import OpenAI, { toFile } from 'openai'
import type {
  ChatCompletionMessageParam,
  ChatCompletionTool
} from 'openai/resources/chat/completions'
import type { ContextInfo, ModelInfo } from '@shared/types'

// Fallback context window used only when no override is configured and the
// server can't be reached to report the loaded model's runtime context.
const DEFAULT_CONTEXT_SIZE = 4096

// Context size a model is loaded with when the caller doesn't pick one. 8192 is
// the smallest power-of-two that fits the tool-heavy agent prompt (~4.3K) AND
// leaves room to generate a file, so it avoids the common overflow error while
// staying memory-frugal. Every LLM this server ships supports far more.
export const DEFAULT_LOAD_CONTEXT = 8192

// Shape (subset) of lemond's /models response.
interface ModelsEntry {
  id?: string
  labels?: string[]
  max_context_window?: number
  size?: number
  downloaded?: boolean
  /** Recipe type, e.g. 'llamacpp' or 'collection.omni' for Omni routers. */
  recipe?: string
  /** For Omni collections: the ids of the component models bundled together. */
  components?: string[]
  /** For Omni collections: a per-collection system prompt / persona, if any. */
  system_prompt?: string
}

// Shape (subset) of lemond's /health response we rely on for context info.
interface HealthModel {
  model_name?: string
  type?: string
  max_context_window?: number
  recipe_options?: { ctx_size?: number }
  loaded?: boolean
}
interface HealthResponse {
  model_loaded?: string
  all_models_loaded?: HealthModel[]
}

// Result of comparing an outgoing request against the model's context budget.
export interface ContextBudget {
  estimatedTokens: number
  /** Effective context window (tokens). */
  contextSize: number
  /** Tokens reserved for the reply. */
  reserve: number
  /** Usable prompt tokens = contextSize - reserve. */
  budget: number
  /** true when the estimate exceeds the usable budget (request should not be sent). */
  overflow: boolean
  /** true when the estimate is close to the budget but not yet over it. */
  warn: boolean
  source: ContextInfo['source']
}

// Thin wrapper over the OpenAI SDK pointed at a running lemond. Lemonade
// implements the OpenAI-compatible surface under /api/v1, so the stock SDK
// works unchanged; we only supply baseURL + (optional) apiKey.
export class LemonadeClient {
  private client: OpenAI
  private model: string
  private baseURL: string
  private contextOverride?: number
  private completionReserve: number
  // Cached server-reported context size, so we don't refetch every step. Only a
  // successful lookup is cached; failures fall back and retry next turn.
  private cachedServerContext?: number
  // Cached model max context window from the last successful /health lookup.
  private cachedMaxContext?: number
  // Cached id of a server-served TTS model, resolved lazily on first speak().
  // Keeps the configured id from going stale if the server renames its voice.
  private cachedTtsModel?: string
  // Cached id of a server-served transcription model, resolved lazily on the
  // first transcribe() call so a stale/renamed model can't silently break it.
  private cachedTranscriptionModel?: string

  constructor(
    baseURL: string,
    apiKey: string,
    model: string,
    contextOverride?: number,
    completionReserve = 512
  ) {
    this.client = new OpenAI({
      baseURL,
      // The SDK requires a non-empty key even when the server doesn't enforce
      // one; send a placeholder so unauthenticated local servers still work.
      apiKey: apiKey || 'lemonade'
    })
    this.model = model
    this.baseURL = baseURL
    this.contextOverride = contextOverride && contextOverride > 0 ? contextOverride : undefined
    this.completionReserve = completionReserve
  }

  /** The id of the model currently configured as the agent's chat model. */
  get activeModel(): string {
    return this.model
  }

  /**
   * Probe the running lemond via its OpenAI-compatible `/health` endpoint.
   * Returns true only on a 2xx response within the timeout; any network error
   * (server not running) or non-2xx status resolves to false rather than
   * throwing, so callers can render a simple online/offline indicator.
   */
  async health(timeoutMs = 3000): Promise<boolean> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)
    try {
      const response = await fetch(`${this.baseURL}/health`, {
        method: 'GET',
        signal: controller.signal
      })
      return response.ok
    } catch {
      return false
    } finally {
      clearTimeout(timer)
    }
  }

  /**
   * Fetch the /health payload and pick out the entry for this chat model (or the
   * server's currently loaded LLM). This is where the *runtime* context size
   * lives: `recipe_options.ctx_size` is what the model was actually loaded with,
   * which can be smaller than the model's advertised `max_context_window`.
   */
  private async fetchModelHealth(timeoutMs = 3000): Promise<HealthModel | null> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)
    try {
      const response = await fetch(`${this.baseURL}/health`, {
        method: 'GET',
        signal: controller.signal
      })
      if (!response.ok) return null
      const body = (await response.json()) as HealthResponse
      const models = body.all_models_loaded ?? []
      // Prefer the entry matching our configured chat model; otherwise fall back
      // to whatever LLM the server currently has loaded. Omni collections have
      // no backend of their own — chat runs on a loaded LLM component — so the
      // collection name never matches an LLM here; the loaded-LLM fallbacks
      // resolve context from that component instead of dropping to the default.
      return (
        models.find((m) => m.type === 'llm' && m.model_name === this.model) ??
        models.find((m) => m.type === 'llm' && m.model_name === body.model_loaded) ??
        models.find((m) => m.type === 'llm' && m.loaded) ??
        models.find((m) => m.type === 'llm') ??
        null
      )
    } catch {
      return null
    } finally {
      clearTimeout(timer)
    }
  }

  /**
   * Resolve the effective context window (in tokens) for the chat model:
   *   1. an explicit override (config/env) always wins;
   *   2. otherwise the loaded model's runtime `ctx_size` from /health;
   *   3. otherwise a conservative built-in fallback.
   * A successful server lookup is cached until the context is changed.
   */
  async resolveContextSize(): Promise<{ size: number; source: ContextInfo['source'] }> {
    if (this.contextOverride) return { size: this.contextOverride, source: 'override' }
    if (this.cachedServerContext) return { size: this.cachedServerContext, source: 'server' }

    const entry = await this.fetchModelHealth()
    if (entry?.max_context_window && entry.max_context_window > 0) {
      this.cachedMaxContext = entry.max_context_window
    }
    const ctx = entry?.recipe_options?.ctx_size
    if (typeof ctx === 'number' && ctx > 0) {
      this.cachedServerContext = ctx
      return { size: ctx, source: 'server' }
    }
    return { size: DEFAULT_CONTEXT_SIZE, source: 'default' }
  }

  /**
   * List the models the server knows about, merged with which one is currently
   * loaded (from /health). A model is flagged `agentReady` when its labels
   * advertise tool-calling — the capability the agent loop depends on.
   */
  async listModels(): Promise<ModelInfo[]> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 5000)
    let entries: ModelsEntry[] = []
    try {
      const response = await fetch(`${this.baseURL}/models`, {
        method: 'GET',
        signal: controller.signal
      })
      if (response.ok) {
        const body = (await response.json()) as { data?: ModelsEntry[] }
        entries = body.data ?? []
      }
    } catch {
      // Server unreachable -> return an empty list; the UI shows an empty state.
    } finally {
      clearTimeout(timer)
    }

    // The default list only reports downloaded models. Omni router models
    // (recipe 'collection.omni') are usually not downloaded, so pull the full
    // catalogue with show_all=true and merge in just the Omni entries — we want
    // to surface them for agentic use without flooding the list with every
    // supported model.
    try {
      const controllerAll = new AbortController()
      const timerAll = setTimeout(() => controllerAll.abort(), 5000)
      try {
        const response = await fetch(`${this.baseURL}/models?show_all=true`, {
          method: 'GET',
          signal: controllerAll.signal
        })
        if (response.ok) {
          const body = (await response.json()) as { data?: ModelsEntry[] }
          const known = new Set(entries.map((e) => e.id))
          for (const e of body.data ?? []) {
            if (e.recipe === 'collection.omni' && !known.has(e.id)) entries.push(e)
          }
        }
      } finally {
        clearTimeout(timerAll)
      }
    } catch {
      // Best-effort; if this fails we simply don't surface undownloaded Omni.
    }

    // Which model ids are currently loaded, per /health.
    const loaded = new Set<string>()
    try {
      const controller2 = new AbortController()
      const timer2 = setTimeout(() => controller2.abort(), 3000)
      try {
        const health = await fetch(`${this.baseURL}/health`, {
          method: 'GET',
          signal: controller2.signal
        })
        if (health.ok) {
          const body = (await health.json()) as HealthResponse
          for (const m of body.all_models_loaded ?? []) {
            if (m.loaded && m.model_name) loaded.add(m.model_name)
          }
        }
      } finally {
        clearTimeout(timer2)
      }
    } catch {
      // Best-effort; loaded flags just stay false.
    }

    return entries.map((e) => {
      const labels = e.labels ?? []
      // Omni is a router recipe that loads several models at once; the server
      // tags these entries with recipe 'collection.omni'. They're multimodal
      // and well-suited to agentic use.
      const omni = e.recipe === 'collection.omni'
      // Infer a coarse modality from labels for grouping in the UI. Omni models
      // are treated as chat models so they surface in the agent's model list.
      const type = omni
        ? 'llm'
        : labels.includes('tts')
          ? 'tts'
          : labels.includes('image')
            ? 'image'
            : labels.includes('transcription') || labels.includes('realtime-transcription')
              ? 'transcription'
              : 'llm'
      return {
        id: e.id ?? '',
        type,
        labels,
        maxContextWindow: e.max_context_window,
        sizeGb: e.size,
        downloaded: e.downloaded ?? false,
        loaded: e.id ? loaded.has(e.id) : false,
        agentReady: labels.includes('tool-calling') || omni,
        omni,
        components: omni ? (e.components ?? []) : undefined,
        systemPrompt: omni ? e.system_prompt : undefined,
        active: e.id === this.model
      }
    })
  }

  /**
   * Load a model on the server via /load and make it the app's active chat
   * model. Loads with `ctxSize` (default 8192) so the model is immediately
   * usable for tool-heavy agent turns without a manual context bump. Clears
   * cached context so the next budget check reflects the new model. Loading
   * (and any download) can take a while, so the timeout is long.
   */
  async loadModel(
    id: string,
    ctxSize: number = DEFAULT_LOAD_CONTEXT
  ): Promise<{ ok: boolean; error?: string }> {
    // Pointer-based Omni collections (e.g. LMX-Omni-*, RPG-HaloTales) resolve
    // their component list only after their manifest is downloaded, and the
    // server's /load intentionally skips downloading for collection recipes.
    // Loading such a collection before it's pulled fails with "GGUF file not
    // found for checkpoint", so pull it first to register + fetch its parts.
    const entry = await this.fetchModelEntry(id)
    if (
      entry?.recipe === 'collection.omni' &&
      entry.downloaded !== true &&
      (entry.components?.length ?? 0) === 0
    ) {
      const pulled = await this.pullModel(id)
      if (!pulled.ok) return pulled
    }

    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 300000)
    try {
      const response = await fetch(`${this.baseURL}/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: id, ctx_size: ctxSize }),
        signal: controller.signal
      })
      if (!response.ok) {
        const detail = await response.text().catch(() => '')
        return { ok: false, error: `Server returned ${response.status}: ${detail || 'load failed'}` }
      }
      this.model = id
      this.cachedServerContext = undefined
      this.cachedMaxContext = undefined
      return { ok: true }
    } catch (err) {
      return { ok: false, error: err instanceof Error ? err.message : String(err) }
    } finally {
      clearTimeout(timer)
    }
  }

  /**
   * Download a model (and, for Omni collections, its manifest and component
   * models) via the server's /pull. Used to prime a not-yet-downloaded Omni
   * collection before /load, since collection routers only resolve their
   * component list once their manifest is present. Downloads can be large, so
   * the timeout is generous.
   */
  async pullModel(id: string): Promise<{ ok: boolean; error?: string }> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 1_800_000)
    try {
      const response = await fetch(`${this.baseURL}/pull`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: id, stream: false }),
        signal: controller.signal
      })
      if (!response.ok) {
        const detail = await response.text().catch(() => '')
        return { ok: false, error: `Server returned ${response.status}: ${detail || 'pull failed'}` }
      }
      return { ok: true }
    } catch (err) {
      return { ok: false, error: err instanceof Error ? err.message : String(err) }
    } finally {
      clearTimeout(timer)
    }
  }

  /** Fetch a single model's registry entry from /models/{id}, or null. */
  private async fetchModelEntry(id: string): Promise<ModelsEntry | null> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 5000)
    try {
      const response = await fetch(`${this.baseURL}/models/${encodeURIComponent(id)}`, {
        method: 'GET',
        signal: controller.signal
      })
      if (!response.ok) return null
      return (await response.json()) as ModelsEntry
    } catch {
      return null
    } finally {
      clearTimeout(timer)
    }
  }

  /**
   * On startup, make sure the active chat model is loaded with at least the
   * default context window. The server (and its own defaults) may have the
   * model unloaded or loaded at a smaller context after a restart, which is why
   * the app's context budget would otherwise revert to the 4096 fallback. This
   * runs best-effort: any failure (server down, model still downloading) is
   * swallowed so it never blocks app launch.
   */
  async ensureModelLoaded(ctxSize: number = DEFAULT_LOAD_CONTEXT): Promise<void> {
    try {
      const entry = await this.fetchModelHealth()
      const current = entry?.recipe_options?.ctx_size
      // Already loaded at a big-enough context -> leave it (don't downgrade a
      // context the user may have raised on purpose).
      if (entry?.loaded && typeof current === 'number' && current >= ctxSize) return
      await this.loadModel(this.model, ctxSize)
    } catch {
      // Best-effort only.
    }
  }

  /** Context-window info for the renderer to display. */
  async getContextInfo(): Promise<ContextInfo> {
    const { size, source } = await this.resolveContextSize()
    return {
      model: this.model,
      contextSize: size,
      maxContextWindow: this.cachedMaxContext,
      reserve: this.completionReserve,
      source
    }
  }

  /**
   * Reload the chat model with a new runtime context size via lemond's /load
   * endpoint. On success the cached budget is updated so the next request is
   * checked against the new window. Reloading can take a while, so the timeout
   * is generous.
   */
  async setContextSize(ctxSize: number): Promise<ContextInfo & { error?: string }> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 180000)
    try {
      // Omni collections have no backend of their own: chat runs through the
      // server's collection orchestrator, which (re)loads each component from
      // its *saved* recipe_options.json — a transient ctx_size sent to the
      // collection is never forwarded to its parts. So target the loaded LLM
      // component directly and persist the size (save_options) so the next
      // collection turn doesn't reload the component back to its old default.
      const isCollection =
        (await this.fetchModelEntry(this.model))?.recipe === 'collection.omni'
      let target = this.model
      let saveOptions = false
      if (isCollection) {
        const component = (await this.fetchModelHealth())?.model_name
        if (component) {
          target = component
          saveOptions = true
        }
      }
      const response = await fetch(`${this.baseURL}/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: target, ctx_size: ctxSize, save_options: saveOptions }),
        signal: controller.signal
      })
      if (!response.ok) {
        const detail = await response.text().catch(() => '')
        const info = await this.getContextInfo()
        return { ...info, error: `Server returned ${response.status}: ${detail || 'load failed'}` }
      }
      // Trust the requested size and refresh the max/window cache from health.
      this.cachedServerContext = ctxSize
      const entry = await this.fetchModelHealth()
      if (entry?.max_context_window && entry.max_context_window > 0) {
        this.cachedMaxContext = entry.max_context_window
      }
      if (typeof entry?.recipe_options?.ctx_size === 'number') {
        this.cachedServerContext = entry.recipe_options.ctx_size
      }
      return {
        model: this.model,
        contextSize: this.cachedServerContext,
        maxContextWindow: this.cachedMaxContext,
        reserve: this.completionReserve,
        // An override still wins for budgeting; report the true source.
        source: this.contextOverride ? 'override' : 'server'
      }
    } catch (err) {
      const info = await this.getContextInfo()
      return { ...info, error: err instanceof Error ? err.message : String(err) }
    } finally {
      clearTimeout(timer)
    }
  }

  /**
   * Rough token estimate for an outgoing request. Serializes the messages and
   * tool schemas and applies the common ~4-chars-per-token heuristic. This is
   * intentionally approximate: it only needs to be good enough to warn before a
   * request would exceed the model's context window.
   */
  estimateTokens(
    messages: ChatCompletionMessageParam[],
    tools: ChatCompletionTool[]
  ): number {
    const payload = JSON.stringify(messages) + JSON.stringify(tools ?? [])
    return Math.ceil(payload.length / 4)
  }

  /**
   * Compare an outgoing request against the resolved context budget so the
   * agent can warn (or block) before the server rejects an over-long prompt.
   */
  async checkBudget(
    messages: ChatCompletionMessageParam[],
    tools: ChatCompletionTool[]
  ): Promise<ContextBudget> {
    const { size, source } = await this.resolveContextSize()
    const estimatedTokens = this.estimateTokens(messages, tools)
    const budget = Math.max(0, size - this.completionReserve)
    const overflow = estimatedTokens > budget
    const warn = !overflow && estimatedTokens > budget * 0.85
    return {
      estimatedTokens,
      contextSize: size,
      reserve: this.completionReserve,
      budget,
      overflow,
      warn,
      source
    }
  }

  /**
   * Single (non-streamed) chat completion. Tools are passed straight through
   * in OpenAI function-tool shape; the model may respond with tool_calls that
   * the agent loop is responsible for executing.
   */
  async chat(
    messages: ChatCompletionMessageParam[],
    tools: ChatCompletionTool[]
  ): Promise<OpenAI.Chat.Completions.ChatCompletion.Choice> {
    const response = await this.client.chat.completions.create({
      model: this.model,
      messages,
      tools: tools.length > 0 ? tools : undefined,
      tool_choice: tools.length > 0 ? 'auto' : undefined,
      stream: false
    })
    const choice = response.choices[0]
    if (!choice) throw new Error('lemond returned no choices')
    return choice
  }

  /**
   * Resolve a TTS model the server actually serves. `preferred` (from config)
   * wins when it's present in the server's model list; otherwise the first
   * model labeled "tts" is used. Resolution is cached so it costs one /models
   * lookup per process. If the server can't be reached, `preferred` is returned
   * unchanged so a transient outage doesn't permanently disable audio.
   */
  private async resolveTtsModel(preferred: string): Promise<string> {
    if (this.cachedTtsModel) return this.cachedTtsModel
    const models = await this.listModels()
    const ttsModels = models.filter((m) => m.type === 'tts')
    if (ttsModels.length === 0) return preferred
    const match = ttsModels.find((m) => m.id === preferred)
    const chosen = match?.id ?? ttsModels[0].id
    this.cachedTtsModel = chosen
    if (chosen !== preferred) {
      console.warn(
        `[tts] configured model "${preferred}" not served; using "${chosen}" instead`
      )
    }
    return chosen
  }

  /**
   * Synthesize speech via lemond's OpenAI-compatible /v1/audio/speech endpoint
   * (backed by Kokoro TTS). Returns base64-encoded audio bytes plus the format
   * so the renderer can wrap them in the right MIME type. `stream=true` isn't
   * needed here — replies are short and we play them as one clip. The model id
   * is resolved against the server so a stale/renamed voice can't silently
   * break playback.
   */
  async speak(
    text: string,
    model: string,
    voice: string,
    format: string
  ): Promise<{ base64: string; format: string }> {
    const resolvedModel = await this.resolveTtsModel(model)
    const response = await this.client.audio.speech.create({
      model: resolvedModel,
      voice: voice as never, // lemond accepts backend voice ids beyond the SDK's enum
      input: text,
      response_format: format as never
    })
    const bytes = Buffer.from(await response.arrayBuffer())
    return { base64: bytes.toString('base64'), format }
  }

  /**
   * Resolve a transcription (speech-to-text) model the server actually serves.
   * `preferred` (from config) wins when it's present in the server's model list;
   * otherwise the first model typed "transcription" is used. Resolution is
   * cached so it costs one /models lookup per process. If the server can't be
   * reached, `preferred` is returned unchanged so a transient outage doesn't
   * permanently disable dictation.
   */
  private async resolveTranscriptionModel(preferred: string): Promise<string> {
    if (this.cachedTranscriptionModel) return this.cachedTranscriptionModel
    const models = await this.listModels()
    const sttModels = models.filter((m) => m.type === 'transcription')
    if (sttModels.length === 0) return preferred
    const match = sttModels.find((m) => m.id === preferred)
    const chosen = match?.id ?? sttModels[0].id
    this.cachedTranscriptionModel = chosen
    if (chosen !== preferred) {
      console.warn(
        `[stt] configured model "${preferred}" not served; using "${chosen}" instead`
      )
    }
    return chosen
  }

  /**
   * Transcribe recorded audio via lemond's OpenAI-compatible
   * /v1/audio/transcriptions endpoint (backed by Whisper). `audioBase64` holds
   * the raw recording bytes (typically WebM/Opus from the browser's
   * MediaRecorder) and `mimeType` its content type, used to pick a sensible
   * filename extension so the server can decode it. The model id is resolved
   * against the server so a stale/renamed model can't silently break dictation.
   */
  async transcribe(audioBase64: string, mimeType: string, model: string): Promise<string> {
    const resolvedModel = await this.resolveTranscriptionModel(model)
    const bytes = Buffer.from(audioBase64, 'base64')
    const ext = mimeType.includes('webm')
      ? 'webm'
      : mimeType.includes('ogg')
        ? 'ogg'
        : mimeType.includes('wav')
          ? 'wav'
          : mimeType.includes('mp4') || mimeType.includes('mp4a') || mimeType.includes('m4a')
            ? 'mp4'
            : mimeType.includes('mpeg') || mimeType.includes('mp3')
              ? 'mp3'
              : 'webm'
    const file = await toFile(bytes, `speech.${ext}`, { type: mimeType || 'audio/webm' })
    const response = await this.client.audio.transcriptions.create({
      model: resolvedModel,
      file
    })
    return response.text ?? ''
  }
}
