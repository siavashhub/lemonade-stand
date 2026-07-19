import OpenAI, { toFile } from 'openai'
import type {
  ChatCompletionMessageParam,
  ChatCompletionTool
} from 'openai/resources/chat/completions'
import type { ContextBreakdown, ContextInfo, DownloadJob, ModelInfo } from '@shared/types'
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

// Shape (subset) of a Lemonade /downloads job snapshot. Byte totals come under
// several aliases depending on server version; we read the most complete one.
interface DownloadEntry {
  id?: string
  model_name?: string
  status?: string
  running?: boolean
  file_index?: number
  total_files?: number
  bytes_total?: number
  total_download_size?: number
  cumulative_bytes_downloaded?: number
  overall_bytes_downloaded?: number
  bytes_downloaded?: number
  percent?: number
  complete?: boolean
  error?: string
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
  private apiKey: string
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
    this.apiKey = apiKey
    this.contextOverride = contextOverride && contextOverride > 0 ? contextOverride : undefined
    this.completionReserve = completionReserve
  }

  /** The id of the model currently configured as the agent's chat model. */
  get activeModel(): string {
    return this.model
  }

  /** The Lemonade server this client currently targets (base URL + API key),
   * surfaced so the UI can show and edit the active connection. */
  get connection(): { baseUrl: string; apiKey: string } {
    return { baseUrl: this.baseURL, apiKey: this.apiKey }
  }

  /**
   * Repoint this client at a different Lemonade server. Recreates the OpenAI
   * client with the new base URL / key and clears every server-derived cache so
   * nothing from the old server leaks across (context sizes, resolved TTS and
   * transcription model ids). Callers persist the choice and re-probe health.
   */
  setConnection(baseURL: string, apiKey: string): void {
    this.baseURL = baseURL
    this.apiKey = apiKey
    this.client = new OpenAI({ baseURL, apiKey: apiKey || 'lemonade' })
    this.cachedServerContext = undefined
    this.cachedMaxContext = undefined
    this.cachedTtsModel = undefined
    this.cachedTranscriptionModel = undefined
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
      // no backend of their own , chat runs on a loaded LLM component , so the
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
   * advertise tool-calling , the capability the agent loop depends on.
   */
  async listModels(): Promise<ModelInfo[]> {
    // 1. The plain /models list reports only *downloaded* models. We use it as
    //    the authoritative "is this on disk?" signal, since the full catalogue
    //    (below) can under-report downloaded state for pointer-based Omni
    //    collections whose components are already present.
    const downloadedIds = new Set<string>()
    let downloadedEntries: ModelsEntry[] = []
    try {
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), 5000)
      try {
        const response = await fetch(`${this.baseURL}/models`, {
          method: 'GET',
          signal: controller.signal
        })
        if (response.ok) {
          const body = (await response.json()) as { data?: ModelsEntry[] }
          downloadedEntries = body.data ?? []
          for (const e of downloadedEntries) if (e.id) downloadedIds.add(e.id)
        }
      } finally {
        clearTimeout(timer)
      }
    } catch {
      // Server unreachable -> handled below; the UI shows an empty state.
    }

    // 2. The full catalogue (show_all=true) is the source of truth for *which*
    //    models to display, so not-yet-downloaded models , and models the user
    //    just uninstalled , stay visible with a Download button instead of
    //    vanishing from the list.
    let entries: ModelsEntry[] = []
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
          entries = body.data ?? []
        }
      } finally {
        clearTimeout(timerAll)
      }
    } catch {
      // Best-effort; fall back to the downloaded-only list below.
    }
    // If the full catalogue couldn't be fetched, fall back to whatever the
    // downloaded-only list returned so the picker still works offline-ish.
    if (entries.length === 0) entries = downloadedEntries

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
        // Trust the authoritative downloaded-only list first; the full catalogue
        // can report a pointer Omni collection as not-downloaded even when its
        // components are already on disk.
        downloaded: (e.id ? downloadedIds.has(e.id) : false) || (e.downloaded ?? false),
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

  /**
   * Delete a downloaded model from local storage via the server's /delete. If
   * the model is currently loaded, the server unloads it first. Note: deleting
   * an Omni collection removes only the collection entry , its component models
   * stay on disk (delete those individually to reclaim their space).
   */
  async deleteModel(id: string): Promise<{ ok: boolean; error?: string }> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 60000)
    try {
      const response = await fetch(`${this.baseURL}/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: id }),
        signal: controller.signal
      })
      if (!response.ok) {
        const detail = await response.text().catch(() => '')
        return {
          ok: false,
          error: `Server returned ${response.status}: ${detail || 'delete failed'}`
        }
      }
      return { ok: true }
    } catch (err) {
      return { ok: false, error: err instanceof Error ? err.message : String(err) }
    } finally {
      clearTimeout(timer)
    }
  }

  /** Normalize a raw /downloads job snapshot into the renderer's DownloadJob. */
  private mapDownload(e: DownloadEntry): DownloadJob {
    const status = (e.status ?? 'downloading') as DownloadJob['status']
    return {
      id: e.id ?? (e.model_name ? `model:${e.model_name}` : ''),
      modelName: e.model_name ?? '',
      status: ['downloading', 'paused', 'cancelled', 'completed', 'error'].includes(status)
        ? status
        : 'downloading',
      running: e.running ?? false,
      percent: typeof e.percent === 'number' ? e.percent : 0,
      // Prefer the whole-job cumulative total; fall back to the per-file count.
      bytesDownloaded:
        e.cumulative_bytes_downloaded ??
        e.overall_bytes_downloaded ??
        e.bytes_downloaded ??
        0,
      bytesTotal: e.total_download_size || e.bytes_total || 0,
      fileIndex: e.file_index,
      totalFiles: e.total_files,
      complete: e.complete ?? false,
      error: e.error
    }
  }

  /**
   * Kick off a *server-owned* background download of a model via /pull with
   * `stream:true, subscribe:false`. The server starts the job and returns a
   * snapshot immediately; the download then proceeds independently of this
   * connection, so it survives a renderer reload. Poll {@link listDownloads} to
   * track progress. Returns the initial job snapshot.
   */
  async startDownload(id: string): Promise<DownloadJob> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 30000)
    try {
      const response = await fetch(`${this.baseURL}/pull`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: id, stream: true, subscribe: false }),
        signal: controller.signal
      })
      if (!response.ok) {
        const detail = await response.text().catch(() => '')
        throw new Error(`Server returned ${response.status}: ${detail || 'download failed'}`)
      }
      const body = (await response.json()) as DownloadEntry
      return this.mapDownload({ ...body, model_name: body.model_name ?? id })
    } finally {
      clearTimeout(timer)
    }
  }

  /** List the server's current model download jobs (for live progress). */
  async listDownloads(): Promise<DownloadJob[]> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 5000)
    try {
      const response = await fetch(`${this.baseURL}/downloads`, {
        method: 'GET',
        signal: controller.signal
      })
      if (!response.ok) return []
      const body = (await response.json()) as DownloadEntry[]
      return (body ?? []).filter((e) => e.model_name).map((e) => this.mapDownload(e))
    } catch {
      return []
    } finally {
      clearTimeout(timer)
    }
  }

  /** Pause, cancel, or remove a server-owned model download job. */
  async controlDownload(id: string, action: 'pause' | 'cancel' | 'remove'): Promise<void> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 10000)
    try {
      await fetch(`${this.baseURL}/downloads/control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, action }),
        signal: controller.signal
      })
    } catch {
      // Best-effort; the next poll reflects the real job state regardless.
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
      // its *saved* recipe_options.json , a transient ctx_size sent to the
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
      // The user explicitly changed context from the UI. Honor that live choice
      // for the current session even when an env/config override was present,
      // so budget checks and warnings reflect what the server is now running.
      this.contextOverride = undefined
      return {
        model: this.model,
        contextSize: this.cachedServerContext,
        maxContextWindow: this.cachedMaxContext,
        reserve: this.completionReserve,
        source: 'server'
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
   * Break the estimated prompt size down by category so the UI can show where
   * the context window is going. Uses the same ~4-chars-per-token heuristic as
   * `estimateTokens`, applied to each subset. `systemPrompt` is folded into the
   * system-instructions bucket when the history doesn't already carry a system
   * message (mirroring what the agent loop injects before sending).
   */
  async contextBreakdown(
    messages: ChatCompletionMessageParam[],
    tools: ChatCompletionTool[],
    systemPrompt: string
  ): Promise<ContextBreakdown> {
    const { size } = await this.resolveContextSize()
    const est = (s: string): number => Math.ceil(s.length / 4)
    const contentOf = (m: ChatCompletionMessageParam): string =>
      typeof m.content === 'string' ? m.content : JSON.stringify(m.content ?? '')

    // The loop injects the system prompt only when none is present already.
    const hasSystem = messages.some((m) => m.role === 'system')
    const systemText =
      messages
        .filter((m) => m.role === 'system')
        .map(contentOf)
        .join('') + (hasSystem ? '' : systemPrompt)

    const messagesText = messages
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map(contentOf)
      .join('')
    const toolResultsText = messages
      .filter((m) => m.role === 'tool')
      .map(contentOf)
      .join('')
    const toolDefsText = JSON.stringify(tools ?? [])

    const systemInstructions = est(systemText)
    const toolDefinitions = est(toolDefsText)
    const messagesTokens = est(messagesText)
    const toolResults = est(toolResultsText)

    // Total is estimated the same way the budget check does, so the indicator
    // agrees with the warn/overflow logic. `other` captures JSON structural
    // overhead (roles, ids, keys) not attributed to a content bucket above.
    const effective = hasSystem
      ? messages
      : [{ role: 'system', content: systemPrompt } as ChatCompletionMessageParam, ...messages]
    const usedTokens = this.estimateTokens(effective, tools)
    const categorized = systemInstructions + toolDefinitions + messagesTokens + toolResults
    const other = Math.max(0, usedTokens - categorized)

    return {
      model: this.model,
      contextSize: size,
      reserve: this.completionReserve,
      usedTokens,
      categories: {
        systemInstructions,
        toolDefinitions,
        messages: messagesTokens,
        toolResults,
        other
      }
    }
  }

  /**
   * Single chat completion. Tools are passed straight through in OpenAI
   * function-tool shape; the model may respond with tool_calls that the agent
   * loop is responsible for executing.
   *
   * The request is *streamed* even though callers only consume the final,
   * reassembled result. This is deliberate and load-bearing for the stop
   * button: with a non-streamed request, aborting the fetch closes the socket,
   * but the OpenAI-compatible backend (llama.cpp / OGA) doesn't notice the
   * client is gone until it writes the response , which never happens until the
   * whole reply is generated , so it keeps the GPU/NPU busy to completion (the
   * fan keeps spinning). Streaming makes the backend write a chunk per token,
   * so the dropped connection is detected on the next token and generation
   * stops promptly when the user hits stop.
   */
  async chat(
    messages: ChatCompletionMessageParam[],
    tools: ChatCompletionTool[],
    signal?: AbortSignal
  ): Promise<OpenAI.Chat.Completions.ChatCompletion.Choice> {
    const stream = this.client.beta.chat.completions.stream(
      {
        model: this.model,
        messages,
        tools: tools.length > 0 ? tools : undefined,
        tool_choice: tools.length > 0 ? 'auto' : undefined
      },
      { signal }
    )
    // Accumulate the model's chain-of-thought ourselves. Reasoning models served
    // by llama.cpp / OGA (e.g. Qwen3) stream their thinking in a non-standard
    // `reasoning_content` (or `reasoning`) delta field. The OpenAI SDK's stream
    // accumulator only *concatenates* the standard delta fields (content,
    // tool_calls, ...); unknown fields are Object.assign'd, so each chunk
    // OVERWRITES the previous , finalChatCompletion() would surface at most the
    // last token. So we append the fragments here and reattach the full text to
    // the final message below, where the agent loop's splitReasoning() reads it.
    let reasoning = ''
    stream.on('chunk', (chunk) => {
      const delta = chunk.choices[0]?.delta as
        | { reasoning_content?: unknown; reasoning?: unknown }
        | undefined
      if (typeof delta?.reasoning_content === 'string') reasoning += delta.reasoning_content
      else if (typeof delta?.reasoning === 'string') reasoning += delta.reasoning
    })
    // If the caller aborts, tear the stream (and its socket) down immediately so
    // the server stops generating instead of running to completion unheard.
    const onAbort = (): void => stream.controller.abort()
    signal?.addEventListener('abort', onAbort, { once: true })
    try {
      const response = await stream.finalChatCompletion()
      const choice = response.choices[0]
      if (!choice) throw new Error('lemond returned no choices')
      // Reattach our fully-accumulated reasoning (the SDK reassembly drops it).
      // Authoritative: our concatenation is the complete text, so always prefer it.
      if (reasoning.trim()) {
        ;(choice.message as { reasoning_content?: string }).reasoning_content = reasoning
      }
      return choice
    } finally {
      signal?.removeEventListener('abort', onAbort)
    }
  }

  /**
   * Compress a slice of conversation into a dense prose summary the agent can
   * carry forward in place of the raw messages. Runs a plain (tool-free) chat
   * completion so it can't trigger side effects. `priorSummary` folds an earlier
   * compaction back in so summaries don't accumulate as the chat grows.
   */
  async summarize(
    messages: ChatCompletionMessageParam[],
    priorSummary?: string,
    signal?: AbortSignal
  ): Promise<string> {
    const transcript = messages
      .map((m) => {
        const content =
          typeof m.content === 'string' ? m.content : JSON.stringify(m.content ?? '')
        return `${m.role.toUpperCase()}: ${content}`
      })
      .join('\n')
    const body =
      (priorSummary ? `Summary of the conversation so far:\n${priorSummary}\n\n` : '') +
      `Newer messages to fold in:\n${transcript}`
    const choice = await this.chat(
      [
        {
          role: 'system',
          content:
            'You compress conversations so they can continue within a limited context ' +
            'window. Produce a concise summary that preserves facts, decisions, task ' +
            'state, file names, identifiers, and any open questions or next steps. Write a ' +
            'few short paragraphs. Do not add commentary, preamble, or a title.'
        },
        { role: 'user', content: body }
      ],
      [],
      signal
    )
    return choice.message.content?.trim() ?? ''
  }

  /**
   * Ask the model for a short, specific conversation title. Best-effort: on any
   * failure the caller falls back to a trimmed first user message, so a slow or
   * offline server never blocks saving a session.
   */
  async generateTitle(
    messages: ChatCompletionMessageParam[],
    signal?: AbortSignal
  ): Promise<string> {
    const snippet = messages
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .slice(0, 4)
      .map((m) => {
        const content =
          typeof m.content === 'string' ? m.content : JSON.stringify(m.content ?? '')
        return `${m.role.toUpperCase()}: ${content}`
      })
      .join('\n')
      .slice(0, 2000)
    const choice = await this.chat(
      [
        {
          role: 'system',
          content:
            'Generate a short, specific title (3 to 6 words) for the conversation. Reply ' +
            'with only the title: no quotes, no trailing punctuation, no "Title:" prefix.'
        },
        { role: 'user', content: snippet }
      ],
      [],
      signal
    )
    const raw = choice.message.content?.trim() ?? ''
    // Strip wrapping quotes and any trailing period the model adds anyway.
    return raw.replace(/^["'\s]+|["'\s.]+$/g, '').slice(0, 60)
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
   * needed here , replies are short and we play them as one clip. The model id
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
  async transcribe(
    audioBase64: string,
    mimeType: string,
    model: string,
    signal?: AbortSignal
  ): Promise<string> {
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
    const response = await this.client.audio.transcriptions.create(
      {
        model: resolvedModel,
        file
      },
      { signal }
    )
    return response.text ?? ''
  }
}
