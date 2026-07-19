// Types shared between the Electron main process and the React renderer.
// Keep this file dependency-free so both build targets can import it.

export type Role = 'system' | 'user' | 'assistant' | 'tool'

export interface ChatMessage {
  role: Role
  content: string
  /** Present on assistant turns that requested tools; opaque passthrough. */
  tool_calls?: unknown
  /** Present on tool-result turns. */
  tool_call_id?: string
  name?: string
}

/** One step in the agent's working plan for a multi-step task. The model
 * creates and revises the whole list via the built-in `update_plan` tool; the
 * UI renders it as a live checklist. */
export interface PlanStep {
  /** Short imperative description of the step, e.g. "Read the config file". */
  title: string
  /** Lifecycle state, driven by the model as it works through the plan. */
  status: 'pending' | 'in-progress' | 'completed'
}

/** A tool the agent can call, flattened from all connected MCP servers. */
export interface AgentTool {
  /** Namespaced as `<serverId>__<toolName>` to avoid collisions. */
  qualifiedName: string
  serverId: string
  toolName: string
  description: string
}

/** The kind of content a Napkin panel renders. Raw HTML / live browsing is
 * intentionally excluded: the model emits declarative content the renderer
 * knows how to display safely (sanitized before it ever hits the DOM). */
export type NapkinKind = 'code' | 'markdown' | 'mermaid' | 'svg' | 'image'

/** A rich artifact the agent asks the app to show in the side "Napkin" panel,
 * to enrich a reply beyond plain chat text. Created via the built-in
 * `show_napkin` tool; the loop intercepts the call and surfaces it to the UI. */
export interface Napkin {
  /** Short heading shown at the top of the panel. */
  title: string
  kind: NapkinKind
  /** The payload: source text for code/markdown/mermaid/svg, or base64-encoded
   * bytes (no `data:` prefix) for an image. */
  content: string
  /** Language hint for `kind:'code'` display, e.g. 'ts', 'python'. */
  language?: string
  /** MIME type for `kind:'image'`, e.g. 'image/png'. Defaults to image/png. */
  mimeType?: string
  /** Descriptive alt text for `kind:'image'`. */
  alt?: string
}

/** One selectable option in an `ask_napkin` clarification prompt. */
export interface NapkinChoice {
  /** Stable id returned to the model when chosen. Defaults to the label. */
  id: string
  /** Human-readable option text shown on the button. */
  label: string
}

/** One entry in config/servers.json. */
export type McpServerConfig =
  | {
      id: string
      transport: 'stdio'
      command: string
      args?: string[]
      env?: Record<string, string>
      enabled: boolean
      note?: string
    }
  | {
      id: string
      transport: 'http'
      url: string
      headers?: Record<string, string>
      enabled: boolean
      note?: string
    }

/** A browsable tool/skill in the Pantry's Market (config/catalog.json). Enabling
 * one materializes an McpServerConfig in config/servers.json. */
export interface CatalogEntry {
  id: string
  /** Display name shown on the card. */
  name: string
  /** One-line pitch. */
  blurb: string
  /** Grouping/badge, e.g. 'Official', 'Files', 'Web'. */
  category: string
  /** Highlighted as a suggested starter. */
  recommended?: boolean
  transport: 'stdio' | 'http'
  /** stdio template. `args` may contain a `{{path}}` placeholder. */
  command?: string
  args?: string[]
  env?: Record<string, string>
  /** http template. */
  url?: string
  headers?: Record<string, string>
  /** When true, the user must supply a filesystem path before enabling. */
  needsPath?: boolean
  /** Whether the required path is a folder or a single file. */
  pathKind?: 'folder' | 'file'
  /** Label shown next to the path input. */
  pathLabel?: string
  /** Upstream docs link. */
  homepage?: string
}

/** Effective context-window budget for the current chat model, surfaced so the
 * UI can show it and the agent can warn before a request overflows it. */
export interface ContextInfo {
  /** The chat model the sizes refer to. */
  model: string
  /** Effective context window (tokens) used for budgeting. */
  contextSize: number
  /** The model's hard maximum context window, when the server reports it. Used
   * as the upper bound when the user changes the runtime context size. */
  maxContextWindow?: number
  /** Tokens held back for the model's reply (not usable by the prompt). */
  reserve: number
  /** Where `contextSize` came from: an explicit override, the loaded model's
   * runtime `ctx_size` reported by the server, or the built-in fallback. */
  source: 'override' | 'server' | 'default'
}

/** Per-category estimate of how the current conversation fills the model's
 * context window, for the live usage indicator. Token counts are approximate
 * (the same ~4-chars-per-token heuristic the budget check uses). */
export interface ContextBreakdown {
  /** The chat model these figures refer to. */
  model: string
  /** Effective context window (tokens). */
  contextSize: number
  /** Tokens held back for the reply. */
  reserve: number
  /** Total estimated prompt tokens for the next request. */
  usedTokens: number
  /** Per-category token estimates that sum (with `other`) to `usedTokens`. */
  categories: {
    /** System prompt / persona and any folded-in summary. */
    systemInstructions: number
    /** JSON schemas for the connected MCP tools. */
    toolDefinitions: number
    /** User and assistant chat turns. */
    messages: number
    /** `role:"tool"` results returned to the model. */
    toolResults: number
    /** Serialization overhead not attributable to a category above. */
    other: number
  }
}

/** A model the Lemonade server knows about, for the model picker. */
export interface ModelInfo {
  /** Model id, e.g. `Qwen3-1.7B-GGUF`. */
  id: string
  /** Modality/type reported by the server: 'llm', 'tts', 'image', etc. */
  type: string
  /** Capability labels from the server, e.g. 'tool-calling', 'reasoning'. */
  labels: string[]
  /** The model's maximum context window, when reported. */
  maxContextWindow?: number
  /** Approx download size in GB, when reported. */
  sizeGb?: number
  /** Whether the model is downloaded locally. */
  downloaded: boolean
  /** Whether the server currently has this model loaded. */
  loaded: boolean
  /** True when labels indicate tool-calling \u2014 best for the agent loop. */
  agentReady: boolean
  /** True for Omni models \u2014 highlighted and recommended for agentic use. */
  omni: boolean
  /** For Omni collections: names of the component models bundled together. */
  components?: string[]
  /** For Omni collections: a per-collection persona/system prompt, if any. */
  systemPrompt?: string
  /** True when the app is currently configured to chat with this model. */
  active: boolean
}

/** Progress snapshot for a server-owned model download job, surfaced so the UI
 * can show a live progress bar, byte counts, and status for a model being
 * pulled. Mirrors (a subset of) the Lemonade server's /downloads job shape. */
export interface DownloadJob {
  /** Stable download id, e.g. `model:Qwen3-4B-GGUF`; used with the control API. */
  id: string
  /** The Lemonade model name this job downloads. */
  modelName: string
  /** Lifecycle state reported by the server. */
  status: 'downloading' | 'paused' | 'cancelled' | 'completed' | 'error'
  /** Whether the download worker is still active. */
  running: boolean
  /** Overall progress across all of the job's files (0–100). */
  percent: number
  /** Total bytes downloaded across the whole job so far. */
  bytesDownloaded: number
  /** Total expected bytes across all files, when known (0 until resolved). */
  bytesTotal: number
  /** 1-based index of the file currently downloading, when reported. */
  fileIndex?: number
  /** Number of files in the job, when reported. */
  totalFiles?: number
  /** True once every file has downloaded successfully. */
  complete: boolean
  /** Populated only for failed jobs. */
  error?: string
}

/** Live state of a configured server, merged with catalog metadata in the UI. */
export interface McpServerState {
  id: string
  enabled: boolean
  transport: 'stdio' | 'http'
  /** Whether it's currently connected (only meaningful when enabled). */
  connected: boolean
  /** Number of tools it currently contributes. */
  toolCount: number
  /** For path-based stdio servers, the folder/file currently configured. */
  path?: string
  /** Populated when the last connection attempt failed. */
  error?: string
}

/** How the user answered a tool-approval prompt. `always` allow-lists the tool
 * for the rest of the session so it won't prompt again. */
export type ApprovalDecision = 'approve' | 'deny' | 'always'

/** One rendered line in the chat transcript. Mirrors the renderer's visual
 * stream (chat turns + tool activity + notices) so a saved conversation can be
 * restored exactly as it looked, not just as the model saw it. Kept here so both
 * processes can persist/round-trip it. */
export type TranscriptEntry =
  | { kind: 'user'; text: string }
  | { kind: 'assistant'; text: string }
  | { kind: 'tool'; label: string; detail: string; ok?: boolean }
  | { kind: 'plan'; steps: PlanStep[] }
  | { kind: 'napkin'; napkin: Napkin }
  | { kind: 'warning'; text: string }
  | { kind: 'error'; text: string }

/** Lightweight metadata for a saved conversation, used to populate the history
 * sidebar without loading every full transcript. */
export interface SessionSummary {
  id: string
  /** Auto-generated (or user-edited) short title. */
  title: string
  createdAt: number
  updatedAt: number
  /** Number of model-facing messages, for a rough size hint in the list. */
  messageCount: number
}

/** A fully persisted conversation: both what the model sees (`history`) and what
 * the user saw (`entries`), plus any compaction summary already folded in. */
export interface StoredSession extends SessionSummary {
  /** The chat model active when the session was last used. */
  model?: string
  /** Model-facing messages (includes tool_calls / tool results). */
  history: ChatMessage[]
  /** The visual transcript, for faithful restore. */
  entries: TranscriptEntry[]
  /** The most recent compaction summary, if the conversation was compacted. */
  summary?: string
}

/** Streamed events the main process pushes to the renderer during a turn. */
export type AgentEvent =
  | { type: 'assistant_text'; text: string }
  | { type: 'tool_call'; server: string; tool: string; args: unknown }
  | { type: 'tool_result'; server: string; tool: string; ok: boolean; preview: string }
  // The model created or revised its working plan via the built-in `update_plan`
  // tool. `steps` is the full, current checklist the renderer should display.
  | { type: 'plan_updated'; steps: PlanStep[] }
  // The agent asked to display a rich artifact in the side Napkin panel via the
  // built-in `show_napkin` tool (code, markdown, a diagram, an SVG, or image).
  | { type: 'napkin_show'; napkin: Napkin }
  // The agent needs the user to pick among choices to clarify the request (the
  // built-in `ask_napkin` tool). Main is blocked awaiting the user's selection;
  // the renderer must call respondNapkinChoice(id, choiceId) to unblock it.
  | { type: 'napkin_choice_request'; id: string; title: string; prompt: string; choices: NapkinChoice[] }
  // Live per-category context usage for the in-flight turn, so the usage badge
  // reflects the real prompt size (including tool calls/results) while the agent
  // works , not just the committed chat history.
  | { type: 'context_usage'; breakdown: ContextBreakdown }
  // The agent used up its step budget without finishing. Main is blocked
  // awaiting the user's choice; the renderer must call respondStepLimit(id, ...)
  // to either grant another budget or stop.
  | { type: 'step_limit_request'; id: string; steps: number }
  // Main is blocked awaiting a decision; renderer must call respondApproval(id, ...).
  | { type: 'tool_approval_request'; id: string; server: string; tool: string; args: unknown }
  // Synthesized speech for an assistant turn (base64-encoded audio bytes).
  | { type: 'audio'; format: string; base64: string }
  // Estimated request size relative to the model's context window. `overflow`
  // means the request was blocked before sending to avoid a hard server error.
  | {
      type: 'context_warning'
      estimatedTokens: number
      contextSize: number
      reserve: number
      overflow: boolean
    }
  // The main process summarized older messages to stay within the context
  // window. `messages` is the new, compacted model-facing history the renderer
  // should adopt in place of what it sent.
  | { type: 'history_compacted'; messages: ChatMessage[] }
  | { type: 'error'; message: string }
  | { type: 'done' }

/** The `window.api` contract exposed by the preload bridge. */
export interface RendererApi {
  /** Verbose diagnostic logging flag (LOG_LEVEL=debug), for renderer-side
   * console tracing. Read synchronously at startup. */
  debug: boolean
  sendMessage(messages: ChatMessage[]): Promise<void>
  /** Halt the in-flight agent turn started by `sendMessage`. */
  cancelMessage(): void
  listTools(): Promise<AgentTool[]>
  /** Playful "agent is working" phrases for the thinking indicator. */
  getThinkingPhrases(): Promise<string[]>
  onAgentEvent(handler: (event: AgentEvent) => void): () => void
  /** Answer a pending `tool_approval_request`. */
  respondApproval(id: string, decision: ApprovalDecision): void
  /** Turn the session-scoped approval bypass on or off. While on, tool calls
   * are approved without prompting. Scoped to the current conversation: the
   * renderer resets it to false whenever a new session starts, so it never
   * outlives the session and a fresh session falls back to the default
   * (env/settings.json) approval behavior. */
  setBypassApprovals(enabled: boolean): void
  /** Answer a pending `step_limit_request`: true to let the agent keep going
   * for another budget, false to stop it. */
  respondStepLimit(id: string, cont: boolean): void
  /** Answer a pending `napkin_choice_request` with the chosen option id. */
  respondNapkinChoice(id: string, choiceId: string): void
  /** Toggle spoken replies (TTS). Returns the effective state. */
  setSpeak(enabled: boolean): Promise<boolean>
  /** Current spoken-reply state, seeded from config at startup. */
  getSpeak(): Promise<boolean>
  /** Transcribe recorded microphone audio (base64-encoded bytes plus its MIME
   * type) to text via the server's speech-to-text model. Returns the recognized
   * text, or an empty string when nothing intelligible was heard. */
  transcribe(audioBase64: string, mimeType: string): Promise<string>
  /** Halt an in-flight transcription started by `transcribe`. */
  cancelTranscribe(): void
  /** App version for the brand tooltip: the semantic version (e.g. 'v0.0.1')
   * in a packaged/installed build, or 'dev' in a local development run. */
  getAppVersion(): Promise<string>
  /** Ping the Lemonade server; true when it's reachable and healthy. */
  checkHealth(): Promise<boolean>
  /** The Lemonade server connection (base URL + API key) the app targets. */
  getConnection(): Promise<{ baseUrl: string; apiKey: string }>
  /** Repoint the app at a different Lemonade server, persist it, and re-probe.
   * Resolves with the saved connection plus its `online` health result. */
  setConnection(opts: {
    baseUrl: string
    apiKey: string
  }): Promise<{ baseUrl: string; apiKey: string; online: boolean }>
  /** Effective context-window budget for the current chat model. */
  getContextInfo(): Promise<ContextInfo>
  /** Per-category breakdown of how the given conversation fills the context
   * window, for the live usage indicator. */
  getContextBreakdown(history: ChatMessage[]): Promise<ContextBreakdown>
  /** Summarize older messages on demand (the "Compact Conversation" button).
   * Returns the compacted model-facing history, or null when nothing was
   * safe to fold. */
  compactHistory(history: ChatMessage[]): Promise<ChatMessage[] | null>
  /** Reload the chat model with a new runtime context size (server `/load`).
   * Returns the refreshed context info, or an `error` string on failure. */
  setContextSize(ctxSize: number): Promise<ContextInfo & { error?: string }>
  /** List models the Lemonade server knows about, for the model picker. */
  listModels(): Promise<ModelInfo[]>
  /** Load a model on the server and make it the app's active chat model.
   * Returns the refreshed model list, or throws on failure. */
  loadModel(id: string): Promise<ModelInfo[]>
  /** Start a background download of a model on the server (server-owned job
   * that survives a renderer reload). Returns the job's initial snapshot. */
  downloadModel(id: string): Promise<DownloadJob>
  /** List in-flight/recent server-owned model download jobs, for progress. */
  listDownloads(): Promise<DownloadJob[]>
  /** Pause, cancel, or remove a server-owned model download job. */
  controlDownload(id: string, action: 'pause' | 'cancel' | 'remove'): Promise<void>
  /** Delete a downloaded model from local storage to free up disk space.
   * Returns the refreshed model list, or throws on failure. */
  deleteModel(id: string): Promise<ModelInfo[]>
  /** The Market catalogue of installable tools/skills. */
  listCatalog(): Promise<CatalogEntry[]>
  /** Current state of every configured server (enabled or not). */
  listServers(): Promise<McpServerState[]>
  /** Enable/disable a tool (adding it from the catalogue if new). Returns the
   * refreshed server states after reconnecting. */
  configureServer(id: string, opts: { enabled: boolean; path?: string }): Promise<McpServerState[]>
  /** Remove a configured server entirely. Returns refreshed states. */
  removeServer(id: string): Promise<McpServerState[]>
  /** Open a native picker so the user can choose a folder or file path. */
  pickPath(kind: 'folder' | 'file'): Promise<string | null>
  /** List saved conversations, newest first, for the history sidebar. */
  listSessions(): Promise<SessionSummary[]>
  /** Load a full saved conversation (history + transcript). Null if missing. */
  loadSession(id: string): Promise<StoredSession | null>
  /** Persist a conversation. Returns the refreshed session list. */
  saveSession(session: StoredSession): Promise<SessionSummary[]>
  /** Delete a saved conversation. Returns the refreshed session list. */
  deleteSession(id: string): Promise<SessionSummary[]>
  /** Rename a saved conversation. Returns the refreshed session list. */
  renameSession(id: string, title: string): Promise<SessionSummary[]>
  /** Delete every saved conversation. Returns the (empty) session list. */
  clearSessions(): Promise<SessionSummary[]>
  /** Ask the model for a short auto-title for a conversation. Falls back to a
   * trimmed first user message when the model is unavailable. */
  suggestTitle(history: ChatMessage[]): Promise<string>
  /** Minimize the window. */
  minimizeWindow(): void
  /** Toggle between maximized and restored. */
  toggleMaximizeWindow(): void
  /** Close the window. */
  closeWindow(): void
}
