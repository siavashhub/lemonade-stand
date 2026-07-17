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

/** A tool the agent can call, flattened from all connected MCP servers. */
export interface AgentTool {
  /** Namespaced as `<serverId>__<toolName>` to avoid collisions. */
  qualifiedName: string
  serverId: string
  toolName: string
  description: string
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

/** Streamed events the main process pushes to the renderer during a turn. */
export type AgentEvent =
  | { type: 'assistant_text'; text: string }
  | { type: 'tool_call'; server: string; tool: string; args: unknown }
  | { type: 'tool_result'; server: string; tool: string; ok: boolean; preview: string }
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
  /** Effective context-window budget for the current chat model. */
  getContextInfo(): Promise<ContextInfo>
  /** Reload the chat model with a new runtime context size (server `/load`).
   * Returns the refreshed context info, or an `error` string on failure. */
  setContextSize(ctxSize: number): Promise<ContextInfo & { error?: string }>
  /** List models the Lemonade server knows about, for the model picker. */
  listModels(): Promise<ModelInfo[]>
  /** Load a model on the server and make it the app's active chat model.
   * Returns the refreshed model list, or throws on failure. */
  loadModel(id: string): Promise<ModelInfo[]>
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
  /** Minimize the window. */
  minimizeWindow(): void
  /** Toggle between maximized and restored. */
  toggleMaximizeWindow(): void
  /** Close the window. */
  closeWindow(): void
}
