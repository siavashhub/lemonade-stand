import { app, BrowserWindow, dialog, ipcMain, Menu, Notification, shell } from 'electron'
import { randomUUID } from 'node:crypto'
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync } from 'node:fs'
import { dirname, isAbsolute, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import type {
  AgentEvent,
  ApprovalDecision,
  ChatMessage,
  McpServerConfig,
  McpServerState,
  Napkin,
  Pitcher,
  PitcherEvent,
  PitcherRunResult,
  StoredSession,
  TranscriptEntry
} from '@shared/types'
import {
  loadCatalog,
  loadConfig,
  loadThinkingPhrases,
  pathForServer,
  readPitchers,
  readServers,
  seedLocalOverrides,
  serverFromCatalog,
  withServerPath,
  writePitchers,
  writeServers,
  writeSettings
} from './config'
import { LemonadeClient } from './lemonade/client'
import { McpManager } from './mcp/manager'
import { Agent, type ApproveFn, type AskNapkinFn, type ContinueFn, type ProposePitcherFn } from './agent/loop'
import { PitcherScheduler } from './pitcher/scheduler'
import { initFileLogging } from './logger'
import {
  clearSessions,
  deleteSession,
  listSessions,
  readSession,
  renameSession,
  writeSession
} from './history/store'
const __dirname = fileURLToPath(new URL('.', import.meta.url))

// Resolve the directory that holds the app's `config/` files (catalog, phrases,
// servers, settings). In development that's the project root. In a packaged
// build the bundled defaults ship read-only under `process.resourcesPath/config`
// (see electron-builder.yml `extraResources`), while the app must also *write*
// to servers.json / settings.json, so on first run we seed those defaults into
// a writable per-user directory and read/write there afterwards. This fixes a
// packaged app starting empty (no Pantry catalogue, no configured servers)
// because it was looking for `config/` inside the read-only asar.
function resolveConfigDir(): string {
  if (!app.isPackaged) {
    // Dev checkout: seed the gitignored *.local.json overrides so the app writes
    // UI edits there instead of dirtying the committed defaults, and a developer
    // cloning the repo discovers the mechanism from the seeded, self-documenting
    // files on first `npm run dev`.
    const devConfigDir = app.getAppPath()
    seedLocalOverrides(devConfigDir)
    return devConfigDir
  }

  const userConfigBase = app.getPath('userData')
  const userConfigDir = join(userConfigBase, 'config')
  const bundledConfigDir = join(process.resourcesPath, 'config')
  try {
    mkdirSync(userConfigDir, { recursive: true })
    for (const name of readdirSync(bundledConfigDir)) {
      const dest = join(userConfigDir, name)
      // Seed each default only once; never clobber the user's own edits.
      if (!existsSync(dest)) copyFileSync(join(bundledConfigDir, name), dest)
    }
    return userConfigBase
  } catch {
    // If seeding fails for any reason, fall back to the bundled (read-only)
    // defaults so the app at least starts with a populated catalogue.
    return process.resourcesPath
  }
}

const appPath = resolveConfigDir()

// Read-only bundled defaults ship under `resourcesPath/config` in a packaged
// build; in a dev checkout they live in the same tree as `appPath`. The Market
// catalogue's built-in entries are read from here so shipped corrections (e.g. a
// fixed homepage link) always reach users, even though their writable per-user
// config is seeded once and never re-copied on upgrade. `loadCatalog` merges any
// user-added entries from the per-user copy back in by id.
const bundledConfigPath = app.isPackaged ? process.resourcesPath : appPath

const config = loadConfig(appPath)

// The bundled Memory MCP server (@modelcontextprotocol/server-memory) defaults
// its knowledge-graph file to `memory.jsonl` *next to its own module*. In a
// packaged build that module lives inside the read-only asar, so every write
// fails with `ENOENT: no such file or directory` , the exact symptom users hit
// where the memory server reads fine but can't create entities. Point it at the
// app's writable data dir instead. `??=` respects an explicit override (real env
// or a `.env` file, already loaded by loadConfig), so a user's own path wins.
process.env.MEMORY_FILE_PATH ??= join(appPath, 'memory.jsonl')
const lemonade = new LemonadeClient(
  config.lemonadeBaseUrl,
  config.lemonadeApiKey,
  config.model,
  config.contextSize,
  config.completionReserve,
  config.maxCompletionTokens
)
const mcp = new McpManager()
const agent = new Agent(lemonade, mcp, config.maxSteps, config.systemPrompt, config.compactThreshold)

// --- Session state -----------------------------------------------------------

// Verbose diagnostic logging, gated by LOG_LEVEL=debug. Errors always log;
// only the chatty progress traces are silenced when debug is off.
const debugLog = config.debug
  ? (...args: unknown[]): void => console.log(...args)
  : (): void => {}

// Spoken replies. Seeded from config; toggled live from the renderer.
let speakEnabled = config.tts.enabled
debugLog(
  `[tts] startup: enabled=${config.tts.enabled} model=${config.tts.model} voice=${config.tts.voice} format=${config.tts.format} (appPath=${appPath})`
)

// Tools the user chose "always allow" for this session (keyed by qualified
// name). Cleared on restart so a persistent grant never outlives the process.
const sessionAllow = new Set<string>()

// Session-scoped "bypass approvals" override. When on, tool calls are approved
// without prompting regardless of config.requireApproval. The renderer turns it
// on/off from the status bar and resets it to false whenever a new conversation
// starts, so a bypass never carries over into a fresh session.
let bypassApprovals = false

// In-flight approval prompts: id -> resolver. The agent loop awaits these;
// the renderer resolves them via the 'agent:approve' channel.
const pendingApprovals = new Map<string, (decision: ApprovalDecision) => void>()

// In-flight step-limit prompts: id -> resolver. When the agent exhausts its
// step budget it asks the renderer whether to keep going; the reply arrives on
// the 'agent:continue' channel.
const pendingLimits = new Map<string, (cont: boolean) => void>()

// In-flight napkin choice prompts: id -> resolver. When the agent asks a
// multiple-choice clarifying question (ask_napkin), it blocks until the user
// picks an option in the Napkin panel; the reply arrives on the
// 'agent:napkin-choice' channel.
const pendingNapkinChoices = new Map<string, (choiceId: string) => void>()

// In-flight create_pitcher proposals: id -> resolver. The agent loop awaits
// these while the user reviews a pre-filled Pitcher editor; the renderer
// resolves via the 'agent:pitcher-proposal-result' channel.
const pendingPitcherProposals = new Map<
  string,
  (result: { saved: boolean; name: string }) => void
>()

// Abort handles for work the user can halt mid-flight. The renderer's stop
// button signals these via the 'agent:cancel' / 'agent:cancel-transcribe'
// channels so a long chat turn or transcription can be interrupted.
let currentAgentAbort: AbortController | null = null
let currentTranscribeAbort: AbortController | null = null

// The main window, captured in createWindow(). Headless Pitcher pours use it to
// mirror events to an open UI and to raise desktop notifications.
let mainWindow: BrowserWindow | null = null

// Guards the periodic MCP reconnect sweep (see whenReady) so a slow attempt
// can't overlap with the next tick.
let serverSweepInFlight = false

// Resolve the app icon for the window. In a packaged build resources/ ship as
// extraResources under process.resourcesPath (they are NOT inside app.asar, so
// app.getAppPath() can't see them); in dev app.getAppPath() is the project root.
function resolveIconPath(): string {
  const iconFile = process.platform === 'win32' ? 'icon.ico' : 'icon.png'
  const base = app.isPackaged ? process.resourcesPath : app.getAppPath()
  return join(base, 'resources', iconFile)
}

function createWindow(): void {
  const window = new BrowserWindow({
    width: 1000,
    height: 720,
    title: 'Lemonade Stand',
    // Frameless: no native OS title bar. The renderer draws its own top bar and
    // window controls (see App.tsx / styles.css).
    frame: false,
    icon: resolveIconPath(),
    webPreferences: {
      preload: join(__dirname, '../preload/index.mjs'),
      sandbox: false,
      contextIsolation: true,
      nodeIntegration: false,
      // Let synthesized TTS replies play without a fresh user gesture. The
      // audio arrives seconds after the user's click (chat + synthesis
      // round-trip), so Chromium's default gesture requirement would otherwise
      // reject Audio.play() and swallow the sound.
      autoplayPolicy: 'no-user-gesture-required',
      // Surface the debug flag to the preload synchronously (read from argv) so
      // the renderer can gate its own diagnostic logging without an IPC round-trip.
      additionalArguments: [`--app-debug=${config.debug ? '1' : '0'}`]
    }
  })

  // Keep a handle so headless Pitcher pours can mirror events to the UI and
  // raise desktop notifications.
  mainWindow = window
  window.on('closed', () => {
    if (mainWindow === window) mainWindow = null
  })

  // Window controls driven by the renderer's custom top bar.
  ipcMain.on('window:minimize', () => window.minimize())
  ipcMain.on('window:toggle-maximize', () => {
    if (window.isMaximized()) window.unmaximize()
    else window.maximize()
  })
  ipcMain.on('window:close', () => window.close())

  // electron-vite injects the dev server URL in development; load the built
  // HTML in production.
  const devUrl = process.env['ELECTRON_RENDERER_URL']
  if (devUrl) {
    window.loadURL(devUrl)
  } else {
    window.loadFile(join(__dirname, '../renderer/index.html'))
  }

  // The app clears the native menu (Menu.setApplicationMenu(null)), which also
  // removes Electron's default DevTools accelerators. Re-add them by hand so
  // F12 / Ctrl+Shift+I (Cmd+Opt+I on macOS) still open the inspector. With
  // LOG_LEVEL=debug, also open DevTools automatically so diagnostic output is
  // visible immediately.
  window.webContents.on('before-input-event', (_event, input) => {
    if (input.type !== 'keyDown') return
    const toggle =
      input.key === 'F12' ||
      ((input.control || input.meta) && input.shift && input.key.toLowerCase() === 'i')
    if (toggle) window.webContents.toggleDevTools()
  })
  if (config.debug) window.webContents.openDevTools({ mode: 'detach' })
}

// --- IPC: renderer <-> agent -------------------------------------------------

ipcMain.handle('agent:list-tools', () => mcp.getTools())

// --- Pantry: browse the Market and manage configured tools -------------------

// Snapshot of every configured server merged with its live connection state.
function serverStates(): McpServerState[] {
  const catalog = loadCatalog(appPath, bundledConfigPath)
  return readServers(appPath).map((s) => {
    const rt = mcp.getRuntime(s.id)
    const entry = catalog.find((c) => c.id === s.id)
    return {
      id: s.id,
      enabled: s.enabled,
      transport: s.transport,
      connected: rt?.connected ?? false,
      toolCount: rt?.toolCount ?? 0,
      path: entry ? pathForServer(entry, s) : undefined,
      error: rt?.error
    }
  })
}

// Id of the built-in Lemonade Gateway entry (see config/catalog.json). Its "/mcp"
// endpoint is the running Lemonade server's OWN, so it must always live on the
// same host/port as the chat API.
const GATEWAY_SERVER_ID = 'lemonade'

// Derive the gateway's "/mcp" URL from the chat base URL. Base URLs end in the
// REST version prefix (".../api/v1"); swap that final segment for the sibling
// "/mcp" path, preserving any reverse-proxy prefix (so
// "http://host/lemonade/api/v1" -> "http://host/lemonade/mcp"). Falls back to
// "<origin>/mcp" when the path doesn't match that shape. Returns undefined for
// an unparseable base URL.
function gatewayUrlFromBase(baseUrl: string): string | undefined {
  let u: URL
  try {
    u = new URL(baseUrl)
  } catch {
    return undefined
  }
  const path = u.pathname.replace(/\/+$/, '')
  const mcpPath = /\/api\/v\d+$/.test(path) ? path.replace(/\/api\/v\d+$/, '/mcp') : '/mcp'
  return `${u.origin}${mcpPath}`
}

// The built-in Lemonade Gateway is the running Lemonade server's own endpoint,
// so it can never live anywhere but the host the chat API points at. The catalog
// ships it as `http://localhost:13305/mcp`, but when the user repoints chat at a
// different host (e.g. a Lemonade server on another machine on the LAN) a
// hardcoded `localhost` gateway can never connect , the exact reason cross-machine
// setups saw the gateway stuck on "can't reach this server". Rewrite the gateway
// entry's URL to follow the live chat connection so the two always track together
// without any hand-editing. Only the built-in gateway id is touched; user-added
// HTTP servers keep their own URLs.
function withGatewayUrl(servers: McpServerConfig[]): McpServerConfig[] {
  const derived = gatewayUrlFromBase(lemonade.connection.baseUrl)
  if (!derived) return servers
  return servers.map((s) =>
    s.id === GATEWAY_SERVER_ID && s.transport === 'http' && s.url !== derived
      ? { ...s, url: derived }
      : s
  )
}

// The Lemonade Gateway MCP server ("/mcp") is the running lemond's OWN endpoint,
// so when that server is launched with an API key the gateway enforces it too.
// The MCP Streamable HTTP transport only sends a server's explicit `headers`,
// which the default gateway entry omits , so a key-protected server rejects the
// connection with "Invalid or missing API key". Attach the configured Bearer
// token to any HTTP server whose URL shares the Lemonade base URL's origin (and
// ONLY that origin, so the key never leaks to unrelated third-party HTTP MCP
// servers), unless the entry already sets its own Authorization header.
function withGatewayAuth(servers: McpServerConfig[]): McpServerConfig[] {
  const { baseUrl, apiKey } = lemonade.connection
  if (!apiKey) return servers
  let gatewayOrigin: string
  try {
    gatewayOrigin = new URL(baseUrl).origin
  } catch {
    return servers
  }
  return servers.map((s) => {
    if (s.transport !== 'http') return s
    let sameOrigin = false
    try {
      sameOrigin = new URL(s.url).origin === gatewayOrigin
    } catch {
      sameOrigin = false
    }
    if (!sameOrigin) return s
    const hasAuth = s.headers
      ? Object.keys(s.headers).some((k) => k.toLowerCase() === 'authorization')
      : false
    if (hasAuth) return s
    return { ...s, headers: { ...s.headers, Authorization: `Bearer ${apiKey}` } }
  })
}

// Prepare the on-disk server list for connecting: first point the built-in
// Lemonade Gateway at the currently-configured chat host, then attach the
// gateway's API key. Order matters , the auth step matches servers by the
// Lemonade base URL's origin, so the URL must be rewritten to that host first
// for the key to land on a cross-machine gateway.
function prepareServers(servers: McpServerConfig[]): McpServerConfig[] {
  return withGatewayAuth(withGatewayUrl(servers))
}

// Reconnect only the enabled servers from the current on-disk config. Called
// after any change so tool availability tracks the user's choices live.
async function reloadServers(): Promise<void> {
  const all = readServers(appPath)
  await mcp.closeAll()
  await mcp.connectAll(prepareServers(all.filter((s) => s.enabled)))
}

ipcMain.handle('catalog:list', () => loadCatalog(appPath, bundledConfigPath))
ipcMain.handle('servers:list', () => serverStates())

// Playful "agent is working" phrases for the thinking indicator.
ipcMain.handle('agent:thinking-phrases', () => loadThinkingPhrases(appPath))

ipcMain.handle(
  'servers:configure',
  async (_event, id: string, opts: { enabled: boolean; path?: string }) => {
    const all = readServers(appPath)
    const idx = all.findIndex((s) => s.id === id)
    if (idx >= 0) {
      let updated = { ...all[idx], enabled: opts.enabled }
      // If a new path was supplied, rewrite the server's `{{path}}` arg so the
      // user can change the folder after the server was first configured.
      if (opts.path) {
        const entry = loadCatalog(appPath, bundledConfigPath).find((c) => c.id === id)
        if (entry) updated = withServerPath(entry, updated, opts.path)
      }
      all[idx] = updated
    } else {
      const entry = loadCatalog(appPath, bundledConfigPath).find((c) => c.id === id)
      if (!entry) throw new Error(`Unknown tool "${id}"`)
      const built = serverFromCatalog(entry, opts.path)
      built.enabled = opts.enabled
      all.push(built)
    }
    writeServers(appPath, all)
    await reloadServers()
    return serverStates()
  }
)

ipcMain.handle('servers:remove', async (_event, id: string) => {
  writeServers(
    appPath,
    readServers(appPath).filter((s) => s.id !== id)
  )
  await reloadServers()
  return serverStates()
})

ipcMain.handle('dialog:pick-path', async (_event, kind: 'folder' | 'file') => {
  const result = await dialog.showOpenDialog({
    properties: [kind === 'folder' ? 'openDirectory' : 'openFile']
  })
  return result.canceled || result.filePaths.length === 0 ? null : result.filePaths[0]
})

// --- Conversation history ----------------------------------------------------
// Saved sessions live one-file-per-conversation under the writable config dir,
// so a user can revisit or continue an earlier chat.

ipcMain.handle('history:list', () => listSessions(appPath))
ipcMain.handle('history:load', (_event, id: string) => readSession(appPath, id))
ipcMain.handle('history:save', (_event, session: StoredSession) =>
  writeSession(appPath, session)
)
ipcMain.handle('history:delete', (_event, id: string) => deleteSession(appPath, id))
ipcMain.handle('history:rename', (_event, id: string, title: string) =>
  renameSession(appPath, id, title)
)
ipcMain.handle('history:clear', () => clearSessions(appPath))

// Auto-title a conversation. Best-effort: fall back to a trimmed first user
// message when the model is slow or offline so saving never blocks on this.
ipcMain.handle('history:suggest-title', async (_event, messages: ChatMessage[]) => {
  const firstUser = messages.find((m) => m.role === 'user')
  const firstText =
    typeof firstUser?.content === 'string'
      ? firstUser.content
      : (firstUser?.content ?? [])
          .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
          .map((p) => p.text)
          .join(' ')
  const fallback = (firstText || 'New conversation').trim().slice(0, 60)
  try {
    const title = await lemonade.generateTitle(messages as never)
    return title || fallback
  } catch {
    return fallback
  }
})

ipcMain.handle('agent:get-speak', () => speakEnabled)
ipcMain.handle('agent:set-speak', (_event, enabled: boolean) => {
  speakEnabled = Boolean(enabled)
  writeSettings(appPath, { speak: speakEnabled })
  return speakEnabled
})

// Transcribe recorded microphone audio to text via the server's speech-to-text
// model, so the user can dictate their message instead of typing it. Tracks an
// AbortController so the renderer's stop button can cancel a slow transcription.
ipcMain.handle('agent:transcribe', async (_event, audioBase64: string, mimeType: string) => {
  const abort = new AbortController()
  currentTranscribeAbort = abort
  try {
    return await lemonade.transcribe(audioBase64, mimeType, config.stt.model, abort.signal)
  } finally {
    if (currentTranscribeAbort === abort) currentTranscribeAbort = null
  }
})

// Renderer's stop button: halt a running transcription.
ipcMain.on('agent:cancel-transcribe', () => currentTranscribeAbort?.abort())

// Version shown in the UI's brand tooltip. A packaged/installed build reports
// the semantic version baked into package.json (e.g. 'v0.0.1'); a local dev run
// (electron-vite dev / preview) reports 'dev' since it isn't a tagged release.
ipcMain.handle('app:version', () => (app.isPackaged ? `v${app.getVersion()}` : 'dev'))

// Health probe for the renderer's server-status indicator.
ipcMain.handle('agent:check-health', () => lemonade.health())

// The Lemonade server connection (base URL + API key) the app currently targets,
// so the renderer's connection editor can show what's active.
ipcMain.handle('agent:get-connection', () => lemonade.connection)

// Repoint the app at a different Lemonade server. Persists the choice so it
// survives restarts, then probes the new server so the caller can immediately
// reflect online/offline. Trims input and drops a trailing slash so
// `${baseUrl}/health` never doubles up.
ipcMain.handle(
  'agent:set-connection',
  async (_event, opts: { baseUrl: string; apiKey: string }) => {
    const baseUrl = String(opts.baseUrl ?? '').trim().replace(/\/+$/, '')
    const apiKey = String(opts.apiKey ?? '').trim()
    if (!/^https?:\/\//i.test(baseUrl)) {
      throw new Error('Enter a full base URL, e.g. http://localhost:13305/api/v1')
    }
    lemonade.setConnection(baseUrl, apiKey)
    writeSettings(appPath, { baseUrl, apiKey })
    // Reconnect so the Lemonade Gateway ("/mcp") picks up the new key/origin:
    // its Authorization header is derived from this connection, so a stale
    // connection would keep failing with "Invalid or missing API key".
    await reloadServers()
    const online = await lemonade.health()
    return { baseUrl, apiKey, online }
  }
)

// Effective context-window budget, surfaced in the UI.
ipcMain.handle('agent:context-info', () => lemonade.getContextInfo())

// Per-category breakdown of how the current conversation fills the context
// window, for the live usage indicator. The tool catalogue and system prompt
// live in main, so the split is computed here from the renderer's history.
ipcMain.handle('agent:context-breakdown', (_event, messages: ChatMessage[]) =>
  lemonade.contextBreakdown(
    messages as never,
    mcp.getOpenAiTools(),
    config.systemPrompt
  )
)

// Manual "Compact Conversation" button: summarize older messages on demand and
// return the compacted history (or null when nothing was safe to fold).
ipcMain.handle('agent:compact', (_event, messages: ChatMessage[]) =>
  agent.compact(messages)
)

// Reload the chat model with a new runtime context size (server /load). Persist
// the chosen size so it survives restarts (dev: config/settings.local.json;
// packaged: the per-user settings.json). Only save when the reload succeeded.
ipcMain.handle('agent:set-context', async (_event, ctxSize: number) => {
  const info = await lemonade.setContextSize(ctxSize)
  if (!info.error) writeSettings(appPath, { contextSize: info.contextSize })
  return info
})

// Update the reply-length cap (max_completion_tokens) live , no server reload,
// so a ruminating model can't run until it fills the context window. Persist the
// choice (dev: settings.local.json; packaged: per-user settings.json) so it
// survives restarts, then return the refreshed budget for the UI.
ipcMain.handle('agent:set-max-completion-tokens', async (_event, tokens: number) => {
  lemonade.setMaxCompletionTokens(tokens)
  writeSettings(appPath, { maxCompletionTokens: lemonade.replyCap })
  return lemonade.getContextInfo()
})

// Models the server knows about, for the model picker.
ipcMain.handle('agent:list-models', () => lemonade.listModels())

// Load a model on the server and make it the active chat model. Returns the
// refreshed model list so the UI reflects the new loaded/active state. The
// choice is persisted so the same model is active again after a restart.
ipcMain.handle('agent:load-model', async (_event, id: string, ctxSize?: number) => {
  const result = await lemonade.loadModel(id, ctxSize)
  if (!result.ok) throw new Error(result.error ?? 'Failed to load model')
  writeSettings(appPath, { model: lemonade.activeModel })
  return lemonade.listModels()
})

// Unload a model from server memory to free RAM, leaving it on disk. Returns the
// refreshed model list so the UI reflects the freed slot. Does not change the
// app's active chat model.
ipcMain.handle('agent:unload-model', async (_event, id: string) => {
  const result = await lemonade.unloadModel(id)
  if (!result.ok) throw new Error(result.error ?? 'Failed to unload model')
  return lemonade.listModels()
})

// Start a server-owned background download of a model. Returns the initial job
// snapshot; the renderer polls agent:list-downloads for live progress.
ipcMain.handle('agent:download-model', (_event, id: string) => lemonade.startDownload(id))

// Current model download jobs, for the live progress indicators.
ipcMain.handle('agent:list-downloads', () => lemonade.listDownloads())

// Pause, cancel, or remove a model download job.
ipcMain.handle('agent:control-download', (_event, id: string, action: 'pause' | 'cancel' | 'remove') =>
  lemonade.controlDownload(id, action)
)

// Delete a downloaded model from local storage to free up disk space. Returns
// the refreshed model list so the UI reflects its now not-downloaded state.
ipcMain.handle('agent:delete-model', async (_event, id: string) => {
  const result = await lemonade.deleteModel(id)
  if (!result.ok) throw new Error(result.error ?? 'Failed to delete model')
  return lemonade.listModels()
})

// Renderer's answer to a tool_approval_request. Resolving the stored promise
// unblocks the agent loop.
ipcMain.on('agent:approve', (_event, id: string, decision: ApprovalDecision) => {
  const resolve = pendingApprovals.get(id)
  if (resolve) {
    pendingApprovals.delete(id)
    resolve(decision)
  }
})

// Toggle the session-scoped approval bypass. The renderer sets it true when the
// user turns on "bypass approvals" and false when a new conversation starts, so
// the override never outlives the session that enabled it.
ipcMain.on('agent:set-bypass', (_event, enabled: boolean) => {
  bypassApprovals = enabled
})

// Roots of every configured path-based stdio server (e.g. the Filesystem
// server's allowed directory). Filesystem tools return paths relative to their
// root, so a napkin's folderPath can be relative (e.g. "notes"); we resolve it
// against these roots before opening.
function configuredServerRoots(): string[] {
  const catalog = loadCatalog(appPath, bundledConfigPath)
  const roots: string[] = []
  for (const server of readServers(appPath)) {
    const entry = catalog.find((c) => c.id === server.id)
    const root = entry ? pathForServer(entry, server) : undefined
    if (root && isAbsolute(root)) roots.push(root)
  }
  return roots
}

// Resolve a possibly-relative folder path to an absolute, existing location.
// Absolute paths are used as-is; relative paths are tried against each
// configured server root. Files resolve to their containing directory.
function resolveExplorerTarget(rawPath: string): string {
  const unquoted = rawPath.replace(/^['"]|['"]$/g, '')
  const toDir = (p: string): string =>
    existsSync(p) && statSync(p).isFile() ? dirname(p) : p

  if (isAbsolute(unquoted)) return toDir(unquoted)

  for (const root of configuredServerRoots()) {
    const candidate = join(root, unquoted)
    if (existsSync(candidate)) return toDir(candidate)
  }
  // Nothing matched; return the best guess so the error names a real path.
  const roots = configuredServerRoots()
  return roots.length > 0 ? join(roots[0], unquoted) : unquoted
}

ipcMain.handle('explorer:open-folder', async (_event, folderPath: string) => {
  try {
    const rawPath = String(folderPath ?? '').trim()
    if (!rawPath) throw new Error('No folder path provided')

    const target = resolveExplorerTarget(rawPath)
    if (!existsSync(target)) throw new Error(`Path not found: ${target}`)
    const openError = await shell.openPath(target)
    if (openError) throw new Error(openError)
  } catch (err) {
    console.error('[explorer] failed to open folder:', err)
    throw err
  }
})

// Renderer's answer to a step_limit_request: true to grant another step budget,
// false to stop. Resolving the stored promise unblocks the agent loop.
ipcMain.on('agent:continue', (_event, id: string, cont: boolean) => {
  const resolve = pendingLimits.get(id)
  if (resolve) {
    pendingLimits.delete(id)
    resolve(cont)
  }
})

// Renderer's answer to a napkin_choice_request: the id of the option the user
// picked. Resolving the stored promise unblocks the agent loop.
ipcMain.on('agent:napkin-choice', (_event, id: string, choiceId: string) => {
  const resolve = pendingNapkinChoices.get(id)
  if (resolve) {
    pendingNapkinChoices.delete(id)
    resolve(choiceId)
  }
})

// Renderer's answer to a pitcher_proposal_request: whether the user saved the
// pre-filled scheduled task. Resolving the stored promise unblocks the loop.
ipcMain.on(
  'agent:pitcher-proposal-result',
  (_event, id: string, result: { saved: boolean; name: string }) => {
    const resolve = pendingPitcherProposals.get(id)
    if (resolve) {
      pendingPitcherProposals.delete(id)
      resolve(result)
    }
  }
)

// Compact one-line description of an agent event for the debug log, so a full
// turn (model completions, tool calls, plan updates, budget) can be read back
// from the log file when diagnosing a run.
function describeEvent(e: AgentEvent): string {
  switch (e.type) {
    case 'tool_call':
      return `tool_call ${e.server}__${e.tool} args=${JSON.stringify(e.args).slice(0, 200)}`
    case 'tool_result':
      return `tool_result ${e.server}__${e.tool} ok=${e.ok} preview=${e.preview.slice(0, 120)}`
    case 'plan_updated':
      return `plan_updated steps=${e.steps.length} [${e.steps
        .map((s) => `${s.status[0]}:${s.title}`)
        .join(' | ')
        .slice(0, 300)}]`
    case 'napkin_show':
      return `napkin_show kind=${e.napkin.kind} title=${e.napkin.title} len=${e.napkin.content.length}`
    case 'napkin_choice_request':
      return `napkin_choice_request choices=${e.choices.length} prompt=${e.prompt.slice(0, 80)}`
    case 'pitcher_proposal_request':
      return `pitcher_proposal_request name=${e.draft.name} trigger=${e.draft.trigger.type}`
    case 'assistant_text':
      return `assistant_text len=${e.text.trim().length}`
    case 'reasoning':
      return `reasoning len=${e.text.trim().length}`
    case 'reasoning_delta':
      return `reasoning_delta len=${e.text.length}`
    case 'tool_approval_request':
      return `tool_approval_request ${e.server}__${e.tool}`
    case 'step_limit_request':
      return `step_limit_request steps=${e.steps}`
    case 'context_usage':
      return `context_usage used=${e.breakdown.usedTokens}/${e.breakdown.contextSize}`
    case 'context_warning':
      return `context_warning est=${e.estimatedTokens} ctx=${e.contextSize} overflow=${e.overflow}`
    case 'history_compacted':
      return `history_compacted messages=${e.messages.length}`
    case 'error':
      return `error ${e.message}`
    case 'done':
      return 'done'
    default:
      return e.type
  }
}

ipcMain.handle('agent:send', async (event, messages: ChatMessage[]) => {
  const send = (agentEvent: AgentEvent): void => {
    if (!event.sender.isDestroyed()) event.sender.send('agent:event', agentEvent)
  }

  // Wrap the emit so a final assistant turn is also synthesized to speech when
  // TTS is on. Fire-and-forget: playback lags the text slightly but never
  // blocks the loop, and a TTS failure only logs.
  // TTS is on. Fire-and-forget: playback lags the text slightly but never
  // blocks the loop, and a TTS failure only logs.
  const emit = (agentEvent: AgentEvent): void => {
    send(agentEvent)
    // Trace every agent event to the log (when debug is on) so a run can be
    // reconstructed from the log file: tool calls, plan updates, budget, etc.
    debugLog(`[agent] ${describeEvent(agentEvent)}`)
    if (agentEvent.type === 'assistant_text') {
      debugLog(
        `[tts] assistant_text: speakEnabled=${speakEnabled} textLen=${agentEvent.text.trim().length}`
      )
    }
    if (agentEvent.type === 'assistant_text' && speakEnabled && agentEvent.text.trim()) {
      debugLog(
        `[tts] synthesizing model=${config.tts.model} voice=${config.tts.voice} format=${config.tts.format}`
      )
      lemonade
        .speak(agentEvent.text, config.tts.model, config.tts.voice, config.tts.format)
        .then((audio) => {
          debugLog(`[tts] synthesis OK: ${audio.base64.length} b64 chars, format=${audio.format}`)
          send({ type: 'audio', format: audio.format, base64: audio.base64 })
        })
        .catch((err) => console.error('[tts] synthesis failed:', err))
    }
  }

  // Approve callback: auto-allow when approval is disabled, the session is
  // bypassing approvals, or the tool was already "always allowed"; otherwise
  // prompt the renderer and await a reply.
  const approve: ApproveFn = ({ server, tool, qualified, args }) => {
    if (!config.requireApproval || bypassApprovals || sessionAllow.has(qualified))
      return Promise.resolve(true)
    const id = randomUUID()
    send({ type: 'tool_approval_request', id, server, tool, args })
    return new Promise<boolean>((resolve) => {
      pendingApprovals.set(id, (decision) => {
        if (decision === 'always') {
          sessionAllow.add(qualified)
          resolve(true)
        } else {
          resolve(decision === 'approve')
        }
      })
    })
  }

  // Step-limit callback: prompt the renderer when the agent runs out of its
  // step budget and await the user's choice to keep going or stop.
  const onLimit: ContinueFn = (steps) => {
    const id = randomUUID()
    send({ type: 'step_limit_request', id, steps })
    return new Promise<boolean>((resolve) => {
      pendingLimits.set(id, resolve)
    })
  }

  // Napkin-choice callback: prompt the renderer with a multiple-choice question
  // (rendered in the Napkin panel) and await the id of the option the user picks.
  const askNapkin: AskNapkinFn = ({ title, prompt, choices }) => {
    const id = randomUUID()
    send({ type: 'napkin_choice_request', id, title, prompt, choices })
    return new Promise<string>((resolve) => {
      pendingNapkinChoices.set(id, resolve)
    })
  }

  // Propose-pitcher callback: open a pre-filled Pitcher editor in the renderer
  // and await the user's decision. Only ever passed to this interactive turn ,
  // pours never get it , so a scheduled task can't create more scheduled tasks.
  const proposePitcher: ProposePitcherFn = (draft) => {
    const id = randomUUID()
    send({ type: 'pitcher_proposal_request', id, draft })
    return new Promise<{ saved: boolean; name: string }>((resolve) => {
      pendingPitcherProposals.set(id, resolve)
    })
  }

  const abort = new AbortController()
  currentAgentAbort = abort
  try {
    await agent.run(messages, emit, approve, abort.signal, onLimit, askNapkin, proposePitcher)
  } finally {
    if (currentAgentAbort === abort) currentAgentAbort = null
  }
})

// Renderer's stop button: halt the running agent turn. Aborting only flips the
// signal, which the loop checks between steps , but if the turn is parked on a
// pending approval prompt it would never reach that check. So also drain any
// in-flight approvals, resolving each as a denial, to unblock the awaited
// `approve(...)` call so the loop can observe the abort and stop.
ipcMain.on('agent:cancel', () => {
  currentAgentAbort?.abort()
  for (const resolve of pendingApprovals.values()) resolve('deny')
  pendingApprovals.clear()
  // Unblock any turn parked on a clarification or a pitcher proposal so the loop
  // can observe the abort and stop instead of hanging on user input.
  for (const resolve of pendingNapkinChoices.values()) resolve('')
  pendingNapkinChoices.clear()
  for (const resolve of pendingPitcherProposals.values()) resolve({ saved: false, name: '' })
  pendingPitcherProposals.clear()
})

// --- Pitcher: scheduled tasks ------------------------------------------------

function emitPitcher(evt: PitcherEvent): void {
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('pitcher:event', evt)
}

// Run one Pitcher end-to-end. Auto-approves only tools on its allowlist; every
// other tool call is denied so a scheduled task can't be steered (e.g. by
// injected web content) into actions it was never granted. Every pour , success
// OR failure , is saved as a conversation the user can reopen, so a failed run
// is inspectable (which tool was blocked, what error occurred) instead of
// vanishing. A desktop notification is raised when the window isn't focused, and
// a bounded retry guards against a flaky local model.
async function pourPitcher(p: Pitcher): Promise<PitcherRunResult> {
  emitPitcher({ type: 'pitcher_started', id: p.id })

  // A napkin pour steers the model to present its answer as a rich artifact via
  // show_napkin; we still synthesize one below as a fallback if it ignores the
  // hint, so "serves napkin" always yields a Napkin.
  const prompt =
    p.output === 'napkin'
      ? `${p.prompt}\n\nPresent your final answer by calling the show_napkin tool (use kind:"markdown" unless another kind fits the content better) rather than only replying in chat.`
      : p.prompt
  const messages: ChatMessage[] = [{ role: 'user', content: prompt }]
  let napkin: Napkin | null = null
  let finalText = ''
  // A headless pour surfaces most failures (step-limit with no answer, context
  // overflow, a blocked tool the model kept retrying) as an 'error' EVENT rather
  // than a thrown exception , the loop emits it then returns normally. Capture
  // it so such a run is recorded as a failure instead of a silent "(no reply)".
  let emittedError: string | null = null
  // A live transcript of the pour so a saved run shows the tool calls, denials,
  // and errors , not just the final text. Reset at the start of each attempt.
  let entries: TranscriptEntry[] = []

  const emit = (e: AgentEvent): void => {
    if (e.type === 'assistant_text') {
      finalText = e.text
      entries.push({ kind: 'assistant', text: e.text })
    } else if (e.type === 'napkin_show') {
      napkin = e.napkin
      entries.push({ kind: 'napkin', napkin: e.napkin })
    } else if (e.type === 'tool_call') {
      entries.push({ kind: 'tool', label: `${e.server} → ${e.tool}`, detail: JSON.stringify(e.args) })
    } else if (e.type === 'tool_result') {
      entries.push({ kind: 'tool', label: `${e.server} ← ${e.tool}`, detail: e.preview, ok: e.ok })
    } else if (e.type === 'error') {
      emittedError = e.message
      entries.push({ kind: 'error', text: e.message })
    }
    // Mirror to the window if the user happens to be watching.
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('agent:event', e)
  }

  // Whitelist approve: allow iff the qualified tool is explicitly permitted.
  // Denied calls still land in the saved transcript (as a blocked tool_result),
  // so a run that failed because a needed tool wasn't enabled is diagnosable.
  // The denial carries a Pitcher-specific reason so the model and transcript say
  // *why* it was blocked (not in this task's allow-list) rather than the
  // misleading "denied by user".
  const approve: ApproveFn = ({ qualified, tool }) =>
    Promise.resolve(
      p.allowedTools.includes(qualified)
        ? true
        : {
            allowed: false,
            reason: `The "${tool}" tool is not in this scheduled task's allowed tools, so it was blocked. To let this task use it, edit the Pitcher and check the tool (it must first be connected in the Pantry).`
          }
    )

  let lastErr: unknown = null
  for (let attempt = 0; attempt < 2; attempt++) {
    napkin = null
    finalText = ''
    emittedError = null
    entries = []
    const abort = new AbortController()
    try {
      // No onLimit/askNapkin: a headless pour must not block on user input, so
      // the loop stops at its step budget and skips clarification instead.
      await agent.run(messages, emit, approve, abort.signal)
      // An emitted error means the loop gave up without a real answer; treat it
      // as a failed attempt so it retries and, if it persists, is recorded as an
      // error rather than a success.
      if (emittedError) {
        lastErr = new Error(emittedError)
        continue
      }
      lastErr = null
      break
    } catch (err) {
      lastErr = err
    }
  }

  const now = Date.now()
  const list = readPitchers(appPath)
  const idx = list.findIndex((x) => x.id === p.id)

  const failed = lastErr != null
  const error = failed
    ? lastErr instanceof Error
      ? lastErr.message
      : String(lastErr)
    : undefined

  // Always persist the run as a saved conversation , success or failure , so the
  // user can reopen it and see exactly what happened. Honor a napkin pour's
  // output preference on success by synthesizing a Napkin when the model didn't.
  const sessionId = randomUUID()
  if (!failed && !napkin && p.output === 'napkin' && finalText.trim())
    napkin = { title: p.name, kind: 'markdown', content: finalText }
  if (napkin && !entries.some((e) => e.kind === 'napkin')) entries.push({ kind: 'napkin', napkin })
  if (entries.length === 0)
    entries.push(
      failed
        ? { kind: 'error', text: error ?? 'The pour failed.' }
        : { kind: 'assistant', text: finalText || '(no reply)' }
    )

  writeSession(appPath, {
    id: sessionId,
    title: `${failed ? '⚠️' : '🥤'} ${p.name}`,
    createdAt: now,
    updatedAt: now,
    messageCount: 2,
    model: config.model,
    history: [...messages, { role: 'assistant', content: finalText }],
    entries
  })

  if (idx >= 0) {
    list[idx] = failed
      ? { ...list[idx], lastStatus: 'error', lastError: error, lastSessionId: sessionId }
      : {
          ...list[idx],
          lastRunAt: now,
          lastStatus: 'ok',
          lastError: undefined,
          lastSessionId: sessionId
        }
    writePitchers(appPath, list)
  }

  if (mainWindow && !mainWindow.isFocused() && Notification.isSupported())
    new Notification(
      failed
        ? { title: `Pitcher failed: ${p.name}`, body: error ?? 'Open the run to see what happened.' }
        : {
            title: `Fresh pour: ${p.name}`,
            body: finalText.slice(0, 120) || 'Ready in your history.'
          }
    ).show()

  emitPitcher({ type: 'pitcher_finished', id: p.id, ok: !failed, sessionId, error })
  return { id: p.id, ok: !failed, sessionId, error }
}

const scheduler = new PitcherScheduler(
  appPath,
  (p) => pourPitcher(p).then(() => undefined),
  // "busy" = an interactive agent turn is in flight; never pour over the user.
  () => currentAgentAbort !== null
)

ipcMain.handle('pitcher:list', () => readPitchers(appPath))

ipcMain.handle('pitcher:save', (_event, pitcher: Pitcher) => {
  const list = readPitchers(appPath)
  const idx = list.findIndex((x) => x.id === pitcher.id)
  if (idx >= 0) list[idx] = pitcher
  else list.push(pitcher)
  writePitchers(appPath, list)
  scheduler.reload()
  return list
})

ipcMain.handle('pitcher:delete', (_event, id: string) => {
  const list = readPitchers(appPath).filter((x) => x.id !== id)
  writePitchers(appPath, list)
  scheduler.reload()
  return list
})

ipcMain.handle('pitcher:run', (_event, id: string): Promise<PitcherRunResult> => {
  const p = readPitchers(appPath).find((x) => x.id === id)
  if (!p) return Promise.resolve({ id, ok: false, error: 'Pitcher not found' })
  return pourPitcher(p)
})

// --- Lifecycle ---------------------------------------------------------------

// Single-instance: only one copy of the app may run at a time. If a second
// launch happens (double-clicked shortcut, "Open" again), the OS hands its
// arguments to the already-running process via the 'second-instance' event and
// the newcomer exits immediately. Without the lock we'd get duplicate windows
// fighting over the same config/history files and the local model server.
const gotSingleInstanceLock = app.requestSingleInstanceLock()
if (!gotSingleInstanceLock) {
  app.quit()
} else {
  app.on('second-instance', () => {
    // A second launch was attempted; surface the existing window instead.
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore()
      mainWindow.show()
      mainWindow.focus()
    }
  })
}

app.whenReady().then(async () => {
  // Turn on file logging first (when LOG_LEVEL=debug or settings.json's
  // "logLevel":"debug"), so MCP connection traces and errors below are captured
  // to a shareable log file. Best-effort; a non-writable path just skips it.
  const logFile = initFileLogging(app.getPath('logs'), config.debug, `v${app.getVersion()}`)
  if (logFile) console.log(`[logger] file logging enabled -> ${logFile}`)

  // Dump the effective config so a run's behaviour can be compared against a
  // known-good one (e.g. dev vs packaged). The model, step budget, and system
  // prompt are the usual sources of dev/packaged divergence since dev reads a
  // project .env + repo config while a packaged build has neither.
  console.log(
    `[config] configDir=${appPath} packaged=${app.isPackaged} model=${config.model} ` +
      `maxSteps=${config.maxSteps} compactThreshold=${config.compactThreshold} ` +
      `requireApproval=${config.requireApproval} contextSize=${config.contextSize ?? 'auto'} ` +
      `systemPromptLen=${config.systemPrompt.length} servers=${config.servers.length}`
  )

  // Give Windows a stable app identity so the taskbar uses our icon and groups
  // windows under one entry. This MUST match electron-builder's appId so the
  // running window maps to the installed shortcut (otherwise the taskbar falls
  // back to a blank/generic icon).
  if (process.platform === 'win32') app.setAppUserModelId('com.lemonadestand.app')

  // Drop the native application menu (File, Edit, View, …). The frameless
  // window has no menu bar to show it in anyway.
  Menu.setApplicationMenu(null)

  // Show the window first, THEN connect MCP servers. In a packaged build the
  // network stack isn't reliably ready the instant whenReady() fires, so an
  // HTTP MCP connect (the Lemonade Gateway) attempted before the first window
  // could fail with a transient "fetch failed" , which is exactly why users saw
  // the gateway stuck on "can't reach this server" until they toggled it. This
  // also keeps the connect's retry/backoff off the window's critical path so a
  // genuinely-offline server never delays the UI appearing.
  createWindow()

  await mcp.connectAll(prepareServers(config.servers))

  // Background reconnect sweep. The Lemonade Gateway MCP entry depends on the
  // Lemonade server, which a user may start AFTER the app. The server-status
  // indicator already recovers on its own (the renderer re-probes /health every
  // few seconds), but an MCP entry that failed its initial connect used to stay
  // stuck on "can't reach this server" until it was manually toggled off and on.
  // Periodically retry any enabled-but-disconnected server so the gateway heals
  // the same way the status pill does, and notify the renderer when something
  // newly connects so the Pantry and the agent's tool list refresh live.
  setInterval(() => {
    if (serverSweepInFlight) return
    serverSweepInFlight = true
    void (async () => {
      try {
        const enabled = prepareServers(readServers(appPath).filter((s) => s.enabled))
        const changed = await mcp.connectMissing(enabled)
        if (changed && mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('servers:changed')
        }
      } catch {
        // Best-effort; the next tick tries again.
      } finally {
        serverSweepInFlight = false
      }
    })()
  }, 5000)

  // Arm scheduled Pitchers now that the window exists: runs on-open tasks and
  // any daily task whose time was missed while the app was closed, then keeps
  // timers for future daily fires.
  scheduler.start()

  // Ensure the active model is loaded at our default context so the budget
  // doesn't revert to the small server fallback after a restart. When the user
  // has pinned a context size in the UI, honor that saved size instead so the
  // server is actually running the window we budget against. Best-effort and
  // non-blocking, the window is already up.
  void lemonade
    .ensureModelLoaded(config.contextSize)
    .then(() => console.log(`[config] active chat model on server: ${lemonade.activeModel}`))
    .catch((err) => console.error('[config] ensureModelLoaded failed:', err))

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', async () => {
  await mcp.closeAll()
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', async () => {
  scheduler.stop()
  await mcp.closeAll()
})
