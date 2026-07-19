import { app, BrowserWindow, dialog, ipcMain, Menu, Notification, shell } from 'electron'
import { randomUUID } from 'node:crypto'
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync } from 'node:fs'
import { dirname, isAbsolute, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import type {
  AgentEvent,
  ApprovalDecision,
  ChatMessage,
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
  serverFromCatalog,
  withServerPath,
  writePitchers,
  writeServers,
  writeSettings
} from './config'
import { LemonadeClient } from './lemonade/client'
import { McpManager } from './mcp/manager'
import { Agent, type ApproveFn, type AskNapkinFn, type ContinueFn } from './agent/loop'
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
  if (!app.isPackaged) return app.getAppPath()

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
const config = loadConfig(appPath)
const lemonade = new LemonadeClient(
  config.lemonadeBaseUrl,
  config.lemonadeApiKey,
  config.model,
  config.contextSize,
  config.completionReserve
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

// Abort handles for work the user can halt mid-flight. The renderer's stop
// button signals these via the 'agent:cancel' / 'agent:cancel-transcribe'
// channels so a long chat turn or transcription can be interrupted.
let currentAgentAbort: AbortController | null = null
let currentTranscribeAbort: AbortController | null = null

// The main window, captured in createWindow(). Headless Pitcher pours use it to
// mirror events to an open UI and to raise desktop notifications.
let mainWindow: BrowserWindow | null = null

function createWindow(): void {
  const window = new BrowserWindow({
    width: 1000,
    height: 720,
    title: 'Lemonade Stand',
    // Frameless: no native OS title bar. The renderer draws its own top bar and
    // window controls (see App.tsx / styles.css).
    frame: false,
    icon: join(app.getAppPath(), 'resources', process.platform === 'win32' ? 'icon.ico' : 'icon.png'),
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
  const catalog = loadCatalog(appPath)
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

// Reconnect only the enabled servers from the current on-disk config. Called
// after any change so tool availability tracks the user's choices live.
async function reloadServers(): Promise<void> {
  const all = readServers(appPath)
  await mcp.closeAll()
  await mcp.connectAll(all.filter((s) => s.enabled))
}

ipcMain.handle('catalog:list', () => loadCatalog(appPath))
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
        const entry = loadCatalog(appPath).find((c) => c.id === id)
        if (entry) updated = withServerPath(entry, updated, opts.path)
      }
      all[idx] = updated
    } else {
      const entry = loadCatalog(appPath).find((c) => c.id === id)
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
  const catalog = loadCatalog(appPath)
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

  const abort = new AbortController()
  currentAgentAbort = abort
  try {
    await agent.run(messages, emit, approve, abort.signal, onLimit, askNapkin)
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
})

// --- Pitcher: scheduled tasks ------------------------------------------------

function emitPitcher(evt: PitcherEvent): void {
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('pitcher:event', evt)
}

// Run one Pitcher end-to-end. Auto-approves only tools on its allowlist; every
// other tool call is denied so a scheduled task can't be steered (e.g. by
// injected web content) into actions it was never granted. The pour is saved as
// a normal conversation the user can reopen, and a desktop notification is
// raised when the window isn't focused. Bounded retry guards against a flaky
// local model that occasionally fails to call its tools.
async function pourPitcher(p: Pitcher): Promise<PitcherRunResult> {
  emitPitcher({ type: 'pitcher_started', id: p.id })

  const messages: ChatMessage[] = [{ role: 'user', content: p.prompt }]
  let napkin: Napkin | null = null
  let finalText = ''

  const emit = (e: AgentEvent): void => {
    if (e.type === 'assistant_text') finalText = e.text
    if (e.type === 'napkin_show') napkin = e.napkin
    // Mirror to the window if the user happens to be watching.
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('agent:event', e)
  }

  // Whitelist approve: allow iff the qualified tool is explicitly permitted.
  const approve: ApproveFn = ({ qualified }) => Promise.resolve(p.allowedTools.includes(qualified))

  let lastErr: unknown = null
  for (let attempt = 0; attempt < 2; attempt++) {
    napkin = null
    finalText = ''
    const abort = new AbortController()
    try {
      // No onLimit/askNapkin: a headless pour must not block on user input, so
      // the loop stops at its step budget and skips clarification instead.
      await agent.run(messages, emit, approve, abort.signal)
      lastErr = null
      break
    } catch (err) {
      lastErr = err
    }
  }

  const now = Date.now()
  const list = readPitchers(appPath)
  const idx = list.findIndex((x) => x.id === p.id)

  if (lastErr) {
    const error = lastErr instanceof Error ? lastErr.message : String(lastErr)
    if (idx >= 0) {
      list[idx] = { ...list[idx], lastStatus: 'error', lastError: error }
      writePitchers(appPath, list)
    }
    if (mainWindow && !mainWindow.isFocused() && Notification.isSupported())
      new Notification({ title: `Pitcher failed: ${p.name}`, body: error }).show()
    emitPitcher({ type: 'pitcher_finished', id: p.id, ok: false, error })
    return { id: p.id, ok: false, error }
  }

  // Persist the pour as a normal saved conversation the user can reopen later.
  const sessionId = randomUUID()
  const entries: TranscriptEntry[] = [{ kind: 'assistant', text: finalText || '(no reply)' }]
  if (napkin) entries.push({ kind: 'napkin', napkin })
  writeSession(appPath, {
    id: sessionId,
    title: `🥤 ${p.name}`,
    createdAt: now,
    updatedAt: now,
    messageCount: 2,
    model: config.model,
    history: [...messages, { role: 'assistant', content: finalText }],
    entries
  })

  if (idx >= 0) {
    list[idx] = { ...list[idx], lastRunAt: now, lastStatus: 'ok', lastError: undefined }
    writePitchers(appPath, list)
  }
  if (mainWindow && !mainWindow.isFocused() && Notification.isSupported())
    new Notification({
      title: `Fresh pour: ${p.name}`,
      body: finalText.slice(0, 120) || 'Ready in your history.'
    }).show()

  emitPitcher({ type: 'pitcher_finished', id: p.id, ok: true, sessionId })
  return { id: p.id, ok: true, sessionId }
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
  // windows under one entry.
  if (process.platform === 'win32') app.setAppUserModelId('com.lemonade.stand')

  // Drop the native application menu (File, Edit, View, …). The frameless
  // window has no menu bar to show it in anyway.
  Menu.setApplicationMenu(null)

  await mcp.connectAll(config.servers)
  createWindow()

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
