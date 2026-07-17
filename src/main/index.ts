import { app, BrowserWindow, dialog, ipcMain, Menu } from 'electron'
import { randomUUID } from 'node:crypto'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { AgentEvent, ApprovalDecision, ChatMessage, McpServerState } from '@shared/types'
import {
  loadCatalog,
  loadConfig,
  loadThinkingPhrases,
  pathForServer,
  readServers,
  serverFromCatalog,
  withServerPath,
  writeServers,
  writeSettings
} from './config'
import { LemonadeClient } from './lemonade/client'
import { McpManager } from './mcp/manager'
import { Agent, type ApproveFn } from './agent/loop'

const __dirname = fileURLToPath(new URL('.', import.meta.url))

const appPath = app.getAppPath()
const config = loadConfig(appPath)
const lemonade = new LemonadeClient(
  config.lemonadeBaseUrl,
  config.lemonadeApiKey,
  config.model,
  config.contextSize,
  config.completionReserve
)
const mcp = new McpManager()
const agent = new Agent(lemonade, mcp, config.maxSteps, config.systemPrompt)

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

// In-flight approval prompts: id -> resolver. The agent loop awaits these;
// the renderer resolves them via the 'agent:approve' channel.
const pendingApprovals = new Map<string, (decision: ApprovalDecision) => void>()

// Abort handles for work the user can halt mid-flight. The renderer's stop
// button signals these via the 'agent:cancel' / 'agent:cancel-transcribe'
// channels so a long chat turn or transcription can be interrupted.
let currentAgentAbort: AbortController | null = null
let currentTranscribeAbort: AbortController | null = null

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

ipcMain.handle('agent:get-speak', () => speakEnabled)
ipcMain.handle('agent:set-speak', (_event, enabled: boolean) => {
  speakEnabled = Boolean(enabled)
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

// Effective context-window budget, surfaced in the UI.
ipcMain.handle('agent:context-info', () => lemonade.getContextInfo())

// Reload the chat model with a new runtime context size (server /load).
ipcMain.handle('agent:set-context', (_event, ctxSize: number) =>
  lemonade.setContextSize(ctxSize)
)

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

// Renderer's answer to a tool_approval_request. Resolving the stored promise
// unblocks the agent loop.
ipcMain.on('agent:approve', (_event, id: string, decision: ApprovalDecision) => {
  const resolve = pendingApprovals.get(id)
  if (resolve) {
    pendingApprovals.delete(id)
    resolve(decision)
  }
})

ipcMain.handle('agent:send', async (event, messages: ChatMessage[]) => {
  const send = (agentEvent: AgentEvent): void => {
    if (!event.sender.isDestroyed()) event.sender.send('agent:event', agentEvent)
  }

  // Wrap the emit so a final assistant turn is also synthesized to speech when
  // TTS is on. Fire-and-forget: playback lags the text slightly but never
  // blocks the loop, and a TTS failure only logs.
  const emit = (agentEvent: AgentEvent): void => {
    send(agentEvent)
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

  // Approve callback: auto-allow when approval is disabled or the tool was
  // already "always allowed"; otherwise prompt the renderer and await a reply.
  const approve: ApproveFn = ({ server, tool, qualified, args }) => {
    if (!config.requireApproval || sessionAllow.has(qualified)) return Promise.resolve(true)
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

  const abort = new AbortController()
  currentAgentAbort = abort
  try {
    await agent.run(messages, emit, approve, abort.signal)
  } finally {
    if (currentAgentAbort === abort) currentAgentAbort = null
  }
})

// Renderer's stop button: halt the running agent turn.
ipcMain.on('agent:cancel', () => currentAgentAbort?.abort())

// --- Lifecycle ---------------------------------------------------------------

app.whenReady().then(async () => {
  // Give Windows a stable app identity so the taskbar uses our icon and groups
  // windows under one entry.
  if (process.platform === 'win32') app.setAppUserModelId('com.lemonade.stand')

  // Drop the native application menu (File, Edit, View, …). The frameless
  // window has no menu bar to show it in anyway.
  Menu.setApplicationMenu(null)

  await mcp.connectAll(config.servers)
  createWindow()

  // Ensure the active model is loaded at our default context so the budget
  // doesn't revert to the small server fallback after a restart. Best-effort
  // and non-blocking — the window is already up.
  void lemonade.ensureModelLoaded()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', async () => {
  await mcp.closeAll()
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', async () => {
  await mcp.closeAll()
})
