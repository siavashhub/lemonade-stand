import { useEffect, useMemo, useRef, useState } from 'react'
import type { JSX, UIEvent } from 'react'
import type {
  AgentEvent,
  AgentTool,
  ApprovalDecision,
  CatalogEntry,
  ChatMessage,
  ContextBreakdown,
  ContextInfo,
  DownloadJob,
  McpServerState,
  MessageAttachment,
  MessageContentPart,
  ModelInfo,
  Napkin,
  NapkinChoice,
  NapkinKind,
  Pitcher,
  PitcherEvent,
  PlanStep,
  SessionSummary,
  TranscriptEntry
} from '@shared/types'
import {
  ArchiveBoxIcon,
  ArrowDownTrayIcon,
  ClockIcon,
  CpuChipIcon,
  LightBulbIcon,
  MicrophoneIcon,
  PitcherIcon,
  ShieldCheckIcon,
  ShieldSlashIcon,
  SpeakerWaveIcon,
  SpeakerXMarkIcon,
  StopIcon,
  StopSpeakingIcon,
  TrashIcon
} from './icons'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

// A "trace" line rendered in the transcript. Chat turns and tool activity share
// the same visual stream so you can watch the agent think. Aliased to the shared
// TranscriptEntry so a conversation can be saved and restored verbatim.
type Entry = TranscriptEntry

// A tool call awaiting the user's approval decision.
interface PendingApproval {
  id: string
  server: string
  tool: string
  args: unknown
}

// Reachability of the Lemonade server. 'checking' is the initial/in-flight
// state before the first probe resolves.
type ServerStatus = 'checking' | 'online' | 'offline'

// The two visual themes offered in the top-bar toggle. Persisted in
// localStorage and applied via the data-theme attribute on <html>.
type Theme = 'light' | 'dark'

// The overlays that can be shown one-at-a-time: top-bar/footer popovers
// (connection, context, usage) and the full modals (pantry, models, history).
// A single active-panel value enforces that opening one closes any other.
type Panel = 'connection' | 'context' | 'usage' | 'pantry' | 'models' | 'history' | 'pitchers' | 'menu'

// Convert base64 audio from lemond's TTS into a playable object URL. Rejects
// (rather than swallowing) so callers can surface a playback failure instead of
// leaving the user wondering why it's silent. The optional `register` callback
// receives the live <audio> element before playback starts so callers can keep
// a handle on it (e.g. to stop spoken replies mid-playback).
function playAudio(
  base64: string,
  format: string,
  register?: (audio: HTMLAudioElement) => void
): Promise<void> {
  const mime = format === 'wav' ? 'audio/wav' : format === 'mp3' ? 'audio/mpeg' : `audio/${format}`
  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0))
  const url = URL.createObjectURL(new Blob([bytes], { type: mime }))
  const audio = new Audio(url)
  audio.onended = () => URL.revokeObjectURL(url)
  register?.(audio)
  return audio.play().catch((err) => {
    URL.revokeObjectURL(url)
    throw err
  })
}

// Base64-encode raw bytes in chunks. Encoding the whole buffer in one
// String.fromCharCode(...bytes) call can overflow the argument stack for longer
// recordings, so walk it in fixed-size windows instead.
function bytesToBase64(bytes: Uint8Array): string {
  let binary = ''
  const chunk = 0x8000
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk))
  }
  return btoa(binary)
}

// Lemonade's /audio/transcriptions endpoint only accepts WAV, but the browser's
// MediaRecorder emits WebM/Opus. Decode the recording and re-encode it as a
// 16 kHz mono 16-bit PCM WAV , the format Whisper wants , so the server can read
// it. Returns the WAV bytes.
async function blobToWavBytes(blob: Blob): Promise<Uint8Array> {
  const AudioCtx: typeof AudioContext =
    window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
  const ctx = new AudioCtx()
  try {
    const decoded = await ctx.decodeAudioData(await blob.arrayBuffer())
    const mono = downmixToMono(decoded)
    const resampled = resampleTo(mono, decoded.sampleRate, 16000)
    return encodeWav(resampled, 16000)
  } finally {
    void ctx.close()
  }
}

// Average all channels into a single mono track.
function downmixToMono(buffer: AudioBuffer): Float32Array {
  const { numberOfChannels, length } = buffer
  const out = new Float32Array(length)
  for (let ch = 0; ch < numberOfChannels; ch++) {
    const data = buffer.getChannelData(ch)
    for (let i = 0; i < length; i++) out[i] += data[i]
  }
  if (numberOfChannels > 1) {
    for (let i = 0; i < length; i++) out[i] /= numberOfChannels
  }
  return out
}

// Linear-resample a mono signal to the target sample rate. Whisper expects
// 16 kHz; linear interpolation is more than good enough for speech.
function resampleTo(input: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) return input
  const ratio = fromRate / toRate
  const outLength = Math.round(input.length / ratio)
  const out = new Float32Array(outLength)
  for (let i = 0; i < outLength; i++) {
    const pos = i * ratio
    const idx = Math.floor(pos)
    const frac = pos - idx
    const a = input[idx] ?? 0
    const b = input[idx + 1] ?? a
    out[i] = a + (b - a) * frac
  }
  return out
}

// Encode a mono Float32 signal as a 16-bit PCM WAV container.
function encodeWav(samples: Float32Array, sampleRate: number): Uint8Array {
  const bytesPerSample = 2
  const blockAlign = bytesPerSample // mono
  const byteRate = sampleRate * blockAlign
  const dataSize = samples.length * bytesPerSample
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)

  const writeString = (offset: number, text: string): void => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i))
  }

  writeString(0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM chunk size
  view.setUint16(20, 1, true) // PCM format
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, byteRate, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, 16, true) // bits per sample
  writeString(36, 'data')
  view.setUint32(40, dataSize, true)

  let offset = 44
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true)
    offset += bytesPerSample
  }
  return new Uint8Array(buffer)
}

// Build an accurate, per-model description for an Omni collection. Omni is a
// router recipe: it bundles several component models and routes each request to
// the right one. Prefer the collection's own persona (system_prompt) when set,
// then fall back to listing the bundled components, then a generic summary.
function describeOmni(model: ModelInfo): string {
  const persona = model.systemPrompt?.trim()
  if (persona) {
    const short = persona.length > 140 ? persona.slice(0, 137) + '…' : persona
    return `Omni router , ${short}`
  }
  const parts = model.components ?? []
  if (parts.length > 0) {
    return (
      `Omni router bundling ${parts.length} models: ${parts.join(', ')}. ` +
      'Routes chat, vision, image, speech and transcription to the right one.'
    )
  }
  return (
    'Omni router , loads several models together (chat + image + speech + ' +
    'transcription) and routes each request to the right one.'
  )
}

// Read a File into a `data:` URL so an image can be previewed and, for vision
// models, embedded inline in the outgoing message. Rejects on read failure so
// callers can skip an unreadable attachment instead of hanging.
function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error ?? new Error('read failed'))
    reader.readAsDataURL(file)
  })
}

// Normalize a filesystem path for prefix comparison: unify separators, strip a
// trailing separator, and lowercase (this app ships on Windows, whose paths are
// case-insensitive). Two normalized paths can then be compared as plain strings.
function normalizeFsPath(p: string): string {
  return p
    .replace(/[\\/]+/g, '\\')
    .replace(/\\+$/, '')
    .toLowerCase()
}

// Whether `filePath` lives inside one of the agent's accessible filesystem roots.
// Used to decide whether to tell the model a dropped file is openable (inside a
// root) or unreachable (outside every root), so it doesn't confidently reach for
// a filesystem tool that the sandboxed server will just deny.
function isPathReachable(filePath: string, roots: string[]): boolean {
  if (roots.length === 0) return false
  const f = normalizeFsPath(filePath)
  return roots.some((root) => {
    const r = normalizeFsPath(root)
    return f === r || f.startsWith(r + '\\')
  })
}

// Build the model-facing content for a user turn. When images are attached the
// content becomes a multimodal array (text + image parts) so a vision model can
// see them; otherwise it stays a plain string. Paths are split by whether the
// agent's filesystem tools can actually reach them: in-root files are listed as
// available on disk, while out-of-root files are called out explicitly so the
// model explains the boundary instead of trying (and failing) to open them.
function buildUserContent(
  text: string,
  attachments: MessageAttachment[],
  fsRoots: string[]
): string | MessageContentPart[] {
  const withPath = attachments.filter(
    (a): a is MessageAttachment & { path: string } => !!a.path
  )
  const reachable = withPath.filter((a) => isPathReachable(a.path, fsRoots))
  // Images still travel inline (below), so a picture the agent can't open on
  // disk isn't truly inaccessible , only non-image files are worth flagging.
  const unreachable = withPath.filter(
    (a) => !isPathReachable(a.path, fsRoots) && !(a.kind === 'image' && a.dataUrl)
  )

  const blocks: string[] = []
  if (reachable.length > 0) {
    const list = reachable.map((a) => `- ${a.path}`).join('\n')
    blocks.push(`Attached files (available on disk):\n${list}`)
  }
  if (unreachable.length > 0) {
    const list = unreachable.map((a) => `- ${a.name} (${a.path})`).join('\n')
    blocks.push(
      'These attached files are OUTSIDE the folders your filesystem tools can access, so you ' +
        'cannot open them. Do not attempt to read them. Tell the user to move the file into an ' +
        `allowed folder, or to grant access to its location from the Pantry:\n${list}`
    )
  }
  let composed = text
  if (blocks.length > 0) composed = (text ? `${text}\n\n` : '') + blocks.join('\n\n')

  const images = attachments.filter((a) => a.kind === 'image' && a.dataUrl)
  if (images.length === 0) {
    return composed || '(see attached files)'
  }
  const parts: MessageContentPart[] = []
  if (composed) parts.push({ type: 'text', text: composed })
  for (const img of images) {
    parts.push({ type: 'image_url', image_url: { url: img.dataUrl as string } })
  }
  return parts
}

export function App(): JSX.Element {
  const [entries, setEntries] = useState<Entry[]>([])
  const [history, setHistory] = useState<ChatMessage[]>([])
  const [tools, setTools] = useState<AgentTool[]>([])
  const [input, setInput] = useState('')
  // Files/images the user attached to the next message (via paste or drag-drop).
  // Images are previewed and shown to vision models; any item dragged from disk
  // also carries its absolute path so the agent's filesystem tools can open it.
  const [attachments, setAttachments] = useState<MessageAttachment[]>([])
  // The folders the agent's filesystem tools can actually read/write, derived
  // from enabled Filesystem MCP servers. Lets the composer tell the model (and
  // the user) when a dropped file lives outside that boundary instead of letting
  // it try a filesystem tool that would be denied.
  const [fsRoots, setFsRoots] = useState<string[]>([])
  // True while a drag hovers the composer, to show the drop affordance.
  const [dragOver, setDragOver] = useState(false)
  const [busy, setBusy] = useState(false)
  const [speak, setSpeak] = useState(false)
  // True while a synthesized reply is actively playing, so the UI can offer a
  // stop control (spoken replies are otherwise unstoppable once started).
  const [speaking, setSpeaking] = useState(false)
  // Session-scoped approval bypass. While on, tool calls run without prompting
  // for the current conversation only; starting a new session resets it so the
  // default (env/settings.json) approval behavior applies again.
  const [bypassApprovals, setBypassApprovals] = useState(false)
  const [recording, setRecording] = useState(false)
  const [transcribing, setTranscribing] = useState(false)
  const [approvals, setApprovals] = useState<PendingApproval[]>([])
  // Pending "keep going past the step limit?" prompt, or null when none.
  const [stepLimit, setStepLimit] = useState<{ id: string; steps: number } | null>(null)
  // The agent's current working plan (from the update_plan tool), shown in a
  // sticky bar above the composer. Null when there's no active plan.
  const [plan, setPlan] = useState<PlanStep[] | null>(null)
  // Whether the sticky plan bar is expanded to show the full checklist.
  const [planExpanded, setPlanExpanded] = useState(false)
  // The artifact currently shown in the side Napkin panel (from show_napkin),
  // or null when nothing is displayed.
  const [napkin, setNapkin] = useState<Napkin | null>(null)
  // A pending ask_napkin clarification the user must answer. The agent loop is
  // blocked until the user picks an option; null when there's no open question.
  const [napkinChoice, setNapkinChoice] = useState<{
    id: string
    title: string
    prompt: string
    choices: NapkinChoice[]
  } | null>(null)
  // User-initiated napkin creation form; shown when the side button is clicked.
  const [showNapkinCreator, setShowNapkinCreator] = useState(false)
  const [serverStatus, setServerStatus] = useState<ServerStatus>('checking')
  const [connectionBusy, setConnectionBusy] = useState(false)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [context, setContext] = useState<ContextInfo | null>(null)
  const [contextBusy, setContextBusy] = useState(false)
  const [contextError, setContextError] = useState<string | null>(null)
  // Live per-category context usage for the indicator. Refreshed whenever the
  // conversation or model context changes.
  const [breakdown, setBreakdown] = useState<ContextBreakdown | null>(null)
  const [compacting, setCompacting] = useState(false)
  // Exactly one overlay (popover or modal) can be open at a time. Opening any
  // panel replaces whatever was showing, so clicking a second control always
  // dismisses the first , no stacking of windows on top of each other.
  const [activePanel, setActivePanel] = useState<Panel | null>(null)
  const togglePanel = (p: Panel): void => setActivePanel((cur) => (cur === p ? null : p))
  const closePanel = (): void => setActivePanel(null)
  // Model download/loading state lifted to the app shell so the Models button in
  // the status bar can double as a progress indicator even when the Models panel
  // is closed. `downloads` mirrors the server's active jobs; `modelBusyId` is set
  // while a model is being loaded into memory.
  const [downloads, setDownloads] = useState<Record<string, DownloadJob>>({})
  const [modelBusyId, setModelBusyId] = useState<string | null>(null)
  const [thinkingPhrases, setThinkingPhrases] = useState<string[]>([])
  const [thinkingPhrase, setThinkingPhrase] = useState('')
  const [thinkingTick, setThinkingTick] = useState(0)
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem('theme') as Theme | null) ?? 'dark'
  )
  const [version, setVersion] = useState('')
  // Saved-conversation state. `sessionId` identifies the live conversation being
  // edited; `sessions` backs the history sidebar. `currentTitle` is the
  // auto-generated title once the first exchange has happened.
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [pitchers, setPitchers] = useState<Pitcher[]>([])
  const [sessionId, setSessionId] = useState<string>(() => crypto.randomUUID())
  const [currentTitle, setCurrentTitle] = useState('')
  const createdAtRef = useRef<number>(Date.now())
  const inputRef = useRef<HTMLTextAreaElement>(null)
  // Set right before loading a saved session so the autosave effect skips the
  // render caused purely by the load (which would otherwise bump updatedAt).
  const suppressSaveRef = useRef(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  // Active microphone recorder and the audio chunks it has produced so far.
  // Held in refs (not state) so the MediaRecorder callbacks always see the
  // latest values without re-rendering on every chunk.
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  // The <audio> element for the spoken reply currently playing, if any. Held in
  // a ref so it can be paused/stopped without threading it through render state.
  const currentAudioRef = useRef<HTMLAudioElement | null>(null)
  // Set when the user halts a transcription via the stop button, so the aborted
  // request rejects quietly instead of surfacing as a transcription error.
  const transcribeCancelledRef = useRef(false)

  // Apply and persist the chosen theme by flipping the data-theme attribute the
  // CSS variables key off of.
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  // Swallow drag-and-drop events outside the composer so a file dropped on the
  // window edge can't make Electron navigate away from the app to open it.
  useEffect(() => {
    const prevent = (e: DragEvent): void => e.preventDefault()
    window.addEventListener('dragover', prevent)
    window.addEventListener('drop', prevent)
    return () => {
      window.removeEventListener('dragover', prevent)
      window.removeEventListener('drop', prevent)
    }
  }, [])

  // Refresh the connected-tool catalogue shown in the top bar. Called on mount
  // and whenever the Pantry changes what's stocked.
  function refreshTools(): void {
    window.api.listTools().then(setTools).catch(() => setTools([]))
  }

  // Re-derive which folders the agent's filesystem tools can reach, from the
  // enabled Filesystem MCP server(s). Refreshed alongside tools so the composer
  // always knows the current file-access boundary.
  function refreshFsRoots(): void {
    window.api
      .listServers()
      .then((servers) =>
        setFsRoots(
          servers
            .filter((s) => s.enabled && s.id === 'filesystem' && !!s.path)
            .map((s) => s.path as string)
        )
      )
      .catch(() => setFsRoots([]))
  }

  // Re-read the active model's context budget for the top-bar badge.
  function refreshContext(): void {
    window.api
      .getContextInfo()
      .then(setContext)
      .catch(() => setContext(null))
  }

  // Poll the server's model-download jobs so the status-bar Models button can
  // show live download progress no matter where it was started (or whether the
  // Models panel is open). Finished jobs are cleared from the server so the
  // indicator settles once a download completes.
  useEffect(() => {
    let alive = true
    const poll = async (): Promise<void> => {
      const jobs = await window.api.listDownloads().catch(() => [] as DownloadJob[])
      if (!alive) return
      const byId: Record<string, DownloadJob> = {}
      for (const j of jobs) byId[j.modelName] = j
      setDownloads(byId)
      for (const j of jobs) {
        if (j.complete && j.status === 'completed') {
          void window.api.controlDownload(j.id, 'remove').catch(() => {})
        }
      }
    }
    void poll()
    const timer = window.setInterval(() => void poll(), 2000)
    return () => {
      alive = false
      window.clearInterval(timer)
    }
  }, [])

  // Start a server-owned download and optimistically show it right away (the
  // poll above then keeps it live). Throws so the Models panel can surface a
  // start failure inline.
  const downloadModel = async (id: string): Promise<void> => {
    const job = await window.api.downloadModel(id)
    setDownloads((prev) => ({ ...prev, [job.modelName]: job }))
  }

  // Cancel + remove a download job, dropping it from the indicator immediately.
  const cancelModelDownload = async (job: DownloadJob): Promise<void> => {
    await window.api.controlDownload(job.id, 'cancel').catch(() => {})
    await window.api.controlDownload(job.id, 'remove').catch(() => {})
    setDownloads((prev) => {
      const next = { ...prev }
      delete next[job.modelName]
      return next
    })
  }

  // Aggregate progress across all in-flight downloads, for the status-bar
  // Models button. `null` when nothing is downloading.
  const activeDownloads = Object.values(downloads).filter(
    (j) => j.running || j.status === 'downloading'
  )
  const downloadPercent = activeDownloads.length
    ? Math.round(
        activeDownloads.reduce((sum, j) => sum + (j.percent || 0), 0) / activeDownloads.length
      )
    : null
  const modelLoading = modelBusyId != null

  // Recompute the per-category context usage for the live indicator. Cheap and
  // local (a size estimate in main), so it's safe to call after every turn.
  function refreshBreakdown(msgs: ChatMessage[]): void {
    window.api
      .getContextBreakdown(msgs)
      .then(setBreakdown)
      .catch(() => setBreakdown(null))
  }

  // The "Compact Conversation" button: summarize older messages on demand,
  // adopt the compacted history, and note it in the transcript.
  async function compactNow(): Promise<void> {
    if (compacting || busy) return
    setCompacting(true)
    try {
      const compacted = await window.api.compactHistory(history)
      if (compacted) {
        setHistory(compacted)
        setEntries((e) => [
          ...e,
          { kind: 'warning', text: 'Compacted older messages to free up context.' }
        ])
        refreshBreakdown(compacted)
      } else {
        setEntries((e) => [
          ...e,
          { kind: 'warning', text: 'Nothing to compact yet , the conversation is still short.' }
        ])
      }
      closePanel()
    } catch (err) {
      setEntries((e) => [...e, { kind: 'error', text: `Couldn't compact: ${String(err)}` }])
    } finally {
      setCompacting(false)
    }
  }

  // Reload the saved-conversation list backing the history sidebar.
  function refreshSessions(): void {
    window.api.listSessions().then(setSessions).catch(() => setSessions([]))
  }

  // Reload the configured Pitchers (scheduled tasks) backing the Pitchers panel.
  function refreshPitchers(): void {
    window.api.listPitchers().then(setPitchers).catch(() => setPitchers([]))
  }

  // Persist the live conversation (model history + visual transcript) under the
  // current session id. Called by the autosave effect once a turn settles.
  function persistCurrent(): void {
    if (history.length === 0) return
    const firstUser = history.find((m) => m.role === 'user')
    const firstUserText =
      typeof firstUser?.content === 'string'
        ? firstUser.content
        : (firstUser?.content ?? [])
            .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
            .map((p) => p.text)
            .join(' ')
    const title = currentTitle || (firstUserText || 'New conversation').trim().slice(0, 60)
    window.api
      .saveSession({
        id: sessionId,
        title,
        createdAt: createdAtRef.current,
        updatedAt: Date.now(),
        messageCount: history.length,
        model: context?.model,
        history,
        entries
      })
      .then(setSessions)
      .catch(() => {})
  }

  // Flush the current conversation, then reset to a fresh, empty session so the
  // old one is preserved and a new topic starts clean.
  function newSession(): void {
    persistCurrent()
    setEntries([])
    setHistory([])
    setAttachments([])
    setCurrentTitle('')
    createdAtRef.current = Date.now()
    setSessionId(crypto.randomUUID())
    setPlan(null)
    setPlanExpanded(false)
    setNapkin(null)
    setNapkinChoice(null)
    // Move focus to the composer so the user can immediately start typing
    // instead of leaving focus on the button (which would swallow keystrokes).
    requestAnimationFrame(() => inputRef.current?.focus())
  }

  // Load a saved conversation into the live view so the user can continue it.
  // Saves whatever is current first so nothing is lost when switching.
  function openSession(id: string): void {
    if (id === sessionId) {
      closePanel()
      return
    }
    persistCurrent()
    window.api
      .loadSession(id)
      .then((session) => {
        if (!session) return
        suppressSaveRef.current = true
        setSessionId(session.id)
        setCurrentTitle(session.title)
        createdAtRef.current = session.createdAt
        setEntries(session.entries)
        setHistory(session.history)
        setAttachments([])
        setPlan(null)
        setPlanExpanded(false)
        setNapkin(null)
        setNapkinChoice(null)
        closePanel()
      })
      .catch(() => {})
  }

  // Delete a saved conversation. If it's the one on screen, start fresh so the
  // view doesn't keep re-saving a just-deleted session.
  function removeSession(id: string): void {
    window.api
      .deleteSession(id)
      .then((list) => {
        setSessions(list)
        if (id === sessionId) {
          setEntries([])
          setHistory([])
          setCurrentTitle('')
          createdAtRef.current = Date.now()
          setSessionId(crypto.randomUUID())
        }
      })
      .catch(() => {})
  }

  // Wipe all saved conversations and reset to a fresh, empty session.
  function clearHistory(): void {
    window.api
      .clearSessions()
      .then((list) => {
        setSessions(list)
        setEntries([])
        setHistory([])
        setCurrentTitle('')
        createdAtRef.current = Date.now()
        setSessionId(crypto.randomUUID())
      })
      .catch(() => {})
  }

  useEffect(() => {
    refreshTools()
    refreshFsRoots()
    window.api.getSpeak().then(setSpeak).catch(() => setSpeak(false))
    refreshContext()
    refreshSessions()
    refreshPitchers()
    window.api.getAppVersion().then(setVersion).catch(() => setVersion(''))
    window.api
      .getThinkingPhrases()
      .then(setThinkingPhrases)
      .catch(() => setThinkingPhrases([]))
  }, [])

  // Keep the Pitchers list fresh as scheduled/manual pours change their status,
  // and refresh the history list when a pour saves a new conversation.
  useEffect(() => {
    return window.api.onPitcherEvent((evt: PitcherEvent) => {
      refreshPitchers()
      if (evt.type === 'pitcher_finished' && evt.ok) refreshSessions()
    })
  }, [])

  // Autosave the live conversation whenever a turn settles (busy clears) or the
  // title arrives. Skips the render triggered purely by loading a saved session
  // so merely viewing an old chat doesn't bump its timestamp.
  useEffect(() => {
    if (suppressSaveRef.current) {
      suppressSaveRef.current = false
      return
    }
    if (busy || history.length === 0) return
    persistCurrent()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [history, busy, currentTitle])

  // Once the first full exchange exists, ask the model for a short title. Best
  // effort and non-blocking; failures leave the trimmed-first-message fallback.
  useEffect(() => {
    if (currentTitle || busy) return
    const hasUser = history.some((m) => m.role === 'user')
    const hasAssistant = history.some((m) => m.role === 'assistant')
    if (!hasUser || !hasAssistant) return
    let cancelled = false
    window.api
      .suggestTitle(history)
      .then((title) => {
        if (!cancelled && title) setCurrentTitle(title)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [history, busy, currentTitle])

  // Keep the context-usage indicator current: recompute the per-category split
  // whenever the conversation settles or the model's context window changes.
  useEffect(() => {
    if (busy) return
    refreshBreakdown(history)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [history, busy, context?.contextSize, tools.length])

  // While the agent is working , or audio is being transcribed , cycle a fresh
  // playful phrase. The first phrase is picked when the work starts; subsequent
  // phrases are advanced by the lemon's squeeze finishing (see onAnimationEnd
  // below), so the text swap stays in sync with the animation.
  useEffect(() => {
    if (!(busy || transcribing) || thinkingPhrases.length === 0) return
    setThinkingPhrase(thinkingPhrases[Math.floor(Math.random() * thinkingPhrases.length)])
    setThinkingTick((t) => t + 1)
  }, [busy, transcribing, thinkingPhrases])

  // Advance to the next phrase once the lemon's squeeze completes, and bump the
  // tick so the lemon remounts and squeezes again for the new phrase.
  function nextThinkingPhrase(): void {
    if (thinkingPhrases.length === 0) return
    setThinkingPhrase(thinkingPhrases[Math.floor(Math.random() * thinkingPhrases.length)])
    setThinkingTick((t) => t + 1)
  }

  // Reload the model with a new runtime context size via the server's /load.
  async function applyContextSize(ctxSize: number): Promise<void> {
    setContextBusy(true)
    setContextError(null)
    try {
      const info = await window.api.setContextSize(ctxSize)
      setContext({
        model: info.model,
        contextSize: info.contextSize,
        maxContextWindow: info.maxContextWindow,
        reserve: info.reserve,
        source: info.source
      })
      if (info.error) setContextError(info.error)
      else closePanel()
    } catch (err) {
      setContextError(String(err))
    } finally {
      setContextBusy(false)
    }
  }

  // Point the app at a different Lemonade server. On success, persist happens in
  // main; here we reflect the fresh online/offline result and re-pull anything
  // that depends on the server (models, context budget, connected tools).
  async function applyConnection(baseUrl: string, apiKey: string): Promise<void> {
    setConnectionBusy(true)
    setConnectionError(null)
    try {
      const result = await window.api.setConnection({ baseUrl, apiKey })
      setServerStatus(result.online ? 'online' : 'offline')
      refreshContext()
      refreshTools()
      if (result.online) closePanel()
      else setConnectionError('Saved, but the server is still unreachable at that URL.')
    } catch (err) {
      setConnectionError(String(err))
    } finally {
      setConnectionBusy(false)
    }
  }

  // Poll the Lemonade server's health so the indicator reflects it going up or
  // down while the app is open. Runs immediately, then every 5s.
  useEffect(() => {
    let active = true
    const probe = (): void => {
      window.api
        .checkHealth()
        .then((ok) => {
          if (active) setServerStatus(ok ? 'online' : 'offline')
        })
        .catch(() => {
          if (active) setServerStatus('offline')
        })
    }
    probe()
    const timer = setInterval(probe, 5000)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [entries, approvals, busy, stepLimit])

  // Persistent listener for synthesized speech. TTS is fire-and-forget in the
  // main process, so the 'audio' event often arrives *after* the per-turn
  // handler in send() has already torn down on 'done'. Handling it here , for
  // the app's lifetime , means late audio still plays instead of being dropped.
  useEffect(() => {
    const off = window.api.onAgentEvent((event: AgentEvent) => {
      if (event.type !== 'audio') return
      if (window.api.debug) {
        console.log(`[tts] renderer received audio event: ${event.base64.length} b64 chars`)
      }
      void playAudio(event.base64, event.format, (audio) => {
        // A newer reply supersedes any still-playing one: stop the old audio so
        // replies don't overlap, then track the new element and mark us speaking.
        currentAudioRef.current?.pause()
        currentAudioRef.current = audio
        setSpeaking(true)
        const clear = (): void => {
          if (currentAudioRef.current === audio) {
            currentAudioRef.current = null
            setSpeaking(false)
          }
        }
        audio.addEventListener('ended', clear)
        audio.addEventListener('pause', clear)
      })
        .then(() => {
          if (window.api.debug) console.log('[tts] renderer playback started')
        })
        .catch((err) => {
          setSpeaking(false)
          console.error('[tts] renderer playback failed:', err)
          setEntries((e) => [
            ...e,
            { kind: 'error', text: `Couldn't play spoken reply: ${String(err)}` }
          ])
        })
    })
    return off
  }, [])

  function decide(id: string, decision: ApprovalDecision): void {
    window.api.respondApproval(id, decision)
    setApprovals((a) => a.filter((p) => p.id !== id))
  }

  // Answer the "keep going past the step limit?" prompt. Continuing grants the
  // agent another budget; stopping ends the turn.
  function decideContinue(cont: boolean): void {
    if (stepLimit) window.api.respondStepLimit(stepLimit.id, cont)
    setStepLimit(null)
  }

  // Answer a pending ask_napkin clarification: send the chosen option id back to
  // the (blocked) agent loop and echo the choice into the transcript as if the
  // user had typed it, so the decision is visible and saved.
  function chooseNapkin(choiceId: string): void {
    if (!napkinChoice) return
    window.api.respondNapkinChoice(napkinChoice.id, choiceId)
    const picked = napkinChoice.choices.find((c) => c.id === choiceId)
    setEntries((e) => [...e, { kind: 'user', text: picked ? picked.label : choiceId }])
    setNapkinChoice(null)
  }

  async function toggleSpeak(): Promise<void> {
    const next = await window.api.setSpeak(!speak)
    setSpeak(next)
    // Turning spoken replies off should also silence anything mid-playback.
    if (!next) stopSpeaking()
  }

  // Flip the session-scoped approval bypass. When on, the main process approves
  // tool calls without prompting; the state is reset to off whenever a new
  // session starts (see the effect keyed on sessionId).
  function toggleBypassApprovals(): void {
    const next = !bypassApprovals
    setBypassApprovals(next)
    window.api.setBypassApprovals(next)
  }

  // Lock the approval bypass to the session that enabled it: any time the live
  // conversation changes (new session, opening/deleting one, clearing history),
  // reset the bypass so a fresh session falls back to the default approval
  // behavior configured via the env/settings.json file.
  useEffect(() => {
    setBypassApprovals(false)
    window.api.setBypassApprovals(false)
  }, [sessionId])

  // Halt the spoken reply currently playing (if any). Pausing fires the 'pause'
  // listener wired up in the audio handler, which clears the ref and state.
  function stopSpeaking(): void {
    const audio = currentAudioRef.current
    if (!audio) return
    audio.pause()
    audio.currentTime = 0
    currentAudioRef.current = null
    setSpeaking(false)
  }

  // Transcribe the just-finished recording and drop the recognized text into
  // the composer so the user can review/edit it before sending.
  async function transcribeRecording(): Promise<void> {
    const chunks = chunksRef.current
    chunksRef.current = []
    if (chunks.length === 0) return
    const blob = new Blob(chunks, { type: chunks[0].type || 'audio/webm' })
    transcribeCancelledRef.current = false
    setTranscribing(true)
    try {
      // The server only accepts WAV, so transcode the WebM/Opus recording to a
      // 16 kHz mono WAV before sending.
      const wav = await blobToWavBytes(blob)
      const text = (await window.api.transcribe(bytesToBase64(wav), 'audio/wav')).trim()
      if (text) setInput((prev) => (prev ? `${prev} ${text}` : text))
    } catch (err) {
      // A user-triggered halt rejects the transcribe call; swallow it rather
      // than reporting a failure the user deliberately caused.
      if (!transcribeCancelledRef.current) {
        setEntries((e) => [...e, { kind: 'error', text: `Couldn't transcribe audio: ${String(err)}` }])
      }
    } finally {
      setTranscribing(false)
    }
  }

  // Start capturing from the microphone. Prompts for mic access on first use;
  // the recorded audio is transcribed once recording stops.
  async function startRecording(): Promise<void> {
    if (recording || busy || transcribing) return
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      chunksRef.current = []
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data)
      }
      recorder.onstop = () => {
        // Release the mic so the OS indicator clears, then transcribe.
        stream.getTracks().forEach((track) => track.stop())
        recorderRef.current = null
        void transcribeRecording()
      }
      recorderRef.current = recorder
      recorder.start()
      setRecording(true)
    } catch (err) {
      setEntries((e) => [
        ...e,
        { kind: 'error', text: `Couldn't access the microphone: ${String(err)}` }
      ])
    }
  }

  // Stop the active recording; onstop kicks off transcription.
  function stopRecording(): void {
    const recorder = recorderRef.current
    if (recorder && recorder.state !== 'inactive') recorder.stop()
    setRecording(false)
  }

  function toggleRecording(): void {
    if (recording) stopRecording()
    else void startRecording()
  }

  // Halt whatever the send lemon is currently busy with , an in-flight agent
  // turn or an audio transcription , so the user can immediately start over.
  function stop(): void {
    if (busy) window.api.cancelMessage()
    if (transcribing) {
      transcribeCancelledRef.current = true
      window.api.cancelTranscribe()
    }
  }

  // Turn dropped/pasted File objects into attachments. Images are read into a
  // data URL for the thumbnail (and vision); every item keeps its on-disk path
  // when the OS provides one so the agent can operate on it via file tools.
  async function addFiles(files: File[]): Promise<void> {
    const next: MessageAttachment[] = []
    for (const file of files) {
      const path = window.api.getPathForFile(file) || undefined
      const isImage = file.type.startsWith('image/')
      let dataUrl: string | undefined
      if (isImage) {
        try {
          dataUrl = await readFileAsDataUrl(file)
        } catch {
          // Unreadable image , skip its preview but still attach by path below.
        }
      }
      // Skip items we can neither read nor reference (nothing useful to send).
      if (!path && !dataUrl) continue
      next.push({
        name: file.name || (isImage ? 'pasted-image.png' : 'attachment'),
        kind: isImage ? 'image' : 'file',
        path,
        dataUrl,
        mimeType: file.type || undefined,
        size: file.size
      })
    }
    if (next.length > 0) setAttachments((a) => [...a, ...next])
  }

  function removeAttachment(index: number): void {
    setAttachments((a) => a.filter((_, i) => i !== index))
  }

  async function send(): Promise<void> {
    const text = input.trim()
    const atts = attachments
    if ((!text && atts.length === 0) || busy) return
    setInput('')
    setAttachments([])
    setBusy(true)
    // Clear any previous turn's plan so the sticky bar reflects only this turn.
    setPlan(null)
    setPlanExpanded(false)

    const userMsg: ChatMessage = { role: 'user', content: buildUserContent(text, atts, fsRoots) }
    const nextHistory = [...history, userMsg]
    setHistory(nextHistory)
    setEntries((e) => [
      ...e,
      { kind: 'user', text, attachments: atts.length > 0 ? atts : undefined }
    ])

    const collected: ChatMessage[] = []
    const off = window.api.onAgentEvent((event: AgentEvent) => {
      if (event.type === 'assistant_text') {
        collected.push({ role: 'assistant', content: event.text })
        setEntries((e) => [...e, { kind: 'assistant', text: event.text }])
      } else if (event.type === 'reasoning_delta') {
        // A chunk of live chain-of-thought. Append to the in-progress reasoning
        // block if one is still streaming; otherwise start a fresh one.
        setEntries((e) => {
          const last = e[e.length - 1]
          if (last && last.kind === 'reasoning' && last.streaming) {
            return [...e.slice(0, -1), { ...last, text: last.text + event.text }]
          }
          return [...e, { kind: 'reasoning', text: event.text, streaming: true }]
        })
      } else if (event.type === 'reasoning') {
        // Display-only chain-of-thought: shown in the transcript but never added
        // to `collected` (the model-facing history), so it can't bloat context.
        // This is the final text for the turn , replace the live-streamed preview
        // (authoritative) and drop `streaming` so the panel collapses to a summary.
        setEntries((e) => {
          const last = e[e.length - 1]
          if (last && last.kind === 'reasoning' && last.streaming) {
            return [...e.slice(0, -1), { kind: 'reasoning', text: event.text }]
          }
          return [...e, { kind: 'reasoning', text: event.text }]
        })
      } else if (event.type === 'tool_call') {
        setEntries((e) => [
          ...e,
          {
            kind: 'tool',
            label: `${event.server} → ${event.tool}`,
            detail: JSON.stringify(event.args)
          }
        ])
      } else if (event.type === 'tool_result') {
        setEntries((e) => [
          ...e,
          { kind: 'tool', label: `${event.server} ← ${event.tool}`, detail: event.preview, ok: event.ok }
        ])
      } else if (event.type === 'plan_updated') {
        // The model laid out or revised its plan. Drive the sticky plan bar
        // above the composer; it stays visible and updates in place as the
        // model works through the steps.
        setPlan(event.steps)
      } else if (event.type === 'napkin_show') {
        // The agent put a rich artifact on the side Napkin panel. Open it and
        // drop a reopenable chip into the transcript so it persists with the
        // saved conversation.
        setNapkin(event.napkin)
        setEntries((e) => [...e, { kind: 'napkin', napkin: event.napkin }])
      } else if (event.type === 'napkin_choice_request') {
        // The agent needs the user to pick a direction. The panel renders the
        // choices; the loop stays blocked until chooseNapkin() answers.
        setNapkinChoice({
          id: event.id,
          title: event.title,
          prompt: event.prompt,
          choices: event.choices
        })
      } else if (event.type === 'context_usage') {
        // Live in-flight prompt size (tool calls/results included) , keep the
        // usage badge honest while the agent works, not just between turns.
        setBreakdown(event.breakdown)
      } else if (event.type === 'tool_approval_request') {
        setApprovals((a) => [
          ...a,
          { id: event.id, server: event.server, tool: event.tool, args: event.args }
        ])
      } else if (event.type === 'step_limit_request') {
        setStepLimit({ id: event.id, steps: event.steps })
      } else if (event.type === 'context_warning') {
        const usable = event.contextSize - event.reserve
        const text = event.overflow
          ? `Request too large: ~${event.estimatedTokens} tokens exceed the usable ${usable} of ${event.contextSize} (reserving ${event.reserve} for the reply). It was not sent , shorten the chat, disable tools, or raise the context size.`
          : `Heads up: this request is ~${event.estimatedTokens} tokens, close to the usable ${usable}-token limit (context ${event.contextSize}).`
        setEntries((e) => [...e, { kind: 'warning', text }])
      } else if (event.type === 'history_compacted') {
        // The agent summarized older messages to reclaim context. Adopt the
        // compacted model-facing history; new assistant turns from this same
        // reply are still appended on 'done'. The visual transcript is kept.
        setHistory(event.messages)
        setEntries((e) => [
          ...e,
          { kind: 'warning', text: 'Compacted older messages to free up context.' }
        ])
      } else if (event.type === 'error') {
        setEntries((e) => [...e, { kind: 'error', text: event.message }])
      } else if (event.type === 'done') {
        off()
        setStepLimit(null)
        setHistory((h) => [...h, ...collected])
        setBusy(false)
      }
    })

    try {
      await window.api.sendMessage(nextHistory)
    } catch (err) {
      off()
      setEntries((e) => [...e, { kind: 'error', text: String(err) }])
      setBusy(false)
    }
  }

  // While the model is actively streaming its chain-of-thought, hide the playful
  // "Working" phrase , the reasoning panel is the live indicator. It returns once
  // the thought finishes (or when there's no reasoning at all).
  const lastEntry = entries[entries.length - 1]
  const reasoningActive = lastEntry?.kind === 'reasoning' && !!lastEntry.streaming

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">
          <svg
            className="brand-logo"
            viewBox="0 0 64 64"
            xmlns="http://www.w3.org/2000/svg"
            role="img"
            aria-label="Lemonade Stand logo"
          >
            <rect x="4" y="4" width="56" height="56" rx="14" fill="#f5c542" />
            <path d="M4 18 H60 V24 H4 Z" fill="#e05a5a" />
            <path
              d="M12 18 V24 M24 18 V24 M36 18 V24 M48 18 V24"
              stroke="#ffffff"
              strokeWidth="4"
            />
            <path
              d="M20 31 V50 H31"
              fill="none"
              stroke="#1a1a1e"
              strokeWidth="5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M46 35 C46 31 36 31 36 36 C36 40 46 40 46 45 C46 50 36 50 36 46"
              fill="none"
              stroke="#1a1a1e"
              strokeWidth="5"
              strokeLinecap="round"
            />
          </svg>
          Lemonade Stand
          <div className="brand-menu-wrap">
            <button
              className="hamburger-btn"
              onClick={() => togglePanel('menu')}
              title="Menu"
              aria-label="Open menu"
              aria-expanded={activePanel === 'menu'}
            >
              <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
                <path
                  d="M2 4 H14 M2 8 H14 M2 12 H14"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                />
              </svg>
            </button>
            {activePanel === 'menu' && (
              <div className="brand-menu-popover" role="menu">
                <span className="brand-menu-label">Theme</span>
                <button
                  className={`brand-menu-item ${theme === 'light' ? 'active' : ''}`}
                  onClick={() => {
                    setTheme('light')
                    closePanel()
                  }}
                  role="menuitemradio"
                  aria-checked={theme === 'light'}
                >
                  ☀️ Light
                </button>
                <button
                  className={`brand-menu-item ${theme === 'dark' ? 'active' : ''}`}
                  onClick={() => {
                    setTheme('dark')
                    closePanel()
                  }}
                  role="menuitemradio"
                  aria-checked={theme === 'dark'}
                >
                  🌙 Dark
                </button>
                <div className="brand-menu-divider" />
                <div className="brand-menu-version">
                  {version ? `Lemonade Stand ${version}` : 'Lemonade Stand'}
                </div>
              </div>
            )}
          </div>
        </span>
        <div className="topbar-center">
          <div className="session-seg">
            <button
              className="session-seg-btn primary"
              onClick={newSession}
              disabled={busy}
              title="Save this chat and start a new one"
            >
              ＋ New Session
            </button>
            <button
              className="session-seg-btn"
              onClick={() => setActivePanel('history')}
              title="Browse and continue past conversations"
            >
              <ClockIcon /> History
            </button>
          </div>
        </div>
        <div className="topbar-right">
          <button
            className="pantry-toggle"
            onClick={() => setActivePanel('pitchers')}
            title="Open Pitchers , scheduled tasks poured fresh on a timer or when the app opens"
          >
            <PitcherIcon /> Pitchers
            {pitchers.filter((p) => p.enabled).length > 0 && (
              <span
                className="tools-badge"
                title={`${pitchers.filter((p) => p.enabled).length} active Pitcher(s)`}
              >
                {pitchers.filter((p) => p.enabled).length}
              </span>
            )}
          </button>
          <button
            className="pantry-toggle pantry-toggle-primary"
            onClick={() => setActivePanel('pantry')}
            title={`Open the Pantry , stock tools & skills (${tools.length} tool${
              tools.length === 1 ? '' : 's'
            } connected)`}
          >
            <ArchiveBoxIcon /> Pantry
            <span className="tools-badge" title={`${tools.length} tool${tools.length === 1 ? '' : 's'} connected`}>
              {tools.length}
            </span>
          </button>
          <div className="window-controls">
            <button
              className="win-btn"
              onClick={() => window.api.minimizeWindow()}
              title="Minimize"
              aria-label="Minimize"
            >
              <svg viewBox="0 0 10 10" width="10" height="10" aria-hidden="true">
                <path d="M0 5 H10" stroke="currentColor" strokeWidth="1" />
              </svg>
            </button>
            <button
              className="win-btn"
              onClick={() => window.api.toggleMaximizeWindow()}
              title="Maximize"
              aria-label="Maximize"
            >
              <svg viewBox="0 0 10 10" width="10" height="10" aria-hidden="true">
                <rect x="0.5" y="0.5" width="9" height="9" fill="none" stroke="currentColor" strokeWidth="1" />
              </svg>
            </button>
            <button
              className="win-btn win-close"
              onClick={() => window.api.closeWindow()}
              title="Close"
              aria-label="Close"
            >
              <svg viewBox="0 0 10 10" width="10" height="10" aria-hidden="true">
                <path d="M0 0 L10 10 M10 0 L0 10" stroke="currentColor" strokeWidth="1" />
              </svg>
            </button>
          </div>
        </div>
      </header>

      <div className="stage">
        {/* Subtle napkin creation button on right edge */}
        <button
          className="napkin-creator-btn"
          onClick={() => setShowNapkinCreator(true)}
          title="Create a napkin artifact (code, markdown, diagram, etc.)"
          aria-label="Create napkin"
        >
          +
        </button>
        <div className="chat-col">
          <div className="transcript" ref={scrollRef}>
        {entries.length === 0 && (
          <div className="empty">
            Ask something. Open the <strong><ArchiveBoxIcon /> Pantry</strong> to stock tools &amp; skills the
            agent can use.
          </div>
        )}
        {entries.map((entry, i) => (
          <Line key={i} entry={entry} onOpenNapkin={setNapkin} />
        ))}

        {(busy || transcribing) && !reasoningActive && (
          <div className="thinking" aria-live="polite">
            <span
              key={thinkingTick}
              className="thinking-lemon"
              aria-hidden="true"
              onAnimationEnd={nextThinkingPhrase}
            >
              🍋
            </span>
            <span className="thinking-text shimmer">{thinkingPhrase || 'Working'}</span>
          </div>
        )}

        {approvals.map((p) => (
          <div key={p.id} className="approval">
            <div className="approval-head">
              Allow <span className="approval-tool">{p.server} → {p.tool}</span> to run?
            </div>
            <pre className="approval-args">{JSON.stringify(p.args, null, 2)}</pre>
            <div className="approval-actions">
              <button className="ok" onClick={() => decide(p.id, 'approve')}>
                Allow once
              </button>
              <button className="always" onClick={() => decide(p.id, 'always')}>
                Always allow
              </button>
              <button className="deny" onClick={() => decide(p.id, 'deny')}>
                Deny
              </button>
            </div>
          </div>
        ))}

        {stepLimit && (
          <div className="approval">
            <div className="approval-head">
              The agent has run {stepLimit.steps} steps without finishing. Keep going?
            </div>
            <div className="approval-actions">
              <button className="ok" onClick={() => decideContinue(true)}>
                Continue
              </button>
              <button className="deny" onClick={() => decideContinue(false)}>
                Stop
              </button>
            </div>
          </div>
        )}
      </div>

      {plan && plan.length > 0 && (
        <PlanBar
          steps={plan}
          expanded={planExpanded}
          onToggle={() => setPlanExpanded((v) => !v)}
        />
      )}

      <div
        className={`composer ${busy || transcribing ? 'working' : ''} ${dragOver ? 'dragover' : ''}`}
        onDragOver={(e) => {
          if (e.dataTransfer?.types?.includes('Files')) {
            e.preventDefault()
            setDragOver(true)
          }
        }}
        onDragLeave={(e) => {
          // Ignore leaves bubbling up from child elements; only clear when the
          // pointer actually leaves the composer.
          if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragOver(false)
        }}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          const files = Array.from(e.dataTransfer?.files ?? [])
          if (files.length > 0) void addFiles(files)
        }}
      >
        {attachments.length > 0 && (
          <div className="attachments">
            {attachments.map((att, i) => (
              <div key={i} className={`attachment ${att.kind}`} title={att.path ?? att.name}>
                {att.kind === 'image' && att.dataUrl ? (
                  <img className="attachment-thumb" src={att.dataUrl} alt={att.name} />
                ) : (
                  <span className="attachment-icon" aria-hidden="true">
                    📎
                  </span>
                )}
                <span className="attachment-name">{att.name}</span>
                <button
                  className="attachment-remove"
                  onClick={() => removeAttachment(i)}
                  aria-label={`Remove ${att.name}`}
                  title="Remove"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}
        <div className="composer-row">
        <textarea
          ref={inputRef}
          value={input}
          placeholder={
            recording
              ? 'Listening… click the mic again to stop'
              : transcribing
                ? 'Transcribing your audio…'
                : 'What do you need help with?'
          }
          onChange={(e) => setInput(e.target.value)}
          onPaste={(e) => {
            const files = Array.from(e.clipboardData?.items ?? [])
              .filter((it) => it.kind === 'file')
              .map((it) => it.getAsFile())
              .filter((f): f is File => !!f)
            if (files.length > 0) {
              e.preventDefault()
              void addFiles(files)
            }
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              void send()
            }
          }}
          disabled={busy}
        />
        <button
          className={`mic-btn ${recording ? 'recording' : ''}`}
          onClick={toggleRecording}
          disabled={busy || transcribing}
          aria-label={recording ? 'Stop recording' : 'Speak your message'}
          title={
            recording
              ? 'Stop recording'
              : transcribing
                ? 'Transcribing…'
                : 'Speak your message'
          }
        >
          <MicrophoneIcon />
        </button>
        {busy || transcribing ? (
          <button
            className="send-btn stop-btn"
            onClick={stop}
            aria-label="Stop"
            title="Stop"
          >
            <StopIcon />
          </button>
        ) : (
          <button
            className="send-btn"
            onClick={() => void send()}
            disabled={!input.trim() && attachments.length === 0}
            aria-label="Send"
            title="Send"
          >
            <span className="send-lemon">🍋</span>
          </button>
        )}
        </div>
      </div>

      <footer className="statusbar">
        {context !== null && (
          <div className="context-control statusbar-item">
            <button
              className="context-size"
              onClick={() => togglePanel('context')}
              title={
                `Model context window: ${context.contextSize.toLocaleString()} tokens` +
                (context.maxContextWindow
                  ? ` (max ${context.maxContextWindow.toLocaleString()})`
                  : '') +
                '\nClick to change'
              }
            >
              {context.contextSize.toLocaleString()} ctx ▴
            </button>
            {activePanel === 'context' && (
              <ContextEditor
                info={context}
                busy={contextBusy}
                error={contextError}
                onApply={applyContextSize}
                onClose={closePanel}
              />
            )}
          </div>
        )}
        {breakdown !== null && (
          <div className="context-control statusbar-item">
            <ContextUsageBadge
              breakdown={breakdown}
              open={activePanel === 'usage'}
              onToggle={() => togglePanel('usage')}
            />
            {activePanel === 'usage' && (
              <ContextUsage
                breakdown={breakdown}
                compacting={compacting}
                canCompact={history.length > 0 && !busy}
                onCompact={() => void compactNow()}
              />
            )}
          </div>
        )}
        <div className="statusbar-right">
          <div className="context-control statusbar-item server-connection">
            <button
              className={`server-status ${serverStatus}`}
              onClick={() => togglePanel('connection')}
              title={
                (serverStatus === 'online'
                  ? 'Lemonade server is running'
                  : serverStatus === 'offline'
                    ? 'Lemonade server is unreachable , start lemond to chat'
                    : 'Checking Lemonade server…') + '\nClick to configure the connection'
              }
            >
              <span className="server-dot" />
              {serverStatus === 'online'
                ? 'Server online'
                : serverStatus === 'offline'
                  ? 'Server offline'
                  : 'Checking…'}
            </button>
            {activePanel === 'connection' && (
              <ConnectionEditor
                busy={connectionBusy}
                error={connectionError}
                onApply={applyConnection}
                onClose={closePanel}
              />
            )}
          </div>
          <button
            className={`statusbar-btn ${
              downloadPercent != null ? 'is-downloading' : modelLoading ? 'is-loading' : ''
            }`}
            onClick={() => setActivePanel('models')}
            title={
              downloadPercent != null
                ? `Downloading ${activeDownloads.length} model${activeDownloads.length > 1 ? 's' : ''}… ${downloadPercent}%`
                : modelLoading
                  ? 'Loading a model into memory…'
                  : 'Choose the model the agent runs on'
            }
          >
            <CpuChipIcon /> Models{context?.model ? ` (${context.model})` : ''}
            {downloadPercent != null && (
              <span className="statusbar-tag">↓ {downloadPercent}%</span>
            )}
            {downloadPercent == null && modelLoading && (
              <span className="statusbar-tag">loading…</span>
            )}
            {downloadPercent != null && (
              <span className="statusbar-progress" style={{ width: `${downloadPercent}%` }} />
            )}
          </button>
          <button
            className={`bypass-toggle ${bypassApprovals ? 'on' : ''}`}
            onClick={toggleBypassApprovals}
            title={
              bypassApprovals
                ? 'Bypassing tool approvals for this session , click to require approvals again.\nResets when you start a new conversation.'
                : 'Tool approvals enforced , click to bypass them for this session only.\nResets when you start a new conversation.'
            }
            aria-pressed={bypassApprovals}
            aria-label={
              bypassApprovals ? 'Approvals bypassed (this session)' : 'Approvals enforced'
            }
          >
            {bypassApprovals ? <ShieldSlashIcon /> : <ShieldCheckIcon />}
          </button>
          <button
            className={`speak-toggle ${speaking ? 'stop-speaking' : speak ? 'on' : ''}`}
            onClick={() => (speaking ? stopSpeaking() : void toggleSpeak())}
            title={speaking ? 'Stop speaking' : speak ? 'Spoken replies on' : 'Spoken replies off'}
            aria-label={speaking ? 'Stop speaking' : speak ? 'Spoken replies on' : 'Spoken replies off'}
          >
            {speaking ? <StopSpeakingIcon /> : speak ? <SpeakerWaveIcon /> : <SpeakerXMarkIcon />}
          </button>
        </div>
      </footer>
        </div>
        {(napkin || napkinChoice) && (
          <NapkinPanel
            napkin={napkin}
            choice={napkinChoice}
            theme={theme}
            onChoose={chooseNapkin}
            onClose={() => setNapkin(null)}
            onOpenFolder={async (path) => {
              try {
                await window.api.openFolderInExplorer(path)
              } catch (err) {
                console.error('Failed to open folder:', err)
                const message = err instanceof Error ? err.message : String(err)
                window.alert(`Couldn't open folder:\n${path}\n\n${message}`)
              }
            }}
            isAutoCreated={napkin?.kind === 'markdown' && napkin?.content?.includes('📂 **Saved to:**')}
          />
        )}
        {showNapkinCreator && (
          <NapkinCreatorModal
            defaultTitle={currentTitle || 'Untitled'}
            onClose={() => setShowNapkinCreator(false)}
            onCreateNapkin={(napkin) => {
              setNapkin(napkin)
              setEntries((e) => [...e, { kind: 'napkin', napkin }])
              setShowNapkinCreator(false)
            }}
          />
        )}
      </div>

      {activePanel === 'pantry' && (
        <Pantry
          onClose={closePanel}
          onChanged={() => {
            refreshTools()
            refreshFsRoots()
          }}
        />
      )}
      {activePanel === 'history' && (
        <History
          sessions={sessions}
          activeId={sessionId}
          onOpen={openSession}
          onDelete={removeSession}
          onClear={clearHistory}
          onClose={closePanel}
        />
      )}
      {activePanel === 'pitchers' && (
        <Pitchers
          pitchers={pitchers}
          tools={tools}
          onChanged={setPitchers}
          onOpenSession={openSession}
          onClose={closePanel}
        />
      )}
      {activePanel === 'models' && (
        <Models
          onClose={closePanel}
          onChanged={refreshContext}
          downloads={downloads}
          busyId={modelBusyId}
          setBusyId={setModelBusyId}
          onDownload={downloadModel}
          onCancelDownload={cancelModelDownload}
        />
      )}
    </div>
  )
}

// A popover for changing the model's runtime context window. Reloads the model
// on the server via /load, so applying can take a few seconds.
// The connection editor popover: lets the user point the app at their Lemonade
// server without touching env files. Prefilled with the currently active base
// URL / key, which it fetches on mount. Saving persists the choice in main and
// re-probes the server so the status pill updates immediately.
function ConnectionEditor({
  busy,
  error,
  onApply,
  onClose
}: {
  busy: boolean
  error: string | null
  onApply: (baseUrl: string, apiKey: string) => void
  onClose: () => void
}): JSX.Element {
  const [baseUrl, setBaseUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    window.api
      .getConnection()
      .then((c) => {
        setBaseUrl(c.baseUrl)
        setApiKey(c.apiKey)
      })
      .catch(() => {})
      .finally(() => setLoaded(true))
  }, [])

  const valid = /^https?:\/\/.+/i.test(baseUrl.trim())

  return (
    <div className="context-editor" role="dialog" aria-label="Configure Lemonade server">
      <div className="context-editor-head">
        <strong>Lemonade server</strong>
        <button className="context-editor-close" onClick={onClose} aria-label="Close">
          ×
        </button>
      </div>
      <p className="context-editor-note">
        Point the app at your running <code>lemonade-server</code>. Use the OpenAI-compatible
        base URL, including the <code>/api/v1</code> prefix. Default port is <code>13305</code>.
      </p>
      <label className="conn-label">
        Base URL
        <input
          type="text"
          placeholder="http://localhost:13305/api/v1"
          value={baseUrl}
          disabled={busy || !loaded}
          spellCheck={false}
          onChange={(e) => setBaseUrl(e.target.value)}
        />
      </label>
      <label className="conn-label">
        API key <span className="conn-optional">(optional)</span>
        <input
          type="password"
          placeholder="Only if lemond was started with a key"
          value={apiKey}
          disabled={busy || !loaded}
          spellCheck={false}
          onChange={(e) => setApiKey(e.target.value)}
        />
      </label>
      <div className="context-editor-row">
        <button
          className="context-apply"
          disabled={!valid || busy || !loaded}
          onClick={() => onApply(baseUrl, apiKey)}
        >
          {busy ? 'Connecting…' : 'Save & connect'}
        </button>
      </div>
      {!valid && loaded && (
        <p className="context-editor-err">Enter a full URL starting with http:// or https://.</p>
      )}
      {error && <p className="context-editor-err">{error}</p>}
    </div>
  )
}

// Compact readable token count: 1234 -> 1.2K, 1_500_000 -> 1.5M.
function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
  return String(n)
}

// The always-visible context-usage pill: a percent plus a thin fill bar that
// turns amber then red as the window fills. Clicking opens the breakdown.
function ContextUsageBadge({
  breakdown,
  open,
  onToggle
}: {
  breakdown: ContextBreakdown
  open: boolean
  onToggle: () => void
}): JSX.Element {
  const pct = breakdown.contextSize > 0
    ? Math.min(100, Math.round((breakdown.usedTokens / breakdown.contextSize) * 100))
    : 0
  const level = pct >= 90 ? 'crit' : pct >= 70 ? 'warn' : 'ok'
  return (
    <button
      className={`context-usage-badge ${level} ${open ? 'open' : ''}`}
      onClick={onToggle}
      title={
        `Context used: ${breakdown.usedTokens.toLocaleString()} of ` +
        `${breakdown.contextSize.toLocaleString()} tokens (${pct}%)\nClick for a breakdown`
      }
    >
      <span className="context-usage-bar" aria-hidden="true">
        <span className="context-usage-fill" style={{ width: `${pct}%` }} />
      </span>
      {pct}%
    </button>
  )
}

// The breakdown popover, modeled on the reference indicator: a header total, a
// segmented bar, per-category rows grouped into System / User Context /
// Uncategorized, and a manual "Compact Conversation" action.
function ContextUsage({
  breakdown,
  compacting,
  canCompact,
  onCompact
}: {
  breakdown: ContextBreakdown
  compacting: boolean
  canCompact: boolean
  onCompact: () => void
}): JSX.Element {
  const { contextSize, usedTokens, categories } = breakdown
  // Guard against a missing/NaN reserve (e.g. an out-of-date backend that omits
  // the field) so it can never poison the gradient string or render as "NaN".
  const reserve = Number.isFinite(breakdown.reserve) ? breakdown.reserve : 0
  const pct = (n: number): number =>
    contextSize > 0 && Number.isFinite(n) ? Math.round((n / contextSize) * 1000) / 10 : 0
  const usedPct = contextSize > 0 ? Math.min(100, Math.round((usedTokens / contextSize) * 100)) : 0

  // Legend rows in draw order; the donut ring is built from the same list so
  // colours and proportions stay in sync.
  const items = [
    { key: 'system', label: 'System', color: 'var(--c-system)', tokens: categories.systemInstructions },
    { key: 'tools', label: 'Tools', color: 'var(--c-tools)', tokens: categories.toolDefinitions },
    { key: 'messages', label: 'Messages', color: 'var(--c-messages)', tokens: categories.messages },
    { key: 'results', label: 'Results', color: 'var(--c-results)', tokens: categories.toolResults },
    { key: 'other', label: 'Other', color: 'var(--c-other)', tokens: categories.other }
  ]

  // Build the conic-gradient ring: each category takes its share of the whole
  // window, followed by the hatched reserve slice, then the empty remainder.
  // `frac` guards non-finite inputs so a single bad value can't produce a
  // `NaN%` stop that would invalidate the whole gradient (and hide the ring).
  const frac = (n: number): number =>
    contextSize > 0 && Number.isFinite(n) ? Math.max(0, n / contextSize) : 0
  let acc = 0
  const stops: string[] = []
  for (const it of items) {
    const start = acc * 100
    acc += frac(it.tokens)
    stops.push(`${it.color} ${start}% ${acc * 100}%`)
  }
  const reserveStart = acc * 100
  acc += frac(reserve)
  const reserveEnd = acc * 100
  const ring =
    `conic-gradient(${stops.join(', ')}, ` +
    `var(--c-reserve) ${reserveStart}% ${reserveEnd}%, ` +
    `var(--border) ${reserveEnd}% 100%)`

  return (
    <div className="context-usage-pop usage-a" onClick={(e) => e.stopPropagation()}>
      <div className="usage-a-top">
        <div className="usage-donut" style={{ background: ring }}>
          <div className="usage-donut-hole">
            <b>{usedPct}%</b>
            <small>used</small>
          </div>
        </div>
        <div className="usage-a-meta">
          <div className="usage-a-big">
            {formatTokens(usedTokens)} / {formatTokens(contextSize)}
          </div>
          <div className="usage-a-sub">tokens in context</div>
          <div className="usage-a-sub">~{formatTokens(reserve)} reserved for reply</div>
        </div>
      </div>

      <div className="usage-a-legend">
        {items.map((it) => (
          <div className="usage-a-li" key={it.key}>
            <span className="usage-dot" style={{ background: it.color }} aria-hidden="true" />
            {it.label}
            <span className="usage-a-v">{pct(it.tokens)}%</span>
          </div>
        ))}
        <div className="usage-a-li">
          <span className="usage-dot seg-reserve" aria-hidden="true" />
          Reserved
          <span className="usage-a-v">{pct(reserve)}%</span>
        </div>
      </div>

      <button
        className="usage-compact"
        onClick={onCompact}
        disabled={!canCompact || compacting}
        title={
          canCompact
            ? 'Summarize older messages to reclaim context'
            : 'Nothing to compact yet'
        }
      >
        {compacting ? 'Compacting…' : 'Compact Conversation'}
      </button>
    </div>
  )
}

function ContextEditor({
  info,
  busy,
  error,
  onApply,
  onClose
}: {
  info: ContextInfo
  busy: boolean
  error: string | null
  onApply: (ctxSize: number) => void
  onClose: () => void
}): JSX.Element {
  // The model's advertised max is a *hint* for the loaded model, not a hard cap:
  // the server ultimately clamps or rejects a value it can't honor, and the
  // reported max isn't always present or accurate for every runtime.
  const max = info.maxContextWindow
  const [value, setValue] = useState(String(info.contextSize))

  // Common context sizes shown as quick picks. We don't filter these by the
  // model max , picks above it stay available and are flagged instead.
  const presets = [4096, 8192, 16384, 32768, 65536, 131072]

  const parsed = Number(value)
  // Only a sane lower bound is enforced; exceeding the model's max is allowed.
  const valid = Number.isFinite(parsed) && parsed >= 512
  const overMax = valid && max != null && parsed > max

  // Slider bounds. The slider is a quick way to push toward the model's max, so
  // its ceiling is that max when known, else the largest preset as a fallback.
  const sliderMax = max ?? 131072
  const sliderVal = Math.min(Math.max(valid ? parsed : 512, 512), sliderMax)
  const fillPct = Math.round(((sliderVal - 512) / Math.max(sliderMax - 512, 1)) * 100)
  const fmt = (n: number): string => (n >= 1024 ? `${Math.round(n / 1024)}K` : String(n))

  return (
    <div className="context-editor" role="dialog" aria-label="Change context size">
      <div className="context-editor-head">
        <strong>Context window</strong>
        <button className="context-editor-close" onClick={onClose} aria-label="Close">
          ×
        </button>
      </div>
      <p className="context-editor-note">
        Reload <code>{info.model}</code> with a new runtime context size.
        {info.source === 'override' ? ' A configured override currently pins the budget.' : ''}
      </p>
      {max != null && (
        <div className="context-max-chip">
          <span className="context-max-icon" aria-hidden="true">
            ⓘ
          </span>
          <span className="context-max-text">
            Selected model’s max is <strong>{max.toLocaleString()}</strong> tokens
          </span>
          <button
            className="context-max-use"
            disabled={busy || parsed === max}
            onClick={() => setValue(String(max))}
          >
            Use max
          </button>
        </div>
      )}
      <div className="context-presets">
        {presets.map((p) => (
          <button
            key={p}
            className={
              (Number(value) === p ? 'active' : '') + (max != null && p > max ? ' over-max' : '')
            }
            disabled={busy}
            title={max != null && p > max ? `Above model max (${max.toLocaleString()})` : undefined}
            onClick={() => setValue(String(p))}
          >
            {p >= 1024 ? `${p / 1024}K` : p}
          </button>
        ))}
      </div>
      <div className="context-slider-row">
        <input
          type="range"
          className={'context-slider' + (overMax ? ' over-max' : '')}
          min={512}
          max={sliderMax}
          step={512}
          value={sliderVal}
          disabled={busy}
          onChange={(e) => setValue(e.target.value)}
          style={{ '--fill': `${fillPct}%` } as React.CSSProperties}
          aria-label="Context size slider"
        />
        <div className="context-slider-scale">
          <span>512</span>
          <span>{fmt(sliderMax)}</span>
        </div>
      </div>
      <div className="context-editor-row">
        <input
          type="number"
          min={512}
          step={512}
          value={value}
          disabled={busy}
          onChange={(e) => setValue(e.target.value)}
        />
        <button
          className="context-apply"
          disabled={!valid || busy || parsed === info.contextSize}
          onClick={() => onApply(parsed)}
        >
          {busy ? 'Reloading…' : 'Apply'}
        </button>
      </div>
      {!valid && <p className="context-editor-err">Enter a value of at least 512 tokens.</p>}
      {overMax && (
        <p className="context-editor-hint">
          <span className="context-editor-hint-icon" aria-hidden="true">
            ⚠
          </span>
          Above the model’s advertised max ({max!.toLocaleString()}). The server may clamp it or
          fail to load.
        </p>
      )}
      {error && <p className="context-editor-err">{error}</p>}
    </div>
  )
}

// The conversation-history slide-over (left side): lists saved conversations
// newest-first and lets the user reopen one to continue it, or delete it. The
// currently open conversation is marked so it's clear what's live.
function History({
  sessions,
  activeId,
  onOpen,
  onDelete,
  onClear,
  onClose
}: {
  sessions: SessionSummary[]
  activeId: string
  onOpen: (id: string) => void
  onDelete: (id: string) => void
  onClear: () => void
  onClose: () => void
}): JSX.Element {
  const [query, setQuery] = useState('')
  // Set once the user clicks "Clear all", so the button turns into a confirm.
  const [confirmClear, setConfirmClear] = useState(false)

  const q = query.trim().toLowerCase()
  const filtered = q
    ? sessions.filter((s) => (s.title || '').toLowerCase().includes(q))
    : sessions

  return (
    <div className="pantry-overlay history-overlay" onClick={onClose}>
      <aside className="pantry history-panel" onClick={(e) => e.stopPropagation()}>
        <header className="pantry-head">
          <div>
            <h2>
              <ClockIcon /> History
            </h2>
            <p className="pantry-sub">Reopen a past conversation to pick up where you left off</p>
          </div>
          <button className="pantry-close" onClick={onClose} aria-label="Close" title="Close">
            ✕
          </button>
        </header>
        <div className="history-search">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search conversations…"
            aria-label="Search conversations"
          />
          {sessions.length > 0 &&
            (confirmClear ? (
              <div className="history-confirm">
                <span>Delete all?</span>
                <button
                  className="history-confirm-yes"
                  onClick={() => {
                    onClear()
                    setConfirmClear(false)
                  }}
                >
                  Yes, clear all
                </button>
                <button className="history-confirm-no" onClick={() => setConfirmClear(false)}>
                  Cancel
                </button>
              </div>
            ) : (
              <button
                className="history-clear"
                onClick={() => setConfirmClear(true)}
                title="Delete every saved conversation"
              >
                <TrashIcon /> Clear all
              </button>
            ))}
        </div>
        <div className="history-list">
          {sessions.length === 0 && (
            <div className="empty">No saved conversations yet. They'll appear here once you chat.</div>
          )}
          {sessions.length > 0 && filtered.length === 0 && (
            <div className="empty">No conversations match "{query}".</div>
          )}
          {filtered.map((s) => (
            <div
              key={s.id}
              className={`history-item ${s.id === activeId ? 'active' : ''}`}
            >
              <button className="history-open" onClick={() => onOpen(s.id)} title="Open this conversation">
                <span className="history-title">{s.title || 'Untitled conversation'}</span>
                <span className="history-meta">
                  {new Date(s.updatedAt).toLocaleString()} · {s.messageCount} message
                  {s.messageCount === 1 ? '' : 's'}
                  {s.id === activeId ? ' · current' : ''}
                </span>
              </button>
              <button
                className="history-del"
                onClick={() => onDelete(s.id)}
                aria-label="Delete conversation"
                title="Delete conversation"
              >
                <TrashIcon />
              </button>
            </div>
          ))}
        </div>
      </aside>
    </div>
  )
}

// The Pitchers slide-over: manage scheduled tasks ("Pitchers") that pour fresh
// results on a timer or when the app opens. Each Pitcher is a saved prompt plus
// a trigger, an output target, and a whitelist of tools it may auto-run.
function Pitchers({
  pitchers,
  tools,
  onChanged,
  onOpenSession,
  onClose
}: {
  pitchers: Pitcher[]
  tools: AgentTool[]
  onChanged: (list: Pitcher[]) => void
  onOpenSession: (id: string) => void
  onClose: () => void
}): JSX.Element {
  const [editing, setEditing] = useState<Pitcher | null>(null)
  const [pouringId, setPouringId] = useState<string | null>(null)

  const blank = (): Pitcher => ({
    id: crypto.randomUUID(),
    name: 'Morning brief',
    enabled: true,
    prompt: 'Fetch a website and summarize the key points in a few bullets.',
    trigger: { type: 'daily', at: '08:00' },
    output: 'napkin',
    allowedTools: []
  })

  const save = async (p: Pitcher): Promise<void> => {
    onChanged(await window.api.savePitcher(p))
    setEditing(null)
  }
  const remove = async (id: string): Promise<void> => {
    onChanged(await window.api.deletePitcher(id))
  }
  const toggle = async (p: Pitcher): Promise<void> => {
    onChanged(await window.api.savePitcher({ ...p, enabled: !p.enabled }))
  }
  const pour = async (id: string): Promise<void> => {
    setPouringId(id)
    try {
      const r = await window.api.runPitcher(id)
      onChanged(await window.api.listPitchers())
      if (r.ok && r.sessionId) onOpenSession(r.sessionId)
    } finally {
      setPouringId(null)
    }
  }

  const describeTrigger = (p: Pitcher): string =>
    p.trigger.type === 'on-open' ? 'When the app opens' : `Daily at ${p.trigger.at}`

  return (
    <div className="pantry-overlay history-overlay" onClick={onClose}>
      <aside className="pantry history-panel" onClick={(e) => e.stopPropagation()}>
        <header className="pantry-head">
          <div>
            <h2>
              <PitcherIcon /> Pitchers
            </h2>
            <p className="pantry-sub">
              Scheduled tasks, poured fresh on a timer or when the app opens
            </p>
          </div>
          <button className="pantry-close" onClick={onClose} aria-label="Close" title="Close">
            ✕
          </button>
        </header>

        {editing ? (
          <PitcherEditor
            key={editing.id}
            initial={editing}
            tools={tools}
            onCancel={() => setEditing(null)}
            onSave={save}
          />
        ) : (
          <>
            <div className="history-search">
              <button className="pantry-add" onClick={() => setEditing(blank())}>
                ＋ New Pitcher
              </button>
            </div>
            <div className="history-list">
              {pitchers.length === 0 && (
                <div className="empty">
                  No Pitchers yet. Create one to have the agent pour a fresh result on a
                  schedule.
                </div>
              )}
              {pitchers.map((p) => (
                <div key={p.id} className={`history-item ${p.enabled ? '' : 'disabled'}`}>
                  <button
                    className="history-open"
                    onClick={() => setEditing(p)}
                    title="Edit this Pitcher"
                  >
                    <span className="history-title">
                      {p.enabled ? '🥤' : '⏸'} {p.name}
                    </span>
                    <span className="history-meta">
                      {describeTrigger(p)} · serves {p.output}
                      {p.lastStatus === 'error' && ' · ⚠ last run failed'}
                      {p.lastStatus === 'ok' && p.lastRunAt
                        ? ` · last ${new Date(p.lastRunAt).toLocaleString()}`
                        : ''}
                    </span>
                  </button>
                  <button
                    className="history-del"
                    onClick={() => pour(p.id)}
                    disabled={pouringId !== null}
                    aria-label="Pour now"
                    title="Pour now"
                  >
                    {pouringId === p.id ? '…' : '▶'}
                  </button>
                  <button
                    className="history-del"
                    onClick={() => toggle(p)}
                    aria-label={p.enabled ? 'Disable' : 'Enable'}
                    title={p.enabled ? 'Disable' : 'Enable'}
                  >
                    {p.enabled ? '⏸' : '⏵'}
                  </button>
                  <button
                    className="history-del"
                    onClick={() => remove(p.id)}
                    aria-label="Delete Pitcher"
                    title="Delete Pitcher"
                  >
                    <TrashIcon />
                  </button>
                </div>
              ))}
            </div>
          </>
        )}
      </aside>
    </div>
  )
}

// The create/edit form for a single Pitcher. The tool whitelist is the security
// crux: only the tools checked here are auto-approved when the Pitcher pours
// unattended, so a scheduled task can never run a tool it wasn't granted.
function PitcherEditor({
  initial,
  tools,
  onCancel,
  onSave
}: {
  initial: Pitcher
  tools: AgentTool[]
  onCancel: () => void
  onSave: (p: Pitcher) => void
}): JSX.Element {
  const [draft, setDraft] = useState<Pitcher>(initial)
  const [toolSearch, setToolSearch] = useState<string>('')

  const set = (patch: Partial<Pitcher>): void => setDraft((d) => ({ ...d, ...patch }))

  const toggleTool = (qualified: string): void =>
    setDraft((d) => ({
      ...d,
      allowedTools: d.allowedTools.includes(qualified)
        ? d.allowedTools.filter((t) => t !== qualified)
        : [...d.allowedTools, qualified]
    }))

  const canSave = draft.name.trim().length > 0 && draft.prompt.trim().length > 0

  const filteredTools = useMemo(() => {
    const query = toolSearch.toLowerCase()
    return tools.filter((t) =>
      `${t.serverId} ${t.toolName}`.toLowerCase().includes(query)
    )
  }, [tools, toolSearch])

  return (
    <div className="pitcher-editor">
      <label className="pitcher-field">
        <span>Name</span>
        <input
          type="text"
          value={draft.name}
          onChange={(e) => set({ name: e.target.value })}
          placeholder="Morning brief"
        />
      </label>

      <label className="pitcher-field">
        <span>Prompt</span>
        <textarea
          rows={4}
          value={draft.prompt}
          onChange={(e) => set({ prompt: e.target.value })}
          placeholder="Fetch https://example.com and summarize it in 5 bullets."
        />
      </label>

      <label className="pitcher-field">
        <span>Trigger</span>
        <div className="pitcher-row">
          <select
            value={draft.trigger.type}
            onChange={(e) =>
              set({
                trigger:
                  e.target.value === 'on-open'
                    ? { type: 'on-open' }
                    : { type: 'daily', at: '08:00' }
              })
            }
          >
            <option value="daily">Daily at</option>
            <option value="on-open">When the app opens</option>
          </select>
          {draft.trigger.type === 'daily' && (
            <input
              type="time"
              value={draft.trigger.at}
              onChange={(e) => set({ trigger: { type: 'daily', at: e.target.value } })}
            />
          )}
        </div>
      </label>

      <label className="pitcher-field">
        <span>Serve to</span>
        <select
          value={draft.output}
          onChange={(e) => set({ output: e.target.value as Pitcher['output'] })}
        >
          <option value="napkin">Napkin (rich artifact)</option>
          <option value="chat">Chat (saved conversation)</option>
        </select>
      </label>

      <div className="pitcher-field">
        <span>
          Allowed tools{' '}
          <em className="pitcher-hint">
            (auto-approved during a pour, everything else is blocked)
          </em>
        </span>
        <input
          type="text"
          className="pitcher-search"
          placeholder="Search tools..."
          value={toolSearch}
          onChange={(e) => setToolSearch(e.target.value)}
        />
        <div className="pitcher-tools">
          {tools.length === 0 && (
            <div className="empty">
              No tools connected. Stock some in the Pantry so Pitchers can fetch, read, or
              write.
            </div>
          )}
          {filteredTools.length === 0 && tools.length > 0 && (
            <div className="empty">No tools match your search.</div>
          )}
          {filteredTools.map((t) => (
            <label key={t.qualifiedName} className="pitcher-tool">
              <input
                type="checkbox"
                checked={draft.allowedTools.includes(t.qualifiedName)}
                onChange={() => toggleTool(t.qualifiedName)}
              />
              <span className="pitcher-tool-name">
                {t.serverId} · {t.toolName}
              </span>
            </label>
          ))}
        </div>
      </div>

      <div className="pitcher-actions">
        <button className="history-confirm-no" onClick={onCancel}>
          Cancel
        </button>
        <button
          className="pantry-add"
          disabled={!canSave}
          onClick={() => onSave({ ...draft, name: draft.name.trim(), prompt: draft.prompt.trim() })}
        >
          Save Pitcher
        </button>
      </div>
    </div>
  )
}

// The Models slide-over: lists the models the Lemonade server knows about and
// lets the user load one as the agent's chat model. Models whose labels
// advertise tool-calling are highlighted and sorted first, since those are the
// ones that work well in the agent loop.
function Models({
  onClose,
  onChanged,
  downloads,
  busyId,
  setBusyId,
  onDownload,
  onCancelDownload
}: {
  onClose: () => void
  onChanged: () => void
  /** Live download jobs keyed by model id, owned by the app shell. */
  downloads: Record<string, DownloadJob>
  /** Id of the model currently being loaded into memory, or null. */
  busyId: string | null
  setBusyId: (id: string | null) => void
  /** Start a server-owned download; rejects if the server refuses to start it. */
  onDownload: (id: string) => Promise<void>
  /** Cancel and remove a download job. */
  onCancelDownload: (job: DownloadJob) => Promise<void>
}): JSX.Element {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  // Quick-access filter across the catalogue.
  const [filter, setFilter] = useState<'all' | 'recommended' | 'agent'>('all')
  // Free-text search across model id, capability labels, and Omni components.
  const [search, setSearch] = useState('')
  // Remembers the last-seen download jobs so we can detect when one finishes
  // (transitions to complete, or disappears from the server list) and refresh
  // the model list so the just-downloaded model flips to a Load action.
  const prevDownloads = useRef<Record<string, DownloadJob>>({})

  function refresh(): void {
    setLoading(true)
    window.api
      .listModels()
      .then(setModels)
      .catch(() => setModels([]))
      .finally(() => setLoading(false))
  }

  useEffect(refresh, [])

  // When a tracked download completes (or is cleared from the server), refresh
  // the model list so its card reflects the new on-disk state.
  useEffect(() => {
    const prev = prevDownloads.current
    let finished = false
    for (const name of Object.keys(prev)) {
      const before = prev[name]
      const now = downloads[name]
      const wasActive = before.running || before.status === 'downloading'
      if (wasActive && (!now || now.complete || now.status === 'completed')) finished = true
    }
    prevDownloads.current = downloads
    if (finished) refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [downloads])

  async function load(id: string): Promise<void> {
    setBusyId(id)
    setError(null)
    try {
      setModels(await window.api.loadModel(id))
      onChanged()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      refresh()
    } finally {
      setBusyId(null)
    }
  }

  async function download(id: string): Promise<void> {
    setError(null)
    try {
      await onDownload(id)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  async function cancelDownload(job: DownloadJob): Promise<void> {
    await onCancelDownload(job)
  }

  async function remove(id: string): Promise<void> {
    setBusyId(id)
    setError(null)
    try {
      setModels(await window.api.deleteModel(id))
      onChanged()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      refresh()
    } finally {
      setBusyId(null)
    }
  }

  // The active filter: 'recommended' = Omni collections, 'agent' = tool-calling
  // capable. 'all' passes everything through.
  const matchesFilter = (m: ModelInfo): boolean =>
    filter === 'all' ? true : filter === 'recommended' ? m.omni : m.agentReady

  // Free-text match: case-insensitive substring across the model id, its
  // capability labels, and any Omni component names. An empty query matches all.
  const query = search.trim().toLowerCase()
  const matchesSearch = (m: ModelInfo): boolean => {
    if (!query) return true
    const haystack = [m.id, ...m.labels, ...(m.components ?? [])].join(' ').toLowerCase()
    return haystack.includes(query)
  }

  // Only chat models can drive the agent; surface those first and separately.
  const llms = models.filter((m) => m.type === 'llm')
  const others = models.filter((m) => m.type !== 'llm')

  // Agent-ready models first, then by name, so the best choices float to top.
  // Omni models lead the list since they're recommended for agentic use.
  const rankedLlms = [...llms]
    .sort((a, b) => {
      if (a.omni !== b.omni) return a.omni ? -1 : 1
      if (a.agentReady !== b.agentReady) return a.agentReady ? -1 : 1
      return a.id.localeCompare(b.id)
    })
    .filter((m) => matchesFilter(m) && matchesSearch(m))
  const filteredOthers = others.filter((m) => matchesFilter(m) && matchesSearch(m))
  const nothingMatches = !loading && models.length > 0 && rankedLlms.length === 0 && filteredOthers.length === 0

  return (
    <div className="pantry-overlay" onClick={onClose}>
      <aside className="pantry" onClick={(e) => e.stopPropagation()}>
        <header className="pantry-head">
          <div>
            <h2><CpuChipIcon /> Models</h2>
            <p className="pantry-sub">
              Download or load a model to run the agent · ★ = great for agentic use
            </p>
          </div>
          <button className="pantry-close" onClick={onClose} aria-label="Close" title="Close">
            ✕
          </button>
        </header>

        <div className="models-search">
          <input
            type="search"
            className="models-search-input"
            placeholder="Search models by name or capability…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Search models"
          />
        </div>

        <div className="models-filters" role="group" aria-label="Filter models">
          <button
            className={`filter-chip ${filter === 'all' ? 'on' : ''}`}
            onClick={() => setFilter('all')}
            aria-pressed={filter === 'all'}
          >
            All
          </button>
          <button
            className={`filter-chip ${filter === 'recommended' ? 'on' : ''}`}
            onClick={() => setFilter('recommended')}
            aria-pressed={filter === 'recommended'}
            title="Omni collections — recommended for agentic use"
          >
            ★ Recommended
          </button>
          <button
            className={`filter-chip ${filter === 'agent' ? 'on' : ''}`}
            onClick={() => setFilter('agent')}
            aria-pressed={filter === 'agent'}
            title="Tool-calling models that work well in the agent loop"
          >
            Great for agents
          </button>
        </div>

        <div className="pantry-list">
          {loading && <div className="pantry-empty">Loading models…</div>}
          {!loading && models.length === 0 && (
            <div className="pantry-empty">
              No models reported. Is the Lemonade server running?
            </div>
          )}
          {nothingMatches && (
            <div className="pantry-empty">
              No models match {query ? 'your search' : 'this filter'}.{' '}
              <button
                className="link-inline"
                onClick={() => {
                  setFilter('all')
                  setSearch('')
                }}
              >
                Show all
              </button>
            </div>
          )}
          {error && <div className="card-status err">{error}</div>}

          {rankedLlms.map((m) => (
            <ModelCard
              key={m.id}
              model={m}
              busy={busyId === m.id}
              download={downloads[m.id]}
              onLoad={() => void load(m.id)}
              onDownload={() => void download(m.id)}
              onCancelDownload={() => downloads[m.id] && void cancelDownload(downloads[m.id])}
              onDelete={() => void remove(m.id)}
            />
          ))}

          {filteredOthers.length > 0 && (
            <>
              <div className="pantry-section">Other models (not for chat/agent)</div>
              {filteredOthers.map((m) => (
                <ModelCard
                  key={m.id}
                  model={m}
                  busy={busyId === m.id}
                  download={downloads[m.id]}
                  onLoad={() => void load(m.id)}
                  onDownload={() => void download(m.id)}
                  onCancelDownload={() => downloads[m.id] && void cancelDownload(downloads[m.id])}
                  onDelete={() => void remove(m.id)}
                />
              ))}
            </>
          )}
        </div>

        <footer className="pantry-foot">
          Agentic tasks need a tool-calling model. Loading swaps the server's active model.
        </footer>
      </aside>
    </div>
  )
}

function ModelCard({
  model,
  busy,
  download,
  onLoad,
  onDownload,
  onCancelDownload,
  onDelete
}: {
  model: ModelInfo
  busy: boolean
  download?: DownloadJob
  onLoad: () => void
  onDownload: () => void
  onCancelDownload: () => void
  onDelete: () => void
}): JSX.Element {
  // Two-step confirm for the destructive uninstall, so a stray click can't
  // wipe a multi-GB download.
  const [confirmDelete, setConfirmDelete] = useState(false)
  const ctx = model.maxContextWindow
    ? `${(model.maxContextWindow / 1024).toFixed(0)}K ctx`
    : null
  const size = model.sizeGb ? `${model.sizeGb.toFixed(2)} GB` : null
  // A job is "in flight" until it either completes or errors out.
  const downloading =
    !!download && !download.complete && download.status !== 'cancelled' && download.status !== 'error'
  const failed = download?.status === 'error'
  return (
    <div className={`pantry-card ${model.active ? 'is-on' : ''}`}>
      <div className="card-main">
        <div className="card-title">
          <span className="card-name">{model.id}</span>
          {model.omni && <span className="badge rec">★ Omni · recommended</span>}
          {!model.omni && model.agentReady && (
            <span className="badge rec">★ Great for agents</span>
          )}
          {model.active && <span className="badge">Active</span>}
          {/* Surface the download size up front for models not yet on disk, so
              the user knows how much they're about to pull before clicking. */}
          {!model.downloaded && !downloading && size && (
            <span className="badge size" title="Download size">
              ↓ {size}
            </span>
          )}
        </div>
        <p className="card-blurb">
          {model.omni
            ? describeOmni(model)
            : model.labels.length > 0
              ? model.labels.join(' · ')
              : 'No capability labels reported.'}
        </p>
        <div className="card-status">
          {[
            ctx,
            size,
            model.downloaded ? 'downloaded' : 'not downloaded',
            model.loaded ? '● loaded on server' : null
          ]
            .filter(Boolean)
            .join(' · ')}
        </div>
        {downloading && (
          <div className="dl-progress" role="status" aria-live="polite">
            <div className="dl-bar">
              <div
                className={`dl-fill ${download!.status === 'paused' ? 'paused' : ''}`}
                style={{ width: `${Math.max(2, Math.round(download!.percent))}%` }}
              />
            </div>
            <div className="dl-meta">
              <span>
                {download!.status === 'paused' ? 'Paused' : 'Downloading'} ·{' '}
                {Math.round(download!.percent)}%
              </span>
              <span>
                {formatBytes(download!.bytesDownloaded)}
                {download!.bytesTotal > 0 ? ` / ${formatBytes(download!.bytesTotal)}` : ''}
                {download!.totalFiles && download!.totalFiles > 1
                  ? ` · file ${download!.fileIndex ?? 1}/${download!.totalFiles}`
                  : ''}
              </span>
            </div>
          </div>
        )}
        {failed && (
          <div className="card-status err">
            Download failed{download?.error ? `: ${download.error}` : ''}
          </div>
        )}
        {busy && (
          <div className="dl-progress" role="status" aria-live="polite">
            <div className="dl-bar indeterminate">
              <div className="dl-fill" />
            </div>
            <div className="dl-meta">
              <span>Loading model into memory…</span>
              <span>This can take a moment</span>
            </div>
          </div>
        )}
        {!model.agentReady && model.type === 'llm' && (
          <div className="card-status warn-note">
            No tool-calling label , may not reliably call tools in the agent loop.
          </div>
        )}
      </div>
      <div className="card-actions">
        {downloading ? (
          <button className="btn-off" onClick={onCancelDownload}>
            Cancel
          </button>
        ) : !model.downloaded ? (
          <button className="btn-on" disabled={busy} onClick={onDownload}>
            <ArrowDownTrayIcon /> Download
          </button>
        ) : (
          <button
            className={model.active ? 'btn-off' : 'btn-on'}
            disabled={busy || model.active}
            onClick={onLoad}
            title={
              model.active
                ? 'This is the active chat model'
                : model.loaded
                  ? 'Already loaded in server memory — switch to it instantly'
                  : 'Load this model into server memory (takes a moment)'
            }
          >
            {busy ? 'Loading…' : model.active ? 'Active' : model.loaded ? 'Use' : 'Load'}
          </button>
        )}
        {/* Uninstall is offered only for models actually on disk. A two-step
            confirm guards the destructive delete. */}
        {model.downloaded && !downloading && !confirmDelete && (
          <button
            className="btn-remove"
            disabled={busy}
            onClick={() => setConfirmDelete(true)}
            title="Delete this model from disk to free up space"
          >
            <TrashIcon /> Uninstall
          </button>
        )}
        {model.downloaded && !downloading && confirmDelete && (
          <>
            <button
              className="btn-danger"
              disabled={busy}
              onClick={() => {
                setConfirmDelete(false)
                onDelete()
              }}
              title="Permanently delete this model's files"
            >
              {busy ? 'Deleting…' : 'Confirm delete'}
            </button>
            <button className="btn-off" disabled={busy} onClick={() => setConfirmDelete(false)}>
              Keep
            </button>
          </>
        )}
      </div>
    </div>
  )
}

// Human-readable byte size for download progress (e.g. "1.34 GB", "512 MB").
function formatBytes(bytes: number): string {
  if (!bytes || bytes < 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)))
  const value = bytes / Math.pow(1024, i)
  return `${value.toFixed(i >= 3 ? 2 : i >= 2 ? 1 : 0)} ${units[i]}`
}

// The sticky plan bar shown above the composer while the agent has an active
// plan. Collapsed, it summarizes the current step and overall progress (e.g.
// "Creating README.md… (2/4)"); expanded, it reveals the full checklist. Driven
// by the update_plan tool's plan_updated events.
function PlanBar({
  steps,
  expanded,
  onToggle
}: {
  steps: PlanStep[]
  expanded: boolean
  onToggle: () => void
}): JSX.Element {
  const total = steps.length
  const done = steps.filter((s) => s.status === 'completed').length
  const allDone = done === total

  // The step to summarize when collapsed: the one in progress, else the first
  // not-yet-done step, else the last (everything complete).
  const activeIdx = (() => {
    const ip = steps.findIndex((s) => s.status === 'in-progress')
    if (ip !== -1) return ip
    const pending = steps.findIndex((s) => s.status !== 'completed')
    return pending !== -1 ? pending : total - 1
  })()
  const active = steps[activeIdx]
  const inProgress = active?.status === 'in-progress'

  const summary = allDone
    ? 'Plan complete'
    : `${active?.title ?? 'Working'}${inProgress ? '…' : ''}`
  // Position shown as (current/total): the active step's place in the list, or
  // total/total once everything is finished.
  const position = allDone ? total : activeIdx + 1
  const fillPct = total > 0 ? Math.round((done / total) * 100) : 0

  return (
    <div className={`plan-bar ${expanded ? 'open' : ''} ${allDone ? 'complete' : ''}`}>
      <button
        className="plan-bar-head"
        onClick={onToggle}
        aria-expanded={expanded}
        title={expanded ? 'Collapse plan' : 'Expand plan'}
      >
        <span className="plan-bar-caret" aria-hidden="true">
          ▸
        </span>
        <span className="plan-bar-summary">{summary}</span>
        <span className="plan-bar-count">
          ({position}/{total})
        </span>
        <span className="plan-bar-progress" aria-hidden="true">
          <span className="plan-bar-fill" style={{ width: `${fillPct}%` }} />
        </span>
      </button>
      {expanded && (
        <ul className="plan-list plan-bar-list">
          {steps.map((step, i) => (
            <li key={i} className={`plan-step ${step.status}`}>
              <span className="plan-mark" aria-hidden="true" />
              <span className="plan-step-text">{step.title}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// Condense a chain-of-thought into a one-line gist for the collapsed reasoning
// panel, in the style of "Reasoned about the Omni router's handling…". We take
// the first sentence, strip a leading first-person lead-in ("I'm looking at…",
// "Let me…") so it reads as a recap, cap the length, and trail off with an
// ellipsis so it reads naturally.
function reasoningSummary(text: string): string {
  const firstLine = text.trim().split(/\n+/)[0] ?? ''
  const firstSentence = (firstLine.split(/(?<=[.!?])\s/)[0] ?? firstLine).trim()
  let gist = firstSentence
    .replace(
      /^(i'?m|i am|i've|i have|i'?ll|i will|i need to|i should|i want to|let me|let's|first,?|now,?|okay,?|ok,?|so,?|looking at|checking|considering)\s+/i,
      ''
    )
    .trim()
  if (!gist) gist = firstSentence
  const max = 72
  if (gist.length > max) gist = gist.slice(0, max)
  // Drop any trailing punctuation/ellipsis before we add our own.
  gist = gist.replace(/[\s.,;:…]+$/, '')
  if (!gist) return 'Reasoned it through…'
  return `Reasoned ${gist.charAt(0).toLowerCase()}${gist.slice(1)}…`
}

// The model's chain-of-thought. While it streams it stays expanded so you can
// watch it think; once the turn moves on it collapses to a one-line gist (click
// to re-expand). The body is capped to a few lines and auto-scrolls so a long
// thought never balloons the transcript.
function ReasoningLine({
  entry
}: {
  entry: Extract<Entry, { kind: 'reasoning' }>
}): JSX.Element {
  const streaming = entry.streaming ?? false
  const [open, setOpen] = useState(streaming)
  const bodyRef = useRef<HTMLDivElement>(null)
  // Stick to the bottom as the thought streams , but back off the moment the
  // user scrolls up to read, and re-engage once they scroll back down.
  const stickRef = useRef(true)

  // Collapse automatically the moment streaming finishes; expand when a fresh
  // thought starts streaming.
  useEffect(() => {
    setOpen(streaming)
  }, [streaming])

  // Follow the latest lines as text streams in, unless the user has scrolled up.
  useEffect(() => {
    const body = bodyRef.current
    if (open && body && stickRef.current) body.scrollTop = body.scrollHeight
  }, [entry.text, open])

  function onBodyScroll(e: UIEvent<HTMLDivElement>): void {
    const el = e.currentTarget
    // Within a few px of the bottom counts as "following".
    stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 8
  }

  return (
    <details
      className={`line reasoning ${streaming ? 'streaming' : ''}`}
      open={open}
      onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}
    >
      <summary className="reasoning-summary">
        <LightBulbIcon className="reasoning-icon" />
        <span className="reasoning-gist">
          {streaming ? 'Thinking…' : reasoningSummary(entry.text)}
        </span>
      </summary>
      <div className="reasoning-body" ref={bodyRef} onScroll={onBodyScroll}>
        {entry.text}
      </div>
    </details>
  )
}

function Line({
  entry,
  onOpenNapkin
}: {
  entry: Entry
  onOpenNapkin?: (napkin: Napkin) => void
}): JSX.Element {
  if (entry.kind === 'tool') {
    return (
      <div className={`line tool ${entry.ok === false ? 'tool-err' : ''}`}>
        <span className="tool-label">{entry.label}</span>
        <span className="tool-detail">{entry.detail}</span>
      </div>
    )
  }
  if (entry.kind === 'napkin') {
    const napkin = entry.napkin
    return (
      <div className="line napkin-chip-line">
        <button
          className="napkin-chip"
          onClick={() => onOpenNapkin?.(napkin)}
          title="Reopen on the napkin"
        >
          <span className="napkin-chip-icon" aria-hidden="true">
            🧾
          </span>
          <span className="napkin-chip-title">{napkin.title}</span>
          <span className="napkin-chip-kind">{napkin.kind}</span>
        </button>
      </div>
    )
  }
  if (entry.kind === 'plan') {
    const done = entry.steps.filter((s) => s.status === 'completed').length
    return (
      <div className="line plan">
        <div className="plan-head">
          <span className="plan-title">Plan</span>
          <span className="plan-count">
            {done}/{entry.steps.length}
          </span>
        </div>
        <ul className="plan-list">
          {entry.steps.map((step, i) => (
            <li key={i} className={`plan-step ${step.status}`}>
              <span className="plan-mark" aria-hidden="true" />
              <span className="plan-step-text">{step.title}</span>
            </li>
          ))}
        </ul>
      </div>
    )
  }
  if (entry.kind === 'error') {
    return <div className="line error">{entry.text}</div>
  }
  if (entry.kind === 'warning') {
    return <div className="line warning">{entry.text}</div>
  }
  if (entry.kind === 'reasoning') {
    return <ReasoningLine entry={entry} />
  }
  const attachments = entry.kind === 'user' ? entry.attachments : undefined
  return (
    <div className={`line ${entry.kind}`}>
      <span className="role">{entry.kind === 'user' ? 'You' : 'Agent'}</span>
      <span className="bubble">
        {attachments && attachments.length > 0 && (
          <span className="bubble-attachments">
            {attachments.map((att, i) => (
              <span key={i} className={`attachment ${att.kind}`} title={att.path ?? att.name}>
                {att.kind === 'image' && att.dataUrl ? (
                  <img className="attachment-thumb" src={att.dataUrl} alt={att.name} />
                ) : (
                  <span className="attachment-icon" aria-hidden="true">
                    📎
                  </span>
                )}
                <span className="attachment-name">{att.name}</span>
              </span>
            ))}
          </span>
        )}
        {entry.text}
      </span>
    </div>
  )
}

// The Pantry: a slide-over that showcases every tool/skill in the Market and
// lets the user stock (enable), unstock (disable), or remove them. Changes are
// persisted to config/servers.json and hot-applied by the main process.
function Pantry({
  onClose,
  onChanged
}: {
  onClose: () => void
  onChanged: () => void
}): JSX.Element {
  const [catalog, setCatalog] = useState<CatalogEntry[]>([])
  const [servers, setServers] = useState<McpServerState[]>([])
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('All')
  const [busyId, setBusyId] = useState<string | null>(null)

  function refresh(): void {
    window.api.listCatalog().then(setCatalog).catch(() => setCatalog([]))
    window.api.listServers().then(setServers).catch(() => setServers([]))
  }

  useEffect(refresh, [])

  const stateOf = (id: string): McpServerState | undefined => servers.find((s) => s.id === id)

  // Servers configured by hand that aren't part of the shipped Market.
  const custom = servers.filter((s) => !catalog.some((c) => c.id === s.id))

  const categories = ['All', ...Array.from(new Set(catalog.map((c) => c.category)))]

  const q = query.trim().toLowerCase()
  const visible = catalog.filter((c) => {
    if (category !== 'All' && c.category !== category) return false
    if (!q) return true
    return (
      c.name.toLowerCase().includes(q) ||
      c.blurb.toLowerCase().includes(q) ||
      c.category.toLowerCase().includes(q)
    )
  })

  async function run(op: Promise<McpServerState[]>, id: string): Promise<void> {
    setBusyId(id)
    try {
      setServers(await op)
      onChanged()
    } catch (err) {
      console.error('[pantry] operation failed:', err)
      refresh()
    } finally {
      setBusyId(null)
    }
  }

  const stockedCount = servers.filter((s) => s.enabled).length

  return (
    <div className="pantry-overlay" onClick={onClose}>
      <aside className="pantry" onClick={(e) => e.stopPropagation()}>
        <header className="pantry-head">
          <div>
            <h2><ArchiveBoxIcon /> The Pantry</h2>
            <p className="pantry-sub">Stock your stand with tools &amp; skills · {stockedCount} stocked</p>
          </div>
          <button className="pantry-close" onClick={onClose} aria-label="Close" title="Close">
            ✕
          </button>
        </header>

        <div className="pantry-search">
          <input
            type="text"
            placeholder="Search the Market…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoFocus
          />
        </div>

        <div className="pantry-cats">
          {categories.map((c) => (
            <button
              key={c}
              className={`cat-pill ${category === c ? 'active' : ''}`}
              onClick={() => setCategory(c)}
            >
              {c}
            </button>
          ))}
        </div>

        <div className="pantry-list">
          {visible.length === 0 && <div className="pantry-empty">No tools match “{query}”.</div>}
          {visible.map((entry) => (
            <PantryCard
              key={entry.id}
              entry={entry}
              state={stateOf(entry.id)}
              busy={busyId === entry.id}
              onConfigure={(opts) => run(window.api.configureServer(entry.id, opts), entry.id)}
              onRemove={() => run(window.api.removeServer(entry.id), entry.id)}
            />
          ))}

          {custom.length > 0 && (
            <>
              <div className="pantry-section">Custom (from servers.json)</div>
              {custom.map((s) => (
                <div key={s.id} className="pantry-card">
                  <div className="card-main">
                    <div className="card-title">
                      <span className="card-name">{s.id}</span>
                      <span className="badge">{s.transport}</span>
                    </div>
                    <p className="card-blurb">Hand-configured server.</p>
                    <ServerStatusRow state={s} />
                  </div>
                  <div className="card-actions">
                    <button
                      className={s.enabled ? 'btn-off' : 'btn-on'}
                      disabled={busyId === s.id}
                      onClick={() =>
                        run(window.api.configureServer(s.id, { enabled: !s.enabled }), s.id)
                      }
                    >
                      {s.enabled ? 'Disable' : 'Enable'}
                    </button>
                    <button
                      className="btn-remove"
                      disabled={busyId === s.id}
                      onClick={() => run(window.api.removeServer(s.id), s.id)}
                    >
                      Remove
                    </button>
                  </div>
                </div>
              ))}
            </>
          )}
        </div>

        <footer className="pantry-foot">
          Only stock tools you trust , their actions run with this app's privileges.
        </footer>
      </aside>
    </div>
  )
}

function ServerStatusRow({ state }: { state: McpServerState | undefined }): JSX.Element | null {
  if (!state || !state.enabled) return null
  if (state.error) return <div className="card-status err">Failed to connect: {state.error}</div>
  if (state.connected)
    return (
      <div className="card-status ok">
        ● Stocked · {state.toolCount} tool{state.toolCount === 1 ? '' : 's'}
      </div>
    )
  return <div className="card-status">Enabled , connecting…</div>
}

function PantryCard({
  entry,
  state,
  busy,
  onConfigure,
  onRemove
}: {
  entry: CatalogEntry
  state: McpServerState | undefined
  busy: boolean
  onConfigure: (opts: { enabled: boolean; path?: string }) => void
  onRemove: () => void
}): JSX.Element {
  const [path, setPath] = useState('')
  const [picking, setPicking] = useState(false)

  const configured = Boolean(state)
  const enabled = state?.enabled ?? false
  const isPathBased = Boolean(entry.needsPath)

  // Once configured, seed the field with the folder the server is actually
  // using so the user can see and change it (rather than it disappearing).
  useEffect(() => {
    if (state?.path) setPath(state.path)
  }, [state?.path])

  const currentPath = state?.path ?? ''
  const pathChanged = isPathBased && configured && path.trim() !== '' && path.trim() !== currentPath

  async function browse(): Promise<void> {
    setPicking(true)
    try {
      const chosen = await window.api.pickPath(entry.pathKind ?? 'folder')
      if (chosen) setPath(chosen)
    } finally {
      setPicking(false)
    }
  }

  return (
    <div className={`pantry-card ${enabled ? 'is-on' : ''}`}>
      <div className="card-main">
        <div className="card-title">
          <span className="card-name">{entry.name}</span>
          <span className="badge">{entry.category}</span>
          {entry.recommended && <span className="badge rec">★ Recommended</span>}
        </div>
        <p className="card-blurb">{entry.blurb}</p>
        <ServerStatusRow state={state} />

        {isPathBased && (
          <div className="card-path">
            <label>{entry.pathLabel ?? 'Path'}</label>
            <div className="card-path-row">
              <input
                type="text"
                value={path}
                placeholder={entry.pathKind === 'file' ? 'Choose a file…' : 'Choose a folder…'}
                onChange={(e) => setPath(e.target.value)}
              />
              <button className="btn-browse" disabled={picking} onClick={() => void browse()}>
                Browse…
              </button>
            </div>
            {configured && (
              <div className="card-path-current">
                Current: <code>{currentPath || '(none)'}</code>
                {pathChanged && (
                  <button
                    className="btn-on btn-inline"
                    disabled={busy}
                    onClick={() => onConfigure({ enabled, path: path.trim() })}
                  >
                    {busy ? 'Updating…' : 'Update folder'}
                  </button>
                )}
              </div>
            )}
          </div>
        )}

        {entry.homepage && (
          <a className="card-link" href={entry.homepage} target="_blank" rel="noreferrer">
            Learn more ↗
          </a>
        )}
      </div>

      <div className="card-actions">
        {!configured && (
          <button
            className="btn-on"
            disabled={busy || (isPathBased ? !path.trim() : false)}
            onClick={() => onConfigure({ enabled: true, path: path.trim() || undefined })}
          >
            {busy ? 'Adding…' : 'Add'}
          </button>
        )}
        {configured && (
          <>
            <button
              className={enabled ? 'btn-off' : 'btn-on'}
              disabled={busy}
              onClick={() => onConfigure({ enabled: !enabled })}
            >
              {enabled ? 'Disable' : 'Enable'}
            </button>
            <button className="btn-remove" disabled={busy} onClick={onRemove}>
              Remove
            </button>
          </>
        )}
      </div>
    </div>
  )
}

// --- The Napkin -------------------------------------------------------------
// A side panel that enriches replies with rich artifacts (code the user can
// copy, rendered Markdown, Mermaid diagrams, SVG, images) and multiple-choice
// clarifying prompts. The model drives it via the built-in show_napkin /
// ask_napkin tools. All model-authored markup is sanitized before it touches
// the DOM; raw HTML and live browsing are intentionally not supported.

function NapkinPanel({
  napkin,
  choice,
  theme,
  onChoose,
  onClose,
  onOpenFolder,
  isAutoCreated
}: {
  napkin: Napkin | null
  choice: { id: string; title: string; prompt: string; choices: NapkinChoice[] } | null
  theme: Theme
  onChoose: (choiceId: string) => void
  onClose: () => void
  onOpenFolder?: (path: string) => void
  isAutoCreated?: boolean
}): JSX.Element {
  const [copied, setCopied] = useState(false)
  // Only text-based artifacts have a copyable source; images don't.
  // Also hide copy for auto-created file save napkins
  const canCopy = napkin !== null && napkin.kind !== 'image' && !isAutoCreated

  async function copySource(): Promise<void> {
    if (!napkin) return
    try {
      await navigator.clipboard.writeText(napkin.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard access denied; leave the button state unchanged.
    }
  }

  return (
    <aside className="napkin-panel">
      <header className="napkin-head">
        <div className="napkin-head-titles">
          <span className="napkin-brand">🧾 Napkin</span>
          {napkin && <span className="napkin-title">{napkin.title}</span>}
        </div>
        <div className="napkin-head-actions">
          {napkin && napkin.folderPath && onOpenFolder && (
            <button
              className="napkin-folder"
              onClick={() => onOpenFolder(napkin.folderPath!)}
              title="Open folder in explorer"
            >
              📁
            </button>
          )}
          {canCopy && (
            <button
              className="napkin-copy"
              onClick={() => void copySource()}
              title="Copy the source"
            >
              {copied ? 'Copied ✓' : 'Copy'}
            </button>
          )}
          <button
            className="pantry-close"
            onClick={onClose}
            aria-label="Close napkin"
            title="Close"
          >
            ✕
          </button>
        </div>
      </header>
      <div className="napkin-body">
        {choice && <NapkinChoicePrompt choice={choice} onChoose={onChoose} />}
        {napkin && <NapkinContent napkin={napkin} theme={theme} />}
        {!choice && !napkin && <div className="napkin-empty">Nothing on the napkin.</div>}
      </div>
    </aside>
  )
}

// The clarify prompt: renders the question and one button per option. Picking
// one unblocks the waiting agent loop via onChoose.
function NapkinChoicePrompt({
  choice,
  onChoose
}: {
  choice: { id: string; title: string; prompt: string; choices: NapkinChoice[] }
  onChoose: (choiceId: string) => void
}): JSX.Element {
  return (
    <div className="napkin-choice">
      <div className="napkin-choice-title">{choice.title}</div>
      <p className="napkin-choice-prompt">{choice.prompt}</p>
      <div className="napkin-choice-options">
        {choice.choices.map((c) => (
          <button key={c.id} className="napkin-choice-btn" onClick={() => onChoose(c.id)}>
            {c.label}
          </button>
        ))}
      </div>
    </div>
  )
}

// Dispatches to the right renderer for the artifact's kind.
function NapkinContent({ napkin, theme }: { napkin: Napkin; theme: Theme }): JSX.Element {
  if (napkin.kind === 'code') return <CodeNapkin napkin={napkin} />
  if (napkin.kind === 'image') return <ImageNapkin napkin={napkin} />
  if (napkin.kind === 'mermaid') return <MermaidNapkin napkin={napkin} theme={theme} />
  if (napkin.kind === 'svg') return <SvgNapkin napkin={napkin} />
  return <MarkdownNapkin napkin={napkin} />
}

function CodeNapkin({ napkin }: { napkin: Napkin }): JSX.Element {
  return (
    <pre className="napkin-code">
      <code data-lang={napkin.language ?? 'text'}>{napkin.content}</code>
    </pre>
  )
}

// Markdown is parsed to HTML and then sanitized, since it comes from the model.
function MarkdownNapkin({ napkin }: { napkin: Napkin }): JSX.Element {
  const html = useMemo(
    () => DOMPurify.sanitize(marked.parse(napkin.content) as string),
    [napkin.content]
  )
  return <div className="napkin-md" dangerouslySetInnerHTML={{ __html: html }} />
}

// Raw SVG is sanitized with DOMPurify's SVG profile before it's injected, so
// scripts/foreignObject/event handlers in model output can't execute.
function SvgNapkin({ napkin }: { napkin: Napkin }): JSX.Element {
  const html = useMemo(
    () => DOMPurify.sanitize(napkin.content, { USE_PROFILES: { svg: true, svgFilters: true } }),
    [napkin.content]
  )
  return <div className="napkin-svg" dangerouslySetInnerHTML={{ __html: html }} />
}

function ImageNapkin({ napkin }: { napkin: Napkin }): JSX.Element {
  const mime =
    napkin.mimeType && /^image\/[a-z0-9.+-]+$/i.test(napkin.mimeType)
      ? napkin.mimeType
      : 'image/png'
  const src = `data:${mime};base64,${napkin.content.replace(/\s+/g, '')}`
  return (
    <div className="napkin-img-wrap">
      <img className="napkin-img" src={src} alt={napkin.alt ?? napkin.title} />
    </div>
  )
}

// Mermaid is loaded lazily (it's heavy) and rendered to a sanitized SVG string
// using the strict security level, so diagram source can't inject scripts.
function MermaidNapkin({ napkin, theme }: { napkin: Napkin; theme: Theme }): JSX.Element {
  const [svg, setSvg] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setSvg(null)
    setError(null)
    void (async () => {
      try {
        const mermaid = (await import('mermaid')).default
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: 'strict',
          theme: theme === 'dark' ? 'dark' : 'default'
        })
        const id = `napkin-mermaid-${Math.random().toString(36).slice(2)}`
        const rendered = await mermaid.render(id, napkin.content)
        if (!cancelled) setSvg(rendered.svg)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [napkin.content, theme])

  if (error) {
    return (
      <div className="napkin-mermaid-err">
        <p>Couldn&apos;t render the diagram: {error}</p>
        <pre className="napkin-code">
          <code>{napkin.content}</code>
        </pre>
      </div>
    )
  }
  if (!svg) return <div className="napkin-empty">Rendering diagram…</div>
  return <div className="napkin-mermaid" dangerouslySetInnerHTML={{ __html: svg }} />
}

// Modal for creating a napkin manually. User picks a kind (code, markdown, etc.)
// and fills in content. Auto-populates smart defaults for diagram types (mermaid, svg).
function NapkinCreatorModal({
  defaultTitle,
  onClose,
  onCreateNapkin
}: {
  defaultTitle: string
  onClose: () => void
  onCreateNapkin: (napkin: Napkin) => void
}): JSX.Element {
  const [selectedKind, setSelectedKind] = useState<NapkinKind>('code')
  const [title, setTitle] = useState(defaultTitle)
  const [content, setContent] = useState('')

  const kinds: NapkinKind[] = ['code', 'markdown', 'mermaid', 'svg', 'image']

  // Smart defaults for diagram types
  const getDefaultContent = (kind: NapkinKind): string => {
    if (kind === 'mermaid') {
      return 'graph TD\n  A[Start] --> B[Process]\n  B --> C{Decision}\n  C -->|Yes| D[End]\n  C -->|No| B'
    }
    if (kind === 'svg') {
      return '<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">\n  <circle cx="100" cy="100" r="80" fill="#f39c12" stroke="#333" stroke-width="2"/>\n  <text x="100" y="105" text-anchor="middle" font-size="20" font-weight="bold">SVG</text>\n</svg>'
    }
    return ''
  }

  function handleKindChange(kind: NapkinKind): void {
    setSelectedKind(kind)
    // Auto-populate content for mermaid and svg
    if (kind === 'mermaid' || kind === 'svg') {
      setContent(getDefaultContent(kind))
    } else {
      setContent('')
    }
  }

  function handleCreate(): void {
    const napkin: Napkin = {
      title: title.trim() || defaultTitle,
      kind: selectedKind,
      content: content.trim(),
      language: selectedKind === 'code' ? 'js' : undefined,
      mimeType: selectedKind === 'image' ? 'image/png' : undefined
    }
    onCreateNapkin(napkin)
  }

  return (
    <div className="napkin-creator-overlay" onClick={onClose}>
      <div className="napkin-creator-modal" onClick={(e) => e.stopPropagation()}>
        <div className="creator-head">
          <h2 className="creator-title">New Napkin</h2>
          <button className="creator-close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="creator-body">
          <div className="creator-label">
            <span>Title</span>
            <input
              type="text"
              placeholder="My artifact"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleCreate()
              }}
            />
          </div>

          <div className="creator-label">
            <span>Type</span>
            <div className="creator-kinds">
              {kinds.map((kind) => (
                <button
                  key={kind}
                  className={`kind-btn ${selectedKind === kind ? 'active' : ''}`}
                  onClick={() => handleKindChange(kind)}
                >
                  {kind}
                </button>
              ))}
            </div>
          </div>

          <div className="creator-label">
            <span>Content</span>
            <textarea
              placeholder={
                selectedKind === 'code'
                  ? 'function hello() {\n  return "world"\n}'
                  : selectedKind === 'markdown'
                    ? '# Hello\n\nSome **markdown** content'
                    : selectedKind === 'mermaid'
                      ? 'Edit the diagram…'
                      : selectedKind === 'svg'
                        ? 'Edit the SVG…'
                        : 'Paste image data URL or base64 encoded PNG'
              }
              value={content}
              onChange={(e) => setContent(e.target.value)}
            />
          </div>
        </div>

        <div className="creator-actions">
          <button className="creator-btn cancel" onClick={onClose}>
            Cancel
          </button>
          <button className="creator-btn primary" onClick={handleCreate}>
            Create Napkin
          </button>
        </div>
      </div>
    </div>
  )
}

