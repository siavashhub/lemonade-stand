import { useEffect, useRef, useState } from 'react'
import type { JSX } from 'react'
import type {
  AgentEvent,
  AgentTool,
  ApprovalDecision,
  CatalogEntry,
  ChatMessage,
  ContextBreakdown,
  ContextInfo,
  McpServerState,
  ModelInfo,
  PlanStep,
  SessionSummary,
  TranscriptEntry
} from '@shared/types'
import {
  ArchiveBoxIcon,
  ClockIcon,
  CpuChipIcon,
  MicrophoneIcon,
  SpeakerWaveIcon,
  SpeakerXMarkIcon,
  StopIcon,
  StopSpeakingIcon,
  TrashIcon
} from './icons'

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
type Panel = 'connection' | 'context' | 'usage' | 'pantry' | 'models' | 'history'

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
// 16 kHz mono 16-bit PCM WAV — the format Whisper wants — so the server can read
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
    return `Omni router — ${short}`
  }
  const parts = model.components ?? []
  if (parts.length > 0) {
    return (
      `Omni router bundling ${parts.length} models: ${parts.join(', ')}. ` +
      'Routes chat, vision, image, speech and transcription to the right one.'
    )
  }
  return (
    'Omni router — loads several models together (chat + image + speech + ' +
    'transcription) and routes each request to the right one.'
  )
}

export function App(): JSX.Element {
  const [entries, setEntries] = useState<Entry[]>([])
  const [history, setHistory] = useState<ChatMessage[]>([])
  const [tools, setTools] = useState<AgentTool[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [speak, setSpeak] = useState(false)
  // True while a synthesized reply is actively playing, so the UI can offer a
  // stop control (spoken replies are otherwise unstoppable once started).
  const [speaking, setSpeaking] = useState(false)
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
  // dismisses the first — no stacking of windows on top of each other.
  const [activePanel, setActivePanel] = useState<Panel | null>(null)
  const togglePanel = (p: Panel): void => setActivePanel((cur) => (cur === p ? null : p))
  const closePanel = (): void => setActivePanel(null)
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
  const [sessionId, setSessionId] = useState<string>(() => crypto.randomUUID())
  const [currentTitle, setCurrentTitle] = useState('')
  const createdAtRef = useRef<number>(Date.now())
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

  // Refresh the connected-tool catalogue shown in the top bar. Called on mount
  // and whenever the Pantry changes what's stocked.
  function refreshTools(): void {
    window.api.listTools().then(setTools).catch(() => setTools([]))
  }

  // Re-read the active model's context budget for the top-bar badge.
  function refreshContext(): void {
    window.api
      .getContextInfo()
      .then(setContext)
      .catch(() => setContext(null))
  }

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
          { kind: 'warning', text: 'Nothing to compact yet — the conversation is still short.' }
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

  // Persist the live conversation (model history + visual transcript) under the
  // current session id. Called by the autosave effect once a turn settles.
  function persistCurrent(): void {
    if (history.length === 0) return
    const firstUser = history.find((m) => m.role === 'user')
    const title =
      currentTitle || (firstUser?.content ?? 'New conversation').trim().slice(0, 60)
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
    setCurrentTitle('')
    createdAtRef.current = Date.now()
    setSessionId(crypto.randomUUID())
    setPlan(null)
    setPlanExpanded(false)
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
        setPlan(null)
        setPlanExpanded(false)
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
    window.api.getSpeak().then(setSpeak).catch(() => setSpeak(false))
    refreshContext()
    refreshSessions()
    window.api.getAppVersion().then(setVersion).catch(() => setVersion(''))
    window.api
      .getThinkingPhrases()
      .then(setThinkingPhrases)
      .catch(() => setThinkingPhrases([]))
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

  // While the agent is working — or audio is being transcribed — cycle a fresh
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
  // handler in send() has already torn down on 'done'. Handling it here — for
  // the app's lifetime — means late audio still plays instead of being dropped.
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

  async function toggleSpeak(): Promise<void> {
    const next = await window.api.setSpeak(!speak)
    setSpeak(next)
    // Turning spoken replies off should also silence anything mid-playback.
    if (!next) stopSpeaking()
  }

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

  // Halt whatever the send lemon is currently busy with — an in-flight agent
  // turn or an audio transcription — so the user can immediately start over.
  function stop(): void {
    if (busy) window.api.cancelMessage()
    if (transcribing) {
      transcribeCancelledRef.current = true
      window.api.cancelTranscribe()
    }
  }

  async function send(): Promise<void> {
    const text = input.trim()
    if (!text || busy) return
    setInput('')
    setBusy(true)
    // Clear any previous turn's plan so the sticky bar reflects only this turn.
    setPlan(null)
    setPlanExpanded(false)

    const userMsg: ChatMessage = { role: 'user', content: text }
    const nextHistory = [...history, userMsg]
    setHistory(nextHistory)
    setEntries((e) => [...e, { kind: 'user', text }])

    const collected: ChatMessage[] = []
    const off = window.api.onAgentEvent((event: AgentEvent) => {
      if (event.type === 'assistant_text') {
        collected.push({ role: 'assistant', content: event.text })
        setEntries((e) => [...e, { kind: 'assistant', text: event.text }])
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
      } else if (event.type === 'context_usage') {
        // Live in-flight prompt size (tool calls/results included) — keep the
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
          ? `Request too large: ~${event.estimatedTokens} tokens exceed the usable ${usable} of ${event.contextSize} (reserving ${event.reserve} for the reply). It was not sent — shorten the chat, disable tools, or raise the context size.`
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

  return (
    <div className="app">
      <header className="topbar">
        <span
          className="brand"
          title={version ? `Lemonade Stand ${version}` : 'Lemonade Stand'}
        >
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
            className="speak-toggle"
            onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
            title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            aria-label="Toggle color theme"
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
          <button
            className="pantry-toggle pantry-toggle-primary"
            onClick={() => setActivePanel('pantry')}
            title={`Open the Pantry — stock tools & skills (${tools.length} tool${
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

      <div className="transcript" ref={scrollRef}>
        {entries.length === 0 && (
          <div className="empty">
            Ask something. Open the <strong><ArchiveBoxIcon /> Pantry</strong> to stock tools &amp; skills the
            agent can use.
          </div>
        )}
        {entries.map((entry, i) => (
          <Line key={i} entry={entry} />
        ))}

        {(busy || transcribing) && (
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

      <div className={`composer ${busy || transcribing ? 'working' : ''}`}>
        <textarea
          value={input}
          placeholder={
            recording
              ? 'Listening… click the mic again to stop'
              : transcribing
                ? 'Transcribing your audio…'
                : 'What do you need help with?'
          }
          onChange={(e) => setInput(e.target.value)}
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
            disabled={!input.trim()}
            aria-label="Send"
            title="Send"
          >
            <span className="send-lemon">🍋</span>
          </button>
        )}
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
                    ? 'Lemonade server is unreachable — start lemond to chat'
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
            className="statusbar-btn"
            onClick={() => setActivePanel('models')}
            title="Choose the model the agent runs on"
          >
            <CpuChipIcon /> Models{context?.model ? ` (${context.model})` : ''}
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

      {activePanel === 'pantry' && (
        <Pantry onClose={closePanel} onChanged={refreshTools} />
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
      {activePanel === 'models' && (
        <Models
          onClose={closePanel}
          onChanged={refreshContext}
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
  const max = info.maxContextWindow ?? 131072
  const [value, setValue] = useState(String(info.contextSize))

  // Common context sizes, capped at the model's advertised maximum.
  const presets = [4096, 8192, 16384, 32768, 65536].filter((p) => p <= max)

  const parsed = Number(value)
  const valid = Number.isFinite(parsed) && parsed >= 512 && parsed <= max

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
        {info.maxContextWindow ? ` Model max: ${info.maxContextWindow.toLocaleString()}.` : ''}
        {info.source === 'override' ? ' A configured override currently pins the budget.' : ''}
      </p>
      <div className="context-presets">
        {presets.map((p) => (
          <button
            key={p}
            className={Number(value) === p ? 'active' : ''}
            disabled={busy}
            onClick={() => setValue(String(p))}
          >
            {p >= 1024 ? `${p / 1024}K` : p}
          </button>
        ))}
      </div>
      <div className="context-editor-row">
        <input
          type="number"
          min={512}
          max={max}
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
      {!valid && (
        <p className="context-editor-err">Enter a value between 512 and {max.toLocaleString()}.</p>
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

// The Models slide-over: lists the models the Lemonade server knows about and
// lets the user load one as the agent's chat model. Models whose labels
// advertise tool-calling are highlighted and sorted first, since those are the
// ones that work well in the agent loop.
function Models({
  onClose,
  onChanged
}: {
  onClose: () => void
  onChanged: () => void
}): JSX.Element {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [busyId, setBusyId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  function refresh(): void {
    setLoading(true)
    window.api
      .listModels()
      .then(setModels)
      .catch(() => setModels([]))
      .finally(() => setLoading(false))
  }

  useEffect(refresh, [])

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

  // Only chat models can drive the agent; surface those first and separately.
  const llms = models.filter((m) => m.type === 'llm')
  const others = models.filter((m) => m.type !== 'llm')

  // Agent-ready models first, then by name, so the best choices float to top.
  // Omni models lead the list since they're recommended for agentic use.
  const rankedLlms = [...llms].sort((a, b) => {
    if (a.omni !== b.omni) return a.omni ? -1 : 1
    if (a.agentReady !== b.agentReady) return a.agentReady ? -1 : 1
    return a.id.localeCompare(b.id)
  })

  return (
    <div className="pantry-overlay" onClick={onClose}>
      <aside className="pantry" onClick={(e) => e.stopPropagation()}>
        <header className="pantry-head">
          <div>
            <h2><CpuChipIcon /> Models</h2>
            <p className="pantry-sub">
              Load a model on the server to run the agent · ★ = great for agentic use
            </p>
          </div>
          <button className="pantry-close" onClick={onClose} aria-label="Close" title="Close">
            ✕
          </button>
        </header>

        <div className="pantry-list">
          {loading && <div className="pantry-empty">Loading models…</div>}
          {!loading && models.length === 0 && (
            <div className="pantry-empty">
              No models reported. Is the Lemonade server running?
            </div>
          )}
          {error && <div className="card-status err">{error}</div>}

          {rankedLlms.map((m) => (
            <ModelCard key={m.id} model={m} busy={busyId === m.id} onLoad={() => void load(m.id)} />
          ))}

          {others.length > 0 && (
            <>
              <div className="pantry-section">Other models (not for chat/agent)</div>
              {others.map((m) => (
                <ModelCard
                  key={m.id}
                  model={m}
                  busy={busyId === m.id}
                  onLoad={() => void load(m.id)}
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
  onLoad
}: {
  model: ModelInfo
  busy: boolean
  onLoad: () => void
}): JSX.Element {
  const ctx = model.maxContextWindow
    ? `${(model.maxContextWindow / 1024).toFixed(0)}K ctx`
    : null
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
            model.sizeGb ? `${model.sizeGb.toFixed(2)} GB` : null,
            model.downloaded ? 'downloaded' : 'not downloaded',
            model.loaded ? '● loaded on server' : null
          ]
            .filter(Boolean)
            .join(' · ')}
        </div>
        {!model.agentReady && model.type === 'llm' && (
          <div className="card-status warn-note">
            No tool-calling label — may not reliably call tools in the agent loop.
          </div>
        )}
      </div>
      <div className="card-actions">
        <button
          className={model.active ? 'btn-off' : 'btn-on'}
          disabled={busy || model.active}
          onClick={onLoad}
        >
          {busy ? 'Loading…' : model.active ? 'Active' : model.loaded ? 'Use' : 'Load'}
        </button>
      </div>
    </div>
  )
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

function Line({ entry }: { entry: Entry }): JSX.Element {
  if (entry.kind === 'tool') {
    return (
      <div className={`line tool ${entry.ok === false ? 'tool-err' : ''}`}>
        <span className="tool-label">{entry.label}</span>
        <span className="tool-detail">{entry.detail}</span>
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
  return (
    <div className={`line ${entry.kind}`}>
      <span className="role">{entry.kind === 'user' ? 'You' : 'Agent'}</span>
      <span className="bubble">{entry.text}</span>
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
          Only stock tools you trust — their actions run with this app's privileges.
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
  return <div className="card-status">Enabled — connecting…</div>
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
