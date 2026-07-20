import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import type { CatalogEntry, McpServerConfig, Pitcher } from '@shared/types'

// Minimal .env loader, avoids a dependency. Parses KEY=VALUE lines, ignores
// comments/blank lines, and does not override variables already in the real
// environment (so OS-level env wins, matching Lemonade's own key handling).
function loadDotEnv(cwd: string): void {
  try {
    const raw = readFileSync(resolve(cwd, '.env'), 'utf8')
    for (const line of raw.split(/\r?\n/)) {
      const trimmed = line.trim()
      if (!trimmed || trimmed.startsWith('#')) continue
      const eq = trimmed.indexOf('=')
      if (eq === -1) continue
      const key = trimmed.slice(0, eq).trim()
      const value = trimmed.slice(eq + 1).trim()
      if (key && process.env[key] === undefined) process.env[key] = value
    }
  } catch {
    // No .env file is fine; fall back to real env + defaults.
  }
}

// Clamp a possibly-bad numeric env value into the 0..1 range, falling back to a
// default when it isn't a finite number.
function clampFraction(value: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback
  return Math.min(1, Math.max(0, value))
}

export interface AppConfig {
  lemonadeBaseUrl: string
  lemonadeApiKey: string
  model: string
  maxSteps: number
  /**
   * Effective context window (tokens) to budget against. When unset, the app
   * asks the server for the model's `max_context_window`. Set this to match the
   * context the server was actually launched with when that runtime limit is
   * smaller than the model's advertised maximum.
   */
  contextSize?: number
  /** Tokens reserved for the model's reply when checking the prompt budget. */
  completionReserve: number
  /** System prompt that primes the model to actually call tools. */
  systemPrompt: string
  /** Prompt the user before every MCP tool call (default true). */
  requireApproval: boolean
  /** Fraction (0-1) of the usable context budget at which older messages are
   * auto-summarized to reclaim room. 0 disables auto-compaction. */
  compactThreshold: number
  tts: {
    enabled: boolean
    model: string
    voice: string
    format: string
  }
  /** Speech-to-text (microphone dictation). `model` is the preferred
   * transcription model; when it isn't served, the first available one is used. */
  stt: {
    model: string
  }
  /** Verbose diagnostic logging, enabled with LOG_LEVEL=debug. */
  debug: boolean
  servers: McpServerConfig[]
}

export function loadConfig(cwd: string = process.cwd()): AppConfig {
  loadDotEnv(cwd)

  // Base defaults live in the committed config/servers.json; a developer's
  // personal edits layer over them from the gitignored config/servers.local.json
  // (see readServers). Only enabled servers are launched.
  const servers = readServers(cwd).filter((s) => s.enabled)

  // Values the user sets in the app's UI (model, server connection) are
  // remembered across restarts in settings.json. A UI choice is explicit, so it
  // takes priority over the matching env var, which acts only as the *initial*
  // default before the user has ever chosen one. Precedence for each:
  // saved choice, then env default, then the built-in default.
  const saved = readSettings(cwd)

  return {
    // 13305 is Lemonade Server's default port (its OpenAI-compatible routes live
    // under /api/v1). Configurable in-app via the server-status menu, or by env.
    lemonadeBaseUrl:
      saved.baseUrl ?? process.env.LEMONADE_BASE_URL ?? 'http://localhost:13305/api/v1',
    lemonadeApiKey: saved.apiKey ?? process.env.LEMONADE_API_KEY ?? '',
    model: saved.model ?? process.env.LEMONADE_MODEL ?? 'Qwen3-1.7B-GGUF',
    // A value saved in settings.json wins so packaged users can tune it without
    // touching env; the env var is only the initial default before that.
    maxSteps: saved.maxSteps ?? Number(process.env.AGENT_MAX_STEPS ?? '20'),
    // The user's last in-app choice wins and survives restarts; the env var is
    // only the initial default before they've ever changed it in the UI.
    contextSize:
      saved.contextSize ??
      (process.env.LEMONADE_CONTEXT_SIZE ? Number(process.env.LEMONADE_CONTEXT_SIZE) : undefined),
    completionReserve: Number(process.env.LEMONADE_COMPLETION_RESERVE ?? '512'),
    systemPrompt: process.env.AGENT_SYSTEM_PROMPT ?? DEFAULT_SYSTEM_PROMPT,
    requireApproval: (process.env.AGENT_REQUIRE_APPROVAL ?? 'true') !== 'false',
    compactThreshold: clampFraction(
      Number(process.env.AGENT_COMPACT_THRESHOLD ?? '0.9'),
      0.9
    ),
    tts: {
      enabled: saved.speak ?? (process.env.LEMONADE_TTS_ENABLED ?? 'false') === 'true',
      model: process.env.LEMONADE_TTS_MODEL ?? 'kokoro-v1',
      voice: process.env.LEMONADE_TTS_VOICE ?? 'af_sky',
      format: process.env.LEMONADE_TTS_FORMAT ?? 'mp3'
    },
    stt: {
      model: process.env.LEMONADE_STT_MODEL ?? 'whisper-base'
    },
    debug:
      (saved.logLevel ?? process.env.LOG_LEVEL ?? 'info').toLowerCase() === 'debug',
    servers
  }
}

// Primes the model to use its tools instead of narrating what the user should
// do. Weak local models default to chatty behavior without this nudge.
const DEFAULT_SYSTEM_PROMPT =
  'You are Lemonade Stand, a local AI agent with tools provided by MCP servers. ' +
  'When the user asks for something a tool can do \u2014 create or edit a file, fetch a URL, ' +
  'query a database, run a command \u2014 you MUST call the appropriate tool to actually do it. ' +
  'Never print code, file contents, or step-by-step instructions for the user to run themselves ' +
  'when a tool can perform the action directly. Only reply in plain text when no tool fits, or to ' +
  'briefly report results after your tool calls have completed. ' +
  'For a task that takes several steps or tool calls, first call the update_plan tool to lay out ' +
  'a short todo list, then call it again as you go to mark steps in-progress and completed. ' +
  'Skip planning for simple, one-step requests. ' +
  'To show an image, a diagram, or other rich/copyable artifact, call the show_napkin tool so it ' +
  'renders in the side panel. Never paste base64 data or a data: URL into your chat reply , the ' +
  'chat cannot render it. When a tool returns an image, it is already shown to the user on the ' +
  'napkin panel; just acknowledge it briefly instead of trying to reproduce the bytes.'

const SERVERS_FILE = 'config/servers.json'
// Optional, gitignored per-developer override merged over SERVERS_FILE. Keeps
// local server tweaks out of the tracked defaults (and out of `git status`).
const SERVERS_LOCAL_FILE = 'config/servers.local.json'
const CATALOG_FILE = 'config/catalog.json'
const PHRASES_FILE = 'config/phrases.json'
const SETTINGS_FILE = 'config/settings.json'
// A developer's personal UI state (model, connection, context size, …) layers
// over the committed settings.json from this gitignored file, so a dev checkout
// never dirties the tracked default with machine-specific choices , exactly the
// way servers.local.json overrides servers.json.
const SETTINGS_LOCAL_FILE = 'config/settings.local.json'
const PITCHERS_FILE = 'config/pitchers.json'
// Optional, gitignored per-developer override merged over PITCHERS_FILE, keyed
// by pitcher id, so local scheduled-task edits stay out of the tracked defaults
// (and out of `git status`), exactly like servers.local.json.
const PITCHERS_LOCAL_FILE = 'config/pitchers.local.json'

/** User-chosen state that must survive restarts (e.g. the active chat model). */
export interface AppSettings {
  /** The model the user last loaded as the agent's chat model, if any. */
  model?: string
  /** Base URL of the Lemonade server the user configured in-app, if any.
   * Includes the OpenAI-compatible prefix, e.g. `http://localhost:13305/api/v1`. */
  baseUrl?: string
  /** API key the user configured in-app, if the server requires one. */
  apiKey?: string
  /** Whether spoken replies (TTS) were last left on or off by the user. */
  speak?: boolean
  /** Diagnostic log level. Set to "debug" to mirror all logs to a file
   * (config/settings.json). Takes precedence over the LOG_LEVEL env var. */
  logLevel?: string
  /** Runtime context window (tokens) the user last picked in the UI. Restored
   * on the next launch so the choice survives restarts. */
  contextSize?: number
  /** Max tool-calling iterations per user turn before the loop stops. Lets a
   * packaged user tune the safety rail from settings.json without env vars. */
  maxSteps?: number
}

/** Read one settings file's JSON object; null when the file is absent/malformed. */
function readSettingsFile(cwd: string, file: string): AppSettings | null {
  try {
    const raw = readFileSync(resolve(cwd, file), 'utf8')
    return JSON.parse(raw) as AppSettings
  } catch {
    return null
  }
}

/** Read persisted UI/app settings, with the gitignored local override
 * (config/settings.local.json) layered over the committed defaults. A
 * developer's personal choices live only in the local file, so the tracked
 * config/settings.json never picks up machine-specific churn. */
export function readSettings(cwd: string): AppSettings {
  const base = readSettingsFile(cwd, SETTINGS_FILE) ?? {}
  const local = readSettingsFile(cwd, SETTINGS_LOCAL_FILE)
  return local ? { ...base, ...local } : base
}

/** Persist app settings, preserving any sibling keys already on disk. Writes to
 * the gitignored local override when it exists (a dev checkout), so UI edits
 * don't dirty the tracked config/settings.json; otherwise writes the base file
 * (e.g. the seeded per-user copy in a packaged build). */
export function writeSettings(cwd: string, settings: AppSettings): void {
  const target = existsSync(resolve(cwd, SETTINGS_LOCAL_FILE))
    ? SETTINGS_LOCAL_FILE
    : SETTINGS_FILE
  const path = resolve(cwd, target)
  let existing: Record<string, unknown> = {}
  try {
    existing = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>
  } catch {
    // Fresh file is fine.
  }
  writeFileSync(path, JSON.stringify({ ...existing, ...settings }, null, 2) + '\n', 'utf8')
}

// Fallback used when config/phrases.json is missing or malformed, so the
// "agent is working" indicator always has something to say.
const DEFAULT_PHRASES = ['Squeezing the lemons', 'Stirring the lemonade']

/** Playful "agent is working" phrases shown while a turn is in flight. Reads
 * the VS Code-style `chat.agent.thinking.phrases` block from phrases.json. */
export function loadThinkingPhrases(cwd: string): string[] {
  try {
    const raw = readFileSync(resolve(cwd, PHRASES_FILE), 'utf8')
    const parsed = JSON.parse(raw) as {
      'chat.agent.thinking.phrases'?: { phrases?: string[] }
    }
    const phrases = parsed['chat.agent.thinking.phrases']?.phrases
    return Array.isArray(phrases) && phrases.length > 0 ? phrases : DEFAULT_PHRASES
  } catch {
    return DEFAULT_PHRASES
  }
}

/** Read one server-config file's `servers` array; null when the file is absent. */
function readServersFile(cwd: string, file: string): McpServerConfig[] | null {
  try {
    const raw = readFileSync(resolve(cwd, file), 'utf8')
    const parsed = JSON.parse(raw) as { servers?: McpServerConfig[] }
    return parsed.servers ?? []
  } catch {
    return null
  }
}

/** Layer a local override list over the base list, keyed by server id: a local
 * entry replaces the same-id base entry in place; brand-new ids are appended. */
function mergeServers(
  base: McpServerConfig[],
  local: McpServerConfig[]
): McpServerConfig[] {
  const byId = new Map(base.map((s) => [s.id, s]))
  const order = base.map((s) => s.id)
  for (const s of local) {
    if (!byId.has(s.id)) order.push(s.id)
    byId.set(s.id, s)
  }
  return order.map((id) => byId.get(id) as McpServerConfig)
}

/** Read every configured server (enabled and disabled), with the gitignored
 * local override (config/servers.local.json) merged over the committed defaults.
 * A developer's personal server edits live only in the local file, so the tracked
 * config/servers.json never picks up local churn. */
export function readServers(cwd: string): McpServerConfig[] {
  const base = readServersFile(cwd, SERVERS_FILE) ?? []
  const local = readServersFile(cwd, SERVERS_LOCAL_FILE)
  return local ? mergeServers(base, local) : base
}

/** Persist the server list, preserving any sibling keys (e.g. the schema-note).
 * Writes to the gitignored local override when it exists, so UI edits in a dev
 * checkout don't dirty the tracked config/servers.json. */
export function writeServers(cwd: string, servers: McpServerConfig[]): void {
  const target = existsSync(resolve(cwd, SERVERS_LOCAL_FILE))
    ? SERVERS_LOCAL_FILE
    : SERVERS_FILE
  const path = resolve(cwd, target)
  let existing: Record<string, unknown> = {}
  try {
    existing = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>
  } catch {
    // Fresh file is fine.
  }
  writeFileSync(path, JSON.stringify({ ...existing, servers }, null, 2) + '\n', 'utf8')
}

/** Read one pitcher-config file's `pitchers` array; null when the file is absent. */
function readPitchersFile(cwd: string, file: string): Pitcher[] | null {
  try {
    const raw = readFileSync(resolve(cwd, file), 'utf8')
    const parsed = JSON.parse(raw) as { pitchers?: Pitcher[] }
    return parsed.pitchers ?? []
  } catch {
    return null
  }
}

/** Layer a local override list over the base list, keyed by pitcher id: a local
 * entry replaces the same-id base entry in place; brand-new ids are appended. */
function mergePitchers(base: Pitcher[], local: Pitcher[]): Pitcher[] {
  const byId = new Map(base.map((p) => [p.id, p]))
  const order = base.map((p) => p.id)
  for (const p of local) {
    if (!byId.has(p.id)) order.push(p.id)
    byId.set(p.id, p)
  }
  return order.map((id) => byId.get(id) as Pitcher)
}

/** Read every configured Pitcher (scheduled task), with the gitignored local
 * override (config/pitchers.local.json) merged over the committed defaults. A
 * developer's personal pitcher edits live only in the local file, so the tracked
 * config/pitchers.json never picks up local churn. */
export function readPitchers(cwd: string): Pitcher[] {
  const base = readPitchersFile(cwd, PITCHERS_FILE) ?? []
  const local = readPitchersFile(cwd, PITCHERS_LOCAL_FILE)
  return local ? mergePitchers(base, local) : base
}

/** Persist the Pitcher list, preserving any sibling keys (e.g. the note). Writes
 * to the gitignored local override when it exists, so UI edits in a dev checkout
 * don't dirty the tracked config/pitchers.json. */
export function writePitchers(cwd: string, pitchers: Pitcher[]): void {
  const target = existsSync(resolve(cwd, PITCHERS_LOCAL_FILE))
    ? PITCHERS_LOCAL_FILE
    : PITCHERS_FILE
  const path = resolve(cwd, target)
  let existing: Record<string, unknown> = {}
  try {
    existing = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>
  } catch {
    // Fresh file is fine.
  }
  writeFileSync(path, JSON.stringify({ ...existing, pitchers }, null, 2) + '\n', 'utf8')
}

// Seed content for the gitignored local override files. Each starts empty so it
// contributes no overrides (the committed defaults still show), but its mere
// existence redirects the app's writes here. The `$schema-note` documents the
// file for a developer who spots it after a fresh clone.
const LOCAL_OVERRIDE_SEEDS: Record<string, Record<string, unknown>> = {
  [SERVERS_LOCAL_FILE]: {
    '$schema-note':
      'LOCAL, gitignored per-developer override. Entries here are merged over config/servers.json by server id (same id replaces the committed default; new ids are appended). The app writes your personal server edits here so the tracked config/servers.json stays clean.',
    servers: []
  },
  [SETTINGS_LOCAL_FILE]: {
    '$schema-note':
      'LOCAL, gitignored per-developer override merged over config/settings.json. The app writes your personal UI state (model, connection, context size, …) here so the tracked config/settings.json stays clean.'
  },
  [PITCHERS_LOCAL_FILE]: {
    '$schema-note':
      'LOCAL, gitignored per-developer override. Entries here are merged over config/pitchers.json by pitcher id (same id replaces the committed default; new ids are appended). The app writes your personal scheduled-task edits here so the tracked config/pitchers.json stays clean.',
    pitchers: []
  }
}

/** Seed the gitignored local override files (servers/settings/pitchers) if they
 * don't exist yet. Call this only in a dev checkout (`!app.isPackaged`): it makes
 * the app write UI edits to the `*.local.json` overrides instead of dirtying the
 * committed defaults, and — because the seeded files are visible and self-
 * documenting — a developer cloning the repo discovers the mechanism on first
 * run. Packaged builds skip this: they seed a writable per-user copy elsewhere. */
export function seedLocalOverrides(cwd: string): void {
  for (const [file, seed] of Object.entries(LOCAL_OVERRIDE_SEEDS)) {
    const path = resolve(cwd, file)
    if (existsSync(path)) continue
    try {
      writeFileSync(path, JSON.stringify(seed, null, 2) + '\n', 'utf8')
    } catch {
      // A failed seed is non-fatal: the app falls back to writing the tracked
      // default, exactly as before this helper existed.
    }
  }
}

/** The Market catalogue of installable tools/skills. */
export function loadCatalog(cwd: string): CatalogEntry[] {
  try {
    const raw = readFileSync(resolve(cwd, CATALOG_FILE), 'utf8')
    const parsed = JSON.parse(raw) as { entries?: CatalogEntry[] }
    return parsed.entries ?? []
  } catch {
    return []
  }
}

/** Build a concrete server config from a catalogue entry, substituting the
 * chosen filesystem path into any `{{path}}` placeholders. */
export function serverFromCatalog(entry: CatalogEntry, path?: string): McpServerConfig {
  if (entry.transport === 'http') {
    return {
      id: entry.id,
      transport: 'http',
      url: entry.url ?? '',
      headers: entry.headers,
      enabled: true
    }
  }
  const fill = (s: string): string => (path ? s.replace('{{path}}', path) : s)
  return {
    id: entry.id,
    transport: 'stdio',
    command: entry.command ?? '',
    args: (entry.args ?? []).map(fill),
    env: entry.env,
    enabled: true
  }
}

const PATH_PLACEHOLDER = '{{path}}'

/** Index of the catalog arg that carries the `{{path}}` placeholder, or -1. */
function pathArgIndex(entry: CatalogEntry): number {
  return (entry.args ?? []).findIndex((a) => a.includes(PATH_PLACEHOLDER))
}

/** Recover the filesystem path a configured path-based server is using, by
 * matching the catalog template's `{{path}}` slot against the stored args. */
export function pathForServer(entry: CatalogEntry, server: McpServerConfig): string | undefined {
  if (server.transport !== 'stdio' || entry.transport !== 'stdio') return undefined
  const idx = pathArgIndex(entry)
  if (idx < 0) return undefined
  return server.args?.[idx]
}

/** Return a copy of `server` with its `{{path}}` arg replaced by `path`. */
export function withServerPath(
  entry: CatalogEntry,
  server: McpServerConfig,
  path: string
): McpServerConfig {
  if (server.transport !== 'stdio') return server
  const idx = pathArgIndex(entry)
  const templateArgs = entry.args ?? []
  const args = (server.args ?? []).slice()
  if (idx >= 0) {
    args[idx] = templateArgs[idx].replace(PATH_PLACEHOLDER, path)
  }
  return { ...server, args }
}
