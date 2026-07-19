import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import type { CatalogEntry, McpServerConfig } from '@shared/types'

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

  let servers: McpServerConfig[] = []
  try {
    const raw = readFileSync(resolve(cwd, 'config/servers.json'), 'utf8')
    const parsed = JSON.parse(raw) as { servers?: McpServerConfig[] }
    servers = (parsed.servers ?? []).filter((s) => s.enabled)
  } catch {
    // No server config -> agent runs chat-only with no external tools.
  }

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
    maxSteps: Number(process.env.AGENT_MAX_STEPS ?? '8'),
    contextSize: process.env.LEMONADE_CONTEXT_SIZE
      ? Number(process.env.LEMONADE_CONTEXT_SIZE)
      : undefined,
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
  'Skip planning for simple, one-step requests.'

const SERVERS_FILE = 'config/servers.json'
const CATALOG_FILE = 'config/catalog.json'
const PHRASES_FILE = 'config/phrases.json'
const SETTINGS_FILE = 'config/settings.json'

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
}

/** Read persisted UI/app settings. Missing or malformed file -> empty. */
export function readSettings(cwd: string): AppSettings {
  try {
    const raw = readFileSync(resolve(cwd, SETTINGS_FILE), 'utf8')
    return JSON.parse(raw) as AppSettings
  } catch {
    return {}
  }
}

/** Persist app settings, preserving any sibling keys already on disk. */
export function writeSettings(cwd: string, settings: AppSettings): void {
  const path = resolve(cwd, SETTINGS_FILE)
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

/** Read every configured server (enabled and disabled) from disk. */
export function readServers(cwd: string): McpServerConfig[] {
  try {
    const raw = readFileSync(resolve(cwd, SERVERS_FILE), 'utf8')
    const parsed = JSON.parse(raw) as { servers?: McpServerConfig[] }
    return parsed.servers ?? []
  } catch {
    return []
  }
}

/** Persist the server list, preserving any sibling keys (e.g. the schema-note). */
export function writeServers(cwd: string, servers: McpServerConfig[]): void {
  const path = resolve(cwd, SERVERS_FILE)
  let existing: Record<string, unknown> = {}
  try {
    existing = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>
  } catch {
    // Fresh file is fine.
  }
  writeFileSync(path, JSON.stringify({ ...existing, servers }, null, 2) + '\n', 'utf8')
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
