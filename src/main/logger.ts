import { appendFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'

// Lightweight file logging for diagnostics. When enabled (LOG_LEVEL=debug or
// `"logLevel": "debug"` in config/settings.json), every console.log/info/warn/
// error is mirrored to a rolling text file the user can share when reporting a
// problem. Disabled by default so normal runs write nothing to disk.

let logFilePath: string | null = null

/**
 * Turn on file logging in `dir`, patching the console so existing log calls are
 * also written to `lemonade-stand.log`. No-op when `enabled` is false. Returns
 * the log file path (or null when disabled / not writable).
 */
export function initFileLogging(dir: string, enabled: boolean, header?: string): string | null {
  if (!enabled) return null
  try {
    mkdirSync(dir, { recursive: true })
    const file = join(dir, 'lemonade-stand.log')
    appendFileSync(
      file,
      `\n===== session started ${new Date().toISOString()}${header ? ` ${header}` : ''} =====\n`
    )
    logFilePath = file
  } catch (err) {
    // If the log location isn't writable, keep running without file logging.
    console.error('[logger] could not open log file:', err)
    return null
  }

  const levels = ['log', 'info', 'warn', 'error'] as const
  for (const level of levels) {
    const original = console[level].bind(console)
    console[level] = (...args: unknown[]): void => {
      original(...args)
      appendLine(level, args)
    }
  }
  return logFilePath
}

/** The active log file path, or null when file logging is off. */
export function getLogFilePath(): string | null {
  return logFilePath
}

function appendLine(level: string, args: unknown[]): void {
  if (!logFilePath) return
  const body = args
    .map((a) => {
      if (typeof a === 'string') return a
      if (a instanceof Error) return a.stack ?? a.message
      try {
        return JSON.stringify(a)
      } catch {
        return String(a)
      }
    })
    .join(' ')
  try {
    appendFileSync(logFilePath, `[${new Date().toISOString()}] [${level}] ${body}\n`)
  } catch {
    // Best-effort: never let logging throw into the app.
  }
}
