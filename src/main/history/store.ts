import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import type { SessionSummary, StoredSession } from '@shared/types'

// Saved conversations live one-file-per-session under `conversations/` inside
// the app's writable config dir (the same per-user location servers.json and
// settings.json are seeded into). One file per session keeps a long history from
// forcing a full rewrite on every save, and isolates a single corrupt file so it
// can't take down the whole list.
const DIR = 'conversations'

function historyDir(cwd: string): string {
  return resolve(cwd, DIR)
}

// Session ids are app-generated UUIDs, but sanitize anyway so a malformed id can
// never escape the conversations directory via `..` or path separators.
function sanitizeId(id: string): string {
  return id.replace(/[^a-zA-Z0-9_-]/g, '')
}

function sessionPath(cwd: string, id: string): string {
  return join(historyDir(cwd), `${sanitizeId(id)}.json`)
}

/** Metadata-only view of a stored session, for the sidebar list. */
function toSummary(s: StoredSession): SessionSummary {
  return {
    id: s.id,
    title: s.title,
    createdAt: s.createdAt,
    updatedAt: s.updatedAt,
    messageCount: s.messageCount
  }
}

/** List saved conversations, newest first. Unreadable files are skipped rather
 * than aborting the whole listing. */
export function listSessions(cwd: string): SessionSummary[] {
  const dir = historyDir(cwd)
  if (!existsSync(dir)) return []
  const summaries: SessionSummary[] = []
  for (const name of readdirSync(dir)) {
    if (!name.endsWith('.json')) continue
    try {
      const raw = readFileSync(join(dir, name), 'utf8')
      const parsed = JSON.parse(raw) as StoredSession
      if (parsed && typeof parsed.id === 'string') summaries.push(toSummary(parsed))
    } catch {
      // Skip a corrupt or partially written file.
    }
  }
  return summaries.sort((a, b) => b.updatedAt - a.updatedAt)
}

/** Read one full saved conversation, or null when it doesn't exist / is corrupt. */
export function readSession(cwd: string, id: string): StoredSession | null {
  try {
    const raw = readFileSync(sessionPath(cwd, id), 'utf8')
    return JSON.parse(raw) as StoredSession
  } catch {
    return null
  }
}

/** Persist a conversation, normalizing its metadata (timestamps, message count)
 * before writing. Returns the refreshed session list. */
export function writeSession(cwd: string, session: StoredSession): SessionSummary[] {
  mkdirSync(historyDir(cwd), { recursive: true })
  const now = Date.now()
  const normalized: StoredSession = {
    ...session,
    createdAt: session.createdAt || now,
    updatedAt: now,
    messageCount: session.history.length
  }
  writeFileSync(
    sessionPath(cwd, session.id),
    JSON.stringify(normalized, null, 2) + '\n',
    'utf8'
  )
  return listSessions(cwd)
}

/** Delete a saved conversation. Missing file is a no-op. Returns the refreshed
 * session list. */
export function deleteSession(cwd: string, id: string): SessionSummary[] {
  const path = sessionPath(cwd, id)
  try {
    if (existsSync(path)) rmSync(path)
  } catch {
    // Ignore; the listing below will still reflect reality.
  }
  return listSessions(cwd)
}

/** Rename a saved conversation in place. Returns the refreshed session list. */
export function renameSession(cwd: string, id: string, title: string): SessionSummary[] {
  const session = readSession(cwd, id)
  if (!session) return listSessions(cwd)
  session.title = title.trim().slice(0, 80) || session.title
  return writeSession(cwd, session)
}

/** Delete every saved conversation. Only removes this app's `*.json` session
 * files, so nothing else that might live in the directory is touched. Returns
 * the (now empty) session list. */
export function clearSessions(cwd: string): SessionSummary[] {
  const dir = historyDir(cwd)
  if (!existsSync(dir)) return []
  for (const name of readdirSync(dir)) {
    if (!name.endsWith('.json')) continue
    try {
      rmSync(join(dir, name))
    } catch {
      // Ignore a file we couldn't remove; the listing reflects what remains.
    }
  }
  return listSessions(cwd)
}
