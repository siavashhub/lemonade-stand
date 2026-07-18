import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { McpManager } from '../src/main/mcp/manager'
import type { McpServerConfig } from '@shared/types'

// Opt-in integration tests for the `uvx`-based Pantry servers (time, sqlite,
// git, fetch). They are SKIPPED unless RUN_UVX_TESTS is set, because they
// require the `uv`/`uvx` toolchain and — for the first run — a package
// download (and, for fetch, network access). Run them locally with:
//   npm run test:optional
// In CI they run in a dedicated, non-blocking job (see .github/workflows/ci.yml).
const enabled = !!process.env.RUN_UVX_TESTS
const d = enabled ? describe : describe.skip

// uvx ships as a real executable on every platform, so no `.cmd` shim dance.
const UVX = 'uvx'

// Boot a single stdio server through the app's McpManager and return it.
async function connect(server: McpServerConfig): Promise<McpManager> {
  const manager = new McpManager()
  await manager.connectAll([server])
  const runtime = manager.getRuntime(server.id)
  if (!runtime?.connected) {
    throw new Error(`server "${server.id}" failed to connect: ${runtime?.error}`)
  }
  return manager
}

d('time MCP server (integration, uvx)', () => {
  let manager: McpManager
  beforeAll(async () => {
    manager = await connect({
      id: 'time',
      transport: 'stdio',
      command: UVX,
      args: ['mcp-server-time'],
      enabled: true
    })
  })
  afterAll(async () => manager?.closeAll())

  it('exposes get_current_time and returns a plausible timestamp', async () => {
    expect(manager.getTools().map((t) => t.qualifiedName)).toContain('time__get_current_time')
    const result = await manager.callTool('time__get_current_time', { timezone: 'UTC' })
    // The server returns JSON containing an ISO datetime; assert it mentions the
    // timezone and looks like a real time value rather than an error string.
    expect(result).toContain('UTC')
    expect(result).toMatch(/\d{4}-\d{2}-\d{2}/)
  })
})

d('sqlite MCP server (integration, uvx)', () => {
  let manager: McpManager
  let dir: string
  beforeAll(async () => {
    dir = mkdtempSync(join(tmpdir(), 'lemonade-sqlite-test-'))
    manager = await connect({
      id: 'sqlite',
      transport: 'stdio',
      command: UVX,
      args: ['mcp-server-sqlite', '--db-path', join(dir, 'stand.db')],
      enabled: true
    })
  })
  afterAll(async () => {
    await manager?.closeAll()
    if (dir) rmSync(dir, { recursive: true, force: true })
  })

  it('creates a table, inserts rows, and queries them back', async () => {
    await manager.callTool('sqlite__create_table', {
      query: 'CREATE TABLE sales (id INTEGER PRIMARY KEY, item TEXT, price REAL)'
    })
    await manager.callTool('sqlite__write_query', {
      query: "INSERT INTO sales (item, price) VALUES ('lemonade', 2.0), ('cookie', 1.0)"
    })
    const tables = await manager.callTool('sqlite__list_tables', {})
    expect(tables).toContain('sales')

    const rows = await manager.callTool('sqlite__read_query', {
      query: 'SELECT item, price FROM sales ORDER BY price DESC'
    })
    expect(rows).toContain('lemonade')
    expect(rows).toContain('cookie')
  })
})

d('git MCP server (integration, uvx)', () => {
  let manager: McpManager
  let repo: string
  beforeAll(async () => {
    repo = mkdtempSync(join(tmpdir(), 'lemonade-git-test-'))
    // Minimal repo so git_status has something real to report.
    execFileSync('git', ['init'], { cwd: repo })
    writeFileSync(join(repo, 'note.txt'), 'hello')
    manager = await connect({
      id: 'git',
      transport: 'stdio',
      command: UVX,
      args: ['mcp-server-git', '--repository', repo],
      enabled: true
    })
  })
  afterAll(async () => {
    await manager?.closeAll()
    if (repo) rmSync(repo, { recursive: true, force: true })
  })

  it('reports the untracked file via git_status', async () => {
    expect(manager.getTools().map((t) => t.qualifiedName)).toContain('git__git_status')
    const status = await manager.callTool('git__git_status', { repo_path: repo })
    expect(status).toContain('note.txt')
  })
})

d('fetch MCP server (integration, uvx, network)', () => {
  let manager: McpManager
  beforeAll(async () => {
    manager = await connect({
      id: 'fetch',
      transport: 'stdio',
      command: UVX,
      args: ['mcp-server-fetch'],
      enabled: true
    })
  })
  afterAll(async () => manager?.closeAll())

  it('fetches a URL and returns page content as markdown', async () => {
    expect(manager.getTools().map((t) => t.qualifiedName)).toContain('fetch__fetch')
    const result = await manager.callTool('fetch__fetch', { url: 'https://example.com' })
    // The server prefixes the response with the URL, so example.com is always
    // present. iana.org is the stable IANA link in the page body, regardless of
    // how mcp-server-fetch renders the page heading.
    expect(result.toLowerCase()).toContain('example.com')
    expect(result.toLowerCase()).toContain('iana.org')
  })
})
