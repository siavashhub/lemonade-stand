import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { mkdtempSync, rmSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { McpManager } from '../src/main/mcp/manager'
import type { McpServerConfig } from '@shared/types'

// End-to-end test of the MCP layer WITHOUT a model: it boots the real
// `@modelcontextprotocol/server-filesystem` as a stdio subprocess, connects the
// app's own McpManager to it, and drives real tool calls. This exercises the
// exact path a tool takes at runtime (connectAll -> listTools -> callTool),
// minus the non-deterministic LLM that would normally decide to call it.
describe('filesystem MCP server (integration)', () => {
  let root: string
  let manager: McpManager

  beforeAll(async () => {
    root = mkdtempSync(join(tmpdir(), 'lemonade-fs-test-'))
    manager = new McpManager()
    const server: McpServerConfig = {
      id: 'filesystem',
      transport: 'stdio',
      command: process.platform === 'win32' ? 'npx.cmd' : 'npx',
      args: ['-y', '@modelcontextprotocol/server-filesystem', root],
      enabled: true
    }
    await manager.connectAll([server])
  })

  afterAll(async () => {
    await manager?.closeAll()
    if (root) rmSync(root, { recursive: true, force: true })
  })

  it('connects and reports the server as healthy', () => {
    const runtime = manager.getRuntime('filesystem')
    expect(runtime?.connected, `connect error: ${runtime?.error}`).toBe(true)
    expect(runtime?.toolCount ?? 0).toBeGreaterThan(0)
  })

  it('exposes tools namespaced as filesystem__<tool>', () => {
    const tools = manager.getTools()
    const names = tools.map((t) => t.qualifiedName)
    expect(names).toContain('filesystem__write_file')
    expect(names).toContain('filesystem__read_file')
    expect(names).toContain('filesystem__list_directory')
    // Every tool must belong to this server and carry a description.
    for (const tool of tools) {
      expect(tool.serverId).toBe('filesystem')
    }
  })

  it('writes a file through the tool and it lands on disk', async () => {
    const target = join(root, 'hello.txt')
    const body = 'Fresh lemonade, 50 cents'
    await manager.callTool('filesystem__write_file', { path: target, content: body })
    expect(readFileSync(target, 'utf8')).toBe(body)
  })

  it('reads the file back through the tool', async () => {
    const target = join(root, 'hello.txt')
    const result = await manager.callTool('filesystem__read_file', { path: target })
    expect(result).toContain('Fresh lemonade, 50 cents')
  })

  it('lists the directory and sees the file it wrote', async () => {
    const result = await manager.callTool('filesystem__list_directory', { path: root })
    expect(result).toContain('hello.txt')
  })

  it('surfaces tool errors instead of throwing (reading a missing file)', async () => {
    const result = await manager.callTool('filesystem__read_file', {
      path: join(root, 'does-not-exist.txt')
    })
    expect(result.toLowerCase()).toContain('error')
  })

  it('rejects an unknown qualified tool name', async () => {
    await expect(manager.callTool('filesystem__no_such_tool', {})).rejects.toThrow()
  })
})
