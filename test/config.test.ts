import { describe, it, expect } from 'vitest'
import { resolve } from 'node:path'
import {
  loadCatalog,
  readServers,
  serverFromCatalog,
  pathForServer,
  withServerPath
} from '../src/main/config'
import type { CatalogEntry } from '@shared/types'

// The repo root — config files (catalog.json / servers.json) live under it.
const ROOT = resolve(__dirname, '..')

describe('catalog.json', () => {
  const catalog = loadCatalog(ROOT)

  it('is non-empty and loads', () => {
    expect(catalog.length).toBeGreaterThan(0)
  })

  it('every entry has the required fields', () => {
    for (const entry of catalog) {
      expect(entry.id, `entry missing id: ${JSON.stringify(entry)}`).toBeTruthy()
      expect(entry.name, `${entry.id} missing name`).toBeTruthy()
      expect(entry.blurb, `${entry.id} missing blurb`).toBeTruthy()
      expect(entry.category, `${entry.id} missing category`).toBeTruthy()
      expect(['stdio', 'http'], `${entry.id} bad transport`).toContain(entry.transport)
    }
  })

  it('has unique ids', () => {
    const ids = catalog.map((e) => e.id)
    expect(new Set(ids).size).toBe(ids.length)
  })

  it('stdio entries declare a command; http entries declare a url', () => {
    for (const entry of catalog) {
      if (entry.transport === 'stdio') {
        expect(entry.command, `${entry.id} stdio needs command`).toBeTruthy()
      } else {
        expect(entry.url, `${entry.id} http needs url`).toBeTruthy()
      }
    }
  })

  it('path-based entries carry a {{path}} placeholder in their args', () => {
    for (const entry of catalog) {
      if (!entry.needsPath) continue
      const hasPlaceholder = (entry.args ?? []).some((a) => a.includes('{{path}}'))
      expect(hasPlaceholder, `${entry.id} needsPath but no {{path}} arg`).toBe(true)
      expect(['folder', 'file'], `${entry.id} bad pathKind`).toContain(entry.pathKind)
    }
  })
})

describe('servers.json', () => {
  it('parses and every enabled server is well-formed', () => {
    // readServers returns all configured servers (enabled or not).
    const servers = readServers(ROOT)
    expect(Array.isArray(servers)).toBe(true)
    for (const server of servers) {
      expect(server.id).toBeTruthy()
      expect(['stdio', 'http']).toContain(server.transport)
      if (server.transport === 'stdio') {
        expect(server.command).toBeTruthy()
      } else {
        expect(server.url).toBeTruthy()
      }
    }
  })
})

describe('serverFromCatalog', () => {
  it('substitutes a chosen path into a {{path}} arg (stdio)', () => {
    const entry: CatalogEntry = {
      id: 'filesystem',
      name: 'Filesystem',
      blurb: 'x',
      category: 'Files',
      transport: 'stdio',
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-filesystem', '{{path}}'],
      needsPath: true,
      pathKind: 'folder'
    }
    const server = serverFromCatalog(entry, '/tmp/work')
    expect(server.transport).toBe('stdio')
    if (server.transport === 'stdio') {
      expect(server.args).toEqual(['-y', '@modelcontextprotocol/server-filesystem', '/tmp/work'])
      expect(server.command).toBe('npx')
    }
    expect(server.enabled).toBe(true)
  })

  it('produces an http server from an http entry', () => {
    const entry: CatalogEntry = {
      id: 'lemonade',
      name: 'Lemonade Gateway',
      blurb: 'x',
      category: 'Official',
      transport: 'http',
      url: 'http://localhost:13305/mcp'
    }
    const server = serverFromCatalog(entry)
    expect(server.transport).toBe('http')
    if (server.transport === 'http') {
      expect(server.url).toBe('http://localhost:13305/mcp')
    }
  })
})

describe('pathForServer / withServerPath round-trip', () => {
  const entry: CatalogEntry = {
    id: 'filesystem',
    name: 'Filesystem',
    blurb: 'x',
    category: 'Files',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', '@modelcontextprotocol/server-filesystem', '{{path}}'],
    needsPath: true,
    pathKind: 'folder'
  }

  it('recovers the path a configured server is using', () => {
    const server = serverFromCatalog(entry, '/tmp/first')
    expect(pathForServer(entry, server)).toBe('/tmp/first')
  })

  it('rewrites the path and is recoverable again', () => {
    const server = serverFromCatalog(entry, '/tmp/first')
    const moved = withServerPath(entry, server, '/tmp/second')
    expect(pathForServer(entry, moved)).toBe('/tmp/second')
  })
})
