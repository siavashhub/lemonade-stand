import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import { existsSync, readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { homedir } from 'node:os'
import { delimiter, dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { ChatCompletionTool } from 'openai/resources/chat/completions'
import type { AgentTool, McpServerConfig } from '@shared/types'

const SEP = '__'

/** An image block produced by an MCP tool, base64-encoded (no `data:` prefix). */
export interface McpToolImage {
  data: string
  mimeType: string
}

/** The outcome of an MCP tool call: text for the model plus any image blocks
 * the tool emitted, kept separate so binary bytes don't flood the context. */
export interface McpToolResult {
  text: string
  images: McpToolImage[]
}

const nodeRequire = createRequire(import.meta.url)
const HERE = fileURLToPath(new URL('.', import.meta.url))

// `npx`-style launchers we transparently reroute through Electron's own Node so
// end users need not install Node/npm themselves.
const NODE_LAUNCHERS = new Set(['npx', 'npx.cmd', 'npx.exe'])

// OpenAI tool names must match ^[a-zA-Z0-9_-]{1,64}$; sanitize any exotic
// server/tool identifiers so the model can address them.
function sanitize(name: string): string {
  return name.replace(/[^a-zA-Z0-9_-]/g, '_')
}

// Drop `undefined` values from a process env so it satisfies the stdio
// transport's Record<string, string> shape while preserving PATH and friends.
function cleanEnv(env: NodeJS.ProcessEnv): Record<string, string> {
  const out: Record<string, string> = {}
  for (const [k, v] of Object.entries(env)) {
    if (typeof v === 'string') out[k] = v
  }
  return out
}

// Directories prepended to a stdio server's PATH so its launcher (uvx/uv) and
// sibling tools resolve even when the app's own PATH is stale (e.g. uv was
// installed after the app started) or the user never installed uv at all , in
// packaged builds we ship uv under resources/bin (-> <resources>/bin).
let cachedBinDirs: string[] | null = null
function bundledBinDirs(): string[] {
  if (cachedBinDirs) return cachedBinDirs
  const candidates = [
    process.resourcesPath ? join(process.resourcesPath, 'bin') : '',
    join(process.cwd(), 'resources', 'bin'),
    join(HERE, '..', '..', 'resources', 'bin'),
    // Where the official uv installer drops uv/uvx for the current user.
    join(homedir(), '.local', 'bin')
  ]
  cachedBinDirs = candidates.filter((d) => d && existsSync(d))
  return cachedBinDirs
}

function withBundledPath(env: Record<string, string>): Record<string, string> {
  // Collapse every case-variant of PATH (Windows exposes `Path`) into a single
  // canonical uppercase `PATH`. The MCP SDK's getDefaultEnvironment() injects an
  // uppercase `PATH` (with the un-augmented value) and merges it BEFORE our env;
  // if we left a differently-cased `Path` key around, the spawned process would
  // carry BOTH keys and cross-spawn could resolve the launcher against the stale
  // one — producing `spawn uvx ENOENT` even though our bin dir was on `Path`.
  let current = ''
  for (const k of Object.keys(env)) {
    if (k.toLowerCase() === 'path') {
      if (!current) current = env[k]
      delete env[k]
    }
  }
  const dirs = bundledBinDirs()
  env.PATH = [...dirs, current].filter(Boolean).join(delimiter)
  return env
}

// Strip a version suffix from an npm spec: '@scope/pkg@1.2.3' -> '@scope/pkg'.
function packageName(spec: string): string {
  const at = spec.indexOf('@', spec.startsWith('@') ? 1 : 0)
  return at === -1 ? spec : spec.slice(0, at)
}

// Resolve an `npx -y <pkg> ...args` invocation to the package's bin entry so it
// can be run directly by Node, skipping npx and its download. Returns null when
// the package isn't bundled as a dependency, so the caller falls back to npx.
function resolveNodePackage(args: string[]): { entry: string; rest: string[] } | null {
  const rest = [...args]
  const flags = new Set(['-y', '--yes', '-q', '--quiet', '--'])
  while (rest.length && flags.has(rest[0])) rest.shift()
  const spec = rest.shift()
  if (!spec) return null
  const name = packageName(spec)
  try {
    const pkgJsonPath = nodeRequire.resolve(`${name}/package.json`)
    const pkg = JSON.parse(readFileSync(pkgJsonPath, 'utf8')) as {
      name?: string
      bin?: string | Record<string, string>
    }
    let binRel: string | undefined
    if (typeof pkg.bin === 'string') binRel = pkg.bin
    else if (pkg.bin) binRel = pkg.bin[name] ?? pkg.bin[pkg.name ?? ''] ?? Object.values(pkg.bin)[0]
    if (!binRel) return null
    return { entry: resolve(dirname(pkgJsonPath), binRel), rest }
  } catch {
    return null
  }
}

interface StdioLaunch {
  command: string
  args: string[]
  env: Record<string, string>
}

// Turn a configured stdio server into the actual command to spawn. Two
// transforms make bundled/ambient toolchains unnecessary for end users:
//   • PATH is augmented with our bundled bin dir + uv's install dir, so `uvx`
//     resolves without a system-wide install or a freshly-restarted shell.
//   • `npx <pkg>` launchers are rerouted through Electron's built-in Node
//     (process.execPath + ELECTRON_RUN_AS_NODE) against the bundled package, so
//     Node itself need not be installed.
function resolveStdioLaunch(server: {
  command: string
  args?: string[]
  env?: Record<string, string>
}): StdioLaunch {
  const env = withBundledPath({ ...cleanEnv(process.env), ...server.env })
  const command = server.command
  const args = server.args ?? []

  if (NODE_LAUNCHERS.has(command.toLowerCase())) {
    const resolved = resolveNodePackage(args)
    if (resolved) {
      return {
        command: process.execPath,
        args: [resolved.entry, ...resolved.rest],
        env: { ...env, ELECTRON_RUN_AS_NODE: '1' }
      }
    }
  }
  return { command, args, env }
}

interface Connected {
  id: string
  client: Client
  /** Sanitized-tool-name -> original MCP tool name. */
  toolNames: Map<string, string>
}

// Owns one MCP client connection per enabled server, aggregates their tools
// into a single flat catalogue, and routes tool calls back to the right
// server. Both stdio (subprocess) and Streamable HTTP transports are handled.
export class McpManager {
  private connected: Connected[] = []
  private tools: AgentTool[] = []
  private openaiTools: ChatCompletionTool[] = []
  // Per-server connection outcome, surfaced to the Pantry UI.
  private runtime = new Map<string, { connected: boolean; toolCount: number; error?: string }>()

  async connectAll(servers: McpServerConfig[]): Promise<void> {
    // Fresh catalogue each time so repeated calls (e.g. after the user toggles
    // a tool) don't accumulate duplicates or stale state.
    this.tools = []
    this.openaiTools = []
    this.runtime = new Map()

    for (const server of servers) {
      try {
        await this.connectOne(server)
      } catch (err) {
        // A single bad server must not sink the whole app; log and continue.
        this.runtime.set(server.id, { connected: false, toolCount: 0, error: String(err) })
        console.error(`[mcp] failed to connect "${server.id}":`, err)
      }
    }
  }

  private async connectOne(server: McpServerConfig): Promise<void> {
    const client = new Client({ name: 'lemonade-stand', version: '0.1.0' })

    if (server.transport === 'stdio') {
      // resolveStdioLaunch augments PATH with our bundled uv (resources/bin)
      // and uv's user install dir, and reroutes npx launchers through
      // Electron's built-in Node , so the uvx/npx toolchains don't need to be
      // installed (or on a freshly-restarted PATH) for a server to start. The
      // MCP SDK otherwise spawns with a stripped env, which is why stdio servers
      // connect in `npm run dev` but silently fail in a packaged build. Config
      // `env` overrides still win.
      const launch = resolveStdioLaunch(server)
      const transport = new StdioClientTransport({
        command: launch.command,
        args: launch.args,
        env: launch.env
      })
      await client.connect(transport)
    } else {
      const transport = new StreamableHTTPClientTransport(new URL(server.url), {
        requestInit: server.headers ? { headers: server.headers } : undefined
      })
      await client.connect(transport)
    }

    const { tools } = await client.listTools()
    const toolNames = new Map<string, string>()

    for (const tool of tools) {
      const qualified = sanitize(`${server.id}${SEP}${tool.name}`)
      toolNames.set(qualified, tool.name)

      this.tools.push({
        qualifiedName: qualified,
        serverId: server.id,
        toolName: tool.name,
        description: tool.description ?? ''
      })

      this.openaiTools.push({
        type: 'function',
        function: {
          name: qualified,
          description: tool.description ?? '',
          parameters: (tool.inputSchema as Record<string, unknown>) ?? {
            type: 'object',
            properties: {}
          }
        }
      })
    }

    this.connected.push({ id: server.id, client, toolNames })
    this.runtime.set(server.id, { connected: true, toolCount: tools.length })
    console.log(`[mcp] connected "${server.id}" (${tools.length} tools)`)
  }

  getTools(): AgentTool[] {
    return this.tools
  }

  getOpenAiTools(): ChatCompletionTool[] {
    return this.openaiTools
  }

  /** Connection outcome for one server, for the Pantry UI. */
  getRuntime(id: string): { connected: boolean; toolCount: number; error?: string } | undefined {
    return this.runtime.get(id)
  }

  /**
   * Execute a qualified tool call. Returns a text rendering of the tool's
   * content blocks (for feeding back to the model as a tool message) plus any
   * image blocks the tool produced, so the caller can surface them to the user
   * (e.g. on the Napkin panel) instead of discarding the bytes.
   */
  async callTool(qualifiedName: string, args: Record<string, unknown>): Promise<McpToolResult> {
    const owner = this.connected.find((c) => c.toolNames.has(qualifiedName))
    if (!owner) throw new Error(`No connected server owns tool "${qualifiedName}"`)
    const originalName = owner.toolNames.get(qualifiedName)!

    const result = await owner.client.callTool({ name: originalName, arguments: args })

    // Flatten MCP content blocks. Text blocks feed the model as usual. Image
    // blocks would blow up the context window if inlined, so they're pulled out
    // and returned separately for the caller to display; the model sees a short
    // placeholder that tells it the image was already shown to the user.
    const blocks = Array.isArray(result.content) ? result.content : []
    const parts: string[] = []
    const images: McpToolImage[] = []
    for (const block of blocks) {
      if (block.type === 'text') {
        parts.push(block.text)
      } else if (block.type === 'image' && typeof block.data === 'string') {
        const mimeType =
          typeof block.mimeType === 'string' && /^image\//i.test(block.mimeType)
            ? block.mimeType
            : 'image/png'
        images.push({ data: block.data, mimeType })
        parts.push('[image shown to the user on the napkin panel]')
      } else {
        parts.push(`[${block.type} content omitted]`)
      }
    }
    const text = parts.join('\n').trim()
    if (result.isError) return { text: `Tool error: ${text || 'unknown error'}`, images }
    return { text: text || '(tool returned no content)', images }
  }

  async closeAll(): Promise<void> {
    for (const c of this.connected) {
      try {
        await c.client.close()
      } catch {
        // Best-effort shutdown.
      }
    }
    this.connected = []
    this.tools = []
    this.openaiTools = []
  }
}
