import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import type { ChatCompletionTool } from 'openai/resources/chat/completions'
import type { AgentTool, McpServerConfig } from '@shared/types'

const SEP = '__'

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
      const transport = new StdioClientTransport({
        command: server.command,
        args: server.args ?? [],
        // Inherit the app's environment (notably PATH) so the launcher — npx,
        // node, uvx, etc. — resolves when the app is started from the OS shell
        // rather than a terminal. The MCP SDK otherwise spawns with a stripped
        // default env, which is why stdio servers connect in `npm run dev` but
        // silently fail in a packaged build. Config `env` overrides still win.
        env: { ...cleanEnv(process.env), ...server.env }
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
   * content blocks suitable for feeding back to the model as a tool message.
   */
  async callTool(qualifiedName: string, args: Record<string, unknown>): Promise<string> {
    const owner = this.connected.find((c) => c.toolNames.has(qualifiedName))
    if (!owner) throw new Error(`No connected server owns tool "${qualifiedName}"`)
    const originalName = owner.toolNames.get(qualifiedName)!

    const result = await owner.client.callTool({ name: originalName, arguments: args })

    // Flatten MCP content blocks to text. Non-text blocks (image/audio) are
    // summarized rather than inlined — this is a text agent loop; binary media
    // would blow up the context window.
    const blocks = Array.isArray(result.content) ? result.content : []
    const parts: string[] = []
    for (const block of blocks) {
      if (block.type === 'text') parts.push(block.text)
      else parts.push(`[${block.type} content omitted]`)
    }
    const text = parts.join('\n').trim()
    if (result.isError) return `Tool error: ${text || 'unknown error'}`
    return text || '(tool returned no content)'
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
