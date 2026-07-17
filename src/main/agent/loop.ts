import type {
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall
} from 'openai/resources/chat/completions'
import type { AgentEvent, ChatMessage } from '@shared/types'
import type { LemonadeClient } from '../lemonade/client'
import type { McpManager } from '../mcp/manager'

// Asks the user to approve a single tool call. Resolves true to proceed, false
// to skip. The main process implements this by prompting the renderer; a
// non-interactive caller can pass a function that always resolves true.
export type ApproveFn = (params: {
  server: string
  tool: string
  qualified: string
  args: unknown
}) => Promise<boolean>

// Bounded tool-calling loop. Each user turn:
//   1. ask lemond for a completion with the current MCP tool catalogue;
//   2. if it returns tool_calls, execute each via the MCP manager, append the
//      results as `tool` messages, and loop;
//   3. otherwise emit the assistant text and finish.
// `maxSteps` caps iterations so a misbehaving model or flaky tool server can't
// spin forever.
export class Agent {
  constructor(
    private lemonade: LemonadeClient,
    private mcp: McpManager,
    private maxSteps: number,
    private systemPrompt: string
  ) {}

  async run(
    history: ChatMessage[],
    emit: (event: AgentEvent) => void,
    approve: ApproveFn,
    signal?: AbortSignal
  ): Promise<void> {
    const messages = history as ChatCompletionMessageParam[]
    const tools = this.mcp.getOpenAiTools()

    // Prime the model to actually call tools rather than describe them. Only
    // inject if the caller hasn't already supplied a system message.
    if (this.systemPrompt && !messages.some((m) => m.role === 'system')) {
      messages.unshift({ role: 'system', content: this.systemPrompt })
    }

    try {
      for (let step = 0; step < this.maxSteps; step++) {
        // The user asked to halt: stop before starting more work.
        if (signal?.aborted) {
          emit({ type: 'done' })
          return
        }
        // Pre-flight: compare the request against the model's context window so
        // we can warn the user (or block) instead of letting the server reject
        // an over-long prompt with an opaque error.
        const budget = await this.lemonade.checkBudget(messages, tools)
        if (budget.overflow) {
          emit({
            type: 'context_warning',
            estimatedTokens: budget.estimatedTokens,
            contextSize: budget.contextSize,
            reserve: budget.reserve,
            overflow: true
          })
          emit({
            type: 'error',
            message:
              `This request needs about ${budget.estimatedTokens} tokens, but the model's ` +
              `context window is ${budget.contextSize} (usable ${budget.budget} after reserving ` +
              `${budget.reserve} for the reply). Shorten the conversation, disable some tools, ` +
              `or raise the context — increase the server's context size or set LEMONADE_CONTEXT_SIZE.`
          })
          emit({ type: 'done' })
          return
        }
        if (budget.warn) {
          emit({
            type: 'context_warning',
            estimatedTokens: budget.estimatedTokens,
            contextSize: budget.contextSize,
            reserve: budget.reserve,
            overflow: false
          })
        }

        const choice = await this.lemonade.chat(messages, tools, signal)
        const message = choice.message

        // Record the assistant turn verbatim so tool results can reference its
        // tool_call ids on the next iteration.
        messages.push(message as ChatCompletionMessageParam)

        const toolCalls = message.tool_calls ?? []
        if (toolCalls.length === 0) {
          emit({ type: 'assistant_text', text: message.content ?? '' })
          emit({ type: 'done' })
          return
        }

        for (const call of toolCalls) {
          await this.executeCall(call, messages, emit, approve)
        }
      }

      emit({
        type: 'error',
        message: `Reached step limit (${this.maxSteps}) without a final answer.`
      })
      emit({ type: 'done' })
    } catch (err) {
      // A user-triggered halt surfaces as an abort; end quietly rather than
      // reporting it as a failure.
      if (signal?.aborted) {
        emit({ type: 'done' })
        return
      }
      emit({ type: 'error', message: err instanceof Error ? err.message : String(err) })
      emit({ type: 'done' })
    }
  }

  private async executeCall(
    call: ChatCompletionMessageToolCall,
    messages: ChatCompletionMessageParam[],
    emit: (event: AgentEvent) => void,
    approve: ApproveFn
  ): Promise<void> {
    if (call.type !== 'function') return
    const qualified = call.function.name

    let args: Record<string, unknown> = {}
    try {
      args = call.function.arguments ? JSON.parse(call.function.arguments) : {}
    } catch {
      // Malformed arguments -> hand the error back to the model to retry.
    }

    const [serverId, ...rest] = qualified.split('__')
    const toolLabel = rest.join('__') || qualified

    // Gate on user approval BEFORE announcing/executing the call. A denied tool
    // gets a synthetic result so the model can react rather than hang.
    const allowed = await approve({ server: serverId, tool: toolLabel, qualified, args })
    if (!allowed) {
      emit({ type: 'tool_result', server: serverId, tool: toolLabel, ok: false, preview: 'Denied by user' })
      messages.push({
        role: 'tool',
        tool_call_id: call.id,
        content: 'The user denied permission to run this tool. Do not retry it; continue without it.'
      })
      return
    }

    emit({ type: 'tool_call', server: serverId, tool: toolLabel, args })

    let resultText: string
    let ok = true
    try {
      resultText = await this.mcp.callTool(qualified, args)
    } catch (err) {
      ok = false
      resultText = `Tool call failed: ${err instanceof Error ? err.message : String(err)}`
    }

    emit({
      type: 'tool_result',
      server: serverId,
      tool: toolLabel,
      ok,
      preview: resultText.slice(0, 280)
    })

    messages.push({
      role: 'tool',
      tool_call_id: call.id,
      content: resultText
    })
  }
}
