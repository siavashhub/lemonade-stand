import type {
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall,
  ChatCompletionTool
} from 'openai/resources/chat/completions'
import type { AgentEvent, ChatMessage, PlanStep } from '@shared/types'
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

// Name of the built-in planning tool. It is not backed by an MCP server: the
// loop synthesizes it, handles its calls internally, and surfaces the plan to
// the UI. Kept flat (no `__` namespace) so it can't collide with a server tool.
const PLAN_TOOL = 'update_plan'

// Schema for the built-in planning tool, appended to the MCP tool catalogue so
// the model can optionally lay out — and revise — a short todo list for
// multi-step work. The model always sends the full list; statuses drive the
// live checklist in the UI.
const PLAN_TOOL_DEF: ChatCompletionTool = {
  type: 'function',
  function: {
    name: PLAN_TOOL,
    description:
      'Create or revise a short todo plan for a multi-step task. Call this before ' +
      'starting non-trivial work to lay out the steps, then call it again as you ' +
      'progress to mark steps in-progress or completed. Always pass the FULL, ' +
      'current list of steps. Skip this for simple one-step requests.',
    parameters: {
      type: 'object',
      properties: {
        steps: {
          type: 'array',
          description: 'The full, ordered list of plan steps in their current state.',
          items: {
            type: 'object',
            properties: {
              title: {
                type: 'string',
                description: 'Short imperative description of the step.'
              },
              status: {
                type: 'string',
                enum: ['pending', 'in-progress', 'completed'],
                description: 'Current state of this step.'
              }
            },
            required: ['title', 'status']
          }
        }
      },
      required: ['steps']
    }
  }
}

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
    private systemPrompt: string,
    /** Fraction (0-1) of the usable context budget at which older messages are
     * summarized to free room. 0 disables auto-compaction. */
    private compactThreshold = 0.75
  ) {}

  async run(
    history: ChatMessage[],
    emit: (event: AgentEvent) => void,
    approve: ApproveFn,
    signal?: AbortSignal
  ): Promise<void> {
    const messages = history as ChatCompletionMessageParam[]
    // Expose the MCP tools plus the built-in planning tool for this turn.
    const tools = [...this.mcp.getOpenAiTools(), PLAN_TOOL_DEF]

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
        // Keep long conversations inside the context window by summarizing the
        // older messages before they overflow. Runs each iteration so a turn
        // that balloons via tool output is also caught.
        await this.maybeCompact(messages, tools, emit, signal)
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

  // Prefix marking the injected summary system message so a later compaction
  // can recognize, fold, and replace it instead of stacking summaries.
  private static readonly SUMMARY_PREFIX = 'Summary of the conversation so far:'

  /**
   * Summarize older messages when the request approaches the context budget, so
   * a long conversation can keep going instead of hitting a hard overflow.
   *
   * Safety rules that must hold for lemond to accept the result:
   *  - the leading system prompt is always preserved;
   *  - the kept "tail" always begins at a `user` message, so we never split an
   *    assistant `tool_calls` message from its matching `tool` results.
   * When no clean boundary exists yet, it does nothing and lets the pre-flight
   * budget check warn/block as before.
   */
  private async maybeCompact(
    messages: ChatCompletionMessageParam[],
    tools: ChatCompletionTool[],
    emit: (event: AgentEvent) => void,
    signal?: AbortSignal
  ): Promise<void> {
    if (this.compactThreshold <= 0) return
    const budget = await this.lemonade.checkBudget(messages, tools)
    if (budget.estimatedTokens <= budget.budget * this.compactThreshold) return
    if (await this.compactMessages(messages, signal)) {
      emit({ type: 'history_compacted', messages: messages as ChatMessage[] })
    }
  }

  /**
   * Force a compaction pass regardless of the budget threshold, for the UI's
   * manual "Compact Conversation" button. Returns the new model-facing history
   * when it summarized, or null when there was nothing safe to fold (so the
   * caller can leave the conversation untouched).
   */
  async compact(history: ChatMessage[], signal?: AbortSignal): Promise<ChatMessage[] | null> {
    const messages = history.slice() as ChatCompletionMessageParam[]
    const changed = await this.compactMessages(messages, signal)
    return changed ? (messages as ChatMessage[]) : null
  }

  /**
   * Core summarize-and-keep-tail compaction, mutating `messages` in place.
   * Returns true when it replaced older messages with a summary, false when no
   * clean boundary existed or summarization produced nothing.
   *
   * Safety rules that must hold for lemond to accept the result:
   *  - leading system messages are always preserved;
   *  - the kept "tail" always begins at a `user` message, so an assistant
   *    `tool_calls` message is never split from its matching `tool` results.
   */
  private async compactMessages(
    messages: ChatCompletionMessageParam[],
    signal?: AbortSignal
  ): Promise<boolean> {
    // Preserve leading system messages; pull out any prior summary to fold in.
    let sysEnd = 0
    while (sysEnd < messages.length && messages[sysEnd].role === 'system') sysEnd++
    const prefix: ChatCompletionMessageParam[] = []
    let priorSummary = ''
    for (let i = 0; i < sysEnd; i++) {
      const m = messages[i]
      const content = typeof m.content === 'string' ? m.content : ''
      if (content.startsWith(Agent.SUMMARY_PREFIX)) {
        priorSummary = content.slice(Agent.SUMMARY_PREFIX.length).trim()
      } else {
        prefix.push(m)
      }
    }

    // Keep the most recent turns verbatim; find a clean `user` boundary at or
    // after the desired tail start so tool sequences stay intact.
    const KEEP_TAIL = 6
    let cut = Math.max(sysEnd, messages.length - KEEP_TAIL)
    while (cut < messages.length && messages[cut].role !== 'user') cut++
    const head = messages.slice(sysEnd, cut)
    if (cut >= messages.length || head.length === 0) return false // nothing safe to fold

    const tail = messages.slice(cut)
    let summary: string
    try {
      summary = await this.lemonade.summarize(head, priorSummary, signal)
    } catch {
      return false // summarization failed; leave history untouched
    }
    if (!summary) return false

    // Rebuild in place: [system prefix, folded summary, recent tail].
    messages.length = 0
    messages.push(...prefix)
    messages.push({ role: 'system', content: `${Agent.SUMMARY_PREFIX}\n${summary}` })
    messages.push(...tail)
    return true
  }

  /**
   * Coerce the model's `update_plan` arguments into a clean PlanStep[]. Tolerant
   * of a weak model's rough output: drops non-object/empty steps, trims titles,
   * and defaults an unknown status to 'pending' so the UI always gets valid data.
   */
  private normalizePlan(args: Record<string, unknown>): PlanStep[] {
    const raw = Array.isArray(args.steps) ? args.steps : []
    const steps: PlanStep[] = []
    for (const item of raw) {
      if (!item || typeof item !== 'object') continue
      const rec = item as Record<string, unknown>
      const title = typeof rec.title === 'string' ? rec.title.trim() : ''
      if (!title) continue
      const status =
        rec.status === 'in-progress' || rec.status === 'completed' ? rec.status : 'pending'
      steps.push({ title, status })
    }
    return steps
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

    // The built-in planning tool is handled in-process: no MCP server owns it
    // and it never needs user approval. Surface the plan to the UI and hand a
    // short acknowledgement back to the model so it keeps working.
    if (qualified === PLAN_TOOL) {
      const steps = this.normalizePlan(args)
      emit({ type: 'plan_updated', steps })
      messages.push({
        role: 'tool',
        tool_call_id: call.id,
        content:
          steps.length > 0
            ? `Plan updated (${steps.length} steps). Continue working through it.`
            : 'Plan cleared.'
      })
      return
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
