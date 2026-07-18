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

// Asked when the agent exhausts its step budget without a final answer.
// Resolves true to grant another budget and keep going, false to stop. The main
// process implements this by prompting the renderer; a non-interactive caller
// can omit it, in which case the agent stops at the limit as before.
export type ContinueFn = (steps: number) => Promise<boolean>

// Safety valve: if the model calls only `update_plan` this many turns in a row
// without doing any real work, stop rather than spin forever on plan edits.
const MAX_PLAN_ONLY_STREAK = 8

// Once a task has done this many real-work turns without ever planning, force a
// single planning round-trip (Option B). Kept at 1 so the plan appears early —
// as soon as the model takes a second action without a plan — while a task that
// finishes in a single tool call stays plan-free.
const FORCE_PLAN_AFTER = 1

// The nudge injected to force that planning round-trip.
const FORCE_PLAN_NUDGE =
  'You are working through a multi-step task but have not laid out a plan yet. ' +
  'Before taking any more actions, call the update_plan tool to list the steps ' +
  'as a short todo list. Then continue with the work.'

// If the model tries to stop while its plan still has unfinished steps (e.g.
// after a mid-task compaction confused it), nudge it to keep going. Bounded so
// a model that genuinely can't finish isn't looped forever.
const MAX_PLAN_FINISH_NUDGES = 3

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
    private compactThreshold = 0.9
  ) {}

  async run(
    history: ChatMessage[],
    emit: (event: AgentEvent) => void,
    approve: ApproveFn,
    signal?: AbortSignal,
    onLimit?: ContinueFn
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
      // `step` counts only turns that did real work; `update_plan`-only turns
      // are free so planning doesn't eat the work budget. `limit` starts at
      // maxSteps and is extended each time the user opts to keep going.
      let step = 0
      let limit = this.maxSteps
      let planOnlyStreak = 0
      // Whether the model has produced a plan this turn, and whether we've
      // already forced one — so the Option-B nudge fires at most once.
      let hasPlanned = false
      let forcedPlan = false
      // The latest plan the model set, and how many times we've nudged it to
      // finish unfinished steps, so a premature stop can be caught and resumed.
      let currentPlan: PlanStep[] = []
      let finishNudges = 0
      for (;;) {
        // The user asked to halt: stop before starting more work.
        if (signal?.aborted) {
          emit({ type: 'done' })
          return
        }
        // Keep long conversations inside the context window by summarizing the
        // older messages before they overflow. Runs each iteration so a turn
        // that balloons via tool output is also caught. The current plan is
        // pinned so it survives compaction intact.
        await this.maybeCompact(messages, tools, emit, signal, currentPlan)
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
        const toolCalls = message.tool_calls ?? []

        // Final answer (no tool calls). If the model is trying to stop while its
        // plan still has unfinished steps (often after a mid-task compaction
        // muddled its state), nudge it to keep going instead of ending.
        if (toolCalls.length === 0) {
          const unfinished = currentPlan.filter((s) => s.status !== 'completed')
          if (unfinished.length > 0 && finishNudges < MAX_PLAN_FINISH_NUDGES) {
            finishNudges++
            messages.push({
              role: 'user',
              content:
                `Your plan still has unfinished steps: ${unfinished
                  .map((s) => s.title)
                  .join('; ')}. Keep working — actually perform each remaining step ` +
                `with the appropriate tool, updating the plan as you complete them, ` +
                `and only give your final summary once every step is done.`
            })
            continue
          }
          messages.push(message as ChatCompletionMessageParam)
          emit({ type: 'assistant_text', text: message.content ?? '' })
          emit({ type: 'done' })
          return
        }

        const wantsRealWork = toolCalls.some(
          (c) => c.type === 'function' && c.function.name !== PLAN_TOOL
        )

        // Option B: the task is clearly multi-step (already did FORCE_PLAN_AFTER
        // real-work turns) but the model still hasn't planned. Drop this
        // un-planned work turn — we never pushed it, so there are no orphaned
        // tool_calls — and re-ask with a nudge so the model plans first.
        if (wantsRealWork && !hasPlanned && !forcedPlan && step >= FORCE_PLAN_AFTER) {
          forcedPlan = true
          messages.push({ role: 'user', content: FORCE_PLAN_NUDGE })
          continue
        }

        // Record the assistant turn verbatim so tool results can reference its
        // tool_call ids on the next iteration.
        messages.push(message as ChatCompletionMessageParam)

        // Execute every requested call. A turn "did real work" if it invoked at
        // least one non-planning tool; a plan-only turn stays free.
        let didRealWork = false
        for (const call of toolCalls) {
          if (call.type === 'function' && call.function.name === PLAN_TOOL) {
            hasPlanned = true
          } else {
            didRealWork = true
          }
          const planned = await this.executeCall(call, messages, emit, approve)
          // Track the latest plan so a premature stop can be detected against it.
          if (planned) currentPlan = planned
        }

        if (didRealWork) {
          step++
          planOnlyStreak = 0
        } else {
          // Free plan-only turn — but guard against a model that loops forever
          // revising its plan without ever acting on it.
          if (++planOnlyStreak >= MAX_PLAN_ONLY_STREAK) {
            emit({
              type: 'error',
              message:
                'The agent kept updating its plan without making progress, so it was stopped.'
            })
            emit({ type: 'done' })
            return
          }
          continue
        }

        // Budget exhausted without a final answer. Ask the user whether to keep
        // going (each yes grants another budget); non-interactive callers stop.
        if (step >= limit) {
          const keepGoing = onLimit ? await onLimit(limit) : false
          if (!keepGoing) {
            if (!onLimit) {
              emit({
                type: 'error',
                message: `Reached step limit (${limit}) without a final answer.`
              })
            }
            emit({ type: 'done' })
            return
          }
          limit += this.maxSteps
        }
      }
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

  // Prefix marking the pinned-plan system message. Kept out of the summary so
  // the plan (with live progress) survives every compaction verbatim and the
  // model never loses track of what's left to do. Re-generated from the current
  // plan on each compaction, replacing any prior pin.
  private static readonly PLAN_PREFIX = 'Current plan (not finished until every step is checked):'

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
    signal?: AbortSignal,
    plan?: PlanStep[]
  ): Promise<void> {
    if (this.compactThreshold <= 0) return
    const budget = await this.lemonade.checkBudget(messages, tools)
    if (budget.estimatedTokens <= budget.budget * this.compactThreshold) return
    if (await this.compactMessages(messages, signal, plan)) {
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
    signal?: AbortSignal,
    plan?: PlanStep[]
  ): Promise<boolean> {
    // Preserve leading system messages; pull out any prior summary to fold in,
    // and drop any prior plan pin (a fresh one is re-added from `plan` below).
    let sysEnd = 0
    while (sysEnd < messages.length && messages[sysEnd].role === 'system') sysEnd++
    const prefix: ChatCompletionMessageParam[] = []
    let priorSummary = ''
    for (let i = 0; i < sysEnd; i++) {
      const m = messages[i]
      const content = typeof m.content === 'string' ? m.content : ''
      if (content.startsWith(Agent.SUMMARY_PREFIX)) {
        priorSummary = content.slice(Agent.SUMMARY_PREFIX.length).trim()
      } else if (content.startsWith(Agent.PLAN_PREFIX)) {
        // Drop stale pin; the live plan is re-pinned after summarizing.
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

    // Rebuild in place: [system prefix, folded summary, pinned plan, recent tail].
    messages.length = 0
    messages.push(...prefix)
    messages.push({ role: 'system', content: `${Agent.SUMMARY_PREFIX}\n${summary}` })
    const pin = this.renderPlanPin(plan)
    if (pin) messages.push({ role: 'system', content: pin })
    messages.push(...tail)
    return true
  }

  /**
   * Render the current plan as a compact checklist to pin into context after a
   * compaction, so the model always sees the remaining work. Returns an empty
   * string when there's no plan to pin.
   */
  private renderPlanPin(plan?: PlanStep[]): string {
    if (!plan || plan.length === 0) return ''
    const mark = (s: PlanStep): string =>
      s.status === 'completed' ? '[x]' : s.status === 'in-progress' ? '[~]' : '[ ]'
    const lines = plan.map((s) => `${mark(s)} ${s.title}`).join('\n')
    return `${Agent.PLAN_PREFIX}\n${lines}`
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
  ): Promise<PlanStep[] | undefined> {
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
      return steps
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
