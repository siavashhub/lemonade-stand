import type {
  ChatCompletionMessage,
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall,
  ChatCompletionTool
} from 'openai/resources/chat/completions'
import type {
  AgentEvent,
  ChatMessage,
  Napkin,
  NapkinChoice,
  PitcherOutput,
  PitcherTrigger,
  PlanStep
} from '@shared/types'
import type { LemonadeClient } from '../lemonade/client'
import type { McpManager } from '../mcp/manager'
import { dirname } from 'path'

// Asks the user to approve a single tool call. Resolves true to proceed, false
// to skip. May instead resolve an object carrying a human-readable `reason` for
// a denial , used so a headless Pitcher pour can explain that a tool was blocked
// because it isn't in the task's allow-list (not "denied by the user"). The main
// process implements this by prompting the renderer; a non-interactive caller
// can pass a function that always resolves true.
export type ApproveResult = boolean | { allowed: boolean; reason?: string }
export type ApproveFn = (params: {
  server: string
  tool: string
  qualified: string
  args: unknown
}) => Promise<ApproveResult>

// Asked when the agent exhausts its step budget without a final answer.
// Resolves true to grant another budget and keep going, false to stop. The main
// process implements this by prompting the renderer; a non-interactive caller
// can omit it, in which case the agent stops at the limit as before.
export type ContinueFn = (steps: number) => Promise<boolean>

// Presents a multiple-choice clarifying question to the user and resolves with
// the id of the option they picked. The main process implements this by
// prompting the renderer (the choices render in the Napkin panel) and awaiting
// the reply. A non-interactive caller can omit it, in which case `ask_napkin`
// falls back to telling the model to ask in plain text.
export type AskNapkinFn = (params: {
  title: string
  prompt: string
  choices: NapkinChoice[]
}) => Promise<string>

// Proposes a new scheduled task ("Pitcher") for the user to review and confirm in
// a pre-filled editor. Resolves once they save it (saved:true) or cancel
// (saved:false); the built-in `create_pitcher` tool feeds that outcome back to
// the model. Deliberately only wired for INTERACTIVE turns , a headless pour is
// never given this callback, so a scheduled task can't create or steer other
// scheduled tasks. The draft carries NO tool whitelist: the user grants tools
// themselves in the editor, so the agent can never auto-grant capabilities.
export type ProposePitcherFn = (draft: {
  name: string
  prompt: string
  trigger: PitcherTrigger
  output: PitcherOutput
  allowedTools: string[]
}) => Promise<{ saved: boolean; name: string }>

// Safety valve: if the model calls only `update_plan` this many turns in a row
// without doing any real work, stop rather than spin forever on plan edits.
const MAX_PLAN_ONLY_STREAK = 8

// Once a task has done this many real-work turns without ever planning, force a
// single planning round-trip (Option B). Kept at 1 so the plan appears early,
// as soon as the model takes a second action without a plan, while a task that
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
// the model can optionally lay out , and revise , a short todo list for
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

// Names of the built-in Napkin tools. Like `update_plan`, they are not backed by
// any MCP server: the loop synthesizes them, handles their calls internally, and
// surfaces the result to the UI. Kept flat (no `__` namespace) so they can't
// collide with a server tool.
const SHOW_NAPKIN_TOOL = 'show_napkin'
const ASK_NAPKIN_TOOL = 'ask_napkin'

// Built-in tool that lets the model PROPOSE a scheduled task the user confirms.
// Like the others it's synthesized (no MCP server) and handled in-process, and
// it's only offered to interactive turns (see `run`).
const CREATE_PITCHER_TOOL = 'create_pitcher'

// Every tool handled in-process rather than dispatched to an MCP server. Used to
// separate real (task-advancing) tool calls from the built-ins when deciding
// whether to force a planning round-trip.
const BUILTIN_TOOLS = new Set([
  PLAN_TOOL,
  SHOW_NAPKIN_TOOL,
  ASK_NAPKIN_TOOL,
  CREATE_PITCHER_TOOL
])

// Schema for `show_napkin`: lets the model display a rich artifact in the side
// panel to enrich a reply beyond plain text. Raw HTML / live browsing is
// deliberately not offered; the renderer sanitizes everything before display.
const SHOW_NAPKIN_TOOL_DEF: ChatCompletionTool = {
  type: 'function',
  function: {
    name: SHOW_NAPKIN_TOOL,
    description:
      'Show a rich artifact in the side "Napkin" panel to enrich your reply beyond plain text: ' +
      'a formatted code block the user can copy, rendered Markdown, a Mermaid diagram, an SVG ' +
      'drawing, or an image. Use it when a visual or copyable artifact genuinely helps (code, ' +
      'diagrams, tables, illustrations). Keep replying in text as usual; the napkin supplements it.',
    parameters: {
      type: 'object',
      properties: {
        title: { type: 'string', description: 'Short heading for the panel.' },
        kind: {
          type: 'string',
          enum: ['code', 'markdown', 'mermaid', 'svg', 'image'],
          description:
            "The artifact type. 'code' = source code; 'markdown' = rich text; 'mermaid' = a " +
            "Mermaid diagram definition; 'svg' = raw SVG markup; 'image' = base64-encoded image bytes."
        },
        content: {
          type: 'string',
          description:
            'The artifact body: source text for code/markdown/mermaid/svg, or base64-encoded ' +
            'bytes (no data: prefix) for an image.'
        },
        language: {
          type: 'string',
          description: "For kind:'code', the language for display, e.g. 'ts', 'python', 'json'."
        },
        mimeType: {
          type: 'string',
          description: "For kind:'image', the image MIME type, e.g. 'image/png'."
        },
        alt: { type: 'string', description: "For kind:'image', descriptive alt text." }
      },
      required: ['title', 'kind', 'content']
    }
  }
}

// Schema for `ask_napkin`: a blocking, multiple-choice clarifying question. The
// loop pauses on the call until the user picks an option in the Napkin panel,
// then feeds their choice back to the model as the tool result.
const ASK_NAPKIN_TOOL_DEF: ChatCompletionTool = {
  type: 'function',
  function: {
    name: ASK_NAPKIN_TOOL,
    description:
      'Ask the user a multiple-choice clarifying question when their request is ambiguous and you ' +
      'need them to pick a direction before continuing. The choices appear as buttons in the ' +
      'Napkin panel; the loop pauses until the user selects one, then returns their choice to you. ' +
      'Prefer this over guessing when a decision materially changes what you do next.',
    parameters: {
      type: 'object',
      properties: {
        title: { type: 'string', description: 'Short heading for the prompt.' },
        prompt: { type: 'string', description: 'The question to ask the user.' },
        choices: {
          type: 'array',
          description: 'The options to offer. Provide 2-6 concise choices.',
          items: {
            type: 'object',
            properties: {
              id: {
                type: 'string',
                description: 'Stable id returned when chosen. Defaults to the label.'
              },
              label: { type: 'string', description: 'The option text shown on the button.' }
            },
            required: ['label']
          }
        }
      },
      required: ['prompt', 'choices']
    }
  }
}

// Schema for `create_pitcher`: propose a scheduled task the user reviews and
// confirms. It does NOT create anything directly , the call opens a pre-filled
// editor (with the tools it needs pre-selected) where the user confirms the
// trigger and tool whitelist, so the model can never silently schedule work or
// grant itself tools , the human still confirms. Only offered to interactive
// turns (never during a pour).
const CREATE_PITCHER_TOOL_DEF: ChatCompletionTool = {
  type: 'function',
  function: {
    name: CREATE_PITCHER_TOOL,
    description:
      'Propose a new scheduled task (a "Pitcher") that re-runs a saved prompt on a trigger. Use ' +
      'this when the user asks to schedule, automate, or repeat a task (e.g. "every morning at ' +
      '8", "each time the app opens"). This does NOT create the task directly: it opens a ' +
      'pre-filled editor for the user to review and confirm, and THEY choose which tools (if any) ' +
      'the task may run , you cannot grant tools yourself. The task runs unattended with no ' +
      'conversation history, so write a clear, fully self-contained prompt.',
    parameters: {
      type: 'object',
      properties: {
        name: {
          type: 'string',
          description: 'Short title for the task, also used as the saved-conversation title.'
        },
        prompt: {
          type: 'string',
          description:
            'The full, self-contained instruction the task runs each time. It has no chat ' +
            'history, so include every piece of context it needs.'
        },
        trigger: {
          type: 'object',
          description: 'When the task runs.',
          properties: {
            type: {
              type: 'string',
              enum: ['daily', 'on-open'],
              description:
                "'daily' fires at a local time each day; 'on-open' runs once each app launch."
            },
            at: {
              type: 'string',
              description: "For type 'daily', the 24-hour local time as HH:MM, e.g. '08:00'."
            }
          },
          required: ['type']
        },
        output: {
          type: 'string',
          enum: ['napkin', 'chat'],
          description:
            "Where results are served: 'chat' (saved conversation) or 'napkin' (rich artifact " +
            "panel). Defaults to 'chat'."
        },
        allowedTools: {
          type: 'array',
          items: { type: 'string' },
          description:
            'The tools this task will need to run unattended, given as their EXACT names as they ' +
            'appear in the tools available to you now (e.g. a filesystem or fetch tool). These are ' +
            'pre-selected in the review editor so the user only has to confirm; names that do not ' +
            'match a real tool are ignored, and the user can add or remove any before saving.'
        }
      },
      required: ['name', 'prompt', 'trigger']
    }
  }
}

// A model reply split into its optional chain-of-thought ("reasoning") and the
// clean, user-facing answer.
interface SplitReply {
  reasoning: string
  content: string
}

// Pull a model's chain-of-thought out of a completion message. Reasoning arrives
// one of two ways: a non-standard `reasoning_content` (or `reasoning`) field
// (DeepSeek / OGA style), or inline as a <think>...</think> block in the content.
// Either way the returned `content` is the clean, user-facing text with any
// <think> block stripped, so it's safe to feed back into the conversation.
function splitReasoning(message: ChatCompletionMessage): SplitReply {
  const rawContent = typeof message.content === 'string' ? message.content : ''
  const extra = message as { reasoning_content?: unknown; reasoning?: unknown }
  const field =
    typeof extra.reasoning_content === 'string'
      ? extra.reasoning_content
      : typeof extra.reasoning === 'string'
        ? extra.reasoning
        : ''
  if (field.trim()) {
    return { reasoning: field.trim(), content: rawContent }
  }
  const match = rawContent.match(/<think>([\s\S]*?)<\/think>/i)
  if (match) {
    return {
      reasoning: match[1].trim(),
      content: rawContent.replace(/<think>[\s\S]*?<\/think>/gi, '').trim()
    }
  }
  return { reasoning: '', content: rawContent }
}

// Return a copy of the assistant message safe to store in the model-facing
// history: the chain-of-thought is stripped (both the inline <think> block and
// any reasoning_content field) so it never re-enters the context window, while
// tool_calls and the rest of the shape are preserved.
function stripReasoningForHistory(
  message: ChatCompletionMessage,
  cleanContent: string
): ChatCompletionMessageParam {
  const copy = { ...(message as unknown as Record<string, unknown>) }
  delete copy.reasoning_content
  delete copy.reasoning
  copy.content = cleanContent
  return copy as unknown as ChatCompletionMessageParam
}

// Bounded tool-calling loop. Each user turn:
//   1. ask lemond for a completion with the current MCP tool catalogue;
//   2. if it returns tool_calls, execute each via the MCP manager, append the
//      results as `tool` messages, and loop;
//   3. otherwise emit the assistant text and finish.
// `maxSteps` caps iterations so a misbehaving model or flaky tool server can't
// spin forever.

// Extract folder path from filesystem tool results. Returns the directory
// containing the file that was created/written, or null if not detected.
function extractFolderPathFromToolResult(
  toolName: string,
  result: string
): string | null {
  // Only match actual file write operations, NOT directory operations
  const fileWriteTools = ['write_file', 'create_file', 'save_file', 'write']
  const lowerName = toolName.toLowerCase()
  if (!fileWriteTools.some((t) => lowerName.includes(t))) {
    return null
  }

  // Try to extract a file path from the result text
  // Look for patterns like "Wrote to /path/to/file.md" with file extensions
  // This regex requires an actual file extension to avoid false matches
  const pathMatch = result.match(/(?:Wrote to|File saved at|Written to|Successfully wrote to)[\s:]+([^\n]+\.[a-z0-9]+)/i)
  if (pathMatch && pathMatch[1]) {
    const filePath = pathMatch[1].trim()
    // Return the directory containing the file
    return dirname(filePath)
  }

  return null
}

// When an image-generation tool fails because the requested image model isn't
// available on the server, a weak local model tends to ruminate over the opaque
// error instead of acting (the "overthinks it and takes forever" failure). The
// Lemonade gateway auto-loads an *installed* image model, but it can't conjure a
// model that was never pulled , so `lemonade_generate_image` just errors. Turn
// that dead-end into a decisive, single next action so the model self-corrects
// in one step: list the installed models, retry with an available image model,
// or stop and tell the user to install one. Returns null when the failure isn't
// an image-model-availability problem (so ordinary errors pass through as-is).
function imageModelRecoveryHint(toolLabel: string, resultText: string): string | null {
  const isImageTool = /generate[_-]?image|text[_-]?to[_-]?image|txt2img/i.test(toolLabel)
  if (!isImageTool) return null
  // Covers both paths: a thrown call ("Tool call failed:") and a returned MCP
  // error result ("Tool error:"), which don't otherwise set ok=false.
  const failed = /^Tool (error|call failed):/i.test(resultText)
  if (!failed) return null
  const availability =
    /(not\s*found|not\s*available|unavailable|missing|unknown|no\s*such|does\s*not\s*exist|failed\s*to\s*load|could\s*not\s*(be\s*)?load)/i.test(
      resultText
    )
  if (!availability) return null
  return (
    `${resultText}\n\n` +
    'The requested image model is not available on the server, and the gateway cannot ' +
    'download one that was never installed. Take ONE of these actions now , do not deliberate ' +
    'at length or retry the same missing model:\n' +
    '1. Call lemonade_list_models to see which models are installed.\n' +
    '2. If any is an image-generation model (its id typically contains "sd", "flux", ' +
    '"stable-diffusion", or "dream"), call lemonade_generate_image again with that id in the ' +
    '`model` argument.\n' +
    '3. If none is installed, tell the user no image model is available and that they can ' +
    'install one (e.g. `lemonade-server pull <model>`), then stop.'
  )
}

// Weak local models sometimes ignore `show_napkin` and instead dump a base64
// image straight into their reply, as a Markdown image `![alt](data:image/...)`
// or a bare `data:image/...;base64,...` URL. That never renders in the chat
// transcript (and would bloat it with a giant blob), so pull every such image
// out into a Napkin artifact and return the text with the blobs removed. Any
// extracted images are surfaced on the napkin panel by the caller.
function extractInlineImages(text: string): {
  cleaned: string
  napkins: Napkin[]
} {
  const napkins: Napkin[] = []
  // Matches an optional Markdown image wrapper around a data: URL, or a bare
  // data: URL. Captures the alt text (when present), the mime subtype, and the
  // base64 payload.
  const pattern =
    /!\[([^\]]*)\]\(\s*data:image\/([a-z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+?)\s*\)|data:image\/([a-z0-9.+-]+);base64,([A-Za-z0-9+/=]+)/gi

  let cleaned = text.replace(pattern, (_m, alt, mdMime, mdData, bareMime, bareData) => {
    const mime = (mdMime ?? bareMime) as string
    const data = ((mdData ?? bareData) as string).replace(/\s+/g, '')
    if (!data) return ''
    napkins.push({
      title: (typeof alt === 'string' && alt.trim()) || `Generated image ${napkins.length + 1}`,
      kind: 'image',
      content: data,
      mimeType: `image/${mime.toLowerCase()}`,
      alt: typeof alt === 'string' && alt.trim() ? alt.trim() : undefined
    })
    return ''
  })

  if (napkins.length > 0) {
    // Tidy up whitespace left where the blobs were removed so the reply reads
    // cleanly (e.g. no empty lines or dangling spaces).
    cleaned = cleaned.replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim()
    if (!cleaned) {
      cleaned =
        napkins.length === 1
          ? 'Here is the image , see the napkin panel.'
          : `Here are ${napkins.length} images , see the napkin panel.`
    }
  }

  return { cleaned, napkins }
}

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
    onLimit?: ContinueFn,
    askNapkin?: AskNapkinFn,
    proposePitcher?: ProposePitcherFn
  ): Promise<void> {
    const messages = history as ChatCompletionMessageParam[]
    // Expose the MCP tools plus the built-in planning + napkin tools for this turn.
    const tools = [
      ...this.mcp.getOpenAiTools(),
      PLAN_TOOL_DEF,
      SHOW_NAPKIN_TOOL_DEF,
      ASK_NAPKIN_TOOL_DEF
    ]
    // Only an interactive turn (proposePitcher present) may propose scheduled
    // tasks. A headless pour never gets the tool, so a scheduled task can't
    // create or steer other scheduled tasks (no privilege escalation).
    if (proposePitcher) tools.push(CREATE_PITCHER_TOOL_DEF)

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
      // already forced one , so the Option-B nudge fires at most once.
      let hasPlanned = false
      let forcedPlan = false
      // The latest plan the model set, and how many times we've nudged it to
      // finish unfinished steps, so a premature stop can be caught and resumed.
      let currentPlan: PlanStep[] = []
      let finishNudges = 0
      // Seed the badge with the starting prompt size (system + tools) before the
      // first model call, so it reflects reality from the outset.
      await this.emitUsage(messages, tools, emit)
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
              `or raise the context , increase the server's context size or set LEMONADE_CONTEXT_SIZE.`
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

        const choice = await this.lemonade.chat(messages, tools, signal, (delta) =>
          emit({ type: 'reasoning_delta', text: delta })
        )
        const message = choice.message
        const toolCalls = message.tool_calls ?? []
        // Split off the chain-of-thought once, right after the completion and
        // before any branch. Emitting the final `reasoning` here finalizes the
        // live-streamed preview (collapsing its panel) for every path , including
        // the dropped forced-plan and finish-nudge turns below , so a streamed
        // thought never gets stranded open. `content` is the clean, user-facing
        // text with any <think> block stripped, reused wherever we store history.
        const { reasoning, content } = splitReasoning(message)
        if (reasoning) emit({ type: 'reasoning', text: reasoning })

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
                  .join('; ')}. Keep working , actually perform each remaining step ` +
                `with the appropriate tool, updating the plan as you complete them, ` +
                `and only give your final summary once every step is done.`
            })
            continue
          }
          messages.push(stripReasoningForHistory(message, content))
          // If the model inlined base64 image(s) into its reply instead of using
          // show_napkin, lift them onto the napkin panel and strip the blobs so
          // the chat shows a clean message rather than a giant data: URL.
          const { cleaned, napkins } = extractInlineImages(content)
          for (const napkin of napkins) emit({ type: 'napkin_show', napkin })
          emit({ type: 'assistant_text', text: cleaned })
          emit({ type: 'done' })
          return
        }

        const realCalls = toolCalls.filter(
          (c) => c.type === 'function' && !BUILTIN_TOOLS.has(c.function.name)
        )
        const wantsRealWork = realCalls.length > 0

        // Option B: force a single planning round-trip when the model is clearly
        // doing multi-step work but hasn't planned. Two triggers cover both
        // model styles: (a) it has already done FORCE_PLAN_AFTER work turns
        // (one-tool-per-turn models), or (b) this single completion batches 2+
        // tool calls (models that request everything at once). A genuinely
        // single-step task never hits either, so trivial requests stay
        // plan-free. We drop this un-planned turn , it was never pushed, so no
        // orphaned tool_calls , and re-ask with a nudge so the model plans first.
        if (
          wantsRealWork &&
          !hasPlanned &&
          !forcedPlan &&
          (step >= FORCE_PLAN_AFTER || realCalls.length >= 2)
        ) {
          forcedPlan = true
          messages.push({ role: 'user', content: FORCE_PLAN_NUDGE })
          continue
        }

        // Record the assistant turn verbatim so tool results can reference its
        // tool_call ids on the next iteration. The chain-of-thought (if any) was
        // already surfaced above; it's stripped from the stored message so it
        // never re-enters the model-facing context.
        messages.push(stripReasoningForHistory(message, content))

        // Execute every requested call. A turn "did real work" if it invoked at
        // least one non-planning tool; a plan-only turn stays free.
        let didRealWork = false
        for (const call of toolCalls) {
          if (call.type === 'function' && call.function.name === PLAN_TOOL) {
            hasPlanned = true
          } else {
            didRealWork = true
          }
          const planned = await this.executeCall(
            call,
            messages,
            emit,
            approve,
            askNapkin,
            proposePitcher
          )
          // Track the latest plan so a premature stop can be detected against it.
          if (planned) currentPlan = planned
        }

        // The prompt grew by this turn's tool calls/results , refresh the badge.
        await this.emitUsage(messages, tools, emit)

        if (didRealWork) {
          step++
          planOnlyStreak = 0
        } else {
          // Free plan-only turn , but guard against a model that loops forever
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
   * Emit the live per-category context usage for the current in-flight prompt so
   * the renderer's usage badge reflects the real size , tool calls and results
   * included , instead of just the committed chat history. Best-effort: a failed
   * estimate just skips this tick's update.
   */
  private async emitUsage(
    messages: ChatCompletionMessageParam[],
    tools: ChatCompletionTool[],
    emit: (event: AgentEvent) => void
  ): Promise<void> {
    try {
      const breakdown = await this.lemonade.contextBreakdown(messages, tools, this.systemPrompt)
      emit({ type: 'context_usage', breakdown })
    } catch {
      // Non-fatal: the badge simply won't update on this iteration.
    }
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

  /**
   * Coerce the model's `show_napkin` arguments into a clean Napkin, or null when
   * the payload is unusable (missing content or an unknown kind). Tolerant of a
   * weak model: trims fields and defaults the title.
   */
  private normalizeNapkin(args: Record<string, unknown>): Napkin | null {
    const kinds: Napkin['kind'][] = ['code', 'markdown', 'mermaid', 'svg', 'image']
    const kind = kinds.find((k) => k === args.kind)
    const content = typeof args.content === 'string' ? args.content : ''
    if (!kind || !content.trim()) return null
    const title =
      typeof args.title === 'string' && args.title.trim() ? args.title.trim() : 'Napkin'
    const napkin: Napkin = { title, kind, content }
    if (typeof args.language === 'string' && args.language.trim())
      napkin.language = args.language.trim()
    if (typeof args.mimeType === 'string' && args.mimeType.trim())
      napkin.mimeType = args.mimeType.trim()
    if (typeof args.alt === 'string' && args.alt.trim()) napkin.alt = args.alt.trim()
    return napkin
  }

  /**
   * Coerce the model's `ask_napkin` arguments into a prompt plus a de-duplicated
   * list of at most six choices. Accepts either `{id?, label}` objects or bare
   * strings, defaults an id to its label, and drops empty/duplicate options.
   */
  private normalizeAsk(args: Record<string, unknown>): {
    title: string
    prompt: string
    choices: NapkinChoice[]
  } {
    const prompt = typeof args.prompt === 'string' ? args.prompt.trim() : ''
    const title =
      typeof args.title === 'string' && args.title.trim() ? args.title.trim() : 'Quick question'
    const raw = Array.isArray(args.choices) ? args.choices : []
    const choices: NapkinChoice[] = []
    const seen = new Set<string>()
    for (const item of raw) {
      let label = ''
      let id = ''
      if (typeof item === 'string') {
        label = item.trim()
      } else if (item && typeof item === 'object') {
        const rec = item as Record<string, unknown>
        label = typeof rec.label === 'string' ? rec.label.trim() : ''
        id = typeof rec.id === 'string' ? rec.id.trim() : ''
      }
      if (!label) continue
      if (!id) id = label
      if (seen.has(id)) continue
      seen.add(id)
      choices.push({ id, label })
      if (choices.length >= 6) break
    }
    return { title, prompt, choices }
  }

  /**
   * Coerce the model's `create_pitcher` arguments into a clean, safe draft, or
   * null when unusable (no prompt). Tolerant of a weak model's rough output:
   * clamps the name, defaults the output to 'chat', and only accepts a daily
   * trigger with a valid 24-hour HH:MM time (else falls back to on-open). The
   * proposed tool whitelist is filtered to tools that actually exist right now
   * (a hallucinated name is dropped) and merely PRE-SELECTS them in the review
   * editor , the user still confirms/edits the list before it's saved, so the
   * model can't grant a task tools the user didn't approve.
   */
  private normalizePitcherDraft(args: Record<string, unknown>): {
    name: string
    prompt: string
    trigger: PitcherTrigger
    output: PitcherOutput
    allowedTools: string[]
  } | null {
    const prompt = typeof args.prompt === 'string' ? args.prompt.trim() : ''
    if (!prompt) return null
    const rawName = typeof args.name === 'string' ? args.name.trim() : ''
    const name = (rawName || 'Scheduled task').slice(0, 80)
    // Default to serving a saved conversation; only 'napkin' opts into the panel.
    const output: PitcherOutput = args.output === 'napkin' ? 'napkin' : 'chat'

    // Accept the trigger as a nested object ({type, at}) or a loose top-level
    // `at` string; anything that can't be read as a time becomes on-open. The
    // parser is deliberately forgiving of how a weak model writes a time , "8",
    // "8am", "8:30 pm", "0800" all work , since it often skips the strict HH:MM.
    const toDaily = (value: unknown): PitcherTrigger | null => {
      if (typeof value !== 'string') return null
      let s = value.trim().toLowerCase()
      if (!s) return null
      const ampm = /(am|pm)/.exec(s)?.[1]
      s = s.replace(/\s*(am|pm)\s*/, '').trim()
      let h: number
      let min = 0
      const m = s.match(/^(\d{1,2})(?::(\d{1,2}))?$/)
      if (m) {
        h = parseInt(m[1], 10)
        min = m[2] != null ? parseInt(m[2], 10) : 0
      } else if (/^\d{3,4}$/.test(s)) {
        // Bare digits like "800" or "0800".
        const digits = s.padStart(4, '0')
        h = parseInt(digits.slice(0, 2), 10)
        min = parseInt(digits.slice(2), 10)
      } else {
        return null
      }
      if (ampm === 'pm' && h < 12) h += 12
      if (ampm === 'am' && h === 12) h = 0
      if (h < 0 || h > 23 || min < 0 || min > 59) return null
      return { type: 'daily', at: `${String(h).padStart(2, '0')}:${String(min).padStart(2, '0')}` }
    }

    let trigger: PitcherTrigger = { type: 'on-open' }
    const t = args.trigger
    if (t && typeof t === 'object') {
      const rec = t as Record<string, unknown>
      if (rec.type === 'daily') trigger = toDaily(rec.at) ?? { type: 'on-open' }
      else if (rec.type === 'on-open') trigger = { type: 'on-open' }
      else trigger = toDaily(rec.at) ?? { type: 'on-open' }
    } else {
      trigger = toDaily(args.at) ?? { type: 'on-open' }
    }

    // Pre-select the proposed tools, but keep only ones that really exist so a
    // made-up name can't slip through; the user reviews the checked list before
    // saving.
    const available = new Set(this.mcp.getOpenAiTools().map((tool) => tool.function.name))
    const rawTools = Array.isArray(args.allowedTools) ? args.allowedTools : []
    const allowedTools = [
      ...new Set(
        rawTools
          .filter((x): x is string => typeof x === 'string')
          .map((x) => x.trim())
          .filter((x) => available.has(x))
      )
    ]

    return { name, prompt, trigger, output, allowedTools }
  }

  private async executeCall(
    call: ChatCompletionMessageToolCall,
    messages: ChatCompletionMessageParam[],
    emit: (event: AgentEvent) => void,
    approve: ApproveFn,
    askNapkin?: AskNapkinFn,
    proposePitcher?: ProposePitcherFn
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

    // `show_napkin`: display a rich artifact in the side panel. Handled in
    // process (no MCP server, no approval); the model gets a short ack so it
    // keeps replying in text alongside the visual.
    if (qualified === SHOW_NAPKIN_TOOL) {
      const napkin = this.normalizeNapkin(args)
      if (napkin) {
        emit({ type: 'napkin_show', napkin })
        messages.push({
          role: 'tool',
          tool_call_id: call.id,
          content: `Displayed "${napkin.title}" (${napkin.kind}) on the napkin panel for the user.`
        })
      } else {
        messages.push({
          role: 'tool',
          tool_call_id: call.id,
          content:
            'Napkin not shown: provide non-empty content and a valid kind ' +
            '(code, markdown, mermaid, svg, image).'
        })
      }
      return
    }

    // `ask_napkin`: block on a multiple-choice clarification. Without an
    // interactive handler (non-interactive caller), tell the model to ask in
    // text instead of hanging.
    if (qualified === ASK_NAPKIN_TOOL) {
      const { title, prompt, choices } = this.normalizeAsk(args)
      if (!askNapkin || choices.length === 0) {
        messages.push({
          role: 'tool',
          tool_call_id: call.id,
          content:
            'Could not present a choice prompt; ask the user directly in your text reply instead.'
        })
        return
      }
      const chosen = await askNapkin({ title, prompt, choices })
      const picked = choices.find((c) => c.id === chosen)
      messages.push({
        role: 'tool',
        tool_call_id: call.id,
        content: `The user selected: ${picked ? picked.label : chosen}. Continue based on this choice.`
      })
      return
    }

    // `create_pitcher`: propose a scheduled task the user reviews and confirms.
    // Handled in-process; blocks on the user's decision. Absent the interactive
    // callback (i.e. during a pour) the tool isn't even exposed, but guard
    // anyway so a stray call is refused rather than silently creating a task.
    if (qualified === CREATE_PITCHER_TOOL) {
      const draft = this.normalizePitcherDraft(args)
      if (!proposePitcher || !draft) {
        messages.push({
          role: 'tool',
          tool_call_id: call.id,
          content: !proposePitcher
            ? 'Scheduling is not available here; tell the user to create the task from the ' +
              'Pitchers panel instead.'
            : 'Could not draft the task: provide a non-empty prompt and a valid trigger ' +
              '(daily with an HH:MM time, or on-open).'
        })
        return
      }
      const result = await proposePitcher(draft)
      messages.push({
        role: 'tool',
        tool_call_id: call.id,
        content: result.saved
          ? `The user reviewed and saved the scheduled task "${result.name}"; it will run on ` +
            `its trigger. Briefly confirm this to the user.`
          : 'The user cancelled the scheduled task, so nothing was created. Do not retry unless ' +
            'they ask again.'
      })
      return
    }

    const [serverId, ...rest] = qualified.split('__')
    const toolLabel = rest.join('__') || qualified

    // Gate on user approval BEFORE announcing/executing the call. A denied tool
    // gets a synthetic result so the model can react rather than hang. A denial
    // may carry a reason (e.g. a pour blocking a tool not in its allow-list) so
    // the transcript and the model see *why*, not a generic "denied by user".
    const decision = await approve({ server: serverId, tool: toolLabel, qualified, args })
    const allowed = typeof decision === 'boolean' ? decision : decision.allowed
    const denyReason = typeof decision === 'boolean' ? undefined : decision.reason
    if (!allowed) {
      emit({
        type: 'tool_result',
        server: serverId,
        tool: toolLabel,
        ok: false,
        preview: denyReason ?? 'Denied by user'
      })
      messages.push({
        role: 'tool',
        tool_call_id: call.id,
        content: denyReason
          ? `${denyReason} Do not retry it; continue without it and tell the user how to enable it if they need it.`
          : 'The user denied permission to run this tool. Do not retry it; continue without it.'
      })
      return
    }

    emit({ type: 'tool_call', server: serverId, tool: toolLabel, args })

    let resultText: string
    let resultImages: { data: string; mimeType: string }[] = []
    let ok = true
    try {
      const result = await this.mcp.callTool(qualified, args)
      resultText = result.text
      resultImages = result.images
    } catch (err) {
      ok = false
      resultText = `Tool call failed: ${err instanceof Error ? err.message : String(err)}`
    }

    // A failed image-generation call (whether it threw or returned an MCP error
    // result) over a missing model gets a decisive recovery instruction instead
    // of the opaque error, so the model lists/retries or stops cleanly rather
    // than ruminating. No-op for successful calls and non-image failures.
    const recovery = imageModelRecoveryHint(toolLabel, resultText)
    if (recovery) resultText = recovery

    // Surface any image the tool produced (e.g. lemonade_generate_image) on the
    // napkin panel so the user actually sees it , the bytes are kept out of the
    // model's context, so without this they'd be lost and the model would only
    // be able to describe the result.
    if (ok && resultImages.length > 0) {
      resultImages.forEach((img, i) => {
        const napkin: Napkin = {
          title: resultImages.length > 1 ? `${toolLabel} (image ${i + 1})` : toolLabel,
          kind: 'image',
          content: img.data,
          mimeType: img.mimeType
        }
        emit({ type: 'napkin_show', napkin })
      })
    }

    // If a filesystem tool succeeds, try to extract the folder path and show it
    if (ok) {
      const folderPath = extractFolderPathFromToolResult(toolLabel, resultText)
      if (folderPath) {
        const napkin: Napkin = {
          title: `Files saved to ${folderPath.split(/[\\\\/]/).pop()}`,
          kind: 'markdown',
          content: `📂 **Saved to:** ${folderPath}\n\nClick the folder icon in the napkin header to open this location in explorer.`,
          folderPath
        }
        emit({ type: 'napkin_show', napkin })
      }
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
