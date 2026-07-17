# Lemonade Stand — Project Context & Handoff

> **Purpose of this file.** This is a self-contained context document so work on
> Lemonade Stand can continue in a fresh GitHub Copilot chat *from this repo*.
> It captures the background decisions, the architecture, a file-by-file map of
> what exists, how to run it, and the open next steps. If you're an AI assistant
> reading this: this is your primer — start here.

---

## 1. What this project is

**Lemonade Stand** is a desktop **MCP client / agent host**. It drives local
models served by [Lemonade](https://github.com/lemonade-sdk/lemonade) (the
`lemond` server) over its **OpenAI-compatible HTTP API**, and lets that local
LLM call tools exposed by **MCP servers** (both stdio subprocesses and
Streamable-HTTP servers).

In one sentence: *it's a local, on-device agent that uses Lemonade as its model
engine and MCP servers as its tools.*

### Relationship to Lemonade (important)

- This is a **separate, greenfield project** — **not a fork** of Lemonade.
- It depends only on Lemonade's **public HTTP API**. It never imports Lemonade's
  source or reaches into its internals.
- That keeps it cleanly **upstreamable later**: if the Lemonade maintainers want
  it, it can be contributed as a companion repo/component without entanglement.
  Their contributing rule is "open an Issue before major PRs."
- Lemonade is a **runtime dependency**: the app assumes a `lemond` is already
  running and connects to it via base URL (+ optional API key). (This was
  "option 1" in the planning discussion; "option 2" was bundling/embedding
  `lemond` via its `embeddable/` target — not chosen, but still possible later.)

### Why these technology choices

The stack was chosen to satisfy three requirements: **TypeScript**, a
**desktop UI**, and **connecting to a running `lemond`**.

- **Electron** (not Tauri): the official MCP client SDK is a Node library, and
  most MCP servers ship as **stdio subprocesses** (via `npx`/`uvx`) that need
  Node's `child_process` to spawn. Tauri's frontend is a plain webview with no
  Node runtime, which would force rewriting the MCP client in Rust. Electron
  keeps *everything* in TypeScript:
  - **Main process (Node)** runs the MCP client, the agent loop, and the
    `lemond` HTTP client.
  - **Renderer (React)** is the chat UI, sandboxed, talking to main over IPC.
- **React 19 + Vite** via `electron-vite` for the build.
- **`openai` SDK** pointed at `lemond` (Lemonade implements the OpenAI-compatible
  surface under `/api/v1`).
- **`@modelcontextprotocol/sdk`** for MCP client transport + protocol.

---

## 2. How Lemonade Stand talks to `lemond`

Lemonade exposes a broad HTTP surface; this app uses these parts:

- **Chat** — `POST /api/v1/chat/completions` (via the `openai` SDK) drives the
  agent's tool-calling loop.
- **TTS** — `POST /v1/audio/speech` (Kokoro backend) for spoken replies.
- **(Available, not yet used)** transcription (`/v1/audio/transcriptions`),
  images (`/v1/images/*`), embeddings (`/v1/embeddings`), reranking, model
  management (`models`, `pull`, `load`, `unload`), system info.

### The `lemond` `/mcp` gateway is *also* a tool source

Lemonade itself exposes an MCP **server** at `POST /mcp` (tools:
`lemonade_chat`, `lemonade_omni`, `lemonade_generate_image`,
`lemonade_transcribe_audio`, `lemonade_list_models`). Because Lemonade Stand is
an MCP **client**, it can treat `lemond`'s `/mcp` as **just another MCP server**
in `config/servers.json` — so Lemonade's own capabilities show up as tools
alongside third-party ones, through the same `tools/list` / `tools/call` path.

Caveats to remember about `/mcp` (from Lemonade's own docs):
- It's **HTTP-only** and **non-streaming** (`stream=true` is ignored). For
  streamed tokens, call `/v1/chat/completions` directly (which is what the agent
  loop does anyway).
- **TTS/embeddings are not exposed as standalone MCP tools** — use the REST
  endpoints. That's exactly why spoken replies use `/v1/audio/speech` directly
  rather than going through `/mcp`.
- If `lemond` runs with `LEMONADE_API_KEY`, every route (REST **and** `/mcp`)
  requires `Authorization: Bearer <key>`.

---

## 3. Architecture

```
┌──────────────────────── Electron ────────────────────────┐
│                                                           │
│  Renderer (React, sandboxed)      Main process (Node)     │
│  ┌───────────────────┐            ┌────────────────────┐  │
│  │ App.tsx chat UI   │            │ index.ts           │  │
│  │  - transcript     │  IPC       │  - IPC handlers    │  │
│  │  - approval cards │◀──────────▶│  - approval registry│ │
│  │  - 🔊 TTS toggle  │  (preload  │  - TTS wrapper     │  │
│  │  - audio playback │   bridge)  │                    │  │
│  └───────────────────┘            │  Agent (loop.ts)   │  │
│                                   │   ├─ LemonadeClient │──┼─HTTP─▶ lemond /api/v1
│                                   │   │   (chat + speak)│  │       (+ /v1/audio/speech)
│                                   │   └─ McpManager     │──┼─stdio/HTTP─▶ MCP servers
│                                   │       (tools/*)     │  │
│                                   └────────────────────┘  │
└───────────────────────────────────────────────────────────┘
```

### The agent loop (bounded)

Per user turn (`src/main/agent/loop.ts`):
1. Call `lemond` for a chat completion with the merged MCP tool catalogue.
2. If the model returns `tool_calls`: for each call, **ask the user to approve**
   (unless disabled or session-allow-listed), execute via `McpManager`, append
   the result as a `role:"tool"` message, and loop.
3. If no tool calls: emit the assistant text (and, if TTS is on, speak it) and
   finish.
4. `AGENT_MAX_STEPS` caps iterations so a bad model or flaky server can't spin
   forever.

### Two features layered on top

- **Per-tool approval.** Before any tool runs, main emits a
  `tool_approval_request`; the renderer shows Allow once / Always allow / Deny.
  "Always allow" adds the tool to a **session-scoped** allow-list (`sessionAllow`)
  that is cleared on restart (a grant never outlives the process). A denial
  feeds the model a synthetic "user denied, don't retry" tool result rather than
  hanging. Toggle via `AGENT_REQUIRE_APPROVAL` (default on).
- **Spoken replies (TTS).** Final assistant turns are synthesized via
  `/v1/audio/speech` and played in the renderer from a `blob:` URL.
  Fire-and-forget so a TTS failure only logs and never blocks the loop. Toggle
  live via the 🔊 button or `LEMONADE_TTS_ENABLED`.

---

## 4. File-by-file map

### Build / config
- `package.json` — scripts (`dev`, `build`, `typecheck`), deps.
- `electron.vite.config.ts` — three build targets: main, preload, renderer.
- `tsconfig.json` / `tsconfig.node.json` / `tsconfig.web.json` — project refs;
  node config covers main+preload+shared, web config covers renderer+shared.
- `.env.example` — all runtime settings (copy to `.env`).
- `config/servers.json` — MCP servers to connect to (all disabled by default).
- `.gitignore` — excludes `.env`, `node_modules/`, `out/`, and
  `config/servers.local.json`.

### Shared
- `src/shared/types.ts` — types used by **both** processes: `ChatMessage`,
  `AgentTool`, `McpServerConfig`, `AgentEvent` (the streamed event union),
  `ApprovalDecision`, and the `RendererApi` (`window.api`) contract.

### Main process (Node)
- `src/main/index.ts` — Electron entry. Creates the window, wires IPC, owns the
  **approval registry** (`pendingApprovals`), the **session allow-list**
  (`sessionAllow`), the **TTS toggle** (`speakEnabled`), and the emit-wrapper
  that triggers speech on assistant text.
- `src/main/config.ts` — dependency-free `.env` loader (real env wins over
  `.env`) + `config/servers.json` reader. Produces `AppConfig`.
- `src/main/lemonade/client.ts` — `LemonadeClient`: `chat()` (tools passthrough)
  and `speak()` (`/v1/audio/speech`, returns base64 audio).
- `src/main/mcp/manager.ts` — `McpManager`: connects one MCP `Client` per
  enabled server (stdio or Streamable HTTP), aggregates tools into a flat
  catalogue with names namespaced as `<serverId>__<toolName>` (sanitized to
  `^[a-zA-Z0-9_-]{1,64}$`), and routes `callTool` back to the owning server.
  Flattens MCP content blocks to text (image/audio summarized, not inlined).
- `src/main/agent/loop.ts` — `Agent` + `ApproveFn`. The bounded tool-calling
  loop described above.

### Preload
- `src/preload/index.ts` — `contextBridge` exposing the minimal `window.api`:
  `sendMessage`, `listTools`, `onAgentEvent`, `respondApproval`, `setSpeak`,
  `getSpeak`.
- `src/preload/api.d.ts` — declares `window.api` for the renderer's TS.

### Renderer (React)
- `src/renderer/index.html` — root HTML + CSP (allows `media-src blob:` for TTS).
- `src/renderer/main.tsx` — React root.
- `src/renderer/App.tsx` — chat UI: transcript (chat + tool trace), approval
  cards, 🔊 toggle, base64→`blob:` audio playback.
- `src/renderer/styles.css` — dark theme.

---

## 5. Configuration reference

### `.env` (copy from `.env.example`)
| Var | Default | Meaning |
|-----|---------|---------|
| `LEMONADE_BASE_URL` | `http://localhost:8000/api/v1` | `lemond` OpenAI-compatible base (include `/api/v1`). |
| `LEMONADE_API_KEY` | *(empty)* | Only if `lemond` was started with a key. |
| `LEMONADE_MODEL` | `Qwen3-1.7B-GGUF` | Chat model; must exist in your server's registry. |
| `AGENT_MAX_STEPS` | `8` | Max tool-calling iterations per turn. |
| `AGENT_REQUIRE_APPROVAL` | `true` | Prompt before each tool call. |
| `LEMONADE_TTS_ENABLED` | `false` | Spoken replies on/off (also live-toggleable). |
| `LEMONADE_TTS_MODEL` | `Kokoro-82M` | TTS model. **Verify against the real registry.** |
| `LEMONADE_TTS_VOICE` | `af_sky` | TTS voice. **Verify against the real registry.** |
| `LEMONADE_TTS_FORMAT` | `mp3` | `mp3` or `wav`. |

### `config/servers.json`
Array of MCP servers. Two transport shapes:
- **stdio**: `{ id, transport: "stdio", command, args?, env?, enabled }`
- **http**: `{ id, transport: "http", url, headers?, enabled }`

All entries are **disabled by default**. Enable only trusted servers — their
tools run with this app's privileges. To use Lemonade's own tools, enable the
`lemonade` entry and point its `url` at your server's `/mcp`.

---

## 6. How to run

```powershell
cd C:\my-files\dev\lemonade-stand
npm install                 # already done once
Copy-Item .env.example .env # then edit to match your lemond
npm run dev                 # launch with hot reload
```

Other scripts:
```powershell
npm run typecheck   # tsc for main+preload and renderer
npm run build       # electron-vite production build into out/
```

**Prereqs:** Node 20+ (developed on 24), and a running `lemond` reachable at
`LEMONADE_BASE_URL`.

---

## 7. Current status

- ✅ Scaffold builds and type-checks cleanly (all three targets).
- ✅ Agent loop with MCP tool calling (stdio + HTTP transports).
- ✅ Per-tool approval prompt (Allow once / Always allow / Deny; session
  allow-list).
- ✅ Spoken replies via `/v1/audio/speech` with live 🔊 toggle.
- ⚠️ **Not yet runtime-tested against a live `lemond`** — only compiled. Needs a
  smoke test with a real server.
- ⚠️ `npm install` reported a few transitive-dependency audit warnings (typical
  of the Electron toolchain). Worth reviewing before publishing.

---

## 8. Known gaps / open next steps

Prioritized backlog for the next session:

1. **Verify TTS defaults.** Confirm `Kokoro-82M` / `af_sky` (and the
   `/v1/audio/speech` request shape) against Lemonade's actual model registry
   and TTS API. Adjust defaults in `src/main/config.ts` + `.env.example` if they
   don't match a real entry. *(This was flagged but not yet done.)*
2. **Live smoke test.** Run against a real `lemond`: a plain chat turn, a
   tool-calling turn (enable an MCP server), an approval Allow/Deny, and a TTS
   playback. Fix whatever breaks.
3. **Streaming chat.** The loop currently uses non-streamed completions. Add
   token streaming (`stream=true` on `/v1/chat/completions`) with incremental
   `assistant_text` events for a snappier UI.
4. **Media tools via REST.** MCP image/audio tool *results* are currently
   summarized as text. For real media, add direct calls to `/v1/images/*` and
   render them inline.
5. **Model picker.** Fetch `/api/v1/models` and let the user choose the chat
   model in-app instead of only via `.env`.
6. **Persist history / sessions.** Currently in-memory only.
7. **Settings UI.** Surface `config/servers.json` editing and `.env` values in
   the app (respecting the secrets-stay-untracked rule).
8. **Dependency audit.** Review the `npm audit` findings.
9. **Packaging.** Add `electron-builder` (or similar) to produce installers.

---

## 9. Conventions & guardrails (please preserve)

- **Don't turn this into a fork of Lemonade.** Depend only on `lemond`'s public
  HTTP API. If you need something Lemonade doesn't expose, prefer contributing
  that endpoint upstream over reaching into internals.
- **Keep the preload surface minimal.** The renderer is sandboxed
  (`contextIsolation: true`, `nodeIntegration: false`); everything crosses
  through the small `window.api` contract in `src/shared/types.ts`.
- **Secrets never get committed.** `LEMONADE_API_KEY` and per-server headers
  live in `.env` / untracked config only. `.gitignore` enforces this.
- **Trust boundary.** MCP tool descriptions and results are third-party text
  flowing into the model — a prompt-injection surface. The approval gate exists
  because of this; keep it on by default.
- **The agent loop must stay bounded** (`AGENT_MAX_STEPS`) so it can't run away.
- **TTS is fire-and-forget** — it must never block or fail the chat loop.

---

## 10. Backstory (how we got here, condensed)

The idea began as "add MCP **client** capability to Lemonade." Lemonade is
already an MCP **server** (`/mcp` gateway) and already has an internal
tool-calling loop (`collection_orchestrator.cpp`) for its Omni collections. The
options considered were:

- **(A) In-process in `lemond`** — reuse the existing orchestrator loop; smallest
  code delta but grows the C++ server's trust/lifecycle/dependency surface.
- **(B) Separate app** — cleaner isolation; must reimplement the loop, but that's
  an acceptable trade.

The decision was a **fully separate project** (B), because the goal is a
standalone agent experience. A fork was explicitly rejected: this app needs
Lemonade's *behavior* (an HTTP service), not its *code*, so it depends on the
API instead of inheriting the whole C++/CMake tree. This keeps future
upstreaming clean and avoids a permanent rebase tax.

The stack (Electron + React + TS) follows directly from wanting a TS desktop app
that can spawn stdio MCP servers and connect to a running `lemond`.
