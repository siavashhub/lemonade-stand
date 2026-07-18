# Lemonade Stand

![Lemonade Stand](resources/banner.svg)

A desktop **MCP client / agent host** that drives local models served by
[Lemonade](https://github.com/lemonade-sdk/lemonade) (`lemond`) over its
OpenAI-compatible HTTP API.

- **Main process (Node)** — runs the agent tool-calling loop, connects to MCP
  servers (stdio + Streamable HTTP), and talks to `lemond`.
- **Renderer (React)** — a chat UI that streams the agent's thinking and tool
  activity.
- **`lemond` is a runtime dependency**, not a fork: this app only speaks to its
  public HTTP API, so it can be upstreamed later without entangling Lemonade's
  internals.

## Architecture

```
┌──────────────── Electron ────────────────┐
│  Renderer (React)                         │
│    chat UI  ──IPC──▶  Main process (Node) │
│                          │                │
│                          ├── LemonadeClient ──HTTP──▶ lemond (/api/v1)
│                          └── McpManager  ──stdio/HTTP──▶ MCP servers
│                                (agent loop dispatches tool calls)
└───────────────────────────────────────────┘
```

The agent can also treat **lemond's own `/mcp` gateway** as just another MCP
server (see `config/servers.json`), so Lemonade's `lemonade_chat` /
`lemonade_omni` / image / transcription tools show up alongside third-party
tools. For streaming chat or standalone TTS/embeddings, call `lemond`'s REST
endpoints directly.

## Prerequisites

- Node.js 20+ (tested on 24).
- A running `lemond` server. Note its base URL and port.

## Setup

```powershell
npm install
Copy-Item .env.example .env
```

Edit `.env` so `LEMONADE_BASE_URL` matches your server (include the `/api/v1`
suffix), set `LEMONADE_MODEL` to a model your server can serve, and set
`LEMONADE_API_KEY` only if you launched `lemond` with one.

## Run

```powershell
npm run dev        # launch the app with hot reload
npm run typecheck  # type-check main + renderer
npm run build      # production build into out/
```

Hover the **Lemonade Stand** brand text to see the version — `dev` in a local
run, or the semantic version in an installed build. To use packaged installers, see the [release guide](docs/RELEASING.md).

### Lemonade Server Run
The quickest way to run a lemonade server would to be install docker and run it using below command:

```powershell
docker run -d `
  --name lemonade-server `
  -p 13305:13305 `
  -v lemonade-cache:/opt/lemonade/.cache/huggingface `
  -v lemonade-llama:/opt/lemonade/llama `
  -v lemonade-recipe:/opt/lemonade/.cache/lemonade `
  ghcr.io/lemonade-sdk/lemonade-server:latest
```
Otherwise you can download the server from Lemonade server it self: (https://github.com/lemonade-sdk/lemonade)

## Enabling tools (MCP servers)

Edit `config/servers.json`. Each entry is `stdio` (spawns a subprocess) or
`http` (Streamable HTTP). Entries are **disabled by default** — only enable
servers you trust, since their tools run with this app's privileges.

To let the agent use Lemonade's own capabilities as tools, set the `lemonade`
entry's `url` to your server's `/mcp` endpoint and flip `enabled` to `true`.

## Security notes

- MCP tool descriptions and results are third-party text that flows into the
  model. Treat enabled servers as trusted code.
- Secrets (`LEMONADE_API_KEY`, any per-server headers) live in `.env` /
  untracked config, never committed. `.gitignore` excludes `.env` and
  `config/servers.local.json`.
- The agent loop is bounded by `AGENT_MAX_STEPS` to prevent runaway tool loops.
