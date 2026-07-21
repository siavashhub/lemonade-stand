# Lemonade Stand

![Lemonade Stand](resources/banner.svg)


A desktop **MCP client / agent host** that drives local models served by
[Lemonade](https://github.com/lemonade-sdk/lemonade) (`lemond`).

- **Main process (Node)**: runs the agent tool-calling loop, connects to MCP
  servers (stdio + Streamable HTTP), and talks to `lemond`.
- **Renderer (React)**: a chat UI that streams the agent's thinking and tool
  activity.
- **`lemond` is a runtime dependency**: can be download and installed from [Lemonade](https://github.com/lemonade-sdk/lemonade) repository
  or pulled and run as a container .

## Architecture

The agent uses HTTP API and  **lemond's own `/mcp` gateway** as another MCP
server (see `config/servers.json`), so Lemonade's `lemonade_chat` /
`lemonade_omni` / image / transcription tools show up alongside third-party
tools.

## Demo

https://github.com/siavashhub/lemonade-stand/raw/main/demo/lemonade-stand-demo.mp4

## Prerequisites

- Node.js 22.12+ (tested on 22.23 and 24).
- A running `lemond` server. Note its base URL and port.

## Quick Run
Headover to the [release page](https://github.com/siavashhub/lemonade-stand/releases) section and download and run the installer.

## Dev Setup

```powershell
npm install
Copy-Item .env.example .env
```

If you switch Node versions (for example via `nvm`), run `npm install` again in
this repo so native/binary dependencies are refreshed for the active runtime.

Edit `.env` so `LEMONADE_BASE_URL` matches your server (include the `/api/v1`
suffix), set `LEMONADE_MODEL` to a model your server can serve, and set
`LEMONADE_API_KEY` only if you launched `lemond` with one.

You can also change the base URL and API key
at any time from inside the app by clicking the **server-status** pill in the
top bar.

## Dev Run

```powershell
npm run dev        # launch the app with hot reload
npm run typecheck  # type-check main + renderer
npm run build      # production build into out/
```

`npm run dev` automatically runs a pre-step that bootstraps Electron if needed.


### Lemonade Server Run
The quickest way to run a lemonade server would to be install docker and run it using below command:

```powershell
docker run -d `
  --name lemonade-server `
  -p 13305:13305 `
  -v "$env:USERPROFILE/.cache/huggingface:/opt/lemonade/.cache/huggingface" `
  -v lemonade-llama:/opt/lemonade/llama `
  -v "$env:USERPROFILE/.cache/lemonade:/opt/lemonade/.cache/lemonade" `
  ghcr.io/lemonade-sdk/lemonade-server:latest
```
Otherwise you can download the server from Lemonade server it self: (https://github.com/lemonade-sdk/lemonade)

## Troubleshooting

- `TypeError: crypto.hash is not a function`
  Use Node 22.12+ (or newer). This error appears when running with older Node
  versions.
- `Error: Electron uninstall`
  Electron binary metadata was not bootstrapped yet for your current runtime.
  Run:

```powershell
npm run predev
```

Then retry:

```powershell
npm run dev
```

## Enabling tools (MCP servers)

Edit `config/servers.json`. Each entry is `stdio` (spawns a subprocess) or
`http` (Streamable HTTP). Entries are **disabled by default**. Only enable
servers you trust, since their tools run with this app's privileges.

To let the agent use Lemonade's own capabilities as tools, set the `lemonade`
entry's `url` to your server's `/mcp` endpoint and flip `enabled` to `true`.

## Security notes

- MCP tool descriptions and results are third-party text that flows into the
  model. Treat enabled servers as trusted code.
- Secrets (`LEMONADE_API_KEY`, any per-server headers) live in `.env` /
  untracked config, never committed. `.gitignore` excludes `.env` and
  `config/servers.local.json`.