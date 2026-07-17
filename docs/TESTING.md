# Testing Lemonade Stand

This guide covers how to run the automated tests locally. Tests run on
[Vitest](https://vitest.dev) and are split into two tiers:

- **Core tests** — deterministic, network-light, and fast. They cover config
  validation, the Pantry catalog, and a real filesystem-MCP integration test.
  These run on every push and pull request to `main`.
- **Optional tests** — integration tests for the `uvx`-based Pantry servers
  (time, sqlite, git, fetch). They need the [`uv`](https://github.com/astral-sh/uv)
  toolchain and network access, so they're opt-in.

> **What isn't tested:** the local LLM deciding *when* to call a tool. That needs
> a running `lemond`, a model, and a GPU, and its output is non-deterministic —
> so it's out of scope for automated tests. The tests exercise the tool
> **plumbing** (`McpManager` → connect → `listTools` → `callTool`), which is the
> code path every tool call travels regardless of the model.

## Prerequisites

- Node.js 20+ (same as the app).
- Dependencies installed: `npm install`.
- For the **optional** tests only: the `uv`/`uvx` toolchain on your PATH
  (see [Optional tests](#optional-uvx-based-server-tests)).

## Running the core tests

```powershell
npm test
```

This runs the full suite once and exits. On a machine without `uvx`, the
optional tests are automatically **skipped** (you'll see them marked as skipped
in the output), so `npm test` is always safe to run.

Expected output looks like:

```
 ✓ test/config.test.ts (10)
 ✓ test/mcp-filesystem.integration.test.ts (7)
 ↓ test/mcp-optional.integration.test.ts (4) [skipped]

 Test Files  2 passed | 1 skipped (3)
      Tests  17 passed | 4 skipped (21)
```

### Watch mode

For fast feedback while editing, run the watcher — it re-runs affected tests on
save:

```powershell
npm run test:watch
```

## Optional (`uvx`-based) server tests

These verify the **time**, **sqlite**, **git**, and **fetch** Pantry servers
end-to-end. They're skipped unless you opt in, because they require `uvx` and,
on first run, download the server packages (and `fetch` needs network access).

### 1. Install `uv` (provides `uvx`)

```powershell
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

See the [uv install docs](https://docs.astral.sh/uv/getting-started/installation/)
for macOS/Linux and alternatives. Confirm it's on your PATH:

```powershell
uvx --version
```

### 2. Run them

```powershell
npm run test:optional
```

This sets `RUN_UVX_TESTS=1` (via `cross-env`), which flips the optional suite
from skipped to active. The `git` test also shells out to `git`, so make sure
that's installed too.

## What each test file covers

| File | Tier | Covers |
|------|------|--------|
| [`test/config.test.ts`](../test/config.test.ts) | Core | `catalog.json` / `servers.json` shape, unique ids, `{{path}}` handling, and the Pantry config helpers (`serverFromCatalog`, `pathForServer`, `withServerPath`). |
| [`test/mcp-filesystem.integration.test.ts`](../test/mcp-filesystem.integration.test.ts) | Core | Boots the real filesystem MCP server against a temp folder and drives `write_file` / `read_file` / `list_directory` through the app's `McpManager`, plus error handling. |
| [`test/mcp-optional.integration.test.ts`](../test/mcp-optional.integration.test.ts) | Optional | Same integration approach for the `uvx` servers: time, sqlite, git, and fetch. Skipped unless `RUN_UVX_TESTS` is set. |

## How tests run in CI

The [`ci.yml`](../.github/workflows/ci.yml) workflow runs on every push and pull
request to `main`:

- **`test`** — type-check + core tests. This is the job to make a **required
  status check** in branch protection so merges are gated on green tests.
- **`test-optional`** — installs `uv` and runs the optional suite. It's marked
  `continue-on-error`, so a flaky download or upstream change never blocks a
  merge; review its result but don't gate on it.

## Troubleshooting

- **Optional tests all skip.** That's expected without `uvx`. Install `uv` and
  use `npm run test:optional`, not `npm test`.
- **First optional run is slow or times out.** The initial `uvx` invocation
  downloads each server package. The suite allows a generous timeout; re-run
  once the packages are cached.
- **Filesystem test fails to connect.** It uses `npx` to fetch
  `@modelcontextprotocol/server-filesystem` on first run — make sure you have
  network access and `npx` works (`npx --version`).
- **`fetch` test fails.** It reaches `https://example.com`; a blocked or offline
  network will fail it. This is why it's in the optional, non-blocking tier.
