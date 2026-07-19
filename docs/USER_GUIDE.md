# Lemonade Stand , User Guide

Lemonade Stand is a desktop agent that runs a local model (served by `lemond`)
and lets it call **tools** provided by **MCP servers**. You stock those tools
from the **Pantry** (the Market), then just talk to the agent in plain English and
it decides when to reach for a tool. Before any tool runs, you'll see an
**approval card** (Allow once / Always allow / Deny) unless you've turned
approvals off.

This guide lists every tool that ships in the Pantry, how to enable it, and
copy‑paste example prompts you can use to test each one.

> **Tip:** After enabling a server in the Pantry (or editing
> `config/servers.json`), restart the app so the agent picks up the new tools.
> Then ask *"What tools do you have?"* to confirm they loaded.

---

## How to enable a tool

1. Open the **Pantry** in the app.
2. Pick a server from the Market and click to stock it.
3. If it asks for a **path** (a folder or a file), choose one.
4. Restart if prompted.

Under the hood this adds an entry to `config/servers.json`. You can also
hand‑edit that file. **Only enable servers you trust**, their tools run with
this app's privileges.

---

### 1. Lemonade Gateway 🍋 (`lemonade`)

Your local `lemond`'s own tools, exposed as agent tools: chat, "omni"
multimodal, image generation, audio transcription, and model listing. No install
needed , just point the entry's `url` at your server's `/mcp` endpoint and enable
it.

**Tools:** `lemonade_list_models`, `lemonade_chat`, `lemonade_omni`,
`lemonade_generate_image`, `lemonade_transcribe_audio`

**Try these prompts:**

- "Use the Lemonade tools to **list the models** my server has available."
- "Ask `lemonade_chat` to write a two‑line poem about lemonade, then show me the reply."
- "**Generate an image** of a sunny lemonade stand on a beach and save it."
- "**Transcribe** the audio file `C:\temp\lemonade-stand\note.wav` for me."
- "Use `lemonade_omni` to describe what's in `C:\temp\lemonade-stand\photo.png`."

> Note: the `/mcp` gateway is HTTP‑only and non‑streaming. TTS (spoken replies)
> and embeddings are **not** gateway tools, the app calls those REST endpoints
> directly.

---

### 2. Filesystem 📁 (`filesystem`)

Read and write files under a single root folder you choose. Great for letting the
agent work with local documents and code.

**Tools:** `read_file`, `read_multiple_files`, `write_file`, `edit_file`,
`create_directory`, `list_directory`, `directory_tree`, `move_file`,
`search_files`, `get_file_info`, `list_allowed_directories`

**Try these prompts:**

- "**List the files** in my allowed folder."
- "Create a file called `hello.txt` that says *Fresh lemonade, 50 cents*."
- "**Read** `hello.txt` back to me."
- "Make a folder called `notes` and put a `todo.md` inside it with three bullet points."
- "**Search** my folder for any file containing the word *recipe*."
- "Show me a **directory tree** of my allowed folder."
- "Append a line to `hello.txt` that says *Now with extra pulp!*"

---

### 3. Web Fetch 🌐 (`fetch`)

Fetches a URL and converts the page to clean markdown so the model can read it.
Requires `uvx` (from the [uv](https://github.com/astral-sh/uv) toolchain) on your
PATH.

**Tools:** `fetch`

**Try these prompts:**

- "**Fetch** https://example.com and summarize the page in two sentences."
- "Grab the content of https://modelcontextprotocol.io and tell me what MCP is."
- "Read https://news.ycombinator.com and list the top 3 headlines."
- "Fetch the Lemonade project page at https://github.com/lemonade-sdk/lemonade and tell me what it does."

---

### 4. Git 🔀 (`git`)

Inspect and operate on a local Git repository , status, diffs, log, and commits.
Requires `uvx`. When enabling, pick the **repository folder**.

**Tools:** `git_status`, `git_diff_unstaged`, `git_diff_staged`, `git_diff`,
`git_log`, `git_show`, `git_add`, `git_commit`, `git_reset`, `git_create_branch`,
`git_checkout`, and more.

**Try these prompts:**

- "What's the **git status** of my repo?"
- "Show me the **last 5 commits** with their messages."
- "**Diff** my unstaged changes and summarize what changed."
- "What files are currently modified but not staged?"
- "Show me the details of the most recent commit."

> The commit/reset/checkout tools change your repo. The approval card lets you
> review each action before it runs , deny anything you're unsure about.

---

### 5. Memory 🧠 (`memory`)

A persistent knowledge‑graph memory the agent can write to and recall across
turns (and across restarts). Good for remembering facts about you or a project.
Requires `npx`.

**Tools:** `create_entities`, `create_relations`, `add_observations`,
`delete_entities`, `read_graph`, `search_nodes`, `open_nodes`

**Try these prompts:**

- "**Remember** that my favorite lemonade recipe uses 4 lemons, 1 cup sugar, and a pinch of salt."
- "Remember that my lemonade stand is named *The Sour Spot* and opens on Saturdays."
- "What do you **remember** about my lemonade recipe?"
- "Search your memory for anything about *The Sour Spot*."
- "Show me everything in your memory graph."

*(Restart the app, then ask again to confirm the memory persisted.)*

---

### 6. Time ⏰ (`time`)

Current time plus timezone conversions. Requires `uvx`.

**Tools:** `get_current_time`, `convert_time`

**Try these prompts:**

- "What **time** is it right now in New York?"
- "What's the current time in Tokyo?"
- "If it's **3 PM in London**, what time is it in Los Angeles?"
- "Convert 9:00 AM Sydney time to UTC."

---

### 7. SQLite 🗃️ (`sqlite`)

Query and modify a local SQLite database file. When enabling, pick the **`.db`
file**. Requires `uvx`.

**Tools:** `read_query`, `write_query`, `create_table`, `list_tables`,
`describe_table`, `append_insight`

**Try these prompts:**

- "**Create a table** called `sales` with columns id, item, and price."
- "Insert a few rows into `sales`: lemonade $2, cookie $1, iced tea $3."
- "**List the tables** in my database."
- "Describe the `sales` table's columns."
- "**Query** the total revenue from the `sales` table."
- "Show me the three most expensive items in `sales`."

---

## Pitchers
(scheduled tasks) run prompts fresh on a timer or when the app opens
  - **On-open trigger**: runs once each app launch
  - **Daily trigger**: runs at a local HH:MM each day (with automatic catch-up on launch if the time was missed while the app was closed)
  - **Per-task tool whitelist**: only explicitly-allowed MCP tools can auto-run during a pour; everything else is denied so scheduled tasks can't be steered into unintended actions
  - **Bounded retry**: if a local model flakily fails to call its tools, a pour retries once
  - **Dual output**: serve poured results to the **Napkin** panel (rich artifacts) or save as a **conversation** in history
  - **Desktop notifications** when results are ready and the window isn't focused
  - **Managed from a new Pitchers panel**: create, edit, enable/disable, manually "Pour now", and see last-run status + errors
  - **Graceful serialization**: pours never collide with interactive chat turns; they queue and run when the user isn't typing

## Quick test checklist

| Tool | One‑line test prompt | Needs a path? |
|------|----------------------|---------------|
| Lemonade Gateway | "List the models my server has." | No (needs `/mcp` URL) |
| Filesystem | "List the files in my allowed folder." | Folder |
| Web Fetch | "Fetch example.com and summarize it." | No |
| Git | "What's the git status of my repo?" | Repo folder |
| Memory | "Remember my recipe uses 4 lemons." | No |
| Time | "What time is it in Tokyo?" | No |
| SQLite | "List the tables in my database." | `.db` file |

---

## Troubleshooting

- **"I don't have that tool."** The server isn't enabled, or the app wasn't
  restarted after enabling it. Check `config/servers.json` and restart.
- **A tool never runs.** You may have denied its approval card, or a
  session‑scoped *Always allow* was cleared on restart. Try again and choose
  *Allow once*.
- **`npx` / `uvx` errors.** Node's `npx` ships with Node.js; `uvx` comes from
  [uv](https://github.com/astral-sh/uv). Make sure the relevant one is installed
  and on your PATH, then restart the app.
- **Nothing happens / server won't start.** The first run of an `npx`/`uvx`
  server downloads it, which can take a moment. Check the app logs for errors.
- **Confirm what's loaded.** Ask the agent *"What tools do you have?"* at any
  time.

