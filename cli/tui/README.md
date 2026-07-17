# LEM terminal workspace

`lem tui` is a persistent Bubble Tea workspace around `go-inference`. It keeps
agent-style chat inside one stable frame, lets conversations keep generating
while another session or panel is open, and shares one loaded model between
the TUI and the OpenAI/Anthropic/Ollama-compatible HTTP service.

The workspace opens asynchronously at `~/.lem/`. Chat is the default panel,
including when no model is loaded.

## Run it

From the nested CLI module:

```sh
go run . tui
go run . tui --model /path/to/model --context 8192 --max-tokens 4096
go run . tui --check
```

`--check` performs a real workspace bootstrap, renders one 100x30 frame without
a TTY, closes the workspace, and exits nonzero if required storage cannot open.

## Primary panels

| Panel | Behaviour |
| --- | --- |
| **Chat** | Durable multi-session transcript, Bubbles viewport and composer, streamed thought/answer channels, cached Glamour Markdown, per-session draft/scroll state, tool receipts, and hidden-session attention. |
| **Work** | Persistent local/provider work records, grouped status, detail timeline, runtime context, and the complete future-agent action catalogue. Execution actions are visibly disabled until a `go/agent/*` provider is connected. |
| **Models** | Asynchronous discovery from the Hugging Face cache and `LEM_MODELS_DIR`, fuzzy filtering, explicit loading, and loaded-model detail. A swap is refused while any chat job is active and drains the HTTP listener before replacing an idle model. |
| **Service** | Start or stop the API over the exact serial model lane used by Chat; inspect address, model, request count, client URLs, and listener errors. Stopping the service never closes the model. |

Settings, sampling modes, built-in tools, local knowledge, runtime detection,
and agent capability detail live in the contextual inspector instead of
competing for primary tabs.

## Layout

- At 120 columns or wider, the main panel and a 32-column inspector are shown
  side by side.
- From 80 to 119 columns, `Ctrl+O` opens the inspector above a compact panel
  preview.
- Below 80 columns, `Ctrl+O` makes the inspector the single content view.
- Very short terminals retain a bounded frame, panel identity, session strip,
  and footer without slicing ANSI sequences.

The transcript follows output only while already at the bottom. Manual upward
scrolling preserves the reading position and shows a `new output` marker;
`End` returns to live output.

## Keys

| Key | Scope | Action |
| --- | --- | --- |
| `Tab` / `Shift+Tab` | global | Next / previous primary panel |
| `Ctrl+N` | global | Create and open a blank session |
| `Ctrl+P` | global | Fuzzy recent-session switcher |
| `Alt+Left` / `Alt+Right` | global | Previous / next recent session |
| `Ctrl+K` | global | Searchable command palette |
| `Ctrl+F` | global | Search durable session titles and turn content |
| `Ctrl+O` | global | Toggle the inspector |
| `Ctrl+S` | global | Commit inspector settings |
| `F1` | global | Full key help |
| `Ctrl+C` | global | Cancel jobs, stop service, close resources, and quit |
| `Enter` | Chat | Send a non-empty prompt when a model is loaded |
| `Alt+Enter` | Chat | Insert a newline in the composer |
| `Esc` | Chat | Cancel the visible session's generation |
| `Ctrl+T` | Chat | Toggle explicit thinking on/off |
| `Home` / `End` | Chat | Top / resume live transcript follow |
| `PgUp` / `PgDn`, `Ctrl+U` / `Ctrl+D`, mouse wheel | Chat | Scroll transcript |
| `/` | Models or Work | Filter the focused list |
| `Enter` | Models | Load the selected model when all jobs are idle |
| `Enter` | Service | Start / stop the listener |
| arrows or `h`/`l` | Service | Change listen preset while stopped |
| arrows or `h`/`j`/`k`/`l` | inspector | Select and adjust settings/actions |
| `Esc` | overlay | Close the topmost overlay |

## Sessions, jobs, and tools

Sessions, turns, events, generation jobs, work records, artifacts, and
knowledge attachments are relational DuckDB records. Switching panels or
sessions never cancels generation. Each session may own one job; every session
and HTTP request queues through one serial scheduler around the single resident
model.

The user turn, assistant placeholder, job state, streamed deltas, final metrics,
tool calls/results, and attention transition are persisted. On restart, queued
or generating jobs and their sessions become `interrupted`, a timeline event
is recorded, partial assistant content remains, and nothing restarts
automatically. Graceful quit drains buffered deltas into the cancelled session
before DuckDB closes.

When built-in tools are enabled, declarations join the system message. A valid
call is recorded, executed locally, persisted as a tool turn/event, and fed
back through a new durable job. Automatic continuation is bounded to two hops.
Malformed and unknown calls produce explicit failure receipts.

The command palette currently exposes session creation, switching, history
search, Markdown/structured-JSON export, panel navigation, settings save, and
the capability catalogue. Session rename/archive/reopen, local Work CRUD, and
knowledge attach/detach exist at the tested domain boundary but do not yet have
interactive editors in this TUI-first slice.

## Local storage

```text
~/.lem/
├── config.yaml       # written only by an explicit settings commit
├── lem.duckdb        # durable relational workspace
├── state.db          # scoped drafts, viewport, and active-session state
├── workspaces/       # go-store workspace state
├── packs/            # local Markdown knowledge documents
└── exports/          # atomic Markdown and JSON session exports
```

File-like state is accessed through a sandboxed `dappco.re/go/io` medium rooted
at `~/.lem/`; DuckDB and go-store receive only their resolved local database
filenames. Exports use same-directory temporary files and atomic rename.

Failure to open the root or DuckDB is blocking and shows Retry/Quit without an
in-memory-history fallback. A go-store failure degrades draft/UI persistence
with a visible warning. Malformed config is preserved, defaults are used for
the run, and commits remain disabled until a successful reload.

Knowledge discovery reads Markdown below `packs/` through medium `List`,
`Stat`, and `Read` only. Stored attachments keep a content snapshot and hash,
so later source changes are marked stale without rewriting conversation
context. The current inspector is read-only and rescans knowledge/runtime state
on the next application start.

## Authority limits

This slice does not import CoreAgent, spawn an agent, create a container, mutate
a repository, contact a forge, or download knowledge packs. `go-container` is
used only for read-only capability detection. All 28 future agent actions are
present with the provider's unavailable reason rather than simulated success.

The next feature slice ports CoreAgent's proven behaviours into focused
`go/agent/*` packages behind the TUI-owned provider boundary.

## Verification

Deterministic tests use temporary DuckDB/go-store files, an in-memory file
medium, fake inference models, and fake runtime/agent adapters:

```sh
go test ./tui -count=1
go test -race ./tui -count=1
go test ./tui -covermode=atomic -coverprofile=/tmp/lem-tui-cover.out -count=1
go test ./tui -run '^$' -bench '^BenchmarkMarkdownTranscript$' -benchmem -benchtime=20x
```

Darwin/Metal live receipts remain opt-in. Set `LTHN_PROBE_MODEL` and
`MLX_METALLIB_PATH` to run a real model through both the chat update loop and
the shared HTTP service.
