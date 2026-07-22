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

`--check` performs a native-execution-disabled workspace bootstrap, renders one
100x30 frame without a TTY, closes the workspace, and exits nonzero if required
storage cannot open. Bootstrap can ensure LEM directories, open DuckDB
read-write, apply migrations, and recover interrupted workspace records. It
does not construct the native agent engine, start Soft Serve, discover or launch
providers, mutate private/source Git, or admit or change native queue work.

## Primary panels

| Panel | Behaviour |
| --- | --- |
| **Chat** | Durable multi-session transcript, Bubbles viewport and composer, streamed thought/answer channels, cached Glamour Markdown, per-session draft/scroll state, tool receipts, and hidden-session attention. |
| **Work** | Durable Work CRUD, native provider dispatch, queue controls, ordered process output, blocked questions, retry/resume, and reviewed acceptance through the connected `go/agent/*` engine. Future actions remain visible with an exact unavailable reason. |
| **Models** | Asynchronous discovery from the Hugging Face cache and `LEM_MODELS_DIR`, fuzzy filtering, explicit loading, and loaded-model detail. A swap is refused while any chat job is active and drains the HTTP listener before replacing an idle model. |
| **Service** | Start or stop the API over the exact serial model lane used by Chat; inspect address, model, request count, client URLs, and listener errors. Stopping the service never closes the model. |
| **Data** | Review surface for the `lem data` training-data loop (`~/.lem/datasets.duckdb`): item list with dataset/status/kind/source/score filters, side-by-side detail (rendered content, score breakdown, lineage, welfare flags), and the review actions — approve, reject, quarantine-clear (note required), edit-as-derived (original archived), tag — each with an uppercase bulk form across the current filter behind a count confirmation. Every action has a mirrored palette entry; unavailable states render honestly. |

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

## Rendering with .ctml

Screens are migrating from hand-composed Lip Gloss to `.ctml` markup rendered
by go-html's terminal renderer (`dappco.re/go/html`). The tab strip
(`tabs.go` + `tabs.ctml`) establishes the idiom each converted screen copies:

1. **Markup file** — the screen's structure lives in a `.ctml` file embedded
   beside its Go file (`//go:embed`). Text content doubles as its own i18n
   key; `class` attributes are static strings; comments record the host
   seams the file exposes.
2. **Bindings** — dynamic rows enter at parse time through `ctml.Bindings`
   `Sequences` and `<each items="..." as="row">`, bound in text as
   `{{row.field}}`. Re-parse and re-bind on every state change — screens
   this size make that free. A per-row style variation cannot ride a class
   attribute (they are static), so the host splits rows into one sequence
   per style (see `panelBarBindings`: `tabsBefore` / `tabsActive` /
   `tabsAfter`).
3. **Theme** — `html.TermTheme.Classes` maps the markup's class tokens onto
   the existing `uiStyles` palette (`panelBarTheme`). The markup carries no
   colours of its own; the palette in `style.go` stays the single source of
   visual truth.
4. **Boxes and mouse** — render through `html.RenderTermBoxes` to receive the
   `html.BoxMap` of every id'd block, then resolve mouse coordinates with
   `teabox.Resolve` inside the app's `tea.MouseMsg` handling (`onMouse`).
   Screen cells map to frame-inner cells by subtracting `frameInsetRows` /
   `frameInsetCols` (the outer border). The renderer boxes block-level
   elements only, so a single-row strip derives its per-item boxes from the
   render itself (`mergePanelTabBoxes`) and merges them into the same map —
   teabox's smallest-box rule then prefers the item over the strip.

A left click on a tab switches panels through exactly the same path as
`Tab`/`Shift+Tab` (`selectPanel`); the wheel keeps scrolling the transcript.

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
| `a` / `r` | Data | Approve / reject the selected item |
| `c` | Data | Clear the selected item's quarantine (note required) |
| `e` | Data | Edit as a derived item (the original is archived, lineage kept) |
| `t` | Data | Tag the selected item |
| `A` / `R` / `C` / `T` | Data | The same action in bulk across the current filter, behind a count confirmation |
| `s` | Data | Toggle sort (date / score) |
| `f` | Data | Edit the dataset/status/kind/source/score filter |
| `Ctrl+K`, then `New Work` / `Edit Work` | Work | Create or edit title, task, and repository; `Ctrl+S` saves the editor |
| `Ctrl+O`, arrows, `Enter` | Work | Select and invoke an available native action in the inspector |
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

The command palette exposes session creation and switching, history search,
Markdown/structured-JSON export, panel navigation, settings save, Work creation
and editing, manual Work refresh, and every native action whose durable state
currently permits it.

## Native Work lifecycle

### Create, register, and dispatch

`New Work` and `Edit Work` persist a title, task, and repository directory in
DuckDB. Dispatch is a sequence of current, explicit reviews rather than one
blind process launch:

1. Select the provider and model. LEM discovers Codex, Claude, and OpenCode at
   startup and shows unavailable executable/version reasons without disabling
   the rest of the workspace.
2. Review the canonical source path, branch, HEAD, included files, and private
   repository name. The source must be clean and on a named branch.
3. For an ad-hoc directory, separately confirm `Enable Git`. This creates local
   `.git` metadata and a baseline commit using the displayed repository-local
   identity; it does not change global Git configuration.
4. Review the exact redacted provider command, source revision, internal
   repository, proposed run worktree, and current queue decision.
5. Confirm launch. The durable run is queued with an immutable ID. `Start queue`
   changes a frozen queue to accepting; `Stop queue` freezes it immediately or
   drains already-running work before becoming frozen.

LEM seeds a private repository without adding or rewriting a source remote.
Each attempt runs on a branch such as `lem/work/<work-id>/run-<number>` in an
internal worktree below `~/.lem/workspaces/`. Stdout and stderr are streamed in
one durable sequence. At terminal exit, LEM commits any remaining non-ignored
changes, pushes the branch, and records completed, waiting, failed, cancelled,
or interrupted state before releasing a recoverable worktree.

### Provider policy and native authority

`~/.lem/agents.yaml` is policy only; live counters and backoff state remain in
DuckDB. The versioned policy selects the default provider, validation commands,
global/provider/model concurrency, delays, quota windows and backoff, plus each
provider executable, default model, credential environment allow-list, and
extra flags. A minimal override is:

```yaml
version: 1
dispatch:
  default_agent: codex
  global_concurrency: 1
  timeout_minutes: 60
  validation:
    - command: go
      args: [test, ./...]
providers:
  codex:
    executable: codex
    default_model: gpt-5
    credential_env: [OPENAI_API_KEY]
```

Native execution has explicit host access. LEM uses an argument vector rather
than a shell command and passes only configured essential and credential
environment keys, but it is not an operating-system sandbox. A provider can
read or contact anything its host account can access. The launch review repeats
this warning; only confirm providers, flags, tasks, and repositories you trust.

### Cancel, shutdown, answer, and recover

`Cancel` stops a queued run or shuts down the running provider process group.
Normal TUI shutdown freezes admission, withdraws queued work, cancels every
owned process group including children, joins output readers, flushes log
chunks, and captures/pushes remaining Git work. If capture or push cannot be
made durable, the internal worktree is retained with a recovery receipt.

A provider can finish in `waiting` with one durable question. `Answer` stores
the response without mutating the parent run; `Resume` creates an immutable
child attempt from that answer and prior ordered output. `Retry` creates a child
attempt from a failed or cancelled run. For an interrupted run, the TUI's
`Resume` action intentionally uses the same durable retry path. Children reuse
the run branch and reconstruct a released worktree from private Git when
needed.

On startup, any queued, preparing, running, or cancelling run that could not
have a live owned process is marked `interrupted`; partial logs remain, the
queue starts frozen, and nothing reconnects or restarts automatically. The user
reviews the retained state and explicitly resumes it.

### Review, validate, accept, or reject

`Review changes` on a completed run fetches its durable branch into a disposable
integration worktree, replays the agent commit range onto the current clean
source revision, renders commits and diff, and executes the configured
validation commands there. Review never mutates the source checkout.

Conflicts or failed validation block acceptance and leave source HEAD, status,
index, and files unchanged. If no validation command is configured, the review
requires a separate acknowledgement. `Accept` requires a final confirmation,
rechecks the reviewed Git facts and clean source, then advances the source to
the exact reviewed result and records the accepted revision. `Reject` records
the reviewed decision without changing the source. A stale, superseded, or
tampered review receipt cannot authorize an apply.

## Local storage

```text
~/.lem/
├── config.yaml       # written only by an explicit settings commit
├── agents.yaml       # provider, queue, rate, and validation policy
├── lem.duckdb        # chat, Work, native runs, logs, queue, and reviews
├── state.db          # scoped drafts, viewport, and active-session state
├── soft-serve/       # private Git service data, keys, owner lock, and log
├── workspaces/       # cached private clones and per-run worktrees
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

Native orchestration uses the `agent_projects`, `agent_runs`, `agent_events`,
`agent_log_chunks`, `agent_questions`, `agent_answers`, `agent_acceptances`,
`agent_queue_state`, and `agent_provider_state` tables in `lem.duckdb`. Native
records are not copied into the chat-oriented `lem_events` table.

Soft Serve starts lazily on `127.0.0.1:23231` when registration or dispatch
needs private Git. One live LEM PID owns `soft-serve/owner.lock`; another TUI
continues to provide non-agent panels but receives a visible agent capability
reason. A stale lock is removed only after its recorded PID is no longer live.
The owning TUI stops the listener, closes Soft Serve's private metadata, and
releases the lock during shutdown. That metadata is not application state and
is never queried as DuckDB domain data.

Knowledge discovery reads Markdown below `packs/` through medium `List`,
`Stat`, and `Read` only. Stored attachments keep a content snapshot and hash,
so later source changes are marked stale without rewriting conversation
context. The current inspector is read-only and rescans knowledge/runtime state
on the next application start.

## Authority limits

This slice ports native lifecycle behaviour into focused `go/agent/*` packages;
it does not import CoreAgent. It does launch Codex, Claude, and OpenCode and can
mutate a source repository only for separately confirmed Git enablement or a
validated, explicitly accepted result. It does not create a container, run a
remote agent, contact a forge, create or merge a pull request, or download a
knowledge pack. Those future actions remain visible with their exact
unavailable reasons rather than simulated success.

LEM runtime metadata stays below `~/.lem/`. It does not create question,
answer, status, Markdown, JSON, YAML, or other LEM control files in source,
cached, execution, review, or validation repositories.

## Verification

Deterministic tests use temporary DuckDB/go-store files, clean temporary Git
repositories, fake native provider executables, fake inference models, and fake
runtime adapters. The native receipts drive Codex, Claude, and OpenCode through
dispatch, ordered output, commit, completion, validation, and acceptance; a
separate receipt kills a provider and its child, verifies interruption and log
flush, reopens the orchestrator, resumes from private Git, and accepts the
result.

```sh
go test ./tui -count=1
go test -race ./tui -count=1
go test ./tui -covermode=atomic -coverprofile=/tmp/lem-tui-cover.out -count=1
go test ./tui -run '^$' -bench '^BenchmarkMarkdownTranscript$' -benchmem -benchtime=20x
cd ../go
go test -race ./agent/work ./agent/queue ./agent/provider ./agent/gitserver ./agent/workspace ./agent/orchestrator -count=1
```

Darwin/Metal live receipts remain opt-in. Set `LTHN_PROBE_MODEL` and
`MLX_METALLIB_PATH` to run a real model through both the chat update loop and
the shared HTTP service.
