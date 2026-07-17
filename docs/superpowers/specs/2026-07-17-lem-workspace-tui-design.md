# LEM Workspace TUI Design

**Date:** 2026-07-17
**Status:** Approved for TUI-first implementation
**Scope:** Interactive workspace shell and local persistence

## Product intent

`lem tui` becomes the canonical interactive workspace for local inference and
agent-assisted work. It is a new product surface inside `go-inference`; it does
not embed, wrap, or reproduce CoreAgent's CLI. CoreAgent remains the behavioural
reference for the later `go/agent/*` port, while this slice builds stable UI
interfaces and honest unavailable states without importing CoreAgent.

The central experience is a familiar agent chat contained inside a stable,
full-terminal frame. Multiple conversations remain open as sessions, continue
working when the user changes panels, and restore after restart. Chat, work,
model management, service hosting, knowledge context, settings, and runtime
status are coordinated as parts of one workspace rather than presented as six
unrelated screens.

This specification deliberately covers the TUI-first vertical slice. It
delivers the complete visual shell, durable multi-session behaviour, local work
tracking, and useful read-side integrations. Agent dispatch, monitoring, Brain,
fleet, forge, and closeout features are represented by capability boundaries
and disabled actions until their inference-native implementations land under
`go/agent/*`.

## Goals

- Replace the current loose row of tabs with a polished, stable workspace
  frame that uses the full terminal deliberately.
- Support durable, searchable multi-session chat with safe switching while
  generation continues in another session.
- Present chat, work, models, and service hosting as coherent primary panels.
- Make settings, modes, tools, knowledge, model information, and runtime
  capabilities available in a contextual inspector.
- Persist relational data in DuckDB and reactive key/value state in
  `dAppCore/go-store`, all below `~/.lem/`.
- Use `dAppCore/config` for explicit, durable settings changes.
- Route file-like state through an injected `dAppCore/go-io` medium. The
  default is a sandboxed local medium rooted at `~/.lem/`; tests use a memory
  medium and later agent backends may supply a remote medium.
- Use `dAppCore/orm` for typed session, turn, event, work-item, attachment, and
  artifact access over DuckDB.
- Define a LEM-owned agent capability interface without importing CoreAgent.
  The future `go/agent/*` implementation must plug into this boundary.
- Discover local runtime capabilities through `dAppCore/go-container` without
  spawning a container in this slice.
- Discover and attach local Markdown knowledge-pack documents as session
  context.
- Retain the existing streaming local-model and shared HTTP-service behaviour.
- Keep the CLI module portable and keep the root inference package free of TUI,
  database, agent, and native-runtime dependencies.

## Non-goals for this slice

- Launching, cancelling, or resuming CoreAgent, Codex, Claude, or other worker
  processes.
- Automatically creating Apple, VZ, Docker, or Podman containers.
- Reusing any CoreAgent CLI code, command parsing, or presentation code.
- Importing CoreAgent Go packages into the TUI-first slice.
- Remote fleet control, forge automation, auto-PR, auto-merge, or auto-QA.
- Automatic network download or update of knowledge packs.
- Semantic/vector retrieval or shared OpenBrain writes.
- Hard deletion of sessions or their history. Sessions can be archived and
  exported.
- Loading more than one inference model concurrently. A single scheduled model
  remains the shared execution lane for TUI chat and HTTP requests.

## Experience architecture

### Stable frame

The application renders one outer frame for the lifetime of the program. The
frame contains, from top to bottom:

1. a brand and primary-panel bar;
2. a recent-session strip;
3. the active panel and optional inspector;
4. the chat composer when Chat is active; and
5. a compact status and key-help footer.

The primary panels are:

- **Chat** — transcript, composer, generation state, and per-session context;
- **Work** — local work items and the future agent activity surface;
- **Models** — discovery, filtering, details, loading, and recent choices; and
- **Service** — HTTP service state, address, request count, and concise actions.

Tab and Shift+Tab change the primary panel. They do not change the active
session. Settings, Tools, and Modes cease to be primary tabs and move into the
inspector where their relationship to the current session is visible.

### Session strip and switcher

The strip shows the active session plus the most recently opened sessions that
fit. Each item carries a text label and a state marker: idle, queued,
generating, needs attention, or failed. Overflow is represented by a session
count rather than squeezed labels.

- `Ctrl+N` creates a session.
- `Ctrl+P` opens a fuzzy-search session switcher.
- `Alt+Left` and `Alt+Right` move between recent sessions.
- Rename, archive, reopen, and export are available from the command palette
  and session context menu.

Export has explicit Markdown and JSONL actions. The default destination is
`~/.lem/exports/`; a user-selected destination overrides it for that export.
Export never changes or closes the source session.

Creating a session does not require a loaded model. Its title starts as
`New session` and is replaced by a compact title derived from the first user
turn unless the user renamed it first. Archive is reversible and does not
delete turns, events, work items, or artifacts.

### Chat panel

The chat panel is a scrollable Bubbles viewport above a Bubbles textarea.
Completed assistant messages are rendered with a custom Glamour style. User
messages, thoughts, tool calls, tool results, errors, and runner events have
distinct but related visual treatments.

Thoughts and structured events render as compact cards. Long bodies are
collapsed by default and can be expanded without changing the transcript's
logical ordering. Status is never communicated by colour alone: cards carry a
label and glyph as well as a semantic colour.

The viewport follows new output only when it was already at the bottom. Manual
scrolling suspends follow mode and displays a `new output` marker. End returns
to live output. Completed message Markdown is cached by turn ID, content hash,
width, and theme. The currently streaming assistant message is refreshed at a
bounded cadence and receives its final rich render when generation completes.

Enter sends. Alt+Enter inserts a newline. Escape moves focus out of the
composer or closes the topmost overlay. A draft is saved per session and
restored when switching back.

### Work panel

Work is grouped into Active, Waiting, and Done. The left side is a filterable
list; the right side shows the selected item's timeline, repository, branch,
runtime, workspace path, question, pull-request URL, and related artifacts.
On smaller terminals the list and detail become sequential views.

The user can create, rename, complete, reopen, and archive a LEM work item.
Work items can be linked to a chat session. Agent activity, questions, logs,
artifacts, and lifecycle actions have stable presentation states, but the
default agent capability reports `not installed` and every execution action is
disabled with that reason. This slice never pretends an unavailable backend
succeeded.

### Models panel

The existing asynchronous model discovery remains, presented as a list/detail
split. The list supports fuzzy filtering and carries loading/current/recent
markers. Details show the discovered path and metadata available from the
inference model. Selection is separate from loading, so browsing never unloads
the current model.

Each session remembers a preferred model plus its mode, generation values, and
enabled tools. Activating a session never loads or unloads a model implicitly.
When the preferred and currently loaded models differ, the inspector explains
the mismatch and offers an explicit load action. Every assistant turn records
the model that actually produced it.

A model change is blocked while any chat generation or HTTP request is in
flight. The UI explains why and offers the relevant cancel or wait action; it
never tears down active work implicitly.

### Service panel

The Service panel retains the existing one-lane scheduler shared with TUI chat.
It presents start/stop, listen address, current model, request count, and base
URLs first. Endpoint and curl examples are collapsed help rather than the main
surface. Listener errors remain associated with this panel and appear in the
global footer until acknowledged.

### Inspector and overlays

On terminals at least 120 columns wide, a 34-to-40-column inspector appears to
the right of the active panel. It contains contextual sections for Model,
Generation, Mode, Tools, Knowledge, Runtime, and Session. Sections may collapse
but retain a one-line summary.

Between 80 and 119 columns, the inspector is an overlay toggled with `Ctrl+O`.
Below 80 columns, the workspace becomes a single-column compact layout and the
same inspector opens as a full content view. The outer frame, active panel, and
footer remain recognisable at every supported width.

`Ctrl+K` opens a command palette containing every global or contextual action.
`Ctrl+F` opens search for the active panel. `F1` opens complete key help.
Overlays own keyboard input until dismissed; background panels do not receive
the same key event.

## Visual language

The default theme is a restrained midnight palette with warm amber focus,
violet identity, and mint success accents. Lip Gloss adaptive colours provide
legible light-terminal equivalents. Borders are structural rather than
decorative: one strong outer frame, subtle panel separators, and a brighter
border only around the focused region.

Whitespace, alignment, and text hierarchy carry most of the design. Animation
is limited to generation, loading, service activity, and work activity.
Spinners stop when their operation stops. Empty states state the next useful
action and its key binding.

The interface must remain usable without colour and with Unicode width
variation. Layout calculations use rendered cell width rather than byte or rune
count. Long paths and titles are middle-truncated where retaining the suffix is
valuable.

## Component architecture

The existing `app` remains the Bubble Tea root but becomes a coordinator rather
than owning every interaction and render decision. It composes focused models:

- `sessionManager` — recent sessions, active session, drafts, and generation
  state;
- `chatPanel` — transcript viewport, composer, follow mode, and chat actions;
- `workPanel` — work-item list/detail and imported agent events;
- `modelPanel` — discovery, selection, loading, and details;
- `servicePanel` — lifecycle and request state over the existing service code;
- `inspector` — contextual settings and capabilities;
- `commandPalette` and `sessionSwitcher` — modal searchable overlays;
- `workspaceLayout` — width/height allocation and responsive composition; and
- `theme` plus `markdownRenderer` — style tokens and cached rich rendering.

The root model owns an explicit focus value: navigation, body, inspector,
composer, or overlay. Global key handling runs first, modal handling second,
and only the focused component receives the remaining message.

All filesystem, database, configuration, model, service, and agent operations
run through `tea.Cmd`. Background goroutines send typed messages back to the
update loop and never mutate a Bubble Tea model directly.

The package stays split into focused sibling files under `cli/tui`. Persistence
and external integrations are represented by small interfaces in the TUI
package and implemented in separate adapter files. This preserves the existing
package shape and avoids a premature internal framework.

## Imported dependency boundaries

### dAppCore ORM and DuckDB

`dappco.re/go/orm` owns typed CRUD for durable LEM records. Its DuckDB medium is
mounted once for `~/.lem/lem.duckdb`. LEM applies explicit, transactional SQL
migrations through the medium's raw connection before registering ORM schemas.
Schema changes are versioned; startup never guesses a migration from struct
shape.

### dAppCore go-store

`dappco.re/go/store` opens `~/.lem/state.db` as the reactive key/value store.
A scoped `lem` namespace contains per-session drafts and small UI values such
as the active session, recent-session ordering, active panel, and collapsed
section state. Chat and work records never live in the key/value database.

Store watchers are bridged into Bubble Tea commands where a live update is
useful. All watchers are unregistered and stores closed during shutdown.

### dAppCore config

`dappco.re/go/config` reads and commits `config.yaml` through
`config.WithMedium`. Defaults cover generation values, appearance, knowledge
paths, recent-session limit, preferred runtime, and whether future execution
features require confirmation.
Environment-derived values may affect the resolved view but are not written
back when the user commits an unrelated setting.

### dAppCore go-io

The composition root resolves the user's home, joins `.lem`, creates
`io.NewSandboxed(resolvedRoot)`, and injects the resulting `io.Medium` into
configuration, knowledge, export, and agent-capability boundaries. Medium paths
are application-relative: `config.yaml`, `packs/`, `exports/`, and
`workspaces/`. Components never join these paths to the host home themselves.

DuckDB and SQLite drivers require local database filenames, so the composition
root separately retains the resolved local paths for `lem.duckdb` and
`state.db`. Those path-only adapters remain behind repository interfaces. A
future remote deployment can replace the file medium without changing panels,
then choose a remote repository adapter independently.

### Future agent capability

The TUI owns a small agent capability interface for availability, work
snapshots, events, and lifecycle commands. The TUI-first implementation is an
unavailable provider with a stable reason. No CoreAgent type, package, status
file, or CLI path crosses into the TUI.

The later `go/agent/*` port will implement this interface using medium-backed
workspaces and inference-native domain values. CoreAgent's runner, queue,
monitor, setup, Brain, session/handoff, provider, fleet, forge, QA, review, PR,
merge, and recovery behaviour are design inputs for that port rather than a
dependency of this slice.

### dAppCore go-container

The runtime adapter performs read-only detection and capability inspection.
The inspector and Work detail show the resolved runtime and available choices,
including native macOS or OCI runtimes where present. Detection failure disables
the capability with a reason and never prevents local chat.

### Knowledge packs

Knowledge packs are Markdown content rather than a required Go runtime. LEM
discovers Markdown below `~/.lem/packs/` and additional paths configured by the
user. The inspector provides search, preview, attach, and detach actions.

An attachment stores its source path, title, content hash, and selected content
snapshot in DuckDB. The snapshot makes an existing conversation reproducible
if the source pack later changes; the UI marks the attachment stale when its
current hash differs. This slice performs no network clone or update.

### Charmbracelet

The implementation stays on the Bubble Tea and Bubbles major versions already
used by the CLI. It adds Glamour for Markdown and reuses Bubbles viewport,
textarea, list, help, spinner, and key-binding primitives. Dependency upgrades
are limited to versions required for compatibility with the existing Bubble
Tea runtime.

### Versioning discipline

Tagged module releases are used where available. Untagged dAppCore modules are
pinned to an exact reachable pseudo-version. The CLI module receives no local
`replace` directive and the repository `go.work` receives no path outside this
repository. This keeps builds reproducible without the sibling checkouts.

## Application data and schema

The default application root is fixed at `~/.lem/`. Production code resolves
the user's home through `dappco.re/go`; tests inject an explicit temporary root
through the composition constructor. The CLI does not add a second public home
setting in this slice.

```text
~/.lem/
├── config.yaml
├── lem.duckdb
├── state.db
├── workspaces/
├── packs/
└── exports/
```

The root and newly created private subdirectories use owner-only directory
permissions. Existing user permissions are not silently rewritten.

DuckDB contains these logical tables:

| Table | Purpose |
| --- | --- |
| `lem_schema_versions` | Applied migration number and timestamp. |
| `sessions` | Identity, title, lifecycle, preferred model, per-session settings, and timestamps. |
| `turns` | Ordered user, assistant, and system content plus completion state and token counts. |
| `events` | Thoughts, tool calls/results, errors, service events, and runner activity. |
| `generation_jobs` | Durable queued, generating, completed, cancelled, failed, or interrupted generation state. |
| `work_items` | Session-linked tasks and imported workspace status. |
| `artifacts` | Files, URLs, exports, and pull requests related to work. |
| `attachments` | Snapshotted knowledge-pack context and source hashes. |

Primary identifiers are UUID strings. Turns have a session-scoped ordinal with
a uniqueness constraint. Session and work-item status values are validated at
the repository boundary. Indexes support recent sessions, ordered turns,
session event timelines, work status, and attachment lookup.

No record is hard-deleted by a normal UI action. Archive timestamps exclude
records from default lists while preserving queryability and export.

## Data flows

### Startup

1. Resolve and create the `~/.lem/` structure.
2. Load configuration and report malformed input without overwriting it.
3. Open DuckDB, apply migrations transactionally, and mount ORM schemas.
4. Open and scope go-store.
5. Restore sessions, active panel, active session, and its draft.
6. Start model discovery, knowledge discovery, and runtime detection
   concurrently; resolve the agent capability immediately.
7. Render immediately with loading placeholders; each result arrives as a
   typed Bubble Tea message.

### Sending a chat turn

1. Validate non-empty input and the loaded model.
2. Persist the user turn before clearing its saved draft.
3. Persist a per-session generation job and enqueue it on the shared model lane.
4. Stream parser events into the owning session regardless of which session is
   currently visible.
5. Record completed thought/tool/error events and render the evolving assistant
   message at a bounded cadence.
6. Persist the completed assistant turn and token counts, or persist an
   interrupted partial turn plus an error event.
7. Mark the session as needing attention if it completed while hidden.

Each session may have one in-flight generation. The shared scheduler serialises
model access across sessions and HTTP requests. Switching sessions never
cancels a generation. Additional sends in other sessions queue visibly.

After an unclean process exit, startup changes persisted queued or generating
jobs to interrupted. It never restarts work automatically.

### Tool loop

Structured tool calls become events before execution. Enabled built-in tools
run through the existing tool registry, and their structured results become a
linked event and tool turn before generation continues. Disabled, malformed,
or failed tool calls produce an explicit result the model can recover from.

### Agent capability states

The Work panel asks the capability provider for availability before offering
agent actions. An unavailable provider returns a reason displayed in the panel,
inspector, and command palette; dispatch, cancel, answer, retry, and resume stay
disabled. Local work-item CRUD remains fully functional. Later providers emit
LEM-owned work snapshots and timeline events, so the UI and DuckDB schema do
not depend on their internal types.

## Failure and recovery behaviour

- Failure to create or open the application root or DuckDB produces a blocking
  storage screen with the exact path, error, Retry, and Quit. There is no silent
  in-memory history mode.
- A malformed configuration file is preserved. The user may continue for the
  current run with defaults or quit and repair it; continuing does not commit
  over the malformed file.
- Migrations run in a transaction. Failure rolls back, closes the database, and
  presents the storage screen.
- go-store failure disables draft/UI-state persistence with a persistent
  warning but does not endanger DuckDB chat history.
- Knowledge, runtime, and agent-capability failures disable only their related
  capability and show a reason in the inspector.
- Model and tool failures are recorded on the owning session, even when hidden.
- A listener failure stops only the HTTP service; local chat remains available.
- Shutdown cancels generation, closes capability providers, tears down the HTTP service,
  closes the model, unregisters watchers, flushes pending state, and closes both
  stores in that order.

## Privacy and authority

All conversation, work, knowledge snapshots, and state remain local below
`~/.lem/` unless the user explicitly exports them. Export writes only to the
chosen destination and preserves the source archive.

This slice detects runtimes but has no authority to spawn agents, create
containers, mutate repositories, or contact a remote forge. Knowledge
discovery reads configured medium paths only. Future execution and network
features require explicit designs and user-visible confirmation boundaries.

## Testing and quality

Production dependencies are hidden behind interfaces so tests use temporary
DuckDB/go-store paths, an in-memory file medium, fake inference models,
deterministic knowledge fixtures, and fake agent/runtime adapters.

Required test coverage includes:

- application-root creation, permissions, and injected test roots;
- every schema migration plus restart persistence;
- session create, title, switch, archive, reopen, search, and export;
- per-session draft preservation;
- queued background generation and hidden-session attention state;
- interrupted and failed generation persistence;
- tool event ordering and replay into model history;
- unavailable agent actions, injected agent snapshots, and event transitions;
- knowledge discovery, attachment snapshots, and stale detection;
- config load, malformed-file handling, and explicit commit behaviour;
- focus routing and overlay key ownership;
- wide, medium, narrow, and short-terminal layout invariants;
- viewport follow suspension and restoration;
- model-change and service scheduler safety; and
- orderly shutdown of every owned resource.

Tests assert behaviour directly rather than only comparing snapshots. ANSI-
stripped render tests check structural invariants such as frame width, panel
presence, truncation, and absence of overflow. Focused benchmarks cover a long
transcript render, Markdown cache hits, and a large session switcher.

Before handback, the implementation must pass focused CLI/TUI tests, the CLI
module suite, `task qa`, `task cover` with the repository's 95% project and
patch target, relevant benchmarks, `git diff --check`, and the documented
core/go compliance audit. Metal-tagged live tests remain separate from the
portable gate.

## Acceptance criteria

The workspace-foundation slice is complete when:

1. `lem tui` opens into a polished full-frame layout at wide, medium, and narrow
   terminal sizes.
2. A user can create, rename, switch, search, archive, reopen, and export
   sessions, then restart the program and recover them from `~/.lem/lem.duckdb`.
3. Generation can continue in one session while the user views or queues work
   in another, with visible per-session state and a globally serial model lane.
4. Markdown, thoughts, tools, errors, and agent events render as distinct,
   scrollable transcript elements without stealing manual scroll position.
5. Work items persist, the future agent activity surface is complete, and all
   unavailable execution actions are visibly disabled rather than simulated.
6. Settings persist through dAppCore config, drafts through go-store, relational
   records through ORM/DuckDB, runtime capabilities through go-container, and
   local knowledge documents through the pack browser.
7. The current model picker and HTTP service remain functional inside the new
   shell.
8. Every unavailable optional capability degrades locally with an actionable
   explanation, while storage failures never masquerade as successful
   persistence.
9. The implementation meets the repository's portable QA, coverage,
   compliance, and benchmark gates.

## Follow-on slices

The stable boundaries in this design support later specs without another UI
rewrite:

1. **Agent execution:** explicit local/container dispatch, approvals, streamed
   logs, questions, cancellation, restart recovery, and workspace cleanup.
2. **Memory and retrieval:** OpenBrain integration, embeddings, semantic search,
   memory consent, and knowledge-pack ranking.
3. **Fleet and collaboration:** multiple runners, remote nodes, hand-off,
   concurrency/rate views, and forge workflows.

CoreAgent's proven behaviours are ported into focused `go/agent/*` packages one
capability at a time. UI components and persisted LEM records remain stable
throughout that migration.
