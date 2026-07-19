# LEM Workspace TUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace the current six-tab LEM terminal UI with a polished, persistent four-panel workspace that supports concurrent chat sessions, searchable DuckDB history, a shared serial model lane, contextual controls, local work/runtime/knowledge views, and complete unavailable states for the later `go/agent/*` capabilities.

**Architecture:** Keep `cli/tui` as one focused Bubble Tea package. A composition root opens a sandboxed `go-io` medium rooted at `~/.lem/`, local DuckDB/go-store repositories, and medium-backed config; the Bubble Tea model owns only LEM-defined interfaces and view state. One scheduler-wrapped model is shared by all session jobs and the HTTP service, while a capability-driven agent boundary renders honest disabled actions until the later `go/agent/*` port implements them.

**Tech Stack:** Go 1.26.2, Bubble Tea 1.3.10, Bubbles 1.0.0, Lip Gloss 1.1.1 pseudo-version already pinned, Glamour 1.0.0, `dappco.re/go/io` 0.15.1, `dappco.re/go/orm` with DuckDB, `dappco.re/go/store` with SQLite, `dappco.re/go/config`, and `dappco.re/go/container` read-only runtime detection.

## Global Constraints

- The approved design is [`docs/superpowers/specs/2026-07-17-lem-workspace-tui-design.md`](../specs/2026-07-17-lem-workspace-tui-design.md). If implementation pressure conflicts with it, stop and update the design before changing behaviour.
- Keep the root `dappco.re/go/inference` package portable. TUI, ORM, storage, agent, container, and Glamour dependencies stay in the nested `cli` module.
- Production state lives only below `~/.lem/`; tests inject temporary local SQL paths plus an in-memory file medium through the internal composition constructor. Do not add a public `--home` or environment override.
- File-like state uses an injected `dappco.re/go/io.Medium`. The production medium is `io.NewSandboxed(~/.lem)`; config, packs, exports, and workspaces use medium-relative paths. Only DuckDB and SQLite adapters receive resolved local filenames.
- Do not add `replace` directives or an out-of-repository `go.work` entry.
- Use `dappco.re/go` wrappers in production for formatting, errors, filesystem access, paths, JSON, strings, logging, bytes, and process execution. Existing banned imports in touched TUI files are part of this refactor.
- Every fallible production helper returns `core.Result`; inspect `r.OK` before reading `r.Value`.
- Write a failing behavioural test before each implementation change. Tests named for a symbol must invoke that symbol directly.
- Add tests beside their source file. Do not add monolithic compliance tests, versioned test files, or `ax7*` files.
- Do not hard-delete sessions, turns, events, work items, attachments, or artifacts from normal UI actions. Archive by flag and timestamp.
- Do not spawn an agent, import CoreAgent, create a container, mutate a repository, contact a forge, or download a knowledge pack in this slice.
- The Work panel and command palette must expose future agent capability states without simulating success. Dispatch, cancel, answer, resume, queue, setup, plan/session, monitor, Brain, fleet, forge, QA, review, PR, and merge actions remain disabled with the provider's reason.
- Only one inference model may be resident. Wrap it once in the serial scheduler and share that exact lane between every TUI session and the HTTP service.
- Switching sessions never cancels generation. Each session may own one job; the shared scheduler serialises actual model access.
- A task is not complete until its focused tests pass and its commit contains only that task's files.

## Verified Dependency Pins

| Module | Pin | Use in this slice |
|---|---|---|
| `dappco.re/go/io` | `v0.15.1` | sandboxed local medium and replaceable file boundary |
| `dappco.re/go/store` | `v0.14.1` | `~/.lem/state.db`, scoped draft/UI state |
| `dappco.re/go/config` | `v0.18.0` | `~/.lem/config.yaml`, defaults and explicit commits |
| `dappco.re/go/orm` | `v0.1.1` | typed CRUD over the official DuckDB v2 driver |
| `dappco.re/go/container` | `v0.11.0` | `DetectAll` and capability projection only |
| `github.com/charmbracelet/glamour` | `v1.0.0` | width-aware Markdown rendering |
| `github.com/google/uuid` | `v1.6.0` | durable record IDs; promote the existing indirect pin to direct |

ORM `v0.1.1` and go-store `v0.14.1` both use the official `github.com/duckdb/duckdb-go/v2` driver. The TUI-first slice deliberately carries no CoreAgent module dependency; later agent behaviours enter through `agentProvider`.

## Target File Map

Create these focused files under `cli/tui`:

- `paths.go` / `paths_test.go` — fixed application-root and sandboxed-medium contract.
- `records.go` / `records_test.go` — LEM-owned ORM record types and schemas.
- `migrations.go` / `migrations_test.go` — explicit DuckDB migrations.
- `repository.go` / `repository_test.go` — typed CRUD, search, recovery, archive.
- `preferences.go` / `preferences_test.go` — config defaults, staging, commit, malformed-file handling.
- `reactive.go` / `reactive_test.go` — scoped go-store draft/UI state.
- `bootstrap.go` / `bootstrap_test.go` — composition and strict/degraded startup.
- `sessions.go` / `sessions_test.go` — multi-session domain state and attention.
- `jobs.go` / `jobs_test.go` — per-session jobs and the shared model lane.
- `layout.go` / `layout_test.go` — responsive geometry and frame composition.
- `keymap.go` / `keymap_test.go` — global keys and help bindings.
- `markdown.go` / `markdown_test.go` / `markdown_bench_test.go` — cached Glamour rendering.
- `palette.go` / `palette_test.go` — command palette, switcher, search, help overlays.
- `inspector.go` / `inspector_test.go` — contextual settings/modes/tools/knowledge/runtime detail.
- `work.go` / `work_test.go` — work list and status transitions.
- `agentcap.go` / `agentcap_test.go` — future agent feature catalogue, availability, snapshots, and disabled action states.
- `runtime.go` / `runtime_test.go` — go-container capability adapter.
- `knowledge.go` / `knowledge_test.go` — local Markdown discovery, snapshot, hash, stale state.
- `export.go` / `export_test.go` — Markdown and JSON session exports.

Modify these existing files:

- `cli/go.mod`, `cli/go.sum` — exact dependency pins.
- `cli/tui/app.go`, `app_test.go` — composition, update routing, persistence, recovery, overlays.
- `cli/tui/stream.go` — session/job-tagged events and parent cancellation.
- `cli/tui/service.go` — consume the shared lane without owning or closing it.
- `cli/tui/style.go` — instance-owned adaptive theme styles.
- `cli/tui/tabs.go` — four primary `panelID` values and labels.
- `cli/tui/settings.go`, `modes.go`, `tools.go` — inspector editors rather than primary tabs.
- `cli/tui/picker.go` — model panel styling and core/go filesystem/string wrappers.
- `cli/tui/tui.go` — asynchronous bootstrap, check render, lifecycle close.
- `cli/tui/README.md` — user workflow, keys, storage, limitations.
- `docs/superpowers/specs/2026-07-17-lem-workspace-tui-design.md` — keep the TUI-first and medium boundaries aligned with implementation.

## Task 1: Establish the Fixed `~/.lem/` Path Contract

**Files:**

- Modify: `cli/go.mod`
- Modify: `cli/go.sum`
- Create: `cli/tui/paths.go`
- Create: `cli/tui/paths_test.go`

**Interfaces produced:**

```go
type appPaths struct {
	Root       string
	Database   string
	State      string
	Config     string
	Workspaces string
	Packs      string
	Exports    string
}

type appFiles struct {
	Paths  appPaths
	Medium coreio.Medium
}

func defaultAppPaths() core.Result
func appPathsAt(root string) core.Result
func openAppFilesAt(root string) core.Result
func ensureAppFiles(medium coreio.Medium, paths appPaths) core.Result
```

- [x] Write `TestAppPaths_Good`, asserting `Database` and `State` resolve below the injected host root while `Config`, `Workspaces`, `Packs`, and `Exports` are exactly the medium-relative values `config.yaml`, `workspaces`, `packs`, and `exports`.
- [x] Write `TestAppFiles_Good`, opening a temporary root, asserting the three medium directories exist, writing `config.yaml` through the returned medium, and proving an absolute/traversal write cannot escape the root.
- [x] Write `TestAppPaths_Bad`, asserting `appPathsAt("")` returns `!OK`.
- [x] Write `TestAppFiles_Ugly`, placing a regular file at the intended root and asserting `openAppFilesAt` fails without deleting or replacing it.
- [x] Run the focused tests and confirm they fail because the path contract does not exist:

```sh
cd /Users/snider/Code/core/go-inference/cli
go test ./tui -run '^TestApp(Paths|Files)_' -count=1
```

- [x] Pin the fixed medium implementation before production code imports it:

```sh
cd /Users/snider/Code/core/go-inference/cli
go get dappco.re/go/io@v0.15.1
```

- [x] Implement `appPathsAt` with `core.Path` only for the root and two database filenames. Implement `openAppFilesAt` with `coreio.NewSandboxed(root)`, then call `EnsureDir("")`, `EnsureDir("workspaces")`, `EnsureDir("packs")`, and `EnsureDir("exports")`. Do not create either database or the config file in this helper.
- [x] Re-run the focused tests, then commit:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/go.mod cli/go.sum cli/tui/paths.go cli/tui/paths_test.go
git commit -m "feat(tui): establish lem application paths"
```

## Task 2: Add Typed Records and Transactional DuckDB Migrations

**Files:**

- Modify: `cli/go.mod`
- Modify: `cli/go.sum`
- Create: `cli/tui/records.go`
- Create: `cli/tui/records_test.go`
- Create: `cli/tui/migrations.go`
- Create: `cli/tui/migrations_test.go`

**Records produced:**

```go
type schemaVersionRecord struct {
	Version   int64     `json:"version"`
	AppliedAt time.Time `json:"applied_at"`
}

type sessionRecord struct {
	ID             string    `json:"id"`
	Title          string    `json:"title"`
	Status         string    `json:"status"`
	PreferredModel string    `json:"preferred_model"`
	Mode           string    `json:"mode"`
	GenerationJSON string    `json:"generation_json"`
	ToolsJSON      string    `json:"tools_json"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
	LastOpenedAt   time.Time `json:"last_opened_at"`
	Archived       bool      `json:"archived"`
	ArchivedAt     time.Time `json:"archived_at"`
}

type turnRecord struct {
	ID             string    `json:"id"`
	SessionID      string    `json:"session_id"`
	Sequence       int64     `json:"sequence"`
	Role           string    `json:"role"`
	Visible        string    `json:"visible"`
	Thought        string    `json:"thought"`
	ToolName       string    `json:"tool_name"`
	ToolCallJSON   string    `json:"tool_call_json"`
	ToolResultJSON string    `json:"tool_result_json"`
	Model          string    `json:"model"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}

type eventRecord struct {
	ID         string    `json:"id"`
	SessionID  string    `json:"session_id"`
	WorkItemID string    `json:"work_item_id"`
	JobID      string    `json:"job_id"`
	Kind       string    `json:"kind"`
	Status     string    `json:"status"`
	Title      string    `json:"title"`
	Detail     string    `json:"detail"`
	PayloadJSON string   `json:"payload_json"`
	CreatedAt  time.Time `json:"created_at"`
}

type generationJobRecord struct {
	ID           string    `json:"id"`
	SessionID    string    `json:"session_id"`
	PromptTurnID string    `json:"prompt_turn_id"`
	AnswerTurnID string    `json:"answer_turn_id"`
	Status       string    `json:"status"`
	Model        string    `json:"model"`
	Error        string    `json:"error"`
	MetricsJSON  string    `json:"metrics_json"`
	CreatedAt    time.Time `json:"created_at"`
	StartedAt    time.Time `json:"started_at"`
	FinishedAt   time.Time `json:"finished_at"`
}

type workItemRecord struct {
	ID         string    `json:"id"`
	ExternalID string    `json:"external_id"`
	Source     string    `json:"source"`
	Title      string    `json:"title"`
	Status     string    `json:"status"`
	Agent      string    `json:"agent"`
	Repo       string    `json:"repo"`
	Org        string    `json:"org"`
	Task       string    `json:"task"`
	Branch     string    `json:"branch"`
	Runtime    string    `json:"runtime"`
	Question   string    `json:"question"`
	PRURL      string    `json:"pr_url"`
	SessionID  string    `json:"session_id"`
	StartedAt  time.Time `json:"started_at"`
	UpdatedAt  time.Time `json:"updated_at"`
	Archived   bool      `json:"archived"`
	ArchivedAt time.Time `json:"archived_at"`
}

type artifactRecord struct {
	ID           string    `json:"id"`
	SessionID    string    `json:"session_id"`
	WorkItemID   string    `json:"work_item_id"`
	Kind         string    `json:"kind"`
	Path         string    `json:"path"`
	Title        string    `json:"title"`
	MetadataJSON string    `json:"metadata_json"`
	CreatedAt    time.Time `json:"created_at"`
	Archived     bool      `json:"archived"`
	ArchivedAt   time.Time `json:"archived_at"`
}

type attachmentRecord struct {
	ID            string    `json:"id"`
	SessionID     string    `json:"session_id"`
	SourcePath    string    `json:"source_path"`
	Title         string    `json:"title"`
	ContentHash   string    `json:"content_hash"`
	Snapshot      string    `json:"snapshot"`
	AddedAt       time.Time `json:"added_at"`
	LastCheckedAt time.Time `json:"last_checked_at"`
	Stale         bool      `json:"stale"`
	Archived      bool      `json:"archived"`
	ArchivedAt    time.Time `json:"archived_at"`
}
```

Use Unix epoch UTC as the explicit “unset” timestamp for non-null archive and finish fields. Each record implements `Schema() orm.Schema`; names are `lem_schema_versions`, `lem_sessions`, `lem_turns`, `lem_events`, `lem_generation_jobs`, `lem_work_items`, `lem_artifacts`, and `lem_attachments`. Declare text IDs as primary keys, `external_id` as unique, and indexes matching the migration below.

- [x] Pin ORM, go-store, and UUID directly, then tidy only after the new source imports them:

```sh
cd /Users/snider/Code/core/go-inference/cli
go get dappco.re/go/orm@v0.1.1
go get dappco.re/go/store@v0.14.1
go get github.com/google/uuid@v1.6.0
```

- [x] Write `TestRecordSchemas_Good`, checking every schema name, primary key, field, unique constraint, and required index directly through each record's `Schema()` method.
- [x] Write `TestMigrations_Good`, opening a temporary DuckDB, applying migrations twice, and asserting one schema-version row and all eight tables.
- [x] Write `TestMigrations_Bad`, injecting an invalid statement into a migration and asserting the version row is not committed.
- [x] Write `TestMigrations_Ugly`, pre-creating version 1 and asserting startup skips it without changing its timestamp.
- [x] Run the tests and observe the missing types/functions failure.
- [x] Implement this exact migration inventory as individually executed statements inside one SQL transaction per version:

```sql
CREATE TABLE IF NOT EXISTS lem_schema_versions (version BIGINT PRIMARY KEY, applied_at TIMESTAMP NOT NULL);
CREATE TABLE IF NOT EXISTS lem_sessions (id TEXT PRIMARY KEY, title TEXT NOT NULL, status TEXT NOT NULL, preferred_model TEXT NOT NULL, mode TEXT NOT NULL, generation_json TEXT NOT NULL, tools_json TEXT NOT NULL, created_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT NULL, last_opened_at TIMESTAMP NOT NULL, archived BOOLEAN NOT NULL, archived_at TIMESTAMP NOT NULL);
CREATE TABLE IF NOT EXISTS lem_turns (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, sequence BIGINT NOT NULL, role TEXT NOT NULL, visible TEXT NOT NULL, thought TEXT NOT NULL, tool_name TEXT NOT NULL, tool_call_json TEXT NOT NULL, tool_result_json TEXT NOT NULL, model TEXT NOT NULL, created_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT NULL, UNIQUE(session_id, sequence));
CREATE TABLE IF NOT EXISTS lem_events (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, work_item_id TEXT NOT NULL, job_id TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL, title TEXT NOT NULL, detail TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TIMESTAMP NOT NULL);
CREATE TABLE IF NOT EXISTS lem_generation_jobs (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, prompt_turn_id TEXT NOT NULL, answer_turn_id TEXT NOT NULL, status TEXT NOT NULL, model TEXT NOT NULL, error TEXT NOT NULL, metrics_json TEXT NOT NULL, created_at TIMESTAMP NOT NULL, started_at TIMESTAMP NOT NULL, finished_at TIMESTAMP NOT NULL);
CREATE TABLE IF NOT EXISTS lem_work_items (id TEXT PRIMARY KEY, external_id TEXT NOT NULL UNIQUE, source TEXT NOT NULL, title TEXT NOT NULL, status TEXT NOT NULL, agent TEXT NOT NULL, repo TEXT NOT NULL, org TEXT NOT NULL, task TEXT NOT NULL, branch TEXT NOT NULL, runtime TEXT NOT NULL, question TEXT NOT NULL, pr_url TEXT NOT NULL, session_id TEXT NOT NULL, started_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT NULL, archived BOOLEAN NOT NULL, archived_at TIMESTAMP NOT NULL);
CREATE TABLE IF NOT EXISTS lem_artifacts (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, work_item_id TEXT NOT NULL, kind TEXT NOT NULL, path TEXT NOT NULL, title TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at TIMESTAMP NOT NULL, archived BOOLEAN NOT NULL, archived_at TIMESTAMP NOT NULL);
CREATE TABLE IF NOT EXISTS lem_attachments (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, source_path TEXT NOT NULL, title TEXT NOT NULL, content_hash TEXT NOT NULL, snapshot TEXT NOT NULL, added_at TIMESTAMP NOT NULL, last_checked_at TIMESTAMP NOT NULL, stale BOOLEAN NOT NULL, archived BOOLEAN NOT NULL, archived_at TIMESTAMP NOT NULL);
CREATE INDEX IF NOT EXISTS lem_sessions_recent_idx ON lem_sessions(archived, last_opened_at);
CREATE INDEX IF NOT EXISTS lem_turns_session_idx ON lem_turns(session_id, sequence);
CREATE INDEX IF NOT EXISTS lem_events_session_idx ON lem_events(session_id, created_at);
CREATE INDEX IF NOT EXISTS lem_jobs_session_idx ON lem_generation_jobs(session_id, status, created_at);
CREATE INDEX IF NOT EXISTS lem_work_status_idx ON lem_work_items(archived, status, updated_at);
CREATE INDEX IF NOT EXISTS lem_artifacts_session_idx ON lem_artifacts(session_id, created_at);
CREATE INDEX IF NOT EXISTS lem_attachments_session_idx ON lem_attachments(session_id, archived, added_at);
```

- [x] After migration, mount the DuckDB medium as ORM `default`, register each schema with both the medium and ORM cache, and expose a close function. If any step fails, close the medium before returning.
- [x] Re-run the focused tests, `go mod tidy`, and commit:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/go.mod cli/go.sum cli/tui/records.go cli/tui/records_test.go cli/tui/migrations.go cli/tui/migrations_test.go
git commit -m "feat(tui): add duckdb workspace schema"
```

## Task 3: Implement the Typed Repository, Search, Archive, and Recovery

**Files:**

- Create: `cli/tui/repository.go`
- Create: `cli/tui/repository_test.go`

**Interface produced:**

```go
type sessionSearchHit struct {
	Session sessionRecord
	Snippet string
}

type workspaceRepository interface {
	Close() core.Result
	Session(id string) core.Result
	ListSessions(includeArchived bool) core.Result
	SearchSessions(query string, limit int) core.Result
	SaveSession(record sessionRecord) core.Result
	Turns(sessionID string) core.Result
	SaveTurn(record turnRecord) core.Result
	Events(sessionID string) core.Result
	SaveEvent(record eventRecord) core.Result
	Jobs(sessionID string) core.Result
	SaveJob(record generationJobRecord) core.Result
	InterruptActiveJobs(at time.Time) core.Result
	ListWorkItems(includeArchived bool) core.Result
	WorkItemByExternalID(externalID string) core.Result
	SaveWorkItem(record workItemRecord) core.Result
	Artifacts(sessionID string) core.Result
	SaveArtifact(record artifactRecord) core.Result
	Attachments(sessionID string) core.Result
	SaveAttachment(record attachmentRecord) core.Result
}
```

- [x] Write `TestDuckRepository_Good`: save a session, ordered user/assistant turns, an event, a job, a work item, an artifact, and an attachment; close and reopen the database; then assert typed round trips and ordering.
- [x] Write `TestDuckRepository_Bad`: save two turn IDs at the same `(session_id, sequence)` and assert the second write fails without changing the first.
- [x] Write `TestDuckRepository_Ugly`: create queued and generating jobs, call `InterruptActiveJobs`, and assert both become `interrupted` while completed jobs remain completed.
- [x] Write `TestDuckRepository_SearchAndArchive_Good`: match title and turn content case-insensitively, return a useful snippet, archive a session through `SaveSession`, and prove normal lists/search exclude it while `ListSessions(true)` retains it.
- [x] Run the focused tests and confirm failure before implementation.
- [x] Implement routine CRUD through `orm.Of[T]`. Use parameterised raw DuckDB SQL only for cross-table search and the set-based interrupted-job recovery update.
- [x] Keep repository return values concrete: `sessionRecord`, `[]sessionRecord`, `[]turnRecord`, `[]eventRecord`, `[]generationJobRecord`, `[]workItemRecord`, `[]artifactRecord`, `[]attachmentRecord`, and `[]sessionSearchHit` inside `core.Result`.
- [x] Re-run focused tests and commit:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/tui/repository.go cli/tui/repository_test.go
git commit -m "feat(tui): persist workspace records"
```

## Task 4: Add Durable Preferences with Safe Malformed-File Handling

**Files:**

- Modify: `cli/go.mod`
- Modify: `cli/go.sum`
- Create: `cli/tui/preferences.go`
- Create: `cli/tui/preferences_test.go`

**Values and interface produced:**

```go
type preferenceValues struct {
	ContextLength       int
	MaxTokens           int
	Thinking            string
	Theme               string
	ShowThinking        bool
	RecentSessionLimit  int
	KnowledgePaths      []string
	KnowledgeMaxBytes   int64
	PreferredRuntime    string
	ConfirmExecution    bool
}

type preferenceStore interface {
	Values() preferenceValues
	Set(key string, value any) core.Result
	Commit() core.Result
	Reload() core.Result
	Warning() error
}
```

Defaults are context `0`, max tokens `4096`, thinking `model`, theme `midnight`, show thinking `true`, recent sessions `12`, knowledge max bytes `65536`, preferred runtime `auto`, and confirmation `true`. The default knowledge path is the medium-relative `packs` directory.

- [x] Pin config exactly:

```sh
cd /Users/snider/Code/core/go-inference/cli
go get dappco.re/go/config@v0.18.0
```

- [x] Write `TestPreferences_Good`: stage theme/max-token changes, prove the file remains absent before `Commit`, commit, reopen, and assert both values.
- [x] Write `TestPreferences_Bad`: write malformed YAML, load preferences, assert defaults are usable with a warning, assert `Commit` fails, and byte-compare the malformed file to prove it was not overwritten.
- [x] Write `TestPreferences_Ugly`: set `LEM_GENERATION_MAX_TOKENS`, commit an unrelated theme change, and assert the environment-derived token value is not written into YAML.
- [x] Run the focused tests and confirm failure.
- [x] Implement the main config with `config.WithMedium(files)`, `config.WithPath("config.yaml")`, `WithEnvPrefix("LEM")`, and `WithDefaults`. On parse failure, construct a defaults-only config against a separate in-memory medium and mark commits disabled until `Reload` succeeds against the real medium/path.
- [x] Do not attach config to go-store: config writes are explicit, while go-store is reserved for drafts and UI state.
- [x] Re-run tests, tidy, and commit:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/go.mod cli/go.sum cli/tui/preferences.go cli/tui/preferences_test.go
git commit -m "feat(tui): persist workspace preferences"
```

## Task 5: Add Scoped Reactive State and the Workspace Bootstrap

**Files:**

- Create: `cli/tui/reactive.go`
- Create: `cli/tui/reactive_test.go`
- Create: `cli/tui/bootstrap.go`
- Create: `cli/tui/bootstrap_test.go`

**Interfaces produced:**

```go
type reactiveState interface {
	Get(group, key string) (string, core.Result)
	Set(group, key, value string) core.Result
	Delete(group, key string) core.Result
	Close() core.Result
}

type workspaceResources struct {
	Paths       appPaths
	Files       coreio.Medium
	Repository  workspaceRepository
	State       reactiveState
	Preferences preferenceStore
	Warnings    []string
}

func openWorkspace(root string, openers workspaceOpeners) core.Result
func openWorkspaceWith(files appFiles, openers workspaceOpeners) core.Result
func (resources *workspaceResources) Close() core.Result
```

Use scoped namespace `lem`; groups are `workspace`, `drafts`, `viewport`, and `collapsed`. Keys include `active_session`, `active_panel`, `inspector_open`, and session IDs. A disabled state adapter returns misses for reads and explicit failures for writes while allowing the in-memory UI to continue.

- [x] Write `TestReactiveState_Good`, round-tripping all four groups through a temporary `state.db`, reopening it, and asserting values survived.
- [x] Write `TestReactiveState_Bad`, passing an unusable database path and asserting an open failure rather than a panic.
- [x] Write `TestReactiveState_Ugly`, deleting a draft and asserting the miss is distinguishable from an empty stored draft.
- [x] Write `TestOpenWorkspace_Good`, injecting an in-memory medium through `openWorkspaceWith` and asserting medium directories, migration, repository, state, preferences, and reverse-order close.
- [x] Write `TestOpenWorkspace_Bad`, inject a repository-open failure and assert startup is blocking with the exact DuckDB path in the error.
- [x] Write `TestOpenWorkspace_Ugly`, inject state and config failures independently and assert startup succeeds with a disabled state/default preferences plus persistent warnings.
- [x] Run the focused tests and confirm failure.
- [x] Implement go-store through `store.New(paths.State, store.WithWorkspaceStateDirectory(core.Path(paths.Root, paths.Workspaces)))` and `store.NewScoped(instance, "lem")`. This is confined to the local SQLite adapter; no panel receives the host path.
- [x] Implement bootstrap order: open/ensure the medium, open repository/apply migrations, interrupt stale jobs, open state, open preferences with the same medium. Repository failure is fatal; state/config failure is degraded exactly as the design specifies.
- [x] Re-run tests and commit:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/tui/reactive.go cli/tui/reactive_test.go cli/tui/bootstrap.go cli/tui/bootstrap_test.go
git commit -m "feat(tui): compose persistent workspace state"
```

## Task 6: Build the Multi-Session Domain Model

**Files:**

- Create: `cli/tui/sessions.go`
- Create: `cli/tui/sessions_test.go`

**State produced:**

```go
type chatSession struct {
	Record         sessionRecord
	Turns          []turnRecord
	Draft          string
	ViewportOffset int
	Follow         bool
	Attention      bool
	ToolHops       int
	ActiveJobID    string
}

type sessionManager struct {
	repository workspaceRepository
	state      reactiveState
	ids        func() string
	now        func() time.Time
	order      []string
	activeID   string
	sessions   map[string]*chatSession
}
```

- [x] Write `TestSessionManager_Good`: start empty, create the first session, add a second, switch both directions, save independent drafts/viewport offsets, reconstruct the manager, and assert active/recent state restores.
- [x] Write `TestSessionManager_Bad`: attempt to switch to an unknown ID and assert no active-state mutation.
- [x] Write `TestSessionManager_Ugly`: complete a hidden session, assert its attention marker, switch to it, and assert only that session's marker clears.
- [x] Write `TestSessionManager_Title_Good`: first user text creates a bounded one-line title; later prompts do not rename a user-edited title.
- [x] Run the tests and observe failure.
- [x] Implement `Ctrl+N`-ready creation, recent ordering, previous/next navigation, active persistence, draft persistence, lazy turn loading, archive exclusion, and attention state. Repository records are the source of truth; go-store stores only draft/UI affordances.
- [x] Re-run and commit:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/tui/sessions.go cli/tui/sessions_test.go
git commit -m "feat(tui): add persistent chat sessions"
```

## Task 7: Route Per-Session Jobs Through One Shared Model Lane

**Files:**

- Create: `cli/tui/jobs.go`
- Create: `cli/tui/jobs_test.go`
- Modify: `cli/tui/stream.go`
- Modify: `cli/tui/service.go`
- Modify: `cli/tui/app_test.go`

**Interfaces produced:**

```go
type modelLane struct {
	base      inference.TextModel
	scheduled *scheduler.Model
	name      string
}

type generation struct {
	SessionID string
	JobID     string
	cancel    context.CancelFunc
	events    chan streamEvent
}

type jobManager struct {
	parent    context.Context
	bySession map[string]*generation
}

func newModelLane(model inference.TextModel, name string) core.Result
func (lane *modelLane) Model() inference.TextModel
func (lane *modelLane) Close() core.Result
func (jobs *jobManager) Start(sessionID, jobID string, model inference.TextModel, history []inference.Message, opts []inference.GenerateOption) core.Result
func (jobs *jobManager) Cancel(sessionID string) core.Result
func (jobs *jobManager) CancelAll() core.Result
```

Every `streamEvent` and `streamMsg` carries `SessionID` and `JobID`. `Start` rejects a second job in the same session but permits jobs in distinct sessions. Scheduler config is serial, max concurrent 1, max queue 64, stream buffer 16, request prefix `tui`.

- [x] Add a deterministic `fakeTextModel` in `jobs_test.go` that satisfies every `inference.TextModel` method, honours cancellation, can block/yield scripted tokens, counts concurrent calls, and records close calls.
- [x] Write `TestModelLane_Good`, submit two concurrent chats and prove the base model's maximum concurrency is one.
- [x] Write `TestModelLane_Bad`, assert an unknown scheduler construction cannot be represented and nil base input fails.
- [x] Write `TestModelLane_Ugly`, close with queued and running jobs and assert both drain, the base closes exactly once, and close is idempotent.
- [x] Write `TestJobManager_Good`, start jobs for two sessions, consume tagged events interleaved, and assert no delta lands in the wrong session.
- [x] Write `TestJobManager_Bad`, start two jobs in one session and assert the second fails.
- [x] Write `TestJobManager_Ugly`, cancel one hidden session and prove the other continues.
- [x] Refactor service tests to prove starting/stopping the listener never creates or closes a scheduler/model. The service resolver receives `lane.Model()` and stopping only owns the HTTP context/listener.
- [x] Run focused tests and confirm failure before implementation.
- [x] Implement the lane once per loaded model. Remove `serviceState.sched`, remove `chatModel`'s service-dependent branch, and let chat always use `lane.Model()`.
- [x] Preserve the reasoning parser and metrics sink, but derive generation contexts from the app parent rather than `context.Background()`.
- [x] Re-run focused tests and commit:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/tui/jobs.go cli/tui/jobs_test.go cli/tui/stream.go cli/tui/service.go cli/tui/app_test.go
git commit -m "feat(tui): share one model lane across sessions"
```

## Task 8: Build the Responsive Frame, Four Panels, Theme, and Key Map

**Files:**

- Create: `cli/tui/layout.go`
- Create: `cli/tui/layout_test.go`
- Create: `cli/tui/keymap.go`
- Create: `cli/tui/keymap_test.go`
- Modify: `cli/tui/style.go`
- Modify: `cli/tui/tabs.go`

**Layout contract:**

```go
type layoutKind uint8

const (
	layoutNarrow layoutKind = iota
	layoutOverlay
	layoutWide
)

type panelID uint8

const (
	panelChat panelID = iota
	panelWork
	panelModels
	panelService
	panelCount
)

func chooseLayout(width int) layoutKind
```

`chooseLayout` returns wide at 120 columns and above, overlay at 80–119, and narrow below 80. Wide renders a 32-column inspector beside the main panel. Overlay renders the inspector above the main panel only while open. Narrow renders one column and short labels without clipping the frame.

- [x] Write `TestChooseLayout_Good` for widths 120 and 160, `TestChooseLayout_Bad` for 80 and 119, and `TestChooseLayout_Ugly` for 0, 1, and 79.
- [x] Write `TestPanelID_Good`, cycling Tab and Shift+Tab through exactly Chat, Work, Models, and Service with wraparound.
- [x] Write `TestKeyMap_Good`, directly matching `Ctrl+N`, `Ctrl+P`, `Alt+Left`, `Alt+Right`, `Ctrl+K`, `Ctrl+O`, `Ctrl+F`, `Ctrl+S`, and `F1` through Bubbles `key.Binding`.
- [x] Write frame tests that assert a stable outer border, product title, panel labels, session strip, main region, inspector policy, and footer at all three widths.
- [x] Run the focused tests and confirm failure.
- [x] Replace global style variables with an app-owned `uiStyles` built from an adaptive `theme`. The `midnight` preset uses cyan focus, violet assistant identity, amber attention, green success, red error, and muted blue-grey; every status also carries a text label or symbol.
- [x] Use Lip Gloss joins and explicit width/height math. Never crop by slicing ANSI strings.
- [x] Re-run and commit:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/tui/layout.go cli/tui/layout_test.go cli/tui/keymap.go cli/tui/keymap_test.go cli/tui/style.go cli/tui/tabs.go
git commit -m "feat(tui): frame the lem agent workspace"
```

## Task 9: Render a Fast Markdown Transcript with Correct Follow Behaviour

**Files:**

- Modify: `cli/go.mod`
- Modify: `cli/go.sum`
- Create: `cli/tui/markdown.go`
- Create: `cli/tui/markdown_test.go`
- Create: `cli/tui/markdown_bench_test.go`
- Modify: `cli/tui/app.go`
- Modify: `cli/tui/app_test.go`

- [x] Pin Glamour exactly:

```sh
cd /Users/snider/Code/core/go-inference/cli
go get github.com/charmbracelet/glamour@v1.0.0
```

- [x] Write `TestMarkdownRenderer_Good`, rendering headings, lists, emphasis, inline code, fenced Go, and a link at two widths; strip ANSI and assert content plus wrapping.
- [x] Write `TestMarkdownRenderer_Bad`, pass an invalid width and assert a plain-text fallback rather than a panic or blank message.
- [x] Write `TestMarkdownRenderer_Ugly`, render the same completed turn twice and assert the second call hits the `(turn ID, content hash, width, theme)` cache.
- [x] Write `TestTranscriptFollow_Good`: at bottom, stream a delta and remain at bottom.
- [x] Write `TestTranscriptFollow_Bad`: scroll upward, stream a delta, and preserve the offset plus a “new output” marker.
- [x] Write `TestTranscriptFollow_Ugly`: finish in a hidden session and assert attention without moving the active viewport.
- [x] Add `BenchmarkMarkdownTranscript` with 20 mixed turns, `ReportAllocs`, and a package sink.
- [x] Run tests and see them fail before implementation.
- [x] Construct Glamour with `WithStandardStyle("dark")`, `WithWordWrap(width)`, `WithTableWrap(true)`, `WithPreservedNewLines()`, and `WithEmoji()`. Cache completed turns; render the active streaming turn as styled plain text until completion, then promote it to cached Markdown.
- [x] Replace unconditional `GotoBottom` with explicit `Follow`. Mouse/wheel, PgUp/PgDn, and arrow scrolling turn follow off; `End` or scrolling to the bottom turns it back on.
- [x] Batch persistence and viewport refresh on a short Bubble Tea tick while streaming, and flush the completed assistant turn/job immediately.
- [x] Re-run tests/benchmark, tidy, and commit:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/go.mod cli/go.sum cli/tui/markdown.go cli/tui/markdown_test.go cli/tui/markdown_bench_test.go cli/tui/app.go cli/tui/app_test.go
git commit -m "feat(tui): render markdown chat transcripts"
```

## Task 10: Add the Command Palette, Session Switcher, Search, and Help Overlays

**Files:**

- Create: `cli/tui/palette.go`
- Create: `cli/tui/palette_test.go`
- Modify: `cli/tui/app.go`
- Modify: `cli/tui/app_test.go`

The always-enabled command registry contains: new session, switch session, search history, toggle inspector, show help, go to each of the four panels, save settings, export active session as Markdown, export as JSON, refresh work, refresh runtimes, and refresh knowledge. Task 12 appends the complete agent feature catalogue as capability-bound commands; unavailable commands stay visible with their reason and cannot be invoked.

- [x] Write `TestCommandPalette_Good`, filter by fuzzy text and invoke a navigation command.
- [x] Write `TestCommandPalette_Bad`, invoke an unknown command ID and assert no state mutation.
- [x] Write `TestSessionSwitcher_Good`, assert recent ordering, status/model metadata, hidden-job markers, keyboard selection, and active-session switch without cancellation.
- [x] Write `TestHistorySearch_Good`, drive the real repository search, select a hit, switch session, and position the viewport at its matching turn.
- [x] Write `TestOverlayRouting_Ugly`, prove an open overlay consumes Enter/Escape/arrows before global or panel handlers.
- [x] Run tests and confirm failure.
- [x] Implement overlays with Bubbles `list.Model` and `help.Model`. Keep one overlay active at a time; Escape closes it without affecting a running job.
- [x] Wire keys: `Ctrl+K` command palette, `Ctrl+P` switcher, `Ctrl+F` search, `F1` help, `Ctrl+N` new session, `Alt+Left/Right` previous/next session.
- [x] Re-run and commit:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/tui/palette.go cli/tui/palette_test.go cli/tui/app.go cli/tui/app_test.go
git commit -m "feat(tui): add workspace navigation overlays"
```

## Task 11: Move Settings, Modes, and Tools into the Contextual Inspector

**Files:**

- Create: `cli/tui/inspector.go`
- Create: `cli/tui/inspector_test.go`
- Modify: `cli/tui/settings.go`
- Modify: `cli/tui/modes.go`
- Modify: `cli/tui/tools.go`
- Modify: `cli/tui/app.go`

- [x] Write `TestInspector_Good`, asserting Chat shows session/model/generation/settings/mode/tools, Work shows work detail/runtime/agent capability, Models shows model detail, and Service shows address/request detail.
- [x] Write `TestInspector_Bad`, open at an overlay width and prove the main panel remains intact behind the overlay.
- [x] Write `TestInspector_Ugly`, open below 80 columns and prove sections remain reachable in one column with no negative dimensions.
- [x] Write `TestInspectorPreferences_Good`, edit max tokens/theme, assert dirty state, press `Ctrl+S`, reopen preferences, and assert persistence.
- [x] Write `TestInspectorTools_Bad`, enable a disabled tool, generate a malformed call, and assert an explicit tool-result turn/event rather than silent execution.
- [x] Run focused tests and confirm failure.
- [x] Reuse the existing settings, modes, and tools domain logic, but remove their primary tab views. Inspector navigation owns its own selected row and edit focus.
- [x] Keep the existing bounded two-hop tool loop. Persist structured call/result JSON and an event before auto-continuation.
- [x] Rebuild styles and Markdown renderer after a saved theme change; do not reload the application.
- [x] Re-run and commit:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/tui/inspector.go cli/tui/inspector_test.go cli/tui/settings.go cli/tui/modes.go cli/tui/tools.go cli/tui/app.go
git commit -m "feat(tui): add contextual workspace inspector"
```

## Task 12: Build the Work Panel and Future Agent Capability Surface

**Files:**

- Create: `cli/tui/agentcap.go`
- Create: `cli/tui/agentcap_test.go`
- Create: `cli/tui/work.go`
- Create: `cli/tui/work_test.go`
- Modify: `cli/tui/palette.go`
- Modify: `cli/tui/inspector.go`
- Modify: `cli/tui/app.go`

**Boundary produced:**

```go
type agentFeature string

const (
	agentFeatureDispatch      agentFeature = "dispatch"
	agentFeatureCancel        agentFeature = "cancel"
	agentFeatureAnswer        agentFeature = "answer"
	agentFeatureRetry         agentFeature = "retry"
	agentFeatureResume        agentFeature = "resume"
	agentFeatureQueueStart    agentFeature = "queue.start"
	agentFeatureQueueStop     agentFeature = "queue.stop"
	agentFeatureSetup         agentFeature = "setup"
	agentFeatureProvider      agentFeature = "provider"
	agentFeatureTemplate      agentFeature = "template"
	agentFeaturePlan          agentFeature = "plan"
	agentFeatureSession       agentFeature = "session"
	agentFeatureHandoff       agentFeature = "handoff"
	agentFeatureScan          agentFeature = "scan"
	agentFeatureAudit         agentFeature = "audit"
	agentFeaturePipeline      agentFeature = "pipeline"
	agentFeatureMonitor       agentFeature = "monitor"
	agentFeatureHarvest       agentFeature = "harvest"
	agentFeatureBrainRecall   agentFeature = "brain.recall"
	agentFeatureBrainRemember agentFeature = "brain.remember"
	agentFeatureMessage       agentFeature = "message"
	agentFeatureFleet         agentFeature = "fleet"
	agentFeatureForge         agentFeature = "forge"
	agentFeatureRemote        agentFeature = "remote"
	agentFeatureQA            agentFeature = "qa"
	agentFeatureReview        agentFeature = "review"
	agentFeaturePRCreate      agentFeature = "pr.create"
	agentFeaturePRMerge       agentFeature = "pr.merge"
)

type agentCapability struct {
	Feature   agentFeature
	Available bool
	Reason    string
}

type agentWorkSnapshot struct {
	ExternalID string
	Title      string
	Status     string
	Agent      string
	Repo       string
	Branch     string
	Runtime    string
	Question   string
	PRURL      string
}

type agentEventSnapshot struct {
	ExternalID string
	WorkID     string
	Kind       string
	Title      string
	Detail     string
	CreatedAt  time.Time
}

type agentSnapshot struct {
	Work   []agentWorkSnapshot
	Events []agentEventSnapshot
}

type agentRequest struct {
	Feature agentFeature
	WorkID  string
	Input   string
}

type agentProvider interface {
	Capabilities() []agentCapability
	Snapshot(ctx context.Context) core.Result
	Run(ctx context.Context, request agentRequest) core.Result
	Close() core.Result
}

func agentFeatureCatalog(reason string) []agentCapability
func newUnavailableAgentProvider(reason string) agentProvider
```

- [x] Write `TestAgentFeatureCatalog_Good`, asserting every constant above appears exactly once and all capabilities carry the supplied unavailable reason.
- [x] Write `TestUnavailableAgentProvider_Bad`, asserting `Snapshot` returns an empty typed snapshot, `Run` fails with the feature and reason, and `Close` succeeds idempotently.
- [x] Write `TestAgentProviderBoundary_Ugly` with a fake provider that returns two work snapshots and three events; refresh twice and assert deterministic display ordering without duplicate persisted events.
- [x] Write `TestWorkPanel_Good`, creating, renaming, completing, reopening, linking, and archiving a local work item through the real repository.
- [x] Write `TestWorkPanel_Bad`, focus every disabled agent action and assert Enter leaves repository/provider state unchanged while displaying the provider reason.
- [x] Write `TestWorkPanel_Ugly`, render empty, local-only, waiting-question, failed, and completed states at wide and narrow widths.
- [x] Write `TestAgentCommandPalette_Good`, asserting all agent feature commands are searchable, visibly disabled, and include the same reason as the Work inspector.
- [x] Run tests and confirm failure.
- [x] Implement the unavailable provider and capability catalogue. `Run` always returns `!OK`; the UI checks `Available` before calling it, so a disabled action has no side effect.
- [x] Implement the Work list/detail panel over `workspaceRepository`; provider snapshots are translated into existing `workItemRecord` and `eventRecord` values at this boundary only.
- [x] Add capability-bound palette commands and Work inspector sections for execution, queue/setup/providers/templates, plans/sessions/handoffs, scan/audit/pipeline/monitor/harvest, Brain/messages, fleet/forge/remote, and QA/review/PR. Keep labels useful even when disabled.
- [x] Enforce the no-CoreAgent boundary with this source scan:

```sh
cd /Users/snider/Code/core/go-inference
if rg -n 'dappco\.re/go/agent' cli; then exit 1; fi
```

- [x] Re-run tests and commit:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/tui/agentcap.go cli/tui/agentcap_test.go cli/tui/work.go cli/tui/work_test.go cli/tui/palette.go cli/tui/inspector.go cli/tui/app.go
git commit -m "feat(tui): add agent capability workspace"
```

## Task 13: Add Read-Only Runtime Capability Inspection

**Files:**

- Modify: `cli/go.mod`
- Modify: `cli/go.sum`
- Create: `cli/tui/runtime.go`
- Create: `cli/tui/runtime_test.go`
- Modify: `cli/tui/inspector.go`

**Boundary produced:**

```go
type runtimeCapability struct {
	Name              string
	Version           string
	Path              string
	GPU               bool
	NetworkIsolation  bool
	VolumeMounts      bool
	Encryption        bool
	HardwareIsolation bool
	SubSecondStart    bool
}

type runtimeDetector interface {
	Detect() core.Result
}
```

- [x] Pin container exactly:

```sh
cd /Users/snider/Code/core/go-inference/cli
go get dappco.re/go/container@v0.11.0
```

- [x] Write `TestRuntimeAdapter_Good`, project scripted Apple, VZ, Docker, and Podman values into LEM capabilities and preserve priority order.
- [x] Write `TestRuntimeAdapter_Bad`, inject a detection failure and assert a disabled inspector section with the exact reason.
- [x] Write `TestRuntimeAdapter_Ugly`, return no runtimes and assert a labelled `none available` state rather than an empty panel.
- [x] Run tests and confirm failure.
- [x] Implement the production adapter with `container.DetectAll` and only capability getters. Do not construct providers and do not call Build, Run, Stop, Exec, Pull, or image APIs.
- [x] Re-run, tidy, and commit:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/go.mod cli/go.sum cli/tui/runtime.go cli/tui/runtime_test.go cli/tui/inspector.go
git commit -m "feat(tui): inspect local runtime capabilities"
```

## Task 14: Discover and Attach Local Knowledge Packs

**Files:**

- Create: `cli/tui/knowledge.go`
- Create: `cli/tui/knowledge_test.go`
- Modify: `cli/tui/inspector.go`
- Modify: `cli/tui/app.go`

**Boundary produced:**

```go
type knowledgeDocument struct {
	Mount       string
	Path        string
	Title       string
	Content     string
	ContentHash string
	ModifiedAt  time.Time
}

type knowledgeMount struct {
	Name   string
	Root   string
	Medium coreio.Medium
}

type knowledgeScanner interface {
	Discover(mounts []knowledgeMount, maxBytes int64) core.Result
}

func knowledgeSystemMessage(attachments []attachmentRecord) string
```

- [x] Write `TestKnowledgeDiscover_Good` against `coreio.NewMemoryMedium`, finding nested `.md` and `.markdown` files, deriving titles from the first heading or basename, sorting by mount/title/path, and hashing exact content.
- [x] Write `TestKnowledgeDiscover_Bad`, include an over-limit file and assert it is rejected with a visible reason while valid documents remain.
- [x] Write `TestKnowledgeDiscover_Ugly`, include symlink loops, unreadable files, duplicate configured roots, and non-Markdown files; assert no loop, duplicate, or network activity.
- [x] Write `TestKnowledgeAttachment_Good`, attach a document to a session, persist its snapshot/hash, build a bounded system message, restart, and assert the same snapshot is used.
- [x] Write `TestKnowledgeAttachmentStale_Good`, change the source file and assert stale state without replacing the persisted snapshot.
- [x] Run tests and confirm failure.
- [x] Implement recursive scanning only through each mount's `List`, `Stat`, and `Read` methods plus `core.SHA256HexString`. Ignore symlink entries, deduplicate `(mount, path)`, restrict the default mount to the medium-relative `packs` root, and never clone or update the linked knowledge-packs repository. Additional configured local roots are each opened as their own sandboxed medium by composition code before being passed to the scanner.
- [x] Attach/detach through archived attachment records. Prepend active snapshots to the single system message before tool declarations, within the configured aggregate byte limit.
- [x] Re-run and commit:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/tui/knowledge.go cli/tui/knowledge_test.go cli/tui/inspector.go cli/tui/app.go
git commit -m "feat(tui): attach local knowledge packs"
```

## Task 15: Add Recoverable Session Export

**Files:**

- Create: `cli/tui/export.go`
- Create: `cli/tui/export_test.go`
- Modify: `cli/tui/palette.go`

**Boundary produced:**

```go
type exportFormat string

const (
	exportMarkdown exportFormat = "markdown"
	exportJSON     exportFormat = "json"
)

type sessionExporter interface {
	Export(medium coreio.Medium, directory, sessionID string, format exportFormat) core.Result
}
```

- [x] Write `TestExportSessionMarkdown_Good`, exporting metadata, visible/thought content according to preference, tool receipts, attachments, events, and artifacts to a collision-safe file below `exports/`.
- [x] Write `TestExportSessionJSON_Good`, export the same data as structured JSON and unmarshal it back into the export DTO.
- [x] Write `TestExportSessionSelectedMedium_Good`, inject a second in-memory medium as a user-selected destination and assert the export exists there while the default medium remains unchanged.
- [x] Write `TestExportSession_Bad`, make the exports path unwritable and assert the exact path/error is surfaced without modifying database records.
- [x] Write `TestExportSession_Ugly`, export two same-title sessions in one second and assert unique filenames.
- [x] Run tests and confirm failure.
- [x] Implement writes through the injected export medium: `WriteMode` a same-directory temporary name, then `Rename` it to the collision-safe final name. Filenames are UTC timestamp, title slug, short session ID, and extension. Export never archives or mutates the session.
- [x] Wire both formats into the command palette and persist an artifact record only after a successful write.
- [x] Re-run and commit:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/tui/export.go cli/tui/export_test.go cli/tui/palette.go
git commit -m "feat(tui): export durable chat sessions"
```

## Task 16: Integrate Bootstrap, Panels, Persistence, Models, and Shutdown

**Files:**

- Modify: `cli/tui/app.go`
- Modify: `cli/tui/app_test.go`
- Modify: `cli/tui/tui.go`
- Modify: `cli/tui/picker.go`
- Modify: `cli/tui/service.go`
- Modify: `cli/tui/engine_darwin_test.go`

The final `app` owns boot state, resources, sessions, jobs, one model lane, four panel models, overlays, inspector, runtime/knowledge/work adapters, an agent provider, and a root context cancel function. Resource-heavy startup runs as a Bubble Tea command and returns either `workspaceReadyMsg` or a blocking `workspaceFailedMsg`.

- [x] Replace the old transition test with focused app tests for booting, ready, degraded warnings, blocking storage failure, Retry, and Quit.
- [x] Add `TestAppSessionGeneration_Good`, using the real app update loop and fake model to send in session A, switch/create session B, send there, consume both streams, and assert isolated persisted transcripts plus hidden attention.
- [x] Add `TestAppSharedServiceLane_Good`, start HTTP service and a TUI job concurrently against the fake model, assert serial base concurrency, stop the service, and prove later chat still works.
- [x] Add `TestAppModelSwap_Bad`, asserting a model change is refused while any session job is active.
- [x] Add `TestAppModelSwap_Good`, asserting the service drains first, the old lane closes exactly once, one new model loads, and a failed load leaves a clear no-model state.
- [x] Add `TestAppQuit_Ugly`, with queued jobs, open service, state/repository resources, and capability providers; assert cancellation, listener stop, flush, close order, and no blocked goroutine.
- [x] Keep the existing `LTHN_PROBE_MODEL` live chat/service tests, updating them to sessions and the shared lane. They remain opt-in and do not replace deterministic tests.
- [x] Run the app tests and confirm failure before integration.
- [x] Make Chat the default primary panel even without a model; render a useful empty state with a Models shortcut. Model selection is disabled while jobs run. If service is running, model change first stops the listener, then closes the old lane, then loads one new model.
- [x] Persist user turn, assistant placeholder, job, deltas on flush ticks, final metrics/status, tool turns/events, session recent time, draft, viewport, and attention transitions.
- [x] Implement startup recovery: queued/generating jobs become interrupted; partial assistant content remains; no work restarts automatically.
- [x] Update `Run` and `--check` to construct the new app. `--check` performs a synchronous real workspace bootstrap, renders 100x30, closes resources, and returns nonzero on blocking storage failure.
- [x] Replace direct `os`, `filepath`, `strings`, `encoding/json`, `fmt`, `bytes`, and process imports in all touched production TUI files with core/go wrappers.
- [x] Re-run focused and package tests, then commit:

```sh
cd /Users/snider/Code/core/go-inference/cli
go test ./tui -count=1

cd /Users/snider/Code/core/go-inference
git add cli/tui/app.go cli/tui/app_test.go cli/tui/tui.go cli/tui/picker.go cli/tui/service.go cli/tui/engine_darwin_test.go
git commit -m "feat(tui): integrate the lem workspace"
```

## Task 17: Document, Audit, and Verify the Complete Slice

**Files:**

- Modify: `cli/tui/README.md`
- Modify: `docs/superpowers/specs/2026-07-17-lem-workspace-tui-design.md`
- Modify: `Taskfile.yml` only if the repository lacks an equivalent CLI QA command at implementation time.

- [x] Update the README with the four panels, responsive modes, complete key table, session/job semantics, inspector, service sharing, `~/.lem/` tree, medium boundary, knowledge snapshots, work/runtime limits, unavailable agent features, config recovery, export, and optional live-test environment.
- [x] Reconcile the design document against actual behaviour and dependency pins. Keep the complete `go/agent/*` capability implementation as the next slice, not a CLI dependency.
- [x] Run formatting and static analysis:

```sh
cd /Users/snider/Code/core/go-inference
gofmt -w cli/tui
test -z "$(gofmt -l cli/tui)"
git diff --check

cd /Users/snider/Code/core/go-inference/cli
go vet ./...
go test ./... -count=1
go test -race ./tui -count=1
```

- [x] Verify standalone nested-module resolution, proving no workspace-only sibling dependency slipped in:

```sh
cd /Users/snider/Code/core/go-inference/cli
GOWORK=off go mod tidy
GOWORK=off go test ./tui -count=1
```

- [x] Measure TUI coverage and inspect uncovered critical branches rather than reporting only an aggregate:

```sh
cd /Users/snider/Code/core/go-inference/cli
go test ./tui -covermode=atomic -coverprofile=/tmp/lem-tui-cover.out -count=1
go tool cover -func=/tmp/lem-tui-cover.out
```

- [x] Run the repository's portable gates and preserve the 95% root-package Codecov contract:

```sh
cd /Users/snider/Code/core/go-inference
task qa
task cover
```

- [x] Run the Markdown benchmark and one headless frame:

```sh
cd /Users/snider/Code/core/go-inference/cli
go test ./tui -run '^$' -bench '^BenchmarkMarkdownTranscript$' -benchmem -benchtime=20x
go run . tui --check
```

- [x] Run the core/go compliance audit and require every counter to be zero:

```sh
cd /Users/snider/Code/core/go-inference
bash /Users/snider/Code/core/go/tests/cli/v090-upgrade/audit.sh .
```

- [x] Confirm dependency and authority boundaries:

```sh
cd /Users/snider/Code/core/go-inference
if rg -n '^replace ' cli/go.mod; then exit 1; fi
if rg -n 'dappco\.re/go/agent' cli; then exit 1; fi
if rg -n '^\s*dappco\.re/go/agent' cli/go.mod; then exit 1; fi
if rg -n '\.(Build|Run|Stop|Exec|Pull)\(' cli/tui/runtime.go; then exit 1; fi
if rg -n '(git clone|http\.Get|https://github\.com/dAppCore/knowledge-packs)' cli/tui/knowledge.go; then exit 1; fi
if rg -n '(core\.Fs|NewUnrestricted)' cli/tui/paths.go cli/tui/preferences.go cli/tui/knowledge.go cli/tui/export.go; then exit 1; fi
```

- [x] Review `git diff --stat`, `git diff`, and `git status --short` for unrelated user changes, then commit documentation/QA wiring:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/tui/README.md docs/superpowers/specs/2026-07-17-lem-workspace-tui-design.md
# Also add cli/go.mod, cli/go.sum, and Taskfile.yml only when the final tidy/QA pass changed them intentionally.
git commit -m "docs(tui): document the lem workspace"
```

## Final Acceptance Checklist

- [x] The outer frame remains stable while switching Chat, Work, Models, and Service.
- [x] Wide, overlay, and narrow layouts pass boundary-width tests.
- [x] Settings, modes, tools, knowledge, model, service, work, and runtime detail are reachable through the inspector.
- [x] Multiple sessions persist and remain independently scrollable/searchable; switching never cancels a job.
- [x] Each session has at most one job; the single model lane serialises all TUI and HTTP work.
- [x] Hidden output and work questions produce attention markers without stealing focus.
- [x] Markdown, thinking, tool calls/results, metrics, cancellation, failure, and interrupted restart states are visible and durable.
- [x] `~/.lem/config.yaml`, `lem.duckdb`, `state.db`, `workspaces/`, `packs/`, and `exports/` are the only default application paths.
- [x] Malformed config is preserved; DuckDB failure blocks with Retry/Quit; state/integration failures degrade visibly.
- [x] The complete future agent feature catalogue is visible with honest unavailable states; the CLI has no CoreAgent dependency.
- [x] Config, knowledge, exports, and future workspace files use an injected `go-io` medium; only local SQL adapters receive host database paths.
- [x] go-container usage is read-only; knowledge usage is local-only; no execution authority is introduced.
- [x] Deterministic TUI tests, standalone module tests, race tests, root QA/coverage, benchmark, headless render, and the core/go audit all pass with fresh output.
