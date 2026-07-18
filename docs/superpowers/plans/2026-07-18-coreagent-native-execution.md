# CoreAgent Native Execution Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the LEM Work panel into a durable, Git-isolated native agent workspace that runs Codex, Claude, and OpenCode through CoreAgent-compatible queue policy and applies work to the user's checkout only after reviewed validation.

**Architecture:** Add reusable domain, queue, provider, Soft Serve, Git-workspace, and orchestration packages below `go/agent/*`; none imports the TUI. The nested CLI supplies the DuckDB persistence implementation and a private adapter to the existing `agentProvider` boundary, then adds focused Bubble Tea overlays for project editing, launch review, questions, change review, and acceptance. Root-module work lands and passes its gates before the CLI consumes a released module version.

**Tech Stack:** Go 1.26.2, `dappco.re/go` v0.11.0, `dappco.re/go/config` v0.18.0, `dappco.re/go/io` v0.15.1, `dappco.re/go/process` v0.16.1, `dappco.re/go/store` v0.14.1, DuckDB v2.10504.0, Charm Soft Serve v0.11.6, Git 2.x, Bubble Tea 1.3.10, Bubbles 1.0.0, and Lip Gloss 1.1.1 pseudo-version already pinned.

## Global Constraints

- The approved design is [`docs/superpowers/specs/2026-07-18-coreagent-native-execution-design.md`](../specs/2026-07-18-coreagent-native-execution-design.md). If implementation pressure changes its authority, lifecycle, persistence, or source-mutation rules, update and re-review the design first.
- Work in the existing `/Users/snider/Code/core/go-inference` checkout and `feat/lem-tui` branch. The user explicitly requested no additional worktree for this implementation.
- Port behaviour from `/Users/snider/Code/core/agent`; never import `dappco.re/go/agent`, invoke its CLI, or copy its filesystem-state protocol.
- Git is mandatory. Established repositories must be clean and on an attached branch; ad-hoc directories require a separately confirmed Git-initialisation review.
- Agents run only in `~/.lem/workspaces/<project-id>/runs/<run-id>/worktree`. No provider process receives the user's source checkout as its working directory.
- All LEM orchestration state, logs, questions, answers, queue state, backoff state, and acceptance receipts live in `~/.lem/lem.duckdb`. Never create LEM Markdown, JSON, YAML, or status files in source, execution, review, or integration worktrees.
- `~/.lem/agents.yaml` contains policy only. Missing policy uses documented defaults; malformed policy freezes dispatch and is never silently replaced.
- Soft Serve data is opaque upstream state below `~/.lem/soft-serve/`. Bind only `127.0.0.1`; acquire one owner lock; start on first project operation; stop after all recoverable branches and DuckDB records are durable.
- Pin `github.com/charmbracelet/soft-serve` to `v0.11.6`. Wrap its public APIs behind `go/agent/gitserver`; no Soft Serve type crosses that package boundary.
- Native processes are non-interactive. Use explicit argument vectors, a fixed worktree, a bounded shutdown grace period, and process-group termination. Never enable Codex, Claude, or OpenCode dangerous bypass flags by default.
- Because go-process v0.16.1 appends to the inherited environment, production launch uses an injected `env -i` executable with explicit `KEY=VALUE` arguments before the provider executable. Tests prove unlisted parent variables do not reach a fake provider.
- The default environment essentials are `PATH`, `HOME`, `USER`, `TMPDIR`, `TMP`, `TEMP`, `LANG`, `LC_ALL`, and `SHELL`; provider policy may add named credential variables, but never wildcard or inherit the complete parent environment.
- Logical command receipts show the provider executable and redacted arguments, not credential values or the `env -i` transport detail.
- Codex defaults to `codex --ask-for-approval never --sandbox workspace-write --cd <worktree> exec --json --color never`; Claude defaults to `claude --print --output-format stream-json --permission-mode acceptEdits --no-session-persistence`; OpenCode defaults to `opencode run --format json --pure --dir <worktree>`. Model flags are added only when non-empty.
- Retry and Resume create immutable child run records. A waiting run owns no live process; Answer persists text and Resume launches a new process on the same internal branch. If capture failure retained the parent worktree, the child reuses that checkout; it reconstructs only when the checkout is absent and refuses to create a second worktree for the same branch.
- Queue Stop disables admissions but does not cancel running work. LEM Close withdraws queued admissions, shuts down all process groups, captures recoverable Git work, marks unfinished runs interrupted, and joins every worker.
- Acceptance prepares and validates in a disposable internal worktree. Final confirmation rechecks source cleanliness, branch, and commit, then fetches and fast-forwards the source; conflicts, validation failure, or source movement leave it untouched.
- Production code uses `dappco.re/go` wrappers for formatting, errors, filesystem access, paths, JSON, strings, logging, bytes, and process execution. Public fallible functions return `core.Result`, and callers inspect `OK` before `Value`.
- Add tests and runnable examples beside their source. Exported symbols receive direct file-aware AX-7 Good/Bad/Ugly coverage named `Test<File>_<Symbol>_<Variant>`; hot paths receive sibling benchmarks. Do not add monolithic compliance files, versioned tests, or `ax7*` files.
- Every task starts with a failing behavioural test, proves the expected failure, implements the smallest contract, proves focused success, and commits only that task's files.
- Do not add a permanent `replace` directive. Before `cli/tui` imports the new root packages, publish or otherwise make the root module version available and update `cli/go.mod` normally; `GOWORK=off` standalone verification is mandatory at that checkpoint.

## Verified External Contracts

| Dependency | Verified contract used here |
| --- | --- |
| Soft Serve v0.11.6 | Go 1.25.8 module; `pkg/config.DefaultConfig`, `pkg/db.Open`, `pkg/db/migrate.Migrate`, `pkg/store/database.New`, `pkg/backend.New`, `pkg/backend.CreateRepository`, and `pkg/ssh.NewSSHServer` are public. |
| Codex CLI 0.144.3 | `exec --json`, `--color never`, `--model`, `--sandbox workspace-write`, `--ask-for-approval never`, and `--cd` are available. |
| Claude Code 2.1.211 | `--print --output-format stream-json`, `--permission-mode acceptEdits`, `--model`, `--session-id`, and `--no-session-persistence` are available. |
| OpenCode 1.16.2 | `run --format json --pure --dir`, `--model`, `--title`, and `--session` are available. |
| go-process v0.16.1 | `RunOptions` supports `Dir`, `Env`, `Detach`, `KillGroup`, and `GracePeriod`; output and exit events are broadcast through the Core action bus; supplied environment entries augment rather than replace inheritance. |

## Target File Map

Create these reusable root-module packages:

- `go/agent/work/work.go` plus sibling tests/examples/benchmark — immutable domain values, statuses, transitions, requests, snapshots, and capability names.
- `go/agent/queue/config.go` plus siblings — CoreAgent-compatible `agents.yaml` loading and validation.
- `go/agent/queue/controller.go` plus siblings — pure queue admission, concurrency, delay, backoff, freeze, and drain policy.
- `go/agent/provider/provider.go` plus siblings — adapter registry, logical command, detection, output, and launch contracts.
- `go/agent/provider/codex.go`, `claude.go`, `opencode.go` plus sibling tests/examples — exact native command adapters.
- `go/agent/provider/parser.go` plus siblings — JSONL event extraction, rate-limit recognition, and final LEM status envelope parsing.
- `go/agent/gitserver/gitserver.go` plus siblings — narrow service interface, options, repository receipt, owner-lock contract.
- `go/agent/gitserver/softserve.go` plus siblings — pinned loopback Soft Serve implementation.
- `go/agent/workspace/git.go` plus siblings — injected Git argv runner, source inspection, receipts, and SSH environment.
- `go/agent/workspace/workspace.go` plus siblings — registration, seeding, cached clone, run worktree, capture, and reconstruction.
- `go/agent/workspace/accept.go` plus siblings — diff review, integration replay, validation, and final fast-forward.
- `go/agent/orchestrator/store.go` plus siblings — persistence and clock/ID/process abstraction contracts.
- `go/agent/orchestrator/orchestrator.go` plus siblings — composition, capabilities, snapshots, queue ownership, recovery, and shutdown.
- `go/agent/orchestrator/run.go` plus siblings — prepare, dispatch, output, cancellation, waiting, retry, answer, and resume.
- `go/agent/orchestrator/accept.go` plus siblings — change review, accept, and reject operations.
- `go/agent/orchestrator/native.go` plus siblings — go-process adapter, action routing, `env -i`, redaction, and process joining.

Modify the root module and documentation:

- `go/go.mod`, `go/go.sum` — pin Soft Serve v0.11.6 and direct configuration dependencies actually imported.
- `docs/superpowers/specs/2026-07-18-coreagent-native-execution-design.md` — preserve the clarified queue/provider DuckDB records.

After the root release checkpoint, create or modify these CLI files:

- `cli/tui/agentstore.go`, `agentstore_test.go` — DuckDB implementation of `orchestrator.Store` sharing the current connection.
- `cli/tui/agentadapter.go`, `agentadapter_test.go` — bridge reusable snapshots/actions to the private TUI provider contract.
- `cli/tui/agentoverlay.go`, `agentoverlay_test.go` — Work editor, Git-enable review, launch review, answer, changes, validation, and final confirmation overlays.
- `cli/tui/records.go`, `records_test.go` — register agent record shapes only where ORM projection is useful.
- `cli/tui/migrations.go`, `migrations_test.go` — add agent schema migration version 2.
- `cli/tui/repository.go`, `repository_test.go` — expose the shared SQL connection safely and persist Work task/repository edits.
- `cli/tui/agentcap.go`, `agentcap_test.go` — add prepare/review request and result values without exposing root types to the view.
- `cli/tui/work.go`, `work_test.go` — edit durable Work, render provider events/logs without duplicate persistence, and contextualise actions.
- `cli/tui/palette.go`, `palette_test.go` — mirror only available contextual actions.
- `cli/tui/app.go`, `app_test.go` — asynchronous agent commands, overlay routing, refresh ticks, and lifecycle ownership.
- `cli/tui/bootstrap.go`, `bootstrap_test.go` — open the policy, store, Soft Serve/workspace/provider/orchestrator composition with degraded reasons.
- `cli/tui/paths.go`, `paths_test.go` — add `agents.yaml` and Soft Serve host paths while keeping medium-relative workspace paths.
- `cli/tui/tui.go`, `tui_test.go` — ensure `--check` never starts Soft Serve or a provider.
- `cli/tui/README.md` — document native execution, Git isolation, storage, queue controls, and acceptance.
- `cli/go.mod`, `cli/go.sum` — consume the released root module version without `replace`.

---

### Task 1: Establish the Agent Work Domain and State Machine

**Files:**

- Create: `go/agent/work/work.go`
- Create: `go/agent/work/work_test.go`
- Create: `go/agent/work/work_example_test.go`
- Create: `go/agent/work/work_bench_test.go`

**Interfaces:**

- Consumes: `core.Result`, `time.Time`.
- Produces:

```go
type RunStatus string

const (
	RunQueued      RunStatus = "queued"
	RunPreparing   RunStatus = "preparing"
	RunRunning     RunStatus = "running"
	RunWaiting     RunStatus = "waiting"
	RunCancelling  RunStatus = "cancelling"
	RunCancelled   RunStatus = "cancelled"
	RunFailed      RunStatus = "failed"
	RunCompleted   RunStatus = "completed"
	RunInterrupted RunStatus = "interrupted"
	RunAccepted    RunStatus = "accepted"
	RunRejected    RunStatus = "rejected"
)

type QueueStatus string
const (QueueFrozen QueueStatus = "frozen"; QueueAccepting QueueStatus = "accepting"; QueueDraining QueueStatus = "draining")

type Project struct { ID, SourcePath, RepositoryRoot, SourceBranch, SourceRevision, RepositoryName, ClonePath string; CreatedAt, UpdatedAt time.Time }
type Item struct { ID, ExternalID, Title, Task, Repository string }
type Run struct { ID, WorkID, ProjectID, ParentRunID, Provider, Model, SourceRevision, ExecutionRevision, AcceptedRevision, Branch, Worktree, CommandReceipt, FailureReason string; Status RunStatus; Number, Attempt, ProcessID, ExitCode int; QueuedAt, StartedAt, FinishedAt, UpdatedAt time.Time }
type Event struct { ID, RunID, WorkID, Kind, Title, Detail, DetailJSON string; CreatedAt time.Time }
type LogChunk struct { RunID string; Sequence int64; Stream, Text string; CreatedAt time.Time }
type Question struct { ID, RunID, Text string; CreatedAt time.Time }
type Answer struct { ID, QuestionID, ResumeRunID, Text string; CreatedAt time.Time }
type Acceptance struct { ID, WorkID, RunID, SourceBase, AgentBase, AgentTip, IntegrationBranch, IntegrationWorktree, ResultRevision, Status, ValidationJSON, FailureReason string; CreatedAt, UpdatedAt time.Time }
type ProviderState struct { Provider, BackoffReason, LastRunID string; BackoffUntil, LastStartedAt, WindowStartedAt, UpdatedAt time.Time; WindowAdmissions int }
type QueueState struct { ID string; Status QueueStatus; Reason string; UpdatedAt time.Time }
type Capability struct { Name string; Available bool; Reason string }
type Snapshot struct { Projects []Project; Runs []Run; Events []Event; Logs []LogChunk; Questions []Question; Acceptances []Acceptance; Queue QueueState; Providers []ProviderState }
type Continuation struct { Run Run; Logs []LogChunk; Question Question; Answer Answer }
type DispatchRequest struct { Work Item; Provider, Model string; ConfirmedSourceRevision string; UnsafeFlags []string }
type ResumeRequest struct { Work Item; ParentRunID, AnswerID, Provider, Model string }

func Transition(from, to RunStatus) core.Result
func ValidateDispatch(request DispatchRequest) core.Result
```

- [ ] Write `TestWork_Transition_Good`, `TestWork_Transition_Bad`, and `TestWork_Transition_Ugly`. Assert `queued -> preparing`, `running -> waiting`, `running -> interrupted`, and `completed -> accepted/rejected` succeed; terminal-to-running and empty statuses fail.
- [ ] Write `TestWork_ValidateDispatch_Good`, `_Bad`, and `_Ugly`, directly checking a complete request, missing task/repository/provider, and blank/whitespace values.
- [ ] Run the focused tests and confirm compilation fails because the package surface is absent:

```sh
cd /Users/snider/Code/core/go-inference/go
go test ./agent/work -run '^TestWork_(Transition|ValidateDispatch)_' -count=1
```

- [ ] Implement an explicit transition map. Shutdown recovery may move any non-terminal status to interrupted; user cancellation may move queued to cancelled and running to cancelling; waiting/completed/failed/cancelled/interrupted never transition back to queued because retry/resume creates a child record.
- [ ] Implement dispatch normalization without mutating the caller's slices. Return the normalized `DispatchRequest` in `core.Result.Value`.
- [ ] Add runnable examples for both functions and benchmark `Transition` through a package sink.
- [ ] Run the package tests, benchmark smoke, formatting, and commit:

```sh
cd /Users/snider/Code/core/go-inference/go
go test ./agent/work -count=1
go test ./agent/work -run '^$' -bench '^BenchmarkTransition$' -benchtime=20x -benchmem
cd ..
gofmt -w go/agent/work
git add go/agent/work
git commit -m "feat(agent): define durable work lifecycle"
```

### Task 2: Port CoreAgent-Compatible Policy and Pure Queue Admission

**Files:**

- Modify: `go/go.mod`
- Modify: `go/go.sum`
- Create: `go/agent/queue/config.go`
- Create: `go/agent/queue/config_test.go`
- Create: `go/agent/queue/config_example_test.go`
- Create: `go/agent/queue/controller.go`
- Create: `go/agent/queue/controller_test.go`
- Create: `go/agent/queue/controller_example_test.go`
- Create: `go/agent/queue/controller_bench_test.go`

**Interfaces:**

- Consumes: `work.Run`, `work.QueueState`, `work.ProviderState`, `dappco.re/go/config`, replaceable `go-io` medium.
- Produces:

```go
type Command struct { Command string `yaml:"command"`; Args []string `yaml:"args"` }
type DispatchConfig struct { DefaultAgent string `yaml:"default_agent"`; GlobalConcurrency, TimeoutMinutes int; Validation []Command `yaml:"validation"` }
type ConcurrencyLimit struct { Total int; Models map[string]int }
type RateConfig struct { ResetUTC string `yaml:"reset_utc"`; DailyLimit, MinDelay, SustainedDelay, BurstWindow, BurstDelay int }
type NativeConfig struct { Executable, DefaultModel string; CredentialEnv, Flags []string }
type Policy struct { Version int; Dispatch DispatchConfig; Concurrency map[string]ConcurrencyLimit; Rates map[string]RateConfig; Providers map[string]NativeConfig }

func LoadPolicy(medium coreio.Medium, path string) core.Result

type Candidate struct { RunID, Provider, Model string; QueuedAt time.Time }
type Runtime struct { Queued, Running []work.Run; Now time.Time }
type Decision struct { Allowed bool; Reason string; NotBefore time.Time }
type Controller struct { mu sync.Mutex; policy Policy; state work.QueueState; providers map[string]work.ProviderState }

func NewController(policy Policy, initial work.QueueState, providers []work.ProviderState) core.Result
func (controller *Controller) Start(at time.Time) core.Result
func (controller *Controller) Stop(active int, at time.Time) core.Result
func (controller *Controller) Decide(candidate Candidate, runtime Runtime) core.Result
func (controller *Controller) RecordStart(provider, runID string, at time.Time) core.Result
func (controller *Controller) RecordBackoff(provider, reason string, until, at time.Time) core.Result
```

- [ ] Copy the CoreAgent YAML fixture shape into test strings, including scalar `codex: 1`, nested `opencode: {total: 3, opencode-go/deepseek-v4-pro: 1}`, reset windows, delays, and the additive `dispatch.global_concurrency`/`dispatch.validation` fields.
- [ ] Add an optional additive `providers:` fixture with explicit executable, default model, credential environment names, and custom flags; reject empty/invalid environment names and preserve unsafe flags only for launch-review display.
- [ ] Write `TestConfig_LoadPolicy_Good`, `_Bad`, and `_Ugly`: valid scalar/nested policy succeeds, malformed YAML fails, and missing policy returns the documented conservative defaults without creating a file.
- [ ] Write controller AX-7 tests for global/provider/model limits, FIFO eligibility, frozen/accepting/draining, minimum/sustained/burst delays, persisted backoff, daily quota reset, and clock boundaries.
- [ ] Prove the tests fail before implementation:

```sh
cd /Users/snider/Code/core/go-inference/go
go test ./agent/queue -count=1
```

- [ ] Pin configuration and the current go-io medium before production imports them:

```sh
cd /Users/snider/Code/core/go-inference/go
go get dappco.re/go/config@v0.18.0
go get dappco.re/go/io@v0.15.1
```

- [ ] Implement `LoadPolicy` with `config.New(config.WithMedium(...), config.WithPath(...))`; decode scalar/nested concurrency through `map[string]any`, reject negative limits/delays, validate `HH:MM`, and deep-copy maps/slices before returning.
- [ ] Implement `Decide` as a pure check in the approved order: queue state, global count, provider count, model count, backoff, minimum/burst delay, daily quota, then FIFO order. The orchestrator rejects an unavailable provider before this call. Never sleep inside the controller.
- [ ] Implement Stop as `draining` when `active > 0`, otherwise `frozen`; a final run completion transitions draining to frozen through a second Stop call.
- [ ] Add examples and a steady-state admission benchmark; run tests, race, benchmark, format, and commit:

```sh
cd /Users/snider/Code/core/go-inference/go
go test ./agent/queue -count=1
go test -race ./agent/queue -count=1
go test ./agent/queue -run '^$' -bench '^BenchmarkController_Decide$' -benchtime=20x -benchmem
cd ..
gofmt -w go/agent/queue
git add go/go.mod go/go.sum go/agent/queue
git commit -m "feat(agent): port queue policy and admission"
```

### Task 3: Add Native Codex, Claude, and OpenCode Adapters

**Files:**

- Create: `go/agent/provider/provider.go`
- Create: `go/agent/provider/provider_test.go`
- Create: `go/agent/provider/provider_example_test.go`
- Create: `go/agent/provider/codex.go`
- Create: `go/agent/provider/codex_test.go`
- Create: `go/agent/provider/codex_example_test.go`
- Create: `go/agent/provider/claude.go`
- Create: `go/agent/provider/claude_test.go`
- Create: `go/agent/provider/claude_example_test.go`
- Create: `go/agent/provider/opencode.go`
- Create: `go/agent/provider/opencode_test.go`
- Create: `go/agent/provider/opencode_example_test.go`
- Create: `go/agent/provider/parser.go`
- Create: `go/agent/provider/parser_test.go`
- Create: `go/agent/provider/parser_example_test.go`
- Create: `go/agent/provider/parser_bench_test.go`

**Interfaces:**

```go
type Launch struct { WorkID, RunID, Title, Task, Worktree, Branch, Model, Continuation string; UnsafeFlags []string }
type Command struct { Provider, Executable string; Args, Environment, CredentialKeys []string; Receipt string }
type Detection struct { Provider, Executable, Version string; Available bool; Reason string }
type Output struct { Kind, Text, DetailJSON, RetryAfter, UsageJSON string }
type FinalStatus struct { Status, Summary, Question, Reason string }
type Finder interface { Find(name string) core.Result; Version(context.Context, string) core.Result }
type Adapter interface { Name() string; Detect(context.Context) core.Result; Build(Launch) core.Result; ParseLine(stream, line string) []Output }
type Config struct { Executable, DefaultModel string; CredentialEnv, Flags []string }
type Registry struct { adapters map[string]Adapter; names []string }

func NewRegistry(adapters ...Adapter) core.Result
func DefaultRegistry(finder Finder, configs map[string]Config) core.Result
func ParseFinalStatus(text string) core.Result
```

- [ ] Write command tests that compare complete argument slices and prove prompts are single arguments rather than shell text. Include models with punctuation and worktree paths containing spaces.
- [ ] Assert no default command contains `dangerously-bypass`, `dangerously-skip`, `bypassPermissions`, `--add-dir`, source checkout paths, or credentials.
- [ ] Assert Codex, Claude, and OpenCode receive the exact safe defaults in Global Constraints, and unsafe custom flags appear only when explicitly supplied.
- [ ] Write parser fixtures for representative JSONL text/progress/usage events, malformed JSON, stdout/stderr, rate-limit text with and without retry duration, and these exact envelopes:

```text
<<<LEM_STATUS>>>{"status":"completed","summary":"tests pass"}<<<END_LEM_STATUS>>>
<<<LEM_STATUS>>>{"status":"waiting","question":"Which API should be canonical?"}<<<END_LEM_STATUS>>>
<<<LEM_STATUS>>>{"status":"failed","reason":"validation failed"}<<<END_LEM_STATUS>>>
```

- [ ] Prove missing/malformed envelopes never create waiting questions; zero/non-zero exit classification remains the orchestrator's responsibility.
- [ ] Run the package and observe failure before implementation.
- [ ] Implement adapters with `coreprocess.Program.Find`, version probes, deep-copied args, credential-key allowlists, and redacted receipts. Prompt construction includes task, branch/worktree context, continuation context, commit guidance, and the final envelope contract.
- [ ] Implement tolerant JSON parsing by extracting provider text/event fields and retaining the raw redacted line as fallback. Parse rate-limit deadlines only when valid; otherwise emit a rate-limit event without inventing a retry time.
- [ ] Add runnable examples and parser benchmark; run tests/race/benchmark, format, and commit:

```sh
cd /Users/snider/Code/core/go-inference/go
go test ./agent/provider -count=1
go test -race ./agent/provider -count=1
go test ./agent/provider -run '^$' -bench '^BenchmarkParse' -benchtime=20x -benchmem
cd ..
gofmt -w go/agent/provider
git add go/agent/provider
git commit -m "feat(agent): add native provider adapters"
```

### Task 4: Wrap a Private Loopback Soft Serve Service

**Files:**

- Modify: `go/go.mod`
- Modify: `go/go.sum`
- Create: `go/agent/gitserver/gitserver.go`
- Create: `go/agent/gitserver/gitserver_test.go`
- Create: `go/agent/gitserver/gitserver_example_test.go`
- Create: `go/agent/gitserver/softserve.go`
- Create: `go/agent/gitserver/softserve_test.go`
- Create: `go/agent/gitserver/softserve_example_test.go`

**Interfaces:**

```go
type Options struct { DataPath, ListenAddress, PublicURL string; ShutdownTimeout time.Duration; PID func() int; ProcessAlive func(int) bool }
type Repository struct { Name, CloneURL, IdentityFile, KnownHostsFile string }
type Health struct { Running bool; Address, Reason string }
type Service interface { Start(context.Context) core.Result; EnsureRepository(context.Context, string) core.Result; Health(context.Context) core.Result; Close() core.Result }

func DefaultOptions(dataPath string) core.Result
func NewSoftServe(options Options) core.Result
```

- [ ] Pin and tidy only after imports exist:

```sh
cd /Users/snider/Code/core/go-inference/go
go get github.com/charmbracelet/soft-serve@v0.11.6
```

- [ ] Write AX-7 tests for default loopback configuration, path validation, nil options, repository-name sanitisation, idempotent start/ensure/close, owner-lock contention, stale-owner recovery, port collision, and a second process receiving an ownership reason.
- [ ] Use a temporary data directory and loopback port fixture. Create a private repository through `backend.CreateRepository`, push a fixture commit over SSH, clone it back, and assert no listener binds a non-loopback address.
- [ ] Prove tests fail before implementation.
- [ ] Implement one owner receipt at `<data>/owner.lock` using `core.OpenFile(..., core.O_CREATE|core.O_EXCL|core.O_WRONLY, 0o600)`. The receipt contains only PID and start time; stale recovery requires `ProcessAlive(pid) == false` before replacement.
- [ ] Configure Soft Serve directly: absolute `DataPath`, SQLite database under that path, SSH enabled at `127.0.0.1:23231` by default, Git/HTTP/stats/LFS disabled, and restrictive directory/key modes. Generate the Ed25519 client key before final config validation, then use its public key as the sole initial admin key.
- [ ] Open/migrate Soft Serve's database, create database store/backend contexts, call `backend.SetAllowKeyless(ctx, false)` and `backend.SetAnonAccess(ctx, access.NoAccess)`, construct only `pkg/ssh.NewSSHServer`, and use the backend directly for private repository creation. Do not import `cmd/soft/serve` or start its cron/HTTP/stats services.
- [ ] Bind the TCP listener inside the wrapper and call `SSHServer.Serve(listener)` in one joined goroutine. This makes `127.0.0.1:0` deterministic in tests and lets the returned clone URL use the listener's actual port.
- [ ] Build `ssh/known_hosts` from the generated Soft Serve host public key and actual `[127.0.0.1]:<port>` authority before reporting healthy, so Git uses `StrictHostKeyChecking=yes` from the first push.
- [ ] Produce SSH clone URLs without credentials. Supply identity and known-host paths separately so workspace Git commands use `GIT_SSH_COMMAND` without changing global SSH configuration.
- [ ] Close SSH, wait for its goroutine, close Soft Serve DB, then remove the owner receipt only if this instance owns it.
- [ ] Run focused tests/race, verify no unexpected files outside the fixture, format, tidy, and commit:

```sh
cd /Users/snider/Code/core/go-inference/go
go test ./agent/gitserver -count=1
go test -race ./agent/gitserver -count=1
go mod tidy
cd ..
gofmt -w go/agent/gitserver
git add go/go.mod go/go.sum go/agent/gitserver
git commit -m "feat(agent): embed private soft serve git"
```

### Task 5: Build Git Registration, Run Worktrees, and Capture

**Files:**

- Create: `go/agent/workspace/git.go`
- Create: `go/agent/workspace/git_test.go`
- Create: `go/agent/workspace/git_example_test.go`
- Create: `go/agent/workspace/workspace.go`
- Create: `go/agent/workspace/workspace_test.go`
- Create: `go/agent/workspace/workspace_example_test.go`
- Create: `go/agent/workspace/workspace_bench_test.go`

**Interfaces:**

```go
type Command struct { Dir, Executable string; Args, Environment []string }
type Runner interface { Run(context.Context, Command) core.Result }
type SourceReview struct { Path, Root, Branch, ProposedBranch, Revision, CommitIdentity string; Git, Clean, Detached bool; Included []string; IncludedHash string }
type RegisterRequest struct { ProjectID, SourcePath, RepositoryName string; EnableGit, Confirmed bool; ExpectedIncludedHash string }
type RunWorkspace struct { Project work.Project; RunID, Branch, Path, BaseRevision string }
type Capture struct { Revision string; Changed, Pushed, Retained bool; Summary string }
type ManagerOptions struct { Root string; Files coreio.Medium; Git Runner; Server gitserver.Service; IDs func() string; Now func() time.Time }
type Manager struct { root string; files coreio.Medium; git Runner; server gitserver.Service; ids func() string; now func() time.Time; mu sync.Mutex; leases map[string]RunWorkspace }

func NewManager(options ManagerOptions) core.Result
func (manager *Manager) ReviewSource(context.Context, string) core.Result
func (manager *Manager) Register(context.Context, RegisterRequest) core.Result
func (manager *Manager) PrepareRun(context.Context, work.Project, work.Run) core.Result
func (manager *Manager) CaptureRun(context.Context, RunWorkspace) core.Result
func (manager *Manager) ReconstructRun(context.Context, work.Project, work.Run) core.Result
func (manager *Manager) ReleaseRun(context.Context, RunWorkspace) core.Result
```

- [ ] Implement tests with real temporary Git repositories and an in-memory fake `gitserver.Service`: existing clean repo, dirty repo, detached HEAD, nested selected path, ad-hoc included-file review, review/hash drift, missing identity, pre-existing remotes, path with spaces, ignored files, cancellation capture, and failed push retention.
- [ ] Assert registration never adds/deletes/rewrites source remotes and never creates LEM control files. Snapshot every tracked/untracked filename before and after established-repository registration.
- [ ] Assert ad-hoc Git enablement fails unless `Confirmed` and `ExpectedIncludedHash` match the fresh review; repository-local LEM identity is displayed and used only when no identity resolves.
- [ ] Assert cached clone and run paths are exactly `<root>/<project-id>/repo.git` and `<root>/<project-id>/runs/<run-id>/worktree`, and branch names are deterministic `lem/work/<escaped-work-id>/run-<number>`.
- [ ] Prove tests fail before implementation.
- [ ] Implement the production runner with `dappco.re/go/process.RunWithOptions`. Every Git call is an explicit argv vector; SSH calls receive `GIT_SSH_COMMAND=ssh -i ... -o IdentitiesOnly=yes -o UserKnownHostsFile=... -o StrictHostKeyChecking=yes`.
- [ ] Require a replaceable go-io medium rooted at the internal workspace root. Use it for internal directory checks and receipts; create a separate sandboxed local medium rooted at the reviewed source only while hashing an ad-hoc inclusion list. Git still receives resolved local paths, so a future remote medium adapter does not masquerade as a local checkout.
- [ ] `ReviewSource` uses `git rev-parse --show-toplevel`, `git symbolic-ref --short HEAD`, `git rev-parse HEAD`, `git status --porcelain=v1`, and `git ls-files --cached --others --exclude-standard`; hash the sorted included list plus content hashes.
- [ ] Registration starts Soft Serve, ensures the private repo, pushes the exact source revision through an explicit URL, creates/fetches a mirror clone without touching source remotes, and returns `work.Project`.
- [ ] Run preparation fetches internal refs, creates the branch and worktree, and refuses any path outside the configured root. Capture stages non-ignored changes, creates a labelled commit when needed, pushes, and retains the worktree whenever push/capture is not durably recoverable.
- [ ] Add examples and path/branch benchmark; run focused tests/race/benchmark, format, and commit:

```sh
cd /Users/snider/Code/core/go-inference/go
go test ./agent/workspace -count=1
go test -race ./agent/workspace -count=1
go test ./agent/workspace -run '^$' -bench '^Benchmark' -benchtime=20x -benchmem
cd ..
gofmt -w go/agent/workspace
git add go/agent/workspace
git commit -m "feat(agent): isolate work in git worktrees"
```

### Task 6: Define the Orchestrator Store and Native Process Boundaries

**Files:**

- Create: `go/agent/orchestrator/store.go`
- Create: `go/agent/orchestrator/store_test.go`
- Create: `go/agent/orchestrator/store_example_test.go`
- Create: `go/agent/orchestrator/native.go`
- Create: `go/agent/orchestrator/native_test.go`
- Create: `go/agent/orchestrator/native_example_test.go`

**Interfaces:**

```go
type Commit struct {
	Project *work.Project
	Run *work.Run
	ExpectedStatus *work.RunStatus
	CreateRun bool
	Event *work.Event
	Logs []work.LogChunk
	Question *work.Question
	Answer *work.Answer
	Acceptance *work.Acceptance
	Queue *work.QueueState
	Provider *work.ProviderState
}

type Store interface {
	Recover(time.Time) core.Result
	Commit(Commit) core.Result
	Project(string) core.Result
	ProjectBySource(string) core.Result
	Run(string) core.Result
	NextRunNumber(string) core.Result
	Continuation(string) core.Result
	Snapshot(string) core.Result
}

type Process interface { ID() string; PID() int; Wait() core.Result; Shutdown() core.Result }
type Launcher interface { DetectEnvironment([]string) core.Result; Start(context.Context, provider.Command, func(stream, line string)) core.Result; Close() core.Result }
type Clock interface { Now() time.Time }
type Identifier interface { New() string }

func NewNativeLauncher(service *coreprocess.Service, essentials []string) core.Result
```

- [ ] Create contract fakes in the tests and assert empty commits, duplicate creates, stale expected statuses, empty IDs, invalid log sequences, and recovery errors propagate without reading invalid `Result.Value`. `CreateRun` requires a non-nil Run and nil `ExpectedStatus`; an existing-run update requires non-nil `ExpectedStatus`; related Event, Log, Question, Answer, Acceptance, Queue, and Provider values must carry valid IDs/sequences before the Store is called.
- [ ] Assert one `Commit` atomically covers queued run plus event, running transition plus provider start state, waiting transition plus question, progress event plus provider backoff, answer, log batch, and accepted/rejected transition plus acceptance receipt.
- [ ] Create fake provider executables that echo selected environment keys, stream stdout/stderr, spawn a child, ignore TERM, and exit 0/non-zero. Assert only configured essentials and credential keys survive `env -i`, output order is preserved, and Shutdown kills the whole group.
- [ ] Assert secrets are redacted from receipts/errors and never appear in output events generated by the launcher.
- [ ] Prove focused tests fail before implementation.
- [ ] Implement a single go-process Core/service and action subscriber. Route raw output/exit events by process ID, buffer events that arrive between process start and callback registration, batch no state here, set `Detach: true`, `KillGroup: true`, `GracePeriod: 3*time.Second`, and explicitly Shutdown/Wait every tracked process on Close.
- [ ] Resolve a platform `env` program, build `env -i KEY=VALUE ... <provider> <args...>`, and retain the logical provider command for review. Reject malformed environment names and credential values containing NUL/newline.
- [ ] Run tests and race, format, and commit:

```sh
cd /Users/snider/Code/core/go-inference/go
go test ./agent/orchestrator -run '^(TestStore|TestNative)' -count=1
go test -race ./agent/orchestrator -run '^(TestStore|TestNative)' -count=1
cd ..
gofmt -w go/agent/orchestrator
git add go/agent/orchestrator/store.go go/agent/orchestrator/store_test.go go/agent/orchestrator/store_example_test.go go/agent/orchestrator/native.go go/agent/orchestrator/native_test.go go/agent/orchestrator/native_example_test.go
git commit -m "feat(agent): define durable execution boundaries"
```

### Task 7: Implement Dispatch, Streaming, Queueing, and Recovery

**Files:**

- Create: `go/agent/orchestrator/orchestrator.go`
- Create: `go/agent/orchestrator/orchestrator_test.go`
- Create: `go/agent/orchestrator/orchestrator_example_test.go`
- Create: `go/agent/orchestrator/run.go`
- Create: `go/agent/orchestrator/run_test.go`
- Create: `go/agent/orchestrator/run_example_test.go`
- Create: `go/agent/orchestrator/run_bench_test.go`

**Interfaces:**

```go
type Options struct { Store Store; GitServer gitserver.Service; Workspaces *workspace.Manager; Providers *provider.Registry; Queue *queue.Controller; Launcher Launcher; Clock Clock; IDs Identifier; LogBatchBytes int; LogBatchDelay time.Duration }
type ProjectReview struct { Work work.Item; Source workspace.SourceReview; RepositoryName string; RequiresGitEnable bool }
type DispatchReview struct { Request work.DispatchRequest; Project work.Project; Source workspace.SourceReview; Detection provider.Detection; Command provider.Command; Queue queue.Decision; WorktreePath, Warning string }
type Orchestrator struct { store Store; gitServer gitserver.Service; workspaces *workspace.Manager; providers *provider.Registry; queue *queue.Controller; launcher Launcher; clock Clock; ids Identifier; ctx context.Context; cancel context.CancelFunc; mu sync.Mutex; runs map[string]Process; wake chan struct{}; workers sync.WaitGroup; closed bool }

func New(options Options) core.Result
func (orchestrator *Orchestrator) Capabilities() []work.Capability
func (orchestrator *Orchestrator) Snapshot(context.Context, string) core.Result
func (orchestrator *Orchestrator) ReviewProject(context.Context, work.Item) core.Result
func (orchestrator *Orchestrator) RegisterProject(context.Context, ProjectReview, bool) core.Result
func (orchestrator *Orchestrator) ReviewDispatch(context.Context, work.DispatchRequest) core.Result
func (orchestrator *Orchestrator) Dispatch(context.Context, DispatchReview) core.Result
func (orchestrator *Orchestrator) Cancel(context.Context, string) core.Result
func (orchestrator *Orchestrator) StartQueue(context.Context) core.Result
func (orchestrator *Orchestrator) StopQueue(context.Context) core.Result
func (orchestrator *Orchestrator) Close() core.Result
```

- [x] Write orchestration tests for existing-project lookup by canonical source, non-mutating project review, separately confirmed ad-hoc Git enablement, unavailable providers, dirty/moved source after review, durable queued record before admission, persistence failure before launch, concurrency order, backoff, Queue Stop drain, Queue Start, live output batches, malformed output, zero/non-zero exits, cancellation grace/escalation, capture failure retention, and idempotent Close.
- [x] Use channel-controlled fakes to prove no child starts until the `running` transition is durable and no goroutine/process remains after Close.
- [x] Prove startup `Recover` marks queued/preparing/running/cancelling interrupted and starts the queue frozen without automatic launch.
- [x] Prove focused tests fail before implementation.
- [x] Implement `ReviewProject` and `RegisterProject` as the sole bridge to workspace registration. `ReviewDispatch` remains non-mutating source/provider/command/queue inspection and requires a registered project; launch review carries the exact source revision and redacted command.
- [x] Implement Dispatch by revalidating the review, atomically committing the queued run/event, signalling the queue loop, preparing the worktree only after admission, atomically committing preparing/running transitions and provider start state, then launching.
- [x] Route each raw launcher line through the selected provider adapter. Batch log chunks at the first of configured byte/time thresholds with monotonic sequence numbers. Persist provider events and parsed backoff immediately; on DuckDB failure cancel the process and fail the run.
- [x] On exit, flush logs, classify final envelope plus exit code, capture/push Git, save waiting question only from a valid waiting envelope, and release only durably recoverable worktrees. A zero exit without a valid envelope completes with an `unclassified provider finish` event; a non-zero exit fails even when the envelope is missing.
- [x] Implement Close in the approved order and combine errors without abandoning later cleanup.
- [x] Add examples and snapshot benchmark; run package/race/benchmark, format, and commit:

```sh
cd /Users/snider/Code/core/go-inference/go
go test ./agent/orchestrator -count=1
go test -race ./agent/orchestrator -count=1
go test ./agent/orchestrator -run '^$' -bench '^BenchmarkOrchestrator_Snapshot$' -benchtime=20x -benchmem
cd ..
gofmt -w go/agent/orchestrator
git add go/agent/orchestrator
git commit -m "feat(agent): orchestrate native agent runs"
```

### Task 8: Add Answer, Resume, Retry, and Immutable Attempts

**Files:**

- Modify: `go/agent/orchestrator/run.go`
- Modify: `go/agent/orchestrator/run_test.go`
- Modify: `go/agent/orchestrator/run_example_test.go`

**Interfaces added:**

```go
func (orchestrator *Orchestrator) Answer(context.Context, runID, text string) core.Result
func (orchestrator *Orchestrator) Resume(context.Context, work.ResumeRequest) core.Result
func (orchestrator *Orchestrator) Retry(context.Context, work.Item, parentRunID string) core.Result
```

- [x] Write direct `TestRun_Orchestrator_{Answer,Resume,Retry}_{Good,Bad,Ugly}` AX-7 tests: Answer only waiting runs with an unanswered question; Resume requires a stored answer; Retry accepts failed/cancelled/interrupted only; empty text/IDs fail; duplicate actions do not rewrite the parent.
- [x] Assert each action creates a new queued run with `ParentRunID`, preserves the parent's `Number` and exact internal `Branch`, increments `Attempt`, preserves the parent status/timestamps, and builds continuation text from `Store.Continuation(parentRunID)` containing the earlier task/output/question/answer. A fresh root dispatch obtains its branch number from `Store.NextRunNumber(workID)`.
- [x] Assert Resume/Retry reuses a retained parent worktree after confirming its branch and internal root, reconstructs from Soft Serve only when the checkout is absent, and returns a durable recovery reason when a retained checkout is corrupt or on the wrong branch. Never attach the same internal branch to a second worktree.
- [x] Assert no question/answer/status file appears in either repository.
- [x] Run the focused tests and see the missing methods fail.
- [x] Commit the answer separately from launch but atomically as one Store commit. Resume/Retry build fresh provider commands and enter the same Dispatch queue path without mutating terminal attempts.
- [x] Run package/race, format, and commit:

```sh
cd /Users/snider/Code/core/go-inference/go
go test ./agent/orchestrator -run '^TestRun_Orchestrator_(Answer|Resume|Retry)_' -count=1
go test -race ./agent/orchestrator -run '^TestRun_Orchestrator_(Answer|Resume|Retry)_' -count=1
cd ..
gofmt -w go/agent/orchestrator
git add go/agent/orchestrator/run.go go/agent/orchestrator/run_test.go go/agent/orchestrator/run_example_test.go
git commit -m "feat(agent): resume immutable agent attempts"
```

### Task 9: Implement Review, Validation, Acceptance, and Rejection

**Files:**

- Create: `go/agent/workspace/accept.go`
- Create: `go/agent/workspace/accept_test.go`
- Create: `go/agent/workspace/accept_example_test.go`
- Create: `go/agent/orchestrator/accept.go`
- Create: `go/agent/orchestrator/accept_test.go`
- Create: `go/agent/orchestrator/accept_example_test.go`
- Modify: `go/agent/orchestrator/orchestrator.go`
- Modify: `go/agent/queue/controller.go`
- Modify: `go/agent/queue/controller_test.go`
- Modify: `go/agent/queue/controller_example_test.go`

**Interfaces:**

```go
type ChangeReview struct { WorkID, RunID, SourceBranch, SourceRevision, AgentBase, AgentTip, IntegrationBranch, IntegrationPath, ResultRevision, Diff, CommitLog string; Validation []ValidationResult; Conflicts []string }
type ValidationResult struct { Command Command; ExitCode int; Output, Receipt string; Passed bool }
type AcceptRequest struct { Review ChangeReview; Project work.Project; Confirmed bool }

func (manager *Manager) ReviewChanges(context.Context, work.Project, work.Run, []queue.Command) core.Result
func (manager *Manager) Apply(context.Context, AcceptRequest) core.Result
func (manager *Manager) Reject(context.Context, ChangeReview) core.Result

func (orchestrator *Orchestrator) ReviewChanges(context.Context, string) core.Result
func (orchestrator *Orchestrator) Accept(context.Context, workspace.AcceptRequest) core.Result
func (orchestrator *Orchestrator) Reject(context.Context, string) core.Result
```

- [x] Write real-Git tests for unchanged source fast-forward, diverged source replay, multi-commit order, merge conflict, validation pass/fail, no validation acknowledgement, source movement after review, dirty source after review, rejected history retention, restart recovery, concurrent decisions, and idempotent acceptance receipt.
- [x] Snapshot source branch, HEAD, index, worktree, remotes, and refs before each conflict/failure test; assert they remain unchanged.
- [x] Prove tests fail before implementation.
- [x] Review fetches source into the cached internal repository, creates a disposable integration worktree, fast-forwards when possible or replays the exact agent range, records conflicts, runs validation argv commands there, and returns the complete receipt without touching source.
- [x] Persist every complete review as canonical `ChangeReview` JSON plus a `changes_reviewed` event in one Store commit, with status `prepared`, `conflicted`, or `validation_failed`, while leaving the completed Run unchanged. Accept/Reject load the latest durable review after restart and reject caller tampering or superseded receipts.
- [x] Apply requires `Confirmed`, replaces the transient `AcceptRequest.Project` with the Store-owned Project, reconstructs and verifies the internal Git receipt, rechecks source branch/HEAD/cleanliness, fetches the validated result only after confirmation, and advances with `git merge --ff-only`. If any recheck differs, return a stale-review failure and prepare nothing automatically.
- [x] Reject records the decision and retains internal refs/worktrees under normal retention.
- [x] One Store commit transactionally writes the final acceptance receipt/event and moves the run from completed to accepted; rejection does the same from completed to rejected. Neither operation deletes the internal branch or prior run/event/log records.
- [x] Run workspace/orchestrator/queue tests and race, format, and commit:

```sh
cd /Users/snider/Code/core/go-inference/go
go test ./agent/workspace ./agent/orchestrator ./agent/queue -count=1
go test -race ./agent/workspace ./agent/orchestrator ./agent/queue -count=1
cd ..
gofmt -w go/agent/workspace go/agent/orchestrator go/agent/queue
git add go/agent/workspace/accept.go go/agent/workspace/accept_test.go go/agent/workspace/accept_example_test.go go/agent/orchestrator/accept.go go/agent/orchestrator/accept_test.go go/agent/orchestrator/accept_example_test.go go/agent/orchestrator/orchestrator.go go/agent/queue/controller.go go/agent/queue/controller_test.go go/agent/queue/controller_example_test.go
git commit -m "feat(agent): validate and accept agent changes"
```

### Task 10: Pass the Root Module Release Checkpoint

**Files:**

- Modify when verified: `go/go.mod`
- Modify when verified: `go/go.sum`
- Modify: `go/engine/metal/composed_attn_core_backend.go`
- Modify: `go/engine/metal/composed_backend.go`
- Modify: `go/engine/metal/composed_bf16_backend.go`
- Modify: `go/engine/metal/composed_bf16_backend_test.go`
- Modify: `go/engine/metal/composed_chain_icb.go`
- Modify: `go/engine/metal/composed_stateful_session_test.go`
- Modify: `go/engine/metal/lthn_gated_delta.go`
- Modify: `go/engine/metal/lthn_gated_delta_test.go`
- Modify: `go/model/composed/composed.go`
- Modify: `docs/superpowers/plans/2026-07-18-coreagent-native-execution.md`

**Interfaces:**

- Consumes: Tasks 1–9.
- Produces: a tag-ready `dappco.re/go/inference` v0.14.0 release candidate for the nested CLI without a local replacement. Publication remains an explicit owner action.

- [x] Run focused portable packages, then the repository gates with fresh output. Record the root audit as the existing migration baseline rather than attributing it to the native-agent change:

```sh
cd /Users/snider/Code/core/go-inference/go
go test ./agent/work ./agent/queue ./agent/provider ./agent/gitserver ./agent/workspace ./agent/orchestrator -count=1
go test -race ./agent/work ./agent/queue ./agent/provider ./agent/gitserver ./agent/workspace ./agent/orchestrator -count=1
GOWORK=off go mod tidy
GOWORK=off go test ./agent/work ./agent/queue ./agent/provider ./agent/gitserver ./agent/workspace ./agent/orchestrator -count=1
cd ..
task qa
bash /Users/snider/Code/core/go/tests/cli/v090-upgrade/audit.sh go
```

- [x] Confirm the six native-agent packages have zero attributed core/go code-shape findings, no repository contains LEM control files, and `git diff --check` is clean. Root-wide legacy migration findings remain separate work.
- [x] Review the dependency size introduced by Soft Serve and prepare the root-module tag `go/v0.14.0`, which is the version `cli/go.mod` will consume. Dependency licensing is project-owner managed and is not an agent release gate.
- [x] Stop before creating or pushing `go/v0.14.0` unless the user explicitly authorises publication. Once that version is actually available, verify it outside the workspace:

```sh
release_probe=$(mktemp -d)
cd "$release_probe"
go mod init lem-agent-release-check
go get dappco.re/go/inference/agent/orchestrator@v0.14.0
go test ./...
```

- [x] Commit the verified checkpoint changes. Run and record the external probe after publication:

```sh
cd /Users/snider/Code/core/go-inference
git add go/go.mod go/go.sum docs/superpowers/plans/2026-07-18-coreagent-native-execution.md go/engine/metal/composed_attn_core_backend.go go/engine/metal/composed_backend.go go/engine/metal/composed_bf16_backend.go go/engine/metal/composed_bf16_backend_test.go go/engine/metal/composed_chain_icb.go go/engine/metal/composed_stateful_session_test.go go/engine/metal/lthn_gated_delta.go go/engine/metal/lthn_gated_delta_test.go go/model/composed/composed.go
git commit -m "chore(agent): prepare native runtime release"
```

### Task 11: Add the DuckDB Agent Schema and Store

**Files:**

- Modify: `cli/go.mod`
- Modify: `cli/go.sum`
- Modify: `cli/tui/migrations.go`
- Modify: `cli/tui/migrations_test.go`
- Modify: `cli/tui/repository.go`
- Modify: `cli/tui/repository_test.go`
- Create: `cli/tui/agentstore.go`
- Create: `cli/tui/agentstore_test.go`

**Interfaces:**

- Consumes: released `dappco.re/go/inference/agent/work` and `agent/orchestrator.Store` from v0.14.0.
- Produces: private `duckAgentStore` sharing `workspaceDatabase.store.Conn()`.

```go
func newDuckAgentStore(repository workspaceRepository) core.Result
```

- [x] Update `cli/go.mod` normally; never add `replace`:

```sh
cd /Users/snider/Code/core/go-inference/cli
go get dappco.re/go/inference@v0.14.0
```

- [x] Add migration version 2 with exact tables: `agent_projects`, `agent_runs`, `agent_events`, `agent_log_chunks`, `agent_questions`, `agent_answers`, `agent_acceptances`, `agent_queue_state`, and `agent_provider_state`. Add unique `(run_id, sequence)` log ordering, run/work/status indexes, event ordering indexes, and one queue-state check constraint.
- [x] Use this exact SQL inventory, split into individually executed statements inside the migration transaction:

```sql
CREATE TABLE agent_projects (id TEXT PRIMARY KEY, source_path TEXT NOT NULL UNIQUE, repository_root TEXT NOT NULL, source_branch TEXT NOT NULL, source_revision TEXT NOT NULL, repository_name TEXT NOT NULL UNIQUE, clone_path TEXT NOT NULL, created_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT NULL);
CREATE TABLE agent_runs (id TEXT PRIMARY KEY, work_id TEXT NOT NULL, project_id TEXT NOT NULL, parent_run_id TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, source_revision TEXT NOT NULL, durable_revision TEXT NOT NULL DEFAULT '', execution_revision TEXT NOT NULL, accepted_revision TEXT NOT NULL, branch TEXT NOT NULL, worktree TEXT NOT NULL, command_receipt TEXT NOT NULL, run_number INTEGER NOT NULL, attempt INTEGER NOT NULL, process_id BIGINT NOT NULL, status TEXT NOT NULL, exit_code INTEGER NOT NULL, failure_reason TEXT NOT NULL, queued_at TIMESTAMP NOT NULL, started_at TIMESTAMP NOT NULL, finished_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT NULL);
CREATE TABLE agent_events (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, work_id TEXT NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL, detail TEXT NOT NULL, detail_json TEXT NOT NULL, created_at TIMESTAMP NOT NULL);
CREATE TABLE agent_log_chunks (run_id TEXT NOT NULL, sequence BIGINT NOT NULL, stream TEXT NOT NULL, text TEXT NOT NULL, created_at TIMESTAMP NOT NULL, PRIMARY KEY (run_id, sequence));
CREATE TABLE agent_questions (id TEXT PRIMARY KEY, run_id TEXT NOT NULL UNIQUE, text TEXT NOT NULL, created_at TIMESTAMP NOT NULL);
CREATE TABLE agent_answers (id TEXT PRIMARY KEY, question_id TEXT NOT NULL UNIQUE, resume_run_id TEXT NOT NULL, text TEXT NOT NULL, created_at TIMESTAMP NOT NULL);
CREATE TABLE agent_acceptances (id TEXT PRIMARY KEY, work_id TEXT NOT NULL, run_id TEXT NOT NULL, source_base TEXT NOT NULL, agent_base TEXT NOT NULL, agent_tip TEXT NOT NULL, integration_branch TEXT NOT NULL, integration_worktree TEXT NOT NULL, result_revision TEXT NOT NULL, status TEXT NOT NULL, validation_json TEXT NOT NULL, failure_reason TEXT NOT NULL, created_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT NULL);
CREATE TABLE agent_queue_state (id TEXT PRIMARY KEY CHECK (id = 'default'), status TEXT NOT NULL CHECK (status IN ('frozen', 'accepting', 'draining')), reason TEXT NOT NULL, updated_at TIMESTAMP NOT NULL);
CREATE TABLE agent_provider_state (provider TEXT PRIMARY KEY, backoff_reason TEXT NOT NULL, last_run_id TEXT NOT NULL, backoff_until TIMESTAMP NOT NULL, last_started_at TIMESTAMP NOT NULL, window_started_at TIMESTAMP NOT NULL, window_admissions INTEGER NOT NULL, updated_at TIMESTAMP NOT NULL);
CREATE UNIQUE INDEX agent_runs_work_number_attempt_idx ON agent_runs(work_id, run_number, attempt);
CREATE INDEX agent_runs_work_idx ON agent_runs(work_id, queued_at);
CREATE INDEX agent_runs_status_idx ON agent_runs(status, provider, model, queued_at);
CREATE INDEX agent_events_run_idx ON agent_events(run_id, created_at, id);
CREATE INDEX agent_acceptances_work_idx ON agent_acceptances(work_id, updated_at);
```
- [x] Write migration Good/Bad/Ugly tests for idempotence, transaction rollback, upgrade from version 1, and reopen.
- [x] Write Store contract tests for every method, transactionally checked transitions, monotonic log sequences, ordered snapshots, interrupted startup recovery, queue/provider reopen, and concurrent writer serialization. Round-trip `durable_revision` through every Run insert/load/Snapshot/Continuation path; existing rows default to empty and must never infer push acknowledgement from cached Git tracking state. Round-trip the full canonical `ChangeReview` JSON for `prepared`, `conflicted`, and `validation_failed` acceptance rows, and preserve deterministic `(updated_at, id)` ordering so the latest durable review remains authoritative even when timestamps are equal.
- [x] Assert SQL tables contain no secret values and repository directory scans contain no LEM state files.
- [x] Prove focused tests fail before implementation.
- [x] Implement `Store.Commit` as one SQL transaction for every non-nil member, with compare-and-swap status enforcement through `ExpectedStatus`. Derive concurrency from run rows instead of mutable counters.
- [x] Run tests/race, format, tidy, and commit:

```sh
cd /Users/snider/Code/core/go-inference/cli
go test ./tui -run '^(TestAgentStore|TestMigrations_Agent)' -count=1
go test -race ./tui -run '^(TestAgentStore|TestMigrations_Agent)' -count=1
go mod tidy
cd ..
gofmt -w cli/tui/agentstore.go cli/tui/agentstore_test.go cli/tui/migrations.go cli/tui/migrations_test.go cli/tui/repository.go cli/tui/repository_test.go
git add cli/go.mod cli/go.sum cli/tui/agentstore.go cli/tui/agentstore_test.go cli/tui/migrations.go cli/tui/migrations_test.go cli/tui/repository.go cli/tui/repository_test.go
git commit -m "feat(tui): persist native agent lifecycle"
```

### Task 12: Compose the Reusable Engine and TUI Adapter

**Files:**

- Create: `cli/tui/agentadapter.go`
- Create: `cli/tui/agentadapter_test.go`
- Modify: `cli/tui/agentcap.go`
- Modify: `cli/tui/agentcap_test.go`
- Modify: `cli/tui/bootstrap.go`
- Modify: `cli/tui/bootstrap_test.go`
- Modify: `cli/tui/paths.go`
- Modify: `cli/tui/paths_test.go`

**Interfaces:**

```go
type agentReviewRequest struct { Feature agentFeature; WorkID, Provider, Model, Input string }
type agentReview struct { Feature agentFeature; Title, Body, Warning string; ConfirmRequired bool; Payload any }

type agentProvider interface {
	Capabilities() []agentCapability
	Snapshot(context.Context) core.Result
	Review(context.Context, agentReviewRequest) core.Result
	Run(context.Context, agentRequest) core.Result
	Close() core.Result
}
```

- [x] Extend `appPaths` with host `SoftServe` and `Workspaces` plus medium-relative `Agents` while preserving existing path tests: `<root>/soft-serve`, `<root>/workspaces`, and `agents.yaml`.
- [x] Write adapter mapping tests for all available first-slice capabilities, unavailable future capabilities with specific reasons, Work/project translation, project review/registration including separately confirmed Git enablement, ordered events/logs/questions, dispatch review, cancel, answer, retry, resume, queue start/stop, review, accept, reject, and close.
- [x] Add private feature names `changes.review`, `accept`, and `reject`; the available first-slice set is exactly Dispatch, Cancel, Answer, Retry, Resume, Queue Start, Queue Stop, Review Changes, Accept, and Reject. Preserve specific unavailable reasons for every other catalogue entry.
- [x] Write bootstrap tests for valid policy, malformed policy frozen reason, Soft Serve owner contention degraded reason, missing Git/provider detection, `--check` composition, and cleanup after partial failures. The check path injects an unavailable read-only agent provider and never calls store recovery or constructs the native launcher.
- [x] Prove tests fail before implementation.
- [x] Map `queue.Policy.Providers` into `provider.Config`, open a go-io medium rooted at the host workspace path, then compose the DuckDB store, Soft Serve options, workspace manager, default provider registry, queue controller, native launcher, and orchestrator only in normal TUI startup. Keep Soft Serve lazy: construction and snapshots do not call Start.
- [x] Ensure workspace resource Close calls orchestrator first, then reactive/config/repository resources, preserving all errors.
- [x] Map reusable types at the adapter boundary so Bubble Tea files do not depend on Soft Serve, process, queue, or workspace implementations.
- [x] Run focused tests/race, format, and commit:

```sh
cd /Users/snider/Code/core/go-inference/cli
go test ./tui -run '^(TestAgentAdapter|TestAgentBootstrap|TestAppPaths_)' -count=1
go test -race ./tui -run '^(TestAgentAdapter|TestAgentBootstrap)' -count=1
cd ..
gofmt -w cli/tui/agentadapter.go cli/tui/agentadapter_test.go cli/tui/agentcap.go cli/tui/agentcap_test.go cli/tui/bootstrap.go cli/tui/bootstrap_test.go cli/tui/paths.go cli/tui/paths_test.go
git add cli/tui/agentadapter.go cli/tui/agentadapter_test.go cli/tui/agentcap.go cli/tui/agentcap_test.go cli/tui/bootstrap.go cli/tui/bootstrap_test.go cli/tui/paths.go cli/tui/paths_test.go
git commit -m "feat(tui): connect native agent engine"
```

### Task 13: Add Work Editing, Git Enablement, and Launch Review Overlays

**Files:**

- Create: `cli/tui/agentoverlay.go`
- Create: `cli/tui/agentoverlay_test.go`
- Modify: `cli/tui/work.go`
- Modify: `cli/tui/work_test.go`
- Modify: `cli/tui/app.go`
- Modify: `cli/tui/app_test.go`
- Modify: `cli/tui/palette.go`
- Modify: `cli/tui/palette_test.go`

**Behaviour:**

```go
type workEditor struct { title textinput.Model; task textarea.Model; repository textinput.Model; focus int; editingID, validation string }
type launchReviewOverlay struct { review agentReview; provider, model string; confirmed bool }
type agentActionMsg struct { feature agentFeature; result core.Result }
```

- [x] Write tests for creating/editing title, full task, and repository; path with spaces; empty fields; existing clean/dirty/detached Git; ad-hoc included-file review; changed included hash; provider/model selection; exact redacted argv; native-access warning; confirm/cancel; narrow/wide layout; focus/tab/escape/enter handling.
- [x] Assert Dispatch is impossible without a selected Work item and always performs project review/registration before dispatch review. Existing clean repositories register without a source mutation prompt; ad-hoc Git enablement requires its own confirmation before launch review, and no run is created until launch confirmation.
- [x] Assert Work creation/editing alone never starts Soft Serve; approving Git enablement may start it and is separately confirmed.
- [x] Prove tests fail before implementation.
- [x] Replace title-only Work creation with the editor and add `workPanel.CreateWork(title, task, repository)` / `EditWork`. Persist task/repository in the existing `lem_work_items` row.
- [x] Add overlay kinds for Work editor, Git enable review, and launch review. Route all provider calls through lifecycle-owned `tea.Cmd`; never block `Update` on Git, process, or validation.
- [x] Rebuild the palette from current capability plus selected-Work status so only contextual actions are invokable.
- [x] Run tests/race, render `--check`, format, and commit:

```sh
cd /Users/snider/Code/core/go-inference/cli
go test ./tui -run '^(TestWorkEditor|TestLaunchReview|TestApp_AgentReview|TestCommandPalette_Agent)' -count=1
go test -race ./tui -run '^(TestWorkEditor|TestLaunchReview|TestApp_AgentReview)' -count=1
go run ./cmd/lem tui --check
cd ..
gofmt -w cli/tui/agentoverlay.go cli/tui/agentoverlay_test.go cli/tui/work.go cli/tui/work_test.go cli/tui/app.go cli/tui/app_test.go cli/tui/palette.go cli/tui/palette_test.go
git add cli/tui/agentoverlay.go cli/tui/agentoverlay_test.go cli/tui/work.go cli/tui/work_test.go cli/tui/app.go cli/tui/app_test.go cli/tui/palette.go cli/tui/palette_test.go
git commit -m "feat(tui): review native agent launches"
```

### Task 14: Render Live Runs, Questions, Changes, and Acceptance

**Files:**

- Modify: `cli/tui/agentoverlay.go`
- Modify: `cli/tui/agentoverlay_test.go`
- Modify: `cli/tui/work.go`
- Modify: `cli/tui/work_test.go`
- Modify: `cli/tui/app.go`
- Modify: `cli/tui/app_test.go`
- Modify: `cli/tui/palette.go`
- Modify: `cli/tui/palette_test.go`

- [x] Write tests for queued/preparing/running/completed/waiting/cancelling/cancelled/failed/interrupted/accepted/rejected rendering; batched stdout/stderr order; refresh while another panel is active; switching Work; attention markers; Answer/Resume; Retry; Queue Start/Stop/draining; cancellation; change diff; conflict; validation pass/fail; no-validation acknowledgement; final confirmation; accept/reject.
- [x] Assert provider events are rendered from `agent_events` snapshots and are not copied into `lem_events`; local Work events remain intact.
- [x] Assert `--check` does not start Soft Serve, run providers, mutate Git, or admit queued work.
- [x] Prove focused tests fail before implementation.
- [x] Merge local Work events and live agent snapshots in memory by stable IDs. Render log chunks through the existing viewport/Markdown styles with stdout/stderr distinction and bounded history.
- [x] Add answer textarea, changes viewport, validation receipt, and final acceptance overlay states. Every long action returns through `agentActionMsg`, refreshes snapshots, and preserves actionable errors.
- [x] Add a one-second refresh command only while non-terminal agent work exists; lifecycle cancellation stops it on shutdown.
- [x] Run focused/full TUI tests and race, render checks at narrow/wide sizes, format, and commit:

```sh
cd /Users/snider/Code/core/go-inference/cli
go test ./tui -run '^(TestWorkPanel_Agent|TestAgentOverlay|TestApp_AgentAction|TestRunWithWorkspace_Check)' -count=1
go test ./tui -count=1
go test -race ./tui -count=1
go run ./cmd/lem tui --check
cd ..
gofmt -w cli/tui/agentoverlay.go cli/tui/agentoverlay_test.go cli/tui/work.go cli/tui/work_test.go cli/tui/app.go cli/tui/app_test.go cli/tui/palette.go cli/tui/palette_test.go
git add cli/tui/agentoverlay.go cli/tui/agentoverlay_test.go cli/tui/work.go cli/tui/work_test.go cli/tui/app.go cli/tui/app_test.go cli/tui/palette.go cli/tui/palette_test.go
git commit -m "feat(tui): operate and accept agent work"
```

### Task 15: Document and Verify the Complete Native Slice

**Files:**

- Modify: `cli/tui/README.md`
- Modify if implementation differs after review: `docs/superpowers/specs/2026-07-18-coreagent-native-execution-design.md`
- Modify: `docs/superpowers/plans/2026-07-18-coreagent-native-execution.md` checkbox state only

- [ ] Document Work creation, Git enablement, launch review, provider discovery, `~/.lem/agents.yaml`, queue controls, DuckDB tables, Soft Serve ownership, cancellation/shutdown, answer/resume, review, validation, acceptance/rejection, recovery, and explicit native host access.
- [ ] Run fake-provider end-to-end receipts for all three adapters: create clean repo, dispatch, stream output, commit, complete, review, validate, accept, and verify source HEAD/content.
- [ ] Run interruption receipts: long-running process with child, close orchestrator, verify both PIDs exit, run becomes interrupted, logs flush, branch pushes or worktree is retained, reopen, and Resume succeeds.
- [ ] Run conflict/validation receipts and prove source status/HEAD/content remain unchanged.
- [ ] Run all fresh gates:

```sh
cd /Users/snider/Code/core/go-inference
git diff --check
task qa
task test
task cover
task bench
cd go
go test -race ./agent/work ./agent/queue ./agent/provider ./agent/gitserver ./agent/workspace ./agent/orchestrator -count=1
GOWORK=off go test ./agent/work ./agent/queue ./agent/provider ./agent/gitserver ./agent/workspace ./agent/orchestrator -count=1
cd ../cli
go test -race ./tui -count=1
GOWORK=off go test ./tui -count=1
go run ./cmd/lem tui --check
cd ..
bash /Users/snider/Code/core/go/tests/cli/v090-upgrade/audit.sh go
```

- [ ] Confirm portable coverage remains at or above the repository's 95% project and patch target, audit verdict is COMPLIANT, the worktree is clean, no LEM control files exist in test repositories, and no process/listener survives the suite.
- [ ] Commit documentation and plan completion:

```sh
cd /Users/snider/Code/core/go-inference
git add cli/tui/README.md docs/superpowers/specs/2026-07-18-coreagent-native-execution-design.md docs/superpowers/plans/2026-07-18-coreagent-native-execution.md
git commit -m "docs(agent): document native work lifecycle"
```

## Execution Order and Checkpoints

1. Tasks 1–3 provide a portable, testable domain/queue/provider foundation without Git or process mutation.
2. Tasks 4–5 establish the private Git control plane and disposable workspaces.
3. Tasks 6–9 produce the complete reusable root orchestration package.
4. Task 10 is a hard module-release checkpoint; do not bypass it with `replace`.
5. Tasks 11–14 connect persistence and the TUI after root v0.14.0 is available.
6. Task 15 is the complete behavioural and repository handback gate.
