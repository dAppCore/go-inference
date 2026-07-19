# CoreAgent Native Execution Port Design

**Date:** 2026-07-18
**Status:** Approved design; awaiting written-spec review
**Scope:** First CoreAgent feature port: Git-isolated native execution lifecycle

## Product intent

LEM's Work panel becomes an operational agent workspace without importing or
reproducing CoreAgent's CLI. CoreAgent remains the behavioural reference for
dispatch, queueing, status, cancellation, blocked answers, resume, rate limits,
and provider selection. Those behaviours move into focused, reusable packages
below `go/agent/*` and are consumed by the TUI through its existing provider
boundary.

This first port runs Codex, Claude, and OpenCode as non-interactive native
processes. It deliberately establishes a clean orchestration core before
container, remote fleet, forge, Brain, planning, QA, review, or pull-request
automation is added.

Git is an invariant. A private, LEM-managed Soft Serve instance is the internal
Git control plane for both established repositories and ad-hoc directories.
Agents work only in disposable internal worktrees. The user's source checkout
is unchanged until an explicit, validated acceptance operation, apart from the
separately reviewed Git initialisation and baseline commit for an ad-hoc
directory.

## Decisions

- Port behaviour into `go/agent/*`; do not import `dappco.re/go/agent`.
- Support native Codex, Claude, and OpenCode providers first.
- Dispatch only from a selected, durable Work item.
- Work creation and editing capture a title, task, and repository directory.
- Require a launch review before every native process starts.
- Run agents non-interactively and stream their output into the Work timeline.
- LEM owns child processes; closing LEM cancels them and records interruption.
- Preserve CoreAgent's provider/model concurrency, rate, backoff, and queue
  policy in `~/.lem/agents.yaml`.
- Run a private loopback Soft Serve service on demand with data below
  `~/.lem/soft-serve/`; stop it with its owning LEM process.
- Keep all orchestration state, logs, questions, answers, and acceptance
  receipts in DuckDB. Never create LEM state files in execution repositories.
- Use internal Git branches and worktrees for execution, review, and acceptance.
- Require a clean source repository before seeding, dispatch, or acceptance.

## Goals

- Make Dispatch, Cancel, Answer, Retry, Resume, Queue Start, and Queue Stop real
  TUI capabilities.
- Preserve the proven CoreAgent lifecycle while improving package isolation,
  persistence, safety, and testability.
- Keep the source repository untouched while an agent works.
- Make every process, Git, and state transition durable and inspectable.
- Recover unfinished work after restart without attempting to reattach to or
  silently restart a native process.
- Support established Git repositories and explicitly Git-enable ad-hoc
  directories.
- Provide a stable foundation for later container and remote providers.

## Non-goals

- Container, Apple Container, VZ, Docker, or Podman execution.
- Interactive PTY sessions.
- Detached or daemon-supervised agents that survive LEM shutdown.
- CoreAgent MCP tools, HTTP service, CLI commands, PHP services, or global
  filesystem layout.
- Plans, phases, sessions, handoffs, Brain, messaging, fleet, forge, scanning,
  monitoring, harvesting, QA, review, PR creation, or merge automation.
- Automatic repository mutation before explicit acceptance.
- Automatic installation or download of Codex, Claude, OpenCode, Git, or Soft
  Serve binaries. Soft Serve is linked through its Go packages.
- Storing LEM control Markdown or JSON files inside a source or execution
  worktree.

## Package architecture

### `go/agent/work`

Defines the stable domain surface without TUI, process, Git, Soft Serve, or
DuckDB dependencies:

- project and repository identity;
- Work and run identifiers;
- dispatch and resume requests;
- provider and model identity;
- run, queue, and acceptance statuses;
- ordered events and log chunks;
- blocked questions and answers;
- Git revisions and acceptance results.

Public values are immutable-by-convention snapshots. Mutable state is owned by
the orchestrator and persisted through interfaces.

### `go/agent/gitserver`

Defines a narrow internal Git control-plane interface and its pinned Soft Serve
implementation:

- start and stop the private loopback service;
- create or resolve a private repository;
- expose an authenticated internal clone URL without leaking credentials;
- query service health;
- prevent concurrent writers to the same Soft Serve data directory.

All Soft Serve types stop at this package boundary. Other agent packages see
only LEM-owned interfaces and values, limiting upstream API churn.

### `go/agent/workspace`

Owns source validation and isolated Git operations:

- recognise an existing repository or explicitly initialise an ad-hoc folder;
- verify clean source state;
- seed and synchronise the selected source revision into Soft Serve without
  changing the source's configured remotes;
- maintain one cached internal clone per project;
- create, reconstruct, and remove per-run worktrees;
- capture remaining non-ignored agent changes as labelled run commits;
- prepare disposable integration worktrees;
- validate and apply accepted commits.

The package executes Git through an injected command interface backed by
`dappco.re/go/process`. It does not import Bubble Tea or know about panels.

### `go/agent/provider`

Defines native provider adapters for Codex, Claude, and OpenCode. Each adapter:

- detects whether its executable is available;
- validates configured model and flags;
- builds an executable plus argument vector, never a shell command string;
- defines safe environment inheritance;
- selects the provider's structured output mode when available;
- parses output into text, progress, usage, rate-limit, question, and terminal
  events;
- redacts credentials from command receipts and errors.

Native defaults do not reproduce CoreAgent's blanket permission-bypass flags.
Unsafe custom flags remain possible only through explicit configuration and are
prominent in the launch review.

### `go/agent/queue`

Ports CoreAgent's dispatch policy from `~/.lem/agents.yaml`:

- default provider and model;
- global, per-provider, and per-model concurrency;
- minimum delay, sustained delay, burst delay, and reset window;
- provider backoff after parsed rate-limit events;
- frozen, accepting, and draining state;
- FIFO admission with provider/model eligibility.

The compatibility loader accepts the proven CoreAgent provider/model and rate
shape. LEM-specific optional fields are additive. Queue state is persisted in
DuckDB; the YAML file is policy, not runtime state.

### `go/agent/orchestrator`

Exposes the sole high-level service used by applications:

- capabilities;
- create or update project registration;
- snapshot;
- dispatch;
- cancel;
- answer and resume;
- retry;
- start and stop queue admission;
- review, accept, and reject;
- close.

It depends on interfaces for Git control, workspace operations, processes,
provider adapters, persistence, identifiers, and clocks. It owns all running
contexts and joins every worker during shutdown.

### `cli/tui` adapter

A small adapter translates public agent snapshots, events, requests, and
capabilities into the existing private `agentProvider` contract. Bubble Tea
commands call the adapter asynchronously; the reusable engine never imports TUI
packages.

## Runtime ownership

The orchestrator owns:

- the queue controller;
- the on-demand Soft Serve service lease;
- cached-project and run-worktree leases;
- every child process context and process-group handle;
- output readers and persistence workers;
- provider backoff state;
- shutdown cancellation and joining.

No goroutine or child process may outlive `Close`. Closing LEM freezes queue
admission, withdraws queued admissions, cancels running process groups, waits
for process groups and output readers, flushes buffered log chunks, attempts to
capture and push each run's remaining work, and records unfinished runs as
interrupted. A worktree is released only when its branch is durably recoverable
from Soft Serve. If capture or push fails, the worktree and a recovery receipt
are retained. Soft Serve stops only after those shutdown records are durable.

## Git control plane

### Private Soft Serve

Soft Serve starts on demand when a project must be registered or dispatched.
It binds only loopback listeners and uses `~/.lem/soft-serve/` for configuration,
keys, repositories, and upstream metadata. LEM generates and owns an internal
credential; it never writes credentials into Git remotes, command output,
DuckDB logs, or UI text.

An exclusive owner lock protects the Soft Serve data directory. A second LEM
process may continue using Chat, Models, and Service, but its agent capabilities
are unavailable with a visible ownership reason. Stale locks are recoverable
only after the recorded owner is no longer alive.

LEM's relational application state remains DuckDB. Soft Serve's own upstream
metadata database is opaque implementation data and is never queried,
mirrored, or treated as LEM domain state.

### Existing repositories

Assigning an existing repository to Work records:

1. canonical source path;
2. repository root;
3. current branch and commit;
4. configured origin identity when present;
5. a stable LEM project identifier.

Dispatch requires a clean source. LEM pushes the chosen commit to a private
Soft Serve repository through an explicit URL and does not add, delete, or
rewrite source remotes. The launch review shows the exact source revision.
LEM re-reads the source branch, commit, and cleanliness when the review opens
and again when dispatch is confirmed. A detached source checkout must first be
placed on an explicit target branch so later acceptance has an unambiguous
destination.

### Ad-hoc directories

An ad-hoc directory cannot dispatch until the user approves “Make Git-backed.”
That review lists the directory, files included by Git, initial branch, commit
identity, and private repository name. Approval:

1. initialises `.git` in the selected directory;
2. stages non-ignored files;
3. creates a baseline commit;
4. creates the private Soft Serve repository;
5. pushes the baseline branch.

Global Git configuration is never changed. If no commit identity exists, LEM
uses an explicitly displayed repository-local LEM identity.

### Internal clone and execution worktrees

Each project has a cached clone under `~/.lem/workspaces/<project-id>/repo.git`.
Runs use worktrees below
`~/.lem/workspaces/<project-id>/runs/<run-id>/worktree` on branches named like
`lem/work/<work-id>/run-<number>`.

The process working directory is always the execution worktree. The source
checkout is not passed to the provider. Runtime metadata and logs never enter
the worktree.

At the end of each run, the workspace layer inspects Git state. Remaining
non-ignored changes are captured in a labelled run commit and pushed to Soft
Serve. This preserves useful work even after cancellation or a failed provider
exit. Generated ignored files remain ignored.

Resume reuses the branch. If its worktree was cleaned up, the workspace layer
reconstructs it from Soft Serve before starting the next run.

## Domain and persistence model

The existing `work_items` record remains the user-owned task. Agent execution
adds relational records to `~/.lem/lem.duckdb`:

### `agent_projects`

- project ID;
- canonical source path;
- source repository identity;
- Soft Serve repository name;
- internal cached-clone path;
- source branch and last seeded commit;
- created and updated timestamps.

### `agent_runs`

- run ID and Work ID;
- parent run ID for retry or resume;
- provider and model;
- source, execution, and accepted revisions;
- internal branch and worktree path;
- redacted command receipt;
- process identifier while live;
- status, exit code, and failure reason;
- queued, started, finished, and updated timestamps.

### `agent_events`

- ordered event ID and run ID;
- kind and title;
- structured detail JSON;
- event timestamp.

Kinds include queued, admitted, preparing, started, progress, rate-limited,
waiting, answered, resumed, cancelling, cancelled, failed, completed,
interrupted, review-started, validation, conflict, accepted, and rejected.

### `agent_log_chunks`

- run ID;
- monotonically increasing sequence;
- stdout or stderr stream;
- text payload;
- timestamp.

Chunks are batched by bounded time and size to avoid one DuckDB transaction per
terminal fragment. Ordering is deterministic across persistence and rendering.

### `agent_questions` and `agent_answers`

Questions store run, text, and creation time. Answers store question, text,
answer time, and the resume run that consumed them. The original task, prior
output, question, and answer are used to build the continuation prompt.

### `agent_acceptances`

- Work ID and source/agent revision range;
- integration branch and worktree;
- validation command receipts and results;
- conflict or failure detail;
- accepted source commits;
- reviewer decision and timestamps.

### `agent_queue_state`

- singleton queue identifier;
- frozen, accepting, or draining status;
- status reason and updated timestamp.

### `agent_provider_state`

- provider identifier;
- persisted backoff deadline and reason;
- last admitted run and start timestamp;
- current quota-window start and admitted count;
- updated timestamp.

Concurrency is derived from non-terminal `agent_runs`; it is never maintained
as a second counter that could drift. Per-model and per-provider rate-window
queries use `agent_runs` together with `agent_provider_state`.

## Lifecycle and state transitions

Runs use these states:

```text
queued -> preparing -> running -> completed
   |          |           |------> waiting -> queued (resume)
   |          |           |------> cancelling -> cancelled
   |          |           |------> failed -> queued (retry)
   |          |           |------> interrupted -> queued (resume/retry)
   |          |           `------> completed -> accepted | rejected
   |          `------------------> failed
   `-----------------------------> cancelled
```

Persistence of a run and queued event happens before admission. A process starts
only after its running transition can be committed. If durable state cannot be
recorded, the process is cancelled and the run becomes failed; LEM never leaves
an untracked native agent running.

On startup, queued, preparing, running, or cancelling runs become interrupted.
The queue starts frozen and never restarts work automatically. The user may
review and explicitly resume or retry an interrupted run.

Each native process attempt has an immutable run record. Retry and Resume
create a new queued run linked through `parent_run_id`; they do not rewrite the
terminal attempt. Because providers are non-interactive, `waiting` means the
provider process has exited after emitting a valid question. Answer stores the
response in DuckDB, and Resume starts a new process on the same internal branch
with the earlier task, output, question, and answer as continuation context.

## Provider process contract

The launch request includes:

- Work title and task;
- repository and branch context;
- provider and model;
- previous-run context for resume;
- structured final-status instructions;
- a statement that the process works in an isolated Git worktree;
- a request to commit coherent changes when appropriate.

Providers use structured CLI output when supported. The parser also recognises
a final LEM status envelope from the assistant response:

- completed with an optional summary;
- waiting with a required question;
- failed with a reason.

Malformed or missing status envelopes never invent a question. A zero exit with
no valid envelope is completed with an “unclassified provider finish” event; a
non-zero exit is failed. The full redacted output remains available for review.

Environment inheritance is an allowlist containing normal process essentials
and configured provider credential variable names. Values never appear in the
launch review, receipts, or logs. Native execution is explicitly labelled as
host-accessing; repository isolation is not misrepresented as an OS sandbox.

## Queue policy

`~/.lem/agents.yaml` is loaded with `dappco.re/go/config` semantics and validated
before the queue can start. Invalid policy keeps dispatch frozen and preserves
the file for correction.

Admission checks, in order:

1. queue is accepting;
2. provider exists and is available;
3. global concurrency permits admission;
4. provider concurrency permits admission;
5. model concurrency permits admission;
6. provider is outside its rate-limit backoff;
7. minimum and burst delay requirements are satisfied;
8. Work and project state remain dispatchable.

Queue Stop disables new admissions immediately and preserves queued records. If
work is still running, the queue reports `draining`; it becomes `frozen` when
the final active run ends. It does not cancel running work. Queue Start returns
the queue to `accepting` and admits eligible work. Cancel removes a queued run
or cancels a running process group. A running cancellation uses a grace period
before force termination.

## TUI experience

### Work creation and editing

Work creation and editing use a focused overlay with:

- title;
- full task;
- repository directory;
- detected Git/project state.

An ad-hoc directory offers the explicit Git-enabling review. A dirty existing
repository shows the blocking reason and cannot dispatch.

### Launch review

Dispatch is available only on the selected Work item. The review shows:

- provider and model selectors;
- source directory, branch, and commit;
- private internal repository and execution branch;
- projected worktree location;
- current queue limits and expected position;
- redacted executable and arguments;
- native host-access warning;
- dry-run preparation result.

Confirm queues the run. Cancel closes the overlay without creating a run.

### Work timeline and actions

The Work detail timeline renders durable events and batched stdout/stderr.
Switching panels or Work items does not affect execution.

Contextual actions are:

- Dispatch for ready Work;
- Cancel for queued or running Work;
- Answer for waiting Work;
- Retry for failed or cancelled Work;
- Resume for waiting or interrupted Work;
- Queue Start and Queue Stop;
- Review Changes for a completed run;
- Accept and Reject after review.

The command palette mirrors only currently available actions. The remaining
CoreAgent capability catalogue stays visible but unavailable with specific
reasons.

## Review and acceptance

Review Changes reads commits and diffs from the internal branch and shows
source divergence. It does not mutate the source.

Accept:

1. verifies the source remains clean;
2. fetches the agent revision into a disposable integration worktree;
3. fast-forwards when the source base is unchanged;
4. otherwise replays the agent commit range onto the current source revision;
5. records conflicts without touching the source checkout;
6. runs configured validation commands in the integration worktree;
7. presents commits, validation, and resulting source revision for final
   confirmation;
8. re-verifies that the source is clean, on the reviewed target branch, and at
   the reviewed commit;
9. advances the source branch and working tree to the already validated result
   with a fast-forward-only operation;
10. records the accepted commits and result.

If the source branch or commit changes between review and final confirmation,
acceptance aborts and must be prepared again. LEM never applies an unvalidated
rebase, merge, or conflict resolution directly in the source checkout.

If no validation command is configured, acceptance requires an explicit “no
validation configured” acknowledgement. Failed validation or conflicts retain
the internal branch and integration receipt for later retry.

Reject records the decision and retains the internal branch until normal
retention cleanup. It never rewrites or deletes execution history.

## Failure handling

| Failure | Behaviour |
| --- | --- |
| Soft Serve cannot start | Agent capabilities are unavailable; the rest of LEM remains usable. |
| Soft Serve owner lock is held | Agent capabilities show the owning-process reason; no second writer opens the data directory. |
| Git is missing | Project registration and dispatch are unavailable. |
| Source is dirty | Dispatch or acceptance is blocked before internal work begins. |
| Source is detached | Dispatch is blocked until an explicit acceptance branch is selected. |
| Provider executable is missing | That provider is unavailable with its detection reason. |
| Invalid `agents.yaml` | Queue stays frozen; no defaults silently replace the invalid policy. |
| DuckDB write fails before launch | No process starts. |
| DuckDB write fails during execution | Process group is cancelled and the run is failed. |
| Output reader fails | Remaining process is cancelled; captured output and failure are retained. |
| Process exits non-zero | Run is failed with exit and redacted stderr detail. |
| Structured output is malformed | Raw output is retained; no false blocked/completed metadata is invented beyond exit classification. |
| Git capture or push fails | Run fails, or remains interrupted during shutdown, and its worktree is retained for recovery. |
| Integration conflicts | Source remains untouched; conflict receipt and integration worktree are retained. |
| Validation fails | Source remains untouched; results are durable and reviewable. |
| LEM exits | Queue freezes; children are cancelled and joined; unfinished runs become interrupted. |

## Security and authority

- Soft Serve binds only loopback addresses.
- Internal Git credentials are generated and stored below `~/.lem/soft-serve/`
  with restrictive permissions.
- Credentials are never embedded in persisted URLs or command receipts.
- User-controlled paths are canonicalised and checked before file or Git
  operations.
- Git commands use explicit argument vectors.
- Provider commands use explicit argument vectors and a fixed worktree.
- No blanket native permission-bypass flags are enabled by default.
- Every external mutation is tied to a selected Work item and an explicit
  review or confirmation.
- Acceptance is the only operation authorised to mutate an established source
  checkout after registration.

## Module and release boundary

The reusable packages land and pass the root Go module gates first. The nested
CLI must not gain a permanent local `replace` directive. Before the TUI imports
the new agent surface, the root module receives an available version and the
CLI dependency is updated normally. Workspace builds exercise the integrated
source throughout development; standalone CLI verification is restored at the
version checkpoint rather than bypassed.

## Testing strategy

### Unit and contract tests

- AX-7 Good, Bad, and Ugly tests for new public symbols in sibling test files.
- Runnable examples for every public surface.
- Command construction and credential redaction for all three providers.
- Queue global/provider/model limits, rate windows, backoff, freeze, cancel,
  retry, and resume.
- State-machine transition and invalid-transition tests.
- Log chunk ordering and batching.
- Shutdown joining and persistence-failure cancellation.

### Git integration tests

Temporary real Git repositories cover:

- clean existing repository registration;
- ad-hoc initialisation and baseline commit;
- source remote preservation;
- dirty-source rejection;
- internal seed and synchronisation;
- branch and worktree creation;
- worktree reconstruction from Soft Serve;
- dirty run capture commits;
- source divergence;
- conflict isolation;
- validation failure;
- successful acceptance;
- rejection and retention.

A loopback Soft Serve fixture uses a temporary data directory and no external
network access.

### Persistence and recovery tests

- DuckDB close/reopen for projects, runs, events, logs, questions, answers, and
  acceptance receipts.
- DuckDB close/reopen for queue admission, provider backoff, last-start, and
  quota-window state.
- Startup interruption of non-terminal runs.
- Queue policy preserved while runtime state remains in DuckDB.
- Repository scans proving no LEM state or status files enter execution or
  source worktrees.

### Process and TUI tests

- Fake Codex, Claude, and OpenCode executables stream structured and malformed
  output, block, fail, rate-limit, ignore graceful cancellation, and exit.
- Race tests cover concurrent output, cancellation, queue admission, snapshot,
  and shutdown.
- TUI tests cover Work editing, Git-enable review, launch review, live timeline,
  answer/resume, review, acceptance, failure states, and narrow layouts.
- `lem tui --check` remains non-destructive and does not start Soft Serve or an
  agent.
- Live provider receipts are opt-in and never run in the portable default
  suite.

## Acceptance criteria

1. A selected Work item can register a clean existing Git repository without
   changing its remotes.
2. An ad-hoc directory can become Git-backed only after explicit review.
3. Soft Serve is private, loopback-only, managed by LEM, and stored below
   `~/.lem/soft-serve/`.
4. Codex, Claude, and OpenCode availability is detected with precise reasons.
5. Launch review shows source, internal branch, provider/model, queue policy,
   redacted command, and native-access warning.
6. Confirmed work queues under CoreAgent-compatible concurrency and rate
   policy.
7. Output and lifecycle state survive DuckDB close/reopen in order.
8. Cancel terminates queued or running work without leaving a child process.
9. Answer/Resume uses DuckDB context and the existing internal branch without
   repository state files.
10. LEM shutdown cancels and joins every process and marks work interrupted.
11. Shutdown retains any worktree whose final changes are not durably pushed.
12. Agent work never changes the source checkout before acceptance.
13. Conflicts, source movement, or failed validation never change the source
    checkout.
14. Successful acceptance records and applies only the validated commits.
15. Source and execution repositories contain no LEM control Markdown or JSON
    state files.
16. The remaining CoreAgent feature catalogue stays honestly unavailable.
17. Root, race, TUI, launch-check, audit-baseline, and standalone module gates
    pass at their defined release checkpoint.

## Follow-on slices

After this foundation:

1. container providers using `dappco.re/go/container`;
2. plans, phases, sessions, artifacts, and handoffs;
3. Brain recall, remember, and messaging;
4. monitoring, harvesting, QA, and review;
5. fleet, forge, remote dispatch, PR creation, and merge.

Each follow-on slice plugs into the same work, queue, event, Git, and TUI
boundaries rather than adding a second orchestration system.

## References

- CoreAgent behavioural source: `/Users/snider/Code/core/agent/go/pkg/agentic`
- CoreAgent queue source: `/Users/snider/Code/core/agent/go/pkg/runner`
- Existing TUI provider boundary: `cli/tui/agentcap.go`
- Existing TUI Work projection: `cli/tui/work.go`
- Soft Serve repository: <https://github.com/charmbracelet/soft-serve>
- Soft Serve configuration: <https://github.com/charmbracelet/soft-serve/blob/main/pkg/config/config.go>
- Soft Serve Git daemon: <https://github.com/charmbracelet/soft-serve/blob/main/pkg/daemon/daemon.go>
