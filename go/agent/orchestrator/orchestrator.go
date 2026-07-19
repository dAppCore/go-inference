// SPDX-License-Identifier: EUPL-1.2

// Package orchestrator coordinates durable native agent work.
package orchestrator

import (
	"context"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/gitserver"
	"dappco.re/go/inference/agent/provider"
	"dappco.re/go/inference/agent/queue"
	"dappco.re/go/inference/agent/work"
	"dappco.re/go/inference/agent/workspace"
)

const (
	defaultLogBatchBytes    = 32 * 1024
	defaultLogBatchDelay    = 250 * time.Millisecond
	defaultAttemptTimeout   = 60 * time.Minute
	defaultCleanupTimeout   = 5 * time.Second
	hardenedRuntimeContract = "go-inference/native-runtime/v1;softserve-fail-closed;bounded-cleanup;attempt-timeout;reviewed-retry-resume"
)

// Options supplies every durable and native dependency owned by an Orchestrator.
type Options struct {
	Store          Store
	GitServer      gitserver.Service
	Workspaces     *workspace.Manager
	Providers      *provider.Registry
	Queue          *queue.Controller
	Launcher       Launcher
	Clock          Clock
	IDs            Identifier
	LogBatchBytes  int
	LogBatchDelay  time.Duration
	AttemptTimeout time.Duration
	CleanupTimeout time.Duration
}

// HardenedRuntimeContract identifies the fail-closed native runtime guarantees linked into this module.
func (Options) HardenedRuntimeContract(ctx context.Context) core.Result {
	if ctx == nil {
		return core.Fail(core.NewError("agent hardened runtime contract context is required"))
	}
	if err := ctx.Err(); err != nil {
		return core.Fail(core.E("orchestrator.Options.HardenedRuntimeContract", "contract context is done", err))
	}
	return core.Ok(hardenedRuntimeContract)
}

// ProjectReview is a mutation-free source review awaiting registration consent.
type ProjectReview struct {
	Work              work.Item
	Source            workspace.SourceReview
	RepositoryName    string
	RequiresGitEnable bool
}

type cleanupRecoveryOutcome struct {
	RecoveryEventID string
	Receipt         workspace.RecoveryReceipt
	Error           string
}

type pendingCleanupRecovery struct {
	Run            work.Run
	Receipt        workspace.RecoveryReceipt
	Kind           string
	UpdateRun      bool
	ExpectedStatus work.RunStatus
	Event          work.Event
	EventReady     bool
}

// DispatchReview is the exact source, provider, command, and queue review awaiting dispatch consent.
type DispatchReview struct {
	Request      work.DispatchRequest
	Project      work.Project
	Source       workspace.SourceReview
	Detection    provider.Detection
	Command      provider.Command
	Queue        queue.Decision
	WorktreePath string
	Warning      string

	exactCommand   bool
	reviewedBranch string
}

// ChildReview is the exact immutable continuation launch awaiting confirmation.
type ChildReview struct {
	Action       string
	Work         work.Item
	Parent       work.Run
	AnswerID     string
	RunID        string
	Provider     string
	Model        string
	Project      work.Project
	Source       workspace.SourceReview
	Detection    provider.Detection
	Command      provider.Command
	Queue        queue.Decision
	WorktreePath string
	Branch       string
	Warning      string
}

// Orchestrator owns queue admission, native processes, worktrees, and shutdown.
type Orchestrator struct {
	store      Store
	gitServer  gitserver.Service
	workspaces *workspace.Manager
	providers  *provider.Registry
	queue      *queue.Controller
	launcher   Launcher
	clock      Clock
	ids        Identifier

	logBatchBytes  int
	logBatchDelay  time.Duration
	attemptTimeout time.Duration
	cleanupTimeout time.Duration
	ctx            context.Context
	cancel         context.CancelFunc
	wake           chan struct{}
	workers        sync.WaitGroup
	closeMu        sync.Mutex
	closeComplete  bool
	closeResult    core.Result
	lifecycle      sync.RWMutex
	queueMu        sync.Mutex
	decisionMu     sync.Mutex

	mu                       sync.Mutex
	runs                     map[string]Process
	pending                  map[string]DispatchReview
	pendingCleanupRecoveries map[string]*pendingCleanupRecovery
	executions               map[string]*runExecution
	closed                   bool
}

// New recovers durable state, freezes admission, and starts an idle queue owner.
func New(options Options) core.Result {
	if options.Store == nil || options.GitServer == nil || options.Workspaces == nil || options.Providers == nil || options.Queue == nil || options.Launcher == nil || options.Clock == nil || options.IDs == nil {
		return core.Fail(core.NewError("agent orchestrator requires store, Git, workspace, provider, queue, launcher, clock, and ID dependencies"))
	}
	if options.LogBatchBytes < 0 || options.LogBatchDelay < 0 || options.AttemptTimeout < 0 || options.CleanupTimeout < 0 {
		return core.Fail(core.NewError("agent orchestrator log batch limits and attempt timeout cannot be negative"))
	}
	if options.LogBatchBytes == 0 {
		options.LogBatchBytes = defaultLogBatchBytes
	}
	if options.LogBatchDelay == 0 {
		options.LogBatchDelay = defaultLogBatchDelay
	}
	if options.AttemptTimeout == 0 {
		options.AttemptTimeout = defaultAttemptTimeout
	}
	if options.CleanupTimeout == 0 {
		options.CleanupTimeout = defaultCleanupTimeout
	}
	at := options.Clock.Now()
	if at.IsZero() {
		return core.Fail(core.NewError("agent orchestrator clock returned zero during recovery"))
	}
	if recovered := recoverStore(options.Store, at); !recovered.OK {
		return core.Fail(core.E("orchestrator.New", "failed to recover durable runs", recovered.Err()))
	}
	frozenResult := options.Queue.Stop(0, at)
	if !frozenResult.OK {
		return core.Fail(core.E("orchestrator.New", "failed to freeze recovered queue", frozenResult.Err()))
	}
	frozen, ok := frozenResult.Value.(work.QueueState)
	if !ok {
		return core.Fail(core.Errorf("agent queue returned %T instead of state", frozenResult.Value))
	}
	if committed := commitStore(options.Store, Commit{Queue: &frozen}); !committed.OK {
		return core.Fail(core.E("orchestrator.New", "failed to persist recovered queue", committed.Err()))
	}

	ctx, cancel := context.WithCancel(context.Background())
	orchestrator := &Orchestrator{
		store: options.Store, gitServer: options.GitServer, workspaces: options.Workspaces,
		providers: options.Providers, queue: options.Queue, launcher: options.Launcher,
		clock: options.Clock, ids: options.IDs, logBatchBytes: options.LogBatchBytes,
		logBatchDelay: options.LogBatchDelay, attemptTimeout: options.AttemptTimeout, cleanupTimeout: options.CleanupTimeout,
		ctx: ctx, cancel: cancel, wake: make(chan struct{}, 1),
		runs: make(map[string]Process), pending: make(map[string]DispatchReview),
		pendingCleanupRecoveries: make(map[string]*pendingCleanupRecovery),
		executions:               make(map[string]*runExecution), closeResult: core.Ok(nil),
	}
	orchestrator.workers.Add(1)
	go orchestrator.queueLoop()
	return core.Ok(orchestrator)
}

func (orchestrator *Orchestrator) cleanupContext() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), orchestrator.cleanupTimeout)
}

// Capabilities reports the reusable native-agent feature catalogue.
func (orchestrator *Orchestrator) Capabilities() []work.Capability {
	closed := orchestrator == nil || orchestrator.isClosed()
	reason := ""
	if closed {
		reason = "agent orchestrator is closed"
	}
	capabilities := []work.Capability{
		{Name: "dispatch", Available: !closed, Reason: reason},
		{Name: "cancel", Available: !closed, Reason: reason},
		{Name: "queue.start", Available: !closed, Reason: reason},
		{Name: "queue.stop", Available: !closed, Reason: reason},
		{Name: "answer", Available: !closed, Reason: reason},
		{Name: "retry", Available: !closed, Reason: reason},
		{Name: "resume", Available: !closed, Reason: reason},
		{Name: "changes.review", Available: !closed, Reason: reason},
		{Name: "accept", Available: !closed, Reason: reason},
		{Name: "reject", Available: !closed, Reason: reason},
		{Name: "recovery.abandon", Available: !closed, Reason: reason},
	}
	return capabilities
}

// Snapshot returns a detached durable view for one Work item or all Work when the ID is empty.
func (orchestrator *Orchestrator) Snapshot(ctx context.Context, workID string) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.lifecycle.RLock()
	defer orchestrator.lifecycle.RUnlock()
	if contextResult := validateContext(ctx, "snapshot"); !contextResult.OK {
		return contextResult
	}
	result := orchestrator.store.Snapshot(core.Trim(workID))
	if !result.OK {
		return result
	}
	snapshot, ok := result.Value.(work.Snapshot)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of snapshot", result.Value))
	}
	return core.Ok(cloneSnapshot(snapshot))
}

// AbandonRecovery explicitly retries verified cleanup for one exact durable recovery event.
func (orchestrator *Orchestrator) AbandonRecovery(ctx context.Context, runID, recoveryEventID string) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.lifecycle.RLock()
	defer orchestrator.lifecycle.RUnlock()
	if contextResult := validateContext(ctx, "recovery abandonment"); !contextResult.OK {
		return contextResult
	}
	if orchestrator.isClosed() {
		return core.Fail(core.NewError("agent orchestrator is closed"))
	}
	runID = core.Trim(runID)
	recoveryEventID = core.Trim(recoveryEventID)
	if runID == "" || recoveryEventID == "" {
		return core.Fail(core.NewError("agent recovery abandonment requires run and recovery event IDs"))
	}

	orchestrator.decisionMu.Lock()
	defer orchestrator.decisionMu.Unlock()
	runResult := orchestrator.store.Run(runID)
	if !runResult.OK {
		return runResult
	}
	run, ok := runResult.Value.(work.Run)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of run", runResult.Value))
	}
	if run.ID != runID {
		return core.Fail(core.NewError("agent store returned a different recovery run identity"))
	}
	switch run.Status {
	case work.RunCancelled, work.RunFailed, work.RunCompleted, work.RunInterrupted, work.RunAccepted, work.RunRejected:
	default:
		return core.Fail(core.NewError("agent cleanup recovery requires a terminal durable run before workspace mutation"))
	}
	snapshotResult := orchestrator.store.Snapshot(run.WorkID)
	if !snapshotResult.OK {
		return snapshotResult
	}
	snapshot, ok := snapshotResult.Value.(work.Snapshot)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of snapshot", snapshotResult.Value))
	}
	retainedResult := cleanupRecoveryEvent(snapshot.Events, run, recoveryEventID)
	if !retainedResult.OK {
		return retainedResult
	}
	retained := retainedResult.Value.(work.Event)
	var receipt workspace.RecoveryReceipt
	if decoded := core.JSONUnmarshalString(retained.DetailJSON, &receipt); !decoded.OK {
		return core.Fail(core.E("orchestrator.Orchestrator.AbandonRecovery", "durable cleanup recovery receipt is invalid", decoded.Err()))
	}
	if validated := validateCleanupRecoveryIdentity(snapshot, run, retained, receipt); !validated.OK {
		return orchestrator.recordCleanupRecoveryFailure(run, retained.ID, receipt, validated)
	}
	terminalResult := terminalCleanupRecovery(snapshot.Events, run, retained.ID, receipt)
	if !terminalResult.OK {
		return orchestrator.recordCleanupRecoveryFailure(run, retained.ID, receipt, terminalResult)
	}
	if terminalResult.Value != nil {
		if receipt.Kind == "run" && (run.Branch != "" || run.Worktree != "") {
			return core.Fail(core.NewError("agent cleanup recovery success did not clear the durable run workspace projection"))
		}
		return core.Ok(terminalResult.Value.(work.Event))
	}
	if validated := validatePendingCleanupRecovery(snapshot.Runs, run, receipt); !validated.OK {
		return orchestrator.recordCleanupRecoveryFailure(run, retained.ID, receipt, validated)
	}

	cleaned := orchestrator.workspaces.AbandonRecovery(ctx, receipt)
	if !cleaned.OK {
		return orchestrator.recordCleanupRecoveryFailure(run, retained.ID, receipt, cleaned)
	}
	return orchestrator.persistCleanupRecoveryOutcome(run, retained.ID, receipt, "")
}

func cleanupRecoveryEvent(events []work.Event, run work.Run, recoveryEventID string) core.Result {
	var retained *work.Event
	for index := range events {
		event := events[index]
		if event.ID != recoveryEventID {
			continue
		}
		if retained != nil {
			return core.Fail(core.NewError("agent cleanup recovery event identity is duplicated"))
		}
		retained = &event
	}
	if retained == nil {
		return core.Fail(core.NewError("agent cleanup recovery event was not found"))
	}
	if retained.RunID != run.ID || retained.WorkID != run.WorkID ||
		(retained.Kind != "workspace_cleanup_retained" && retained.Kind != "review_cleanup_retained") {
		return core.Fail(core.NewError("agent cleanup recovery event does not belong to the requested run"))
	}
	return core.Ok(*retained)
}

func validateCleanupRecoveryIdentity(snapshot work.Snapshot, run work.Run, retained work.Event, receipt workspace.RecoveryReceipt) core.Result {
	expectedKind := "run"
	if retained.Kind == "review_cleanup_retained" {
		expectedKind = "review"
	}
	if receipt.Kind != expectedKind || receipt.ProjectID != run.ProjectID || receipt.WorkID != run.WorkID ||
		receipt.RunID != run.ID || receipt.RunNumber != run.Number || retained.Detail != receipt.Worktree {
		return core.Fail(core.NewError("agent cleanup recovery receipt differs from its durable event or run"))
	}
	if receipt.Kind == "review" {
		if receipt.WorkspaceRunID != "" {
			return core.Fail(core.NewError("agent cleanup recovery review cannot name a run workspace lineage"))
		}
	} else if receipt.ReviewID != "" {
		return core.Fail(core.NewError("agent cleanup recovery run cannot name a review"))
	}
	for _, snapshotRun := range snapshot.Runs {
		if snapshotRun.ID == run.ID {
			if snapshotRun != run {
				return core.Fail(core.NewError("agent cleanup recovery run differs from its durable snapshot"))
			}
			return core.Ok(nil)
		}
	}
	return core.Fail(core.NewError("agent cleanup recovery run is missing from its durable snapshot"))
}

func validatePendingCleanupRecovery(runs []work.Run, run work.Run, receipt workspace.RecoveryReceipt) core.Result {
	if receipt.Kind == "review" {
		return core.Ok(nil)
	}
	if run.Branch != receipt.Branch || run.Worktree != receipt.Worktree {
		return core.Fail(core.NewError("agent cleanup recovery run differs from its durable workspace lineage"))
	}
	return validateCleanupRecoveryLineage(runs, run, receipt)
}

func validateCleanupRecoveryLineage(runs []work.Run, run work.Run, receipt workspace.RecoveryReceipt) core.Result {
	byID := make(map[string]work.Run, len(runs))
	for _, candidate := range runs {
		if _, exists := byID[candidate.ID]; exists {
			return core.Fail(core.NewError("agent cleanup recovery durable run lineage contains duplicate IDs"))
		}
		byID[candidate.ID] = candidate
	}
	snapshotRun, exists := byID[run.ID]
	if !exists || snapshotRun != run {
		return core.Fail(core.NewError("agent cleanup recovery run differs from its durable snapshot"))
	}
	visited := make(map[string]bool)
	current := run
	for {
		if current.ID == "" || visited[current.ID] {
			return core.Fail(core.NewError("agent cleanup recovery durable run lineage is cyclic"))
		}
		visited[current.ID] = true
		if current.ProjectID != receipt.ProjectID || current.WorkID != receipt.WorkID || current.Number != receipt.RunNumber ||
			current.Branch != receipt.Branch || current.Worktree != receipt.Worktree {
			return core.Fail(core.NewError("agent cleanup recovery durable run lineage differs from the receipt"))
		}
		if current.ParentRunID == "" {
			if current.ID != receipt.WorkspaceRunID {
				return core.Fail(core.NewError("agent cleanup recovery durable run lineage root differs from the workspace owner"))
			}
			return core.Ok(nil)
		}
		parent, exists := byID[current.ParentRunID]
		if !exists {
			return core.Fail(core.NewError("agent cleanup recovery durable run lineage is incomplete"))
		}
		current = parent
	}
}

func terminalCleanupRecovery(events []work.Event, run work.Run, recoveryEventID string, receipt workspace.RecoveryReceipt) core.Result {
	var terminal *work.Event
	for _, event := range events {
		if event.Kind != "cleanup_recovery_succeeded" && event.Kind != "cleanup_recovery_failed" {
			continue
		}
		var outcome cleanupRecoveryOutcome
		decoded := core.JSONUnmarshalString(event.DetailJSON, &outcome)
		if !decoded.OK {
			if event.RunID == run.ID && event.WorkID == run.WorkID && event.Detail == receipt.Worktree {
				return core.Fail(core.E("orchestrator.terminalCleanupRecovery", "agent cleanup recovery outcome is invalid", decoded.Err()))
			}
			continue
		}
		if outcome.RecoveryEventID != recoveryEventID {
			continue
		}
		if event.RunID != run.ID || event.WorkID != run.WorkID || event.Detail != receipt.Worktree ||
			!sameCleanupRecoveryReceipt(outcome.Receipt, receipt) {
			return core.Fail(core.NewError("agent cleanup recovery outcome differs from its durable receipt"))
		}
		if event.Kind == "cleanup_recovery_succeeded" {
			if outcome.Error != "" {
				return core.Fail(core.NewError("agent cleanup recovery success outcome contains a failure"))
			}
			if terminal == nil {
				copy := event
				terminal = &copy
			}
			continue
		}
		if core.Trim(outcome.Error) == "" {
			return core.Fail(core.NewError("agent cleanup recovery failure outcome has no error"))
		}
	}
	if terminal != nil {
		return core.Ok(*terminal)
	}
	return core.Ok(nil)
}

func sameCleanupRecoveryReceipt(left, right workspace.RecoveryReceipt) bool {
	return left.Kind == right.Kind && left.ProjectID == right.ProjectID && left.WorkID == right.WorkID &&
		left.RunID == right.RunID && left.RunNumber == right.RunNumber && left.WorkspaceRunID == right.WorkspaceRunID &&
		left.ReviewID == right.ReviewID && left.Branch == right.Branch && left.Worktree == right.Worktree
}

func (orchestrator *Orchestrator) recordCleanupRecoveryFailure(run work.Run, recoveryEventID string, receipt workspace.RecoveryReceipt, failure core.Result) core.Result {
	persisted := orchestrator.persistCleanupRecoveryOutcome(run, recoveryEventID, receipt, failure.Error())
	if persisted.OK {
		return failure
	}
	return core.Fail(core.E(
		"orchestrator.Orchestrator.AbandonRecovery",
		core.Concat(failure.Error(), "; cleanup recovery audit failed for exact receipt ", core.JSONMarshalString(receipt)),
		persisted.Err(),
	))
}

func (orchestrator *Orchestrator) persistCleanupRecoveryOutcome(run work.Run, recoveryEventID string, receipt workspace.RecoveryReceipt, failure string) core.Result {
	kind := "cleanup_recovery_succeeded"
	title := "cleanup recovery succeeded"
	if core.Trim(failure) != "" {
		kind = "cleanup_recovery_failed"
		title = "cleanup recovery failed"
	}
	receiptJSON := core.JSONMarshalString(receipt)
	eventResult := orchestrator.newEvent(run, kind, title, receipt.Worktree)
	if !eventResult.OK {
		return core.Fail(core.E("orchestrator.persistCleanupRecoveryOutcome", core.Concat("cleanup recovery audit failed for exact receipt ", receiptJSON), eventResult.Err()))
	}
	event := eventResult.Value.(work.Event)
	event.DetailJSON = core.JSONMarshalString(cleanupRecoveryOutcome{
		RecoveryEventID: recoveryEventID, Receipt: receipt, Error: core.Trim(failure),
	})
	commit := Commit{Event: &event}
	if kind == "cleanup_recovery_succeeded" && receipt.Kind == "run" {
		cleared := run
		cleared.Branch = ""
		cleared.Worktree = ""
		expected := run.Status
		commit.Run = &cleared
		commit.ExpectedStatus = &expected
	}
	if committed := commitStore(orchestrator.store, commit); !committed.OK {
		return core.Fail(core.E("orchestrator.persistCleanupRecoveryOutcome", core.Concat("cleanup recovery audit failed for exact receipt ", receiptJSON), committed.Err()))
	}
	return core.Ok(event)
}

// ReviewProject inspects a Work source without changing Git, the source, or durable state.
func (orchestrator *Orchestrator) ReviewProject(ctx context.Context, item work.Item) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.lifecycle.RLock()
	defer orchestrator.lifecycle.RUnlock()
	if contextResult := validateContext(ctx, "project review"); !contextResult.OK {
		return contextResult
	}
	item = normalizeWorkItem(item)
	if item.ID == "" || item.Repository == "" {
		return core.Fail(core.NewError("agent project review requires Work ID and repository"))
	}
	if orchestrator.isClosed() {
		return core.Fail(core.NewError("agent orchestrator is closed"))
	}
	reviewResult := orchestrator.workspaces.ReviewSource(ctx, item.Repository)
	if !reviewResult.OK {
		return reviewResult
	}
	source, ok := reviewResult.Value.(workspace.SourceReview)
	if !ok {
		return core.Fail(core.Errorf("agent workspace returned %T instead of source review", reviewResult.Value))
	}
	repositoryName := item.ID
	projectResult := orchestrator.store.ProjectBySource(source.Path)
	if !projectResult.OK {
		return projectResult
	}
	if projectResult.Value != nil {
		project, projectOK := projectResult.Value.(work.Project)
		if !projectOK {
			return core.Fail(core.Errorf("agent store returned %T instead of project", projectResult.Value))
		}
		repositoryName = project.RepositoryName
	}
	return core.Ok(ProjectReview{
		Work: item, Source: source, RepositoryName: repositoryName, RequiresGitEnable: !source.Git,
	})
}

// RegisterProject confirms one reviewed source and persists its private Git identity.
func (orchestrator *Orchestrator) RegisterProject(ctx context.Context, review ProjectReview, confirmed bool) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.lifecycle.RLock()
	defer orchestrator.lifecycle.RUnlock()
	if contextResult := validateContext(ctx, "project registration"); !contextResult.OK {
		return contextResult
	}
	if orchestrator.isClosed() {
		return core.Fail(core.NewError("agent orchestrator is closed"))
	}
	if !confirmed {
		return core.Fail(core.NewError("agent project registration requires explicit confirmation"))
	}
	review.Work = normalizeWorkItem(review.Work)
	review.RepositoryName = core.Trim(review.RepositoryName)
	if review.Work.ID == "" || review.Source.Path == "" || review.Source.IncludedHash == "" || review.RepositoryName == "" {
		return core.Fail(core.NewError("agent project registration review is incomplete"))
	}
	freshResult := orchestrator.workspaces.ReviewSource(ctx, review.Source.Path)
	if !freshResult.OK {
		return freshResult
	}
	fresh, ok := freshResult.Value.(workspace.SourceReview)
	if !ok {
		return core.Fail(core.Errorf("agent workspace returned %T instead of source review", freshResult.Value))
	}
	if !sameSourceReview(review.Source, fresh) || review.RequiresGitEnable != !fresh.Git {
		return core.Fail(core.NewError("agent project source changed after review; review the project again"))
	}
	projectID := review.Work.ID
	repositoryName := review.RepositoryName
	var existingProject *work.Project
	existingResult := orchestrator.store.ProjectBySource(fresh.Path)
	if !existingResult.OK {
		return existingResult
	}
	if existingResult.Value != nil {
		existing, existingOK := existingResult.Value.(work.Project)
		if !existingOK {
			return core.Fail(core.Errorf("agent store returned %T instead of project", existingResult.Value))
		}
		if review.RepositoryName != existing.RepositoryName {
			return core.Fail(core.NewError("agent project repository identity changed after review"))
		}
		if existing.SourcePath == fresh.Path && existing.RepositoryRoot == fresh.Root &&
			existing.SourceBranch == fresh.Branch && existing.SourceRevision == fresh.Revision {
			return core.Ok(existing)
		}
		projectID = existing.ID
		repositoryName = existing.RepositoryName
		existingProject = &existing
	}
	registered := orchestrator.workspaces.Register(ctx, workspace.RegisterRequest{
		ProjectID: projectID, SourcePath: fresh.Path, RepositoryName: repositoryName,
		EnableGit: review.RequiresGitEnable, Confirmed: true, ExpectedIncludedHash: review.Source.IncludedHash,
	})
	if !registered.OK {
		return registered
	}
	project, ok := registered.Value.(work.Project)
	if !ok {
		return core.Fail(core.Errorf("agent workspace returned %T instead of project", registered.Value))
	}
	if existingProject != nil {
		project.CreatedAt = existingProject.CreatedAt
	}
	if committed := commitStore(orchestrator.store, Commit{Project: &project}); !committed.OK {
		return committed
	}
	return core.Ok(project)
}

// Close withdraws queued work, joins native processes and workers, then closes private Git.
func (orchestrator *Orchestrator) Close() core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.closeMu.Lock()
	defer orchestrator.closeMu.Unlock()
	if orchestrator.closeComplete {
		return orchestrator.closeResult
	}
	orchestrator.lifecycle.Lock()
	orchestrator.closeResult = orchestrator.close()
	orchestrator.lifecycle.Unlock()
	orchestrator.closeComplete = orchestrator.closeResult.OK
	return orchestrator.closeResult
}

func (orchestrator *Orchestrator) close() core.Result {
	failures := make([]string, 0, 8)
	orchestrator.mu.Lock()
	orchestrator.closed = true
	processes := make([]Process, 0, len(orchestrator.runs))
	for _, process := range orchestrator.runs {
		processes = append(processes, process)
	}
	orchestrator.mu.Unlock()
	if persisted := orchestrator.flushPendingCleanupRecoveries(); !persisted.OK {
		return persisted
	}

	if withdrawn := orchestrator.withdrawQueued(); !withdrawn.OK {
		failures = append(failures, withdrawn.Error())
	}
	at := orchestrator.now()
	if at.OK {
		orchestrator.queueMu.Lock()
		queueSnapshotResult := orchestrator.durableQueueSnapshot()
		if !queueSnapshotResult.OK {
			failures = append(failures, queueSnapshotResult.Error())
		} else {
			queueSnapshot := queueSnapshotResult.Value.(work.Snapshot)
			stopped := orchestrator.queue.Stop(len(processes), at.Value.(time.Time))
			if stopped.OK {
				state, stateOK := stopped.Value.(work.QueueState)
				if !stateOK {
					failure := core.Fail(core.Errorf("agent queue returned %T instead of state", stopped.Value))
					restored := orchestrator.restoreQueue(queueSnapshot, "orchestrator.Close", failure)
					failures = append(failures, restored.Error())
				} else if committed := commitStore(orchestrator.store, Commit{Queue: &state}); !committed.OK {
					restored := orchestrator.restoreQueue(queueSnapshot, "orchestrator.Close", committed)
					failures = append(failures, restored.Error())
				}
			} else {
				failures = append(failures, stopped.Error())
			}
		}
		orchestrator.queueMu.Unlock()
	} else {
		failures = append(failures, at.Error())
	}

	orchestrator.cancel()
	for _, process := range processes {
		if shutdown := process.Shutdown(); !shutdown.OK {
			failures = append(failures, shutdown.Error())
		}
	}
	if closed := orchestrator.launcher.Close(); !closed.OK {
		failures = append(failures, closed.Error())
	}
	orchestrator.workers.Wait()
	recoveryAt := orchestrator.now()
	if !recoveryAt.OK {
		failures = append(failures, recoveryAt.Error())
	} else if recovered := recoverStore(orchestrator.store, recoveryAt.Value.(time.Time)); !recovered.OK {
		failures = append(failures, recovered.Error())
	} else {
		orchestrator.mu.Lock()
		orchestrator.runs = make(map[string]Process)
		orchestrator.executions = make(map[string]*runExecution)
		orchestrator.pending = make(map[string]DispatchReview)
		orchestrator.mu.Unlock()
		orchestrator.finishDrainingQueue()
	}
	if closed := orchestrator.gitServer.Close(); !closed.OK {
		failures = append(failures, closed.Error())
	}
	if len(failures) > 0 {
		return core.Fail(core.NewError(core.Join("; ", failures...)))
	}
	return core.Ok(nil)
}

func (orchestrator *Orchestrator) withdrawQueued() core.Result {
	snapshotResult := orchestrator.store.Snapshot("")
	if !snapshotResult.OK {
		return snapshotResult
	}
	snapshot, ok := snapshotResult.Value.(work.Snapshot)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of snapshot", snapshotResult.Value))
	}
	failures := make([]string, 0)
	for _, current := range snapshot.Runs {
		if current.Status != work.RunQueued {
			continue
		}
		at := orchestrator.now()
		if !at.OK {
			failures = append(failures, at.Error())
			continue
		}
		current.Status = work.RunCancelled
		current.FinishedAt = at.Value.(time.Time)
		current.UpdatedAt = current.FinishedAt
		expected := work.RunQueued
		eventResult := orchestrator.newEvent(current, "cancelled", "queued run withdrawn during shutdown", "")
		if !eventResult.OK {
			failures = append(failures, eventResult.Error())
			continue
		}
		event := eventResult.Value.(work.Event)
		if committed := commitStore(orchestrator.store, Commit{Run: &current, ExpectedStatus: &expected, Event: &event}); !committed.OK {
			failures = append(failures, committed.Error())
			continue
		}
		orchestrator.mu.Lock()
		delete(orchestrator.pending, current.ID)
		orchestrator.mu.Unlock()
	}
	if len(failures) > 0 {
		return core.Fail(core.NewError(core.Join("; ", failures...)))
	}
	return core.Ok(nil)
}

func (orchestrator *Orchestrator) isClosed() bool {
	if orchestrator == nil {
		return true
	}
	orchestrator.mu.Lock()
	defer orchestrator.mu.Unlock()
	return orchestrator.closed
}

func (orchestrator *Orchestrator) now() core.Result {
	at := orchestrator.clock.Now()
	if at.IsZero() {
		return core.Fail(core.NewError("agent orchestrator clock returned zero"))
	}
	return core.Ok(at)
}

func (orchestrator *Orchestrator) nextID(label string) core.Result {
	id := core.Trim(orchestrator.ids.New())
	if id == "" {
		return core.Fail(core.Errorf("agent orchestrator ID source returned empty %s ID", label))
	}
	return core.Ok(id)
}

func validateContext(ctx context.Context, operation string) core.Result {
	if ctx == nil {
		return core.Fail(core.Errorf("agent orchestrator %s context is required", operation))
	}
	if err := ctx.Err(); err != nil {
		return core.Fail(core.E(core.Concat("orchestrator.", operation), "context is done", err))
	}
	return core.Ok(nil)
}

func normalizeWorkItem(item work.Item) work.Item {
	item.ID = core.Trim(item.ID)
	item.ExternalID = core.Trim(item.ExternalID)
	item.Title = core.Trim(item.Title)
	item.Task = core.Trim(item.Task)
	item.Repository = core.Trim(item.Repository)
	return item
}

func sameSourceReview(review, fresh workspace.SourceReview) bool {
	return review.Path == fresh.Path && review.Root == fresh.Root && review.Branch == fresh.Branch &&
		review.ProposedBranch == fresh.ProposedBranch && review.Revision == fresh.Revision &&
		review.CommitIdentity == fresh.CommitIdentity && review.Git == fresh.Git && review.Clean == fresh.Clean &&
		review.Detached == fresh.Detached && review.IncludedHash == fresh.IncludedHash &&
		sameStrings(review.Included, fresh.Included)
}

func cloneSnapshot(snapshot work.Snapshot) work.Snapshot {
	snapshot.Projects = append([]work.Project(nil), snapshot.Projects...)
	snapshot.Runs = append([]work.Run(nil), snapshot.Runs...)
	snapshot.Events = append([]work.Event(nil), snapshot.Events...)
	snapshot.Logs = append([]work.LogChunk(nil), snapshot.Logs...)
	snapshot.Questions = append([]work.Question(nil), snapshot.Questions...)
	snapshot.Acceptances = append([]work.Acceptance(nil), snapshot.Acceptances...)
	snapshot.Providers = append([]work.ProviderState(nil), snapshot.Providers...)
	return snapshot
}
