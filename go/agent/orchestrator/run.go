// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	"context"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/provider"
	"dappco.re/go/inference/agent/queue"
	"dappco.re/go/inference/agent/work"
	"dappco.re/go/inference/agent/workspace"
)

const (
	nativeHostWarning     = "Native provider runs on the host with access limited by its own CLI policy; Git worktree isolation is not an OS sandbox."
	queueRetryDelay       = time.Second
	terminalRetryDelay    = 100 * time.Millisecond
	retainedEventAttempts = 2
)

type rawLine struct {
	stream string
	text   string
}

type queuedPreparation struct {
	run    work.Run
	review DispatchReview
}

type queuedAdmission struct {
	decision    queue.Decision
	preparation queuedPreparation
}

type runExecution struct {
	orchestrator *Orchestrator
	run          work.Run
	workspace    workspace.RunWorkspace
	adapter      provider.Adapter
	process      Process

	incomingMu sync.Mutex
	incoming   []rawLine
	signal     chan struct{}
	logs       []work.LogChunk
	logBytes   int
	sequence   int64
	response   []string
	failure    string
	timedOut   bool
}

// ReviewDispatch revalidates source and provider state without creating a run or worktree.
func (orchestrator *Orchestrator) ReviewDispatch(ctx context.Context, request work.DispatchRequest) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.lifecycle.RLock()
	defer orchestrator.lifecycle.RUnlock()
	return orchestrator.reviewDispatch(ctx, request)
}

func (orchestrator *Orchestrator) reviewDispatch(ctx context.Context, request work.DispatchRequest) core.Result {
	if contextResult := validateContext(ctx, "dispatch review"); !contextResult.OK {
		return contextResult
	}
	if orchestrator.isClosed() {
		return core.Fail(core.NewError("agent orchestrator is closed"))
	}
	validated := work.ValidateDispatch(request)
	if !validated.OK {
		return validated
	}
	request = validated.Value.(work.DispatchRequest)

	sourceResult := orchestrator.workspaces.ReviewSource(ctx, request.Work.Repository)
	if !sourceResult.OK {
		return sourceResult
	}
	source, ok := sourceResult.Value.(workspace.SourceReview)
	if !ok {
		return core.Fail(core.Errorf("agent workspace returned %T instead of source review", sourceResult.Value))
	}
	if !source.Git {
		return core.Fail(core.NewError("agent dispatch requires a registered Git source"))
	}
	if !source.Clean {
		return core.Fail(core.NewError("agent dispatch source must be clean"))
	}
	if source.Detached || source.Branch == "" {
		return core.Fail(core.NewError("agent dispatch source must be on an attached branch"))
	}
	if source.Revision != request.ConfirmedSourceRevision {
		return core.Fail(core.NewError("agent dispatch source revision differs from the confirmed review"))
	}

	projectResult := orchestrator.store.ProjectBySource(source.Path)
	if !projectResult.OK {
		return projectResult
	}
	if projectResult.Value == nil {
		return core.Fail(core.NewError("agent dispatch source is not registered"))
	}
	project, ok := projectResult.Value.(work.Project)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of project", projectResult.Value))
	}
	if project.SourceBranch != source.Branch || project.SourceRevision != source.Revision {
		return core.Fail(core.NewError("agent dispatch source changed after project registration; register the project again"))
	}

	adapterResult := orchestrator.providers.Adapter(request.Provider)
	if !adapterResult.OK {
		return adapterResult
	}
	adapter, ok := adapterResult.Value.(provider.Adapter)
	if !ok {
		return core.Fail(core.Errorf("agent provider registry returned %T instead of adapter", adapterResult.Value))
	}
	detectionResult := adapter.Detect(ctx)
	if !detectionResult.OK {
		return detectionResult
	}
	detection, ok := detectionResult.Value.(provider.Detection)
	if !ok {
		return core.Fail(core.Errorf("agent provider returned %T instead of detection", detectionResult.Value))
	}
	if !detection.Available {
		reason := core.Trim(detection.Reason)
		if reason == "" {
			reason = core.Concat(request.Provider, " is unavailable")
		}
		return core.Fail(core.NewError(reason))
	}

	numberResult := orchestrator.store.NextRunNumber(request.Work.ID)
	if !numberResult.OK {
		return numberResult
	}
	number, ok := numberResult.Value.(int)
	if !ok || number <= 0 {
		return core.Fail(core.Errorf("agent store returned invalid next run number %v", numberResult.Value))
	}
	worktreePath := core.PathJoin(core.PathDir(project.ClonePath), "runs", "pending-run", "worktree")
	branchResult := reviewRunBranch(request.Work.ID, number)
	if !branchResult.OK {
		return branchResult
	}
	commandResult := adapter.Build(provider.Launch{
		WorkID: request.Work.ID, RunID: "pending-run", Title: request.Work.Title, Task: request.Work.Task,
		Worktree: worktreePath, Branch: branchResult.Value.(string), Model: request.Model,
		UnsafeFlags: request.UnsafeFlags,
	})
	if !commandResult.OK {
		return commandResult
	}
	command, ok := commandResult.Value.(provider.Command)
	if !ok {
		return core.Fail(core.Errorf("agent provider returned %T instead of command", commandResult.Value))
	}

	snapshotResult := orchestrator.durableQueueSnapshot()
	if !snapshotResult.OK {
		return snapshotResult
	}
	snapshot := snapshotResult.Value.(work.Snapshot)
	at := orchestrator.now()
	if !at.OK {
		return at
	}
	decisionResult := orchestrator.queue.Decide(queue.Candidate{
		RunID: core.Concat("review-", request.Work.ID), Provider: request.Provider,
		Model: request.Model, QueuedAt: at.Value.(time.Time),
	}, queue.Runtime{Queued: snapshot.Runs, Running: snapshot.Runs, Now: at.Value.(time.Time)})
	if !decisionResult.OK {
		return decisionResult
	}
	decision, ok := decisionResult.Value.(queue.Decision)
	if !ok {
		return core.Fail(core.Errorf("agent queue returned %T instead of decision", decisionResult.Value))
	}
	return core.Ok(DispatchReview{
		Request: request, Project: project, Source: source, Detection: detection,
		Command: command, Queue: decision, WorktreePath: worktreePath, Warning: nativeHostWarning,
	})
}

// Dispatch revalidates a launch review and durably queues one immutable run.
func (orchestrator *Orchestrator) Dispatch(ctx context.Context, review DispatchReview) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.lifecycle.RLock()
	defer orchestrator.lifecycle.RUnlock()
	if contextResult := validateContext(ctx, "dispatch"); !contextResult.OK {
		return contextResult
	}
	if orchestrator.isClosed() {
		return core.Fail(core.NewError("agent orchestrator is closed"))
	}
	freshResult := orchestrator.reviewDispatch(ctx, review.Request)
	if !freshResult.OK {
		return freshResult
	}
	fresh := freshResult.Value.(DispatchReview)
	if !sameDispatchReview(review, fresh) {
		return core.Fail(core.NewError("agent dispatch review is stale; review the launch again"))
	}

	project := fresh.Project

	numberResult := orchestrator.store.NextRunNumber(fresh.Request.Work.ID)
	if !numberResult.OK {
		return numberResult
	}
	number, ok := numberResult.Value.(int)
	if !ok || number <= 0 {
		return core.Fail(core.Errorf("agent store returned invalid next run number %v", numberResult.Value))
	}
	runIDResult := orchestrator.nextID("run")
	if !runIDResult.OK {
		return runIDResult
	}
	atResult := orchestrator.now()
	if !atResult.OK {
		return atResult
	}
	at := atResult.Value.(time.Time)
	run := work.Run{
		ID: runIDResult.Value.(string), WorkID: fresh.Request.Work.ID, ProjectID: project.ID,
		Provider: fresh.Request.Provider, Model: fresh.Request.Model, SourceRevision: fresh.Source.Revision,
		CommandReceipt: fresh.Command.Receipt, Status: work.RunQueued, Number: number, Attempt: 1,
		QueuedAt: at, UpdatedAt: at,
	}
	eventResult := orchestrator.newEvent(run, "queued", "run queued for native admission", fresh.Request.Work.Task)
	if !eventResult.OK {
		return eventResult
	}
	event := eventResult.Value.(work.Event)
	if committed := commitStore(orchestrator.store, Commit{Project: &project, Run: &run, CreateRun: true, Event: &event}); !committed.OK {
		return committed
	}
	fresh.Project = project
	orchestrator.mu.Lock()
	orchestrator.pending[run.ID] = fresh
	orchestrator.mu.Unlock()
	orchestrator.wakeQueue()
	return core.Ok(run)
}

// Answer durably records one response and reserves the exact later resume run identity.
func (orchestrator *Orchestrator) Answer(ctx context.Context, runID, text string) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.lifecycle.RLock()
	defer orchestrator.lifecycle.RUnlock()
	if contextResult := validateContext(ctx, "answer"); !contextResult.OK {
		return contextResult
	}
	if orchestrator.isClosed() {
		return core.Fail(core.NewError("agent orchestrator is closed"))
	}
	runID = core.Trim(runID)
	text = core.Trim(text)
	if runID == "" || text == "" {
		return core.Fail(core.NewError("agent answer requires a run ID and non-empty text"))
	}

	orchestrator.queueMu.Lock()
	defer orchestrator.queueMu.Unlock()
	continuationResult := orchestrator.continuation(runID)
	if !continuationResult.OK {
		return continuationResult
	}
	continuation := continuationResult.Value.(work.Continuation)
	if continuation.Run.Status != work.RunWaiting {
		return core.Fail(core.Errorf("agent run %s in %s cannot be answered", continuation.Run.ID, continuation.Run.Status))
	}
	if continuation.Question.ID == "" || continuation.Question.RunID != continuation.Run.ID {
		return core.Fail(core.NewError("agent waiting run has no durable question"))
	}
	if continuation.Answer.ID != "" {
		return core.Fail(core.NewError("agent waiting question is already answered"))
	}
	answerIDResult := orchestrator.nextID("answer")
	if !answerIDResult.OK {
		return answerIDResult
	}
	resumeRunIDResult := orchestrator.nextID("run")
	if !resumeRunIDResult.OK {
		return resumeRunIDResult
	}
	atResult := orchestrator.now()
	if !atResult.OK {
		return atResult
	}
	answer := work.Answer{
		ID: answerIDResult.Value.(string), QuestionID: continuation.Question.ID,
		ResumeRunID: resumeRunIDResult.Value.(string), Text: text, CreatedAt: atResult.Value.(time.Time),
	}
	if committed := commitStore(orchestrator.store, Commit{Answer: &answer}); !committed.OK {
		return committed
	}
	return core.Ok(answer)
}

// Resume refuses the legacy unreviewed child-launch path.
func (orchestrator *Orchestrator) Resume(ctx context.Context, request work.ResumeRequest) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	if contextResult := validateContext(ctx, "resume"); !contextResult.OK {
		return contextResult
	}
	return core.Fail(core.NewError("agent resume requires an explicitly confirmed child launch review"))
}

// Retry refuses the legacy unreviewed child-launch path.
func (orchestrator *Orchestrator) Retry(ctx context.Context, item work.Item, parentRunID string) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	if contextResult := validateContext(ctx, "retry"); !contextResult.OK {
		return contextResult
	}
	return core.Fail(core.NewError("agent retry requires an explicitly confirmed child launch review"))
}

// ReviewRetry revalidates and renders one retry launch without creating a child run.
func (orchestrator *Orchestrator) ReviewRetry(ctx context.Context, item work.Item, parentRunID string) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.lifecycle.RLock()
	defer orchestrator.lifecycle.RUnlock()
	if contextResult := validateContext(ctx, "retry review"); !contextResult.OK {
		return contextResult
	}
	if orchestrator.isClosed() {
		return core.Fail(core.NewError("agent orchestrator is closed"))
	}
	runIDResult := orchestrator.nextID("run")
	if !runIDResult.OK {
		return runIDResult
	}
	orchestrator.queueMu.Lock()
	defer orchestrator.queueMu.Unlock()
	return orchestrator.reviewChild(ctx, "retry", item, parentRunID, "", "", "", runIDResult.Value.(string))
}

// ConfirmRetry queues only the exact retry launch that was explicitly reviewed.
func (orchestrator *Orchestrator) ConfirmRetry(ctx context.Context, supplied any) core.Result {
	review, ok := supplied.(ChildReview)
	if !ok {
		return core.Fail(core.Errorf("agent retry confirmation received %T instead of child review", supplied))
	}
	return orchestrator.confirmChild(ctx, "retry", review)
}

// ReviewResume revalidates and renders one answered continuation without creating a child run.
func (orchestrator *Orchestrator) ReviewResume(ctx context.Context, request work.ResumeRequest) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.lifecycle.RLock()
	defer orchestrator.lifecycle.RUnlock()
	if contextResult := validateContext(ctx, "resume review"); !contextResult.OK {
		return contextResult
	}
	if orchestrator.isClosed() {
		return core.Fail(core.NewError("agent orchestrator is closed"))
	}
	orchestrator.queueMu.Lock()
	defer orchestrator.queueMu.Unlock()
	return orchestrator.reviewChild(ctx, "resume", request.Work, request.ParentRunID, request.AnswerID, request.Provider, request.Model, "")
}

// ConfirmResume queues only the exact answered continuation that was explicitly reviewed.
func (orchestrator *Orchestrator) ConfirmResume(ctx context.Context, supplied any) core.Result {
	review, ok := supplied.(ChildReview)
	if !ok {
		return core.Fail(core.Errorf("agent resume confirmation received %T instead of child review", supplied))
	}
	return orchestrator.confirmChild(ctx, "resume", review)
}

func (orchestrator *Orchestrator) confirmChild(ctx context.Context, action string, review ChildReview) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.lifecycle.RLock()
	defer orchestrator.lifecycle.RUnlock()
	if contextResult := validateContext(ctx, core.Concat(action, " confirmation")); !contextResult.OK {
		return contextResult
	}
	if orchestrator.isClosed() {
		return core.Fail(core.NewError("agent orchestrator is closed"))
	}
	if review.Action != action || core.Trim(review.RunID) == "" {
		return core.Fail(core.NewError("agent child launch review has the wrong action or run identity"))
	}
	orchestrator.queueMu.Lock()
	defer orchestrator.queueMu.Unlock()
	freshResult := orchestrator.reviewChild(ctx, action, review.Work, review.Parent.ID, review.AnswerID,
		review.Provider, review.Model, review.RunID)
	if !freshResult.OK {
		return freshResult
	}
	fresh := freshResult.Value.(ChildReview)
	if !sameChildReview(review, fresh) {
		return core.Fail(core.NewError("agent child launch review is stale; review the continuation again"))
	}
	return orchestrator.queueReviewedChild(fresh)
}

func (orchestrator *Orchestrator) continuation(runID string) core.Result {
	result := orchestrator.store.Continuation(runID)
	if !result.OK {
		return result
	}
	continuation, ok := result.Value.(work.Continuation)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of continuation", result.Value))
	}
	if continuation.Run.ID != runID {
		return core.Fail(core.NewError("agent store continuation does not match the requested run"))
	}
	return core.Ok(continuation)
}

func validateChildRequest(item work.Item, parentRunID string) core.Result {
	if item.ID == "" || item.Title == "" || item.Task == "" || item.Repository == "" || parentRunID == "" {
		return core.Fail(core.NewError("agent child attempt requires Work identity, title, task, repository, and parent run ID"))
	}
	return core.Ok(item)
}

func (orchestrator *Orchestrator) reviewChild(ctx context.Context, action string, item work.Item, parentRunID, answerID, providerName, model, runID string) core.Result {
	item = normalizeWorkItem(item)
	parentRunID = core.Trim(parentRunID)
	answerID = core.Trim(answerID)
	providerName = core.Lower(core.Trim(providerName))
	model = core.Trim(model)
	runID = core.Trim(runID)
	if invalid := validateChildRequest(item, parentRunID); !invalid.OK {
		return invalid
	}
	continuationResult := orchestrator.continuation(parentRunID)
	if !continuationResult.OK {
		return continuationResult
	}
	continuation := continuationResult.Value.(work.Continuation)
	parent := continuation.Run
	if parent.WorkID != item.ID {
		return core.Fail(core.NewError("agent child review Work does not match the parent run"))
	}
	if durableTask := core.Trim(continuation.Task); durableTask == "" || item.Task != durableTask {
		return core.Fail(core.NewError("agent child review Work task does not match the durable parent task"))
	}
	switch action {
	case "retry":
		switch parent.Status {
		case work.RunFailed, work.RunCancelled, work.RunInterrupted:
		default:
			return core.Fail(core.Errorf("agent run %s in %s cannot be retried", parent.ID, parent.Status))
		}
		providerName, model = parent.Provider, parent.Model
		if runID == "" {
			return core.Fail(core.NewError("agent retry review requires a reserved run ID"))
		}
	case "resume":
		if parent.Status != work.RunWaiting {
			return core.Fail(core.Errorf("agent run %s in %s cannot be resumed", parent.ID, parent.Status))
		}
		answer := continuation.Answer
		if answer.ID == "" {
			return core.Fail(core.NewError("agent resume review requires a stored answer"))
		}
		if answer.ID != answerID || answer.QuestionID != continuation.Question.ID || core.Trim(answer.ResumeRunID) == "" {
			return core.Fail(core.NewError("agent resume review answer does not match the durable question linkage"))
		}
		if providerName == "" {
			return core.Fail(core.NewError("agent resume review requires a provider ID"))
		}
		if runID == "" {
			runID = core.Trim(answer.ResumeRunID)
		}
		if runID != core.Trim(answer.ResumeRunID) {
			return core.Fail(core.NewError("agent resume review run ID differs from the durable answer reservation"))
		}
	default:
		return core.Fail(core.NewError("agent child review requires retry or resume action"))
	}
	if parent.Number <= 0 || parent.Attempt <= 0 || core.Trim(parent.Branch) == "" || core.Trim(parent.Worktree) == "" || core.Trim(parent.DurableRevision) == "" {
		return core.Fail(core.NewError("agent parent run has no durable workspace identity"))
	}
	projectResult := orchestrator.store.Project(parent.ProjectID)
	if !projectResult.OK {
		return projectResult
	}
	project, ok := projectResult.Value.(work.Project)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of project", projectResult.Value))
	}
	sourceResult := orchestrator.workspaces.ReviewSource(ctx, item.Repository)
	if !sourceResult.OK {
		return sourceResult
	}
	source, ok := sourceResult.Value.(workspace.SourceReview)
	if !ok {
		return core.Fail(core.Errorf("agent workspace returned %T instead of source review", sourceResult.Value))
	}
	if !source.Git || !source.Clean || source.Detached || source.Branch == "" {
		return core.Fail(core.NewError("agent child review source must be a clean attached Git repository"))
	}
	if source.Path != project.SourcePath || source.Root != project.RepositoryRoot || source.Branch != project.SourceBranch ||
		source.Revision != project.SourceRevision || parent.SourceRevision != project.SourceRevision {
		return core.Fail(core.NewError("agent child review source or project registration changed after the parent launch"))
	}
	adapterResult := orchestrator.providers.Adapter(providerName)
	if !adapterResult.OK {
		return adapterResult
	}
	adapter, ok := adapterResult.Value.(provider.Adapter)
	if !ok {
		return core.Fail(core.Errorf("agent provider registry returned %T instead of adapter", adapterResult.Value))
	}
	detectionResult := adapter.Detect(ctx)
	if !detectionResult.OK {
		return detectionResult
	}
	detection, ok := detectionResult.Value.(provider.Detection)
	if !ok {
		return core.Fail(core.Errorf("agent provider returned %T instead of detection", detectionResult.Value))
	}
	if !detection.Available {
		reason := core.Trim(detection.Reason)
		if reason == "" {
			reason = core.Concat(providerName, " is unavailable")
		}
		return core.Fail(core.NewError(reason))
	}
	commandResult := adapter.Build(provider.Launch{
		WorkID: item.ID, RunID: runID, Title: item.Title, Task: item.Task,
		Worktree: parent.Worktree, Branch: parent.Branch, Model: model,
		Continuation: renderContinuation(continuation),
	})
	if !commandResult.OK {
		return commandResult
	}
	command, ok := commandResult.Value.(provider.Command)
	if !ok {
		return core.Fail(core.Errorf("agent provider returned %T instead of command", commandResult.Value))
	}
	snapshotResult := orchestrator.durableQueueSnapshot()
	if !snapshotResult.OK {
		return snapshotResult
	}
	snapshot := snapshotResult.Value.(work.Snapshot)
	for _, existing := range snapshot.Runs {
		if existing.ParentRunID == parent.ID || existing.ID == runID {
			return core.Fail(core.Errorf("agent run %s already has child attempt %s", parent.ID, existing.ID))
		}
	}
	atResult := orchestrator.now()
	if !atResult.OK {
		return atResult
	}
	decisionResult := orchestrator.queue.Decide(queue.Candidate{
		RunID: runID, Provider: providerName, Model: model, QueuedAt: atResult.Value.(time.Time),
	}, queue.Runtime{Queued: snapshot.Runs, Running: snapshot.Runs, Now: atResult.Value.(time.Time)})
	if !decisionResult.OK {
		return decisionResult
	}
	decision, ok := decisionResult.Value.(queue.Decision)
	if !ok {
		return core.Fail(core.Errorf("agent queue returned %T instead of decision", decisionResult.Value))
	}
	return core.Ok(ChildReview{
		Action: action, Work: item, Parent: parent, AnswerID: answerID, RunID: runID,
		Provider: providerName, Model: model, Project: project, Source: source, Detection: detection,
		Command: command, Queue: decision, WorktreePath: parent.Worktree, Branch: parent.Branch,
		Warning: nativeHostWarning,
	})
}

func (orchestrator *Orchestrator) queueReviewedChild(review ChildReview) core.Result {
	item, parent := review.Work, review.Parent
	if parent.Number <= 0 || parent.Attempt <= 0 || core.Trim(parent.Branch) == "" || core.Trim(parent.Worktree) == "" {
		return core.Fail(core.NewError("agent parent run has no durable workspace identity"))
	}
	snapshotResult := orchestrator.store.Snapshot(parent.WorkID)
	if !snapshotResult.OK {
		return snapshotResult
	}
	snapshot, ok := snapshotResult.Value.(work.Snapshot)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of snapshot", snapshotResult.Value))
	}
	for _, existing := range snapshot.Runs {
		if existing.ParentRunID == parent.ID {
			return core.Fail(core.Errorf("agent run %s already has child attempt %s", parent.ID, existing.ID))
		}
	}
	atResult := orchestrator.now()
	if !atResult.OK {
		return atResult
	}
	at := atResult.Value.(time.Time)
	child := work.Run{
		ID: review.RunID, WorkID: parent.WorkID, ProjectID: parent.ProjectID, ParentRunID: parent.ID,
		Provider: review.Provider, Model: review.Model, SourceRevision: parent.SourceRevision,
		DurableRevision: parent.DurableRevision,
		Branch:          parent.Branch, Worktree: parent.Worktree, Status: work.RunQueued,
		Number: parent.Number, Attempt: parent.Attempt + 1, QueuedAt: at, UpdatedAt: at,
		CommandReceipt: review.Command.Receipt,
	}
	if child.ID == "" || child.Provider == "" {
		return core.Fail(core.NewError("agent child attempt requires run and provider IDs"))
	}
	eventResult := orchestrator.newEvent(child, "queued", "continuation attempt queued for native admission", item.Task)
	if !eventResult.OK {
		return eventResult
	}
	event := eventResult.Value.(work.Event)
	if committed := commitStore(orchestrator.store, Commit{Run: &child, CreateRun: true, Event: &event}); !committed.OK {
		return committed
	}
	pendingReview := DispatchReview{
		Request: work.DispatchRequest{Work: item, Provider: child.Provider, Model: child.Model, ConfirmedSourceRevision: review.Source.Revision},
		Project: review.Project, Source: review.Source, Detection: review.Detection, Command: review.Command,
		Queue: review.Queue, WorktreePath: child.Worktree, Warning: review.Warning,
		exactCommand: true, reviewedBranch: review.Branch,
	}
	orchestrator.mu.Lock()
	orchestrator.pending[child.ID] = pendingReview
	orchestrator.mu.Unlock()
	orchestrator.wakeQueue()
	return core.Ok(child)
}

// Cancel withdraws a queued run or gracefully terminates one live native process group.
func (orchestrator *Orchestrator) Cancel(ctx context.Context, runID string) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.lifecycle.RLock()
	defer orchestrator.lifecycle.RUnlock()
	if contextResult := validateContext(ctx, "cancel"); !contextResult.OK {
		return contextResult
	}
	if orchestrator.isClosed() {
		return core.Fail(core.NewError("agent orchestrator is closed"))
	}
	runID = core.Trim(runID)
	if runID == "" {
		return core.Fail(core.NewError("agent cancellation requires a run ID"))
	}
	runResult := orchestrator.store.Run(runID)
	if !runResult.OK {
		return runResult
	}
	run, ok := runResult.Value.(work.Run)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of run", runResult.Value))
	}
	if run.Status == work.RunCancelling {
		return core.Ok(run)
	}
	atResult := orchestrator.now()
	if !atResult.OK {
		return atResult
	}
	at := atResult.Value.(time.Time)
	expected := run.Status
	switch run.Status {
	case work.RunQueued:
		run.Status = work.RunCancelled
		run.FinishedAt = at
	case work.RunRunning:
		run.Status = work.RunCancelling
	default:
		return core.Fail(core.Errorf("agent run %s in %s cannot be cancelled", run.ID, run.Status))
	}
	run.UpdatedAt = at
	eventResult := orchestrator.newEvent(run, string(run.Status), "cancellation requested", "")
	if !eventResult.OK {
		return eventResult
	}
	event := eventResult.Value.(work.Event)
	if committed := commitStore(orchestrator.store, Commit{Run: &run, ExpectedStatus: &expected, Event: &event}); !committed.OK {
		return committed
	}
	orchestrator.mu.Lock()
	process := orchestrator.runs[run.ID]
	if run.Status == work.RunCancelled {
		delete(orchestrator.pending, run.ID)
	}
	orchestrator.mu.Unlock()
	if process != nil {
		if shutdown := process.Shutdown(); !shutdown.OK {
			return shutdown
		}
	}
	orchestrator.wakeQueue()
	return core.Ok(run)
}

// StartQueue explicitly enables admission for durable queued runs.
func (orchestrator *Orchestrator) StartQueue(ctx context.Context) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.lifecycle.RLock()
	defer orchestrator.lifecycle.RUnlock()
	if contextResult := validateContext(ctx, "queue start"); !contextResult.OK {
		return contextResult
	}
	if orchestrator.isClosed() {
		return core.Fail(core.NewError("agent orchestrator is closed"))
	}
	at := orchestrator.now()
	if !at.OK {
		return at
	}
	orchestrator.queueMu.Lock()
	defer orchestrator.queueMu.Unlock()
	snapshotResult := orchestrator.store.Snapshot("")
	if !snapshotResult.OK {
		return snapshotResult
	}
	snapshot, ok := snapshotResult.Value.(work.Snapshot)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of snapshot", snapshotResult.Value))
	}
	started := orchestrator.queue.Start(at.Value.(time.Time))
	if !started.OK {
		return started
	}
	state, ok := started.Value.(work.QueueState)
	if !ok {
		failure := core.Fail(core.Errorf("agent queue returned %T instead of state", started.Value))
		return orchestrator.restoreQueue(snapshot, "orchestrator.StartQueue", failure)
	}
	if committed := commitStore(orchestrator.store, Commit{Queue: &state}); !committed.OK {
		return orchestrator.restoreQueue(snapshot, "orchestrator.StartQueue", committed)
	}
	orchestrator.wakeQueue()
	return core.Ok(state)
}

// StopQueue freezes new admissions and lets active runs drain.
func (orchestrator *Orchestrator) StopQueue(ctx context.Context) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.lifecycle.RLock()
	defer orchestrator.lifecycle.RUnlock()
	if contextResult := validateContext(ctx, "queue stop"); !contextResult.OK {
		return contextResult
	}
	if orchestrator.isClosed() {
		return core.Fail(core.NewError("agent orchestrator is closed"))
	}
	orchestrator.queueMu.Lock()
	defer orchestrator.queueMu.Unlock()
	snapshotResult := orchestrator.durableQueueSnapshot()
	if !snapshotResult.OK {
		return snapshotResult
	}
	snapshot := snapshotResult.Value.(work.Snapshot)
	at := orchestrator.now()
	if !at.OK {
		return at
	}
	stopped := orchestrator.queue.Stop(activeRuns(snapshot.Runs), at.Value.(time.Time))
	if !stopped.OK {
		return stopped
	}
	state, ok := stopped.Value.(work.QueueState)
	if !ok {
		failure := core.Fail(core.Errorf("agent queue returned %T instead of state", stopped.Value))
		return orchestrator.restoreQueue(snapshot, "orchestrator.StopQueue", failure)
	}
	if committed := commitStore(orchestrator.store, Commit{Queue: &state}); !committed.OK {
		return orchestrator.restoreQueue(snapshot, "orchestrator.StopQueue", committed)
	}
	return core.Ok(state)
}

func (orchestrator *Orchestrator) queueLoop() {
	defer orchestrator.workers.Done()
	var timer *time.Timer
	var timerChannel <-chan time.Time
	for {
		select {
		case <-orchestrator.ctx.Done():
			if timer != nil {
				timer.Stop()
			}
			return
		case <-orchestrator.wake:
		case <-timerChannel:
		}
		next := orchestrator.drainQueue()
		if timer != nil {
			timer.Stop()
			timer = nil
			timerChannel = nil
		}
		if next.IsZero() || orchestrator.isClosed() {
			continue
		}
		nowResult := orchestrator.now()
		if !nowResult.OK {
			continue
		}
		delay := next.Sub(nowResult.Value.(time.Time))
		if delay <= 0 {
			delay = time.Millisecond
		}
		timer = time.NewTimer(delay)
		timerChannel = timer.C
	}
}

func (orchestrator *Orchestrator) drainQueue() time.Time {
	if orchestrator.isClosed() {
		return time.Time{}
	}
	snapshotResult := orchestrator.store.Snapshot("")
	if !snapshotResult.OK {
		return time.Time{}
	}
	snapshot, ok := snapshotResult.Value.(work.Snapshot)
	if !ok {
		return time.Time{}
	}
	queued := make([]work.Run, 0, len(snapshot.Runs))
	for _, run := range snapshot.Runs {
		if run.Status == work.RunQueued {
			queued = append(queued, run)
		}
	}
	core.SliceSortFunc(queued, func(left, right work.Run) bool {
		if left.QueuedAt.Equal(right.QueuedAt) {
			return left.ID < right.ID
		}
		return left.QueuedAt.Before(right.QueuedAt)
	})
	var next time.Time
	for _, run := range queued {
		orchestrator.mu.Lock()
		_, pending := orchestrator.pending[run.ID]
		orchestrator.mu.Unlock()
		if !pending {
			continue
		}
		at := orchestrator.now()
		if !at.OK {
			return next
		}
		admissionResult := orchestrator.admitQueuedRun(run, queued, snapshot.Runs, at.Value.(time.Time))
		if !admissionResult.OK {
			retryAt := at.Value.(time.Time).Add(queueRetryDelay)
			if next.IsZero() || retryAt.Before(next) {
				next = retryAt
			}
			break
		}
		admission, ok := admissionResult.Value.(queuedAdmission)
		if !ok {
			continue
		}
		decision := admission.decision
		if !decision.Allowed {
			if !decision.NotBefore.IsZero() && (next.IsZero() || decision.NotBefore.Before(next)) {
				next = decision.NotBefore
			}
			continue
		}
		started := orchestrator.startPreparingRun(admission.preparation)
		if !started.OK {
			retryAt := at.Value.(time.Time).Add(queueRetryDelay)
			if next.IsZero() || retryAt.Before(next) {
				next = retryAt
			}
			break
		}
		leftQueue, leftQueueOK := started.Value.(bool)
		if !leftQueueOK || !leftQueue {
			break
		}
		for index := range queued {
			if queued[index].ID == run.ID {
				queued[index].Status = work.RunPreparing
				break
			}
		}
		refreshed := orchestrator.store.Snapshot("")
		if refreshed.OK {
			if current, currentOK := refreshed.Value.(work.Snapshot); currentOK {
				snapshot = current
			}
		}
	}
	return next
}

func (orchestrator *Orchestrator) admitQueuedRun(run work.Run, queued, running []work.Run, at time.Time) core.Result {
	orchestrator.queueMu.Lock()
	defer orchestrator.queueMu.Unlock()
	decisionResult := orchestrator.queue.Decide(queue.Candidate{
		RunID: run.ID, Provider: run.Provider, Model: run.Model, QueuedAt: run.QueuedAt,
	}, queue.Runtime{Queued: queued, Running: running, Now: at})
	if !decisionResult.OK {
		return core.Ok(queuedAdmission{})
	}
	decision, ok := decisionResult.Value.(queue.Decision)
	if !ok {
		return core.Ok(queuedAdmission{})
	}
	admission := queuedAdmission{decision: decision}
	if !decision.Allowed {
		return core.Ok(admission)
	}
	preparationResult := orchestrator.beginQueuedRun(run)
	if !preparationResult.OK {
		return preparationResult
	}
	preparation, ok := preparationResult.Value.(queuedPreparation)
	if !ok {
		return core.Fail(core.Errorf("agent queue preparation returned %T instead of queued preparation", preparationResult.Value))
	}
	admission.preparation = preparation
	return core.Ok(admission)
}

func (orchestrator *Orchestrator) startQueuedRun(run work.Run) core.Result {
	orchestrator.queueMu.Lock()
	preparationResult := orchestrator.beginQueuedRun(run)
	orchestrator.queueMu.Unlock()
	if !preparationResult.OK {
		return preparationResult
	}
	preparation, ok := preparationResult.Value.(queuedPreparation)
	if !ok {
		return core.Fail(core.Errorf("agent queue preparation returned %T instead of queued preparation", preparationResult.Value))
	}
	return orchestrator.startPreparingRun(preparation)
}

func (orchestrator *Orchestrator) beginQueuedRun(run work.Run) core.Result {
	if orchestrator.isClosed() {
		return core.Fail(core.NewError("agent orchestrator is closed"))
	}
	orchestrator.mu.Lock()
	review, exists := orchestrator.pending[run.ID]
	orchestrator.mu.Unlock()
	if !exists {
		return core.Fail(core.Errorf("agent run %s has no pending launch review", run.ID))
	}
	atResult := orchestrator.now()
	if !atResult.OK {
		return atResult
	}
	run.Status = work.RunPreparing
	run.UpdatedAt = atResult.Value.(time.Time)
	expectedQueued := work.RunQueued
	preparingTitle := "preparing isolated Git worktree"
	if run.ParentRunID != "" {
		preparingTitle = "recovering durable parent Git worktree"
	}
	eventResult := orchestrator.newEvent(run, "preparing", preparingTitle, "")
	if !eventResult.OK {
		return eventResult
	}
	event := eventResult.Value.(work.Event)
	if committed := commitStore(orchestrator.store, Commit{Run: &run, ExpectedStatus: &expectedQueued, Event: &event}); !committed.OK {
		return committed
	}
	return core.Ok(queuedPreparation{run: run, review: review})
}

func (orchestrator *Orchestrator) startPreparingRun(preparation queuedPreparation) core.Result {
	run := preparation.run
	review := preparation.review
	adapterResult := orchestrator.providers.Adapter(run.Provider)
	if !adapterResult.OK {
		orchestrator.finishWithoutProcess(run, work.RunPreparing, workspace.RunWorkspace{}, adapterResult.Error())
		return core.Ok(true)
	}
	adapter := adapterResult.Value.(provider.Adapter)
	detection := adapter.Detect(orchestrator.ctx)
	if !detection.OK {
		orchestrator.finishWithoutProcess(run, work.RunPreparing, workspace.RunWorkspace{}, detection.Error())
		return core.Ok(true)
	}
	detected, ok := detection.Value.(provider.Detection)
	if !ok || !detected.Available {
		reason := "provider became unavailable after launch review"
		if ok && core.Trim(detected.Reason) != "" {
			reason = detected.Reason
		}
		orchestrator.finishWithoutProcess(run, work.RunPreparing, workspace.RunWorkspace{}, reason)
		return core.Ok(true)
	}

	var preparedResult core.Result
	if run.ParentRunID == "" {
		preparedResult = orchestrator.workspaces.PrepareRun(orchestrator.ctx, review.Project, run)
	} else {
		preparedResult = orchestrator.workspaces.ReconstructRun(orchestrator.ctx, review.Project, run)
	}
	if !preparedResult.OK {
		reason := preparedResult.Error()
		recoveryResult := orchestrator.workspaceRecovery(run.ID, "run")
		if recoveryResult.OK && recoveryResult.Value != nil {
			receipt := recoveryResult.Value.(workspace.RecoveryReceipt)
			run.Branch = receipt.Branch
			run.Worktree = receipt.Worktree
			receiptJSON := core.JSONMarshalString(receipt)
			reason = core.Join("; ", reason, core.Concat("workspace cleanup recovery: ", receiptJSON))
			if persisted := orchestrator.persistCleanupRecovery(run, receipt, "workspace_cleanup_retained", true); !persisted.OK {
				reason = core.Join("; ", reason, persisted.Error())
			}
		} else if !recoveryResult.OK {
			reason = core.Join("; ", reason, recoveryResult.Error())
		}
		orchestrator.finishWithoutProcess(run, work.RunPreparing, workspace.RunWorkspace{}, reason)
		return core.Ok(true)
	}
	prepared, ok := preparedResult.Value.(workspace.RunWorkspace)
	if !ok {
		orchestrator.finishWithoutProcess(run, work.RunPreparing, workspace.RunWorkspace{}, core.Sprintf("workspace returned %T instead of run worktree", preparedResult.Value))
		return core.Ok(true)
	}
	run.Branch = prepared.Branch
	run.Worktree = prepared.Path
	run.ExecutionRevision = prepared.BaseRevision
	command := review.Command
	if review.exactCommand {
		if prepared.Path != review.WorktreePath || prepared.Path != command.Dir || prepared.Branch != review.reviewedBranch {
			orchestrator.finishWithoutProcess(run, work.RunPreparing, prepared, "reconstructed child workspace differs from the confirmed launch review")
			return core.Ok(true)
		}
	} else {
		commandResult := adapter.Build(provider.Launch{
			WorkID: run.WorkID, RunID: run.ID, Title: review.Request.Work.Title, Task: review.Request.Work.Task,
			Worktree: prepared.Path, Branch: prepared.Branch, Model: run.Model,
			UnsafeFlags: review.Request.UnsafeFlags,
		})
		if !commandResult.OK {
			orchestrator.finishWithoutProcess(run, work.RunPreparing, prepared, commandResult.Error())
			return core.Ok(true)
		}
		var commandOK bool
		command, commandOK = commandResult.Value.(provider.Command)
		if !commandOK {
			orchestrator.finishWithoutProcess(run, work.RunPreparing, prepared, core.Sprintf("provider returned %T instead of command", commandResult.Value))
			return core.Ok(true)
		}
	}
	atResult := orchestrator.now()
	if !atResult.OK {
		orchestrator.finishWithoutProcess(run, work.RunPreparing, prepared, atResult.Error())
		return core.Ok(true)
	}
	at := atResult.Value.(time.Time)
	run.Status = work.RunRunning
	run.StartedAt = at
	run.UpdatedAt = at
	run.CommandReceipt = command.Receipt
	expectedPreparing := work.RunPreparing
	eventResult := orchestrator.newEvent(run, "running", "native provider admitted", command.Receipt)
	if !eventResult.OK {
		orchestrator.finishWithoutProcess(run, work.RunPreparing, prepared, eventResult.Error())
		return core.Ok(true)
	}
	event := eventResult.Value.(work.Event)
	orchestrator.queueMu.Lock()
	queueSnapshotResult := orchestrator.durableQueueSnapshot()
	if !queueSnapshotResult.OK {
		orchestrator.queueMu.Unlock()
		orchestrator.finishWithoutProcess(run, work.RunPreparing, prepared, queueSnapshotResult.Error())
		return core.Ok(true)
	}
	queueSnapshot := queueSnapshotResult.Value.(work.Snapshot)
	providerStateResult := orchestrator.queue.RecordStart(run.Provider, run.ID, at)
	if !providerStateResult.OK {
		orchestrator.queueMu.Unlock()
		orchestrator.finishWithoutProcess(run, work.RunPreparing, prepared, providerStateResult.Error())
		return core.Ok(true)
	}
	providerState, ok := providerStateResult.Value.(work.ProviderState)
	if !ok {
		failure := core.Fail(core.Errorf("queue returned %T instead of provider state", providerStateResult.Value))
		restored := orchestrator.restoreQueue(queueSnapshot, "orchestrator.startQueuedRun", failure)
		orchestrator.queueMu.Unlock()
		orchestrator.finishWithoutProcess(run, work.RunPreparing, prepared, restored.Error())
		return core.Ok(true)
	}
	if committed := commitStore(orchestrator.store, Commit{Run: &run, ExpectedStatus: &expectedPreparing, Event: &event, Provider: &providerState}); !committed.OK {
		restored := orchestrator.restoreQueue(queueSnapshot, "orchestrator.startQueuedRun", committed)
		orchestrator.queueMu.Unlock()
		orchestrator.finishWithoutProcess(run, work.RunPreparing, prepared, restored.Error())
		return core.Ok(true)
	}
	orchestrator.queueMu.Unlock()

	execution := &runExecution{
		orchestrator: orchestrator, run: run, workspace: prepared, adapter: adapter,
		signal: make(chan struct{}, 1), incoming: make([]rawLine, 0, 32),
		logs: make([]work.LogChunk, 0, 32), response: make([]string, 0, 64),
	}
	started := orchestrator.launcher.Start(orchestrator.ctx, command, execution.enqueue)
	if !started.OK {
		orchestrator.finishWithoutProcess(run, work.RunRunning, prepared, started.Error())
		return core.Ok(true)
	}
	process, ok := started.Value.(Process)
	if !ok {
		orchestrator.finishWithoutProcess(run, work.RunRunning, prepared, core.Sprintf("launcher returned %T instead of process", started.Value))
		return core.Ok(true)
	}
	execution.process = process
	run.ProcessID = process.PID()
	atResult = orchestrator.now()
	if !atResult.OK {
		orchestrator.abortUntrackedProcess(execution, work.RunRunning, atResult.Error())
		return core.Ok(true)
	}
	run.UpdatedAt = atResult.Value.(time.Time)
	expectedProcessStatus := work.RunRunning
	currentResult := orchestrator.store.Run(run.ID)
	if currentResult.OK {
		if current, currentOK := currentResult.Value.(work.Run); currentOK && current.Status == work.RunCancelling {
			run.Status = work.RunCancelling
			expectedProcessStatus = work.RunCancelling
		}
	}
	processEventResult := orchestrator.newEvent(run, "process_started", "native process group started", core.Sprintf("pid=%d", run.ProcessID))
	if !processEventResult.OK {
		orchestrator.abortUntrackedProcess(execution, expectedProcessStatus, processEventResult.Error())
		return core.Ok(true)
	}
	processEvent := processEventResult.Value.(work.Event)
	if committed := commitStore(orchestrator.store, Commit{Run: &run, ExpectedStatus: &expectedProcessStatus, Event: &processEvent}); !committed.OK {
		orchestrator.abortUntrackedProcess(execution, expectedProcessStatus, committed.Error())
		return core.Ok(true)
	}
	execution.run = run
	if run.Status == work.RunCancelling {
		if shutdown := process.Shutdown(); !shutdown.OK {
			execution.failure = shutdown.Error()
		}
	}
	orchestrator.mu.Lock()
	orchestrator.runs[run.ID] = process
	orchestrator.executions[run.ID] = execution
	orchestrator.mu.Unlock()
	orchestrator.workers.Add(1)
	go orchestrator.monitorRun(execution)
	return core.Ok(true)
}

func (orchestrator *Orchestrator) abortUntrackedProcess(execution *runExecution, expected work.RunStatus, reason string) {
	shutdown := execution.process.Shutdown()
	waited := execution.process.Wait()
	if !shutdown.OK {
		reason = core.Concat(reason, "; ", shutdown.Error())
	}
	if !waited.OK {
		reason = core.Concat(reason, "; ", waited.Error())
	}
	orchestrator.finishWithoutProcess(execution.run, expected, execution.workspace, reason)
}

func (orchestrator *Orchestrator) finishWithoutProcess(run work.Run, expected work.RunStatus, prepared workspace.RunWorkspace, reason string) {
	cleanupContext, cancelCleanup := orchestrator.cleanupContext()
	defer cancelCleanup()
	atResult := orchestrator.now()
	if !atResult.OK {
		return
	}
	status := work.RunFailed
	if orchestrator.isClosed() {
		status = work.RunInterrupted
	}
	run.Status = status
	run.FailureReason = core.Trim(reason)
	run.FinishedAt = atResult.Value.(time.Time)
	run.UpdatedAt = run.FinishedAt
	var capture workspace.Capture
	if prepared.Path != "" {
		captured := orchestrator.workspaces.CaptureRun(cleanupContext, prepared)
		if captured.OK {
			capturedValue, captureOK := captured.Value.(workspace.Capture)
			if !captureOK {
				run.FailureReason = core.Join("; ", run.FailureReason, core.Sprintf("workspace returned %T instead of capture", captured.Value))
			} else {
				capture = capturedValue
				run.DurableRevision = capture.DurableRevision
				if capture.Revision != "" {
					run.ExecutionRevision = capture.Revision
				}
				if !capture.Pushed {
					run.FailureReason = core.Join("; ", run.FailureReason, capture.Summary)
				}
			}
		} else {
			run.FailureReason = core.Join("; ", run.FailureReason, captured.Error())
		}
	}
	eventResult := orchestrator.newEvent(run, string(status), run.FailureReason, "")
	if !eventResult.OK {
		return
	}
	event := eventResult.Value.(work.Event)
	if committed := orchestrator.commitTerminal(Commit{Run: &run, ExpectedStatus: &expected, Event: &event}); !committed.OK {
		return
	}
	if prepared.Path != "" && capture.Pushed {
		released := orchestrator.workspaces.ReleaseRun(cleanupContext, prepared)
		if !released.OK {
			orchestrator.persistWorkspaceRetained(run, released.Error())
		}
	}
	orchestrator.finishRunOwnership(run.ID)
}

func (execution *runExecution) enqueue(stream, line string) {
	execution.incomingMu.Lock()
	execution.incoming = append(execution.incoming, rawLine{stream: core.Lower(core.Trim(stream)), text: line})
	execution.incomingMu.Unlock()
	select {
	case execution.signal <- struct{}{}:
	default:
	}
}

func (execution *runExecution) takeLines() []rawLine {
	execution.incomingMu.Lock()
	lines := execution.incoming
	execution.incoming = nil
	execution.incomingMu.Unlock()
	return lines
}

func (orchestrator *Orchestrator) monitorRun(execution *runExecution) {
	defer orchestrator.workers.Done()
	waited := make(chan core.Result, 1)
	go func() { waited <- execution.process.Wait() }()
	ticker := time.NewTicker(orchestrator.logBatchDelay)
	defer ticker.Stop()
	timeout := time.NewTimer(orchestrator.attemptTimeout)
	defer timeout.Stop()
	for {
		select {
		case <-execution.signal:
			orchestrator.consumeLines(execution)
		case <-ticker.C:
			orchestrator.consumeLines(execution)
			orchestrator.flushLogs(execution)
		case <-timeout.C:
			execution.timedOut = true
			execution.failure = core.Concat("native provider attempt timeout after ", orchestrator.attemptTimeout.String())
			if shutdown := execution.process.Shutdown(); !shutdown.OK {
				execution.failure = core.Join("; ", execution.failure, shutdown.Error())
			}
		case result := <-waited:
			orchestrator.consumeLines(execution)
			orchestrator.flushLogs(execution)
			orchestrator.finishExecution(execution, result)
			return
		}
	}
}

func (orchestrator *Orchestrator) consumeLines(execution *runExecution) {
	for _, line := range execution.takeLines() {
		orchestrator.consumeLine(execution, line)
	}
}

func (orchestrator *Orchestrator) consumeLine(execution *runExecution, line rawLine) {
	if line.text != "" {
		atResult := orchestrator.now()
		if !atResult.OK {
			orchestrator.failExecution(execution, atResult.Error())
			return
		}
		execution.sequence++
		execution.logs = append(execution.logs, work.LogChunk{
			RunID: execution.run.ID, Sequence: execution.sequence, Stream: line.stream,
			Text: line.text, CreatedAt: atResult.Value.(time.Time),
		})
		execution.logBytes += len(line.stream) + len(line.text)
	}
	for _, output := range execution.adapter.ParseLine(line.stream, line.text) {
		if core.Trim(output.Text) != "" {
			execution.response = append(execution.response, output.Text)
		}
		switch output.Kind {
		case "", "text", "stderr", "raw":
			continue
		default:
			orchestrator.persistProviderOutput(execution, output)
		}
	}
	if execution.logBytes >= orchestrator.logBatchBytes {
		orchestrator.flushLogs(execution)
	}
}

func (orchestrator *Orchestrator) persistProviderOutput(execution *runExecution, output provider.Output) {
	eventResult := orchestrator.newEvent(execution.run, output.Kind, output.Text, "")
	if !eventResult.OK {
		orchestrator.failExecution(execution, eventResult.Error())
		return
	}
	event := eventResult.Value.(work.Event)
	event.DetailJSON = core.Trim(output.DetailJSON)
	commit := Commit{Event: &event}
	var queueSnapshot work.Snapshot
	queueMutation := false
	if output.Kind == "rate_limit" && core.Trim(output.RetryAfter) != "" {
		parsed := core.ParseDuration(output.RetryAfter)
		if parsed.OK {
			if duration, ok := parsed.Value.(time.Duration); ok && duration > 0 {
				atResult := orchestrator.now()
				if !atResult.OK {
					orchestrator.failExecution(execution, atResult.Error())
					return
				}
				at := atResult.Value.(time.Time)
				orchestrator.queueMu.Lock()
				snapshotResult := orchestrator.durableQueueSnapshot()
				if !snapshotResult.OK {
					orchestrator.queueMu.Unlock()
					orchestrator.failExecution(execution, snapshotResult.Error())
					return
				}
				queueSnapshot = snapshotResult.Value.(work.Snapshot)
				stateResult := orchestrator.queue.RecordBackoff(execution.run.Provider, output.Text, at.Add(duration), at)
				if !stateResult.OK {
					orchestrator.queueMu.Unlock()
					orchestrator.failExecution(execution, stateResult.Error())
					return
				}
				state, stateOK := stateResult.Value.(work.ProviderState)
				if !stateOK {
					failure := core.Fail(core.Errorf("agent queue returned %T instead of provider state", stateResult.Value))
					restored := orchestrator.restoreQueue(queueSnapshot, "orchestrator.persistProviderOutput", failure)
					orchestrator.queueMu.Unlock()
					orchestrator.failExecution(execution, restored.Error())
					return
				}
				commit.Provider = &state
				queueMutation = true
			}
		}
	}
	if committed := commitStore(orchestrator.store, commit); !committed.OK {
		if queueMutation {
			committed = orchestrator.restoreQueue(queueSnapshot, "orchestrator.persistProviderOutput", committed)
			orchestrator.queueMu.Unlock()
		}
		orchestrator.failExecution(execution, committed.Error())
		return
	}
	if queueMutation {
		orchestrator.queueMu.Unlock()
	}
}

func (orchestrator *Orchestrator) flushLogs(execution *runExecution) {
	if len(execution.logs) == 0 {
		return
	}
	logs := append([]work.LogChunk(nil), execution.logs...)
	if committed := commitStore(orchestrator.store, Commit{Logs: logs}); !committed.OK {
		orchestrator.failExecution(execution, committed.Error())
		return
	}
	execution.logs = execution.logs[:0]
	execution.logBytes = 0
}

func (orchestrator *Orchestrator) failExecution(execution *runExecution, reason string) {
	if execution.failure != "" {
		return
	}
	execution.failure = core.Trim(reason)
	if execution.process != nil {
		if shutdown := execution.process.Shutdown(); !shutdown.OK {
			execution.failure = core.Join("; ", execution.failure, shutdown.Error())
		}
	}
}

func (orchestrator *Orchestrator) finishExecution(execution *runExecution, waited core.Result) {
	cleanupContext, cancelCleanup := orchestrator.cleanupContext()
	defer cancelCleanup()
	runResult := orchestrator.store.Run(execution.run.ID)
	run := execution.run
	if runResult.OK {
		if stored, ok := runResult.Value.(work.Run); ok {
			run = stored
		}
	}
	exitCode := -1
	if waited.OK {
		if code, ok := waited.Value.(int); ok {
			exitCode = code
		} else {
			execution.failure = core.Join("; ", execution.failure, core.Sprintf("launcher wait returned %T instead of exit code", waited.Value))
		}
	} else {
		execution.failure = core.Join("; ", execution.failure, waited.Error())
	}
	run.ExitCode = exitCode

	captured := orchestrator.workspaces.CaptureRun(cleanupContext, execution.workspace)
	var capture workspace.Capture
	if captured.OK {
		var captureOK bool
		capture, captureOK = captured.Value.(workspace.Capture)
		if !captureOK {
			execution.failure = core.Join("; ", execution.failure, core.Sprintf("workspace returned %T instead of capture", captured.Value))
		} else {
			run.DurableRevision = capture.DurableRevision
		}
		if capture.Revision != "" {
			run.ExecutionRevision = capture.Revision
		}
		if !capture.Pushed {
			execution.failure = core.Join("; ", execution.failure, capture.Summary)
		}
	} else {
		execution.failure = core.Join("; ", execution.failure, captured.Error())
	}

	status := work.RunCompleted
	title := "native provider completed"
	var question *work.Question
	if orchestrator.isClosed() {
		status = work.RunInterrupted
		title = "native provider interrupted during shutdown"
	} else if run.Status == work.RunCancelling {
		status = work.RunCancelled
		title = "native provider cancelled"
	} else if execution.failure != "" || exitCode != 0 {
		status = work.RunFailed
		if execution.failure == "" {
			execution.failure = core.Sprintf("native provider exited with code %d", exitCode)
		}
		title = execution.failure
	} else {
		finalResult := provider.ParseFinalStatus(core.Join("\n", execution.response...))
		if !finalResult.OK {
			title = "unclassified provider finish"
		} else {
			final := finalResult.Value.(provider.FinalStatus)
			switch final.Status {
			case "waiting":
				status = work.RunWaiting
				title = final.Question
				questionID := orchestrator.nextID("question")
				atResult := orchestrator.now()
				if !questionID.OK || !atResult.OK {
					status = work.RunFailed
					execution.failure = "failed to create durable provider question"
					title = execution.failure
				} else {
					question = &work.Question{ID: questionID.Value.(string), RunID: run.ID, Text: final.Question, CreatedAt: atResult.Value.(time.Time)}
				}
			case "failed":
				status = work.RunFailed
				execution.failure = final.Reason
				title = final.Reason
			case "completed":
				if final.Summary != "" {
					title = final.Summary
				}
			}
		}
	}
	if !capture.Pushed && status != work.RunInterrupted {
		status = work.RunFailed
		if execution.failure == "" {
			execution.failure = "agent workspace was retained because capture was not durable"
		}
		title = execution.failure
		question = nil
	}
	atResult := orchestrator.now()
	if !atResult.OK {
		return
	}
	expected := run.Status
	run.Status = status
	run.FailureReason = execution.failure
	run.FinishedAt = atResult.Value.(time.Time)
	run.UpdatedAt = run.FinishedAt
	eventKind := string(status)
	if status == work.RunFailed && execution.timedOut {
		eventKind = "timeout"
	}
	if status == work.RunCompleted && title == "unclassified provider finish" {
		eventKind = "unclassified_provider_finish"
	}
	eventResult := orchestrator.newEvent(run, eventKind, title, capture.Summary)
	if !eventResult.OK {
		return
	}
	event := eventResult.Value.(work.Event)
	if committed := orchestrator.commitTerminal(Commit{Run: &run, ExpectedStatus: &expected, Event: &event, Question: question}); !committed.OK {
		return
	}
	if capture.Pushed {
		released := orchestrator.workspaces.ReleaseRun(cleanupContext, execution.workspace)
		if !released.OK {
			orchestrator.persistWorkspaceRetained(run, released.Error())
		}
	}
	orchestrator.finishRunOwnership(run.ID)
}

func (orchestrator *Orchestrator) finishRunOwnership(runID string) {
	orchestrator.mu.Lock()
	delete(orchestrator.runs, runID)
	delete(orchestrator.executions, runID)
	delete(orchestrator.pending, runID)
	orchestrator.mu.Unlock()
	orchestrator.finishDrainingQueue()
	orchestrator.wakeQueue()
}

func (orchestrator *Orchestrator) commitTerminal(commit Commit) core.Result {
	for {
		committed := commitStore(orchestrator.store, commit)
		if committed.OK {
			return committed
		}
		if commit.Run == nil || commit.ExpectedStatus == nil {
			return committed
		}
		currentResult := orchestrator.store.Run(commit.Run.ID)
		if currentResult.OK {
			current, ok := currentResult.Value.(work.Run)
			if !ok {
				return core.Fail(core.Errorf("agent store returned %T instead of run", currentResult.Value))
			}
			if current.Status == commit.Run.Status || terminalOwnershipStatus(current.Status) {
				return core.Ok(nil)
			}
			if current.Status == work.RunCancelling && *commit.ExpectedStatus == work.RunRunning {
				rebuilt := orchestrator.cancelledTerminalCommit(commit, current)
				if rebuilt.OK {
					commit = rebuilt.Value.(Commit)
					continue
				}
				committed = rebuilt
			} else if current.Status != *commit.ExpectedStatus {
				return committed
			}
		}
		if !orchestrator.waitTerminalRetry() {
			return committed
		}
	}
}

func (orchestrator *Orchestrator) cancelledTerminalCommit(commit Commit, current work.Run) core.Result {
	atResult := orchestrator.now()
	if !atResult.OK {
		return atResult
	}
	run := *commit.Run
	run.Status = work.RunCancelled
	run.FinishedAt = atResult.Value.(time.Time)
	run.UpdatedAt = run.FinishedAt
	detail := ""
	if commit.Event != nil {
		detail = commit.Event.Detail
	}
	eventResult := orchestrator.newEvent(run, "cancelled", "native provider cancelled", detail)
	if !eventResult.OK {
		return eventResult
	}
	event := eventResult.Value.(work.Event)
	expected := current.Status
	commit.Run = &run
	commit.ExpectedStatus = &expected
	commit.Event = &event
	commit.Question = nil
	return core.Ok(commit)
}

func (orchestrator *Orchestrator) waitTerminalRetry() bool {
	timer := time.NewTimer(terminalRetryDelay)
	select {
	case <-orchestrator.ctx.Done():
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
		return false
	case <-timer.C:
		return true
	}
}

func (orchestrator *Orchestrator) persistWorkspaceRetained(run work.Run, reason string) {
	for attempt := 0; attempt < retainedEventAttempts; attempt++ {
		eventResult := orchestrator.newEvent(run, "workspace_retained", "captured worktree retained after cleanup failure", reason)
		if !eventResult.OK {
			continue
		}
		event := eventResult.Value.(work.Event)
		if committed := commitStore(orchestrator.store, Commit{Event: &event}); committed.OK {
			return
		}
	}
}

func (orchestrator *Orchestrator) workspaceRecovery(runID, kind string) core.Result {
	result := orchestrator.workspaces.Recovery(runID)
	if !result.OK {
		return result
	}
	receipts, ok := result.Value.([]workspace.RecoveryReceipt)
	if !ok {
		return core.Fail(core.Errorf("agent workspace returned %T instead of recovery receipts", result.Value))
	}
	for _, receipt := range receipts {
		if receipt.Kind == kind {
			return core.Ok(receipt)
		}
	}
	return core.Ok(nil)
}

func (orchestrator *Orchestrator) persistCleanupRecovery(run work.Run, receipt workspace.RecoveryReceipt, kind string, updateRun bool) core.Result {
	receiptJSON := core.JSONMarshalString(receipt)
	eventResult := orchestrator.newEvent(run, kind, "provisional workspace cleanup retained", receipt.Worktree)
	if !eventResult.OK {
		return core.Fail(core.E("orchestrator.persistCleanupRecovery", core.Concat("cleanup recovery ", receiptJSON), eventResult.Err()))
	}
	event := eventResult.Value.(work.Event)
	event.DetailJSON = receiptJSON
	commit := Commit{Event: &event}
	if updateRun {
		expected := run.Status
		commit.Run = &run
		commit.ExpectedStatus = &expected
	}
	if committed := commitStore(orchestrator.store, commit); !committed.OK {
		return core.Fail(core.E("orchestrator.persistCleanupRecovery", core.Concat("cleanup recovery ", receiptJSON), committed.Err()))
	}
	return core.Ok(event)
}

func (orchestrator *Orchestrator) finishDrainingQueue() {
	orchestrator.queueMu.Lock()
	defer orchestrator.queueMu.Unlock()
	snapshotResult := orchestrator.durableQueueSnapshot()
	if !snapshotResult.OK {
		return
	}
	snapshot := snapshotResult.Value.(work.Snapshot)
	if snapshot.Queue.Status != work.QueueDraining || activeRuns(snapshot.Runs) > 0 {
		return
	}
	atResult := orchestrator.now()
	if !atResult.OK {
		return
	}
	stopped := orchestrator.queue.Stop(0, atResult.Value.(time.Time))
	if !stopped.OK {
		return
	}
	state, ok := stopped.Value.(work.QueueState)
	if !ok {
		failure := core.Fail(core.Errorf("agent queue returned %T instead of state", stopped.Value))
		if restored := orchestrator.restoreQueue(snapshot, "orchestrator.finishDrainingQueue", failure); !restored.OK {
			return
		}
		return
	}
	committed := commitStore(orchestrator.store, Commit{Queue: &state})
	if !committed.OK {
		if restored := orchestrator.restoreQueue(snapshot, "orchestrator.finishDrainingQueue", committed); !restored.OK {
			return
		}
		return
	}
}

func (orchestrator *Orchestrator) durableQueueSnapshot() core.Result {
	snapshotResult := orchestrator.store.Snapshot("")
	if !snapshotResult.OK {
		return snapshotResult
	}
	snapshot, ok := snapshotResult.Value.(work.Snapshot)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of snapshot", snapshotResult.Value))
	}
	return core.Ok(snapshot)
}

func (orchestrator *Orchestrator) restoreQueue(snapshot work.Snapshot, operation string, failure core.Result) core.Result {
	if failure.OK {
		failure = core.Fail(core.NewError("agent queue restoration requires a failure"))
	}
	restored := orchestrator.queue.Restore(snapshot.Queue, snapshot.Providers)
	if !restored.OK {
		return core.Fail(core.E(operation, failure.Error(), restored.Err()))
	}
	return failure
}

func (orchestrator *Orchestrator) newEvent(run work.Run, kind, title, detail string) core.Result {
	idResult := orchestrator.nextID("event")
	if !idResult.OK {
		return idResult
	}
	atResult := orchestrator.now()
	if !atResult.OK {
		return atResult
	}
	kind = core.Trim(kind)
	if kind == "" {
		return core.Fail(core.NewError("agent event kind is required"))
	}
	return core.Ok(work.Event{
		ID: idResult.Value.(string), RunID: run.ID, WorkID: run.WorkID, Kind: kind,
		Title: core.Trim(title), Detail: core.Trim(detail), CreatedAt: atResult.Value.(time.Time),
	})
}

func (orchestrator *Orchestrator) wakeQueue() {
	select {
	case orchestrator.wake <- struct{}{}:
	default:
	}
}

func sameDispatchReview(review, fresh DispatchReview) bool {
	return review.Request.Work.ID == fresh.Request.Work.ID &&
		review.Request.Work.ExternalID == fresh.Request.Work.ExternalID &&
		review.Request.Work.Title == fresh.Request.Work.Title &&
		review.Request.Work.Task == fresh.Request.Work.Task &&
		review.Request.Work.Repository == fresh.Request.Work.Repository &&
		review.Request.Provider == fresh.Request.Provider && review.Request.Model == fresh.Request.Model &&
		review.Request.ConfirmedSourceRevision == fresh.Request.ConfirmedSourceRevision &&
		sameStrings(review.Request.UnsafeFlags, fresh.Request.UnsafeFlags) &&
		review.Project.ID == fresh.Project.ID && review.Project.RepositoryName == fresh.Project.RepositoryName &&
		review.Project.SourcePath == fresh.Project.SourcePath && review.Project.RepositoryRoot == fresh.Project.RepositoryRoot &&
		review.Project.SourceBranch == fresh.Project.SourceBranch && review.Project.SourceRevision == fresh.Project.SourceRevision &&
		review.Project.ClonePath == fresh.Project.ClonePath && review.Project.CreatedAt.Equal(fresh.Project.CreatedAt) &&
		review.Project.UpdatedAt.Equal(fresh.Project.UpdatedAt) &&
		sameSourceReview(review.Source, fresh.Source) &&
		review.Detection.Provider == fresh.Detection.Provider && review.Detection.Executable == fresh.Detection.Executable &&
		review.Detection.Version == fresh.Detection.Version && review.Detection.Available == fresh.Detection.Available &&
		review.Detection.Reason == fresh.Detection.Reason &&
		review.Command.Provider == fresh.Command.Provider && review.Command.Executable == fresh.Command.Executable &&
		review.Command.Dir == fresh.Command.Dir && review.Command.Receipt == fresh.Command.Receipt &&
		sameStrings(review.Command.Args, fresh.Command.Args) && sameStrings(review.Command.Environment, fresh.Command.Environment) &&
		sameStrings(review.Command.CredentialKeys, fresh.Command.CredentialKeys) &&
		review.Queue.Allowed == fresh.Queue.Allowed && review.Queue.Reason == fresh.Queue.Reason &&
		review.Queue.NotBefore.Equal(fresh.Queue.NotBefore) && review.WorktreePath == fresh.WorktreePath &&
		review.Warning == fresh.Warning
}

func sameChildReview(review, fresh ChildReview) bool {
	if review.Action != fresh.Action || review.AnswerID != fresh.AnswerID || review.RunID != fresh.RunID ||
		review.Provider != fresh.Provider || review.Model != fresh.Model || review.Branch != fresh.Branch ||
		core.JSONMarshalString(review.Work) != core.JSONMarshalString(fresh.Work) ||
		core.JSONMarshalString(review.Parent) != core.JSONMarshalString(fresh.Parent) {
		return false
	}
	return sameDispatchReview(childDispatchReview(review), childDispatchReview(fresh))
}

func childDispatchReview(review ChildReview) DispatchReview {
	return DispatchReview{
		Request: work.DispatchRequest{
			Work: review.Work, Provider: review.Provider, Model: review.Model,
			ConfirmedSourceRevision: review.Source.Revision,
		},
		Project: review.Project, Source: review.Source, Detection: review.Detection, Command: review.Command,
		Queue: review.Queue, WorktreePath: review.WorktreePath, Warning: review.Warning,
	}
}

func sameStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func renderContinuation(continuation work.Continuation) string {
	sections := []string{core.Concat("Earlier Work task:\n", core.Trim(continuation.Task))}
	logs := append([]work.LogChunk(nil), continuation.Logs...)
	core.SliceSortFunc(logs, func(left, right work.LogChunk) bool {
		return left.Sequence < right.Sequence
	})
	if len(logs) > 0 {
		lines := make([]string, 0, len(logs)+1)
		lines = append(lines, "Ordered durable output:")
		for _, log := range logs {
			lines = append(lines, core.Sprintf("%d %s: %s", log.Sequence, core.Trim(log.Stream), log.Text))
		}
		sections = append(sections, core.Join("\n", lines...))
	}
	if question := core.Trim(continuation.Question.Text); question != "" {
		sections = append(sections, core.Concat("Earlier question:\n", question))
	}
	if answer := core.Trim(continuation.Answer.Text); answer != "" {
		sections = append(sections, core.Concat("User answer:\n", answer))
	}
	return core.Join("\n\n", sections...)
}

func activeRuns(runs []work.Run) int {
	active := 0
	for _, run := range runs {
		if run.Status == work.RunPreparing || run.Status == work.RunRunning || run.Status == work.RunCancelling {
			active++
		}
	}
	return active
}

func terminalOwnershipStatus(status work.RunStatus) bool {
	switch status {
	case work.RunWaiting, work.RunCancelled, work.RunFailed, work.RunCompleted,
		work.RunInterrupted, work.RunAccepted, work.RunRejected:
		return true
	default:
		return false
	}
}

func reviewRunBranch(workID string, number int) core.Result {
	if number <= 0 {
		return core.Fail(core.NewError("agent dispatch preview run number must be positive"))
	}
	input := []byte(core.Trim(workID))
	component := make([]byte, 0, len(input))
	separator := false
	for _, character := range input {
		allowed := character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' || character >= '0' && character <= '9' || character == '_' || character == '.' || character == '-'
		if allowed {
			component = append(component, character)
			separator = false
			continue
		}
		if len(component) > 0 && !separator {
			component = append(component, '-')
			separator = true
		}
	}
	for len(component) > 0 && (component[0] == '.' || component[0] == '-') {
		component = component[1:]
	}
	for len(component) > 0 && (component[len(component)-1] == '.' || component[len(component)-1] == '-') {
		component = component[:len(component)-1]
	}
	if len(component) == 0 {
		return core.Fail(core.NewError("agent dispatch Work ID cannot form a Git branch"))
	}
	return core.Ok(core.Concat("lem/work/", string(component), "/run-", core.Itoa(number)))
}
