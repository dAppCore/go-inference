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
	eventResult := orchestrator.newEvent(run, "queued", "run queued for native admission", "")
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
		decisionResult := orchestrator.queue.Decide(queue.Candidate{
			RunID: run.ID, Provider: run.Provider, Model: run.Model, QueuedAt: run.QueuedAt,
		}, queue.Runtime{Queued: queued, Running: snapshot.Runs, Now: at.Value.(time.Time)})
		if !decisionResult.OK {
			continue
		}
		decision, ok := decisionResult.Value.(queue.Decision)
		if !ok {
			continue
		}
		if !decision.Allowed {
			if !decision.NotBefore.IsZero() && (next.IsZero() || decision.NotBefore.Before(next)) {
				next = decision.NotBefore
			}
			continue
		}
		started := orchestrator.startQueuedRun(run)
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

func (orchestrator *Orchestrator) startQueuedRun(run work.Run) core.Result {
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
	eventResult := orchestrator.newEvent(run, "preparing", "preparing isolated Git worktree", "")
	if !eventResult.OK {
		return eventResult
	}
	event := eventResult.Value.(work.Event)
	if committed := commitStore(orchestrator.store, Commit{Run: &run, ExpectedStatus: &expectedQueued, Event: &event}); !committed.OK {
		return committed
	}

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

	preparedResult := orchestrator.workspaces.PrepareRun(orchestrator.ctx, review.Project, run)
	if !preparedResult.OK {
		orchestrator.finishWithoutProcess(run, work.RunPreparing, workspace.RunWorkspace{}, preparedResult.Error())
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
	commandResult := adapter.Build(provider.Launch{
		WorkID: run.WorkID, RunID: run.ID, Title: review.Request.Work.Title, Task: review.Request.Work.Task,
		Worktree: prepared.Path, Branch: prepared.Branch, Model: run.Model, UnsafeFlags: review.Request.UnsafeFlags,
	})
	if !commandResult.OK {
		orchestrator.finishWithoutProcess(run, work.RunPreparing, prepared, commandResult.Error())
		return core.Ok(true)
	}
	command, ok := commandResult.Value.(provider.Command)
	if !ok {
		orchestrator.finishWithoutProcess(run, work.RunPreparing, prepared, core.Sprintf("provider returned %T instead of command", commandResult.Value))
		return core.Ok(true)
	}
	atResult = orchestrator.now()
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
	eventResult = orchestrator.newEvent(run, "running", "native provider admitted", command.Receipt)
	if !eventResult.OK {
		orchestrator.finishWithoutProcess(run, work.RunPreparing, prepared, eventResult.Error())
		return core.Ok(true)
	}
	event = eventResult.Value.(work.Event)
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
		captured := orchestrator.workspaces.CaptureRun(context.WithoutCancel(orchestrator.ctx), prepared)
		if captured.OK {
			capturedValue, captureOK := captured.Value.(workspace.Capture)
			if !captureOK {
				run.FailureReason = core.Join("; ", run.FailureReason, core.Sprintf("workspace returned %T instead of capture", captured.Value))
			} else {
				capture = capturedValue
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
		released := orchestrator.workspaces.ReleaseRun(context.WithoutCancel(orchestrator.ctx), prepared)
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
	for {
		select {
		case <-execution.signal:
			orchestrator.consumeLines(execution)
		case <-ticker.C:
			orchestrator.consumeLines(execution)
			orchestrator.flushLogs(execution)
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

	captured := orchestrator.workspaces.CaptureRun(context.WithoutCancel(orchestrator.ctx), execution.workspace)
	var capture workspace.Capture
	if captured.OK {
		var captureOK bool
		capture, captureOK = captured.Value.(workspace.Capture)
		if !captureOK {
			execution.failure = core.Join("; ", execution.failure, core.Sprintf("workspace returned %T instead of capture", captured.Value))
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
		released := orchestrator.workspaces.ReleaseRun(context.WithoutCancel(orchestrator.ctx), execution.workspace)
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
