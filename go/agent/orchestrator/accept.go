// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	"context"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/work"
	"dappco.re/go/inference/agent/workspace"
)

// ReviewChanges prepares a mutation-free integration and validation receipt.
func (orchestrator *Orchestrator) ReviewChanges(ctx context.Context, runID string) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.lifecycle.RLock()
	defer orchestrator.lifecycle.RUnlock()
	if contextResult := validateContext(ctx, "change review"); !contextResult.OK {
		return contextResult
	}
	if orchestrator.isClosed() {
		return core.Fail(core.NewError("agent orchestrator is closed"))
	}
	runID = core.Trim(runID)
	if runID == "" {
		return core.Fail(core.NewError("agent change review requires a run ID"))
	}
	runResult := orchestrator.store.Run(runID)
	if !runResult.OK {
		return runResult
	}
	run, ok := runResult.Value.(work.Run)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of run", runResult.Value))
	}
	if run.Status != work.RunCompleted {
		return core.Fail(core.NewError("agent change review requires a completed run"))
	}
	projectResult := orchestrator.store.Project(run.ProjectID)
	if !projectResult.OK {
		return projectResult
	}
	project, ok := projectResult.Value.(work.Project)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of project", projectResult.Value))
	}
	return orchestrator.workspaces.ReviewChanges(ctx, project, run, orchestrator.queue.Validation())
}

// Accept applies a confirmed review and atomically records its durable decision.
func (orchestrator *Orchestrator) Accept(ctx context.Context, request workspace.AcceptRequest) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.lifecycle.RLock()
	defer orchestrator.lifecycle.RUnlock()
	if contextResult := validateContext(ctx, "acceptance"); !contextResult.OK {
		return contextResult
	}
	if orchestrator.isClosed() {
		return core.Fail(core.NewError("agent orchestrator is closed"))
	}
	if !request.Confirmed {
		return core.Fail(core.NewError("agent acceptance requires explicit final confirmation"))
	}
	review := request.Review
	if core.Trim(review.RunID) == "" || core.Trim(review.WorkID) == "" {
		return core.Fail(core.NewError("agent acceptance review requires Work and run IDs"))
	}
	runResult := orchestrator.store.Run(review.RunID)
	if !runResult.OK {
		return runResult
	}
	run, ok := runResult.Value.(work.Run)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of run", runResult.Value))
	}
	if run.WorkID != review.WorkID {
		return core.Fail(core.NewError("agent acceptance review does not belong to the stored run"))
	}
	if run.Status == work.RunAccepted {
		return orchestrator.acceptanceForRun(run.ID, "accepted")
	}
	if run.Status != work.RunCompleted || run.SourceRevision != review.AgentBase || run.DurableRevision != review.AgentTip {
		return core.Fail(core.NewError("agent acceptance requires the unchanged completed run receipt"))
	}
	atResult := orchestrator.acceptanceTime()
	if !atResult.OK {
		return atResult
	}
	at := atResult.Value.(time.Time)
	idResult := orchestrator.nextID("acceptance")
	if !idResult.OK {
		return idResult
	}
	id := idResult.Value.(string)
	validationResult := core.JSONMarshal(review.Validation)
	if !validationResult.OK {
		return core.Fail(core.E("orchestrator.Accept", "failed to encode validation receipt", validationResult.Err()))
	}
	validationJSON := string(validationResult.Value.([]byte))
	receipt := work.Acceptance{
		ID: id, WorkID: run.WorkID, RunID: run.ID, SourceBase: review.SourceRevision,
		AgentBase: review.AgentBase, AgentTip: review.AgentTip,
		IntegrationBranch: review.IntegrationBranch, IntegrationWorktree: review.IntegrationPath,
		ResultRevision: review.ResultRevision, Status: "accepted", ValidationJSON: validationJSON,
		CreatedAt: at, UpdatedAt: at,
	}
	expected := work.RunCompleted
	run.Status = work.RunAccepted
	run.AcceptedRevision = review.ResultRevision
	run.UpdatedAt = at
	eventIDResult := orchestrator.nextID("event")
	if !eventIDResult.OK {
		return eventIDResult
	}
	event := work.Event{
		ID: eventIDResult.Value.(string), RunID: run.ID, WorkID: run.WorkID, Kind: "accepted",
		Title: "agent changes accepted", Detail: review.ResultRevision, DetailJSON: validationJSON, CreatedAt: at,
	}
	commit := Commit{Run: &run, ExpectedStatus: &expected, Event: &event, Acceptance: &receipt}
	if validated := validateCommit(commit); !validated.OK {
		return validated
	}
	applied := orchestrator.workspaces.Apply(ctx, request)
	if !applied.OK {
		return applied
	}
	if committed := orchestrator.store.Commit(commit); !committed.OK {
		rollback, rollbackOK := applied.Value.(interface {
			Rollback(context.Context) core.Result
		})
		if !rollbackOK {
			return core.Fail(core.E("orchestrator.Accept", committed.Error(), core.NewError("workspace did not return an acceptance rollback")))
		}
		restored := rollback.Rollback(context.WithoutCancel(ctx))
		if !restored.OK {
			return core.Fail(core.E("orchestrator.Accept", committed.Error(), restored.Err()))
		}
		return committed
	}
	return core.Ok(receipt)
}

// Reject atomically records a completed run as rejected without deleting history.
func (orchestrator *Orchestrator) Reject(ctx context.Context, runID string) core.Result {
	if orchestrator == nil {
		return core.Fail(core.NewError("agent orchestrator is required"))
	}
	orchestrator.lifecycle.RLock()
	defer orchestrator.lifecycle.RUnlock()
	if contextResult := validateContext(ctx, "rejection"); !contextResult.OK {
		return contextResult
	}
	if orchestrator.isClosed() {
		return core.Fail(core.NewError("agent orchestrator is closed"))
	}
	runID = core.Trim(runID)
	if runID == "" {
		return core.Fail(core.NewError("agent rejection requires a run ID"))
	}
	runResult := orchestrator.store.Run(runID)
	if !runResult.OK {
		return runResult
	}
	run, ok := runResult.Value.(work.Run)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of run", runResult.Value))
	}
	if run.Status == work.RunRejected {
		return orchestrator.acceptanceForRun(run.ID, "rejected")
	}
	if run.Status != work.RunCompleted {
		return core.Fail(core.NewError("agent rejection requires a completed run"))
	}
	review := workspace.ChangeReview{WorkID: run.WorkID, RunID: run.ID, AgentBase: run.SourceRevision, AgentTip: run.DurableRevision}
	if rejected := orchestrator.workspaces.Reject(ctx, review); !rejected.OK {
		return rejected
	}
	atResult := orchestrator.acceptanceTime()
	if !atResult.OK {
		return atResult
	}
	at := atResult.Value.(time.Time)
	idResult := orchestrator.nextID("acceptance")
	if !idResult.OK {
		return idResult
	}
	eventIDResult := orchestrator.nextID("event")
	if !eventIDResult.OK {
		return eventIDResult
	}
	id := idResult.Value.(string)
	eventID := eventIDResult.Value.(string)
	receipt := work.Acceptance{
		ID: id, WorkID: run.WorkID, RunID: run.ID, SourceBase: run.SourceRevision,
		AgentBase: run.SourceRevision, AgentTip: run.DurableRevision, Status: "rejected",
		ValidationJSON: "[]", CreatedAt: at, UpdatedAt: at,
	}
	expected := work.RunCompleted
	run.Status = work.RunRejected
	run.UpdatedAt = at
	event := work.Event{ID: eventID, RunID: run.ID, WorkID: run.WorkID, Kind: "rejected", Title: "agent changes rejected", CreatedAt: at}
	if committed := commitStore(orchestrator.store, Commit{Run: &run, ExpectedStatus: &expected, Event: &event, Acceptance: &receipt}); !committed.OK {
		return committed
	}
	return core.Ok(receipt)
}

func (orchestrator *Orchestrator) acceptanceForRun(runID, status string) core.Result {
	snapshotResult := orchestrator.store.Snapshot("")
	if !snapshotResult.OK {
		return snapshotResult
	}
	snapshot, ok := snapshotResult.Value.(work.Snapshot)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of snapshot", snapshotResult.Value))
	}
	for _, receipt := range snapshot.Acceptances {
		if receipt.RunID == runID && receipt.Status == status {
			return core.Ok(receipt)
		}
	}
	return core.Fail(core.NewError("agent durable decision is missing its acceptance receipt"))
}

func (orchestrator *Orchestrator) acceptanceTime() core.Result {
	at := orchestrator.clock.Now()
	if at.IsZero() {
		return core.Fail(core.NewError("agent orchestrator clock returned zero during acceptance decision"))
	}
	return core.Ok(at)
}
