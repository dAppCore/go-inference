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
	orchestrator.decisionMu.Lock()
	defer orchestrator.decisionMu.Unlock()
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
	reviewResult := orchestrator.workspaces.ReviewChanges(ctx, project, run, orchestrator.queue.Validation())
	if !reviewResult.OK {
		return reviewResult
	}
	review, ok := reviewResult.Value.(workspace.ChangeReview)
	if !ok {
		return core.Fail(core.Errorf("agent workspace returned %T instead of change review", reviewResult.Value))
	}
	reviewJSONResult := encodeChangeReview(review)
	if !reviewJSONResult.OK {
		return reviewJSONResult
	}
	reviewJSON := reviewJSONResult.Value.(string)
	atResult := orchestrator.acceptanceTime()
	if !atResult.OK {
		return atResult
	}
	at := atResult.Value.(time.Time)
	idResult := orchestrator.nextID("acceptance")
	eventIDResult := orchestrator.nextID("event")
	if !idResult.OK {
		return idResult
	}
	if !eventIDResult.OK {
		return eventIDResult
	}
	status := preparedReviewStatus(review)
	receipt := work.Acceptance{
		ID: idResult.Value.(string), WorkID: run.WorkID, RunID: run.ID,
		SourceBase: review.SourceRevision, AgentBase: review.AgentBase, AgentTip: review.AgentTip,
		IntegrationBranch: review.IntegrationBranch, IntegrationWorktree: review.IntegrationPath,
		ResultRevision: review.ResultRevision, Status: status, ValidationJSON: reviewJSON,
		CreatedAt: at, UpdatedAt: at,
	}
	event := work.Event{
		ID: eventIDResult.Value.(string), RunID: run.ID, WorkID: run.WorkID, Kind: "changes_reviewed",
		Title: core.Concat("agent changes review ", status), Detail: review.ResultRevision,
		DetailJSON: reviewJSON, CreatedAt: at,
	}
	if committed := commitStore(orchestrator.store, Commit{Event: &event, Acceptance: &receipt}); !committed.OK {
		return committed
	}
	return decodeChangeReview(reviewJSON)
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
	orchestrator.decisionMu.Lock()
	defer orchestrator.decisionMu.Unlock()
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
	if run.Status != work.RunCompleted {
		return core.Fail(core.NewError("agent acceptance requires a completed run"))
	}
	durableResult := orchestrator.latestPreparedReview(run.ID)
	if !durableResult.OK {
		return durableResult
	}
	durableReceipt := durableResult.Value.(work.Acceptance)
	durableReviewResult := decodeChangeReview(durableReceipt.ValidationJSON)
	if !durableReviewResult.OK {
		return durableReviewResult
	}
	durableReview := durableReviewResult.Value.(workspace.ChangeReview)
	suppliedJSON := core.JSONMarshalString(review)
	durableJSON := core.JSONMarshalString(durableReview)
	if suppliedJSON != durableJSON {
		return core.Fail(core.NewError("agent acceptance review is tampered or superseded"))
	}
	review = durableReview
	if run.SourceRevision != review.AgentBase || run.DurableRevision != review.AgentTip {
		return core.Fail(core.NewError("agent acceptance requires the unchanged completed run receipt"))
	}
	projectResult := orchestrator.store.Project(run.ProjectID)
	if !projectResult.OK {
		return projectResult
	}
	project, ok := projectResult.Value.(work.Project)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of project", projectResult.Value))
	}
	request.Review = review
	request.Project = project
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
	validationJSON := durableReceipt.ValidationJSON
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
		reconciledRun := orchestrator.store.Run(run.ID)
		if reconciledRun.OK {
			if current, currentOK := reconciledRun.Value.(work.Run); currentOK && current.Status == work.RunAccepted && current.AcceptedRevision == review.ResultRevision {
				reconciled := orchestrator.acceptanceForRun(run.ID, "accepted")
				if reconciled.OK && reconciled.Value.(work.Acceptance).ResultRevision == review.ResultRevision {
					return reconciled
				}
			}
		}
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
	orchestrator.decisionMu.Lock()
	defer orchestrator.decisionMu.Unlock()
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
	durableResult := orchestrator.latestPreparedReview(run.ID)
	if !durableResult.OK {
		return durableResult
	}
	durableReceipt := durableResult.Value.(work.Acceptance)
	reviewResult := decodeChangeReview(durableReceipt.ValidationJSON)
	if !reviewResult.OK {
		return reviewResult
	}
	review := reviewResult.Value.(workspace.ChangeReview)
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
		ID: id, WorkID: run.WorkID, RunID: run.ID, SourceBase: review.SourceRevision,
		AgentBase: review.AgentBase, AgentTip: review.AgentTip,
		IntegrationBranch: review.IntegrationBranch, IntegrationWorktree: review.IntegrationPath,
		ResultRevision: review.ResultRevision, Status: "rejected",
		ValidationJSON: durableReceipt.ValidationJSON, CreatedAt: at, UpdatedAt: at,
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
	for index := len(snapshot.Acceptances) - 1; index >= 0; index-- {
		receipt := snapshot.Acceptances[index]
		if receipt.RunID == runID && receipt.Status == status {
			return core.Ok(receipt)
		}
	}
	return core.Fail(core.NewError("agent durable decision is missing its acceptance receipt"))
}

func (orchestrator *Orchestrator) latestPreparedReview(runID string) core.Result {
	snapshotResult := orchestrator.store.Snapshot("")
	if !snapshotResult.OK {
		return snapshotResult
	}
	snapshot, ok := snapshotResult.Value.(work.Snapshot)
	if !ok {
		return core.Fail(core.Errorf("agent store returned %T instead of snapshot", snapshotResult.Value))
	}
	var latest *work.Acceptance
	for index := range snapshot.Acceptances {
		receipt := snapshot.Acceptances[index]
		if receipt.RunID != runID || !preparedAcceptanceStatus(receipt.Status) {
			continue
		}
		if latest == nil || receipt.UpdatedAt.After(latest.UpdatedAt) || receipt.UpdatedAt.Equal(latest.UpdatedAt) {
			copyReceipt := receipt
			latest = &copyReceipt
		}
	}
	if latest == nil {
		return core.Fail(core.NewError("agent completed run has no durable prepared review receipt"))
	}
	return core.Ok(*latest)
}

func preparedReviewStatus(review workspace.ChangeReview) string {
	if len(review.Conflicts) > 0 {
		return "conflicted"
	}
	for _, validation := range review.Validation {
		if !validation.Passed {
			return "validation_failed"
		}
	}
	return "prepared"
}

func preparedAcceptanceStatus(status string) bool {
	return status == "prepared" || status == "conflicted" || status == "validation_failed"
}

func encodeChangeReview(review workspace.ChangeReview) core.Result {
	encoded := core.JSONMarshal(review)
	if !encoded.OK {
		return core.Fail(core.E("orchestrator.encodeChangeReview", "failed to encode durable change review", encoded.Err()))
	}
	return core.Ok(string(encoded.Value.([]byte)))
}

func decodeChangeReview(encoded string) core.Result {
	var review workspace.ChangeReview
	decoded := core.JSONUnmarshalString(encoded, &review)
	if !decoded.OK {
		return core.Fail(core.E("orchestrator.decodeChangeReview", "failed to decode durable change review", decoded.Err()))
	}
	return core.Ok(review)
}

func (orchestrator *Orchestrator) acceptanceTime() core.Result {
	at := orchestrator.clock.Now()
	if at.IsZero() {
		return core.Fail(core.NewError("agent orchestrator clock returned zero during acceptance decision"))
	}
	return core.Ok(at)
}
