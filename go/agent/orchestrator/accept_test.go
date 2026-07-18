// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/work"
	"dappco.re/go/inference/agent/workspace"
)

func orchestratorCompletedAcceptanceRun(t *testing.T, fixture *orchestratorFixture) (work.Project, work.Run) {
	t.Helper()
	_, project, _ := fixture.registerRepository()
	run := work.Run{ID: "review-run", WorkID: "work-1", ProjectID: project.ID, Provider: "fake", SourceRevision: project.SourceRevision, Number: 1, Attempt: 1, Status: work.RunCompleted, UpdatedAt: fixture.at}
	preparedResult := fixture.manager.PrepareRun(context.Background(), project, run)
	core.AssertTrue(t, preparedResult.OK, preparedResult.Error())
	prepared := preparedResult.Value.(workspace.RunWorkspace)
	core.AssertTrue(t, core.WriteFile(core.PathJoin(prepared.Path, "agent.txt"), []byte("agent\n"), 0o600).OK)
	captureResult := fixture.manager.CaptureRun(context.Background(), prepared)
	core.AssertTrue(t, captureResult.OK, captureResult.Error())
	capture := captureResult.Value.(workspace.Capture)
	core.AssertTrue(t, capture.Pushed, capture.Summary)
	run.Branch = prepared.Branch
	run.DurableRevision = capture.DurableRevision
	run.ExecutionRevision = capture.Revision
	run.Worktree = prepared.Path
	fixture.store.runs[run.ID] = run
	return project, run
}

func orchestratorAcceptanceSourceSnapshot(t *testing.T, source string) string {
	t.Helper()
	parts := []string{
		orchestratorRunGit(t, source, "symbolic-ref", "--short", "HEAD"),
		orchestratorRunGit(t, source, "rev-parse", "HEAD"),
		orchestratorRunGit(t, source, "status", "--porcelain=v1", "--untracked-files=all"),
		orchestratorRunGit(t, source, "diff", "--cached", "--binary"),
		orchestratorRunGit(t, source, "remote", "-v"),
		orchestratorRunGit(t, source, "for-each-ref", "--format=%(refname) %(objectname)"),
	}
	return core.Join("\n--snapshot--\n", parts...)
}

func TestAccept_Orchestrator_ReviewChanges_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	_, run := orchestratorCompletedAcceptanceRun(t, fixture)
	result := fixture.orchestrator.ReviewChanges(context.Background(), run.ID)
	core.AssertTrue(t, result.OK, result.Error())
	review := result.Value.(workspace.ChangeReview)
	core.AssertEqual(t, run.ID, review.RunID)
	core.AssertEqual(t, run.DurableRevision, review.AgentTip)
}

func TestAccept_Orchestrator_ReviewChanges_Bad(t *testing.T) {
	var orchestrator *Orchestrator
	core.AssertFalse(t, orchestrator.ReviewChanges(context.Background(), "run").OK)
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.ReviewChanges(context.Background(), "missing").OK)
}

func TestAccept_Orchestrator_ReviewChanges_Ugly(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.ReviewChanges(nil, "run").OK)
	core.AssertFalse(t, fixture.orchestrator.ReviewChanges(context.Background(), " ").OK)
}

func TestAccept_Orchestrator_Accept_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	_, run := orchestratorCompletedAcceptanceRun(t, fixture)
	review := fixture.orchestrator.ReviewChanges(context.Background(), run.ID).Value.(workspace.ChangeReview)
	result := fixture.orchestrator.Accept(context.Background(), workspace.AcceptRequest{Review: review, Confirmed: true})
	core.AssertTrue(t, result.OK, result.Error())
	receipt := result.Value.(work.Acceptance)
	core.AssertEqual(t, "accepted", receipt.Status)
	core.AssertEqual(t, work.RunAccepted, fixture.store.runs[run.ID].Status)
	core.AssertEqual(t, 1, len(fixture.store.acceptances))
	last := fixture.store.commits[len(fixture.store.commits)-1]
	core.AssertTrue(t, last.Run != nil && last.Event != nil && last.Acceptance != nil)
	repeated := fixture.orchestrator.Accept(context.Background(), workspace.AcceptRequest{Review: review, Confirmed: true})
	core.AssertTrue(t, repeated.OK, repeated.Error())
	core.AssertEqual(t, receipt, repeated.Value.(work.Acceptance))
	core.AssertEqual(t, 1, len(fixture.store.acceptances))
}

func TestAccept_Orchestrator_Accept_Bad(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	_, run := orchestratorCompletedAcceptanceRun(t, fixture)
	review := fixture.orchestrator.ReviewChanges(context.Background(), run.ID).Value.(workspace.ChangeReview)
	core.AssertFalse(t, fixture.orchestrator.Accept(context.Background(), workspace.AcceptRequest{Review: review}).OK)
	core.AssertEqual(t, work.RunCompleted, fixture.store.runs[run.ID].Status)
}

func TestAccept_Orchestrator_Accept_Ugly(t *testing.T) {
	var orchestrator *Orchestrator
	core.AssertFalse(t, orchestrator.Accept(context.Background(), workspace.AcceptRequest{}).OK)
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.Accept(nil, workspace.AcceptRequest{Confirmed: true}).OK)
}

func TestAcceptStoreFailureRestoresSourceAndCompletedRun(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	project, run := orchestratorCompletedAcceptanceRun(t, fixture)
	review := fixture.orchestrator.ReviewChanges(context.Background(), run.ID).Value.(workspace.ChangeReview)
	before := orchestratorAcceptanceSourceSnapshot(t, project.RepositoryRoot)
	fixture.store.failNext(func(commit Commit) bool { return commit.Acceptance != nil })
	result := fixture.orchestrator.Accept(context.Background(), workspace.AcceptRequest{Review: review, Confirmed: true})
	core.AssertFalse(t, result.OK)
	core.AssertEqual(t, before, orchestratorAcceptanceSourceSnapshot(t, project.RepositoryRoot))
	core.AssertEqual(t, work.RunCompleted, fixture.store.runs[run.ID].Status)
	core.AssertEqual(t, 0, len(fixture.store.acceptances))
}

func TestAccept_Orchestrator_Reject_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	_, run := orchestratorCompletedAcceptanceRun(t, fixture)
	core.AssertTrue(t, fixture.orchestrator.ReviewChanges(context.Background(), run.ID).OK)
	result := fixture.orchestrator.Reject(context.Background(), run.ID)
	core.AssertTrue(t, result.OK, result.Error())
	receipt := result.Value.(work.Acceptance)
	core.AssertEqual(t, "rejected", receipt.Status)
	core.AssertEqual(t, work.RunRejected, fixture.store.runs[run.ID].Status)
	core.AssertEqual(t, 1, len(fixture.store.acceptances))
}

func TestAccept_Orchestrator_Reject_Bad(t *testing.T) {
	var orchestrator *Orchestrator
	core.AssertFalse(t, orchestrator.Reject(context.Background(), "run").OK)
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.Reject(context.Background(), "missing").OK)
}

func TestAccept_Orchestrator_Reject_Ugly(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.Reject(nil, "run").OK)
	core.AssertFalse(t, fixture.orchestrator.Reject(context.Background(), " ").OK)
}
