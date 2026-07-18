// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	"context"
	"sync"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/queue"
	"dappco.re/go/inference/agent/work"
	"dappco.re/go/inference/agent/workspace"
	coreio "dappco.re/go/io"
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

func orchestratorAcceptanceCount(store *orchestratorTestStore, status string) int {
	store.mu.Lock()
	defer store.mu.Unlock()
	count := 0
	for _, receipt := range store.acceptances {
		if receipt.Status == status {
			count++
		}
	}
	return count
}

func TestAccept_Orchestrator_ReviewChanges_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	_, run := orchestratorCompletedAcceptanceRun(t, fixture)
	result := fixture.orchestrator.ReviewChanges(context.Background(), run.ID)
	core.AssertTrue(t, result.OK, result.Error())
	review := result.Value.(workspace.ChangeReview)
	core.AssertEqual(t, run.ID, review.RunID)
	core.AssertEqual(t, run.DurableRevision, review.AgentTip)
	core.AssertEqual(t, work.RunCompleted, fixture.store.runs[run.ID].Status)
	core.AssertEqual(t, 1, orchestratorAcceptanceCount(fixture.store, "prepared"))
	last := fixture.store.commits[len(fixture.store.commits)-1]
	core.AssertTrue(t, last.Run == nil && last.Event != nil && last.Acceptance != nil)
	core.AssertEqual(t, "changes_reviewed", last.Event.Kind)
	core.AssertEqual(t, "prepared", last.Acceptance.Status)
	durable := decodeChangeReview(last.Acceptance.ValidationJSON)
	core.AssertTrue(t, durable.OK, durable.Error())
	core.AssertEqual(t, review, durable.Value.(workspace.ChangeReview))
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
	core.AssertEqual(t, 1, orchestratorAcceptanceCount(fixture.store, "accepted"))
	last := fixture.store.commits[len(fixture.store.commits)-1]
	core.AssertTrue(t, last.Run != nil && last.Event != nil && last.Acceptance != nil)
	repeated := fixture.orchestrator.Accept(context.Background(), workspace.AcceptRequest{Review: review, Confirmed: true})
	core.AssertTrue(t, repeated.OK, repeated.Error())
	core.AssertEqual(t, receipt, repeated.Value.(work.Acceptance))
	core.AssertEqual(t, 1, orchestratorAcceptanceCount(fixture.store, "accepted"))
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
	core.AssertEqual(t, 0, orchestratorAcceptanceCount(fixture.store, "accepted"))
	core.AssertEqual(t, 1, orchestratorAcceptanceCount(fixture.store, "prepared"))
}

func TestAcceptStoreFailureReconcilesMatchingDurableDecision(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	project, run := orchestratorCompletedAcceptanceRun(t, fixture)
	review := fixture.orchestrator.ReviewChanges(context.Background(), run.ID).Value.(workspace.ChangeReview)
	fixture.store.beforeCommit = func(commit Commit) {
		if commit.Run == nil || commit.Acceptance == nil || commit.Acceptance.Status != "accepted" {
			return
		}
		fixture.store.mu.Lock()
		fixture.store.runs[commit.Run.ID] = *commit.Run
		fixture.store.acceptances = append(fixture.store.acceptances, *commit.Acceptance)
		if commit.Event != nil {
			fixture.store.events = append(fixture.store.events, *commit.Event)
		}
		fixture.store.beforeCommit = nil
		fixture.store.mu.Unlock()
	}
	fixture.store.failNext(func(commit Commit) bool {
		return commit.Acceptance != nil && commit.Acceptance.Status == "accepted"
	})
	result := fixture.orchestrator.Accept(context.Background(), workspace.AcceptRequest{Review: review, Confirmed: true})
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertEqual(t, "accepted", result.Value.(work.Acceptance).Status)
	core.AssertEqual(t, 1, orchestratorAcceptanceCount(fixture.store, "accepted"))
	core.AssertEqual(t, review.ResultRevision, orchestratorRunGit(t, project.RepositoryRoot, "rev-parse", "HEAD"))
	core.AssertEqual(t, work.RunAccepted, fixture.store.runs[run.ID].Status)
}

func TestAcceptConcurrentCallsKeepWinningReceiptAndResult(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	project, run := orchestratorCompletedAcceptanceRun(t, fixture)
	review := fixture.orchestrator.ReviewChanges(context.Background(), run.ID).Value.(workspace.ChangeReview)
	firstCommit := make(chan struct{})
	secondCommit := make(chan struct{})
	releaseFirst := make(chan struct{})
	var once sync.Once
	fixture.store.beforeCommit = func(commit Commit) {
		if commit.Acceptance == nil {
			return
		}
		blocked := false
		once.Do(func() {
			blocked = true
			close(firstCommit)
		})
		if blocked {
			select {
			case <-secondCommit:
			case <-time.After(500 * time.Millisecond):
			}
			<-releaseFirst
			return
		}
		select {
		case <-secondCommit:
		default:
			close(secondCommit)
		}
	}
	results := make(chan core.Result, 2)
	request := workspace.AcceptRequest{Review: review, Confirmed: true}
	go func() { results <- fixture.orchestrator.Accept(context.Background(), request) }()
	<-firstCommit
	go func() { results <- fixture.orchestrator.Accept(context.Background(), request) }()
	select {
	case <-secondCommit:
	case <-time.After(750 * time.Millisecond):
	}
	close(releaseFirst)
	first := <-results
	second := <-results
	core.AssertTrue(t, first.OK, first.Error())
	core.AssertTrue(t, second.OK, second.Error())
	core.AssertEqual(t, 1, orchestratorAcceptanceCount(fixture.store, "accepted"))
	core.AssertEqual(t, review.ResultRevision, orchestratorRunGit(t, project.RepositoryRoot, "rev-parse", "HEAD"))
	core.AssertEqual(t, work.RunAccepted, fixture.store.runs[run.ID].Status)
}

func TestAcceptReturnedValidationMutationCannotAuthorizeApply(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	project, run := orchestratorCompletedAcceptanceRun(t, fixture)
	queueResult := queue.NewController(queue.Policy{
		Version: 1, Dispatch: queue.DispatchConfig{DefaultAgent: "fake", GlobalConcurrency: 1, TimeoutMinutes: 60,
			Validation: []queue.Command{{Command: "git", Args: []string{"rev-parse", "--verify", "refs/heads/missing"}}}},
		Concurrency: map[string]queue.ConcurrencyLimit{"fake": {Total: 1}}, Rates: map[string]queue.RateConfig{}, Providers: map[string]queue.NativeConfig{},
	}, fixture.store.queue, nil)
	core.AssertTrue(t, queueResult.OK, queueResult.Error())
	fixture.orchestrator.queue = queueResult.Value.(*queue.Controller)
	reviewResult := fixture.orchestrator.ReviewChanges(context.Background(), run.ID)
	core.AssertTrue(t, reviewResult.OK, reviewResult.Error())
	review := reviewResult.Value.(workspace.ChangeReview)
	core.AssertFalse(t, review.Validation[0].Passed)
	before := orchestratorAcceptanceSourceSnapshot(t, project.RepositoryRoot)
	review.Validation[0].Passed = true
	review.Validation[0].Command.Args[0] = "status"
	result := fixture.orchestrator.Accept(context.Background(), workspace.AcceptRequest{Review: review, Confirmed: true})
	core.AssertFalse(t, result.OK)
	core.AssertEqual(t, before, orchestratorAcceptanceSourceSnapshot(t, project.RepositoryRoot))
}

func TestAcceptDurableReviewSurvivesManagerRestart(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	project, run := orchestratorCompletedAcceptanceRun(t, fixture)
	review := fixture.orchestrator.ReviewChanges(context.Background(), run.ID).Value.(workspace.ChangeReview)
	root := core.PathDir(core.PathDir(project.ClonePath))
	files, filesErr := coreio.NewSandboxed(root)
	if filesErr != nil {
		t.Fatalf("NewSandboxed failed: %s", filesErr)
	}
	reopenedResult := workspace.NewManager(workspace.ManagerOptions{
		Root: root, Files: files, Git: fixture.git, Server: fixture.server,
		IDs: fixture.ids.New, Now: fixture.clock.Now,
	})
	core.AssertTrue(t, reopenedResult.OK, reopenedResult.Error())
	fixture.orchestrator.workspaces = reopenedResult.Value.(*workspace.Manager)
	result := fixture.orchestrator.Accept(context.Background(), workspace.AcceptRequest{Review: review, Confirmed: true})
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertEqual(t, review.ResultRevision, orchestratorRunGit(t, project.RepositoryRoot, "rev-parse", "HEAD"))
}

func TestAcceptSupersededOrTamperedDurableReviewCannotApply(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	project, run := orchestratorCompletedAcceptanceRun(t, fixture)
	first := fixture.orchestrator.ReviewChanges(context.Background(), run.ID).Value.(workspace.ChangeReview)
	second := fixture.orchestrator.ReviewChanges(context.Background(), run.ID).Value.(workspace.ChangeReview)
	before := orchestratorAcceptanceSourceSnapshot(t, project.RepositoryRoot)
	core.AssertFalse(t, fixture.orchestrator.Accept(context.Background(), workspace.AcceptRequest{Review: first, Confirmed: true}).OK)
	tampered := second
	tampered.Diff = core.Concat(tampered.Diff, "tampered")
	core.AssertFalse(t, fixture.orchestrator.Accept(context.Background(), workspace.AcceptRequest{Review: tampered, Confirmed: true}).OK)
	core.AssertEqual(t, before, orchestratorAcceptanceSourceSnapshot(t, project.RepositoryRoot))
	accepted := fixture.orchestrator.Accept(context.Background(), workspace.AcceptRequest{Review: second, Confirmed: true})
	core.AssertTrue(t, accepted.OK, accepted.Error())
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
	core.AssertEqual(t, 1, orchestratorAcceptanceCount(fixture.store, "rejected"))
}

func TestAccept_Orchestrator_Reject_Bad(t *testing.T) {
	var orchestrator *Orchestrator
	core.AssertFalse(t, orchestrator.Reject(context.Background(), "run").OK)
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.Reject(context.Background(), "missing").OK)
	_, run := orchestratorCompletedAcceptanceRun(t, fixture)
	core.AssertFalse(t, fixture.orchestrator.Reject(context.Background(), run.ID).OK)
	core.AssertEqual(t, work.RunCompleted, fixture.store.runs[run.ID].Status)
}

func TestAccept_Orchestrator_Reject_Ugly(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.Reject(nil, "run").OK)
	core.AssertFalse(t, fixture.orchestrator.Reject(context.Background(), " ").OK)
}
