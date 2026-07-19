// SPDX-License-Identifier: EUPL-1.2

package workspace

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/queue"
	"dappco.re/go/inference/agent/work"
)

type acceptanceFixture struct {
	workspaceRunFixture
	source string
}

func newAcceptanceFixture(t *testing.T) acceptanceFixture {
	t.Helper()
	fixture := workspaceNewFixture(t)
	source := core.PathJoin(t.TempDir(), "acceptance source")
	workspaceCreateRepository(t, fixture.runner, source)
	project := workspaceRegisterRepository(t, fixture, source, "accept-project", "accept-project")
	run := work.Run{ID: "accept-run", WorkID: "accept-work", ProjectID: project.ID, Number: 1, Attempt: 1, SourceRevision: project.SourceRevision, Status: work.RunCompleted}
	prepared := workspacePrepareRun(t, fixture, project, run)
	return acceptanceFixture{workspaceRunFixture: workspaceRunFixture{workspaceFixture: fixture, project: project, run: run, prepared: prepared}, source: source}
}

func (fixture *acceptanceFixture) agentCommit(t *testing.T, path, content, message string) string {
	t.Helper()
	workspaceWriteFile(t, core.PathJoin(fixture.prepared.Path, path), content)
	workspaceRunGit(t, fixture.runner, fixture.prepared.Path, "add", "--all")
	workspaceRunGit(t, fixture.runner, fixture.prepared.Path, "commit", "-m", message)
	return workspaceRunGit(t, fixture.runner, fixture.prepared.Path, "rev-parse", "HEAD")
}

func (fixture *acceptanceFixture) sourceCommit(t *testing.T, path, content, message string) string {
	t.Helper()
	workspaceWriteFile(t, core.PathJoin(fixture.source, path), content)
	workspaceRunGit(t, fixture.runner, fixture.source, "add", "--all")
	workspaceRunGit(t, fixture.runner, fixture.source, "commit", "-m", message)
	return workspaceRunGit(t, fixture.runner, fixture.source, "rev-parse", "HEAD")
}

func (fixture *acceptanceFixture) capture(t *testing.T) string {
	t.Helper()
	result := fixture.manager.CaptureRun(context.Background(), fixture.prepared)
	core.AssertTrue(t, result.OK, result.Error())
	capture := result.Value.(Capture)
	core.AssertTrue(t, capture.Pushed, capture.Summary)
	fixture.run.DurableRevision = capture.DurableRevision
	fixture.run.Branch = fixture.prepared.Branch
	fixture.run.Worktree = fixture.prepared.Path
	return capture.DurableRevision
}

func acceptanceSourceSnapshot(t *testing.T, fixture acceptanceFixture) string {
	t.Helper()
	parts := []string{
		workspaceRunGit(t, fixture.runner, fixture.source, "symbolic-ref", "--short", "HEAD"),
		workspaceRunGit(t, fixture.runner, fixture.source, "rev-parse", "HEAD"),
		workspaceRunGit(t, fixture.runner, fixture.source, "status", "--porcelain=v1", "--untracked-files=all"),
		workspaceRunGit(t, fixture.runner, fixture.source, "diff", "--cached", "--binary"),
		workspaceRunGit(t, fixture.runner, fixture.source, "remote", "-v"),
		workspaceRunGit(t, fixture.runner, fixture.source, "for-each-ref", "--format=%(refname) %(objectname)"),
	}
	return core.Join("\n--snapshot--\n", parts...)
}

func TestAccept_ChangeReview_Good(t *testing.T) {
	review := ChangeReview{WorkID: "work", RunID: "run", ResultRevision: "result"}
	core.AssertEqual(t, "work", review.WorkID)
	core.AssertEqual(t, "run", review.RunID)
	core.AssertEqual(t, "result", review.ResultRevision)
}

func TestAccept_ChangeReview_Bad(t *testing.T) {
	review := ChangeReview{Conflicts: []string{"README.md"}}
	core.AssertEqual(t, []string{"README.md"}, review.Conflicts)
	core.AssertEqual(t, "", review.ResultRevision)
}

func TestAccept_ChangeReview_Ugly(t *testing.T) {
	review := ChangeReview{}
	core.AssertEqual(t, "", review.WorkID)
	core.AssertEqual(t, 0, len(review.Validation))
}

func TestAccept_ValidationResult_Good(t *testing.T) {
	validation := ValidationResult{Command: Command{Executable: "go", Args: []string{"test", "./..."}}, ExitCode: 0, Passed: true}
	core.AssertEqual(t, "go", validation.Command.Executable)
	core.AssertEqual(t, 0, validation.ExitCode)
	core.AssertTrue(t, validation.Passed)
}

func TestAccept_ValidationResult_Bad(t *testing.T) {
	validation := ValidationResult{ExitCode: 1, Output: "failed", Passed: false}
	core.AssertEqual(t, 1, validation.ExitCode)
	core.AssertEqual(t, "failed", validation.Output)
	core.AssertFalse(t, validation.Passed)
}

func TestAccept_ValidationResult_Ugly(t *testing.T) {
	validation := ValidationResult{}
	core.AssertEqual(t, "", validation.Receipt)
	core.AssertFalse(t, validation.Passed)
}

func TestAccept_AcceptRequest_Good(t *testing.T) {
	request := AcceptRequest{Review: ChangeReview{RunID: "run"}, Confirmed: true}
	core.AssertEqual(t, "run", request.Review.RunID)
	core.AssertTrue(t, request.Confirmed)
}

func TestAccept_AcceptRequest_Bad(t *testing.T) {
	request := AcceptRequest{Review: ChangeReview{RunID: "run"}}
	core.AssertEqual(t, "run", request.Review.RunID)
	core.AssertFalse(t, request.Confirmed)
}

func TestAccept_AcceptRequest_Ugly(t *testing.T) {
	request := AcceptRequest{}
	core.AssertEqual(t, "", request.Review.RunID)
	core.AssertFalse(t, request.Confirmed)
}

func TestAccept_Manager_ReviewChanges_Good(t *testing.T) {
	fixture := newAcceptanceFixture(t)
	first := fixture.agentCommit(t, "first.txt", "first\n", "agent first")
	second := fixture.agentCommit(t, "second.txt", "second\n", "agent second")
	core.AssertTrue(t, first != second)
	fixture.capture(t)
	before := acceptanceSourceSnapshot(t, fixture)
	result := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, []queue.Command{{Command: "git", Args: []string{"rev-parse", "--verify", "HEAD"}}})
	core.AssertTrue(t, result.OK, result.Error())
	review := result.Value.(ChangeReview)
	core.AssertEqual(t, fixture.project.SourceRevision, review.SourceRevision)
	core.AssertEqual(t, fixture.run.SourceRevision, review.AgentBase)
	core.AssertEqual(t, fixture.run.DurableRevision, review.AgentTip)
	core.AssertEqual(t, fixture.run.DurableRevision, review.ResultRevision)
	core.AssertContains(t, review.CommitLog, "agent first")
	core.AssertContains(t, review.CommitLog, "agent second")
	core.AssertEqual(t, 1, len(review.Validation))
	core.AssertTrue(t, review.Validation[0].Passed)
	core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
}

func TestAccept_Manager_ReviewChanges_Bad(t *testing.T) {
	var manager *Manager
	core.AssertFalse(t, manager.ReviewChanges(context.Background(), work.Project{}, work.Run{}, nil).OK)
	fixture := newAcceptanceFixture(t)
	fixture.agentCommit(t, "change.txt", "change\n", "agent change")
	fixture.capture(t)
	fixture.run.Status = work.RunFailed
	core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
}

func TestAccept_Manager_ReviewChanges_Ugly(t *testing.T) {
	fixture := newAcceptanceFixture(t)
	fixture.agentCommit(t, "change.txt", "change\n", "agent change")
	fixture.capture(t)
	core.AssertFalse(t, fixture.manager.ReviewChanges(nil, fixture.project, fixture.run, nil).OK)
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	core.AssertFalse(t, fixture.manager.ReviewChanges(cancelled, fixture.project, fixture.run, nil).OK)
	fixture.run.DurableRevision = ""
	core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
}

func TestAccept_Manager_RetainReview_Good(t *testing.T) {
	fixture := newAcceptanceFixture(t)
	fixture.agentCommit(t, "retained-review.txt", "retained\n", "retained review")
	fixture.capture(t)
	review := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).Value.(ChangeReview)
	result := fixture.manager.RetainReview(review)
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertEqual(t, 0, len(fixture.manager.reviews))
	core.AssertTrue(t, core.Stat(review.IntegrationPath).OK)
}

func TestAccept_Manager_RetainReview_Bad(t *testing.T) {
	fixture := newAcceptanceFixture(t)
	fixture.agentCommit(t, "forged-review.txt", "forged\n", "forged review")
	fixture.capture(t)
	review := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).Value.(ChangeReview)
	review.IntegrationBranch = "lem/integration/forged"
	core.AssertFalse(t, fixture.manager.RetainReview(review).OK)
	core.AssertEqual(t, 1, len(fixture.manager.reviews))
}

func TestAccept_Manager_RetainReview_Ugly(t *testing.T) {
	var manager *Manager
	core.AssertFalse(t, manager.RetainReview(ChangeReview{}).OK)
	fixture := newAcceptanceFixture(t)
	core.AssertFalse(t, fixture.manager.RetainReview(ChangeReview{}).OK)
}

func TestAccept_Manager_AbandonReview_Good(t *testing.T) {
	fixture := newAcceptanceFixture(t)
	fixture.agentCommit(t, "abandoned-review.txt", "abandoned\n", "abandoned review")
	fixture.capture(t)
	review := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).Value.(ChangeReview)
	result := fixture.manager.AbandonReview(context.Background(), review)
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertFalse(t, core.Stat(review.IntegrationPath).OK)
	core.AssertEqual(t, "", workspaceRunGit(t, fixture.runner, fixture.root, "--git-dir", fixture.project.ClonePath, "branch", "--list", review.IntegrationBranch))
}

func TestAccept_Manager_AbandonReview_Bad(t *testing.T) {
	fixture := newAcceptanceFixture(t)
	fixture.agentCommit(t, "mismatched-review.txt", "mismatched\n", "mismatched review")
	fixture.capture(t)
	review := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).Value.(ChangeReview)
	review.RunID = "different-run"
	core.AssertFalse(t, fixture.manager.AbandonReview(context.Background(), review).OK)
	core.AssertTrue(t, core.Stat(review.IntegrationPath).OK)
}

func TestAccept_Manager_AbandonReview_Ugly(t *testing.T) {
	var manager *Manager
	core.AssertFalse(t, manager.AbandonReview(context.Background(), ChangeReview{}).OK)
	fixture := newAcceptanceFixture(t)
	core.AssertFalse(t, fixture.manager.AbandonReview(nil, ChangeReview{}).OK)
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	core.AssertFalse(t, fixture.manager.AbandonReview(cancelled, ChangeReview{}).OK)
}

func TestAcceptManagerReviewChangesReportsGitAndBoundaryFailures(t *testing.T) {
	ready := func(t *testing.T) acceptanceFixture {
		t.Helper()
		fixture := newAcceptanceFixture(t)
		fixture.agentCommit(t, "boundary.txt", "boundary\n", "agent boundary")
		fixture.capture(t)
		return fixture
	}
	t.Run("run identity", func(t *testing.T) {
		fixture := ready(t)
		fixture.run.ProjectID = "other-project"
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("project clone", func(t *testing.T) {
		fixture := ready(t)
		fixture.project.ClonePath = core.PathJoin(t.TempDir(), "outside.git")
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("missing source", func(t *testing.T) {
		fixture := ready(t)
		fixture.project.RepositoryRoot = core.PathJoin(t.TempDir(), "missing")
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("dirty source", func(t *testing.T) {
		fixture := ready(t)
		workspaceWriteFile(t, core.PathJoin(fixture.source, "dirty.txt"), "dirty\n")
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("private repository failure", func(t *testing.T) {
		fixture := ready(t)
		fixture.server.setFailure(false, true)
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("private repository type", func(t *testing.T) {
		fixture := ready(t)
		fixture.server.setEnsureValue("not a repository")
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("durable fetch", func(t *testing.T) {
		fixture := ready(t)
		fixture.runner.setFailure(func(command Command) bool {
			return workspaceContainsArgument(command.Args, "fetch") && workspaceCommandContains(command, core.Concat("refs/heads/", fixture.run.Branch))
		})
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("durable tip", func(t *testing.T) {
		fixture := ready(t)
		fixture.run.DurableRevision = fixture.run.SourceRevision
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("durable ancestry", func(t *testing.T) {
		fixture := ready(t)
		fixture.run.SourceRevision = fixture.sourceCommit(t, "source-only.txt", "source\n", "source only")
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("review ID", func(t *testing.T) {
		fixture := ready(t)
		fixture.manager.ids = func() string { return "../review" }
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("run ID", func(t *testing.T) {
		fixture := ready(t)
		fixture.run.ID = "../run"
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("integration directory", func(t *testing.T) {
		fixture := ready(t)
		fixture.manager.files = &workspaceFaultMedium{
			Medium: fixture.files, failEnsure: core.PathJoin(fixture.project.ID, "reviews", fixture.run.ID, "review-id"),
		}
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("source fetch", func(t *testing.T) {
		fixture := ready(t)
		fixture.runner.setFailure(func(command Command) bool {
			return workspaceContainsArgument(command.Args, "fetch") && workspaceCommandContains(command, fixture.source)
		})
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("worktree add", func(t *testing.T) {
		fixture := ready(t)
		fixture.runner.setFailure(func(command Command) bool {
			return workspaceContainsArgument(command.Args, "worktree") && workspaceContainsArgument(command.Args, "add")
		})
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("commit log", func(t *testing.T) {
		fixture := ready(t)
		fixture.runner.setFailure(func(command Command) bool { return workspaceContainsArgument(command.Args, "log") })
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("integration merge", func(t *testing.T) {
		fixture := ready(t)
		fixture.runner.setFailure(func(command Command) bool { return len(command.Args) > 0 && command.Args[0] == "merge" })
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("result revision", func(t *testing.T) {
		fixture := ready(t)
		fixture.runner.setFailure(func(command Command) bool {
			return core.Contains(command.Dir, core.PathJoin("reviews", fixture.run.ID)) && workspaceCommandContains(command, "rev-parse\x00HEAD")
		})
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("result diff", func(t *testing.T) {
		fixture := ready(t)
		fixture.runner.setFailure(func(command Command) bool {
			return core.Contains(command.Dir, core.PathJoin("reviews", fixture.run.ID)) && workspaceContainsArgument(command.Args, "diff")
		})
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
}

func TestAcceptManagerReviewChangesPostAddFailureCleansProvisionalWorktreeAndBranch(t *testing.T) {
	fixture := newAcceptanceFixture(t)
	fixture.agentCommit(t, "cleanup.txt", "cleanup\n", "agent cleanup")
	fixture.capture(t)
	integrationPath := core.PathJoin(fixture.root, fixture.project.ID, "reviews", fixture.run.ID, "review-id", "worktree")
	integrationBranch := core.Concat("lem/integration/", branchComponent(fixture.run.ID), "/", branchComponent("review-id"))
	fixture.runner.setFailure(func(command Command) bool { return workspaceContainsArgument(command.Args, "log") })

	result := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil)
	core.AssertFalse(t, result.OK)
	fixture.runner.setFailure(nil)
	core.AssertFalse(t, core.Stat(integrationPath).OK)
	core.AssertEqual(t, "", workspaceRunGit(t, fixture.runner, fixture.root, "--git-dir", fixture.project.ClonePath, "branch", "--list", integrationBranch))
}

func TestAcceptReviewReplaysDivergedMultiCommitRangeInOrder(t *testing.T) {
	fixture := newAcceptanceFixture(t)
	fixture.agentCommit(t, "ordered.txt", "one\n", "ordered one")
	fixture.agentCommit(t, "ordered.txt", "one\ntwo\n", "ordered two")
	fixture.capture(t)
	sourceTip := fixture.sourceCommit(t, "source.txt", "source diverged\n", "source diverged")
	before := acceptanceSourceSnapshot(t, fixture)
	result := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil)
	core.AssertTrue(t, result.OK, result.Error())
	review := result.Value.(ChangeReview)
	core.AssertEqual(t, sourceTip, review.SourceRevision)
	core.AssertTrue(t, review.ResultRevision != fixture.run.DurableRevision)
	content := core.ReadFile(core.PathJoin(review.IntegrationPath, "ordered.txt"))
	core.AssertTrue(t, content.OK, content.Error())
	core.AssertEqual(t, "one\ntwo\n", string(content.Value.([]byte)))
	core.AssertTrue(t, core.Index(review.CommitLog, "ordered one") < core.Index(review.CommitLog, "ordered two"))
	core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
}

func TestAcceptReviewConflictAndValidationFailureLeaveSourceUntouched(t *testing.T) {
	t.Run("conflict", func(t *testing.T) {
		fixture := newAcceptanceFixture(t)
		fixture.agentCommit(t, "README.md", "agent\n", "agent conflict")
		fixture.capture(t)
		fixture.sourceCommit(t, "README.md", "source\n", "source conflict")
		before := acceptanceSourceSnapshot(t, fixture)
		beforeContent := core.ReadFile(core.PathJoin(fixture.source, "README.md"))
		core.AssertTrue(t, beforeContent.OK, beforeContent.Error())
		result := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil)
		core.AssertTrue(t, result.OK, result.Error())
		review := result.Value.(ChangeReview)
		core.AssertTrue(t, len(review.Conflicts) > 0)
		core.AssertEqual(t, "", review.ResultRevision)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
		afterContent := core.ReadFile(core.PathJoin(fixture.source, "README.md"))
		core.AssertTrue(t, afterContent.OK, afterContent.Error())
		core.AssertEqual(t, string(beforeContent.Value.([]byte)), string(afterContent.Value.([]byte)))
	})
	t.Run("validation", func(t *testing.T) {
		fixture := newAcceptanceFixture(t)
		fixture.agentCommit(t, "change.txt", "agent\n", "agent validation")
		fixture.capture(t)
		before := acceptanceSourceSnapshot(t, fixture)
		beforeContent := core.ReadFile(core.PathJoin(fixture.source, "README.md"))
		core.AssertTrue(t, beforeContent.OK, beforeContent.Error())
		result := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, []queue.Command{
			{Command: "git", Args: []string{"rev-parse", "--verify", "refs/heads/does-not-exist"}},
			{Command: "lem-command-that-does-not-exist"},
		})
		core.AssertTrue(t, result.OK, result.Error())
		review := result.Value.(ChangeReview)
		core.AssertEqual(t, 2, len(review.Validation))
		core.AssertFalse(t, review.Validation[0].Passed)
		core.AssertTrue(t, review.Validation[0].ExitCode != 0)
		core.AssertFalse(t, review.Validation[1].Passed)
		core.AssertContains(t, review.Validation[1].Output, "lem-command-that-does-not-exist")
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
		afterContent := core.ReadFile(core.PathJoin(fixture.source, "README.md"))
		core.AssertTrue(t, afterContent.OK, afterContent.Error())
		core.AssertEqual(t, string(beforeContent.Value.([]byte)), string(afterContent.Value.([]byte)))
		core.AssertFalse(t, core.Stat(core.PathJoin(fixture.source, "change.txt")).OK)
	})
}

func TestAccept_Manager_Apply_Good(t *testing.T) {
	fixture := newAcceptanceFixture(t)
	tip := fixture.agentCommit(t, "accepted.txt", "accepted\n", "agent accepted")
	fixture.capture(t)
	reviewResult := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil)
	core.AssertTrue(t, reviewResult.OK, reviewResult.Error())
	review := reviewResult.Value.(ChangeReview)
	result := fixture.manager.Apply(context.Background(), AcceptRequest{Review: review, Project: fixture.project, Confirmed: true})
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertEqual(t, tip, workspaceRunGit(t, fixture.runner, fixture.source, "rev-parse", "HEAD"))
	repeated := fixture.manager.Apply(context.Background(), AcceptRequest{Review: review, Project: fixture.project, Confirmed: true})
	core.AssertTrue(t, repeated.OK, repeated.Error())
	core.AssertEqual(t, tip, workspaceRunGit(t, fixture.runner, fixture.source, "rev-parse", "HEAD"))
}

func TestAcceptApplicationDoesNotExposeDestructiveRollback(t *testing.T) {
	fixture := newAcceptanceFixture(t)
	fixture.agentCommit(t, "retained.txt", "retained\n", "agent retained")
	fixture.capture(t)
	review := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).Value.(ChangeReview)
	applied := fixture.manager.Apply(context.Background(), AcceptRequest{Review: review, Project: fixture.project, Confirmed: true})
	core.AssertTrue(t, applied.OK, applied.Error())
	_, rollback := applied.Value.(interface {
		Rollback(context.Context) core.Result
	})
	core.AssertFalse(t, rollback)
	core.AssertEqual(t, review.ResultRevision, workspaceRunGit(t, fixture.runner, fixture.source, "rev-parse", "HEAD"))
}

func TestAccept_Manager_Apply_Bad(t *testing.T) {
	fixture := newAcceptanceFixture(t)
	fixture.agentCommit(t, "accepted.txt", "accepted\n", "agent accepted")
	fixture.capture(t)
	review := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).Value.(ChangeReview)
	before := acceptanceSourceSnapshot(t, fixture)
	core.AssertFalse(t, fixture.manager.Apply(context.Background(), AcceptRequest{Review: review}).OK)
	core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	failed := review
	failed.Validation = []ValidationResult{{Passed: false}}
	core.AssertFalse(t, fixture.manager.Apply(context.Background(), AcceptRequest{Review: failed, Project: fixture.project, Confirmed: true}).OK)
	core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
}

func TestAcceptPreparedReviewSurvivesManagerRestart(t *testing.T) {
	fixture := newAcceptanceFixture(t)
	tip := fixture.agentCommit(t, "restart.txt", "restart\n", "agent restart")
	fixture.capture(t)
	reviewResult := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil)
	core.AssertTrue(t, reviewResult.OK, reviewResult.Error())
	review := reviewResult.Value.(ChangeReview)
	reopenedResult := NewManager(ManagerOptions{
		Root: fixture.root, Files: fixture.files, Git: fixture.runner, Server: fixture.server,
		IDs: func() string { return "reopened-review" }, Now: fixture.manager.now,
	})
	core.AssertTrue(t, reopenedResult.OK, reopenedResult.Error())
	reopened := reopenedResult.Value.(*Manager)
	applied := reopened.Apply(context.Background(), AcceptRequest{Review: review, Project: fixture.project, Confirmed: true})
	core.AssertTrue(t, applied.OK, applied.Error())
	core.AssertEqual(t, tip, workspaceRunGit(t, fixture.runner, fixture.source, "rev-parse", "HEAD"))
}

func TestAccept_Manager_Apply_Ugly(t *testing.T) {
	fixture := newAcceptanceFixture(t)
	fixture.agentCommit(t, "accepted.txt", "accepted\n", "agent accepted")
	fixture.capture(t)
	review := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).Value.(ChangeReview)
	core.AssertFalse(t, fixture.manager.Apply(nil, AcceptRequest{Review: review, Project: fixture.project, Confirmed: true}).OK)
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	core.AssertFalse(t, fixture.manager.Apply(cancelled, AcceptRequest{Review: review, Project: fixture.project, Confirmed: true}).OK)
	core.AssertFalse(t, fixture.manager.Apply(context.Background(), AcceptRequest{Confirmed: true}).OK)
}

func TestAcceptManagerApplyRejectsChangedGitReceiptFacts(t *testing.T) {
	fixture := newAcceptanceFixture(t)
	fixture.agentCommit(t, "verified.txt", "verified\n", "agent verified")
	fixture.capture(t)
	review := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).Value.(ChangeReview)
	before := acceptanceSourceSnapshot(t, fixture)

	wrongProject := fixture.project
	wrongProject.SourceBranch = "other"
	core.AssertFalse(t, fixture.manager.Apply(context.Background(), AcceptRequest{Review: review, Project: wrongProject, Confirmed: true}).OK)
	badClone := fixture.project
	badClone.ClonePath = core.PathJoin(t.TempDir(), "outside.git")
	core.AssertFalse(t, fixture.manager.Apply(context.Background(), AcceptRequest{Review: review, Project: badClone, Confirmed: true}).OK)
	outside := review
	outside.IntegrationPath = core.PathJoin(t.TempDir(), "outside-worktree")
	core.AssertFalse(t, fixture.manager.Apply(context.Background(), AcceptRequest{Review: outside, Project: fixture.project, Confirmed: true}).OK)
	missing := review
	missing.IntegrationPath = core.PathJoin(fixture.root, "missing-integration")
	core.AssertFalse(t, fixture.manager.Apply(context.Background(), AcceptRequest{Review: missing, Project: fixture.project, Confirmed: true}).OK)
	wrongBranch := review
	wrongBranch.IntegrationBranch = "lem/integration/other"
	core.AssertFalse(t, fixture.manager.Apply(context.Background(), AcceptRequest{Review: wrongBranch, Project: fixture.project, Confirmed: true}).OK)
	wrongAncestry := review
	wrongAncestry.AgentBase = core.Repeat("0", 40)
	core.AssertFalse(t, fixture.manager.Apply(context.Background(), AcceptRequest{Review: wrongAncestry, Project: fixture.project, Confirmed: true}).OK)
	wrongLog := review
	wrongLog.CommitLog = core.Concat(review.CommitLog, "\ntampered")
	core.AssertFalse(t, fixture.manager.Apply(context.Background(), AcceptRequest{Review: wrongLog, Project: fixture.project, Confirmed: true}).OK)
	wrongDiff := review
	wrongDiff.Diff = core.Concat(review.Diff, "tampered")
	core.AssertFalse(t, fixture.manager.Apply(context.Background(), AcceptRequest{Review: wrongDiff, Project: fixture.project, Confirmed: true}).OK)
	workspaceWriteFile(t, core.PathJoin(review.IntegrationPath, "dirty.txt"), "dirty\n")
	core.AssertFalse(t, fixture.manager.Apply(context.Background(), AcceptRequest{Review: review, Project: fixture.project, Confirmed: true}).OK)
	core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
}

func TestAcceptManagerApplyReportsFetchAndMergeFailures(t *testing.T) {
	t.Run("fetch", func(t *testing.T) {
		fixture := newAcceptanceFixture(t)
		fixture.agentCommit(t, "fetch.txt", "fetch\n", "agent fetch")
		fixture.capture(t)
		review := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).Value.(ChangeReview)
		before := acceptanceSourceSnapshot(t, fixture)
		fixture.runner.setFailure(func(command Command) bool {
			return workspaceContainsArgument(command.Args, "fetch")
		})
		core.AssertFalse(t, fixture.manager.Apply(context.Background(), AcceptRequest{Review: review, Project: fixture.project, Confirmed: true}).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("merge", func(t *testing.T) {
		fixture := newAcceptanceFixture(t)
		fixture.agentCommit(t, "merge.txt", "merge\n", "agent merge")
		fixture.capture(t)
		review := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).Value.(ChangeReview)
		before := acceptanceSourceSnapshot(t, fixture)
		fixture.runner.setFailure(func(command Command) bool {
			return len(command.Args) > 0 && command.Args[0] == "merge"
		})
		core.AssertFalse(t, fixture.manager.Apply(context.Background(), AcceptRequest{Review: review, Project: fixture.project, Confirmed: true}).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
}

func TestAcceptApplyRejectsMovedOrDirtySourceWithoutPreparingAnything(t *testing.T) {
	t.Run("moved", func(t *testing.T) {
		fixture := newAcceptanceFixture(t)
		fixture.agentCommit(t, "agent.txt", "agent\n", "agent")
		fixture.capture(t)
		review := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).Value.(ChangeReview)
		fixture.sourceCommit(t, "moved.txt", "moved\n", "source moved")
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.Apply(context.Background(), AcceptRequest{Review: review, Project: fixture.project, Confirmed: true}).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
	t.Run("dirty", func(t *testing.T) {
		fixture := newAcceptanceFixture(t)
		fixture.agentCommit(t, "agent.txt", "agent\n", "agent")
		fixture.capture(t)
		review := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).Value.(ChangeReview)
		workspaceWriteFile(t, core.PathJoin(fixture.source, "dirty.txt"), "dirty\n")
		before := acceptanceSourceSnapshot(t, fixture)
		core.AssertFalse(t, fixture.manager.Apply(context.Background(), AcceptRequest{Review: review, Project: fixture.project, Confirmed: true}).OK)
		core.AssertEqual(t, before, acceptanceSourceSnapshot(t, fixture))
	})
}

func TestAccept_Manager_Reject_Good(t *testing.T) {
	fixture := newAcceptanceFixture(t)
	fixture.agentCommit(t, "rejected.txt", "rejected\n", "agent rejected")
	fixture.capture(t)
	review := fixture.manager.ReviewChanges(context.Background(), fixture.project, fixture.run, nil).Value.(ChangeReview)
	result := fixture.manager.Reject(context.Background(), review)
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertTrue(t, core.Stat(review.IntegrationPath).OK)
	core.AssertEqual(t, review.ResultRevision, workspaceRunGit(t, fixture.runner, review.IntegrationPath, "rev-parse", "HEAD"))
}

func TestAccept_Manager_Reject_Bad(t *testing.T) {
	var manager *Manager
	core.AssertFalse(t, manager.Reject(context.Background(), ChangeReview{}).OK)
	fixture := newAcceptanceFixture(t)
	core.AssertFalse(t, fixture.manager.Reject(context.Background(), ChangeReview{}).OK)
}

func TestAccept_Manager_Reject_Ugly(t *testing.T) {
	fixture := newAcceptanceFixture(t)
	core.AssertFalse(t, fixture.manager.Reject(nil, ChangeReview{RunID: "run"}).OK)
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	core.AssertFalse(t, fixture.manager.Reject(cancelled, ChangeReview{WorkID: "work", RunID: "run"}).OK)
}
