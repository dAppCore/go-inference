// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/gitserver"
	"dappco.re/go/inference/agent/provider"
	"dappco.re/go/inference/agent/queue"
	"dappco.re/go/inference/agent/work"
	"dappco.re/go/inference/agent/workspace"
)

func (fixture *orchestratorFixture) reviewDispatch(item work.Item, revision string) DispatchReview {
	fixture.t.Helper()
	result := fixture.orchestrator.ReviewDispatch(context.Background(), work.DispatchRequest{
		Work: item, Provider: "fake", Model: "test-model", ConfirmedSourceRevision: revision,
	})
	core.AssertTrue(fixture.t, result.OK, result.Error())
	return result.Value.(DispatchReview)
}

func (fixture *orchestratorFixture) queueDispatch(item work.Item, revision string) work.Run {
	fixture.t.Helper()
	review := fixture.reviewDispatch(item, revision)
	result := fixture.orchestrator.Dispatch(context.Background(), review)
	core.AssertTrue(fixture.t, result.OK, result.Error())
	return result.Value.(work.Run)
}

func orchestratorWaitRunStatus(t *testing.T, store *orchestratorTestStore, runID string, statuses ...work.RunStatus) work.Run {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		result := store.Run(runID)
		if result.OK {
			run := result.Value.(work.Run)
			for _, status := range statuses {
				if run.Status == status {
					return run
				}
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	result := store.Run(runID)
	if result.OK {
		t.Fatalf("run %s remained %s; wanted one of %v", runID, result.Value.(work.Run).Status, statuses)
	}
	t.Fatalf("run %s was not found: %s", runID, result.Error())
	return work.Run{}
}

func orchestratorCompletedEnvelope(summary string) string {
	return core.Concat(`<<<LEM_STATUS>>>{"status":"completed","summary":"`, summary, `"}<<<END_LEM_STATUS>>>`)
}

func orchestratorWaitingEnvelope(question string) string {
	return core.Concat(`<<<LEM_STATUS>>>{"status":"waiting","question":"`, question, `"}<<<END_LEM_STATUS>>>`)
}

func orchestratorWaitReleased(t *testing.T, fixture *orchestratorFixture, runID string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		fixture.orchestrator.mu.Lock()
		_, running := fixture.orchestrator.runs[runID]
		_, executing := fixture.orchestrator.executions[runID]
		_, pending := fixture.orchestrator.pending[runID]
		fixture.orchestrator.mu.Unlock()
		if !running && !executing && !pending {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("run %s retained runtime ownership", runID)
}

func orchestratorTerminalParent(t *testing.T, fixture *orchestratorFixture, item work.Item, revision string, status work.RunStatus, retain bool) work.Run {
	t.Helper()
	run := fixture.queueDispatch(item, revision)
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	launch := fixture.launcher.WaitStart(t)
	running := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunRunning)
	if retain {
		fixture.git.setAfterPush(func(directory string) {
			result := core.WriteFile(core.PathJoin(directory, "retained-after-capture.txt"), []byte("retain\n"), 0o600)
			core.AssertTrue(t, result.OK, result.Error())
		})
	}
	launch.callback("stdout", "earlier durable output")
	switch status {
	case work.RunWaiting:
		launch.callback("stdout", orchestratorWaitingEnvelope("Which API should remain canonical?"))
		launch.process.Finish(0)
	case work.RunFailed:
		launch.process.Finish(7)
	case work.RunCancelled:
		cancelled := fixture.orchestrator.Cancel(context.Background(), run.ID)
		core.AssertTrue(t, cancelled.OK, cancelled.Error())
	default:
		t.Fatalf("unsupported terminal parent status %s", status)
	}
	parent := orchestratorWaitRunStatus(t, fixture.store, run.ID, status)
	orchestratorWaitReleased(t, fixture, run.ID)
	core.AssertEqual(t, running.Branch, parent.Branch)
	core.AssertEqual(t, running.Worktree, parent.Worktree)
	return parent
}

func orchestratorAssertImmutableChild(t *testing.T, fixture *orchestratorFixture, parent, child work.Run) {
	t.Helper()
	core.AssertEqual(t, work.RunQueued, child.Status)
	core.AssertEqual(t, parent.ID, child.ParentRunID)
	core.AssertEqual(t, parent.WorkID, child.WorkID)
	core.AssertEqual(t, parent.ProjectID, child.ProjectID)
	core.AssertEqual(t, parent.Number, child.Number)
	core.AssertEqual(t, parent.Attempt+1, child.Attempt)
	core.AssertEqual(t, parent.Branch, child.Branch)
	core.AssertEqual(t, parent.Worktree, child.Worktree)
	core.AssertTrue(t, parent.DurableRevision != "")
	core.AssertEqual(t, parent.DurableRevision, child.DurableRevision)
	storedParent := fixture.store.Run(parent.ID)
	core.AssertTrue(t, storedParent.OK, storedParent.Error())
	core.AssertEqual(t, parent, storedParent.Value.(work.Run))
	parentContinuation := fixture.store.Continuation(parent.ID)
	core.AssertTrue(t, parentContinuation.OK, parentContinuation.Error())
	childContinuation := fixture.store.Continuation(child.ID)
	core.AssertTrue(t, childContinuation.OK, childContinuation.Error())
	core.AssertTrue(t, parentContinuation.Value.(work.Continuation).Task != "")
	core.AssertEqual(t, parentContinuation.Value.(work.Continuation).Task, childContinuation.Value.(work.Continuation).Task)
}

func orchestratorAssertContinuationOrder(t *testing.T, continuation string, pieces ...string) {
	t.Helper()
	offset := 0
	for _, piece := range pieces {
		index := core.Index(continuation[offset:], piece)
		if index < 0 {
			t.Fatalf("continuation %q does not contain ordered piece %q", continuation, piece)
		}
		offset += index + len(piece)
	}
}

func TestRun_RepositoryControlFilesRecursiveReceipt(t *testing.T) {
	t.Run("allows normal repository files containing lem", func(t *testing.T) {
		root := t.TempDir()
		core.AssertTrue(t, core.MkdirAll(core.PathJoin(root, "docs", "nested"), 0o700).OK)
		core.AssertTrue(t, core.WriteFile(core.PathJoin(root, "docs", "problem.json"), []byte("{}\n"), 0o600).OK)
		core.AssertTrue(t, core.WriteFile(core.PathJoin(root, "docs", "nested", "implementation.md"), []byte("normal docs\n"), 0o600).OK)
		result := orchestratorRepositoryControlFiles(root)
		core.AssertTrue(t, result.OK, result.Error())
	})

	for _, path := range []string{
		"nested/.lem/record",
		"nested/.lem",
		"nested/QUESTION.md",
		"nested/deeper/lem_status.json",
		"nested/deeper/.lem-control.yaml",
		"nested/deeper/lem.status",
		"nested/deeper/lem-run.json",
		"nested/deeper/lem-log.json",
		"nested/deeper/lem-queue.yaml",
		"nested/deeper/lem-backoff.json",
		"nested/deeper/lem-acceptance.json",
	} {
		t.Run(path, func(t *testing.T) {
			root := t.TempDir()
			target := core.PathJoin(root, path)
			core.AssertTrue(t, core.MkdirAll(core.PathDir(target), 0o700).OK)
			core.AssertTrue(t, core.WriteFile(target, []byte("state\n"), 0o600).OK)
			result := orchestratorRepositoryControlFiles(root)
			core.AssertFalse(t, result.OK)
			wantPath := target
			if core.Contains(path, "/.lem/") {
				wantPath = core.PathDir(target)
			}
			core.AssertContains(t, result.Error(), wantPath)
		})
	}
}

func orchestratorRepositoryControlFiles(root string) core.Result {
	root = core.Trim(root)
	if root == "" || !core.Stat(root).OK {
		return core.Ok(nil)
	}
	return core.PathWalkDir(root, func(path string, entry core.FsDirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		name := core.Lower(entry.Name())
		if name == ".lem" {
			return core.NewError(core.Concat("unexpected LEM control path: ", path))
		}
		if entry.IsDir() {
			return nil
		}
		extension := core.Lower(core.PathExt(name))
		allowedExtension := false
		for _, allowed := range []string{".md", ".json", ".yaml", ".yml", ".status"} {
			if extension == allowed {
				allowedExtension = true
				break
			}
		}
		if !allowedExtension {
			return nil
		}
		stem := core.TrimPrefix(core.TrimSuffix(name, extension), ".")
		stem = core.Replace(core.Replace(stem, "_", "-"), ".", "-")
		controlName := false
		for _, exact := range []string{"lem", "question", "answer", "status", "state", "control"} {
			if stem == exact {
				controlName = true
				break
			}
		}
		if !controlName && core.HasPrefix(stem, "lem-") {
			controlName = true
		}
		if controlName {
			return core.NewError(core.Concat("unexpected LEM state/control file: ", path))
		}
		return nil
	})
}

func orchestratorAssertNoControlFiles(t *testing.T, roots ...string) {
	t.Helper()
	for _, root := range roots {
		if result := orchestratorRepositoryControlFiles(root); !result.OK {
			t.Fatalf("repository control-file scan failed: %s", result.Error())
		}
	}
}

func orchestratorSeedContinuation(t *testing.T, status work.RunStatus, answered bool) (*orchestratorFixture, work.Item, work.Run) {
	t.Helper()
	fixture := newOrchestratorFixture(t)
	project := storeTestProject()
	parent := storeTestRun(status)
	parent.Provider = "fake"
	parent.Model = "test-model"
	item := work.Item{
		ID: parent.WorkID, Title: "Boundary Work", Task: "Exercise child boundaries", Repository: project.SourcePath,
	}
	question := work.Question{ID: "question-boundary", RunID: parent.ID, Text: "Which boundary?", CreatedAt: fixture.at}
	fixture.store.mu.Lock()
	fixture.store.projects[project.ID] = project
	fixture.store.runs[parent.ID] = parent
	fixture.store.events = append(fixture.store.events, work.Event{
		ID: "event-boundary", RunID: parent.ID, Kind: "queued", Detail: item.Task, CreatedAt: fixture.at,
	})
	fixture.store.questions = append(fixture.store.questions, question)
	if answered {
		fixture.store.answers = append(fixture.store.answers, work.Answer{
			ID: "answer-boundary", QuestionID: question.ID, ResumeRunID: "resume-boundary",
			Text: "Use the durable path", CreatedAt: fixture.at,
		})
	}
	fixture.store.mu.Unlock()
	return fixture, item, parent
}

func TestRun_Orchestrator_ReviewDispatch_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, project, revision := fixture.registerRepository()
	result := fixture.orchestrator.ReviewDispatch(context.Background(), work.DispatchRequest{
		Work: item, Provider: "fake", Model: "model-a", ConfirmedSourceRevision: revision,
	})
	core.AssertTrue(t, result.OK, result.Error())
	review := result.Value.(DispatchReview)
	core.AssertEqual(t, project.ID, review.Project.ID)
	core.AssertEqual(t, revision, review.Source.Revision)
	core.AssertTrue(t, review.Detection.Available)
	core.AssertEqual(t, review.WorktreePath, review.Command.Dir)
	core.AssertContains(t, review.Warning, "not an OS sandbox")
	core.AssertFalse(t, review.Queue.Allowed)
	core.AssertEqual(t, 0, len(fixture.store.runs))
}

func TestRun_Orchestrator_ReviewDispatch_Bad(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(nil, work.DispatchRequest{}).OK)
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), work.DispatchRequest{}).OK)
	item, _, revision := fixture.registerRepository()
	fixture.adapter.mu.Lock()
	fixture.adapter.available = false
	fixture.adapter.mu.Unlock()
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), work.DispatchRequest{
		Work: item, Provider: "fake", ConfirmedSourceRevision: revision,
	}).OK)
}

func TestRun_Orchestrator_ReviewDispatch_Ugly(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	missingSource := work.DispatchRequest{
		Work: work.Item{
			ID: "missing-source", Title: "Missing source", Task: "Inspect the missing source",
			Repository: core.PathJoin(t.TempDir(), "does-not-exist"),
		},
		Provider: "fake", ConfirmedSourceRevision: "missing-revision",
	}
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), missingSource).OK)
	item, _, revision := fixture.registerRepository()
	core.AssertTrue(t, core.WriteFile(core.PathJoin(item.Repository, "dirty.txt"), []byte("dirty\n"), 0o600).OK)
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), work.DispatchRequest{
		Work: item, Provider: "fake", ConfirmedSourceRevision: revision,
	}).OK)
}

func TestRunReviewDispatchBoundaryFailures(t *testing.T) {
	var missing *Orchestrator
	core.AssertFalse(t, missing.ReviewDispatch(context.Background(), work.DispatchRequest{}).OK)
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	request := work.DispatchRequest{Work: item, Provider: "fake", ConfirmedSourceRevision: revision}

	wrongRevision := request
	wrongRevision.ConfirmedSourceRevision = "different"
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), wrongRevision).OK)
	unregistered := item
	unregistered.ID = "unregistered"
	unregistered.Repository = core.PathJoin(t.TempDir(), "unregistered source")
	unregisteredRevision := orchestratorCreateRepository(t, unregistered.Repository)
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), work.DispatchRequest{
		Work: unregistered, Provider: "fake", ConfirmedSourceRevision: unregisteredRevision,
	}).OK)
	adHoc := item
	adHoc.ID = "ad-hoc-review"
	adHoc.Repository = core.PathJoin(t.TempDir(), "ad hoc review")
	core.AssertTrue(t, core.MkdirAll(adHoc.Repository, 0o700).OK)
	core.AssertTrue(t, core.WriteFile(core.PathJoin(adHoc.Repository, "file.txt"), []byte("file\n"), 0o600).OK)
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), work.DispatchRequest{
		Work: adHoc, Provider: "fake", ConfirmedSourceRevision: "none",
	}).OK)

	orchestratorRunGit(t, item.Repository, "checkout", "--detach")
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), request).OK)
	orchestratorRunGit(t, item.Repository, "checkout", "main")
	fixture.store.setProjectFailure(true)
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), request).OK)
	fixture.store.setProjectFailure(false)
	fixture.store.overrideProject("wrong project")
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), request).OK)
	fixture.store.clearProjectOverride()
	missingProvider := request
	missingProvider.Provider = "missing"
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), missingProvider).OK)

	fixture.adapter.mu.Lock()
	fixture.adapter.failDetect = true
	fixture.adapter.mu.Unlock()
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), request).OK)
	fixture.adapter.mu.Lock()
	fixture.adapter.failDetect = false
	fixture.adapter.detectSet = true
	fixture.adapter.detectValue = "wrong detection"
	fixture.adapter.mu.Unlock()
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), request).OK)
	fixture.adapter.mu.Lock()
	fixture.adapter.detectValue = provider.Detection{Provider: "fake"}
	fixture.adapter.mu.Unlock()
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), request).OK)
	fixture.adapter.mu.Lock()
	fixture.adapter.detectSet = false
	fixture.adapter.mu.Unlock()

	fixture.store.overrideNext(core.Fail(core.NewError("next failed")))
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), request).OK)
	fixture.store.overrideNext(core.Ok("wrong number"))
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), request).OK)
	fixture.store.clearNextOverride()
	punctuation := request
	punctuation.Work.ID = "..."
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), punctuation).OK)

	fixture.adapter.mu.Lock()
	fixture.adapter.failBuild = true
	fixture.adapter.mu.Unlock()
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), request).OK)
	fixture.adapter.mu.Lock()
	fixture.adapter.failBuild = false
	fixture.adapter.buildSet = true
	fixture.adapter.buildValue = "wrong command"
	fixture.adapter.mu.Unlock()
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), request).OK)
	fixture.adapter.mu.Lock()
	fixture.adapter.buildSet = false
	fixture.adapter.mu.Unlock()

	fixture.store.setSnapshotFailure(true)
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), request).OK)
	fixture.store.setSnapshotFailure(false)
	fixture.store.overrideSnapshot("wrong snapshot")
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), request).OK)
	fixture.store.clearSnapshotOverride()
	at := fixture.clock.Now()
	fixture.clock.Set(time.Time{})
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), request).OK)
	fixture.clock.Set(at)
	core.AssertTrue(t, fixture.orchestrator.Close().OK)
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), request).OK)
}

func TestRun_Orchestrator_Dispatch_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	head := orchestratorRunGit(t, item.Repository, "rev-parse", "HEAD")
	review := fixture.reviewDispatch(item, revision)
	result := fixture.orchestrator.Dispatch(context.Background(), review)
	core.AssertTrue(t, result.OK, result.Error())
	run := result.Value.(work.Run)
	core.AssertEqual(t, work.RunQueued, run.Status)
	core.AssertEqual(t, 0, fixture.launcher.Count())
	stored := fixture.store.Run(run.ID)
	core.AssertTrue(t, stored.OK, stored.Error())
	core.AssertEqual(t, work.RunQueued, stored.Value.(work.Run).Status)
	core.AssertEqual(t, head, orchestratorRunGit(t, item.Repository, "rev-parse", "HEAD"))
}

func TestRun_Orchestrator_Dispatch_Bad(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.Dispatch(nil, DispatchReview{}).OK)
	core.AssertFalse(t, fixture.orchestrator.Dispatch(context.Background(), DispatchReview{}).OK)
}

func TestRun_Orchestrator_Dispatch_Ugly(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	review := fixture.reviewDispatch(item, revision)
	core.AssertTrue(t, core.WriteFile(core.PathJoin(item.Repository, "moved.txt"), []byte("moved\n"), 0o600).OK)
	core.AssertFalse(t, fixture.orchestrator.Dispatch(context.Background(), review).OK)
	core.AssertEqual(t, 0, len(fixture.store.runs))
}

func TestRunDispatchRegistrationAndIdentityFailures(t *testing.T) {
	var missing *Orchestrator
	core.AssertFalse(t, missing.Dispatch(context.Background(), DispatchReview{}).OK)
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	review := fixture.reviewDispatch(item, revision)
	fixture.server.mu.Lock()
	fixture.server.failEnsure = true
	fixture.server.mu.Unlock()
	dispatched := fixture.orchestrator.Dispatch(context.Background(), review)
	core.AssertTrue(t, dispatched.OK, dispatched.Error())
	core.AssertTrue(t, fixture.orchestrator.Cancel(context.Background(), dispatched.Value.(work.Run).ID).OK)
	fixture.server.mu.Lock()
	fixture.server.failEnsure = false
	fixture.server.mu.Unlock()
	review = fixture.reviewDispatch(item, revision)
	fixture.ids.mu.Lock()
	fixture.ids.empty = true
	fixture.ids.mu.Unlock()
	core.AssertFalse(t, fixture.orchestrator.Dispatch(context.Background(), review).OK)
	fixture.ids.mu.Lock()
	fixture.ids.empty = false
	fixture.ids.mu.Unlock()
}

func TestRunDispatchDurabilityBoundaries(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	review := fixture.reviewDispatch(item, revision)

	fixture.store.overrideNextAfter(1, core.Fail(core.NewError("injected next run failure")))
	failed := fixture.orchestrator.Dispatch(context.Background(), review)
	core.AssertFalse(t, failed.OK)
	core.AssertContains(t, failed.Error(), "next run failure")

	fixture.store.overrideNextAfter(1, core.Ok("not a run number"))
	failed = fixture.orchestrator.Dispatch(context.Background(), review)
	core.AssertFalse(t, failed.OK)
	core.AssertContains(t, failed.Error(), "invalid next run number")
	fixture.store.clearNextOverride()

	fixture.clock.ZeroAfter(2)
	failed = fixture.orchestrator.Dispatch(context.Background(), review)
	core.AssertFalse(t, failed.OK)
	core.AssertContains(t, failed.Error(), "clock")

	fixture.clock.ZeroAfter(3)
	failed = fixture.orchestrator.Dispatch(context.Background(), review)
	core.AssertFalse(t, failed.OK)
	core.AssertContains(t, failed.Error(), "clock")
	core.AssertEqual(t, 0, len(fixture.store.runs))
}

func TestRunDispatchRequiresReregisteredSourceRevision(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, project, revision := fixture.registerRepository()
	core.AssertTrue(t, core.WriteFile(core.PathJoin(item.Repository, "next.txt"), []byte("next\n"), 0o600).OK)
	orchestratorRunGit(t, item.Repository, "add", "--all")
	orchestratorRunGit(t, item.Repository, "commit", "-m", "next")
	nextRevision := orchestratorRunGit(t, item.Repository, "rev-parse", "HEAD")
	core.AssertFalse(t, nextRevision == revision)
	stale := fixture.orchestrator.ReviewDispatch(context.Background(), work.DispatchRequest{
		Work: item, Provider: "fake", ConfirmedSourceRevision: nextRevision,
	})
	core.AssertFalse(t, stale.OK)
	stored := fixture.store.Project(project.ID)
	core.AssertTrue(t, stored.OK, stored.Error())
	core.AssertEqual(t, revision, stored.Value.(work.Project).SourceRevision)

	reviewed := fixture.orchestrator.ReviewProject(context.Background(), item)
	core.AssertTrue(t, reviewed.OK, reviewed.Error())
	registered := fixture.orchestrator.RegisterProject(context.Background(), reviewed.Value.(ProjectReview), true)
	core.AssertTrue(t, registered.OK, registered.Error())
	core.AssertEqual(t, nextRevision, registered.Value.(work.Project).SourceRevision)
	refreshed := fixture.orchestrator.ReviewDispatch(context.Background(), work.DispatchRequest{
		Work: item, Provider: "fake", ConfirmedSourceRevision: nextRevision,
	})
	core.AssertTrue(t, refreshed.OK, refreshed.Error())
}

func TestRunDispatchRejectsAnyMaterialReviewChange(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	tests := []struct {
		name   string
		mutate func(*DispatchReview)
	}{
		{name: "unsafe flags", mutate: func(review *DispatchReview) { review.Request.UnsafeFlags = []string{"--dangerously-skip-permissions"} }},
		{name: "provider version", mutate: func(review *DispatchReview) { review.Detection.Version = "different" }},
		{name: "command executable", mutate: func(review *DispatchReview) { review.Command.Executable = "/different/provider" }},
		{name: "queue decision", mutate: func(review *DispatchReview) { review.Queue.Reason = "different" }},
		{name: "worktree", mutate: func(review *DispatchReview) { review.WorktreePath = "/different/worktree" }},
		{name: "warning", mutate: func(review *DispatchReview) { review.Warning = "different" }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			review := fixture.reviewDispatch(item, revision)
			test.mutate(&review)
			result := fixture.orchestrator.Dispatch(context.Background(), review)
			core.AssertFalse(t, result.OK)
			core.AssertContains(t, result.Error(), "stale")
		})
	}
}

func TestRun_Orchestrator_Answer_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	parent := orchestratorTerminalParent(t, fixture, item, revision, work.RunWaiting, false)
	result := fixture.orchestrator.Answer(context.Background(), parent.ID, "  Keep the Adapter API.  ")
	core.AssertTrue(t, result.OK, result.Error())
	answer := result.Value.(work.Answer)
	core.AssertEqual(t, "Keep the Adapter API.", answer.Text)
	core.AssertTrue(t, answer.ID != "")
	core.AssertTrue(t, answer.ResumeRunID != "")
	continuation := fixture.store.Continuation(parent.ID)
	core.AssertTrue(t, continuation.OK, continuation.Error())
	stored := continuation.Value.(work.Continuation)
	core.AssertEqual(t, stored.Question.ID, answer.QuestionID)
	core.AssertEqual(t, answer, stored.Answer)
	core.AssertEqual(t, 1, len(fixture.store.runs))
	fixture.store.mu.Lock()
	commit := fixture.store.commits[len(fixture.store.commits)-1]
	fixture.store.mu.Unlock()
	core.AssertTrue(t, commit.Answer != nil)
	core.AssertTrue(t, commit.Run == nil)
	core.AssertEqual(t, parent, fixture.store.Run(parent.ID).Value.(work.Run))
	orchestratorAssertNoControlFiles(t, item.Repository)
}

func TestRun_Orchestrator_Answer_Bad(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.Answer(nil, "run", "answer").OK)
	core.AssertFalse(t, fixture.orchestrator.Answer(context.Background(), "", "answer").OK)
	core.AssertFalse(t, fixture.orchestrator.Answer(context.Background(), "run", "   ").OK)
	item, _, revision := fixture.registerRepository()
	completed := orchestratorTerminalParent(t, fixture, item, revision, work.RunFailed, false)
	core.AssertFalse(t, fixture.orchestrator.Answer(context.Background(), completed.ID, "not waiting").OK)
}

func TestRun_Orchestrator_Answer_Ugly(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	parent := orchestratorTerminalParent(t, fixture, item, revision, work.RunWaiting, false)
	first := fixture.orchestrator.Answer(context.Background(), parent.ID, "Use v2")
	core.AssertTrue(t, first.OK, first.Error())
	duplicate := fixture.orchestrator.Answer(context.Background(), parent.ID, "Use v3")
	core.AssertFalse(t, duplicate.OK)
	core.AssertContains(t, duplicate.Error(), "already")
	continuation := fixture.store.Continuation(parent.ID).Value.(work.Continuation)
	core.AssertEqual(t, first.Value.(work.Answer), continuation.Answer)
	core.AssertEqual(t, parent, fixture.store.Run(parent.ID).Value.(work.Run))
	fixture.store.mu.Lock()
	answerCount := len(fixture.store.answers)
	fixture.store.mu.Unlock()
	core.AssertEqual(t, 1, answerCount)
}

func TestRun_Orchestrator_Resume_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	parent := orchestratorTerminalParent(t, fixture, item, revision, work.RunWaiting, false)
	answerResult := fixture.orchestrator.Answer(context.Background(), parent.ID, "Keep the Adapter API")
	core.AssertTrue(t, answerResult.OK, answerResult.Error())
	answer := answerResult.Value.(work.Answer)
	core.AssertTrue(t, fixture.orchestrator.StopQueue(context.Background()).OK)
	core.AssertFalse(t, core.Stat(parent.Worktree).OK)

	result := fixture.orchestrator.Resume(context.Background(), work.ResumeRequest{
		Work: item, ParentRunID: parent.ID, AnswerID: answer.ID, Provider: "fake", Model: "resume-model",
	})
	core.AssertTrue(t, result.OK, result.Error())
	child := result.Value.(work.Run)
	orchestratorAssertImmutableChild(t, fixture, parent, child)
	core.AssertEqual(t, answer.ResumeRunID, child.ID)
	core.AssertEqual(t, "fake", child.Provider)
	core.AssertEqual(t, "resume-model", child.Model)

	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	launch := fixture.launcher.WaitStart(t)
	core.AssertEqual(t, parent.Worktree, launch.command.Dir)
	core.AssertTrue(t, core.Stat(parent.Worktree).OK)
	fixture.adapter.mu.Lock()
	built := fixture.adapter.builds[len(fixture.adapter.builds)-1]
	fixture.adapter.mu.Unlock()
	core.AssertEqual(t, parent.Branch, built.Branch)
	core.AssertEqual(t, parent.Worktree, built.Worktree)
	orchestratorAssertContinuationOrder(t, built.Continuation,
		"Make a tested change", "earlier durable output", "Which API should remain canonical?", "Keep the Adapter API")
	orchestratorAssertNoControlFiles(t, item.Repository, parent.Worktree)
	launch.callback("stdout", orchestratorCompletedEnvelope("resumed"))
	launch.process.Finish(0)
	orchestratorWaitRunStatus(t, fixture.store, child.ID, work.RunCompleted)
	orchestratorWaitReleased(t, fixture, child.ID)

	core.AssertTrue(t, fixture.orchestrator.StopQueue(context.Background()).OK)
	freshRoot := fixture.queueDispatch(item, revision)
	core.AssertEqual(t, parent.Number+1, freshRoot.Number)
	fixture.adapter.mu.Lock()
	freshBuild := fixture.adapter.builds[len(fixture.adapter.builds)-1]
	fixture.adapter.mu.Unlock()
	core.AssertContains(t, freshBuild.Branch, "run-2")
}

func TestRun_Orchestrator_Resume_Bad(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.Resume(nil, work.ResumeRequest{}).OK)
	core.AssertFalse(t, fixture.orchestrator.Resume(context.Background(), work.ResumeRequest{}).OK)
	item, _, revision := fixture.registerRepository()
	parent := orchestratorTerminalParent(t, fixture, item, revision, work.RunWaiting, false)
	request := work.ResumeRequest{Work: item, ParentRunID: parent.ID, AnswerID: "missing", Provider: "fake"}
	core.AssertFalse(t, fixture.orchestrator.Resume(context.Background(), request).OK)
	answer := fixture.orchestrator.Answer(context.Background(), parent.ID, "Use v2")
	core.AssertTrue(t, answer.OK, answer.Error())
	request.AnswerID = "different-answer"
	core.AssertFalse(t, fixture.orchestrator.Resume(context.Background(), request).OK)
	core.AssertTrue(t, fixture.orchestrator.StopQueue(context.Background()).OK)
	request.AnswerID = answer.Value.(work.Answer).ID
	request.Work.Task = "Tampered task for the same work ID"
	tampered := fixture.orchestrator.Resume(context.Background(), request)
	core.AssertFalse(t, tampered.OK)
	core.AssertContains(t, tampered.Error(), "task")
	core.AssertEqual(t, parent, fixture.store.Run(parent.ID).Value.(work.Run))
	fixture.store.mu.Lock()
	core.AssertEqual(t, 1, len(fixture.store.runs))
	fixture.store.mu.Unlock()
}

func TestRun_Orchestrator_Resume_Ugly(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	parent := orchestratorTerminalParent(t, fixture, item, revision, work.RunWaiting, false)
	answer := fixture.orchestrator.Answer(context.Background(), parent.ID, "Use v2")
	core.AssertTrue(t, answer.OK, answer.Error())
	core.AssertTrue(t, fixture.orchestrator.StopQueue(context.Background()).OK)
	request := work.ResumeRequest{
		Work: item, ParentRunID: parent.ID, AnswerID: answer.Value.(work.Answer).ID, Provider: "fake",
	}
	first := fixture.orchestrator.Resume(context.Background(), request)
	core.AssertTrue(t, first.OK, first.Error())
	duplicate := fixture.orchestrator.Resume(context.Background(), request)
	core.AssertFalse(t, duplicate.OK)
	core.AssertEqual(t, parent, fixture.store.Run(parent.ID).Value.(work.Run))
	fixture.store.mu.Lock()
	core.AssertEqual(t, 2, len(fixture.store.runs))
	fixture.store.mu.Unlock()
}

func TestRun_Orchestrator_Retry_Good(t *testing.T) {
	statuses := []work.RunStatus{work.RunFailed, work.RunCancelled, work.RunInterrupted}
	for _, status := range statuses {
		t.Run(string(status), func(t *testing.T) {
			fixture := newOrchestratorFixture(t)
			item, project, revision := fixture.registerRepository()
			terminal := status
			if terminal == work.RunInterrupted {
				terminal = work.RunFailed
			}
			parent := orchestratorTerminalParent(t, fixture, item, revision, terminal, true)
			if status == work.RunInterrupted {
				parent.Status = work.RunInterrupted
				fixture.store.mu.Lock()
				fixture.store.runs[parent.ID] = parent
				fixture.store.mu.Unlock()
			}
			core.AssertTrue(t, fixture.orchestrator.StopQueue(context.Background()).OK)
			result := fixture.orchestrator.Retry(context.Background(), item, parent.ID)
			core.AssertTrue(t, result.OK, result.Error())
			child := result.Value.(work.Run)
			orchestratorAssertImmutableChild(t, fixture, parent, child)
			core.AssertEqual(t, parent.Provider, child.Provider)
			core.AssertEqual(t, parent.Model, child.Model)

			if status != work.RunFailed {
				return
			}
			core.AssertTrue(t, core.Stat(parent.Worktree).OK)
			core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
			launch := fixture.launcher.WaitStart(t)
			core.AssertEqual(t, parent.Worktree, launch.command.Dir)
			fixture.adapter.mu.Lock()
			built := fixture.adapter.builds[len(fixture.adapter.builds)-1]
			fixture.adapter.mu.Unlock()
			orchestratorAssertContinuationOrder(t, built.Continuation, "Make a tested change", "earlier durable output")
			worktrees := orchestratorRunGit(t, project.ClonePath, "worktree", "list", "--porcelain")
			core.AssertEqual(t, 1, core.Count(worktrees, parent.Worktree))
			orchestratorAssertNoControlFiles(t, item.Repository, parent.Worktree)
			writeResult := core.WriteFile(core.PathJoin(parent.Worktree, "durable-child.txt"), []byte("child\n"), 0o600)
			core.AssertTrue(t, writeResult.OK, writeResult.Error())
			launch.callback("stdout", orchestratorCompletedEnvelope("retried"))
			launch.process.Finish(0)
			finished := orchestratorWaitRunStatus(t, fixture.store, child.ID, work.RunCompleted, work.RunFailed)
			if finished.Status != work.RunCompleted {
				t.Fatalf("retried child failed: %s", finished.FailureReason)
			}
			core.AssertTrue(t, finished.DurableRevision != parent.DurableRevision)
			core.AssertEqual(t, parent, fixture.store.Run(parent.ID).Value.(work.Run))
		})
	}
}

func TestRun_Orchestrator_Retry_Bad(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.Retry(nil, work.Item{}, "run").OK)
	core.AssertFalse(t, fixture.orchestrator.Retry(context.Background(), work.Item{}, "").OK)
	item, _, revision := fixture.registerRepository()
	waiting := orchestratorTerminalParent(t, fixture, item, revision, work.RunWaiting, false)
	core.AssertFalse(t, fixture.orchestrator.Retry(context.Background(), item, waiting.ID).OK)
	waiting.Status = work.RunCompleted
	fixture.store.mu.Lock()
	fixture.store.runs[waiting.ID] = waiting
	fixture.store.mu.Unlock()
	core.AssertFalse(t, fixture.orchestrator.Retry(context.Background(), item, waiting.ID).OK)
	core.AssertTrue(t, fixture.orchestrator.StopQueue(context.Background()).OK)
	waiting.Status = work.RunFailed
	fixture.store.mu.Lock()
	fixture.store.runs[waiting.ID] = waiting
	fixture.store.mu.Unlock()
	tamperedItem := item
	tamperedItem.Task = "Tampered task for the same work ID"
	tampered := fixture.orchestrator.Retry(context.Background(), tamperedItem, waiting.ID)
	core.AssertFalse(t, tampered.OK)
	core.AssertContains(t, tampered.Error(), "task")
	core.AssertEqual(t, waiting, fixture.store.Run(waiting.ID).Value.(work.Run))
	fixture.store.mu.Lock()
	core.AssertEqual(t, 1, len(fixture.store.runs))
	fixture.store.mu.Unlock()
}

func TestRun_Orchestrator_Retry_Ugly(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	parent := orchestratorTerminalParent(t, fixture, item, revision, work.RunFailed, true)
	core.AssertTrue(t, fixture.orchestrator.StopQueue(context.Background()).OK)
	first := fixture.orchestrator.Retry(context.Background(), item, parent.ID)
	core.AssertTrue(t, first.OK, first.Error())
	duplicate := fixture.orchestrator.Retry(context.Background(), item, parent.ID)
	core.AssertFalse(t, duplicate.OK)
	core.AssertEqual(t, parent, fixture.store.Run(parent.ID).Value.(work.Run))
	orchestratorRunGit(t, parent.Worktree, "checkout", "-b", "wrong-retained-branch")
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	child := orchestratorWaitRunStatus(t, fixture.store, first.Value.(work.Run).ID, work.RunFailed)
	core.AssertContains(t, child.FailureReason, "expected branch")
	core.AssertEqual(t, 1, fixture.launcher.Count())
	core.AssertEqual(t, parent, fixture.store.Run(parent.ID).Value.(work.Run))
}

func TestRunAcknowledgedCaptureRevisionIsDurable(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, project, revision := fixture.registerRepository()
	run := fixture.queueDispatch(item, revision)
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	launch := fixture.launcher.WaitStart(t)
	fixture.git.setFailure(func(command workspace.Command) bool {
		return orchestratorWorkspaceCommandHasArgument(command, "update-ref") || orchestratorWorkspaceCommandHasArgument(command, "fetch")
	})
	launch.callback("stdout", orchestratorCompletedEnvelope("captured"))
	launch.process.Finish(0)
	finished := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
	fixture.git.setFailure(nil)
	core.AssertTrue(t, finished.DurableRevision != "")
	core.AssertEqual(t, finished.ExecutionRevision, finished.DurableRevision)
	core.AssertContains(t, finished.FailureReason, "tracking")
	core.AssertTrue(t, core.Stat(finished.Worktree).OK)
	remote := fixture.server.EnsureRepository(context.Background(), project.RepositoryName).Value.(gitserver.Repository)
	core.AssertEqual(t, finished.DurableRevision, orchestratorRunGit(t, remote.CloneURL, "rev-parse", core.Concat("refs/heads/", finished.Branch)))
}

func TestRunChildAttemptBoundaryFailures(t *testing.T) {
	t.Run("closed actions", func(t *testing.T) {
		fixture, item, parent := orchestratorSeedContinuation(t, work.RunWaiting, true)
		core.AssertTrue(t, fixture.orchestrator.Close().OK)
		core.AssertFalse(t, fixture.orchestrator.Answer(context.Background(), parent.ID, "answer").OK)
		core.AssertFalse(t, fixture.orchestrator.Resume(context.Background(), work.ResumeRequest{
			Work: item, ParentRunID: parent.ID, AnswerID: "answer-boundary", Provider: "fake",
		}).OK)
		core.AssertFalse(t, fixture.orchestrator.Retry(context.Background(), item, parent.ID).OK)
	})

	t.Run("continuation shapes", func(t *testing.T) {
		fixture, item, parent := orchestratorSeedContinuation(t, work.RunWaiting, true)
		fixture.store.overrideContinuation(core.Fail(core.NewError("injected continuation failure")))
		core.AssertFalse(t, fixture.orchestrator.Answer(context.Background(), parent.ID, "answer").OK)
		core.AssertFalse(t, fixture.orchestrator.Resume(context.Background(), work.ResumeRequest{
			Work: item, ParentRunID: parent.ID, AnswerID: "answer-boundary", Provider: "fake",
		}).OK)
		parent.Status = work.RunFailed
		fixture.store.mu.Lock()
		fixture.store.runs[parent.ID] = parent
		fixture.store.mu.Unlock()
		core.AssertFalse(t, fixture.orchestrator.Retry(context.Background(), item, parent.ID).OK)
		fixture.store.overrideContinuation(core.Ok("wrong continuation"))
		core.AssertFalse(t, fixture.orchestrator.Answer(context.Background(), parent.ID, "answer").OK)
		fixture.store.overrideContinuation(core.Ok(work.Continuation{Run: work.Run{ID: "different"}}))
		core.AssertFalse(t, fixture.orchestrator.Answer(context.Background(), parent.ID, "answer").OK)
		fixture.store.clearContinuationOverride()
	})

	t.Run("answer durability", func(t *testing.T) {
		fixture, _, parent := orchestratorSeedContinuation(t, work.RunWaiting, false)
		fixture.store.mu.Lock()
		fixture.store.questions = nil
		fixture.store.mu.Unlock()
		core.AssertFalse(t, fixture.orchestrator.Answer(context.Background(), parent.ID, "answer").OK)
		fixture.store.mu.Lock()
		fixture.store.questions = append(fixture.store.questions, work.Question{
			ID: "question", RunID: parent.ID, Text: "Question?", CreatedAt: fixture.at,
		})
		fixture.store.mu.Unlock()
		fixture.ids.mu.Lock()
		fixture.ids.empty = true
		fixture.ids.mu.Unlock()
		core.AssertFalse(t, fixture.orchestrator.Answer(context.Background(), parent.ID, "answer").OK)
		fixture.ids.mu.Lock()
		fixture.ids.empty = false
		fixture.ids.mu.Unlock()
		fixture.clock.Set(time.Time{})
		core.AssertFalse(t, fixture.orchestrator.Answer(context.Background(), parent.ID, "answer").OK)
		fixture.clock.Set(fixture.at)
		fixture.store.failNext(func(commit Commit) bool { return commit.Answer != nil })
		core.AssertFalse(t, fixture.orchestrator.Answer(context.Background(), parent.ID, "answer").OK)
	})

	t.Run("resume linkage", func(t *testing.T) {
		fixture, item, parent := orchestratorSeedContinuation(t, work.RunWaiting, true)
		core.AssertFalse(t, fixture.orchestrator.Resume(context.Background(), work.ResumeRequest{
			Work: item, ParentRunID: parent.ID,
		}).OK)
		parent.Status = work.RunFailed
		fixture.store.mu.Lock()
		fixture.store.runs[parent.ID] = parent
		fixture.store.mu.Unlock()
		core.AssertFalse(t, fixture.orchestrator.Resume(context.Background(), work.ResumeRequest{
			Work: item, ParentRunID: parent.ID, AnswerID: "answer-boundary", Provider: "fake",
		}).OK)
		parent.Status = work.RunWaiting
		fixture.store.mu.Lock()
		fixture.store.runs[parent.ID] = parent
		fixture.store.mu.Unlock()
		wrongWork := item
		wrongWork.ID = "different-work"
		core.AssertFalse(t, fixture.orchestrator.Resume(context.Background(), work.ResumeRequest{
			Work: wrongWork, ParentRunID: parent.ID, AnswerID: "answer-boundary", Provider: "fake",
		}).OK)
	})

	t.Run("retry identity", func(t *testing.T) {
		fixture, item, parent := orchestratorSeedContinuation(t, work.RunFailed, false)
		wrongWork := item
		wrongWork.ID = "different-work"
		core.AssertFalse(t, fixture.orchestrator.Retry(context.Background(), wrongWork, parent.ID).OK)
		fixture.ids.mu.Lock()
		fixture.ids.empty = true
		fixture.ids.mu.Unlock()
		core.AssertFalse(t, fixture.orchestrator.Retry(context.Background(), item, parent.ID).OK)
	})

	t.Run("child queue persistence", func(t *testing.T) {
		newCase := func(t *testing.T) (*orchestratorFixture, work.Item, work.Run) {
			return orchestratorSeedContinuation(t, work.RunFailed, false)
		}
		t.Run("workspace", func(t *testing.T) {
			fixture, item, parent := newCase(t)
			parent.Branch = ""
			core.AssertFalse(t, fixture.orchestrator.queueChildAttempt(item, parent, "child", "fake", "").OK)
		})
		t.Run("project missing", func(t *testing.T) {
			fixture, item, parent := newCase(t)
			parent.ProjectID = "missing"
			core.AssertFalse(t, fixture.orchestrator.queueChildAttempt(item, parent, "child", "fake", "").OK)
		})
		t.Run("project shape", func(t *testing.T) {
			fixture, item, parent := newCase(t)
			fixture.store.overrideProjectID("wrong project")
			core.AssertFalse(t, fixture.orchestrator.queueChildAttempt(item, parent, "child", "fake", "").OK)
			fixture.store.clearProjectIDOverride()
		})
		t.Run("snapshot failure", func(t *testing.T) {
			fixture, item, parent := newCase(t)
			fixture.store.setSnapshotFailure(true)
			core.AssertFalse(t, fixture.orchestrator.queueChildAttempt(item, parent, "child", "fake", "").OK)
		})
		t.Run("snapshot shape", func(t *testing.T) {
			fixture, item, parent := newCase(t)
			fixture.store.overrideSnapshot("wrong snapshot")
			core.AssertFalse(t, fixture.orchestrator.queueChildAttempt(item, parent, "child", "fake", "").OK)
		})
		t.Run("clock", func(t *testing.T) {
			fixture, item, parent := newCase(t)
			fixture.clock.Set(time.Time{})
			core.AssertFalse(t, fixture.orchestrator.queueChildAttempt(item, parent, "child", "fake", "").OK)
		})
		t.Run("identity", func(t *testing.T) {
			fixture, item, parent := newCase(t)
			core.AssertFalse(t, fixture.orchestrator.queueChildAttempt(item, parent, "", "fake", "").OK)
			core.AssertFalse(t, fixture.orchestrator.queueChildAttempt(item, parent, "child", "", "").OK)
		})
		t.Run("event", func(t *testing.T) {
			fixture, item, parent := newCase(t)
			fixture.ids.mu.Lock()
			fixture.ids.empty = true
			fixture.ids.mu.Unlock()
			core.AssertFalse(t, fixture.orchestrator.queueChildAttempt(item, parent, "child", "fake", "").OK)
		})
		t.Run("commit", func(t *testing.T) {
			fixture, item, parent := newCase(t)
			fixture.store.failNext(func(commit Commit) bool { return commit.CreateRun })
			core.AssertFalse(t, fixture.orchestrator.queueChildAttempt(item, parent, "child", "fake", "").OK)
		})
	})
}

func TestRun_Orchestrator_StartQueue_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	run := fixture.queueDispatch(item, revision)
	durable := make(chan work.RunStatus, 1)
	fixture.launcher.beforeStart = func(provider.Command) {
		stored := fixture.store.Run(run.ID)
		if stored.OK {
			durable <- stored.Value.(work.Run).Status
		}
	}
	result := fixture.orchestrator.StartQueue(context.Background())
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertEqual(t, work.QueueAccepting, result.Value.(work.QueueState).Status)
	launch := fixture.launcher.WaitStart(t)
	select {
	case status := <-durable:
		core.AssertEqual(t, work.RunRunning, status)
	case <-time.After(time.Second):
		t.Fatal("launcher did not observe durable running state")
	}
	launch.callback("stdout", orchestratorCompletedEnvelope("done"))
	launch.process.Finish(0)
	orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunCompleted)
}

func TestRun_Orchestrator_StartQueue_Bad(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.StartQueue(nil).OK)
	var orchestrator *Orchestrator
	core.AssertFalse(t, orchestrator.StartQueue(context.Background()).OK)
}

func TestRun_Orchestrator_StartQueue_Ugly(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertTrue(t, fixture.orchestrator.Close().OK)
	core.AssertFalse(t, fixture.orchestrator.StartQueue(context.Background()).OK)
}

func TestRun_Orchestrator_StopQueue_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	run := fixture.queueDispatch(item, revision)
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	launch := fixture.launcher.WaitStart(t)
	result := fixture.orchestrator.StopQueue(context.Background())
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertEqual(t, work.QueueDraining, result.Value.(work.QueueState).Status)
	launch.callback("stdout", orchestratorCompletedEnvelope("done"))
	launch.process.Finish(0)
	orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunCompleted)
	deadline := time.Now().Add(2 * time.Second)
	queueStatus := work.QueueDraining
	for queueStatus != work.QueueFrozen && time.Now().Before(deadline) {
		fixture.store.mu.Lock()
		queueStatus = fixture.store.queue.Status
		fixture.store.mu.Unlock()
		time.Sleep(10 * time.Millisecond)
	}
	core.AssertEqual(t, work.QueueFrozen, queueStatus)
}

func TestRun_Orchestrator_StopQueue_Bad(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.StopQueue(nil).OK)
	var orchestrator *Orchestrator
	core.AssertFalse(t, orchestrator.StopQueue(context.Background()).OK)
}

func TestRun_Orchestrator_StopQueue_Ugly(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	fixture.store.setSnapshotFailure(true)
	core.AssertFalse(t, fixture.orchestrator.StopQueue(context.Background()).OK)
}

func TestRunQueueBoundaryFailures(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	at := fixture.clock.Now()
	fixture.clock.Set(time.Time{})
	core.AssertFalse(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	fixture.clock.Set(at)
	fixture.store.setSnapshotFailure(true)
	core.AssertFalse(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	fixture.store.setSnapshotFailure(false)
	fixture.store.overrideSnapshot("wrong snapshot")
	core.AssertFalse(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	core.AssertFalse(t, fixture.orchestrator.StopQueue(context.Background()).OK)
	fixture.store.clearSnapshotOverride()
	core.AssertTrue(t, fixture.orchestrator.Close().OK)
	core.AssertFalse(t, fixture.orchestrator.StopQueue(context.Background()).OK)
}

func TestRun_Orchestrator_Cancel_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	queued := fixture.queueDispatch(item, revision)
	cancelled := fixture.orchestrator.Cancel(context.Background(), queued.ID)
	core.AssertTrue(t, cancelled.OK, cancelled.Error())
	core.AssertEqual(t, work.RunCancelled, cancelled.Value.(work.Run).Status)
	core.AssertEqual(t, 0, fixture.launcher.Count())

	second := fixture.queueDispatch(item, revision)
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	launch := fixture.launcher.WaitStart(t)
	live := fixture.orchestrator.Cancel(context.Background(), second.ID)
	core.AssertTrue(t, live.OK, live.Error())
	core.AssertEqual(t, work.RunCancelling, live.Value.(work.Run).Status)
	finished := orchestratorWaitRunStatus(t, fixture.store, second.ID, work.RunCancelled)
	core.AssertEqual(t, -1, finished.ExitCode)
	core.AssertTrue(t, launch.process.shutdown)
}

func TestRun_Orchestrator_Cancel_Bad(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.Cancel(nil, "").OK)
	core.AssertFalse(t, fixture.orchestrator.Cancel(context.Background(), "").OK)
	core.AssertFalse(t, fixture.orchestrator.Cancel(context.Background(), "missing").OK)
}

func TestRun_Orchestrator_Cancel_Ugly(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	run := fixture.queueDispatch(item, revision)
	core.AssertTrue(t, fixture.orchestrator.Cancel(context.Background(), run.ID).OK)
	core.AssertFalse(t, fixture.orchestrator.Cancel(context.Background(), run.ID).OK)
}

func TestRunCancelBoundaryFailures(t *testing.T) {
	var missing *Orchestrator
	core.AssertFalse(t, missing.Cancel(context.Background(), "run").OK)
	fixture := newOrchestratorFixture(t)
	fixture.store.overrideRun("wrong run")
	core.AssertFalse(t, fixture.orchestrator.Cancel(context.Background(), "run").OK)
	fixture.store.clearRunOverride()
	cancelling := storeTestRun(work.RunCancelling)
	cancelling.ID = "already-cancelling"
	fixture.store.mu.Lock()
	fixture.store.runs[cancelling.ID] = cancelling
	fixture.store.mu.Unlock()
	result := fixture.orchestrator.Cancel(context.Background(), cancelling.ID)
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertEqual(t, work.RunCancelling, result.Value.(work.Run).Status)

	queued := storeTestRun(work.RunQueued)
	queued.ID = "zero-clock"
	fixture.store.mu.Lock()
	fixture.store.runs[queued.ID] = queued
	fixture.store.mu.Unlock()
	at := fixture.clock.Now()
	fixture.clock.Set(time.Time{})
	core.AssertFalse(t, fixture.orchestrator.Cancel(context.Background(), queued.ID).OK)
	fixture.clock.Set(at)
	fixture.ids.mu.Lock()
	fixture.ids.empty = true
	fixture.ids.mu.Unlock()
	core.AssertFalse(t, fixture.orchestrator.Cancel(context.Background(), queued.ID).OK)
	fixture.ids.mu.Lock()
	fixture.ids.empty = false
	fixture.ids.mu.Unlock()

	running := storeTestRun(work.RunRunning)
	running.ID = "shutdown-failure"
	process := newOrchestratorTestProcess(88)
	process.shutdownFail = true
	fixture.store.mu.Lock()
	fixture.store.runs[running.ID] = running
	fixture.store.mu.Unlock()
	fixture.orchestrator.mu.Lock()
	fixture.orchestrator.runs[running.ID] = process
	fixture.orchestrator.mu.Unlock()
	core.AssertFalse(t, fixture.orchestrator.Cancel(context.Background(), running.ID).OK)
	fixture.orchestrator.mu.Lock()
	delete(fixture.orchestrator.runs, running.ID)
	fixture.orchestrator.mu.Unlock()
	process.shutdownFail = false
	process.Finish(-1)
}

func TestRunStreamingBatchBackoffAndFinalStatus(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	run := fixture.queueDispatch(item, revision)
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	launch := fixture.launcher.WaitStart(t)
	launch.callback("stdout", "alpha")
	launch.callback("stderr", "beta")
	launch.callback("stdout", "PROGRESS:halfway")
	launch.callback("stderr", "RATE:quota reached")
	launch.callback("stdout", orchestratorCompletedEnvelope("all done"))
	launch.process.Finish(0)
	finished := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunCompleted)
	core.AssertEqual(t, 0, finished.ExitCode)
	snapshot := fixture.store.Snapshot(item.ID).Value.(work.Snapshot)
	core.AssertTrue(t, len(snapshot.Logs) >= 5)
	for index, log := range snapshot.Logs {
		core.AssertEqual(t, int64(index+1), log.Sequence)
	}
	core.AssertEqual(t, "alpha", snapshot.Logs[0].Text)
	core.AssertEqual(t, "beta", snapshot.Logs[1].Text)
	state := fixture.store.providers["fake"]
	core.AssertTrue(t, state.BackoffUntil.Equal(fixture.at.Add(time.Hour)))
	core.AssertContains(t, state.BackoffReason, "quota")

	deferredItem := item
	deferredItem.ID = "work-deferred"
	deferredItem.Title = "Wait for provider backoff"
	deferred := fixture.queueDispatch(deferredItem, revision)
	fixture.orchestrator.wakeQueue()
	time.Sleep(100 * time.Millisecond)
	core.AssertEqual(t, 1, fixture.launcher.Count())
	deferredResult := fixture.store.Run(deferred.ID)
	core.AssertTrue(t, deferredResult.OK, deferredResult.Error())
	core.AssertEqual(t, work.RunQueued, deferredResult.Value.(work.Run).Status)
}

func TestRunAdmissionFailuresBecomeDurable(t *testing.T) {
	tests := []struct {
		name      string
		configure func(*orchestratorFixture)
		contains  string
	}{
		{
			name: "provider detection fails",
			configure: func(fixture *orchestratorFixture) {
				fixture.adapter.mu.Lock()
				fixture.adapter.failDetect = true
				fixture.adapter.mu.Unlock()
			},
			contains: "detection failure",
		},
		{
			name: "provider becomes unavailable",
			configure: func(fixture *orchestratorFixture) {
				fixture.adapter.mu.Lock()
				fixture.adapter.available = false
				fixture.adapter.mu.Unlock()
			},
			contains: "provider executable is unavailable",
		},
		{
			name: "private Git becomes unavailable",
			configure: func(fixture *orchestratorFixture) {
				fixture.server.mu.Lock()
				fixture.server.failEnsure = true
				fixture.server.mu.Unlock()
			},
			contains: "private Git failure",
		},
		{
			name: "provider build fails",
			configure: func(fixture *orchestratorFixture) {
				fixture.adapter.mu.Lock()
				fixture.adapter.failBuild = true
				fixture.adapter.mu.Unlock()
			},
			contains: "provider build failure",
		},
		{
			name: "provider returns wrong command",
			configure: func(fixture *orchestratorFixture) {
				fixture.adapter.mu.Lock()
				fixture.adapter.buildSet = true
				fixture.adapter.buildValue = "not a command"
				fixture.adapter.mu.Unlock()
			},
			contains: "instead of command",
		},
		{
			name: "launcher start fails",
			configure: func(fixture *orchestratorFixture) {
				fixture.launcher.mu.Lock()
				fixture.launcher.failStart = true
				fixture.launcher.mu.Unlock()
			},
			contains: "launcher start failure",
		},
		{
			name: "launcher returns wrong process",
			configure: func(fixture *orchestratorFixture) {
				fixture.launcher.mu.Lock()
				fixture.launcher.startSet = true
				fixture.launcher.startValue = "not a process"
				fixture.launcher.mu.Unlock()
			},
			contains: "instead of process",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fixture := newOrchestratorFixture(t)
			item, _, revision := fixture.registerRepository()
			run := fixture.queueDispatch(item, revision)
			test.configure(fixture)
			core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
			failed := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
			core.AssertContains(t, failed.FailureReason, test.contains)
		})
	}
}

func TestRunProcessBoundaryFailuresBecomeDurable(t *testing.T) {
	t.Run("process receipt cannot be persisted", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		fixture.store.failNext(func(commit Commit) bool {
			return commit.Event != nil && commit.Event.Kind == "process_started"
		})
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		launch := fixture.launcher.WaitStart(t)
		failed := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
		core.AssertContains(t, failed.FailureReason, "injected commit failure")
		core.AssertTrue(t, launch.process.shutdown)
	})

	t.Run("process receipt abort collects shutdown and wait failures", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		fixture.store.failNext(func(commit Commit) bool {
			return commit.Event != nil && commit.Event.Kind == "process_started"
		})
		fixture.launcher.mu.Lock()
		fixture.launcher.configure = func(process *orchestratorTestProcess) {
			process.shutdownFail = true
			process.waitFail = true
			process.Finish(-1)
		}
		fixture.launcher.mu.Unlock()
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		fixture.launcher.WaitStart(t)
		failed := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
		core.AssertContains(t, failed.FailureReason, "shutdown failure")
		core.AssertContains(t, failed.FailureReason, "wait failure")
	})

	tests := []struct {
		name      string
		configure func(*orchestratorTestProcess)
		line      string
		contains  string
	}{
		{
			name:      "process wait fails",
			configure: func(process *orchestratorTestProcess) { process.waitFail = true },
			line:      orchestratorCompletedEnvelope("done"),
			contains:  "wait failure",
		},
		{
			name:      "process wait returns wrong type",
			configure: func(process *orchestratorTestProcess) { process.waitValue = "not an exit code" },
			line:      orchestratorCompletedEnvelope("done"),
			contains:  "instead of exit code",
		},
		{
			name:      "provider reports failure",
			configure: func(*orchestratorTestProcess) {},
			line:      `<<<LEM_STATUS>>>{"status":"failed","reason":"provider rejected the task"}<<<END_LEM_STATUS>>>`,
			contains:  "provider rejected the task",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fixture := newOrchestratorFixture(t)
			item, _, revision := fixture.registerRepository()
			run := fixture.queueDispatch(item, revision)
			fixture.launcher.mu.Lock()
			fixture.launcher.configure = test.configure
			fixture.launcher.mu.Unlock()
			core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
			launch := fixture.launcher.WaitStart(t)
			launch.callback("stdout", test.line)
			launch.process.Finish(0)
			failed := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
			core.AssertContains(t, failed.FailureReason, test.contains)
		})
	}
}

func TestRunOutputAndQuestionPersistenceFailuresStopExecution(t *testing.T) {
	t.Run("provider progress event fails", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		launch := fixture.launcher.WaitStart(t)
		fixture.store.failNext(func(commit Commit) bool {
			return commit.Event != nil && commit.Event.Kind == "progress"
		})
		launch.callback("stdout", "PROGRESS:halfway")
		failed := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
		core.AssertContains(t, failed.FailureReason, "injected commit failure")
		core.AssertTrue(t, launch.process.shutdown)
	})

	t.Run("waiting question identity fails once", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		launch := fixture.launcher.WaitStart(t)
		fixture.ids.mu.Lock()
		fixture.ids.failNext = 1
		fixture.ids.mu.Unlock()
		launch.callback("stdout", `<<<LEM_STATUS>>>{"status":"waiting","question":"Which API?"}<<<END_LEM_STATUS>>>`)
		launch.process.Finish(0)
		failed := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
		core.AssertContains(t, failed.FailureReason, "durable provider question")
		core.AssertEqual(t, 0, len(fixture.store.questions))
	})
}

func TestRunInternalFailureBoundaries(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, reviewRunBranch("work", 0).OK)
	core.AssertFalse(t, reviewRunBranch("...", 1).OK)
	branch := reviewRunBranch(" .odd / branch. ", 2)
	core.AssertTrue(t, branch.OK, branch.Error())
	core.AssertEqual(t, "lem/work/odd-branch/run-2", branch.Value.(string))

	run := storeTestRun(work.RunQueued)
	core.AssertFalse(t, fixture.orchestrator.newEvent(run, "", "", "").OK)
	fixture.ids.mu.Lock()
	fixture.ids.failNext = 1
	fixture.ids.mu.Unlock()
	core.AssertFalse(t, fixture.orchestrator.newEvent(run, "queued", "", "").OK)
	at := fixture.clock.Now()
	fixture.clock.Set(time.Time{})
	core.AssertFalse(t, fixture.orchestrator.newEvent(run, "queued", "", "").OK)
	fixture.orchestrator.startQueuedRun(run)
	fixture.clock.Set(at)

	fixture.store.setSnapshotFailure(true)
	core.AssertTrue(t, fixture.orchestrator.drainQueue().IsZero())
	fixture.orchestrator.finishDrainingQueue()
	fixture.store.setSnapshotFailure(false)
	fixture.store.overrideSnapshot("not a snapshot")
	core.AssertTrue(t, fixture.orchestrator.drainQueue().IsZero())
	fixture.orchestrator.finishDrainingQueue()
	fixture.store.clearSnapshotOverride()
	durable := fixture.orchestrator.durableQueueSnapshot()
	core.AssertTrue(t, durable.OK, durable.Error())
	restoredFailure := fixture.orchestrator.restoreQueue(durable.Value.(work.Snapshot), "test.restore", core.Ok(nil))
	core.AssertFalse(t, restoredFailure.OK)
	core.AssertContains(t, restoredFailure.Error(), "requires a failure")
	invalidSnapshot := durable.Value.(work.Snapshot)
	invalidSnapshot.Queue.ID = "invalid"
	restoredFailure = fixture.orchestrator.restoreQueue(invalidSnapshot, "test.restore", core.Fail(core.NewError("original failure")))
	core.AssertFalse(t, restoredFailure.OK)
	core.AssertContains(t, restoredFailure.Error(), "original failure")
	core.AssertContains(t, restoredFailure.Error(), "state ID")

	process := newOrchestratorTestProcess(77)
	process.shutdownFail = true
	execution := &runExecution{process: process}
	fixture.orchestrator.failExecution(execution, "first failure")
	fixture.orchestrator.failExecution(execution, "ignored second failure")
	core.AssertContains(t, execution.failure, "shutdown failure")
	core.AssertFalse(t, core.Contains(execution.failure, "ignored"))
	process.shutdownFail = false
	process.Finish(-1)

	fixture.store.mu.Lock()
	fixture.store.queue.Status = work.QueueDraining
	fixture.store.queue.Reason = "durable drain"
	active := storeTestRun(work.RunRunning)
	active.ID = "active-drain"
	fixture.store.runs[active.ID] = active
	fixture.store.mu.Unlock()
	fixture.orchestrator.finishDrainingQueue()
	fixture.store.mu.Lock()
	delete(fixture.store.runs, active.ID)
	fixture.store.mu.Unlock()
	fixture.clock.Set(time.Time{})
	fixture.orchestrator.finishDrainingQueue()
	fixture.clock.Set(at)
	fixture.store.failNext(func(commit Commit) bool { return commit.Queue != nil })
	fixture.store.mu.Lock()
	core.AssertEqual(t, work.QueueDraining, fixture.store.queue.Status)
	fixture.store.mu.Unlock()
	fixture.orchestrator.finishDrainingQueue()
	fixture.store.mu.Lock()
	queueStatus := fixture.store.queue.Status
	failureConsumed := fixture.store.failCommitOnce == nil
	fixture.store.mu.Unlock()
	core.AssertTrue(t, failureConsumed)
	core.AssertEqual(t, work.QueueDraining, queueStatus)
	decision := fixture.queue.Decide(queue.Candidate{
		RunID: "drain-rollback", Provider: "fake", QueuedAt: at,
	}, queue.Runtime{Now: at})
	core.AssertTrue(t, decision.OK, decision.Error())
	core.AssertEqual(t, "durable drain", decision.Value.(queue.Decision).Reason)

	orphaned := storeTestRun(work.RunQueued)
	orphaned.ID = "orphaned-queue-record"
	fixture.store.mu.Lock()
	fixture.store.runs[orphaned.ID] = orphaned
	fixture.store.mu.Unlock()
	core.AssertTrue(t, fixture.orchestrator.drainQueue().IsZero())
	fixture.orchestrator.mu.Lock()
	fixture.orchestrator.pending[orphaned.ID] = DispatchReview{}
	fixture.orchestrator.mu.Unlock()
	fixture.clock.Set(time.Time{})
	core.AssertTrue(t, fixture.orchestrator.drainQueue().IsZero())
	fixture.clock.Set(at)
	fixture.store.mu.Lock()
	orphaned = fixture.store.runs[orphaned.ID]
	orphaned.QueuedAt = time.Time{}
	fixture.store.runs[orphaned.ID] = orphaned
	fixture.store.mu.Unlock()
	core.AssertTrue(t, fixture.orchestrator.drainQueue().IsZero())

	fixture.orchestrator.mu.Lock()
	fixture.orchestrator.closed = true
	fixture.orchestrator.mu.Unlock()
	fixture.orchestrator.startQueuedRun(orphaned)
	fixture.orchestrator.mu.Lock()
	fixture.orchestrator.closed = false
	fixture.orchestrator.mu.Unlock()
	orphaned.QueuedAt = at
	fixture.store.mu.Lock()
	fixture.store.runs[orphaned.ID] = orphaned
	fixture.store.mu.Unlock()
	fixture.clock.Set(time.Time{})
	fixture.orchestrator.startQueuedRun(orphaned)
	fixture.clock.Set(at)

	clockRun := storeTestRun(work.RunPreparing)
	clockRun.ID = "finish-clock-failure"
	fixture.store.mu.Lock()
	fixture.store.runs[clockRun.ID] = clockRun
	fixture.store.mu.Unlock()
	fixture.clock.ZeroAfter(1)
	fixture.orchestrator.finishWithoutProcess(clockRun, work.RunPreparing, workspace.RunWorkspace{}, "clock failure")

	shutdownRun := storeTestRun(work.RunPreparing)
	shutdownRun.ID = "finish-during-shutdown"
	fixture.store.mu.Lock()
	fixture.store.runs[shutdownRun.ID] = shutdownRun
	fixture.store.mu.Unlock()
	fixture.orchestrator.mu.Lock()
	fixture.orchestrator.closed = true
	fixture.orchestrator.mu.Unlock()
	fixture.orchestrator.finishWithoutProcess(shutdownRun, work.RunPreparing, workspace.RunWorkspace{}, "shutdown")
	fixture.orchestrator.mu.Lock()
	fixture.orchestrator.closed = false
	fixture.orchestrator.mu.Unlock()
	finished := fixture.store.Run(shutdownRun.ID)
	core.AssertTrue(t, finished.OK, finished.Error())
	core.AssertEqual(t, work.RunInterrupted, finished.Value.(work.Run).Status)

	eventRun := storeTestRun(work.RunPreparing)
	eventRun.ID = "finish-event-failure"
	fixture.store.mu.Lock()
	fixture.store.runs[eventRun.ID] = eventRun
	fixture.store.mu.Unlock()
	fixture.ids.mu.Lock()
	fixture.ids.failNext = 1
	fixture.ids.mu.Unlock()
	fixture.orchestrator.finishWithoutProcess(eventRun, work.RunPreparing, workspace.RunWorkspace{}, "event failure")
	unfinished := fixture.store.Run(eventRun.ID)
	core.AssertTrue(t, unfinished.OK, unfinished.Error())
	core.AssertEqual(t, work.RunPreparing, unfinished.Value.(work.Run).Status)
}

func TestRunMissingWorkspaceLeaseIsDurable(t *testing.T) {
	t.Run("before process start", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		run := storeTestRun(work.RunPreparing)
		run.ID = "missing-pre-process-lease"
		fixture.store.mu.Lock()
		fixture.store.runs[run.ID] = run
		fixture.store.mu.Unlock()
		fixture.orchestrator.finishWithoutProcess(run, work.RunPreparing, workspace.RunWorkspace{
			RunID: run.ID, Path: core.PathJoin(t.TempDir(), "missing-worktree"),
		}, "native launch failed")
		failed := fixture.store.Run(run.ID)
		core.AssertTrue(t, failed.OK, failed.Error())
		core.AssertEqual(t, work.RunFailed, failed.Value.(work.Run).Status)
		core.AssertContains(t, failed.Value.(work.Run).FailureReason, "lease")
	})

	t.Run("after process exit", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		run := storeTestRun(work.RunRunning)
		run.ID = "missing-terminal-lease"
		fixture.store.mu.Lock()
		fixture.store.runs[run.ID] = run
		fixture.store.mu.Unlock()
		process := newOrchestratorTestProcess(88)
		process.Finish(0)
		execution := &runExecution{
			orchestrator: fixture.orchestrator, run: run, process: process, adapter: fixture.adapter,
			workspace: workspace.RunWorkspace{RunID: run.ID, Path: core.PathJoin(t.TempDir(), "missing-worktree")},
		}
		fixture.orchestrator.finishExecution(execution, core.Ok(0))
		failed := fixture.store.Run(run.ID)
		core.AssertTrue(t, failed.OK, failed.Error())
		core.AssertEqual(t, work.RunFailed, failed.Value.(work.Run).Status)
		core.AssertContains(t, failed.Value.(work.Run).FailureReason, "lease")
	})
}

func TestRunPublicMutationFailureBoundaries(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	missingSource := work.DispatchRequest{
		Work:     work.Item{ID: "missing", Repository: core.PathJoin(t.TempDir(), "does-not-exist")},
		Provider: "fake", ConfirmedSourceRevision: "missing",
	}
	core.AssertFalse(t, fixture.orchestrator.ReviewDispatch(context.Background(), missingSource).OK)

	item, _, revision := fixture.registerRepository()
	review := fixture.reviewDispatch(item, revision)
	fixture.ids.mu.Lock()
	fixture.ids.failNext = 1
	fixture.ids.mu.Unlock()
	core.AssertFalse(t, fixture.orchestrator.Dispatch(context.Background(), review).OK)

	run := fixture.queueDispatch(item, revision)
	fixture.store.failNext(func(commit Commit) bool {
		return commit.Run != nil && commit.Run.ID == run.ID && commit.Run.Status == work.RunCancelled
	})
	core.AssertFalse(t, fixture.orchestrator.Cancel(context.Background(), run.ID).OK)
	at := fixture.clock.Now()
	fixture.clock.Set(time.Time{})
	core.AssertFalse(t, fixture.orchestrator.StopQueue(context.Background()).OK)
	fixture.clock.Set(at)
	fixture.store.failNext(func(commit Commit) bool { return commit.Queue != nil })
	core.AssertFalse(t, fixture.orchestrator.StopQueue(context.Background()).OK)

	core.AssertTrue(t, fixture.orchestrator.Close().OK)
	core.AssertFalse(t, fixture.orchestrator.Dispatch(context.Background(), review).OK)
	core.AssertFalse(t, fixture.orchestrator.Cancel(context.Background(), run.ID).OK)
}

func TestRunAdmissionEventAndProviderFailureBoundaries(t *testing.T) {
	t.Run("provider registration disappears", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		fixture.store.mu.Lock()
		stored := fixture.store.runs[run.ID]
		stored.Provider = "missing"
		fixture.store.runs[run.ID] = stored
		fixture.store.mu.Unlock()
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		failed := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
		core.AssertContains(t, failed.FailureReason, "missing")
	})

	t.Run("preparing event identity is unavailable", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		fixture.ids.mu.Lock()
		fixture.ids.empty = true
		fixture.ids.mu.Unlock()
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		time.Sleep(100 * time.Millisecond)
		stored := fixture.store.Run(run.ID)
		core.AssertTrue(t, stored.OK, stored.Error())
		core.AssertEqual(t, work.RunQueued, stored.Value.(work.Run).Status)
		fixture.ids.mu.Lock()
		fixture.ids.empty = false
		fixture.ids.mu.Unlock()
		core.AssertTrue(t, fixture.orchestrator.Cancel(context.Background(), run.ID).OK)
	})

	t.Run("preparing transition cannot be persisted", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		fixture.store.failNext(func(commit Commit) bool {
			return commit.Run != nil && commit.Run.Status == work.RunPreparing
		})
		fixture.orchestrator.startQueuedRun(run)
		stored := fixture.store.Run(run.ID)
		core.AssertTrue(t, stored.OK, stored.Error())
		core.AssertEqual(t, work.RunQueued, stored.Value.(work.Run).Status)
		core.AssertTrue(t, fixture.orchestrator.Cancel(context.Background(), run.ID).OK)
	})

	t.Run("running event identity fails after workspace preparation", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		fixture.adapter.mu.Lock()
		fixture.adapter.afterBuild = func() {
			fixture.ids.mu.Lock()
			fixture.ids.failNext = 1
			fixture.ids.mu.Unlock()
			fixture.server.mu.Lock()
			fixture.server.failEnsure = true
			fixture.server.mu.Unlock()
		}
		fixture.adapter.mu.Unlock()
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		failed := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
		core.AssertContains(t, failed.FailureReason, "ID source")
		core.AssertContains(t, failed.FailureReason, "private repository unavailable")
		fixture.server.mu.Lock()
		fixture.server.failEnsure = false
		fixture.server.mu.Unlock()
	})

	t.Run("durable queue snapshot fails after workspace preparation", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		fixture.adapter.mu.Lock()
		fixture.adapter.afterBuild = func() { fixture.store.setSnapshotFailure(true) }
		fixture.adapter.mu.Unlock()
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		failed := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
		core.AssertContains(t, failed.FailureReason, "snapshot failure")
		fixture.store.setSnapshotFailure(false)
	})

	t.Run("durable queue snapshot has wrong shape after workspace preparation", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		fixture.adapter.mu.Lock()
		fixture.adapter.afterBuild = func() { fixture.store.overrideSnapshot("not a snapshot") }
		fixture.adapter.mu.Unlock()
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		failed := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
		core.AssertContains(t, failed.FailureReason, "instead of snapshot")
		fixture.store.clearSnapshotOverride()
	})

	t.Run("clock fails after provider command build", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		fixture.adapter.mu.Lock()
		fixture.adapter.afterBuild = func() { fixture.clock.ZeroAfter(1) }
		fixture.adapter.mu.Unlock()
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		failed := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
		core.AssertContains(t, failed.FailureReason, "clock")
		core.AssertEqual(t, 0, fixture.launcher.Count())
	})

	t.Run("clock fails after native process start", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		fixture.launcher.mu.Lock()
		fixture.launcher.beforeStart = func(provider.Command) { fixture.clock.ZeroAfter(1) }
		fixture.launcher.mu.Unlock()
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		launch := fixture.launcher.WaitStart(t)
		failed := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
		core.AssertContains(t, failed.FailureReason, "clock")
		core.AssertTrue(t, launch.process.shutdown)
	})

	t.Run("process event identity fails after native start", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		fixture.launcher.mu.Lock()
		fixture.launcher.beforeStart = func(provider.Command) {
			fixture.ids.mu.Lock()
			fixture.ids.failNext = 1
			fixture.ids.mu.Unlock()
		}
		fixture.launcher.mu.Unlock()
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		launch := fixture.launcher.WaitStart(t)
		failed := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
		core.AssertContains(t, failed.FailureReason, "ID source")
		core.AssertTrue(t, launch.process.shutdown)
	})
}

func TestRunProviderOutputFailureBoundaries(t *testing.T) {
	newExecution := func(process *orchestratorTestProcess) *runExecution {
		run := storeTestRun(work.RunRunning)
		run.Provider = "fake"
		return &runExecution{run: run, process: process}
	}

	fixture := newOrchestratorFixture(t)
	process := newOrchestratorTestProcess(101)
	execution := newExecution(process)
	fixture.clock.Set(time.Time{})
	fixture.orchestrator.consumeLine(execution, rawLine{stream: "stdout", text: "line"})
	core.AssertContains(t, execution.failure, "clock")
	core.AssertTrue(t, process.shutdown)
	fixture.clock.Set(fixture.at)

	process = newOrchestratorTestProcess(102)
	execution = newExecution(process)
	fixture.ids.mu.Lock()
	fixture.ids.failNext = 1
	fixture.ids.mu.Unlock()
	fixture.orchestrator.persistProviderOutput(execution, provider.Output{Kind: "progress", Text: "halfway"})
	core.AssertContains(t, execution.failure, "ID source")
	core.AssertTrue(t, process.shutdown)

	process = newOrchestratorTestProcess(103)
	execution = newExecution(process)
	fixture.clock.ZeroAfter(2)
	fixture.orchestrator.persistProviderOutput(execution, provider.Output{
		Kind: "rate_limit", Text: "quota", DetailJSON: `{}`, RetryAfter: "1h",
	})
	core.AssertContains(t, execution.failure, "clock")
	core.AssertTrue(t, process.shutdown)

	process = newOrchestratorTestProcess(104)
	execution = newExecution(process)
	fixture.orchestrator.persistProviderOutput(execution, provider.Output{
		Kind: "rate_limit", RetryAfter: "1h",
	})
	core.AssertContains(t, execution.failure, "backoff requires")
	core.AssertTrue(t, process.shutdown)

	process = newOrchestratorTestProcess(105)
	execution = newExecution(process)
	fixture.orchestrator.persistProviderOutput(execution, provider.Output{
		Kind: "progress", Text: "halfway", DetailJSON: `{"step":1}`,
	})
	fixture.store.mu.Lock()
	event := fixture.store.events[len(fixture.store.events)-1]
	fixture.store.mu.Unlock()
	core.AssertEqual(t, "", event.Detail)
	core.AssertEqual(t, `{"step":1}`, event.DetailJSON)

	process = newOrchestratorTestProcess(106)
	execution = newExecution(process)
	fixture.store.setSnapshotFailure(true)
	fixture.orchestrator.persistProviderOutput(execution, provider.Output{
		Kind: "rate_limit", Text: "quota", RetryAfter: "1h",
	})
	core.AssertContains(t, execution.failure, "snapshot failure")
	core.AssertTrue(t, process.shutdown)
	fixture.store.setSnapshotFailure(false)

	process = newOrchestratorTestProcess(107)
	execution = newExecution(process)
	fixture.store.overrideSnapshot("not a snapshot")
	fixture.orchestrator.persistProviderOutput(execution, provider.Output{
		Kind: "rate_limit", Text: "quota", RetryAfter: "1h",
	})
	core.AssertContains(t, execution.failure, "instead of snapshot")
	core.AssertTrue(t, process.shutdown)
	fixture.store.clearSnapshotOverride()
	core.AssertFalse(t, sameStrings([]string{"left"}, []string{"right"}))
}

func TestRunMalformedAndNonZeroClassification(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	malformed := fixture.queueDispatch(item, revision)
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	first := fixture.launcher.WaitStart(t)
	first.callback("stdout", "MALFORMED")
	first.process.Finish(0)
	orchestratorWaitRunStatus(t, fixture.store, malformed.ID, work.RunCompleted)
	foundUnclassified := false
	for _, event := range fixture.store.events {
		if event.RunID == malformed.ID && event.Kind == "unclassified_provider_finish" {
			foundUnclassified = true
		}
	}
	core.AssertTrue(t, foundUnclassified)

	nonzero := fixture.queueDispatch(item, revision)
	second := fixture.launcher.WaitStart(t)
	second.callback("stdout", orchestratorCompletedEnvelope("looks done"))
	second.process.Finish(9)
	failed := orchestratorWaitRunStatus(t, fixture.store, nonzero.ID, work.RunFailed)
	core.AssertEqual(t, 9, failed.ExitCode)
}

func TestRunWaitingQuestionRequiresValidEnvelope(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	run := fixture.queueDispatch(item, revision)
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	launch := fixture.launcher.WaitStart(t)
	launch.callback("stdout", `<<<LEM_STATUS>>>{"status":"waiting","question":"Which API?"}<<<END_LEM_STATUS>>>`)
	launch.process.Finish(0)
	orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunWaiting)
	core.AssertEqual(t, 1, len(fixture.store.questions))
	core.AssertEqual(t, "Which API?", fixture.store.questions[0].Text)
}

func TestRunPersistenceFailureStopsBeforeOrDuringLaunch(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	review := fixture.reviewDispatch(item, revision)
	fixture.store.failNext(func(commit Commit) bool { return commit.CreateRun })
	core.AssertFalse(t, fixture.orchestrator.Dispatch(context.Background(), review).OK)
	core.AssertEqual(t, 0, len(fixture.store.runs))
	core.AssertEqual(t, 0, fixture.launcher.Count())

	first := fixture.queueDispatch(item, revision)
	fixture.store.failNext(func(commit Commit) bool {
		return commit.Run != nil && commit.Run.Status == work.RunRunning
	})
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	orchestratorWaitRunStatus(t, fixture.store, first.ID, work.RunFailed)
	core.AssertEqual(t, 0, fixture.launcher.Count())

	fixture.store.failNext(func(commit Commit) bool { return len(commit.Logs) > 0 })
	second := fixture.queueDispatch(item, revision)
	fixture.orchestrator.wakeQueue()
	launch := fixture.launcher.WaitStart(t)
	launch.callback("stdout", "a line long enough to flush immediately")
	failed := orchestratorWaitRunStatus(t, fixture.store, second.ID, work.RunFailed)
	core.AssertContains(t, failed.FailureReason, "injected commit failure")
	core.AssertTrue(t, launch.process.shutdown)
}

func TestRunTerminalPersistenceFailureRetainsOwnership(t *testing.T) {
	t.Run("transient terminal failure retries atomically", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		fixture.store.failNext(func(commit Commit) bool {
			return commit.Run != nil && commit.Run.Status == work.RunCompleted
		})
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		launch := fixture.launcher.WaitStart(t)
		launch.callback("stdout", orchestratorCompletedEnvelope("done"))
		launch.process.Finish(0)
		orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunCompleted)
		deadline := time.Now().Add(5 * time.Second)
		ownsRun := true
		for ownsRun && time.Now().Before(deadline) {
			fixture.orchestrator.mu.Lock()
			_, ownsRun = fixture.orchestrator.runs[run.ID]
			fixture.orchestrator.mu.Unlock()
			if ownsRun {
				time.Sleep(10 * time.Millisecond)
			}
		}
		core.AssertFalse(t, ownsRun)
	})

	t.Run("live execution", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		terminalAttempted := make(chan struct{}, 1)
		fixture.store.mu.Lock()
		fixture.store.beforeCommit = func(commit Commit) {
			if commit.Run != nil && commit.Run.Status == work.RunCompleted {
				terminalAttempted <- struct{}{}
			}
		}
		fixture.store.mu.Unlock()
		fixture.store.mu.Lock()
		fixture.store.failCommit = func(commit Commit) bool {
			return commit.Run != nil && commit.Run.Status == work.RunCompleted
		}
		fixture.store.mu.Unlock()
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		launch := fixture.launcher.WaitStart(t)
		launch.callback("stdout", orchestratorCompletedEnvelope("done"))
		launch.process.Finish(0)
		select {
		case <-terminalAttempted:
		case <-time.After(5 * time.Second):
			t.Fatal("terminal commit was not attempted")
		}
		time.Sleep(50 * time.Millisecond)
		stored := fixture.store.Run(run.ID)
		core.AssertTrue(t, stored.OK, stored.Error())
		core.AssertEqual(t, work.RunRunning, stored.Value.(work.Run).Status)
		fixture.orchestrator.mu.Lock()
		_, ownsRun := fixture.orchestrator.runs[run.ID]
		_, ownsExecution := fixture.orchestrator.executions[run.ID]
		fixture.orchestrator.mu.Unlock()
		core.AssertTrue(t, ownsRun)
		core.AssertTrue(t, ownsExecution)
		fixture.store.mu.Lock()
		fixture.store.failCommit = nil
		fixture.store.mu.Unlock()
		orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunCompleted)
		deadline := time.Now().Add(5 * time.Second)
		for ownsRun && time.Now().Before(deadline) {
			fixture.orchestrator.mu.Lock()
			_, ownsRun = fixture.orchestrator.runs[run.ID]
			fixture.orchestrator.mu.Unlock()
			if ownsRun {
				time.Sleep(10 * time.Millisecond)
			}
		}
		core.AssertFalse(t, ownsRun)
		core.AssertTrue(t, fixture.orchestrator.Close().OK)
	})

	t.Run("pre-process failure", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		fixture.launcher.mu.Lock()
		fixture.launcher.failStart = true
		fixture.launcher.mu.Unlock()
		terminalAttempted := make(chan struct{}, 1)
		fixture.store.mu.Lock()
		fixture.store.beforeCommit = func(commit Commit) {
			if commit.Run != nil && commit.Run.Status == work.RunFailed {
				terminalAttempted <- struct{}{}
			}
		}
		fixture.store.mu.Unlock()
		fixture.store.mu.Lock()
		fixture.store.failCommit = func(commit Commit) bool {
			return commit.Run != nil && commit.Run.Status == work.RunFailed
		}
		fixture.store.mu.Unlock()
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		select {
		case <-terminalAttempted:
		case <-time.After(5 * time.Second):
			t.Fatal("failure commit was not attempted")
		}
		time.Sleep(50 * time.Millisecond)
		stored := fixture.store.Run(run.ID)
		core.AssertTrue(t, stored.OK, stored.Error())
		core.AssertEqual(t, work.RunRunning, stored.Value.(work.Run).Status)
		fixture.orchestrator.mu.Lock()
		_, ownsReview := fixture.orchestrator.pending[run.ID]
		fixture.orchestrator.mu.Unlock()
		core.AssertTrue(t, ownsReview)
		fixture.store.mu.Lock()
		fixture.store.failCommit = nil
		fixture.store.mu.Unlock()
		orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
		deadline := time.Now().Add(5 * time.Second)
		for ownsReview && time.Now().Before(deadline) {
			fixture.orchestrator.mu.Lock()
			_, ownsReview = fixture.orchestrator.pending[run.ID]
			fixture.orchestrator.mu.Unlock()
			if ownsReview {
				time.Sleep(10 * time.Millisecond)
			}
		}
		core.AssertFalse(t, ownsReview)
		core.AssertTrue(t, fixture.orchestrator.Close().OK)
	})

	t.Run("shutdown interrupts persistent retry", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		terminalAttempted := make(chan struct{}, 1)
		fixture.store.mu.Lock()
		fixture.store.beforeCommit = func(commit Commit) {
			if commit.Run != nil && commit.Run.Status == work.RunCompleted {
				select {
				case terminalAttempted <- struct{}{}:
				default:
				}
			}
		}
		fixture.store.failCommit = func(commit Commit) bool {
			return commit.Run != nil && commit.Run.Status == work.RunCompleted
		}
		fixture.store.mu.Unlock()
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		launch := fixture.launcher.WaitStart(t)
		launch.callback("stdout", orchestratorCompletedEnvelope("done"))
		launch.process.Finish(0)
		select {
		case <-terminalAttempted:
		case <-time.After(5 * time.Second):
			t.Fatal("terminal retry was not attempted")
		}
		core.AssertTrue(t, fixture.orchestrator.Close().OK)
		recovered := fixture.store.Run(run.ID)
		core.AssertTrue(t, recovered.OK, recovered.Error())
		core.AssertEqual(t, work.RunInterrupted, recovered.Value.(work.Run).Status)
	})
}

func TestRunRetainedWorkspaceDiagnosticFailureDoesNotLeakOwnership(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	run := fixture.queueDispatch(item, revision)
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	launch := fixture.launcher.WaitStart(t)
	fixture.git.setAfterPush(func(directory string) {
		result := core.WriteFile(core.PathJoin(directory, "changed-after-capture.txt"), []byte("retain\n"), 0o600)
		core.AssertTrue(t, result.OK, result.Error())
	})
	fixture.store.mu.Lock()
	fixture.store.failCommit = func(commit Commit) bool {
		return commit.Event != nil && commit.Event.Kind == "workspace_retained"
	}
	fixture.store.mu.Unlock()
	launch.callback("stdout", orchestratorCompletedEnvelope("done"))
	launch.process.Finish(0)
	completed := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunCompleted)
	core.AssertTrue(t, core.Stat(core.PathJoin(completed.Worktree, "changed-after-capture.txt")).OK)
	deadline := time.Now().Add(5 * time.Second)
	ownsRun := true
	for ownsRun && time.Now().Before(deadline) {
		fixture.orchestrator.mu.Lock()
		_, ownsRun = fixture.orchestrator.runs[run.ID]
		fixture.orchestrator.mu.Unlock()
		if ownsRun {
			time.Sleep(10 * time.Millisecond)
		}
	}
	fixture.store.mu.Lock()
	fixture.store.failCommit = nil
	fixture.store.mu.Unlock()
	core.AssertFalse(t, ownsRun)
}

func TestRunTerminalMetadataFailureRetainsUntilCloseRecovery(t *testing.T) {
	tests := []struct {
		name      string
		configure func(*orchestratorFixture)
		consumed  func(*orchestratorFixture) bool
	}{
		{
			name: "clock",
			configure: func(fixture *orchestratorFixture) {
				fixture.clock.ZeroAfter(1)
			},
			consumed: func(fixture *orchestratorFixture) bool {
				fixture.clock.mu.Lock()
				defer fixture.clock.mu.Unlock()
				return fixture.clock.zeroAfter == 0
			},
		},
		{
			name: "event identity",
			configure: func(fixture *orchestratorFixture) {
				fixture.ids.mu.Lock()
				fixture.ids.failNext = 1
				fixture.ids.mu.Unlock()
			},
			consumed: func(fixture *orchestratorFixture) bool {
				fixture.ids.mu.Lock()
				defer fixture.ids.mu.Unlock()
				return fixture.ids.failNext == 0
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fixture := newOrchestratorFixture(t)
			item, _, revision := fixture.registerRepository()
			run := fixture.queueDispatch(item, revision)
			core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
			launch := fixture.launcher.WaitStart(t)
			launch.callback("stdout", orchestratorCompletedEnvelope("done"))
			deadline := time.Now().Add(5 * time.Second)
			for {
				fixture.store.mu.Lock()
				logged := len(fixture.store.logs) > 0
				fixture.store.mu.Unlock()
				if logged {
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("provider output was not consumed")
				}
				time.Sleep(10 * time.Millisecond)
			}
			test.configure(fixture)
			launch.process.Finish(0)
			consumedDeadline := time.Now().Add(5 * time.Second)
			for !test.consumed(fixture) {
				if time.Now().After(consumedDeadline) {
					t.Fatalf("%s one-shot terminal metadata failure was not consumed", test.name)
				}
				time.Sleep(10 * time.Millisecond)
			}
			stored := fixture.store.Run(run.ID)
			core.AssertTrue(t, stored.OK, stored.Error())
			core.AssertEqual(t, work.RunRunning, stored.Value.(work.Run).Status)
			fixture.orchestrator.mu.Lock()
			_, retained := fixture.orchestrator.executions[run.ID]
			fixture.orchestrator.mu.Unlock()
			core.AssertTrue(t, retained)
			core.AssertTrue(t, fixture.orchestrator.Close().OK)
			recovered := fixture.store.Run(run.ID)
			core.AssertTrue(t, recovered.OK, recovered.Error())
			core.AssertEqual(t, work.RunInterrupted, recovered.Value.(work.Run).Status)
		})
	}
}

func TestRunQueueStartPersistenceRollback(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	run := fixture.queueDispatch(item, revision)
	acceptingCommitReached := make(chan struct{}, 1)
	releaseAcceptingCommit := make(chan struct{})
	preparingCommitReached := make(chan struct{}, 1)
	fixture.store.mu.Lock()
	fixture.store.beforeCommit = func(commit Commit) {
		if commit.Queue != nil && commit.Queue.Status == work.QueueAccepting {
			acceptingCommitReached <- struct{}{}
			<-releaseAcceptingCommit
		}
		if commit.Run != nil && commit.Run.Status == work.RunPreparing {
			preparingCommitReached <- struct{}{}
		}
	}
	fixture.store.mu.Unlock()
	fixture.store.failNext(func(commit Commit) bool {
		return commit.Queue != nil && commit.Queue.Status == work.QueueAccepting
	})
	started := make(chan core.Result, 1)
	go func() { started <- fixture.orchestrator.StartQueue(context.Background()) }()
	select {
	case <-acceptingCommitReached:
	case <-time.After(5 * time.Second):
		t.Fatal("queue start did not reach the blocked persistence boundary")
	}
	drained := make(chan time.Time, 1)
	go func() { drained <- fixture.orchestrator.drainQueue() }()
	prematureAdmission := false
	select {
	case <-preparingCommitReached:
		prematureAdmission = true
	case <-time.After(100 * time.Millisecond):
	}
	close(releaseAcceptingCommit)
	core.AssertFalse(t, (<-started).OK)
	select {
	case <-drained:
	case <-time.After(5 * time.Second):
		t.Fatal("queue drain did not return after persistence rollback")
	}
	fixture.store.mu.Lock()
	fixture.store.beforeCommit = nil
	fixture.store.mu.Unlock()
	core.AssertFalse(t, prematureAdmission)
	core.AssertEqual(t, 0, fixture.launcher.Count())
	core.AssertEqual(t, work.RunQueued, fixture.store.Run(run.ID).Value.(work.Run).Status)
	fixture.store.mu.Lock()
	queueStatus := fixture.store.queue.Status
	fixture.store.mu.Unlock()
	core.AssertEqual(t, work.QueueFrozen, queueStatus)
}

func TestRunQueueStopSerializesNewAdmission(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	frozenCommitReached := make(chan struct{}, 1)
	releaseFrozenCommit := make(chan struct{})
	fixture.store.mu.Lock()
	fixture.store.beforeCommit = func(commit Commit) {
		if commit.Queue != nil && commit.Queue.Status == work.QueueFrozen {
			frozenCommitReached <- struct{}{}
			<-releaseFrozenCommit
		}
	}
	fixture.store.mu.Unlock()
	stopped := make(chan core.Result, 1)
	go func() { stopped <- fixture.orchestrator.StopQueue(context.Background()) }()
	select {
	case <-frozenCommitReached:
	case <-time.After(5 * time.Second):
		t.Fatal("queue stop did not reach the blocked persistence boundary")
	}
	run := fixture.queueDispatch(item, revision)
	close(releaseFrozenCommit)
	stopResult := <-stopped
	core.AssertTrue(t, stopResult.OK, stopResult.Error())
	fixture.store.mu.Lock()
	fixture.store.beforeCommit = nil
	fixture.store.mu.Unlock()
	core.AssertTrue(t, fixture.orchestrator.drainQueue().IsZero())
	core.AssertEqual(t, 0, fixture.launcher.Count())
	core.AssertEqual(t, work.RunQueued, fixture.store.Run(run.ID).Value.(work.Run).Status)
}

func TestRunPreparingPersistenceFailurePreservesFIFO(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	first := fixture.queueDispatch(item, revision)
	secondItem := item
	secondItem.ID = "work-second"
	second := fixture.queueDispatch(secondItem, revision)
	time.Sleep(100 * time.Millisecond)
	started := fixture.queue.Start(fixture.at)
	core.AssertTrue(t, started.OK, started.Error())
	state := started.Value.(work.QueueState)
	core.AssertTrue(t, fixture.store.Commit(Commit{Queue: &state}).OK)
	attempts := 0
	fixture.store.mu.Lock()
	fixture.store.failCommit = func(commit Commit) bool {
		if commit.Run != nil && commit.Run.Status == work.RunPreparing {
			attempts++
			return true
		}
		return false
	}
	fixture.store.mu.Unlock()
	next := fixture.orchestrator.drainQueue()
	core.AssertEqual(t, 1, attempts)
	core.AssertFalse(t, next.IsZero())
	core.AssertEqual(t, 0, fixture.launcher.Count())
	core.AssertEqual(t, work.RunQueued, fixture.store.Run(first.ID).Value.(work.Run).Status)
	core.AssertEqual(t, work.RunQueued, fixture.store.Run(second.ID).Value.(work.Run).Status)
	fixture.store.mu.Lock()
	fixture.store.failCommit = nil
	fixture.store.mu.Unlock()
}

func TestRunQueueMutationFailuresRestoreControllerState(t *testing.T) {
	t.Run("stop persistence", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		fixture.store.failNext(func(commit Commit) bool {
			return commit.Queue != nil && commit.Queue.Status == work.QueueFrozen
		})
		core.AssertFalse(t, fixture.orchestrator.StopQueue(context.Background()).OK)
		decision := fixture.queue.Decide(queue.Candidate{
			RunID: "probe", Provider: "fake", QueuedAt: fixture.at,
		}, queue.Runtime{Now: fixture.at})
		core.AssertTrue(t, decision.OK, decision.Error())
		core.AssertTrue(t, decision.Value.(queue.Decision).Allowed)
	})

	t.Run("provider start persistence", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		fixture.store.failNext(func(commit Commit) bool {
			return commit.Run != nil && commit.Run.Status == work.RunRunning
		})
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
		state := fixture.queue.RecordStart("fake", "probe", fixture.at.Add(time.Minute))
		core.AssertTrue(t, state.OK, state.Error())
		core.AssertEqual(t, 1, state.Value.(work.ProviderState).WindowAdmissions)
	})

	t.Run("backoff persistence", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		item, _, revision := fixture.registerRepository()
		run := fixture.queueDispatch(item, revision)
		core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
		launch := fixture.launcher.WaitStart(t)
		fixture.store.failNext(func(commit Commit) bool {
			return commit.Event != nil && commit.Event.Kind == "rate_limit"
		})
		launch.callback("stderr", "RATE:quota")
		orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
		decision := fixture.queue.Decide(queue.Candidate{
			RunID: "probe", Provider: "fake", QueuedAt: fixture.at,
		}, queue.Runtime{Now: fixture.at})
		core.AssertTrue(t, decision.OK, decision.Error())
		core.AssertTrue(t, decision.Value.(queue.Decision).Allowed)
	})
}

func TestRunCancelBetweenDurabilityAndProcessRegistration(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	run := fixture.queueDispatch(item, revision)
	startReached := make(chan struct{}, 1)
	releaseStart := make(chan struct{})
	fixture.launcher.beforeStart = func(provider.Command) {
		startReached <- struct{}{}
		<-releaseStart
	}
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	select {
	case <-startReached:
	case <-time.After(5 * time.Second):
		t.Fatal("launcher boundary was not reached")
	}
	cancelled := fixture.orchestrator.Cancel(context.Background(), run.ID)
	core.AssertTrue(t, cancelled.OK, cancelled.Error())
	core.AssertEqual(t, work.RunCancelling, cancelled.Value.(work.Run).Status)
	close(releaseStart)
	launch := fixture.launcher.WaitStart(t)
	finished := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunCancelled)
	core.AssertEqual(t, -1, finished.ExitCode)
	core.AssertTrue(t, launch.process.shutdown)
}

func TestRunCancellingStartupShutdownFailureIsDurableAndRaceFree(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	run := fixture.queueDispatch(item, revision)
	startReached := make(chan struct{}, 1)
	releaseStart := make(chan struct{})
	fixture.launcher.mu.Lock()
	fixture.launcher.beforeStart = func(provider.Command) {
		startReached <- struct{}{}
		<-releaseStart
	}
	fixture.launcher.configure = func(process *orchestratorTestProcess) {
		process.shutdownFail = true
	}
	fixture.launcher.mu.Unlock()
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	select {
	case <-startReached:
	case <-time.After(5 * time.Second):
		t.Fatal("launcher boundary was not reached")
	}
	cancelled := fixture.orchestrator.Cancel(context.Background(), run.ID)
	core.AssertTrue(t, cancelled.OK, cancelled.Error())
	close(releaseStart)
	launch := fixture.launcher.WaitStart(t)
	deadline := time.Now().Add(5 * time.Second)
	for {
		launch.process.mu.Lock()
		failedShutdown := launch.process.shutdownFail
		launch.process.mu.Unlock()
		fixture.orchestrator.mu.Lock()
		execution := fixture.orchestrator.executions[run.ID]
		fixture.orchestrator.mu.Unlock()
		if execution != nil && execution.failure != "" && failedShutdown {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("cancelling launch did not record shutdown failure")
		}
		time.Sleep(10 * time.Millisecond)
	}
	launch.process.mu.Lock()
	launch.process.shutdownFail = false
	launch.process.mu.Unlock()
	launch.process.Finish(-1)
	finished := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunCancelled)
	core.AssertContains(t, finished.FailureReason, "shutdown failure")
}

func TestRunCancelRacingTerminalCommitFinalizesCancelled(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	run := fixture.queueDispatch(item, revision)
	terminalReached := make(chan struct{}, 1)
	releaseTerminal := make(chan struct{})
	fixture.store.mu.Lock()
	fixture.store.beforeCommit = func(commit Commit) {
		if commit.Run == nil || commit.Run.Status != work.RunCompleted {
			return
		}
		select {
		case terminalReached <- struct{}{}:
			<-releaseTerminal
		default:
		}
	}
	fixture.store.mu.Unlock()
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	launch := fixture.launcher.WaitStart(t)
	launch.callback("stdout", orchestratorCompletedEnvelope("done"))
	launch.process.Finish(0)
	select {
	case <-terminalReached:
	case <-time.After(5 * time.Second):
		t.Fatal("terminal commit boundary was not reached")
	}
	cancelled := fixture.orchestrator.Cancel(context.Background(), run.ID)
	core.AssertTrue(t, cancelled.OK, cancelled.Error())
	core.AssertEqual(t, work.RunCancelling, cancelled.Value.(work.Run).Status)
	close(releaseTerminal)
	finished := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunCancelled)
	core.AssertEqual(t, 0, finished.ExitCode)
	core.AssertTrue(t, launch.process.shutdown)
	deadline := time.Now().Add(5 * time.Second)
	ownsRun := true
	for ownsRun && time.Now().Before(deadline) {
		fixture.orchestrator.mu.Lock()
		_, ownsRun = fixture.orchestrator.runs[run.ID]
		fixture.orchestrator.mu.Unlock()
		if ownsRun {
			time.Sleep(10 * time.Millisecond)
		}
	}
	core.AssertFalse(t, ownsRun)
}

func TestRunTerminalCommitReconcilesDurableState(t *testing.T) {
	fixture := newOrchestratorFixture(t)

	t.Run("rejects incomplete commit", func(t *testing.T) {
		core.AssertFalse(t, fixture.orchestrator.commitTerminal(Commit{}).OK)
	})

	target := storeTestRun(work.RunFailed)
	expected := work.RunRunning
	current := target
	current.Status = work.RunCompleted
	fixture.store.mu.Lock()
	fixture.store.runs[current.ID] = current
	fixture.store.mu.Unlock()

	t.Run("accepts an already durable terminal outcome", func(t *testing.T) {
		result := fixture.orchestrator.commitTerminal(Commit{Run: &target, ExpectedStatus: &expected})
		core.AssertTrue(t, result.OK, result.Error())
	})

	t.Run("rejects an invalid durable run value", func(t *testing.T) {
		fixture.store.overrideRun("not a run")
		result := fixture.orchestrator.commitTerminal(Commit{Run: &target, ExpectedStatus: &expected})
		fixture.store.clearRunOverride()
		core.AssertFalse(t, result.OK)
		core.AssertContains(t, result.Error(), "instead of run")
	})

	t.Run("does not overwrite an unrelated nonterminal state", func(t *testing.T) {
		current.Status = work.RunQueued
		fixture.store.mu.Lock()
		fixture.store.runs[current.ID] = current
		fixture.store.mu.Unlock()
		result := fixture.orchestrator.commitTerminal(Commit{Run: &target, ExpectedStatus: &expected})
		core.AssertFalse(t, result.OK)
		core.AssertContains(t, result.Error(), "stale expected status")
	})
}

func TestRunConcurrencyAndDrainOrder(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	first := fixture.queueDispatch(item, revision)
	secondItem := item
	secondItem.ID = "work-2"
	secondItem.Title = "Second"
	second := fixture.queueDispatch(secondItem, revision)
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	firstLaunch := fixture.launcher.WaitStart(t)
	time.Sleep(100 * time.Millisecond)
	core.AssertEqual(t, 1, fixture.launcher.Count())
	firstLaunch.callback("stdout", orchestratorCompletedEnvelope("first"))
	firstLaunch.process.Finish(0)
	orchestratorWaitRunStatus(t, fixture.store, first.ID, work.RunCompleted)
	secondLaunch := fixture.launcher.WaitStart(t)
	core.AssertEqual(t, 2, fixture.launcher.Count())
	secondLaunch.callback("stdout", orchestratorCompletedEnvelope("second"))
	secondLaunch.process.Finish(0)
	orchestratorWaitRunStatus(t, fixture.store, second.ID, work.RunCompleted)
}

func TestRunCaptureFailureRetainsWorktree(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	run := fixture.queueDispatch(item, revision)
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	launch := fixture.launcher.WaitStart(t)
	running := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunRunning)
	core.AssertTrue(t, core.WriteFile(core.PathJoin(running.Worktree, "agent.txt"), []byte("agent work\n"), 0o600).OK)
	fixture.server.mu.Lock()
	fixture.server.failEnsure = true
	fixture.server.mu.Unlock()
	launch.callback("stdout", orchestratorCompletedEnvelope("done"))
	launch.process.Finish(0)
	failed := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunFailed)
	core.AssertContains(t, failed.FailureReason, "private repository unavailable")
	core.AssertTrue(t, core.Stat(running.Worktree).OK)
}

func TestRunCloseInterruptsAndJoins(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	run := fixture.queueDispatch(item, revision)
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	launch := fixture.launcher.WaitStart(t)
	core.AssertTrue(t, fixture.orchestrator.Close().OK)
	finished := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunInterrupted)
	core.AssertEqual(t, -1, finished.ExitCode)
	core.AssertTrue(t, launch.process.shutdown)
	fixture.orchestrator.mu.Lock()
	core.AssertEqual(t, 0, len(fixture.orchestrator.runs))
	core.AssertEqual(t, 0, len(fixture.orchestrator.executions))
	fixture.orchestrator.mu.Unlock()
}
