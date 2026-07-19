// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/orchestrator"
	"dappco.re/go/inference/agent/work"
	"dappco.re/go/inference/agent/workspace"
)

func TestAgentStore_Good(t *testing.T) {
	_, repository, agentStore := openTestAgentStore(t)
	defer closeTestDuckRepository(t, repository)
	at := time.Date(2026, time.July, 18, 9, 0, 0, 0, time.UTC)
	project := testAgentProject(at)
	run := testAgentRun("run-1", work.RunQueued, at)
	event := work.Event{ID: "event-queued", RunID: run.ID, WorkID: run.WorkID, Kind: "queued", Title: "queued", Detail: "implement task 11", CreatedAt: at}
	logs := []work.LogChunk{
		{RunID: run.ID, Sequence: 1, Stream: "stdout", Text: "first", CreatedAt: at},
		{RunID: run.ID, Sequence: 2, Stream: "stderr", Text: "second", CreatedAt: at.Add(time.Second)},
	}
	queue := work.QueueState{ID: "default", Status: work.QueueAccepting, Reason: "ready", UpdatedAt: at}
	provider := work.ProviderState{Provider: "codex", LastRunID: run.ID, WindowStartedAt: at, WindowAdmissions: 1, UpdatedAt: at}
	if result := agentStore.Commit(orchestrator.Commit{Project: &project, Run: &run, CreateRun: true, Event: &event, Logs: logs, Queue: &queue, Provider: &provider}); !result.OK {
		t.Fatalf("create durable run: %v", result.Value)
	}

	if got := requireAgentValue[work.Project](t, "Project", agentStore.Project(project.ID)); got != project {
		t.Fatalf("Project = %#v, want %#v", got, project)
	}
	if got := requireAgentValue[work.Project](t, "ProjectBySource", agentStore.ProjectBySource(project.SourcePath)); got != project {
		t.Fatalf("ProjectBySource = %#v, want %#v", got, project)
	}
	gotRun := requireAgentValue[work.Run](t, "Run", agentStore.Run(run.ID))
	if gotRun.DurableRevision != run.DurableRevision {
		t.Fatalf("Run durable revision = %q, want %q", gotRun.DurableRevision, run.DurableRevision)
	}
	if next := requireAgentValue[int](t, "NextRunNumber", agentStore.NextRunNumber(run.WorkID)); next != 2 {
		t.Fatalf("NextRunNumber = %d, want 2", next)
	}

	question := work.Question{ID: "question-1", RunID: run.ID, Text: "continue?", CreatedAt: at.Add(2 * time.Second)}
	answer := work.Answer{ID: "answer-1", QuestionID: question.ID, ResumeRunID: run.ID, Text: "yes", CreatedAt: at.Add(3 * time.Second)}
	if result := agentStore.Commit(orchestrator.Commit{Question: &question, Answer: &answer}); !result.OK {
		t.Fatalf("commit continuation records: %v", result.Value)
	}
	continuation := requireAgentValue[work.Continuation](t, "Continuation", agentStore.Continuation(run.ID))
	if continuation.Run.DurableRevision != run.DurableRevision || continuation.Task != event.Detail || len(continuation.Logs) != 2 || continuation.Logs[0].Sequence != 1 || continuation.Question != question || continuation.Answer != answer {
		t.Fatalf("Continuation = %#v", continuation)
	}

	acceptanceTime := at.Add(4 * time.Second)
	reviews := []struct {
		id     string
		status string
		review workspace.ChangeReview
	}{
		{id: "acceptance-a", status: "prepared", review: testChangeReview(run, nil, true)},
		{id: "acceptance-b", status: "conflicted", review: testChangeReview(run, []string{"README.md"}, true)},
		{id: "acceptance-c", status: "validation_failed", review: testChangeReview(run, nil, false)},
	}
	for _, index := range []int{2, 0, 1} {
		fixture := reviews[index]
		encoded := core.JSONMarshalString(fixture.review)
		receipt := work.Acceptance{ID: fixture.id, WorkID: run.WorkID, RunID: run.ID, SourceBase: "source", AgentBase: "base", AgentTip: "tip", IntegrationBranch: "integration", IntegrationWorktree: "/tmp/integration", ResultRevision: "result", Status: fixture.status, ValidationJSON: encoded, CreatedAt: acceptanceTime, UpdatedAt: acceptanceTime}
		if result := agentStore.Commit(orchestrator.Commit{Acceptance: &receipt}); !result.OK {
			t.Fatalf("commit %s acceptance: %v", fixture.status, result.Value)
		}
	}
	snapshot := requireAgentValue[work.Snapshot](t, "Snapshot", agentStore.Snapshot(run.WorkID))
	if len(snapshot.Runs) != 1 || snapshot.Runs[0].DurableRevision != run.DurableRevision || len(snapshot.Acceptances) != 3 {
		t.Fatalf("Snapshot = %#v", snapshot)
	}
	for index, fixture := range reviews {
		if snapshot.Acceptances[index].ID != fixture.id {
			t.Fatalf("acceptance order[%d] = %q, want %q", index, snapshot.Acceptances[index].ID, fixture.id)
		}
		encoded := core.JSONMarshalString(fixture.review)
		if snapshot.Acceptances[index].ValidationJSON != encoded {
			t.Fatalf("%s ChangeReview JSON changed\ngot:  %s\nwant: %s", fixture.status, snapshot.Acceptances[index].ValidationJSON, encoded)
		}
		var decoded workspace.ChangeReview
		if result := core.JSONUnmarshalString(snapshot.Acceptances[index].ValidationJSON, &decoded); !result.OK {
			t.Fatalf("decode %s ChangeReview: %v", fixture.status, result.Value)
		}
		if decoded.WorkID != fixture.review.WorkID || decoded.RunID != fixture.review.RunID || decoded.Diff != fixture.review.Diff || len(decoded.Validation) != len(fixture.review.Validation) || len(decoded.Conflicts) != len(fixture.review.Conflicts) {
			t.Fatalf("decoded %s ChangeReview = %#v, want %#v", fixture.status, decoded, fixture.review)
		}
	}
	if snapshot.Queue != queue || len(snapshot.Providers) != 1 || snapshot.Providers[0] != provider {
		t.Fatalf("queue/provider snapshot = %#v / %#v", snapshot.Queue, snapshot.Providers)
	}
}

func TestAgentStoreSnapshotIncludesDurableAnswerResumeProjection(t *testing.T) {
	_, repository, agentStore := openTestAgentStore(t)
	defer closeTestDuckRepository(t, repository)
	at := time.Date(2026, time.July, 19, 9, 0, 0, 0, time.UTC)
	run := testAgentRun("waiting-run", work.RunWaiting, at)
	question := work.Question{ID: "question-reopen", RunID: run.ID, Text: "Which target?", CreatedAt: at.Add(time.Second)}
	answer := work.Answer{ID: "answer-reopen", QuestionID: question.ID, ResumeRunID: "resume-run", Text: "Target A", CreatedAt: at.Add(2 * time.Second)}
	if result := agentStore.Commit(orchestrator.Commit{Run: &run, CreateRun: true, Question: &question, Answer: &answer}); !result.OK {
		t.Fatalf("commit answered waiting run: %s", result.Error())
	}

	snapshot := requireAgentValue[work.Snapshot](t, "answered Snapshot", agentStore.Snapshot(run.WorkID))
	for _, event := range snapshot.Events {
		if event.RunID == run.ID && event.Kind == "answered" && strings.Contains(event.DetailJSON, answer.ID) && strings.Contains(event.DetailJSON, answer.ResumeRunID) {
			return
		}
	}
	t.Fatalf("snapshot events omit durable answer/resume projection: %#v", snapshot.Events)
}

func TestAgentStore_OrderedSnapshot(t *testing.T) {
	_, repository, agentStore := openTestAgentStore(t)
	defer closeTestDuckRepository(t, repository)
	at := time.Date(2026, time.July, 18, 9, 30, 0, 0, time.UTC)

	projectZ := testAgentProject(at)
	projectZ.ID = "project-z"
	projectZ.SourcePath = "/source-z"
	projectZ.RepositoryRoot = "/source-z"
	projectZ.RepositoryName = "source-z.git"
	projectZ.ClonePath = "/private/source-z.git"
	projectA := projectZ
	projectA.ID = "project-a"
	projectA.SourcePath = "/source-a"
	projectA.RepositoryRoot = "/source-a"
	projectA.RepositoryName = "source-a.git"
	projectA.ClonePath = "/private/source-a.git"

	runZ := testAgentRun("run-z", work.RunQueued, at)
	runZ.ProjectID = projectZ.ID
	runA := testAgentRun("run-a", work.RunQueued, at)
	runA.ProjectID = projectA.ID
	runA.Number = 2

	eventZ := work.Event{ID: "event-z", RunID: runZ.ID, WorkID: runZ.WorkID, Kind: "ordered", Title: "z", CreatedAt: at}
	eventA := work.Event{ID: "event-a", RunID: runA.ID, WorkID: runA.WorkID, Kind: "ordered", Title: "a", CreatedAt: at}
	logZ := work.LogChunk{RunID: runZ.ID, Sequence: 1, Stream: "stdout", Text: "z", CreatedAt: at}
	logA := work.LogChunk{RunID: runA.ID, Sequence: 1, Stream: "stdout", Text: "a", CreatedAt: at}
	questionZ := work.Question{ID: "question-z", RunID: runZ.ID, Text: "z", CreatedAt: at}
	questionA := work.Question{ID: "question-a", RunID: runA.ID, Text: "a", CreatedAt: at}
	acceptanceZ := work.Acceptance{ID: "acceptance-z", WorkID: runZ.WorkID, RunID: runZ.ID, Status: "prepared", ValidationJSON: "{}", CreatedAt: at, UpdatedAt: at}
	acceptanceA := work.Acceptance{ID: "acceptance-a", WorkID: runA.WorkID, RunID: runA.ID, Status: "prepared", ValidationJSON: "{}", CreatedAt: at, UpdatedAt: at}
	providerZ := work.ProviderState{Provider: "provider-z", WindowStartedAt: at, UpdatedAt: at}
	providerA := work.ProviderState{Provider: "provider-a", WindowStartedAt: at, UpdatedAt: at}

	for _, commit := range []orchestrator.Commit{
		{Project: &projectZ, Run: &runZ, CreateRun: true, Event: &eventZ, Logs: []work.LogChunk{logZ}, Question: &questionZ, Acceptance: &acceptanceZ, Provider: &providerZ},
		{Project: &projectA, Run: &runA, CreateRun: true, Event: &eventA, Logs: []work.LogChunk{logA}, Question: &questionA, Acceptance: &acceptanceA, Provider: &providerA},
	} {
		if result := agentStore.Commit(commit); !result.OK {
			t.Fatalf("commit adversarial snapshot fixture: %v", result.Value)
		}
	}

	snapshot := requireAgentValue[work.Snapshot](t, "ordered Snapshot", agentStore.Snapshot(runA.WorkID))
	if got := []string{snapshot.Projects[0].ID, snapshot.Projects[1].ID}; got[0] != "project-a" || got[1] != "project-z" {
		t.Fatalf("project order = %#v, want [project-a project-z]", got)
	}
	if got := []string{snapshot.Runs[0].ID, snapshot.Runs[1].ID}; got[0] != "run-a" || got[1] != "run-z" {
		t.Fatalf("run order = %#v, want [run-a run-z]", got)
	}
	if got := []string{snapshot.Events[0].ID, snapshot.Events[1].ID}; got[0] != "event-a" || got[1] != "event-z" {
		t.Fatalf("event order = %#v, want [event-a event-z]", got)
	}
	if got := []string{snapshot.Logs[0].RunID, snapshot.Logs[1].RunID}; got[0] != "run-a" || got[1] != "run-z" {
		t.Fatalf("log order = %#v, want [run-a run-z]", got)
	}
	if got := []string{snapshot.Questions[0].ID, snapshot.Questions[1].ID}; got[0] != "question-a" || got[1] != "question-z" {
		t.Fatalf("question order = %#v, want [question-a question-z]", got)
	}
	if got := []string{snapshot.Acceptances[0].ID, snapshot.Acceptances[1].ID}; got[0] != "acceptance-a" || got[1] != "acceptance-z" {
		t.Fatalf("acceptance order = %#v, want [acceptance-a acceptance-z]", got)
	}
	if got := []string{snapshot.Providers[0].Provider, snapshot.Providers[1].Provider}; got[0] != "provider-a" || got[1] != "provider-z" {
		t.Fatalf("provider order = %#v, want [provider-a provider-z]", got)
	}
}

func TestAgentStore_Bad(t *testing.T) {
	_, repository, agentStore := openTestAgentStore(t)
	defer closeTestDuckRepository(t, repository)
	at := time.Date(2026, time.July, 18, 10, 0, 0, 0, time.UTC)
	orphanExpected := work.RunQueued
	orphanQueue := work.QueueState{ID: "default", Status: work.QueueFrozen, Reason: "must roll back", UpdatedAt: at}
	if result := agentStore.Commit(orchestrator.Commit{ExpectedStatus: &orphanExpected, Queue: &orphanQueue}); result.OK {
		t.Fatal("commit accepted expected status without a run")
	}
	orphanSnapshot := requireAgentValue[work.Snapshot](t, "Snapshot after orphan expected status", agentStore.Snapshot(""))
	if orphanSnapshot.Queue.ID != "" {
		t.Fatalf("orphan expected-status commit persisted queue state: %#v", orphanSnapshot.Queue)
	}
	project := testAgentProject(at)
	run := testAgentRun("run-cas", work.RunQueued, at)
	if result := agentStore.Commit(orchestrator.Commit{Project: &project, Run: &run, CreateRun: true}); !result.OK {
		t.Fatalf("create run: %v", result.Value)
	}
	firstLog := work.LogChunk{RunID: run.ID, Sequence: 1, Stream: "stdout", Text: "first", CreatedAt: at}
	if result := agentStore.Commit(orchestrator.Commit{Logs: []work.LogChunk{firstLog}}); !result.OK {
		t.Fatalf("commit first log: %v", result.Value)
	}

	updated := run
	updated.Status = work.RunRunning
	updated.UpdatedAt = at.Add(time.Minute)
	expected := work.RunQueued
	duplicate := work.LogChunk{RunID: run.ID, Sequence: 1, Stream: "stdout", Text: "duplicate", CreatedAt: at}
	if result := agentStore.Commit(orchestrator.Commit{Run: &updated, ExpectedStatus: &expected, Logs: []work.LogChunk{duplicate}}); result.OK {
		t.Fatal("atomic commit with duplicate log sequence succeeded")
	}
	if got := requireAgentValue[work.Run](t, "Run after rollback", agentStore.Run(run.ID)); got.Status != work.RunQueued {
		t.Fatalf("run status after rollback = %q, want queued", got.Status)
	}

	wrong := work.RunPreparing
	if result := agentStore.Commit(orchestrator.Commit{Run: &updated, ExpectedStatus: &wrong}); result.OK {
		t.Fatal("stale compare-and-swap update succeeded")
	}
	if result := agentStore.Commit(orchestrator.Commit{Run: &updated, ExpectedStatus: &expected}); !result.OK {
		t.Fatalf("valid compare-and-swap update: %v", result.Value)
	}
	if result := agentStore.Commit(orchestrator.Commit{Run: &run, CreateRun: true}); result.OK {
		t.Fatal("duplicate run creation succeeded")
	}
	if result := agentStore.Project("missing"); result.OK {
		t.Fatal("missing project lookup succeeded")
	}
	missing := agentStore.ProjectBySource("/missing")
	if !missing.OK || missing.Value != nil {
		t.Fatalf("missing ProjectBySource = %#v (%v), want successful nil", missing.Value, missing.Err())
	}
}

func TestAgentStore_LegacyDurableRevisionRemainsEmpty(t *testing.T) {
	_, repository, agentStore := openTestAgentStore(t)
	defer closeTestDuckRepository(t, repository)
	at := time.Date(2026, time.July, 18, 10, 30, 0, 0, time.UTC)
	connection := repository.(workspaceConnectionProvider).workspaceConnection()
	_, err := connection.Exec(`INSERT INTO agent_runs (
		id, work_id, project_id, parent_run_id, provider, model, source_revision,
		execution_revision, accepted_revision, branch, worktree, command_receipt,
		run_number, attempt, process_id, status, exit_code, failure_reason,
		queued_at, started_at, finished_at, updated_at
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		"legacy-run", "legacy-work", "project-agent", "", "codex", "", "source", "", "", "branch", "worktree", "",
		1, 1, 0, work.RunQueued, 0, "", at, time.Time{}, time.Time{}, at)
	if err != nil {
		t.Fatalf("insert legacy run: %v", err)
	}
	if run := requireAgentValue[work.Run](t, "legacy Run", agentStore.Run("legacy-run")); run.DurableRevision != "" {
		t.Fatalf("legacy Run durable revision = %q, want empty", run.DurableRevision)
	}
	if continuation := requireAgentValue[work.Continuation](t, "legacy Continuation", agentStore.Continuation("legacy-run")); continuation.Run.DurableRevision != "" {
		t.Fatalf("legacy Continuation durable revision = %q, want empty", continuation.Run.DurableRevision)
	}
	snapshot := requireAgentValue[work.Snapshot](t, "legacy Snapshot", agentStore.Snapshot("legacy-work"))
	if len(snapshot.Runs) != 1 || snapshot.Runs[0].DurableRevision != "" {
		t.Fatalf("legacy Snapshot runs = %#v, want empty durable revision", snapshot.Runs)
	}
}

func TestAgentStore_UglyRecoveryAndReopen(t *testing.T) {
	root, repository, agentStore := openTestAgentStore(t)
	at := time.Date(2026, time.July, 18, 11, 0, 0, 0, time.UTC)
	project := testAgentProject(at)
	if result := agentStore.Commit(orchestrator.Commit{Project: &project}); !result.OK {
		t.Fatalf("commit project: %v", result.Value)
	}
	statuses := []work.RunStatus{work.RunQueued, work.RunPreparing, work.RunRunning, work.RunCancelling, work.RunCompleted}
	for index, status := range statuses {
		run := testAgentRun(core.Sprintf("recover-%d", index), status, at.Add(time.Duration(index)*time.Second))
		run.Number = index + 1
		if result := agentStore.Commit(orchestrator.Commit{Run: &run, CreateRun: true}); !result.OK {
			t.Fatalf("commit %s run: %v", status, result.Value)
		}
	}
	queue := work.QueueState{ID: "default", Status: work.QueueDraining, Reason: "shutdown", UpdatedAt: at}
	provider := work.ProviderState{Provider: "codex", BackoffReason: "rate", BackoffUntil: at.Add(time.Hour), WindowStartedAt: at, WindowAdmissions: 4, UpdatedAt: at}
	if result := agentStore.Commit(orchestrator.Commit{Queue: &queue, Provider: &provider}); !result.OK {
		t.Fatalf("commit reopen state: %v", result.Value)
	}
	recoveredAt := at.Add(2 * time.Hour)
	if count := requireAgentValue[int](t, "Recover", agentStore.Recover(recoveredAt)); count != 4 {
		t.Fatalf("Recover count = %d, want 4", count)
	}
	if result := repository.Close(); !result.OK {
		t.Fatalf("close repository: %v", result.Value)
	}

	reopenedResult := openDuckRepository(root + "/lem.duckdb")
	if !reopenedResult.OK {
		t.Fatalf("reopen repository: %v", reopenedResult.Value)
	}
	reopened := reopenedResult.Value.(workspaceRepository)
	defer closeTestDuckRepository(t, reopened)
	reopenedStore := requireAgentValue[*duckAgentStore](t, "newDuckAgentStore reopened", newDuckAgentStore(reopened))
	snapshot := requireAgentValue[work.Snapshot](t, "reopened Snapshot", reopenedStore.Snapshot("work-agent"))
	if snapshot.Queue != queue || len(snapshot.Providers) != 1 || snapshot.Providers[0] != provider {
		t.Fatalf("reopened queue/provider = %#v / %#v", snapshot.Queue, snapshot.Providers)
	}
	for _, run := range snapshot.Runs {
		if run.Status == work.RunCompleted {
			continue
		}
		if run.Status != work.RunInterrupted || !run.FinishedAt.Equal(recoveredAt) || !run.UpdatedAt.Equal(recoveredAt) || run.DurableRevision != "durable-revision" {
			t.Fatalf("recovered run = %#v", run)
		}
	}
}

type agentStoreCleanupRecoveryReceipt struct {
	Kind           string
	ProjectID      string
	WorkID         string
	RunID          string
	RunNumber      int
	WorkspaceRunID string
	ReviewID       string
	Branch         string
	Worktree       string
}

type agentStoreCleanupRecoveryOutcome struct {
	RecoveryEventID string
	Receipt         agentStoreCleanupRecoveryReceipt
	Error           string
}

func TestAgentStoreCleanupRecoveryReceiptsSurviveReopen(t *testing.T) {
	root, repository, agentStore := openTestAgentStore(t)
	at := time.Date(2026, time.July, 19, 12, 0, 0, 0, time.UTC)
	project := testAgentProject(at)
	run := testAgentRun("cleanup-recovery-run", work.RunPreparing, at)
	workspaceRoot := core.PathJoin(root, "workspaces")
	run.Branch = "lem/work/work-agent/run-1"
	run.Worktree = core.PathJoin(workspaceRoot, project.ID, "runs", run.ID, "worktree")
	runReceipt := agentStoreCleanupRecoveryReceipt{
		Kind: "run", ProjectID: project.ID, WorkID: run.WorkID, RunID: run.ID,
		RunNumber: run.Number, WorkspaceRunID: run.ID, Branch: run.Branch, Worktree: run.Worktree,
	}
	reviewReceipt := agentStoreCleanupRecoveryReceipt{
		Kind: "review", ProjectID: project.ID, WorkID: run.WorkID, RunID: run.ID, ReviewID: "cleanup-review",
		RunNumber: run.Number,
		Branch:    "lem/integration/cleanup-recovery-run/cleanup-review",
		Worktree:  core.PathJoin(workspaceRoot, project.ID, "reviews", run.ID, "cleanup-review", "worktree"),
	}
	runFailure := agentStoreCleanupRecoveryOutcome{RecoveryEventID: "cleanup-recovery-run-event", Receipt: runReceipt, Error: "worktree remove failed"}
	reviewSuccess := agentStoreCleanupRecoveryOutcome{RecoveryEventID: "cleanup-recovery-review-event", Receipt: reviewReceipt}
	events := []work.Event{
		{ID: "cleanup-recovery-run-event", RunID: run.ID, WorkID: run.WorkID, Kind: "workspace_cleanup_retained", Title: "provisional workspace cleanup retained", Detail: runReceipt.Worktree, DetailJSON: core.JSONMarshalString(runReceipt), CreatedAt: at},
		{ID: "cleanup-recovery-review-event", RunID: run.ID, WorkID: run.WorkID, Kind: "review_cleanup_retained", Title: "provisional workspace cleanup retained", Detail: reviewReceipt.Worktree, DetailJSON: core.JSONMarshalString(reviewReceipt), CreatedAt: at.Add(time.Second)},
		{ID: "cleanup-recovery-run-failed", RunID: run.ID, WorkID: run.WorkID, Kind: "cleanup_recovery_failed", Title: "retained cleanup failed", Detail: runReceipt.Worktree, DetailJSON: core.JSONMarshalString(runFailure), CreatedAt: at.Add(2 * time.Second)},
		{ID: "cleanup-recovery-review-succeeded", RunID: run.ID, WorkID: run.WorkID, Kind: "cleanup_recovery_succeeded", Title: "retained cleanup succeeded", Detail: reviewReceipt.Worktree, DetailJSON: core.JSONMarshalString(reviewSuccess), CreatedAt: at.Add(3 * time.Second)},
	}
	if result := agentStore.Commit(orchestrator.Commit{Project: &project, Run: &run, CreateRun: true, Event: &events[0]}); !result.OK {
		t.Fatalf("commit run cleanup recovery: %s", result.Error())
	}
	for index := 1; index < len(events); index++ {
		if result := agentStore.Commit(orchestrator.Commit{Event: &events[index]}); !result.OK {
			t.Fatalf("commit cleanup recovery event %d: %s", index, result.Error())
		}
	}
	if result := repository.Close(); !result.OK {
		t.Fatalf("close repository: %s", result.Error())
	}

	reopenedResult := openDuckRepository(core.PathJoin(root, "lem.duckdb"))
	core.AssertTrue(t, reopenedResult.OK, reopenedResult.Error())
	reopened := reopenedResult.Value.(workspaceRepository)
	defer closeTestDuckRepository(t, reopened)
	reopenedStore := requireAgentValue[*duckAgentStore](t, "newDuckAgentStore reopened cleanup recovery", newDuckAgentStore(reopened))
	snapshot := requireAgentValue[work.Snapshot](t, "reopened cleanup recovery Snapshot", reopenedStore.Snapshot(run.WorkID))
	core.AssertEqual(t, 1, len(snapshot.Runs))
	core.AssertEqual(t, run.Branch, snapshot.Runs[0].Branch)
	core.AssertEqual(t, run.Worktree, snapshot.Runs[0].Worktree)

	wantJSON := map[string]string{
		"workspace_cleanup_retained": core.JSONMarshalString(runReceipt),
		"review_cleanup_retained":    core.JSONMarshalString(reviewReceipt),
		"cleanup_recovery_failed":    core.JSONMarshalString(runFailure),
		"cleanup_recovery_succeeded": core.JSONMarshalString(reviewSuccess),
	}
	for _, event := range snapshot.Events {
		want, exists := wantJSON[event.Kind]
		if !exists {
			continue
		}
		core.AssertEqual(t, want, event.DetailJSON)
		delete(wantJSON, event.Kind)
	}
	core.AssertEqual(t, 0, len(wantJSON))
	mapped := mapAgentSnapshot(snapshot)
	core.AssertEqual(t, 1, len(mapped.Work))
	core.AssertEqual(t, 1, mapped.Work[0].RecoveryCount)
	core.AssertEqual(t, events[0].ID, mapped.Work[0].Recovery.EventID)
}

func TestAgentStore_ConcurrentWriterSerialization(t *testing.T) {
	_, repository, agentStore := openTestAgentStore(t)
	defer closeTestDuckRepository(t, repository)
	stores := []*duckAgentStore{agentStore}
	for len(stores) < 8 {
		stores = append(stores, requireAgentValue[*duckAgentStore](t, "newDuckAgentStore concurrent", newDuckAgentStore(repository)))
	}
	for index, candidate := range stores[1:] {
		if candidate.writeMu != agentStore.writeMu {
			t.Fatalf("store %d has independent write mutex", index+1)
		}
	}
	at := time.Date(2026, time.July, 18, 12, 0, 0, 0, time.UTC)
	run := testAgentRun("run-concurrent", work.RunQueued, at)
	if result := agentStore.Commit(orchestrator.Commit{Run: &run, CreateRun: true}); !result.OK {
		t.Fatalf("create run: %v", result.Value)
	}
	expected := work.RunQueued
	var wait sync.WaitGroup
	results := make(chan bool, 8)
	for index := 0; index < 8; index++ {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			candidate := run
			candidate.Status = work.RunPreparing
			candidate.UpdatedAt = at.Add(time.Duration(index+1) * time.Second)
			results <- stores[index].Commit(orchestrator.Commit{Run: &candidate, ExpectedStatus: &expected}).OK
		}(index)
	}
	wait.Wait()
	close(results)
	succeeded := 0
	for ok := range results {
		if ok {
			succeeded++
		}
	}
	if succeeded != 1 {
		t.Fatalf("concurrent CAS successes = %d, want 1", succeeded)
	}
	if next := requireAgentValue[int](t, "NextRunNumber after concurrency", agentStore.NextRunNumber(run.WorkID)); next != 2 {
		t.Fatalf("next run number = %d, want 2", next)
	}
}

func TestAgentStore_NoSecretOrRepositoryStateFiles(t *testing.T) {
	root, repository, agentStore := openTestAgentStore(t)
	repositoryRoot := root + "/source-repository"
	if err := os.MkdirAll(repositoryRoot, 0o700); err != nil {
		t.Fatalf("create source repository fixture: %v", err)
	}
	if err := os.WriteFile(repositoryRoot+"/README.md", []byte("ordinary repository documentation\n"), 0o600); err != nil {
		t.Fatalf("write ordinary repository fixture: %v", err)
	}
	at := time.Date(2026, time.July, 18, 13, 0, 0, 0, time.UTC)
	rawSecret := "provider-token-7f993"
	rawReceipt := core.Concat("Authorization: Bearer ", rawSecret)
	redactedSentinel := "[REDACTED:provider-credential]"
	redactedReceipt := strings.ReplaceAll(rawReceipt, rawSecret, redactedSentinel)
	if !strings.Contains(rawReceipt, rawSecret) || strings.Contains(redactedReceipt, rawSecret) || !strings.Contains(redactedReceipt, redactedSentinel) {
		t.Fatalf("invalid redaction fixture: raw=%q redacted=%q", rawReceipt, redactedReceipt)
	}
	project := testAgentProject(at)
	run := testAgentRun("run-no-secret", work.RunQueued, at)
	run.CommandReceipt = redactedReceipt
	event := work.Event{ID: "event-no-secret", RunID: run.ID, WorkID: run.WorkID, Kind: "receipt", Title: "redacted", DetailJSON: redactedReceipt, CreatedAt: at}
	log := work.LogChunk{RunID: run.ID, Sequence: 1, Stream: "stdout", Text: redactedReceipt, CreatedAt: at}
	question := work.Question{ID: "question-no-secret", RunID: run.ID, Text: redactedReceipt, CreatedAt: at}
	answer := work.Answer{ID: "answer-no-secret", QuestionID: question.ID, ResumeRunID: run.ID, Text: redactedReceipt, CreatedAt: at}
	acceptance := work.Acceptance{ID: "acceptance-no-secret", WorkID: run.WorkID, RunID: run.ID, Status: "prepared", ValidationJSON: core.Concat(`{"receipt":`, core.JSONMarshalString(redactedReceipt), `}`), CreatedAt: at, UpdatedAt: at}
	queue := work.QueueState{ID: "default", Status: work.QueueFrozen, Reason: redactedReceipt, UpdatedAt: at}
	provider := work.ProviderState{Provider: "codex", BackoffReason: redactedReceipt, WindowStartedAt: at, UpdatedAt: at}
	if result := agentStore.Commit(orchestrator.Commit{Project: &project, Run: &run, CreateRun: true, Event: &event, Logs: []work.LogChunk{log}, Question: &question, Answer: &answer, Acceptance: &acceptance, Queue: &queue, Provider: &provider}); !result.OK {
		t.Fatalf("commit redacted records: %v", result.Value)
	}
	connection := repository.(workspaceConnectionProvider).workspaceConnection()
	tables := []struct {
		name    string
		columns string
	}{
		{name: "agent_projects", columns: "id, source_path, repository_root, source_branch, source_revision, repository_name, clone_path"},
		{name: "agent_runs", columns: "id, work_id, project_id, parent_run_id, provider, model, source_revision, durable_revision, execution_revision, accepted_revision, branch, worktree, command_receipt, status, failure_reason"},
		{name: "agent_events", columns: "id, run_id, work_id, kind, title, detail, detail_json"},
		{name: "agent_log_chunks", columns: "run_id, stream, text"},
		{name: "agent_questions", columns: "id, run_id, text"},
		{name: "agent_answers", columns: "id, question_id, resume_run_id, text"},
		{name: "agent_acceptances", columns: "id, work_id, run_id, source_base, agent_base, agent_tip, integration_branch, integration_worktree, result_revision, status, validation_json, failure_reason"},
		{name: "agent_queue_state", columns: "id, status, reason"},
		{name: "agent_provider_state", columns: "provider, backoff_reason, last_run_id"},
	}
	redactedRows := 0
	for _, table := range tables {
		query := core.Sprintf("SELECT COUNT(*) FROM %s WHERE strpos(concat_ws('|', %s), ?) > 0", table.name, table.columns)
		var rawRows int
		if err := connection.QueryRow(query, rawSecret).Scan(&rawRows); err != nil {
			t.Fatalf("inspect raw values in %s: %v", table.name, err)
		}
		if rawRows != 0 {
			t.Fatalf("%s persisted %d rows containing raw secret", table.name, rawRows)
		}
		var tableRedactedRows int
		if err := connection.QueryRow(query, redactedSentinel).Scan(&tableRedactedRows); err != nil {
			t.Fatalf("inspect redacted values in %s: %v", table.name, err)
		}
		redactedRows += tableRedactedRows
	}
	if redactedRows < 8 {
		t.Fatalf("persisted rows containing redacted receipt = %d, want at least 8", redactedRows)
	}
	if result := repository.Close(); !result.OK {
		t.Fatalf("close repository: %v", result.Value)
	}
	databaseBytes, err := os.ReadFile(root + "/lem.duckdb")
	if err != nil {
		t.Fatalf("read DuckDB file: %v", err)
	}
	if strings.Contains(string(databaseBytes), rawSecret) {
		t.Fatal("DuckDB contains test secret")
	}
	err = filepath.WalkDir(repositoryRoot, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			return nil
		}
		name := strings.ToLower(entry.Name())
		stateSuffix := strings.HasSuffix(name, ".md") || strings.HasSuffix(name, ".json") || strings.HasSuffix(name, ".yaml") || strings.HasSuffix(name, ".yml") || strings.HasSuffix(name, ".status")
		if name == ".lem" || strings.Contains(name, "lem") && stateSuffix {
			t.Errorf("unexpected repository state file: %s", path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("scan repository directory: %v", err)
	}
}

func openTestAgentStore(t *testing.T) (string, workspaceRepository, *duckAgentStore) {
	t.Helper()
	root := t.TempDir()
	opened := openDuckRepository(root + "/lem.duckdb")
	if !opened.OK {
		t.Fatalf("open repository: %v", opened.Value)
	}
	repository := opened.Value.(workspaceRepository)
	storeResult := newDuckAgentStore(repository)
	if !storeResult.OK {
		t.Fatalf("newDuckAgentStore: %v", storeResult.Value)
	}
	agentStore, ok := storeResult.Value.(*duckAgentStore)
	if !ok {
		t.Fatalf("newDuckAgentStore value = %T, want *duckAgentStore", storeResult.Value)
	}
	return root, repository, agentStore
}

func testAgentProject(at time.Time) work.Project {
	return work.Project{ID: "project-agent", SourcePath: "/source", RepositoryRoot: "/source", SourceBranch: "main", SourceRevision: "source-revision", RepositoryName: "source.git", ClonePath: "/private/source.git", CreatedAt: at, UpdatedAt: at}
}

func testAgentRun(id string, status work.RunStatus, at time.Time) work.Run {
	return work.Run{ID: id, WorkID: "work-agent", ProjectID: "project-agent", Provider: "codex", Model: "gpt-5", SourceRevision: "source-revision", DurableRevision: "durable-revision", ExecutionRevision: "execution-revision", AcceptedRevision: "accepted-revision", Branch: "lem/work-agent/run-1", Worktree: "/private/run-1", CommandReceipt: "receipt", Status: status, Number: 1, Attempt: 1, ProcessID: 123, QueuedAt: at, StartedAt: at, FinishedAt: at, UpdatedAt: at}
}

func testChangeReview(run work.Run, conflicts []string, passed bool) workspace.ChangeReview {
	return workspace.ChangeReview{WorkID: run.WorkID, RunID: run.ID, SourceBranch: "main", SourceRevision: "source", AgentBase: "base", AgentTip: "tip", IntegrationBranch: "integration", IntegrationPath: "/tmp/integration", ResultRevision: "result", Diff: "diff --git a/a b/a\n", CommitLog: "abc\tchange", Validation: []workspace.ValidationResult{{Command: workspace.Command{Dir: "/tmp/integration", Executable: "go", Args: []string{"test", "./..."}, Environment: []string{"SAFE=1"}}, ExitCode: 0, Output: "ok", Receipt: "validation-receipt", Passed: passed}}, Conflicts: append([]string(nil), conflicts...)}
}

func requireAgentValue[T any](t *testing.T, operation string, result core.Result) T {
	t.Helper()
	if !result.OK {
		t.Fatalf("%s failed: %v", operation, result.Value)
	}
	value, ok := result.Value.(T)
	if !ok {
		t.Fatalf("%s value = %T, want %T", operation, result.Value, *new(T))
	}
	return value
}
