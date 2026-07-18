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
	for _, fixture := range reviews {
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

func TestAgentStore_Bad(t *testing.T) {
	_, repository, agentStore := openTestAgentStore(t)
	defer closeTestDuckRepository(t, repository)
	at := time.Date(2026, time.July, 18, 10, 0, 0, 0, time.UTC)
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

func TestAgentStore_ConcurrentWriterSerialization(t *testing.T) {
	_, repository, agentStore := openTestAgentStore(t)
	defer closeTestDuckRepository(t, repository)
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
			results <- agentStore.Commit(orchestrator.Commit{Run: &candidate, ExpectedStatus: &expected}).OK
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
	run := testAgentRun("run-no-secret", work.RunQueued, at)
	if result := agentStore.Commit(orchestrator.Commit{Run: &run, CreateRun: true}); !result.OK {
		t.Fatalf("commit run: %v", result.Value)
	}
	secret := "LEM_TEST_SECRET_DO_NOT_PERSIST_7f993"
	for _, table := range []string{"agent_projects", "agent_runs", "agent_events", "agent_log_chunks", "agent_questions", "agent_answers", "agent_acceptances", "agent_queue_state", "agent_provider_state"} {
		var columns int
		query := core.Sprintf(`SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = 'main' AND table_name = ? AND LOWER(column_name) LIKE '%%secret%%'`)
		if err := repository.(*duckRepository).database.store.Conn().QueryRow(query, table).Scan(&columns); err != nil {
			t.Fatalf("inspect %s columns: %v", table, err)
		}
		if columns != 0 {
			t.Fatalf("%s has %d secret-bearing columns", table, columns)
		}
	}
	if result := repository.Close(); !result.OK {
		t.Fatalf("close repository: %v", result.Value)
	}
	databaseBytes, err := os.ReadFile(root + "/lem.duckdb")
	if err != nil {
		t.Fatalf("read DuckDB file: %v", err)
	}
	if strings.Contains(string(databaseBytes), secret) {
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
