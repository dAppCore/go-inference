// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	"sync"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/work"
)

type storeContractFake struct {
	mu          sync.Mutex
	commits     []Commit
	runs        map[string]work.Run
	recoverFail bool
}

func newStoreContractFake() *storeContractFake {
	return &storeContractFake{runs: make(map[string]work.Run)}
}

func (store *storeContractFake) Recover(time.Time) core.Result {
	if store.recoverFail {
		return core.Fail(core.NewError("injected recovery failure"))
	}
	return core.Ok(0)
}

func (store *storeContractFake) Commit(commit Commit) core.Result {
	store.mu.Lock()
	defer store.mu.Unlock()
	if commit.Run != nil {
		existing, exists := store.runs[commit.Run.ID]
		if commit.CreateRun {
			if exists {
				return core.Fail(core.NewError("duplicate run"))
			}
			store.runs[commit.Run.ID] = *commit.Run
		} else {
			if !exists || commit.ExpectedStatus == nil || existing.Status != *commit.ExpectedStatus {
				return core.Fail(core.NewError("stale expected status"))
			}
			store.runs[commit.Run.ID] = *commit.Run
		}
	}
	store.commits = append(store.commits, commit)
	return core.Ok(nil)
}

func (store *storeContractFake) Project(string) core.Result {
	return core.Fail(core.NewError("not found"))
}

func (store *storeContractFake) ProjectBySource(string) core.Result {
	return core.Fail(core.NewError("not found"))
}

func (store *storeContractFake) Run(id string) core.Result {
	store.mu.Lock()
	defer store.mu.Unlock()
	run, exists := store.runs[id]
	if !exists {
		return core.Fail(core.NewError("not found"))
	}
	return core.Ok(run)
}

func (store *storeContractFake) NextRunNumber(string) core.Result {
	return core.Ok(1)
}

func (store *storeContractFake) Continuation(string) core.Result {
	return core.Ok(work.Continuation{})
}

func (store *storeContractFake) Snapshot(string) core.Result {
	return core.Ok(work.Snapshot{})
}

func storeTestTime() time.Time {
	return time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
}

func storeTestRun(status work.RunStatus) work.Run {
	return work.Run{
		ID: "run-1", WorkID: "work-1", ProjectID: "project-1", Provider: "codex",
		SourceRevision: "abc123", Branch: "lem/work/work-1/run-1", Worktree: "/tmp/worktree",
		Status: status, Number: 1, Attempt: 1, QueuedAt: storeTestTime(), UpdatedAt: storeTestTime(),
	}
}

func storeTestEvent(kind string) work.Event {
	return work.Event{ID: core.Concat("event-", kind), RunID: "run-1", WorkID: "work-1", Kind: kind, Title: kind, CreatedAt: storeTestTime()}
}

func storeTestProject() work.Project {
	return work.Project{
		ID: "project-1", SourcePath: "/source", RepositoryRoot: "/source", SourceBranch: "main",
		SourceRevision: "abc123", RepositoryName: "project-1", ClonePath: "/internal/project-1/repo.git",
		CreatedAt: storeTestTime(), UpdatedAt: storeTestTime(),
	}
}

func TestStoreCommitValidation(t *testing.T) {
	store := newStoreContractFake()
	core.AssertFalse(t, commitStore(nil, Commit{}).OK)
	core.AssertFalse(t, commitStore(store, Commit{}).OK)

	run := storeTestRun(work.RunQueued)
	expected := work.RunQueued
	project := storeTestProject()
	projectWithoutTime := project
	projectWithoutTime.CreatedAt = time.Time{}
	runWithoutNumber := run
	runWithoutNumber.Number = 0
	runWithoutTime := run
	runWithoutTime.UpdatedAt = time.Time{}
	unknownStatus := work.RunStatus("unknown")
	invalid := []Commit{
		{CreateRun: true},
		{Run: &run, CreateRun: true, ExpectedStatus: &expected},
		{Run: &run},
		{ExpectedStatus: &expected, Event: func() *work.Event { event := storeTestEvent("progress"); return &event }()},
		{Run: &work.Run{Status: work.RunQueued}, CreateRun: true},
		{Run: &work.Run{ID: "run", WorkID: "work", ProjectID: "project", Status: work.RunStatus("lost")}, CreateRun: true},
		{Project: &work.Project{ID: "project"}},
		{Project: &projectWithoutTime},
		{Run: &runWithoutNumber, CreateRun: true},
		{Run: &runWithoutTime, CreateRun: true},
		{Run: &run, ExpectedStatus: &unknownStatus},
		{Event: &work.Event{ID: "event"}},
		{Logs: []work.LogChunk{{RunID: "run-1", Sequence: 0, Stream: "stdout", Text: "line", CreatedAt: storeTestTime()}}},
		{Logs: []work.LogChunk{{RunID: "run-1", Sequence: 2, Stream: "stdout", Text: "two", CreatedAt: storeTestTime()}, {RunID: "run-1", Sequence: 1, Stream: "stdout", Text: "one", CreatedAt: storeTestTime()}}},
		{Question: &work.Question{ID: "question"}},
		{Answer: &work.Answer{ID: "answer"}},
		{Acceptance: &work.Acceptance{ID: "acceptance"}},
		{Queue: &work.QueueState{ID: "wrong", Status: work.QueueAccepting, UpdatedAt: storeTestTime()}},
		{Queue: &work.QueueState{ID: "default", Status: work.QueueStatus("unknown"), UpdatedAt: storeTestTime()}},
		{Provider: &work.ProviderState{}},
	}
	for index, commit := range invalid {
		result := commitStore(store, commit)
		core.AssertFalse(t, result.OK, core.Sprintf("invalid commit %d unexpectedly passed", index))
	}
	core.AssertEqual(t, 0, len(store.commits))
}

func TestStoreCommitAtomicShapes(t *testing.T) {
	store := newStoreContractFake()
	queued := storeTestRun(work.RunQueued)
	queuedEvent := storeTestEvent("queued")
	queueState := work.QueueState{ID: "default", Status: work.QueueAccepting, UpdatedAt: storeTestTime()}
	core.AssertTrue(t, commitStore(store, Commit{Run: &queued, CreateRun: true, Event: &queuedEvent, Queue: &queueState}).OK)
	project := storeTestProject()
	core.AssertTrue(t, commitStore(store, Commit{Project: &project}).OK)

	running := queued
	running.Status = work.RunRunning
	running.StartedAt = storeTestTime().Add(time.Minute)
	running.UpdatedAt = running.StartedAt
	expectedQueued := work.RunQueued
	providerState := work.ProviderState{Provider: "codex", LastRunID: running.ID, LastStartedAt: running.StartedAt, UpdatedAt: running.StartedAt, WindowAdmissions: 1}
	core.AssertTrue(t, commitStore(store, Commit{Run: &running, ExpectedStatus: &expectedQueued, Provider: &providerState}).OK)

	waiting := running
	waiting.Status = work.RunWaiting
	waiting.FinishedAt = storeTestTime().Add(2 * time.Minute)
	waiting.UpdatedAt = waiting.FinishedAt
	expectedRunning := work.RunRunning
	question := work.Question{ID: "question-1", RunID: waiting.ID, Text: "Which API?", CreatedAt: waiting.FinishedAt}
	core.AssertTrue(t, commitStore(store, Commit{Run: &waiting, ExpectedStatus: &expectedRunning, Question: &question}).OK)

	backoffEvent := storeTestEvent("rate_limit")
	providerState.BackoffReason = "quota"
	providerState.BackoffUntil = storeTestTime().Add(time.Hour)
	providerState.UpdatedAt = storeTestTime().Add(3 * time.Minute)
	core.AssertTrue(t, commitStore(store, Commit{Event: &backoffEvent, Provider: &providerState}).OK)

	answer := work.Answer{ID: "answer-1", QuestionID: question.ID, ResumeRunID: "run-2", Text: "Use v2", CreatedAt: storeTestTime().Add(4 * time.Minute)}
	core.AssertTrue(t, commitStore(store, Commit{Answer: &answer}).OK)

	logs := []work.LogChunk{
		{RunID: waiting.ID, Sequence: 1, Stream: "stdout", Text: "one", CreatedAt: storeTestTime()},
		{RunID: waiting.ID, Sequence: 2, Stream: "stderr", Text: "two", CreatedAt: storeTestTime()},
	}
	core.AssertTrue(t, commitStore(store, Commit{Logs: logs}).OK)

	accepted := waiting
	accepted.Status = work.RunAccepted
	accepted.AcceptedRevision = "def456"
	accepted.UpdatedAt = storeTestTime().Add(5 * time.Minute)
	expectedWaiting := work.RunWaiting
	acceptance := work.Acceptance{
		ID: "acceptance-1", WorkID: accepted.WorkID, RunID: accepted.ID,
		SourceBase: "abc123", AgentBase: "abc123", AgentTip: "def456", ResultRevision: "def456",
		Status: "accepted", CreatedAt: accepted.UpdatedAt, UpdatedAt: accepted.UpdatedAt,
	}
	core.AssertTrue(t, commitStore(store, Commit{Run: &accepted, ExpectedStatus: &expectedWaiting, Acceptance: &acceptance}).OK)
	core.AssertEqual(t, 8, len(store.commits))
}

func TestStoreCommitConflictsPropagate(t *testing.T) {
	store := newStoreContractFake()
	run := storeTestRun(work.RunQueued)
	core.AssertTrue(t, commitStore(store, Commit{Run: &run, CreateRun: true}).OK)
	duplicate := commitStore(store, Commit{Run: &run, CreateRun: true})
	core.AssertFalse(t, duplicate.OK)
	core.AssertContains(t, duplicate.Error(), "duplicate")

	running := run
	running.Status = work.RunRunning
	wrong := work.RunPreparing
	stale := commitStore(store, Commit{Run: &running, ExpectedStatus: &wrong})
	core.AssertFalse(t, stale.OK)
	core.AssertContains(t, stale.Error(), "stale")
}

func TestStoreCommitRelations(t *testing.T) {
	store := newStoreContractFake()
	run := storeTestRun(work.RunQueued)
	event := storeTestEvent("queued")
	event.RunID = "another"
	core.AssertFalse(t, commitStore(store, Commit{Run: &run, CreateRun: true, Event: &event}).OK)
	log := work.LogChunk{RunID: "another", Sequence: 1, Stream: "stdout", Text: "line", CreatedAt: storeTestTime()}
	core.AssertFalse(t, commitStore(store, Commit{Run: &run, CreateRun: true, Logs: []work.LogChunk{log}}).OK)
	question := work.Question{ID: "question", RunID: "another", Text: "question", CreatedAt: storeTestTime()}
	core.AssertFalse(t, commitStore(store, Commit{Run: &run, CreateRun: true, Question: &question}).OK)
	answer := work.Answer{ID: "answer", QuestionID: "question", ResumeRunID: "another", Text: "answer", CreatedAt: storeTestTime()}
	core.AssertFalse(t, commitStore(store, Commit{Run: &run, CreateRun: true, Answer: &answer}).OK)
	acceptance := work.Acceptance{ID: "acceptance", WorkID: run.WorkID, RunID: "another", Status: "accepted", CreatedAt: storeTestTime(), UpdatedAt: storeTestTime()}
	core.AssertFalse(t, commitStore(store, Commit{Run: &run, CreateRun: true, Acceptance: &acceptance}).OK)
}

func TestStoreRecoveryBoundary(t *testing.T) {
	store := newStoreContractFake()
	core.AssertFalse(t, recoverStore(nil, storeTestTime()).OK)
	core.AssertFalse(t, recoverStore(store, time.Time{}).OK)
	store.recoverFail = true
	result := recoverStore(store, storeTestTime())
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "recovery")
	store.recoverFail = false
	core.AssertTrue(t, recoverStore(store, storeTestTime()).OK)
}
