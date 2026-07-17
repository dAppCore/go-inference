// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
)

func TestDuckRepository_Good(t *testing.T) {
	path := t.TempDir() + "/lem.duckdb"
	opened := openDuckRepository(path)
	if !opened.OK {
		t.Fatalf("openDuckRepository(%q) failed: %v", path, opened.Value)
	}
	repository, ok := opened.Value.(workspaceRepository)
	if !ok {
		t.Fatalf("openDuckRepository value = %T, want workspaceRepository", opened.Value)
	}

	createdAt := time.Date(2026, time.July, 17, 10, 0, 0, 0, time.UTC)
	session := testSessionRecord("session-good", "Persistent workspace", createdAt)
	userTurn := testTurnRecord("turn-user", session.ID, 1, "user", "Remember this session", createdAt.Add(time.Minute))
	assistantTurn := testTurnRecord("turn-assistant", session.ID, 2, "assistant", "It will survive a restart.", createdAt.Add(2*time.Minute))
	event := eventRecord{
		ID:          "event-good",
		SessionID:   session.ID,
		Kind:        "generation",
		Status:      "completed",
		Title:       "Answer completed",
		Detail:      "The response was persisted.",
		PayloadJSON: "{}",
		CreatedAt:   createdAt.Add(3 * time.Minute),
	}
	job := generationJobRecord{
		ID:           "job-good",
		SessionID:    session.ID,
		PromptTurnID: userTurn.ID,
		AnswerTurnID: assistantTurn.ID,
		Status:       "completed",
		Model:        "fixture-model",
		MetricsJSON:  "{}",
		CreatedAt:    createdAt.Add(time.Minute),
		StartedAt:    createdAt.Add(time.Minute),
		FinishedAt:   createdAt.Add(2 * time.Minute),
	}
	work := workItemRecord{
		ID:         "work-good",
		ExternalID: "forge-42",
		Source:     "forge",
		Title:      "Persist the workspace",
		Status:     "completed",
		SessionID:  session.ID,
		StartedAt:  createdAt,
		UpdatedAt:  createdAt.Add(4 * time.Minute),
		ArchivedAt: unsetRecordTime(),
	}
	artifact := artifactRecord{
		ID:           "artifact-good",
		SessionID:    session.ID,
		WorkItemID:   work.ID,
		Kind:         "patch",
		Path:         "workspaces/session-good/change.patch",
		Title:        "Workspace patch",
		MetadataJSON: "{}",
		CreatedAt:    createdAt.Add(4 * time.Minute),
		ArchivedAt:   unsetRecordTime(),
	}
	attachment := attachmentRecord{
		ID:            "attachment-good",
		SessionID:     session.ID,
		SourcePath:    "/tmp/context.md",
		Title:         "Context",
		ContentHash:   "sha256:fixture",
		Snapshot:      "A durable local snapshot.",
		AddedAt:       createdAt,
		LastCheckedAt: createdAt.Add(4 * time.Minute),
		ArchivedAt:    unsetRecordTime(),
	}

	for _, save := range []struct {
		label string
		call  func() core.Result
	}{
		{label: "session", call: func() core.Result { return repository.SaveSession(session) }},
		{label: "user turn", call: func() core.Result { return repository.SaveTurn(userTurn) }},
		{label: "answer turn", call: func() core.Result { return repository.SaveTurn(assistantTurn) }},
		{label: "event", call: func() core.Result { return repository.SaveEvent(event) }},
		{label: "job", call: func() core.Result { return repository.SaveJob(job) }},
		{label: "work item", call: func() core.Result { return repository.SaveWorkItem(work) }},
		{label: "artifact", call: func() core.Result { return repository.SaveArtifact(artifact) }},
		{label: "attachment", call: func() core.Result { return repository.SaveAttachment(attachment) }},
	} {
		if result := save.call(); !result.OK {
			t.Fatalf("save %s failed: %v", save.label, result.Value)
		}
	}
	if result := repository.Close(); !result.OK {
		t.Fatalf("close first repository: %v", result.Value)
	}

	reopened := openDuckRepository(path)
	if !reopened.OK {
		t.Fatalf("reopen repository: %v", reopened.Value)
	}
	repository, ok = reopened.Value.(workspaceRepository)
	if !ok {
		t.Fatalf("reopened value = %T, want workspaceRepository", reopened.Value)
	}
	defer func() {
		if result := repository.Close(); !result.OK {
			t.Errorf("close reopened repository: %v", result.Value)
		}
	}()

	sessionResult := repository.Session(session.ID)
	if !sessionResult.OK {
		t.Fatalf("read session: %v", sessionResult.Value)
	}
	gotSession, ok := sessionResult.Value.(sessionRecord)
	if !ok {
		t.Fatalf("Session value = %T, want sessionRecord", sessionResult.Value)
	}
	if gotSession.ID != session.ID || gotSession.Title != session.Title || !gotSession.CreatedAt.Equal(session.CreatedAt) {
		t.Fatalf("session round trip = %#v, want core fields from %#v", gotSession, session)
	}

	turnsResult := repository.Turns(session.ID)
	if !turnsResult.OK {
		t.Fatalf("read turns: %v", turnsResult.Value)
	}
	turns, ok := turnsResult.Value.([]turnRecord)
	if !ok || len(turns) != 2 {
		t.Fatalf("Turns value = %#v (%T), want two []turnRecord", turnsResult.Value, turnsResult.Value)
	}
	if turns[0].ID != userTurn.ID || turns[1].ID != assistantTurn.ID {
		t.Fatalf("turn order = %q, %q, want %q, %q", turns[0].ID, turns[1].ID, userTurn.ID, assistantTurn.ID)
	}

	assertRecordSliceLength[eventRecord](t, "Events", repository.Events(session.ID), 1)
	assertRecordSliceLength[generationJobRecord](t, "Jobs", repository.Jobs(session.ID), 1)
	assertRecordSliceLength[workItemRecord](t, "ListWorkItems", repository.ListWorkItems(false), 1)
	assertRecordSliceLength[artifactRecord](t, "Artifacts", repository.Artifacts(session.ID), 1)
	assertRecordSliceLength[attachmentRecord](t, "Attachments", repository.Attachments(session.ID), 1)

	workResult := repository.WorkItemByExternalID(work.ExternalID)
	if !workResult.OK {
		t.Fatalf("read work item by external ID: %v", workResult.Value)
	}
	gotWork, ok := workResult.Value.(workItemRecord)
	if !ok || gotWork.ID != work.ID {
		t.Fatalf("WorkItemByExternalID value = %#v (%T), want ID %q", workResult.Value, workResult.Value, work.ID)
	}
}

func TestDuckRepository_Bad(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)

	createdAt := time.Date(2026, time.July, 17, 11, 0, 0, 0, time.UTC)
	session := testSessionRecord("session-duplicate", "Duplicate sequence", createdAt)
	if result := repository.SaveSession(session); !result.OK {
		t.Fatalf("save session: %v", result.Value)
	}
	first := testTurnRecord("turn-first", session.ID, 1, "user", "First", createdAt)
	duplicate := testTurnRecord("turn-duplicate", session.ID, 1, "assistant", "Must fail", createdAt.Add(time.Second))
	if result := repository.SaveTurn(first); !result.OK {
		t.Fatalf("save first turn: %v", result.Value)
	}
	if result := repository.SaveTurn(duplicate); result.OK {
		t.Fatalf("SaveTurn duplicate sequence = %#v, want failure", result.Value)
	}

	result := repository.Turns(session.ID)
	if !result.OK {
		t.Fatalf("read turns after duplicate: %v", result.Value)
	}
	turns, ok := result.Value.([]turnRecord)
	if !ok || len(turns) != 1 || turns[0].ID != first.ID || turns[0].Visible != first.Visible {
		t.Fatalf("turns after duplicate = %#v (%T), want only first turn", result.Value, result.Value)
	}
}

func TestDuckRepository_Ugly(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)

	createdAt := time.Date(2026, time.July, 17, 12, 0, 0, 0, time.UTC)
	session := testSessionRecord("session-recovery", "Recovery", createdAt)
	if result := repository.SaveSession(session); !result.OK {
		t.Fatalf("save session: %v", result.Value)
	}
	jobs := []generationJobRecord{
		testJobRecord("job-queued", session.ID, "queued", createdAt, unsetRecordTime()),
		testJobRecord("job-generating", session.ID, "generating", createdAt.Add(time.Second), unsetRecordTime()),
		testJobRecord("job-completed", session.ID, "completed", createdAt.Add(2*time.Second), createdAt.Add(3*time.Second)),
	}
	for _, job := range jobs {
		if result := repository.SaveJob(job); !result.OK {
			t.Fatalf("save %s: %v", job.ID, result.Value)
		}
	}

	interruptedAt := createdAt.Add(time.Hour)
	if result := repository.InterruptActiveJobs(interruptedAt); !result.OK {
		t.Fatalf("InterruptActiveJobs: %v", result.Value)
	}
	result := repository.Jobs(session.ID)
	if !result.OK {
		t.Fatalf("read recovered jobs: %v", result.Value)
	}
	got, ok := result.Value.([]generationJobRecord)
	if !ok || len(got) != 3 {
		t.Fatalf("Jobs value = %#v (%T), want three jobs", result.Value, result.Value)
	}
	byID := make(map[string]generationJobRecord, len(got))
	for _, job := range got {
		byID[job.ID] = job
	}
	for _, id := range []string{"job-queued", "job-generating"} {
		if byID[id].Status != "interrupted" || !byID[id].FinishedAt.Equal(interruptedAt) {
			t.Errorf("%s after recovery = %#v, want interrupted at %v", id, byID[id], interruptedAt)
		}
	}
	if completed := byID["job-completed"]; completed.Status != "completed" || !completed.FinishedAt.Equal(jobs[2].FinishedAt) {
		t.Errorf("completed job changed to %#v", completed)
	}
}

func TestDuckRepository_SearchAndArchive_Good(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)

	createdAt := time.Date(2026, time.July, 17, 13, 0, 0, 0, time.UTC)
	titleMatch := testSessionRecord("session-title-match", "Database cleanup", createdAt)
	turnMatch := testSessionRecord("session-turn-match", "Query planning", createdAt.Add(time.Minute))
	for _, session := range []sessionRecord{titleMatch, turnMatch} {
		if result := repository.SaveSession(session); !result.OK {
			t.Fatalf("save %s: %v", session.ID, result.Value)
		}
	}
	if result := repository.SaveTurn(testTurnRecord(
		"turn-search", turnMatch.ID, 1, "user", "Please tune this DUCKDB query", createdAt.Add(2*time.Minute),
	)); !result.OK {
		t.Fatalf("save search turn: %v", result.Value)
	}

	titleResult := repository.SearchSessions("DATABASE", 10)
	if !titleResult.OK {
		t.Fatalf("search title case-insensitively: %v", titleResult.Value)
	}
	titleHits, ok := titleResult.Value.([]sessionSearchHit)
	if !ok || len(titleHits) != 1 || titleHits[0].Session.ID != titleMatch.ID {
		t.Fatalf("title search hits = %#v (%T), want %q", titleResult.Value, titleResult.Value, titleMatch.ID)
	}
	if !strings.Contains(strings.ToLower(titleHits[0].Snippet), "database") {
		t.Fatalf("title snippet = %q, want useful matching text", titleHits[0].Snippet)
	}

	turnResult := repository.SearchSessions("duckdb", 10)
	if !turnResult.OK {
		t.Fatalf("search turn case-insensitively: %v", turnResult.Value)
	}
	turnHits, ok := turnResult.Value.([]sessionSearchHit)
	if !ok || len(turnHits) != 1 || turnHits[0].Session.ID != turnMatch.ID {
		t.Fatalf("turn search hits = %#v (%T), want %q", turnResult.Value, turnResult.Value, turnMatch.ID)
	}
	if !strings.Contains(strings.ToLower(turnHits[0].Snippet), "duckdb") {
		t.Fatalf("turn snippet = %q, want matching turn content", turnHits[0].Snippet)
	}

	turnMatch.Archived = true
	turnMatch.ArchivedAt = createdAt.Add(3 * time.Minute)
	if result := repository.SaveSession(turnMatch); !result.OK {
		t.Fatalf("archive session: %v", result.Value)
	}
	visibleResult := repository.ListSessions(false)
	if !visibleResult.OK {
		t.Fatalf("list visible sessions: %v", visibleResult.Value)
	}
	visible, ok := visibleResult.Value.([]sessionRecord)
	if !ok || len(visible) != 1 || visible[0].ID != titleMatch.ID {
		t.Fatalf("visible sessions = %#v (%T), want only %q", visibleResult.Value, visibleResult.Value, titleMatch.ID)
	}
	archivedSearch := repository.SearchSessions("duckdb", 10)
	if !archivedSearch.OK {
		t.Fatalf("search after archive: %v", archivedSearch.Value)
	}
	if hits, ok := archivedSearch.Value.([]sessionSearchHit); !ok || len(hits) != 0 {
		t.Fatalf("archived search hits = %#v (%T), want empty", archivedSearch.Value, archivedSearch.Value)
	}
	allResult := repository.ListSessions(true)
	if !allResult.OK {
		t.Fatalf("list sessions including archived: %v", allResult.Value)
	}
	all, ok := allResult.Value.([]sessionRecord)
	if !ok || len(all) != 2 {
		t.Fatalf("all sessions = %#v (%T), want two", allResult.Value, allResult.Value)
	}
}

func openTestDuckRepository(t *testing.T) workspaceRepository {
	t.Helper()
	result := openDuckRepository(t.TempDir() + "/lem.duckdb")
	if !result.OK {
		t.Fatalf("open test repository: %v", result.Value)
	}
	repository, ok := result.Value.(workspaceRepository)
	if !ok {
		t.Fatalf("openDuckRepository value = %T, want workspaceRepository", result.Value)
	}
	return repository
}

func closeTestDuckRepository(t *testing.T, repository workspaceRepository) {
	t.Helper()
	if result := repository.Close(); !result.OK {
		t.Errorf("close test repository: %v", result.Value)
	}
}

func testSessionRecord(id, title string, createdAt time.Time) sessionRecord {
	return sessionRecord{
		ID:             id,
		Title:          title,
		Status:         "ready",
		Mode:           "chat",
		GenerationJSON: "{}",
		ToolsJSON:      "[]",
		CreatedAt:      createdAt,
		UpdatedAt:      createdAt,
		LastOpenedAt:   createdAt,
		ArchivedAt:     unsetRecordTime(),
	}
}

func testTurnRecord(id, sessionID string, sequence int64, role, visible string, createdAt time.Time) turnRecord {
	return turnRecord{
		ID:             id,
		SessionID:      sessionID,
		Sequence:       sequence,
		Role:           role,
		Visible:        visible,
		ToolCallJSON:   "{}",
		ToolResultJSON: "{}",
		CreatedAt:      createdAt,
		UpdatedAt:      createdAt,
	}
}

func testJobRecord(id, sessionID, status string, createdAt, finishedAt time.Time) generationJobRecord {
	return generationJobRecord{
		ID:          id,
		SessionID:   sessionID,
		Status:      status,
		Model:       "fixture-model",
		MetricsJSON: "{}",
		CreatedAt:   createdAt,
		StartedAt:   unsetRecordTime(),
		FinishedAt:  finishedAt,
	}
}

func assertRecordSliceLength[T any](t *testing.T, label string, result core.Result, want int) {
	t.Helper()
	if !result.OK {
		t.Fatalf("%s failed: %v", label, result.Value)
	}
	records, ok := result.Value.([]T)
	if !ok || len(records) != want {
		t.Fatalf("%s value = %#v (%T), want %d records", label, result.Value, result.Value, want)
	}
}
