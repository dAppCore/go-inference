// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"

	core "dappco.re/go"
)

func TestWorkPanel_AgentSnapshotKeepsLocalEventsAndOrderedLogs(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	now := time.Date(2026, time.July, 18, 11, 0, 0, 0, time.UTC)
	provider := &fixtureAgentProvider{snapshot: agentSnapshot{
		Work: []agentWorkSnapshot{{ExternalID: "local-1", NativeRunID: "run-9", Title: "Provider title", Repo: "/provider/repo", Status: "running"}},
		Events: []agentEventSnapshot{
			{ExternalID: "stderr-2", WorkID: "local-1", RunID: "run-9", Sequence: 2, Stream: "stderr", Kind: "log.stderr", Detail: "second", CreatedAt: now},
			{ExternalID: "stdout-1", WorkID: "local-1", RunID: "run-9", Sequence: 1, Stream: "stdout", Kind: "log.stdout", Detail: "first", CreatedAt: now},
		},
	}}
	opened := newWorkPanel(repository, provider, sequenceIDs("local-1"), func() time.Time { return now })
	if !opened.OK {
		t.Fatalf("newWorkPanel: %v", opened.Value)
	}
	panel := opened.Value.(*workPanel)
	if result := panel.CreateWork("Local", "keep local identity", "/tmp/local"); !result.OK {
		t.Fatalf("CreateWork: %v", result.Value)
	}
	local := panel.Items()[0]
	if result := repository.SaveEvent(eventRecord{ID: "local-event", SessionID: workEventSessionID(local), WorkItemID: local.ID, Kind: "local.note", Status: "recorded", Title: "Local note", PayloadJSON: "{}", CreatedAt: now}); !result.OK {
		t.Fatalf("SaveEvent: %v", result.Value)
	}
	if result := panel.Refresh(context.Background()); !result.OK {
		t.Fatalf("Refresh: %v", result.Value)
	}
	if items := panel.Items(); len(items) != 1 || items[0].ID != local.ID || items[0].Title != "Local" || items[0].Repo != "/tmp/local" || panel.AgentState(items[0]).NativeRunID != "run-9" {
		t.Fatalf("local identity/run = %#v / %#v", items, panel.AgentState(items[0]))
	}
	events := panel.Events(local.ID)
	if len(events) != 3 || events[0].Kind != "log.stdout" || events[1].Kind != "log.stderr" || events[2].ID != "local-event" {
		t.Fatalf("merged events = %#v", events)
	}
	if stored := repository.Events(workEventSessionID(local)).Value.([]eventRecord); len(stored) != 1 || stored[0].ID != "local-event" {
		t.Fatalf("provider events were copied into lem_events: %#v", stored)
	}
	view := panel.View(120, 24, newUIStyles(midnightTheme()))
	if !strings.Contains(view, "STDOUT") || !strings.Contains(view, "STDERR") || !strings.Contains(view, "first") || !strings.Contains(view, "second") {
		t.Fatalf("live log view:\n%s", view)
	}
}

func TestWorkPanel_AgentStatusRenderingMatrix(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	statuses := []string{"queued", "preparing", "running", "completed", "waiting", "cancelling", "cancelled", "failed", "interrupted", "accepted", "rejected"}
	for index, status := range statuses {
		record := testWorkRecord("status-"+status, status, status, time.Now().Add(time.Duration(index)*time.Second))
		if result := repository.SaveWorkItem(record); !result.OK {
			t.Fatalf("SaveWorkItem(%s): %v", status, result.Value)
		}
	}
	opened := newWorkPanel(repository, newUnavailableAgentProvider(defaultAgentUnavailableReason), nil, time.Now)
	if !opened.OK {
		t.Fatalf("newWorkPanel: %v", opened.Value)
	}
	panel := opened.Value.(*workPanel)
	for _, status := range statuses {
		panel.selectWork("status-" + status)
		view := panel.View(120, 22, newUIStyles(midnightTheme()))
		if !strings.Contains(view, strings.ToUpper(status)) {
			t.Fatalf("status %s was not rendered:\n%s", status, view)
		}
	}
}

func TestWorkPanel_AgentLogsUseSequenceBoundAndRemainWorkIsolated(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	at := time.Date(2026, time.July, 18, 12, 0, 0, 0, time.UTC)
	opened := newWorkPanel(repository, newUnavailableAgentProvider(defaultAgentUnavailableReason), sequenceIDs("work-a", "work-b"), func() time.Time { return at })
	if !opened.OK {
		t.Fatalf("newWorkPanel: %v", opened.Value)
	}
	panel := opened.Value.(*workPanel)
	first := panel.CreateWork("First", "first timeline", "/src/first").Value.(workItemRecord)
	second := panel.CreateWork("Second", "second timeline", "/src/second").Value.(workItemRecord)
	for index := 0; index < 3; index++ {
		if result := repository.SaveEvent(eventRecord{
			ID: core.Sprintf("local-%d", index), SessionID: workEventSessionID(first), WorkItemID: first.ID,
			Kind: "local.note", Status: "recorded", Title: core.Sprintf("local note %d", index),
			PayloadJSON: "{}", CreatedAt: at.Add(time.Duration(index) * time.Second),
		}); !result.OK {
			t.Fatalf("SaveEvent local %d: %v", index, result.Value)
		}
	}
	events := []agentEventSnapshot{
		{ExternalID: "provider-started", WorkID: first.ID, RunID: "run-a", Kind: "started", Title: "Provider started", Detail: "durable provider event", CreatedAt: at},
		{ExternalID: "question-current", WorkID: first.ID, RunID: "run-a", Kind: "question", Title: "Agent question", Detail: "durable provider question", CreatedAt: at.Add(time.Second)},
	}
	for sequence := int64(1); sequence <= 205; sequence++ {
		createdAt := at.Add(time.Duration(500-sequence) * time.Second)
		stream := "stdout"
		if sequence%2 == 0 {
			stream = "stderr"
		}
		events = append(events, agentEventSnapshot{
			ExternalID: core.Sprintf("chunk-%03d", sequence), WorkID: first.ID, RunID: "run-a",
			Sequence: sequence, Stream: stream, Kind: "log." + stream,
			Detail: core.Sprintf("chunk %03d", sequence), CreatedAt: createdAt,
		})
	}
	events = append(events, agentEventSnapshot{
		ExternalID: "work-b-only", WorkID: second.ID, RunID: "run-b", Sequence: 1,
		Stream: "stdout", Kind: "log.stdout", Detail: "second-work-only", CreatedAt: at,
	})
	if result := panel.ApplyAgentSnapshot(agentSnapshot{
		Work: []agentWorkSnapshot{
			{ExternalID: first.ID, NativeRunID: "run-a", Status: "running"},
			{ExternalID: second.ID, NativeRunID: "run-b", Status: "running"},
		},
		Events: events,
	}); !result.OK {
		t.Fatalf("ApplyAgentSnapshot: %v", result.Value)
	}
	merged := panel.Events(first.ID)
	logs := make([]eventRecord, 0, maxRenderedAgentEvents)
	providerEvents := make([]eventRecord, 0, 2)
	local := make([]eventRecord, 0, 3)
	for _, event := range merged {
		switch {
		case event.Status != "live":
			local = append(local, event)
		case strings.HasPrefix(event.Kind, "log."):
			logs = append(logs, event)
		default:
			providerEvents = append(providerEvents, event)
		}
	}
	if len(logs) != maxRenderedAgentEvents || len(providerEvents) != 2 || len(local) != 3 {
		t.Fatalf("bounded timeline = %d logs / %d provider / %d local, want %d / 2 / 3", len(logs), len(providerEvents), len(local), maxRenderedAgentEvents)
	}
	for index, event := range logs {
		want := core.Sprintf("chunk %03d", index+6)
		if event.Detail != want {
			t.Fatalf("logs[%d] = %q, want %q", index, event.Detail, want)
		}
	}
	if providerEvents[0].Detail != "durable provider event" || providerEvents[1].Detail != "durable provider question" {
		t.Fatalf("durable provider events = %#v", providerEvents)
	}
	if events := panel.Events(second.ID); len(events) != 1 || events[0].Detail != "second-work-only" {
		t.Fatalf("second timeline = %#v", events)
	}
	panel.selectWork(first.ID)
	firstView := panel.View(120, 24, newUIStyles(midnightTheme()))
	panel.selectWork(second.ID)
	secondView := panel.View(120, 24, newUIStyles(midnightTheme()))
	if strings.Contains(firstView, "second-work-only") || strings.Contains(secondView, "chunk 205") || !strings.Contains(secondView, "second-work-only") {
		t.Fatalf("switched timelines leaked:\nFIRST\n%s\nSECOND\n%s", firstView, secondView)
	}
	stored := repository.Events(workEventSessionID(first)).Value.([]eventRecord)
	if len(stored) != 3 {
		t.Fatalf("provider logs copied into local events: %#v", stored)
	}
}

func TestWorkPanel_Good(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	now := time.Date(2026, time.July, 17, 17, 0, 0, 0, time.UTC)
	opened := newWorkPanel(repository, newUnavailableAgentProvider(defaultAgentUnavailableReason), sequenceIDs("work-local"), func() time.Time { return now })
	if !opened.OK {
		t.Fatalf("newWorkPanel: %v", opened.Value)
	}
	panel := opened.Value.(*workPanel)
	created := panel.CreateWork("Draft release notes", "Prepare the release notes for review.", "/tmp/release-notes")
	if !created.OK {
		t.Fatalf("Create: %v", created.Value)
	}
	record := created.Value.(workItemRecord)
	if record.Status != workStatusActive || record.Source != "local" {
		t.Fatalf("created work = %#v", record)
	}
	if result := panel.Rename(record.ID, "Ship release notes"); !result.OK {
		t.Fatalf("Rename: %v", result.Value)
	}
	if result := panel.Complete(record.ID); !result.OK {
		t.Fatalf("Complete: %v", result.Value)
	}
	if result := panel.Reopen(record.ID); !result.OK {
		t.Fatalf("Reopen: %v", result.Value)
	}
	if result := panel.Link(record.ID, "session-42"); !result.OK {
		t.Fatalf("Link: %v", result.Value)
	}
	if result := panel.Archive(record.ID); !result.OK {
		t.Fatalf("Archive: %v", result.Value)
	}
	assertRecordSliceLength[workItemRecord](t, "active work", repository.ListWorkItems(false), 0)
	allResult := repository.ListWorkItems(true)
	assertRecordSliceLength[workItemRecord](t, "all work", allResult, 1)
	stored := allResult.Value.([]workItemRecord)[0]
	if stored.Title != "Ship release notes" || stored.Task != "Prepare the release notes for review." || stored.Repo != "/tmp/release-notes" || stored.Status != workStatusActive || stored.SessionID != "session-42" || !stored.Archived {
		t.Fatalf("stored work = %#v", stored)
	}
}

func TestWorkEditor_PersistsTitleTaskAndRepository(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	now := time.Date(2026, time.July, 18, 10, 0, 0, 0, time.UTC)
	opened := newWorkPanel(repository, newUnavailableAgentProvider(defaultAgentUnavailableReason), sequenceIDs("work-editor"), func() time.Time { return now })
	if !opened.OK {
		t.Fatalf("newWorkPanel: %v", opened.Value)
	}
	panel := opened.Value.(*workPanel)
	created := panel.CreateWork("Ship agent review", "Exercise the complete launch flow.", "/tmp/repositories/with spaces")
	if !created.OK {
		t.Fatalf("CreateWork: %v", created.Value)
	}
	record := created.Value.(workItemRecord)
	if record.Title != "Ship agent review" || record.Task != "Exercise the complete launch flow." || record.Repo != "/tmp/repositories/with spaces" {
		t.Fatalf("created work = %#v", record)
	}
	edited := panel.EditWork(record.ID, "Ship reviewed agent flow", "Keep the source untouched.", "/tmp/other repository")
	if !edited.OK {
		t.Fatalf("EditWork: %v", edited.Value)
	}
	stored := repository.ListWorkItems(false).Value.([]workItemRecord)[0]
	if stored.Title != "Ship reviewed agent flow" || stored.Task != "Keep the source untouched." || stored.Repo != "/tmp/other repository" {
		t.Fatalf("stored work = %#v", stored)
	}
	for _, args := range [][3]string{{"", "task", "/tmp/repo"}, {"title", "", "/tmp/repo"}, {"title", "task", ""}} {
		if result := panel.CreateWork(args[0], args[1], args[2]); result.OK {
			t.Fatalf("CreateWork%q succeeded", args)
		}
	}
}

func TestWorkPanel_Bad(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	reason := "go/agent provider is not installed"
	provider := &fixtureAgentProvider{caps: agentFeatureCatalog(reason)}
	opened := newWorkPanel(repository, provider, sequenceIDs("unused"), time.Now)
	if !opened.OK {
		t.Fatalf("newWorkPanel: %v", opened.Value)
	}
	panel := opened.Value.(*workPanel)
	before := repository.ListWorkItems(true)
	for _, capability := range provider.Capabilities() {
		if !panel.SelectAction(capability.Feature) {
			t.Fatalf("SelectAction(%q) = false", capability.Feature)
		}
		selected := panel.SelectedAction()
		if selected.Available || !strings.Contains(selected.Reason, reason) {
			t.Fatalf("disabled %q = %#v", capability.Feature, selected)
		}
	}
	after := repository.ListWorkItems(true)
	if provider.runs != 0 || len(before.Value.([]workItemRecord)) != len(after.Value.([]workItemRecord)) {
		t.Fatalf("disabled actions mutated state: runs=%d before=%#v after=%#v", provider.runs, before.Value, after.Value)
	}
}

func TestWorkPanel_Ugly(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	now := time.Date(2026, time.July, 17, 18, 0, 0, 0, time.UTC)
	opened := newWorkPanel(repository, newUnavailableAgentProvider(defaultAgentUnavailableReason), sequenceIDs("unused"), func() time.Time { return now })
	if !opened.OK {
		t.Fatalf("newWorkPanel: %v", opened.Value)
	}
	panel := opened.Value.(*workPanel)
	empty := panel.View(72, 18, newUIStyles(midnightTheme()))
	compact := strings.Join(strings.Fields(ansi.Strip(empty)), " ")
	wantEmpty := "A connected provider or workspace action will create work here. Open the inspector and dispatch one."
	if !strings.Contains(compact, "○ No work yet") || !strings.Contains(compact, wantEmpty) || strings.Contains(compact, "command palette") {
		t.Fatalf("empty work view:\n%s", empty)
	}
	fixtures := []workItemRecord{
		testWorkRecord("active", "Local task", workStatusActive, now),
		testWorkRecord("waiting", "Needs decision", workStatusWaiting, now.Add(-time.Minute)),
		testWorkRecord("failed", "Broken check", workStatusFailed, now.Add(-2*time.Minute)),
		testWorkRecord("completed", "Finished task", workStatusCompleted, now.Add(-3*time.Minute)),
	}
	fixtures[1].Question = "Which release target?"
	for _, fixture := range fixtures {
		if result := repository.SaveWorkItem(fixture); !result.OK {
			t.Fatalf("save fixture: %v", result.Value)
		}
	}
	if result := panel.RefreshLocal(); !result.OK {
		t.Fatalf("RefreshLocal: %v", result.Value)
	}
	for _, width := range []int{72, 132} {
		view := panel.View(width, 22, newUIStyles(midnightTheme()))
		for _, want := range []string{"ACTIVE", "WAITING", "DONE", "Local task", "Needs decision", "Broken check", "Finished task"} {
			if !strings.Contains(view, want) {
				t.Fatalf("width %d missing %q:\n%s", width, want, view)
			}
		}
		for line, text := range strings.Split(view, "\n") {
			if got := lipgloss.Width(text); got > width {
				t.Fatalf("width %d line %d overflows at %d", width, line, got)
			}
		}
		for _, want := range []string{"source", "TIMELINE"} {
			if !strings.Contains(view, want) {
				t.Fatalf("width %d cropped detail %q:\n%s", width, want, view)
			}
		}
	}
}

func TestWorkPanel_PreservesAgentExecutionStatuses(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	provider := &fixtureAgentProvider{snapshot: agentSnapshot{Work: []agentWorkSnapshot{
		{ExternalID: "queued", Title: "Queued", Status: "queued"},
		{ExternalID: "running", Title: "Running", Status: "running"},
		{ExternalID: "interrupted", Title: "Interrupted", Status: "interrupted"},
	}}}
	opened := newWorkPanel(repository, provider, sequenceIDs("queued-id", "running-id", "interrupted-id"), time.Now)
	if !opened.OK {
		t.Fatalf("newWorkPanel: %v", opened.Value)
	}
	panel := opened.Value.(*workPanel)
	if result := panel.Refresh(context.Background()); !result.OK {
		t.Fatalf("Refresh: %v", result.Value)
	}
	if items := panel.Items(); len(items) != 0 {
		t.Fatalf("provider state persisted as Work rows: %#v", items)
	}
}

func TestAgentCommandPalette_Good(t *testing.T) {
	reason := defaultAgentUnavailableReason
	palette := newCommandPalette(newUIStyles(midnightTheme()))
	for _, capability := range agentFeatureCatalog(reason) {
		id := agentCommandID(capability.Feature)
		command, exists := palette.byID[id]
		if !exists || command.Available || command.Reason != reason {
			t.Fatalf("agent command %q = %#v, exists=%v", id, command, exists)
		}
		matches := palette.Filter(string(capability.Feature))
		found := false
		for _, match := range matches {
			found = found || match.ID == id
		}
		if !found {
			t.Fatalf("agent command %q is not searchable", id)
		}
	}

	a := newApp("", 0, 64)
	a.activePanel = panelWork
	view := a.inspector.View(a, 48, 80)
	for _, want := range []string{"EXECUTION", "QUEUE + SETUP", "PLANS + SESSIONS", "SCAN + MONITOR", "BRAIN + MESSAGES", "FLEET + FORGE", "QA + REVIEW + PR", "agent capability not installed", "provider has not been connected"} {
		if !strings.Contains(view, want) {
			t.Fatalf("Work inspector missing %q:\n%s", want, view)
		}
	}
}

func testWorkRecord(id, title, status string, updatedAt time.Time) workItemRecord {
	return workItemRecord{
		ID:         id,
		ExternalID: "local:" + id,
		Source:     "local",
		Title:      title,
		Status:     status,
		StartedAt:  updatedAt,
		UpdatedAt:  updatedAt,
		ArchivedAt: unsetRecordTime(),
	}
}
