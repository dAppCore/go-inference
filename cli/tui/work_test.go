// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/lipgloss"
)

func TestWorkPanel_Good(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	now := time.Date(2026, time.July, 17, 17, 0, 0, 0, time.UTC)
	opened := newWorkPanel(repository, newUnavailableAgentProvider(defaultAgentUnavailableReason), sequenceIDs("work-local"), func() time.Time { return now })
	if !opened.OK {
		t.Fatalf("newWorkPanel: %v", opened.Value)
	}
	panel := opened.Value.(*workPanel)
	created := panel.Create("Draft release notes")
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
	if stored.Title != "Ship release notes" || stored.Status != workStatusActive || stored.SessionID != "session-42" || !stored.Archived {
		t.Fatalf("stored work = %#v", stored)
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
		result := panel.ActivateSelectedAction(context.Background(), "")
		if result.OK || !strings.Contains(result.Error(), reason) {
			t.Fatalf("disabled %q result = %#v", capability.Feature, result)
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
	if !strings.Contains(empty, "No work yet") || !strings.Contains(empty, "workspace action") || strings.Contains(empty, "command palette") {
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
