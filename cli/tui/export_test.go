// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"errors"
	"io/fs"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

func TestExportSessionMarkdown_Good(t *testing.T) {
	repository, session := exportFixture(t, "11111111-1111-4111-8111-111111111111", "A thoughtful session")
	defer closeTestDuckRepository(t, repository)
	medium := coreio.NewMemoryMedium()
	now := time.Date(2026, time.July, 17, 21, 30, 0, 0, time.UTC)
	exporter := newWorkspaceSessionExporter(repository, true, func() time.Time { return now }, newRecordID)
	result := exporter.Export(medium, "exports", session.ID, exportMarkdown)
	if !result.OK {
		t.Fatalf("Export Markdown: %v", result.Value)
	}
	receipt := result.Value.(exportReceipt)
	content, err := medium.Read(receipt.Path)
	if err != nil {
		t.Fatalf("read export %s: %v", receipt.Path, err)
	}
	for _, want := range []string{
		"A thoughtful session", "Visible answer", "private thought", "word_count",
		"2 words", "Knowledge snapshot", "Runner started", "https://example.test/pr/1",
	} {
		if !strings.Contains(content, want) {
			t.Fatalf("Markdown export missing %q:\n%s", want, content)
		}
	}
	withoutThoughts := newWorkspaceSessionExporter(repository, false, func() time.Time { return now.Add(time.Second) }, newRecordID)
	hidden := withoutThoughts.Export(medium, "exports", session.ID, exportMarkdown)
	if !hidden.OK {
		t.Fatalf("Export without thoughts: %v", hidden.Value)
	}
	hiddenContent, _ := medium.Read(hidden.Value.(exportReceipt).Path)
	if strings.Contains(hiddenContent, "private thought") {
		t.Fatalf("thought preference was ignored:\n%s", hiddenContent)
	}

	palette := newCommandPalette(newUIStyles(midnightTheme()))
	palette.SetExporter(exporter, medium, "exports")
	a := newApp("", 0, 64)
	a.palette = palette
	a.repository = repository
	a.sessionID = session.ID
	if invoked := palette.Invoke(commandExportMarkdown, &a); !invoked.OK {
		t.Fatalf("palette export: %v", invoked.Value)
	}
	if invoked := palette.Invoke(commandExportJSON, &a); !invoked.OK {
		t.Fatalf("palette JSON export: %v", invoked.Value)
	}
	artifacts := repository.Artifacts(session.ID)
	if !artifacts.OK || len(artifacts.Value.([]artifactRecord)) != 3 {
		t.Fatalf("export artifact persistence = %#v", artifacts.Value)
	}
}

func TestExportSessionJSON_Good(t *testing.T) {
	repository, session := exportFixture(t, "22222222-2222-4222-8222-222222222222", "Structured session")
	defer closeTestDuckRepository(t, repository)
	medium := coreio.NewMemoryMedium()
	exporter := newWorkspaceSessionExporter(repository, true, func() time.Time {
		return time.Date(2026, time.July, 17, 22, 0, 0, 0, time.UTC)
	}, newRecordID)
	result := exporter.Export(medium, "exports", session.ID, exportJSON)
	if !result.OK {
		t.Fatalf("Export JSON: %v", result.Value)
	}
	content, err := medium.Read(result.Value.(exportReceipt).Path)
	if err != nil {
		t.Fatalf("read JSON export: %v", err)
	}
	var decoded sessionExport
	if unmarshaled := core.JSONUnmarshalString(content, &decoded); !unmarshaled.OK {
		t.Fatalf("unmarshal JSON export: %v", unmarshaled.Value)
	}
	if decoded.Session.ID != session.ID || len(decoded.Turns) != 2 || len(decoded.Events) != 1 || len(decoded.Artifacts) != 1 || len(decoded.Attachments) != 1 {
		t.Fatalf("decoded export = %#v", decoded)
	}
}

func TestExportSessionSelectedMedium_Good(t *testing.T) {
	repository, session := exportFixture(t, "33333333-3333-4333-8333-333333333333", "Selected destination")
	defer closeTestDuckRepository(t, repository)
	defaultMedium := coreio.NewMemoryMedium()
	selectedMedium := coreio.NewMemoryMedium()
	exporter := newWorkspaceSessionExporter(repository, true, time.Now, newRecordID)
	result := exporter.Export(selectedMedium, "chosen", session.ID, exportMarkdown)
	if !result.OK {
		t.Fatalf("selected-medium export: %v", result.Value)
	}
	if !selectedMedium.Exists(result.Value.(exportReceipt).Path) {
		t.Fatal("selected destination did not receive export")
	}
	entries, err := defaultMedium.List("")
	if err != nil || len(entries) != 0 {
		t.Fatalf("default destination changed: entries=%#v err=%v", entries, err)
	}
}

func TestExportSession_Bad(t *testing.T) {
	repository, session := exportFixture(t, "44444444-4444-4444-8444-444444444444", "Read only destination")
	defer closeTestDuckRepository(t, repository)
	medium := &failingExportMedium{Medium: coreio.NewMemoryMedium(), reason: errors.New("read-only medium")}
	exporter := newWorkspaceSessionExporter(repository, true, time.Now, newRecordID)
	palette := newCommandPalette(newUIStyles(midnightTheme()))
	palette.SetExporter(exporter, medium, "exports")
	a := newApp("", 0, 64)
	a.palette = palette
	a.repository = repository
	a.sessionID = session.ID
	before := repository.Artifacts(session.ID).Value.([]artifactRecord)
	result := palette.Invoke(commandExportMarkdown, &a)
	if result.OK || !strings.Contains(result.Error(), "exports/") || !strings.Contains(result.Error(), "read-only medium") {
		t.Fatalf("failed export result = %#v", result)
	}
	after := repository.Artifacts(session.ID).Value.([]artifactRecord)
	if len(after) != len(before) {
		t.Fatalf("failed export mutated artifacts: before=%#v after=%#v", before, after)
	}
}

func TestExportSession_Ugly(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	now := time.Date(2026, time.July, 17, 23, 0, 0, 0, time.UTC)
	first := testSessionRecord("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "Same title", now)
	second := testSessionRecord("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", "Same title", now)
	for _, session := range []sessionRecord{first, second} {
		if result := repository.SaveSession(session); !result.OK {
			t.Fatalf("save session: %v", result.Value)
		}
	}
	medium := coreio.NewMemoryMedium()
	exporter := newWorkspaceSessionExporter(repository, true, func() time.Time { return now }, newRecordID)
	firstResult := exporter.Export(medium, "exports", first.ID, exportMarkdown)
	secondResult := exporter.Export(medium, "exports", second.ID, exportMarkdown)
	if !firstResult.OK || !secondResult.OK {
		t.Fatalf("collision exports: first=%#v second=%#v", firstResult.Value, secondResult.Value)
	}
	firstPath := firstResult.Value.(exportReceipt).Path
	secondPath := secondResult.Value.(exportReceipt).Path
	if firstPath == secondPath || !medium.Exists(firstPath) || !medium.Exists(secondPath) {
		t.Fatalf("collision paths = %q / %q", firstPath, secondPath)
	}
}

func exportFixture(t *testing.T, sessionID, title string) (workspaceRepository, sessionRecord) {
	t.Helper()
	repository := openTestDuckRepository(t)
	now := time.Date(2026, time.July, 17, 21, 0, 0, 0, time.UTC)
	session := testSessionRecord(sessionID, title, now)
	if result := repository.SaveSession(session); !result.OK {
		t.Fatalf("save export session: %v", result.Value)
	}
	turns := []turnRecord{
		testTurnRecord("turn-user-"+sessionID[:8], session.ID, 1, "user", "Question", now),
		{
			ID: "turn-assistant-" + sessionID[:8], SessionID: session.ID, Sequence: 2, Role: "assistant",
			Visible: "Visible answer", Thought: "private thought", ToolName: "word_count",
			ToolCallJSON: `{"name":"word_count","text":"two words"}`, ToolResultJSON: `{"result":"2 words"}`,
			Model: "fixture-model", CreatedAt: now.Add(time.Minute), UpdatedAt: now.Add(time.Minute),
		},
	}
	for _, turn := range turns {
		if result := repository.SaveTurn(turn); !result.OK {
			t.Fatalf("save export turn: %v", result.Value)
		}
	}
	event := eventRecord{
		ID: "event-" + sessionID[:8], SessionID: session.ID, Kind: "runner.started", Status: "completed",
		Title: "Runner started", Detail: "local", PayloadJSON: `{}`, CreatedAt: now.Add(2 * time.Minute),
	}
	if result := repository.SaveEvent(event); !result.OK {
		t.Fatalf("save export event: %v", result.Value)
	}
	artifact := artifactRecord{
		ID: "artifact-" + sessionID[:8], SessionID: session.ID, Kind: "pull_request",
		Path: "https://example.test/pr/1", Title: "Pull request", MetadataJSON: `{}`,
		CreatedAt: now.Add(3 * time.Minute), ArchivedAt: unsetRecordTime(),
	}
	if result := repository.SaveArtifact(artifact); !result.OK {
		t.Fatalf("save export artifact: %v", result.Value)
	}
	attachment := attachmentRecord{
		ID: "attachment-" + sessionID[:8], SessionID: session.ID, SourcePath: "local:packs/knowledge.md",
		Title: "Knowledge", ContentHash: core.SHA256HexString("Knowledge snapshot"), Snapshot: "Knowledge snapshot",
		AddedAt: now.Add(4 * time.Minute), LastCheckedAt: now.Add(4 * time.Minute), ArchivedAt: unsetRecordTime(),
	}
	if result := repository.SaveAttachment(attachment); !result.OK {
		t.Fatalf("save export attachment: %v", result.Value)
	}
	return repository, session
}

type failingExportMedium struct {
	coreio.Medium
	reason error
}

func (medium *failingExportMedium) WriteMode(string, string, fs.FileMode) error {
	return medium.reason
}
