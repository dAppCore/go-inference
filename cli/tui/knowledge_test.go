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

func TestKnowledgeDiscover_Good(t *testing.T) {
	local := coreio.NewMemoryMedium()
	additional := coreio.NewMemoryMedium()
	writeKnowledgeFixture(t, local, "packs/zeta.md", "# Zeta\n\nLast document.")
	writeKnowledgeFixture(t, local, "packs/nested/guide.markdown", "No heading here.\n")
	writeKnowledgeFixture(t, additional, "alpha.md", "# Alpha\n\nFirst document.")

	result := newKnowledgeScanner().Discover([]knowledgeMount{
		{Name: "local", Root: "packs", Medium: local},
		{Name: "additional", Root: "", Medium: additional},
	}, 1024)
	if !result.OK {
		t.Fatalf("Discover: %v", result.Value)
	}
	discovery := result.Value.(knowledgeDiscovery)
	if len(discovery.Warnings) != 0 || len(discovery.Documents) != 3 {
		t.Fatalf("discovery = %#v", discovery)
	}
	want := []struct{ mount, title, path string }{
		{"additional", "Alpha", "alpha.md"},
		{"local", "guide", "packs/nested/guide.markdown"},
		{"local", "Zeta", "packs/zeta.md"},
	}
	for index, expected := range want {
		document := discovery.Documents[index]
		if document.Mount != expected.mount || document.Title != expected.title || document.Path != expected.path {
			t.Fatalf("document %d = %#v, want %#v", index, document, expected)
		}
		if document.ContentHash != core.SHA256HexString(document.Content) {
			t.Fatalf("document %d hash = %q", index, document.ContentHash)
		}
	}
}

func TestKnowledgeDiscover_Bad(t *testing.T) {
	medium := coreio.NewMemoryMedium()
	writeKnowledgeFixture(t, medium, "packs/small.md", "# Small\nvalid")
	writeKnowledgeFixture(t, medium, "packs/oversized.md", "# Huge\n"+strings.Repeat("x", 256))
	result := newKnowledgeScanner().Discover([]knowledgeMount{{Name: "local", Root: "packs", Medium: medium}}, 32)
	if !result.OK {
		t.Fatalf("Discover should retain valid documents: %v", result.Value)
	}
	discovery := result.Value.(knowledgeDiscovery)
	if len(discovery.Documents) != 1 || discovery.Documents[0].Title != "Small" || len(discovery.Warnings) != 1 {
		t.Fatalf("partial discovery = %#v", discovery)
	}
	warning := discovery.Warnings[0]
	if warning.Path != "packs/oversized.md" || !strings.Contains(warning.Reason, "32 bytes") {
		t.Fatalf("oversized warning = %#v", warning)
	}
	a := newApp("", 0, 64)
	a.activePanel = panelChat
	a.inspector.ApplyKnowledge(result)
	view := a.inspector.View(a, 72, 40)
	if !strings.Contains(view, "KNOWLEDGE") || !strings.Contains(view, "oversized.md") || !strings.Contains(view, "32 bytes") {
		t.Fatalf("visible knowledge warning:\n%s", view)
	}
}

func TestKnowledgeDiscover_Ugly(t *testing.T) {
	base := coreio.NewMemoryMedium()
	writeKnowledgeFixture(t, base, "packs/real.md", "# Real\n")
	writeKnowledgeFixture(t, base, "packs/unreadable.md", "# Secret\n")
	writeKnowledgeFixture(t, base, "packs/ignore.txt", "not markdown")
	medium := &fixtureKnowledgeMedium{
		Medium:     base,
		unreadable: map[string]bool{"packs/unreadable.md": true},
		reads:      map[string]int{},
		lists:      map[string]int{},
	}
	mount := knowledgeMount{Name: "local", Root: "packs", Medium: medium}
	result := newKnowledgeScanner().Discover([]knowledgeMount{mount, mount}, 1024)
	if !result.OK {
		t.Fatalf("Discover ugly: %v", result.Value)
	}
	discovery := result.Value.(knowledgeDiscovery)
	if len(discovery.Documents) != 1 || discovery.Documents[0].Title != "Real" || len(discovery.Warnings) != 1 {
		t.Fatalf("ugly discovery = %#v", discovery)
	}
	if medium.reads["packs/real.md"] != 1 || medium.lists["packs/loop"] != 0 {
		t.Fatalf("duplicate/symlink traversal: reads=%#v lists=%#v", medium.reads, medium.lists)
	}
}

func TestKnowledgeAttachment_Good(t *testing.T) {
	databasePath := t.TempDir() + "/lem.duckdb"
	repositoryResult := openDuckRepository(databasePath)
	if !repositoryResult.OK {
		t.Fatalf("open repository: %v", repositoryResult.Value)
	}
	repository := repositoryResult.Value.(workspaceRepository)
	createdAt := time.Date(2026, time.July, 17, 19, 0, 0, 0, time.UTC)
	document := knowledgeDocument{
		Mount: "local", Path: "packs/guide.md", Title: "Guide",
		Content: "# Guide\n\nUse the durable snapshot.", ContentHash: core.SHA256HexString("# Guide\n\nUse the durable snapshot."),
	}
	opened := newKnowledgeLibrary(repository, 4096, sequenceIDs("attachment-1"), func() time.Time { return createdAt })
	if !opened.OK {
		t.Fatalf("newKnowledgeLibrary: %v", opened.Value)
	}
	library := opened.Value.(*knowledgeLibrary)
	attached := library.Attach("session-knowledge", document)
	if !attached.OK {
		t.Fatalf("Attach: %v", attached.Value)
	}
	attachments := library.Attachments("session-knowledge")
	if !attachments.OK {
		t.Fatalf("Attachments: %v", attachments.Value)
	}
	records := attachments.Value.([]attachmentRecord)
	message := knowledgeSystemMessage(records)
	if len(message) > knowledgeSystemMessageMaxBytes || !strings.Contains(message, "Guide") || !strings.Contains(message, document.Content) {
		t.Fatalf("knowledge system message = %q", message)
	}
	a := newApp("", 0, 64)
	a.knowledge = library
	a.attachments = records
	a.tools.setEnabled(true)
	history := a.history()
	if len(history) != 1 || history[0].Role != "system" {
		t.Fatalf("knowledge/tool history = %#v", history)
	}
	knowledgeAt := strings.Index(history[0].Content, "Local knowledge snapshots")
	toolsAt := strings.Index(history[0].Content, a.tools.declarations())
	if knowledgeAt < 0 || toolsAt <= knowledgeAt {
		t.Fatalf("system message ordering = %q", history[0].Content)
	}
	if result := repository.Close(); !result.OK {
		t.Fatalf("close repository: %v", result.Value)
	}

	reopened := openDuckRepository(databasePath)
	if !reopened.OK {
		t.Fatalf("reopen repository: %v", reopened.Value)
	}
	repository = reopened.Value.(workspaceRepository)
	defer closeTestDuckRepository(t, repository)
	restarted := newKnowledgeLibrary(repository, 4096, sequenceIDs("unused"), func() time.Time { return createdAt.Add(time.Hour) })
	if !restarted.OK {
		t.Fatalf("restart library: %v", restarted.Value)
	}
	restored := restarted.Value.(*knowledgeLibrary).Attachments("session-knowledge")
	if !restored.OK {
		t.Fatalf("restored attachments: %v", restored.Value)
	}
	stored := restored.Value.([]attachmentRecord)
	if len(stored) != 1 || stored[0].Snapshot != document.Content || knowledgeSystemMessage(stored) != message {
		t.Fatalf("restored snapshot = %#v", stored)
	}
	if result := restarted.Value.(*knowledgeLibrary).Detach("session-knowledge", stored[0].ID); !result.OK {
		t.Fatalf("Detach: %v", result.Value)
	}
	if active := restarted.Value.(*knowledgeLibrary).Attachments("session-knowledge"); !active.OK || len(active.Value.([]attachmentRecord)) != 0 {
		t.Fatalf("active attachments after detach = %#v", active.Value)
	}
}

func TestKnowledgeAttachmentStale_Good(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	medium := coreio.NewMemoryMedium()
	writeKnowledgeFixture(t, medium, "packs/context.md", "# Context\noriginal")
	scanner := newKnowledgeScanner()
	discovered := scanner.Discover([]knowledgeMount{{Name: "local", Root: "packs", Medium: medium}}, 1024)
	document := discovered.Value.(knowledgeDiscovery).Documents[0]
	now := time.Date(2026, time.July, 17, 20, 0, 0, 0, time.UTC)
	libraryResult := newKnowledgeLibrary(repository, 4096, sequenceIDs("attachment-stale"), func() time.Time { return now })
	if !libraryResult.OK {
		t.Fatalf("newKnowledgeLibrary: %v", libraryResult.Value)
	}
	library := libraryResult.Value.(*knowledgeLibrary)
	if result := library.Attach("session-stale", document); !result.OK {
		t.Fatalf("Attach: %v", result.Value)
	}
	original := document.Content
	writeKnowledgeFixture(t, medium, "packs/context.md", "# Context\nchanged")
	changed := scanner.Discover([]knowledgeMount{{Name: "local", Root: "packs", Medium: medium}}, 1024)
	if result := library.RefreshStaleness("session-stale", changed.Value.(knowledgeDiscovery).Documents); !result.OK {
		t.Fatalf("RefreshStaleness: %v", result.Value)
	}
	attachments := library.Attachments("session-stale").Value.([]attachmentRecord)
	if len(attachments) != 1 || !attachments[0].Stale || attachments[0].Snapshot != original {
		t.Fatalf("stale attachment = %#v", attachments)
	}
}

func writeKnowledgeFixture(t *testing.T, medium coreio.Medium, path, content string) {
	t.Helper()
	if err := medium.Write(path, content); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

type fixtureKnowledgeMedium struct {
	coreio.Medium
	unreadable map[string]bool
	reads      map[string]int
	lists      map[string]int
}

func (medium *fixtureKnowledgeMedium) Read(path string) (string, error) {
	medium.reads[path]++
	if medium.unreadable[path] {
		return "", errors.New("fixture unreadable")
	}
	return medium.Medium.Read(path)
}

func (medium *fixtureKnowledgeMedium) List(path string) ([]fs.DirEntry, error) {
	medium.lists[path]++
	entries, err := medium.Medium.List(path)
	if err != nil {
		return nil, err
	}
	if path == "packs" {
		info := coreio.NewFileInfo("loop", 0, core.ModeSymlink|0777, time.Time{}, false)
		entries = append(entries, coreio.NewDirEntry("loop", false, core.ModeSymlink|0777, info))
	}
	return entries, nil
}
