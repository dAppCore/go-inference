// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

func TestCommandPalette_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	palette := newCommandPalette(styles)
	matches := palette.Filter("models panel")
	if len(matches) == 0 || matches[0].ID != commandPanelModels {
		t.Fatalf("models filter = %#v", matches)
	}
	a := newApp("", 0, 64)
	a.activePanel = panelWork
	if result := palette.Invoke(commandPanelModels, &a); !result.OK {
		t.Fatalf("Invoke models: %v", result.Value)
	}
	if a.activePanel != panelModels {
		t.Fatalf("active panel = %d, want Models", a.activePanel)
	}
}

func TestCommandPalette_Bad(t *testing.T) {
	palette := newCommandPalette(newUIStyles(midnightTheme()))
	a := newApp("", 0, 64)
	a.activePanel = panelService
	before := a.activePanel
	result := palette.Invoke(commandID("missing.command"), &a)
	if result.OK {
		t.Fatal("unknown command invocation succeeded")
	}
	if a.activePanel != before || a.inspectorOpen {
		t.Fatalf("unknown command mutated app: panel=%d inspector=%v", a.activePanel, a.inspectorOpen)
	}
}

func TestCommandPaletteLocalRefresh_Bad(t *testing.T) {
	palette := newCommandPalette(newUIStyles(midnightTheme()))
	a := newApp("", 0, 64)
	for _, id := range []commandID{commandRefreshRuntimes, commandRefreshKnowledge} {
		command := palette.byID[id]
		if command.Available || !strings.Contains(command.Reason, "restart") {
			t.Fatalf("refresh command %q = %#v", id, command)
		}
		if result := palette.Invoke(id, &a); result.OK {
			t.Fatalf("unwired refresh command %q reported success", id)
		}
	}
	if command := palette.byID[commandExportJSON]; command.Title != "Export JSON" || strings.Contains(command.Description, "JSON Lines") {
		t.Fatalf("JSON export command = %#v", command)
	}
}

func TestSessionSwitcher_Good(t *testing.T) {
	manager := openTestSessionManager(t, sequenceIDs("session-one", "session-two"))
	first := manager.Create().Value.(*chatSession)
	second := manager.Create().Value.(*chatSession)
	first.Record.Status = "idle"
	first.Record.PreferredModel = "gemma-4"
	second.Record.Status = "generating"
	second.Record.PreferredModel = "qwen-3"
	second.ActiveJobID = "job-hidden"
	second.Attention = true
	if result := manager.repository.SaveSession(first.Record); !result.OK {
		t.Fatalf("save first metadata: %v", result.Value)
	}
	if result := manager.repository.SaveSession(second.Record); !result.OK {
		t.Fatalf("save second metadata: %v", result.Value)
	}
	if result := manager.Switch(first.Record.ID); !result.OK {
		t.Fatalf("switch first: %v", result.Value)
	}
	second.Attention = true

	switcherResult := newSessionSwitcher(manager, newUIStyles(midnightTheme()), 72, 14)
	if !switcherResult.OK {
		t.Fatalf("newSessionSwitcher: %v", switcherResult.Value)
	}
	switcher := switcherResult.Value.(*sessionSwitcher)
	items := switcher.Items()
	if len(items) != 2 || items[0].SessionID != first.Record.ID || items[1].SessionID != second.Record.ID {
		t.Fatalf("recent switcher order = %#v", items)
	}
	if !strings.Contains(items[1].Title, "!") || !strings.Contains(items[1].Description, "generating") || !strings.Contains(items[1].Description, "qwen-3") {
		t.Fatalf("hidden session metadata = %#v", items[1])
	}

	switcher.list.Select(0)
	switcher.Update(tea.KeyMsg{Type: tea.KeyDown})
	if result := switcher.ActivateSelected(); !result.OK {
		t.Fatalf("activate selected: %v", result.Value)
	}
	if manager.Active().Record.ID != second.Record.ID {
		t.Fatalf("active session = %q, want %q", manager.Active().Record.ID, second.Record.ID)
	}
	if second.ActiveJobID != "job-hidden" {
		t.Fatalf("switch cancelled hidden job: %q", second.ActiveJobID)
	}
}

func TestHistorySearch_Good(t *testing.T) {
	manager := openTestSessionManager(t, sequenceIDs("session-match", "session-other"))
	matched := manager.Create().Value.(*chatSession)
	first := testTurnRecord("turn-before", matched.Record.ID, 1, "user", "ordinary opening", time.Now().UTC())
	second := testTurnRecord("turn-match", matched.Record.ID, 2, "assistant", "the durable needle is here", time.Now().UTC())
	if result := manager.AddTurn(first); !result.OK {
		t.Fatalf("add first turn: %v", result.Value)
	}
	if result := manager.AddTurn(second); !result.OK {
		t.Fatalf("add matching turn: %v", result.Value)
	}
	other := manager.Create().Value.(*chatSession)
	if manager.Active().Record.ID != other.Record.ID {
		t.Fatal("search fixture did not leave another session active")
	}

	searchResult := newHistorySearch(manager.repository, manager, newUIStyles(midnightTheme()), 72, 14)
	if !searchResult.OK {
		t.Fatalf("newHistorySearch: %v", searchResult.Value)
	}
	search := searchResult.Value.(*historySearch)
	if result := search.Search("durable needle"); !result.OK {
		t.Fatalf("Search: %v", result.Value)
	}
	if len(search.Hits()) != 1 || search.Hits()[0].Session.ID != matched.Record.ID {
		t.Fatalf("search hits = %#v", search.Hits())
	}
	if result := search.ActivateSelected(); !result.OK {
		t.Fatalf("activate search hit: %v", result.Value)
	}
	if manager.Active().Record.ID != matched.Record.ID {
		t.Fatalf("active after hit = %q", manager.Active().Record.ID)
	}
	if manager.Active().ViewportOffset != 1 || manager.Active().Follow {
		t.Fatalf("matched viewport = offset %d follow %v, want 1/false", manager.Active().ViewportOffset, manager.Active().Follow)
	}
	if search.MatchTurnID() != second.ID {
		t.Fatalf("match turn = %q, want %q", search.MatchTurnID(), second.ID)
	}
}

func TestOverlayRouting_Ugly(t *testing.T) {
	a := newApp("", 0, 64)
	m, _ := a.Update(tea.WindowSizeMsg{Width: 100, Height: 24})
	a = m.(app)
	a.activePanel = panelService
	a.generating = true

	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyCtrlK})
	a = m.(app)
	if a.activeOverlay != overlayCommands {
		t.Fatalf("ctrl+k overlay = %d, want commands", a.activeOverlay)
	}
	a.palette.list.SetFilterText("models panel")
	beforeAddress := a.svc.addrIdx
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyDown})
	a = m.(app)
	if a.activePanel != panelService || a.svc.addrIdx != beforeAddress {
		t.Fatal("overlay arrow leaked into the Service panel")
	}
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if a.activePanel != panelModels || a.svc.running || a.activeOverlay != overlayNone {
		t.Fatalf("overlay Enter: panel=%d service=%v overlay=%d", a.activePanel, a.svc.running, a.activeOverlay)
	}

	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyCtrlK})
	a = m.(app)
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyEsc})
	a = m.(app)
	if a.activeOverlay != overlayNone || !a.generating {
		t.Fatalf("overlay Escape: overlay=%d generating=%v", a.activeOverlay, a.generating)
	}
}
