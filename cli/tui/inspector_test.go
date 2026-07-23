// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"

	"dappco.re/go/inference/decode/parser"
	coreio "dappco.re/go/io"
	tea "dappco.re/go/render/display/tui"
	"dappco.re/go/render/display/tui/list"
	"github.com/charmbracelet/lipgloss"
)

func TestInspector_Good(t *testing.T) {
	a := newApp("", 0, 64)
	a.modelName = "gemma-4"
	a.generating = true
	a.svc.running = true
	a.svc.requests.Store(7)
	a.picker.SetItems([]list.Item{modelItem{path: "/models/qwen", name: "qwen", modelType: "qwen3"}})

	tests := []struct {
		panel panelID
		want  []string
	}{
		{panelChat, []string{"SESSION", "MODEL", "GENERATION", "● generating", "SETTINGS", "MODE", "TOOLS", "gemma-4"}},
		{panelWork, []string{"WORK DETAIL", "RUNTIME", "AGENT CAPABILITY", "not installed"}},
		{panelModels, []string{"MODEL DETAIL", "qwen", "/models/qwen", "LOADED", "○ none"}},
		{panelService, []string{"ADDRESS", "REQUESTS", "STATE", a.svc.addr(), "7", "● listening"}},
	}
	for _, test := range tests {
		a.activePanel = test.panel
		view := a.inspector.View(a, 36, 20)
		for _, want := range test.want {
			if !strings.Contains(view, want) {
				t.Fatalf("panel %d inspector missing %q:\n%s", test.panel, want, view)
			}
		}
	}

	a.activePanel = panelModels
	a.modelName = "qwen"
	if view := a.inspector.View(a, 36, 20); !strings.Contains(view, "● loaded") {
		t.Fatalf("selected loaded model must use the canonical loaded receipt:\n%s", view)
	}
}

func TestInspector_Bad(t *testing.T) {
	a := newApp("", 0, 64)
	m, _ := a.Update(tea.WindowSizeMsg{Width: 100, Height: 24})
	a = m.(app)
	a.activePanel = panelWork
	a.inspectorOpen = true
	view := a.View().Content
	if !strings.Contains(view, "WORK") || !strings.Contains(view, "AGENT CAPABILITY") {
		t.Fatalf("overlay inspector did not retain main Work panel:\n%s", view)
	}
}

func TestInspector_Ugly(t *testing.T) {
	a := newApp("", 0, 64)
	m, _ := a.Update(tea.WindowSizeMsg{Width: 72, Height: 22})
	a = m.(app)
	a.activePanel = panelChat
	a.inspectorOpen = true
	view := a.View().Content
	for _, section := range []string{"SESSION", "SETTINGS", "MODE", "TOOLS"} {
		if !strings.Contains(view, section) {
			t.Fatalf("narrow inspector missing %q:\n%s", section, view)
		}
	}
	for line, text := range strings.Split(view, "\n") {
		if width := lipgloss.Width(text); width > 72 {
			t.Fatalf("narrow line %d width = %d", line, width)
		}
	}
	_ = a.inspector.View(a, 0, 0) // zero dimensions are a valid compact boundary
}

func TestInspectorPreferences_Good(t *testing.T) {
	medium := coreio.NewMockMedium()
	opened := openPreferences(medium, appConfigPath)
	if !opened.OK {
		t.Fatalf("open preferences: %v", opened.Value)
	}
	preferences := opened.Value.(preferenceStore)
	a := newApp("", 0, 64)
	a.attachPreferences(preferences)
	if !a.inspector.Select(inspectorControlMaxTokens) {
		t.Fatal("max-tokens control unavailable")
	}
	if result := a.inspector.Adjust(&a, 1); !result.OK {
		t.Fatalf("adjust max tokens: %v", result.Value)
	}
	if !a.inspector.Select(inspectorControlTheme) {
		t.Fatal("theme control unavailable")
	}
	if result := a.inspector.Adjust(&a, 1); !result.OK {
		t.Fatalf("adjust theme: %v", result.Value)
	}
	if !a.inspector.Dirty() || a.cfg.maxTokens() != 8192 || a.inspector.Theme() != "aurora" {
		t.Fatalf("edited inspector: dirty=%v max=%d theme=%q", a.inspector.Dirty(), a.cfg.maxTokens(), a.inspector.Theme())
	}

	m, _ := a.Update(testModifiedKeyPress('s', tea.ModCtrl))
	a = m.(app)
	if a.inspector.Dirty() {
		t.Fatal("Ctrl+S left inspector dirty")
	}
	if a.styles.theme.name != "aurora" || a.markdown.theme != "aurora" {
		t.Fatalf("theme was not rebuilt in place: styles=%q markdown=%q", a.styles.theme.name, a.markdown.theme)
	}

	reopened := openPreferences(medium, appConfigPath)
	if !reopened.OK {
		t.Fatalf("reopen preferences: %v", reopened.Value)
	}
	values := reopened.Value.(preferenceStore).Values()
	if values.MaxTokens != 8192 || values.Theme != "aurora" {
		t.Fatalf("persisted inspector values = %#v", values)
	}
}

func TestInspectorTools_Bad(t *testing.T) {
	manager := openTestSessionManager(t, sequenceIDs("session-tools"))
	created := manager.Create()
	if !created.OK {
		t.Fatalf("create tools session: %v", created.Value)
	}
	a := newApp("", 0, 64)
	a.sessions = manager
	a.repository = manager.repository
	a.activateManagedSession(created.Value.(*chatSession))
	if !a.inspector.Select(inspectorControlTools) {
		t.Fatal("tools control unavailable")
	}
	if result := a.inspector.Adjust(&a, 1); !result.OK || !a.tools.enabled {
		t.Fatalf("enable tools = %#v enabled=%v", result.Value, a.tools.enabled)
	}

	malformed := parser.ToolCallOpenMarker + "broken payload" + parser.ToolCallCloseMarker
	a.turns = append(a.turns, turn{id: "assistant-malformed", role: "assistant", text: malformed})
	if command := a.runToolLoop(); command != nil {
		t.Fatal("malformed tool call unexpectedly auto-continued")
	}
	last := a.turns[len(a.turns)-1]
	if last.role != "tool" || !strings.Contains(last.text, "malformed tool call") {
		t.Fatalf("explicit malformed tool result = %#v", last)
	}
	events := manager.repository.Events(manager.Active().Record.ID)
	if !events.OK {
		t.Fatalf("load tool events: %v", events.Value)
	}
	stored := events.Value.([]eventRecord)
	if len(stored) != 1 || stored[0].Kind != "tool.parse" || stored[0].Status != "failed" {
		t.Fatalf("stored malformed tool event = %#v", stored)
	}
}
