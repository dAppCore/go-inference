// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"testing"

	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"
)

func TestKeyMap_Good(t *testing.T) {
	keys := newKeyMap()
	tests := []struct {
		name    string
		message tea.KeyMsg
		binding key.Binding
	}{
		{"new session", tea.KeyMsg{Type: tea.KeyCtrlN}, keys.NewSession},
		{"session switcher", tea.KeyMsg{Type: tea.KeyCtrlP}, keys.SwitchSession},
		{"previous session", tea.KeyMsg{Type: tea.KeyLeft, Alt: true}, keys.PreviousSession},
		{"next session", tea.KeyMsg{Type: tea.KeyRight, Alt: true}, keys.NextSession},
		{"command palette", tea.KeyMsg{Type: tea.KeyCtrlK}, keys.CommandPalette},
		{"inspector", tea.KeyMsg{Type: tea.KeyCtrlO}, keys.ToggleInspector},
		{"search", tea.KeyMsg{Type: tea.KeyCtrlF}, keys.Search},
		{"save", tea.KeyMsg{Type: tea.KeyCtrlS}, keys.Save},
		{"settings", tea.KeyMsg{Type: tea.KeyF2}, keys.Settings},
		{"help", tea.KeyMsg{Type: tea.KeyF1}, keys.Help},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if !key.Matches(test.message, test.binding) {
				t.Fatalf("%q did not match %q", test.message.String(), test.binding.Keys())
			}
		})
	}
}
