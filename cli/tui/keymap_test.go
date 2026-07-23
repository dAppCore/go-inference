// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"testing"

	tea "dappco.re/go/render/display/tui"
	"dappco.re/go/render/display/tui/key"
)

func TestKeyMap_Good(t *testing.T) {
	keys := newKeyMap()
	tests := []struct {
		name    string
		message tea.KeyPressMsg
		binding key.Binding
	}{
		{"new session", testModifiedKeyPress('n', tea.ModCtrl), keys.NewSession},
		{"session switcher", testModifiedKeyPress('p', tea.ModCtrl), keys.SwitchSession},
		{"previous session", testModifiedKeyPress(tea.KeyLeft, tea.ModAlt), keys.PreviousSession},
		{"next session", testModifiedKeyPress(tea.KeyRight, tea.ModAlt), keys.NextSession},
		{"command palette", testModifiedKeyPress('k', tea.ModCtrl), keys.CommandPalette},
		{"inspector", testModifiedKeyPress('o', tea.ModCtrl), keys.ToggleInspector},
		{"search", testModifiedKeyPress('f', tea.ModCtrl), keys.Search},
		{"save", testModifiedKeyPress('s', tea.ModCtrl), keys.Save},
		{"settings", testKeyPress(tea.KeyF2), keys.Settings},
		{"help", testKeyPress(tea.KeyF1), keys.Help},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if !key.Matches(test.message, test.binding) {
				t.Fatalf("%q did not match %q", test.message.String(), test.binding.Keys())
			}
		})
	}
}

func testKeyPress(code rune) tea.KeyPressMsg {
	return tea.KeyPressMsg{Code: code}
}

func testModifiedKeyPress(code rune, mod tea.KeyMod) tea.KeyPressMsg {
	return tea.KeyPressMsg{Code: code, Mod: mod}
}

func testTextPress(code rune) tea.KeyPressMsg {
	return tea.KeyPressMsg{Code: code, Text: string(code)}
}
