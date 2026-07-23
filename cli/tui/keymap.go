// SPDX-Licence-Identifier: EUPL-1.2

package tui

import "dappco.re/go/html/tui/key"

type keyMap struct {
	NewSession      key.Binding
	SwitchSession   key.Binding
	PreviousSession key.Binding
	NextSession     key.Binding
	CommandPalette  key.Binding
	ToggleInspector key.Binding
	Search          key.Binding
	Save            key.Binding
	Settings        key.Binding
	Help            key.Binding
}

func newKeyMap() keyMap {
	return keyMap{
		NewSession:      key.NewBinding(key.WithKeys("ctrl+n"), key.WithHelp("ctrl+n", "new session")),
		SwitchSession:   key.NewBinding(key.WithKeys("ctrl+p"), key.WithHelp("ctrl+p", "switch session")),
		PreviousSession: key.NewBinding(key.WithKeys("alt+left"), key.WithHelp("alt+←", "previous session")),
		NextSession:     key.NewBinding(key.WithKeys("alt+right"), key.WithHelp("alt+→", "next session")),
		CommandPalette:  key.NewBinding(key.WithKeys("ctrl+k"), key.WithHelp("ctrl+k", "commands")),
		ToggleInspector: key.NewBinding(key.WithKeys("ctrl+o"), key.WithHelp("ctrl+o", "inspector")),
		Search:          key.NewBinding(key.WithKeys("ctrl+f"), key.WithHelp("ctrl+f", "search")),
		Save:            key.NewBinding(key.WithKeys("ctrl+s"), key.WithHelp("ctrl+s", "save")),
		Settings:        key.NewBinding(key.WithKeys("f2"), key.WithHelp("f2", "settings")),
		Help:            key.NewBinding(key.WithKeys("f1"), key.WithHelp("f1", "help")),
	}
}

func (keys keyMap) ShortHelp() []key.Binding {
	return []key.Binding{keys.CommandPalette, keys.SwitchSession, keys.ToggleInspector, keys.Help}
}

func (keys keyMap) FullHelp() [][]key.Binding {
	return [][]key.Binding{
		{keys.NewSession, keys.SwitchSession, keys.PreviousSession, keys.NextSession},
		{keys.CommandPalette, keys.ToggleInspector, keys.Search, keys.Save, keys.Settings, keys.Help},
	}
}
