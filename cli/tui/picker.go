// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"sort"

	"github.com/charmbracelet/bubbles/list"
	tea "github.com/charmbracelet/bubbletea"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// modelItem is one discovered checkpoint in the picker list.
type modelItem struct {
	path      string
	name      string
	modelType string
}

func (m modelItem) Title() string       { return m.name }
func (m modelItem) Description() string { return m.modelType + "  " + m.path }
func (m modelItem) FilterValue() string { return m.name + " " + m.modelType }

// discoveredMsg carries the picker's items once the scan completes.
type discoveredMsg struct{ items []list.Item }

// discoverModels scans the HuggingFace hub cache (and LEM_MODELS_DIR when set)
// for loadable checkpoints via inference.Discover — the same walk pkg/discover
// documents. Runs as a tea.Cmd so a slow disk never blocks the first frame.
func discoverModels() tea.Msg {
	dirs := []string{}
	if homeResult := core.UserHomeDir(); homeResult.OK {
		if home := homeResult.String(); core.Trim(home) != "" {
			dirs = append(dirs, core.Path(home, ".cache", "huggingface", "hub"))
		}
	}
	if extra := core.Trim(core.Getenv("LEM_MODELS_DIR")); extra != "" {
		dirs = append(dirs, extra)
	}
	var items []list.Item
	seen := map[string]bool{}
	for _, dir := range dirs {
		for m := range inference.Discover(dir) {
			if seen[m.Path] {
				continue
			}
			seen[m.Path] = true
			items = append(items, modelItem{
				path:      m.Path,
				name:      displayName(m.Path),
				modelType: m.ModelType,
			})
		}
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].(modelItem).name < items[j].(modelItem).name
	})
	return discoveredMsg{items: items}
}

// displayName compresses a hub snapshot path to its repo name — the hub layout
// is models--ORG--NAME/snapshots/HASH, anything else keeps its base name.
func displayName(path string) string {
	for _, part := range core.Split(path, string(core.PathSeparator)) {
		if rest, ok := core.CutPrefix(part, "models--"); ok {
			if _, name, found := core.Cut(rest, "--"); found {
				return name
			}
			return rest
		}
	}
	return core.PathBase(path)
}

// newPicker builds the model list with the house dark styling.
func newPicker(styles uiStyles) list.Model {
	delegate := list.NewDefaultDelegate()
	l := list.New(nil, delegate, 0, 0)
	l.Title = "Models"
	l.SetShowStatusBar(false)
	l.SetFilteringEnabled(true)
	l.Styles.Title = styles.title
	return l
}
