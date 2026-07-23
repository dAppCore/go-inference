// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	_ "embed"
	"sort"

	tea "dappco.re/go/render/display/tui"
	"dappco.re/go/render/display/tui/list"
	"dappco.re/go/render/display/tui/style"

	core "dappco.re/go"
	"dappco.re/go/render/engine/html"
	"dappco.re/go/render/engine/ctml"
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

// newPicker builds the model list. The list.Model owns picker STATE only —
// items, cursor, filter, pagination — and its default delegate survives as
// the page-size driver (height 2 + spacing 1, the same three cells a
// picker.ctml row occupies); rendering goes through renderPicker, so the
// list carries no styling of its own.
func newPicker() list.Model {
	l := list.New(nil, list.NewDefaultDelegate(), 0, 0)
	l.Title = "Models"
	l.SetShowStatusBar(false)
	l.SetFilteringEnabled(true)
	return l
}

// pickerCTML is the Models panel's markup — see picker.ctml for the seams it
// exposes (row/filter/empty/page sequences, class tokens, the model-picker
// block id).
//
//go:embed picker.ctml
var pickerCTML []byte

// modelPickerBindings derives the panel's rows from the list model's own
// state: the current page of visible items as ONE sequence — selection
// styling rides the row-scoped class bind (class="{{row.state}}", go-html
// v0.13.0) and the marker glyph rides the row, so no before/active/after
// split is needed — plus the zero-or-one-row conditional sections for the
// typed filter, the empty state, and the page position. Each row also
// binds its box id (the model path — unique, discovery de-duplicates on
// it). Name and hint are truncated host-side to the row budget because the
// page math budgets exactly three cells per row — a <dt> wraps to the
// render width, and a wrapped row would overflow the page the list
// delegate sized.
func modelPickerBindings(picker list.Model, width int) ctml.Bindings {
	sequences := map[string][]map[string]any{
		"filter": {},
		"rows":   {},
		"empty":  {},
		"page":   {},
	}
	if picker.FilterState() == list.Filtering {
		sequences["filter"] = append(sequences["filter"], map[string]any{"value": picker.FilterValue()})
	}
	visible := picker.VisibleItems()
	start, end := picker.Paginator.GetSliceBounds(len(visible))
	active := picker.Index() - start
	budget := max(1, width-2)
	for index, item := range visible[start:end] {
		entry, ok := item.(list.DefaultItem)
		if !ok {
			continue
		}
		state, marker := "row-idle", "○"
		if index == active {
			state, marker = "row-active", "›"
		}
		id := entry.Title()
		if model, ok := item.(modelItem); ok {
			id = model.path
		}
		sequences["rows"] = append(sequences["rows"], map[string]any{
			"state":  state,
			"marker": marker,
			"id":     id,
			"name":   style.Truncate(entry.Title(), budget, "…"),
			"detail": style.Truncate(entry.Description(), budget, "…"),
		})
	}
	if len(visible) == 0 && picker.FilterState() != list.Filtering {
		sequences["empty"] = append(sequences["empty"], map[string]any{"text": "No items."})
	}
	if picker.Paginator.TotalPages > 1 {
		sequences["page"] = append(sequences["page"], map[string]any{
			"label": core.Sprintf("page %d/%d", picker.Paginator.Page+1, picker.Paginator.TotalPages),
		})
	}
	return ctml.Bindings{Sequences: sequences}
}

// modelPickerTheme maps the markup's class tokens onto the existing palette,
// so the .ctml render reuses uiStyles paint exactly — no colours of its own.
func modelPickerTheme(styles uiStyles) *html.TermTheme {
	theme := html.DefaultTermTheme()
	theme.Text = styles.answer
	theme.Heading = styles.title // the <h2> panel title
	theme.Classes = map[string]style.Style{
		"row-idle":      styles.answer,
		"row-active":    styles.accent,
		"row-hint":      styles.thought,
		"filter-prompt": styles.accent,
		"filter-value":  styles.answer,
		"picker-empty":  styles.status,
		"picker-page":   styles.status,
		"picker-keys":   styles.status,
	}
	return theme
}

// renderPicker parses picker.ctml with bindings derived from the list
// model's current state and renders it through the go-html terminal
// renderer: the panel title, the filter line while filtering, one <dl> row
// per model on the current page (marker + name, indented type/path hint),
// the empty and page states, and the key footer. Cursor movement, fuzzy
// filtering, and pagination stay in list.Model — this is the render swap
// over a.picker.Update.
func renderPicker(picker list.Model, width int, styles uiStyles) string {
	if width <= 0 {
		return ""
	}
	tree, err := ctml.Parse(pickerCTML, modelPickerBindings(picker, width))
	if err != nil {
		// picker.ctml is embedded and static, so a parse failure is a build
		// defect; TestRenderPicker_Good pins the markup as parseable.
		return ""
	}
	return html.RenderTerm(tree, html.NewContext(), html.TermOptions{Width: width, Theme: modelPickerTheme(styles)})
}
