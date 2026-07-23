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

const modelPickerChromeRows = 3

// newPicker builds the model list. The list.Model owns picker STATE only —
// items, cursor, filter, pagination — while picker.ctml renders the mockup's
// dense one-line table rows.
func newPicker() list.Model {
	delegate := list.NewDefaultDelegate()
	delegate.ShowDescription = false
	delegate.SetSpacing(0)
	l := list.New(nil, delegate, 0, 0)
	l.Title = "Models"
	l.SetShowTitle(false)
	l.SetShowFilter(false)
	l.SetShowStatusBar(false)
	l.SetShowPagination(false)
	l.SetShowHelp(false)
	l.SetFilteringEnabled(true)
	return l
}

// resizeModelPicker reserves the three structural rows owned by picker.ctml:
// its toolbar, table heading, and optional page receipt.
func resizeModelPicker(picker *list.Model, width, height int) {
	if picker == nil {
		return
	}
	picker.SetSize(max(1, width), max(1, height-modelPickerChromeRows))
}

// pickerCTML is the Models panel's markup — see picker.ctml for the seams it
// exposes (row/filter/empty/page sequences, class tokens, the model-picker
// block id).
//
//go:embed picker.ctml
var pickerCTML []byte

// modelPickerBindings derives the panel's toolbar and current page from the
// list model. Table columns are padded host-side because terminal markup has
// no borderless fixed-grid primitive; state, filtering, and pagination stay
// entirely in list.Model.
func modelPickerBindings(picker list.Model, width int) ctml.Bindings {
	sequences := map[string][]map[string]any{
		"filterIdle":   {},
		"filterActive": {},
		"rows":         {},
		"empty":        {},
		"page":         {},
	}
	filterValue := core.Trim(picker.FilterValue())
	if filterValue == "" {
		sequences["filterIdle"] = append(sequences["filterIdle"], map[string]any{})
	} else {
		sequences["filterActive"] = append(sequences["filterActive"], map[string]any{"value": filterValue})
	}

	visible := picker.VisibleItems()
	count := core.Sprintf("%d local", len(picker.Items()))
	if filterValue != "" {
		count = core.Sprintf("%d of %d", len(visible), len(picker.Items()))
	}

	start, end := picker.Paginator.GetSliceBounds(len(visible))
	active := picker.Index() - start
	nameWidth, engineWidth, pathWidth := modelPickerColumnWidths(width)
	for index, item := range visible[start:end] {
		entry, ok := item.(list.DefaultItem)
		if !ok {
			continue
		}
		state, marker := "row-idle", "  "
		if index == active {
			state, marker = "row-active", "│ "
		}
		id := entry.Title()
		name, engine, path := entry.Title(), "", entry.Description()
		if model, ok := item.(modelItem); ok {
			id = model.path
			name, engine, path = model.name, model.modelType, model.path
		}
		sequences["rows"] = append(sequences["rows"], map[string]any{
			"state":  state,
			"marker": marker,
			"id":     id,
			"name":   padPickerCell(style.Truncate(name, max(1, nameWidth-2), "…"), max(1, nameWidth-2)),
			"engine": padPickerCell(style.Truncate(engine, engineWidth, "…"), engineWidth),
			"path":   padPickerCell(truncatePickerTail(path, pathWidth), pathWidth),
		})
	}
	if len(picker.Items()) == 0 {
		sequences["empty"] = append(sequences["empty"], map[string]any{})
	}
	if picker.Paginator.TotalPages > 1 {
		sequences["page"] = append(sequences["page"], map[string]any{
			"label": core.Sprintf("page %d/%d", picker.Paginator.Page+1, picker.Paginator.TotalPages),
		})
	}
	return ctml.Bindings{
		Sequences: sequences,
		Values: map[string]any{
			"count":      count,
			"headName":   "  " + padPickerCell("Model", max(1, nameWidth-2)),
			"headEngine": padPickerCell("Engine", engineWidth),
			"headPath":   padPickerCell("Snapshot path", pathWidth),
		},
	}
}

func modelPickerColumnWidths(width int) (name, engine, path int) {
	width = max(1, width)
	name = max(8, width*31/100)
	engine = max(7, width*15/100)
	if name+engine >= width {
		name = max(1, width/2)
		engine = max(1, min(7, width-name-1))
	}
	path = max(1, width-name-engine)
	return name, engine, path
}

func padPickerCell(value string, width int) string {
	width = max(1, width)
	value = style.Truncate(value, width, "…")
	return value + core.Repeat(" ", max(0, width-style.Measure(value)))
}

func truncatePickerTail(value string, width int) string {
	width = max(1, width)
	if style.Measure(value) <= width {
		return value
	}
	if width == 1 {
		return "…"
	}
	runes := []rune(value)
	for len(runes) > 0 && style.Measure("…"+string(runes)) > width {
		runes = runes[1:]
	}
	return "…" + string(runes)
}

// modelPickerTheme maps the markup's class tokens onto the existing palette,
// so the .ctml render reuses uiStyles paint exactly — no colours of its own.
func modelPickerTheme(styles uiStyles) *html.TermTheme {
	theme := html.DefaultTermTheme()
	theme.Text = styles.answer
	theme.Classes = map[string]style.Style{
		"picker-title":  styles.title,
		"picker-count":  styles.status,
		"table-heading": styles.status,
		"row-idle":      styles.title,
		"row-active":    styles.accent.Bold(true),
		"row-engine":    styles.status,
		"row-path":      styles.thought,
		"filter-prompt": styles.accent,
		"filter-value":  styles.answer,
		"filter-idle":   styles.status,
		"picker-empty":  styles.status,
		"picker-page":   styles.status,
	}
	return theme
}

// renderPicker parses picker.ctml with bindings derived from the list
// model's current state and renders it through the go-html terminal renderer:
// toolbar, borderless Model / Engine / Snapshot path table, empty state, and
// page receipt. Cursor movement, fuzzy filtering, and pagination stay in
// list.Model — this is the render swap over a.picker.Update.
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
