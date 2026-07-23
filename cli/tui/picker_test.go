// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"

	core "dappco.re/go"
	"dappco.re/go/render/display/tui/list"
)

func pickerWithItems(items ...list.Item) list.Model {
	picker := newPicker()
	picker.SetSize(80, 30)
	picker.SetItems(items)
	return picker
}

func TestRenderPicker_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	picker := pickerWithItems(
		modelItem{path: "/models/alpha", name: "alpha", modelType: "fake"},
		modelItem{path: "/models/beta", name: "beta", modelType: "fake"},
		modelItem{path: "/models/gamma", name: "gamma", modelType: "fake"},
	)
	picker.Select(1)
	view := renderPicker(picker, 80, styles)
	plain := ansi.Strip(view)
	lines := strings.Split(plain, "\n")

	if strings.TrimSpace(lines[0]) != "Models" {
		t.Fatalf("panel must open with its title line: %q", lines[0])
	}
	if strings.TrimSpace(lines[1]) != "" {
		t.Fatalf("title must be followed by a blank separator: %q", lines[1])
	}
	if !strings.Contains(plain, "› beta") {
		t.Fatalf("selected row must carry the active marker: %q", plain)
	}
	for _, text := range []string{"○ alpha", "○ gamma"} {
		if !strings.Contains(plain, text) {
			t.Fatalf("unselected row missing the idle marker %q: %q", text, plain)
		}
	}
	for _, text := range []string{"fake  /models/alpha", "fake  /models/beta", "fake  /models/gamma"} {
		if !strings.Contains(plain, text) {
			t.Fatalf("row missing the type/path hint %q: %q", text, plain)
		}
	}
	if !strings.Contains(plain, "↑/↓ select · / filter · enter load") {
		t.Fatalf("panel missing the key footer: %q", plain)
	}
	if strings.Contains(plain, "No items.") || strings.Contains(plain, "page ") {
		t.Fatalf("a populated single page must carry no empty or page state: %q", plain)
	}

	row := -1
	for index, line := range lines {
		if strings.Contains(line, "› beta") {
			row = index
			break
		}
	}
	if row < 0 || row+1 >= len(lines) {
		t.Fatalf("selected row not found in the rendered lines: %q", plain)
	}
	if !strings.Contains(lines[row+1], "fake  /models/beta") {
		t.Fatalf("hint must sit directly beneath its name line: %q", lines[row+1])
	}

	first := strings.Index(plain, "alpha")
	second := strings.Index(plain, "beta")
	third := strings.Index(plain, "gamma")
	if first < 0 || second < first || third < second {
		t.Fatalf("the sequence split must preserve row order: %q", plain)
	}

	for index, line := range lines {
		if got := lipgloss.Width(line); got > 80 {
			t.Fatalf("line %d width = %d, exceeds 80: %q", index, got, line)
		}
	}
}

func TestRenderPicker_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	picker := pickerWithItems(modelItem{path: "/models/alpha", name: "alpha", modelType: "fake"})
	for _, width := range []int{0, -4} {
		if got := renderPicker(picker, width, styles); got != "" {
			t.Fatalf("renderPicker(width=%d) = %q, want empty", width, got)
		}
	}
}

func TestRenderPicker_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())

	// A pickerless start renders the honest empty state, not a bare pane.
	empty := newPicker()
	empty.SetSize(40, 20)
	plain := ansi.Strip(renderPicker(empty, 40, styles))
	if !strings.Contains(plain, "No items.") {
		t.Fatalf("an empty picker must render its empty state: %q", plain)
	}

	// While the filter is being typed, the filter line shows the typed value,
	// matches stay live, and misses leave — with no "No items." mislabel.
	filtered := pickerWithItems(
		modelItem{path: "/models/alpha", name: "alpha", modelType: "fake"},
		modelItem{path: "/models/beta", name: "beta", modelType: "fake"},
	)
	filtered.SetFilterText("beta")
	filtered.SetFilterState(list.Filtering)
	plain = ansi.Strip(renderPicker(filtered, 80, styles))
	if !strings.Contains(plain, "Filter: beta") {
		t.Fatalf("filtering must surface the typed filter: %q", plain)
	}
	if !strings.Contains(plain, "› beta") || strings.Contains(plain, "alpha") {
		t.Fatalf("filtering must render matches only, cursor on the first: %q", plain)
	}
	filtered.SetFilterText("zzz")
	filtered.SetFilterState(list.Filtering)
	plain = ansi.Strip(renderPicker(filtered, 80, styles))
	if strings.Contains(plain, "No items.") || strings.Contains(plain, "○ ") {
		t.Fatalf("a match-free filter renders no rows and no empty mislabel: %q", plain)
	}

	// A long snapshot path truncates to the row budget instead of wrapping,
	// so a page of rows keeps the density the list model paginates by.
	long := pickerWithItems(modelItem{
		path:      "/home/user/.cache/huggingface/hub/models--org--very-long-name/snapshots/0123456789abcdef",
		name:      "very-long-name",
		modelType: "gemma3_text",
	})
	long.SetSize(40, 20)
	view := renderPicker(long, 40, styles)
	plain = ansi.Strip(view)
	if !strings.Contains(plain, "…") {
		t.Fatalf("an over-budget hint must truncate with an ellipsis: %q", plain)
	}
	for index, line := range strings.Split(plain, "\n") {
		if got := lipgloss.Width(line); got > 40 {
			t.Fatalf("line %d width = %d, exceeds 40: %q", index, got, line)
		}
	}

	// More items than one short pane holds: only the current page renders,
	// with the page position stated.
	items := make([]list.Item, 0, 30)
	for index := range 30 {
		letter := string(rune('a' + index%26))
		items = append(items, modelItem{
			path:      "/models/m" + letter + core.Sprintf("%02d", index),
			name:      "m" + core.Sprintf("%02d", index),
			modelType: "fake",
		})
	}
	paged := newPicker()
	paged.SetSize(40, 10)
	paged.SetItems(items)
	plain = ansi.Strip(renderPicker(paged, 40, styles))
	if !strings.Contains(plain, "page 1/") {
		t.Fatalf("a paginated list must state its page position: %q", plain)
	}
	if !strings.Contains(plain, "› m00") {
		t.Fatalf("the first page must open on the first item: %q", plain)
	}
	if strings.Contains(plain, "m29") {
		t.Fatalf("rows beyond the current page must not render: %q", plain)
	}
}
