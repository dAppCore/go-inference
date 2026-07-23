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

	if !strings.Contains(lines[0], "MODELS  3 local") || !strings.Contains(lines[0], "/ filter by name…") {
		t.Fatalf("panel must open with its canonical toolbar: %q", lines[0])
	}
	if !strings.Contains(plain, "/ filter by name…") {
		t.Fatalf("panel must keep the canonical filter prompt visible: %q", plain)
	}
	header := ""
	for _, line := range lines {
		if strings.Contains(line, "Model") && strings.Contains(line, "Engine") && strings.Contains(line, "Snapshot path") {
			header = line
			break
		}
	}
	if header == "" || strings.Index(header, "Model") >= strings.Index(header, "Engine") || strings.Index(header, "Engine") >= strings.Index(header, "Snapshot path") {
		t.Fatalf("canonical table headings missing or out of order: %q", plain)
	}
	if !strings.Contains(plain, "│ beta") {
		t.Fatalf("selected row must carry the table selection rule: %q", plain)
	}
	if strings.Contains(plain, "No items.") || strings.Contains(plain, "page ") {
		t.Fatalf("a populated single page must carry no empty or page state: %q", plain)
	}

	for _, model := range []struct {
		name, engine, path string
	}{
		{"alpha", "fake", "/models/alpha"},
		{"beta", "fake", "/models/beta"},
		{"gamma", "fake", "/models/gamma"},
	} {
		row := ""
		for _, line := range lines {
			if strings.Contains(line, model.name) {
				row = line
				break
			}
		}
		if row == "" {
			t.Fatalf("table row for %q missing: %q", model.name, plain)
		}
		nameAt, engineAt, pathAt := strings.Index(row, model.name), strings.Index(row, model.engine), strings.Index(row, model.path)
		if nameAt < 0 || engineAt <= nameAt || pathAt <= engineAt {
			t.Fatalf("table row %q does not keep Model / Engine / Snapshot path order", row)
		}
	}
	if strings.Contains(plain, "○ alpha") || strings.Contains(plain, "○ gamma") {
		t.Fatalf("table rows must not retain the old list-item circle markers: %q", plain)
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
	if !strings.Contains(plain, "MODELS  0 local") || !strings.Contains(plain, "/ filter by name…") || !strings.Contains(plain, "No local models found") {
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
	if !strings.Contains(plain, "/ beta") || !strings.Contains(plain, "1 of 2") {
		t.Fatalf("filtering must surface the typed filter: %q", plain)
	}
	if !strings.Contains(plain, "│ beta") || strings.Contains(plain, "alpha") {
		t.Fatalf("filtering must render matches only, cursor on the first: %q", plain)
	}
	filtered.SetFilterText("zzz")
	filtered.SetFilterState(list.Filtering)
	plain = ansi.Strip(renderPicker(filtered, 80, styles))
	if !strings.Contains(plain, "0 of 2") || strings.Contains(plain, "No local models found") || strings.Contains(plain, "○ ") {
		t.Fatalf("a match-free filter renders its count but no discovery-empty mislabel: %q", plain)
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
		t.Fatalf("an over-budget path must truncate with an ellipsis: %q", plain)
	}
	if !strings.Contains(plain, "0123456789abcdef") {
		t.Fatalf("the truncated Snapshot path must retain its identifying tail: %q", plain)
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
	if !strings.Contains(plain, "│ m00") {
		t.Fatalf("the first page must open on the first item: %q", plain)
	}
	if strings.Contains(plain, "m29") {
		t.Fatalf("rows beyond the current page must not render: %q", plain)
	}
}
