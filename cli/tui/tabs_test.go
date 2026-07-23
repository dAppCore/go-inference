// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"

	"dappco.re/go/html/engine/html"
)

func TestRenderPanelBar_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	view := renderPanelBar(panelChat, 120, layoutWide, styles)
	plain := ansi.Strip(view)
	if strings.Contains(plain, "\n") {
		t.Fatalf("panel bar must render one row: %q", plain)
	}
	for _, text := range []string{"LEM", "● Chat", "○ Work", "○ Models", "○ Service", "○ Data"} {
		if !strings.Contains(plain, text) {
			t.Fatalf("panel bar missing %q: %q", text, plain)
		}
	}
	if strings.Contains(plain, "○ Chat") {
		t.Fatalf("active tab must not carry the inactive marker: %q", plain)
	}
	if got := lipgloss.Width(view); got > 120 {
		t.Fatalf("panel bar width = %d, exceeds 120", got)
	}
}

func TestRenderPanelBar_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	for _, width := range []int{0, -4} {
		if got := renderPanelBar(panelChat, width, layoutWide, styles); got != "" {
			t.Fatalf("renderPanelBar(width=%d) = %q, want empty", width, got)
		}
	}
}

func TestRenderPanelBar_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	view := renderPanelBar(panelData, 72, layoutNarrow, styles)
	plain := ansi.Strip(view)
	if !strings.Contains(plain, "○ API") || strings.Contains(plain, "Service") {
		t.Fatalf("narrow bar must use compact labels: %q", plain)
	}
	if !strings.Contains(plain, "● Data") || strings.Contains(plain, "○ Data") {
		t.Fatalf("last tab active must carry the active marker: %q", plain)
	}
	if got := lipgloss.Width(view); got > 72 {
		t.Fatalf("narrow bar width = %d, exceeds 72", got)
	}
}

// panelTabSpan mirrors panelBarHit's cell walk for assertions: the strip
// coordinates [start, end) one tab's marker+label occupies, from the C
// slot's box and the same panelBarCells the bindings rendered.
func panelTabSpan(boxes html.BoxMap, active panelID, kind layoutKind, panel panelID) (start, end int, ok bool) {
	tabs, exists := boxes[tabsSlotID]
	if !exists {
		return 0, 0, false
	}
	edge := tabs.Col + tabs.Width
	pos := tabs.Col + 1
	for _, cell := range panelBarCells(active, kind) {
		cellStart := pos + cell.gap
		cellEnd := min(pos+lipgloss.Width(cell.cell), edge)
		if cellStart >= edge {
			return 0, 0, false
		}
		if cell.panel == panel {
			return cellStart, cellEnd, cellStart < cellEnd
		}
		pos += lipgloss.Width(cell.cell)
	}
	return 0, 0, false
}

func TestRenderPanelBarBoxes_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	line, boxes := renderPanelBarBoxes(panelWork, 120, layoutWide, styles)
	brand, ok := boxes[brandSlotID]
	if !ok {
		t.Fatal("box map missing the brand slot")
	}
	if brand.Row != 0 || brand.Col != 0 || brand.Height != 1 {
		t.Fatalf("brand box = %+v, want a single row at the origin", brand)
	}
	tabs, ok := boxes[tabsSlotID]
	if !ok {
		t.Fatal("box map missing the tabs slot")
	}
	if tabs.Row != 0 || tabs.Height != 1 {
		t.Fatalf("tabs box = %+v, want single row 0", tabs)
	}
	if tabs.Col != brand.Col+brand.Width {
		t.Fatalf("tabs box starts at %d, want it abutting the brand slot ending at %d", tabs.Col, brand.Col+brand.Width)
	}
	if tabs.Col+tabs.Width > 120 {
		t.Fatalf("tabs box = %+v, exceeds the render width", tabs)
	}
	lastEnd := 0
	for panel := panelID(0); panel < panelCount; panel++ {
		start, end, ok := panelTabSpan(boxes, panelWork, layoutWide, panel)
		if !ok {
			t.Fatalf("no span for panel %v", panel)
		}
		if start < lastEnd {
			t.Fatalf("panel %v span starts at %d, overlaps the previous tab ending at %d", panel, start, lastEnd)
		}
		if end > tabs.Col+tabs.Width {
			t.Fatalf("panel %v span [%d, %d) exceeds the tabs slot", panel, start, end)
		}
		lastEnd = end
	}
	if got := lipgloss.Width(line); got != 120 {
		t.Fatalf("fitted strip width = %d, want 120", got)
	}
}

func TestRenderPanelBarBoxes_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	line, boxes := renderPanelBarBoxes(panelChat, 0, layoutWide, styles)
	if line != "" || len(boxes) != 0 {
		t.Fatalf("zero width must render nothing: line=%q boxes=%d", line, len(boxes))
	}
}

// TestRenderPanelBarBoxes_Ugly: the fitted strip truncates at the render
// width, so the recorded boxes are clamped to it — a tab pushed past the
// edge keeps no hit span (previously asserted through the derived per-tab
// boxes, which the native slot boxes replaced).
func TestRenderPanelBarBoxes_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	_, boxes := renderPanelBarBoxes(panelChat, 24, layoutNarrow, styles)
	tabs, ok := boxes[tabsSlotID]
	if !ok {
		t.Fatal("tiny bar still records the tabs slot")
	}
	if tabs.Col+tabs.Width > 24 {
		t.Fatalf("tabs box = %+v, want it clamped to the fitted width", tabs)
	}
	if _, _, ok := panelTabSpan(boxes, panelChat, layoutNarrow, panelChat); !ok {
		t.Fatal("tiny bar must still span the first tab")
	}
	if _, _, ok := panelTabSpan(boxes, panelChat, layoutNarrow, panelData); ok {
		t.Fatal("a tab pushed off the visible row must not keep a hit span")
	}
}

func TestPanelBarHit_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	_, boxes := renderPanelBarBoxes(panelChat, 120, layoutWide, styles)
	for panel := panelID(0); panel < panelCount; panel++ {
		start, end, ok := panelTabSpan(boxes, panelChat, layoutWide, panel)
		if !ok {
			t.Fatalf("no span for panel %v", panel)
		}
		for _, x := range []int{start, (start + end) / 2, end - 1} {
			got, hit := panelBarHit(boxes, x, 0, panelChat, layoutWide)
			if !hit || got != panel {
				t.Fatalf("panelBarHit(%d, 0) = (%v, %v), want (%v, true)", x, got, hit, panel)
			}
		}
	}
}

func TestPanelBarHit_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	_, boxes := renderPanelBarBoxes(panelChat, 120, layoutWide, styles)
	if _, hit := panelBarHit(boxes, 0, 0, panelChat, layoutWide); hit {
		t.Fatal("the brand cell must not resolve to a tab")
	}
	if _, hit := panelBarHit(boxes, 0, 5, panelChat, layoutWide); hit {
		t.Fatal("a coordinate below the strip must not resolve to a tab")
	}
}

func TestPanelBarHit_Ugly(t *testing.T) {
	if _, hit := panelBarHit(nil, 3, 0, panelChat, layoutWide); hit {
		t.Fatal("a nil box map must resolve to no tab")
	}
	styles := newUIStyles(midnightTheme())
	_, boxes := renderPanelBarBoxes(panelChat, 120, layoutWide, styles)
	start, _, ok := panelTabSpan(boxes, panelChat, layoutWide, panelWork)
	if !ok {
		t.Fatal("no span for the Work tab")
	}
	gap := start - 1 // the inter-tab spacing resolves to the strip, not a tab
	if _, hit := panelBarHit(boxes, gap, 0, panelChat, layoutWide); hit {
		t.Fatal("an inter-tab gap must not resolve to a tab")
	}
}
