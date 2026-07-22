// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
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

func TestRenderPanelBarBoxes_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	_, boxes := renderPanelBarBoxes(panelWork, 120, layoutWide, styles)
	bar, ok := boxes[panelBarBlockID]
	if !ok {
		t.Fatal("box map missing the panel-bar block")
	}
	if bar.Row != 0 || bar.Col != 0 || bar.Height != 1 || bar.Width != 120 {
		t.Fatalf("panel-bar box = %+v, want full-width single row at origin", bar)
	}
	lastEnd := 0
	for panel := panelID(0); panel < panelCount; panel++ {
		box, ok := boxes[panelTabBlockID(panel)]
		if !ok {
			t.Fatalf("box map missing %s", panelTabBlockID(panel))
		}
		if box.Row != 0 || box.Height != 1 {
			t.Fatalf("%s box = %+v, want single row 0", panelTabBlockID(panel), box)
		}
		if box.Col < lastEnd {
			t.Fatalf("%s box starts at %d, overlaps previous tab ending at %d", panelTabBlockID(panel), box.Col, lastEnd)
		}
		if box.Col+box.Width > 120 {
			t.Fatalf("%s box = %+v, exceeds the render width", panelTabBlockID(panel), box)
		}
		lastEnd = box.Col + box.Width
	}
}

func TestRenderPanelBarBoxes_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	line, boxes := renderPanelBarBoxes(panelChat, 0, layoutWide, styles)
	if line != "" || len(boxes) != 0 {
		t.Fatalf("zero width must render nothing: line=%q boxes=%d", line, len(boxes))
	}
}

func TestRenderPanelBarBoxes_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	_, boxes := renderPanelBarBoxes(panelChat, 24, layoutNarrow, styles)
	if _, ok := boxes[panelBarBlockID]; !ok {
		t.Fatal("tiny bar still records the panel-bar block")
	}
	if _, ok := boxes[panelTabBlockID(panelChat)]; !ok {
		t.Fatal("tiny bar must still box the first tab")
	}
	if _, ok := boxes[panelTabBlockID(panelData)]; ok {
		t.Fatal("a tab pushed off the visible row must not receive a box")
	}
}

func TestPanelBarHit_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	_, boxes := renderPanelBarBoxes(panelChat, 120, layoutWide, styles)
	for panel := panelID(0); panel < panelCount; panel++ {
		box, ok := boxes[panelTabBlockID(panel)]
		if !ok {
			t.Fatalf("box map missing %s", panelTabBlockID(panel))
		}
		for _, x := range []int{box.Col, box.Col + box.Width/2, box.Col + box.Width - 1} {
			got, hit := panelBarHit(boxes, x, box.Row)
			if !hit || got != panel {
				t.Fatalf("panelBarHit(%d, %d) = (%v, %v), want (%v, true)", x, box.Row, got, hit, panel)
			}
		}
	}
}

func TestPanelBarHit_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	_, boxes := renderPanelBarBoxes(panelChat, 120, layoutWide, styles)
	if _, hit := panelBarHit(boxes, 0, 0); hit {
		t.Fatal("the brand cell must not resolve to a tab")
	}
	if _, hit := panelBarHit(boxes, 0, 5); hit {
		t.Fatal("a coordinate below the strip must not resolve to a tab")
	}
}

func TestPanelBarHit_Ugly(t *testing.T) {
	if _, hit := panelBarHit(nil, 3, 0); hit {
		t.Fatal("a nil box map must resolve to no tab")
	}
	styles := newUIStyles(midnightTheme())
	_, boxes := renderPanelBarBoxes(panelChat, 120, layoutWide, styles)
	work := boxes[panelTabBlockID(panelWork)]
	gap := work.Col - 1 // the inter-tab spacing resolves to the strip, not a tab
	if _, hit := panelBarHit(boxes, gap, 0); hit {
		t.Fatal("an inter-tab gap must not resolve to a tab")
	}
}
