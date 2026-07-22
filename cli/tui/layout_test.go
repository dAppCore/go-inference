// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"

	"dappco.re/go/html"
)

func TestChooseLayout_Good(t *testing.T) {
	for _, width := range []int{120, 160} {
		if got := chooseLayout(width); got != layoutWide {
			t.Fatalf("chooseLayout(%d) = %d, want wide", width, got)
		}
	}
}

func TestChooseLayout_Bad(t *testing.T) {
	for _, width := range []int{80, 119} {
		if got := chooseLayout(width); got != layoutOverlay {
			t.Fatalf("chooseLayout(%d) = %d, want overlay", width, got)
		}
	}
}

func TestChooseLayout_Ugly(t *testing.T) {
	for _, width := range []int{0, 1, 79} {
		if got := chooseLayout(width); got != layoutNarrow {
			t.Fatalf("chooseLayout(%d) = %d, want narrow", width, got)
		}
	}
}

// TestRegionAsideWidth_MatchesGoHTML pins regionAsideWidth (layout.go)
// against a live go-html render: R's fixed outer-width budget is an
// unexported constant (termAsideWidth, ctml.md S:15.1/S:15.5), so this
// package duplicates it locally rather than reading it live every frame.
// If go-html ever changes that budget, this test fails loudly instead of
// shellwide.ctml's frame silently reflowing.
func TestRegionAsideWidth_MatchesGoHTML(t *testing.T) {
	theme := html.DefaultTermTheme()
	theme.Content = lipgloss.NewStyle()
	theme.Aside = lipgloss.NewStyle()
	page := html.NewLayout("CR").C(html.Verbatim("")).R(html.Verbatim(""))
	_, boxes := page.RenderTermBoxes(html.NewContext(), html.TermOptions{Width: 120, Theme: theme})
	if got := boxes["R"].Width; got != regionAsideWidth {
		t.Fatalf("go-html's live R budget = %d, regionAsideWidth const = %d -- update the constant (ctml.md S:15.1)", got, regionAsideWidth)
	}
}

func TestPanelID_Good(t *testing.T) {
	want := []panelID{panelChat, panelWork, panelModels, panelService, panelData}
	panel := panelChat
	for i, expected := range want {
		if panel != expected {
			t.Fatalf("forward panel %d = %d, want %d", i, panel, expected)
		}
		panel = panel.next()
	}
	if panel != panelChat {
		t.Fatalf("forward wrap = %d, want chat", panel)
	}
	panel = panelChat
	for i := len(want) - 1; i >= 0; i-- {
		panel = panel.prev()
		if panel != want[i] {
			t.Fatalf("reverse panel %d = %d, want %d", i, panel, want[i])
		}
	}
}

func TestWorkspaceFrame_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	view := renderFrame(frameSpec{
		Width:        140,
		Height:       26,
		Active:       panelChat,
		SessionStrip: "● New session   ○ Refactor scheduler",
		Main:         "MAIN REGION\nchat transcript",
		Inspector:    "INSPECTOR\nmodel ready",
		Footer:       "FOOTER ctrl+k commands",
	}, styles)
	assertFrameBasics(t, view, 140)
	if !strings.Contains(view, "INSPECTOR") {
		t.Fatal("wide frame did not render its permanent inspector")
	}
}

func TestWorkspaceFrame_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	closed := renderFrame(frameSpec{
		Width: 100, Height: 24, Active: panelModels,
		SessionStrip: "● New session", Main: "MAIN REGION", Inspector: "INSPECTOR", Footer: "FOOTER",
	}, styles)
	assertFrameBasics(t, closed, 100)
	if strings.Contains(closed, "INSPECTOR") {
		t.Fatal("closed overlay frame rendered the inspector")
	}
	open := renderFrame(frameSpec{
		Width: 100, Height: 24, Active: panelModels, InspectorOpen: true,
		SessionStrip: "● New session", Main: "MAIN REGION", Inspector: "INSPECTOR", Footer: "FOOTER",
	}, styles)
	assertFrameBasics(t, open, 100)
	if !strings.Contains(open, "INSPECTOR") || !strings.Contains(open, "MAIN REGION") {
		t.Fatal("open overlay must retain both inspector and main panel")
	}
}

func TestWorkspaceFrame_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	closed := renderFrame(frameSpec{
		Width: 72, Height: 22, Active: panelWork,
		SessionStrip: "● New session", Main: "MAIN REGION", Inspector: "INSPECTOR", Footer: "FOOTER",
	}, styles)
	assertFrameBasics(t, closed, 72)
	if strings.Contains(closed, "INSPECTOR") {
		t.Fatal("closed narrow frame rendered the inspector")
	}
	open := renderFrame(frameSpec{
		Width: 72, Height: 22, Active: panelWork, InspectorOpen: true,
		SessionStrip: "● New session", Main: "MAIN REGION", Inspector: "INSPECTOR", Footer: "FOOTER",
	}, styles)
	if !strings.Contains(open, "INSPECTOR") {
		t.Fatal("open narrow frame did not render inspector as the content view")
	}
	if strings.Contains(open, "MAIN REGION") {
		t.Fatal("open narrow inspector should own the single content column")
	}
	assertFrameWidth(t, open, 72)
}

func assertFrameBasics(t *testing.T, view string, width int) {
	t.Helper()
	for _, text := range []string{"LEM", "Chat", "Work", "Models", "Data", "SESSIONS", "MAIN REGION", "FOOTER"} {
		if !strings.Contains(view, text) {
			t.Fatalf("frame missing %q\n%s", text, view)
		}
	}
	if !strings.HasPrefix(view, "╭") || !strings.Contains(view, "╰") {
		t.Fatalf("frame missing stable rounded outer border\n%s", view)
	}
	assertFrameWidth(t, view, width)
}

func assertFrameWidth(t *testing.T, view string, width int) {
	t.Helper()
	for line, text := range strings.Split(view, "\n") {
		if got := lipgloss.Width(text); got > width {
			t.Fatalf("line %d width = %d, exceeds %d: %q", line, got, width, text)
		}
	}
}
