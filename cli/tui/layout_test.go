// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"

	"dappco.re/go/html/engine/ctml"
	"dappco.re/go/html/engine/html"
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

// TestWideInspectorWidth_MatchesRequest pins wideInspectorWidth (layout.go)
// against a live render of the REAL shellwide.ctml shell path: since go-html
// v0.15.0, R's outer width is a REQUEST (TermOptions.AsideWidth, docs/ctml.md
// S:15.1) the caller makes going in, not a fixed upstream constant to mirror
// -- S:15.5's own doctrine is "the box map is the render-time source of
// truth", so this test reads the requested width back from the render's own
// BoxMap rather than asserting a hardcoded upstream default. If go-html ever
// stops honouring the AsideWidth request, this fails loudly here instead of
// shellwide.ctml's frame silently reflowing back to go-html's unrequested
// 28-column default.
func TestWideInspectorWidth_MatchesRequest(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	layout, err := ctml.ParseLayout(shellWideCTML, shellWideBindings("HEADER", "FOOTER", "MAIN", "INSPECTOR"))
	if err != nil {
		t.Fatalf("parse shellwide.ctml: %v", err)
	}
	_, boxes := layout.RenderTermBoxes(html.NewContext(), html.TermOptions{
		Width: 140, Theme: shellWideTheme(styles), AsideWidth: wideInspectorWidth,
	})
	if got := boxes["R"].Width; got != wideInspectorWidth {
		t.Fatalf("shellwide.ctml's live R box width = %d, want the requested wideInspectorWidth = %d (ctml.md S:15.1) -- go-html stopped honouring the AsideWidth request", got, wideInspectorWidth)
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
