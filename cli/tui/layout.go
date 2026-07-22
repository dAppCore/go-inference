// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	_ "embed"

	"github.com/charmbracelet/lipgloss"

	core "dappco.re/go"
	"dappco.re/go/html"
	"dappco.re/go/html/ctml"
)

type layoutKind uint8

const (
	layoutNarrow layoutKind = iota
	layoutOverlay
	layoutWide
)

const wideInspectorWidth = 32

// frameInsetRows and frameInsetCols map screen cells to frame-inner cells:
// renderFrame's outer rounded border occupies row 0 and column 0, so inner
// content — the panel bar first — begins one cell in on both axes. Mouse
// hit-testing subtracts these before resolving against a component box map.
const (
	frameInsetRows = 1
	frameInsetCols = 1
)

func chooseLayout(width int) layoutKind {
	switch {
	case width >= 120:
		return layoutWide
	case width >= 80:
		return layoutOverlay
	default:
		return layoutNarrow
	}
}

type frameSpec struct {
	Width         int
	Height        int
	Active        panelID
	SessionStrip  string
	Main          string
	Inspector     string
	Footer        string
	InspectorOpen bool
}

type frameMetrics struct {
	kind            layoutKind
	innerWidth      int
	innerHeight     int
	regionHeight    int
	mainWidth       int
	mainHeight      int
	inspectorWidth  int
	inspectorHeight int
}

func measureFrame(width, height int, inspectorOpen bool) frameMetrics {
	metrics := frameMetrics{
		kind:         chooseLayout(width),
		innerWidth:   max(0, width-2),
		innerHeight:  max(0, height-2),
		regionHeight: max(1, height-5), // border + header + sessions + footer
	}
	metrics.mainWidth = metrics.innerWidth
	metrics.mainHeight = metrics.regionHeight
	switch metrics.kind {
	case layoutWide:
		metrics.inspectorWidth = wideInspectorWidth
		metrics.inspectorHeight = metrics.regionHeight
		metrics.mainWidth = max(1, metrics.innerWidth-wideInspectorWidth-1)
	case layoutOverlay:
		if inspectorOpen {
			metrics.inspectorWidth = metrics.innerWidth
			// Contextual inspectors carry several short sections. Give them room
			// to remain useful while preserving a compact main-panel preview.
			metrics.inspectorHeight = min(12, max(3, metrics.regionHeight-5))
			metrics.mainHeight = max(1, metrics.regionHeight-metrics.inspectorHeight-1)
		}
	case layoutNarrow:
		if inspectorOpen {
			metrics.inspectorWidth = metrics.innerWidth
			metrics.inspectorHeight = metrics.regionHeight
		}
	}
	return metrics
}

// shellCTML is the app shell's markup -- see shell.ctml for the seams it
// exposes (the header/footer verbatim values, and why the region rides
// between them rather than through a middle-band slot).
//
//go:embed shell.ctml
var shellCTML []byte

// shellFrameTheme maps the shell markup onto the existing palette. The H/F
// bands are stripped of the default theme's border and padding -- the same
// technique overlayFrameTheme and dataListTheme already use -- because both
// slots here carry content the host has already fitted to the exact band
// width (renderPanelBar's own fitLine call and footerLine's, both below);
// any theme chrome would offset that pre-fitted content instead of leaving
// it byte-exact.
func shellFrameTheme(styles uiStyles) *html.TermTheme {
	theme := html.DefaultTermTheme()
	theme.Header = lipgloss.NewStyle()
	theme.Footer = lipgloss.NewStyle()
	return theme
}

// shellBindings supplies the shell's two verbatim values. header and footer
// arrive already pre-rendered and pre-fitted to metrics.innerWidth by
// renderFrame, exactly as the pre-.ctml shell computed them.
func shellBindings(header, footer string) ctml.Bindings {
	return ctml.Bindings{Values: map[string]any{"header": header, "footer": footer}}
}

// renderFrame composes the permanent workspace shell. All truncation is done
// by Lip Gloss on rendered cell widths; no ANSI string is sliced manually.
//
// The outer rounded border and the region (the active main panel, plus the
// wide/toggled inspector) stay hand composed. The border because go-html's
// TermTheme has no whole-page border concept -- Header/Footer/Sidebar/Aside
// each style their own band, nothing wraps H+middle+F together (S:15 has no
// such style). The region because the one slot that can fill remaining
// width -- Content -- hardcodes a (0,1) padding with no theme override
// (unlike Header/Footer/Sidebar/Aside), so a full-width verbatim placed
// there is corrupted: content already fitted to the slot's own width
// arrives one column too wide once that padding is added, and the slot's
// own Width() enforcement then word-wraps the overflow onto a spurious
// extra row rather than leaving it byte-exact (shell.ctml's header comment
// has the full account; L/R share the defect through their own hardcoded
// offsets). The region must keep the SAME width contract every panel body
// and the inspector already render against, so it is computed exactly as
// before (renderWorkspaceRegion, unchanged) and rides between the shell's
// H and F bands -- the same HF+host-composition idiom a widget-carrying
// overlay already uses for a live Bubbles widget between its own bands.
//
// The tab strip and session strip (both already fully styled, single-line,
// pre-fitted to metrics.innerWidth) join as the shell's one "header" value;
// the status/key-hint line is the "footer" value. Verbatim is emitted
// exactly as supplied with no added blank, so the "\n" join below is the
// only seam deciding header/session adjacency -- matching the pre-.ctml
// JoinVertical's own row order byte-for-byte.
func renderFrame(spec frameSpec, styles uiStyles) string {
	if spec.Width <= 0 || spec.Height <= 0 {
		return ""
	}
	if spec.Width < 3 || spec.Height < 4 {
		return minimalFrame(spec.Width, spec.Height)
	}
	metrics := measureFrame(spec.Width, spec.Height, spec.InspectorOpen)
	tabstrip := renderPanelBar(spec.Active, metrics.innerWidth, metrics.kind, styles)
	sessions := fitLine(styles.title.Render("SESSIONS")+"  "+styles.session.Render(spec.SessionStrip), metrics.innerWidth, styles.session)
	header := core.Join("\n", tabstrip, sessions)
	footer := fitLine(spec.Footer, metrics.innerWidth, styles.footer)
	region := renderWorkspaceRegion(spec, metrics, styles)
	head, foot := renderBandFrame(shellCTML, metrics.innerWidth, shellFrameTheme(styles), shellBindings(header, footer))
	inside := lipgloss.JoinVertical(lipgloss.Left, head, region, foot)
	return styles.outerFrame.
		Width(metrics.innerWidth).
		Height(metrics.innerHeight).
		MaxWidth(spec.Width).
		MaxHeight(spec.Height).
		Render(inside)
}

func renderWorkspaceRegion(spec frameSpec, metrics frameMetrics, styles uiStyles) string {
	switch metrics.kind {
	case layoutWide:
		main := fitPane(spec.Main, metrics.mainWidth, metrics.mainHeight, styles.panel)
		separator := fitPane(core.Repeat("│\n", max(0, metrics.regionHeight-1))+"│", 1, metrics.regionHeight, styles.separator)
		inspector := fitPane(spec.Inspector, metrics.inspectorWidth, metrics.inspectorHeight, styles.inspector)
		return lipgloss.JoinHorizontal(lipgloss.Top, main, separator, inspector)
	case layoutOverlay:
		if spec.InspectorOpen {
			inspector := fitPane(spec.Inspector, metrics.inspectorWidth, metrics.inspectorHeight, styles.inspector)
			separator := fitLine(core.Repeat("─", metrics.innerWidth), metrics.innerWidth, styles.separator)
			main := fitPane(spec.Main, metrics.mainWidth, metrics.mainHeight, styles.panel)
			return lipgloss.JoinVertical(lipgloss.Left, inspector, separator, main)
		}
	case layoutNarrow:
		if spec.InspectorOpen {
			return fitPane(spec.Inspector, metrics.innerWidth, metrics.regionHeight, styles.inspector)
		}
	}
	return fitPane(spec.Main, metrics.mainWidth, metrics.mainHeight, styles.panel)
}

func fitLine(content string, width int, style lipgloss.Style) string {
	if width <= 0 {
		return ""
	}
	return style.Width(width).MaxWidth(width).MaxHeight(1).Render(content)
}

func fitPane(content string, width, height int, style lipgloss.Style) string {
	if width <= 0 || height <= 0 {
		return ""
	}
	return style.Width(width).Height(height).MaxWidth(width).MaxHeight(height).Render(content)
}

func minimalFrame(width, height int) string {
	if width <= 0 || height <= 0 {
		return ""
	}
	line := core.Repeat("─", width)
	lines := make([]string, height)
	for i := range lines {
		lines[i] = line
	}
	return core.Join("\n", lines...)
}
