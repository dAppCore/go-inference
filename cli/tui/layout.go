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

// regionAsideWidth is go-html's fixed R-slot outer-width budget in
// non-FitSlots mode at >=80 columns (the unexported termAsideWidth
// constant; docs/ctml.md S:15.1) -- there is no TermOptions or Layout lever
// to request a different width for R (confirmed by reading term_layout.go
// and by a throwaway RenderTermBoxes probe during this slice, not part of
// the shipped diff). go-html exports no accessor for the fixed budgets
// (docs/ctml.md S:15.5: "the budgets stay unexported"), so this constant
// duplicates the value rather than reading it live every frame;
// TestRegionAsideWidth_MatchesGoHTML pins it against a real RenderTermBoxes
// call so any upstream drift fails loudly here instead of silently
// reflowing the Wide frame. Replaces the pre-.ctml wideInspectorWidth=32 --
// see shellwide.ctml's own header comment for the width-contract delta this
// causes.
const regionAsideWidth = 28

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
		metrics.inspectorWidth = regionAsideWidth
		metrics.inspectorHeight = metrics.regionHeight
		// -1: go-html's own C/R gutter column (S:15.1) -- shellwide.ctml.
		metrics.mainWidth = max(1, metrics.innerWidth-regionAsideWidth-1)
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

// shellCTML is the app shell's markup for the ONE region shape that stays
// host composition -- Overlay layouts with the inspector open -- see
// shell.ctml for the seams it exposes and renderInspectorStack for why that
// one shape cannot join shellRegionCTML/shellWideCTML below.
//
//go:embed shell.ctml
var shellCTML []byte

// shellRegionCTML is the shell shape for a single active region pane
// (Narrow either way; Overlay with the inspector closed) -- see
// shellregion.ctml for the seams it exposes.
//
//go:embed shellregion.ctml
var shellRegionCTML []byte

// shellWideCTML is the shell shape for the Wide layout kind's side-by-side
// main panel and inspector -- see shellwide.ctml for the seams it exposes
// and the R-slot width contract.
//
//go:embed shellwide.ctml
var shellWideCTML []byte

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

// shellRegionTheme extends shellFrameTheme with a zero-chrome Content slot:
// C's content -- the single active pane -- arrives pre-fitted to its exact
// target width by renderFrame, exactly as H/F already do, so TermTheme's
// default (0,1) Content gutter is stripped the same way Header/Footer
// already are (go-html v0.14.0's themeable Content lever, docs/ctml.md
// S:15.2) -- otherwise the pre-fitted content lands one column over budget
// and gets word-wrapped onto a spurious extra row (S:15.5).
func shellRegionTheme(styles uiStyles) *html.TermTheme {
	theme := shellFrameTheme(styles)
	theme.Content = lipgloss.NewStyle()
	return theme
}

// shellWideTheme extends shellRegionTheme with a zero-chrome Aside (R) slot
// for the Wide layout's side-by-side main+inspector: both C and R arrive
// pre-fitted to their own exact width by renderFrame, so both slots' default
// chrome is stripped (Aside was already themeable pre-v0.14.0; Content
// joined it in round 4, S:15.2).
func shellWideTheme(styles uiStyles) *html.TermTheme {
	theme := shellRegionTheme(styles)
	theme.Aside = lipgloss.NewStyle()
	return theme
}

// shellBindings supplies the shell's two verbatim values. header and footer
// arrive already pre-rendered and pre-fitted to metrics.innerWidth by
// renderFrame, exactly as the pre-.ctml shell computed them.
func shellBindings(header, footer string) ctml.Bindings {
	return ctml.Bindings{Values: map[string]any{"header": header, "footer": footer}}
}

// shellRegionBindings supplies shellregion.ctml's three verbatim values.
// content is whichever pane is active, pre-fitted by renderFrame to
// metrics.innerWidth x metrics.regionHeight.
func shellRegionBindings(header, footer, content string) ctml.Bindings {
	return ctml.Bindings{Values: map[string]any{"header": header, "footer": footer, "content": content}}
}

// shellWideBindings supplies shellwide.ctml's four verbatim values. main and
// inspector are pre-fitted by renderFrame to their own exact target width
// (metrics.mainWidth, regionAsideWidth) x metrics.regionHeight.
func shellWideBindings(header, footer, main, inspector string) ctml.Bindings {
	return ctml.Bindings{Values: map[string]any{
		"header": header, "footer": footer, "main": main, "inspector": inspector,
	}}
}

// renderFrame composes the permanent workspace shell. All truncation is done
// by Lip Gloss on rendered cell widths; no ANSI string is sliced manually.
//
// The outer rounded border stays hand composed: go-html's TermTheme has no
// whole-page border concept -- Header/Footer/Sidebar/Aside each style their
// own band, nothing wraps H+middle+F together (S:15 has no such style).
//
// The region (the active main panel, plus the wide/toggled inspector) joins
// the shell's own declarative surface for every shape but one, since
// go-html v0.14.0: a zero-chrome Content slot (and the pre-existing
// themeable Aside) now passes pre-fitted content through byte-exact at the
// slot's full width (docs/ctml.md S:15.2/S:15.5 -- shellregion.ctml's and
// shellwide.ctml's own header comments have the width contract in full), so
// Wide's main+inspector and every single-pane shape render through ONE
// go-html RenderTerm call (renderBandLayout) instead of
// lipgloss.JoinHorizontal/fitPane host composition. The ONE shape that
// still cannot join them -- Overlay with the inspector open, stacking two
// independently-sized panes vertically with a rule between -- has no HLCRF
// slot vocabulary to express it in (H/F are whole-page top/bottom bands,
// not a reusable mid-page pair), so it stays exactly as before this slice:
// shell.ctml's H/F-only layout, split by renderBandFrame, with
// renderInspectorStack composing the region between the bands -- the same
// HF+host-composition idiom a widget-carrying overlay uses for a live
// Bubbles widget between its own bands.
//
// The tab strip and session strip (both already fully styled, single-line,
// pre-fitted to metrics.innerWidth) join as the shell's one "header" value;
// the status/key-hint line is the "footer" value. Verbatim is emitted
// exactly as supplied with no added blank, so the "\n" join building header
// is the only seam deciding header/session adjacency -- matching the
// pre-.ctml JoinVertical's own row order byte-for-byte.
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

	var inside string
	switch {
	case metrics.kind == layoutOverlay && spec.InspectorOpen:
		region := renderInspectorStack(spec, metrics, styles)
		head, foot := renderBandFrame(shellCTML, metrics.innerWidth, shellFrameTheme(styles), shellBindings(header, footer))
		inside = lipgloss.JoinVertical(lipgloss.Left, head, region, foot)
	case metrics.kind == layoutWide:
		main := fitPane(spec.Main, metrics.mainWidth, metrics.mainHeight, styles.panel)
		inspector := fitPane(spec.Inspector, metrics.inspectorWidth, metrics.inspectorHeight, styles.inspector)
		inside = renderBandLayout(shellWideCTML, metrics.innerWidth, shellWideTheme(styles), shellWideBindings(header, footer, main, inspector))
	default: // layoutNarrow (either pane) and layoutOverlay with the inspector closed
		content, style := spec.Main, styles.panel
		width, height := metrics.mainWidth, metrics.mainHeight
		if metrics.kind == layoutNarrow && spec.InspectorOpen {
			content, style = spec.Inspector, styles.inspector
			width, height = metrics.inspectorWidth, metrics.inspectorHeight
		}
		fitted := fitPane(content, width, height, style)
		inside = renderBandLayout(shellRegionCTML, metrics.innerWidth, shellRegionTheme(styles), shellRegionBindings(header, footer, fitted))
	}

	return styles.outerFrame.
		Width(metrics.innerWidth).
		Height(metrics.innerHeight).
		MaxWidth(spec.Width).
		MaxHeight(spec.Height).
		Render(inside)
}

// renderInspectorStack composes the ONE region shape go-html's <layout>
// still cannot express in a single render: Overlay layouts (80-119 cols)
// with the inspector open, stacking the full-width inspector above a
// compact main-panel preview with a horizontal rule between. H/L/C/R/F is a
// closed, five-letter vocabulary -- H and F are whole-page top/bottom bands,
// not a reusable mid-page pair, and go-html's own automatic L/C/R vertical
// stacking triggers only below 80 columns (termStackThreshold, docs/ctml.md
// S:15.1), above this layout kind's own floor -- so two independently-sized
// panes stacked vertically here has no slot to bind through. This is the
// friction this slice reports rather than works around; it stays exactly
// the pre-.ctml host composition (lipgloss.JoinVertical + a manual rule
// line), called from renderFrame only for this one case.
func renderInspectorStack(spec frameSpec, metrics frameMetrics, styles uiStyles) string {
	inspector := fitPane(spec.Inspector, metrics.inspectorWidth, metrics.inspectorHeight, styles.inspector)
	separator := fitLine(core.Repeat("─", metrics.innerWidth), metrics.innerWidth, styles.separator)
	main := fitPane(spec.Main, metrics.mainWidth, metrics.mainHeight, styles.panel)
	return lipgloss.JoinVertical(lipgloss.Left, inspector, separator, main)
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
