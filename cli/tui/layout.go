// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	_ "embed"

	core "dappco.re/go"
	"dappco.re/go/render/display/tui/style"
	"dappco.re/go/render/engine/ctml"
	"dappco.re/go/render/engine/html"
)

type layoutKind uint8

const (
	layoutNarrow layoutKind = iota
	layoutOverlay
	layoutWide
)

// wideInspectorWidth is the width REQUESTED for the R slot (the inspector)
// in the Wide layout kind's side-by-side band, via TermOptions.AsideWidth
// (docs/ctml.md S:15.1, go-html v0.15.0) -- restoring both the pre-.ctml
// value AND name (git history: layout.go's renderWorkspaceRegion, before
// slice 7's shell.ctml conversion dropped to go-html's unrequested default).
// go-html's own default R budget when no request is made is a fixed 28 (the
// unexported termAsideWidth constant, docs/ctml.md S:15.5: "the budgets stay
// unexported"); AsideWidth overrides it per render rather than changing
// what go-html ships, so this constant is the caller's REQUEST, not a
// mirrored upstream budget -- go-html sizes C (the main panel) to absorb
// the difference. TestWideInspectorWidth_MatchesRequest pins the request
// against a live RenderTermBoxes call on the real shellwide.ctml render
// path (the box map is the render-time source of truth, docs/ctml.md
// S:15.5) so any upstream regression in honouring AsideWidth fails loudly
// here instead of silently reflowing the Wide frame back to 28.
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
		// -1: go-html's own C/R gutter column (S:15.1), now painted with the
		// GutterRule "│" (S:15.6, shellWideTheme) instead of left blank.
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

// shellCTML is the app shell's H/F-only markup for the ONE shape whose
// header+region+footer still cannot join into a single render call --
// Overlay layouts with the inspector open -- see shell.ctml for the seams
// it exposes and renderFrame's own doc comment for why that one join stays
// host-side even though the region between the bands (renderInspectorStack)
// no longer hand-composes.
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
	theme.Header = style.New()
	theme.Footer = style.New()
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
	theme.Content = style.New()
	return theme
}

// shellWideTheme extends shellRegionTheme with a zero-chrome Aside (R) slot
// for the Wide layout's side-by-side main+inspector: both C and R arrive
// pre-fitted to their own exact width by renderFrame, so both slots' default
// chrome is stripped (Aside was already themeable pre-v0.14.0; Content
// joined it in round 4, S:15.2). GutterRule repaints the C/R junction column
// go-html always reserves (S:15.1) with the historic "│" divider instead of
// leaving it blank (S:15.6, go-html v0.15.0); Rule is repointed at
// styles.separator (Foreground(t.border)) so the glyph matches the exact
// colour the pre-.ctml renderWorkspaceRegion's own styles.separator pane
// drew (git history), rather than go-html's own similar-but-different
// default border colour.
func shellWideTheme(styles uiStyles) *html.TermTheme {
	theme := shellRegionTheme(styles)
	theme.Aside = style.New()
	theme.GutterRule = "│"
	theme.Rule = styles.separator
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
// (metrics.mainWidth, wideInspectorWidth) x metrics.regionHeight.
func shellWideBindings(header, footer, main, inspector string) ctml.Bindings {
	return ctml.Bindings{Values: map[string]any{
		"header": header, "footer": footer, "main": main, "inspector": inspector,
	}}
}

// renderWideLayout is renderBandLayout's Wide-shape sibling (layout.go only
// -- overlayframe.go's shared renderBandLayout stays untouched, since every
// other caller renders at go-html's default side-slot widths): it
// additionally requests AsideWidth so R renders at wideInspectorWidth
// instead of go-html's unrequested 28-column default (docs/ctml.md S:15.1,
// go-html v0.15.0) -- the one render path in this package that needs a
// non-default side-slot width.
func renderWideLayout(src []byte, width int, theme *html.TermTheme, bindings ...ctml.Bindings) string {
	layout, err := ctml.ParseLayout(src, bindings...)
	if err != nil {
		// Embedded static markup -- a parse failure is a build defect (see
		// renderBandFrame, overlayframe.go).
		return ""
	}
	return layout.RenderTerm(html.NewContext(), html.TermOptions{Width: width, Theme: theme, AsideWidth: wideInspectorWidth})
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
// go-html RenderTerm call (renderBandLayout/renderWideLayout) instead of
// lipgloss.JoinHorizontal/fitPane host composition. Wide's R additionally
// rides go-html v0.15.0's AsideWidth request (S:15.1, renderWideLayout) for
// its historic 32-column inspector, and its C/R junction repaints go-html's
// always-reserved gutter column with the historic "│" rule (S:15.6,
// shellWideTheme) instead of leaving it blank.
//
// The ONE shape that still cannot join header+region+footer into a single
// call -- Overlay with the inspector open -- stays a host-side
// style.Column of three independently rendered pieces: shell.ctml's
// H/F-only layout (split by renderBandFrame, unchanged), and
// renderInspectorStack's own region render sat between them. What changed
// this slice (docs/ctml.md S:15.7, go-html v0.15.0) is that the REGION
// itself is no longer hand-stacked: renderInspectorStack now renders a
// genuine go-html "HC" layout (shellinspectorpair.ctml) -- H the inspector,
// its own bottom border the divider rule, C the main preview -- through its
// OWN RenderTerm call with its own theme. It cannot join the OUTER
// header/footer render into that same call: TermTheme is one flat struct
// threaded through every nested Layout in a single render (term_layout.go's
// termRenderer carries exactly one theme field), so a unified call would
// force the page header's H (needing zero chrome, pre-fitted byte-exact
// like every sibling shell) and the pair's own H (needing its natural
// bordered chrome, since that border IS the rule) to share one Header
// style -- confirmed by a throwaway probe during this slice (not part of
// the shipped diff): zeroing Header for the page header silently zeroed the
// pair's divider too. See shellinspectorpair.ctml's header comment and
// renderInspectorStack's own comment for the full account.
//
// The tab strip and the composed SESSIONS action/chip band (both single-line,
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
	sessionLine := styles.title.Render("SESSIONS") +
		"  " + styles.accent.Render("●") + " " + styles.session.Render("New session")
	if core.Trim(spec.SessionStrip) != "" {
		chipStyle := styles.session
		if core.Trim(spec.SessionStrip) == "session 1" {
			chipStyle = styles.accent
		}
		sessionLine += "  " + styles.separator.Render("·") + "  " + chipStyle.Render(spec.SessionStrip)
	}
	sessions := fitLine(sessionLine, metrics.innerWidth, styles.session)
	header := core.Join("\n", tabstrip, sessions)
	footer := fitLine(spec.Footer, metrics.innerWidth, styles.footer)

	var inside string
	switch {
	case metrics.kind == layoutOverlay && spec.InspectorOpen:
		region := renderInspectorStack(spec, metrics, styles)
		head, foot := renderBandFrame(shellCTML, metrics.innerWidth, shellFrameTheme(styles), shellBindings(header, footer))
		inside = style.Column(style.Left, head, region, foot)
	case metrics.kind == layoutWide:
		main := fitPane(spec.Main, metrics.mainWidth, metrics.mainHeight, styles.panel)
		inspector := fitPane(spec.Inspector, metrics.inspectorWidth, metrics.inspectorHeight, styles.inspector)
		inside = renderWideLayout(shellWideCTML, metrics.innerWidth, shellWideTheme(styles), shellWideBindings(header, footer, main, inspector))
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
		Width(spec.Width).
		Height(spec.Height).
		MaxWidth(spec.Width).
		MaxHeight(spec.Height).
		Render(inside)
}

// shellInspectorPairCTML is the mid-page vertical pane pair markup for the
// Overlay layout kind (80-119 cols) with the inspector open -- see
// shellinspectorpair.ctml for the seams it exposes and why it renders
// standalone rather than nested inside shell.ctml's own H/F bands.
//
//go:embed shellinspectorpair.ctml
var shellInspectorPairCTML []byte

// inspectorPairTheme keeps Header's default bottom border -- the S:15.7
// divider rule -- but strips its bold/violet text styling and (0,1) padding
// so the inspector's own pre-fitted, pre-styled content (fitPane,
// styles.inspector) passes through unmodified except for gaining that one
// extra border row; Content is zeroed the same way shellRegionTheme zeroes
// it, so the main preview passes through byte-exact too. BorderForeground
// is repointed at styles.theme.border (the same colour styles.separator
// itself paints with) so the divider matches the exact shade the
// pre-.ctml renderWorkspaceRegion's own styles.separator pane drew (git
// history), not go-html's own similar-but-different default border colour.
func inspectorPairTheme(styles uiStyles) *html.TermTheme {
	theme := html.DefaultTermTheme()
	theme.Header = style.New().
		Border(style.Normal(), false, false, true, false).
		BorderForeground(styles.theme.border)
	theme.Content = style.New()
	return theme
}

// inspectorPairBindings supplies shellinspectorpair.ctml's two verbatim
// values. Both arrive pre-fitted by renderInspectorStack to their own exact
// target width x height (metrics.inspectorWidth/Height for inspector,
// metrics.mainWidth/Height for main).
func inspectorPairBindings(inspector, main string) ctml.Bindings {
	return ctml.Bindings{Values: map[string]any{"inspector": inspector, "main": main}}
}

// renderInspectorStack composes the region for Overlay layouts (80-119
// cols) with the inspector open: the full-width inspector stacked above a
// compact main-panel preview, with a rule between. go-html v0.15.0's
// documented mid-page vertical-pair idiom (docs/ctml.md S:15.7) replaces the
// old lipgloss.JoinVertical plus a manually core.Repeat("─", ...) rule line
// with one go-html "HC" layout render (shellinspectorpair.ctml): H the
// inspector -- its own bottom border the divider -- C the main preview.
//
// This still renders through its OWN RenderTerm call rather than nesting
// inside shell.ctml's H/F bands in one page-wide call: see
// shellinspectorpair.ctml's header comment and renderFrame's own doc
// comment for why a single unified call cannot give the page header's H
// (zero-chrome) and this pair's own H (bordered) different styles from the
// one flat TermTheme a render call threads through every nested Layout.
// renderFrame still joins this function's output between the (unchanged,
// separately rendered) header/footer bands with style.Column --
// that outer join, not this region's own composition, is what remains
// host-side.
func renderInspectorStack(spec frameSpec, metrics frameMetrics, styles uiStyles) string {
	inspector := fitPane(spec.Inspector, metrics.inspectorWidth, metrics.inspectorHeight, styles.inspector)
	main := fitPane(spec.Main, metrics.mainWidth, metrics.mainHeight, styles.panel)
	return renderBandLayout(shellInspectorPairCTML, metrics.innerWidth, inspectorPairTheme(styles), inspectorPairBindings(inspector, main))
}

func fitLine(content string, width int, style style.Style) string {
	if width <= 0 {
		return ""
	}
	return style.Width(width).MaxWidth(width).MaxHeight(1).Render(content)
}

func fitPane(content string, width, height int, style style.Style) string {
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
