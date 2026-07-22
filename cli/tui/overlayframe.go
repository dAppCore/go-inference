// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"github.com/charmbracelet/lipgloss"

	core "dappco.re/go"
	"dappco.re/go/html"
	"dappco.re/go/html/ctml"
)

// The overlay layer renders its text chrome through .ctml <layout>
// documents (HLCRF). Two idioms, chosen per overlay and noted in each
// file's header comment:
//
//   - An all-text overlay is a full <layout variant="HCF">: the title
//     band, the content region, and the key-hint footer all render in one
//     RenderTerm call (databulk.ctml, launchreview.ctml).
//   - A widget-carrying overlay is a <layout variant="HF">: live Bubbles
//     widgets (textinput/textarea/viewport) emit pre-styled ANSI, which
//     cannot ride a .ctml document, so the layout renders the header and
//     footer bands and the host composes the widgets between them —
//     exactly as the transcript composes Glamour output around
//     ctml-rendered chrome.
//
// renderOverlayFrame is the HF idiom's seam: it renders the layout once
// through RenderTermBoxes and splits the output at the H slot's own
// recorded box height — the renderer's receipt for where the header band
// ends — so the host never re-measures rendered chrome.
func renderOverlayFrame(src []byte, width int, styles uiStyles, bindings ...ctml.Bindings) (head, foot string) {
	layout, err := ctml.ParseLayout(src, bindings...)
	if err != nil {
		// Overlay markup is embedded and static, so a parse failure is a
		// build defect; the TestRender<Overlay>_Good tests pin each file
		// as parseable.
		return "", ""
	}
	rendered, boxes := layout.RenderTermBoxes(html.NewContext(), html.TermOptions{Width: width, Theme: overlayFrameTheme(styles)})
	lines := core.Split(rendered, "\n")
	split := min(boxes["H"].Height, len(lines))
	return core.Join("\n", lines[:split]...), core.Join("\n", lines[split:]...)
}

// renderOverlayLayout is the HCF idiom's seam: one RenderTerm call for an
// overlay whose every region is text.
func renderOverlayLayout(src []byte, width int, styles uiStyles, bindings ...ctml.Bindings) string {
	layout, err := ctml.ParseLayout(src, bindings...)
	if err != nil {
		// Embedded static markup — a parse failure is a build defect (see
		// renderOverlayFrame).
		return ""
	}
	return layout.RenderTerm(html.NewContext(), html.TermOptions{Width: width, Theme: overlayFrameTheme(styles)})
}

// overlayFrameTheme maps overlay markup onto the existing palette, so the
// .ctml renders reuse uiStyles paint exactly — no colours of their own.
// The layout bands are stripped of the default theme's borders: the
// floating overlay box (renderOverlay) already draws the frame. The
// footer band keeps one row of top padding — the blank line every overlay
// draws above its key hints — so that spacing lives in the theme, not in
// host composition.
func overlayFrameTheme(styles uiStyles) *html.TermTheme {
	theme := html.DefaultTermTheme()
	theme.Text = styles.answer
	theme.Heading = styles.title // the <h2> overlay titles
	theme.Header = lipgloss.NewStyle()
	theme.Footer = lipgloss.NewStyle().Padding(1, 0, 0, 0)
	theme.Classes = map[string]lipgloss.Style{
		"overlay-hint":  styles.thought,
		"overlay-warn":  styles.attention,
		"overlay-error": styles.err,
		"overlay-keys":  styles.status,
	}
	return theme
}
