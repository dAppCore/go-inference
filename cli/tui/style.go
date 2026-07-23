// SPDX-Licence-Identifier: EUPL-1.2

// Package tui is the lem terminal UI (`lem tui`): a model picker over
// inference.Discover and a streaming chat over inference.TextModel.Chat,
// built on Bubble Tea + Bubbles + Lip Gloss.
package tui

import (
	termlipgloss "charm.land/lipgloss/v2"
	"github.com/charmbracelet/lipgloss"
)

// theme contains semantic colours rather than component-specific paint. Every
// colour is adaptive so the same hierarchy remains legible on light terminals.
type theme struct {
	name      string
	text      lipgloss.AdaptiveColor
	muted     lipgloss.AdaptiveColor
	border    lipgloss.AdaptiveColor
	focus     lipgloss.AdaptiveColor
	assistant lipgloss.AdaptiveColor
	attention lipgloss.AdaptiveColor
	success   lipgloss.AdaptiveColor
	error     lipgloss.AdaptiveColor
}

func midnightTheme() theme {
	return theme{
		name:      "midnight",
		text:      lipgloss.AdaptiveColor{Light: "#172033", Dark: "#E2E8F0"},
		muted:     lipgloss.AdaptiveColor{Light: "#475569", Dark: "#718096"},
		border:    lipgloss.AdaptiveColor{Light: "#94A3B8", Dark: "#334155"},
		focus:     lipgloss.AdaptiveColor{Light: "#007A89", Dark: "#67E8F9"},
		assistant: lipgloss.AdaptiveColor{Light: "#6D28D9", Dark: "#C4B5FD"},
		attention: lipgloss.AdaptiveColor{Light: "#92400E", Dark: "#FBBF24"},
		success:   lipgloss.AdaptiveColor{Light: "#047857", Dark: "#6EE7B7"},
		error:     lipgloss.AdaptiveColor{Light: "#BE123C", Dark: "#FB7185"},
	}
}

func themeForName(name string) theme {
	switch name {
	case "aurora":
		value := midnightTheme()
		value.name = "aurora"
		value.focus = lipgloss.AdaptiveColor{Light: "#047857", Dark: "#5EEAD4"}
		value.assistant = lipgloss.AdaptiveColor{Light: "#7E22CE", Dark: "#E879F9"}
		value.attention = lipgloss.AdaptiveColor{Light: "#B45309", Dark: "#FDE68A"}
		return value
	case "daylight":
		value := midnightTheme()
		value.name = "daylight"
		value.focus = lipgloss.AdaptiveColor{Light: "#0369A1", Dark: "#7DD3FC"}
		value.assistant = lipgloss.AdaptiveColor{Light: "#5B21B6", Dark: "#DDD6FE"}
		return value
	default:
		return midnightTheme()
	}
}

// uiStyles is owned by app. Components receive it explicitly, which lets a
// future preference change rebuild the complete visual language in place.
type uiStyles struct {
	theme theme

	title     lipgloss.Style
	user      lipgloss.Style
	answer    lipgloss.Style
	thought   lipgloss.Style
	err       lipgloss.Style
	status    lipgloss.Style
	accent    lipgloss.Style
	assistant lipgloss.Style
	attention lipgloss.Style
	success   lipgloss.Style

	brand       lipgloss.Style
	navActive   lipgloss.Style
	navInactive lipgloss.Style
	header      lipgloss.Style
	session     lipgloss.Style
	panel       lipgloss.Style
	inspector   lipgloss.Style
	footer      lipgloss.Style
	separator   lipgloss.Style
	outerFrame  lipgloss.Style
	inputBorder lipgloss.Style
}

func newUIStyles(t theme) uiStyles {
	return uiStyles{
		theme:       t,
		title:       lipgloss.NewStyle().Bold(true).Foreground(t.text),
		user:        lipgloss.NewStyle().Bold(true).Foreground(t.focus),
		answer:      lipgloss.NewStyle().Foreground(t.text),
		thought:     lipgloss.NewStyle().Italic(true).Foreground(t.muted),
		err:         lipgloss.NewStyle().Foreground(t.error),
		status:      lipgloss.NewStyle().Foreground(t.muted),
		accent:      lipgloss.NewStyle().Foreground(t.focus),
		assistant:   lipgloss.NewStyle().Bold(true).Foreground(t.assistant),
		attention:   lipgloss.NewStyle().Bold(true).Foreground(t.attention),
		success:     lipgloss.NewStyle().Foreground(t.success),
		brand:       lipgloss.NewStyle().Bold(true).Foreground(t.focus),
		navActive:   lipgloss.NewStyle().Bold(true).Foreground(t.text).Underline(true).UnderlineSpaces(false),
		navInactive: lipgloss.NewStyle().Foreground(t.muted),
		header:      lipgloss.NewStyle().Foreground(t.text),
		session:     lipgloss.NewStyle().Foreground(t.muted),
		panel:       lipgloss.NewStyle().Foreground(t.text),
		inspector:   lipgloss.NewStyle().Foreground(t.muted),
		footer:      lipgloss.NewStyle().Foreground(t.muted),
		separator:   lipgloss.NewStyle().Foreground(t.border),
		outerFrame:  lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(t.border),
		inputBorder: lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(t.border).Padding(0, 1),
	}
}

// termStyle adapts the v1 styles owned by the Bubble Tea shell to the v2
// styles required by go-render's terminal theme. It is intentionally a
// renderer-boundary copy: the shell continues to own and render v1 styles.
func termStyle(source lipgloss.Style) termlipgloss.Style {
	target := termlipgloss.NewStyle()
	if colour := source.GetForeground(); hasTermColour(colour) {
		target = target.Foreground(colour)
	}
	if source.GetBold() {
		target = target.Bold(true)
	}
	if source.GetItalic() {
		target = target.Italic(true)
	}
	if source.GetUnderline() {
		target = target.Underline(true).UnderlineSpaces(source.GetUnderlineSpaces())
	}

	if top, right, bottom, left := source.GetPadding(); top != 0 || right != 0 || bottom != 0 || left != 0 {
		target = target.Padding(top, right, bottom, left)
	}
	if border, top, right, bottom, left := source.GetBorder(); border != (lipgloss.Border{}) {
		target = target.Border(termlipgloss.Border(border), top, right, bottom, left)
	}
	if colour := source.GetBorderTopForeground(); hasTermColour(colour) {
		target = target.BorderTopForeground(colour)
	}
	if colour := source.GetBorderRightForeground(); hasTermColour(colour) {
		target = target.BorderRightForeground(colour)
	}
	if colour := source.GetBorderBottomForeground(); hasTermColour(colour) {
		target = target.BorderBottomForeground(colour)
	}
	if colour := source.GetBorderLeftForeground(); hasTermColour(colour) {
		target = target.BorderLeftForeground(colour)
	}
	return target
}

func termStyles(styles map[string]lipgloss.Style) map[string]termlipgloss.Style {
	converted := make(map[string]termlipgloss.Style, len(styles))
	for name, style := range styles {
		converted[name] = termStyle(style)
	}
	return converted
}

// termOutput applies Lip Gloss v2's output-layer profile conversion before a
// go-render fragment enters the Bubble Tea v1 shell.
func termOutput(rendered string) string {
	return termlipgloss.Sprint(rendered)
}

func hasTermColour(colour lipgloss.TerminalColor) bool {
	switch colour.(type) {
	case nil, lipgloss.NoColor, *lipgloss.NoColor:
		return false
	default:
		return true
	}
}
