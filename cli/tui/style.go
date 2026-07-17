// SPDX-Licence-Identifier: EUPL-1.2

// Package tui is the lem terminal UI (`lem tui`): a model picker over
// inference.Discover and a streaming chat over inference.TextModel.Chat,
// built on Bubble Tea + Bubbles + Lip Gloss.
package tui

import "github.com/charmbracelet/lipgloss"

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
