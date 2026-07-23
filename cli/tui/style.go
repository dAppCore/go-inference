// SPDX-Licence-Identifier: EUPL-1.2

// Package tui is the lem terminal UI (`lem tui`): a model picker over
// inference.Discover and a streaming chat over inference.TextModel.Chat,
// built on Bubble Tea + Bubbles + Lip Gloss.
package tui

import "dappco.re/go/render/display/tui/style"

// theme contains semantic colours rather than component-specific paint. Every
// colour is adaptive so the same hierarchy remains legible on light terminals.
type theme struct {
	name      string
	text      style.Paint
	muted     style.Paint
	border    style.Paint
	focus     style.Paint
	assistant style.Paint
	attention style.Paint
	success   style.Paint
	error     style.Paint
}

func midnightTheme() theme {
	return theme{
		name:      "midnight",
		text:      style.AdaptiveColor{Light: "#172033", Dark: "#E2E8F0"}.Resolve(true),
		muted:     style.AdaptiveColor{Light: "#475569", Dark: "#718096"}.Resolve(true),
		border:    style.AdaptiveColor{Light: "#94A3B8", Dark: "#334155"}.Resolve(true),
		focus:     style.AdaptiveColor{Light: "#007A89", Dark: "#67E8F9"}.Resolve(true),
		assistant: style.AdaptiveColor{Light: "#6D28D9", Dark: "#C4B5FD"}.Resolve(true),
		attention: style.AdaptiveColor{Light: "#92400E", Dark: "#FBBF24"}.Resolve(true),
		success:   style.AdaptiveColor{Light: "#047857", Dark: "#6EE7B7"}.Resolve(true),
		error:     style.AdaptiveColor{Light: "#BE123C", Dark: "#FB7185"}.Resolve(true),
	}
}

func themeForName(name string) theme {
	switch name {
	case "aurora":
		value := midnightTheme()
		value.name = "aurora"
		value.focus = style.AdaptiveColor{Light: "#047857", Dark: "#5EEAD4"}.Resolve(true)
		value.assistant = style.AdaptiveColor{Light: "#7E22CE", Dark: "#E879F9"}.Resolve(true)
		value.attention = style.AdaptiveColor{Light: "#B45309", Dark: "#FDE68A"}.Resolve(true)
		return value
	case "daylight":
		value := midnightTheme()
		value.name = "daylight"
		value.focus = style.AdaptiveColor{Light: "#0369A1", Dark: "#7DD3FC"}.Resolve(true)
		value.assistant = style.AdaptiveColor{Light: "#5B21B6", Dark: "#DDD6FE"}.Resolve(true)
		return value
	default:
		return midnightTheme()
	}
}

// uiStyles is owned by app. Components receive it explicitly, which lets a
// future preference change rebuild the complete visual language in place.
type uiStyles struct {
	theme theme

	title     style.Style
	user      style.Style
	answer    style.Style
	thought   style.Style
	err       style.Style
	status    style.Style
	accent    style.Style
	assistant style.Style
	attention style.Style
	success   style.Style

	brand       style.Style
	navActive   style.Style
	navInactive style.Style
	header      style.Style
	session     style.Style
	panel       style.Style
	inspector   style.Style
	footer      style.Style
	separator   style.Style
	outerFrame  style.Style
	inputBorder style.Style
}

func newUIStyles(t theme) uiStyles {
	return uiStyles{
		theme:       t,
		title:       style.New().Bold(true).Foreground(t.text),
		user:        style.New().Bold(true).Foreground(t.focus),
		answer:      style.New().Foreground(t.text),
		thought:     style.New().Italic(true).Foreground(t.muted),
		err:         style.New().Foreground(t.error),
		status:      style.New().Foreground(t.muted),
		accent:      style.New().Foreground(t.focus),
		assistant:   style.New().Bold(true).Foreground(t.assistant),
		attention:   style.New().Bold(true).Foreground(t.attention),
		success:     style.New().Foreground(t.success),
		brand:       style.New().Bold(true).Foreground(t.focus),
		navActive:   style.New().Bold(true).Foreground(t.text).Underline(true).UnderlineSpaces(false),
		navInactive: style.New().Foreground(t.muted),
		header:      style.New().Foreground(t.text),
		session:     style.New().Foreground(t.muted),
		panel:       style.New().Foreground(t.text),
		inspector:   style.New().Foreground(t.muted),
		footer:      style.New().Foreground(t.muted),
		separator:   style.New().Foreground(t.border),
		outerFrame:  style.New().Border(style.Rounded()).BorderForeground(t.border),
		inputBorder: style.New().Border(style.Rounded()).BorderForeground(t.border).Padding(0, 1),
	}
}
