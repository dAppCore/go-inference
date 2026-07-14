// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"

	"dappco.re/go/inference"
)

// The Modes tab: sampling presets applied to every turn. "Balanced" leaves the
// checkpoint's own declared sampling defaults in charge (the declares-
// discipline default); the others pin explicit values.

type mode struct {
	name string
	hint string
	opts func() []inference.GenerateOption
}

var modes = []mode{
	{"Balanced", "the checkpoint's declared sampling defaults — no overrides", func() []inference.GenerateOption { return nil }},
	{"Greedy", "temperature 0 — deterministic, benchmark-comparable", func() []inference.GenerateOption {
		return []inference.GenerateOption{inference.WithTemperature(0)}
	}},
	{"Creative", "temperature 1.2, top-p 0.95 — wander", func() []inference.GenerateOption {
		return []inference.GenerateOption{inference.WithTemperature(1.2), inference.WithTopP(0.95)}
	}},
	{"Coder", "temperature 0.2, thinking off — terse and literal", func() []inference.GenerateOption {
		off := false
		return []inference.GenerateOption{inference.WithTemperature(0.2), inference.WithEnableThinking(&off)}
	}},
}

type modeState struct{ selected int }

func (m modeState) current() mode { return modes[m.selected] }

func (m modeState) move(delta int) modeState {
	n := len(modes)
	m.selected = ((m.selected+delta)%n + n) % n
	return m
}

func (m modeState) view(width int) string {
	var b strings.Builder
	b.WriteString(styleTitle.Render("modes") + "\n\n")
	for i, md := range modes {
		cursor, name := "  ", styleAnswer.Render(md.name)
		if i == m.selected {
			cursor, name = styleAccent.Render("› "), styleAccent.Render(md.name)
		}
		b.WriteString(cursor + name + "\n    " + styleThought.Render(md.hint) + "\n\n")
	}
	b.WriteString(styleStatus.Render("↑/↓ select — applies to every following turn"))
	return b.String()
}
