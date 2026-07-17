// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"

	core "dappco.re/go"
)

// The Settings tab: a cursor list of knobs adjusted with ←/→ (or h/l). Values
// apply to the NEXT load (context) or the next turn (the rest) — the honest
// scope of each knob is stated in its hint.

type settings struct {
	cursor    int
	ctxIdx    int // index into ctxSteps; 0 = model default (applies at load)
	maxTokIdx int // index into maxTokSteps (applies per turn)
	thinkIdx  int // 0 model default · 1 on · 2 off (applies per turn)
}

var (
	ctxSteps    = []int{0, 4096, 6144, 8192, 16384, 32768}
	maxTokSteps = []int{512, 1024, 2048, 4096, 8192}
	thinkNames  = []string{"model default", "on", "off"}
)

func newSettings() settings {
	return settings{maxTokIdx: 3} // 4096 — thinking spends from the budget
}

func (s settings) contextLen() int { return ctxSteps[s.ctxIdx] }
func (s settings) maxTokens() int  { return maxTokSteps[s.maxTokIdx] }

// thinking returns the per-turn override: nil = the model family default.
func (s settings) thinking() *bool {
	switch s.thinkIdx {
	case 1:
		on := true
		return &on
	case 2:
		off := false
		return &off
	}
	return nil
}

type settingRow struct {
	name  string
	value string
	hint  string
}

func (s settings) rows() []settingRow {
	ctx := "model default"
	if s.contextLen() > 0 {
		ctx = core.Sprintf("%d", s.contextLen())
	}
	return []settingRow{
		{"context length", ctx, "KV cache size — applies when a model is (re)loaded from Models"},
		{"max tokens", core.Sprintf("%d", s.maxTokens()), "per-reply budget — thinking spends from it"},
		{"thinking", thinkNames[s.thinkIdx], "the reasoning channel — gemma4's family default is on"},
	}
}

// adjust moves the value under the cursor by delta (wrapping).
func (s settings) adjust(delta int) settings {
	wrap := func(i, n int) int { return ((i+delta)%n + n) % n }
	switch s.cursor {
	case 0:
		s.ctxIdx = wrap(s.ctxIdx, len(ctxSteps))
	case 1:
		s.maxTokIdx = wrap(s.maxTokIdx, len(maxTokSteps))
	case 2:
		s.thinkIdx = wrap(s.thinkIdx, len(thinkNames))
	}
	return s
}

func (s settings) move(delta int) settings {
	n := len(s.rows())
	s.cursor = ((s.cursor+delta)%n + n) % n
	return s
}

func (s settings) view(width int, styles uiStyles) string {
	var b strings.Builder
	b.WriteString(styles.title.Render("settings") + "\n\n")
	for i, row := range s.rows() {
		cursor := "  "
		name := styles.answer.Render(row.name)
		if i == s.cursor {
			cursor = styles.accent.Render("› ")
			name = styles.accent.Render(row.name)
		}
		b.WriteString(cursor + name + "  " + styles.title.Render("‹ "+row.value+" ›") + "\n")
		b.WriteString("    " + styles.thought.Render(row.hint) + "\n\n")
	}
	b.WriteString(styles.status.Render("↑/↓ select · ←/→ change · values apply as hinted"))
	return b.String()
}
