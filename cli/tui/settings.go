// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	_ "embed"

	"github.com/charmbracelet/lipgloss"

	core "dappco.re/go"
	"dappco.re/go/html"
	"dappco.re/go/html/ctml"
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

func (s settings) withPreferenceValues(values preferenceValues) settings {
	for index, value := range ctxSteps {
		if value == values.ContextLength {
			s.ctxIdx = index
			break
		}
	}
	for index, value := range maxTokSteps {
		if value == values.MaxTokens {
			s.maxTokIdx = index
			break
		}
	}
	switch values.Thinking {
	case "on":
		s.thinkIdx = 1
	case "off":
		s.thinkIdx = 2
	default:
		s.thinkIdx = 0
	}
	return s
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

// settingsCTML is the Settings form's markup — see settings.ctml for the
// seams it exposes (row sequences, class tokens, the settings-form block id).
//
//go:embed settings.ctml
var settingsCTML []byte

// settingsFormBindings binds ONE row per knob — selection styling rides the
// row-scoped class bind (class="{{row.state}}", go-html v0.13.0) and the
// marker glyph rides the row, so no before/active/after sequence split is
// needed. A form this size re-binds on every change for free.
func settingsFormBindings(form settings) ctml.Bindings {
	rows := make([]map[string]any, 0, len(form.rows()))
	for index, row := range form.rows() {
		state, marker := "row-idle", "○"
		if index == form.cursor {
			state, marker = "row-active", "›"
		}
		rows = append(rows, map[string]any{
			"state": state, "marker": marker,
			"name": row.name, "value": row.value, "hint": row.hint,
		})
	}
	return ctml.Bindings{Sequences: map[string][]map[string]any{"rows": rows}}
}

// settingsFormTheme maps the markup's class tokens onto the existing palette,
// so the .ctml render reuses uiStyles paint exactly — no colours of its own.
func settingsFormTheme(styles uiStyles) *html.TermTheme {
	theme := html.DefaultTermTheme()
	theme.Text = styles.answer
	theme.Heading = styles.title // the <h2> form title
	theme.Classes = map[string]lipgloss.Style{
		"row-idle":   styles.answer,
		"row-active": styles.accent,
		"row-value":  styles.title,
		"row-hint":   styles.thought,
		"form-keys":  styles.status,
	}
	return theme
}

// renderSettings parses settings.ctml with the current row bindings and
// renders it through the go-html terminal renderer: the form title, one
// <dl> row per knob (value line + indented hint), and the key footer.
func renderSettings(form settings, width int, styles uiStyles) string {
	if width <= 0 {
		return ""
	}
	tree, err := ctml.Parse(settingsCTML, settingsFormBindings(form))
	if err != nil {
		// settings.ctml is embedded and static, so a parse failure is a build
		// defect; TestRenderSettings_Good pins the markup as parseable.
		return ""
	}
	return html.RenderTerm(tree, html.NewContext(), html.TermOptions{Width: width, Theme: settingsFormTheme(styles)})
}
