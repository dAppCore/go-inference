// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	_ "embed"
	"time"

	"github.com/charmbracelet/lipgloss"

	core "dappco.re/go"
	"dappco.re/go/html"
	"dappco.re/go/html/ctml"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/parser"
)

// The Tools tab: built-in demo tools the model can call. When enabled, the
// chat injects RenderGemmaToolDeclarations as the system turn; a completed
// reply is parsed with ParseGemmaToolCalls, each call executed locally, the
// result handed back via RenderGemmaToolResponse, and the turn auto-continues
// — the full agent round trip, live in the terminal.

type builtinTool struct {
	decl parser.ToolDecl
	run  func(args map[string]any) string
}

type toolState struct {
	enabled bool
	cursor  int
	tools   []builtinTool
	lastRun []string // rendered call receipts for the tab view
}

func newTools() toolState {
	return toolState{tools: []builtinTool{
		{
			decl: parser.ToolDecl{
				Name:        "get_time",
				Description: "Get the current local date and time.",
			},
			run: func(map[string]any) string {
				return time.Now().Format("Monday 2 January 2006, 15:04:05 MST")
			},
		},
		{
			decl: parser.ToolDecl{
				Name:        "word_count",
				Description: "Count the words in the given text.",
				Properties: map[string]parser.ToolParam{
					"text": {Type: "string", Description: "the text to count"},
				},
				Required: []string{"text"},
			},
			run: func(args map[string]any) string {
				text, _ := args["text"].(string)
				return core.Sprintf("%d words", len(core.Fields(text)))
			},
		},
	}}
}

func (t *toolState) setEnabled(enabled bool) {
	if t != nil {
		t.enabled = enabled
	}
}

func (t *toolState) toggle() {
	if t != nil {
		t.enabled = !t.enabled
	}
}

// declarations renders the system-turn tool preamble for the enabled set.
func (t toolState) declarations() string {
	if !t.enabled {
		return ""
	}
	decls := make([]parser.ToolDecl, len(t.tools))
	for i, bt := range t.tools {
		decls[i] = bt.decl
	}
	return parser.RenderGemmaToolDeclarations(decls)
}

// execute runs one parsed call against the built-in set; unknown tools get an
// honest error string the model can recover from.
func (t *toolState) execute(call inference.ToolCall) string {
	for _, bt := range t.tools {
		if bt.decl.Name == call.Name {
			args := map[string]any{}
			if core.Trim(call.ArgumentsJSON) != "" {
				if decoded := core.JSONUnmarshal([]byte(call.ArgumentsJSON), &args); !decoded.OK {
					args = map[string]any{}
				}
			}
			out := bt.run(args)
			t.lastRun = append(t.lastRun, call.Name+" → "+out)
			return out
		}
	}
	t.lastRun = append(t.lastRun, call.Name+" → (unknown tool)")
	return "error: unknown tool " + call.Name
}

// toolsCTML is the Tools tab's markup — see tools.ctml for the seams it
// exposes (the state/tools/recent sequences, class tokens, the tools-panel
// block id).
//
//go:embed tools.ctml
var toolsCTML []byte

// toolsPanelBindings binds the tab's dynamic content: the enabled-state
// text (an always-present lone scalar riding Bindings.Values — the one-row
// sequence it rode before Values existed is retired), one row per built-in
// tool, and — only when calls have run — the "recent calls" heading (a
// zero-or-one-row sequence, the conditional-section idiom) with the last
// five receipts. The two-cell gutter between a tool name and its
// description travels in the bound desc value, because a whitespace-only
// source run between sibling spans drops in the parser.
func toolsPanelBindings(state toolState) ctml.Bindings {
	stateText := "disabled — replies are plain chat"
	if state.enabled {
		stateText = "enabled — declarations ride the system turn; calls run locally and feed back"
	}
	sequences := map[string][]map[string]any{
		"tools":       {},
		"recentTitle": {},
		"recent":      {},
	}
	for _, tool := range state.tools {
		sequences["tools"] = append(sequences["tools"], map[string]any{
			"name": tool.decl.Name,
			"desc": "  " + tool.decl.Description,
		})
	}
	if len(state.lastRun) > 0 {
		sequences["recentTitle"] = append(sequences["recentTitle"], map[string]any{})
		start := max(0, len(state.lastRun)-5)
		for _, receipt := range state.lastRun[start:] {
			sequences["recent"] = append(sequences["recent"], map[string]any{"receipt": receipt})
		}
	}
	return ctml.Bindings{Sequences: sequences, Values: map[string]any{"state": stateText}}
}

// toolsPanelTheme maps the markup's class tokens onto the existing palette,
// so the .ctml render reuses uiStyles paint exactly — no colours of its own.
func toolsPanelTheme(styles uiStyles) *html.TermTheme {
	theme := html.DefaultTermTheme()
	theme.Text = styles.answer
	theme.Heading = styles.title // the <h2> section titles
	theme.Classes = map[string]lipgloss.Style{
		"tool-label":   styles.accent,
		"tool-state":   styles.answer,
		"tool-name":    styles.answer,
		"tool-desc":    styles.thought,
		"tool-receipt": styles.thought,
		"tool-keys":    styles.status,
	}
	return theme
}

// renderTools parses tools.ctml with the current tool bindings and renders
// it through the go-html terminal renderer: the tab title, the
// function-calling state line, one line per built-in tool, the recent-call
// receipts when any exist, and the key footer.
func renderTools(state toolState, width int, styles uiStyles) string {
	if width <= 0 {
		return ""
	}
	tree, err := ctml.Parse(toolsCTML, toolsPanelBindings(state))
	if err != nil {
		// tools.ctml is embedded and static, so a parse failure is a build
		// defect; TestRenderTools_Good pins the markup as parseable.
		return ""
	}
	return html.RenderTerm(tree, html.NewContext(), html.TermOptions{Width: width, Theme: toolsPanelTheme(styles)})
}
