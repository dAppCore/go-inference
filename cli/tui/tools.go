// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"time"

	core "dappco.re/go"
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
				return core.Sprintf("%d words", len(strings.Fields(text)))
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
			if strings.TrimSpace(call.ArgumentsJSON) != "" {
				_ = core.JSONUnmarshal([]byte(call.ArgumentsJSON), &args)
			}
			out := bt.run(args)
			t.lastRun = append(t.lastRun, call.Name+" → "+out)
			return out
		}
	}
	t.lastRun = append(t.lastRun, call.Name+" → (unknown tool)")
	return "error: unknown tool " + call.Name
}

func (t toolState) view(width int, styles uiStyles) string {
	var b strings.Builder
	b.WriteString(styles.title.Render("tools") + "\n\n")
	state := "disabled — replies are plain chat"
	if t.enabled {
		state = "enabled — declarations ride the system turn; calls run locally and feed back"
	}
	b.WriteString("  " + styles.accent.Render("function calling: ") + styles.answer.Render(state) + "\n\n")
	for _, bt := range t.tools {
		b.WriteString("  " + styles.answer.Render(bt.decl.Name) + "  " + styles.thought.Render(bt.decl.Description) + "\n")
	}
	if len(t.lastRun) > 0 {
		b.WriteString("\n" + styles.title.Render("recent calls") + "\n")
		start := len(t.lastRun) - 5
		if start < 0 {
			start = 0
		}
		for _, r := range t.lastRun[start:] {
			b.WriteString("  " + styles.thought.Render(r) + "\n")
		}
	}
	b.WriteString("\n" + styles.status.Render("enter toggles · calls appear dim in the chat transcript"))
	return b.String()
}
