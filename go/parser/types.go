// SPDX-Licence-Identifier: EUPL-1.2

// Package parser is the driver-neutral output-parsing layer — reasoning
// channels (`<think>...</think>`), tool-call payloads, and a thinking-mode
// processor for streaming or batched generation output.
//
//	r := parser.ForHint(parser.Hint{Architecture: "qwen3"}).ParseReasoning(nil, text)
package parser

//	hint := parser.Hint{Architecture: "qwen3", AdapterName: "lora-coder"}
//	out := parser.ForHint(hint).ParseReasoning(nil, response)
type Hint struct {
	Architecture string
	AdapterName  string
}

//	cfg := parser.Config{Mode: parser.Capture, Capture: func(c parser.Chunk) { log.Print(c.Text) }}
type Config struct {
	Mode    Mode        `json:"mode,omitempty"`
	Capture func(Chunk) `json:"-"`
}

//	parser.Show     // leave reasoning markers + content in the visible output
//	parser.Hide     // strip recognised reasoning blocks from visible output
//	parser.Capture  // strip from visible + emit blocks via Config.Capture
type Mode string

const (
	Show    Mode = "show"
	Hide    Mode = "hide"
	Capture Mode = "capture"
)

//	chunk := parser.Chunk{Text: "let me think...", Channel: "thinking", Model: "qwen"}
type Chunk struct {
	Text    string `json:"text"`
	Channel string `json:"channel,omitempty"`
	Model   string `json:"model,omitempty"`
}

//	result := parser.Filter(text, parser.Config{Mode: parser.Capture}, hint)
//	visible := result.Text
type Result struct {
	Text      string  `json:"text"`
	Reasoning string  `json:"reasoning,omitempty"`
	Chunks    []Chunk `json:"chunks,omitempty"`
}

type reasoningMarker struct {
	start string
	ends  []string
	kind  string
}

type thinkingMarker struct {
	start   string
	end     string
	channel string
	model   string
}

type toolBlockMarker struct {
	start string
	end   string
}
