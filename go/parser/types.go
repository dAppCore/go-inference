// SPDX-Licence-Identifier: EUPL-1.2

// Package parser is the driver-neutral output-parsing layer — reasoning
// channels (`<think>...</think>`), tool-call payloads, and a thinking-mode
// processor for streaming or batched generation output.
//
//	r := parser.ForHint(parser.Hint{Architecture: "qwen3"}).ParseReasoning(nil, text)
package parser

import "dappco.re/go/inference"

//	hint := parser.Hint{Architecture: "qwen3", AdapterName: "lora-coder"}
//	out := parser.ForHint(hint).ParseReasoning(nil, response)
type Hint struct {
	Architecture string
	AdapterName  string
}

// The thinking trio (Config/Mode/Chunk) is declared in the inference root so
// inference.GenerateConfig can carry a ThinkingConfig without an import cycle
// (this package imports inference for Token and the parse-result contracts).
// The aliases keep every parser.* consumer unchanged.
//
//	cfg := parser.Config{Mode: parser.Capture, Capture: func(c parser.Chunk) { log.Print(c.Text) }}
type Config = inference.ThinkingConfig

//	parser.Show     // leave reasoning markers + content in the visible output
//	parser.Hide     // strip recognised reasoning blocks from visible output
//	parser.Capture  // strip from visible + emit blocks via Config.Capture
type Mode = inference.ThinkingMode

const (
	Show    = inference.ThinkingShow
	Hide    = inference.ThinkingHide
	Capture = inference.ThinkingCapture
)

//	chunk := parser.Chunk{Text: "let me think...", Channel: "thinking", Model: "qwen"}
type Chunk = inference.ThinkingChunk

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
