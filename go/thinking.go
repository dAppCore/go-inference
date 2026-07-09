// SPDX-Licence-Identifier: EUPL-1.2

package inference

// Thinking-channel configuration — the engine-level form of the reasoning
// controls. EnableThinking on [GenerateConfig] is the API-level intent
// (on/off/model-default); ThinkingConfig is the resolved processing policy an
// engine applies to the thought channel while decoding. The parser package
// aliases these types (`parser.Config = inference.ThinkingConfig`), so parser
// consumers are unchanged; they live here so GenerateConfig can carry them
// without an import cycle (parser imports this package for Token and the
// parse-result contracts).

// ThinkingMode selects what happens to recognised reasoning blocks in the
// visible output stream.
//
//	inference.ThinkingShow     // leave reasoning markers + content in the visible output
//	inference.ThinkingHide     // strip recognised reasoning blocks from visible output
//	inference.ThinkingCapture  // strip from visible + emit blocks via ThinkingConfig.Capture
type ThinkingMode string

const (
	ThinkingShow    ThinkingMode = "show"
	ThinkingHide    ThinkingMode = "hide"
	ThinkingCapture ThinkingMode = "capture"
)

// ThinkingChunk is one captured reasoning block.
//
//	chunk := inference.ThinkingChunk{Text: "let me think...", Channel: "thinking", Model: "qwen"}
type ThinkingChunk struct {
	Text    string `json:"text"`
	Channel string `json:"channel,omitempty"`
	Model   string `json:"model,omitempty"`
}

// ThinkingConfig is the thought-channel processing policy for one generation.
//
//	cfg := inference.ThinkingConfig{Mode: inference.ThinkingCapture, Capture: func(c inference.ThinkingChunk) { _ = c.Text }}
type ThinkingConfig struct {
	Mode    ThinkingMode        `json:"mode,omitempty"`
	Capture func(ThinkingChunk) `json:"-"`
}
