// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	"testing"

	"dappco.re/go/inference"
)

// TestTypes_Hint_Good pins the normal construction shape: both fields set.
func TestTypes_Hint_Good(t *testing.T) {
	h := Hint{Architecture: "qwen3", AdapterName: "lora-coder"}
	if h.Architecture != "qwen3" || h.AdapterName != "lora-coder" {
		t.Fatalf("Hint = %+v, want architecture/adapter set verbatim", h)
	}
}

// TestTypes_Hint_Bad pins the zero value: an unset Hint carries empty fields
// rather than a sentinel, so Family(Hint{}) falls through to "generic".
func TestTypes_Hint_Bad(t *testing.T) {
	var h Hint
	if h.Architecture != "" || h.AdapterName != "" {
		t.Fatalf("zero Hint = %+v, want both fields empty", h)
	}
	if got := Family(h); got != "generic" {
		t.Fatalf("Family(zero Hint) = %q, want generic", got)
	}
}

// TestTypes_Result_Ugly pins the JSON-tag contract: Reasoning and Chunks are
// `omitempty` so a plain-content result (no reasoning) serialises without
// the noise, while Text always renders even when empty.
func TestTypes_Result_Ugly(t *testing.T) {
	full := Result{Text: "answer", Reasoning: "plan", Chunks: []Chunk{{Text: "plan", Channel: "thinking"}}}
	if full.Text != "answer" || full.Reasoning != "plan" || len(full.Chunks) != 1 {
		t.Fatalf("Result = %+v, want all fields populated", full)
	}

	bare := Result{Text: "answer"}
	if bare.Reasoning != "" || bare.Chunks != nil {
		t.Fatalf("bare Result = %+v, want Reasoning empty and Chunks nil", bare)
	}
}

// TestTypes_Mode_Good pins the alias identity: Config/Mode/Chunk are
// type aliases (not distinct types) over their inference.Thinking*
// counterparts, and Show/Hide/Capture equal the inference constants —
// callers must be able to pass either spelling interchangeably.
func TestTypes_Mode_Good(t *testing.T) {
	var cfg Config = inference.ThinkingConfig{Mode: Capture}
	var mode Mode = inference.ThinkingShow
	var chunk Chunk = inference.ThinkingChunk{Text: "x"}

	if cfg.Mode != Capture {
		t.Fatalf("Config alias round-trip = %q, want Capture", cfg.Mode)
	}
	if mode != Show {
		t.Fatalf("Mode alias round-trip = %q, want Show", mode)
	}
	if chunk.Text != "x" {
		t.Fatalf("Chunk alias round-trip = %+v, want Text=x", chunk)
	}
	if Show != inference.ThinkingShow || Hide != inference.ThinkingHide || Capture != inference.ThinkingCapture {
		t.Fatalf("mode constants drifted from inference.Thinking*: show=%q hide=%q capture=%q", Show, Hide, Capture)
	}
}
