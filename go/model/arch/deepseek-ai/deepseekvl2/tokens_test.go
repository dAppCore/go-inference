// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"testing"

	"dappco.re/go/inference/decode/tokenizer"
)

// tokens_test.go covers BuildPromptEmbeds/GreedyDecode's GUARDS hermetically (a zero-value
// *tokenizer.Tokenizer safely returns "not found" from every lookup — Go's nil-map reads never
// panic — so the checks that fire BEFORE any real tokenisation are testable without a loaded
// vocab). tokenizer.Tokenizer's fields are all unexported (encode/decode state), so a hermetic
// Tokenizer carrying a WORKING vocab/BOS/merges cannot be hand-built from this package — Encode/
// TokenID's happy path (and therefore BuildPromptEmbeds'/GreedyDecode's own happy paths) is
// proven instead by live_test.go's TestLive_RealCheckpoint_OCR against the real checkpoint's real
// tokenizer.json — the strongest available evidence (the exact reference transcript, not a
// synthetic approximation).

// TestBuildPromptEmbeds_Bad proves a prompt without exactly one "<image>" placeholder is refused
// before any tokenizer/weights state is even consulted (BuildPromptEmbeds' doc comment: the
// cheapest, model-independent check runs first).
func TestBuildPromptEmbeds_Bad(t *testing.T) {
	tok := tokenizer.NewForDecode(nil)
	w := &Weights{}
	for _, prompt := range []string{"no placeholder here", "<image> one <image> two"} {
		if _, _, err := BuildPromptEmbeds(prompt, nil, tok, w); err == nil {
			t.Fatalf("BuildPromptEmbeds(%q, ...) accepted a prompt without exactly one <image> placeholder", prompt)
		}
	}
}

// TestBuildPromptEmbeds_Ugly proves a visionFeatures buffer that doesn't match
// NumImageTokens*hidden is refused — distinct from _Bad's placeholder-count case (this fires on
// the SECOND check, once the prompt shape itself is valid).
func TestBuildPromptEmbeds_Ugly(t *testing.T) {
	tok := tokenizer.NewForDecode(nil)
	w := &Weights{} // hidden = 0
	if _, _, err := BuildPromptEmbeds("<image>\nprompt", make([]float32, 10), tok, w); err == nil {
		t.Fatal("BuildPromptEmbeds accepted a visionFeatures buffer that does not match NumImageTokens*hidden")
	}
}

// TestBuildPromptEmbeds_Good proves the nil tokenizer/weights guard — the one BuildPromptEmbeds
// check reachable with no real tokenizer/checkpoint at all — fires cleanly and distinctly from
// the shape guards above (a *_Good test for an always-refuses guard mirrors
// deepseekvl2.Config's own ExampleConfig_Arch pattern: the "recognised, correctly-refuses"
// behaviour IS the happy path for this particular entry point). See the file doc comment for
// where BuildPromptEmbeds' actual assembly happy path is proven.
func TestBuildPromptEmbeds_Good(t *testing.T) {
	if _, _, err := BuildPromptEmbeds("<image>", nil, nil, nil); err == nil {
		t.Fatal("BuildPromptEmbeds accepted nil tokenizer and weights")
	}
}

// TestGreedyDecode_Bad proves a config/weights geometry mismatch (embeds not a whole number of
// hidden-width rows) is refused before any decode step runs.
func TestGreedyDecode_Bad(t *testing.T) {
	cfg := &Config{HiddenSize: 8, MaxPositionEmbeddings: 100, NumHiddenLayers: 1, NumAttentionHeads: 2}
	w := &Weights{Decoder: DecoderWeights{Layers: make([]DecoderLayerWeights, 1)}}
	if _, err := GreedyDecode(make([]float32, 5), cfg, w, 10); err == nil {
		t.Fatal("GreedyDecode accepted an embeds buffer that is not a whole number of hidden-width rows")
	}
}

// TestGreedyDecode_Ugly proves a prompt already at or past the position-embedding bound is
// refused (a truthful, named refusal — mirroring whisper's ">30s window" bound — rather than
// silently truncating or wrapping) — distinct from _Bad's malformed-buffer case.
func TestGreedyDecode_Ugly(t *testing.T) {
	cfg := &Config{HiddenSize: 8, MaxPositionEmbeddings: 4, NumHiddenLayers: 1, NumAttentionHeads: 2}
	w := &Weights{Decoder: DecoderWeights{Layers: make([]DecoderLayerWeights, 1)}}
	promptEmbeds := make([]float32, 8*4) // 4 rows == MaxPositionEmbeddings already
	if _, err := GreedyDecode(promptEmbeds, cfg, w, 10); err == nil {
		t.Fatal("GreedyDecode accepted a prompt already at the max_position_embeddings bound")
	}
}
