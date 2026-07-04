// SPDX-Licence-Identifier: EUPL-1.2

package generate

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/serving"
)

// TestSpineModelInfo_CopiesFields_Good proves the inference→spine model-info
// bridge carries the fields the durable session needs.
func TestSpineModelInfo_CopiesFields_Good(t *testing.T) {
	got := spineModelInfo(inference.ModelInfo{
		Architecture: "gemma4",
		VocabSize:    262144,
		NumLayers:    26,
		HiddenSize:   2304,
		QuantBits:    4,
		QuantGroup:   64,
	}, 8192)
	if got.Architecture != "gemma4" || got.VocabSize != 262144 || got.NumLayers != 26 ||
		got.HiddenSize != 2304 || got.QuantBits != 4 || got.QuantGroup != 64 {
		t.Fatalf("field mapping wrong: %+v", got)
	}
	if got.ContextLength != 8192 {
		t.Fatalf("ContextLength = %d, want 8192", got.ContextLength)
	}
}

// TestSpineModelInfo_DefaultContext_Bad proves a non-positive context length
// falls back to the 4096 default rather than producing a zero-length KV cache.
func TestSpineModelInfo_DefaultContext_Bad(t *testing.T) {
	if got := spineModelInfo(inference.ModelInfo{Architecture: "gemma4"}, 0); got.ContextLength != 4096 {
		t.Fatalf("default ContextLength = %d, want 4096", got.ContextLength)
	}
}

// TestResolvedDraftBlock_FlagWins_Good proves an explicit draft block overrides
// the engine default.
func TestResolvedDraftBlock_FlagWins_Good(t *testing.T) {
	if got := resolvedDraftBlock(7); got != 7 {
		t.Fatalf("resolvedDraftBlock(7) = %d, want 7", got)
	}
}

// TestResolvedDraftBlock_DefaultWhenZero_Bad proves a zero flag falls back to
// the shared MTP engine default.
func TestResolvedDraftBlock_DefaultWhenZero_Bad(t *testing.T) {
	if got := resolvedDraftBlock(0); got != serving.MTPDefaultDraftBlock {
		t.Fatalf("resolvedDraftBlock(0) = %d, want %d", got, serving.MTPDefaultDraftBlock)
	}
}
