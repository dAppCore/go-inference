// SPDX-Licence-Identifier: EUPL-1.2

package spine

import (
	"testing"

	"dappco.re/go/inference/train/lora"
	"dappco.re/go/inference/parser"
)

// Tests for the ModelInfo projections. These are projections, not full
// copies — the only non-tautological thing to assert is which fields make
// it across and which are deliberately dropped, so each test pins the
// carried fields AND the dropped ones (zero on the far side).

func sampleModelInfo() ModelInfo {
	return ModelInfo{
		Architecture:  "gemma3",
		VocabSize:     262144,
		NumLayers:     26,
		NumHeads:      8,
		NumKVHeads:    4,
		HeadDim:       256,
		HiddenSize:    1152,
		QuantBits:     4,
		QuantGroup:    64,
		ContextLength: 8192,
		Adapter:       lora.AdapterInfo{Name: "lem", Rank: 8},
	}
}

// --- ModelInfoToBundle ---

func TestModelInfo_ModelInfoToBundle_Good(t *testing.T) {
	got := ModelInfoToBundle(sampleModelInfo())
	if got.Architecture != "gemma3" || got.VocabSize != 262144 || got.NumLayers != 26 {
		t.Fatalf("bundle = %+v, want gemma3/262144/26", got)
	}
	if got.HiddenSize != 1152 || got.QuantBits != 4 || got.QuantGroup != 64 || got.ContextLength != 8192 {
		t.Fatalf("bundle = %+v, want hidden1152/q4/group64/ctx8192", got)
	}
	if got.Adapter.Name != "lem" || got.Adapter.Rank != 8 {
		t.Fatalf("bundle Adapter = %+v, want {lem,8}", got.Adapter)
	}
}

func TestModelInfo_ModelInfoToBundle_Bad(t *testing.T) {
	// Zero ModelInfo → zero bundle.ModelInfo, no panic.
	got := ModelInfoToBundle(ModelInfo{})
	if got.Architecture != "" || got.VocabSize != 0 || got.Adapter.Name != "" {
		t.Fatalf("zero in → %+v, want all-zero bundle", got)
	}
}

// --- ModelInfoToMemory ---

func TestModelInfo_ModelInfoToMemory_Good(t *testing.T) {
	got := ModelInfoToMemory(sampleModelInfo())
	if got.Architecture != "gemma3" || got.VocabSize != 262144 || got.NumLayers != 26 {
		t.Fatalf("memory = %+v, want gemma3/262144/26", got)
	}
	if got.HiddenSize != 1152 || got.QuantBits != 4 || got.QuantGroup != 64 || got.ContextLength != 8192 {
		t.Fatalf("memory = %+v, want hidden1152/q4/group64/ctx8192", got)
	}
}

func TestModelInfo_ModelInfoToMemory_Ugly(t *testing.T) {
	// memory.ModelInfo HAS NumKVHeads + HeadDim slots, but the converter
	// does not wire them — assert the current (partial-projection)
	// behaviour: both come out zero even though the source set them.
	// This is documented in the report as left-as-is for the coverage
	// task, not fixed here (tests adapt to the code).
	got := ModelInfoToMemory(sampleModelInfo())
	if got.NumKVHeads != 0 || got.HeadDim != 0 {
		t.Fatalf("memory NumKVHeads=%d HeadDim=%d, want both 0 (converter drops them)", got.NumKVHeads, got.HeadDim)
	}
}

// --- ParserHint ---

func TestModelInfo_ParserHint_Good(t *testing.T) {
	got := ParserHint(sampleModelInfo())
	want := parser.Hint{Architecture: "gemma3", AdapterName: "lem"}
	if got != want {
		t.Fatalf("ParserHint = %+v, want %+v", got, want)
	}
}

func TestModelInfo_ParserHint_Bad(t *testing.T) {
	// No adapter set → AdapterName must be empty (sourced from the empty
	// AdapterInfo.Name), not a panic on the nested struct.
	got := ParserHint(ModelInfo{Architecture: "qwen3"})
	if got.Architecture != "qwen3" || got.AdapterName != "" {
		t.Fatalf("ParserHint = %+v, want {qwen3, empty adapter}", got)
	}
}
