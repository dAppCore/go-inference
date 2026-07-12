// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// TestDiffusionEOSTokens covers the EOS-token config polymorphism: HF ships eos_token_id as either
// a bare number or a list, and the JSON decode lands it as float64 / []any. A scalar becomes a
// one-element slice, a list keeps its float64 members and drops any non-number, and any other type
// (or absence) yields nil. This drives the diffusion generation stop condition, so each shape is
// load-bearing.
func TestDiffusionEOSTokens(t *testing.T) {
	if got := diffusionEOSTokens(float64(7)); len(got) != 1 || got[0] != 7 {
		t.Fatalf("scalar eos = %v, want [7]", got)
	}
	if got := diffusionEOSTokens([]any{float64(1), float64(2), float64(3)}); len(got) != 3 || got[0] != 1 || got[2] != 3 {
		t.Fatalf("list eos = %v, want [1 2 3]", got)
	}
	// A list with a non-number member drops it rather than coercing garbage.
	if got := diffusionEOSTokens([]any{float64(1), "eos", float64(2)}); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("mixed list eos = %v, want [1 2] (non-number dropped)", got)
	}
	// An absent / wrong-typed eos_token_id → nil (no stop tokens declared).
	if got := diffusionEOSTokens(nil); got != nil {
		t.Fatalf("nil eos = %v, want nil", got)
	}
	if got := diffusionEOSTokens("2"); got != nil {
		t.Fatalf("string eos = %v, want nil", got)
	}
	t.Logf("diffusionEOSTokens: scalar→[n], list→filtered ints, non-number dropped, absent/wrong-type→nil")
}

// TestAssembleDiffusion_NilConfig covers the nil-config opt-out: a pack with no text config carries
// no diffusion trunk, returning (nil, nil) rather than erroring.
func TestAssembleDiffusion_NilConfig(t *testing.T) {
	d, err := AssembleDiffusion(map[string]safetensors.Tensor{}, nil)
	if d != nil || err != nil {
		t.Fatalf("nil config should yield (nil,nil), got (%v,%v)", d, err)
	}
	t.Logf("AssembleDiffusion: nil config → (nil,nil)")
}

// TestAssembleDiffusion_Incomplete covers the error arm: a config declaring a diffusion trunk whose
// self-conditioning block is absent from the checkpoint is rejected, not partially built.
func TestAssembleDiffusion_Incomplete(t *testing.T) {
	cfg := &Gemma4TextConfig{}
	cfg.HiddenSize = 64
	cfg.IntermediateSize = 128
	cfg.NumHiddenLayers = 2
	// Empty weight set → self_conditioning.pre_norm / gate / up / down all missing.
	if _, err := AssembleDiffusion(map[string]safetensors.Tensor{}, cfg); err == nil {
		t.Fatal("AssembleDiffusion should reject a checkpoint missing the self-conditioning block")
	}
	t.Logf("AssembleDiffusion: incomplete self-conditioning block → error")
}

// TestDiffusionEncoderScalarsShortfall covers the scalar-count guard: the self-conditioning block is
// present but fewer per-layer scalars are found than num_hidden_layers declares, so the trunk is
// rejected (a truncated diffusion encoder would silently mis-index layers).
func TestDiffusionEncoderScalarsShortfall(t *testing.T) {
	cfg := &Gemma4TextConfig{}
	cfg.HiddenSize = 8
	cfg.IntermediateSize = 16
	cfg.NumHiddenLayers = 2

	w := map[string]safetensors.Tensor{
		"self_conditioning.pre_norm.weight":  bf16Vec(8),
		"self_conditioning.gate_proj.weight": mat2D(16, 8),
		"self_conditioning.up_proj.weight":   mat2D(16, 8),
		"self_conditioning.down_proj.weight": mat2D(8, 16),
		// only one of the two declared encoder-layer scalars is present
		"model.encoder.language_model.layers.0.layer_scalar": bf16Vec(1),
	}
	if _, err := AssembleDiffusion(w, cfg); err == nil {
		t.Fatal("AssembleDiffusion should reject fewer encoder-layer scalars than num_hidden_layers")
	}
	t.Logf("AssembleDiffusion: encoder-layer-scalar shortfall (1 of 2) → error")
}

func bf16Vec(n int) safetensors.Tensor {
	return safetensors.Tensor{Dtype: "BF16", Shape: []int{n}, Data: make([]byte, n*2)}
}

func mat2D(out, in int) safetensors.Tensor {
	return safetensors.Tensor{Dtype: "BF16", Shape: []int{out, in}, Data: make([]byte, out*in*2)}
}
