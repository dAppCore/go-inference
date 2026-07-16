// SPDX-Licence-Identifier: EUPL-1.2

package gemma3

import (
	"math"
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// TestGemma3_Config_Arch_Good verifies the gemma3 config→Arch derivation against the specifics confirmed
// from metal gemma3: scale 1/sqrt(head_dim), full rotary, no softcap, no value-norm, and the
// sliding/global layer pattern (global when (i+1)%pattern == 0).
func TestGemma3_Config_Arch_Good(t *testing.T) {
	const layers, pattern, headDim = 12, 6, 256
	c := &Config{
		HiddenSize: 2048, NumHiddenLayers: layers, IntermediateSize: 8192,
		NumAttentionHeads: 8, NumKeyValueHeads: 4, HeadDim: headDim, VocabSize: 262144,
		RMSNormEps: 1e-6, RopeTheta: 1_000_000, RopeLocalBaseFreq: 10_000,
		SlidingWindow: 1024, SlidingWindowPattern: pattern,
	}
	a, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if want := float32(1.0 / math.Sqrt(headDim)); a.AttnScale != want {
		t.Errorf("AttnScale = %v, want 1/sqrt(head_dim) = %v", a.AttnScale, want)
	}
	if a.RotaryDim != headDim || a.RotaryDimLocal != headDim {
		t.Errorf("rotary = %d/%d, want full %d (gemma3 has no partial rotary)", a.RotaryDim, a.RotaryDimLocal, headDim)
	}
	if a.SoftCap != 0 {
		t.Errorf("SoftCap = %v, want 0 (gemma3 dropped logit softcapping)", a.SoftCap)
	}
	if a.ValueNorm {
		t.Error("ValueNorm = true, want false (gemma3 does not value-norm V)")
	}
	if a.RopeBase != 1_000_000 || a.RopeLocalBase != 10_000 {
		t.Errorf("rope bases = %v/%v, want 1e6/1e4", a.RopeBase, a.RopeLocalBase)
	}
	if a.Hidden != 2048 || a.Heads != 8 || a.KVHeads != 4 || a.HeadDim != headDim || a.FF != 8192 || a.Vocab != 262144 {
		t.Errorf("dims wrong: %+v", a)
	}
	if len(a.Layer) != layers {
		t.Fatalf("layers = %d, want %d", len(a.Layer), layers)
	}
	globals := 0
	for i := range a.Layer {
		isGlobal := a.Layer[i].Attention == model.GlobalAttention
		wantGlobal := (i+1)%pattern == 0
		if isGlobal != wantGlobal {
			t.Errorf("layer %d: global=%v, want %v", i, isGlobal, wantGlobal)
		}
		if isGlobal {
			globals++
		}
	}
	t.Logf("gemma3 Arch: scale=1/sqrt(%d), full rotary, no softcap/value-norm, %d layers (%d global per pattern %d)", headDim, layers, globals, pattern)
}

func TestGemma3_Config_Arch_Bad(t *testing.T) {
	if _, err := (&Config{}).Arch(); err == nil {
		t.Fatal("empty config accepted")
	}
}

// TestGemma3_Config_Arch_Ugly proves the multimodal-wrapper delegation: a
// config with only TextConfig set resolves the NESTED dims, not the (empty)
// outer ones — the distinct edge Arch()'s TextConfig != nil branch exists for.
func TestGemma3_Config_Arch_Ugly(t *testing.T) {
	inner := &Config{HiddenSize: 8, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 16}
	c := &Config{TextConfig: inner}
	arch, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch (multimodal wrapper delegation): %v", err)
	}
	if arch.Hidden != 8 || arch.Vocab != 16 {
		t.Fatalf("delegated arch = %+v, want the nested text_config dims (hidden 8, vocab 16)", arch)
	}
}

func TestGemma3_Config_ResolvedQuant_Good(t *testing.T) {
	q := &model.QuantConfig{Bits: 4, Mode: "affine"}
	c := &Config{Quantization: q}
	if got := c.ResolvedQuant(); got != q {
		t.Fatalf("ResolvedQuant = %+v, want the top-level quantization block", got)
	}
}

// TestGemma3_Config_ResolvedQuant_Bad proves absence is reported honestly: a
// config with no quantization block anywhere returns nil (bf16), not a
// fabricated default.
func TestGemma3_Config_ResolvedQuant_Bad(t *testing.T) {
	c := &Config{}
	if got := c.ResolvedQuant(); got != nil {
		t.Fatalf("ResolvedQuant = %+v, want nil (no quant declared)", got)
	}
}

// TestGemma3_Config_ResolvedQuant_Ugly exercises the nested fallback: the
// multimodal wrapper's own Quantization is absent but text_config carries
// one — a distinct code path from both the top-level hit and the true-nil case.
func TestGemma3_Config_ResolvedQuant_Ugly(t *testing.T) {
	q := &model.QuantConfig{Bits: 8, Mode: "mxfp8"}
	c := &Config{TextConfig: &Config{Quantization: q}}
	if got := c.ResolvedQuant(); got != q {
		t.Fatalf("ResolvedQuant = %+v, want the nested text_config quantization block", got)
	}
}

func TestGemma3_Config_InferFromWeights_Good(t *testing.T) {
	c := &Config{NumHiddenLayers: 1, NumAttentionHeads: 8}
	weights := map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": {Shape: []int{2048, 1024}}, // 8 heads * 256 head_dim
		"model.embed_tokens.weight":              {Shape: []int{262144, 1024}},
	}
	c.InferFromWeights(weights)
	if c.HeadDim != 256 {
		t.Fatalf("HeadDim = %d, want 256 (2048/8)", c.HeadDim)
	}
	if c.VocabSize != 262144 {
		t.Fatalf("VocabSize = %d, want 262144", c.VocabSize)
	}
}

// TestGemma3_Config_InferFromWeights_Bad proves missing checkpoint tensors
// degrade to the declared-dims fallback rather than leaving garbage.
func TestGemma3_Config_InferFromWeights_Bad(t *testing.T) {
	c := &Config{HiddenSize: 2048, NumAttentionHeads: 8}
	c.InferFromWeights(nil)
	if c.HeadDim != 256 { // falls back to hidden/heads when weights absent
		t.Fatalf("HeadDim = %d, want 256 fallback (hidden/heads)", c.HeadDim)
	}
	if c.VocabSize != 0 {
		t.Fatalf("VocabSize = %d, want 0 (no embed weight, no fallback)", c.VocabSize)
	}
}

// TestGemma3_Config_InferFromWeights_Ugly proves the multimodal-wrapper
// delegation mutates the NESTED TextConfig, leaving the outer wrapper's own
// fields untouched — distinct from _Bad's flat-config fallback path.
func TestGemma3_Config_InferFromWeights_Ugly(t *testing.T) {
	inner := &Config{NumHiddenLayers: 1, NumAttentionHeads: 4}
	c := &Config{TextConfig: inner}
	weights := map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": {Shape: []int{1024, 512}}, // 4 heads * 256 head_dim
	}
	c.InferFromWeights(weights)
	if inner.HeadDim != 256 {
		t.Fatalf("nested TextConfig.HeadDim = %d, want 256 (delegation)", inner.HeadDim)
	}
	if c.HeadDim != 0 {
		t.Fatalf("outer wrapper HeadDim = %d, want 0 (only TextConfig is mutated)", c.HeadDim)
	}
}

// TestGemma3Registered confirms gemma3 is in the reactive arch registry with the gemma (1+w) RMSNorm
// convention enabled (NormBiasOne) — so model.Load assembles a gemma3 checkpoint with folded norms.
func TestGemma3Registered(t *testing.T) {
	var spec model.ArchSpec
	for _, mt := range []string{"gemma3", "gemma3_text"} { // both declared aliases must resolve
		s, ok := model.LookupArch(mt)
		if !ok {
			t.Fatalf("gemma3 not registered in the arch registry under %q", mt)
		}
		spec = s
	}
	if !spec.Weights.NormBiasOne {
		t.Error("gemma3 ArchSpec must set Weights.NormBiasOne for the (1+w) RMSNorm convention")
	}
	// Parse a minimal config and derive the arch through the registered spec.
	cfg, err := spec.Parse([]byte(`{"model_type":"gemma3","hidden_size":1152,"num_hidden_layers":4,"intermediate_size":6912,"num_attention_heads":4,"num_key_value_heads":1,"head_dim":256,"vocab_size":262144,"sliding_window_pattern":6}`))
	if err != nil {
		t.Fatalf("registered Parse: %v", err)
	}
	a, err := cfg.Arch()
	if err != nil {
		t.Fatalf("registered Arch: %v", err)
	}
	if len(a.Layer) != 4 || a.HeadDim != 256 {
		t.Fatalf("parsed arch wrong: layers=%d headDim=%d", len(a.Layer), a.HeadDim)
	}
	t.Log("gemma3 registered: model.Load can parse + assemble a gemma3 checkpoint via the reactive loader")
}
