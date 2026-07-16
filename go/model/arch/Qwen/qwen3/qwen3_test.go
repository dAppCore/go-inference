// SPDX-Licence-Identifier: EUPL-1.2

package qwen3

import (
	"math"
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// TestQwen3_Config_Arch_Good verifies the dense qwen3 config→Arch: a standard GQA transformer, scale
// 1/sqrt(head_dim), full rotary, all-global attention, no value-norm/softcap.
func TestQwen3_Config_Arch_Good(t *testing.T) {
	const layers, headDim = 28, 128
	c := &Config{
		HiddenSize: 2048, NumHiddenLayers: layers, NumAttentionHeads: 16, NumKeyValueHeads: 8,
		HeadDim: headDim, IntermediateSize: 6144, VocabSize: 151936, RMSNormEps: 1e-6, RopeTheta: 1_000_000,
	}
	a, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if want := float32(1.0 / math.Sqrt(headDim)); a.AttnScale != want {
		t.Errorf("AttnScale = %v, want %v", a.AttnScale, want)
	}
	if a.RotaryDim != headDim || a.RopeBase != 1_000_000 || a.SoftCap != 0 || a.ValueNorm {
		t.Errorf("qwen3 arch specifics wrong: rotary=%d rope=%v softcap=%v valuenorm=%v", a.RotaryDim, a.RopeBase, a.SoftCap, a.ValueNorm)
	}
	if len(a.Layer) != layers {
		t.Fatalf("layers = %d, want %d", len(a.Layer), layers)
	}
	for i := range a.Layer { // qwen3 dense: every layer global
		if a.Layer[i].Attention != model.GlobalAttention {
			t.Fatalf("layer %d not global (qwen3 dense has no sliding)", i)
		}
	}
	t.Logf("qwen3 Arch: %d global layers, scale=1/sqrt(%d), full rotary, QK-norm via weight names", layers, headDim)
}

func TestQwen3_Config_Arch_Bad(t *testing.T) {
	if _, err := (&Config{}).Arch(); err == nil {
		t.Fatal("empty config accepted")
	}
}

// TestQwen3_Config_Arch_Ugly proves the multimodal-wrapper delegation: a
// config with only TextConfig set resolves the NESTED dims, not the (empty)
// outer ones.
func TestQwen3_Config_Arch_Ugly(t *testing.T) {
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

// TestQwen3_Config_InferFromWeights_Good covers the dim-from-shape resolution: the q_proj rows
// outrank the hidden/heads arithmetic (proven with a geometry where the two answers DIFFER —
// hidden 2560 / 16 heads would give 160, the weight says 128, the real qwen3 shape where head_dim
// ≠ hidden/heads), a missing layer-0 q_proj falls through to a later layer's, and a declared
// head_dim is never overridden by a weight shape.
func TestQwen3_Config_InferFromWeights_Good(t *testing.T) {
	qProj := safetensors.Tensor{Shape: []int{16 * 128, 2560}}
	embed := safetensors.Tensor{Shape: []int{151936, 2560}}

	// weight shape outranks arithmetic: 2560/16 = 160, but the q_proj rows say 128.
	c := &Config{HiddenSize: 2560, NumHiddenLayers: 2, NumAttentionHeads: 16}
	c.InferFromWeights(map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": qProj,
		"model.embed_tokens.weight":              embed,
	})
	if c.HeadDim != 128 {
		t.Fatalf("HeadDim = %d, want 128 from the q_proj rows (NOT 160 = hidden/heads arithmetic)", c.HeadDim)
	}
	if c.VocabSize != 151936 {
		t.Fatalf("VocabSize = %d, want 151936 from the embedding rows", c.VocabSize)
	}

	// layer-0 q_proj absent → the per-layer walk finds layer 1's.
	c = &Config{HiddenSize: 2560, NumHiddenLayers: 2, NumAttentionHeads: 16}
	c.InferFromWeights(map[string]safetensors.Tensor{
		"model.layers.1.self_attn.q_proj.weight": qProj,
	})
	if c.HeadDim != 128 {
		t.Fatalf("HeadDim = %d, want 128 found on layer 1 (the walk must not stop at a missing layer 0)", c.HeadDim)
	}

	// a declared head_dim is authoritative — weights must not override it.
	c = &Config{HiddenSize: 2560, NumHiddenLayers: 1, NumAttentionHeads: 16, HeadDim: 64}
	c.InferFromWeights(map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": qProj,
	})
	if c.HeadDim != 64 {
		t.Fatalf("HeadDim = %d, want the declared 64 untouched", c.HeadDim)
	}
}

// TestQwen3_Config_InferFromWeights_Bad proves absent weights degrade to the
// hidden/heads arithmetic fallback for HeadDim, and leave VocabSize at 0 —
// no fabricated geometry from nothing.
func TestQwen3_Config_InferFromWeights_Bad(t *testing.T) {
	c := &Config{HiddenSize: 2048, NumHiddenLayers: 2, NumAttentionHeads: 16}
	c.InferFromWeights(map[string]safetensors.Tensor{})
	if c.HeadDim != 128 {
		t.Fatalf("HeadDim = %d, want 128 (2048/16 arithmetic fallback)", c.HeadDim)
	}
	if c.VocabSize != 0 {
		t.Fatalf("VocabSize = %d, want 0 (no embed weight, no fallback)", c.VocabSize)
	}
}

// TestQwen3_Config_InferFromWeights_Ugly proves the multimodal-wrapper
// delegation mutates the NESTED TextConfig, leaving the outer wrapper's own
// fields untouched — distinct from _Good's flat-config cases.
func TestQwen3_Config_InferFromWeights_Ugly(t *testing.T) {
	qProj := safetensors.Tensor{Shape: []int{16 * 128, 2560}}
	embed := safetensors.Tensor{Shape: []int{151936, 2560}}
	c := &Config{TextConfig: &Config{HiddenSize: 2560, NumHiddenLayers: 1, NumAttentionHeads: 16}}
	c.InferFromWeights(map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": qProj,
		"model.embed_tokens.weight":              embed,
	})
	if c.TextConfig.HeadDim != 128 || c.TextConfig.VocabSize != 151936 {
		t.Fatalf("wrapper delegation: nested HeadDim/Vocab = %d/%d, want 128/151936", c.TextConfig.HeadDim, c.TextConfig.VocabSize)
	}
	if c.HeadDim != 0 || c.VocabSize != 0 {
		t.Fatalf("outer wrapper fields = %d/%d, want 0/0 (only TextConfig is mutated)", c.HeadDim, c.VocabSize)
	}
}

func TestQwen3_Config_ResolvedQuant_Good(t *testing.T) {
	top := &model.QuantConfig{GroupSize: 64, Bits: 4}
	nested := &model.QuantConfig{GroupSize: 32, Bits: 8}
	if got := (&Config{Quantization: top, TextConfig: &Config{Quantization: nested}}).ResolvedQuant(); got != top {
		t.Fatalf("ResolvedQuant = %+v, want the top-level block", got)
	}
}

// TestQwen3_Config_ResolvedQuant_Bad proves absence is reported honestly: no
// quant block anywhere returns nil (bf16), whether TextConfig is absent
// entirely or present but quant-free.
func TestQwen3_Config_ResolvedQuant_Bad(t *testing.T) {
	if got := (&Config{}).ResolvedQuant(); got != nil {
		t.Fatalf("ResolvedQuant = %+v, want nil (bf16)", got)
	}
	if got := (&Config{TextConfig: &Config{}}).ResolvedQuant(); got != nil {
		t.Fatalf("ResolvedQuant = %+v, want nil (text_config present but quant-free)", got)
	}
}

// TestQwen3_Config_ResolvedQuant_Ugly exercises the nested fallback: the
// wrapper's own Quantization is absent but text_config carries one — a
// distinct code path from both the top-level hit and the true-nil case.
func TestQwen3_Config_ResolvedQuant_Ugly(t *testing.T) {
	nested := &model.QuantConfig{GroupSize: 32, Bits: 8}
	if got := (&Config{TextConfig: &Config{Quantization: nested}}).ResolvedQuant(); got != nested {
		t.Fatalf("ResolvedQuant = %+v, want the nested text_config block", got)
	}
}

// TestQwen3Registered confirms qwen3 is registered with the llama/mistral norm layout (MLP norm =
// post_attention_layernorm, no gemma post-attn norm), QK-norm names kept, and plain RMSNorm.
func TestQwen3Registered(t *testing.T) {
	spec, ok := model.LookupArch("qwen3")
	if !ok {
		t.Fatal("qwen3 not registered")
	}
	if spec.Weights.MLPNorm != ".post_attention_layernorm.weight" {
		t.Errorf("MLPNorm = %q, want llama-style post_attention_layernorm", spec.Weights.MLPNorm)
	}
	if spec.Weights.PostAttnNorm != "" {
		t.Errorf("PostAttnNorm = %q, want empty (qwen3 has no gemma post-attn norm)", spec.Weights.PostAttnNorm)
	}
	if spec.Weights.QNorm != ".self_attn.q_norm.weight" {
		t.Errorf("QNorm = %q, want QK-norm bound by name", spec.Weights.QNorm)
	}
	if spec.Weights.NormBiasOne {
		t.Error("NormBiasOne must be false for qwen3 (plain RMSNorm, not gemma)")
	}
	t.Log("qwen3 registered: dense qwen3 loads via the reactive loader (mistral-style norms + QK-norm, plain RMSNorm)")
}
