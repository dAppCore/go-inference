// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// weights_test.go covers LoadWeights' error paths (missing/malformed tensors, incomplete config)
// and loadDecoderWeights' happy path at a small, fully hand-built scale (the decoder's geometry is
// config-parametrised, unlike the vision tower's hardcoded real dims — see weights_sam.go/
// weights_clip.go's doc comments). loadSAMWeights/loadCLIPWeights' own happy paths are proven by
// live_test.go's TestLive_RealCheckpoint_Load against the real ~6.7GB checkpoint (they cannot be
// exercised at a smaller hand-built scale at all — every dimension is a hardcoded constant); every
// block-level golden test in vision_sam_test.go/vision_clip_test.go additionally proves each
// individual tensor SHAPE this package expects is the one the real checkpoint carries.

// f32Tensor builds an F32 safetensors.Tensor from f32 values — mirrors whisper's helper of the
// same name (arch/openai/whisper/weights_test.go).
func f32Tensor(vals []float32, shape ...int) safetensors.Tensor {
	data := make([]byte, len(vals)*4)
	for i, v := range vals {
		bits := math.Float32bits(v)
		data[4*i] = byte(bits)
		data[4*i+1] = byte(bits >> 8)
		data[4*i+2] = byte(bits >> 16)
		data[4*i+3] = byte(bits >> 24)
	}
	return safetensors.Tensor{Dtype: "F32", Shape: shape, Data: data}
}

func seqVals(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(i)
	}
	return v
}

// tinyDecoderTensors builds a hermetic 2-layer (1 dense, 1 MoE with 3 toy experts) decoder
// checkpoint using the REAL tensor names loadDecoderWeights reads (not simplified), small enough
// to hand-construct: hidden=4, intermediate=6, moeIntermediate=3, vocab=5, 3 routed experts, 2
// shared experts (combined width 6).
func tinyDecoderTensors() (map[string]safetensors.Tensor, *Config) {
	const hidden, intermediate, moeIntermediate, vocab, nExperts, nShared = 4, 6, 3, 5, 3, 2
	tensors := map[string]safetensors.Tensor{
		"model.embed_tokens.weight": f32Tensor(seqVals(vocab*hidden), vocab, hidden),
		"model.norm.weight":         f32Tensor(seqVals(hidden), hidden),
		"lm_head.weight":            f32Tensor(seqVals(vocab*hidden), vocab, hidden),

		"model.layers.0.input_layernorm.weight":          f32Tensor(seqVals(hidden), hidden),
		"model.layers.0.post_attention_layernorm.weight": f32Tensor(seqVals(hidden), hidden),
		"model.layers.0.self_attn.q_proj.weight":         f32Tensor(seqVals(hidden*hidden), hidden, hidden),
		"model.layers.0.self_attn.k_proj.weight":         f32Tensor(seqVals(hidden*hidden), hidden, hidden),
		"model.layers.0.self_attn.v_proj.weight":         f32Tensor(seqVals(hidden*hidden), hidden, hidden),
		"model.layers.0.self_attn.o_proj.weight":         f32Tensor(seqVals(hidden*hidden), hidden, hidden),
		"model.layers.0.mlp.gate_proj.weight":            f32Tensor(seqVals(intermediate*hidden), intermediate, hidden),
		"model.layers.0.mlp.up_proj.weight":              f32Tensor(seqVals(intermediate*hidden), intermediate, hidden),
		"model.layers.0.mlp.down_proj.weight":            f32Tensor(seqVals(hidden*intermediate), hidden, intermediate),

		"model.layers.1.input_layernorm.weight":              f32Tensor(seqVals(hidden), hidden),
		"model.layers.1.post_attention_layernorm.weight":     f32Tensor(seqVals(hidden), hidden),
		"model.layers.1.self_attn.q_proj.weight":             f32Tensor(seqVals(hidden*hidden), hidden, hidden),
		"model.layers.1.self_attn.k_proj.weight":             f32Tensor(seqVals(hidden*hidden), hidden, hidden),
		"model.layers.1.self_attn.v_proj.weight":             f32Tensor(seqVals(hidden*hidden), hidden, hidden),
		"model.layers.1.self_attn.o_proj.weight":             f32Tensor(seqVals(hidden*hidden), hidden, hidden),
		"model.layers.1.mlp.gate.weight":                     f32Tensor(seqVals(nExperts*hidden), nExperts, hidden),
		"model.layers.1.mlp.shared_experts.gate_proj.weight": f32Tensor(seqVals(moeIntermediate*nShared*hidden), moeIntermediate*nShared, hidden),
		"model.layers.1.mlp.shared_experts.up_proj.weight":   f32Tensor(seqVals(moeIntermediate*nShared*hidden), moeIntermediate*nShared, hidden),
		"model.layers.1.mlp.shared_experts.down_proj.weight": f32Tensor(seqVals(hidden*moeIntermediate*nShared), hidden, moeIntermediate*nShared),
	}
	for e := range nExperts {
		p := core.Sprintf("model.layers.1.mlp.experts.%d", e)
		tensors[p+".gate_proj.weight"] = f32Tensor(seqVals(moeIntermediate*hidden), moeIntermediate, hidden)
		tensors[p+".up_proj.weight"] = f32Tensor(seqVals(moeIntermediate*hidden), moeIntermediate, hidden)
		tensors[p+".down_proj.weight"] = f32Tensor(seqVals(hidden*moeIntermediate), hidden, moeIntermediate)
	}

	cfg := &Config{
		HiddenSize: hidden, IntermediateSize: intermediate, MoEIntermediateSize: moeIntermediate,
		VocabSize: vocab, NumHiddenLayers: 2, NumAttentionHeads: 2, NumKeyValueHeads: 2,
		NRoutedExperts: nExperts, NSharedExperts: nShared, NumExpertsPerTok: 2, FirstKDenseReplace: 1,
	}
	return tensors, cfg
}

// TestLoadDecoderWeights_Good pins the happy path: every real tensor name loadDecoderWeights
// reads resolves, dense-vs-MoE layer selection matches FirstKDenseReplace, and expert count
// matches NRoutedExperts.
func TestLoadDecoderWeights_Good(t *testing.T) {
	tensors, cfg := tinyDecoderTensors()
	l := weightLoader{tensors: tensors}
	w, err := loadDecoderWeights(l, cfg)
	if err != nil {
		t.Fatalf("loadDecoderWeights: %v", err)
	}
	if len(w.Layers) != 2 {
		t.Fatalf("loaded %d layers, want 2", len(w.Layers))
	}
	if w.Layers[0].IsMoE {
		t.Fatal("layer 0 loaded as MoE, want dense")
	}
	if !w.Layers[1].IsMoE || len(w.Layers[1].Experts) != 3 {
		t.Fatalf("layer 1 IsMoE=%v with %d experts, want true/3", w.Layers[1].IsMoE, len(w.Layers[1].Experts))
	}
	if len(w.EmbedTokens) != 5*4 || len(w.LMHeadWeight) != 5*4 {
		t.Fatalf("EmbedTokens/LMHeadWeight lens = %d/%d, want 20/20", len(w.EmbedTokens), len(w.LMHeadWeight))
	}
}

// TestLoadDecoderWeights_Bad proves a missing tensor is refused by name, not silently zero-valued.
func TestLoadDecoderWeights_Bad(t *testing.T) {
	tensors, cfg := tinyDecoderTensors()
	delete(tensors, "model.layers.1.mlp.gate.weight")
	l := weightLoader{tensors: tensors}
	if _, err := loadDecoderWeights(l, cfg); err == nil {
		t.Fatal("loadDecoderWeights accepted a tensor map missing the MoE router weight")
	}
}

// TestLoadDecoderWeights_Ugly proves a WRONG-SHAPED tensor (present, but the wrong element count)
// is refused — distinct from _Bad's missing-entirely case.
func TestLoadDecoderWeights_Ugly(t *testing.T) {
	tensors, cfg := tinyDecoderTensors()
	tensors["model.embed_tokens.weight"] = f32Tensor(seqVals(3), 3) // far too short for vocab=5,hidden=4
	l := weightLoader{tensors: tensors}
	if _, err := loadDecoderWeights(l, cfg); err == nil {
		t.Fatal("loadDecoderWeights accepted a wrong-shaped embed_tokens.weight")
	}
}

// TestLoadWeights_Bad proves LoadWeights refuses an incomplete decoder config (the first guard,
// firing before any tensor lookup) — mirrors whisper.LoadWeights' identical geometry guard.
func TestLoadWeights_Bad(t *testing.T) {
	if _, err := LoadWeights(map[string]safetensors.Tensor{}, &Config{}); err == nil {
		t.Fatal("LoadWeights accepted a config with zero-value decoder geometry")
	}
}

// TestLoadWeights_Ugly proves a nil config is refused distinctly from an empty-but-non-nil one.
func TestLoadWeights_Ugly(t *testing.T) {
	if _, err := LoadWeights(map[string]safetensors.Tensor{}, nil); err == nil {
		t.Fatal("LoadWeights accepted a nil config")
	}
}

// TestLoadWeights_Good proves LoadWeights fails at the SAM tower (the first sub-loader it calls)
// when given a valid decoder config but an otherwise-empty tensor map — i.e. it does not silently
// skip a missing vision tower just because the decoder config looks complete. (The real happy
// path — every tower AND the decoder loading together — is live_test.go's
// TestLive_RealCheckpoint_Load; assembling a full 768/1024-wide vision tower fixture by hand is
// impractical, see the file doc comment.)
func TestLoadWeights_Good(t *testing.T) {
	_, cfg := tinyDecoderTensors()
	_, err := LoadWeights(map[string]safetensors.Tensor{}, cfg)
	if err == nil {
		t.Fatal("LoadWeights accepted an empty tensor map (no SAM tower tensors at all)")
	}
	if !core.Contains(err.Error(), "SAM tower") {
		t.Fatalf("LoadWeights error %q must name which sub-loader failed (SAM tower)", err.Error())
	}
}
