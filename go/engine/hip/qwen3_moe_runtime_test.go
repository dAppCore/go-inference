// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
	sharedmodel "dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// qwen3MoESeeded is a tiny deterministic value generator for synthetic weight fixtures —
// no real PRNG needed, just enough variation across positions that a shape/wiring bug
// (e.g. two tensors accidentally aliased) shows up as a numeric mismatch rather than
// silently cancelling out.
type qwen3MoESeeded struct{ n int }

func (s *qwen3MoESeeded) values(count int) []float32 {
	out := make([]float32, count)
	for i := range out {
		s.n++
		out[i] = float32(s.n%13-6) / 4
	}
	return out
}

func qwen3MoEF32Tensor(values []float32, shape ...int) safetensors.Tensor {
	return safetensors.Tensor{Dtype: "F32", Shape: shape, Data: safetensors.EncodeFloat32(values)}
}

// tinyQwen3MoETensors builds a complete, minimal qwen3_moe checkpoint tensor map — same
// HF tensor naming convention as model/arch/Qwen/qwenmoe's own synthetic fixtures
// (load_test.go's tinyQwen2MoEWeights), plus the q_norm/k_norm tensors a real Qwen3-MoE
// checkpoint always carries (this package's loader requires them; the metal factory
// route tolerates their absence, this v1 forward pass does not — see
// hip_qwen3_moe_config.go's loadHIPQwen3MoEWeights).
func tinyQwen3MoETensors(hidden, vocab, expertFF, heads, kvHeads, headDim, experts, layers int) map[string]safetensors.Tensor {
	s := &qwen3MoESeeded{}
	tensors := map[string]safetensors.Tensor{
		"model.embed_tokens.weight": qwen3MoEF32Tensor(s.values(vocab*hidden), vocab, hidden),
		"model.norm.weight":         qwen3MoEF32Tensor(s.values(hidden), hidden),
	}
	for layer := 0; layer < layers; layer++ {
		p := core.Sprintf("model.layers.%d.", layer)
		tensors[p+"input_layernorm.weight"] = qwen3MoEF32Tensor(s.values(hidden), hidden)
		tensors[p+"post_attention_layernorm.weight"] = qwen3MoEF32Tensor(s.values(hidden), hidden)
		tensors[p+"self_attn.q_proj.weight"] = qwen3MoEF32Tensor(s.values(heads*headDim*hidden), heads*headDim, hidden)
		tensors[p+"self_attn.k_proj.weight"] = qwen3MoEF32Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden)
		tensors[p+"self_attn.v_proj.weight"] = qwen3MoEF32Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden)
		tensors[p+"self_attn.o_proj.weight"] = qwen3MoEF32Tensor(s.values(hidden*heads*headDim), hidden, heads*headDim)
		tensors[p+"self_attn.q_norm.weight"] = qwen3MoEF32Tensor(s.values(headDim), headDim)
		tensors[p+"self_attn.k_norm.weight"] = qwen3MoEF32Tensor(s.values(headDim), headDim)
		tensors[p+"mlp.gate.weight"] = qwen3MoEF32Tensor(s.values(experts*hidden), experts, hidden)
		for e := 0; e < experts; e++ {
			ep := core.Sprintf("%smlp.experts.%d.", p, e)
			tensors[ep+"gate_proj.weight"] = qwen3MoEF32Tensor(s.values(expertFF*hidden), expertFF, hidden)
			tensors[ep+"up_proj.weight"] = qwen3MoEF32Tensor(s.values(expertFF*hidden), expertFF, hidden)
			tensors[ep+"down_proj.weight"] = qwen3MoEF32Tensor(s.values(hidden*expertFF), hidden, expertFF)
		}
	}
	return tensors
}

func tinyQwen3MoEGeometry() hipQwen3MoEConfig {
	return hipQwen3MoEConfig{
		HiddenSize: 8, NumLayers: 1, VocabSize: 16,
		Heads: 2, KVHeads: 1, HeadDim: 4,
		NumExperts: 4, TopK: 2, ExpertFF: 6,
		Epsilon: 1e-5, RopeTheta: 10000, NormaliseTopK: true,
	}
}

// TestLoadHIPQwen3MoEWeights_Good proves loadHIPQwen3MoEWeights resolves every tensor
// name a real qwen3_moe checkpoint carries (attention, QK-norm, router, every expert,
// final norm) and ties the LM head to the embedding table when the checkpoint omits its
// own — the tied-embeddings convention most published Qwen3-MoE checkpoints use.
func TestLoadHIPQwen3MoEWeights_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	geometry := tinyQwen3MoEGeometry()
	tensors := tinyQwen3MoETensors(geometry.HiddenSize, geometry.VocabSize, geometry.ExpertFF, geometry.Heads, geometry.KVHeads, geometry.HeadDim, geometry.NumExperts, geometry.NumLayers)

	weights, err := loadHIPQwen3MoEWeights(driver, tensors, geometry)
	core.AssertNoError(t, err)
	defer weights.Close()

	core.AssertEqual(t, geometry.VocabSize*geometry.HiddenSize, len(weights.Embed))
	core.AssertEqual(t, 1, len(weights.Layers))
	lw := weights.Layers[0]
	for name, buf := range map[string]*hipDeviceByteBuffer{
		"InputNorm": lw.InputNorm, "QProj": lw.QProj, "KProj": lw.KProj, "VProj": lw.VProj, "OProj": lw.OProj,
		"QNorm": lw.QNorm, "KNorm": lw.KNorm, "PostAttnNorm": lw.PostAttnNorm, "Router": lw.Router,
	} {
		if buf == nil || buf.Pointer() == 0 {
			t.Fatalf("%s device buffer not loaded", name)
		}
	}
	core.AssertEqual(t, geometry.NumExperts, len(lw.ExpertGate))
	for e := 0; e < geometry.NumExperts; e++ {
		if lw.ExpertGate[e] == nil || lw.ExpertUp[e] == nil || lw.ExpertDown[e] == nil {
			t.Fatalf("expert %d weights not loaded", e)
		}
	}
	if weights.LMHead.Count() != geometry.VocabSize*geometry.HiddenSize {
		t.Fatalf("lm_head tied to embed_tokens has wrong element count: %d", weights.LMHead.Count())
	}
}

// TestLoadHIPQwen3MoEWeights_MissingTensor_Bad names the exact absent tensor rather than
// failing generically — q_norm is required (real Qwen3-MoE checkpoints always carry it;
// see tinyQwen3MoETensors' doc for why this loader does not tolerate its absence the way
// the metal factory route does).
func TestLoadHIPQwen3MoEWeights_MissingTensor_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	geometry := tinyQwen3MoEGeometry()
	tensors := tinyQwen3MoETensors(geometry.HiddenSize, geometry.VocabSize, geometry.ExpertFF, geometry.Heads, geometry.KVHeads, geometry.HeadDim, geometry.NumExperts, geometry.NumLayers)
	delete(tensors, "model.layers.0.self_attn.q_norm.weight")

	_, err := loadHIPQwen3MoEWeights(driver, tensors, geometry)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "self_attn.q_norm.weight")
}

// TestHIPQwen3MoEModel_GenerateGreedy_Good is the greedy generate smoke: a synthetic
// (non-checkpoint-backed — no real Qwen MoE snapshot exists on this box, see the worker
// brief's step 4) qwen3_moe model, driven token-ID-in/token-ID-out through the SAME
// shared generation loop every backend uses (model.Generate), exercising the whole
// pipeline this family's NextWork names: embed -> N-layer attention+router+SwiGLU-expert
// forward -> final norm -> LM head -> greedy sample, repeated. Proves the assembly runs
// end-to-end and stays in-vocabulary; it is not a text/tokenizer smoke (no tokenizer.json
// is built here — loadHIPQwen3MoETextModel's directory+tokenizer path is covered by
// build+vet and by mirroring loadHIPMamba2TextModel's proven shape, not by an on-disk
// fixture in this pass; named as the follow-up in the final report).
func TestHIPQwen3MoEModel_GenerateGreedy_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	geometry := tinyQwen3MoEGeometry()
	tensors := tinyQwen3MoETensors(geometry.HiddenSize, geometry.VocabSize, geometry.ExpertFF, geometry.Heads, geometry.KVHeads, geometry.HeadDim, geometry.NumExperts, geometry.NumLayers)
	weights, err := loadHIPQwen3MoEWeights(driver, tensors, geometry)
	core.AssertNoError(t, err)
	defer weights.Close()

	model := &hipQwen3MoEModel{driver: driver, cfg: geometry, weights: weights}
	core.AssertEqual(t, geometry.VocabSize, model.Vocab())

	prompt := []int32{1, 2, 3}
	generated, err := sharedmodel.Generate(model, prompt, 4, -1)
	core.AssertNoError(t, err)
	core.AssertEqual(t, 4, len(generated))
	for _, id := range generated {
		if id < 0 || int(id) >= geometry.VocabSize {
			t.Fatalf("generated token %d is out of vocabulary range [0,%d)", id, geometry.VocabSize)
		}
	}
}

// TestHIPQwen3MoEModel_GenerateGreedy_Deterministic_Good proves greedy generation is
// deterministic: two runs from the same weights and prompt produce the identical token
// sequence — a basic sanity check that no uninitialised/shared scratch state leaks
// between calls.
func TestHIPQwen3MoEModel_GenerateGreedy_Deterministic_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	geometry := tinyQwen3MoEGeometry()
	tensors := tinyQwen3MoETensors(geometry.HiddenSize, geometry.VocabSize, geometry.ExpertFF, geometry.Heads, geometry.KVHeads, geometry.HeadDim, geometry.NumExperts, geometry.NumLayers)
	weights, err := loadHIPQwen3MoEWeights(driver, tensors, geometry)
	core.AssertNoError(t, err)
	defer weights.Close()
	model := &hipQwen3MoEModel{driver: driver, cfg: geometry, weights: weights}

	prompt := []int32{1, 2, 3}
	first, err := sharedmodel.Generate(model, prompt, 3, -1)
	core.AssertNoError(t, err)
	second, err := sharedmodel.Generate(model, prompt, 3, -1)
	core.AssertNoError(t, err)
	core.AssertEqual(t, len(first), len(second))
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("greedy generation is not deterministic at token %d: %d != %d", i, first[i], second[i])
		}
	}
}

// qwen3MoETestTokenizerJSON is a minimal but COMPLETE (every vocab-size-16 token id
// covered) tokenizer.json — mirrors writeHIPMamba2TestModel's minimal fixture shape, sized
// to this model's vocab so a generated token id always decodes.
const qwen3MoETestTokenizerJSON = `{
  "model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7,"i":8,"j":9,"k":10,"l":11,"m":12,"n":13,"o":14,"p":15},"merges":[]},
  "added_tokens":[{"id":15,"content":"<eos>","special":true}]
}`

// qwen3MoETestConfigJSON is tinyQwen3MoEGeometry's geometry as a real checkpoint
// config.json — the shape loadHIPQwen3MoETextModel's ParseDenseConfig call parses.
const qwen3MoETestConfigJSON = `{
  "model_type":"qwen3_moe",
  "hidden_size":8,
  "intermediate_size":10,
  "moe_intermediate_size":6,
  "num_hidden_layers":1,
  "num_attention_heads":2,
  "num_key_value_heads":1,
  "head_dim":4,
  "num_experts":4,
  "num_experts_per_tok":2,
  "vocab_size":16,
  "norm_topk_prob":true,
  "rms_norm_eps":0.00001,
  "rope_theta":10000,
  "max_position_embeddings":2048
}`

// writeHIPQwen3MoETestModel materialises a complete on-disk checkpoint directory (real
// .safetensors file, config.json, tokenizer.json) — the SAME fixture shape
// writeHIPMamba2TestModel (mamba2_runtime_test.go) uses for its own loader smoke, so
// loadHIPQwen3MoETextModel is exercised through the identical on-disk path a real
// checkpoint takes, not just the in-memory tensor map TestLoadHIPQwen3MoEWeights_Good and
// TestHIPQwen3MoEModel_GenerateGreedy_Good use.
func writeHIPQwen3MoETestModel(t *testing.T) string {
	t.Helper()
	geometry := tinyQwen3MoEGeometry()
	tensors := tinyQwen3MoETensors(geometry.HiddenSize, geometry.VocabSize, geometry.ExpertFF, geometry.Heads, geometry.KVHeads, geometry.HeadDim, geometry.NumExperts, geometry.NumLayers)
	encoded, err := safetensors.Encode(tensors)
	core.RequireNoError(t, err)
	dir := t.TempDir()
	core.RequireNoError(t, coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(encoded)))
	core.RequireNoError(t, coreio.Local.Write(core.PathJoin(dir, "config.json"), qwen3MoETestConfigJSON))
	core.RequireNoError(t, coreio.Local.Write(core.PathJoin(dir, "tokenizer.json"), qwen3MoETestTokenizerJSON))
	return dir
}

// TestROCmBackend_LoadsQwen3MoEBeforeNativeRuntime_Good is the full end-to-end loader
// smoke: the real backend entry point (loadModelWithROCmConfigMode, the same one a live
// `lem` load call reaches), a real on-disk directory (safetensors + config.json +
// tokenizer.json, not an in-memory tensor map), text prompt in, greedy generation out —
// proving loadHIPQwen3MoETextModel's directory probe + tokenizer wiring (the part
// TestHIPQwen3MoEModel_GenerateGreedy_Good's in-memory construction does NOT cover) work,
// and that the native GGUF runtime is never reached (hipComposedRouteRuntime errors if it
// is — the SAME regression shape TestROCmBackend_LoadsMamba2BeforeNativeRuntime_Good
// proves for Mamba2).
//
// Hardware-gated (unlike every other loader-shaped test in this package): this path goes
// through loadHIPQwen3MoETextModel's own newSystemHIPDriver() call — a REAL driver, not a
// fakeHIPDriver — so generation needs the linked custom-kernel HSACO the same way the
// hip_qwen3_moe_hardware_test.go tier does; loading the checkpoint (config, tensors,
// tokenizer) itself needs no custom kernels and would succeed either way.
func TestROCmBackend_LoadsQwen3MoEBeforeNativeRuntime_Good(t *testing.T) {
	hipQwen3MoESkipUnlessHardware(t)
	dir := writeHIPQwen3MoETestModel(t)
	runtime := &hipComposedRouteRuntime{}
	loaded, err := newROCmBackendWithRuntime(runtime).loadModelWithROCmConfigMode(
		dir,
		inference.LoadConfig{},
		ROCmLoadConfig{},
		false,
	)
	core.RequireNoError(t, err)
	defer loaded.Close()
	core.AssertEqual(t, 0, runtime.calls)
	core.AssertEqual(t, "qwen3_moe", loaded.ModelType())
	core.AssertEqual(t, inference.ModelInfo{Architecture: "qwen3_moe", VocabSize: 16, NumLayers: 1, HiddenSize: 8}, loaded.Info())

	var generated []int32
	for token := range loaded.Generate(context.Background(), "a", inference.WithMaxTokens(3)) {
		generated = append(generated, token.ID)
	}
	if result := loaded.(*engine.TextModel).Err(); !result.OK {
		t.Fatalf("generation failed: %+v", result)
	}
	core.AssertEqual(t, 3, len(generated))
	for _, id := range generated {
		if id < 0 || id >= 16 {
			t.Fatalf("generated token %d is out of the tokenizer's vocabulary", id)
		}
	}
}
