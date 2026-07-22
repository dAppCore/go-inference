// SPDX-Licence-Identifier: EUPL-1.2

package qwenmoe_test

import (
	"bytes"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/Qwen/qwenmoe" // register the qwenmoe loaders
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// writeTinyQwenMoEDir materialises tensors + config as a checkpoint directory, so model.Load (a
// dir-reading entry point, unlike spec.Composed's in-memory tensors) can exercise the factory route.
func writeTinyQwenMoEDir(t *testing.T, tensors map[string]safetensors.Tensor, config string) string {
	t.Helper()
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("encode weights: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir
}

// tinyQwen2MoEConfig is the same synthetic checkpoint config TestQwen2MoEForward_Good (the Composed
// route, integration_test.go) exercises: 1 layer, 4 experts, top-2, WITH a shared expert whose
// intermediate size (10) deliberately differs from the routed experts' (6) — mirroring the real
// Qwen1.5-MoE-A2.7B ratio (moe_intermediate_size=1408 vs shared_expert_intermediate_size=5632, see
// weights.go's KNOWN LIMITATION doc). Reused here so the factory route below is proven against the
// identical fixture the composed route already passes — same tensors, same config, two load paths.
const tinyQwen2MoEConfig = `{"model_type":"qwen2_moe","hidden_size":8,"intermediate_size":10,"moe_intermediate_size":6,` +
	`"shared_expert_intermediate_size":10,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,` +
	`"num_experts":4,"num_experts_per_tok":2,"vocab_size":32,"norm_topk_prob":false,"tie_word_embeddings":false}`

// tinyQwen3MoEConfig mirrors TestQwen3MoEForward_Good's config: same tiny checkpoint MINUS the
// shared-expert tensors, matching real Qwen3-MoE (no shared expert).
const tinyQwen3MoEConfig = `{"model_type":"qwen3_moe","hidden_size":8,"intermediate_size":10,"moe_intermediate_size":6,` +
	`"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"num_experts":4,` +
	`"num_experts_per_tok":2,"vocab_size":32,"norm_topk_prob":true,"tie_word_embeddings":false}`

// TestTinyQwen2MoEFactoryLoad_Good is the #50 bar for this arch: model.Load (the factory route —
// model.Assemble + arch_session) now succeeds for Qwen2-MoE, where it used to
// fail (Weights was the zero-value model.WeightNames{}, so Assemble rejected the checkpoint as missing
// model.embed_tokens rather than routing it correctly). It also proves the "same tensor maps" parity
// method for the routed experts (packed bytes == the checkpoint's own per-expert tensors, concatenated in
// index order) and confirms the shared expert loads too (see weights.go's KNOWN LIMITATION doc for the
// one open gap: SharedDown's InDim metadata, not its Weight bytes, which this test DOES pin).
func TestTinyQwen2MoEFactoryLoad_Good(t *testing.T) {
	tensors, _ := tinyQwen2MoEWeights()
	dir := writeTinyQwenMoEDir(t, tensors, tinyQwen2MoEConfig)

	loaded, mapping, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load: %v (Qwen2-MoE must load through the factory route alone)", err)
	}
	defer func() { _ = mapping.Close() }()

	const hidden, experts, topK, expertFF = 8, 4, 2, 6
	if loaded.Arch.Experts != experts || loaded.Arch.TopK != topK || loaded.Arch.ExpertFF != expertFF {
		t.Fatalf("Arch MoE geometry = experts %d topK %d expertFF %d, want %d/%d/%d", loaded.Arch.Experts, loaded.Arch.TopK, loaded.Arch.ExpertFF, experts, topK, expertFF)
	}
	if loaded.Arch.SharedExperts != 1 {
		t.Fatalf("Arch.SharedExperts = %d, want 1 (Qwen2-MoE carries a shared expert)", loaded.Arch.SharedExperts)
	}
	if len(loaded.Layers) != 1 {
		t.Fatalf("layers = %d, want 1", len(loaded.Layers))
	}
	L := loaded.Layers[0]
	if L.MoE == nil {
		t.Fatal("layer 0 MoE nil — Assemble did not route through the MoE branch")
	}
	if L.MoE.Router == nil || L.MoE.ExpGate == nil || L.MoE.ExpUp == nil || L.MoE.ExpDown == nil {
		t.Fatalf("routed MoE weights not loaded: router=%v gate=%v up=%v down=%v", L.MoE.Router, L.MoE.ExpGate, L.MoE.ExpUp, L.MoE.ExpDown)
	}
	if L.MoE.SharedGate == nil || L.MoE.SharedUp == nil || L.MoE.SharedDown == nil || L.MoE.SharedSigmoid == nil {
		t.Fatalf("shared-expert weights not loaded: gate=%v up=%v down=%v sigmoid=%v", L.MoE.SharedGate, L.MoE.SharedUp, L.MoE.SharedDown, L.MoE.SharedSigmoid)
	}
	// Dense-MLP fields stay unset on a MoE layer (Assemble's spec.MoE branch never touches them).
	if L.Gate != nil || L.Up != nil || L.Down != nil {
		t.Error("a MoE layer should leave the dense Gate/Up/Down fields nil")
	}

	// "same tensor maps": routed-expert packed bytes == the source per-expert tensors, concatenated in
	// index order (packExperts' contract, already unit-tested directly in weights_test.go — this proves
	// the SAME contract holds end-to-end through model.Load).
	var wantGate, wantUp, wantDown []byte
	for e := 0; e < experts; e++ {
		wantGate = append(wantGate, tensors[core.Sprintf("model.layers.0.mlp.experts.%d.gate_proj.weight", e)].Data...)
		wantUp = append(wantUp, tensors[core.Sprintf("model.layers.0.mlp.experts.%d.up_proj.weight", e)].Data...)
		wantDown = append(wantDown, tensors[core.Sprintf("model.layers.0.mlp.experts.%d.down_proj.weight", e)].Data...)
	}
	if !bytes.Equal(L.MoE.ExpGate.Weight, wantGate) {
		t.Error("ExpGate.Weight bytes != source per-expert gate_proj tensors concatenated in order")
	}
	if !bytes.Equal(L.MoE.ExpUp.Weight, wantUp) {
		t.Error("ExpUp.Weight bytes != source per-expert up_proj tensors concatenated in order")
	}
	if !bytes.Equal(L.MoE.ExpDown.Weight, wantDown) {
		t.Error("ExpDown.Weight bytes != source per-expert down_proj tensors concatenated in order")
	}
	if !bytes.Equal(L.MoE.Router.Weight, tensors["model.layers.0.mlp.gate.weight"].Data) {
		t.Error("Router.Weight bytes != source mlp.gate.weight tensor")
	}

	// Shared expert: a direct alias (no packing), so bytes must be untouched regardless of the
	// documented InDim-metadata limitation.
	if !bytes.Equal(L.MoE.SharedGate.Weight, tensors["model.layers.0.mlp.shared_expert.gate_proj.weight"].Data) {
		t.Error("SharedGate.Weight bytes != source mlp.shared_expert.gate_proj.weight tensor")
	}
	if !bytes.Equal(L.MoE.SharedUp.Weight, tensors["model.layers.0.mlp.shared_expert.up_proj.weight"].Data) {
		t.Error("SharedUp.Weight bytes != source mlp.shared_expert.up_proj.weight tensor")
	}
	if !bytes.Equal(L.MoE.SharedDown.Weight, tensors["model.layers.0.mlp.shared_expert.down_proj.weight"].Data) {
		t.Error("SharedDown.Weight bytes != source mlp.shared_expert.down_proj.weight tensor")
	}
	if !bytes.Equal(L.MoE.SharedSigmoid.Weight, tensors["model.layers.0.mlp.shared_expert_gate.weight"].Data) {
		t.Error("SharedSigmoid.Weight bytes != source mlp.shared_expert_gate.weight tensor")
	}
	if hidden != loaded.Arch.Hidden {
		t.Fatalf("Arch.Hidden = %d, want %d", loaded.Arch.Hidden, hidden)
	}
}

// TestTinyQwen3MoEFactoryLoad_Good proves the factory route also serves Qwen3-MoE, which drops the
// shared expert entirely — the Shared* fields must stay nil (LoadLinear's nil-safe absence), not error.
func TestTinyQwen3MoEFactoryLoad_Good(t *testing.T) {
	tensors, _ := tinyQwen2MoEWeights()
	delete(tensors, "model.layers.0.mlp.shared_expert.gate_proj.weight")
	delete(tensors, "model.layers.0.mlp.shared_expert.up_proj.weight")
	delete(tensors, "model.layers.0.mlp.shared_expert.down_proj.weight")
	delete(tensors, "model.layers.0.mlp.shared_expert_gate.weight")
	dir := writeTinyQwenMoEDir(t, tensors, tinyQwen3MoEConfig)

	loaded, mapping, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load: %v (Qwen3-MoE must load through the factory route alone)", err)
	}
	defer func() { _ = mapping.Close() }()

	if loaded.Arch.SharedExperts != 0 {
		t.Fatalf("Arch.SharedExperts = %d, want 0 (Qwen3-MoE has no shared expert)", loaded.Arch.SharedExperts)
	}
	L := loaded.Layers[0]
	if L.MoE == nil {
		t.Fatal("layer 0 MoE nil")
	}
	if L.MoE.SharedGate != nil || L.MoE.SharedUp != nil || L.MoE.SharedDown != nil || L.MoE.SharedSigmoid != nil {
		t.Errorf("Qwen3-MoE Shared* fields = gate %v up %v down %v sigmoid %v, want all nil", L.MoE.SharedGate, L.MoE.SharedUp, L.MoE.SharedDown, L.MoE.SharedSigmoid)
	}
	if L.MoE.ExpGate == nil || L.MoE.ExpUp == nil || L.MoE.ExpDown == nil || L.MoE.Router == nil {
		t.Fatal("Qwen3-MoE routed-expert weights not loaded")
	}
}

// TestTinyQwen2MoEFactoryLoad_QuantisedSharedExpert_Good is the #57 regression: a shared expert whose
// down_proj is QUANTISED and whose intermediate size (24) genuinely differs from the routed experts'
// (6) — mirroring the real Qwen1.5-MoE-A2.7B ratio (moe_intermediate_size=1408 vs
// shared_expert_intermediate_size=5632, also 4x) — must derive its affine GroupSize/Bits from ITS OWN
// width, not the routed experts'. Before #57, assembleMoE forced SharedDown's declared inDim to
// arch.ExpertFF (6), so LoadLinear's affineGeometry(inDim=6, scalesShape=[8,3], weightShape=[8,3])
// silently produced GroupSize=2/Bits=16 — both wrong for a weight whose real width is 24. The fix
// resolves inDim from arch.SharedExpertFF (24), giving the correct GroupSize=8/Bits=4.
func TestTinyQwen2MoEFactoryLoad_QuantisedSharedExpert_Good(t *testing.T) {
	tensors := tinyQwen2MoEQuantSharedWeights()
	dir := writeTinyQwenMoEDir(t, tensors, tinyQwen2MoEQuantSharedConfig)

	loaded, mapping, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load: %v", err)
	}
	defer func() { _ = mapping.Close() }()

	const routedFF, sharedFF = 6, 24
	if loaded.Arch.ExpertFF != routedFF {
		t.Fatalf("Arch.ExpertFF = %d, want %d (the routed experts' width)", loaded.Arch.ExpertFF, routedFF)
	}
	if loaded.Arch.SharedExpertFF != sharedFF {
		t.Fatalf("Arch.SharedExpertFF = %d, want %d (the shared expert's OWN width)", loaded.Arch.SharedExpertFF, sharedFF)
	}

	L := loaded.Layers[0]
	if L.MoE == nil || L.MoE.SharedDown == nil {
		t.Fatal("layer 0 MoE.SharedDown nil — the quantised shared-expert weight did not load")
	}
	sd := L.MoE.SharedDown
	if sd.InDim != sharedFF {
		t.Fatalf("SharedDown.InDim = %d, want %d (the shared width — #57 used to force ExpertFF=%d here)", sd.InDim, sharedFF, routedFF)
	}
	if !sd.Quantised() {
		t.Fatal("SharedDown should be quantised (a .scales tensor is present)")
	}
	const wantGroupSize, wantBits = 8, 4
	if sd.GroupSize != wantGroupSize || sd.Bits != wantBits {
		t.Fatalf("SharedDown geometry = GroupSize %d Bits %d, want GroupSize %d Bits %d (affineGeometry(inDim=%d, ...) post-#57)",
			sd.GroupSize, sd.Bits, wantGroupSize, wantBits, sharedFF)
	}
	const wrongGroupSize, wrongBits = 2, 16 // affineGeometry(inDim=6, ...) — the pre-#57 bug value
	if sd.GroupSize == wrongGroupSize && sd.Bits == wrongBits {
		t.Fatalf("SharedDown geometry = GroupSize %d Bits %d — the PRE-#57 bug value (derived from the routed ExpertFF=%d instead of SharedExpertFF=%d)",
			sd.GroupSize, sd.Bits, routedFF, sharedFF)
	}

	// SharedGate/SharedUp project FROM hidden regardless of the shared FF width, so #57 never touched
	// them — confirm they still resolve to the ordinary dense (unquantised) InDim=hidden.
	if L.MoE.SharedGate == nil || L.MoE.SharedGate.InDim != 8 || L.MoE.SharedGate.Quantised() {
		t.Fatalf("SharedGate = %+v, want a dense Linear with InDim 8", L.MoE.SharedGate)
	}
	if L.MoE.SharedUp == nil || L.MoE.SharedUp.InDim != 8 || L.MoE.SharedUp.Quantised() {
		t.Fatalf("SharedUp = %+v, want a dense Linear with InDim 8", L.MoE.SharedUp)
	}

	// The routed experts' own geometry is untouched by #57 (assembleMoE still passes arch.ExpertFF for
	// ExpDown) — confirm no collateral change there.
	if L.MoE.ExpDown == nil || L.MoE.ExpDown.InDim != routedFF {
		t.Fatalf("ExpDown.InDim = %v, want %d (routed experts unaffected by the shared-FF fix)", L.MoE.ExpDown, routedFF)
	}
}

// --- synthetic checkpoint fixtures (moved from the deleted composed-route integration_test.go, #50) ---

func tinyQwen2MoEWeights() (map[string]safetensors.Tensor, []float32) {
	const hidden, vocab, expertFF, sharedFF, heads, kvHeads, headDim, experts = 8, 32, 6, 10, 2, 1, 4, 4
	s := seededWeights{state: 0x2a3e1234}
	router := s.values(experts * hidden)
	tensors := map[string]safetensors.Tensor{
		"model.embed_tokens.weight":                         f32Tensor(s.values(vocab*hidden), vocab, hidden),
		"model.norm.weight":                                 f32Tensor(s.values(hidden), hidden),
		"lm_head.weight":                                    f32Tensor(s.values(vocab*hidden), vocab, hidden),
		"model.layers.0.input_layernorm.weight":             f32Tensor(s.values(hidden), hidden),
		"model.layers.0.post_attention_layernorm.weight":    f32Tensor(s.values(hidden), hidden),
		"model.layers.0.self_attn.q_proj.weight":            f32Tensor(s.values(heads*headDim*hidden), heads*headDim, hidden),
		"model.layers.0.self_attn.k_proj.weight":            f32Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.v_proj.weight":            f32Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.o_proj.weight":            f32Tensor(s.values(hidden*heads*headDim), hidden, heads*headDim),
		"model.layers.0.mlp.gate.weight":                    f32Tensor(router, experts, hidden),
		"model.layers.0.mlp.shared_expert.gate_proj.weight": f32Tensor(s.values(sharedFF*hidden), sharedFF, hidden),
		"model.layers.0.mlp.shared_expert.up_proj.weight":   f32Tensor(s.values(sharedFF*hidden), sharedFF, hidden),
		"model.layers.0.mlp.shared_expert.down_proj.weight": f32Tensor(s.values(hidden*sharedFF), hidden, sharedFF),
		"model.layers.0.mlp.shared_expert_gate.weight":      f32Tensor(s.values(hidden), 1, hidden),
	}
	for expert := range experts {
		prefix := core.Sprintf("model.layers.0.mlp.experts.%d.", expert)
		tensors[prefix+"gate_proj.weight"] = f32Tensor(s.values(expertFF*hidden), expertFF, hidden)
		tensors[prefix+"up_proj.weight"] = f32Tensor(s.values(expertFF*hidden), expertFF, hidden)
		tensors[prefix+"down_proj.weight"] = f32Tensor(s.values(hidden*expertFF), hidden, expertFF)
	}
	return tensors, router
}

// tinyQwen2MoEQuantSharedConfig mirrors tinyQwen2MoEConfig but gives the shared expert its OWN
// intermediate size (24) that is a clean 4x multiple of the routed experts' (6) — mirroring the real
// Qwen1.5-MoE-A2.7B ratio (moe_intermediate_size=1408, shared_expert_intermediate_size=5632, also 4x) at
// a size small enough for a fast synthetic checkpoint. Feeds
// TestTinyQwen2MoEFactoryLoad_QuantisedSharedExpert_Good, the #57 regression.
const tinyQwen2MoEQuantSharedConfig = `{"model_type":"qwen2_moe","hidden_size":8,"intermediate_size":10,"moe_intermediate_size":6,` +
	`"shared_expert_intermediate_size":24,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,` +
	`"num_experts":4,"num_experts_per_tok":2,"vocab_size":32,"norm_topk_prob":false,"tie_word_embeddings":false}`

// quantTensor builds a placeholder affine-quantised tensor: dtype and shape are what safetensors.Encode
// validates (byte span == dtype size × ∏shape) and what LoadLinear's affineGeometry reads its GroupSize/
// Bits derivation from; the byte VALUES are irrelevant to a geometry-only fixture (nothing here
// dequantises a real number), so they are left zero.
func quantTensor(dtype string, elemBytes int, shape ...int) safetensors.Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return safetensors.Tensor{Dtype: dtype, Shape: shape, Data: make([]byte, n*elemBytes)}
}

// tinyQwen2MoEQuantSharedWeights is tinyQwen2MoEWeights' shape with the shared expert's down_proj
// QUANTISED 4-bit/group-8 at its OWN width (sharedFF=24, 4x the routed experts' expertFF=6) — the exact
// shape #57 got wrong. SharedDown is [hidden=8, sharedFF=24] logically: packed 4-bit codes are
// [8, sharedFF·4/32] = [8,3] U32 (8 int4 codes per uint32 row-word); scales/biases are one value per
// group of 8 along the 24-wide input, [8, sharedFF/8] = [8,3] BF16. affineGeometry(inDim=24, scales
// shape [8,3], weight shape [8,3]) derives GroupSize=24/3=8, Bits=3·32/24=4 — the correct geometry. The
// pre-#57 bug passed inDim=arch.ExpertFF=6 (the ROUTED width) into that same shape pair, giving
// GroupSize=6/3=2, Bits=3·32/6=16 — both wrong, silently.
//
// SharedGate/SharedUp stay dense F32 (their InDim is always the hidden size, unaffected by the shared-FF
// width regardless of the bug), isolating the fixture to the one weight #57 actually broke.
func tinyQwen2MoEQuantSharedWeights() map[string]safetensors.Tensor {
	const hidden, vocab, expertFF, sharedFF, heads, kvHeads, headDim, experts = 8, 32, 6, 24, 2, 1, 4, 4
	const packedCols, nGroups = 3, 3 // sharedFF·4/32 = 24·4/32 = 3; sharedFF/groupSize(8) = 24/8 = 3
	s := seededWeights{state: 0x51a7e001}
	router := s.values(experts * hidden)
	tensors := map[string]safetensors.Tensor{
		"model.embed_tokens.weight":                         f32Tensor(s.values(vocab*hidden), vocab, hidden),
		"model.norm.weight":                                 f32Tensor(s.values(hidden), hidden),
		"lm_head.weight":                                    f32Tensor(s.values(vocab*hidden), vocab, hidden),
		"model.layers.0.input_layernorm.weight":             f32Tensor(s.values(hidden), hidden),
		"model.layers.0.post_attention_layernorm.weight":    f32Tensor(s.values(hidden), hidden),
		"model.layers.0.self_attn.q_proj.weight":            f32Tensor(s.values(heads*headDim*hidden), heads*headDim, hidden),
		"model.layers.0.self_attn.k_proj.weight":            f32Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.v_proj.weight":            f32Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.o_proj.weight":            f32Tensor(s.values(hidden*heads*headDim), hidden, heads*headDim),
		"model.layers.0.mlp.gate.weight":                    f32Tensor(router, experts, hidden),
		"model.layers.0.mlp.shared_expert.gate_proj.weight": f32Tensor(s.values(sharedFF*hidden), sharedFF, hidden),
		"model.layers.0.mlp.shared_expert.up_proj.weight":   f32Tensor(s.values(sharedFF*hidden), sharedFF, hidden),
		// SharedDown: QUANTISED 4-bit/group-8 at the shared width (24), not the routed width (6) — the
		// #57 regression proof (see the function doc for the arithmetic).
		"model.layers.0.mlp.shared_expert.down_proj.weight": quantTensor("U32", 4, hidden, packedCols),
		"model.layers.0.mlp.shared_expert.down_proj.scales": quantTensor("BF16", 2, hidden, nGroups),
		"model.layers.0.mlp.shared_expert.down_proj.biases": quantTensor("BF16", 2, hidden, nGroups),
		"model.layers.0.mlp.shared_expert_gate.weight":      f32Tensor(s.values(hidden), 1, hidden),
	}
	for expert := range experts {
		prefix := core.Sprintf("model.layers.0.mlp.experts.%d.", expert)
		tensors[prefix+"gate_proj.weight"] = f32Tensor(s.values(expertFF*hidden), expertFF, hidden)
		tensors[prefix+"up_proj.weight"] = f32Tensor(s.values(expertFF*hidden), expertFF, hidden)
		tensors[prefix+"down_proj.weight"] = f32Tensor(s.values(hidden*expertFF), hidden, expertFF)
	}
	return tensors
}

type seededWeights struct{ state uint32 }

func f32Tensor(values []float32, shape ...int) safetensors.Tensor {
	data := make([]byte, len(values)*4)
	for i, value := range values {
		bits := math.Float32bits(value)
		data[4*i], data[4*i+1], data[4*i+2], data[4*i+3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
	}
	return safetensors.Tensor{Dtype: "F32", Shape: shape, Data: data}
}

func (s *seededWeights) values(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		s.state = 1664525*s.state + 1013904223
		out[i] = float32(int32(s.state>>24)-128) / 512
	}
	return out
}
