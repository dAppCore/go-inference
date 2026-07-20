// SPDX-Licence-Identifier: EUPL-1.2

package qwenmoe_test

import (
	"bytes"
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
// model.Assemble + arch_session, no model/composed involved) now succeeds for Qwen2-MoE, where it used to
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
		t.Fatalf("model.Load: %v (Qwen2-MoE must no longer require model/composed to load)", err)
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
		t.Fatalf("model.Load: %v (Qwen3-MoE must no longer require model/composed to load)", err)
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

// TestQwenmoeFactoryAndComposed_AgreeOnRegistration proves both qwen2_moe and qwen3_moe carry the dual
// route: the SAME model_type resolves to a spec carrying BOTH Parse+Weights (the factory route this
// file's Good tests exercise) AND a working Composed hook (the existing TestQwen2MoEForward_Good/
// TestQwen3MoEForward_Good route in integration_test.go) — composed is demoted to an available
// alternative, not removed, exactly mirroring the qwen35/mixtral/granitemoe dual-route pattern (#18/#50).
func TestQwenmoeFactoryAndComposed_AgreeOnRegistration(t *testing.T) {
	for _, modelType := range []string{"qwen2_moe", "qwen3_moe"} {
		spec, ok := model.LookupArch(modelType)
		if !ok {
			t.Fatalf("%s not registered", modelType)
		}
		if spec.Parse == nil || spec.Weights.MoE.ExpGate == "" {
			t.Fatalf("%s spec missing the factory route (Parse + Weights)", modelType)
		}
		if spec.Composed == nil {
			t.Fatalf("%s spec missing the Composed route — composed must stay available, not be removed", modelType)
		}
	}

	tensors, _ := tinyQwen2MoEWeights()
	dir := writeTinyQwenMoEDir(t, tensors, tinyQwen2MoEConfig)
	if _, _, err := model.Load(dir); err != nil {
		t.Fatalf("factory route: %v", err)
	}
	tm, ok, err := model.LoadComposedDir(dir)
	if err != nil || !ok || tm == nil {
		t.Fatalf("composed route: ok=%v err=%v tm=%v — the escape hatch must still work", ok, err, tm)
	}
}
