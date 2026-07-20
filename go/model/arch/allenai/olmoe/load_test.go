// SPDX-Licence-Identifier: EUPL-1.2

package olmoe_test

import (
	"bytes"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// tinyOLMoEConfig is the same synthetic checkpoint shape TestTinyOLMoEForward_Good (the Composed route)
// exercises: 1 layer, 4 experts, top-2. Reused here so the factory route below is proven against the
// identical fixture the composed route already passes — same tensors, same config, two different load
// paths.
const tinyOLMoEConfig = `{"model_type":"olmoe","hidden_size":8,"intermediate_size":12,"num_hidden_layers":1,` +
	`"num_attention_heads":2,"num_key_value_heads":1,"num_experts":4,"num_experts_per_tok":2,"vocab_size":32,` +
	`"rms_norm_eps":1e-5,"rope_theta":10000,"norm_topk_prob":false,"tie_word_embeddings":false}`

// writeTinyOLMoEDir materialises tinyOLMoEWeights() + tinyOLMoEConfig as an on-disk checkpoint directory —
// OLMoE's real HF naming needs no NormalizeWeights step, so this is already the layout model.Load reads
// straight off disk.
func writeTinyOLMoEDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), tinyOLMoEConfig); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	blob, err := safetensors.Encode(tinyOLMoEWeights())
	if err != nil {
		t.Fatalf("encode weights: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir
}

// TestTinyOLMoEFactoryLoad_Good is the #50 bar for this arch: model.Load (the factory route —
// model.Assemble, no model/composed involved) now succeeds for OLMoE, where previously spec.Weights was the
// WeightNames zero value (every lookup missed) and model.Assemble rejected the checkpoint outright. It also
// proves the first half of the #18 parity method — "same tensor maps": the packed MoE expert weights
// model.Assemble loads are byte-identical to the checkpoint's own per-expert tensors, concatenated in
// expert-index order, not re-derived or approximated.
func TestTinyOLMoEFactoryLoad_Good(t *testing.T) {
	dir := writeTinyOLMoEDir(t)

	loaded, mapping, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load: %v (OLMoE must no longer require model/composed to load)", err)
	}
	defer func() { _ = mapping.Close() }()

	const hidden, ff, experts = 8, 12, 4
	if loaded.Arch.Experts != experts || loaded.Arch.TopK != 2 || loaded.Arch.ExpertFF != ff {
		t.Fatalf("Arch MoE geometry = experts %d topK %d expertFF %d, want %d/2/%d", loaded.Arch.Experts, loaded.Arch.TopK, loaded.Arch.ExpertFF, experts, ff)
	}
	if len(loaded.Layers) != 1 {
		t.Fatalf("layers = %d, want 1", len(loaded.Layers))
	}
	L := loaded.Layers[0]
	if L.MoE == nil {
		t.Fatal("layer 0 MoE nil — Assemble did not route through the MoE branch")
	}
	if L.MoE.Router == nil || L.MoE.ExpGate == nil || L.MoE.ExpUp == nil || L.MoE.ExpDown == nil {
		t.Fatalf("MoE weights not loaded: router=%v gate=%v up=%v down=%v", L.MoE.Router, L.MoE.ExpGate, L.MoE.ExpUp, L.MoE.ExpDown)
	}
	if L.MoE.ExpGate.OutDim != experts*ff || L.MoE.ExpGate.InDim != hidden {
		t.Errorf("ExpGate dims = %d->%d, want %d->%d (packed across %d experts)", L.MoE.ExpGate.InDim, L.MoE.ExpGate.OutDim, hidden, experts*ff, experts)
	}
	if L.MoE.ExpDown.OutDim != experts*hidden || L.MoE.ExpDown.InDim != ff {
		t.Errorf("ExpDown dims = %d->%d, want %d->%d", L.MoE.ExpDown.InDim, L.MoE.ExpDown.OutDim, ff, experts*hidden)
	}
	// Dense-MLP fields stay unset on a MoE layer (Assemble's spec.MoE branch never touches them).
	if L.Gate != nil || L.Up != nil || L.Down != nil {
		t.Error("a MoE layer should leave the dense Gate/Up/Down fields nil")
	}
	// OLMoE's per-head QK-norm must flow through the factory route too (StandardWeightNames' default
	// names — FactoryWeightNames does not override them).
	if L.QNorm == nil || L.KNorm == nil {
		t.Error("OLMoE's per-head QK-norm should load through the standard QNorm/KNorm names")
	}

	// "same tensor maps": packed bytes == the source per-expert tensors, concatenated in index order.
	tensors := tinyOLMoEWeights()
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
}

// TestTinyOLMoEFactoryAndComposed_AgreeOnRegistration proves OLMoE's dual route: the SAME model_type
// resolves to a spec carrying BOTH Parse+Weights (the factory route this file's Good test exercises) AND a
// working Composed hook (the existing TestTinyOLMoEForward_Good route in integration_test.go) — composed is
// demoted to an available alternative, not removed, mirroring the mixtral/qwen35 dual-route pattern
// (#18/#50).
func TestTinyOLMoEFactoryAndComposed_AgreeOnRegistration(t *testing.T) {
	spec, ok := model.LookupArch("olmoe")
	if !ok {
		t.Fatal("olmoe not registered")
	}
	if spec.Parse == nil || spec.Weights.MoE.ExpGate == "" {
		t.Fatal("olmoe spec missing the factory route (Parse + Weights)")
	}
	if spec.Composed == nil {
		t.Fatal("olmoe spec missing the Composed route — composed must stay available, not be removed")
	}

	dir := writeTinyOLMoEDir(t)
	if _, _, err := model.Load(dir); err != nil {
		t.Fatalf("factory route: %v", err)
	}
	tm, ok, err := model.LoadComposedDir(dir)
	if err != nil || !ok || tm == nil {
		t.Fatalf("composed route: ok=%v err=%v tm=%v — the escape hatch must still work", ok, err, tm)
	}
}
