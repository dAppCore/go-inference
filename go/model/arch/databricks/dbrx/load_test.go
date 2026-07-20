// SPDX-Licence-Identifier: EUPL-1.2

package dbrx

import (
	"bytes"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// tinyDBRXConfig is the same synthetic checkpoint shape TestTinyDBRXForward_Good (the Composed route)
// exercises: 1 layer, 4 experts, top-2. Reused here so the factory route below is proven against the
// identical fixture the composed route already passes — same tensors, same config, two different load
// paths.
const tinyDBRXConfig = `{"model_type":"dbrx","d_model":8,"n_heads":2,"n_layers":1,"vocab_size":32,` +
	`"attn_config":{"kv_n_heads":1,"rope_theta":10000},"ffn_config":{"ffn_hidden_size":12,"moe_num_experts":4,"moe_top_k":2}}`

// writeTinyDBRXDir materialises tinyDBRXWeights(fill) + tinyDBRXConfig as an on-disk checkpoint directory —
// the RAW fused-tensor layout (transformer.wte.weight, ffn.experts.mlp.w1/v1/w2, …) model.Load reads
// straight off disk; model.Load runs NormalizeConfig (NormalizeWeights + packExperts) itself, so this
// fixture must NOT be pre-normalised.
func writeTinyDBRXDir(t *testing.T, fill float32) string {
	t.Helper()
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), tinyDBRXConfig); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	blob, err := safetensors.Encode(tinyDBRXWeights(fill))
	if err != nil {
		t.Fatalf("encode weights: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir
}

// TestTinyDBRXFactoryLoad_Good is the #50 bar for this arch: model.Load (the factory route —
// model.Assemble, no model/composed involved) now succeeds for DBRX, where previously spec.Weights was the
// WeightNames zero value (every lookup missed) and model.Assemble rejected the checkpoint outright. It also
// proves the first half of the #18 parity method — "same tensor maps": the packed MoE expert weights
// model.Assemble loads are byte-identical to NormalizeWeights' own (separately-tested) decode of the fused
// checkpoint tensors, concatenated in expert-index order, not re-derived or approximated a second way.
func TestTinyDBRXFactoryLoad_Good(t *testing.T) {
	dir := writeTinyDBRXDir(t, 0.02)

	loaded, mapping, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load: %v (DBRX must no longer require model/composed to load)", err)
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

	// "same tensor maps": the packed bytes equal NormalizeWeights' own per-expert decode of the fused
	// checkpoint, concatenated in index order — proving packExperts didn't silently re-derive the fused
	// w1/v1/w2 split differently from the tested Composed-route decode.
	var cfg Config
	if !core.JSONUnmarshal([]byte(tinyDBRXConfig), &cfg).OK {
		t.Fatal("tiny config parse")
	}
	normalised := NormalizeWeights(tinyDBRXWeights(0.02), cfg)
	var wantGate, wantUp, wantDown []byte
	for e := 0; e < experts; e++ {
		wantGate = append(wantGate, normalised[core.Sprintf("model.layers.0.mlp.experts.%d.gate_proj.weight", e)].Data...)
		wantUp = append(wantUp, normalised[core.Sprintf("model.layers.0.mlp.experts.%d.up_proj.weight", e)].Data...)
		wantDown = append(wantDown, normalised[core.Sprintf("model.layers.0.mlp.experts.%d.down_proj.weight", e)].Data...)
	}
	if !bytes.Equal(L.MoE.ExpGate.Weight, wantGate) {
		t.Error("ExpGate.Weight bytes != NormalizeWeights' per-expert gate_proj tensors concatenated in order")
	}
	if !bytes.Equal(L.MoE.ExpUp.Weight, wantUp) {
		t.Error("ExpUp.Weight bytes != NormalizeWeights' per-expert up_proj tensors concatenated in order")
	}
	if !bytes.Equal(L.MoE.ExpDown.Weight, wantDown) {
		t.Error("ExpDown.Weight bytes != NormalizeWeights' per-expert down_proj tensors concatenated in order")
	}
}

// TestTinyDBRXFactoryAndComposed_AgreeOnRegistration proves DBRX's dual route: the SAME model_type resolves
// to a spec carrying BOTH Parse+Weights (the factory route this file's Good test exercises) AND a working
// Composed hook (the existing TestTinyDBRXForward_Good route in integration_test.go) — composed is demoted
// to an available alternative, not removed, mirroring the mixtral/qwen35 dual-route pattern (#18/#50).
func TestTinyDBRXFactoryAndComposed_AgreeOnRegistration(t *testing.T) {
	spec, ok := model.LookupArch("dbrx")
	if !ok {
		t.Fatal("dbrx not registered")
	}
	if spec.Parse == nil || spec.Weights.MoE.ExpGate == "" {
		t.Fatal("dbrx spec missing the factory route (Parse + Weights)")
	}
	if spec.Composed == nil {
		t.Fatal("dbrx spec missing the Composed route — composed must stay available, not be removed")
	}

	dir := writeTinyDBRXDir(t, 0.02)
	if _, _, err := model.Load(dir); err != nil {
		t.Fatalf("factory route: %v", err)
	}
	tm, ok, err := model.LoadComposedDir(dir)
	if err != nil || !ok || tm == nil {
		t.Fatalf("composed route: ok=%v err=%v tm=%v — the escape hatch must still work", ok, err, tm)
	}
}
