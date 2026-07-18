// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// TestAssembleMoE_SharedExpert_Good drives Assemble over a single MoE layer carrying the qwen3_5_moe shared
// expert (router + the shared_expert SwiGLU trio + the σ gate), asserting the shared-expert branch loads
// each weight as a Linear of the width the tensor shape declares, and that the absent routed-expert weights
// stay nil (the assembler's nil-safe rule). This is the build-side receipt for the shared-expert load path;
// gemma leaves every Shared* name "" so its MoE layers are byte-identical (the shared fields stay nil).
func TestAssembleMoE_SharedExpert_Good(t *testing.T) {
	mat := func(rows, cols int) safetensors.Tensor {
		return safetensors.Tensor{Shape: []int{rows, cols}, Data: make([]byte, rows*cols*2), Dtype: "BF16"}
	}
	vec := func(n int) safetensors.Tensor {
		return safetensors.Tensor{Shape: []int{n}, Data: make([]byte, n*2), Dtype: "BF16"}
	}
	const (
		d  = 4 // hidden
		ff = 8 // shared-expert SwiGLU intermediate width
	)
	tensors := map[string]safetensors.Tensor{
		"embed.weight":                  mat(8, d),
		"norm.weight":                   vec(d),
		"layer.0.attn_norm.weight":      vec(d),
		"layer.0.q.weight":              mat(d, d),
		"layer.0.k.weight":              mat(d, d),
		"layer.0.o.weight":              mat(d, d),
		"layer.0.router.weight":         mat(2, d), // 2 routed experts (weights absent → nil below)
		"layer.0.shared.gate.weight":    mat(ff, d),
		"layer.0.shared.up.weight":      mat(ff, d),
		"layer.0.shared.down.weight":    mat(d, ff),
		"layer.0.shared.sigmoid.weight": mat(1, d), // σ gate: hidden → 1
	}
	names := WeightNames{
		Embed: "embed", FinalNorm: "norm.weight", LayerPrefix: "layer.%d",
		AttnNorm: ".attn_norm.weight", Q: ".q", K: ".k", O: ".o",
		MoE: MoEWeightNames{
			Router:     ".router",
			SharedGate: ".shared.gate", SharedUp: ".shared.up", SharedDown: ".shared.down",
			SharedSigmoid: ".shared.sigmoid",
		},
	}
	arch := Arch{
		Hidden: d, Heads: 2, Experts: 2, TopK: 1, ExpertFF: ff,
		Layer: []LayerSpec{{Attention: GlobalAttention, CacheIndex: 0, KVShareFrom: 0, HeadDim: 2, KVHeads: 2, MoE: true}},
	}
	m, err := Assemble(tensors, arch, names)
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	L := m.Layers[0]
	if L.MoE == nil {
		t.Fatal("layer 0 MoE nil, want a MoE layer routed through assembleMoE")
	}
	if L.MoE.SharedGate == nil || L.MoE.SharedUp == nil || L.MoE.SharedDown == nil {
		t.Fatalf("shared SwiGLU trio not loaded: gate=%v up=%v down=%v", L.MoE.SharedGate, L.MoE.SharedUp, L.MoE.SharedDown)
	}
	if L.MoE.SharedSigmoid == nil {
		t.Fatal("shared_expert_gate σ not loaded")
	}
	if L.MoE.SharedSigmoid.InDim != d || L.MoE.SharedSigmoid.OutDim != 1 {
		t.Errorf("σ gate dims = %d→%d, want %d→1", L.MoE.SharedSigmoid.InDim, L.MoE.SharedSigmoid.OutDim, d)
	}
	if L.MoE.SharedDown.InDim != ff {
		t.Errorf("shared down InDim = %d, want %d (the SwiGLU ff width read from the shape)", L.MoE.SharedDown.InDim, ff)
	}
	if L.MoE.SharedGate.OutDim != ff {
		t.Errorf("shared gate OutDim = %d, want %d", L.MoE.SharedGate.OutDim, ff)
	}
	// Routed experts absent → nil, isolating the shared-expert branch (the nil-safe rule).
	if L.MoE.ExpGate != nil || L.MoE.ExpDown != nil {
		t.Error("routed expert weights should be nil when absent (nil-safe)")
	}
}
