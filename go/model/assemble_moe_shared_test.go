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

// TestAssembleMoE_NoDistinctSharedWidth_Good is the #57 zero-change guarantee: an arch WITHOUT a
// distinct shared-expert width (Arch.SharedExpertFF left at its zero value — every arch but qwenmoe's
// shared-expert family, today) must derive byte-identical MoE geometry to before #57 added the field.
// Shaped after GraniteMoE (see model/arch/ibm-granite/granitemoe: fused ExpGateUp naming, no shared
// expert at all — GraniteMoE's own TestFactoryWeightNames_Bad guards that its WeightNames never grows
// Shared* names) rather than editing that package directly: read-only reference, not a fixture change.
// ExpGateUp/ExpDown are the SAME two lines assembleMoE used before #57 (unconditionally arch.ExpertFF/d,
// never touched by the SharedExpertFF branch) — this test locks that so a future change to assembleMoE
// can't accidentally couple them to the new field.
func TestAssembleMoE_NoDistinctSharedWidth_Good(t *testing.T) {
	mat := func(rows, cols int) safetensors.Tensor {
		return safetensors.Tensor{Shape: []int{rows, cols}, Data: make([]byte, rows*cols*2), Dtype: "BF16"}
	}
	vec := func(n int) safetensors.Tensor {
		return safetensors.Tensor{Shape: []int{n}, Data: make([]byte, n*2), Dtype: "BF16"}
	}
	const (
		d       = 8 // hidden
		ff      = 4 // routed experts' intermediate size (arch.ExpertFF)
		experts = 4
	)
	tensors := map[string]safetensors.Tensor{
		"embed.weight":                 mat(16, d),
		"norm.weight":                  vec(d),
		"layer.0.attn_norm.weight":     vec(d),
		"layer.0.q.weight":             mat(d, d),
		"layer.0.k.weight":             mat(d, d),
		"layer.0.o.weight":             mat(d, d),
		"layer.0.router.weight":        mat(experts, d),
		"layer.0.expert_gateup.weight": mat(experts*ff*2, d), // GraniteMoE-shaped: gate‖up fused, no separate roles
		"layer.0.expert_down.weight":   mat(d, ff),
	}
	names := WeightNames{
		Embed: "embed", FinalNorm: "norm.weight", LayerPrefix: "layer.%d",
		AttnNorm: ".attn_norm.weight", Q: ".q", K: ".k", O: ".o",
		MoE: MoEWeightNames{
			Router:    ".router",
			ExpGateUp: ".expert_gateup",
			ExpDown:   ".expert_down",
			// SharedGate/Up/Down/SharedSigmoid deliberately left "" — GraniteMoE carries no shared expert.
		},
	}
	arch := Arch{
		Hidden: d, Heads: 2, Experts: experts, TopK: 2, ExpertFF: ff, // SharedExpertFF left unset (0)
		Layer: []LayerSpec{{Attention: GlobalAttention, CacheIndex: 0, KVShareFrom: 0, HeadDim: 4, KVHeads: 2, MoE: true}},
	}
	if arch.SharedExpertFF != 0 {
		t.Fatalf("test setup: SharedExpertFF = %d, want the zero value (this fixture proves the unset-field path)", arch.SharedExpertFF)
	}
	m, err := Assemble(tensors, arch, names)
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	L := m.Layers[0]
	if L.MoE == nil {
		t.Fatal("layer 0 MoE nil, want a MoE layer routed through assembleMoE")
	}
	// The routed-expert geometry (the two lines assembleMoE has ALWAYS derived from arch.ExpertFF/d) is
	// unchanged by #57 — SharedExpertFF plays no part when an arch never populates it.
	if L.MoE.ExpGateUp == nil || L.MoE.ExpGateUp.InDim != d {
		t.Fatalf("ExpGateUp = %+v, want a Linear with InDim %d (hidden, unaffected by #57)", L.MoE.ExpGateUp, d)
	}
	if L.MoE.ExpDown == nil || L.MoE.ExpDown.InDim != ff {
		t.Fatalf("ExpDown = %+v, want a Linear with InDim %d (arch.ExpertFF, unaffected by #57)", L.MoE.ExpDown, ff)
	}
	// No shared expert declared → every Shared* field stays nil, exactly as before #57 (the new field
	// has nothing to plumb: assembleMoE's sharedFF fallback is computed but never consulted, since
	// SharedDown's name is "" and LoadLinear is nil-safe on an absent name).
	if L.MoE.SharedGate != nil || L.MoE.SharedUp != nil || L.MoE.SharedDown != nil || L.MoE.SharedSigmoid != nil {
		t.Fatalf("Shared* fields = gate %v up %v down %v sigmoid %v, want all nil (no shared expert declared)",
			L.MoE.SharedGate, L.MoE.SharedUp, L.MoE.SharedDown, L.MoE.SharedSigmoid)
	}
}
