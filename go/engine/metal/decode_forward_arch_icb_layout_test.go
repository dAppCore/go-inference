// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

// TestResolveArchICBOpLayout_Good: the full-feature fused stack (the gemma4-assistant
// shape — all four norms + value-norm declared, PLE present, every fused kernel
// available) resolves to the layout the E2B recorder actually allocates: 21 ops/layer,
// with the K cache-row rebind moved to buffer index 2 by the fused kNorm+rope op.
func TestResolveArchICBOpLayout_Good(t *testing.T) {
	specs := []model.LayerSpec{{
		AttentionQNorm: true, AttentionKNorm: true, PostAttnNorm: true, PostFFNorm: true,
	}}
	lay := resolveArchICBOpLayout(specs, archICBOpCaps{
		valueNorm: true, ple: true,
		fusedGELU: true, fusedQKRope: true, fusedResRMS: true,
	})
	if !lay.hasQN || !lay.hasKN || !lay.hasPA || !lay.hasPF {
		t.Fatalf("declared norm selections must resolve on: %+v", lay)
	}
	// 24 base + 4 norms + 1 valueNorm − 9 fusedGELU − 2 fusedQKRope + 5 fused PLE − 2 fusedResRMS
	if lay.opsPerLayer != 21 {
		t.Fatalf("opsPerLayer = %d, want 21 (the E2B fused full-feature stride)", lay.opsPerLayer)
	}
	if lay.kRopeBindIdx != 2 {
		t.Fatalf("kRopeBindIdx = %d, want 2 (fused kNorm+rope writes the cache at its out slot)", lay.kRopeBindIdx)
	}
}

// TestResolveArchICBOpLayout_Bad: a hand-built caller declaring NOTHING with no fused
// kernels gets the plain composed layout — 24 ops/layer, cache-row rebind at index 1 —
// and no optional stage sneaks in without a declaration or a buffer.
func TestResolveArchICBOpLayout_Bad(t *testing.T) {
	lay := resolveArchICBOpLayout([]model.LayerSpec{{}}, archICBOpCaps{})
	if lay.hasQN || lay.hasKN || lay.hasPA || lay.hasPF {
		t.Fatalf("nothing declared, nothing built — no norm op may resolve on: %+v", lay)
	}
	if lay.opsPerLayer != 24 {
		t.Fatalf("opsPerLayer = %d, want the plain composed 24", lay.opsPerLayer)
	}
	if lay.kRopeBindIdx != 1 {
		t.Fatalf("kRopeBindIdx = %d, want 1 (plain rope writes the cache at index 1)", lay.kRopeBindIdx)
	}
}

// TestResolveArchICBOpLayout_Ugly: two contracts that keep hand-built callers and
// capability probes honest. (a) declared-versus-self-heal equivalence — a caller that
// only hands buffers (specs say nothing) resolves the IDENTICAL layout to one that
// declares on its specs (the slice-2/3 bind-declared-or-infer contract). (b) an engine
// capability without its op selection is inert — fusedQKRope on a stack with no QK
// norms changes nothing and leaves the rebind at index 1.
func TestResolveArchICBOpLayout_Ugly(t *testing.T) {
	declared := resolveArchICBOpLayout([]model.LayerSpec{{
		AttentionQNorm: true, AttentionKNorm: true, PostAttnNorm: true, PostFFNorm: true,
	}}, archICBOpCaps{layerScalar: true, fusedGELU: true, fusedQKRope: true})
	healed := resolveArchICBOpLayout([]model.LayerSpec{{}}, archICBOpCaps{
		qNormBuf: true, kNormBuf: true, postAttnBuf: true, postFFBuf: true,
		layerScalar: true, fusedGELU: true, fusedQKRope: true,
	})
	if declared != healed {
		t.Fatalf("declared %+v != buffer-healed %+v — the bind-declared-or-infer contract broke", declared, healed)
	}
	plain := resolveArchICBOpLayout([]model.LayerSpec{{}}, archICBOpCaps{})
	inert := resolveArchICBOpLayout([]model.LayerSpec{{}}, archICBOpCaps{fusedQKRope: true})
	if inert != plain {
		t.Fatalf("fusedQKRope with no QK norms must be inert: %+v vs %+v", inert, plain)
	}
}

// TestArchICBOpLayoutTotal_Good: the whole-stack command count is the uniform per-layer
// stride times the layer count plus the stack-global additions — the 2-pass SDPA op per
// recorded GLOBAL layer, the two quantise-store ops per q8 owner, and the TurboQuant
// lane's ops (2 stores per TQ owner + rot/unrot per TQ-reading layer). This total is what
// the recorder allocates and what the record loop's running counter must land on.
func TestArchICBOpLayoutTotal_Good(t *testing.T) {
	lay := archICBOpLayout{opsPerLayer: 24}
	if got := lay.total(30, 0, 0, 0); got != 720 {
		t.Fatalf("uniform total = %d, want 720", got)
	}
	// 30 layers, 6 global layers recording 2-pass SDPA, 6 q8 owners (2 stores each)
	if got := lay.total(30, 6, 12, 0); got != 738 {
		t.Fatalf("total with stack-global additions = %d, want 738", got)
	}
	// the TurboQuant lane instead: 6 TQ owners = 12 store ops + 6 reading layers × (rot+unrot) = 24
	if got := lay.total(30, 6, 0, 24); got != 750 {
		t.Fatalf("total with TQ additions = %d, want 750", got)
	}
}
