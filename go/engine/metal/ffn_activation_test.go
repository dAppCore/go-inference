// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

// TestFFNActivation_ffnUsesSiLU pins the feed-forward activation selector: SiLU/swish declared
// activations (llama/mistral/qwen SwiGLU) route to the silu gate; everything else — including the
// unset default and the gemma GELU spellings — keeps GELU, so the gemma path stays byte-identical.
func TestFFNActivation_ffnUsesSiLU(t *testing.T) {
	silu := []string{"silu", "swish"}
	gelu := []string{"", "gelu", "gelu_pytorch_tanh", "gelu_new", "relu"}
	for _, a := range silu {
		if !ffnUsesSiLU(a) {
			t.Errorf("ffnUsesSiLU(%q) = false, want true", a)
		}
	}
	for _, a := range gelu {
		if ffnUsesSiLU(a) {
			t.Errorf("ffnUsesSiLU(%q) = true, want false (gemma/GELU path must stay unchanged)", a)
		}
	}
}

// TestFFNActivation_archRopeScale pins the RoPE position scale resolver: it returns arch.RopeScale,
// defaulting an unset (0) arch to 1.0 (standard rope) — NEVER the attention scale. This is the
// value the decode feeds the rope kernel (angle = ropeScale·pos·inv_freq), distinct from AttnScale.
func TestFFNActivation_archRopeScale(t *testing.T) {
	if got := archRopeScale(model.Arch{RopeScale: 0}); got != 1 {
		t.Errorf("archRopeScale(unset) = %v, want 1", got)
	}
	if got := archRopeScale(model.Arch{RopeScale: 0.25}); got != 0.25 {
		t.Errorf("archRopeScale(0.25) = %v, want 0.25", got)
	}
	// A qwen-like arch: attention scale 1/√headDim, rope scale 1 — the resolver must NOT return AttnScale.
	if got := archRopeScale(model.Arch{RopeScale: 1, AttnScale: 0.08838835}); got != 1 {
		t.Errorf("archRopeScale(qwen-like) = %v, want 1 (not the attention scale)", got)
	}
}

// TestFFNActivation_projectorSiLUFlag checks the projector carries the SwiGLU flag through withSiLU:
// the seam the session constructor stamps so the MLP encode picks silu(gate)·up. A gemma projector
// (flag unset) reports GELU.
func TestFFNActivation_projectorSiLUFlag(t *testing.T) {
	var bf bf16Projector
	if bf.usesSiLU() {
		t.Fatal("bf16Projector default usesSiLU() = true, want false (gemma GELU)")
	}
	if !bf.withSiLU(true).usesSiLU() {
		t.Fatal("bf16Projector.withSiLU(true).usesSiLU() = false")
	}
	var qm qmvProjector
	if qm.usesSiLU() {
		t.Fatal("qmvProjector default usesSiLU() = true, want false (gemma GELU)")
	}
	if !qm.withSiLU(true).usesSiLU() {
		t.Fatal("qmvProjector.withSiLU(true).usesSiLU() = false")
	}
}
