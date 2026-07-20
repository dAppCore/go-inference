// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

func TestQWMapsModelLinear(t *testing.T) {
	if got := qw(nil); got.Packed != nil || got.GroupSize != 0 || got.Bits != 0 {
		t.Fatalf("qw(nil) = %+v, want zero QuantWeight", got)
	}
	lin := &model.Linear{
		Weight:    []byte{1, 2, 3},
		Scales:    []byte{4, 5},
		Biases:    []byte{6, 7},
		GroupSize: 64,
		Bits:      4,
	}
	got := qw(lin)
	if string(got.Packed) != string(lin.Weight) || string(got.Scales) != string(lin.Scales) || string(got.Biases) != string(lin.Biases) {
		t.Fatalf("qw did not preserve linear byte slices: got %+v", got)
	}
	if got.GroupSize != 64 || got.Bits != 4 {
		t.Fatalf("qw geometry = gs%d bits%d, want gs64 bits4", got.GroupSize, got.Bits)
	}
}

func TestLoadedToQuantRejectsNilModel(t *testing.T) {
	if _, err := loadedToQuant(nil, 64, 4); err == nil {
		t.Fatal("expected loadedToQuant to reject a nil model")
	}
}

// TestMoeToQuant_SharedExpertGeometryIsPassThrough_Good is the #57 engine-side verification: load_shared.go
// must NOT independently re-derive or reassume the shared expert's affine geometry from arch.ExpertFF —
// moeToQuant's qw() maps each Linear's OWN GroupSize/Bits verbatim (read straight off the model.Linear
// model.Assemble already resolved), so the #57 fix — which corrects SharedDown's InDim upstream in
// model.Assemble, not here — is preserved end-to-end with NO change needed in this file. SharedDown and
// ExpDown are given DELIBERATELY DIFFERENT geometry (mirroring what a genuinely mismatched-width
// checkpoint like Qwen1.5-MoE-A2.7B produces post-fix: GroupSize 8/Bits 4 for the 5632-wide shared
// expert vs GroupSize 32/Bits 8 for the 1408-wide routed experts) to prove the two are never conflated
// or overwritten by a model-wide value.
func TestMoeToQuant_SharedExpertGeometryIsPassThrough_Good(t *testing.T) {
	e := &model.LoadedMoE{
		Router:  &model.Linear{Weight: make([]byte, 8)},
		ExpGate: &model.Linear{Weight: make([]byte, 8), GroupSize: 32, Bits: 8},
		ExpUp:   &model.Linear{Weight: make([]byte, 8), GroupSize: 32, Bits: 8},
		ExpDown: &model.Linear{Weight: make([]byte, 8), GroupSize: 32, Bits: 8},
		// SharedGate/Up/Down carry the geometry #57's upstream fix derives for a genuinely-distinct-width
		// shared expert — deliberately different from ExpGate/Up/Down above.
		SharedGate: &model.Linear{Weight: make([]byte, 8), GroupSize: 8, Bits: 4},
		SharedUp:   &model.Linear{Weight: make([]byte, 8), GroupSize: 8, Bits: 4},
		SharedDown: &model.Linear{Weight: make([]byte, 8), GroupSize: 8, Bits: 4},
	}
	arch := model.Arch{Experts: 8, TopK: 2, ExpertFF: 1408, SharedExpertFF: 5632, Hidden: 2048}
	q := moeToQuant(e, arch)

	if q.SharedDown.GroupSize != 8 || q.SharedDown.Bits != 4 {
		t.Fatalf("SharedDown geometry = GroupSize %d Bits %d, want GroupSize 8 Bits 4 (the Linear's OWN geometry, not re-derived from arch.ExpertFF)", q.SharedDown.GroupSize, q.SharedDown.Bits)
	}
	if q.SharedGate.GroupSize != 8 || q.SharedGate.Bits != 4 || q.SharedUp.GroupSize != 8 || q.SharedUp.Bits != 4 {
		t.Fatalf("SharedGate/Up geometry = %+v/%+v, want GroupSize 8 Bits 4 on both", q.SharedGate, q.SharedUp)
	}
	if q.ExpDown.GroupSize != 32 || q.ExpDown.Bits != 8 {
		t.Fatalf("ExpDown geometry = GroupSize %d Bits %d, want GroupSize 32 Bits 8 — moeToQuant must not conflate the routed and shared geometries", q.ExpDown.GroupSize, q.ExpDown.Bits)
	}
	// ExpertDFF is model-wide and carries ONLY the routed width (arch.ExpertFF) — MoEQuantLayerWeights has
	// no shared-width counterpart field today (the #57 follow-up load_shared.go's doc now names: the
	// non-fused decode dispatch in arch_qwen_moe.go still sizes its shared-expert matvec off ExpertDFF).
	// Pinning this here (rather than assuming a field that doesn't exist) keeps this test honest about
	// what #57 did and did not reach.
	if q.ExpertDFF != 1408 {
		t.Fatalf("ExpertDFF = %d, want 1408 (arch.ExpertFF — the routed width)", q.ExpertDFF)
	}
}
