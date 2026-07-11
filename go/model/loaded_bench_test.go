// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

// The loaded-model benches baseline the once-per-load required-weight check (AX-11):
// ValidateRequired walks every layer confirming the always-present weights are there, so a
// malformed checkpoint fails cleanly instead of nil-derefing deep in decode. It is
// pointer-nil-check bound with no allocation on the success path; the bench pins that a
// clean N-layer validation allocates nothing. Realistic input: a 48-layer dense model.

func benchLoadedModel(n int) (*LoadedModel, Arch) {
	arch := Arch{Layer: make([]LayerSpec, n)}
	m := &LoadedModel{Embed: &Linear{OutDim: 4}, FinalNorm: []byte{1, 2}, Layers: make([]LoadedLayer, n)}
	for i := range m.Layers {
		arch.Layer[i] = LayerSpec{CacheIndex: i}
		m.Layers[i] = LoadedLayer{
			AttnNorm: []byte{1, 2}, Q: &Linear{}, K: &Linear{}, O: &Linear{},
			MLPNorm: []byte{1, 2}, Gate: &Linear{}, Up: &Linear{}, Down: &Linear{},
		}
	}
	return m, arch
}

// BenchmarkValidateRequired — the full per-load walk over 48 dense layers, all weights
// present (the pass path runs every check). Expected zero allocs — the validation is
// pure nil-checks over the already-built model.
func BenchmarkValidateRequired(b *testing.B) {
	m, arch := benchLoadedModel(48)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := m.ValidateRequired(arch); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkLoadedModel_Tied — the tied-LM-head check a backend runs once to decide whether
// the output projection reuses the token embedding: a single nil compare, no allocation.
func BenchmarkLoadedModel_Tied(b *testing.B) {
	m := &LoadedModel{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.Tied()
	}
}
