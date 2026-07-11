// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

// The assemble benches baseline the generic weight assembler (AX-11): Assemble is the ONE
// arch.Layer-driven loop that maps a tensor set onto the neutral LoadedModel, running once
// per load. Its allocation is the LoadedModel + per-layer Linear structs + the
// Sprintf(LayerPrefix, i) name joins — the byte views are zero-copy, so this measures the
// assembly bookkeeping, not the weight bytes. StandardWeightNames is the canonical
// weight-name table an arch reads once; it is a pure struct literal, expected alloc-free.
// The fixtures (minimalDenseTensors / minimalDenseNames / minimalDenseArch) are shared with
// assemble_test.go — a hermetic single-layer dense set, no checkpoint read.

// BenchmarkAssemble — the per-load assembly of a single-layer dense model: the name joins,
// the map lookups, and the LoadedModel + LoadedLayer allocation. Small but real — the
// per-layer cost multiplies with layer count at load.
func BenchmarkAssemble(b *testing.B) {
	tensors := minimalDenseTensors("BF16")
	names := minimalDenseNames()
	arch := minimalDenseArch()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Assemble(tensors, arch, names); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkStandardWeightNames — the canonical HF weight-name table build: a struct literal
// of compile-time string constants, expected zero heap allocation. Pins that reading the
// weight-name convention at load is free.
func BenchmarkStandardWeightNames(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = StandardWeightNames()
	}
}
