// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

// The arch-registry benches baseline the reactive loader's dispatch cost (AX-11):
// LookupArch is the model_type → ArchSpec resolve the loader runs to route a checkpoint,
// and RegisterArch is the init-time write a model package pays once. The registry is the
// shared core.NewRegistry primitive; these measure its get/set + the interface unbox, not
// any load work. Synthetic — a spec registered under a bench-only model_type.

// BenchmarkLookupArch_Hit — the resolve path: one registry get + the ArchSpec type
// assertion. The cost the loader pays per checkpoint to find its architecture.
func BenchmarkLookupArch_Hit(b *testing.B) {
	RegisterArch(ArchSpec{ModelTypes: []string{"bench-arch"}})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok := LookupArch("bench-arch"); !ok {
			b.Fatal("registered arch not resolved")
		}
	}
}

// BenchmarkLookupArch_Miss — the unregistered lookup: a failed registry get, which must
// stay as cheap as a hit (the "no arch for this model_type" detection on the load path).
func BenchmarkLookupArch_Miss(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok := LookupArch("no-such-bench-arch"); ok {
			b.Fatal("unexpected hit on an unregistered model_type")
		}
	}
}
