// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

// The quant-registry benches baseline the (backend,kind) dispatch cost (AX-11): the
// per-decode lookup the engine pays to resolve a weight's compute, plus the registration
// the backend pays once at init. They measure CONTRACT overhead — bkKey's "backend/kind"
// concat + the registry map — NOT matvec arithmetic (that lives in pkg/native/pkg/metal;
// fakeQuant.MatVec is a stub, so its bench is a dispatch floor, not a perf number).

// BenchmarkBackendQuant_Resolve — the hot path: resolve a registered (backend,kind) per
// decode. One bkKey concat (an alloc) + a map get. The cost the engine pays per weight.
func BenchmarkBackendQuant_Resolve(b *testing.B) {
	_ = RegisterBackendQuant("benchbackend", fakeQuant{kind: "affine", bits: 4, tag: 0x7})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok := BackendQuant("benchbackend", "affine"); !ok {
			b.Fatal("registered (backend,kind) not resolved")
		}
	}
}

// BenchmarkBackendQuant_Miss — the unregistered lookup (concat + a failed map get): the
// cost of detecting "no backend serves this format", which must stay as cheap as a hit.
func BenchmarkBackendQuant_Miss(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok := BackendQuant("nosuchbenchbackend", "affine"); ok {
			b.Fatal("unexpected hit on an unregistered (backend,kind)")
		}
	}
}

// BenchmarkRegisterBackendQuant — the init-time write (concat + map set). Run once per
// backend per kind in practice; benched for completeness of the registry surface.
func BenchmarkRegisterBackendQuant(b *testing.B) {
	q := fakeQuant{kind: "affine", bits: 4, tag: 0x9}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if r := RegisterBackendQuant("benchregister", q); !r.OK {
			b.Fatal("register failed")
		}
	}
}

// BenchmarkBkKey — the key composition in isolation: the "backend/kind" concat that is the
// one allocation on the resolve path. Pins the lower bound of BackendQuant's cost.
func BenchmarkBkKey(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bkKey("native", "affine")
	}
}
