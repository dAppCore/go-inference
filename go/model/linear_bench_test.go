// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// The LoadLinear benches baseline the per-weight load path (AX-11): the map lookups +
// the per-weight quant decision (.scales present? derive geometry from shapes) run once
// per weight at model construction. Synthetic — a tensor SET in memory, no checkpoint
// read. The byte slices VIEW the tensor data (zero-copy), so allocation here is the
// Linear struct + the lookups, not the weight bytes.

func benchTensor(shape ...int) safetensors.Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return safetensors.Tensor{Shape: shape, Data: make([]byte, n)}
}

// BenchmarkLoadLinear_Dense — the bf16 path: one .weight lookup + two absent-tensor
// lookups (.bias, .scales), no geometry math. The cheap branch.
func BenchmarkLoadLinear_Dense(b *testing.B) {
	const out, in = 4096, 4096
	t := map[string]safetensors.Tensor{"w.weight": benchTensor(out, in)}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if l := LoadLinear(t, "w", in, "affine"); l == nil {
			b.Fatal("nil Linear")
		}
	}
}

// BenchmarkLoadLinear_Quant — the affine path: .weight + .scales + .biases lookups plus
// affineGeometry deriving group size + bit-width from the shapes. The full decision.
func BenchmarkLoadLinear_Quant(b *testing.B) {
	const out, in = 4096, 4096
	t := map[string]safetensors.Tensor{
		"w.weight": benchTensor(out, in*4/32), // 4-bit packed
		"w.scales": benchTensor(out, in/32),   // group 32
		"w.biases": benchTensor(out, in/32),
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l := LoadLinear(t, "w", in, "affine")
		if l == nil || !l.Quantised() {
			b.Fatal("expected a quantised Linear")
		}
	}
}
