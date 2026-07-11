// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "testing"

// The tensors benches baseline the GGUF dequantise-to-F16 load path (AX-11): a GGUF weight
// stored Q8_0/Q4_0 is widened to F16 at load for the native engine — the inverse of the
// quantise kernels, run once per quantised tensor at load. Each allocates the F16 result
// buffer; the per-block unpack is the cost. ggufFloat32ToFloat16 is the scalar half-pack the
// widening leans on. Valid quantised input is produced by the quantise kernels themselves.
// Pure Go, synthetic — no file.

// BenchmarkGGUFDequantizeQ8_0ToF16 — the Q8_0 → F16 widen: per-block scale × int8 into F16.
// The per-tensor load cost of an 8-bit GGUF weight.
func BenchmarkGGUFDequantizeQ8_0ToF16(b *testing.B) {
	const elems = 256 * 512
	raw, err := Quantize(QuantizeQ8_0, benchQuantF32(elems))
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ggufDequantizeQ8_0ToF16(raw, elems); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGGUFDequantizeQ4_0ToF16 — the Q4_0 → F16 widen: per-block scale × nibble into F16.
// The per-tensor load cost of a 4-bit GGUF weight.
func BenchmarkGGUFDequantizeQ4_0ToF16(b *testing.B) {
	const elems = 256 * 512
	raw, err := Quantize(QuantizeQ4_0, benchQuantF32(elems))
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ggufDequantizeQ4_0ToF16(raw, elems); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGGUFFloat32ToFloat16 — the scalar f32→f16 pack the widen leans on: bit-twiddling
// with subnormal/overflow handling, zero allocation.
func BenchmarkGGUFFloat32ToFloat16(b *testing.B) {
	vals := benchQuantF32(1024)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var acc uint16
		for _, v := range vals {
			acc ^= ggufFloat32ToFloat16(v)
		}
		_ = acc
	}
}
