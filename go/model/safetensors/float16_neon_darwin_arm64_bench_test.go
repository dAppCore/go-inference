// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package safetensors

import "testing"

// The float16SliceToFloat32 bench baselines the NEON FCVTL half→single conversion (AX-11) —
// the darwin/arm64 hardware path DecodeFloatData drives for F16 weights at load. It is a
// vectorised fill into a caller-provided dst, so it allocates nothing; the bench pins the
// SIMD throughput of a per-tensor F16 upcast. This is CPU SIMD (the ARMv8 FCVTL instruction),
// not GPU, and needs no model. Sized to a realistic 1M-element per-tensor chunk.

func benchU16(n int) []uint16 {
	s := make([]uint16, n)
	for i := range s {
		s[i] = uint16(i * 137) // full pattern spread
	}
	return s
}

// BenchmarkFloat16SliceToFloat32 — the NEON FCVTL loop over 1M F16 values into a caller dst:
// zero allocation, the hardware-accelerated upcast the F16 load path uses on this platform.
func BenchmarkFloat16SliceToFloat32(b *testing.B) {
	const n = 1 << 20
	src := benchU16(n)
	dst := make([]float32, n)
	b.SetBytes(int64(n * 2))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		float16SliceToFloat32(src, dst, n)
	}
}
