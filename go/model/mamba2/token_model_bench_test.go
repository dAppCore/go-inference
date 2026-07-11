// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import "testing"

// The token-model benches baseline the bf16 boundary conversions (AX-11): the
// model.TokenModel seam is bf16 []byte but Mamba runs f32, so every token crossing the seam
// (Embed in, Step/DecodeForward hidden out) converts. f32ToBF16Bytes packs with
// round-to-nearest-even; bf16BytesToF32 widens back. Each allocates one result buffer, a
// per-token allocation the recurrent decode pays twice per step. Dims: a single [D] hidden
// (D=2048, a small-model d_model).

const benchMambaD = 2048

func benchMambaF32(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32((i*131)%4096-2048) * 0.001
	}
	return s
}

// BenchmarkF32ToBF16Bytes — the f32→bf16 pack a hidden pays leaving the model: per-element
// round-to-nearest-even into a fresh [2·len] byte buffer.
func BenchmarkF32ToBF16Bytes(b *testing.B) {
	v := benchMambaF32(benchMambaD)
	b.SetBytes(int64(len(v) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = f32ToBF16Bytes(v)
	}
}

// BenchmarkBF16BytesToF32 — the bf16→f32 widen an embedding pays entering the model: a
// left-shift unpack into a fresh [len/2] f32 buffer.
func BenchmarkBF16BytesToF32(b *testing.B) {
	data := f32ToBF16Bytes(benchMambaF32(benchMambaD))
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bf16BytesToF32(data)
	}
}
