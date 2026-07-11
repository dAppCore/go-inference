// SPDX-Licence-Identifier: EUPL-1.2

package composed

import "testing"

// The token-model benches baseline the bf16 boundary conversions (AX-11): the
// model.TokenModel seam is bf16 []byte but the hybrid runs f32, so every token crossing the
// seam (Embed in, Step/DecodeForward hidden out) converts. f32ToBF16Bytes does a
// round-to-nearest-even repack per element; bf16BytesToF32 the reverse widen. Each allocates
// one result buffer, so the conversion is a per-token allocation the decode loop pays twice
// (embed in, hidden out). Dims: a single [D] hidden (D=1024).

// BenchmarkF32ToBF16Bytes — the f32→bf16 pack a hidden/logit vector pays leaving the model:
// per-element round-to-nearest-even into a fresh [2·len] byte buffer.
func BenchmarkF32ToBF16Bytes(b *testing.B) {
	v := benchF32(benchD)
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
	data := f32ToBF16Bytes(benchF32(benchD))
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bf16BytesToF32(data)
	}
}
