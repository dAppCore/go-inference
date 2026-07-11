// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

// The foldNormBiasOne benches baseline the gemma "(1 + weight)" RMSNorm fold (AX-11): a
// once-per-norm-weight byte transform at load that lets the plain RMSNorm kernel reproduce
// gemma's (1+w)·rms(x). It runs per norm tensor across every layer at load, so its
// allocation shape — the single out[len(data)] byte slice — is the per-weight fold cost.
// bf16 does a round-to-nearest-even repack per element (the hot arithmetic); f32 is the
// plainer four-byte path. Realistic input: a hidden-sized norm weight.

func benchNormBytes(n int, bytesPer int) []byte { return make([]byte, n*bytesPer) }

// BenchmarkFoldNormBiasOne_BF16 — the shipping dtype: each 2-byte element is unpacked to
// f32, +1, and repacked with round-to-nearest-even. The per-element repack is the cost.
func BenchmarkFoldNormBiasOne_BF16(b *testing.B) {
	data := benchNormBytes(benchMatHidden, 2)
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := foldNormBiasOne(data, "BF16"); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkFoldNormBiasOne_F32 — the four-byte path: unpack, +1, repack, no rounding. The
// same single out allocation, so the bf16↔f32 gap is arithmetic-only, not allocation.
func BenchmarkFoldNormBiasOne_F32(b *testing.B) {
	data := benchNormBytes(benchMatHidden, 4)
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := foldNormBiasOne(data, "F32"); err != nil {
			b.Fatal(err)
		}
	}
}
