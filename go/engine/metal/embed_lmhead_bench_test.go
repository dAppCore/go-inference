// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

func BenchmarkEmbedTokensBF16Batch16(b *testing.B) {
	const vocab, dModel = 128, 64
	table := toBF16Bytes(syntheticFloat32(vocab*dModel, 11))
	tokenIDs := []int32{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
	scale := float32(math.Sqrt(float64(dModel)))
	b.SetBytes(int64(len(tokenIDs) * dModel * bf16Size))
	for i := 0; i < b.N; i++ {
		if _, err := EmbedTokensBF16(table, tokenIDs, vocab, dModel, scale); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLMHeadBF16_64x128(b *testing.B) {
	requireNativeRuntime(b)

	const vocab, dModel = 128, 64
	hidden := toBF16Bytes(syntheticFloat32(dModel, 3))
	finalNorm := toBF16Bytes(syntheticFloat32(dModel, 5))
	head := toBF16Bytes(syntheticFloat32(vocab*dModel, 7))
	b.SetBytes(int64(len(hidden) + len(finalNorm) + len(head)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := LMHeadBF16(hidden, finalNorm, head, dModel, vocab, 1e-5, 0); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := LMHeadBF16(hidden, finalNorm, head, dModel, vocab, 1e-5, 0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLMHeadBF16Into64x128(b *testing.B) {
	requireNativeRuntime(b)

	const vocab, dModel = 128, 64
	hidden := toBF16Bytes(syntheticFloat32(dModel, 3))
	finalNorm := toBF16Bytes(syntheticFloat32(dModel, 5))
	head := toBF16Bytes(syntheticFloat32(vocab*dModel, 7))
	out := make([]byte, vocab*bf16Size)
	b.SetBytes(int64(len(hidden) + len(finalNorm) + len(head)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := LMHeadBF16Into(out, hidden, finalNorm, head, dModel, vocab, 1e-5, 0); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = LMHeadBF16Into(out, hidden, finalNorm, head, dModel, vocab, 1e-5, 0)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLMHeadQuant64x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab, groupSize, bits = 64, 128, 32, 4
	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	finalNorm := toBF16Bytes(syntheticFloat32(dModel, 7))
	qw := quantWeightFixture(b, vocab, dModel, groupSize, bits, 53)
	b.SetBytes(int64(len(hidden) + len(finalNorm) + len(qw.Packed) + len(qw.Scales) + len(qw.Biases)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := LMHeadQuant(hidden, finalNorm, qw.Packed, qw.Scales, qw.Biases, dModel, vocab, groupSize, bits, 1e-6, 0); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := LMHeadQuant(hidden, finalNorm, qw.Packed, qw.Scales, qw.Biases, dModel, vocab, groupSize, bits, 1e-6, 0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLMHeadQuantInto64x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab, groupSize, bits = 64, 128, 32, 4
	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	finalNorm := toBF16Bytes(syntheticFloat32(dModel, 7))
	qw := quantWeightFixture(b, vocab, dModel, groupSize, bits, 53)
	out := make([]byte, vocab*bf16Size)
	b.SetBytes(int64(len(hidden) + len(finalNorm) + len(qw.Packed) + len(qw.Scales) + len(qw.Biases)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := LMHeadQuantInto(out, hidden, finalNorm, qw.Packed, qw.Scales, qw.Biases, dModel, vocab, groupSize, bits, 1e-6, 0); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = LMHeadQuantInto(out, hidden, finalNorm, qw.Packed, qw.Scales, qw.Biases, dModel, vocab, groupSize, bits, 1e-6, 0)
		if err != nil {
			b.Fatal(err)
		}
	}
}
