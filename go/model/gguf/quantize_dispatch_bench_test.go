// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "testing"

// The quantize-dispatch benches baseline the GGUF quantisation kernels (AX-11) through the ONE
// public entry (Quantize/AppendQuantize live in quantize_dispatch.go) every engine streams
// weights through — one bench per format, so each of the nine kernels (the _0 family over
// 32-value blocks, the K family over 256-value super-blocks) is measured at a realistic
// per-tensor size. The kernel is the whole cost; the packed output is the single (pre-sized)
// allocation. This covers quantize_kernels.go's kernels via the dispatch entry (no separate
// direct-kernel benches — that would double-bench the same symbols). Pure Go, synthetic — no file.

const benchQuantElems = 256 * 512 // multiple of 256 (K super-block) and 32 (_0 block)

func benchQuantF32(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32((i*131)%4096-2048) * 0.001
	}
	return s
}

func benchQuantize(b *testing.B, format QuantizeFormat) {
	b.Helper()
	vals := benchQuantF32(benchQuantElems)
	b.SetBytes(int64(len(vals) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Quantize(format, vals); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkQuantize_Q8_0 — 8-bit, 32-value blocks: the highest-fidelity _0 format, one scale
// per block. The common "keep it simple" export.
func BenchmarkQuantize_Q8_0(b *testing.B) { benchQuantize(b, QuantizeQ8_0) }

// BenchmarkQuantize_Q4_0 — 4-bit, 32-value blocks: the smallest _0 format, nibble-packed.
func BenchmarkQuantize_Q4_0(b *testing.B) { benchQuantize(b, QuantizeQ4_0) }

// BenchmarkQuantize_Q5_0 — 5-bit, 32-value blocks: nibbles + a high-bit plane.
func BenchmarkQuantize_Q5_0(b *testing.B) { benchQuantize(b, QuantizeQ5_0) }

// BenchmarkQuantize_Q4_K — 4-bit K super-block (256 values, 8 sub-blocks with packed scales):
// the workhorse llama.cpp quant, the heaviest of the common formats.
func BenchmarkQuantize_Q4_K(b *testing.B) { benchQuantize(b, QuantizeQ4_K) }

// BenchmarkQuantize_Q5_K — 5-bit K super-block.
func BenchmarkQuantize_Q5_K(b *testing.B) { benchQuantize(b, QuantizeQ5_K) }

// BenchmarkQuantize_Q6_K — 6-bit K super-block: near-lossless, the common "high quality" quant.
func BenchmarkQuantize_Q6_K(b *testing.B) { benchQuantize(b, QuantizeQ6_K) }

// BenchmarkQuantize_Q8_K — 8-bit K super-block (used as the intermediate for K-quant matmul).
func BenchmarkQuantize_Q8_K(b *testing.B) { benchQuantize(b, QuantizeQ8_K) }

// BenchmarkQuantize_Q3_K — 3-bit K super-block: the low-bit-rate quant, the heaviest packing.
func BenchmarkQuantize_Q3_K(b *testing.B) { benchQuantize(b, QuantizeQ3_K) }

// BenchmarkQuantize_Q2_K — 2-bit K super-block: the smallest, most aggressive quant.
func BenchmarkQuantize_Q2_K(b *testing.B) { benchQuantize(b, QuantizeQ2_K) }
