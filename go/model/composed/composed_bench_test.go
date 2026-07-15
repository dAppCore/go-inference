// SPDX-Licence-Identifier: EUPL-1.2

package composed

import "testing"

// The composed core benches baseline the hybrid stack's per-token CPU primitives (AX-11):
// the SwiGLU MLP (a layer's dense FFN slot), the plain RMSNorm applied twice per layer, and
// the final argmax over the vocab logits. All run per decode token, so their allocation
// shape is the per-token cost the decode loop pays outside the mixer. Dims are a small-model
// layer (D=1024, FF=4096) at L=1 (single-token decode); the matmuls use the package matNT
// f64-accumulation host reference. Pure Go, no model load, no device GEMM.

const (
	benchD  = 1024   // hidden size
	benchFF = 4096   // dense FFN intermediate
	benchV  = 151936 // qwen-family vocab (for the argmax scan)
)

// benchF32 fills n floats with a deterministic spread — realistic magnitudes without a RNG
// dependency, so every run measures the same work.
func benchF32(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32((i*131)%4096-2048) * 0.001
	}
	return s
}

// BenchmarkMLP_Forward — one token through the SwiGLU MLP: three matNT projections (gate,
// up, down) plus the SiLU-gate elementwise. The g/u/h scratch + the down result are the
// per-token FFN allocation a dense layer pays.
func BenchmarkMLP_Forward(b *testing.B) {
	mlp := &MLP{Gate: benchF32(benchFF * benchD), Up: benchF32(benchFF * benchD), Down: benchF32(benchD * benchFF), FF: benchFF}
	x := benchF32(benchD)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = mlp.forward(x, 1, benchD)
	}
}

// BenchmarkRMSNormRowsPlain — the plain RMSNorm applied to a single [D] row (run twice per
// layer, pre-mixer and pre-MLP). One out allocation of [D]; the scan + rescale is the cost.
func BenchmarkRMSNormRowsPlain(b *testing.B) {
	x := benchF32(benchD)
	w := benchF32(benchD)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rmsNormRowsPlain(x, w, 1, benchD, 1e-6)
	}
}

// BenchmarkArgmaxF32 — the greedy token pick over a vocab-sized logit vector: a single scan,
// expected zero allocs. The deterministic close of a decode step.
func BenchmarkArgmaxF32(b *testing.B) {
	logits := benchF32(benchV)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = argmaxF32(logits)
	}
}
