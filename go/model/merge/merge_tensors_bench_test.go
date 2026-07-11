// SPDX-Licence-Identifier: EUPL-1.2

package merge

import "testing"

// The merge-tensors benches baseline the per-element model-merge kernels (AX-11), run once per
// tensor at merge time: linearMerge is the weighted sum across sources (the primary merge
// path), slerpMerge the spherical interpolation of two tensors, normalizedWeights the
// per-source coefficient normalise. linearMerge/slerpMerge allocate the single out slice and
// run an O(sources·elements) f32/f64 loop — the merge's whole compute. Sized to a realistic
// weight tensor. Pure Go, synthetic values — no file.

const benchMergeElems = 1 << 20 // 1M elements — a realistic weight tensor

func benchMergeF32(n int, seed int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32((i*seed)%4096-2048) * 0.001
	}
	return s
}

// BenchmarkLinearMerge — the weighted sum across three sources: the += accumulation over 1M
// elements. The out slice is the single allocation; the loop is the merge cost.
func BenchmarkLinearMerge(b *testing.B) {
	values := [][]float32{benchMergeF32(benchMergeElems, 131), benchMergeF32(benchMergeElems, 97), benchMergeF32(benchMergeElems, 59)}
	weights := []float64{0.5, 0.3, 0.2}
	b.SetBytes(int64(benchMergeElems * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := linearMerge(values, weights); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSLERPMerge — the spherical interpolation of two tensors: the dot/norm scan (f64)
// then a linear blend at the SLERP scales. The norm pass + the blend are the cost.
func BenchmarkSLERPMerge(b *testing.B) {
	values := [][]float32{benchMergeF32(benchMergeElems, 131), benchMergeF32(benchMergeElems, 97)}
	b.SetBytes(int64(benchMergeElems * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := slerpMerge(values, 0.5); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkNormalizedWeights — the per-source coefficient normalise (sum→1): a small pass over
// the sources, run once per merge. The weights slice is the allocation.
func BenchmarkNormalizedWeights(b *testing.B) {
	sources := []Source{{Weight: 0.5}, {Weight: 0.3}, {Weight: 0.2}, {Weight: 0.1}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := normalizedWeights(sources); err != nil {
			b.Fatal(err)
		}
	}
}
