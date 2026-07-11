// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// The compare benches baseline the per-tensor delta computation (AX-11), run once per tensor
// when comparing a base vs a fine-tuned pack: compareTensorEntries decodes both tensors and
// runs the stats scan (mean-abs / RMS / max-abs / L2 / cosine) in one f64 pass;
// compareCosine is the cosine close from the accumulated dot + norms. The decode buffers +
// the single scan are the cost. Sized to a realistic weight tensor. Pure Go, synthetic F32
// bytes — no file.

func benchCompareEntry(elems, seed int) tensorEntry {
	vals := make([]float32, elems)
	for i := range vals {
		vals[i] = float32((i*seed)%4096-2048) * 0.001
	}
	return tensorEntry{DType: "F32", Shape: []int{1024, elems / 1024}, Raw: safetensors.EncodeFloat32(vals)}
}

// BenchmarkCompareTensorEntries — decode base + tuned (1M elements each) then the single-pass
// delta scan: the two decode allocations + the O(elements) f64 stats loop. The per-tensor
// cost of a pack comparison.
func BenchmarkCompareTensorEntries(b *testing.B) {
	const elems = 1 << 20
	base := benchCompareEntry(elems, 131)
	tuned := benchCompareEntry(elems, 137)
	b.SetBytes(int64(elems * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := compareTensorEntries("model.layer.0.weight", base, tuned); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkCompareCosine — the cosine close from accumulated dot + norms: a couple of sqrts
// and a clamp, no allocation. The per-tensor scalar finish.
func BenchmarkCompareCosine(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = compareCosine(0.87, 1.2, 1.3)
	}
}
