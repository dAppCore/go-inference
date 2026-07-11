// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// The tensorF32 bench baselines the per-weight widening at load (AX-11): each bf16 checkpoint
// tensor is widened to a flat f32 slice (the precision the host scan runs in) once per weight
// at load — its allocation is the whole f32 copy, sized to the weight. Not per-token; this
// pins the per-weight load cost (the biggest single load allocation). Synthetic tensor — no
// checkpoint read. Dims: an in_proj-sized weight.

// BenchmarkTensorF32_BF16 — widening a projection-sized bf16 weight to f32: the per-element
// left-shift unpack into a fresh [len/2] f32 buffer.
func BenchmarkTensorF32_BF16(b *testing.B) {
	const rows, cols = 8192, 2048
	t := safetensors.Tensor{Shape: []int{rows, cols}, Dtype: "BF16", Data: make([]byte, rows*cols*2)}
	b.SetBytes(int64(len(t.Data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := tensorF32(t); err != nil {
			b.Fatal(err)
		}
	}
}
