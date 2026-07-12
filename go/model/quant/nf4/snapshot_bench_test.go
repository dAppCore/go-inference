// SPDX-Licence-Identifier: EUPL-1.2
package nf4

import (
	"context"
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
	"testing"
)

func BenchmarkConvertSnapshot(b *testing.B) {
	src, out := b.TempDir(), core.PathJoin(b.TempDir(), "out")
	blob, err := safetensors.Encode(map[string]safetensors.Tensor{"layer.weight": {Dtype: "F32", Shape: []int{64, 64}, Data: safetensors.EncodeFloat32(make([]float32, 4096))}})
	if err != nil {
		b.Fatal(err)
	}
	if r := core.WriteFile(core.PathJoin(src, "model.safetensors"), blob, 0o644); !r.OK {
		b.Fatal(r.Err())
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := ConvertSnapshot(context.Background(), src, out, nil); err != nil {
			b.Fatal(err)
		}
	}
}
