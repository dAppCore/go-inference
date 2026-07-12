// SPDX-Licence-Identifier: EUPL-1.2

package awq

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

var benchmarkResult *Result

func BenchmarkConvertSnapshot(b *testing.B) {
	root := b.TempDir()
	src := core.PathJoin(root, "src")
	if result := core.MkdirAll(src, 0o755); !result.OK {
		b.Fatal(result.Err())
	}
	values := make([]float32, 32*128)
	blob, err := safetensors.Encode(map[string]safetensors.Tensor{
		"model.layers.0.mlp.up_proj.weight": {Dtype: "F32", Shape: []int{32, 128}, Data: safetensors.EncodeFloat32(values)},
	})
	if err != nil {
		b.Fatal(err)
	}
	if result := core.WriteFile(core.PathJoin(src, "model.safetensors"), blob, 0o644); !result.OK {
		b.Fatal(result.Err())
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := range b.N {
		benchmarkResult, err = ConvertSnapshot(context.Background(), src, core.PathJoin(root, core.Sprintf("out-%d", i)), Options{Bits: 4, GroupSize: 128, ZeroPoint: true}, nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}
