// SPDX-Licence-Identifier: EUPL-1.2

package mlxaffine_test

import (
	"testing"

	"dappco.re/go/inference/model/quant/mlxaffine"
)

// mlxaffine_bench_test.go pins the cost of the two whole-matrix walks the
// quantiser runs on EVERY eligible weight tensor of a model at convert time:
// QuantizeTensor visits every element once (min/max per group, then a code per
// element) and DequantizeTensor unpacks and reconstructs every element. A 12B
// model has hundreds of these tensors, so the per-tensor allocation shape is the
// convert-time memory story — the benches report B/op + allocs/op so a regression
// in the fixed output-buffer allocations is caught.

// benchOut/benchIn are a representative attention/MLP projection shape: a
// [1024×1024] matrix at 4-bit, group size 64 — the MLX default the engine loads.
const (
	benchOut  = 1024
	benchIn   = 1024
	benchBits = 4
	benchGS   = 64
)

// BenchmarkQuantizeTensor measures the forward group-affine walk at the default
// [1024×1024] 4-bit gs64 shape. Output is three fixed buffers (packed + scales +
// biases), so allocs/op should be a small constant independent of N.
func BenchmarkQuantizeTensor(b *testing.B) {
	w := synthWeight(benchOut, benchIn)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, _, err := mlxaffine.QuantizeTensor(w, benchOut, benchIn, benchBits, benchGS); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDequantizeTensor measures the inverse walk — unpack every code and
// reconstruct scale·q+bias — on the same shape. The single output float32 slice
// is the only allocation the reconstruction should make.
func BenchmarkDequantizeTensor(b *testing.B) {
	w := synthWeight(benchOut, benchIn)
	packed, scales, biases, err := mlxaffine.QuantizeTensor(w, benchOut, benchIn, benchBits, benchGS)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := mlxaffine.DequantizeTensor(packed, scales, biases, benchOut, benchIn, benchBits, benchGS); err != nil {
			b.Fatal(err)
		}
	}
}
