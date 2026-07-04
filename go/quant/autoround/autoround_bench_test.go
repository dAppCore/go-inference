// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import "testing"

// benchSyntheticWeights builds a deterministic synthetic weight tensor sized
// like a small projection row block so per-group quant/pack/export allocations
// are exercised across many groups (AX-11 load-path bench, no model load).
func benchSyntheticWeights(elements int) []float32 {
	weights := make([]float32, elements)
	for i := range weights {
		// Spread values across a representative range without RNG so the
		// quantised output is byte-stable run to run.
		weights[i] = float32((i%257)-128) * 0.0123
	}
	return weights
}

func benchQuantized(b *testing.B, elements, groupSize int) QuantizedWeights {
	b.Helper()
	quantized, err := QuantizeWeights(benchSyntheticWeights(elements), QuantizeConfig{
		Scheme:    SchemeW4A16,
		GroupSize: groupSize,
		Iters:     0,
	})
	if err != nil {
		b.Fatalf("QuantizeWeights() error = %v", err)
	}
	return quantized
}

func BenchmarkQuantizeWeights(b *testing.B) {
	weights := benchSyntheticWeights(4096)
	cfg := QuantizeConfig{Scheme: SchemeW4A16, GroupSize: 128, Iters: 0}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := QuantizeWeights(weights, cfg); err != nil {
			b.Fatalf("QuantizeWeights() error = %v", err)
		}
	}
}

// BenchmarkQuantizeWeightsSignRound exercises the SignRound rounding-search
// branch (Iters > 0 with per-weight gradients) that the Iters:0 bench skips, so
// the rounding-adjust path is measured for allocations rather than reasoned
// about. Gradients are built once outside the timed loop and are deterministic
// so the quantised output stays byte-stable run to run.
func BenchmarkQuantizeWeightsSignRound(b *testing.B) {
	weights := benchSyntheticWeights(4096)
	gradients := make([]float32, len(weights))
	for i := range gradients {
		gradients[i] = float32((i%97)-48) * 0.01
	}
	cfg := QuantizeConfig{Scheme: SchemeW4A16, GroupSize: 128, Iters: 200, Gradients: gradients}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := QuantizeWeights(weights, cfg); err != nil {
			b.Fatalf("QuantizeWeights() error = %v", err)
		}
	}
}

func BenchmarkPackQuantizedWeights(b *testing.B) {
	quantized := benchQuantized(b, 4096, 128)
	shape := []int32{32, 128}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PackQuantizedWeights(quantized, shape); err != nil {
			b.Fatalf("PackQuantizedWeights() error = %v", err)
		}
	}
}

func BenchmarkDequantizePackedWeights(b *testing.B) {
	quantized := benchQuantized(b, 4096, 128)
	packed, err := PackQuantizedWeights(quantized, []int32{32, 128})
	if err != nil {
		b.Fatalf("PackQuantizedWeights() error = %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DequantizePackedWeights(packed); err != nil {
			b.Fatalf("DequantizePackedWeights() error = %v", err)
		}
	}
}

func benchProjection(b *testing.B, elements, groupSize int) PackedProjection {
	b.Helper()
	quantized := benchQuantized(b, elements, groupSize)
	shape := []int32{int32(elements / groupSize), int32(groupSize)}
	packed, err := PackQuantizedWeights(quantized, shape)
	if err != nil {
		b.Fatalf("PackQuantizedWeights() error = %v", err)
	}
	groups := (elements + groupSize - 1) / groupSize
	packedBytes := (elements*packed.Bits + 7) / 8
	return PackedProjection{
		Tensor: PackTensor{
			Name:        "model.layers.0.self_attn.q_proj.weight",
			Packed:      "model.layers.0.self_attn.q_proj.weight.packed",
			Scales:      "model.layers.0.self_attn.q_proj.weight.scales",
			ZeroPoints:  "model.layers.0.self_attn.q_proj.weight.zeros",
			Shape:       shape,
			Bits:        packed.Bits,
			GroupSize:   groupSize,
			Symmetric:   packed.Symmetric,
			PackedBytes: packedBytes,
			Groups:      groups,
			QMin:        packed.QMin,
			QMax:        packed.QMax,
		},
		Weights: packed,
	}
}

// BenchmarkPackedProjectionTensors drives the in-memory safetensors-tensor
// build (no file IO) so the per-projection packed-buffer handling shows up
// cleanly in -alloc_space.
func BenchmarkPackedProjectionTensors(b *testing.B) {
	projection := benchProjection(b, 4096, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := packedProjectionSafetensorsTensors(projection); err != nil {
			b.Fatalf("packedProjectionSafetensorsTensors() error = %v", err)
		}
	}
}
