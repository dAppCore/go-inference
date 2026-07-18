// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"bytes"
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// moe_batched_test.go gates sliceBatchedExpertQuant (loader.go) — the per-expert carve of the mlx
// switch_mlp batched MoE layout (one [numExperts, outDim, packed] tensor per projection), which is what
// makes the real mlx-community Qwen3.5-35B-A3B-4bit checkpoint loadable in the composed lane.

// TestSliceBatchedExpertQuant pins the round trip: concatenating N per-expert quant weights into one
// [N, outDim, packed] tensor (+ its batched scales/biases) and slicing expert e back out is BYTE-IDENTICAL
// to the original — the exact inverse of mlx-lm's fused-MoE batching. Every expert's codes, scales, biases
// and derived geometry (bits, group size, out/in dims) survive.
func TestSliceBatchedExpertQuant(t *testing.T) {
	const nE, FF, D, bits, gs = 4, 24, 64, 4, 8
	var packed, scales, biases []byte
	originals := make([]*model.QuantWeight, nE)
	for e := range nE {
		qw, _ := quantiseSynthetic(t, FF, D, bits, gs, e*3+1)
		originals[e] = qw
		packed = append(packed, qw.Packed...)
		scales = append(scales, qw.Scales...)
		biases = append(biases, qw.Biases...)
	}
	tensors := map[string]safetensors.Tensor{
		"sm.gate_proj.weight": {Dtype: "U32", Shape: []int{nE, FF, D * bits / 32}, Data: packed},
		"sm.gate_proj.scales": {Dtype: "BF16", Shape: []int{nE, FF, D / gs}, Data: scales},
		"sm.gate_proj.biases": {Dtype: "BF16", Shape: []int{nE, FF, D / gs}, Data: biases},
	}
	get := func(n string) (safetensors.Tensor, bool) { tt, ok := tensors[n]; return tt, ok }

	for e := range nE {
		got, err := sliceBatchedExpertQuant(get, "sm.gate_proj", e, FF, D)
		if err != nil {
			t.Fatalf("expert %d: %v", e, err)
		}
		o := originals[e]
		if !bytes.Equal(got.Packed, o.Packed) || !bytes.Equal(got.Scales, o.Scales) || !bytes.Equal(got.Biases, o.Biases) {
			t.Fatalf("expert %d: sliced bytes != original — the round trip is not byte-identical", e)
		}
		if got.OutDim != FF || got.InDim != D || got.Bits != bits || got.GroupSize != gs {
			t.Fatalf("expert %d: geometry (out %d, in %d, bits %d, gs %d), want (%d, %d, %d, %d)",
				e, got.OutDim, got.InDim, got.Bits, got.GroupSize, FF, D, bits, gs)
		}
		// slices must be OWNED copies, not aliases into the batched buffer.
		got.Packed[0] ^= 0xFF
		if bytes.Equal(got.Packed, packed[e*len(o.Packed):(e+1)*len(o.Packed)]) {
			t.Fatalf("expert %d: sliced Packed aliases the batched tensor (not copied)", e)
		}
	}
}

// TestSliceBatchedExpertQuant_Bad pins the guards: a missing sibling or a non-3-D tensor errors rather than
// slicing garbage.
func TestSliceBatchedExpertQuant_Bad(t *testing.T) {
	get := func(map[string]safetensors.Tensor) func(string) (safetensors.Tensor, bool) {
		return func(string) (safetensors.Tensor, bool) { return safetensors.Tensor{}, false }
	}(nil)
	if _, err := sliceBatchedExpertQuant(get, "sm.gate_proj", 0, 24, 64); err == nil {
		t.Fatal("missing weight/scales/biases must error")
	}
	twoD := map[string]safetensors.Tensor{
		"sm.gate_proj.weight": {Dtype: "U32", Shape: []int{24, 32}, Data: make([]byte, 24*32*4)},
		"sm.gate_proj.scales": {Dtype: "BF16", Shape: []int{24, 8}, Data: make([]byte, 24*8*2)},
		"sm.gate_proj.biases": {Dtype: "BF16", Shape: []int{24, 8}, Data: make([]byte, 24*8*2)},
	}
	get2 := func(n string) (safetensors.Tensor, bool) { tt, ok := twoD[n]; return tt, ok }
	if _, err := sliceBatchedExpertQuant(get2, "sm.gate_proj", 0, 24, 64); err == nil {
		t.Fatal("a 2-D (non-batched) tensor must error")
	}
}
