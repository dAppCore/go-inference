// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"bytes"
	"testing"

	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
)

// TestLoadLinear_QuantAgnostic is the R2 proof: one load path, the format decided per weight
// by .scales, the affine width read from the tensor shapes — so bf16 / 4-bit / 8-bit (and a
// weight one quant leaves bf16 while another quantises) all load with no per-weight branch.
func TestLoadLinear_QuantAgnostic(t *testing.T) {
	const out, in = 4, 64
	mk := func(shape ...int) safetensors.Tensor {
		n := 1
		for _, d := range shape {
			n *= d
		}
		return safetensors.Tensor{Shape: shape, Data: make([]byte, n)} // bytes irrelevant to geometry
	}
	cases := []struct {
		name             string
		t                map[string]safetensors.Tensor
		wantQuant        bool
		wantGS, wantBits int
	}{
		{
			name: "dense bf16 (no .scales)",
			t:    map[string]safetensors.Tensor{"w.weight": mk(out, in)},
		},
		{
			name:      "4-bit affine, group 32",
			t:         map[string]safetensors.Tensor{"w.weight": mk(out, in*4/32), "w.scales": mk(out, in/32), "w.biases": mk(out, in/32)},
			wantQuant: true, wantGS: 32, wantBits: 4,
		},
		{
			name:      "8-bit affine, group 64",
			t:         map[string]safetensors.Tensor{"w.weight": mk(out, in*8/32), "w.scales": mk(out, in/64), "w.biases": mk(out, in/64)},
			wantQuant: true, wantGS: 64, wantBits: 8,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			l := LoadLinear(c.t, "w", in, "affine")
			if l == nil {
				t.Fatal("LoadLinear returned nil for a present weight")
			}
			if l.OutDim != out {
				t.Fatalf("OutDim=%d derived from shape, want %d", l.OutDim, out)
			}
			if l.Quantised() != c.wantQuant {
				t.Fatalf("Quantised()=%v want %v", l.Quantised(), c.wantQuant)
			}
			if c.wantQuant && (l.GroupSize != c.wantGS || l.Bits != c.wantBits) {
				t.Fatalf("geometry gs=%d bits=%d, want gs=%d bits=%d", l.GroupSize, l.Bits, c.wantGS, c.wantBits)
			}
			if !c.wantQuant && l.Kind != "" {
				t.Fatalf("dense weight got Kind=%q, want empty", l.Kind)
			}
		})
	}
}

// TestLoadLinear_AbsentReturnsNil — an optional weight that isn't in the checkpoint loads as
// nil (the caller treats nil as "feature absent"), never a zero-value mistaken for present.
func TestLoadLinear_AbsentReturnsNil(t *testing.T) {
	if l := LoadLinear(map[string]safetensors.Tensor{}, "missing", 64, "affine"); l != nil {
		t.Fatalf("absent weight should return nil, got %+v", l)
	}
}

// TestLoadLinear_AdditiveBias — a present ".bias" tensor is carried as the optional additive
// bias (orthogonal to quant: a dense or a quantised weight may both have one), and its absence
// leaves Bias nil. The .bias load branch the geometry cases never touched.
func TestLoadLinear_AdditiveBias(t *testing.T) {
	const out, in = 4, 64
	mk := func(shape ...int) safetensors.Tensor {
		n := 1
		for _, d := range shape {
			n *= d
		}
		return safetensors.Tensor{Shape: shape, Data: make([]byte, n)}
	}
	withBias := map[string]safetensors.Tensor{"w.weight": mk(out, in), "w.bias": mk(out)}
	l := LoadLinear(withBias, "w", in, "affine")
	if l == nil {
		t.Fatal("LoadLinear returned nil for a present weight")
	}
	if l.Bias == nil {
		t.Fatal("present .bias should be carried, got nil")
	}
	if len(l.Bias) != out {
		t.Fatalf("Bias len = %d, want %d (the .bias tensor data)", len(l.Bias), out)
	}
	if l.Quantised() {
		t.Fatal("a bias does not make a weight quantised (no .scales)")
	}
	// the same weight without .bias leaves Bias nil — the bias is genuinely optional.
	if nb := LoadLinear(map[string]safetensors.Tensor{"w.weight": mk(out, in)}, "w", in, "affine"); nb.Bias != nil {
		t.Fatalf("absent .bias should leave Bias nil, got %d bytes", len(nb.Bias))
	}
}

// TestLoadLinear_B1RepacksToB2 — a 1-bit affine weight (Bonsai's width: no b_1 device kernel
// ships) loads as its EXACT b2 widening: Bits reports 2, the packed codes equal
// mlxaffine.RepackB1ToB2's output, the repacked weight is written back into the tensor map
// (Data + packed shape) so a tied second read loads the b2 form instead of repacking again,
// and scales/biases keep the ORIGINAL tensor views (invariant under the widening — they stay
// on the zero-copy path).
func TestLoadLinear_B1RepacksToB2(t *testing.T) {
	const out, in, gs = 4, 64, 32
	packed := make([]byte, out*(in/32)*4)
	for i := range packed {
		packed[i] = byte(i*37 + 11)
	}
	scales := make([]byte, out*(in/gs)*2)
	biases := make([]byte, out*(in/gs)*2)
	for i := range scales {
		scales[i], biases[i] = byte(i+1), byte(i+2)
	}
	tensors := map[string]safetensors.Tensor{
		"w.weight": {Dtype: "U32", Shape: []int{out, in / 32}, Data: packed},
		"w.scales": {Dtype: "BF16", Shape: []int{out, in / gs}, Data: scales},
		"w.biases": {Dtype: "BF16", Shape: []int{out, in / gs}, Data: biases},
	}
	wantPacked, _, _, err := mlxaffine.RepackB1ToB2(packed, scales, biases, out, in, gs)
	if err != nil {
		t.Fatalf("reference RepackB1ToB2: %v", err)
	}
	l := LoadLinear(tensors, "w", in, "affine")
	if l == nil || !l.Quantised() {
		t.Fatalf("LoadLinear = %+v, want a quantised weight", l)
	}
	if l.Bits != 2 || l.GroupSize != gs {
		t.Fatalf("geometry after repack = bits %d gs %d, want bits 2 gs %d", l.Bits, l.GroupSize, gs)
	}
	if !bytes.Equal(l.Weight, wantPacked) {
		t.Fatal("repacked Weight bytes != RepackB1ToB2 reference")
	}
	if &l.Scales[0] != &scales[0] || &l.Biases[0] != &biases[0] {
		t.Fatal("scales/biases must keep the original tensor views (zero-copy invariant)")
	}
	// Writeback: the map now holds the b2 tensor, so a tied second read derives b2 from the
	// SAME backing array — no second repack, no second owned buffer.
	w2 := tensors["w.weight"]
	if lastDim(w2.Shape) != in*2/32 {
		t.Fatalf("written-back packed shape = %v, want last dim %d (b2 words per row)", w2.Shape, in*2/32)
	}
	l2 := LoadLinear(tensors, "w", in, "affine")
	if l2.Bits != 2 || &l2.Weight[0] != &l.Weight[0] {
		t.Fatalf("tied second read = bits %d, want 2 over the SAME repacked buffer", l2.Bits)
	}
}

// TestLoadLinear_B1RepackError — a malformed b1 pack (scales length not [out, in/gs]) passes
// through UNMODIFIED (Bits stays 1, bytes untouched, no writeback): the missing-b1-kernel
// condition then surfaces at session build, named, instead of a silent half-repack.
func TestLoadLinear_B1RepackError(t *testing.T) {
	const out, in = 4, 64
	packed := make([]byte, out*(in/32)*4)
	shortScales := make([]byte, 8) // not out*(in/gs)*2 for any gs the shape declares
	tensors := map[string]safetensors.Tensor{
		"w.weight": {Dtype: "U32", Shape: []int{out, in / 32}, Data: packed},
		"w.scales": {Dtype: "BF16", Shape: []int{out, 2}, Data: shortScales},
		"w.biases": {Dtype: "BF16", Shape: []int{out, 2}, Data: shortScales},
	}
	l := LoadLinear(tensors, "w", in, "affine")
	if l == nil {
		t.Fatal("LoadLinear returned nil for a present weight")
	}
	if l.Bits != 1 {
		t.Fatalf("Bits = %d after a failed repack, want 1 (pass through unmodified)", l.Bits)
	}
	if &l.Weight[0] != &packed[0] {
		t.Fatal("failed repack must leave the original packed view in place")
	}
	if lastDim(tensors["w.weight"].Shape) != in/32 {
		t.Fatal("failed repack must not write back a changed shape")
	}
}

// TestAffineGeometry_Guards covers the geometry helper's edge returns directly: a
// non-positive inDim can't encode a group size (division would be meaningless), and
// empty shapes yield zeroes — the "not a quantised weight" signal LoadLinear relies on.
func TestAffineGeometry_Guards(t *testing.T) {
	if gs, bits := affineGeometry(0, []int{4, 2}, []int{4, 8}); gs != 0 || bits != 0 {
		t.Fatalf("affineGeometry(inDim=0) = (%d,%d), want (0,0) — guard against a meaningless group size", gs, bits)
	}
	if gs, bits := affineGeometry(-1, []int{4, 2}, []int{4, 8}); gs != 0 || bits != 0 {
		t.Fatalf("affineGeometry(inDim<0) = (%d,%d), want (0,0)", gs, bits)
	}
	// positive inDim but empty shapes → 0,0 (lastDim guards both reads).
	if gs, bits := affineGeometry(64, nil, nil); gs != 0 || bits != 0 {
		t.Fatalf("affineGeometry(empty shapes) = (%d,%d), want (0,0)", gs, bits)
	}
}

// TestDimHelpers — lastDim/firstDim return 0 for a rank-0/empty shape (the guard that lets
// the geometry math treat "no shape" as "not encoded") and the boundary dims otherwise.
func TestDimHelpers(t *testing.T) {
	if got := lastDim(nil); got != 0 {
		t.Fatalf("lastDim(nil) = %d, want 0", got)
	}
	if got := firstDim(nil); got != 0 {
		t.Fatalf("firstDim(nil) = %d, want 0", got)
	}
	if got := lastDim([]int{4, 7}); got != 7 {
		t.Fatalf("lastDim([4,7]) = %d, want 7", got)
	}
	if got := firstDim([]int{4, 7}); got != 4 {
		t.Fatalf("firstDim([4,7]) = %d, want 4", got)
	}
}

// TestLinear_LoadLinear_Good covers the ordinary quantised load: OutDim from the shape,
// Kind/GroupSize/Bits derived from the scales tensor.
func TestLinear_LoadLinear_Good(t *testing.T) {
	const out, in = 4, 64
	mk := func(shape ...int) safetensors.Tensor {
		n := 1
		for _, d := range shape {
			n *= d
		}
		return safetensors.Tensor{Shape: shape, Data: make([]byte, n)}
	}
	t2 := map[string]safetensors.Tensor{
		"w.weight": mk(out, in*4/32), "w.scales": mk(out, in/32), "w.biases": mk(out, in/32),
	}
	l := LoadLinear(t2, "w", in, "affine")
	if l == nil || l.OutDim != out || !l.Quantised() || l.GroupSize != 32 || l.Bits != 4 {
		t.Fatalf("LoadLinear = %+v, want a quantised 4-bit/group-32 weight of OutDim %d", l, out)
	}
}

// TestLinear_LoadLinear_Bad covers the absent-weight case: an optional weight not in the
// checkpoint loads as nil, the caller's "feature absent" signal.
func TestLinear_LoadLinear_Bad(t *testing.T) {
	if l := LoadLinear(map[string]safetensors.Tensor{}, "missing", 64, "affine"); l != nil {
		t.Fatalf("LoadLinear(absent) = %+v, want nil", l)
	}
}

// TestLinear_LoadLinear_Ugly covers a present .scales tensor with EMPTY data (len(s.Data)
// == 0): LoadLinear must treat this as "not quantised" (dense bf16), the guard in the
// ok && len(s.Data) > 0 check — a subtly different edge from .scales absent entirely.
func TestLinear_LoadLinear_Ugly(t *testing.T) {
	const out, in = 4, 64
	t2 := map[string]safetensors.Tensor{
		"w.weight": {Shape: []int{out, in}, Data: make([]byte, out*in)},
		"w.scales": {Shape: []int{out, in / 32}, Data: []byte{}}, // present but empty
	}
	l := LoadLinear(t2, "w", in, "affine")
	if l == nil {
		t.Fatal("LoadLinear returned nil for a present weight")
	}
	if l.Quantised() {
		t.Fatalf("LoadLinear with empty .scales data should NOT be quantised: %+v", l)
	}
}

// TestLinear_Linear_Quantised_Good covers the quantised weight: Scales set AND Kind set
// both required for Quantised() to report true.
func TestLinear_Linear_Quantised_Good(t *testing.T) {
	l := &Linear{Scales: []byte{1}, Kind: "affine"}
	if !l.Quantised() {
		t.Fatal("Quantised() = false with Scales+Kind both set, want true")
	}
}

// TestLinear_Linear_Quantised_Bad covers the dense bf16 weight: no Scales and no Kind
// means Quantised() reports false.
func TestLinear_Linear_Quantised_Bad(t *testing.T) {
	l := &Linear{Weight: []byte{1, 2}}
	if l.Quantised() {
		t.Fatal("Quantised() = true with no Scales/Kind, want false")
	}
}

// TestLinear_Linear_Quantised_Ugly covers a nil receiver AND a half-set weight (Scales
// without Kind, or Kind without Scales): Quantised must not panic on nil, and must
// require BOTH fields, not just one.
func TestLinear_Linear_Quantised_Ugly(t *testing.T) {
	var nilLin *Linear
	if nilLin.Quantised() {
		t.Fatal("Quantised() on a nil *Linear = true, want false")
	}
	if (&Linear{Scales: []byte{1}}).Quantised() {
		t.Fatal("Quantised() with Scales but no Kind = true, want false (both required)")
	}
	if (&Linear{Kind: "affine"}).Quantised() {
		t.Fatal("Quantised() with Kind but no Scales = true, want false (both required)")
	}
}
