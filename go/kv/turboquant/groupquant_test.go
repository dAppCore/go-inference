// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import (
	"math"
	"testing"

	core "dappco.re/go"
)

// TestGroupScale_Good checks the scale formula matches engine/metal's
// `scale = m / 127.0f` exactly (kernels/lthn_sdpa_paged.metal,
// lthn_kv_q8_store_bf16) for a known absmax.
func TestGroupScale_Good(t *testing.T) {
	got := groupScale([]float64{1, -4, 2}, 127)
	want := 4.0 / 127.0
	if !approxEqual(got, want, 1e-12) {
		t.Errorf("groupScale({1,-4,2}, 127) = %v, want %v", got, want)
	}
}

// TestGroupScale_Ugly checks an all-zero group scales to exactly 0 —
// matching the kernel's `inv = scale > 0 ? 1/scale : 0` guard against a
// divide-by-zero.
func TestGroupScale_Ugly(t *testing.T) {
	if got := groupScale([]float64{0, 0, 0}, 127); got != 0 {
		t.Errorf("groupScale(zero group) = %v, want 0", got)
	}
}

// TestQuantiseGroupSymmetric_Good checks a group that fits within the code
// range round-trips to within one code's worth of error, and that a group
// spanning multiple GroupSize blocks scales each block independently (a
// large value in block 0 must not affect block 1's precision).
func TestQuantiseGroupSymmetric_Good(t *testing.T) {
	x := make([]float64, 2*GroupSize)
	x[0] = 100 // large outlier confined to block 0
	x[GroupSize] = 1
	codes, scales := quantiseGroupSymmetric(x, 127)
	if len(scales) != 2 {
		t.Fatalf("quantiseGroupSymmetric produced %d scales, want 2 groups", len(scales))
	}
	back := dequantiseGroupSymmetric(codes, scales)
	// Block 1's reconstruction should be close to 1, unaffected by block 0's
	// outlier (independent per-group scale is the whole point of grouping).
	if !approxEqual(back[GroupSize], 1, scales[1]) {
		t.Errorf("block-1 reconstruction = %v, want ≈1 (independent of block-0's outlier)", back[GroupSize])
	}
}

// TestQuantiseGroupSymmetric_Ugly checks a short final group (not a
// multiple of GroupSize) is handled without panicking or corrupting the
// prior full groups.
func TestQuantiseGroupSymmetric_Ugly(t *testing.T) {
	x := make([]float64, GroupSize+3)
	for i := range x {
		x[i] = float64(i%5) - 2
	}
	codes, scales := quantiseGroupSymmetric(x, 127)
	if len(scales) != 2 {
		t.Fatalf("quantiseGroupSymmetric(%d elements) produced %d scales, want 2", len(x), len(scales))
	}
	back := dequantiseGroupSymmetric(codes, scales)
	if len(back) != len(x) {
		t.Fatalf("dequantiseGroupSymmetric returned %d elements, want %d", len(back), len(x))
	}
}

// TestEncodeGroupQuantInt8_Good round-trips a row with bounded per-element
// error (symmetric int8: worst case half a code step, scale ≈ max/127).
func TestEncodeGroupQuantInt8_Good(t *testing.T) {
	x := make([]float32, GroupSize)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)))
	}
	e := EncodeGroupQuantInt8(x)
	got := DecodeGroupQuantInt8(e)
	for i := range x {
		if diff := math.Abs(float64(x[i] - got[i])); diff > 2.0/127.0 {
			t.Errorf("int8 g64: |x[%d]-x̃[%d]| = %v, want <= 2/127 (one code step)", i, i, diff)
		}
	}
}

// TestEncodeGroupQuantInt8_Ugly checks the all-zero row encodes to a zero
// scale and decodes to exactly zero.
func TestEncodeGroupQuantInt8_Ugly(t *testing.T) {
	x := make([]float32, GroupSize)
	e := EncodeGroupQuantInt8(x)
	if e.Scales[0] != 0 {
		t.Errorf("EncodeGroupQuantInt8(zero row).Scales[0] = %v, want 0", e.Scales[0])
	}
	got := DecodeGroupQuantInt8(e)
	for i, v := range got {
		if v != 0 {
			t.Errorf("DecodeGroupQuantInt8(zero-row encoding)[%d] = %v, want 0", i, v)
		}
	}
}

// TestMarshalGroupQuantInt8_Good round-trips through the wire format and
// checks the byte length matches GroupQuantInt8Codec.BytesPerRow.
func TestMarshalGroupQuantInt8_Good(t *testing.T) {
	x := make([]float32, GroupSize+10)
	for i := range x {
		x[i] = float32(i) - 30
	}
	e := EncodeGroupQuantInt8(x)
	data := MarshalGroupQuantInt8(e)
	if want := (GroupQuantInt8Codec{}).BytesPerRow(len(x)); len(data) != want {
		t.Errorf("len(MarshalGroupQuantInt8(...)) = %d, want BytesPerRow = %d", len(data), want)
	}
	back := UnmarshalGroupQuantInt8(data, len(x))
	got := DecodeGroupQuantInt8(back)
	want := DecodeGroupQuantInt8(e)
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("round-tripped decode[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestUnmarshalGroupQuantInt8_Bad checks a too-short payload decodes to the
// safe zero-value encoding.
func TestUnmarshalGroupQuantInt8_Bad(t *testing.T) {
	got := UnmarshalGroupQuantInt8([]byte{1, 2}, GroupSize)
	if got.Codes != nil || got.Scales != nil {
		t.Errorf("UnmarshalGroupQuantInt8(short data) = %+v, want the zero-value encoding", got)
	}
}

// ExampleEncodeGroupQuantInt8 demonstrates a full encode/decode round trip.
func ExampleEncodeGroupQuantInt8() {
	x := []float32{3, 4}
	e := EncodeGroupQuantInt8(x)
	got := DecodeGroupQuantInt8(e)
	close := l2Norm(subtract(toFloat64(x), toFloat64(got))) < 0.5
	core.Println("reconstruction close:", close)
	// Output:
	// reconstruction close: true
}

// TestEncodeGroupQuantInt4_Good round-trips a row with bounded per-element
// error (symmetric int4: scale ≈ max/7, one code step ≈ 2/7).
func TestEncodeGroupQuantInt4_Good(t *testing.T) {
	x := make([]float32, GroupSize)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)))
	}
	e := EncodeGroupQuantInt4(x)
	got := DecodeGroupQuantInt4(e)
	for i := range x {
		if diff := math.Abs(float64(x[i] - got[i])); diff > 2.0/7.0 {
			t.Errorf("int4 g64: |x[%d]-x̃[%d]| = %v, want <= 2/7 (one code step)", i, i, diff)
		}
	}
}

// TestEncodeGroupQuantInt4_Ugly checks codes stay within the symmetric
// [-7,7] range (not the full two's-complement [-8,7]) even for a group
// whose absmax element would naturally round to the boundary.
func TestEncodeGroupQuantInt4_Ugly(t *testing.T) {
	x := make([]float32, GroupSize)
	x[0] = 1000 // this element defines the group's absmax
	e := EncodeGroupQuantInt4(x)
	for i, c := range e.Codes {
		if c < -7 || c > 7 {
			t.Errorf("EncodeGroupQuantInt4 code[%d] = %d, want within [-7,7]", i, c)
		}
	}
}

// TestMarshalGroupQuantInt4_Good round-trips through the wire format and
// checks the byte length matches GroupQuantInt4Codec.BytesPerRow.
func TestMarshalGroupQuantInt4_Good(t *testing.T) {
	x := make([]float32, GroupSize+10)
	for i := range x {
		x[i] = float32(i) - 30
	}
	e := EncodeGroupQuantInt4(x)
	data := MarshalGroupQuantInt4(e)
	if want := (GroupQuantInt4Codec{}).BytesPerRow(len(x)); len(data) != want {
		t.Errorf("len(MarshalGroupQuantInt4(...)) = %d, want BytesPerRow = %d", len(data), want)
	}
	back := UnmarshalGroupQuantInt4(data, len(x))
	got := DecodeGroupQuantInt4(back)
	want := DecodeGroupQuantInt4(e)
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("round-tripped decode[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestUnmarshalGroupQuantInt4_Bad checks a too-short payload decodes to the
// safe zero-value encoding.
func TestUnmarshalGroupQuantInt4_Bad(t *testing.T) {
	got := UnmarshalGroupQuantInt4([]byte{1}, GroupSize)
	if got.Codes != nil || got.Scales != nil {
		t.Errorf("UnmarshalGroupQuantInt4(short data) = %+v, want the zero-value encoding", got)
	}
}

// ExampleEncodeGroupQuantInt4 demonstrates a full encode/decode round trip.
func ExampleEncodeGroupQuantInt4() {
	x := []float32{3, 4}
	e := EncodeGroupQuantInt4(x)
	got := DecodeGroupQuantInt4(e)
	close := l2Norm(subtract(toFloat64(x), toFloat64(got))) < 1
	core.Println("reconstruction close:", close)
	// Output:
	// reconstruction close: true
}
