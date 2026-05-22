// SPDX-Licence-Identifier: EUPL-1.2

package jang

import (
	"testing"

	core "dappco.re/go"
)

func testJANGTQInfo() *Info {
	return &Info{
		Version:          2,
		WeightFormat:     "mxtq",
		Profile:          "JANGTQ",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		AttentionBits:    8,
		SharedExpertBits: 8,
		RoutedExpertBits: 2,
		EmbedTokensBits:  8,
		LMHeadBits:       8,
	}
}

func TestJang_PackedTensorDescriptorMXTQRoutedExpert_Good(t *testing.T) {
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.17.w1.weight", []uint64{2, 4}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewPackedTensorDescriptor() error = %v", err)
	}

	if desc.Type != "jangtq" || desc.Format != "mxtq" || desc.Profile != "JANGTQ" {
		t.Fatalf("profile = type:%q format:%q profile:%q", desc.Type, desc.Format, desc.Profile)
	}
	if desc.Role != TensorRoleRoutedExpert || desc.Bits != 2 || desc.GroupSize != 4 {
		t.Fatalf("descriptor = %+v, want routed expert 2-bit group 4", desc)
	}
	if desc.Elements != 8 || desc.Groups != 2 || desc.PackedBytes != 2 || desc.ScaleCount != 2 || desc.BiasCount != 2 {
		t.Fatalf("descriptor sizes = %+v, want 8 elements, 2 groups, 2 packed bytes", desc)
	}
	if desc.BitOrder != BitOrderLSB0 || desc.Encoding != EncodingAffine {
		t.Fatalf("layout = bit_order:%q encoding:%q", desc.BitOrder, desc.Encoding)
	}
}

func TestJang_PackedTensorDescriptorAttentionUsesWideBits_Good(t *testing.T) {
	desc, err := NewPackedTensorDescriptor("model.layers.0.self_attn.q_proj.weight", []uint64{2, 4}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewPackedTensorDescriptor() error = %v", err)
	}

	if desc.Role != TensorRoleAttention || desc.Bits != 8 || desc.PackedBytes != 8 {
		t.Fatalf("descriptor = %+v, want attention 8-bit un-nibbled bytes", desc)
	}
}

func TestJang_PackedTensorDescriptorBadUnsupportedBits(t *testing.T) {
	info := testJANGTQInfo()
	info.RoutedExpertBits = 5

	_, err := NewPackedTensorDescriptor("model.layers.0.mlp.experts.0.down_proj.weight", []uint64{4, 4}, info)
	if err == nil || !core.Contains(err.Error(), "unsupported") || !core.Contains(err.Error(), "5-bit") {
		t.Fatalf("error = %v, want explicit unsupported 5-bit error", err)
	}
}

func TestJang_DequantizePackedTensor_Good(t *testing.T) {
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.3.w2.weight", []uint64{8}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewPackedTensorDescriptor() error = %v", err)
	}
	packed, err := PackQuantizedValues(desc, []uint8{0, 1, 2, 3, 0, 1, 2, 3})
	if err != nil {
		t.Fatalf("PackQuantizedValues() error = %v", err)
	}

	out, err := DequantizePackedTensor(desc, packed, []float32{0.5, 1}, []float32{-1, 10})
	if err != nil {
		t.Fatalf("DequantizePackedTensor() error = %v", err)
	}

	want := []float32{-1, -0.5, 0, 0.5, 10, 11, 12, 13}
	if len(out) != len(want) {
		t.Fatalf("out length = %d, want %d", len(out), len(want))
	}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out[%d] = %v, want %v (all=%v)", i, out[i], want[i], out)
		}
	}
}

func TestJang_ValidatePackedTensorBadPackedLength(t *testing.T) {
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.3.w2.weight", []uint64{8}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewPackedTensorDescriptor() error = %v", err)
	}

	err = ValidatePackedTensor(desc, []byte{0}, []float32{1, 1}, []float32{0, 0})
	if err == nil || !core.Contains(err.Error(), "packed length") {
		t.Fatalf("error = %v, want packed length validation", err)
	}
}

// roundTripFixture builds a descriptor at the requested bit width with the
// MXTQ routed-expert tensor name (the inferTensorRole route that picks up
// RoutedExpertBits) and feeds it crafted values such that every group is
// exercised. Returns descriptor + the values written in.
func roundTripFixture(t *testing.T, bits int, elements int, groupSize int) (PackedTensorDescriptor, []uint8, []byte, []float32, []float32) {
	t.Helper()
	info := &Info{
		Version:          2,
		WeightFormat:     "mxtq",
		Profile:          "JANGTQ",
		Method:           "affine+mxtq",
		GroupSize:        groupSize,
		BitsDefault:      bits,
		RoutedExpertBits: bits,
	}
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.0.w1.weight", []uint64{uint64(elements)}, info)
	if err != nil {
		t.Fatalf("NewPackedTensorDescriptor(%d-bit): %v", bits, err)
	}
	maxValue := uint8((1 << bits) - 1)
	values := make([]uint8, desc.Elements)
	for i := range values {
		// Walk the full 0..maxValue range so every nibble/lane is touched.
		values[i] = uint8(i) & maxValue
	}
	packed, err := PackQuantizedValues(desc, values)
	if err != nil {
		t.Fatalf("PackQuantizedValues(%d-bit): %v", bits, err)
	}
	// Distinct per-group scale + bias so a regression that mis-indexes groups
	// surfaces as a wrong magnitude, not a hidden silent identity.
	scales := make([]float32, desc.ScaleCount)
	biases := make([]float32, desc.BiasCount)
	for i := range scales {
		scales[i] = 0.25 + float32(i)*0.0625
		biases[i] = -1 - float32(i)*0.5
	}
	return desc, values, packed, scales, biases
}

// expectedDequantize is the smallest possible reference dequant — pure
// per-element arithmetic with the generic unpack walk used by upstream
// before the W10-N specialisation. Used as the bit-exact oracle.
func expectedDequantize(t *testing.T, values []uint8, scales, biases []float32, groupSize int) []float32 {
	t.Helper()
	out := make([]float32, len(values))
	for i, v := range values {
		group := i / groupSize
		out[i] = float32(v)*scales[group] + biases[group]
	}
	return out
}

func TestJang_DequantizePackedTensor_RoundTrip_1bit(t *testing.T) {
	// 4096 elements with groupSize=64 to exercise the multi-group dispatch.
	desc, values, packed, scales, biases := roundTripFixture(t, 1, 4096, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(1-bit): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

func TestJang_DequantizePackedTensor_RoundTrip_2bit(t *testing.T) {
	desc, values, packed, scales, biases := roundTripFixture(t, 2, 4096, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(2-bit): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

func TestJang_DequantizePackedTensor_RoundTrip_3bit(t *testing.T) {
	// 3-bit hits the generic-walk default branch — the dequant must still
	// be bit-exact against the pre-specialisation oracle.
	desc, values, packed, scales, biases := roundTripFixture(t, 3, 4096, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(3-bit): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

func TestJang_DequantizePackedTensor_RoundTrip_4bit(t *testing.T) {
	desc, values, packed, scales, biases := roundTripFixture(t, 4, 4096, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(4-bit): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

func TestJang_DequantizePackedTensor_RoundTrip_8bit(t *testing.T) {
	desc, values, packed, scales, biases := roundTripFixture(t, 8, 4096, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(8-bit): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

// TestJang_DequantizePackedTensor_RoundTrip_2bit_ShortTail exercises the
// case where the tensor's element count is NOT a multiple of groupSize,
// so the final group runs short and the 2-bit suffix-drain path covers
// the tail.
func TestJang_DequantizePackedTensor_RoundTrip_2bit_ShortTail(t *testing.T) {
	// 130 elements with groupSize=64 → 3 groups, last group has 2 elements.
	desc, values, packed, scales, biases := roundTripFixture(t, 2, 130, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(2-bit short tail): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

// TestJang_DequantizePackedTensor_RoundTrip_2bit_GroupSize2 exercises the
// case where groupSize < 4 — the 2-bit batched fast path can't fire on a
// 4-elements-per-byte stride, so the per-element prefix path must cover
// every element.
func TestJang_DequantizePackedTensor_RoundTrip_2bit_GroupSize2(t *testing.T) {
	desc, values, packed, scales, biases := roundTripFixture(t, 2, 32, 2)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(2-bit groupSize=2): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

// TestJang_DequantizePackedTensor_RoundTrip_4bit_ShortTail covers the
// 4-bit prefix + suffix drains around the batched 2-per-byte fast path
// when the final group is shorter than groupSize.
func TestJang_DequantizePackedTensor_RoundTrip_4bit_ShortTail(t *testing.T) {
	// 67 elements with groupSize=64 → last group has 3 elements; the
	// 2-per-byte batched path takes 2 of them, the suffix drains the 1.
	desc, values, packed, scales, biases := roundTripFixture(t, 4, 67, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(4-bit short tail): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

// TestJang_DequantizePackedTensor_RoundTrip_4bit_GroupSize1 covers the
// degenerate case where groupSize=1, forcing every element into the
// suffix-drain path (no batched stride can fire).
func TestJang_DequantizePackedTensor_RoundTrip_4bit_GroupSize1(t *testing.T) {
	desc, values, packed, scales, biases := roundTripFixture(t, 4, 16, 1)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(4-bit groupSize=1): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

// TestJang_DequantizePackedTensor_RoundTrip_1bit_ShortTail covers the
// 1-bit prefix + suffix drains around the batched 8-per-byte fast path
// when the final group is shorter than groupSize.
func TestJang_DequantizePackedTensor_RoundTrip_1bit_ShortTail(t *testing.T) {
	// 133 elements with groupSize=64 → last group has 5 elements; the
	// 8-per-byte batched path can't fire, suffix-drain takes all 5.
	desc, values, packed, scales, biases := roundTripFixture(t, 1, 133, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(1-bit short tail): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

// TestJang_DequantizePackedTensor_RoundTrip_1bit_GroupSize4 covers the
// case where groupSize=4 < 8, so the 8-per-byte batched fast path can
// never fire and the prefix path must cover every element.
func TestJang_DequantizePackedTensor_RoundTrip_1bit_GroupSize4(t *testing.T) {
	desc, values, packed, scales, biases := roundTripFixture(t, 1, 32, 4)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(1-bit groupSize=4): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

func assertBitExact(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("dequant[%d] = %v, want %v (delta=%v)", i, got[i], want[i], got[i]-want[i])
		}
	}
}

func TestJang_BuildPackedProfile_Good(t *testing.T) {
	profile := BuildPackedProfile(testJANGTQInfo())
	if profile == nil {
		t.Fatal("profile = nil")
	}
	if profile.Type != "jangtq" || profile.Format != "mxtq" || !profile.Mixed {
		t.Fatalf("profile = %+v, want JANGTQ/MXTQ mixed profile", profile)
	}
	if profile.MinBits != 2 || profile.MaxBits != 8 || profile.RoleBits[string(TensorRoleRoutedExpert)] != 2 || profile.RoleBits[string(TensorRoleAttention)] != 8 {
		t.Fatalf("role bits = %+v, min/max=%d/%d", profile.RoleBits, profile.MinBits, profile.MaxBits)
	}
}
