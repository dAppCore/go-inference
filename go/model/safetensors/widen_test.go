// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"bytes"
	"math"
	"testing"
)

// le16 (the little-endian uint16 packer the safetensors payload uses) is shared from values_test.go.

// TestWiden_WidenF16ToBF16_KnownValues pins the widen against hand-computed bfloat16 encodings —
// the independent oracle the "F16 tensor widens to the correct bf16 bytes" gate wants (not a
// round-trip through the same conversion). Each F16 input is paired with the bf16 bit pattern its
// exact float32 value rounds to.
func TestWiden_WidenF16ToBF16_KnownValues(t *testing.T) {
	cases := []struct {
		name     string
		f16      uint16 // IEEE-754 half input
		wantBF16 uint16 // expected bfloat16 output
	}{
		{"one", 0x3C00, 0x3F80},         // 1.0
		{"two", 0x4000, 0x4000},         // 2.0
		{"half", 0x3800, 0x3F00},        // 0.5
		{"neg_one", 0xBC00, 0xBF80},     // -1.0
		{"zero", 0x0000, 0x0000},        // +0.0
		{"neg_zero", 0x8000, 0x8000},    // -0.0
		{"round_up", 0x3C05, 0x3F81},    // 1.0048828125 → rounds up (low16 0xA000 > 0x8000)
		{"round_down", 0x3C01, 0x3F80},  // 1.0009765625 → rounds down to 1.0
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := WidenF16ToBF16(le16(c.f16))
			if want := le16(c.wantBF16); !bytes.Equal(got, want) {
				t.Fatalf("WidenF16ToBF16(%#04x) = % x, want % x (bf16 %#04x)", c.f16, got, want, c.wantBF16)
			}
		})
	}
}

// TestWiden_WidenF16ToBF16_LengthAndMulti checks that widening preserves the 2-bytes/element byte
// length across a multi-element payload and a trailing odd byte is dropped (never a valid F16 pair).
func TestWiden_WidenF16ToBF16_LengthAndMulti(t *testing.T) {
	in := append(append(le16(0x3C00), le16(0x4000)...), le16(0xBC00)...) // 1.0, 2.0, -1.0
	got := WidenF16ToBF16(in)
	want := append(append(le16(0x3F80), le16(0x4000)...), le16(0xBF80)...)
	if !bytes.Equal(got, want) {
		t.Fatalf("multi widen = % x, want % x", got, want)
	}
	if len(got) != len(in) {
		t.Fatalf("byte length changed: in %d, out %d", len(in), len(got))
	}
	if odd := WidenF16ToBF16([]byte{0x00, 0x3C, 0x00}); len(odd) != 2 {
		t.Fatalf("trailing odd byte not dropped: len %d", len(odd))
	}
	if empty := WidenF16ToBF16(nil); len(empty) != 0 {
		t.Fatalf("empty input widened to %d bytes", len(empty))
	}
}

// TestWiden_Float32ToBFloat16_RoundToNearestEven pins the scalar round matching mlx's AsType,
// including the NaN-stays-quiet and signed-zero edges.
func TestWiden_Float32ToBFloat16_RoundToNearestEven(t *testing.T) {
	if got := Float32ToBFloat16(1.0); got != 0x3F80 {
		t.Fatalf("Float32ToBFloat16(1.0) = %#04x, want 0x3F80", got)
	}
	// Round-trip a bf16 value: bf16 → f32 → bf16 is the identity (no low mantissa bits to round).
	for _, b := range []uint16{0x3F80, 0x4000, 0xBF80, 0x0000, 0x8000, 0x3F00} {
		if got := Float32ToBFloat16(BFloat16ToFloat32(b)); got != b {
			t.Fatalf("round-trip bf16 %#04x → %#04x", b, got)
		}
	}
	// NaN stays a quiet NaN (exponent all ones, non-zero mantissa).
	nan := Float32ToBFloat16(math.Float32frombits(0x7FC00000))
	if nan&0x7F80 != 0x7F80 || nan&0x007F == 0 {
		t.Fatalf("Float32ToBFloat16(NaN) = %#04x, not a quiet NaN", nan)
	}
}

// TestWiden_WidenF16TensorsToBF16_FlipsF16LeavesOthers checks the DirMapping pass widens only F16
// tensors: the F16 tensor flips to BF16 with widened bytes, and the BF16 + U32 tensors stay
// byte-identical (a bf16 pack loads unchanged).
func TestWiden_WidenF16TensorsToBF16_FlipsF16LeavesOthers(t *testing.T) {
	bf16Orig := le16(0x3F80)
	u32Orig := []byte{0xDE, 0xAD, 0xBE, 0xEF}
	dm := &DirMapping{Tensors: map[string]Tensor{
		"norm.weight":   {Dtype: "F16", Shape: []int{1}, Data: le16(0x3C00)},
		"other.weight":  {Dtype: "BF16", Shape: []int{1}, Data: bf16Orig},
		"packed.weight": {Dtype: "U32", Shape: []int{1}, Data: u32Orig},
	}}
	if n := dm.WidenF16TensorsToBF16(); n != 1 {
		t.Fatalf("widened count = %d, want 1", n)
	}
	norm := dm.Tensors["norm.weight"]
	if norm.Dtype != "BF16" {
		t.Fatalf("widened tensor dtype = %q, want BF16", norm.Dtype)
	}
	if !bytes.Equal(norm.Data, le16(0x3F80)) {
		t.Fatalf("widened norm data = % x, want % x", norm.Data, le16(0x3F80))
	}
	if other := dm.Tensors["other.weight"]; other.Dtype != "BF16" || !bytes.Equal(other.Data, bf16Orig) {
		t.Fatalf("bf16 tensor mutated: dtype %q data % x", other.Dtype, other.Data)
	}
	if packed := dm.Tensors["packed.weight"]; packed.Dtype != "U32" || !bytes.Equal(packed.Data, u32Orig) {
		t.Fatalf("u32 tensor mutated: dtype %q data % x", packed.Dtype, packed.Data)
	}
}

// TestWiden_IsWidened_TracksWidenedRanges checks the binder's discriminator: a widened tensor's Data
// reports true, an untouched shard-view tensor reports false, and a nil mapping is safe.
func TestWiden_IsWidened_TracksWidenedRanges(t *testing.T) {
	notWidened := le16(0x3F80)
	dm := &DirMapping{Tensors: map[string]Tensor{
		"norm.weight": {Dtype: "F16", Shape: []int{1}, Data: le16(0x3C00)},
	}}
	dm.WidenF16TensorsToBF16()
	if !dm.IsWidened(dm.Tensors["norm.weight"].Data) {
		t.Fatal("widened tensor Data reported not widened")
	}
	if dm.IsWidened(notWidened) {
		t.Fatal("unrelated buffer reported widened")
	}
	var nilDM *DirMapping
	if nilDM.IsWidened(notWidened) {
		t.Fatal("nil mapping reported widened")
	}
}
