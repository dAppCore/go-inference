// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"bytes"
	"math"
	"testing"
)

// le16 (the little-endian uint16 packer the safetensors payload uses) is shared from values_test.go.

// TestWiden_Float32ToBFloat16_Good documents the exact (no-rounding-needed) happy path: values
// whose low mantissa bits are already zero convert with nothing to round.
func TestWiden_Float32ToBFloat16_Good(t *testing.T) {
	if got := Float32ToBFloat16(1.0); got != 0x3F80 {
		t.Fatalf("Float32ToBFloat16(1.0) = %#04x, want 0x3F80", got)
	}
	if got := Float32ToBFloat16(2.0); got != 0x4000 {
		t.Fatalf("Float32ToBFloat16(2.0) = %#04x, want 0x4000", got)
	}
}

// TestWiden_Float32ToBFloat16_Bad documents round-to-nearest-even: a value whose low 16 bits sit
// above the halfway point rounds UP, not down — the bug a naive truncating implementation would
// introduce. 1.0048828125 is the exact float32 value F16 0x3C05 decodes to (see
// TestWiden_WidenF16ToBF16_KnownValues below); its correct bf16 rounding is 0x3F81, one ULP above
// the truncated 0x3F80.
func TestWiden_Float32ToBFloat16_Bad(t *testing.T) {
	if got := Float32ToBFloat16(1.0048828125); got != 0x3F81 {
		t.Fatalf("Float32ToBFloat16(1.0048828125) = %#04x, want 0x3F81 (round up, not truncate)", got)
	}
}

// TestWiden_Float32ToBFloat16_Ugly covers signed zero and infinity: both survive the conversion
// exactly, carrying their sign bit through untouched.
func TestWiden_Float32ToBFloat16_Ugly(t *testing.T) {
	if got := Float32ToBFloat16(0.0); got != 0x0000 {
		t.Fatalf("Float32ToBFloat16(+0.0) = %#04x, want 0x0000", got)
	}
	if got := Float32ToBFloat16(float32(math.Copysign(0, -1))); got != 0x8000 {
		t.Fatalf("Float32ToBFloat16(-0.0) = %#04x, want 0x8000", got)
	}
	if got := Float32ToBFloat16(float32(math.Inf(1))); got != 0x7F80 {
		t.Fatalf("Float32ToBFloat16(+Inf) = %#04x, want 0x7F80", got)
	}
	if got := Float32ToBFloat16(float32(math.Inf(-1))); got != 0xFF80 {
		t.Fatalf("Float32ToBFloat16(-Inf) = %#04x, want 0xFF80", got)
	}
}

// TestWiden_WidenF16ToBF16_Good documents the happy path on a clean two-element payload: both
// values need no rounding, and the output stays at the same byte length as the input.
func TestWiden_WidenF16ToBF16_Good(t *testing.T) {
	in := append(le16(0x3C00), le16(0x4000)...) // 1.0, 2.0
	got := WidenF16ToBF16(in)
	want := append(le16(0x3F80), le16(0x4000)...)
	if !bytes.Equal(got, want) {
		t.Fatalf("WidenF16ToBF16 = % x, want % x", got, want)
	}
}

// TestWiden_WidenF16ToBF16_Bad documents that a trailing odd byte (never a valid F16 element) is
// dropped rather than causing a panic or an out-of-range element.
func TestWiden_WidenF16ToBF16_Bad(t *testing.T) {
	got := WidenF16ToBF16([]byte{0x00, 0x3C, 0x00})
	if len(got) != 2 {
		t.Fatalf("trailing odd byte not dropped: len %d, want 2", len(got))
	}
}

// TestWiden_WidenF16ToBF16_Ugly covers empty input, both nil and non-nil.
func TestWiden_WidenF16ToBF16_Ugly(t *testing.T) {
	if got := WidenF16ToBF16(nil); len(got) != 0 {
		t.Fatalf("nil input widened to %d bytes, want 0", len(got))
	}
	if got := WidenF16ToBF16([]byte{}); len(got) != 0 {
		t.Fatalf("empty (non-nil) input widened to %d bytes, want 0", len(got))
	}
}

// TestWiden_DirMapping_WidenF16TensorsToBF16_Good documents the minimal happy path: a single F16
// tensor widens to BF16 and the reported widened count is 1.
func TestWiden_DirMapping_WidenF16TensorsToBF16_Good(t *testing.T) {
	dm := &DirMapping{Tensors: map[string]Tensor{
		"w": {Dtype: "F16", Shape: []int{1}, Data: le16(0x3C00)},
	}}
	if n := dm.WidenF16TensorsToBF16(); n != 1 {
		t.Fatalf("widened count = %d, want 1", n)
	}
	if dtype := dm.Tensors["w"].Dtype; dtype != "BF16" {
		t.Fatalf("widened dtype = %q, want BF16", dtype)
	}
}

// TestWiden_DirMapping_WidenF16TensorsToBF16_Bad documents the nil-receiver guard: a nil
// *DirMapping is a safe no-op reporting zero tensors widened.
func TestWiden_DirMapping_WidenF16TensorsToBF16_Bad(t *testing.T) {
	var nilDM *DirMapping
	if n := nilDM.WidenF16TensorsToBF16(); n != 0 {
		t.Fatalf("nil DirMapping widened count = %d, want 0", n)
	}
}

// TestWiden_DirMapping_WidenF16TensorsToBF16_Ugly covers an F16 tensor with empty Data: the
// length guard skips it rather than widening a zero-byte payload.
func TestWiden_DirMapping_WidenF16TensorsToBF16_Ugly(t *testing.T) {
	dm := &DirMapping{Tensors: map[string]Tensor{
		"empty": {Dtype: "F16", Shape: []int{0}, Data: nil},
	}}
	if n := dm.WidenF16TensorsToBF16(); n != 0 {
		t.Fatalf("widened count = %d, want 0 (empty Data must be skipped)", n)
	}
	if dtype := dm.Tensors["empty"].Dtype; dtype != "F16" {
		t.Fatalf("skipped tensor dtype = %q, want unchanged F16", dtype)
	}
}

// TestWiden_DirMapping_IsWidened_Good documents the happy path: a tensor's Data this mapping
// widened reports true.
func TestWiden_DirMapping_IsWidened_Good(t *testing.T) {
	dm := &DirMapping{Tensors: map[string]Tensor{
		"w": {Dtype: "F16", Shape: []int{1}, Data: le16(0x3C00)},
	}}
	dm.WidenF16TensorsToBF16()
	if !dm.IsWidened(dm.Tensors["w"].Data) {
		t.Fatal("widened tensor Data reported not widened")
	}
}

// TestWiden_DirMapping_IsWidened_Bad documents that an unrelated buffer — never produced by this
// mapping's WidenF16TensorsToBF16 — reports false, even after real widening has occurred.
func TestWiden_DirMapping_IsWidened_Bad(t *testing.T) {
	dm := &DirMapping{Tensors: map[string]Tensor{
		"w": {Dtype: "F16", Shape: []int{1}, Data: le16(0x3C00)},
	}}
	dm.WidenF16TensorsToBF16()
	if dm.IsWidened(le16(0x3F80)) {
		t.Fatal("unrelated buffer reported widened")
	}
}

// TestWiden_DirMapping_IsWidened_Ugly covers both length-zero early-return shapes: a nil
// receiver and an empty (non-nil) buffer.
func TestWiden_DirMapping_IsWidened_Ugly(t *testing.T) {
	var nilDM *DirMapping
	if nilDM.IsWidened(le16(0x3F80)) {
		t.Fatal("nil mapping reported widened")
	}
	dm := &DirMapping{}
	if dm.IsWidened([]byte{}) {
		t.Fatal("empty buffer reported widened")
	}
}

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
		{"one", 0x3C00, 0x3F80},        // 1.0
		{"two", 0x4000, 0x4000},        // 2.0
		{"half", 0x3800, 0x3F00},       // 0.5
		{"neg_one", 0xBC00, 0xBF80},    // -1.0
		{"zero", 0x0000, 0x0000},       // +0.0
		{"neg_zero", 0x8000, 0x8000},   // -0.0
		{"round_up", 0x3C05, 0x3F81},   // 1.0048828125 → rounds up (low16 0xA000 > 0x8000)
		{"round_down", 0x3C01, 0x3F80}, // 1.0009765625 → rounds down to 1.0
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
