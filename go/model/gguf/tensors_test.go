// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	pkgsafetensors "dappco.re/go/inference/model/safetensors"
)

// The expected f16 bit patterns below are the canonical IEEE-754 binary16
// encodings (1 sign / 5 exponent, bias 15 / 10 mantissa), derived by hand
// rather than from the implementation under test:
//   0.0 → 0x0000   1.0 → 0x3C00   2.0 → 0x4000   3.0 → 0x4200
//  -1.0 → 0xBC00  -2.0 → 0xC000   4.0 → 0x4400  -4.0 → 0xC400
//  +Inf → 0x7C00  -Inf → 0xFC00   NaN → 0x7E00  2^-24 → 0x0001

// TestTensors_GgufFloat16ToFloat32_Good pins the widen of normal + zero f16 bit
// patterns to their exact float32 values.
func TestTensors_GgufFloat16ToFloat32_Good(t *testing.T) {
	cases := []struct {
		bits uint16
		want float32
	}{
		{0x0000, 0},
		{0x8000, 0}, // negative zero widens to 0.0 numerically
		{0x3C00, 1},
		{0xBC00, -1},
		{0x4000, 2},
		{0xC000, -2},
		{0x4200, 3},
		{0x3800, 0.5},
		{0x0001, math.Float32frombits(0x33800000)}, // smallest subnormal = 2^-24
	}
	for _, c := range cases {
		if got := ggufFloat16ToFloat32(c.bits); got != c.want {
			t.Fatalf("ggufFloat16ToFloat32(%#04x) = %v, want %v", c.bits, got, c.want)
		}
	}
}

// TestTensors_GgufFloat16ToFloat32_Bad pins the exp==31 specials: the two
// infinities and a NaN payload.
func TestTensors_GgufFloat16ToFloat32_Bad(t *testing.T) {
	if got := ggufFloat16ToFloat32(0x7C00); !math.IsInf(float64(got), 1) {
		t.Fatalf("ggufFloat16ToFloat32(0x7C00) = %v, want +Inf", got)
	}
	if got := ggufFloat16ToFloat32(0xFC00); !math.IsInf(float64(got), -1) {
		t.Fatalf("ggufFloat16ToFloat32(0xFC00) = %v, want -Inf", got)
	}
	if got := ggufFloat16ToFloat32(0x7E00); !math.IsNaN(float64(got)) {
		t.Fatalf("ggufFloat16ToFloat32(0x7E00) = %v, want NaN", got)
	}
}

// TestTensors_GgufFloat32ToFloat16_Good pins the pack of exactly-representable
// float32 values to their canonical f16 bit patterns.
func TestTensors_GgufFloat32ToFloat16_Good(t *testing.T) {
	cases := []struct {
		value float32
		want  uint16
	}{
		{0, 0x0000},
		{1, 0x3C00},
		{-1, 0xBC00},
		{2, 0x4000},
		{-2, 0xC000},
		{3, 0x4200},
		{0.5, 0x3800},
		{4, 0x4400},
	}
	for _, c := range cases {
		if got := ggufFloat32ToFloat16(c.value); got != c.want {
			t.Fatalf("ggufFloat32ToFloat16(%v) = %#04x, want %#04x", c.value, got, c.want)
		}
	}
}

// TestTensors_GgufFloat32ToFloat16_Bad pins the exp==255 and overflow arms:
// a finite value beyond f16 range saturates to Inf, and the infinities / NaN
// map to their f16 specials.
func TestTensors_GgufFloat32ToFloat16_Bad(t *testing.T) {
	if got := ggufFloat32ToFloat16(70000); got != 0x7C00 { // > f16 max 65504 → +Inf
		t.Fatalf("ggufFloat32ToFloat16(70000) = %#04x, want 0x7C00", got)
	}
	if got := ggufFloat32ToFloat16(float32(math.Inf(1))); got != 0x7C00 {
		t.Fatalf("ggufFloat32ToFloat16(+Inf) = %#04x, want 0x7C00", got)
	}
	if got := ggufFloat32ToFloat16(float32(math.Inf(-1))); got != 0xFC00 {
		t.Fatalf("ggufFloat32ToFloat16(-Inf) = %#04x, want 0xFC00", got)
	}
	if got := ggufFloat32ToFloat16(float32(math.NaN())); got != 0x7E00 {
		t.Fatalf("ggufFloat32ToFloat16(NaN) = %#04x, want 0x7E00", got)
	}
}

// TestTensors_GgufFloat32ToFloat16_Ugly pins the subnormal boundary: 2^-24 is
// the smallest representable f16 (0x0001), while 2^-26 underflows to zero.
func TestTensors_GgufFloat32ToFloat16_Ugly(t *testing.T) {
	if got := ggufFloat32ToFloat16(math.Float32frombits(0x33800000)); got != 0x0001 { // 2^-24
		t.Fatalf("ggufFloat32ToFloat16(2^-24) = %#04x, want 0x0001", got)
	}
	if got := ggufFloat32ToFloat16(math.Float32frombits(0x32800000)); got != 0x0000 { // 2^-26
		t.Fatalf("ggufFloat32ToFloat16(2^-26) = %#04x, want 0x0000", got)
	}
}

// TestTensors_GgufCheckedMul_Good pins the non-overflowing products, including
// the zero short-circuit.
func TestTensors_GgufCheckedMul_Good(t *testing.T) {
	if got, ok := ggufCheckedMul(6, 7); got != 42 || !ok {
		t.Fatalf("ggufCheckedMul(6, 7) = %d, %v, want 42, true", got, ok)
	}
	if got, ok := ggufCheckedMul(0, math.MaxUint64); got != 0 || !ok {
		t.Fatalf("ggufCheckedMul(0, MaxUint64) = %d, %v, want 0, true", got, ok)
	}
}

// TestTensors_GgufCheckedMul_Bad pins the overflow guard: a product past
// MaxUint64 reports ok=false without wrapping.
func TestTensors_GgufCheckedMul_Bad(t *testing.T) {
	if got, ok := ggufCheckedMul(math.MaxUint64, 2); ok || got != 0 {
		t.Fatalf("ggufCheckedMul(MaxUint64, 2) = %d, %v, want 0, false", got, ok)
	}
}

// TestTensors_GgufTensorShapeElements_Good pins the row-major element product
// and shape []int conversion.
func TestTensors_GgufTensorShapeElements_Good(t *testing.T) {
	shape, elements, err := ggufTensorShapeElements(TensorInfo{Name: "w", Shape: []uint64{2, 3, 4}})
	if err != nil {
		t.Fatalf("ggufTensorShapeElements: %v", err)
	}
	if elements != 24 {
		t.Fatalf("elements = %d, want 24", elements)
	}
	if len(shape) != 3 || shape[0] != 2 || shape[1] != 3 || shape[2] != 4 {
		t.Fatalf("shape = %v, want [2 3 4]", shape)
	}
}

// TestTensors_GgufTensorShapeElements_Bad pins the element-count overflow guard.
func TestTensors_GgufTensorShapeElements_Bad(t *testing.T) {
	_, _, err := ggufTensorShapeElements(TensorInfo{Name: "huge", Shape: []uint64{math.MaxUint64, 2}})
	if err == nil {
		t.Fatal("ggufTensorShapeElements(MaxUint64 × 2) = nil error, want element-count overflow")
	}
}

// TestTensors_GgufTensorShapeElements_Ugly pins the zero-dimension case: a
// dimension of 0 yields 0 elements and skips the overflow multiply.
func TestTensors_GgufTensorShapeElements_Ugly(t *testing.T) {
	shape, elements, err := ggufTensorShapeElements(TensorInfo{Name: "z", Shape: []uint64{4, 0}})
	if err != nil {
		t.Fatalf("ggufTensorShapeElements(zero dim): %v", err)
	}
	if elements != 0 {
		t.Fatalf("elements = %d, want 0", elements)
	}
	if len(shape) != 2 || shape[0] != 4 || shape[1] != 0 {
		t.Fatalf("shape = %v, want [4 0]", shape)
	}
}

// TestTensors_GgufTensorNativeStorage_Good pins the dtype + byte-size that each
// supported GGUF type maps to for the native load: dense types keep their
// width, Q4_0/Q8_0 report the F16 dequantised size.
func TestTensors_GgufTensorNativeStorage_Good(t *testing.T) {
	cases := []struct {
		typ       uint32
		elements  uint64
		wantDtype string
		wantSize  uint64
	}{
		{ggufTensorTypeF32, 4, "F32", 16},
		{ggufTensorTypeF16, 4, "F16", 8},
		{ggufTensorTypeBF16, 4, "BF16", 8},
		{TensorTypeQ4_0, 32, "F16", 18}, // 1 block × 18 bytes on-disk
		{TensorTypeQ8_0, 32, "F16", 34}, // 1 block × 34 bytes on-disk
	}
	for _, c := range cases {
		dtype, size, err := ggufTensorNativeStorage(TensorInfo{Name: "w", Type: c.typ}, c.elements)
		if err != nil {
			t.Fatalf("ggufTensorNativeStorage(type %d): %v", c.typ, err)
		}
		if dtype != c.wantDtype || size != c.wantSize {
			t.Fatalf("ggufTensorNativeStorage(type %d) = %q/%d, want %q/%d", c.typ, dtype, size, c.wantDtype, c.wantSize)
		}
	}
}

// TestTensors_GgufTensorNativeStorage_Bad pins the unsupported-type rejection.
func TestTensors_GgufTensorNativeStorage_Bad(t *testing.T) {
	_, _, err := ggufTensorNativeStorage(TensorInfo{Name: "w", Type: ggufTensorTypeQ4_0_8_8}, 32)
	if err == nil {
		t.Fatal("ggufTensorNativeStorage(unsupported type) = nil error, want rejection")
	}
}

// TestTensors_GgufTensorNativeBlockStorage_Bad pins the block-alignment guard:
// an element count not divisible by the block size is rejected.
func TestTensors_GgufTensorNativeBlockStorage_Bad(t *testing.T) {
	_, _, err := ggufTensorNativeStorage(TensorInfo{Name: "w", Type: TensorTypeQ4_0}, 5)
	if err == nil {
		t.Fatal("ggufTensorNativeStorage(Q4_0, 5 elements) = nil error, want block-alignment rejection")
	}
}

// TestTensors_GgufDequantizeQ8_0ToF16_Good widens a hand-built Q8_0 block
// (scale 1.0, so each stored int8 becomes that integer's f16 encoding) and
// asserts the whole F16 output buffer against independently-derived patterns.
func TestTensors_GgufDequantizeQ8_0ToF16_Good(t *testing.T) {
	block := make([]byte, 34)
	binary.LittleEndian.PutUint16(block[:2], 0x3C00) // scale = f16(1.0)
	// int8 samples in the first eight lanes; the rest stay 0.
	int8s := []int8{0, 1, 2, 3, -1, -2, 4, -4}
	for i, v := range int8s {
		block[2+i] = byte(v)
	}

	out, err := ggufDequantizeQ8_0ToF16(block, 32)
	if err != nil {
		t.Fatalf("ggufDequantizeQ8_0ToF16: %v", err)
	}
	if len(out) != 64 {
		t.Fatalf("output len = %d, want 64 bytes (32 × f16)", len(out))
	}

	want := make([]uint16, 32)
	for i, p := range []uint16{0x0000, 0x3C00, 0x4000, 0x4200, 0xBC00, 0xC000, 0x4400, 0xC400} {
		want[i] = p
	}
	for i := 0; i < 32; i++ {
		got := binary.LittleEndian.Uint16(out[i*2 : i*2+2])
		if got != want[i] {
			t.Fatalf("Q8_0 element %d = %#04x, want %#04x", i, got, want[i])
		}
	}
}

// TestTensors_GgufDequantizeQ8_0ToF16_Bad pins both length guards: an element
// count not divisible by 32, and a payload whose byte length disagrees with
// the block count.
func TestTensors_GgufDequantizeQ8_0ToF16_Bad(t *testing.T) {
	if _, err := ggufDequantizeQ8_0ToF16(make([]byte, 34), 31); err == nil {
		t.Fatal("ggufDequantizeQ8_0ToF16(elements=31) = nil error, want non-block-aligned rejection")
	}
	if _, err := ggufDequantizeQ8_0ToF16(make([]byte, 33), 32); err == nil {
		t.Fatal("ggufDequantizeQ8_0ToF16(short payload) = nil error, want length-mismatch rejection")
	}
}

// TestTensors_GgufDequantizeQ4_0ToF16_Good widens a hand-built Q4_0 block
// (scale 1.0). Each packed byte carries lane i in the low nibble and lane i+16
// in the high nibble, offset by -8; the assertion is the full 32-lane output.
func TestTensors_GgufDequantizeQ4_0ToF16_Good(t *testing.T) {
	block := make([]byte, 18)
	binary.LittleEndian.PutUint16(block[:2], 0x3C00) // scale = f16(1.0)
	// packed[0]: lo nibble 9 → +1 (lane 0), hi nibble 8 → 0 (lane 16).
	block[2] = (8 << 4) | 9
	// packed[1]: lo nibble 10 → +2 (lane 1), hi nibble 6 → -2 (lane 17).
	block[3] = (6 << 4) | 10
	// packed[2..15] stay 0x88 → both nibbles 8 → 0.
	for i := 4; i < 18; i++ {
		block[i] = (8 << 4) | 8
	}

	out, err := ggufDequantizeQ4_0ToF16(block, 32)
	if err != nil {
		t.Fatalf("ggufDequantizeQ4_0ToF16: %v", err)
	}
	if len(out) != 64 {
		t.Fatalf("output len = %d, want 64 bytes (32 × f16)", len(out))
	}

	want := make([]uint16, 32)
	want[0] = 0x3C00  // +1
	want[1] = 0x4000  // +2
	want[17] = 0xC000 // -2
	for i := 0; i < 32; i++ {
		got := binary.LittleEndian.Uint16(out[i*2 : i*2+2])
		if got != want[i] {
			t.Fatalf("Q4_0 element %d = %#04x, want %#04x", i, got, want[i])
		}
	}
}

// TestTensors_GgufDequantizeQ4_0ToF16_Bad pins the length guards for the 18-byte
// block format.
func TestTensors_GgufDequantizeQ4_0ToF16_Bad(t *testing.T) {
	if _, err := ggufDequantizeQ4_0ToF16(make([]byte, 18), 31); err == nil {
		t.Fatal("ggufDequantizeQ4_0ToF16(elements=31) = nil error, want non-block-aligned rejection")
	}
	if _, err := ggufDequantizeQ4_0ToF16(make([]byte, 17), 32); err == nil {
		t.Fatal("ggufDequantizeQ4_0ToF16(short payload) = nil error, want length-mismatch rejection")
	}
}

// TestTensors_LoadTensors_Good writes a GGUF file carrying a dense F32 tensor
// and a Q8_0 tensor, then loads it: the dense tensor views the file bytes
// unchanged, and the Q8_0 tensor is dequantised to an F16 payload.
func TestTensors_LoadTensors_Good(t *testing.T) {
	dense := []float32{1, 2, 3, 4}
	denseBytes := make([]byte, len(dense)*4)
	for i, v := range dense {
		binary.LittleEndian.PutUint32(denseBytes[i*4:i*4+4], math.Float32bits(v))
	}

	path := core.PathJoin(t.TempDir(), "load.gguf")
	tensors := []Tensor{
		{Name: "dense.weight", Type: ggufTensorTypeF32, Shape: []uint64{4}, Data: denseBytes},
		{Name: "quant.weight", Type: TensorTypeQ8_0, Shape: []uint64{32}, Data: quantizeQ8_0(rampBlock(32))},
	}
	metadata := []MetadataEntry{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: "llama"},
	}
	if err := WriteFile(path, metadata, tensors); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	mapping, err := LoadTensors(path)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}
	defer mapping.Close()

	got, ok := mapping.Tensors["dense.weight"]
	if !ok {
		t.Fatal("dense.weight missing from loaded tensors")
	}
	if got.Dtype != "F32" {
		t.Fatalf("dense dtype = %q, want F32", got.Dtype)
	}
	if len(got.Shape) != 1 || got.Shape[0] != 4 {
		t.Fatalf("dense shape = %v, want [4]", got.Shape)
	}
	for i := range denseBytes {
		if got.Data[i] != denseBytes[i] {
			t.Fatalf("dense byte %d = %d, want %d (dense tensors must view the file bytes)", i, got.Data[i], denseBytes[i])
		}
	}

	quant, ok := mapping.Tensors["quant.weight"]
	if !ok {
		t.Fatal("quant.weight missing from loaded tensors")
	}
	if quant.Dtype != "F16" {
		t.Fatalf("quant dtype = %q, want F16 (Q8_0 dequantises on load)", quant.Dtype)
	}
	if len(quant.Data) != 32*2 {
		t.Fatalf("quant payload = %d bytes, want 64 (32 × f16)", len(quant.Data))
	}
}

// TestTensors_LoadTensors_Bad rejects a path that is not a GGUF file.
func TestTensors_LoadTensors_Bad(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "not.gguf")
	if w := core.WriteFile(path, []byte("this is not a gguf file"), 0o644); !w.OK {
		t.Fatalf("seed non-gguf file: %v", w.Value)
	}
	if _, err := LoadTensors(path); err == nil {
		t.Fatal("LoadTensors(non-gguf) = nil error, want parse rejection")
	}
}

// TestTensors_Close_Good pins the nil-receiver no-op, the close-func handoff +
// field clear, and idempotency: the second Close is a no-op that never
// re-invokes the release func.
func TestTensors_Close_Good(t *testing.T) {
	if err := (*TensorMapping)(nil).Close(); err != nil {
		t.Fatalf("(*TensorMapping)(nil).Close() = %v, want nil", err)
	}

	closed := 0
	mapping := &TensorMapping{
		Data:    []byte{1, 2, 3},
		Tensors: map[string]pkgsafetensors.Tensor{"w": {}},
		close:   func() error { closed++; return nil },
	}
	if err := mapping.Close(); err != nil {
		t.Fatalf("Close() = %v, want nil", err)
	}
	if closed != 1 {
		t.Fatalf("release func called %d times, want 1", closed)
	}
	if mapping.Data != nil || mapping.Tensors != nil {
		t.Fatal("Close() left Data/Tensors set, want both cleared")
	}
	if err := mapping.Close(); err != nil {
		t.Fatalf("second Close() = %v, want nil", err)
	}
	if closed != 1 {
		t.Fatalf("release func re-invoked on second Close (%d), want 1", closed)
	}
}
