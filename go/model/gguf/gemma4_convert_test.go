// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"math"
	"testing"
)

func gemma4DecodeF32(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

// TestGemma4Convert_isGemma4Config checks model_type detection.
func TestGemma4Convert_isGemma4Config(t *testing.T) {
	if !isGemma4Config([]byte(`{"model_type": "gemma4"}`)) {
		t.Error("gemma4 config not detected")
	}
	if isGemma4Config([]byte(`{"model_type": "llama"}`)) {
		t.Error("llama config detected as gemma4")
	}
	if isGemma4Config([]byte("{ not json")) {
		t.Error("malformed config detected as gemma4")
	}
}

// TestGemma4Convert_gemma4CanonicalLayerIndex checks block-index extraction.
func TestGemma4Convert_gemma4CanonicalLayerIndex(t *testing.T) {
	cases := map[string]int{
		"blk.0.attn_q.weight":    0,
		"blk.31.ffn_down.weight": 31,
		"token_embd.weight":      -1,
		"output_norm.weight":     -1,
		"rope_freqs.weight":      -1,
	}
	for name, want := range cases {
		if got := gemma4CanonicalLayerIndex(name); got != want {
			t.Errorf("gemma4CanonicalLayerIndex(%q) = %d, want %d", name, got, want)
		}
	}
}

// TestGemma4Convert_gemma4RopeFreqsTensor_Gemma4E2B checks the computed
// rope_freqs mask matches the oracle shape: 256 entries, the first 64 rotating
// (1.0) and the remaining 192 disabled (1e30), for dimension_count 512 /
// partial_rotary_factor 0.25.
func TestGemma4Convert_gemma4RopeFreqsTensor_Gemma4E2B(t *testing.T) {
	tensor, err := gemma4RopeFreqsTensor(512, 0.25)
	if err != nil {
		t.Fatalf("gemma4RopeFreqsTensor: %v", err)
	}
	if tensor.Name != "rope_freqs.weight" || tensor.Type != ggufTensorTypeF32 {
		t.Errorf("tensor name/type = %q/%d, want rope_freqs.weight/F32", tensor.Name, tensor.Type)
	}
	if len(tensor.Shape) != 1 || tensor.Shape[0] != 256 {
		t.Fatalf("shape = %v, want [256]", tensor.Shape)
	}
	freqs := gemma4DecodeF32(tensor.Data)
	if len(freqs) != 256 {
		t.Fatalf("decoded %d freqs, want 256", len(freqs))
	}
	for i, v := range freqs {
		want := float32(1.0)
		if i >= 64 {
			want = gemma4RopeFreqsDisabled
		}
		if v != want {
			t.Errorf("rope_freqs[%d] = %g, want %g", i, v, want)
		}
	}
}

// TestGemma4Convert_gemma4RopeFreqsTensor_Bad rejects invalid rope geometry.
func TestGemma4Convert_gemma4RopeFreqsTensor_Bad(t *testing.T) {
	for name, tc := range map[string]struct {
		dim    int
		factor float32
	}{
		"zero dim":        {0, 0.25},
		"odd dim":         {511, 0.25},
		"zero factor":     {512, 0},
		"factor over one": {512, 1.5},
	} {
		if _, err := gemma4RopeFreqsTensor(tc.dim, tc.factor); err == nil {
			t.Errorf("gemma4RopeFreqsTensor(%s): want error, got nil", name)
		}
	}
}

// TestGemma4Convert_float32ToBF16 checks bfloat16 rounding, including that a
// value already representable in bf16 (all decoded gemma-4 bf16 weights) is
// exact.
func TestGemma4Convert_float32ToBF16(t *testing.T) {
	// 1.0f = 0x3F800000; bf16 high half = 0x3F80.
	if got := float32ToBF16(1.0); got != 0x3F80 {
		t.Errorf("float32ToBF16(1.0) = %#04x, want 0x3f80", got)
	}
	// A bf16-origin value: bf16 0xC0A0 -> f32 0xC0A00000 = -5.0; round-trips.
	f := math.Float32frombits(0xC0A00000)
	if got := float32ToBF16(f); got != 0xC0A0 {
		t.Errorf("float32ToBF16(%g) = %#04x, want 0xc0a0", f, got)
	}
	// Round-to-nearest-even: 0x3F808000 rounds half-to-even down to 0x3F80.
	if got := float32ToBF16(math.Float32frombits(0x3F808000)); got != 0x3F80 {
		t.Errorf("float32ToBF16(half-even) = %#04x, want 0x3f80", got)
	}
}

// TestGemma4Convert_encodeGemma4TensorData_Unsupported errors on a type it has
// no encoder for (Q2_K — out of scope for the gemma4 lane per
// docs/design-quant-formats.md, gated on an operator decision) and on a
// K-quant tensor that is not a whole number of blocks.
func TestGemma4Convert_encodeGemma4TensorData_Unsupported(t *testing.T) {
	if _, err := encodeGemma4TensorData([]float32{1, 2, 3}, ggufTensorTypeQ2K); err == nil {
		t.Error("want error for unsupported encoder type")
	}
	if _, err := encodeGemma4TensorData(make([]float32, 100), ggufTensorTypeQ4K); err == nil {
		t.Error("want error for non-block-aligned Q4_K tensor")
	}
}

// TestGemma4Convert_encodeGemma4TensorData_Q8_0 checks the Q8_0 encoder
// (added for #53's q8_0 export lane) accepts a block-aligned (32-element)
// tensor and rejects a non-aligned one, mirroring the existing Q4_K/Q5_K/
// Q6_K coverage above.
func TestGemma4Convert_encodeGemma4TensorData_Q8_0(t *testing.T) {
	data, err := encodeGemma4TensorData(make([]float32, 64), TensorTypeQ8_0)
	if err != nil {
		t.Fatalf("encodeGemma4TensorData(64 elements, Q8_0): %v", err)
	}
	if len(data) != 2*34 { // 64/32 blocks * 34 bytes/block.
		t.Errorf("len(data) = %d, want %d", len(data), 2*34)
	}
	if _, err := encodeGemma4TensorData(make([]float32, 33), TensorTypeQ8_0); err == nil {
		t.Error("want error for non-block-aligned (33-element) Q8_0 tensor")
	}
}

// TestGemma4Convert_encodeGemma4TensorData_Q3_K checks the Q3_K encoder
// (added for #53's q3_k_m export lane) accepts a block-aligned (256-element)
// tensor and rejects a non-aligned one.
func TestGemma4Convert_encodeGemma4TensorData_Q3_K(t *testing.T) {
	data, err := encodeGemma4TensorData(make([]float32, 512), ggufTensorTypeQ3K)
	if err != nil {
		t.Fatalf("encodeGemma4TensorData(512 elements, Q3_K): %v", err)
	}
	if len(data) != 2*110 { // 512/256 blocks * 110 bytes/block.
		t.Errorf("len(data) = %d, want %d", len(data), 2*110)
	}
	if _, err := encodeGemma4TensorData(make([]float32, 100), ggufTensorTypeQ3K); err == nil {
		t.Error("want error for non-block-aligned (100-element) Q3_K tensor")
	}
}
