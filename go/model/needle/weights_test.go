// SPDX-Licence-Identifier: EUPL-1.2

package needle

import "testing"

// TestWeights_widenBF16_Exact confirms bf16 -> f32 is an exact left-shift by 16:
// 1.0 (0x3f80) and 2.0 (0x4000) widen with no rounding.
func TestWeights_widenBF16_Exact(t *testing.T) {
	// little-endian bf16 bytes for 1.0, 2.0, -1.0
	raw := []byte{0x80, 0x3f, 0x00, 0x40, 0x80, 0xbf}
	got := widenBF16(raw)
	want := []float32{1, 2, -1}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("widenBF16[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestWeights_widenF32_Exact confirms f32 bytes reinterpret losslessly.
func TestWeights_widenF32_Exact(t *testing.T) {
	// little-endian f32 bytes for 1.0 and 0.5
	raw := []byte{0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x3f}
	got := widenF32(raw)
	want := []float32{1, 0.5}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("widenF32[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestWeights_LoadWeights_RealCheckpoint loads the published weights and spot
// checks a known tensor's shape, exercising the safetensors + widen path.
func TestWeights_LoadWeights_RealCheckpoint(t *testing.T) {
	path := snapshotDir + "/model.safetensors"
	w, err := LoadWeights(path)
	if err != nil {
		t.Skipf("needle checkpoint not loadable: %v", err)
	}
	embed := w.shape("model.embed_tokens.weight")
	if len(embed) != 2 || embed[0] != 8192 || embed[1] != 512 {
		t.Fatalf("embed_tokens shape = %v, want [8192 512]", embed)
	}
	if got := w.get("model.encoder.layers.0.attn_gate"); len(got) != 1 {
		t.Fatalf("attn_gate len = %d, want 1 (scalar gate)", len(got))
	}
}
