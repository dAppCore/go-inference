// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"math"
	"testing"
)

func foldBF16Bytes(vals []float32) []byte {
	out := make([]byte, len(vals)*2)
	for i, v := range vals {
		bits := math.Float32bits(v)
		r := uint16((bits + 0x7fff + ((bits >> 16) & 1)) >> 16)
		out[2*i], out[2*i+1] = byte(r), byte(r>>8)
	}
	return out
}

func foldBF16At(b []byte, i int) float32 {
	return math.Float32frombits(uint32(uint16(b[2*i])|uint16(b[2*i+1])<<8) << 16)
}

// TestFoldNormBiasOne verifies the gemma (1+weight) RMSNorm fold adds exactly 1 to every norm element,
// in bf16 and f32 — the convention metal applies via AddScalar(weight, 1.0).
func TestFoldNormBiasOne(t *testing.T) {
	in := []float32{0.5, -0.25, 2.0, 0.0, -1.0, 0.125}

	folded, err := foldNormBiasOne(foldBF16Bytes(in), "BF16")
	if err != nil {
		t.Fatalf("bf16 fold: %v", err)
	}
	for i, v := range in {
		got := foldBF16At(folded, i)
		want := foldBF16At(foldBF16Bytes([]float32{v + 1}), 0) // bf16(v+1)
		if got != want {
			t.Errorf("bf16[%d]: folded %v, want bf16(%v+1)=%v", i, got, v, want)
		}
	}

	f32 := make([]byte, len(in)*4)
	for i, v := range in {
		b := math.Float32bits(v)
		f32[4*i], f32[4*i+1], f32[4*i+2], f32[4*i+3] = byte(b), byte(b>>8), byte(b>>16), byte(b>>24)
	}
	ff, err := foldNormBiasOne(f32, "F32")
	if err != nil {
		t.Fatalf("f32 fold: %v", err)
	}
	for i, v := range in {
		bits := uint32(ff[4*i]) | uint32(ff[4*i+1])<<8 | uint32(ff[4*i+2])<<16 | uint32(ff[4*i+3])<<24
		if got := math.Float32frombits(bits); got != v+1 {
			t.Errorf("f32[%d]: folded %v, want %v", i, got, v+1)
		}
	}

	if _, err := foldNormBiasOne([]byte{1, 2}, "I8"); err == nil {
		t.Error("unsupported dtype should error, not silently mis-fold")
	}
	t.Log("gemma (1+w) norm fold verified: +1 added byte-correct in bf16 and f32")
}
