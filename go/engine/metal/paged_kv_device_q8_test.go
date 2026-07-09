// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"
	"unsafe"
)

func bf16FromF32(f float32) [2]byte {
	bits := math.Float32bits(f)
	bits += 0x7FFF + ((bits >> 16) & 1)
	return [2]byte{byte(bits >> 16), byte(bits >> 24)}
}

func f32FromBF16(lo, hi byte) float32 {
	return math.Float32frombits((uint32(lo) | uint32(hi)<<8) << 16)
}

// TestDevicePagedKVQ8RoundTrip loads bf16 rows into a q8 cache through
// loadLinearSnapshot (the host quant path every prefill/restore rides) and
// reads them back through linearSnapshot (the host dequant path the batched
// verify, state save and drafter export ride). Group-symmetric int8 with an
// f32 scale bounds the per-element error at maxabs/127 within each 64-group.
func TestDevicePagedKVQ8RoundTrip(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatalf("ensureInit: %v", err)
	}
	const (
		nKVHeads = 2
		headDim  = 128 // kvDim 256 = 4 groups per row
		rows     = 3000
	)
	c, err := newDevicePagedKVCache(nKVHeads, headDim, 0, 2048)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	c.quantQ8 = true
	kvDim := nKVHeads * headDim
	rowBytes := kvDim * bf16Size
	kLin := make([]byte, rows*rowBytes)
	vLin := make([]byte, rows*rowBytes)
	want := make([]float32, rows*kvDim)
	for i := range want {
		// per-group varying magnitudes exercise the scale path
		g := i / kvQ8GroupSize
		want[i] = float32((i%97)-48) * (0.001 + float32(g%7)*0.03)
		b := bf16FromF32(want[i])
		kLin[i*2], kLin[i*2+1] = b[0], b[1]
		vLin[i*2], vLin[i*2+1] = b[1], b[0] // different bytes down the V path
	}
	if err := c.loadLinearSnapshot(kLin, vLin, rows); err != nil {
		t.Fatalf("loadLinearSnapshot: %v", err)
	}
	if got := c.rowElemBytes(); got != 1 {
		t.Fatalf("rowElemBytes = %d, want 1", got)
	}
	if len(c.kScalePages) != len(c.kPages) {
		t.Fatalf("scale pages = %d, data pages = %d", len(c.kScalePages), len(c.kPages))
	}
	_, _, kPtr, _, err := c.linearSnapshot(rows)
	if err != nil {
		t.Fatalf("linearSnapshot: %v", err)
	}
	out := unsafe.Slice(kPtr, rows*rowBytes)
	worst := float64(0)
	for g := 0; g < rows*kvDim/kvQ8GroupSize; g++ {
		maxAbs := float32(0)
		for i := range kvQ8GroupSize {
			// the bf16 SOURCE values bound the group scale
			src := f32FromBF16(kLin[(g*kvQ8GroupSize+i)*2], kLin[(g*kvQ8GroupSize+i)*2+1])
			if a := absF32(src); a > maxAbs {
				maxAbs = a
			}
		}
		tol := float64(maxAbs)/127*0.5 + 1e-6 + float64(maxAbs)*(1.0/128)
		for i := range kvQ8GroupSize {
			idx := g*kvQ8GroupSize + i
			src := f32FromBF16(kLin[idx*2], kLin[idx*2+1])
			got := f32FromBF16(out[idx*2], out[idx*2+1])
			if d := math.Abs(float64(got - src)); d > tol {
				t.Fatalf("group %d elem %d: got %v want %v (|d|=%g > tol %g)", g, i, got, src, d, tol)
			} else if d > worst {
				worst = d
			}
		}
	}
	t.Logf("q8 round-trip worst abs error %.5g over %d elems", worst, rows*kvDim)
}
