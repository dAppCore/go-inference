// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"
)

func TestRoPEDimsBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const batch, nHeads, headDim, rotaryDim = 1, 8, 64, 32
	x := toBF16Bytes(syntheticFloat32(batch*nHeads*headDim, 5))
	if _, err := RoPEDimsBF16(x, batch, nHeads, headDim, rotaryDim, 10000, 1, 7, false); err != nil {
		t.Fatalf("RoPEDimsBF16 warmup: %v", err)
	}

	var ropeErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, ropeErr = RoPEDimsBF16(x, batch, nHeads, headDim, rotaryDim, 10000, 1, 7, false)
	})
	if ropeErr != nil {
		t.Fatalf("RoPEDimsBF16: %v", ropeErr)
	}
	if allocs > 10 {
		t.Fatalf("RoPEDimsBF16 allocations = %.0f, want <= 10", allocs)
	}
}

func TestRoPEDimsBF16IntoUsesCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const batch, nHeads, headDim, rotaryDim = 1, 8, 64, 32
	x := toBF16Bytes(syntheticFloat32(batch*nHeads*headDim, 5))
	out := make([]byte, len(x))
	for i := range out {
		out[i] = 0xA5
	}

	got, err := RoPEDimsBF16Into(out, x, batch, nHeads, headDim, rotaryDim, 10000, 1, 7, false)
	if err != nil {
		t.Fatalf("RoPEDimsBF16Into: %v", err)
	}
	if len(got) != len(out) {
		t.Fatalf("RoPEDimsBF16Into len = %d, want %d", len(got), len(out))
	}
	if unsafe.Pointer(&got[0]) != unsafe.Pointer(&out[0]) {
		t.Fatal("RoPEDimsBF16Into did not return caller-owned output backing")
	}
	want, err := RoPEDimsBF16(x, batch, nHeads, headDim, rotaryDim, 10000, 1, 7, false)
	if err != nil {
		t.Fatalf("RoPEDimsBF16 reference: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("RoPEDimsBF16Into output differs from allocating wrapper")
	}
}

// TestRoPEDimsPartial gates partial rotary: rotaryDim == headDim is byte-identical to RoPEBF16,
// and rotaryDim < headDim rotates only the first rotaryDim (its block ≡ a full RoPE of that
// sub-vector — kernel vs kernel, so byte-exact) while the tail [rotaryDim:headDim] passes
// through unchanged. This is gemma4's partial_rotary_factor (full_attention 0.25).
func TestRoPEDimsPartial(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const b, nHeads, headDim, rotaryDim = 1, 2, 8, 4
	const base, scale, offset = float32(10000), float32(1), 5
	xf := make([]float32, b*nHeads*headDim)
	for i := range xf {
		xf[i] = float32((i*7+3)%17-8) * 0.1
	}
	x := toBF16Bytes(xf)

	// rotaryDim == headDim ≡ RoPEBF16, byte-for-byte (full rotary unchanged).
	full, err := RoPEDimsBF16(x, b, nHeads, headDim, headDim, base, scale, offset, false)
	if err != nil {
		t.Fatalf("full RoPEDimsBF16: %v", err)
	}
	ref, err := RoPEBF16(x, b, nHeads, headDim, base, scale, offset, false)
	if err != nil {
		t.Fatalf("RoPEBF16: %v", err)
	}
	if !bytes.Equal(full, ref) {
		t.Fatal("rotaryDim == headDim diverged from RoPEBF16")
	}

	// partial: rotate the first rotaryDim, pass the tail through.
	part, err := RoPEDimsBF16(x, b, nHeads, headDim, rotaryDim, base, scale, offset, false)
	if err != nil {
		t.Fatalf("partial RoPEDimsBF16: %v", err)
	}
	rowB, rotB := headDim*bf16Size, rotaryDim*bf16Size
	for h := 0; h < nHeads; h++ {
		head := x[h*rowB : (h+1)*rowB]
		// the rotated block must equal a full RoPE of just the first rotaryDim as its own head.
		subRoped, err := RoPEBF16(head[:rotB], 1, 1, rotaryDim, base, scale, offset, false)
		if err != nil {
			t.Fatalf("sub RoPEBF16: %v", err)
		}
		got := part[h*rowB : (h+1)*rowB]
		if !bytes.Equal(got[:rotB], subRoped) {
			t.Fatalf("head %d: rotated block != full RoPE of the sub-vector", h)
		}
		if !bytes.Equal(got[rotB:], head[rotB:]) {
			t.Fatalf("head %d: tail [rotaryDim:headDim] was not passed through", h)
		}
	}
	if bytes.Equal(part, full) {
		t.Fatal("partial == full — rotaryDim had no effect")
	}

	if _, err := RoPEDimsBF16(x, b, nHeads, headDim, 3, base, scale, offset, false); err == nil {
		t.Fatal("odd rotaryDim: expected an error")
	}
	if _, err := RoPEDimsBF16(x, b, nHeads, headDim, headDim+2, base, scale, offset, false); err == nil {
		t.Fatal("rotaryDim > headDim: expected an error")
	}
	t.Logf("partial rotary: rotaryDim=%d/%d rotates the first block (≡ full RoPE of the sub-vector) and passes the tail through; full rotary byte-identical to RoPEBF16", rotaryDim, headDim)
}
