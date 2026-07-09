// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// TestSDPAPagedQ8Parity runs the q8 pass-1/pass-2 pair against the bf16 pair
// over IDENTICAL content: rows are quantised host-side into a q8 cache, then
// the bf16 reference cache is loaded with the DEQUANTISED rows, so both
// kernels see the same values and only the arithmetic path differs. Covers
// the multi-cell route and the single-cell final at gqa2 geometry.
func TestSDPAPagedQ8Parity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatalf("ensureInit: %v", err)
	}
	const (
		nHeads   = 16
		nKVHeads = 8
		headDim  = 256
	)
	kvDim := nKVHeads * headDim
	rowBytes := kvDim * bf16Size
	scale := float32(1.0 / 16.0)

	run := func(name string, rows int) {
		q8c, err := newDevicePagedKVCache(nKVHeads, headDim, 0, 2048)
		if err != nil {
			t.Fatalf("%s: new q8: %v", name, err)
		}
		q8c.quantQ8 = true
		bfc, err := newDevicePagedKVCache(nKVHeads, headDim, 0, 2048)
		if err != nil {
			t.Fatalf("%s: new bf16: %v", name, err)
		}
		kLin := make([]byte, rows*rowBytes)
		vLin := make([]byte, rows*rowBytes)
		for i := 0; i < rows*kvDim; i++ {
			f := float32(math.Sin(float64(i)*0.37)) * (0.3 + float32(i%5)*0.4)
			b := bf16FromF32(f)
			kLin[i*2], kLin[i*2+1] = b[0], b[1]
			g := bf16FromF32(f * 0.7)
			vLin[i*2], vLin[i*2+1] = g[0], g[1]
		}
		if err := q8c.loadLinearSnapshot(kLin, vLin, rows); err != nil {
			t.Fatalf("%s: q8 load: %v", name, err)
		}
		// dequantise the q8 cache back to bf16 rows for the reference cache
		_, _, kPtr, vPtr, err := q8c.linearSnapshot(rows)
		if err != nil {
			t.Fatalf("%s: q8 snapshot: %v", name, err)
		}
		kDeq := append([]byte(nil), unsafe.Slice(kPtr, rows*rowBytes)...)
		vDeq := append([]byte(nil), unsafe.Slice(vPtr, rows*rowBytes)...)
		if err := bfc.loadLinearSnapshot(kDeq, vDeq, rows); err != nil {
			t.Fatalf("%s: bf16 load: %v", name, err)
		}

		q := scratchBF16(nHeads * headDim)
		qs := unsafe.Slice((*byte)(q.Contents()), nHeads*headDim*2)
		for i := 0; i < nHeads*headDim; i++ {
			b := bf16FromF32(float32(math.Cos(float64(i) * 0.11)))
			qs[i*2], qs[i*2+1] = b[0], b[1]
		}
		outQ8 := scratchBF16(nHeads * headDim)
		outBF := scratchBF16(nHeads * headDim)

		attend := func(c *devicePagedKVCache, out metal.MTLBuffer, q8 bool) {
			keys, values, lens, kh, ks, vh, vs, serr := c.state()
			if serr != nil {
				t.Fatalf("%s: state: %v", name, serr)
			}
			scratch, serr2 := newSDPAPagedDecodeScratch(nHeads, headDim)
			if serr2 != nil {
				t.Fatalf("%s: scratch: %v", name, serr2)
			}
			plan, perr := buildSDPAPagedDecodePlan(q, keys, values, lens, kh, ks, vh, vs, out, scratch, nHeads, nKVHeads, headDim, scale)
			if perr != nil {
				t.Fatalf("%s: plan: %v", name, perr)
			}
			if q8 {
				kSc, vSc := c.scaleState()
				if aerr := plan.attachQ8(kSc, vSc); aerr != nil {
					t.Fatalf("%s: attachQ8: %v", name, aerr)
				}
			}
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			plan.emitP1s(enc)
			plan.emitP2(enc)
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
		}
		attend(q8c, outQ8, true)
		attend(bfc, outBF, false)

		a := unsafe.Slice((*byte)(outQ8.Contents()), nHeads*headDim*2)
		b := unsafe.Slice((*byte)(outBF.Contents()), nHeads*headDim*2)
		worst := 0.0
		for i := 0; i < nHeads*headDim; i++ {
			fa := float64(f32FromBF16(a[i*2], a[i*2+1]))
			fb := float64(f32FromBF16(b[i*2], b[i*2+1]))
			d := math.Abs(fa - fb)
			if d > worst {
				worst = d
			}
			if d > 0.02 {
				t.Fatalf("%s: head-dim %d: q8 %v vs bf16 %v (|d|=%g)", name, i, fa, fb, d)
			}
		}
		t.Logf("%s: worst |q8-bf16| = %.5g over %d dims", name, worst, nHeads*headDim)
	}

	run("multi-cell 5000 rows", 5000)
	run("single-cell 100 rows", 100)
}
