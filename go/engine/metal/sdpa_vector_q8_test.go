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

// TestSDPAVectorQ8Parity runs the linear q8 decode SDPA pair against the MLX
// bf16 kernels over IDENTICAL content — the paged-q8 method: rows are
// quantised host-side (the lthn_kv_q8_store math), the bf16 reference reads
// the DEQUANTISED rows, so both kernels see the same values and only the
// arithmetic path differs. Shapes are the dense family's GLOBAL layers (the
// deep-context scan this lane exists for): 12B/e2b kv=1 hd=512 (gqa 16/8),
// 31B kv=4 hd=512 (gqa 8) — plus hd=256 gqa2 (the sliding shape). Covers the
// single-pass and the 2-pass pair.
func TestSDPAVectorQ8Parity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatalf("ensureInit: %v", err)
	}
	type shape struct {
		name             string
		nHeads, nKVHeads int
		headDim          int
		n                int
	}
	shapes := []shape{
		{"12B-global kv1 hd512 gqa16", 16, 1, 512, 700},
		{"31B-global kv4 hd512 gqa8", 32, 4, 512, 700},
		{"e2b-global kv1 hd512 gqa8", 8, 1, 512, 300},
		{"sliding kv2 hd256 gqa2", 4, 2, 256, 500},
	}
	for _, sh := range shapes {
		kvDim := sh.nKVHeads * sh.headDim
		scale := float32(1.0 / math.Sqrt(float64(sh.headDim)))

		// content: seq-major [row][kvHead][headDim] rows, deterministic
		kRows := make([]byte, sh.n*kvDim*2)
		vRows := make([]byte, sh.n*kvDim*2)
		for i := 0; i < sh.n*kvDim; i++ {
			f := float32(math.Sin(float64(i)*0.37)) * (0.3 + float32(i%5)*0.4)
			b := bf16FromF32(f)
			kRows[i*2], kRows[i*2+1] = b[0], b[1]
			g := bf16FromF32(f * 0.7)
			vRows[i*2], vRows[i*2+1] = g[0], g[1]
		}
		kCodes, kScales := kvQ8QuantiseRows(kRows, kvDim)
		vCodes, vScales := kvQ8QuantiseRows(vRows, kvDim)
		kDeq := kvQ8DequantiseRows(kCodes, kScales)
		vDeq := kvQ8DequantiseRows(vCodes, vScales)

		q := make([]byte, sh.nHeads*sh.headDim*2)
		for i := 0; i < sh.nHeads*sh.headDim; i++ {
			b := bf16FromF32(float32(math.Cos(float64(i) * 0.11)))
			q[i*2], q[i*2+1] = b[0], b[1]
		}

		// strides in elements: seq stride = kvDim, head stride = headDim
		khs, kss := int64(sh.headDim), int64(kvDim)

		readOut := func(buf metal.MTLBuffer) []float64 {
			raw := unsafe.Slice((*byte)(buf.Contents()), sh.nHeads*sh.headDim*2)
			out := make([]float64, sh.nHeads*sh.headDim)
			for i := range out {
				out[i] = float64(f32FromBF16(raw[i*2], raw[i*2+1]))
			}
			return out
		}
		compare := func(lane string, got, want []float64) {
			t.Helper()
			worst := 0.0
			for i := range want {
				d := math.Abs(got[i] - want[i])
				if d > worst {
					worst = d
				}
				if d > 0.02 {
					t.Fatalf("%s %s: dim %d: q8 %v vs bf16 %v (|d|=%g)", sh.name, lane, i, got[i], want[i], d)
				}
			}
			t.Logf("%s %s: worst |q8-bf16| = %.5g over %d dims", sh.name, lane, worst, len(want))
		}

		var singleQ8, singleBF, twoQ8, twoBF []float64
		var encErr error
		withAutoreleasePool(func() {
			qBuf := sharedBytes(q)
			kQ8, vQ8 := sharedBytes(kCodes), sharedBytes(vCodes)
			kScB, vScB := shared(kScales), shared(vScales)
			kBF, vBF := sharedBytes(kDeq), sharedBytes(vDeq)
			outQ8, outBF := scratchBF16(sh.nHeads*sh.headDim), scratchBF16(sh.nHeads*sh.headDim)

			q8PSO, err := sdpaVectorQ8Pipeline(sh.headDim)
			if err != nil {
				encErr = err
				return
			}
			bfPSO, err := sdpaVectorPipelineICBForHeadDim(sh.headDim)
			if err != nil {
				encErr = err
				return
			}
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			emitSDPAVectorQ8(encSink{enc}, q8PSO, qBuf, kQ8, vQ8, outQ8, kScB, vScB, 0, 0, nil,
				sh.nHeads, sh.nKVHeads, sh.n, khs, kss, khs, kss, scale)
			emitSDPA(encSink{enc}, bfPSO, qBuf, kBF, vBF, outBF, 0, nil,
				sh.nHeads, sh.nKVHeads, sh.n, khs, kss, khs, kss, scale)
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			singleQ8, singleBF = readOut(outQ8), readOut(outBF)

			// 2-pass pair: q8 pass 1 + the unchanged MLX pass 2, vs the bf16 pair
			const blocks = 32
			partQ8 := scratchBF16(sh.nHeads * blocks * sh.headDim)
			partBF := scratchBF16(sh.nHeads * blocks * sh.headDim)
			sumsQ8, maxsQ8 := scratchF32(sh.nHeads*blocks), scratchF32(sh.nHeads*blocks)
			sumsBF, maxsBF := scratchF32(sh.nHeads*blocks), scratchF32(sh.nHeads*blocks)
			out2Q8, out2BF := scratchBF16(sh.nHeads*sh.headDim), scratchBF16(sh.nHeads*sh.headDim)

			p1Q8, err := sdpaVector2Pass1Q8Pipeline(sh.headDim, blocks)
			if err != nil {
				encErr = err
				return
			}
			p1BF, err := sdpaVector2Pass1PipelineICB(sh.headDim, blocks)
			if err != nil {
				encErr = err
				return
			}
			p2, err := sdpaVector2Pass2PipelineICB(sh.headDim)
			if err != nil {
				encErr = err
				return
			}
			cb2 := commandBufferFast(queue)
			enc2 := computeCommandEncoderFast(cb2)
			emitSDPAVector2Pass1Q8(encSink{enc2}, p1Q8, qBuf, kQ8, vQ8, partQ8, sumsQ8, maxsQ8, kScB, vScB, 0, 0, nil,
				sh.nHeads, sh.nKVHeads, sh.n, blocks, khs, kss, khs, kss, scale)
			emitSDPA2Pass1NAt(encSink{enc2}, p1BF, qBuf, 0, kBF, vBF, partBF, sumsBF, maxsBF, 0, nil,
				1, sh.nHeads, sh.nKVHeads, sh.n, blocks, khs, kss, khs, kss, scale)
			memoryBarrierObject(enc2, metal.MTLBarrierScopeBuffers)
			emitSDPA2Pass2(encSink{enc2}, p2, partQ8, sumsQ8, maxsQ8, out2Q8, 1, sh.nHeads, blocks)
			emitSDPA2Pass2(encSink{enc2}, p2, partBF, sumsBF, maxsBF, out2BF, 1, sh.nHeads, blocks)
			endEncodingFast(enc2)
			commitCommandBufferFast(cb2)
			waitUntilCompletedFast(cb2)
			twoQ8, twoBF = readOut(out2Q8), readOut(out2BF)
		})
		if encErr != nil {
			t.Fatalf("%s: %v", sh.name, encErr)
		}
		compare("single-pass", singleQ8, singleBF)
		compare("2-pass", twoQ8, twoBF)
	}
}

// TestKVQ8StoreKernelMatchesHostQuantiser pins the GPU quantise-at-store
// kernel (lthn_kv_q8_store_bf16) byte-for-byte against the host quantiser the
// parity test and the reference dequant use — one math, two implementations,
// gated exactly.
func TestKVQ8StoreKernelMatchesHostQuantiser(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatalf("ensureInit: %v", err)
	}
	const kvDim = 512
	row := make([]byte, kvDim*2)
	for i := 0; i < kvDim; i++ {
		b := bf16FromF32(float32(math.Sin(float64(i)*0.53)) * (0.1 + float32(i%7)*0.9))
		row[i*2], row[i*2+1] = b[0], b[1]
	}
	wantCodes, wantScales := kvQ8QuantiseRows(row, kvDim)

	var gotCodes []byte
	var gotScales []float32
	var encErr error
	withAutoreleasePool(func() {
		rowBuf := sharedBytes(row)
		outBuf := device.NewBufferWithLengthOptions(uint(kvDim), metal.MTLResourceStorageModeShared)
		scBuf := device.NewBufferWithLengthOptions(uint(kvDim/kvQ8GroupSize*4), metal.MTLResourceStorageModeShared)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if encErr = encKVQ8Store(enc, rowBuf, outBuf, 0, scBuf, 0, kvDim); encErr != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		gotCodes = append([]byte(nil), unsafe.Slice((*byte)(outBuf.Contents()), kvDim)...)
		gotScales = append([]float32(nil), unsafe.Slice((*float32)(scBuf.Contents()), kvDim/kvQ8GroupSize)...)
	})
	if encErr != nil {
		t.Fatalf("encKVQ8Store: %v", encErr)
	}
	for i := range wantCodes {
		if gotCodes[i] != wantCodes[i] {
			t.Fatalf("code %d: kernel %d != host %d", i, int8(gotCodes[i]), int8(wantCodes[i]))
		}
	}
	for i := range wantScales {
		if gotScales[i] != wantScales[i] {
			t.Fatalf("scale %d: kernel %v != host %v", i, gotScales[i], wantScales[i])
		}
	}
	t.Logf("store kernel == host quantiser over %d codes + %d scales", len(wantCodes), len(wantScales))
}

// TestDiagQ8ReadKernelCost times the 2-pass q8 pair against the bf16 pair on a
// 32K-row kv=1 hd=512 cache (the 12B/e2b global shape at depth) — the slice-D
// item-1 instrument: is the q8 read kernel giving back the bandwidth saving?
func TestDiagQ8ReadKernelCost(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatalf("ensureInit: %v", err)
	}
	const nHeads, nKVHeads, headDim, n, blocks, iters = 16, 1, 512, 32768, 256, 50
	kvDim := nKVHeads * headDim
	scale := float32(1.0 / 16)

	kRows := make([]byte, n*kvDim*2)
	for i := range kRows {
		kRows[i] = byte((i*13 + 7) % 251)
	}
	kCodes, kScales := kvQ8QuantiseRows(kRows, kvDim)
	q := make([]byte, nHeads*headDim*2)
	for i := range q {
		q[i] = byte((i*31 + 5) % 250)
	}

	time2pass := func(q8 bool) float64 {
		var ms float64
		withAutoreleasePool(func() {
			qBuf := sharedBytes(q)
			outP := scratchBF16(nHeads * blocks * headDim)
			sums, maxs := scratchF32(nHeads*blocks), scratchF32(nHeads*blocks)
			out := scratchBF16(nHeads * headDim)
			var kB, vB, kSc, vSc metal.MTLBuffer
			if q8 {
				kB, vB = sharedBytes(kCodes), sharedBytes(kCodes)
				kSc, vSc = shared(kScales), shared(kScales)
			} else {
				kB, vB = sharedBytes(kRows), sharedBytes(kRows)
			}
			khs, kss := int64(headDim), int64(kvDim)
			var p1, p2 metal.MTLComputePipelineState
			var err error
			if q8 {
				p1, err = sdpaVector2Pass1Q8Pipeline(headDim, blocks)
			} else {
				p1, err = sdpaVector2Pass1PipelineICB(headDim, blocks)
			}
			if err != nil {
				t.Fatal(err)
			}
			if p2, err = sdpaVector2Pass2PipelineICB(headDim); err != nil {
				t.Fatal(err)
			}
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			for it := 0; it < iters; it++ {
				if q8 {
					emitSDPAVector2Pass1Q8(encSink{enc}, p1, qBuf, kB, vB, outP, sums, maxs, kSc, vSc, 0, 0, nil,
						nHeads, nKVHeads, n, blocks, khs, kss, khs, kss, scale)
				} else {
					emitSDPA2Pass1NAt(encSink{enc}, p1, qBuf, 0, kB, vB, outP, sums, maxs, 0, nil,
						1, nHeads, nKVHeads, n, blocks, khs, kss, khs, kss, scale)
				}
				memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
				emitSDPA2Pass2(encSink{enc}, p2, outP, sums, maxs, out, 1, nHeads, blocks)
				memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			}
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			ms = float64(cb.GPUEndTime()-cb.GPUStartTime()) * 1e3 / float64(iters)
		})
		return ms
	}
	// warm both once, then measure
	time2pass(false)
	time2pass(true)
	bf := time2pass(false)
	q8t := time2pass(true)
	bytesBF := float64(2*n*kvDim*2) / 1e9
	bytesQ8 := float64(2*n*kvDim+2*n*(kvDim/64)*4) / 1e9
	t.Logf("2-pass @N=%d kv=1 hd=512: bf16 %.3f ms (%.0f GB/s) vs q8 %.3f ms (%.0f GB/s) — q8/bf16 wall %.2fx (byte ratio 0.52x)",
		n, bf, bytesBF/(bf/1e3), q8t, bytesQ8/(q8t/1e3), q8t/bf)
}
