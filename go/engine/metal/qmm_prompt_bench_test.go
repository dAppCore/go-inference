// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"syscall"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// TestDiagQMMTvsSteelGatherAtPromptShape times the prompt fold's projection
// tier (the simple affine_qmm_t) against MLX's steel gather_qmm_rhs_nt driven
// with identity indices — SAME transposed weight layout, same math, steel
// simdgroup-mma tiles. The 2026-07-13 prefill hunt eliminated q8, dtype,
// launch count and chunk width; the fold's GEMMs run at ~19% ALU while
// mlx-lm hits 11.8K tok/s prefill on the identical snapshot. If the steel
// tier wins big here, the fix is a routing change (the MoE lane already
// ships this kernel), not new kernels.
func TestDiagQMMTvsSteelGatherAtPromptShape(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	requireNativeRuntime(t)
	const gs, bits = 64, 4
	const iters = 20
	shapes := []struct{ m, k, n int }{
		{512, 2048, 8192},  // reference fat shape
		{2048, 2048, 8192}, // 4-window chunk, fat
		{512, 1536, 6144},  // e2b TRUE gate/up at the default chunk
		{2048, 1536, 6144}, // e2b TRUE gate/up
		{4096, 1536, 6144}, // e2b TRUE gate/up at the 8-window chunk
		{2048, 6144, 1536}, // e2b TRUE down (skinny N)
		{2048, 1536, 2048}, // e2b TRUE q-proj (8×256)
	}
	for _, sh := range shapes {
		m, k, n := sh.m, sh.k, sh.n
		rng := rand.New(rand.NewPCG(42, 99))
		wq := device.NewBufferWithLengthOptions(uint(n*k/8*4), metal.MTLResourceStorageModeShared)
		scales := device.NewBufferWithLengthOptions(uint(n*(k/gs)*2), metal.MTLResourceStorageModeShared)
		biases := device.NewBufferWithLengthOptions(uint(n*(k/gs)*2), metal.MTLResourceStorageModeShared)
		x := device.NewBufferWithLengthOptions(uint(m*k*2), metal.MTLResourceStorageModeShared)
		outA := device.NewBufferWithLengthOptions(uint(m*n*2), metal.MTLResourceStorageModeShared)
		outB := device.NewBufferWithLengthOptions(uint(m*n*2), metal.MTLResourceStorageModeShared)
		idx := device.NewBufferWithLengthOptions(uint(m*4), metal.MTLResourceStorageModeShared)
		for _, b := range []metal.MTLBuffer{wq, scales, biases, x, outA, outB, idx} {
			if b == nil {
				t.Fatal("buffer allocation failed")
			}
		}
		wqs := unsafe.Slice((*uint32)(wq.Contents()), n*k/8)
		for i := range wqs {
			wqs[i] = rng.Uint32()
		}
		fill := func(buf metal.MTLBuffer, elems int, scale float32) {
			s := unsafe.Slice((*uint16)(buf.Contents()), elems)
			for i := range s {
				s[i] = f32ToBF16(rng.Float32()*scale - scale/2)
			}
		}
		fill(scales, n*(k/gs), 0.02)
		fill(biases, n*(k/gs), 0.02)
		fill(x, m*k, 2.0)
		clear(unsafe.Slice((*uint32)(idx.Contents()), m)) // identity: every row reads expert 0

		psoT, err := pipelineFor(qmmTKernelName(n, gs, bits))
		if err != nil {
			t.Fatalf("qmm_t pipeline: %v", err)
		}
		psoS, ok := moeGatherQMMRHSPipeline(gs, bits, m%moeGroupedBM == 0, n%moeGroupedBN == 0, k%moeGroupedBK == 0)
		if !ok {
			t.Fatal("steel gather_qmm_rhs pipeline unavailable")
		}

		runT := func(reps int) time.Duration {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			for range reps {
				emitQMMT(encSink{enc}, psoT, wq, 0, scales, 0, biases, 0, x, 0, outA, 0, m, n, k)
			}
			endEncodingFast(enc)
			start := time.Now()
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			return time.Since(start)
		}
		runS := func(reps int) time.Duration {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			for range reps {
				emitMoEGatherQMMRHS(encSink{enc}, psoS, x, 0, wq, 0, scales, 0, biases, 0, idx, outB, m, n, k)
			}
			endEncodingFast(enc)
			start := time.Now()
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			return time.Since(start)
		}
		runT(2) // warm both pipelines + buffers
		runS(2)
		dT := runT(iters)
		dS := runS(iters)

		// parity: same transposed layout, same affine dequant — outputs must agree
		// within accumulation-order noise.
		a := unsafe.Slice((*uint16)(outA.Contents()), m*n)
		b := unsafe.Slice((*uint16)(outB.Contents()), m*n)
		worst := 0.0
		for i := 0; i < m*n; i += 97 {
			av, bv := f64FromBF16Bits(a[i]), f64FromBF16Bits(b[i])
			if d := math.Abs(av - bv); d > worst {
				worst = d
			}
		}
		flops := 2.0 * float64(m) * float64(n) * float64(k)
		msT := dT.Seconds() * 1000 / iters
		msS := dS.Seconds() * 1000 / iters
		t.Logf("M=%d K=%d N=%d: qmm_t %.3fms (%.1f TFLOPS)  steel gather %.3fms (%.1f TFLOPS)  speedup %.2fx  worst|Δ|=%.4f",
			m, k, n, msT, flops/(msT/1000)/1e12, msS, flops/(msS/1000)/1e12, msT/msS, worst)
		if worst > 1.0 {
			t.Fatalf("steel gather diverges from qmm_t: worst |Δ| = %.4f", worst)
		}

		// mmap arm: the SAME qmm_t dispatch with the weight bound as a no-copy view
		// over file-backed pages — the fold binds shard-mmap views, the arms above
		// bind dedicated anonymous buffers. A rate drop here convicts the mmap
		// aperture as the fold's ~1.45x tax.
		wqBytes := unsafe.Slice((*byte)(wq.Contents()), n*k/8*4)
		dir := t.TempDir()
		fpath := dir + "/wq.bin"
		if err := os.WriteFile(fpath, wqBytes, 0o644); err != nil {
			t.Fatalf("write wq file: %v", err)
		}
		f, err := os.Open(fpath)
		if err != nil {
			t.Fatalf("open wq file: %v", err)
		}
		defer f.Close()
		mm, err := syscall.Mmap(int(f.Fd()), 0, len(wqBytes), syscall.PROT_READ, syscall.MAP_SHARED)
		if err != nil {
			t.Fatalf("mmap: %v", err)
		}
		defer syscall.Munmap(mm)
		var pinner runtime.Pinner
		pinner.Pin(&mm[0])
		defer pinner.Unpin()
		wqMM := newNoCopyBuffer(unsafe.Pointer(&mm[0]), uint(len(mm)))
		if wqMM == nil || wqMM.GetID() == 0 {
			t.Fatal("no-copy mmap buffer failed")
		}
		runMM := func(reps int) time.Duration {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			for range reps {
				emitQMMT(encSink{enc}, psoT, wqMM, 0, scales, 0, biases, 0, x, 0, outA, 0, m, n, k)
			}
			endEncodingFast(enc)
			start := time.Now()
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			return time.Since(start)
		}
		runMM(2)
		dMM := runMM(iters)
		msMM := dMM.Seconds() * 1000 / iters
		t.Logf("M=%d K=%d N=%d: qmm_t over FILE-MMAP weights %.3fms (%.1f TFLOPS)  vs dedicated %.3fms",
			m, k, n, msMM, flops/(msMM/1000)/1e12, msT)
	}
}

// f64FromBF16Bits widens one bf16 (as raw uint16 bits) for the parity check.
func f64FromBF16Bits(v uint16) float64 {
	return float64(math.Float32frombits(uint32(v) << 16))
}

// TestDiagMLPChainEncoderModes times the fold's REAL per-layer MLP chain
// (rms -> gate -> up -> gelu -> down) under the encoder/residency variables
// the black-box hunt could not separate: serial vs concurrent+barriers
// encoders, and one hot weight set vs cycling weight sets (the fold cycles
// 35 layers' weights; the standalone bench re-read ONE — SLC-flattered).
// Whichever arm reproduces the fold's ~14 TFLOPS convicts its mechanism.
func TestDiagMLPChainEncoderModes(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	requireNativeRuntime(t)
	const gs, bits = 64, 4
	const m, dModel, dFF = 512, 1536, 6144
	const layers, iters = 35, 3
	type wset struct{ gateW, upW, downW, gateS, upS, downS, gateB, upB, downB metal.MTLBuffer }
	mkW := func(rng *rand.Rand, n, k int) (metal.MTLBuffer, metal.MTLBuffer, metal.MTLBuffer) {
		wq := device.NewBufferWithLengthOptions(uint(n*k/8*4), metal.MTLResourceStorageModeShared)
		sc := device.NewBufferWithLengthOptions(uint(n*(k/gs)*2), metal.MTLResourceStorageModeShared)
		bi := device.NewBufferWithLengthOptions(uint(n*(k/gs)*2), metal.MTLResourceStorageModeShared)
		ws := unsafe.Slice((*uint32)(wq.Contents()), n*k/8)
		for i := range ws {
			ws[i] = rng.Uint32()
		}
		ss := unsafe.Slice((*uint16)(sc.Contents()), n*(k/gs))
		bs := unsafe.Slice((*uint16)(bi.Contents()), n*(k/gs))
		for i := range ss {
			ss[i] = f32ToBF16(rng.Float32()*0.02 - 0.01)
			bs[i] = f32ToBF16(rng.Float32()*0.02 - 0.01)
		}
		return wq, sc, bi
	}
	rng := rand.New(rand.NewPCG(7, 13))
	mkSet := func() wset {
		var w wset
		w.gateW, w.gateS, w.gateB = mkW(rng, dFF, dModel)
		w.upW, w.upS, w.upB = mkW(rng, dFF, dModel)
		w.downW, w.downS, w.downB = mkW(rng, dModel, dFF)
		return w
	}
	hot := mkSet()
	cyc := make([]wset, layers)
	for i := range cyc {
		cyc[i] = mkSet()
	}
	h := scratchBF16(m * dModel)
	normW := scratchBF16(dModel)
	fillBF := func(buf metal.MTLBuffer, n int) {
		s := unsafe.Slice((*uint16)(buf.Contents()), n)
		for i := range s {
			s[i] = f32ToBF16(rng.Float32() - 0.5)
		}
	}
	fillBF(h, m*dModel)
	fillBF(normW, dModel)
	normed := scratchBF16(m * dModel)
	gate := scratchBF16(m * dFF)
	up := scratchBF16(m * dFF)
	gated := scratchBF16(m * dFF)
	down := scratchBF16(m * dModel)
	psoGate, err := pipelineFor(qmmTKernelName(dFF, gs, bits))
	if err != nil {
		t.Fatalf("gate pso: %v", err)
	}
	psoDown, err := pipelineFor(qmmTKernelName(dModel, gs, bits))
	if err != nil {
		t.Fatalf("down pso: %v", err)
	}
	_ = psoGate
	_ = psoDown

	run := func(conc, groupGateUp, cycle bool) time.Duration {
		cb := commandBufferFast(queue)
		var enc metal.MTLComputeCommandEncoderObject
		if conc {
			enc = concurrentComputeEncoderFast(cb)
		} else {
			enc = computeCommandEncoderFast(cb)
		}
		bar := func() {
			if conc {
				memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			}
		}
		for it := 0; it < iters; it++ {
			for li := 0; li < layers; li++ {
				w := hot
				if cycle {
					w = cyc[li]
				}
				if err := encRMSNormRowsBF16(enc, h, normW, normed, 0, 0, 0, m, dModel, 1e-6); err != nil {
					t.Fatalf("rms: %v", err)
				}
				bar()
				if err := encQMMTBF16At(enc, w.gateW, w.gateS, w.gateB, normed, gate, 0, 0, 0, 0, 0, m, dFF, dModel, gs, bits); err != nil {
					t.Fatalf("gate: %v", err)
				}
				if !groupGateUp {
					bar()
				}
				if err := encQMMTBF16At(enc, w.upW, w.upS, w.upB, normed, up, 0, 0, 0, 0, 0, m, dFF, dModel, gs, bits); err != nil {
					t.Fatalf("up: %v", err)
				}
				bar()
				if err := encGeluGateMulFused(enc, gate, up, gated, m*dFF); err != nil {
					t.Fatalf("gelu: %v", err)
				}
				bar()
				if err := encQMMTBF16At(enc, w.downW, w.downS, w.downB, gated, down, 0, 0, 0, 0, 0, m, dModel, dFF, gs, bits); err != nil {
					t.Fatalf("down: %v", err)
				}
				bar()
			}
		}
		endEncodingFast(enc)
		start := time.Now()
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		return time.Since(start)
	}
	run(false, false, false) // warm PSOs + buffers
	flops := 2.0 * float64(m) * float64(layers) * (2*float64(dModel)*float64(dFF) + float64(dFF)*float64(dModel)) * float64(iters)
	for _, arm := range []struct {
		name                     string
		conc, groupGateUp, cycle bool
	}{
		{"serial hot-weights", false, false, false},
		{"serial cycling-weights", false, false, true},
		{"conc+barriers hot", true, false, false},
		{"conc+barriers cycling", true, false, true},
		{"conc gate||up cycling", true, true, true},
	} {
		d := run(arm.conc, arm.groupGateUp, arm.cycle)
		ms := d.Seconds() * 1000
		t.Logf("%-24s %8.2fms  %.1f TFLOPS", arm.name, ms, flops/d.Seconds()/1e12)
	}
}
