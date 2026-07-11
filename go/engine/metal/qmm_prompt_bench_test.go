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

// TestDiagMLPChainWithSDPASlabTraffic adds the one ingredient the real fold
// has that TestDiagMLPChainEncoderModes lacked: SDPA-GEMM-scale S-slab
// streaming between layers (sdpaPromptS pairs run to hundreds of MB and are
// written+read once per global layer). If interleaving that traffic drops
// the chain from ~19 TFLOPS to the fold's ~13-14, the SLC-poisoning
// mechanism is convicted and the prefill fix is S-traffic shaping
// (flash-style streaming softmax / tighter S tiles), not encoder work.
func TestDiagMLPChainWithSDPASlabTraffic(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	requireNativeRuntime(t)
	const gs, bits = 64, 4
	const m, dModel, dFF = 512, 1536, 6144
	const layers, iters = 35, 3
	rng := rand.New(rand.NewPCG(21, 34))
	mkW := func(n, k int) (metal.MTLBuffer, metal.MTLBuffer, metal.MTLBuffer) {
		wq := device.NewBufferWithLengthOptions(uint(n*k/8*4), metal.MTLResourceStorageModeShared)
		sc := device.NewBufferWithLengthOptions(uint(n*(k/gs)*2), metal.MTLResourceStorageModeShared)
		bi := device.NewBufferWithLengthOptions(uint(n*(k/gs)*2), metal.MTLResourceStorageModeShared)
		ws := unsafe.Slice((*uint32)(wq.Contents()), n*k/8)
		for i := range ws {
			ws[i] = rng.Uint32()
		}
		return wq, sc, bi
	}
	gateW, gateS, gateB := mkW(dFF, dModel)
	upW, upS, upB := mkW(dFF, dModel)
	downW, downS, downB := mkW(dModel, dFF)
	h := scratchBF16(m * dModel)
	normW := scratchBF16(dModel)
	normed := scratchBF16(m * dModel)
	gate := scratchBF16(m * dFF)
	up := scratchBF16(m * dFF)
	gated := scratchBF16(m * dFF)
	down := scratchBF16(m * dModel)
	// S-slab stand-ins at the real scale: the 8K-prompt sdpaS pair holds
	// headBatch*K x (basePos+K) bf16 — ~64MB per slab mid-prompt. Stream one
	// slab-to-slab copy per layer, exactly one write+read of 64MB.
	const sElems = 32 << 20 // 64MB bf16
	s0 := scratchBF16(sElems)
	s1 := scratchBF16(sElems)
	run := func(withS bool) time.Duration {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		for it := 0; it < iters; it++ {
			for li := 0; li < layers; li++ {
				if err := encRMSNormRowsBF16(enc, h, normW, normed, 0, 0, 0, m, dModel, 1e-6); err != nil {
					t.Fatalf("rms: %v", err)
				}
				if withS { // the SDPA seat: S-scale streaming between norm and the gemms
					if err := encCopyBF16Contig(enc, s0, s1, 0, 0, sElems); err != nil {
						t.Fatalf("scopy: %v", err)
					}
				}
				if err := encQMMTBF16At(enc, gateW, gateS, gateB, normed, gate, 0, 0, 0, 0, 0, m, dFF, dModel, gs, bits); err != nil {
					t.Fatalf("gate: %v", err)
				}
				if err := encQMMTBF16At(enc, upW, upS, upB, normed, up, 0, 0, 0, 0, 0, m, dFF, dModel, gs, bits); err != nil {
					t.Fatalf("up: %v", err)
				}
				if err := encGeluGateMulFused(enc, gate, up, gated, m*dFF); err != nil {
					t.Fatalf("gelu: %v", err)
				}
				if err := encQMMTBF16At(enc, downW, downS, downB, gated, down, 0, 0, 0, 0, 0, m, dModel, dFF, gs, bits); err != nil {
					t.Fatalf("down: %v", err)
				}
			}
		}
		endEncodingFast(enc)
		start := time.Now()
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		return time.Since(start)
	}
	run(false) // warm
	gemmFlops := 2.0 * float64(m) * float64(layers) * 3 * float64(dModel) * float64(dFF) * float64(iters)
	dPlain := run(false)
	dS := run(true)
	sBytes := float64(layers*iters) * float64(sElems) * 2 * 2 // write+read per layer
	sSeconds := sBytes / 700e9                                // ~DRAM-rate cost of the copies themselves
	t.Logf("plain chain      %8.2fms  %.1f TFLOPS", dPlain.Seconds()*1000, gemmFlops/dPlain.Seconds()/1e12)
	t.Logf("with S-traffic   %8.2fms  gemm-effective %.1f TFLOPS after subtracting ~%.0fms copy cost",
		dS.Seconds()*1000, gemmFlops/(dS.Seconds()-sSeconds)/1e12, sSeconds*1000)
}

// TestDiagQMMTPipelinedRotatingOutputs measures qmm_t's TRUE pipelined rate at
// the e2b gate shape: reps encoded in ONE command buffer as in the diag above,
// but each rep writes a DIFFERENT output buffer — no WAW hazard chain, so the
// GPU may overlap dispatches exactly as a real forward with per-op outputs
// (mlx's model) does. The delta against the shared-output run quantifies how
// much of the in-situ prefill GEMM tax is hazard serialisation on reused
// scratch slabs (#381).
func TestDiagQMMTPipelinedRotatingOutputs(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	requireNativeRuntime(t)
	const gs, bits = 64, 4
	const m, k, n = 2048, 1536, 6144 // e2b TRUE gate/up at the default chunk
	const iters = 32
	const rot = 8
	rng := rand.New(rand.NewPCG(42, 99))
	wq := device.NewBufferWithLengthOptions(uint(n*k/8*4), metal.MTLResourceStorageModeShared)
	scales := device.NewBufferWithLengthOptions(uint(n*(k/gs)*2), metal.MTLResourceStorageModeShared)
	biases := device.NewBufferWithLengthOptions(uint(n*(k/gs)*2), metal.MTLResourceStorageModeShared)
	x := device.NewBufferWithLengthOptions(uint(m*k*2), metal.MTLResourceStorageModeShared)
	outs := make([]metal.MTLBuffer, rot)
	for i := range outs {
		outs[i] = device.NewBufferWithLengthOptions(uint(m*n*2), metal.MTLResourceStorageModeShared)
		if outs[i] == nil {
			t.Fatal("out buffer allocation failed")
		}
	}
	for _, b := range []metal.MTLBuffer{wq, scales, biases, x} {
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

	psoT, err := pipelineFor(qmmTKernelName(n, gs, bits))
	if err != nil {
		t.Fatalf("qmm_t pipeline: %v", err)
	}
	run := func(rotate bool, reps int) time.Duration {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		for r := range reps {
			out := outs[0]
			if rotate {
				out = outs[r%rot]
			}
			emitQMMT(encSink{enc}, psoT, wq, 0, scales, 0, biases, 0, x, 0, out, 0, m, n, k)
		}
		endEncodingFast(enc)
		start := time.Now()
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		return time.Since(start)
	}
	run(true, 2)
	run(false, 2)
	flops := float64(2*m) * float64(k) * float64(n) * float64(iters)
	dShared := run(false, iters)
	dRot := run(true, iters)
	t.Logf("qmm_t %dx%dx%d ×%d: shared-out %.1fms = %.1f TFLOPS · rotating-out %.1fms = %.1f TFLOPS (%.2fx)",
		m, k, n, iters,
		float64(dShared.Microseconds())/1000, flops/dShared.Seconds()/1e12,
		float64(dRot.Microseconds())/1000, flops/dRot.Seconds()/1e12,
		dShared.Seconds()/dRot.Seconds())
}

// TestDiagQMMTMlxWorkloadReplay prices the EXACT qmm workload a spied mlx-lm
// 8K-prompt run dispatched (shape × count table captured 2026-07-15, 32.98
// TFLOP total, their wall 0.74s) on OUR qmm_t pipelined dispatches — the
// apples-to-apples that says whether the prefill gap is per-op kernel
// efficiency (this sum >> theirs) or work outside the GEMMs (#381).
func TestDiagQMMTMlxWorkloadReplay(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	requireNativeRuntime(t)
	const gs, bits = 64, 4
	shapes := []struct {
		m, k, n, count int
	}{
		{2048, 1536, 12288, 120}, {2048, 1536, 262144, 3}, {2048, 12288, 1536, 60},
		{2048, 1536, 6144, 90}, {2048, 6144, 1536, 45}, {1087, 1536, 12288, 40},
		{2048, 1536, 2048, 84}, {2048, 2048, 1536, 84}, {1087, 1536, 262144, 1},
		{1087, 12288, 1536, 20}, {1087, 1536, 6144, 30}, {2048, 1536, 4096, 21},
		{2048, 4096, 1536, 21}, {1087, 6144, 1536, 15}, {2048, 1536, 256, 177},
		{1087, 1536, 2048, 28}, {1087, 2048, 1536, 28}, {2048, 1536, 8960, 3},
		{2048, 256, 1536, 105}, {1087, 1536, 4096, 7}, {1087, 4096, 1536, 7},
		{2048, 1536, 512, 18}, {1087, 1536, 256, 59},
	}
	var totalTime time.Duration
	var totalFLOPs float64
	for _, sh := range shapes {
		m, k, n := sh.m, sh.k, sh.n
		wq := device.NewBufferWithLengthOptions(uint(n*k/8*4), metal.MTLResourceStorageModeShared)
		scales := device.NewBufferWithLengthOptions(uint(n*(k/gs)*2), metal.MTLResourceStorageModeShared)
		biases := device.NewBufferWithLengthOptions(uint(n*(k/gs)*2), metal.MTLResourceStorageModeShared)
		x := device.NewBufferWithLengthOptions(uint(m*k*2), metal.MTLResourceStorageModeShared)
		out := device.NewBufferWithLengthOptions(uint(m*n*2), metal.MTLResourceStorageModeShared)
		if wq == nil || scales == nil || biases == nil || x == nil || out == nil {
			t.Fatalf("alloc failed at %dx%dx%d", m, k, n)
		}
		pso, err := pipelineFor(qmmTKernelName(n, gs, bits))
		if err != nil {
			t.Fatalf("pipeline %dx%dx%d: %v", m, k, n, err)
		}
		run := func(reps int) time.Duration {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			for range reps {
				emitQMMT(encSink{enc}, pso, wq, 0, scales, 0, biases, 0, x, 0, out, 0, m, n, k)
			}
			endEncodingFast(enc)
			start := time.Now()
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			return time.Since(start)
		}
		run(2)
		reps := min(sh.count, 24)
		d := run(reps)
		per := d / time.Duration(reps)
		shapeTotal := per * time.Duration(sh.count)
		fl := 2 * float64(m) * float64(k) * float64(n)
		totalTime += shapeTotal
		totalFLOPs += fl * float64(sh.count)
		t.Logf("%5dx%5dx%6d x%3d: %7.2fms total (%5.1f TFLOPS)",
			m, k, n, sh.count, float64(shapeTotal.Microseconds())/1000, fl/per.Seconds()/1e12)
		releaseDeviceBuffers(wq, scales, biases, x, out)
	}
	t.Logf("OUR price for mlx's workload: %.2fs (%.1f TFLOPS overall) vs their 0.74s wall — %.2fx",
		totalTime.Seconds(), totalFLOPs/totalTime.Seconds()/1e12, totalTime.Seconds()/0.74)
}

// TestDiagQMMTFloat16VsBfloat16 A/Bs the SAME qmm_t kernel at the e2b gate
// shape with float16 vs bfloat16 activations — pre-M4 Apple GPUs emulate
// bf16 (fp32 convert in-shader) while fp16 runs native ALU rate (#381).
func TestDiagQMMTFloat16VsBfloat16(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	requireNativeRuntime(t)
	const gs, bits = 64, 4
	const m, k, n = 2048, 1536, 12288
	const iters = 24
	wq := device.NewBufferWithLengthOptions(uint(n*k/8*4), metal.MTLResourceStorageModeShared)
	scales := device.NewBufferWithLengthOptions(uint(n*(k/gs)*2), metal.MTLResourceStorageModeShared)
	biases := device.NewBufferWithLengthOptions(uint(n*(k/gs)*2), metal.MTLResourceStorageModeShared)
	x := device.NewBufferWithLengthOptions(uint(m*k*2), metal.MTLResourceStorageModeShared)
	out := device.NewBufferWithLengthOptions(uint(m*n*2), metal.MTLResourceStorageModeShared)
	if wq == nil || scales == nil || biases == nil || x == nil || out == nil {
		t.Fatal("alloc failed")
	}
	run := func(kname string) float64 {
		pso, err := pipelineFor(kname)
		if err != nil {
			t.Fatalf("pipeline %s: %v", kname, err)
		}
		bench := func(reps int) time.Duration {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			for range reps {
				emitQMMT(encSink{enc}, pso, wq, 0, scales, 0, biases, 0, x, 0, out, 0, m, n, k)
			}
			endEncodingFast(enc)
			start := time.Now()
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			return time.Since(start)
		}
		bench(2)
		d := bench(iters)
		return 2 * float64(m) * float64(k) * float64(n) * float64(iters) / d.Seconds() / 1e12
	}
	bf := run("affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0")
	fp := run("affine_qmm_t_float16_t_gs_64_b_4_alN_true_batch_0")
	t.Logf("qmm_t %dx%dx%d: bfloat16 %.1f TFLOPS · float16 %.1f TFLOPS (%.2fx)", m, k, n, bf, fp, fp/bf)
	if os.Getenv("LTHN_DIAG_SUSTAINED") != "" {
		pso, err := pipelineFor("affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0")
		if err != nil {
			t.Fatal(err)
		}
		for _, reps := range []int{24, 96, 384} {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			for range reps {
				emitQMMT(encSink{enc}, pso, wq, 0, scales, 0, biases, 0, x, 0, out, 0, m, n, k)
			}
			endEncodingFast(enc)
			start := time.Now()
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			d := time.Since(start)
			t.Logf("sustained x%d: %.0fms = %.1f TFLOPS", reps, float64(d.Microseconds())/1000,
				2*float64(m)*float64(k)*float64(n)*float64(reps)/d.Seconds()/1e12)
		}
	}
}
