// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
)

// hopTaxTime encodes via fn into one command buffer, then times ONLY
// commit→completion (encode cost excluded — the campaign target is the
// GPU-side chain, the host encode is already hidden by submit-ahead).
// Three reps, minimum taken.
func hopTaxTime(t *testing.T, fn func(cb metal.MTLCommandBufferObject)) time.Duration {
	t.Helper()
	best := time.Duration(0)
	for rep := 0; rep < 3; rep++ {
		var d time.Duration
		withAutoreleasePool(func() {
			cb := commandBufferFast(queue)
			fn(cb)
			t0 := time.Now()
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			d = time.Since(t0)
		})
		if rep == 0 || d < best {
			best = d
		}
	}
	return best
}

// TestHopTaxMicrobench measures the per-dispatch constants the #341 campaign
// designs against, on this hardware, with the decode's real kernels:
//
//	dependent hop   — N chained dispatches on a serial encoder (the decode's shape)
//	independent     — the same N with no data dependencies, serial vs concurrent
//	                  encoder (does hazard tracking alone serialise? what is the
//	                  parallel ceiling per dispatch?)
//	barrier hop     — N chained dispatches on a concurrent encoder with an
//	                  explicit buffer-scope barrier per hop (the concurrent
//	                  passes' idiom)
//	seam            — one encoder PER dispatch (the profiler/pass boundary tax)
//	work scaling    — a big qmv (2816×2816, ~350 TGs) vs a tiny qmv (128×128
//	                  router-shaped, ~16 TGs): if the dependent hop cost is
//	                  similar despite ~500× less work, hops are DRAIN-bound and
//	                  phase 2 (split-K widening) is pointless; if the tiny hop is
//	                  much cheaper, hops scale with occupancy.
//
// It asserts nothing beyond sanity — it is an instrument; the numbers go into
// the #341 task. Run with -v.
func TestHopTaxMicrobench(t *testing.T) {
	requireNativeRuntime(t)
	const dModel = 2816
	const n = 512
	eps := float32(1e-6)

	rmsPSO, err := pipelineFor(rmsKernelBF16(dModel))
	if err != nil {
		t.Fatalf("rms pipeline: %v", err)
	}
	rmsTG := rmsThreadgroup(dModel, rmsPSO)

	w := toBF16Bytes(syntheticFloat32(dModel, 7))
	wBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&w[0]), uint(len(w)), metal.MTLResourceStorageModeShared)
	ping := device.NewBufferWithLengthOptions(uint(dModel*bf16Size), metal.MTLResourceStorageModeShared)
	pong := device.NewBufferWithLengthOptions(uint(dModel*bf16Size), metal.MTLResourceStorageModeShared)

	// disjoint buffer pairs for the independent cases
	indepIn := make([]metal.MTLBuffer, n)
	indepOut := make([]metal.MTLBuffer, n)
	for i := range n {
		indepIn[i] = device.NewBufferWithLengthOptions(uint(dModel*bf16Size), metal.MTLResourceStorageModeShared)
		indepOut[i] = device.NewBufferWithLengthOptions(uint(dModel*bf16Size), metal.MTLResourceStorageModeShared)
	}

	empty := hopTaxTime(t, func(cb metal.MTLCommandBufferObject) {})

	depSerial := hopTaxTime(t, func(cb metal.MTLCommandBufferObject) {
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		a, b := ping, pong
		for i := 0; i < n; i++ {
			emitRMSNorm(sink, rmsPSO, a, wBuf, b, 0, dModel, eps, rmsTG)
			a, b = b, a
		}
		endEncodingFast(enc)
	})

	indepSerial := hopTaxTime(t, func(cb metal.MTLCommandBufferObject) {
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		for i := 0; i < n; i++ {
			emitRMSNorm(sink, rmsPSO, indepIn[i], wBuf, indepOut[i], 0, dModel, eps, rmsTG)
		}
		endEncodingFast(enc)
	})

	indepConcurrent := hopTaxTime(t, func(cb metal.MTLCommandBufferObject) {
		enc := concurrentComputeEncoderFast(cb)
		sink := encSink{metal.MTLComputeCommandEncoder(enc)}
		for i := 0; i < n; i++ {
			emitRMSNorm(sink, rmsPSO, indepIn[i], wBuf, indepOut[i], 0, dModel, eps, rmsTG)
		}
		endEncodingFast(enc)
	})

	barrierHop := hopTaxTime(t, func(cb metal.MTLCommandBufferObject) {
		enc := concurrentComputeEncoderFast(cb)
		sink := encSink{metal.MTLComputeCommandEncoder(enc)}
		a, b := ping, pong
		for i := 0; i < n; i++ {
			emitRMSNorm(sink, rmsPSO, a, wBuf, b, 0, dModel, eps, rmsTG)
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			a, b = b, a
		}
		endEncodingFast(enc)
	})

	const nSeam = 128 // one encoder per dispatch is expensive — keep the case short
	seam := hopTaxTime(t, func(cb metal.MTLCommandBufferObject) {
		a, b := ping, pong
		for i := 0; i < nSeam; i++ {
			enc := computeCommandEncoderFast(cb)
			emitRMSNorm(encSink{enc}, rmsPSO, a, wBuf, b, 0, dModel, eps, rmsTG)
			endEncodingFast(enc)
			a, b = b, a
		}
	})

	// qmv work-scaling pair: big (dModel square) vs tiny (router-shaped square).
	qmvCase := func(dim int) (dep, indep time.Duration, ok bool) {
		const gs, bits = 64, 4
		pso, err := pipelineFor(qmvBF16KernelName(dim, dim, gs, bits))
		if err != nil {
			return 0, 0, false
		}
		packed := make([]byte, dim*dim*bits/8)
		for i := range packed {
			packed[i] = byte((i*131 + 17) % 256)
		}
		nSB := dim * (dim / gs)
		scales := toBF16Bytes(syntheticFloat32(nSB, 11))
		biases := toBF16Bytes(syntheticFloat32(nSB, 13))
		wq := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&packed[0]), uint(len(packed)), metal.MTLResourceStorageModeShared)
		sc := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&scales[0]), uint(len(scales)), metal.MTLResourceStorageModeShared)
		bi := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&biases[0]), uint(len(biases)), metal.MTLResourceStorageModeShared)
		qa := device.NewBufferWithLengthOptions(uint(dim*bf16Size), metal.MTLResourceStorageModeShared)
		qb := device.NewBufferWithLengthOptions(uint(dim*bf16Size), metal.MTLResourceStorageModeShared)
		outs := device.NewBufferWithLengthOptions(uint(n*dim*bf16Size), metal.MTLResourceStorageModeShared)

		dep = hopTaxTime(t, func(cb metal.MTLCommandBufferObject) {
			enc := computeCommandEncoderFast(cb)
			sink := encSink{enc}
			a, b := qa, qb
			for i := 0; i < n; i++ {
				emitQMV(sink, pso, wq, 0, sc, 0, bi, 0, a, b, 0, dim, dim)
				a, b = b, a
			}
			endEncodingFast(enc)
		})
		indep = hopTaxTime(t, func(cb metal.MTLCommandBufferObject) {
			enc := concurrentComputeEncoderFast(cb)
			sink := encSink{metal.MTLComputeCommandEncoder(enc)}
			for i := 0; i < n; i++ {
				emitQMV(sink, pso, wq, 0, sc, 0, bi, 0, qa, outs, uint(i*dim*bf16Size), dim, dim)
			}
			endEncodingFast(enc)
		})
		return dep, indep, true
	}

	per := func(d time.Duration, count int) float64 {
		net := d - empty
		if net < 0 {
			net = 0
		}
		return float64(net.Nanoseconds()) / 1e3 / float64(count)
	}

	t.Logf("empty cb commit+wait: %v", empty)
	t.Logf("rms %d-wide, n=%d:", dModel, n)
	t.Logf("  dependent hop, serial encoder:      %7.2f µs/dispatch", per(depSerial, n))
	t.Logf("  independent, serial encoder:        %7.2f µs/dispatch  (>> concurrent ⇒ hazard tracking serialises)", per(indepSerial, n))
	t.Logf("  independent, concurrent encoder:    %7.2f µs/dispatch  (the parallel ceiling)", per(indepConcurrent, n))
	t.Logf("  dependent + barrier, concurrent:    %7.2f µs/dispatch  (the concurrent passes' hop)", per(barrierHop, n))
	t.Logf("  one encoder per dispatch (seam):    %7.2f µs/dispatch  (n=%d)", per(seam, nSeam), nSeam)

	if depBig, indepBig, ok := qmvCase(2816); ok {
		t.Logf("qmv 2816x2816 (~350 TGs): dependent %7.2f µs/hop, independent-concurrent %7.2f µs/dispatch", per(depBig, n), per(indepBig, n))
	}
	if depTiny, indepTiny, ok := qmvCase(128); ok {
		t.Logf("qmv 128x128 (router-shaped, 16 TGs): dependent %7.2f µs/hop, independent-concurrent %7.2f µs/dispatch", per(depTiny, n), per(indepTiny, n))
		t.Logf("  → tiny-vs-big dependent hop ratio decides phase 2: ≈1 ⇒ DRAIN-bound (skip split-K); ≪1 ⇒ occupancy-bound")
	}

	if depSerial <= empty {
		t.Fatal("dependent chain measured no GPU time — the bench is broken")
	}
}

// TestHopTaxICBMicrobench is the phase-3 go/no-go instrument (#341): ICB
// commands within one executeCommandsInBuffer are CONCURRENT-typed (no
// ordering inside an execute), so a dependent per-layer chain recorded as an
// ICB must issue one execute per stage — the decisive constant is therefore
// the cost of a DEPENDENT execute on a serial encoder, against the carried
// concurrent encoder's 4.13µs barrier hop (the shape the decode already runs).
//
// Pre-registered rule: dependent-execute ≥ the barrier hop ⇒ a stage-ICB MoE
// decode buys nothing GPU-side (and submit-ahead already hides the host
// encode) ⇒ phase 3 closes by rule. Only a dramatically cheaper execute
// (< ~2µs) earns the build.
func TestHopTaxICBMicrobench(t *testing.T) {
	requireNativeRuntime(t)
	const dModel = 2816
	const n = 512
	eps := float32(1e-6)

	rmsPSO, err := pipelineForICB(rmsKernelBF16(dModel))
	if err != nil {
		t.Skipf("ICB-capable rms pipeline unavailable: %v", err)
	}
	rmsTG := rmsThreadgroup(dModel, rmsPSO)

	w := toBF16Bytes(syntheticFloat32(dModel, 7))
	wBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&w[0]), uint(len(w)), metal.MTLResourceStorageModeShared)
	ping := device.NewBufferWithLengthOptions(uint(dModel*bf16Size), metal.MTLResourceStorageModeShared)
	pong := device.NewBufferWithLengthOptions(uint(dModel*bf16Size), metal.MTLResourceStorageModeShared)
	indepOut := device.NewBufferWithLengthOptions(uint(n*dModel*bf16Size), metal.MTLResourceStorageModeShared)

	icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
	icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
	icbDesc.SetInheritBuffers(false)
	icbDesc.SetInheritPipelineState(false)
	icbDesc.SetMaxKernelBufferBindCount(8)

	// dependent chain: command i reads the previous command's output (ping↔pong),
	// replayed as n SEPARATE executes so the serial encoder orders them.
	depICB := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, uint(n), metal.MTLResourceStorageModeShared)
	a, b := ping, pong
	for i := 0; i < n; i++ {
		cmd := indirectComputeCommandAtIndexFast(depICB, uint(i))
		emitRMSNorm(fastICBSink{cmd}, rmsPSO, a, wBuf, b, 0, dModel, eps, rmsTG)
		a, b = b, a
	}
	// independent: n commands with disjoint outputs, ONE execute (the ICB's
	// internal concurrency ceiling).
	indepICB := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, uint(n), metal.MTLResourceStorageModeShared)
	for i := 0; i < n; i++ {
		cmd := indirectComputeCommandAtIndexFast(indepICB, uint(i))
		emitRMSNormAt(fastICBSink{cmd}, rmsPSO, ping, wBuf, indepOut, 0, 0, uint(i*dModel*bf16Size), dModel, eps, rmsTG)
	}

	resident := []metal.MTLResource{wBuf, ping, pong, indepOut}
	residentIDs := resourceIDsForFastUse(nil, resident)
	empty := hopTaxTime(t, func(cb metal.MTLCommandBufferObject) {})

	depExec := hopTaxTime(t, func(cb metal.MTLCommandBufferObject) {
		enc := computeCommandEncoderFast(cb)
		useResourcesIDsFast(enc, resident, residentIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
		for i := 0; i < n; i++ {
			executeCommandsInBufferWithRangeFast(enc, depICB, foundation.NSRange{Location: uint(i), Length: 1})
		}
		endEncodingFast(enc)
	})
	oneExec := hopTaxTime(t, func(cb metal.MTLCommandBufferObject) {
		enc := computeCommandEncoderFast(cb)
		useResourcesIDsFast(enc, resident, residentIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
		executeCommandsInBufferWithRangeFast(enc, indepICB, foundation.NSRange{Location: 0, Length: uint(n)})
		endEncodingFast(enc)
	})

	per := func(d time.Duration, count int) float64 {
		net := d - empty
		if net < 0 {
			net = 0
		}
		return float64(net.Nanoseconds()) / 1e3 / float64(count)
	}
	t.Logf("ICB rms %d-wide, n=%d:", dModel, n)
	t.Logf("  dependent, one execute per hop (serial enc): %7.2f µs/hop   — compare the 4.13µs barrier hop", per(depExec, n))
	t.Logf("  independent, ONE execute of %d commands:     %7.2f µs/dispatch — the ICB concurrency ceiling", n, per(oneExec, n))
}
