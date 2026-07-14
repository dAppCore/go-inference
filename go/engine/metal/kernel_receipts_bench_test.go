// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"github.com/tmc/apple/metal"
)

// kernel_receipts_bench_test.go — per-kernel perf receipts (#393 slice 2). Each benchmark
// dispatches ONE kernel back-to-back (barriered, so dispatches serialise) in its own
// command buffer at the REAL shapes the engine dispatches, then reports the GPU span per
// dispatch and the achieved GB/s. This is the in-tree half of the kernel instrument (the
// .gputrace capture is the in-situ half): it answers "what does this kernel achieve
// running alone" without opening Xcode.
//
// FIRST RECEIPTS (2026-07-14, M3 Ultra): the fat bf16 head gemv measured 792 GB/s — THAT
// is the box's real roofline (the ~650 napkin was low). QMV at the fat attention shape:
// 567. The THIN MoE expert down (704→2816): 178 — a quarter of the fat qmv's rate, the
// kernel-level confirmation of #392's occupancy suspicion. SDPA vector @ kv 4096: 326.
// RMSNorm 2816: ~3.8µs/op — the serialisation floor every thin barriered op pays (the
// round is ~700 of them). K=1 decode streams ~2.2GB/9.3ms ≈ 237 GB/s ≈ 30% of the
// MEASURED roofline: the gap is thin-dispatch occupancy + the per-op floor, now with
// numbers attached.
//
//	MLX_METALLIB_PATH=…/mlx.metallib go test -tags metal_runtime -run '^$' \
//	  -bench BenchmarkKernelReceipt -benchtime 1x ./engine/metal/
//
// GPU time comes from the command buffer's own GPUStart/EndTime (whole-cb span over N
// serialised dispatches ÷ N) — deliberately NOT MTLCounterSampleBuffer: the cb span needs
// no counter resolution or CPU/GPU timestamp calibration, and for an isolated single-kernel
// cb it measures the same thing. Counter sampling stays reserved for per-dispatch splits
// inside MIXED production rounds (a later slice, likely via the dappcore/apple fork).

// benchKernelReceipt encodes `iters` barriered dispatches via encode into one command
// buffer, waits, and reports gpu-ns/op and GB/s from bytesPerDispatch.
func benchKernelReceipt(b *testing.B, bytesPerDispatch int64, iters int, encode func(enc metal.MTLComputeCommandEncoderObject) error) {
	b.Helper()
	requireNativeRuntime(b)
	for n := 0; n < b.N; n++ {
		withAutoreleasePool(func() {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			for i := 0; i < iters; i++ {
				if err := encode(enc); err != nil {
					b.Fatalf("encode: %v", err)
				}
				memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			}
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			spanNs := (cb.GPUEndTime() - cb.GPUStartTime()) * 1e9
			perDispatchNs := spanNs / float64(iters)
			b.ReportMetric(perDispatchNs, "gpu-ns/op")
			b.ReportMetric(float64(bytesPerDispatch)/perDispatchNs, "GB/s")
		})
	}
}

// sharedBuf allocates a zero-filled shared device buffer (values are irrelevant to
// timing; zeros keep bf16 arithmetic NaN-free).
func sharedBuf(n int) metal.MTLBuffer {
	return device.NewBufferWithLengthOptions(uint(n), metal.MTLResourceStorageModeShared)
}

// qmvWeightBytes is the 4-bit affine layout's bytes for one [outDim, inDim] weight:
// packed nibbles + bf16 scales and biases per group.
func qmvWeightBytes(outDim, inDim, groupSize int) (wq, scales, biases int64) {
	wq = int64(outDim) * int64(inDim) / 2
	groups := int64(outDim) * int64(inDim/groupSize)
	return wq, groups * 2, groups * 2
}

// BenchmarkKernelReceipt_QMV_Attn26B: the 4-bit affine qmv at the 26B's fattest dense
// attention shape (Q projection, dModel 2816 → qDim 4096, gs 64) — the weight-stream
// workhorse of every decode round.
func BenchmarkKernelReceipt_QMV_Attn26B(b *testing.B) {
	const inDim, outDim, gs, bits = 2816, 4096, 64, 4
	wqB, scB, biB := qmvWeightBytes(outDim, inDim, gs)
	wq, sc, bi := sharedBuf(int(wqB)), sharedBuf(int(scB)), sharedBuf(int(biB))
	x, out := sharedBuf(inDim*bf16Size), sharedBuf(outDim*bf16Size)
	bytes := wqB + scB + biB + int64(inDim+outDim)*bf16Size
	benchKernelReceipt(b, bytes, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
		return encQMVBF16(enc, wq, sc, bi, x, out, 0, 0, 0, 0, outDim, inDim, gs, bits)
	})
}

// BenchmarkKernelReceipt_QMV_ExpertDown26B: the 26B MoE expert's down projection
// (704 → 2816, gs 64) — the THIN per-expert dispatch that fires k× per MoE layer; the
// occupancy question for the K=1 gap (#392) is whether a matrix this small can ever
// stream near the roofline.
func BenchmarkKernelReceipt_QMV_ExpertDown26B(b *testing.B) {
	const inDim, outDim, gs, bits = 704, 2816, 64, 4
	wqB, scB, biB := qmvWeightBytes(outDim, inDim, gs)
	wq, sc, bi := sharedBuf(int(wqB)), sharedBuf(int(scB)), sharedBuf(int(biB))
	x, out := sharedBuf(inDim*bf16Size), sharedBuf(outDim*bf16Size)
	bytes := wqB + scB + biB + int64(inDim+outDim)*bf16Size
	benchKernelReceipt(b, bytes, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
		return encQMVBF16(enc, wq, sc, bi, x, out, 0, 0, 0, 0, outDim, inDim, gs, bits)
	})
}

// BenchmarkKernelReceipt_GEMVBF16_HeadE2B: the bf16 LM head at E2B shape (262144-token
// vocab × 2048 hidden) — the single fattest dispatch of a bf16 decode round (~1.07 GB of
// weights per token). Its achieved GB/s IS the head's share of the round.
func BenchmarkKernelReceipt_GEMVBF16_HeadE2B(b *testing.B) {
	const inDim, outDim = 2048, 262144
	mat := sharedBuf(outDim * inDim * bf16Size)
	vec, out := sharedBuf(inDim*bf16Size), sharedBuf(outDim*bf16Size)
	bytes := int64(outDim)*int64(inDim)*bf16Size + int64(inDim+outDim)*bf16Size
	benchKernelReceipt(b, bytes, 8, func(enc metal.MTLComputeCommandEncoderObject) error {
		return encGemvBF16VecAt(enc, mat, vec, out, 0, 0, 0, outDim, inDim)
	})
}

// BenchmarkKernelReceipt_SDPAVector26B: single-token decode attention at the 26B's
// global-layer shape (16 heads / 8 KV heads / head_dim 256) over a 4096-row cache —
// the "attn pass" itself. Bytes = the K+V rows the kernel streams.
func BenchmarkKernelReceipt_SDPAVector26B(b *testing.B) {
	const nHeads, nKV, hd, n = 16, 8, 256, 4096
	const kvd = nKV * hd
	requireNativeRuntime(b)
	pso, err := sdpaVectorPipelineForHeadDim(hd)
	if err != nil {
		b.Fatalf("sdpa pso: %v", err)
	}
	q := sharedBuf(nHeads * hd * bf16Size)
	k := sharedBuf(n * kvd * bf16Size)
	v := sharedBuf(n * kvd * bf16Size)
	out := sharedBuf(nHeads * hd * bf16Size)
	nBuf := scalarI32(int32(n))
	bytes := int64(2*n*kvd)*bf16Size + int64(2*nHeads*hd)*bf16Size
	benchKernelReceipt(b, bytes, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
		emitSDPA(encSink{enc}, pso, q, k, v, out, 0, nBuf, nHeads, nKV, n,
			int64(hd), int64(kvd), int64(hd), int64(kvd), 0.0625)
		return nil
	})
}

// BenchmarkKernelReceipt_RMSNorm26B: the full-dModel RMSNorm at 26B width (2816) — a
// THIN op. Its gpu-ns/op is the launch/serialisation floor every one of the round's
// ~700 barriered ops pays; the GB/s here is expected to be tiny and that is the point.
func BenchmarkKernelReceipt_RMSNorm26B(b *testing.B) {
	const dModel = 2816
	requireNativeRuntime(b)
	pso, err := pipelineFor(rmsKernelBF16(dModel))
	if err != nil {
		b.Fatalf("rms pso: %v", err)
	}
	in := sharedBuf(dModel * bf16Size)
	w := bf16ConstBuffer(dModel, 1.0)
	out := sharedBuf(dModel * bf16Size)
	tg := rmsThreadgroup(dModel, pso)
	bytes := int64(3*dModel) * bf16Size
	benchKernelReceipt(b, bytes, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
		emitRMSNorm(encSink{enc}, pso, in, w, out, 0, dModel, 1e-5, tg)
		return nil
	})
}
