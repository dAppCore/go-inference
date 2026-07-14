// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// kernel_receipts_bench_test.go — per-kernel perf receipts (#393 slice 2): the SIGHT.
// Every kernel family with a clean encode seam is measured here at the REAL shapes the
// engine dispatches, with sweeps wherever the interesting answer is a curve (row count,
// kv depth, matrix thinness, axis width) — so no future fix has to guess which view it
// needs. Each benchmark dispatches ONE kernel back-to-back (barriered, so dispatches
// serialise) in its own command buffer and reports the GPU span per dispatch plus the
// achieved GB/s. This is the in-tree half of the kernel instrument (the .gputrace
// capture — `task capture:serve` / `task capture:fire` — is the in-situ half).
//
//	MLX_METALLIB_PATH=…/mlx.metallib go test -tags metal_runtime -run '^$' \
//	  -bench BenchmarkKernelReceipt -benchtime 3x ./engine/metal/
//
// GPU time comes from the command buffer's own GPUStart/EndTime (whole-cb span over N
// serialised dispatches ÷ N) — deliberately NOT MTLCounterSampleBuffer: an isolated
// single-kernel cb measures the same thing with no counter resolution or CPU/GPU
// timestamp calibration. Counter sampling stays reserved for per-dispatch splits inside
// MIXED production rounds (a later slice, likely via the dappcore/apple fork).
//
// READING THE NUMBERS:
//   - GB/s vs the DRAM roofline is only meaningful when the working set exceeds the
//     system-level cache — the 1–1.9GB HEAD rows are the true roofline receipts
//     (~790–810 GB/s). Small-weight rows (attnQ bf16 at 23MB reads "1412") measure SLC
//     re-reads across the barriered loop; their value is the CURVE against siblings at
//     matched sizes, not the absolute GB/s.
//   - the QMVRows fattening curve reads as gpu-ns/op vs M × the single-row qmv's
//     gpu-ns/op (weight bytes are shared, so reported GB/s falls with M by construction).
//
// FIRST RECEIPTS (2026-07-14, M3 Ultra, -benchtime 3x):
//   - roofline: bf16 head gemv 262k×2048 = 807 GB/s, 262k×2816 = 789 (the ~650 napkin
//     was low). K=1 decode streams ~237 GB/s ≈ 30% of it.
//   - the thinness curve (qmv, 4-bit): attnO 4096→2816 = 651 · attnQ = 567 · dense
//     2112/2816 ≈ 320–373 · expert down 704→2816 = 174 · expert gate 2816→704 = 109 ·
//     PLE 256-wide = 50. The 26B's MoE expert dispatches run at ~⅕–¼ of the fat shapes —
//     #392's occupancy gap, now a curve.
//   - the fattening curve (lthn_qmv_rows M=2/3/4): 13.4/19.3/26.6µs vs 11.4µs single-row
//     ⇒ verify-4 ≈ 1.7× the projection throughput of serial rows — the kernel-level
//     price of MTP, consistent with its +26–38% e2e receipts.
//   - deep attention: single-pass sdpa_vector @ kv32768 = 1037µs/259 GB/s; the 2-pass
//     pair = 342µs/785 GB/s — 3× and near-roofline (#365 confirmed at kernel level).
//   - fusion receipts: qknorm+rope fused 4.5µs vs composed 7.5µs; rms+residual fused
//     4.7µs vs composed 7.2µs — each fusion buys ≈ one thin-op floor, as theorised.
//   - the floor table: add/mul 2.1µs · kvq8-store 3.2µs · gelu 3.6µs · rope 3.6µs ·
//     rmsnorm 4.6–5.2µs · embed row gather 2.4µs. ~700 barriered ops/round at these
//     floors is the serialisation tax fusion must be priced against.
//   - the PRODUCTION expert dispatch (lean lthn_gather_qmv, all 8 routes in one launch):
//     gate/up 21.5µs @ 415 GB/s, down 23.8µs @ 378 — ~4× the naive per-expert loop and
//     half-roofline through the route indirection. The thin single-expert rows above are
//     the worst-case bound only; production was already right.
//   - FFN MEGAKERNEL REFUTED (2026-07-14): at its own receipted shape (1536×6144) the
//     live lthn_ffn_megakernel runs 1330µs @ 12 GB/s vs the composed 4-dispatch chain's
//     60µs @ 268 — 22× SLOWER. Its persistent-threadgroup grid-sync (64 tgs × 128
//     threads) is ~5% occupancy by construction on an 80-core M3 Ultra. Production only
//     reaches it through the MoE block's megaLocal gate, which the 26B's inverted local
//     shape fails — the geometry gate is (accidentally) saving us. Do not widen the
//     gate; the whole persistent-tg megakernel class is the wrong shape for this GPU,
//     and any fusion revisit should be barrier-elision (the 2–5µs floor table), never
//     grid-sync.
//
// NOT COVERED HERE (no clean standalone seam — measured in-situ via the capture):
// lthn_gather_qmv + the MoE router/weighted-sum set (closure-bound metas inside
// moe_block), the q4 LM-head argmax (bespoke no-copy ABI), paged SDPA (page tables),
// steel attention / flash prompt (prefill-only shapes), and the four dormant
// megakernels (slice 3 — priced against the thin-op floor this file measures).

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

// --- matmul family -----------------------------------------------------------

// BenchmarkKernelReceipt_QMV sweeps the 4-bit affine qmv across every projection shape a
// decode round dispatches (26B dims; E2B PLE dims) — the THINNESS CURVE. The fat shapes
// should sit near the roofline; how far the thin expert/PLE shapes fall below it is the
// occupancy loss #392 chases.
func BenchmarkKernelReceipt_QMV(b *testing.B) {
	const gs, bits = 64, 4
	shapes := []struct {
		name          string
		inDim, outDim int
	}{
		{"attnQ_2816x4096", 2816, 4096},
		{"attnKV_2816x2048", 2816, 2048},
		{"attnO_4096x2816", 4096, 2816},
		{"denseGate_2816x2112", 2816, 2112},
		{"denseDown_2112x2816", 2112, 2816},
		{"expertGate_2816x704", 2816, 704},
		{"expertDown_704x2816", 704, 2816},
		{"pleGate_2048x256", 2048, 256},
		{"pleProj_256x2048", 256, 2048},
	}
	for _, s := range shapes {
		b.Run(s.name, func(b *testing.B) {
			wqB, scB, biB := qmvWeightBytes(s.outDim, s.inDim, gs)
			wq, sc, bi := sharedBuf(int(wqB)), sharedBuf(int(scB)), sharedBuf(int(biB))
			x, out := sharedBuf(s.inDim*bf16Size), sharedBuf(s.outDim*bf16Size)
			bytes := wqB + scB + biB + int64(s.inDim+s.outDim)*bf16Size
			benchKernelReceipt(b, bytes, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
				return encQMVBF16(enc, wq, sc, bi, x, out, 0, 0, 0, 0, s.outDim, s.inDim, gs, bits)
			})
		})
	}
}

// BenchmarkKernelReceipt_QMVWide races MLX's affine_qmv_wide (k_lanes=8 splitting each
// row's K-reduction across lanes — shipped in our metallib since v0.32.0, gated by MLX to
// gen-15+ GPUs, and NEVER dispatched by this engine) against the qmv variants the engine
// uses today, at the same shapes as the thinness sweep above. The wide impl clamps both
// M and the row index, and its group loop needs only group_size|K — so it is LEGAL at
// every shape the fast variant's %512 gate rejects (2816, 2112, 704). Its shuffle-ladder
// reduction sums in a different order, so adoption would be token-identity tier, not
// byte tier. M=1 rides the nv_2 instance half-idle (nv_1 is not instantiated).
//
// VERDICT (2026-07-14, M3 Ultra, M=1): REFUTED — wide loses at EVERY shape, −7% to −18%
// (attnQ 13.7µs/474 vs 11.2/581; expertDown 6.9/164 vs 5.8/192; pleProj a wash). Its
// M-tiling is the whole design and M=1 burns half the nv_2 tile, so MLX's gen-15 gate
// does not transfer to single-token decode: the engine's existing variant choice is
// CORRECT. Do not revisit for M=1; a revisit is only interesting for the M=2–5 batch
// shapes (MTP verify / small-K fold) where nv_2..5 tiles run full. The rows stay as
// regression sentinels documenting the choice. NOTE the thin-EXPERT story is separate:
// production MoE decode does NOT dispatch these per-expert qmvs — it batches all topK
// experts in one gather_qmv (grid.z = topK, moe_block.go) — so the expert rows above
// bound the WORST case, not the production dispatch; the production number comes from
// the .gputrace capture.
func BenchmarkKernelReceipt_QMVWide(b *testing.B) {
	const gs, bits = 64, 4
	shapes := []struct {
		name          string
		inDim, outDim int
	}{
		{"attnQ_2816x4096", 2816, 4096},
		{"attnKV_2816x2048", 2816, 2048},
		{"denseGate_2816x2112", 2816, 2112},
		{"expertGate_2816x704", 2816, 704},
		{"expertDown_704x2816", 704, 2816},
		{"pleProj_256x2048", 256, 2048},
	}
	for _, s := range shapes {
		b.Run(s.name, func(b *testing.B) {
			requireNativeRuntime(b)
			pso, err := pipelineFor(core.Sprintf("affine_qmv_wide_bfloat16_t_gs_%d_b_%d_nv_2_kl_8_batch_0", gs, bits))
			if err != nil {
				b.Skipf("affine_qmv_wide unavailable: %v", err)
			}
			wqB, scB, biB := qmvWeightBytes(s.outDim, s.inDim, gs)
			wq, sc, bi := sharedBuf(int(wqB)), sharedBuf(int(scB)), sharedBuf(int(biB))
			x, out := sharedBuf(s.inDim*bf16Size), sharedBuf(s.outDim*bf16Size)
			bytes := wqB + scB + biB + int64(s.inDim+s.outDim)*bf16Size
			// dispatch geometry per MLX's qmv_wide dispatcher at M=1: grid x = ceil(M/nv)=1
			// threadgroup column, y = ceil(N / rows_per_tg) with rows_per_tg = (32/kl_8)*2 = 8;
			// threadgroup = (SIMD_SIZE, num_simdgroups) = (32, 2).
			grid := metal.MTLSize{Width: 1, Height: uint((s.outDim + 7) / 8), Depth: 1}
			tg := metal.MTLSize{Width: 32, Height: 2, Depth: 1}
			benchKernelReceipt(b, bytes, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
				sink := encSink{enc}
				sink.setPSO(pso)
				sink.setBuf(wq, 0, 0)
				sink.setBuf(sc, 0, 1)
				sink.setBuf(bi, 0, 2)
				sink.setBuf(x, 0, 3)
				sink.setBuf(out, 0, 4)
				sink.setI32(int32(s.inDim), 5)
				sink.setI32(int32(s.outDim), 6)
				sink.setI32(1, 7) // M — one decode row
				sink.dispatchThreadgroups(grid, tg)
				return nil
			})
		})
	}
}

// BenchmarkKernelReceipt_QMVRows sweeps the register-tiled M-row qmv (lthn_qmv_rows) at
// M=2..4 (its occupancy cap) — the FATTENING CURVE: how much bandwidth each extra
// speculative/batched row buys on the same weight read. This directly prices MTP rows
// and the CB fold at the kernel level.
func BenchmarkKernelReceipt_QMVRows(b *testing.B) {
	const gs, bits = 64, 4
	shapes := []struct {
		name          string
		inDim, outDim int
	}{
		{"attnQ_2816x4096", 2816, 4096},
		{"expertDown_704x2816", 704, 2816},
	}
	for _, s := range shapes {
		for m := 2; m <= lthnQMVRowsMaxM; m++ {
			b.Run(core.Sprintf("%s/rows_%d", s.name, m), func(b *testing.B) {
				requireNativeRuntime(b)
				pso, ok := lthnQMVRowsPipeline(lthnQMVRowsKey{groupSize: gs, bits: bits, m: m})
				if !ok {
					b.Skip("lthn_qmv_rows unavailable (custom metallib not loaded)")
				}
				wqB, scB, biB := qmvWeightBytes(s.outDim, s.inDim, gs)
				wq, sc, bi := sharedBuf(int(wqB)), sharedBuf(int(scB)), sharedBuf(int(biB))
				in, out := sharedBuf(m*s.inDim*bf16Size), sharedBuf(m*s.outDim*bf16Size)
				bytes := wqB + scB + biB + int64(m)*int64(s.inDim+s.outDim)*bf16Size
				benchKernelReceipt(b, bytes, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
					emitQMVRowsTiled(encSink{enc}, pso, wq, 0, sc, 0, bi, 0, in, 0, out, 0, s.inDim, s.outDim)
					return nil
				})
			})
		}
	}
}

// BenchmarkKernelReceipt_GEMVBF16 measures the bf16 tiled gemv at the head shapes (the
// fattest single dispatches of a bf16 round) and a projection shape for contrast.
func BenchmarkKernelReceipt_GEMVBF16(b *testing.B) {
	shapes := []struct {
		name          string
		inDim, outDim int
	}{
		{"headE2B_262144x2048", 2048, 262144},
		{"head26B_262144x2816", 2816, 262144},
		{"attnQ_2816x4096", 2816, 4096},
	}
	for _, s := range shapes {
		b.Run(s.name, func(b *testing.B) {
			mat := sharedBuf(s.outDim * s.inDim * bf16Size)
			vec, out := sharedBuf(s.inDim*bf16Size), sharedBuf(s.outDim*bf16Size)
			bytes := int64(s.outDim)*int64(s.inDim)*bf16Size + int64(s.inDim+s.outDim)*bf16Size
			benchKernelReceipt(b, bytes, 8, func(enc metal.MTLComputeCommandEncoderObject) error {
				return encGemvBF16VecAt(enc, mat, vec, out, 0, 0, 0, s.outDim, s.inDim)
			})
		})
	}
}

// BenchmarkKernelReceipt_GEMVBF16Batched sweeps the grid-Z batched gemv (one weight read
// shared by `batch` independent rows — the MTP verify / small-K fold shape) at the fat
// attention projection: the bf16 fattening curve.
func BenchmarkKernelReceipt_GEMVBF16Batched(b *testing.B) {
	const inDim, outDim = 2816, 4096
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	for _, batch := range []int{2, 4, 8} {
		b.Run(core.Sprintf("attnQ_2816x4096/batch_%d", batch), func(b *testing.B) {
			requireNativeRuntime(b)
			pso, err := pipelineFor(gemvKernelName("bfloat16", bm, bn, sm, sn, tm, tn))
			if err != nil {
				b.Fatalf("gemv pso: %v", err)
			}
			mat := sharedBuf(outDim * inDim * bf16Size)
			vec := sharedBuf(batch * inDim * bf16Size)
			out := sharedBuf(batch * outDim * bf16Size)
			bytes := int64(outDim)*int64(inDim)*bf16Size + int64(batch)*int64(inDim+outDim)*bf16Size
			benchKernelReceipt(b, bytes, 32, func(enc metal.MTLComputeCommandEncoderObject) error {
				emitGemvBatchedVecAt(encSink{enc}, pso, mat, 0, vec, 0, out, 0, inDim, outDim, batch, bm, bn, sm, tm)
				return nil
			})
		})
	}
}

// --- attention family --------------------------------------------------------

// BenchmarkKernelReceipt_SDPAVector sweeps single-token decode attention over kv depth
// at the 26B global-layer shape — where the single-pass kernel's one-threadgroup-per-head
// reduction stops scaling is exactly why the 2-pass exists.
func BenchmarkKernelReceipt_SDPAVector(b *testing.B) {
	const nHeads, nKV, hd = 16, 8, 256
	const kvd = nKV * hd
	for _, n := range []int{512, 4096, 32768} {
		b.Run(core.Sprintf("g16h8kv256/kv_%d", n), func(b *testing.B) {
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
			benchKernelReceipt(b, bytes, 32, func(enc metal.MTLComputeCommandEncoderObject) error {
				emitSDPA(encSink{enc}, pso, q, k, v, out, 0, nBuf, nHeads, nKV, n,
					int64(hd), int64(kvd), int64(hd), int64(kvd), 0.0625)
				return nil
			})
		})
	}
}

// BenchmarkKernelReceipt_SDPA2Pass prices the deep-decode pair (pass-1 blocks fan +
// pass-2 merge) at kv 32768 — comparable directly against SDPAVector/kv_32768 above:
// the two rows together ARE the 2-pass go/no-go receipt.
func BenchmarkKernelReceipt_SDPA2Pass(b *testing.B) {
	const nHeads, nKV, hd, n = 16, 8, 256, 32768
	const kvd = nKV * hd
	requireNativeRuntime(b)
	blocks := int(sdpa2PassBlocks(n, nKV))
	pso1, err := sdpaVector2Pass1Pipeline(core.Sprintf("sdpa_vector_2pass_1_bfloat16_t_%d_%d", hd, hd), int32(blocks))
	if err != nil {
		b.Fatalf("2pass1 pso: %v", err)
	}
	pso2, err := sdpaVector2Pass2Pipeline(core.Sprintf("sdpa_vector_2pass_2_bfloat16_t_%d", hd))
	if err != nil {
		b.Fatalf("2pass2 pso: %v", err)
	}
	q := sharedBuf(nHeads * hd * bf16Size)
	k := sharedBuf(n * kvd * bf16Size)
	v := sharedBuf(n * kvd * bf16Size)
	out := sharedBuf(nHeads * hd * bf16Size)
	partials := sharedBuf(nHeads * blocks * hd * 4)
	sums := sharedBuf(nHeads * blocks * 4)
	maxs := sharedBuf(nHeads * blocks * 4)
	nBuf := scalarI32(int32(n))
	bytes := int64(2*n*kvd)*bf16Size + int64(2*nHeads*hd)*bf16Size
	b.Run(core.Sprintf("g16h8kv256/kv_%d_blocks_%d", n, blocks), func(b *testing.B) {
		benchKernelReceipt(b, bytes, 32, func(enc metal.MTLComputeCommandEncoderObject) error {
			emitSDPA2Pass1NAt(encSink{enc}, pso1, q, 0, k, v, partials, sums, maxs, 0,
				nBuf, 1, nHeads, nKV, n, blocks, int64(hd), int64(kvd), int64(hd), int64(kvd), 0.0625)
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			emitSDPA2Pass2(encSink{enc}, pso2, partials, sums, maxs, out, 1, nHeads, blocks)
			return nil
		})
	})
}

// BenchmarkKernelReceipt_QKNormRope prices the fused per-head QK-norm+RoPE against its
// composed pair (rms-rows then rope) at the 26B Q shape — a live fusion receipt: the
// difference is exactly one thin-op floor if the fusion theory holds.
func BenchmarkKernelReceipt_QKNormRope(b *testing.B) {
	const nHeads, hd = 16, 256
	requireNativeRuntime(b)
	x := sharedBuf(nHeads * hd * bf16Size)
	w := bf16ConstBuffer(hd, 1.0)
	out := sharedBuf(nHeads * hd * bf16Size)
	pos := scalarI32(64)
	dummy := scalarI32(0)
	bytes := int64(2*nHeads*hd+hd) * bf16Size

	b.Run("fused", func(b *testing.B) {
		pso, err := qkNormRopePipelineICB()
		if err != nil {
			b.Skip("lthn_qknorm_rope unavailable (custom metallib not loaded)")
		}
		benchKernelReceipt(b, bytes, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
			emitQKNormRope(encSink{enc}, pso, x, w, out, 0, 0, 0, pos, nil, dummy,
				nHeads, hd, hd, 1e-5, 1.0, 13.2877)
			return nil
		})
	})
	b.Run("composed_rms_then_rope", func(b *testing.B) {
		rmsPSO, err := pipelineFor("rmsbfloat16")
		if err != nil {
			b.Fatalf("rms pso: %v", err)
		}
		ropePSO, err := ropePipelineBF16(false)
		if err != nil {
			b.Fatalf("rope pso: %v", err)
		}
		tg := uint(rmsSimdSize * ((((hd + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
		mid := sharedBuf(nHeads * hd * bf16Size)
		benchKernelReceipt(b, bytes, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
			emitRMSNormRows(encSink{enc}, rmsPSO, x, w, mid, 0, 0, 0, hd, 1e-5, nHeads, tg)
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			emitRope(encSink{enc}, ropePSO, mid, out, 0, 0, pos, nil, nHeads, hd, hd, 1.0, 13.2877)
			return nil
		})
	})
}

// --- thin-op family (the serialisation floor table) ---------------------------

// BenchmarkKernelReceipt_RMSNorm sweeps the full-width RMSNorm across the model widths
// we ship (E2B 2048, 26B 2816, 31B 5376 — the last takes the looped kernel) plus the
// fused residual variant against its composed pair. gpu-ns/op here IS the floor each of
// the round's ~700 barriered thin ops pays.
func BenchmarkKernelReceipt_RMSNorm(b *testing.B) {
	for _, dModel := range []int{2048, 2816, 5376} {
		b.Run(core.Sprintf("width_%d", dModel), func(b *testing.B) {
			requireNativeRuntime(b)
			pso, err := pipelineFor(rmsKernelBF16(dModel))
			if err != nil {
				b.Fatalf("rms pso: %v", err)
			}
			in := sharedBuf(dModel * bf16Size)
			w := bf16ConstBuffer(dModel, 1.0)
			out := sharedBuf(dModel * bf16Size)
			tg := rmsThreadgroup(dModel, pso)
			benchKernelReceipt(b, int64(3*dModel)*bf16Size, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
				emitRMSNorm(encSink{enc}, pso, in, w, out, 0, dModel, 1e-5, tg)
				return nil
			})
		})
	}
	const dModel = 2816
	b.Run("residual_fused_2816", func(b *testing.B) {
		requireNativeRuntime(b)
		pso, err := rmsResidualPipelineICB(dModel)
		if err != nil {
			b.Skip("lthn_rmsnorm_residual unavailable (custom metallib not loaded)")
		}
		x := sharedBuf(dModel * bf16Size)
		w := bf16ConstBuffer(dModel, 1.0)
		res := sharedBuf(dModel * bf16Size)
		out := sharedBuf(dModel * bf16Size)
		tg := rmsThreadgroup(dModel, pso)
		benchKernelReceipt(b, int64(4*dModel)*bf16Size, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
			emitRMSNormResidual(encSink{enc}, pso, x, w, res, out, 0, dModel, 1e-5, tg)
			return nil
		})
	})
	b.Run("residual_composed_2816", func(b *testing.B) {
		requireNativeRuntime(b)
		rmsPSO, err := pipelineFor(rmsKernelBF16(dModel))
		if err != nil {
			b.Fatalf("rms pso: %v", err)
		}
		addPSO, err := pipelineFor("vv_Addbfloat16")
		if err != nil {
			b.Fatalf("add pso: %v", err)
		}
		x := sharedBuf(dModel * bf16Size)
		w := bf16ConstBuffer(dModel, 1.0)
		res := sharedBuf(dModel * bf16Size)
		mid := sharedBuf(dModel * bf16Size)
		out := sharedBuf(dModel * bf16Size)
		tg := rmsThreadgroup(dModel, rmsPSO)
		benchKernelReceipt(b, int64(4*dModel)*bf16Size, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
			emitRMSNorm(encSink{enc}, rmsPSO, x, w, mid, 0, dModel, 1e-5, tg)
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			emitBinary(encSink{enc}, addPSO, res, 0, mid, 0, out, 0, dModel)
			return nil
		})
	})
}

// BenchmarkKernelReceipt_ThinOps is the floor table for the round's small elementwise
// and rope ops — each row's gpu-ns/op is pure launch+barrier tax at these sizes.
func BenchmarkKernelReceipt_ThinOps(b *testing.B) {
	const dModel = 2816
	const nHeads, hd = 16, 256
	b.Run("add_2816", func(b *testing.B) {
		requireNativeRuntime(b)
		pso, err := pipelineFor("vv_Addbfloat16")
		if err != nil {
			b.Fatalf("add pso: %v", err)
		}
		x, y, out := sharedBuf(dModel*bf16Size), sharedBuf(dModel*bf16Size), sharedBuf(dModel*bf16Size)
		benchKernelReceipt(b, int64(3*dModel)*bf16Size, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
			emitBinary(encSink{enc}, pso, x, 0, y, 0, out, 0, dModel)
			return nil
		})
	})
	b.Run("mul_2816", func(b *testing.B) {
		requireNativeRuntime(b)
		pso, err := pipelineFor("vv_Multiplybfloat16")
		if err != nil {
			b.Fatalf("mul pso: %v", err)
		}
		x, y, out := sharedBuf(dModel*bf16Size), sharedBuf(dModel*bf16Size), sharedBuf(dModel*bf16Size)
		benchKernelReceipt(b, int64(3*dModel)*bf16Size, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
			emitBinary(encSink{enc}, pso, x, 0, y, 0, out, 0, dModel)
			return nil
		})
	})
	b.Run("rope_16x256", func(b *testing.B) {
		requireNativeRuntime(b)
		pso, err := ropePipelineBF16(false)
		if err != nil {
			b.Fatalf("rope pso: %v", err)
		}
		x, out := sharedBuf(nHeads*hd*bf16Size), sharedBuf(nHeads*hd*bf16Size)
		pos := scalarI32(64)
		benchKernelReceipt(b, int64(2*nHeads*hd)*bf16Size, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
			emitRope(encSink{enc}, pso, x, out, 0, 0, pos, nil, nHeads, hd, hd, 1.0, 13.2877)
			return nil
		})
	})
	b.Run("gelu_gate_mul_2112", func(b *testing.B) {
		requireNativeRuntime(b)
		pso, err := geluPipelineICB()
		if err != nil {
			b.Skip("lthn_gelu_gate_mul unavailable (custom metallib not loaded)")
		}
		const dFF = 2112
		gate, up, out := sharedBuf(dFF*bf16Size), sharedBuf(dFF*bf16Size), sharedBuf(dFF*bf16Size)
		benchKernelReceipt(b, int64(3*dFF)*bf16Size, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
			emitBinary(encSink{enc}, pso, gate, 0, up, 0, out, 0, dFF)
			return nil
		})
	})
}

// --- cache + embedding family --------------------------------------------------

// BenchmarkKernelReceipt_KVQ8Store prices the q8 owner layers' per-token quantise-store
// (one K or V row → int8 cache row + f32 group scales) at the 26B kv width.
func BenchmarkKernelReceipt_KVQ8Store(b *testing.B) {
	const kvd = 2048
	requireNativeRuntime(b)
	pso, err := kvQ8StorePipeline()
	if err != nil {
		b.Skip("kv q8 store kernel unavailable")
	}
	row := sharedBuf(kvd * bf16Size)
	out := sharedBuf(kvd)
	scales := sharedBuf(kvd / kvQ8GroupSize * 4)
	bytes := int64(kvd)*bf16Size + int64(kvd) + int64(kvd/kvQ8GroupSize*4)
	benchKernelReceipt(b, bytes, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
		emitKVQ8Store(encSink{enc}, pso, row, out, 0, scales, 0, kvd)
		return nil
	})
}

// BenchmarkKernelReceipt_EmbedGatherRow prices the single-token bf16 embedding row
// gather at the 262k-vocab table — the first op of every decode round.
func BenchmarkKernelReceipt_EmbedGatherRow(b *testing.B) {
	const vocab, dModel = 262144, 2048
	requireNativeRuntime(b)
	pso, err := embedGatherPipeline()
	if err != nil {
		b.Skip("embed gather kernel unavailable")
	}
	table := sharedBuf(vocab * dModel * bf16Size)
	token := scalarI32(12345)
	out := sharedBuf(dModel * bf16Size)
	bytes := int64(2*dModel) * bf16Size // one row read + one row written
	benchKernelReceipt(b, bytes, 64, func(enc metal.MTLComputeCommandEncoderObject) error {
		emitEmbedGatherRowBF16(encSink{enc}, pso, token, table, out, 0, 0, dModel, 1.0)
		return nil
	})
}

// --- MoE + megakernel family ---------------------------------------------------

// gatherRouteIdxBuf builds a [routes] int32 device buffer of expert ids / iota.
func gatherRouteIdxBuf(vals []int32) metal.MTLBuffer {
	buf := sharedBuf(len(vals) * 4)
	copy(unsafe.Slice((*byte)(bufferContentsFast(buf)), len(vals)*4), unsafe.Slice((*byte)(unsafe.Pointer(&vals[0])), len(vals)*4))
	return buf
}

// BenchmarkKernelReceipt_GatherQMVAllRoutes measures the PRODUCTION expert dispatch — one
// gather_qmv whose grid.z carries all topK routed experts (the lean lthn_gather_qmv when
// the custom lib ships it, MLX's steel gather otherwise; the log line says which ran).
// This is what the 26B MoE decode actually fires (moe_block.go all-routes lane), against
// which the synthetic per-expert qmv rows above are the worst-case bound. 26B shapes:
// 128 experts, topK 8, dModel 2816, expert dFF 704.
func BenchmarkKernelReceipt_GatherQMVAllRoutes(b *testing.B) {
	requireNativeRuntime(b)
	const numExperts, topK, dModel, expertDFF, gs, bits = 128, 8, 2816, 704, 64, 4
	routeIdx := gatherRouteIdxBuf([]int32{3, 17, 42, 63, 80, 99, 110, 127})
	routeZeros := gatherRouteIdxBuf(make([]int32, topK))
	iota := make([]int32, topK)
	for i := range iota {
		iota[i] = int32(i)
	}
	routeIota := gatherRouteIdxBuf(iota)

	expertWeights := func(outDim, inDim int) (wq, sc, bi metal.MTLBuffer, bytesPerExpert int64) {
		wqB, scB, biB := qmvWeightBytes(outDim, inDim, gs)
		return sharedBuf(int(wqB) * numExperts), sharedBuf(int(scB) * numExperts), sharedBuf(int(biB) * numExperts), wqB + scB + biB
	}

	b.Run("gateUp_2816x704_routes8", func(b *testing.B) {
		requireNativeRuntime(b)
		lean0 := leanGatherDispatches.Load()
		meta, err := gatherQMVAllRoutesMetadata(numExperts, expertDFF, dModel, gs, bits, expertDFF, topK, 1, false)
		if err != nil {
			b.Fatalf("meta: %v", err)
		}
		key := gatherQMVAllRoutesMetaKey{numExperts: numExperts, outDim: expertDFF, inDim: dModel, groupSize: gs, bits: bits, expertRows: expertDFF, routes: topK, xRows: 1, batchedX: false}
		pso, err := gatherQMVBF16SteelPipeline(expertDFF, dModel, gs, bits)
		if err != nil {
			b.Fatalf("steel pso: %v", err)
		}
		wq, sc, bi, perExpert := expertWeights(expertDFF, dModel)
		x := sharedBuf(dModel * bf16Size)
		out := sharedBuf(topK * expertDFF * bf16Size)
		bytes := int64(topK)*perExpert + int64(dModel+topK*expertDFF)*bf16Size
		benchKernelReceipt(b, bytes, 32, func(enc metal.MTLComputeCommandEncoderObject) error {
			emitGatherQMVAllRoutes(encSink{enc}, pso, meta, key, x, 0, wq, 0, sc, 0, bi, 0, routeZeros, routeIdx, 0, out, 0, expertDFF, dModel, gs, bits, 0, topK)
			return nil
		})
		if leanGatherDispatches.Load() > lean0 {
			b.Logf("path: lean lthn_gather_qmv")
		} else {
			b.Logf("path: MLX steel gather_qmv (lean unavailable)")
		}
	})

	b.Run("down_704x2816_routes8", func(b *testing.B) {
		requireNativeRuntime(b)
		meta, err := gatherQMVAllRoutesMetadata(numExperts, dModel, expertDFF, gs, bits, dModel, topK, topK, true)
		if err != nil {
			b.Fatalf("meta: %v", err)
		}
		key := gatherQMVAllRoutesMetaKey{numExperts: numExperts, outDim: dModel, inDim: expertDFF, groupSize: gs, bits: bits, expertRows: dModel, routes: topK, xRows: topK, batchedX: true}
		pso, err := gatherQMVBF16SteelPipeline(dModel, expertDFF, gs, bits)
		if err != nil {
			b.Fatalf("steel pso: %v", err)
		}
		wq, sc, bi, perExpert := expertWeights(dModel, expertDFF)
		x := sharedBuf(topK * expertDFF * bf16Size)
		out := sharedBuf(topK * dModel * bf16Size)
		bytes := int64(topK)*perExpert + int64(topK*(expertDFF+dModel))*bf16Size
		benchKernelReceipt(b, bytes, 32, func(enc metal.MTLComputeCommandEncoderObject) error {
			emitGatherQMVAllRoutes(encSink{enc}, pso, meta, key, x, 0, wq, 0, sc, 0, bi, 0, routeIota, routeIdx, 0, out, 0, dModel, expertDFF, gs, bits, 0, topK)
			return nil
		})
	})
}

// BenchmarkKernelReceipt_FFNMega races the LIVE ffn megakernel (one grid-synced dispatch
// covering gate+up+gelu+down — engaged in production behind ffnMegaDefaultGeometry) against
// the composed 4-dispatch chain, at the receipted shape family (E2B-class 1536×6144,
// ratio-4; the 26B's inverted local 2816×2112 FAILS the geometry gate by design). One
// dispatch per command buffer: the mega's grid-barrier arrive counter is host-reset at
// encode time, which is only safe before the cb runs.
func BenchmarkKernelReceipt_FFNMega(b *testing.B) {
	requireNativeRuntime(b)
	const dModel, dFF, gs, bits = 1536, 6144, 64, 4
	mkView := func(outDim, inDim int) (quantMLPProjView, int64) {
		wqB, scB, biB := qmvWeightBytes(outDim, inDim, gs)
		return quantMLPProjView{
			packed:    bufView{buf: sharedBuf(int(wqB))},
			scales:    bufView{buf: sharedBuf(int(scB))},
			biases:    bufView{buf: sharedBuf(int(biB))},
			groupSize: gs, bits: bits,
		}, wqB + scB + biB
	}
	gate, gateB := mkView(dFF, dModel)
	up, upB := mkView(dFF, dModel)
	down, downB := mkView(dModel, dFF)
	x := sharedBuf(dModel * bf16Size)
	gated := sharedBuf(dFF * bf16Size)
	out := sharedBuf(dModel * bf16Size)
	bytes := gateB + upB + downB + int64(2*dModel+2*dFF)*bf16Size

	b.Run("mega_1536x6144", func(b *testing.B) {
		requireNativeRuntime(b)
		if !ffnMegaDefaultGeometry(dModel, dFF) {
			b.Fatal("shape must pass the mega geometry gate")
		}
		pso, err := ffnMegaPipelineBits(bits)
		if err != nil {
			b.Skip("lthn_ffn_megakernel unavailable (custom metallib not loaded)")
		}
		arrive := sharedBuf(4)
		arrivePtr := (*uint32)(bufferContentsFast(arrive))
		benchKernelReceipt(b, bytes, 1, func(enc metal.MTLComputeCommandEncoderObject) error {
			*arrivePtr = 0 // encode-time reset — the cb has not run yet
			emitFFNMega(encSink{enc}, pso, x, 0, gate, up, down, gated, out, 0, arrive, dModel, dFF)
			return nil
		})
	})
	b.Run("composed_4dispatch_1536x6144", func(b *testing.B) {
		requireNativeRuntime(b)
		geluPSO, err := geluPipelineICB()
		if err != nil {
			b.Skip("lthn_gelu_gate_mul unavailable (custom metallib not loaded)")
		}
		gateOut := sharedBuf(dFF * bf16Size)
		upOut := sharedBuf(dFF * bf16Size)
		benchKernelReceipt(b, bytes, 1, func(enc metal.MTLComputeCommandEncoderObject) error {
			if err := encQMVBF16(enc, gate.packed.buf, gate.scales.buf, gate.biases.buf, x, gateOut, 0, 0, 0, 0, dFF, dModel, gs, bits); err != nil {
				return err
			}
			if err := encQMVBF16(enc, up.packed.buf, up.scales.buf, up.biases.buf, x, upOut, 0, 0, 0, 0, dFF, dModel, gs, bits); err != nil {
				return err
			}
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			emitBinary(encSink{enc}, geluPSO, gateOut, 0, upOut, 0, gated, 0, dFF)
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			return encQMVBF16(enc, down.packed.buf, down.scales.buf, down.biases.buf, gated, out, 0, 0, 0, 0, dModel, dFF, gs, bits)
		})
	})
}
