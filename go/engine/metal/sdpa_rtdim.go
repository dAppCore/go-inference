// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// sdpa_rtdim.go is the Go driver for kernels/lthn_sdpa_rtdim.metal — the runtime-head-dim decode
// SDPA fallback (#28). The shipped MLX metallib only instantiates sdpa_vector for a fixed class of
// head dims (64/96/128/256); head_dim 32 has no pipeline anywhere, and before this file every call
// site's absent-width behaviour differed (some errored immediately, some fell back to a different
// kernel family that also lacked the width). sdpaVectorDispatchForHeadDim is the SINGLE chokepoint
// every non-sinks, non-q8 single-pass SDPA call site in this package now resolves its pipeline
// through: the fixed per-headDim pipeline when the metallib carries it (byte-identical fast path,
// untouched), or — logged exactly once per distinct absent width — the lthn runtime-dim fallback.
// Never an error for a missing width; never a different fallback per call site. The 2-pass
// long-context pair has a proven, tested pass-1 runtime-dim kernel (below) but is not fully wired
// end to end — see the comment above emitSDPA2Pass1RTDimAt for why and what encSDPADecodeAt does
// instead (falls through to the single-pass fallback, which has no length ceiling).

const (
	// sdpaRTDimBD is the SIMD-group width every MLX/lthn sdpa_vector variant divides the head
	// dimension across — Apple GPU simdgroups are hardware-fixed at 32 lanes, so this is not a
	// tuning choice. The runtime-dim fallback therefore only serves head dims that are exact
	// multiples of it (a constraint the fixed 64/96/128/256 instantiations already satisfy).
	sdpaRTDimBD = 32
	// sdpaVectorRTDimMaxHeadDim mirrors kernels/lthn_sdpa_rtdim.metal's LTHN_SDPA_RTDIM_MAX_D —
	// the per-thread register-array cap, sized to the largest FIXED sdpa_vector instantiation
	// (256). A head_dim above it is refused with a named error rather than silently truncated.
	sdpaVectorRTDimMaxHeadDim = 256
)

var (
	sdpaRTDimMu sync.Mutex

	sdpaRTDimPSO      metal.MTLComputePipelineState
	sdpaRTDimPSOBuilt bool

	sdpaRTDim2Pass1PSO      metal.MTLComputePipelineState
	sdpaRTDim2Pass1PSOBuilt bool

	// sdpaVectorHeadDimMissing records head dims confirmed absent from the FIXED metallib (mirrors
	// sdpa_multiq.go's sdpaMultiQMissing idiom) — so a repeated decode call skips straight to the
	// runtime-dim fallback instead of re-attempting (and re-failing) the fixed lookup every token,
	// and so the one-time notice fires exactly once per width.
	sdpaVectorHeadDimMissing = map[int]bool{}
)

// sdpaVectorRTDimValidHeadDim reports whether headDim is inside the runtime-dim fallback's
// supported domain: a positive multiple of the simdgroup width, no larger than the register-array
// cap. Both bounds are enforced BEFORE dispatch — never silently truncated or corrupted.
func sdpaVectorRTDimValidHeadDim(headDim int) bool {
	return headDim > 0 && headDim%sdpaRTDimBD == 0 && headDim <= sdpaVectorRTDimMaxHeadDim
}

// sdpaVectorRTDimPipeline resolves (and caches) the single-pass runtime-dim decode SDPA. One
// pipeline total — headDim is a runtime buffer parameter, not a template argument, so no
// per-headDim PSO variants are needed here (contrast sdpaVectorHeadDimPSOCache's per-width cache).
func sdpaVectorRTDimPipeline() (metal.MTLComputePipelineState, error) {
	sdpaRTDimMu.Lock()
	defer sdpaRTDimMu.Unlock()
	if sdpaRTDimPSOBuilt {
		return sdpaRTDimPSO, nil
	}
	// Cache SUCCESS only. A nil/torn-down customLibrary is a transient state some
	// test suites pass through — latching the error here poisoned every later
	// caller in the process (the 199-failure suite cascade, 2026-07-19).
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.sdpaVectorRTDimPipeline: custom library unavailable")
	}
	fn := customLibrary.NewFunctionWithName("lthn_sdpa_vector_rtdim_bf16")
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.sdpaVectorRTDimPipeline: kernel lthn_sdpa_vector_rtdim_bf16 not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, err
	}
	sdpaRTDimPSO, sdpaRTDimPSOBuilt = pso, true
	return sdpaRTDimPSO, nil
}

// sdpaVector2Pass1RTDimPipeline resolves (and caches) the 2-pass pass-1 runtime-dim decode SDPA —
// see kernels/lthn_sdpa_rtdim.metal for the batch=1 constraint. One pipeline total: blocks is a
// runtime buffer parameter here (not the fixed kernel's function-constant 26), so unlike
// sdpaVector2Pass1HeadDimCache no (headDim, blocks) product of PSOs is ever built.
func sdpaVector2Pass1RTDimPipeline() (metal.MTLComputePipelineState, error) {
	sdpaRTDimMu.Lock()
	defer sdpaRTDimMu.Unlock()
	if sdpaRTDim2Pass1PSOBuilt {
		return sdpaRTDim2Pass1PSO, nil
	}
	// Cache SUCCESS only — see sdpaVectorRTDimPipeline's latch note.
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass1RTDimPipeline: custom library unavailable")
	}
	fn := customLibrary.NewFunctionWithName("lthn_sdpa_vector_2pass_1_rtdim_bf16")
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass1RTDimPipeline: kernel lthn_sdpa_vector_2pass_1_rtdim_bf16 not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, err
	}
	sdpaRTDim2Pass1PSO, sdpaRTDim2Pass1PSOBuilt = pso, true
	return sdpaRTDim2Pass1PSO, nil
}

// sdpaRTDimNotice fires the ONE-TIME (per kind+headDim, via the caller's missing-map gate)
// diagnostic naming the fallback's engagement — visible without spamming the decode hot loop.
func sdpaRTDimNotice(kind string, headDim int) {
	nativeTraceLog(core.Sprintf(
		"sdpa-rtdim: head_dim=%d has no fixed %s pipeline in the metallib — routing to the lthn runtime-dim fallback (#28)\n",
		headDim, kind))
}

// sdpaVectorDispatchForHeadDim resolves the single-pass decode SDPA pipeline for headDim: the
// FIXED per-headDim pipeline when the shipped MLX metallib carries that instantiation (rtDim=false
// — the fast path, byte-identical to before this fallback existed), or — when it doesn't, e.g.
// headDim 32 against the metallib's 64/96/128/256-class widths (#28) — the lthn runtime-dim
// fallback (rtDim=true), logged once per distinct absent width. Callers branch their emit on
// rtDim: false uses emitSDPA/emitSDPAAt (the existing 0..10 ABI, unchanged); true uses
// emitSDPARTDim/emitSDPARTDimAt (the same ABI plus headDim at buffer(11)).
func sdpaVectorDispatchForHeadDim(headDim int) (pso metal.MTLComputePipelineState, rtDim bool, err error) {
	// ALWAYS try the fixed pipeline first — the missing-map gates only the
	// one-time notice, never the routing. Latching the route poisoned every
	// later caller at that width when the fixed lookup failed once during a
	// transient library-teardown window (the 199-failure suite cascade).
	if fixed, ferr := sdpaVectorPipelineForHeadDim(headDim); ferr == nil {
		return fixed, false, nil
	}
	sdpaRTDimMu.Lock()
	first := !sdpaVectorHeadDimMissing[headDim]
	sdpaVectorHeadDimMissing[headDim] = true
	sdpaRTDimMu.Unlock()
	if first {
		sdpaRTDimNotice("sdpa_vector", headDim)
	}
	if !sdpaVectorRTDimValidHeadDim(headDim) {
		return nil, true, core.NewError(core.Sprintf(
			"native.sdpaVectorDispatchForHeadDim: head_dim=%d has no fixed pipeline and is not eligible for the runtime-dim fallback (must be a positive multiple of %d, <= %d)",
			headDim, sdpaRTDimBD, sdpaVectorRTDimMaxHeadDim))
	}
	pso, err = sdpaVectorRTDimPipeline()
	return pso, true, err
}

// NAMED REMAINING BOUNDARY (#28): there is no sdpaVector2Pass1DispatchForHeadDim wrapper. The
// 2-pass long-context kernel is a PAIR — pass 1 (sdpaVector2Pass1RTDimPipeline /
// emitSDPA2Pass1RTDimAt below, both implemented and gated in isolation) and pass 2, the merge
// kernel (sdpaVector2Pass2PipelineForHeadDim in sdpa.go) — and pass 2 is ALSO a per-headDim
// fixed-only pipeline (`sdpa_vector_2pass_2_bfloat16_t_<D>`, MLX-shipped, same 64/96/128/256
// class) with no runtime-dim port. Wiring pass 1's fallback alone would resolve a pipeline, fire
// the one-time notice, and then still fail at pass 2 — a misleading half-success. So
// encSDPADecodeAt instead falls all the way through to the single-pass runtime-dim fallback
// whenever EITHER fixed 2-pass pipeline is unavailable: correct at any kvLen (the single-pass
// kernel has no length ceiling, just no long-context parallelism), never an error, never a
// notice that promises more than it delivers. Porting pass 2 is the honest remaining gap — TRUTHFUL
// PARTIAL: this file lands a proven, tested pass-1 kernel ready for that follow-up, not a rushed,
// unvalidated pass-2 written from scratch without MLX's source to cross-check against.

// emitSDPARTDim records the runtime-dim single-pass decode SDPA through any sink — emitSDPA's
// counterpart for lthn_sdpa_vector_rtdim_bf16.
func emitSDPARTDim[S dispatchSink](sink S, pso metal.MTLComputePipelineState, q, k, v, out metal.MTLBuffer, kvByteOff uint, nBuf metal.MTLBuffer, nHeads, nKVHeads, headDim, n int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) {
	emitSDPARTDimAt(sink, pso, q, 0, k, v, out, 0, kvByteOff, nBuf, nHeads, nKVHeads, headDim, n, kHeadStride, kSeqStride, vHeadStride, vSeqStride, scale)
}

// emitSDPARTDimAt is emitSDPAAt's runtime-dim counterpart: the SAME 0..10 ABI (q=0 k=1 v=2 out=3
// gqa_factor=4 N=5 strides=6..9 scale=10) plus headDim bound at buffer(11) — the ONE addition
// lthn_sdpa_vector_rtdim_bf16 needs since it cannot get D from a C++ template. Same grid — one
// threadgroup per head, 1024-wide — as the fixed kernel, so callers only branch the pso/emit
// choice, never the dispatch geometry.
func emitSDPARTDimAt[S dispatchSink](sink S, pso metal.MTLComputePipelineState, q metal.MTLBuffer, qOff uint, k, v, out metal.MTLBuffer, outOff, kvByteOff uint, nBuf metal.MTLBuffer, nHeads, nKVHeads, headDim, n int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) {
	sink.setPSO(pso)
	sink.setBuf(q, qOff, 0)
	sink.setBuf(k, kvByteOff, 1)
	sink.setBuf(v, kvByteOff, 2)
	sink.setBuf(out, outOff, 3)
	sink.setI32(int32(nHeads/nKVHeads), 4) // gqa_factor
	if nBuf != nil {
		sink.setBuf(nBuf, 0, 5) // ICB: the N buffer, rebound per token at replay
	} else {
		sink.setI32(int32(n), 5) // live: inline N (the live cache length this token)
	}
	sink.setI64(kHeadStride, 6)
	sink.setI64(kSeqStride, 7)
	sink.setI64(vHeadStride, 8)
	sink.setI64(vSeqStride, 9)
	sink.setF32(scale, 10)
	sink.setI32(int32(headDim), 11)
	sink.dispatchThreadgroups(metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1}, metal.MTLSize{Width: 1024, Height: 1, Depth: 1})
}

// emitSDPA2Pass1RTDimAt is emitSDPA2Pass1At's runtime-dim counterpart for
// lthn_sdpa_vector_2pass_1_rtdim_bf16 — BATCH-1 ONLY (see the kernel's own comment): the batch
// parameter is accepted for signature symmetry with emitSDPA2Pass1At but MUST be 1, guarded here
// rather than silently mis-addressed. ABI: q=0 k=1 v=2 partials=3 sums=4 maxs=5 N=6 strides=7..10
// scale=11 headDim=12 blocks=13 — blocks is bound BOTH as buffer(13) (the kernel's own
// block-strided loop and o_offset stride — a runtime value here, not a function constant, since
// ONE fallback pipeline serves every headDim/blocks combination) and as the dispatch grid's Depth
// (how many threadgroups actually run).
//
// NOT YET WIRED into any call site (see the comment above this function's neighbours) — pass 2
// (the merge kernel) has no runtime-dim port, so completing pass 1 alone would resolve a pipeline
// and then still fail at pass 2. This function IS proven correct in isolation (its emitted
// partials/sums/maxs are gated against the fixed pass-1 kernel's own outputs at headDim=64 in
// sdpa_rtdim_test.go) — it is the ready building block for the pass-2 follow-up, not dead code.
func emitSDPA2Pass1RTDimAt[S dispatchSink](sink S, pso metal.MTLComputePipelineState, q metal.MTLBuffer, qOff uint, k, v, partials, sums, maxs metal.MTLBuffer, kvByteOff uint, nBuf metal.MTLBuffer, batch, nHeads, nKVHeads, headDim, n, blocks int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) error {
	if batch != 1 {
		return core.NewError("native.emitSDPA2Pass1RTDimAt: the runtime-dim 2-pass fallback only supports batch=1")
	}
	sink.setPSO(pso)
	sink.setBuf(q, qOff, 0)
	sink.setBuf(k, kvByteOff, 1)
	sink.setBuf(v, kvByteOff, 2)
	sink.setBuf(partials, 0, 3)
	sink.setBuf(sums, 0, 4)
	sink.setBuf(maxs, 0, 5)
	if nBuf != nil {
		sink.setBuf(nBuf, 0, 6) // ICB: the N buffer, rebound per token at replay
	} else {
		sink.setI32(int32(n), 6)
	}
	sink.setI64(kHeadStride, 7)
	sink.setI64(kSeqStride, 8)
	sink.setI64(vHeadStride, 9)
	sink.setI64(vSeqStride, 10)
	sink.setF32(scale, 11)
	sink.setI32(int32(headDim), 12)
	sink.setI32(int32(blocks), 13)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(nKVHeads), Height: uint(batch), Depth: uint(blocks)},
		metal.MTLSize{Width: 32, Height: uint(nHeads / nKVHeads), Depth: 1},
	)
	return nil
}
