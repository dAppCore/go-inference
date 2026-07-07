// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"github.com/tmc/apple/metal"
)

// dispatchSink abstracts "record one compute dispatch" over the two Metal targets the decode path
// drives: the live MTLComputeCommandEncoder (re-encode every token) and the MTLIndirectComputeCommand
// (record-once ICB replay). An op written against a sink — its pipeline, buffer bindings, and dispatch
// geometry, i.e. the binding ABI — records into EITHER target from ONE body, instead of the two parallel
// emit-helper sets (the live enc* funcs and the ICB recorder's set*/rec* closures) that drifted. That
// drift is not hypothetical: the 12B/31B kvHeads gate sat closed for a long time on a believed-but-false
// recorder divergence that lived in exactly the gap between the two copies.
//
// The asymmetries the sink hides:
//   - scalars: live encoders bind inline bytes through the raw fast-send path; ICB commands bind
//     process-memoised scalar buffers (scalarI32/…), because ICB commands cannot set bytes inline.
//   - dispatch: DispatchThreads* / DispatchThreadgroups* on the encoder vs the ConcurrentDispatch*
//     variants on the ICB command.
//
// What the sink does NOT hide (caller-provided, because they legitimately differ per target):
//   - the pipeline: ICB ops need a supportIndirectCommandBuffers variant (pipelineForICB); the live path
//     uses pipelineFor — different PSO objects for the same kernel, so the caller passes the right one.
//   - per-token-VARYING scalars (the SDPA live length, the sliding read offset): those are the ICB
//     orchestration's rebindable buffers, passed in as buffers; the sink owns only constant scalars.
type dispatchSink interface {
	setPSO(pso metal.MTLComputePipelineState)
	setBuf(buf metal.MTLBuffer, off, idx uint)
	setI32(v int32, idx uint)
	setI64(v int64, idx uint)
	setF32(v float32, idx uint)
	dispatchThreads(grid, group metal.MTLSize)
	dispatchThreadgroups(grid, group metal.MTLSize)
}

// encSink records into a live compute encoder: scalar buffers, plain dispatch.
type encSink struct {
	enc metal.MTLComputeCommandEncoder
}

func (s encSink) setPSO(pso metal.MTLComputePipelineState) { setPSO(s.enc, pso) }
func (s encSink) setBuf(buf metal.MTLBuffer, off, idx uint) {
	setBuf(s.enc, buf, off, idx)
}
func (s encSink) setI32(v int32, idx uint)   { setBytesI32(s.enc, v, idx) }
func (s encSink) setI64(v int64, idx uint)   { setBytesI64(s.enc, v, idx) }
func (s encSink) setF32(v float32, idx uint) { setBytesF32(s.enc, v, idx) }
func (s encSink) dispatchThreads(grid, group metal.MTLSize) {
	dispatchThreads(s.enc, grid, group)
}
func (s encSink) dispatchThreadgroups(grid, group metal.MTLSize) {
	dispatchThreadgroups(s.enc, grid, group)
}

// encObjectSink is the same live encoder target as encSink, but keeps the
// generated concrete object type for hot paths that already have it. This avoids
// allocating when a concrete encoder is converted through the protocol
// interface just to reach the raw fast-send helpers.
type encObjectSink struct {
	enc metal.MTLComputeCommandEncoderObject
}

func (s encObjectSink) setPSO(pso metal.MTLComputePipelineState) {
	setPSOObject(s.enc, pso)
}
func (s encObjectSink) setBuf(buf metal.MTLBuffer, off, idx uint) {
	setBufObject(s.enc, buf, off, idx)
}
func (s encObjectSink) setI32(v int32, idx uint)   { setBytesI32Object(s.enc, v, idx) }
func (s encObjectSink) setI64(v int64, idx uint)   { setBytesI64Object(s.enc, v, idx) }
func (s encObjectSink) setF32(v float32, idx uint) { setBytesF32Object(s.enc, v, idx) }
func (s encObjectSink) dispatchThreads(grid, group metal.MTLSize) {
	dispatchThreadsObject(s.enc, grid, group)
}
func (s encObjectSink) dispatchThreadgroups(grid, group metal.MTLSize) {
	dispatchThreadgroupsObject(s.enc, grid, group)
}

// icbSink records into an ICB command: scalars bound as (process-memoised) buffers — an ICB command
// cannot SetBytes inline — and concurrent dispatch. The scalar buffers come from scalarI32/I64/F32, which
// memoise by value, so binding a scalar adds no per-record allocation and reuses the recorder's own
// resident scalar handles (created via the same scalar* helpers).
type icbSink struct {
	cmd metal.MTLIndirectComputeCommand
}

func (s icbSink) setPSO(pso metal.MTLComputePipelineState) { s.cmd.SetComputePipelineState(pso) }
func (s icbSink) setBuf(buf metal.MTLBuffer, off, idx uint) {
	s.cmd.SetKernelBufferOffsetAtIndex(buf, off, idx)
}
func (s icbSink) setI32(v int32, idx uint) { s.cmd.SetKernelBufferOffsetAtIndex(scalarI32(v), 0, idx) }
func (s icbSink) setI64(v int64, idx uint) { s.cmd.SetKernelBufferOffsetAtIndex(scalarI64(v), 0, idx) }
func (s icbSink) setF32(v float32, idx uint) {
	s.cmd.SetKernelBufferOffsetAtIndex(scalarF32(v), 0, idx)
}
func (s icbSink) dispatchThreads(grid, group metal.MTLSize) {
	s.cmd.ConcurrentDispatchThreadsThreadsPerThreadgroup(grid, group)
}
func (s icbSink) dispatchThreadgroups(grid, group metal.MTLSize) {
	s.cmd.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(grid, group)
}

type fastICBSink struct {
	cmd metal.MTLIndirectComputeCommand
}

func (s fastICBSink) setPSO(pso metal.MTLComputePipelineState) { setICBPSO(s.cmd, pso) }
func (s fastICBSink) setBuf(buf metal.MTLBuffer, off, idx uint) {
	setICBKernelBuffer(s.cmd, buf, off, idx)
}
func (s fastICBSink) setI32(v int32, idx uint) {
	setICBKernelBuffer(s.cmd, scalarI32(v), 0, idx)
}
func (s fastICBSink) setI64(v int64, idx uint) {
	setICBKernelBuffer(s.cmd, scalarI64(v), 0, idx)
}
func (s fastICBSink) setF32(v float32, idx uint) {
	setICBKernelBuffer(s.cmd, scalarF32(v), 0, idx)
}
func (s fastICBSink) dispatchThreads(grid, group metal.MTLSize) {
	concurrentDispatchThreads(s.cmd, grid, group)
}
func (s fastICBSink) dispatchThreadgroups(grid, group metal.MTLSize) {
	concurrentDispatchThreadgroups(s.cmd, grid, group)
}

// emitRMSNorm records a single-row bf16 RMSNorm (out = rmsnorm(x, w@wOff), axisSize ≤ the kernel cap)
// through any sink: the binding ABI (x=0, w=1, out=2, eps=3, axisSize=4, ws=5) + a square single-row
// threadgroup. pso + tg are caller-provided — the ICB needs a supportIndirectCommandBuffers pipeline
// and carries its own tg. This is the ONE body behind both encRMSNormBF16 (live, encSink) and the ICB
// recorder's setRMS (icbSink); byte-parity with the re-encode path is gated by the ICB parity suite.
func emitRMSNorm[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x, w, out metal.MTLBuffer, wOff uint, axisSize int, eps float32, tg uint) {
	emitRMSNormAt(sink, pso, x, w, out, 0, wOff, 0, axisSize, eps, tg)
}

// emitRMSNormAt is emitRMSNorm with the input and output bound at byte offsets — the SAME
// single-row specialised pipeline, so a row living at an offset inside a shared K-row buffer
// norms bit-identically to the sequential path (the generic rows kernel reduces in a different
// order and drifts by ulps).
func emitRMSNormAt[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, axisSize int, eps float32, tg uint) {
	sink.setPSO(pso)
	sink.setBuf(x, xOff, 0)
	sink.setBuf(w, wOff, 1)
	sink.setBuf(out, outOff, 2)
	sink.setF32(eps, 3)
	sink.setI32(int32(axisSize), 4)
	sink.setI32(1, 5) // ws (row stride = 1, single row)
	sink.dispatchThreads(metal.MTLSize{Width: tg, Height: 1, Depth: 1}, metal.MTLSize{Width: tg, Height: 1, Depth: 1})
}

// emitRMSNormRows records a per-row bf16 RMSNorm — `rows` independent rows of axisSize each (each at its
// byte offset) — through any sink: same binding ABI as emitRMSNorm (x=0, w=1, out=2, eps=3, axisSize=4,
// ws=5) but dispatched as rows·tg threads in tg-wide groups. The body behind encRMSNormRowsBF16 (live)
// and the recorder's setRMSRows (gemma4 per-head QK-norm). pso + tg caller-provided.
func emitRMSNormRows[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, axisSize int, eps float32, rows int, tg uint) {
	sink.setPSO(pso)
	sink.setBuf(x, xOff, 0)
	sink.setBuf(w, wOff, 1)
	sink.setBuf(out, outOff, 2)
	sink.setF32(eps, 3)
	sink.setI32(int32(axisSize), 4)
	sink.setI32(1, 5)
	sink.dispatchThreads(metal.MTLSize{Width: uint(rows) * tg, Height: 1, Depth: 1}, metal.MTLSize{Width: tg, Height: 1, Depth: 1})
}

// emitRMSNormResidual records the FUSED post-norm tail out = res + rmsnorm(x, w@wOff) in one dispatch
// (lthn_rmsnorm_residual_bf16) through any sink: x=0, w=1, res=2, out=3, eps=4, axisSize=5, ws=6. The
// body behind encRMSNormResidualBF16 (live) and the recorder's setRMSResidual. pso + tg caller-provided.
func emitRMSNormResidual[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x, w, res, out metal.MTLBuffer, wOff uint, axisSize int, eps float32, tg uint) {
	emitRMSNormResidualAt(sink, pso, x, w, res, out, 0, wOff, 0, 0, axisSize, eps, tg)
}

// emitRMSNormResidualAt is emitRMSNormResidual with the branch input, residual and output bound
// at byte offsets — the SAME fused pipeline, so a batched row living at an offset inside a shared
// K-row buffer runs the identical fused tail the sequential step records.
func emitRMSNormResidualAt[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x, w, res, out metal.MTLBuffer, xOff, wOff, resOff, outOff uint, axisSize int, eps float32, tg uint) {
	sink.setPSO(pso)
	sink.setBuf(x, xOff, 0)
	sink.setBuf(w, wOff, 1)
	sink.setBuf(res, resOff, 2)
	sink.setBuf(out, outOff, 3)
	sink.setF32(eps, 4)
	sink.setI32(int32(axisSize), 5)
	sink.setI32(1, 6)
	sink.dispatchThreads(metal.MTLSize{Width: tg, Height: 1, Depth: 1}, metal.MTLSize{Width: tg, Height: 1, Depth: 1})
}

// emitLayerNorm records per-row LayerNorm over `rows` rows of axisSize each. Binding ABI:
// x=0, weight=1, bias=2, out=3, eps=4, axisSize=5, weightStride=6, biasStride=7.
func emitLayerNorm[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x, w, b, out metal.MTLBuffer, axisSize, rows int, eps float32, tg uint) {
	sink.setPSO(pso)
	sink.setBuf(x, 0, 0)
	sink.setBuf(w, 0, 1)
	sink.setBuf(b, 0, 2)
	sink.setBuf(out, 0, 3)
	sink.setF32(eps, 4)
	sink.setI32(int32(axisSize), 5)
	sink.setI32(1, 6)
	sink.setI32(1, 7)
	sink.dispatchThreads(
		metal.MTLSize{Width: uint(rows) * tg, Height: 1, Depth: 1},
		metal.MTLSize{Width: tg, Height: 1, Depth: 1},
	)
}

// emitSoftmax records row-wise float32 softmax over `rows` rows of axisSize each. Binding ABI:
// in=0, out=1, axisSize=2; one threadgroup per row.
func emitSoftmax[S dispatchSink](sink S, pso metal.MTLComputePipelineState, in, out metal.MTLBuffer, axisSize, rows int, tg uint) {
	sink.setPSO(pso)
	sink.setBuf(in, 0, 0)
	sink.setBuf(out, 0, 1)
	sink.setI32(int32(axisSize), 2)
	sink.dispatchThreads(
		metal.MTLSize{Width: uint(rows) * tg, Height: 1, Depth: 1},
		metal.MTLSize{Width: tg, Height: 1, Depth: 1},
	)
}

// emitSteelGemm records one MLX steel GEMM dispatch. Binding ABI: A=0, B=1, D=3, params=4.
func emitSteelGemm[S dispatchSink](sink S, pso metal.MTLComputePipelineState, a, b, out, params metal.MTLBuffer, tn, tm int, wn, wm uint) {
	sink.setPSO(pso)
	sink.setBuf(a, 0, 0)
	sink.setBuf(b, 0, 1)
	sink.setBuf(out, 0, 3)
	sink.setBuf(params, 0, 4)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(tn), Height: uint(tm), Depth: 1},
		metal.MTLSize{Width: 32, Height: wn, Depth: wm},
	)
}

// emitSteelSplitKGemm records the first MLX split-K steel GEMM pass. Binding ABI:
// A=0, B=1, C_split=2, params=3.
func emitSteelSplitKGemm[S dispatchSink](sink S, pso metal.MTLComputePipelineState, a, b, split, params metal.MTLBuffer, tn, tm, partitions int, wn, wm uint) {
	sink.setPSO(pso)
	sink.setBuf(a, 0, 0)
	sink.setBuf(b, 0, 1)
	sink.setBuf(split, 0, 2)
	sink.setBuf(params, 0, 3)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(tn), Height: uint(tm), Depth: uint(partitions)},
		metal.MTLSize{Width: 32, Height: wn, Depth: wm},
	)
}

// emitSteelSplitKAccum records the second split-K pass that reduces C_split into the final D buffer.
// Binding ABI: C_split=0, D=1, partitions=2, stride=3, N=4.
func emitSteelSplitKAccum[S dispatchSink](sink S, pso metal.MTLComputePipelineState, split, out metal.MTLBuffer, partitions, stride, M, N int, bd0, bd1, bd2 uint) {
	sink.setPSO(pso)
	sink.setBuf(split, 0, 0)
	sink.setBuf(out, 0, 1)
	sink.setI32(int32(partitions), 2)
	sink.setI32(int32(stride), 3)
	sink.setI32(int32(N), 4)
	sink.dispatchThreads(
		metal.MTLSize{Width: uint(N), Height: uint(M), Depth: 1},
		metal.MTLSize{Width: bd0, Height: bd1, Depth: bd2},
	)
}

// emitUnary records a contiguous unary op over n elements. Binding ABI: in=0, out=1, count=2.
func emitUnary[S dispatchSink](sink S, pso metal.MTLComputePipelineState, in, out metal.MTLBuffer, n int) {
	sink.setPSO(pso)
	sink.setBuf(in, 0, 0)
	sink.setBuf(out, 0, 1)
	sink.setI32(int32(n), 2)
	group := min(uint(n), uint(256))
	sink.dispatchThreads(
		metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
		metal.MTLSize{Width: group, Height: 1, Depth: 1},
	)
}

// emitBinary records an element-wise binary op (vv_Add/vv_Multiply…) out = a⊙b over n elements through
// any sink: a=0, b=1, out=2 (each at its byte offset), count=3, dispatched as n threads in min(n,256)-wide
// groups. The body behind encBinaryDT (live) and the recorder's setBin. pso caller-provided (the ICB
// needs its supportIndirectCommandBuffers variant); the count routes through the sink — inline on the
// encoder, a memoised (resident) scalar buffer on the ICB.
func emitBinary[S dispatchSink](sink S, pso metal.MTLComputePipelineState, a metal.MTLBuffer, aOff uint, b metal.MTLBuffer, bOff uint, out metal.MTLBuffer, oOff uint, n int) {
	sink.setPSO(pso)
	sink.setBuf(a, aOff, 0)
	sink.setBuf(b, bOff, 1)
	sink.setBuf(out, oOff, 2)
	sink.setI32(int32(n), 3)
	g := min(uint(n), uint(256))
	sink.dispatchThreads(metal.MTLSize{Width: uint(n), Height: 1, Depth: 1}, metal.MTLSize{Width: g, Height: 1, Depth: 1})
}

// emitRope records partial-rotary RoPE (rotated width rd ≤ headDim) over nHeads heads through any sink:
// in=0, out=1, pos=2 (the per-token position buffer — a VARYING buffer the ICB rebinds, passed in), scale=3,
// headStride=4, then EITHER periods@10 + freqStride@11 (the freqs form, periods != nil) OR log2base@10 (the
// base form). 2D dispatch (rd/2 × nHeads). The body behind encRoPEBF16To / encRoPEFreqsBF16To (live) and
// the recorder's setRope. pso caller-provided — the ICB variant, and base vs freqs are different pipelines.
func emitRope[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x, out metal.MTLBuffer, inOff, outOff uint, pos, periods metal.MTLBuffer, nHeads, rd, headDim int, scale, log2base float32) {
	emitRopeAt(sink, pso, x, out, inOff, outOff, pos, 0, periods, nHeads, rd, headDim, scale, log2base)
}

func emitRopeAt[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x, out metal.MTLBuffer, inOff, outOff uint, pos metal.MTLBuffer, posOff uint, periods metal.MTLBuffer, nHeads, rd, headDim int, scale, log2base float32) {
	sink.setPSO(pso)
	sink.setBuf(x, inOff, 0)
	sink.setBuf(out, outOff, 1)
	sink.setBuf(pos, posOff, 2)
	sink.setF32(scale, 3)
	sink.setI64(int64(headDim), 4)
	if periods != nil {
		sink.setBuf(periods, 0, 10)
		sink.setI64(1, 11) // freq_stride = 1
	} else {
		sink.setF32(log2base, 10)
	}
	d0 := uint(rd / 2)
	sink.dispatchThreads(metal.MTLSize{Width: d0, Height: uint(nHeads), Depth: 1}, metal.MTLSize{Width: d0, Height: 1, Depth: 1})
}

// emitQKNormRope records the FUSED per-head QK-norm + RoPE (out = RoPE(RMSNorm(in, w))) in ONE op through
// any sink: in=0, w=1, out=2, eps=3, headDim=4, rd=5, scale=6, pos=7 (the per-token position buffer), then
// log2base=8, periods=9 (real or a dummy when periods==nil), useFreqs=10 (1/0). One threadgroup per head
// (headDim threads). The body behind encQKNormRope (live) and the recorder's setQKNormRope. `dummy` is the
// caller's bound-but-unread periods buffer for the base form (each path supplies its own — content ignored
// when useFreqs=0). pso caller-provided (ICB variant).
func emitQKNormRope[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, pos, periods, dummy metal.MTLBuffer, nHeads, headDim, rd int, eps, scale, log2base float32) {
	emitQKNormRopeAt(sink, pso, x, w, out, xOff, wOff, outOff, pos, 0, periods, dummy, nHeads, headDim, rd, eps, scale, log2base)
}

func emitQKNormRopeAt[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, pos metal.MTLBuffer, posOff uint, periods, dummy metal.MTLBuffer, nHeads, headDim, rd int, eps, scale, log2base float32) {
	sink.setPSO(pso)
	sink.setBuf(x, xOff, 0)
	sink.setBuf(w, wOff, 1)
	sink.setBuf(out, outOff, 2)
	sink.setF32(eps, 3)
	sink.setI32(int32(headDim), 4)
	sink.setI32(int32(rd), 5)
	sink.setF32(scale, 6)
	sink.setBuf(pos, posOff, 7)
	sink.setF32(log2base, 8)
	if periods != nil {
		sink.setBuf(periods, 0, 9)
		sink.setI32(1, 10)
	} else {
		sink.setBuf(dummy, 0, 9)
		sink.setI32(0, 10)
	}
	sink.dispatchThreads(metal.MTLSize{Width: uint(nHeads * headDim), Height: 1, Depth: 1}, metal.MTLSize{Width: uint(headDim), Height: 1, Depth: 1})
}

// emitSDPA records single-query single-pass scaled-dot-product attention (the sdpa_vector kernel) through
// any sink: q=0, k=1 (at kvByteOff — the sliding read offset), v=2 (kvByteOff), out=3, gqa=4, N=5,
// strides=6..9, scale=10, one threadgroup per head (1024-wide). The body behind encSDPAStrided (live) and
// the recorder's SDPA op — the op that STARTED the path-unification (the 2-pass had to be wired twice).
//
// N is the one truly per-token-VARYING scalar: the ICB binds its rebindable nBuf (rebound each token at
// replay), the live path inlines the value. So nBuf != nil binds the buffer at 5; nBuf == nil inlines n.
// Everything else is constant (gqa/strides/scale) and routes through the sink's memoised scalars — the
// recorder's gqaOf/sdpaStrideOf/sdpaScaleB buffers ARE those memoised scalars. pso caller-provided.
func emitSDPA[S dispatchSink](sink S, pso metal.MTLComputePipelineState, q, k, v, out metal.MTLBuffer, kvByteOff uint, nBuf metal.MTLBuffer, nHeads, nKVHeads, n int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) {
	emitSDPAAt(sink, pso, q, 0, k, v, out, 0, kvByteOff, nBuf, nHeads, nKVHeads, n, kHeadStride, kSeqStride, vHeadStride, vSeqStride, scale)
}

// emitSDPAAt is emitSDPA with the query and output bound at byte offsets — the batched pass's
// attention fold keeps each row's q/attn inside shared K-row slabs instead of dedicated scratch.
func emitSDPAAt[S dispatchSink](sink S, pso metal.MTLComputePipelineState, q metal.MTLBuffer, qOff uint, k, v, out metal.MTLBuffer, outOff, kvByteOff uint, nBuf metal.MTLBuffer, nHeads, nKVHeads, n int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) {
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
	sink.dispatchThreadgroups(metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1}, metal.MTLSize{Width: 1024, Height: 1, Depth: 1})
}

// emitSDPA2Pass1 records the first long-context SDPA pass. It writes one partial
// weighted-V sum plus online-softmax sum/max per (batch, kv-head, block).
func emitSDPA2Pass1[S dispatchSink](sink S, pso metal.MTLComputePipelineState, q, k, v, partials, sums, maxs metal.MTLBuffer, kvByteOff uint, batch, nHeads, nKVHeads, n, blocks int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) {
	emitSDPA2Pass1At(sink, pso, q, 0, k, v, partials, sums, maxs, kvByteOff, batch, nHeads, nKVHeads, n, blocks, kHeadStride, kSeqStride, vHeadStride, vSeqStride, scale)
}

// emitSDPA2Pass1At is emitSDPA2Pass1 with the query bound at a byte offset (the attention fold's
// slab rows). The partials/sums/maxs stay whole-buffer — they are per-dispatch scratch.
func emitSDPA2Pass1At[S dispatchSink](sink S, pso metal.MTLComputePipelineState, q metal.MTLBuffer, qOff uint, k, v, partials, sums, maxs metal.MTLBuffer, kvByteOff uint, batch, nHeads, nKVHeads, n, blocks int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) {
	sink.setPSO(pso)
	sink.setBuf(q, qOff, 0)
	sink.setBuf(k, kvByteOff, 1)
	sink.setBuf(v, kvByteOff, 2)
	sink.setBuf(partials, 0, 3)
	sink.setBuf(sums, 0, 4)
	sink.setBuf(maxs, 0, 5)
	sink.setI32(int32(n), 7)
	sink.setI64(kHeadStride, 8)
	sink.setI64(kSeqStride, 9)
	sink.setI64(vHeadStride, 10)
	sink.setI64(vSeqStride, 11)
	sink.setF32(scale, 12)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(nKVHeads), Height: uint(batch), Depth: uint(blocks)},
		metal.MTLSize{Width: 32, Height: uint(nHeads / nKVHeads), Depth: 1},
	)
}

// emitSDPA2Pass2 records the merge pass that combines per-block partials into the
// final per-head output.
func emitSDPA2Pass2[S dispatchSink](sink S, pso metal.MTLComputePipelineState, partials, sums, maxs, out metal.MTLBuffer, batch, nHeads, blocks int) {
	emitSDPA2Pass2At(sink, pso, partials, sums, maxs, out, 0, batch, nHeads, blocks)
}

// emitSDPA2Pass2At is emitSDPA2Pass2 with the output bound at a byte offset (the attention fold's
// slab rows).
func emitSDPA2Pass2At[S dispatchSink](sink S, pso metal.MTLComputePipelineState, partials, sums, maxs, out metal.MTLBuffer, outOff uint, batch, nHeads, blocks int) {
	sink.setPSO(pso)
	sink.setBuf(partials, 0, 0)
	sink.setBuf(sums, 0, 1)
	sink.setBuf(maxs, 0, 2)
	sink.setBuf(out, outOff, 3)
	sink.setI32(int32(blocks), 4)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(batch * nHeads), Height: 1, Depth: 1},
		metal.MTLSize{Width: 1024, Height: 1, Depth: 1},
	)
}

// emitQMV records a 4-bit affine quantised matvec (out = x @ Wᵀ, affine_qmv kernel) through any sink:
// wq=0, scales=1, biases=2 (each at its byte offset into the shard mmap), x=3, out=4, K=5, N=6, grid
// (1, ceil(N/8)) of (32, 2) threads. The body behind encQMVBF16 (live) and the recorder's setQMV — the
// COMMON decode matmul (e2b/12b/31b are 4-bit). K/N bind the memoised scalars the recorder's count
// buffers (kDModel/nQDimByHd/…) hold; pso caller-provided (the qmv kernel name encodes groupSize/bits).
func emitQMV[S dispatchSink](sink S, pso metal.MTLComputePipelineState, wq metal.MTLBuffer, wqOff uint, scales metal.MTLBuffer, scalesOff uint, biases metal.MTLBuffer, biasesOff uint, x, out metal.MTLBuffer, outOff uint, inDim, outDim int) {
	emitQMVAt(sink, pso, wq, wqOff, scales, scalesOff, biases, biasesOff, x, 0, out, outOff, inDim, outDim)
}

// emitQMVAt is emitQMV with the activation vector bound at a byte offset — the batched dense
// forward's rows live at byte offsets inside shared K-row buffers, and a row's quant gate reads
// its input in place.
func emitQMVAt[S dispatchSink](sink S, pso metal.MTLComputePipelineState, wq metal.MTLBuffer, wqOff uint, scales metal.MTLBuffer, scalesOff uint, biases metal.MTLBuffer, biasesOff uint, x metal.MTLBuffer, xOff uint, out metal.MTLBuffer, outOff uint, inDim, outDim int) {
	sink.setPSO(pso)
	sink.setBuf(wq, wqOff, 0)
	sink.setBuf(scales, scalesOff, 1)
	sink.setBuf(biases, biasesOff, 2)
	sink.setBuf(x, xOff, 3)
	sink.setBuf(out, outOff, 4)
	sink.setI32(int32(inDim), 5)  // K
	sink.setI32(int32(outDim), 6) // N
	const bn, bk = 8, 32
	nTgp := uint((outDim + bn - 1) / bn)
	sink.dispatchThreadgroups(metal.MTLSize{Width: 1, Height: nTgp, Depth: 1}, metal.MTLSize{Width: bk, Height: 2, Depth: 1})
}

// emitQMMT records MLX's quantised GEMM (out[M,N] = x[M,K] @ dequant(w[N,K])ᵀ, affine qmm_t
// kernel) through any sink — the BATCHED sibling of emitQMV, one weight pass for all M rows
// (the quant prompt-prefill fold). Binding ABI copied from mlx/backend/metal/quantized.cpp
// qmm(): w=0, scales=1, biases=2, x=3, out=4, K=5, N=6, M=7 (batch_0 skips the stride block);
// grid ((N+31)/32, (M+31)/32) threadgroups of (32, 2, 2). x/out bind at byte offsets so rows
// living inside shared slabs dispatch in place.
func emitQMMT[S dispatchSink](sink S, pso metal.MTLComputePipelineState, wq metal.MTLBuffer, wqOff uint, scales metal.MTLBuffer, scalesOff uint, biases metal.MTLBuffer, biasesOff uint, x metal.MTLBuffer, xOff uint, out metal.MTLBuffer, outOff uint, m, n, k int) {
	sink.setPSO(pso)
	sink.setBuf(wq, wqOff, 0)
	sink.setBuf(scales, scalesOff, 1)
	sink.setBuf(biases, biasesOff, 2)
	sink.setBuf(x, xOff, 3)
	sink.setBuf(out, outOff, 4)
	sink.setI32(int32(k), 5) // K
	sink.setI32(int32(n), 6) // N
	sink.setI32(int32(m), 7) // M
	const bm, bn = 32, 32
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint((n + bn - 1) / bn), Height: uint((m + bm - 1) / bm), Depth: 1},
		metal.MTLSize{Width: 32, Height: 2, Depth: 2},
	)
}

// emitRMSQMV records the fused BF16 input RMSNorm + quant QMV fast kernel through any sink:
// wq/scales/biases=0/1/2, x=3, out=4, K=5, N=6, normW=7, eps=8. The kernel uses the same
// qmv-fast threadgroup geometry as emitQMV, but folds the norm into the projection prologue.
func emitRMSQMV[S dispatchSink](sink S, pso metal.MTLComputePipelineState, wq metal.MTLBuffer, wqOff uint, scales metal.MTLBuffer, scalesOff uint, biases metal.MTLBuffer, biasesOff uint, x, out metal.MTLBuffer, outOff uint, normW metal.MTLBuffer, normWOff uint, inDim, outDim int, eps float32) {
	sink.setPSO(pso)
	sink.setBuf(wq, wqOff, 0)
	sink.setBuf(scales, scalesOff, 1)
	sink.setBuf(biases, biasesOff, 2)
	sink.setBuf(x, 0, 3)
	sink.setBuf(out, outOff, 4)
	sink.setI32(int32(inDim), 5)
	sink.setI32(int32(outDim), 6)
	sink.setBuf(normW, normWOff, 7)
	sink.setF32(eps, 8)
	const bn, bk = 8, 32
	nTgp := uint((outDim + bn - 1) / bn)
	sink.dispatchThreadgroups(metal.MTLSize{Width: 1, Height: nTgp, Depth: 1}, metal.MTLSize{Width: bk, Height: 2, Depth: 1})
}

// emitVProjHeadRMS records the fused Gemma V path: input RMSNorm + quantised V projection +
// per-head value RMSNorm. Binding ABI: wq=0, scales=1, biases=2, x=3, normW=4, out=5,
// inDim=6, eps=8; index 7 is intentionally unused because headDim is the threadgroup width.
func emitVProjHeadRMS[S dispatchSink](sink S, pso metal.MTLComputePipelineState, wq, scales, biases, x, normW, out metal.MTLBuffer, inDim, nKVHeads, headDim int, eps float32) {
	sink.setPSO(pso)
	sink.setBuf(wq, 0, 0)
	sink.setBuf(scales, 0, 1)
	sink.setBuf(biases, 0, 2)
	sink.setBuf(x, 0, 3)
	sink.setBuf(normW, 0, 4)
	sink.setBuf(out, 0, 5)
	sink.setI32(int32(inDim), 6)
	sink.setF32(eps, 8)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(nKVHeads), Height: 1, Depth: 1},
		metal.MTLSize{Width: uint(headDim), Height: 1, Depth: 1},
	)
}

// emitEmbedGatherQuant records the GPU dequant-gather for a token embedding row. Binding ABI:
// token=0, packed=1, scales=2, biases=3, out=4, dModel=5, groupSize=6, embedScale=7,
// rowPacked=8, rowSB=9. The token buffer is intentionally caller-provided so decode can bind the
// previous GPU argmax output without a host round-trip.
func emitEmbedGatherQuant[S dispatchSink](sink S, pso metal.MTLComputePipelineState, tokenBuf, packed, scales, biases, out metal.MTLBuffer, packedOff, scalesOff, biasesOff uint, dModel, groupSize, bits int, embedScale float32) {
	rowPacked := dModel * bits / 8
	rowSB := dModel / groupSize
	sink.setPSO(pso)
	sink.setBuf(tokenBuf, 0, 0)
	sink.setBuf(packed, packedOff, 1)
	sink.setBuf(scales, scalesOff, 2)
	sink.setBuf(biases, biasesOff, 3)
	sink.setBuf(out, 0, 4)
	sink.setI32(int32(dModel), 5)
	sink.setI32(int32(groupSize), 6)
	sink.setF32(embedScale, 7)
	sink.setI32(int32(rowPacked), 8)
	sink.setI32(int32(rowSB), 9)
	sink.dispatchThreads(
		metal.MTLSize{Width: uint(dModel), Height: 1, Depth: 1},
		metal.MTLSize{Width: uint(elemGroupTG(dModel)), Height: 1, Depth: 1},
	)
}

// emitGemv records a bf16 tiled gemv (out = mat @ vec, mat row-major outDim×inDim) through any sink:
// mat=0, vec=1, out=3, K=4, N=5, ld=6, then the single-gemv batch params (batch_ndim=1@9, batch_shape=1
// @10, vec/mat batch strides=0@11/@12), grid ceil(outDim/(bm·sm·tm)) of (32, bn, bm) threads. The body
// behind encGemvBF16To (live) and the recorder's setGemv. K/N/ld/batch bind the same memoised scalars
// the recorder's count buffers hold; pso + the bm/bn/sm/tm tiling caller-provided (both from gemvTiles).
func emitGemv[S dispatchSink](sink S, pso metal.MTLComputePipelineState, mat metal.MTLBuffer, matOff uint, vec, out metal.MTLBuffer, outOff uint, inDim, outDim, bm, bn, sm, tm int) {
	emitGemvVecAt(sink, pso, mat, matOff, vec, 0, out, outOff, inDim, outDim, bm, bn, sm, tm)
}

// emitGemvVecAt is emitGemv with the input VECTOR bound at vecOff BYTES — the batched dense
// prefill's rows live at offsets inside shared K-row buffers, so per-row consumers (the PLE
// input gate) bind the hidden at its row offset instead of copying it out first.
func emitGemvVecAt[S dispatchSink](sink S, pso metal.MTLComputePipelineState, mat metal.MTLBuffer, matOff uint, vec metal.MTLBuffer, vecOff uint, out metal.MTLBuffer, outOff uint, inDim, outDim, bm, bn, sm, tm int) {
	emitGemvBatchedVecAt(sink, pso, mat, matOff, vec, vecOff, out, outOff, inDim, outDim, 1, bm, bn, sm, tm)
}

// emitGemvBatchedVecAt is emitGemvVecAt across `batch` contiguous input rows in ONE dispatch: the
// grid's Z carries the batch through the kernel's nc0 stride branch (in_vec += z·vecStride,
// mat += z·matStride, out_vec += z·out_vec_size), so with matStride=0 every z-slice runs the
// single-row tile loop unchanged against the SHARED weight matrix — each row's bytes identical
// to `batch` separate dispatches, the weight swept once through the cache instead of `batch`
// times. vec rows contiguous at vecOff + z·inDim elements; out rows land at outOff + z·outDim.
func emitGemvBatchedVecAt[S dispatchSink](sink S, pso metal.MTLComputePipelineState, mat metal.MTLBuffer, matOff uint, vec metal.MTLBuffer, vecOff uint, out metal.MTLBuffer, outOff uint, inDim, outDim, batch, bm, bn, sm, tm int) {
	sink.setPSO(pso)
	sink.setBuf(mat, matOff, 0)
	sink.setBuf(vec, vecOff, 1)
	sink.setBuf(out, outOff, 3)
	sink.setI32(int32(inDim), 4)
	sink.setI32(int32(outDim), 5)
	sink.setI32(int32(inDim), 6) // leading dim
	sink.setI32(1, 9)            // batch_ndim
	sink.setI32(int32(batch), 10)
	vecStride := int64(0)
	if batch > 1 {
		vecStride = int64(inDim) // element stride between the contiguous input rows
	}
	sink.setI64(vecStride, 11)
	sink.setI64(0, 12) // mat batch stride: one weight matrix shared by every row
	nTgp := uint((outDim + bm*sm*tm - 1) / (bm * sm * tm))
	sink.dispatchThreadgroups(metal.MTLSize{Width: nTgp, Height: 1, Depth: uint(batch)}, metal.MTLSize{Width: 32, Height: uint(bn), Depth: uint(bm)})
}
