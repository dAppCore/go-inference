// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
)

// DecodeLayerICB records a full gemma transformer decode layer — the same 21-op
// sequence as DecodeLayer (attention half then MLP half, both residuals) — once
// into an indirect command buffer, then replays it `replays` times. This is the
// full-layer encode-bypass: a decode step's command sequence is fixed across
// tokens, so recording it once skips the per-token host re-encode of all 21 ops.
//
// It extends AttentionBlockICB (ops 0-5) with the MLP block (ops 6-20). The ICB
// rules from AttentionBlockICB hold throughout: every scalar param is a tiny
// persistent buffer (ICB commands cannot setBytes inline), each consumer command
// carries a SetBarrier so its read of a prior op's output is ordered, and the
// replay encoder marks every referenced buffer resident with UseResource. The
// gelu scalar operands are dense dFF-length bf16 constant buffers (bf16ConstBytes),
// exactly as DecodeLayer/MLPBlockBF16 build them, so the in-line gelu is identical.
//
// Same arguments and shapes as DecodeLayer; inputs/outputs raw bf16 bytes. With
// replays=1 it must equal DecodeLayer byte-for-byte — same kernels, same data,
// only the submission path differs.
func DecodeLayerICB(
	x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, kvLen, dFF int,
	base, scale float32, offset int, eps float32,
	replays int,
) ([]byte, error) {
	return DecodeLayerICBInto(nil, x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKVHeads, headDim, kvLen, dFF, base, scale, offset, eps, replays)
}

// DecodeLayerICBInto runs DecodeLayerICB and writes into caller-owned bf16 output when possible.
func DecodeLayerICBInto(
	out []byte,
	x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, kvLen, dFF int,
	base, scale float32, offset int, eps float32,
	replays int,
) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if replays < 1 {
		replays = 1
	}
	qDim := nHeads * headDim
	if len(x) != dModel*bf16Size || len(attnNormW) != dModel*bf16Size || len(mlpNormW) != dModel*bf16Size {
		return nil, core.NewError("native.DecodeLayerICB: x/attnNormW/mlpNormW must be dModel bf16 bytes")
	}
	if len(wQ) != qDim*dModel*bf16Size || len(wO) != dModel*qDim*bf16Size {
		return nil, core.NewError("native.DecodeLayerICB: wQ/wO size mismatch")
	}
	if len(wGate) != dFF*dModel*bf16Size || len(wUp) != dFF*dModel*bf16Size || len(wDown) != dModel*dFF*bf16Size {
		return nil, core.NewError("native.DecodeLayerICB: MLP weight size mismatch")
	}
	if len(kCache) != nKVHeads*kvLen*headDim*bf16Size || len(vCache) != nKVHeads*kvLen*headDim*bf16Size {
		return nil, core.NewError("native.DecodeLayerICB: kCache/vCache size mismatch")
	}
	outLen := dModel * bf16Size
	callerOut := cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}

	// ICB-capable pipelines. gemv tiles depend on (inDim, outDim) so there are
	// four distinct gemv PSOs: Q (dModel→qDim), O (qDim→dModel), gate/up
	// (dModel→dFF, shared) and down (dFF→dModel).
	rmsPSO, err := pipelineForICB(rmsKernelBF16(dModel))
	if err != nil {
		return nil, err
	}
	bmQ, bnQ, smQ, snQ, tmQ, tnQ := gemvTiles(dModel, qDim)
	gemvQPSO, err := pipelineForICB(gemvKernelName("bfloat16", bmQ, bnQ, smQ, snQ, tmQ, tnQ))
	if err != nil {
		return nil, err
	}
	bmO, bnO, smO, snO, tmO, tnO := gemvTiles(qDim, dModel)
	gemvOPSO, err := pipelineForICB(gemvKernelName("bfloat16", bmO, bnO, smO, snO, tmO, tnO))
	if err != nil {
		return nil, err
	}
	bmF, bnF, smF, snF, tmF, tnF := gemvTiles(dModel, dFF)
	gemvFPSO, err := pipelineForICB(gemvKernelName("bfloat16", bmF, bnF, smF, snF, tmF, tnF))
	if err != nil {
		return nil, err
	}
	bmD, bnD, smD, snD, tmD, tnD := gemvTiles(dFF, dModel)
	gemvDPSO, err := pipelineForICB(gemvKernelName("bfloat16", bmD, bnD, smD, snD, tmD, tnD))
	if err != nil {
		return nil, err
	}
	ropePSO, err := ropePipelineICB(false)
	if err != nil {
		return nil, err
	}
	sdpaPSO, err := sdpaVectorPipelineICBForHeadDim(headDim)
	if err != nil {
		return nil, err
	}
	addPSO, err := pipelineForICB("vv_Addbfloat16")
	if err != nil {
		return nil, err
	}
	hasFusedGELU := gpuHasGeluKernel()
	var mulPSO, tanhPSO metal.MTLComputePipelineState
	var geluICBPSO metal.MTLComputePipelineState
	if hasFusedGELU {
		if geluICBPSO, err = geluPipelineICB(); err != nil {
			return nil, err
		}
	} else {
		mulPSO, err = pipelineForICB("vv_Multiplybfloat16")
		if err != nil {
			return nil, err
		}
		tanhPSO, err = pipelineForICB("v_Tanhbfloat16bfloat16")
		if err != nil {
			return nil, err
		}
	}
	// fused gelu is one command (cmd 9) vs the composed chain's ten (cmd 9-18), so the down-proj +
	// residual shift to 10/11 and the layer records 12 commands instead of 21.
	nCmds, dpIdx := 21, 19
	if hasFusedGELU {
		nCmds, dpIdx = 12, 10
	}

	var encErr error
	withAutoreleasePool(func() {
		ioScratch, err := getQMVBF16Scratch(dModel, dModel)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVBF16Scratch(ioScratch)
		xBuf, outBuf, err := ioScratch.buffers(x)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if callerOut {
			if tmp, ok := ioScratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}
		// --- data buffers ---
		anwBuf, mnwBuf := residentBytes(attnNormW), residentBytes(mlpNormW)
		wqBuf, woBuf := residentBytes(wQ), residentBytes(wO)
		kvScratch, err := getAttentionBlockKVScratch(len(kCache), len(vCache))
		if err != nil {
			encErr = err
			return
		}
		defer putAttentionBlockKVScratch(kvScratch)
		kBuf, vBuf, err := kvScratch.buffers(kCache, vCache)
		if err != nil {
			encErr = err
			return
		}
		wgBuf, wuBuf, wdBuf := residentBytes(wGate), residentBytes(wUp), residentBytes(wDown)

		asc := getAttnScratch(dModel, qDim, nKVHeads*headDim, nHeads, 0)
		defer putAttnScratch(asc)
		msc := getMLPScratch(dModel, dFF)
		defer putMLPScratch(msc)
		layerScratch := getDecodeLayerResidualScratch(dModel)
		defer putDecodeLayerResidualScratch(layerScratch)

		// attention intermediates
		attnNormed := asc.normed
		q, qr, attn := asc.q, asc.qr, asc.attn
		attnOut, h := asc.attnOut, layerScratch.h
		// mlp intermediates
		mlpNormed := msc.mlpNormed
		gate, up := msc.gate, msc.up
		gated := msc.gated
		down := msc.down
		var c044, c079, c1c, c05 metal.MTLBuffer
		var x2, x3, x3s, inner metal.MTLBuffer
		var scaled, tnh, onePlus, halfG metal.MTLBuffer
		var gelu metal.MTLBuffer
		if !hasFusedGELU {
			// gelu scalar operands as dense dFF-length bf16 constant buffers.
			c044, c079, c1c, c05 = msc.c044, msc.c079, msc.c1, msc.c05
			x2, x3, x3s, inner = msc.x2, msc.x3, msc.x3s, msc.inner
			scaled, tnh, onePlus, halfG = msc.scaled, msc.tnh, msc.onePlus, msc.halfG
			gelu = msc.gelu
		}

		// --- scalar buffers ---
		offBuf := scalarI32(int32(offset))
		epsBuf, axisBuf, wsBuf := scalarF32(eps), scalarI32(int32(dModel)), scalarI32(1)
		// gemv scalars (inDim, outDim, ld vary per op; batch scalars are shared)
		qInB, qOutB, qLdB := scalarI32(int32(dModel)), scalarI32(int32(qDim)), scalarI32(int32(dModel))
		oInB, oOutB, oLdB := scalarI32(int32(qDim)), scalarI32(int32(dModel)), scalarI32(int32(qDim))
		fInB, fOutB, fLdB := scalarI32(int32(dModel)), scalarI32(int32(dFF)), scalarI32(int32(dModel))
		dInB, dOutB, dLdB := scalarI32(int32(dFF)), scalarI32(int32(dModel)), scalarI32(int32(dFF))
		bndB, bshB, vsB, msB := scalarI32(1), scalarI32(1), scalarI64(0), scalarI64(0)
		// rope scalars
		ropeScaleB := scalarF32(scale)
		ropeMatB := scalarI64(int64(headDim))
		ropeBaseB := scalarF32(float32(math.Log2(float64(base))))
		// sdpa scalars
		gqaB, nB := scalarI32(int32(nHeads/nKVHeads)), scalarI32(int32(kvLen))
		khsB, kssB := scalarI64(int64(kvLen*headDim)), scalarI64(int64(headDim))
		vhsB, vssB := scalarI64(int64(kvLen*headDim)), scalarI64(int64(headDim))
		sdpaScaleB := scalarF32(scale)
		// element-wise counts: dModel for the two residual adds, dFF for the
		// MLP element-wise chain. tanh's count is bound as a buffer at index 2
		// (the encode path uses setBytes there; the ICB cannot, so it is a buffer).
		addModelB := scalarI32(int32(dModel))
		cntFFB := scalarI32(int32(dFF))
		tanhCntB := scalarI32(int32(dFF))

		resident := []metal.MTLResource{
			xBuf, anwBuf, mnwBuf, wqBuf, woBuf, kBuf, vBuf, wgBuf, wuBuf, wdBuf,
			attnNormed, q, qr, attn, attnOut, h,
			mlpNormed, gate, up, gated, down, outBuf,
			offBuf, epsBuf, axisBuf, wsBuf,
			qInB, qOutB, qLdB, oInB, oOutB, oLdB, fInB, fOutB, fLdB, dInB, dOutB, dLdB,
			bndB, bshB, vsB, msB,
			ropeScaleB, ropeMatB, ropeBaseB,
			gqaB, nB, khsB, kssB, vhsB, vssB, sdpaScaleB,
			addModelB, cntFFB, tanhCntB,
		}
		if !hasFusedGELU {
			resident = append(resident,
				c044, c079, c1c, c05,
				x2, x3, x3s, inner, scaled, tnh, onePlus, halfG, gelu,
			)
		}

		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch | metal.MTLIndirectCommandTypeConcurrentDispatchThreads)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, uint(nCmds), metal.MTLResourceStorageModeShared)

		rmsTG := rmsThreadgroup(dModel, rmsPSO)
		log2base := float32(math.Log2(float64(base)))

		// helper closures so each op's binding matches its encode-form exactly.
		setRMS := func(c metal.MTLIndirectComputeCommand, in, w, o metal.MTLBuffer) {
			emitRMSNorm(fastICBSink{c}, rmsPSO, in, w, o, 0, dModel, eps, rmsTG)
		}
		setGemv := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, mat, vec, o metal.MTLBuffer, inDim, outDim, bm, bn, sm, tm int) {
			emitGemv(fastICBSink{c}, pso, mat, 0, vec, o, 0, inDim, outDim, bm, bn, sm, tm)
		}
		setBinary := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, a, b, o, cntB metal.MTLBuffer, n int) {
			emitBinary(fastICBSink{c}, pso, a, 0, b, 0, o, 0, n)
		}

		// ===== attention half (ops 0-5): h = x + Wo·sdpa(rope(Wq·rms(x))) =====
		// 0: rms x -> attnNormed
		setRMS(indirectComputeCommandAtIndexFast(icb, 0), xBuf, anwBuf, attnNormed)

		// 1: gemv Wq @ attnNormed -> q  (dModel -> qDim)
		c := indirectComputeCommandAtIndexFast(icb, 1)
		setICBBarrier(c)
		setGemv(c, gemvQPSO, wqBuf, attnNormed, q, dModel, qDim, bmQ, bnQ, smQ, tmQ)

		// 2: rope q -> qr
		c = indirectComputeCommandAtIndexFast(icb, 2)
		setICBBarrier(c)
		emitRope(fastICBSink{c}, ropePSO, q, qr, 0, 0, offBuf, nil, nHeads, headDim, headDim, scale, log2base)

		// 3: sdpa qr, k, v -> attn
		c = indirectComputeCommandAtIndexFast(icb, 3)
		setICBBarrier(c)
		emitSDPA(fastICBSink{c}, sdpaPSO, qr, kBuf, vBuf, attn, 0, nB, nHeads, nKVHeads, 0, int64(kvLen*headDim), int64(headDim), int64(kvLen*headDim), int64(headDim), scale)

		// 4: gemv Wo @ attn -> attnOut  (qDim -> dModel)
		c = indirectComputeCommandAtIndexFast(icb, 4)
		setICBBarrier(c)
		setGemv(c, gemvOPSO, woBuf, attn, attnOut, qDim, dModel, bmO, bnO, smO, tmO)

		// 5: add x + attnOut -> h
		c = indirectComputeCommandAtIndexFast(icb, 5)
		setICBBarrier(c)
		setBinary(c, addPSO, xBuf, attnOut, h, addModelB, dModel)

		// ===== MLP half (ops 6-20): out = h + Wdown·(gelu(Wgate·rms(h))·(Wup·rms(h))) =====
		// 6: rms h -> mlpNormed
		c = indirectComputeCommandAtIndexFast(icb, 6)
		setICBBarrier(c)
		setRMS(c, h, mnwBuf, mlpNormed)

		// 7: gemv Wgate @ mlpNormed -> gate  (dModel -> dFF)
		c = indirectComputeCommandAtIndexFast(icb, 7)
		setICBBarrier(c)
		setGemv(c, gemvFPSO, wgBuf, mlpNormed, gate, dModel, dFF, bmF, bnF, smF, tmF)

		// 8: gemv Wup @ mlpNormed -> up  (dModel -> dFF)
		c = indirectComputeCommandAtIndexFast(icb, 8)
		setICBBarrier(c)
		setGemv(c, gemvFPSO, wuBuf, mlpNormed, up, dModel, dFF, bmF, bnF, smF, tmF)

		// gelu(gate)·up — fused kernel (one command, cmd 9) when loaded; composed chain (cmd 9-18) otherwise
		if hasFusedGELU {
			c = indirectComputeCommandAtIndexFast(icb, 9)
			setICBBarrier(c)
			emitBinary(fastICBSink{c}, geluICBPSO, gate, 0, up, 0, gated, 0, dFF)
		} else {
			// gelu_approx(gate): x2=g·g; x3=x2·g; x3s=0.044715·x3; inner=g+x3s;
			//                    scaled=0.7978…·inner; tnh=tanh(scaled);
			//                    onePlus=tnh+1; halfG=0.5·g; gelu=halfG·onePlus
			c = indirectComputeCommandAtIndexFast(icb, 9)
			setICBBarrier(c)
			setBinary(c, mulPSO, gate, gate, x2, cntFFB, dFF)
			c = indirectComputeCommandAtIndexFast(icb, 10)
			setICBBarrier(c)
			setBinary(c, mulPSO, x2, gate, x3, cntFFB, dFF)
			c = indirectComputeCommandAtIndexFast(icb, 11)
			setICBBarrier(c)
			setBinary(c, mulPSO, x3, c044, x3s, cntFFB, dFF)
			c = indirectComputeCommandAtIndexFast(icb, 12)
			setICBBarrier(c)
			setBinary(c, addPSO, gate, x3s, inner, cntFFB, dFF)
			c = indirectComputeCommandAtIndexFast(icb, 13)
			setICBBarrier(c)
			setBinary(c, mulPSO, inner, c079, scaled, cntFFB, dFF)
			c = indirectComputeCommandAtIndexFast(icb, 14)
			setICBBarrier(c)
			emitUnary(fastICBSink{c}, tanhPSO, scaled, tnh, dFF)
			c = indirectComputeCommandAtIndexFast(icb, 15)
			setICBBarrier(c)
			setBinary(c, addPSO, tnh, c1c, onePlus, cntFFB, dFF)
			c = indirectComputeCommandAtIndexFast(icb, 16)
			setICBBarrier(c)
			setBinary(c, mulPSO, gate, c05, halfG, cntFFB, dFF)
			c = indirectComputeCommandAtIndexFast(icb, 17)
			setICBBarrier(c)
			setBinary(c, mulPSO, halfG, onePlus, gelu, cntFFB, dFF)
			c = indirectComputeCommandAtIndexFast(icb, 18)
			setICBBarrier(c)
			setBinary(c, mulPSO, gelu, up, gated, cntFFB, dFF)
		}

		// down-proj: gemv Wdown @ gated -> down  (dFF -> dModel) — cmd dpIdx (10 fused / 19 composed)
		c = indirectComputeCommandAtIndexFast(icb, uint(dpIdx))
		setICBBarrier(c)
		setGemv(c, gemvDPSO, wdBuf, gated, down, dFF, dModel, bmD, bnD, smD, tmD)

		// residual: add h + down -> outBuf — cmd dpIdx+1
		c = indirectComputeCommandAtIndexFast(icb, uint(dpIdx+1))
		setICBBarrier(c)
		setBinary(c, addPSO, h, down, outBuf, addModelB, dModel)

		rng := foundation.NSRange{Location: 0, Length: uint(nCmds)}
		ioScratch.residentIDs = resourceIDsForFastUse(ioScratch.residentIDs, resident)
		residentIDs := ioScratch.residentIDs
		for r := 0; r < replays; r++ {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			useResourcesIDsFast(enc, resident, residentIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			executeCommandsInBufferWithRangeFast(enc, icb, rng)
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
		}
		if !directOut {
			copy(out, ioScratch.out.bytes[:len(out)])
		}
	})
	return out, encErr
}

// DecodeTokenICB records a full nLayers-deep decode TOKEN — nLayers copies of the
// 21-op DecodeLayer sequence chained through a residual-stream ping-pong — once
// into one indirect command buffer (21*nLayers commands), then replays the whole
// stack with a SINGLE executeCommandsInBuffer + commit + wait per token, `replays`
// times. This is the per-token analogue of DecodeLayerICB and the un-diluted
// encode-bypass headline: a real decode step submits its entire layer stack at
// once, so the one commit+wait is amortised across all the layers and the A/B vs
// tokenReEncode(reps) reflects the true per-token host saving (DecodeLayerICB paid
// a commit+wait per layer, diluting the ratio with GPU+submit time).
//
// Same per-layer ICB rules as DecodeLayerICB hold: every scalar is a persistent
// buffer, every command carries a SetBarrier so its reads of prior writes are
// ordered (here that also serialises layer L+1's input rms after layer L's output
// add) EXCEPT the very first command, and the replay encoder marks every buffer
// resident. Layers share weights/scratch/KV (the host encode cost per command is
// independent of which buffer it binds; keeps it AX-11-light); only the residual
// stream ping-pongs between two model-dim buffers. With nLayers=1, replays=1 it is
// DecodeLayerICB; chained it equals nLayers applications of DecodeLayer — both
// gated byte-for-byte in the tests. Inputs/outputs raw bf16 bytes.
func DecodeTokenICB(
	x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, kvLen, dFF, nLayers int,
	base, scale float32, offset int, eps float32,
	replays int,
) ([]byte, error) {
	return DecodeTokenICBInto(nil, x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKVHeads, headDim, kvLen, dFF, nLayers, base, scale, offset, eps, replays)
}

// DecodeTokenICBInto runs DecodeTokenICB and writes into caller-owned bf16 output when possible.
func DecodeTokenICBInto(
	out []byte,
	x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, kvLen, dFF, nLayers int,
	base, scale float32, offset int, eps float32,
	replays int,
) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if nLayers < 1 {
		nLayers = 1
	}
	if replays < 1 {
		replays = 1
	}
	qDim := nHeads * headDim
	if len(x) != dModel*bf16Size || len(attnNormW) != dModel*bf16Size || len(mlpNormW) != dModel*bf16Size {
		return nil, core.NewError("native.DecodeTokenICB: x/attnNormW/mlpNormW must be dModel bf16 bytes")
	}
	if len(wQ) != qDim*dModel*bf16Size || len(wO) != dModel*qDim*bf16Size {
		return nil, core.NewError("native.DecodeTokenICB: wQ/wO size mismatch")
	}
	if len(wGate) != dFF*dModel*bf16Size || len(wUp) != dFF*dModel*bf16Size || len(wDown) != dModel*dFF*bf16Size {
		return nil, core.NewError("native.DecodeTokenICB: MLP weight size mismatch")
	}
	if len(kCache) != nKVHeads*kvLen*headDim*bf16Size || len(vCache) != nKVHeads*kvLen*headDim*bf16Size {
		return nil, core.NewError("native.DecodeTokenICB: kCache/vCache size mismatch")
	}
	outLen := dModel * bf16Size
	callerOut := cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}

	rmsPSO, err := pipelineForICB(rmsKernelBF16(dModel))
	if err != nil {
		return nil, err
	}
	bmQ, bnQ, smQ, snQ, tmQ, tnQ := gemvTiles(dModel, qDim)
	gemvQPSO, err := pipelineForICB(gemvKernelName("bfloat16", bmQ, bnQ, smQ, snQ, tmQ, tnQ))
	if err != nil {
		return nil, err
	}
	bmO, bnO, smO, snO, tmO, tnO := gemvTiles(qDim, dModel)
	gemvOPSO, err := pipelineForICB(gemvKernelName("bfloat16", bmO, bnO, smO, snO, tmO, tnO))
	if err != nil {
		return nil, err
	}
	bmF, bnF, smF, snF, tmF, tnF := gemvTiles(dModel, dFF)
	gemvFPSO, err := pipelineForICB(gemvKernelName("bfloat16", bmF, bnF, smF, snF, tmF, tnF))
	if err != nil {
		return nil, err
	}
	bmD, bnD, smD, snD, tmD, tnD := gemvTiles(dFF, dModel)
	gemvDPSO, err := pipelineForICB(gemvKernelName("bfloat16", bmD, bnD, smD, snD, tmD, tnD))
	if err != nil {
		return nil, err
	}
	ropePSO, err := ropePipelineICB(false)
	if err != nil {
		return nil, err
	}
	sdpaPSO, err := sdpaVectorPipelineICBForHeadDim(headDim)
	if err != nil {
		return nil, err
	}
	addPSO, err := pipelineForICB("vv_Addbfloat16")
	if err != nil {
		return nil, err
	}
	hasFusedGELU := gpuHasGeluKernel()
	var mulPSO, tanhPSO metal.MTLComputePipelineState
	var geluICBPSO metal.MTLComputePipelineState
	if hasFusedGELU {
		if geluICBPSO, err = geluPipelineICB(); err != nil {
			return nil, err
		}
	} else {
		mulPSO, err = pipelineForICB("vv_Multiplybfloat16")
		if err != nil {
			return nil, err
		}
		tanhPSO, err = pipelineForICB("v_Tanhbfloat16bfloat16")
		if err != nil {
			return nil, err
		}
	}
	// fused gelu is one command (cmd 9) vs the composed chain's ten; the down-proj + residual shift
	// to 10/11 and a layer records 12 commands instead of 21.
	opsPerLayer, dpIdx := 21, 19
	if hasFusedGELU {
		opsPerLayer, dpIdx = 12, 10
	}

	var encErr error
	withAutoreleasePool(func() {
		ioScratch, err := getQMVBF16Scratch(dModel, dModel)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVBF16Scratch(ioScratch)
		xA, xB, err := ioScratch.buffers(x)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		// Odd layer counts finish in xB; even counts finish in xA, which must be
		// seeded with the input token, so only xB can safely alias caller output.
		if callerOut && nLayers%2 == 1 {
			if tmp, ok := ioScratch.outputView(out); ok {
				xB = tmp
				directOut = true
			}
		}
		// --- weight / KV / gelu-const data buffers (shared across layers) ---
		anwBuf, mnwBuf := residentBytes(attnNormW), residentBytes(mlpNormW)
		wqBuf, woBuf := residentBytes(wQ), residentBytes(wO)
		kvScratch, err := getAttentionBlockKVScratch(len(kCache), len(vCache))
		if err != nil {
			encErr = err
			return
		}
		defer putAttentionBlockKVScratch(kvScratch)
		kBuf, vBuf, err := kvScratch.buffers(kCache, vCache)
		if err != nil {
			encErr = err
			return
		}
		wgBuf, wuBuf, wdBuf := residentBytes(wGate), residentBytes(wUp), residentBytes(wDown)

		// residual-stream ping-pong: xA seeded with the token input, xB scratch.

		asc := getAttnScratch(dModel, qDim, nKVHeads*headDim, nHeads, 0)
		defer putAttnScratch(asc)
		msc := getMLPScratch(dModel, dFF)
		defer putMLPScratch(msc)
		layerScratch := getDecodeLayerResidualScratch(dModel)
		defer putDecodeLayerResidualScratch(layerScratch)

		// shared per-layer intermediates
		attnNormed := asc.normed
		q, qr, attn := asc.q, asc.qr, asc.attn
		attnOut, h := asc.attnOut, layerScratch.h
		mlpNormed := msc.mlpNormed
		gate, up := msc.gate, msc.up
		gated := msc.gated
		down := msc.down
		var c044, c079, c1c, c05 metal.MTLBuffer
		var x2, x3, x3s, inner metal.MTLBuffer
		var scaled, tnh, onePlus, halfG metal.MTLBuffer
		var gelu metal.MTLBuffer
		if !hasFusedGELU {
			c044, c079, c1c, c05 = msc.c044, msc.c079, msc.c1, msc.c05
			x2, x3, x3s, inner = msc.x2, msc.x3, msc.x3s, msc.inner
			scaled, tnh, onePlus, halfG = msc.scaled, msc.tnh, msc.onePlus, msc.halfG
			gelu = msc.gelu
		}

		// --- scalar buffers (shared) ---
		offBuf := scalarI32(int32(offset))
		epsBuf, axisBuf, wsBuf := scalarF32(eps), scalarI32(int32(dModel)), scalarI32(1)
		qInB, qOutB, qLdB := scalarI32(int32(dModel)), scalarI32(int32(qDim)), scalarI32(int32(dModel))
		oInB, oOutB, oLdB := scalarI32(int32(qDim)), scalarI32(int32(dModel)), scalarI32(int32(qDim))
		fInB, fOutB, fLdB := scalarI32(int32(dModel)), scalarI32(int32(dFF)), scalarI32(int32(dModel))
		dInB, dOutB, dLdB := scalarI32(int32(dFF)), scalarI32(int32(dModel)), scalarI32(int32(dFF))
		bndB, bshB, vsB, msB := scalarI32(1), scalarI32(1), scalarI64(0), scalarI64(0)
		ropeScaleB := scalarF32(scale)
		ropeMatB := scalarI64(int64(headDim))
		ropeBaseB := scalarF32(float32(math.Log2(float64(base))))
		gqaB, nB := scalarI32(int32(nHeads/nKVHeads)), scalarI32(int32(kvLen))
		khsB, kssB := scalarI64(int64(kvLen*headDim)), scalarI64(int64(headDim))
		vhsB, vssB := scalarI64(int64(kvLen*headDim)), scalarI64(int64(headDim))
		sdpaScaleB := scalarF32(scale)
		addModelB := scalarI32(int32(dModel))
		cntFFB := scalarI32(int32(dFF))
		tanhCntB := scalarI32(int32(dFF))

		resident := []metal.MTLResource{
			xA, xB, anwBuf, mnwBuf, wqBuf, woBuf, kBuf, vBuf, wgBuf, wuBuf, wdBuf,
			attnNormed, q, qr, attn, attnOut, h,
			mlpNormed, gate, up, gated, down,
			offBuf, epsBuf, axisBuf, wsBuf,
			qInB, qOutB, qLdB, oInB, oOutB, oLdB, fInB, fOutB, fLdB, dInB, dOutB, dLdB,
			bndB, bshB, vsB, msB,
			ropeScaleB, ropeMatB, ropeBaseB,
			gqaB, nB, khsB, kssB, vhsB, vssB, sdpaScaleB,
			addModelB, cntFFB, tanhCntB,
		}
		if !hasFusedGELU {
			resident = append(resident,
				c044, c079, c1c, c05,
				x2, x3, x3s, inner, scaled, tnh, onePlus, halfG, gelu,
			)
		}

		total := opsPerLayer * nLayers
		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch | metal.MTLIndirectCommandTypeConcurrentDispatchThreads)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, uint(total), metal.MTLResourceStorageModeShared)

		rmsTG := rmsThreadgroup(dModel, rmsPSO)
		log2base := float32(math.Log2(float64(base)))
		setRMS := func(c metal.MTLIndirectComputeCommand, in, w, o metal.MTLBuffer) {
			emitRMSNorm(fastICBSink{c}, rmsPSO, in, w, o, 0, dModel, eps, rmsTG)
		}
		setGemv := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, mat, vec, o metal.MTLBuffer, inDim, outDim, bm, bn, sm, tm int) {
			emitGemv(fastICBSink{c}, pso, mat, 0, vec, o, 0, inDim, outDim, bm, bn, sm, tm)
		}
		setBinary := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, a, b, o, cntB metal.MTLBuffer, n int) {
			emitBinary(fastICBSink{c}, pso, a, 0, b, 0, o, 0, n)
		}

		// recordLayer writes the 21 commands of one layer at [base, base+20],
		// reading inBuf, writing outBuf. Every command barriers except global 0 —
		// which also orders this layer's input rms after the previous layer's
		// output add (the shared scratch is reused, so the stack must serialise).
		recordLayer := func(base int, inBuf, outBuf metal.MTLBuffer) {
			cmd := func(op int) metal.MTLIndirectComputeCommand {
				c := indirectComputeCommandAtIndexFast(icb, uint(base+op))
				if base+op != 0 {
					setICBBarrier(c)
				}
				return c
			}
			// ===== attention half: h = in + Wo·sdpa(rope(Wq·rms(in))) =====
			setRMS(cmd(0), inBuf, anwBuf, attnNormed)
			setGemv(cmd(1), gemvQPSO, wqBuf, attnNormed, q, dModel, qDim, bmQ, bnQ, smQ, tmQ)
			c := cmd(2)
			emitRope(fastICBSink{c}, ropePSO, q, qr, 0, 0, offBuf, nil, nHeads, headDim, headDim, scale, log2base)

			c = cmd(3)
			emitSDPA(fastICBSink{c}, sdpaPSO, qr, kBuf, vBuf, attn, 0, nB, nHeads, nKVHeads, 0, int64(kvLen*headDim), int64(headDim), int64(kvLen*headDim), int64(headDim), scale)

			setGemv(cmd(4), gemvOPSO, woBuf, attn, attnOut, qDim, dModel, bmO, bnO, smO, tmO)
			setBinary(cmd(5), addPSO, inBuf, attnOut, h, addModelB, dModel)

			// ===== MLP half: out = h + Wdown·(gelu(Wgate·rms(h))·(Wup·rms(h))) =====
			setRMS(cmd(6), h, mnwBuf, mlpNormed)
			setGemv(cmd(7), gemvFPSO, wgBuf, mlpNormed, gate, dModel, dFF, bmF, bnF, smF, tmF)
			setGemv(cmd(8), gemvFPSO, wuBuf, mlpNormed, up, dModel, dFF, bmF, bnF, smF, tmF)
			if hasFusedGELU {
				cg := cmd(9) // fused gelu(gate)·up -> gated (cntFFB = dFF as the n buffer)
				emitBinary(fastICBSink{cg}, geluICBPSO, gate, 0, up, 0, gated, 0, dFF)
			} else {
				setBinary(cmd(9), mulPSO, gate, gate, x2, cntFFB, dFF)
				setBinary(cmd(10), mulPSO, x2, gate, x3, cntFFB, dFF)
				setBinary(cmd(11), mulPSO, x3, c044, x3s, cntFFB, dFF)
				setBinary(cmd(12), addPSO, gate, x3s, inner, cntFFB, dFF)
				setBinary(cmd(13), mulPSO, inner, c079, scaled, cntFFB, dFF)
				c = cmd(14)
				emitUnary(fastICBSink{c}, tanhPSO, scaled, tnh, dFF)
				setBinary(cmd(15), addPSO, tnh, c1c, onePlus, cntFFB, dFF)
				setBinary(cmd(16), mulPSO, gate, c05, halfG, cntFFB, dFF)
				setBinary(cmd(17), mulPSO, halfG, onePlus, gelu, cntFFB, dFF)
				setBinary(cmd(18), mulPSO, gelu, up, gated, cntFFB, dFF)
			}
			setGemv(cmd(dpIdx), gemvDPSO, wdBuf, gated, down, dFF, dModel, bmD, bnD, smD, tmD)
			setBinary(cmd(dpIdx+1), addPSO, h, down, outBuf, addModelB, dModel)
		}

		in, outB := xA, xB
		for L := 0; L < nLayers; L++ {
			recordLayer(opsPerLayer*L, in, outB)
			in, outB = outB, in
		}
		lastOut := in // after the final swap, `in` is the last layer's output

		rng := foundation.NSRange{Location: 0, Length: uint(total)}
		ioScratch.residentIDs = resourceIDsForFastUse(ioScratch.residentIDs, resident)
		residentIDs := ioScratch.residentIDs
		for r := 0; r < replays; r++ {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			useResourcesIDsFast(enc, resident, residentIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			executeCommandsInBufferWithRangeFast(enc, icb, rng)
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
		}
		if !directOut {
			if lastOut.GetID() == ioScratch.x.buf.GetID() {
				copy(out, ioScratch.x.bytes[:len(out)])
			} else {
				copy(out, ioScratch.out.bytes[:len(out)])
			}
		}
	})
	return out, encErr
}
