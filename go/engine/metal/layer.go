// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"sync"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

type decodeLayerResidualScratch struct {
	dModel int
	h      metal.MTLBuffer
}

type bf16GemvPlan struct {
	pso        metal.MTLComputePipelineState
	bm, bn, sm int
	tm         int
}

var decodeLayerResidualScratchPools sync.Map

type decodeLayerResidualScratchPool struct {
	core.Pool[*decodeLayerResidualScratch]
}

func decodeLayerResidualScratchPoolFor(dModel int) *decodeLayerResidualScratchPool {
	if v, ok := decodeLayerResidualScratchPools.Load(dModel); ok {
		return v.(*decodeLayerResidualScratchPool)
	}
	pool := new(decodeLayerResidualScratchPool)
	if v, loaded := decodeLayerResidualScratchPools.LoadOrStore(dModel, pool); loaded {
		return v.(*decodeLayerResidualScratchPool)
	}
	return pool
}

func newBF16GemvPlan(outDim, inDim int) (bf16GemvPlan, error) {
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	pso, err := pipelineFor(gemvKernelName("bfloat16", bm, bn, sm, sn, tm, tn))
	if err != nil {
		return bf16GemvPlan{}, err
	}
	return bf16GemvPlan{pso: pso, bm: bm, bn: bn, sm: sm, tm: tm}, nil
}

func emitBF16GemvPlan[S dispatchSink](sink S, plan bf16GemvPlan, mat, vec, out metal.MTLBuffer, inDim, outDim int) {
	emitGemv(sink, plan.pso, mat, 0, vec, out, 0, inDim, outDim, plan.bm, plan.bn, plan.sm, plan.tm)
}

func getDecodeLayerResidualScratch(dModel int) *decodeLayerResidualScratch {
	pool := decodeLayerResidualScratchPoolFor(dModel)
	if sc := pool.Get(); sc != nil {
		if sc.dModel == dModel && sc.h != nil {
			return sc
		}
	}
	return &decodeLayerResidualScratch{dModel: dModel, h: scratchBF16(dModel)}
}

func putDecodeLayerResidualScratch(sc *decodeLayerResidualScratch) {
	if sc != nil && sc.dModel > 0 && sc.h != nil {
		decodeLayerResidualScratchPoolFor(sc.dModel).Put(sc)
	}
}

// DecodeLayer runs a full gemma transformer decode layer on-device, in bf16, in
// ONE command buffer — the attention block feeding the MLP block, each with its
// residual, every intermediate resident:
//
//	h   = x + Wo·sdpa(rope(Wq·rms(x)), kCache, vCache)      // attention + residual
//	out = h + Wdown·( gelu(Wgate·rms(h)) · (Wup·rms(h)) )   // MLP + residual
//
// ~21 dispatches, one commit, no host round-trip. Attention attends over a given
// KV cache (the cache-write half is a follow-up). wQ is (nHeads·headDim × dModel),
// wO is (dModel × nHeads·headDim), wGate/wUp are (dFF × dModel), wDown is (dModel
// × dFF); kCache/vCache are (nKVHeads, kvLen, headDim). All inputs/outputs raw
// bf16 bytes. The result equals AttentionBlock then MLPBlockBF16 run separately —
// proven in the tests. This is a complete transformer layer on the no-cgo path.
func DecodeLayer(
	x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, kvLen, dFF int,
	base, scale float32, offset int, eps float32,
) ([]byte, error) {
	return decodeLayerInto(nil, x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKVHeads, headDim, kvLen, dFF, base, scale, offset, eps, false)
}

// DecodeLayerInto is DecodeLayer with caller-owned output storage. If out has
// enough capacity, the final MLP residual add writes directly into out through a
// pinned no-copy Metal buffer; otherwise a correctly sized output is allocated
// and returned.
func DecodeLayerInto(
	out []byte,
	x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, kvLen, dFF int,
	base, scale float32, offset int, eps float32,
) ([]byte, error) {
	return decodeLayerInto(out, x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKVHeads, headDim, kvLen, dFF, base, scale, offset, eps, true)
}

func decodeLayerInto(
	out []byte,
	x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, kvLen, dFF int,
	base, scale float32, offset int, eps float32,
	useCallerOut bool,
) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	qDim := nHeads * headDim
	if len(x) != dModel*bf16Size || len(attnNormW) != dModel*bf16Size || len(mlpNormW) != dModel*bf16Size {
		return nil, core.NewError("native.DecodeLayer: x/attnNormW/mlpNormW must be dModel bf16 bytes")
	}
	if len(wQ) != qDim*dModel*bf16Size || len(wO) != dModel*qDim*bf16Size {
		return nil, core.NewError("native.DecodeLayer: wQ/wO size mismatch")
	}
	if len(wGate) != dFF*dModel*bf16Size || len(wUp) != dFF*dModel*bf16Size || len(wDown) != dModel*dFF*bf16Size {
		return nil, core.NewError("native.DecodeLayer: MLP weight size mismatch")
	}
	if len(kCache) != nKVHeads*kvLen*headDim*bf16Size || len(vCache) != nKVHeads*kvLen*headDim*bf16Size {
		return nil, core.NewError("native.DecodeLayer: kCache/vCache size mismatch")
	}

	outLen := dModel * bf16Size
	callerOut := useCallerOut && cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
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
		// inputs
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
		offBuf := scalarI32(int32(offset))

		asc := getAttnScratch(dModel, qDim, nKVHeads*headDim, nHeads, 0)
		defer putAttnScratch(asc)
		msc := getMLPScratch(dModel, dFF)
		defer putMLPScratch(msc)
		layerScratch := getDecodeLayerResidualScratch(dModel)
		defer putDecodeLayerResidualScratch(layerScratch)

		rmsPSO, err := pipelineFor(rmsKernelBF16(dModel))
		if err != nil {
			encErr = err
			return
		}
		rmsTG := rmsThreadgroup(dModel, rmsPSO)
		qPlan, err := newBF16GemvPlan(qDim, dModel)
		if err != nil {
			encErr = err
			return
		}
		oPlan, err := newBF16GemvPlan(dModel, qDim)
		if err != nil {
			encErr = err
			return
		}
		gatePlan, err := newBF16GemvPlan(dFF, dModel)
		if err != nil {
			encErr = err
			return
		}
		downPlan, err := newBF16GemvPlan(dModel, dFF)
		if err != nil {
			encErr = err
			return
		}
		ropePSO, err := ropePipelineBF16(false)
		if err != nil {
			encErr = err
			return
		}
		sdpaPSO, sdpaRTDim, err := sdpaVectorDispatchForHeadDim(headDim)
		if err != nil {
			encErr = err
			return
		}
		addPSO, err := pipelineFor("vv_Addbfloat16")
		if err != nil {
			encErr = err
			return
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		emitRMSNorm(sink, rmsPSO, xBuf, anwBuf, asc.normed, 0, dModel, eps, rmsTG)
		emitBF16GemvPlan(sink, qPlan, wqBuf, asc.normed, asc.q, dModel, qDim)
		emitRopeAt(sink, ropePSO, asc.q, asc.qr, 0, 0, offBuf, 0, nil, nHeads, headDim, headDim, scale, float32(math.Log2(float64(base))))
		// sdpaRTDim: headDim has no fixed pipeline (#28) — the runtime-dim fallback needs one extra binding.
		if sdpaRTDim {
			emitSDPARTDim(sink, sdpaPSO, asc.qr, kBuf, vBuf, asc.attn, 0, nil, nHeads, nKVHeads, headDim, kvLen, int64(kvLen*headDim), int64(headDim), int64(kvLen*headDim), int64(headDim), scale)
		} else {
			emitSDPA(sink, sdpaPSO, asc.qr, kBuf, vBuf, asc.attn, 0, nil, nHeads, nKVHeads, kvLen, int64(kvLen*headDim), int64(headDim), int64(kvLen*headDim), int64(headDim), scale)
		}
		emitBF16GemvPlan(sink, oPlan, woBuf, asc.attn, asc.attnOut, qDim, dModel)
		emitBinary(sink, addPSO, xBuf, 0, asc.attnOut, 0, layerScratch.h, 0, dModel)
		emitRMSNorm(sink, rmsPSO, layerScratch.h, mnwBuf, msc.mlpNormed, 0, dModel, eps, rmsTG)
		emitBF16GemvPlan(sink, gatePlan, wgBuf, msc.mlpNormed, msc.gate, dModel, dFF)
		emitBF16GemvPlan(sink, gatePlan, wuBuf, msc.mlpNormed, msc.up, dModel, dFF)
		if encErr = encGeluGateMul(enc, msc.gate, msc.up, msc.gated, *msc, dFF); encErr != nil {
			endEncodingFast(enc)
			return
		}
		emitBF16GemvPlan(sink, downPlan, wdBuf, msc.gated, msc.down, dFF, dModel)
		emitBinary(sink, addPSO, layerScratch.h, 0, msc.down, 0, outBuf, 0, dModel)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, ioScratch.out.bytes[:len(out)])
		}
	})
	return out, encErr
}
