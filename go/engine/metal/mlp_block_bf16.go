// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// MLPBlockBF16 runs a full gemma feed-forward block on-device in one command
// buffer, in bf16 (the dtype the decode path actually runs in) — the bf16
// sibling of MLPBlock:
//
//	normed = rmsnorm(x, normWeight)
//	gate   = Wgate · normed     up = Wup · normed      (dModel → dFF)
//	gated  = gelu(gate) · up    (gelu_approx composed in-line, each step rounded)
//	down   = Wdown · gated      (dFF → dModel)
//	out    = x + down           (residual)
//
// Every intermediate stays resident; one commit. Wgate/Wup are row-major
// (dFF × dModel), Wdown is (dModel × dFF). The composed-fallback gelu scalar
// operands are resident bf16 constant buffers, so the in-line gelu matches
// GeluGateMulBF16 byte-for-byte without re-uploading them per call. All
// inputs/outputs are raw bf16 bytes; the result equals the same native bf16 ops
// run separately — proven in the tests. This is a real decode sub-block on the
// no-cgo path.
func MLPBlockBF16(x, normWeight, wGate, wUp, wDown []byte, dModel, dFF int, eps float32) ([]byte, error) {
	return mlpBlockBF16Into(nil, x, normWeight, wGate, wUp, wDown, dModel, dFF, eps, false)
}

// MLPBlockBF16Into is MLPBlockBF16 with caller-owned output storage. If out has
// enough capacity, the command buffer writes the final residual directly into
// out through a pinned no-copy Metal buffer; otherwise a correctly sized output
// is allocated and returned.
func MLPBlockBF16Into(out []byte, x, normWeight, wGate, wUp, wDown []byte, dModel, dFF int, eps float32) ([]byte, error) {
	return mlpBlockBF16Into(out, x, normWeight, wGate, wUp, wDown, dModel, dFF, eps, true)
}

func mlpBlockBF16Into(out []byte, x, normWeight, wGate, wUp, wDown []byte, dModel, dFF int, eps float32, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dModel*bf16Size || len(normWeight) != dModel*bf16Size {
		return nil, core.NewError("native.MLPBlockBF16: x/normWeight must be dModel bf16 bytes")
	}
	if len(wGate) != dFF*dModel*bf16Size || len(wUp) != dFF*dModel*bf16Size {
		return nil, core.NewError("native.MLPBlockBF16: wGate/wUp must be dFF*dModel bf16 bytes")
	}
	if len(wDown) != dModel*dFF*bf16Size {
		return nil, core.NewError("native.MLPBlockBF16: wDown must be dModel*dFF bf16 bytes")
	}

	outLen := dModel * bf16Size
	callerOut := useCallerOut && cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}
	if dModel == 0 || dFF == 0 {
		clear(out)
		return out, nil
	}
	rmsPSO, err := pipelineFor(rmsKernelBF16(dModel))
	if err != nil {
		return nil, err
	}
	rmsTG := rmsThreadgroup(dModel, rmsPSO)
	inBM, inBN, inSM, inSN, inTM, inTN := gemvTiles(dModel, dFF)
	inPSO, err := pipelineFor(gemvKernelName("bfloat16", inBM, inBN, inSM, inSN, inTM, inTN))
	if err != nil {
		return nil, err
	}
	downBM, downBN, downSM, downSN, downTM, downTN := gemvTiles(dFF, dModel)
	downPSO, err := pipelineFor(gemvKernelName("bfloat16", downBM, downBN, downSM, downSN, downTM, downTN))
	if err != nil {
		return nil, err
	}
	addPSO, err := pipelineFor("vv_Addbfloat16")
	if err != nil {
		return nil, err
	}
	var geluPSO metal.MTLComputePipelineState
	useFusedGelu := gpuHasGeluKernel()
	if useFusedGelu {
		geluPSO, err = geluPipeline()
		if err != nil {
			return nil, err
		}
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
		nwBuf := residentBytes(normWeight)
		wgBuf, wuBuf, wdBuf := residentBytes(wGate), residentBytes(wUp), residentBytes(wDown)
		mlp := getMLPScratch(dModel, dFF)
		defer putMLPScratch(mlp)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		emitRMSNorm(sink, rmsPSO, xBuf, nwBuf, mlp.mlpNormed, 0, dModel, eps, rmsTG)
		emitGemv(sink, inPSO, wgBuf, 0, mlp.mlpNormed, mlp.gate, 0, dModel, dFF, inBM, inBN, inSM, inTM)
		emitGemv(sink, inPSO, wuBuf, 0, mlp.mlpNormed, mlp.up, 0, dModel, dFF, inBM, inBN, inSM, inTM)
		if useFusedGelu {
			emitBinary(sink, geluPSO, mlp.gate, 0, mlp.up, 0, mlp.gated, 0, dFF)
		} else {
			encErr = encGeluGateMul(enc, mlp.gate, mlp.up, mlp.gated, *mlp, dFF)
		}
		if encErr != nil {
			endEncodingFast(enc)
			return
		}
		emitGemv(sink, downPSO, wdBuf, 0, mlp.gated, mlp.down, 0, dFF, dModel, downBM, downBN, downSM, downTM)
		emitBinary(sink, addPSO, xBuf, 0, mlp.down, 0, outBuf, 0, dModel)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, ioScratch.out.bytes[:len(out)])
		}
	})
	return out, encErr
}
