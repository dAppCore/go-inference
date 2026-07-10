// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

var (
	rmsResidualPSOOnce       sync.Once
	rmsResidualPSO           metal.MTLComputePipelineState
	rmsResidualPSOErr        error
	rmsResidualScratchPools  sync.Map
	errRMSResidualScratchDim = core.NewError("native.rmsNormResidualScratch: dimension mismatch")
)

type rmsNormResidualBF16Scratch struct {
	axisSize       int
	x, res, out    *pinnedNoCopyBytes
	xView, resView cachedNoCopyBytesView
	noCopyOutputView
}

func newRMSNormResidualBF16Scratch(axisSize int) (*rmsNormResidualBF16Scratch, error) {
	if axisSize <= 0 {
		return nil, core.NewError("native.newRMSNormResidualBF16Scratch: invalid axis size")
	}
	n := axisSize * bf16Size
	x, err := newPinnedNoCopyBytes(n)
	if err != nil {
		return nil, err
	}
	res, err := newPinnedNoCopyBytes(n)
	if err != nil {
		x.Close()
		return nil, err
	}
	out, err := newPinnedNoCopyBytes(n)
	if err != nil {
		x.Close()
		res.Close()
		return nil, err
	}
	return &rmsNormResidualBF16Scratch{axisSize: axisSize, x: x, res: res, out: out}, nil
}

func rmsResidualScratchPoolFor(axisSize int) *scratchLIFOPool[*rmsNormResidualBF16Scratch] {
	if v, ok := rmsResidualScratchPools.Load(axisSize); ok {
		return v.(*scratchLIFOPool[*rmsNormResidualBF16Scratch])
	}
	pool := &scratchLIFOPool[*rmsNormResidualBF16Scratch]{}
	actual, _ := rmsResidualScratchPools.LoadOrStore(axisSize, pool)
	return actual.(*scratchLIFOPool[*rmsNormResidualBF16Scratch])
}

func getRMSNormResidualBF16Scratch(axisSize int) (*rmsNormResidualBF16Scratch, error) {
	if s := rmsResidualScratchPoolFor(axisSize).Get(); s != nil {
		if s.axisSize == axisSize && s.x != nil && s.res != nil && s.out != nil {
			return s, nil
		}
		s.Close()
	}
	return newRMSNormResidualBF16Scratch(axisSize)
}

func putRMSNormResidualBF16Scratch(s *rmsNormResidualBF16Scratch) {
	if s != nil && s.axisSize > 0 {
		rmsResidualScratchPoolFor(s.axisSize).Put(s)
	}
}

func (s *rmsNormResidualBF16Scratch) Close() {
	if s == nil {
		return
	}
	if s.x != nil {
		s.x.Close()
		s.x = nil
	}
	if s.res != nil {
		s.res.Close()
		s.res = nil
	}
	s.xView.Close()
	s.resView.Close()
	if s.out != nil {
		s.out.Close()
		s.out = nil
	}
	s.closeOutputView()
	s.axisSize = 0
}

func (s *rmsNormResidualBF16Scratch) buffers(x, res []byte) (metal.MTLBuffer, metal.MTLBuffer, metal.MTLBuffer, error) {
	if s == nil || s.x == nil || s.res == nil || s.out == nil {
		return nil, nil, nil, core.NewError("native.rmsNormResidualBF16Scratch.buffers: scratch is nil")
	}
	n := s.axisSize * bf16Size
	if len(x) != n || len(res) != n || len(s.out.bytes) != n {
		return nil, nil, nil, errRMSResidualScratchDim
	}
	var err error
	xBuf, xNoCopy := s.xView.buffer(x)
	if !xNoCopy {
		xBuf, err = s.x.copyBuffer(x)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	resBuf, resNoCopy := s.resView.buffer(res)
	if !resNoCopy {
		resBuf, err = s.res.copyBuffer(res)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	return xBuf, resBuf, s.out.buf, nil
}

// rmsNormResidualPipeline builds (once) the fused rms-norm+residual pipeline from the custom kernels
// library (lthn_kernels.metallib). Shares the customLibraryLoaded gate with the gelu kernel.
// rmsNormResidualPipelineFor selects the fused kernel by axis: the single-row kernel
// (one pass, ≤1024 threads × N_READS) up to rmsLoopedLimit, the grid-striding looped
// variant beyond it (gemma4 31B hidden 5376 — the #348 tail-drop class). One dispatch
// either way, so every lane keeps its barrier discipline (the paged lane's CONCURRENT
// encoder orders stages with explicit barriers; a composed rms+add pair would race).
var (
	rmsResidualLoopedPSOOnce sync.Once
	rmsResidualLoopedPSO     metal.MTLComputePipelineState
	rmsResidualLoopedPSOErr  error
)

func rmsNormResidualPipelineFor(axisSize int) (metal.MTLComputePipelineState, error) {
	if axisSize > rmsLoopedLimit {
		return rmsNormResidualLoopedPipeline()
	}
	return rmsNormResidualPipeline()
}

func rmsNormResidualLoopedPipeline() (metal.MTLComputePipelineState, error) {
	rmsResidualLoopedPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			rmsResidualLoopedPSOErr = core.NewError("native.rmsNormResidualLoopedPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_rmsnorm_residual_looped_bf16")
		if fn == nil || fn.GetID() == 0 {
			rmsResidualLoopedPSOErr = core.NewError("native.rmsNormResidualLoopedPipeline: kernel lthn_rmsnorm_residual_looped_bf16 not found — rebuild lthn_kernels.metallib")
			return
		}
		rmsResidualLoopedPSO, rmsResidualLoopedPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return rmsResidualLoopedPSO, rmsResidualLoopedPSOErr
}

func rmsNormResidualPipeline() (metal.MTLComputePipelineState, error) {
	rmsResidualPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			rmsResidualPSOErr = core.NewError("native.rmsNormResidualPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_rmsnorm_residual_bf16")
		if fn == nil || fn.GetID() == 0 {
			rmsResidualPSOErr = core.NewError("native.rmsNormResidualPipeline: kernel lthn_rmsnorm_residual_bf16 not found")
			return
		}
		rmsResidualPSO, rmsResidualPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return rmsResidualPSO, rmsResidualPSOErr
}

// RMSNormResidualBF16 computes, in ONE dispatch, the fused gemma4 post-attention / post-FF tail:
//
//	out = res + RMSNorm(x, weight)
//
// x/res/weight are bf16 bytes of length axisSize (single row); out is axisSize bf16 bytes. The kernel
// copies MLX's rms_single_row reduction verbatim and rounds the normed value to bf16 before the add,
// so the result is byte-identical to AddBF16(res, RMSNormBF16(x, weight)) — gated in the parity test.
// axisSize must be ≤ rmsLoopedLimit (the single-row kernel; every gemma hidden/head size qualifies).
// Guard with gpuHasGeluKernel (same custom library) before calling on the decode path.
// encRMSNormResidualBF16 encodes the fused out = res + RMSNorm(x, weight) into `enc` (no commit) — the
// encoder-level form of RMSNormResidualBF16, for the re-encode decode path to stay LOCKSTEP with the ICB's
// setRMSResidual (same kernel, so the two paths are byte-equal). wOff offsets the weight binding.
func encRMSNormResidualBF16(enc metal.MTLComputeCommandEncoder, x, weight, res, out metal.MTLBuffer, wOff uint, axisSize int, eps float32) error {
	return encRMSNormResidualBF16At(enc, x, weight, res, out, 0, wOff, 0, 0, axisSize, eps)
}

// encRMSNormResidualBF16At is encRMSNormResidualBF16 with the branch input, residual and output
// bound at byte offsets — the SAME fused pipeline, for batched rows living at offsets inside
// shared K-row buffers (bit-identical per row to the offset-0 sequential encode).
func encRMSNormResidualBF16At(enc metal.MTLComputeCommandEncoder, x, weight, res, out metal.MTLBuffer, xOff, wOff, resOff, outOff uint, axisSize int, eps float32) error {
	pso, err := rmsNormResidualPipelineFor(axisSize)
	if err != nil {
		return err
	}
	emitRMSNormResidualAt(encSink{enc}, pso, x, weight, res, out, xOff, wOff, resOff, outOff, axisSize, eps, rmsThreadgroup(axisSize, pso))
	return nil
}

func RMSNormResidualBF16(x, weight, res []byte, axisSize int, eps float32) ([]byte, error) {
	return RMSNormResidualBF16Into(nil, x, weight, res, axisSize, eps)
}

func RMSNormResidualBF16Into(out []byte, x, weight, res []byte, axisSize int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != axisSize*bf16Size || len(res) != axisSize*bf16Size {
		return nil, core.NewError("native.RMSNormResidualBF16: x and res must each be axisSize bf16 bytes")
	}
	if len(weight) != axisSize*bf16Size {
		return nil, core.NewError("native.RMSNormResidualBF16: weight must be axisSize bf16 bytes")
	}
	if axisSize > rmsLoopedLimit {
		return nil, core.NewError("native.RMSNormResidualBF16: axisSize exceeds the single-row kernel limit")
	}
	pso, err := rmsNormResidualPipeline()
	if err != nil {
		return nil, err
	}

	outLen := axisSize * bf16Size
	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	var encErr error
	withAutoreleasePool(func() {
		wBuf := residentBytes(weight)
		scratch, err := getRMSNormResidualBF16Scratch(axisSize)
		if err != nil {
			encErr = err
			return
		}
		defer putRMSNormResidualBF16Scratch(scratch)
		xBuf, rBuf, oBuf, err := scratch.buffers(x, res)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				oBuf = tmp
				directOut = true
			}
		}
		tgSize := rmsThreadgroup(axisSize, pso) // ceil(axis/N_READS) rounded up to a simd — one threadgroup, one row

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitRMSNormResidual(encSink{enc}, pso, xBuf, wBuf, rBuf, oBuf, 0, axisSize, eps, tgSize)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, scratch.out.bytes[:outLen])
		}
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

// encRMSNormResidualRowsBF16At encodes out rows = res rows + RMSNorm(x rows, weight)
// for `rows` CONTIGUOUS rows in one dispatch: the fused kernel's gid already indexes
// rows (base = gid·axisSize), so the rows form just widens the grid — each row's
// bytes identical to the sequential single-row encodes. The batched epilogue's
// rms-rows + add pair collapses into it (#372: launch-bound at MTP-verify K).
func encRMSNormResidualRowsBF16At(enc metal.MTLComputeCommandEncoder, x, weight, res, out metal.MTLBuffer, xOff, wOff, resOff, outOff uint, rows, axisSize int, eps float32) error {
	pso, err := rmsNormResidualPipelineFor(axisSize)
	if err != nil {
		return err
	}
	sink := encSink{enc}
	sink.setPSO(pso)
	sink.setBuf(x, xOff, 0)
	sink.setBuf(weight, wOff, 1)
	sink.setBuf(res, resOff, 2)
	sink.setBuf(out, outOff, 3)
	sink.setF32(eps, 4)
	sink.setI32(int32(axisSize), 5)
	sink.setI32(1, 6)
	tg := rmsThreadgroup(axisSize, pso)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(rows), Height: 1, Depth: 1},
		metal.MTLSize{Width: tg, Height: 1, Depth: 1},
	)
	return nil
}
