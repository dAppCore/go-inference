// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// gemvTiles mirrors MLX's non-transposed gemv tile selection
// (mlx/backend/metal/matmul.cpp, gemv_axbpy) verbatim, so the kernel name we
// assemble resolves to the exact variant mlx-c would dispatch for this shape.
// The returned tile parameters are baked into the kernel name as
// bm/bn/sm/sn/tm/tn — they are template specialisations, not function constants,
// so picking the right variant is the whole game. bn stays 1 for the shapes
// decode cares about, which means the kernel needs no threadgroup memory.
func gemvTiles(k, outVecLen int) (bm, bn, sm, sn, tm, tn int) {
	tm, tn = 4, 4
	sm, sn = 1, 32
	bm, bn = 1, 1

	bm = 4
	if outVecLen >= 4096 {
		bm = 8
	}
	sn = 32
	switch {
	case k <= 64:
		bm, sm, sn = 1, 8, 4
	case k >= 16*outVecLen:
		bm, bn = 1, 8
	}
	if outVecLen < tm {
		tm = 1
	}
	return bm, bn, sm, sn, tm, tn
}

type gemvKernelNameKey struct {
	dtype              string
	bm, bn, sm, sn, tm int
	tn                 int
}

var (
	gemvKernelNameMu    sync.Mutex
	gemvKernelNameCache = map[gemvKernelNameKey]string{}
)

func gemvKernelName(dtype string, bm, bn, sm, sn, tm, tn int) string {
	key := gemvKernelNameKey{dtype: dtype, bm: bm, bn: bn, sm: sm, sn: sn, tm: tm, tn: tn}
	gemvKernelNameMu.Lock()
	if name, ok := gemvKernelNameCache[key]; ok {
		gemvKernelNameMu.Unlock()
		return name
	}
	gemvKernelNameMu.Unlock()

	name := core.Sprintf("gemv_%s_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", dtype, bm, bn, sm, sn, tm, tn)

	gemvKernelNameMu.Lock()
	if existing, ok := gemvKernelNameCache[key]; ok {
		gemvKernelNameMu.Unlock()
		return existing
	}
	gemvKernelNameCache[key] = name
	gemvKernelNameMu.Unlock()
	return name
}

// MatVec computes out = mat @ vec, where mat is a row-major (outDim x inDim)
// matrix and vec has length inDim, returning a fresh slice of length outDim. It
// drives MLX's gemv kernel directly through the no-cgo path: the tile variant is
// chosen exactly as mlx-c chooses it, and a single size-1 batch is configured so
// the kernel's batch-offset arithmetic resolves to zero. float32 only.
//
// This is the first hard kernel on the native path — threadgroup-parallel, a
// real parameter ABI, tile-specialised — proving the dual path reaches the
// kernels that actually carry inference cost, not just elementwise ops. Buffer
// ABI (mlx gemv [[kernel]] entry): mat(0) vec(1) out(3) in_vec_size(4)
// out_vec_size(5) matrix_ld(6) batch_ndim(9) batch_shape(10) vec_stride(11)
// mat_stride(12); dispatched as ceil(outDim/(bm*sm*tm)) threadgroups of
// (32, bn, bm) threads. Byte-for-byte parity with pkg/metal.Matmul of
// (outDim x inDim) @ (inDim x 1) is gated in parity_test.go.
func MatVec(mat, vec []float32, outDim, inDim int) ([]float32, error) {
	return MatVecInto(nil, mat, vec, outDim, inDim)
}

// MatVecInto is MatVec with caller-owned output storage when cap(out) >= outDim.
func MatVecInto(out []float32, mat, vec []float32, outDim, inDim int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(mat) != outDim*inDim {
		return nil, core.NewError("native.MatVec: len(mat) must equal outDim*inDim")
	}
	if len(vec) != inDim {
		return nil, core.NewError("native.MatVec: len(vec) must equal inDim")
	}
	if outDim == 0 || inDim == 0 {
		if cap(out) < outDim {
			return make([]float32, outDim), nil
		}
		return out[:outDim], nil
	}

	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	name := gemvKernelName("float32", bm, bn, sm, sn, tm, tn)
	pso, err := pipelineFor(name)
	if err != nil {
		return nil, err
	}

	callerOut := cap(out) >= outDim
	if !callerOut {
		out = make([]float32, outDim)
	} else {
		out = out[:outDim]
	}
	var encErr error
	withAutoreleasePool(func() {
		matBuf := residentFloat32(mat)
		scratch, err := getQMVFloatScratch(outDim, inDim)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVFloatScratch(scratch)
		vecBuf, outBuf, err := scratch.buffers(vec)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitGemv(encSink{enc}, pso, matBuf, 0, vecBuf, outBuf, 0, inDim, outDim, bm, bn, sm, tm)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		if !directOut {
			copy(float32Bytes(out), scratch.out.bytes[:outDim*4])
		}
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

// setEncInt32 binds a single int32 as an inline constant at a buffer index
// (the gemv scalar params: sizes, leading dimension, batch ndim/shape).
func setEncInt32(enc metal.MTLComputeCommandEncoder, v int32, idx uint) {
	setBytesI32(enc, v, idx)
}

// setEncInt64 binds a single int64 as an inline constant at a buffer index
// (the gemv batch strides, which the kernel types as int64_t*).
func setEncInt64(enc metal.MTLComputeCommandEncoder, v int64, idx uint) {
	setBytesI64(enc, v, idx)
}
