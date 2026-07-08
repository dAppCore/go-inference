// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// qmv_rows.go — the multi-row qmv: ONE dispatch projects M contiguous
// activation rows through a quant weight, riding the lean gather-qmv kernel
// (grid Z = the row, rhs indices pinned to weight 0, lhs = the identity map).
// The dot body is MLX's qmv[_fast]_impl, so each row's output bytes are
// identical to the per-row decode qmv — unlike the qmm_t tier — while the
// weight streams to Z concurrent threadgroups instead of M serialised
// dispatches. This is the MTP verify's projection: the per-row interleave
// paid M full weight reads (M× a plain decode step), the qmm_t fold read the
// weight once but at small-M GEMM occupancy (~5× off the qmv floor on 12B).

// qmvRowsMax caps the multi-row route: verify blocks are draft+carry
// (≤ 17 at the largest -draft-block the fold admits); past steelGEMMMinRows
// the qmm_t fold amortises properly on its own.
const qmvRowsMax = 24

var (
	qmvRowsIdxOnce sync.Once
	qmvRowsLHS     metal.MTLBuffer // identity u32[0..qmvRowsMax): route z reads x row z
	qmvRowsRHS     metal.MTLBuffer // zero u32[qmvRowsMax]: every route dereferences weight 0
)

func qmvRowsIndexBuffers() (metal.MTLBuffer, metal.MTLBuffer, bool) {
	qmvRowsIdxOnce.Do(func() {
		n := uint(qmvRowsMax * 4)
		lhs := device.NewBufferWithLengthOptions(n, metal.MTLResourceStorageModeShared)
		rhs := device.NewBufferWithLengthOptions(n, metal.MTLResourceStorageModeShared)
		if lhs == nil || rhs == nil {
			return
		}
		ids := unsafe.Slice((*uint32)(lhs.Contents()), qmvRowsMax)
		zeros := unsafe.Slice((*uint32)(rhs.Contents()), qmvRowsMax)
		for i := range qmvRowsMax {
			ids[i] = uint32(i)
			zeros[i] = 0
		}
		qmvRowsLHS, qmvRowsRHS = lhs, rhs
	})
	return qmvRowsLHS, qmvRowsRHS, qmvRowsLHS != nil && qmvRowsRHS != nil
}

// encQMVRowsBF16At encodes the multi-row qmv when the geometry has a lean
// gather-qmv instantiation, reporting handled=false (no encode) otherwise so
// the caller keeps its qmm_t/per-row route. in rows are contiguous bf16 at
// inOff + z·inDim·2; out rows land at outOff + z·outDim·2.
func encQMVRowsBF16At(enc metal.MTLComputeCommandEncoder, wq, scales, biases, in, out metal.MTLBuffer, wqOff, scalesOff, biasesOff, inOff, outOff uint, rows, outDim, inDim, gs, bits int) (bool, error) {
	if rows < 2 || rows > qmvRowsMax || inDim <= 0 || gs <= 0 || inDim%gs != 0 || (inDim*bits)%32 != 0 {
		return false, nil
	}
	pso, ok := lthnGatherQMVPipeline(lthnGatherQMVKey{
		groupSize: gs, bits: bits, expertRows: 0, batchedX: true,
		fast: outDim%8 == 0 && inDim%512 == 0,
	})
	if !ok {
		return false, nil
	}
	lhs, rhs, ok := qmvRowsIndexBuffers()
	if !ok {
		return false, nil
	}
	emitLthnGatherQMVRoutes(encSink{enc}, pso, in, inOff, wq, wqOff, scales, scalesOff, biases, biasesOff, lhs, rhs, 0, out, outOff, outDim, inDim, gs, bits, 0, rows)
	return true, nil
}
