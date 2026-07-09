// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
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

// lthnQMVRowsMaxM caps the register-tiled kernel: each thread holds M x-slices
// plus M×4 accumulators, and past M=4 the register pressure collapses
// occupancy — measured on the 12B verify: M=3 gpuTotal 18.7ms (vs 20 gather),
// M=5 35.3ms (vs 27.6 gather). Wider rows ride the gather fallback.
const lthnQMVRowsMaxM = 4

type lthnQMVRowsKey struct {
	groupSize, bits, m int
}

var (
	lthnQMVRowsPSOMu    sync.Mutex
	lthnQMVRowsPSOCache = map[lthnQMVRowsKey]metal.MTLComputePipelineState{}
)

// lthnQMVRowsPipeline resolves (and caches, including failures) the M-row qmv
// for a geometry: M bakes as a function constant on the group_size/bits
// template instance. A miss caches nil so the caller falls back per dispatch
// without re-probing.
func lthnQMVRowsPipeline(key lthnQMVRowsKey) (metal.MTLComputePipelineState, bool) {
	lthnQMVRowsPSOMu.Lock()
	defer lthnQMVRowsPSOMu.Unlock()
	if pso, ok := lthnQMVRowsPSOCache[key]; ok {
		return pso, pso != nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		lthnQMVRowsPSOCache[key] = nil
		return nil, false
	}
	name := core.Sprintf("lthn_qmv_rows_bfloat16_t_gs_%d_b_%d", key.groupSize, key.bits)
	fc := metal.NewMTLFunctionConstantValues()
	m := int32(key.m)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&m), metal.MTLDataTypeInt, 0)
	fn, err := customLibrary.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil || fn == nil || fn.GetID() == 0 {
		lthnQMVRowsPSOCache[key] = nil
		return nil, false
	}
	pso, perr := device.NewComputePipelineStateWithFunctionError(fn)
	if perr != nil {
		lthnQMVRowsPSOCache[key] = nil
		return nil, false
	}
	lthnQMVRowsPSOCache[key] = pso
	return pso, true
}

// encQMVRowsBF16At encodes the multi-row qmv, reporting handled=false (no
// encode) when the geometry has no kernel so the caller keeps its
// qmm_t/per-row route. in rows are contiguous bf16 at inOff + z·inDim·2; out
// rows land at outOff + z·outDim·2. Rows 2..lthnQMVRowsMaxM take the
// register-tiled lthn_qmv_rows (the weight stream read ONCE, rows
// byte-identical to MLX's plain qmv); wider rows ride the lean gather kernel
// (grid-Z, qmv_fast bytes, weight re-streamed per row but still ~2× the
// serialised per-row interleave).
func encQMVRowsBF16At(enc metal.MTLComputeCommandEncoder, wq, scales, biases, in, out metal.MTLBuffer, wqOff, scalesOff, biasesOff, inOff, outOff uint, rows, outDim, inDim, gs, bits int) (bool, error) {
	if rows < 2 || rows > qmvRowsMax || inDim <= 0 || gs <= 0 || inDim%gs != 0 || (inDim*bits)%32 != 0 {
		return false, nil
	}
	// The register-tiled kernel ports qmv_impl's MAIN loop only (no in-tail):
	// every production projection dim is a 256-multiple (block_size = 8 values
	// × 32 lanes), anything else keeps the gather fallback.
	if rows <= lthnQMVRowsMaxM && outDim%8 == 0 && inDim%256 == 0 {
		if pso, ok := lthnQMVRowsPipeline(lthnQMVRowsKey{groupSize: gs, bits: bits, m: rows}); ok {
			sink := encSink{enc}
			sink.setPSO(pso)
			sink.setBuf(wq, wqOff, 0)
			sink.setBuf(scales, scalesOff, 1)
			sink.setBuf(biases, biasesOff, 2)
			sink.setBuf(in, inOff, 3)
			sink.setBuf(out, outOff, 4)
			sink.setI32(int32(inDim), 5)
			sink.setI32(int32(outDim), 6)
			sink.dispatchThreadgroups(
				metal.MTLSize{Width: 1, Height: uint((outDim + 7) / 8), Depth: 1},
				metal.MTLSize{Width: 32, Height: 2, Depth: 1},
			)
			return true, nil
		}
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
