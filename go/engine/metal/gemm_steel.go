// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// Steel GEMM — the true tiled matmul for the batched pass's large-row projections. Below
// steelGEMMMinRows the grid-Z batched gemv runs each row's tile loop unchanged (byte-identical to
// the sequential oracle — the MTP verify and every parity fixture live there). At or above it, the
// projections route to MLX's steel_gemm_fused kernel: one simdgroup-matrix GEMM reading the weight
// ONCE for all rows. Steel accumulates per output tile (simdgroup MMA over BK panels), a different
// summation order from the per-row gemv — so large-row prefill trades byte- for token-identity,
// exactly as pkg/metal's GEMM prefill always has. Production quality is pinned by the closeness
// test (per-element bf16 agreement within tolerance) and the live output remaining coherent.

// steelGEMMMinRows is the row count at which the batched projections switch from the grid-Z gemv
// to the steel GEMM. 64 = one full BM tile; MTP verify blocks (K ≤ 16) stay on the gemv and keep
// strict byte-identity with the sequential lane.
const steelGEMMMinRows = 32

// steelGEMMDisabledForTest forces the batched projections back onto the grid-Z gemv at any row
// count — the A/B lever for the GEMM closeness/engagement tests. Production never sets it.
var steelGEMMDisabledForTest bool

// steelGEMMDispatchesForTest counts steel GEMM dispatches while pieceTimingOn — the engagement
// receipt (a GEMM and a gemv are one dispatch each, so plain dispatch counts cannot tell them
// apart). Zero cost in production (one bool test).
var steelGEMMDispatchesForTest int64

// steelGEMMParams mirrors mlx::steel::GEMMParams (lib/mlx .../steel/gemm/params.h) — 8 int32, 3
// int64 (8-aligned at offset 32), 3 int32, padded to 72 bytes. Bound as the constant params
// struct at buffer(4).
type steelGEMMParams struct {
	M, N, K                int32
	LDA, LDB, LDD          int32
	TilesN, TilesM         int32
	BatchStrideA           int64
	BatchStrideB           int64
	BatchStrideD           int64
	SwizzleLog             int32
	GemmKIterationsAligned int32
	BatchNDim              int32
	_                      int32 // trailing pad to the struct's 8-byte alignment
}

type steelGEMMKey struct{ alignM, alignN, alignK bool }

var (
	steelGEMMMu       sync.Mutex
	steelGEMMPSOCache = map[steelGEMMKey]metal.MTLComputePipelineState{}
	steelGEMMBroken   bool
)

const (
	steelGEMMBM = 64
	steelGEMMBN = 64
	steelGEMMBK = 16
	steelGEMMWM = 2
	steelGEMMWN = 2
)

// steelGEMMPipeline builds (and caches) the nt bf16 steel GEMM pipeline for an alignment combo.
// The alignment booleans are function constants (they select the no-bounds-check fast paths), so
// each combo is its own PSO. has_batch/use_out_source/do_axpby are baked false — the batched pass
// runs plain single-batch D = A @ Bᵀ.
func steelGEMMPipeline(alignM, alignN, alignK bool) (metal.MTLComputePipelineState, bool) {
	steelGEMMMu.Lock()
	defer steelGEMMMu.Unlock()
	if steelGEMMBroken {
		return nil, false
	}
	key := steelGEMMKey{alignM: alignM, alignN: alignN, alignK: alignK}
	if pso, ok := steelGEMMPSOCache[key]; ok {
		return pso, true
	}
	if library == nil || library.GetID() == 0 {
		steelGEMMBroken = true
		return nil, false
	}
	name := core.Sprintf("steel_gemm_fused_nt_bfloat16_bfloat16_bm%d_bn%d_bk%d_wm%d_wn%d",
		steelGEMMBM, steelGEMMBN, steelGEMMBK, steelGEMMWM, steelGEMMWN)
	fc := metal.NewMTLFunctionConstantValues()
	off := uint8(0)
	for _, idx := range []uint{10, 100, 110} { // has_batch, use_out_source, do_axpby
		fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&off), metal.MTLDataTypeBool, idx)
	}
	aM, aN, aK := alignM, alignN, alignK
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&aM), metal.MTLDataTypeBool, 200)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&aN), metal.MTLDataTypeBool, 201)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&aK), metal.MTLDataTypeBool, 202)
	fn, err := library.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil || fn == nil || fn.GetID() == 0 {
		steelGEMMBroken = true
		return nil, false
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil || pso == nil || pso.GetID() == 0 {
		steelGEMMBroken = true
		return nil, false
	}
	steelGEMMPSOCache[key] = pso
	return pso, true
}

// encGemmBF16NT encodes D[rows × outDim] = act[rows × inDim] @ W[outDim × inDim]ᵀ as ONE steel
// GEMM: A = the contiguous activation rows at vecOff (lda = inDim), B = the row-major weight at
// matOff (the nt variant reads it transposed, ldb = inDim), D = contiguous output rows at outOff
// (ldd = outDim). Reports false when the steel pipeline is unavailable (the caller keeps the
// batched gemv).
func encGemmBF16NT(enc metal.MTLComputeCommandEncoder, mat, vec, out metal.MTLBuffer, matOff, vecOff, outOff uint, outDim, inDim, rows int) bool {
	pso, ok := steelGEMMPipeline(rows%steelGEMMBM == 0, outDim%steelGEMMBN == 0, inDim%steelGEMMBK == 0)
	if !ok {
		return false
	}
	if pieceTimingOn {
		steelGEMMDispatchesForTest++
	}
	tilesM := (rows + steelGEMMBM - 1) / steelGEMMBM
	tilesN := (outDim + steelGEMMBN - 1) / steelGEMMBN
	// threadblock swizzle (mlx matmul.cpp): interleave the tile walk so neighbouring threadgroups
	// share B panels in L2 — 0 for short grids, 2 on this device class for tall ones.
	swizzle := 0
	if tilesM > 3 {
		swizzle = 2
	}
	params := steelGEMMParams{
		M: int32(rows), N: int32(outDim), K: int32(inDim),
		LDA: int32(inDim), LDB: int32(inDim), LDD: int32(outDim),
		TilesN: int32(tilesN), TilesM: int32(tilesM),
		SwizzleLog: int32(swizzle), GemmKIterationsAligned: int32(inDim / steelGEMMBK), BatchNDim: 1,
	}
	sink := encSink{enc}
	sink.setPSO(pso)
	sink.setBuf(vec, vecOff, 0) // A: the activation rows
	sink.setBuf(mat, matOff, 1) // B: the weight, read transposed (nt)
	sink.setBuf(out, outOff, 3) // D
	setBytes(enc, unsafe.Pointer(&params), uint(unsafe.Sizeof(params)), 4)
	tile := 1 << swizzle
	gridX := tilesN * tile
	gridY := (tilesM + tile - 1) / tile
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(gridX), Height: uint(gridY), Depth: 1},
		metal.MTLSize{Width: 32, Height: steelGEMMWN, Depth: steelGEMMWM},
	)
	return true
}
