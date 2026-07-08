// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// moe_grouped.go — the grouped-GEMM expert MLP for the batched MoE prefill (#347).
//
// The all-routes gather (emitGatherQMVAllRoutes) runs one GEMV per routed pair, re-reading
// each expert's quant weights once per pair — at prompt scale that is K·topK weight sweeps
// per projection and the MoE stage measured ~76% of the 26B prefill GPU time. This lane
// sorts the pairs by expert (one small bucket-sort kernel), gathers the pair inputs into
// sorted order, and drives MLX's affine_gather_qmm_rhs_nax steel kernel — each expert's
// weights read once per 64-row block — then scatters the down outputs back to pair order so
// the weighted-sum + combine tail is byte-identical to the all-routes path's ordering.
// Token-identity tier (simdgroup-MMA accumulation vs the per-pair GEMV), the same boundary
// the prompt fold's qmm_t already trades at.

// moeGroupedDisabledForTest forces the batched MoE block back onto the all-routes gathers
// (the A/B lever; also honoured via LTHN_MOE_GEMM=0).
var moeGroupedDisabledForTest bool

func moeGroupedEnvDisabled() bool { return os.Getenv("LTHN_MOE_GEMM") == "0" }

// The steel gather_qmm_rhs tile family. The NAX variant (bm/bn/bk 64) is gated on
// metal::is_nax_available() hardware (M4-era tile units) — on this machine class the
// portable steel tiles are the correct family.
const (
	moeGroupedBM = 16
	moeGroupedBN = 32
	moeGroupedBK = 32
	moeGroupedWM = 1
	moeGroupedWN = 2
	// lthn_moe_pair_sort holds one threadgroup histogram of numExperts atomics.
	moePairSortMaxExperts = 1024
	moePairSortThreads    = 1024
)

type moeGroupedPSOKey struct {
	groupSize, bits         int
	alignM, alignN, alignK  bool
}

var (
	moeGroupedPSOMu    sync.Mutex
	moeGroupedPSOCache = map[moeGroupedPSOKey]metal.MTLComputePipelineState{}
	moeGroupedBroken   bool

	moePairSortPSOOnce sync.Once
	moePairSortPSO     metal.MTLComputePipelineState
	moePairSortPSOErr  error

	moeGatherRowsPSOOnce sync.Once
	moeGatherRowsPSO     metal.MTLComputePipelineState
	moeGatherRowsPSOErr  error

	moeScatterRowsPSOOnce sync.Once
	moeScatterRowsPSO     metal.MTLComputePipelineState
	moeScatterRowsPSOErr  error

	moeGeluGateUpPSOOnce sync.Once
	moeGeluGateUpPSO     metal.MTLComputePipelineState
	moeGeluGateUpPSOErr  error
)

func lthnPipeline(name string, once *sync.Once, pso *metal.MTLComputePipelineState, psoErr *error) (metal.MTLComputePipelineState, error) {
	once.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			*psoErr = core.NewError("native.lthnPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName(name)
		if fn == nil || fn.GetID() == 0 {
			*psoErr = core.NewError("native.lthnPipeline: kernel " + name + " not found")
			return
		}
		*pso, *psoErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return *pso, *psoErr
}

func moePairSortPipeline() (metal.MTLComputePipelineState, error) {
	return lthnPipeline("lthn_moe_pair_sort", &moePairSortPSOOnce, &moePairSortPSO, &moePairSortPSOErr)
}

func moeGatherRowsPipeline() (metal.MTLComputePipelineState, error) {
	return lthnPipeline("lthn_moe_gather_rows_bf16", &moeGatherRowsPSOOnce, &moeGatherRowsPSO, &moeGatherRowsPSOErr)
}

func moeScatterRowsPipeline() (metal.MTLComputePipelineState, error) {
	return lthnPipeline("lthn_moe_scatter_rows_bf16", &moeScatterRowsPSOOnce, &moeScatterRowsPSO, &moeScatterRowsPSOErr)
}

func moeGeluGateUpPipeline() (metal.MTLComputePipelineState, error) {
	return lthnPipeline("lthn_moe_gelu_gate_up_bf16", &moeGeluGateUpPSOOnce, &moeGeluGateUpPSO, &moeGeluGateUpPSOErr)
}

// moeGatherQMMRHSPipeline resolves MLX's affine_gather_qmm_rhs_nax_nt steel kernel (the
// transposed/qmm_t weight layout) with the alignment function constants at 200/201/202 —
// the same FC shape steelGEMMPipelineTrans drives for the fused GEMM.
func moeGatherQMMRHSPipeline(groupSize, bits int, alignM, alignN, alignK bool) (metal.MTLComputePipelineState, bool) {
	moeGroupedPSOMu.Lock()
	defer moeGroupedPSOMu.Unlock()
	if moeGroupedBroken {
		return nil, false
	}
	key := moeGroupedPSOKey{groupSize: groupSize, bits: bits, alignM: alignM, alignN: alignN, alignK: alignK}
	if pso, ok := moeGroupedPSOCache[key]; ok {
		return pso, pso != nil
	}
	if library == nil || library.GetID() == 0 {
		moeGroupedBroken = true
		return nil, false
	}
	name := core.Sprintf("affine_gather_qmm_rhs_nt_bfloat16_t_gs_%d_b_%d_bm_%d_bn_%d_bk_%d_wm_%d_wn_%d",
		groupSize, bits, moeGroupedBM, moeGroupedBN, moeGroupedBK, moeGroupedWM, moeGroupedWN)
	fc := metal.NewMTLFunctionConstantValues()
	aM, aN, aK := alignM, alignN, alignK
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&aM), metal.MTLDataTypeBool, 200)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&aN), metal.MTLDataTypeBool, 201)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&aK), metal.MTLDataTypeBool, 202)
	fn, err := library.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil || fn == nil || fn.GetID() == 0 {
		moeGroupedPSOCache[key] = nil // this gs/bits variant is absent; other variants may exist
		return nil, false
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil || pso == nil || pso.GetID() == 0 {
		moeGroupedPSOCache[key] = nil
		return nil, false
	}
	moeGroupedPSOCache[key] = pso
	return pso, true
}

// moeGroupedScratch holds the sorted-order slabs. Owned by moeBatchScratch, grown with it.
type moeGroupedScratch struct {
	sortedPair   metal.MTLBuffer // pairs int32: pair id at sorted position
	sortedExpert metal.MTLBuffer // pairs uint32: expert id per sorted row (qmm_rhs indices)
	xSorted      metal.MTLBuffer // pairs × dModel bf16
	guSorted     metal.MTLBuffer // pairs × 2·expertDFF bf16 (fused gate_up; halves for split packs)
	gatedSorted  metal.MTLBuffer // pairs × expertDFF bf16
	downSorted   metal.MTLBuffer // pairs × dModel bf16

	pairsCap, dModelCap, expertDFFCap int
}

func (g *moeGroupedScratch) ensure(pairs, dModel, expertDFF int) error {
	if g.pairsCap >= pairs && g.dModelCap == dModel && g.expertDFFCap >= expertDFF {
		return nil
	}
	g.sortedPair = device.NewBufferWithLengthOptions(uint(pairs*4), metal.MTLResourceStorageModeShared)
	g.sortedExpert = device.NewBufferWithLengthOptions(uint(pairs*4), metal.MTLResourceStorageModeShared)
	g.xSorted = scratchBF16(pairs * dModel)
	g.guSorted = scratchBF16(pairs * 2 * expertDFF)
	g.gatedSorted = scratchBF16(pairs * expertDFF)
	g.downSorted = scratchBF16(pairs * dModel)
	if g.sortedPair == nil || g.sortedExpert == nil || g.xSorted == nil || g.guSorted == nil || g.gatedSorted == nil || g.downSorted == nil {
		return core.NewError("native.moeGroupedScratch: buffer allocation failed")
	}
	g.pairsCap, g.dModelCap, g.expertDFFCap = pairs, dModel, expertDFF
	return nil
}

// moeGroupedUsable reports whether the grouped-GEMM expert lane can run this layer — checked
// once per block so the emission never declines mid-encode.
func moeGroupedUsable(w MoEQuantLayerWeights, expertBits, expertGS int) bool {
	if moeGroupedDisabledForTest || moeGroupedEnvDisabled() {
		return false
	}
	if w.NumExperts > moePairSortMaxExperts {
		return false
	}
	if _, err := moePairSortPipeline(); err != nil {
		return false
	}
	if _, err := moeGatherRowsPipeline(); err != nil {
		return false
	}
	if _, err := moeScatterRowsPipeline(); err != nil {
		return false
	}
	if _, err := moeGeluGateUpPipeline(); err != nil {
		return false
	}
	// both projections share the expert gs/bits (validated upstream); resolve the aligned
	// variant to prove the kernel family exists — per-call alignment picks the exact PSO.
	if _, ok := moeGatherQMMRHSPipeline(expertGS, expertBits, true, true, true); !ok {
		return false
	}
	return true
}

// emitMoEGatherQMMRHS encodes one sorted-rows grouped projection: out[M,N] = x[M,K] @
// dequant(w[e(row)])ᵀ with e = sortedExpert. Bindings mirror mlx's gather_qmm_rhs_nax host
// call: x(0) w(1) scales(2) biases(3) indices(4) out(5) M(6) N(7) K(8).
func emitMoEGatherQMMRHS(sink encSink, pso metal.MTLComputePipelineState, x metal.MTLBuffer, xOff uint, wq metal.MTLBuffer, wqOff uint, scales metal.MTLBuffer, scalesOff uint, biases metal.MTLBuffer, biasesOff uint, indices, out metal.MTLBuffer, m, n, k int) {
	sink.setPSO(pso)
	sink.setBuf(x, xOff, 0)
	sink.setBuf(wq, wqOff, 1)
	sink.setBuf(scales, scalesOff, 2)
	sink.setBuf(biases, biasesOff, 3)
	sink.setBuf(indices, 0, 4)
	sink.setBuf(out, 0, 5)
	sink.setI32(int32(m), 6)
	sink.setI32(int32(n), 7)
	sink.setI32(int32(k), 8)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint((n + moeGroupedBN - 1) / moeGroupedBN), Height: uint((m + moeGroupedBM - 1) / moeGroupedBM), Depth: 1},
		metal.MTLSize{Width: 32, Height: moeGroupedWN, Depth: moeGroupedWM},
	)
}

// encMoEExpertsGrouped encodes the expert MLP (gate/up → gelu → down) for all K·topK routed
// pairs in sorted-by-expert order, writing mb.downAll in PAIR order — a drop-in replacement
// for the three all-routes gathers + gelu in encMoEBlockQuantBatched. The caller pre-checked
// moeGroupedUsable.
func (s *archDecodeState) encMoEExpertsGrouped(
	enc metal.MTLComputeCommandEncoderObject,
	mb *moeBatchScratch,
	fusedExperts bool,
	expGateUpPacked, expGateUpScales, expGateUpBiases bufView,
	expGatePacked, expGateScales, expGateBiases bufView,
	expUpPacked, expUpScales, expUpBiases bufView,
	expDownPacked, expDownScales, expDownBiases bufView,
	inGS, inBits, expDownGS, expDownBits int,
	numExperts, topK, expertDFF, dModel, pairs int,
) error {
	gs := s.denseBatch.moeGrouped
	if gs == nil {
		gs = &moeGroupedScratch{}
		s.denseBatch.moeGrouped = gs
	}
	if err := gs.ensure(pairs, dModel, expertDFF); err != nil {
		return err
	}
	sortPSO, err := moePairSortPipeline()
	if err != nil {
		return err
	}
	gatherPSO, err := moeGatherRowsPipeline()
	if err != nil {
		return err
	}
	scatterPSO, err := moeScatterRowsPipeline()
	if err != nil {
		return err
	}
	geluPSO, err := moeGeluGateUpPipeline()
	if err != nil {
		return err
	}
	alignM := pairs%moeGroupedBM == 0
	inN := expertDFF
	if fusedExperts {
		inN = 2 * expertDFF
	}
	inPSO, ok := moeGatherQMMRHSPipeline(inGS, inBits, alignM, inN%moeGroupedBN == 0, dModel%moeGroupedBK == 0)
	if !ok {
		return core.NewError("native.encMoEExpertsGrouped: gather_qmm_rhs in-projection pipeline unavailable")
	}
	downPSO, ok := moeGatherQMMRHSPipeline(expDownGS, expDownBits, alignM, dModel%moeGroupedBN == 0, expertDFF%moeGroupedBK == 0)
	if !ok {
		return core.NewError("native.encMoEExpertsGrouped: gather_qmm_rhs down pipeline unavailable")
	}
	sink := encSink{enc}

	// 1. bucket-sort the pairs by expert (one threadgroup).
	sink.setPSO(sortPSO)
	sink.setBuf(mb.idx, 0, 0)
	sink.setBuf(gs.sortedPair, 0, 1)
	sink.setBuf(gs.sortedExpert, 0, 2)
	sink.setI32(int32(pairs), 3)
	sink.setI32(int32(numExperts), 4)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: 1, Height: 1, Depth: 1},
		metal.MTLSize{Width: moePairSortThreads, Height: 1, Depth: 1},
	)

	// 2. gather the pair inputs into sorted order (row = pairToToken[sortedPair[i]]).
	rowBytes := dModel * bf16Size
	sink.setPSO(gatherPSO)
	sink.setBuf(mb.expertIn, 0, 0)
	sink.setBuf(gs.sortedPair, 0, 1)
	sink.setBuf(mb.pairToToken, 0, 2)
	sink.setBuf(gs.xSorted, 0, 3)
	sink.setI32(int32(rowBytes), 4)
	sink.setI32(1, 5)
	sink.dispatchThreads(
		metal.MTLSize{Width: 256, Height: uint(pairs), Depth: 1},
		metal.MTLSize{Width: 256, Height: 1, Depth: 1},
	)

	// 3. gate/up in sorted order — one fused sweep (N = 2·expertDFF) or two split sweeps.
	if fusedExperts {
		emitMoEGatherQMMRHS(sink, inPSO, gs.xSorted, 0, expGateUpPacked.buf, expGateUpPacked.off, expGateUpScales.buf, expGateUpScales.off, expGateUpBiases.buf, expGateUpBiases.off, gs.sortedExpert, gs.guSorted, pairs, 2*expertDFF, dModel)
		sink.setPSO(geluPSO)
		sink.setBuf(gs.guSorted, 0, 0)
		sink.setBuf(gs.gatedSorted, 0, 1)
		sink.setI32(int32(expertDFF), 2)
		sink.dispatchThreads(
			metal.MTLSize{Width: uint(expertDFF), Height: uint(pairs), Depth: 1},
			metal.MTLSize{Width: uint(min(expertDFF, 256)), Height: 1, Depth: 1},
		)
	} else {
		gateSorted, upSorted := gs.guSorted, gs.downSorted // borrow: both pairs×expertDFF fit
		emitMoEGatherQMMRHS(sink, inPSO, gs.xSorted, 0, expGatePacked.buf, expGatePacked.off, expGateScales.buf, expGateScales.off, expGateBiases.buf, expGateBiases.off, gs.sortedExpert, gateSorted, pairs, expertDFF, dModel)
		emitMoEGatherQMMRHS(sink, inPSO, gs.xSorted, 0, expUpPacked.buf, expUpPacked.off, expUpScales.buf, expUpScales.off, expUpBiases.buf, expUpBiases.off, gs.sortedExpert, upSorted, pairs, expertDFF, dModel)
		if err := encGeluGateMulFused(enc, gateSorted, upSorted, gs.gatedSorted, pairs*expertDFF); err != nil {
			return err
		}
	}

	// 4. down in sorted order (input already sorted-contiguous — no second gather).
	emitMoEGatherQMMRHS(sink, downPSO, gs.gatedSorted, 0, expDownPacked.buf, expDownPacked.off, expDownScales.buf, expDownScales.off, expDownBiases.buf, expDownBiases.off, gs.sortedExpert, gs.downSorted, pairs, dModel, expertDFF)

	// 5. scatter the down rows back to PAIR order — the weighted-sum + combine tail then
	// reads exactly what the all-routes path would have written.
	sink.setPSO(scatterPSO)
	sink.setBuf(gs.downSorted, 0, 0)
	sink.setBuf(gs.sortedPair, 0, 1)
	sink.setBuf(mb.downAll, 0, 2)
	sink.setI32(int32(rowBytes), 3)
	sink.dispatchThreads(
		metal.MTLSize{Width: 256, Height: uint(pairs), Depth: 1},
		metal.MTLSize{Width: 256, Height: 1, Depth: 1},
	)
	return nil
}
