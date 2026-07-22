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

// lthnQMVRowsMaxM caps the FLAT register tile: each thread holds M x-slices
// plus M×4 accumulators, and past M=4 the register pressure collapses
// occupancy — measured on the 12B verify: M=3 gpuTotal 18.7ms (vs 20 gather),
// M=5 35.3ms (vs 27.6 gather). Rows past it ride the WIDE halved tile below
// (kernel cases 5..8), or the gather fallback when the wide lane is off.
const lthnQMVRowsMaxM = 4

// lthnQMVRowsWideMaxM caps the halved-tile wide kernel (#53): only ceil(M/2)
// x-slices stay register-live per k-block, which is what the M=5 occupancy
// collapse was about, at the cost of the second half re-touching the k-block's
// weight bytes from L1. 8 covers the MTP verify band (draft block + carry at
// the shipped -draft-block defaults); wider rows keep the gather.
const lthnQMVRowsWideMaxM = 8

var (
	qmvRowsWideOnce sync.Once
	qmvRowsWideOn   bool
)

// qmvRowsWideEnabled reads the LTHN_QMV_ROWS_WIDE lever once. OPT-IN ("1"
// arms rows 5..8 onto the wide tile): the halved tile is byte-identical to
// the per-row qmv (TestLthnQMVRowsWideByteBand) but LOST its live A/B on the
// 26B MTP pair — 400-tok greedy 3-run medians, wide on 120.2 vs
// off 122.7 tok/s with the chained draft armed — the gather's grid-Z L2
// amortisation still wins at these dims (the small-dims pattern the chunked
// tier hit first). The kernel + gate stay as the banked instrument: re-probe
// per geometry before re-defaulting.
func qmvRowsWideEnabled() bool {
	qmvRowsWideOnce.Do(func() { qmvRowsWideOn = os.Getenv("LTHN_QMV_ROWS_WIDE") == "1" })
	return qmvRowsWideOn
}

// qmvRowsTiledCap is the live register-tiled band: the flat tile's 4, or the
// wide tile's 8 when the lane is armed. Every tiled-band decision (the plan
// gate, the byte tier, the fold entry) consults THIS so live, byte-tier and
// ICB-recorded routing stay one rule.
func qmvRowsTiledCap() int {
	if qmvRowsWideEnabled() {
		return lthnQMVRowsWideMaxM
	}
	return lthnQMVRowsMaxM
}

type lthnQMVRowsKey struct {
	groupSize, bits, m int
	// general selects the qmv_impl M-variant (lthn_qmv_rows_general) — the
	// byte-parity twin of the NON-fast per-row route, for dims outside the
	// qmv_fast envelope (outDim%8==0 && inDim%512==0). false = the fast twin.
	general bool
}

// lthnQMVRowsKernelName derives the metallib host name a key resolves — ONE
// place, consulted by both the plain and ICB pipeline resolvers, so the
// fast/general twin split can never skew between live encode and ICB record.
func lthnQMVRowsKernelName(key lthnQMVRowsKey) string {
	variant := ""
	if key.general {
		variant = "_general"
	}
	return core.Sprintf("lthn_qmv_rows%s_bfloat16_t_gs_%d_b_%d", variant, key.groupSize, key.bits)
}

// qmvRowsTiledKeyFor is THE tiled-band rule: the register-tiled kernel key for
// one M-sized dispatch at this geometry, or ok=false when no tiled kernel
// serves it. On the qmv_fast envelope (outDim%8==0 && inDim%512==0 — the same
// rule qmvBF16KernelName routes the per-row decode by) M rides the fast twin up
// to qmvRowsTiledCap() (wide cases included when the lane is armed). Every
// other dim routes per-row to qmv_impl, so M rides the general twin
// (lthn_qmv_rows_general — qmv_impl's M-variant, byte-identical to THAT
// oracle), capped at the flat tile's 4: there is no general wide variant
// (docs/design-qmv-rows-unaligned.md — the fast wide twin was live-refuted and
// the unaligned rows>4 band is chunk-composed byte-tier only). The plan gate,
// the chunked fold, the byte-exact encoder and the servability probe ALL
// consult this, so live, record and probe stay one rule.
func qmvRowsTiledKeyFor(m, outDim, inDim, gs, bits int) (lthnQMVRowsKey, bool) {
	if m < 2 {
		return lthnQMVRowsKey{}, false
	}
	if outDim%8 == 0 && inDim%512 == 0 {
		if m > qmvRowsTiledCap() {
			return lthnQMVRowsKey{}, false
		}
		return lthnQMVRowsKey{groupSize: gs, bits: bits, m: m}, true
	}
	if m > lthnQMVRowsMaxM {
		return lthnQMVRowsKey{}, false
	}
	return lthnQMVRowsKey{groupSize: gs, bits: bits, m: m, general: true}, true
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
	fc := metal.NewMTLFunctionConstantValues()
	m := int32(key.m)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&m), metal.MTLDataTypeInt, 0)
	fn, err := customLibrary.NewFunctionWithNameConstantValuesError(lthnQMVRowsKernelName(key), fc)
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

// qmvRowsPlan is the routing decision encQMVRowsBF16At takes for a geometry —
// extracted so the verify-tail ICB recorder consults the SAME decision and
// records exactly the dispatch the live path encodes (record==live by
// construction, never by parallel logic).
type qmvRowsPlan struct {
	tiled     bool // register-tiled lthn_qmv_rows, one dispatch (rows ≤ qmvRowsTiledCap())
	tiledKey  lthnQMVRowsKey
	gatherKey lthnGatherQMVKey
}

var (
	qmvChunksOnce sync.Once
	qmvChunksOn   bool
)

// qmvChunksEnabled reads the LTHN_QMV_CHUNKS kill switch once: "0" removes the
// chunked byte-tier route (the laneSet GEMM fold then declines quant at K >
// lthnQMVRowsMaxM); anything else (including unset) keeps it.
func qmvChunksEnabled() bool {
	qmvChunksOnce.Do(func() { qmvChunksOn = os.Getenv("LTHN_QMV_CHUNKS") != "0" })
	return qmvChunksOn
}

// qmvRowsChunks decomposes rows (> lthnQMVRowsMaxM) into tiled chunk sizes in
// {2,3,4} — never 1: a 1-row remainder converts the final [4,1] into [3,2] so
// every chunk rides lthn_qmv_rows. Each chunk's rows are byte-identical to the
// per-row qmv (row math is row-independent), which is what extends the laneSet
// GEMM byte tier past K=4 (encQMVRowsBF16ChunkedAt). This is a BYTE-TIER route
// only: for pure throughput the gather beats it at small dims (e2b MTP verify
// A/B — chunked 178/180 vs gather 222 tok/s; grid-Z L2-amortises a
// small weight stream better than two half-parallel tiled dispatches), so the
// plain encQMVRowsBF16At keeps the gather for rows > lthnQMVRowsMaxM.
func qmvRowsChunks(rows int) []int {
	if rows <= lthnQMVRowsMaxM {
		return nil
	}
	var out []int
	r := rows
	for r > lthnQMVRowsMaxM+1 {
		out = append(out, lthnQMVRowsMaxM)
		r -= lthnQMVRowsMaxM
	}
	if r == lthnQMVRowsMaxM+1 {
		out = append(out, 3, 2)
	} else {
		out = append(out, r)
	}
	return out
}

// encQMVRowsBF16ChunkedAt is the BYTE-TIER multi-row qmv for rows >
// lthnQMVRowsMaxM: qmvRowsChunks tiled dispatches of 2..4 rows each, every
// chunk byte-identical to the per-row qmv — the fast twin on the qmv_fast
// envelope, the general (qmv_impl M-variant) twin on every other dim
// (qmvRowsTiledKeyFor picks per chunk; 2..4 chunks always fit the general flat
// cap). handled=false when a chunk key/PSO or the kill switch declines — the
// caller keeps its throughput route. Only the laneSet GEMM fold
// (projectRowsByteTier) and the byte-exact MoE lane take this: the plain
// encQMVRowsBF16At keeps the gather, which wins on throughput at small dims.
func encQMVRowsBF16ChunkedAt(enc metal.MTLComputeCommandEncoder, wq, scales, biases, in, out metal.MTLBuffer, wqOff, scalesOff, biasesOff, inOff, outOff uint, rows, outDim, inDim, gs, bits int) (bool, error) {
	if rows <= lthnQMVRowsMaxM || rows > qmvRowsMax || !qmvChunksEnabled() {
		return false, nil
	}
	chunks := qmvRowsChunks(rows)
	psos := make([]metal.MTLComputePipelineState, len(chunks))
	for i, m := range chunks {
		key, ok := qmvRowsTiledKeyFor(m, outDim, inDim, gs, bits)
		if !ok {
			return false, nil
		}
		pso, ok := lthnQMVRowsPipeline(key)
		if !ok {
			return false, nil
		}
		psos[i] = pso
	}
	row := 0
	for i, m := range chunks {
		emitQMVRowsTiled(encSink{enc}, psos[i], wq, wqOff, scales, scalesOff, biases, biasesOff, in, inOff+uint(row*inDim*bf16Size), out, outOff+uint(row*outDim*bf16Size), inDim, outDim)
		row += m
	}
	return true, nil
}

// qmvRowsPlanFor reports the multi-row route for a geometry: ok=false means no
// multi-row kernel applies (the caller keeps its qmm_t/per-row route). The
// tiled preference is probed against the LIVE pipeline cache — kernel
// availability, not just geometry, picks the branch, exactly as the encoder
// does.
func qmvRowsPlanFor(rows, outDim, inDim, gs, bits int) (qmvRowsPlan, bool) {
	if rows < 2 || rows > qmvRowsMax || inDim <= 0 || gs <= 0 || inDim%gs != 0 || (inDim*bits)%32 != 0 {
		return qmvRowsPlan{}, false
	}
	// Each envelope gets its per-row oracle's OWN M-variant (qmvRowsTiledKeyFor —
	// the one tiled-band rule): fast dims (outDim%8==0 && inDim%512==0, the
	// qmvBF16KernelName rule) ride qmv_fast_impl's M-variant, rows 5..8
	// resolving the WIDE halved-tile cases behind the same host name when the
	// lane is armed; every other dim rides qmv_impl's M-variant
	// (lthn_qmv_rows_general, flat 2..4 only). A geometry with no tiled key —
	// or a missing PSO — keeps the gather fallback.
	if key, ok := qmvRowsTiledKeyFor(rows, outDim, inDim, gs, bits); ok {
		if _, ok := lthnQMVRowsPipeline(key); ok {
			return qmvRowsPlan{tiled: true, tiledKey: key}, true
		}
	}
	key := lthnGatherQMVKey{
		groupSize: gs, bits: bits, expertRows: 0, batchedX: true,
		fast: outDim%8 == 0 && inDim%512 == 0,
	}
	if _, ok := lthnGatherQMVPipeline(key); !ok {
		return qmvRowsPlan{}, false
	}
	if _, _, ok := qmvRowsIndexBuffers(); !ok {
		return qmvRowsPlan{}, false
	}
	return qmvRowsPlan{gatherKey: key}, true
}

// emitQMVRowsTiled records the register-tiled multi-row qmv through any sink —
// binding ABI w=0, scales=1, biases=2, in=3, out=4, K=5, N=6 on the qmv-fast
// grid. One body behind encQMVRowsBF16At's tiled branch (live) and the
// verify-tail ICB recorder; pso caller-provided (ICB variant differs).
func emitQMVRowsTiled[S dispatchSink](sink S, pso metal.MTLComputePipelineState, wq metal.MTLBuffer, wqOff uint, scales metal.MTLBuffer, scalesOff uint, biases metal.MTLBuffer, biasesOff uint, in metal.MTLBuffer, inOff uint, out metal.MTLBuffer, outOff uint, inDim, outDim int) {
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
}

var (
	lthnQMVRowsICBPSOMu    sync.Mutex
	lthnQMVRowsICBPSOCache = map[lthnQMVRowsKey]metal.MTLComputePipelineState{}
)

// lthnQMVRowsPipelineICB is lthnQMVRowsPipeline with supportIndirectCommandBuffers
// set — the variant the MTP verify-tail ICB records.
func lthnQMVRowsPipelineICB(key lthnQMVRowsKey) (metal.MTLComputePipelineState, bool) {
	lthnQMVRowsICBPSOMu.Lock()
	defer lthnQMVRowsICBPSOMu.Unlock()
	if pso, ok := lthnQMVRowsICBPSOCache[key]; ok {
		return pso, pso != nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		lthnQMVRowsICBPSOCache[key] = nil
		return nil, false
	}
	fc := metal.NewMTLFunctionConstantValues()
	m := int32(key.m)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&m), metal.MTLDataTypeInt, 0)
	fn, err := customLibrary.NewFunctionWithNameConstantValuesError(lthnQMVRowsKernelName(key), fc)
	if err != nil || fn == nil || fn.GetID() == 0 {
		lthnQMVRowsICBPSOCache[key] = nil
		return nil, false
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, perr := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if perr != nil {
		lthnQMVRowsICBPSOCache[key] = nil
		return nil, false
	}
	lthnQMVRowsICBPSOCache[key] = pso
	return pso, true
}

// encQMVRowsBF16At encodes the multi-row qmv, reporting handled=false (no
// encode) when the geometry has no kernel so the caller keeps its
// qmm_t/per-row route. in rows are contiguous bf16 at inOff + z·inDim·2; out
// rows land at outOff + z·outDim·2. Rows 2..qmvRowsTiledCap() take one
// register-tiled lthn_qmv_rows dispatch (flat tile to 4, halved wide tile to
// 8 — the weight stream read ONCE); wider rows ride the lean gather kernel
// (grid-Z, qmv_fast bytes, weight re-streamed per row — the THROUGHPUT winner
// at small dims; the byte-tier chunked variant is encQMVRowsBF16ChunkedAt,
// fold-only).
// Byte identity: the tiled kernel is the per-row oracle's OWN M-variant —
// qmv_fast_impl's on fast-twin dims (outDim%8==0 && inDim%512==0, the rule
// qmvBF16KernelName routes the per-row decode by), qmv_impl's
// (lthn_qmv_rows_general, rows 2..4) everywhere else — qmvRowsTiledKeyFor
// matches twin to envelope, so a tiled encode is byte-identical to the per-row
// qmv row for row on EVERY dim it serves. The refuted predecessor
// (one packs=1 kernel claiming parity with BOTH per-row twins; ~1 ulp
// value-dependent accumulation drift on fast dims) is exactly why the twins
// stay separate and envelope-matched. The gather fallback remains a THROUGHPUT
// route (MTP verify draft blocks), not a byte-identity claim —
// qmvProjector.rowsByteTier accepts tiled plans only.
func encQMVRowsBF16At(enc metal.MTLComputeCommandEncoder, wq, scales, biases, in, out metal.MTLBuffer, wqOff, scalesOff, biasesOff, inOff, outOff uint, rows, outDim, inDim, gs, bits int) (bool, error) {
	plan, ok := qmvRowsPlanFor(rows, outDim, inDim, gs, bits)
	if !ok {
		return false, nil
	}
	if plan.tiled {
		pso, ok := lthnQMVRowsPipeline(plan.tiledKey)
		if !ok {
			return false, nil
		}
		emitQMVRowsTiled(encSink{enc}, pso, wq, wqOff, scales, scalesOff, biases, biasesOff, in, inOff, out, outOff, inDim, outDim)
		return true, nil
	}
	pso, ok := lthnGatherQMVPipeline(plan.gatherKey)
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
