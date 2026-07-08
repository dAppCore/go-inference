// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// sdpaSingleCellDispatches counts paged-SDPA decodes that ran the single-cell
// P1-final fast path (pass 2 skipped, #340). Engagement counter for the A/B
// tests — a compare that never engaged the lane proves nothing.
var sdpaSingleCellDispatches atomic.Int64

type sdpaPagedP1Params struct {
	NHeads      uint32
	NKVHeads    uint32
	HeadDim     uint32
	PageLen     uint32
	KHeadStride uint32
	KSeqStride  uint32
	VHeadStride uint32
	VSeqStride  uint32
	SplitRows   uint32
	Splits      uint32
	CellBase    uint32
	CellCount   uint32
	Scale       float32
}

// sdpaPagedSplitRowsOverride: LTHN_SDPA_SPLIT probe lever for the #356 grain
// sweep — 0 (unset) keeps the computed grain.
var sdpaPagedSplitRowsOverride = func() int {
	if v := os.Getenv("LTHN_SDPA_SPLIT"); v != "" {
		if r := core.ParseInt(v, 10, 32); r.OK {
			return int(r.Value.(int64))
		}
	}
	return 0
}()

// sdpaPagedSplitRows is the depth-parallelism grain: each pass-1 threadgroup owns one
// split window of a page, so the grid grows with context (nHeads × ceil(len/splitRows)
// threadgroups) instead of pinning at nHeads while simdgroups serialise the rows.
const sdpaPagedSplitRows = 256

type sdpaPagedP2Params struct {
	HeadDim   uint32
	CellCount uint32
}

// sdpaPagedDecodeScratch holds the parallel two-pass partials: one (max, sum,
// acc[headDim]) cell per (head, page). Pass 1 fully overwrites the cells it owns
// and pass 2 reads only [0, pageCount) cells, so no host reset is needed between
// tokens — ensure only reallocates when the page count outgrows capacity. The
// buffers stay hazard-TRACKED: the per-layer page count is small (pageSize 256),
// each pass-1 dispatch is ~90µs, and measurement showed tracked serialisation of
// those few dispatches costs the same as untracked-plus-explicit-barrier — so the
// simpler tracked form wins (ordering pass 1 → pass 2 comes for free).
type sdpaPagedDecodeScratch struct {
	nHeads, headDim, maxPages int
	maxs, sums, acc           metal.MTLBuffer
}

var (
	sdpaPagedP1PSOOnce sync.Once
	sdpaPagedP1PSO     metal.MTLComputePipelineState
	sdpaPagedP1PSOErr  error

	sdpaPagedP2PSOOnce sync.Once
	sdpaPagedP2PSO     metal.MTLComputePipelineState
	sdpaPagedP2PSOErr  error

	sdpaPagedP1FinalPSOOnce sync.Once
	sdpaPagedP1FinalPSO     metal.MTLComputePipelineState
	sdpaPagedP1FinalPSOErr  error

	sdpaPagedP1GQA2PSOOnce sync.Once
	sdpaPagedP1GQA2PSO     metal.MTLComputePipelineState
	sdpaPagedP1GQA2PSOErr  error

	sdpaPagedP1FinalGQA2PSOOnce sync.Once
	sdpaPagedP1FinalGQA2PSO     metal.MTLComputePipelineState
	sdpaPagedP1FinalGQA2PSOErr  error
)

func newSDPAPagedDecodeScratch(nHeads, headDim int) (*sdpaPagedDecodeScratch, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if nHeads <= 0 || headDim <= 0 {
		return nil, core.NewError("native.newSDPAPagedDecodeScratch: dimensions must be > 0")
	}
	s := &sdpaPagedDecodeScratch{nHeads: nHeads, headDim: headDim}
	if err := s.ensure(nHeads, headDim, 1); err != nil {
		return nil, err
	}
	return s, nil
}

// ensure sizes the per-(head, page) partials for at least pages cells, reallocating
// only on growth or shape change.
func (s *sdpaPagedDecodeScratch) ensure(nHeads, headDim, pages int) error {
	if s == nil {
		return core.NewError("native.sdpaPagedDecodeScratch.ensure: nil scratch")
	}
	if nHeads <= 0 || headDim <= 0 || pages <= 0 {
		return core.NewError("native.sdpaPagedDecodeScratch.ensure: dimensions must be > 0")
	}
	if s.maxs != nil && s.nHeads == nHeads && s.headDim == headDim && pages <= s.maxPages {
		return nil
	}
	capPages := pages
	if s.maxPages*2 > capPages && s.nHeads == nHeads && s.headDim == headDim {
		capPages = s.maxPages * 2 // grow geometrically as the context adds pages
	}
	cells := nHeads * capPages
	maxs := device.NewBufferWithLengthOptions(uint(cells*4), metal.MTLResourceStorageModeShared)
	sums := device.NewBufferWithLengthOptions(uint(cells*4), metal.MTLResourceStorageModeShared)
	acc := device.NewBufferWithLengthOptions(uint(cells*headDim*4), metal.MTLResourceStorageModeShared)
	if maxs == nil || sums == nil || acc == nil ||
		maxs.GetID() == 0 || sums.GetID() == 0 || acc.GetID() == 0 {
		return core.NewError("native.sdpaPagedDecodeScratch.ensure: failed to allocate scratch buffers")
	}
	s.nHeads, s.headDim, s.maxPages = nHeads, headDim, capPages
	s.maxs, s.sums, s.acc = maxs, sums, acc
	return nil
}

func sdpaPagedP1Pipeline() (metal.MTLComputePipelineState, error) {
	sdpaPagedP1PSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			sdpaPagedP1PSOErr = core.NewError("native.sdpaPagedP1Pipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_sdpa_paged_p1_bf16")
		if fn == nil || fn.GetID() == 0 {
			sdpaPagedP1PSOErr = core.NewError("native.sdpaPagedP1Pipeline: kernel lthn_sdpa_paged_p1_bf16 not found")
			return
		}
		sdpaPagedP1PSO, sdpaPagedP1PSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return sdpaPagedP1PSO, sdpaPagedP1PSOErr
}

// sdpaPagedP1FinalPipeline resolves the single-cell P1 variant — the same pass-1
// body with the final normalise applied at store (see the kernel's fast-path
// notes). A separate host_name, deliberately: a STALE metallib fails this lookup
// and the plan falls back to two passes, rather than silently running a kernel
// without the branch.
func sdpaPagedP1FinalPipeline() (metal.MTLComputePipelineState, error) {
	sdpaPagedP1FinalPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			sdpaPagedP1FinalPSOErr = core.NewError("native.sdpaPagedP1FinalPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_sdpa_paged_p1_final_bf16")
		if fn == nil || fn.GetID() == 0 {
			sdpaPagedP1FinalPSOErr = core.NewError("native.sdpaPagedP1FinalPipeline: kernel lthn_sdpa_paged_p1_final_bf16 not found")
			return
		}
		sdpaPagedP1FinalPSO, sdpaPagedP1FinalPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return sdpaPagedP1FinalPSO, sdpaPagedP1FinalPSOErr
}

// sdpaPagedP1GQA2Pipeline resolves the GQA-shared pass 1: one threadgroup per
// (KV head, split) streams rows once for BOTH query heads of the group (#356 —
// the per-head kernel paid 2x the bandwidth floor on GQA-2 models). Separate
// host_name so a stale metallib falls back to the per-head kernel loudly.
func sdpaPagedP1GQA2Pipeline() (metal.MTLComputePipelineState, error) {
	sdpaPagedP1GQA2PSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			sdpaPagedP1GQA2PSOErr = core.NewError("native.sdpaPagedP1GQA2Pipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_sdpa_paged_p1_gqa2_bf16")
		if fn == nil || fn.GetID() == 0 {
			sdpaPagedP1GQA2PSOErr = core.NewError("native.sdpaPagedP1GQA2Pipeline: kernel lthn_sdpa_paged_p1_gqa2_bf16 not found")
			return
		}
		sdpaPagedP1GQA2PSO, sdpaPagedP1GQA2PSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return sdpaPagedP1GQA2PSO, sdpaPagedP1GQA2PSOErr
}

func sdpaPagedP1FinalGQA2Pipeline() (metal.MTLComputePipelineState, error) {
	sdpaPagedP1FinalGQA2PSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			sdpaPagedP1FinalGQA2PSOErr = core.NewError("native.sdpaPagedP1FinalGQA2Pipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_sdpa_paged_p1_final_gqa2_bf16")
		if fn == nil || fn.GetID() == 0 {
			sdpaPagedP1FinalGQA2PSOErr = core.NewError("native.sdpaPagedP1FinalGQA2Pipeline: kernel lthn_sdpa_paged_p1_final_gqa2_bf16 not found")
			return
		}
		sdpaPagedP1FinalGQA2PSO, sdpaPagedP1FinalGQA2PSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return sdpaPagedP1FinalGQA2PSO, sdpaPagedP1FinalGQA2PSOErr
}

func sdpaPagedP2Pipeline() (metal.MTLComputePipelineState, error) {
	sdpaPagedP2PSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			sdpaPagedP2PSOErr = core.NewError("native.sdpaPagedP2Pipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_sdpa_paged_p2_bf16")
		if fn == nil || fn.GetID() == 0 {
			sdpaPagedP2PSOErr = core.NewError("native.sdpaPagedP2Pipeline: kernel lthn_sdpa_paged_p2_bf16 not found")
			return
		}
		sdpaPagedP2PSO, sdpaPagedP2PSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return sdpaPagedP2PSO, sdpaPagedP2PSOErr
}

func encSDPAPagedDecode(
	enc metal.MTLComputeCommandEncoder,
	q metal.MTLBuffer,
	keyPages, valuePages []metal.MTLBuffer,
	pageLens, pageSpans []int,
	out metal.MTLBuffer,
	scratch *sdpaPagedDecodeScratch,
	nHeads, nKVHeads, headDim int,
	scale float32,
) error {
	if len(pageLens) != len(keyPages) || len(pageSpans) != len(keyPages) {
		return core.NewError("native.encSDPAPagedDecode: page lengths and spans must match page buffers")
	}
	keyHeadStrides := make([]int, len(pageSpans))
	keySeqStrides := make([]int, len(pageSpans))
	valueHeadStrides := make([]int, len(pageSpans))
	valueSeqStrides := make([]int, len(pageSpans))
	for i, span := range pageSpans {
		if span < pageLens[i] {
			return core.NewError("native.encSDPAPagedDecode: visible page length must fit physical span")
		}
		keyHeadStrides[i] = span * headDim
		keySeqStrides[i] = headDim
		valueHeadStrides[i] = span * headDim
		valueSeqStrides[i] = headDim
	}
	return encSDPAPagedDecodeStrided(enc, q, keyPages, valuePages, pageLens, keyHeadStrides, keySeqStrides, valueHeadStrides, valueSeqStrides, out, scratch, nHeads, nKVHeads, headDim, scale)
}

func encSDPAPagedDecodeStrided(
	enc metal.MTLComputeCommandEncoder,
	q metal.MTLBuffer,
	keyPages, valuePages []metal.MTLBuffer,
	pageLens, keyHeadStrides, keySeqStrides, valueHeadStrides, valueSeqStrides []int,
	out metal.MTLBuffer,
	scratch *sdpaPagedDecodeScratch,
	nHeads, nKVHeads, headDim int,
	scale float32,
) error {
	plan, err := buildSDPAPagedDecodePlan(q, keyPages, valuePages, pageLens, keyHeadStrides, keySeqStrides, valueHeadStrides, valueSeqStrides, out, scratch, nHeads, nKVHeads, headDim, scale)
	if err != nil {
		return err
	}
	// serial pass: buffer hazard tracking orders pass 1 → pass 2; a CONCURRENT pass must
	// place an explicit barrier between emitP1s and emitP2 instead.
	plan.emitP1s(enc)
	plan.emitP2(enc)
	return nil
}

// sdpaPagedDecodePlan is the validated paged-decode SDPA dispatch: pass 1 (per-page split
// windows) and pass 2 (the cell merge) emit separately so the concurrent attention pass can
// barrier between them.
type sdpaPagedDecodePlan struct {
	q, out               metal.MTLBuffer
	keyPages, valuePages []metal.MTLBuffer
	pageLens             []int
	kHead, kSeq          []int
	vHead, vSeq          []int
	scratch              *sdpaPagedDecodeScratch
	p1PSO, p2PSO         metal.MTLComputePipelineState
	// gqaShared: pass 1 runs one threadgroup per (KV head, split) computing both
	// query heads of the GQA-2 group over rows read ONCE (#356); the grid width
	// shrinks to nKVHeads x splits (splitRows halves to keep the threadgroup
	// count) and the kernel derives h = kvh*2 + g.
	gqaShared bool
	splitRows int
	// p1FinalPSO drives the single-cell fast path: pass 1 writes the final
	// normalised row directly and emitP2 no-ops. Set only when cellCount == 1
	// and the variant kernel resolved (see singleCell).
	p1FinalPSO         metal.MTLComputePipelineState
	singleCell         bool
	nHeads, nKVHeads   int
	headDim, cellCount int
	scale              float32
}

func buildSDPAPagedDecodePlan(
	q metal.MTLBuffer,
	keyPages, valuePages []metal.MTLBuffer,
	pageLens, keyHeadStrides, keySeqStrides, valueHeadStrides, valueSeqStrides []int,
	out metal.MTLBuffer,
	scratch *sdpaPagedDecodeScratch,
	nHeads, nKVHeads, headDim int,
	scale float32,
) (sdpaPagedDecodePlan, error) {
	if nHeads <= 0 || nKVHeads <= 0 || headDim <= 0 {
		return sdpaPagedDecodePlan{}, core.NewError("native.encSDPAPagedDecodeStrided: dimensions must be > 0")
	}
	if nHeads%nKVHeads != 0 {
		return sdpaPagedDecodePlan{}, core.NewError("native.encSDPAPagedDecodeStrided: nHeads must be a multiple of nKVHeads")
	}
	if q == nil || q.GetID() == 0 || out == nil || out.GetID() == 0 {
		return sdpaPagedDecodePlan{}, core.NewError("native.encSDPAPagedDecodeStrided: nil input/output buffer")
	}
	if len(keyPages) == 0 || len(keyPages) != len(valuePages) || len(keyPages) != len(pageLens) ||
		len(keyPages) != len(keyHeadStrides) || len(keyPages) != len(keySeqStrides) ||
		len(keyPages) != len(valueHeadStrides) || len(keyPages) != len(valueSeqStrides) {
		return sdpaPagedDecodePlan{}, core.NewError("native.encSDPAPagedDecodeStrided: page buffers and strides must be non-empty and matched")
	}
	for i := range keyPages {
		if keyPages[i] == nil || keyPages[i].GetID() == 0 || valuePages[i] == nil || valuePages[i].GetID() == 0 {
			return sdpaPagedDecodePlan{}, core.NewError("native.encSDPAPagedDecodeStrided: nil page buffer")
		}
		if pageLens[i] <= 0 {
			// allocated-but-unwritten pages are a legitimate cache state (slot() allocates
			// eagerly up to the written page; a ring's prefill sync fills only the visible
			// rows): they contribute zero cells and pass 1 skips them.
			continue
		}
		if keyHeadStrides[i] <= 0 || keySeqStrides[i] <= 0 || valueHeadStrides[i] <= 0 || valueSeqStrides[i] <= 0 {
			return sdpaPagedDecodePlan{}, core.NewError("native.encSDPAPagedDecodeStrided: page strides must be > 0")
		}
	}
	// the lane slicing owns headDim/32 dims per lane — every shipped head dim (64,
	// 128, 256, 512) is a multiple of 32; reject anything else loudly rather than
	// silently dropping tail dims.
	if headDim%32 != 0 || headDim/32 > 16 {
		return sdpaPagedDecodePlan{}, core.NewError("native.encSDPAPagedDecodeStrided: headDim must be a multiple of 32, at most 512")
	}
	// GQA-2 models take the row-shared pass 1 (both query heads of a KV group
	// computed over rows read ONCE — half the K/V traffic, #356). The grid loses
	// its query-head factor, so the split grain HALVES to keep the threadgroup
	// count — 8 kvHeads x 8 splits was 64 threadgroups on an 80-core GPU, and
	// the first cut of this kernel measured SLOWER than the per-head one (4.15
	// vs 3.82 ms/token) purely from that under-occupancy. Gated on headDim<=256
	// (every gqa2 model's shape): the kernel sizes its two accumulator sets for
	// per<=8 to keep the register budget at the per-head kernel's level.
	gqaShared := false
	totalRows := 0
	for i := range pageLens {
		if pageLens[i] > 0 {
			totalRows += pageLens[i]
		}
	}
	// Short windows keep the per-head kernel: a near-single-cell dispatch is
	// already under-occupied, and halving its threadgroups measured 143.7 to
	// 138.1 tok/s on the 26B short decode. The traffic halving only matters
	// once the scan is deep.
	if nHeads == 2*nKVHeads && headDim <= 256 && totalRows > sdpaPagedSplitRows {
		if _, gerr := sdpaPagedP1GQA2Pipeline(); gerr == nil {
			gqaShared = true
		}
	}
	splitRows := sdpaPagedSplitRows
	if gqaShared {
		splitRows = sdpaPagedSplitRows / 2
	}
	if v := sdpaPagedSplitRowsOverride; v > 0 { // LTHN_SDPA_SPLIT probe (#356)
		splitRows = v
	}
	// each page fans out over ceil(len/splitRows) independent split cells — the grid grows
	// with context (#339: 16 fixed threadgroups measured 12.4 ms/token of attention at
	// position ~3500; split windows keep the whole GPU busy at any depth).
	cellCount := 0
	for i := range keyPages {
		if pageLens[i] > 0 {
			cellCount += (pageLens[i] + splitRows - 1) / splitRows
		}
	}
	if err := scratch.ensure(nHeads, headDim, cellCount); err != nil {
		return sdpaPagedDecodePlan{}, err
	}
	p1PSO, err := sdpaPagedP1Pipeline()
	if err != nil {
		return sdpaPagedDecodePlan{}, err
	}
	p2PSO, err := sdpaPagedP2Pipeline()
	if err != nil {
		return sdpaPagedDecodePlan{}, err
	}
	if gqaShared {
		if pso, gerr := sdpaPagedP1GQA2Pipeline(); gerr == nil {
			p1PSO = pso
		} else {
			gqaShared = false
		}
	}
	// Single-cell fast path (#340): the whole visible cache is one split window of
	// one page, so pass 2's merge of one cell is an identity rescale — pass 1
	// writes the final row and pass 2 is skipped. Falls back to two passes when
	// the variant kernel is missing (stale metallib) or levered off.
	var p1FinalPSO metal.MTLComputePipelineState
	singleCell := false
	if cellCount == 1 && !sdpaSingleCellDisabled {
		if gqaShared {
			if pso, ferr := sdpaPagedP1FinalGQA2Pipeline(); ferr == nil {
				p1FinalPSO, singleCell = pso, true
			}
		}
		if !singleCell {
			if pso, ferr := sdpaPagedP1FinalPipeline(); ferr == nil {
				p1FinalPSO, singleCell = pso, true
				gqaShared = false // per-head final variant: dispatch per query head
			}
		}
	}
	// returned BY VALUE (borrowed slices only): the callers hold it as a local, so it stays
	// on the stack — the sampled-retained allocation budgets count every per-layer alloc.
	return sdpaPagedDecodePlan{
		q: q, out: out,
		keyPages: keyPages, valuePages: valuePages, pageLens: pageLens,
		kHead: keyHeadStrides, kSeq: keySeqStrides, vHead: valueHeadStrides, vSeq: valueSeqStrides,
		scratch: scratch, p1PSO: p1PSO, p2PSO: p2PSO,
		p1FinalPSO: p1FinalPSO, singleCell: singleCell, gqaShared: gqaShared,
		splitRows: splitRows,
		nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim, cellCount: cellCount,
		scale: scale,
	}, nil
}

// emitP1s encodes pass 1: one dispatch per page, nHeads × splits threadgroups, each writing
// its OWN (head, cell) partial — every dispatch independent of the others. On a single-cell
// plan the one dispatch runs the P1-final variant instead, writing the normalised row
// straight to out (emitP2 then no-ops).
func (p *sdpaPagedDecodePlan) emitP1s(enc metal.MTLComputeCommandEncoder) {
	if p.singleCell {
		for i := range p.keyPages {
			if p.pageLens[i] <= 0 {
				continue // empty page: zero cells (see the plan validation)
			}
			sdpaSingleCellDispatches.Add(1)
			params := sdpaPagedP1Params{
				NHeads:      uint32(p.nHeads),
				NKVHeads:    uint32(p.nKVHeads),
				HeadDim:     uint32(p.headDim),
				PageLen:     uint32(p.pageLens[i]),
				KHeadStride: uint32(p.kHead[i]),
				KSeqStride:  uint32(p.kSeq[i]),
				VHeadStride: uint32(p.vHead[i]),
				VSeqStride:  uint32(p.vSeq[i]),
				SplitRows:   uint32(p.splitRows),
				Splits:      1,
				CellBase:    0,
				CellCount:   1,
				Scale:       p.scale,
			}
			setPSO(enc, p.p1FinalPSO)
			setBuf(enc, p.q, 0, 0)
			setBuf(enc, p.keyPages[i], 0, 1)
			setBuf(enc, p.valuePages[i], 0, 2)
			setBuf(enc, p.out, 0, 3)
			setBytes(enc, unsafe.Pointer(&params), uint(unsafe.Sizeof(params)), 4)
			tgWidth := p.nHeads
			if p.gqaShared {
				tgWidth = p.nKVHeads
			}
			dispatchThreadgroups(enc,
				metal.MTLSize{Width: uint(tgWidth), Height: 1, Depth: 1},
				metal.MTLSize{Width: 256, Height: 1, Depth: 1}, // 8 simdgroups split the window rows
			)
			return
		}
		return
	}
	cellBase := 0
	for i := range p.keyPages {
		if p.pageLens[i] <= 0 {
			continue // empty page: zero cells (see the plan validation)
		}
		splits := (p.pageLens[i] + p.splitRows - 1) / p.splitRows
		params := sdpaPagedP1Params{
			NHeads:      uint32(p.nHeads),
			NKVHeads:    uint32(p.nKVHeads),
			HeadDim:     uint32(p.headDim),
			PageLen:     uint32(p.pageLens[i]),
			KHeadStride: uint32(p.kHead[i]),
			KSeqStride:  uint32(p.kSeq[i]),
			VHeadStride: uint32(p.vHead[i]),
			VSeqStride:  uint32(p.vSeq[i]),
			SplitRows:   uint32(p.splitRows),
			Splits:      uint32(splits),
			CellBase:    uint32(cellBase),
			CellCount:   uint32(p.cellCount),
			Scale:       p.scale,
		}
		cellBase += splits
		setPSO(enc, p.p1PSO)
		setBuf(enc, p.q, 0, 0)
		setBuf(enc, p.keyPages[i], 0, 1)
		setBuf(enc, p.valuePages[i], 0, 2)
		setBuf(enc, p.scratch.maxs, 0, 3)
		setBuf(enc, p.scratch.sums, 0, 4)
		setBuf(enc, p.scratch.acc, 0, 5)
		setBytes(enc, unsafe.Pointer(&params), uint(unsafe.Sizeof(params)), 6)
		tgWidth := p.nHeads
		if p.gqaShared {
			tgWidth = p.nKVHeads
		}
		dispatchThreadgroups(enc,
			metal.MTLSize{Width: uint(tgWidth * splits), Height: 1, Depth: 1},
			metal.MTLSize{Width: 256, Height: 1, Depth: 1}, // 8 simdgroups split the window rows
		)
	}
}

// emitP2 encodes pass 2: the per-head merge across every partial cell. A
// single-cell plan already wrote the final row in pass 1 — nothing to merge.
func (p *sdpaPagedDecodePlan) emitP2(enc metal.MTLComputeCommandEncoder) {
	if p.singleCell {
		return
	}
	p2 := sdpaPagedP2Params{HeadDim: uint32(p.headDim), CellCount: uint32(p.cellCount)}
	setPSO(enc, p.p2PSO)
	setBuf(enc, p.scratch.maxs, 0, 0)
	setBuf(enc, p.scratch.sums, 0, 1)
	setBuf(enc, p.scratch.acc, 0, 2)
	setBuf(enc, p.out, 0, 3)
	setBytes(enc, unsafe.Pointer(&p2), uint(unsafe.Sizeof(p2)), 4)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: uint(p.nHeads), Height: 1, Depth: 1},
		metal.MTLSize{Width: 256, Height: 1, Depth: 1}, // one thread per head dim
	)
}

func sdpaPagedTransientBuffer(b []byte, pinners *[]*runtime.Pinner) metal.MTLBuffer {
	if buf, ok := registeredPinnedNoCopyBytes(b); ok {
		return buf
	}
	buf, pinner, noCopy := residentNoCopyBytes(b)
	if noCopy && pinner != nil {
		*pinners = append(*pinners, pinner)
	}
	return buf
}

func sdpaPagedOutputBuffer(out []byte) (metal.MTLBuffer, *runtime.Pinner, bool) {
	if buf, ok := registeredPinnedNoCopyBytes(out); ok {
		return buf, nil, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(out)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, nil, false
	}
	return buf, pinner, true
}

func sdpaPagedValidate(qb []byte, keyPages, valuePages [][]byte, pageLens []int, nHeads, nKVHeads, headDim int) ([]int, int, error) {
	if nHeads <= 0 || nKVHeads <= 0 || headDim <= 0 {
		return nil, 0, core.NewError("native.SDPAPagedBF16: dimensions must be > 0")
	}
	if nHeads%nKVHeads != 0 {
		return nil, 0, core.NewError("native.SDPAPagedBF16: nHeads must be a multiple of nKVHeads")
	}
	if len(qb) != nHeads*headDim*bf16Size {
		return nil, 0, core.NewError("native.SDPAPagedBF16: query length mismatch")
	}
	if len(keyPages) == 0 || len(keyPages) != len(valuePages) {
		return nil, 0, core.NewError("native.SDPAPagedBF16: key/value pages must be non-empty and matched")
	}
	if pageLens != nil && len(pageLens) != len(keyPages) {
		return nil, 0, core.NewError("native.SDPAPagedBF16: page lens must match key/value pages")
	}
	pageStride := nKVHeads * headDim * bf16Size
	lens := make([]int, len(keyPages))
	total := 0
	for i := range keyPages {
		if len(keyPages[i]) == 0 || len(valuePages[i]) == 0 {
			return nil, 0, core.NewError("native.SDPAPagedBF16: page length must be > 0")
		}
		if len(keyPages[i]) != len(valuePages[i]) {
			return nil, 0, core.NewError("native.SDPAPagedBF16: key/value page byte lengths differ")
		}
		if len(keyPages[i])%pageStride != 0 {
			return nil, 0, core.NewError("native.SDPAPagedBF16: page byte length is not aligned to KV heads and headDim")
		}
		pageLen := len(keyPages[i]) / pageStride
		if pageLens != nil {
			pageLen = pageLens[i]
			physicalLen := len(keyPages[i]) / pageStride
			if pageLen <= 0 || pageLen > physicalLen {
				return nil, 0, core.NewError("native.SDPAPagedBF16: page lens must fit the physical page")
			}
		}
		lens[i] = pageLen
		total += pageLen
	}
	return lens, total, nil
}

// SDPAPagedBF16 computes single-token scaled-dot-product attention over paged BF16
// KV cache rows without concatenating the pages on the host. Page layout is
// head-major [nKVHeads, pageLen, headDim], matching pkg/metal's paged-cache ABI.
func SDPAPagedBF16(qb []byte, keyPages, valuePages [][]byte, nHeads, nKVHeads, headDim int, scale float32) ([]byte, error) {
	return SDPAPagedBF16Into(nil, qb, keyPages, valuePages, nHeads, nKVHeads, headDim, scale)
}

func SDPAPagedBF16Into(out []byte, qb []byte, keyPages, valuePages [][]byte, nHeads, nKVHeads, headDim int, scale float32) ([]byte, error) {
	return sdpaPagedBF16IntoPageLens(out, qb, keyPages, valuePages, nil, nHeads, nKVHeads, headDim, scale)
}

func sdpaPagedBF16IntoPageLens(out []byte, qb []byte, keyPages, valuePages [][]byte, pageLens []int, nHeads, nKVHeads, headDim int, scale float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	pageLens, _, err := sdpaPagedValidate(qb, keyPages, valuePages, pageLens, nHeads, nKVHeads, headDim)
	if err != nil {
		return nil, err
	}

	outLen := nHeads * headDim * bf16Size
	callerOut := cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}

	var encErr error
	withAutoreleasePool(func() {
		outBuf := scratchBF16(nHeads * headDim)
		if outBuf == nil || outBuf.GetID() == 0 {
			encErr = core.NewError("native.SDPAPagedBF16: failed to allocate scratch buffers")
			return
		}
		scratch, err := newSDPAPagedDecodeScratch(nHeads, headDim)
		if err != nil {
			encErr = err
			return
		}

		var outPinner *runtime.Pinner
		directOut := false
		if callerOut {
			if tmp, pinner, ok := sdpaPagedOutputBuffer(out); ok {
				outBuf = tmp
				outPinner = pinner
				directOut = true
			}
		}
		defer func() {
			if outPinner != nil {
				outPinner.Unpin()
			}
		}()

		pinners := make([]*runtime.Pinner, 0, 1+len(keyPages)*2)
		defer func() {
			for _, pinner := range pinners {
				if pinner != nil {
					pinner.Unpin()
				}
			}
		}()
		qBuf := sdpaPagedTransientBuffer(qb, &pinners)
		keyBufs := make([]metal.MTLBuffer, len(keyPages))
		valueBufs := make([]metal.MTLBuffer, len(valuePages))
		pageSpans := make([]int, len(keyPages))
		for i := range keyPages {
			keyBufs[i] = sdpaPagedTransientBuffer(keyPages[i], &pinners)
			valueBufs[i] = sdpaPagedTransientBuffer(valuePages[i], &pinners)
			pageSpans[i] = len(keyPages[i]) / (nKVHeads * headDim * bf16Size)
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		encErr = encSDPAPagedDecode(enc, qBuf, keyBufs, valueBufs, pageLens, pageSpans, outBuf, scratch, nHeads, nKVHeads, headDim, scale)
		endEncodingFast(enc)
		if encErr != nil {
			return
		}
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		if !directOut {
			copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outLen))
		}
		runtime.KeepAlive(qb)
		runtime.KeepAlive(keyPages)
		runtime.KeepAlive(valuePages)
		runtime.KeepAlive(out)
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}
