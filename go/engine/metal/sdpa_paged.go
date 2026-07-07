// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"runtime"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

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
	if nHeads <= 0 || nKVHeads <= 0 || headDim <= 0 {
		return core.NewError("native.encSDPAPagedDecodeStrided: dimensions must be > 0")
	}
	if nHeads%nKVHeads != 0 {
		return core.NewError("native.encSDPAPagedDecodeStrided: nHeads must be a multiple of nKVHeads")
	}
	if q == nil || q.GetID() == 0 || out == nil || out.GetID() == 0 {
		return core.NewError("native.encSDPAPagedDecodeStrided: nil input/output buffer")
	}
	if len(keyPages) == 0 || len(keyPages) != len(valuePages) || len(keyPages) != len(pageLens) ||
		len(keyPages) != len(keyHeadStrides) || len(keyPages) != len(keySeqStrides) ||
		len(keyPages) != len(valueHeadStrides) || len(keyPages) != len(valueSeqStrides) {
		return core.NewError("native.encSDPAPagedDecodeStrided: page buffers and strides must be non-empty and matched")
	}
	for i := range keyPages {
		if keyPages[i] == nil || keyPages[i].GetID() == 0 || valuePages[i] == nil || valuePages[i].GetID() == 0 {
			return core.NewError("native.encSDPAPagedDecodeStrided: nil page buffer")
		}
		if pageLens[i] <= 0 || keyHeadStrides[i] <= 0 || keySeqStrides[i] <= 0 || valueHeadStrides[i] <= 0 || valueSeqStrides[i] <= 0 {
			return core.NewError("native.encSDPAPagedDecodeStrided: page lengths and strides must be > 0")
		}
	}
	// the lane slicing owns headDim/32 dims per lane — every shipped head dim (64,
	// 128, 256, 512) is a multiple of 32; reject anything else loudly rather than
	// silently dropping tail dims.
	if headDim%32 != 0 || headDim/32 > 16 {
		return core.NewError("native.encSDPAPagedDecodeStrided: headDim must be a multiple of 32, at most 512")
	}
	// each page fans out over ceil(len/splitRows) independent split cells — the grid grows
	// with context (#339: 16 fixed threadgroups measured 12.4 ms/token of attention at
	// position ~3500; split windows keep the whole GPU busy at any depth).
	cellCount := 0
	for i := range keyPages {
		cellCount += (pageLens[i] + sdpaPagedSplitRows - 1) / sdpaPagedSplitRows
	}
	if err := scratch.ensure(nHeads, headDim, cellCount); err != nil {
		return err
	}
	p1PSO, err := sdpaPagedP1Pipeline()
	if err != nil {
		return err
	}
	p2PSO, err := sdpaPagedP2Pipeline()
	if err != nil {
		return err
	}

	// pass 1: one dispatch per page, nHeads × splits threadgroups, each writing its OWN
	// (head, cell) partial; pass 2 merges per head across every cell. Hazard tracking on
	// the scratch orders pass 1 → pass 2 for free. (The previous kernel carried the online
	// softmax ACROSS pages at one scalar thread per head, looping each page twice — a
	// serialised chain whose length grew with context: the #252 decode collapse.)
	cellBase := 0
	for i := range keyPages {
		splits := (pageLens[i] + sdpaPagedSplitRows - 1) / sdpaPagedSplitRows
		params := sdpaPagedP1Params{
			NHeads:      uint32(nHeads),
			NKVHeads:    uint32(nKVHeads),
			HeadDim:     uint32(headDim),
			PageLen:     uint32(pageLens[i]),
			KHeadStride: uint32(keyHeadStrides[i]),
			KSeqStride:  uint32(keySeqStrides[i]),
			VHeadStride: uint32(valueHeadStrides[i]),
			VSeqStride:  uint32(valueSeqStrides[i]),
			SplitRows:   uint32(sdpaPagedSplitRows),
			Splits:      uint32(splits),
			CellBase:    uint32(cellBase),
			CellCount:   uint32(cellCount),
			Scale:       scale,
		}
		cellBase += splits
		setPSO(enc, p1PSO)
		setBuf(enc, q, 0, 0)
		setBuf(enc, keyPages[i], 0, 1)
		setBuf(enc, valuePages[i], 0, 2)
		setBuf(enc, scratch.maxs, 0, 3)
		setBuf(enc, scratch.sums, 0, 4)
		setBuf(enc, scratch.acc, 0, 5)
		setBytes(enc, unsafe.Pointer(&params), uint(unsafe.Sizeof(params)), 6)
		dispatchThreadgroups(enc,
			metal.MTLSize{Width: uint(nHeads * splits), Height: 1, Depth: 1},
			metal.MTLSize{Width: 256, Height: 1, Depth: 1}, // 8 simdgroups split the window rows
		)
	}
	p2 := sdpaPagedP2Params{HeadDim: uint32(headDim), CellCount: uint32(cellCount)}
	setPSO(enc, p2PSO)
	setBuf(enc, scratch.maxs, 0, 0)
	setBuf(enc, scratch.sums, 0, 1)
	setBuf(enc, scratch.acc, 0, 2)
	setBuf(enc, out, 0, 3)
	setBytes(enc, unsafe.Pointer(&p2), uint(unsafe.Sizeof(p2)), 4)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
	return nil
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
