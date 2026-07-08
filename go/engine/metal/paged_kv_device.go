// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math/bits"
	"sync/atomic"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

type devicePagedKVCache struct {
	kPages, vPages []metal.MTLBuffer
	kPagePtrs      []*byte
	vPagePtrs      []*byte
	pageLens       []int

	keyScratch, valueScratch           []metal.MTLBuffer
	lensScratch                        []int
	kHeadStrides, kSeqStrides          []int
	vHeadStrides, vSeqStrides          []int
	snapshotK, snapshotV               metal.MTLBuffer
	snapshotKPtr, snapshotVPtr         *byte
	snapshotBytes                      int
	nKVHeads, headDim, kvDim, pageSize int
	maxSize, length, offset            int
	ring                               bool
	linearSynced                       int
	sdpaScratch                        []*sdpaPagedDecodeScratch
	sdpaScratchCursor                  int
}

func newDevicePagedKVCache(nKVHeads, headDim, maxSize, pageSize int) (*devicePagedKVCache, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if nKVHeads <= 0 || headDim <= 0 {
		return nil, core.NewError("native.newDevicePagedKVCache: dimensions must be > 0")
	}
	if maxSize < 0 {
		return nil, core.NewError("native.newDevicePagedKVCache: maxSize must be >= 0")
	}
	if pageSize <= 0 {
		pageSize = defaultPagedKVPageSize
	}
	if maxSize > 0 && pageSize > maxSize {
		pageSize = maxSize
	}
	return &devicePagedKVCache{
		nKVHeads: nKVHeads,
		headDim:  headDim,
		kvDim:    nKVHeads * headDim,
		pageSize: pageSize,
		maxSize:  maxSize,
	}, nil
}

// maxPagedKVPageRows caps the geometric page growth. Pages double from the
// base size (2048 -> 4096 -> 8192 -> 16384, then 16384 flat): the paged SDPA
// runs ONE pass-1 dispatch per visited page, and the #356 anatomy bench
// measured the 16K scan 26% faster in one page than in eight — dispatch
// boundaries, not hazards, were the cost. Doubling keeps the allocation
// granularity of small pages for short sessions (a request that never leaves
// page 0 allocates 2048 rows) while a deep scan converges to a handful of
// dispatches; the cap bounds the worst-case over-allocation to one 16K page.
const maxPagedKVPageRows = 16384

// pageRowsFor is page i's row capacity under the geometric schedule, based at
// the cache's (possibly maxSize-clamped) pageSize.
func (c *devicePagedKVCache) pageRowsFor(page int) int {
	if c.pageSize >= maxPagedKVPageRows {
		return c.pageSize
	}
	doublings := 0
	for sz := c.pageSize; sz < maxPagedKVPageRows; sz <<= 1 {
		doublings++
	}
	if page >= doublings {
		return maxPagedKVPageRows
	}
	return c.pageSize << page
}

// pageStartFor is the first cache position stored in page i (the prefix sum of
// pageRowsFor, in closed form: base·(2^i − 1) through the doubling run, then
// flat cap-sized steps).
func (c *devicePagedKVCache) pageStartFor(page int) int {
	if c.pageSize >= maxPagedKVPageRows {
		return page * c.pageSize
	}
	doublings := 0
	for sz := c.pageSize; sz < maxPagedKVPageRows; sz <<= 1 {
		doublings++
	}
	if page <= doublings {
		return c.pageSize * ((1 << page) - 1)
	}
	rampEnd := c.pageSize * ((1 << doublings) - 1)
	return rampEnd + (page-doublings)*maxPagedKVPageRows
}

// pageForPos maps a cache position to its page index under the schedule —
// the doubling run resolves with one bit-length, the flat tail by division.
func (c *devicePagedKVCache) pageForPos(cachePos int) int {
	if c.pageSize >= maxPagedKVPageRows {
		return cachePos / c.pageSize
	}
	doublings := 0
	for sz := c.pageSize; sz < maxPagedKVPageRows; sz <<= 1 {
		doublings++
	}
	rampEnd := c.pageSize * ((1 << doublings) - 1)
	if cachePos < rampEnd {
		return bits.Len(uint(cachePos/c.pageSize+1)) - 1
	}
	return doublings + (cachePos-rampEnd)/maxPagedKVPageRows
}

func (c *devicePagedKVCache) Close() {
	if c == nil {
		return
	}
	c.kPages = nil
	c.vPages = nil
	c.kPagePtrs = nil
	c.vPagePtrs = nil
	c.pageLens = nil
	c.keyScratch = nil
	c.valueScratch = nil
	c.lensScratch = nil
	c.kHeadStrides = nil
	c.kSeqStrides = nil
	c.vHeadStrides = nil
	c.vSeqStrides = nil
	c.snapshotK = nil
	c.snapshotV = nil
	c.snapshotKPtr = nil
	c.snapshotVPtr = nil
	c.snapshotBytes = 0
	c.sdpaScratch = nil
	c.sdpaScratchCursor = 0
	c.length = 0
	c.offset = 0
	c.linearSynced = 0
}

func (c *devicePagedKVCache) slot(pos int) (kPage, vPage metal.MTLBuffer, rowOff uint, err error) {
	if c == nil {
		return nil, nil, 0, core.NewError("native.devicePagedKVCache.slot: nil cache")
	}
	if pos < 0 {
		return nil, nil, 0, core.NewError("native.devicePagedKVCache.slot: negative position")
	}
	if c.maxSize > 0 && !c.ring && pos >= c.maxSize {
		return nil, nil, 0, core.NewError("native.devicePagedKVCache.slot: position exceeds maxSize")
	}
	cachePos := pos
	if c.ring && c.maxSize > 0 {
		cachePos = pos % c.maxSize
	}
	page := c.pageForPos(cachePos)
	slot := cachePos - c.pageStartFor(page)
	for len(c.kPages) <= page {
		k, v, kPtr, vPtr, allocErr := c.newPage(c.pageRowsFor(len(c.kPages)))
		if allocErr != nil {
			return nil, nil, 0, allocErr
		}
		c.kPages = append(c.kPages, k)
		c.vPages = append(c.vPages, v)
		c.kPagePtrs = append(c.kPagePtrs, kPtr)
		c.vPagePtrs = append(c.vPagePtrs, vPtr)
		c.pageLens = append(c.pageLens, 0)
	}
	if n := slot + 1; n > c.pageLens[page] {
		c.pageLens[page] = n
	}
	if n := pos + 1; c.ring && c.maxSize > 0 && n > c.maxSize {
		c.length = c.maxSize
	} else if n > c.length {
		c.length = n
	}
	if n := pos + 1; n > c.offset {
		c.offset = n
	}
	if cachePos < c.linearSynced {
		c.linearSynced = cachePos
	}
	return c.kPages[page], c.vPages[page], uint(slot * c.kvDim * bf16Size), nil
}

func (c *devicePagedKVCache) newPage(rows int) (metal.MTLBuffer, metal.MTLBuffer, *byte, *byte, error) {
	bytes := uint(rows * c.kvDim * bf16Size)
	k := device.NewBufferWithLengthOptions(bytes, metal.MTLResourceStorageModeShared)
	v := device.NewBufferWithLengthOptions(bytes, metal.MTLResourceStorageModeShared)
	if k == nil || v == nil || k.GetID() == 0 || v.GetID() == 0 {
		return nil, nil, nil, nil, core.NewError("native.devicePagedKVCache.newPage: failed to allocate page buffers")
	}
	return k, v, (*byte)(k.Contents()), (*byte)(v.Contents()), nil
}

func (c *devicePagedKVCache) preallocPages() error {
	if c == nil {
		return core.NewError("native.devicePagedKVCache.preallocPages: nil cache")
	}
	if c.maxSize <= 0 {
		return nil
	}
	need := c.pageForPos(c.maxSize-1) + 1
	for len(c.kPages) < need {
		k, v, kPtr, vPtr, err := c.newPage(c.pageRowsFor(len(c.kPages)))
		if err != nil {
			return err
		}
		c.kPages = append(c.kPages, k)
		c.vPages = append(c.vPages, v)
		c.kPagePtrs = append(c.kPagePtrs, kPtr)
		c.vPagePtrs = append(c.vPagePtrs, vPtr)
		c.pageLens = append(c.pageLens, 0)
	}
	return nil
}

func (c *devicePagedKVCache) linearSnapshot(rows int) (kBuf, vBuf metal.MTLBuffer, kPtr, vPtr *byte, err error) {
	if c == nil {
		return nil, nil, nil, nil, core.NewError("native.devicePagedKVCache.linearSnapshot: nil cache")
	}
	if rows < c.length {
		return nil, nil, nil, nil, core.NewError("native.devicePagedKVCache.linearSnapshot: rows shorter than cache")
	}
	if rows < 0 {
		return nil, nil, nil, nil, core.NewError("native.devicePagedKVCache.linearSnapshot: rows must be >= 0")
	}
	rowBytes := c.kvDim * bf16Size
	nBytes := rows * rowBytes
	if nBytes == 0 {
		return nil, nil, nil, nil, core.NewError("native.devicePagedKVCache.linearSnapshot: empty snapshot")
	}
	if c.snapshotK == nil || c.snapshotBytes != nBytes {
		c.snapshotK = device.NewBufferWithLengthOptions(uint(nBytes), metal.MTLResourceStorageModeShared)
	}
	if c.snapshotV == nil || c.snapshotBytes != nBytes {
		c.snapshotV = device.NewBufferWithLengthOptions(uint(nBytes), metal.MTLResourceStorageModeShared)
	}
	if c.snapshotK == nil || c.snapshotK.GetID() == 0 || c.snapshotV == nil || c.snapshotV.GetID() == 0 {
		return nil, nil, nil, nil, core.NewError("native.devicePagedKVCache.linearSnapshot: failed to allocate snapshot buffers")
	}
	if c.snapshotBytes != nBytes || c.snapshotKPtr == nil || c.snapshotVPtr == nil {
		c.snapshotKPtr = (*byte)(c.snapshotK.Contents())
		c.snapshotVPtr = (*byte)(c.snapshotV.Contents())
		c.snapshotBytes = nBytes
	}
	kPtr = c.snapshotKPtr
	vPtr = c.snapshotVPtr
	kBytes := unsafe.Slice(kPtr, nBytes)
	vBytes := unsafe.Slice(vPtr, nBytes)
	clear(kBytes)
	clear(vBytes)
	for pageIdx, pageLen := range c.pageLens {
		if pageLen <= 0 {
			continue
		}
		start := c.pageStartFor(pageIdx)
		if start >= rows {
			break
		}
		if start+pageLen > rows {
			pageLen = rows - start
		}
		copyBytes := pageLen * rowBytes
		dstOff := start * rowBytes
		srcK := unsafe.Slice(c.kPagePtrs[pageIdx], copyBytes)
		srcV := unsafe.Slice(c.vPagePtrs[pageIdx], copyBytes)
		copy(kBytes[dstOff:dstOff+copyBytes], srcK)
		copy(vBytes[dstOff:dstOff+copyBytes], srcV)
	}
	return c.snapshotK, c.snapshotV, kPtr, vPtr, nil
}

func (c *devicePagedKVCache) loadLinearSnapshot(kRows, vRows []byte, tokens int) error {
	if c == nil {
		return core.NewError("native.devicePagedKVCache.loadLinearSnapshot: nil cache")
	}
	if tokens < 0 {
		return core.NewError("native.devicePagedKVCache.loadLinearSnapshot: tokens must be >= 0")
	}
	if c.maxSize > 0 && tokens > c.maxSize {
		return core.NewError("native.devicePagedKVCache.loadLinearSnapshot: tokens exceed maxSize")
	}
	rowBytes := c.kvDim * bf16Size
	need := tokens * rowBytes
	if len(kRows) < need || len(vRows) < need {
		return core.NewError("native.devicePagedKVCache.loadLinearSnapshot: snapshot bytes too short")
	}
	for i := range c.pageLens {
		c.pageLens[i] = 0
	}
	c.length = 0
	c.offset = 0
	for pos := range tokens {
		_, _, rowOff, err := c.slot(pos)
		if err != nil {
			return err
		}
		srcOff := pos * rowBytes
		page := c.pageForPos(pos)
		copy(unsafe.Slice((*byte)(unsafe.Add(unsafe.Pointer(c.kPagePtrs[page]), uintptr(rowOff))), rowBytes), kRows[srcOff:srcOff+rowBytes])
		copy(unsafe.Slice((*byte)(unsafe.Add(unsafe.Pointer(c.vPagePtrs[page]), uintptr(rowOff))), rowBytes), vRows[srcOff:srcOff+rowBytes])
	}
	c.linearSynced = tokens
	return nil
}

func (c *devicePagedKVCache) truncate(tokens int) error {
	if c == nil {
		return core.NewError("native.devicePagedKVCache.truncate: nil cache")
	}
	if tokens < 0 {
		return core.NewError("native.devicePagedKVCache.truncate: tokens must be >= 0")
	}
	if c.ring && c.maxSize > 0 && tokens > c.maxSize {
		c.length = c.maxSize
		c.offset = tokens
		if c.linearSynced > c.length {
			c.linearSynced = c.length
		}
		return nil
	}
	if c.maxSize > 0 && tokens > c.maxSize {
		return core.NewError("native.devicePagedKVCache.truncate: tokens exceed maxSize")
	}
	if tokens > c.length {
		return core.NewError("native.devicePagedKVCache.truncate: cannot extend cache")
	}
	for page := range c.pageLens {
		start := c.pageStartFor(page)
		rows := c.pageRowsFor(page)
		switch {
		case tokens <= start:
			c.pageLens[page] = 0
		case tokens-start >= rows:
			c.pageLens[page] = rows
		default:
			c.pageLens[page] = tokens - start
		}
	}
	c.length = tokens
	c.offset = tokens
	if c.linearSynced > tokens {
		c.linearSynced = tokens
	}
	return nil
}

func (c *devicePagedKVCache) state() (keys, values []metal.MTLBuffer, lens, kHead, kSeq, vHead, vSeq []int, err error) {
	if c == nil || len(c.kPages) == 0 || len(c.kPages) != len(c.vPages) || len(c.kPages) != len(c.pageLens) {
		return nil, nil, nil, nil, nil, nil, nil, core.NewError("native.devicePagedKVCache.state: invalid page state")
	}
	n := len(c.kPages)
	if cap(c.keyScratch) < n {
		c.keyScratch = make([]metal.MTLBuffer, n)
	}
	if cap(c.valueScratch) < n {
		c.valueScratch = make([]metal.MTLBuffer, n)
	}
	if cap(c.lensScratch) < n {
		c.lensScratch = make([]int, n)
	}
	if cap(c.kHeadStrides) < n {
		c.kHeadStrides = make([]int, n)
		c.kSeqStrides = make([]int, n)
		c.vHeadStrides = make([]int, n)
		c.vSeqStrides = make([]int, n)
	}
	keys = c.keyScratch[:n]
	values = c.valueScratch[:n]
	lens = c.lensScratch[:n]
	kHead = c.kHeadStrides[:n]
	kSeq = c.kSeqStrides[:n]
	vHead = c.vHeadStrides[:n]
	vSeq = c.vSeqStrides[:n]
	for i := range n {
		keys[i] = c.kPages[i]
		values[i] = c.vPages[i]
		lens[i] = c.pageLens[i]
		kHead[i] = c.headDim
		kSeq[i] = c.kvDim
		vHead[i] = c.headDim
		vSeq[i] = c.kvDim
	}
	return keys, values, lens, kHead, kSeq, vHead, vSeq, nil
}

func (c *devicePagedKVCache) attentionScratch(nHeads int) (*sdpaPagedDecodeScratch, error) {
	if c == nil {
		return nil, core.NewError("native.devicePagedKVCache.attentionScratch: nil cache")
	}
	idx := c.sdpaScratchCursor
	c.sdpaScratchCursor++
	if idx < len(c.sdpaScratch) {
		scratch := c.sdpaScratch[idx]
		if scratch != nil && scratch.nHeads == nHeads && scratch.headDim == c.headDim {
			return scratch, nil
		}
	}
	scratch, err := newSDPAPagedDecodeScratch(nHeads, c.headDim)
	if err != nil {
		return nil, err
	}
	if idx < len(c.sdpaScratch) {
		c.sdpaScratch[idx] = scratch
	} else {
		c.sdpaScratch = append(c.sdpaScratch, scratch)
	}
	return scratch, nil
}

func (c *devicePagedKVCache) resetAttentionScratchCursor() {
	if c != nil {
		c.sdpaScratchCursor = 0
	}
}

// attnConcurrentPasses counts concurrent-pass attention encodes — the engagement receipt
// (a silent gate regression reads as zero, not as a perf blur).
var attnConcurrentPasses atomic.Int64

// concEncoderCarries counts passes that CONTINUED on an incoming concurrent
// encoder instead of closing it and reopening (#341 phase 1.5): the hop-tax
// bench measured every encoder end+open seam at ~4.7µs and every serial-tracked
// hop at 7.0µs vs the barrier idiom's 4.13µs, so carrying one concurrent
// encoder across the attn pass, the MoE pass and the per-layer scalar removes
// ~4 seams per layer per token. Engagement counter for the A/B tests.
var concEncoderCarries atomic.Int64

// encConc marks enc as an OPEN CONCURRENT encoder carried from a previous pass
// (#341 phase 1.5): the pass then joins it behind one buffer barrier instead of
// paying an encoder seam, and returns encConc=true itself when it leaves its
// own concurrent encoder open for the next pass to carry. A false return means
// enc is a plain serial encoder (hazard-tracked), exactly the pre-carry
// contract.
func encAttnHalfKVPaged(
	enc metal.MTLComputeCommandEncoderObject,
	cb metal.MTLCommandBufferObject,
	prof *gpuCounterProfiler,
	encConc bool,
	x metal.MTLBuffer, cache *devicePagedKVCache, offBuf, h metal.MTLBuffer, offOff uint,
	attnNormW, postAttnNorm, qNorm, kNorm bufView, valueNorm metal.MTLBuffer,
	sc attnScratch, proj projector,
	dModel, nHeads, nKVHeads, headDim, pos, slideW, rotaryDim int, base, scale, eps float32,
	ropeFreqs metal.MTLBuffer,
) (metal.MTLComputeCommandEncoderObject, bool, error) {
	if slideW > 0 {
		if cache == nil {
			return enc, encConc, core.NewError("native.encAttnHalfKVPaged: sliding window requires ring pages")
		}
		if !cache.ring {
			// The builder skips ring pages when the window covers the whole cache
			// (max pos = maxSize-1 < slideW ⇒ the mask can never clip): the window is
			// inert here, so attend fully. A window that CAN clip still requires ring.
			if cache.maxSize > slideW {
				return enc, encConc, core.NewError("native.encAttnHalfKVPaged: sliding window requires ring pages")
			}
			slideW = 0
		}
	}
	kPage, vPage, rowOff, err := cache.slot(pos)
	if err != nil {
		return enc, encConc, err
	}
	// one interface value per encoder: the Object struct is not pointer-shaped, so every
	// implicit conversion at an interface-taking call would allocate — 13 boxes per layer
	// showed up straight in the sampled-wake allocation budgets.
	encI := metal.MTLComputeCommandEncoder(enc)
	fusedQKRope := gpuHasGeluKernel() && qNorm.buf != nil
	fusedKRope := gpuHasGeluKernel() && kNorm.buf != nil

	// The paged SDPA plan up front: the concurrent pass needs its pass-1/pass-2 seam, and a
	// plan failure must decline BEFORE any dispatch is encoded.
	keyPages, valuePages, pageLens, kHead, kSeq, vHead, vSeq, err := cache.state()
	if err != nil {
		return enc, encConc, err
	}
	pagedScratch, err := cache.attentionScratch(nHeads)
	if err != nil {
		return enc, encConc, err
	}
	sdpaPlan, err := buildSDPAPagedDecodePlan(sc.q, keyPages, valuePages, pageLens, kHead, kSeq, vHead, vSeq, sc.attn, pagedScratch, nHeads, nKVHeads, headDim, scale)
	if err != nil {
		return enc, encConc, err
	}

	// ---- concurrent pass: the q/k/v projections all read the same normed row, their
	// rope/norm stages are pairwise independent, and the SDPA's per-page pass-1 dispatches
	// are independent by construction — a serial encoder never overlapped ANY of them.
	// Explicit buffer barriers mark the true edges; values are unchanged (same kernels,
	// same rounding — only the schedule differs). The fused-rope shape only (the plain
	// rms+rope fallback keeps the serial path), and never under the profiler's seams.
	if prof == nil && !attnConcurrentDisabled && fusedQKRope && (kNorm.buf == nil || fusedKRope) {
		attnConcurrentPasses.Add(1)
		if encConc && !encCarryDisabled {
			// Carry the previous pass's open concurrent encoder: one buffer barrier
			// orders its writes ahead of this pass's reads, no encoder seam paid.
			concEncoderCarries.Add(1)
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			encI = metal.MTLComputeCommandEncoder(enc)
		} else {
			endEncodingFast(enc)
			enc = concurrentComputeEncoderFast(cb)
			encI = metal.MTLComputeCommandEncoder(enc)
		}
		// stage 1: the shared input norm
		if err := encRMSNormBF16(encI, x, attnNormW.buf, sc.normed, attnNormW.off, dModel, eps); err != nil {
			endEncodingFast(enc)
			return computeCommandEncoderFast(cb), false, err
		}
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// stage 2: q ∥ k ∥ v projections
		if err := proj.project(encI, sc.normed, sc.q, 0, projQ); err != nil {
			endEncodingFast(enc)
			return computeCommandEncoderFast(cb), false, err
		}
		if err := proj.project(encI, sc.normed, kPage, rowOff, projK); err != nil {
			endEncodingFast(enc)
			return computeCommandEncoderFast(cb), false, err
		}
		vIdx := projV
		if !proj.hasV() {
			vIdx = projK
		}
		if err := proj.project(encI, sc.normed, vPage, rowOff, vIdx); err != nil {
			endEncodingFast(enc)
			return computeCommandEncoderFast(cb), false, err
		}
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// stage 3: q rope ∥ k rope ∥ v norm
		if err := encQKNormRopeAt(encI, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, scale, eps); err != nil {
			endEncodingFast(enc)
			return computeCommandEncoderFast(cb), false, err
		}
		if kNorm.buf != nil {
			if err := encQKNormRopeAt(encI, kPage, kNorm.buf, kPage, rowOff, kNorm.off, rowOff, offBuf, offOff, ropeFreqs, nKVHeads, headDim, rotaryDim, base, scale, eps); err != nil {
				endEncodingFast(enc)
				return computeCommandEncoderFast(cb), false, err
			}
		}
		if valueNorm != nil {
			if err := encRMSNormRowsBF16(encI, vPage, valueNorm, vPage, rowOff, 0, rowOff, nKVHeads, headDim, eps); err != nil {
				endEncodingFast(enc)
				return computeCommandEncoderFast(cb), false, err
			}
		}
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// stage 4: SDPA pass 1 — the per-page/window partials, genuinely overlapped
		sdpaPlan.emitP1s(encI)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// stage 5: SDPA pass 2 — the cell merge
		sdpaPlan.emitP2(encI)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// stage 6: output projection
		if err := proj.project(encI, sc.attn, sc.attnOut, 0, projO); err != nil {
			endEncodingFast(enc)
			return computeCommandEncoderFast(cb), false, err
		}
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// stage 7: residual (+ post-attention norm)
		if err := encResidualMaybeNorm(encI, x, sc.attnOut, sc.normed, h, postAttnNorm, dModel, eps); err != nil {
			endEncodingFast(enc)
			return computeCommandEncoderFast(cb), false, err
		}
		if encCarryDisabled {
			endEncodingFast(enc)
			return computeCommandEncoderFast(cb), false, nil
		}
		// Leave the concurrent encoder OPEN — the caller carries it into the next
		// pass (a barrier at that pass's entry orders this pass's writes).
		return enc, true, nil
	}

	// ---- serial path (hazard tracking orders every edge) ----
	// A carried concurrent encoder has no hazard tracking — the serial fallback
	// must close it and reopen a tracked serial encoder before dispatching.
	if encConc {
		endEncodingFast(enc)
		enc = computeCommandEncoderFast(cb)
		encI = metal.MTLComputeCommandEncoder(enc)
		encConc = false
	}
	// Under the profiler the attention half splits at its family seams — proj
	// (norm + q/k/v projections + ropes) | sdpa (both passes) | tail (o-proj +
	// residual) — so the ranked table can tell weight-read time from attention
	// math from launch overhead. prof==nil (production) encodes exactly as before.
	if prof != nil {
		endEncodingFast(enc)
		enc = prof.encoderFor(cb, "attn.proj")
		encI = metal.MTLComputeCommandEncoder(enc)
	}
	if err := encRMSNormBF16(encI, x, attnNormW.buf, sc.normed, attnNormW.off, dModel, eps); err != nil {
		return enc, false, err
	}
	if err := proj.project(encI, sc.normed, sc.q, 0, projQ); err != nil {
		return enc, false, err
	}
	if fusedQKRope {
		if err := encQKNormRopeAt(encI, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, scale, eps); err != nil {
			return enc, false, err
		}
	} else {
		if qNorm.buf != nil {
			if err := encRMSNormRowsBF16(encI, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, nHeads, headDim, eps); err != nil {
				return enc, false, err
			}
		}
		if err := encRopeDecodeAt(encI, sc.q, sc.q, 0, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, scale); err != nil {
			return enc, false, err
		}
	}
	if err := proj.project(encI, sc.normed, kPage, rowOff, projK); err != nil {
		return enc, false, err
	}
	if fusedKRope {
		if err := encQKNormRopeAt(encI, kPage, kNorm.buf, kPage, rowOff, kNorm.off, rowOff, offBuf, offOff, ropeFreqs, nKVHeads, headDim, rotaryDim, base, scale, eps); err != nil {
			return enc, false, err
		}
	} else {
		if kNorm.buf != nil {
			if err := encRMSNormRowsBF16(encI, kPage, kNorm.buf, kPage, rowOff, kNorm.off, rowOff, nKVHeads, headDim, eps); err != nil {
				return enc, false, err
			}
		}
		if err := encRopeDecodeAt(encI, kPage, kPage, rowOff, rowOff, offBuf, offOff, ropeFreqs, nKVHeads, headDim, rotaryDim, base, scale); err != nil {
			return enc, false, err
		}
	}
	vIdx := projV
	if !proj.hasV() {
		vIdx = projK
	}
	if err := proj.project(encI, sc.normed, vPage, rowOff, vIdx); err != nil {
		return enc, false, err
	}
	if valueNorm != nil {
		if err := encRMSNormRowsBF16(encI, vPage, valueNorm, vPage, rowOff, 0, rowOff, nKVHeads, headDim, eps); err != nil {
			return enc, false, err
		}
	}
	if prof != nil {
		endEncodingFast(enc)
		enc = prof.encoderFor(cb, "attn.sdpa")
		encI = metal.MTLComputeCommandEncoder(enc)
	}
	sdpaPlan.emitP1s(encI)
	sdpaPlan.emitP2(encI)
	if prof != nil {
		endEncodingFast(enc)
		enc = prof.encoderFor(cb, "attn.tail")
		encI = metal.MTLComputeCommandEncoder(enc)
	}
	if err := proj.project(encI, sc.attn, sc.attnOut, 0, projO); err != nil {
		return enc, false, err
	}
	return enc, false, encResidualMaybeNorm(encI, x, sc.attnOut, sc.normed, h, postAttnNorm, dModel, eps)
}

func encAttnHalfSharedPaged(
	enc metal.MTLComputeCommandEncoder,
	x metal.MTLBuffer, cache *devicePagedKVCache, offBuf, h metal.MTLBuffer, offOff uint,
	attnNormW, postAttnNorm, qNorm bufView,
	sc attnScratch, proj projector,
	dModel, nHeads, nKVHeads, headDim, pos, slideW, rotaryDim int, base, scale, eps float32,
	ropeFreqs metal.MTLBuffer,
) error {
	if cache == nil {
		return core.NewError("native.encAttnHalfSharedPaged: nil cache")
	}
	if pos < 0 {
		return core.NewError("native.encAttnHalfSharedPaged: negative position")
	}
	if cache.length < pos+1 {
		need := pos + 1
		if cache.ring && cache.maxSize > 0 && need > cache.maxSize {
			need = cache.maxSize
		}
		if cache.length < need {
			return core.NewError("native.encAttnHalfSharedPaged: cache shorter than position")
		}
	}
	if slideW > 0 && !cache.ring {
		// Same inert-window carve-out as encAttnHalfKVPaged: a window covering the whole
		// cache can never clip, so the builder deliberately built linear pages.
		if cache.maxSize > slideW {
			return core.NewError("native.encAttnHalfSharedPaged: sliding window requires ring pages")
		}
		slideW = 0
	}
	if err := encRMSNormBF16(enc, x, attnNormW.buf, sc.normed, attnNormW.off, dModel, eps); err != nil {
		return err
	}
	if err := proj.project(enc, sc.normed, sc.q, 0, projQ); err != nil {
		return err
	}
	if gpuHasGeluKernel() && qNorm.buf != nil {
		if err := encQKNormRopeAt(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, scale, eps); err != nil {
			return err
		}
	} else {
		if qNorm.buf != nil {
			if err := encRMSNormRowsBF16(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, nHeads, headDim, eps); err != nil {
				return err
			}
		}
		if err := encRopeDecodeAt(enc, sc.q, sc.q, 0, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, scale); err != nil {
			return err
		}
	}
	keyPages, valuePages, pageLens, kHead, kSeq, vHead, vSeq, err := cache.state()
	if err != nil {
		return err
	}
	pagedScratch, err := cache.attentionScratch(nHeads)
	if err != nil {
		return err
	}
	if err := encSDPAPagedDecodeStrided(enc, sc.q, keyPages, valuePages, pageLens, kHead, kSeq, vHead, vSeq, sc.attn, pagedScratch, nHeads, nKVHeads, headDim, scale); err != nil {
		return err
	}
	if err := proj.project(enc, sc.attn, sc.attnOut, 0, projO); err != nil {
		return err
	}
	return encResidualMaybeNorm(enc, x, sc.attnOut, sc.normed, h, postAttnNorm, dModel, eps)
}
