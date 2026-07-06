// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"github.com/tmc/apple/metal"
	"unsafe"
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
	page := cachePos / c.pageSize
	slot := cachePos % c.pageSize
	for len(c.kPages) <= page {
		k, v, kPtr, vPtr, allocErr := c.newPage()
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

func (c *devicePagedKVCache) newPage() (metal.MTLBuffer, metal.MTLBuffer, *byte, *byte, error) {
	bytes := uint(c.pageSize * c.kvDim * bf16Size)
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
	need := (c.maxSize + c.pageSize - 1) / c.pageSize
	for len(c.kPages) < need {
		k, v, kPtr, vPtr, err := c.newPage()
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
		start := pageIdx * c.pageSize
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
		page := pos / c.pageSize
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
		start := page * c.pageSize
		switch {
		case tokens <= start:
			c.pageLens[page] = 0
		case tokens-start >= c.pageSize:
			c.pageLens[page] = c.pageSize
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

func encAttnHalfKVPaged(
	enc metal.MTLComputeCommandEncoder,
	x metal.MTLBuffer, cache *devicePagedKVCache, offBuf, h metal.MTLBuffer, offOff uint,
	attnNormW, postAttnNorm, qNorm, kNorm bufView, valueNorm metal.MTLBuffer,
	sc attnScratch, proj projector,
	dModel, nHeads, nKVHeads, headDim, pos, slideW, rotaryDim int, base, scale, eps float32,
	ropeFreqs metal.MTLBuffer,
) error {
	if slideW > 0 {
		if cache == nil || !cache.ring {
			return core.NewError("native.encAttnHalfKVPaged: sliding window requires ring pages")
		}
	}
	kPage, vPage, rowOff, err := cache.slot(pos)
	if err != nil {
		return err
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
	if err := proj.project(enc, sc.normed, kPage, rowOff, projK); err != nil {
		return err
	}
	if gpuHasGeluKernel() && kNorm.buf != nil {
		if err := encQKNormRopeAt(enc, kPage, kNorm.buf, kPage, rowOff, kNorm.off, rowOff, offBuf, offOff, ropeFreqs, nKVHeads, headDim, rotaryDim, base, scale, eps); err != nil {
			return err
		}
	} else {
		if kNorm.buf != nil {
			if err := encRMSNormRowsBF16(enc, kPage, kNorm.buf, kPage, rowOff, kNorm.off, rowOff, nKVHeads, headDim, eps); err != nil {
				return err
			}
		}
		if err := encRopeDecodeAt(enc, kPage, kPage, rowOff, rowOff, offBuf, offOff, ropeFreqs, nKVHeads, headDim, rotaryDim, base, scale); err != nil {
			return err
		}
	}
	vIdx := projV
	if !proj.hasV() {
		vIdx = projK
	}
	if err := proj.project(enc, sc.normed, vPage, rowOff, vIdx); err != nil {
		return err
	}
	if valueNorm != nil {
		if err := encRMSNormRowsBF16(enc, vPage, valueNorm, vPage, rowOff, 0, rowOff, nKVHeads, headDim, eps); err != nil {
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
		return core.NewError("native.encAttnHalfSharedPaged: sliding window requires ring pages")
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
