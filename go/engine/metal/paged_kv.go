// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import core "dappco.re/go"

const defaultPagedKVPageSize = 256

// PagedKVState is a borrowed view of a native paged BF16 K/V cache. KeyPages
// and ValuePages include each page's physical backing; PageLens carries the
// visible token count for each page. The slices remain valid until the cache is
// mutated, reset, or closed.
type PagedKVState struct {
	KeyPages   [][]byte
	ValuePages [][]byte
	PageLens   []int
	Offset     int
	Length     int
}

// PagedKVCache stores BF16 K/V rows in pinned pages, avoiding the growing
// contiguous cache copies that pkg/metal's paged path was introduced to remove.
// Pages use head-major layout [nKVHeads, pageCapacity, headDim].
type PagedKVCache struct {
	kPages  []*pinnedNoCopyBytes
	vPages  []*pinnedNoCopyBytes
	lengths []int

	keyScratch   [][]byte
	valueScratch [][]byte
	lensScratch  []int

	nKVHeads int
	headDim  int
	rowBytes int
	maxSize  int
	pageSize int
	offset   int
	length   int
}

func NewPagedKVCache(nKVHeads, headDim, maxSize, pageSize int) (*PagedKVCache, error) {
	if nKVHeads <= 0 || headDim <= 0 {
		return nil, core.NewError("native.NewPagedKVCache: dimensions must be > 0")
	}
	if maxSize < 0 {
		return nil, core.NewError("native.NewPagedKVCache: maxSize must be >= 0")
	}
	if pageSize <= 0 {
		pageSize = defaultPagedKVPageSize
	}
	if maxSize > 0 && pageSize > maxSize {
		pageSize = maxSize
	}
	return &PagedKVCache{
		nKVHeads: nKVHeads,
		headDim:  headDim,
		rowBytes: nKVHeads * headDim * bf16Size,
		maxSize:  maxSize,
		pageSize: pageSize,
	}, nil
}

func (c *PagedKVCache) Close() {
	if c == nil {
		return
	}
	for _, page := range c.kPages {
		page.Close()
	}
	for _, page := range c.vPages {
		page.Close()
	}
	c.kPages = nil
	c.vPages = nil
	c.lengths = nil
	c.keyScratch = nil
	c.valueScratch = nil
	c.lensScratch = nil
	c.offset = 0
	c.length = 0
}

func (c *PagedKVCache) Reset() {
	c.Close()
}

func (c *PagedKVCache) Offset() int {
	if c == nil {
		return 0
	}
	return c.offset
}

func (c *PagedKVCache) Len() int {
	if c == nil {
		return 0
	}
	return c.length
}

func (c *PagedKVCache) PageSize() int {
	if c == nil {
		return 0
	}
	return c.pageSize
}

func (c *PagedKVCache) Update(k, v []byte, seqLen int) (PagedKVState, error) {
	if c == nil {
		return PagedKVState{}, core.NewError("native.PagedKVCache.Update: nil cache")
	}
	added, err := c.appendPages(k, v, seqLen)
	if err != nil {
		return PagedKVState{}, err
	}
	c.offset += added
	c.length += added
	if err := c.trimToMaxSize(); err != nil {
		return PagedKVState{}, err
	}
	return c.State(), nil
}

func (c *PagedKVCache) AttentionInto(out []byte, q []byte, nHeads int, scale float32) ([]byte, error) {
	if c == nil {
		return nil, core.NewError("native.PagedKVCache.AttentionInto: nil cache")
	}
	state := c.State()
	if state.Length == 0 {
		return nil, core.NewError("native.PagedKVCache.AttentionInto: empty cache")
	}
	return sdpaPagedBF16IntoPageLens(out, q, state.KeyPages, state.ValuePages, state.PageLens, nHeads, c.nKVHeads, c.headDim, scale)
}

func (c *PagedKVCache) Attention(q []byte, nHeads int, scale float32) ([]byte, error) {
	return c.AttentionInto(nil, q, nHeads, scale)
}

func (c *PagedKVCache) State() PagedKVState {
	if c == nil || len(c.kPages) == 0 || len(c.vPages) == 0 {
		return PagedKVState{}
	}
	n := len(c.kPages)
	if cap(c.keyScratch) < n {
		c.keyScratch = make([][]byte, n)
	}
	if cap(c.valueScratch) < n {
		c.valueScratch = make([][]byte, n)
	}
	if cap(c.lensScratch) < n {
		c.lensScratch = make([]int, n)
	}
	keys := c.keyScratch[:n]
	values := c.valueScratch[:n]
	lens := c.lensScratch[:n]
	clear(keys)
	clear(values)
	clear(lens)
	for i := range c.kPages {
		if c.kPages[i] != nil {
			keys[i] = c.kPages[i].bytes
		}
		if c.vPages[i] != nil {
			values[i] = c.vPages[i].bytes
		}
		lens[i] = c.lengths[i]
	}
	return PagedKVState{KeyPages: keys, ValuePages: values, PageLens: lens, Offset: c.offset, Length: c.length}
}

func (c *PagedKVCache) appendPages(k, v []byte, seqLen int) (int, error) {
	totalLen, err := c.validateKV(k, v, seqLen)
	if err != nil {
		return 0, err
	}
	if seqLen <= 0 || seqLen > totalLen {
		seqLen = totalLen
	}
	for start := 0; start < seqLen; {
		last := len(c.kPages) - 1
		if last >= 0 && c.lengths[last] < c.pageSize {
			room := c.pageSize - c.lengths[last]
			take := min(room, seqLen-start)
			c.appendToPage(last, k, v, totalLen, start, take)
			start += take
			continue
		}
		take := min(c.pageSize, seqLen-start)
		if err := c.appendNewPage(k, v, totalLen, start, take); err != nil {
			return start, err
		}
		start += take
	}
	return seqLen, nil
}

func (c *PagedKVCache) validateKV(k, v []byte, seqLen int) (int, error) {
	if len(k) == 0 || len(v) == 0 {
		return 0, core.NewError("native.PagedKVCache.Update: K/V must be non-empty")
	}
	if len(k) != len(v) {
		return 0, core.NewError("native.PagedKVCache.Update: K/V byte lengths differ")
	}
	if len(k)%c.rowBytes != 0 {
		return 0, core.NewError("native.PagedKVCache.Update: K/V length is not aligned to cache shape")
	}
	totalLen := len(k) / c.rowBytes
	if seqLen < 0 || seqLen > totalLen {
		return 0, core.NewError("native.PagedKVCache.Update: seqLen outside K/V rows")
	}
	return totalLen, nil
}

func (c *PagedKVCache) appendNewPage(k, v []byte, srcLen, start, take int) error {
	kPage, err := newPinnedNoCopyBytes(c.pageSize * c.rowBytes)
	if err != nil {
		return err
	}
	vPage, err := newPinnedNoCopyBytes(c.pageSize * c.rowBytes)
	if err != nil {
		kPage.Close()
		return err
	}
	c.kPages = append(c.kPages, kPage)
	c.vPages = append(c.vPages, vPage)
	c.lengths = append(c.lengths, 0)
	c.appendToPage(len(c.kPages)-1, k, v, srcLen, start, take)
	return nil
}

func (c *PagedKVCache) appendToPage(page int, k, v []byte, srcLen, start, take int) {
	dstStart := c.lengths[page]
	copyPagedKVTokens(c.kPages[page].bytes, c.pageSize, dstStart, k, srcLen, start, take, c.nKVHeads, c.headDim)
	copyPagedKVTokens(c.vPages[page].bytes, c.pageSize, dstStart, v, srcLen, start, take, c.nKVHeads, c.headDim)
	c.lengths[page] += take
}

func copyPagedKVTokens(dst []byte, dstSpan, dstStart int, src []byte, srcSpan, srcStart, take, nKVHeads, headDim int) {
	headBytes := headDim * bf16Size
	for h := range nKVHeads {
		dstOff := (h*dstSpan + dstStart) * headBytes
		srcOff := (h*srcSpan + srcStart) * headBytes
		copy(dst[dstOff:dstOff+take*headBytes], src[srcOff:srcOff+take*headBytes])
	}
}

func (c *PagedKVCache) trimToMaxSize() error {
	if c.maxSize <= 0 || c.length <= c.maxSize {
		return nil
	}
	drop := c.length - c.maxSize
	for drop > 0 && len(c.kPages) > 0 {
		firstLen := c.lengths[0]
		if drop >= firstLen {
			c.kPages[0].Close()
			c.vPages[0].Close()
			copy(c.kPages, c.kPages[1:])
			copy(c.vPages, c.vPages[1:])
			copy(c.lengths, c.lengths[1:])
			c.kPages = c.kPages[:len(c.kPages)-1]
			c.vPages = c.vPages[:len(c.vPages)-1]
			c.lengths = c.lengths[:len(c.lengths)-1]
			c.length -= firstLen
			drop -= firstLen
			continue
		}
		c.trimFirstPage(drop)
		c.length -= drop
		drop = 0
	}
	return nil
}

func (c *PagedKVCache) trimFirstPage(tokens int) {
	if tokens <= 0 || len(c.kPages) == 0 {
		return
	}
	remaining := c.lengths[0] - tokens
	if remaining <= 0 {
		return
	}
	movePagedKVTokensToFront(c.kPages[0].bytes, c.pageSize, tokens, remaining, c.nKVHeads, c.headDim)
	movePagedKVTokensToFront(c.vPages[0].bytes, c.pageSize, tokens, remaining, c.nKVHeads, c.headDim)
	c.lengths[0] = remaining
}

func movePagedKVTokensToFront(page []byte, span, start, count, nKVHeads, headDim int) {
	headBytes := headDim * bf16Size
	for h := range nKVHeads {
		base := h * span * headBytes
		src := base + start*headBytes
		dst := base
		copy(page[dst:dst+count*headBytes], page[src:src+count*headBytes])
	}
}
