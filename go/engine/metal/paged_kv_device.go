// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"math/bits"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// float helpers for the host q8 paths (kept local: the package's other float
// plumbing is GPU-side).
func mathFloat32bits(f float32) uint32     { return math.Float32bits(f) }
func mathFloat32frombits(b uint32) float32 { return math.Float32frombits(b) }
func absF32(f float32) float32 {
	if f < 0 {
		return -f
	}
	return f
}
func roundF32(f float32) float32 { return float32(math.Round(float64(f))) }

type devicePagedKVCache struct {
	kPages, vPages []metal.MTLBuffer
	kPagePtrs      []*byte
	vPagePtrs      []*byte
	pageLens       []int

	// q8 mode (#357): pages hold int8 rows quantised symmetrically per
	// kvQ8GroupSize elements, with f32 group scales in PARALLEL pages — the
	// same [row][kvHead][...] order, element strides unchanged, so the SDPA
	// kernels' addressing carries over with a byte (not bf16) element and one
	// scale load per lane slice. The linear twin stays bf16: the host
	// snapshot paths quantise on load and dequantise on read, so prefill, the
	// batched verify, state save/restore, and the drafter export all flow
	// through unchanged.
	quantQ8                  bool
	kScalePages, vScalePages []metal.MTLBuffer
	kScalePtrs, vScalePtrs   []*byte

	keyScratch, valueScratch           []metal.MTLBuffer
	lensScratch                        []int
	kHeadStrides, kSeqStrides          []int
	vHeadStrides, vSeqStrides          []int
	kScaleScratch, vScaleScratch       []metal.MTLBuffer
	snapshotK, snapshotV               metal.MTLBuffer
	snapshotKPtr, snapshotVPtr         *byte
	snapshotBytes                      int
	nKVHeads, headDim, kvDim, pageSize int
	maxSize, length, offset            int
	ring                               bool
	linearSynced                       int
	// linearSyncedAbs is the RING caches' mirror watermark in ABSOLUTE token
	// positions: rows [0, linearSyncedAbs) hold identical content in this cache
	// and the linear lb twin. Ring caches cannot reuse linearSynced for this —
	// slot() maintains that field in wrapped SLOT space, which is ambiguous
	// across ring wraps. Maintained by the batched-verify seam
	// (syncLinearKVFromDevicePaged / reloadDevicePagedKVFromLinear); lowered by
	// slot() overwrites and truncate(); zeroed by loadLinearSnapshot, whose
	// callers (state restore) write page rows of unknown lb provenance.
	linearSyncedAbs   int
	sdpaScratch       []*sdpaPagedDecodeScratch
	sdpaScratchCursor int
}

// kvQ8GroupSize is the q8 quantisation group: 64 elements per scale keeps the
// P1 kernels' lane slices (headDim/32 = 8 elements at headDim 256) inside one
// group, so a lane loads exactly one scale per row.
const kvQ8GroupSize = 64

// rowElemBytes is the per-element byte width of a K/V page row.
func (c *devicePagedKVCache) rowElemBytes() int {
	if c.quantQ8 {
		return 1
	}
	return bf16Size
}

// scaleRowBytes is one row's group-scale bytes (0 when not quantised).
func (c *devicePagedKVCache) scaleRowBytes() int {
	if !c.quantQ8 {
		return 0
	}
	return c.kvDim / kvQ8GroupSize * 4
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
	c.kScalePages = nil
	c.vScalePages = nil
	c.kScalePtrs = nil
	c.vScalePtrs = nil
	c.kScaleScratch = nil
	c.vScaleScratch = nil
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
	c.linearSyncedAbs = 0
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
		if allocErr := c.appendPage(); allocErr != nil {
			return nil, nil, 0, allocErr
		}
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
	if pos < c.linearSyncedAbs { // an overwrite below the absolute mirror watermark stales the lb twin
		c.linearSyncedAbs = pos
	}
	return c.kPages[page], c.vPages[page], uint(slot * c.kvDim * c.rowElemBytes()), nil
}

// appendPage allocates the next page under the geometric schedule — the q8
// mode grows the parallel scale pages in lockstep.
func (c *devicePagedKVCache) appendPage() error {
	rows := c.pageRowsFor(len(c.kPages))
	k, v, kPtr, vPtr, err := c.newPage(rows)
	if err != nil {
		return err
	}
	if c.quantQ8 {
		sBytes := uint(rows * c.scaleRowBytes())
		ks := device.NewBufferWithLengthOptions(sBytes, metal.MTLResourceStorageModeShared)
		vs := device.NewBufferWithLengthOptions(sBytes, metal.MTLResourceStorageModeShared)
		if ks == nil || vs == nil || ks.GetID() == 0 || vs.GetID() == 0 {
			return core.NewError("native.devicePagedKVCache.appendPage: failed to allocate scale pages")
		}
		c.kScalePages = append(c.kScalePages, ks)
		c.vScalePages = append(c.vScalePages, vs)
		c.kScalePtrs = append(c.kScalePtrs, (*byte)(ks.Contents()))
		c.vScalePtrs = append(c.vScalePtrs, (*byte)(vs.Contents()))
	}
	c.kPages = append(c.kPages, k)
	c.vPages = append(c.vPages, v)
	c.kPagePtrs = append(c.kPagePtrs, kPtr)
	c.vPagePtrs = append(c.vPagePtrs, vPtr)
	c.pageLens = append(c.pageLens, 0)
	return nil
}

// scaleSlot returns the q8 scale page + row byte offset for a landed row —
// callers pair it with slot(pos) (nil when the cache is not quantised).
func (c *devicePagedKVCache) scaleSlot(pos int) (kScale, vScale metal.MTLBuffer, scaleOff uint) {
	if c == nil || !c.quantQ8 {
		return nil, nil, 0
	}
	cachePos := pos
	if c.ring && c.maxSize > 0 {
		cachePos = pos % c.maxSize
	}
	page := c.pageForPos(cachePos)
	if page >= len(c.kScalePages) {
		return nil, nil, 0
	}
	slot := cachePos - c.pageStartFor(page)
	return c.kScalePages[page], c.vScalePages[page], uint(slot * c.scaleRowBytes())
}

func (c *devicePagedKVCache) newPage(rows int) (metal.MTLBuffer, metal.MTLBuffer, *byte, *byte, error) {
	bytes := uint(rows * c.kvDim * c.rowElemBytes())
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
		if err := c.appendPage(); err != nil {
			return err
		}
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
		dstOff := start * rowBytes
		if c.quantQ8 {
			elems := pageLen * c.kvDim
			srcK := unsafe.Slice((*int8)(unsafe.Pointer(c.kPagePtrs[pageIdx])), elems)
			srcV := unsafe.Slice((*int8)(unsafe.Pointer(c.vPagePtrs[pageIdx])), elems)
			scales := pageLen * c.kvDim / kvQ8GroupSize
			sK := unsafe.Slice((*float32)(unsafe.Pointer(c.kScalePtrs[pageIdx])), scales)
			sV := unsafe.Slice((*float32)(unsafe.Pointer(c.vScalePtrs[pageIdx])), scales)
			kvQ8DequantRows(kBytes[dstOff:dstOff+pageLen*rowBytes], srcK, sK)
			kvQ8DequantRows(vBytes[dstOff:dstOff+pageLen*rowBytes], srcV, sV)
			continue
		}
		copyBytes := pageLen * rowBytes
		srcK := unsafe.Slice(c.kPagePtrs[pageIdx], copyBytes)
		srcV := unsafe.Slice(c.vPagePtrs[pageIdx], copyBytes)
		copy(kBytes[dstOff:dstOff+copyBytes], srcK)
		copy(vBytes[dstOff:dstOff+copyBytes], srcV)
	}
	return c.snapshotK, c.snapshotV, kPtr, vPtr, nil
}

// kvQ8DequantRows expands int8 group-quantised elements into bf16 bytes:
// one f32 scale per kvQ8GroupSize elements, x = q·scale.
func kvQ8DequantRows(dst []byte, q []int8, scales []float32) {
	for g := range scales {
		s := scales[g]
		base := g * kvQ8GroupSize
		for i := range kvQ8GroupSize {
			f := float32(q[base+i]) * s
			bits := mathFloat32bits(f)
			// round-to-nearest-even bf16 truncation
			bits += 0x7FFF + ((bits >> 16) & 1)
			off := (base + i) * 2
			dst[off] = byte(bits >> 16)
			dst[off+1] = byte(bits >> 24)
		}
	}
}

// kvQ8QuantRows quantises bf16 bytes into int8 groups with f32 scales:
// scale = maxabs/127 per group, q = round(x/scale) clamped to ±127.
func kvQ8QuantRows(dstQ []int8, dstScales []float32, src []byte) {
	groups := len(dstScales)
	for g := range groups {
		base := g * kvQ8GroupSize
		maxAbs := float32(0)
		var vals [kvQ8GroupSize]float32
		for i := range kvQ8GroupSize {
			off := (base + i) * 2
			bits := uint32(src[off]) | uint32(src[off+1])<<8
			f := mathFloat32frombits(bits << 16)
			vals[i] = f
			if a := absF32(f); a > maxAbs {
				maxAbs = a
			}
		}
		scale := maxAbs / 127
		dstScales[g] = scale
		inv := float32(0)
		if scale > 0 {
			inv = 1 / scale
		}
		for i := range kvQ8GroupSize {
			q := int32(roundF32(vals[i] * inv))
			if q > 127 {
				q = 127
			} else if q < -127 {
				q = -127
			}
			dstQ[base+i] = int8(q)
		}
	}
}

// kvQuantQMax returns the positive saturation code for a bit-depth: the
// symmetric grid runs -qmax..qmax, so qmax = (1<<(bits-1))-1 → 7/31/127 for
// bits 4/6/8. Any other bit-depth is a programmer error and panics.
func kvQuantQMax(bits int) int {
	switch bits {
	case 4, 6, 8:
		return (1 << (bits - 1)) - 1
	default:
		panic(core.NewError("native.kvQuant: unsupported bits (want 4, 6 or 8)"))
	}
}

// kvQuantPackedLen is the byte length a group-quantised row of elems elements
// occupies at a given bit-depth: 1 byte/elem at q8, two nibbles/byte at q4, a
// contiguous 6-bit stream at q6. kvQ8GroupSize (64) is divisible by 8, so every
// group stays byte-aligned at q6 (64·6 = 384 bits = 48 bytes).
func kvQuantPackedLen(elems, bits int) int {
	return elems * bits / 8
}

// kvQuantPack writes the signed code q for one element into dstPacked. q is
// assumed already clamped to ±kvQuantQMax(bits). Layouts:
//   - bits==8: one int8 per byte (byte(int8(q)) — identical to kvQ8QuantRows).
//   - bits==4: two two's-complement nibbles per byte, element 2i in the low
//     nibble (assignment clears the byte), 2i+1 in the high nibble (OR-in).
//   - bits==6: little-endian bit stream, element e occupies bits e·6 .. e·6+5;
//     bits are OR-ed in, so dstPacked must be zeroed for the q6 region first.
func kvQuantPack(dstPacked []byte, elem int, q int32, bits int) {
	switch bits {
	case 8:
		dstPacked[elem] = byte(int8(q))
	case 4:
		code := byte(q & 0x0F)
		if elem&1 == 0 {
			dstPacked[elem>>1] = code
		} else {
			dstPacked[elem>>1] |= code << 4
		}
	case 6:
		code := uint32(q) & 0x3F
		bitPos := elem * 6
		for k := 0; k < 6; k++ {
			if code&(uint32(1)<<uint(k)) != 0 {
				gb := bitPos + k
				dstPacked[gb>>3] |= 1 << uint(gb&7)
			}
		}
	}
}

// kvQuantUnpack reads and sign-extends the code for one element — the inverse
// of kvQuantPack. bits is assumed already validated by the caller.
func kvQuantUnpack(packed []byte, elem int, bits int) int32 {
	switch bits {
	case 8:
		return int32(int8(packed[elem]))
	case 4:
		var nib byte
		if elem&1 == 0 {
			nib = packed[elem>>1] & 0x0F
		} else {
			nib = packed[elem>>1] >> 4
		}
		if nib >= 8 { // sign-extend the 4-bit two's-complement code
			return int32(nib) - 16
		}
		return int32(nib)
	case 6:
		bitPos := elem * 6
		var code uint32
		for k := 0; k < 6; k++ {
			gb := bitPos + k
			if packed[gb>>3]&(1<<uint(gb&7)) != 0 {
				code |= uint32(1) << uint(k)
			}
		}
		if code >= 32 { // sign-extend the 6-bit two's-complement code
			return int32(code) - 64
		}
		return int32(code)
	default:
		return 0 // unreachable: callers validate bits via kvQuantQMax
	}
}

// kvQuantRows quantises bf16 rows into symmetric per-group N-bit integers with
// f32 group scales — the bit-parameterised generalisation of kvQ8QuantRows.
// bits ∈ {4,6,8} selects the grid: qmax = (1<<(bits-1))-1, scale = maxabs/qmax
// per kvQ8GroupSize-element group, q = round(clamp(x/scale, -qmax..qmax)). The
// group count is taken from len(dstScales); src is bf16 (2 bytes/element) and
// dstPacked must be kvQuantPackedLen(groups·kvQ8GroupSize, bits) bytes. This is
// the host byte layer only — a follow-up kernel reads the packed bytes.
func kvQuantRows(dstPacked []byte, dstScales []float32, src []byte, bits int) {
	qmax := kvQuantQMax(bits)
	if bits != 8 {
		// q4/q6 share bytes or cross byte boundaries and are OR-ed in, so start
		// from a cleared buffer (q8 assigns every byte outright).
		clear(dstPacked)
	}
	for g := range dstScales {
		base := g * kvQ8GroupSize
		maxAbs := float32(0)
		var vals [kvQ8GroupSize]float32
		for i := range kvQ8GroupSize {
			off := (base + i) * 2
			raw := uint32(src[off]) | uint32(src[off+1])<<8
			f := mathFloat32frombits(raw << 16)
			vals[i] = f
			if a := absF32(f); a > maxAbs {
				maxAbs = a
			}
		}
		scale := maxAbs / float32(qmax)
		dstScales[g] = scale
		inv := float32(0)
		if scale > 0 {
			inv = 1 / scale
		}
		for i := range kvQ8GroupSize {
			q := int32(roundF32(vals[i] * inv))
			if q > int32(qmax) {
				q = int32(qmax)
			} else if q < -int32(qmax) {
				q = -int32(qmax)
			}
			kvQuantPack(dstPacked, base+i, q, bits)
		}
	}
}

// kvDequantRows expands N-bit group-quantised codes back into bf16 bytes — the
// exact inverse of kvQuantRows, with the same round-to-nearest-even bf16
// truncation as kvQ8DequantRows. bits ∈ {4,6,8}; the group count is len(scales).
func kvDequantRows(dst []byte, packed []byte, scales []float32, bits int) {
	_ = kvQuantQMax(bits) // validate bits once (panics on unsupported)
	for g := range scales {
		s := scales[g]
		base := g * kvQ8GroupSize
		for i := range kvQ8GroupSize {
			q := kvQuantUnpack(packed, base+i, bits)
			f := float32(q) * s
			bits32 := mathFloat32bits(f)
			// round-to-nearest-even bf16 truncation
			bits32 += 0x7FFF + ((bits32 >> 16) & 1)
			off := (base + i) * 2
			dst[off] = byte(bits32 >> 16)
			dst[off+1] = byte(bits32 >> 24)
		}
	}
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
	if c.quantQ8 {
		// materialise the pages/lengths first (slot is stateful), then quantise
		// rows in parallel — the scalar loop cost ~10s for a 16K-row 26B prefill
		// reload single-threaded, and every row is independent.
		for pos := range tokens {
			if _, _, _, err := c.slot(pos); err != nil {
				return err
			}
		}
		rowGroups := c.kvDim / kvQ8GroupSize
		parallelRows(tokens, func(pos int) {
			page := c.pageForPos(pos)
			slotIdx := pos - c.pageStartFor(page)
			qOff := uintptr(slotIdx * c.kvDim)
			sOff := uintptr(slotIdx * c.scaleRowBytes())
			srcOff := pos * rowBytes
			kQ := unsafe.Slice((*int8)(unsafe.Add(unsafe.Pointer(c.kPagePtrs[page]), qOff)), c.kvDim)
			vQ := unsafe.Slice((*int8)(unsafe.Add(unsafe.Pointer(c.vPagePtrs[page]), qOff)), c.kvDim)
			kS := unsafe.Slice((*float32)(unsafe.Add(unsafe.Pointer(c.kScalePtrs[page]), sOff)), rowGroups)
			vS := unsafe.Slice((*float32)(unsafe.Add(unsafe.Pointer(c.vScalePtrs[page]), sOff)), rowGroups)
			kvQ8QuantRows(kQ, kS, kRows[srcOff:srcOff+rowBytes])
			kvQ8QuantRows(vQ, vS, vRows[srcOff:srcOff+rowBytes])
		})
		c.linearSynced = tokens
		c.linearSyncedAbs = 0 // caller-provided rows: lb-mirror provenance unknown (state restore)
		return nil
	}
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
	c.linearSyncedAbs = 0 // caller-provided rows: lb-mirror provenance unknown (state restore)
	return nil
}

// pagedRowRef resolves an absolute position to its resident page + slot index
// WITHOUT mutating the cache (slot() extends pageLens/length/offset — correct
// for writes, wrong for reads). Errors when the row was never landed.
func (c *devicePagedKVCache) pagedRowRef(pos int) (page, slotIdx int, err error) {
	if c == nil {
		return 0, 0, core.NewError("native.devicePagedKVCache.pagedRowRef: nil cache")
	}
	if pos < 0 {
		return 0, 0, core.NewError("native.devicePagedKVCache.pagedRowRef: negative position")
	}
	cachePos := pos
	if c.ring && c.maxSize > 0 {
		cachePos = pos % c.maxSize
	}
	page = c.pageForPos(cachePos)
	if page >= len(c.kPages) || page >= len(c.pageLens) {
		return 0, 0, core.NewError("native.devicePagedKVCache.pagedRowRef: row page not resident")
	}
	slotIdx = cachePos - c.pageStartFor(page)
	if slotIdx < 0 || slotIdx >= c.pageLens[page] {
		return 0, 0, core.NewError("native.devicePagedKVCache.pagedRowRef: row slot not resident")
	}
	return page, slotIdx, nil
}

// syncRowsToLinear copies rows [from, to) (ABSOLUTE positions) from the pages
// into the linear lb K/V twins, dequantising q8 pages per row — the O(delta)
// sibling of the linearSnapshot+memcpy full path (#372: the MTP verify seam
// paid an O(position) snapshot per round). Row addressing on both sides is
// pos%capacity (ring slots when bounded; the modulo is the identity when the
// twin holds every position), matching the batched pass's landing math.
func (c *devicePagedKVCache) syncRowsToLinear(kDst, vDst []byte, lbRows, from, to int) error {
	if c == nil {
		return core.NewError("native.devicePagedKVCache.syncRowsToLinear: nil cache")
	}
	if lbRows <= 0 || from < 0 || to < from {
		return core.NewError("native.devicePagedKVCache.syncRowsToLinear: invalid row range")
	}
	rowBytes := c.kvDim * bf16Size
	if len(kDst) < lbRows*rowBytes || len(vDst) < lbRows*rowBytes {
		return core.NewError("native.devicePagedKVCache.syncRowsToLinear: linear twin too short")
	}
	rowGroups := c.kvDim / kvQ8GroupSize
	for pos := from; pos < to; pos++ {
		page, slotIdx, err := c.pagedRowRef(pos)
		if err != nil {
			return err
		}
		dstOff := (pos % lbRows) * rowBytes
		if c.quantQ8 {
			qOff := uintptr(slotIdx * c.kvDim)
			sOff := uintptr(slotIdx * c.scaleRowBytes())
			kQ := unsafe.Slice((*int8)(unsafe.Add(unsafe.Pointer(c.kPagePtrs[page]), qOff)), c.kvDim)
			vQ := unsafe.Slice((*int8)(unsafe.Add(unsafe.Pointer(c.vPagePtrs[page]), qOff)), c.kvDim)
			kS := unsafe.Slice((*float32)(unsafe.Add(unsafe.Pointer(c.kScalePtrs[page]), sOff)), rowGroups)
			vS := unsafe.Slice((*float32)(unsafe.Add(unsafe.Pointer(c.vScalePtrs[page]), sOff)), rowGroups)
			kvQ8DequantRows(kDst[dstOff:dstOff+rowBytes], kQ, kS)
			kvQ8DequantRows(vDst[dstOff:dstOff+rowBytes], vQ, vS)
			continue
		}
		srcOff := uintptr(slotIdx * rowBytes)
		copy(kDst[dstOff:dstOff+rowBytes], unsafe.Slice((*byte)(unsafe.Add(unsafe.Pointer(c.kPagePtrs[page]), srcOff)), rowBytes))
		copy(vDst[dstOff:dstOff+rowBytes], unsafe.Slice((*byte)(unsafe.Add(unsafe.Pointer(c.vPagePtrs[page]), srcOff)), rowBytes))
	}
	return nil
}

// loadRowsFromLinear writes rows [from, to) (ABSOLUTE positions) from the
// linear lb K/V twins into the pages via slot() — the mutating direction, so
// pageLens/length/offset extend exactly as a landing would — quantising into
// q8 pages per row. The O(delta) sibling of loadLinearSnapshot for the MTP
// verify seam: only the batch's K landed rows need reloading, not the prefix.
// Unlike the full reload's re-quantise-everything sweep, committed q8 rows are
// never re-coded — each row is quantised once from its fresh bf16 landing, the
// same single-coding the serial chained lane produces.
func (c *devicePagedKVCache) loadRowsFromLinear(kSrc, vSrc []byte, lbRows, from, to int) error {
	if c == nil {
		return core.NewError("native.devicePagedKVCache.loadRowsFromLinear: nil cache")
	}
	if lbRows <= 0 || from < 0 || to < from {
		return core.NewError("native.devicePagedKVCache.loadRowsFromLinear: invalid row range")
	}
	rowBytes := c.kvDim * bf16Size
	if len(kSrc) < lbRows*rowBytes || len(vSrc) < lbRows*rowBytes {
		return core.NewError("native.devicePagedKVCache.loadRowsFromLinear: linear twin too short")
	}
	rowGroups := c.kvDim / kvQ8GroupSize
	for pos := from; pos < to; pos++ {
		_, _, rowOff, err := c.slot(pos)
		if err != nil {
			return err
		}
		cachePos := pos
		if c.ring && c.maxSize > 0 {
			cachePos = pos % c.maxSize
		}
		page := c.pageForPos(cachePos)
		srcOff := (pos % lbRows) * rowBytes
		if c.quantQ8 {
			slotIdx := int(rowOff) / (c.kvDim * c.rowElemBytes())
			qOff := uintptr(slotIdx * c.kvDim)
			sOff := uintptr(slotIdx * c.scaleRowBytes())
			kQ := unsafe.Slice((*int8)(unsafe.Add(unsafe.Pointer(c.kPagePtrs[page]), qOff)), c.kvDim)
			vQ := unsafe.Slice((*int8)(unsafe.Add(unsafe.Pointer(c.vPagePtrs[page]), qOff)), c.kvDim)
			kS := unsafe.Slice((*float32)(unsafe.Add(unsafe.Pointer(c.kScalePtrs[page]), sOff)), rowGroups)
			vS := unsafe.Slice((*float32)(unsafe.Add(unsafe.Pointer(c.vScalePtrs[page]), sOff)), rowGroups)
			kvQ8QuantRows(kQ, kS, kSrc[srcOff:srcOff+rowBytes])
			kvQ8QuantRows(vQ, vS, vSrc[srcOff:srcOff+rowBytes])
			continue
		}
		copy(unsafe.Slice((*byte)(unsafe.Add(unsafe.Pointer(c.kPagePtrs[page]), uintptr(rowOff))), rowBytes), kSrc[srcOff:srcOff+rowBytes])
		copy(unsafe.Slice((*byte)(unsafe.Add(unsafe.Pointer(c.vPagePtrs[page]), uintptr(rowOff))), rowBytes), vSrc[srcOff:srcOff+rowBytes])
	}
	return nil
}

// parallelRows fans fn over [0, n) across the machine's cores — the q8 host
// quantise/dequantise loops are embarrassingly parallel per row.
func parallelRows(n int, fn func(int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		for i := range n {
			fn(i)
		}
		return
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			const stride = 64 // batch to keep the atomic off the hot path
			for {
				start := int(next.Add(stride)) - stride
				if start >= n {
					return
				}
				end := min(start+stride, n)
				for i := start; i < end; i++ {
					fn(i)
				}
			}
		}()
	}
	wg.Wait()
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
		if c.linearSyncedAbs > tokens {
			c.linearSyncedAbs = tokens
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
	if c.linearSyncedAbs > tokens {
		c.linearSyncedAbs = tokens
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

// scaleState mirrors state() for the q8 group-scale pages (nil, nil when the
// cache is not quantised) — the SDPA plan binds them beside the int8 pages.
func (c *devicePagedKVCache) scaleState() (kScales, vScales []metal.MTLBuffer) {
	if c == nil || !c.quantQ8 || len(c.kScalePages) != len(c.kPages) {
		return nil, nil
	}
	n := len(c.kScalePages)
	if cap(c.kScaleScratch) < n {
		c.kScaleScratch = make([]metal.MTLBuffer, n)
		c.vScaleScratch = make([]metal.MTLBuffer, n)
	}
	kScales = c.kScaleScratch[:n]
	vScales = c.vScaleScratch[:n]
	copy(kScales, c.kScalePages)
	copy(vScales, c.vScalePages)
	return kScales, vScales
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
	dModel, nHeads, nKVHeads, headDim, pos, slideW, rotaryDim int, base, scale, ropeScale, eps float32,
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
	// q8 landing targets (#357): the projections cannot write int8 pages, so
	// K/V project into scratch rows, the norms/ropes run there, and the
	// quantise-store hop writes the page row + group scales before the SDPA
	// reads. bf16 landings keep writing the page row directly.
	kDst, vDst := kPage, vPage
	kDstOff, vDstOff := rowOff, rowOff
	var kScalePage, vScalePage metal.MTLBuffer
	var scaleOff uint
	if cache.quantQ8 {
		kSc, vSc := cache.scaleState()
		if err := sdpaPlan.attachQ8(kSc, vSc); err != nil {
			return enc, encConc, err
		}
		kScalePage, vScalePage, scaleOff = cache.scaleSlot(pos)
		if kScalePage == nil || vScalePage == nil || sc.vProj == nil {
			return enc, encConc, core.NewError("native.encAttnHalfKVPaged: q8 cache missing scale pages or vProj scratch")
		}
		kDst, vDst = sc.kProj, sc.vProj
		kDstOff, vDstOff = 0, 0
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
		if err := proj.project(encI, sc.normed, kDst, kDstOff, projK); err != nil {
			endEncodingFast(enc)
			return computeCommandEncoderFast(cb), false, err
		}
		vIdx := projV
		if !proj.hasV() {
			vIdx = projK
		}
		if err := proj.project(encI, sc.normed, vDst, vDstOff, vIdx); err != nil {
			endEncodingFast(enc)
			return computeCommandEncoderFast(cb), false, err
		}
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		// stage 3: q rope ∥ k rope ∥ v norm
		if err := encQKNormRopeAt(encI, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, ropeScale, eps); err != nil {
			endEncodingFast(enc)
			return computeCommandEncoderFast(cb), false, err
		}
		if kNorm.buf != nil {
			if err := encQKNormRopeAt(encI, kDst, kNorm.buf, kDst, kDstOff, kNorm.off, kDstOff, offBuf, offOff, ropeFreqs, nKVHeads, headDim, rotaryDim, base, ropeScale, eps); err != nil {
				endEncodingFast(enc)
				return computeCommandEncoderFast(cb), false, err
			}
		}
		if valueNorm != nil {
			if err := encRMSNormRowsBF16(encI, vDst, valueNorm, vDst, vDstOff, 0, vDstOff, nKVHeads, headDim, eps); err != nil {
				endEncodingFast(enc)
				return computeCommandEncoderFast(cb), false, err
			}
		}
		if cache.quantQ8 {
			// stage 3.5: the quantise-store hop — a barrier orders the staged
			// rows' writes ahead, then both stores land before stage 4's reads.
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			if err := encKVQ8Store(encI, sc.kProj, kPage, rowOff, kScalePage, scaleOff, nKVHeads*headDim); err != nil {
				endEncodingFast(enc)
				return computeCommandEncoderFast(cb), false, err
			}
			if err := encKVQ8Store(encI, sc.vProj, vPage, rowOff, vScalePage, scaleOff, nKVHeads*headDim); err != nil {
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
		if err := encQKNormRopeAt(encI, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, ropeScale, eps); err != nil {
			return enc, false, err
		}
	} else {
		if qNorm.buf != nil {
			if err := encRMSNormRowsBF16(encI, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, nHeads, headDim, eps); err != nil {
				return enc, false, err
			}
		}
		if err := encRopeDecodeAt(encI, sc.q, sc.q, 0, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, ropeScale); err != nil {
			return enc, false, err
		}
	}
	if err := proj.project(encI, sc.normed, kDst, kDstOff, projK); err != nil {
		return enc, false, err
	}
	if fusedKRope {
		if err := encQKNormRopeAt(encI, kDst, kNorm.buf, kDst, kDstOff, kNorm.off, kDstOff, offBuf, offOff, ropeFreqs, nKVHeads, headDim, rotaryDim, base, ropeScale, eps); err != nil {
			return enc, false, err
		}
	} else {
		if kNorm.buf != nil {
			if err := encRMSNormRowsBF16(encI, kDst, kNorm.buf, kDst, kDstOff, kNorm.off, kDstOff, nKVHeads, headDim, eps); err != nil {
				return enc, false, err
			}
		}
		if err := encRopeDecodeAt(encI, kDst, kDst, kDstOff, kDstOff, offBuf, offOff, ropeFreqs, nKVHeads, headDim, rotaryDim, base, ropeScale); err != nil {
			return enc, false, err
		}
	}
	vIdx := projV
	if !proj.hasV() {
		vIdx = projK
	}
	if err := proj.project(encI, sc.normed, vDst, vDstOff, vIdx); err != nil {
		return enc, false, err
	}
	if valueNorm != nil {
		if err := encRMSNormRowsBF16(encI, vDst, valueNorm, vDst, vDstOff, 0, vDstOff, nKVHeads, headDim, eps); err != nil {
			return enc, false, err
		}
	}
	if cache.quantQ8 {
		// the quantise-store hop (hazard tracking orders it after the norms
		// and ahead of the SDPA's page reads on this serial encoder).
		if err := encKVQ8Store(encI, sc.kProj, kPage, rowOff, kScalePage, scaleOff, nKVHeads*headDim); err != nil {
			return enc, false, err
		}
		if err := encKVQ8Store(encI, sc.vProj, vPage, rowOff, vScalePage, scaleOff, nKVHeads*headDim); err != nil {
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
	dModel, nHeads, nKVHeads, headDim, pos, slideW, rotaryDim int, base, scale, ropeScale, eps float32,
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
		if err := encQKNormRopeAt(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, ropeScale, eps); err != nil {
			return err
		}
	} else {
		if qNorm.buf != nil {
			if err := encRMSNormRowsBF16(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, nHeads, headDim, eps); err != nil {
				return err
			}
		}
		if err := encRopeDecodeAt(enc, sc.q, sc.q, 0, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, ropeScale); err != nil {
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
	plan, err := buildSDPAPagedDecodePlan(sc.q, keyPages, valuePages, pageLens, kHead, kSeq, vHead, vSeq, sc.attn, pagedScratch, nHeads, nKVHeads, headDim, scale)
	if err != nil {
		return err
	}
	if cache.quantQ8 {
		// sharers attend the OWNER's q8 pages (read-only — no landing here).
		kSc, vSc := cache.scaleState()
		if err := plan.attachQ8(kSc, vSc); err != nil {
			return err
		}
	}
	plan.emitP1s(enc)
	plan.emitP2(enc)
	if err := proj.project(enc, sc.attn, sc.attnOut, 0, projO); err != nil {
		return err
	}
	return encResidualMaybeNorm(enc, x, sc.attnOut, sc.normed, h, postAttnNorm, dModel, eps)
}
