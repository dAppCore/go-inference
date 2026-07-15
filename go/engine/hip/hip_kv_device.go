// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"strconv"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const (
	rocmDeviceKVDescriptorVersion                              uint32 = 1
	rocmDeviceKVDescriptorHeaderBytes                                 = 32
	rocmDeviceKVDescriptorPageBytes                                   = 64
	rocmDeviceKVLaunchDescriptorBytes                                 = 64
	hipKVEncodeTokenLaunchArgsVersion                          uint32 = 1
	hipKVEncodeTokenLaunchArgsBytes                                   = 96
	hipKVEncodeTokenValueNormLaunchArgsVersion                 uint32 = 1
	hipKVEncodeTokenValueNormLaunchArgsBytes                          = 112
	hipKVEncodeTokenValueNormDescriptorAppendLaunchArgsVersion uint32 = 1
	hipKVEncodeTokenValueNormDescriptorAppendLaunchArgsBytes          = 248
	hipKVEncodeTokenValueNormMaxHeads                                 = 64
	hipKVEncodeTokenBlockSize                                  uint32 = 256
	hipKVDescriptorAppendLaunchArgsVersion                     uint32 = 1
	hipKVDescriptorAppendLaunchArgsBytes                              = 128
	hipKVDescriptorAppendBlockSize                             uint32 = 64
)

const (
	rocmDeviceKVDescriptorModeFP16   uint32 = 1
	rocmDeviceKVDescriptorModeQ8     uint32 = 2
	rocmDeviceKVDescriptorModeKQ8VQ4 uint32 = 3
)

const (
	rocmDeviceKVHotPageCapacity         = 512
	rocmDeviceKVPagePoolMinCapacity     = 16
	rocmDeviceKVPagePoolMaxCapacity     = 128 * 1024
	rocmDeviceKVCachePoolMax            = 4096
	rocmDeviceKVDescriptorTablePoolMax  = 4096
	rocmDeviceKVHostPoolWarmDepth       = 128
	rocmGemma4Q4DeviceKVBlockSize       = 512
	rocmGemma4Q4GlobalDeviceKVBlockSize = 512
)

const (
	rocmDeviceKVDescriptorPointerPoolMaxBytes   = 32 << 20
	rocmDeviceKVDescriptorPointerPoolMaxPerSize = 4096
)

const (
	rocmDeviceKVDescriptorEncodingFP16        uint32 = 1
	rocmDeviceKVDescriptorEncodingQ8          uint32 = 2
	rocmDeviceKVDescriptorEncodingQ4          uint32 = 3
	rocmDeviceKVDescriptorEncodingQ8Rows      uint32 = 4
	rocmDeviceKVDescriptorEncodingQ4Rows      uint32 = 5
	rocmDeviceKVDescriptorEncodingQ8RowsI     uint32 = 6
	rocmDeviceKVDescriptorEncodingQ4RowsI     uint32 = 7
	rocmKVDescriptorAppendModeGrowLastPage    uint64 = 1
	rocmKVDescriptorAppendModeBuildSinglePage uint64 = 2
)

type rocmDeviceKVCache struct {
	driver     nativeHIPDriver
	mode       string
	blockSize  int
	pages      []rocmDeviceKVPage
	tokenCount int
	closed     bool
	borrowed   bool
}

var rocmDeviceKVCachePool = struct {
	sync.Mutex
	caches []*rocmDeviceKVCache
}{
	caches: make([]*rocmDeviceKVCache, 0, rocmDeviceKVCachePoolMax),
}

func rocmBorrowDeviceKVCache(driver nativeHIPDriver, mode string, blockSize, tokenCount int, pages []rocmDeviceKVPage, borrowed bool) *rocmDeviceKVCache {
	rocmDeviceKVCachePool.Lock()
	count := len(rocmDeviceKVCachePool.caches)
	if count > 0 {
		cache := rocmDeviceKVCachePool.caches[count-1]
		rocmDeviceKVCachePool.caches[count-1] = nil
		rocmDeviceKVCachePool.caches = rocmDeviceKVCachePool.caches[:count-1]
		rocmDeviceKVCachePool.Unlock()
		*cache = rocmDeviceKVCache{
			driver:     driver,
			mode:       mode,
			blockSize:  blockSize,
			pages:      pages,
			tokenCount: tokenCount,
			borrowed:   borrowed,
		}
		return cache
	}
	rocmDeviceKVCachePool.Unlock()
	cache := &rocmDeviceKVCache{}
	*cache = rocmDeviceKVCache{
		driver:     driver,
		mode:       mode,
		blockSize:  blockSize,
		pages:      pages,
		tokenCount: tokenCount,
		borrowed:   borrowed,
	}
	return cache
}

func rocmReleaseDeviceKVCache(cache *rocmDeviceKVCache) {
	if cache == nil {
		return
	}
	*cache = rocmDeviceKVCache{closed: true}
	rocmDeviceKVCachePool.Lock()
	if len(rocmDeviceKVCachePool.caches) < rocmDeviceKVCachePoolMax {
		rocmDeviceKVCachePool.caches = append(rocmDeviceKVCachePool.caches, cache)
	}
	rocmDeviceKVCachePool.Unlock()
}

func rocmPrewarmDeviceKVHostPools() {
	for i := 0; i < rocmDeviceKVHostPoolWarmDepth; i++ {
		rocmReleaseDeviceKVCache(&rocmDeviceKVCache{closed: true})
		rocmReleaseDeviceKVDescriptorTable(&rocmDeviceKVDescriptorTable{poolable: true})
	}
	for _, capacity := range []int{rocmDeviceKVPagePoolMinCapacity, rocmDeviceKVHotPageCapacity} {
		for i := 0; i < rocmDeviceKVHostPoolWarmDepth; i++ {
			rocmDeviceKVReleasePageSlice(make([]rocmDeviceKVPage, 0, capacity))
		}
	}
	for _, capacity := range []int{
		rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVDescriptorPageBytes,
		rocmDeviceKVDescriptorHeaderBytes + 2*rocmDeviceKVDescriptorPageBytes,
		rocmDeviceKVDescriptorHeaderBytes + 4*rocmDeviceKVDescriptorPageBytes,
		rocmDeviceKVDescriptorHeaderBytes + 8*rocmDeviceKVDescriptorPageBytes,
		rocmDeviceKVDescriptorHeaderBytes + 16*rocmDeviceKVDescriptorPageBytes,
		int(rocmDeviceKVDescriptorHotTableBytes()),
	} {
		for i := 0; i < rocmDeviceKVHostPoolWarmDepth; i++ {
			rocmDeviceKVReleaseDescriptorBytes(make([]byte, 0, capacity))
		}
	}
}

type rocmDeviceKVPage struct {
	tokenStart int
	tokenCount int
	keyWidth   int
	valueWidth int
	key        rocmDeviceKVTensor
	value      rocmDeviceKVTensor
	owned      bool
}

type rocmDeviceKVTensor struct {
	pointer           nativeDevicePointer
	sizeBytes         uint64
	encoding          string
	allocationPointer nativeDevicePointer
	allocationBytes   uint64
}

type rocmDeviceKVDescriptorTable struct {
	driver          nativeHIPDriver
	pointer         nativeDevicePointer
	sizeBytes       uint64
	allocationBytes uint64
	version         uint32
	pageCount       int
	closed          bool
	borrowed        bool
	poolable        bool
}

var rocmDeviceKVDescriptorTablePool = struct {
	sync.Mutex
	entries []*rocmDeviceKVDescriptorTable
}{}

type rocmDeviceKVDescriptorPointerPoolEntry struct {
	driver  nativeHIPDriver
	pointer nativeDevicePointer
}

var rocmDeviceKVDescriptorPointerPool = struct {
	sync.Mutex
	entries map[uint64][]rocmDeviceKVDescriptorPointerPoolEntry
	bytes   uint64
}{
	entries: make(map[uint64][]rocmDeviceKVDescriptorPointerPoolEntry),
}

type rocmDeviceKVLaunchDescriptor struct {
	DescriptorPointer nativeDevicePointer
	DescriptorBytes   uint64
	DescriptorVersion uint32
	Mode              string
	ModeCode          uint32
	BlockSize         int
	PageCount         int
	TokenCount        int
	KeyWidth          int
	ValueWidth        int
	StatusPointer     nativeDevicePointer
	StatusValue       uint32
}

func rocmBorrowDeviceKVDescriptorTable(driver nativeHIPDriver, pointer nativeDevicePointer, sizeBytes uint64, version uint32, pageCount int, borrowed, poolable bool) *rocmDeviceKVDescriptorTable {
	return rocmBorrowDeviceKVDescriptorTableAllocated(driver, pointer, sizeBytes, sizeBytes, version, pageCount, borrowed, poolable)
}

func rocmBorrowDeviceKVDescriptorTableAllocated(driver nativeHIPDriver, pointer nativeDevicePointer, sizeBytes, allocationBytes uint64, version uint32, pageCount int, borrowed, poolable bool) *rocmDeviceKVDescriptorTable {
	var table *rocmDeviceKVDescriptorTable
	if poolable {
		rocmDeviceKVDescriptorTablePool.Lock()
		count := len(rocmDeviceKVDescriptorTablePool.entries)
		if count > 0 {
			table = rocmDeviceKVDescriptorTablePool.entries[count-1]
			rocmDeviceKVDescriptorTablePool.entries[count-1] = nil
			rocmDeviceKVDescriptorTablePool.entries = rocmDeviceKVDescriptorTablePool.entries[:count-1]
		}
		rocmDeviceKVDescriptorTablePool.Unlock()
	}
	if table == nil {
		table = &rocmDeviceKVDescriptorTable{}
	}
	if allocationBytes == 0 {
		allocationBytes = sizeBytes
	}
	*table = rocmDeviceKVDescriptorTable{
		driver:          driver,
		pointer:         pointer,
		sizeBytes:       sizeBytes,
		allocationBytes: allocationBytes,
		version:         version,
		pageCount:       pageCount,
		borrowed:        borrowed,
		poolable:        poolable,
	}
	return table
}

func rocmReleaseDeviceKVDescriptorTable(table *rocmDeviceKVDescriptorTable) {
	if table == nil {
		return
	}
	*table = rocmDeviceKVDescriptorTable{closed: true}
	rocmDeviceKVDescriptorTablePool.Lock()
	if len(rocmDeviceKVDescriptorTablePool.entries) < rocmDeviceKVDescriptorTablePoolMax {
		rocmDeviceKVDescriptorTablePool.entries = append(rocmDeviceKVDescriptorTablePool.entries, table)
	}
	rocmDeviceKVDescriptorTablePool.Unlock()
}

func rocmDeviceKVDescriptorHotTableBytes() uint64 {
	return uint64(rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVHotPageCapacity*rocmDeviceKVDescriptorPageBytes)
}

func rocmDeviceKVDescriptorTableAllocationBytes(sizeBytes uint64) uint64 {
	if sizeBytes <= uint64(rocmDeviceKVDescriptorHeaderBytes) {
		return sizeBytes
	}
	pageBytes := uint64(rocmDeviceKVDescriptorPageBytes)
	pageCount := int((sizeBytes - uint64(rocmDeviceKVDescriptorHeaderBytes) + pageBytes - 1) / pageBytes)
	pageCapacity := rocmDeviceKVDescriptorPageCapacity(pageCount)
	if pageCapacity > rocmDeviceKVPagePoolMaxCapacity {
		return sizeBytes
	}
	return uint64(rocmDeviceKVDescriptorHeaderBytes + pageCapacity*rocmDeviceKVDescriptorPageBytes)
}

func rocmDeviceKVDescriptorPageCapacity(pageCount int) int {
	if pageCount <= 0 {
		return 0
	}
	capacity := rocmDeviceKVHotPageCapacity
	for capacity < pageCount && capacity < rocmDeviceKVPagePoolMaxCapacity {
		capacity *= 2
	}
	if capacity < pageCount {
		return pageCount
	}
	return capacity
}

func rocmDeviceKVDescriptorPointerPoolable(sizeBytes uint64) bool {
	return sizeBytes >= rocmDeviceKVDescriptorHotTableBytes() &&
		sizeBytes <= uint64(rocmDeviceKVDescriptorHeaderBytes+rocmDeviceKVPagePoolMaxCapacity*rocmDeviceKVDescriptorPageBytes)
}

func rocmDeviceKVDescriptorExactPointerPoolable(sizeBytes uint64) bool {
	return sizeBytes > uint64(rocmDeviceKVDescriptorHeaderBytes) &&
		sizeBytes < rocmDeviceKVDescriptorHotTableBytes()
}

func rocmDeviceKVDescriptorPointerPoolTake(driver nativeHIPDriver, sizeBytes uint64) (nativeDevicePointer, bool) {
	if driver == nil || sizeBytes == 0 {
		return 0, false
	}
	rocmDeviceKVDescriptorPointerPool.Lock()
	entries := rocmDeviceKVDescriptorPointerPool.entries[sizeBytes]
	for index := len(entries) - 1; index >= 0; index-- {
		entry := entries[index]
		if entry.driver != driver {
			continue
		}
		entries[index] = entries[len(entries)-1]
		entries[len(entries)-1] = rocmDeviceKVDescriptorPointerPoolEntry{}
		entries = entries[:len(entries)-1]
		rocmDeviceKVDescriptorPointerPool.entries[sizeBytes] = entries
		rocmDeviceKVDescriptorPointerPool.bytes -= sizeBytes
		rocmDeviceKVDescriptorPointerPool.Unlock()
		return entry.pointer, true
	}
	rocmDeviceKVDescriptorPointerPool.Unlock()
	return 0, false
}

func rocmPrewarmDeviceKVDescriptorPointerPool(driver nativeHIPDriver, exactCount, hotCount int) {
	if driver == nil || !driver.Available() {
		return
	}
	prewarm := func(sizeBytes uint64, count int) {
		if sizeBytes == 0 || count <= 0 {
			return
		}
		for i := 0; i < count; i++ {
			pointer, err := driver.Malloc(sizeBytes)
			if err != nil {
				return
			}
			if err := rocmDeviceKVDescriptorTableFree(driver, pointer, sizeBytes); err != nil {
				_ = driver.Free(pointer)
				return
			}
		}
	}
	for pageCount := 1; pageCount <= 32; pageCount++ {
		count := hotCount
		if pageCount == 1 {
			count = exactCount
		}
		prewarm(uint64(rocmDeviceKVDescriptorHeaderBytes+pageCount*rocmDeviceKVDescriptorPageBytes), count)
	}
	prewarm(rocmDeviceKVDescriptorHotTableBytes(), hotCount)
}

func rocmDeviceKVDescriptorTableMallocExact(driver nativeHIPDriver, sizeBytes uint64) (nativeDevicePointer, uint64, error) {
	if driver == nil {
		return 0, 0, core.E("rocm.KVCache.DeviceDescriptor", "HIP driver is nil", nil)
	}
	if rocmDeviceKVDescriptorExactPointerPoolable(sizeBytes) {
		if pointer, ok := rocmDeviceKVDescriptorPointerPoolTake(driver, sizeBytes); ok {
			return pointer, sizeBytes, nil
		}
	}
	pointer, err := hipMallocLabeled(driver, "rocm.KVCache.DeviceDescriptor", "KV descriptor table", sizeBytes)
	return pointer, sizeBytes, err
}

func rocmDeviceKVDescriptorTableMalloc(driver nativeHIPDriver, sizeBytes uint64) (nativeDevicePointer, uint64, error) {
	if driver == nil {
		return 0, 0, core.E("rocm.KVCache.DeviceDescriptor", "HIP driver is nil", nil)
	}
	allocationBytes := rocmDeviceKVDescriptorTableAllocationBytes(sizeBytes)
	if rocmDeviceKVDescriptorPointerPoolable(allocationBytes) {
		if pointer, ok := rocmDeviceKVDescriptorPointerPoolTake(driver, allocationBytes); ok {
			return pointer, allocationBytes, nil
		}
	}
	pointer, err := hipMallocLabeled(driver, "rocm.KVCache.DeviceDescriptor", "KV descriptor table", allocationBytes)
	return pointer, allocationBytes, err
}

func rocmDeviceKVDescriptorTableFree(driver nativeHIPDriver, pointer nativeDevicePointer, sizeBytes uint64) error {
	if pointer == 0 {
		return nil
	}
	if driver == nil {
		return core.E("rocm.KVCache.DeviceDescriptor", "HIP driver is nil", nil)
	}
	if rocmDeviceKVDescriptorPointerPoolable(sizeBytes) || rocmDeviceKVDescriptorExactPointerPoolable(sizeBytes) {
		rocmDeviceKVDescriptorPointerPool.Lock()
		entries := rocmDeviceKVDescriptorPointerPool.entries[sizeBytes]
		if rocmDeviceKVDescriptorPointerPool.bytes+sizeBytes <= rocmDeviceKVDescriptorPointerPoolMaxBytes &&
			len(entries) < rocmDeviceKVDescriptorPointerPoolMaxPerSize {
			rocmDeviceKVDescriptorPointerPool.entries[sizeBytes] = append(entries, rocmDeviceKVDescriptorPointerPoolEntry{
				driver:  driver,
				pointer: pointer,
			})
			rocmDeviceKVDescriptorPointerPool.bytes += sizeBytes
			rocmDeviceKVDescriptorPointerPool.Unlock()
			return nil
		}
		rocmDeviceKVDescriptorPointerPool.Unlock()
	}
	return driver.Free(pointer)
}

type hipKVEncodeTokenLaunchArgs struct {
	KeyInputPointer    nativeDevicePointer
	ValueInputPointer  nativeDevicePointer
	KeyOutputPointer   nativeDevicePointer
	ValueOutputPointer nativeDevicePointer
	KeyCount           int
	ValueCount         int
	KeyInputBytes      uint64
	ValueInputBytes    uint64
	KeyOutputBytes     uint64
	ValueOutputBytes   uint64
	KeyEncoding        uint32
	ValueEncoding      uint32
	KeyWidth           int
	ValueWidth         int
	TokenCount         int
}

type hipKVEncodeTokenValueNormLaunchArgs struct {
	KeyInputPointer    nativeDevicePointer
	ValueInputPointer  nativeDevicePointer
	KeyOutputPointer   nativeDevicePointer
	ValueOutputPointer nativeDevicePointer
	KeyCount           int
	ValueCount         int
	KeyInputBytes      uint64
	ValueInputBytes    uint64
	KeyOutputBytes     uint64
	ValueOutputBytes   uint64
	KeyEncoding        uint32
	ValueEncoding      uint32
	KeyWidth           int
	ValueWidth         int
	TokenCount         int
	ValueHeadDim       int
	ValueHeadCount     int
	Epsilon            float32
}

type hipKVDescriptorAppendLaunchArgs struct {
	PreviousDescriptorPointer nativeDevicePointer
	OutputDescriptorPointer   nativeDevicePointer
	NewKeyPointer             nativeDevicePointer
	NewValuePointer           nativeDevicePointer
	PreviousDescriptorBytes   uint64
	OutputDescriptorBytes     uint64
	NewKeyBytes               uint64
	NewValueBytes             uint64
	ModeCode                  uint32
	BlockSize                 int
	OutputPageCount           int
	OutputTokenCount          int
	KeyWidth                  int
	ValueWidth                int
	NewKeyEncoding            uint32
	NewValueEncoding          uint32
	TrimStart                 int
	Reserved0                 uint64
	Reserved1                 uint64
}

type hipKVEncodeTokenValueNormDescriptorAppendLaunchArgs struct {
	Encode     hipKVEncodeTokenValueNormLaunchArgs
	Descriptor hipKVDescriptorAppendLaunchArgs
}

type rocmDeviceKVDescriptor struct {
	Mode       string
	BlockSize  int
	TokenCount int
	Pages      []rocmDeviceKVPageDescriptor
}

type rocmDeviceKVPageDescriptor struct {
	TokenStart    int
	TokenCount    int
	KeyWidth      int
	ValueWidth    int
	KeyPointer    nativeDevicePointer
	ValuePointer  nativeDevicePointer
	KeyBytes      uint64
	ValueBytes    uint64
	KeyEncoding   string
	ValueEncoding string
}

type rocmDeviceKVPageSlicePool struct {
	sync.Mutex
	pages [][]rocmDeviceKVPage
}

var rocmDeviceKVPageSlicePools sync.Map

const rocmDeviceKVPageSlicePoolMaxPerCapacity = 512

type rocmDeviceKVDescriptorBytePool struct {
	sync.Mutex
	buffers [][]byte
}

var rocmDeviceKVDescriptorBytePools sync.Map
var rocmDeviceKVPayloadBytePools sync.Map

const (
	rocmDeviceKVDescriptorBytePoolMaxPerCapacity = 512
	rocmDeviceKVDescriptorBytePoolMinBytes       = rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVDescriptorPageBytes
	rocmDeviceKVDescriptorBytePoolMaxBytes       = rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVPagePoolMaxCapacity*rocmDeviceKVDescriptorPageBytes
	rocmDeviceKVPayloadBytePoolMaxPerCapacity    = 512
	rocmDeviceKVPayloadBytePoolMinBytes          = 8
	rocmDeviceKVPayloadBytePoolMaxBytes          = 4096
	rocmDeviceKVLabelIntMax                      = 65536
)

var rocmDeviceKVLabelInts = func() [rocmDeviceKVLabelIntMax + 1]string {
	var labels [rocmDeviceKVLabelIntMax + 1]string
	for value := range labels {
		labels[value] = strconv.Itoa(value)
	}
	return labels
}()

type rocmDeviceKVTensorPoolEntry struct {
	driver  nativeHIPDriver
	pointer nativeDevicePointer
}

type rocmDeviceKVTensorPoolBucket struct {
	first rocmDeviceKVTensorPoolEntry
	rest  []rocmDeviceKVTensorPoolEntry
}

func (bucket rocmDeviceKVTensorPoolBucket) len() int {
	if bucket.first.pointer == 0 {
		return 0
	}
	return 1 + len(bucket.rest)
}

var rocmDeviceKVTensorPool = struct {
	sync.Mutex
	entries map[uint64]rocmDeviceKVTensorPoolBucket
	bytes   uint64
}{
	entries: make(map[uint64]rocmDeviceKVTensorPoolBucket),
}

const (
	rocmDeviceKVTensorPoolMaxPerSize = 4096
	rocmDeviceKVTensorPoolMaxBytes   = 512 << 20
	// Covers local/SWA and retained global q6 interleaved pages while keeping oversized pages uncached.
	rocmDeviceKVTensorPoolDefaultBytes = 2 << 20
)

func rocmDeviceKVBorrowPageSlice(length, minCapacity int) []rocmDeviceKVPage {
	if minCapacity < length {
		minCapacity = length
	}
	minCapacity = rocmDeviceKVPageSliceCapacity(minCapacity)
	if minCapacity >= rocmDeviceKVPagePoolMinCapacity && minCapacity <= rocmDeviceKVPagePoolMaxCapacity {
		poolValue, ok := rocmDeviceKVPageSlicePools.Load(minCapacity)
		if !ok {
			pool := &rocmDeviceKVPageSlicePool{}
			poolValue, _ = rocmDeviceKVPageSlicePools.LoadOrStore(minCapacity, pool)
		}
		pool := poolValue.(*rocmDeviceKVPageSlicePool)
		pool.Lock()
		if index := len(pool.pages) - 1; index >= 0 {
			pages := pool.pages[index]
			pool.pages[index] = nil
			pool.pages = pool.pages[:index]
			pool.Unlock()
			return pages[:length]
		}
		pool.Unlock()
	}
	return make([]rocmDeviceKVPage, length, minCapacity)
}

func rocmDeviceKVPageSliceCapacity(minCapacity int) int {
	if minCapacity <= 0 {
		return 0
	}
	capacity := rocmDeviceKVPagePoolMinCapacity
	for capacity < minCapacity && capacity < rocmDeviceKVPagePoolMaxCapacity {
		capacity *= 2
	}
	if capacity < minCapacity {
		return minCapacity
	}
	return capacity
}

func rocmDeviceKVCopyPagesWithExtra(pages []rocmDeviceKVPage, extra int) []rocmDeviceKVPage {
	out := rocmDeviceKVBorrowPageSlice(len(pages), len(pages)+extra)
	copy(out, pages)
	return out
}

func rocmDeviceKVReleasePageSlice(pages []rocmDeviceKVPage) {
	if cap(pages) < rocmDeviceKVPagePoolMinCapacity || cap(pages) > rocmDeviceKVPagePoolMaxCapacity {
		return
	}
	full := pages[:cap(pages)]
	for index := range full {
		full[index] = rocmDeviceKVPage{}
	}
	poolValue, ok := rocmDeviceKVPageSlicePools.Load(cap(full))
	if !ok {
		pool := &rocmDeviceKVPageSlicePool{}
		poolValue, _ = rocmDeviceKVPageSlicePools.LoadOrStore(cap(full), pool)
	}
	pool := poolValue.(*rocmDeviceKVPageSlicePool)
	pool.Lock()
	if len(pool.pages) < rocmDeviceKVPageSlicePoolMaxPerCapacity {
		pool.pages = append(pool.pages, full[:0])
	}
	pool.Unlock()
}

func rocmDeviceKVBorrowDescriptorBytes(length int) []byte {
	if length <= 0 {
		return nil
	}
	capacity := rocmDeviceKVDescriptorByteCapacity(length)
	if capacity >= rocmDeviceKVDescriptorBytePoolMinBytes && capacity <= rocmDeviceKVDescriptorBytePoolMaxBytes {
		poolValue, ok := rocmDeviceKVDescriptorBytePools.Load(capacity)
		if !ok {
			pool := &rocmDeviceKVDescriptorBytePool{}
			poolValue, _ = rocmDeviceKVDescriptorBytePools.LoadOrStore(capacity, pool)
		}
		pool := poolValue.(*rocmDeviceKVDescriptorBytePool)
		pool.Lock()
		if index := len(pool.buffers) - 1; index >= 0 {
			buffer := pool.buffers[index]
			pool.buffers[index] = nil
			pool.buffers = pool.buffers[:index]
			pool.Unlock()
			return buffer[:length]
		}
		pool.Unlock()
	}
	return make([]byte, length, capacity)
}

func rocmDeviceKVDescriptorByteCapacity(length int) int {
	if length <= 0 {
		return 0
	}
	if length < int(rocmDeviceKVDescriptorHotTableBytes()) {
		return length
	}
	if length <= rocmDeviceKVDescriptorHeaderBytes {
		return length
	}
	pageBytes := rocmDeviceKVDescriptorPageBytes
	pageCount := (length - rocmDeviceKVDescriptorHeaderBytes + pageBytes - 1) / pageBytes
	pageCapacity := rocmDeviceKVPageSliceCapacity(pageCount)
	if pageCapacity > rocmDeviceKVPagePoolMaxCapacity {
		return length
	}
	return rocmDeviceKVDescriptorHeaderBytes + pageCapacity*pageBytes
}

func rocmDeviceKVReleaseDescriptorBytes(payload []byte) {
	if cap(payload) < rocmDeviceKVDescriptorBytePoolMinBytes || cap(payload) > rocmDeviceKVDescriptorBytePoolMaxBytes {
		return
	}
	full := payload[:cap(payload)]
	clear(full)
	poolValue, ok := rocmDeviceKVDescriptorBytePools.Load(cap(full))
	if !ok {
		pool := &rocmDeviceKVDescriptorBytePool{}
		poolValue, _ = rocmDeviceKVDescriptorBytePools.LoadOrStore(cap(full), pool)
	}
	pool := poolValue.(*rocmDeviceKVDescriptorBytePool)
	pool.Lock()
	if len(pool.buffers) < rocmDeviceKVDescriptorBytePoolMaxPerCapacity {
		pool.buffers = append(pool.buffers, full[:0])
	}
	pool.Unlock()
}

func rocmDeviceKVBorrowPayloadBytes(length int) []byte {
	if length <= 0 {
		return nil
	}
	capacity := rocmDeviceKVPayloadByteCapacity(length)
	if capacity >= rocmDeviceKVPayloadBytePoolMinBytes && capacity <= rocmDeviceKVPayloadBytePoolMaxBytes {
		poolValue, ok := rocmDeviceKVPayloadBytePools.Load(capacity)
		if !ok {
			pool := &rocmDeviceKVDescriptorBytePool{}
			poolValue, _ = rocmDeviceKVPayloadBytePools.LoadOrStore(capacity, pool)
		}
		pool := poolValue.(*rocmDeviceKVDescriptorBytePool)
		pool.Lock()
		if index := len(pool.buffers) - 1; index >= 0 {
			buffer := pool.buffers[index]
			pool.buffers[index] = nil
			pool.buffers = pool.buffers[:index]
			pool.Unlock()
			return buffer[:length]
		}
		pool.Unlock()
	}
	return make([]byte, length, capacity)
}

func rocmDeviceKVPayloadByteCapacity(length int) int {
	if length <= 0 {
		return 0
	}
	capacity := 8
	for capacity < length && capacity < rocmDeviceKVPayloadBytePoolMaxBytes {
		capacity *= 2
	}
	if capacity < length {
		return length
	}
	return capacity
}

func rocmDeviceKVReleasePayloadBytes(payload []byte) {
	if cap(payload) < rocmDeviceKVPayloadBytePoolMinBytes || cap(payload) > rocmDeviceKVPayloadBytePoolMaxBytes {
		return
	}
	full := payload[:cap(payload)]
	clear(full)
	poolValue, ok := rocmDeviceKVPayloadBytePools.Load(cap(full))
	if !ok {
		pool := &rocmDeviceKVDescriptorBytePool{}
		poolValue, _ = rocmDeviceKVPayloadBytePools.LoadOrStore(cap(full), pool)
	}
	pool := poolValue.(*rocmDeviceKVDescriptorBytePool)
	pool.Lock()
	if len(pool.buffers) < rocmDeviceKVPayloadBytePoolMaxPerCapacity {
		pool.buffers = append(pool.buffers, full[:0])
	}
	pool.Unlock()
}

type rocmDeviceKVTensorPoolDefaultDriver interface {
	rocmDefaultKVTensorPool()
}

type rocmNativeHIPDriverUnwrapper interface {
	rocmUnwrapNativeHIPDriver() nativeHIPDriver
}

func rocmDeviceKVTensorPoolDefaultDriverEnabled(driver nativeHIPDriver) bool {
	for depth := 0; driver != nil && depth < 4; depth++ {
		if _, ok := driver.(rocmDeviceKVTensorPoolDefaultDriver); ok {
			return true
		}
		unwrapper, ok := driver.(rocmNativeHIPDriverUnwrapper)
		if !ok {
			return false
		}
		driver = unwrapper.rocmUnwrapNativeHIPDriver()
	}
	return false
}

func rocmDeviceKVTensorPoolEnabled(driver nativeHIPDriver, sizeBytes uint64) bool {
	if os.Getenv("GO_ROCM_DISABLE_KV_TENSOR_POOL") == "1" {
		return false
	}
	if os.Getenv("GO_ROCM_ENABLE_KV_TENSOR_POOL") == "1" {
		return true
	}
	return sizeBytes > 0 &&
		sizeBytes <= rocmDeviceKVTensorPoolDefaultBytes &&
		rocmDeviceKVTensorPoolDefaultDriverEnabled(driver)
}

func rocmDeviceKVTensorMalloc(driver nativeHIPDriver, sizeBytes uint64) (nativeDevicePointer, error) {
	if !rocmDeviceKVTensorPoolEnabled(driver, sizeBytes) {
		return hipMallocLabeled(driver, "rocm.KVCache.DeviceTensor", "KV tensor", sizeBytes)
	}
	rocmDeviceKVTensorPool.Lock()
	bucket := rocmDeviceKVTensorPool.entries[sizeBytes]
	if bucket.first.pointer != 0 {
		if bucket.first.driver == driver {
			pointer := bucket.first.pointer
			if count := len(bucket.rest); count > 0 {
				bucket.first = bucket.rest[count-1]
				bucket.rest[count-1] = rocmDeviceKVTensorPoolEntry{}
				bucket.rest = bucket.rest[:count-1]
			} else {
				bucket.first = rocmDeviceKVTensorPoolEntry{}
			}
			rocmDeviceKVTensorPool.entries[sizeBytes] = bucket
			rocmDeviceKVTensorPool.bytes -= sizeBytes
			rocmDeviceKVTensorPool.Unlock()
			return pointer, nil
		}
		for index := len(bucket.rest) - 1; index >= 0; index-- {
			entry := bucket.rest[index]
			if entry.driver != driver {
				continue
			}
			pointer := entry.pointer
			bucket.rest[index] = bucket.rest[len(bucket.rest)-1]
			bucket.rest[len(bucket.rest)-1] = rocmDeviceKVTensorPoolEntry{}
			bucket.rest = bucket.rest[:len(bucket.rest)-1]
			rocmDeviceKVTensorPool.entries[sizeBytes] = bucket
			rocmDeviceKVTensorPool.bytes -= sizeBytes
			rocmDeviceKVTensorPool.Unlock()
			return pointer, nil
		}
	}
	rocmDeviceKVTensorPool.Unlock()
	return hipMallocLabeled(driver, "rocm.KVCache.DeviceTensor", "KV tensor", sizeBytes)
}

func rocmDeviceKVTensorFree(driver nativeHIPDriver, pointer nativeDevicePointer, sizeBytes uint64) error {
	if pointer == 0 {
		return nil
	}
	if rocmDeviceKVTensorPoolEnabled(driver, sizeBytes) && driver != nil && sizeBytes > 0 {
		rocmDeviceKVTensorPool.Lock()
		bucket := rocmDeviceKVTensorPool.entries[sizeBytes]
		if bucket.len() < rocmDeviceKVTensorPoolMaxPerSize &&
			rocmDeviceKVTensorPool.bytes+sizeBytes <= rocmDeviceKVTensorPoolMaxBytes {
			entry := rocmDeviceKVTensorPoolEntry{driver: driver, pointer: pointer}
			if bucket.first.pointer == 0 {
				bucket.first = entry
			} else {
				if bucket.rest == nil {
					bucket.rest = make([]rocmDeviceKVTensorPoolEntry, 0, 8)
				}
				bucket.rest = append(bucket.rest, entry)
			}
			rocmDeviceKVTensorPool.entries[sizeBytes] = bucket
			rocmDeviceKVTensorPool.bytes += sizeBytes
			rocmDeviceKVTensorPool.Unlock()
			return nil
		}
		rocmDeviceKVTensorPool.Unlock()
	}
	return driver.Free(pointer)
}

func rocmPrewarmDeviceKVTensorPool(driver nativeHIPDriver, sizeBytes uint64, count int) {
	if driver == nil || !driver.Available() || sizeBytes == 0 || count <= 0 {
		return
	}
	for i := 0; i < count; i++ {
		pointer, err := driver.Malloc(sizeBytes)
		if err != nil {
			return
		}
		if err := rocmDeviceKVTensorFree(driver, pointer, sizeBytes); err != nil {
			_ = driver.Free(pointer)
			return
		}
	}
}

func rocmDeviceKVAllocateEncodedTensorPair(driver nativeHIPDriver, keyBytes, valueBytes uint64, keyEncoding, valueEncoding string) (rocmDeviceKVTensor, rocmDeviceKVTensor, error) {
	if driver == nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if keyBytes == 0 || valueBytes == 0 {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "encoded KV tensor sizes must be positive", nil)
	}
	if valueBytes > ^uint64(0)-keyBytes {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "encoded KV tensor allocation size overflow", nil)
	}
	allocationBytes := keyBytes + valueBytes
	pointer, err := rocmDeviceKVTensorMalloc(driver, allocationBytes)
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "allocate encoded KV token pair", err)
	}
	key := rocmDeviceKVTensor{
		pointer:           pointer,
		sizeBytes:         keyBytes,
		encoding:          keyEncoding,
		allocationPointer: pointer,
		allocationBytes:   allocationBytes,
	}
	value := rocmDeviceKVTensor{
		pointer:           pointer + nativeDevicePointer(keyBytes),
		sizeBytes:         valueBytes,
		encoding:          valueEncoding,
		allocationPointer: pointer,
		allocationBytes:   allocationBytes,
	}
	return key, value, nil
}

func rocmKVInterleavedEncodingsForMode(mode string) (string, string, bool) {
	switch mode {
	case rocmKVCacheModeKQ8VQ4:
		return rocmKVEncodingQ8RowsI, rocmKVEncodingQ4RowsI, true
	default:
		return "", "", false
	}
}

func rocmKVInterleavedRowStride(encoding string, width int) (uint64, error) {
	if width <= 0 {
		return 0, core.E("rocm.KVCache.DeviceAppend", "interleaved KV row width must be positive", nil)
	}
	switch encoding {
	case rocmKVEncodingQ8RowsI:
		return uint64(4 + width), nil
	case rocmKVEncodingQ4RowsI:
		return uint64(4 + (width+1)/2), nil
	default:
		return 0, core.E("rocm.KVCache.DeviceAppend", core.Sprintf("unsupported interleaved KV encoding %q", encoding), nil)
	}
}

func rocmDeviceKVAllocateInterleavedTensorPair(driver nativeHIPDriver, keyWidth, valueWidth, capacity int, keyEncoding, valueEncoding string) (rocmDeviceKVTensor, rocmDeviceKVTensor, uint64, uint64, error) {
	if driver == nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, 0, 0, core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if keyWidth <= 0 || valueWidth <= 0 || capacity <= 0 {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, 0, 0, core.E("rocm.KVCache.DeviceAppend", "interleaved KV dimensions must be positive", nil)
	}
	keyStride, err := rocmKVInterleavedRowStride(keyEncoding, keyWidth)
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, 0, 0, err
	}
	valueStride, err := rocmKVInterleavedRowStride(valueEncoding, valueWidth)
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, 0, 0, err
	}
	keyCapacityBytes := keyStride * uint64(capacity)
	valueCapacityBytes := valueStride * uint64(capacity)
	if valueCapacityBytes > ^uint64(0)-keyCapacityBytes {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, 0, 0, core.E("rocm.KVCache.DeviceAppend", "interleaved KV tensor allocation size overflow", nil)
	}
	allocationBytes := keyCapacityBytes + valueCapacityBytes
	pointer, err := rocmDeviceKVTensorMalloc(driver, allocationBytes)
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, 0, 0, core.E("rocm.KVCache.DeviceAppend", "allocate interleaved KV page pair", err)
	}
	key := rocmDeviceKVTensor{
		pointer:           pointer,
		sizeBytes:         keyStride,
		encoding:          keyEncoding,
		allocationPointer: pointer,
		allocationBytes:   allocationBytes,
	}
	value := rocmDeviceKVTensor{
		pointer:           pointer + nativeDevicePointer(keyCapacityBytes),
		sizeBytes:         valueStride,
		encoding:          valueEncoding,
		allocationPointer: pointer,
		allocationBytes:   allocationBytes,
	}
	return key, value, keyStride, valueStride, nil
}

func rocmDeviceKVTensorAllocation(tensor rocmDeviceKVTensor) (nativeDevicePointer, uint64) {
	if tensor.allocationPointer != 0 && tensor.allocationBytes > 0 {
		return tensor.allocationPointer, tensor.allocationBytes
	}
	return tensor.pointer, tensor.sizeBytes
}

func rocmDeviceKVTensorsShareAllocation(key, value rocmDeviceKVTensor) bool {
	keyPointer, keyBytes := rocmDeviceKVTensorAllocation(key)
	valuePointer, valueBytes := rocmDeviceKVTensorAllocation(value)
	return key.allocationPointer != 0 &&
		value.allocationPointer != 0 &&
		keyPointer == valuePointer &&
		keyBytes == valueBytes
}

func rocmDeviceKVTensorFreeTensor(driver nativeHIPDriver, tensor rocmDeviceKVTensor) error {
	pointer, sizeBytes := rocmDeviceKVTensorAllocation(tensor)
	if pointer == 0 || sizeBytes == 0 {
		return nil
	}
	return rocmDeviceKVTensorFree(driver, pointer, sizeBytes)
}

func rocmDeviceKVTensorFreePair(driver nativeHIPDriver, key, value rocmDeviceKVTensor) error {
	if rocmDeviceKVTensorsShareAllocation(key, value) {
		return rocmDeviceKVTensorFreeTensor(driver, key)
	}
	var lastErr error
	if err := rocmDeviceKVTensorFreeTensor(driver, key); err != nil {
		lastErr = err
	}
	if err := rocmDeviceKVTensorFreeTensor(driver, value); err != nil {
		lastErr = err
	}
	return lastErr
}

func (cache *rocmKVCache) MirrorToDevice(driver nativeHIPDriver) (*rocmDeviceKVCache, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.DeviceMirror", "cache is nil", nil)
	}
	if driver == nil {
		return nil, core.E("rocm.KVCache.DeviceMirror", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceMirror", "HIP driver is not available", nil)
	}
	if len(cache.blocks) == 0 {
		return nil, core.E("rocm.KVCache.DeviceMirror", "cache has no pages", nil)
	}
	device := &rocmDeviceKVCache{
		driver:     driver,
		mode:       cache.mode,
		blockSize:  cache.blockSize,
		tokenCount: cache.TokenCount(),
		pages:      make([]rocmDeviceKVPage, 0, len(cache.blocks)),
	}
	for _, block := range cache.blocks {
		page := rocmDeviceKVPage{
			tokenStart: block.tokenStart,
			tokenCount: block.tokenCount,
			keyWidth:   block.keyWidth,
			valueWidth: block.valueWidth,
			owned:      true,
		}
		key, err := mirrorROCmKVTensorToDevice(driver, block.key)
		if err != nil {
			_ = device.Close()
			return nil, core.E("rocm.KVCache.DeviceMirror", "copy KV key page", err)
		}
		page.key = key
		value, err := mirrorROCmKVTensorToDevice(driver, block.value)
		if err != nil {
			_ = rocmDeviceKVTensorFree(driver, key.pointer, key.sizeBytes)
			_ = device.Close()
			return nil, core.E("rocm.KVCache.DeviceMirror", "copy KV value page", err)
		}
		page.value = value
		device.pages = append(device.pages, page)
	}
	return device, nil
}

func mirrorROCmKVTensorToDevice(driver nativeHIPDriver, tensor rocmKVEncodedTensor) (rocmDeviceKVTensor, error) {
	payload, err := tensor.deviceBytes()
	if err != nil {
		return rocmDeviceKVTensor{}, err
	}
	return mirrorROCmKVPayloadToDevice(driver, tensor.encoding, payload)
}

func mirrorROCmKVValuesToDevice(driver nativeHIPDriver, encoding string, values []float32) (rocmDeviceKVTensor, error) {
	payload, err := encodeROCmKVValuesDeviceBytes(encoding, values)
	if err != nil {
		return rocmDeviceKVTensor{}, err
	}
	defer rocmDeviceKVReleasePayloadBytes(payload)
	return mirrorROCmKVPayloadToDevice(driver, encoding, payload)
}

func mirrorROCmKVPayloadToDevice(driver nativeHIPDriver, encoding string, payload []byte) (rocmDeviceKVTensor, error) {
	pointer, err := rocmDeviceKVTensorMalloc(driver, uint64(len(payload)))
	if err != nil {
		return rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceMirror", "allocate KV tensor", err)
	}
	if err := hipCopyPinnedHostToDevice(driver, pointer, payload); err != nil {
		_ = rocmDeviceKVTensorFree(driver, pointer, uint64(len(payload)))
		return rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceMirror", "copy KV tensor", err)
	}
	return rocmDeviceKVTensor{pointer: pointer, sizeBytes: uint64(len(payload)), encoding: encoding}, nil
}

func rocmDeviceKVPageFromRawPayload(driver nativeHIPDriver, payload []byte) (rocmDeviceKVPage, error) {
	if driver == nil {
		return rocmDeviceKVPage{}, core.E("rocm.KVCache.DeviceRestore", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return rocmDeviceKVPage{}, core.E("rocm.KVCache.DeviceRestore", "HIP driver is not available", nil)
	}
	meta, keyPayload, valuePayload, err := rocmKVBlockRawPayloadParts(payload)
	if err != nil {
		return rocmDeviceKVPage{}, err
	}
	keyPointer, err := rocmDeviceKVTensorMalloc(driver, uint64(len(keyPayload)))
	if err != nil {
		return rocmDeviceKVPage{}, core.E("rocm.KVCache.DeviceRestore", "allocate KV key page", err)
	}
	if err := hipCopyPinnedHostToDevice(driver, keyPointer, keyPayload); err != nil {
		_ = rocmDeviceKVTensorFree(driver, keyPointer, uint64(len(keyPayload)))
		return rocmDeviceKVPage{}, core.E("rocm.KVCache.DeviceRestore", "copy KV key page", err)
	}
	valuePointer, err := rocmDeviceKVTensorMalloc(driver, uint64(len(valuePayload)))
	if err != nil {
		_ = rocmDeviceKVTensorFree(driver, keyPointer, uint64(len(keyPayload)))
		return rocmDeviceKVPage{}, core.E("rocm.KVCache.DeviceRestore", "allocate KV value page", err)
	}
	if err := hipCopyPinnedHostToDevice(driver, valuePointer, valuePayload); err != nil {
		_ = rocmDeviceKVTensorFree(driver, valuePointer, uint64(len(valuePayload)))
		_ = rocmDeviceKVTensorFree(driver, keyPointer, uint64(len(keyPayload)))
		return rocmDeviceKVPage{}, core.E("rocm.KVCache.DeviceRestore", "copy KV value page", err)
	}
	return rocmDeviceKVPage{
		tokenStart: meta.tokenStart,
		tokenCount: meta.tokenCount,
		keyWidth:   meta.keyWidth,
		valueWidth: meta.valueWidth,
		key: rocmDeviceKVTensor{
			pointer:   keyPointer,
			sizeBytes: uint64(len(keyPayload)),
			encoding:  meta.keyEncoding,
		},
		value: rocmDeviceKVTensor{
			pointer:   valuePointer,
			sizeBytes: uint64(len(valuePayload)),
			encoding:  meta.valueEncoding,
		},
		owned: true,
	}, nil
}

func (cache *rocmDeviceKVCache) withAppendedToken(key, value []float32) (*rocmDeviceKVCache, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache is nil", nil)
	}
	if cache.closed {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache is closed", nil)
	}
	if cache.driver == nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !cache.driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	keyWidth, valueWidth, ok := cache.LastVectorWidths()
	if !ok {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache has no pages", nil)
	}
	if len(key) != keyWidth || len(value) != valueWidth {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV vector widths must match device cache shape", nil)
	}
	keyEncoding, valueEncoding := rocmKVEncodingsForMode(cache.mode)
	deviceKey, err := mirrorROCmKVValuesToDevice(cache.driver, keyEncoding, key)
	if err != nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "copy KV key page", err)
	}
	deviceValue, err := mirrorROCmKVValuesToDevice(cache.driver, valueEncoding, value)
	if err != nil {
		_ = rocmDeviceKVTensorFree(cache.driver, deviceKey.pointer, deviceKey.sizeBytes)
		return nil, core.E("rocm.KVCache.DeviceAppend", "copy KV value page", err)
	}
	tokenStart := cache.TokenCount()
	next := rocmBorrowDeviceKVCache(cache.driver, cache.mode, cache.blockSize, tokenStart+1, rocmDeviceKVCopyPagesWithExtra(cache.pages, 1), false)
	for index := range next.pages {
		next.pages[index].owned = false
	}
	next.pages = append(next.pages, rocmDeviceKVPage{
		tokenStart: tokenStart,
		tokenCount: 1,
		keyWidth:   keyWidth,
		valueWidth: valueWidth,
		key:        deviceKey,
		value:      deviceValue,
		owned:      true,
	})
	return next, nil
}

func (cache *rocmDeviceKVCache) withAppendedTokenWindow(key, value []float32, window int) (*rocmDeviceKVCache, error) {
	next, err := cache.withAppendedToken(key, value)
	if err != nil {
		return nil, err
	}
	if window <= 0 || next.TokenCount() <= window {
		return next, nil
	}
	oldTokenCount := next.TokenCount()
	trimStart := oldTokenCount - window
	trimmed := rocmDeviceKVBorrowPageSlice(0, len(next.pages))
	for _, page := range next.pages {
		pageEnd := page.tokenStart + page.tokenCount
		if pageEnd <= trimStart {
			continue
		}
		if page.tokenStart < trimStart {
			// The hot Gemma4 generation path appends one-token pages. If a
			// multi-token page straddles the window boundary, keep the untrimmed
			// cache rather than making a descriptor that points into the middle of
			// an encoded page we cannot slice safely.
			rocmDeviceKVReleasePageSlice(trimmed)
			return next, nil
		}
		page.tokenStart -= trimStart
		trimmed = append(trimmed, page)
	}
	if len(trimmed) == 0 {
		rocmDeviceKVReleasePageSlice(trimmed)
		return next, nil
	}
	pages := next.pages
	next.pages = trimmed
	next.tokenCount = oldTokenCount - trimStart
	rocmDeviceKVReleasePageSlice(pages)
	return next, nil
}

func (cache *rocmDeviceKVCache) withAppendedDeviceTokenWindow(ctx context.Context, key, value *hipDeviceByteBuffer, window int) (*rocmDeviceKVCache, error) {
	return cache.withAppendedDeviceTokenWindowWithWorkspace(ctx, key, value, window, nil)
}

func (cache *rocmDeviceKVCache) withAppendedDeviceTokenWindowWithWorkspace(ctx context.Context, key, value *hipDeviceByteBuffer, window int, workspace *hipAttentionHeadsChunkedWorkspace) (*rocmDeviceKVCache, error) {
	return cache.withAppendedDeviceTokenWindowWithWorkspaceAndEngineConfig(ctx, key, value, window, workspace, defaultHIPGemma4Q4EngineConfig())
}

func (cache *rocmDeviceKVCache) withAppendedDeviceTokenWindowWithWorkspaceAndEngineConfig(ctx context.Context, key, value *hipDeviceByteBuffer, window int, workspace *hipAttentionHeadsChunkedWorkspace, engineConfig hipGemma4Q4EngineConfig) (*rocmDeviceKVCache, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache is nil", nil)
	}
	if cache.closed {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache is closed", nil)
	}
	keyWidth, valueWidth, ok := cache.LastVectorWidths()
	if !ok {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache has no pages", nil)
	}
	if next, ok, err := cache.withAppendedDeviceTokenInterleavedBlockWithWorkspaceAndEngineConfig(ctx, key, value, keyWidth, valueWidth, window, workspace, engineConfig); ok || err != nil {
		return next, err
	}
	encodedKey, encodedValue, err := hipRunKVEncodeTokenKernelWithWorkspace(ctx, cache.driver, key, value, cache.mode, workspace)
	if err != nil {
		return nil, err
	}
	next, err := cache.withAppendedEncodedTokenWindow(encodedKey, encodedValue, keyWidth, valueWidth, window)
	if err != nil {
		_ = rocmDeviceKVTensorFreePair(cache.driver, encodedKey, encodedValue)
		return nil, err
	}
	return next, nil
}

func (cache *rocmDeviceKVCache) withAppendedDeviceTokenInterleavedBlock(ctx context.Context, key, value *hipDeviceByteBuffer, keyWidth, valueWidth, window int) (*rocmDeviceKVCache, bool, error) {
	return cache.withAppendedDeviceTokenInterleavedBlockWithWorkspace(ctx, key, value, keyWidth, valueWidth, window, nil)
}

func (cache *rocmDeviceKVCache) withAppendedDeviceTokenInterleavedBlockWithWorkspace(ctx context.Context, key, value *hipDeviceByteBuffer, keyWidth, valueWidth, window int, workspace *hipAttentionHeadsChunkedWorkspace) (*rocmDeviceKVCache, bool, error) {
	return cache.withAppendedDeviceTokenInterleavedBlockWithWorkspaceAndEngineConfig(ctx, key, value, keyWidth, valueWidth, window, workspace, defaultHIPGemma4Q4EngineConfig())
}

func (cache *rocmDeviceKVCache) withAppendedDeviceTokenInterleavedBlockWithWorkspaceAndEngineConfig(ctx context.Context, key, value *hipDeviceByteBuffer, keyWidth, valueWidth, window int, workspace *hipAttentionHeadsChunkedWorkspace, engineConfig hipGemma4Q4EngineConfig) (*rocmDeviceKVCache, bool, error) {
	if cache == nil || cache.blockSize <= 1 {
		return nil, false, nil
	}
	keyEncoding, valueEncoding, ok := rocmKVInterleavedEncodingsForMode(cache.mode)
	if !ok {
		return nil, false, nil
	}
	if key == nil || value == nil || key.Pointer() == 0 || value.Pointer() == 0 || key.Count() != keyWidth || value.Count() != valueWidth {
		return nil, false, core.E("rocm.KVCache.DeviceAppend", "device KV token buffers must match cache widths", nil)
	}
	tokenStart := cache.TokenCount()
	keyStride, err := rocmKVInterleavedRowStride(keyEncoding, keyWidth)
	if err != nil {
		return nil, false, err
	}
	valueStride, err := rocmKVInterleavedRowStride(valueEncoding, valueWidth)
	if err != nil {
		return nil, false, err
	}
	if len(cache.pages) > 0 {
		last := cache.pages[len(cache.pages)-1]
		if last.tokenStart+last.tokenCount == tokenStart &&
			last.tokenCount > 0 && last.tokenCount < cache.blockSize &&
			last.keyWidth == keyWidth && last.valueWidth == valueWidth &&
			last.key.encoding == keyEncoding && last.value.encoding == valueEncoding &&
			rocmDeviceKVInterleavedPageHasCapacity(last, keyStride, valueStride, cache.blockSize) {
			rowOffset := last.tokenCount
			keyOutputPointer := last.key.pointer + nativeDevicePointer(keyStride*uint64(rowOffset))
			valueOutputPointer := last.value.pointer + nativeDevicePointer(valueStride*uint64(rowOffset))
			if err := hipRunKVEncodeRowsKernelIntoWithWorkspace(ctx, cache.driver, key, value, keyWidth, valueWidth, 1, keyOutputPointer, valueOutputPointer, keyStride, valueStride, keyEncoding, valueEncoding, workspace); err != nil {
				return nil, true, err
			}
			pages := rocmDeviceKVCopyPagesWithExtra(cache.pages, 0)
			for index := range pages {
				pages[index].owned = false
			}
			pages[len(pages)-1].tokenCount++
			pages[len(pages)-1].key.sizeBytes += keyStride
			pages[len(pages)-1].value.sizeBytes += valueStride
			next := rocmBorrowDeviceKVCache(cache.driver, cache.mode, cache.blockSize, tokenStart+1, pages, false)
			if window > 0 && next.TokenCount() > window {
				next = next.trimDeviceTokenWindowForAppendWithEngineConfig(window, engineConfig)
			}
			return next, true, nil
		}
	}
	deviceKey, deviceValue, keyStride, valueStride, err := rocmDeviceKVAllocateInterleavedTensorPair(cache.driver, keyWidth, valueWidth, cache.blockSize, keyEncoding, valueEncoding)
	if err != nil {
		return nil, true, err
	}
	if err := hipRunKVEncodeRowsKernelIntoWithWorkspace(ctx, cache.driver, key, value, keyWidth, valueWidth, 1, deviceKey.pointer, deviceValue.pointer, keyStride, valueStride, keyEncoding, valueEncoding, workspace); err != nil {
		_ = rocmDeviceKVTensorFreePair(cache.driver, deviceKey, deviceValue)
		return nil, true, err
	}
	next := rocmBorrowDeviceKVCache(cache.driver, cache.mode, cache.blockSize, tokenStart+1, rocmDeviceKVCopyPagesWithExtra(cache.pages, 1), false)
	for index := range next.pages {
		next.pages[index].owned = false
	}
	next.pages = append(next.pages, rocmDeviceKVPage{
		tokenStart: tokenStart,
		tokenCount: 1,
		keyWidth:   keyWidth,
		valueWidth: valueWidth,
		key:        deviceKey,
		value:      deviceValue,
		owned:      true,
	})
	if window > 0 && next.TokenCount() > window {
		next = next.trimDeviceTokenWindowForAppendWithEngineConfig(window, engineConfig)
	}
	return next, true, nil
}

func rocmDeviceKVInterleavedPageHasCapacity(page rocmDeviceKVPage, keyStride, valueStride uint64, blockSize int) bool {
	return rocmDeviceKVInterleavedPageHasRowCapacity(page, keyStride, valueStride, blockSize, 1)
}

func rocmDeviceKVInterleavedPageHasRowCapacity(page rocmDeviceKVPage, keyStride, valueStride uint64, blockSize, appendRows int) bool {
	if blockSize <= 0 || keyStride == 0 || valueStride == 0 || page.key.pointer == 0 || page.value.pointer == 0 {
		return false
	}
	if appendRows <= 0 || page.tokenCount < 0 || page.tokenCount+appendRows > blockSize {
		return false
	}
	appendKeyBytes := keyStride * uint64(appendRows)
	appendValueBytes := valueStride * uint64(appendRows)
	neededKeyBytes := keyStride * uint64(page.tokenCount+appendRows)
	neededValueBytes := valueStride * uint64(page.tokenCount+appendRows)
	if page.key.sizeBytes+appendKeyBytes != neededKeyBytes || page.value.sizeBytes+appendValueBytes != neededValueBytes {
		return false
	}
	if !rocmDeviceKVTensorsShareAllocation(page.key, page.value) || page.value.pointer <= page.key.pointer {
		return page.key.allocationBytes >= keyStride*uint64(blockSize) && page.value.allocationBytes >= valueStride*uint64(blockSize)
	}
	keyCapacity := uint64(page.value.pointer - page.key.pointer)
	allocationEnd := page.key.allocationPointer + nativeDevicePointer(page.key.allocationBytes)
	if allocationEnd <= page.value.pointer {
		return false
	}
	valueCapacity := uint64(allocationEnd - page.value.pointer)
	return keyCapacity >= keyStride*uint64(blockSize) && valueCapacity >= valueStride*uint64(blockSize)
}

func (cache *rocmDeviceKVCache) withAppendedDeviceRowsWindow(ctx context.Context, key, value *hipDeviceByteBuffer, keyWidth, valueWidth, tokenCount, window int) (*rocmDeviceKVCache, error) {
	return cache.withAppendedDeviceRowsWindowWithEngineConfig(ctx, key, value, keyWidth, valueWidth, tokenCount, window, defaultHIPGemma4Q4EngineConfig())
}

func (cache *rocmDeviceKVCache) withAppendedDeviceRowsWindowWithEngineConfig(ctx context.Context, key, value *hipDeviceByteBuffer, keyWidth, valueWidth, tokenCount, window int, engineConfig hipGemma4Q4EngineConfig) (*rocmDeviceKVCache, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache is nil", nil)
	}
	if cache.closed {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache is closed", nil)
	}
	if cache.driver == nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !cache.driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	if key == nil || key.Pointer() == 0 || value == nil || value.Pointer() == 0 {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV row buffers are required", nil)
	}
	if keyWidth <= 0 || valueWidth <= 0 || tokenCount <= 0 {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV row widths and token count must be positive", nil)
	}
	if key.Count() != keyWidth*tokenCount || value.Count() != valueWidth*tokenCount ||
		key.SizeBytes() != uint64(key.Count()*4) || value.SizeBytes() != uint64(value.Count()*4) {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV row buffer shape mismatch", nil)
	}
	if priorKeyWidth, priorValueWidth, ok := cache.LastVectorWidths(); ok && (priorKeyWidth != keyWidth || priorValueWidth != valueWidth) {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV row widths must match device cache shape", nil)
	}
	if next, ok, err := cache.withAppendedDeviceRowsInterleavedBlockWithEngineConfig(ctx, key, value, keyWidth, valueWidth, tokenCount, window, engineConfig); ok || err != nil {
		return next, err
	}
	mode := firstNonEmptyString(cache.mode, rocmKVCacheModeFP16)
	if !isROCmKVCacheMode(mode) {
		return nil, core.E("rocm.KVCache.DeviceAppend", core.Sprintf("unsupported cache mode %q", mode), nil)
	}
	blockSize := cache.blockSize
	if blockSize <= 0 {
		blockSize = defaultROCmKVBlockSize
	}
	pageCount := (tokenCount + blockSize - 1) / blockSize
	tokenStart := cache.TokenCount()
	next := rocmBorrowDeviceKVCache(cache.driver, mode, blockSize, tokenStart+tokenCount, rocmDeviceKVCopyPagesWithExtra(cache.pages, pageCount), false)
	for index := range next.pages {
		next.pages[index].owned = false
	}
	basePageCount := len(cache.pages)
	success := false
	defer func() {
		if !success {
			_ = next.closePagesFrom(basePageCount)
		}
	}()
	for tokenOffset := 0; tokenOffset < tokenCount; tokenOffset += blockSize {
		tokenEnd := tokenOffset + blockSize
		if tokenEnd > tokenCount {
			tokenEnd = tokenCount
		}
		pageTokens := tokenEnd - tokenOffset
		keyCount := pageTokens * keyWidth
		valueCount := pageTokens * valueWidth
		keyByteOffset := nativeDevicePointer(tokenOffset * keyWidth * 4)
		valueByteOffset := nativeDevicePointer(tokenOffset * valueWidth * 4)
		keyPage := hipBorrowDeviceByteBufferValue(cache.driver, "device KV key row page", key.Pointer()+keyByteOffset, uint64(keyCount*4), keyCount)
		valuePage := hipBorrowDeviceByteBufferValue(cache.driver, "device KV value row page", value.Pointer()+valueByteOffset, uint64(valueCount*4), valueCount)
		encodedKey, encodedValue, err := rocmDeviceKVCacheEncodeDeviceRowsPageWithEngineConfig(ctx, cache.driver, mode, blockSize, &keyPage, &valuePage, keyWidth, valueWidth, pageTokens, engineConfig)
		if err != nil {
			return nil, err
		}
		next.pages = append(next.pages, rocmDeviceKVPage{
			tokenStart: tokenStart + tokenOffset,
			tokenCount: pageTokens,
			keyWidth:   keyWidth,
			valueWidth: valueWidth,
			key:        encodedKey,
			value:      encodedValue,
			owned:      true,
		})
	}
	success = true
	if window > 0 && next.TokenCount() > window {
		return next.trimDeviceTokenWindowForAppendWithEngineConfig(window, engineConfig), nil
	}
	return next, nil
}

func (cache *rocmDeviceKVCache) withAppendedDeviceRowsValueNormWindowWithWorkspaceAndEngineConfig(ctx context.Context, key, rawValue *hipDeviceByteBuffer, keyWidth, valueWidth, tokenCount int, valueHeadDim, valueHeadCount int, valueEpsilon float32, window int, workspace *hipAttentionHeadsChunkedWorkspace, engineConfig hipGemma4Q4EngineConfig) (*rocmDeviceKVCache, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache is nil", nil)
	}
	if cache.closed {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache is closed", nil)
	}
	if cache.driver == nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !cache.driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	if tokenCount != 1 {
		return nil, core.E("rocm.KVCache.DeviceAppend", "fused value norm KV append currently supports one token", nil)
	}
	if key == nil || key.Pointer() == 0 || rawValue == nil || rawValue.Pointer() == 0 {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV row buffers are required", nil)
	}
	if keyWidth <= 0 || valueWidth <= 0 || key.Count() != keyWidth || rawValue.Count() != valueWidth ||
		key.SizeBytes() != uint64(key.Count()*4) || rawValue.SizeBytes() != uint64(rawValue.Count()*4) {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV row buffer shape mismatch", nil)
	}
	if valueHeadDim <= 0 || valueHeadCount <= 0 || valueHeadCount > hipKVEncodeTokenValueNormMaxHeads || valueWidth != valueHeadDim*valueHeadCount {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV value norm head shape mismatch", nil)
	}
	if priorKeyWidth, priorValueWidth, ok := cache.LastVectorWidths(); ok && (priorKeyWidth != keyWidth || priorValueWidth != valueWidth) {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV row widths must match device cache shape", nil)
	}
	if next, ok, err := cache.withAppendedDeviceRowsValueNormInterleavedBlockWithWorkspaceAndEngineConfig(ctx, key, rawValue, keyWidth, valueWidth, tokenCount, valueHeadDim, valueHeadCount, valueEpsilon, window, workspace, engineConfig); ok || err != nil {
		return next, err
	}
	mode := firstNonEmptyString(cache.mode, rocmKVCacheModeFP16)
	if !isROCmKVCacheMode(mode) {
		return nil, core.E("rocm.KVCache.DeviceAppend", core.Sprintf("unsupported cache mode %q", mode), nil)
	}
	encodedKey, encodedValue, err := hipRunKVEncodeRowsValueNormKernelWithWorkspace(ctx, cache.driver, key, rawValue, keyWidth, valueWidth, tokenCount, mode, valueHeadDim, valueHeadCount, valueEpsilon, workspace)
	if err != nil {
		return nil, err
	}
	next, err := cache.withAppendedEncodedTokenWindow(encodedKey, encodedValue, keyWidth, valueWidth, window)
	if err != nil {
		_ = rocmDeviceKVTensorFreePair(cache.driver, encodedKey, encodedValue)
		return nil, err
	}
	return next, nil
}

func (cache *rocmDeviceKVCache) withAppendedDeviceRowsValueNormInterleavedBlockWithWorkspaceAndEngineConfig(ctx context.Context, key, rawValue *hipDeviceByteBuffer, keyWidth, valueWidth, tokenCount int, valueHeadDim, valueHeadCount int, valueEpsilon float32, window int, workspace *hipAttentionHeadsChunkedWorkspace, engineConfig hipGemma4Q4EngineConfig) (*rocmDeviceKVCache, bool, error) {
	if cache == nil || cache.blockSize <= 1 || tokenCount != 1 || !engineConfig.interleavedRowPagesEnabled() {
		return nil, false, nil
	}
	keyEncoding, valueEncoding, ok := rocmKVInterleavedEncodingsForMode(cache.mode)
	if !ok {
		return nil, false, nil
	}
	if key == nil || rawValue == nil || key.Pointer() == 0 || rawValue.Pointer() == 0 ||
		key.Count() != keyWidth || rawValue.Count() != valueWidth {
		return nil, false, core.E("rocm.KVCache.DeviceAppend", "device KV row buffers must match cache widths", nil)
	}
	keyStride, err := rocmKVInterleavedRowStride(keyEncoding, keyWidth)
	if err != nil {
		return nil, true, err
	}
	valueStride, err := rocmKVInterleavedRowStride(valueEncoding, valueWidth)
	if err != nil {
		return nil, true, err
	}
	tokenStart := cache.TokenCount()
	if len(cache.pages) > 0 {
		last := cache.pages[len(cache.pages)-1]
		if last.tokenStart+last.tokenCount == tokenStart &&
			last.tokenCount > 0 && last.tokenCount < cache.blockSize &&
			last.keyWidth == keyWidth && last.valueWidth == valueWidth &&
			last.key.encoding == keyEncoding && last.value.encoding == valueEncoding &&
			rocmDeviceKVInterleavedPageHasRowCapacity(last, keyStride, valueStride, cache.blockSize, tokenCount) {
			rowOffset := last.tokenCount
			keyOutputPointer := last.key.pointer + nativeDevicePointer(keyStride*uint64(rowOffset))
			valueOutputPointer := last.value.pointer + nativeDevicePointer(valueStride*uint64(rowOffset))
			if err := hipRunKVEncodeRowsValueNormKernelIntoWithWorkspace(ctx, cache.driver, key, rawValue, keyWidth, valueWidth, tokenCount, valueHeadDim, valueHeadCount, valueEpsilon, keyOutputPointer, valueOutputPointer, keyStride*uint64(tokenCount), valueStride*uint64(tokenCount), keyEncoding, valueEncoding, workspace); err != nil {
				return nil, true, err
			}
			pages := rocmDeviceKVCopyPagesWithExtra(cache.pages, 0)
			for index := range pages {
				pages[index].owned = false
			}
			pages[len(pages)-1].tokenCount += tokenCount
			pages[len(pages)-1].key.sizeBytes += keyStride * uint64(tokenCount)
			pages[len(pages)-1].value.sizeBytes += valueStride * uint64(tokenCount)
			next := rocmBorrowDeviceKVCache(cache.driver, cache.mode, cache.blockSize, tokenStart+tokenCount, pages, false)
			if window > 0 && next.TokenCount() > window {
				next = next.trimDeviceTokenWindowForAppendWithEngineConfig(window, engineConfig)
			}
			return next, true, nil
		}
	}
	deviceKey, deviceValue, keyStride, valueStride, err := rocmDeviceKVAllocateInterleavedTensorPair(cache.driver, keyWidth, valueWidth, cache.blockSize, keyEncoding, valueEncoding)
	if err != nil {
		return nil, true, err
	}
	keyBytes := keyStride * uint64(tokenCount)
	valueBytes := valueStride * uint64(tokenCount)
	if err := hipRunKVEncodeRowsValueNormKernelIntoWithWorkspace(ctx, cache.driver, key, rawValue, keyWidth, valueWidth, tokenCount, valueHeadDim, valueHeadCount, valueEpsilon, deviceKey.pointer, deviceValue.pointer, keyBytes, valueBytes, keyEncoding, valueEncoding, workspace); err != nil {
		_ = rocmDeviceKVTensorFreePair(cache.driver, deviceKey, deviceValue)
		return nil, true, err
	}
	deviceKey.sizeBytes = keyBytes
	deviceValue.sizeBytes = valueBytes
	pages := rocmDeviceKVCopyPagesWithExtra(cache.pages, 1)
	for index := range pages {
		pages[index].owned = false
	}
	pages = append(pages, rocmDeviceKVPage{
		tokenStart: tokenStart,
		tokenCount: tokenCount,
		keyWidth:   keyWidth,
		valueWidth: valueWidth,
		key:        deviceKey,
		value:      deviceValue,
		owned:      true,
	})
	next := rocmBorrowDeviceKVCache(cache.driver, cache.mode, cache.blockSize, tokenStart+tokenCount, pages, false)
	if window > 0 && next.TokenCount() > window {
		next = next.trimDeviceTokenWindowForAppendWithEngineConfig(window, engineConfig)
	}
	return next, true, nil
}

func (cache *rocmDeviceKVCache) withAppendedDeviceRowsValueNormDescriptorGrowLastPageWithWorkspaceAndEngineConfig(ctx context.Context, key, rawValue *hipDeviceByteBuffer, previousTable *rocmDeviceKVDescriptorTable, keyWidth, valueWidth, tokenCount int, valueHeadDim, valueHeadCount int, valueEpsilon float32, window int, workspace *hipAttentionHeadsChunkedWorkspace, engineConfig hipGemma4Q4EngineConfig) (*rocmDeviceKVCache, *rocmDeviceKVDescriptorTable, bool, error) {
	if cache == nil || previousTable == nil || workspace == nil || cache.blockSize <= 1 || tokenCount != 1 || !engineConfig.interleavedRowPagesEnabled() {
		return nil, nil, false, nil
	}
	if err := hipContextErr(ctx); err != nil {
		return nil, nil, true, err
	}
	if cache.closed {
		return nil, nil, true, core.E("rocm.KVCache.DeviceAppend", "device KV cache is closed", nil)
	}
	if cache.driver == nil || !cache.driver.Available() {
		return nil, nil, true, core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	if previousTable.closed {
		return nil, nil, true, core.E("rocm.KVCache.DeviceDescriptor", "previous descriptor table is closed", nil)
	}
	if err := previousTable.CompatibleWith(cache); err != nil {
		return nil, nil, true, core.E("rocm.KVCache.DeviceDescriptor", "previous descriptor table does not match device KV cache", err)
	}
	if window > 0 && cache.TokenCount()+tokenCount > window {
		return nil, nil, false, nil
	}
	keyEncoding, valueEncoding, ok := rocmKVInterleavedEncodingsForMode(cache.mode)
	if !ok {
		return nil, nil, false, nil
	}
	if key == nil || rawValue == nil || key.Pointer() == 0 || rawValue.Pointer() == 0 ||
		key.Count() != keyWidth || rawValue.Count() != valueWidth ||
		key.SizeBytes() != uint64(key.Count()*4) || rawValue.SizeBytes() != uint64(rawValue.Count()*4) {
		return nil, nil, true, core.E("rocm.KVCache.DeviceAppend", "device KV row buffers must match cache widths", nil)
	}
	if valueHeadDim <= 0 || valueHeadCount <= 0 || valueHeadCount > hipKVEncodeTokenValueNormMaxHeads || valueWidth != valueHeadDim*valueHeadCount {
		return nil, nil, true, core.E("rocm.KVCache.DeviceAppend", "device KV value norm head shape mismatch", nil)
	}
	if priorKeyWidth, priorValueWidth, ok := cache.LastVectorWidths(); ok && (priorKeyWidth != keyWidth || priorValueWidth != valueWidth) {
		return nil, nil, true, core.E("rocm.KVCache.DeviceAppend", "KV row widths must match device cache shape", nil)
	}
	keyStride, err := rocmKVInterleavedRowStride(keyEncoding, keyWidth)
	if err != nil {
		return nil, nil, true, err
	}
	valueStride, err := rocmKVInterleavedRowStride(valueEncoding, valueWidth)
	if err != nil {
		return nil, nil, true, err
	}
	tokenStart := cache.TokenCount()
	if len(cache.pages) == 0 {
		return nil, nil, false, nil
	}
	last := cache.pages[len(cache.pages)-1]
	if last.tokenStart+last.tokenCount != tokenStart ||
		last.tokenCount <= 0 ||
		last.tokenCount >= cache.blockSize ||
		last.keyWidth != keyWidth ||
		last.valueWidth != valueWidth ||
		last.key.encoding != keyEncoding ||
		last.value.encoding != valueEncoding ||
		!rocmDeviceKVInterleavedPageHasRowCapacity(last, keyStride, valueStride, cache.blockSize, tokenCount) {
		return nil, nil, false, nil
	}
	keyEncodingCode, err := rocmDeviceKVEncodingCode(keyEncoding)
	if err != nil {
		return nil, nil, true, err
	}
	valueEncodingCode, err := rocmDeviceKVEncodingCode(valueEncoding)
	if err != nil {
		return nil, nil, true, err
	}
	modeCode, err := rocmDeviceKVModeCode(cache.mode)
	if err != nil {
		return nil, nil, true, err
	}
	outputBytes := uint64(rocmDeviceKVDescriptorHeaderBytes + cache.PageCount()*rocmDeviceKVDescriptorPageBytes)
	descriptorPointer := previousTable.Pointer()
	allocationBytes := previousTable.AllocationBytes()
	inPlace := !previousTable.borrowed && descriptorPointer != 0 && allocationBytes >= outputBytes
	if !inPlace {
		descriptorPointer, allocationBytes, err = rocmDeviceKVDescriptorTableMalloc(cache.driver, outputBytes)
		if err != nil {
			return nil, nil, true, core.E("rocm.KVCache.DeviceDescriptor", "allocate fused appended descriptor table", err)
		}
	}
	rowOffset := last.tokenCount
	keyOutputBytes := keyStride * uint64(tokenCount)
	valueOutputBytes := valueStride * uint64(tokenCount)
	keyOutputPointer := last.key.pointer + nativeDevicePointer(keyStride*uint64(rowOffset))
	valueOutputPointer := last.value.pointer + nativeDevicePointer(valueStride*uint64(rowOffset))
	nextKeyBytes := last.key.sizeBytes + keyOutputBytes
	nextValueBytes := last.value.sizeBytes + valueOutputBytes
	encodeArgs := hipKVEncodeTokenValueNormLaunchArgs{
		KeyInputPointer:    key.Pointer(),
		ValueInputPointer:  rawValue.Pointer(),
		KeyOutputPointer:   keyOutputPointer,
		ValueOutputPointer: valueOutputPointer,
		KeyCount:           key.Count(),
		ValueCount:         rawValue.Count(),
		KeyInputBytes:      key.SizeBytes(),
		ValueInputBytes:    rawValue.SizeBytes(),
		KeyOutputBytes:     keyOutputBytes,
		ValueOutputBytes:   valueOutputBytes,
		KeyEncoding:        keyEncodingCode,
		ValueEncoding:      valueEncodingCode,
		KeyWidth:           keyWidth,
		ValueWidth:         valueWidth,
		TokenCount:         tokenCount,
		ValueHeadDim:       valueHeadDim,
		ValueHeadCount:     valueHeadCount,
		Epsilon:            valueEpsilon,
	}
	descriptorArgs := hipKVDescriptorAppendLaunchArgs{
		PreviousDescriptorPointer: previousTable.Pointer(),
		OutputDescriptorPointer:   descriptorPointer,
		NewKeyPointer:             last.key.pointer,
		NewValuePointer:           last.value.pointer,
		PreviousDescriptorBytes:   previousTable.SizeBytes(),
		OutputDescriptorBytes:     outputBytes,
		NewKeyBytes:               nextKeyBytes,
		NewValueBytes:             nextValueBytes,
		ModeCode:                  modeCode,
		BlockSize:                 cache.blockSize,
		OutputPageCount:           cache.PageCount(),
		OutputTokenCount:          tokenStart + tokenCount,
		KeyWidth:                  last.keyWidth,
		ValueWidth:                last.valueWidth,
		NewKeyEncoding:            keyEncodingCode,
		NewValueEncoding:          valueEncodingCode,
		Reserved0:                 rocmKVDescriptorAppendModeGrowLastPage,
	}
	if err := hipRunKVEncodeTokenValueNormDescriptorAppendKernelWithWorkspace(ctx, cache.driver, encodeArgs, descriptorArgs, workspace); err != nil {
		if !inPlace {
			_ = rocmDeviceKVDescriptorTableFree(cache.driver, descriptorPointer, allocationBytes)
		}
		return nil, nil, true, err
	}
	pages := rocmDeviceKVCopyPagesWithExtra(cache.pages, 0)
	for index := range pages {
		pages[index].owned = false
	}
	pages[len(pages)-1].tokenCount += tokenCount
	pages[len(pages)-1].key.sizeBytes = nextKeyBytes
	pages[len(pages)-1].value.sizeBytes = nextValueBytes
	next := rocmBorrowDeviceKVCache(cache.driver, cache.mode, cache.blockSize, tokenStart+tokenCount, pages, false)
	if inPlace {
		previousTable.sizeBytes = outputBytes
		previousTable.pageCount = next.PageCount()
		previousTable.version = rocmDeviceKVDescriptorVersion
		return next, previousTable, true, nil
	}
	table := rocmBorrowDeviceKVDescriptorTableAllocated(cache.driver, descriptorPointer, outputBytes, allocationBytes, rocmDeviceKVDescriptorVersion, next.PageCount(), false, true)
	return next, table, true, nil
}

func (cache *rocmDeviceKVCache) withAppendedDeviceRowsInterleavedBlockWithEngineConfig(ctx context.Context, key, value *hipDeviceByteBuffer, keyWidth, valueWidth, tokenCount, window int, engineConfig hipGemma4Q4EngineConfig) (*rocmDeviceKVCache, bool, error) {
	if cache == nil || cache.blockSize <= 1 || tokenCount <= 0 || !engineConfig.interleavedRowPagesEnabled() {
		return nil, false, nil
	}
	keyEncoding, valueEncoding, ok := rocmKVInterleavedEncodingsForMode(cache.mode)
	if !ok {
		return nil, false, nil
	}
	if key == nil || value == nil || key.Pointer() == 0 || value.Pointer() == 0 ||
		key.Count() != keyWidth*tokenCount || value.Count() != valueWidth*tokenCount {
		return nil, false, core.E("rocm.KVCache.DeviceAppend", "device KV row buffers must match cache widths", nil)
	}
	if len(cache.pages) == 0 {
		return nil, false, nil
	}
	tokenStart := cache.TokenCount()
	last := cache.pages[len(cache.pages)-1]
	if last.tokenStart+last.tokenCount != tokenStart ||
		last.tokenCount <= 0 || last.tokenCount+tokenCount > cache.blockSize ||
		last.keyWidth != keyWidth || last.valueWidth != valueWidth ||
		last.key.encoding != keyEncoding || last.value.encoding != valueEncoding {
		return nil, false, nil
	}
	keyStride, err := rocmKVInterleavedRowStride(keyEncoding, keyWidth)
	if err != nil {
		return nil, true, err
	}
	valueStride, err := rocmKVInterleavedRowStride(valueEncoding, valueWidth)
	if err != nil {
		return nil, true, err
	}
	if !rocmDeviceKVInterleavedPageHasRowCapacity(last, keyStride, valueStride, cache.blockSize, tokenCount) {
		return nil, false, nil
	}
	rowOffset := last.tokenCount
	keyOutputPointer := last.key.pointer + nativeDevicePointer(keyStride*uint64(rowOffset))
	valueOutputPointer := last.value.pointer + nativeDevicePointer(valueStride*uint64(rowOffset))
	if err := hipRunKVEncodeRowsKernelInto(ctx, cache.driver, key, value, keyWidth, valueWidth, tokenCount, keyOutputPointer, valueOutputPointer, keyStride*uint64(tokenCount), valueStride*uint64(tokenCount), keyEncoding, valueEncoding); err != nil {
		return nil, true, err
	}
	pages := rocmDeviceKVCopyPagesWithExtra(cache.pages, 0)
	for index := range pages {
		pages[index].owned = false
	}
	pages[len(pages)-1].tokenCount += tokenCount
	pages[len(pages)-1].key.sizeBytes += keyStride * uint64(tokenCount)
	pages[len(pages)-1].value.sizeBytes += valueStride * uint64(tokenCount)
	next := rocmBorrowDeviceKVCache(cache.driver, cache.mode, cache.blockSize, tokenStart+tokenCount, pages, false)
	if window > 0 && next.TokenCount() > window {
		next = next.trimDeviceTokenWindowForAppendWithEngineConfig(window, engineConfig)
	}
	return next, true, nil
}

func rocmDeviceKVCacheEncodeDeviceRowsPage(ctx context.Context, driver nativeHIPDriver, mode string, blockSize int, key, value *hipDeviceByteBuffer, keyWidth, valueWidth, pageTokens int) (rocmDeviceKVTensor, rocmDeviceKVTensor, error) {
	return rocmDeviceKVCacheEncodeDeviceRowsPageWithEngineConfig(ctx, driver, mode, blockSize, key, value, keyWidth, valueWidth, pageTokens, defaultHIPGemma4Q4EngineConfig())
}

func rocmDeviceKVCacheEncodeDeviceRowsPageWithEngineConfig(ctx context.Context, driver nativeHIPDriver, mode string, blockSize int, key, value *hipDeviceByteBuffer, keyWidth, valueWidth, pageTokens int, engineConfig hipGemma4Q4EngineConfig) (rocmDeviceKVTensor, rocmDeviceKVTensor, error) {
	if blockSize > 1 && engineConfig.interleavedRowPagesEnabled() {
		if keyEncoding, valueEncoding, ok := rocmKVInterleavedEncodingsForMode(mode); ok {
			deviceKey, deviceValue, keyStride, valueStride, err := rocmDeviceKVAllocateInterleavedTensorPair(driver, keyWidth, valueWidth, blockSize, keyEncoding, valueEncoding)
			if err != nil {
				return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
			}
			keyBytes := keyStride * uint64(pageTokens)
			valueBytes := valueStride * uint64(pageTokens)
			if err := hipRunKVEncodeRowsKernelInto(ctx, driver, key, value, keyWidth, valueWidth, pageTokens, deviceKey.pointer, deviceValue.pointer, keyBytes, valueBytes, keyEncoding, valueEncoding); err != nil {
				_ = rocmDeviceKVTensorFreePair(driver, deviceKey, deviceValue)
				return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
			}
			deviceKey.sizeBytes = keyBytes
			deviceValue.sizeBytes = valueBytes
			return deviceKey, deviceValue, nil
		}
	}
	return hipRunKVEncodeRowsKernel(ctx, driver, key, value, keyWidth, valueWidth, pageTokens, mode)
}

func newROCmDeviceKVCacheFromDeviceToken(ctx context.Context, driver nativeHIPDriver, mode string, blockSize int, key, value *hipDeviceByteBuffer, window int) (*rocmDeviceKVCache, error) {
	return newROCmDeviceKVCacheFromDeviceTokenWithWorkspace(ctx, driver, mode, blockSize, key, value, window, nil)
}

func newROCmDeviceKVCacheFromDeviceTokenWithWorkspace(ctx context.Context, driver nativeHIPDriver, mode string, blockSize int, key, value *hipDeviceByteBuffer, window int, workspace *hipAttentionHeadsChunkedWorkspace) (*rocmDeviceKVCache, error) {
	if driver == nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	if key == nil || value == nil || key.Pointer() == 0 || value.Pointer() == 0 {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV token buffers are required", nil)
	}
	mode = firstNonEmptyString(mode, rocmKVCacheModeFP16)
	if !isROCmKVCacheMode(mode) {
		return nil, core.E("rocm.KVCache.DeviceAppend", core.Sprintf("unsupported cache mode %q", mode), nil)
	}
	if blockSize <= 0 {
		blockSize = defaultROCmKVBlockSize
	}
	if blockSize > 1 {
		if keyEncoding, valueEncoding, ok := rocmKVInterleavedEncodingsForMode(mode); ok {
			deviceKey, deviceValue, keyStride, valueStride, err := rocmDeviceKVAllocateInterleavedTensorPair(driver, key.Count(), value.Count(), blockSize, keyEncoding, valueEncoding)
			if err != nil {
				return nil, err
			}
			if err := hipRunKVEncodeRowsKernelIntoWithWorkspace(ctx, driver, key, value, key.Count(), value.Count(), 1, deviceKey.pointer, deviceValue.pointer, keyStride, valueStride, keyEncoding, valueEncoding, workspace); err != nil {
				_ = rocmDeviceKVTensorFreePair(driver, deviceKey, deviceValue)
				return nil, err
			}
			pages := rocmDeviceKVBorrowPageSlice(0, 1)
			pages = append(pages, rocmDeviceKVPage{
				tokenStart: 0,
				tokenCount: 1,
				keyWidth:   key.Count(),
				valueWidth: value.Count(),
				key:        deviceKey,
				value:      deviceValue,
				owned:      true,
			})
			return rocmBorrowDeviceKVCache(driver, mode, blockSize, 1, pages, false), nil
		}
	}
	encodedKey, encodedValue, err := hipRunKVEncodeTokenKernelWithWorkspace(ctx, driver, key, value, mode, workspace)
	if err != nil {
		return nil, err
	}
	cache := rocmBorrowDeviceKVCache(driver, mode, blockSize, 0, nil, false)
	next, err := cache.withAppendedEncodedToken(encodedKey, encodedValue, key.Count(), value.Count())
	rocmReleaseDeviceKVCache(cache)
	if err != nil {
		_ = rocmDeviceKVTensorFreePair(driver, encodedKey, encodedValue)
		return nil, err
	}
	if window > 0 && next.TokenCount() > window {
		return next.trimDeviceTokenWindowForAppend(window), nil
	}
	return next, nil
}

func newROCmDeviceKVCacheFromDeviceRows(ctx context.Context, driver nativeHIPDriver, mode string, blockSize int, key, value *hipDeviceByteBuffer, keyWidth, valueWidth, tokenCount, window int) (*rocmDeviceKVCache, error) {
	return newROCmDeviceKVCacheFromDeviceRowsWithEngineConfig(ctx, driver, mode, blockSize, key, value, keyWidth, valueWidth, tokenCount, window, defaultHIPGemma4Q4EngineConfig())
}

func newROCmDeviceKVCacheFromDeviceRowsWithEngineConfig(ctx context.Context, driver nativeHIPDriver, mode string, blockSize int, key, value *hipDeviceByteBuffer, keyWidth, valueWidth, tokenCount, window int, engineConfig hipGemma4Q4EngineConfig) (*rocmDeviceKVCache, error) {
	if driver == nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	mode = firstNonEmptyString(mode, rocmKVCacheModeFP16)
	if !isROCmKVCacheMode(mode) {
		return nil, core.E("rocm.KVCache.DeviceAppend", core.Sprintf("unsupported cache mode %q", mode), nil)
	}
	if blockSize <= 0 {
		blockSize = defaultROCmKVBlockSize
	}
	cache := rocmBorrowDeviceKVCache(driver, mode, blockSize, 0, nil, false)
	next, err := cache.withAppendedDeviceRowsWindowWithEngineConfig(ctx, key, value, keyWidth, valueWidth, tokenCount, window, engineConfig)
	rocmReleaseDeviceKVCache(cache)
	return next, err
}

func (cache *rocmDeviceKVCache) withAppendedEncodedToken(key, value rocmDeviceKVTensor, keyWidth, valueWidth int) (*rocmDeviceKVCache, error) {
	return cache.withAppendedEncodedRows(key, value, keyWidth, valueWidth, 1)
}

func (cache *rocmDeviceKVCache) withAppendedEncodedTokenWindow(key, value rocmDeviceKVTensor, keyWidth, valueWidth, window int) (*rocmDeviceKVCache, error) {
	if window <= 0 || cache == nil || cache.TokenCount()+1 <= window {
		return cache.withAppendedEncodedToken(key, value, keyWidth, valueWidth)
	}
	next, ok, err := cache.withAppendedEncodedTokenTrimmed(key, value, keyWidth, valueWidth, window)
	if err != nil {
		return nil, err
	}
	if ok {
		return next, nil
	}
	next, err = cache.withAppendedEncodedToken(key, value, keyWidth, valueWidth)
	if err != nil {
		return nil, err
	}
	return next.trimDeviceTokenWindow(window), nil
}

func (cache *rocmDeviceKVCache) withAppendedEncodedTokenTrimmed(key, value rocmDeviceKVTensor, keyWidth, valueWidth, window int) (*rocmDeviceKVCache, bool, error) {
	if err := cache.validateAppendedEncodedRows(key, value, keyWidth, valueWidth, 1); err != nil {
		return nil, false, err
	}
	trimStart := cache.TokenCount() + 1 - window
	pages := rocmDeviceKVBorrowPageSlice(0, window)
	for _, page := range cache.pages {
		pageEnd := page.tokenStart + page.tokenCount
		if pageEnd <= trimStart {
			continue
		}
		if page.tokenStart < trimStart {
			rocmDeviceKVReleasePageSlice(pages)
			return nil, false, nil
		}
		page.tokenStart -= trimStart
		page.owned = false
		pages = append(pages, page)
	}
	tokenStart := cache.TokenCount() - trimStart
	pages = append(pages, rocmDeviceKVPage{
		tokenStart: tokenStart,
		tokenCount: 1,
		keyWidth:   keyWidth,
		valueWidth: valueWidth,
		key:        key,
		value:      value,
		owned:      true,
	})
	return rocmBorrowDeviceKVCache(cache.driver, cache.mode, cache.blockSize, window, pages, false), true, nil
}

func (cache *rocmDeviceKVCache) withAppendedEncodedRows(key, value rocmDeviceKVTensor, keyWidth, valueWidth, tokenCount int) (*rocmDeviceKVCache, error) {
	if err := cache.validateAppendedEncodedRows(key, value, keyWidth, valueWidth, tokenCount); err != nil {
		return nil, err
	}
	tokenStart := cache.TokenCount()
	next := rocmBorrowDeviceKVCache(cache.driver, cache.mode, cache.blockSize, tokenStart+tokenCount, rocmDeviceKVCopyPagesWithExtra(cache.pages, 1), false)
	for index := range next.pages {
		next.pages[index].owned = false
	}
	next.pages = append(next.pages, rocmDeviceKVPage{
		tokenStart: tokenStart,
		tokenCount: tokenCount,
		keyWidth:   keyWidth,
		valueWidth: valueWidth,
		key:        key,
		value:      value,
		owned:      true,
	})
	return next, nil
}

func (cache *rocmDeviceKVCache) validateAppendedEncodedRows(key, value rocmDeviceKVTensor, keyWidth, valueWidth, tokenCount int) error {
	if cache == nil {
		return core.E("rocm.KVCache.DeviceAppend", "device KV cache is nil", nil)
	}
	if cache.closed {
		return core.E("rocm.KVCache.DeviceAppend", "device KV cache is closed", nil)
	}
	if cache.driver == nil {
		return core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !cache.driver.Available() {
		return core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	if key.pointer == 0 || value.pointer == 0 || key.sizeBytes == 0 || value.sizeBytes == 0 {
		return core.E("rocm.KVCache.DeviceAppend", "encoded device KV row tensors are required", nil)
	}
	if keyWidth <= 0 || valueWidth <= 0 || tokenCount <= 0 {
		return core.E("rocm.KVCache.DeviceAppend", "KV row widths and token count must be positive", nil)
	}
	if priorKeyWidth, priorValueWidth, ok := cache.LastVectorWidths(); ok && (priorKeyWidth != keyWidth || priorValueWidth != valueWidth) {
		return core.E("rocm.KVCache.DeviceAppend", "KV row widths must match device cache shape", nil)
	}
	expectedKeyEncoding, expectedValueEncoding := rocmKVEncodingsForMode(cache.mode)
	if !rocmDeviceKVEncodingCompatible(key.encoding, expectedKeyEncoding, tokenCount) ||
		!rocmDeviceKVEncodingCompatible(value.encoding, expectedValueEncoding, tokenCount) {
		return core.E("rocm.KVCache.DeviceAppend", "encoded device KV row encodings do not match cache mode", nil)
	}
	expectedKeyBytes, err := rocmKVTensorDeviceByteCountRows(key.encoding, keyWidth*tokenCount, tokenCount)
	if err != nil {
		return err
	}
	expectedValueBytes, err := rocmKVTensorDeviceByteCountRows(value.encoding, valueWidth*tokenCount, tokenCount)
	if err != nil {
		return err
	}
	if key.sizeBytes != expectedKeyBytes || value.sizeBytes != expectedValueBytes {
		return core.E("rocm.KVCache.DeviceAppend", "encoded device KV row byte count mismatch", nil)
	}
	return nil
}

func rocmDeviceKVEncodingCompatible(got, want string, tokenCount int) bool {
	if got == want {
		return true
	}
	if tokenCount <= 1 {
		return false
	}
	switch want {
	case rocmKVEncodingQ8:
		return got == rocmKVEncodingQ8Rows || got == rocmKVEncodingQ8RowsI
	case rocmKVEncodingQ4:
		return got == rocmKVEncodingQ4Rows || got == rocmKVEncodingQ4RowsI
	default:
		return false
	}
}

func (cache *rocmDeviceKVCache) trimDeviceTokenWindow(window int) *rocmDeviceKVCache {
	if cache == nil || window <= 0 || cache.TokenCount() <= window {
		return cache
	}
	oldTokenCount := cache.TokenCount()
	trimStart := oldTokenCount - window
	trimmed := rocmDeviceKVBorrowPageSlice(0, len(cache.pages))
	for _, page := range cache.pages {
		pageEnd := page.tokenStart + page.tokenCount
		if pageEnd <= trimStart {
			continue
		}
		if page.tokenStart < trimStart {
			sliced, ok := rocmDeviceKVSliceInterleavedPage(page, trimStart)
			if !ok {
				rocmDeviceKVReleasePageSlice(trimmed)
				return cache
			}
			trimmed = append(trimmed, sliced)
			continue
		}
		page.tokenStart -= trimStart
		trimmed = append(trimmed, page)
	}
	if len(trimmed) == 0 {
		rocmDeviceKVReleasePageSlice(trimmed)
		return cache
	}
	pages := cache.pages
	cache.pages = trimmed
	cache.tokenCount = oldTokenCount - trimStart
	rocmDeviceKVReleasePageSlice(pages)
	return cache
}

func (cache *rocmDeviceKVCache) trimDeviceTokenWindowForAppend(window int) *rocmDeviceKVCache {
	return cache.trimDeviceTokenWindowForAppendWithEngineConfig(window, defaultHIPGemma4Q4EngineConfig())
}

func (cache *rocmDeviceKVCache) trimDeviceTokenWindowForAppendWithEngineConfig(window int, engineConfig hipGemma4Q4EngineConfig) *rocmDeviceKVCache {
	if cache == nil || window <= 0 || cache.TokenCount() <= window {
		return cache
	}
	if engineConfig.pageAlignedLocalKVEnabled() && cache.blockSize > 1 {
		if trimmed, ok := cache.trimDeviceTokenWindowPageAligned(window); ok {
			return trimmed
		}
	}
	return cache.trimDeviceTokenWindow(window)
}

func (cache *rocmDeviceKVCache) truncateDeviceTokenCount(tokenCount int) error {
	if cache == nil {
		return core.E("rocm.KVCache.DeviceAppend", "device KV cache is nil", nil)
	}
	if cache.closed {
		return core.E("rocm.KVCache.DeviceAppend", "device KV cache is closed", nil)
	}
	if tokenCount <= 0 {
		return core.E("rocm.KVCache.DeviceAppend", "device KV truncate token count must be positive", nil)
	}
	if tokenCount >= cache.TokenCount() {
		return nil
	}
	if cache.borrowed {
		return core.E("rocm.KVCache.DeviceAppend", "borrowed device KV cache cannot be truncated", nil)
	}
	trimmed := rocmDeviceKVBorrowPageSlice(0, len(cache.pages))
	var lastErr error
	for _, page := range cache.pages {
		if page.tokenStart >= tokenCount {
			rocmDeviceKVFreeOwnedPage(cache.driver, &page, &lastErr)
			continue
		}
		pageEnd := page.tokenStart + page.tokenCount
		if pageEnd > tokenCount {
			keepTokens := tokenCount - page.tokenStart
			if keepTokens <= 0 {
				rocmDeviceKVFreeOwnedPage(cache.driver, &page, &lastErr)
				continue
			}
			truncated, err := rocmDeviceKVPagePrefix(page, keepTokens)
			if err != nil {
				rocmDeviceKVReleasePageSlice(trimmed)
				return err
			}
			page = truncated
		}
		trimmed = append(trimmed, page)
	}
	if len(trimmed) == 0 {
		rocmDeviceKVReleasePageSlice(trimmed)
		return core.E("rocm.KVCache.DeviceAppend", "device KV truncate removed every page", nil)
	}
	pages := cache.pages
	cache.pages = trimmed
	cache.tokenCount = tokenCount
	rocmDeviceKVReleasePageSlice(pages)
	return lastErr
}

func rocmDeviceKVPagePrefix(page rocmDeviceKVPage, tokenCount int) (rocmDeviceKVPage, error) {
	if tokenCount <= 0 || tokenCount >= page.tokenCount {
		return page, nil
	}
	if page.key.allocationPointer == 0 || page.key.allocationBytes == 0 ||
		page.value.allocationPointer == 0 || page.value.allocationBytes == 0 {
		return rocmDeviceKVPage{}, core.E("rocm.KVCache.DeviceAppend", "device KV page cannot be prefix-truncated without allocation metadata", nil)
	}
	keyBytes, err := rocmKVTensorDeviceByteCountRows(page.key.encoding, page.keyWidth*tokenCount, tokenCount)
	if err != nil {
		return rocmDeviceKVPage{}, err
	}
	valueBytes, err := rocmKVTensorDeviceByteCountRows(page.value.encoding, page.valueWidth*tokenCount, tokenCount)
	if err != nil {
		return rocmDeviceKVPage{}, err
	}
	if keyBytes == 0 || valueBytes == 0 || keyBytes > page.key.sizeBytes || valueBytes > page.value.sizeBytes {
		return rocmDeviceKVPage{}, core.E("rocm.KVCache.DeviceAppend", "device KV prefix truncate byte count mismatch", nil)
	}
	page.tokenCount = tokenCount
	page.key.sizeBytes = keyBytes
	page.value.sizeBytes = valueBytes
	return page, nil
}

func (cache *rocmDeviceKVCache) trimDeviceTokenWindowPageAligned(window int) (*rocmDeviceKVCache, bool) {
	if cache == nil || window <= 0 || cache.TokenCount() <= window {
		return cache, true
	}
	oldTokenCount := cache.TokenCount()
	maxRetainedTokens := window + cache.blockSize - 1
	if oldTokenCount <= maxRetainedTokens {
		return cache, true
	}
	trimStart := oldTokenCount - window
	dropStart := 0
	firstRetained := -1
	for index, page := range cache.pages {
		pageEnd := page.tokenStart + page.tokenCount
		if pageEnd <= trimStart {
			dropStart = pageEnd
			continue
		}
		firstRetained = index
		break
	}
	if firstRetained <= 0 || dropStart <= 0 {
		return cache, false
	}
	trimmed := rocmDeviceKVBorrowPageSlice(0, len(cache.pages)-firstRetained)
	for _, page := range cache.pages[firstRetained:] {
		if page.tokenStart < dropStart {
			rocmDeviceKVReleasePageSlice(trimmed)
			return cache, false
		}
		page.tokenStart -= dropStart
		trimmed = append(trimmed, page)
	}
	if len(trimmed) == 0 {
		rocmDeviceKVReleasePageSlice(trimmed)
		return cache, false
	}
	pages := cache.pages
	cache.pages = trimmed
	cache.tokenCount = oldTokenCount - dropStart
	rocmDeviceKVReleasePageSlice(pages)
	return cache, true
}

func rocmDeviceKVSliceInterleavedPage(page rocmDeviceKVPage, trimStart int) (rocmDeviceKVPage, bool) {
	if trimStart <= page.tokenStart || trimStart >= page.tokenStart+page.tokenCount {
		return rocmDeviceKVPage{}, false
	}
	skipTokens := trimStart - page.tokenStart
	keyStride, err := rocmKVInterleavedRowStride(page.key.encoding, page.keyWidth)
	if err != nil {
		return rocmDeviceKVPage{}, false
	}
	valueStride, err := rocmKVInterleavedRowStride(page.value.encoding, page.valueWidth)
	if err != nil {
		return rocmDeviceKVPage{}, false
	}
	if page.key.sizeBytes != keyStride*uint64(page.tokenCount) || page.value.sizeBytes != valueStride*uint64(page.tokenCount) {
		return rocmDeviceKVPage{}, false
	}
	keySkipBytes := keyStride * uint64(skipTokens)
	valueSkipBytes := valueStride * uint64(skipTokens)
	if keySkipBytes >= page.key.sizeBytes || valueSkipBytes >= page.value.sizeBytes {
		return rocmDeviceKVPage{}, false
	}
	page.tokenStart = 0
	page.tokenCount -= skipTokens
	page.key.pointer += nativeDevicePointer(keySkipBytes)
	page.key.sizeBytes -= keySkipBytes
	page.value.pointer += nativeDevicePointer(valueSkipBytes)
	page.value.sizeBytes -= valueSkipBytes
	return page, true
}

func hipRunKVEncodeTokenKernel(ctx context.Context, driver nativeHIPDriver, key, value *hipDeviceByteBuffer, mode string) (rocmDeviceKVTensor, rocmDeviceKVTensor, error) {
	return hipRunKVEncodeTokenKernelWithWorkspace(ctx, driver, key, value, mode, nil)
}

func hipRunKVEncodeTokenKernelWithWorkspace(ctx context.Context, driver nativeHIPDriver, key, value *hipDeviceByteBuffer, mode string, workspace *hipAttentionHeadsChunkedWorkspace) (rocmDeviceKVTensor, rocmDeviceKVTensor, error) {
	keyWidth := 0
	if key != nil {
		keyWidth = key.Count()
	}
	valueWidth := 0
	if value != nil {
		valueWidth = value.Count()
	}
	return hipRunKVEncodeRowsKernelWithWorkspace(ctx, driver, key, value, keyWidth, valueWidth, 1, mode, workspace)
}

func hipRunKVEncodeRowsKernel(ctx context.Context, driver nativeHIPDriver, key, value *hipDeviceByteBuffer, keyWidth, valueWidth, tokenCount int, mode string) (rocmDeviceKVTensor, rocmDeviceKVTensor, error) {
	return hipRunKVEncodeRowsKernelWithWorkspace(ctx, driver, key, value, keyWidth, valueWidth, tokenCount, mode, nil)
}

func hipRunKVEncodeRowsKernelWithWorkspace(ctx context.Context, driver nativeHIPDriver, key, value *hipDeviceByteBuffer, keyWidth, valueWidth, tokenCount int, mode string, workspace *hipAttentionHeadsChunkedWorkspace) (rocmDeviceKVTensor, rocmDeviceKVTensor, error) {
	if err := hipContextErr(ctx); err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	if driver == nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	if key == nil || key.Pointer() == 0 || key.Count() <= 0 || key.SizeBytes() != uint64(key.Count())*4 {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "device KV key token buffer is required", nil)
	}
	if value == nil || value.Pointer() == 0 || value.Count() <= 0 || value.SizeBytes() != uint64(value.Count())*4 {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "device KV value token buffer is required", nil)
	}
	if keyWidth <= 0 || valueWidth <= 0 || tokenCount <= 0 || key.Count() != keyWidth*tokenCount || value.Count() != valueWidth*tokenCount {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "device KV row shape mismatch", nil)
	}
	mode = firstNonEmptyString(mode, rocmKVCacheModeFP16)
	if !isROCmKVCacheMode(mode) {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", core.Sprintf("unsupported cache mode %q", mode), nil)
	}
	keyEncoding, valueEncoding := rocmKVEncodingsForMode(mode)
	if tokenCount > 1 {
		if keyEncoding == rocmKVEncodingQ8 {
			keyEncoding = rocmKVEncodingQ8Rows
		}
		if valueEncoding == rocmKVEncodingQ4 {
			valueEncoding = rocmKVEncodingQ4Rows
		}
		if valueEncoding == rocmKVEncodingQ8 {
			valueEncoding = rocmKVEncodingQ8Rows
		}
	}
	keyEncodingCode, err := rocmDeviceKVEncodingCode(keyEncoding)
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	valueEncodingCode, err := rocmDeviceKVEncodingCode(valueEncoding)
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	keyBytes, err := rocmKVTensorDeviceByteCountRows(keyEncoding, key.Count(), tokenCount)
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	valueBytes, err := rocmKVTensorDeviceByteCountRows(valueEncoding, value.Count(), tokenCount)
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	encodedKey, encodedValue, err := rocmDeviceKVAllocateEncodedTensorPair(driver, keyBytes, valueBytes, keyEncoding, valueEncoding)
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	launchArgs := hipKVEncodeTokenLaunchArgs{
		KeyInputPointer:    key.Pointer(),
		ValueInputPointer:  value.Pointer(),
		KeyOutputPointer:   encodedKey.pointer,
		ValueOutputPointer: encodedValue.pointer,
		KeyCount:           key.Count(),
		ValueCount:         value.Count(),
		KeyInputBytes:      key.SizeBytes(),
		ValueInputBytes:    value.SizeBytes(),
		KeyOutputBytes:     keyBytes,
		ValueOutputBytes:   valueBytes,
		KeyEncoding:        keyEncodingCode,
		ValueEncoding:      valueEncodingCode,
		KeyWidth:           keyWidth,
		ValueWidth:         valueWidth,
		TokenCount:         tokenCount,
	}
	var payload []byte
	if workspace != nil {
		payload, err = launchArgs.BinaryInto(workspace.KVEncodeTokenArgs[:])
	} else {
		payload, err = launchArgs.Binary()
	}
	if err != nil {
		_ = rocmDeviceKVTensorFreePair(driver, encodedKey, encodedValue)
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameKVEncodeToken,
		Args:   payload,
		GridX:  2,
		GridY:  1,
		GridZ:  1,
		BlockX: hipKVEncodeTokenBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		_ = rocmDeviceKVTensorFreePair(driver, encodedKey, encodedValue)
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	return encodedKey, encodedValue, nil
}

func hipRunKVEncodeRowsKernelInto(ctx context.Context, driver nativeHIPDriver, key, value *hipDeviceByteBuffer, keyWidth, valueWidth, tokenCount int, keyOutputPointer, valueOutputPointer nativeDevicePointer, keyOutputBytes, valueOutputBytes uint64, keyEncoding, valueEncoding string) error {
	return hipRunKVEncodeRowsKernelIntoWithWorkspace(ctx, driver, key, value, keyWidth, valueWidth, tokenCount, keyOutputPointer, valueOutputPointer, keyOutputBytes, valueOutputBytes, keyEncoding, valueEncoding, nil)
}

func hipRunKVEncodeRowsKernelIntoWithWorkspace(ctx context.Context, driver nativeHIPDriver, key, value *hipDeviceByteBuffer, keyWidth, valueWidth, tokenCount int, keyOutputPointer, valueOutputPointer nativeDevicePointer, keyOutputBytes, valueOutputBytes uint64, keyEncoding, valueEncoding string, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil {
		return core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	if key == nil || key.Pointer() == 0 || key.Count() <= 0 || key.SizeBytes() != uint64(key.Count())*4 {
		return core.E("rocm.KVCache.DeviceAppend", "device KV key token buffer is required", nil)
	}
	if value == nil || value.Pointer() == 0 || value.Count() <= 0 || value.SizeBytes() != uint64(value.Count())*4 {
		return core.E("rocm.KVCache.DeviceAppend", "device KV value token buffer is required", nil)
	}
	if keyWidth <= 0 || valueWidth <= 0 || tokenCount <= 0 || key.Count() != keyWidth*tokenCount || value.Count() != valueWidth*tokenCount {
		return core.E("rocm.KVCache.DeviceAppend", "device KV row shape mismatch", nil)
	}
	if keyOutputPointer == 0 || valueOutputPointer == 0 || keyOutputBytes == 0 || valueOutputBytes == 0 {
		return core.E("rocm.KVCache.DeviceAppend", "encoded KV output buffers are required", nil)
	}
	keyEncodingCode, err := rocmDeviceKVEncodingCode(keyEncoding)
	if err != nil {
		return err
	}
	valueEncodingCode, err := rocmDeviceKVEncodingCode(valueEncoding)
	if err != nil {
		return err
	}
	launchArgs := hipKVEncodeTokenLaunchArgs{
		KeyInputPointer:    key.Pointer(),
		ValueInputPointer:  value.Pointer(),
		KeyOutputPointer:   keyOutputPointer,
		ValueOutputPointer: valueOutputPointer,
		KeyCount:           key.Count(),
		ValueCount:         value.Count(),
		KeyInputBytes:      key.SizeBytes(),
		ValueInputBytes:    value.SizeBytes(),
		KeyOutputBytes:     keyOutputBytes,
		ValueOutputBytes:   valueOutputBytes,
		KeyEncoding:        keyEncodingCode,
		ValueEncoding:      valueEncodingCode,
		KeyWidth:           keyWidth,
		ValueWidth:         valueWidth,
		TokenCount:         tokenCount,
	}
	var payload []byte
	if workspace != nil {
		payload, err = launchArgs.BinaryInto(workspace.KVEncodeTokenArgs[:])
	} else {
		payload, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameKVEncodeToken,
		Args:   payload,
		GridX:  2,
		GridY:  1,
		GridZ:  1,
		BlockX: hipKVEncodeTokenBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return hipLaunchKernel(driver, config)
}

func hipRunKVEncodeTokenValueNormKernelWithWorkspace(ctx context.Context, driver nativeHIPDriver, key, rawValue *hipDeviceByteBuffer, mode string, valueHeadDim, valueHeadCount int, valueEpsilon float32, workspace *hipAttentionHeadsChunkedWorkspace) (rocmDeviceKVTensor, rocmDeviceKVTensor, error) {
	keyWidth := 0
	if key != nil {
		keyWidth = key.Count()
	}
	valueWidth := 0
	if rawValue != nil {
		valueWidth = rawValue.Count()
	}
	return hipRunKVEncodeRowsValueNormKernelWithWorkspace(ctx, driver, key, rawValue, keyWidth, valueWidth, 1, mode, valueHeadDim, valueHeadCount, valueEpsilon, workspace)
}

func hipRunKVEncodeRowsValueNormKernelWithWorkspace(ctx context.Context, driver nativeHIPDriver, key, rawValue *hipDeviceByteBuffer, keyWidth, valueWidth, tokenCount int, mode string, valueHeadDim, valueHeadCount int, valueEpsilon float32, workspace *hipAttentionHeadsChunkedWorkspace) (rocmDeviceKVTensor, rocmDeviceKVTensor, error) {
	if err := hipContextErr(ctx); err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	if driver == nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	if key == nil || key.Pointer() == 0 || key.Count() <= 0 || key.SizeBytes() != uint64(key.Count())*4 {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "device KV key token buffer is required", nil)
	}
	if rawValue == nil || rawValue.Pointer() == 0 || rawValue.Count() <= 0 || rawValue.SizeBytes() != uint64(rawValue.Count())*4 {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "device KV raw value token buffer is required", nil)
	}
	if keyWidth <= 0 || valueWidth <= 0 || tokenCount <= 0 || key.Count() != keyWidth*tokenCount || rawValue.Count() != valueWidth*tokenCount {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "device KV row shape mismatch", nil)
	}
	mode = firstNonEmptyString(mode, rocmKVCacheModeFP16)
	if !isROCmKVCacheMode(mode) {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", core.Sprintf("unsupported cache mode %q", mode), nil)
	}
	keyEncoding, valueEncoding := rocmKVEncodingsForMode(mode)
	if tokenCount > 1 {
		if keyEncoding == rocmKVEncodingQ8 {
			keyEncoding = rocmKVEncodingQ8Rows
		}
		if valueEncoding == rocmKVEncodingQ4 {
			valueEncoding = rocmKVEncodingQ4Rows
		}
		if valueEncoding == rocmKVEncodingQ8 {
			valueEncoding = rocmKVEncodingQ8Rows
		}
	}
	keyBytes, err := rocmKVTensorDeviceByteCountRows(keyEncoding, key.Count(), tokenCount)
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	valueBytes, err := rocmKVTensorDeviceByteCountRows(valueEncoding, rawValue.Count(), tokenCount)
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	encodedKey, encodedValue, err := rocmDeviceKVAllocateEncodedTensorPair(driver, keyBytes, valueBytes, keyEncoding, valueEncoding)
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	if err := hipRunKVEncodeRowsValueNormKernelIntoWithWorkspace(ctx, driver, key, rawValue, keyWidth, valueWidth, tokenCount, valueHeadDim, valueHeadCount, valueEpsilon, encodedKey.pointer, encodedValue.pointer, keyBytes, valueBytes, keyEncoding, valueEncoding, workspace); err != nil {
		_ = rocmDeviceKVTensorFreePair(driver, encodedKey, encodedValue)
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	return encodedKey, encodedValue, nil
}

func hipRunKVEncodeRowsValueNormKernelIntoWithWorkspace(ctx context.Context, driver nativeHIPDriver, key, rawValue *hipDeviceByteBuffer, keyWidth, valueWidth, tokenCount int, valueHeadDim, valueHeadCount int, valueEpsilon float32, keyOutputPointer, valueOutputPointer nativeDevicePointer, keyOutputBytes, valueOutputBytes uint64, keyEncoding, valueEncoding string, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil {
		return core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	if key == nil || key.Pointer() == 0 || key.Count() <= 0 || key.SizeBytes() != uint64(key.Count())*4 {
		return core.E("rocm.KVCache.DeviceAppend", "device KV key token buffer is required", nil)
	}
	if rawValue == nil || rawValue.Pointer() == 0 || rawValue.Count() <= 0 || rawValue.SizeBytes() != uint64(rawValue.Count())*4 {
		return core.E("rocm.KVCache.DeviceAppend", "device KV raw value token buffer is required", nil)
	}
	if keyWidth <= 0 || valueWidth <= 0 || tokenCount <= 0 || key.Count() != keyWidth*tokenCount || rawValue.Count() != valueWidth*tokenCount {
		return core.E("rocm.KVCache.DeviceAppend", "device KV row shape mismatch", nil)
	}
	if valueHeadDim <= 0 || valueHeadCount <= 0 || valueHeadCount > hipKVEncodeTokenValueNormMaxHeads || valueWidth != valueHeadDim*valueHeadCount {
		return core.E("rocm.KVCache.DeviceAppend", "device KV value norm head shape mismatch", nil)
	}
	if math.IsNaN(float64(valueEpsilon)) || math.IsInf(float64(valueEpsilon), 0) || valueEpsilon < 0 {
		return core.E("rocm.KVCache.DeviceAppend", "device KV value norm epsilon must be finite and non-negative", nil)
	}
	if keyOutputPointer == 0 || valueOutputPointer == 0 || keyOutputBytes == 0 || valueOutputBytes == 0 {
		return core.E("rocm.KVCache.DeviceAppend", "encoded KV output buffers are required", nil)
	}
	keyEncodingCode, err := rocmDeviceKVEncodingCode(keyEncoding)
	if err != nil {
		return err
	}
	valueEncodingCode, err := rocmDeviceKVEncodingCode(valueEncoding)
	if err != nil {
		return err
	}
	launchArgs := hipKVEncodeTokenValueNormLaunchArgs{
		KeyInputPointer:    key.Pointer(),
		ValueInputPointer:  rawValue.Pointer(),
		KeyOutputPointer:   keyOutputPointer,
		ValueOutputPointer: valueOutputPointer,
		KeyCount:           key.Count(),
		ValueCount:         rawValue.Count(),
		KeyInputBytes:      key.SizeBytes(),
		ValueInputBytes:    rawValue.SizeBytes(),
		KeyOutputBytes:     keyOutputBytes,
		ValueOutputBytes:   valueOutputBytes,
		KeyEncoding:        keyEncodingCode,
		ValueEncoding:      valueEncodingCode,
		KeyWidth:           keyWidth,
		ValueWidth:         valueWidth,
		TokenCount:         tokenCount,
		ValueHeadDim:       valueHeadDim,
		ValueHeadCount:     valueHeadCount,
		Epsilon:            valueEpsilon,
	}
	var payload []byte
	if workspace != nil {
		payload, err = launchArgs.BinaryInto(workspace.KVEncodeTokenValueNormArgs[:])
	} else {
		payload, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameKVEncodeTokenValueNorm,
		Args:   payload,
		GridX:  2,
		GridY:  1,
		GridZ:  1,
		BlockX: hipKVEncodeTokenBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return hipLaunchKernel(driver, config)
}

func hipRunKVEncodeTokenValueNormDescriptorAppendKernelWithWorkspace(ctx context.Context, driver nativeHIPDriver, encodeArgs hipKVEncodeTokenValueNormLaunchArgs, descriptorArgs hipKVDescriptorAppendLaunchArgs, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil {
		return core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	launchArgs := hipKVEncodeTokenValueNormDescriptorAppendLaunchArgs{
		Encode:     encodeArgs,
		Descriptor: descriptorArgs,
	}
	var payload []byte
	var err error
	if workspace != nil {
		payload, err = launchArgs.BinaryInto(workspace.KVEncodeTokenValueNormDescriptorAppendArgs[:])
	} else {
		payload, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameKVEncodeTokenValueNormDescriptorAppend,
		Args:   payload,
		GridX:  3,
		GridY:  1,
		GridZ:  1,
		BlockX: hipKVEncodeTokenBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return hipLaunchKernel(driver, config)
}

func (cache *rocmDeviceKVCache) borrowedAlias() (*rocmDeviceKVCache, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.DeviceAlias", "device KV cache is nil", nil)
	}
	if cache.closed {
		return nil, core.E("rocm.KVCache.DeviceAlias", "device KV cache is closed", nil)
	}
	if cache.driver == nil {
		return nil, core.E("rocm.KVCache.DeviceAlias", "HIP driver is nil", nil)
	}
	if len(cache.pages) == 0 {
		return nil, core.E("rocm.KVCache.DeviceAlias", "device KV cache has no pages", nil)
	}
	return rocmBorrowDeviceKVCache(cache.driver, cache.mode, cache.blockSize, cache.TokenCount(), cache.pages, true), nil
}

func (cache *rocmDeviceKVCache) closePagesFrom(index int) error {
	if cache == nil {
		return nil
	}
	if cache.borrowed {
		rocmReleaseDeviceKVCache(cache)
		return nil
	}
	if index < 0 {
		index = 0
	}
	if index > len(cache.pages) {
		index = len(cache.pages)
	}
	var lastErr error
	pages := cache.pages
	for pageIndex := index; pageIndex < len(cache.pages); pageIndex++ {
		page := &cache.pages[pageIndex]
		if !page.owned {
			continue
		}
		if err := rocmDeviceKVTensorFreePair(cache.driver, page.key, page.value); err != nil {
			lastErr = core.E("rocm.KVCache.DeviceAppend", "free appended KV page", err)
		}
		page.key = rocmDeviceKVTensor{}
		page.value = rocmDeviceKVTensor{}
		page.owned = false
	}
	cache.pages = nil
	cache.tokenCount = 0
	cache.closed = true
	rocmDeviceKVReleasePageSlice(pages)
	return lastErr
}

func (cache *rocmDeviceKVCache) transferPagesTo(next *rocmDeviceKVCache) error {
	if cache == nil {
		return nil
	}
	if next == nil {
		return core.E("rocm.KVCache.DeviceAppend", "next device KV cache is nil", nil)
	}
	if cache.driver != next.driver || cache.mode != next.mode || cache.blockSize != next.blockSize {
		return core.E("rocm.KVCache.DeviceAppend", "device KV cache ownership target does not match", nil)
	}
	if len(next.pages) < len(cache.pages) {
		return core.E("rocm.KVCache.DeviceAppend", "device KV cache ownership target is missing source pages", nil)
	}
	for index := range cache.pages {
		if cache.pages[index].key.pointer != next.pages[index].key.pointer || cache.pages[index].value.pointer != next.pages[index].value.pointer {
			return core.E("rocm.KVCache.DeviceAppend", "device KV cache ownership target page pointers do not match source", nil)
		}
		if cache.pages[index].owned {
			next.pages[index].owned = true
			cache.pages[index].key.pointer = 0
			cache.pages[index].value.pointer = 0
		}
		cache.pages[index].owned = false
	}
	pages := cache.pages
	cache.pages = nil
	cache.tokenCount = 0
	cache.closed = true
	rocmDeviceKVReleasePageSlice(pages)
	return nil
}

func (cache *rocmDeviceKVCache) transferSharedPagesTo(next *rocmDeviceKVCache) error {
	if cache == nil {
		return nil
	}
	if next == nil {
		return core.E("rocm.KVCache.DeviceAppend", "next device KV cache is nil", nil)
	}
	if cache.driver != next.driver || cache.mode != next.mode || cache.blockSize != next.blockSize {
		return core.E("rocm.KVCache.DeviceAppend", "device KV cache ownership target does not match", nil)
	}
	var lastErr error
	sourcePages := cache.pages
	targetPages := next.pages
	if len(sourcePages) > 0 && len(targetPages) >= len(sourcePages) &&
		rocmDeviceKVPagePointersEqual(&sourcePages[0], &targetPages[0]) &&
		rocmDeviceKVPagePointersEqual(&sourcePages[len(sourcePages)-1], &targetPages[len(sourcePages)-1]) {
		for index := range sourcePages {
			rocmDeviceKVTransferPageOwnership(&sourcePages[index], &targetPages[index])
		}
		cache.finishTransferSharedPages()
		return nil
	}
	if suffixOffset := len(sourcePages) - len(targetPages); suffixOffset > 0 && len(targetPages) > 0 &&
		rocmDeviceKVPagePointersEqual(&sourcePages[suffixOffset], &targetPages[0]) &&
		rocmDeviceKVPagePointersEqual(&sourcePages[len(sourcePages)-1], &targetPages[len(targetPages)-1]) {
		for sourceIndex := 0; sourceIndex < suffixOffset; sourceIndex++ {
			rocmDeviceKVFreeOwnedPage(cache.driver, &sourcePages[sourceIndex], &lastErr)
		}
		for targetIndex := range targetPages {
			rocmDeviceKVTransferPageOwnership(&sourcePages[targetIndex+suffixOffset], &targetPages[targetIndex])
		}
		cache.finishTransferSharedPages()
		return lastErr
	}
	if len(sourcePages) > 1 && len(sourcePages) == len(targetPages) &&
		rocmDeviceKVPagePointersEqual(&sourcePages[1], &targetPages[0]) &&
		rocmDeviceKVPagePointersEqual(&sourcePages[len(sourcePages)-1], &targetPages[len(targetPages)-2]) {
		rocmDeviceKVFreeOwnedPage(cache.driver, &sourcePages[0], &lastErr)
		for targetIndex := 0; targetIndex < len(targetPages)-1; targetIndex++ {
			rocmDeviceKVTransferPageOwnership(&sourcePages[targetIndex+1], &targetPages[targetIndex])
		}
		cache.finishTransferSharedPages()
		return lastErr
	}
	slowStorageMatch := rocmDeviceKVPageSliceHasSlicedStorage(sourcePages) || rocmDeviceKVPageSliceHasSlicedStorage(targetPages)
	for sourceIndex := range sourcePages {
		source := &sourcePages[sourceIndex]
		matched := false
		for targetIndex := range targetPages {
			target := &targetPages[targetIndex]
			if rocmDeviceKVPagePointersEqual(source, target) ||
				(slowStorageMatch && rocmDeviceKVPageSharesStorage(source, target)) {
				rocmDeviceKVTransferPageOwnership(source, target)
				matched = true
				break
			}
		}
		if matched || !source.owned {
			continue
		}
		rocmDeviceKVFreeOwnedPage(cache.driver, source, &lastErr)
	}
	cache.finishTransferSharedPages()
	return lastErr
}

func (cache *rocmDeviceKVCache) finishTransferSharedPages() {
	pages := cache.pages
	cache.pages = nil
	cache.tokenCount = 0
	cache.closed = true
	rocmDeviceKVReleasePageSlice(pages)
}

func rocmDeviceKVPagePointersEqual(source, target *rocmDeviceKVPage) bool {
	return source != nil && target != nil &&
		source.key.pointer == target.key.pointer &&
		source.value.pointer == target.value.pointer
}

func rocmDeviceKVPageSliceHasSlicedStorage(pages []rocmDeviceKVPage) bool {
	for index := range pages {
		if rocmDeviceKVPageHasSlicedStorage(&pages[index]) {
			return true
		}
	}
	return false
}

func rocmDeviceKVPageHasSlicedStorage(page *rocmDeviceKVPage) bool {
	if page == nil {
		return false
	}
	return rocmDeviceKVTensorHasSlicedStorage(page.key) || rocmDeviceKVTensorHasSlicedStorage(page.value)
}

func rocmDeviceKVTensorHasSlicedStorage(tensor rocmDeviceKVTensor) bool {
	return tensor.allocationPointer != 0 && tensor.allocationBytes != 0 && tensor.pointer != tensor.allocationPointer
}

func rocmDeviceKVPageSharesStorage(source, target *rocmDeviceKVPage) bool {
	if source == nil || target == nil {
		return false
	}
	return rocmDeviceKVTensorSharesStorage(source.key, target.key) &&
		rocmDeviceKVTensorSharesStorage(source.value, target.value)
}

func rocmDeviceKVTensorSharesStorage(source, target rocmDeviceKVTensor) bool {
	if source.pointer == 0 || target.pointer == 0 {
		return false
	}
	if source.pointer == target.pointer {
		return true
	}
	sourcePointer, sourceBytes := rocmDeviceKVTensorAllocation(source)
	targetPointer, targetBytes := rocmDeviceKVTensorAllocation(target)
	return sourcePointer != 0 &&
		targetPointer != 0 &&
		sourceBytes != 0 &&
		targetBytes != 0 &&
		sourcePointer == targetPointer &&
		sourceBytes == targetBytes
}

func rocmDeviceKVTransferPageOwnership(source, target *rocmDeviceKVPage) {
	if source.owned {
		target.owned = true
		if target.key.allocationPointer == 0 {
			target.key.allocationPointer = source.key.allocationPointer
			target.key.allocationBytes = source.key.allocationBytes
		}
		if target.value.allocationPointer == 0 {
			target.value.allocationPointer = source.value.allocationPointer
			target.value.allocationBytes = source.value.allocationBytes
		}
		source.key = rocmDeviceKVTensor{}
		source.value = rocmDeviceKVTensor{}
	}
	source.owned = false
}

func rocmDeviceKVFreeOwnedPage(driver nativeHIPDriver, page *rocmDeviceKVPage, lastErr *error) {
	if page == nil || !page.owned {
		return
	}
	if err := rocmDeviceKVTensorFreePair(driver, page.key, page.value); err != nil && lastErr != nil {
		*lastErr = core.E("rocm.KVCache.DeviceAppend", "free trimmed KV page", err)
	}
	page.key = rocmDeviceKVTensor{}
	page.value = rocmDeviceKVTensor{}
	page.owned = false
}

func (cache *rocmDeviceKVCache) borrowsPagesFrom(source *rocmDeviceKVCache) bool {
	if cache == nil || source == nil {
		return false
	}
	if cache.driver != source.driver || cache.mode != source.mode || cache.blockSize != source.blockSize {
		return false
	}
	if len(cache.pages) < len(source.pages) {
		return false
	}
	for index := range source.pages {
		if cache.pages[index].key.pointer != source.pages[index].key.pointer ||
			cache.pages[index].value.pointer != source.pages[index].value.pointer {
			return false
		}
	}
	return true
}

func (cache *rocmDeviceKVCache) sharesPagesFrom(source *rocmDeviceKVCache) bool {
	if cache == nil || source == nil {
		return false
	}
	if cache.driver != source.driver || cache.mode != source.mode || cache.blockSize != source.blockSize {
		return false
	}
	for sourceIndex := range source.pages {
		for targetIndex := range cache.pages {
			if rocmDeviceKVPagePointersEqual(&source.pages[sourceIndex], &cache.pages[targetIndex]) {
				return true
			}
		}
	}
	if !rocmDeviceKVPageSliceHasSlicedStorage(source.pages) && !rocmDeviceKVPageSliceHasSlicedStorage(cache.pages) {
		return false
	}
	for sourceIndex := range source.pages {
		for targetIndex := range cache.pages {
			if rocmDeviceKVPageSharesStorage(&source.pages[sourceIndex], &cache.pages[targetIndex]) {
				return true
			}
		}
	}
	return false
}

func (cache *rocmDeviceKVCache) ownsAnyPages() bool {
	if cache == nil || cache.borrowed {
		return false
	}
	for _, page := range cache.pages {
		if page.owned {
			return true
		}
	}
	return false
}

func (cache *rocmDeviceKVCache) Close() error {
	if cache == nil || cache.closed {
		return nil
	}
	if cache.borrowed {
		cache.pages = nil
		cache.tokenCount = 0
		cache.closed = true
		return nil
	}
	var lastErr error
	pages := cache.pages
	for index := range cache.pages {
		page := &cache.pages[index]
		if !page.owned {
			continue
		}
		if err := rocmDeviceKVTensorFreePair(cache.driver, page.key, page.value); err != nil {
			lastErr = core.E("rocm.KVCache.DeviceMirror", "free KV page", err)
		}
		page.key = rocmDeviceKVTensor{}
		page.value = rocmDeviceKVTensor{}
		page.owned = false
	}
	cache.pages = nil
	cache.tokenCount = 0
	cache.closed = true
	rocmDeviceKVReleasePageSlice(pages)
	return lastErr
}

func (cache *rocmDeviceKVCache) PageCount() int {
	if cache == nil {
		return 0
	}
	return len(cache.pages)
}

func (cache *rocmDeviceKVCache) TokenCount() int {
	if cache == nil {
		return 0
	}
	if cache.tokenCount > 0 || len(cache.pages) == 0 {
		return cache.tokenCount
	}
	return rocmDeviceKVPagesTokenCount(cache.pages)
}

func rocmDeviceKVPagesTokenCount(pages []rocmDeviceKVPage) int {
	var maxEnd int
	for _, page := range pages {
		if end := page.tokenStart + page.tokenCount; end > maxEnd {
			maxEnd = end
		}
	}
	return maxEnd
}

func (cache *rocmDeviceKVCache) MemoryBytes() uint64 {
	if cache == nil {
		return 0
	}
	var total uint64
	for _, page := range cache.pages {
		total += page.key.sizeBytes + page.value.sizeBytes
	}
	return total
}

func (cache *rocmDeviceKVCache) Stats() inference.CacheStats {
	if cache == nil {
		return inference.CacheStats{}
	}
	labels := make(map[string]string, 8)
	cache.addStatsLabels(labels)
	labels = rocmApplyCacheProfileLabels(labels, cache.CacheProfile(""))
	return inference.CacheStats{
		Blocks:      len(cache.pages),
		MemoryBytes: cache.MemoryBytes(),
		CacheMode:   cache.mode,
		Labels:      labels,
	}
}

func (cache *rocmDeviceKVCache) addStatsLabels(labels map[string]string) {
	if cache == nil || labels == nil {
		return
	}
	labels["kv_backing"] = "hip_device_mirror"
	labels["kv_block_size"] = rocmDeviceKVLabelInt(cache.blockSize)
	labels["kv_cache_block_size"] = labels["kv_block_size"]
	labels["kv_device_backing"] = "mirrored"
	labels["kv_pages"] = rocmDeviceKVLabelInt(cache.PageCount())
	labels["kv_tokens"] = rocmDeviceKVLabelInt(cache.TokenCount())
	if keyWidth, valueWidth, ok := cache.LastVectorWidths(); ok {
		labels["kv_key_width"] = rocmDeviceKVLabelInt(keyWidth)
		labels["kv_value_width"] = rocmDeviceKVLabelInt(valueWidth)
	}
}

func rocmDeviceKVLabelInt(value int) string {
	if value >= 0 && value <= rocmDeviceKVLabelIntMax {
		return rocmDeviceKVLabelInts[value]
	}
	return strconv.Itoa(value)
}

func rocmDeviceKVLabelUint64(value uint64) string {
	if value <= rocmDeviceKVLabelIntMax {
		return rocmDeviceKVLabelInts[int(value)]
	}
	return strconv.FormatUint(value, 10)
}

func (cache *rocmDeviceKVCache) Snapshot() ([]byte, error) {
	host, err := cache.hostCache()
	if err != nil {
		return nil, err
	}
	return host.Snapshot()
}

func (cache *rocmDeviceKVCache) hostCache() (*rocmKVCache, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.DeviceSnapshot", "device KV cache is nil", nil)
	}
	if cache.closed {
		return nil, core.E("rocm.KVCache.DeviceSnapshot", "device KV cache is closed", nil)
	}
	if cache.driver == nil {
		return nil, core.E("rocm.KVCache.DeviceSnapshot", "HIP driver is nil", nil)
	}
	if !cache.driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceSnapshot", "HIP driver is not available", nil)
	}
	if len(cache.pages) == 0 {
		return nil, core.E("rocm.KVCache.DeviceSnapshot", "device KV cache has no pages", nil)
	}
	host, err := newROCmKVCache(cache.mode, cache.blockSize)
	if err != nil {
		return nil, err
	}
	for _, page := range cache.pages {
		key, err := copyROCmDeviceKVTensorRowsToHost(cache.driver, page.key, page.tokenCount*page.keyWidth, page.tokenCount)
		if err != nil {
			return nil, core.E("rocm.KVCache.DeviceSnapshot", "copy KV key page", err)
		}
		value, err := copyROCmDeviceKVTensorRowsToHost(cache.driver, page.value, page.tokenCount*page.valueWidth, page.tokenCount)
		if err != nil {
			return nil, core.E("rocm.KVCache.DeviceSnapshot", "copy KV value page", err)
		}
		if err := host.validateVectorShape(page.keyWidth, page.valueWidth); err != nil {
			return nil, err
		}
		block := rocmKVCacheBlock{
			tokenStart: page.tokenStart,
			tokenCount: page.tokenCount,
			keyWidth:   page.keyWidth,
			valueWidth: page.valueWidth,
			key:        key,
			value:      value,
		}
		host.blocks, err = insertROCmKVCacheBlock(host.blocks, block)
		if err != nil {
			return nil, err
		}
		host.setVectorShape(page.keyWidth, page.valueWidth)
	}
	return host, nil
}

func (cache *rocmDeviceKVCache) LastVectorWidths() (int, int, bool) {
	if cache == nil || len(cache.pages) == 0 {
		return 0, 0, false
	}
	page := cache.pages[len(cache.pages)-1]
	return page.keyWidth, page.valueWidth, true
}

func (cache *rocmDeviceKVCache) CompatibleWith(host *rocmKVCache) error {
	if cache == nil {
		return nil
	}
	if cache.closed {
		return core.E("rocm.KVCache.DeviceMirror", "device KV cache is closed", nil)
	}
	if host == nil {
		return core.E("rocm.KVCache.DeviceMirror", "package KV cache is nil", nil)
	}
	if cache.mode != host.mode {
		return core.E("rocm.KVCache.DeviceMirror", "cache mode mismatch", nil)
	}
	if cache.blockSize != host.blockSize {
		return core.E("rocm.KVCache.DeviceMirror", "cache block size mismatch", nil)
	}
	if cache.PageCount() != host.PageCount() {
		return core.E("rocm.KVCache.DeviceMirror", "page count mismatch", nil)
	}
	if cache.TokenCount() != host.TokenCount() {
		return core.E("rocm.KVCache.DeviceMirror", "token count mismatch", nil)
	}
	keyWidth, valueWidth, ok := cache.LastVectorWidths()
	hostKeyWidth, hostValueWidth, hostOK := host.LastVectorWidths()
	if ok != hostOK || keyWidth != hostKeyWidth || valueWidth != hostValueWidth {
		return core.E("rocm.KVCache.DeviceMirror", "KV vector width mismatch", nil)
	}
	return nil
}

func (cache *rocmDeviceKVCache) KernelDescriptor() (rocmDeviceKVDescriptor, error) {
	if cache == nil {
		return rocmDeviceKVDescriptor{}, core.E("rocm.KVCache.DeviceMirror", "device KV cache is nil", nil)
	}
	if cache.closed {
		return rocmDeviceKVDescriptor{}, core.E("rocm.KVCache.DeviceMirror", "device KV cache is closed", nil)
	}
	if len(cache.pages) == 0 {
		return rocmDeviceKVDescriptor{}, core.E("rocm.KVCache.DeviceMirror", "device KV cache has no pages", nil)
	}
	descriptor := rocmDeviceKVDescriptor{
		Mode:       cache.mode,
		BlockSize:  cache.blockSize,
		TokenCount: cache.TokenCount(),
		Pages:      make([]rocmDeviceKVPageDescriptor, 0, len(cache.pages)),
	}
	for _, page := range cache.pages {
		if page.key.pointer == 0 || page.value.pointer == 0 {
			return rocmDeviceKVDescriptor{}, core.E("rocm.KVCache.DeviceMirror", "device KV page has nil pointer", nil)
		}
		descriptor.Pages = append(descriptor.Pages, rocmDeviceKVPageDescriptor{
			TokenStart:    page.tokenStart,
			TokenCount:    page.tokenCount,
			KeyWidth:      page.keyWidth,
			ValueWidth:    page.valueWidth,
			KeyPointer:    page.key.pointer,
			ValuePointer:  page.value.pointer,
			KeyBytes:      page.key.sizeBytes,
			ValueBytes:    page.value.sizeBytes,
			KeyEncoding:   page.key.encoding,
			ValueEncoding: page.value.encoding,
		})
	}
	return descriptor, nil
}

func (cache *rocmDeviceKVCache) KernelDescriptorBytes() ([]byte, error) {
	return cache.kernelDescriptorBytesInto(nil)
}

func (cache *rocmDeviceKVCache) kernelDescriptorBytesInto(payload []byte) ([]byte, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.DeviceMirror", "device KV cache is nil", nil)
	}
	if cache.closed {
		return nil, core.E("rocm.KVCache.DeviceMirror", "device KV cache is closed", nil)
	}
	if len(cache.pages) == 0 {
		return nil, core.E("rocm.KVCache.DeviceMirror", "device KV cache has no pages", nil)
	}
	modeCode, err := rocmDeviceKVModeCode(cache.mode)
	if err != nil {
		return nil, err
	}
	pageCount, err := rocmDeviceKVUint32("page count", len(cache.pages))
	if err != nil {
		return nil, err
	}
	blockSize, err := rocmDeviceKVPositiveUint32("block size", cache.blockSize)
	if err != nil {
		return nil, err
	}
	tokenCount, err := rocmDeviceKVUint64("token count", cache.TokenCount())
	if err != nil {
		return nil, err
	}
	if tokenCount == 0 {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "token count must be positive", nil)
	}
	descriptorBytes := rocmDeviceKVDescriptorHeaderBytes + len(cache.pages)*rocmDeviceKVDescriptorPageBytes
	if cap(payload) < descriptorBytes {
		payload = make([]byte, descriptorBytes)
	} else {
		payload = payload[:descriptorBytes]
	}
	binary.LittleEndian.PutUint32(payload[0:], rocmDeviceKVDescriptorVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(rocmDeviceKVDescriptorHeaderBytes))
	binary.LittleEndian.PutUint32(payload[8:], uint32(rocmDeviceKVDescriptorPageBytes))
	binary.LittleEndian.PutUint32(payload[12:], modeCode)
	binary.LittleEndian.PutUint32(payload[16:], pageCount)
	binary.LittleEndian.PutUint32(payload[20:], blockSize)
	binary.LittleEndian.PutUint64(payload[24:], tokenCount)

	var lastPageEnd uint64
	for index, page := range cache.pages {
		offset := rocmDeviceKVDescriptorHeaderBytes + index*rocmDeviceKVDescriptorPageBytes
		tokenStart, err := rocmDeviceKVUint64("page token start", page.tokenStart)
		if err != nil {
			return nil, err
		}
		pageTokenCount, err := rocmDeviceKVUint64("page token count", page.tokenCount)
		if err != nil {
			return nil, err
		}
		if pageTokenCount == 0 {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "page token count must be positive", nil)
		}
		pageEnd := tokenStart + pageTokenCount
		if pageEnd > tokenCount {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "page token range exceeds descriptor token count", nil)
		}
		if index > 0 && tokenStart < lastPageEnd {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV descriptor pages must be sorted and non-overlapping", nil)
		}
		lastPageEnd = pageEnd
		keyWidth, err := rocmDeviceKVPositiveUint32("page key width", page.keyWidth)
		if err != nil {
			return nil, err
		}
		valueWidth, err := rocmDeviceKVPositiveUint32("page value width", page.valueWidth)
		if err != nil {
			return nil, err
		}
		keyEncoding, err := rocmDeviceKVEncodingCode(page.key.encoding)
		if err != nil {
			return nil, err
		}
		valueEncoding, err := rocmDeviceKVEncodingCode(page.value.encoding)
		if err != nil {
			return nil, err
		}
		if page.key.pointer == 0 || page.value.pointer == 0 {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV descriptor page has nil pointer", nil)
		}
		if page.key.sizeBytes == 0 || page.value.sizeBytes == 0 {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV descriptor page has empty tensor bytes", nil)
		}
		binary.LittleEndian.PutUint64(payload[offset:], tokenStart)
		binary.LittleEndian.PutUint64(payload[offset+8:], pageTokenCount)
		binary.LittleEndian.PutUint32(payload[offset+16:], keyWidth)
		binary.LittleEndian.PutUint32(payload[offset+20:], valueWidth)
		binary.LittleEndian.PutUint32(payload[offset+24:], keyEncoding)
		binary.LittleEndian.PutUint32(payload[offset+28:], valueEncoding)
		binary.LittleEndian.PutUint64(payload[offset+32:], uint64(page.key.pointer))
		binary.LittleEndian.PutUint64(payload[offset+40:], uint64(page.value.pointer))
		binary.LittleEndian.PutUint64(payload[offset+48:], page.key.sizeBytes)
		binary.LittleEndian.PutUint64(payload[offset+56:], page.value.sizeBytes)
	}
	return payload, nil
}

func (cache *rocmDeviceKVCache) KernelDescriptorTable() (*rocmDeviceKVDescriptorTable, error) {
	return cache.kernelDescriptorTableLabeled("", "")
}

func (cache *rocmDeviceKVCache) kernelDescriptorTableLabeled(operation, label string) (*rocmDeviceKVDescriptorTable, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV cache is nil", nil)
	}
	if cache.driver == nil {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "HIP driver is nil", nil)
	}
	if !cache.driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "HIP driver is not available", nil)
	}
	if len(cache.pages) == 1 {
		return cache.kernelSinglePageDescriptorTableOnDevice()
	}
	payloadLength := rocmDeviceKVDescriptorHeaderBytes + len(cache.pages)*rocmDeviceKVDescriptorPageBytes
	payload := rocmDeviceKVBorrowDescriptorBytes(payloadLength)
	payload, err := cache.kernelDescriptorBytesInto(payload)
	if err != nil {
		rocmDeviceKVReleaseDescriptorBytes(payload)
		return nil, err
	}
	sizeBytes := uint64(len(payload))
	allocationBytes := sizeBytes
	poolable := sizeBytes >= rocmDeviceKVDescriptorHotTableBytes() || rocmDeviceKVDescriptorExactPointerPoolable(sizeBytes)
	var pointer nativeDevicePointer
	if sizeBytes >= rocmDeviceKVDescriptorHotTableBytes() {
		pointer, allocationBytes, err = rocmDeviceKVDescriptorTableMalloc(cache.driver, sizeBytes)
		if err != nil {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "allocate descriptor table", err)
		}
	} else if rocmDeviceKVDescriptorExactPointerPoolable(sizeBytes) {
		pointer, allocationBytes, err = rocmDeviceKVDescriptorTableMallocExact(cache.driver, sizeBytes)
		if err != nil {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "allocate descriptor table", err)
		}
	} else {
		pointer, err = hipMallocLabeled(cache.driver, "rocm.KVCache.DeviceDescriptor", "KV descriptor table", sizeBytes)
		if err != nil {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "allocate descriptor table", err)
		}
	}
	if err := hipCopyPinnedHostToDeviceLabeled(cache.driver, pointer, payload, operation, label); err != nil {
		rocmDeviceKVReleaseDescriptorBytes(payload)
		if poolable {
			_ = rocmDeviceKVDescriptorTableFree(cache.driver, pointer, allocationBytes)
		} else {
			_ = cache.driver.Free(pointer)
		}
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "copy descriptor table", err)
	}
	rocmDeviceKVReleaseDescriptorBytes(payload)
	return rocmBorrowDeviceKVDescriptorTableAllocated(cache.driver, pointer, sizeBytes, allocationBytes, rocmDeviceKVDescriptorVersion, cache.PageCount(), false, poolable), nil
}

func (cache *rocmDeviceKVCache) kernelSinglePageDescriptorTableOnDevice() (*rocmDeviceKVDescriptorTable, error) {
	return cache.kernelSinglePageDescriptorTableOnDeviceWithWorkspace(nil)
}

func (cache *rocmDeviceKVCache) kernelSinglePageDescriptorTableOnDeviceWithWorkspace(workspace *hipAttentionHeadsChunkedWorkspace) (*rocmDeviceKVDescriptorTable, error) {
	if cache == nil || len(cache.pages) != 1 {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "single-page device descriptor requires one page", nil)
	}
	page := cache.pages[0]
	if page.tokenStart != 0 || page.tokenCount != cache.TokenCount() || page.tokenCount <= 0 {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "single-page device descriptor cache shape mismatch", nil)
	}
	if page.key.pointer == 0 || page.value.pointer == 0 || page.key.sizeBytes == 0 || page.value.sizeBytes == 0 {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "single-page device descriptor page is empty", nil)
	}
	modeCode, err := rocmDeviceKVModeCode(cache.mode)
	if err != nil {
		return nil, err
	}
	keyEncoding, err := rocmDeviceKVEncodingCode(page.key.encoding)
	if err != nil {
		return nil, err
	}
	valueEncoding, err := rocmDeviceKVEncodingCode(page.value.encoding)
	if err != nil {
		return nil, err
	}
	sizeBytes := uint64(rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVDescriptorPageBytes)
	pointer, allocationBytes, err := rocmDeviceKVDescriptorTableMallocExact(cache.driver, sizeBytes)
	if err != nil {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "allocate single-page descriptor table", err)
	}
	launchArgs := hipKVDescriptorAppendLaunchArgs{
		OutputDescriptorPointer: pointer,
		NewKeyPointer:           page.key.pointer,
		NewValuePointer:         page.value.pointer,
		OutputDescriptorBytes:   sizeBytes,
		NewKeyBytes:             page.key.sizeBytes,
		NewValueBytes:           page.value.sizeBytes,
		ModeCode:                modeCode,
		BlockSize:               cache.blockSize,
		OutputPageCount:         1,
		OutputTokenCount:        cache.TokenCount(),
		KeyWidth:                page.keyWidth,
		ValueWidth:              page.valueWidth,
		NewKeyEncoding:          keyEncoding,
		NewValueEncoding:        valueEncoding,
		Reserved0:               rocmKVDescriptorAppendModeBuildSinglePage,
	}
	var args []byte
	if workspace != nil {
		args, err = launchArgs.BinaryInto(workspace.KVDescriptorAppendArgs[:])
	} else {
		args, err = launchArgs.Binary()
	}
	if err != nil {
		_ = rocmDeviceKVDescriptorTableFree(cache.driver, pointer, allocationBytes)
		return nil, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameKVDescriptorAppend,
		Args:   args,
		GridX:  1,
		GridY:  1,
		GridZ:  1,
		BlockX: hipKVDescriptorAppendBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	if err := hipLaunchKernel(cache.driver, config); err != nil {
		_ = rocmDeviceKVDescriptorTableFree(cache.driver, pointer, allocationBytes)
		return nil, err
	}
	return rocmBorrowDeviceKVDescriptorTableAllocated(cache.driver, pointer, sizeBytes, allocationBytes, rocmDeviceKVDescriptorVersion, 1, false, true), nil
}

func (cache *rocmDeviceKVCache) KernelDescriptorTableFromAppendedToken(ctx context.Context, previous *rocmDeviceKVCache, previousTable *rocmDeviceKVDescriptorTable) (*rocmDeviceKVDescriptorTable, error) {
	return cache.KernelDescriptorTableFromAppendedTokenWithWorkspace(ctx, previous, previousTable, nil)
}

func (cache *rocmDeviceKVCache) KernelDescriptorTableFromAppendedTokenWithWorkspace(ctx context.Context, previous *rocmDeviceKVCache, previousTable *rocmDeviceKVDescriptorTable, workspace *hipAttentionHeadsChunkedWorkspace) (*rocmDeviceKVDescriptorTable, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if cache == nil || previous == nil {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV append descriptor caches are required", nil)
	}
	if cache.closed || previous.closed {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV append descriptor cache is closed", nil)
	}
	if cache.driver == nil || !cache.driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "HIP driver is not available", nil)
	}
	if cache.driver != previous.driver || cache.mode != previous.mode || cache.blockSize != previous.blockSize {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV append descriptor cache shape mismatch", nil)
	}
	if previousTable == nil {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "previous descriptor table is required", nil)
	}
	if err := previousTable.CompatibleWith(previous); err != nil {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "previous descriptor table does not match device KV cache", err)
	}
	growTrimStart, growLastPage := rocmDeviceKVGrowsDescriptorLastPageWithTrim(previous, cache)
	trimStart, copiedPages := 0, 0
	if growLastPage {
		trimStart = growTrimStart
		copiedPages = cache.PageCount()
	} else {
		var ok bool
		trimStart, copiedPages, ok = rocmDeviceKVAppendDescriptorShape(previous, cache)
		if !ok {
			return cache.kernelDescriptorTableLabeled("rocm.KVCache.DeviceDescriptor", "append_rebuild")
		}
	}
	if !growLastPage && copiedPages+1 != cache.PageCount() {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV append descriptor page count mismatch", nil)
	}
	lastPage := cache.pages[len(cache.pages)-1]
	modeCode, err := rocmDeviceKVModeCode(cache.mode)
	if err != nil {
		return nil, err
	}
	keyEncoding, err := rocmDeviceKVEncodingCode(lastPage.key.encoding)
	if err != nil {
		return nil, err
	}
	valueEncoding, err := rocmDeviceKVEncodingCode(lastPage.value.encoding)
	if err != nil {
		return nil, err
	}
	outputBytes := uint64(rocmDeviceKVDescriptorHeaderBytes + cache.PageCount()*rocmDeviceKVDescriptorPageBytes)
	pointer := previousTable.Pointer()
	allocationBytes := previousTable.AllocationBytes()
	inPlace := !previousTable.borrowed && pointer != 0 && allocationBytes >= outputBytes
	if !inPlace {
		var err error
		pointer, allocationBytes, err = rocmDeviceKVDescriptorTableMalloc(cache.driver, outputBytes)
		if err != nil {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "allocate appended descriptor table", err)
		}
	}
	appendMode := uint64(0)
	if growLastPage {
		appendMode = rocmKVDescriptorAppendModeGrowLastPage
	}
	launchArgs := hipKVDescriptorAppendLaunchArgs{
		PreviousDescriptorPointer: previousTable.Pointer(),
		OutputDescriptorPointer:   pointer,
		NewKeyPointer:             lastPage.key.pointer,
		NewValuePointer:           lastPage.value.pointer,
		PreviousDescriptorBytes:   previousTable.SizeBytes(),
		OutputDescriptorBytes:     outputBytes,
		NewKeyBytes:               lastPage.key.sizeBytes,
		NewValueBytes:             lastPage.value.sizeBytes,
		ModeCode:                  modeCode,
		BlockSize:                 cache.blockSize,
		OutputPageCount:           cache.PageCount(),
		OutputTokenCount:          cache.TokenCount(),
		KeyWidth:                  lastPage.keyWidth,
		ValueWidth:                lastPage.valueWidth,
		NewKeyEncoding:            keyEncoding,
		NewValueEncoding:          valueEncoding,
		TrimStart:                 trimStart,
		Reserved0:                 appendMode,
	}
	var args []byte
	if workspace != nil {
		args, err = launchArgs.BinaryInto(workspace.KVDescriptorAppendArgs[:])
	} else {
		args, err = launchArgs.Binary()
	}
	if err != nil {
		if !inPlace {
			_ = rocmDeviceKVDescriptorTableFree(cache.driver, pointer, allocationBytes)
		}
		return nil, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameKVDescriptorAppend,
		Args:   args,
		GridX:  1,
		GridY:  1,
		GridZ:  1,
		BlockX: hipKVDescriptorAppendBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	if err := hipLaunchKernel(cache.driver, config); err != nil {
		if !inPlace {
			_ = rocmDeviceKVDescriptorTableFree(cache.driver, pointer, allocationBytes)
		}
		return nil, err
	}
	if inPlace {
		previousTable.sizeBytes = outputBytes
		previousTable.pageCount = cache.PageCount()
		previousTable.version = rocmDeviceKVDescriptorVersion
		return previousTable, nil
	}
	return rocmBorrowDeviceKVDescriptorTableAllocated(cache.driver, pointer, outputBytes, allocationBytes, rocmDeviceKVDescriptorVersion, cache.PageCount(), false, true), nil
}

func rocmDeviceKVAppendDescriptorShape(previous, next *rocmDeviceKVCache) (int, int, bool) {
	if previous == nil || next == nil || len(previous.pages) == 0 || len(next.pages) == 0 {
		return 0, 0, false
	}
	lastPage := next.pages[len(next.pages)-1]
	if lastPage.tokenCount <= 0 || lastPage.tokenStart+lastPage.tokenCount != next.TokenCount() {
		return 0, 0, false
	}
	if previous.TokenCount()+lastPage.tokenCount < next.TokenCount() {
		return 0, 0, false
	}
	trimStart := previous.TokenCount() + lastPage.tokenCount - next.TokenCount()
	copiedPages := 0
	for _, page := range previous.pages {
		retainedPage, retained := rocmDeviceKVPageAfterTrim(page, trimStart)
		if !retained {
			continue
		}
		if copiedPages >= len(next.pages)-1 {
			return 0, 0, false
		}
		nextPage := next.pages[copiedPages]
		if !rocmDeviceKVPageShapeEqual(retainedPage, nextPage) {
			return 0, 0, false
		}
		copiedPages++
	}
	if copiedPages != len(next.pages)-1 {
		return 0, 0, false
	}
	return trimStart, copiedPages, true
}

func rocmDeviceKVPageAfterTrim(page rocmDeviceKVPage, trimStart int) (rocmDeviceKVPage, bool) {
	pageEnd := page.tokenStart + page.tokenCount
	if pageEnd <= trimStart {
		return rocmDeviceKVPage{}, false
	}
	if page.tokenStart < trimStart {
		return rocmDeviceKVSliceInterleavedPage(page, trimStart)
	}
	page.tokenStart -= trimStart
	return page, true
}

func rocmDeviceKVGrowsDescriptorLastPageWithTrim(previous, next *rocmDeviceKVCache) (int, bool) {
	if previous == nil || next == nil || len(previous.pages) == 0 || len(next.pages) == 0 {
		return 0, false
	}
	if previous.driver != next.driver || previous.mode != next.mode || previous.blockSize != next.blockSize {
		return 0, false
	}
	if previous.blockSize <= 1 {
		return 0, false
	}
	nextLast := next.pages[len(next.pages)-1]
	if nextLast.tokenStart >= previous.TokenCount() || nextLast.tokenStart+nextLast.tokenCount != next.TokenCount() {
		return 0, false
	}
	maxAppendCount := nextLast.tokenCount
	if maxAppendCount > previous.blockSize {
		maxAppendCount = previous.blockSize
	}
	lastIndex := len(previous.pages) - 1
	outputLastIndex := len(next.pages) - 1
	for appendCount := 1; appendCount <= maxAppendCount; appendCount++ {
		if previous.TokenCount()+appendCount < next.TokenCount() {
			continue
		}
		trimStart := previous.TokenCount() + appendCount - next.TokenCount()
		outputIndex := 0
		for index := 0; index < lastIndex; index++ {
			retainedPage, retained := rocmDeviceKVPageAfterTrim(previous.pages[index], trimStart)
			if !retained {
				continue
			}
			if outputIndex >= outputLastIndex || !rocmDeviceKVPageShapeEqual(retainedPage, next.pages[outputIndex]) {
				outputIndex = -1
				break
			}
			outputIndex++
		}
		if outputIndex != outputLastIndex {
			continue
		}
		prevLast, retained := rocmDeviceKVPageAfterTrim(previous.pages[lastIndex], trimStart)
		if !retained {
			continue
		}
		if prevLast.tokenStart != nextLast.tokenStart ||
			prevLast.tokenCount+appendCount != nextLast.tokenCount ||
			prevLast.keyWidth != nextLast.keyWidth || prevLast.valueWidth != nextLast.valueWidth ||
			prevLast.key.pointer != nextLast.key.pointer || prevLast.value.pointer != nextLast.value.pointer ||
			prevLast.key.encoding != nextLast.key.encoding || prevLast.value.encoding != nextLast.value.encoding {
			continue
		}
		return trimStart, nextLast.key.sizeBytes > prevLast.key.sizeBytes && nextLast.value.sizeBytes > prevLast.value.sizeBytes
	}
	return 0, false
}

func rocmDeviceKVGrowsDescriptorLastPage(previous, next *rocmDeviceKVCache) bool {
	trimStart, ok := rocmDeviceKVGrowsDescriptorLastPageWithTrim(previous, next)
	return ok && trimStart == 0
}

func rocmDeviceKVPageShapeEqual(left, right rocmDeviceKVPage) bool {
	return left.tokenStart == right.tokenStart &&
		left.tokenCount == right.tokenCount &&
		left.keyWidth == right.keyWidth &&
		left.valueWidth == right.valueWidth &&
		left.key.pointer == right.key.pointer &&
		left.value.pointer == right.value.pointer &&
		left.key.sizeBytes == right.key.sizeBytes &&
		left.value.sizeBytes == right.value.sizeBytes &&
		left.key.encoding == right.key.encoding &&
		left.value.encoding == right.value.encoding
}

func (table *rocmDeviceKVDescriptorTable) Pointer() nativeDevicePointer {
	if table == nil || table.closed {
		return 0
	}
	return table.pointer
}

func (table *rocmDeviceKVDescriptorTable) SizeBytes() uint64 {
	if table == nil || table.closed {
		return 0
	}
	return table.sizeBytes
}

func (table *rocmDeviceKVDescriptorTable) AllocationBytes() uint64 {
	if table == nil || table.closed {
		return 0
	}
	if table.allocationBytes != 0 {
		return table.allocationBytes
	}
	return table.sizeBytes
}

func (table *rocmDeviceKVDescriptorTable) CompatibleWith(cache *rocmDeviceKVCache) error {
	if table == nil {
		return nil
	}
	if table.closed || table.pointer == 0 {
		return core.E("rocm.KVCache.DeviceDescriptor", "descriptor table is closed", nil)
	}
	if cache == nil {
		return core.E("rocm.KVCache.DeviceDescriptor", "device KV cache is nil", nil)
	}
	if cache.closed {
		return core.E("rocm.KVCache.DeviceDescriptor", "device KV cache is closed", nil)
	}
	if table.version != rocmDeviceKVDescriptorVersion {
		return core.E("rocm.KVCache.DeviceDescriptor", "descriptor table version mismatch", nil)
	}
	if table.pageCount != cache.PageCount() {
		return core.E("rocm.KVCache.DeviceDescriptor", "descriptor table page count mismatch", nil)
	}
	expectedBytes := uint64(rocmDeviceKVDescriptorHeaderBytes + table.pageCount*rocmDeviceKVDescriptorPageBytes)
	if table.sizeBytes != expectedBytes {
		return core.E("rocm.KVCache.DeviceDescriptor", "descriptor table size mismatch", nil)
	}
	return nil
}

func (cache *rocmDeviceKVCache) KernelLaunchDescriptor(table *rocmDeviceKVDescriptorTable) (rocmDeviceKVLaunchDescriptor, error) {
	if cache == nil {
		return rocmDeviceKVLaunchDescriptor{}, core.E("rocm.KVCache.DeviceLaunch", "device KV cache is nil", nil)
	}
	if cache.closed {
		return rocmDeviceKVLaunchDescriptor{}, core.E("rocm.KVCache.DeviceLaunch", "device KV cache is closed", nil)
	}
	if table == nil {
		return rocmDeviceKVLaunchDescriptor{}, core.E("rocm.KVCache.DeviceLaunch", "descriptor table is required", nil)
	}
	if err := table.CompatibleWith(cache); err != nil {
		return rocmDeviceKVLaunchDescriptor{}, core.E("rocm.KVCache.DeviceLaunch", "descriptor table does not match device KV cache", err)
	}
	modeCode, err := rocmDeviceKVModeCode(cache.mode)
	if err != nil {
		return rocmDeviceKVLaunchDescriptor{}, err
	}
	keyWidth, valueWidth, ok := cache.LastVectorWidths()
	if !ok {
		return rocmDeviceKVLaunchDescriptor{}, core.E("rocm.KVCache.DeviceLaunch", "device KV cache has no pages", nil)
	}
	return rocmDeviceKVLaunchDescriptor{
		DescriptorPointer: table.Pointer(),
		DescriptorBytes:   table.SizeBytes(),
		DescriptorVersion: table.version,
		Mode:              cache.mode,
		ModeCode:          modeCode,
		BlockSize:         cache.blockSize,
		PageCount:         cache.PageCount(),
		TokenCount:        cache.TokenCount(),
		KeyWidth:          keyWidth,
		ValueWidth:        valueWidth,
	}, nil
}

func (launch rocmDeviceKVLaunchDescriptor) Binary() ([]byte, error) {
	return launch.BinaryInto(nil)
}

func (launch rocmDeviceKVLaunchDescriptor) BinaryInto(payload []byte) ([]byte, error) {
	if launch.DescriptorPointer == 0 {
		return nil, core.E("rocm.KVCache.DeviceLaunch", "descriptor pointer is nil", nil)
	}
	if launch.DescriptorBytes == 0 {
		return nil, core.E("rocm.KVCache.DeviceLaunch", "descriptor bytes must be positive", nil)
	}
	if launch.DescriptorVersion != rocmDeviceKVDescriptorVersion {
		return nil, core.E("rocm.KVCache.DeviceLaunch", "descriptor version mismatch", nil)
	}
	if err := rocmDeviceKVValidateModeCode(launch.ModeCode); err != nil {
		return nil, err
	}
	if launch.Mode != "" {
		modeCode, err := rocmDeviceKVModeCode(launch.Mode)
		if err != nil {
			return nil, err
		}
		if modeCode != launch.ModeCode {
			return nil, core.E("rocm.KVCache.DeviceLaunch", "mode code mismatch", nil)
		}
	}
	blockSize, err := rocmDeviceKVPositiveUint32("block size", launch.BlockSize)
	if err != nil {
		return nil, err
	}
	pageCount, err := rocmDeviceKVPositiveUint32("page count", launch.PageCount)
	if err != nil {
		return nil, err
	}
	tokenCount, err := rocmDeviceKVUint64("token count", launch.TokenCount)
	if err != nil {
		return nil, err
	}
	if tokenCount == 0 {
		return nil, core.E("rocm.KVCache.DeviceLaunch", "token count must be positive", nil)
	}
	keyWidth, err := rocmDeviceKVPositiveUint32("key width", launch.KeyWidth)
	if err != nil {
		return nil, err
	}
	valueWidth, err := rocmDeviceKVPositiveUint32("value width", launch.ValueWidth)
	if err != nil {
		return nil, err
	}
	if cap(payload) < rocmDeviceKVLaunchDescriptorBytes {
		payload = hipBorrowLaunchPacket(rocmDeviceKVLaunchDescriptorBytes)
	} else {
		payload = payload[:rocmDeviceKVLaunchDescriptorBytes]
		clear(payload)
	}
	statusValue := launch.StatusValue
	if launch.StatusPointer != 0 && statusValue == 0 {
		statusValue = hipDecodeLaunchStatusOK
	}
	binary.LittleEndian.PutUint64(payload[0:], uint64(launch.DescriptorPointer))
	binary.LittleEndian.PutUint64(payload[8:], launch.DescriptorBytes)
	binary.LittleEndian.PutUint32(payload[16:], launch.DescriptorVersion)
	binary.LittleEndian.PutUint32(payload[20:], launch.ModeCode)
	binary.LittleEndian.PutUint32(payload[24:], blockSize)
	binary.LittleEndian.PutUint32(payload[28:], pageCount)
	binary.LittleEndian.PutUint64(payload[32:], tokenCount)
	binary.LittleEndian.PutUint32(payload[40:], keyWidth)
	binary.LittleEndian.PutUint32(payload[44:], valueWidth)
	binary.LittleEndian.PutUint64(payload[48:], uint64(launch.StatusPointer))
	binary.LittleEndian.PutUint32(payload[56:], statusValue)
	return payload, nil
}

func (table *rocmDeviceKVDescriptorTable) Close() error {
	if table == nil || table.closed {
		return nil
	}
	if table.borrowed {
		if table.poolable {
			rocmReleaseDeviceKVDescriptorTable(table)
			return nil
		}
		table.closed = true
		return nil
	}
	if table.pointer != 0 {
		if err := rocmDeviceKVDescriptorTableFree(table.driver, table.pointer, table.AllocationBytes()); err != nil {
			return core.E("rocm.KVCache.DeviceDescriptor", "free descriptor table", err)
		}
		table.pointer = 0
	}
	if table.poolable {
		rocmReleaseDeviceKVDescriptorTable(table)
		return nil
	}
	table.closed = true
	return nil
}

func (table *rocmDeviceKVDescriptorTable) borrowedAlias() (*rocmDeviceKVDescriptorTable, error) {
	if table == nil {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "descriptor table is nil", nil)
	}
	if table.closed || table.pointer == 0 {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "descriptor table is closed", nil)
	}
	return rocmBorrowDeviceKVDescriptorTable(table.driver, table.pointer, table.sizeBytes, table.version, table.pageCount, true, true), nil
}

func (descriptor rocmDeviceKVDescriptor) Binary() ([]byte, error) {
	if len(descriptor.Pages) == 0 {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV descriptor has no pages", nil)
	}
	modeCode, err := rocmDeviceKVModeCode(descriptor.Mode)
	if err != nil {
		return nil, err
	}
	pageCount, err := rocmDeviceKVUint32("page count", len(descriptor.Pages))
	if err != nil {
		return nil, err
	}
	blockSize, err := rocmDeviceKVPositiveUint32("block size", descriptor.BlockSize)
	if err != nil {
		return nil, err
	}
	tokenCount, err := rocmDeviceKVUint64("token count", descriptor.TokenCount)
	if err != nil {
		return nil, err
	}
	if tokenCount == 0 {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "token count must be positive", nil)
	}
	payload := make([]byte, rocmDeviceKVDescriptorHeaderBytes+len(descriptor.Pages)*rocmDeviceKVDescriptorPageBytes)
	binary.LittleEndian.PutUint32(payload[0:], rocmDeviceKVDescriptorVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(rocmDeviceKVDescriptorHeaderBytes))
	binary.LittleEndian.PutUint32(payload[8:], uint32(rocmDeviceKVDescriptorPageBytes))
	binary.LittleEndian.PutUint32(payload[12:], modeCode)
	binary.LittleEndian.PutUint32(payload[16:], pageCount)
	binary.LittleEndian.PutUint32(payload[20:], blockSize)
	binary.LittleEndian.PutUint64(payload[24:], tokenCount)

	var lastPageEnd uint64
	for index, page := range descriptor.Pages {
		offset := rocmDeviceKVDescriptorHeaderBytes + index*rocmDeviceKVDescriptorPageBytes
		tokenStart, err := rocmDeviceKVUint64("page token start", page.TokenStart)
		if err != nil {
			return nil, err
		}
		pageTokenCount, err := rocmDeviceKVUint64("page token count", page.TokenCount)
		if err != nil {
			return nil, err
		}
		if pageTokenCount == 0 {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "page token count must be positive", nil)
		}
		pageEnd := tokenStart + pageTokenCount
		if pageEnd > tokenCount {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "page token range exceeds descriptor token count", nil)
		}
		if index > 0 && tokenStart < lastPageEnd {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV descriptor pages must be sorted and non-overlapping", nil)
		}
		lastPageEnd = pageEnd
		keyWidth, err := rocmDeviceKVPositiveUint32("page key width", page.KeyWidth)
		if err != nil {
			return nil, err
		}
		valueWidth, err := rocmDeviceKVPositiveUint32("page value width", page.ValueWidth)
		if err != nil {
			return nil, err
		}
		keyEncoding, err := rocmDeviceKVEncodingCode(page.KeyEncoding)
		if err != nil {
			return nil, err
		}
		valueEncoding, err := rocmDeviceKVEncodingCode(page.ValueEncoding)
		if err != nil {
			return nil, err
		}
		if page.KeyPointer == 0 || page.ValuePointer == 0 {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV descriptor page has nil pointer", nil)
		}
		if page.KeyBytes == 0 || page.ValueBytes == 0 {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV descriptor page has empty tensor bytes", nil)
		}
		binary.LittleEndian.PutUint64(payload[offset:], tokenStart)
		binary.LittleEndian.PutUint64(payload[offset+8:], pageTokenCount)
		binary.LittleEndian.PutUint32(payload[offset+16:], keyWidth)
		binary.LittleEndian.PutUint32(payload[offset+20:], valueWidth)
		binary.LittleEndian.PutUint32(payload[offset+24:], keyEncoding)
		binary.LittleEndian.PutUint32(payload[offset+28:], valueEncoding)
		binary.LittleEndian.PutUint64(payload[offset+32:], uint64(page.KeyPointer))
		binary.LittleEndian.PutUint64(payload[offset+40:], uint64(page.ValuePointer))
		binary.LittleEndian.PutUint64(payload[offset+48:], page.KeyBytes)
		binary.LittleEndian.PutUint64(payload[offset+56:], page.ValueBytes)
	}
	return payload, nil
}

func rocmDeviceKVModeCode(mode string) (uint32, error) {
	switch mode {
	case rocmKVCacheModeFP16:
		return rocmDeviceKVDescriptorModeFP16, nil
	case rocmKVCacheModeQ8:
		return rocmDeviceKVDescriptorModeQ8, nil
	case rocmKVCacheModeKQ8VQ4:
		return rocmDeviceKVDescriptorModeKQ8VQ4, nil
	default:
		return 0, core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("unsupported cache mode %q", mode), nil)
	}
}

func rocmDeviceKVValidateModeCode(code uint32) error {
	switch code {
	case rocmDeviceKVDescriptorModeFP16, rocmDeviceKVDescriptorModeQ8, rocmDeviceKVDescriptorModeKQ8VQ4:
		return nil
	default:
		return core.E("rocm.KVCache.DeviceLaunch", core.Sprintf("unsupported cache mode code %d", code), nil)
	}
}

func rocmDeviceKVEncodingCode(encoding string) (uint32, error) {
	switch encoding {
	case rocmKVEncodingFP16:
		return rocmDeviceKVDescriptorEncodingFP16, nil
	case rocmKVEncodingQ8:
		return rocmDeviceKVDescriptorEncodingQ8, nil
	case rocmKVEncodingQ4:
		return rocmDeviceKVDescriptorEncodingQ4, nil
	case rocmKVEncodingQ8Rows:
		return rocmDeviceKVDescriptorEncodingQ8Rows, nil
	case rocmKVEncodingQ4Rows:
		return rocmDeviceKVDescriptorEncodingQ4Rows, nil
	case rocmKVEncodingQ8RowsI:
		return rocmDeviceKVDescriptorEncodingQ8RowsI, nil
	case rocmKVEncodingQ4RowsI:
		return rocmDeviceKVDescriptorEncodingQ4RowsI, nil
	default:
		return 0, core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("unsupported tensor encoding %q", encoding), nil)
	}
}

func rocmDeviceKVValidateEncodingCode(code uint32) error {
	switch code {
	case rocmDeviceKVDescriptorEncodingFP16, rocmDeviceKVDescriptorEncodingQ8, rocmDeviceKVDescriptorEncodingQ4, rocmDeviceKVDescriptorEncodingQ8Rows, rocmDeviceKVDescriptorEncodingQ4Rows, rocmDeviceKVDescriptorEncodingQ8RowsI, rocmDeviceKVDescriptorEncodingQ4RowsI:
		return nil
	default:
		return core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("unsupported tensor encoding code %d", code), nil)
	}
}

func rocmDeviceKVEncodingName(code uint32) (string, error) {
	switch code {
	case rocmDeviceKVDescriptorEncodingFP16:
		return rocmKVEncodingFP16, nil
	case rocmDeviceKVDescriptorEncodingQ8:
		return rocmKVEncodingQ8, nil
	case rocmDeviceKVDescriptorEncodingQ4:
		return rocmKVEncodingQ4, nil
	case rocmDeviceKVDescriptorEncodingQ8Rows:
		return rocmKVEncodingQ8Rows, nil
	case rocmDeviceKVDescriptorEncodingQ4Rows:
		return rocmKVEncodingQ4Rows, nil
	case rocmDeviceKVDescriptorEncodingQ8RowsI:
		return rocmKVEncodingQ8RowsI, nil
	case rocmDeviceKVDescriptorEncodingQ4RowsI:
		return rocmKVEncodingQ4RowsI, nil
	default:
		return "", core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("unsupported tensor encoding code %d", code), nil)
	}
}

func rocmKVTensorDeviceByteCount(encoding string, length int) (uint64, error) {
	return rocmKVTensorDeviceByteCountRows(encoding, length, 1)
}

func rocmKVTensorDeviceByteCountRows(encoding string, length, rows int) (uint64, error) {
	if length <= 0 {
		return 0, core.E("rocm.KVCache.DeviceDescriptor", "tensor length must be positive", nil)
	}
	switch encoding {
	case rocmKVEncodingFP16:
		return uint64(length) * 2, nil
	case rocmKVEncodingQ8:
		return uint64(length) + 4, nil
	case rocmKVEncodingQ4:
		return uint64((length+1)/2) + 4, nil
	case rocmKVEncodingQ8Rows:
		if rows <= 0 {
			return 0, core.E("rocm.KVCache.DeviceDescriptor", "row-scaled tensor row count must be positive", nil)
		}
		return uint64(length) + uint64(rows)*4, nil
	case rocmKVEncodingQ4Rows:
		if rows <= 0 {
			return 0, core.E("rocm.KVCache.DeviceDescriptor", "row-scaled tensor row count must be positive", nil)
		}
		return uint64((length+1)/2) + uint64(rows)*4, nil
	case rocmKVEncodingQ8RowsI:
		if rows <= 0 || length%rows != 0 {
			return 0, core.E("rocm.KVCache.DeviceDescriptor", "interleaved row tensor shape mismatch", nil)
		}
		rowWidth := length / rows
		return uint64(rows * (4 + rowWidth)), nil
	case rocmKVEncodingQ4RowsI:
		if rows <= 0 || length%rows != 0 {
			return 0, core.E("rocm.KVCache.DeviceDescriptor", "interleaved row tensor shape mismatch", nil)
		}
		rowWidth := length / rows
		return uint64(rows * (4 + (rowWidth+1)/2)), nil
	default:
		return 0, core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("unsupported tensor encoding %q", encoding), nil)
	}
}

func rocmDeviceKVUint32Bytes(field string, value uint64) (uint32, error) {
	if value > uint64(^uint32(0)) {
		return 0, core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("%s are out of uint32 range", field), nil)
	}
	return uint32(value), nil
}

func (args hipKVEncodeTokenLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipKVEncodeTokenLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.KeyInputPointer == 0 || args.ValueInputPointer == 0 || args.KeyOutputPointer == 0 || args.ValueOutputPointer == 0 {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV encode token pointers are required", nil)
	}
	keyCount, err := rocmDeviceKVPositiveUint32("key count", args.KeyCount)
	if err != nil {
		return nil, err
	}
	valueCount, err := rocmDeviceKVPositiveUint32("value count", args.ValueCount)
	if err != nil {
		return nil, err
	}
	if args.KeyInputBytes != uint64(keyCount)*4 || args.ValueInputBytes != uint64(valueCount)*4 {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV encode token input byte count mismatch", nil)
	}
	if err := rocmDeviceKVValidateEncodingCode(args.KeyEncoding); err != nil {
		return nil, err
	}
	if err := rocmDeviceKVValidateEncodingCode(args.ValueEncoding); err != nil {
		return nil, err
	}
	keyEncoding, err := rocmDeviceKVEncodingName(args.KeyEncoding)
	if err != nil {
		return nil, err
	}
	valueEncoding, err := rocmDeviceKVEncodingName(args.ValueEncoding)
	if err != nil {
		return nil, err
	}
	tokenCount := 1
	if args.TokenCount > 0 {
		tokenCount = args.TokenCount
	}
	if args.KeyWidth > 0 || args.ValueWidth > 0 || args.TokenCount > 0 {
		if args.KeyWidth <= 0 || args.ValueWidth <= 0 || args.TokenCount <= 0 ||
			int(keyCount) != args.KeyWidth*args.TokenCount ||
			int(valueCount) != args.ValueWidth*args.TokenCount {
			return nil, core.E("rocm.KVCache.DeviceAppend", "KV encode token row shape mismatch", nil)
		}
	}
	expectedKeyBytes, err := rocmKVTensorDeviceByteCountRows(keyEncoding, int(keyCount), tokenCount)
	if err != nil {
		return nil, err
	}
	expectedValueBytes, err := rocmKVTensorDeviceByteCountRows(valueEncoding, int(valueCount), tokenCount)
	if err != nil {
		return nil, err
	}
	if args.KeyOutputBytes != expectedKeyBytes || args.ValueOutputBytes != expectedValueBytes {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV encode token output byte count mismatch", nil)
	}
	keyInputBytes, err := rocmDeviceKVUint32Bytes("key input bytes", args.KeyInputBytes)
	if err != nil {
		return nil, err
	}
	valueInputBytes, err := rocmDeviceKVUint32Bytes("value input bytes", args.ValueInputBytes)
	if err != nil {
		return nil, err
	}
	keyOutputBytes, err := rocmDeviceKVUint32Bytes("key output bytes", args.KeyOutputBytes)
	if err != nil {
		return nil, err
	}
	valueOutputBytes, err := rocmDeviceKVUint32Bytes("value output bytes", args.ValueOutputBytes)
	if err != nil {
		return nil, err
	}
	if cap(payload) < hipKVEncodeTokenLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipKVEncodeTokenLaunchArgsBytes)
	} else {
		payload = payload[:hipKVEncodeTokenLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipKVEncodeTokenLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(hipKVEncodeTokenLaunchArgsBytes))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.KeyInputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.ValueInputPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.KeyOutputPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.ValueOutputPointer))
	binary.LittleEndian.PutUint32(payload[40:], keyCount)
	binary.LittleEndian.PutUint32(payload[44:], valueCount)
	binary.LittleEndian.PutUint32(payload[48:], keyInputBytes)
	binary.LittleEndian.PutUint32(payload[52:], valueInputBytes)
	binary.LittleEndian.PutUint32(payload[56:], keyOutputBytes)
	binary.LittleEndian.PutUint32(payload[60:], valueOutputBytes)
	binary.LittleEndian.PutUint32(payload[64:], args.KeyEncoding)
	binary.LittleEndian.PutUint32(payload[68:], args.ValueEncoding)
	binary.LittleEndian.PutUint64(payload[72:], uint64(args.KeyWidth))
	binary.LittleEndian.PutUint64(payload[80:], uint64(args.ValueWidth))
	binary.LittleEndian.PutUint64(payload[88:], uint64(args.TokenCount))
	return payload, nil
}

func (args hipKVEncodeTokenValueNormLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipKVEncodeTokenValueNormLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.KeyInputPointer == 0 || args.ValueInputPointer == 0 || args.KeyOutputPointer == 0 || args.ValueOutputPointer == 0 {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV encode value norm pointers are required", nil)
	}
	keyCount, err := rocmDeviceKVPositiveUint32("key count", args.KeyCount)
	if err != nil {
		return nil, err
	}
	valueCount, err := rocmDeviceKVPositiveUint32("value count", args.ValueCount)
	if err != nil {
		return nil, err
	}
	if args.KeyInputBytes != uint64(keyCount)*4 || args.ValueInputBytes != uint64(valueCount)*4 {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV encode value norm input byte count mismatch", nil)
	}
	if err := rocmDeviceKVValidateEncodingCode(args.KeyEncoding); err != nil {
		return nil, err
	}
	if err := rocmDeviceKVValidateEncodingCode(args.ValueEncoding); err != nil {
		return nil, err
	}
	keyEncoding, err := rocmDeviceKVEncodingName(args.KeyEncoding)
	if err != nil {
		return nil, err
	}
	valueEncoding, err := rocmDeviceKVEncodingName(args.ValueEncoding)
	if err != nil {
		return nil, err
	}
	tokenCount := 1
	if args.TokenCount > 0 {
		tokenCount = args.TokenCount
	}
	if args.KeyWidth <= 0 || args.ValueWidth <= 0 || args.TokenCount <= 0 ||
		int(keyCount) != args.KeyWidth*args.TokenCount ||
		int(valueCount) != args.ValueWidth*args.TokenCount {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV encode value norm row shape mismatch", nil)
	}
	if args.ValueHeadDim <= 0 || args.ValueHeadCount <= 0 || args.ValueHeadCount > hipKVEncodeTokenValueNormMaxHeads || args.ValueWidth != args.ValueHeadDim*args.ValueHeadCount {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV encode value norm head shape mismatch", nil)
	}
	if math.IsNaN(float64(args.Epsilon)) || math.IsInf(float64(args.Epsilon), 0) || args.Epsilon < 0 {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV encode value norm epsilon must be finite and non-negative", nil)
	}
	expectedKeyBytes, err := rocmKVTensorDeviceByteCountRows(keyEncoding, int(keyCount), tokenCount)
	if err != nil {
		return nil, err
	}
	expectedValueBytes, err := rocmKVTensorDeviceByteCountRows(valueEncoding, int(valueCount), tokenCount)
	if err != nil {
		return nil, err
	}
	if args.KeyOutputBytes != expectedKeyBytes || args.ValueOutputBytes != expectedValueBytes {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV encode value norm output byte count mismatch", nil)
	}
	keyInputBytes, err := rocmDeviceKVUint32Bytes("key input bytes", args.KeyInputBytes)
	if err != nil {
		return nil, err
	}
	valueInputBytes, err := rocmDeviceKVUint32Bytes("value input bytes", args.ValueInputBytes)
	if err != nil {
		return nil, err
	}
	keyOutputBytes, err := rocmDeviceKVUint32Bytes("key output bytes", args.KeyOutputBytes)
	if err != nil {
		return nil, err
	}
	valueOutputBytes, err := rocmDeviceKVUint32Bytes("value output bytes", args.ValueOutputBytes)
	if err != nil {
		return nil, err
	}
	valueHeadDim, err := rocmDeviceKVPositiveUint32("value head dim", args.ValueHeadDim)
	if err != nil {
		return nil, err
	}
	valueHeadCount, err := rocmDeviceKVPositiveUint32("value head count", args.ValueHeadCount)
	if err != nil {
		return nil, err
	}
	if cap(payload) < hipKVEncodeTokenValueNormLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipKVEncodeTokenValueNormLaunchArgsBytes)
	} else {
		payload = payload[:hipKVEncodeTokenValueNormLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipKVEncodeTokenValueNormLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(hipKVEncodeTokenValueNormLaunchArgsBytes))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.KeyInputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.ValueInputPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.KeyOutputPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.ValueOutputPointer))
	binary.LittleEndian.PutUint32(payload[40:], keyCount)
	binary.LittleEndian.PutUint32(payload[44:], valueCount)
	binary.LittleEndian.PutUint32(payload[48:], keyInputBytes)
	binary.LittleEndian.PutUint32(payload[52:], valueInputBytes)
	binary.LittleEndian.PutUint32(payload[56:], keyOutputBytes)
	binary.LittleEndian.PutUint32(payload[60:], valueOutputBytes)
	binary.LittleEndian.PutUint32(payload[64:], args.KeyEncoding)
	binary.LittleEndian.PutUint32(payload[68:], args.ValueEncoding)
	binary.LittleEndian.PutUint64(payload[72:], uint64(args.KeyWidth))
	binary.LittleEndian.PutUint64(payload[80:], uint64(args.ValueWidth))
	binary.LittleEndian.PutUint64(payload[88:], uint64(args.TokenCount))
	binary.LittleEndian.PutUint32(payload[96:], valueHeadDim)
	binary.LittleEndian.PutUint32(payload[100:], valueHeadCount)
	binary.LittleEndian.PutUint32(payload[104:], math.Float32bits(args.Epsilon))
	return payload, nil
}

func (args hipKVDescriptorAppendLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipKVDescriptorAppendLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	buildSinglePage := args.Reserved0 == rocmKVDescriptorAppendModeBuildSinglePage
	if args.OutputDescriptorPointer == 0 || args.NewKeyPointer == 0 || args.NewValuePointer == 0 {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "KV descriptor append pointers are required", nil)
	}
	if !buildSinglePage && args.PreviousDescriptorPointer == 0 {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "KV descriptor append previous pointer is required", nil)
	}
	if !buildSinglePage && args.PreviousDescriptorBytes < rocmDeviceKVDescriptorHeaderBytes {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "KV descriptor append byte counts must include headers", nil)
	}
	if args.OutputDescriptorBytes < rocmDeviceKVDescriptorHeaderBytes {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "KV descriptor append byte counts must include headers", nil)
	}
	if err := rocmDeviceKVValidateModeCode(args.ModeCode); err != nil {
		return nil, err
	}
	blockSize, err := rocmDeviceKVPositiveUint32("block size", args.BlockSize)
	if err != nil {
		return nil, err
	}
	outputPageCount, err := rocmDeviceKVPositiveUint32("output page count", args.OutputPageCount)
	if err != nil {
		return nil, err
	}
	outputTokenCount, err := rocmDeviceKVPositiveUint32("output token count", args.OutputTokenCount)
	if err != nil {
		return nil, err
	}
	keyWidth, err := rocmDeviceKVPositiveUint32("key width", args.KeyWidth)
	if err != nil {
		return nil, err
	}
	valueWidth, err := rocmDeviceKVPositiveUint32("value width", args.ValueWidth)
	if err != nil {
		return nil, err
	}
	if err := rocmDeviceKVValidateEncodingCode(args.NewKeyEncoding); err != nil {
		return nil, err
	}
	if err := rocmDeviceKVValidateEncodingCode(args.NewValueEncoding); err != nil {
		return nil, err
	}
	if args.NewKeyBytes == 0 || args.NewValueBytes == 0 {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "KV descriptor append page metadata mismatch", nil)
	}
	expectedOutputBytes := uint64(rocmDeviceKVDescriptorHeaderBytes) + uint64(outputPageCount)*uint64(rocmDeviceKVDescriptorPageBytes)
	if args.OutputDescriptorBytes != expectedOutputBytes {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "KV descriptor append output byte count mismatch", nil)
	}
	if buildSinglePage && outputPageCount != 1 {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "KV descriptor single-page output count mismatch", nil)
	}
	trimStart, err := rocmDeviceKVUint64("trim start", args.TrimStart)
	if err != nil {
		return nil, err
	}
	if cap(payload) < hipKVDescriptorAppendLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipKVDescriptorAppendLaunchArgsBytes)
	} else {
		payload = payload[:hipKVDescriptorAppendLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipKVDescriptorAppendLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(hipKVDescriptorAppendLaunchArgsBytes))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.PreviousDescriptorPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputDescriptorPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.NewKeyPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.NewValuePointer))
	binary.LittleEndian.PutUint64(payload[40:], args.PreviousDescriptorBytes)
	binary.LittleEndian.PutUint64(payload[48:], args.OutputDescriptorBytes)
	binary.LittleEndian.PutUint64(payload[56:], args.NewKeyBytes)
	binary.LittleEndian.PutUint64(payload[64:], args.NewValueBytes)
	binary.LittleEndian.PutUint32(payload[72:], args.ModeCode)
	binary.LittleEndian.PutUint32(payload[76:], blockSize)
	binary.LittleEndian.PutUint32(payload[80:], outputPageCount)
	binary.LittleEndian.PutUint32(payload[84:], outputTokenCount)
	binary.LittleEndian.PutUint32(payload[88:], keyWidth)
	binary.LittleEndian.PutUint32(payload[92:], valueWidth)
	binary.LittleEndian.PutUint32(payload[96:], args.NewKeyEncoding)
	binary.LittleEndian.PutUint32(payload[100:], args.NewValueEncoding)
	binary.LittleEndian.PutUint64(payload[104:], trimStart)
	binary.LittleEndian.PutUint64(payload[112:], args.Reserved0)
	binary.LittleEndian.PutUint64(payload[120:], args.Reserved1)
	return payload, nil
}

func (args hipKVEncodeTokenValueNormDescriptorAppendLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipKVEncodeTokenValueNormDescriptorAppendLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if cap(payload) < hipKVEncodeTokenValueNormDescriptorAppendLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipKVEncodeTokenValueNormDescriptorAppendLaunchArgsBytes)
	} else {
		payload = payload[:hipKVEncodeTokenValueNormDescriptorAppendLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipKVEncodeTokenValueNormDescriptorAppendLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(hipKVEncodeTokenValueNormDescriptorAppendLaunchArgsBytes))
	if _, err := args.Encode.BinaryInto(payload[8:]); err != nil {
		return nil, err
	}
	if _, err := args.Descriptor.BinaryInto(payload[8+hipKVEncodeTokenValueNormLaunchArgsBytes:]); err != nil {
		return nil, err
	}
	return payload, nil
}

func rocmDeviceKVUint32(field string, value int) (uint32, error) {
	if value < 0 || value > int(^uint32(0)) {
		return 0, core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("%s is out of uint32 range", field), nil)
	}
	return uint32(value), nil
}

func rocmDeviceKVPositiveUint32(field string, value int) (uint32, error) {
	out, err := rocmDeviceKVUint32(field, value)
	if err != nil {
		return 0, err
	}
	if out == 0 {
		return 0, core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("%s must be positive", field), nil)
	}
	return out, nil
}

func rocmDeviceKVUint64(field string, value int) (uint64, error) {
	if value < 0 {
		return 0, core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("%s is out of uint64 range", field), nil)
	}
	return uint64(value), nil
}

func encodeROCmKVValuesDeviceBytes(encoding string, values []float32) ([]byte, error) {
	if len(values) == 0 {
		return nil, core.E("rocm.KVCache.DeviceMirror", "KV tensor values are required", nil)
	}
	switch encoding {
	case rocmKVEncodingFP16:
		payload := rocmDeviceKVBorrowPayloadBytes(len(values) * 2)
		for i, value := range values {
			binary.LittleEndian.PutUint16(payload[i*2:], rocmFloat32ToFloat16(value))
		}
		return payload, nil
	case rocmKVEncodingQ8:
		scale := rocmQuantScale(values, 127)
		payload := rocmDeviceKVBorrowPayloadBytes(4 + len(values))
		binary.LittleEndian.PutUint32(payload, math.Float32bits(scale))
		for i, value := range values {
			payload[4+i] = byte(int8(clampInt(int(math.Round(float64(value/scale))), -127, 127)))
		}
		return payload, nil
	case rocmKVEncodingQ4:
		scale := rocmQuantScale(values, 7)
		payload := rocmDeviceKVBorrowPayloadBytes(4 + (len(values)+1)/2)
		binary.LittleEndian.PutUint32(payload, math.Float32bits(scale))
		for i, value := range values {
			quantized := int8(clampInt(int(math.Round(float64(value/scale))), -8, 7))
			packed := packSignedQ4(quantized)
			if i%2 == 0 {
				payload[4+i/2] = packed
			} else {
				payload[4+i/2] |= packed << 4
			}
		}
		return payload, nil
	default:
		return nil, core.E("rocm.KVCache.DeviceMirror", core.Sprintf("unsupported direct KV tensor encoding %q", encoding), nil)
	}
}

func (tensor rocmKVEncodedTensor) deviceBytes() ([]byte, error) {
	switch tensor.encoding {
	case rocmKVEncodingFP16:
		payload := make([]byte, len(tensor.f16)*2)
		for i, value := range tensor.f16 {
			binary.LittleEndian.PutUint16(payload[i*2:], value)
		}
		return payload, nil
	case rocmKVEncodingQ8:
		payload := make([]byte, 4+len(tensor.q8))
		binary.LittleEndian.PutUint32(payload, math.Float32bits(tensor.scale))
		for i, value := range tensor.q8 {
			payload[4+i] = byte(value)
		}
		return payload, nil
	case rocmKVEncodingQ8Rows:
		payload := make([]byte, len(tensor.scales)*4+len(tensor.q8))
		for i, scale := range tensor.scales {
			binary.LittleEndian.PutUint32(payload[i*4:], math.Float32bits(scale))
		}
		offset := len(tensor.scales) * 4
		for i, value := range tensor.q8 {
			payload[offset+i] = byte(value)
		}
		return payload, nil
	case rocmKVEncodingQ8RowsI:
		rows := len(tensor.scales)
		if rows <= 0 || tensor.length%rows != 0 {
			return nil, core.E("rocm.KVCache.DeviceMirror", "q8 interleaved row tensor shape mismatch", nil)
		}
		rowWidth := tensor.length / rows
		rowStride := 4 + rowWidth
		payload := make([]byte, rows*rowStride)
		for row, scale := range tensor.scales {
			rowOffset := row * rowStride
			binary.LittleEndian.PutUint32(payload[rowOffset:], math.Float32bits(scale))
			for i, value := range tensor.q8[row*rowWidth : row*rowWidth+rowWidth] {
				payload[rowOffset+4+i] = byte(value)
			}
		}
		return payload, nil
	case rocmKVEncodingQ4:
		payload := make([]byte, 4+len(tensor.packedQ4))
		binary.LittleEndian.PutUint32(payload, math.Float32bits(tensor.scale))
		copy(payload[4:], tensor.packedQ4)
		return payload, nil
	case rocmKVEncodingQ4Rows:
		payload := make([]byte, len(tensor.scales)*4+len(tensor.packedQ4))
		for i, scale := range tensor.scales {
			binary.LittleEndian.PutUint32(payload[i*4:], math.Float32bits(scale))
		}
		copy(payload[len(tensor.scales)*4:], tensor.packedQ4)
		return payload, nil
	case rocmKVEncodingQ4RowsI:
		rows := len(tensor.scales)
		if rows <= 0 || tensor.length%rows != 0 {
			return nil, core.E("rocm.KVCache.DeviceMirror", "q4 interleaved row tensor shape mismatch", nil)
		}
		rowWidth := tensor.length / rows
		rowPacked := (rowWidth + 1) / 2
		rowStride := 4 + rowPacked
		payload := make([]byte, rows*rowStride)
		for row, scale := range tensor.scales {
			rowOffset := row * rowStride
			binary.LittleEndian.PutUint32(payload[rowOffset:], math.Float32bits(scale))
			copy(payload[rowOffset+4:rowOffset+4+rowPacked], tensor.packedQ4[row*rowPacked:row*rowPacked+rowPacked])
		}
		return payload, nil
	default:
		return nil, core.E("rocm.KVCache.DeviceMirror", core.Sprintf("unsupported tensor encoding %q", tensor.encoding), nil)
	}
}

func copyROCmDeviceKVTensorToHost(driver nativeHIPDriver, tensor rocmDeviceKVTensor, length int) (rocmKVEncodedTensor, error) {
	return copyROCmDeviceKVTensorRowsToHost(driver, tensor, length, 1)
}

func copyROCmDeviceKVTensorRowsToHost(driver nativeHIPDriver, tensor rocmDeviceKVTensor, length, rows int) (rocmKVEncodedTensor, error) {
	if driver == nil {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "HIP driver is nil", nil)
	}
	if tensor.pointer == 0 {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "device tensor pointer is nil", nil)
	}
	if tensor.sizeBytes == 0 {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "device tensor byte count is zero", nil)
	}
	maxInt := uint64(int(^uint(0) >> 1))
	if tensor.sizeBytes > maxInt {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "device tensor byte count exceeds addressable memory", nil)
	}
	payload := make([]byte, int(tensor.sizeBytes))
	if err := driver.CopyDeviceToHost(tensor.pointer, payload); err != nil {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "copy device tensor", err)
	}
	return rocmKVTensorFromDeviceBytesRows(tensor.encoding, length, rows, payload)
}

func rocmKVTensorFromDeviceBytes(encoding string, length int, payload []byte) (rocmKVEncodedTensor, error) {
	return rocmKVTensorFromDeviceBytesRows(encoding, length, 1, payload)
}

func rocmKVInt8View(payload []byte) []int8 {
	if len(payload) == 0 {
		return nil
	}
	return unsafe.Slice((*int8)(unsafe.Pointer(&payload[0])), len(payload))
}

func rocmKVTensorFromDeviceBytesRows(encoding string, length, rows int, payload []byte) (rocmKVEncodedTensor, error) {
	if length <= 0 {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "tensor length must be positive", nil)
	}
	tensor := rocmKVEncodedTensor{encoding: encoding, length: length, sizeBytes: uint64(len(payload))}
	switch encoding {
	case rocmKVEncodingFP16:
		if len(payload) != length*2 {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "fp16 tensor byte count mismatch", nil)
		}
		tensor.f16 = make([]uint16, length)
		for index := range tensor.f16 {
			tensor.f16[index] = binary.LittleEndian.Uint16(payload[index*2:])
		}
	case rocmKVEncodingQ8:
		if len(payload) != length+4 {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q8 tensor byte count mismatch", nil)
		}
		tensor.scale = math.Float32frombits(binary.LittleEndian.Uint32(payload[0:]))
		if tensor.scale <= 0 || math.IsNaN(float64(tensor.scale)) || math.IsInf(float64(tensor.scale), 0) {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q8 scale must be positive and finite", nil)
		}
		tensor.q8 = rocmKVInt8View(payload[4:])
	case rocmKVEncodingQ8Rows:
		if rows <= 0 || len(payload) != length+rows*4 {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q8 row tensor byte count mismatch", nil)
		}
		tensor.scales = make([]float32, rows)
		for index := range tensor.scales {
			tensor.scales[index] = math.Float32frombits(binary.LittleEndian.Uint32(payload[index*4:]))
			if tensor.scales[index] <= 0 || math.IsNaN(float64(tensor.scales[index])) || math.IsInf(float64(tensor.scales[index]), 0) {
				return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q8 row scale must be positive and finite", nil)
			}
		}
		offset := rows * 4
		tensor.q8 = rocmKVInt8View(payload[offset:])
	case rocmKVEncodingQ8RowsI:
		if rows <= 0 || length%rows != 0 {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q8 interleaved row tensor shape mismatch", nil)
		}
		rowWidth := length / rows
		rowStride := 4 + rowWidth
		if len(payload) != rows*rowStride {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q8 interleaved row tensor byte count mismatch", nil)
		}
		tensor.scales = make([]float32, rows)
		tensor.q8 = make([]int8, length)
		for row := 0; row < rows; row++ {
			rowOffset := row * rowStride
			tensor.scales[row] = math.Float32frombits(binary.LittleEndian.Uint32(payload[rowOffset:]))
			if tensor.scales[row] <= 0 || math.IsNaN(float64(tensor.scales[row])) || math.IsInf(float64(tensor.scales[row]), 0) {
				return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q8 interleaved row scale must be positive and finite", nil)
			}
			for index, value := range payload[rowOffset+4 : rowOffset+4+rowWidth] {
				tensor.q8[row*rowWidth+index] = int8(value)
			}
		}
	case rocmKVEncodingQ4:
		packedLength := (length + 1) / 2
		if len(payload) != packedLength+4 {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q4 tensor byte count mismatch", nil)
		}
		tensor.scale = math.Float32frombits(binary.LittleEndian.Uint32(payload[0:]))
		if tensor.scale <= 0 || math.IsNaN(float64(tensor.scale)) || math.IsInf(float64(tensor.scale), 0) {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q4 scale must be positive and finite", nil)
		}
		tensor.packedQ4 = payload[4:]
	case rocmKVEncodingQ4Rows:
		packedLength := (length + 1) / 2
		if rows <= 0 || len(payload) != packedLength+rows*4 {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q4 row tensor byte count mismatch", nil)
		}
		tensor.scales = make([]float32, rows)
		for index := range tensor.scales {
			tensor.scales[index] = math.Float32frombits(binary.LittleEndian.Uint32(payload[index*4:]))
			if tensor.scales[index] <= 0 || math.IsNaN(float64(tensor.scales[index])) || math.IsInf(float64(tensor.scales[index]), 0) {
				return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q4 row scale must be positive and finite", nil)
			}
		}
		tensor.packedQ4 = payload[rows*4:]
	case rocmKVEncodingQ4RowsI:
		if rows <= 0 || length%rows != 0 {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q4 interleaved row tensor shape mismatch", nil)
		}
		rowWidth := length / rows
		rowPacked := (rowWidth + 1) / 2
		rowStride := 4 + rowPacked
		if len(payload) != rows*rowStride {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q4 interleaved row tensor byte count mismatch", nil)
		}
		tensor.scales = make([]float32, rows)
		tensor.packedQ4 = make([]byte, rows*rowPacked)
		for row := 0; row < rows; row++ {
			rowOffset := row * rowStride
			tensor.scales[row] = math.Float32frombits(binary.LittleEndian.Uint32(payload[rowOffset:]))
			if tensor.scales[row] <= 0 || math.IsNaN(float64(tensor.scales[row])) || math.IsInf(float64(tensor.scales[row]), 0) {
				return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q4 interleaved row scale must be positive and finite", nil)
			}
			copy(tensor.packedQ4[row*rowPacked:row*rowPacked+rowPacked], payload[rowOffset+4:rowOffset+4+rowPacked])
		}
	default:
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", core.Sprintf("unsupported tensor encoding %q", encoding), nil)
	}
	return tensor, nil
}

func rocmKVTensorPrefixFromDeviceBytesRows(encoding string, length, rows int, payload []byte, prefixRows int) (rocmKVEncodedTensor, error) {
	if prefixRows <= 0 || prefixRows > rows {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "prefix row count mismatch", nil)
	}
	if rows <= 0 || length <= 0 || length%rows != 0 {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "tensor row shape mismatch", nil)
	}
	if prefixRows == rows {
		return rocmKVTensorFromDeviceBytesRows(encoding, length, rows, payload)
	}
	rowWidth := length / rows
	prefixLength := prefixRows * rowWidth
	switch encoding {
	case rocmKVEncodingFP16:
		prefixBytes := prefixLength * 2
		if len(payload) < prefixBytes {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "fp16 tensor byte count mismatch", nil)
		}
		return rocmKVTensorFromDeviceBytesRows(encoding, prefixLength, prefixRows, payload[:prefixBytes])
	case rocmKVEncodingQ8:
		prefixBytes := 4 + prefixLength
		if len(payload) < prefixBytes {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q8 tensor byte count mismatch", nil)
		}
		return rocmKVTensorFromDeviceBytesRows(encoding, prefixLength, prefixRows, payload[:prefixBytes])
	case rocmKVEncodingQ8Rows:
		if len(payload) < rows*4+length {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q8 row tensor byte count mismatch", nil)
		}
		tensor := rocmKVEncodedTensor{encoding: encoding, length: prefixLength, sizeBytes: uint64(prefixRows*4 + prefixLength), scales: make([]float32, prefixRows)}
		for index := range tensor.scales {
			tensor.scales[index] = math.Float32frombits(binary.LittleEndian.Uint32(payload[index*4:]))
			if tensor.scales[index] <= 0 || math.IsNaN(float64(tensor.scales[index])) || math.IsInf(float64(tensor.scales[index]), 0) {
				return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q8 row scale must be positive and finite", nil)
			}
		}
		tensor.q8 = rocmKVInt8View(payload[rows*4 : rows*4+prefixLength])
		return tensor, nil
	case rocmKVEncodingQ8RowsI:
		rowStride := 4 + rowWidth
		prefixBytes := prefixRows * rowStride
		if len(payload) < prefixBytes {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q8 interleaved row tensor byte count mismatch", nil)
		}
		return rocmKVTensorFromDeviceBytesRows(encoding, prefixLength, prefixRows, payload[:prefixBytes])
	case rocmKVEncodingQ4:
		prefixBytes := 4 + (prefixLength+1)/2
		if len(payload) < prefixBytes {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q4 tensor byte count mismatch", nil)
		}
		tensor, err := rocmKVTensorFromDeviceBytesRows(encoding, prefixLength, prefixRows, payload[:prefixBytes])
		if err != nil {
			return rocmKVEncodedTensor{}, err
		}
		if prefixLength%2 == 1 {
			tensor.packedQ4 = append([]byte(nil), tensor.packedQ4...)
			tensor.packedQ4[len(tensor.packedQ4)-1] &= 0x0f
		}
		return tensor, nil
	case rocmKVEncodingQ4Rows:
		fullPacked := (length + 1) / 2
		prefixPacked := (prefixLength + 1) / 2
		if len(payload) < rows*4+fullPacked {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q4 row tensor byte count mismatch", nil)
		}
		tensor := rocmKVEncodedTensor{encoding: encoding, length: prefixLength, sizeBytes: uint64(prefixRows*4 + prefixPacked), scales: make([]float32, prefixRows)}
		for index := range tensor.scales {
			tensor.scales[index] = math.Float32frombits(binary.LittleEndian.Uint32(payload[index*4:]))
			if tensor.scales[index] <= 0 || math.IsNaN(float64(tensor.scales[index])) || math.IsInf(float64(tensor.scales[index]), 0) {
				return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q4 row scale must be positive and finite", nil)
			}
		}
		tensor.packedQ4 = payload[rows*4 : rows*4+prefixPacked]
		if prefixLength%2 == 1 {
			tensor.packedQ4 = append([]byte(nil), tensor.packedQ4...)
			tensor.packedQ4[len(tensor.packedQ4)-1] &= 0x0f
		}
		return tensor, nil
	case rocmKVEncodingQ4RowsI:
		rowPacked := (rowWidth + 1) / 2
		rowStride := 4 + rowPacked
		prefixBytes := prefixRows * rowStride
		if len(payload) < prefixBytes {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q4 interleaved row tensor byte count mismatch", nil)
		}
		return rocmKVTensorFromDeviceBytesRows(encoding, prefixLength, prefixRows, payload[:prefixBytes])
	default:
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", core.Sprintf("unsupported tensor encoding %q", encoding), nil)
	}
}
