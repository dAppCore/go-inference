// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"runtime"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/engine/scheme"
	"github.com/tmc/apple/kernel"
	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
)

// This file assembles the attention half of a decode step on-device, in bf16
// (the dtype attention actually runs in). The enc* helpers each encode one
// dispatch into a caller-supplied encoder — the bf16 siblings of chain.go's
// float32 encode helpers, with bindings copied verbatim from the parity-proven
// bf16 ops in bf16.go / sdpa.go. AttentionBlock chains them in one command
// buffer with every intermediate resident.

func sharedBytes(b []byte) metal.MTLBuffer {
	return device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&b[0]), uint(len(b)), metal.MTLResourceStorageModeShared)
}

type attentionBlockKVScratch struct {
	kBytes, vBytes int
	k, v           *pinnedNoCopyBytes
	kViewPtr       uintptr
	kViewLen       int
	kView          metal.MTLBuffer
	kViewPinned    *pinnedNoCopyBytes
	vViewPtr       uintptr
	vViewLen       int
	vView          metal.MTLBuffer
	vViewPinned    *pinnedNoCopyBytes
}

type attentionBlockKVScratchKey struct {
	kBytes, vBytes int
}

type attentionBlockKVScratchPool struct {
	core.Pool[*attentionBlockKVScratch]
}

var attentionBlockKVScratchPools sync.Map

func attentionBlockKVScratchPoolFor(kBytes, vBytes int) *attentionBlockKVScratchPool {
	key := attentionBlockKVScratchKey{kBytes: kBytes, vBytes: vBytes}
	if v, ok := attentionBlockKVScratchPools.Load(key); ok {
		return v.(*attentionBlockKVScratchPool)
	}
	pool := &attentionBlockKVScratchPool{}
	if v, loaded := attentionBlockKVScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*attentionBlockKVScratchPool)
	}
	return pool
}

func newAttentionBlockKVScratch(kBytes, vBytes int) (*attentionBlockKVScratch, error) {
	if kBytes <= 0 || vBytes <= 0 {
		return nil, core.NewError("native.newAttentionBlockKVScratch: invalid dimensions")
	}
	k, err := newPinnedNoCopyBytes(kBytes)
	if err != nil {
		return nil, err
	}
	v, err := newPinnedNoCopyBytes(vBytes)
	if err != nil {
		k.Close()
		return nil, err
	}
	return &attentionBlockKVScratch{kBytes: kBytes, vBytes: vBytes, k: k, v: v}, nil
}

func getAttentionBlockKVScratch(kBytes, vBytes int) (*attentionBlockKVScratch, error) {
	pool := attentionBlockKVScratchPoolFor(kBytes, vBytes)
	if s := pool.Get(); s != nil {
		if s.kBytes == kBytes && s.vBytes == vBytes && s.k != nil && s.v != nil {
			return s, nil
		}
		s.Close()
	}
	return newAttentionBlockKVScratch(kBytes, vBytes)
}

func putAttentionBlockKVScratch(s *attentionBlockKVScratch) {
	if s != nil && s.kBytes > 0 && s.vBytes > 0 && s.k != nil && s.v != nil {
		attentionBlockKVScratchPoolFor(s.kBytes, s.vBytes).Put(s)
	}
}

func (s *attentionBlockKVScratch) Close() {
	if s == nil {
		return
	}
	if s.k != nil {
		s.k.Close()
		s.k = nil
	}
	if s.v != nil {
		s.v.Close()
		s.v = nil
	}
	s.closeCacheViews()
	s.kBytes, s.vBytes = 0, 0
}

func (s *attentionBlockKVScratch) closeCacheViews() {
	if s == nil {
		return
	}
	if s.kViewPinned != nil {
		s.kViewPinned.Close()
	}
	if s.vViewPinned != nil {
		s.vViewPinned.Close()
	}
	s.kViewPtr = 0
	s.kViewLen = 0
	s.kView = nil
	s.kViewPinned = nil
	s.vViewPtr = 0
	s.vViewLen = 0
	s.vView = nil
	s.vViewPinned = nil
}

func (s *attentionBlockKVScratch) buffers(kCache, vCache []byte) (metal.MTLBuffer, metal.MTLBuffer, error) {
	if s == nil || s.k == nil || s.v == nil {
		return nil, nil, core.NewError("native.attentionBlockKVScratch.buffers: scratch is nil")
	}
	if len(kCache) != s.kBytes || len(vCache) != s.vBytes {
		return nil, nil, core.NewError("native.attentionBlockKVScratch.buffers: cache length mismatch")
	}
	kBuf, err := s.k.copyBuffer(kCache)
	if err != nil {
		return nil, nil, err
	}
	vBuf, err := s.v.copyBuffer(vCache)
	if err != nil {
		return nil, nil, err
	}
	return kBuf, vBuf, nil
}

func (s *attentionBlockKVScratch) buffersNoCopy(kCache, vCache []byte) (metal.MTLBuffer, metal.MTLBuffer, bool, error) {
	if s == nil || s.k == nil || s.v == nil {
		return nil, nil, false, core.NewError("native.attentionBlockKVScratch.buffersNoCopy: scratch is nil")
	}
	if len(kCache) != s.kBytes || len(vCache) != s.vBytes {
		return nil, nil, false, core.NewError("native.attentionBlockKVScratch.buffersNoCopy: cache length mismatch")
	}
	if len(kCache) == 0 || len(vCache) == 0 {
		return nil, nil, false, core.NewError("native.attentionBlockKVScratch.buffersNoCopy: cache slices are empty")
	}
	kPtr := uintptr(unsafe.Pointer(&kCache[0]))
	vPtr := uintptr(unsafe.Pointer(&vCache[0]))
	if s.kView != nil && s.vView != nil &&
		s.kViewPtr == kPtr && s.kViewLen == len(kCache) &&
		s.vViewPtr == vPtr && s.vViewLen == len(vCache) {
		return s.kView, s.vView, true, nil
	}
	s.closeCacheViews()
	kBuf, kRegistered := registeredPinnedNoCopyBytes(kCache)
	var kPinner *runtime.Pinner
	if !kRegistered {
		var kNoCopy bool
		kBuf, kPinner, kNoCopy = residentNoCopyBytes(kCache)
		if !kNoCopy {
			if kPinner != nil {
				kPinner.Unpin()
			}
			return nil, nil, false, nil
		}
	}
	vBuf, vRegistered := registeredPinnedNoCopyBytes(vCache)
	var vPinner *runtime.Pinner
	if !vRegistered {
		var vNoCopy bool
		vBuf, vPinner, vNoCopy = residentNoCopyBytes(vCache)
		if !vNoCopy {
			if kPinner != nil {
				kPinner.Unpin()
			}
			if vPinner != nil {
				vPinner.Unpin()
			}
			return nil, nil, false, nil
		}
	}
	var kPinned, vPinned *pinnedNoCopyBytes
	if !kRegistered {
		kPinned = &pinnedNoCopyBytes{bytes: kCache, buf: kBuf, pinner: kPinner}
		runtime.SetFinalizer(kPinned, (*pinnedNoCopyBytes).Close)
	}
	if !vRegistered {
		vPinned = &pinnedNoCopyBytes{bytes: vCache, buf: vBuf, pinner: vPinner}
		runtime.SetFinalizer(vPinned, (*pinnedNoCopyBytes).Close)
	}
	s.kViewPtr = kPtr
	s.kViewLen = len(kCache)
	s.kView = kBuf
	s.kViewPinned = kPinned
	s.vViewPtr = vPtr
	s.vViewLen = len(vCache)
	s.vView = vBuf
	s.vViewPinned = vPinned
	return kBuf, vBuf, true, nil
}

func withPinnedNoCopyBytes(b []byte, fn func(metal.MTLBuffer) error) error {
	if len(b) == 0 {
		return core.NewError("native.withPinnedNoCopyBytes: empty byte slice")
	}
	var pinner runtime.Pinner
	pinner.Pin(&b[0])
	defer func() {
		pinner.Unpin()
		runtime.KeepAlive(b)
	}()
	buf := newNoCopyBuffer(unsafe.Pointer(&b[0]), uint(len(b)))
	if buf == nil || buf.GetID() == 0 {
		return core.NewError("native.withPinnedNoCopyBytes: failed to create no-copy Metal buffer")
	}
	return fn(buf)
}

func temporaryPinnedNoCopyBytes(b []byte, pinner *runtime.Pinner) (metal.MTLBuffer, error) {
	if len(b) == 0 {
		return nil, core.NewError("native.temporaryPinnedNoCopyBytes: empty byte slice")
	}
	pinner.Pin(&b[0])
	buf := newNoCopyBuffer(unsafe.Pointer(&b[0]), uint(len(b)))
	if buf == nil || buf.GetID() == 0 {
		pinner.Unpin()
		return nil, core.NewError("native.temporaryPinnedNoCopyBytes: failed to create no-copy Metal buffer")
	}
	return buf, nil
}

type pinnedNoCopyBytes struct {
	bytes  []byte
	buf    metal.MTLBuffer
	pinner *runtime.Pinner
}

type pinnedNoCopyBytesKey struct {
	ptr uintptr
	n   int
}

var pinnedNoCopyByteBuffers sync.Map

func pinnedNoCopyKey(b []byte) (pinnedNoCopyBytesKey, bool) {
	if len(b) == 0 {
		return pinnedNoCopyBytesKey{}, false
	}
	return pinnedNoCopyBytesKey{ptr: uintptr(unsafe.Pointer(&b[0])), n: len(b)}, true
}

func registerPinnedNoCopyBytes(p *pinnedNoCopyBytes) {
	if p == nil || p.buf == nil {
		return
	}
	key, ok := pinnedNoCopyKey(p.bytes)
	if !ok {
		return
	}
	pinnedNoCopyByteBuffers.Store(key, p.buf)
}

func unregisterPinnedNoCopyBytes(p *pinnedNoCopyBytes) {
	if p == nil {
		return
	}
	key, ok := pinnedNoCopyKey(p.bytes)
	if !ok {
		return
	}
	pinnedNoCopyByteBuffers.Delete(key)
}

func registeredPinnedNoCopyBytes(b []byte) (metal.MTLBuffer, bool) {
	key, ok := pinnedNoCopyKey(b)
	if !ok {
		return nil, false
	}
	v, ok := pinnedNoCopyByteBuffers.Load(key)
	if !ok {
		return nil, false
	}
	buf, ok := v.(metal.MTLBuffer)
	if !ok || buf == nil {
		pinnedNoCopyByteBuffers.Delete(key)
		return nil, false
	}
	return buf, true
}

func newPinnedNoCopyBytes(n int) (*pinnedNoCopyBytes, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if n <= 0 {
		return nil, core.NewError("native.newPinnedNoCopyBytes: size must be > 0")
	}
	b := make([]byte, n)
	pinner := pinGoBytes(b)
	if pinner == nil {
		return nil, core.NewError("native.newPinnedNoCopyBytes: failed to pin backing bytes")
	}
	buf := newNoCopyBuffer(unsafe.Pointer(&b[0]), uint(len(b)))
	if buf == nil || buf.GetID() == 0 {
		pinner.Unpin()
		return nil, core.NewError("native.newPinnedNoCopyBytes: failed to create no-copy Metal buffer")
	}
	p := &pinnedNoCopyBytes{bytes: b, buf: buf, pinner: pinner}
	registerPinnedNoCopyBytes(p)
	runtime.SetFinalizer(p, (*pinnedNoCopyBytes).Close)
	return p, nil
}

func (p *pinnedNoCopyBytes) copyBuffer(src []byte) (metal.MTLBuffer, error) {
	if p == nil || p.buf == nil {
		return nil, core.NewError("native.pinnedNoCopyBytes.copyBuffer: nil pinned buffer")
	}
	if len(src) != len(p.bytes) {
		return nil, core.NewError("native.pinnedNoCopyBytes.copyBuffer: source length mismatch")
	}
	copy(p.bytes, src)
	return p.buf, nil
}

func (p *pinnedNoCopyBytes) copyPrefixBuffer(src []byte) (metal.MTLBuffer, error) {
	if p == nil || p.buf == nil {
		return nil, core.NewError("native.pinnedNoCopyBytes.copyPrefixBuffer: nil pinned buffer")
	}
	if len(src) > len(p.bytes) {
		return nil, core.NewError("native.pinnedNoCopyBytes.copyPrefixBuffer: source length exceeds backing")
	}
	copy(p.bytes[:len(src)], src)
	return p.buf, nil
}

func (p *pinnedNoCopyBytes) Close() {
	if p == nil {
		return
	}
	runtime.SetFinalizer(p, nil)
	unregisterPinnedNoCopyBytes(p)
	if p.pinner != nil {
		p.pinner.Unpin()
		p.pinner = nil
	}
	runtime.KeepAlive(p.bytes)
	p.bytes = nil
	p.buf = nil
}

// residentBufs caches the GPU buffer for a RESIDENT weight slice. The MoE expert weights are the
// SAME mmap bytes every token, but the host-orchestrated MoE compute re-uploaded (sharedBytes COPIES)
// each selected expert's weight EVERY token. Those buffers are objc-"new" RETAINED, which
// withAutoreleasePool cannot free, so a long generation leaked tens of MB/token → 26B-A4B OOM'd at
// ~70 tokens (badLayers=0 throughout — a leak, not a decode bug). residentBytes uploads each distinct
// weight slice ONCE — keyed by its start address in the stable safetensors mmap — and reuses it, the
// resident pattern the dense projector already uses. Process-lifetime: model weights live as long as
// the model (a model swap would want eviction, not a concern for a single served model). The mutex
// guards concurrent sessions; the decode itself is single-goroutine.
var (
	residentBufMu sync.Mutex
	residentBufs  = map[uintptr]residentBuf{}
)

// residentBuf pins the backing slice alongside its uploaded buffer: caching by &b[0] is only sound
// while that address stays valid, which is automatic for the safetensors mmap (never moved) but NOT
// for a Go-managed slice (GC can free it and reuse the address → a stale cache hit). Holding b keeps
// it alive, so the key can never be re-issued for different data.
type residentBuf struct {
	buf    metal.MTLBuffer
	pin    []byte
	pinner *runtime.Pinner
	noCopy bool
}

func closeResidentBuf(r residentBuf) {
	if r.pinner != nil {
		r.pinner.Unpin()
	}
}

func residentKeyInRanges(key uintptr, bases, ends []uintptr) bool {
	for i, start := range bases {
		if i >= len(ends) {
			break
		}
		end := ends[i]
		if start != 0 && end > start && key >= start && key < end {
			return true
		}
	}
	return false
}

func evictResidentBufsForRanges(bases, ends []uintptr) {
	residentBufMu.Lock()
	defer residentBufMu.Unlock()
	for key, r := range residentBufs {
		if !residentKeyInRanges(key, bases, ends) {
			continue
		}
		closeResidentBuf(r)
		delete(residentBufs, key)
	}
}

func residentBytes(b []byte) metal.MTLBuffer {
	key := uintptr(unsafe.Pointer(&b[0]))
	residentBufMu.Lock()
	defer residentBufMu.Unlock()
	if r, ok := residentBufs[key]; ok {
		return r.buf
	}
	var (
		buf    metal.MTLBuffer
		pinner *runtime.Pinner
		noCopy bool
	)
	// An ODD base pointer cannot bind no-copy: the wrap comes back looking
	// perfect from the CPU (Contents()==base, full Length()) but the GPU's
	// view is sheared off the element boundary — every bf16 read is garbage
	// and dot products go NaN (TestNoCopyOffsetAlignmentRule: odd offsets NaN,
	// any even offset is exact). Odd bases only arise as interior slices of
	// loaded file blobs — a tensor at an arbitrary byte offset, immutable —
	// never as Go allocations (≥8-aligned), so a one-time upload copy is safe
	// here while every mutation-aliasing consumer keeps the live no-copy view.
	// (#352: the 12B assistant's pre_projection at blob+9423 drafted all-NaN
	// from clean operands; the 31B pair's 0% acceptance shared the cause.)
	if key%2 != 0 {
		buf = sharedBytes(b)
	} else {
		buf, pinner, noCopy = residentNoCopyBytes(b)
	}
	residentBufs[key] = residentBuf{buf: buf, pin: b, pinner: pinner, noCopy: noCopy}
	return buf
}

// noopResidentDealloc is the single, process-lifetime deallocator block shared by
// every no-copy resident buffer. The bytes are owned by our own runtime.Pinner
// (unpinned on residentBufs eviction — closeResidentBuf), so Metal must never free
// them: the deallocator is a no-op. Materialising an ObjC block from a Go closure
// costs a reflect.MakeFunc + a retained purego callback, and the tmc/apple binding
// does that per NewBuffer call — it profiled as ~13% of hot-path alloc_objects, one
// MakeFunc per buffer all wrapping this same empty closure. We build it ONCE and
// name it directly. It is deliberately never Released: the +1 from NewBlock keeps it
// alive for every buffer that references it, so — unlike the binding — we do NOT
// associate it per buffer (association would free it when any one buffer deallocs).
var noopResidentDealloc = objc.NewBlock(func(objc.Block, kernel.Pointer, uint64) {})

// newNoCopyBuffer wraps [ptr, ptr+length) as a no-copy Metal buffer with the shared
// no-op deallocator above, via a direct msgSend — replacing the tmc/apple binding's
// per-call objc.NewBlock(closure) (a reflect.MakeFunc + a retained purego callback on
// every call). GetID()==0 on failure. Every no-copy site owns its bytes' lifetime
// elsewhere (a runtime.Pinner, or the mmap's Close), so the shared no-op deallocator
// is correct for all of them.
func newNoCopyBuffer(ptr unsafe.Pointer, length uint) metal.MTLBuffer {
	return metal.MTLBufferObjectFromID(objc.Send[objc.ID](device.ID,
		objc.Sel("newBufferWithBytesNoCopy:length:options:deallocator:"),
		ptr, length, metal.MTLResourceStorageModeShared, objc.ID(noopResidentDealloc)))
}

func residentNoCopyBytes(b []byte) (metal.MTLBuffer, *runtime.Pinner, bool) {
	if isMappedShardBytes(b) {
		return sharedBytes(b), nil, false
	}
	pinner := pinGoBytes(b)
	if pinner == nil {
		return sharedBytes(b), nil, false
	}
	buf := newNoCopyBuffer(unsafe.Pointer(&b[0]), uint(len(b)))
	if buf.GetID() == 0 {
		if pinner != nil {
			pinner.Unpin()
		}
		return sharedBytes(b), nil, false
	}
	return buf, pinner, true
}

func pinGoBytes(b []byte) (pinner *runtime.Pinner) {
	defer func() {
		if recover() != nil {
			if pinner != nil {
				pinner.Unpin()
			}
			pinner = nil
		}
	}()
	pinner = new(runtime.Pinner)
	pinner.Pin(&b[0])
	return pinner
}

// sharedOrNil is sharedBytes for an optional weight: nil/empty → a nil MTLBuffer (the
// half-encoders treat a nil norm buffer as "skip"), so callers can pass an absent gemma4
// post-norm straight through without a length guard.
func sharedOrNil(b []byte) metal.MTLBuffer {
	if len(b) == 0 {
		return nil
	}
	return sharedBytes(b)
}

func scratchBF16(nElems int) metal.MTLBuffer {
	return device.NewBufferWithLengthOptions(uint(nElems*bf16Size), metal.MTLResourceStorageModeShared)
}

// scratchF32 allocates a shared float32 scratch buffer of nElems — the 2-pass SDPA
// per-block sums/maxs intermediates are float32 (the online-softmax accumulators).
func scratchF32(nElems int) metal.MTLBuffer {
	return device.NewBufferWithLengthOptions(uint(nElems*4), metal.MTLResourceStorageModeShared)
}

// encRMSNormBF16 encodes a single-row bf16 RMSNorm (axisSize ≤ 4096) into enc. wOff offsets the
// WEIGHT binding (bytes) — the zero-copy weight path binds the norm weight at its offset into the
// shared shard mmap buffer rather than uploading it; wOff=0 is the plain (copied-buffer) binding.
func encRMSNormBF16(enc metal.MTLComputeCommandEncoder, x, w, out metal.MTLBuffer, wOff uint, axisSize int, eps float32) error {
	return encRMSNormBF16At(enc, x, w, out, 0, wOff, 0, axisSize, eps)
}

// encRMSNormBF16At is encRMSNormBF16 with the input row bound at xOff and the output at outOff
// BYTES — the SAME single-row specialised pipeline, for rows living at offsets inside shared
// K-row buffers (the batched dense interleave). Bit-identical per row to the sequential path;
// the generic rows kernel (encRMSNormRowsBF16) reduces in a different order and drifts by ulps.
func encRMSNormBF16At(enc metal.MTLComputeCommandEncoder, x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, axisSize int, eps float32) error {
	pso, err := pipelineFor(rmsKernelBF16(axisSize))
	if err != nil {
		return err
	}
	// single-row up to the limit, else the looped kernel (a max-threads threadgroup that grid-strides
	// the axis) — a single row of axis > 4096 (gemma4 31B hidden 5376) overruns the single-row cap.
	// One shared body (emitRMSNormAt) records the binding ABI into the live encoder here and into the
	// ICB recorder's setRMS — the path-unifying dispatchSink (one math, two targets).
	emitRMSNormAt(encSink{enc}, pso, x, w, out, xOff, wOff, outOff, axisSize, eps, rmsThreadgroup(axisSize, pso))
	return nil
}

// encRMSNormRowsBF16 RMS-norms `rows` contiguous rows of axisSize each, independently,
// with the single shared weight (axisSize) — one threadgroup per row (the grid carries
// the batch, exactly as the standalone RMSNormBF16's rows path). gemma4 QK-norm uses this
// to norm each attention head's headDim slice (rows = nHeads, axisSize = headDim) with the
// shared q_norm/k_norm weight. wOff offsets the WEIGHT binding (the zero-copy path binds it at its
// offset into the shared shard buffer; 0 is the plain binding). Safe in-place (the per-row
// reduction barriers before the write phase, and each thread writes only its own element).
func encRMSNormRowsBF16(enc metal.MTLComputeCommandEncoder, x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, rows, axisSize int, eps float32) error {
	// single-row kernel up to rmsLoopedLimit, the looped (grid-striding) kernel past it —
	// the raw tg formula exceeds 1024 threads/threadgroup beyond 4096 dims and Metal DROPS
	// the dispatch silently (31B dModel=5376 prefilled all-zero caches through exactly this).
	pso, err := pipelineFor(rmsKernelBF16(axisSize))
	if err != nil {
		return err
	}
	emitRMSNormRows(encSink{enc}, pso, x, w, out, xOff, wOff, outOff, axisSize, eps, rows, rmsThreadgroup(axisSize, pso))
	return nil
}

func encRMSNormRowsBF16Object(enc metal.MTLComputeCommandEncoderObject, x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, rows, axisSize int, eps float32) error {
	pso, err := pipelineFor(rmsKernelBF16(axisSize))
	if err != nil {
		return err
	}
	emitRMSNormRows(encObjectSink{enc}, pso, x, w, out, xOff, wOff, outOff, axisSize, eps, rows, rmsThreadgroup(axisSize, pso))
	return nil
}

// encGemvBF16 encodes out = mat @ vec (bf16, mat row-major outDim×inDim) into enc.
func encGemvBF16(enc metal.MTLComputeCommandEncoder, mat, vec, out metal.MTLBuffer, outDim, inDim int) error {
	return encGemvBF16To(enc, mat, vec, out, 0, 0, outDim, inDim)
}

// encGemvBF16To is encGemvBF16 that binds the weight MATRIX at matOff BYTES and writes the result
// starting at outOff BYTES into out. matOff lets the zero-copy weight path bind the projection
// weight at its offset into the shared shard mmap buffer (vs an uploaded copy); outOff lets the
// decode KV path project K/V straight into the (seq-major) cache at the current token's row, so
// the projection IS the cache append (no copy kernel; the gemv output index is relative to the
// bound buffer offset). matOff=outOff=0 is the plain projection.
func encGemvBF16To(enc metal.MTLComputeCommandEncoder, mat, vec, out metal.MTLBuffer, matOff, outOff uint, outDim, inDim int) error {
	return encGemvBF16VecAt(enc, mat, vec, out, matOff, 0, outOff, outDim, inDim)
}

// encGemvBF16VecAt is encGemvBF16To that additionally binds the input VECTOR at vecOff BYTES —
// used where the activation lives at a row offset inside a shared multi-row buffer (the batched
// dense prefill's per-row PLE gate) rather than at the start of a dedicated buffer.
func encGemvBF16VecAt(enc metal.MTLComputeCommandEncoder, mat, vec, out metal.MTLBuffer, matOff, vecOff, outOff uint, outDim, inDim int) error {
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	pso, err := pipelineFor(gemvKernelName("bfloat16", bm, bn, sm, sn, tm, tn))
	if err != nil {
		return err
	}
	// bf16 tiled gemv through the SHARED emitGemv body (with the ICB recorder's setGemv).
	emitGemvVecAt(encSink{enc}, pso, mat, matOff, vec, vecOff, out, outOff, inDim, outDim, bm, bn, sm, tm)
	return nil
}

// encGemvBF16BatchedAt encodes `batch` independent gemvs against ONE shared weight matrix in a
// single dispatch (grid Z carries the batch): out row z = mat @ vec row z. vec rows are contiguous
// bf16 at vecOff + z·inDim elements; out rows land at outOff + z·outDim. The kernel variant and
// per-row tile loop are exactly encGemvBF16VecAt's (gemvTiles ignores batch), so each row's output
// is byte-identical to `batch` single-row dispatches — the weight matrix is just swept once. This
// is the batched dense pass's MLP fold: K rows' gate/up/down share each layer's weight read.
func encGemvBF16BatchedAt(enc metal.MTLComputeCommandEncoder, mat, vec, out metal.MTLBuffer, matOff, vecOff, outOff uint, outDim, inDim, batch int) error {
	// large row counts take the true tiled GEMM — the weight read once for ALL rows, trading the
	// per-row gemv's byte-identity for token-identity (pkg/metal's GEMM prefill trade). Small
	// batches (the MTP verify, every parity fixture) stay on the grid-Z gemv and its strict
	// byte-identity with the sequential lane.
	if batch >= steelGEMMMinRows && !steelGEMMDisabledForTest &&
		encGemmBF16NT(enc, mat, vec, out, matOff, vecOff, outOff, outDim, inDim, batch) {
		return nil
	}
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	pso, err := pipelineFor(gemvKernelName("bfloat16", bm, bn, sm, sn, tm, tn))
	if err != nil {
		return err
	}
	emitGemvBatchedVecAt(encSink{enc}, pso, mat, matOff, vec, vecOff, out, outOff, inDim, outDim, batch, bm, bn, sm, tm)
	return nil
}

func encGemvBF16ToObject(enc metal.MTLComputeCommandEncoderObject, mat, vec, out metal.MTLBuffer, matOff, outOff uint, outDim, inDim int) error {
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	pso, err := pipelineFor(gemvKernelName("bfloat16", bm, bn, sm, sn, tm, tn))
	if err != nil {
		return err
	}
	emitGemv(encObjectSink{enc}, pso, mat, matOff, vec, out, outOff, inDim, outDim, bm, bn, sm, tm)
	return nil
}

// encQMVBF16 encodes a bf16-activation 4-bit quantised matvec (out = x @ Wᵀ) into
// enc — the chained sibling of QMVBF16 for the quantised decode layer. Same kernel
// (affine_qmv[_fast]_bfloat16_t) and ABI as QMVBF16. wqOff/scalesOff/biasesOff bind the three
// quant weight tensors at their offsets into the shared shard mmap buffer(s) (the zero-copy weight
// path; each tensor can sit in a different shard, hence three offsets) — 0/0/0 is the plain
// (uploaded-copy) binding. outOff lets the projection write its result straight into a cache row
// (the V projection), exactly like encGemvBF16To. wq is packed 4-bit; scales/biases bf16.
type qmvBF16KernelKey struct {
	groupSize, bits int
	fast            bool
}

var qmvBF16KernelNames sync.Map

func qmvBF16KernelName(outDim, inDim, groupSize, bits int) string {
	fast := outDim%8 == 0 && inDim%512 == 0
	key := qmvBF16KernelKey{groupSize: groupSize, bits: bits, fast: fast}
	if v, ok := qmvBF16KernelNames.Load(key); ok {
		return v.(string)
	}
	variant := "_qmv_"
	if fast {
		variant = "_qmv_fast_"
	}
	name := core.Sprintf("affine%sbfloat16_t_gs_%d_b_%d_batch_0", variant, groupSize, bits)
	if v, loaded := qmvBF16KernelNames.LoadOrStore(key, name); loaded {
		return v.(string)
	}
	return name
}

func encQMVBF16(enc metal.MTLComputeCommandEncoder, wq, scales, biases, x, out metal.MTLBuffer, wqOff, scalesOff, biasesOff, outOff uint, outDim, inDim, groupSize, bits int) error {
	return encQMVBF16At(enc, wq, scales, biases, x, out, wqOff, scalesOff, biasesOff, 0, outOff, outDim, inDim, groupSize, bits)
}

// encQMVBF16At is encQMVBF16 with the activation vector bound at xOff BYTES — the batched dense
// forward's per-row quant PLE gate reads its row in place inside the shared K-row buffer.
func encQMVBF16At(enc metal.MTLComputeCommandEncoder, wq, scales, biases, x, out metal.MTLBuffer, wqOff, scalesOff, biasesOff, xOff, outOff uint, outDim, inDim, groupSize, bits int) error {
	pso, err := pipelineFor(qmvBF16KernelName(outDim, inDim, groupSize, bits))
	if err != nil {
		return err
	}
	// 4-bit quantised matvec through the SHARED emitQMV body (with the ICB recorder's setQMV).
	emitQMVAt(encSink{enc}, pso, wq, wqOff, scales, scalesOff, biases, biasesOff, x, xOff, out, outOff, inDim, outDim)
	return nil
}

// qmmTKernelName builds MLX's transposed quantised-GEMM kernel name for the affine mode at
// gs/bits — the batched sibling of qmvBF16KernelName. aligned keys the N%32 template variant;
// batch_0 = one 2-D x (the prompt fold's case; the strides block is skipped by that variant).
func qmmTKernelName(outDim, groupSize, bits int) string {
	aligned := "false"
	if outDim%32 == 0 {
		aligned = "true"
	}
	return core.Sprintf("affine_qmm_t_bfloat16_t_gs_%d_b_%d_alN_%s_batch_0", groupSize, bits, aligned)
}

// encQMMTBF16At encodes out[M,N] = x[M,K] @ dequant(w[N,K])ᵀ — MLX's affine qmm_t, the ONE
// weight pass scoring all M rows (the quant prompt-prefill fold; the per-row qmv re-read the
// weights M times). x rows are contiguous [M,K] bf16 at xOff; out rows [M,N] bf16 at outOff.
func encQMMTBF16At(enc metal.MTLComputeCommandEncoder, wq, scales, biases, x, out metal.MTLBuffer, wqOff, scalesOff, biasesOff, xOff, outOff uint, m, outDim, inDim, groupSize, bits int) error {
	pso, err := pipelineFor(qmmTKernelName(outDim, groupSize, bits))
	if err != nil {
		return err
	}
	emitQMMT(encSink{enc}, pso, wq, wqOff, scales, scalesOff, biases, biasesOff, x, xOff, out, outOff, m, outDim, inDim)
	return nil
}

// encRoPEBF16 encodes single-token bf16 RoPE over x (b=1, nHeads, 1, headDim) at
// the position in offBuf into enc. offBuf holds one int32.
func encRoPEBF16(enc metal.MTLComputeCommandEncoder, x, out, offBuf metal.MTLBuffer, nHeads, headDim, rotaryDim int, base, scale float32) error {
	return encRoPEBF16To(enc, x, out, 0, 0, offBuf, nHeads, headDim, rotaryDim, base, scale)
}

// encRoPEBF16To is encRoPEBF16 that reads from inOff and writes the rotated result starting at
// outOff BYTES — used to RoPE the new token's K in place within the (seq-major) KV cache row.
// rotaryDim rotates only the first rotaryDim of each head (gemma4 partial rotary; == headDim is
// full); the kernel writes only the rotated dims, so for partial rotary call it IN PLACE
// (in==out, inOff==outOff) so the untouched [rotaryDim:headDim] tail keeps its input value.
func encRoPEBF16To(enc metal.MTLComputeCommandEncoder, x, out metal.MTLBuffer, inOff, outOff uint, offBuf metal.MTLBuffer, nHeads, headDim, rotaryDim int, base, scale float32) error {
	return encRoPEBF16ToAt(enc, x, out, inOff, outOff, offBuf, 0, nHeads, headDim, rotaryDim, base, scale)
}

func encRoPEBF16ToAt(enc metal.MTLComputeCommandEncoder, x, out metal.MTLBuffer, inOff, outOff uint, offBuf metal.MTLBuffer, offOff uint, nHeads, headDim, rotaryDim int, base, scale float32) error {
	pso, err := ropePipelineBF16(false)
	if err != nil {
		return err
	}
	rd := headDim
	if rotaryDim > 0 && rotaryDim < headDim {
		rd = rotaryDim
	}
	// base partial-rotary RoPE through the SHARED emitRope body (with encRoPEFreqsBF16To + the ICB setRope);
	// periods=nil selects the base form, log2(base) at index 10.
	emitRopeAt(encSink{enc}, pso, x, out, inOff, outOff, offBuf, offOff, nil, nHeads, rd, headDim, scale, float32(math.Log2(float64(base))))
	return nil
}

// encSDPA encodes single-query bf16 attention over a HEAD-MAJOR cache into enc:
// q (1, nHeads, 1, headDim), k/v (1, nKVHeads, kvLen, headDim) → out (1, nHeads,
// 1, headDim). No mask / not causal.
func encSDPA(enc metal.MTLComputeCommandEncoder, q, k, v, out metal.MTLBuffer, nHeads, nKVHeads, headDim, kvLen int, scale float32) error {
	// head-major: head h, seq i, dim d at (h*kvLen + i)*headDim + d
	return encSDPAStrided(enc, q, k, v, out, nHeads, nKVHeads, headDim, kvLen,
		int64(kvLen*headDim), int64(headDim), int64(kvLen*headDim), int64(headDim), scale, 0)
}

// slideWindow returns the cache window the SDPA attends for a layer decoding at
// position pos: the full prefix [0..pos] (start 0, n pos+1) for a global layer
// (slideW <= 0), or the last slideW rows once the window is exceeded — the
// correctness of sliding-window attention. (The cache still stores all rows; the
// rotating W-sized buffer is a separate memory optimisation.)
func slideWindow(pos, slideW int) (start, n int) {
	if slideW > 0 && pos+1 > slideW {
		return pos + 1 - slideW, slideW
	}
	return 0, pos + 1
}

// encSDPAStrided encodes single-query bf16 attention with explicit element
// strides — the sdpa_vector kernel indexes keys as kv_head*k_head_stride +
// seq*k_seq_stride + d with headDim contiguous (innermost), so the cache layout
// is the caller's choice. The decode KV path uses a SEQ-MAJOR cache
// [seq, nKVHeads, headDim] (k_head_stride=headDim, k_seq_stride=nKVHeads*headDim)
// so appending a token is one contiguous row write; encSDPA passes the head-major
// strides. n is the live cache length (the grown window).
// kvByteOff offsets the K and V bindings (bytes) — used to attend a window of the
// cache starting at a non-zero row (sliding-window attention reads the last W rows).
func encSDPAStrided(enc metal.MTLComputeCommandEncoder, q, k, v, out metal.MTLBuffer, nHeads, nKVHeads, headDim, n int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32, kvByteOff uint) error {
	pso, err := sdpaVectorPipelineForHeadDim(headDim)
	if err != nil {
		return err
	}
	// single-pass SDPA through the SHARED emitSDPA body (with the ICB recorder's SDPA op). nBuf=nil → N
	// is inlined here (the re-encode path knows the live length); the ICB binds its rebound N buffer.
	emitSDPA(encSink{enc}, pso, q, k, v, out, kvByteOff, nil, nHeads, nKVHeads, n, kHeadStride, kSeqStride, vHeadStride, vSeqStride, scale)
	return nil
}

// encSDPA2PassStrided encodes the TWO-pass long-context SDPA into enc (b=1 decode):
// pass 1 (sdpa_vector_2pass_1) fans the cache reduction over `blocks` threadgroups,
// each writing its segment's online-softmax partials (weighted-V sum + sum/max) into
// the caller's once-allocated intermediates; pass 2 (sdpa_vector_2pass_2) merges them
// into the head output. Same q/k/v/out + element strides + kvByteOff as
// encSDPAStrided (the strides describe the caller's cache layout, the offset selects a
// sliding window) — the two dispatches are serial in enc so pass 2 sees pass 1's
// writes. Token-identical to encSDPAStrided (sdpa_2pass_test.go), differing only in how
// the reduction parallelises — so it keeps scaling where the single-pass kernel stalls.
func encSDPA2PassStrided(enc metal.MTLComputeCommandEncoder, q, k, v, out, partials, sums, maxs metal.MTLBuffer, nHeads, nKVHeads, headDim, n int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32, kvByteOff uint) error {
	blocks := sdpa2PassBlocks(n)
	pso1, err := sdpaVector2Pass1PipelineForHeadDim(headDim, blocks)
	if err != nil {
		return err
	}
	pso2, err := sdpaVector2Pass2PipelineForHeadDim(headDim)
	if err != nil {
		return err
	}
	sink := encSink{enc}
	emitSDPA2Pass1(sink, pso1, q, k, v, partials, sums, maxs, kvByteOff, 1, nHeads, nKVHeads, n, int(blocks), kHeadStride, kSeqStride, vHeadStride, vSeqStride, scale)
	emitSDPA2Pass2(sink, pso2, partials, sums, maxs, out, 1, nHeads, int(blocks))
	return nil
}

// encSDPADecode routes a single-query decode SDPA to the 2-pass long-context kernels
// once the attended window n reaches the single-pass knee AND the scratch carries the
// (once-allocated) 2-pass intermediates; otherwise the proven single-pass kernel. Same
// buffers/strides/offset either way, so the choice is invisible to the caller and
// token-identical — only the cache-reduction parallelism differs. The intermediates
// live in sc so the long-context path adds NO per-token allocation.
func encSDPADecode(enc metal.MTLComputeCommandEncoder, sc attnScratch, q, k, v, out metal.MTLBuffer, nHeads, nKVHeads, headDim, n int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32, kvByteOff uint) error {
	return encSDPADecodeAt(enc, sc, q, 0, k, v, out, 0, nHeads, nKVHeads, headDim, n, kHeadStride, kSeqStride, vHeadStride, vSeqStride, scale, kvByteOff)
}

// encSDPADecodeAt is encSDPADecode with the query and output bound at byte offsets — the batched
// pass's attention fold keeps each row's q/attn in shared K-row slabs. Same 2-pass routing; the
// 2-pass intermediates stay the shared per-session scratch (the rows hazard-serialise on them,
// exactly as they did on the shared single-row scratch).
func encSDPADecodeAt(enc metal.MTLComputeCommandEncoder, sc attnScratch, q metal.MTLBuffer, qOff uint, k, v, out metal.MTLBuffer, outOff uint, nHeads, nKVHeads, headDim, n int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32, kvByteOff uint) error {
	if n >= sdpa2PassMinKV && sc.p2Partials != nil && !sdpa2PassDisabledForTest {
		blocks := sdpa2PassBlocks(n)
		pso1, err := sdpaVector2Pass1PipelineForHeadDim(headDim, blocks)
		if err != nil {
			return err
		}
		pso2, err := sdpaVector2Pass2PipelineForHeadDim(headDim)
		if err != nil {
			return err
		}
		sink := encSink{enc}
		emitSDPA2Pass1At(sink, pso1, q, qOff, k, v, sc.p2Partials, sc.p2Sums, sc.p2Maxs, kvByteOff, 1, nHeads, nKVHeads, n, int(blocks), kHeadStride, kSeqStride, vHeadStride, vSeqStride, scale)
		emitSDPA2Pass2At(sink, pso2, sc.p2Partials, sc.p2Sums, sc.p2Maxs, out, outOff, 1, nHeads, int(blocks))
		return nil
	}
	pso, err := sdpaVectorPipelineForHeadDim(headDim)
	if err != nil {
		return err
	}
	emitSDPAAt(encSink{enc}, pso, q, qOff, k, v, out, outOff, kvByteOff, nil, nHeads, nKVHeads, n, kHeadStride, kSeqStride, vHeadStride, vSeqStride, scale)
	return nil
}

// encBinaryDT encodes the element-wise binary op (op = "Add" | "Multiply") in the
// activation dtype dt — kernel "vv_<op><dt.Name>" — over n elements into enc. The
// dtype is resolved from the registered scheme (scheme.BFloat16, scheme.Float32, …),
// so a new activation dtype is a registered scheme, not a new hardcoded encoder.
func encBinaryDT(enc metal.MTLComputeCommandEncoder, op string, dt scheme.DType, a, b, out metal.MTLBuffer, n int) error {
	return encBinaryDTTo(enc, op, dt, a, b, out, 0, 0, 0, n)
}

func encBinaryDTTo(enc metal.MTLComputeCommandEncoder, op string, dt scheme.DType, a, b, out metal.MTLBuffer, aOff, bOff, outOff uint, n int) error {
	pso, err := pipelineFor("vv_" + op + dt.Name())
	if err != nil {
		return err
	}
	emitBinary(encSink{enc}, pso, a, aOff, b, bOff, out, outOff, n)
	return nil
}

func encBinaryLiteralTo(enc metal.MTLComputeCommandEncoder, name string, a, b, out metal.MTLBuffer, aOff, bOff, outOff uint, n int) error {
	pso, err := pipelineFor(name)
	if err != nil {
		return err
	}
	emitBinary(encSink{enc}, pso, a, aOff, b, bOff, out, outOff, n)
	return nil
}

func encBinaryLiteralToObject(enc metal.MTLComputeCommandEncoderObject, name string, a, b, out metal.MTLBuffer, aOff, bOff, outOff uint, n int) error {
	pso, err := pipelineFor(name)
	if err != nil {
		return err
	}
	emitBinary(encObjectSink{enc}, pso, a, aOff, b, bOff, out, outOff, n)
	return nil
}

// encAddBF16 / encMulBF16 are the bf16-bound conveniences for gemma's MLP and
// residual paths. They use literal kernel names to avoid rebuilding the generic
// "vv_"+op+dtype string in the per-token decode loop.
func encAddBF16(enc metal.MTLComputeCommandEncoder, a, b, out metal.MTLBuffer, n int) error {
	return encAddBF16To(enc, a, b, out, 0, 0, 0, n)
}
func encAddBF16To(enc metal.MTLComputeCommandEncoder, a, b, out metal.MTLBuffer, aOff, bOff, outOff uint, n int) error {
	return encBinaryLiteralTo(enc, "vv_Addbfloat16", a, b, out, aOff, bOff, outOff, n)
}
func encAddBF16Object(enc metal.MTLComputeCommandEncoderObject, a, b, out metal.MTLBuffer, n int) error {
	return encBinaryLiteralToObject(enc, "vv_Addbfloat16", a, b, out, 0, 0, 0, n)
}
func encMulBF16(enc metal.MTLComputeCommandEncoder, a, b, out metal.MTLBuffer, n int) error {
	return encMulBF16To(enc, a, b, out, 0, 0, 0, n)
}
func encMulBF16To(enc metal.MTLComputeCommandEncoder, a, b, out metal.MTLBuffer, aOff, bOff, outOff uint, n int) error {
	return encBinaryLiteralTo(enc, "vv_Multiplybfloat16", a, b, out, aOff, bOff, outOff, n)
}

// encUnaryDT encodes the element-wise unary op (op = "Tanh", …) in the activation
// dtype dt — kernel "v_<op><dt.Name><dt.Name>" (the metallib repeats the dtype for
// in+out) — over n elements. The count is a uint32 at index 2 (SetBytes), matching
// TanhBF16. Dtype resolved from the registered scheme, not hardcoded.
func encUnaryDT(enc metal.MTLComputeCommandEncoder, op string, dt scheme.DType, in, out metal.MTLBuffer, n int) error {
	pso, err := pipelineFor("v_" + op + dt.Name() + dt.Name())
	if err != nil {
		return err
	}
	emitUnary(encSink{enc}, pso, in, out, n)
	return nil
}

func encUnaryDTObject(enc metal.MTLComputeCommandEncoderObject, op string, dt scheme.DType, in, out metal.MTLBuffer, n int) error {
	pso, err := pipelineFor("v_" + op + dt.Name() + dt.Name())
	if err != nil {
		return err
	}
	emitUnary(encObjectSink{enc: enc}, pso, in, out, n)
	return nil
}

// encTanhBF16 is the bf16-bound tanh (gemma's gelu nonlinearity) — scheme.BFloat16 through encUnaryDT.
func encTanhBF16(enc metal.MTLComputeCommandEncoder, in, out metal.MTLBuffer, n int) error {
	return encUnaryDT(enc, "Tanh", scheme.BFloat16, in, out, n)
}

func encTanhBF16Object(enc metal.MTLComputeCommandEncoderObject, in, out metal.MTLBuffer, n int) error {
	pso, err := pipelineFor("v_Tanhbfloat16bfloat16")
	if err != nil {
		return err
	}
	emitUnary(encObjectSink{enc: enc}, pso, in, out, n)
	return nil
}

// AttentionBlock runs the attention half of a gemma decode step on-device, in
// bf16, over a given KV cache (the read path of a single new token):
//
//	normed  = rmsnorm(x, normWeight)
//	q       = wQ · normed                 (dModel → nHeads·headDim)
//	q       = rope(q, offset)             (per head, full rotary)
//	attn    = sdpa(q, kCache, vCache)     (single query over the cache)
//	attnOut = wO · attn                   (nHeads·headDim → dModel)
//	out     = x + attnOut                 (residual)
//
// Every buffer is bf16 and stays resident; the whole block is one command
// buffer, one commit. kCache/vCache are the post-RoPE cache (1, nKVHeads, kvLen,
// headDim). The cache-write half (wK/wV projections, RoPE on the new K, append)
// is a separate follow-up. All inputs/outputs are raw bf16 bytes. The result
// equals the same native bf16 ops run separately — proven in the tests.
func AttentionBlock(x, normWeight, wQ, wO, kCache, vCache []byte, dModel, nHeads, nKVHeads, headDim, kvLen int, base, scale float32, offset int, eps float32) ([]byte, error) {
	return attentionBlockInto(nil, x, normWeight, wQ, wO, kCache, vCache, dModel, nHeads, nKVHeads, headDim, kvLen, base, scale, offset, eps, false)
}

// AttentionBlockInto is AttentionBlock with caller-owned output storage. If out
// has enough capacity, the final residual add writes directly into out through a
// pinned no-copy Metal buffer; otherwise a correctly sized output is allocated
// and returned.
func AttentionBlockInto(out []byte, x, normWeight, wQ, wO, kCache, vCache []byte, dModel, nHeads, nKVHeads, headDim, kvLen int, base, scale float32, offset int, eps float32) ([]byte, error) {
	return attentionBlockInto(out, x, normWeight, wQ, wO, kCache, vCache, dModel, nHeads, nKVHeads, headDim, kvLen, base, scale, offset, eps, true)
}

func attentionBlockInto(out []byte, x, normWeight, wQ, wO, kCache, vCache []byte, dModel, nHeads, nKVHeads, headDim, kvLen int, base, scale float32, offset int, eps float32, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	qDim := nHeads * headDim
	if len(x) != dModel*bf16Size || len(normWeight) != dModel*bf16Size {
		return nil, core.NewError("native.AttentionBlock: x/normWeight must be dModel bf16 bytes")
	}
	if len(wQ) != qDim*dModel*bf16Size || len(wO) != dModel*qDim*bf16Size {
		return nil, core.NewError("native.AttentionBlock: wQ/wO size mismatch")
	}
	if len(kCache) != nKVHeads*kvLen*headDim*bf16Size || len(vCache) != nKVHeads*kvLen*headDim*bf16Size {
		return nil, core.NewError("native.AttentionBlock: kCache/vCache size mismatch")
	}

	outLen := dModel * bf16Size
	callerOut := useCallerOut && cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}
	var encErr error
	withAutoreleasePool(func() {
		ioScratch, err := getQMVBF16Scratch(dModel, dModel)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVBF16Scratch(ioScratch)
		xBuf, outBuf, err := ioScratch.buffers(x)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if callerOut {
			if tmp, ok := ioScratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}
		nwBuf := residentBytes(normWeight)
		wqBuf, woBuf := residentBytes(wQ), residentBytes(wO)
		kvScratch, err := getAttentionBlockKVScratch(len(kCache), len(vCache))
		if err != nil {
			encErr = err
			return
		}
		defer putAttentionBlockKVScratch(kvScratch)
		kBuf, vBuf, ok, err := kvScratch.buffersNoCopy(kCache, vCache)
		if err != nil {
			encErr = err
			return
		}
		if !ok {
			kBuf, vBuf, err = kvScratch.buffers(kCache, vCache)
			if err != nil {
				encErr = err
				return
			}
		}
		off := int32(offset)
		offBuf := scalarI32(off)
		sc := getAttnScratch(dModel, qDim, nKVHeads*headDim, nHeads, 0)
		defer putAttnScratch(sc)

		rmsPSO, err := pipelineFor(rmsKernelBF16(dModel))
		if err != nil {
			encErr = err
			return
		}
		rmsTG := rmsThreadgroup(dModel, rmsPSO)
		qPlan, err := newBF16GemvPlan(qDim, dModel)
		if err != nil {
			encErr = err
			return
		}
		oPlan, err := newBF16GemvPlan(dModel, qDim)
		if err != nil {
			encErr = err
			return
		}
		ropePSO, err := ropePipelineBF16(false)
		if err != nil {
			encErr = err
			return
		}
		sdpaPSO, err := sdpaVectorPipelineForHeadDim(headDim)
		if err != nil {
			encErr = err
			return
		}
		addPSO, err := pipelineFor("vv_Addbfloat16")
		if err != nil {
			encErr = err
			return
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		emitRMSNorm(sink, rmsPSO, xBuf, nwBuf, sc.normed, 0, dModel, eps, rmsTG)
		emitBF16GemvPlan(sink, qPlan, wqBuf, sc.normed, sc.q, dModel, qDim)
		emitRopeAt(sink, ropePSO, sc.q, sc.qr, 0, 0, offBuf, 0, nil, nHeads, headDim, headDim, scale, float32(math.Log2(float64(base))))
		emitSDPA(sink, sdpaPSO, sc.qr, kBuf, vBuf, sc.attn, 0, nil, nHeads, nKVHeads, kvLen, int64(kvLen*headDim), int64(headDim), int64(kvLen*headDim), int64(headDim), scale)
		emitBF16GemvPlan(sink, oPlan, woBuf, sc.attn, sc.attnOut, qDim, dModel)
		emitBinary(sink, addPSO, xBuf, 0, sc.attnOut, 0, outBuf, 0, dModel)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, ioScratch.out.bytes[:len(out)])
		}
	})
	return out, encErr
}
