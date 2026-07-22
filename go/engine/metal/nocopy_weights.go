// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
	"github.com/tmc/apple/metal"
)

// This file is the zero-copy weight path's resolver. A checkpoint is memory-mapped once
// (safetensors.LoadDirMmap → *DirMapping), each shard's page-aligned mmap is wrapped in ONE
// Metal no-copy buffer (bytesNoCopy — proven GPU-readable by TestNoCopyMmapGPURead), and every
// weight tensor (which VIEWS its shard's mmap) is addressed by its byte offset INTO that shard's
// buffer rather than uploaded into a fresh owned buffer. So a multi-GB checkpoint is never
// duplicated in heap or GPU memory, and the per-token LM-head re-upload balloon is gone (the
// head binds the persistent shard buffer at the embedding's offset, every token, with no upload).

// bufView is a weight bound zero-copy: a shard's no-copy Metal buffer plus the weight's byte
// offset into it. It replaces a per-weight owned (copied) buffer everywhere the decode binds
// weights — the projectors, the layer norms, the head/embed. off is fed to the enc ops'
// input-offset params (mirrors the output-offset the cache-write path already uses).
type bufView struct {
	buf metal.MTLBuffer
	off uint
}

// copyView uploads a weight into a fresh owned Metal buffer and returns it as a bufView at offset
// 0 — the COPY path's bufView constructor (the in-memory weight bytes the directory zero-copy path
// does not apply to: DecodeForwardArch/Quant from test bytes, the standalone step helpers). It is
// the bufView form of sharedBytes.
func copyView(b []byte) bufView { return bufView{buf: sharedBytes(b)} }

// copyOrNilView is copyView for an optional weight: an empty weight yields the zero bufView (nil
// buf), which the projector/norm bindings treat as "skip". The bufView form of sharedOrNil.
func copyOrNilView(b []byte) bufView {
	if len(b) == 0 {
		return bufView{}
	}
	return bufView{buf: sharedBytes(b)}
}

// shardBuffers owns a memory-mapped checkpoint and one no-copy Metal buffer per shard, and
// resolves a weight's []byte view (Tensor.Data, a sub-slice of a shard's mmap) to the (buffer,
// offset) that addresses it. It MUST outlive every bufView and every command buffer that binds
// one — the session holds it and Closes it on unload (which unmaps the shards AFTER the buffers
// are done). Build it inside a withAutoreleasePool (the no-copy buffers are objc "new" = retained,
// so they survive the pool; the Go reference here keeps them alive).
type shardBuffers struct {
	dm   *safetensors.DirMapping
	bufs []metal.MTLBuffer // one no-copy buffer per dm.Shards[i], same index
	// bases caches the start pointer of each shard's Data so bufFor avoids re-reading &Data[0]
	// (and stays correct even though Data is a field on a heap *Mapping).
	bases []uintptr
	ends  []uintptr
}

type mappedShardRange struct {
	start uintptr
	end   uintptr
}

// totalMappedBytes sums the mapped shard sizes — the checkpoint's weight bytes
// as the RAM-aware context default budgets them (zero-copy weights go resident
// as decode touches them, so steady state is the full mapping).
func (sb *shardBuffers) totalMappedBytes() uint64 {
	if sb == nil {
		return 0
	}
	var n uint64
	for i := range sb.bases {
		if sb.ends[i] > sb.bases[i] {
			n += uint64(sb.ends[i] - sb.bases[i])
		}
	}
	return n
}

var (
	mappedShardRangeMu sync.Mutex
	mappedShardRanges  []mappedShardRange
)

func registerMappedShardRanges(bases, ends []uintptr) {
	mappedShardRangeMu.Lock()
	defer mappedShardRangeMu.Unlock()
	for i := range bases {
		if bases[i] != 0 && ends[i] > bases[i] {
			mappedShardRanges = append(mappedShardRanges, mappedShardRange{start: bases[i], end: ends[i]})
		}
	}
}

func unregisterMappedShardRanges(bases, ends []uintptr) {
	mappedShardRangeMu.Lock()
	defer mappedShardRangeMu.Unlock()
	for i := range bases {
		start, end := bases[i], ends[i]
		if start == 0 || end <= start {
			continue
		}
		out := mappedShardRanges[:0]
		for _, r := range mappedShardRanges {
			if r.start == start && r.end == end {
				continue
			}
			out = append(out, r)
		}
		mappedShardRanges = out
	}
}

func isMappedShardBytes(b []byte) bool {
	if len(b) == 0 {
		return false
	}
	p := uintptr(unsafe.Pointer(&b[0]))
	mappedShardRangeMu.Lock()
	defer mappedShardRangeMu.Unlock()
	for _, r := range mappedShardRanges {
		if p >= r.start && p < r.end {
			return true
		}
	}
	return false
}

// newShardBuffers wraps each shard's page-aligned mmap in a no-copy Metal buffer, the validated
// pattern from TestNoCopyMmapGPURead: NewBufferWithBytesNoCopyLengthOptionsDeallocator over
// &Data[0] with a non-nil no-op deallocator (the binding always invokes it; the mmap's lifetime
// is owned by dm.Close, not the buffer). MUST be called inside a withAutoreleasePool. The
// returned shardBuffers takes ownership of dm — its Close unmaps the shards.
func newShardBuffers(dm *safetensors.DirMapping) (*shardBuffers, error) {
	// The directory loaders build the shard buffers BEFORE the session constructor's ensureInit, so
	// ensure the shared device exists here — otherwise device is nil and every no-copy buffer comes
	// back unbacked (Contents != base). (Latent until a process's FIRST native call is a Dir load.)
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if dm == nil {
		return nil, core.NewError("native.newShardBuffers: nil DirMapping")
	}
	sb := &shardBuffers{
		dm:    dm,
		bufs:  make([]metal.MTLBuffer, len(dm.Shards)),
		bases: make([]uintptr, len(dm.Shards)),
		ends:  make([]uintptr, len(dm.Shards)),
	}
	for i, m := range dm.Shards {
		if m == nil || len(m.Data) == 0 {
			return nil, core.NewError("native.newShardBuffers: empty shard mapping")
		}
		base := unsafe.Pointer(&m.Data[0])
		buf := newNoCopyBuffer(base, uint(len(m.Data)))
		if buf.Contents() != base {
			return nil, core.NewError("native.newShardBuffers: no-copy buffer not backed by the mmap (bytesNoCopy rejected the page-aligned mapping)")
		}
		sb.bufs[i] = buf
		sb.bases[i] = uintptr(base)
		sb.ends[i] = uintptr(base) + uintptr(len(m.Data))
	}
	registerMappedShardRanges(sb.bases, sb.ends)
	return sb, nil
}

// bufFor resolves a weight (a Tensor.Data view into one shard's mmap) to the no-copy buffer that
// backs it and the weight's byte offset into that buffer. An empty weight ([]byte of len 0 — an
// absent optional weight, e.g. a K==V layer's missing v_proj) returns the zero bufView (nil buf),
// which the projector/norm bindings already treat as "skip". A non-empty weight whose first byte
// lies in no shard is a programming error (the weight didn't come from this mapping) and errors.
// bufForAligned resolves a weight to its no-copy shard view, OR an aligned owned copy when the weight's
// byte offset into the shard isn't a multiple of align. Metal's setBuffer:offset cannot do a misaligned
// read of the element type — bf16 reads need 2-byte alignment, the 4-bit affine_qmv's packed uint32
// weights need 4. A non-element-length tensor early in the checkpoint shifts every weight after it
// off-alignment (E4B-bf16: 1777/2076 tensors → odd offsets; the GPU reads a WRONG-but-valid weight →
// NaN). Copies go through residentBytes, which caches+pins by address so a tied/re-resolved weight
// copies once. Empty weight ([]byte len 0 — an absent optional) returns the zero bufView ("skip").
func (s *shardBuffers) bufForAligned(weight []byte, align uint) (bufView, error) {
	if len(weight) == 0 {
		return bufView{}, nil
	}
	p := uintptr(unsafe.Pointer(&weight[0]))
	for i := range s.bufs {
		if p >= s.bases[i] && p < s.ends[i] {
			off := uint(p - s.bases[i])
			if off%align != 0 {
				return bufView{buf: residentBytes(weight), off: 0}, nil
			}
			return bufView{buf: s.bufs[i], off: off}, nil
		}
	}
	// A tensor widened from F16 to BF16 at load (WidenF16TensorsToBF16) is a fresh heap buffer, not a
	// shard mmap view — legitimately off-mmap. Bind it resident (a small companion copy). Likewise a
	// REGISTERED OWNED tensor — a load-time synthesis the mapping adopted (a packExperts MoE pack, a
	// b1→b2-repacked weight; safetensors.AdoptOwnedTensors): residentBytes pins the slice beside its
	// device buffer, so the heap bytes can neither move nor be collected while bound, and Close
	// evicts the owned ranges with the session. This keeps the strict guard below for a NON-registered
	// off-shard weight, which is still a wrong-mapping bug.
	if s.dm != nil && (s.dm.IsWidened(weight) || s.dm.IsOwned(weight)) {
		return bufView{buf: residentBytes(weight), off: 0}, nil
	}
	return bufView{}, core.NewError("native.shardBuffers.bufForAligned: weight is not a view into any mapped shard")
}

// bufFor resolves a bf16 weight (2-byte element alignment). See bufForAligned.
func (s *shardBuffers) bufFor(weight []byte) (bufView, error) {
	return s.bufForAligned(weight, bf16Size)
}

// bufForNorm resolves a NORM weight: a no-copy shard view when the weight IS a view into a mapped
// shard (the qwen/gemma4 path — norms bind zero-copy like the projections), or a small resident copy
// when it is SYNTHESISED. A gemma-family checkpoint folds the "(1 + weight)" RMSNorm convention into
// every norm at load (ArchSpec.NormBiasOne → foldNormBiasOne), producing a FRESH heap buffer that is
// not a view into any mapped shard — so the strict projection resolver (bufFor/mustBufFor) rightly
// errors on it ("weight is not a view"), because a PROJECTION that is not a shard view is a wrong-
// mapping bug and must stay zero-copy. A folded norm is legitimate and, unlike the tied head or a
// projection, is a tiny vector bound ONCE per session, so the resident copy carries no per-token
// balloon. This mirrors the MoE router-norm treatment (RouterNormWScaled, itself a RootSize-folded
// synthesis, already binds resident). Empty weight (an absent optional norm) yields the zero bufView.
func (s *shardBuffers) bufForNorm(weight []byte) bufView {
	if len(weight) == 0 {
		return bufView{}
	}
	if v, err := s.bufForAligned(weight, bf16Size); err == nil {
		return v
	}
	return bufView{buf: residentBytes(weight)}
}

// mustBufForAligned is bufForAligned with the error folded into a shared ferr — the assembler/build
// pattern so a long sequence of resolutions short-circuits on the first failure. A nil receiver (the
// copy path: no shardBuffers) returns the zero bufView so callers branch on s == nil once.
func (s *shardBuffers) mustBufForAligned(weight []byte, align uint, ferr *error) bufView {
	if s == nil || *ferr != nil {
		return bufView{}
	}
	v, err := s.bufForAligned(weight, align)
	if err != nil {
		*ferr = err
	}
	return v
}

// mustBufFor resolves a bf16 (2-byte) weight; mustBufFor4 a 4-bit packed uint32 (4-byte) weight.
func (s *shardBuffers) mustBufFor(weight []byte, ferr *error) bufView {
	return s.mustBufForAligned(weight, bf16Size, ferr)
}
func (s *shardBuffers) mustBufFor4(weight []byte, ferr *error) bufView {
	return s.mustBufForAligned(weight, 4, ferr)
}

// Close unmaps the checkpoint. Call exactly once, AFTER every command buffer that bound a shard
// buffer has completed (using a no-copy buffer over an unmapped shard is a use-after-unmap). The
// no-copy Metal buffers reference the mmap, so they must be done first — in practice the session
// has finished decoding before unload. Safe on a nil/already-closed shardBuffers.
func (s *shardBuffers) Close() error {
	if s == nil || s.dm == nil {
		return nil
	}
	// Registered owned tensors (repacked/synthesised heap buffers — see bufForAligned) bound
	// resident through this session: release their device buffers and unpin with the session,
	// exactly like the shard-range eviction below. Owned packs can be GBs (a fully repacked
	// checkpoint), so session-scoped eviction is load-bearing, not tidiness.
	evictResidentBufsForRanges(s.dm.OwnedRanges())
	evictResidentBufsForRanges(s.bases, s.ends)
	unregisterMappedShardRanges(s.bases, s.ends)
	err := s.dm.Close()
	s.dm = nil
	s.bufs = nil
	s.bases = nil
	s.ends = nil
	return err
}
