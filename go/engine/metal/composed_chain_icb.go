// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
)

// composed_chain_icb.go — CB recording for the composed whole-token chain (#18 queue item 4):
// the L=1 decode command stream is FIXED across tokens (same kernels, same buffers, same
// constants — only the position and the hidden's bytes change), so the chain records it once
// into an MTLIndirectCommandBuffer and every later token replays it with ONE
// executeCommandsInBuffer call instead of re-encoding ~10³ dispatches on the host. The probe
// that sized this: encode+host is 22% of a 0.8B decode token and 10% of a 27B one.
//
// The mechanism mirrors the arch lane's ICB replay (decode_forward_arch_icb.go):
//   - every scalar binds a process-memoised buffer (ICB commands cannot SetBytes);
//   - the per-token POSITION binds the recording's posBuf — the host bumps one int32 and the
//     recorded qprep/kprep/vappend/sdpa commands read the new position;
//   - dependency edges carry recorded SetBarriers (the ICB has no hazard tracking);
//   - the replay encoder marks every recorded-bound buffer resident with UseResources.
//
// chainTarget is the seam that lets ONE body serve both paths: the live chain bodies encode
// through it onto their per-layer encoders exactly as before, and the recording pass drives
// the SAME bodies with ICB command sinks — no parallel emit stream to drift.

// customPipelineForICB builds (and caches, alongside pipelineForICB's cache) an ICB-capable
// pipeline for a CUSTOM-library kernel (the lthn_* set — pipelineForICB resolves the main
// metallib only).
func customPipelineForICB(name string) (metal.MTLComputePipelineState, error) {
	key := name + "|custom-icb"
	icbPSOMu.Lock()
	defer icbPSOMu.Unlock()
	if pso, ok := icbPSOCache[key]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.customPipelineForICB: custom library unavailable for " + name)
	}
	fn := customLibrary.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.customPipelineForICB: kernel " + name + " not found")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.customPipelineForICB", name, err)
	}
	icbPSOCache[key] = pso
	return pso, nil
}

// composedChainRecording is one recorded L=1 chain: the ICB, its bound-resource set (for the
// replay's UseResources), the position buffer every position-dependent command reads, the
// hidden pins the stream reads/writes, and the invalidation stamps (the attn caches' pin
// identities — a cap-doubling realloc rebinds, so the recording dies with the old pins).
type composedChainRecording struct {
	icb    metal.MTLIndirectCommandBuffer
	count  uint // commands recorded
	maxCmd uint

	posBuf metal.MTLBuffer
	posPtr *int32

	hA, hB *pinnedNoCopyBytes // the chain's ping-pong hiddens — hA is the input, final is finalPin
	final  *pinnedNoCopyBytes
	D      int

	// head fold: recorded terminal norm + head; logits read from logitsPin after the wait.
	logitsPin *pinnedNoCopyBytes
	headVocab int

	// resources the recorded commands bind — marked resident at every replay.
	resources   []metal.MTLResource
	resourceIDs []objc.ID
	resourceSet map[uintptr]struct{}

	// pins the recording owns (scratch staging shared across layers) — held alive here.
	pins []*pinnedNoCopyBytes

	// invalidation: the attn device states the stream binds, with the pin identities recorded.
	// A replay at a position past a state's capacity (or after a realloc) invalidates.
	attnStates []*attnKVDeviceState
	attnKPins  []uintptr
	attnVPins  []uintptr

	gdScratch *gatedDeltaQuantLayerScratch // shared gd staging, owned by the recording
	gdKey     gatedDeltaQuantLayerKey
	hasGD     bool
}

// track adds a buffer to the replay's UseResources set (deduplicated by buffer identity).
func (r *composedChainRecording) track(buf metal.MTLBuffer) {
	if buf == nil {
		return
	}
	id := buf.GetID()
	if id == 0 {
		return
	}
	key := uintptr(id)
	if _, ok := r.resourceSet[key]; ok {
		return
	}
	r.resourceSet[key] = struct{}{}
	r.resources = append(r.resources, buf)
	r.resourceIDs = append(r.resourceIDs, objc.ID(id))
}

// recSink is the recording-mode dispatchSink: one ICB command per emitted dispatch, scalars
// through the memoised scalar buffers, every binding tracked for UseResources.
type recSink struct {
	cmd metal.MTLIndirectComputeCommand
	rec *composedChainRecording
}

func (s recSink) setPSO(pso metal.MTLComputePipelineState) { setICBPSO(s.cmd, pso) }
func (s recSink) setBuf(buf metal.MTLBuffer, off, idx uint) {
	s.rec.track(buf)
	setICBKernelBuffer(s.cmd, buf, off, idx)
}
func (s recSink) setI32(v int32, idx uint) {
	b := scalarI32(v)
	s.rec.track(b)
	setICBKernelBuffer(s.cmd, b, 0, idx)
}
func (s recSink) setI64(v int64, idx uint) {
	b := scalarI64(v)
	s.rec.track(b)
	setICBKernelBuffer(s.cmd, b, 0, idx)
}
func (s recSink) setF32(v float32, idx uint) {
	b := scalarF32(v)
	s.rec.track(b)
	setICBKernelBuffer(s.cmd, b, 0, idx)
}
func (s recSink) dispatchThreads(grid, group metal.MTLSize) {
	concurrentDispatchThreads(s.cmd, grid, group)
}
func (s recSink) dispatchThreadgroups(grid, group metal.MTLSize) {
	concurrentDispatchThreadgroups(s.cmd, grid, group)
}

// chainTarget is where a chain-layer body lands its dispatches: the live per-layer encoder
// (re-encode every token) or the ICB recording. One body, both targets — the drift-killer.
type chainTarget struct {
	enc     metal.MTLComputeCommandEncoderObject // live mode
	rec     *composedChainRecording              // record mode
	pending bool                                 // record mode: next command carries a SetBarrier
	full    bool                                 // record mode: command capacity exhausted (recording aborts)
	err     error                                // record mode: first recording error
}

func (t *chainTarget) recording() bool { return t.rec != nil }

// cmd hands back the sink for ONE dispatch: the live encoder, or the next ICB command (with
// a barrier stamped when a dependency edge was marked since the previous command).
func (t *chainTarget) cmd() dispatchSink {
	if t.rec == nil {
		return encObjectSink{t.enc}
	}
	if t.rec.count >= t.rec.maxCmd {
		t.full = true
		if t.err == nil {
			t.err = core.NewError("native.composedChainRecording: command capacity exhausted")
		}
		// hand back a dead command sink target — the caller checks t.err at body end; binding
		// into command maxCmd-1 twice is harmless (the recording is abandoned).
		return recSink{cmd: indirectComputeCommandAtIndexFast(t.rec.icb, t.rec.maxCmd-1), rec: t.rec}
	}
	c := indirectComputeCommandAtIndexFast(t.rec.icb, t.rec.count)
	t.rec.count++
	if t.pending {
		setICBBarrier(c)
		t.pending = false
	}
	return recSink{cmd: c, rec: t.rec}
}

// barrier marks a dependency edge: the live path drains the encoder now; the recording stamps
// the NEXT command with a SetBarrier (ICB barriers attach to commands, not the stream).
func (t *chainTarget) barrier() {
	if t.rec == nil {
		memoryBarrierObject(t.enc, metal.MTLBarrierScopeBuffers)
		return
	}
	t.pending = true
}

// pso resolves a MAIN-metallib kernel for this target (live pipeline vs ICB-capable pipeline).
func (t *chainTarget) pso(name string) (metal.MTLComputePipelineState, error) {
	if t.rec == nil {
		return pipelineFor(name)
	}
	return pipelineForICB(name)
}

// customPSO resolves a CUSTOM-library kernel for this target: the caller's memoised live
// resolver vs the ICB-capable build.
func (t *chainTarget) customPSO(live func() (metal.MTLComputePipelineState, error), name string) (metal.MTLComputePipelineState, error) {
	if t.rec == nil {
		return live()
	}
	return customPipelineForICB(name)
}

// posBind binds the per-token position at idx: the recording binds the posBuf (so a replayed
// command reads the bumped position); the live path binds it too when the ctx carries one —
// identical streams — and falls back to the inline scalar for target-less callers.
func (t *chainTarget) posBind(sink dispatchSink, posBuf metal.MTLBuffer, pos0 int, idx uint) {
	if posBuf != nil {
		sink.setBuf(posBuf, 0, idx)
		return
	}
	sink.setI32(int32(pos0), idx)
}

// composedChainReplay re-issues a recorded chain for one token: writes the input hidden into
// the recorded hA pin, bumps the position buffer, marks the bound set resident, executes the
// whole ICB range on one serial encoder, waits, and reads back the final hidden (+ the head
// logits when the recording folded the head). ok=false (no error) means the recording no
// longer serves (position past a recorded cache's capacity, or a cache pin was reallocated)
// — the caller falls back to the re-encode chain and re-records.
func composedChainReplay(rec *composedChainRecording, h []float32, pos int) (y []float32, logits []float32, ok bool, err error) {
	if rec == nil || rec.icb == nil {
		return nil, nil, false, nil
	}
	for i, st := range rec.attnStates {
		if st == nil || st.kBuf == nil || st.vBuf == nil ||
			uintptr(unsafe.Pointer(st.kBuf)) != rec.attnKPins[i] ||
			uintptr(unsafe.Pointer(st.vBuf)) != rec.attnVPins[i] {
			return nil, nil, false, nil // cache realloc — the recorded bindings are stale
		}
		if st.n != pos {
			return nil, nil, false, nil // position desync — never replay over the wrong slot
		}
		if (pos+1)*st.KVH*st.HD*4 > len(st.kBuf.bytes) {
			return nil, nil, false, nil // past capacity — the live path grows, then re-records
		}
	}
	copy(rec.hA.bytes, float32Bytes(h))
	*rec.posPtr = int32(pos)
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		useResourcesIDsFastObject(enc, rec.resources, rec.resourceIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
		executeCommandsInBufferWithRangeObjectFast(enc, rec.icb, foundation.NSRange{Location: 0, Length: rec.count})
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
	})
	for _, st := range rec.attnStates {
		st.n = pos + 1
	}
	y = make([]float32, rec.D)
	copy(y, unsafe.Slice((*float32)(unsafe.Pointer(&rec.final.bytes[0])), rec.D))
	if rec.logitsPin != nil {
		logits = make([]float32, rec.headVocab)
		copy(logits, unsafe.Slice((*float32)(unsafe.Pointer(&rec.logitsPin.bytes[0])), rec.headVocab))
	}
	return y, logits, true, nil
}

// composedChainRecordingPool has no pool: a recording is session-lifetime state (its pins and
// ICB bind that session's caches). Dropping the reference releases it; the gd scratch returns
// to its pool through release.
func (r *composedChainRecording) release() {
	if r == nil {
		return
	}
	if r.gdScratch != nil {
		putGatedDeltaQuantLayerScratch(r.gdKey, r.gdScratch)
		r.gdScratch = nil
	}
	r.icb = nil
	r.pins = nil
	r.resources = nil
	r.resourceIDs = nil
	r.resourceSet = nil
}

var composedChainICBMu sync.Mutex // recording is single-flight per process (cheap, rare)
