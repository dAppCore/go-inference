// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"runtime"
	"slices"
	"sync"
	"unsafe"

	core "dappco.re/go"
	icbop "dappco.re/go/inference/engine/metal/ops/icb"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
)

type archICBPLEPlan struct {
	runtime                *archDecodePLEInputs
	pliDim                 int
	postNormBufs           []metal.MTLBuffer
	resident               []metal.MTLBuffer
	recordGate, recordProj func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer)
}

func (p *archICBPLEPlan) enabled() bool {
	return p != nil && p.runtime != nil && p.pliDim > 0
}

type archICBGemvShape struct {
	pso            metal.MTLComputePipelineState
	bm, bn, sm, tm int
}

type archICBLayerProjBuffers struct {
	wq, wk, wv, wo, wg, wu, wd metal.MTLBuffer
}

type archICBPLEProjBuffers struct {
	gate, proj metal.MTLBuffer
}

type archICBFFNScalarBuffers struct {
	fOut, dIn, dLd metal.MTLBuffer
}

type archICBSetupScratch struct {
	lFF, ffnWidthIndex []int
	uniqueDFF          []int
	ffUp, ffDown       []archICBGemvShape
	ffnScalars         []archICBFFNScalarBuffers
	anwBufs, mnwBufs   []metal.MTLBuffer
	qNormBufs          []metal.MTLBuffer
	kNormBufs          []metal.MTLBuffer
	postAttnBufs       []metal.MTLBuffer
	postFFBufs         []metal.MTLBuffer
	layerScalarBufs    []metal.MTLBuffer
	kCaches, vCaches   []metal.MTLBuffer
	lb                 []archICBLayerProjBuffers
	pleLB              []archICBPLEProjBuffers
	plePostNorms       []metal.MTLBuffer
	projResident       []metal.MTLBuffer
	pleResident        []metal.MTLBuffer
}

var archICBSetupScratchPool sync.Pool

func newArchICBSetupScratch(nLayers int) *archICBSetupScratch {
	return &archICBSetupScratch{
		lFF:             make([]int, nLayers),
		ffnWidthIndex:   make([]int, nLayers),
		uniqueDFF:       make([]int, 0, nLayers),
		ffUp:            make([]archICBGemvShape, 0, nLayers),
		ffDown:          make([]archICBGemvShape, 0, nLayers),
		ffnScalars:      make([]archICBFFNScalarBuffers, 0, nLayers),
		anwBufs:         make([]metal.MTLBuffer, nLayers),
		mnwBufs:         make([]metal.MTLBuffer, nLayers),
		qNormBufs:       make([]metal.MTLBuffer, nLayers),
		kNormBufs:       make([]metal.MTLBuffer, nLayers),
		postAttnBufs:    make([]metal.MTLBuffer, nLayers),
		postFFBufs:      make([]metal.MTLBuffer, nLayers),
		layerScalarBufs: make([]metal.MTLBuffer, nLayers),
		kCaches:         make([]metal.MTLBuffer, nLayers),
		vCaches:         make([]metal.MTLBuffer, nLayers),
		lb:              make([]archICBLayerProjBuffers, nLayers),
		pleLB:           make([]archICBPLEProjBuffers, nLayers),
		plePostNorms:    make([]metal.MTLBuffer, nLayers),
		projResident:    make([]metal.MTLBuffer, 0, nLayers*10+16),
		pleResident:     make([]metal.MTLBuffer, 0, nLayers*2+6),
	}
}

func (s *archICBSetupScratch) fits(nLayers int) bool {
	return s != nil &&
		cap(s.lFF) >= nLayers &&
		cap(s.ffnWidthIndex) >= nLayers &&
		cap(s.uniqueDFF) >= nLayers &&
		cap(s.ffUp) >= nLayers &&
		cap(s.ffDown) >= nLayers &&
		cap(s.ffnScalars) >= nLayers &&
		cap(s.anwBufs) >= nLayers &&
		cap(s.mnwBufs) >= nLayers &&
		cap(s.qNormBufs) >= nLayers &&
		cap(s.kNormBufs) >= nLayers &&
		cap(s.postAttnBufs) >= nLayers &&
		cap(s.postFFBufs) >= nLayers &&
		cap(s.layerScalarBufs) >= nLayers &&
		cap(s.kCaches) >= nLayers &&
		cap(s.vCaches) >= nLayers &&
		cap(s.lb) >= nLayers &&
		cap(s.pleLB) >= nLayers &&
		cap(s.plePostNorms) >= nLayers &&
		cap(s.projResident) >= nLayers*10+16 &&
		cap(s.pleResident) >= nLayers*2+6
}

func (s *archICBSetupScratch) reset(nLayers int) *archICBSetupScratch {
	clear(s.lFF)
	clear(s.ffnWidthIndex)
	clear(s.uniqueDFF)
	clear(s.ffUp)
	clear(s.ffDown)
	clear(s.ffnScalars)
	clear(s.anwBufs)
	clear(s.mnwBufs)
	clear(s.qNormBufs)
	clear(s.kNormBufs)
	clear(s.postAttnBufs)
	clear(s.postFFBufs)
	clear(s.layerScalarBufs)
	clear(s.kCaches)
	clear(s.vCaches)
	clear(s.lb)
	clear(s.pleLB)
	clear(s.plePostNorms)
	clear(s.projResident)
	clear(s.pleResident)
	s.lFF = s.lFF[:nLayers]
	s.ffnWidthIndex = s.ffnWidthIndex[:nLayers]
	s.uniqueDFF = s.uniqueDFF[:0]
	s.ffUp = s.ffUp[:0]
	s.ffDown = s.ffDown[:0]
	s.ffnScalars = s.ffnScalars[:0]
	s.anwBufs = s.anwBufs[:nLayers]
	s.mnwBufs = s.mnwBufs[:nLayers]
	s.qNormBufs = s.qNormBufs[:nLayers]
	s.kNormBufs = s.kNormBufs[:nLayers]
	s.postAttnBufs = s.postAttnBufs[:nLayers]
	s.postFFBufs = s.postFFBufs[:nLayers]
	s.layerScalarBufs = s.layerScalarBufs[:nLayers]
	s.kCaches = s.kCaches[:nLayers]
	s.vCaches = s.vCaches[:nLayers]
	s.lb = s.lb[:nLayers]
	s.pleLB = s.pleLB[:nLayers]
	s.plePostNorms = s.plePostNorms[:nLayers]
	s.projResident = s.projResident[:0]
	s.pleResident = s.pleResident[:0]
	return s
}

func getArchICBSetupScratch(nLayers int) *archICBSetupScratch {
	if v := archICBSetupScratchPool.Get(); v != nil {
		s := v.(*archICBSetupScratch)
		if s.fits(nLayers) {
			return s.reset(nLayers)
		}
	}
	return newArchICBSetupScratch(nLayers)
}

func putArchICBSetupScratch(s *archICBSetupScratch) {
	if s != nil {
		archICBSetupScratchPool.Put(s.reset(0))
	}
}

// archICBReplay is a recorded arch ICB held for incremental replay: recordArchICB builds it ONCE
// (the decode stack baked into icb) and each stepBody replays it for ONE token over the growing
// cache with cheap per-token offset rebinds. The batch core records it + runBatch-loops every
// token (byte-identical to the old single-call core); the ArchSession holds it across StepWithID
// calls for the per-token encode-bypass. Every buffer + the icb is retained (scratchBF16 /
// device.New* return owned objects, like the session's own caches), so the struct survives the
// per-step autorelease pools.
type archICBReplay struct {
	icb                               metal.MTLIndirectCommandBuffer
	rng                               foundation.NSRange
	residentRes                       []metal.MTLResource
	residentResIDs                    []objc.ID
	scratch                           *archICBReplayScratch
	specs                             []model.LayerSpec
	nLayers                           int
	vOutBind                          uint
	kRopeBind                         uint // K cache-write buffer index: 1 for plain rope, 2 for the fused qk-norm+rope op
	hasValueNorm                      bool
	kRopeIdx, vIdx, vNormIdx, sdpaIdx []int
	barrierOps                        []int // fine-grained replay: op indices to insert an encoder memory barrier before
	kCaches, vCaches                  []metal.MTLBuffer
	// q8 KV (#367): per-layer int8+scale caches on GLOBAL owners; the store
	// ops' output offsets (kStoreIdx/vStoreIdx binds 1+2) are the per-token
	// rebind targets — the rope/vProj/vNorm binds stay fixed on staging.
	kvQ8                            *archICBKVQ8
	kStoreIdx, vStoreIdx            []int
	kCachePtrs, vCachePtrs          []*byte
	offBuf, nGlobalBuf, nSlidingBuf metal.MTLBuffer
	offPtr, nGlobalPtr, nSlidingPtr *int32
	ping                            [2]metal.MTLBuffer
	ping0, lastOut, pleInput        metal.MTLBuffer
	ping0Ptr, pleInputPtr           *byte
	lastOutPtr                      *byte
	finalOutIdx                     int
	finalOutBind                    uint
	finalOutBufID                   objc.ID
	hasFinalOut                     bool
	hasPLE                          bool
	plePliDim                       int
	pleRuntime                      *archDecodePLEInputs
	opsPerLayer                     uint
	rowBytes                        []int // per-layer KV cache row stride (nKVHeads·hd·bf16Size) — gemma4 global layers are wider
	cacheRows                       []int // per-layer physical row CAPACITY of kCaches[li]/vCaches[li] (bufferLength/rowBytes).
	// A sliding owner allocated at slidingWindow rows (the bounded-memory fix) makes this a
	// ring; a global (or not-yet-bounded) owner allocated at maxLen makes pos%cacheRows a
	// no-op (pos < maxLen always), so prepareStepRebind is byte-identical to the old
	// unconditional linear write/read in that case.
	slidingWindow, dModel int
}

type archICBReplayScratch struct {
	dModel, maxQd, maxKvd, maxDFF, maxGelu int
	nLayers, pleInputElems, pleDim         int
	hasFusedGELU, hasPLE                   bool
	normed, q, qr, kProj, attn, attnOut    metal.MTLBuffer
	kThrow, vThrow, mlpNormed              metal.MTLBuffer
	gate, up, gated, down                  metal.MTLBuffer
	x2, x3, x3s, inner                     metal.MTLBuffer
	scaled, tnh, onePlus, halfG, gelu      metal.MTLBuffer
	c044, c079, c1c, c05                   metal.MTLBuffer
	pleInput, pleGate, pleGated            metal.MTLBuffer
	pleProj, pleNorm                       metal.MTLBuffer
	ping                                   [2]metal.MTLBuffer
	hBufs                                  []metal.MTLBuffer
	offBuf, nGlobalBuf, nSlidingBuf        metal.MTLBuffer
	kRopeIdx, vIdx, vNormIdx, sdpaIdx      []int
	barrierOps, rowBytes, cacheRows        []int
	residentRes                            []metal.MTLResource
	residentResIDs                         []objc.ID
	outputViewPtrs                         []uintptr
	outputViewLens                         []int
	outputViewBufs                         []metal.MTLBuffer
	outputViewPinned                       []*pinnedNoCopyBytes
	outputResidentRes                      []metal.MTLResource
	outputResidentIDs                      []objc.ID
}

var archICBReplayScratchPool sync.Pool

func newArchICBReplayScratch(dModel, maxQd, maxKvd, maxDFF, maxGelu, nLayers, pleInputElems, pleDim int, hasFusedGELU, hasPLE bool) *archICBReplayScratch {
	s := &archICBReplayScratch{
		dModel: dModel, maxQd: maxQd, maxKvd: maxKvd, maxDFF: maxDFF, maxGelu: maxGelu,
		nLayers: nLayers, pleInputElems: pleInputElems, pleDim: pleDim, hasFusedGELU: hasFusedGELU, hasPLE: hasPLE,
		normed:      scratchBF16(dModel),
		q:           scratchBF16(maxQd),
		qr:          scratchBF16(maxQd),
		kProj:       scratchBF16(maxKvd),
		attn:        scratchBF16(maxQd),
		attnOut:     scratchBF16(dModel),
		kThrow:      scratchBF16(maxKvd),
		vThrow:      scratchBF16(maxKvd),
		mlpNormed:   scratchBF16(dModel),
		gate:        scratchBF16(maxDFF),
		up:          scratchBF16(maxDFF),
		gated:       scratchBF16(maxDFF),
		down:        scratchBF16(dModel),
		ping:        [2]metal.MTLBuffer{scratchBF16(dModel), scratchBF16(dModel)},
		hBufs:       make([]metal.MTLBuffer, nLayers),
		offBuf:      device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared),
		nGlobalBuf:  device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared),
		nSlidingBuf: device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared),
		kRopeIdx:    make([]int, nLayers),
		vIdx:        make([]int, nLayers),
		vNormIdx:    make([]int, nLayers),
		sdpaIdx:     make([]int, nLayers),
		barrierOps:  make([]int, 0, nLayers*24),
		rowBytes:    make([]int, nLayers),
		cacheRows:   make([]int, nLayers),
		residentRes: make([]metal.MTLResource, 0, nLayers*48+96),
	}
	for i := range s.hBufs {
		s.hBufs[i] = scratchBF16(dModel)
	}
	if !hasFusedGELU {
		s.x2, s.x3, s.x3s, s.inner = scratchBF16(maxGelu), scratchBF16(maxGelu), scratchBF16(maxGelu), scratchBF16(maxGelu)
		s.scaled, s.tnh, s.onePlus, s.halfG = scratchBF16(maxGelu), scratchBF16(maxGelu), scratchBF16(maxGelu), scratchBF16(maxGelu)
		s.gelu = scratchBF16(maxGelu)
		s.c044 = bf16ConstBuffer(maxGelu, 0.044715)
		s.c079 = bf16ConstBuffer(maxGelu, 0.7978845608028654)
		s.c1c = bf16ConstBuffer(maxGelu, 1.0)
		s.c05 = bf16ConstBuffer(maxGelu, 0.5)
	}
	if hasPLE {
		s.pleInput = scratchBF16(pleInputElems)
		s.pleGate = scratchBF16(pleDim)
		s.pleGated = scratchBF16(pleDim)
		s.pleProj = scratchBF16(dModel)
		s.pleNorm = scratchBF16(dModel)
	}
	return s
}

func (s *archICBReplayScratch) matches(dModel, maxQd, maxKvd, maxDFF, maxGelu, nLayers, pleInputElems, pleDim int, hasFusedGELU, hasPLE bool) bool {
	if s == nil || s.dModel != dModel || s.maxQd != maxQd || s.maxKvd != maxKvd || s.maxDFF != maxDFF || s.maxGelu != maxGelu ||
		s.nLayers != nLayers || s.pleInputElems != pleInputElems || s.pleDim != pleDim || s.hasFusedGELU != hasFusedGELU || s.hasPLE != hasPLE {
		return false
	}
	if s.normed == nil || s.q == nil || s.qr == nil || s.kProj == nil || s.attn == nil || s.attnOut == nil ||
		s.kThrow == nil || s.vThrow == nil || s.mlpNormed == nil || s.gate == nil || s.up == nil || s.gated == nil || s.down == nil ||
		s.ping[0] == nil || s.ping[1] == nil || s.offBuf == nil || s.nGlobalBuf == nil || s.nSlidingBuf == nil {
		return false
	}
	if len(s.hBufs) != nLayers || len(s.kRopeIdx) != nLayers || len(s.vIdx) != nLayers || len(s.vNormIdx) != nLayers || len(s.sdpaIdx) != nLayers || len(s.rowBytes) != nLayers || len(s.cacheRows) != nLayers {
		return false
	}
	for _, h := range s.hBufs {
		if h == nil {
			return false
		}
	}
	if !hasFusedGELU && (s.x2 == nil || s.x3 == nil || s.x3s == nil || s.inner == nil || s.scaled == nil || s.tnh == nil || s.onePlus == nil || s.halfG == nil || s.gelu == nil || s.c044 == nil || s.c079 == nil || s.c1c == nil || s.c05 == nil) {
		return false
	}
	if hasPLE && (s.pleInput == nil || s.pleGate == nil || s.pleGated == nil || s.pleProj == nil || s.pleNorm == nil) {
		return false
	}
	return true
}

func getArchICBReplayScratch(dModel, maxQd, maxKvd, maxDFF, maxGelu, nLayers, pleInputElems, pleDim int, hasFusedGELU, hasPLE bool) *archICBReplayScratch {
	if v := archICBReplayScratchPool.Get(); v != nil {
		s := v.(*archICBReplayScratch)
		if s.matches(dModel, maxQd, maxKvd, maxDFF, maxGelu, nLayers, pleInputElems, pleDim, hasFusedGELU, hasPLE) {
			return s
		}
	}
	return newArchICBReplayScratch(dModel, maxQd, maxKvd, maxDFF, maxGelu, nLayers, pleInputElems, pleDim, hasFusedGELU, hasPLE)
}

func putArchICBReplayScratch(s *archICBReplayScratch) {
	if s != nil {
		archICBReplayScratchPool.Put(s)
	}
}

func (s *archICBReplayScratch) closeOutputViewAt(i int) {
	if s == nil || i < 0 || i >= len(s.outputViewBufs) {
		return
	}
	if i < len(s.outputViewPinned) && s.outputViewPinned[i] != nil {
		s.outputViewPinned[i].Close()
		s.outputViewPinned[i] = nil
	}
	s.outputViewPtrs[i] = 0
	s.outputViewLens[i] = 0
	s.outputViewBufs[i] = nil
}

func (s *archICBReplayScratch) closeOutputViews() {
	if s == nil {
		return
	}
	for i := range s.outputViewBufs {
		s.closeOutputViewAt(i)
	}
	s.outputViewPtrs = nil
	s.outputViewLens = nil
	s.outputViewBufs = nil
	s.outputViewPinned = nil
}

func (s *archICBReplayScratch) outputViews(outputs [][]byte, outLen int) ([]metal.MTLBuffer, bool) {
	if s == nil || outLen <= 0 || len(outputs) == 0 {
		return nil, false
	}
	for i := range outputs {
		if len(outputs[i]) != outLen {
			return nil, false
		}
	}
	T := len(outputs)
	if cap(s.outputViewBufs) < T {
		s.closeOutputViews()
		s.outputViewPtrs = make([]uintptr, T)
		s.outputViewLens = make([]int, T)
		s.outputViewBufs = make([]metal.MTLBuffer, T)
		s.outputViewPinned = make([]*pinnedNoCopyBytes, T)
	} else {
		for i := T; i < len(s.outputViewBufs); i++ {
			s.closeOutputViewAt(i)
		}
		s.outputViewPtrs = s.outputViewPtrs[:T]
		s.outputViewLens = s.outputViewLens[:T]
		s.outputViewBufs = s.outputViewBufs[:T]
		s.outputViewPinned = s.outputViewPinned[:T]
	}
	for i := range outputs {
		ptr := uintptr(unsafe.Pointer(&outputs[i][0]))
		if s.outputViewBufs[i] != nil && s.outputViewPtrs[i] == ptr && s.outputViewLens[i] == outLen {
			continue
		}
		s.closeOutputViewAt(i)
		if buf, ok := registeredPinnedNoCopyBytes(outputs[i]); ok {
			s.outputViewPtrs[i] = ptr
			s.outputViewLens[i] = outLen
			s.outputViewBufs[i] = buf
			s.outputViewPinned[i] = nil
			continue
		}
		buf, pinner, noCopy := residentNoCopyBytes(outputs[i])
		if !noCopy {
			if pinner != nil {
				pinner.Unpin()
			}
			return nil, false
		}
		pinned := &pinnedNoCopyBytes{bytes: outputs[i], buf: buf, pinner: pinner}
		runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
		s.outputViewPtrs[i] = ptr
		s.outputViewLens[i] = outLen
		s.outputViewBufs[i] = buf
		s.outputViewPinned[i] = pinned
	}
	return s.outputViewBufs, true
}

func (s *archICBReplayScratch) outputResidentResources(base []metal.MTLResource, baseIDs []objc.ID, views []metal.MTLBuffer) ([]metal.MTLResource, []objc.ID) {
	if s == nil || len(views) == 0 {
		return nil, nil
	}
	n := len(base) + len(views)
	if cap(s.outputResidentRes) < n {
		s.outputResidentRes = make([]metal.MTLResource, n)
	} else {
		s.outputResidentRes = s.outputResidentRes[:n]
	}
	copy(s.outputResidentRes, base)
	for i, view := range views {
		s.outputResidentRes[len(base)+i] = view
	}
	if cap(s.outputResidentIDs) < n {
		s.outputResidentIDs = make([]objc.ID, n)
	} else {
		s.outputResidentIDs = s.outputResidentIDs[:n]
	}
	if len(baseIDs) == len(base) {
		copy(s.outputResidentIDs, baseIDs)
	} else {
		for i, res := range base {
			if res != nil {
				s.outputResidentIDs[i] = res.GetID()
			} else {
				s.outputResidentIDs[i] = 0
			}
		}
	}
	for i, view := range views {
		if view != nil {
			s.outputResidentIDs[len(base)+i] = view.GetID()
		} else {
			s.outputResidentIDs[len(base)+i] = 0
		}
	}
	return s.outputResidentRes, s.outputResidentIDs
}

func (s *archICBReplayScratch) outputResidentResource(base []metal.MTLResource, baseIDs []objc.ID, view metal.MTLBuffer) ([]metal.MTLResource, []objc.ID) {
	if s == nil || view == nil {
		return nil, nil
	}
	n := len(base) + 1
	if cap(s.outputResidentRes) < n {
		s.outputResidentRes = make([]metal.MTLResource, n)
	} else {
		s.outputResidentRes = s.outputResidentRes[:n]
	}
	copy(s.outputResidentRes, base)
	s.outputResidentRes[len(base)] = view
	if cap(s.outputResidentIDs) < n {
		s.outputResidentIDs = make([]objc.ID, n)
	} else {
		s.outputResidentIDs = s.outputResidentIDs[:n]
	}
	if len(baseIDs) == len(base) {
		copy(s.outputResidentIDs, baseIDs)
	} else {
		for i, res := range base {
			if res != nil {
				s.outputResidentIDs[i] = res.GetID()
			} else {
				s.outputResidentIDs[i] = 0
			}
		}
	}
	s.outputResidentIDs[len(base)] = view.GetID()
	return s.outputResidentRes, s.outputResidentIDs
}

func (r *archICBReplay) releaseScratch() {
	if r != nil && r.scratch != nil {
		putArchICBReplayScratch(r.scratch)
		r.scratch = nil
	}
}

func (r *archICBReplay) cacheKVContents() {
	if r == nil {
		return
	}
	if len(r.kCachePtrs) != len(r.kCaches) {
		r.kCachePtrs = make([]*byte, len(r.kCaches))
	}
	if len(r.vCachePtrs) != len(r.vCaches) {
		r.vCachePtrs = make([]*byte, len(r.vCaches))
	}
	for i, b := range r.kCaches {
		if b != nil {
			r.kCachePtrs[i] = (*byte)(bufferContentsFast(b))
		}
	}
	for i, b := range r.vCaches {
		if b != nil {
			r.vCachePtrs[i] = (*byte)(bufferContentsFast(b))
		}
	}
}

func (r *archICBReplay) cacheLastOutContents() {
	if r == nil || r.lastOut == nil {
		return
	}
	r.lastOutPtr = (*byte)(bufferContentsFast(r.lastOut))
}

func (r *archICBReplay) cacheStepContents() {
	if r == nil {
		return
	}
	if r.offBuf != nil {
		r.offPtr = (*int32)(bufferContentsFast(r.offBuf))
	}
	if r.nGlobalBuf != nil {
		r.nGlobalPtr = (*int32)(bufferContentsFast(r.nGlobalBuf))
	}
	if r.nSlidingBuf != nil {
		r.nSlidingPtr = (*int32)(bufferContentsFast(r.nSlidingBuf))
	}
	if r.ping0 != nil {
		r.ping0Ptr = (*byte)(bufferContentsFast(r.ping0))
	}
	if r.pleInput != nil {
		r.pleInputPtr = (*byte)(bufferContentsFast(r.pleInput))
	}
}

func (r *archICBReplay) copyLastOutInto(dst []byte) {
	if r == nil || r.lastOutPtr == nil {
		return
	}
	copy(dst, unsafe.Slice(r.lastOutPtr, r.dModel*bf16Size))
}

// stepBody replays the recorded ICB for ONE token at position pos over the growing cache. pli is
// this token's [nLayers·pliDim] PerLayerInputs tensor (nil for non-PLE); the caller computes it
// (ArchSession.StepWithID from the token id, runBatch from the batch token ids). Returns a
// fresh hidden copy (read out of the device buffer, so it survives the caller's pool). The caller
// wraps the call in withAutoreleasePool (StepWithID + runBatch both do).
func (r *archICBReplay) stepBody(inputEmb []byte, pos int, pli []byte) []byte {
	return r.stepBodyResult(inputEmb, pos, pli, true)
}

func (r *archICBReplay) stepBodyInto(inputEmb []byte, pos int, pli []byte, out []byte) []byte {
	r.stepBodyResult(inputEmb, pos, pli, false)
	r.copyLastOutInto(out)
	return out
}

func (r *archICBReplay) stepBodyIntoBuffer(inputEmb []byte, pos int, pli []byte, out metal.MTLBuffer) bool {
	if r == nil || r.scratch == nil || !r.hasFinalOut || r.icb == nil || out == nil {
		return false
	}
	if !r.bindStepOutput(out) {
		return false
	}
	residentRes, residentIDs := r.scratch.outputResidentResource(r.residentRes, r.residentResIDs, out)
	r.stepBodyResultWithResources(inputEmb, pos, pli, false, residentRes, residentIDs)
	return true
}

func (r *archICBReplay) encodeStepBodyIntoBuffer(enc metal.MTLComputeCommandEncoderObject, inputEmb []byte, pos int, pli []byte, out metal.MTLBuffer) (metal.MTLBuffer, bool) {
	if r == nil || r.scratch == nil || !r.hasFinalOut || r.icb == nil || out == nil {
		return nil, false
	}
	if !r.bindStepOutput(out) {
		return nil, false
	}
	r.prepareStep(inputEmb, pos, pli)
	residentRes, residentIDs := r.scratch.outputResidentResource(r.residentRes, r.residentResIDs, out)
	useResourcesIDsFastObject(enc, residentRes, residentIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
	executeCommandsInBufferWithRangeObjectFast(enc, r.icb, r.rng)
	return out, true
}

func (r *archICBReplay) bindStepOutput(out metal.MTLBuffer) bool {
	if r == nil || !r.hasFinalOut || r.icb == nil || out == nil {
		return false
	}
	if outID := out.GetID(); outID != 0 && r.finalOutBufID == outID {
		return true
	}
	cmd := indirectComputeCommandAtIndexFast(r.icb, uint(r.finalOutIdx))
	return r.bindStepOutputCommand(cmd, out)
}

func (r *archICBReplay) directOutputResources(outputs [][]byte, outLen int) ([]metal.MTLBuffer, []metal.MTLResource, []objc.ID, bool) {
	if r == nil || r.scratch == nil || !r.hasFinalOut {
		return nil, nil, nil, false
	}
	views, ok := r.scratch.outputViews(outputs, outLen)
	if !ok {
		r.scratch.closeOutputViews()
		return nil, nil, nil, false
	}
	resources, ids := r.scratch.outputResidentResources(r.residentRes, r.residentResIDs, views)
	return views, resources, ids, true
}

func (r *archICBReplay) bindStepOutputCommand(cmd metal.MTLIndirectComputeCommand, out metal.MTLBuffer) bool {
	if r == nil || !r.hasFinalOut || cmd == nil || out == nil {
		return false
	}
	setICBKernelBuffer(cmd, out, 0, r.finalOutBind)
	r.finalOutBufID = out.GetID()
	return true
}

func (r *archICBReplay) stepBodyDirectOutput(inputEmb []byte, pos int, pli []byte, out []byte, outCmd metal.MTLIndirectComputeCommand, outBuf metal.MTLBuffer, residentRes []metal.MTLResource, residentIDs []objc.ID) []byte {
	if !r.bindStepOutputCommand(outCmd, outBuf) {
		return r.stepBodyInto(inputEmb, pos, pli, out)
	}
	r.stepBodyResultWithResources(inputEmb, pos, pli, false, residentRes, residentIDs)
	return out
}

func (r *archICBReplay) stepBodyNoResult(inputEmb []byte, pos int, pli []byte) {
	r.stepBodyResult(inputEmb, pos, pli, false)
}

// encodeStepBody records this token's ICB replay into the caller-owned `enc` WITHOUT committing, so the
// caller can append more GPU work (the LM head + argmax) to the SAME command buffer and sync once per
// token instead of twice. Returns the device buffer holding this layer-stack's final hidden (r.lastOut),
// which the caller reads after the command buffer completes. Must run inside an autorelease pool.
func (r *archICBReplay) encodeStepBody(enc metal.MTLComputeCommandEncoderObject, inputEmb []byte, pos int, pli []byte) metal.MTLBuffer {
	r.bindStepOutput(r.lastOut)
	r.prepareStep(inputEmb, pos, pli)
	useResourcesIDsFastObject(enc, r.residentRes, r.residentResIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
	executeCommandsInBufferWithRangeObjectFast(enc, r.icb, r.rng)
	return r.lastOut
}

func (r *archICBReplay) stepBodyResult(inputEmb []byte, pos int, pli []byte, readResult bool) []byte {
	r.bindStepOutput(r.lastOut)
	return r.stepBodyResultWithResources(inputEmb, pos, pli, readResult, r.residentRes, r.residentResIDs)
}

func (r *archICBReplay) stepBodyResultWithResources(inputEmb []byte, pos int, pli []byte, readResult bool, residentRes []metal.MTLResource, residentIDs []objc.ID) []byte {
	r.prepareStep(inputEmb, pos, pli)
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	useResourcesIDsFastObject(enc, residentRes, residentIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
	if fineGrainedReplay && len(r.barrierOps) > 0 {
		// replay barrier-free ICB ranges with an encoder memory barrier at each recorded dep point —
		// resource-scoped coherency instead of the coarse all-prior drain.
		start := r.rng.Location
		for _, b := range r.barrierOps {
			bb := uint(b)
			executeCommandsInBufferWithRangeObjectFast(enc, r.icb, foundation.NSRange{Location: start, Length: bb - start})
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			start = bb
		}
		executeCommandsInBufferWithRangeObjectFast(enc, r.icb, foundation.NSRange{Location: start, Length: r.rng.Location + r.rng.Length - start})
	} else {
		executeCommandsInBufferWithRangeObjectFast(enc, r.icb, r.rng)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	if pieceTimingOn { // GPU execution span of the replay — vs the wall, splits GPU-side from host submit/wait
		icbGPUNs += int64(float64(cb.GPUEndTime()-cb.GPUStartTime()) * 1e9)
	}
	if !readResult {
		return nil
	}
	out := make([]byte, r.dModel*bf16Size)
	r.copyLastOutInto(out)
	return out
}

func (r *archICBReplay) stepBodyCapture(inputEmb []byte, pos int, pli []byte) (final []byte, perLayer [][]byte) {
	r.prepareStep(inputEmb, pos, pli)
	perLayer = make([][]byte, r.nLayers)
	for li := 0; li < r.nLayers; li++ {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		useResourcesIDsFastObject(enc, r.residentRes, r.residentResIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
		executeCommandsInBufferWithRangeObjectFast(enc, r.icb, foundation.NSRange{
			Location: uint(li) * r.opsPerLayer,
			Length:   r.opsPerLayer,
		})
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		row := make([]byte, r.dModel*bf16Size)
		copy(row, unsafe.Slice((*byte)(bufferContentsFast(r.ping[(li+1)%2])), r.dModel*bf16Size))
		perLayer[li] = row
	}
	if len(perLayer) > 0 {
		final = append([]byte(nil), perLayer[len(perLayer)-1]...)
	}
	return final, perLayer
}

func (r *archICBReplay) prepareStep(inputEmb []byte, pos int, pli []byte) {
	r.prepareStepRebind(pos)
	if r.hasPLE && pli != nil {
		want := r.nLayers * r.plePliDim * bf16Size
		copy(unsafe.Slice(r.pleInputPtr, want), pli)
	}
	copy(unsafe.Slice(r.ping0Ptr, r.dModel*bf16Size), inputEmb)
}

// prepareStepRebind does the position-dependent ICB rebind for one decode step — the offset/window
// counters + per-layer cache-row offsets — WITHOUT writing the input emb/pli. The chained-GPU decode
// path uses this: the next step's emb (→ping0) and pli (→pleInput) are produced on-GPU by the prior
// step's encNextInputsGPU, so the host must not overwrite them, only re-point the caches for `pos`.
func (r *archICBReplay) prepareStepRebind(pos int) {
	*r.offPtr = int32(pos)
	*r.nGlobalPtr = int32(pos + 1)
	win := pos + 1
	start := 0
	if r.slidingWindow > 0 && win > r.slidingWindow {
		start = win - r.slidingWindow
		win = r.slidingWindow
	}
	*r.nSlidingPtr = int32(win)
	for li := 0; li < r.nLayers; li++ {
		if r.specs[li].OwnsCache() {
			// Re-acquire the command from the retained icb each step: the handle from
			// IndirectComputeCommandAtIndex is a pool-scoped view that does NOT survive the
			// record pool's drain, but the icb + its recorded commands persist — so rebind by
			// op index. (The buffers + the icb are device.New*-owned, hence retained.)
			//
			// rowOff wraps into the owner's ACTUAL cache capacity, not the absolute position:
			// a sliding owner allocated at slidingWindow rows (the bounded-memory fix) turns
			// this into the ring write, evicting the slot that just left the window; an owner
			// still allocated at the full maxLen (global layers, or any not-yet-bounded caller)
			// has cacheRows>pos always, so pos%cacheRows==pos — byte-identical to the old
			// unconditional linear write.
			if r.kvQ8.on(li) {
				// q8 owner: the landing writes staging (fixed binds); rebind the two
				// quantise-store ops' cache row (int8: kvd bytes) + scale row.
				kvd := r.rowBytes[li] / bf16Size
				row := pos
				if rows := r.cacheRows[li]; rows > 0 {
					row = pos % rows
				}
				rowOff := uint(row * kvd)
				scOff := uint(row * (kvd / kvQ8GroupSize) * 4)
				setICBKernelBufferAtCommandIndexFast(r.icb, uint(r.kStoreIdx[li]), r.kCaches[li], rowOff, 1)
				setICBKernelBufferAtCommandIndexFast(r.icb, uint(r.kStoreIdx[li]), r.kvQ8.kScales[li], scOff, 2)
				setICBKernelBufferAtCommandIndexFast(r.icb, uint(r.vStoreIdx[li]), r.vCaches[li], rowOff, 1)
				setICBKernelBufferAtCommandIndexFast(r.icb, uint(r.vStoreIdx[li]), r.kvQ8.vScales[li], scOff, 2)
				continue
			}
			rowOff := uint(pos * r.rowBytes[li]) // per-layer: global layers' rows are wider (larger head_dim)
			if rows := r.cacheRows[li]; rows > 0 {
				rowOff = uint((pos % rows) * r.rowBytes[li])
			}
			setICBKernelBufferAtCommandIndexFast(r.icb, uint(r.kRopeIdx[li]), r.kCaches[li], rowOff, r.kRopeBind)
			setICBKernelBufferAtCommandIndexFast(r.icb, uint(r.vIdx[li]), r.vCaches[li], rowOff, r.vOutBind)
			if r.hasValueNorm {
				setICBKernelBufferAtCommandIndexFast(r.icb, uint(r.vNormIdx[li]), r.vCaches[li], rowOff, 0)
				setICBKernelBufferAtCommandIndexFast(r.icb, uint(r.vNormIdx[li]), r.vCaches[li], rowOff, 2)
			}
		}
		if r.specs[li].Attention == model.SlidingAttention {
			own := r.specs[li].KVShareFrom
			// A bounded ring (owner capacity <= slidingWindow) always attends from slot 0: once
			// the ring is full the whole physical buffer IS the live window (rows in slot order,
			// not chronological order — sound because softmax is permutation-invariant and every
			// cached row carries its own absolute-position RoPE baked in at write time; the same
			// reasoning the non-ICB sliding ring already relies on). An owner still on the linear
			// maxLen buffer keeps the old absolute offset into its untouched history.
			ownStart := start
			if rows := r.cacheRows[own]; rows > 0 && rows <= r.slidingWindow {
				ownStart = 0
			}
			slideOff := uint(ownStart * r.rowBytes[own]) // read the owner's cache at its row stride
			setICBKernelBufferAtCommandIndexFast(r.icb, uint(r.sdpaIdx[li]), r.kCaches[own], slideOff, 1)
			setICBKernelBufferAtCommandIndexFast(r.icb, uint(r.sdpaIdx[li]), r.vCaches[own], slideOff, 2)
		}
	}
}

// encodeStepBodyNoInput replays one decode step with the input emb+pli ALREADY in ping0/pleInput (the
// chained-GPU path: produced on-GPU by the prior step's encNextInputsGPU). It rebinds the caches for
// `pos` and replays — no host emb/pli write — returning lastOut (the post-stack hidden).
func (r *archICBReplay) encodeStepBodyNoInput(enc metal.MTLComputeCommandEncoderObject, pos int) metal.MTLBuffer {
	r.bindStepOutput(r.lastOut)
	r.prepareStepRebind(pos)
	useResourcesIDsFastObject(enc, r.residentRes, r.residentResIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
	if fineGrainedReplay && len(r.barrierOps) > 0 {
		// Replay barrier-free ICB ranges separated by a RESOURCE-SCOPED encoder memory barrier at each
		// true dependency — buffer-coherency sync instead of the coarse all-prior SetBarrier full drain,
		// so the tiny decode kernels can pipeline. The ICB must have been recorded barrier-free.
		start := r.rng.Location
		for _, b := range r.barrierOps {
			bb := uint(b)
			executeCommandsInBufferWithRangeObjectFast(enc, r.icb, foundation.NSRange{Location: start, Length: bb - start})
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			start = bb
		}
		executeCommandsInBufferWithRangeObjectFast(enc, r.icb, foundation.NSRange{Location: start, Length: r.rng.Location + r.rng.Length - start})
		return r.lastOut
	}
	executeCommandsInBufferWithRangeObjectFast(enc, r.icb, r.rng)
	return r.lastOut
}

func (r *archICBReplay) encodeStepBodyNoInputIntoBuffer(enc metal.MTLComputeCommandEncoderObject, pos int, out metal.MTLBuffer) (metal.MTLBuffer, bool) {
	if r == nil || r.scratch == nil || !r.hasFinalOut || r.icb == nil || out == nil {
		return nil, false
	}
	if !r.bindStepOutput(out) {
		return nil, false
	}
	r.prepareStepRebind(pos)
	residentRes, residentIDs := r.scratch.outputResidentResource(r.residentRes, r.residentResIDs, out)
	useResourcesIDsFastObject(enc, residentRes, residentIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
	if fineGrainedReplay && len(r.barrierOps) > 0 {
		start := r.rng.Location
		for _, b := range r.barrierOps {
			bb := uint(b)
			executeCommandsInBufferWithRangeObjectFast(enc, r.icb, foundation.NSRange{Location: start, Length: bb - start})
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			start = bb
		}
		executeCommandsInBufferWithRangeObjectFast(enc, r.icb, foundation.NSRange{Location: start, Length: r.rng.Location + r.rng.Length - start})
		return out, true
	}
	executeCommandsInBufferWithRangeObjectFast(enc, r.icb, r.rng)
	return out, true
}

// runBatchInto replays the recorded ICB across a whole T-token sequence — the batch
// encode-bypass, one autorelease pool for the run. PLE tensors are computed per
// token from the recorded runtime's batch token ids.
func (r *archICBReplay) runBatchInto(outputs [][]byte, inputs [][]byte, useCallerOut bool) ([][]byte, error) {
	if r.hasPLE && len(r.pleRuntime.tokenIDs) != len(inputs) {
		return nil, core.NewError("native.archICBReplay.runBatch: PLE token id count must equal inputs")
	}
	outLen := r.dModel * bf16Size
	if cap(outputs) < len(inputs) {
		outputs = make([][]byte, len(inputs))
	} else {
		outputs = outputs[:len(inputs)]
	}
	for t := range outputs {
		if useCallerOut && cap(outputs[t]) >= outLen {
			outputs[t] = outputs[t][:outLen]
			continue
		}
		outputs[t] = make([]byte, outLen)
	}
	var directOutputViews []metal.MTLBuffer
	directOutput := false
	residentRes, residentIDs := r.residentRes, r.residentResIDs
	if useCallerOut {
		if views, resources, ids, ok := r.directOutputResources(outputs, outLen); ok {
			directOutputViews = views
			directOutput = true
			residentRes, residentIDs = resources, ids
		}
	} else if r.scratch != nil {
		r.scratch.closeOutputViews()
	}
	var coreErr error
	withAutoreleasePool(func() {
		var directOutputCmd metal.MTLIndirectComputeCommand
		if directOutput {
			directOutputCmd = indirectComputeCommandAtIndexFast(r.icb, uint(r.finalOutIdx))
		}
		for t := range inputs {
			var pli []byte
			if r.hasPLE {
				p, err := r.pleRuntime.compute(r.pleRuntime.tokenIDs[t], inputs[t])
				if err != nil {
					coreErr = err
					return
				}
				if len(p) != r.nLayers*r.plePliDim*bf16Size {
					coreErr = core.NewError("native.archICBReplay.runBatch: PLE tensor size mismatch")
					return
				}
				pli = p
			}
			if directOutput {
				outputs[t] = r.stepBodyDirectOutput(inputs[t], t, pli, outputs[t], directOutputCmd, directOutputViews[t], residentRes, residentIDs)
				continue
			}
			outputs[t] = r.stepBodyInto(inputs[t], t, pli, outputs[t])
		}
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return outputs, nil
}

func (r *archICBReplay) runBatchLastInto(out []byte, inputs [][]byte) ([]byte, error) {
	if len(inputs) == 0 {
		return nil, core.NewError("native.archICBReplay.runBatchLastInto: empty batch")
	}
	if r.hasPLE && len(r.pleRuntime.tokenIDs) != len(inputs) {
		return nil, core.NewError("native.archICBReplay.runBatchLastInto: PLE token id count must equal inputs")
	}
	outLen := r.dModel * bf16Size
	if len(out) != outLen {
		return nil, core.NewError("native.archICBReplay.runBatchLastInto: output must be hidden bf16 bytes")
	}
	var directOutputView metal.MTLBuffer
	directOutput := false
	residentRes, residentIDs := r.residentRes, r.residentResIDs
	if views, resources, ids, ok := r.directOutputResources([][]byte{out}, outLen); ok {
		directOutputView = views[0]
		directOutput = true
		residentRes, residentIDs = resources, ids
	} else if r.scratch != nil {
		r.scratch.closeOutputViews()
	}
	var coreErr error
	withAutoreleasePool(func() {
		var directOutputCmd metal.MTLIndirectComputeCommand
		if directOutput {
			directOutputCmd = indirectComputeCommandAtIndexFast(r.icb, uint(r.finalOutIdx))
		}
		last := len(inputs) - 1
		for t := range inputs {
			var pli []byte
			if r.hasPLE {
				p, err := r.pleRuntime.compute(r.pleRuntime.tokenIDs[t], inputs[t])
				if err != nil {
					coreErr = err
					return
				}
				if len(p) != r.nLayers*r.plePliDim*bf16Size {
					coreErr = core.NewError("native.archICBReplay.runBatchLastInto: PLE tensor size mismatch")
					return
				}
				pli = p
			}
			if t == last {
				if directOutput {
					r.stepBodyDirectOutput(inputs[t], t, pli, out, directOutputCmd, directOutputView, residentRes, residentIDs)
					continue
				}
				r.stepBodyInto(inputs[t], t, pli, out)
				continue
			}
			r.stepBodyNoResult(inputs[t], t, pli)
		}
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return out, nil
}

// runBatchPipelinedInto replays the sequence DOUBLE-BUFFERED across r and r2 — two ICBs recorded over the
// SAME KV caches. Token t's host prep+submit on rs[t%2] overlaps token t-1's GPU compute on rs[(t-1)%2],
// reclaiming the per-token WaitUntilCompleted/submit/read idle (~40% of the wall — the GPU sits stalled
// between tokens in the serial runBatch). The shared-cache hazard serialises the GPU side correctly
// (token t's attention waits t-1's KV write), so it's byte-identical to runBatchInto. r2 must be recorded
// against the same caches/runtime as r. ~1.6× on e2b prefill.
func (r *archICBReplay) runBatchPipelinedInto(r2 *archICBReplay, outputs [][]byte, inputs [][]byte, useCallerOut bool) ([][]byte, error) {
	if r.hasPLE && len(r.pleRuntime.tokenIDs) != len(inputs) {
		return nil, core.NewError("native.archICBReplay.runBatchPipelined: PLE token id count must equal inputs")
	}
	rs := [2]*archICBReplay{r, r2}
	outLen := r.dModel * bf16Size
	if cap(outputs) < len(inputs) {
		outputs = make([][]byte, len(inputs))
	} else {
		outputs = outputs[:len(inputs)]
	}
	for t := range outputs {
		if useCallerOut && cap(outputs[t]) >= outLen {
			outputs[t] = outputs[t][:outLen]
			continue
		}
		outputs[t] = make([]byte, outLen)
	}
	readOut := func(rr *archICBReplay, out []byte) []byte {
		rr.copyLastOutInto(out)
		return out
	}
	var directOutputViews [2][]metal.MTLBuffer
	var directResidentRes [2][]metal.MTLResource
	var directResidentIDs [2][]objc.ID
	directOutput := false
	if useCallerOut {
		if views0, resources0, ids0, ok0 := r.directOutputResources(outputs, outLen); ok0 {
			if views1, resources1, ids1, ok1 := r2.directOutputResources(outputs, outLen); ok1 {
				directOutput = true
				directOutputViews = [2][]metal.MTLBuffer{views0, views1}
				directResidentRes = [2][]metal.MTLResource{resources0, resources1}
				directResidentIDs = [2][]objc.ID{ids0, ids1}
			}
		}
	} else {
		if r.scratch != nil {
			r.scratch.closeOutputViews()
		}
		if r2.scratch != nil {
			r2.scratch.closeOutputViews()
		}
	}
	var coreErr error
	withAutoreleasePool(func() {
		var directOutputCmds [2]metal.MTLIndirectComputeCommand
		if directOutput {
			directOutputCmds[0] = indirectComputeCommandAtIndexFast(r.icb, uint(r.finalOutIdx))
			directOutputCmds[1] = indirectComputeCommandAtIndexFast(r2.icb, uint(r2.finalOutIdx))
		}
		var prev *archICBReplay
		var prevCB metal.MTLCommandBufferObject
		var prevT int
		prevReady := false
		for t := range inputs {
			rr := rs[t%2]
			var pli []byte
			if rr.hasPLE {
				p, err := rr.pleRuntime.compute(rr.pleRuntime.tokenIDs[t], inputs[t])
				if err != nil {
					coreErr = err
					return
				}
				if len(p) != rr.nLayers*rr.plePliDim*bf16Size {
					coreErr = core.NewError("native.archICBReplay.runBatchPipelined: PLE tensor size mismatch")
					return
				}
				pli = p
			}
			slot := t % 2
			if directOutput {
				rr.bindStepOutputCommand(directOutputCmds[slot], directOutputViews[slot][t])
			}
			rr.prepareStep(inputs[t], t, pli)
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			if directOutput {
				useResourcesIDsFastObject(enc, directResidentRes[slot], directResidentIDs[slot], metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			} else {
				useResourcesIDsFastObject(enc, rr.residentRes, rr.residentResIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			}
			executeCommandsInBufferWithRangeObjectFast(enc, rr.icb, rr.rng)
			endEncodingFast(enc)
			commitCommandBufferFast(cb) // submit t WITHOUT waiting — overlaps t-1's GPU compute with this host turn
			if prevReady {
				waitUntilCompletedFast(prevCB)
				if !directOutput {
					outputs[prevT] = readOut(prev, outputs[prevT])
				}
			}
			prev, prevCB, prevT, prevReady = rr, cb, t, true
		}
		if prevReady {
			waitUntilCompletedFast(prevCB)
			if !directOutput {
				outputs[prevT] = readOut(prev, outputs[prevT])
			}
		}
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return outputs, nil
}

// icbRope bundles the per-layer rope geometry the ICB records: the global theta `base` + the
// sliding theta `localBase`, the partial-rotary dims (`rotaryDim` global, `rotaryDimLocal` sliding),
// the `globalHeadDim` proportional-global layers rope over, and the explicit-periods buffers
// (`globalFreqs` proportional-global, `freqs` YaRN; nil ⇒ base-derived). A uniform model sets
// localBase==base, rotary==headDim, nil freqs ⇒ every layer ropes on `base` (the old single-base
// behaviour, byte-identical).
type icbRope struct {
	base, localBase                          float32
	rotaryDim, rotaryDimLocal, globalHeadDim int
	globalFreqs, freqs                       metal.MTLBuffer
}

// simpleICBRope is the uniform rope (every layer on `base`, full rotary, no freqs) — the
// byte-identical default for callers that carry no per-layer rope (the bf16/quant batch entries).
func simpleICBRope(base float32, headDim int) icbRope {
	return icbRope{base: base, localBase: base, rotaryDim: headDim, rotaryDimLocal: headDim, globalHeadDim: headDim}
}

// decodeForwardArchICBCore is the ARCH-AWARE cache-grow ICB recorder + replay: like
// decodeForwardICBCore it records the decode stack ONCE and replays per token over a
// growing seq-major KV cache with cheap per-token offset rebinds, but it is DRIVEN by
// the declared arch (specs) — honouring the KV-cache topology (sharer layers attend an
// earlier owner's cache instead of their own) and per-layer sliding-window attention
// (the SDPA reads only the last W rows). MoE is NOT supported here (the router's host
// top-k can't live inside a single recorded/replayed command buffer).
//
// Layout: a uniform 24 ops/layer (base = 24·li) keeps indexing simple. A SHARER layer
// still records its K/V projections (ops 3-5) but to THROWAWAY scratch — its SDPA (op
// 6) reads the OWNER's cache. (Truly eliding the sharer's K/V matmuls would need a
// variable op layout; that's a perf micro-opt, not correctness — the output is identical.)
//
// Per-token rebind: offBuf (rope position), the two window-length buffers (nGlobalBuf =
// t+1, nSlidingBuf = min(t+1,W)), each OWNER layer's two cache-WRITE offsets (advancing
// row t), and each SLIDING layer's SDPA K/V READ offset (the window start). recordProj
// records the seven projections (gemv or qmv) exactly as the non-arch core; vOutBind is
// the projection output's bind index (gemv 3 / qmv 4).
//
// perLayerDFF carries each layer's FFN width (gemma4 E2B/E4B MatFormer varies it per
// layer): the FFN scratch + GeLU-constant buffers are sized to the WIDEST layer and the
// per-layer FFN dispatch widths / element-count buffers read only that layer's lff. A nil
// or short entry (or 0) falls back to the uniform dFF, so the existing uniform callers are
// byte-identical. The recordProj seam keys the gate/up/down PSOs per layer (it already
// receives li), so it must select the matching (outDim,inDim) shape for that layer's lff.
// (Per-layer headDim — gemma4 global layers' larger head_dim — is a later step: it would
// also make kvDim/rowBytes/SDPA-PSO per-layer; this core keeps headDim uniform.)
func recordArchICB(
	specs []model.LayerSpec,
	anwBufs, mnwBufs, kCaches, vCaches, projResident []metal.MTLBuffer,
	qNormBufs, kNormBufs, postAttnBufs, postFFBufs []metal.MTLBuffer,
	layerScalarBufs []metal.MTLBuffer, ple *archICBPLEPlan,
	recordProj func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer, outOff uint, p projIndex),
	recordFusedRMSProj func(li int, c metal.MTLIndirectComputeCommand, rawIn, normW, epsB, out metal.MTLBuffer, outOff uint, p projIndex),
	vOutBind uint, valueNormOnes metal.MTLBuffer,
	dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int,
	perLayerDFF []int,
	rope icbRope, scale, eps float32,
	kvQ8 *archICBKVQ8,
) (*archICBReplay, error) {
	nLayers := len(anwBufs)
	// The value projection reads the DECLARED per-layer op selection (model.LayerSpec.AttentionKEqV):
	// a K==V layer (or a KV-shared layer) has no own v_proj, so V rides the k-proj output. The arch
	// no longer hands the engine a v-proj-index hook — the family declares this on the spec.
	vProjOf := func(li int) projIndex {
		if specs[li].AttentionKEqV {
			return projK
		}
		return projV
	}
	// per-layer head dim AND kv heads (gemma4 full_attention layers attend with a LARGER head_dim than
	// sliding, and the 12B/31B global layers use MQA — kvHeads=1 — vs GQA on the sliding layers): hdOf(li)
	// / kvOf(li) are the layer's geometry; maxHd·maxKv size the shared attention scratch; each layer binds a
	// per-hd SDPA PSO + a per-(hd,kv) stride/axis set + a per-kv GQA-ratio buffer. Uniform models
	// (maxHd==headDim, maxKv==nKVHeads) are byte-identical to the pre-per-layer recorder.
	hdOf := func(li int) int { return headDimOf(specs[li], headDim) }
	kvOf := func(li int) int { return kvHeadsOf(specs[li], nKVHeads) }
	kvdOf := func(li int) int { return kvOf(li) * hdOf(li) }
	maxHd, maxKv := headDim, nKVHeads
	for li := range nLayers {
		if h := hdOf(li); h > maxHd {
			maxHd = h
		}
		if k := kvOf(li); k > maxKv {
			maxKv = k
		}
	}
	maxQd, maxKvd := nHeads*maxHd, maxKv*maxHd
	// per-layer FFN width: lffOf(li) is this layer's FFN dim (gemma4 MatFormer); maxDFF
	// sizes the shared FFN scratch + GeLU constants to the widest layer. Falls back to the
	// uniform dFF when perLayerDFF is absent/0 ⇒ uniform callers are byte-identical.
	lffOf := func(li int) int {
		if li < len(perLayerDFF) && perLayerDFF[li] > 0 {
			return perLayerDFF[li]
		}
		return dFF
	}
	maxDFF := dFF
	for li := range nLayers {
		if l := lffOf(li); l > maxDFF {
			maxDFF = l
		}
	}
	hasPLE := ple.enabled()
	if hasPLE {
		if len(ple.postNormBufs) != nLayers {
			return nil, core.NewError("native.recordArchICB: PLE post norm count must equal layers")
		}
	}
	hasLayerScalar := false
	for _, b := range layerScalarBufs {
		if b != nil {
			hasLayerScalar = true
			break
		}
	}
	maxGelu := maxDFF
	if hasPLE && ple.pliDim > maxGelu {
		maxGelu = ple.pliDim
	}

	// looped-aware: dModel past rmsLoopedLimit takes the grid-striding rms kernel — the
	// single-row kernel's one pass cannot cover it (#348, gemma4 31B hidden 5376). The
	// per-HEAD rms rows (axis = headDim ≤ 512) stay on the single-row kernel below.
	rmsPSO, err := pipelineForICB(rmsKernelBF16(dModel))
	if err != nil {
		return nil, err
	}
	rmsHeadPSO, err := pipelineForICB("rmsbfloat16")
	if err != nil {
		return nil, err
	}
	ropePSO, err := ropePipelineICB(false)
	if err != nil {
		return nil, err
	}
	var ropeFreqsPSO metal.MTLComputePipelineState
	if rope.globalFreqs != nil || rope.freqs != nil {
		if ropeFreqsPSO, err = ropeFreqsPipelineICB(false); err != nil {
			return nil, err
		}
	}
	// per-hd SDPA PSO (gemma4 global 512 vs sliding 256 head dim) — one per distinct hd, picked per layer.
	sdpaPSOByHd := make(map[int]metal.MTLComputePipelineState)
	for li := range nLayers {
		hd := hdOf(li)
		if _, ok := sdpaPSOByHd[hd]; !ok {
			pso, e := sdpaVectorPipelineICBForHeadDim(hd)
			if e != nil {
				return nil, e
			}
			sdpaPSOByHd[hd] = pso
		}
	}
	// Deep-decode: GLOBAL layers record the 2-pass SDPA pair instead of the single-pass
	// kernel. The single-pass sdpa_vector runs ONE threadgroup per head over the whole
	// cache — recorded once, it can never re-parallelise as the KV grows, which collapsed
	// deep decode far below bandwidth physics (E2B @52K: 44 tok/s measured vs ~115
	// expected). blocks is fixed at record time from maxLen and is safe at ANY smaller n:
	// a block whose strided key walk starts past N writes finite_min/0 partials that the
	// pass-2 merge zeroes. Sliding layers stay single-pass — their window bounds n below
	// the 2-pass knee. Sessions too short to ever cross the knee keep the pure
	// single-pass layout (and the existing byte-parity fixtures with it).
	sdpa2PassICBBlocks := 0
	if maxLen >= sdpa2PassMinKV {
		// blocks bakes into the recorded pipelines, so fan for the most-starved
		// GLOBAL layer (fewest KV heads): the occupancy floor in sdpa2PassBlocks
		// keys on the layer's KV heads (#365 — E2B's single KV head needs 4x the
		// ladder's fan to feed the GPU). Wider layers pay only their pass-2 merge.
		minKV := 0
		for li := range nLayers {
			if specs[li].Attention != model.GlobalAttention {
				continue
			}
			if kv := kvOf(li); minKV == 0 || kv < minKV {
				minKV = kv
			}
		}
		sdpa2PassICBBlocks = int(sdpa2PassBlocks(maxLen, minKV))
	}
	sdpa2Pass1PSOByHd := make(map[int]metal.MTLComputePipelineState)
	sdpa2Pass2PSOByHd := make(map[int]metal.MTLComputePipelineState)
	nGlobal2Pass := 0
	if sdpa2PassICBBlocks > 0 {
		for li := range nLayers {
			if specs[li].Attention != model.GlobalAttention {
				continue
			}
			nGlobal2Pass++
			hd := hdOf(li)
			if _, ok := sdpa2Pass1PSOByHd[hd]; !ok {
				p1, e := sdpaVector2Pass1PipelineICB(hd, int32(sdpa2PassICBBlocks))
				if e != nil {
					return nil, e
				}
				p2, e2 := sdpaVector2Pass2PipelineICB(hd)
				if e2 != nil {
					return nil, e2
				}
				sdpa2Pass1PSOByHd[hd] = p1
				sdpa2Pass2PSOByHd[hd] = p2
			}
		}
	}
	// q8 KV global layers (#367): resolve the quantise-store + q8 SDPA read
	// pipelines up front — a q8 cache with a missing q8 kernel is a hard error
	// at record time, never a silent bf16 misread of int8 bytes.
	q8On := func(li int) bool { return kvQ8.on(li) }
	nQ8Store := 0
	var kvQ8StoreICB metal.MTLComputePipelineState
	sdpaQ8PSOByHd := make(map[int]metal.MTLComputePipelineState)
	sdpaQ8P1PSOByHd := make(map[int]metal.MTLComputePipelineState)
	if kvQ8.any() {
		var qerr error
		if kvQ8StoreICB, qerr = kvQ8StorePipelineICB(); qerr != nil {
			return nil, qerr
		}
		for li := range nLayers {
			if !q8On(li) {
				continue
			}
			if !specs[li].OwnsCache() {
				return nil, core.NewError("native.recordArchICB: q8 KV on a non-owner layer")
			}
			nQ8Store += 2
			hd := hdOf(li)
			if _, ok := sdpaQ8PSOByHd[hd]; !ok {
				pso, e := sdpaVectorQ8PipelineICB(hd)
				if e != nil {
					return nil, e
				}
				sdpaQ8PSOByHd[hd] = pso
				if sdpa2PassICBBlocks > 0 {
					p1, e1 := sdpaVector2Pass1Q8PipelineICB(hd, int32(sdpa2PassICBBlocks))
					if e1 != nil {
						return nil, e1
					}
					sdpaQ8P1PSOByHd[hd] = p1
				}
			}
		}
	}
	addPSO, err := pipelineForICB("vv_Addbfloat16")
	if err != nil {
		return nil, err
	}
	hasFusedGELU := gpuHasGeluKernel()
	var mulPSO, tanhPSO metal.MTLComputePipelineState
	var geluICBPSO metal.MTLComputePipelineState
	if hasFusedGELU {
		if geluICBPSO, err = geluPipelineICB(); err != nil {
			return nil, err
		}
	} else {
		mulPSO, err = pipelineForICB("vv_Multiplybfloat16")
		if err != nil {
			return nil, err
		}
		tanhPSO, err = pipelineForICB("v_Tanhbfloat16bfloat16")
		if err != nil {
			return nil, err
		}
	}
	if hasFusedGELU && hasLayerScalar {
		mulPSO, err = pipelineForICB("vv_Multiplybfloat16")
		if err != nil {
			return nil, err
		}
	}
	// Fused residual-RMSNorm: gemma4's post-attn / post-FF norm-then-add (out = res + rms(branch)) collapses
	// from two barriered ICB ops (rms in-place + vv_Add) to ONE — removing 2 full-drain barriers/layer (the
	// no-barrier ceiling probe showed each coarse SetBarrier drain costs ~7.5µs at decode batch=1).
	var rmsResPSO metal.MTLComputePipelineState
	useFusedResRMS := hasFusedGELU
	if useFusedResRMS {
		if rmsResPSO, err = rmsResidualPipelineICB(dModel); err != nil {
			return nil, err
		}
	}
	// Fused per-head QK-norm + RoPE: qNorm+ropeQ (and kNorm+ropeK) collapse from two barriered ICB ops
	// to one — the high-value element-wise fusion (the probe: per-head norms ~+7.5, rope ~+5.5 tok/s).
	// Soft (fall back to the composed pair on miss). Lockstep with the re-encode encQKNormRope (same
	// kernel) so ICB ≡ re-encode stays byte-equal; ~1 ULP from the old composed path.
	var qkRopeICBPSO metal.MTLComputePipelineState
	useFusedQKRope := false
	if hasFusedGELU { // same custom library as gelu — if that built, this builds (hard, like gelu)
		if qkRopeICBPSO, err = qkNormRopePipelineICB(); err != nil {
			return nil, err
		}
		useFusedQKRope = true
	}

	var r *archICBReplay
	var coreErr error
	withAutoreleasePool(func() {
		pleInputElems, pleDim := 0, 0
		if hasPLE {
			pleInputElems, pleDim = nLayers*ple.pliDim, ple.pliDim
		}
		sc := getArchICBReplayScratch(dModel, maxQd, maxKvd, maxDFF, maxGelu, nLayers, pleInputElems, pleDim, hasFusedGELU, hasPLE)

		normed := sc.normed
		q, qr, kProj, attn := sc.q, sc.qr, sc.kProj, sc.attn
		attnOut := sc.attnOut
		kThrow, vThrow := sc.kThrow, sc.vThrow // sharer's discarded K/V
		mlpNormed := sc.mlpNormed
		// FFN scratch + GeLU constants sized to the WIDEST layer (gemma4 MatFormer varies dFF
		// per layer); each layer dispatches only its own lff elements, so a narrower layer reads
		// a prefix of these buffers. Uniform callers (maxDFF==dFF) are byte-identical.
		gate, up := sc.gate, sc.up
		gated, down := sc.gated, sc.down
		var x2, x3, x3s, inner metal.MTLBuffer
		var scaled, tnh, onePlus, halfG metal.MTLBuffer
		var gelu metal.MTLBuffer
		var c044, c079, c1c, c05 metal.MTLBuffer
		if !hasFusedGELU {
			x2, x3, x3s, inner = sc.x2, sc.x3, sc.x3s, sc.inner
			scaled, tnh, onePlus, halfG = sc.scaled, sc.tnh, sc.onePlus, sc.halfG
			gelu = sc.gelu
			c044, c079, c1c, c05 = sc.c044, sc.c079, sc.c1c, sc.c05
		}
		var pleInput, pleGate, pleGated, pleProj, pleNorm metal.MTLBuffer
		if hasPLE {
			pleInput, pleGate, pleGated = sc.pleInput, sc.pleGate, sc.pleGated
			pleProj, pleNorm = sc.pleProj, sc.pleNorm
		}
		ping := sc.ping
		hBufs := sc.hBufs

		offBuf, nGlobalBuf, nSlidingBuf := sc.offBuf, sc.nGlobalBuf, sc.nSlidingBuf
		// scalarI32/F32 memoise by value, so a sink-driven op (emitRMSNorm via icbSink) binds the SAME
		// eps/axis/ws buffers these named handles hold — no duplicate scalar buffers, no per-record alloc.
		epsBuf, axisBuf, wsBuf := scalarF32(eps), scalarI32(int32(dModel)), scalarI32(1)
		ropeScaleB := scalarF32(scale)
		ropeBaseB := scalarF32(float32(math.Log2(float64(rope.base))))
		ropeLocalBaseB := scalarF32(float32(math.Log2(float64(rope.localBase))))
		freqStride1B := scalarI64(1)
		// per-kv GQA ratio buffer (nHeads/kvHeads): one per distinct kvHeads (gemma4 12B/31B mix MQA
		// global layers kv=1 with GQA sliding layers kv=8), shared across layers of that kv, resident below.
		gqaBy := make(map[int]metal.MTLBuffer)
		gqaOf := func(kv int) metal.MTLBuffer {
			b, ok := gqaBy[kv]
			if !ok {
				b = scalarI32(int32(nHeads / kv))
				gqaBy[kv] = b
			}
			return b
		}
		// per-hd axis scalars (QK-norm axis = hd, rope head-stride = hd): hd-only, one per distinct head dim.
		type hdAxis struct{ axisHead, ropeMat metal.MTLBuffer }
		hdAxisBy := make(map[int]hdAxis)
		hdAxisOf := func(hd int) hdAxis {
			a, ok := hdAxisBy[hd]
			if !ok {
				a = hdAxis{axisHead: scalarI32(int32(hd)), ropeMat: scalarI64(int64(hd))} // memoised, so emitRMSNormRows binds this same buffer
				hdAxisBy[hd] = a
			}
			return a
		}
		// per-(hd,kv) SDPA strides: head stride = hd, seq stride = kvHeads·hd — the seq stride varies with kv
		// (12B/31B global layers are MQA, kv=1). One set per distinct (hd,kv), all made resident below.
		type sdpaStrides struct{ khs, kss, vhs, vss metal.MTLBuffer }
		sdpaStrideBy := make(map[[2]int]sdpaStrides)
		sdpaStrideOf := func(hd, kv int) sdpaStrides {
			key := [2]int{hd, kv}
			s, ok := sdpaStrideBy[key]
			if !ok {
				kvd := kv * hd
				s = sdpaStrides{khs: scalarI64(int64(hd)), kss: scalarI64(int64(kvd)), vhs: scalarI64(int64(hd)), vss: scalarI64(int64(kvd))}
				sdpaStrideBy[key] = s
			}
			return s
		}
		for li := range nLayers {
			hdAxisOf(hdOf(li))
			sdpaStrideOf(hdOf(li), kvOf(li))
			gqaOf(kvOf(li))
		}
		sdpaScaleB := scalarF32(scale)
		addModelB := scalarI32(int32(dModel)) // memoised, so a sink-driven binary op binds this same resident buffer
		var pleCntB metal.MTLBuffer
		if hasPLE {
			pleCntB = scalarI32(int32(ple.pliDim)) // memoised, so the sink-driven PLE gelu binds this same resident buffer
		}
		// per-distinct-dFF element-count buffers (the FFN binary/gelu/tanh ops take the count
		// as a buffer): one scalar per distinct width, shared across layers of that width. Every
		// one is appended to resident below so the ICB replay's UseResources covers it — a
		// non-resident count buffer is read as garbage on the layer that uses it.
		ffCntBufs := make(map[int]metal.MTLBuffer)
		ffCntOf := func(n int) metal.MTLBuffer {
			b, ok := ffCntBufs[n]
			if !ok {
				b = scalarI32(int32(n)) // memoised; still tracked here for residency
				ffCntBufs[n] = b
			}
			return b
		}
		for li := range nLayers {
			ffCntOf(lffOf(li))
		}
		// fused QK-norm+rope per-layer params: ropeParamsOf mirrors setRope's per-layer base/rotDim/freqs
		// pick; a rotary-dim scalar per distinct rotaryDim + the use-freqs flags + a dummy periods buffer,
		// all made resident below (a non-resident param buffer reads garbage on the layer that uses it).
		// per-layer rope params, matching the host stepToken pick: sliding → localBase/rotaryDimLocal;
		// proportional-global → globalFreqs/globalHeadDim; else base/rotaryDim. Returns log2(base) as a
		// VALUE — the sink derives the (memoised) buffer — so setRope/setQKNormRope share one selection.
		ropeParamsOf := func(li int) (log2base float64, freqs metal.MTLBuffer, rotDim int) {
			hd := hdOf(li)
			log2base, rotDim, freqs = math.Log2(float64(rope.base)), rope.rotaryDim, rope.freqs
			if specs[li].Attention == model.SlidingAttention {
				log2base, rotDim, freqs = math.Log2(float64(rope.localBase)), rope.rotaryDimLocal, rope.freqs
			} else if rope.globalFreqs != nil {
				rotDim, freqs = rope.globalHeadDim, rope.globalFreqs
			}
			if rotDim <= 0 || rotDim > hd {
				rotDim = hd
			}
			return
		}
		rotDimBufs := make(map[int]metal.MTLBuffer)
		rotDimBufOf := func(rd int) metal.MTLBuffer {
			b, ok := rotDimBufs[rd]
			if !ok {
				b = scalarI32(int32(rd))
				rotDimBufs[rd] = b
			}
			return b
		}
		useFreqs0B, useFreqs1B := scalarI32(0), scalarI32(1)
		qkDummyPeriodsB := qkRopeDummyBuf()
		if useFusedQKRope {
			for li := range nLayers {
				_, _, rd := ropeParamsOf(li)
				rotDimBufOf(rd)
			}
		}

		resident := []metal.MTLBuffer{
			ping[0], ping[1], normed, q, qr, kProj, attn, attnOut, kThrow, vThrow, mlpNormed,
			gate, up, gated, down,
			offBuf, nGlobalBuf, nSlidingBuf, epsBuf, axisBuf, wsBuf,
			ropeScaleB, ropeBaseB, ropeLocalBaseB, freqStride1B, sdpaScaleB, addModelB,
		}
		// 2-pass SDPA intermediates for the GLOBAL layers (shared across layers — the replay's
		// dependency barriers already serialise each layer's attention on the shared scratch,
		// exactly as the single-row attn scratch). f32 per the kernel ABI; sized at the widest
		// head dim. Owned by the replay via residentRes for the session's lifetime.
		var p2Partials, p2Sums, p2Maxs metal.MTLBuffer
		if nGlobal2Pass > 0 {
			p2Partials = device.NewBufferWithLengthOptions(uint(nHeads*sdpa2PassICBBlocks*maxHd*4), metal.MTLResourceStorageModeShared)
			p2Sums = device.NewBufferWithLengthOptions(uint(nHeads*sdpa2PassICBBlocks*4), metal.MTLResourceStorageModeShared)
			p2Maxs = device.NewBufferWithLengthOptions(uint(nHeads*sdpa2PassICBBlocks*4), metal.MTLResourceStorageModeShared)
			// pass-2 binds blocks via the memoised scalar — a value no other op declares, so
			// register it resident explicitly (an ICB op reading a non-resident buffer is
			// undefined; the strides/scale/N binds all reuse scalars already listed above).
			resident = append(resident, p2Partials, p2Sums, p2Maxs, scalarI32(int32(sdpa2PassICBBlocks)))
		}
		// q8 staging rows (the K rope/norm and V projection land here, FIXED
		// binds; the store ops quantise staging into the int8 cache + scale
		// rows). Shared across q8 layers — the recorded barriers serialise the
		// layer stack on them, exactly as the attention scratch. Owned by the
		// replay via residentRes for the session's lifetime.
		var kStageQ8, vStageQ8 metal.MTLBuffer
		if kvQ8.any() {
			kStageQ8 = device.NewBufferWithLengthOptions(uint(maxKvd*bf16Size), metal.MTLResourceStorageModeShared)
			vStageQ8 = device.NewBufferWithLengthOptions(uint(maxKvd*bf16Size), metal.MTLResourceStorageModeShared)
			resident = append(resident, kStageQ8, vStageQ8)
			for li := range nLayers {
				if !q8On(li) {
					continue
				}
				resident = append(resident, kvQ8.kScales[li], kvQ8.vScales[li])
				// the store op binds kvDim via the memoised i32 scalar; register it
				resident = append(resident, scalarI32(int32(kvdOf(li))))
			}
		}
		if !hasFusedGELU {
			resident = append(resident, x2, x3, x3s, inner, scaled, tnh, onePlus, halfG, gelu, c044, c079, c1c, c05)
		}
		for _, a := range hdAxisBy {
			resident = append(resident, a.axisHead, a.ropeMat)
		}
		for _, s := range sdpaStrideBy {
			resident = append(resident, s.khs, s.kss, s.vhs, s.vss)
		}
		for _, b := range gqaBy {
			resident = append(resident, b)
		}
		if rope.globalFreqs != nil {
			resident = append(resident, rope.globalFreqs)
		}
		if rope.freqs != nil {
			resident = append(resident, rope.freqs)
		}
		resident = append(resident, useFreqs0B, useFreqs1B, qkDummyPeriodsB)
		for _, b := range rotDimBufs {
			resident = append(resident, b)
		}
		var layerScalarOnes metal.MTLBuffer
		if hasPLE {
			resident = append(resident, pleInput, pleGate, pleGated, pleProj, pleNorm, pleCntB)
			resident = append(resident, ple.resident...)
			for _, b := range ple.postNormBufs {
				resident = append(resident, b)
			}
		}
		if hasLayerScalar {
			layerScalarOnes = bf16ConstBuffer(dModel, 1.0)
			resident = append(resident, layerScalarOnes)
			for _, b := range layerScalarBufs {
				if b != nil {
					resident = append(resident, b)
				}
			}
		}
		for _, b := range ffCntBufs { // the per-distinct-dFF FFN count buffers must be resident for the replay
			resident = append(resident, b)
		}
		// reserve the upper-bound capacity for the appends that follow (projResident + the per-layer
		// weight/norm/cache slices, ≤16 buffers/layer + the 19 projResident scalars) so the resident
		// slice never geometrically regrows its backing array. Grow changes capacity only — the
		// literal contents, the appended buffers, and every kernel binding are unchanged.
		resident = slices.Grow(resident, 16*nLayers+20)
		resident = append(resident, projResident...)
		resident = append(resident, anwBufs...)
		resident = append(resident, mnwBufs...)
		// gemma4 norm buffers (uniform presence across layers); add the non-nil ones.
		for _, bufs := range [][]metal.MTLBuffer{qNormBufs, kNormBufs, postAttnBufs, postFFBufs} {
			for _, b := range bufs {
				if b != nil {
					resident = append(resident, b)
				}
			}
		}
		if valueNormOnes != nil {
			resident = append(resident, valueNormOnes)
		}
		for _, b := range kCaches {
			if b != nil {
				resident = append(resident, b)
			}
		}
		for _, b := range vCaches {
			if b != nil {
				resident = append(resident, b)
			}
		}
		resident = append(resident, hBufs...)

		// Norm-op selection is family-declared (LayerSpec.Attention{Q,K}Norm /
		// Post{Attn,FF}Norm, resolved once in model.Assemble) with a buffer-presence
		// self-heal for hand-built callers that do not declare — the slice-2 K==V
		// pattern. Uniform across layers by contract (layer 0 speaks for the stack);
		// each present norm adds one op per layer, so the layout grows but stays
		// uniform → a single running op counter.
		hasQN := specs[0].AttentionQNorm || (len(qNormBufs) > 0 && qNormBufs[0] != nil)
		hasKN := specs[0].AttentionKNorm || (len(kNormBufs) > 0 && kNormBufs[0] != nil)
		hasPA := specs[0].PostAttnNorm || (len(postAttnBufs) > 0 && postAttnBufs[0] != nil)
		hasPF := specs[0].PostFFNorm || (len(postFFBufs) > 0 && postFFBufs[0] != nil)
		extra := 0
		for _, h := range []bool{hasQN, hasKN, hasPA, hasPF} {
			if h {
				extra++
			}
		}
		if valueNormOnes != nil { // gemma4 value-norm adds one op/layer (owner: the V row; sharer: discarded)
			extra++
		}
		opsPerLayer := 24 + extra
		if hasFusedGELU { // fused gelu is 1 command vs the composed chain's 10
			opsPerLayer -= 9
		}
		// fused QK-norm+rope collapses (qNorm + ropeQ) and (kNorm + ropeK) from 2 ops to 1 each when the
		// layer has QK-norm. The fused K op writes the cache at buffer index 2 (its `out`), not the plain
		// rope's index 1 — so the per-token kRopeIdx rebind (prepareStep) uses kRopeBindIdx.
		kRopeBindIdx := uint(1)
		if useFusedQKRope && hasQN {
			opsPerLayer-- // qNorm+ropeQ
		}
		if useFusedQKRope && hasKN {
			opsPerLayer-- // kNorm+ropeK
			kRopeBindIdx = 2
		}
		if hasPLE {
			if hasFusedGELU {
				opsPerLayer += 5 // qmv gate, fused gelu*pli, qmv proj, rms, residual add
			} else {
				opsPerLayer += 14 // qmv gate, 10-op gelu*pli chain, qmv proj, rms, residual add
			}
		}
		if hasLayerScalar {
			opsPerLayer++
		}
		// fused input-RMSNorm+qmv folds the attn-input rms and the mlp-input rms INTO their following
		// projections (Q/K/V read inBuf+attnNormW; gate/up read hBuf+mlpNormW), removing both setRMS ops.
		if recordFusedRMSProj != nil {
			opsPerLayer -= 2
		}
		// fused residual-RMSNorm folds each post-norm + its residual add into one op (out = res + rms(branch)).
		if useFusedResRMS {
			if hasPA {
				opsPerLayer--
			}
			if hasPF {
				opsPerLayer--
			}
		}
		// GLOBAL layers' 2-pass SDPA is pass-1 + pass-2 where the single-pass was one op;
		// q8 owner layers add two quantise-store ops (K row, V row).
		total := opsPerLayer*nLayers + nGlobal2Pass + nQ8Store
		recorder := icbop.New(device, uint(total), 16)
		icb := recorder.Buffer

		rmsTG := rmsThreadgroup(dModel, rmsPSO)
		// per-head rows: hd is bounded at 512 by the attention plan guard, so the raw
		// single-row threadgroup (≤ 128 lanes) never nears the 1024 cap the full-dModel
		// sites needed rmsThreadgroup for.
		headTGOf := func(hd int) uint {
			return uint(rmsSimdSize * ((((hd + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
		}
		elemGroup := func(n int) uint {
			if uint(n) < 256 {
				return uint(n)
			}
			return 256
		}
		// full-dModel RMSNorm through the SHARED emitRMSNorm body (the same one encRMSNormBF16 drives) via
		// icbSink — the path-unifying dispatchSink, one math recorded into both the encoder and the ICB.
		// icbSink binds eps/axis/ws as the memoised scalar buffers (== epsBuf/axisBuf/wsBuf bound above).
		setRMS := func(c metal.MTLIndirectComputeCommand, in, w, o metal.MTLBuffer) {
			emitRMSNorm(fastICBSink{c}, rmsPSO, in, w, o, 0, dModel, eps, rmsTG)
		}
		// fused post-norm tail out = res + rmsnorm(x, w) in ONE ICB command (lthn_rmsnorm_residual_bf16,
		// one fewer barrier than RMS + vv_Add) through the SHARED emitRMSNormResidual body.
		setRMSResidual := func(c metal.MTLIndirectComputeCommand, x, w, res, o metal.MTLBuffer) {
			emitRMSNormResidual(fastICBSink{c}, rmsResPSO, x, w, res, o, 0, dModel, eps, rmsTG)
		}
		// per-head RMSNorm (gemma4 QK-norm: rows of headDim each) through the SHARED emitRMSNormRows body;
		// axisSize = hd binds the same memoised buffer hdAxisOf(hd).axisHead holds.
		setRMSRows := func(c metal.MTLIndirectComputeCommand, in, w, o metal.MTLBuffer, rows, hd int) {
			emitRMSNormRows(fastICBSink{c}, rmsHeadPSO, in, w, o, 0, 0, 0, hd, eps, rows, headTGOf(hd))
		}
		// element-wise binary op through the SHARED emitBinary body (with encBinaryDT). The count binds the
		// memoised scalar buffer addModelB/ffCntOf hold — no separate count param.
		setBinOffsets := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, a metal.MTLBuffer, aOff uint, b metal.MTLBuffer, bOff uint, o metal.MTLBuffer, oOff uint, n int) {
			emitBinary(fastICBSink{c}, pso, a, aOff, b, bOff, o, oOff, n)
		}
		setBin := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, a, b, o metal.MTLBuffer, n int) {
			setBinOffsets(c, pso, a, 0, b, 0, o, 0, n)
		}
		// per-layer rope through the SHARED emitRope body (with encRoPEBF16To/encRoPEFreqsBF16To), matching
		// the host stepToken pick: sliding → localBase/rotaryDimLocal; proportional-global → the globalFreqs
		// spectrum over globalHeadDim; else base/rotaryDim. log2base/scale/ropeMat bind the same memoised
		// scalar buffers ropeBaseB/ropeScaleB/hdAxisOf(hd).ropeMat hold.
		setRope := func(c metal.MTLIndirectComputeCommand, in, out metal.MTLBuffer, heads, li int) {
			log2base, freqs, rotDim := ropeParamsOf(li)
			pso := ropePSO
			if freqs != nil {
				pso = ropeFreqsPSO
			}
			emitRope(fastICBSink{c}, pso, in, out, 0, 0, offBuf, freqs, heads, rotDim, hdOf(li), scale, float32(log2base))
		}
		// setQKNormRope records the fused per-head QK-norm + RoPE (out = RoPE(RMSNorm(in, w))) in ONE op:
		// per-head rms then rotate, replacing setRMSRows+setRope. One threadgroup per head, hd threads.
		// in/out byte offsets carry the K cache row when fusing K (the projection wrote it there).
		// fused per-head QK-norm + RoPE through the SHARED emitQKNormRope body (with encQKNormRope). eps/
		// headDim/rd/scale/log2base bind the same memoised scalars epsBuf/axisHead/rotDimBufOf/ropeScaleB/
		// ropeBaseB hold; the base form binds qkDummyPeriodsB at 9 (unread, useFreqs=0).
		setQKNormRope := func(c metal.MTLIndirectComputeCommand, in metal.MTLBuffer, inOff uint, w metal.MTLBuffer, out metal.MTLBuffer, outOff uint, heads, li int) {
			log2base, freqs, rd := ropeParamsOf(li)
			emitQKNormRope(fastICBSink{c}, qkRopeICBPSO, in, w, out, inOff, 0, outOff, offBuf, freqs, qkDummyPeriodsB,
				heads, hdOf(li), rd, eps, scale, float32(log2base))
		}
		layerScalarFor := func(li int) metal.MTLBuffer {
			if li < len(layerScalarBufs) && layerScalarBufs[li] != nil {
				return layerScalarBufs[li]
			}
			return layerScalarOnes
		}

		// per-layer commands whose bindings advance per token
		kRopeIdx := sc.kRopeIdx[:nLayers] // owner cache-write (K) op index — re-acquired per token
		vIdx := sc.vIdx[:nLayers]         // owner cache-write (V) op index
		vNormIdx := sc.vNormIdx[:nLayers] // owner value-norm op index (rebound/token)
		sdpaIdx := sc.sdpaIdx[:nLayers]   // SDPA op index (sliding: read offset rebound/token)
		clear(kRopeIdx)
		clear(vIdx)
		clear(vNormIdx)
		clear(sdpaIdx)

		// one running command index across the whole stack (the conditional norm ops make
		// per-layer offsets uneven, but the count is uniform so the running counter stays
		// aligned). The barrier on every command but the first makes execution sequential.
		// q8 store-op indices (the per-token rebind targets for q8 owner layers)
		kStoreIdx := make([]int, nLayers)
		vStoreIdx := make([]int, nLayers)
		opIdx := 0
		finalOutIdx := -1
		finalOutBind := uint(0)
		hasFinalOut := false
		barrierOps := sc.barrierOps[:0] // op indices that carry a barrier-before — used by the fine-grained replay
		emit := func() metal.MTLIndirectComputeCommand {
			c := indirectComputeCommandAtIndexFast(icb, uint(opIdx))
			if opIdx != 0 {
				if fineGrainedReplay {
					// record barrier-free; the replay enforces the dep with an encoder memory barrier
					// (resource-scoped, may pipeline) instead of the coarse all-prior ICB SetBarrier.
					barrierOps = append(barrierOps, opIdx)
				} else if !allBarriersOffForTest { // allBarriersOff: TIMING-ONLY ceiling probe (output races/garbage)
					setICBBarrier(c)
				}
			}
			opIdx++
			return c
		}
		// emitNB records a command WITHOUT a barrier — for an INDEPENDENT SECONDARY consumer of a
		// producer whose FIRST consumer already barriered (and so flushed) it. The op reads the
		// already-visible producer and overlaps its sibling ops instead of draining the pipeline.
		// q/kProj/vProj all read `normed` (q barriers, kProj+vProj ride free); gate/up read
		// `mlpNormed` (gate barriers, up rides free — the big FFN-gemv overlap). Each op that READS
		// one of these (kNorm, kRope, valueNorm, SDPA, gelu) still barriers, so the only relaxed
		// ordering is sibling-vs-sibling, which has no data hazard. Byte-parity-gated.
		emitNB := func() metal.MTLIndirectComputeCommand {
			c := indirectComputeCommandAtIndexFast(icb, uint(opIdx))
			opIdx++
			return c
		}
		// emitFFN is emit() in production but emitNB() under ffnBarriersOffForTest — the FFN-only no-barrier
		// ceiling probe (racy output; measures the GPU-span a fused FFN megakernel could reclaim).
		emitFFN := func() metal.MTLIndirectComputeCommand {
			if ffnBarriersOffForTest {
				return emitNB()
			}
			return emit()
		}
		// recInputProj records an input-rms-fed projection (Q/K/V/gate/up): the FUSED rms+qmv (rms folded
		// in, reads rawIn+normW) when available, else the plain projection over the pre-normed buffer. The
		// caller emits the command (emit/emitNB) so the barrier structure stays visible at the call site,
		// and emits-or-skips the matching setRMS itself.
		recInputProj := func(c metal.MTLIndirectComputeCommand, li int, rawIn, normW, normed, out metal.MTLBuffer, outOff uint, p projIndex) {
			if recordFusedRMSProj != nil {
				recordFusedRMSProj(li, c, rawIn, normW, epsBuf, out, outOff, p)
			} else {
				recordProj(li, c, normed, out, outOff, p)
			}
		}

		for li := range nLayers {
			owns := specs[li].OwnsCache()
			ownerIdx := specs[li].KVShareFrom
			sliding := specs[li].Attention == model.SlidingAttention
			attendK, attendV := kCaches[ownerIdx], vCaches[ownerIdx]
			nBufForLayer := nGlobalBuf
			if sliding {
				nBufForLayer = nSlidingBuf
			}
			inBuf, outBuf := ping[li%2], ping[(li+1)%2]
			hBuf := hBufs[li]

			// --- attention half ---
			if recordFusedRMSProj == nil { // fused path folds this rms into q/kProj/vProj below
				setRMS(emit(), inBuf, anwBufs[li], normed)
			}
			recInputProj(emit(), li, inBuf, anwBufs[li], normed, q, 0, projQ)
			if useFusedQKRope && hasQN { // fused: qr = RoPE(RMSNorm(q, qNormW)) in one op
				setQKNormRope(emit(), q, 0, qNormBufs[li], qr, 0, nHeads, li)
			} else {
				if hasQN { // gemma4 per-head QK-norm on Q before RoPE (in-place)
					setRMSRows(emit(), q, qNormBufs[li], q, nHeads, hdOf(li))
				}
				setRope(emit(), q, qr, nHeads, li)
			}
			recInputProj(emitNB(), li, inBuf, anwBufs[li], normed, kProj, 0, projK) // 2nd consumer (q barriered it) — overlap
			fuseK := useFusedQKRope && hasKN                                        // fuse kNorm+ropeK into one op (writes the cache at buf 2)
			if owns && q8On(li) {
				// q8 GLOBAL owner (#367): K rope/norm and the V projection land in
				// the FIXED bf16 staging rows, then one quantise-store op each
				// writes the int8 cache row + f32 scale row — those two stores'
				// output offsets are what prepareStepRebind rebinds per token.
				if fuseK {
					setQKNormRope(emit(), kProj, 0, kNormBufs[li], kStageQ8, 0, kvOf(li), li)
				} else {
					if hasKN {
						setRMSRows(emit(), kProj, kNormBufs[li], kProj, kvOf(li), hdOf(li))
					}
					setRope(emit(), kProj, kStageQ8, kvOf(li), li)
				}
				cks := emit()
				emitKVQ8Store(fastICBSink{cks}, kvQ8StoreICB, kStageQ8, kCaches[li], 0, kvQ8.kScales[li], 0, kvdOf(li))
				kStoreIdx[li] = opIdx - 1
				cv := emitNB() // 2nd consumer of `normed` (q barriered it) — overlap
				recInputProj(cv, li, inBuf, anwBufs[li], normed, vStageQ8, 0, vProjOf(li))
				if valueNormOnes != nil { // gemma4 value-norm on the staged V row (fixed bind)
					setRMSRows(emit(), vStageQ8, valueNormOnes, vStageQ8, kvOf(li), hdOf(li))
				}
				cvs := emit()
				emitKVQ8Store(fastICBSink{cvs}, kvQ8StoreICB, vStageQ8, vCaches[li], 0, kvQ8.vScales[li], 0, kvdOf(li))
				vStoreIdx[li] = opIdx - 1
			} else if owns {
				if fuseK {
					ck := emit()
					setQKNormRope(ck, kProj, 0, kNormBufs[li], kCaches[li], 0, kvOf(li), li) // kNorm+rope -> kCache @ row pos (rebound/token)
					kRopeIdx[li] = opIdx - 1
				} else {
					if hasKN {
						setRMSRows(emit(), kProj, kNormBufs[li], kProj, kvOf(li), hdOf(li))
					}
					ck := emit()
					setRope(ck, kProj, kCaches[li], kvOf(li), li) // -> kCache @ row pos (rebound/token)
					kRopeIdx[li] = opIdx - 1
				}
				cv := emitNB()                                                                   // 2nd consumer of `normed` (q barriered it) — overlap
				recInputProj(cv, li, inBuf, anwBufs[li], normed, vCaches[li], 0, vProjOf(li)) // -> vCache @ row pos (rebound/token); K==V layers project via wK
				vIdx[li] = opIdx - 1
				if valueNormOnes != nil { // gemma4 value-norm on the new V row (per head; rebound/token)
					cvn := emit()
					setRMSRows(cvn, vCaches[li], valueNormOnes, vCaches[li], kvOf(li), hdOf(li))
					vNormIdx[li] = opIdx - 1
				}
			} else {
				if fuseK {
					setQKNormRope(emit(), kProj, 0, kNormBufs[li], kThrow, 0, kvOf(li), li) // kNorm+rope -> discard
				} else {
					if hasKN {
						setRMSRows(emit(), kProj, kNormBufs[li], kProj, kvOf(li), hdOf(li))
					}
					setRope(emit(), kProj, kThrow, kvOf(li), li) // discarded
				}
				recInputProj(emitNB(), li, inBuf, anwBufs[li], normed, vThrow, 0, vProjOf(li)) // discarded; 2nd consumer of `normed` — overlap
				if valueNormOnes != nil {
					setRMSRows(emit(), vThrow, valueNormOnes, vThrow, kvOf(li), hdOf(li)) // discarded (keeps the op layout uniform)
				}
			}
			// SDPA over the owner's cache; sliding layers read the windowed slice.
			// SDPA over the owner's cache through the SHARED emitSDPA body (with encSDPAStrided). nBufForLayer
			// is the per-token-VARYING N buffer (rebound at replay if sliding); k/v bind at offset 0 here and
			// the replay rebinds the sliding read offset. gqa/strides/scale bind the same memoised scalars
			// gqaOf/sdpaStrideOf/sdpaScaleB hold. attendK read offset rebound/token if sliding.
			hd, kv := hdOf(li), kvOf(li)
			kvd := int64(kv * hd)
			q8Read := kvQ8.on(ownerIdx) // sharers of a q8 owner read q8 too
			if sdpa2PassICBBlocks > 0 && specs[li].Attention == model.GlobalAttention {
				// GLOBAL layer deep-decode: the 2-pass pair fans the growing-cache reduction over
				// blocks threadgroups (pass 1) and merges the partials (pass 2) — the recorded
				// replacement for the single-pass kernel that serialised the whole cache on one
				// threadgroup per head. N binds the same rebindable nGlobalBuf; K/V bind at slots
				// 1/2 exactly as the single-pass op, so the replay's rebind indices are unchanged.
				if q8Read {
					emitSDPAVector2Pass1Q8(fastICBSink{emit()}, sdpaQ8P1PSOByHd[hd], qr, attendK, attendV,
						p2Partials, p2Sums, p2Maxs, kvQ8.kScales[ownerIdx], kvQ8.vScales[ownerIdx], 0, 0,
						nBufForLayer, nHeads, kv, 0, sdpa2PassICBBlocks, int64(hd), kvd, int64(hd), kvd, scale)
				} else {
					emitSDPA2Pass1NAt(fastICBSink{emit()}, sdpa2Pass1PSOByHd[hd], qr, 0, attendK, attendV,
						p2Partials, p2Sums, p2Maxs, 0, nBufForLayer, 1, nHeads, kv, 0, sdpa2PassICBBlocks,
						int64(hd), kvd, int64(hd), kvd, scale)
				}
				sdpaIdx[li] = opIdx - 1
				emitSDPA2Pass2(fastICBSink{emit()}, sdpa2Pass2PSOByHd[hd], p2Partials, p2Sums, p2Maxs,
					attn, 1, nHeads, sdpa2PassICBBlocks)
			} else if q8Read {
				emitSDPAVectorQ8(fastICBSink{emit()}, sdpaQ8PSOByHd[hd], qr, attendK, attendV, attn,
					kvQ8.kScales[ownerIdx], kvQ8.vScales[ownerIdx], 0, 0, nBufForLayer,
					nHeads, kv, 0, int64(hd), kvd, int64(hd), kvd, scale)
				sdpaIdx[li] = opIdx - 1
			} else {
				emitSDPA(fastICBSink{emit()}, sdpaPSOByHd[hd], qr, attendK, attendV, attn, 0, nBufForLayer,
					nHeads, kv, 0, int64(hd), kvd, int64(hd), kvd, scale)
				sdpaIdx[li] = opIdx - 1
			}
			recordProj(li, emit(), attn, attnOut, 0, projO)
			if hasPA && useFusedResRMS { // fused: hBuf = inBuf + rms(Wo·attn) — one op, one fewer barrier
				setRMSResidual(emit(), attnOut, postAttnBufs[li], inBuf, hBuf)
			} else {
				if hasPA { // gemma4 post-attention norm on Wo·attn before the residual (in-place)
					setRMS(emit(), attnOut, postAttnBufs[li], attnOut)
				}
				setBin(emit(), addPSO, inBuf, attnOut, hBuf, dModel)
			}

			// --- MLP half --- (lff = this layer's FFN width; the FFN ops dispatch only lff
			// elements + bind this width's count buffer — gemma4 MatFormer varies it per layer)
			lff := lffOf(li)
			ffCntB := ffCntOf(lff)
			if recordFusedRMSProj == nil { // fused path folds this rms into gate/up below
				setRMS(emit(), hBuf, mnwBufs[li], mlpNormed)
			}
			recInputProj(emitFFN(), li, hBuf, mnwBufs[li], mlpNormed, gate, 0, projGate)
			recInputProj(emitNB(), li, hBuf, mnwBufs[li], mlpNormed, up, 0, projUp) // 2nd consumer of `mlpNormed` (gate barriered it) — overlap gate
			if hasFusedGELU {                                                       // fused gelu(gate)·up — one ICB command, the binary-op ABI with the gelu pipeline
				setBin(emitFFN(), geluICBPSO, gate, up, gated, lff)
			} else {
				setBin(emit(), mulPSO, gate, gate, x2, lff)
				setBin(emit(), mulPSO, x2, gate, x3, lff)
				setBin(emit(), mulPSO, x3, c044, x3s, lff)
				setBin(emit(), addPSO, gate, x3s, inner, lff)
				setBin(emit(), mulPSO, inner, c079, scaled, lff)
				ct := emit()
				setICBPSO(ct, tanhPSO)
				setICBKernelBuffer(ct, scaled, 0, 0)
				setICBKernelBuffer(ct, tnh, 0, 1)
				setICBKernelBuffer(ct, ffCntB, 0, 2)
				concurrentDispatchThreads(ct, metal.MTLSize{Width: uint(lff), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(lff), Height: 1, Depth: 1})
				setBin(emit(), addPSO, tnh, c1c, onePlus, lff)
				setBin(emit(), mulPSO, gate, c05, halfG, lff)
				setBin(emit(), mulPSO, halfG, onePlus, gelu, lff)
				setBin(emit(), mulPSO, gelu, up, gated, lff)
			}
			recordProj(li, emitFFN(), gated, down, 0, projDown)
			if hasPF && useFusedResRMS { // fused: outBuf = hBuf + rms(Wdown·…) — one op, one fewer barrier
				c := emit()
				setRMSResidual(c, down, postFFBufs[li], hBuf, outBuf)
				if li == nLayers-1 {
					finalOutIdx, finalOutBind, hasFinalOut = opIdx-1, 3, true
				}
			} else {
				if hasPF { // gemma4 post-feed-forward norm on Wdown·… before the residual (in-place)
					setRMS(emit(), down, postFFBufs[li], down)
				}
				c := emit()
				setBin(c, addPSO, hBuf, down, outBuf, dModel)
				if li == nLayers-1 {
					finalOutIdx, finalOutBind, hasFinalOut = opIdx-1, 2, true
				}
			}
			if hasPLE {
				pleOff := uint(li * ple.pliDim * bf16Size)
				if hasFusedGELU { // fused gelu(pleGate)·pleInput — the binary-op ABI with the gelu pipeline (pleInput at offset)
					ple.recordGate(li, emit(), outBuf, pleGate)
					setBinOffsets(emit(), geluICBPSO, pleGate, 0, pleInput, pleOff, pleGated, 0, ple.pliDim)
				} else {
					ple.recordGate(li, emit(), outBuf, pleGate)
					setBin(emit(), mulPSO, pleGate, pleGate, x2, ple.pliDim)
					setBin(emit(), mulPSO, x2, pleGate, x3, ple.pliDim)
					setBin(emit(), mulPSO, x3, c044, x3s, ple.pliDim)
					setBin(emit(), addPSO, pleGate, x3s, inner, ple.pliDim)
					setBin(emit(), mulPSO, inner, c079, scaled, ple.pliDim)
					ct := emit()
					setICBPSO(ct, tanhPSO)
					setICBKernelBuffer(ct, scaled, 0, 0)
					setICBKernelBuffer(ct, tnh, 0, 1)
					setICBKernelBuffer(ct, pleCntB, 0, 2)
					concurrentDispatchThreads(ct, metal.MTLSize{Width: uint(ple.pliDim), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(ple.pliDim), Height: 1, Depth: 1})
					setBin(emit(), addPSO, tnh, c1c, onePlus, ple.pliDim)
					setBin(emit(), mulPSO, pleGate, c05, halfG, ple.pliDim)
					setBin(emit(), mulPSO, halfG, onePlus, gelu, ple.pliDim)
					setBinOffsets(emit(), mulPSO, gelu, 0, pleInput, pleOff, pleGated, 0, ple.pliDim)
				}
				ple.recordProj(li, emit(), pleGated, pleProj)
				// (the PLE post-norm residual stays un-fused: the fused kernel diverges ~2 ULP from the
				// PerLayerInputGate* re-encode / its CPU reference on the dModel axis — byte-parity-hostile.)
				setRMS(emit(), pleProj, ple.postNormBufs[li], pleNorm)
				c := emit()
				setBin(c, addPSO, outBuf, pleNorm, outBuf, dModel)
				if li == nLayers-1 {
					finalOutIdx, finalOutBind, hasFinalOut = opIdx-1, 2, true
				}
			}
			if hasLayerScalar {
				c := emit()
				setBin(c, mulPSO, outBuf, layerScalarFor(li), outBuf, dModel)
				if li == nLayers-1 {
					finalOutIdx, finalOutBind, hasFinalOut = opIdx-1, 2, true
				}
			}
		}
		// the per-layer op-count is invariant to dFF (the gelu/no-gelu + owner/sharer branches
		// are fixed-count), so the running index must land exactly on `total`. A mismatch means
		// the recorded layout diverged from opsPerLayer·nLayers — a recorder bug, not a numeric
		// drift; fail loud rather than replay a misaligned ICB.
		if opIdx != total {
			coreErr = core.NewError(core.Sprintf("native.decodeForwardArchICBCore: recorded %d ops, expected %d (opsPerLayer=%d × %d layers + %d global 2-pass + %d q8 stores) — heterogeneous layout misaligned", opIdx, total, opsPerLayer, nLayers, nGlobal2Pass, nQ8Store))
			return
		}

		lastOut := ping[nLayers%2]
		if cap(sc.residentRes) < len(resident) {
			sc.residentRes = make([]metal.MTLResource, len(resident))
		}
		residentRes := sc.residentRes[:len(resident)]
		for i, bb := range resident {
			residentRes[i] = bb
		}
		sc.residentResIDs = resourceIDsForFastUse(sc.residentResIDs, residentRes)
		residentResIDs := sc.residentResIDs
		rng := recorder.Range

		optCb := commandBufferFast(queue)
		blit := blitCommandEncoderFast(optCb)
		optimizeIndirectCommandBufferWithRangeFast(blit, icb, rng)
		endBlitEncodingFast(blit)
		commitCommandBufferFast(optCb)
		waitUntilCompletedFast(optCb)

		plePliDim, pleRuntime := 0, (*archDecodePLEInputs)(nil)
		if hasPLE {
			plePliDim, pleRuntime = ple.pliDim, ple.runtime
		}
		rowBytesByLayer := sc.rowBytes[:nLayers]
		cacheRowsByLayer := sc.cacheRows[:nLayers]
		for li := range nLayers {
			rowBytesByLayer[li] = kvdOf(li) * bf16Size
			cacheRowsByLayer[li] = 0
			if specs[li].OwnsCache() && kCaches[li] != nil {
				// Capacity as actually ALLOCATED (rows), not maxLen — a caller that bounded a
				// sliding owner's buffer to slidingWindow rows gets ring rebind for free;
				// a caller that kept the old maxLen-sized buffer gets the old linear rebind
				// (pos%maxLen == pos), byte-identical. q8 rows are int8: kvd bytes/row —
				// dividing by the bf16 stride would HALVE the capacity and ring-wrap a
				// global cache at maxLen/2.
				physRow := rowBytesByLayer[li]
				if q8On(li) {
					physRow = kvdOf(li)
				}
				cacheRowsByLayer[li] = int(bufferLengthFast(kCaches[li])) / physRow
			}
		}
		r = &archICBReplay{
			icb: icb, rng: rng, residentRes: residentRes, residentResIDs: residentResIDs,
			scratch: sc,
			specs:   specs, nLayers: nLayers, vOutBind: vOutBind, kRopeBind: kRopeBindIdx, hasValueNorm: valueNormOnes != nil,
			kRopeIdx: kRopeIdx, vIdx: vIdx, vNormIdx: vNormIdx, sdpaIdx: sdpaIdx, barrierOps: barrierOps,
			kCaches: kCaches, vCaches: vCaches,
			kvQ8: kvQ8, kStoreIdx: kStoreIdx, vStoreIdx: vStoreIdx,
			offBuf: offBuf, nGlobalBuf: nGlobalBuf, nSlidingBuf: nSlidingBuf,
			ping: ping, ping0: ping[0], lastOut: lastOut, pleInput: pleInput,
			finalOutIdx: finalOutIdx, finalOutBind: finalOutBind, hasFinalOut: hasFinalOut,
			hasPLE: hasPLE, plePliDim: plePliDim, pleRuntime: pleRuntime,
			opsPerLayer: uint(opsPerLayer),
			rowBytes:    rowBytesByLayer, cacheRows: cacheRowsByLayer, slidingWindow: slidingWindow, dModel: dModel,
		}
		r.cacheKVContents()
		r.cacheStepContents()
		r.cacheLastOutContents()
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return r, nil
}

// decodeForwardArchICBCore records the arch ICB then replays it across the whole input sequence —
// the batch encode-bypass. It is recordArchICB + runBatch; byte-identical to the pre-split core.
func decodeForwardArchICBCore(
	outputs [][]byte, inputs [][]byte, specs []model.LayerSpec,
	anwBufs, mnwBufs, kCaches, vCaches, projResident []metal.MTLBuffer,
	qNormBufs, kNormBufs, postAttnBufs, postFFBufs []metal.MTLBuffer,
	layerScalarBufs []metal.MTLBuffer, ple *archICBPLEPlan,
	recordProj func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer, outOff uint, p projIndex),
	recordFusedRMSProj func(li int, c metal.MTLIndirectComputeCommand, rawIn, normW, epsB, out metal.MTLBuffer, outOff uint, p projIndex),
	vOutBind uint, valueNormOnes metal.MTLBuffer,
	dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int,
	perLayerDFF []int,
	base, scale, eps float32,
	useCallerOut bool,
) ([][]byte, error) {
	r, err := recordArchICB(specs, anwBufs, mnwBufs, kCaches, vCaches, projResident, qNormBufs, kNormBufs, postAttnBufs, postFFBufs, layerScalarBufs, ple, recordProj, recordFusedRMSProj, vOutBind, valueNormOnes, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, perLayerDFF, simpleICBRope(base, headDim), scale, eps, nil)
	if err != nil {
		return nil, err
	}
	outputs, err = r.runBatchInto(outputs, inputs, useCallerOut)
	r.releaseScratch()
	return outputs, err
}

// recordArchICBBF16 records the bf16 arch ICB and returns the held *archICBReplay — the bf16
// sibling of recordArchICBQuant, for the ArchSession (record once at open, stepBody per token).
// It rides the QUANT recorder with every projection wrapped as a sidecar-less QuantWeight
// (dense bf16 bytes, no scales/biases): the recorder's mkW/psoFor/setQMV already dispatch
// sidecar-less weights through the tiled bf16 gemv (the pack-level-fusion contract on
// QuantWeight), so the per-layer head-dim, K==V, MatFormer-DFF and PLE handling are shared
// rather than forked — and the recorder derives vOutBind=3 (gemv out) from the dense V.
// Unlike the whole-seq batch DecodeForwardArchICB below, this therefore records gemma4's
// MIXED head-dim (wider global layers) fine. Caches + the PLE runtime are parameters exactly
// as in the quant recorder; a nil WV (gemma4 K==V) wraps to a nil Packed ⇒ V rides the k-proj.
func recordArchICBBF16(
	layers []DecodeLayerWeights, specs []model.LayerSpec,
	kCaches, vCaches []metal.MTLBuffer,
	pleRuntime *archDecodePLEInputs, pliDim int,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	rope icbRope, scale, eps float32, valueNorm bool,
	kvQ8 *archICBKVQ8,
) (*archICBReplay, error) {
	qlayers := make([]QuantizedLayerWeights, len(layers))
	dense := func(b []byte) QuantWeight { return QuantWeight{Packed: b} }
	for li := range layers {
		w := layers[li]
		qlayers[li] = QuantizedLayerWeights{
			AttnNormW: w.AttnNormW, MLPNormW: w.MLPNormW,
			Q: dense(w.WQ), K: dense(w.WK), V: dense(w.WV), O: dense(w.WO),
			Gate: dense(w.WGate), Up: dense(w.WUp), Down: dense(w.WDown),
			DFF:           w.DFF,
			PostAttnNormW: w.PostAttnNormW, PostFFNormW: w.PostFFNormW,
			QNormW: w.QNormW, KNormW: w.KNormW,
			LayerScalarW:           w.LayerScalarW,
			PerLayerGate:           dense(w.PerLayerGate),
			PerLayerProjection:     dense(w.PerLayerProjection),
			PostPerLayerInputNormW: w.PostPerLayerInputNormW,
		}
	}
	return recordArchICBQuant(qlayers, specs, kCaches, vCaches, pleRuntime, pliDim, 0, 0, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, rope, scale, eps, valueNorm, kvQ8)
}

// DecodeForwardArchICB is the bf16 ARCH-driven cache-grow ICB: the encode-bypass replay
// of DecodeForwardArch (KV-share + sliding-window), recorded once and replayed per token.
// It builds a gemv recorder + the per-layer weight/cache buffers (caches for OWNER layers
// only) and runs decodeForwardArchICBCore. Byte-for-byte equal to DecodeForwardArch on
// the same arch (gated). MoE layers are not supported (rejected). All bf16.
func DecodeForwardArchICB(
	inputs [][]byte, layers []DecodeLayerWeights, specs []model.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32, valueNorm bool,
	pleArgs ...ArchPLEBF16,
) ([][]byte, error) {
	return decodeForwardArchICBInto(nil, inputs, layers, specs, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, base, scale, eps, valueNorm, false, pleArgs...)
}

// DecodeForwardArchICBInto is DecodeForwardArchICB with caller-owned per-token
// output storage. Output slices with enough capacity are reused for the final
// hidden readback from each ICB replay.
func DecodeForwardArchICBInto(
	outputs [][]byte, inputs [][]byte, layers []DecodeLayerWeights, specs []model.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32, valueNorm bool,
	pleArgs ...ArchPLEBF16,
) ([][]byte, error) {
	return decodeForwardArchICBInto(outputs, inputs, layers, specs, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, base, scale, eps, valueNorm, true, pleArgs...)
}

func decodeForwardArchICBInto(
	outputs [][]byte, inputs [][]byte, layers []DecodeLayerWeights, specs []model.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32, valueNorm bool,
	useCallerOut bool,
	pleArgs ...ArchPLEBF16,
) ([][]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	nLayers, T := len(layers), len(inputs)
	if nLayers == 0 || T == 0 {
		return nil, core.NewError("native.DecodeForwardArchICB: need layers and inputs")
	}
	if len(specs) != nLayers {
		return nil, core.NewError("native.DecodeForwardArchICB: specs length must equal layers")
	}
	if T > maxLen {
		return nil, core.NewError("native.DecodeForwardArchICB: more tokens than maxLen cache rows")
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardArchICB: each input must be dModel bf16 bytes")
		}
	}
	hasMoE, mixedHeadDim := false, false
	for li := range specs {
		o := specs[li].KVShareFrom
		if o < 0 || o > li || (o != li && !specs[o].OwnsCache()) {
			return nil, core.NewError("native.DecodeForwardArchICB: KVShareFrom must reference an earlier owner layer")
		}
		if specs[li].MoE {
			hasMoE = true
		}
		if headDimOf(specs[li], headDim) != headDim {
			mixedHeadDim = true // gemma4 global layers are WIDER (e.g. 512 vs sliding 256)
		}
	}
	// This whole-sequence recorder records ONE uniform projection shape + a single base-rope spectrum
	// for every layer (qDim/kvDim/psoQ/psoKV and simpleICBRope are computed once below). It therefore
	// cannot represent MoE (host router) NOR gemma4's per-layer head dim (the global layers' wider
	// head_dim + proportional partial rope). For those, fall back to the per-layer-correct re-encode
	// forward — byte-identical, just not the ICB fast path for this (cold, batch) call. The SESSION
	// path keeps the fast per-hd ICB (it records per-head-dim); this is only the whole-seq batch API.
	if hasMoE || mixedHeadDim {
		if useCallerOut {
			return DecodeForwardArchInto(outputs, inputs, layers, specs, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, base, scale, eps, valueNorm, pleArgs...)
		}
		return DecodeForwardArch(inputs, layers, specs, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, base, scale, eps, valueNorm, pleArgs...)
	}

	setup := getArchICBSetupScratch(nLayers)
	defer putArchICBSetupScratch(setup)

	// per-layer FFN width (gemma4 E2B/E4B MatFormer): lFF[li] (from w.DFF, fallback dFF).
	lFF := setup.lFF
	ffnWidthIndex := setup.ffnWidthIndex
	uniqueDFF := setup.uniqueDFF
	for li := range layers {
		lFF[li] = dFF
		if layers[li].DFF > 0 {
			lFF[li] = layers[li].DFF
		}
		idx := slices.Index(uniqueDFF, lFF[li])
		if idx < 0 {
			idx = len(uniqueDFF)
			uniqueDFF = append(uniqueDFF, lFF[li])
		}
		ffnWidthIndex[li] = idx
	}
	setup.uniqueDFF = uniqueDFF
	plePayload, err := singleArchPLEBF16("native.DecodeForwardArchICB", pleArgs)
	if err != nil {
		return nil, err
	}
	pleRuntime, pliDim, err := archPLEBF16Runtime("native.DecodeForwardArchICB", plePayload, nLayers, T, dModel, eps)
	if err != nil {
		return nil, err
	}
	var pleLayers []pleLayer
	if pleRuntime != nil {
		pleLayers, err = bf16PLELayers("native.DecodeForwardArchICB", layers, dModel, pliDim)
		if err != nil {
			return nil, err
		}
	}

	gemvPSO := func(inDim, outDim int) (metal.MTLComputePipelineState, int, int, int, int, error) {
		bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
		p, e := pipelineForICB(gemvKernelName("bfloat16", bm, bn, sm, sn, tm, tn))
		return p, bm, bn, sm, tm, e
	}
	psoQ, bmQ, bnQ, smQ, tmQ, err := gemvPSO(dModel, qDim)
	if err != nil {
		return nil, err
	}
	psoKV, bmKV, bnKV, smKV, tmKV, err := gemvPSO(dModel, kvDim)
	if err != nil {
		return nil, err
	}
	psoO, bmO, bnO, smO, tmO, err := gemvPSO(qDim, dModel)
	if err != nil {
		return nil, err
	}
	// gate/up (dModel→lff) and down (lff→dModel) gemv PSOs + tiles, one per distinct FFN width.
	ffUp := setup.ffUp[:len(uniqueDFF)]
	ffDown := setup.ffDown[:len(uniqueDFF)]
	for i, lff := range uniqueDFF {
		p, bm, bn, sm, tm, e := gemvPSO(dModel, lff)
		if e != nil {
			return nil, e
		}
		ffUp[i] = archICBGemvShape{p, bm, bn, sm, tm}
		p2, bm2, bn2, sm2, tm2, e2 := gemvPSO(lff, dModel)
		if e2 != nil {
			return nil, e2
		}
		ffDown[i] = archICBGemvShape{p2, bm2, bn2, sm2, tm2}
	}
	setup.ffUp = ffUp
	setup.ffDown = ffDown
	var pleGateShape, pleProjShape archICBGemvShape
	if pleRuntime != nil {
		p, bm, bn, sm, tm, e := gemvPSO(dModel, pliDim)
		if e != nil {
			return nil, e
		}
		pleGateShape = archICBGemvShape{p, bm, bn, sm, tm}
		p, bm, bn, sm, tm, e = gemvPSO(pliDim, dModel)
		if e != nil {
			return nil, e
		}
		pleProjShape = archICBGemvShape{p, bm, bn, sm, tm}
	}

	var coreErr error
	withAutoreleasePool(func() {
		anwBufs := setup.anwBufs
		mnwBufs := setup.mnwBufs
		qNormBufs := setup.qNormBufs
		kNormBufs := setup.kNormBufs
		postAttnBufs := setup.postAttnBufs
		postFFBufs := setup.postFFBufs
		layerScalarBufs := setup.layerScalarBufs
		kCaches := setup.kCaches
		vCaches := setup.vCaches
		lb := setup.lb
		pleLB := setup.pleLB
		plePostNorms := setup.plePostNorms
		cacheBytesFull := uint(maxLen * kvDim * bf16Size)
		cacheBytesSliding := cacheBytesFull
		if slidingWindow > 0 && slidingWindow < maxLen {
			// Bounded ring — the sliding-window KV memory fix: a sliding owner only ever
			// attends its own window, so it only ever needs slidingWindow rows of storage.
			// prepareStepRebind detects the smaller allocation (via the actual buffer length)
			// and rebinds pos%cacheRows instead of the absolute position.
			cacheBytesSliding = uint(slidingWindow * kvDim * bf16Size)
		}
		// presized to the upper bound (every layer's ≤7 projection buffers, the 16 shared trailing
		// scalar buffers, plus ≤3 FFN dim scalars per distinct dFF width) so the per-forward build
		// never geometrically regrows its backing array — K==V layers leave the v-proj slot unused.
		// Byte-identical.
		projResident := setup.projResident
		residentOrNil := func(b []byte) metal.MTLBuffer {
			if len(b) == 0 {
				return nil
			}
			return residentBytes(b)
		}
		for li := range layers {
			w := layers[li]
			anwBufs[li] = residentBytes(w.AttnNormW)
			mnwBufs[li] = residentBytes(w.MLPNormW)
			qNormBufs[li] = residentOrNil(w.QNormW)
			kNormBufs[li] = residentOrNil(w.KNormW)
			postAttnBufs[li] = residentOrNil(w.PostAttnNormW)
			postFFBufs[li] = residentOrNil(w.PostFFNormW)
			layerScalarBufs[li] = layerScalarBuf(w.LayerScalarW, dModel)
			if specs[li].OwnsCache() {
				cacheBytes := cacheBytesFull
				if specs[li].Attention != model.GlobalAttention {
					cacheBytes = cacheBytesSliding
				}
				kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
				vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			}
			lb[li] = archICBLayerProjBuffers{residentBytes(w.WQ), residentBytes(w.WK), residentOrNil(w.WV), residentBytes(w.WO), residentBytes(w.WGate), residentBytes(w.WUp), residentBytes(w.WDown)}
			projResident = append(projResident, lb[li].wq, lb[li].wk, lb[li].wo, lb[li].wg, lb[li].wu, lb[li].wd)
			if lb[li].wv != nil { // gemma4 K==V layers carry no v_proj
				projResident = append(projResident, lb[li].wv)
			}
			if pleRuntime != nil {
				pleLB[li] = archICBPLEProjBuffers{residentBytes(pleLayers[li].gate.Packed), residentBytes(pleLayers[li].proj.Packed)}
				plePostNorms[li] = residentBytes(pleLayers[li].postNorm)
			}
		}
		qInB, qOutB, qLdB := scalarI32(int32(dModel)), scalarI32(int32(qDim)), scalarI32(int32(dModel))
		kvInB, kvOutB, kvLdB := scalarI32(int32(dModel)), scalarI32(int32(kvDim)), scalarI32(int32(dModel))
		oInB, oOutB, oLdB := scalarI32(int32(qDim)), scalarI32(int32(dModel)), scalarI32(int32(qDim))
		// FFN gemv dim scalars: the dModel-side (up's in/ld, down's out) are shared; the lff-side
		// (up's out, down's in/ld) is one buffer per distinct width. All appended to projResident.
		fInB, fLdB, dOutB := scalarI32(int32(dModel)), scalarI32(int32(dModel)), scalarI32(int32(dModel))
		ffnScalars := setup.ffnScalars[:len(uniqueDFF)]
		for i, lff := range uniqueDFF {
			ffnScalars[i] = archICBFFNScalarBuffers{
				fOut: scalarI32(int32(lff)),
				dIn:  scalarI32(int32(lff)),
				dLd:  scalarI32(int32(lff)),
			}
		}
		setup.ffnScalars = ffnScalars
		bndB, bshB, vsB, msB := scalarI32(1), scalarI32(1), scalarI64(0), scalarI64(0)
		projResident = append(projResident, qInB, qOutB, qLdB, kvInB, kvOutB, kvLdB, oInB, oOutB, oLdB, fInB, fLdB, dOutB, bndB, bshB, vsB, msB)
		for _, s := range ffnScalars {
			projResident = append(projResident, s.fOut, s.dIn, s.dLd)
		}
		setup.projResident = projResident

		// bf16 tiled gemv through the SHARED emitGemv body (with encGemvBF16To). K/N/ld/batch bind the same
		// memoised scalars inB/outB/ldB/bndB/bshB/vsB/msB hold, so the call passes inDim/outDim values.
		setGemv := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, mat, vec, o metal.MTLBuffer, outOff uint, inDim, outDim, bm, bn, sm, tm int) {
			emitGemv(fastICBSink{c}, pso, mat, 0, vec, o, outOff, inDim, outDim, bm, bn, sm, tm)
		}
		var plePlan *archICBPLEPlan
		if pleRuntime != nil {
			pleGateInB, pleGateOutB, pleGateLdB := scalarI32(int32(dModel)), scalarI32(int32(pliDim)), scalarI32(int32(dModel))
			pleProjInB, pleProjOutB, pleProjLdB := scalarI32(int32(pliDim)), scalarI32(int32(dModel)), scalarI32(int32(pliDim))
			pleResident := append(setup.pleResident, pleGateInB, pleGateOutB, pleGateLdB, pleProjInB, pleProjOutB, pleProjLdB)
			for li := range pleLB {
				pleResident = append(pleResident, pleLB[li].gate, pleLB[li].proj)
			}
			setup.pleResident = pleResident
			plePlan = &archICBPLEPlan{
				runtime: pleRuntime, pliDim: pliDim, postNormBufs: plePostNorms, resident: pleResident,
			}
			plePlan.recordGate = func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer) {
				g := pleGateShape
				setGemv(c, g.pso, pleLB[li].gate, vec, out, 0, dModel, pliDim, g.bm, g.bn, g.sm, g.tm)
			}
			plePlan.recordProj = func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer) {
				g := pleProjShape
				setGemv(c, g.pso, pleLB[li].proj, vec, out, 0, pliDim, dModel, g.bm, g.bn, g.sm, g.tm)
			}
		}
		recordProj := func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer, outOff uint, p projIndex) {
			l := lb[li]
			switch p {
			case projQ:
				setGemv(c, psoQ, l.wq, vec, out, outOff, dModel, qDim, bmQ, bnQ, smQ, tmQ)
			case projK:
				setGemv(c, psoKV, l.wk, vec, out, outOff, dModel, kvDim, bmKV, bnKV, smKV, tmKV)
			case projV:
				setGemv(c, psoKV, l.wv, vec, out, outOff, dModel, kvDim, bmKV, bnKV, smKV, tmKV)
			case projO:
				setGemv(c, psoO, l.wo, vec, out, outOff, qDim, dModel, bmO, bnO, smO, tmO)
			case projGate:
				lff := lFF[li]
				u := ffUp[ffnWidthIndex[li]]
				setGemv(c, u.pso, l.wg, vec, out, outOff, dModel, lff, u.bm, u.bn, u.sm, u.tm)
			case projUp:
				lff := lFF[li]
				u := ffUp[ffnWidthIndex[li]]
				setGemv(c, u.pso, l.wu, vec, out, outOff, dModel, lff, u.bm, u.bn, u.sm, u.tm)
			case projDown:
				lff := lFF[li]
				d := ffDown[ffnWidthIndex[li]]
				setGemv(c, d.pso, l.wd, vec, out, outOff, lff, dModel, d.bm, d.bn, d.sm, d.tm)
			}
		}
		valueNormOnes := valueNormOnesBuf(valueNorm, maxHeadDimOf(specs, headDim))
		// Resolve the per-layer K==V op selection (LayerSpec.AttentionKEqV) that recordArchICB reads:
		// models loaded through model.Assemble DECLARE it (authoritative); a hand-built caller of this
		// whole-seq API that did not declare falls back to weight presence — a layer with no v_proj has
		// its value ride the k-proj (12B: sliding layers carry V, global layers don't). Cloned so the
		// caller's specs stay untouched.
		keqvSpecs := slices.Clone(specs)
		for li := range keqvSpecs {
			keqvSpecs[li].AttentionKEqV = keqvSpecs[li].AttentionKEqV || len(layers[li].WV) == 0
		}
		outputs, coreErr = decodeForwardArchICBCore(outputs, inputs, keqvSpecs, anwBufs, mnwBufs, kCaches, vCaches, projResident, qNormBufs, kNormBufs, postAttnBufs, postFFBufs, layerScalarBufs, plePlan, recordProj, nil, 3, valueNormOnes, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, lFF, base, scale, eps, useCallerOut)
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return outputs, nil
}
