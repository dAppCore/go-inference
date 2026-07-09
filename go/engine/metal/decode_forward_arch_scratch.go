// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"

	"github.com/tmc/apple/metal"
)

type archBF16LayerBufScratch struct {
	lb         []archLayerBufs
	moeWeights []*MoELayerWeights
	kCaches    []metal.MTLBuffer
	vCaches    []metal.MTLBuffer
	kBytes     []uint
	vBytes     []uint
	kPtrs      []*byte
	vPtrs      []*byte
}

var archBF16LayerBufScratchPool sync.Pool

func newArchBF16LayerBufScratch(nLayers int) *archBF16LayerBufScratch {
	return &archBF16LayerBufScratch{
		lb:         make([]archLayerBufs, nLayers),
		moeWeights: make([]*MoELayerWeights, nLayers),
		kCaches:    make([]metal.MTLBuffer, nLayers),
		vCaches:    make([]metal.MTLBuffer, nLayers),
		kBytes:     make([]uint, nLayers),
		vBytes:     make([]uint, nLayers),
		kPtrs:      make([]*byte, nLayers),
		vPtrs:      make([]*byte, nLayers),
	}
}

func (s *archBF16LayerBufScratch) fits(nLayers int) bool {
	return s != nil &&
		cap(s.lb) >= nLayers && cap(s.moeWeights) >= nLayers &&
		cap(s.kCaches) >= nLayers && cap(s.vCaches) >= nLayers &&
		cap(s.kBytes) >= nLayers && cap(s.vBytes) >= nLayers &&
		cap(s.kPtrs) >= nLayers && cap(s.vPtrs) >= nLayers
}

func (s *archBF16LayerBufScratch) reset(nLayers int) *archBF16LayerBufScratch {
	clear(s.lb)
	clear(s.moeWeights)
	s.lb = s.lb[:nLayers]
	s.moeWeights = s.moeWeights[:nLayers]
	s.kCaches = s.kCaches[:nLayers]
	s.vCaches = s.vCaches[:nLayers]
	s.kBytes = s.kBytes[:nLayers]
	s.vBytes = s.vBytes[:nLayers]
	s.kPtrs = s.kPtrs[:nLayers]
	s.vPtrs = s.vPtrs[:nLayers]
	return s
}

func (s *archBF16LayerBufScratch) kvCache(li int, cacheBytes uint) (metal.MTLBuffer, metal.MTLBuffer, *byte, *byte) {
	if s.kCaches[li] == nil || s.kBytes[li] != cacheBytes {
		s.kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
		s.kBytes[li] = cacheBytes
		s.kPtrs[li] = (*byte)(s.kCaches[li].Contents())
	}
	if s.vCaches[li] == nil || s.vBytes[li] != cacheBytes {
		s.vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
		s.vBytes[li] = cacheBytes
		s.vPtrs[li] = (*byte)(s.vCaches[li].Contents())
	}
	return s.kCaches[li], s.vCaches[li], s.kPtrs[li], s.vPtrs[li]
}

func getArchBF16LayerBufScratch(nLayers int) *archBF16LayerBufScratch {
	if v := archBF16LayerBufScratchPool.Get(); v != nil {
		if s, ok := v.(*archBF16LayerBufScratch); ok && s.fits(nLayers) {
			return s.reset(nLayers)
		}
	}
	return newArchBF16LayerBufScratch(nLayers)
}

func putArchBF16LayerBufScratch(s *archBF16LayerBufScratch) {
	if s != nil {
		archBF16LayerBufScratchPool.Put(s.reset(0))
	}
}

type archQuantLayerBufScratch struct {
	lb      []archLayerBufs
	moe     []*MoEQuantLayerWeights
	moeVals []MoEQuantLayerWeights
	kCaches []metal.MTLBuffer
	vCaches []metal.MTLBuffer
	kBytes  []uint
	vBytes  []uint
	kPtrs   []*byte
	vPtrs   []*byte
}

var archQuantLayerBufScratchPool sync.Pool

func newArchQuantLayerBufScratch(nLayers int) *archQuantLayerBufScratch {
	return &archQuantLayerBufScratch{
		lb:      make([]archLayerBufs, nLayers),
		moe:     make([]*MoEQuantLayerWeights, nLayers),
		moeVals: make([]MoEQuantLayerWeights, nLayers),
		kCaches: make([]metal.MTLBuffer, nLayers),
		vCaches: make([]metal.MTLBuffer, nLayers),
		kBytes:  make([]uint, nLayers),
		vBytes:  make([]uint, nLayers),
		kPtrs:   make([]*byte, nLayers),
		vPtrs:   make([]*byte, nLayers),
	}
}

func (s *archQuantLayerBufScratch) fits(nLayers int) bool {
	return s != nil &&
		cap(s.lb) >= nLayers && cap(s.moe) >= nLayers && cap(s.moeVals) >= nLayers &&
		cap(s.kCaches) >= nLayers && cap(s.vCaches) >= nLayers &&
		cap(s.kBytes) >= nLayers && cap(s.vBytes) >= nLayers &&
		cap(s.kPtrs) >= nLayers && cap(s.vPtrs) >= nLayers
}

func (s *archQuantLayerBufScratch) reset(nLayers int) *archQuantLayerBufScratch {
	clear(s.lb)
	clear(s.moe)
	clear(s.moeVals)
	s.lb = s.lb[:nLayers]
	s.moe = s.moe[:nLayers]
	s.moeVals = s.moeVals[:nLayers]
	s.kCaches = s.kCaches[:nLayers]
	s.vCaches = s.vCaches[:nLayers]
	s.kBytes = s.kBytes[:nLayers]
	s.vBytes = s.vBytes[:nLayers]
	s.kPtrs = s.kPtrs[:nLayers]
	s.vPtrs = s.vPtrs[:nLayers]
	return s
}

func (s *archQuantLayerBufScratch) kvCache(li int, cacheBytes uint) (metal.MTLBuffer, metal.MTLBuffer, *byte, *byte) {
	if s.kCaches[li] == nil || s.kBytes[li] != cacheBytes {
		s.kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
		s.kBytes[li] = cacheBytes
		s.kPtrs[li] = (*byte)(s.kCaches[li].Contents())
	}
	if s.vCaches[li] == nil || s.vBytes[li] != cacheBytes {
		s.vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
		s.vBytes[li] = cacheBytes
		s.vPtrs[li] = (*byte)(s.vCaches[li].Contents())
	}
	return s.kCaches[li], s.vCaches[li], s.kPtrs[li], s.vPtrs[li]
}

func getArchQuantLayerBufScratch(nLayers int) *archQuantLayerBufScratch {
	if v := archQuantLayerBufScratchPool.Get(); v != nil {
		if s, ok := v.(*archQuantLayerBufScratch); ok && s.fits(nLayers) {
			return s.reset(nLayers)
		}
	}
	return newArchQuantLayerBufScratch(nLayers)
}

func putArchQuantLayerBufScratch(s *archQuantLayerBufScratch) {
	if s != nil {
		archQuantLayerBufScratchPool.Put(s.reset(0))
	}
}

type archDecodeCoreScratch struct {
	dModel, qDim, kvDim, nHeads, maxLen, dFF int
	asc                                      attnScratch
	msc                                      mlpScratch
	hBuf, xA, xB, offBuf                     metal.MTLBuffer
	offPtr                                   *int32
	hBufPtr, xAPtr, xBPtr                    *byte
	hostPinned                               *pinnedNoCopyBytes
}

var archDecodeCoreScratchPool sync.Pool

func newArchDecodeCoreScratch(dModel, qDim, kvDim, nHeads, maxLen, dFF int) *archDecodeCoreScratch {
	hBuf := scratchBF16(dModel)
	xA := scratchBF16(dModel)
	xB := scratchBF16(dModel)
	offBuf := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
	sc := &archDecodeCoreScratch{
		dModel: dModel, qDim: qDim, kvDim: kvDim, nHeads: nHeads, maxLen: maxLen, dFF: dFF,
		asc:     newAttnScratch(dModel, qDim, kvDim, nHeads, maxLen),
		msc:     newMLPScratch(dModel, dFF),
		hBuf:    hBuf,
		xA:      xA,
		xB:      xB,
		offBuf:  offBuf,
		offPtr:  (*int32)(offBuf.Contents()),
		hBufPtr: (*byte)(hBuf.Contents()),
		xAPtr:   (*byte)(xA.Contents()),
		xBPtr:   (*byte)(xB.Contents()),
	}
	return sc.reset()
}

func (s *archDecodeCoreScratch) fits(dModel, qDim, kvDim, nHeads, maxLen, dFF int) bool {
	return s != nil &&
		s.dModel == dModel && s.qDim == qDim && s.kvDim == kvDim && s.nHeads == nHeads && s.maxLen == maxLen && s.dFF == dFF &&
		s.hBuf != nil && s.xA != nil && s.xB != nil && s.offBuf != nil &&
		s.offPtr != nil && s.hBufPtr != nil && s.xAPtr != nil && s.xBPtr != nil &&
		s.asc.normed != nil && s.asc.q != nil && s.asc.qr != nil && s.asc.kProj != nil && s.asc.attn != nil && s.asc.attnOut != nil &&
		s.msc.mlpNormed != nil && s.msc.gate != nil && s.msc.up != nil && s.msc.gated != nil && s.msc.down != nil
}

func (s *archDecodeCoreScratch) reset() *archDecodeCoreScratch {
	if s != nil && s.offPtr != nil {
		*s.offPtr = 0
	}
	return s
}

func (s *archDecodeCoreScratch) hostPinnedScratch(byteLen int) (*pinnedNoCopyBytes, error) {
	if s == nil {
		return nil, nil
	}
	if s.hostPinned == nil || len(s.hostPinned.bytes) != byteLen {
		if s.hostPinned != nil {
			s.hostPinned.Close()
			s.hostPinned = nil
		}
		p, err := newPinnedNoCopyBytes(byteLen)
		if err != nil {
			return nil, err
		}
		s.hostPinned = p
	}
	return s.hostPinned, nil
}

func getArchDecodeCoreScratch(dModel, qDim, kvDim, nHeads, maxLen, dFF int) *archDecodeCoreScratch {
	if v := archDecodeCoreScratchPool.Get(); v != nil {
		if s, ok := v.(*archDecodeCoreScratch); ok && s.fits(dModel, qDim, kvDim, nHeads, maxLen, dFF) {
			return s.reset()
		}
	}
	return newArchDecodeCoreScratch(dModel, qDim, kvDim, nHeads, maxLen, dFF)
}

func putArchDecodeCoreScratch(s *archDecodeCoreScratch) {
	if s != nil {
		archDecodeCoreScratchPool.Put(s.reset())
	}
}
