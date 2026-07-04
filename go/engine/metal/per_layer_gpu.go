// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"runtime"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

type perLayerInputsGPUScratchKey struct {
	plDim, dModel int
	projScale     [2]byte
}

type perLayerInputsGPUScratchPool struct {
	mu    sync.Mutex
	items []*perLayerInputsGPUScratch
}

var perLayerInputsGPUScratchPools sync.Map

func perLayerInputsGPUScratchPoolFor(plDim, dModel int, projScale float32) *perLayerInputsGPUScratchPool {
	key := perLayerInputsGPUScratchKey{plDim: plDim, dModel: dModel, projScale: bf16ScalarBytes(projScale)}
	if v, ok := perLayerInputsGPUScratchPools.Load(key); ok {
		return v.(*perLayerInputsGPUScratchPool)
	}
	pool := &perLayerInputsGPUScratchPool{}
	if v, loaded := perLayerInputsGPUScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*perLayerInputsGPUScratchPool)
	}
	return pool
}

// plGPUScratch is the device-buffer scratch for the on-GPU PLE (one set per in-flight pipeline slot).
type plGPUScratch struct {
	perLayer, projected, scaled, projNormed, combined, out metal.MTLBuffer
	projScaleBuf, combineScaleBuf                          metal.MTLBuffer
	projScaleBytes, combineScaleBytes                      [2]byte
	outPtr                                                 *byte
	outPinned                                              *pinnedNoCopyBytes
}

func (s *plGPUScratch) Close() {
	if s == nil {
		return
	}
	if s.outPinned != nil {
		s.outPinned.Close()
		s.outPinned = nil
	}
	s.perLayer, s.projected, s.scaled, s.projNormed, s.combined, s.out = nil, nil, nil, nil, nil, nil
	s.projScaleBuf, s.combineScaleBuf = nil, nil
	s.outPtr = nil
}

func newPLGPUScratch(plDim int, projScale float32) *plGPUScratch {
	nb := func() metal.MTLBuffer {
		return device.NewBufferWithLengthOptions(uint(plDim*bf16Size), metal.MTLResourceStorageModeShared)
	}
	s := &plGPUScratch{
		perLayer: nb(), projected: nb(), scaled: nb(), projNormed: nb(), combined: nb(),
	}
	if pinned, err := newPinnedNoCopyBytes(plDim * bf16Size); err == nil {
		s.outPinned = pinned
		s.out = pinned.buf
		s.outPtr = (*byte)(unsafe.Pointer(&pinned.bytes[0]))
	} else {
		s.out = nb()
		s.outPtr = (*byte)(s.out.Contents())
	}
	s.projScaleBytes = bf16ScalarBytes(projScale)
	s.combineScaleBytes = bf16ScalarBytes(gemma4PerLayerCombineScale)
	s.projScaleBuf = bf16ConstBuffer(1, projScale)
	s.combineScaleBuf = bf16ConstBuffer(1, gemma4PerLayerCombineScale)
	return s
}

type perLayerInputsGPUScratch struct {
	plDim, dModel int
	projScale     float32
	token, emb    *pinnedNoCopyBytes
	embView       cachedNoCopyBytesView
	pl            *plGPUScratch
	outViewPtr    uintptr
	outViewLen    int
	outView       metal.MTLBuffer
	outViewPinned *pinnedNoCopyBytes
}

func newPerLayerInputsGPUScratch(plDim, dModel int, projScale float32) (*perLayerInputsGPUScratch, error) {
	token, err := newPinnedNoCopyBytes(4)
	if err != nil {
		return nil, err
	}
	emb, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		token.Close()
		return nil, err
	}
	return &perLayerInputsGPUScratch{
		plDim:     plDim,
		dModel:    dModel,
		projScale: projScale,
		token:     token,
		emb:       emb,
		pl:        newPLGPUScratch(plDim, projScale),
	}, nil
}

func getPerLayerInputsGPUScratch(plDim, dModel int, projScale float32) (*perLayerInputsGPUScratch, error) {
	pool := perLayerInputsGPUScratchPoolFor(plDim, dModel, projScale)
	if s := pool.Get(); s != nil {
		if s.plDim == plDim && s.dModel == dModel && s.projScale == projScale && s.token != nil && s.emb != nil && s.pl != nil && s.pl.out != nil {
			return s, nil
		}
		s.Close()
	}
	return newPerLayerInputsGPUScratch(plDim, dModel, projScale)
}

func putPerLayerInputsGPUScratch(s *perLayerInputsGPUScratch) {
	if s != nil && s.plDim > 0 && s.dModel > 0 && s.token != nil && s.emb != nil && s.pl != nil && s.pl.out != nil {
		perLayerInputsGPUScratchPoolFor(s.plDim, s.dModel, s.projScale).Put(s)
	}
}

func (p *perLayerInputsGPUScratchPool) Get() *perLayerInputsGPUScratch {
	p.mu.Lock()
	defer p.mu.Unlock()
	n := len(p.items)
	if n == 0 {
		return nil
	}
	s := p.items[n-1]
	p.items[n-1] = nil
	p.items = p.items[:n-1]
	return s
}

func (p *perLayerInputsGPUScratchPool) Put(s *perLayerInputsGPUScratch) {
	if s == nil {
		return
	}
	p.mu.Lock()
	p.items = append(p.items, s)
	p.mu.Unlock()
}

func (s *perLayerInputsGPUScratch) Close() {
	if s == nil {
		return
	}
	s.closeOutputView()
	if s.token != nil {
		s.token.Close()
		s.token = nil
	}
	if s.emb != nil {
		s.emb.Close()
		s.emb = nil
	}
	s.embView.Close()
	if s.pl != nil {
		s.pl.Close()
		s.pl = nil
	}
	s.plDim, s.dModel = 0, 0
	s.projScale = 0
}

func (s *perLayerInputsGPUScratch) closeOutputView() {
	if s == nil {
		return
	}
	if s.outViewPinned != nil {
		s.outViewPinned.Close()
	}
	s.outViewPtr = 0
	s.outViewLen = 0
	s.outView = nil
	s.outViewPinned = nil
}

func (s *perLayerInputsGPUScratch) outputView(out []byte) (metal.MTLBuffer, *byte, bool) {
	if s == nil || len(out) == 0 {
		return nil, nil, false
	}
	ptr := uintptr(unsafe.Pointer(&out[0]))
	if s.outView != nil && s.outViewPtr == ptr && s.outViewLen == len(out) {
		return s.outView, (*byte)(unsafe.Pointer(&out[0])), true
	}
	s.closeOutputView()
	if buf, ok := registeredPinnedNoCopyBytes(out); ok {
		s.outViewPtr = ptr
		s.outViewLen = len(out)
		s.outView = buf
		s.outViewPinned = nil
		return buf, (*byte)(unsafe.Pointer(&out[0])), true
	}
	buf, pinner, noCopy := residentNoCopyBytes(out)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: out, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.outViewPtr = ptr
	s.outViewLen = len(out)
	s.outView = buf
	s.outViewPinned = pinned
	return buf, (*byte)(unsafe.Pointer(&out[0])), true
}

func (s *perLayerInputsGPUScratch) buffers(tokenID int32, emb []byte) (metal.MTLBuffer, metal.MTLBuffer, *plGPUScratch, error) {
	if s == nil || s.token == nil || s.emb == nil || s.pl == nil {
		return nil, nil, nil, core.NewError("native.perLayerInputsGPUScratch.buffers: scratch is nil")
	}
	if len(emb) != s.dModel*bf16Size || len(s.token.bytes) != 4 || len(s.emb.bytes) != s.dModel*bf16Size {
		return nil, nil, nil, core.NewError("native.perLayerInputsGPUScratch.buffers: dimension mismatch")
	}
	*(*int32)(unsafe.Pointer(&s.token.bytes[0])) = tokenID
	embBuf, ok := s.embView.buffer(emb)
	if !ok {
		var err error
		embBuf, err = s.emb.copyBuffer(emb)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	return s.token.buf, embBuf, s.pl, nil
}

// encPerLayerInputsGPU encodes the WHOLE gemma4 PLE for one token into `enc` (no commit): the per-layer
// embedding is gathered+dequantised on the GPU from `tokenBuf` (the LM-head argmax output), the main
// embedding `embBuf` is projected → ×projScale → RMSNorm(rows) → +perLayer → ×combineScale. Output is
// scratch.out ([numLayers·pliDim] bf16). The token never round-trips to host — the seam the submit-ahead
// decode pipeline needs for PLE models (e2b/e4b). bf16 projection (e2b); 4-bit per-layer embedding.
func encPerLayerInputsGPU(enc metal.MTLComputeCommandEncoder, embedGatherPSO metal.MTLComputePipelineState,
	tokenBuf, embBuf metal.MTLBuffer,
	embedPacked, embedScales, embedBiases metal.MTLBuffer, embedPackedOff, embedScalesOff, embedBiasesOff uint,
	projW metal.MTLBuffer, projWOff uint, projNormW metal.MTLBuffer,
	sc *plGPUScratch, numLayers, pliDim, dModel, embGS, embBits int, embScale float32, eps float32) error {
	plDim := numLayers * pliDim
	// (1) per-layer embedding: gather token's [plDim] row × √pliDim on the GPU.
	encEmbedGatherQuant(enc, embedGatherPSO, tokenBuf, embedPacked, embedScales, embedBiases, sc.perLayer, embedPackedOff, embedScalesOff, embedBiasesOff, plDim, embGS, embBits, embScale)
	// (2-6) project the main embedding → ×projScale → RMSNorm(rows) → +perLayer → ×combineScale.
	// (projScale is baked into sc.projScaleBuf by newPLGPUScratch.)
	if err := encGemvBF16To(enc, projW, embBuf, sc.projected, projWOff, 0, plDim, dModel); err != nil {
		return err
	}
	if err := encScaleBF16(enc, sc.projected, sc.projScaleBuf, sc.scaled, 0, sc.projScaleBytes[:], plDim); err != nil {
		return err
	}
	if err := encRMSNormRowsBF16(enc, sc.scaled, projNormW, sc.projNormed, 0, 0, 0, numLayers, pliDim, eps); err != nil {
		return err
	}
	if err := encAddBF16(enc, sc.projNormed, sc.perLayer, sc.combined, plDim); err != nil {
		return err
	}
	return encScaleBF16(enc, sc.combined, sc.combineScaleBuf, sc.out, 0, sc.combineScaleBytes[:], plDim)
}

func encPerLayerInputsGPUObject(enc metal.MTLComputeCommandEncoderObject, embedGatherPSO metal.MTLComputePipelineState,
	tokenBuf, embBuf metal.MTLBuffer,
	embedPacked, embedScales, embedBiases metal.MTLBuffer, embedPackedOff, embedScalesOff, embedBiasesOff uint,
	projW metal.MTLBuffer, projWOff uint, projNormW metal.MTLBuffer,
	sc *plGPUScratch, numLayers, pliDim, dModel, embGS, embBits int, embScale float32, eps float32) error {
	plDim := numLayers * pliDim
	encEmbedGatherQuantObject(enc, embedGatherPSO, tokenBuf, embedPacked, embedScales, embedBiases, sc.perLayer, embedPackedOff, embedScalesOff, embedBiasesOff, plDim, embGS, embBits, embScale)
	if err := encGemvBF16ToObject(enc, projW, embBuf, sc.projected, projWOff, 0, plDim, dModel); err != nil {
		return err
	}
	if err := encScaleBF16Object(enc, sc.projected, sc.projScaleBuf, sc.scaled, 0, sc.projScaleBytes[:], plDim); err != nil {
		return err
	}
	if err := encRMSNormRowsBF16Object(enc, sc.scaled, projNormW, sc.projNormed, 0, 0, 0, numLayers, pliDim, eps); err != nil {
		return err
	}
	if err := encAddBF16Object(enc, sc.projNormed, sc.perLayer, sc.combined, plDim); err != nil {
		return err
	}
	return encScaleBF16Object(enc, sc.combined, sc.combineScaleBuf, sc.out, 0, sc.combineScaleBytes[:], plDim)
}

// nextInputsGPU computes one token's NEXT-step decode inputs — the main embedding (dModel) and the PLE
// tensor (numLayers·pliDim) — fully on the GPU via the session's resident weights, reading both back.
// The host-visible check that encNextInputsGPU matches s.embed + s.perLayerInput. ok=false when the
// session has no GPU PLE seam (non-e2b shape). Single-shot (own command buffer); the pipeline drives
// encNextInputsGPU directly into the ICB input buffers instead.
func (s *ArchSession) nextInputsGPU(tokenID int32) (emb, pli []byte, ok bool, err error) {
	if s.encNextInputsGPU == nil || s.plScratchNew == nil {
		return nil, nil, false, nil
	}
	dModel := s.arch.Hidden
	plDim := len(s.arch.Layer) * s.arch.PerLayerInputHidden
	withAutoreleasePool(func() {
		tokBuf := s.nextInputTokenBuffer(tokenID)
		embBuf := s.nextInputEmbBuffer(dModel)
		sc := s.nextInputPLScratchBuffer()
		emb = s.nextInputEmbReadback(dModel)
		pli = s.nextInputPLEReadback(plDim)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if err = s.encNextInputsGPU(enc, tokBuf, embBuf, sc); err != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if len(emb) > 0 && unsafe.Pointer(&emb[0]) != unsafe.Pointer(s.nextInputEmbPtr) {
			copy(emb, unsafe.Slice(s.nextInputEmbPtr, dModel*bf16Size))
		}
		if len(pli) > 0 && unsafe.Pointer(&pli[0]) != unsafe.Pointer(sc.outPtr) {
			copy(pli, unsafe.Slice(sc.outPtr, plDim*bf16Size))
		}
	})
	if err != nil {
		return nil, nil, false, err
	}
	return emb, pli, true, nil
}

// PerLayerInputsGPU is the standalone host entry over encPerLayerInputsGPU: computes one token's PLE
// tensor fully on the GPU (token id + main embedding in, [numLayers·pliDim] bf16 out). bf16 projection
// (e2b). Byte/cosine-tracks the host PerLayerInputs.
func PerLayerInputsGPU(tokenID int32, emb []byte, embedPacked, embedScales, embedBiases, projW, projNormW []byte, vocabPLI, numLayers, pliDim, dModel, embGS, embBits int, eps float32) ([]byte, error) {
	return perLayerInputsGPUInto(nil, tokenID, emb, embedPacked, embedScales, embedBiases, projW, projNormW, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, eps)
}

func perLayerInputsGPUInto(out []byte, tokenID int32, emb []byte, embedPacked, embedScales, embedBiases, projW, projNormW []byte, vocabPLI, numLayers, pliDim, dModel, embGS, embBits int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if embBits != 4 {
		return nil, core.NewError("native.PerLayerInputsGPU: per-layer embedding must be 4-bit")
	}
	plDim := numLayers * pliDim
	gpso, err := embedGatherPipeline()
	if err != nil {
		return nil, err
	}
	embScale := float32(math.Sqrt(float64(pliDim)))
	projScale := float32(1.0 / math.Sqrt(float64(dModel)))
	outBytes := plDim * bf16Size
	callerOut := cap(out) >= outBytes
	if !callerOut {
		out = make([]byte, outBytes)
	} else {
		out = out[:outBytes]
	}
	var ferr error
	withAutoreleasePool(func() {
		scratch, err := getPerLayerInputsGPUScratch(plDim, dModel, projScale)
		if err != nil {
			ferr = err
			return
		}
		defer putPerLayerInputsGPUScratch(scratch)
		tokBuf, embBuf, sc, err := scratch.buffers(tokenID, emb)
		if err != nil {
			ferr = err
			return
		}
		ePacked, eScales, eBiases := residentBytes(embedPacked), residentBytes(embedScales), residentBytes(embedBiases)
		projWBuf, projNormWBuf := residentBytes(projW), residentBytes(projNormW)
		scForCall := sc
		directOut := false
		if callerOut {
			outBuf, outPtr, ok := scratch.outputView(out)
			directOut = ok
			if ok {
				tmp := *sc
				tmp.out = outBuf
				tmp.outPtr = outPtr
				scForCall = &tmp
			}
		}
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if ferr = encPerLayerInputsGPU(enc, gpso, tokBuf, embBuf, ePacked, eScales, eBiases, 0, 0, 0, projWBuf, 0, projNormWBuf, scForCall, numLayers, pliDim, dModel, embGS, embBits, embScale, eps); ferr != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, unsafe.Slice(sc.outPtr, plDim*bf16Size))
		}
	})
	return out, ferr
}
