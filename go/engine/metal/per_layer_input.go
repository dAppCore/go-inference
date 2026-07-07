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

// gemma4PerLayerCombineScale is gemma4's 1/√2 factor that combines the two per-layer-input
// branches (the per-layer embedding and the projected main embedding).
const gemma4PerLayerCombineScale = 0.70710678118654752440

type perLayerInputGateScratch struct {
	dModel, pliDim                            int
	hNext, perLayer                           *pinnedNoCopyBytes
	hNextView, perLayerView                   cachedNoCopyBytesView
	gate, gelu, multiplied, projected, normed metal.MTLBuffer
	out                                       metal.MTLBuffer
	noCopyOutputView
}

type perLayerInputGateScratchKey struct {
	dModel, pliDim int
}

type cachedNoCopyBytesView struct {
	ptr           uintptr
	len           int
	buf           metal.MTLBuffer
	pinned        *pinnedNoCopyBytes
	candidatePtr  uintptr
	candidateLen  int
	candidateHits int
}

func (v *cachedNoCopyBytesView) Close() {
	if v == nil {
		return
	}
	v.closePinned()
	v.candidatePtr = 0
	v.candidateLen = 0
	v.candidateHits = 0
}

func (v *cachedNoCopyBytesView) closePinned() {
	if v.pinned != nil {
		v.pinned.Close()
	}
	v.ptr = 0
	v.len = 0
	v.buf = nil
	v.pinned = nil
}

func (v *cachedNoCopyBytesView) buffer(src []byte) (metal.MTLBuffer, bool) {
	return v.bufferAfterStable(src, 1)
}

func (v *cachedNoCopyBytesView) bufferAfterStable(src []byte, minHits int) (metal.MTLBuffer, bool) {
	if v == nil || len(src) == 0 {
		return nil, false
	}
	if minHits < 1 {
		minHits = 1
	}
	ptr := uintptr(unsafe.Pointer(&src[0]))
	if v.buf != nil && v.ptr == ptr && v.len == len(src) {
		return v.buf, true
	}
	if v.buf != nil {
		v.closePinned()
	}
	if buf, ok := registeredPinnedNoCopyBytes(src); ok {
		v.candidatePtr = ptr
		v.candidateLen = len(src)
		v.candidateHits = minHits
		v.ptr = ptr
		v.len = len(src)
		v.buf = buf
		v.pinned = nil
		return buf, true
	}
	if v.candidatePtr == ptr && v.candidateLen == len(src) {
		v.candidateHits++
	} else {
		v.candidatePtr = ptr
		v.candidateLen = len(src)
		v.candidateHits = 1
	}
	if v.candidateHits < minHits {
		return nil, false
	}
	buf, pinner, noCopy := residentNoCopyBytes(src)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: src, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	v.ptr = ptr
	v.len = len(src)
	v.buf = buf
	v.pinned = pinned
	return buf, true
}

var perLayerInputGateScratchPools sync.Map

func perLayerInputGateScratchPoolFor(dModel, pliDim int) *scratchLIFOPool[*perLayerInputGateScratch] {
	key := perLayerInputGateScratchKey{dModel: dModel, pliDim: pliDim}
	if v, ok := perLayerInputGateScratchPools.Load(key); ok {
		return v.(*scratchLIFOPool[*perLayerInputGateScratch])
	}
	pool := &scratchLIFOPool[*perLayerInputGateScratch]{}
	if v, loaded := perLayerInputGateScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*scratchLIFOPool[*perLayerInputGateScratch])
	}
	return pool
}

func getPerLayerInputGateScratch(dModel, pliDim int) *perLayerInputGateScratch {
	pool := perLayerInputGateScratchPoolFor(dModel, pliDim)
	if s := pool.Get(); s != nil {
		if s.dModel == dModel && s.pliDim == pliDim && s.gate != nil && s.out != nil {
			return s
		}
		s.Close()
	}
	return newPerLayerInputGateScratch(dModel, pliDim)
}

func newPerLayerInputGateScratch(dModel, pliDim int) *perLayerInputGateScratch {
	return &perLayerInputGateScratch{
		dModel:     dModel,
		pliDim:     pliDim,
		gate:       scratchBF16(pliDim),
		gelu:       scratchBF16(pliDim),
		multiplied: scratchBF16(pliDim),
		projected:  scratchBF16(dModel),
		normed:     scratchBF16(dModel),
		out:        scratchBF16(dModel),
	}
}

func (s *perLayerInputGateScratch) Close() {
	if s == nil {
		return
	}
	if s.hNext != nil {
		s.hNext.Close()
		s.hNext = nil
	}
	if s.perLayer != nil {
		s.perLayer.Close()
		s.perLayer = nil
	}
	s.hNextView.Close()
	s.perLayerView.Close()
	s.closeOutputView()
	s.gate, s.gelu, s.multiplied, s.projected, s.normed, s.out = nil, nil, nil, nil, nil, nil
	s.dModel, s.pliDim = 0, 0
}

func (s *perLayerInputGateScratch) inputBuffers(hNext, perLayerInput []byte) (metal.MTLBuffer, metal.MTLBuffer, error) {
	if s == nil {
		return nil, nil, core.NewError("native.perLayerInputGateScratch.inputBuffers: scratch is nil")
	}
	hLen, plLen := s.dModel*bf16Size, s.pliDim*bf16Size
	if len(hNext) != hLen || len(perLayerInput) != plLen {
		return nil, nil, core.NewError("native.perLayerInputGateScratch.inputBuffers: input size mismatch")
	}
	var err error
	hBuf, hNoCopy := s.hNextView.buffer(hNext)
	if !hNoCopy && s.hNext == nil {
		s.hNext, err = newPinnedNoCopyBytes(hLen)
		if err != nil {
			return nil, nil, err
		}
	}
	perLayerBuf, perLayerNoCopy := s.perLayerView.buffer(perLayerInput)
	if !perLayerNoCopy && s.perLayer == nil {
		s.perLayer, err = newPinnedNoCopyBytes(plLen)
		if err != nil {
			return nil, nil, err
		}
	}
	if !hNoCopy {
		hBuf, err = s.hNext.copyBuffer(hNext)
		if err != nil {
			return nil, nil, err
		}
	}
	if !perLayerNoCopy {
		perLayerBuf, err = s.perLayer.copyBuffer(perLayerInput)
		if err != nil {
			return nil, nil, err
		}
	}
	return hBuf, perLayerBuf, nil
}

func putPerLayerInputGateScratch(s *perLayerInputGateScratch) {
	if s != nil && s.dModel > 0 && s.pliDim > 0 && s.gate != nil && s.out != nil {
		perLayerInputGateScratchPoolFor(s.dModel, s.pliDim).Put(s)
	}
}

// PerLayerInputs computes gemma4's per-layer-input tensor for ONE token — the auxiliary
// embedding each layer's per-layer-input gate (PerLayerInputGateBF16) consumes, returned as
// [numLayers · pliDim] bf16 (numLayers contiguous rows of pliDim). Mirrors pkg/metal/model/
// gemma4 perLayerInputTensor op-for-op:
//
//	perLayer  = embed_tokens_per_layer[token] · √pliDim                        (4-bit gather + scale)
//	projected = rms( (per_layer_model_projection · hidden) · 1/√dModel, projNorm )  (per layer-row)
//	combined  = (projected + perLayer) · 1/√2
//
// Mixed weights, matching the checkpoint: the per-layer embedding is 4-bit (packed/scales/
// biases), the model projection + projection norm are bf16. hidden is the main token embedding
// (dModel bf16). projNormW is the PLAIN [pliDim] norm weight, applied per layer-row (rows =
// numLayers, axis = pliDim). Composed from the parity-proven ops.
func PerLayerInputs(
	embedPacked, embedScales, embedBiases []byte,
	projW, projScales, projBiases, projNormW []byte,
	tokenID int32, hidden []byte,
	vocabPLI, numLayers, pliDim, dModel, groupSize, bits, projGS, projBits int, eps float32, projView bufView,
	scratchArg ...*plHostScratch,
) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(hidden) != dModel*bf16Size {
		return nil, core.NewError("native.PerLayerInputs: hidden must be dModel bf16 bytes")
	}
	plDim := numLayers * pliDim
	// projScales present ⇒ the model projection is 4-bit (qat packs, e4b); its packed weight has a
	// different byte span, so only validate the bf16 span when the projection is dense (e2b).
	if len(projScales) == 0 && len(projW) != plDim*dModel*bf16Size {
		return nil, core.NewError("native.PerLayerInputs: bf16 projW must be (numLayers·pliDim)*dModel bf16 bytes")
	}
	if len(projNormW) != pliDim*bf16Size {
		return nil, core.NewError("native.PerLayerInputs: projNormW must be pliDim bf16 bytes")
	}
	embScale := float32(math.Sqrt(float64(pliDim)))
	projScale := float32(1.0 / math.Sqrt(float64(dModel)))
	var scratch *plHostScratch
	if len(scratchArg) > 0 {
		scratch = scratchArg[0]
	}
	borrowedScratch := false
	var err error
	if scratch == nil {
		scratch, err = getPLHostScratch(plDim, dModel, projScale)
		if err != nil {
			return nil, err
		}
		borrowedScratch = true
		defer putPLHostScratch(scratch)
	}

	// (1) per-layer embedding: gather the token's [numLayers·pliDim] row, × √pliDim. bf16 in regular
	// packs (e2b), 4-bit in qat packs (e4b) — dispatch on the .scales decision, exactly like the
	// projection below, so a bf16 model is a non-event (the shared loader already decided the format).
	var perLayer []byte
	if scratch != nil && scratch.perLayer != nil && scratch.plDim == plDim && len(scratch.perLayer.bytes) == plDim*bf16Size {
		perLayer = scratch.perLayer.bytes[:plDim*bf16Size]
		if len(embedScales) > 0 {
			_, err = embedTokenQuantInto(perLayer, embedPacked, embedScales, embedBiases, tokenID, vocabPLI, plDim, groupSize, bits, embScale)
		} else {
			_, err = embedTokenBF16Into(perLayer, embedPacked, tokenID, vocabPLI, plDim, embScale)
		}
		if err != nil {
			return nil, err
		}
	} else if len(embedScales) > 0 {
		var embs [][]byte
		if embs, err = EmbedTokensQuant(embedPacked, embedScales, embedBiases, []int32{tokenID}, vocabPLI, plDim, groupSize, bits, embScale); err != nil {
			return nil, err
		}
		perLayer = embs[0]
	} else {
		var embs [][]byte
		if embs, err = EmbedTokensBF16(embedPacked, []int32{tokenID}, vocabPLI, plDim, embScale); err != nil {
			return nil, err
		}
		perLayer = embs[0]
	}
	// (2) project the main embedding → [numLayers·pliDim], × 1/√dModel. The model projection is
	// bf16 in regular packs (e2b) and 4-bit in qat packs (e4b); dispatch on the presence of scales,
	// so a quantised projection is a non-event — the shared loader already made the .scales decision.
	// (2-6) run the whole projection chain as ONE command buffer (five GPU round-trips → one).
	// Byte-identical to the unbatched ops below for both dense bf16 and quant projection weights.
	if len(projScales) == 0 {
		projView = bf16WeightView(projW, projView)
		out, err := perLayerProjBatched(projView, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, scratch)
		if err != nil || !borrowedScratch {
			return out, err
		}
		return append([]byte(nil), out...), nil
	}
	if len(projScales) > 0 {
		out, err := perLayerProjQuantBatched(QuantWeight{Packed: projW, Scales: projScales, Biases: projBiases}, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, projGS, projBits, eps, scratch)
		if err != nil || !borrowedScratch {
			return out, err
		}
		return append([]byte(nil), out...), nil
	}
	return nil, core.NewError("native.PerLayerInputs: unreachable projection format")
}

func perLayerInputsResidentBuffer(
	embedPacked, embedScales, embedBiases []byte,
	projW, projNormW []byte,
	tokenID int32, hidden []byte,
	vocabPLI, numLayers, pliDim, dModel, groupSize, bits int, eps float32, projView bufView, scratch *plHostScratch,
) (metal.MTLBuffer, int, error) {
	return perLayerInputsResidentBufferCore(embedPacked, embedScales, embedBiases, projW, projNormW, tokenID, hidden, nil, vocabPLI, numLayers, pliDim, dModel, groupSize, bits, eps, projView, scratch)
}

func perLayerInputsResidentMetalBuffer(
	embedPacked, embedScales, embedBiases []byte,
	projW, projNormW []byte,
	tokenID int32, hiddenBuf metal.MTLBuffer,
	vocabPLI, numLayers, pliDim, dModel, groupSize, bits int, eps float32, projView bufView, scratch *plHostScratch,
) (metal.MTLBuffer, int, error) {
	return perLayerInputsResidentBufferCore(embedPacked, embedScales, embedBiases, projW, projNormW, tokenID, nil, hiddenBuf, vocabPLI, numLayers, pliDim, dModel, groupSize, bits, eps, projView, scratch)
}

func perLayerInputsQuantResidentBuffer(
	embedPacked, embedScales, embedBiases []byte,
	proj QuantWeight, projNormW []byte,
	tokenID int32, hidden []byte,
	vocabPLI, numLayers, pliDim, dModel, groupSize, bits, projGroupSize, projBits int, eps float32, scratch *plHostScratch,
) (metal.MTLBuffer, int, error) {
	return perLayerInputsQuantResidentBufferCore(embedPacked, embedScales, embedBiases, proj, projNormW, tokenID, hidden, nil, vocabPLI, numLayers, pliDim, dModel, groupSize, bits, projGroupSize, projBits, eps, scratch)
}

func perLayerInputsQuantResidentMetalBuffer(
	embedPacked, embedScales, embedBiases []byte,
	proj QuantWeight, projNormW []byte,
	tokenID int32, hiddenBuf metal.MTLBuffer,
	vocabPLI, numLayers, pliDim, dModel, groupSize, bits, projGroupSize, projBits int, eps float32, scratch *plHostScratch,
) (metal.MTLBuffer, int, error) {
	return perLayerInputsQuantResidentBufferCore(embedPacked, embedScales, embedBiases, proj, projNormW, tokenID, nil, hiddenBuf, vocabPLI, numLayers, pliDim, dModel, groupSize, bits, projGroupSize, projBits, eps, scratch)
}

func perLayerInputsQuantResidentBufferCore(
	embedPacked, embedScales, embedBiases []byte,
	proj QuantWeight, projNormW []byte,
	tokenID int32, hidden []byte, hiddenBuf metal.MTLBuffer,
	vocabPLI, numLayers, pliDim, dModel, groupSize, bits, projGroupSize, projBits int, eps float32, scratch *plHostScratch,
) (metal.MTLBuffer, int, error) {
	if err := ensureInit(); err != nil {
		return nil, 0, err
	}
	if scratch == nil || scratch.perLayer == nil {
		return nil, 0, core.NewError("native.perLayerInputsQuantResidentBuffer: scratch is required")
	}
	if hiddenBuf == nil && len(hidden) != dModel*bf16Size {
		return nil, 0, core.NewError("native.perLayerInputsQuantResidentBuffer: hidden must be dModel bf16 bytes")
	}
	plDim := numLayers * pliDim
	if plDim <= 0 || dModel <= 0 {
		return nil, 0, core.NewError("native.perLayerInputsQuantResidentBuffer: invalid dimensions")
	}
	if len(projNormW) != pliDim*bf16Size {
		return nil, 0, core.NewError("native.perLayerInputsQuantResidentBuffer: projNormW must be pliDim bf16 bytes")
	}
	if scratch.plDim != plDim || scratch.dModel != dModel || len(scratch.perLayer.bytes) != plDim*bf16Size {
		return nil, 0, core.NewError("native.perLayerInputsQuantResidentBuffer: scratch dimension mismatch")
	}

	perLayer := scratch.perLayer.bytes[:plDim*bf16Size]
	embScale := float32(math.Sqrt(float64(pliDim)))
	var err error
	if len(embedScales) > 0 {
		_, err = embedTokenQuantInto(perLayer, embedPacked, embedScales, embedBiases, tokenID, vocabPLI, plDim, groupSize, bits, embScale)
	} else {
		_, err = embedTokenBF16Into(perLayer, embedPacked, tokenID, vocabPLI, plDim, embScale)
	}
	if err != nil {
		return nil, 0, err
	}

	projScale := float32(1.0 / math.Sqrt(float64(dModel)))
	var buf metal.MTLBuffer
	if hiddenBuf != nil {
		buf, err = perLayerProjQuantBatchedResidentBuffer(proj, hiddenBuf, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, projGroupSize, projBits, eps, scratch)
	} else {
		buf, err = perLayerProjQuantBatchedResident(proj, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, projGroupSize, projBits, eps, scratch)
	}
	if err != nil {
		return nil, 0, err
	}
	return buf, plDim * bf16Size, nil
}

func perLayerInputsResidentBufferCore(
	embedPacked, embedScales, embedBiases []byte,
	projW, projNormW []byte,
	tokenID int32, hidden []byte, hiddenBuf metal.MTLBuffer,
	vocabPLI, numLayers, pliDim, dModel, groupSize, bits int, eps float32, projView bufView, scratch *plHostScratch,
) (metal.MTLBuffer, int, error) {
	if err := ensureInit(); err != nil {
		return nil, 0, err
	}
	if scratch == nil || scratch.perLayer == nil {
		return nil, 0, core.NewError("native.perLayerInputsResidentBuffer: scratch is required")
	}
	if hiddenBuf == nil && len(hidden) != dModel*bf16Size {
		return nil, 0, core.NewError("native.perLayerInputsResidentBuffer: hidden must be dModel bf16 bytes")
	}
	plDim := numLayers * pliDim
	if plDim <= 0 || dModel <= 0 {
		return nil, 0, core.NewError("native.perLayerInputsResidentBuffer: invalid dimensions")
	}
	if len(projW) != plDim*dModel*bf16Size {
		return nil, 0, core.NewError("native.perLayerInputsResidentBuffer: bf16 projW must be (numLayers·pliDim)*dModel bf16 bytes")
	}
	if len(projNormW) != pliDim*bf16Size {
		return nil, 0, core.NewError("native.perLayerInputsResidentBuffer: projNormW must be pliDim bf16 bytes")
	}
	if projView.buf == nil {
		return nil, 0, core.NewError("native.perLayerInputsResidentBuffer: resident projection buffer is nil")
	}
	if scratch.plDim != plDim || scratch.dModel != dModel || len(scratch.perLayer.bytes) != plDim*bf16Size {
		return nil, 0, core.NewError("native.perLayerInputsResidentBuffer: scratch dimension mismatch")
	}

	perLayer := scratch.perLayer.bytes[:plDim*bf16Size]
	embScale := float32(math.Sqrt(float64(pliDim)))
	var err error
	if len(embedScales) > 0 {
		_, err = embedTokenQuantInto(perLayer, embedPacked, embedScales, embedBiases, tokenID, vocabPLI, plDim, groupSize, bits, embScale)
	} else {
		_, err = embedTokenBF16Into(perLayer, embedPacked, tokenID, vocabPLI, plDim, embScale)
	}
	if err != nil {
		return nil, 0, err
	}

	projScale := float32(1.0 / math.Sqrt(float64(dModel)))
	var buf metal.MTLBuffer
	if hiddenBuf != nil {
		buf, err = perLayerProjBatchedResidentBuffer(projView, hiddenBuf, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, scratch)
	} else {
		buf, err = perLayerProjBatchedResident(projView, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, scratch)
	}
	if err != nil {
		return nil, 0, err
	}
	return buf, plDim * bf16Size, nil
}

// PerLayerInputGateBF16 applies the gemma4 per-layer-input gate to a layer's output
// hNext (dModel) and returns the gated result. Mirrors pkg/metal/model/gemma4
// decoder_layer.go's per-layer-input block op-for-op:
//
//	gate       = WGate · hNext            (dModel → pliDim)
//	multiplied = gelu(gate) · perLayerInput   (pliDim, the SwiGLU gate-mul)
//	projected  = WProj · multiplied       (pliDim → dModel)
//	hNext      = hNext + rms(projected, PostPerLayerInputNorm)
//
// perLayerInput is this layer's per-token, per-layer input (pliDim bf16) — the slice
// of the per-layer embedding the layer consumes. PostPerLayerInputNorm is the PLAIN
// norm weight (NOT RootSize-scaled like the router's — metal caches this one as a
// plain Copy). Bias-free, matching the rest of the gemma4 native path (q/k/v/o/
// gate/up/down are all bias-free); a checkpoint with per-layer biases is a
// cross-cutting load-time concern. Composed from the parity-proven bf16 ops.
func PerLayerInputGateBF16(hNext, gateW, perLayerInput, projW, postNormW []byte, dModel, pliDim int, eps float32) ([]byte, error) {
	return perLayerInputGateBF16Into(nil, hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, eps)
}

func perLayerInputGateBF16Into(out []byte, hNext, gateW, perLayerInput, projW, postNormW []byte, dModel, pliDim int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(hNext) != dModel*bf16Size {
		return nil, core.NewError("native.PerLayerInputGateBF16: hNext must be dModel bf16 bytes")
	}
	if len(perLayerInput) != pliDim*bf16Size {
		return nil, core.NewError("native.PerLayerInputGateBF16: perLayerInput must be pliDim bf16 bytes")
	}
	if len(gateW) != pliDim*dModel*bf16Size {
		return nil, core.NewError("native.PerLayerInputGateBF16: gateW must be pliDim*dModel bf16 bytes")
	}
	if len(projW) != dModel*pliDim*bf16Size {
		return nil, core.NewError("native.PerLayerInputGateBF16: projW must be dModel*pliDim bf16 bytes")
	}
	if len(postNormW) != dModel*bf16Size {
		return nil, core.NewError("native.PerLayerInputGateBF16: postNormW must be dModel bf16 bytes")
	}
	if dModel == 0 {
		return []byte{}, nil
	}
	outLen := dModel * bf16Size
	if pliDim == 0 {
		if cap(out) < outLen {
			return append([]byte(nil), hNext...), nil
		}
		out = out[:outLen]
		copy(out, hNext)
		return out, nil
	}

	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	scratch := getPerLayerInputGateScratch(dModel, pliDim)
	defer putPerLayerInputGateScratch(scratch)
	hBuf, perLayerBuf, err := scratch.inputBuffers(hNext, perLayerInput)
	if err != nil {
		return nil, err
	}
	if callerOut {
		err = perLayerInputGateBF16EncodedInto(
			scratch, out, hBuf, residentBytes(gateW), perLayerBuf, residentBytes(projW), residentBytes(postNormW),
			dModel, pliDim, eps,
		)
	} else {
		err = perLayerInputGateBF16Encoded(
			scratch, out, hBuf, residentBytes(gateW), perLayerBuf, residentBytes(projW), residentBytes(postNormW),
			dModel, pliDim, eps,
		)
	}
	if err != nil {
		return nil, err
	}
	return out, nil
}

// PerLayerInputGateQuant is PerLayerInputGateBF16 for a 4-bit checkpoint: the gate and
// projection are affine-quantised (per_layer_input_gate / per_layer_projection are 4-bit in the
// served E2B/E4B packs), the post-norm stays bf16. gate is the [pliDim × dModel] quant weight,
// proj the [dModel × pliDim] quant weight; the chain matches PerLayerInputGateBF16 with QMVBF16
// in place of the two bf16 matvecs. The quant gate/projection weights are fixed per
// layer and stay resident across tokens. perLayerInput is this layer's pliDim slice
// of the PerLayerInputs tensor.
func PerLayerInputGateQuant(hNext []byte, gate QuantWeight, perLayerInput []byte, proj QuantWeight, postNormW []byte, dModel, pliDim, groupSize, bits int, eps float32) ([]byte, error) {
	return perLayerInputGateQuantInto(nil, hNext, gate, perLayerInput, proj, postNormW, dModel, pliDim, groupSize, bits, eps)
}

func perLayerInputGateQuantInto(out []byte, hNext []byte, gate QuantWeight, perLayerInput []byte, proj QuantWeight, postNormW []byte, dModel, pliDim, groupSize, bits int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(hNext) != dModel*bf16Size {
		return nil, core.NewError("native.PerLayerInputGateQuant: hNext must be dModel bf16 bytes")
	}
	if len(perLayerInput) != pliDim*bf16Size {
		return nil, core.NewError("native.PerLayerInputGateQuant: perLayerInput must be pliDim bf16 bytes")
	}
	if len(postNormW) != dModel*bf16Size {
		return nil, core.NewError("native.PerLayerInputGateQuant: postNormW must be dModel bf16 bytes")
	}
	if dModel == 0 {
		return []byte{}, nil
	}
	outLen := dModel * bf16Size
	if pliDim == 0 {
		if cap(out) < outLen {
			return append([]byte(nil), hNext...), nil
		}
		out = out[:outLen]
		copy(out, hNext)
		return out, nil
	}
	gateGroupSize, gateBits, err := validatePerLayerInputGateQuantWeight("gate", gate, pliDim, dModel, groupSize, bits)
	if err != nil {
		return nil, err
	}
	projGroupSize, projBits, err := validatePerLayerInputGateQuantWeight("projection", proj, dModel, pliDim, groupSize, bits)
	if err != nil {
		return nil, err
	}

	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	scratch := getPerLayerInputGateScratch(dModel, pliDim)
	defer putPerLayerInputGateScratch(scratch)
	hBuf, perLayerBuf, err := scratch.inputBuffers(hNext, perLayerInput)
	if err != nil {
		return nil, err
	}
	gatePacked, gateScales, gateBiases := quantWeightViews(gate)
	projPacked, projScales, projBiases := quantWeightViews(proj)
	if callerOut {
		err = perLayerInputGateQuantEncodedInto(
			scratch, out, hBuf, gatePacked, gateScales, gateBiases, perLayerBuf, projPacked, projScales, projBiases,
			residentBytes(postNormW), dModel, pliDim, gateGroupSize, gateBits, projGroupSize, projBits, eps,
		)
	} else {
		err = perLayerInputGateQuantEncoded(
			scratch, out, hBuf, gatePacked, gateScales, gateBiases, perLayerBuf, projPacked, projScales, projBiases,
			residentBytes(postNormW), dModel, pliDim, gateGroupSize, gateBits, projGroupSize, projBits, eps,
		)
	}
	if err != nil {
		return nil, err
	}
	return out, nil
}

func validatePerLayerInputGateQuantWeight(name string, w QuantWeight, outDim, inDim, groupSize, bits int) (int, int, error) {
	groupSize, bits = quantWeightGeometryForShape(w, outDim, inDim, groupSize, bits)
	if groupSize <= 0 || bits <= 0 || inDim%groupSize != 0 {
		return 0, 0, core.NewError("native.PerLayerInputGateQuant: invalid " + name + " quant geometry")
	}
	wantPacked := outDim * inDim * bits / 8
	wantSB := outDim * (inDim / groupSize) * bf16Size
	if len(w.Packed) != wantPacked || len(w.Scales) != wantSB || len(w.Biases) != wantSB {
		return 0, 0, core.NewError("native.PerLayerInputGateQuant: " + name + " quant weight size mismatch")
	}
	return groupSize, bits, nil
}

func perLayerInputGateBF16Encoded(scratch *perLayerInputGateScratch, out []byte, hBuf, gateWBuf, perLayerBuf, projWBuf, postNormWBuf metal.MTLBuffer, dModel, pliDim int, eps float32) error {
	var encErr error
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		encErr = encPerLayerInputGateBF16Scratch(enc, scratch, hBuf, gateWBuf, perLayerBuf, projWBuf, postNormWBuf, scratch.out, 0, dModel, pliDim, eps)
		endEncodingFast(enc)
		if encErr != nil {
			return
		}
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(out, unsafe.Slice((*byte)(scratch.out.Contents()), len(out)))
	})
	return encErr
}

func perLayerInputGateBF16EncodedInto(scratch *perLayerInputGateScratch, out []byte, hBuf, gateWBuf, perLayerBuf, projWBuf, postNormWBuf metal.MTLBuffer, dModel, pliDim int, eps float32) error {
	var encErr error
	withAutoreleasePool(func() {
		outBuf := scratch.out
		directOut := false
		if tmp, ok := scratch.outputView(out); ok {
			outBuf = tmp
			directOut = true
		}
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		encErr = encPerLayerInputGateBF16Scratch(enc, scratch, hBuf, gateWBuf, perLayerBuf, projWBuf, postNormWBuf, outBuf, 0, dModel, pliDim, eps)
		endEncodingFast(enc)
		if encErr != nil {
			return
		}
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, unsafe.Slice((*byte)(scratch.out.Contents()), len(out)))
		}
	})
	return encErr
}

func perLayerInputGateQuantEncoded(
	scratch *perLayerInputGateScratch,
	out []byte,
	hBuf metal.MTLBuffer,
	gatePacked, gateScales, gateBiases bufView,
	perLayerBuf metal.MTLBuffer,
	projPacked, projScales, projBiases bufView,
	postNormWBuf metal.MTLBuffer,
	dModel, pliDim, gateGroupSize, gateBits, projGroupSize, projBits int,
	eps float32,
) error {
	var encErr error
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		encErr = encPerLayerInputGateQuantScratch(enc, scratch, hBuf, gatePacked, gateScales, gateBiases, perLayerBuf, projPacked, projScales, projBiases, postNormWBuf, scratch.out, 0, dModel, pliDim, gateGroupSize, gateBits, projGroupSize, projBits, eps)
		endEncodingFast(enc)
		if encErr != nil {
			return
		}
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(out, unsafe.Slice((*byte)(scratch.out.Contents()), len(out)))
	})
	return encErr
}

func perLayerInputGateQuantEncodedInto(
	scratch *perLayerInputGateScratch,
	out []byte,
	hBuf metal.MTLBuffer,
	gatePacked, gateScales, gateBiases bufView,
	perLayerBuf metal.MTLBuffer,
	projPacked, projScales, projBiases bufView,
	postNormWBuf metal.MTLBuffer,
	dModel, pliDim, gateGroupSize, gateBits, projGroupSize, projBits int,
	eps float32,
) error {
	var encErr error
	withAutoreleasePool(func() {
		outBuf := scratch.out
		directOut := false
		if tmp, ok := scratch.outputView(out); ok {
			outBuf = tmp
			directOut = true
		}
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		encErr = encPerLayerInputGateQuantScratch(enc, scratch, hBuf, gatePacked, gateScales, gateBiases, perLayerBuf, projPacked, projScales, projBiases, postNormWBuf, outBuf, 0, dModel, pliDim, gateGroupSize, gateBits, projGroupSize, projBits, eps)
		endEncodingFast(enc)
		if encErr != nil {
			return
		}
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, unsafe.Slice((*byte)(scratch.out.Contents()), len(out)))
		}
	})
	return encErr
}

func encPerLayerInputGateBF16Scratch(enc metal.MTLComputeCommandEncoder, scratch *perLayerInputGateScratch, hBuf, gateWBuf, perLayerBuf, projWBuf, postNormWBuf, outBuf metal.MTLBuffer, perLayerOff uint, dModel, pliDim int, eps float32) error {
	return encPerLayerInputGateBF16ScratchAt(enc, scratch, hBuf, 0, gateWBuf, perLayerBuf, projWBuf, postNormWBuf, outBuf, 0, perLayerOff, dModel, pliDim, eps)
}

// encPerLayerInputGateBF16ScratchAt is encPerLayerInputGateBF16Scratch with the layer hidden
// bound at hOff and the output written at outOff — the batched dense prefill's rows live at
// byte offsets inside shared K-row buffers, and each row applies the gate in place at its own
// offset. The scratch is shared across rows within one command buffer: Metal's hazard tracking
// on the scratch buffers serialises the gate chain row-by-row, preserving the sequential
// byte-identity contract.
func encPerLayerInputGateBF16ScratchAt(enc metal.MTLComputeCommandEncoder, scratch *perLayerInputGateScratch, hBuf metal.MTLBuffer, hOff uint, gateWBuf, perLayerBuf, projWBuf, postNormWBuf, outBuf metal.MTLBuffer, outOff, perLayerOff uint, dModel, pliDim int, eps float32) error {
	if scratch == nil || scratch.dModel != dModel || scratch.pliDim != pliDim {
		return core.NewError("native.encPerLayerInputGateBF16Scratch: scratch dimension mismatch")
	}
	if err := encGemvBF16VecAt(enc, gateWBuf, hBuf, scratch.gate, 0, hOff, 0, pliDim, dModel); err != nil {
		return err
	}
	if err := encPerLayerGeluGateMulBF16(enc, scratch.gate, perLayerBuf, scratch.gelu, scratch.multiplied, 0, perLayerOff, 0, pliDim); err != nil {
		return err
	}
	if err := encGemvBF16To(enc, projWBuf, scratch.multiplied, scratch.projected, 0, 0, dModel, pliDim); err != nil {
		return err
	}
	if err := encRMSNormBF16(enc, scratch.projected, postNormWBuf, scratch.normed, 0, dModel, eps); err != nil {
		return err
	}
	return encAddBF16To(enc, hBuf, scratch.normed, outBuf, hOff, 0, outOff, dModel)
}

func encPerLayerInputGateQuantScratch(
	enc metal.MTLComputeCommandEncoder,
	scratch *perLayerInputGateScratch,
	hBuf metal.MTLBuffer,
	gatePacked, gateScales, gateBiases bufView,
	perLayerBuf metal.MTLBuffer,
	projPacked, projScales, projBiases bufView,
	postNormWBuf, outBuf metal.MTLBuffer,
	perLayerOff uint,
	dModel, pliDim, gateGroupSize, gateBits, projGroupSize, projBits int,
	eps float32,
) error {
	return encPerLayerInputGateQuantScratchAt(enc, scratch, hBuf, 0, gatePacked, gateScales, gateBiases, perLayerBuf, projPacked, projScales, projBiases, postNormWBuf, outBuf, 0, perLayerOff, dModel, pliDim, gateGroupSize, gateBits, projGroupSize, projBits, eps)
}

// encPerLayerInputGateQuantScratchAt is encPerLayerInputGateQuantScratch with the layer hidden
// bound at hOff and the output written at outOff — the quant sibling of
// encPerLayerInputGateBF16ScratchAt for the batched dense forward, whose rows live at byte
// offsets inside shared K-row buffers. The scratch is shared across rows within one command
// buffer: Metal's hazard tracking on the scratch buffers serialises the gate chain row-by-row,
// preserving the sequential byte-identity contract.
func encPerLayerInputGateQuantScratchAt(
	enc metal.MTLComputeCommandEncoder,
	scratch *perLayerInputGateScratch,
	hBuf metal.MTLBuffer,
	hOff uint,
	gatePacked, gateScales, gateBiases bufView,
	perLayerBuf metal.MTLBuffer,
	projPacked, projScales, projBiases bufView,
	postNormWBuf, outBuf metal.MTLBuffer,
	outOff, perLayerOff uint,
	dModel, pliDim, gateGroupSize, gateBits, projGroupSize, projBits int,
	eps float32,
) error {
	if scratch == nil || scratch.dModel != dModel || scratch.pliDim != pliDim {
		return core.NewError("native.encPerLayerInputGateQuantScratch: scratch dimension mismatch")
	}
	if err := encQMVBF16At(enc, gatePacked.buf, gateScales.buf, gateBiases.buf, hBuf, scratch.gate, gatePacked.off, gateScales.off, gateBiases.off, hOff, 0, pliDim, dModel, gateGroupSize, gateBits); err != nil {
		return err
	}
	if err := encPerLayerGeluGateMulBF16(enc, scratch.gate, perLayerBuf, scratch.gelu, scratch.multiplied, 0, perLayerOff, 0, pliDim); err != nil {
		return err
	}
	if err := encQMVBF16(enc, projPacked.buf, projScales.buf, projBiases.buf, scratch.multiplied, scratch.projected, projPacked.off, projScales.off, projBiases.off, 0, dModel, pliDim, projGroupSize, projBits); err != nil {
		return err
	}
	if err := encRMSNormBF16(enc, scratch.projected, postNormWBuf, scratch.normed, 0, dModel, eps); err != nil {
		return err
	}
	return encAddBF16To(enc, hBuf, scratch.normed, outBuf, hOff, 0, outOff, dModel)
}

func encPerLayerGeluGateMulBF16(enc metal.MTLComputeCommandEncoder, gate, up, gelu, out metal.MTLBuffer, gateOff, upOff, outOff uint, n int) error {
	if gpuHasGeluKernel() {
		return encGeluGateMulFusedTo(enc, gate, up, out, gateOff, upOff, outOff, n)
	}
	if err := encGeluBF16Composed(enc, gate, gelu, n); err != nil {
		return err
	}
	return encMulBF16To(enc, gelu, up, out, 0, upOff, outOff, n)
}
