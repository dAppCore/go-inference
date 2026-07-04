// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// DecodeForward runs a real multi-layer, multi-token decode forward on the no-cgo
// path: each token flows through every layer (residual stream layer→layer), each
// layer APPENDS its K/V to its OWN growing cache at the token's position, and the
// whole N-layer stack for a token is submitted in ONE command buffer + commit
// (how a real decode step submits). It is DecodeStepKV (the parity-proven real
// layer) wired into the autoregressive loop with resident per-layer caches and
// shared scratch — no per-token/per-layer buffer churn, so the per-token cost is
// the encode + the growing-window GPU work, nothing else.
//
// inputs are the T token hidden vectors (each dModel bf16) — the embedding/lm_head
// /sampler are separate concerns (a real model load, Snider's call); this exercises
// the transformer stack + KV growth. Returns the T per-token output vectors. With
// the same weights/inputs it equals stepping DecodeStepKV token-by-token,
// layer-by-layer (gated byte-for-byte in the tests). All raw bf16.

// DecodeLayerWeights is one decode layer's weights (raw bf16 bytes): attention
// norm, Q/K/V/O projections, MLP norm, gate/up/down. wQ is (nHeads·headDim ×
// dModel), wK/wV are (nKVHeads·headDim × dModel), wO is (dModel × nHeads·headDim),
// wGate/wUp are (dFF × dModel), wDown is (dModel × dFF).
type DecodeLayerWeights struct {
	AttnNormW, WQ, WK, WV, WO   []byte
	MLPNormW, WGate, WUp, WDown []byte
	// MoE, when non-nil, replaces the dense MLP half with the gemma4 dual-branch MoE
	// feed-forward (MoEBlockBF16) for this layer. The dense MLPNormW/WGate/WUp/WDown
	// are then unused (the local MLP lives in MoE.WGate/WUp/WDown). Only honoured by
	// the arch executor (DecodeForwardArch) when the layer's spec.MoE is set.
	MoE *MoELayerWeights
	// gemma4 norms the loader populates but the decode does NOT consume yet: QK-norm
	// (per-head RMSNorm on Q/K before RoPE), post-attention norm, post-feed-forward
	// norm. The native dense decode currently does pre-attn + pre-FF only; wiring these
	// four into encAttnHalfKV/encMLPHalfBF16 is the "gemma4 norm reconciliation" slice.
	// nil when the checkpoint omits them. (MLPNormW is the pre-feed-forward norm.)
	QNormW, KNormW, PostAttnNormW, PostFFNormW []byte
	// LayerScalarW is gemma4's per-layer output scalar (shape [1] bf16): the layer's final
	// hidden is multiplied by it before the next layer (applied by the arch executor). nil
	// when the checkpoint omits it.
	LayerScalarW []byte
	// gemma4 per-layer-input tower (E2B/E4B), bf16: the per-layer-input gate + projection and the
	// post-per-layer-input norm, applied host-side by PerLayerInputGateBF16 (the bf16 sibling of
	// the quant path). nil when the model has no PLE tower.
	PerLayerGate, PerLayerProjection, PostPerLayerInputNormW []byte
	// DFF is the per-layer MatFormer FFN width (E2B/E4B vary it, 6144/12288); 0 ⇒ the arch default.
	// The bf16 decode reads it so the MLP projector matches each layer's actual gate/up/down width.
	DFF int
}

type decodeForwardStepScratch struct {
	hBuf, xA, xB metal.MTLBuffer
	offBuf       metal.MTLBuffer
	offPtr       *int32
	hBufPtr      *byte
	xAPtr, xBPtr *byte
	dModel       int
}

func newDecodeForwardStepScratch(dModel int) decodeForwardStepScratch {
	off := int32(0)
	hBuf := scratchBF16(dModel)
	xA, xB := scratchBF16(dModel), scratchBF16(dModel)
	offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
	return decodeForwardStepScratch{
		hBuf:    hBuf,
		xA:      xA,
		xB:      xB,
		offBuf:  offBuf,
		offPtr:  (*int32)(offBuf.Contents()),
		hBufPtr: (*byte)(hBuf.Contents()),
		xAPtr:   (*byte)(xA.Contents()),
		xBPtr:   (*byte)(xB.Contents()),
		dModel:  dModel,
	}
}

func (s *decodeForwardStepScratch) bufferPtr(buf metal.MTLBuffer) *byte {
	if s == nil || buf == nil {
		return nil
	}
	switch buf {
	case s.hBuf:
		if s.hBufPtr != nil {
			return s.hBufPtr
		}
	case s.xA:
		if s.xAPtr != nil {
			return s.xAPtr
		}
	case s.xB:
		if s.xBPtr != nil {
			return s.xBPtr
		}
	}
	return (*byte)(buf.Contents())
}

func (s *decodeForwardStepScratch) bufferBytes(buf metal.MTLBuffer) []byte {
	return unsafe.Slice(s.bufferPtr(buf), s.dModel*bf16Size)
}

func (s *decodeForwardStepScratch) seed(pos int, input []byte) {
	*s.offPtr = int32(pos)
	copy(s.bufferBytes(s.xA), input)
}

func (s *decodeForwardStepScratch) copyBuffer(dst []byte, src metal.MTLBuffer) {
	copy(dst, s.bufferBytes(src))
}

type decodeForwardLayerBufs struct {
	anw, wq, wk, wv, wo, mnw, wg, wu, wd metal.MTLBuffer
	pan, pfn                             metal.MTLBuffer
	qn, kn                               metal.MTLBuffer
	kCache, vCache                       metal.MTLBuffer
}

type decodeForwardLayerScratch struct {
	lb      []decodeForwardLayerBufs
	projs   []bf16Projector
	kCaches []metal.MTLBuffer
	vCaches []metal.MTLBuffer
	kBytes  []uint
	vBytes  []uint
}

var decodeForwardLayerScratchPool sync.Pool

func newDecodeForwardLayerScratch(nLayers int) *decodeForwardLayerScratch {
	return &decodeForwardLayerScratch{
		lb:      make([]decodeForwardLayerBufs, nLayers),
		projs:   make([]bf16Projector, nLayers),
		kCaches: make([]metal.MTLBuffer, nLayers),
		vCaches: make([]metal.MTLBuffer, nLayers),
		kBytes:  make([]uint, nLayers),
		vBytes:  make([]uint, nLayers),
	}
}

func (s *decodeForwardLayerScratch) fits(nLayers int) bool {
	return s != nil &&
		cap(s.lb) >= nLayers && cap(s.projs) >= nLayers &&
		cap(s.kCaches) >= nLayers && cap(s.vCaches) >= nLayers &&
		cap(s.kBytes) >= nLayers && cap(s.vBytes) >= nLayers
}

func (s *decodeForwardLayerScratch) reset(nLayers int) *decodeForwardLayerScratch {
	clear(s.lb)
	clear(s.projs)
	s.lb = s.lb[:nLayers]
	s.projs = s.projs[:nLayers]
	s.kCaches = s.kCaches[:nLayers]
	s.vCaches = s.vCaches[:nLayers]
	s.kBytes = s.kBytes[:nLayers]
	s.vBytes = s.vBytes[:nLayers]
	return s
}

func (s *decodeForwardLayerScratch) kvCache(li int, cacheBytes uint) (metal.MTLBuffer, metal.MTLBuffer) {
	if s.kCaches[li] == nil || s.kBytes[li] != cacheBytes {
		s.kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
		s.kBytes[li] = cacheBytes
	}
	if s.vCaches[li] == nil || s.vBytes[li] != cacheBytes {
		s.vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
		s.vBytes[li] = cacheBytes
	}
	return s.kCaches[li], s.vCaches[li]
}

func getDecodeForwardLayerScratch(nLayers int) *decodeForwardLayerScratch {
	if v := decodeForwardLayerScratchPool.Get(); v != nil {
		if s, ok := v.(*decodeForwardLayerScratch); ok && s.fits(nLayers) {
			return s.reset(nLayers)
		}
	}
	return newDecodeForwardLayerScratch(nLayers)
}

func putDecodeForwardLayerScratch(s *decodeForwardLayerScratch) {
	if s != nil {
		decodeForwardLayerScratchPool.Put(s.reset(0))
	}
}

type decodeForwardCoreScratch struct {
	dModel, qDim, kvDim, nHeads, dFF int
	asc                              attnScratch
	msc                              mlpScratch
	step                             decodeForwardStepScratch
}

var decodeForwardCoreScratchPool sync.Pool

func newDecodeForwardCoreScratch(dModel, qDim, kvDim, nHeads, dFF int) *decodeForwardCoreScratch {
	return &decodeForwardCoreScratch{
		dModel: dModel, qDim: qDim, kvDim: kvDim, nHeads: nHeads, dFF: dFF,
		asc:  newAttnScratch(dModel, qDim, kvDim, nHeads, 0),
		msc:  newMLPScratch(dModel, dFF),
		step: newDecodeForwardStepScratch(dModel),
	}
}

func (s *decodeForwardCoreScratch) fits(dModel, qDim, kvDim, nHeads, dFF int) bool {
	return s != nil &&
		s.dModel == dModel && s.qDim == qDim && s.kvDim == kvDim && s.nHeads == nHeads && s.dFF == dFF &&
		s.asc.normed != nil && s.asc.q != nil && s.asc.qr != nil && s.asc.kProj != nil && s.asc.attn != nil && s.asc.attnOut != nil &&
		s.msc.mlpNormed != nil && s.msc.gate != nil && s.msc.up != nil && s.msc.gated != nil && s.msc.down != nil &&
		s.step.hBuf != nil && s.step.xA != nil && s.step.xB != nil && s.step.offBuf != nil &&
		s.step.offPtr != nil && s.step.hBufPtr != nil && s.step.xAPtr != nil && s.step.xBPtr != nil
}

func (s *decodeForwardCoreScratch) reset() *decodeForwardCoreScratch {
	if s != nil && s.step.offPtr != nil {
		*s.step.offPtr = 0
	}
	return s
}

func getDecodeForwardCoreScratch(dModel, qDim, kvDim, nHeads, dFF int) *decodeForwardCoreScratch {
	if v := decodeForwardCoreScratchPool.Get(); v != nil {
		if s, ok := v.(*decodeForwardCoreScratch); ok && s.fits(dModel, qDim, kvDim, nHeads, dFF) {
			return s.reset()
		}
	}
	return newDecodeForwardCoreScratch(dModel, qDim, kvDim, nHeads, dFF)
}

func putDecodeForwardCoreScratch(s *decodeForwardCoreScratch) {
	if s != nil {
		decodeForwardCoreScratchPool.Put(s.reset())
	}
}

// DecodeForward — see file header.
func DecodeForward(
	inputs [][]byte, layers []DecodeLayerWeights,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF int,
	base, scale, eps float32,
) ([][]byte, error) {
	return decodeForwardInto(nil, inputs, layers, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, base, scale, eps, false)
}

// DecodeForwardInto is DecodeForward with caller-owned per-token output storage.
// Output slices with enough capacity are reused for the final host readback,
// avoiding per-token output allocation in streaming callers.
func DecodeForwardInto(
	outputs [][]byte, inputs [][]byte, layers []DecodeLayerWeights,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF int,
	base, scale, eps float32,
) ([][]byte, error) {
	return decodeForwardInto(outputs, inputs, layers, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, base, scale, eps, true)
}

func decodeForwardInto(
	outputs [][]byte, inputs [][]byte, layers []DecodeLayerWeights,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF int,
	base, scale, eps float32,
	useCallerOut bool,
) ([][]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	nLayers := len(layers)
	if nLayers == 0 {
		return nil, core.NewError("native.DecodeForward: no layers")
	}
	T := len(inputs)
	if T == 0 {
		return nil, core.NewError("native.DecodeForward: no inputs")
	}
	if T > maxLen {
		return nil, core.NewError("native.DecodeForward: more tokens than maxLen cache rows")
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForward: each input must be dModel bf16 bytes")
		}
	}
	for li := range layers {
		w := layers[li]
		if len(w.AttnNormW) != dModel*bf16Size || len(w.MLPNormW) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForward: layer norm weight size mismatch")
		}
		if len(w.WQ) != qDim*dModel*bf16Size || len(w.WO) != dModel*qDim*bf16Size {
			return nil, core.NewError("native.DecodeForward: layer wQ/wO size mismatch")
		}
		if len(w.WK) != kvDim*dModel*bf16Size || len(w.WV) != kvDim*dModel*bf16Size {
			return nil, core.NewError("native.DecodeForward: layer wK/wV size mismatch")
		}
		if len(w.WGate) != dFF*dModel*bf16Size || len(w.WUp) != dFF*dModel*bf16Size || len(w.WDown) != dModel*dFF*bf16Size {
			return nil, core.NewError("native.DecodeForward: layer MLP weight size mismatch")
		}
	}

	outLen := dModel * bf16Size
	if cap(outputs) < T {
		outputs = make([][]byte, T)
	} else {
		outputs = outputs[:T]
	}
	for i := range outputs {
		if useCallerOut && cap(outputs[i]) >= outLen {
			outputs[i] = outputs[i][:outLen]
			continue
		}
		outputs[i] = make([]byte, outLen)
	}
	var encErr error
	withAutoreleasePool(func() {
		// resident per-layer weight buffers + per-layer caches (caches zeroed; rows
		// fill as tokens append). Created once for the whole forward.
		layerScratch := getDecodeForwardLayerScratch(nLayers)
		defer putDecodeForwardLayerScratch(layerScratch)
		lb := layerScratch.lb
		cacheBytes := uint(maxLen * kvDim * bf16Size)
		residentOrNil := func(b []byte) metal.MTLBuffer {
			if len(b) == 0 {
				return nil
			}
			return residentBytes(b)
		}
		for li := range layers {
			w := layers[li]
			kCache, vCache := layerScratch.kvCache(li, cacheBytes)
			lb[li] = decodeForwardLayerBufs{
				anw: residentBytes(w.AttnNormW), wq: residentBytes(w.WQ), wk: residentBytes(w.WK),
				wv: residentBytes(w.WV), wo: residentBytes(w.WO), mnw: residentBytes(w.MLPNormW),
				wg: residentBytes(w.WGate), wu: residentBytes(w.WUp), wd: residentBytes(w.WDown),
				pan: residentOrNil(w.PostAttnNormW), pfn: residentOrNil(w.PostFFNormW),
				qn: residentOrNil(w.QNormW), kn: residentOrNil(w.KNormW),
				kCache: kCache, vCache: vCache,
			}
		}

		// one bf16 projector per layer (holds that layer's 7 weight buffers); the
		// half-encoders project through it, so a quantised forward differs only in
		// building qmvProjectors here.
		projs := layerScratch.projs
		for li := range lb {
			l := lb[li]
			projs[li] = bf16Projector{
				wQ: bufView{buf: l.wq}, wK: bufView{buf: l.wk}, wV: bufView{buf: l.wv}, wO: bufView{buf: l.wo},
				wGate: bufView{buf: l.wg}, wUp: bufView{buf: l.wu}, wDown: bufView{buf: l.wd},
				dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: dFF,
			}
		}

		// shared scratch (reused across every layer and token; serial dispatch +
		// per-token commit make reuse safe) and the residual-stream ping-pong.
		coreScratch := getDecodeForwardCoreScratch(dModel, qDim, kvDim, nHeads, dFF)
		defer putDecodeForwardCoreScratch(coreScratch)
		asc := coreScratch.asc
		msc := coreScratch.msc
		sc := coreScratch.step

		for t := 0; t < T; t++ {
			sc.seed(t, inputs[t])

			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			in, out := sc.xA, sc.xB
			for li := 0; li < nLayers; li++ {
				l := lb[li]
				if encErr = encAttnHalfKV(enc, in, l.kCache, l.vCache, sc.offBuf, sc.hBuf, bufView{buf: l.anw}, bufView{buf: l.pan}, bufView{buf: l.qn}, bufView{buf: l.kn}, nil, asc, projs[li], dModel, nHeads, nKVHeads, headDim, t, 0, headDim, base, scale, eps, nil); encErr != nil {
					endEncodingFast(enc)
					return
				}
				if encErr = encMLPHalfBF16(enc, sc.hBuf, out, bufView{buf: l.mnw}, bufView{buf: l.pfn}, msc, projs[li], dModel, dFF, eps); encErr != nil {
					endEncodingFast(enc)
					return
				}
				in, out = out, in // next layer reads this layer's output
			}
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			sc.copyBuffer(outputs[t], in) // `in` holds the last layer's output after the final swap
		}
	})
	return outputs, encErr
}
