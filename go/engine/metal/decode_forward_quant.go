// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// QuantWeight is one projection's affine-quantised weight: MLX's packed codes + bf16 scales +
// bf16 biases (one scale/bias per group per row). Sidecar-less Packed is also accepted by the
// arch quant path as a dense bf16 matrix after pack-level fusion. GroupSize/Bits are the weight's
// OWN affine geometry — mixed-precision packs (e4b-qat: the MLP is 8-bit while attention is 4-bit)
// vary it per weight; 0 ⇒ fall back to the projector's layer-default groupSize/bits (uniform packs).
type QuantWeight struct {
	Packed, Scales, Biases []byte
	GroupSize, Bits        int
	resident               bool // synthesised heap buffer (e.g. the fused ExpGateUp), not a mapped-shard view → resident-copy, never shard-lookup
	packedView             bufView
	scalesView             bufView
	biasesView             bufView
}

// QuantizedLayerWeights is one decode layer with 4-bit projections: the two
// RMSNorm weights stay bf16 (norms aren't quantised — tiny vectors), the seven
// matmuls are quantised. GroupSize ∈ {32,64,128}, Bits = 4 for the models we serve.
type QuantizedLayerWeights struct {
	AttnNormW, MLPNormW        []byte
	Q, K, V, O, Gate, Up, Down QuantWeight
	GroupSize, Bits            int
	// DFF is this layer's FFN width — gemma4 E2B/E4B (MatFormer) vary it per layer, so the decode
	// can't assume a single arch.FF. 0 ⇒ use the arch default (uniform models).
	DFF int
	// gemma4 norms (bf16, not quantised), applied when non-nil: PostAttnNormW /
	// PostFFNormW before their residual add; QNormW / KNormW per-head on Q/K before RoPE.
	PostAttnNormW, PostFFNormW []byte
	QNormW, KNormW             []byte
	// LayerScalarW is gemma4's per-layer output scalar (shape [1] bf16, not quantised); the
	// arch executor multiplies the layer's final hidden by it. nil when omitted.
	LayerScalarW []byte
	// per-layer-input gate (gemma4 E2B/E4B): the 4-bit gate (pliDim×dModel) + projection
	// (dModel×pliDim) and the bf16 post-norm (dModel). All nil for models without the PLE
	// tower (the dense 12B). Applied at the layer tail by PerLayerInputGateQuant.
	PerLayerGate, PerLayerProjection QuantWeight
	PostPerLayerInputNormW           []byte
	// MoE, when non-nil (gemma4 26B-A4B), replaces the dense MLP half with the 4-bit dual-branch
	// MoEBlockQuant for this layer; the dense MLPNormW/Gate/Up/Down are then unused.
	MoE *MoEQuantLayerWeights
}

type decodeForwardQuantLayerBufs struct {
	anw, mnw, pan, pfn, qn, kn metal.MTLBuffer
	kCache, vCache             metal.MTLBuffer
}

type decodeForwardQuantLayerScratch struct {
	lb      []decodeForwardQuantLayerBufs
	projs   []qmvProjector
	kCaches []metal.MTLBuffer
	vCaches []metal.MTLBuffer
	kBytes  []uint
	vBytes  []uint
}

var decodeForwardQuantLayerScratchPool sync.Pool

func newDecodeForwardQuantLayerScratch(nLayers int) *decodeForwardQuantLayerScratch {
	return &decodeForwardQuantLayerScratch{
		lb:      make([]decodeForwardQuantLayerBufs, nLayers),
		projs:   make([]qmvProjector, nLayers),
		kCaches: make([]metal.MTLBuffer, nLayers),
		vCaches: make([]metal.MTLBuffer, nLayers),
		kBytes:  make([]uint, nLayers),
		vBytes:  make([]uint, nLayers),
	}
}

func (s *decodeForwardQuantLayerScratch) fits(nLayers int) bool {
	return s != nil &&
		cap(s.lb) >= nLayers && cap(s.projs) >= nLayers &&
		cap(s.kCaches) >= nLayers && cap(s.vCaches) >= nLayers &&
		cap(s.kBytes) >= nLayers && cap(s.vBytes) >= nLayers
}

func (s *decodeForwardQuantLayerScratch) reset(nLayers int) *decodeForwardQuantLayerScratch {
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

func (s *decodeForwardQuantLayerScratch) kvCache(li int, cacheBytes uint) (metal.MTLBuffer, metal.MTLBuffer) {
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

func getDecodeForwardQuantLayerScratch(nLayers int) *decodeForwardQuantLayerScratch {
	if v := decodeForwardQuantLayerScratchPool.Get(); v != nil {
		if s, ok := v.(*decodeForwardQuantLayerScratch); ok && s.fits(nLayers) {
			return s.reset(nLayers)
		}
	}
	return newDecodeForwardQuantLayerScratch(nLayers)
}

func putDecodeForwardQuantLayerScratch(s *decodeForwardQuantLayerScratch) {
	if s != nil {
		decodeForwardQuantLayerScratchPool.Put(s.reset(0))
	}
}

// DecodeForwardQuant is DecodeForward with 4-bit-quantised projections: identical
// in every other respect (bf16 activations, growing seq-major KV cache, one
// commit+wait per token, residual stream layer→layer), because the only thing that
// changes is the projector — qmvProjector (affine_qmv_bfloat16_t) instead of
// bf16Projector. This is the whole 4-bit decode forward running with NO mlx-c at
// runtime. With the same logical weights it equals DecodeForward up to quantisation
// (gated against the parity-proven standalone ops in the tests). All raw bf16 I/O.
func DecodeForwardQuant(
	inputs [][]byte, qlayers []QuantizedLayerWeights,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF int,
	base, scale, eps float32,
) ([][]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	nLayers, T := len(qlayers), len(inputs)
	if nLayers == 0 || T == 0 {
		return nil, core.NewError("native.DecodeForwardQuant: need layers and inputs")
	}
	if T > maxLen {
		return nil, core.NewError("native.DecodeForwardQuant: more tokens than maxLen cache rows")
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardQuant: each input must be dModel bf16 bytes")
		}
	}
	// validate per-layer: norms bf16; each projection's packed/scales/biases sizes
	type pj struct {
		w           QuantWeight
		outDim, inD int
	}
	for li := range qlayers {
		ql := qlayers[li]
		if ql.GroupSize == 0 || ql.Bits == 0 {
			return nil, core.NewError("native.DecodeForwardQuant: GroupSize/Bits unset")
		}
		if len(ql.AttnNormW) != dModel*bf16Size || len(ql.MLPNormW) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardQuant: norm weight size mismatch")
		}
		for _, p := range []pj{
			{ql.Q, qDim, dModel}, {ql.K, kvDim, dModel}, {ql.V, kvDim, dModel}, {ql.O, dModel, qDim},
			{ql.Gate, dFF, dModel}, {ql.Up, dFF, dModel}, {ql.Down, dModel, dFF},
		} {
			if !quantWeightShapeOK(p.w, p.outDim, p.inD, ql.GroupSize, ql.Bits) {
				return nil, core.NewError("native.DecodeForwardQuant: quantised weight size mismatch")
			}
		}
	}

	outputs := make([][]byte, T)
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}
	var encErr error
	withAutoreleasePool(func() {
		// per-layer resident: bf16 norms + the quantised projector + growing caches
		layerScratch := getDecodeForwardQuantLayerScratch(nLayers)
		defer putDecodeForwardQuantLayerScratch(layerScratch)
		lb := layerScratch.lb
		projs := layerScratch.projs
		cacheBytes := uint(maxLen * kvDim * bf16Size)
		residentView := func(b []byte) bufView { return bufView{buf: residentBytes(b)} }
		residentOrNil := func(b []byte) metal.MTLBuffer {
			if len(b) == 0 {
				return nil
			}
			return residentBytes(b)
		}
		mkW := func(qw QuantWeight) qmvWeight {
			return qmvWeight{wq: residentView(qw.Packed), scales: residentView(qw.Scales), biases: residentView(qw.Biases), gs: qw.GroupSize, bits: qw.Bits}
		}
		for li := range qlayers {
			ql := qlayers[li]
			kCache, vCache := layerScratch.kvCache(li, cacheBytes)
			lb[li] = decodeForwardQuantLayerBufs{
				anw: residentBytes(ql.AttnNormW), mnw: residentBytes(ql.MLPNormW),
				pan: residentOrNil(ql.PostAttnNormW), pfn: residentOrNil(ql.PostFFNormW),
				qn: residentOrNil(ql.QNormW), kn: residentOrNil(ql.KNormW),
				kCache: kCache, vCache: vCache,
			}
			projs[li] = qmvProjector{
				q: mkW(ql.Q), k: mkW(ql.K), v: mkW(ql.V), o: mkW(ql.O),
				gate: mkW(ql.Gate), up: mkW(ql.Up), down: mkW(ql.Down),
				dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: dFF,
				groupSize: ql.GroupSize, bits: ql.Bits,
			}
		}

		coreScratch := getDecodeForwardCoreScratch(dModel, qDim, kvDim, nHeads, dFF)
		defer putDecodeForwardCoreScratch(coreScratch)
		asc := coreScratch.asc
		msc := coreScratch.msc
		sc := coreScratch.step

		for t := range T {
			sc.seed(t, inputs[t])

			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			in, out := sc.xA, sc.xB
			for li := range nLayers {
				l := lb[li]
				if encErr = encAttnHalfKV(enc, in, l.kCache, l.vCache, sc.offBuf, sc.hBuf, bufView{buf: l.anw}, bufView{buf: l.pan}, bufView{buf: l.qn}, bufView{buf: l.kn}, nil, asc, projs[li], dModel, nHeads, nKVHeads, headDim, t, 0, headDim, base, scale, eps, nil); encErr != nil {
					endEncodingFast(enc)
					return
				}
				if encErr = encMLPHalfBF16(enc, sc.hBuf, out, bufView{buf: l.mnw}, bufView{buf: l.pfn}, msc, projs[li], dModel, dFF, eps); encErr != nil {
					endEncodingFast(enc)
					return
				}
				in, out = out, in
			}
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			sc.copyBuffer(outputs[t], in)
		}
	})
	return outputs, encErr
}
