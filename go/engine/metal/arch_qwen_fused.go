// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/attn"
	"github.com/tmc/apple/metal"
)

// arch_qwen_fused.go — the DEVICE decode for the factory session's Qwen hybrid layers (#18 fusion).
// The host halves (encGatedDeltaHalf / encGatedAttnHalf) pay ~15 submit+wait round trips per layer per
// token; these lanes run the WHOLE layer through the composed lane's proven single-command-buffer
// seams instead — gatedDeltaQuantLayerRun (norm → 4 packed projections → conv/gates/recurrence with
// device-resident state → out_proj → SiLU FFN tail) and AttnQuantFullLayerDevice (norm → packed q/k/v
// → rope/QK-norm prep → resident-KV SDPA → σ output gate → o_proj → SiLU FFN tail). One x upload and
// one y readback per layer. The host halves stay as the fallback for geometries the device block
// cannot serve (and for the captureLayerHiddens diff probe, which needs the split halves). Attention
// layers without the output gate and every gemma path are untouched.

// archQuantToModel views a native QuantWeight as the model-typed form the composed seams take. The
// packed/scales/biases bytes are shared, not copied; dims are the projection's logical [out,in].
// nil for an absent or bf16 (unquantised) weight — the fused lanes are packed-only.
func archQuantToModel(w QuantWeight, outDim, inDim int) *model.QuantWeight {
	if len(w.Packed) == 0 || len(w.Scales) == 0 || w.Bits <= 0 || w.GroupSize <= 0 {
		return nil
	}
	return &model.QuantWeight{
		Packed: w.Packed, Scales: w.Scales, Biases: w.Biases,
		Bits: w.Bits, GroupSize: w.GroupSize, OutDim: outDim, InDim: inDim,
	}
}

// bindQwenFusedDense captures the fused-lane weight views on the two Qwen holder kinds after the
// host-path binds have run — dense-FFN layers only (a MoE layer's FFN half stays with its own
// encoder path). Weight-side eligibility latches here; device-side usability latches on first use
// (the metallib is not loaded yet at bind time).
func (s *archDecodeState) bindQwenFusedDense(layers []QuantizedLayerWeights) {
	for i := range layers {
		L := &layers[i]
		if L.MoE != nil || len(L.AttnNormW) == 0 || len(L.MLPNormW) == 0 {
			continue
		}
		ff := s.dFF
		if L.DFF > 0 {
			ff = L.DFF
		}
		gate := archQuantToModel(L.Gate, ff, s.dModel)
		up := archQuantToModel(L.Up, ff, s.dModel)
		down := archQuantToModel(L.Down, s.dModel, ff)
		if gate == nil || up == nil || down == nil || ff <= 0 {
			continue
		}
		if gd := s.layerGatedDelta(i); gd != nil {
			gd.inNorm = bf16VecToF32(L.AttnNormW)
			gd.ffNorm = bf16VecToF32(L.MLPNormW)
			gd.ffGate, gd.ffUp, gd.ffDown, gd.dff = gate, up, down, ff
			gd.fusedDense = true
			continue
		}
		if s.gatedAttn == nil || s.gatedAttn[i] == nil {
			continue
		}
		ga := s.gatedAttn[i]
		if ga.bq != nil || ga.bk != nil || ga.bv != nil { // the device seam carries no additive biases
			continue
		}
		lhd, lkv := headDimOf(s.specs[i], s.headDim), kvHeadsOf(s.specs[i], s.nKVHeads)
		mq := archQuantToModel(ga.q, 2*s.nHeads*lhd, s.dModel)
		mk := archQuantToModel(ga.k, lkv*lhd, s.dModel)
		mv := archQuantToModel(ga.v, lkv*lhd, s.dModel)
		mo := archQuantToModel(ga.o, s.dModel, s.nHeads*lhd)
		if mq == nil || mk == nil || mv == nil || mo == nil {
			continue
		}
		ga.inNorm = bf16VecToF32(L.AttnNormW)
		ga.ffNorm = bf16VecToF32(L.MLPNormW)
		ga.mq, ga.mk, ga.mv, ga.mo = mq, mk, mv, mo
		ga.ffGate, ga.ffUp, ga.ffDown, ga.dff = gate, up, down, ff
		ga.fusedDense = true
	}
}

// layerGatedDelta returns the bound gated-delta holder for a layer, nil otherwise.
func (s *archDecodeState) layerGatedDelta(li int) *gatedDeltaLayer {
	if li >= len(s.gatedDelta) {
		return nil
	}
	return s.gatedDelta[li]
}

// hasRecurrentLayers reports whether any layer carries session state OUTSIDE the paged KV pool —
// the gated-delta recurrence or a gated-attention holder (host or device KV). Such state cannot
// unwind a speculative token, so the submit-ahead decode tails must decline the session (a stop
// with links in flight would leave the recurrence one token past the sequence).
func (s *archDecodeState) hasRecurrentLayers() bool {
	for _, gd := range s.gatedDelta {
		if gd != nil {
			return true
		}
	}
	for _, ga := range s.gatedAttn {
		if ga != nil {
			return true
		}
	}
	return false
}

// qwenFusedDisabled is the kill-switch: LTHN_QWEN_FUSED=0 forces the host halves (the A/B baseline +
// revert-safety, latched once — the house rule for wall-clock-adaptive paths).
var qwenFusedDisabled = os.Getenv("LTHN_QWEN_FUSED") == "0"

// gatedDeltaFusedReady reports whether this MixerGatedDelta layer decodes on the fused device lane.
// Weight-side eligibility latched at bind; device-side usability latched on the first call (after
// ensureInit, so the customLibrary check is meaningful). The captureLayerHiddens diff probe needs the
// split halves, so it forces the host path for the whole run.
func (s *archDecodeState) gatedDeltaFusedReady(li int) bool {
	gd := s.gatedDelta[li]
	if gd == nil || !gd.fusedDense || captureLayerHiddens || qwenFusedDisabled {
		return false
	}
	if !gd.devChecked {
		gd.devChecked = true
		if err := ensureInit(); err == nil {
			gd.devOK = gatedDeltaBlockUsable(gd.cfg.HeadDim, gd.cfg.HeadDim, gd.cfg.KeyHeads, gd.cfg.ValueHeads, gd.cfg.ConvKernel)
		}
	}
	return gd.devOK
}

// gatedAttnFusedReady is the gated-attention twin of gatedDeltaFusedReady.
func (s *archDecodeState) gatedAttnFusedReady(li, heads, kvHeads, headDim, rotDim int) bool {
	ga := s.gatedAttn[li]
	if ga == nil || !ga.fusedDense || captureLayerHiddens || qwenFusedDisabled {
		return false
	}
	if !ga.devChecked {
		ga.devChecked = true
		if err := ensureInit(); err == nil {
			ga.devOK = attnCoreUsable(heads, kvHeads, headDim, rotDim)
		}
	}
	return ga.devOK
}

// bf16WriteBuf narrows a [D] f32 row into a device buffer's bf16 contents.
func bf16WriteBuf(buf metal.MTLBuffer, x []float32) {
	b := unsafe.Slice((*byte)(buf.Contents()), len(x)*2)
	for i, v := range x {
		u := f32ToBF16(v)
		b[2*i], b[2*i+1] = byte(u), byte(u>>8)
	}
}

// encGatedDeltaFusedLayer decodes one WHOLE MixerGatedDelta layer (mixer + dense FFN tail) for a
// single token on the device lane: reads the running hidden from `in`, runs gatedDeltaQuantLayerRun
// (state device-resident on gd.sc.Device), writes the layer output to `out`. The caller skips the
// loop's FFN half for this layer.
func (s *archDecodeState) encGatedDeltaFusedLayer(li int, in, out metal.MTLBuffer) error {
	gd := s.gatedDelta[li]
	if gd == nil {
		return core.NewError("native.encGatedDeltaFusedLayer: gated-delta layer weights missing")
	}
	if gd.sc == nil {
		gd.sc = &attn.GatedDeltaScratch{}
	}
	h, _ := gd.sc.Device.(*gatedDeltaDeviceState)
	if h == nil {
		nh, err := newGatedDeltaDeviceState(gd.cfg.KeyHeads, gd.cfg.ValueHeads, gd.cfg.HeadDim, gd.cfg.HeadDim, gd.cfg.ConvKernel, 1)
		if err != nil {
			return err
		}
		h = nh
	}
	D := s.dModel
	x := bf16BufToF32(in, 0, D)
	if gd.y == nil {
		gd.y = make([]float32, D)
	}
	if err := gatedDeltaQuantLayerRun(h, x, gd.inNorm, gd.w, gd.ffNorm, gd.ffGate, gd.ffUp, gd.ffDown, 1, D, gd.dff, s.eps, gd.conv, gd.delta, gd.y); err != nil {
		return err
	}
	gd.sc.Device = h
	bf16WriteBuf(out, gd.y)
	return nil
}

// qwenChainReady latches whether EVERY layer of this session can ride the whole-token chain — the
// all-or-nothing gate for the composed chain walk (one command buffer for the entire layer stack,
// state resident, ComposedChainBegin/End). Pre-flighted per layer BEFORE the first walk so a failed
// usability check can never leave the recurrent state part-advanced mid-token.
func (s *archDecodeState) qwenChainReady() bool {
	if s.qwenChainChecked {
		return s.qwenChainAll
	}
	s.qwenChainChecked = true
	s.qwenChainAll = false
	if captureLayerHiddens || qwenFusedDisabled || len(s.specs) == 0 || ensureInit() != nil {
		return false
	}
	for li := range s.specs {
		if gd := s.layerGatedDelta(li); gd != nil {
			if !gd.fusedDense || !gatedDeltaBlockUsable(gd.cfg.HeadDim, gd.cfg.HeadDim, gd.cfg.KeyHeads, gd.cfg.ValueHeads, gd.cfg.ConvKernel) {
				return false
			}
			continue
		}
		if s.gatedAttn == nil || s.gatedAttn[li] == nil {
			return false // a plain (non-qwen) layer — the chain walk serves the pure hybrid only
		}
		ga := s.gatedAttn[li]
		lhd := headDimOf(s.specs[li], s.headDim)
		lkv := kvHeadsOf(s.specs[li], s.nKVHeads)
		if !ga.fusedDense || !attnCoreUsable(s.nHeads, lkv, lhd, s.rotaryDim) {
			return false
		}
	}
	s.qwenChainAll = true
	return true
}

// stepTokenQwenChain decodes ONE token through the composed chain walk: the whole layer stack
// encodes into a single command buffer (ComposedChainBeginDevice → per-layer chain calls with
// resident state → ComposedChainEndDevice), replacing the per-layer submit+wait of the fused
// seams. Writes the final hidden (bf16) into `out` and returns it. The caller skips the
// per-layer loop entirely.
func (s *archDecodeState) stepTokenQwenChain(inputBuf, out metal.MTLBuffer, pos int) (metal.MTLBuffer, error) {
	D := s.dModel
	x := bf16BufToF32(inputBuf, 0, D)
	ctx, err := ComposedChainBeginDevice(x, 1, D)
	if err != nil {
		return nil, err
	}
	for li := range s.specs {
		if gd := s.layerGatedDelta(li); gd != nil {
			if gd.sc == nil {
				gd.sc = &attn.GatedDeltaScratch{}
			}
			if err := gatedDeltaQuantChainLayerDevice(ctx, gd.sc, gd.inNorm, gd.w, gd.cfg, gd.ffNorm, gd.ffGate, gd.ffUp, gd.ffDown, gd.conv, gd.delta, gd.dff, s.eps); err != nil {
				return nil, err
			}
			continue
		}
		ga := s.gatedAttn[li]
		lhd := headDimOf(s.specs[li], s.headDim)
		lkv := kvHeadsOf(s.specs[li], s.nKVHeads)
		slideW, rbase, rotDim := 0, s.base, s.rotaryDim
		if s.specs[li].Attention == model.SlidingAttention {
			slideW, rbase, rotDim = s.slidingWindow, s.localBase, s.rotaryDimLocal
		}
		qkNorm := 0
		if len(ga.qNorm) > 0 || len(ga.kNorm) > 0 {
			qkNorm = 1
		}
		dev, aerr := attnQuantChainLayerDevice(ctx, ga.dev, ga.inNorm, ga.mq, ga.mk, ga.mv, ga.mo,
			ga.qNorm, ga.kNorm, ga.ffNorm, ga.ffGate, ga.ffUp, ga.ffDown,
			nil, nil, s.nHeads, lkv, lhd, rotDim, pos, slideW, 1, qkNorm, ga.dff, s.eps, rbase)
		if aerr != nil {
			return nil, aerr
		}
		ga.dev = dev
	}
	y, err := ComposedChainEndDevice(ctx)
	if err != nil {
		return nil, err
	}
	bf16WriteBuf(out, y)
	return out, nil
}

// encGatedAttnFusedLayer decodes one WHOLE gated full-attention layer (attention + dense FFN tail)
// for a single token on the device lane via AttnQuantFullLayerDevice — resident device KV on ga.dev,
// σ output gate, partial rotary, per-head QK-norm. Writes the layer output to `out`; the caller
// skips the loop's FFN half for this layer.
func (s *archDecodeState) encGatedAttnFusedLayer(li, pos, heads, kvHeads, headDim, rotDim int, theta float32, slideW int, in, out metal.MTLBuffer) error {
	ga := s.gatedAttn[li]
	if ga == nil {
		return core.NewError("native.encGatedAttnFusedLayer: gated-attention layer weights missing")
	}
	D := s.dModel
	x := bf16BufToF32(in, 0, D)
	qkNorm := 0
	if len(ga.qNorm) > 0 || len(ga.kNorm) > 0 {
		qkNorm = 1
	}
	y, dev, err := AttnQuantFullLayerDevice(ga.dev, x, ga.inNorm, ga.mq, ga.mk, ga.mv, ga.mo,
		ga.qNorm, ga.kNorm, ga.ffNorm, ga.ffGate, ga.ffUp, ga.ffDown,
		nil, nil, 1, D, heads, kvHeads, headDim, rotDim, pos, slideW, 1, qkNorm, ga.dff, s.eps, theta)
	if err != nil {
		return err
	}
	ga.dev = dev
	bf16WriteBuf(out, y)
	return nil
}
