// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/attn"
	"dappco.re/go/inference/model/composed"
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

// qwenChainMoE builds the minimal composed.MoEMLP view the chain's MoE tail resolves its weights
// from, out of the factory's native MoE holder — batched switch_mlp experts, softmax top-k router
// (dequantised to f32 once), the shared expert trio and its σ gate. Dims are DERIVED from the packed
// byte strides (the don't-guess rule). nil when the layer is not a chain-servable qwen MoE.
func qwenChainMoE(moe *MoEQuantLayerWeights, D int) *composed.MoEMLP {
	if moe == nil || len(moe.SharedGate.Packed) == 0 || moe.NumExperts <= 0 || moe.TopK <= 0 || moe.ExpertDFF <= 0 {
		return nil
	}
	var router []float32
	if moe.Router.Bits > 0 {
		r, err := dequantizeAffineRowsF32(moe.Router.Packed, moe.Router.Scales, moe.Router.Biases, moe.NumExperts, D, moe.Router.GroupSize, moe.Router.Bits)
		if err != nil {
			return nil
		}
		router = r
	} else {
		router = bf16VecToF32(moe.Router.Packed)
	}
	if len(router) != moe.NumExperts*D {
		return nil
	}
	gate := archQuantToModel(moe.ExpGate, moe.ExpertDFF, D)
	up := archQuantToModel(moe.ExpUp, moe.ExpertDFF, D)
	down := archQuantToModel(moe.ExpDown, D, moe.ExpertDFF)
	if gate == nil || up == nil || down == nil {
		return nil
	}
	// shared expert FF from the packed row stride of its gate ([FF, D] at Bits): rowBytes = D·Bits/8.
	sg := moe.SharedGate
	if sg.Bits <= 0 || D*sg.Bits%8 != 0 {
		return nil
	}
	sharedFF := len(sg.Packed) / (D * sg.Bits / 8)
	sGate := archQuantToModel(moe.SharedGate, sharedFF, D)
	sUp := archQuantToModel(moe.SharedUp, sharedFF, D)
	sDown := archQuantToModel(moe.SharedDown, D, sharedFF)
	if sGate == nil || sUp == nil || sDown == nil || sharedFF <= 0 {
		return nil
	}
	var sharedGate []float32
	if len(moe.SharedSigmoid.Packed) > 0 {
		if moe.SharedSigmoid.Bits > 0 {
			gg, err := dequantizeAffineRowsF32(moe.SharedSigmoid.Packed, moe.SharedSigmoid.Scales, moe.SharedSigmoid.Biases, 1, D, moe.SharedSigmoid.GroupSize, moe.SharedSigmoid.Bits)
			if err != nil {
				return nil
			}
			sharedGate = gg
		} else {
			sharedGate = bf16VecToF32(moe.SharedSigmoid.Packed)
		}
		if len(sharedGate) != D {
			return nil
		}
	}
	return &composed.MoEMLP{
		Router: router, Experts: make([]composed.MoEExpert, moe.NumExperts), TopK: moe.TopK,
		NormTopKProb: true, Gating: model.MoEGatingSoftmax,
		GateBatchedQ: gate, UpBatchedQ: up, DownBatchedQ: down,
		Shared:     &composed.MoEExpert{GateQ: sGate, UpQ: sUp, DownQ: sDown},
		SharedGate: sharedGate,
		MoEBits:    moe.ExpGate.Bits, MoEGroupSize: moe.ExpGate.GroupSize,
	}
}

// gatedDeltaQuantServable mirrors gatedDeltaQuantLayerRun's projection geometry gate at BIND time:
// every packed projection must be a width the affine qmv emitters serve (quantGeometryOK — Bonsai's
// 1-bit packs are not). A mid-walk geometry error would leave earlier layers' recurrent state
// advanced; failing eligibility here keeps such checkpoints on the host halves instead.
func gatedDeltaQuantServable(w *model.GatedDeltaWeights, cfg model.GatedDeltaConfig, d int) bool {
	if w == nil {
		return false
	}
	convDim, vDim := cfg.ConvDim(), cfg.VDim()
	return quantGeometryOK(w.InProjQKVQ, convDim, d) && quantGeometryOK(w.InProjZQ, vDim, d) &&
		quantGeometryOK(w.InProjAQ, cfg.ValueHeads, d) && quantGeometryOK(w.InProjBQ, cfg.ValueHeads, d) &&
		quantGeometryOK(w.OutProjQ, d, vDim)
}

// bindQwenFusedDense captures the fused-lane weight views on the two Qwen holder kinds after the
// host-path binds have run. Dense-FFN layers latch fusedDense (the whole-layer seams carry the SiLU
// tail); qwen MoE layers latch fusedMoE with the chain's MoEMLP view (the chain MoE tail owns the
// FFN half there). Weight-side eligibility latches here; device-side usability latches on first use
// (the metallib is not loaded yet at bind time).
func (s *archDecodeState) bindQwenFusedDense(layers []QuantizedLayerWeights) {
	for i := range layers {
		L := &layers[i]
		if len(L.AttnNormW) == 0 {
			continue
		}
		if L.MoE != nil {
			mm := qwenChainMoE(L.MoE, s.dModel)
			if mm == nil || len(L.MoE.PreFFNormW) == 0 {
				continue
			}
			preFF := bf16VecToF32(L.MoE.PreFFNormW)
			if gd := s.layerGatedDelta(i); gd != nil {
				if !gatedDeltaQuantServable(gd.w, gd.cfg, s.dModel) {
					continue
				}
				gd.inNorm = bf16VecToF32(L.AttnNormW)
				gd.ffNorm = preFF
				gd.moe = mm
				gd.fusedMoE = true
				continue
			}
			if s.gatedAttn == nil || s.gatedAttn[i] == nil {
				continue
			}
			ga := s.gatedAttn[i]
			if ga.bq != nil || ga.bk != nil || ga.bv != nil {
				continue
			}
			lhd, lkv := headDimOf(s.specs[i], s.headDim), kvHeadsOf(s.specs[i], s.nKVHeads)
			mq := archQuantToModel(ga.q, 2*s.nHeads*lhd, s.dModel)
			mk := archQuantToModel(ga.k, lkv*lhd, s.dModel)
			mv := archQuantToModel(ga.v, lkv*lhd, s.dModel)
			mo := archQuantToModel(ga.o, s.dModel, s.nHeads*lhd)
			if mq == nil || mk == nil || mv == nil || mo == nil ||
				!quantGeometryOK(mq, 2*s.nHeads*lhd, s.dModel) || !quantGeometryOK(mk, lkv*lhd, s.dModel) ||
				!quantGeometryOK(mv, lkv*lhd, s.dModel) || !quantGeometryOK(mo, s.dModel, s.nHeads*lhd) {
				continue
			}
			ga.inNorm = bf16VecToF32(L.AttnNormW)
			ga.ffNorm = preFF
			ga.mq, ga.mk, ga.mv, ga.mo = mq, mk, mv, mo
			ga.moe = mm
			ga.fusedMoE = true
			continue
		}
		if len(L.MLPNormW) == 0 {
			continue
		}
		ff := s.dFF
		if L.DFF > 0 {
			ff = L.DFF
		}
		gate := archQuantToModel(L.Gate, ff, s.dModel)
		up := archQuantToModel(L.Up, ff, s.dModel)
		down := archQuantToModel(L.Down, s.dModel, ff)
		if gate == nil || up == nil || down == nil || ff <= 0 ||
			!quantGeometryOK(gate, ff, s.dModel) || !quantGeometryOK(up, ff, s.dModel) || !quantGeometryOK(down, s.dModel, ff) {
			continue
		}
		if gd := s.layerGatedDelta(i); gd != nil {
			if !gatedDeltaQuantServable(gd.w, gd.cfg, s.dModel) {
				continue
			}
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
		if mq == nil || mk == nil || mv == nil || mo == nil ||
			!quantGeometryOK(mq, 2*s.nHeads*lhd, s.dModel) || !quantGeometryOK(mk, lkv*lhd, s.dModel) ||
			!quantGeometryOK(mv, lkv*lhd, s.dModel) || !quantGeometryOK(mo, s.dModel, s.nHeads*lhd) {
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
			if !gd.fusedDense && !gd.fusedMoE {
				return false
			}
			if !gatedDeltaBlockUsable(gd.cfg.HeadDim, gd.cfg.HeadDim, gd.cfg.KeyHeads, gd.cfg.ValueHeads, gd.cfg.ConvKernel) {
				return false
			}
			if gd.fusedMoE && !moeChainRecordable(gd.moe) {
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
		if !ga.fusedDense && !ga.fusedMoE {
			return false
		}
		if !attnCoreUsable(s.nHeads, lkv, lhd, s.rotaryDim) {
			return false
		}
		if ga.fusedMoE && !moeChainRecordable(ga.moe) {
			return false
		}
	}
	s.qwenChainAll = true
	return true
}

// qwenChainWalk drives every layer's chain call against a live or recording chain context — the
// one walk both the record pass and the live encode share (mirroring composed.chainWalk).
func (s *archDecodeState) qwenChainWalk(ctx any, pos int) error {
	for li := range s.specs {
		if gd := s.layerGatedDelta(li); gd != nil {
			if gd.sc == nil {
				gd.sc = &attn.GatedDeltaScratch{}
			}
			if gd.fusedMoE {
				if err := gatedDeltaQuantChainMoELayerDevice(ctx, gd.sc, gd.inNorm, gd.w, gd.cfg, gd.ffNorm, gd.moe, gd.conv, gd.delta, s.eps); err != nil {
					return err
				}
			} else if err := gatedDeltaQuantChainLayerDevice(ctx, gd.sc, gd.inNorm, gd.w, gd.cfg, gd.ffNorm, gd.ffGate, gd.ffUp, gd.ffDown, gd.conv, gd.delta, gd.dff, s.eps); err != nil {
				return err
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
		var dev any
		var aerr error
		if ga.fusedMoE {
			dev, aerr = attnQuantChainMoELayerDevice(ctx, ga.dev, ga.inNorm, ga.mq, ga.mk, ga.mv, ga.mo,
				ga.qNorm, ga.kNorm, ga.ffNorm, ga.moe,
				nil, nil, s.nHeads, lkv, lhd, rotDim, pos, slideW, 1, qkNorm, s.eps, rbase)
		} else {
			dev, aerr = attnQuantChainLayerDevice(ctx, ga.dev, ga.inNorm, ga.mq, ga.mk, ga.mv, ga.mo,
				ga.qNorm, ga.kNorm, ga.ffNorm, ga.ffGate, ga.ffUp, ga.ffDown,
				nil, nil, s.nHeads, lkv, lhd, rotDim, pos, slideW, 1, qkNorm, ga.dff, s.eps, rbase)
		}
		if aerr != nil {
			return aerr
		}
		ga.dev = dev
	}
	return nil
}

// stepTokenQwenChain decodes ONE token through the composed chain machinery: replay the recorded
// command stream when one is valid (one executeCommandsInBuffer, no re-encode — mirroring
// composed.forwardChain), else record once and run the live walk (one command buffer for the whole
// layer stack, resident state). Writes the final hidden (bf16) into `out` and returns it. The
// caller skips the per-layer loop entirely.
func (s *archDecodeState) stepTokenQwenChain(inputBuf, out metal.MTLBuffer, pos int) (metal.MTLBuffer, error) {
	D := s.dModel
	x := bf16BufToF32(inputBuf, 0, D)

	if s.qwenChainRec != nil {
		y, _, ok, rerr := ComposedChainReplayDevice(s.qwenChainRec, x)
		if rerr == nil && ok {
			// composedChainReplay advanced ALL the state itself — the device KV rows, ring and delta
			// through the executed commands, and the attnKVDeviceState.n counters host-side (these ARE
			// the factory's ga.dev handles; bumping them again here double-advances the position and
			// silently corrupts the KV slot addressing — the debugged-in-anger receipt).
			bf16WriteBuf(out, y)
			return out, nil
		}
		ComposedChainRecordingRelease(s.qwenChainRec)
		s.qwenChainRec = nil // stale (cache grew/realloc'd) — re-encode this token, re-record later
	}

	// Record BEFORE the live encode (states not yet advanced, so the recorded desync checks see the
	// same position the live pass uses). A recording failure latches off — the stack won't change
	// shape mid-session.
	if s.qwenChainRec == nil && !s.qwenChainRecFailed {
		if rctx, rerr := ComposedChainRecordBegin(1, D, len(s.specs)); rerr == nil {
			if recErr := s.qwenChainWalk(rctx, pos); recErr == nil {
				if rec, eerr := ComposedChainRecordEnd(rctx); eerr == nil {
					s.qwenChainRec = rec
				} else {
					s.qwenChainRecFailed = true
				}
			} else {
				s.qwenChainRecFailed = true
			}
		}
	}

	ctx, err := ComposedChainBeginDevice(x, 1, D)
	if err != nil {
		return nil, err
	}
	if err := s.qwenChainWalk(ctx, pos); err != nil {
		return nil, err
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
