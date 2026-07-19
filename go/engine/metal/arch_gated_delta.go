// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/attn"
	"dappco.re/go/inference/model/composed"
	"github.com/tmc/apple/metal"
)

// arch_gated_delta.go decodes a MixerGatedDelta layer in the arch (factory) session — the unified
// engine's gated-delta mixer, replacing the composed lane's per-seam decode (#18). Correctness-first:
// the recurrence runs on the HOST through model/attn (the proven GatedDeltaForwardScratchF32), the state
// (conv ring + delta) threads on the layer struct, and the pre-norm residual is applied exactly as
// encAttnHalfKV does for attention (h = in + maybe_postNorm(mixer(RMSNorm(in)))). The attention layers
// keep their device path untouched, so gemma4 is byte-identical. This host path is now the FALLBACK —
// the fused device lanes (arch_qwen_fused.go: whole-layer seams + the whole-token chain walk) are the
// default for servable geometries; this file remains the correctness reference (LTHN_QWEN_FUSED=0).

// gatedDeltaLayer is one MixerGatedDelta layer's weights + geometry + recurrent state for arch decode.
type gatedDeltaLayer struct {
	w           *model.GatedDeltaWeights
	cfg         model.GatedDeltaConfig
	sc          *attn.GatedDeltaScratch
	conv, delta []float32 // the conv ring + delta state, advanced each decode token (host path)

	// fused device lane (arch_qwen_fused.go): the whole layer through gatedDeltaQuantLayerRun (dense)
	// or the chain's gated-delta + MoE-tail walk (fusedMoE; ffNorm then holds the pre-FF norm and moe
	// the chain's MoEMLP view). fusedDense/fusedMoE latch weight-side eligibility at bind;
	// devChecked/devOK latch device-side usability on first use. y is the reused layer output row.
	inNorm, ffNorm       []float32
	ffGate, ffUp, ffDown *model.QuantWeight
	dff                  int
	moe                  *composed.MoEMLP
	fusedDense, fusedMoE bool
	devChecked, devOK    bool
	y                    []float32
}

// bf16BufToF32 widens n bf16 elements at a device buffer's offset to a fresh []float32.
func bf16BufToF32(buf metal.MTLBuffer, off uint, n int) []float32 {
	b := unsafe.Slice((*byte)(unsafe.Add(buf.Contents(), off)), n*2)
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(uint32(uint16(b[2*i])|uint16(b[2*i+1])<<8) << 16)
	}
	return out
}

// rmsNormHostF32 applies RMSNorm(x)·w over a single [D] row on the host — the f32 reference norm for the
// gated-delta layer's input + optional post-mixer norms (bf16 weight widened).
func rmsNormHostF32(x []float32, w []float32, eps float32) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(1.0 / math.Sqrt(ss/float64(len(x))+float64(eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv * w[i]
	}
	return out
}

// bindGatedDeltaQuant / bindGatedDeltaBF16 build the per-layer recurrence holders from the session's
// converted layer weights — the weight-flow that lets encGatedDeltaHalf read a MixerGatedDelta layer's
// projections + geometry. nil holder for an attention layer (the branch is never taken there). Called
// once by NewArchSession(Quant) after newArchDecodeState.
func (s *archDecodeState) bindGatedDeltaQuant(layers []QuantizedLayerWeights) {
	s.gatedDelta = make([]*gatedDeltaLayer, len(layers))
	for i := range layers {
		if i < len(s.specs) && s.specs[i].Mixer == model.MixerGatedDelta && layers[i].GatedDelta != nil {
			s.gatedDelta[i] = &gatedDeltaLayer{w: layers[i].GatedDelta, cfg: layers[i].GatedDeltaCfg}
		}
	}
}

func (s *archDecodeState) bindGatedDeltaBF16(layers []DecodeLayerWeights) {
	s.gatedDelta = make([]*gatedDeltaLayer, len(layers))
	for i := range layers {
		if i < len(s.specs) && s.specs[i].Mixer == model.MixerGatedDelta && layers[i].GatedDelta != nil {
			s.gatedDelta[i] = &gatedDeltaLayer{w: layers[i].GatedDelta, cfg: layers[i].GatedDeltaCfg}
		}
	}
}

// encGatedDeltaHalf computes one MixerGatedDelta layer's mixer for a single decode token and writes the
// residual output (in + maybe_postNorm(gatedDelta(RMSNorm(in)))) to hBuf — the recurrence twin of
// encAttnHalfKV. Host path (correctness-first). Advances the layer's conv/delta state.
func (s *archDecodeState) encGatedDeltaHalf(li int, in metal.MTLBuffer) error {
	if li >= len(s.gatedDelta) || s.gatedDelta[li] == nil {
		return core.NewError("native.encGatedDeltaHalf: gated-delta layer weights missing")
	}
	gd := s.gatedDelta[li]
	if gd.sc == nil {
		gd.sc = &attn.GatedDeltaScratch{}
	}
	D := s.dModel
	inF := bf16BufToF32(in, 0, D)
	normW := bf16BufToF32(s.lb[li].anw.buf, s.lb[li].anw.off, D)
	normed := rmsNormHostF32(inF, normW, s.eps)

	out, nc, nd, err := attn.GatedDeltaForwardScratchF32(normed, gd.w, gd.cfg, gd.conv, gd.delta, 1, D, gd.sc)
	if err != nil {
		return err
	}
	gd.conv, gd.delta = nc, nd

	if s.lb[li].postAttnNorm.buf != nil { // post-mixer norm applied to the mixer output before the residual
		out = rmsNormHostF32(out, bf16BufToF32(s.lb[li].postAttnNorm.buf, s.lb[li].postAttnNorm.off, D), s.eps)
	}
	for d := 0; d < D; d++ {
		inF[d] += out[d]
	}

	// write the [D] residual back to hBuf as bf16
	hb := unsafe.Slice((*byte)(s.hBuf.Contents()), D*2)
	for d := 0; d < D; d++ {
		u := f32ToBF16(inF[d])
		hb[2*d], hb[2*d+1] = byte(u), byte(u>>8)
	}
	return nil
}
