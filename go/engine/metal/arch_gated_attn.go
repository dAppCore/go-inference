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

// arch_gated_attn.go decodes a GATED full-attention layer (Qwen3.5 attn_output_gate) in the arch (factory)
// session — the host attention twin of encGatedDeltaHalf. Correctness-first: the whole attention forward
// (input RMSNorm → q/k/v proj → per-head QK-norm + partial rotary → causal softmax over a host KV cache →
// σ-gate → o_proj → residual) runs on the HOST, ported from model/composed's continueFromQKV. Only the
// full_attention layers of a checkpoint that DECLARES attn_output_gate take this path; a plain (ungated)
// attention layer keeps the device encAttnHalfKV, so gemma4 is byte-identical. Device fusion is a later slice.

// gatedAttnLayer is one gated full-attention layer's host-accessible weights + KV state. Geometry
// (heads/kv/head_dim/rotary/theta/window) is passed per call from the decode loop's per-layer resolution,
// matching how encAttnHalfKV is driven — the holder carries only what is layer-resident.
type gatedAttnLayer struct {
	q, k, v, o   QuantWeight // packed projections; q emits [q ; gate] per head (2·HD blocks) when gated
	qNorm, kNorm []float32   // per-head QK-norm weights (widened bf16); nil ⇒ no QK-norm
	bq, bk, bv   []float32   // optional additive q/k/v biases (widened bf16); nil ⇒ bias-free (qwen3_5)

	kCache, vCache []float32 // host KV cache [n · kvHeads·headDim], grown one token per decode step
	n              int       // cached positions (host path)

	// fused device lane (arch_qwen_fused.go): the whole layer through AttnQuantFullLayerDevice with
	// resident device KV on dev (dense), or the chain's gated-attention + MoE-tail walk (fusedMoE;
	// ffNorm then holds the pre-FF norm and moe the chain's MoEMLP view). fusedDense/fusedMoE latch
	// weight-side eligibility at bind; devChecked/devOK latch device-side usability on first use.
	inNorm, ffNorm       []float32
	mq, mk, mv, mo       *model.QuantWeight
	ffGate, ffUp, ffDown *model.QuantWeight
	dff                  int
	moe                  *composed.MoEMLP
	fusedDense, fusedMoE bool
	devChecked, devOK    bool
	dev                  any
}

// bf16VecToF32 widens a bf16 []byte vector (a norm/bias tensor) to a fresh []float32; nil for empty.
func bf16VecToF32(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	out := make([]float32, len(b)/2)
	for i := range out {
		out[i] = math.Float32frombits(uint32(uint16(b[2*i])|uint16(b[2*i+1])<<8) << 16)
	}
	return out
}

// rmsNormHeadF32 / applyRotaryHalfF32 are the per-head QK-norm + partial rotary, ported from
// model/composed (rmsNormHead / applyRotaryHalf) so the factory attention is bit-for-bit the composed math.
func rmsNormHeadF32(x, w []float32, eps float32) {
	if len(w) == 0 {
		return
	}
	var ss float64
	for _, e := range x {
		ss += float64(e) * float64(e)
	}
	r := math.Sqrt(ss/float64(len(x)) + float64(eps))
	for i := range x {
		x[i] = float32(float64(x[i]) / r * float64(w[i]))
	}
}

func applyRotaryHalfF32(x []float32, pos, rotaryDim int, theta float64) {
	half := rotaryDim / 2
	for i := 0; i < half; i++ {
		freq := 1.0 / math.Pow(theta, float64(2*i)/float64(rotaryDim))
		ang := float64(pos) * freq
		c, s := math.Cos(ang), math.Sin(ang)
		a, b := float64(x[i]), float64(x[i+half])
		x[i] = float32(a*c - b*s)
		x[i+half] = float32(b*c + a*s)
	}
}

// projQuantAttn applies out[N] = x[K] @ dequant(w)ᵀ for a single row through the shared host quant matmul
// seam (attn.ProjQuantMatMulInto), plus an optional additive bias — the q/k/v/o projections' host path.
func projQuantAttn(w QuantWeight, x []float32, k, n int, bias []float32) ([]float32, error) {
	out := make([]float32, n)
	res, err := attn.ProjQuantMatMulInto(out, x, w.Packed, w.Scales, w.Biases, 1, k, n, w.GroupSize, w.Bits)
	if err != nil {
		return nil, err
	}
	if len(bias) == n {
		for i := range res {
			res[i] += bias[i]
		}
	}
	return res, nil
}

// bindGatedAttnQuant builds the per-layer gated-attention holders from the converted quant layer weights —
// only when the Arch declares attn_output_gate, and only for the attention (non-gated-delta) layers. A plain
// attention layer keeps its device path (nil holder ⇒ the branch is never taken). Called once by
// NewArchSessionQuant after newArchDecodeState + bindGatedDeltaQuant.
func (s *archDecodeState) bindGatedAttnQuant(layers []QuantizedLayerWeights) {
	if !s.attnOutputGate {
		return
	}
	s.gatedAttn = make([]*gatedAttnLayer, len(layers))
	for i := range layers {
		if i >= len(s.specs) || s.specs[i].Mixer == model.MixerGatedDelta {
			continue // gated-delta layers decode via encGatedDeltaHalf, not here
		}
		L := &layers[i]
		s.gatedAttn[i] = &gatedAttnLayer{
			q: L.Q, k: L.K, v: L.V, o: L.O,
			qNorm: bf16VecToF32(L.QNormW), kNorm: bf16VecToF32(L.KNormW),
			bq: bf16VecToF32(L.BQ), bk: bf16VecToF32(L.BK), bv: bf16VecToF32(L.BV),
		}
	}
}

// encGatedAttnHalf computes one gated full-attention layer for a single decode token (position pos) and
// writes the residual output (in + σ(gate)·attn(RMSNorm(in))·o_proj) to hBuf — the gated twin of
// encAttnHalfKV. Host path (correctness-first), ported from composed.continueFromQKV at L=1. Grows the
// layer's host KV cache. heads/kvHeads/headDim/rotDim/theta/slideW are the loop's per-layer geometry.
func (s *archDecodeState) encGatedAttnHalf(li, pos, heads, kvHeads, headDim, rotDim int, theta float32, slideW int, in metal.MTLBuffer) error {
	ga := s.gatedAttn[li]
	if ga == nil {
		return core.NewError("native.encGatedAttnHalf: gated-attention layer weights missing")
	}
	if heads <= 0 || kvHeads <= 0 || headDim <= 0 || heads%kvHeads != 0 {
		return core.NewError("native.encGatedAttnHalf: bad attention geometry")
	}
	D := s.dModel
	inF := bf16BufToF32(in, 0, D)
	normed := rmsNormHostF32(inF, bf16BufToF32(s.lb[li].anw.buf, s.lb[li].anw.off, D), s.eps)

	// q/k/v projections. q is the [q ; gate] raw (2·heads·headDim), k/v are kvHeads·headDim.
	qRaw, err := projQuantAttn(ga.q, normed, D, 2*heads*headDim, ga.bq)
	if err != nil {
		return err
	}
	k, err := projQuantAttn(ga.k, normed, D, kvHeads*headDim, ga.bk)
	if err != nil {
		return err
	}
	v, err := projQuantAttn(ga.v, normed, D, kvHeads*headDim, ga.bv)
	if err != nil {
		return err
	}

	// De-interleave the per-head output gate: within each head's 2·HD block, [q_h ; gate_h].
	q := make([]float32, heads*headDim)
	gate := make([]float32, heads*headDim)
	for h := 0; h < heads; h++ {
		src := qRaw[h*2*headDim:]
		copy(q[h*headDim:h*headDim+headDim], src[:headDim])
		copy(gate[h*headDim:h*headDim+headDim], src[headDim:2*headDim])
	}

	// Per-head QK-norm + partial rotary at absolute position pos (gate is never normed or rotated).
	th := float64(theta)
	for h := 0; h < heads; h++ {
		row := q[h*headDim : h*headDim+headDim]
		rmsNormHeadF32(row, ga.qNorm, s.eps)
		applyRotaryHalfF32(row, pos, rotDim, th)
	}
	for h := 0; h < kvHeads; h++ {
		row := k[h*headDim : h*headDim+headDim]
		rmsNormHeadF32(row, ga.kNorm, s.eps)
		applyRotaryHalfF32(row, pos, rotDim, th)
	}

	// Grow the host KV cache by this token (decode is sequential: pos == ga.n).
	ga.kCache = append(ga.kCache, k...)
	ga.vCache = append(ga.vCache, v...)
	ga.n = pos + 1
	N := ga.n

	// Causal softmax attention over the cached keys/values, GQA (rep query heads per kv head), with the
	// optional sliding window. scale = 1/√headDim.
	rep := heads / kvHeads
	scale := 1.0 / math.Sqrt(float64(headDim))
	first := 0
	if slideW > 0 && N > slideW {
		first = N - slideW
	}
	last := pos // inclusive
	out := make([]float32, heads*headDim)
	scores := make([]float64, N)
	for h := 0; h < heads; h++ {
		kvh := h / rep
		qrow := q[h*headDim:]
		maxS := math.Inf(-1)
		for j := first; j <= last; j++ {
			krow := ga.kCache[j*kvHeads*headDim+kvh*headDim:]
			var dot float64
			for d := 0; d < headDim; d++ {
				dot += float64(qrow[d]) * float64(krow[d])
			}
			dot *= scale
			scores[j] = dot
			if dot > maxS {
				maxS = dot
			}
		}
		var sum float64
		for j := first; j <= last; j++ {
			scores[j] = math.Exp(scores[j] - maxS)
			sum += scores[j]
		}
		orow := out[h*headDim:]
		for d := 0; d < headDim; d++ {
			var acc float64
			for j := first; j <= last; j++ {
				acc += scores[j] * float64(ga.vCache[j*kvHeads*headDim+kvh*headDim+d])
			}
			orow[d] = float32(acc / sum)
		}
	}

	// attn_output_gate: σ(gate) · attn output, per element, before o_proj (qwen3_5 hardcodes sigmoid).
	for i := range out {
		out[i] = float32(float64(out[i]) * (1.0 / (1.0 + math.Exp(-float64(gate[i])))))
	}

	// o_proj + residual, then write the [D] result to hBuf as bf16 (as encGatedDeltaHalf does).
	o, err := projQuantAttn(ga.o, out, heads*headDim, D, nil)
	if err != nil {
		return err
	}
	for d := 0; d < D; d++ {
		inF[d] += o[d]
	}
	hb := unsafe.Slice((*byte)(s.hBuf.Contents()), D*2)
	for d := 0; d < D; d++ {
		u := f32ToBF16(inF[d])
		hb[2*d], hb[2*d+1] = byte(u), byte(u>>8)
	}
	return nil
}
