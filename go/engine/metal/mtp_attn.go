// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"

	core "dappco.re/go"
)

// mtp_attn.go is the multi-query causal attention the MTP batched verify needs — byte-identical to
// metal.ScaledDotProductAttention(q, k, v, scale, causal=true). The MTP verify runs K draft queries
// against the resident cache in one pass; gemma4's headDim is 256, which the fused steel attention
// does NOT support (it ships only bd128/64/80), so metal falls back to the f32-decomposed attention
// (instrumented: f32 QK^T → f32 softmax → f32 probs·V, output rounded to bf16 — bf16 intermediates
// diverge badly, f32 matches). Native composes the SAME with MatMulF32 (QK^T) + SoftmaxF32 (the GPU
// softmax that matches metal's) + MatMulF32 (probs·V) — the audio-attention pattern.

// sdpaCausalAttnInvalid is the masked-logit fill (underflows to 0 probability, like metal's -inf).
const sdpaCausalAttnInvalid = float32(-1e30)

type sdpaCausalBF16ScratchKey struct {
	H, Hkv, qL, kL, D int
}

type sdpaCausalBF16Scratch struct {
	H, Hkv, qL, kL, D      int
	qf, kf, vf             []float32
	scores, probs, headOut []float32
}

var sdpaCausalBF16ScratchPools sync.Map

type sdpaCausalBF16ScratchPool struct {
	core.Pool[*sdpaCausalBF16Scratch]
}

func sdpaCausalBF16ScratchPoolFor(key sdpaCausalBF16ScratchKey) *sdpaCausalBF16ScratchPool {
	if v, ok := sdpaCausalBF16ScratchPools.Load(key); ok {
		return v.(*sdpaCausalBF16ScratchPool)
	}
	pool := new(sdpaCausalBF16ScratchPool)
	if v, loaded := sdpaCausalBF16ScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*sdpaCausalBF16ScratchPool)
	}
	return pool
}

func sdpaCausalBF16ScratchReady(s *sdpaCausalBF16Scratch, key sdpaCausalBF16ScratchKey) bool {
	return s != nil &&
		s.H == key.H && s.Hkv == key.Hkv && s.qL == key.qL && s.kL == key.kL && s.D == key.D &&
		len(s.qf) == key.H*key.qL*key.D && len(s.kf) == key.Hkv*key.kL*key.D &&
		len(s.vf) == key.Hkv*key.kL*key.D && len(s.scores) == key.qL*key.kL &&
		len(s.probs) == key.qL*key.kL && len(s.headOut) == key.qL*key.D
}

func newSDPACausalBF16Scratch(H, Hkv, qL, kL, D int) *sdpaCausalBF16Scratch {
	return &sdpaCausalBF16Scratch{
		H: H, Hkv: Hkv, qL: qL, kL: kL, D: D,
		qf:      make([]float32, H*qL*D),
		kf:      make([]float32, Hkv*kL*D),
		vf:      make([]float32, Hkv*kL*D),
		scores:  make([]float32, qL*kL),
		probs:   make([]float32, qL*kL),
		headOut: make([]float32, qL*D),
	}
}

func getSDPACausalBF16Scratch(H, Hkv, qL, kL, D int) *sdpaCausalBF16Scratch {
	key := sdpaCausalBF16ScratchKey{H: H, Hkv: Hkv, qL: qL, kL: kL, D: D}
	pool := sdpaCausalBF16ScratchPoolFor(key)
	if s := pool.Get(); s != nil {
		if sdpaCausalBF16ScratchReady(s, key) {
			return s
		}
	}
	return newSDPACausalBF16Scratch(H, Hkv, qL, kL, D)
}

func putSDPACausalBF16Scratch(s *sdpaCausalBF16Scratch) {
	if s == nil {
		return
	}
	key := sdpaCausalBF16ScratchKey{H: s.H, Hkv: s.Hkv, qL: s.qL, kL: s.kL, D: s.D}
	if sdpaCausalBF16ScratchReady(s, key) {
		sdpaCausalBF16ScratchPoolFor(key).Put(s)
	}
}

func bf16ToF32Into(out []float32, b []byte) {
	for i := range out {
		o := i * bf16Size
		out[i] = bf16ToF32(b[o], b[o+1])
	}
}

// SDPACausalBF16 is causal scaled-dot-product attention on bf16 q/k/v in head-major [H, L, D] layout
// (within batch 1), returning bf16 [H, qL, D] — byte-identical to metal.ScaledDotProductAttention with
// causal=true. q has H heads, k/v have Hkv heads (GQA: head h reads kv head h/(H/Hkv)); query i (the
// last qL positions) attends keys [0 .. kL-qL+i]. Computed in f32 (widened weights), rounded to bf16.
func SDPACausalBF16(q, k, v []byte, H, Hkv, qL, kL, D int, scale float32) ([]byte, error) {
	return SDPACausalBF16Into(nil, q, k, v, H, Hkv, qL, kL, D, scale)
}

// SDPACausalBF16Into is SDPACausalBF16 with caller-owned output storage when cap(out) is large enough.
func SDPACausalBF16Into(out, q, k, v []byte, H, Hkv, qL, kL, D int, scale float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(q) != H*qL*D*bf16Size || len(k) != Hkv*kL*D*bf16Size || len(v) != Hkv*kL*D*bf16Size {
		return nil, core.NewError("native.SDPACausalBF16: q/k/v sizes must match [H,qL,D]/[Hkv,kL,D] bf16")
	}
	if H%Hkv != 0 {
		return nil, core.NewError("native.SDPACausalBF16: H must be a multiple of Hkv")
	}
	outLen := H * qL * D * bf16Size
	if cap(out) < outLen {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	scratch := getSDPACausalBF16Scratch(H, Hkv, qL, kL, D)
	defer putSDPACausalBF16Scratch(scratch)
	qf, kf, vf := scratch.qf, scratch.kf, scratch.vf
	bf16ToF32Into(qf, q)
	bf16ToF32Into(kf, k)
	bf16ToF32Into(vf, v)
	gqa := H / Hkv
	scores := scratch.scores
	probs := scratch.probs
	oh := scratch.headOut
	for h := range H {
		hk := h / gqa
		qh := qf[h*qL*D : (h+1)*qL*D]   // [qL, D]
		kh := kf[hk*kL*D : (hk+1)*kL*D] // [kL, D]
		vh := vf[hk*kL*D : (hk+1)*kL*D] // [kL, D]

		// scores = (qh · khᵀ)·scale, causal-masked: [qL, kL].
		var err error
		scores, err = matMulF32NTIntoPublic(scores, qh, kh, qL, D, kL, false)
		if err != nil {
			return nil, err
		}
		for i := range qL {
			lim := kL - qL + i
			for j := range kL {
				if j <= lim {
					scores[i*kL+j] *= scale
				} else {
					scores[i*kL+j] = sdpaCausalAttnInvalid
				}
			}
		}
		if err := softmaxF32Into(probs, scores, kL, false); err != nil {
			return nil, err
		}
		// out_h = probs · vh : [qL, kL]·[kL, D] = [qL, D].
		oh, err = matMulF32Into(oh, probs, vh, qL, kL, D, false)
		if err != nil {
			return nil, err
		}
		base := h * qL * D * bf16Size
		for i, val := range oh {
			hh := f32ToBF16(val)
			out[base+i*bf16Size], out[base+i*bf16Size+1] = byte(hh), byte(hh>>8)
		}
	}
	return out, nil
}
