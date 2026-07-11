// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"

	core "dappco.re/go"
)

// attention.go is the full_attention mixer for the hybrid stack — qwen3-style dense attention (per-head
// QK-norm → partial rotary → GQA → causal softmax) with a growing KV cache as its state. It is the cut-2
// peer of the gated-delta mixer: same Mixer interface, but its state is a KV cache instead of a recurrent
// matrix, exactly the per-layer cache-typing metal's composed model does. Host f32; the projections use
// the package matNT (the device-GEMM path is a later optimisation, shared with the gated-delta seam).

// AttnConfig is the per-layer attention geometry. RotaryDim ≤ HeadDim (partial rotary; Qwen 3.6 uses
// 0.25·HeadDim). KVHeads ≤ Heads (GQA). OutputGate is the gated-attention flag (attn_output_gate): when
// set, q_proj emits [q ; gate] per head and the attention output is σ(gate)-gated before o_proj.
type AttnConfig struct {
	Heads, KVHeads, HeadDim, RotaryDim int
	RopeTheta, NormEps                 float32
	OutputGate                         bool
}

// AttnWeights is one layer's attention weights. QProj is [Heads*HeadDim, D] (or [2·Heads*HeadDim, D] when
// OutputGate — the [q ; gate] projection); K/VProj [KVHeads*HeadDim, D]; OProj [D, Heads*HeadDim];
// QNorm/KNorm [HeadDim] (per-head RMSNorm, plain — qwen is not gemma).
type AttnWeights struct {
	QProj, KProj, VProj, OProj []float32
	QNorm, KNorm               []float32
}

type attnMixer struct {
	w   *AttnWeights
	cfg AttnConfig
}

// attnState is the KV cache: n past tokens, k/v laid out [n, KVHeads, HeadDim] (rotary already applied to
// the cached keys).
type attnState struct {
	k, v []float32
	n    int
}

// NewAttnMixer builds a full-attention mixer for one layer.
func NewAttnMixer(w *AttnWeights, cfg AttnConfig) Mixer { return &attnMixer{w: w, cfg: cfg} }

func (m *attnMixer) Kind() string { return "full_attention" }

// rmsNormHead RMS-normalises a single [HeadDim] vector in place by weight w.
func rmsNormHead(x, w []float32, eps float32) {
	var ss float64
	for _, e := range x {
		ss += float64(e) * float64(e)
	}
	r := math.Sqrt(ss/float64(len(x)) + float64(eps))
	for i := range x {
		x[i] = float32(float64(x[i]) / r * float64(w[i]))
	}
}

// applyRotaryHalf rotates the first rotaryDim dims of a [HeadDim] vector at position pos (the rotate_half
// convention: pair i with i+rotaryDim/2), leaving dims [rotaryDim:] unchanged.
func applyRotaryHalf(x []float32, pos, rotaryDim int, theta float64) {
	half := rotaryDim / 2
	for i := range half {
		freq := 1.0 / math.Pow(theta, float64(2*i)/float64(rotaryDim))
		ang := float64(pos) * freq
		c, s := math.Cos(ang), math.Sin(ang)
		a, b := float64(x[i]), float64(x[i+half])
		x[i] = float32(a*c - b*s)
		x[i+half] = float32(b*c + a*s)
	}
}

// Forward runs attention over hidden [L,D], appending the new K/V to the cache and attending causally over
// all cached tokens. Returns out [L,D] and the grown cache.
func (m *attnMixer) Forward(h []float32, L, D int, prior any) ([]float32, any, error) {
	cfg := m.cfg
	H, KVH, HD, RD := cfg.Heads, cfg.KVHeads, cfg.HeadDim, cfg.RotaryDim
	if H <= 0 || KVH <= 0 || HD <= 0 || H%KVH != 0 {
		return nil, nil, core.NewError("composed.attnMixer: bad geometry")
	}
	theta := float64(cfg.RopeTheta)
	if theta == 0 {
		theta = 1e6
	}
	qCols := H * HD
	if cfg.OutputGate {
		qCols = 2 * H * HD // q_proj emits [q ; gate] per head
	}
	if len(m.w.QProj) != qCols*D {
		return nil, nil, core.NewError("composed.attnMixer: q_proj size mismatch (OutputGate?)")
	}
	var st attnState
	if p, ok := prior.(attnState); ok {
		st = p
	}
	pos0 := st.n
	scale := 1.0 / math.Sqrt(float64(HD))
	rep := H / KVH

	// q_proj: [L, H*HD] ungated, or [L, 2·H*HD] gated — de-interleaved per head into q [L,H*HD] and the
	// gate [L,H*HD] ([q_h ; gate_h] within each head's 2·HD block, per the transformers qwen3_5 chunk).
	var q, gate []float32
	if cfg.OutputGate {
		raw := matNT(h, m.w.QProj, L, D, 2*H*HD)
		q = make([]float32, L*H*HD)
		gate = make([]float32, L*H*HD)
		for t := range L {
			for hd := range H {
				src := raw[t*2*H*HD+hd*2*HD:]
				copy(q[(t*H+hd)*HD:(t*H+hd)*HD+HD], src[:HD])
				copy(gate[(t*H+hd)*HD:(t*H+hd)*HD+HD], src[HD:2*HD])
			}
		}
	} else {
		q = matNT(h, m.w.QProj, L, D, H*HD) // [L, H*HD]
	}
	k := matNT(h, m.w.KProj, L, D, KVH*HD) // [L, KVH*HD]
	v := matNT(h, m.w.VProj, L, D, KVH*HD) // [L, KVH*HD]

	// QK-norm (per head) + partial rotary at absolute positions pos0+t.
	for t := range L {
		for hd := range H {
			row := q[t*H*HD+hd*HD : t*H*HD+hd*HD+HD]
			rmsNormHead(row, m.w.QNorm, cfg.NormEps)
			applyRotaryHalf(row, pos0+t, RD, theta)
		}
		for hd := range KVH {
			row := k[t*KVH*HD+hd*HD : t*KVH*HD+hd*HD+HD]
			rmsNormHead(row, m.w.KNorm, cfg.NormEps)
			applyRotaryHalf(row, pos0+t, RD, theta)
		}
	}

	// grow the cache: [pos0+L, KVH*HD]. ck and cv are the returned state — one backing slab, two
	// non-overlapping capped windows. Both are copied out (read-only) at the head of the next call
	// before any write, so sharing one array between the K and V caches is bit-identical and saves
	// one alloc per token on the decode path.
	N := pos0 + L
	cacheN := N * KVH * HD
	ckcv := make([]float32, 2*cacheN)
	ck := ckcv[0:cacheN:cacheN]
	cv := ckcv[cacheN : 2*cacheN : 2*cacheN]
	copy(ck, st.k)
	copy(cv, st.v)
	copy(ck[pos0*KVH*HD:], k)
	copy(cv[pos0*KVH*HD:], v)

	// causal attention: query t (position pos0+t) attends to cached keys 0..pos0+t. The query buffer q
	// is [L,H,HD] = the output shape and is dead after this loop; each head's qrow is fully consumed by
	// its score dot-products before that head's orow is written (orow is at the same offset), so the
	// attention output is written in place over q — bit-identical, one fewer alloc per token.
	out := q
	scores := make([]float64, N)
	for t := range L {
		last := pos0 + t // inclusive
		for hd := range H {
			kvh := hd / rep
			qrow := q[t*H*HD+hd*HD:]
			// scores over keys 0..last
			maxS := math.Inf(-1)
			for j := 0; j <= last; j++ {
				krow := ck[j*KVH*HD+kvh*HD:]
				var dot float64
				for d := range HD {
					dot += float64(qrow[d]) * float64(krow[d])
				}
				dot *= scale
				scores[j] = dot
				if dot > maxS {
					maxS = dot
				}
			}
			// softmax
			var sum float64
			for j := 0; j <= last; j++ {
				scores[j] = math.Exp(scores[j] - maxS)
				sum += scores[j]
			}
			// weighted sum of values
			orow := out[t*H*HD+hd*HD:]
			for d := range HD {
				var acc float64
				for j := 0; j <= last; j++ {
					acc += scores[j] * float64(cv[j*KVH*HD+kvh*HD+d])
				}
				orow[d] = float32(acc / sum)
			}
		}
	}
	// attn_output_gate: gate the attention output (per head, per dim) by σ(gate) before o_proj. The gate
	// is the second half of each head's q_proj block — never QK-normed or rotated. The transformers
	// qwen3_5 reference hardcodes sigmoid here (output_gate_type is not consumed by the reference forward).
	if gate != nil {
		for i := range out {
			s := 1.0 / (1.0 + math.Exp(-float64(gate[i])))
			out[i] = float32(float64(out[i]) * s)
		}
	}
	o := matNT(out, m.w.OProj, L, H*HD, D)
	return o, attnState{k: ck, v: cv, n: N}, nil
}
