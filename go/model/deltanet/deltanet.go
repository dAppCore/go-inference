// SPDX-Licence-Identifier: EUPL-1.2

// Package deltanet is the native (no-cgo) gated delta-rule linear-attention recurrence — the mixer of the
// Qwen 3.5 / 3.6 hybrid family (Yang et al. 2024, "Gated Delta Networks"). Qwen 3.6 is, with gemma4, one
// of the two model families that matter for local inference, and its hybrid layers interleave this gated
// delta rule with full attention — so this is the FLA family with a real fleet target (unlike rwkv7/gla/
// gsa, which are unwired research mixers). The recurrence here mirrors metal's GatedDeltaRuleChunkSequential
// exactly but as pure Go over f32 host slices — engine-neutral, verifiable with plain `go test`. The
// chunked-parallel prefill form (decay folded into the WY system) is a later optimisation over this exact
// O(L) recurrence, which serves both prefill and decode correctly.
package deltanet

import (
	"math"

	core "dappco.re/go"
)

// GatedDeltaRuleF32 runs the gated delta-rule recurrence for one sequence (batch 1) and returns the output
// o [L,H,D] and the advanced state [H,D,D] (square: Dk = Dv = D, the per-head dim). Per timestep, with q/k/v
// [H,D] and per-(token,head) scalars α (decay, ∈(0,1]) and β (write strength); k is L2-normalised and q is
// scaled by `scale` (1/√D) inside, matching metal:
//
//	S      ← α_t · S_{t-1}                       // decay the whole prior state
//	read   = k̂_t · S                            // [D]  read at the (normalised) key
//	err    = v_t − read                          // [D]
//	S      = S + k̂_t ⊗ (β_t · err)              // rank-1 delta write
//	o_t    = (scale·q_t) · S                      // [D]  read out with the scaled query (post-write)
//
// prior is the carried state [H,D,D] (decode) or nil for a fresh sequence (prefill, zero state). α ≡ 1
// recovers the plain (ungated) delta rule. Row-major: q/k/v/o[t*H*D + h*D + i], α/β[t*H + h],
// S[h*D*D + i*D + j]. f32 state with f64 within-step accumulation — the higher-precision host reference.
func GatedDeltaRuleF32(q, k, v, beta, alpha, prior []float32, L, H, D int, scale, normEps float32) (o, state []float32, err error) {
	if L <= 0 || H <= 0 || D <= 0 {
		return nil, nil, core.NewError("deltanet.GatedDeltaRuleF32: L,H,D must be > 0")
	}
	if len(q) != L*H*D || len(k) != L*H*D || len(v) != L*H*D {
		return nil, nil, core.NewError("deltanet.GatedDeltaRuleF32: q/k/v must each be [L,H,D]")
	}
	if len(beta) != L*H || len(alpha) != L*H {
		return nil, nil, core.NewError("deltanet.GatedDeltaRuleF32: beta/alpha must each be [L,H]")
	}
	if prior != nil && len(prior) != H*D*D {
		return nil, nil, core.NewError("deltanet.GatedDeltaRuleF32: prior state must be [H,D,D] or nil")
	}
	if normEps <= 0 {
		normEps = 1e-6
	}
	o = make([]float32, L*H*D)
	state = make([]float32, H*D*D)
	if prior != nil {
		copy(state, prior)
	}
	kn := make([]float64, D)   // L2-normalised key
	read := make([]float64, D) // read at the key
	be := make([]float64, D)   // β · error
	for t := range L {
		for h := range H {
			sBase := h * D * D
			row := t*H*D + h*D
			// L2-normalise k_t (over D).
			var ss float64
			for i := range D {
				kv := float64(k[row+i])
				ss += kv * kv
			}
			inv := 1.0 / math.Sqrt(ss+float64(normEps))
			for i := range D {
				kn[i] = float64(k[row+i]) * inv
			}
			a := float64(alpha[t*H+h])
			bta := float64(beta[t*H+h])

			// decay: S ← α · S
			for idx := sBase; idx < sBase+D*D; idx++ {
				state[idx] = float32(a * float64(state[idx]))
			}
			// read[vv] = Σ_kk k̂[kk] · S[kk,vv]   (decayed state)
			for vv := range D {
				read[vv] = 0
			}
			for kk := range D {
				knk := kn[kk]
				sr := sBase + kk*D
				for vv := range D {
					read[vv] += knk * float64(state[sr+vv])
				}
			}
			// be[vv] = β · (v[vv] − read[vv])
			for vv := range D {
				be[vv] = bta * (float64(v[row+vv]) - read[vv])
			}
			// write: S[kk,vv] += k̂[kk] · be[vv]
			for kk := range D {
				knk := kn[kk]
				sr := sBase + kk*D
				for vv := range D {
					state[sr+vv] = float32(float64(state[sr+vv]) + knk*be[vv])
				}
			}
			// o[vv] = Σ_kk (scale·q[kk]) · S_new[kk,vv]
			sc := float64(scale)
			for vv := range D {
				var acc float64
				for kk := range D {
					acc += sc * float64(q[row+kk]) * float64(state[sBase+kk*D+vv])
				}
				o[row+vv] = float32(acc)
			}
		}
	}
	return o, state, nil
}
