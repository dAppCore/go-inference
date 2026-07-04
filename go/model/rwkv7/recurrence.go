// SPDX-Licence-Identifier: EUPL-1.2

// Package rwkv7 is the native (no-cgo) RWKV-7 ("Goose") mixer: the generalised delta-rule WKV7 recurrence
// and the pieces an RWKV-7 time-mixing block composes from it. It is the second SSM/FLA family ported off
// metal, on the same scaffold as pkg/model/mamba2 (recurrence + decode-carry test + recurrent session +
// loader + SessionModel wrapper + device-GEMM seam). The recurrence here mirrors metal's WKV7 exactly
// (github.com/fla-org/flash-linear-attention rwkv7/fused_recurrent) but as pure Go over f32 host slices,
// so it is engine-neutral and verifiable with plain `go test`. A device / chunked-parallel form is a
// later optimisation over this exact O(L) recurrence.
package rwkv7

import (
	"math"

	core "dappco.re/go"
)

// WKV7F32 runs the RWKV-7 recurrence for one sequence (batch 1) and returns the mixed output o [L,H,V]
// and the advanced state [H,K,V]. Per head the state S is a [K,V] matrix; per timestep, with r/w/k/a/b
// [H,K] and v [H,V] (w the per-channel LOG-decay, ≤ 0, so exp(w) ∈ (0,1] is the forget gate):
//
//	Sa     = aᵀ · S_old                                  // [V]   contract K (OLD state)
//	S      = diag(exp(w)) · S_old  +  b ⊗ Sa  +  k ⊗ v   // [K,V] decay + rank-1 transition + rank-1 write
//	o      = Sᵀ · r                                      // [V]   contract K (NEW state)
//
// prior is the carried state [H,K,V] from the previous chunk (decode) or nil for a fresh sequence
// (prefill, zero state). Row-major: r/w/k/a/b[t*H*K + h*K + i], v/o[t*H*V + h*V + j], S[h*K*V + i*V + j].
// f32 throughout — the precision metal's WKV7 and the fla reference keep through the recurrence.
func WKV7F32(r, w, k, v, a, b, prior []float32, L, H, K, V int) (o, state []float32, err error) {
	if L <= 0 || H <= 0 || K <= 0 || V <= 0 {
		return nil, nil, core.NewError("rwkv7.WKV7F32: L,H,K,V must be > 0")
	}
	if len(r) != L*H*K || len(w) != L*H*K || len(k) != L*H*K || len(a) != L*H*K || len(b) != L*H*K {
		return nil, nil, core.NewError("rwkv7.WKV7F32: r/w/k/a/b must each be [L,H,K]")
	}
	if len(v) != L*H*V {
		return nil, nil, core.NewError("rwkv7.WKV7F32: v must be [L,H,V]")
	}
	if prior != nil && len(prior) != H*K*V {
		return nil, nil, core.NewError("rwkv7.WKV7F32: prior state must be [H,K,V] or nil")
	}
	o = make([]float32, L*H*V)
	state = make([]float32, H*K*V)
	if prior != nil {
		copy(state, prior)
	}
	sa := make([]float64, V) // Sa[v], reused per (t,h)
	for t := 0; t < L; t++ {
		for h := 0; h < H; h++ {
			sBase := h * K * V
			kBase := t*H*K + h*K // r/w/k/a/b row [t,h,:]
			vBase := t*H*V + h*V // v/o row [t,h,:]

			// Sa[vv] = Σ_kk a[kk] · S_old[kk,vv]
			for vv := range sa {
				sa[vv] = 0
			}
			for kk := 0; kk < K; kk++ {
				ak := float64(a[kBase+kk])
				row := sBase + kk*V
				for vv := 0; vv < V; vv++ {
					sa[vv] += ak * float64(state[row+vv])
				}
			}
			// S[kk,vv] = exp(w[kk])·S_old[kk,vv] + b[kk]·Sa[vv] + k[kk]·v[vv]
			for kk := 0; kk < K; kk++ {
				ew := math.Exp(float64(w[kBase+kk]))
				bk := float64(b[kBase+kk])
				kv := float64(k[kBase+kk])
				row := sBase + kk*V
				for vv := 0; vv < V; vv++ {
					state[row+vv] = float32(ew*float64(state[row+vv]) + bk*sa[vv] + kv*float64(v[vBase+vv]))
				}
			}
			// o[vv] = Σ_kk r[kk] · S_new[kk,vv]
			for vv := 0; vv < V; vv++ {
				var acc float64
				for kk := 0; kk < K; kk++ {
					acc += float64(r[kBase+kk]) * float64(state[sBase+kk*V+vv])
				}
				o[vBase+vv] = float32(acc)
			}
		}
	}
	return o, state, nil
}
