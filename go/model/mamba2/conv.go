// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import core "dappco.re/go"

// conv.go is the Mamba-2 causal depthwise conv1d — the short (kernel ~4) per-channel causal convolution
// applied to the xBC stream before the scan. Each of the convDim channels has its own K-tap filter; the
// output at step t mixes only the current and past K-1 inputs (causal), so a streaming decode carries the
// last K-1 inputs as a "conv state" ring across calls. Pure Go over f32 host slices, engine-neutral.

// CausalConv1dF32 runs the causal depthwise conv1d over in [L, convDim] with per-channel weight
// [convDim, K] and optional bias [convDim], returning out [L, convDim] and the conv-state ring
// newState [(K-1), convDim] for the next chunk. prior is the carried [(K-1), convDim] ring (the previous
// chunk's last K-1 inputs) or nil for a fresh sequence (zero-padded). The window: out[t,ch] =
// bias[ch] + Σ_k weight[ch,k]·x[t-K+1+k, ch], so weight[ch,K-1] multiplies the current input (the
// standard causal orientation). Row-major: in[t*convDim+ch], weight[ch*K+k]. SiLU is applied by the
// caller (it follows the conv in the block).
func CausalConv1dF32(in, weight, bias, prior []float32, L, convDim, K int) (out, newState []float32, err error) {
	if L <= 0 || convDim <= 0 || K <= 0 {
		return nil, nil, core.NewError("mamba2.CausalConv1dF32: L,convDim,K must be > 0")
	}
	if len(in) != L*convDim || len(weight) != convDim*K {
		return nil, nil, core.NewError("mamba2.CausalConv1dF32: in[L,convDim]/weight[convDim,K] size mismatch")
	}
	pad := K - 1
	if bias != nil && len(bias) != convDim {
		return nil, nil, core.NewError("mamba2.CausalConv1dF32: bias must be [convDim] or nil")
	}
	if prior != nil && len(prior) != pad*convDim {
		return nil, nil, core.NewError("mamba2.CausalConv1dF32: prior must be [(K-1),convDim] or nil")
	}
	// padded row r: r<pad → the prior ring (or 0); r>=pad → input row r-pad.
	get := func(r, ch int) float64 {
		if r < pad {
			if prior == nil {
				return 0
			}
			return float64(prior[r*convDim+ch])
		}
		return float64(in[(r-pad)*convDim+ch])
	}
	out = make([]float32, L*convDim)
	for t := 0; t < L; t++ {
		for ch := 0; ch < convDim; ch++ {
			acc := 0.0
			if bias != nil {
				acc = float64(bias[ch])
			}
			for k := 0; k < K; k++ {
				acc += float64(weight[ch*K+k]) * get(t+k, ch) // weight[K-1] hits padded[t+K-1] = current input
			}
			out[t*convDim+ch] = float32(acc)
		}
	}
	// the next chunk's ring = the last K-1 inputs = padded rows [L .. L+pad-1].
	newState = make([]float32, pad*convDim)
	for r := 0; r < pad; r++ {
		for ch := 0; ch < convDim; ch++ {
			newState[r*convDim+ch] = float32(get(L+r, ch))
		}
	}
	return out, newState, nil
}
