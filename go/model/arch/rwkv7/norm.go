// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import "math"

// norm.go holds the small numeric primitives the real RWKV-7 "Goose" chain needs beyond the WKV7
// recurrence (recurrence.go) and the simplified projection block (block.go): token-shift, the addcmul
// mixing fla's rwkv7 layer uses to build every projection's input, real (mean+variance) LayerNorm — RWKV-7
// uses nn.LayerNorm for pre_norm/attn_norm/ffn_norm/the final norm, NOT the plain RMSNorm mamba2 uses — and
// the per-head GroupNorm applied to the WKV7 read-out. All f64-accumulate/f32-store, matching the
// convention block.go/recurrence.go/mamba2 already keep through this host reference.

// tokenShift computes RWKV's shift-by-one delta over one chunk: delta[t] = prev(t) - x[t], where prev(0)
// is `prior` (the carried last-token hidden from a previous call, nil ⇒ the zero vector for a fresh
// sequence) and prev(t>0) = x[t-1] — fla.modules.token_shift.token_shift_ref's semantics
// (nn.ZeroPad2d((0,0,1,-1)) then shifted-x). Returns delta [L,D] and newPrior [D] (x's own last row,
// threaded into the next call so a chunked decode reproduces a one-pass prefill exactly, mirroring the
// mamba2/WKV7 carry invariant).
func tokenShift(x, prior []float32, L, D int) (delta, newPrior []float32) {
	delta = make([]float32, L*D)
	for i := range D {
		var p float32
		if prior != nil {
			p = prior[i]
		}
		delta[i] = p - x[i]
	}
	for t := 1; t < L; t++ {
		prevBase, curBase := (t-1)*D, t*D
		for i := range D {
			delta[curBase+i] = x[prevBase+i] - x[curBase+i]
		}
	}
	newPrior = append([]float32(nil), x[(L-1)*D:L*D]...)
	return delta, newPrior
}

// addcmulRows computes out[t,c] = x[t,c] + delta[t,c]*mix[c] over `rows` rows of width d — fla's
// torch.addcmul(hidden_states, delta, mix), the per-channel token-shift-mixed projection input (RWKV-7's
// xr/xw/xk/xv/xa/xg, and channel-mix's xk).
func addcmulRows(x, delta, mix []float32, rows, d int) []float32 {
	out := make([]float32, rows*d)
	for r := range rows {
		rb := r * d
		for i := range d {
			out[rb+i] = x[rb+i] + delta[rb+i]*mix[i]
		}
	}
	return out
}

// layerNormRows applies standard LayerNorm — mean-centred, variance-normalised, affine weight and OPTIONAL
// bias (b nil ⇒ norm_bias=false) — to each of the `rows` rows of x [rows,d]. RWKV-7's pre_norm/attn_norm/
// ffn_norm/final norm are real nn.LayerNorm (mean+variance), unlike mamba2's plain RMSNorm — a distinct
// primitive from mamba2/model.go's rmsNormRowsPlain, not a reuse of it.
func layerNormRows(x, w, b []float32, rows, d int, eps float32) []float32 {
	out := make([]float32, rows*d)
	for r := range rows {
		xr := x[r*d : (r+1)*d]
		var sum float64
		for i := range d {
			sum += float64(xr[i])
		}
		mean := sum / float64(d)
		var vs float64
		for i := range d {
			dv := float64(xr[i]) - mean
			vs += dv * dv
		}
		inv := 1.0 / math.Sqrt(vs/float64(d)+float64(eps))
		for i := range d {
			v := (float64(xr[i]) - mean) * inv * float64(w[i])
			if b != nil {
				v += float64(b[i])
			}
			out[r*d+i] = float32(v)
		}
	}
	return out
}

// groupNormHeads applies PyTorch's nn.GroupNorm with num_groups=H to x [rows, H*V] — normalising each
// head's V channels independently per row (its own mean+variance over exactly those V values), then an
// affine weight+bias indexed per FULL channel (h*V+j). This is RWKV-7's "g_norm" on the WKV7 read-out
// (fla.layers.rwkv7.RWKV7Attention's non-fuse_norm path: nn.GroupNorm(num_groups=H, num_channels=H*V,
// eps=head_dim*norm_eps)) — a different grouping from mamba2/gated-delta's per-head RMSNorm-then-gate.
func groupNormHeads(x, w, b []float32, rows, H, V int, eps float32) []float32 {
	out := make([]float32, rows*H*V)
	for r := range rows {
		base := r * H * V
		for h := range H {
			hb := base + h*V
			var sum float64
			for j := range V {
				sum += float64(x[hb+j])
			}
			mean := sum / float64(V)
			var vs float64
			for j := range V {
				dv := float64(x[hb+j]) - mean
				vs += dv * dv
			}
			inv := 1.0 / math.Sqrt(vs/float64(V)+float64(eps))
			cb := h * V
			for j := range V {
				v := (float64(x[hb+j]) - mean) * inv * float64(w[cb+j])
				if b != nil {
					v += float64(b[cb+j])
				}
				out[hb+j] = float32(v)
			}
		}
	}
	return out
}

// sigmoidF32 and tanhF32 evaluate in float64 and round back to float32 — the same higher-internal-
// precision convention as this package's matNT/WKV7F32 accumulation.
func sigmoidF32(x float32) float32 { return float32(1.0 / (1.0 + math.Exp(-float64(x)))) }
func tanhF32(x float32) float32    { return float32(math.Tanh(float64(x))) }
