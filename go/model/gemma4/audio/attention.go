// SPDX-Licence-Identifier: EUPL-1.2

package audio

import (
	"dappco.re/go/inference/model"
)

// attention.go is the host float32 port of the Gemma 4 Conformer chunked relative-position attention
// (engine/metal Gemma4AudioAttention.Forward). The tower runs it in f32: q/k/v/post/relative-K
// projections are bf16-widened GEMMs, the blocked-context windowing + Transformer-XL rel-shift + the
// validity mask are pure index remaps, the per-dim q-scale is pre-folded into QScalePerDim by the
// assembler, and the soft-cap uses tanh (host math.Tanh, not the engine's GPU kernel — the HF golden,
// not metal byte-identity, is the gate here).

// contextSize is chunk + past + future.
func contextSize(cfg model.LoadedAudioConfig) int {
	return cfg.ChunkSize + cfg.PastHorizon + cfg.FutureHorizon
}

// blockContext pads the time axis of x [T,H,D] by [past, future+chunk-1] (zeros) and unfolds
// overlapping windows strided by chunk → [nB,ctx,H,D]. Port of _extract_block_context.
func blockContext(x []float32, t, h, d, nB, chunk, past, future int) []float32 {
	ctx := chunk + past + future
	out := make([]float32, nB*ctx*h*d)
	for b := range nB {
		for c := range ctx {
			it := b*chunk + c - past // padded index = b*chunk + c; original time = padded - past.
			if it < 0 || it >= t {
				continue // zero pad
			}
			copy(out[((b*ctx+c)*h)*d:((b*ctx+c)*h+h)*d], x[(it*h)*d:(it*h+h)*d])
		}
	}
	return out
}

// relShiftInto is the Transformer-XL relative shift on [H,nB,chunk,P] → [H,nB,chunk,ctx] by padding the
// position axis to ctx+1, folding chunk·(ctx+1), truncating to chunk·ctx, refolding. Pure index remap.
func relShiftInto(out, x []float32, h, nB, chunk, p, ctx int) {
	padP := ctx + 1
	for hi := range h {
		for b := range nB {
			base := (hi*nB + b) * chunk
			for i := range chunk {
				for c := range ctx {
					fi := i*ctx + c // index into the folded chunk·(ctx+1) stream
					row, col := fi/padP, fi%padP
					var v float32
					if col < p {
						v = x[((base+row)*p)+col]
					}
					out[((base+i)*ctx)+c] = v
				}
			}
		}
	}
}

// blockedMask builds the [nB,chunk,ctx] validity mask: query q=blk·chunk+i may attend key
// kv=blk·chunk-past+j iff both in-sequence and kv∈[q-past, q+future]. Port of blockedMask.
func blockedMask(seqLen, nB, chunk, ctx, past, future int) []bool {
	m := make([]bool, nB*chunk*ctx)
	for b := range nB {
		for i := range chunk {
			q := b*chunk + i
			for j := range ctx {
				kv := b*chunk - past + j
				if q < seqLen && kv >= 0 && kv < seqLen && kv >= q-past && kv <= q+future {
					m[(b*chunk+i)*ctx+j] = true
				}
			}
		}
	}
	return m
}

// attention runs the Conformer chunked relative-position attention on f32 [T,hidden] → f32 [T,hidden].
func attention(x []float32, w model.LoadedAudioAttention, cfg model.LoadedAudioConfig) []float32 {
	hd := cfg.NumHeads * cfg.HeadDim
	t := len(x) / cfg.Hidden
	qf := linear(x, w.Q, t, cfg.Hidden, hd)
	kf := linear(x, w.K, t, cfg.Hidden, hd)
	vf := linear(x, w.V, t, cfg.Hidden, hd)

	merged := attentionCore(qf, kf, vf, w, cfg, t)
	return linear(merged, w.Post, t, hd, cfg.Hidden)
}

// attentionCore runs the f32 chunked relative-position attention math on the projections qf/kf/vf
// ([T,H,D]; q-scale/k-scale applied here), returning the merged context [T*hd] (pre output-projection).
func attentionCore(qf, kf, vf []float32, w model.LoadedAudioAttention, cfg model.LoadedAudioConfig, t int) []float32 {
	h, d := cfg.NumHeads, cfg.HeadDim
	hd := h * d
	chunk := cfg.ChunkSize
	nB := (t + chunk - 1) / chunk
	ctx := contextSize(cfg)
	past, future := cfg.PastHorizon, cfg.FutureHorizon

	// q *= QScalePerDim[d] (per-dim, broadcast over T and heads); k *= KScale.
	for i := 0; i < t*h; i++ {
		for dd := range d {
			qf[i*d+dd] *= w.QScalePerDim[dd]
		}
	}
	for i := range kf {
		kf[i] *= cfg.KScale
	}

	kc := blockContext(kf, t, h, d, nB, chunk, past, future)
	vc := blockContext(vf, t, h, d, nB, chunk, past, future)

	// relK = RelativeKProj.Forward(PosEmbed) = PosEmbed[P,hidden] · Wᵀ → [P,hd].
	relK := matMulMixedNT(w.PosEmbed, w.RelativeKProj, w.PosCount, cfg.Hidden, hd)

	mask := blockedMask(t, nB, chunk, ctx, past, future)
	merged := make([]float32, nB*chunk*hd)
	qh := make([]float32, nB*chunk*d)
	relKh := make([]float32, w.PosCount*d)
	bdShift := make([]float32, nB*chunk*ctx)
	kh := make([]float32, ctx*d)
	vh := make([]float32, ctx*d)
	invCap := float32(1) / cfg.LogitCap

	for head := range h {
		// gather this head's blocked q [nB·chunk, D] (zero for t>=T).
		clear(qh)
		for b := range nB {
			for i := range chunk {
				tt := b*chunk + i
				if tt < t {
					copy(qh[(b*chunk+i)*d:(b*chunk+i)*d+d], qf[(tt*h+head)*d:(tt*h+head)*d+d])
				}
			}
		}
		// bd[nB·chunk, P] = qh · relK_hᵀ, then per-block rel-shift to [nB·chunk, ctx].
		for p := 0; p < w.PosCount; p++ {
			copy(relKh[p*d:p*d+d], relK[(p*h+head)*d:(p*h+head)*d+d])
		}
		relKhT := transpose(relKh, w.PosCount, d)
		bd := matMulNN(qh, relKhT, nB*chunk, d, w.PosCount)
		relShiftInto(bdShift, bd, 1, nB, chunk, w.PosCount, ctx)

		for b := range nB {
			for c := range ctx {
				copy(kh[c*d:c*d+d], kc[((b*ctx+c)*h+head)*d:((b*ctx+c)*h+head)*d+d])
				copy(vh[c*d:c*d+d], vc[((b*ctx+c)*h+head)*d:((b*ctx+c)*h+head)*d+d])
			}
			khT := transpose(kh, ctx, d)
			ac := matMulNN(qh[b*chunk*d:(b+1)*chunk*d], khT, chunk, d, ctx)
			// soft-cap = LogitCap·tanh(logits/LogitCap).
			scaled := make([]float32, chunk*ctx)
			for i := range chunk {
				for j := range ctx {
					scaled[i*ctx+j] = (ac[i*ctx+j] + bdShift[(b*chunk+i)*ctx+j]) * invCap
				}
			}
			capped := tanh(scaled)
			for i := range chunk {
				for j := range ctx {
					s := capped[i*ctx+j] * cfg.LogitCap
					if !mask[(b*chunk+i)*ctx+j] {
						s = cfg.InvalidLogit
					}
					capped[i*ctx+j] = s
				}
			}
			probs := softmax(capped, chunk, ctx)
			blockOut := matMulNN(probs, vh, chunk, ctx, d)
			for i := range chunk {
				copy(merged[((b*chunk+i)*hd)+head*d:((b*chunk+i)*hd)+head*d+d], blockOut[i*d:i*d+d])
			}
		}
	}
	return merged[:t*hd]
}
