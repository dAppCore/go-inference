// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
)

// audio_attention.go ports the gemma4 Conformer chunked relative-position attention to the no-cgo
// path, BYTE-IDENTICAL to metal's Gemma4AudioAttention.Forward. The attention runs in float32 (metal
// .float()s q/k/v), so its matmuls go through MatMulF32 (the fused steel GEMM, byte-identical to
// metal.Matmul-f32) and its softmax through SoftmaxF32; the per-dim q-scale and tanh soft-cap use the
// byte-parity f32 Mul/Tanh; the blocked-context windowing, the Transformer-XL relShift, the validity
// mask and the masked select are host byte-copies/selects (no arithmetic, so byte-identical). The
// projections are bf16 (MatRowsBF16) widened to f32 (an exact AsType), and the result is rounded back
// to bf16 (f32ToBF16) before the bf16 output projection — exactly metal's dtype dance.

// AudioAttentionWeights holds the attention's weights: q/k/v/post projections (bf16, [H·D,hidden] /
// [hidden,H·D] for post), the relative-key projection (bf16, [H·D,hidden]), the per-dim q-scale
// (f32, [H·D] = q_scale·softplus(per_dim_scale), precomputed) and the sinusoid position table (f32,
// [P,hidden]). Projection clips (gradient clipping) are applied via the layer's ClipMin/ClipMax.
type AudioAttentionWeights struct {
	QProj, KProj, VProj, Post []byte
	// optional per-projection activation clamps (zero value = none, == metal nil InputMin/OutputMin).
	QClip, KClip, VClip, PostClip ClipPair
	RelativeKProj                 []byte
	QScalePerDim                  []float32 // [headDim] — broadcast over heads (metal's [1,1,1,headDim])
	PosEmbed                      []float32 // [P·hidden]
	PosCount                      int       // P
}

// audioContextSizeOf is chunk + past + future.
func audioContextSizeOf(cfg AudioConfig) int {
	return cfg.ChunkSize + cfg.PastHorizon + cfg.FutureHorizon
}

// audioBlockContextF32 pads the time axis of x [T, H, D] (fp32) by [past, future+chunk-1] (zeros) and
// unfolds overlapping windows strided by chunk → [nB, ctx, H, D] (fp32). Port of extractBlockContext.
func audioBlockContextF32(x []float32, T, H, D, nB, chunk, past, future int) []float32 {
	ctx := chunk + past + future
	out := make([]float32, nB*ctx*H*D)
	for b := range nB {
		for c := range ctx {
			// padded index = b*chunk + c; original time = padded - past.
			it := b*chunk + c - past
			if it < 0 || it >= T {
				continue // zero pad
			}
			copy(out[((b*ctx+c)*H)*D:((b*ctx+c)*H+H)*D], x[(it*H)*D:(it*H+H)*D])
		}
	}
	return out
}

// audioRelShiftF32Into is the Transformer-XL relative shift: [H, nB, chunk, P] → [H, nB, chunk, ctx] by
// padding the position axis to ctx+1, folding chunk·(ctx+1), truncating to chunk·ctx, refolding. Port
// of relShift (B=1). Pure index remap (byte-copy / zero-pad), so byte-identical.
func audioRelShiftF32Into(out, x []float32, H, nB, chunk, P, ctx int) {
	padP := ctx + 1
	for h := range H {
		for b := range nB {
			// folded[i*padP + p] = x[h,b,i,p] (p<P), else 0; then out[i,c] = folded[i*ctx + c].
			base := ((h*nB + b) * chunk)
			for i := range chunk {
				for c := range ctx {
					fi := i*ctx + c // index into the folded chunk·(ctx+1) stream
					row, col := fi/padP, fi%padP
					var v float32
					if col < P {
						v = x[((base+row)*P)+col]
					}
					out[((base+i)*ctx)+c] = v
				}
			}
		}
	}
}

// audioBlockedMask builds the [nB, chunk, ctx] validity mask: query q=blk·chunk+i may attend key
// kv=blk·chunk-past+j iff both in-sequence, kv∈[q-past, q+future], AND kv is a valid (non-padding)
// frame. Port of blockedMask AND'd with HF's _convert_4d_mask_to_blocked_5d key-padding term. validity
// is the per-soft-token (post-subsample) mask over [0,seqLen); nil means every position is valid (a
// fully-valid clip), byte-identical to the purely-positional mask (all-true validity is the same no-op).
func audioBlockedMask(seqLen, nB, chunk, ctx, past, future int, validity []bool) []bool {
	m := make([]bool, nB*chunk*ctx)
	for b := range nB {
		for i := range chunk {
			q := b*chunk + i
			for j := range ctx {
				kv := b*chunk - past + j
				if q < seqLen && kv >= 0 && kv < seqLen && kv >= q-past && kv <= q+future {
					if validity == nil || validity[kv] {
						m[(b*chunk+i)*ctx+j] = true
					}
				}
			}
		}
	}
	return m
}

// AudioAttention runs the Conformer attention on bf16 [T, hidden] (a standalone bf16 turn), returning
// bf16 [T, hidden] — byte-identical to metal's Gemma4AudioAttention.Forward with a bf16 input.
func AudioAttention(x []byte, w *AudioAttentionWeights, cfg AudioConfig) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	hd := cfg.NumHeads * cfg.HeadDim
	T := len(x) / (cfg.Hidden * bf16Size)
	proj := func(weight []byte, clip ClipPair) ([]float32, error) {
		p, err := clippedMatRowsBF16(weight, x, T, hd, cfg.Hidden, clip)
		if err != nil {
			return nil, err
		}
		return bf16ToF32Slice(p), nil
	}
	qf, err := proj(w.QProj, w.QClip)
	if err != nil {
		return nil, err
	}
	kf, err := proj(w.KProj, w.KClip)
	if err != nil {
		return nil, err
	}
	vf, err := proj(w.VProj, w.VClip)
	if err != nil {
		return nil, err
	}
	merged, err := audioAttentionCore(qf, kf, vf, w, cfg, T, nil)
	if err != nil {
		return nil, err
	}
	return clippedMatRowsBF16(w.Post, f32ToBf16Slice(merged), T, cfg.Hidden, hd, w.PostClip)
}

// AudioAttentionF32 runs the Conformer attention on fp32 [T, hidden] — the TOWER path (the layer feeds
// fp32 after the GC clamp promotes the activation). Projections + output projection are fp32 mixed-
// dtype matmuls (bf16 weights widened); the attention math is the same fp32 core. Byte-identical to
// metal's Gemma4AudioAttention.Forward with an fp32 input. validity is the optional per-soft-token
// (length T) key-padding mask; nil (a fully-valid clip) keeps the purely-positional blocked mask.
func AudioAttentionF32(x []float32, w *AudioAttentionWeights, cfg AudioConfig, validity []bool) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	hd := cfg.NumHeads * cfg.HeadDim
	T := len(x) / cfg.Hidden
	proj := func(weight []byte, clip ClipPair) ([]float32, error) {
		return clippedMatF32(x, weight, T, hd, cfg.Hidden, clip)
	}
	qf, err := proj(w.QProj, w.QClip)
	if err != nil {
		return nil, err
	}
	kf, err := proj(w.KProj, w.KClip)
	if err != nil {
		return nil, err
	}
	vf, err := proj(w.VProj, w.VClip)
	if err != nil {
		return nil, err
	}
	merged, err := audioAttentionCore(qf, kf, vf, w, cfg, T, validity)
	if err != nil {
		return nil, err
	}
	return clippedMatF32(merged, w.Post, T, cfg.Hidden, hd, w.PostClip)
}

// audioAttentionCore runs the fp32 chunked relative-position attention math on the fp32 projections
// qf/kf/vf ([T,H,D]; q-scale/k-scale applied here), returning the merged context [T*hd] fp32 (pre
// output-projection). Shared by the bf16 and fp32 entry points. validity (nil ⇒ all-valid) ANDs into
// the blocked mask so padding keys are never attended.
func audioAttentionCore(qf, kf, vf []float32, w *AudioAttentionWeights, cfg AudioConfig, T int, validity []bool) ([]float32, error) {
	H, D := cfg.NumHeads, cfg.HeadDim
	hd := H * D
	chunk := cfg.ChunkSize
	nB := (T + chunk - 1) / chunk
	ctx := audioContextSizeOf(cfg)
	past, future := cfg.PastHorizon, cfg.FutureHorizon

	// q *= QScalePerDim[d] (per-dim, broadcast over T and heads); k *= KScale.
	for i := 0; i < T*H; i++ {
		for d := range D {
			qf[i*D+d] *= w.QScalePerDim[d]
		}
	}
	for i := range kf {
		kf[i] *= cfg.KScale
	}

	// context windows for k,v: [nB, ctx, H, D].
	kc := audioBlockContextF32(kf, T, H, D, nB, chunk, past, future)
	vc := audioBlockContextF32(vf, T, H, D, nB, chunk, past, future)

	// relK = RelativeKProj.Forward(PosEmbed) = Matmul(PosEmbed, Transpose(weight)) → [P, H·D] (f32),
	// the bf16 weight widened, the NT steel kernel with split-K dispatch (a 1-ULP-sensitive shape).
	relK, err := MatMulF32NT(w.PosEmbed, bf16ToF32Slice(w.RelativeKProj), w.PosCount, cfg.Hidden, hd)
	if err != nil {
		return nil, err
	}

	// per query head h: matrix_ac[i,j] = Σ_d q[blk,i,h,d]·k_ctx[blk,j,h,d]; bd[i,p] = Σ_d q·relK[p,h,d];
	// logits = ac + relShift(bd); soft-cap; mask; softmax over ctx; out = Σ_j w[i,j]·v_ctx[blk,j,h,d].
	mask := audioBlockedMask(T, nB, chunk, ctx, past, future, validity)
	merged := make([]float32, nB*chunk*hd)
	qh := make([]float32, nB*chunk*D)
	relKh := make([]float32, w.PosCount*D)
	relKhT := make([]float32, D*w.PosCount)
	bd := make([]float32, nB*chunk*w.PosCount)
	bdShift := make([]float32, nB*chunk*ctx)
	kh := make([]float32, ctx*D)
	vh := make([]float32, ctx*D)
	khT := make([]float32, D*ctx)
	ac := make([]float32, chunk*ctx)
	scaled := make([]float32, chunk*ctx)
	capped := make([]float32, chunk*ctx)
	masked := make([]float32, chunk*ctx)
	probs := make([]float32, chunk*ctx)
	blockOut := make([]float32, chunk*D)
	for h := range H {
		// gather this head's blocked q [nB·chunk, D], context k/v [nB,ctx,D].
		clear(qh)
		for b := range nB {
			for i := range chunk {
				t := b*chunk + i
				if t < T {
					copy(qh[(b*chunk+i)*D:(b*chunk+i)*D+D], qf[(t*H+h)*D:(t*H+h)*D+D])
				}
			}
		}
		// bd over all positions then per-block relShift: bd[nB·chunk, P] = qh @ relK_hᵀ.
		for p := 0; p < w.PosCount; p++ {
			copy(relKh[p*D:p*D+D], relK[(p*H+h)*D:(p*H+h)*D+D])
		}
		transposeF32Into(relKhT, relKh, w.PosCount, D)
		bd, err = matMulF32Into(bd, qh, relKhT, nB*chunk, D, w.PosCount, false) // [nB·chunk, P]
		if err != nil {
			return nil, err
		}
		audioRelShiftF32Into(bdShift, bd, 1, nB, chunk, w.PosCount, ctx) // treat as [1,nB,chunk,P]→[1,nB,chunk,ctx]

		for b := range nB {
			for c := range ctx {
				copy(kh[c*D:c*D+D], kc[((b*ctx+c)*H+h)*D:((b*ctx+c)*H+h)*D+D])
				copy(vh[c*D:c*D+D], vc[((b*ctx+c)*H+h)*D:((b*ctx+c)*H+h)*D+D])
			}
			transposeF32Into(khT, kh, ctx, D)
			ac, err = matMulF32Into(ac, qh[b*chunk*D:(b+1)*chunk*D], khT, chunk, D, ctx, false) // [chunk, ctx]
			if err != nil {
				return nil, err
			}
			// soft-cap = LogitCap·tanh(logits/LogitCap), tanh via the GPU kernel (host math.Tanh is NOT
			// byte-identical to v_Tanhfloat32). MulScalar/Add are single f32 ops → byte-identical host-side.
			invCap := float32(1) / cfg.LogitCap
			for i := range chunk {
				for j := range ctx {
					scaled[i*ctx+j] = (ac[i*ctx+j] + bdShift[(b*chunk+i)*ctx+j]) * invCap
				}
			}
			if err := RunUnaryInto("v_Tanhfloat32float32", scaled, capped); err != nil {
				return nil, err
			}
			for i := range chunk {
				for j := range ctx {
					s := capped[i*ctx+j] * cfg.LogitCap
					if !mask[(b*chunk+i)*ctx+j] {
						s = cfg.InvalidLogit
					}
					masked[i*ctx+j] = s
				}
			}
			if err := softmaxF32Into(probs, masked, ctx, false); err != nil {
				return nil, err
			}
			blockOut, err = matMulF32Into(blockOut, probs, vh, chunk, ctx, D, false) // [chunk, D]
			if err != nil {
				return nil, err
			}
			for i := range chunk {
				copy(merged[((b*chunk+i)*hd)+h*D:((b*chunk+i)*hd)+h*D+D], blockOut[i*D:i*D+D])
			}
		}
	}

	// trim to T, round to bf16, Post projection.
	if len(merged) < T*hd {
		return nil, core.NewError("native.audioAttentionCore: internal merge size")
	}
	return merged[:T*hd], nil
}
