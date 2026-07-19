// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import core "dappco.re/go"

// train_real_shared.go extends the FD-gated real-arch layer reference (train_real_layer.go, #40)
// across KV-CACHE SHARING (#42) — the gemma4 E2B/E4B tail, where a CONSUMER layer
// (model.LayerSpec.KVShareFrom < own index) projects only its query and attends the OWNER layer's
// cached K/V rows (encAttnHalfShared / encAttnHalfKV are the decode ground truth):
//
//   - the owner's cached K row j is rope'd + K-normed AT THE OWNER'S WRITE POSITION j
//     (encAttnHalfKV projects into the cache row, then K-norm + rope in place) — the owner
//     tape's kr rows;
//   - the owner's cached V row j is the post-value-norm value (or, on a K==V owner, the
//     value-normed RAW key projection) — the owner tape's v rows;
//   - the consumer applies its OWN pre-attn norm, q projection, per-head Q-norm and rope (at its
//     own row position), attends the owner's rows over its own window, then its own o-proj,
//     post-attn norm, residual, MLP, PLE gate and layer scalar (the standard tail).
//
// The chain helpers here mirror that wiring host-pure (f32 storage, f64 accumulation) and are the
// forward/backward the per-layer LoRA trainer walks on a shared-KV stack; every gradient claim is
// finite-difference-gated in train_real_shared_test.go through a genuinely shared multi-layer chain.

// realConsumerForwardTape runs ONE KV-share consumer layer forward: h [T,DModel] through the
// consumer attention half (own q only) over the OWNER's cached rows extK/extV
// ([T, KVHeads·HeadDim] each — the owner tape's kr and v), then the shared tail. The tape's kr/v
// alias extK/extV (read-only) so the backward's head gathers work unchanged; k0/kn/v0 stay nil (the
// consumer has no key/value stations of its own — encAttnHalfShared applies none).
func realConsumerForwardTape(h []float32, L *RealTrainLayerF32, extK, extV []float32, wQ, wO, wGate, wUp, wDown []float32) (*realLayerTape, error) {
	if err := L.validate(); err != nil {
		return nil, err
	}
	if !L.SharesKV {
		return nil, core.NewError("native.realConsumerForwardTape: layer does not declare SharesKV — use realLayerForwardTape")
	}
	T, D, d := L.T, L.DModel, L.HeadDim
	qDim, kvDim := L.Heads*d, L.KVHeads*d
	if len(h) != T*D {
		return nil, core.NewError("native.realConsumerForwardTape: h must be [T,DModel]")
	}
	if len(extK) != T*kvDim || len(extV) != T*kvDim {
		return nil, core.NewError("native.realConsumerForwardTape: extK/extV must be [T, KVHeads·HeadDim] (the owner's cached rows)")
	}
	tp := &realLayerTape{}

	// consumer attention half (encAttnHalfShared): pre-norm → OWN q projection → per-head Q-norm →
	// rope(q) at the row position → windowed GQA SDPA over the owner's cached rows (already
	// K-normed + rope'd at the owner's write positions, already value-normed — read AS-IS, no
	// consumer-side key/value op) → o-proj → (post-attn norm) → residual.
	tp.normed = rmsNormForwardF32(h, L.AttnNormW, T, D, L.Eps)
	tp.q0 = hostLinearF32(tp.normed, wQ, T, D, qDim)
	tp.qn = tp.q0
	if len(L.QNormW) > 0 {
		tp.qn = rmsNormForwardF32(tp.q0, L.QNormW, T*L.Heads, d, L.Eps)
	}
	tp.qr = make([]float32, T*qDim)
	for i := range T {
		copy(tp.qr[i*qDim:(i+1)*qDim], realRopeForwardF32(tp.qn[i*qDim:(i+1)*qDim], i, L.Heads, d, L.RopePairHalf, L.RopeInvFreq, L.RopeScale))
	}
	tp.kr, tp.v = extK, extV
	gqa := L.Heads / L.KVHeads
	tp.o = make([]float32, T*qDim)
	tp.probs = make([][]float32, L.Heads)
	for hh := range L.Heads {
		hk := hh / gqa
		qh := gatherHeadF32(tp.qr, T, L.Heads, d, hh)
		kh := gatherHeadF32(tp.kr, T, L.KVHeads, d, hk)
		vh := gatherHeadF32(tp.v, T, L.KVHeads, d, hk)
		p := hostSDPAProbsF32(qh, kh, L)
		tp.probs[hh] = p
		oh := make([]float32, T*d)
		for i := range T {
			for j := L.attnLow(i); j <= i; j++ {
				pij := float64(p[i*T+j])
				for c := range d {
					oh[i*d+c] += float32(pij * float64(vh[j*d+c]))
				}
			}
		}
		scatterAddHeadF32(tp.o, oh, T, L.Heads, d, hh)
	}
	tp.attnBranchPre = hostLinearF32(tp.o, wO, T, qDim, D)
	tp.attnBranch = tp.attnBranchPre
	if len(L.PostAttnNormW) > 0 {
		tp.attnBranch = rmsNormForwardF32(tp.attnBranchPre, L.PostAttnNormW, T, D, L.Eps)
	}
	tp.h1 = make([]float32, T*D)
	for i := range tp.h1 {
		tp.h1[i] = h[i] + tp.attnBranch[i]
	}
	realLayerTailForward(tp, L, wGate, wUp, wDown)
	return tp, nil
}

// realConsumerBackward walks a consumer tape in reverse: given dout [T,DModel] it returns the
// consumer's OWN weight gradients (q/o/gate/up/down — a consumer has no k/v), dH to its input, and
// dExtK/dExtV — the gradients w.r.t. the OWNER's cached post-rope K / post-value-norm V rows it
// attended. The chain backward routes dExtK/dExtV; this function only produces them.
func realConsumerBackward(dout, h []float32, L *RealTrainLayerF32, tp *realLayerTape, wQ, wO, wGate, wUp, wDown []float32) (*realLayerGrads, error) {
	if len(dout) != L.T*L.DModel {
		return nil, core.NewError("native.realConsumerBackward: dout must be [T,DModel]")
	}
	T, D, d := L.T, L.DModel, L.HeadDim
	qDim, kvDim := L.Heads*d, L.KVHeads*d
	g := &realLayerGrads{}

	dH1, err := realLayerTailBackward(dout, L, tp, wGate, wUp, wDown, g)
	if err != nil {
		return nil, err
	}

	// consumer attention half backward: residual → (post-attention sandwich norm) → o-proj →
	// SDPA core (dScores route to dQ locally AND to the owner's cached rows as dExtK/dExtV) →
	// rope → Q-norm → q projection → pre-attn norm → residual join.
	dAttnBranchPre := dH1
	if len(L.PostAttnNormW) > 0 {
		var nerr error
		dAttnBranchPre, _, nerr = RMSNormBackwardF32(dH1, tp.attnBranchPre, L.PostAttnNormW, T, D, L.Eps)
		if nerr != nil {
			return nil, nerr
		}
	}
	var dO []float32
	dO, g.dWO = hostLinearBackwardF32(dAttnBranchPre, tp.o, wO, T, qDim, D)
	gqa := L.Heads / L.KVHeads
	dQr := make([]float32, T*qDim)
	g.dExtK = make([]float32, T*kvDim)
	g.dExtV = make([]float32, T*kvDim)
	for hh := range L.Heads {
		hk := hh / gqa
		qh := gatherHeadF32(tp.qr, T, L.Heads, d, hh)
		kh := gatherHeadF32(tp.kr, T, L.KVHeads, d, hk)
		vh := gatherHeadF32(tp.v, T, L.KVHeads, d, hk)
		doh := gatherHeadF32(dO, T, L.Heads, d, hh)
		p := tp.probs[hh]

		dqh := make([]float32, T*d)
		dkh := make([]float32, T*d)
		dvh := make([]float32, T*d)
		for i := range T {
			lo := L.attnLow(i)
			// dV_j += P[i,j]·dO_i ; dP[i,j] = dO_i·v_j
			dp := make([]float64, i-lo+1)
			var dot float64 // Σ_j P[i,j]·dP[i,j] (softmax VJP reduction)
			for j := lo; j <= i; j++ {
				var acc float64
				for c := range d {
					acc += float64(doh[i*d+c]) * float64(vh[j*d+c])
					dvh[j*d+c] += float32(float64(p[i*T+j]) * float64(doh[i*d+c]))
				}
				dp[j-lo] = acc
				dot += float64(p[i*T+j]) * acc
			}
			// dS[i,j] = P[i,j]·(dP[i,j] − dot); dQ_i += dS·K_j·scale; dK_j += dS·Q_i·scale
			for j := lo; j <= i; j++ {
				ds := float64(p[i*T+j]) * (dp[j-lo] - dot) * float64(L.AttnScale)
				for c := range d {
					dqh[i*d+c] += float32(ds * float64(kh[j*d+c]))
					dkh[j*d+c] += float32(ds * float64(qh[i*d+c]))
				}
			}
		}
		scatterAddHeadF32(dQr, dqh, T, L.Heads, d, hh)
		scatterAddHeadF32(g.dExtK, dkh, T, L.KVHeads, d, hk)
		scatterAddHeadF32(g.dExtV, dvh, T, L.KVHeads, d, hk)
	}
	dQ0 := make([]float32, T*qDim)
	for i := range T {
		copy(dQ0[i*qDim:(i+1)*qDim], realRopeBackwardF32(dQr[i*qDim:(i+1)*qDim], i, L.Heads, d, L.RopePairHalf, L.RopeInvFreq, L.RopeScale))
	}
	if len(L.QNormW) > 0 {
		dQ0, _, err = QKNormBackwardF32(dQ0, tp.q0, L.QNormW, T, L.Heads, d, L.Eps)
		if err != nil {
			return nil, err
		}
	}
	dNormedQ, dWQ := hostLinearBackwardF32(dQ0, tp.normed, wQ, T, D, qDim)
	g.dWQ = dWQ
	dHNorm, _, err := RMSNormBackwardF32(dNormedQ, h, L.AttnNormW, T, D, L.Eps)
	if err != nil {
		return nil, err
	}
	g.dH = make([]float32, T*D)
	for i := range g.dH {
		g.dH[i] = dH1[i] + dHNorm[i]
	}
	return g, nil
}

// validateShareTopology checks a chain's KV-share map once: shareFrom[i] == i declares an
// owner/dense layer; shareFrom[i] < i declares a consumer of that earlier OWNER (an owner never
// consumes — DeriveLayers' rule), with each layer's SharesKV flag consistent and the consumer's
// cache geometry (KVHeads·HeadDim) matching its owner's (the SDPA reads the owner's rows with the
// consumer's geometry — encAttnHalfShared — so a mismatch is un-mirrorable, and un-servable).
func validateShareTopology(layers []*RealTrainLayerF32, shareFrom []int) error {
	if len(shareFrom) != len(layers) {
		return core.NewError("native.validateShareTopology: shareFrom must carry one entry per layer")
	}
	for li, L := range layers {
		own := shareFrom[li]
		if own == li {
			if L.SharesKV {
				return core.NewError(core.Concat("native.validateShareTopology: layer ", core.Sprintf("%d", li),
					" declares SharesKV but shareFrom names it an owner"))
			}
			continue
		}
		if own < 0 || own >= li {
			return core.NewError(core.Concat("native.validateShareTopology: layer ", core.Sprintf("%d", li),
				" shares layer ", core.Sprintf("%d", own), " — the owner must be an EARLIER layer"))
		}
		if !L.SharesKV {
			return core.NewError(core.Concat("native.validateShareTopology: layer ", core.Sprintf("%d", li),
				" shares layer ", core.Sprintf("%d", own), "'s cache but does not declare SharesKV"))
		}
		if shareFrom[own] != own {
			return core.NewError(core.Concat("native.validateShareTopology: layer ", core.Sprintf("%d", li),
				" shares layer ", core.Sprintf("%d", own), ", which is itself a consumer — owners own"))
		}
		O := layers[own]
		if O.KVHeads != L.KVHeads || O.HeadDim != L.HeadDim {
			return core.NewError(core.Concat("native.validateShareTopology: layer ", core.Sprintf("%d", li),
				" reads layer ", core.Sprintf("%d", own), "'s cache with a different KV geometry (",
				core.Sprintf("%d×%d", L.KVHeads, L.HeadDim), " vs ", core.Sprintf("%d×%d", O.KVHeads, O.HeadDim), ")"))
		}
	}
	return nil
}

// realSharedChainForward runs the host layer chain over x [T,DModel] with per-layer weight sets,
// dispatching each layer by its share role: owners/dense through realLayerForwardTape (their tapes
// bank the cached rows), consumers through realConsumerForwardTape reading the OWNER tape's
// kr (post-rope K) and v (post-value-norm V) — exactly the rows encAttnHalfKV wrote and
// encAttnHalfShared attends. Returns each layer's input hidden and forward tape. The forward is
// exact for ANY adapter placement (the cached rows are recomputed under the current effective
// weights every call).
func realSharedChainForward(x []float32, layers []*RealTrainLayerF32, shareFrom []int, sets []layerWeightSet) (inputs [][]float32, tapes []*realLayerTape, err error) {
	if err := validateShareTopology(layers, shareFrom); err != nil {
		return nil, nil, err
	}
	if len(sets) != len(layers) {
		return nil, nil, core.NewError("native.realSharedChainForward: one weight set per layer required")
	}
	inputs = make([][]float32, len(layers))
	tapes = make([]*realLayerTape, len(layers))
	for li, L := range layers {
		inputs[li] = x
		s := sets[li]
		var tp *realLayerTape
		if shareFrom[li] != li {
			ownTape := tapes[shareFrom[li]]
			tp, err = realConsumerForwardTape(x, L, ownTape.kr, ownTape.v, s.wQ, s.wO, s.wGate, s.wUp, s.wDown)
		} else {
			tp, err = realLayerForwardTape(x, L, s.wQ, s.wK, s.wV, s.wO, s.wGate, s.wUp, s.wDown)
		}
		if err != nil {
			return nil, nil, err
		}
		tapes[li] = tp
		x = tp.out
	}
	return inputs, tapes, nil
}

// realSharedChainBackward walks the chain top-down from dout (the gradient at the LAST layer's
// output), calling visit(li, g) with each layer's gradients — dense/owner layers carry the full
// seven projection gradients, consumers carry q/o/gate/up/down plus dExtK/dExtV — and returns the
// gradient at the chain input.
//
// STAGE-1 EXACTNESS RULE (cached rows treated as CONSTANTS — this backward does not yet route a
// consumer's dExtK/dExtV back into its owner): an adapter at (layer li, target t) influences owner
// o's cached rows iff li < o (ANY target moves the owner's input hidden and therefore its cache),
// or li == o with t ∈ {k_proj, v_proj} (the projections that write the cache; on a K==V owner
// k_proj writes both). Discarding dExtK/dExtV is therefore EXACT iff, for EVERY owner with at
// least one consumer, no adapter sits strictly below it and none of its own k_proj/v_proj is
// adapted — equivalently: every adapter sits at li ≥ the HIGHEST consumed owner, with k_proj/
// v_proj excluded at that owner (consumer-side q/o/mlp adapters, at-or-above the last consumed
// owner). validateSharedKVAdapterSubset enforces exactly this rule at trainer open;
// train_real_shared_test.go finite-difference-gates it through a genuinely shared stack. The dH
// returned at the chain input is NOT exact under sharing (the input feeds the lowest owner's
// cache) — it is unused by the trainer (the embedding is frozen) and must not be consumed while
// this stage-1 boundary stands.
func realSharedChainBackward(dout []float32, inputs [][]float32, tapes []*realLayerTape, layers []*RealTrainLayerF32, shareFrom []int, sets []layerWeightSet, visit func(li int, g *realLayerGrads) error) ([]float32, error) {
	if err := validateShareTopology(layers, shareFrom); err != nil {
		return nil, err
	}
	if len(inputs) != len(layers) || len(tapes) != len(layers) || len(sets) != len(layers) {
		return nil, core.NewError("native.realSharedChainBackward: inputs/tapes/sets must carry one entry per layer")
	}
	dH := dout
	for li := len(layers) - 1; li >= 0; li-- {
		L := layers[li]
		s := sets[li]
		var g *realLayerGrads
		var err error
		if shareFrom[li] != li {
			g, err = realConsumerBackward(dH, inputs[li], L, tapes[li], s.wQ, s.wO, s.wGate, s.wUp, s.wDown)
		} else {
			g, err = realLayerBackward(dH, inputs[li], L, tapes[li], s.wQ, s.wK, s.wV, s.wO, s.wGate, s.wUp, s.wDown)
		}
		if err != nil {
			return nil, err
		}
		if visit != nil {
			if verr := visit(li, g); verr != nil {
				return nil, verr
			}
		}
		dH = g.dH
	}
	return dH, nil
}

// consumedOwnerSet returns the owner indices that have at least one consumer — the layers whose
// cached rows create cross-layer gradient paths.
func consumedOwnerSet(shareFrom []int) map[int]bool {
	owners := map[int]bool{}
	for li, own := range shareFrom {
		if own != li {
			owners[own] = true
		}
	}
	return owners
}

// validateSharedKVAdapterSubset enforces the stage-1 exactness rule (realSharedChainBackward's
// doc) on a resolved adapter set: on a stack with KV-cache sharing, every adapter must sit
// AT-OR-ABOVE every consumed owner, and a consumed owner's own k_proj/v_proj must not be adapted —
// the combinations where the cached rows are constants of every trainable parameter, so the
// consumer backward is exact WITHOUT the owner-routed dK/dV path. Anything else refuses, naming
// the missing path (the stage-2 backward: dK/dV from every consumer routed back through the
// owner's k_proj/v_proj and hidden). A share-free stack passes vacuously.
func validateSharedKVAdapterSubset(shareFrom []int, adapters []*layerLoRAAdapter) error {
	owners := consumedOwnerSet(shareFrom)
	if len(owners) == 0 {
		return nil
	}
	for _, ad := range adapters {
		for own := range owners {
			below := ad.layer < own
			cacheWriting := ad.layer == own && (ad.target == ProjK || ad.target == ProjV)
			if !below && !cacheWriting {
				continue
			}
			reason := "its k_proj/v_proj write the rows layer "
			if below {
				reason = "every target moves the hidden that feeds the cached rows layer "
			}
			return core.NewError(core.Concat(
				"native.NewLoRATrainer: adapter (layer ", core.Sprintf("%d", ad.layer), ", ", ad.target,
				") can influence layer ", core.Sprintf("%d", own), "'s SHARED KV cache (", reason,
				core.Sprintf("%d", own), "'s consumers attend) — the owner-routed cross-layer gradient path",
				" (dK/dV from every consumer back through the owner's k_proj/v_proj and hidden) is not wired;",
				" restrict per-layer adapters to layers at-or-above every consumed owner (k_proj/v_proj",
				" excluded on the owner itself), or train the ", loraTargetHead, " adapter"))
		}
	}
	return nil
}
