// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// train_trainer_layers.go is stage 3 of #40: the PER-LAYER projection LoRA path of LoRATrainer,
// wired ONLY for model shapes whose every layer feature is covered by the FD-gated real-arch
// reference (train_real_layer.go + the owner-routed KV-share chain of train_real_shared.go, #42).
// The #31 refusal REMAINS for everything else — a shape with any un-gated feature (MoE, recurrent
// mixers, projection biases, logit soft-caps, …) refuses per-layer targets loudly, naming the
// blocking feature (validatePerLayerLoRAShape).
//
// The training maths: a per-layer adapter changes EVERY layer's forward, so — unlike the head
// seam, where the frozen capture is exact — each step re-runs the layer chain HOST-SIDE
// (realLayerForwardTape) under the CURRENT effective weights, seeded by the engine's own token
// embeddings and per-layer inputs (ForwardCaptureHiddens + the session's PLE closure: the frozen,
// engine-computed inputs). The backward walks the chain top-down (realLayerBackward), folding each
// adapted projection's weight gradient onto its LoRA factors (LoRAFactorGradsF32). With B = 0 the
// host chain reproduces the engine's own captured hiddens (the parity anchor the e2e test pins),
// so the reference mirrors the engine, not a paper. batch.LossMask is honoured with exactly the
// head path's semantics (#39): masked positions contribute zero loss and zero gradient, means
// divide by unmasked counts.

// loraLayerTargets is the canonical per-layer projection vocabulary (the TargetKeys spelling).
var loraLayerTargets = map[string]bool{
	ProjQ: true, ProjK: true, ProjV: true, ProjO: true, ProjGate: true, ProjUp: true, ProjDown: true,
}

// loraTargetMode classifies a LoRA config's TargetKeys: the HEAD mode (empty, or every key
// lm_head — the #31-era trainer), or the LAYERS mode (every key a canonical per-layer projection).
// A mix of head and layer keys, or an unknown key, is refused — the caller trains the two seams
// separately, and a typo never trains silently as something else.
func loraTargetMode(cfg inference.LoRAConfig) (perLayer bool, err error) {
	head, layers := 0, 0
	for _, key := range cfg.TargetKeys {
		switch {
		case key == loraTargetHead:
			head++
		case loraLayerTargets[key]:
			layers++
		default:
			return false, core.NewError(core.Concat(
				"native.NewLoRATrainer: unknown LoRA target ", core.Sprintf("%q", key),
				" — supported: ", loraTargetHead, " (the head) or the per-layer projections (",
				ProjQ, " ", ProjK, " ", ProjV, " ", ProjO, " ", ProjGate, " ", ProjUp, " ", ProjDown, ")"))
		}
	}
	if head > 0 && layers > 0 {
		return false, core.NewError(core.Concat(
			"native.NewLoRATrainer: TargetKeys ", core.Sprintf("%v", cfg.TargetKeys),
			" mix the head adapter (", loraTargetHead, ") with per-layer projections — train the two seams separately"))
	}
	return layers > 0, nil
}

// validatePerLayerLoRAShape decides whether a loaded model's shape is FULLY covered by the
// FD-gated real-arch reference — the stage-3 honesty gate. Every feature named here either has a
// green finite-difference gate in train_real_layer_test.go (and is accepted) or has none (and
// refuses per-layer training, naming itself). The head adapter (lm_head) remains available for
// every shape.
func validatePerLayerLoRAShape(tm *NativeTokenModel) error {
	refuse := func(feature string) error {
		return core.NewError(core.Concat(
			"native.NewLoRATrainer: per-layer projection LoRA is not wired for this model shape — ",
			feature, " has no finite-difference-gated backward (train_real_layer.go); ",
			"the ", loraTargetHead, " adapter trains on every shape"))
	}
	if tm == nil || tm.bf16 == nil {
		return refuse("a non-bf16 (quantised) base")
	}
	arch := tm.arch
	if arch.Experts > 0 {
		return refuse("the MoE feed-forward")
	}
	// arch.SoftCap is WIRED: the trainer's head forward caps like serving and the CE gradient is
	// scaled by the cap's derivative (train_softcap.go) — FD-gated by TestSoftcapHeadBackward_FD.
	if arch.LogitsScaling != 0 || arch.LogitScale != 0 {
		return refuse("final-logit scaling in the head backward")
	}
	if arch.AttnOutputGate {
		return refuse("the attention output gate")
	}
	if arch.ALiBi || arch.LearnedAbsolutePositions {
		return refuse("non-rotary position encoding")
	}
	if arch.ParallelResidual {
		return refuse("the parallel-residual block order")
	}
	if arch.LayerNorm || arch.NonParametricLayerNorm || arch.NormPlacement == model.NormPlacementPost {
		return refuse("centred/post-placement LayerNorm")
	}
	if arch.QKVClip != 0 {
		return refuse("the projected-QKV clamp")
	}
	if arch.ResidualMultiplier != 0 && arch.ResidualMultiplier != 1 {
		return refuse("the residual-branch multiplier")
	}
	if len(arch.RopeShortFreqs) > 0 || arch.RopeOriginalContext != 0 {
		return refuse("position-dependent (LongRoPE) rope switching")
	}
	for li := range arch.Layer {
		spec := arch.Layer[li]
		if spec.Mixer != model.MixerAttention {
			return refuse("a non-attention sequence mixer")
		}
		if spec.MoE {
			return refuse("the MoE feed-forward")
		}
		if spec.KVShareFrom != li {
			// KV-cache sharing itself is covered (#42): a consumer layer mirrors
			// encAttnHalfShared (own q only, the owner's cached rows), FD-gated in
			// train_real_shared_test.go. Only a topology the decode itself could not serve —
			// and the mirror could not reproduce — refuses here; WHICH (layer, target)
			// adapter combinations train on a shared stack is the separate, config-dependent
			// gate (validateSharedKVAdapterSubset).
			own := spec.KVShareFrom
			if own < 0 || own >= li || arch.Layer[own].KVShareFrom != own {
				return refuse(core.Concat("a malformed KV-share topology (layer ", core.Sprintf("%d", li),
					" names owner ", core.Sprintf("%d", own), ", which is not an earlier cache owner)"))
			}
			ospec := arch.Layer[own]
			if kvHeadsOf(spec, arch.KVHeads) != kvHeadsOf(ospec, arch.KVHeads) ||
				headDimOf(spec, arch.HeadDim) != headDimOf(ospec, arch.HeadDim) {
				return refuse(core.Concat("a KV-share consumer (layer ", core.Sprintf("%d", li),
					") whose cache geometry differs from its owner's (layer ", core.Sprintf("%d", own), ")"))
			}
		}
		if li < len(tm.bf16.Layers) {
			w := &tm.bf16.Layers[li]
			if len(w.BQ) > 0 || len(w.BK) > 0 || len(w.BV) > 0 {
				return refuse("additive q/k/v projection biases")
			}
			if w.MoE != nil || w.GatedDelta != nil {
				return refuse("a MoE/recurrent layer body")
			}
		}
	}
	// Per-layer head-dim switching (gemma4 global layers run hd 512 vs 256 sliding) is WIRED: the
	// template builder resolves each layer's geometry exactly as the decode loop does
	// (headDimOf/kvHeadsOf per spec) and each class's rope from the engine's own table sources
	// (realRopeInvFreqs for sliding; the inverted finite prefix of globalRopePeriodsFromFolded —
	// the globalRopeFreqs construction — for the proportional Inf-padded global form). Receipts:
	// the mixed-geometry chain FD gate (TestRealSharedChainBackward_MixedHeadDim — all seven
	// targets + dH0 across an hd/kv switch with consumers of both owner classes) and the real-E2B
	// host probe (TestRealChainE2BMirrorVsReference_Good — the live loader path's templates match
	// the ecosystem reference per layer, worst cosine 1.000000 across all four layer classes).
	// The #42-era refusal here traced to the live anchor's engine-capture side, not the mirror.
	return nil
}

// layerLoRAAdapter is one trainable per-layer projection adapter: the factors, their optimiser
// state, and the (layer, target) address.
type layerLoRAAdapter struct {
	layer      int
	target     string
	out, in    int
	a, b       []float32
	optA, optB *AdamW
}

// buildRealLayerTemplates widens the loaded bf16 model into per-layer RealTrainLayerF32 templates
// (frozen weights + per-layer geometry + rope shape). T and PLEInput are per-sequence and filled by
// the step; everything else is fixed at open.
func buildRealLayerTemplates(g *BF16Model, arch model.Arch) ([]*RealTrainLayerF32, error) {
	layers := make([]*RealTrainLayerF32, len(g.Layers))
	for li := range g.Layers {
		w := &g.Layers[li]
		spec := arch.Layer[li]
		lhd := headDimOf(spec, arch.HeadDim)
		lkv := kvHeadsOf(spec, arch.KVHeads)
		dFF := arch.FF
		if w.DFF > 0 {
			dFF = w.DFF
		}
		L := &RealTrainLayerF32{
			AttnNormW: bf16ToF32Slice(w.AttnNormW),
			WQ:        bf16ToF32Slice(w.WQ),
			WO:        bf16ToF32Slice(w.WO),
			MLPNormW:  bf16ToF32Slice(w.MLPNormW),
			WGate:     bf16ToF32Slice(w.WGate),
			WUp:       bf16ToF32Slice(w.WUp),
			WDown:     bf16ToF32Slice(w.WDown),
			DModel:    arch.Hidden, DFF: dFF,
			Heads: arch.Heads, KVHeads: lkv, HeadDim: lhd,
			ValueNorm: arch.ValueNorm,
			AttnScale: arch.AttnScale,
			Eps:       arch.Eps,
			RopeScale: arch.RopeScale,
		}
		if spec.KVShareFrom != li {
			// a KV-share CONSUMER (#42): projects only its query and attends the owner's cached
			// rows — no k/v projection, no K-norm, no value-norm of its own (encAttnHalfShared;
			// the checkpoint carries no such tensors for it). The chain forward feeds it the
			// owner tape's rows.
			L.SharesKV = true
			L.ValueNorm = false
		} else {
			L.WK = bf16ToF32Slice(w.WK)
			if len(w.WV) > 0 {
				L.WV = bf16ToF32Slice(w.WV) // nil WV = the K==V layer
			}
			if len(w.KNormW) > 0 {
				L.KNormW = bf16ToF32Slice(w.KNormW)
			}
		}
		if len(w.QNormW) > 0 {
			L.QNormW = bf16ToF32Slice(w.QNormW)
		}
		if len(w.PostAttnNormW) > 0 {
			L.PostAttnNormW = bf16ToF32Slice(w.PostAttnNormW)
		}
		if len(w.PostFFNormW) > 0 {
			L.PostFFNormW = bf16ToF32Slice(w.PostFFNormW)
		}
		if len(w.LayerScalarW) > 0 {
			L.LayerScalar = bf16ToF32Slice(w.LayerScalarW)[0]
		}
		if g.HasPLE() && len(w.PerLayerGate) > 0 {
			L.PLIDim = arch.PerLayerInputHidden
			L.PLEGateW = bf16ToF32Slice(w.PerLayerGate)
			L.PLEProjW = bf16ToF32Slice(w.PerLayerProjection)
			L.PLEPostNormW = bf16ToF32Slice(w.PostPerLayerInputNormW)
		}
		// rope shape: sliding layers use the base-derived local rope (standard pairing over
		// RotaryDimLocal); global layers use the arch's explicit spectrum when it carries one
		// (YaRN — standard pairing), the proportional FULL-HEAD pairing when the rotary is
		// partial (globalRopePeriodsFromFolded's shape), or the plain base-derived rope.
		if spec.Attention == model.SlidingAttention {
			rot := arch.RotaryDimLocal
			if rot <= 0 || rot > lhd {
				rot = lhd
			}
			L.RopePairHalf = rot / 2
			L.RopeInvFreq = realRopeInvFreqs(rot, arch.RopeLocalBase)
			L.Window = arch.SlidingWindow
		} else {
			rot := arch.RotaryDim
			if rot <= 0 || rot > lhd {
				rot = lhd
			}
			switch {
			case len(arch.RopeFreqs) > 0:
				L.RopePairHalf = rot / 2
				L.RopeInvFreq = append([]float32(nil), arch.RopeFreqs...)
			case rot < lhd:
				// gemma4 proportional partial rotary: pairs span the whole head; the arch's
				// RopeBase is pre-folded to raw^(rot/headDim) (config.go), so recover the raw
				// theta and build the full-head spectrum (rope_freqs.go).
				periods := globalRopePeriodsFromFolded(lhd, rot, arch.RopeBase)
				inv := make([]float32, 0, rot/2)
				for _, p := range periods[:rot/2] {
					inv = append(inv, 1/p)
				}
				L.RopePairHalf = lhd / 2
				L.RopeInvFreq = inv
			default:
				L.RopePairHalf = lhd / 2
				L.RopeInvFreq = realRopeInvFreqs(lhd, arch.RopeBase)
			}
		}
		if L.RopeScale == 0 {
			L.RopeScale = 1
		}
		layers[li] = L
	}
	return layers, nil
}

// buildLayerAdapters resolves the requested projection targets against the model's layers into the
// trainable adapter set. Every (layer, target) pair that exists is adapted; the ecosystem
// absences — v_proj on a K==V layer (no value projection), and k_proj/v_proj on a KV-share
// consumer (it attends the owner's cache and carries no k/v tensors at all) — are skipped (the
// saved adapter shows exactly what trained). A request that resolves to NOTHING is refused.
func buildLayerAdapters(layers []*RealTrainLayerF32, targets []string, rank int, lr float32) ([]*layerLoRAAdapter, error) {
	seen := map[string]bool{}
	var adapters []*layerLoRAAdapter
	for _, target := range targets {
		if seen[target] {
			continue // a repeated key is one adapter, not two
		}
		seen[target] = true
		for li, L := range layers {
			if (target == ProjK || target == ProjV) && L.SharesKV {
				continue // a KV-share consumer has no k/v projection — the owner's k/v feed the shared rows
			}
			if target == ProjV && L.WV == nil {
				continue // the K==V layer has no v_proj — the documented ecosystem skip
			}
			out, in, err := L.projDims(target)
			if err != nil {
				return nil, err
			}
			adapters = append(adapters, &layerLoRAAdapter{
				layer: li, target: target, out: out, in: in,
				a:    initLoRAFactorA(rank*in, in),
				b:    make([]float32, out*rank),
				optA: NewAdamW(rank*in, lr, 0),
				optB: NewAdamW(out*rank, lr, 0),
			})
		}
	}
	if len(adapters) == 0 {
		return nil, core.NewError("native.NewLoRATrainer: the requested per-layer targets resolve to no trainable projection on this model")
	}
	return adapters, nil
}

// layerWeightSet is one layer's weight slices for a step — the frozen template weights with the
// layer's adapted projections substituted by their CURRENT effective weights.
type layerWeightSet struct {
	wQ, wK, wV, wO, wGate, wUp, wDown []float32
}

// effectiveWeightSets computes each layer's weight set under the current adapters (eff = W +
// scaling·B·A per adapted projection). Recomputed every step — the adapters moved.
func (t *LoRATrainer) effectiveWeightSets() ([]layerWeightSet, error) {
	sets := make([]layerWeightSet, len(t.layers))
	for li, L := range t.layers {
		sets[li] = layerWeightSet{wQ: L.WQ, wK: L.WK, wV: L.WV, wO: L.WO, wGate: L.WGate, wUp: L.WUp, wDown: L.WDown}
	}
	for _, ad := range t.adapters {
		L := t.layers[ad.layer]
		eff, err := LoRAEffectiveWeightF32(L.projWeight(ad.target), ad.a, ad.b, ad.out, ad.in, t.rank, t.scaling)
		if err != nil {
			return nil, err
		}
		s := &sets[ad.layer]
		switch ad.target {
		case ProjQ:
			s.wQ = eff
		case ProjK:
			s.wK = eff
		case ProjV:
			s.wV = eff
		case ProjO:
			s.wO = eff
		case ProjGate:
			s.wGate = eff
		case ProjUp:
			s.wUp = eff
		case ProjDown:
			s.wDown = eff
		}
	}
	return sets, nil
}

// layerChainForward seeds the per-layer templates with this sequence (T + the engine's PLE inputs),
// then runs the host layer chain under the given weight sets from the engine's token embeddings —
// share-aware (#42): owners/dense layers run the full attention half, KV-share consumers attend
// their owner tape's cached rows (realSharedChainForward). Returns each layer's input hidden
// (inputs[l], [T,DModel]) and forward tape.
func (t *LoRATrainer) layerChainForward(ids []int32, embeds [][]byte, sets []layerWeightSet) (inputs [][]float32, tapes []*realLayerTape, err error) {
	T := len(ids)
	x := make([]float32, 0, T*t.dModel)
	for tok := range T {
		x = append(x, bf16ToF32Slice(embeds[tok])...)
	}
	// per-sequence template fill: T everywhere; the PLE rows from the session's own closure
	// (the engine-computed per-layer inputs — frozen functions of token id + embedding).
	var pliRows [][]float32 // [tok][nLayers·pliDim]
	if t.hasPLE {
		pliRows = make([][]float32, T)
		for tok := range T {
			pli, perr := t.sess.perLayerInput(ids[tok], embeds[tok])
			if perr != nil {
				return nil, nil, perr
			}
			pliRows[tok] = bf16ToF32Slice(pli)
		}
	}
	for li, L := range t.layers {
		L.T = T
		if t.hasPLE && L.PLIDim > 0 {
			pin := make([]float32, T*L.PLIDim)
			for tok := range T {
				copy(pin[tok*L.PLIDim:(tok+1)*L.PLIDim], pliRows[tok][li*L.PLIDim:(li+1)*L.PLIDim])
			}
			L.PLEInput = pin
		}
	}
	return realSharedChainForward(x, t.layers, t.shareFrom, sets)
}

// perLayerSeqGrads runs one sequence's forward + backward under the current adapters and
// accumulates each adapter's factor gradients into sums. Returns the sequence's mean loss over its
// unmasked rows and the contributing row count (0 = fully masked, nothing accumulated).
func (t *LoRATrainer) perLayerSeqGrads(ids []int32, mask inference.LossMask, sample int, sumDA, sumDB [][]float32) (loss float32, rows int, err error) {
	if len(ids) < 2 {
		return 0, 0, core.NewError("native.LoRATrainer: a training sequence needs at least 2 tokens")
	}
	maskRows, masked, err := lossMaskRows(mask, sample, len(ids))
	if err != nil {
		return 0, 0, err
	}
	if masked && len(maskRows) == 0 {
		return 0, 0, nil
	}
	if !masked {
		maskRows = make([]int, len(ids)-1)
		for i := range maskRows {
			maskRows[i] = i
		}
	}
	T := len(ids)

	embeds, _, err := t.sess.ForwardCaptureHiddens(ids)
	if err != nil {
		return 0, 0, err
	}
	sets, err := t.effectiveWeightSets()
	if err != nil {
		return 0, 0, err
	}
	inputs, tapes, err := t.layerChainForward(ids, embeds, sets)
	if err != nil {
		return 0, 0, err
	}
	hN := tapes[len(tapes)-1].out

	// head (frozen) forward + cross-entropy on the contributing rows only.
	rows = len(maskRows)
	hRows := gatherRowsF32(hN, maskRows, t.dModel)
	normedRows := rmsNormForwardF32(hRows, t.finalNorm, rows, t.dModel, t.eps)
	baseLogits, err := MatMulF32NT(normedRows, t.lmHead, rows, t.dModel, t.vocab)
	if err != nil {
		return 0, 0, err
	}
	softcapForwardF32(baseLogits, t.softCap) // match serving's capped head (no-op when 0)
	targets := make([]int32, rows)
	for i, p := range maskRows {
		targets[i] = ids[p+1]
	}
	loss, dLogits, err := CrossEntropyBackwardF32Auto(baseLogits, targets, rows, t.vocab)
	if err != nil {
		return 0, 0, err
	}
	softcapBackwardScaleF32(dLogits, baseLogits, t.softCap)

	// head backward: dNormed = dLogits·lmHead, scattered to the full rows (masked rows zero),
	// then through the frozen final norm onto the top layer's output.
	dNormedRows, err := MatMulF32(dLogits, t.lmHead, rows, t.vocab, t.dModel)
	if err != nil {
		return 0, 0, err
	}
	dNormedFull := make([]float32, T*t.dModel)
	for i, p := range maskRows {
		copy(dNormedFull[p*t.dModel:(p+1)*t.dModel], dNormedRows[i*t.dModel:(i+1)*t.dModel])
	}
	dH, _, err := RMSNormBackwardF32(dNormedFull, hN, t.finalNorm, T, t.dModel, t.eps)
	if err != nil {
		return 0, 0, err
	}

	// walk the chain top-down (share-aware, #42); at each layer take every adapter's dW under the
	// SUBSTITUTED weight set (all adapters active at once — the exact multi-adapter gradient) and
	// fold it onto the factors.
	_, err = realSharedChainBackward(dH, inputs, tapes, t.layers, t.shareFrom, sets, func(li int, g *realLayerGrads) error {
		for ai, ad := range t.adapters {
			if ad.layer != li {
				continue
			}
			var dW []float32
			switch ad.target {
			case ProjQ:
				dW = g.dWQ
			case ProjK:
				dW = g.dWK
			case ProjV:
				dW = g.dWV
			case ProjO:
				dW = g.dWO
			case ProjGate:
				dW = g.dWGate
			case ProjUp:
				dW = g.dWUp
			case ProjDown:
				dW = g.dWDown
			}
			dA, dB, ferr := LoRAFactorGradsF32(dW, ad.a, ad.b, ad.out, ad.in, t.rank, t.scaling)
			if ferr != nil {
				return ferr
			}
			for i := range sumDA[ai] {
				sumDA[ai][i] += dA[i]
			}
			for i := range sumDB[ai] {
				sumDB[ai][i] += dB[i]
			}
		}
		return nil
	})
	if err != nil {
		return 0, 0, err
	}
	return loss, len(maskRows), nil
}

// perLayerAccumulate sums the per-sequence loss and every adapter's factor gradients across batch
// (no step) — the layers-mode twin of accumulate, honouring batch.LossMask identically.
func (t *LoRATrainer) perLayerAccumulate(batch inference.Batch) (lossSum float64, sumDA, sumDB [][]float32, n int, err error) {
	sumDA = make([][]float32, len(t.adapters))
	sumDB = make([][]float32, len(t.adapters))
	for ai, ad := range t.adapters {
		sumDA[ai] = make([]float32, len(ad.a))
		sumDB[ai] = make([]float32, len(ad.b))
	}
	for si, ids := range batch.TokenIDs {
		loss, rows, e := t.perLayerSeqGrads(ids, batch.LossMask, si, sumDA, sumDB)
		if e != nil {
			return 0, nil, nil, 0, e
		}
		if rows == 0 {
			continue
		}
		lossSum += float64(loss)
		n++
	}
	return lossSum, sumDA, sumDB, n, nil
}

// perLayerApplyMeanStep scales every adapter's summed gradients by 1/count and applies one AdamW
// update per factor.
func (t *LoRATrainer) perLayerApplyMeanStep(sumDA, sumDB [][]float32, count int) error {
	inv := float32(1.0 / float64(count))
	for ai, ad := range t.adapters {
		for i := range sumDA[ai] {
			sumDA[ai][i] *= inv
		}
		for i := range sumDB[ai] {
			sumDB[ai][i] *= inv
		}
		if err := ad.optA.Step(ad.a, sumDA[ai]); err != nil {
			return err
		}
		if err := ad.optB.Step(ad.b, sumDB[ai]); err != nil {
			return err
		}
	}
	return nil
}

// perLayerLoss is the forward-only mean loss under the current adapters — the layers-mode Loss.
func (t *LoRATrainer) perLayerLoss(batch inference.Batch) (float64, error) {
	var lossSum float64
	n := 0
	for si, ids := range batch.TokenIDs {
		if len(ids) < 2 {
			return 0, core.NewError("native.LoRATrainer.Loss: a sequence needs at least 2 tokens")
		}
		maskRows, masked, err := lossMaskRows(batch.LossMask, si, len(ids))
		if err != nil {
			return 0, err
		}
		if masked && len(maskRows) == 0 {
			continue
		}
		if !masked {
			maskRows = make([]int, len(ids)-1)
			for i := range maskRows {
				maskRows[i] = i
			}
		}
		embeds, _, err := t.sess.ForwardCaptureHiddens(ids)
		if err != nil {
			return 0, err
		}
		sets, err := t.effectiveWeightSets()
		if err != nil {
			return 0, err
		}
		_, tapes, err := t.layerChainForward(ids, embeds, sets)
		if err != nil {
			return 0, err
		}
		hN := tapes[len(tapes)-1].out
		rows := len(maskRows)
		hRows := gatherRowsF32(hN, maskRows, t.dModel)
		normedRows := rmsNormForwardF32(hRows, t.finalNorm, rows, t.dModel, t.eps)
		baseLogits, err := MatMulF32NT(normedRows, t.lmHead, rows, t.dModel, t.vocab)
		if err != nil {
			return 0, err
		}
		softcapForwardF32(baseLogits, t.softCap)
		targets := make([]int32, rows)
		for i, p := range maskRows {
			targets[i] = ids[p+1]
		}
		loss, _, err := CrossEntropyBackwardF32Auto(baseLogits, targets, rows, t.vocab)
		if err != nil {
			return 0, err
		}
		lossSum += float64(loss)
		n++
	}
	if n == 0 {
		return 0, core.NewError("native.LoRATrainer.Loss: batch produced no scoreable sequences (a set LossMask masks every position)")
	}
	return lossSum / float64(n), nil
}

// layerAdapterTensorName is the on-disk tensor prefix of one per-layer adapter — the mlx
// per-layer format (model.layers.<i>.self_attn.q_proj / .mlp.gate_proj), so the emitted
// identifiers stay canonical for the ecosystem.
func layerAdapterTensorName(layer int, target string) string {
	module := "self_attn"
	switch target {
	case ProjGate, ProjUp, ProjDown:
		module = "mlp"
	}
	return core.Concat("model.layers.", core.Sprintf("%d", layer), ".", module, ".", target)
}

// perLayerSave writes the trained per-layer adapters as one reloadable package —
// adapter.safetensors (each projection's A/B as F32 under the canonical mlx per-layer names) +
// adapter_config.json (rank/alpha, the adapted layer count, the target list).
func (t *LoRATrainer) perLayerSave(path string) error {
	if path == "" {
		return core.NewError("native.LoRATrainer.Save: path is required")
	}
	if res := core.MkdirAll(path, core.FileMode(0o755)); !res.OK {
		return core.E("native.LoRATrainer.Save", "ensure adapter dir", resultErr(res))
	}
	tensors := map[string]safetensors.Tensor{}
	layerSet := map[int]bool{}
	targetSet := map[string]bool{}
	var targetList []string
	for _, ad := range t.adapters {
		name := layerAdapterTensorName(ad.layer, ad.target)
		tensors[name+".lora_a"] = safetensors.Tensor{Dtype: "F32", Shape: []int{t.rank, ad.in}, Data: safetensors.EncodeFloat32(ad.a)}
		tensors[name+".lora_b"] = safetensors.Tensor{Dtype: "F32", Shape: []int{ad.out, t.rank}, Data: safetensors.EncodeFloat32(ad.b)}
		layerSet[ad.layer] = true
		if !targetSet[ad.target] {
			targetSet[ad.target] = true
			targetList = append(targetList, ad.target)
		}
	}
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		return core.E("native.LoRATrainer.Save", "encode adapter safetensors", err)
	}
	if werr := coreio.Local.Write(core.PathJoin(path, "adapter.safetensors"), string(blob)); werr != nil {
		return core.E("native.LoRATrainer.Save", "write adapter.safetensors", werr)
	}
	cfg := adapterConfigJSON{Rank: t.rank, Alpha: t.alpha, NumLayers: len(layerSet), LoRALayers: targetList}
	cj := core.JSONMarshal(cfg)
	if !cj.OK {
		return core.E("native.LoRATrainer.Save", "marshal adapter_config.json", nil)
	}
	if werr := coreio.Local.Write(core.PathJoin(path, "adapter_config.json"), string(cj.Value.([]byte))); werr != nil {
		return core.E("native.LoRATrainer.Save", "write adapter_config.json", werr)
	}
	return nil
}
