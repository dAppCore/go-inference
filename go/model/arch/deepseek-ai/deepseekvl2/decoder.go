// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"math"
	"sort"

	core "dappco.re/go"
)

// decoder.go is the DeepSeek-V2-lite MoE decoder ported host-side: token/vision embeddings (built
// by tokens.go) → cfg.NumHiddenLayers pre-norm layers (causal rotary self-attention — use_mla=
// false resolves to plain LlamaAttention, not DeepSeek's multi-latent attention, see Config's doc
// comment — then EITHER a dense SwiGLU MLP (layer_idx < FirstKDenseReplace) OR a routed-plus-
// shared MoE) → a final RMSNorm → the (untied) lm_head projection.
//
// SELF-ATTENTION KV CACHE from the start (unlike whisper's phased v1-then-cache-later shape):
// this decoder's per-layer cost is dominated by its 64-expert MoE MLPs (moe_intermediate_size
// 896, hidden 1280) — recomputing the WHOLE growing sequence from scratch on every greedy-decode
// step (whisper's original v1 shape, whisper/decoder.go's doc comment) would reprocess every
// earlier token's MoE routing+combine on every later step too; for an 8-token generation over a
// ~280-token prompt that's roughly an 80x redundancy over processing each token once — impractical
// for a gate, not a device-fusion concern. DecodeLogitsStep (cached) is the same
// bit-identical-to-whole-sequence-recompute argument whisper/decoder.go's file doc comment makes
// (every op here is row-wise except causal self-attention, which reads exactly [0,limit) in a
// fixed order regardless of which call populated the cache) — DecodeLogits (whole-sequence, no
// cache) is kept as that equivalence's test oracle, decoder_test.go's parity test proves it.

const (
	decoderRopeTheta    = 10000.0 // DeepseekV2Config's own default — config.json never overrides it (Config's doc comment)
	decoderRMSNormEps   = 1e-6    // ditto
	decoderNormTopkProb = false   // ditto — top-k expert weights are NOT renormalised to sum to 1
	decoderRoutedScale  = 1.0     // ditto — routed_scaling_factor's default is a no-op multiply
)

// ropeCosSin returns the [headDim] cos/sin rotation vectors for one absolute position — the
// standard rotate-half convention (Llama/NeoX-style, NOT interleaved pairs): inv_freq[i] =
// theta^(-2i/headDim) for i in [0,headDim/2), and the returned vectors DUPLICATE that half twice
// (emb = cat(freqs,freqs)) so index d and d+headDim/2 share the same angle — apply_rotary_pos_emb
// pairs (d, d+half) via rotate_half, exactly the shape decoderApplyRope below mirrors.
func ropeCosSin(pos, headDim int) (cos, sin []float32) {
	half := headDim / 2
	cos = make([]float32, headDim)
	sin = make([]float32, headDim)
	for i := range half {
		invFreq := 1.0 / math.Pow(decoderRopeTheta, float64(2*i)/float64(headDim))
		angle := float64(pos) * invFreq
		c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
		cos[i], cos[i+half] = c, c
		sin[i], sin[i+half] = s, s
	}
	return cos, sin
}

// decoderApplyRope rotates one [headDim] q/k vector IN PLACE at absolute position pos —
// rotate_half(x) = cat(-x[half:], x[:half]); x_embed = x*cos + rotate_half(x)*sin.
func decoderApplyRope(x []float32, pos, headDim int) {
	cos, sin := ropeCosSin(pos, headDim)
	half := headDim / 2
	out := make([]float32, headDim)
	for d := range half {
		out[d] = x[d]*cos[d] - x[d+half]*sin[d]
		out[d+half] = x[d+half]*cos[d+half] + x[d]*sin[d+half]
	}
	copy(x, out)
}

// SelfAttnCache is one decoder layer's growing self-attention K/V cache, flat [n,hidden] (n =
// tokens appended so far) — mirrors whisper.SelfAttnCache exactly (arch/openai/whisper/decoder.go).
type SelfAttnCache struct {
	K, V []float32
}

// NewSelfAttnCache allocates one empty cache per decoder layer.
func NewSelfAttnCache(numLayers int) []SelfAttnCache { return make([]SelfAttnCache, numLayers) }

// decoderAttnCore runs causal scaled-dot-product attention given already-projected+roped q[Tq,H]/
// k[Tk,H]/v[Tk,H] (H=hidden, num_heads==num_key_value_heads in this checkpoint — plain MHA, no
// GQA repeat needed). offset is the absolute position of k/v row 0 (0 for a whole-sequence pass,
// the pre-batch cache length for a cached pass) — query row i may attend keys [0,offset+i].
func decoderAttnCore(q, k, v []float32, tq, tk, numHeads, headDim, offset int) []float32 {
	hidden := numHeads * headDim
	out := make([]float32, tq*hidden)
	scale := 1.0 / math.Sqrt(float64(headDim))
	// Parallelise over the flattened (head, query-row) space — scores allocated per-unit, never
	// shared across goroutines (mathops.go's parallelFor doc comment). This is the decoder's
	// second-largest cost after moeForward (up to ~370 prompt+generated tokens x 12 layers).
	parallelFor(numHeads*tq, func(unit int) {
		h, i := unit/tq, unit%tq
		off := h * headDim
		limit := offset + i + 1
		if limit > tk {
			limit = tk
		}
		scores := make([]float64, limit)
		qi := q[i*hidden+off : i*hidden+off+headDim]
		maxScore := math.Inf(-1)
		for j := range limit {
			kj := k[j*hidden+off : j*hidden+off+headDim]
			var dot float64
			for d := range headDim {
				dot += float64(qi[d]) * float64(kj[d])
			}
			dot *= scale
			scores[j] = dot
			if dot > maxScore {
				maxScore = dot
			}
		}
		var sum float64
		for j := range limit {
			e := math.Exp(scores[j] - maxScore)
			scores[j] = e
			sum += e
		}
		oi := out[i*hidden+off : i*hidden+off+headDim]
		for d := range headDim {
			var acc float64
			for j := range limit {
				acc += scores[j] * float64(v[j*hidden+off+d])
			}
			oi[d] = float32(acc / sum)
		}
	})
	return out
}

// decoderSelfAttention projects q/k/v from x[T,hidden], applies RoPE at absolute positions
// [startPos,startPos+T), runs the whole-sequence (no-cache) causal core, out_proj.
func decoderSelfAttention(x []float32, startPos, hidden, numHeads, headDim int, ly DecoderLayerWeights) []float32 {
	t := len(x) / hidden
	q := linear(x, ly.QW, hidden, hidden, nil)
	k := linear(x, ly.KW, hidden, hidden, nil)
	v := linear(x, ly.VW, hidden, hidden, nil)
	for i := range t {
		for h := range numHeads {
			decoderApplyRope(q[i*hidden+h*headDim:i*hidden+h*headDim+headDim], startPos+i, headDim)
			decoderApplyRope(k[i*hidden+h*headDim:i*hidden+h*headDim+headDim], startPos+i, headDim)
		}
	}
	attn := decoderAttnCore(q, k, v, t, t, numHeads, headDim, 0)
	return linear(attn, ly.OW, hidden, hidden, nil)
}

// decoderSelfAttentionCached is decoderSelfAttention's incremental twin: projects q/k/v for ONLY
// the new rows x[Tn,hidden], applies RoPE at their true absolute positions, appends the new k/v
// onto cache (mutated in place), attends over the FULL cache (history + these new rows).
func decoderSelfAttentionCached(x []float32, startPos, hidden, numHeads, headDim int, ly DecoderLayerWeights, cache *SelfAttnCache) []float32 {
	tn := len(x) / hidden
	q := linear(x, ly.QW, hidden, hidden, nil)
	kNew := linear(x, ly.KW, hidden, hidden, nil)
	v := linear(x, ly.VW, hidden, hidden, nil)
	for i := range tn {
		for h := range numHeads {
			decoderApplyRope(q[i*hidden+h*headDim:i*hidden+h*headDim+headDim], startPos+i, headDim)
			decoderApplyRope(kNew[i*hidden+h*headDim:i*hidden+h*headDim+headDim], startPos+i, headDim)
		}
	}
	offset := len(cache.K) / hidden
	cache.K = append(cache.K, kNew...)
	cache.V = append(cache.V, v...)
	tk := len(cache.K) / hidden
	attn := decoderAttnCore(q, cache.K, cache.V, tn, tk, numHeads, headDim, offset)
	return linear(attn, ly.OW, hidden, hidden, nil)
}

// denseMLPForward is DeepseekV2MLP: down(SiLU(gate(x)) * up(x)) — the dense layer (layer_idx <
// cfg.FirstKDenseReplace) and, at MoE-layer width, the shared-expert MLP below.
func denseMLPForward(x []float32, gateW, upW, downW []float32, hidden, intermediate int) []float32 {
	g := linear(x, gateW, hidden, intermediate, nil)
	u := linear(x, upW, hidden, intermediate, nil)
	h := make([]float32, len(g))
	for i := range h {
		h[i] = silu(g[i]) * u[i]
	}
	return linear(h, downW, intermediate, hidden, nil)
}

// expertScore is one (expert index, gate weight) pair — moeRoute's per-token top-k result.
type expertScore struct {
	idx    int
	weight float32
}

// moeRoute runs MoEGate over one token's hidden vector x[hidden]: logits = x·GateWeightᵀ (no
// bias), softmax over all n_routed_experts, then the checkpoint's topk_method "greedy" — plain
// top-num_experts_per_tok by score (config.json's n_group=topk_group=1 makes the "group_limited"
// paths in the reference degenerate to this same plain top-k anyway). norm_topk_prob is False
// (decoderNormTopkProb) — the returned weights are the RAW softmax scores for the selected
// experts, NOT renormalised to sum to 1 (MoEGate.forward's "else: topk_weight = topk_weight *
// routed_scaling_factor" branch — see Config's doc comment for why these unset-in-config.json
// fields resolve to DeepseekV2Config's own Python defaults, confirmed, not assumed).
func moeRoute(x []float32, gateWeight []float32, hidden, numExperts, topK int) []expertScore {
	logits := linear(x, gateWeight, hidden, numExperts, nil)
	scores := make([]float64, numExperts)
	logits64 := make([]float64, numExperts)
	for i, v := range logits {
		logits64[i] = float64(v)
	}
	copy(scores, logits64)
	softmaxInPlace(scores)

	all := make([]expertScore, numExperts)
	for i := range numExperts {
		all[i] = expertScore{idx: i, weight: float32(scores[i])}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].weight > all[j].weight })
	top := append([]expertScore(nil), all[:topK]...)
	if decoderNormTopkProb {
		var sum float32
		for _, e := range top {
			sum += e.weight
		}
		for i := range top {
			top[i].weight = top[i].weight / sum * decoderRoutedScale
		}
	} else {
		for i := range top {
			top[i].weight *= decoderRoutedScale
		}
	}
	return top
}

// moeForward is DeepseekV2MoE: route each token independently to its own top-K experts, combine
// their weighted outputs, and ALWAYS add the one combined shared-expert MLP (n_shared_experts
// folded into a single wider intermediate — weights_decoder.go's DecoderLayerWeights doc comment).
// Tokens are fully independent (own routing, own accumulator, disjoint output row) —
// parallelFor'd (mathops.go's file doc comment): this is the decoder's dominant cost (up to 64
// experts per MoE layer per token), confirmed impractically slow single-threaded on a real
// checkpoint's ~280+-token prompt prefill.
func moeForward(x []float32, ly DecoderLayerWeights, hidden, moeIntermediate, numExperts, topK int) []float32 {
	t := len(x) / hidden
	out := make([]float32, t*hidden)
	parallelFor(t, func(i int) {
		xi := x[i*hidden : (i+1)*hidden]
		picked := moeRoute(xi, ly.GateWeight, hidden, numExperts, topK)
		acc := make([]float64, hidden)
		for _, e := range picked {
			ex := ly.Experts[e.idx]
			y := denseMLPForward(xi, ex.GateW, ex.UpW, ex.DownW, hidden, moeIntermediate)
			for d := range hidden {
				acc[d] += float64(e.weight) * float64(y[d])
			}
		}
		oi := out[i*hidden : (i+1)*hidden]
		for d := range hidden {
			oi[d] = float32(acc[d])
		}
	})
	// SharedGateW is [sharedIntermediate,hidden] flat — sharedIntermediate = moe_intermediate_size *
	// n_shared_experts (weights_decoder.go's doc comment), read back from the loaded weight's own
	// length rather than re-deriving n_shared_experts here (moeForward doesn't otherwise need it).
	sharedIntermediate := len(ly.SharedGateW) / hidden
	shared := denseMLPForward(x, ly.SharedGateW, ly.SharedUpW, ly.SharedDownW, hidden, sharedIntermediate)
	return addRows(out, shared)
}

// decoderLayerForward runs one pre-norm layer (whole-sequence, no cache — DecodeLogits' path):
// RMSNorm → self-attention → residual, RMSNorm → dense-or-MoE MLP → residual.
func decoderLayerForward(x []float32, startPos, hidden, numHeads, headDim, moeIntermediate, numExperts, topK int, ly DecoderLayerWeights) []float32 {
	residual := x
	normed := rmsNorm(x, ly.InputNormW, hidden, decoderRMSNormEps)
	attnOut := decoderSelfAttention(normed, startPos, hidden, numHeads, headDim, ly)
	hiddenX := addRows(residual, attnOut)

	residual = hiddenX
	normed = rmsNorm(hiddenX, ly.PostAttnNormW, hidden, decoderRMSNormEps)
	var mlpOut []float32
	if ly.IsMoE {
		mlpOut = moeForward(normed, ly, hidden, moeIntermediate, numExperts, topK)
	} else {
		mlpOut = denseMLPForward(normed, ly.DenseGateW, ly.DenseUpW, ly.DenseDownW, hidden, len(ly.DenseGateW)/hidden)
	}
	return addRows(residual, mlpOut)
}

// decoderLayerForwardCached is decoderLayerForward's incremental twin (DecodeLogitsStep's path):
// only self-attention differs (decoderSelfAttentionCached) — every other op is row-wise, so it
// runs identically over just the new rows (see the file doc comment's equivalence argument).
func decoderLayerForwardCached(x []float32, startPos, hidden, numHeads, headDim, moeIntermediate, numExperts, topK int, ly DecoderLayerWeights, cache *SelfAttnCache) []float32 {
	residual := x
	normed := rmsNorm(x, ly.InputNormW, hidden, decoderRMSNormEps)
	attnOut := decoderSelfAttentionCached(normed, startPos, hidden, numHeads, headDim, ly, cache)
	hiddenX := addRows(residual, attnOut)

	residual = hiddenX
	normed = rmsNorm(hiddenX, ly.PostAttnNormW, hidden, decoderRMSNormEps)
	var mlpOut []float32
	if ly.IsMoE {
		mlpOut = moeForward(normed, ly, hidden, moeIntermediate, numExperts, topK)
	} else {
		mlpOut = denseMLPForward(normed, ly.DenseGateW, ly.DenseUpW, ly.DenseDownW, hidden, len(ly.DenseGateW)/hidden)
	}
	return addRows(residual, mlpOut)
}

// tiedLMHead-equivalent: DeepSeek-OCR's lm_head is NOT tied (weights_decoder.go's doc comment) —
// this is a plain output projection over one final hidden row.
func lmHead(hiddenRow []float32, w *Weights) []float32 {
	return linear(hiddenRow, w.Decoder.LMHeadWeight, w.Decoder.hiddenSize(), len(w.Decoder.LMHeadWeight)/w.Decoder.hiddenSize(), nil)
}

// DecodeLogits runs the decoder stack over the WHOLE embeds sequence embeds[T,hidden] from
// scratch (no cache — the whole-sequence reference/test oracle, see the file doc comment) and
// returns the vocabulary logits at the LAST position only.
func DecodeLogits(embeds []float32, cfg *Config, w *Weights) ([]float32, error) {
	if cfg == nil || w == nil {
		return nil, core.NewError("deepseekvl2.DecodeLogits: nil config/weights")
	}
	hidden := cfg.HiddenSize
	if hidden <= 0 || len(embeds)%hidden != 0 {
		return nil, core.NewError("deepseekvl2.DecodeLogits: embeds buffer is not a whole number of hidden-width rows")
	}
	headDim := hidden / cfg.NumAttentionHeads
	hiddenX := embeds
	for _, ly := range w.Decoder.Layers {
		hiddenX = decoderLayerForward(hiddenX, 0, hidden, cfg.NumAttentionHeads, headDim, cfg.MoEIntermediateSize, cfg.NRoutedExperts, cfg.NumExpertsPerTok, ly)
	}
	final := rmsNorm(hiddenX, w.Decoder.FinalNormW, hidden, decoderRMSNormEps)
	t := len(final) / hidden
	return lmHead(final[(t-1)*hidden:t*hidden], w), nil
}

// DecodeLogitsStep runs the decoder stack over ONLY the newly appended embedding rows newEmbeds
// (a multi-row PREFILL batch on the first call — the whole initial prompt — or a single row on
// every call after), starting at absolute position startPos, appending each layer's self-
// attention K/V onto cache (mutated in place) and attending over the full cache. Returns the
// vocabulary logits at the LAST new position only — bit-identical to what DecodeLogits(wholeSoFar,
// …) would return for that position (see the file doc comment; decoder_test.go's parity test).
func DecodeLogitsStep(newEmbeds []float32, startPos int, cache []SelfAttnCache, cfg *Config, w *Weights) ([]float32, error) {
	if cfg == nil || w == nil {
		return nil, core.NewError("deepseekvl2.DecodeLogitsStep: nil config/weights")
	}
	hidden := cfg.HiddenSize
	if hidden <= 0 || len(newEmbeds)%hidden != 0 {
		return nil, core.NewError("deepseekvl2.DecodeLogitsStep: newEmbeds buffer is not a whole number of hidden-width rows")
	}
	if len(cache) != len(w.Decoder.Layers) {
		return nil, core.NewError("deepseekvl2.DecodeLogitsStep: cache must have one entry per decoder layer")
	}
	headDim := hidden / cfg.NumAttentionHeads
	hiddenX := newEmbeds
	for li, ly := range w.Decoder.Layers {
		hiddenX = decoderLayerForwardCached(hiddenX, startPos, hidden, cfg.NumAttentionHeads, headDim, cfg.MoEIntermediateSize, cfg.NRoutedExperts, cfg.NumExpertsPerTok, ly, &cache[li])
	}
	final := rmsNorm(hiddenX, w.Decoder.FinalNormW, hidden, decoderRMSNormEps)
	t := len(final) / hidden
	return lmHead(final[(t-1)*hidden:t*hidden], w), nil
}

// argmaxF32 returns the index of the largest value (ties keep the first, matching torch.argmax).
func argmaxF32(v []float32) int32 {
	best := int32(0)
	bestVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > bestVal {
			bestVal = v[i]
			best = int32(i)
		}
	}
	return best
}
