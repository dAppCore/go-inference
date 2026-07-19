// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import (
	"math"

	core "dappco.re/go"
)

// decoder.go is Qwen2ForCausalLM ported host-side — DOTS-OCR's text decoder is an UNMODIFIED
// Qwen2 decoder-only causal LM (modeling_dots_ocr.py's DotsOCRForCausalLM subclasses it directly
// and only overrides embedding preparation, never the transformer stack itself): token embedding
// → N pre-norm layers (causal GQA self-attention with 1D rotary + SwiGLU MLP, both RMSNorm) →
// final RMSNorm → an UNTIED lm_head (tie_word_embeddings is false — a standalone lm_head.weight
// tensor, unlike whisper's tied proj_out). Self-attention carries a growing per-layer KV cache
// from the first call (decodeLayersStep processes the whole prompt as one "new" batch against an
// empty cache — the prefill) — see decoder_test.go's parity proof that batching the SAME token
// sequence differently (one shot vs one token at a time) produces BIT-IDENTICAL logits at every
// shared position, for the same row-wise/causal argument whisper's decoder.go doc comment gives
// (every op here except self-attention is row-wise, and causal self-attention's row i depends
// only on rows [0,i], present in the cache regardless of which call added them).

// SelfAttnCache is one decoder layer's growing self-attention K/V cache, flat [n, KVHeads·HeadDim]
// (n = tokens appended so far). Mirrors whisper.SelfAttnCache's shape.
type SelfAttnCache struct {
	K, V []float32
}

// NewSelfAttnCache allocates one empty cache per decoder layer.
func NewSelfAttnCache(numLayers int) []SelfAttnCache {
	return make([]SelfAttnCache, numLayers)
}

// EmbedTokens looks up ids[Tn] in the token embedding table, returning [Tn,HiddenSize] flat —
// the row-wise input ocr.go's vision scatter overwrites at image-token positions before the first
// decodeLayersStep call.
func EmbedTokens(ids []int32, w *Weights, cfg *Config) ([]float32, error) {
	d := cfg.HiddenSize
	out := make([]float32, len(ids)*d)
	for t, id := range ids {
		if int(id) < 0 || int(id) >= cfg.VocabSize {
			return nil, core.NewError(core.Sprintf("dotsocr.EmbedTokens: token id %d at position %d is out of vocab range [0,%d)", id, t, cfg.VocabSize))
		}
		copy(out[t*d:(t+1)*d], w.EmbedTokens[int(id)*d:int(id)*d+d])
	}
	return out, nil
}

// textRotaryCosSin computes the text decoder's 1D rotary half-tables at absolute position pos —
// invFreq[j] = theta^(-2j/headDim), j in [0,headDim/2) (Qwen2RotaryEmbedding.
// compute_default_rope_parameters); cosHalf[j] = cos(pos·invFreq[j]), matching applyRotaryHalf's
// expected half-length table directly (no h/w split — that is the vision tower's 2D rotary only,
// see visionRotaryTable).
func textRotaryCosSin(headDim int, theta float32, pos int) (cosHalf, sinHalf []float32) {
	half := headDim / 2
	cosHalf = make([]float32, half)
	sinHalf = make([]float32, half)
	for j := range half {
		angle := float64(pos) / math.Pow(float64(theta), float64(2*j)/float64(headDim))
		cosHalf[j] = float32(math.Cos(angle))
		sinHalf[j] = float32(math.Sin(angle))
	}
	return cosHalf, sinHalf
}

// decoderSelfAttention is one Qwen2Attention pass: projects q/k/v for the new rows x[Tn,D],
// applies rotary to q/k per head at their true absolute positions, appends k/v onto cache
// (mutated in place), then attends each new row's query over the FULL cache (history + this
// batch, causal) with GQA head grouping — query head h reads KV head h/(heads/kvHeads), matching
// repeat_kv's expansion order exactly (modeling_qwen2.py: kv head index repeated n_rep times
// contiguously, so head h's group is h/n_rep, not h%kvHeads).
func decoderSelfAttention(x []float32, tn, d, heads, kvHeads, headDim int, q, k, v, o LinearWeights, cache *SelfAttnCache, startPos int, theta float32) ([]float32, error) {
	if d%heads != 0 {
		return nil, core.NewError("dotsocr.decoderSelfAttention: hidden_size not divisible by num_attention_heads")
	}
	if heads%kvHeads != 0 {
		return nil, core.NewError("dotsocr.decoderSelfAttention: num_attention_heads not divisible by num_key_value_heads")
	}
	kvDim := kvHeads * headDim
	qProj := linearForward(x, q, tn)
	kNew := linearForward(x, k, tn)
	vNew := linearForward(x, v, tn)

	for t := range tn {
		cosHalf, sinHalf := textRotaryCosSin(headDim, theta, startPos+t)
		for h := range heads {
			off := t*d + h*headDim
			applyRotaryHalf(qProj[off:off+headDim], cosHalf, sinHalf)
		}
		for h := range kvHeads {
			off := t*kvDim + h*headDim
			applyRotaryHalf(kNew[off:off+headDim], cosHalf, sinHalf)
		}
	}

	offset := len(cache.K) / kvDim
	cache.K = append(cache.K, kNew...)
	cache.V = append(cache.V, vNew...)
	tk := len(cache.K) / kvDim

	groupSize := heads / kvHeads
	scale := 1.0 / math.Sqrt(float64(headDim))
	out := make([]float32, tn*d)
	scores := make([]float64, tk)
	for h := range heads {
		kvh := h / groupSize
		qOff := h * headDim
		kvOff := kvh * headDim
		for i := range tn {
			limit := offset + i + 1
			qi := qProj[i*d+qOff : i*d+qOff+headDim]
			maxScore := math.Inf(-1)
			for j := range limit {
				kj := cache.K[j*kvDim+kvOff : j*kvDim+kvOff+headDim]
				var dot float64
				for c := range headDim {
					dot += float64(qi[c]) * float64(kj[c])
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
			oi := out[i*d+qOff : i*d+qOff+headDim]
			for c := range headDim {
				var acc float64
				for j := range limit {
					acc += scores[j] * float64(cache.V[j*kvDim+kvOff+c])
				}
				oi[c] = float32(acc / sum)
			}
		}
	}
	return linearForward(out, o, tn), nil
}

// decodeLayersStep runs the transformer stack over PRE-BUILT input embeddings embeds[tn,D]
// (ocr.go's vision scatter has already spliced image embeddings in before the prefill call) at
// absolute position startPos, mutating cache, and returns vocabulary logits at the LAST position
// only — the one a greedy decode step ever needs.
func decodeLayersStep(embeds []float32, tn, startPos int, cache []SelfAttnCache, w *Weights, cfg *Config) ([]float32, error) {
	if w == nil || cfg == nil {
		return nil, core.NewError("dotsocr.decodeLayersStep: nil weights/config")
	}
	if len(cache) != len(w.Layers) {
		return nil, core.NewError("dotsocr.decodeLayersStep: cache must have one entry per decoder layer")
	}
	d := cfg.HiddenSize
	if len(embeds) != tn*d {
		return nil, core.NewError(core.Sprintf("dotsocr.decodeLayersStep: embeds has %d elements, want %d (%d×%d)", len(embeds), tn*d, tn, d))
	}
	headDim := d / cfg.NumAttentionHeads
	hidden := embeds
	for li := range w.Layers {
		layer := w.Layers[li]
		residual := hidden
		normed := rmsNormForward(hidden, layer.InputNorm, tn, d, cfg.RMSNormEps)
		attnOut, err := decoderSelfAttention(normed, tn, d, cfg.NumAttentionHeads, cfg.NumKeyValueHeads, headDim, layer.Q, layer.K, layer.V, layer.O, &cache[li], startPos, cfg.RopeTheta)
		if err != nil {
			return nil, err
		}
		hidden = addRows(residual, attnOut)

		residual = hidden
		normed = rmsNormForward(hidden, layer.PostAttnNorm, tn, d, cfg.RMSNormEps)
		ff := swiGLU(normed, layer.Gate, layer.Up, layer.Down, tn)
		hidden = addRows(residual, ff)
	}
	normedFinal := rmsNormForward(hidden, w.FinalNorm, tn, d, cfg.RMSNormEps)
	lastRow := normedFinal[(tn-1)*d : tn*d]
	return linearForward(lastRow, w.LMHead, 1), nil
}

// DecodeLogitsStep embeds newIDs via the token table (no image tokens: every per-token decode
// step after the prefill is plain text) and runs decodeLayersStep — the incremental path
// GreedyDecode calls for every step after the first.
func DecodeLogitsStep(newIDs []int32, startPos int, cache []SelfAttnCache, w *Weights, cfg *Config) ([]float32, error) {
	if len(newIDs) == 0 {
		return nil, core.NewError("dotsocr.DecodeLogitsStep: no new tokens")
	}
	embeds, err := EmbedTokens(newIDs, w, cfg)
	if err != nil {
		return nil, err
	}
	return decodeLayersStep(embeds, len(newIDs), startPos, cache, w, cfg)
}

// DecodeLogits runs the decoder stack over the WHOLE token sequence ids in one call against a
// fresh cache — the whole-sequence recompute reference decoder_test.go's parity test checks
// DecodeLogitsStep's incremental path against (see this file's doc comment for why they must be
// bit-identical, not merely close).
func DecodeLogits(ids []int32, w *Weights, cfg *Config) ([]float32, error) {
	if w == nil {
		return nil, core.NewError("dotsocr.DecodeLogits: nil weights")
	}
	cache := NewSelfAttnCache(len(w.Layers))
	return DecodeLogitsStep(ids, 0, cache, w, cfg)
}

// argmax32 returns the index of the largest value in logits (ties keep the first, matching
// torch.argmax's default tie-break).
func argmax32(logits []float32) int32 {
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return int32(best)
}

// GreedyDecode runs the full greedy generation loop: prefill promptIDs (which ocr.go has already
// embedded and vision-scattered into promptEmbeds), then repeatedly append the argmax token until
// eosIDs is hit or maxNewTokens is reached. Returns the GENERATED ids only (never the prompt).
func GreedyDecode(promptEmbeds []float32, promptLen int, w *Weights, cfg *Config, maxNewTokens int, eosIDs map[int32]bool) ([]int32, error) {
	if maxNewTokens < 0 {
		return nil, core.NewError("dotsocr.GreedyDecode: negative maxNewTokens")
	}
	cache := NewSelfAttnCache(len(w.Layers))
	logits, err := decodeLayersStep(promptEmbeds, promptLen, 0, cache, w, cfg)
	if err != nil {
		return nil, err
	}
	generated := make([]int32, 0, maxNewTokens)
	pos := promptLen
	for range maxNewTokens {
		next := argmax32(logits)
		if eosIDs[next] {
			break
		}
		generated = append(generated, next)
		logits, err = DecodeLogitsStep([]int32{next}, pos, cache, w, cfg)
		if err != nil {
			return nil, err
		}
		pos++
	}
	return generated, nil
}
