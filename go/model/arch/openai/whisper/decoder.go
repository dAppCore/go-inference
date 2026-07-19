// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import core "dappco.re/go"

// decoder.go is WhisperDecoder ported host-side: token embedding + the learned position table (absolute
// position = index into the sequence, matching WhisperPositionalEmbedding's plain weight[pos] lookup) →
// N pre-LN layers (causal self-attention → cross-attention over the PRECOMPUTED encoder K/V → FFN) →
// one final top-level LayerNorm → the tied LM head (proj_out.weight IS model.decoder.embed_tokens.weight
// — see weights.go's doc comment).
//
// DELIBERATE v1 SIMPLIFICATION (host-f32 correctness-first, per the design's own phasing): DecodeLogits
// recomputes the FULL causal self-attention over every token generated so far on EVERY call — there is
// no incremental self-attention KV cache. This is the exact shape validated against the reference
// (transformers' generate() with no custom cache, same greedy argmax, same suppress lists) before this
// file was written. Whisper-tiny's decode is 4 layers × d_model 384 over at most MaxTargetPositions=448
// steps — O(T²) total self-attention work that stays well under a second on CPU host math. Cross-
// attention K/V IS precomputed once per request (PrecomputeCrossKV), matching the design's explicit ask
// — the standard Whisper serving trick, since cross K/V depend only on the fixed encoder output. A later
// perf slice can add a self-attention KV cache without changing this function's outputs.

// CrossKV is one decoder layer's precomputed cross-attention K/V — computed once per request from the
// (fixed) encoder output.
type CrossKV struct {
	K, V []float32 // [Tenc, DModel] each
}

// PrecomputeCrossKV projects the encoder output through every decoder layer's cross-attention K/V once
// — reused for every subsequent decode step (encoder output never changes mid-request).
func PrecomputeCrossKV(encOut []float32, tenc int, w *Weights) []CrossKV {
	out := make([]CrossKV, len(w.DecoderLayers))
	for i, layer := range w.DecoderLayers {
		k, v := precomputeCrossKV(encOut, tenc, layer.CrossAttn)
		out[i] = CrossKV{K: k, V: v}
	}
	return out
}

// DecodeLogits runs the decoder stack over the full token sequence ids (see the file doc comment for why
// this recomputes from scratch every call) and returns the vocabulary logits at the LAST position only
// — the one a greedy decode step or the language-detection step ever needs.
func DecodeLogits(ids []int32, crossKV []CrossKV, tenc int, w *Weights, cfg *Config) ([]float32, error) {
	if w == nil || cfg == nil {
		return nil, core.NewError("whisper.DecodeLogits: nil weights/config")
	}
	T := len(ids)
	if T == 0 {
		return nil, core.NewError("whisper.DecodeLogits: empty token sequence")
	}
	if T > cfg.MaxTargetPositions {
		return nil, core.NewError(core.Sprintf("whisper.DecodeLogits: sequence length %d exceeds max_target_positions %d", T, cfg.MaxTargetPositions))
	}
	if len(crossKV) != len(w.DecoderLayers) {
		return nil, core.NewError("whisper.DecodeLogits: crossKV has one entry per decoder layer, got a mismatched count")
	}
	D := cfg.DModel
	hidden := make([]float32, T*D)
	for t, id := range ids {
		if int(id) < 0 || int(id) >= cfg.VocabSize {
			return nil, core.NewError(core.Sprintf("whisper.DecodeLogits: token id %d at position %d is out of vocab range [0,%d)", id, t, cfg.VocabSize))
		}
		embedRow := w.EmbedTokens[int(id)*D : int(id)*D+D]
		posRow := w.DecoderPos[t*D : t*D+D]
		for d := range D {
			hidden[t*D+d] = embedRow[d] + posRow[d]
		}
	}

	for li, layer := range w.DecoderLayers {
		residual := hidden
		normed := layerNormForward(hidden, layer.SelfAttnNorm, T, D)
		selfOut, err := selfAttentionForward(normed, T, D, cfg.DecoderAttentionHeads, true, layer.SelfAttn)
		if err != nil {
			return nil, err
		}
		hidden = addRows(residual, selfOut)

		residual = hidden
		normed = layerNormForward(hidden, layer.CrossAttnNorm, T, D)
		crossOut, err := crossAttentionForward(normed, T, D, cfg.DecoderAttentionHeads, layer.CrossAttn, crossKV[li].K, crossKV[li].V, tenc)
		if err != nil {
			return nil, err
		}
		hidden = addRows(residual, crossOut)

		residual = hidden
		normed = layerNormForward(hidden, layer.FinalNorm, T, D)
		ff := linearForward(geluRow(linearForward(normed, layer.FC1, T)), layer.FC2, T)
		hidden = addRows(residual, ff)
	}

	normedFinal := layerNormForward(hidden, w.DecoderFinalNorm, T, D)
	lastRow := normedFinal[(T-1)*D : T*D]
	return tiedLMHead(lastRow, w), nil
}

// tiedLMHead projects one final hidden row [DModel] to vocabulary logits [VocabSize] via the embedding
// table transposed (proj_out.weight IS embed_tokens.weight — Whisper ties them, no separate tensor).
func tiedLMHead(hiddenRow []float32, w *Weights) []float32 {
	logits := make([]float32, w.VocabSize)
	for v := range w.VocabSize {
		row := w.EmbedTokens[v*w.DModel : v*w.DModel+w.DModel]
		var acc float64
		for d := range w.DModel {
			acc += float64(hiddenRow[d]) * float64(row[d])
		}
		logits[v] = float32(acc)
	}
	return logits
}
