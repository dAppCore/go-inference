// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import core "dappco.re/go"

// decoder.go is WhisperDecoder ported host-side: token embedding + the learned position table (absolute
// position = index into the sequence, matching WhisperPositionalEmbedding's plain weight[pos] lookup) →
// N pre-LN layers (causal self-attention → cross-attention over the PRECOMPUTED encoder K/V → FFN) →
// one final top-level LayerNorm → the tied LM head (proj_out.weight IS model.decoder.embed_tokens.weight
// — see weights.go's doc comment).
//
// SELF-ATTENTION KV CACHE (the #37-tail perf slice): GreedyDecode/DetectLanguage now drive
// DecodeLogitsStep, which processes ONLY the newly appended token(s) each call and attends over a
// per-layer SelfAttnCache that grows by one batch per step — replacing the v1 "recompute the full
// sequence from scratch every call" shape this file originally shipped with (host-f32 correctness-first,
// per the design's own phasing). DecodeLogits (below) is UNCHANGED and kept as the whole-sequence
// recompute reference: decoder_test.go's parity test proves DecodeLogitsStep produces BIT-IDENTICAL
// logits at every step, not merely close ones. That equivalence holds because every op in the layer
// stack except self-attention is row-wise (embedding/position lookup, LayerNorm, the FFN's two linear
// projections, GELU, the residual adds — see attention.go/encoder.go) — a row's output depends only on
// that row's own input, never on any other row in the same call — and self-attention itself is causal, so
// row j's output depends only on rows [0,j], which are bit-identical regardless of how many further rows
// a later call also happens to process. Concretely: linearForward/layerNormForward/geluRow/addRows all
// loop per-row independently (T=1 or T=beginIndex, same arithmetic per row either way), and mhaCore's
// weighted-sum loop reads exactly [0,limit) in the same fixed order whether those keys/values came from
// THIS call's own batch or an earlier one now sitting in the cache — see attention.go's file doc comment
// for mhaCore's offset parameter, the mechanism that makes this true. Cross-attention K/V stays
// precomputed once per request (PrecomputeCrossKV, unchanged) — the standard Whisper serving trick, since
// cross K/V depend only on the fixed encoder output; DecodeLogitsStep projects cross-attention Q for only
// the new row(s) too (crossAttentionForward's Tq shrinks from the whole sequence to the new batch), which
// is bit-identical per row for the same row-wise reason.

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

// SelfAttnCache is one decoder layer's growing self-attention K/V cache — appended one step's new row(s)
// at a time by DecodeLogitsStep, attended over in full (via mhaCore's offset parameter) on every
// subsequent step. K/V are flat [n,DModel] (n = tokens appended so far, i.e. len(K)/DModel) so appending
// a step's rows is a plain slice append, matching CrossKV's flat layout.
type SelfAttnCache struct {
	K, V []float32
}

// NewSelfAttnCache allocates one empty cache per decoder layer, ready for DecodeLogitsStep — mirrors
// PrecomputeCrossKV's one-entry-per-layer shape, empty rather than precomputed because self-attention K/V
// depend on the tokens generated so far, not on anything known before decoding starts.
func NewSelfAttnCache(numLayers int) []SelfAttnCache {
	return make([]SelfAttnCache, numLayers)
}

// DecodeLogitsStep is DecodeLogits' incremental twin: runs the decoder stack over ONLY the newly
// appended token ids newIDs (a multi-token PREFILL batch on the very first call — GreedyDecode's whole
// init prompt — or a single token on every call after), starting at absolute position startPos (the
// count of tokens already in cache), appending each layer's self-attention K/V onto cache (mutated in
// place) and attending over the FULL cache (history + this batch, causal). Returns the vocabulary logits
// at the LAST new position only, BIT-IDENTICAL to what DecodeLogits(fullSequenceSoFar, ...) would return
// for that same position — see the file doc comment for why, and decoder_test.go's
// TestDecodeLogitsStep_MatchesDecodeLogits_Good for the proof. This is the function GreedyDecode/
// DetectLanguage actually call (the ~23s → seconds wall-clock receipt on the live whisper-tiny fixture —
// see live_test.go and docs/handover.md).
func DecodeLogitsStep(newIDs []int32, startPos int, cache []SelfAttnCache, crossKV []CrossKV, tenc int, w *Weights, cfg *Config) ([]float32, error) {
	if w == nil || cfg == nil {
		return nil, core.NewError("whisper.DecodeLogitsStep: nil weights/config")
	}
	Tn := len(newIDs)
	if Tn == 0 {
		return nil, core.NewError("whisper.DecodeLogitsStep: no new tokens")
	}
	if startPos < 0 {
		return nil, core.NewError("whisper.DecodeLogitsStep: negative startPos")
	}
	if startPos+Tn > cfg.MaxTargetPositions {
		return nil, core.NewError(core.Sprintf("whisper.DecodeLogitsStep: sequence length %d exceeds max_target_positions %d", startPos+Tn, cfg.MaxTargetPositions))
	}
	if len(crossKV) != len(w.DecoderLayers) || len(cache) != len(w.DecoderLayers) {
		return nil, core.NewError("whisper.DecodeLogitsStep: cache/crossKV must have one entry per decoder layer")
	}
	D := cfg.DModel
	hidden := make([]float32, Tn*D)
	for t, id := range newIDs {
		if int(id) < 0 || int(id) >= cfg.VocabSize {
			return nil, core.NewError(core.Sprintf("whisper.DecodeLogitsStep: token id %d at position %d is out of vocab range [0,%d)", id, startPos+t, cfg.VocabSize))
		}
		embedRow := w.EmbedTokens[int(id)*D : int(id)*D+D]
		posRow := w.DecoderPos[(startPos+t)*D : (startPos+t)*D+D]
		for d := range D {
			hidden[t*D+d] = embedRow[d] + posRow[d]
		}
	}

	for li := range w.DecoderLayers {
		layer := w.DecoderLayers[li]
		residual := hidden
		normed := layerNormForward(hidden, layer.SelfAttnNorm, Tn, D)
		selfOut, err := selfAttentionForwardCached(normed, Tn, D, cfg.DecoderAttentionHeads, layer.SelfAttn, &cache[li])
		if err != nil {
			return nil, err
		}
		hidden = addRows(residual, selfOut)

		residual = hidden
		normed = layerNormForward(hidden, layer.CrossAttnNorm, Tn, D)
		crossOut, err := crossAttentionForward(normed, Tn, D, cfg.DecoderAttentionHeads, layer.CrossAttn, crossKV[li].K, crossKV[li].V, tenc)
		if err != nil {
			return nil, err
		}
		hidden = addRows(residual, crossOut)

		residual = hidden
		normed = layerNormForward(hidden, layer.FinalNorm, Tn, D)
		ff := linearForward(geluRow(linearForward(normed, layer.FC1, Tn)), layer.FC2, Tn)
		hidden = addRows(residual, ff)
	}

	normedFinal := layerNormForward(hidden, w.DecoderFinalNorm, Tn, D)
	lastRow := normedFinal[(Tn-1)*D : Tn*D]
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
