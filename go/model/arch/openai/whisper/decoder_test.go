// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import "testing"

// TestPrecomputeCrossKVPerLayer_Good proves the exported, per-layer PrecomputeCrossKV (decoder.go) fans
// the unexported single-layer precomputeCrossKV (attention.go, already golden-tested in
// TestPrecomputeCrossKV_Good/attention_test.go) out across every decoder layer, returning one CrossKV
// per layer in order.
func TestPrecomputeCrossKVPerLayer_Good(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	if len(crossKV) != len(w.DecoderLayers) {
		t.Fatalf("PrecomputeCrossKV returned %d entries, want %d (one per decoder layer)", len(crossKV), len(w.DecoderLayers))
	}
	wantK, wantV := precomputeCrossKV(encOut, cfg.MaxSourcePositions, w.DecoderLayers[0].CrossAttn)
	if d := maxAbsDiff32(t, crossKV[0].K, wantK); d != 0 {
		t.Fatalf("PrecomputeCrossKV[0].K diverges from the direct single-layer call by %g", d)
	}
	if d := maxAbsDiff32(t, crossKV[0].V, wantV); d != 0 {
		t.Fatalf("PrecomputeCrossKV[0].V diverges from the direct single-layer call by %g", d)
	}
}

// TestDecoderLayerForward_Good replays toy_block_goldens.json's decoder_layer: a real transformers
// WhisperDecoderLayer(config) forward (causal self-attn → cross-attn over a synthetic encoder output →
// FFN) on toy hand-built weights — the exact per-layer sequence DecodeLogits' loop runs. DecodeLogits'
// own end-to-end wiring (embed+pos lookup → N of these layers → final norm → tied head) is proven at
// REAL scale by live_test.go's exact-transcript gate, same reasoning as encoder_test.go's
// TestEncoderLayer_Good.
func TestDecoderLayerForward_Good(t *testing.T) {
	g := readToyBlockGoldens(t)
	dl := g.DecoderLayer
	geo := g.Geometry
	w := DecoderLayerWeights{
		SelfAttnNorm: dl.Weights.SelfAttnLayerNorm.layerNorm(),
		SelfAttn: AttnWeights{
			Q: dl.Weights.SelfQProj.linear(geo.DModel, geo.DModel), K: dl.Weights.SelfKProj.linear(geo.DModel, geo.DModel),
			V: dl.Weights.SelfVProj.linear(geo.DModel, geo.DModel), Out: dl.Weights.SelfOutProj.linear(geo.DModel, geo.DModel),
		},
		CrossAttnNorm: dl.Weights.EncoderAttnLayerNorm.layerNorm(),
		CrossAttn: AttnWeights{
			Q: dl.Weights.CrossQProj.linear(geo.DModel, geo.DModel), K: dl.Weights.CrossKProj.linear(geo.DModel, geo.DModel),
			V: dl.Weights.CrossVProj.linear(geo.DModel, geo.DModel), Out: dl.Weights.CrossOutProj.linear(geo.DModel, geo.DModel),
		},
		FinalNorm: dl.Weights.FinalLayerNorm.layerNorm(),
		FC1:       dl.Weights.FC1.linear(geo.DModel, geo.FFN),
		FC2:       dl.Weights.FC2.linear(geo.FFN, geo.DModel),
	}
	encK, encV := precomputeCrossKV(dl.EncInput, dl.Tenc, w.CrossAttn)

	hidden := dl.DecInput
	residual := hidden
	normed := layerNormForward(hidden, w.SelfAttnNorm, dl.Td, geo.DModel)
	selfOut, err := selfAttentionForward(normed, dl.Td, geo.DModel, geo.Heads, true, w.SelfAttn)
	if err != nil {
		t.Fatalf("selfAttentionForward: %v", err)
	}
	hidden = addRows(residual, selfOut)

	residual = hidden
	normed = layerNormForward(hidden, w.CrossAttnNorm, dl.Td, geo.DModel)
	crossOut, err := crossAttentionForward(normed, dl.Td, geo.DModel, geo.Heads, w.CrossAttn, encK, encV, dl.Tenc)
	if err != nil {
		t.Fatalf("crossAttentionForward: %v", err)
	}
	hidden = addRows(residual, crossOut)

	residual = hidden
	normed = layerNormForward(hidden, w.FinalNorm, dl.Td, geo.DModel)
	ff := linearForward(geluRow(linearForward(normed, w.FC1, dl.Td)), w.FC2, dl.Td)
	hidden = addRows(residual, ff)

	if d := maxAbsDiff32(t, hidden, dl.Output); d > 1e-3 {
		t.Fatalf("decoder layer max abs diff vs reference = %g, want <= 1e-3", d)
	}
}

// TestTiedLMHead_Good replays toy_block_goldens.json's lm_head: a real transformers
// nn.functional.linear(normed_hidden, embed) tied-weight projection — the exact op DecodeLogits' final
// step runs (proj_out.weight IS embed_tokens.weight — weights.go's doc comment).
func TestTiedLMHead_Good(t *testing.T) {
	g := readToyBlockGoldens(t)
	lm := g.LMHead
	normed := layerNormForward(lm.Hidden, LayerNormWeights{Weight: lm.LNWeight, Bias: lm.LNBias}, 1, len(lm.Hidden))
	if d := maxAbsDiff32(t, normed, lm.LayerNormOutput); d > 1e-4 {
		t.Fatalf("lm_head final layer_norm max abs diff = %g, want <= 1e-4", d)
	}
	w := &Weights{EmbedTokens: lm.Embed, VocabSize: lm.Vocab, DModel: len(lm.Hidden)}
	logits := tiedLMHead(normed, w)
	if d := maxAbsDiff32(t, logits, lm.Logits); d > 1e-3 {
		t.Fatalf("tiedLMHead max abs diff vs reference = %g, want <= 1e-3", d)
	}
	if got := argmaxF32(logits); got != int32(lm.Argmax) {
		t.Fatalf("argmax(tiedLMHead logits) = %d, want %d (the reference's own argmax)", got, lm.Argmax)
	}
}

// TestDecodeLogits_Good is a hermetic WIRING test (multi-layer loop, per-layer crossKV indexing, embed+
// pos lookup, bounds) using tinyWhisperTensors — self-consistent shape/no-panic proof; bit-exact
// parity against the reference is TestDecoderLayerForward_Good + TestTiedLMHead_Good (isolated) plus
// live_test.go's real-checkpoint E2E transcript match (composed, at real scale).
func TestDecodeLogits_Good(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	logits, err := DecodeLogits([]int32{0, 1, 2}, crossKV, cfg.MaxSourcePositions, w, cfg)
	if err != nil {
		t.Fatalf("DecodeLogits: %v", err)
	}
	if len(logits) != cfg.VocabSize {
		t.Fatalf("len(logits) = %d, want VocabSize %d", len(logits), cfg.VocabSize)
	}
}

func TestDecodeLogits_Bad(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	if _, err := DecodeLogits(nil, nil, cfg.MaxSourcePositions, w, cfg); err == nil {
		t.Fatal("DecodeLogits accepted an empty token sequence")
	}
}

// TestDecodeLogits_Ugly proves a token id outside the vocab is refused with a specific message rather
// than an out-of-range panic on the embedding lookup.
func TestDecodeLogits_Ugly(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	if _, err := DecodeLogits([]int32{9999}, crossKV, cfg.MaxSourcePositions, w, cfg); err == nil {
		t.Fatal("DecodeLogits accepted a token id outside the vocabulary")
	}
}

// TestNewSelfAttnCache_Good proves one empty cache entry per decoder layer — mirrors PrecomputeCrossKV's
// one-entry-per-layer shape, but empty (self-attention K/V are not known before decoding starts, unlike
// cross-attention K/V which are fixed from the encoder output).
func TestNewSelfAttnCache_Good(t *testing.T) {
	cache := NewSelfAttnCache(4)
	if len(cache) != 4 {
		t.Fatalf("NewSelfAttnCache(4) returned %d entries, want 4", len(cache))
	}
	for i, c := range cache {
		if len(c.K) != 0 || len(c.V) != 0 {
			t.Fatalf("cache[%d] = %+v, want an empty K/V (nothing decoded yet)", i, c)
		}
	}
}

// TestDecodeLogitsStep_Good proves DecodeLogitsStep reproduces DecodeLogits' output BIT-FOR-BIT at every
// position across a growing sequence, exercising the three call shapes GreedyDecode/DetectLanguage
// actually use: (1) a single token from a fresh (empty) cache — DetectLanguage's shape; (2) a multi-token
// PREFILL batch from a fresh cache — GreedyDecode's first iteration, the whole init prompt at once; and
// (3) a single new token appended onto an already-populated cache — GreedyDecode's every iteration after
// the first. See decoder.go's file doc comment for why bit-identity, not mere closeness, is the right bar
// — this is the hermetic half of the #37-tail KV-cache gate (the live half is live_test.go's exact-
// transcript assertion, now running through this same DecodeLogitsStep path).
func TestDecodeLogitsStep_Good(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	ids := []int32{0, 1, 2} // fills tinyWhisperTensors' MaxTargetPositions=3 exactly

	// (1) single token, fresh cache — DetectLanguage's shape.
	wantAt1, err := DecodeLogits(ids[:1], crossKV, cfg.MaxSourcePositions, w, cfg)
	if err != nil {
		t.Fatalf("DecodeLogits(T=1): %v", err)
	}
	freshCache := NewSelfAttnCache(len(w.DecoderLayers))
	gotAt1, err := DecodeLogitsStep(ids[:1], 0, freshCache, crossKV, cfg.MaxSourcePositions, w, cfg)
	if err != nil {
		t.Fatalf("DecodeLogitsStep(fresh, 1 token): %v", err)
	}
	if d := maxAbsDiff32(t, gotAt1, wantAt1); d != 0 {
		t.Fatalf("DecodeLogitsStep(fresh, 1 token) diverges from DecodeLogits(T=1) by %g, want bit-identical", d)
	}

	// (2) multi-token prefill batch, fresh cache — GreedyDecode's first iteration.
	wantAt2, err := DecodeLogits(ids[:2], crossKV, cfg.MaxSourcePositions, w, cfg)
	if err != nil {
		t.Fatalf("DecodeLogits(T=2): %v", err)
	}
	cache := NewSelfAttnCache(len(w.DecoderLayers))
	gotAt2, err := DecodeLogitsStep(ids[:2], 0, cache, crossKV, cfg.MaxSourcePositions, w, cfg)
	if err != nil {
		t.Fatalf("DecodeLogitsStep(fresh, 2-token prefill): %v", err)
	}
	if d := maxAbsDiff32(t, gotAt2, wantAt2); d != 0 {
		t.Fatalf("DecodeLogitsStep(fresh, 2-token prefill) diverges from DecodeLogits(T=2) by %g, want bit-identical", d)
	}

	// (3) single new token appended onto the now-populated cache — GreedyDecode's every later iteration.
	wantAt3, err := DecodeLogits(ids[:3], crossKV, cfg.MaxSourcePositions, w, cfg)
	if err != nil {
		t.Fatalf("DecodeLogits(T=3): %v", err)
	}
	gotAt3, err := DecodeLogitsStep(ids[2:3], 2, cache, crossKV, cfg.MaxSourcePositions, w, cfg)
	if err != nil {
		t.Fatalf("DecodeLogitsStep(continuation, 1 token): %v", err)
	}
	if d := maxAbsDiff32(t, gotAt3, wantAt3); d != 0 {
		t.Fatalf("DecodeLogitsStep(continuation) diverges from DecodeLogits(T=3) by %g, want bit-identical", d)
	}
}

// TestDecodeLogitsStep_Bad mirrors TestDecodeLogits_Bad: an empty new-token batch is refused.
func TestDecodeLogitsStep_Bad(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	cache := NewSelfAttnCache(len(w.DecoderLayers))
	if _, err := DecodeLogitsStep(nil, 0, cache, nil, cfg.MaxSourcePositions, w, cfg); err == nil {
		t.Fatal("DecodeLogitsStep accepted an empty new-token batch")
	}
}

// TestDecodeLogitsStep_Ugly mirrors TestDecodeLogits_Ugly: a token id outside the vocab is refused with a
// specific message rather than an out-of-range panic on the embedding lookup.
func TestDecodeLogitsStep_Ugly(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	cache := NewSelfAttnCache(len(w.DecoderLayers))
	if _, err := DecodeLogitsStep([]int32{9999}, 0, cache, crossKV, cfg.MaxSourcePositions, w, cfg); err == nil {
		t.Fatal("DecodeLogitsStep accepted a token id outside the vocabulary")
	}
}
