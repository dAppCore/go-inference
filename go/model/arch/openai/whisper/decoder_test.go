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
