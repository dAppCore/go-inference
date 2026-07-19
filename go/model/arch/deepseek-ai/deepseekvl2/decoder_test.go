// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"strconv"
	"testing"
)

// decoder_test.go gates the MoE decoder's ported ops against decoder_toy_golden.json (a REAL
// 2-layer DeepseekV2Model — layer 0 dense, layer 1 MoE with 4 toy routed experts — constructed
// directly and run with use_cache=False, see golden_test.go's doc comment) and
// moe_gate_real_golden.json (the REAL n_routed_experts=64/num_experts_per_tok=6 routing scale).

func decoderConfigFromGolden(g decoderToyConfigGolden) *Config {
	return &Config{
		VocabSize: g.VocabSize, HiddenSize: g.HiddenSize, IntermediateSize: g.IntermediateSize,
		MoEIntermediateSize: g.MoEIntermediateSize, NumHiddenLayers: g.NumHiddenLayers,
		NumAttentionHeads: g.NumAttentionHeads, NumKeyValueHeads: g.NumKeyValueHeads,
		NSharedExperts: g.NSharedExperts, NRoutedExperts: g.NRoutedExperts,
		NumExpertsPerTok: g.NumExpertsPerTok, FirstKDenseReplace: g.FirstKDenseReplace,
		MaxPositionEmbeddings: 4096, EOSTokenID: -1, // never hit in these tests
	}
}

func decoderWeightsFromGolden(t *testing.T, g decoderToyGolden) *Weights {
	t.Helper()
	w := g.Weights
	cfg := g.Config

	layer0 := DecoderLayerWeights{
		InputNormW: w.mustGet(t, "layers.0.input_layernorm.weight"),
		QW:         w.mustGet(t, "layers.0.self_attn.q_proj.weight"), KW: w.mustGet(t, "layers.0.self_attn.k_proj.weight"),
		VW: w.mustGet(t, "layers.0.self_attn.v_proj.weight"), OW: w.mustGet(t, "layers.0.self_attn.o_proj.weight"),
		PostAttnNormW: w.mustGet(t, "layers.0.post_attention_layernorm.weight"),
		IsMoE:         false,
		DenseGateW:    w.mustGet(t, "layers.0.mlp.gate_proj.weight"), DenseUpW: w.mustGet(t, "layers.0.mlp.up_proj.weight"),
		DenseDownW: w.mustGet(t, "layers.0.mlp.down_proj.weight"),
	}

	experts := make([]DecoderExpertWeights, cfg.NRoutedExperts)
	for e := range experts {
		p := "layers.1.mlp.experts." + strconv.Itoa(e) + "."
		experts[e] = DecoderExpertWeights{
			GateW: w.mustGet(t, p+"gate_proj.weight"), UpW: w.mustGet(t, p+"up_proj.weight"), DownW: w.mustGet(t, p+"down_proj.weight"),
		}
	}
	layer1 := DecoderLayerWeights{
		InputNormW: w.mustGet(t, "layers.1.input_layernorm.weight"),
		QW:         w.mustGet(t, "layers.1.self_attn.q_proj.weight"), KW: w.mustGet(t, "layers.1.self_attn.k_proj.weight"),
		VW: w.mustGet(t, "layers.1.self_attn.v_proj.weight"), OW: w.mustGet(t, "layers.1.self_attn.o_proj.weight"),
		PostAttnNormW: w.mustGet(t, "layers.1.post_attention_layernorm.weight"),
		IsMoE:         true,
		GateWeight:    w.mustGet(t, "layers.1.mlp.gate.weight"),
		Experts:       experts,
		SharedGateW:   w.mustGet(t, "layers.1.mlp.shared_experts.gate_proj.weight"),
		SharedUpW:     w.mustGet(t, "layers.1.mlp.shared_experts.up_proj.weight"),
		SharedDownW:   w.mustGet(t, "layers.1.mlp.shared_experts.down_proj.weight"),
	}

	return &Weights{
		Decoder: DecoderWeights{
			EmbedTokens:  w.mustGet(t, "embed_tokens.weight"),
			Layers:       []DecoderLayerWeights{layer0, layer1},
			FinalNormW:   w.mustGet(t, "norm.weight"),
			LMHeadWeight: g.LMHeadWeight,
		},
	}
}

func embedsFromIDs(ids []int32, embedTokens []float32, hidden int) []float32 {
	out := make([]float32, len(ids)*hidden)
	for i, id := range ids {
		copy(out[i*hidden:(i+1)*hidden], embedTokens[int(id)*hidden:int(id)*hidden+hidden])
	}
	return out
}

// TestDecoderLayerForward_Dense_Golden_Good pins the embedding lookup AND layer 0's dense
// (RoPE causal attention + SwiGLU MLP) forward against the real DeepseekV2Model's captured
// per-layer hidden states (output_hidden_states=True — see golden_test.go's doc comment for why
// index 0 is the raw embedding and index 1 is layer 0's output).
func TestDecoderLayerForward_Dense_Golden_Good(t *testing.T) {
	g := readDecoderToyGolden(t)
	w := decoderWeightsFromGolden(t, g)
	h := g.Config.HiddenSize
	headDim := h / g.Config.NumAttentionHeads

	embeds := embedsFromIDs(g.InputIDs, w.Decoder.EmbedTokens, h)
	if d := maxAbsDiff32(t, embeds, g.HiddenStatesPerLayer[0]); d > 1e-4 {
		t.Fatalf("embedding lookup max abs diff %g, want <=1e-4", d)
	}

	afterLayer0 := decoderLayerForward(embeds, 0, h, g.Config.NumAttentionHeads, headDim, g.Config.MoEIntermediateSize, g.Config.NRoutedExperts, g.Config.NumExpertsPerTok, w.Decoder.Layers[0])
	if d := maxAbsDiff32(t, afterLayer0, g.HiddenStatesPerLayer[1]); d > 1e-3 {
		t.Fatalf("layer 0 (dense) output max abs diff %g, want <=1e-3", d)
	}
}

// TestDecoderLayerForward_MoE_Golden_Good pins layer 1's MoE (routed-expert combine + shared
// expert) forward, THEN the final RMSNorm, against the real reference's post-norm
// last_hidden_state (hidden_states_per_layer[2] IS post-norm — DeepseekV2Model.forward appends
// its output_hidden_states entries BEFORE each layer runs and once more AFTER the final norm, see
// golden_test.go's doc comment) — cross-checked against the golden's separate FinalHidden field
// too (both must agree, since they are the same reference tensor read two different ways).
func TestDecoderLayerForward_MoE_Golden_Good(t *testing.T) {
	g := readDecoderToyGolden(t)
	w := decoderWeightsFromGolden(t, g)
	h := g.Config.HiddenSize
	headDim := h / g.Config.NumAttentionHeads

	embeds := embedsFromIDs(g.InputIDs, w.Decoder.EmbedTokens, h)
	afterLayer0 := decoderLayerForward(embeds, 0, h, g.Config.NumAttentionHeads, headDim, g.Config.MoEIntermediateSize, g.Config.NRoutedExperts, g.Config.NumExpertsPerTok, w.Decoder.Layers[0])
	afterLayer1 := decoderLayerForward(afterLayer0, 0, h, g.Config.NumAttentionHeads, headDim, g.Config.MoEIntermediateSize, g.Config.NRoutedExperts, g.Config.NumExpertsPerTok, w.Decoder.Layers[1])
	final := rmsNorm(afterLayer1, w.Decoder.FinalNormW, h, float32(g.Config.RMSNormEps))

	if d := maxAbsDiff32(t, final, g.HiddenStatesPerLayer[2]); d > 1e-3 {
		t.Fatalf("post-norm final hidden max abs diff %g, want <=1e-3", d)
	}
	if d := maxAbsDiff32(t, final, g.FinalHidden); d > 1e-3 {
		t.Fatalf("post-norm final hidden vs FinalHidden max abs diff %g, want <=1e-3", d)
	}
}

// TestDecodeLogits_Golden_Good pins the WHOLE top-level orchestration (every layer, final norm,
// lm_head) against the golden's logits and argmax in one call. The capture script's lm_head runs
// over EVERY position (logits shape [1,5,vocab], matching what `self.lm_head(hidden_states)` -- no
// slicing -- produces in the reference's own forward), but DecodeLogits only ever returns the
// LAST position (the one a greedy decode step needs, mirroring whisper.DecodeLogits) — so the
// comparison slices the golden down to its own last row first.
func TestDecodeLogits_Golden_Good(t *testing.T) {
	g := readDecoderToyGolden(t)
	w := decoderWeightsFromGolden(t, g)
	cfg := decoderConfigFromGolden(g.Config)
	embeds := embedsFromIDs(g.InputIDs, w.Decoder.EmbedTokens, cfg.HiddenSize)

	logits, err := DecodeLogits(embeds, cfg, w)
	if err != nil {
		t.Fatalf("DecodeLogits: %v", err)
	}
	lastRowLogits := g.Logits[len(g.Logits)-cfg.VocabSize:]
	if d := maxAbsDiff32(t, logits, lastRowLogits); d > 1e-2 {
		t.Fatalf("logits max abs diff %g, want <=1e-2", d)
	}
	if argmaxF32(logits) != g.ArgmaxLast {
		t.Fatalf("argmax = %d, want %d", argmaxF32(logits), g.ArgmaxLast)
	}
}

// TestDecodeLogitsStep_MatchesDecodeLogits_Good proves the cached incremental path
// (DecodeLogitsStep, fed the WHOLE prompt as one prefill batch) is BIT-IDENTICAL to the
// whole-sequence no-cache reference (DecodeLogits) for the same input — the equivalence
// decoder.go's file doc comment argues for and every real GreedyDecode call depends on. Mirrors
// whisper's TestDecodeLogitsStep_MatchesDecodeLogits_Good precedent (arch/openai/whisper/
// decoder_test.go).
func TestDecodeLogitsStep_MatchesDecodeLogits_Good(t *testing.T) {
	g := readDecoderToyGolden(t)
	w := decoderWeightsFromGolden(t, g)
	cfg := decoderConfigFromGolden(g.Config)
	embeds := embedsFromIDs(g.InputIDs, w.Decoder.EmbedTokens, cfg.HiddenSize)

	want, err := DecodeLogits(embeds, cfg, w)
	if err != nil {
		t.Fatalf("DecodeLogits: %v", err)
	}
	cache := NewSelfAttnCache(len(w.Decoder.Layers))
	got, err := DecodeLogitsStep(embeds, 0, cache, cfg, w)
	if err != nil {
		t.Fatalf("DecodeLogitsStep: %v", err)
	}
	if d := maxAbsDiff32(t, got, want); d > 1e-6 {
		t.Fatalf("DecodeLogitsStep vs DecodeLogits max abs diff %g, want bit-identical (<=1e-6)", d)
	}
}

// TestMoERoute_Toy_Golden_Good pins moeRoute's per-token top-K SELECTION (the SET of expert
// indices and their raw softmax weights — never the return order, which torch.topk(sorted=False)
// does not guarantee either, so comparing order would be a fragile, unspecified assertion) against
// the toy golden's captured per-token routing.
func TestMoERoute_Toy_Golden_Good(t *testing.T) {
	g := readDecoderToyGolden(t)
	mg := g.MoEGateLayer1
	gateWeight := g.Weights.mustGet(t, "layers.1.mlp.gate.weight")
	hidden := g.Config.HiddenSize
	tokens := len(mg.Input) / hidden

	for tk := range tokens {
		xi := mg.Input[tk*hidden : (tk+1)*hidden]
		got := moeRoute(xi, gateWeight, hidden, g.Config.NRoutedExperts, g.Config.NumExpertsPerTok)
		gotMap := make(map[int]float32, len(got))
		for _, e := range got {
			gotMap[e.idx] = e.weight
		}
		wantIdx := mg.TopkIdx[tk]
		if len(wantIdx) != len(got) {
			t.Fatalf("token %d: selected %d experts, want %d", tk, len(got), len(wantIdx))
		}
		for wi, idx := range wantIdx {
			wantWeight := mg.TopkWeight[tk*len(wantIdx)+wi]
			gotWeight, ok := gotMap[idx]
			if !ok {
				t.Fatalf("token %d: expert %d not selected, want selected (weight %g)", tk, idx, wantWeight)
			}
			if diff := absF32(gotWeight - wantWeight); diff > 1e-4 {
				t.Fatalf("token %d expert %d: weight %g, want %g (diff %g)", tk, idx, gotWeight, wantWeight, diff)
			}
		}
	}
}

// TestMoERoute_RealScale_Golden_Good pins moeRoute at the CHECKPOINT'S REAL routing dimensions
// (n_routed_experts=64, num_experts_per_tok=6 — the toy golden above only exercises 4
// experts/top-2, too small to meaningfully prove top-6-of-64 selection).
func TestMoERoute_RealScale_Golden_Good(t *testing.T) {
	g := readMoEGateRealGolden(t)
	tokens := len(g.Input) / g.HiddenSize
	for tk := range tokens {
		xi := g.Input[tk*g.HiddenSize : (tk+1)*g.HiddenSize]
		got := moeRoute(xi, g.GateWeight, g.HiddenSize, g.NRoutedExperts, g.NumExpertsPerTok)
		if len(got) != g.NumExpertsPerTok {
			t.Fatalf("token %d: selected %d experts, want %d", tk, len(got), g.NumExpertsPerTok)
		}
		gotMap := make(map[int]float32, len(got))
		for _, e := range got {
			gotMap[e.idx] = e.weight
		}
		wantIdx := g.TopkIdx[tk]
		for wi, idx := range wantIdx {
			wantWeight := g.TopkWeight[tk*len(wantIdx)+wi]
			gotWeight, ok := gotMap[idx]
			if !ok {
				t.Fatalf("token %d: expert %d not selected at real scale, want selected (weight %g)", tk, idx, wantWeight)
			}
			if diff := absF32(gotWeight - wantWeight); diff > 1e-4 {
				t.Fatalf("token %d expert %d: weight %g, want %g (diff %g)", tk, idx, gotWeight, wantWeight, diff)
			}
		}
	}
}

func absF32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// TestDecodeLogits_Bad proves a hidden-width mismatch (embeds not a whole number of hidden-width
// rows) is refused before any layer runs.
func TestDecodeLogits_Bad(t *testing.T) {
	cfg := &Config{HiddenSize: 8, NumHiddenLayers: 1, NumAttentionHeads: 2}
	w := &Weights{Decoder: DecoderWeights{Layers: make([]DecoderLayerWeights, 1)}}
	if _, err := DecodeLogits(make([]float32, 5), cfg, w); err == nil {
		t.Fatal("DecodeLogits accepted an embeds buffer that is not a whole number of hidden-width rows")
	}
}

// TestDecodeLogitsStep_Ugly proves a cache slice whose length doesn't match the layer count is
// refused — distinct from _Bad's malformed-embeds-buffer case.
func TestDecodeLogitsStep_Ugly(t *testing.T) {
	cfg := &Config{HiddenSize: 8, NumHiddenLayers: 2, NumAttentionHeads: 2}
	w := &Weights{Decoder: DecoderWeights{Layers: make([]DecoderLayerWeights, 2)}}
	wrongCache := NewSelfAttnCache(1) // 1 entry, but the config declares 2 layers
	if _, err := DecodeLogitsStep(make([]float32, 8), 0, wrongCache, cfg, w); err == nil {
		t.Fatal("DecodeLogitsStep accepted a cache slice whose length does not match the layer count")
	}
}
