// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// addPLETensors adds the gemma4 per-layer-input tower tensors (E2B/E4B) to a quant checkpoint:
// the 4-bit per-layer embedding, the bf16 model projection + norm, and per-layer 4-bit gate +
// projection + bf16 post-norm — sized from the Arch's PLE dims.
func addPLETensors(t testing.TB, ts map[string]safetensors.Tensor, arch model.Arch, gs, bits int) {
	t.Helper()
	vocabPLI, numLayers, pliDim, dModel := arch.PerLayerInputVocab, len(arch.Layer), arch.PerLayerInputHidden, arch.Hidden
	plDim := numLayers * pliDim
	salt := 50
	mkBF16 := func(name string, elems int) {
		f := make([]float32, elems)
		for i := range f {
			f[i] = float32((i*salt+7)%83-41) * 0.02
		}
		ts[name] = safetensors.Tensor{Dtype: "BF16", Shape: []int{elems}, Data: toBF16Bytes(f)}
		salt++
	}
	mkQuant := func(prefix string, outDim, inDim int) {
		p, s, b := quantizeProj(t, outDim, inDim, gs, bits, salt)
		salt++
		ts[prefix+".weight"] = safetensors.Tensor{Dtype: "U32", Shape: []int{outDim, inDim * bits / 32}, Data: p}
		ts[prefix+".scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: s}
		ts[prefix+".biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: b}
	}
	mkQuant("model.embed_tokens_per_layer", vocabPLI, plDim)
	mkBF16("model.per_layer_model_projection.weight", plDim*dModel)
	mkBF16("model.per_layer_projection_norm.weight", pliDim)
	for i := range numLayers {
		p := core.Sprintf("model.layers.%d", i)
		mkQuant(p+".per_layer_input_gate", pliDim, dModel)
		mkQuant(p+".per_layer_projection", dModel, pliDim)
		mkBF16(p+".post_per_layer_input_norm.weight", dModel)
	}
}

// TestLoadGemma4QuantPLE gates the whole E2B/E4B integration: a synthetic 4-bit gemma4 WITH the
// per-layer-input tower assembles (HasPLE), the session generates, the first token equals a
// manual per-token chain (embed → PerLayerInputs → stepToken-with-gate → lm_head → greedy —
// proving the session computes + threads the per-layer-input tensor each token), and a config +
// weights written to a dir load to the same tokens through LoadDir.
func TestLoadGemma4QuantPLE(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim, gs, bits = 2, 64, 64, 4
	const maxLen, n = 16, 4
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	addPLETensors(t, ts, arch, gs, bits)
	prompt := []int32{1, 5, 3}

	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	if !g.HasPLE() {
		t.Fatal("assembled model should have the per-layer-input tower")
	}
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	gen, err := sess.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	for i, id := range gen {
		if id < 0 || int(id) >= vocab {
			t.Fatalf("token %d = %d out of range", i, id)
		}
	}

	// manual per-token chain: replicate what the session must do (PerLayerInputs each token,
	// fed this token's embedding, gating every layer) and check the first generated token.
	attnScale := arch.AttnScale // the model-declared scale (gemma4 1.0), matching the session
	embedScale := float32(math.Sqrt(float64(dModel)))
	var manualFirst int32
	withAutoreleasePool(func() {
		lb, _, _ := buildQuantArchLayerBufs(g.Layers, arch.Layer, dModel, nHeads, nKV, headDim, dFF, maxLen, arch.SlidingWindow, nil)
		st := newArchDecodeState(arch.Layer, lb, make([]*MoELayerWeights, numLayers), dModel, nHeads, nKV, headDim, dFF, arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, attnScale, arch.Eps, false, 0)
		st.pliDim = pliDim
		st.ple = make([]pleLayer, numLayers)
		for i := range g.Layers {
			st.ple[i] = pleLayer{gate: g.Layers[i].PerLayerGate, proj: g.Layers[i].PerLayerProjection, postNorm: g.Layers[i].PostPerLayerInputNormW, groupSize: gs, bits: bits}
		}
		var hidden []byte
		for p, id := range prompt {
			embs, err := EmbedTokensQuant(g.Embed, g.EmbedScales, g.EmbedBiases, []int32{id}, vocab, dModel, gs, bits, embedScale)
			if err != nil {
				t.Fatalf("EmbedTokensQuant: %v", err)
			}
			pli, err := PerLayerInputs(g.EmbedPerLayer, g.EmbedPerLayerScales, g.EmbedPerLayerBiases, g.PerLayerModelProjW, g.PerLayerModelProjScales, g.PerLayerModelProjBiases, g.PerLayerProjNormW, id, embs[0], arch.PerLayerInputVocab, numLayers, pliDim, dModel, gs, bits, g.PerLayerModelProjGS, g.PerLayerModelProjBits, arch.Eps, bufView{})
			if err != nil {
				t.Fatalf("PerLayerInputs: %v", err)
			}
			st.perLayerInput = pli
			if hidden, err = st.stepToken(embs[0], p); err != nil {
				t.Fatalf("stepToken: %v", err)
			}
		}
		logits, err := LMHeadQuant(hidden, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, dModel, vocab, gs, bits, arch.Eps, arch.SoftCap)
		if err != nil {
			t.Fatalf("LMHeadQuant: %v", err)
		}
		if manualFirst, err = model.Greedy(logits, vocab); err != nil {
			t.Fatalf("Greedy: %v", err)
		}
	})
	if gen[0] != manualFirst {
		t.Fatalf("session first token %d != manual PLE chain %d", gen[0], manualFirst)
	}

	// dir-load: config + weights on disk → LoadDir ≡ in-memory.
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(gemma4ConfigJSON(t, cfg))); err != nil {
		t.Fatalf("write config: %v", err)
	}
	blob, err := safetensors.Encode(ts)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	dirSess, err := LoadDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	genDir, err := dirSess.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("dir Generate: %v", err)
	}
	if !idsEqual(genDir, gen) {
		t.Fatalf("dir-loaded PLE model %v != in-memory %v", genDir, gen)
	}
	t.Logf("E2B/E4B PLE end to end: assemble (HasPLE) → session generates %v; first token ≡ manual per-token PLE chain; dir-load ≡ in-memory", gen)
}
