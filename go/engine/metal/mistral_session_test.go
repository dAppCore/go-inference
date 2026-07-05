// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/mistral"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// mistralBF16Tensors builds a synthetic Ministral-3 bf16 checkpoint under the real multimodal
// wrapper prefix (language_model.model.*) with two stray vision tensors that the text assembler
// must drop. A Mistral layer carries exactly two norms (input_layernorm + post_attention_layernorm)
// and no gemma4 extras. Tied embeddings (no lm_head.weight).
func mistralBF16Tensors(t *testing.T, dModel, nHeads, nKV, headDim, dFF, vocab, numLayers int) map[string]safetensors.Tensor {
	t.Helper()
	ts := map[string]safetensors.Tensor{}
	salt := 1
	mk := func(name string, shape ...int) {
		elems := 1
		for _, dim := range shape {
			elems *= dim
		}
		f := make([]float32, elems)
		for i := range f {
			f[i] = float32((i*salt+7)%83-41) * 0.02
		}
		ts[name] = safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: toBF16Bytes(f)}
		salt++
	}
	qDim, kvDim := nHeads*headDim, nKV*headDim
	mk("language_model.model.embed_tokens.weight", vocab, dModel)
	mk("language_model.model.norm.weight", dModel)
	for i := 0; i < numLayers; i++ {
		p := core.Sprintf("language_model.model.layers.%d", i)
		mk(p+".input_layernorm.weight", dModel)
		mk(p+".post_attention_layernorm.weight", dModel)
		mk(p+".self_attn.q_proj.weight", qDim, dModel)
		mk(p+".self_attn.k_proj.weight", kvDim, dModel)
		mk(p+".self_attn.v_proj.weight", kvDim, dModel)
		mk(p+".self_attn.o_proj.weight", dModel, qDim)
		mk(p+".mlp.gate_proj.weight", dFF, dModel)
		mk(p+".mlp.up_proj.weight", dFF, dModel)
		mk(p+".mlp.down_proj.weight", dModel, dFF)
	}
	// stray non-text towers the assembler must drop (they're not under language_model.)
	mk("vision_tower.transformer.layers.0.attention.q_proj.weight", dModel, dModel)
	mk("multi_modal_projector.linear_1.weight", dModel, dModel)
	return ts
}

// TestLoadMistralBF16 gates the whole Ministral-3 bf16 path: a synthetic Mistral checkpoint
// (under the multimodal wrapper, tied embeddings, two-norm layers, vision tensors to drop)
// assembles into a session that generates; the first token equals the manual chain (embed →
// stepToken → lm_head → greedy) proving the assembler maps the Mistral names — crucially the
// post_attention_layernorm → pre-MLP-norm mapping — and the shared executor runs the faithful
// Mistral layer; and a config.json dir-loads to the same tokens.
func TestLoadMistralBF16(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, maxLen, n = 2, 16, 4
	cfg := mistral.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		RopeParameters: &mistral.RopeParams{RopeTheta: 1_000_000, RopeType: "yarn", Factor: 16},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := mistralBF16Tensors(t, dModel, nHeads, nKV, headDim, dFF, vocab, numLayers)
	prompt := []int32{1, 5, 3}

	// mistral's weight layout = the standard names with two overrides (the same spec
	// pkg/model/mistral/register.go registers): pre-MLP norm is post_attention_layernorm,
	// and there is no gemma-style post-attention norm.
	w := model.StandardWeightNames()
	w.MLPNorm = ".post_attention_layernorm.weight"
	w.PostAttnNorm = ""
	lm, err := model.Assemble(ts, arch, w)
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g := loadedToBF16(lm)
	if !g.Tied {
		t.Fatal("Ministral-3 ties embeddings — LMHead should alias Embed")
	}
	if len(g.Layers[0].QNormW) != 0 || len(g.Layers[0].PostAttnNormW) != 0 || len(g.Layers[0].PostFFNormW) != 0 {
		t.Fatal("a Mistral layer must carry none of the gemma4 norm extras")
	}
	if len(g.Layers[0].MLPNormW) != dModel*bf16Size {
		t.Fatal("pre-MLP norm (post_attention_layernorm) not mapped to MLPNormW")
	}
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
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

	// manual chain: embed → stepToken (the shared executor, all gemma4 extras off) → lm_head → greedy.
	attnScale := float32(1.0 / math.Sqrt(float64(headDim)))
	embedScale := float32(math.Sqrt(float64(dModel)))
	var manualFirst int32
	withAutoreleasePool(func() {
		lb, moeW, _ := buildBF16ArchLayerBufs(g.Layers, arch.Layer, dModel, nHeads, nKV, headDim, dFF, maxLen, arch.SlidingWindow, nil)
		st := newArchDecodeState(arch.Layer, lb, moeW, dModel, nHeads, nKV, headDim, dFF, arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, attnScale, arch.Eps, false, 0)
		var hidden []byte
		for p, id := range prompt {
			embs, err := EmbedTokensBF16(g.Embed, []int32{id}, vocab, dModel, embedScale)
			if err != nil {
				t.Fatalf("EmbedTokensBF16: %v", err)
			}
			if hidden, err = st.stepToken(embs[0], p); err != nil {
				t.Fatalf("stepToken: %v", err)
			}
		}
		logits, err := LMHeadBF16(hidden, g.FinalNorm, g.LMHead, dModel, vocab, arch.Eps, arch.SoftCap)
		if err != nil {
			t.Fatalf("LMHeadBF16: %v", err)
		}
		if manualFirst, err = model.Greedy(logits, vocab); err != nil {
			t.Fatalf("Greedy: %v", err)
		}
	})
	if gen[0] != manualFirst {
		t.Fatalf("session first token %d != manual Mistral chain %d", gen[0], manualFirst)
	}

	// dir-load: config.json (the Mistral config) + weights on disk → LoadDir ≡ in-memory.
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(configJSONWithModelType(t, cfg, "mistral"))); err != nil {
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
		t.Fatalf("dir-loaded Mistral %v != in-memory %v", genDir, gen)
	}
	t.Logf("Ministral-3 bf16 end to end: multimodal-wrapped tensors assemble (tied, two-norm layers, vision dropped) → session generates %v; first token ≡ manual Mistral chain; dir-load ≡ in-memory", gen)
}
