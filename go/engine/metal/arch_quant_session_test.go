// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/safetensors"
)

// quantizeProj and quantGemma4Tensors (this file's synthetic 4-bit gemma4 checkpoint builders) now
// live in test_helpers_test.go, reimplemented in pure Go (no cgo/metal) — they are shared by many
// other untagged test files across the package, so they can't depend on the metal_runtime lane.

// TestLoadGemma4TokenModelDir gates the contract loader: a synthetic 4-bit gemma4 on
// disk loads via LoadTokenModelDir into a model.TokenModel that model.Generate
// drives to the SAME tokens as the model assembled in memory — the dir → contract
// path the no-cgo serve adapter (mlx.LoadNativeTextModel) builds on.
func TestLoadGemma4TokenModelDir(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const gs, bits = 32, 4
	const maxLen, n = 16, 4
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	prompt := []int32{1, 5, 3}

	// in-memory reference: assemble (registry) + NewQuantTokenModel + model.Generate.
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	refTM, err := NewQuantTokenModel(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewQuantTokenModel: %v", err)
	}
	want, err := model.Generate(refTM, prompt, n, -1)
	if err != nil {
		t.Fatalf("ref Generate: %v", err)
	}

	// on disk → LoadTokenModelDir → model.Generate.
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
	tm, err := LoadTokenModelDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadTokenModelDir: %v", err)
	}
	got, err := model.Generate(tm, prompt, n, -1)
	if err != nil {
		t.Fatalf("dir Generate: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("dir-loaded %d tokens, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("dir-loaded token %d = %d, in-memory = %d (%v vs %v)", i, got[i], want[i], got, want)
		}
	}
	t.Logf("contract loader: LoadTokenModelDir ≡ in-memory NewQuantTokenModel = %v", got)
}

// TestLoadGemma4Quant4Dir gates the whole 4-bit load+session path: a synthetic 4-bit gemma4
// assembles into a quant session that generates; the FIRST generated token equals the gated
// whole-sequence quant chain (EmbedTokensQuant → DecodeForwardArchQuant → LMHeadQuant →
// greedy); and a config.json + weights written to a temp dir — single AND sharded — load to
// the same tokens. The model is all-global so the session's per-type RoPE coincides with
// DecodeForwardArchQuant's one base (a sliding model would legitimately diverge there).
func TestLoadGemma4Quant4Dir(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const gs, bits = 32, 4
	const maxLen, n = 16, 4
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	prompt := []int32{1, 5, 3}

	// direct: assemble in memory (registry) → quant session → generate.
	lmDirect, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	gDirect, err := loadedToQuant(lmDirect, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	sd, err := NewArchQuantSession(gDirect, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	genDirect, err := sd.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("direct Generate: %v", err)
	}
	if len(genDirect) != n {
		t.Fatalf("generated %d tokens, want %d", len(genDirect), n)
	}
	for i, id := range genDirect {
		if id < 0 || int(id) >= arch.Vocab {
			t.Fatalf("token %d = %d out of [0,%d)", i, id, arch.Vocab)
		}
	}

	// correctness: the first generated token ≡ the gated whole-seq quant chain.
	embs, err := EmbedTokensQuant(gDirect.Embed, gDirect.EmbedScales, gDirect.EmbedBiases, prompt, arch.Vocab, arch.Hidden, gs, bits, float32(math.Sqrt(float64(arch.Hidden))))
	if err != nil {
		t.Fatalf("EmbedTokensQuant: %v", err)
	}
	attnScale := arch.AttnScale // the model-declared scale (gemma4 1.0), matching the session
	hs, err := DecodeForwardArchQuant(embs, gDirect.Layers, arch.Layer, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, maxLen, arch.FF, arch.SlidingWindow, arch.RopeBase, attnScale, arch.Eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant: %v", err)
	}
	logits, err := LMHeadQuant(hs[len(hs)-1], gDirect.FinalNorm, gDirect.LMHead, gDirect.LMHeadScales, gDirect.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, arch.SoftCap)
	if err != nil {
		t.Fatalf("LMHeadQuant: %v", err)
	}
	wantFirst, err := model.Greedy(logits, arch.Vocab)
	if err != nil {
		t.Fatalf("Greedy: %v", err)
	}
	if genDirect[0] != wantFirst {
		t.Fatalf("quant session first token %d != whole-seq quant chain %d", genDirect[0], wantFirst)
	}

	// dir round-trip: write config.json + weights, single AND sharded → LoadDir ≡ direct.
	configJSON := gemma4ConfigJSON(t, cfg)
	genFromDir := func(dir string) []int32 {
		if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(configJSON)); err != nil {
			t.Fatalf("write config.json: %v", err)
		}
		s, err := LoadDir(dir, maxLen)
		if err != nil {
			t.Fatalf("LoadDir(%s): %v", dir, err)
		}
		out, err := s.Generate(prompt, n, -1)
		if err != nil {
			t.Fatalf("dir Generate: %v", err)
		}
		return out
	}

	single := t.TempDir()
	blob, err := safetensors.Encode(ts)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(single, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write single: %v", err)
	}
	if got := genFromDir(single); !idsEqual(got, genDirect) {
		t.Fatalf("single-file dir %v != in-memory %v", got, genDirect)
	}

	sharded := t.TempDir()
	half1, half2 := map[string]safetensors.Tensor{}, map[string]safetensors.Tensor{}
	wm := map[string]string{}
	i := 0
	for name, tns := range ts {
		if i%2 == 0 {
			half1[name], wm[name] = tns, "model-00001-of-00002.safetensors"
		} else {
			half2[name], wm[name] = tns, "model-00002-of-00002.safetensors"
		}
		i++
	}
	b1, err := safetensors.Encode(half1)
	if err != nil {
		t.Fatalf("Encode shard1: %v", err)
	}
	b2, err := safetensors.Encode(half2)
	if err != nil {
		t.Fatalf("Encode shard2: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(sharded, "model-00001-of-00002.safetensors"), string(b1)); err != nil {
		t.Fatalf("write shard1: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(sharded, "model-00002-of-00002.safetensors"), string(b2)); err != nil {
		t.Fatalf("write shard2: %v", err)
	}
	idx := core.JSONMarshal(map[string]any{"weight_map": wm})
	if !idx.OK {
		t.Fatalf("marshal index")
	}
	if err := coreio.Local.Write(core.PathJoin(sharded, "model.safetensors.index.json"), string(idx.Value.([]byte))); err != nil {
		t.Fatalf("write index: %v", err)
	}
	if got := genFromDir(sharded); !idsEqual(got, genDirect) {
		t.Fatalf("sharded dir %v != in-memory %v", got, genDirect)
	}

	t.Logf("4-bit dir-load: assemble → quant session generates %v; first token ≡ whole-seq quant chain; single + sharded dirs ≡ in-memory (the path mlx-community 4-bit takes)", genDirect)
}
