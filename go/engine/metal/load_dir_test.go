// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"reflect"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/model/safetensors"
)

func mustEncode(t *testing.T, tensors map[string]safetensors.Tensor) []byte {
	t.Helper()
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	return blob
}

// TestLoadDirBF16 gates the on-disk directory path: a config.json + safetensors written
// to a temp dir — as BOTH a single model.safetensors AND a 2-shard index.json + shards —
// loads via LoadDir into a session generating IDENTICALLY to the in-memory assemble pipe. Proves
// the thin dir/sharded I/O layer (safetensors.LoadDir) + the registry dispatch feed the assembler
// unchanged: real gemma4 checkpoints are always sharded, so this is the load path a real model
// actually takes.
func TestLoadDirBF16(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const headDim, vocab = 64, 32 // headDim 64 so the SDPA kernel exists
	const maxLen = 16
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	tensors, _ := gemma4Tensors(arch, false)
	prompt := []int32{1, 5, 3}
	const n = 4

	configJSON := gemma4ConfigJSON(t, cfg)

	// reference: assemble the tensors in memory (registry) → session → generate.
	lm, err := model.Assemble(tensors, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	refSess, err := NewArchSession(loadedToBF16(lm), arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	want, err := refSess.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("ref Generate: %v", err)
	}

	// write config.json into dir, load it via the registry dir loader, generate.
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

	// (a) single model.safetensors.
	single := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(single, "model.safetensors"), string(mustEncode(t, tensors))); err != nil {
		t.Fatalf("write single: %v", err)
	}
	if got := genFromDir(single); !idsEqual(got, want) {
		t.Fatalf("single-file dir %v != in-memory pipe %v", got, want)
	}

	// (b) two shards + index.json — split the gemma4 tensor set across two files.
	sharded := t.TempDir()
	half1, half2 := map[string]safetensors.Tensor{}, map[string]safetensors.Tensor{}
	wm := map[string]string{}
	i := 0
	for name, tns := range tensors {
		if i%2 == 0 {
			half1[name], wm[name] = tns, "model-00001-of-00002.safetensors"
		} else {
			half2[name], wm[name] = tns, "model-00002-of-00002.safetensors"
		}
		i++
	}
	if err := coreio.Local.Write(core.PathJoin(sharded, "model-00001-of-00002.safetensors"), string(mustEncode(t, half1))); err != nil {
		t.Fatalf("write shard1: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(sharded, "model-00002-of-00002.safetensors"), string(mustEncode(t, half2))); err != nil {
		t.Fatalf("write shard2: %v", err)
	}
	idx := core.JSONMarshal(map[string]any{"weight_map": wm})
	if !idx.OK {
		t.Fatalf("marshal index")
	}
	if err := coreio.Local.Write(core.PathJoin(sharded, "model.safetensors.index.json"), string(idx.Value.([]byte))); err != nil {
		t.Fatalf("write index: %v", err)
	}
	if got := genFromDir(sharded); !idsEqual(got, want) {
		t.Fatalf("sharded dir %v != in-memory pipe %v", got, want)
	}

	t.Logf("dir-load: single + 2-shard checkpoints both → session ≡ in-memory pipe %v (the path a real sharded gemma4 takes)", want)
}

func TestLoadDirDiffusionDecoderTrunk(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const headDim, vocab = 64, 32
	const maxLen = 16
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: headDim, GlobalHeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6, SlidingWindow: 4, MaxPositionEmbeddings: maxLen,
		LayerTypes: []string{"sliding_attention", "full_attention"},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	decoder := make(map[string]safetensors.Tensor)
	for name, tensor := range gemma4TensorsMust(t, arch) {
		decoder["model.decoder."+name] = tensor
	}
	decoder["self_conditioning.pre_norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{128}, Data: make([]byte, 128*2)}
	decoder["self_conditioning.gate_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{256, 128}, Data: make([]byte, 256*128*2)}
	decoder["self_conditioning.up_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{256, 128}, Data: make([]byte, 256*128*2)}
	decoder["self_conditioning.down_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{128, 256}, Data: make([]byte, 128*256*2)}
	decoder["model.encoder.language_model.layers.0.layer_scalar"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{1}, Data: make([]byte, 2)}
	decoder["model.encoder.language_model.layers.1.layer_scalar"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{1}, Data: make([]byte, 2)}

	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(diffusionGemmaConfigJSON(t, cfg))); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(mustEncode(t, decoder))); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}

	s, err := LoadDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadDir(diffusion_gemma trunk): %v", err)
	}
	defer func() { _ = s.Close() }()
	if _, err := s.Generate([]int32{1, 5, 3}, 1, -1); err != nil {
		t.Fatalf("Generate from diffusion trunk: %v", err)
	}
	tm, err := LoadTokenModelDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadTokenModelDir(diffusion_gemma trunk): %v", err)
	}
	nativeTM, ok := tm.(*NativeTokenModel)
	if !ok {
		t.Fatalf("LoadTokenModelDir returned %T, want *NativeTokenModel", tm)
	}
	diffusion := reflect.ValueOf(nativeTM).Elem().FieldByName("diffusion")
	if !diffusion.IsValid() || diffusion.IsNil() {
		t.Fatal("native token model dropped diffusion extras")
	}
}

func diffusionGemmaConfigJSON(t testing.TB, cfg g4.Config) []byte {
	t.Helper()
	var m map[string]any
	if r := core.JSONUnmarshal(configJSONWithModelType(t, cfg, "diffusion_gemma"), &m); !r.OK {
		t.Fatalf("parse diffusion config fixture: %s", r.Error())
	}
	m["canvas_length"] = 4
	m["eos_token_id"] = []int{1, 2}
	out := core.JSONMarshal(m)
	if !out.OK {
		t.Fatalf("marshal diffusion config fixture: %s", out.Error())
	}
	return out.Value.([]byte)
}
