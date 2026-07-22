// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/gguf"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/gemma4"
)

func TestResolveGemma4ArchitectureDeclaration_Good(t *testing.T) {
	config := []byte(`{
		"model_type":"Gemma4ForCausalLM",
		"architectures":["Gemma4ForCausalLM"],
		"hidden_size":256,"num_hidden_layers":3,"intermediate_size":512,
		"num_attention_heads":8,"num_key_value_heads":2,"head_dim":32,
		"vocab_size":1000,"rms_norm_eps":0.00001,"sliding_window":128,
		"max_position_embeddings":2048,"num_kv_shared_layers":1,
		"layer_types":["full_attention","sliding_attention","full_attention"]
	}`)

	declaration, err := ResolveGemma4ArchitectureDeclaration(config)
	if err != nil {
		t.Fatalf("ResolveGemma4ArchitectureDeclaration: %v", err)
	}
	if declaration.Resolution.Architecture != "gemma4_text" {
		t.Fatalf("resolved architecture = %q, want gemma4_text", declaration.Resolution.Architecture)
	}
	if declaration.Arch.Hidden != 256 || declaration.Arch.Heads != 8 || len(declaration.Arch.Layer) != 3 {
		t.Fatalf("shared Arch = %+v, want 256 hidden / 8 heads / 3 layers", declaration.Arch)
	}
	if !declaration.Matched() {
		t.Fatalf("declaration = %+v, want identity and topology match", declaration)
	}

	want := model.DeriveLayers([]string{"full_attention", "sliding_attention", "full_attention"}, 1)
	for i, layer := range want {
		layer.HeadDim, layer.KVHeads = 32, 2
		if declaration.Arch.Layer[i] != layer {
			t.Fatalf("layer %d = %+v, want %+v", i, declaration.Arch.Layer[i], layer)
		}
	}
}

func TestResolveGemma4ArchitectureDeclaration_DiffusionGemma_Good(t *testing.T) {
	config := []byte(`{
		"model_type":"diffusion_gemma",
		"architectures":["DiffusionGemmaForBlockDiffusion"],
		"hidden_size":256,"num_hidden_layers":3,"intermediate_size":512,
		"num_attention_heads":8,"num_key_value_heads":2,"head_dim":32,
		"global_head_dim":32,"vocab_size":1000,"rms_norm_eps":0.00001,
		"sliding_window":128,"max_position_embeddings":2048,
		"num_kv_shared_layers":1,
		"layer_types":["full_attention","sliding_attention","full_attention"]
	}`)

	declaration, err := ResolveGemma4ArchitectureDeclaration(config)
	if err != nil {
		t.Fatalf("ResolveGemma4ArchitectureDeclaration(diffusion_gemma): %v", err)
	}
	if declaration.Resolution.Architecture != "diffusion_gemma" {
		t.Fatalf("resolved architecture = %q, want diffusion_gemma", declaration.Resolution.Architecture)
	}
	if declaration.Arch.Hidden != 256 || declaration.Arch.Heads != 8 || len(declaration.Arch.Layer) != 3 {
		t.Fatalf("shared Arch = %+v, want 256 hidden / 8 heads / 3 layers", declaration.Arch)
	}
	if !declaration.Matched() {
		t.Fatalf("declaration = %+v, want DiffusionGemma trunk identity and topology match", declaration)
	}
}

func TestResolveGemma4ArchitectureDeclaration_AliasFallback(t *testing.T) {
	config := []byte(`{
		"architectures":["Gemma4UnifiedForConditionalGeneration"],
		"hidden_size":128,"num_hidden_layers":2,"intermediate_size":256,
		"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,
		"vocab_size":1000,"rms_norm_eps":0.00001,"sliding_window":64,
		"max_position_embeddings":1024,
		"layer_types":["sliding_attention","full_attention"]
	}`)

	declaration, err := ResolveGemma4ArchitectureDeclaration(config)
	if err != nil {
		t.Fatalf("ResolveGemma4ArchitectureDeclaration: %v", err)
	}
	if declaration.Resolution.Architecture != "gemma4_unified" {
		t.Fatalf("resolved architecture = %q, want gemma4_unified", declaration.Resolution.Architecture)
	}
	if declaration.Resolution.Source != "architectures" {
		t.Fatalf("resolution source = %q, want architectures", declaration.Resolution.Source)
	}
	if declaration.Topology.LayerTypes[0] != "sliding_attention" || declaration.Topology.CacheIndex[0] != 0 {
		t.Fatalf("normalized topology = %+v, want the shared model topology", declaration.Topology)
	}
}

func TestResolveGemma4ArchitectureDeclaration_Bad(t *testing.T) {
	for _, config := range [][]byte{
		[]byte(`{"model_type":"not_gemma4"}`),
		[]byte(`{"model_type":"gemma4","hidden_size":128}`),
	} {
		if _, err := ResolveGemma4ArchitectureDeclaration(config); err == nil {
			t.Fatalf("ResolveGemma4ArchitectureDeclaration(%s) returned nil error", config)
		}
	}
}

func TestResolveGemma4ArchitectureDeclaration_Ugly(t *testing.T) {
	if _, err := ResolveGemma4ArchitectureDeclaration([]byte(`{"model_type":"gemma4"`)); err == nil {
		t.Fatal("malformed JSON should return an explicit error")
	}
}

func TestResolveGemma4GGUFArchitectureDeclaration_Good(t *testing.T) {
	declaration, err := ResolveGemma4GGUFArchitectureDeclaration(gguf.Metadata{
		Architecture:                  "gemma4",
		BlockCount:                    productionLaneGemma4E2BLayers,
		EmbeddingLength:               productionLaneGemma4E2BHiddenSize,
		FeedForwardLength:             12288,
		ExpertCount:                   128,
		ExpertUsedCount:               8,
		ExpertFeedForwardLength:       704,
		AttentionHeadCount:            8,
		AttentionHeadCountKV:          1,
		AttentionKeyLength:            512,
		AttentionValueLength:          512,
		AttentionKeyLengthSWA:         256,
		AttentionValueLengthSWA:       256,
		AttentionSlidingWindow:        512,
		AttentionSharedKVLayers:       20,
		AttentionSharedKVLayersSet:    true,
		EmbeddingLengthPerLayerInput:  256,
		AttentionSlidingWindowPattern: true,
		RopeFreqBase:                  1_000_000,
		RopeFreqBaseSWA:               10_000,
		RopeDimensionCount:            128,
		RopeDimensionCountSWA:         256,
		FinalLogitSoftcap:             30,
	}, inference.ModelInfo{
		Architecture: "gemma4",
		NumLayers:    productionLaneGemma4E2BLayers,
		HiddenSize:   productionLaneGemma4E2BHiddenSize,
		VocabSize:    productionLaneGemma4E2BVocabSize,
	})
	if err != nil {
		t.Fatalf("ResolveGemma4GGUFArchitectureDeclaration: %v", err)
	}
	if !declaration.Matched() {
		t.Fatalf("GGUF declaration = %+v, want a matched shared architecture", declaration)
	}
	arch := declaration.Arch
	if arch.HeadDim != 256 || arch.GlobalHeadDim != 512 || arch.Experts != 128 || arch.TopK != 8 || arch.ExpertFF != 704 || !arch.HasMoE() {
		t.Fatalf("GGUF shared Arch = %+v, want Gemma4 attention and MoE geometry", arch)
	}
	if arch.RotaryDim != 128 || arch.RotaryDimLocal != 256 {
		t.Fatalf("GGUF rotary dims = (%d,%d), want (128,256)", arch.RotaryDim, arch.RotaryDimLocal)
	}
	if declaration.Topology.KVShareFrom[15] != 13 || declaration.Topology.KVShareFrom[19] != 14 || declaration.Topology.KVShareFrom[34] != 14 {
		t.Fatalf("GGUF shared topology = %+v, want E2B shared-cache owners", declaration.Topology.KVShareFrom)
	}
}

func TestResolveGemma4GGUFArchitectureDeclarationWeightDerivedKVHeads_Good(t *testing.T) {
	metadata := gguf.Metadata{
		Architecture:                  "gemma4",
		BlockCount:                    6,
		EmbeddingLength:               2816,
		FeedForwardLength:             2112,
		AttentionHeadCount:            16,
		AttentionKeyLength:            512,
		AttentionValueLength:          512,
		AttentionKeyLengthSWA:         256,
		AttentionValueLengthSWA:       256,
		AttentionSlidingWindow:        1024,
		AttentionSlidingWindowPattern: true,
	}
	tensors := []nativeTensorInfo{
		{Name: "blk.0.attn_k.weight", Dimensions: []uint64{2816, 2048}},
		{Name: "blk.5.attn_k.weight", Dimensions: []uint64{2816, 1024}},
	}
	declaration, err := resolveGemma4GGUFArchitectureDeclarationWithTensors(metadata, inference.ModelInfo{
		Architecture: "gemma4",
		NumLayers:    6,
		HiddenSize:   2816,
		VocabSize:    262144,
	}, tensors)
	if err != nil {
		t.Fatalf("resolveGemma4GGUFArchitectureDeclarationWithTensors: %v", err)
	}
	if declaration.Arch.KVHeads != 8 || declaration.Arch.GlobalKVHeads != 2 {
		t.Fatalf("shared KV heads = (%d,%d), want sliding/global (8,2)", declaration.Arch.KVHeads, declaration.Arch.GlobalKVHeads)
	}
	if declaration.Arch.Layer[0].KVHeads != 8 || declaration.Arch.Layer[5].KVHeads != 2 {
		t.Fatalf("shared per-layer KV heads = (%d,%d), want (8,2)", declaration.Arch.Layer[0].KVHeads, declaration.Arch.Layer[5].KVHeads)
	}
}

func TestHIPLoadedGemma4LayerSpecsPreferSharedDeclaration_Good(t *testing.T) {
	layers := model.DeriveLayers([]string{"sliding_attention", "full_attention", "sliding_attention"}, 1)
	for index := range layers {
		layers[index].HeadDim = 256
		layers[index].KVHeads = 2
	}
	loaded := &hipLoadedModel{
		gemma4Architecture: Gemma4ArchitectureDeclaration{
			Contract: Gemma4ArchitectureDeclarationContract,
			Resolution: ROCmArchitectureResolution{
				Architecture: "gemma4_text",
				Profile:      ROCmArchitectureProfile{ID: "gemma4_text"},
			},
			Arch: model.Arch{Layer: layers},
			Topology: Gemma4ArchitectureTopology{
				LayerTypes:  []string{"sliding_attention", "full_attention", "sliding_attention"},
				KVShareFrom: []int{0, 1, 0},
				CacheIndex:  []int{0, 1, -1},
			},
		},
		gemma4TextConfig: nativeGemma4TextConfig{
			LayerTypes:        []string{"full_attention", "full_attention", "full_attention"},
			KVSharedLayers:    0,
			KVSharedLayersSet: true,
		},
	}

	got, ok := loaded.sharedGemma4LayerSpecs(3)
	if !ok {
		t.Fatal("sharedGemma4LayerSpecs declined a matched shared declaration")
	}
	for index := range layers {
		if got[index] != layers[index] {
			t.Fatalf("layer %d = %+v, want shared declaration %+v", index, got[index], layers[index])
		}
	}
}
