// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestHIPRuntime_ApplyChatTemplateUsesQwenChatML_Good(t *testing.T) {
	model := &hipLoadedModel{modelInfo: inference.ModelInfo{Architecture: "qwen3_6"}}
	messages := []inference.Message{
		{Role: "system", Content: "Be terse."},
		{Role: "user", Content: "Hi"},
	}

	prompt, err := model.ApplyChatTemplate(messages)
	core.RequireNoError(t, err)
	core.AssertEqual(t,
		"<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
		prompt,
	)
}

func TestHIPRuntime_ApplyChatTemplateHonorsQwenThinking_Bad(t *testing.T) {
	model := &hipLoadedModel{modelInfo: inference.ModelInfo{Architecture: "qwen3_6"}}
	enabled := true
	prompt, err := model.applyChatTemplateWithGenerateConfig(
		[]inference.Message{{Role: "user", Content: "Hi"}},
		inference.GenerateConfig{EnableThinking: &enabled},
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", prompt)
}

func TestHIPRuntime_ApplyChatTemplateKeepsUnknownFallback_Ugly(t *testing.T) {
	model := &hipLoadedModel{modelInfo: inference.ModelInfo{Architecture: "unknown"}}
	prompt, err := model.ApplyChatTemplate([]inference.Message{{Role: "user", Content: "Hi"}})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "user: Hi\n", prompt)
}

func TestHIPRuntime_LoadModelAllocatesAndCopiesGGUFTensors_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	driver := &fakeHIPDriver{
		available: true,
		device:    nativeDeviceInfo{Name: "gfx1100", MemoryBytes: 16 * memoryGiB, FreeBytes: 12 * memoryGiB, Driver: "fake"},
	}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3", NumLayers: 1, QuantBits: 32},
		DataOffset: dataOffset,
		Tensors: []nativeTensorInfo{{
			Name:     "tok_embeddings.weight",
			Type:     0,
			Offset:   0,
			ByteSize: 16,
		}, {
			Name:     "output.weight",
			Type:     0,
			Offset:   16,
			ByteSize: 16,
		}},
	})

	core.AssertNoError(t, err)
	core.AssertNotNil(t, model)
	core.AssertEqual(t, []uint64{16, 16}, driver.allocations)
	core.AssertEqual(t, []uint64{16, 16}, driver.copies)
	core.AssertEqual(t, 2, driver.pinnedCopies)
	stream, errFn := model.Generate(context.Background(), "hello", inference.DefaultGenerateConfig())
	for range stream {
	}
	core.AssertError(t, errFn())
	core.AssertNoError(t, model.Close())
	core.AssertEqual(t, 2, len(driver.frees))
}

func TestHIPGemma4HostResidentExpertTensor_Good(t *testing.T) {
	moe := nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "gemma4"},
		Gemma4TextConfig: nativeGemma4TextConfig{
			EnableMoEBlock: true,
		},
	}
	expert := nativeTensorInfo{Name: "blk.0.ffn_gate_up_exps.weight", Dimensions: []uint64{2816, 1408, 128}}
	core.AssertEqual(t, true, hipGemma4HostResidentExpertTensor(moe, expert))
	moe.Gemma4TextConfig.EnableMoEBlock = false
	core.AssertEqual(t, true, hipGemma4HostResidentExpertTensor(moe, expert))

	for _, architecture := range []string{"gemma4", "diffusion_gemma"} {
		moe.ModelInfo.Architecture = architecture
		for _, name := range []string{
			"language_model.model.layers.0.experts.gate_up_proj.weight",
			"language_model.model.layers.0.experts.gate_up_proj.scales",
			"language_model.model.layers.0.experts.gate_up_proj.biases",
			"language_model.model.layers.0.experts.down_proj.weight",
			"language_model.model.layers.0.experts.down_proj.scales",
			"language_model.model.layers.0.experts.down_proj.biases",
		} {
			core.AssertEqual(t, true, hipGemma4HostResidentExpertTensor(moe, nativeTensorInfo{
				Name:       name,
				Dimensions: []uint64{2816, 1408, 128},
			}))
		}
	}
}

func TestHIPGemma4HostResidentExpertTensor_Bad(t *testing.T) {
	moe := nativeLoadConfig{ModelInfo: inference.ModelInfo{Architecture: "gemma4"}}
	for _, name := range []string{
		"language_model.model.layers.0.experts.gate_up_proj.scale",
		"language_model.model.layers.0.experts.gate_up_proj.weight.extra",
		"language_model.model.layers.0.experts.gate_up_proj.weightish",
		"language_model.model.layers.0.mlp.gate_up_proj.weight",
		"language_model.model.layers.0.experts.down_proj.bias",
	} {
		core.AssertEqual(t, false, hipGemma4HostResidentExpertTensor(moe, nativeTensorInfo{
			Name:       name,
			Dimensions: []uint64{2816, 1408, 128},
		}))
	}
	moe.ModelInfo.Architecture = "qwen3_moe"
	core.AssertEqual(t, false, hipGemma4HostResidentExpertTensor(moe, nativeTensorInfo{
		Name:       "language_model.model.layers.0.experts.gate_up_proj.weight",
		Dimensions: []uint64{2816, 1408, 128},
	}))
	core.AssertEqual(t, false, hipGemma4HostResidentExpertTensor(moe, nativeTensorInfo{
		Name:       "blk.0.ffn_down_exps.scale",
		Dimensions: []uint64{128},
	}))
}

func TestHIPGemma4HostResidentExpertTensor_Ugly(t *testing.T) {
	moe := nativeLoadConfig{ModelInfo: inference.ModelInfo{Architecture: "diffusion_gemma"}}
	for _, dimensions := range [][]uint64{
		nil,
		{2816, 1408},
		{2816, 1408, 128, 1},
	} {
		core.AssertEqual(t, false, hipGemma4HostResidentExpertTensor(moe, nativeTensorInfo{
			Name:       "language_model.model.layers.0.experts.gate_up_proj.weight",
			Dimensions: dimensions,
		}))
	}
	core.AssertEqual(t, false, hipGemma4HostResidentExpertTensor(moe, nativeTensorInfo{
		Name:       "",
		Dimensions: []uint64{2816, 1408, 128},
	}))
	core.AssertEqual(t, false, hipGemma4HostResidentExpertTensor(moe, nativeTensorInfo{
		Name:       "blk.0.ffn_gate.weight",
		Dimensions: []uint64{2816, 2112},
	}))
}

func TestHIPRuntime_LoadModelCarriesDeviceKVMode_Good(t *testing.T) {
	driver := &fakeHIPDriver{
		available: true,
		device:    nativeDeviceInfo{Name: "gfx1100", MemoryBytes: 16 * memoryGiB, FreeBytes: 12 * memoryGiB, Driver: "fake"},
	}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)
	cfg := validHIPDriverFakeLoadConfigWithOffset(dataOffset)
	cfg.DeviceKVMode = rocmKVCacheModeQ8

	model, err := runtime.LoadModel(path, cfg)
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	if loaded.gemma4Q4EngineConfig().DeviceKVMode != rocmKVCacheModeQ8 {
		t.Fatalf("loaded Gemma4 q4 engine config = %+v, want q8 device KV mode", loaded.gemma4Q4EngineConfig())
	}
}

func TestHIPRuntime_Gemma4EngineConfigCarriesBidirectionalSpans_Good(t *testing.T) {
	loaded := &hipLoadedModel{gemma4Q4Config: hipGemma4Q4EngineConfig{
		DeviceKVMode:            rocmKVCacheModeQ8,
		BidirectionalSpanTokens: [2]int32{22, 23},
	}}
	cfg := loaded.gemma4Q4EngineConfig()
	core.AssertEqual(t, rocmKVCacheModeQ8, cfg.DeviceKVMode)
	core.AssertEqual(t, [2]int32{22, 23}, cfg.BidirectionalSpanTokens)
}

func TestHIPRuntime_LoadModelClonesSequenceMixerPlan_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)
	plan := &SequenceMixerLoadPlan{
		Contract: SequenceMixerRegistryContract,
		Runtime:  SequenceMixerRuntimePlannedHIP,
		Layers: []SequenceMixerLayerPlan{{
			Layer:   0,
			Kind:    "full_attention",
			State:   SequenceMixerStateKVCache,
			Source:  "generic_softmax",
			Runtime: SequenceMixerRuntimePlannedHIP,
			Subpath: "self_attn",
		}},
		Subpaths: SequenceMixerSubpathPlan{
			LayerCount: 1,
			Subpaths:   map[int]string{0: "self_attn"},
		},
		Cache: SequenceMixerCachePlan{
			Contract: SequenceMixerCachePlanContract,
			Layers: []SequenceMixerCacheLayerPlan{{
				Layer:  0,
				Kind:   "full_attention",
				State:  SequenceMixerStateKVCache,
				Holder: SequenceMixerStateKVCache,
			}},
		},
	}

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:         inference.ModelInfo{Architecture: "qwen3", NumLayers: 1, QuantBits: 32},
		ModelLabels:       map[string]string{"sequence_mixer_load_plan_status": "valid"},
		SequenceMixerPlan: plan,
		DataOffset:        dataOffset,
		Tensors: []nativeTensorInfo{{
			Name:     "tok_embeddings.weight",
			Type:     0,
			Offset:   0,
			ByteSize: 4,
		}, {
			Name:     "output.weight",
			Type:     0,
			Offset:   4,
			ByteSize: 4,
		}, {
			Name:     "model.layers.0.self_attn.q_proj.weight",
			Type:     0,
			Offset:   8,
			ByteSize: 4,
		}, {
			Name:     "model.layers.0.self_attn.k_proj.weight",
			Type:     0,
			Offset:   12,
			ByteSize: 4,
		}, {
			Name:     "model.layers.0.self_attn.v_proj.weight",
			Type:     0,
			Offset:   16,
			ByteSize: 4,
		}, {
			Name:     "model.layers.0.self_attn.o_proj.weight",
			Type:     0,
			Offset:   20,
			ByteSize: 4,
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()

	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	if loaded.sequenceMixerPlan == nil || len(loaded.sequenceMixerPlan.Layers) != 1 {
		t.Fatalf("loaded sequence mixer plan = %+v, want cloned plan", loaded.sequenceMixerPlan)
	}
	plan.Layers[0].Kind = "mutated"
	plan.Subpaths.Subpaths[0] = "mutated"
	plan.Cache.Layers[0].Holder = "mutated"
	if loaded.sequenceMixerPlan.Layers[0].Kind != "full_attention" ||
		loaded.sequenceMixerPlan.Subpaths.Subpaths[0] != "self_attn" ||
		loaded.sequenceMixerPlan.Cache.Layers[0].Holder != SequenceMixerStateKVCache {
		t.Fatalf("loaded sequence mixer plan mutated with input: %+v", loaded.sequenceMixerPlan)
	}
	if loaded.sequenceMixerBindings == nil || len(loaded.sequenceMixerBindings.Layers) != 1 {
		t.Fatalf("loaded sequence mixer bindings = %+v, want one bound layer", loaded.sequenceMixerBindings)
	}
	if loaded.sequenceMixerBindings.Cache.Contract != SequenceMixerCachePlanContract ||
		len(loaded.sequenceMixerBindings.Cache.Layers) != 1 ||
		loaded.sequenceMixerBindings.Cache.Layers[0].Holder != SequenceMixerStateKVCache ||
		loaded.sequenceMixerBindings.Cache.Layers[0].Mode != SequenceMixerCacheModeDefault {
		t.Fatalf("loaded sequence mixer binding cache = %+v, want bound default kv-cache holder", loaded.sequenceMixerBindings.Cache)
	}
	binding := loaded.sequenceMixerBindings.Layers[0]
	if binding.Plan.Kind != "full_attention" || binding.Plan.Subpath != "self_attn" {
		t.Fatalf("loaded sequence mixer binding plan = %+v, want full_attention self_attn", binding.Plan)
	}
	if got := binding.Tensors["q_proj.weight"].info.Name; got != "model.layers.0.self_attn.q_proj.weight" {
		t.Fatalf("bound q_proj tensor = %q, want self_attn q_proj", got)
	}
	if len(binding.Tensors) != 4 {
		t.Fatalf("bound full_attention tensor count = %d, want required q/k/v/o tensors", len(binding.Tensors))
	}
}

func TestHIPRuntime_LoadModelBindsSequenceMixerRecurrentSubpath_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)
	plan := &SequenceMixerLoadPlan{
		Contract: SequenceMixerRegistryContract,
		Runtime:  SequenceMixerRuntimePlannedHIP,
		Layers: []SequenceMixerLayerPlan{{
			Layer:   0,
			Kind:    "mamba2",
			State:   SequenceMixerStateRecurrent,
			Source:  "fla",
			Runtime: SequenceMixerRuntimePlannedHIP,
			Subpath: "mixer",
		}},
		Subpaths: SequenceMixerSubpathPlan{
			LayerCount: 1,
			Subpaths:   map[int]string{0: "mixer"},
		},
		Cache: SequenceMixerCachePlan{
			Contract: SequenceMixerCachePlanContract,
			Layers: []SequenceMixerCacheLayerPlan{{
				Layer:  0,
				Kind:   "mamba2",
				State:  SequenceMixerStateRecurrent,
				Holder: SequenceMixerStateRecurrent,
			}},
		},
	}

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:         inference.ModelInfo{Architecture: "qwen3", NumLayers: 1, QuantBits: 32},
		ModelLabels:       map[string]string{"sequence_mixer_load_plan_status": "valid"},
		SequenceMixerPlan: plan,
		DataOffset:        dataOffset,
		Tensors: []nativeTensorInfo{{
			Name:     "tok_embeddings.weight",
			Type:     0,
			Offset:   0,
			ByteSize: 4,
		}, {
			Name:     "output.weight",
			Type:     0,
			Offset:   4,
			ByteSize: 4,
		}, {
			Name:     "language_model.model.layers.0.mixer.in_proj.weight",
			Type:     0,
			Offset:   8,
			ByteSize: 4,
		}, {
			Name:     "language_model.model.layers.0.mixer.out_proj.weight",
			Type:     0,
			Offset:   12,
			ByteSize: 4,
		}, {
			Name:     "language_model.model.layers.0.mixer.conv1d.weight",
			Type:     0,
			Offset:   16,
			ByteSize: 4,
		}, {
			Name:     "language_model.model.layers.0.mixer.A_log",
			Type:     0,
			Offset:   20,
			ByteSize: 4,
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()

	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	if loaded.sequenceMixerBindings == nil || len(loaded.sequenceMixerBindings.Layers) != 1 {
		t.Fatalf("loaded sequence mixer bindings = %+v, want one bound layer", loaded.sequenceMixerBindings)
	}
	if loaded.sequenceMixerBindings.Cache.Contract != SequenceMixerCachePlanContract ||
		len(loaded.sequenceMixerBindings.Cache.Layers) != 1 ||
		loaded.sequenceMixerBindings.Cache.Layers[0].Holder != SequenceMixerStateRecurrent ||
		loaded.sequenceMixerBindings.Cache.Layers[0].Mode != SequenceMixerCacheModeRecurrent {
		t.Fatalf("loaded sequence mixer binding cache = %+v, want bound recurrent holder", loaded.sequenceMixerBindings.Cache)
	}
	binding := loaded.sequenceMixerBindings.Layers[0]
	if binding.Plan.Kind != "mamba2" || binding.Plan.State != "recurrent" || binding.Plan.Subpath != "mixer" {
		t.Fatalf("loaded sequence mixer binding plan = %+v, want recurrent mamba2 mixer", binding.Plan)
	}
	if got := binding.Tensors["in_proj.weight"].info.Name; got != "language_model.model.layers.0.mixer.in_proj.weight" {
		t.Fatalf("bound recurrent tensor = %q, want language_model alias in_proj", got)
	}
	for _, leaf := range []string{"in_proj.weight", "out_proj.weight", "conv1d.weight", "A_log"} {
		if _, ok := binding.Tensors[leaf]; !ok {
			t.Fatalf("bound recurrent tensors = %v, missing required %s", binding.Tensors, leaf)
		}
	}
}

func TestHIPRuntime_LoadModelBindsSequenceMixerMissingRecurrentTensor_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)
	plan := &SequenceMixerLoadPlan{
		Contract: SequenceMixerRegistryContract,
		Runtime:  SequenceMixerRuntimePlannedHIP,
		Layers: []SequenceMixerLayerPlan{{
			Layer:   0,
			Kind:    "mamba2",
			State:   SequenceMixerStateRecurrent,
			Source:  "fla",
			Runtime: SequenceMixerRuntimePlannedHIP,
			Subpath: "mixer",
		}},
		Subpaths: SequenceMixerSubpathPlan{
			LayerCount: 1,
			Subpaths:   map[int]string{0: "mixer"},
		},
		Cache: SequenceMixerCachePlan{
			Contract: SequenceMixerCachePlanContract,
			Layers: []SequenceMixerCacheLayerPlan{{
				Layer:  0,
				Kind:   "mamba2",
				State:  SequenceMixerStateRecurrent,
				Holder: SequenceMixerStateRecurrent,
			}},
		},
	}

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:         inference.ModelInfo{Architecture: "qwen3", NumLayers: 1, QuantBits: 32},
		ModelLabels:       map[string]string{"sequence_mixer_load_plan_status": "valid"},
		SequenceMixerPlan: plan,
		DataOffset:        dataOffset,
		Tensors: []nativeTensorInfo{{
			Name:     "tok_embeddings.weight",
			Type:     0,
			Offset:   0,
			ByteSize: 4,
		}, {
			Name:     "output.weight",
			Type:     0,
			Offset:   4,
			ByteSize: 4,
		}, {
			Name:     "model.layers.0.mixer.in_proj.weight",
			Type:     0,
			Offset:   8,
			ByteSize: 4,
		}, {
			Name:     "model.layers.0.mixer.out_proj.weight",
			Type:     0,
			Offset:   12,
			ByteSize: 4,
		}, {
			Name:     "model.layers.0.mixer.conv1d.weight",
			Type:     0,
			Offset:   16,
			ByteSize: 4,
		}},
	})

	core.AssertError(t, err)
	core.AssertNil(t, model)
	core.AssertContains(t, err.Error(), "bind sequence mixer plan")
	core.AssertContains(t, err.Error(), "mamba2 missing A_log tensor")
	core.AssertEqual(t, len(driver.allocations), len(driver.frees))
}

func TestHIPSequenceMixerRequiredLeavesCoverRegistry_Good(t *testing.T) {
	for _, family := range DefaultSequenceMixerFamilies() {
		leaves, ok := SequenceMixerRequiredLeaves(family.Kind)
		if !ok || len(leaves) == 0 {
			t.Fatalf("required leaves for %s = %v, %v; want non-empty required set", family.Kind, leaves, ok)
		}
	}
}

func TestHIPRuntime_LoadModelBindsSequenceMixerMissingAttentionTensor_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)
	plan := &SequenceMixerLoadPlan{
		Contract: SequenceMixerRegistryContract,
		Runtime:  SequenceMixerRuntimePlannedHIP,
		Layers: []SequenceMixerLayerPlan{{
			Layer:   0,
			Kind:    "full_attention",
			State:   SequenceMixerStateKVCache,
			Source:  "generic_softmax",
			Runtime: SequenceMixerRuntimePlannedHIP,
			Subpath: "self_attn",
		}},
		Subpaths: SequenceMixerSubpathPlan{
			LayerCount: 1,
			Subpaths:   map[int]string{0: "self_attn"},
		},
		Cache: SequenceMixerCachePlan{
			Contract: SequenceMixerCachePlanContract,
			Layers: []SequenceMixerCacheLayerPlan{{
				Layer:  0,
				Kind:   "full_attention",
				State:  SequenceMixerStateKVCache,
				Holder: SequenceMixerStateKVCache,
			}},
		},
	}

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:         inference.ModelInfo{Architecture: "qwen3", NumLayers: 1, QuantBits: 32},
		ModelLabels:       map[string]string{"sequence_mixer_load_plan_status": "valid"},
		SequenceMixerPlan: plan,
		DataOffset:        dataOffset,
		Tensors: []nativeTensorInfo{{
			Name:     "tok_embeddings.weight",
			Type:     0,
			Offset:   0,
			ByteSize: 4,
		}, {
			Name:     "output.weight",
			Type:     0,
			Offset:   4,
			ByteSize: 4,
		}, {
			Name:     "model.layers.0.self_attn.q_proj.weight",
			Type:     0,
			Offset:   8,
			ByteSize: 4,
		}, {
			Name:     "model.layers.0.self_attn.k_proj.weight",
			Type:     0,
			Offset:   12,
			ByteSize: 4,
		}, {
			Name:     "model.layers.0.self_attn.v_proj.weight",
			Type:     0,
			Offset:   16,
			ByteSize: 4,
		}},
	})

	core.AssertError(t, err)
	core.AssertNil(t, model)
	core.AssertContains(t, err.Error(), "bind sequence mixer plan")
	core.AssertContains(t, err.Error(), "full_attention missing o_proj.weight tensor")
	core.AssertEqual(t, len(driver.allocations), len(driver.frees))
}

func TestHIPRuntime_SequenceMixerCachePlanValidation_GoodAndBad(t *testing.T) {
	legacy, err := sequenceMixerCachePlanForLoadPlan(&SequenceMixerLoadPlan{
		Layers: []SequenceMixerLayerPlan{{
			Layer:   0,
			Kind:    "mamba2",
			State:   SequenceMixerStateRecurrent,
			Source:  "fla",
			Runtime: SequenceMixerRuntimePlannedHIP,
			Subpath: "mixer",
		}},
	})
	core.AssertNoError(t, err)
	core.AssertEqual(t, SequenceMixerCachePlanContract, legacy.Contract)
	core.AssertEqual(t, SequenceMixerStateRecurrent, legacy.Layers[0].Holder)
	core.AssertEqual(t, SequenceMixerCacheModeRecurrent, legacy.Layers[0].Mode)
	core.AssertEqual(t, []string{"conv_state", "ssm_state"}, legacy.Layers[0].StateSlots)

	_, err = sequenceMixerCachePlanForLoadPlan(&SequenceMixerLoadPlan{
		Layers: []SequenceMixerLayerPlan{{
			Layer:   0,
			Kind:    "full_attention",
			State:   SequenceMixerStateKVCache,
			Source:  "generic_softmax",
			Runtime: SequenceMixerRuntimePlannedHIP,
			Subpath: "self_attn",
		}},
		Cache: SequenceMixerCachePlan{
			Contract: SequenceMixerCachePlanContract,
			Layers: []SequenceMixerCacheLayerPlan{{
				Layer:  0,
				Kind:   "full_attention",
				State:  SequenceMixerStateKVCache,
				Holder: SequenceMixerStateRecurrent,
			}},
		},
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "cache plan mismatch")

	_, err = sequenceMixerCachePlanForLoadPlan(&SequenceMixerLoadPlan{
		Layers: []SequenceMixerLayerPlan{{
			Layer:   0,
			Kind:    "mla",
			State:   SequenceMixerStateKVCache,
			Source:  "fla",
			Runtime: SequenceMixerRuntimePlannedHIP,
			Subpath: "self_attn",
		}},
		Cache: SequenceMixerCachePlan{
			Contract: SequenceMixerCachePlanContract,
			Layers: []SequenceMixerCacheLayerPlan{{
				Layer:  0,
				Kind:   "mla",
				State:  SequenceMixerStateKVCache,
				Holder: SequenceMixerStateKVCache,
				Mode:   SequenceMixerCacheModeDefault,
			}},
		},
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "cache plan mismatch")
}

func TestHIPRuntime_CopyTensorToDeviceReusesReadBuffer_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	path := core.PathJoin(t.TempDir(), "weights.bin")
	payload := []byte("0123456789abcdef0123456789abcdef")
	result := core.WriteFile(path, payload, 0o644)
	core.RequireTrue(t, result.OK)

	first := hipTensor{
		info: nativeTensorInfo{
			Name:     "first.weight",
			Offset:   0,
			ByteSize: 16,
		},
		pointer: nativeDevicePointer(0x1000),
	}
	second := hipTensor{
		info: nativeTensorInfo{
			Name:     "second.weight",
			Offset:   16,
			ByteSize: 8,
		},
		pointer: nativeDevicePointer(0x2000),
	}

	fileCache := map[string]*core.OSFile{}
	defer closeTensorSourceFiles(fileCache)

	buffer, err := copyTensorToDevice(driver, path, 0, first, nil, fileCache)
	core.RequireNoError(t, err)
	firstCap := cap(buffer)
	buffer, err = copyTensorToDevice(driver, path, 0, second, buffer, fileCache)
	core.RequireNoError(t, err)

	core.AssertEqual(t, firstCap, cap(buffer))
	core.AssertEqual(t, 1, len(fileCache))
	core.AssertEqual(t, []uint64{16, 8}, driver.copies)
	core.AssertEqual(t, 2, driver.pinnedCopies)
}

func BenchmarkHIPRuntimeCopyTensorToDevice_ReusedBuffer(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	path := core.PathJoin(b.TempDir(), "weights.bin")
	payload := make([]byte, nativeTensorCopyChunkBytes+4096)
	for index := range payload {
		payload[index] = byte(index)
	}
	result := core.WriteFile(path, payload, 0o644)
	core.RequireTrue(b, result.OK)
	tensor := hipTensor{
		info: nativeTensorInfo{
			Name:     "bench.weight",
			Offset:   0,
			ByteSize: uint64(len(payload)),
		},
		pointer: nativeDevicePointer(0x1000),
	}
	fileCache := map[string]*core.OSFile{}
	defer closeTensorSourceFiles(fileCache)
	buffer, err := copyTensorToDevice(driver, path, 0, tensor, nil, fileCache)
	core.RequireNoError(b, err)
	driver.copies = make([]uint64, 0, b.N*2)
	driver.pinnedCopies = 0

	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buffer, err = copyTensorToDevice(driver, path, 0, tensor, buffer, fileCache)
		if err != nil {
			b.Fatalf("copy tensor: %v", err)
		}
	}
}

func TestHIPRuntime_LoadModelLinksProjectionKernelWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-projection.hsaco")
	driver := &fakeHIPDriver{
		available: true,
		device:    nativeDeviceInfo{Name: "gfx1100", MemoryBytes: 16 * memoryGiB, FreeBytes: 12 * memoryGiB, Driver: "fake"},
	}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3", NumLayers: 1, QuantBits: 32},
		DataOffset: dataOffset,
		Tensors: []nativeTensorInfo{{
			Name:     "tok_embeddings.weight",
			Type:     0,
			Offset:   0,
			ByteSize: 16,
		}, {
			Name:     "output.weight",
			Type:     0,
			Offset:   16,
			ByteSize: 16,
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	status := loaded.KernelStatus()
	core.AssertEqual(t, hipKernelStatusNotLinked, status.Decode)
	core.AssertEqual(t, hipKernelStatusNotLinked, status.Prefill)
	core.AssertEqual(t, hipKernelStatusLinked, status.Projection)
	core.AssertEqual(t, hipKernelStatusLinked, status.CrossEntropy)
	core.AssertEqual(t, hipKernelStatusLinked, status.Distillation)
	core.AssertEqual(t, hipKernelStatusLinked, status.GRPO)

	projected, err := loaded.Project(context.Background(), hipProjectionRequest{
		Input: []float32{1, 2},
		FP16:  []uint16{0x3c00, 0x4000},
		Bias:  []float32{0.5},
		Rows:  1,
		Cols:  2,
	})
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{5.5}, projected, 0)
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameProjection, driver.launches[0].Name)

	stream, streamErr := loaded.Generate(context.Background(), "hello", inference.DefaultGenerateConfig())
	for range stream {
	}
	core.AssertError(t, streamErr())
	core.AssertContains(t, streamErr().Error(), "native decode kernels are not linked yet")
}

func TestHIPRuntime_LoadModelRunsTinyPrefillDecodeWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-tiny.hsaco")
	fixture := hipReferenceTinyLMFixture()
	for _, tt := range []struct {
		name             string
		outputType       uint32
		outputTypeName   string
		outputPayload    []byte
		codebookPayload  []byte
		codebookValues   []float32
		outputEncoding   uint32
		outputScale      float32
		outputWeightByte uint32
		wantJANGTQ       bool
		wantCodebook     bool
	}{{
		name:             "f32-output",
		outputType:       0,
		outputEncoding:   hipTinyOutputWeightEncodingFP32,
		outputWeightByte: 24,
	}, {
		name:             "f16-output",
		outputType:       1,
		outputEncoding:   hipTinyOutputWeightEncodingFP16,
		outputWeightByte: 12,
	}, {
		name:             "q8-output",
		outputType:       24,
		outputTypeName:   "q8:0.5",
		outputPayload:    hipInt8Payload(hipTinyOutputWeightsQ8Fixture()),
		outputEncoding:   hipTinyOutputWeightEncodingQ8,
		outputScale:      0.5,
		outputWeightByte: 6,
	}, {
		name:             "jangtq-output",
		outputType:       999,
		outputTypeName:   "jangtq:bits=2:group=2:scale=1",
		outputPayload:    []byte{0x41, 0x05},
		outputEncoding:   hipTinyOutputWeightEncodingFP32,
		outputWeightByte: 24,
		wantJANGTQ:       true,
	}, {
		name:             "codebook-output",
		outputType:       1000,
		outputTypeName:   "codebook:vq:dim=1",
		outputPayload:    []byte{1, 0, 0, 1, 1, 1},
		codebookValues:   []float32{0, 1},
		outputEncoding:   hipTinyOutputWeightEncodingFP32,
		outputWeightByte: 24,
		wantCodebook:     true,
	}} {
		t.Run(tt.name, func(t *testing.T) {
			embeddingPayload, err := hipFloat32Payload(fixture.EmbeddingTable)
			core.RequireNoError(t, err)
			outputPayload := tt.outputPayload
			if len(outputPayload) == 0 {
				switch tt.outputType {
				case 0:
					outputPayload, err = hipFloat32Payload(fixture.OutputWeights)
				case 1:
					outputPayload, err = hipUint16Payload(hipTinyOutputWeightsFP16Fixture())
				case 24:
					outputPayload = hipInt8Payload(hipTinyOutputWeightsQ8Fixture())
				default:
					t.Fatalf("unsupported output type %d", tt.outputType)
				}
				core.RequireNoError(t, err)
			}
			codebookPayload := tt.codebookPayload
			if len(codebookPayload) == 0 && len(tt.codebookValues) > 0 {
				codebookPayload, err = hipFloat32Payload(tt.codebookValues)
				core.RequireNoError(t, err)
			}
			modelPath := core.PathJoin(t.TempDir(), "tiny.bin")
			payload := append(append([]byte(nil), embeddingPayload...), outputPayload...)
			payload = append(payload, codebookPayload...)
			write := core.WriteFile(modelPath, payload, 0o644)
			core.RequireTrue(t, write.OK)
			driver := &fakeHIPDriver{available: true}
			runtime := newHIPRuntime(driver)
			tensors := []nativeTensorInfo{{
				Name:       "tok_embeddings.weight",
				Type:       0,
				Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
				Offset:     0,
				ByteSize:   uint64(len(embeddingPayload)),
			}, {
				Name:       "output.weight",
				Type:       tt.outputType,
				TypeName:   tt.outputTypeName,
				Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
				Offset:     uint64(len(embeddingPayload)),
				ByteSize:   uint64(len(outputPayload)),
			}}
			if len(codebookPayload) > 0 {
				tensors = append(tensors, nativeTensorInfo{
					Name:       "output.codebook",
					Type:       0,
					Dimensions: []uint64{uint64(len(tt.codebookValues)), 1},
					Offset:     uint64(len(embeddingPayload) + len(outputPayload)),
					ByteSize:   uint64(len(codebookPayload)),
				})
			}
			model, err := runtime.LoadModel(modelPath, nativeLoadConfig{
				ModelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: fixture.VocabSize, HiddenSize: fixture.HiddenSize, QuantBits: 32},
				Tensors:   tensors,
			})
			core.RequireNoError(t, err)
			defer model.Close()
			loaded, ok := model.(*hipLoadedModel)
			core.RequireTrue(t, ok)
			status := loaded.KernelStatus()
			core.AssertEqual(t, hipKernelStatusLinked, status.Prefill)
			core.AssertEqual(t, hipKernelStatusLinked, status.Decode)
			core.AssertEqual(t, hipKernelStatusLinked, status.Projection)

			prefill, err := loaded.Prefill(context.Background(), hipPrefillRequest{TokenIDs: []int32{0, 1}})
			core.RequireNoError(t, err)
			core.AssertEqual(t, 2, prefill.PromptTokens)
			core.AssertEqual(t, hipKernelNameTinyPrefill, prefill.Labels["prefill_kernel_name"])
			if tt.wantJANGTQ {
				core.AssertEqual(t, hipKernelNameJANGTQ, prefill.Labels["output_projection_kernel_name"])
				core.AssertEqual(t, "2", prefill.Labels["output_jangtq_bits"])
				core.AssertEqual(t, "2", prefill.Labels["output_jangtq_group_size"])
			}
			if tt.wantCodebook {
				core.AssertEqual(t, hipKernelNameCodebook, prefill.Labels["output_lookup_kernel_name"])
				core.AssertEqual(t, hipKernelNameProjection, prefill.Labels["output_projection_kernel_name"])
				core.AssertEqual(t, "2", prefill.Labels["output_codebook_entries"])
				core.AssertEqual(t, "1", prefill.Labels["output_codebook_dim"])
			}
			assertFloat32SlicesNear(t, []float32{0.3302, 0.6698, 1}, prefill.Logits, 0.0001)
			keys, values, err := prefill.KV.Restore(0, 2)
			core.RequireNoError(t, err)
			assertFloat32SlicesNear(t, []float32{1, 0, 0, 1}, keys, 0.0001)
			assertFloat32SlicesNear(t, []float32{1, 0, 0, 1}, values, 0.0001)
			core.AssertNotNil(t, prefill.DeviceKV)
			core.AssertNotNil(t, prefill.DescriptorTable)

			decoded, err := loaded.DecodeToken(context.Background(), hipDecodeRequest{
				TokenID:         2,
				KV:              prefill.KV,
				DeviceKV:        prefill.DeviceKV,
				DescriptorTable: prefill.DescriptorTable,
			})
			core.RequireNoError(t, err)
			defer decoded.DeviceKV.Close()
			defer decoded.DescriptorTable.Close()
			core.AssertEqual(t, int32(2), decoded.Token.ID)
			core.AssertEqual(t, hipKernelNameTinyDecode, decoded.Labels["decode_kernel_name"])
			if tt.wantJANGTQ {
				core.AssertEqual(t, hipKernelNameJANGTQ, decoded.Labels["output_projection_kernel_name"])
			}
			if tt.wantCodebook {
				core.AssertEqual(t, hipKernelNameCodebook, decoded.Labels["output_lookup_kernel_name"])
				core.AssertEqual(t, hipKernelNameProjection, decoded.Labels["output_projection_kernel_name"])
				core.AssertEqual(t, "2", decoded.Labels["output_codebook_entries"])
				core.AssertEqual(t, "1", decoded.Labels["output_codebook_dim"])
			}
			assertFloat32SlicesNear(t, []float32{0.7517, 0.7517, 1.5035}, decoded.Logits, 0.0001)
			core.AssertEqual(t, 3, decoded.KV.TokenCount())
			core.AssertEqual(t, "append_token", decoded.Labels["kv_device_update"])
			core.AssertEqual(t, "1", decoded.Labels["kv_device_update_pages"])
			core.AssertEqual(t, "1", decoded.Labels["kv_device_update_from_pages"])
			core.AssertEqual(t, "2", decoded.Labels["kv_device_update_from_tokens"])
			core.AssertEqual(t, "2", decoded.Labels["kv_device_update_to_pages"])
			core.AssertEqual(t, "3", decoded.Labels["kv_device_update_to_tokens"])
			core.AssertEqual(t, "success", decoded.Labels["kv_device_update_descriptor_refresh"])
			if !prefill.DeviceKV.closed || !prefill.DescriptorTable.closed {
				t.Fatalf("prefill device resources should be closed after successful tiny decode")
			}

			stream, streamErr := loaded.Generate(context.Background(), "hello", inference.GenerateConfig{MaxTokens: 2})
			var generated []int32
			for token := range stream {
				generated = append(generated, token.ID)
			}
			core.RequireNoError(t, streamErr())
			core.AssertEqual(t, []int32{1, 1}, generated)

			classified, err := loaded.Classify(context.Background(), []string{"hello"}, inference.GenerateConfig{ReturnLogits: true})
			core.RequireNoError(t, err)
			core.AssertEqual(t, int32(1), classified[0].Token.ID)
			assertFloat32SlicesNear(t, []float32{0, 1, 1}, classified[0].Logits, 0.0001)
			classifiedNoLogits, err := loaded.Classify(context.Background(), []string{"hello"}, inference.DefaultGenerateConfig())
			core.RequireNoError(t, err)
			core.AssertEqual(t, 0, len(classifiedNoLogits[0].Logits))

			launchNames := make([]string, len(driver.launches))
			for index, launch := range driver.launches {
				launchNames[index] = launch.Name
			}
			core.AssertContains(t, core.Join(",", launchNames...), hipKernelNameTinyPrefill)
			core.AssertContains(t, core.Join(",", launchNames...), hipKernelNameTinyDecode)
			if tt.wantJANGTQ {
				core.AssertContains(t, core.Join(",", launchNames...), hipKernelNameJANGTQ)
			}
			if tt.wantCodebook {
				core.AssertContains(t, core.Join(",", launchNames...), hipKernelNameCodebook)
				core.AssertContains(t, core.Join(",", launchNames...), hipKernelNameProjection)
			}
			var checkedPrefillLaunch bool
			for _, launch := range driver.launches {
				if launch.Name != hipKernelNameTinyPrefill || len(launch.Args) != hipTinyPrefillLaunchArgsBytes {
					continue
				}
				core.AssertEqual(t, tt.outputWeightByte, binary.LittleEndian.Uint32(launch.Args[92:]))
				core.AssertEqual(t, tt.outputEncoding, binary.LittleEndian.Uint32(launch.Args[116:]))
				core.AssertEqual(t, math.Float32bits(tt.outputScale), binary.LittleEndian.Uint32(launch.Args[120:]))
				checkedPrefillLaunch = true
				break
			}
			core.AssertTrue(t, checkedPrefillLaunch)
		})
	}
}

func TestHIPRuntime_LoadedTinyTextPathsPreflightRequests_Bad(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-tiny-preflight.hsaco")
	loaded, _ := loadHIPTinyF32FixtureModel(t, &fakeHIPDriver{available: true})

	stream, streamErr := loaded.Chat(context.Background(), nil, inference.DefaultGenerateConfig())
	for range stream {
		t.Fatal("Chat(nil) yielded token, want empty stream")
	}
	core.AssertError(t, streamErr())
	core.AssertContains(t, streamErr().Error(), "messages are required")

	stream, streamErr = loaded.Chat(context.Background(), []inference.Message{{Role: "moderator", Content: "hello"}}, inference.DefaultGenerateConfig())
	for range stream {
		t.Fatal("Chat(invalid role) yielded token, want empty stream")
	}
	core.AssertError(t, streamErr())
	core.AssertContains(t, streamErr().Error(), "message 0 role")

	_, err := loaded.Classify(context.Background(), nil, inference.DefaultGenerateConfig())
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompts are required")

	_, err = loaded.Classify(context.Background(), []string{"hello", ""}, inference.DefaultGenerateConfig())
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompt 1 is empty")

	_, err = loaded.BatchGenerate(context.Background(), nil, inference.DefaultGenerateConfig())
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompts are required")

	_, err = loaded.BatchGenerate(context.Background(), []string{"hello", " "}, inference.DefaultGenerateConfig())
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompt 1 is empty")

	results, err := loaded.BatchGenerate(context.Background(), []string{"hello"}, inference.GenerateConfig{MaxTokens: 1})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(results))
	core.AssertEqual(t, 1, len(results[0].Tokens))
}

func TestHIPRuntime_LoadedTinyRequestValidation_Bad(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-tiny-request-validation.hsaco")
	loaded, _ := loadHIPTinyF32FixtureModel(t, &fakeHIPDriver{available: true})

	_, err := loaded.Prefill(context.Background(), hipPrefillRequest{TokenIDs: []int32{99}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token ID is outside vocabulary")

	_, err = loaded.Prefill(context.Background(), hipPrefillRequest{TokenIDs: []int32{0}, KeyWidth: 1, ValueWidth: 1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "KV widths to match hidden size")

	cache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{1, 0, 0, 1}))

	_, err = loaded.DecodeToken(context.Background(), hipDecodeRequest{TokenID: 99, KV: cache})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token ID is outside vocabulary")
}

func BenchmarkHIPLoadedTinyDecodePriorKVRestoreInto_Reused(b *testing.B) {
	const (
		tokenCount = 512
		hiddenSize = 16
	)
	keys := make([]float32, tokenCount*hiddenSize)
	values := make([]float32, tokenCount*hiddenSize)
	for i := range keys {
		keys[i] = float32((i%17)-8) * 0.125
		values[i] = float32((i%19)-9) * 0.0625
	}
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, defaultROCmKVBlockSize)
	if err != nil {
		b.Fatalf("create KV cache: %v", err)
	}
	if err := cache.AppendVectors(0, hiddenSize, hiddenSize, keys, values); err != nil {
		b.Fatalf("append KV cache vectors: %v", err)
	}
	model := &hipLoadedModel{}
	req := hipDecodeRequest{TokenID: 1, KV: cache}
	if _, _, err := model.restoreLoadedTinyDecodePriorKV(req, hiddenSize); err != nil {
		b.Fatalf("warm restore prior KV: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gotKeys, gotValues, err := model.restoreLoadedTinyDecodePriorKV(req, hiddenSize)
		if err != nil {
			b.Fatalf("restore prior KV: %v", err)
		}
		if len(gotKeys) != len(keys) || len(gotValues) != len(values) {
			b.Fatalf("restored KV lengths = %d/%d, want %d/%d", len(gotKeys), len(gotValues), len(keys), len(values))
		}
	}
}

func TestHIPRuntime_LoadedTinyTextPathsPreferCancelledContext_Ugly(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-tiny-cancel.hsaco")
	loaded, _ := loadHIPTinyF32FixtureModel(t, &fakeHIPDriver{available: true})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	stream, streamErr := loaded.Generate(ctx, "hello", inference.GenerateConfig{MaxTokens: 0})
	for range stream {
		t.Fatal("Generate(cancelled) yielded token, want empty stream")
	}
	if !errors.Is(streamErr(), context.Canceled) {
		t.Fatalf("Generate error = %v, want context.Canceled", streamErr())
	}

	stream, streamErr = loaded.Chat(ctx, nil, inference.DefaultGenerateConfig())
	for range stream {
		t.Fatal("Chat(cancelled) yielded token, want empty stream")
	}
	if !errors.Is(streamErr(), context.Canceled) {
		t.Fatalf("Chat error = %v, want context.Canceled", streamErr())
	}

	_, err := loaded.Classify(ctx, nil, inference.DefaultGenerateConfig())
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Classify error = %v, want context.Canceled", err)
	}

	_, err = loaded.BatchGenerate(ctx, nil, inference.DefaultGenerateConfig())
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("BatchGenerate error = %v, want context.Canceled", err)
	}
}

func TestHIPRuntime_LoadedTinyLMConfigShapeValidation_Bad(t *testing.T) {
	baseModel := func() *hipLoadedModel {
		return &hipLoadedModel{
			driver:    &fakeHIPDriver{available: true},
			modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 3, HiddenSize: 2, QuantBits: 32},
			tensors: map[string]hipTensor{
				"tok_embeddings.weight": {
					info: nativeTensorInfo{
						Name:       "tok_embeddings.weight",
						Type:       0,
						Dimensions: []uint64{3, 2},
						ByteSize:   24,
					},
					pointer: 1,
				},
				"output.weight": {
					info: nativeTensorInfo{
						Name:       "output.weight",
						Type:       0,
						Dimensions: []uint64{3, 2},
						ByteSize:   24,
					},
					pointer: 2,
				},
			},
		}
	}

	for _, tt := range []struct {
		name   string
		mutate func(*hipLoadedModel)
		want   string
	}{{
		name: "embedding-rank",
		mutate: func(model *hipLoadedModel) {
			tensor := model.tensors["tok_embeddings.weight"]
			tensor.info.Dimensions = []uint64{6}
			model.tensors["tok_embeddings.weight"] = tensor
		},
		want: "embedding shape",
	}, {
		name: "output-dimension-mismatch",
		mutate: func(model *hipLoadedModel) {
			tensor := model.tensors["output.weight"]
			tensor.info.Dimensions = []uint64{3, 3}
			tensor.info.ByteSize = 36
			model.tensors["output.weight"] = tensor
		},
		want: "output shape",
	}, {
		name: "output-shape-mismatch",
		mutate: func(model *hipLoadedModel) {
			model.modelInfo = inference.ModelInfo{Architecture: "tiny", QuantBits: 32}
			tensor := model.tensors["output.weight"]
			tensor.info.Dimensions = []uint64{4, 2}
			tensor.info.ByteSize = 32
			model.tensors["output.weight"] = tensor
		},
		want: "embedding and output tensor shapes must match",
	}, {
		name: "embedding-byte-count",
		mutate: func(model *hipLoadedModel) {
			tensor := model.tensors["tok_embeddings.weight"]
			tensor.info.ByteSize = 20
			model.tensors["tok_embeddings.weight"] = tensor
		},
		want: "embedding byte count",
	}, {
		name: "output-byte-count",
		mutate: func(model *hipLoadedModel) {
			tensor := model.tensors["output.weight"]
			tensor.info.ByteSize = 20
			model.tensors["output.weight"] = tensor
		},
		want: "output byte count",
	}, {
		name: "zero-pointer",
		mutate: func(model *hipLoadedModel) {
			tensor := model.tensors["output.weight"]
			tensor.pointer = 0
			model.tensors["output.weight"] = tensor
		},
		want: "embedding and output tensor pointers",
	}} {
		t.Run(tt.name, func(t *testing.T) {
			model := baseModel()
			tt.mutate(model)
			_, err := model.loadedTinyLMConfig()
			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}
}

func TestHIPRuntime_LoadModelRunsTinyEmbedAndRerankWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-embedding.hsaco")
	fixture := hipReferenceTinyLMFixture()
	embeddingPayload, err := hipFloat32Payload(fixture.EmbeddingTable)
	core.RequireNoError(t, err)
	outputPayload, err := hipFloat32Payload(fixture.OutputWeights)
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "tiny-embedding.bin")
	payload := append(append([]byte(nil), embeddingPayload...), outputPayload...)
	write := core.WriteFile(modelPath, payload, 0o644)
	core.RequireTrue(t, write.OK)
	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: fixture.VocabSize, HiddenSize: fixture.HiddenSize, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "tok_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "output.weight",
			Type:       0,
			Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(outputPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)

	status := loaded.KernelStatus()
	core.AssertEqual(t, hipKernelStatusLinked, status.Embedding)
	core.AssertEqual(t, hipKernelStatusLinked, status.Rerank)
	embedded, err := loaded.Embed(context.Background(), inference.EmbeddingRequest{
		Input:     []string{"hello", "hello world"},
		Normalize: true,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, len(embedded.Vectors))
	core.AssertEqual(t, hipKernelNameEmbedMean, embedded.Labels["embedding_kernel_name"])
	assertFloat32SlicesNear(t, []float32{0, 1}, embedded.Vectors[0], 0.0001)
	assertFloat32SlicesNear(t, []float32{0.4472136, 0.8944272}, embedded.Vectors[1], 0.0001)
	core.AssertEqual(t, 3, embedded.Usage.PromptTokens)

	reranked, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(reranked.Results))
	core.AssertEqual(t, 1, reranked.Results[0].Index)
	core.AssertEqual(t, "hello", reranked.Results[0].Text)
	core.AssertEqual(t, hipKernelNameRerank, reranked.Labels["rerank_kernel_name"])

	var sawEmbedding, sawRerank bool
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameEmbedMean {
			sawEmbedding = true
		}
		if launch.Name == hipKernelNameRerank {
			sawRerank = true
		}
	}
	core.AssertTrue(t, sawEmbedding)
	core.AssertTrue(t, sawRerank)
}

func TestHIPRuntime_LoadModelRunsBERTEmbedAndRerankWithoutOutputHeadWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-bert-embedding.hsaco")
	embeddingTable := []float32{
		1, 0,
		0, 1,
		1, 1,
	}
	embeddingPayload, err := hipFloat32Payload(embeddingTable)
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "bert-embedding.bin")
	write := core.WriteFile(modelPath, embeddingPayload, 0o644)
	core.RequireTrue(t, write.OK)
	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 3, HiddenSize: 2, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "embeddings.word_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{3, 2},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)

	status := loaded.KernelStatus()
	core.AssertEqual(t, hipKernelStatusLinked, status.Embedding)
	core.AssertEqual(t, hipKernelStatusLinked, status.Rerank)
	core.AssertEqual(t, hipKernelStatusNotLinked, status.Prefill)
	core.AssertEqual(t, hipKernelStatusNotLinked, status.Decode)
	embedded, err := loaded.Embed(context.Background(), inference.EmbeddingRequest{
		Input:     []string{"hello", "hello world"},
		Normalize: true,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, len(embedded.Vectors))
	core.AssertEqual(t, "bert", embedded.Labels["embedding_model_family"])
	core.AssertEqual(t, "experimental_loaded_f32_table", embedded.Labels["embedding_model_status"])
	assertFloat32SlicesNear(t, []float32{0, 1}, embedded.Vectors[0], 0.0001)
	assertFloat32SlicesNear(t, []float32{0.4472136, 0.8944272}, embedded.Vectors[1], 0.0001)

	reranked, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(reranked.Results))
	core.AssertEqual(t, 1, reranked.Results[0].Index)
	core.AssertEqual(t, "experimental_embedding_cosine", reranked.Labels["rerank_model_status"])

	stream, streamErr := loaded.Generate(context.Background(), "hello", inference.GenerateConfig{MaxTokens: 1})
	for range stream {
	}
	core.AssertError(t, streamErr())
	core.AssertContains(t, streamErr().Error(), "native decode kernels are not linked yet")
}

func TestHIPRuntime_LoadModelRunsBERTSequenceClassifierRerankWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-bert-sequence-classifier.hsaco")
	embeddingTable := []float32{
		0, 0,
		0, 1,
		0, 0,
		0, 1,
		1, 0,
	}
	classifierWeights := []float32{
		0, 0,
		1, 0,
	}
	classifierBias := []float32{0, 0}
	embeddingPayload, err := hipFloat32Payload(embeddingTable)
	core.RequireNoError(t, err)
	classifierPayload, err := hipFloat32Payload(classifierWeights)
	core.RequireNoError(t, err)
	biasPayload, err := hipFloat32Payload(classifierBias)
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "bert-sequence-classifier.bin")
	payload := append(append(append([]byte(nil), embeddingPayload...), classifierPayload...), biasPayload...)
	write := core.WriteFile(modelPath, payload, 0o644)
	core.RequireTrue(t, write.OK)
	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 5, HiddenSize: 2, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "embeddings.word_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{5, 2},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "classifier.weight",
			Type:       0,
			Dimensions: []uint64{2, 2},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(classifierPayload)),
		}, {
			Name:       "classifier.bias",
			Type:       0,
			Dimensions: []uint64{2},
			Offset:     uint64(len(embeddingPayload) + len(classifierPayload)),
			ByteSize:   uint64(len(biasPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)

	wrapper := &rocmModel{
		modelType: "bert",
		modelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 5, HiddenSize: 2, QuantBits: 32},
		native:    loaded,
	}
	report := wrapper.Capabilities()
	classifyCapability, ok := report.Capability(inference.CapabilityClassify)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, inference.CapabilityStatusExperimental, classifyCapability.Status)
	core.AssertEqual(t, "bert_sequence_classifier", classifyCapability.Labels["classify_path"])
	evalCapability, ok := report.Capability(inference.CapabilityEvaluation)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, "bert_sequence_classifier", evalCapability.Labels["classify_path"])

	noTargetEval, err := wrapper.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{Text: "hello world"}}, inference.EvalConfig{MaxSamples: 1})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "not_requested", noTargetEval.Labels["loss_status"])
	core.AssertEqual(t, "not_requested", noTargetEval.Labels["perplexity_status"])
	core.AssertEqual(t, "bert_sequence_classifier", noTargetEval.Labels["classify_path"])
	core.AssertEqual(t, hipKernelStatusLinked, noTargetEval.Labels["loss_kernel"])
	core.AssertEqual(t, hipKernelNameCrossEntropy, noTargetEval.Labels["loss_kernel_name"])

	var evalEvents []inference.ProbeEvent
	wrapper.SetProbeSink(inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
		evalEvents = append(evalEvents, event)
	}))
	lossEval, err := wrapper.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{
		Prompt: "hello world again now",
		Labels: map[string]string{"target_token_id": "1"},
	}}, inference.EvalConfig{MaxSamples: 1})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "experimental", lossEval.Labels["loss_status"])
	core.AssertEqual(t, "experimental", lossEval.Labels["perplexity_status"])
	core.AssertEqual(t, "1", lossEval.Labels["eval.loss_tokens"])
	core.AssertEqual(t, "bert_sequence_classifier", lossEval.Labels["classify_path"])
	core.AssertEqual(t, "hip", lossEval.Labels["loss_backend"])
	core.AssertEqual(t, hipKernelStatusLinked, lossEval.Labels["loss_kernel"])
	core.AssertEqual(t, hipKernelNameCrossEntropy, lossEval.Labels["loss_kernel_name"])
	logitEvent, ok := nativeContractProbeEvent(evalEvents, inference.ProbeEventLogits)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, "classification", logitEvent.Labels["source"])
	core.AssertEqual(t, "0", logitEvent.Labels["classify_prompt_index"])
	entropyEvent, ok := nativeContractProbeEvent(evalEvents, inference.ProbeEventEntropy)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, "classification", entropyEvent.Labels["source"])

	classified, err := loaded.Classify(context.Background(), []string{"hello world again now"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(1), classified[0].Token.ID)
	core.AssertEqual(t, "label_1", classified[0].Token.Text)
	assertFloat32SlicesNear(t, []float32{0, 0.25}, classified[0].Logits, 0.0001)
	classifiedNoLogits, err := loaded.Classify(context.Background(), []string{"hello world again now"}, inference.DefaultGenerateConfig())
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, len(classifiedNoLogits[0].Logits))

	reranked, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(reranked.Results))
	core.AssertEqual(t, 0, reranked.Results[0].Index)
	core.AssertEqual(t, "hello world", reranked.Results[0].Text)
	core.AssertEqual(t, "experimental_bert_sequence_classifier", reranked.Labels["rerank_model_status"])
	core.AssertEqual(t, "classifier_positive_logit", reranked.Labels["rerank_score_source"])
	core.AssertEqual(t, hipKernelNameProjection, reranked.Labels["projection_kernel_name"])
	core.AssertEqual(t, "1", reranked.Results[0].Labels["rerank_classifier_index"])

	var sawEmbedding, sawProjection, sawCosine bool
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameEmbedMean {
			sawEmbedding = true
		}
		if launch.Name == hipKernelNameProjection {
			sawProjection = true
			core.AssertEqual(t, hipProjectionWeightEncodingF32, binary.LittleEndian.Uint32(launch.Args[80:]))
		}
		if launch.Name == hipKernelNameRerank {
			sawCosine = true
		}
	}
	core.AssertTrue(t, sawEmbedding)
	core.AssertTrue(t, sawProjection)
	core.AssertFalse(t, sawCosine)
}

func TestHIPRuntime_LoadModelRunsBERTSequenceClassifierRerankWithF16Head_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-bert-sequence-classifier-f16.hsaco")
	embeddingPayload, err := hipFloat32Payload([]float32{
		0, 0,
		0, 1,
		0, 0,
		0, 1,
		1, 0,
	})
	core.RequireNoError(t, err)
	classifierPayload, err := hipUint16Payload([]uint16{0, 0, 0x3c00, 0})
	core.RequireNoError(t, err)
	biasPayload, err := hipUint16Payload([]uint16{0, 0})
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "bert-sequence-classifier-f16.bin")
	write := core.WriteFile(modelPath, append(append(append([]byte(nil), embeddingPayload...), classifierPayload...), biasPayload...), 0o644)
	core.RequireTrue(t, write.OK)
	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 5, HiddenSize: 2, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "embeddings.word_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{5, 2},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "classifier.weight",
			Type:       1,
			Dimensions: []uint64{2, 2},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(classifierPayload)),
		}, {
			Name:       "classifier.bias",
			Type:       1,
			Dimensions: []uint64{2},
			Offset:     uint64(len(embeddingPayload) + len(classifierPayload)),
			ByteSize:   uint64(len(biasPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)

	classified, err := loaded.Classify(context.Background(), []string{"hello world again now"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(1), classified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 0.25}, classified[0].Logits, 0.0001)

	reranked, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, reranked.Results[0].Index)
	core.AssertEqual(t, "fp16", reranked.Labels["rerank_classifier_encoding"])
	core.AssertEqual(t, "fp16", reranked.Labels["rerank_classifier_bias_encoding"])
	var sawProjection bool
	for _, launch := range driver.launches {
		if launch.Name != hipKernelNameProjection {
			continue
		}
		sawProjection = true
		core.AssertEqual(t, hipProjectionWeightEncodingFP16, binary.LittleEndian.Uint32(launch.Args[80:]))
		core.AssertEqual(t, hipProjectionLaunchFlagBias, binary.LittleEndian.Uint32(launch.Args[84:]))
	}
	core.AssertTrue(t, sawProjection)
}

func TestHIPRuntime_LoadModelRunsBERTSequenceClassifierLoRAAdapterWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-bert-classifier-lora.hsaco")
	embeddingPayload, err := hipFloat32Payload([]float32{
		0, 0,
		0, 1,
		0, 0,
		0, 1,
		1, 0,
	})
	core.RequireNoError(t, err)
	classifierPayload, err := hipFloat32Payload([]float32{
		0, 0,
		0, 1,
	})
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "bert-sequence-classifier-lora.bin")
	write := core.WriteFile(modelPath, append(append([]byte(nil), embeddingPayload...), classifierPayload...), 0o644)
	core.RequireTrue(t, write.OK)
	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 5, HiddenSize: 2, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "embeddings.word_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{5, 2},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "classifier.weight",
			Type:       0,
			Dimensions: []uint64{2, 2},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(classifierPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, hipKernelStatusLinked, loaded.KernelStatus().LoRA)

	base, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, base.Results[0].Index)

	baseClassified, err := loaded.Classify(context.Background(), []string{"hello world again now"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(1), baseClassified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 0.5}, baseClassified[0].Logits, 0.0001)

	adapterPath := core.PathJoin(t.TempDir(), "classifier_lora.json")
	write = core.WriteFile(adapterPath, []byte(`{
		"format":"rocm-classifier-lora",
		"name":"bert-rerank-domain",
		"target":"classifier.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"num_labels":2,
		"lora_a":[1,0],
		"lora_b":[0,4]
	}`), 0o644)
	core.RequireTrue(t, write.OK)
	identity, err := loaded.LoadAdapter(adapterPath)
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmClassifierLoRAFormat, identity.Format)
	core.AssertEqual(t, "hip_bert_classifier", identity.Labels["adapter_runtime"])
	core.AssertEqual(t, adapterPath, loaded.ActiveAdapter().Path)

	classified, err := loaded.Classify(context.Background(), []string{"hello world again now"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(1), classified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 1.5}, classified[0].Logits, 0.0001)

	reranked, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, reranked.Results[0].Index)
	core.AssertEqual(t, hipKernelNameLoRA, reranked.Labels["projection_kernel_name"])
	core.AssertEqual(t, hipKernelNameLoRA, reranked.Labels["lora_kernel_name"])
	core.AssertEqual(t, "hip_bert_classifier", reranked.Labels["adapter_runtime"])

	var sawLoRA, sawCosine bool
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameLoRA {
			sawLoRA = true
		}
		if launch.Name == hipKernelNameRerank {
			sawCosine = true
		}
	}
	core.AssertTrue(t, sawLoRA)
	core.AssertFalse(t, sawCosine)
	core.AssertNoError(t, loaded.UnloadAdapter())
	if !adapterIdentityIsZero(loaded.ActiveAdapter()) {
		t.Fatalf("active adapter = %+v, want zero after unload", loaded.ActiveAdapter())
	}
}

func TestHIPRuntime_LoadModelRunsBERTScoreTensorLoRAAdapterWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-bert-score-lora.hsaco")
	embeddingPayload, err := hipFloat32Payload([]float32{
		0, 0,
		0, 1,
		0, 0,
		0, 1,
		1, 0,
	})
	core.RequireNoError(t, err)
	scorePayload, err := hipFloat32Payload([]float32{
		0, 0,
		0, 1,
	})
	core.RequireNoError(t, err)
	biasPayload, err := hipFloat32Payload([]float32{0, 0})
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "bert-score-lora.bin")
	write := core.WriteFile(modelPath, append(append(append([]byte(nil), embeddingPayload...), scorePayload...), biasPayload...), 0o644)
	core.RequireTrue(t, write.OK)
	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 5, HiddenSize: 2, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "embeddings.word_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{5, 2},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "score.weight",
			Type:       0,
			Dimensions: []uint64{2, 2},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(scorePayload)),
		}, {
			Name:       "score.bias",
			Type:       0,
			Dimensions: []uint64{2},
			Offset:     uint64(len(embeddingPayload) + len(scorePayload)),
			ByteSize:   uint64(len(biasPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)

	base, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, base.Results[0].Index)
	core.AssertEqual(t, "score.weight", base.Labels["rerank_classifier_tensor"])
	core.AssertEqual(t, "score.bias", base.Labels["rerank_classifier_bias"])

	baseClassified, err := loaded.Classify(context.Background(), []string{"hello world again now"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(1), baseClassified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 0.5}, baseClassified[0].Logits, 0.0001)

	adapterPath := core.PathJoin(t.TempDir(), "score_lora.json")
	write = core.WriteFile(adapterPath, []byte(`{
		"format":"rocm-classifier-lora",
		"name":"bert-score-domain",
		"target":"score.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"num_labels":2,
		"lora_a":[1,0],
		"lora_b":[0,4]
	}`), 0o644)
	core.RequireTrue(t, write.OK)
	identity, err := loaded.LoadAdapter(adapterPath)
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmClassifierLoRAFormat, identity.Format)
	core.AssertEqual(t, "score.weight", identity.TargetKeys[0])
	core.AssertEqual(t, "score.weight", identity.Labels["target"])
	core.AssertEqual(t, "score.weight", identity.Labels["classifier_tensor"])

	classified, err := loaded.Classify(context.Background(), []string{"hello world again now"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(1), classified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 1.5}, classified[0].Logits, 0.0001)

	reranked, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, reranked.Results[0].Index)
	core.AssertEqual(t, hipKernelNameLoRA, reranked.Labels["projection_kernel_name"])
	core.AssertEqual(t, hipKernelNameLoRA, reranked.Labels["lora_kernel_name"])
	core.AssertEqual(t, "score.weight", reranked.Labels["rerank_classifier_tensor"])
	core.AssertEqual(t, "score.bias", reranked.Labels["rerank_classifier_bias"])

	var sawLoRA bool
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameLoRA {
			sawLoRA = true
		}
	}
	core.AssertTrue(t, sawLoRA)
}

func TestHIPRuntime_LoadedSequenceClassifierConfigPairsCanonicalHeadBias_Good(t *testing.T) {
	model := &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "bert", HiddenSize: 2},
		tensors: map[string]hipTensor{
			"score.bias": {
				info:    nativeTensorInfo{Name: "score.bias", Type: 0, Dimensions: []uint64{2}, ByteSize: 8},
				pointer: 11,
			},
			"score.weight": {
				info:    nativeTensorInfo{Name: "score.weight", Type: 0, Dimensions: []uint64{2, 2}, ByteSize: 16},
				pointer: 12,
			},
			"classifier.bias": {
				info:    nativeTensorInfo{Name: "classifier.bias", Type: 0, Dimensions: []uint64{2}, ByteSize: 8},
				pointer: 13,
			},
			"classifier.weight": {
				info:    nativeTensorInfo{Name: "classifier.weight", Type: 0, Dimensions: []uint64{2, 2}, ByteSize: 16},
				pointer: 14,
			},
		},
	}

	cfg, hasClassifier, err := model.loadedSequenceClassifierConfig()

	core.RequireNoError(t, err)
	core.RequireTrue(t, hasClassifier)
	core.AssertEqual(t, "classifier.weight", cfg.WeightTensor)
	core.AssertEqual(t, "classifier.bias", cfg.BiasTensor)
	core.AssertEqual(t, nativeDevicePointer(14), cfg.WeightPointer)
	core.AssertEqual(t, nativeDevicePointer(13), cfg.BiasPointer)
}

func TestHIPRuntime_LoadedSequenceClassifierConfigDoesNotPairForeignBias_Good(t *testing.T) {
	model := &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "bert", HiddenSize: 2},
		tensors: map[string]hipTensor{
			"score.weight": {
				info:    nativeTensorInfo{Name: "score.weight", Type: 0, Dimensions: []uint64{2, 2}, ByteSize: 16},
				pointer: 21,
			},
			"classifier.bias": {
				info:    nativeTensorInfo{Name: "classifier.bias", Type: 0, Dimensions: []uint64{2}, ByteSize: 8},
				pointer: 22,
			},
		},
	}

	cfg, hasClassifier, err := model.loadedSequenceClassifierConfig()

	core.RequireNoError(t, err)
	core.RequireTrue(t, hasClassifier)
	core.AssertEqual(t, "score.weight", cfg.WeightTensor)
	core.AssertEqual(t, "", cfg.BiasTensor)
	core.AssertEqual(t, nativeDevicePointer(21), cfg.WeightPointer)
	core.AssertEqual(t, nativeDevicePointer(0), cfg.BiasPointer)
}

func TestHIPRuntime_LoadModelBERTSequenceClassifierLoRAAdapterRejectsBadShape_Bad(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-bert-classifier-lora.hsaco")
	embeddingPayload, err := hipFloat32Payload([]float32{
		0, 0,
		0, 1,
		0, 0,
	})
	core.RequireNoError(t, err)
	classifierPayload, err := hipFloat32Payload([]float32{0, 0, 0, 1})
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "bert-bad-classifier-lora.bin")
	write := core.WriteFile(modelPath, append(append([]byte(nil), embeddingPayload...), classifierPayload...), 0o644)
	core.RequireTrue(t, write.OK)
	model, err := newHIPRuntime(&fakeHIPDriver{available: true}).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 3, HiddenSize: 2, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "embeddings.word_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{3, 2},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "classifier.weight",
			Type:       0,
			Dimensions: []uint64{2, 2},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(classifierPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	adapterPath := core.PathJoin(t.TempDir(), "bad_classifier_lora.json")
	write = core.WriteFile(adapterPath, []byte(`{
		"format":"rocm-classifier-lora",
		"target":"classifier.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"num_labels":2,
		"lora_a":[1,0],
		"lora_b":[4]
	}`), 0o644)
	core.RequireTrue(t, write.OK)

	_, err = loaded.LoadAdapter(adapterPath)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "LoRA B length")
	if !adapterIdentityIsZero(loaded.ActiveAdapter()) {
		t.Fatalf("active adapter = %+v, want zero after failed load", loaded.ActiveAdapter())
	}
}

func TestHIPRuntime_LoadModelBERTSequenceClassifierRerankRejectsBadHead_Bad(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-bert-sequence-classifier.hsaco")
	embeddingPayload, err := hipFloat32Payload([]float32{
		0, 0,
		0, 1,
		0, 0,
		0, 1,
	})
	core.RequireNoError(t, err)
	classifierPayload, err := hipFloat32Payload([]float32{1, 0, 0, 1, 1, 1})
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "bad-bert-sequence-classifier.bin")
	write := core.WriteFile(modelPath, append(append([]byte(nil), embeddingPayload...), classifierPayload...), 0o644)
	core.RequireTrue(t, write.OK)
	model, err := newHIPRuntime(&fakeHIPDriver{available: true}).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 4, HiddenSize: 2, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "embeddings.word_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{4, 2},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "classifier.weight",
			Type:       0,
			Dimensions: []uint64{2, 3},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(classifierPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)

	_, err = loaded.Rerank(context.Background(), inference.RerankRequest{Query: "hello", Documents: []string{"hello"}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "classifier hidden size")
}

func TestHIPRuntime_LoadModelRunsTinyLoRAAdapterWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-lora.hsaco")
	driver := &fakeHIPDriver{available: true}
	loaded, _ := loadHIPTinyF32FixtureModel(t, driver)

	status := loaded.KernelStatus()
	core.AssertEqual(t, hipKernelStatusLinked, status.LoRA)

	adapterDir := t.TempDir()
	adapterPath := core.PathJoin(adapterDir, "rocm_tiny_lora.json")
	writeTinyLoRAAdapterFile(t, adapterPath, `{
		"format":"rocm-tiny-lora",
		"name":"boost-two",
		"target":"output.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"vocab_size":3,
		"lora_a":[0,1],
		"lora_b":[0,0,2]
	}`)

	identity, err := loaded.LoadAdapter(adapterDir)
	core.RequireNoError(t, err)
	core.AssertEqual(t, adapterDir, identity.Path)
	core.AssertEqual(t, rocmTinyLoRAFormat, identity.Format)
	core.AssertEqual(t, 1, identity.Rank)
	core.AssertEqual(t, float32(1), identity.Alpha)
	core.AssertNotEmpty(t, identity.Hash)
	core.AssertEqual(t, hipKernelStatusLinked, identity.Labels["lora_kernel"])
	core.AssertEqual(t, hipKernelNameLoRA, identity.Labels["lora_kernel_name"])
	core.AssertEqual(t, adapterPath, identity.Labels["adapter_file"])
	core.AssertEqual(t, identity.Hash, loaded.ActiveAdapter().Hash)
	identity.TargetKeys[0] = "mutated"
	identity.Labels["lora_kernel"] = "mutated"
	active := loaded.ActiveAdapter()
	core.AssertEqual(t, "output.weight", active.TargetKeys[0])
	core.AssertEqual(t, hipKernelStatusLinked, active.Labels["lora_kernel"])

	classified, err := loaded.Classify(context.Background(), []string{"hello"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(2), classified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 1, 3}, classified[0].Logits, 0.0001)

	prefill, err := loaded.Prefill(context.Background(), hipPrefillRequest{TokenIDs: []int32{1}})
	core.RequireNoError(t, err)
	defer prefill.DeviceKV.Close()
	defer prefill.DescriptorTable.Close()
	core.AssertEqual(t, identity.Hash, prefill.Labels["adapter_hash"])
	core.AssertEqual(t, hipKernelStatusLinked, prefill.Labels["lora_kernel"])
	core.AssertEqual(t, hipKernelNameLoRA, prefill.Labels["lora_kernel_name"])
	assertFloat32SlicesNear(t, []float32{0, 1, 3}, prefill.Logits, 0.0001)

	stream, streamErr := loaded.Generate(context.Background(), "hello", inference.GenerateConfig{MaxTokens: 1})
	var generated []int32
	for token := range stream {
		generated = append(generated, token.ID)
	}
	core.RequireNoError(t, streamErr())
	core.AssertEqual(t, []int32{2}, generated)

	var sawLoRA bool
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameLoRA {
			sawLoRA = true
		}
	}
	core.AssertTrue(t, sawLoRA)

	core.RequireNoError(t, loaded.UnloadAdapter())
	if !adapterIdentityIsZero(loaded.ActiveAdapter()) {
		t.Fatalf("active adapter after unload = %+v, want zero", loaded.ActiveAdapter())
	}
	classified, err = loaded.Classify(context.Background(), []string{"hello"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(1), classified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 1, 1}, classified[0].Logits, 0.0001)
}

func TestHIPRuntime_LoadModelRunsTinyLoRAAdapterWithCodebookOutputWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-codebook-lora.hsaco")
	fixture := hipReferenceTinyLMFixture()
	embeddingPayload, err := hipFloat32Payload(fixture.EmbeddingTable)
	core.RequireNoError(t, err)
	codePayload := []byte{1, 0, 0, 1, 1, 1}
	codebookPayload, err := hipFloat32Payload([]float32{0, 1})
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "tiny-codebook-lora.bin")
	payload := append(append(append([]byte(nil), embeddingPayload...), codePayload...), codebookPayload...)
	write := core.WriteFile(modelPath, payload, 0o644)
	core.RequireTrue(t, write.OK)
	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: fixture.VocabSize, HiddenSize: fixture.HiddenSize, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "tok_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "output.weight",
			Type:       1000,
			TypeName:   "codebook:vq:dim=1",
			Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(codePayload)),
		}, {
			Name:       "output.codebook",
			Type:       0,
			Dimensions: []uint64{2, 1},
			Offset:     uint64(len(embeddingPayload) + len(codePayload)),
			ByteSize:   uint64(len(codebookPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)

	adapterPath := core.PathJoin(t.TempDir(), "rocm_tiny_lora.json")
	writeTinyLoRAAdapterFile(t, adapterPath, `{
		"format":"rocm-tiny-lora",
		"target":"output.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"vocab_size":3,
		"lora_a":[0,1],
		"lora_b":[0,0,2]
	}`)
	identity, err := loaded.LoadAdapter(adapterPath)
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmTinyLoRAFormat, identity.Format)

	classified, err := loaded.Classify(context.Background(), []string{"hello"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(2), classified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 1, 3}, classified[0].Logits, 0.0001)

	prefill, err := loaded.Prefill(context.Background(), hipPrefillRequest{TokenIDs: []int32{1}})
	core.RequireNoError(t, err)
	defer prefill.DeviceKV.Close()
	defer prefill.DescriptorTable.Close()
	core.AssertEqual(t, hipKernelNameCodebook, prefill.Labels["output_lookup_kernel_name"])
	core.AssertEqual(t, hipKernelNameLoRA, prefill.Labels["lora_kernel_name"])
	assertFloat32SlicesNear(t, []float32{0, 1, 3}, prefill.Logits, 0.0001)

	var sawCodebook, sawLoRA bool
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameCodebook {
			sawCodebook = true
		}
		if launch.Name == hipKernelNameLoRA {
			sawLoRA = true
		}
	}
	core.AssertTrue(t, sawCodebook)
	core.AssertTrue(t, sawLoRA)
}

func TestHIPRuntime_LoadTinyLoRAAdapterBadValidationKeepsActiveAdapter_Bad(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-lora.hsaco")
	loaded, _ := loadHIPTinyF32FixtureModel(t, &fakeHIPDriver{available: true})
	validPath := core.PathJoin(t.TempDir(), "valid-lora.json")
	writeTinyLoRAAdapterFile(t, validPath, `{
		"format":"rocm-tiny-lora",
		"target":"output.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"vocab_size":3,
		"lora_a":[0,1],
		"lora_b":[0,0,2]
	}`)
	previous, err := loaded.LoadAdapter(validPath)
	core.RequireNoError(t, err)

	invalidPath := core.PathJoin(t.TempDir(), "invalid-lora.json")
	writeTinyLoRAAdapterFile(t, invalidPath, `{
		"format":"rocm-tiny-lora",
		"target":"output.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"vocab_size":3,
		"lora_a":[0,1],
		"lora_b":[0,2]
	}`)

	identity, err := loaded.LoadAdapter(invalidPath)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "adapter LoRA B length must match vocab*rank")
	if !adapterIdentityIsZero(identity) {
		t.Fatalf("identity = %+v, want zero", identity)
	}
	active := loaded.ActiveAdapter()
	core.AssertEqual(t, previous.Path, active.Path)
	core.AssertEqual(t, previous.Hash, active.Hash)

	classified, err := loaded.Classify(context.Background(), []string{"hello"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(2), classified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 1, 3}, classified[0].Logits, 0.0001)
}

func TestHIPRuntime_LoadedTinyEmbedAndRerankNotLinked_Bad(t *testing.T) {
	loaded := &hipLoadedModel{kernels: newDefaultHIPKernelSet()}

	_, err := loaded.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"hello"}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native embedding kernels are not linked yet")
	_, err = loaded.Rerank(context.Background(), inference.RerankRequest{Query: "hello", Documents: []string{"doc"}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native rerank kernels are not linked yet")
}

func TestHIPRuntime_LoadedTinyEmbedAndRerankPreflightBeforeNotLinked_Bad(t *testing.T) {
	loaded := &hipLoadedModel{kernels: newDefaultHIPKernelSet()}

	_, err := loaded.Embed(context.Background(), inference.EmbeddingRequest{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input text is required")

	_, err = loaded.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"ok", "   "}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input 1 is empty")

	_, err = loaded.Rerank(context.Background(), inference.RerankRequest{Documents: []string{"doc"}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "query is required")

	_, err = loaded.Rerank(context.Background(), inference.RerankRequest{Query: "hello"})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "documents are required")

	_, err = loaded.Rerank(context.Background(), inference.RerankRequest{Query: "hello", Documents: []string{"doc", ""}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "document 1 is empty")
}

func TestHIPRuntime_LoadedTinyQ8ScaleValidation_Bad(t *testing.T) {
	for _, tt := range []struct {
		name     string
		typeName string
		want     string
	}{{
		name:     "empty-scale",
		typeName: "q8:",
		want:     "parse q8 output scale",
	}, {
		name:     "zero-scale",
		typeName: "q8:0",
		want:     "q8 output scale must be positive and finite",
	}, {
		name:     "negative-scale",
		typeName: "q8:-0.5",
		want:     "q8 output scale must be positive and finite",
	}, {
		name:     "nan-scale",
		typeName: "q8:NaN",
		want:     "q8 output scale must be positive and finite",
	}, {
		name:     "inf-scale",
		typeName: "q8:+Inf",
		want:     "q8 output scale must be positive and finite",
	}} {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, _, err := hipTinyLoadedOutputEncoding(nativeTensorInfo{Type: 24, TypeName: tt.typeName})
			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}
}

func TestHIPRuntime_LoadedTinyJANGTQOutputValidation_Bad(t *testing.T) {
	for _, tt := range []struct {
		name     string
		typeName string
		want     string
	}{{
		name:     "bad-bits",
		typeName: "jangtq:bits=3:group=2:scale=1",
		want:     "unsupported bit layout",
	}, {
		name:     "bad-group",
		typeName: "jangtq:bits=2:group=3:scale=1",
		want:     "group size must be a positive power of two",
	}, {
		name:     "bad-scale",
		typeName: "mxtq:bits=2:group=2:scale=0",
		want:     "JANGTQ scale must be positive and finite",
	}} {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, _, err := hipTinyLoadedOutputEncoding(nativeTensorInfo{Type: 999, TypeName: tt.typeName})
			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}

	model := &hipLoadedModel{
		driver:    &fakeHIPDriver{available: true},
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 3, HiddenSize: 2, QuantBits: 2},
		tensors: map[string]hipTensor{
			"tok_embeddings.weight": {
				info:    nativeTensorInfo{Name: "tok_embeddings.weight", Type: 0, Dimensions: []uint64{3, 2}, ByteSize: 24},
				pointer: 1,
			},
			"output.weight": {
				info:    nativeTensorInfo{Name: "output.weight", Type: 999, TypeName: "jangtq:bits=2:group=2:scale=1", Dimensions: []uint64{3, 2}, ByteSize: 1},
				pointer: 2,
			},
		},
	}
	_, err := model.loadedTinyLMConfig()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "output byte count")
}

func TestHIPRuntime_LoadedTinyCodebookOutputValidation_Bad(t *testing.T) {
	_, _, _, _, err := hipTinyLoadedOutputEncoding(nativeTensorInfo{Type: 1000, TypeName: "codebook:vq:dim=0"})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "codebook dimension must be positive")

	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-tiny.hsaco")
	fixture := hipReferenceTinyLMFixture()
	embeddingPayload, err := hipFloat32Payload(fixture.EmbeddingTable)
	core.RequireNoError(t, err)
	codePayload := []byte{1, 0, 0, 1, 1, 1}
	codebookPayload, err := hipFloat32Payload([]float32{0, 1})
	core.RequireNoError(t, err)
	codebookFP16Payload := []byte{0, 0, 0, 0}

	for _, tt := range []struct {
		name           string
		outputTypeName string
		outputByteSize uint64
		tensors        []nativeTensorInfo
		payload        []byte
		want           string
	}{{
		name:           "missing-table",
		outputTypeName: "codebook:vq:dim=1",
		tensors:        nil,
		payload:        append(append([]byte(nil), embeddingPayload...), codePayload...),
		want:           "codebook output table tensor is required",
	}, {
		name:           "vector-codes",
		outputTypeName: "codebook:vq:dim=2",
		tensors:        nil,
		payload:        append(append([]byte(nil), embeddingPayload...), codePayload...),
		want:           "codebook output code dimension must be 1",
	}, {
		name:           "output-code-byte-count",
		outputTypeName: "codebook:vq:dim=1",
		outputByteSize: uint64(len(codePayload) - 1),
		tensors:        nil,
		payload:        append(append([]byte(nil), embeddingPayload...), codePayload...),
		want:           "output byte count",
	}, {
		name:           "table-not-f32",
		outputTypeName: "codebook:vq:dim=1",
		tensors: []nativeTensorInfo{{
			Name:       "output.codebook",
			Type:       1,
			TypeName:   "f16",
			Dimensions: []uint64{2, 1},
			Offset:     uint64(len(embeddingPayload) + len(codePayload)),
			ByteSize:   uint64(len(codebookFP16Payload)),
		}},
		payload: append(append(append([]byte(nil), embeddingPayload...), codePayload...), codebookFP16Payload...),
		want:    "codebook output table must be f32",
	}, {
		name:           "table-rank",
		outputTypeName: "codebook:vq:dim=1",
		tensors: []nativeTensorInfo{{
			Name:       "output.codebook",
			Type:       0,
			Dimensions: []uint64{2},
			Offset:     uint64(len(embeddingPayload) + len(codePayload)),
			ByteSize:   uint64(len(codebookPayload)),
		}},
		payload: append(append(append([]byte(nil), embeddingPayload...), codePayload...), codebookPayload...),
		want:    "codebook output table tensor must be rank 2",
	}, {
		name:           "table-dimension-mismatch",
		outputTypeName: "codebook:vq:dim=1",
		tensors: []nativeTensorInfo{{
			Name:       "output.codebook",
			Type:       0,
			Dimensions: []uint64{1, 2},
			Offset:     uint64(len(embeddingPayload) + len(codePayload)),
			ByteSize:   uint64(len(codebookPayload)),
		}},
		payload: append(append(append([]byte(nil), embeddingPayload...), codePayload...), codebookPayload...),
		want:    "codebook output table dimension mismatch",
	}} {
		t.Run(tt.name, func(t *testing.T) {
			modelPath := core.PathJoin(t.TempDir(), "tiny-codebook-bad.bin")
			write := core.WriteFile(modelPath, tt.payload, 0o644)
			core.RequireTrue(t, write.OK)
			driver := &fakeHIPDriver{available: true}
			runtime := newHIPRuntime(driver)
			tensors := []nativeTensorInfo{{
				Name:       "tok_embeddings.weight",
				Type:       0,
				Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
				Offset:     0,
				ByteSize:   uint64(len(embeddingPayload)),
			}, {
				Name:       "output.weight",
				Type:       1000,
				TypeName:   tt.outputTypeName,
				Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
				Offset:     uint64(len(embeddingPayload)),
				ByteSize:   uint64(len(codePayload)),
			}}
			if tt.outputByteSize > 0 {
				tensors[1].ByteSize = tt.outputByteSize
			}
			tensors = append(tensors, tt.tensors...)
			model, err := runtime.LoadModel(modelPath, nativeLoadConfig{
				ModelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: fixture.VocabSize, HiddenSize: fixture.HiddenSize, QuantBits: 32},
				Tensors:   tensors,
			})
			core.RequireNoError(t, err)
			defer model.Close()
			loaded, ok := model.(*hipLoadedModel)
			core.RequireTrue(t, ok)

			_, err = loaded.loadedTinyLMConfig()
			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}

	for _, tt := range []struct {
		name     string
		byteSize uint64
		pointer  nativeDevicePointer
		want     string
	}{{
		name:     "table-byte-count",
		byteSize: 4,
		pointer:  3,
		want:     "codebook table byte count",
	}, {
		name:     "table-pointer",
		byteSize: uint64(len(codebookPayload)),
		pointer:  0,
		want:     "codebook output table tensor pointer",
	}} {
		t.Run(tt.name, func(t *testing.T) {
			model := &hipLoadedModel{
				driver:    &fakeHIPDriver{available: true},
				modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: fixture.VocabSize, HiddenSize: fixture.HiddenSize, QuantBits: 8},
				tensors: map[string]hipTensor{
					"tok_embeddings.weight": {
						info:    nativeTensorInfo{Name: "tok_embeddings.weight", Type: 0, Dimensions: []uint64{3, 2}, ByteSize: uint64(len(embeddingPayload))},
						pointer: 1,
					},
					"output.weight": {
						info:    nativeTensorInfo{Name: "output.weight", Type: 1000, TypeName: "codebook:vq:dim=1", Dimensions: []uint64{3, 2}, ByteSize: uint64(len(codePayload))},
						pointer: 2,
					},
					"output.codebook": {
						info:    nativeTensorInfo{Name: "output.codebook", Type: 0, Dimensions: []uint64{2, 1}, ByteSize: tt.byteSize},
						pointer: tt.pointer,
					},
				},
			}
			_, err = model.loadedTinyLMConfig()
			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}
}

func TestHIPRuntime_LoadModelBadFreeOnCopyFailure_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	driver := &fakeHIPDriver{available: true, copyErr: core.NewError("copy failed")}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3"},
		DataOffset: dataOffset,
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", ByteSize: 16},
			{Name: "output.weight", Offset: 16, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertNil(t, model)
	core.AssertEqual(t, 1, len(driver.frees))
}

func TestHIPRuntime_LoadModelBadFreesAllTensorsOnSecondCopyFailure_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true, copyErr: core.NewError("copy failed"), copyErrAt: 2}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3"},
		DataOffset: dataOffset,
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", ByteSize: 16},
			{Name: "output.weight", Offset: 16, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertNil(t, model)
	core.AssertEqual(t, []uint64{16, 16}, driver.allocations)
	core.AssertEqual(t, []uint64{16, 16}, driver.copies)
	core.AssertEqual(t, 2, driver.pinnedCopies)
	core.AssertEqual(t, 2, len(driver.frees))
}

func TestHIPRuntime_LoadModelBadShortTensorReadRejectedBeforeAllocation_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3"},
		DataOffset: dataOffset,
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", ByteSize: 16},
			{Name: "output.weight", Offset: 24, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tensor byte range exceeds file size")
	core.AssertNil(t, model)
	core.AssertEqual(t, 0, len(driver.allocations))
	core.AssertEqual(t, 0, len(driver.copies))
	core.AssertEqual(t, 0, len(driver.frees))
}

func TestHIPRuntime_LoadModelUglyEmptyTensorMap_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	driver := &fakeHIPDriver{available: true}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3"},
		DataOffset: dataOffset,
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "missing token embedding tensor")
	core.AssertNil(t, model)
	core.AssertEqual(t, 0, len(driver.allocations))
}

func TestHIPRuntime_Validate_BadMissingOutputHead(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors:   []nativeTensorInfo{{Name: "tok_embeddings.weight", Type: 0, ByteSize: 16}},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "missing output head tensor")
}

func TestHIPRuntime_Validate_BadRequiredTensorHasZeroBytes(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, ByteSize: 0},
			{Name: "output.weight", Type: 0, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "zero byte size")
}

func TestHIPRuntime_Validate_BadMismatchedLayerCount(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3", NumLayers: 2},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, ByteSize: 16},
			{Name: "model.layers.0.attn.weight", Type: 0, ByteSize: 16},
			{Name: "output.weight", Type: 0, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "mismatched layer count")
}

func TestHIPRuntime_Validate_GoodKnownNumericQuantizedDTypeWithoutName(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 15, ByteSize: 16},
			{Name: "output.weight", Type: 15, ByteSize: 16},
		},
	})

	core.AssertNoError(t, err)
}

func TestHIPRuntime_Validate_GoodKnownGGUFQuantizedTypeName(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 15, TypeName: "Q8_K", ByteSize: 16},
			{Name: "output.weight", Type: 15, TypeName: "Q8_K", ByteSize: 16},
		},
	})

	core.AssertNoError(t, err)
}

func TestHIPRuntime_Validate_GoodGGUFTokenEmbeddingAlias(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "gemma3"},
		Tensors: []nativeTensorInfo{
			{Name: "token_embd.weight", Type: 15, TypeName: "Q4_K", ByteSize: 16},
			{Name: "output.weight", Type: 15, TypeName: "Q4_K", ByteSize: 16},
		},
	})

	core.AssertNoError(t, err)
}

func TestHIPRuntime_Validate_GoodGemma4TiedSafetensorsEmbedding(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo:          inference.ModelInfo{Architecture: "gemma4", VocabSize: 8, HiddenSize: 16, NumLayers: 1, QuantBits: 4, QuantGroup: 64},
		TiedWordEmbeddings: true,
		Tensors: []nativeTensorInfo{
			{Name: "language_model.model.embed_tokens.weight", Dimensions: []uint64{8, 2}, Type: 26, TypeName: "U32", ByteSize: 64},
			{Name: "language_model.model.embed_tokens.biases", Dimensions: []uint64{8, 2}, Type: 30, TypeName: "BF16", ByteSize: 32},
			{Name: "language_model.model.layers.0.input_layernorm.weight", Dimensions: []uint64{16}, Type: 30, TypeName: "BF16", ByteSize: 32},
		},
	})

	core.AssertNoError(t, err)
}

func TestHIPRuntime_LoadModelCopiesShardedSafetensorsSources_Good(t *testing.T) {
	dir := t.TempDir()
	shardA := core.PathJoin(dir, "model-00001-of-00002.safetensors")
	shardB := core.PathJoin(dir, "model-00002-of-00002.safetensors")
	writeNativeContractFile(t, shardA, string(make([]byte, 8+64)))
	writeNativeContractFile(t, shardB, string(make([]byte, 8+32)))
	driver := &fakeHIPDriver{available: true}
	runtime := newHIPRuntime(driver)

	model, err := runtime.LoadModel(dir, nativeLoadConfig{
		ModelInfo:          inference.ModelInfo{Architecture: "gemma4", VocabSize: 8, HiddenSize: 16, NumLayers: 1, QuantBits: 4, QuantGroup: 64},
		TiedWordEmbeddings: true,
		Tensors: []nativeTensorInfo{
			{Name: "language_model.model.embed_tokens.weight", SourcePath: shardA, DataOffset: 8, Dimensions: []uint64{8, 2}, Type: 26, TypeName: "U32", ByteSize: 64},
			{Name: "language_model.model.layers.0.input_layernorm.weight", SourcePath: shardB, DataOffset: 8, Dimensions: []uint64{16}, Type: 30, TypeName: "BF16", ByteSize: 32},
		},
	})

	core.AssertNoError(t, err)
	core.AssertNotNil(t, model)
	core.AssertEqual(t, []uint64{64, 32}, driver.allocations)
	core.AssertEqual(t, []uint64{64, 32}, driver.copies)
	core.AssertEqual(t, 2, driver.pinnedCopies)
	core.AssertNoError(t, model.Close())
	core.AssertEqual(t, 2, len(driver.frees))
}

func TestHIPRuntime_Validate_BadUnsupportedDType(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 999, TypeName: "q9", ByteSize: 16},
			{Name: "output.weight", Type: 0, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported tensor dtype")
}

func TestHIPRuntime_Validate_BadEmptyTensorName(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors: []nativeTensorInfo{
			{Name: "", Type: 0, ByteSize: 16},
			{Name: "output.weight", Type: 0, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tensor name is required")
}

func TestHIPRuntime_Validate_BadDuplicateTensorName(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, ByteSize: 16},
			{Name: "TOK_EMBEDDINGS.WEIGHT", Type: 0, ByteSize: 16},
			{Name: "output.weight", Type: 0, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "duplicate tensor name")
}

func TestHIPRuntime_Validate_BadUnsupportedQuantization(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3", QuantBits: 12},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, ByteSize: 16},
			{Name: "output.weight", Type: 0, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported quantization")
}

func TestHIPRuntime_Validate_BadNegativeDataOffset(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3"},
		DataOffset: -1,
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, ByteSize: 16},
			{Name: "output.weight", Type: 0, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "data offset")
}

func TestHIPRuntime_Validate_BadTensorDataOffsetOverflow(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3"},
		DataOffset: 1,
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, ByteSize: 16},
			{Name: "output.weight", Type: 0, Offset: 1 << 63, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "offset overflows")
}

func TestHIPRuntime_Validate_BadTensorFileRangeOverflow(t *testing.T) {
	_, err := hipTensorFileEnd(1, 1<<63)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "overflows")
}

func TestHIPRuntime_Validate_GoodProjectionShapes(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3", VocabSize: 4, HiddenSize: 2},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, Dimensions: []uint64{2, 4}, ByteSize: 32},
			{Name: "output.weight", Type: 1, Dimensions: []uint64{4, 2}, ByteSize: 16},
		},
	})

	core.AssertNoError(t, err)
}

func TestHIPRuntime_Validate_GoodMLXAffinePerModuleBits(t *testing.T) {
	const (
		vocab      = 4
		hidden     = 2816
		packedCols = hidden * 8 / 32
	)
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "diffusion_gemma", VocabSize: vocab, HiddenSize: hidden, QuantBits: 4, QuantGroup: 64},
		Tensors: []nativeTensorInfo{
			{Name: "language_model.model.embed_tokens.weight", Type: 26, TypeName: "U32", Dimensions: []uint64{vocab, packedCols}, ByteSize: vocab * packedCols * 4},
			{Name: "language_model.lm_head.weight", Type: 26, TypeName: "U32", Dimensions: []uint64{vocab, packedCols}, ByteSize: vocab * packedCols * 4},
		},
	})

	core.AssertNoError(t, err)
}

func TestHIPRuntime_Validate_BadProjectionRank(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3", VocabSize: 4, HiddenSize: 2},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, Dimensions: []uint64{8}, ByteSize: 32},
			{Name: "output.weight", Type: 0, Dimensions: []uint64{2, 4}, ByteSize: 32},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "projection tensor must be rank 2")
}

func TestHIPRuntime_Validate_BadProjectionIdentityMismatch(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3", VocabSize: 32000, HiddenSize: 4096},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 1, Dimensions: []uint64{128, 4096}, ByteSize: 1048576},
			{Name: "output.weight", Type: 1, Dimensions: []uint64{4096, 128}, ByteSize: 1048576},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "missing vocab size")
}

func TestHIPRuntime_Validate_BadByteSizeMismatch(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, Dimensions: []uint64{2, 4}, ByteSize: 16},
			{Name: "output.weight", Type: 0, Dimensions: []uint64{2, 4}, ByteSize: 32},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tensor byte size mismatch")
}

func TestHIPRuntime_Validate_UglyZeroDimension(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, Dimensions: []uint64{0, 4}, ByteSize: 0},
			{Name: "output.weight", Type: 0, Dimensions: []uint64{2, 4}, ByteSize: 32},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "zero dimension")
}

func TestHIPRuntime_DecodeKernelsNotLinked_Bad(t *testing.T) {
	model := &hipLoadedModel{}

	stream, streamErr := model.Generate(context.Background(), "hello", inference.DefaultGenerateConfig())
	for range stream {
	}
	err := streamErr()

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native decode kernels are not linked yet")
}

func TestHIPRuntime_CloseGoodIdempotentClearsRuntimeState(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	model := &hipLoadedModel{
		driver:  driver,
		tensors: map[string]hipTensor{"tok_embeddings.weight": {info: nativeTensorInfo{ByteSize: 16}, pointer: 7}},
		adapter: inference.AdapterIdentity{Path: "domain.safetensors", Format: "lora"},
	}

	core.AssertNoError(t, model.Close())
	core.AssertNoError(t, model.Close())

	core.AssertEqual(t, 1, len(driver.frees))
	if !adapterIdentityIsZero(model.ActiveAdapter()) {
		t.Fatalf("active adapter = %+v, want zero after close", model.ActiveAdapter())
	}
	core.AssertEqual(t, uint64(0), model.Metrics().ActiveMemoryBytes)
}

func TestHIPRuntime_LoadAdapterBadEmptyPath_Bad(t *testing.T) {
	model := &hipLoadedModel{}

	identity, err := model.LoadAdapter(" \t")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "adapter path is required")
	if !adapterIdentityIsZero(identity) {
		t.Fatalf("identity = %+v, want zero", identity)
	}
	if !adapterIdentityIsZero(model.ActiveAdapter()) {
		t.Fatalf("active adapter = %+v, want zero", model.ActiveAdapter())
	}
}

func TestHIPRuntime_LoadAdapterBadNotLinkedKeepsActiveAdapter_Bad(t *testing.T) {
	model := &hipLoadedModel{adapter: inference.AdapterIdentity{Path: "previous.safetensors", Format: "lora"}}

	identity, err := model.LoadAdapter("domain.safetensors")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native LoRA adapter application is not linked yet")
	core.AssertContains(t, err.Error(), "domain.safetensors")
	if !adapterIdentityIsZero(identity) {
		t.Fatalf("identity = %+v, want zero", identity)
	}
	if got := model.ActiveAdapter(); got.Path != "previous.safetensors" || got.Format != "lora" {
		t.Fatalf("active adapter = %+v, want previous adapter", got)
	}
}

func loadHIPTinyF32FixtureModel(t *testing.T, driver *fakeHIPDriver) (*hipLoadedModel, hipReferenceTinyLMConfig) {
	t.Helper()
	fixture := hipReferenceTinyLMFixture()
	embeddingPayload, err := hipFloat32Payload(fixture.EmbeddingTable)
	core.RequireNoError(t, err)
	outputPayload, err := hipFloat32Payload(fixture.OutputWeights)
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "tiny.bin")
	payload := append(append([]byte(nil), embeddingPayload...), outputPayload...)
	write := core.WriteFile(modelPath, payload, 0o644)
	core.RequireTrue(t, write.OK)
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: fixture.VocabSize, HiddenSize: fixture.HiddenSize, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "tok_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "output.weight",
			Type:       0,
			Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(outputPayload)),
		}},
	})
	core.RequireNoError(t, err)
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	t.Cleanup(func() {
		core.AssertNoError(t, loaded.Close())
	})
	return loaded, fixture
}

func writeTinyLoRAAdapterFile(t *testing.T, path, payload string) {
	t.Helper()
	write := core.WriteFile(path, []byte(payload), 0o644)
	core.RequireTrue(t, write.OK)
}

type fakeHIPDriver struct {
	available                bool
	device                   nativeDeviceInfo
	nextPointer              nativeDevicePointer
	allocations              []uint64
	copies                   []uint64
	frees                    []nativeDevicePointer
	launches                 []hipKernelLaunchConfig
	memory                   map[nativeDevicePointer][]byte
	copyErr                  error
	copyErrAt                int
	copyHostErrAfterLaunches int
	pinnedCopies             int
	jangtqInputScratch       []float32
	jangtqBiasScratch        []float32
	jangtqOutputScratch      []float32
	jangtqQuantizedScratch   []int8
	memsets                  []uint64
	launchErr                error
	skipLaunchRecording      bool
	skipDriverRecording      bool
	releaseLaunchPackets     bool
	maxLiveBytes             uint64
	liveBytes                uint64
	allocationBytes          map[nativeDevicePointer]uint64
}

func (driver *fakeHIPDriver) Available() bool { return driver.available }
func (driver *fakeHIPDriver) DeviceInfo() nativeDeviceInfo {
	return driver.device
}
func (driver *fakeHIPDriver) Malloc(size uint64) (nativeDevicePointer, error) {
	if !driver.skipDriverRecording {
		driver.allocations = append(driver.allocations, size)
	}
	if driver.maxLiveBytes > 0 && driver.liveBytes+size > driver.maxLiveBytes {
		return 0, core.E("rocm.hip.hipMalloc", "HIP returned 2", nil)
	}
	if driver.nextPointer == 0 {
		driver.nextPointer = 0x1000
	}
	pointer := driver.nextPointer
	driver.nextPointer += nativeDevicePointer(size) + 0x1000
	if driver.memory == nil {
		driver.memory = map[nativeDevicePointer][]byte{}
	}
	if driver.allocationBytes == nil {
		driver.allocationBytes = map[nativeDevicePointer]uint64{}
	}
	driver.memory[pointer] = make([]byte, int(size))
	driver.allocationBytes[pointer] = size
	driver.liveBytes += size
	return pointer, nil
}
func (driver *fakeHIPDriver) Free(pointer nativeDevicePointer) error {
	if !driver.skipDriverRecording {
		driver.frees = append(driver.frees, pointer)
	}
	if size := driver.allocationBytes[pointer]; size > 0 {
		if driver.liveBytes >= size {
			driver.liveBytes -= size
		} else {
			driver.liveBytes = 0
		}
		delete(driver.allocationBytes, pointer)
	}
	delete(driver.memory, pointer)
	return nil
}
func (driver *fakeHIPDriver) CopyHostToDevice(pointer nativeDevicePointer, data []byte) error {
	if !driver.skipDriverRecording {
		driver.copies = append(driver.copies, uint64(len(data)))
	}
	if driver.shouldFailCopy(true) {
		return driver.copyErr
	}
	if target, offset, ok := driver.memoryForPointer(pointer, len(data)); ok {
		copy(target[offset:], data)
	}
	return nil
}
func (driver *fakeHIPDriver) CopyPinnedHostToDevice(pointer nativeDevicePointer, host unsafe.Pointer, sizeBytes int) error {
	driver.pinnedCopies++
	if sizeBytes <= 0 {
		if !driver.skipDriverRecording {
			driver.copies = append(driver.copies, 0)
		}
		return nil
	}
	data := unsafe.Slice((*byte)(host), sizeBytes)
	if !driver.skipDriverRecording {
		driver.copies = append(driver.copies, uint64(len(data)))
	}
	if driver.shouldFailCopy(true) {
		return driver.copyErr
	}
	if target, offset, ok := driver.memoryForPointer(pointer, len(data)); ok {
		copy(target[offset:], data)
	}
	return nil
}
func (driver *fakeHIPDriver) CopyDeviceToHost(pointer nativeDevicePointer, data []byte) error {
	if !driver.skipDriverRecording {
		driver.copies = append(driver.copies, uint64(len(data)))
	}
	if driver.shouldFailCopy(false) {
		return driver.copyErr
	}
	if source, offset, ok := driver.memoryForPointer(pointer, len(data)); ok {
		copy(data, source[offset:offset+len(data)])
	}
	return nil
}
func (driver *fakeHIPDriver) shouldFailCopy(hostToDevice bool) bool {
	if driver.copyErr == nil {
		return false
	}
	if driver.copyErrAt > 0 && len(driver.copies) == driver.copyErrAt {
		return true
	}
	if hostToDevice && driver.copyHostErrAfterLaunches > 0 && len(driver.launches) >= driver.copyHostErrAfterLaunches {
		return true
	}
	return driver.copyErrAt == 0 && driver.copyHostErrAfterLaunches == 0
}
func (driver *fakeHIPDriver) MemsetAsync(pointer nativeDevicePointer, value byte, size uint64) error {
	if !driver.skipDriverRecording {
		driver.memsets = append(driver.memsets, size)
	}
	if size == 0 {
		return nil
	}
	if size > uint64(int(^uint(0)>>1)) {
		return core.E("rocm.hip.FakeMemset", "memset size is out of range", nil)
	}
	target, offset, ok := driver.memoryForPointer(pointer, int(size))
	if !ok {
		return core.E("rocm.hip.FakeMemset", "memset buffer is missing", nil)
	}
	for index := 0; index < int(size); index++ {
		target[offset+index] = value
	}
	return nil
}
func (driver *fakeHIPDriver) LaunchKernel(config hipKernelLaunchConfig) error {
	if driver.releaseLaunchPackets {
		defer hipReleaseLaunchPacket(config.Args)
	}
	if !driver.skipLaunchRecording {
		copied := config
		copied.Args = append([]byte(nil), config.Args...)
		driver.launches = append(driver.launches, copied)
	}
	if driver.launchErr != nil {
		return driver.launchErr
	}
	switch config.Name {
	case hipKernelNamePrefill:
		return driver.launchPrefill(config.Args)
	case hipKernelNameDecode:
		return driver.launchDecode(config.Args)
	case hipKernelNameKVEncodeToken:
		return driver.launchKVEncodeToken(config.Args)
	case hipKernelNameKVEncodeTokenValueNorm:
		return driver.launchKVEncodeTokenValueNorm(config.Args)
	case hipKernelNameKVEncodeTokenValueNormDescriptorAppend:
		return driver.launchKVEncodeTokenValueNormDescriptorAppend(config.Args)
	case hipKernelNameKVDescriptorAppend:
		return driver.launchKVDescriptorAppend(config.Args)
	case hipKernelNameProjection:
		return driver.launchProjection(config.Args)
	case hipKernelNameProjectionBatch:
		return driver.launchProjectionBatch(config.Args)
	case hipKernelNameMLXQ4Proj:
		return driver.launchMLXQ4Projection(config.Args)
	case hipKernelNameMLXQ4ProjQ4G32Rows3840Cols15360:
		return driver.launchMLXQ4Projection(config.Args)
	case hipKernelNameMLXQ4ProjCols256:
		return driver.launchMLXQ4Projection(config.Args)
	case hipKernelNameMLXQ4ProjQ6G16Row16:
		return driver.launchMLXQ4Projection(config.Args)
	case hipKernelNameMLXQ4ProjQ6Row16:
		return driver.launchMLXQ4Projection(config.Args)
	case hipKernelNameMLXQ4ProjQ6Row32:
		return driver.launchMLXQ4Projection(config.Args)
	case hipKernelNameMLXQ4ProjQ6Row64:
		return driver.launchMLXQ4Projection(config.Args)
	case hipKernelNameMLXQ4ProjBatch:
		return driver.launchMLXQ4ProjectionBatch(config.Args)
	case hipKernelNameMLXQ4ProjBatchQ6Row16:
		return driver.launchMLXQ4ProjectionBatch(config.Args)
	case hipKernelNameMLXQ4ProjGreedy:
		return driver.launchMLXQ4ProjectionGreedy(config.Args)
	case hipKernelNameMLXQ4ProjGreedyQ6Row64:
		return driver.launchMLXQ4ProjectionGreedy(config.Args)
	case hipKernelNameMLXQ4ProjGreedyBatch:
		return driver.launchMLXQ4ProjectionGreedyBatch(config.Args)
	case hipKernelNameMLXQ4ProjGreedyBatchQ6Row64:
		return driver.launchMLXQ4ProjectionGreedyBatch(config.Args)
	case hipKernelNameMLXQ4ProjScores:
		return driver.launchMLXQ4ProjectionScores(config.Args)
	case hipKernelNameMLXQ4ProjScoresQ6Row64:
		return driver.launchMLXQ4ProjectionScores(config.Args)
	case hipKernelNameMLXQ4ProjSelectedGreedy:
		return driver.launchMLXQ4ProjectionSelectedGreedy(config.Args)
	case hipKernelNameMLXQ4ProjSelectedGreedyQ6Row64:
		return driver.launchMLXQ4ProjectionSelectedGreedy(config.Args)
	case hipKernelNameOrderedEmbeddingCandidates:
		return driver.launchOrderedEmbeddingCandidates(config.Args)
	case hipKernelNamePackedTopK:
		return driver.launchPackedTopK(config.Args)
	case hipKernelNamePackedTopKSample:
		return driver.launchPackedTopKSample(config.Args)
	case hipKernelNameMLXQ4TripleProj:
		return driver.launchMLXQ4TripleProjection(config.Args)
	case hipKernelNameMLXQ4TripleProjQ6Row16:
		return driver.launchMLXQ4TripleProjection(config.Args)
	case hipKernelNameMLXQ4TripleProjQ6Row64:
		return driver.launchMLXQ4TripleProjection(config.Args)
	case hipKernelNameMLXQ4PairProj:
		return driver.launchMLXQ4TripleProjection(config.Args)
	case hipKernelNameMLXQ4GELUTanhMul:
		return driver.launchMLXQ4GELUTanhMultiply(config.Args)
	case hipKernelNameMLXQ4GELUTanhMulQ4G32Cols1536Row16:
		return driver.launchMLXQ4GELUTanhMultiply(config.Args)
	case hipKernelNameMLXQ4GELUTanhMulQ4G32Rows15360Cols3840:
		return driver.launchMLXQ4GELUTanhMultiply(config.Args)
	case hipKernelNameMLXQ4GELUTanhMulQ4G32Rows15360Cols3840Row8:
		return driver.launchMLXQ4GELUTanhMultiply(config.Args)
	case hipKernelNameMLXQ4GELUTanhMLPQ4G32Cols1536Persistent:
		return driver.launchMLXQ4GELUTanhMLPPersistent(config.Args)
	case hipKernelNameMLXQ4GELUTanhMulQ6Cols1536:
		return driver.launchMLXQ4GELUTanhMultiply(config.Args)
	case hipKernelNameMLXQ4GELUTanhMulQ6Cols1536Row32:
		return driver.launchMLXQ4GELUTanhMultiply(config.Args)
	case hipKernelNameMLXQ4GELUTanhMulQ6Cols1536Row64:
		return driver.launchMLXQ4GELUTanhMultiply(config.Args)
	case hipKernelNameMLXQ4GELUTanhMulBatch:
		return driver.launchMLXQ4GELUTanhMultiplyBatch(config.Args)
	case hipKernelNameMLXQ4GELUTanhProj:
		return driver.launchMLXQ4GELUTanhProjection(config.Args)
	case hipKernelNameMLXQ4GELUTanhProjQ6Row16:
		return driver.launchMLXQ4GELUTanhProjection(config.Args)
	case hipKernelNameMLXQ4GELUTanhProjBatch:
		return driver.launchMLXQ4GELUTanhProjectionBatch(config.Args)
	case hipKernelNameRMSNormResidualAddGELUTanhProj:
		return driver.launchRMSNormResidualAddGELUTanhProjection(config.Args)
	case hipKernelNameRMSNorm:
		return driver.launchRMSNorm(config.Args)
	case hipKernelNameRMSNormResidualAdd:
		return driver.launchRMSNormResidualAdd(config.Args)
	case hipKernelNameRMSNormResAddNorm:
		return driver.launchRMSNormResidualAddNorm(config.Args)
	case hipKernelNameRMSNormHeads:
		return driver.launchRMSNormHeads(config.Args)
	case hipKernelNameRMSNormRoPEHeads:
		return driver.launchRMSNormRoPEHeads(config.Args)
	case hipKernelNameRMSNormRoPEHeadsPair:
		return driver.launchRMSNormRoPEHeadsPair(config.Args)
	case hipKernelNameRMSNormRoPEHeadsBatch:
		return driver.launchRMSNormRoPEHeadsBatch(config.Args)
	case hipKernelNameRMSNormRoPEHeadsPairLaneBatch:
		return driver.launchRMSNormRoPEHeadsPairLaneBatch(config.Args)
	case hipKernelNameMoECombineNorms:
		return driver.launchMoECombineNorms(config.Args)
	case hipKernelNameRoPE:
		return driver.launchRoPE(config.Args)
	case hipKernelNameRoPEHeads:
		return driver.launchRoPEHeads(config.Args)
	case hipKernelNameGreedy:
		return driver.launchGreedySample(config.Args)
	case hipKernelNameSoftcapGreedy:
		return driver.launchSoftcapGreedySample(config.Args)
	case hipKernelNameAttention:
		return driver.launchAttention(config.Args)
	case hipKernelNameAttentionHeads:
		return driver.launchAttentionHeads(config.Args)
	case hipKernelNameAttentionHeadsBatchCausal:
		return driver.launchAttentionHeadsBatchCausal(config.Args)
	case hipKernelNameAttentionHeadsBatchCapped:
		return driver.launchAttentionHeadsBatchCausal(config.Args)
	case hipKernelNameAttentionHeadsLaneBatch:
		return driver.launchAttentionHeadsLaneBatch(config.Args)
	case hipKernelNameAttentionHeadsBatchCausalQueryRMSRoPE:
		return driver.launchAttentionHeadsBatchCausalQueryRMSRoPE(config.Args)
	case hipKernelNameAttentionHeadsBatchChunkedStage1:
		return driver.launchAttentionHeadsBatchChunked(config.Args, false)
	case hipKernelNameAttentionHeadsBatchChunkedStage1GQA2:
		return driver.launchAttentionHeadsBatchChunked(config.Args, false)
	case hipKernelNameAttentionHeadsBatchChunkedStage2:
		return driver.launchAttentionHeadsBatchChunked(config.Args, true)
	case hipKernelNameVectorAdd:
		return driver.launchVectorAdd(config.Args)
	case hipKernelNameVectorAddScaled:
		return driver.launchVectorAddScaled(config.Args)
	case hipKernelNameVectorScale:
		return driver.launchVectorScale(config.Args)
	case hipKernelNamePerLayerInputTranspose:
		return driver.launchPerLayerInputTranspose(config.Args)
	case hipKernelNameSwiGLU:
		return driver.launchSwiGLU(config.Args)
	case hipKernelNameGELUTanhMul:
		return driver.launchGELUTanhMultiply(config.Args)
	case hipKernelNameMoERouter:
		return driver.launchMoERouter(config.Args)
	case hipKernelNameMoELazy:
		return driver.launchMoELazyExperts(config.Args)
	case hipKernelNameJANGTQ:
		return driver.launchJANGTQProjection(config.Args)
	case hipKernelNameCodebook:
		return driver.launchCodebookLookup(config.Args)
	case hipKernelNameLoRA:
		return driver.launchLoRAProjection(config.Args)
	case hipKernelNameEmbedLookup:
		return driver.launchEmbeddingLookup(config.Args, false)
	case hipKernelNameEmbedLookupGreedyToken:
		return driver.launchEmbeddingLookup(config.Args, true)
	case hipKernelNameEmbedMean:
		return driver.launchEmbeddingMeanPool(config.Args)
	case hipKernelNameRerank:
		return driver.launchRerankCosine(config.Args)
	case hipKernelNameTinyPrefill:
		return driver.launchTinyPrefill(config.Args)
	case hipKernelNameTinyDecode:
		return driver.launchTinyDecode(config.Args)
	case hipKernelNameCrossEntropy:
		return driver.launchCrossEntropyLoss(config.Args)
	case hipKernelNameDistillKL:
		return driver.launchDistillationKLLoss(config.Args)
	case hipKernelNameGRPOAdvantage:
		return driver.launchGRPOAdvantage(config.Args)
	case hipKernelNameAdamWUpdate:
		return driver.launchAdamWUpdate(config.Args)
	}
	return nil
}

func (driver *fakeHIPDriver) launchCrossEntropyLoss(args []byte) error {
	if len(args) != hipCrossEntropyLossLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "cross entropy launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipCrossEntropyLossLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipCrossEntropyLossLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "cross entropy launch header mismatch", nil)
	}
	logitPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	targetPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	batch := int(binary.LittleEndian.Uint32(args[32:]))
	vocab := int(binary.LittleEndian.Uint32(args[36:]))
	logitBytes := int(binary.LittleEndian.Uint32(args[40:]))
	targetBytes := int(binary.LittleEndian.Uint32(args[44:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[48:]))
	if batch <= 0 || vocab <= 0 || logitBytes != batch*vocab*4 || targetBytes != batch*4 || outputBytes != hipCrossEntropyLossOutputBytes {
		return core.E("rocm.hip.FakeLaunch", "cross entropy shape metadata mismatch", nil)
	}
	logitData, logitOffset, ok := driver.memoryForPointer(logitPointer, logitBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "cross entropy logits buffer is missing", nil)
	}
	targetData, targetOffset, ok := driver.memoryForPointer(targetPointer, targetBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "cross entropy target buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "cross entropy output buffer is missing", nil)
	}
	logits, err := hipFloat32PayloadValues(logitData[logitOffset : logitOffset+logitBytes])
	if err != nil {
		return err
	}
	targets := make([]int, batch)
	for index := range targets {
		targets[index] = int(int32(binary.LittleEndian.Uint32(targetData[targetOffset+index*4:])))
	}
	loss, perplexity, err := rocmReferenceCrossEntropyLoss(splitFloat32Vectors(logits, vocab), targets)
	if err != nil {
		return err
	}
	binary.LittleEndian.PutUint64(outputData[outputOffset:], math.Float64bits(loss))
	binary.LittleEndian.PutUint64(outputData[outputOffset+8:], math.Float64bits(perplexity))
	return nil
}

func (driver *fakeHIPDriver) launchDistillationKLLoss(args []byte) error {
	if len(args) != hipDistillationKLLossLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "distillation KL launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipDistillationKLLossLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipDistillationKLLossLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "distillation KL launch header mismatch", nil)
	}
	studentPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	teacherPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	batch := int(binary.LittleEndian.Uint32(args[32:]))
	vocab := int(binary.LittleEndian.Uint32(args[36:]))
	studentBytes := int(binary.LittleEndian.Uint32(args[40:]))
	teacherBytes := int(binary.LittleEndian.Uint32(args[44:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[48:]))
	temperature := math.Float64frombits(binary.LittleEndian.Uint64(args[56:]))
	if batch <= 0 || vocab <= 0 || studentBytes != batch*vocab*4 || teacherBytes != batch*vocab*4 ||
		outputBytes != hipDistillationKLLossOutputBytes || temperature <= 0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
		return core.E("rocm.hip.FakeLaunch", "distillation KL shape metadata mismatch", nil)
	}
	studentData, studentOffset, ok := driver.memoryForPointer(studentPointer, studentBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "distillation student buffer is missing", nil)
	}
	teacherData, teacherOffset, ok := driver.memoryForPointer(teacherPointer, teacherBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "distillation teacher buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "distillation output buffer is missing", nil)
	}
	students, err := hipFloat32PayloadValues(studentData[studentOffset : studentOffset+studentBytes])
	if err != nil {
		return err
	}
	teachers, err := hipFloat32PayloadValues(teacherData[teacherOffset : teacherOffset+teacherBytes])
	if err != nil {
		return err
	}
	kl, err := rocmReferenceDistillationKL(splitFloat32Vectors(students, vocab), splitFloat32Vectors(teachers, vocab), temperature)
	if err != nil {
		return err
	}
	binary.LittleEndian.PutUint64(outputData[outputOffset:], math.Float64bits(kl))
	return nil
}

func (driver *fakeHIPDriver) launchGRPOAdvantage(args []byte) error {
	if len(args) != hipGRPOAdvantageLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "GRPO advantage launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipGRPOAdvantageLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipGRPOAdvantageLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "GRPO advantage launch header mismatch", nil)
	}
	rewardPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	count := int(binary.LittleEndian.Uint32(args[24:]))
	rewardBytes := int(binary.LittleEndian.Uint32(args[28:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[32:]))
	if count <= 0 || rewardBytes != count*8 || outputBytes != count*8 {
		return core.E("rocm.hip.FakeLaunch", "GRPO advantage shape metadata mismatch", nil)
	}
	rewardData, rewardOffset, ok := driver.memoryForPointer(rewardPointer, rewardBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "GRPO reward buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "GRPO advantage output buffer is missing", nil)
	}
	rewards, err := hipFloat64PayloadValues(rewardData[rewardOffset : rewardOffset+rewardBytes])
	if err != nil {
		return err
	}
	advantages, err := rocmReferenceNormalizeAdvantages(rewards)
	if err != nil {
		return err
	}
	payload, err := hipFloat64Payload(advantages)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchAdamWUpdate(args []byte) error {
	if len(args) != hipAdamWUpdateLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "AdamW update launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipAdamWUpdateLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipAdamWUpdateLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "AdamW update launch header mismatch", nil)
	}
	parameterPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	momentMPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	momentVPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	gradientPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	paramCount := int(binary.LittleEndian.Uint32(args[40:]))
	tensorCount := int(binary.LittleEndian.Uint32(args[44:]))
	step := int(binary.LittleEndian.Uint32(args[48:]))
	parameterBytes := int(binary.LittleEndian.Uint32(args[52:]))
	momentBytes := int(binary.LittleEndian.Uint32(args[56:]))
	gradientBytes := int(binary.LittleEndian.Uint32(args[60:]))
	learningRate := math.Float64frombits(binary.LittleEndian.Uint64(args[64:]))
	beta1 := math.Float64frombits(binary.LittleEndian.Uint64(args[72:]))
	beta2 := math.Float64frombits(binary.LittleEndian.Uint64(args[80:]))
	eps := math.Float64frombits(binary.LittleEndian.Uint64(args[88:]))
	weightDecay := math.Float64frombits(binary.LittleEndian.Uint64(args[96:]))
	if paramCount <= 0 || tensorCount <= 0 || step <= 0 ||
		parameterBytes != paramCount*4 || momentBytes != paramCount*4 || gradientBytes != paramCount*4 ||
		learningRate <= 0 || math.IsNaN(learningRate) || math.IsInf(learningRate, 0) ||
		beta1 < 0 || beta1 >= 1 || math.IsNaN(beta1) || math.IsInf(beta1, 0) ||
		beta2 < 0 || beta2 >= 1 || math.IsNaN(beta2) || math.IsInf(beta2, 0) ||
		eps <= 0 || math.IsNaN(eps) || math.IsInf(eps, 0) ||
		weightDecay < 0 || math.IsNaN(weightDecay) || math.IsInf(weightDecay, 0) {
		return core.E("rocm.hip.FakeLaunch", "AdamW update shape metadata mismatch", nil)
	}
	parameterData, parameterOffset, ok := driver.memoryForPointer(parameterPointer, parameterBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "AdamW parameter buffer is missing", nil)
	}
	momentMData, momentMOffset, ok := driver.memoryForPointer(momentMPointer, momentBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "AdamW first moment buffer is missing", nil)
	}
	momentVData, momentVOffset, ok := driver.memoryForPointer(momentVPointer, momentBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "AdamW second moment buffer is missing", nil)
	}
	gradientData, gradientOffset, ok := driver.memoryForPointer(gradientPointer, gradientBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "AdamW gradient buffer is missing", nil)
	}
	biasCorrection1 := 1 - math.Pow(beta1, float64(step))
	biasCorrection2 := 1 - math.Pow(beta2, float64(step))
	for index := 0; index < paramCount; index++ {
		byteOffset := index * 4
		param := float64(math.Float32frombits(binary.LittleEndian.Uint32(parameterData[parameterOffset+byteOffset:])))
		momentM := float64(math.Float32frombits(binary.LittleEndian.Uint32(momentMData[momentMOffset+byteOffset:])))
		momentV := float64(math.Float32frombits(binary.LittleEndian.Uint32(momentVData[momentVOffset+byteOffset:])))
		gradient := float64(math.Float32frombits(binary.LittleEndian.Uint32(gradientData[gradientOffset+byteOffset:])))
		nextM := beta1*momentM + (1-beta1)*gradient
		nextV := beta2*momentV + (1-beta2)*gradient*gradient
		decayed := param * (1 - learningRate*weightDecay)
		next := decayed - learningRate*(nextM/biasCorrection1)/(math.Sqrt(nextV/biasCorrection2)+eps)
		if math.IsNaN(next) || math.IsInf(next, 0) {
			return core.E("rocm.hip.FakeLaunch", "AdamW update produced non-finite parameter", nil)
		}
		binary.LittleEndian.PutUint32(parameterData[parameterOffset+byteOffset:], math.Float32bits(float32(next)))
		binary.LittleEndian.PutUint32(momentMData[momentMOffset+byteOffset:], math.Float32bits(float32(nextM)))
		binary.LittleEndian.PutUint32(momentVData[momentVOffset+byteOffset:], math.Float32bits(float32(nextV)))
	}
	return nil
}

func (driver *fakeHIPDriver) launchMoERouter(args []byte) error {
	if len(args) != hipMoERouterLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MoE router launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMoERouterLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMoERouterLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MoE router launch header mismatch", nil)
	}
	logitPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	idPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	probPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	expertCount := int(binary.LittleEndian.Uint32(args[32:]))
	topK := int(binary.LittleEndian.Uint32(args[36:]))
	logitBytes := int(binary.LittleEndian.Uint32(args[40:]))
	idBytes := int(binary.LittleEndian.Uint32(args[44:]))
	probBytes := int(binary.LittleEndian.Uint32(args[48:]))
	layer := int(binary.LittleEndian.Uint32(args[52:]))
	statusPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[56:]))
	if expertCount <= 0 || topK <= 0 || topK > expertCount || logitBytes != expertCount*4 || idBytes != topK*4 || probBytes != topK*4 {
		return core.E("rocm.hip.FakeLaunch", "MoE router shape metadata mismatch", nil)
	}
	logitData, logitOffset, ok := driver.memoryForPointer(logitPointer, logitBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MoE router logits buffer is missing", nil)
	}
	idData, idOffset, ok := driver.memoryForPointer(idPointer, idBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MoE router id output buffer is missing", nil)
	}
	probData, probOffset, ok := driver.memoryForPointer(probPointer, probBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MoE router probability output buffer is missing", nil)
	}
	logits, err := hipFloat32PayloadValues(logitData[logitOffset : logitOffset+logitBytes])
	if err != nil {
		return err
	}
	routes, err := rocmReferenceRouteExperts(logits, topK, layer, nil)
	if err != nil {
		return err
	}
	for index, route := range routes {
		binary.LittleEndian.PutUint32(idData[idOffset+index*4:], uint32(int32(route.ID)))
		binary.LittleEndian.PutUint32(probData[probOffset+index*4:], math.Float32bits(route.Prob))
	}
	if statusPointer != 0 {
		status, offset, ok := driver.memoryForPointer(statusPointer, 4)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "MoE router status buffer is missing", nil)
		}
		binary.LittleEndian.PutUint32(status[offset:], hipMoERouterLaunchStatusOK)
	}
	return nil
}

func (driver *fakeHIPDriver) launchMoELazyExperts(args []byte) error {
	if len(args) != hipMoELazyLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MoE lazy expert launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMoELazyLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMoELazyLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MoE lazy expert launch header mismatch", nil)
	}
	idPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	residentPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	selected := int(binary.LittleEndian.Uint32(args[24:]))
	total := int(binary.LittleEndian.Uint32(args[28:]))
	idBytes := int(binary.LittleEndian.Uint32(args[32:]))
	residentBytes := int(binary.LittleEndian.Uint32(args[36:]))
	if selected <= 0 || total <= 0 || idBytes != selected*4 || residentBytes != total {
		return core.E("rocm.hip.FakeLaunch", "MoE lazy expert shape metadata mismatch", nil)
	}
	idData, idOffset, ok := driver.memoryForPointer(idPointer, idBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MoE lazy expert ID buffer is missing", nil)
	}
	residentData, residentOffset, ok := driver.memoryForPointer(residentPointer, residentBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MoE lazy expert output buffer is missing", nil)
	}
	routes := make([]rocmExpertRoute, selected)
	for index := range routes {
		routes[index] = rocmExpertRoute{ID: int(int32(binary.LittleEndian.Uint32(idData[idOffset+index*4:])))}
	}
	resident, err := rocmReferenceLazyExpertResidency(routes, total)
	if err != nil {
		return err
	}
	for index, value := range resident {
		if value {
			residentData[residentOffset+index] = 1
		} else {
			residentData[residentOffset+index] = 0
		}
	}
	return nil
}
func (driver *fakeHIPDriver) memoryForPointer(pointer nativeDevicePointer, size int) ([]byte, int, bool) {
	if driver.memory == nil || pointer == 0 || size < 0 {
		return nil, 0, false
	}
	if data, ok := driver.memory[pointer]; ok && len(data) >= size {
		return data, 0, true
	}
	for base, data := range driver.memory {
		if pointer < base {
			continue
		}
		offset := int(pointer - base)
		if offset >= 0 && offset+size <= len(data) {
			return data, offset, true
		}
	}
	return nil, 0, false
}
func (driver *fakeHIPDriver) launchPrefill(args []byte) error {
	if len(args) != hipPrefillLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "prefill launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipPrefillLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipPrefillLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "prefill launch header mismatch", nil)
	}
	tokenPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	tokenCount := binary.LittleEndian.Uint64(args[16:])
	tokenBytes := binary.LittleEndian.Uint64(args[24:])
	modeCode := binary.LittleEndian.Uint32(args[32:])
	blockSize := binary.LittleEndian.Uint32(args[36:])
	keyWidth := binary.LittleEndian.Uint32(args[40:])
	valueWidth := binary.LittleEndian.Uint32(args[44:])
	statusPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	statusValue := binary.LittleEndian.Uint32(args[56:])
	if tokenCount == 0 || tokenBytes != tokenCount*4 {
		return core.E("rocm.hip.FakeLaunch", "prefill token metadata mismatch", nil)
	}
	if blockSize == 0 || keyWidth == 0 || valueWidth == 0 {
		return core.E("rocm.hip.FakeLaunch", "prefill KV shape metadata is invalid", nil)
	}
	if err := rocmDeviceKVValidateModeCode(modeCode); err != nil {
		return err
	}
	if _, _, ok := driver.memoryForPointer(tokenPointer, int(tokenBytes)); !ok {
		return core.E("rocm.hip.FakeLaunch", "prefill token buffer is missing", nil)
	}
	if statusPointer != 0 {
		if statusValue == 0 {
			statusValue = hipPrefillLaunchStatusOK
		}
		status, offset, ok := driver.memoryForPointer(statusPointer, 4)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "prefill status buffer is missing", nil)
		}
		binary.LittleEndian.PutUint32(status[offset:], statusValue)
	}
	return nil
}
func (driver *fakeHIPDriver) launchDecode(args []byte) error {
	if len(args) != hipDecodeLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "decode launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipDecodeLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipDecodeLaunchArgsHeaderBytes) ||
		binary.LittleEndian.Uint32(args[8:]) != uint32(hipDecodeLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "decode launch header mismatch", nil)
	}
	position := binary.LittleEndian.Uint64(args[16:])
	kvBytes := binary.LittleEndian.Uint32(args[24:])
	if kvBytes != rocmDeviceKVLaunchDescriptorBytes {
		return core.E("rocm.hip.FakeLaunch", "decode KV launch descriptor size mismatch", nil)
	}
	kv := args[hipDecodeLaunchArgsHeaderBytes:]
	descriptorPointer := nativeDevicePointer(binary.LittleEndian.Uint64(kv[0:]))
	descriptorBytes := binary.LittleEndian.Uint64(kv[8:])
	descriptorVersion := binary.LittleEndian.Uint32(kv[16:])
	modeCode := binary.LittleEndian.Uint32(kv[20:])
	pageCount := binary.LittleEndian.Uint32(kv[28:])
	tokenCount := binary.LittleEndian.Uint64(kv[32:])
	keyWidth := binary.LittleEndian.Uint32(kv[40:])
	valueWidth := binary.LittleEndian.Uint32(kv[44:])
	statusPointer := nativeDevicePointer(binary.LittleEndian.Uint64(kv[48:]))
	statusValue := binary.LittleEndian.Uint32(kv[56:])
	if descriptorVersion != rocmDeviceKVDescriptorVersion {
		return core.E("rocm.hip.FakeLaunch", "decode descriptor version mismatch", nil)
	}
	if err := rocmDeviceKVValidateModeCode(modeCode); err != nil {
		return err
	}
	if position != tokenCount || tokenCount == 0 || pageCount == 0 || keyWidth == 0 || valueWidth == 0 {
		return core.E("rocm.hip.FakeLaunch", "decode KV metadata mismatch", nil)
	}
	descriptor, offset, ok := driver.memoryForPointer(descriptorPointer, int(descriptorBytes))
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "decode descriptor table is missing", nil)
	}
	table := descriptor[offset : offset+int(descriptorBytes)]
	if len(table) < rocmDeviceKVDescriptorHeaderBytes ||
		binary.LittleEndian.Uint32(table[0:]) != rocmDeviceKVDescriptorVersion ||
		binary.LittleEndian.Uint32(table[12:]) != modeCode ||
		binary.LittleEndian.Uint32(table[16:]) != pageCount ||
		binary.LittleEndian.Uint64(table[24:]) != tokenCount {
		return core.E("rocm.hip.FakeLaunch", "decode descriptor table header mismatch", nil)
	}
	if statusPointer != 0 {
		if statusValue == 0 {
			statusValue = hipDecodeLaunchStatusOK
		}
		status, offset, ok := driver.memoryForPointer(statusPointer, 4)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "decode status buffer is missing", nil)
		}
		binary.LittleEndian.PutUint32(status[offset:], statusValue)
	}
	return nil
}
func (driver *fakeHIPDriver) launchProjection(args []byte) error {
	if len(args) != hipProjectionLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "projection launch args size mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	inputCount := int(binary.LittleEndian.Uint32(args[16:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[20:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	weightBytes := int(binary.LittleEndian.Uint64(args[32:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	biasBytes := int(binary.LittleEndian.Uint64(args[48:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[56:]))
	outputBytes := int(binary.LittleEndian.Uint64(args[64:]))
	rows := int(binary.LittleEndian.Uint32(args[72:]))
	cols := int(binary.LittleEndian.Uint32(args[76:]))
	encoding := binary.LittleEndian.Uint32(args[80:])
	flags := binary.LittleEndian.Uint32(args[84:])
	q8Scale := math.Float32frombits(binary.LittleEndian.Uint32(args[88:]))
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "projection input buffer is missing", nil)
	}
	weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "projection weight buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "projection output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	var bias []float32
	if flags&hipProjectionLaunchFlagBias != 0 {
		biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "projection bias buffer is missing", nil)
		}
		bias, err = hipFloat32PayloadValues(biasData[biasOffset : biasOffset+biasBytes])
		if err != nil {
			return err
		}
	}
	var output []float32
	switch encoding {
	case hipProjectionWeightEncodingFP16:
		weights := make([]uint16, weightBytes/2)
		for index := range weights {
			weights[index] = binary.LittleEndian.Uint16(weightData[weightOffset+index*2:])
		}
		output, err = hipReferenceFP16Projection(input[:inputCount], weights, rows, cols, bias)
	case hipProjectionWeightEncodingBF16:
		weights := make([]uint16, weightBytes/2)
		for index := range weights {
			weights[index] = binary.LittleEndian.Uint16(weightData[weightOffset+index*2:])
		}
		output, err = hipReferenceBF16Projection(input[:inputCount], weights, rows, cols, bias)
	case hipProjectionWeightEncodingQ8:
		weights := make([]int8, weightBytes)
		for index := range weights {
			weights[index] = int8(weightData[weightOffset+index])
		}
		output, err = hipReferenceQ8Projection(input[:inputCount], weights, q8Scale, rows, cols, bias)
	case hipProjectionWeightEncodingF32:
		weights, decodeErr := hipFloat32PayloadValues(weightData[weightOffset : weightOffset+weightBytes])
		if decodeErr != nil {
			return decodeErr
		}
		output, err = hipReferenceF32Projection(input[:inputCount], weights, rows, cols, bias)
	default:
		err = core.E("rocm.hip.FakeLaunch", "unsupported projection encoding", nil)
	}
	if err != nil {
		return err
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchProjectionBatch(args []byte) error {
	if len(args) != hipProjectionBatchLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "projection batch launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipProjectionBatchLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipProjectionBatchLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "projection batch launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	weightBytes := int(binary.LittleEndian.Uint64(args[24:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	biasBytes := int(binary.LittleEndian.Uint64(args[40:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	outputBytes := int(binary.LittleEndian.Uint64(args[56:]))
	rows := int(binary.LittleEndian.Uint32(args[64:]))
	cols := int(binary.LittleEndian.Uint32(args[68:]))
	batch := int(binary.LittleEndian.Uint32(args[72:]))
	encoding := binary.LittleEndian.Uint32(args[76:])
	flags := binary.LittleEndian.Uint32(args[80:])
	q8Scale := math.Float32frombits(binary.LittleEndian.Uint32(args[84:]))
	inputBytes := int(binary.LittleEndian.Uint64(args[88:]))
	if rows <= 0 || cols <= 0 || batch <= 0 || inputBytes != batch*cols*4 || outputBytes != batch*rows*4 {
		return core.E("rocm.hip.FakeLaunch", "projection batch shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "projection batch input buffer is missing", nil)
	}
	weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "projection batch weight buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "projection batch output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	var bias []float32
	if flags&hipProjectionLaunchFlagBias != 0 {
		biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "projection batch bias buffer is missing", nil)
		}
		bias, err = hipFloat32PayloadValues(biasData[biasOffset : biasOffset+biasBytes])
		if err != nil {
			return err
		}
	}
	output := make([]float32, 0, batch*rows)
	for item := 0; item < batch; item++ {
		start := item * cols
		end := start + cols
		var projected []float32
		switch encoding {
		case hipProjectionWeightEncodingFP16:
			weights := make([]uint16, weightBytes/2)
			for index := range weights {
				weights[index] = binary.LittleEndian.Uint16(weightData[weightOffset+index*2:])
			}
			projected, err = hipReferenceFP16Projection(input[start:end], weights, rows, cols, bias)
		case hipProjectionWeightEncodingBF16:
			weights := make([]uint16, weightBytes/2)
			for index := range weights {
				weights[index] = binary.LittleEndian.Uint16(weightData[weightOffset+index*2:])
			}
			projected, err = hipReferenceBF16Projection(input[start:end], weights, rows, cols, bias)
		case hipProjectionWeightEncodingQ8:
			weights := make([]int8, weightBytes)
			for index := range weights {
				weights[index] = int8(weightData[weightOffset+index])
			}
			projected, err = hipReferenceQ8Projection(input[start:end], weights, q8Scale, rows, cols, bias)
		case hipProjectionWeightEncodingF32:
			weights, decodeErr := hipFloat32PayloadValues(weightData[weightOffset : weightOffset+weightBytes])
			if decodeErr != nil {
				return decodeErr
			}
			projected, err = hipReferenceF32Projection(input[start:end], weights, rows, cols, bias)
		default:
			err = core.E("rocm.hip.FakeLaunch", "unsupported projection batch encoding", nil)
		}
		if err != nil {
			return err
		}
		output = append(output, projected...)
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchMLXQ4Projection(args []byte) error {
	if len(args) != hipMLXQ4ProjectionLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMLXQ4ProjectionLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMLXQ4ProjectionLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	scalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	rows := int(binary.LittleEndian.Uint32(args[48:]))
	cols := int(binary.LittleEndian.Uint32(args[52:]))
	groupSize := int(binary.LittleEndian.Uint32(args[56:]))
	bits := int(binary.LittleEndian.Uint32(args[60:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[64:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[68:]))
	scaleBytes := int(binary.LittleEndian.Uint32(args[72:]))
	biasBytes := int(binary.LittleEndian.Uint32(args[76:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[80:]))
	if !hipMLXAffineSupportedBits(bits) ||
		validateHIPMLXAffineProjectionShape(cols, weightBytes/4, scaleBytes/2, biasBytes/2, rows, cols, groupSize, bits) != nil ||
		inputBytes != cols*4 ||
		outputBytes != rows*4 {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection input buffer is missing", nil)
	}
	weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection packed weight buffer is missing", nil)
	}
	scaleData, scaleOffset, ok := driver.memoryForPointer(scalePointer, scaleBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection scale buffer is missing", nil)
	}
	biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection bias buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	weights := make([]uint32, weightBytes/4)
	for index := range weights {
		weights[index] = binary.LittleEndian.Uint32(weightData[weightOffset+index*4:])
	}
	scales := make([]uint16, scaleBytes/2)
	for index := range scales {
		scales[index] = binary.LittleEndian.Uint16(scaleData[scaleOffset+index*2:])
	}
	biases := make([]uint16, biasBytes/2)
	for index := range biases {
		biases[index] = binary.LittleEndian.Uint16(biasData[biasOffset+index*2:])
	}
	output, err := hipReferenceMLXAffineProjection(input, weights, scales, biases, rows, cols, groupSize, bits)
	if err != nil {
		return err
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchMLXQ4ProjectionBatch(args []byte) error {
	if len(args) != hipMLXQ4ProjectionBatchLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection batch launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMLXQ4ProjectionBatchLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMLXQ4ProjectionBatchLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection batch launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	scalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	rows := int(binary.LittleEndian.Uint32(args[48:]))
	cols := int(binary.LittleEndian.Uint32(args[52:]))
	batch := int(binary.LittleEndian.Uint32(args[56:]))
	groupSize := int(binary.LittleEndian.Uint32(args[60:]))
	bits := int(binary.LittleEndian.Uint32(args[64:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[68:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[72:]))
	scaleBytes := int(binary.LittleEndian.Uint32(args[76:]))
	biasBytes := int(binary.LittleEndian.Uint32(args[80:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[84:]))
	if !hipMLXAffineSupportedBits(bits) ||
		batch <= 0 ||
		validateHIPMLXAffineProjectionShape(cols, weightBytes/4, scaleBytes/2, biasBytes/2, rows, cols, groupSize, bits) != nil ||
		inputBytes != batch*cols*4 ||
		outputBytes != batch*rows*4 {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection batch shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection batch input buffer is missing", nil)
	}
	weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection batch packed weight buffer is missing", nil)
	}
	scaleData, scaleOffset, ok := driver.memoryForPointer(scalePointer, scaleBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection batch scale buffer is missing", nil)
	}
	biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection batch bias buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection batch output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	weights := make([]uint32, weightBytes/4)
	for index := range weights {
		weights[index] = binary.LittleEndian.Uint32(weightData[weightOffset+index*4:])
	}
	scales := make([]uint16, scaleBytes/2)
	for index := range scales {
		scales[index] = binary.LittleEndian.Uint16(scaleData[scaleOffset+index*2:])
	}
	biases := make([]uint16, biasBytes/2)
	for index := range biases {
		biases[index] = binary.LittleEndian.Uint16(biasData[biasOffset+index*2:])
	}
	output := make([]float32, 0, batch*rows)
	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		start := batchIndex * cols
		projected, err := hipReferenceMLXAffineProjection(input[start:start+cols], weights, scales, biases, rows, cols, groupSize, bits)
		if err != nil {
			return err
		}
		output = append(output, projected...)
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchMLXQ4TripleProjection(args []byte) error {
	if len(args) != hipMLXQ4TripleProjLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 triple projection launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMLXQ4TripleProjLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMLXQ4TripleProjLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 triple projection launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	weightPointers := [3]nativeDevicePointer{
		nativeDevicePointer(binary.LittleEndian.Uint64(args[24:])),
		nativeDevicePointer(binary.LittleEndian.Uint64(args[48:])),
		nativeDevicePointer(binary.LittleEndian.Uint64(args[72:])),
	}
	scalePointers := [3]nativeDevicePointer{
		nativeDevicePointer(binary.LittleEndian.Uint64(args[32:])),
		nativeDevicePointer(binary.LittleEndian.Uint64(args[56:])),
		nativeDevicePointer(binary.LittleEndian.Uint64(args[80:])),
	}
	biasPointers := [3]nativeDevicePointer{
		nativeDevicePointer(binary.LittleEndian.Uint64(args[40:])),
		nativeDevicePointer(binary.LittleEndian.Uint64(args[64:])),
		nativeDevicePointer(binary.LittleEndian.Uint64(args[88:])),
	}
	rows := [3]int{
		int(binary.LittleEndian.Uint32(args[96:])),
		int(binary.LittleEndian.Uint32(args[100:])),
		int(binary.LittleEndian.Uint32(args[104:])),
	}
	cols := int(binary.LittleEndian.Uint32(args[108:]))
	groupSize := int(binary.LittleEndian.Uint32(args[112:]))
	bits := int(binary.LittleEndian.Uint32(args[116:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[120:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[124:]))
	weightBytes := [3]int{
		int(binary.LittleEndian.Uint32(args[128:])),
		int(binary.LittleEndian.Uint32(args[140:])),
		int(binary.LittleEndian.Uint32(args[152:])),
	}
	scaleBytes := [3]int{
		int(binary.LittleEndian.Uint32(args[132:])),
		int(binary.LittleEndian.Uint32(args[144:])),
		int(binary.LittleEndian.Uint32(args[156:])),
	}
	biasBytes := [3]int{
		int(binary.LittleEndian.Uint32(args[136:])),
		int(binary.LittleEndian.Uint32(args[148:])),
		int(binary.LittleEndian.Uint32(args[160:])),
	}
	totalRows := rows[0] + rows[1] + rows[2]
	if !hipMLXAffineSupportedBits(bits) || inputBytes != cols*4 || outputBytes != totalRows*4 {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 triple projection shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 triple projection input buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 triple projection output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	combined := make([]float32, 0, totalRows)
	for index := 0; index < 3; index++ {
		if rows[index] == 0 {
			if weightBytes[index] != 0 || scaleBytes[index] != 0 || biasBytes[index] != 0 {
				return core.E("rocm.hip.FakeLaunch", "MLX q4 triple projection zero-row byte metadata mismatch", nil)
			}
			continue
		}
		if validateHIPMLXAffineProjectionShape(cols, weightBytes[index]/4, scaleBytes[index]/2, biasBytes[index]/2, rows[index], cols, groupSize, bits) != nil {
			return core.E("rocm.hip.FakeLaunch", "MLX q4 triple projection shape metadata mismatch", nil)
		}
		weightData, weightOffset, ok := driver.memoryForPointer(weightPointers[index], weightBytes[index])
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "MLX q4 triple projection packed weight buffer is missing", nil)
		}
		scaleData, scaleOffset, ok := driver.memoryForPointer(scalePointers[index], scaleBytes[index])
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "MLX q4 triple projection scale buffer is missing", nil)
		}
		biasData, biasOffset, ok := driver.memoryForPointer(biasPointers[index], biasBytes[index])
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "MLX q4 triple projection bias buffer is missing", nil)
		}
		weights := make([]uint32, weightBytes[index]/4)
		for weightIndex := range weights {
			weights[weightIndex] = binary.LittleEndian.Uint32(weightData[weightOffset+weightIndex*4:])
		}
		scales := make([]uint16, scaleBytes[index]/2)
		for scaleIndex := range scales {
			scales[scaleIndex] = binary.LittleEndian.Uint16(scaleData[scaleOffset+scaleIndex*2:])
		}
		biases := make([]uint16, biasBytes[index]/2)
		for biasIndex := range biases {
			biases[biasIndex] = binary.LittleEndian.Uint16(biasData[biasOffset+biasIndex*2:])
		}
		output, err := hipReferenceMLXAffineProjection(input, weights, scales, biases, rows[index], cols, groupSize, bits)
		if err != nil {
			return err
		}
		combined = append(combined, output...)
	}
	payload, err := hipFloat32Payload(combined)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchMLXQ4GELUTanhMultiply(args []byte) error {
	if len(args) != hipMLXQ4GELUTanhMulLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMLXQ4GELUTanhMulLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMLXQ4GELUTanhMulLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	gateWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	gateScalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	gateBiasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	upWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	upScalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	upBiasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[56:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[64:]))
	rows := int(binary.LittleEndian.Uint32(args[72:]))
	cols := int(binary.LittleEndian.Uint32(args[76:]))
	groupSize := int(binary.LittleEndian.Uint32(args[80:]))
	bits := int(binary.LittleEndian.Uint32(args[84:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[88:]))
	gateWeightBytes := int(binary.LittleEndian.Uint32(args[92:]))
	gateScaleBytes := int(binary.LittleEndian.Uint32(args[96:]))
	gateBiasBytes := int(binary.LittleEndian.Uint32(args[100:]))
	upWeightBytes := int(binary.LittleEndian.Uint32(args[104:]))
	upScaleBytes := int(binary.LittleEndian.Uint32(args[108:]))
	upBiasBytes := int(binary.LittleEndian.Uint32(args[112:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[116:]))
	if !hipMLXAffineSupportedBits(bits) ||
		validateHIPMLXAffineProjectionShape(cols, gateWeightBytes/4, gateScaleBytes/2, gateBiasBytes/2, rows, cols, groupSize, bits) != nil ||
		validateHIPMLXAffineProjectionShape(cols, upWeightBytes/4, upScaleBytes/2, upBiasBytes/2, rows, cols, groupSize, bits) != nil ||
		inputBytes != cols*4 ||
		outputBytes != rows*4 {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply input buffer is missing", nil)
	}
	gateWeightData, gateWeightOffset, ok := driver.memoryForPointer(gateWeightPointer, gateWeightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply gate packed weight buffer is missing", nil)
	}
	gateScaleData, gateScaleOffset, ok := driver.memoryForPointer(gateScalePointer, gateScaleBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply gate scale buffer is missing", nil)
	}
	gateBiasData, gateBiasOffset, ok := driver.memoryForPointer(gateBiasPointer, gateBiasBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply gate bias buffer is missing", nil)
	}
	upWeightData, upWeightOffset, ok := driver.memoryForPointer(upWeightPointer, upWeightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply up packed weight buffer is missing", nil)
	}
	upScaleData, upScaleOffset, ok := driver.memoryForPointer(upScalePointer, upScaleBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply up scale buffer is missing", nil)
	}
	upBiasData, upBiasOffset, ok := driver.memoryForPointer(upBiasPointer, upBiasBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply up bias buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	gateWeights := make([]uint32, gateWeightBytes/4)
	for index := range gateWeights {
		gateWeights[index] = binary.LittleEndian.Uint32(gateWeightData[gateWeightOffset+index*4:])
	}
	gateScales := make([]uint16, gateScaleBytes/2)
	for index := range gateScales {
		gateScales[index] = binary.LittleEndian.Uint16(gateScaleData[gateScaleOffset+index*2:])
	}
	gateBiases := make([]uint16, gateBiasBytes/2)
	for index := range gateBiases {
		gateBiases[index] = binary.LittleEndian.Uint16(gateBiasData[gateBiasOffset+index*2:])
	}
	upWeights := make([]uint32, upWeightBytes/4)
	for index := range upWeights {
		upWeights[index] = binary.LittleEndian.Uint32(upWeightData[upWeightOffset+index*4:])
	}
	upScales := make([]uint16, upScaleBytes/2)
	for index := range upScales {
		upScales[index] = binary.LittleEndian.Uint16(upScaleData[upScaleOffset+index*2:])
	}
	upBiases := make([]uint16, upBiasBytes/2)
	for index := range upBiases {
		upBiases[index] = binary.LittleEndian.Uint16(upBiasData[upBiasOffset+index*2:])
	}
	gate, err := hipReferenceMLXAffineProjection(input, gateWeights, gateScales, gateBiases, rows, cols, groupSize, bits)
	if err != nil {
		return err
	}
	up, err := hipReferenceMLXAffineProjection(input, upWeights, upScales, upBiases, rows, cols, groupSize, bits)
	if err != nil {
		return err
	}
	out := make([]float32, rows)
	const sqrt2OverPi = 0.7978845608028654
	const coeff = 0.044715
	for index := range out {
		value := float64(gate[index])
		gelu := 0.5 * value * (1 + math.Tanh(sqrt2OverPi*(value+coeff*value*value*value)))
		out[index] = float32(gelu) * up[index]
	}
	payload, err := hipFloat32Payload(out)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchMLXQ4GELUTanhMLPPersistent(args []byte) error {
	if len(args) != hipMLXQ4GELUTanhMLPPersistentLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh MLP persistent launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMLXQ4GELUTanhMLPPersistentLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMLXQ4GELUTanhMLPPersistentLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh MLP persistent launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	gateWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	gateScalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	gateBiasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	upWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	upScalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	upBiasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[56:]))
	downWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[64:]))
	downScalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[72:]))
	downBiasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[80:]))
	activationPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[88:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[96:]))
	barrierPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[104:]))
	rows := int(binary.LittleEndian.Uint32(args[112:]))
	cols := int(binary.LittleEndian.Uint32(args[116:]))
	downRows := int(binary.LittleEndian.Uint32(args[120:]))
	groupSize := int(binary.LittleEndian.Uint32(args[124:]))
	bits := int(binary.LittleEndian.Uint32(args[128:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[132:]))
	gateWeightBytes := int(binary.LittleEndian.Uint32(args[136:]))
	gateScaleBytes := int(binary.LittleEndian.Uint32(args[140:]))
	gateBiasBytes := int(binary.LittleEndian.Uint32(args[144:]))
	upWeightBytes := int(binary.LittleEndian.Uint32(args[148:]))
	upScaleBytes := int(binary.LittleEndian.Uint32(args[152:]))
	upBiasBytes := int(binary.LittleEndian.Uint32(args[156:]))
	downWeightBytes := int(binary.LittleEndian.Uint32(args[160:]))
	downScaleBytes := int(binary.LittleEndian.Uint32(args[164:]))
	downBiasBytes := int(binary.LittleEndian.Uint32(args[168:]))
	activationBytes := int(binary.LittleEndian.Uint32(args[172:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[176:]))
	barrierBytes := int(binary.LittleEndian.Uint32(args[180:]))
	if !hipMLXAffineSupportedBits(bits) ||
		cols != 1536 ||
		downRows != 1536 ||
		groupSize != 32 ||
		bits != 4 ||
		validateHIPMLXAffineProjectionShape(cols, gateWeightBytes/4, gateScaleBytes/2, gateBiasBytes/2, rows, cols, groupSize, bits) != nil ||
		validateHIPMLXAffineProjectionShape(cols, upWeightBytes/4, upScaleBytes/2, upBiasBytes/2, rows, cols, groupSize, bits) != nil ||
		validateHIPMLXAffineProjectionShape(rows, downWeightBytes/4, downScaleBytes/2, downBiasBytes/2, downRows, rows, groupSize, bits) != nil ||
		inputBytes != cols*4 ||
		activationBytes != rows*4 ||
		outputBytes != downRows*4 ||
		barrierBytes != 8 {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh MLP persistent shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh MLP persistent input buffer is missing", nil)
	}
	activationData, activationOffset, ok := driver.memoryForPointer(activationPointer, activationBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh MLP persistent activation buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh MLP persistent output buffer is missing", nil)
	}
	if _, _, ok := driver.memoryForPointer(barrierPointer, barrierBytes); !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh MLP persistent barrier buffer is missing", nil)
	}
	readU32 := func(pointer nativeDevicePointer, byteCount int, label string) ([]uint32, error) {
		data, offset, ok := driver.memoryForPointer(pointer, byteCount)
		if !ok {
			return nil, core.E("rocm.hip.FakeLaunch", label+" buffer is missing", nil)
		}
		values := make([]uint32, byteCount/4)
		for index := range values {
			values[index] = binary.LittleEndian.Uint32(data[offset+index*4:])
		}
		return values, nil
	}
	readU16 := func(pointer nativeDevicePointer, byteCount int, label string) ([]uint16, error) {
		data, offset, ok := driver.memoryForPointer(pointer, byteCount)
		if !ok {
			return nil, core.E("rocm.hip.FakeLaunch", label+" buffer is missing", nil)
		}
		values := make([]uint16, byteCount/2)
		for index := range values {
			values[index] = binary.LittleEndian.Uint16(data[offset+index*2:])
		}
		return values, nil
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	gateWeights, err := readU32(gateWeightPointer, gateWeightBytes, "MLX q4 GELU tanh MLP persistent gate packed weight")
	if err != nil {
		return err
	}
	gateScales, err := readU16(gateScalePointer, gateScaleBytes, "MLX q4 GELU tanh MLP persistent gate scale")
	if err != nil {
		return err
	}
	gateBiases, err := readU16(gateBiasPointer, gateBiasBytes, "MLX q4 GELU tanh MLP persistent gate bias")
	if err != nil {
		return err
	}
	upWeights, err := readU32(upWeightPointer, upWeightBytes, "MLX q4 GELU tanh MLP persistent up packed weight")
	if err != nil {
		return err
	}
	upScales, err := readU16(upScalePointer, upScaleBytes, "MLX q4 GELU tanh MLP persistent up scale")
	if err != nil {
		return err
	}
	upBiases, err := readU16(upBiasPointer, upBiasBytes, "MLX q4 GELU tanh MLP persistent up bias")
	if err != nil {
		return err
	}
	downWeights, err := readU32(downWeightPointer, downWeightBytes, "MLX q4 GELU tanh MLP persistent down packed weight")
	if err != nil {
		return err
	}
	downScales, err := readU16(downScalePointer, downScaleBytes, "MLX q4 GELU tanh MLP persistent down scale")
	if err != nil {
		return err
	}
	downBiases, err := readU16(downBiasPointer, downBiasBytes, "MLX q4 GELU tanh MLP persistent down bias")
	if err != nil {
		return err
	}
	gate, err := hipReferenceMLXAffineProjection(input, gateWeights, gateScales, gateBiases, rows, cols, groupSize, bits)
	if err != nil {
		return err
	}
	up, err := hipReferenceMLXAffineProjection(input, upWeights, upScales, upBiases, rows, cols, groupSize, bits)
	if err != nil {
		return err
	}
	activation := make([]float32, rows)
	const sqrt2OverPi = 0.7978845608028654
	const coeff = 0.044715
	for index := range activation {
		value := float64(gate[index])
		gelu := 0.5 * value * (1 + math.Tanh(sqrt2OverPi*(value+coeff*value*value*value)))
		activation[index] = float32(gelu) * up[index]
	}
	activationPayload, err := hipFloat32Payload(activation)
	if err != nil {
		return err
	}
	copy(activationData[activationOffset:activationOffset+activationBytes], activationPayload)
	output, err := hipReferenceMLXAffineProjection(activation, downWeights, downScales, downBiases, downRows, rows, groupSize, bits)
	if err != nil {
		return err
	}
	outputPayload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], outputPayload)
	return nil
}

func (driver *fakeHIPDriver) launchMLXQ4GELUTanhMultiplyBatch(args []byte) error {
	if len(args) != hipMLXQ4GELUTanhMulBatchLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply batch launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMLXQ4GELUTanhMulBatchLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMLXQ4GELUTanhMulBatchLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply batch launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	gateWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	gateScalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	gateBiasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	upWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	upScalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	upBiasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[56:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[64:]))
	rows := int(binary.LittleEndian.Uint32(args[72:]))
	cols := int(binary.LittleEndian.Uint32(args[76:]))
	groupSize := int(binary.LittleEndian.Uint32(args[80:]))
	bits := int(binary.LittleEndian.Uint32(args[84:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[88:]))
	gateWeightBytes := int(binary.LittleEndian.Uint32(args[92:]))
	gateScaleBytes := int(binary.LittleEndian.Uint32(args[96:]))
	gateBiasBytes := int(binary.LittleEndian.Uint32(args[100:]))
	upWeightBytes := int(binary.LittleEndian.Uint32(args[104:]))
	upScaleBytes := int(binary.LittleEndian.Uint32(args[108:]))
	upBiasBytes := int(binary.LittleEndian.Uint32(args[112:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[116:]))
	batch := int(binary.LittleEndian.Uint32(args[120:]))
	if !hipMLXAffineSupportedBits(bits) ||
		batch <= 0 ||
		validateHIPMLXAffineProjectionShape(cols, gateWeightBytes/4, gateScaleBytes/2, gateBiasBytes/2, rows, cols, groupSize, bits) != nil ||
		validateHIPMLXAffineProjectionShape(cols, upWeightBytes/4, upScaleBytes/2, upBiasBytes/2, rows, cols, groupSize, bits) != nil ||
		inputBytes != batch*cols*4 ||
		outputBytes != batch*rows*4 {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply batch shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply batch input buffer is missing", nil)
	}
	gateWeightData, gateWeightOffset, ok := driver.memoryForPointer(gateWeightPointer, gateWeightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply batch gate packed weight buffer is missing", nil)
	}
	gateScaleData, gateScaleOffset, ok := driver.memoryForPointer(gateScalePointer, gateScaleBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply batch gate scale buffer is missing", nil)
	}
	gateBiasData, gateBiasOffset, ok := driver.memoryForPointer(gateBiasPointer, gateBiasBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply batch gate bias buffer is missing", nil)
	}
	upWeightData, upWeightOffset, ok := driver.memoryForPointer(upWeightPointer, upWeightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply batch up packed weight buffer is missing", nil)
	}
	upScaleData, upScaleOffset, ok := driver.memoryForPointer(upScalePointer, upScaleBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply batch up scale buffer is missing", nil)
	}
	upBiasData, upBiasOffset, ok := driver.memoryForPointer(upBiasPointer, upBiasBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply batch up bias buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh multiply batch output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	gateWeights := make([]uint32, gateWeightBytes/4)
	for index := range gateWeights {
		gateWeights[index] = binary.LittleEndian.Uint32(gateWeightData[gateWeightOffset+index*4:])
	}
	gateScales := make([]uint16, gateScaleBytes/2)
	for index := range gateScales {
		gateScales[index] = binary.LittleEndian.Uint16(gateScaleData[gateScaleOffset+index*2:])
	}
	gateBiases := make([]uint16, gateBiasBytes/2)
	for index := range gateBiases {
		gateBiases[index] = binary.LittleEndian.Uint16(gateBiasData[gateBiasOffset+index*2:])
	}
	upWeights := make([]uint32, upWeightBytes/4)
	for index := range upWeights {
		upWeights[index] = binary.LittleEndian.Uint32(upWeightData[upWeightOffset+index*4:])
	}
	upScales := make([]uint16, upScaleBytes/2)
	for index := range upScales {
		upScales[index] = binary.LittleEndian.Uint16(upScaleData[upScaleOffset+index*2:])
	}
	upBiases := make([]uint16, upBiasBytes/2)
	for index := range upBiases {
		upBiases[index] = binary.LittleEndian.Uint16(upBiasData[upBiasOffset+index*2:])
	}
	out := make([]float32, 0, batch*rows)
	const sqrt2OverPi = 0.7978845608028654
	const coeff = 0.044715
	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		start := batchIndex * cols
		gate, err := hipReferenceMLXAffineProjection(input[start:start+cols], gateWeights, gateScales, gateBiases, rows, cols, groupSize, bits)
		if err != nil {
			return err
		}
		up, err := hipReferenceMLXAffineProjection(input[start:start+cols], upWeights, upScales, upBiases, rows, cols, groupSize, bits)
		if err != nil {
			return err
		}
		for index := range gate {
			value := float64(gate[index])
			gelu := 0.5 * value * (1 + math.Tanh(sqrt2OverPi*(value+coeff*value*value*value)))
			out = append(out, float32(gelu)*up[index])
		}
	}
	payload, err := hipFloat32Payload(out)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchMLXQ4GELUTanhProjection(args []byte) error {
	if len(args) != hipMLXQ4GELUTanhProjLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMLXQ4GELUTanhProjLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMLXQ4GELUTanhProjLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	scalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	multiplierPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	rows := int(binary.LittleEndian.Uint32(args[56:]))
	cols := int(binary.LittleEndian.Uint32(args[60:]))
	groupSize := int(binary.LittleEndian.Uint32(args[64:]))
	bits := int(binary.LittleEndian.Uint32(args[68:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[72:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[76:]))
	scaleBytes := int(binary.LittleEndian.Uint32(args[80:]))
	biasBytes := int(binary.LittleEndian.Uint32(args[84:]))
	multiplierBytes := int(binary.LittleEndian.Uint32(args[88:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[92:]))
	if !hipMLXAffineSupportedBits(bits) ||
		validateHIPMLXAffineProjectionShape(cols, weightBytes/4, scaleBytes/2, biasBytes/2, rows, cols, groupSize, bits) != nil ||
		inputBytes != cols*4 ||
		multiplierBytes != rows*4 ||
		outputBytes != rows*4 {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection input buffer is missing", nil)
	}
	weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection packed weight buffer is missing", nil)
	}
	scaleData, scaleOffset, ok := driver.memoryForPointer(scalePointer, scaleBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection scale buffer is missing", nil)
	}
	biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection bias buffer is missing", nil)
	}
	multiplierData, multiplierOffset, ok := driver.memoryForPointer(multiplierPointer, multiplierBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection multiplier buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	weights := make([]uint32, weightBytes/4)
	for index := range weights {
		weights[index] = binary.LittleEndian.Uint32(weightData[weightOffset+index*4:])
	}
	scales := make([]uint16, scaleBytes/2)
	for index := range scales {
		scales[index] = binary.LittleEndian.Uint16(scaleData[scaleOffset+index*2:])
	}
	biases := make([]uint16, biasBytes/2)
	for index := range biases {
		biases[index] = binary.LittleEndian.Uint16(biasData[biasOffset+index*2:])
	}
	multiplier, err := hipFloat32PayloadValues(multiplierData[multiplierOffset : multiplierOffset+multiplierBytes])
	if err != nil {
		return err
	}
	projected, err := hipReferenceMLXAffineProjection(input, weights, scales, biases, rows, cols, groupSize, bits)
	if err != nil {
		return err
	}
	out := make([]float32, rows)
	const sqrt2OverPi = 0.7978845608028654
	const coeff = 0.044715
	for index := range out {
		value := float64(projected[index])
		gelu := 0.5 * value * (1 + math.Tanh(sqrt2OverPi*(value+coeff*value*value*value)))
		out[index] = float32(gelu) * multiplier[index]
	}
	payload, err := hipFloat32Payload(out)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchMLXQ4GELUTanhProjectionBatch(args []byte) error {
	if len(args) != hipMLXQ4GELUTanhProjBatchLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection batch launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMLXQ4GELUTanhProjBatchLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMLXQ4GELUTanhProjBatchLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection batch launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	scalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	multiplierPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	rows := int(binary.LittleEndian.Uint32(args[56:]))
	cols := int(binary.LittleEndian.Uint32(args[60:]))
	batch := int(binary.LittleEndian.Uint32(args[64:]))
	groupSize := int(binary.LittleEndian.Uint32(args[68:]))
	bits := int(binary.LittleEndian.Uint32(args[72:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[76:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[80:]))
	scaleBytes := int(binary.LittleEndian.Uint32(args[84:]))
	biasBytes := int(binary.LittleEndian.Uint32(args[88:]))
	multiplierBytes := int(binary.LittleEndian.Uint32(args[92:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[96:]))
	if !hipMLXAffineSupportedBits(bits) ||
		validateHIPMLXAffineProjectionShape(cols, weightBytes/4, scaleBytes/2, biasBytes/2, rows, cols, groupSize, bits) != nil ||
		batch <= 0 ||
		inputBytes != batch*cols*4 ||
		multiplierBytes != batch*rows*4 ||
		outputBytes != batch*rows*4 {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection batch shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection batch input buffer is missing", nil)
	}
	weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection batch packed weight buffer is missing", nil)
	}
	scaleData, scaleOffset, ok := driver.memoryForPointer(scalePointer, scaleBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection batch scale buffer is missing", nil)
	}
	biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection batch bias buffer is missing", nil)
	}
	multiplierData, multiplierOffset, ok := driver.memoryForPointer(multiplierPointer, multiplierBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection batch multiplier buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 GELU tanh projection batch output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	weights := make([]uint32, weightBytes/4)
	for index := range weights {
		weights[index] = binary.LittleEndian.Uint32(weightData[weightOffset+index*4:])
	}
	scales := make([]uint16, scaleBytes/2)
	for index := range scales {
		scales[index] = binary.LittleEndian.Uint16(scaleData[scaleOffset+index*2:])
	}
	biases := make([]uint16, biasBytes/2)
	for index := range biases {
		biases[index] = binary.LittleEndian.Uint16(biasData[biasOffset+index*2:])
	}
	multiplier, err := hipFloat32PayloadValues(multiplierData[multiplierOffset : multiplierOffset+multiplierBytes])
	if err != nil {
		return err
	}
	out := make([]float32, 0, batch*rows)
	const sqrt2OverPi = 0.7978845608028654
	const coeff = 0.044715
	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		inputStart := batchIndex * cols
		projected, err := hipReferenceMLXAffineProjection(input[inputStart:inputStart+cols], weights, scales, biases, rows, cols, groupSize, bits)
		if err != nil {
			return err
		}
		multiplierStart := batchIndex * rows
		for index := range projected {
			value := float64(projected[index])
			gelu := 0.5 * value * (1 + math.Tanh(sqrt2OverPi*(value+coeff*value*value*value)))
			out = append(out, float32(gelu)*multiplier[multiplierStart+index])
		}
	}
	payload, err := hipFloat32Payload(out)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchRMSNormResidualAddGELUTanhProjection(args []byte) error {
	if len(args) != hipRMSResidualAddGELUTanhProjLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "rms residual-add GELU tanh projection launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipRMSResidualAddGELUTanhProjLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipRMSResidualAddGELUTanhProjLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "rms residual-add GELU tanh projection launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	rmsWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	residualPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	scalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	multiplierPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[56:]))
	residualOutputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[64:]))
	activationOutputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[72:]))
	rows := int(binary.LittleEndian.Uint32(args[80:]))
	cols := int(binary.LittleEndian.Uint32(args[84:]))
	groupSize := int(binary.LittleEndian.Uint32(args[88:]))
	bits := int(binary.LittleEndian.Uint32(args[92:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[96:]))
	rmsWeightBytes := int(binary.LittleEndian.Uint32(args[100:]))
	residualBytes := int(binary.LittleEndian.Uint32(args[104:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[108:]))
	scaleBytes := int(binary.LittleEndian.Uint32(args[112:]))
	biasBytes := int(binary.LittleEndian.Uint32(args[116:]))
	multiplierBytes := int(binary.LittleEndian.Uint32(args[120:]))
	residualOutputBytes := int(binary.LittleEndian.Uint32(args[124:]))
	activationOutputBytes := int(binary.LittleEndian.Uint32(args[128:]))
	epsilon := math.Float32frombits(binary.LittleEndian.Uint32(args[132:]))
	encoding := binary.LittleEndian.Uint32(args[136:])
	flags := binary.LittleEndian.Uint32(args[140:])
	outputScale := math.Float32frombits(binary.LittleEndian.Uint32(args[144:]))
	if !hipMLXAffineSupportedBits(bits) ||
		validateHIPMLXAffineProjectionShape(cols, weightBytes/4, scaleBytes/2, biasBytes/2, rows, cols, groupSize, bits) != nil ||
		inputBytes != cols*4 ||
		residualBytes != cols*4 ||
		residualOutputBytes != cols*4 ||
		multiplierBytes != rows*4 ||
		activationOutputBytes != rows*4 {
		return core.E("rocm.hip.FakeLaunch", "rms residual-add GELU tanh projection shape metadata mismatch", nil)
	}
	if flags&^hipRMSNormLaunchFlagAddUnitWeight != 0 {
		return core.E("rocm.hip.FakeLaunch", "unsupported rms residual-add GELU tanh projection flags", nil)
	}
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		if rmsWeightPointer != 0 || rmsWeightBytes != 0 || flags != 0 {
			return core.E("rocm.hip.FakeLaunch", "rms residual-add GELU tanh projection unit weight metadata mismatch", nil)
		}
	case hipRMSNormWeightEncodingF32:
		if rmsWeightPointer == 0 || rmsWeightBytes != cols*4 {
			return core.E("rocm.hip.FakeLaunch", "rms residual-add GELU tanh projection f32 weight byte count mismatch", nil)
		}
	case hipRMSNormWeightEncodingBF16:
		if rmsWeightPointer == 0 || rmsWeightBytes != cols*2 {
			return core.E("rocm.hip.FakeLaunch", "rms residual-add GELU tanh projection bf16 weight byte count mismatch", nil)
		}
	default:
		return core.E("rocm.hip.FakeLaunch", "unsupported rms residual-add GELU tanh projection weight encoding", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms residual-add GELU tanh projection input buffer is missing", nil)
	}
	residualData, residualOffset, ok := driver.memoryForPointer(residualPointer, residualBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms residual-add GELU tanh projection residual buffer is missing", nil)
	}
	weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms residual-add GELU tanh projection packed weight buffer is missing", nil)
	}
	scaleData, scaleOffset, ok := driver.memoryForPointer(scalePointer, scaleBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms residual-add GELU tanh projection scale buffer is missing", nil)
	}
	biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms residual-add GELU tanh projection bias buffer is missing", nil)
	}
	multiplierData, multiplierOffset, ok := driver.memoryForPointer(multiplierPointer, multiplierBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms residual-add GELU tanh projection multiplier buffer is missing", nil)
	}
	residualOutputData, residualOutputOffset, ok := driver.memoryForPointer(residualOutputPointer, residualOutputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms residual-add GELU tanh projection residual output buffer is missing", nil)
	}
	activationOutputData, activationOutputOffset, ok := driver.memoryForPointer(activationOutputPointer, activationOutputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms residual-add GELU tanh projection activation output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	residual, err := hipFloat32PayloadValues(residualData[residualOffset : residualOffset+residualBytes])
	if err != nil {
		return err
	}
	rmsWeight := make([]float32, cols)
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		for index := range rmsWeight {
			rmsWeight[index] = 1
		}
	case hipRMSNormWeightEncodingF32:
		rmsWeightData, rmsWeightOffset, ok := driver.memoryForPointer(rmsWeightPointer, rmsWeightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "rms residual-add GELU tanh projection weight buffer is missing", nil)
		}
		rmsWeight, err = hipFloat32PayloadValues(rmsWeightData[rmsWeightOffset : rmsWeightOffset+rmsWeightBytes])
		if err != nil {
			return err
		}
	case hipRMSNormWeightEncodingBF16:
		rmsWeightData, rmsWeightOffset, ok := driver.memoryForPointer(rmsWeightPointer, rmsWeightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "rms residual-add GELU tanh projection weight buffer is missing", nil)
		}
		for index := range rmsWeight {
			rmsWeight[index] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(rmsWeightData[rmsWeightOffset+index*2:]))
		}
	}
	if flags&hipRMSNormLaunchFlagAddUnitWeight != 0 {
		for index := range rmsWeight {
			rmsWeight[index] += 1
		}
	}
	normalized, err := hipReferenceRMSNorm(input, rmsWeight, epsilon)
	if err != nil {
		return err
	}
	residualOutput := make([]float32, len(normalized))
	for index := range residualOutput {
		residualOutput[index] = (normalized[index] + residual[index]) * outputScale
	}
	residualPayload, err := hipFloat32Payload(residualOutput)
	if err != nil {
		return err
	}
	copy(residualOutputData[residualOutputOffset:residualOutputOffset+residualOutputBytes], residualPayload)
	weights := make([]uint32, weightBytes/4)
	for index := range weights {
		weights[index] = binary.LittleEndian.Uint32(weightData[weightOffset+index*4:])
	}
	scales := make([]uint16, scaleBytes/2)
	for index := range scales {
		scales[index] = binary.LittleEndian.Uint16(scaleData[scaleOffset+index*2:])
	}
	biases := make([]uint16, biasBytes/2)
	for index := range biases {
		biases[index] = binary.LittleEndian.Uint16(biasData[biasOffset+index*2:])
	}
	multiplier, err := hipFloat32PayloadValues(multiplierData[multiplierOffset : multiplierOffset+multiplierBytes])
	if err != nil {
		return err
	}
	projected, err := hipReferenceMLXAffineProjection(residualOutput, weights, scales, biases, rows, cols, groupSize, bits)
	if err != nil {
		return err
	}
	activation := make([]float32, rows)
	const sqrt2OverPi = 0.7978845608028654
	const coeff = 0.044715
	for index := range activation {
		value := float64(projected[index])
		gelu := 0.5 * value * (1 + math.Tanh(sqrt2OverPi*(value+coeff*value*value*value)))
		activation[index] = float32(gelu) * multiplier[index]
	}
	activationPayload, err := hipFloat32Payload(activation)
	if err != nil {
		return err
	}
	copy(activationOutputData[activationOutputOffset:activationOutputOffset+activationOutputBytes], activationPayload)
	return nil
}

func (driver *fakeHIPDriver) launchMLXQ4ProjectionGreedy(args []byte) error {
	if len(args) != hipMLXQ4ProjectionLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy projection launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMLXQ4ProjectionLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMLXQ4ProjectionLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy projection launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	scalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	rows := int(binary.LittleEndian.Uint32(args[48:]))
	cols := int(binary.LittleEndian.Uint32(args[52:]))
	groupSize := int(binary.LittleEndian.Uint32(args[56:]))
	bits := int(binary.LittleEndian.Uint32(args[60:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[64:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[68:]))
	scaleBytes := int(binary.LittleEndian.Uint32(args[72:]))
	biasBytes := int(binary.LittleEndian.Uint32(args[76:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[80:]))
	suppressCount := int(binary.LittleEndian.Uint32(args[84:]))
	suppressPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[88:]))
	if !hipMLXAffineSupportedBits(bits) ||
		validateHIPMLXAffineProjectionShape(cols, weightBytes/4, scaleBytes/2, biasBytes/2, rows, cols, groupSize, bits) != nil ||
		inputBytes != cols*4 ||
		outputBytes != hipMLXQ4ProjectionBestBytes ||
		(suppressCount > 0 && suppressPointer == 0) {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy projection shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy projection input buffer is missing", nil)
	}
	weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy projection packed weight buffer is missing", nil)
	}
	scaleData, scaleOffset, ok := driver.memoryForPointer(scalePointer, scaleBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy projection scale buffer is missing", nil)
	}
	biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy projection bias buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy projection output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	weights := make([]uint32, weightBytes/4)
	for index := range weights {
		weights[index] = binary.LittleEndian.Uint32(weightData[weightOffset+index*4:])
	}
	scales := make([]uint16, scaleBytes/2)
	for index := range scales {
		scales[index] = binary.LittleEndian.Uint16(scaleData[scaleOffset+index*2:])
	}
	biases := make([]uint16, biasBytes/2)
	for index := range biases {
		biases[index] = binary.LittleEndian.Uint16(biasData[biasOffset+index*2:])
	}
	output, err := hipReferenceMLXAffineProjection(input, weights, scales, biases, rows, cols, groupSize, bits)
	if err != nil {
		return err
	}
	var suppressTokens []int32
	if suppressCount > 0 {
		suppressBytes := suppressCount * 4
		suppressData, suppressOffset, ok := driver.memoryForPointer(suppressPointer, suppressBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy suppress token buffer is missing", nil)
		}
		suppressTokens = make([]int32, suppressCount)
		for index := range suppressTokens {
			suppressTokens[index] = int32(binary.LittleEndian.Uint32(suppressData[suppressOffset+index*4:]))
		}
	}
	bestIndex, bestScore, err := hipReferenceGreedySampleSuppress(output, suppressTokens)
	if err != nil {
		return err
	}
	binary.LittleEndian.PutUint64(outputData[outputOffset:outputOffset+outputBytes], hipPackGreedyBest(bestScore, bestIndex))
	return nil
}

func (driver *fakeHIPDriver) launchMLXQ4ProjectionSelectedGreedy(args []byte) error {
	if len(args) != hipMLXQ4ProjectionLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 selected greedy projection launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMLXQ4ProjectionLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMLXQ4ProjectionLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 selected greedy projection launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	scalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	rows := int(binary.LittleEndian.Uint32(args[48:]))
	cols := int(binary.LittleEndian.Uint32(args[52:]))
	groupSize := int(binary.LittleEndian.Uint32(args[56:]))
	bits := int(binary.LittleEndian.Uint32(args[60:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[64:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[68:]))
	scaleBytes := int(binary.LittleEndian.Uint32(args[72:]))
	biasBytes := int(binary.LittleEndian.Uint32(args[76:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[80:]))
	selectedCount := int(binary.LittleEndian.Uint32(args[84:]))
	selectedPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[88:]))
	if !hipMLXAffineSupportedBits(bits) ||
		validateHIPMLXAffineProjectionShape(cols, weightBytes/4, scaleBytes/2, biasBytes/2, rows, cols, groupSize, bits) != nil ||
		inputBytes != cols*4 ||
		outputBytes != hipMLXQ4ProjectionBestBytes ||
		selectedCount <= 0 ||
		selectedPointer == 0 {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 selected greedy projection shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 selected greedy projection input buffer is missing", nil)
	}
	weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 selected greedy projection packed weight buffer is missing", nil)
	}
	scaleData, scaleOffset, ok := driver.memoryForPointer(scalePointer, scaleBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 selected greedy projection scale buffer is missing", nil)
	}
	biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 selected greedy projection bias buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 selected greedy projection output buffer is missing", nil)
	}
	selectedBytes := selectedCount * 4
	selectedData, selectedOffset, ok := driver.memoryForPointer(selectedPointer, selectedBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 selected greedy token buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	weights := make([]uint32, weightBytes/4)
	for index := range weights {
		weights[index] = binary.LittleEndian.Uint32(weightData[weightOffset+index*4:])
	}
	scales := make([]uint16, scaleBytes/2)
	for index := range scales {
		scales[index] = binary.LittleEndian.Uint16(scaleData[scaleOffset+index*2:])
	}
	biases := make([]uint16, biasBytes/2)
	for index := range biases {
		biases[index] = binary.LittleEndian.Uint16(biasData[biasOffset+index*2:])
	}
	logits, err := hipReferenceMLXAffineProjection(input, weights, scales, biases, rows, cols, groupSize, bits)
	if err != nil {
		return err
	}
	best := uint64(0)
	for index := 0; index < selectedCount; index++ {
		token := int(int32(binary.LittleEndian.Uint32(selectedData[selectedOffset+index*4:])))
		if token < 0 || token >= rows {
			continue
		}
		packed := hipPackGreedyBest(logits[token], token)
		if packed > best {
			best = packed
		}
	}
	binary.LittleEndian.PutUint64(outputData[outputOffset:outputOffset+outputBytes], best)
	return nil
}

func (driver *fakeHIPDriver) launchMLXQ4ProjectionGreedyBatch(args []byte) error {
	if len(args) != hipMLXQ4ProjectionGreedyBatchLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy batch projection launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMLXQ4ProjectionGreedyBatchLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMLXQ4ProjectionGreedyBatchLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy batch projection launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	scalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	suppressPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	rows := int(binary.LittleEndian.Uint32(args[56:]))
	cols := int(binary.LittleEndian.Uint32(args[60:]))
	batch := int(binary.LittleEndian.Uint32(args[64:]))
	groupSize := int(binary.LittleEndian.Uint32(args[68:]))
	bits := int(binary.LittleEndian.Uint32(args[72:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[76:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[80:]))
	scaleBytes := int(binary.LittleEndian.Uint32(args[84:]))
	biasBytes := int(binary.LittleEndian.Uint32(args[88:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[92:]))
	suppressCount := int(binary.LittleEndian.Uint32(args[96:]))
	if !hipMLXAffineSupportedBits(bits) ||
		validateHIPMLXAffineProjectionShape(cols, weightBytes/4, scaleBytes/2, biasBytes/2, rows, cols, groupSize, bits) != nil ||
		inputBytes != batch*cols*4 ||
		outputBytes != batch*hipMLXQ4ProjectionBestBytes ||
		(suppressCount > 0 && suppressPointer == 0) {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy batch projection shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy batch projection input buffer is missing", nil)
	}
	weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy batch projection packed weight buffer is missing", nil)
	}
	scaleData, scaleOffset, ok := driver.memoryForPointer(scalePointer, scaleBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy batch projection scale buffer is missing", nil)
	}
	biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy batch projection bias buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy batch projection output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	weights := make([]uint32, weightBytes/4)
	for index := range weights {
		weights[index] = binary.LittleEndian.Uint32(weightData[weightOffset+index*4:])
	}
	scales := make([]uint16, scaleBytes/2)
	for index := range scales {
		scales[index] = binary.LittleEndian.Uint16(scaleData[scaleOffset+index*2:])
	}
	biases := make([]uint16, biasBytes/2)
	for index := range biases {
		biases[index] = binary.LittleEndian.Uint16(biasData[biasOffset+index*2:])
	}
	var suppressTokens []int32
	if suppressCount > 0 {
		suppressBytes := suppressCount * 4
		suppressData, suppressOffset, ok := driver.memoryForPointer(suppressPointer, suppressBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "MLX q4 greedy batch suppress token buffer is missing", nil)
		}
		suppressTokens = make([]int32, suppressCount)
		for index := range suppressTokens {
			suppressTokens[index] = int32(binary.LittleEndian.Uint32(suppressData[suppressOffset+index*4:]))
		}
	}
	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		inputStart := batchIndex * cols
		output, err := hipReferenceMLXAffineProjection(input[inputStart:inputStart+cols], weights, scales, biases, rows, cols, groupSize, bits)
		if err != nil {
			return err
		}
		bestIndex, bestScore, err := hipReferenceGreedySampleSuppress(output, suppressTokens)
		if err != nil {
			return err
		}
		binary.LittleEndian.PutUint64(outputData[outputOffset+batchIndex*hipMLXQ4ProjectionBestBytes:], hipPackGreedyBest(bestScore, bestIndex))
	}
	return nil
}

func (driver *fakeHIPDriver) launchMLXQ4ProjectionScores(args []byte) error {
	if len(args) != hipMLXQ4ProjectionLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 score projection launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMLXQ4ProjectionLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMLXQ4ProjectionLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 score projection launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	scalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	rows := int(binary.LittleEndian.Uint32(args[48:]))
	cols := int(binary.LittleEndian.Uint32(args[52:]))
	groupSize := int(binary.LittleEndian.Uint32(args[56:]))
	bits := int(binary.LittleEndian.Uint32(args[60:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[64:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[68:]))
	scaleBytes := int(binary.LittleEndian.Uint32(args[72:]))
	biasBytes := int(binary.LittleEndian.Uint32(args[76:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[80:]))
	suppressCount := int(binary.LittleEndian.Uint32(args[84:]))
	suppressPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[88:]))
	if !hipMLXAffineSupportedBits(bits) ||
		validateHIPMLXAffineProjectionShape(cols, weightBytes/4, scaleBytes/2, biasBytes/2, rows, cols, groupSize, bits) != nil ||
		inputBytes != cols*4 ||
		outputBytes != rows*hipMLXQ4ProjectionBestBytes ||
		(suppressCount > 0 && suppressPointer == 0) {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 score projection shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 score projection input buffer is missing", nil)
	}
	weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 score projection packed weight buffer is missing", nil)
	}
	scaleData, scaleOffset, ok := driver.memoryForPointer(scalePointer, scaleBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 score projection scale buffer is missing", nil)
	}
	biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 score projection bias buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 score projection output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	weights := make([]uint32, weightBytes/4)
	for index := range weights {
		weights[index] = binary.LittleEndian.Uint32(weightData[weightOffset+index*4:])
	}
	scales := make([]uint16, scaleBytes/2)
	for index := range scales {
		scales[index] = binary.LittleEndian.Uint16(scaleData[scaleOffset+index*2:])
	}
	biases := make([]uint16, biasBytes/2)
	for index := range biases {
		biases[index] = binary.LittleEndian.Uint16(biasData[biasOffset+index*2:])
	}
	output, err := hipReferenceMLXAffineProjection(input, weights, scales, biases, rows, cols, groupSize, bits)
	if err != nil {
		return err
	}
	var suppressTokens []int32
	if suppressCount > 0 {
		suppressBytes := suppressCount * 4
		suppressData, suppressOffset, ok := driver.memoryForPointer(suppressPointer, suppressBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "MLX q4 score suppress token buffer is missing", nil)
		}
		suppressTokens = make([]int32, suppressCount)
		for index := range suppressTokens {
			suppressTokens[index] = int32(binary.LittleEndian.Uint32(suppressData[suppressOffset+index*4:]))
		}
	}
	for index, score := range output {
		packed := uint64(0)
		if !hipTokenIsSuppressed(int32(index), suppressTokens) {
			packed = hipPackGreedyBest(score, index)
		}
		binary.LittleEndian.PutUint64(outputData[outputOffset+index*8:], packed)
	}
	return nil
}

func (driver *fakeHIPDriver) launchPackedTopK(args []byte) error {
	if len(args) != hipPackedTopKLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "packed top-k launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipPackedTopKLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipPackedTopKLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "packed top-k launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	inputCount := int(binary.LittleEndian.Uint32(args[24:]))
	outputCount := int(binary.LittleEndian.Uint32(args[28:]))
	topK := int(binary.LittleEndian.Uint32(args[32:]))
	chunkSize := int(binary.LittleEndian.Uint32(args[36:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[40:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[44:]))
	if inputCount <= 0 || outputCount <= 0 || topK <= 0 || topK > hipPackedTopKMaxK || chunkSize != hipPackedTopKChunkSize ||
		inputBytes != inputCount*hipMLXQ4ProjectionBestBytes ||
		outputBytes != outputCount*hipMLXQ4ProjectionBestBytes ||
		outputCount != ((inputCount+chunkSize-1)/chunkSize)*topK {
		return core.E("rocm.hip.FakeLaunch", "packed top-k shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "packed top-k input buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "packed top-k output buffer is missing", nil)
	}
	chunkCount := (inputCount + chunkSize - 1) / chunkSize
	for chunk := 0; chunk < chunkCount; chunk++ {
		begin := inputOffset + chunk*chunkSize*hipMLXQ4ProjectionBestBytes
		endIndex := (chunk + 1) * chunkSize
		if endIndex > inputCount {
			endIndex = inputCount
		}
		end := inputOffset + endIndex*hipMLXQ4ProjectionBestBytes
		top := hipTopPackedScoresBytes(inputData[begin:end], topK)
		for index := 0; index < topK; index++ {
			value := uint64(0)
			if index < len(top) {
				value = top[index]
			}
			binary.LittleEndian.PutUint64(outputData[outputOffset+(chunk*topK+index)*hipMLXQ4ProjectionBestBytes:], value)
		}
	}
	return nil
}

func (driver *fakeHIPDriver) launchOrderedEmbeddingCandidates(args []byte) error {
	if len(args) != hipOrderedEmbeddingCandidatesLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "ordered embedding candidates launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipOrderedEmbeddingCandidatesLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipOrderedEmbeddingCandidatesLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "ordered embedding candidates launch header mismatch", nil)
	}
	topKPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	orderingPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	suppressPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	topKCount := int(binary.LittleEndian.Uint32(args[40:]))
	centroids := int(binary.LittleEndian.Uint32(args[44:]))
	tokensPerCentroid := int(binary.LittleEndian.Uint32(args[48:]))
	elementBytes := int(binary.LittleEndian.Uint32(args[52:]))
	orderingCount := int(binary.LittleEndian.Uint32(args[56:]))
	outputCount := int(binary.LittleEndian.Uint32(args[60:]))
	suppressCount := int(binary.LittleEndian.Uint32(args[64:]))
	topKBytes := int(binary.LittleEndian.Uint32(args[68:]))
	orderingBytes := int(binary.LittleEndian.Uint32(args[72:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[76:]))
	if topKCount <= 0 || centroids <= 0 || tokensPerCentroid <= 0 ||
		(elementBytes != 4 && elementBytes != 8) ||
		orderingCount != centroids*tokensPerCentroid ||
		outputCount != topKCount*tokensPerCentroid ||
		topKBytes != topKCount*hipMLXQ4ProjectionBestBytes ||
		orderingBytes != orderingCount*elementBytes ||
		outputBytes != outputCount*4 {
		return core.E("rocm.hip.FakeLaunch", "ordered embedding candidates shape metadata mismatch", nil)
	}
	topKData, topKOffset, ok := driver.memoryForPointer(topKPointer, topKBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "ordered embedding top-k buffer is missing", nil)
	}
	orderingData, orderingOffset, ok := driver.memoryForPointer(orderingPointer, orderingBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "ordered embedding token-ordering buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "ordered embedding output buffer is missing", nil)
	}
	suppressed := map[int32]struct{}{}
	if suppressCount > 0 {
		suppressBytes := suppressCount * 4
		suppressData, suppressOffset, ok := driver.memoryForPointer(suppressPointer, suppressBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "ordered embedding suppress buffer is missing", nil)
		}
		for index := 0; index < suppressCount; index++ {
			suppressed[int32(binary.LittleEndian.Uint32(suppressData[suppressOffset+index*4:]))] = struct{}{}
		}
	}
	for rank := 0; rank < topKCount; rank++ {
		packed := binary.LittleEndian.Uint64(topKData[topKOffset+rank*hipMLXQ4ProjectionBestBytes:])
		centroid := -1
		if packed != 0 {
			centroid = int(^uint32(packed))
		}
		for tokenOffset := 0; tokenOffset < tokensPerCentroid; tokenOffset++ {
			selected := int32(-1)
			if centroid >= 0 && centroid < centroids {
				orderIndex := centroid*tokensPerCentroid + tokenOffset
				var id int64
				if elementBytes == 4 {
					id = int64(int32(binary.LittleEndian.Uint32(orderingData[orderingOffset+orderIndex*4:])))
				} else {
					id = int64(binary.LittleEndian.Uint64(orderingData[orderingOffset+orderIndex*8:]))
				}
				if id >= 0 && id <= math.MaxInt32 {
					if _, skip := suppressed[int32(id)]; !skip {
						selected = int32(id)
					}
				}
			}
			binary.LittleEndian.PutUint32(outputData[outputOffset+(rank*tokensPerCentroid+tokenOffset)*4:], uint32(selected))
		}
	}
	return nil
}

func (driver *fakeHIPDriver) launchPackedTopKSample(args []byte) error {
	if len(args) != hipPackedTopKSampleLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "packed top-k sample launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipPackedTopKSampleLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipPackedTopKSampleLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "packed top-k sample launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	inputCount := int(binary.LittleEndian.Uint32(args[24:]))
	topK := int(binary.LittleEndian.Uint32(args[28:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[32:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[36:]))
	temperature := math.Float32frombits(binary.LittleEndian.Uint32(args[40:]))
	topP := math.Float32frombits(binary.LittleEndian.Uint32(args[44:]))
	draw := math.Float64frombits(binary.LittleEndian.Uint64(args[48:]))
	softcap := math.Float32frombits(binary.LittleEndian.Uint32(args[56:]))
	if inputCount <= 0 || topK <= 0 || topK > inputCount || topK > hipPackedTopKMaxK ||
		inputBytes != inputCount*hipMLXQ4ProjectionBestBytes ||
		outputBytes != hipMLXQ4ProjectionBestBytes ||
		temperature < 0 || math.IsNaN(float64(temperature)) || math.IsInf(float64(temperature), 0) ||
		topP < 0 || topP > 1 || math.IsNaN(float64(topP)) || math.IsInf(float64(topP), 0) ||
		softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) ||
		math.IsNaN(draw) || math.IsInf(draw, 0) {
		return core.E("rocm.hip.FakeLaunch", "packed top-k sample shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "packed top-k sample input buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "packed top-k sample output buffer is missing", nil)
	}
	candidates := make([]hipGreedySampleResult, 0, topK)
	for index := 0; index < topK; index++ {
		packed := binary.LittleEndian.Uint64(inputData[inputOffset+index*hipMLXQ4ProjectionBestBytes:])
		if packed == 0 {
			continue
		}
		candidate, err := hipUnpackGreedyBest(packed, 0, math.MaxInt32)
		if err != nil {
			return err
		}
		candidates = append(candidates, candidate)
	}
	if len(candidates) == 0 {
		binary.LittleEndian.PutUint64(outputData[outputOffset:], 0)
		return nil
	}
	samplingCandidates := append([]hipGreedySampleResult(nil), candidates...)
	if softcap > 0 {
		for index := range samplingCandidates {
			samplingCandidates[index].Score = float32(math.Tanh(float64(samplingCandidates[index].Score/softcap))) * softcap
		}
	}
	result, err := hipGemma4Q4HostSampleSortedCandidateResultWorkspace(samplingCandidates, inference.GenerateConfig{
		Temperature:   temperature,
		TopK:          topK,
		TopP:          topP,
		RepeatPenalty: 1,
	}, nil, draw, nil)
	if err != nil {
		return err
	}
	for _, candidate := range candidates {
		if candidate.TokenID == result.TokenID {
			binary.LittleEndian.PutUint64(outputData[outputOffset:], hipPackGreedyBest(candidate.Score, candidate.TokenID))
			return nil
		}
	}
	return core.E("rocm.hip.FakeLaunch", "sampled token is not in packed candidates", nil)
}

func (driver *fakeHIPDriver) launchJANGTQProjection(args []byte) error {
	if len(args) != hipJANGTQLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "JANGTQ launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipJANGTQLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipJANGTQLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "JANGTQ launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	packedPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	inputCount := int(binary.LittleEndian.Uint32(args[40:]))
	rows := int(binary.LittleEndian.Uint32(args[44:]))
	cols := int(binary.LittleEndian.Uint32(args[48:]))
	bits := int(binary.LittleEndian.Uint32(args[52:]))
	groupSize := int(binary.LittleEndian.Uint32(args[56:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[60:]))
	packedBytes := int(binary.LittleEndian.Uint32(args[64:]))
	biasBytes := int(binary.LittleEndian.Uint32(args[68:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[72:]))
	scale := math.Float32frombits(binary.LittleEndian.Uint32(args[76:]))
	flags := binary.LittleEndian.Uint32(args[80:])
	if inputCount != cols || rows <= 0 || cols <= 0 || inputBytes != cols*4 || outputBytes != rows*4 || packedBytes < packedROCmJANGTQBytes(bits, rows*cols) {
		return core.E("rocm.hip.FakeLaunch", "JANGTQ shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "JANGTQ input buffer is missing", nil)
	}
	packedData, packedOffset, ok := driver.memoryForPointer(packedPointer, packedBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "JANGTQ packed weight buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "JANGTQ output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValuesInto(driver.float32Scratch("JANGTQ input", inputCount), inputData[inputOffset:inputOffset+inputBytes])
	if err != nil {
		return err
	}
	var bias []float32
	if flags&hipJANGTQLaunchFlagBias != 0 {
		biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "JANGTQ bias buffer is missing", nil)
		}
		bias, err = hipFloat32PayloadValuesInto(driver.float32Scratch("JANGTQ bias", rows), biasData[biasOffset:biasOffset+biasBytes])
		if err != nil {
			return err
		}
	}
	output := driver.float32Scratch("JANGTQ output", rows)
	if err := rocmReferenceJANGTQProjectionInto(
		output,
		input[:inputCount],
		packedData[packedOffset:packedOffset+packedBytes],
		driver.int8Scratch("JANGTQ quantized", rows*cols),
		rocmJANGTQDescriptor{WeightFormat: "mxtq", Bits: bits, GroupSize: groupSize},
		rows,
		cols,
		scale,
		bias,
	); err != nil {
		return err
	}
	_, err = hipFloat32PayloadInto(outputData[outputOffset:outputOffset+outputBytes], output)
	if err != nil {
		return err
	}
	return nil
}

func (driver *fakeHIPDriver) float32Scratch(label string, count int) []float32 {
	if count <= 0 {
		return nil
	}
	var scratch *[]float32
	switch label {
	case "JANGTQ input":
		scratch = &driver.jangtqInputScratch
	case "JANGTQ bias":
		scratch = &driver.jangtqBiasScratch
	default:
		scratch = &driver.jangtqOutputScratch
	}
	if cap(*scratch) < count {
		*scratch = make([]float32, count)
	}
	return (*scratch)[:count]
}

func (driver *fakeHIPDriver) int8Scratch(_ string, count int) []int8 {
	if count <= 0 {
		return nil
	}
	if cap(driver.jangtqQuantizedScratch) < count {
		driver.jangtqQuantizedScratch = make([]int8, count)
	}
	return driver.jangtqQuantizedScratch[:count]
}

func (driver *fakeHIPDriver) launchCodebookLookup(args []byte) error {
	if len(args) != hipCodebookLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "codebook launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipCodebookLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipCodebookLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "codebook launch header mismatch", nil)
	}
	codePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	codebookPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	codeCount := int(binary.LittleEndian.Uint32(args[32:]))
	codebookCount := int(binary.LittleEndian.Uint32(args[36:]))
	codeDim := int(binary.LittleEndian.Uint32(args[40:]))
	codeBytes := int(binary.LittleEndian.Uint32(args[44:]))
	codebookBytes := int(binary.LittleEndian.Uint32(args[48:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[52:]))
	if codeCount <= 0 || codebookCount <= 0 || codeDim <= 0 || codeBytes != codeCount || codebookBytes != codebookCount*codeDim*4 || outputBytes != codeCount*codeDim*4 {
		return core.E("rocm.hip.FakeLaunch", "codebook shape metadata mismatch", nil)
	}
	codeData, codeOffset, ok := driver.memoryForPointer(codePointer, codeBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "codebook code buffer is missing", nil)
	}
	codebookData, codebookOffset, ok := driver.memoryForPointer(codebookPointer, codebookBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "codebook table buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "codebook output buffer is missing", nil)
	}
	for codeIndex, code := range codeData[codeOffset : codeOffset+codeBytes] {
		if int(code) >= codebookCount {
			return core.E("rocm.hip.FakeLaunch", core.Sprintf("code %d outside codebook size %d", int(code), codebookCount), nil)
		}
		sourceBegin := codebookOffset + int(code)*codeDim*4
		sourceEnd := sourceBegin + codeDim*4
		targetBegin := outputOffset + codeIndex*codeDim*4
		for offset := sourceBegin; offset < sourceEnd; offset += 4 {
			value := math.Float32frombits(binary.LittleEndian.Uint32(codebookData[offset:]))
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return core.E("rocm.hip.FakeLaunch", "codebook values must be finite", nil)
			}
		}
		copy(outputData[targetBegin:targetBegin+codeDim*4], codebookData[sourceBegin:sourceEnd])
	}
	return nil
}

func (driver *fakeHIPDriver) launchLoRAProjection(args []byte) error {
	if len(args) != hipLoRALaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "LoRA launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipLoRALaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipLoRALaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "LoRA launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	basePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	aPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	bPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	inputCount := int(binary.LittleEndian.Uint32(args[56:]))
	rows := int(binary.LittleEndian.Uint32(args[60:]))
	cols := int(binary.LittleEndian.Uint32(args[64:]))
	rank := int(binary.LittleEndian.Uint32(args[68:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[72:]))
	baseBytes := int(binary.LittleEndian.Uint32(args[76:]))
	aBytes := int(binary.LittleEndian.Uint32(args[80:]))
	bBytes := int(binary.LittleEndian.Uint32(args[84:]))
	biasBytes := int(binary.LittleEndian.Uint32(args[88:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[92:]))
	alpha := math.Float32frombits(binary.LittleEndian.Uint32(args[96:]))
	flags := binary.LittleEndian.Uint32(args[100:])
	if inputCount != cols || rows <= 0 || cols <= 0 || rank <= 0 || inputBytes != cols*4 ||
		baseBytes != rows*cols*4 || aBytes != rank*cols*4 || bBytes != rows*rank*4 ||
		outputBytes != rows*4 || !hipQ8ScaleIsPositiveFinite(alpha) {
		return core.E("rocm.hip.FakeLaunch", "LoRA shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "LoRA input buffer is missing", nil)
	}
	baseData, baseOffset, ok := driver.memoryForPointer(basePointer, baseBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "LoRA base weight buffer is missing", nil)
	}
	aData, aOffset, ok := driver.memoryForPointer(aPointer, aBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "LoRA A buffer is missing", nil)
	}
	bData, bOffset, ok := driver.memoryForPointer(bPointer, bBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "LoRA B buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "LoRA output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	base, err := hipFloat32PayloadValues(baseData[baseOffset : baseOffset+baseBytes])
	if err != nil {
		return err
	}
	loraA, err := hipFloat32PayloadValues(aData[aOffset : aOffset+aBytes])
	if err != nil {
		return err
	}
	loraB, err := hipFloat32PayloadValues(bData[bOffset : bOffset+bBytes])
	if err != nil {
		return err
	}
	var bias []float32
	if flags&hipLoRALaunchFlagBias != 0 {
		biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "LoRA bias buffer is missing", nil)
		}
		bias, err = hipFloat32PayloadValues(biasData[biasOffset : biasOffset+biasBytes])
		if err != nil {
			return err
		}
	}
	output, err := rocmReferenceLoRAProjection(input[:inputCount], base, loraA, loraB, rows, cols, rank, alpha, bias)
	if err != nil {
		return err
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchEmbeddingLookup(args []byte, greedyToken bool) error {
	if len(args) != hipEmbeddingLookupLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "embedding lookup launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipEmbeddingLookupLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipEmbeddingLookupLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "embedding lookup launch header mismatch", nil)
	}
	tokenPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	embeddingPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	tokenCount := int(binary.LittleEndian.Uint32(args[32:]))
	vocabSize := int(binary.LittleEndian.Uint32(args[36:]))
	hiddenSize := int(binary.LittleEndian.Uint32(args[40:]))
	tokenBytes := int(binary.LittleEndian.Uint32(args[44:]))
	embeddingBytes := int(binary.LittleEndian.Uint64(args[48:]))
	outputBytes := int(binary.LittleEndian.Uint64(args[56:]))
	encoding := binary.LittleEndian.Uint32(args[64:])
	groupSize := int(binary.LittleEndian.Uint32(args[68:]))
	scalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[72:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[80:]))
	scaleBytes := int(binary.LittleEndian.Uint32(args[88:]))
	biasBytes := int(binary.LittleEndian.Uint32(args[92:]))
	quantBits := int(binary.LittleEndian.Uint32(args[100:]))
	outputScale := float32(1)
	if bits := binary.LittleEndian.Uint32(args[96:]); bits != 0 {
		outputScale = math.Float32frombits(bits)
		if math.IsNaN(float64(outputScale)) || math.IsInf(float64(outputScale), 0) {
			return core.E("rocm.hip.FakeLaunch", "embedding lookup output scale must be finite", nil)
		}
	}
	wantTokenBytes := tokenCount * 4
	if greedyToken {
		wantTokenBytes = hipMLXQ4ProjectionBestBytes
	}
	if tokenCount <= 0 || vocabSize <= 0 || hiddenSize <= 0 || tokenBytes != wantTokenBytes || outputBytes != tokenCount*hiddenSize*4 {
		return core.E("rocm.hip.FakeLaunch", "embedding lookup shape metadata mismatch", nil)
	}
	if greedyToken && tokenCount != 1 {
		return core.E("rocm.hip.FakeLaunch", "embedding lookup greedy token count mismatch", nil)
	}
	tokenData, tokenOffset, ok := driver.memoryForPointer(tokenPointer, tokenBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "embedding lookup token buffer is missing", nil)
	}
	embeddingData, embeddingOffset, ok := driver.memoryForPointer(embeddingPointer, embeddingBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "embedding lookup table buffer is missing", nil)
	}
	var scaleData []byte
	var scaleOffset int
	var biasData []byte
	var biasOffset int
	packedPerRow := 0
	if encoding == hipEmbeddingTableEncodingMLXQ4 {
		quantBits = hipMLXQ4ProjectionBitsOrDefault(quantBits)
		var err error
		packedPerRow, err = hipMLXAffinePackedCols(hiddenSize, quantBits)
		if err != nil || groupSize <= 0 || hiddenSize%groupSize != 0 {
			return core.E("rocm.hip.FakeLaunch", "embedding lookup MLX affine shape metadata mismatch", err)
		}
		if scaleBytes != vocabSize*(hiddenSize/groupSize)*2 || biasBytes != scaleBytes {
			return core.E("rocm.hip.FakeLaunch", "embedding lookup MLX affine scale/bias byte count mismatch", nil)
		}
		var ok bool
		scaleData, scaleOffset, ok = driver.memoryForPointer(scalePointer, scaleBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "embedding lookup q4 scale buffer is missing", nil)
		}
		biasData, biasOffset, ok = driver.memoryForPointer(biasPointer, biasBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "embedding lookup q4 bias buffer is missing", nil)
		}
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "embedding lookup output buffer is missing", nil)
	}
	output := make([]float32, tokenCount*hiddenSize)
	for tokenIndex := 0; tokenIndex < tokenCount; tokenIndex++ {
		id := int(int32(binary.LittleEndian.Uint32(tokenData[tokenOffset+tokenIndex*4:])))
		if greedyToken {
			id = int(^uint32(binary.LittleEndian.Uint64(tokenData[tokenOffset:])))
		}
		if id < 0 || id >= vocabSize {
			return core.E("rocm.hip.FakeLaunch", "embedding lookup token ID is outside vocabulary", nil)
		}
		for dim := 0; dim < hiddenSize; dim++ {
			tableIndex := id*hiddenSize + dim
			switch encoding {
			case hipEmbeddingTableEncodingF32:
				if embeddingBytes != vocabSize*hiddenSize*4 {
					return core.E("rocm.hip.FakeLaunch", "embedding lookup f32 byte count mismatch", nil)
				}
				output[tokenIndex*hiddenSize+dim] = math.Float32frombits(binary.LittleEndian.Uint32(embeddingData[embeddingOffset+tableIndex*4:]))
			case hipEmbeddingTableEncodingBF16:
				if embeddingBytes != vocabSize*hiddenSize*2 {
					return core.E("rocm.hip.FakeLaunch", "embedding lookup bf16 byte count mismatch", nil)
				}
				output[tokenIndex*hiddenSize+dim] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(embeddingData[embeddingOffset+tableIndex*2:]))
			case hipEmbeddingTableEncodingMLXQ4:
				groupsPerRow := hiddenSize / groupSize
				if embeddingBytes != vocabSize*packedPerRow*4 {
					return core.E("rocm.hip.FakeLaunch", "embedding lookup MLX affine byte count mismatch", nil)
				}
				rowWeights := make([]uint32, packedPerRow)
				rowOffset := embeddingOffset + id*packedPerRow*4
				for index := range rowWeights {
					rowWeights[index] = binary.LittleEndian.Uint32(embeddingData[rowOffset+index*4:])
				}
				q, err := hipMLXAffineUnpackValue(rowWeights, dim, quantBits)
				if err != nil {
					return err
				}
				quantized := float32(q)
				group := id*groupsPerRow + dim/groupSize
				scale := hipBFloat16ToFloat32(binary.LittleEndian.Uint16(scaleData[scaleOffset+group*2:]))
				bias := hipBFloat16ToFloat32(binary.LittleEndian.Uint16(biasData[biasOffset+group*2:]))
				output[tokenIndex*hiddenSize+dim] = quantized*scale + bias
			default:
				return core.E("rocm.hip.FakeLaunch", "unsupported embedding lookup encoding", nil)
			}
		}
	}
	for index := range output {
		output[index] *= outputScale
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchEmbeddingMeanPool(args []byte) error {
	if len(args) != hipEmbeddingMeanPoolLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "embedding mean-pool launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipEmbeddingMeanPoolLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipEmbeddingMeanPoolLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "embedding mean-pool launch header mismatch", nil)
	}
	tokenPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	tokenCount := int(binary.LittleEndian.Uint32(args[24:]))
	dim := int(binary.LittleEndian.Uint32(args[28:]))
	tokenBytes := int(binary.LittleEndian.Uint32(args[32:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[36:]))
	flags := binary.LittleEndian.Uint32(args[40:])
	if tokenCount <= 0 || dim <= 0 || tokenBytes != tokenCount*dim*4 || outputBytes != dim*4 {
		return core.E("rocm.hip.FakeLaunch", "embedding mean-pool shape metadata mismatch", nil)
	}
	tokenData, tokenOffset, ok := driver.memoryForPointer(tokenPointer, tokenBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "embedding token buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "embedding output buffer is missing", nil)
	}
	tokens, err := hipFloat32PayloadValues(tokenData[tokenOffset : tokenOffset+tokenBytes])
	if err != nil {
		return err
	}
	output, err := rocmReferenceMeanPoolEmbedding(splitFloat32Vectors(tokens, dim), flags&hipEmbeddingMeanPoolLaunchFlagNormalize != 0)
	if err != nil {
		return err
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchRerankCosine(args []byte) error {
	if len(args) != hipRerankCosineLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "rerank cosine launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipRerankCosineLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipRerankCosineLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "rerank cosine launch header mismatch", nil)
	}
	queryPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	documentPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	documentCount := int(binary.LittleEndian.Uint32(args[32:]))
	dim := int(binary.LittleEndian.Uint32(args[36:]))
	queryBytes := int(binary.LittleEndian.Uint32(args[40:]))
	documentBytes := int(binary.LittleEndian.Uint32(args[44:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[48:]))
	if documentCount <= 0 || dim <= 0 || queryBytes != dim*4 || documentBytes != documentCount*dim*4 || outputBytes != documentCount*4 {
		return core.E("rocm.hip.FakeLaunch", "rerank cosine shape metadata mismatch", nil)
	}
	queryData, queryOffset, ok := driver.memoryForPointer(queryPointer, queryBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rerank query buffer is missing", nil)
	}
	documentData, documentOffset, ok := driver.memoryForPointer(documentPointer, documentBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rerank document buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rerank output buffer is missing", nil)
	}
	query, err := hipFloat32PayloadValues(queryData[queryOffset : queryOffset+queryBytes])
	if err != nil {
		return err
	}
	documents, err := hipFloat32PayloadValues(documentData[documentOffset : documentOffset+documentBytes])
	if err != nil {
		return err
	}
	scores := make([]float32, documentCount)
	for index := 0; index < documentCount; index++ {
		start := index * dim
		score, err := rocmReferenceCosineSimilarity(query, documents[start:start+dim])
		if err != nil {
			return err
		}
		scores[index] = float32(score)
	}
	payload, err := hipFloat32Payload(scores)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchMoECombineNorms(args []byte) error {
	if len(args) != hipMoECombineNormsLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MoE combine norms launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMoECombineNormsLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMoECombineNormsLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MoE combine norms launch header mismatch", nil)
	}
	localInputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	localWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	expertInputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	expertWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	count := int(binary.LittleEndian.Uint32(args[48:]))
	localInputBytes := int(binary.LittleEndian.Uint32(args[52:]))
	localWeightBytes := int(binary.LittleEndian.Uint32(args[56:]))
	expertInputBytes := int(binary.LittleEndian.Uint32(args[60:]))
	expertWeightBytes := int(binary.LittleEndian.Uint32(args[64:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[68:]))
	localEpsilon := math.Float32frombits(binary.LittleEndian.Uint32(args[72:]))
	localEncoding := binary.LittleEndian.Uint32(args[76:])
	localFlags := binary.LittleEndian.Uint32(args[80:])
	expertEpsilon := math.Float32frombits(binary.LittleEndian.Uint32(args[84:]))
	expertEncoding := binary.LittleEndian.Uint32(args[88:])
	expertFlags := binary.LittleEndian.Uint32(args[92:])
	if count <= 0 || localInputBytes != count*4 || expertInputBytes != count*4 || outputBytes != count*4 ||
		math.IsNaN(float64(localEpsilon)) || math.IsInf(float64(localEpsilon), 0) || localEpsilon < 0 ||
		math.IsNaN(float64(expertEpsilon)) || math.IsInf(float64(expertEpsilon), 0) || expertEpsilon < 0 {
		return core.E("rocm.hip.FakeLaunch", "MoE combine norms shape metadata mismatch", nil)
	}
	localWeight, err := driver.moECombineNormsWeight(localWeightPointer, localWeightBytes, count, localEncoding, localFlags)
	if err != nil {
		return err
	}
	expertWeight, err := driver.moECombineNormsWeight(expertWeightPointer, expertWeightBytes, count, expertEncoding, expertFlags)
	if err != nil {
		return err
	}
	localData, localOffset, ok := driver.memoryForPointer(localInputPointer, localInputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MoE combine norms local input buffer is missing", nil)
	}
	expertData, expertOffset, ok := driver.memoryForPointer(expertInputPointer, expertInputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MoE combine norms expert input buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MoE combine norms output buffer is missing", nil)
	}
	localInput, err := hipFloat32PayloadValues(localData[localOffset : localOffset+localInputBytes])
	if err != nil {
		return err
	}
	expertInput, err := hipFloat32PayloadValues(expertData[expertOffset : expertOffset+expertInputBytes])
	if err != nil {
		return err
	}
	localOutput, err := hipReferenceRMSNorm(localInput, localWeight, localEpsilon)
	if err != nil {
		return err
	}
	expertOutput, err := hipReferenceRMSNorm(expertInput, expertWeight, expertEpsilon)
	if err != nil {
		return err
	}
	for index := range localOutput {
		localOutput[index] += expertOutput[index]
	}
	payload, err := hipFloat32Payload(localOutput)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) moECombineNormsWeight(pointer nativeDevicePointer, bytes, count int, encoding, flags uint32) ([]float32, error) {
	if flags&^hipRMSNormLaunchFlagAddUnitWeight != 0 {
		return nil, core.E("rocm.hip.FakeLaunch", "MoE combine norms unsupported RMSNorm weight flags", nil)
	}
	weight := make([]float32, count)
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		if pointer != 0 || bytes != 0 || flags != 0 {
			return nil, core.E("rocm.hip.FakeLaunch", "MoE combine norms unit weight metadata mismatch", nil)
		}
		for index := range weight {
			weight[index] = 1
		}
	case hipRMSNormWeightEncodingF32:
		if pointer == 0 || bytes != count*4 {
			return nil, core.E("rocm.hip.FakeLaunch", "MoE combine norms f32 weight byte count mismatch", nil)
		}
		data, offset, ok := driver.memoryForPointer(pointer, bytes)
		if !ok {
			return nil, core.E("rocm.hip.FakeLaunch", "MoE combine norms weight buffer is missing", nil)
		}
		values, err := hipFloat32PayloadValues(data[offset : offset+bytes])
		if err != nil {
			return nil, err
		}
		copy(weight, values)
	case hipRMSNormWeightEncodingBF16:
		if pointer == 0 || bytes != count*2 {
			return nil, core.E("rocm.hip.FakeLaunch", "MoE combine norms bf16 weight byte count mismatch", nil)
		}
		data, offset, ok := driver.memoryForPointer(pointer, bytes)
		if !ok {
			return nil, core.E("rocm.hip.FakeLaunch", "MoE combine norms weight buffer is missing", nil)
		}
		for index := range weight {
			weight[index] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(data[offset+index*2:]))
		}
	default:
		return nil, core.E("rocm.hip.FakeLaunch", "MoE combine norms unsupported RMSNorm weight encoding", nil)
	}
	if flags&hipRMSNormLaunchFlagAddUnitWeight != 0 {
		for index := range weight {
			weight[index] += 1
		}
	}
	return weight, nil
}

func (driver *fakeHIPDriver) launchRMSNorm(args []byte) error {
	if len(args) != hipRMSNormLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "rms norm launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipRMSNormLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipRMSNormLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "rms norm launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	count := int(binary.LittleEndian.Uint32(args[32:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[36:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[40:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[44:]))
	epsilon := math.Float32frombits(binary.LittleEndian.Uint32(args[48:]))
	encoding := binary.LittleEndian.Uint32(args[52:])
	flags := binary.LittleEndian.Uint32(args[56:])
	if count <= 0 || inputBytes != count*4 || outputBytes != count*4 {
		return core.E("rocm.hip.FakeLaunch", "rms norm shape metadata mismatch", nil)
	}
	if flags&^hipRMSNormLaunchFlagAddUnitWeight != 0 {
		return core.E("rocm.hip.FakeLaunch", "unsupported rms norm flags", nil)
	}
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		if weightPointer != 0 || weightBytes != 0 || flags != 0 {
			return core.E("rocm.hip.FakeLaunch", "rms norm unit weight metadata mismatch", nil)
		}
	case hipRMSNormWeightEncodingF32:
		if weightPointer == 0 || weightBytes != count*4 {
			return core.E("rocm.hip.FakeLaunch", "rms norm f32 weight byte count mismatch", nil)
		}
	case hipRMSNormWeightEncodingBF16:
		if weightPointer == 0 || weightBytes != count*2 {
			return core.E("rocm.hip.FakeLaunch", "rms norm bf16 weight byte count mismatch", nil)
		}
	default:
		return core.E("rocm.hip.FakeLaunch", "unsupported rms norm weight encoding", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm input buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	weight := make([]float32, count)
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		for index := range weight {
			weight[index] = 1
		}
	case hipRMSNormWeightEncodingF32:
		weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "rms norm weight buffer is missing", nil)
		}
		weight, err = hipFloat32PayloadValues(weightData[weightOffset : weightOffset+weightBytes])
		if err != nil {
			return err
		}
	case hipRMSNormWeightEncodingBF16:
		weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "rms norm weight buffer is missing", nil)
		}
		for index := range weight {
			weight[index] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(weightData[weightOffset+index*2:]))
		}
	}
	if flags&hipRMSNormLaunchFlagAddUnitWeight != 0 {
		for index := range weight {
			weight[index] += 1
		}
	}
	output, err := hipReferenceRMSNorm(input, weight, epsilon)
	if err != nil {
		return err
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchRMSNormResidualAdd(args []byte) error {
	if len(args) != hipRMSNormResidualAddArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "rms norm residual-add launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipRMSNormResidualAddArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipRMSNormResidualAddArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "rms norm residual-add launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	residualPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	count := int(binary.LittleEndian.Uint32(args[40:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[44:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[48:]))
	residualBytes := int(binary.LittleEndian.Uint32(args[52:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[56:]))
	epsilon := math.Float32frombits(binary.LittleEndian.Uint32(args[60:]))
	encoding := binary.LittleEndian.Uint32(args[64:])
	flags := binary.LittleEndian.Uint32(args[68:])
	outputScale := float32(1)
	if bits := binary.LittleEndian.Uint32(args[72:]); bits != 0 {
		outputScale = math.Float32frombits(bits)
	}
	if count <= 0 || inputBytes != count*4 || residualBytes != count*4 || outputBytes != count*4 {
		return core.E("rocm.hip.FakeLaunch", "rms norm residual-add shape metadata mismatch", nil)
	}
	if math.IsNaN(float64(outputScale)) || math.IsInf(float64(outputScale), 0) {
		return core.E("rocm.hip.FakeLaunch", "rms norm residual-add output scale must be finite", nil)
	}
	if flags&^hipRMSNormLaunchFlagAddUnitWeight != 0 {
		return core.E("rocm.hip.FakeLaunch", "unsupported rms norm residual-add flags", nil)
	}
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		if weightPointer != 0 || weightBytes != 0 || flags != 0 {
			return core.E("rocm.hip.FakeLaunch", "rms norm residual-add unit weight metadata mismatch", nil)
		}
	case hipRMSNormWeightEncodingF32:
		if weightPointer == 0 || weightBytes != count*4 {
			return core.E("rocm.hip.FakeLaunch", "rms norm residual-add f32 weight byte count mismatch", nil)
		}
	case hipRMSNormWeightEncodingBF16:
		if weightPointer == 0 || weightBytes != count*2 {
			return core.E("rocm.hip.FakeLaunch", "rms norm residual-add bf16 weight byte count mismatch", nil)
		}
	default:
		return core.E("rocm.hip.FakeLaunch", "unsupported rms norm residual-add weight encoding", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm residual-add input buffer is missing", nil)
	}
	residualData, residualOffset, ok := driver.memoryForPointer(residualPointer, residualBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm residual-add residual buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm residual-add output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	residual, err := hipFloat32PayloadValues(residualData[residualOffset : residualOffset+residualBytes])
	if err != nil {
		return err
	}
	weight := make([]float32, count)
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		for index := range weight {
			weight[index] = 1
		}
	case hipRMSNormWeightEncodingF32:
		weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "rms norm residual-add weight buffer is missing", nil)
		}
		weight, err = hipFloat32PayloadValues(weightData[weightOffset : weightOffset+weightBytes])
		if err != nil {
			return err
		}
	case hipRMSNormWeightEncodingBF16:
		weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "rms norm residual-add weight buffer is missing", nil)
		}
		for index := range weight {
			weight[index] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(weightData[weightOffset+index*2:]))
		}
	}
	if flags&hipRMSNormLaunchFlagAddUnitWeight != 0 {
		for index := range weight {
			weight[index] += 1
		}
	}
	normalized, err := hipReferenceRMSNorm(input, weight, epsilon)
	if err != nil {
		return err
	}
	for index := range normalized {
		normalized[index] = (normalized[index] + residual[index]) * outputScale
	}
	payload, err := hipFloat32Payload(normalized)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchRMSNormResidualAddNorm(args []byte) error {
	if len(args) != hipRMSNormResAddNormArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "rms norm residual-add-norm launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipRMSNormResAddNormArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipRMSNormResAddNormArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "rms norm residual-add-norm launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	residualPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	residualOutputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	normWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	normOutputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	count := int(binary.LittleEndian.Uint32(args[56:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[60:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[64:]))
	residualBytes := int(binary.LittleEndian.Uint32(args[68:]))
	residualOutputBytes := int(binary.LittleEndian.Uint32(args[72:]))
	normWeightBytes := int(binary.LittleEndian.Uint32(args[76:]))
	normOutputBytes := int(binary.LittleEndian.Uint32(args[80:]))
	epsilon := math.Float32frombits(binary.LittleEndian.Uint32(args[84:]))
	encoding := binary.LittleEndian.Uint32(args[88:])
	flags := binary.LittleEndian.Uint32(args[92:])
	normEpsilon := math.Float32frombits(binary.LittleEndian.Uint32(args[96:]))
	normEncoding := binary.LittleEndian.Uint32(args[100:])
	normFlags := binary.LittleEndian.Uint32(args[104:])
	outputScale := float32(1)
	if bits := binary.LittleEndian.Uint32(args[108:]); bits != 0 {
		outputScale = math.Float32frombits(bits)
	}
	if count <= 0 || inputBytes != count*4 || residualBytes != count*4 ||
		residualOutputBytes != count*4 || normOutputBytes != count*4 {
		return core.E("rocm.hip.FakeLaunch", "rms norm residual-add-norm shape metadata mismatch", nil)
	}
	if math.IsNaN(float64(outputScale)) || math.IsInf(float64(outputScale), 0) {
		return core.E("rocm.hip.FakeLaunch", "rms norm residual-add-norm output scale must be finite", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm residual-add-norm input buffer is missing", nil)
	}
	residualData, residualOffset, ok := driver.memoryForPointer(residualPointer, residualBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm residual-add-norm residual buffer is missing", nil)
	}
	residualOutputData, residualOutputOffset, ok := driver.memoryForPointer(residualOutputPointer, residualOutputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm residual-add-norm residual output buffer is missing", nil)
	}
	normOutputData, normOutputOffset, ok := driver.memoryForPointer(normOutputPointer, normOutputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm residual-add-norm norm output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	residual, err := hipFloat32PayloadValues(residualData[residualOffset : residualOffset+residualBytes])
	if err != nil {
		return err
	}
	weight, err := driver.rmsNormWeightValues(weightPointer, weightBytes, count, encoding, flags, "rms norm residual-add-norm weight")
	if err != nil {
		return err
	}
	normWeight, err := driver.rmsNormWeightValues(normWeightPointer, normWeightBytes, count, normEncoding, normFlags, "rms norm residual-add-norm norm weight")
	if err != nil {
		return err
	}
	normalized, err := hipReferenceRMSNorm(input, weight, epsilon)
	if err != nil {
		return err
	}
	for index := range normalized {
		normalized[index] = (normalized[index] + residual[index]) * outputScale
	}
	residualPayload, err := hipFloat32Payload(normalized)
	if err != nil {
		return err
	}
	copy(residualOutputData[residualOutputOffset:residualOutputOffset+residualOutputBytes], residualPayload)
	normOutput, err := hipReferenceRMSNorm(normalized, normWeight, normEpsilon)
	if err != nil {
		return err
	}
	normPayload, err := hipFloat32Payload(normOutput)
	if err != nil {
		return err
	}
	copy(normOutputData[normOutputOffset:normOutputOffset+normOutputBytes], normPayload)
	return nil
}

func (driver *fakeHIPDriver) rmsNormWeightValues(pointer nativeDevicePointer, bytes, count int, encoding, flags uint32, label string) ([]float32, error) {
	if flags&^hipRMSNormLaunchFlagAddUnitWeight != 0 {
		return nil, core.E("rocm.hip.FakeLaunch", "unsupported "+label+" flags", nil)
	}
	weight := make([]float32, count)
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		if pointer != 0 || bytes != 0 || flags != 0 {
			return nil, core.E("rocm.hip.FakeLaunch", label+" unit metadata mismatch", nil)
		}
		for index := range weight {
			weight[index] = 1
		}
	case hipRMSNormWeightEncodingF32:
		if pointer == 0 || bytes != count*4 {
			return nil, core.E("rocm.hip.FakeLaunch", label+" f32 byte count mismatch", nil)
		}
		weightData, weightOffset, ok := driver.memoryForPointer(pointer, bytes)
		if !ok {
			return nil, core.E("rocm.hip.FakeLaunch", label+" buffer is missing", nil)
		}
		values, err := hipFloat32PayloadValues(weightData[weightOffset : weightOffset+bytes])
		if err != nil {
			return nil, err
		}
		weight = values
	case hipRMSNormWeightEncodingBF16:
		if pointer == 0 || bytes != count*2 {
			return nil, core.E("rocm.hip.FakeLaunch", label+" bf16 byte count mismatch", nil)
		}
		weightData, weightOffset, ok := driver.memoryForPointer(pointer, bytes)
		if !ok {
			return nil, core.E("rocm.hip.FakeLaunch", label+" buffer is missing", nil)
		}
		for index := range weight {
			weight[index] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(weightData[weightOffset+index*2:]))
		}
	default:
		return nil, core.E("rocm.hip.FakeLaunch", "unsupported "+label+" encoding", nil)
	}
	if flags&hipRMSNormLaunchFlagAddUnitWeight != 0 {
		for index := range weight {
			weight[index] += 1
		}
	}
	return weight, nil
}

func (driver *fakeHIPDriver) launchRMSNormHeads(args []byte) error {
	if len(args) != hipRMSNormHeadsLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "rms norm heads launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipRMSNormHeadsLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipRMSNormHeadsLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "rms norm heads launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	headDim := int(binary.LittleEndian.Uint32(args[32:]))
	headCount := int(binary.LittleEndian.Uint32(args[36:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[40:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[44:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[48:]))
	epsilon := math.Float32frombits(binary.LittleEndian.Uint32(args[52:]))
	encoding := binary.LittleEndian.Uint32(args[56:])
	flags := binary.LittleEndian.Uint32(args[60:])
	totalCount := headDim * headCount
	if headDim <= 0 || headCount <= 0 || inputBytes != totalCount*4 || outputBytes != totalCount*4 {
		return core.E("rocm.hip.FakeLaunch", "rms norm heads shape metadata mismatch", nil)
	}
	if flags&^hipRMSNormLaunchFlagAddUnitWeight != 0 {
		return core.E("rocm.hip.FakeLaunch", "unsupported rms norm heads flags", nil)
	}
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		if weightPointer != 0 || weightBytes != 0 || flags != 0 {
			return core.E("rocm.hip.FakeLaunch", "rms norm heads unit weight metadata mismatch", nil)
		}
	case hipRMSNormWeightEncodingF32:
		if weightPointer == 0 || weightBytes != headDim*4 {
			return core.E("rocm.hip.FakeLaunch", "rms norm heads f32 weight byte count mismatch", nil)
		}
	case hipRMSNormWeightEncodingBF16:
		if weightPointer == 0 || weightBytes != headDim*2 {
			return core.E("rocm.hip.FakeLaunch", "rms norm heads bf16 weight byte count mismatch", nil)
		}
	default:
		return core.E("rocm.hip.FakeLaunch", "unsupported rms norm heads weight encoding", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm heads input buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm heads output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	weight := make([]float32, headDim)
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		for index := range weight {
			weight[index] = 1
		}
	case hipRMSNormWeightEncodingF32:
		weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "rms norm heads weight buffer is missing", nil)
		}
		weight, err = hipFloat32PayloadValues(weightData[weightOffset : weightOffset+weightBytes])
		if err != nil {
			return err
		}
	case hipRMSNormWeightEncodingBF16:
		weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "rms norm heads weight buffer is missing", nil)
		}
		for index := range weight {
			weight[index] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(weightData[weightOffset+index*2:]))
		}
	}
	if flags&hipRMSNormLaunchFlagAddUnitWeight != 0 {
		for index := range weight {
			weight[index] += 1
		}
	}
	output := make([]float32, 0, totalCount)
	for head := 0; head < headCount; head++ {
		start := head * headDim
		normalized, err := hipReferenceRMSNorm(input[start:start+headDim], weight, epsilon)
		if err != nil {
			return err
		}
		output = append(output, normalized...)
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchRMSNormRoPEHeads(args []byte) error {
	if len(args) != hipRMSNormRoPEHeadsLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipRMSNormRoPEHeadsLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipRMSNormRoPEHeadsLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	headDim := int(binary.LittleEndian.Uint32(args[32:]))
	headCount := int(binary.LittleEndian.Uint32(args[36:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[40:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[44:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[48:]))
	epsilon := math.Float32frombits(binary.LittleEndian.Uint32(args[52:]))
	encoding := binary.LittleEndian.Uint32(args[56:])
	flags := binary.LittleEndian.Uint32(args[60:])
	position := int(binary.LittleEndian.Uint32(args[64:]))
	base := math.Float32frombits(binary.LittleEndian.Uint32(args[68:]))
	frequencyDim := int(binary.LittleEndian.Uint32(args[72:]))
	rotaryCount := int(binary.LittleEndian.Uint32(args[76:]))
	frequencyScale := math.Float32frombits(binary.LittleEndian.Uint32(args[80:]))
	totalCount := headDim * headCount
	if headDim <= 0 || headDim%2 != 0 || headCount <= 0 || inputBytes != totalCount*4 || outputBytes != totalCount*4 {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads shape metadata mismatch", nil)
	}
	if frequencyDim > 0 && frequencyDim < headDim {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads frequency dimension mismatch", nil)
	}
	if rotaryCount < 0 || rotaryCount > headDim || rotaryCount%2 != 0 {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads rotary count mismatch", nil)
	}
	if rotaryCount == 0 {
		rotaryCount = headDim
	}
	if frequencyScale <= 0 || math.IsNaN(float64(frequencyScale)) || math.IsInf(float64(frequencyScale), 0) {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads frequency scale mismatch", nil)
	}
	effectiveFrequencyDim := frequencyDim
	if effectiveFrequencyDim == 0 {
		effectiveFrequencyDim = headDim
	}
	if flags&^hipRMSNormLaunchFlagMask != 0 {
		return core.E("rocm.hip.FakeLaunch", "unsupported rms norm rope heads flags", nil)
	}
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		if weightPointer != 0 || weightBytes != 0 || flags&hipRMSNormLaunchFlagAddUnitWeight != 0 {
			return core.E("rocm.hip.FakeLaunch", "rms norm rope heads unit weight metadata mismatch", nil)
		}
	case hipRMSNormWeightEncodingF32:
		if weightPointer == 0 || weightBytes != headDim*4 {
			return core.E("rocm.hip.FakeLaunch", "rms norm rope heads f32 weight byte count mismatch", nil)
		}
	case hipRMSNormWeightEncodingBF16:
		if weightPointer == 0 || weightBytes != headDim*2 {
			return core.E("rocm.hip.FakeLaunch", "rms norm rope heads bf16 weight byte count mismatch", nil)
		}
	default:
		return core.E("rocm.hip.FakeLaunch", "unsupported rms norm rope heads weight encoding", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads input buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	weight := make([]float32, headDim)
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		for index := range weight {
			weight[index] = 1
		}
	case hipRMSNormWeightEncodingF32:
		weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "rms norm rope heads weight buffer is missing", nil)
		}
		weight, err = hipFloat32PayloadValues(weightData[weightOffset : weightOffset+weightBytes])
		if err != nil {
			return err
		}
	case hipRMSNormWeightEncodingBF16:
		weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "rms norm rope heads weight buffer is missing", nil)
		}
		for index := range weight {
			weight[index] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(weightData[weightOffset+index*2:]))
		}
	}
	if flags&hipRMSNormLaunchFlagAddUnitWeight != 0 {
		for index := range weight {
			weight[index] += 1
		}
	}
	output := make([]float32, 0, totalCount)
	for head := 0; head < headCount; head++ {
		start := head * headDim
		normalized, err := hipReferenceRMSNorm(input[start:start+headDim], weight, epsilon)
		if err != nil {
			return err
		}
		var rotated []float32
		if flags&hipRMSNormLaunchFlagRoPENeoX != 0 {
			rotated, err = hipReferenceRoPENeoXWithFrequencyDimScale(normalized, position, float64(base), effectiveFrequencyDim, rotaryCount, float64(frequencyScale))
		} else {
			rotated = append([]float32(nil), normalized...)
			var rotary []float32
			rotary, err = hipReferenceRoPEWithFrequencyDimScale(normalized[:rotaryCount], position, float64(base), effectiveFrequencyDim, float64(frequencyScale))
			if err == nil {
				copy(rotated[:rotaryCount], rotary)
			}
		}
		if err != nil {
			return err
		}
		output = append(output, rotated...)
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchRMSNormRoPEHeadsPair(args []byte) error {
	if len(args) != hipRMSNormRoPEHeadsPairLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads pair launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipRMSNormRoPEHeadsPairLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipRMSNormRoPEHeadsPairLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads pair launch header mismatch", nil)
	}
	headDim := int(binary.LittleEndian.Uint32(args[56:]))
	queryHeadCount := int(binary.LittleEndian.Uint32(args[60:]))
	keyHeadCount := int(binary.LittleEndian.Uint32(args[64:]))
	position := int(binary.LittleEndian.Uint32(args[116:]))
	base := math.Float32frombits(binary.LittleEndian.Uint32(args[120:]))
	frequencyDim := int(binary.LittleEndian.Uint32(args[124:]))
	rotaryCount := int(binary.LittleEndian.Uint32(args[128:]))
	frequencyScale := math.Float32frombits(binary.LittleEndian.Uint32(args[132:]))
	queryLaunch, err := (hipRMSNormRoPEHeadsLaunchArgs{
		InputPointer:   nativeDevicePointer(binary.LittleEndian.Uint64(args[8:])),
		WeightPointer:  nativeDevicePointer(binary.LittleEndian.Uint64(args[16:])),
		OutputPointer:  nativeDevicePointer(binary.LittleEndian.Uint64(args[24:])),
		HeadDim:        headDim,
		HeadCount:      queryHeadCount,
		InputBytes:     uint64(binary.LittleEndian.Uint32(args[68:])),
		WeightBytes:    uint64(binary.LittleEndian.Uint32(args[72:])),
		OutputBytes:    uint64(binary.LittleEndian.Uint32(args[76:])),
		Epsilon:        math.Float32frombits(binary.LittleEndian.Uint32(args[92:])),
		WeightEncoding: binary.LittleEndian.Uint32(args[96:]),
		Flags:          binary.LittleEndian.Uint32(args[100:]),
		Position:       position,
		Base:           base,
		FrequencyDim:   frequencyDim,
		RotaryCount:    rotaryCount,
		FrequencyScale: frequencyScale,
	}).Binary()
	if err != nil {
		return err
	}
	if err := driver.launchRMSNormRoPEHeads(queryLaunch); err != nil {
		return err
	}
	keyLaunch, err := (hipRMSNormRoPEHeadsLaunchArgs{
		InputPointer:   nativeDevicePointer(binary.LittleEndian.Uint64(args[32:])),
		WeightPointer:  nativeDevicePointer(binary.LittleEndian.Uint64(args[40:])),
		OutputPointer:  nativeDevicePointer(binary.LittleEndian.Uint64(args[48:])),
		HeadDim:        headDim,
		HeadCount:      keyHeadCount,
		InputBytes:     uint64(binary.LittleEndian.Uint32(args[80:])),
		WeightBytes:    uint64(binary.LittleEndian.Uint32(args[84:])),
		OutputBytes:    uint64(binary.LittleEndian.Uint32(args[88:])),
		Epsilon:        math.Float32frombits(binary.LittleEndian.Uint32(args[104:])),
		WeightEncoding: binary.LittleEndian.Uint32(args[108:]),
		Flags:          binary.LittleEndian.Uint32(args[112:]),
		Position:       position,
		Base:           base,
		FrequencyDim:   frequencyDim,
		RotaryCount:    rotaryCount,
		FrequencyScale: frequencyScale,
	}).Binary()
	if err != nil {
		return err
	}
	return driver.launchRMSNormRoPEHeads(keyLaunch)
}

func (driver *fakeHIPDriver) launchRMSNormRoPEHeadsPairLaneBatch(args []byte) error {
	if len(args) != hipRMSNormRoPEHeadsPairLaneBatchLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads pair lane batch launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipRMSNormRoPEHeadsPairLaneBatchLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipRMSNormRoPEHeadsPairLaneBatchLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads pair lane batch launch header mismatch", nil)
	}
	headDim := int(binary.LittleEndian.Uint32(args[64:]))
	queryHeadCount := int(binary.LittleEndian.Uint32(args[68:]))
	keyHeadCount := int(binary.LittleEndian.Uint32(args[72:]))
	batch := int(binary.LittleEndian.Uint32(args[76:]))
	queryInputBytes := int(binary.LittleEndian.Uint32(args[80:]))
	queryWeightBytes := int(binary.LittleEndian.Uint32(args[84:]))
	queryOutputBytes := int(binary.LittleEndian.Uint32(args[88:]))
	keyInputBytes := int(binary.LittleEndian.Uint32(args[92:]))
	keyWeightBytes := int(binary.LittleEndian.Uint32(args[96:]))
	keyOutputBytes := int(binary.LittleEndian.Uint32(args[100:]))
	positionsPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[56:]))
	positionsBytes := int(binary.LittleEndian.Uint32(args[128:]))
	if headDim <= 0 || headDim%2 != 0 || queryHeadCount <= 0 || keyHeadCount <= 0 || batch <= 0 ||
		queryInputBytes != headDim*queryHeadCount*batch*4 || queryOutputBytes != headDim*queryHeadCount*batch*4 ||
		keyInputBytes != headDim*keyHeadCount*batch*4 || keyOutputBytes != headDim*keyHeadCount*batch*4 ||
		positionsPointer == 0 || positionsBytes != batch*4 {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads pair lane batch shape metadata mismatch", nil)
	}
	positionsData, positionsOffset, ok := driver.memoryForPointer(positionsPointer, positionsBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads pair lane batch positions buffer is missing", nil)
	}
	queryInputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	queryWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	queryOutputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	keyInputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	keyWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	keyOutputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	queryRowBytes := headDim * queryHeadCount * 4
	keyRowBytes := headDim * keyHeadCount * 4
	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		position := int(binary.LittleEndian.Uint32(positionsData[positionsOffset+batchIndex*4:]))
		launchBytes, err := (hipRMSNormRoPEHeadsPairLaunchArgs{
			QueryInputPointer:   nativeDevicePointer(uint64(queryInputPointer) + uint64(batchIndex*queryRowBytes)),
			QueryWeightPointer:  queryWeightPointer,
			QueryOutputPointer:  nativeDevicePointer(uint64(queryOutputPointer) + uint64(batchIndex*queryRowBytes)),
			KeyInputPointer:     nativeDevicePointer(uint64(keyInputPointer) + uint64(batchIndex*keyRowBytes)),
			KeyWeightPointer:    keyWeightPointer,
			KeyOutputPointer:    nativeDevicePointer(uint64(keyOutputPointer) + uint64(batchIndex*keyRowBytes)),
			HeadDim:             headDim,
			QueryHeadCount:      queryHeadCount,
			KeyHeadCount:        keyHeadCount,
			QueryInputBytes:     uint64(queryRowBytes),
			QueryWeightBytes:    uint64(queryWeightBytes),
			QueryOutputBytes:    uint64(queryRowBytes),
			KeyInputBytes:       uint64(keyRowBytes),
			KeyWeightBytes:      uint64(keyWeightBytes),
			KeyOutputBytes:      uint64(keyRowBytes),
			QueryEpsilon:        math.Float32frombits(binary.LittleEndian.Uint32(args[104:])),
			QueryWeightEncoding: binary.LittleEndian.Uint32(args[108:]),
			QueryFlags:          binary.LittleEndian.Uint32(args[112:]),
			KeyEpsilon:          math.Float32frombits(binary.LittleEndian.Uint32(args[116:])),
			KeyWeightEncoding:   binary.LittleEndian.Uint32(args[120:]),
			KeyFlags:            binary.LittleEndian.Uint32(args[124:]),
			Position:            position,
			Base:                math.Float32frombits(binary.LittleEndian.Uint32(args[132:])),
			FrequencyDim:        int(binary.LittleEndian.Uint32(args[136:])),
			RotaryCount:         int(binary.LittleEndian.Uint32(args[140:])),
			FrequencyScale:      math.Float32frombits(binary.LittleEndian.Uint32(args[144:])),
		}).BinaryInto(make([]byte, hipRMSNormRoPEHeadsPairLaunchArgsBytes))
		if err != nil {
			return err
		}
		if err := driver.launchRMSNormRoPEHeadsPair(launchBytes); err != nil {
			return err
		}
	}
	return nil
}

func (driver *fakeHIPDriver) launchRMSNormRoPEHeadsBatch(args []byte) error {
	if len(args) != hipRMSNormRoPEHeadsBatchLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads batch launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipRMSNormRoPEHeadsBatchLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipRMSNormRoPEHeadsBatchLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads batch launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	headDim := int(binary.LittleEndian.Uint32(args[32:]))
	headCount := int(binary.LittleEndian.Uint32(args[36:]))
	batch := int(binary.LittleEndian.Uint32(args[40:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[44:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[48:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[52:]))
	epsilon := math.Float32frombits(binary.LittleEndian.Uint32(args[56:]))
	encoding := binary.LittleEndian.Uint32(args[60:])
	flags := binary.LittleEndian.Uint32(args[64:])
	startPosition := int(binary.LittleEndian.Uint32(args[68:]))
	base := math.Float32frombits(binary.LittleEndian.Uint32(args[72:]))
	frequencyDim := int(binary.LittleEndian.Uint32(args[76:]))
	rotaryCount := int(binary.LittleEndian.Uint32(args[80:]))
	frequencyScale := math.Float32frombits(binary.LittleEndian.Uint32(args[84:]))
	totalCount := headDim * headCount * batch
	if headDim <= 0 || headDim%2 != 0 || headCount <= 0 || batch <= 0 || inputBytes != totalCount*4 || outputBytes != totalCount*4 {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads batch shape metadata mismatch", nil)
	}
	if frequencyDim > 0 && frequencyDim < headDim {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads batch frequency dimension mismatch", nil)
	}
	if rotaryCount < 0 || rotaryCount > headDim || rotaryCount%2 != 0 {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads batch rotary count mismatch", nil)
	}
	if rotaryCount == 0 {
		rotaryCount = headDim
	}
	if frequencyScale <= 0 || math.IsNaN(float64(frequencyScale)) || math.IsInf(float64(frequencyScale), 0) {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads batch frequency scale mismatch", nil)
	}
	effectiveFrequencyDim := frequencyDim
	if effectiveFrequencyDim == 0 {
		effectiveFrequencyDim = headDim
	}
	if flags&^hipRMSNormLaunchFlagMask != 0 {
		return core.E("rocm.hip.FakeLaunch", "unsupported rms norm rope heads batch flags", nil)
	}
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		if weightPointer != 0 || weightBytes != 0 || flags&hipRMSNormLaunchFlagAddUnitWeight != 0 {
			return core.E("rocm.hip.FakeLaunch", "rms norm rope heads batch unit weight metadata mismatch", nil)
		}
	case hipRMSNormWeightEncodingF32:
		if weightPointer == 0 || weightBytes != headDim*4 {
			return core.E("rocm.hip.FakeLaunch", "rms norm rope heads batch f32 weight byte count mismatch", nil)
		}
	case hipRMSNormWeightEncodingBF16:
		if weightPointer == 0 || weightBytes != headDim*2 {
			return core.E("rocm.hip.FakeLaunch", "rms norm rope heads batch bf16 weight byte count mismatch", nil)
		}
	default:
		return core.E("rocm.hip.FakeLaunch", "unsupported rms norm rope heads batch weight encoding", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads batch input buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm rope heads batch output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	weight := make([]float32, headDim)
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		for index := range weight {
			weight[index] = 1
		}
	case hipRMSNormWeightEncodingF32:
		weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "rms norm rope heads batch weight buffer is missing", nil)
		}
		weight, err = hipFloat32PayloadValues(weightData[weightOffset : weightOffset+weightBytes])
		if err != nil {
			return err
		}
	case hipRMSNormWeightEncodingBF16:
		weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "rms norm rope heads batch weight buffer is missing", nil)
		}
		for index := range weight {
			weight[index] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(weightData[weightOffset+index*2:]))
		}
	}
	if flags&hipRMSNormLaunchFlagAddUnitWeight != 0 {
		for index := range weight {
			weight[index] += 1
		}
	}
	output := make([]float32, 0, totalCount)
	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for head := 0; head < headCount; head++ {
			start := (batchIndex*headCount + head) * headDim
			normalized, err := hipReferenceRMSNorm(input[start:start+headDim], weight, epsilon)
			if err != nil {
				return err
			}
			var rotated []float32
			if flags&hipRMSNormLaunchFlagRoPENeoX != 0 {
				rotated, err = hipReferenceRoPENeoXWithFrequencyDimScale(normalized, startPosition+batchIndex, float64(base), effectiveFrequencyDim, rotaryCount, float64(frequencyScale))
			} else {
				rotated = append([]float32(nil), normalized...)
				var rotary []float32
				rotary, err = hipReferenceRoPEWithFrequencyDimScale(normalized[:rotaryCount], startPosition+batchIndex, float64(base), effectiveFrequencyDim, float64(frequencyScale))
				if err == nil {
					copy(rotated[:rotaryCount], rotary)
				}
			}
			if err != nil {
				return err
			}
			output = append(output, rotated...)
		}
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchRoPE(args []byte) error {
	if len(args) != hipRoPELaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "rope launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipRoPELaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipRoPELaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "rope launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	count := int(binary.LittleEndian.Uint32(args[24:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[28:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[32:]))
	position := int(binary.LittleEndian.Uint32(args[36:]))
	base := math.Float32frombits(binary.LittleEndian.Uint32(args[40:]))
	frequencyDim := int(binary.LittleEndian.Uint32(args[44:]))
	rotaryCount := int(binary.LittleEndian.Uint32(args[48:]))
	if count <= 0 || count%2 != 0 || inputBytes != count*4 || outputBytes != count*4 {
		return core.E("rocm.hip.FakeLaunch", "rope shape metadata mismatch", nil)
	}
	if frequencyDim > 0 && frequencyDim < count {
		return core.E("rocm.hip.FakeLaunch", "rope frequency dimension mismatch", nil)
	}
	if rotaryCount < 0 || rotaryCount > count || rotaryCount%2 != 0 {
		return core.E("rocm.hip.FakeLaunch", "rope rotary count mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rope input buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rope output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	if rotaryCount == 0 {
		rotaryCount = count
	}
	output := append([]float32(nil), input...)
	rotated, err := hipReferenceRoPEWithFrequencyDim(input[:rotaryCount], position, float64(base), frequencyDim)
	if err != nil {
		return err
	}
	copy(output[:rotaryCount], rotated)
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchRoPEHeads(args []byte) error {
	if len(args) != hipRoPEHeadsLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "rope heads launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipRoPEHeadsLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipRoPEHeadsLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "rope heads launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	headDim := int(binary.LittleEndian.Uint32(args[24:]))
	headCount := int(binary.LittleEndian.Uint32(args[28:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[32:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[36:]))
	position := int(binary.LittleEndian.Uint32(args[40:]))
	base := math.Float32frombits(binary.LittleEndian.Uint32(args[44:]))
	frequencyDim := int(binary.LittleEndian.Uint32(args[48:]))
	rotaryCount := int(binary.LittleEndian.Uint32(args[52:]))
	totalCount := headDim * headCount
	if headDim <= 0 || headDim%2 != 0 || headCount <= 0 || inputBytes != totalCount*4 || outputBytes != totalCount*4 {
		return core.E("rocm.hip.FakeLaunch", "rope heads shape metadata mismatch", nil)
	}
	if frequencyDim > 0 && frequencyDim < headDim {
		return core.E("rocm.hip.FakeLaunch", "rope heads frequency dimension mismatch", nil)
	}
	if rotaryCount < 0 || rotaryCount > headDim || rotaryCount%2 != 0 {
		return core.E("rocm.hip.FakeLaunch", "rope heads rotary count mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rope heads input buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rope heads output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	if rotaryCount == 0 {
		rotaryCount = headDim
	}
	effectiveFrequencyDim := frequencyDim
	if effectiveFrequencyDim == 0 {
		effectiveFrequencyDim = headDim
	}
	output := append([]float32(nil), input...)
	for head := 0; head < headCount; head++ {
		start := head * headDim
		rotated, err := hipReferenceRoPEWithFrequencyDim(input[start:start+rotaryCount], position, float64(base), effectiveFrequencyDim)
		if err != nil {
			return err
		}
		copy(output[start:start+rotaryCount], rotated)
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchGreedySample(args []byte) error {
	if len(args) != hipGreedyLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "greedy launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipGreedyLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipGreedyLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "greedy launch header mismatch", nil)
	}
	logitsPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	count := int(binary.LittleEndian.Uint32(args[24:]))
	logitsBytes := int(binary.LittleEndian.Uint32(args[28:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[32:]))
	if count <= 0 || logitsBytes != count*4 || outputBytes != hipGreedyResultBytes {
		return core.E("rocm.hip.FakeLaunch", "greedy shape metadata mismatch", nil)
	}
	logitsData, logitsOffset, ok := driver.memoryForPointer(logitsPointer, logitsBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "greedy logits buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "greedy output buffer is missing", nil)
	}
	logits, err := hipFloat32PayloadValues(logitsData[logitsOffset : logitsOffset+logitsBytes])
	if err != nil {
		return err
	}
	index, score, err := hipReferenceGreedySample(logits)
	if err != nil {
		return err
	}
	binary.LittleEndian.PutUint32(outputData[outputOffset:], uint32(int32(index)))
	binary.LittleEndian.PutUint32(outputData[outputOffset+4:], math.Float32bits(score))
	return nil
}

func (driver *fakeHIPDriver) launchSoftcapGreedySample(args []byte) error {
	if len(args) != hipSoftcapGreedyLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "softcap greedy launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipSoftcapGreedyLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipSoftcapGreedyLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "softcap greedy launch header mismatch", nil)
	}
	logitsPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	count := int(binary.LittleEndian.Uint32(args[24:]))
	logitsBytes := int(binary.LittleEndian.Uint32(args[28:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[32:]))
	softcap := math.Float32frombits(binary.LittleEndian.Uint32(args[36:]))
	suppressPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	suppressCount := int(binary.LittleEndian.Uint32(args[48:]))
	if count <= 0 || logitsBytes != count*4 || outputBytes != hipGreedyResultBytes ||
		softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) ||
		(suppressCount > 0 && (suppressPointer == 0 || suppressCount >= count)) {
		return core.E("rocm.hip.FakeLaunch", "softcap greedy shape metadata mismatch", nil)
	}
	logitsData, logitsOffset, ok := driver.memoryForPointer(logitsPointer, logitsBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "softcap greedy logits buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "softcap greedy output buffer is missing", nil)
	}
	logits, err := hipFloat32PayloadValues(logitsData[logitsOffset : logitsOffset+logitsBytes])
	if err != nil {
		return err
	}
	var suppressTokens []int32
	if suppressCount > 0 {
		suppressData, suppressOffset, ok := driver.memoryForPointer(suppressPointer, suppressCount*4)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "softcap greedy suppress buffer is missing", nil)
		}
		suppressTokens = make([]int32, suppressCount)
		for index := range suppressTokens {
			suppressTokens[index] = int32(binary.LittleEndian.Uint32(suppressData[suppressOffset+index*4:]))
		}
	}
	index, score, err := hipReferenceGreedySampleSuppress(logits, suppressTokens)
	if err != nil {
		return err
	}
	if softcap > 0 {
		score = float32(math.Tanh(float64(score/softcap))) * softcap
	}
	binary.LittleEndian.PutUint32(outputData[outputOffset:], uint32(int32(index)))
	binary.LittleEndian.PutUint32(outputData[outputOffset+4:], math.Float32bits(score))
	return nil
}

func (driver *fakeHIPDriver) launchAttention(args []byte) error {
	if len(args) != hipAttentionLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "attention launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipAttentionLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipAttentionLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "attention launch header mismatch", nil)
	}
	queryPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	keyPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	valuePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	dim := int(binary.LittleEndian.Uint32(args[48:]))
	tokenCount := int(binary.LittleEndian.Uint32(args[52:]))
	queryBytes := int(binary.LittleEndian.Uint32(args[56:]))
	keyBytes := int(binary.LittleEndian.Uint32(args[60:]))
	valueBytes := int(binary.LittleEndian.Uint32(args[64:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[68:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[72:]))
	kvSource := binary.LittleEndian.Uint32(args[76:])
	scale := math.Float32frombits(binary.LittleEndian.Uint32(args[80:]))
	descriptorPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[88:]))
	descriptorBytes := int(binary.LittleEndian.Uint64(args[96:]))
	if dim <= 0 || tokenCount <= 0 || queryBytes != dim*4 || outputBytes != dim*4 || weightBytes != tokenCount*4 {
		return core.E("rocm.hip.FakeLaunch", "attention shape metadata mismatch", nil)
	}
	if scale < 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return core.E("rocm.hip.FakeLaunch", "attention scale is invalid", nil)
	}
	queryData, queryOffset, ok := driver.memoryForPointer(queryPointer, queryBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention query buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention output buffer is missing", nil)
	}
	weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention weight buffer is missing", nil)
	}
	query, err := hipFloat32PayloadValues(queryData[queryOffset : queryOffset+queryBytes])
	if err != nil {
		return err
	}
	var keyFlat []float32
	var valueFlat []float32
	switch kvSource {
	case hipAttentionKVSourceContiguous:
		if keyBytes != dim*tokenCount*4 || valueBytes != dim*tokenCount*4 {
			return core.E("rocm.hip.FakeLaunch", "attention shape metadata mismatch", nil)
		}
		keyData, keyOffset, ok := driver.memoryForPointer(keyPointer, keyBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention key buffer is missing", nil)
		}
		valueData, valueOffset, ok := driver.memoryForPointer(valuePointer, valueBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention value buffer is missing", nil)
		}
		keyFlat, err = hipFloat32PayloadValues(keyData[keyOffset : keyOffset+keyBytes])
		if err != nil {
			return err
		}
		valueFlat, err = hipFloat32PayloadValues(valueData[valueOffset : valueOffset+valueBytes])
		if err != nil {
			return err
		}
	case hipAttentionKVSourceDevice:
		if keyPointer != 0 || valuePointer != 0 || keyBytes != 0 || valueBytes != 0 {
			return core.E("rocm.hip.FakeLaunch", "attention device KV source must not include contiguous KV buffers", nil)
		}
		keyFlat, valueFlat, err = driver.readDeviceKVDescriptorForAttention(descriptorPointer, descriptorBytes, tokenCount, dim)
		if err != nil {
			return err
		}
	default:
		return core.E("rocm.hip.FakeLaunch", "attention KV source is unsupported", nil)
	}
	keys, err := splitHIPReferenceVectors(keyFlat, dim)
	if err != nil {
		return err
	}
	values, err := splitHIPReferenceVectors(valueFlat, dim)
	if err != nil {
		return err
	}
	output, weights, err := hipReferenceSingleHeadAttentionWithScale(query, keys, values, scale)
	if err != nil {
		return err
	}
	outputPayload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	weightPayload, err := hipFloat32Payload(weights)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], outputPayload)
	copy(weightData[weightOffset:weightOffset+weightBytes], weightPayload)
	return nil
}

func (driver *fakeHIPDriver) launchAttentionHeads(args []byte) error {
	if len(args) != hipAttentionHeadsLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "attention heads launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipAttentionHeadsLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipAttentionHeadsLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "attention heads launch header mismatch", nil)
	}
	queryPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	keyPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	valuePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	dim := int(binary.LittleEndian.Uint32(args[48:]))
	tokenCount := int(binary.LittleEndian.Uint32(args[52:]))
	headCount := int(binary.LittleEndian.Uint32(args[56:]))
	queryBytes := int(binary.LittleEndian.Uint32(args[60:]))
	keyBytes := int(binary.LittleEndian.Uint32(args[64:]))
	valueBytes := int(binary.LittleEndian.Uint32(args[68:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[72:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[76:]))
	kvSource := binary.LittleEndian.Uint32(args[80:])
	scale := math.Float32frombits(binary.LittleEndian.Uint32(args[84:]))
	descriptorPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[88:]))
	descriptorBytes := int(binary.LittleEndian.Uint64(args[96:]))
	keyHeads := int(binary.LittleEndian.Uint32(args[116:]))
	if dim <= 0 || tokenCount <= 0 || headCount <= 0 ||
		keyHeads <= 0 || keyHeads > headCount || headCount%keyHeads != 0 ||
		queryBytes != headCount*dim*4 ||
		outputBytes != headCount*dim*4 {
		return core.E("rocm.hip.FakeLaunch", "attention heads shape metadata mismatch", nil)
	}
	useSharedWeights := weightPointer == 0
	if useSharedWeights {
		if weightBytes != 0 || tokenCount > hipAttentionHeadsSharedMaxTokens {
			return core.E("rocm.hip.FakeLaunch", "attention heads shared weight metadata mismatch", nil)
		}
	} else if weightBytes != headCount*tokenCount*4 {
		return core.E("rocm.hip.FakeLaunch", "attention heads weight metadata mismatch", nil)
	}
	if scale < 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return core.E("rocm.hip.FakeLaunch", "attention heads scale is invalid", nil)
	}
	queryData, queryOffset, ok := driver.memoryForPointer(queryPointer, queryBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention heads query buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention heads output buffer is missing", nil)
	}
	var weightData []byte
	var weightOffset int
	if !useSharedWeights {
		var ok bool
		weightData, weightOffset, ok = driver.memoryForPointer(weightPointer, weightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention heads weight buffer is missing", nil)
		}
	}
	var keyFlat []float32
	var valueFlat []float32
	var err error
	switch kvSource {
	case hipAttentionKVSourceContiguous:
		if keyBytes != dim*tokenCount*keyHeads*4 || valueBytes != dim*tokenCount*keyHeads*4 {
			return core.E("rocm.hip.FakeLaunch", "attention heads shape metadata mismatch", nil)
		}
		keyData, keyOffset, ok := driver.memoryForPointer(keyPointer, keyBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention heads key buffer is missing", nil)
		}
		valueData, valueOffset, ok := driver.memoryForPointer(valuePointer, valueBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention heads value buffer is missing", nil)
		}
		keyFlat, err = hipFloat32PayloadValues(keyData[keyOffset : keyOffset+keyBytes])
		if err != nil {
			return err
		}
		valueFlat, err = hipFloat32PayloadValues(valueData[valueOffset : valueOffset+valueBytes])
		if err != nil {
			return err
		}
	case hipAttentionKVSourceDevice:
		if keyPointer != 0 || valuePointer != 0 || keyBytes != 0 || valueBytes != 0 {
			return core.E("rocm.hip.FakeLaunch", "attention heads device KV source must not include contiguous KV buffers", nil)
		}
		keyFlat, valueFlat, err = driver.readDeviceKVDescriptorForAttention(descriptorPointer, descriptorBytes, tokenCount, dim*keyHeads)
		if err != nil {
			return err
		}
	default:
		return core.E("rocm.hip.FakeLaunch", "attention heads KV source is unsupported", nil)
	}
	for head := 0; head < headCount; head++ {
		queryStart := queryOffset + head*dim*4
		query, err := hipFloat32PayloadValues(queryData[queryStart : queryStart+dim*4])
		if err != nil {
			return err
		}
		keys, err := fakeROCmAttentionHeadVectors(keyFlat, tokenCount, keyHeads, dim, headCount, head)
		if err != nil {
			return err
		}
		values, err := fakeROCmAttentionHeadVectors(valueFlat, tokenCount, keyHeads, dim, headCount, head)
		if err != nil {
			return err
		}
		output, weights, err := hipReferenceSingleHeadAttentionWithScale(query, keys, values, scale)
		if err != nil {
			return err
		}
		outputPayload, err := hipFloat32Payload(output)
		if err != nil {
			return err
		}
		copy(outputData[outputOffset+head*dim*4:outputOffset+(head+1)*dim*4], outputPayload)
		if !useSharedWeights {
			weightPayload, err := hipFloat32Payload(weights)
			if err != nil {
				return err
			}
			copy(weightData[weightOffset+head*tokenCount*4:weightOffset+(head+1)*tokenCount*4], weightPayload)
		}
	}
	return nil
}

func fakeROCmAttentionHeadVectors(flat []float32, tokenCount, keyHeads, dim, headCount, head int) ([][]float32, error) {
	if tokenCount <= 0 || keyHeads <= 0 || dim <= 0 || headCount <= 0 || head < 0 || head >= headCount || headCount%keyHeads != 0 {
		return nil, core.E("rocm.hip.FakeLaunch", "attention heads GQA metadata mismatch", nil)
	}
	if len(flat) != tokenCount*keyHeads*dim {
		return nil, core.E("rocm.hip.FakeLaunch", "attention heads GQA tensor shape mismatch", nil)
	}
	kvHead := head / (headCount / keyHeads)
	out := make([][]float32, 0, tokenCount)
	for token := 0; token < tokenCount; token++ {
		start := (token*keyHeads + kvHead) * dim
		out = append(out, flat[start:start+dim])
	}
	return out, nil
}

func (driver *fakeHIPDriver) launchAttentionHeadsBatchCausal(args []byte) error {
	if len(args) != hipAttentionHeadsBatchCausalLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipAttentionHeadsBatchCausalLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipAttentionHeadsBatchCausalLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal launch header mismatch", nil)
	}
	queryPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	keyPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	valuePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	dim := int(binary.LittleEndian.Uint32(args[48:]))
	tokenCount := int(binary.LittleEndian.Uint32(args[52:]))
	headCount := int(binary.LittleEndian.Uint32(args[56:]))
	queryCount := int(binary.LittleEndian.Uint32(args[60:]))
	queryStartToken := int(binary.LittleEndian.Uint32(args[64:]))
	queryBytes := int(binary.LittleEndian.Uint32(args[68:]))
	keyBytes := int(binary.LittleEndian.Uint32(args[72:]))
	valueBytes := int(binary.LittleEndian.Uint32(args[76:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[80:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[84:]))
	kvSource := binary.LittleEndian.Uint32(args[88:])
	scale := math.Float32frombits(binary.LittleEndian.Uint32(args[92:]))
	descriptorPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[96:]))
	descriptorBytes := int(binary.LittleEndian.Uint64(args[104:]))
	windowSize := int(binary.LittleEndian.Uint32(args[120:]))
	keyHeads := int(binary.LittleEndian.Uint32(args[124:]))
	visibleCapPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[128:]))
	visibleCapBytes := int(binary.LittleEndian.Uint64(args[136:]))
	if dim <= 0 || tokenCount <= 0 || headCount <= 0 || queryCount <= 0 ||
		keyHeads <= 0 || keyHeads > headCount || headCount%keyHeads != 0 ||
		queryStartToken < 0 || windowSize < 0 || uint64(queryStartToken)+uint64(queryCount) > uint64(tokenCount) ||
		queryBytes != queryCount*headCount*dim*4 ||
		outputBytes != queryCount*headCount*dim*4 {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal shape metadata mismatch", nil)
	}
	if (visibleCapPointer == 0) != (visibleCapBytes == 0) || (visibleCapPointer != 0 && visibleCapBytes != queryCount*4) {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch visible cap metadata mismatch", nil)
	}
	useSharedWeights := weightPointer == 0
	if useSharedWeights {
		if weightBytes != 0 || tokenCount > hipAttentionHeadsSharedMaxTokens {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal shared weight metadata mismatch", nil)
		}
	} else if weightBytes != queryCount*headCount*tokenCount*4 {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal weight metadata mismatch", nil)
	}
	if scale < 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal scale is invalid", nil)
	}
	queryData, queryOffset, ok := driver.memoryForPointer(queryPointer, queryBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal output buffer is missing", nil)
	}
	var weightData []byte
	var weightOffset int
	if !useSharedWeights {
		var ok bool
		weightData, weightOffset, ok = driver.memoryForPointer(weightPointer, weightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal weight buffer is missing", nil)
		}
	}
	var keyFlat []float32
	var valueFlat []float32
	var visibleCaps []int32
	var err error
	if visibleCapPointer != 0 {
		capData, capOffset, ok := driver.memoryForPointer(visibleCapPointer, visibleCapBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch visible cap buffer is missing", nil)
		}
		visibleCaps = make([]int32, queryCount)
		for index := range visibleCaps {
			visibleCaps[index] = int32(binary.LittleEndian.Uint32(capData[capOffset+index*4:]))
		}
	}
	switch kvSource {
	case hipAttentionKVSourceContiguous:
		if keyBytes != dim*tokenCount*keyHeads*4 || valueBytes != dim*tokenCount*keyHeads*4 {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal shape metadata mismatch", nil)
		}
		keyData, keyOffset, ok := driver.memoryForPointer(keyPointer, keyBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal key buffer is missing", nil)
		}
		valueData, valueOffset, ok := driver.memoryForPointer(valuePointer, valueBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal value buffer is missing", nil)
		}
		keyFlat, err = hipFloat32PayloadValues(keyData[keyOffset : keyOffset+keyBytes])
		if err != nil {
			return err
		}
		valueFlat, err = hipFloat32PayloadValues(valueData[valueOffset : valueOffset+valueBytes])
		if err != nil {
			return err
		}
	case hipAttentionKVSourceDevice:
		if keyPointer != 0 || valuePointer != 0 || keyBytes != 0 || valueBytes != 0 {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal device KV source must not include contiguous KV buffers", nil)
		}
		keyFlat, valueFlat, err = driver.readDeviceKVDescriptorForAttention(descriptorPointer, descriptorBytes, tokenCount, dim*keyHeads)
		if err != nil {
			return err
		}
	default:
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal KV source is unsupported", nil)
	}
	for queryIndex := 0; queryIndex < queryCount; queryIndex++ {
		visibleTokens := queryStartToken + queryIndex + 1
		if len(visibleCaps) > 0 {
			visibleTokens = int(visibleCaps[queryIndex])
		}
		if visibleTokens <= 0 || visibleTokens > tokenCount {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch visible cap is out of range", nil)
		}
		windowStart := 0
		if windowSize > 0 && visibleTokens > windowSize {
			windowStart = visibleTokens - windowSize
		}
		for head := 0; head < headCount; head++ {
			baseIndex := queryIndex*headCount + head
			queryStart := queryOffset + baseIndex*dim*4
			query, err := hipFloat32PayloadValues(queryData[queryStart : queryStart+dim*4])
			if err != nil {
				return err
			}
			keys, err := fakeROCmAttentionHeadVectors(keyFlat, tokenCount, keyHeads, dim, headCount, head)
			if err != nil {
				return err
			}
			values, err := fakeROCmAttentionHeadVectors(valueFlat, tokenCount, keyHeads, dim, headCount, head)
			if err != nil {
				return err
			}
			output, weights, err := hipReferenceSingleHeadAttentionWithScale(query, keys[windowStart:visibleTokens], values[windowStart:visibleTokens], scale)
			if err != nil {
				return err
			}
			outputPayload, err := hipFloat32Payload(output)
			if err != nil {
				return err
			}
			copy(outputData[outputOffset+baseIndex*dim*4:outputOffset+(baseIndex+1)*dim*4], outputPayload)
			if !useSharedWeights {
				weightPayload, err := hipFloat32Payload(weights)
				if err != nil {
					return err
				}
				weightStart := weightOffset + baseIndex*tokenCount*4
				copy(weightData[weightStart+windowStart*4:weightStart+visibleTokens*4], weightPayload)
			}
		}
	}
	return nil
}

func (driver *fakeHIPDriver) launchAttentionHeadsLaneBatch(args []byte) error {
	if len(args) != hipAttentionHeadsLaneBatchLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "attention heads lane batch launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipAttentionHeadsLaneBatchLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipAttentionHeadsLaneBatchLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "attention heads lane batch launch header mismatch", nil)
	}
	queryPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	lanePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	dim := int(binary.LittleEndian.Uint32(args[40:]))
	laneCount := int(binary.LittleEndian.Uint32(args[44:]))
	headCount := int(binary.LittleEndian.Uint32(args[48:]))
	keyHeads := int(binary.LittleEndian.Uint32(args[52:]))
	maxTokenCount := int(binary.LittleEndian.Uint32(args[56:]))
	queryBytes := int(binary.LittleEndian.Uint32(args[60:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[64:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[68:]))
	laneBytes := int(binary.LittleEndian.Uint32(args[72:]))
	scale := math.Float32frombits(binary.LittleEndian.Uint32(args[76:]))
	if dim <= 0 || laneCount <= 0 || headCount <= 0 || keyHeads <= 0 ||
		keyHeads > headCount || headCount%keyHeads != 0 || maxTokenCount <= 0 ||
		queryBytes != laneCount*headCount*dim*4 || outputBytes != queryBytes ||
		laneBytes != laneCount*hipAttentionHeadsLaneDescriptorBytes {
		return core.E("rocm.hip.FakeLaunch", "attention heads lane batch shape metadata mismatch", nil)
	}
	useSharedWeights := weightPointer == 0
	if useSharedWeights {
		if weightBytes != 0 || maxTokenCount > hipAttentionHeadsSharedMaxTokens {
			return core.E("rocm.hip.FakeLaunch", "attention heads lane batch shared weight metadata mismatch", nil)
		}
	} else if weightBytes != laneCount*headCount*maxTokenCount*4 {
		return core.E("rocm.hip.FakeLaunch", "attention heads lane batch weight metadata mismatch", nil)
	}
	if scale < 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return core.E("rocm.hip.FakeLaunch", "attention heads lane batch scale is invalid", nil)
	}
	queryData, queryOffset, ok := driver.memoryForPointer(queryPointer, queryBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention heads lane batch query buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention heads lane batch output buffer is missing", nil)
	}
	laneData, laneOffset, ok := driver.memoryForPointer(lanePointer, laneBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention heads lane descriptor buffer is missing", nil)
	}
	var weightData []byte
	var weightOffset int
	if !useSharedWeights {
		weightData, weightOffset, ok = driver.memoryForPointer(weightPointer, weightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention heads lane batch weight buffer is missing", nil)
		}
	}
	for lane := 0; lane < laneCount; lane++ {
		descriptorOffset := laneOffset + lane*hipAttentionHeadsLaneDescriptorBytes
		descriptorPointer := nativeDevicePointer(binary.LittleEndian.Uint64(laneData[descriptorOffset:]))
		descriptorBytes := int(binary.LittleEndian.Uint64(laneData[descriptorOffset+8:]))
		tokenCount := int(binary.LittleEndian.Uint32(laneData[descriptorOffset+16:]))
		windowSize := int(binary.LittleEndian.Uint32(laneData[descriptorOffset+20:]))
		if tokenCount <= 0 || tokenCount > maxTokenCount || windowSize < 0 {
			return core.E("rocm.hip.FakeLaunch", "attention heads lane descriptor metadata mismatch", nil)
		}
		keyFlat, valueFlat, err := driver.readDeviceKVDescriptorForAttention(descriptorPointer, descriptorBytes, tokenCount, dim*keyHeads)
		if err != nil {
			return err
		}
		windowStart := 0
		if windowSize > 0 && tokenCount > windowSize {
			windowStart = tokenCount - windowSize
		}
		for head := 0; head < headCount; head++ {
			baseIndex := lane*headCount + head
			queryStart := queryOffset + baseIndex*dim*4
			query, err := hipFloat32PayloadValues(queryData[queryStart : queryStart+dim*4])
			if err != nil {
				return err
			}
			keys, err := fakeROCmAttentionHeadVectors(keyFlat, tokenCount, keyHeads, dim, headCount, head)
			if err != nil {
				return err
			}
			values, err := fakeROCmAttentionHeadVectors(valueFlat, tokenCount, keyHeads, dim, headCount, head)
			if err != nil {
				return err
			}
			result, weights, err := hipReferenceSingleHeadAttentionWithScale(query, keys[windowStart:], values[windowStart:], scale)
			if err != nil {
				return err
			}
			resultPayload, err := hipFloat32Payload(result)
			if err != nil {
				return err
			}
			copy(outputData[outputOffset+baseIndex*dim*4:outputOffset+(baseIndex+1)*dim*4], resultPayload)
			if !useSharedWeights {
				weightPayload, err := hipFloat32Payload(weights)
				if err != nil {
					return err
				}
				weightStart := weightOffset + baseIndex*maxTokenCount*4 + windowStart*4
				copy(weightData[weightStart:weightStart+len(weightPayload)], weightPayload)
			}
		}
	}
	return nil
}

func (driver *fakeHIPDriver) launchAttentionHeadsBatchCausalQueryRMSRoPE(args []byte) error {
	if len(args) != hipAttentionHeadsBatchCausalQueryRMSRoPELaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipAttentionHeadsBatchCausalQueryRMSRoPELaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipAttentionHeadsBatchCausalQueryRMSRoPELaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE launch header mismatch", nil)
	}
	queryPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	queryWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	keyPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	valuePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	attentionWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	dim := int(binary.LittleEndian.Uint32(args[56:]))
	tokenCount := int(binary.LittleEndian.Uint32(args[60:]))
	headCount := int(binary.LittleEndian.Uint32(args[64:]))
	queryCount := int(binary.LittleEndian.Uint32(args[68:]))
	queryStartToken := int(binary.LittleEndian.Uint32(args[72:]))
	queryBytes := int(binary.LittleEndian.Uint32(args[76:]))
	queryWeightBytes := int(binary.LittleEndian.Uint32(args[80:]))
	keyBytes := int(binary.LittleEndian.Uint32(args[84:]))
	valueBytes := int(binary.LittleEndian.Uint32(args[88:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[92:]))
	attentionWeightBytes := int(binary.LittleEndian.Uint32(args[96:]))
	kvSource := binary.LittleEndian.Uint32(args[100:])
	scale := math.Float32frombits(binary.LittleEndian.Uint32(args[104:]))
	keyHeads := int(binary.LittleEndian.Uint32(args[108:]))
	descriptorPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[112:]))
	descriptorBytes := int(binary.LittleEndian.Uint64(args[120:]))
	windowSize := int(binary.LittleEndian.Uint32(args[136:]))
	queryEpsilon := math.Float32frombits(binary.LittleEndian.Uint32(args[140:]))
	queryWeightEncoding := binary.LittleEndian.Uint32(args[144:])
	queryFlags := binary.LittleEndian.Uint32(args[148:])
	ropeStartPosition := int(binary.LittleEndian.Uint32(args[152:]))
	ropeBase := math.Float32frombits(binary.LittleEndian.Uint32(args[156:]))
	ropeFrequencyDim := int(binary.LittleEndian.Uint32(args[160:]))
	ropeRotaryCount := int(binary.LittleEndian.Uint32(args[164:]))
	ropeFrequencyScale := math.Float32frombits(binary.LittleEndian.Uint32(args[168:]))
	if dim <= 0 || dim%2 != 0 || tokenCount <= 0 || headCount <= 0 || queryCount <= 0 ||
		keyHeads <= 0 || keyHeads > headCount || headCount%keyHeads != 0 ||
		queryStartToken < 0 || windowSize < 0 || uint64(queryStartToken)+uint64(queryCount) > uint64(tokenCount) ||
		queryBytes != queryCount*headCount*dim*4 ||
		outputBytes != queryCount*headCount*dim*4 {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE shape metadata mismatch", nil)
	}
	useSharedWeights := attentionWeightPointer == 0
	if useSharedWeights {
		if attentionWeightBytes != 0 || tokenCount > hipAttentionHeadsSharedMaxTokens {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE shared weight metadata mismatch", nil)
		}
	} else if attentionWeightBytes != queryCount*headCount*tokenCount*4 {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE weight metadata mismatch", nil)
	}
	if scale < 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE scale is invalid", nil)
	}
	if queryEpsilon < 0 || math.IsNaN(float64(queryEpsilon)) || math.IsInf(float64(queryEpsilon), 0) {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE epsilon is invalid", nil)
	}
	if ropeStartPosition < 0 || ropeBase <= 0 || math.IsNaN(float64(ropeBase)) || math.IsInf(float64(ropeBase), 0) ||
		ropeFrequencyScale <= 0 || math.IsNaN(float64(ropeFrequencyScale)) || math.IsInf(float64(ropeFrequencyScale), 0) {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE RoPE metadata is invalid", nil)
	}
	if ropeFrequencyDim == 0 {
		ropeFrequencyDim = dim
	}
	if ropeFrequencyDim < dim || ropeRotaryCount < 0 || ropeRotaryCount > dim || ropeRotaryCount%2 != 0 {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE RoPE shape metadata mismatch", nil)
	}
	if ropeRotaryCount == 0 {
		ropeRotaryCount = dim
	}
	if queryFlags&^hipRMSNormLaunchFlagMask != 0 {
		return core.E("rocm.hip.FakeLaunch", "unsupported attention heads batch causal query RMS RoPE flags", nil)
	}
	queryData, queryOffset, ok := driver.memoryForPointer(queryPointer, queryBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE query buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE output buffer is missing", nil)
	}
	var attentionWeightData []byte
	var attentionWeightOffset int
	if !useSharedWeights {
		var ok bool
		attentionWeightData, attentionWeightOffset, ok = driver.memoryForPointer(attentionWeightPointer, attentionWeightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE weight buffer is missing", nil)
		}
	}
	queryWeight := make([]float32, dim)
	switch queryWeightEncoding {
	case hipRMSNormWeightEncodingNone:
		if queryWeightPointer != 0 || queryWeightBytes != 0 || queryFlags&hipRMSNormLaunchFlagAddUnitWeight != 0 {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE unit weight metadata mismatch", nil)
		}
		for index := range queryWeight {
			queryWeight[index] = 1
		}
	case hipRMSNormWeightEncodingF32:
		if queryWeightPointer == 0 || queryWeightBytes != dim*4 {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE f32 weight metadata mismatch", nil)
		}
		weightData, weightOffset, ok := driver.memoryForPointer(queryWeightPointer, queryWeightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE query weight buffer is missing", nil)
		}
		var err error
		queryWeight, err = hipFloat32PayloadValues(weightData[weightOffset : weightOffset+queryWeightBytes])
		if err != nil {
			return err
		}
	case hipRMSNormWeightEncodingBF16:
		if queryWeightPointer == 0 || queryWeightBytes != dim*2 {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE bf16 weight metadata mismatch", nil)
		}
		weightData, weightOffset, ok := driver.memoryForPointer(queryWeightPointer, queryWeightBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE query weight buffer is missing", nil)
		}
		for index := range queryWeight {
			queryWeight[index] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(weightData[weightOffset+index*2:]))
		}
	default:
		return core.E("rocm.hip.FakeLaunch", "unsupported attention heads batch causal query RMS RoPE weight encoding", nil)
	}
	if queryFlags&hipRMSNormLaunchFlagAddUnitWeight != 0 {
		for index := range queryWeight {
			queryWeight[index] += 1
		}
	}
	var keyFlat []float32
	var valueFlat []float32
	var err error
	switch kvSource {
	case hipAttentionKVSourceContiguous:
		if keyBytes != dim*tokenCount*keyHeads*4 || valueBytes != dim*tokenCount*keyHeads*4 {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE shape metadata mismatch", nil)
		}
		keyData, keyOffset, ok := driver.memoryForPointer(keyPointer, keyBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE key buffer is missing", nil)
		}
		valueData, valueOffset, ok := driver.memoryForPointer(valuePointer, valueBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE value buffer is missing", nil)
		}
		keyFlat, err = hipFloat32PayloadValues(keyData[keyOffset : keyOffset+keyBytes])
		if err != nil {
			return err
		}
		valueFlat, err = hipFloat32PayloadValues(valueData[valueOffset : valueOffset+valueBytes])
		if err != nil {
			return err
		}
	case hipAttentionKVSourceDevice:
		if keyPointer != 0 || valuePointer != 0 || keyBytes != 0 || valueBytes != 0 {
			return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE device KV source must not include contiguous KV buffers", nil)
		}
		keyFlat, valueFlat, err = driver.readDeviceKVDescriptorForAttention(descriptorPointer, descriptorBytes, tokenCount, dim*keyHeads)
		if err != nil {
			return err
		}
	default:
		return core.E("rocm.hip.FakeLaunch", "attention heads batch causal query RMS RoPE KV source is unsupported", nil)
	}
	for queryIndex := 0; queryIndex < queryCount; queryIndex++ {
		visibleTokens := queryStartToken + queryIndex + 1
		windowStart := 0
		if windowSize > 0 && visibleTokens > windowSize {
			windowStart = visibleTokens - windowSize
		}
		for head := 0; head < headCount; head++ {
			baseIndex := queryIndex*headCount + head
			queryStart := queryOffset + baseIndex*dim*4
			query, err := hipFloat32PayloadValues(queryData[queryStart : queryStart+dim*4])
			if err != nil {
				return err
			}
			normalized, err := hipReferenceRMSNorm(query, queryWeight, queryEpsilon)
			if err != nil {
				return err
			}
			var rotated []float32
			position := ropeStartPosition + queryIndex
			if queryFlags&hipRMSNormLaunchFlagRoPENeoX != 0 {
				rotated, err = hipReferenceRoPENeoXWithFrequencyDimScale(normalized, position, float64(ropeBase), ropeFrequencyDim, ropeRotaryCount, float64(ropeFrequencyScale))
			} else {
				rotated = append([]float32(nil), normalized...)
				var rotary []float32
				rotary, err = hipReferenceRoPEWithFrequencyDimScale(normalized[:ropeRotaryCount], position, float64(ropeBase), ropeFrequencyDim, float64(ropeFrequencyScale))
				if err == nil {
					copy(rotated[:ropeRotaryCount], rotary)
				}
			}
			if err != nil {
				return err
			}
			keys, err := fakeROCmAttentionHeadVectors(keyFlat, tokenCount, keyHeads, dim, headCount, head)
			if err != nil {
				return err
			}
			values, err := fakeROCmAttentionHeadVectors(valueFlat, tokenCount, keyHeads, dim, headCount, head)
			if err != nil {
				return err
			}
			output, weights, err := hipReferenceSingleHeadAttentionWithScale(rotated, keys[windowStart:visibleTokens], values[windowStart:visibleTokens], scale)
			if err != nil {
				return err
			}
			outputPayload, err := hipFloat32Payload(output)
			if err != nil {
				return err
			}
			copy(outputData[outputOffset+baseIndex*dim*4:outputOffset+(baseIndex+1)*dim*4], outputPayload)
			if !useSharedWeights {
				weightPayload, err := hipFloat32Payload(weights)
				if err != nil {
					return err
				}
				weightStart := attentionWeightOffset + baseIndex*tokenCount*4
				copy(attentionWeightData[weightStart+windowStart*4:weightStart+visibleTokens*4], weightPayload)
			}
		}
	}
	return nil
}

func (driver *fakeHIPDriver) launchAttentionHeadsBatchChunked(args []byte, writeOutput bool) error {
	if len(args) != hipAttentionHeadsBatchChunkedLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch chunked launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipAttentionHeadsBatchChunkedLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipAttentionHeadsBatchChunkedLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch chunked launch header mismatch", nil)
	}
	queryPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	descriptorPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	partialPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	statsPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	dim := int(binary.LittleEndian.Uint32(args[48:]))
	tokenCount := int(binary.LittleEndian.Uint32(args[52:]))
	headCount := int(binary.LittleEndian.Uint32(args[56:]))
	queryCount := int(binary.LittleEndian.Uint32(args[60:]))
	queryStartToken := int(binary.LittleEndian.Uint32(args[64:]))
	chunkSize := int(binary.LittleEndian.Uint32(args[68:]))
	chunkCount := int(binary.LittleEndian.Uint32(args[72:]))
	queryBytes := int(binary.LittleEndian.Uint32(args[76:]))
	descriptorBytes := int(binary.LittleEndian.Uint64(args[80:]))
	partialBytes := int(binary.LittleEndian.Uint32(args[88:]))
	statsBytes := int(binary.LittleEndian.Uint32(args[92:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[96:]))
	scale := math.Float32frombits(binary.LittleEndian.Uint32(args[100:]))
	windowSize := int(binary.LittleEndian.Uint32(args[104:]))
	chunkStartToken := int(binary.LittleEndian.Uint32(args[108:]))
	keyHeads := int(binary.LittleEndian.Uint32(args[112:]))
	activeEnd := queryStartToken + queryCount
	if activeEnd > tokenCount {
		activeEnd = tokenCount
	}
	expectedChunkCount := 0
	if chunkSize > 0 && activeEnd > chunkStartToken {
		expectedChunkCount = (activeEnd - chunkStartToken + chunkSize - 1) / chunkSize
	}
	if dim <= 0 || dim > hipAttentionHeadsChunkedBlockSize || tokenCount <= 0 || headCount <= 0 || keyHeads <= 0 || keyHeads > headCount || headCount%keyHeads != 0 || queryCount <= 0 ||
		queryStartToken < 0 || windowSize < 0 || uint64(queryStartToken)+uint64(queryCount) > uint64(tokenCount) ||
		chunkSize <= 0 || chunkStartToken < 0 || chunkStartToken > activeEnd || chunkCount != expectedChunkCount ||
		queryBytes != queryCount*headCount*dim*4 ||
		partialBytes != queryCount*headCount*chunkCount*dim*4 ||
		statsBytes != queryCount*headCount*chunkCount*2*4 ||
		outputBytes != queryCount*headCount*dim*4 {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch chunked shape metadata mismatch", nil)
	}
	if scale < 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch chunked scale is invalid", nil)
	}
	queryData, queryOffset, ok := driver.memoryForPointer(queryPointer, queryBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch chunked query buffer is missing", nil)
	}
	if _, _, ok := driver.memoryForPointer(partialPointer, partialBytes); !ok {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch chunked partial buffer is missing", nil)
	}
	if _, _, ok := driver.memoryForPointer(statsPointer, statsBytes); !ok {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch chunked stats buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention heads batch chunked output buffer is missing", nil)
	}
	if !writeOutput {
		return nil
	}
	keyFlat, valueFlat, err := driver.readDeviceKVDescriptorForAttention(descriptorPointer, descriptorBytes, tokenCount, keyHeads*dim)
	if err != nil {
		return err
	}
	for queryIndex := 0; queryIndex < queryCount; queryIndex++ {
		visibleTokens := queryStartToken + queryIndex + 1
		windowStart := 0
		if windowSize > 0 && visibleTokens > windowSize {
			windowStart = visibleTokens - windowSize
		}
		for head := 0; head < headCount; head++ {
			keys, err := fakeROCmAttentionHeadVectors(keyFlat, tokenCount, keyHeads, dim, headCount, head)
			if err != nil {
				return err
			}
			values, err := fakeROCmAttentionHeadVectors(valueFlat, tokenCount, keyHeads, dim, headCount, head)
			if err != nil {
				return err
			}
			baseIndex := queryIndex*headCount + head
			queryStart := queryOffset + baseIndex*dim*4
			query, err := hipFloat32PayloadValues(queryData[queryStart : queryStart+dim*4])
			if err != nil {
				return err
			}
			output, _, err := hipReferenceSingleHeadAttentionWithScale(query, keys[windowStart:visibleTokens], values[windowStart:visibleTokens], scale)
			if err != nil {
				return err
			}
			outputPayload, err := hipFloat32Payload(output)
			if err != nil {
				return err
			}
			copy(outputData[outputOffset+baseIndex*dim*4:outputOffset+(baseIndex+1)*dim*4], outputPayload)
		}
	}
	return nil
}

func (driver *fakeHIPDriver) launchKVEncodeToken(args []byte) error {
	if len(args) != hipKVEncodeTokenLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "KV encode token launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipKVEncodeTokenLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipKVEncodeTokenLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "KV encode token launch header mismatch", nil)
	}
	keyInputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	valueInputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	keyOutputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	valueOutputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	keyCount := int(binary.LittleEndian.Uint32(args[40:]))
	valueCount := int(binary.LittleEndian.Uint32(args[44:]))
	keyInputBytes := int(binary.LittleEndian.Uint32(args[48:]))
	valueInputBytes := int(binary.LittleEndian.Uint32(args[52:]))
	keyOutputBytes := int(binary.LittleEndian.Uint32(args[56:]))
	valueOutputBytes := int(binary.LittleEndian.Uint32(args[60:]))
	keyEncoding := fakeROCmKVEncoding(binary.LittleEndian.Uint32(args[64:]))
	valueEncoding := fakeROCmKVEncoding(binary.LittleEndian.Uint32(args[68:]))
	keyWidth := int(binary.LittleEndian.Uint64(args[72:]))
	valueWidth := int(binary.LittleEndian.Uint64(args[80:]))
	tokenCount := int(binary.LittleEndian.Uint64(args[88:]))
	if keyCount <= 0 || valueCount <= 0 || keyInputBytes != keyCount*4 || valueInputBytes != valueCount*4 || keyEncoding == "" || valueEncoding == "" {
		return core.E("rocm.hip.FakeLaunch", "KV encode token shape metadata mismatch", nil)
	}
	if tokenCount == 0 {
		tokenCount = 1
	}
	if keyWidth == 0 {
		keyWidth = keyCount
	}
	if valueWidth == 0 {
		valueWidth = valueCount
	}
	if tokenCount <= 0 || keyWidth <= 0 || valueWidth <= 0 || keyWidth*tokenCount != keyCount || valueWidth*tokenCount != valueCount {
		return core.E("rocm.hip.FakeLaunch", "KV encode token row shape metadata mismatch", nil)
	}
	expectedKeyOutputBytes, err := rocmKVTensorDeviceByteCountRows(keyEncoding, keyCount, tokenCount)
	if err != nil {
		return err
	}
	expectedValueOutputBytes, err := rocmKVTensorDeviceByteCountRows(valueEncoding, valueCount, tokenCount)
	if err != nil {
		return err
	}
	if uint64(keyOutputBytes) != expectedKeyOutputBytes || uint64(valueOutputBytes) != expectedValueOutputBytes {
		return core.E("rocm.hip.FakeLaunch", "KV encode token output byte count mismatch", nil)
	}
	keyInputData, keyInputOffset, ok := driver.memoryForPointer(keyInputPointer, keyInputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "KV encode key input buffer is missing", nil)
	}
	valueInputData, valueInputOffset, ok := driver.memoryForPointer(valueInputPointer, valueInputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "KV encode value input buffer is missing", nil)
	}
	keyOutputData, keyOutputOffset, ok := driver.memoryForPointer(keyOutputPointer, keyOutputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "KV encode key output buffer is missing", nil)
	}
	valueOutputData, valueOutputOffset, ok := driver.memoryForPointer(valueOutputPointer, valueOutputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "KV encode value output buffer is missing", nil)
	}
	keyValues, err := hipFloat32PayloadValues(keyInputData[keyInputOffset : keyInputOffset+keyInputBytes])
	if err != nil {
		return err
	}
	valueValues, err := hipFloat32PayloadValues(valueInputData[valueInputOffset : valueInputOffset+valueInputBytes])
	if err != nil {
		return err
	}
	keyTensor, err := encodeROCmKVTensorRows(keyEncoding, keyValues, keyWidth, tokenCount)
	if err != nil {
		return err
	}
	keyPayload, err := keyTensor.deviceBytes()
	if err != nil {
		return err
	}
	valueTensor, err := encodeROCmKVTensorRows(valueEncoding, valueValues, valueWidth, tokenCount)
	if err != nil {
		return err
	}
	valuePayload, err := valueTensor.deviceBytes()
	if err != nil {
		return err
	}
	copy(keyOutputData[keyOutputOffset:keyOutputOffset+keyOutputBytes], keyPayload)
	copy(valueOutputData[valueOutputOffset:valueOutputOffset+valueOutputBytes], valuePayload)
	return nil
}

func (driver *fakeHIPDriver) launchKVEncodeTokenValueNorm(args []byte) error {
	if len(args) != hipKVEncodeTokenValueNormLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "KV encode value norm launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipKVEncodeTokenValueNormLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipKVEncodeTokenValueNormLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "KV encode value norm launch header mismatch", nil)
	}
	keyInputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	valueInputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	keyOutputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	valueOutputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	keyCount := int(binary.LittleEndian.Uint32(args[40:]))
	valueCount := int(binary.LittleEndian.Uint32(args[44:]))
	keyInputBytes := int(binary.LittleEndian.Uint32(args[48:]))
	valueInputBytes := int(binary.LittleEndian.Uint32(args[52:]))
	keyOutputBytes := int(binary.LittleEndian.Uint32(args[56:]))
	valueOutputBytes := int(binary.LittleEndian.Uint32(args[60:]))
	keyEncoding := fakeROCmKVEncoding(binary.LittleEndian.Uint32(args[64:]))
	valueEncoding := fakeROCmKVEncoding(binary.LittleEndian.Uint32(args[68:]))
	keyWidth := int(binary.LittleEndian.Uint64(args[72:]))
	valueWidth := int(binary.LittleEndian.Uint64(args[80:]))
	tokenCount := int(binary.LittleEndian.Uint64(args[88:]))
	valueHeadDim := int(binary.LittleEndian.Uint32(args[96:]))
	valueHeadCount := int(binary.LittleEndian.Uint32(args[100:]))
	epsilon := math.Float32frombits(binary.LittleEndian.Uint32(args[104:]))
	if keyCount <= 0 || valueCount <= 0 || keyInputBytes != keyCount*4 || valueInputBytes != valueCount*4 || keyEncoding == "" || valueEncoding == "" {
		return core.E("rocm.hip.FakeLaunch", "KV encode value norm shape metadata mismatch", nil)
	}
	if tokenCount == 0 {
		tokenCount = 1
	}
	if keyWidth == 0 {
		keyWidth = keyCount
	}
	if valueWidth == 0 {
		valueWidth = valueCount
	}
	if tokenCount <= 0 || keyWidth <= 0 || valueWidth <= 0 || keyWidth*tokenCount != keyCount || valueWidth*tokenCount != valueCount ||
		valueHeadDim <= 0 || valueHeadCount <= 0 || valueWidth != valueHeadDim*valueHeadCount ||
		math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) || epsilon < 0 {
		return core.E("rocm.hip.FakeLaunch", "KV encode value norm row shape metadata mismatch", nil)
	}
	expectedKeyOutputBytes, err := rocmKVTensorDeviceByteCountRows(keyEncoding, keyCount, tokenCount)
	if err != nil {
		return err
	}
	expectedValueOutputBytes, err := rocmKVTensorDeviceByteCountRows(valueEncoding, valueCount, tokenCount)
	if err != nil {
		return err
	}
	if uint64(keyOutputBytes) != expectedKeyOutputBytes || uint64(valueOutputBytes) != expectedValueOutputBytes {
		return core.E("rocm.hip.FakeLaunch", "KV encode value norm output byte count mismatch", nil)
	}
	keyInputData, keyInputOffset, ok := driver.memoryForPointer(keyInputPointer, keyInputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "KV encode value norm key input buffer is missing", nil)
	}
	valueInputData, valueInputOffset, ok := driver.memoryForPointer(valueInputPointer, valueInputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "KV encode value norm value input buffer is missing", nil)
	}
	keyOutputData, keyOutputOffset, ok := driver.memoryForPointer(keyOutputPointer, keyOutputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "KV encode value norm key output buffer is missing", nil)
	}
	valueOutputData, valueOutputOffset, ok := driver.memoryForPointer(valueOutputPointer, valueOutputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "KV encode value norm value output buffer is missing", nil)
	}
	keyValues, err := hipFloat32PayloadValues(keyInputData[keyInputOffset : keyInputOffset+keyInputBytes])
	if err != nil {
		return err
	}
	valueValues, err := hipFloat32PayloadValues(valueInputData[valueInputOffset : valueInputOffset+valueInputBytes])
	if err != nil {
		return err
	}
	keyTensor, err := encodeROCmKVTensorRows(keyEncoding, keyValues, keyWidth, tokenCount)
	if err != nil {
		return err
	}
	keyPayload, err := keyTensor.deviceBytes()
	if err != nil {
		return err
	}
	valueWeight := make([]float32, valueHeadDim)
	for index := range valueWeight {
		valueWeight[index] = 1
	}
	normalizedValues := make([]float32, len(valueValues))
	for tokenIndex := 0; tokenIndex < tokenCount; tokenIndex++ {
		tokenOffset := tokenIndex * valueWidth
		for headIndex := 0; headIndex < valueHeadCount; headIndex++ {
			headOffset := tokenOffset + headIndex*valueHeadDim
			normalized, err := hipReferenceRMSNorm(valueValues[headOffset:headOffset+valueHeadDim], valueWeight, epsilon)
			if err != nil {
				return err
			}
			copy(normalizedValues[headOffset:headOffset+valueHeadDim], normalized)
		}
	}
	valueTensor, err := encodeROCmKVTensorRows(valueEncoding, normalizedValues, valueWidth, tokenCount)
	if err != nil {
		return err
	}
	valuePayload, err := valueTensor.deviceBytes()
	if err != nil {
		return err
	}
	copy(keyOutputData[keyOutputOffset:keyOutputOffset+keyOutputBytes], keyPayload)
	copy(valueOutputData[valueOutputOffset:valueOutputOffset+valueOutputBytes], valuePayload)
	return nil
}

func (driver *fakeHIPDriver) launchKVEncodeTokenValueNormDescriptorAppend(args []byte) error {
	if len(args) != hipKVEncodeTokenValueNormDescriptorAppendLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "KV encode value norm descriptor append launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipKVEncodeTokenValueNormDescriptorAppendLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipKVEncodeTokenValueNormDescriptorAppendLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "KV encode value norm descriptor append launch header mismatch", nil)
	}
	encode := args[8 : 8+hipKVEncodeTokenValueNormLaunchArgsBytes]
	if binary.LittleEndian.Uint32(encode[0:]) != hipKVEncodeTokenValueNormLaunchArgsVersion ||
		binary.LittleEndian.Uint32(encode[4:]) != uint32(hipKVEncodeTokenValueNormLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "KV encode value norm descriptor append encode header mismatch", nil)
	}
	if err := driver.launchKVEncodeTokenValueNorm(encode); err != nil {
		return err
	}
	descriptor := args[8+hipKVEncodeTokenValueNormLaunchArgsBytes:]
	return driver.launchKVDescriptorAppend(descriptor)
}

func (driver *fakeHIPDriver) launchKVDescriptorAppend(args []byte) error {
	if len(args) != hipKVDescriptorAppendLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "KV descriptor append launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipKVDescriptorAppendLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipKVDescriptorAppendLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "KV descriptor append launch header mismatch", nil)
	}
	previousPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	newKeyPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	newValuePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	previousBytes := int(binary.LittleEndian.Uint64(args[40:]))
	outputBytes := int(binary.LittleEndian.Uint64(args[48:]))
	newKeyBytes := uint64(binary.LittleEndian.Uint64(args[56:]))
	newValueBytes := uint64(binary.LittleEndian.Uint64(args[64:]))
	modeCode := binary.LittleEndian.Uint32(args[72:])
	blockSize := int(binary.LittleEndian.Uint32(args[76:]))
	outputPageCount := int(binary.LittleEndian.Uint32(args[80:]))
	outputTokenCount := int(binary.LittleEndian.Uint32(args[84:]))
	keyWidth := int(binary.LittleEndian.Uint32(args[88:]))
	valueWidth := int(binary.LittleEndian.Uint32(args[92:]))
	keyEncodingCode := binary.LittleEndian.Uint32(args[96:])
	valueEncodingCode := binary.LittleEndian.Uint32(args[100:])
	trimStart := int(binary.LittleEndian.Uint64(args[104:]))
	appendMode := binary.LittleEndian.Uint64(args[112:])
	if appendMode != rocmKVDescriptorAppendModeBuildSinglePage && previousBytes < rocmDeviceKVDescriptorHeaderBytes {
		return core.E("rocm.hip.FakeLaunch", "KV descriptor append previous byte count mismatch", nil)
	}
	if outputBytes != rocmDeviceKVDescriptorHeaderBytes+outputPageCount*rocmDeviceKVDescriptorPageBytes ||
		outputPageCount <= 0 || outputTokenCount <= 0 || blockSize <= 0 || keyWidth <= 0 || valueWidth <= 0 ||
		fakeROCmKVEncoding(keyEncodingCode) == "" || fakeROCmKVEncoding(valueEncodingCode) == "" {
		return core.E("rocm.hip.FakeLaunch", "KV descriptor append shape metadata mismatch", nil)
	}
	if newKeyBytes == 0 || newValueBytes == 0 || newKeyPointer == 0 || newValuePointer == 0 {
		return core.E("rocm.hip.FakeLaunch", "KV descriptor append page metadata mismatch", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "KV descriptor append output descriptor is missing", nil)
	}
	output := outputData[outputOffset : outputOffset+outputBytes]
	if appendMode == rocmKVDescriptorAppendModeBuildSinglePage {
		if outputPageCount != 1 || trimStart != 0 {
			return core.E("rocm.hip.FakeLaunch", "KV descriptor build single-page shape mismatch", nil)
		}
		binary.LittleEndian.PutUint32(output[0:], rocmDeviceKVDescriptorVersion)
		binary.LittleEndian.PutUint32(output[4:], uint32(rocmDeviceKVDescriptorHeaderBytes))
		binary.LittleEndian.PutUint32(output[8:], uint32(rocmDeviceKVDescriptorPageBytes))
		binary.LittleEndian.PutUint32(output[12:], modeCode)
		binary.LittleEndian.PutUint32(output[16:], uint32(outputPageCount))
		binary.LittleEndian.PutUint32(output[20:], uint32(blockSize))
		binary.LittleEndian.PutUint64(output[24:], uint64(outputTokenCount))
		page := output[rocmDeviceKVDescriptorHeaderBytes : rocmDeviceKVDescriptorHeaderBytes+rocmDeviceKVDescriptorPageBytes]
		binary.LittleEndian.PutUint64(page[0:], 0)
		binary.LittleEndian.PutUint64(page[8:], uint64(outputTokenCount))
		binary.LittleEndian.PutUint32(page[16:], uint32(keyWidth))
		binary.LittleEndian.PutUint32(page[20:], uint32(valueWidth))
		binary.LittleEndian.PutUint32(page[24:], keyEncodingCode)
		binary.LittleEndian.PutUint32(page[28:], valueEncodingCode)
		binary.LittleEndian.PutUint64(page[32:], uint64(newKeyPointer))
		binary.LittleEndian.PutUint64(page[40:], uint64(newValuePointer))
		binary.LittleEndian.PutUint64(page[48:], newKeyBytes)
		binary.LittleEndian.PutUint64(page[56:], newValueBytes)
		return nil
	}
	previousData, previousOffset, ok := driver.memoryForPointer(previousPointer, previousBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "KV descriptor append previous descriptor is missing", nil)
	}
	previous := previousData[previousOffset : previousOffset+previousBytes]
	if binary.LittleEndian.Uint32(previous[0:]) != rocmDeviceKVDescriptorVersion ||
		int(binary.LittleEndian.Uint32(previous[4:])) != rocmDeviceKVDescriptorHeaderBytes ||
		int(binary.LittleEndian.Uint32(previous[8:])) != rocmDeviceKVDescriptorPageBytes ||
		binary.LittleEndian.Uint32(previous[12:]) != modeCode ||
		int(binary.LittleEndian.Uint32(previous[20:])) != blockSize {
		return core.E("rocm.hip.FakeLaunch", "KV descriptor append previous descriptor header mismatch", nil)
	}
	previousPageCount := int(binary.LittleEndian.Uint32(previous[16:]))
	previousTokenCount := int(binary.LittleEndian.Uint64(previous[24:]))
	appendCount := trimStart + outputTokenCount - previousTokenCount
	if previousBytes != rocmDeviceKVDescriptorHeaderBytes+previousPageCount*rocmDeviceKVDescriptorPageBytes ||
		appendCount <= 0 ||
		appendCount > blockSize {
		return core.E("rocm.hip.FakeLaunch", "KV descriptor append previous descriptor size mismatch", nil)
	}
	if appendMode == rocmKVDescriptorAppendModeGrowLastPage {
		if outputPageCount > previousPageCount {
			return core.E("rocm.hip.FakeLaunch", "KV descriptor grow page count mismatch", nil)
		}
		if trimStart == 0 {
			if outputTokenCount != previousTokenCount+appendCount {
				return core.E("rocm.hip.FakeLaunch", "KV descriptor grow page count mismatch", nil)
			}
			copy(output, previous)
			lastOffset := rocmDeviceKVDescriptorHeaderBytes + (previousPageCount-1)*rocmDeviceKVDescriptorPageBytes
			if int(binary.LittleEndian.Uint64(output[lastOffset:])+binary.LittleEndian.Uint64(output[lastOffset+8:])) != previousTokenCount ||
				nativeDevicePointer(binary.LittleEndian.Uint64(output[lastOffset+32:])) != newKeyPointer ||
				nativeDevicePointer(binary.LittleEndian.Uint64(output[lastOffset+40:])) != newValuePointer {
				return core.E("rocm.hip.FakeLaunch", "KV descriptor grow last page mismatch", nil)
			}
			binary.LittleEndian.PutUint64(output[lastOffset+8:], binary.LittleEndian.Uint64(output[lastOffset+8:])+uint64(appendCount))
			binary.LittleEndian.PutUint64(output[lastOffset+48:], newKeyBytes)
			binary.LittleEndian.PutUint64(output[lastOffset+56:], newValueBytes)
			binary.LittleEndian.PutUint64(output[24:], uint64(outputTokenCount))
			return nil
		}
		outputIndex := 0
		for pageIndex := 0; pageIndex < previousPageCount-1; pageIndex++ {
			pageOffset := rocmDeviceKVDescriptorHeaderBytes + pageIndex*rocmDeviceKVDescriptorPageBytes
			retained, err := fakeROCmKVDescriptorTrimPage(previous[pageOffset:pageOffset+rocmDeviceKVDescriptorPageBytes], trimStart)
			if err != nil {
				return err
			}
			if !retained.ok {
				continue
			}
			if outputIndex >= outputPageCount-1 {
				return core.E("rocm.hip.FakeLaunch", "KV descriptor grow retained page mismatch", nil)
			}
			outOffset := rocmDeviceKVDescriptorHeaderBytes + outputIndex*rocmDeviceKVDescriptorPageBytes
			copy(output[outOffset:outOffset+rocmDeviceKVDescriptorPageBytes], retained.payload[:])
			outputIndex++
		}
		if outputIndex != outputPageCount-1 {
			return core.E("rocm.hip.FakeLaunch", "KV descriptor grow output page count mismatch", nil)
		}
		lastOffset := rocmDeviceKVDescriptorHeaderBytes + (previousPageCount-1)*rocmDeviceKVDescriptorPageBytes
		previousLast := previous[lastOffset : lastOffset+rocmDeviceKVDescriptorPageBytes]
		retainedLast, err := fakeROCmKVDescriptorTrimPage(previousLast, trimStart)
		if err != nil {
			return err
		}
		if int(binary.LittleEndian.Uint64(previousLast[0:])+binary.LittleEndian.Uint64(previousLast[8:])) != previousTokenCount ||
			!retainedLast.ok ||
			nativeDevicePointer(binary.LittleEndian.Uint64(retainedLast.payload[32:])) != newKeyPointer ||
			nativeDevicePointer(binary.LittleEndian.Uint64(retainedLast.payload[40:])) != newValuePointer {
			return core.E("rocm.hip.FakeLaunch", "KV descriptor grow last page mismatch", nil)
		}
		lastOutOffset := rocmDeviceKVDescriptorHeaderBytes + (outputPageCount-1)*rocmDeviceKVDescriptorPageBytes
		copy(output[lastOutOffset:lastOutOffset+rocmDeviceKVDescriptorPageBytes], retainedLast.payload[:])
		binary.LittleEndian.PutUint64(output[lastOutOffset+8:], binary.LittleEndian.Uint64(retainedLast.payload[8:])+uint64(appendCount))
		binary.LittleEndian.PutUint64(output[lastOutOffset+48:], newKeyBytes)
		binary.LittleEndian.PutUint64(output[lastOutOffset+56:], newValueBytes)
		binary.LittleEndian.PutUint32(output[0:], rocmDeviceKVDescriptorVersion)
		binary.LittleEndian.PutUint32(output[4:], uint32(rocmDeviceKVDescriptorHeaderBytes))
		binary.LittleEndian.PutUint32(output[8:], uint32(rocmDeviceKVDescriptorPageBytes))
		binary.LittleEndian.PutUint32(output[12:], modeCode)
		binary.LittleEndian.PutUint32(output[16:], uint32(outputPageCount))
		binary.LittleEndian.PutUint32(output[20:], uint32(blockSize))
		binary.LittleEndian.PutUint64(output[24:], uint64(outputTokenCount))
		return nil
	}
	outputIndex := 0
	for pageIndex := 0; pageIndex < previousPageCount; pageIndex++ {
		pageOffset := rocmDeviceKVDescriptorHeaderBytes + pageIndex*rocmDeviceKVDescriptorPageBytes
		page := previous[pageOffset : pageOffset+rocmDeviceKVDescriptorPageBytes]
		tokenStart := int(binary.LittleEndian.Uint64(page[0:]))
		tokenCount := int(binary.LittleEndian.Uint64(page[8:]))
		if tokenStart+tokenCount <= trimStart {
			continue
		}
		if outputIndex+1 >= outputPageCount {
			return core.E("rocm.hip.FakeLaunch", "KV descriptor append retained page mismatch", nil)
		}
		retained, err := fakeROCmKVDescriptorTrimPage(page, trimStart)
		if err != nil {
			return err
		}
		if !retained.ok {
			return core.E("rocm.hip.FakeLaunch", "KV descriptor append retained page mismatch", nil)
		}
		outOffset := rocmDeviceKVDescriptorHeaderBytes + outputIndex*rocmDeviceKVDescriptorPageBytes
		copy(output[outOffset:outOffset+rocmDeviceKVDescriptorPageBytes], retained.payload[:])
		outputIndex++
	}
	if outputIndex+1 != outputPageCount {
		return core.E("rocm.hip.FakeLaunch", "KV descriptor append output page count mismatch", nil)
	}
	newOffset := rocmDeviceKVDescriptorHeaderBytes + outputIndex*rocmDeviceKVDescriptorPageBytes
	binary.LittleEndian.PutUint64(output[newOffset:], uint64(outputTokenCount-appendCount))
	binary.LittleEndian.PutUint64(output[newOffset+8:], uint64(appendCount))
	binary.LittleEndian.PutUint32(output[newOffset+16:], uint32(keyWidth))
	binary.LittleEndian.PutUint32(output[newOffset+20:], uint32(valueWidth))
	binary.LittleEndian.PutUint32(output[newOffset+24:], keyEncodingCode)
	binary.LittleEndian.PutUint32(output[newOffset+28:], valueEncodingCode)
	binary.LittleEndian.PutUint64(output[newOffset+32:], uint64(newKeyPointer))
	binary.LittleEndian.PutUint64(output[newOffset+40:], uint64(newValuePointer))
	binary.LittleEndian.PutUint64(output[newOffset+48:], newKeyBytes)
	binary.LittleEndian.PutUint64(output[newOffset+56:], newValueBytes)
	binary.LittleEndian.PutUint32(output[0:], rocmDeviceKVDescriptorVersion)
	binary.LittleEndian.PutUint32(output[4:], uint32(rocmDeviceKVDescriptorHeaderBytes))
	binary.LittleEndian.PutUint32(output[8:], uint32(rocmDeviceKVDescriptorPageBytes))
	binary.LittleEndian.PutUint32(output[12:], modeCode)
	binary.LittleEndian.PutUint32(output[16:], uint32(outputPageCount))
	binary.LittleEndian.PutUint32(output[20:], uint32(blockSize))
	binary.LittleEndian.PutUint64(output[24:], uint64(outputTokenCount))
	return nil
}

type fakeROCmKVDescriptorTrimmedPage struct {
	payload [rocmDeviceKVDescriptorPageBytes]byte
	ok      bool
}

func fakeROCmKVDescriptorTrimPage(page []byte, trimStart int) (fakeROCmKVDescriptorTrimmedPage, error) {
	if len(page) < rocmDeviceKVDescriptorPageBytes {
		return fakeROCmKVDescriptorTrimmedPage{}, core.E("rocm.hip.FakeLaunch", "KV descriptor trim page is too short", nil)
	}
	tokenStart := int(binary.LittleEndian.Uint64(page[0:]))
	tokenCount := int(binary.LittleEndian.Uint64(page[8:]))
	pageEnd := tokenStart + tokenCount
	if pageEnd <= trimStart {
		return fakeROCmKVDescriptorTrimmedPage{}, nil
	}
	var retained fakeROCmKVDescriptorTrimmedPage
	copy(retained.payload[:], page[:rocmDeviceKVDescriptorPageBytes])
	retained.ok = true
	if tokenStart >= trimStart {
		binary.LittleEndian.PutUint64(retained.payload[0:], uint64(tokenStart-trimStart))
		return retained, nil
	}
	keyWidth := int(binary.LittleEndian.Uint32(page[16:]))
	valueWidth := int(binary.LittleEndian.Uint32(page[20:]))
	keyEncoding := fakeROCmKVEncoding(binary.LittleEndian.Uint32(page[24:]))
	valueEncoding := fakeROCmKVEncoding(binary.LittleEndian.Uint32(page[28:]))
	keyStride, err := rocmKVInterleavedRowStride(keyEncoding, keyWidth)
	if err != nil {
		return fakeROCmKVDescriptorTrimmedPage{}, core.E("rocm.hip.FakeLaunch", "KV descriptor cannot trim key page", err)
	}
	valueStride, err := rocmKVInterleavedRowStride(valueEncoding, valueWidth)
	if err != nil {
		return fakeROCmKVDescriptorTrimmedPage{}, core.E("rocm.hip.FakeLaunch", "KV descriptor cannot trim value page", err)
	}
	keyBytes := binary.LittleEndian.Uint64(page[48:])
	valueBytes := binary.LittleEndian.Uint64(page[56:])
	if keyBytes != keyStride*uint64(tokenCount) || valueBytes != valueStride*uint64(tokenCount) {
		return fakeROCmKVDescriptorTrimmedPage{}, core.E("rocm.hip.FakeLaunch", "KV descriptor trim page byte count mismatch", nil)
	}
	skipTokens := trimStart - tokenStart
	retainedTokens := pageEnd - trimStart
	binary.LittleEndian.PutUint64(retained.payload[0:], 0)
	binary.LittleEndian.PutUint64(retained.payload[8:], uint64(retainedTokens))
	binary.LittleEndian.PutUint64(retained.payload[32:], binary.LittleEndian.Uint64(page[32:])+keyStride*uint64(skipTokens))
	binary.LittleEndian.PutUint64(retained.payload[40:], binary.LittleEndian.Uint64(page[40:])+valueStride*uint64(skipTokens))
	binary.LittleEndian.PutUint64(retained.payload[48:], keyStride*uint64(retainedTokens))
	binary.LittleEndian.PutUint64(retained.payload[56:], valueStride*uint64(retainedTokens))
	return retained, nil
}

func (driver *fakeHIPDriver) readDeviceKVDescriptorForAttention(pointer nativeDevicePointer, sizeBytes, tokenCount, dim int) ([]float32, []float32, error) {
	if pointer == 0 || sizeBytes < rocmDeviceKVDescriptorHeaderBytes {
		return nil, nil, core.E("rocm.hip.FakeLaunch", "attention device KV descriptor is missing", nil)
	}
	data, offset, ok := driver.memoryForPointer(pointer, sizeBytes)
	if !ok {
		return nil, nil, core.E("rocm.hip.FakeLaunch", "attention device KV descriptor buffer is missing", nil)
	}
	descriptor := data[offset : offset+sizeBytes]
	if binary.LittleEndian.Uint32(descriptor[0:]) != rocmDeviceKVDescriptorVersion ||
		int(binary.LittleEndian.Uint32(descriptor[4:])) != rocmDeviceKVDescriptorHeaderBytes ||
		int(binary.LittleEndian.Uint32(descriptor[8:])) != rocmDeviceKVDescriptorPageBytes ||
		int(binary.LittleEndian.Uint64(descriptor[24:])) != tokenCount {
		return nil, nil, core.E("rocm.hip.FakeLaunch", "attention device KV descriptor header mismatch", nil)
	}
	pageCount := int(binary.LittleEndian.Uint32(descriptor[16:]))
	if sizeBytes != rocmDeviceKVDescriptorHeaderBytes+pageCount*rocmDeviceKVDescriptorPageBytes {
		return nil, nil, core.E("rocm.hip.FakeLaunch", "attention device KV descriptor size mismatch", nil)
	}
	keys := make([]float32, tokenCount*dim)
	values := make([]float32, tokenCount*dim)
	for pageIndex := 0; pageIndex < pageCount; pageIndex++ {
		pageOffset := rocmDeviceKVDescriptorHeaderBytes + pageIndex*rocmDeviceKVDescriptorPageBytes
		page := descriptor[pageOffset : pageOffset+rocmDeviceKVDescriptorPageBytes]
		tokenStart := int(binary.LittleEndian.Uint64(page[0:]))
		pageTokens := int(binary.LittleEndian.Uint64(page[8:]))
		keyWidth := int(binary.LittleEndian.Uint32(page[16:]))
		valueWidth := int(binary.LittleEndian.Uint32(page[20:]))
		keyEncoding := fakeROCmKVEncoding(binary.LittleEndian.Uint32(page[24:]))
		valueEncoding := fakeROCmKVEncoding(binary.LittleEndian.Uint32(page[28:]))
		keyPointer := nativeDevicePointer(binary.LittleEndian.Uint64(page[32:]))
		valuePointer := nativeDevicePointer(binary.LittleEndian.Uint64(page[40:]))
		keyBytes := int(binary.LittleEndian.Uint64(page[48:]))
		valueBytes := int(binary.LittleEndian.Uint64(page[56:]))
		if tokenStart < 0 || pageTokens <= 0 || tokenStart+pageTokens > tokenCount || keyWidth != dim || valueWidth != dim || keyEncoding == "" || valueEncoding == "" {
			return nil, nil, core.E("rocm.hip.FakeLaunch", "attention device KV descriptor page shape mismatch", nil)
		}
		pageKeys, err := driver.readDeviceKVTensorRows(keyPointer, keyBytes, keyEncoding, pageTokens*keyWidth, pageTokens)
		if err != nil {
			return nil, nil, err
		}
		pageValues, err := driver.readDeviceKVTensorRows(valuePointer, valueBytes, valueEncoding, pageTokens*valueWidth, pageTokens)
		if err != nil {
			return nil, nil, err
		}
		copy(keys[tokenStart*dim:(tokenStart+pageTokens)*dim], pageKeys)
		copy(values[tokenStart*dim:(tokenStart+pageTokens)*dim], pageValues)
	}
	return keys, values, nil
}

func (driver *fakeHIPDriver) readDeviceKVTensor(pointer nativeDevicePointer, sizeBytes int, encoding string, length int) ([]float32, error) {
	return driver.readDeviceKVTensorRows(pointer, sizeBytes, encoding, length, 1)
}

func (driver *fakeHIPDriver) readDeviceKVTensorRows(pointer nativeDevicePointer, sizeBytes int, encoding string, length, rows int) ([]float32, error) {
	data, offset, ok := driver.memoryForPointer(pointer, sizeBytes)
	if !ok {
		return nil, core.E("rocm.hip.FakeLaunch", "attention device KV tensor buffer is missing", nil)
	}
	tensor, err := rocmKVTensorFromDeviceBytesRows(encoding, length, rows, append([]byte(nil), data[offset:offset+sizeBytes]...))
	if err != nil {
		return nil, err
	}
	rowWidth := length
	if rows > 0 {
		rowWidth = length / rows
	}
	return tensor.decodeRows(rowWidth), nil
}

func fakeROCmKVEncoding(code uint32) string {
	switch code {
	case rocmDeviceKVDescriptorEncodingFP16:
		return rocmKVEncodingFP16
	case rocmDeviceKVDescriptorEncodingQ8:
		return rocmKVEncodingQ8
	case rocmDeviceKVDescriptorEncodingQ4:
		return rocmKVEncodingQ4
	case rocmDeviceKVDescriptorEncodingQ8Rows:
		return rocmKVEncodingQ8Rows
	case rocmDeviceKVDescriptorEncodingQ4Rows:
		return rocmKVEncodingQ4Rows
	case rocmDeviceKVDescriptorEncodingQ8RowsI:
		return rocmKVEncodingQ8RowsI
	case rocmDeviceKVDescriptorEncodingQ4RowsI:
		return rocmKVEncodingQ4RowsI
	default:
		return ""
	}
}

func (driver *fakeHIPDriver) launchVectorAdd(args []byte) error {
	if len(args) != hipVectorAddLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "vector add launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipVectorAddLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipVectorAddLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "vector add launch header mismatch", nil)
	}
	leftPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	rightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	count := int(binary.LittleEndian.Uint32(args[32:]))
	leftBytes := int(binary.LittleEndian.Uint32(args[36:]))
	rightBytes := int(binary.LittleEndian.Uint32(args[40:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[44:]))
	if count <= 0 || leftBytes != count*4 || rightBytes != count*4 || outputBytes != count*4 {
		return core.E("rocm.hip.FakeLaunch", "vector add shape metadata mismatch", nil)
	}
	leftData, leftOffset, ok := driver.memoryForPointer(leftPointer, leftBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "vector add left buffer is missing", nil)
	}
	rightData, rightOffset, ok := driver.memoryForPointer(rightPointer, rightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "vector add right buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "vector add output buffer is missing", nil)
	}
	left, err := hipFloat32PayloadValues(leftData[leftOffset : leftOffset+leftBytes])
	if err != nil {
		return err
	}
	right, err := hipFloat32PayloadValues(rightData[rightOffset : rightOffset+rightBytes])
	if err != nil {
		return err
	}
	out := make([]float32, count)
	for index := range out {
		out[index] = left[index] + right[index]
	}
	payload, err := hipFloat32Payload(out)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchVectorAddScaled(args []byte) error {
	if len(args) != hipVectorAddScaledLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "vector add-scaled launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipVectorAddScaledLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipVectorAddScaledLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "vector add-scaled launch header mismatch", nil)
	}
	leftPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	rightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	count := int(binary.LittleEndian.Uint32(args[32:]))
	leftBytes := int(binary.LittleEndian.Uint32(args[36:]))
	rightBytes := int(binary.LittleEndian.Uint32(args[40:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[44:]))
	scale := math.Float32frombits(binary.LittleEndian.Uint32(args[48:]))
	if count <= 0 || leftBytes != count*4 || rightBytes != count*4 || outputBytes != count*4 ||
		math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return core.E("rocm.hip.FakeLaunch", "vector add-scaled shape metadata mismatch", nil)
	}
	leftData, leftOffset, ok := driver.memoryForPointer(leftPointer, leftBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "vector add-scaled left buffer is missing", nil)
	}
	rightData, rightOffset, ok := driver.memoryForPointer(rightPointer, rightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "vector add-scaled right buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "vector add-scaled output buffer is missing", nil)
	}
	left, err := hipFloat32PayloadValues(leftData[leftOffset : leftOffset+leftBytes])
	if err != nil {
		return err
	}
	right, err := hipFloat32PayloadValues(rightData[rightOffset : rightOffset+rightBytes])
	if err != nil {
		return err
	}
	out := make([]float32, count)
	for index := range out {
		out[index] = (left[index] + right[index]) * scale
	}
	payload, err := hipFloat32Payload(out)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchVectorScale(args []byte) error {
	if len(args) != hipVectorScaleLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "vector scale launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipVectorScaleLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipVectorScaleLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "vector scale launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	count := int(binary.LittleEndian.Uint32(args[24:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[28:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[32:]))
	scale := math.Float32frombits(binary.LittleEndian.Uint32(args[36:]))
	if count <= 0 || inputBytes != count*4 || outputBytes != count*4 ||
		math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return core.E("rocm.hip.FakeLaunch", "vector scale shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "vector scale input buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "vector scale output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	out := make([]float32, count)
	for index := range out {
		out[index] = input[index] * scale
	}
	payload, err := hipFloat32Payload(out)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchPerLayerInputTranspose(args []byte) error {
	if len(args) != hipPerLayerInputTransposeLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "per-layer input transpose launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipPerLayerInputTransposeLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipPerLayerInputTransposeLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "per-layer input transpose launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	inputBytes := int(binary.LittleEndian.Uint64(args[24:]))
	outputBytes := int(binary.LittleEndian.Uint64(args[32:]))
	batch := int(binary.LittleEndian.Uint32(args[40:]))
	layerCount := int(binary.LittleEndian.Uint32(args[44:]))
	inputSize := int(binary.LittleEndian.Uint32(args[48:]))
	count := batch * layerCount * inputSize
	if batch <= 0 || layerCount <= 0 || inputSize <= 0 || inputBytes != count*4 || outputBytes != count*4 {
		return core.E("rocm.hip.FakeLaunch", "per-layer input transpose shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "per-layer input transpose input buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "per-layer input transpose output buffer is missing", nil)
	}
	for token := 0; token < batch; token++ {
		for layer := 0; layer < layerCount; layer++ {
			for item := 0; item < inputSize; item++ {
				src := ((token*layerCount+layer)*inputSize + item) * 4
				dst := ((layer*batch+token)*inputSize + item) * 4
				copy(outputData[outputOffset+dst:outputOffset+dst+4], inputData[inputOffset+src:inputOffset+src+4])
			}
		}
	}
	return nil
}

func (driver *fakeHIPDriver) launchSwiGLU(args []byte) error {
	if len(args) != hipSwiGLULaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "swiglu launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipSwiGLULaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipSwiGLULaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "swiglu launch header mismatch", nil)
	}
	gatePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	upPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	count := int(binary.LittleEndian.Uint32(args[32:]))
	gateBytes := int(binary.LittleEndian.Uint32(args[36:]))
	upBytes := int(binary.LittleEndian.Uint32(args[40:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[44:]))
	if count <= 0 || gateBytes != count*4 || upBytes != count*4 || outputBytes != count*4 {
		return core.E("rocm.hip.FakeLaunch", "swiglu shape metadata mismatch", nil)
	}
	gateData, gateOffset, ok := driver.memoryForPointer(gatePointer, gateBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "swiglu gate buffer is missing", nil)
	}
	upData, upOffset, ok := driver.memoryForPointer(upPointer, upBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "swiglu up buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "swiglu output buffer is missing", nil)
	}
	gate, err := hipFloat32PayloadValues(gateData[gateOffset : gateOffset+gateBytes])
	if err != nil {
		return err
	}
	up, err := hipFloat32PayloadValues(upData[upOffset : upOffset+upBytes])
	if err != nil {
		return err
	}
	out := make([]float32, count)
	for index := range out {
		out[index] = gate[index] / (1 + float32(math.Exp(float64(-gate[index])))) * up[index]
	}
	payload, err := hipFloat32Payload(out)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchGELUTanhMultiply(args []byte) error {
	if len(args) != hipGELUTanhMulLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "GELU tanh multiply launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipGELUTanhMulLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipGELUTanhMulLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "GELU tanh multiply launch header mismatch", nil)
	}
	gatePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	upPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	count := int(binary.LittleEndian.Uint32(args[32:]))
	gateBytes := int(binary.LittleEndian.Uint32(args[36:]))
	upBytes := int(binary.LittleEndian.Uint32(args[40:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[44:]))
	if count <= 0 || gateBytes != count*4 || upBytes != count*4 || outputBytes != count*4 {
		return core.E("rocm.hip.FakeLaunch", "GELU tanh multiply shape metadata mismatch", nil)
	}
	gateData, gateOffset, ok := driver.memoryForPointer(gatePointer, gateBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "GELU tanh multiply gate buffer is missing", nil)
	}
	upData, upOffset, ok := driver.memoryForPointer(upPointer, upBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "GELU tanh multiply up buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "GELU tanh multiply output buffer is missing", nil)
	}
	gate, err := hipFloat32PayloadValues(gateData[gateOffset : gateOffset+gateBytes])
	if err != nil {
		return err
	}
	up, err := hipFloat32PayloadValues(upData[upOffset : upOffset+upBytes])
	if err != nil {
		return err
	}
	out := make([]float32, count)
	const sqrt2OverPi = 0.7978845608028654
	const coeff = 0.044715
	for index := range out {
		value := float64(gate[index])
		gelu := 0.5 * value * (1 + math.Tanh(sqrt2OverPi*(value+coeff*value*value*value)))
		out[index] = float32(gelu) * up[index]
	}
	payload, err := hipFloat32Payload(out)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchTinyPrefill(args []byte) error {
	if len(args) != hipTinyPrefillLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipTinyPrefillLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipTinyPrefillLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill launch header mismatch", nil)
	}
	tokenPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	embeddingPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	logitPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	attentionPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	resultPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	keyPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[56:]))
	valuePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[64:]))
	tokenCount := int(binary.LittleEndian.Uint32(args[72:]))
	vocabSize := int(binary.LittleEndian.Uint32(args[76:]))
	hiddenSize := int(binary.LittleEndian.Uint32(args[80:]))
	tokenBytes := int(binary.LittleEndian.Uint32(args[84:]))
	embeddingBytes := int(binary.LittleEndian.Uint32(args[88:]))
	outputWeightBytes := int(binary.LittleEndian.Uint32(args[92:]))
	logitBytes := int(binary.LittleEndian.Uint32(args[96:]))
	attentionBytes := int(binary.LittleEndian.Uint32(args[100:]))
	resultBytes := int(binary.LittleEndian.Uint32(args[104:]))
	keyBytes := int(binary.LittleEndian.Uint32(args[108:]))
	valueBytes := int(binary.LittleEndian.Uint32(args[112:]))
	outputWeightEncoding := binary.LittleEndian.Uint32(args[116:])
	q8Scale := math.Float32frombits(binary.LittleEndian.Uint32(args[120:]))
	expectedOutputWeightBytes, err := hipTinyOutputWeightByteCount(outputWeightEncoding, uint64(outputWeightBytes), uint64(vocabSize*hiddenSize), q8Scale)
	if err != nil {
		return err
	}
	stateBytes := tokenCount * hiddenSize * 4
	if tokenCount <= 0 || vocabSize <= 0 || hiddenSize <= 0 ||
		tokenBytes != tokenCount*4 ||
		embeddingBytes != vocabSize*hiddenSize*4 ||
		outputWeightBytes != int(expectedOutputWeightBytes) ||
		logitBytes != vocabSize*4 ||
		attentionBytes != tokenCount*4 ||
		keyBytes != stateBytes ||
		valueBytes != stateBytes ||
		resultBytes != hipGreedyResultBytes {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill shape metadata mismatch", nil)
	}
	tokenData, tokenOffset, ok := driver.memoryForPointer(tokenPointer, tokenBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill token buffer is missing", nil)
	}
	embeddingData, embeddingOffset, ok := driver.memoryForPointer(embeddingPointer, embeddingBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill embedding buffer is missing", nil)
	}
	outputWeightData, outputWeightOffset, ok := driver.memoryForPointer(outputWeightPointer, outputWeightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill output weight buffer is missing", nil)
	}
	logitData, logitOffset, ok := driver.memoryForPointer(logitPointer, logitBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill logit buffer is missing", nil)
	}
	attentionData, attentionOffset, ok := driver.memoryForPointer(attentionPointer, attentionBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill attention buffer is missing", nil)
	}
	keyData, keyOffset, ok := driver.memoryForPointer(keyPointer, keyBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill key buffer is missing", nil)
	}
	valueData, valueOffset, ok := driver.memoryForPointer(valuePointer, valueBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill value buffer is missing", nil)
	}
	resultData, resultOffset, ok := driver.memoryForPointer(resultPointer, resultBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill result buffer is missing", nil)
	}
	tokens := make([]int32, tokenCount)
	for index := range tokens {
		tokens[index] = int32(binary.LittleEndian.Uint32(tokenData[tokenOffset+index*4:]))
	}
	embedding, err := hipFloat32PayloadValues(embeddingData[embeddingOffset : embeddingOffset+embeddingBytes])
	if err != nil {
		return err
	}
	outputWeights, err := hipTinyOutputWeightValues(outputWeightData[outputWeightOffset:outputWeightOffset+outputWeightBytes], outputWeightEncoding, q8Scale)
	if err != nil {
		return err
	}
	result, err := hipReferenceTinyPrefill(hipReferenceTinyLMConfig{
		EmbeddingTable: embedding,
		OutputWeights:  outputWeights,
		VocabSize:      vocabSize,
		HiddenSize:     hiddenSize,
	}, tokens)
	if err != nil {
		return err
	}
	logitPayload, err := hipFloat32Payload(result.Logits)
	if err != nil {
		return err
	}
	attentionPayload, err := hipFloat32Payload(result.Attention)
	if err != nil {
		return err
	}
	keyPayload, err := hipFloat32Payload(flattenHIPReferenceMatrix(result.State.Keys))
	if err != nil {
		return err
	}
	valuePayload, err := hipFloat32Payload(flattenHIPReferenceMatrix(result.State.Values))
	if err != nil {
		return err
	}
	copy(logitData[logitOffset:logitOffset+logitBytes], logitPayload)
	copy(attentionData[attentionOffset:attentionOffset+attentionBytes], attentionPayload)
	copy(keyData[keyOffset:keyOffset+keyBytes], keyPayload)
	copy(valueData[valueOffset:valueOffset+valueBytes], valuePayload)
	binary.LittleEndian.PutUint32(resultData[resultOffset:], uint32(int32(result.NextTokenID)))
	binary.LittleEndian.PutUint32(resultData[resultOffset+4:], math.Float32bits(result.NextScore))
	return nil
}

func (driver *fakeHIPDriver) launchTinyDecode(args []byte) error {
	if len(args) != hipTinyDecodeLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "tiny decode launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipTinyDecodeLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipTinyDecodeLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "tiny decode launch header mismatch", nil)
	}
	priorKeyPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	priorValuePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	embeddingPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	outputWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	logitPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	attentionPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	updatedKeyPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[56:]))
	updatedValuePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[64:]))
	resultPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[72:]))
	tokenID := int32(binary.LittleEndian.Uint32(args[80:]))
	priorTokenCount := int(binary.LittleEndian.Uint32(args[84:]))
	vocabSize := int(binary.LittleEndian.Uint32(args[88:]))
	hiddenSize := int(binary.LittleEndian.Uint32(args[92:]))
	priorKeyBytes := int(binary.LittleEndian.Uint32(args[96:]))
	priorValueBytes := int(binary.LittleEndian.Uint32(args[100:]))
	embeddingBytes := int(binary.LittleEndian.Uint32(args[104:]))
	outputWeightBytes := int(binary.LittleEndian.Uint32(args[108:]))
	logitBytes := int(binary.LittleEndian.Uint32(args[112:]))
	attentionBytes := int(binary.LittleEndian.Uint32(args[116:]))
	updatedKeyBytes := int(binary.LittleEndian.Uint32(args[120:]))
	updatedValueBytes := int(binary.LittleEndian.Uint32(args[124:]))
	resultBytes := int(binary.LittleEndian.Uint32(args[128:]))
	outputWeightEncoding := binary.LittleEndian.Uint32(args[132:])
	q8Scale := math.Float32frombits(binary.LittleEndian.Uint32(args[136:]))
	expectedOutputWeightBytes, err := hipTinyOutputWeightByteCount(outputWeightEncoding, uint64(outputWeightBytes), uint64(vocabSize*hiddenSize), q8Scale)
	if err != nil {
		return err
	}
	if tokenID < 0 || priorTokenCount <= 0 || vocabSize <= 0 || hiddenSize <= 0 ||
		int(tokenID) >= vocabSize ||
		priorKeyBytes != priorTokenCount*hiddenSize*4 ||
		priorValueBytes != priorTokenCount*hiddenSize*4 ||
		embeddingBytes != vocabSize*hiddenSize*4 ||
		outputWeightBytes != int(expectedOutputWeightBytes) ||
		logitBytes != vocabSize*4 ||
		attentionBytes != (priorTokenCount+1)*4 ||
		updatedKeyBytes != (priorTokenCount+1)*hiddenSize*4 ||
		updatedValueBytes != (priorTokenCount+1)*hiddenSize*4 ||
		resultBytes != hipGreedyResultBytes {
		return core.E("rocm.hip.FakeLaunch", "tiny decode shape metadata mismatch", nil)
	}
	priorKeyData, priorKeyOffset, ok := driver.memoryForPointer(priorKeyPointer, priorKeyBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode prior key buffer is missing", nil)
	}
	priorValueData, priorValueOffset, ok := driver.memoryForPointer(priorValuePointer, priorValueBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode prior value buffer is missing", nil)
	}
	embeddingData, embeddingOffset, ok := driver.memoryForPointer(embeddingPointer, embeddingBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode embedding buffer is missing", nil)
	}
	outputWeightData, outputWeightOffset, ok := driver.memoryForPointer(outputWeightPointer, outputWeightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode output weight buffer is missing", nil)
	}
	logitData, logitOffset, ok := driver.memoryForPointer(logitPointer, logitBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode logit buffer is missing", nil)
	}
	attentionData, attentionOffset, ok := driver.memoryForPointer(attentionPointer, attentionBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode attention buffer is missing", nil)
	}
	updatedKeyData, updatedKeyOffset, ok := driver.memoryForPointer(updatedKeyPointer, updatedKeyBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode updated key buffer is missing", nil)
	}
	updatedValueData, updatedValueOffset, ok := driver.memoryForPointer(updatedValuePointer, updatedValueBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode updated value buffer is missing", nil)
	}
	resultData, resultOffset, ok := driver.memoryForPointer(resultPointer, resultBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode result buffer is missing", nil)
	}
	priorKeysFlat, err := hipFloat32PayloadValues(priorKeyData[priorKeyOffset : priorKeyOffset+priorKeyBytes])
	if err != nil {
		return err
	}
	priorValuesFlat, err := hipFloat32PayloadValues(priorValueData[priorValueOffset : priorValueOffset+priorValueBytes])
	if err != nil {
		return err
	}
	priorKeys, err := splitHIPReferenceVectors(priorKeysFlat, hiddenSize)
	if err != nil {
		return err
	}
	priorValues, err := splitHIPReferenceVectors(priorValuesFlat, hiddenSize)
	if err != nil {
		return err
	}
	embedding, err := hipFloat32PayloadValues(embeddingData[embeddingOffset : embeddingOffset+embeddingBytes])
	if err != nil {
		return err
	}
	outputWeights, err := hipTinyOutputWeightValues(outputWeightData[outputWeightOffset:outputWeightOffset+outputWeightBytes], outputWeightEncoding, q8Scale)
	if err != nil {
		return err
	}
	result, err := hipReferenceTinyDecode(hipReferenceTinyLMConfig{
		EmbeddingTable: embedding,
		OutputWeights:  outputWeights,
		VocabSize:      vocabSize,
		HiddenSize:     hiddenSize,
	}, hipReferenceTinyLMState{Keys: priorKeys, Values: priorValues}, tokenID)
	if err != nil {
		return err
	}
	logitPayload, err := hipFloat32Payload(result.Logits)
	if err != nil {
		return err
	}
	attentionPayload, err := hipFloat32Payload(result.Attention)
	if err != nil {
		return err
	}
	updatedKeysPayload, err := hipFloat32Payload(flattenHIPReferenceMatrix(result.State.Keys))
	if err != nil {
		return err
	}
	updatedValuesPayload, err := hipFloat32Payload(flattenHIPReferenceMatrix(result.State.Values))
	if err != nil {
		return err
	}
	copy(logitData[logitOffset:logitOffset+logitBytes], logitPayload)
	copy(attentionData[attentionOffset:attentionOffset+attentionBytes], attentionPayload)
	copy(updatedKeyData[updatedKeyOffset:updatedKeyOffset+updatedKeyBytes], updatedKeysPayload)
	copy(updatedValueData[updatedValueOffset:updatedValueOffset+updatedValueBytes], updatedValuesPayload)
	binary.LittleEndian.PutUint32(resultData[resultOffset:], uint32(int32(result.NextTokenID)))
	binary.LittleEndian.PutUint32(resultData[resultOffset+4:], math.Float32bits(result.NextScore))
	return nil
}

func TestCGOHIPLaunchArgArenaSlotPointer_Good(t *testing.T) {
	base := nativeDevicePointer(0x1000)
	slotIndex := cgoHIPLaunchArgRingSize - 1

	pointer, ok := cgoHIPLaunchArgArenaSlotPointer(base, slotIndex, 248)

	core.AssertTrue(t, ok)
	core.AssertEqual(t, base+nativeDevicePointer(slotIndex*cgoHIPLaunchArgSlotBytes), pointer)
}

func TestCGOHIPLaunchArgArenaSlotPointer_Bad(t *testing.T) {
	pointer, ok := cgoHIPLaunchArgArenaSlotPointer(0x1000, 0, cgoHIPLaunchArgSlotBytes+1)

	core.AssertFalse(t, ok)
	core.AssertEqual(t, nativeDevicePointer(0), pointer)
}

func TestCGOHIPLaunchArgArenaSlotPointer_Ugly(t *testing.T) {
	for _, slotIndex := range []int{-1, cgoHIPLaunchArgRingSize} {
		pointer, ok := cgoHIPLaunchArgArenaSlotPointer(0x1000, slotIndex, cgoHIPLaunchArgSlotBytes)
		core.AssertFalse(t, ok)
		core.AssertEqual(t, nativeDevicePointer(0), pointer)
	}
}

func nativeHIPTensorGGUF(t *testing.T) (string, int64) {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "weights.gguf")
	result := core.WriteFile(path, []byte("0123456789abcdef0123456789abcdef"), 0o644)
	core.RequireTrue(t, result.OK)
	return path, 0
}
