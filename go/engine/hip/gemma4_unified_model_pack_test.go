// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestModelPackInspectorGemma4Unified12BQ6(t *testing.T) {
	dir := core.PathJoin(t.TempDir(), "lmstudio-community-gemma-4-12b-it-6bit")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q): %v", dir, err)
	}
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures":["Gemma4UnifiedForConditionalGeneration"],
		"model_type":"gemma4_unified",
		"text_config":{
			"model_type":"gemma4_unified_text",
			"hidden_size":3840,
			"intermediate_size":15360,
			"num_hidden_layers":48,
			"num_attention_heads":16,
			"num_key_value_heads":8,
			"num_global_key_value_heads":1,
			"head_dim":256,
			"global_head_dim":512,
			"attention_k_eq_v":true,
			"max_position_embeddings":262144,
			"sliding_window":1024,
			"vocab_size":262144,
			"vocab_size_per_layer_input":262144
		},
		"quantization":{
			"bits":6,
			"group_size":64
		}
	}`)
	writeNativeContractSafetensors(t, core.PathJoin(dir, "model.safetensors"))

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{
		available: true,
		device:    nativeDeviceInfo{Name: "AMD Radeon RX 7800 XT", MemoryBytes: 16 * memoryGiB, FreeBytes: 12 * memoryGiB, Driver: "hip-test"},
	}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !inspection.Supported || inspection.Model.Architecture != "gemma4_unified" {
		t.Fatalf("inspection = %+v labels=%+v, want supported Gemma4 unified pack", inspection, inspection.Labels)
	}
	if inspection.Model.ContextLength != 262144 ||
		inspection.Model.NumLayers != 48 ||
		inspection.Model.HiddenSize != 3840 ||
		inspection.Model.VocabSize != 262144 ||
		inspection.Model.QuantBits != 6 ||
		inspection.Model.QuantGroup != 64 {
		t.Fatalf("model = %+v, want 12B unified q6 dimensions", inspection.Model)
	}
	if inspection.Labels["architecture_supported"] != "true" ||
		inspection.Labels["gemma4_size"] != "12B" ||
		inspection.Labels["gemma4_quant_mode"] != "q6" ||
		inspection.Labels["gemma4_pack_supported"] != "true" ||
		inspection.Labels["gemma4_runtime"] != Gemma4RuntimeMLXAffine ||
		inspection.Labels["gemma4_generate_status"] != Gemma4GenerateLinked ||
		inspection.Labels["engine_state_context_route_contract"] != ROCmStateContextRegistryContract ||
		inspection.Labels["engine_state_context_window"] != "262144" ||
		inspection.Labels["engine_state_context_prompt_replay_refused"] != "true" ||
		inspection.Labels["engine_state_context_remaining_context_default"] != "true" ||
		inspection.Labels["engine_state_context_gemma4_size"] != "12B" ||
		inspection.Labels["engine_state_context_gemma4_quant_mode"] != "q6" ||
		inspection.Labels["engine_attached_drafter_route_contract"] != ROCmAttachedDrafterRegistryContract ||
		inspection.Labels["engine_attached_drafter_target_architecture"] != "gemma4_unified" ||
		inspection.Labels["engine_attached_drafter_assistant_architecture"] != "gemma4_assistant" ||
		inspection.Labels["engine_attached_drafter_native_attachment"] != hipKernelStatusNotLinked ||
		inspection.Labels["engine_attached_drafter_retained_state_required"] != "true" ||
		inspection.Labels["engine_attached_drafter_prompt_replay_refused"] != "true" ||
		inspection.Labels["attention_global_kv_heads"] != "1" ||
		inspection.Labels["attention_global_head_dim"] != "512" ||
		inspection.Labels["sliding_window"] != "1024" {
		t.Fatalf("labels = %+v, want unified Gemma4 12B q6 support and attention metadata", inspection.Labels)
	}
	stateRoute, ok := ROCmStateContextRouteForInspection(inspection)
	if !ok ||
		stateRoute.ContextWindow != 262144 ||
		stateRoute.Gemma4Size != "12B" ||
		stateRoute.Gemma4QuantMode != "q6" ||
		!stateRoute.RuntimeOwnedKV ||
		!stateRoute.PromptReplayRefused {
		t.Fatalf("state context route = %+v ok=%v, want retained-state 12B q6 inspection route", stateRoute, ok)
	}
	attachedRoute, ok := ROCmAttachedDrafterRouteForInspection(inspection)
	if !ok ||
		attachedRoute.TargetArchitecture != "gemma4_unified" ||
		attachedRoute.AssistantArchitecture != "gemma4_assistant" ||
		attachedRoute.NativeAttachment != hipKernelStatusNotLinked ||
		!attachedRoute.RetainedStateRequired ||
		!attachedRoute.PromptReplayRefused ||
		!attachedRoute.DraftDetection {
		t.Fatalf("attached drafter route = %+v ok=%v, want native-pending Gemma4 assistant inspection route", attachedRoute, ok)
	}
	generate, ok := nativeInspectionCapability(inspection, inference.CapabilityGenerate)
	if !ok || generate.Status != inference.CapabilityStatusExperimental ||
		generate.Labels["gemma4_size"] != "12B" ||
		generate.Labels["gemma4_generate_status"] != Gemma4GenerateLinked {
		t.Fatalf("generate capability = %+v ok=%v, want 12B q6 linked generation metadata", generate, ok)
	}
}

func TestModelPackInspectorGemma4E4BPathQuantSupport(t *testing.T) {
	root := core.PathJoin(t.TempDir(), "gemma-4-e4b-it-6bit")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q): %v", root, err)
	}
	writeNativeContractFile(t, core.PathJoin(root, "config.json"), `{
		"architectures":["Gemma4ForCausalLM"],
		"model_type":"gemma4_text",
		"hidden_size":2304,
		"num_hidden_layers":26,
		"num_attention_heads":8,
		"num_key_value_heads":4,
		"head_dim":256,
		"vocab_size":262144,
		"max_position_embeddings":131072,
		"quantization":{"bits":6,"group_size":64}
	}`)
	writeNativeContractSafetensors(t, core.PathJoin(root, "model.safetensors"))

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), root)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !inspection.Supported ||
		inspection.Labels["gemma4_size"] != "E4B" ||
		inspection.Labels["gemma4_quant_mode"] != "q6" ||
		inspection.Labels["gemma4_pack_supported"] != "true" ||
		inspection.Labels["gemma4_generate_status"] != Gemma4GenerateLinked {
		t.Fatalf("inspection = %+v labels=%+v, want E4B q6 support labels from path", inspection, inspection.Labels)
	}
}

func TestModelPackInspectorGemma4PathOnlyQuantSupport(t *testing.T) {
	for _, tc := range []struct {
		name        string
		path        string
		size        string
		hiddenSize  int
		layers      int
		quantMode   string
		quantBits   int
		quantGroup  int
		status      string
		runtime     string
		supported   bool
		hasGenerate bool
	}{
		{name: "e2b-bf16", path: "lmstudio-community-gemma-4-e2b-it-bf16", size: "E2B", hiddenSize: 1536, layers: 35, quantMode: "bf16", quantBits: 16, status: Gemma4GenerateLinked, runtime: Gemma4RuntimeBF16, supported: true, hasGenerate: true},
		{name: "e2b-8bit", path: "lmstudio-community-gemma-4-e2b-it-8bit", size: "E2B", hiddenSize: 1536, layers: 35, quantMode: "q8", quantBits: 8, quantGroup: 64, status: Gemma4GenerateLinked, runtime: Gemma4RuntimeMLXAffine, supported: true, hasGenerate: true},
		{name: "e2b-6bit", path: "lmstudio-community-gemma-4-e2b-it-6bit", size: "E2B", hiddenSize: 1536, layers: 35, quantMode: "q6", quantBits: 6, quantGroup: 64, status: Gemma4GenerateLinked, runtime: Gemma4RuntimeMLXAffine, supported: true, hasGenerate: true},
		{name: "e2b-4bit", path: "lmstudio-community-gemma-4-e2b-it-4bit", size: "E2B", hiddenSize: 1536, layers: 35, quantMode: "q4", quantBits: 4, quantGroup: 64, status: Gemma4GenerateLinked, runtime: Gemma4RuntimeMLXAffine, supported: true, hasGenerate: true},
		{name: "e2b-mxfp8", path: "lmstudio-community-gemma-4-e2b-it-mxfp8", size: "E2B", hiddenSize: 1536, layers: 35, quantMode: "mxfp8", quantBits: 8, quantGroup: 32, status: Gemma4GeneratePlannedOnly, runtime: Gemma4RuntimePlanned},
		{name: "e2b-mxfp4", path: "lmstudio-community-gemma-4-e2b-it-mxfp4", size: "E2B", hiddenSize: 1536, layers: 35, quantMode: "mxfp4", quantBits: 4, quantGroup: 32, status: Gemma4GeneratePlannedOnly, runtime: Gemma4RuntimePlanned},
		{name: "e4b-bf16", path: "lmstudio-community-gemma-4-e4b-it-bf16", size: "E4B", hiddenSize: 2304, layers: 26, quantMode: "bf16", quantBits: 16, status: Gemma4GenerateLinked, runtime: Gemma4RuntimeBF16, supported: true, hasGenerate: true},
		{name: "e4b-8bit", path: "lmstudio-community-gemma-4-e4b-it-8bit", size: "E4B", hiddenSize: 2304, layers: 26, quantMode: "q8", quantBits: 8, quantGroup: 64, status: Gemma4GenerateLinked, runtime: Gemma4RuntimeMLXAffine, supported: true, hasGenerate: true},
		{name: "e4b-6bit", path: "lmstudio-community-gemma-4-e4b-it-6bit", size: "E4B", hiddenSize: 2304, layers: 26, quantMode: "q6", quantBits: 6, quantGroup: 64, status: Gemma4GenerateLinked, runtime: Gemma4RuntimeMLXAffine, supported: true, hasGenerate: true},
		{name: "e4b-4bit", path: "lmstudio-community-gemma-4-e4b-it-4bit", size: "E4B", hiddenSize: 2304, layers: 26, quantMode: "q4", quantBits: 4, quantGroup: 64, status: Gemma4GenerateLinked, runtime: Gemma4RuntimeMLXAffine, supported: true, hasGenerate: true},
		{name: "e4b-mxfp8", path: "lmstudio-community-gemma-4-e4b-it-mxfp8", size: "E4B", hiddenSize: 2304, layers: 26, quantMode: "mxfp8", quantBits: 8, quantGroup: 32, status: Gemma4GeneratePlannedOnly, runtime: Gemma4RuntimePlanned},
		{name: "e4b-mxfp4", path: "lmstudio-community-gemma-4-e4b-it-mxfp4", size: "E4B", hiddenSize: 2304, layers: 26, quantMode: "mxfp4", quantBits: 4, quantGroup: 32, status: Gemma4GeneratePlannedOnly, runtime: Gemma4RuntimePlanned},
		{name: "12b-6bit", path: "lmstudio-community-gemma-4-12b-it-6bit", size: "12B", hiddenSize: 3840, layers: 48, quantMode: "q6", quantBits: 6, quantGroup: 64, status: Gemma4GenerateLinked, runtime: Gemma4RuntimeMLXAffine, supported: true, hasGenerate: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := core.PathJoin(t.TempDir(), tc.path)
			if err := os.MkdirAll(root, 0o755); err != nil {
				t.Fatalf("MkdirAll(%q): %v", root, err)
			}
			writeNativeContractFile(t, core.PathJoin(root, "config.json"), core.Sprintf(`{
				"architectures":["Gemma4ForCausalLM"],
				"model_type":"gemma4_text",
				"hidden_size":%d,
				"num_hidden_layers":%d,
				"num_attention_heads":8,
				"num_key_value_heads":4,
				"head_dim":256,
				"vocab_size":262144,
				"max_position_embeddings":131072
			}`, tc.hiddenSize, tc.layers))
			writeNativeContractSafetensors(t, core.PathJoin(root, "model.safetensors"))

			inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), root)
			if err != nil {
				t.Fatalf("InspectModelPack: %v", err)
			}
			if inspection.Supported != tc.supported ||
				inspection.Model.QuantBits != tc.quantBits ||
				inspection.Model.QuantGroup != tc.quantGroup ||
				inspection.Labels["gemma4_size"] != tc.size ||
				inspection.Labels["gemma4_quant_mode"] != tc.quantMode ||
				inspection.Labels["gemma4_pack_supported"] != "true" ||
				inspection.Labels["gemma4_runtime"] != tc.runtime ||
				inspection.Labels["gemma4_generate_status"] != tc.status {
				t.Fatalf("inspection = %+v labels=%+v, want %s %s path-only support", inspection, inspection.Labels, tc.size, tc.quantMode)
			}
			generate, ok := nativeInspectionCapability(inspection, inference.CapabilityGenerate)
			if ok != tc.hasGenerate {
				t.Fatalf("generate capability = %+v ok=%v, want ok=%v", generate, ok, tc.hasGenerate)
			}
			if tc.status == Gemma4GeneratePlannedOnly {
				modelLoad, ok := nativeInspectionCapability(inspection, inference.CapabilityModelLoad)
				if !ok ||
					modelLoad.Labels["gemma4_size"] != tc.size ||
					modelLoad.Labels["gemma4_quant_mode"] != tc.quantMode ||
					modelLoad.Labels["gemma4_generate_status"] != Gemma4GeneratePlannedOnly {
					t.Fatalf("model-load capability = %+v ok=%v, want planned-only %s %s labels", modelLoad, ok, tc.size, tc.quantMode)
				}
			}
		})
	}
}

func TestModelPackInspectorGemma4ShapeQuantFailsClosed(t *testing.T) {
	root := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(root, "config.json"), `{
		"architectures":["Gemma4ForCausalLM"],
		"model_type":"gemma4_text",
		"hidden_size":2304,
		"num_hidden_layers":26,
		"num_attention_heads":8,
		"num_key_value_heads":4,
		"head_dim":256,
		"vocab_size":262144,
		"max_position_embeddings":131072,
		"quantization":{"bits":6,"group_size":64}
	}`)
	writeNativeContractSafetensors(t, core.PathJoin(root, "model.safetensors"))

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), root)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !inspection.Supported ||
		inspection.Model.Architecture != "gemma4_text" ||
		inspection.Model.HiddenSize != 2304 ||
		inspection.Model.NumLayers != 26 ||
		inspection.Model.QuantBits != 6 {
		t.Fatalf("inspection = %+v labels=%+v, want supported Gemma4 metadata without shape-derived linked generation", inspection, inspection.Labels)
	}
	if inspection.Labels["gemma4_size"] != "" ||
		inspection.Labels["gemma4_quant_mode"] != "q6" ||
		inspection.Labels["gemma4_pack_supported"] != "" ||
		inspection.Labels["gemma4_generate_status"] != "" {
		t.Fatalf("labels = %+v, shape-only metadata must record quant mode without declaring Gemma4 size/generate support", inspection.Labels)
	}
	if generate, ok := nativeInspectionCapability(inspection, inference.CapabilityGenerate); ok {
		t.Fatalf("generate capability = %+v, shape-only metadata must not expose linked generation", generate)
	}
}

func TestModelPackInspectorGemma4GGUFLoadOnlySupport(t *testing.T) {
	path := writeGemma4ModelPackGGUF(t)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), path)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !inspection.Supported ||
		inspection.Format != "gguf" ||
		inspection.Model.Architecture != "gemma4" ||
		inspection.Model.ContextLength != 131072 ||
		inspection.Model.NumLayers != productionLaneGemma4E2BLayers ||
		inspection.Model.QuantBits != 4 ||
		inspection.Labels["gemma4_size"] != "E2B" ||
		inspection.Labels["gemma4_quant_mode"] != "q4" ||
		inspection.Labels["gemma4_pack_supported"] != "true" ||
		inspection.Labels["gemma4_runtime"] != Gemma4RuntimeGGUF ||
		inspection.Labels["gemma4_source_format"] != "gguf" ||
		inspection.Labels["gemma4_generate_status"] != Gemma4GenerateLoadOnly {
		t.Fatalf("inspection = %+v labels=%+v, want Gemma4 E2B q4 GGUF load-only support", inspection, inspection.Labels)
	}
	modelLoad, ok := nativeInspectionCapability(inspection, inference.CapabilityModelLoad)
	if !ok || modelLoad.Labels["gemma4_runtime"] != Gemma4RuntimeGGUF ||
		modelLoad.Labels["gemma4_source_format"] != "gguf" ||
		modelLoad.Labels["gemma4_generate_status"] != Gemma4GenerateLoadOnly ||
		modelLoad.Labels["production_quant_runtime"] != Gemma4RuntimeGGUF ||
		modelLoad.Labels["production_quant_generate_status"] != Gemma4GenerateLoadOnly {
		t.Fatalf("model-load capability = %+v ok=%v, want GGUF load-only metadata", modelLoad, ok)
	}
	chatTemplate, ok := nativeInspectionCapability(inspection, inference.CapabilityChatTemplate)
	if !ok ||
		chatTemplate.Labels["chat_template"] != "gemma4_hf_turn" ||
		chatTemplate.Labels["engine_tokenizer_route_contract"] != ROCmModelTokenizerRegistryContract ||
		chatTemplate.Labels["engine_tokenizer_chat_template_id"] != "gemma4_hf_turn" ||
		chatTemplate.Labels["gemma4_source_format"] != "gguf" ||
		chatTemplate.Labels["gemma4_generate_status"] != Gemma4GenerateLoadOnly ||
		chatTemplate.Labels["production_quant_runtime"] != Gemma4RuntimeGGUF ||
		chatTemplate.Labels["production_quant_generate_status"] != Gemma4GenerateLoadOnly {
		t.Fatalf("chat-template capability = %+v ok=%v, want GGUF Gemma4 template metadata", chatTemplate, ok)
	}
	for _, id := range []inference.CapabilityID{
		inference.CapabilityModelFit,
		inference.CapabilityMemoryPlanning,
		inference.CapabilityKVCachePlanning,
	} {
		capability, ok := nativeInspectionCapability(inspection, id)
		if !ok ||
			capability.Labels["gemma4_source_format"] != "gguf" ||
			capability.Labels["gemma4_generate_status"] != Gemma4GenerateLoadOnly ||
			capability.Labels["production_quant_policy"] != "gemma4_mlx_affine" ||
			capability.Labels["production_quant_runtime"] != Gemma4RuntimeGGUF ||
			capability.Labels["production_quant_generate_status"] != Gemma4GenerateLoadOnly {
			t.Fatalf("capability %s = %+v ok=%v, want GGUF Gemma4 load-only planning metadata", id, capability, ok)
		}
	}
	if generate, ok := nativeInspectionCapability(inspection, inference.CapabilityGenerate); ok {
		t.Fatalf("generate capability = %+v, GGUF pack must not claim linked MLX-affine generation", generate)
	}
}

func TestModelPackInspectorGemma4BF16LinkedSupport(t *testing.T) {
	for _, tc := range []struct {
		name   string
		path   string
		size   string
		config string
	}{
		{
			name: "e2b",
			size: "E2B",
			config: `{
				"architectures":["Gemma4ForCausalLM"],
				"model_type":"gemma4_text",
				"dtype":"bfloat16",
				"hidden_size":1536,
				"num_hidden_layers":35,
				"num_attention_heads":8,
				"num_key_value_heads":4,
				"head_dim":256,
				"vocab_size":262144,
				"max_position_embeddings":131072
			}`,
		},
		{
			name: "e4b",
			path: "gemma-4-e4b-it-bf16",
			size: "E4B",
			config: `{
				"architectures":["Gemma4ForCausalLM"],
				"model_type":"gemma4_text",
				"dtype":"bfloat16",
				"hidden_size":2304,
				"num_hidden_layers":26,
				"num_attention_heads":8,
				"num_key_value_heads":4,
				"head_dim":256,
				"vocab_size":262144,
				"max_position_embeddings":131072
			}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			if tc.path != "" {
				root = core.PathJoin(root, tc.path)
				if err := os.MkdirAll(root, 0o755); err != nil {
					t.Fatalf("MkdirAll(%q): %v", root, err)
				}
			}
			writeNativeContractFile(t, core.PathJoin(root, "config.json"), tc.config)
			writeNativeContractSafetensors(t, core.PathJoin(root, "model.safetensors"))

			inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), root)
			if err != nil {
				t.Fatalf("InspectModelPack: %v", err)
			}
			if !inspection.Supported ||
				inspection.Model.QuantType != "bf16" ||
				inspection.Labels["gemma4_size"] != tc.size ||
				inspection.Labels["gemma4_quant_mode"] != "bf16" ||
				inspection.Labels["gemma4_pack_supported"] != "true" ||
				inspection.Labels["gemma4_runtime"] != Gemma4RuntimeBF16 ||
				inspection.Labels["gemma4_generate_status"] != Gemma4GenerateLinked {
				t.Fatalf("inspection = %+v labels=%+v, want %s BF16 linked support", inspection, inspection.Labels, tc.size)
			}
			for _, id := range []inference.CapabilityID{
				inference.CapabilityChatTemplate,
				inference.CapabilityModelFit,
				inference.CapabilityMemoryPlanning,
				inference.CapabilityKVCachePlanning,
			} {
				capability, ok := nativeInspectionCapability(inspection, id)
				if !ok ||
					capability.Labels["gemma4_size"] != tc.size ||
					capability.Labels["gemma4_quant_mode"] != "bf16" ||
					capability.Labels["gemma4_generate_status"] != Gemma4GenerateLinked ||
					capability.Labels["production_quant_policy"] != "gemma4_mlx_affine" ||
					capability.Labels["production_quant_runtime"] != Gemma4RuntimeBF16 ||
					capability.Labels["production_quant_generate_status"] != Gemma4GenerateLinked {
					t.Fatalf("capability %s = %+v ok=%v, want %s BF16 linked metadata", id, capability, ok, tc.size)
				}
				if id == inference.CapabilityChatTemplate &&
					(capability.Labels["engine_tokenizer_route_contract"] != ROCmModelTokenizerRegistryContract ||
						capability.Labels["engine_tokenizer_chat_template_id"] != "gemma4_hf_turn") {
					t.Fatalf("chat-template capability = %+v ok=%v, want %s BF16 tokenizer route labels", capability, ok, tc.size)
				}
			}
			if generate, ok := nativeInspectionCapability(inspection, inference.CapabilityGenerate); !ok || generate.Labels["gemma4_generate_status"] != Gemma4GenerateLinked {
				t.Fatalf("generate capability = %+v ok=%v, BF16 pack must claim linked generation", generate, ok)
			}
		})
	}
}

func writeGemma4ModelPackGGUF(t *testing.T) string {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "gemma-4-e2b-it-q4.gguf")
	buf := core.NewBuffer()
	writeUint32 := func(v uint32) { core.RequireNoError(t, binary.Write(buf, binary.LittleEndian, v)) }
	writeUint64 := func(v uint64) { core.RequireNoError(t, binary.Write(buf, binary.LittleEndian, v)) }
	writeString := func(v string) {
		writeUint64(uint64(len(v)))
		_, err := buf.Write([]byte(v))
		core.RequireNoError(t, err)
	}
	writeKVString := func(key, value string) {
		writeString(key)
		writeUint32(8)
		writeString(value)
	}
	writeKVUint32 := func(key string, value uint32) {
		writeString(key)
		writeUint32(4)
		writeUint32(value)
	}

	writeUint32(0x46554747)
	writeUint32(3)
	writeUint64(0)
	writeUint64(6)
	writeKVString("general.architecture", "gemma4")
	writeKVString("general.name", "gemma4-e2b-test")
	writeKVString("general.size_label", "E2B")
	writeKVUint32("general.file_type", 15)
	writeKVUint32("gemma4.context_length", 131072)
	writeKVUint32("gemma4.block_count", productionLaneGemma4E2BLayers)

	result := core.WriteFile(path, buf.Bytes(), 0o644)
	core.RequireTrue(t, result.OK)
	return path
}

func writeGemma4ModelPackGGUFWithEmbeddingTensor(t *testing.T) string {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "gemma-4-e2b-it-q4.gguf")
	buf := core.NewBuffer()
	writeUint32 := func(v uint32) { core.RequireNoError(t, binary.Write(buf, binary.LittleEndian, v)) }
	writeUint64 := func(v uint64) { core.RequireNoError(t, binary.Write(buf, binary.LittleEndian, v)) }
	writeString := func(v string) {
		writeUint64(uint64(len(v)))
		_, err := buf.Write([]byte(v))
		core.RequireNoError(t, err)
	}
	writeKVString := func(key, value string) {
		writeString(key)
		writeUint32(8)
		writeString(value)
	}
	writeKVUint32 := func(key string, value uint32) {
		writeString(key)
		writeUint32(4)
		writeUint32(value)
	}
	writeKVFloat32 := func(key string, value float32) {
		writeString(key)
		writeUint32(6)
		core.RequireNoError(t, binary.Write(buf, binary.LittleEndian, value))
	}
	writeKVBool := func(key string, value bool) {
		writeString(key)
		writeUint32(7)
		if value {
			buf.WriteByte(1)
		} else {
			buf.WriteByte(0)
		}
	}
	writeTensorInfo := func(name string, tensorType uint32, dimensions ...uint64) {
		writeString(name)
		writeUint32(uint32(len(dimensions)))
		for _, dimension := range dimensions {
			writeUint64(dimension)
		}
		writeUint32(tensorType)
		writeUint64(0)
	}

	writeUint32(0x46554747)
	writeUint32(3)
	writeUint64(3)
	writeUint64(23)
	writeKVString("general.architecture", "gemma4")
	writeKVString("general.name", "gemma4-e2b-test")
	writeKVString("general.size_label", "E2B")
	writeKVUint32("general.file_type", 15)
	writeKVUint32("gemma4.context_length", 131072)
	writeKVUint32("gemma4.block_count", productionLaneGemma4E2BLayers)
	writeKVUint32("gemma4.embedding_length", productionLaneGemma4E2BHiddenSize)
	writeKVUint32("gemma4.feed_forward_length", 12288)
	writeKVUint32("gemma4.attention.head_count", 8)
	writeKVUint32("gemma4.attention.head_count_kv", 1)
	writeKVUint32("gemma4.attention.key_length", 512)
	writeKVUint32("gemma4.attention.value_length", 512)
	writeKVUint32("gemma4.attention.key_length_swa", 256)
	writeKVUint32("gemma4.attention.value_length_swa", 256)
	writeKVUint32("gemma4.attention.sliding_window", 512)
	writeKVUint32("gemma4.attention.shared_kv_layers", 20)
	writeKVUint32("gemma4.embedding_length_per_layer_input", 256)
	writeKVBool("gemma4.attention.sliding_window_pattern", false)
	writeKVFloat32("gemma4.rope.freq_base", 1000000)
	writeKVFloat32("gemma4.rope.freq_base_swa", 10000)
	writeKVUint32("gemma4.rope.dimension_count", 512)
	writeKVUint32("gemma4.rope.dimension_count_swa", 256)
	writeKVFloat32("gemma4.final_logit_softcapping", 30)
	writeTensorInfo("per_layer_token_embd.weight", 13, 8960, productionLaneGemma4E2BVocabSize)
	writeTensorInfo("token_embd.weight", 12, productionLaneGemma4E2BHiddenSize, productionLaneGemma4E2BVocabSize)
	writeTensorInfo("blk.0.attn_output.weight", 12, 2048, productionLaneGemma4E2BHiddenSize)

	result := core.WriteFile(path, buf.Bytes(), 0o644)
	core.RequireTrue(t, result.OK)
	return path
}

func TestModelPackInspectorGemma4Unified12BQ4QATSupported(t *testing.T) {
	dir := core.PathJoin(t.TempDir(), "lmstudio-community-gemma-4-12b-it-4bit")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q): %v", dir, err)
	}
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures":["Gemma4UnifiedForConditionalGeneration"],
		"model_type":"gemma4_unified",
		"text_config":{
			"model_type":"gemma4_unified_text",
			"hidden_size":3840,
			"num_hidden_layers":48,
			"num_attention_heads":16,
			"num_key_value_heads":8,
			"head_dim":256,
			"vocab_size":262144,
			"max_position_embeddings":262144
		},
		"quantization":{"bits":4,"group_size":64}
	}`)
	writeNativeContractSafetensors(t, core.PathJoin(dir, "model.safetensors"))

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{
		available: true,
		device:    nativeDeviceInfo{Name: "AMD Radeon RX 7800 XT", MemoryBytes: 16 * memoryGiB, FreeBytes: 12 * memoryGiB, Driver: "hip-test"},
	}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !inspection.Supported ||
		inspection.Labels["gemma4_size"] != "12B" ||
		inspection.Labels["gemma4_quant_mode"] != "q4" ||
		inspection.Labels["gemma4_pack_supported"] != "true" ||
		inspection.Labels["gemma4_runtime"] != Gemma4RuntimeMLXAffine ||
		inspection.Labels["gemma4_generate_status"] != Gemma4GenerateLinked ||
		inspection.Labels["gemma4_runnable_on_card"] != "true" {
		t.Fatalf("inspection = %+v labels=%+v, want supported 12B q4 QAT target", inspection, inspection.Labels)
	}
	for _, id := range []inference.CapabilityID{
		inference.CapabilityChatTemplate,
		inference.CapabilityModelFit,
		inference.CapabilityMemoryPlanning,
		inference.CapabilityKVCachePlanning,
	} {
		capability, ok := nativeInspectionCapability(inspection, id)
		if !ok ||
			capability.Labels["gemma4_size"] != "12B" ||
			capability.Labels["gemma4_quant_mode"] != "q4" ||
			capability.Labels["gemma4_pack_supported"] != "true" {
			t.Fatalf("capability %s = %+v ok=%v, want supported Gemma4 12B q4 metadata", id, capability, ok)
		}
		if id == inference.CapabilityChatTemplate &&
			(capability.Labels["engine_tokenizer_route_contract"] != ROCmModelTokenizerRegistryContract ||
				capability.Labels["engine_tokenizer_chat_template_id"] != "gemma4_hf_turn") {
			t.Fatalf("chat-template capability = %+v ok=%v, want Gemma4 tokenizer route labels", capability, ok)
		}
	}
	generate, ok := nativeInspectionCapability(inspection, inference.CapabilityGenerate)
	if !ok ||
		generate.Labels["gemma4_size"] != "12B" ||
		generate.Labels["gemma4_quant_mode"] != "q4" ||
		generate.Labels["gemma4_pack_supported"] != "true" ||
		generate.Labels["gemma4_generate_status"] != Gemma4GenerateLinked {
		t.Fatalf("generate capability = %+v ok=%v, want linked Gemma4 12B q4 generation metadata", generate, ok)
	}
}

func TestModelPackInspectorGemma4LargestPacksStatusOnly(t *testing.T) {
	for _, tc := range []struct {
		name string
		path string
		size string
		mode string
		bits int
	}{
		{name: "26b-a4b-q8", path: "gemma-4-26b-a4b-it-8bit", size: "26B-A4B", mode: "q8-status", bits: 8},
		{name: "26b-a4b-q6", path: "gemma-4-26b-a4b-it-6bit", size: "26B-A4B", mode: "q6-status", bits: 6},
		{name: "26b-a4b-q4", path: "gemma-4-26b-a4b-it-4bit", size: "26B-A4B", mode: "q4-status", bits: 4},
		{name: "31b-q8", path: "gemma-4-31b-it-8bit", size: "31B", mode: "q8-status", bits: 8},
		{name: "31b-q6", path: "gemma-4-31b-it-6bit", size: "31B", mode: "q6-status", bits: 6},
		{name: "31b-q4", path: "gemma-4-31b-it-4bit", size: "31B", mode: "q4-status", bits: 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := core.PathJoin(t.TempDir(), tc.path)
			if err := os.MkdirAll(root, 0o755); err != nil {
				t.Fatalf("MkdirAll(%q): %v", root, err)
			}
			writeNativeContractFile(t, core.PathJoin(root, "config.json"), core.Sprintf(`{
				"architectures":["Gemma4ForCausalLM"],
				"model_type":"gemma4_text",
				"hidden_size":4096,
				"num_hidden_layers":64,
				"num_attention_heads":16,
				"num_key_value_heads":8,
				"head_dim":256,
				"vocab_size":262144,
				"max_position_embeddings":131072,
				"quantization":{"bits":%d,"group_size":64}
			}`, tc.bits))
			writeNativeContractSafetensors(t, core.PathJoin(root, "model.safetensors"))

			inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), root)
			if err != nil {
				t.Fatalf("InspectModelPack: %v", err)
			}
			if inspection.Supported ||
				inspection.Labels["gemma4_size"] != tc.size ||
				inspection.Labels["gemma4_quant_mode"] != tc.mode ||
				inspection.Labels["gemma4_pack_supported"] != "true" ||
				inspection.Labels["gemma4_runtime"] != Gemma4RuntimePlanned ||
				inspection.Labels["gemma4_generate_status"] != Gemma4GeneratePlannedOnly ||
				inspection.Labels["gemma4_runnable_on_card"] != "false" {
				t.Fatalf("inspection = %+v labels=%+v, want %s %s status-only planned pack", inspection, inspection.Labels, tc.size, tc.mode)
			}
			modelLoad, ok := nativeInspectionCapability(inspection, inference.CapabilityModelLoad)
			if !ok ||
				modelLoad.Status != inference.CapabilityStatusPlanned ||
				modelLoad.Labels["gemma4_size"] != tc.size ||
				modelLoad.Labels["gemma4_quant_mode"] != tc.mode ||
				modelLoad.Labels["gemma4_generate_status"] != Gemma4GeneratePlannedOnly ||
				modelLoad.Labels["gemma4_runnable_on_card"] != "false" {
				t.Fatalf("model-load capability = %+v ok=%v, want %s %s planned status-only metadata", modelLoad, ok, tc.size, tc.mode)
			}
			if generate, ok := nativeInspectionCapability(inspection, inference.CapabilityGenerate); ok {
				t.Fatalf("generate capability = %+v, %s status-only pack must not claim linked generation", generate, tc.size)
			}
		})
	}
}
