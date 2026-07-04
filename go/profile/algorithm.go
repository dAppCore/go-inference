// SPDX-Licence-Identifier: EUPL-1.2

package profile

import (
	"dappco.re/go/inference"
	"dappco.re/go/inference/quant/autoround"
)

// AlgorithmRuntimeStatus is the go-mlx implementation state for a shared runtime algorithm.
type AlgorithmRuntimeStatus = inference.FeatureRuntimeStatus

const (
	AlgorithmRuntimeNative       = inference.FeatureRuntimeNative
	AlgorithmRuntimeExperimental = inference.FeatureRuntimeExperimental
	AlgorithmRuntimeMetadataOnly = inference.FeatureRuntimeMetadataOnly
	AlgorithmRuntimePlanned      = inference.FeatureRuntimePlanned
)

// AlgorithmProfile describes one backend-neutral algorithm or feature surface.
type AlgorithmProfile = inference.AlgorithmProfile

// BuiltinAlgorithmProfiles returns the algorithm feature matrix used in
// capability reports and backend planning.
func BuiltinAlgorithmProfiles() []AlgorithmProfile {
	profiles := builtinAlgorithmProfiles()
	out := make([]AlgorithmProfile, len(profiles))
	for i, profile := range profiles {
		out[i] = inference.CloneAlgorithmProfile(profile)
	}
	return out
}

// LookupAlgorithmProfile returns the built-in profile for id.
func LookupAlgorithmProfile(id inference.CapabilityID) (AlgorithmProfile, bool) {
	idx, ok := builtinAlgorithmProfileIndex[id]
	if !ok {
		return AlgorithmProfile{}, false
	}
	return inference.CloneAlgorithmProfile(builtinAlgorithmProfilesData[idx]), true
}

// builtinAlgorithmProfilesData is the singleton backing list — built once
// at package init, exposed through builtinAlgorithmProfiles. Callers must
// not mutate this slice or its entries; the public API clones before
// returning.
var builtinAlgorithmProfilesData = []AlgorithmProfile{}

// builtinAlgorithmProfileIndex maps each profile ID to its position in
// builtinAlgorithmProfilesData so LookupAlgorithmProfile resolves in
// O(1) instead of a linear scan over the 14-entry matrix.
var builtinAlgorithmProfileIndex = map[inference.CapabilityID]int{}

func init() {
	builtinAlgorithmProfilesData = buildBuiltinAlgorithmProfiles()
	builtinAlgorithmProfileIndex = make(map[inference.CapabilityID]int, len(builtinAlgorithmProfilesData))
	for i, profile := range builtinAlgorithmProfilesData {
		builtinAlgorithmProfileIndex[profile.ID] = i
	}
}

func builtinAlgorithmProfiles() []AlgorithmProfile {
	return builtinAlgorithmProfilesData
}

func buildBuiltinAlgorithmProfiles() []AlgorithmProfile {
	return []AlgorithmProfile{
		algorithmNative(inference.CapabilityScheduler, inference.CapabilityGroupRuntime, "scheduler", "bounded request queueing, stream backpressure, cancellation IDs, and latency metrics are implemented"),
		algorithmNative(inference.CapabilityRequestCancel, inference.CapabilityGroupRuntime, "request-cancel", "generation and scheduled requests can be cancelled through context/cancellation IDs"),
		algorithmNative(inference.CapabilityCacheBlocks, inference.CapabilityGroupRuntime, "block-prefix-cache", "block-prefix cache identity and State-backed KV block warm are implemented"),
		algorithmNative(inference.CapabilityCacheWarm, inference.CapabilityGroupRuntime, "cache-warm", "prompt and KV block warm paths are implemented"),
		algorithmNative(inference.CapabilityReasoningParse, inference.CapabilityGroupModel, "reasoning-parser", "model-aware thinking/reasoning parsers are available"),
		algorithmNative(inference.CapabilityToolParse, inference.CapabilityGroupModel, "tool-parser", "XML and OpenAI-style JSON tool-call parsing is available"),
		{
			ID:               inference.CapabilityJANGTQ,
			Group:            inference.CapabilityGroupRuntime,
			CapabilityStatus: inference.CapabilityStatusExperimental,
			RuntimeStatus:    AlgorithmRuntimeMetadataOnly,
			Algorithm:        "jangtq",
			Detail:           "JANG/JANGTQ metadata, packed tensor descriptors, CPU reference dequant, native q2/q8 Metal dequant parity, composed and fused packed expert projection, selected-expert safetensor loading, MiniMax packed layer skeleton with dense router projection, memory planning, parser hints, and model-pack validation are wired; full model execution is pending",
			Architectures:    []string{"minimax_m2"},
			Provides:         []string{"quantization.profile", "packed_tensor.descriptor", "reference.dequant", "memory.hints"},
		},
		{
			ID:               inference.CapabilityCodebookVQ,
			Group:            inference.CapabilityGroupRuntime,
			CapabilityStatus: inference.CapabilityStatusExperimental,
			RuntimeStatus:    AlgorithmRuntimeExperimental,
			Algorithm:        "codebook-vq",
			Detail:           "codebook/VQ tensor metadata, payload validation, CPU reference matvec, tiny native Metal matvec, model-pack feature flags, and clear unsupported full-model load diagnostics are available",
			Provides:         []string{"codebook.metadata", "codebook.validation", "codebook.matvec", "model-pack.flag"},
		},
		{
			ID:               inference.CapabilityQuantization,
			Group:            inference.CapabilityGroupRuntime,
			CapabilityStatus: inference.CapabilityStatusExperimental,
			RuntimeStatus:    AlgorithmRuntimeExperimental,
			Algorithm:        "auto-round",
			Detail:           "AutoRound profile metadata, native group RTN/SignRound weight-rounding primitives, packed byte layout, native tensor-map validation/loading, native pack sidecar + safetensors export, model-pack inspection for validated tensor maps, and CPU/Metal dequant/projection helpers are available; loaded projection payloads can feed the fused Metal adapter, while GGUF export orchestration and model generate validation remain pending",
			Architectures:    []string{"gemma4", "qwen3", "qwen3_moe", "llama"},
			Provides: []string{
				"quantization.profile." + string(autoround.ProfileAutoRound),
				"quantization.profile." + string(autoround.ProfileAutoRoundBest),
				"quantization.profile." + string(autoround.ProfileAutoRoundLight),
				"weight_rounding.rtn",
				"weight_rounding.signround",
				"packed_weight.tensor_map",
				"packed_weight.load_safetensors",
				"packed_weight.write_safetensors_projection",
				"packed_weight.write_safetensors_pack",
				"packed_weight.write_native_pack_sidecar",
				"model_pack.inspect_native_tensor_map",
				"packed_weight.dequant",
				"packed_weight.linear_fused",
				"packed_weight.linear_fused_loaded",
				"gguf.export.profile",
			},
			Notes: []string{
				"Native profile surface follows upstream AutoRound recipe names without depending on the Python runtime.",
				"GGUF export and round-trip model generate validation are intentionally separate from the native safetensors pack primitive.",
			},
		},
		{
			ID:               inference.CapabilityEmbeddings,
			Group:            inference.CapabilityGroupModel,
			CapabilityStatus: inference.CapabilityStatusPlanned,
			RuntimeStatus:    AlgorithmRuntimeMetadataOnly,
			Algorithm:        "embeddings",
			Detail:           "embedding model contracts and BERT metadata profiles are available; native encoder kernels are pending",
			Architectures:    []string{"bert"},
			Provides:         []string{"model-pack.profile", "memory.hints"},
		},
		{
			ID:               inference.CapabilityRerank,
			Group:            inference.CapabilityGroupModel,
			CapabilityStatus: inference.CapabilityStatusPlanned,
			RuntimeStatus:    AlgorithmRuntimeMetadataOnly,
			Algorithm:        "rerank",
			Detail:           "rerank contracts and BERT cross-encoder metadata profiles are available; native scorer kernels are pending",
			Architectures:    []string{"bert_rerank"},
			Provides:         []string{"contract", "model-pack.profile", "memory.hints"},
		},
		{
			ID:               inference.CapabilityMoERouting,
			Group:            inference.CapabilityGroupModel,
			CapabilityStatus: inference.CapabilityStatusPlanned,
			RuntimeStatus:    AlgorithmRuntimeMetadataOnly,
			Algorithm:        "moe-routing",
			Detail:           "MoE architecture detection, MiniMax M2 router/expert tensor planning, dense router projection, selected-expert safetensor resolution, fake dispatch, fused packed layer skeleton, router probe events, and memory hints are wired; full native sparse kernels are pending",
			Architectures:    []string{"gemma4", "qwen3_moe", "minimax_m2", "mixtral", "deepseek", "gpt_oss", "kimi"},
			Provides:         []string{"architecture.profile", "tensor.plan", "fake.router.dispatch", "probe.router_decision"},
		},
		{
			ID:               inference.CapabilityMoELazyExperts,
			Group:            inference.CapabilityGroupRuntime,
			CapabilityStatus: inference.CapabilityStatusExperimental,
			RuntimeStatus:    AlgorithmRuntimeExperimental,
			Algorithm:        "moe-lazy-experts",
			Detail:           "MiniMax-style expert residency planning, hot-start loading, cold expert page-in/eviction accounting, probe events, and workload bench summaries are implemented; native fused sparse kernels remain backend-gated",
			Architectures:    []string{"minimax_m2", "mixtral", "deepseek", "gpt_oss", "kimi"},
			Requires:         []inference.CapabilityID{inference.CapabilityMoERouting},
			Provides:         []string{"memory.hints", "expert.residency.plan", "expert.page_in", "expert.eviction", "expert.residency.probe", "bench.report"},
		},
		{
			ID:               inference.CapabilitySpeculativeDecode,
			Group:            inference.CapabilityGroupModel,
			CapabilityStatus: inference.CapabilityStatusExperimental,
			RuntimeStatus:    AlgorithmRuntimeExperimental,
			Algorithm:        "speculative-decode",
			Detail:           "package-first draft/target acceptance metrics and bench reports are available; native batched verification remains opt-in and benchmark-gated",
			Requires:         []inference.CapabilityID{inference.CapabilityScheduler, inference.CapabilityCacheBlocks},
			Provides:         []string{"acceptance.metrics", "bench.report"},
		},
		{
			ID:               inference.CapabilityPromptLookupDecode,
			Group:            inference.CapabilityGroupModel,
			CapabilityStatus: inference.CapabilityStatusExperimental,
			RuntimeStatus:    AlgorithmRuntimeExperimental,
			Algorithm:        "prompt-lookup",
			Detail:           "explicit prompt-token lookup candidates can be measured for repeated-context workloads; native decode shortcut remains opt-in and benchmark-gated",
			Requires:         []inference.CapabilityID{inference.CapabilityCacheBlocks},
			Provides:         []string{"acceptance.metrics", "bench.report"},
		},
		{
			ID:               inference.CapabilityCacheDisk,
			Group:            inference.CapabilityGroupRuntime,
			CapabilityStatus: inference.CapabilityStatusPlanned,
			RuntimeStatus:    AlgorithmRuntimePlanned,
			Algorithm:        "disk-cache",
			Detail:           "disk-backed KV block cache is pending beyond State block manifests",
			Requires:         []inference.CapabilityID{inference.CapabilityCacheBlocks},
		},
	}
}

func algorithmNative(id inference.CapabilityID, group inference.CapabilityGroup, algorithm, detail string) AlgorithmProfile {
	return AlgorithmProfile{
		ID:               id,
		Group:            group,
		CapabilityStatus: inference.CapabilityStatusSupported,
		RuntimeStatus:    AlgorithmRuntimeNative,
		Algorithm:        algorithm,
		Detail:           detail,
	}
}

func AlgorithmCapabilities() []inference.Capability {
	profiles := builtinAlgorithmProfiles()
	out := make([]inference.Capability, 0, len(profiles))
	for _, profile := range profiles {
		out = append(out, profile.Capability())
	}
	return out
}
