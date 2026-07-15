// SPDX-Licence-Identifier: EUPL-1.2

package profile

import "dappco.re/go/inference"

// AlgorithmRuntimeStatus is the ROCm implementation state for a shared runtime
// algorithm.
type AlgorithmRuntimeStatus = inference.FeatureRuntimeStatus

const (
	AlgorithmRuntimeNative       = inference.FeatureRuntimeNative
	AlgorithmRuntimeExperimental = inference.FeatureRuntimeExperimental
	AlgorithmRuntimeMetadataOnly = inference.FeatureRuntimeMetadataOnly
	AlgorithmRuntimePlanned      = inference.FeatureRuntimePlanned
)

// AlgorithmProfile describes one backend-neutral algorithm or runtime feature
// surface in ROCm terms.
type AlgorithmProfile = inference.AlgorithmProfile

const AlgorithmProfileRegistryContract = "rocm-algorithm-profile-registry-v1"

var builtinAlgorithmProfilesData = []AlgorithmProfile{}
var builtinAlgorithmProfileIndex = map[inference.CapabilityID]int{}

func init() {
	builtinAlgorithmProfilesData = buildBuiltinAlgorithmProfiles()
	builtinAlgorithmProfileIndex = make(map[inference.CapabilityID]int, len(builtinAlgorithmProfilesData))
	for index, profile := range builtinAlgorithmProfilesData {
		builtinAlgorithmProfileIndex[profile.ID] = index
	}
}

// BuiltinAlgorithmProfiles returns the built-in algorithm matrix exposed by
// discovery, daemon registry, and API consumers.
func BuiltinAlgorithmProfiles() []AlgorithmProfile {
	out := make([]AlgorithmProfile, len(builtinAlgorithmProfilesData))
	for index, profile := range builtinAlgorithmProfilesData {
		out[index] = inference.CloneAlgorithmProfile(profile)
	}
	return out
}

// LookupAlgorithmProfile returns the registered profile for id.
func LookupAlgorithmProfile(id inference.CapabilityID) (AlgorithmProfile, bool) {
	index, ok := builtinAlgorithmProfileIndex[id]
	if !ok {
		return AlgorithmProfile{}, false
	}
	return inference.CloneAlgorithmProfile(builtinAlgorithmProfilesData[index]), true
}

// AlgorithmCapabilities returns the algorithm matrix as capability rows.
func AlgorithmCapabilities() []inference.Capability {
	profiles := BuiltinAlgorithmProfiles()
	out := make([]inference.Capability, 0, len(profiles))
	for _, profile := range profiles {
		out = append(out, profile.Capability())
	}
	return out
}

func buildBuiltinAlgorithmProfiles() []AlgorithmProfile {
	return []AlgorithmProfile{
		algorithmNative(inference.CapabilityScheduler, inference.CapabilityGroupRuntime, "scheduler", "bounded request queueing, stream backpressure, cancellation IDs, and latency metrics are implemented"),
		algorithmNative(inference.CapabilityRequestCancel, inference.CapabilityGroupRuntime, "request-cancel", "generation and scheduled requests can be cancelled through context and cancellation IDs"),
		algorithmNative(inference.CapabilityCacheBlocks, inference.CapabilityGroupRuntime, "block-prefix-cache", "block-prefix cache identity, state-backed KV block refs, and warm routes are implemented"),
		algorithmNative(inference.CapabilityCacheWarm, inference.CapabilityGroupRuntime, "cache-warm", "prompt and KV block warm paths are exposed through the cache registry"),
		algorithmNative(inference.CapabilityReasoningParse, inference.CapabilityGroupModel, "reasoning-parser", "model-aware thinking and reasoning parsers are available"),
		algorithmNative(inference.CapabilityToolParse, inference.CapabilityGroupModel, "tool-parser", "XML and OpenAI-style JSON tool-call parsing is available"),
		{
			ID:               inference.CapabilityJANGTQ,
			Group:            inference.CapabilityGroupRuntime,
			CapabilityStatus: inference.CapabilityStatusExperimental,
			RuntimeStatus:    AlgorithmRuntimeMetadataOnly,
			Algorithm:        "jangtq",
			Detail:           "JANG/JANGTQ metadata, packed tensor descriptors, CPU reference dequant, HIP launch scaffolding, and model-pack validation are wired; full model execution is pending",
			Architectures:    []string{"minimax_m2"},
			Provides:         []string{"quantization.profile", "packed_tensor.descriptor", "reference.dequant", "memory.hints"},
		},
		{
			ID:               inference.CapabilityCodebookVQ,
			Group:            inference.CapabilityGroupRuntime,
			CapabilityStatus: inference.CapabilityStatusExperimental,
			RuntimeStatus:    AlgorithmRuntimeExperimental,
			Algorithm:        "codebook-vq",
			Detail:           "codebook/VQ tensor metadata, payload validation, CPU reference matvec, HIP launch scaffolding, model-pack flags, and clear unsupported full-model load diagnostics are available",
			Provides:         []string{"codebook.metadata", "codebook.validation", "codebook.matvec", "model-pack.flag"},
		},
		{
			ID:               inference.CapabilityQuantization,
			Group:            inference.CapabilityGroupRuntime,
			CapabilityStatus: inference.CapabilityStatusExperimental,
			RuntimeStatus:    AlgorithmRuntimeExperimental,
			Algorithm:        "auto-round",
			Detail:           "AutoRound profile metadata, native group RTN/SignRound update passes, packed byte layout, model-pack inspection, and HIP quant launch surfaces are available; GGUF export and promoted generate validation remain separate",
			Architectures:    []string{"gemma4", "qwen3", "qwen3_moe", "llama"},
			Provides: []string{
				"quantization.profile.auto-round",
				"quantization.profile.auto-round-best",
				"quantization.profile.auto-round-light",
				"weight_rounding.rtn",
				"weight_rounding.signround",
				"packed_weight.tensor_map",
				"packed_weight.dequant",
				"packed_weight.linear_fused",
				"model_pack.inspect_autoround",
				"autoround.calibration.plan",
				"autoround.calibration.evidence",
				"autoround.calibration.decision",
				"hip.autoround_quantize.launch_args",
				"hip.autoround_quantize.kernel",
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
			Detail:           "MoE architecture detection, router/expert tensor planning, dense router projection, selected-expert safetensor resolution, probe events, and memory hints are wired; full native sparse kernels are pending",
			Architectures:    []string{"gemma4", "qwen3_moe", "minimax_m2", "mixtral", "deepseek", "gpt-oss", "kimi"},
			Provides:         []string{"architecture.profile", "tensor.plan", "probe.router_decision", "memory.hints"},
		},
		{
			ID:               inference.CapabilityMoELazyExperts,
			Group:            inference.CapabilityGroupRuntime,
			CapabilityStatus: inference.CapabilityStatusExperimental,
			RuntimeStatus:    AlgorithmRuntimeExperimental,
			Algorithm:        "moe-lazy-experts",
			Detail:           "expert residency planning, hot-start loading, cold expert page-in and eviction accounting, probe events, and workload bench summaries are implemented; native fused sparse kernels remain backend-gated",
			Architectures:    []string{"minimax_m2", "mixtral", "deepseek", "gpt-oss", "kimi"},
			Requires:         []inference.CapabilityID{inference.CapabilityMoERouting},
			Provides:         []string{"memory.hints", "expert.residency.plan", "expert.page_in", "expert.eviction", "expert.residency.probe", "bench.report"},
		},
		{
			ID:               inference.CapabilitySpeculativeDecode,
			Group:            inference.CapabilityGroupModel,
			CapabilityStatus: inference.CapabilityStatusExperimental,
			RuntimeStatus:    AlgorithmRuntimeExperimental,
			Algorithm:        "speculative-decode",
			Detail:           "package-first draft/target acceptance metrics, reactive Gemma-4 MTP planning, and benchmark reports are available; native batched verification remains pending",
			Requires:         []inference.CapabilityID{inference.CapabilityScheduler, inference.CapabilityCacheBlocks},
			Provides:         []string{"acceptance.metrics", "bench.report", "mtp.attached_drafter.plan"},
		},
		{
			ID:               inference.CapabilityPromptLookupDecode,
			Group:            inference.CapabilityGroupModel,
			CapabilityStatus: inference.CapabilityStatusExperimental,
			RuntimeStatus:    AlgorithmRuntimeExperimental,
			Algorithm:        "prompt-lookup",
			Detail:           "explicit prompt-token lookup candidates can be measured for repeated-context workloads; native decode shortcut remains benchmark-gated",
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
