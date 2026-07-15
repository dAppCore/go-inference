// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"strconv"
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/profile"
)

const RuntimeAuthorPlanContract = "rocm-runtime-author-plan-v1"

// RuntimeAuthorCapabilityID names an exported runtime-author operation. The
// IDs mirror go-mlx's runtime_author.go accessors while remaining ROCm-owned
// and backend-neutral.
type RuntimeAuthorCapabilityID string

const (
	RuntimeAuthorUnderlyingModel               RuntimeAuthorCapabilityID = "underlying_model"
	RuntimeAuthorRuntimeTokenizer              RuntimeAuthorCapabilityID = "runtime_tokenizer"
	RuntimeAuthorRequireTextRuntime            RuntimeAuthorCapabilityID = "require_text_runtime"
	RuntimeAuthorAcquireSlot                   RuntimeAuthorCapabilityID = "acquire_slot"
	RuntimeAuthorAcquirePromptCache            RuntimeAuthorCapabilityID = "acquire_prompt_cache"
	RuntimeAuthorWithDevice                    RuntimeAuthorCapabilityID = "with_device"
	RuntimeAuthorNewCachesWithRequestFixedSize RuntimeAuthorCapabilityID = "new_caches_with_request_fixed_size"
	RuntimeAuthorGenerationFixedCacheSize      RuntimeAuthorCapabilityID = "generation_fixed_sliding_cache_size"
	RuntimeAuthorRuntimeCachesSnapshotSafe     RuntimeAuthorCapabilityID = "runtime_caches_snapshot_safe"
	RuntimeAuthorPromptCacheEnabled            RuntimeAuthorCapabilityID = "prompt_cache_enabled"
	RuntimeAuthorPrefillChunkSize              RuntimeAuthorCapabilityID = "prefill_chunk_size"
	RuntimeAuthorPromptCacheMinimum            RuntimeAuthorCapabilityID = "prompt_cache_minimum"
	RuntimeAuthorSetLastErr                    RuntimeAuthorCapabilityID = "set_last_err"
	RuntimeAuthorSetLastMetrics                RuntimeAuthorCapabilityID = "set_last_metrics"
	RuntimeAuthorAdapterCacheKey               RuntimeAuthorCapabilityID = "adapter_cache_key"
	RuntimeAuthorPromptCacheMatchWithHidden    RuntimeAuthorCapabilityID = "prompt_cache_match_with_hidden"
	RuntimeAuthorStorePromptCacheEntry         RuntimeAuthorCapabilityID = "store_prompt_cache_entry"
	RuntimeAuthorPromptCacheEntryLogits        RuntimeAuthorCapabilityID = "prompt_cache_entry_logits"
	RuntimeAuthorPromptCacheEntryHidden        RuntimeAuthorCapabilityID = "prompt_cache_entry_hidden"
	RuntimeAuthorRestoreCaches                 RuntimeAuthorCapabilityID = "restore_caches"
	RuntimeAuthorCacheProfile                  RuntimeAuthorCapabilityID = "cache_profile"
	RuntimeAuthorModelProfile                  RuntimeAuthorCapabilityID = "model_profile"
	RuntimeAuthorModelRoutePlan                RuntimeAuthorCapabilityID = "model_route_plan"
	RuntimeAuthorAttachedDrafterRuntime        RuntimeAuthorCapabilityID = "attached_drafter_runtime"
)

// RuntimeAuthorPlan is the model-owned ROCm answer to go-mlx's
// runtime_author.go surface: it describes which private-runtime hooks a
// concrete loaded model can safely expose to a runtime author.
type RuntimeAuthorPlan struct {
	Contract               string                         `json:"contract,omitempty"`
	Architecture           string                         `json:"architecture,omitempty"`
	Family                 string                         `json:"family,omitempty"`
	Runtime                string                         `json:"runtime,omitempty"`
	RuntimeStatus          inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	NativeRuntime          bool                           `json:"native_runtime,omitempty"`
	TextRuntime            bool                           `json:"text_runtime,omitempty"`
	ModelAccess            bool                           `json:"model_access,omitempty"`
	TokenCodec             bool                           `json:"token_codec,omitempty"`
	RuntimeGuard           bool                           `json:"runtime_guard,omitempty"`
	ParallelSlotGate       bool                           `json:"parallel_slot_gate,omitempty"`
	PromptCacheLock        bool                           `json:"prompt_cache_lock,omitempty"`
	DeviceGuard            bool                           `json:"device_guard,omitempty"`
	RequestFixedCache      bool                           `json:"request_fixed_cache,omitempty"`
	FixedSlidingCacheSize  bool                           `json:"fixed_sliding_cache_size,omitempty"`
	CacheSnapshotSafe      bool                           `json:"cache_snapshot_safe,omitempty"`
	PromptCache            bool                           `json:"prompt_cache,omitempty"`
	PrefillChunking        bool                           `json:"prefill_chunking,omitempty"`
	PromptCacheMinimum     bool                           `json:"prompt_cache_minimum,omitempty"`
	LastErrorSink          bool                           `json:"last_error_sink,omitempty"`
	LastMetricsSink        bool                           `json:"last_metrics_sink,omitempty"`
	AdapterCacheKey        bool                           `json:"adapter_cache_key,omitempty"`
	HiddenPromptCache      bool                           `json:"hidden_prompt_cache,omitempty"`
	PromptCacheStore       bool                           `json:"prompt_cache_store,omitempty"`
	PromptCacheEntryLogits bool                           `json:"prompt_cache_entry_logits,omitempty"`
	PromptCacheEntryHidden bool                           `json:"prompt_cache_entry_hidden,omitempty"`
	CacheRestore           bool                           `json:"cache_restore,omitempty"`
	CacheProfile           bool                           `json:"cache_profile,omitempty"`
	ModelProfile           bool                           `json:"model_profile,omitempty"`
	ModelRoutePlan         bool                           `json:"model_route_plan,omitempty"`
	AttachedDrafterRuntime bool                           `json:"attached_drafter_runtime,omitempty"`
	CapabilityIDs          []RuntimeAuthorCapabilityID    `json:"capability_ids,omitempty"`
	Labels                 map[string]string              `json:"labels,omitempty"`
}

func (plan RuntimeAuthorPlan) Matched() bool {
	return plan.Contract != "" && plan.Architecture != "" && len(plan.CapabilityIDs) > 0
}

func (plan RuntimeAuthorPlan) Clone() RuntimeAuthorPlan {
	plan.CapabilityIDs = append([]RuntimeAuthorCapabilityID(nil), plan.CapabilityIDs...)
	plan.Labels = cloneStringMap(plan.Labels)
	return plan
}

func (plan RuntimeAuthorPlan) HasCapability(id RuntimeAuthorCapabilityID) bool {
	return slices.Contains(plan.CapabilityIDs, id)
}

func RuntimeAuthorPlanForIdentity(path string, identity inference.ModelIdentity) (RuntimeAuthorPlan, bool) {
	if identity.Path == "" {
		identity.Path = path
	}
	featureRoute, _ := FeatureRouteForIdentity(path, identity)
	cacheRoute, _ := CacheRouteForIdentity(path, identity)
	stateRoute, _ := StateContextRouteForIdentity(path, identity)
	drafterRoute, _ := AttachedDrafterRouteForIdentity(path, identity)
	runtimeRoute, _ := RuntimeContractRouteForIdentity(path, identity)
	gatePlan := RuntimeGatePlanForRoutes(firstNonEmpty(
		featureRoute.Architecture,
		cacheRoute.Architecture,
		stateRoute.Architecture,
		drafterRoute.Architecture,
		runtimeRoute.Architecture,
		identity.Labels["engine_architecture_resolved"],
		identity.Labels["architecture_resolved"],
		identity.Architecture,
	), firstNonEmpty(
		featureRoute.Family,
		cacheRoute.Family,
		stateRoute.Family,
		drafterRoute.Family,
		runtimeRoute.Family,
	), featureRoute, runtimeRoute, identity.Labels)
	plan := RuntimeAuthorPlanForRoutes(featureRoute.Architecture, featureRoute.Family, featureRoute, cacheRoute, stateRoute, drafterRoute, runtimeRoute, gatePlan, identity.Labels)
	if !plan.Matched() {
		return RuntimeAuthorPlan{}, false
	}
	return plan, true
}

func RuntimeAuthorPlanForRouteSet(set RouteSet) RuntimeAuthorPlan {
	return RuntimeAuthorPlanForRoutes(set.Architecture, set.Family, set.FeatureRoute, set.CacheRoute, set.StateContextRoute, set.AttachedDrafterRoute, set.RuntimeContractRoute, set.RuntimeGatePlan, set.Model.Labels)
}

func RuntimeAuthorPlanForRoutes(architecture, family string, featureRoute FeatureRoute, cacheRoute CacheRoute, stateRoute StateContextRoute, drafterRoute AttachedDrafterRoute, runtimeRoute RuntimeContractRoute, gatePlan RuntimeGatePlan, labels map[string]string) RuntimeAuthorPlan {
	if !runtimeAuthorHasRoute(featureRoute, cacheRoute, stateRoute, drafterRoute, runtimeRoute, gatePlan) {
		return RuntimeAuthorPlan{}
	}
	architecture = profile.ArchitectureID(firstNonEmpty(
		architecture,
		featureRoute.Architecture,
		cacheRoute.Architecture,
		stateRoute.Architecture,
		drafterRoute.Architecture,
		runtimeRoute.Architecture,
		gatePlan.Architecture,
		labels["engine_architecture_resolved"],
		labels["architecture_resolved"],
	))
	if architecture == "" {
		return RuntimeAuthorPlan{}
	}
	family = firstNonEmpty(family, featureRoute.Family, cacheRoute.Family, stateRoute.Family, drafterRoute.Family, runtimeRoute.Family, gatePlan.Family, architecture)
	runtimeStatus := runtimeAuthorRuntimeStatus(featureRoute, cacheRoute, stateRoute, drafterRoute, runtimeRoute, gatePlan)
	nativeRuntime := featureRoute.NativeRuntime ||
		cacheRoute.NativeRuntime ||
		stateRoute.NativeRuntime ||
		drafterRoute.NativeRuntime ||
		runtimeRoute.NativeRuntime ||
		runtimeStatus == inference.FeatureRuntimeNative ||
		runtimeStatus == inference.FeatureRuntimeExperimental
	textRuntime := featureRoute.TextGenerate || runtimeRoute.TextGenerate
	promptCache := cacheRoute.Matched() || stateRoute.PackageLocalKV || stateRoute.BlockBundleRefs || stateRoute.PortableRefs
	hiddenPromptCache := runtimeRoute.LastTokenLogits || drafterRoute.BorrowTargetKV || drafterRoute.NativeStateGeneration || drafterRoute.RetainedStateRequired || stateRoute.AttachedDrafterState

	builder := runtimeAuthorPlanBuilder{
		plan: RuntimeAuthorPlan{
			Contract:      RuntimeAuthorPlanContract,
			Architecture:  architecture,
			Family:        family,
			Runtime:       "rocm",
			RuntimeStatus: runtimeStatus,
			NativeRuntime: nativeRuntime,
			TextRuntime:   textRuntime,
		},
		seen: map[RuntimeAuthorCapabilityID]bool{},
	}
	builder.set(RuntimeAuthorUnderlyingModel, true)
	builder.set(RuntimeAuthorModelProfile, featureRoute.Matched() || runtimeRoute.Matched())
	builder.set(RuntimeAuthorModelRoutePlan, featureRoute.Matched() || cacheRoute.Matched() || runtimeRoute.Matched() || gatePlan.Matched())
	builder.set(RuntimeAuthorRuntimeTokenizer, featureRoute.Matched() || runtimeRoute.Matched())
	builder.set(RuntimeAuthorRequireTextRuntime, textRuntime || runtimeRoute.DecodeUnavailableReporter)
	builder.set(RuntimeAuthorAcquireSlot, textRuntime || gatePlan.GateEnabled(GateGenerationStream))
	builder.set(RuntimeAuthorAcquirePromptCache, promptCache)
	builder.set(RuntimeAuthorWithDevice, nativeRuntime)
	builder.set(RuntimeAuthorNewCachesWithRequestFixedSize, cacheRoute.SupportsKV || runtimeRoute.FixedSlidingCache || stateRoute.RuntimeOwnedKV)
	builder.set(RuntimeAuthorGenerationFixedCacheSize, runtimeRoute.FixedSlidingPrefillLimit || runtimeRoute.FixedSlidingCache)
	builder.set(RuntimeAuthorRuntimeCachesSnapshotSafe, stateRoute.SleepState || stateRoute.WakeState || stateRoute.PackageLocalKV || stateRoute.BlockBundleRefs || cacheRoute.SupportsKV)
	builder.set(RuntimeAuthorPromptCacheEnabled, promptCache)
	builder.set(RuntimeAuthorPrefillChunkSize, textRuntime && (cacheRoute.SupportsKV || stateRoute.StateSession))
	builder.set(RuntimeAuthorPromptCacheMinimum, promptCache)
	builder.set(RuntimeAuthorSetLastErr, true)
	builder.set(RuntimeAuthorSetLastMetrics, true)
	builder.set(RuntimeAuthorAdapterCacheKey, promptCache || drafterRoute.Matched())
	builder.set(RuntimeAuthorPromptCacheMatchWithHidden, hiddenPromptCache)
	builder.set(RuntimeAuthorStorePromptCacheEntry, promptCache)
	builder.set(RuntimeAuthorPromptCacheEntryLogits, runtimeRoute.LastTokenLogits)
	builder.set(RuntimeAuthorPromptCacheEntryHidden, hiddenPromptCache)
	builder.set(RuntimeAuthorRestoreCaches, cacheRoute.SupportsKV || stateRoute.RestoreState || stateRoute.WakeState || stateRoute.RuntimeOwnedKV)
	builder.set(RuntimeAuthorCacheProfile, cacheRoute.Matched() || runtimeRoute.CacheTopology || stateRoute.StateSession)
	builder.set(RuntimeAuthorAttachedDrafterRuntime, drafterRoute.Matched())
	plan := builder.plan
	plan.Labels = runtimeAuthorPlanLabels(plan)
	if !plan.Matched() {
		return RuntimeAuthorPlan{}
	}
	return plan.Clone()
}

func runtimeAuthorHasRoute(featureRoute FeatureRoute, cacheRoute CacheRoute, stateRoute StateContextRoute, drafterRoute AttachedDrafterRoute, runtimeRoute RuntimeContractRoute, gatePlan RuntimeGatePlan) bool {
	return featureRoute.Matched() ||
		cacheRoute.Matched() ||
		stateRoute.Matched() ||
		drafterRoute.Matched() ||
		runtimeRoute.Matched() ||
		gatePlan.Matched()
}

type runtimeAuthorPlanBuilder struct {
	plan RuntimeAuthorPlan
	seen map[RuntimeAuthorCapabilityID]bool
}

func (builder *runtimeAuthorPlanBuilder) set(id RuntimeAuthorCapabilityID, enabled bool) {
	if id == "" || !enabled || builder.seen[id] {
		return
	}
	builder.seen[id] = true
	builder.plan.CapabilityIDs = append(builder.plan.CapabilityIDs, id)
	switch id {
	case RuntimeAuthorUnderlyingModel:
		builder.plan.ModelAccess = true
	case RuntimeAuthorRuntimeTokenizer:
		builder.plan.TokenCodec = true
	case RuntimeAuthorRequireTextRuntime:
		builder.plan.RuntimeGuard = true
	case RuntimeAuthorAcquireSlot:
		builder.plan.ParallelSlotGate = true
	case RuntimeAuthorAcquirePromptCache:
		builder.plan.PromptCacheLock = true
	case RuntimeAuthorWithDevice:
		builder.plan.DeviceGuard = true
	case RuntimeAuthorNewCachesWithRequestFixedSize:
		builder.plan.RequestFixedCache = true
	case RuntimeAuthorGenerationFixedCacheSize:
		builder.plan.FixedSlidingCacheSize = true
	case RuntimeAuthorRuntimeCachesSnapshotSafe:
		builder.plan.CacheSnapshotSafe = true
	case RuntimeAuthorPromptCacheEnabled:
		builder.plan.PromptCache = true
	case RuntimeAuthorPrefillChunkSize:
		builder.plan.PrefillChunking = true
	case RuntimeAuthorPromptCacheMinimum:
		builder.plan.PromptCacheMinimum = true
	case RuntimeAuthorSetLastErr:
		builder.plan.LastErrorSink = true
	case RuntimeAuthorSetLastMetrics:
		builder.plan.LastMetricsSink = true
	case RuntimeAuthorAdapterCacheKey:
		builder.plan.AdapterCacheKey = true
	case RuntimeAuthorPromptCacheMatchWithHidden:
		builder.plan.HiddenPromptCache = true
	case RuntimeAuthorStorePromptCacheEntry:
		builder.plan.PromptCacheStore = true
	case RuntimeAuthorPromptCacheEntryLogits:
		builder.plan.PromptCacheEntryLogits = true
	case RuntimeAuthorPromptCacheEntryHidden:
		builder.plan.PromptCacheEntryHidden = true
	case RuntimeAuthorRestoreCaches:
		builder.plan.CacheRestore = true
	case RuntimeAuthorCacheProfile:
		builder.plan.CacheProfile = true
	case RuntimeAuthorModelProfile:
		builder.plan.ModelProfile = true
	case RuntimeAuthorModelRoutePlan:
		builder.plan.ModelRoutePlan = true
	case RuntimeAuthorAttachedDrafterRuntime:
		builder.plan.AttachedDrafterRuntime = true
	}
}

func runtimeAuthorRuntimeStatus(featureRoute FeatureRoute, cacheRoute CacheRoute, stateRoute StateContextRoute, drafterRoute AttachedDrafterRoute, runtimeRoute RuntimeContractRoute, gatePlan RuntimeGatePlan) inference.FeatureRuntimeStatus {
	for _, status := range []inference.FeatureRuntimeStatus{
		featureRoute.RuntimeStatus,
		cacheRoute.RuntimeStatus,
		stateRoute.RuntimeStatus,
		drafterRoute.RuntimeStatus,
		runtimeRoute.RuntimeStatus,
		gatePlan.RuntimeStatus,
	} {
		if status != "" {
			return status
		}
	}
	return ""
}

func runtimeAuthorPlanLabels(plan RuntimeAuthorPlan) map[string]string {
	if plan.Contract == "" || plan.Architecture == "" {
		return nil
	}
	labels := map[string]string{
		"engine_runtime_author_plan_contract":             plan.Contract,
		"engine_runtime_author_architecture":              plan.Architecture,
		"engine_runtime_author_runtime":                   plan.Runtime,
		"engine_runtime_author_capability_count":          strconv.Itoa(len(plan.CapabilityIDs)),
		"engine_runtime_author_capability_ids":            runtimeAuthorCapabilityIDsCSV(plan.CapabilityIDs),
		"engine_runtime_author_native_runtime":            strconv.FormatBool(plan.NativeRuntime),
		"engine_runtime_author_text_runtime":              strconv.FormatBool(plan.TextRuntime),
		"engine_runtime_author_model_access":              strconv.FormatBool(plan.ModelAccess),
		"engine_runtime_author_token_codec":               strconv.FormatBool(plan.TokenCodec),
		"engine_runtime_author_runtime_guard":             strconv.FormatBool(plan.RuntimeGuard),
		"engine_runtime_author_parallel_slot_gate":        strconv.FormatBool(plan.ParallelSlotGate),
		"engine_runtime_author_prompt_cache_lock":         strconv.FormatBool(plan.PromptCacheLock),
		"engine_runtime_author_device_guard":              strconv.FormatBool(plan.DeviceGuard),
		"engine_runtime_author_request_fixed_cache":       strconv.FormatBool(plan.RequestFixedCache),
		"engine_runtime_author_fixed_sliding_cache_size":  strconv.FormatBool(plan.FixedSlidingCacheSize),
		"engine_runtime_author_cache_snapshot_safe":       strconv.FormatBool(plan.CacheSnapshotSafe),
		"engine_runtime_author_prompt_cache":              strconv.FormatBool(plan.PromptCache),
		"engine_runtime_author_prefill_chunking":          strconv.FormatBool(plan.PrefillChunking),
		"engine_runtime_author_prompt_cache_minimum":      strconv.FormatBool(plan.PromptCacheMinimum),
		"engine_runtime_author_last_error_sink":           strconv.FormatBool(plan.LastErrorSink),
		"engine_runtime_author_last_metrics_sink":         strconv.FormatBool(plan.LastMetricsSink),
		"engine_runtime_author_adapter_cache_key":         strconv.FormatBool(plan.AdapterCacheKey),
		"engine_runtime_author_hidden_prompt_cache":       strconv.FormatBool(plan.HiddenPromptCache),
		"engine_runtime_author_prompt_cache_store":        strconv.FormatBool(plan.PromptCacheStore),
		"engine_runtime_author_prompt_cache_entry_logits": strconv.FormatBool(plan.PromptCacheEntryLogits),
		"engine_runtime_author_prompt_cache_entry_hidden": strconv.FormatBool(plan.PromptCacheEntryHidden),
		"engine_runtime_author_cache_restore":             strconv.FormatBool(plan.CacheRestore),
		"engine_runtime_author_cache_profile":             strconv.FormatBool(plan.CacheProfile),
		"engine_runtime_author_model_profile":             strconv.FormatBool(plan.ModelProfile),
		"engine_runtime_author_model_route_plan":          strconv.FormatBool(plan.ModelRoutePlan),
		"engine_runtime_author_attached_drafter_runtime":  strconv.FormatBool(plan.AttachedDrafterRuntime),
	}
	if plan.Family != "" {
		labels["engine_runtime_author_family"] = plan.Family
	}
	if plan.RuntimeStatus != "" {
		labels["engine_runtime_author_runtime_status"] = string(plan.RuntimeStatus)
	}
	for _, id := range plan.CapabilityIDs {
		if id != "" {
			labels["engine_runtime_author_"+string(id)] = "true"
		}
	}
	return labels
}

func runtimeAuthorCapabilityIDsCSV(ids []RuntimeAuthorCapabilityID) string {
	parts := make([]string, 0, len(ids))
	for _, id := range ids {
		if id != "" {
			parts = append(parts, string(id))
		}
	}
	return strings.Join(parts, ",")
}
