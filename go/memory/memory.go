// SPDX-Licence-Identifier: EUPL-1.2

// Package memory is the go-mlx local-inference memory planner. It maps
// measured Apple-silicon hardware + optional model metadata to a
// runtime policy (context length, KV cache shape, batch size, prompt
// cache, MoE expert residency) that fits the device class without
// over-allocating.
//
//	plan := memory.NewPlan(memory.Input{Device: dev, Pack: pack, ModelInfo: info})
//	if plan.ContextLength > 0 { … }
package memory

import (
	"time"

	mp "dappco.re/go/inference/modelpack"
	"dappco.re/go/inference/profile"
	"dappco.re/go/inference/quant/jang"
)

// GiB is the number of bytes in a gibibyte.
const GiB uint64 = 1 << 30

// Class names the local Apple memory tier driving runtime policy.
type Class string

const (
	ClassUnknown    Class = "unknown"
	ClassApple16GB  Class = "apple-silicon-16gb"
	ClassApple24GB  Class = "apple-silicon-24gb"
	ClassApple32GB  Class = "apple-silicon-32gb"
	ClassApple64GB  Class = "apple-silicon-64gb"
	ClassApple96GB  Class = "apple-silicon-96gb"
	ClassApple128GB Class = "apple-silicon-128gb-plus"
)

// KVCachePolicy names the cache shape selected by the planner.
type KVCachePolicy string

const (
	KVCacheDefault  KVCachePolicy = ""
	KVCacheRotating KVCachePolicy = "rotating"
	KVCacheFull     KVCachePolicy = "full"
)

// KVCacheMode names the physical KV storage strategy used by the native cache.
type KVCacheMode string

const (
	KVCacheModeDefault    KVCacheMode = ""
	KVCacheModeFP16       KVCacheMode = "fp16"
	KVCacheModeQ8         KVCacheMode = "q8"
	KVCacheModeKQ8VQ4     KVCacheMode = "k-q8-v-q4"
	KVCacheModePaged      KVCacheMode = "paged"
	KVCacheModeTurboQuant KVCacheMode = "turboquant"
)

// IsKnownKVCacheMode reports whether mode is part of the public KV-cache
// mode contract. TurboQuant is a research mode; backends may still fail
// closed until their native cache implementation exists.
func IsKnownKVCacheMode(mode KVCacheMode) bool {
	switch mode {
	case KVCacheModeDefault, KVCacheModeFP16, KVCacheModeQ8, KVCacheModeKQ8VQ4, KVCacheModePaged, KVCacheModeTurboQuant:
		return true
	default:
		return false
	}
}

// ExpertResidencyMode names how routed MoE experts are kept resident.
type ExpertResidencyMode string

const (
	ExpertResidencyModeOff    ExpertResidencyMode = ""
	ExpertResidencyModePinned ExpertResidencyMode = "pinned"
	ExpertResidencyModeLazy   ExpertResidencyMode = "lazy"
)

// ExpertEvictionPolicy names the cold-expert eviction strategy.
type ExpertEvictionPolicy string

const (
	ExpertEvictionLRU ExpertEvictionPolicy = "lru"
)

// DeviceInfo carries the measured device memory the planner consults.
// Mirrors the mlx-root metal.DeviceInfo struct so the memory package
// stays driver-internal-free.
type DeviceInfo struct {
	Architecture                 string
	MaxBufferLength              uint64
	MaxRecommendedWorkingSetSize uint64
	MemorySize                   uint64
}

// ModelInfo carries the optional model metadata the planner consults.
// Mirrors the mlx-root ModelInfo identity used at the package boundary.
type ModelInfo struct {
	Architecture  string
	VocabSize     int
	NumLayers     int
	HiddenSize    int
	NumKVHeads    int
	HeadDim       int
	QuantBits     int
	QuantGroup    int
	ContextLength int
}

// Input supplies measured hardware and optional model metadata.
type Input struct {
	Device    DeviceInfo
	Pack      *mp.ModelPack
	ModelInfo *ModelInfo
}

// ExpertResidencyStats records measured hot-load, page-in, and eviction
// behaviour. Backends can feed this directly into workload bench reports.
type ExpertResidencyStats struct {
	ResidentExperts     int           `json:"resident_experts,omitempty"`
	PeakResidentExperts int           `json:"peak_resident_experts,omitempty"`
	HotLoads            int           `json:"hot_loads,omitempty"`
	ColdLoads           int           `json:"cold_loads,omitempty"`
	PageIns             int           `json:"page_ins,omitempty"`
	PageOuts            int           `json:"page_outs,omitempty"`
	Hits                int           `json:"hits,omitempty"`
	LoadedBytes         uint64        `json:"loaded_bytes,omitempty"`
	EvictedBytes        uint64        `json:"evicted_bytes,omitempty"`
	FirstUseLatency     time.Duration `json:"first_use_latency,omitempty"`
	TotalLoadDuration   time.Duration `json:"total_load_duration,omitempty"`
}

// ExpertResidencyPlan is a backend-neutral MoE residency policy. It is
// small enough for memory planners and benchmark reports while still
// explicit about hot experts, resident limits, and expected first-use
// pressure.
type ExpertResidencyPlan struct {
	Enabled                 bool                 `json:"enabled"`
	Mode                    ExpertResidencyMode  `json:"mode,omitempty"`
	Architecture            string               `json:"architecture,omitempty"`
	TotalExperts            int                  `json:"total_experts,omitempty"`
	ExpertsPerToken         int                  `json:"experts_per_token,omitempty"`
	HotExpertIDs            []int                `json:"hot_expert_ids,omitempty"`
	StartupExpertIDs        []int                `json:"startup_expert_ids,omitempty"`
	HotExperts              int                  `json:"hot_experts,omitempty"`
	MaxResidentExperts      int                  `json:"max_resident_experts,omitempty"`
	PageInBatchSize         int                  `json:"page_in_batch_size,omitempty"`
	EvictionPolicy          ExpertEvictionPolicy `json:"eviction_policy,omitempty"`
	EstimatedExpertBytes    uint64               `json:"estimated_expert_bytes,omitempty"`
	EstimatedResidentBytes  uint64               `json:"estimated_resident_bytes,omitempty"`
	MaxResidentBytes        uint64               `json:"max_resident_bytes,omitempty"`
	FirstUseLatencyExpected bool                 `json:"first_use_latency_expected,omitempty"`
	Notes                   []string             `json:"notes,omitempty"`
}

// Plan is the local runtime policy derived from measured device memory.
type Plan struct {
	MachineClass                  Class               `json:"machine_class"`
	Architecture                  string              `json:"architecture,omitempty"`
	DeviceMemoryBytes             uint64              `json:"device_memory_bytes,omitempty"`
	RecommendedWorkingSetBytes    uint64              `json:"recommended_working_set_bytes,omitempty"`
	ContextLength                 int                 `json:"context_length"`
	CachePolicy                   KVCachePolicy       `json:"cache_policy"`
	CacheMode                     KVCacheMode         `json:"cache_mode,omitempty"`
	BatchSize                     int                 `json:"batch_size"`
	PrefillChunkSize              int                 `json:"prefill_chunk_size"`
	ParallelSlots                 int                 `json:"parallel_slots"`
	PromptCache                   bool                `json:"prompt_cache"`
	PromptCacheMinTokens          int                 `json:"prompt_cache_min_tokens"`
	ModelQuantization             int                 `json:"model_quantization,omitempty"`
	ModelQuantizationType         string              `json:"model_quantization_type,omitempty"`
	ModelQuantizationFamily       string              `json:"model_quantization_family,omitempty"`
	ModelPackedQuantization       *jang.PackedProfile `json:"model_packed_quantization,omitempty"`
	ModelWeightBytes              uint64              `json:"model_weight_bytes,omitempty"`
	ModelForwardSkeletonValidated bool                `json:"model_forward_skeleton_validated,omitempty"`
	ModelForwardSkeletonBytes     uint64              `json:"model_forward_skeleton_bytes,omitempty"`
	ExpertResidency               ExpertResidencyPlan `json:"expert_residency"`
	MemoryLimitBytes              uint64              `json:"memory_limit_bytes,omitempty"`
	CacheLimitBytes               uint64              `json:"cache_limit_bytes,omitempty"`
	WiredLimitBytes               uint64              `json:"wired_limit_bytes,omitempty"`
	EstimatedKVCacheBytes         uint64              `json:"estimated_kv_cache_bytes,omitempty"`
	EstimatedKVCacheModeBytes     uint64              `json:"estimated_kv_cache_mode_bytes,omitempty"`
	KVCacheSavingsRatio           float64             `json:"kv_cache_savings_ratio,omitempty"`
	Notes                         []string            `json:"notes,omitempty"`
}

// Defaults that mirror the mlx-root local-inference baselines. Kept
// here so the memory package is self-contained.
const (
	defaultLocalContextLength   = 131072
	defaultLocalParallelSlots   = 1
	defaultPromptCacheMinTokens = 2048
	// planNotesPresizedCap is the headroom NewPlan reserves on
	// plan.Notes when a Pack/ModelInfo is supplied. The hottest plans
	// emit 1-4 notes (context cap, model-quant warning, architecture
	// hint, MoE residency, optional JANGTQ note). Reserving 4 fits the
	// common case in a single 64-byte slice backing array and saves
	// 1-2 slice-grow allocs per plan.
	planNotesPresizedCap = 4
)

// NewPlan chooses opinionated local inference settings from measured memory.
//
//	plan := memory.NewPlan(memory.Input{Device: dev, Pack: pack})
func NewPlan(input Input) Plan {
	deviceMemory := input.Device.MemorySize
	workingSet := input.Device.MaxRecommendedWorkingSetSize
	if workingSet == 0 {
		workingSet = deviceMemory
	}
	class := classForBytes(deviceMemory)
	// Copy the matching pre-built per-class baseline. The previous
	// fillBaseClassPlan(*Plan, Class) shape paid for both a 480-byte
	// stack zero-init AND ~8 individual field writes per call; here
	// a single memcpy from a compile-time-resolved global gives the
	// runtime the freedom to SIMD-copy the whole struct in one shot.
	plan := classDefaultPlans[classBaselineIndex(class)]
	plan.MachineClass = class
	plan.Architecture = input.Device.Architecture
	plan.DeviceMemoryBytes = deviceMemory
	plan.RecommendedWorkingSetBytes = workingSet
	plan.MemoryLimitBytes = percentBytes(workingSet, 85)
	plan.CacheLimitBytes = percentBytes(workingSet, 8)
	plan.WiredLimitBytes = percentBytes(workingSet, 75)

	modelContext, modelQuant, modelQuantType, modelQuantFamily, modelArchitecture, modelWeightBytes := modelHints(input)
	// Pre-size the Notes slice once when a Pack is supplied with an
	// architecture string — that is the path through applyArchitectureHints
	// + applyGenericMoEResidency + (possibly) applyQuantizationHints that
	// emits 2-3 notes per plan on top of the optional context-cap +
	// model-quant warning. Pre-sizing collapses the slice-grow chain
	// (cap 1 → 2 → 4) into a single 4-element backing array, saving 1-2
	// grow allocs per Pack plan and pushing MiniMax M2 + Qwen3-MoE
	// plans down a full tier in alloc count.
	//
	// ModelInfo-only with architecture is left on the natural path —
	// it typically emits a single architecture note (no MoE/JANGTQ/etc),
	// and a 4-cap pre-allocation would be ~3x oversized for one entry.
	// No-Pack/no-ModelInfo plans (the cold-start NoPack benches) stay
	// at zero allocs as before.
	if input.Pack != nil && input.Pack.Architecture != "" {
		plan.Notes = make([]string, 0, planNotesPresizedCap)
	}
	// Derive the concurrency capacity from truth — how many full model-context
	// windows this machine's post-weights KV budget actually holds — and use it
	// for both ParallelSlots and BatchSize, in place of a per-RAM-class slot/
	// batch baseline that guessed the same numbers for every model AND made a
	// larger machine derive a SMALLER context (its bigger slot count divided the
	// KV budget harder than the extra memory grew it). One derived number keeps
	// the concurrency semaphore and the decode-batch KV multiplier coherent.
	// Generation models with a real fit only — encoders/rerankers keep the local
	// default, and a no-model plan keeps the honest one-foreground-slot baseline.
	if usesGenerationKVCacheWithProfile(input, nil) {
		if cc := concurrentContextsThatFit(plan, modelContext, modelWeightBytes, input); cc > 0 {
			plan.ParallelSlots = cc
			plan.BatchSize = cc
			plan.Notes = append(plan.Notes, "parallel slots + batch derived from device memory budget")
		}
	}
	// Derive context length from truth — the model's declared maximum bounded
	// by what this machine's memory budget actually holds — instead of leaving
	// it pinned at the RAM-class baseline, which could only ever cap DOWN and so
	// could never rise to a 256K model's capability on a machine that fits it.
	// Falls back to the plain metadata cap when the fit inputs (model weight
	// bytes + KV shape) are unavailable, so ModelInfo-only / cold-start plans
	// behave exactly as before.
	if fit := fitContextLength(plan, modelContext, modelWeightBytes, input); fit > 0 {
		if fit != plan.ContextLength {
			plan.ContextLength = fit
			plan.Notes = append(plan.Notes, "context length derived from device memory budget")
		}
	} else if modelContext > 0 && modelContext < plan.ContextLength {
		plan.ContextLength = modelContext
		plan.Notes = append(plan.Notes, "context capped by model metadata")
	}
	plan.ModelQuantization = modelQuant
	plan.ModelQuantizationType = modelQuantType
	plan.ModelQuantizationFamily = modelQuantFamily
	if input.Pack != nil {
		plan.ModelPackedQuantization = jang.ClonePackedProfile(input.Pack.PackedQuantization)
	}
	plan.ModelWeightBytes = modelWeightBytes
	// Resolve the canonical architecture once and look up the
	// profile registry exactly once for the whole NewPlan call. The
	// three downstream sites — applyArchitectureHints,
	// applyGenericMoEResidency, and usesGenerationKVCache — used to
	// each call profile.LookupArchitectureProfile, and the profile
	// package clones the entry on every lookup. Caching here saves
	// two clones (plus their child-slice allocations) per plan.
	//
	// The three sites had subtly different architecture precedence
	// in the original code: applyArchitectureHints used
	// modelArchitecture (ModelInfo > Pack), while
	// applyGenericMoEResidency + usesGenerationKVCache used the
	// Pack-precedence resolution (Pack > ModelInfo when both set).
	// Resolve both forms and only fall back to a second lookup when
	// the two strings differ; in the steady-state case where only
	// one of ModelInfo/Pack is populated they agree and we get one
	// lookup total.
	hintsArch := modelArchitecture
	packArch := modelArchitecture
	if input.Pack != nil && input.Pack.Architecture != "" {
		packArch = input.Pack.Architecture
	}
	// Pack carries its own ArchitectureProfile when the pack-creation
	// path has already resolved it — typical for native-loaded packs.
	// Use that instead of re-running profile.LookupArchitectureProfile,
	// which clones the registered profile on every call (~70% of plan
	// alloc footprint when a Pack is present). Only fall back to a
	// registry lookup when the Pack does not have the profile cached.
	var hintsPtr *profile.ModelArchitectureProfile
	var packPtr *profile.ModelArchitectureProfile
	if input.Pack != nil && input.Pack.ArchitectureProfile != nil {
		packPtr = input.Pack.ArchitectureProfile
		// hintsArch may still differ from packArch when ModelInfo
		// overrides the architecture. When they agree, the cached
		// profile is correct for both call sites.
		if packArch == hintsArch {
			hintsPtr = packPtr
		}
	}
	// Skip the lookups entirely when both architecture strings are
	// empty — NoPack/Device-only plans have no architecture to look
	// up and the registry would return (nil, false) for empty input
	// anyway. Saves two function calls per cold-start plan.
	if hintsPtr == nil && hintsArch != "" {
		if hintsProfile, hintsFound := profile.LookupArchitectureProfileRef(hintsArch); hintsFound {
			hintsPtr = hintsProfile
			if packArch == hintsArch {
				packPtr = hintsPtr
			}
		}
	}
	if packPtr == nil && packArch != hintsArch && packArch != "" {
		if packProfile, ok := profile.LookupArchitectureProfileRef(packArch); ok {
			packPtr = packProfile
		}
	}
	applyArchitectureHints(&plan, hintsArch, hintsPtr)
	applyQuantizationHints(&plan)
	applyGenericMoEResidency(&plan, input.Pack, packPtr)
	// Both KV-cache estimates use the same gating + shape — compute
	// once, scale the element count for each mode. usesGenerationKV
	// + kvEstimateShape used to run twice per plan.
	if usesGenerationKVCacheWithProfile(input, packPtr) && plan.ContextLength > 0 {
		if layers, hidden := kvEstimateShape(input, plan.MachineClass); layers > 0 && hidden > 0 {
			elements := uint64(plan.ContextLength) * uint64(layers) * uint64(hidden) * 2
			plan.EstimatedKVCacheBytes = elements * 2 // FP16 = 2 bytes/element
			plan.EstimatedKVCacheModeBytes = scaleKVElements(elements, plan.CacheMode)
		}
	}
	if plan.EstimatedKVCacheBytes > 0 && plan.EstimatedKVCacheModeBytes > 0 && plan.EstimatedKVCacheModeBytes < plan.EstimatedKVCacheBytes {
		plan.KVCacheSavingsRatio = 1 - float64(plan.EstimatedKVCacheModeBytes)/float64(plan.EstimatedKVCacheBytes)
	}
	return plan
}

// contextKVBudgetPercent is the conservative share of post-weights memory the
// planner allots to the KV cache when deriving context length from the actual
// machine, leaving headroom for activations, scratch, and runtime overhead. It
// is the single tunable safety reserve in the derivation — start conservative
// so a derived context never OOMs at serve, then bench per model to tune it.
const contextKVBudgetPercent uint64 = 70

// contextLengthAlignment rounds a derived context down to a clean token
// boundary so the limit reads as a deliberate value, not a raw division.
const contextLengthAlignment uint64 = 4096

// kvWidthPerLayer returns the per-layer KV-cache width (num_kv_heads * head_dim)
// the model declares, or 0 when the config did not carry it. This is the true
// grouped-query-attention cache width — far smaller than hidden_size on GQA
// models — so the planner sizes context from the real KV cost instead of an
// over-estimate that under-derives the context a machine actually fits.
func kvWidthPerLayer(input Input) int {
	if input.ModelInfo != nil && input.ModelInfo.NumKVHeads > 0 && input.ModelInfo.HeadDim > 0 {
		return input.ModelInfo.NumKVHeads * input.ModelInfo.HeadDim
	}
	if input.Pack != nil && input.Pack.NumKVHeads > 0 && input.Pack.HeadDim > 0 {
		return input.Pack.NumKVHeads * input.Pack.HeadDim
	}
	return 0
}

// perTokenKVBytes is the KV-cache cost of a single token across all layers for
// the planned cache mode: num_layers × (num_kv_heads × head_dim) × 2 (K and V),
// scaled by the mode's bytes-per-element. Per-layer width is the true grouped-
// query width when the model declares its KV dims (far below hidden_size), and
// falls back to hidden_size only when the config did not carry them — which
// over-estimates KV and so under-derives, never over-commits. Always positive:
// kvEstimateShape resolves a class-default shape when the model declares none,
// so the per-token cost is never zero. Shared by every memory-budget derivation
// so they size KV identically.
func perTokenKVBytes(plan Plan, input Input) uint64 {
	layers, hidden := kvEstimateShape(input, plan.MachineClass)
	width := kvWidthPerLayer(input)
	if width <= 0 {
		width = hidden
	}
	return scaleKVElements(uint64(layers)*uint64(width)*2, plan.CacheMode)
}

// fitContextLength derives the context length from truth: the model's declared
// maximum, bounded by the number of KV-cache tokens this machine's memory
// budget actually holds for the planned cache mode and parallel slots. It
// returns 0 — telling NewPlan to keep the class baseline / metadata-cap path —
// when the inputs to a real fit (model weight bytes and KV shape) are missing,
// so ModelInfo-only and cold-start plans are unaffected. The plan's baseline
// cache mode / parallel slots are used (architecture hints may shrink KV later),
// which only ever makes the estimate more conservative, never an over-commit.
func fitContextLength(plan Plan, modelContext int, modelWeightBytes uint64, input Input) int {
	if modelWeightBytes == 0 || plan.MemoryLimitBytes <= modelWeightBytes {
		return 0
	}
	// perToken is always > 0 (perTokenKVBytes resolves a class-default shape) and
	// slots is normalised to >= 1, so the divisor below is never zero.
	perToken := perTokenKVBytes(plan, input)
	slots := uint64(plan.ParallelSlots)
	if slots == 0 {
		slots = 1
	}
	kvBudget := percentBytes(plan.MemoryLimitBytes-modelWeightBytes, contextKVBudgetPercent)
	fit := kvBudget / (perToken * slots)
	if fit < contextLengthAlignment {
		return 0
	}
	fit -= fit % contextLengthAlignment
	// The model's declared maximum is the ceiling — never page positions the
	// model was never trained for, even when memory could hold more. When the
	// model declares no maximum, the class baseline stays the ceiling so an
	// unknown-context model is never raised past its conservative default.
	ceiling := uint64(modelContext)
	if modelContext <= 0 {
		ceiling = uint64(plan.ContextLength)
	}
	if ceiling > 0 && ceiling < fit {
		return int(ceiling)
	}
	return int(fit)
}

// concurrentContextsThatFit derives the single capacity that drives both
// ParallelSlots (the concurrency semaphore) and BatchSize (the decode-batch
// limit and the KV ×batch multiplier in estimateModelKVBytes): how many full
// model-context windows the machine's post-weights KV budget actually holds.
// Deriving one number keeps the two coherent — fitContextLength divides the KV
// budget by ParallelSlots, the KV estimate multiplies it by BatchSize, and both
// describe the same concurrent-sequence reservation.
//
// It is monotonic in memory: more RAM never reduces the count, so a larger
// machine can never derive fewer slots — and therefore never a smaller per-slot
// context — than a smaller one. That is the structural fix for the inversion
// the old per-RAM-class slot baseline produced. Returns 0 when a real fit
// cannot be computed (no weight bytes, no KV shape), telling NewPlan to keep
// the honest one-slot local default.
func concurrentContextsThatFit(plan Plan, modelContext int, modelWeightBytes uint64, input Input) int {
	if modelContext <= 0 || modelWeightBytes == 0 || plan.MemoryLimitBytes <= modelWeightBytes {
		return 0
	}
	// perToken is always > 0 and modelContext is > 0 here, so windowBytes (and the
	// divisor below) is never zero.
	perToken := perTokenKVBytes(plan, input)
	windowBytes := perToken * uint64(modelContext)
	kvBudget := percentBytes(plan.MemoryLimitBytes-modelWeightBytes, contextKVBudgetPercent)
	if windows := kvBudget / windowBytes; windows >= 1 {
		return int(windows)
	}
	return 1
}

// ClassForBytes returns the Class corresponding to the supplied memory
// size in bytes. Exported so callers that already know the device
// memory can pre-compute the class without a full plan.
//
//	class := memory.ClassForBytes(96 * memory.GiB)
func ClassForBytes(bytes uint64) Class { return classForBytes(bytes) }

func classForBytes(bytes uint64) Class {
	if bytes == 0 {
		return ClassUnknown
	}
	switch gib := (bytes + GiB - 1) / GiB; {
	case gib <= 18:
		return ClassApple16GB
	case gib <= 26:
		return ClassApple24GB
	case gib <= 40:
		return ClassApple32GB
	case gib <= 80:
		return ClassApple64GB
	case gib <= 112:
		return ClassApple96GB
	default:
		return ClassApple128GB
	}
}

// classDefaultPlans holds the immutable per-Class baseline used by
// NewPlan. Each entry carries only the class-specific fields; every
// other Plan field stays at its zero value. NewPlan dereferences the
// matching entry and copies it into the caller's local — one memcpy
// of 480 bytes is faster than the previous in-place fill (which paid
// for the zero-init AND ~8 ordinary field writes per call) because
// the runtime can use unrolled SIMD memcpy and the source is a
// compile-time-resolved global.
//
// All populated classes use KVCacheRotating; the Unknown/default
// fallback also lives here so the lookup never misses.
//
// ParallelSlots and BatchSize are the honest one-foreground-slot cold
// default (1) in every entry — they are NOT class-specific. NewPlan
// derives the real concurrency capacity from the model's footprint when a
// model is known (concurrentContextsThatFit); this baseline stands only
// when there is no model to size against.
var classDefaultPlans = [...]Plan{
	indexClassApple16GB: {
		CachePolicy:      KVCacheRotating,
		ContextLength:    8192,
		CacheMode:        KVCacheModeKQ8VQ4,
		BatchSize:        1,
		PrefillChunkSize: 512,
		ParallelSlots:    1,
	},
	indexClassApple24GB: {
		CachePolicy:          KVCacheRotating,
		ContextLength:        16384,
		CacheMode:            KVCacheModeQ8,
		BatchSize:            1,
		PrefillChunkSize:     768,
		ParallelSlots:        1,
		PromptCache:          true,
		PromptCacheMinTokens: 4096,
	},
	indexClassApple32GB: {
		CachePolicy:          KVCacheRotating,
		ContextLength:        32768,
		CacheMode:            KVCacheModeQ8,
		BatchSize:            1,
		PrefillChunkSize:     1024,
		ParallelSlots:        1,
		PromptCache:          true,
		PromptCacheMinTokens: 4096,
	},
	indexClassApple64GB: {
		CachePolicy:          KVCacheRotating,
		ContextLength:        32768,
		CacheMode:            KVCacheModeDefault,
		BatchSize:            1,
		PrefillChunkSize:     4096,
		ParallelSlots:        1,
		PromptCache:          true,
		PromptCacheMinTokens: defaultPromptCacheMinTokens,
	},
	indexClassApple96GB: {
		CachePolicy:          KVCacheRotating,
		ContextLength:        defaultLocalContextLength,
		CacheMode:            KVCacheModeDefault,
		BatchSize:            1,
		PrefillChunkSize:     4096,
		ParallelSlots:        1,
		PromptCache:          true,
		PromptCacheMinTokens: defaultPromptCacheMinTokens,
	},
	indexClassApple128GB: {
		CachePolicy:          KVCacheRotating,
		ContextLength:        defaultLocalContextLength,
		CacheMode:            KVCacheModeDefault,
		BatchSize:            1,
		PrefillChunkSize:     4096,
		ParallelSlots:        1,
		PromptCache:          true,
		PromptCacheMinTokens: defaultPromptCacheMinTokens,
	},
	indexClassUnknown: {
		CachePolicy:          KVCacheRotating,
		ContextLength:        defaultLocalContextLength,
		CacheMode:            KVCacheModeQ8,
		BatchSize:            1,
		PrefillChunkSize:     1024,
		ParallelSlots:        defaultLocalParallelSlots,
		PromptCache:          true,
		PromptCacheMinTokens: defaultPromptCacheMinTokens,
	},
}

// classBaselineIndex maps a Class to its slot in classDefaultPlans.
// Inlined into NewPlan so the lookup is a single switch + array
// index (~3 ns) instead of a function call plus per-field-write.
func classBaselineIndex(class Class) int {
	switch class {
	case ClassApple16GB:
		return indexClassApple16GB
	case ClassApple24GB:
		return indexClassApple24GB
	case ClassApple32GB:
		return indexClassApple32GB
	case ClassApple64GB:
		return indexClassApple64GB
	case ClassApple96GB:
		return indexClassApple96GB
	case ClassApple128GB:
		return indexClassApple128GB
	default:
		return indexClassUnknown
	}
}

const (
	indexClassApple16GB = iota
	indexClassApple24GB
	indexClassApple32GB
	indexClassApple64GB
	indexClassApple96GB
	indexClassApple128GB
	indexClassUnknown
)

func estimateKVCacheBytes(plan Plan, input Input, mode KVCacheMode) uint64 {
	return estimateKVCacheBytesWithProfile(plan, input, mode, nil)
}

func estimateKVCacheBytesWithProfile(plan Plan, input Input, mode KVCacheMode, profileHint *profile.ModelArchitectureProfile) uint64 {
	if !usesGenerationKVCacheWithProfile(input, profileHint) {
		return 0
	}
	if plan.ContextLength <= 0 {
		return 0
	}
	// kvEstimateShape always resolves a positive (layers, hidden) — a class
	// default when the model declares none — so no zero-shape guard is needed.
	layers, hidden := kvEstimateShape(input, plan.MachineClass)
	elements := uint64(plan.ContextLength) * uint64(layers) * uint64(hidden) * 2
	return scaleKVElements(elements, mode)
}

// scaleKVElements maps the raw element count to bytes for the given
// KV cache mode. Hoisted from estimateKVCacheBytes so NewPlan can
// run the gating + shape compute once and call this twice instead.
func scaleKVElements(elements uint64, mode KVCacheMode) uint64 {
	switch mode {
	case KVCacheModeKQ8VQ4:
		return elements * 3 / 4
	case KVCacheModeQ8:
		return elements
	case KVCacheModeTurboQuant:
		return scaleElementsByByteRatioCeil(elements, 7, 16) // 3.5 bits per KV element.
	default:
		return elements * 2
	}
}

func scaleElementsByByteRatioCeil(elements, numerator, denominator uint64) uint64 {
	if elements == 0 || numerator == 0 || denominator == 0 {
		return 0
	}
	return (elements*numerator + denominator - 1) / denominator
}

func kvEstimateShape(input Input, class Class) (layers, hidden int) {
	if input.ModelInfo != nil {
		layers = input.ModelInfo.NumLayers
		hidden = input.ModelInfo.HiddenSize
	}
	if input.Pack != nil {
		if layers == 0 {
			layers = input.Pack.NumLayers
		}
		if hidden == 0 {
			hidden = input.Pack.HiddenSize
		}
	}
	if layers > 0 && hidden > 0 {
		return layers, hidden
	}
	switch class {
	case ClassApple16GB, ClassApple24GB:
		return 28, 2048
	case ClassApple32GB:
		return 32, 3072
	case ClassApple64GB:
		return 40, 4096
	default:
		return 48, 5120
	}
}

func modelHints(input Input) (contextLength, quantization int, quantType, quantFamily, architecture string, weightBytes uint64) {
	if input.Pack != nil {
		contextLength = input.Pack.ContextLength
		quantization = input.Pack.QuantBits
		quantType = input.Pack.QuantType
		quantFamily = input.Pack.QuantFamily
		architecture = input.Pack.Architecture
		weightBytes = input.Pack.WeightBytes
	}
	if input.ModelInfo != nil {
		if input.ModelInfo.Architecture != "" {
			architecture = input.ModelInfo.Architecture
		}
		if input.ModelInfo.ContextLength > 0 {
			contextLength = input.ModelInfo.ContextLength
		}
		if input.ModelInfo.QuantBits > 0 {
			quantization = input.ModelInfo.QuantBits
		}
	}
	return contextLength, quantization, quantType, quantFamily, architecture, weightBytes
}

func applyArchitectureHints(plan *Plan, architecture string, profileHint *profile.ModelArchitectureProfile) {
	// Profile registry is authoritative when it matches — skip the
	// normalize allocation entirely in that case. NewPlan has already
	// looked the architecture up in the registry and only passes a
	// non-nil profileHint on hit, so a nil profileHint means the
	// registry does not know this architecture and we go straight to
	// the normalize fallback. The prior default branch repeated the
	// LookupArchitectureProfile call (which clones the profile every
	// call — 70% of the alloc footprint on NewPlan_Qwen3MoEPack).
	var normalized string
	if profileHint != nil {
		normalized = profileHint.ID
	} else if architecture != "" {
		// Empty architecture short-circuit — NoPack plans hit this
		// path with arch="" on every call. Avoid the normalize jump
		// for a guaranteed-empty result, which would no-op through the
		// switch anyway.
		normalized = profile.NormalizeArchitecture(architecture)
	}
	switch normalized {
	case "qwen2":
		plan.Notes = append(plan.Notes, "Qwen2.x uses the native Qwen decoder; long contexts benefit from paged or compact KV cache modes on Apple unified memory")
	case "qwen3_moe":
		plan.Notes = append(plan.Notes, "Qwen3-MoE sparse expert routing increases memory pressure; prefer compact KV cache modes on constrained Apple memory")
		if plan.MachineClass == ClassApple24GB || plan.MachineClass == ClassApple32GB {
			plan.CacheMode = KVCacheModeKQ8VQ4
			plan.Notes = append(plan.Notes, "Qwen3-MoE uses asymmetric K@q8,V@q4 cache below 64GB")
		}
	case "qwen3_6":
		plan.Notes = append(plan.Notes, "Qwen3.6 uses hybrid linear attention; native Go kernels are pending")
		plan.ParallelSlots = 1
		if plan.PrefillChunkSize > 2048 {
			plan.PrefillChunkSize = 2048
		}
	case "qwen3_6_moe":
		plan.Notes = append(plan.Notes, "Qwen3.6-MoE uses hybrid linear attention plus routed experts; native Go kernels are pending")
		plan.ParallelSlots = 1
		if plan.PrefillChunkSize > 2048 {
			plan.PrefillChunkSize = 2048
		}
		if plan.MachineClass == ClassApple16GB || plan.MachineClass == ClassApple24GB || plan.MachineClass == ClassApple32GB {
			plan.CacheMode = KVCacheModeKQ8VQ4
			plan.Notes = append(plan.Notes, "Qwen3.6-MoE uses asymmetric K@q8,V@q4 cache below 64GB")
		}
	case "qwen3_next":
		plan.Notes = append(plan.Notes, "Qwen3-Next uses nested text_config metadata; keep context and cache policy tied to text model limits")
	case "minimax_m2":
		plan.Notes = append(plan.Notes, "MiniMax M2 MoE has a large routed-expert footprint; keep prefill narrow and prefer paged cache on Apple unified memory")
		plan.ParallelSlots = 1
		plan.BatchSize = 1
		if plan.PrefillChunkSize > 2048 {
			plan.PrefillChunkSize = 2048
		}
		if plan.ContextLength > 32768 {
			plan.ContextLength = 32768
			plan.Notes = append(plan.Notes, "MiniMax M2 context capped for 96GB-class local inference")
		}
		if plan.MachineClass == ClassApple16GB || plan.MachineClass == ClassApple24GB || plan.MachineClass == ClassApple32GB {
			plan.ContextLength = minPositive(plan.ContextLength, 8192)
			plan.CacheMode = KVCacheModeKQ8VQ4
			plan.Notes = append(plan.Notes, "MiniMax M2 requires asymmetric compact KV cache below 64GB")
		}
	case "bert":
		applyEncoderHints(plan, encoderHintBert)
	case "bert_rerank":
		applyEncoderHints(plan, encoderHintBertRerank)
	}
}

func applyEncoderHints(plan *Plan, label string) {
	plan.CachePolicy = KVCacheDefault
	plan.CacheMode = KVCacheModeDefault
	plan.PromptCache = false
	plan.PromptCacheMinTokens = 0
	if plan.PrefillChunkSize == 0 || plan.PrefillChunkSize > 512 {
		plan.PrefillChunkSize = 512
	}
	switch plan.MachineClass {
	case ClassApple16GB, ClassApple24GB:
		if plan.BatchSize < 8 {
			plan.BatchSize = 8
		}
	case ClassApple32GB:
		if plan.BatchSize < 16 {
			plan.BatchSize = 16
		}
	case ClassApple64GB, ClassApple96GB:
		if plan.BatchSize < 32 {
			plan.BatchSize = 32
		}
	case ClassApple128GB:
		if plan.BatchSize < 48 {
			plan.BatchSize = 48
		}
	default:
		if plan.BatchSize < 4 {
			plan.BatchSize = 4
		}
	}
	plan.Notes = append(plan.Notes, label)
}

// Pre-computed encoder hint strings — applyEncoderHints used to build
// these by concatenating a per-call label with a constant suffix at
// runtime. With only two call sites it is cheaper to pre-compute the
// full strings as package-level constants and pass the matching one in.
const (
	encoderHintBert       = "BERT embedding encoder uses pooled sequence outputs and does not allocate generation KV cache"
	encoderHintBertRerank = "BERT cross-encoder rerank uses pooled sequence outputs and does not allocate generation KV cache"
)

func usesGenerationKVCache(input Input) bool {
	return usesGenerationKVCacheWithProfile(input, nil)
}

func usesGenerationKVCacheWithProfile(input Input, profileHint *profile.ModelArchitectureProfile) bool {
	// Cheapest checks first — Pack-resident flags short-circuit
	// without touching the architecture string or the profile
	// registry. Most callers that pass Embedding/Rerank packs return
	// here.
	if input.Pack != nil {
		if input.Pack.Embedding != nil || input.Pack.Rerank != nil {
			return false
		}
		if input.Pack.ArchitectureProfile != nil && (input.Pack.ArchitectureProfile.Embeddings || input.Pack.ArchitectureProfile.Rerank) {
			return false
		}
	}
	// Caller may have already done the registry lookup — use the
	// cached profile instead of touching the registry again.
	if profileHint != nil {
		if profileHint.Embeddings || profileHint.Rerank {
			return false
		}
		return true
	}
	// Fall through to the legacy single-call path.
	architecture := ""
	if input.Pack != nil && input.Pack.Architecture != "" {
		architecture = input.Pack.Architecture
	} else if input.ModelInfo != nil {
		architecture = input.ModelInfo.Architecture
	}
	if p, ok := profile.LookupArchitectureProfileRef(architecture); ok && (p.Embeddings || p.Rerank) {
		return false
	}
	return true
}

func applyQuantizationHints(plan *Plan) {
	if plan.ModelQuantizationFamily != "jang" && plan.ModelQuantizationType != "jangtq" {
		return
	}
	plan.Notes = append(plan.Notes, "JANGTQ/JANG mixed precision protects attention while compressing routed experts; fit estimates should use measured weight bytes over uniform-bit heuristics")
}

// genericMoENotes is the static Notes slice for the generic MoE
// residency plan — every MoE pack lands here so the same slice is
// safe to share. The Notes field is read-only after the plan is
// returned (the ExpertResidencyPlan is value-copied into Plan, so
// callers cannot mutate this slice without first copying it).
var genericMoENotes = []string{"MoE model uses lazy expert residency until backend-specific expert byte estimates are available"}

func applyGenericMoEResidency(plan *Plan, pack *mp.ModelPack, profileHint *profile.ModelArchitectureProfile) {
	if plan == nil {
		return
	}
	if profileHint == nil || !profileHint.MoE {
		return
	}
	// Reach through the pointer for the single field we use rather
	// than copying the whole 200-byte ModelArchitectureProfile struct
	// onto the stack for one string read. The Plan-bound ID field is
	// just the architecture name, not a clone of the profile.
	plan.ExpertResidency = ExpertResidencyPlan{
		Enabled:                 true,
		Mode:                    ExpertResidencyModeLazy,
		Architecture:            profileHint.ID,
		MaxResidentExperts:      genericMoEResidentExpertLimit(plan.MachineClass),
		PageInBatchSize:         1,
		EvictionPolicy:          ExpertEvictionLRU,
		FirstUseLatencyExpected: true,
		Notes:                   genericMoENotes,
	}
	plan.Notes = append(plan.Notes, "lazy expert residency enabled for MoE architecture")
}

func genericMoEResidentExpertLimit(class Class) int {
	switch class {
	case ClassApple16GB, ClassApple24GB:
		return 2
	case ClassApple32GB:
		return 4
	case ClassApple64GB:
		return 8
	case ClassApple96GB:
		return 16
	case ClassApple128GB:
		return 24
	default:
		return 2
	}
}

func minPositive(a, b int) int {
	if a <= 0 {
		return b
	}
	if b <= 0 {
		return a
	}
	if a < b {
		return a
	}
	return b
}

func percentBytes(value uint64, percent uint64) uint64 {
	if value == 0 {
		return 0
	}
	return value * percent / 100
}
