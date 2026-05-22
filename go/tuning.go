// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"context"
	"strconv"

	core "dappco.re/go"
)

// TuningWorkload identifies the user-facing job a local model profile is
// being optimised for. The values are stable so UIs can persist profiles.
type TuningWorkload string

const (
	TuningWorkloadChat        TuningWorkload = "chat"
	TuningWorkloadCoding      TuningWorkload = "coding"
	TuningWorkloadLongContext TuningWorkload = "long_context"
	TuningWorkloadAgentState  TuningWorkload = "agent_state"
	TuningWorkloadThroughput  TuningWorkload = "throughput"
	TuningWorkloadLowLatency  TuningWorkload = "low_latency"
)

var defaultTuningWorkloads = []TuningWorkload{
	TuningWorkloadChat,
	TuningWorkloadCoding,
	TuningWorkloadLongContext,
	TuningWorkloadAgentState,
	TuningWorkloadThroughput,
	TuningWorkloadLowLatency,
}

// DefaultTuningWorkloads returns the standard set shown by local tuning UIs.
func DefaultTuningWorkloads() []TuningWorkload {
	return append([]TuningWorkload(nil), defaultTuningWorkloads...)
}

// MachineDiscoverer is implemented by runtimes that can report local hardware,
// supported settings, and optionally discovered model packs without loading
// weights.
type MachineDiscoverer interface {
	DiscoverMachine(context.Context, MachineDiscoveryRequest) (*MachineDiscoveryReport, error)
}

// TuningPlanner is implemented by runtimes that can propose candidate load
// settings for a model/workload pair.
type TuningPlanner interface {
	PlanTuning(context.Context, TuningPlanRequest) (*TuningPlan, error)
}

// MachineDeviceInfo records the backend-neutral hardware facts a driver can
// expose before any model is loaded.
type MachineDeviceInfo struct {
	Name                         string            `json:"name,omitempty"`
	Architecture                 string            `json:"architecture,omitempty"`
	MaxBufferLength              uint64            `json:"max_buffer_length,omitempty"`
	MaxRecommendedWorkingSetSize uint64            `json:"max_recommended_working_set_size,omitempty"`
	MemorySize                   uint64            `json:"memory_size,omitempty"`
	Labels                       map[string]string `json:"labels,omitempty"`
}

// MachineDiscoveryRequest controls cheap local discovery. Drivers should keep
// this metadata-first and avoid loading weights.
type MachineDiscoveryRequest struct {
	ModelDirs         []string          `json:"model_dirs,omitempty"`
	Workloads         []TuningWorkload  `json:"workloads,omitempty"`
	MaxModels         int               `json:"max_models,omitempty"`
	IncludeModels     bool              `json:"include_models,omitempty"`
	IncludeCandidates bool              `json:"include_candidates,omitempty"`
	Labels            map[string]string `json:"labels,omitempty"`
}

// MachineDiscoveryReport is the UI-facing summary of a local backend plus any
// models and candidate settings discovered cheaply.
type MachineDiscoveryReport struct {
	Runtime      RuntimeIdentity   `json:"runtime,omitempty"`
	Device       MachineDeviceInfo `json:"device,omitempty"`
	Available    bool              `json:"available"`
	Capabilities []Capability      `json:"capabilities,omitempty"`
	CacheModes   []string          `json:"cache_modes,omitempty"`
	Models       []DiscoveredModel `json:"models,omitempty"`
	Workloads    []TuningWorkload  `json:"workloads,omitempty"`
	Candidates   []TuningCandidate `json:"candidates,omitempty"`
	Warnings     []string          `json:"warnings,omitempty"`
	Labels       map[string]string `json:"labels,omitempty"`
}

// TuningBudget bounds optional autotuning work. Zero values mean the driver
// picks a short smoke-test default.
type TuningBudget struct {
	MaxCandidates     int  `json:"max_candidates,omitempty"`
	SmokeTokens       int  `json:"smoke_tokens,omitempty"`
	Runs              int  `json:"runs,omitempty"`
	AllowStateBench   bool `json:"allow_state_bench,omitempty"`
	AllowModelReloads bool `json:"allow_model_reloads,omitempty"`
}

// TuningPlanRequest asks a backend to turn known hardware/model facts into
// candidate settings. It is intentionally metadata-only.
type TuningPlanRequest struct {
	Runtime   RuntimeIdentity   `json:"runtime,omitempty"`
	Device    MachineDeviceInfo `json:"device,omitempty"`
	Model     ModelIdentity     `json:"model,omitempty"`
	Adapter   AdapterIdentity   `json:"adapter,omitempty"`
	Workloads []TuningWorkload  `json:"workloads,omitempty"`
	Budget    TuningBudget      `json:"budget,omitempty"`
	Labels    map[string]string `json:"labels,omitempty"`
}

// TuningCandidate is one concrete model-load shape the UI can try or persist.
type TuningCandidate struct {
	ID                   string            `json:"id,omitempty"`
	Workload             TuningWorkload    `json:"workload,omitempty"`
	Model                ModelIdentity     `json:"model,omitempty"`
	Adapter              AdapterIdentity   `json:"adapter,omitempty"`
	Runtime              RuntimeIdentity   `json:"runtime,omitempty"`
	ContextLength        int               `json:"context_length,omitempty"`
	ParallelSlots        int               `json:"parallel_slots,omitempty"`
	PromptCache          bool              `json:"prompt_cache,omitempty"`
	PromptCacheMinTokens int               `json:"prompt_cache_min_tokens,omitempty"`
	CachePolicy          string            `json:"cache_policy,omitempty"`
	CacheMode            string            `json:"cache_mode,omitempty"`
	BatchSize            int               `json:"batch_size,omitempty"`
	PrefillChunkSize     int               `json:"prefill_chunk_size,omitempty"`
	ExpectedQuantization int               `json:"expected_quantization,omitempty"`
	MemoryLimitBytes     uint64            `json:"memory_limit_bytes,omitempty"`
	CacheLimitBytes      uint64            `json:"cache_limit_bytes,omitempty"`
	WiredLimitBytes      uint64            `json:"wired_limit_bytes,omitempty"`
	Reasons              []string          `json:"reasons,omitempty"`
	Labels               map[string]string `json:"labels,omitempty"`
}

// TuningPlan is a compact set of candidates and per-workload recommendations.
type TuningPlan struct {
	Runtime     RuntimeIdentity           `json:"runtime,omitempty"`
	Device      MachineDeviceInfo         `json:"device,omitempty"`
	Model       ModelIdentity             `json:"model,omitempty"`
	Adapter     AdapterIdentity           `json:"adapter,omitempty"`
	Workloads   []TuningWorkload          `json:"workloads,omitempty"`
	Candidates  []TuningCandidate         `json:"candidates,omitempty"`
	Recommended map[TuningWorkload]string `json:"recommended,omitempty"`
	Warnings    []string                  `json:"warnings,omitempty"`
	Labels      map[string]string         `json:"labels,omitempty"`
}

// TuningMeasurements is the driver-neutral subset of a bench result used for
// scoring and persisted profiles.
type TuningMeasurements struct {
	PromptTokens            int     `json:"prompt_tokens,omitempty"`
	GeneratedTokens         int     `json:"generated_tokens,omitempty"`
	LoadMilliseconds        float64 `json:"load_milliseconds,omitempty"`
	FirstTokenMilliseconds  float64 `json:"first_token_milliseconds,omitempty"`
	PrefillTokensPerSec     float64 `json:"prefill_tokens_per_sec,omitempty"`
	DecodeTokensPerSec      float64 `json:"decode_tokens_per_sec,omitempty"`
	PromptCacheHitRate      float64 `json:"prompt_cache_hit_rate,omitempty"`
	KVRestoreMilliseconds   float64 `json:"kv_restore_milliseconds,omitempty"`
	StateBundleMilliseconds float64 `json:"state_bundle_milliseconds,omitempty"`
	TotalMilliseconds       float64 `json:"total_milliseconds,omitempty"`
	PeakMemoryBytes         uint64  `json:"peak_memory_bytes,omitempty"`
	ActiveMemoryBytes       uint64  `json:"active_memory_bytes,omitempty"`
	CorrectnessSmokeResult  string  `json:"correctness_smoke_result,omitempty"`
	CorrectnessSmokeChecks  int     `json:"correctness_smoke_checks,omitempty"`
}

// TuningScore records a comparable score plus the raw metrics that drove it.
type TuningScore struct {
	Workload               TuningWorkload    `json:"workload,omitempty"`
	Score                  float64           `json:"score,omitempty"`
	FirstTokenMilliseconds float64           `json:"first_token_milliseconds,omitempty"`
	PrefillTokensPerSec    float64           `json:"prefill_tokens_per_sec,omitempty"`
	DecodeTokensPerSec     float64           `json:"decode_tokens_per_sec,omitempty"`
	PromptCacheHitRate     float64           `json:"prompt_cache_hit_rate,omitempty"`
	KVRestoreMilliseconds  float64           `json:"kv_restore_milliseconds,omitempty"`
	PeakMemoryBytes        uint64            `json:"peak_memory_bytes,omitempty"`
	Labels                 map[string]string `json:"labels,omitempty"`
}

// TuningResult is emitted after each candidate finishes or fails.
type TuningResult struct {
	Candidate    TuningCandidate    `json:"candidate,omitempty"`
	Measurements TuningMeasurements `json:"measurements,omitempty"`
	Score        TuningScore        `json:"score,omitempty"`
	Error        string             `json:"error,omitempty"`
	Labels       map[string]string  `json:"labels,omitempty"`
}

// TuningEventKind names the streamed lifecycle events an autotune runner emits.
type TuningEventKind string

const (
	TuningEventCandidate TuningEventKind = "candidate"
	TuningEventResult    TuningEventKind = "result"
	TuningEventSelected  TuningEventKind = "selected"
)

// TuningEvent lets UIs update as each candidate starts and finishes.
type TuningEvent struct {
	Kind      TuningEventKind   `json:"kind"`
	Candidate TuningCandidate   `json:"candidate,omitempty"`
	Result    *TuningResult     `json:"result,omitempty"`
	Labels    map[string]string `json:"labels,omitempty"`
}

// TuningProfileKey identifies a persisted winner for one machine/model/workload.
type TuningProfileKey struct {
	MachineHash string          `json:"machine_hash,omitempty"`
	Runtime     RuntimeIdentity `json:"runtime,omitempty"`
	Model       ModelIdentity   `json:"model,omitempty"`
	Adapter     AdapterIdentity `json:"adapter,omitempty"`
	Workload    TuningWorkload  `json:"workload,omitempty"`
}

// TuningProfile stores a proven candidate for later fast reloads.
type TuningProfile struct {
	Key           TuningProfileKey   `json:"key,omitempty"`
	Candidate     TuningCandidate    `json:"candidate,omitempty"`
	Measurements  TuningMeasurements `json:"measurements,omitempty"`
	Score         TuningScore        `json:"score,omitempty"`
	CreatedAtUnix int64              `json:"created_at_unix,omitempty"`
	Labels        map[string]string  `json:"labels,omitempty"`
}

// ScoreTuningMeasurements turns measured smoke-test counters into a simple
// workload-aware score. It deliberately stays transparent rather than claiming
// a universal benchmark.
func ScoreTuningMeasurements(workload TuningWorkload, m TuningMeasurements) TuningScore {
	// Labels map is lazy: most workloads emit zero label entries (Chat,
	// Throughput, Default — and LongContext/AgentState/LowLatency when
	// the optional measurements are missing). Eager-init then nil-out
	// pays an empty-map alloc per call (~48 B/op) which escapes to heap
	// because TuningScore returns the labels pointer. Lazy-init defers
	// the alloc to the moment the first label key is written, and the
	// no-label paths stay at zero heap allocs for the labels slot. When
	// a label IS written, the map is pre-sized to the small upper bound
	// for that workload to skip the default grow-from-empty.
	var labels map[string]string
	score := m.DecodeTokensPerSec
	switch workload {
	case TuningWorkloadLongContext:
		score += m.PrefillTokensPerSec * 0.2
		if m.PromptCacheHitRate > 0 {
			score += m.PromptCacheHitRate * 100
			labels = make(map[string]string, 1)
			labels["prompt_cache"] = "enabled"
		}
	case TuningWorkloadAgentState:
		score += m.PrefillTokensPerSec * 0.1
		score += m.PromptCacheHitRate * 120
		if m.KVRestoreMilliseconds > 0 {
			score += 1000 / (m.KVRestoreMilliseconds + 1)
			if labels == nil {
				labels = make(map[string]string, 2)
			}
			labels["state_restore"] = "enabled"
		}
		if m.StateBundleMilliseconds > 0 {
			score += 500 / (m.StateBundleMilliseconds + 1)
			if labels == nil {
				labels = make(map[string]string, 2)
			}
			labels["state_bundle"] = "enabled"
		}
	case TuningWorkloadThroughput:
		score += m.PrefillTokensPerSec * 0.05
	case TuningWorkloadLowLatency:
		if m.FirstTokenMilliseconds > 0 {
			score += 1000 / (m.FirstTokenMilliseconds + 1)
			labels = make(map[string]string, 1)
			labels["first_token"] = "measured"
		}
		if m.TotalMilliseconds > 0 {
			score += 1000 / m.TotalMilliseconds
		}
	default:
		score += m.PrefillTokensPerSec * 0.02
	}
	return TuningScore{
		Workload:               workload,
		Score:                  score,
		FirstTokenMilliseconds: m.FirstTokenMilliseconds,
		PrefillTokensPerSec:    m.PrefillTokensPerSec,
		DecodeTokensPerSec:     m.DecodeTokensPerSec,
		PromptCacheHitRate:     m.PromptCacheHitRate,
		KVRestoreMilliseconds:  m.KVRestoreMilliseconds,
		PeakMemoryBytes:        m.PeakMemoryBytes,
		Labels:                 labels,
	}
}

// ModelReplaceAction describes the safest way to move between loaded models
// or settings while preserving useful state where possible.
type ModelReplaceAction string

const (
	ModelReplaceReuseState      ModelReplaceAction = "reuse_state"
	ModelReplaceCheckpointState ModelReplaceAction = "checkpoint_state"
	ModelReplaceSummaryWindow   ModelReplaceAction = "summary_window"
)

// ModelReplaceRequest compares the current runtime/model/adapter against the
// requested replacement.
type ModelReplaceRequest struct {
	CurrentModel   ModelIdentity   `json:"current_model,omitempty"`
	NextModel      ModelIdentity   `json:"next_model,omitempty"`
	CurrentRuntime RuntimeIdentity `json:"current_runtime,omitempty"`
	NextRuntime    RuntimeIdentity `json:"next_runtime,omitempty"`
	CurrentAdapter AdapterIdentity `json:"current_adapter,omitempty"`
	NextAdapter    AdapterIdentity `json:"next_adapter,omitempty"`
}

// ModelReplacePlan tells the UI whether state can be reused directly or should
// be compacted into a summary/new window before reload.
type ModelReplacePlan struct {
	Action     ModelReplaceAction `json:"action"`
	Compatible bool               `json:"compatible"`
	Reasons    []string           `json:"reasons,omitempty"`
}

// PlanModelReplace returns a conservative state-reuse decision for model swaps.
func PlanModelReplace(req ModelReplaceRequest) ModelReplacePlan {
	sameModel := sameModelIdentity(req.CurrentModel, req.NextModel)
	sameRuntime := sameRuntimeIdentity(req.CurrentRuntime, req.NextRuntime)
	sameAdapter := sameAdapterIdentity(req.CurrentAdapter, req.NextAdapter)
	switch {
	case sameModel && sameRuntime && sameAdapter:
		return ModelReplacePlan{Action: ModelReplaceReuseState, Compatible: true, Reasons: []string{"model, runtime, and adapter match"}}
	case sameModel && sameAdapter:
		// CheckpointState path: 0 or 1 reason. Pre-size the backing
		// array so the append (when it fires) does not trigger an
		// extra grow alloc; when sameRuntime keeps it empty the slice
		// is still nil so json.Marshal honours omitempty correctly.
		var reasons []string
		if !sameRuntime {
			reasons = make([]string, 0, 1)
			reasons = append(reasons, "runtime or cache settings changed")
		}
		return ModelReplacePlan{Action: ModelReplaceCheckpointState, Compatible: true, Reasons: reasons}
	default:
		// SummaryWindow path: up to 2 reasons (model + adapter). The
		// previous shape allocated `[]string{}` and then grew on each
		// append — two allocs by the second append. Pre-sizing to 2
		// drops the grow.
		reasons := make([]string, 0, 2)
		if !sameModel {
			reasons = append(reasons, "model identity changed")
		}
		if !sameAdapter {
			reasons = append(reasons, "adapter identity changed")
		}
		return ModelReplacePlan{Action: ModelReplaceSummaryWindow, Compatible: false, Reasons: reasons}
	}
}

func sameModelIdentity(a, b ModelIdentity) bool {
	if a.Hash != "" || b.Hash != "" {
		return a.Hash != "" && a.Hash == b.Hash
	}
	if a.Path != "" || b.Path != "" {
		return a.Path != "" && a.Path == b.Path && a.QuantBits == b.QuantBits && a.QuantType == b.QuantType
	}
	return a.Architecture == b.Architecture && a.QuantBits == b.QuantBits && a.ContextLength == b.ContextLength
}

func sameRuntimeIdentity(a, b RuntimeIdentity) bool {
	return a.Backend == b.Backend && a.Device == b.Device && a.CacheMode == b.CacheMode
}

func sameAdapterIdentity(a, b AdapterIdentity) bool {
	if a.Hash != "" || b.Hash != "" {
		return a.Hash != "" && a.Hash == b.Hash
	}
	return a.Path == b.Path && a.Format == b.Format && a.Rank == b.Rank && a.Alpha == b.Alpha
}

// CandidateID builds a stable readable ID when a planner has not supplied one.
//
// Hand-built via strconv.AppendInt + core.AsString — saves the fmt
// formatter pipeline that Sprintf would walk for every tuning lookup.
func CandidateID(workload TuningWorkload, cacheMode string, contextLength, batchSize int) string {
	buf := make([]byte, 0, len(workload)+len(cacheMode)+32)
	buf = append(buf, string(workload)...)
	buf = append(buf, ':')
	buf = append(buf, cacheMode...)
	buf = append(buf, ':', 'c', 't', 'x')
	buf = strconv.AppendInt(buf, int64(contextLength), 10)
	buf = append(buf, ':', 'b', 'a', 't', 'c', 'h')
	buf = strconv.AppendInt(buf, int64(batchSize), 10)
	return core.AsString(buf)
}
