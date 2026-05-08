// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"context"
	"maps"
	"slices"
)

// CapabilityGroup identifies the layer a capability belongs to.
type CapabilityGroup string

const (
	// CapabilityGroupModel covers model-facing inference and model-pack features.
	CapabilityGroupModel CapabilityGroup = "model"
	// CapabilityGroupRuntime covers hardware/runtime planning and loading.
	CapabilityGroupRuntime CapabilityGroup = "runtime"
	// CapabilityGroupTraining covers native training and adapter update loops.
	CapabilityGroupTraining CapabilityGroup = "training"
	// CapabilityGroupProbe covers research telemetry and model-state probing.
	CapabilityGroupProbe CapabilityGroup = "probe"
)

// CapabilityStatus records whether a feature is usable today.
type CapabilityStatus string

const (
	CapabilityStatusSupported    CapabilityStatus = "supported"
	CapabilityStatusExperimental CapabilityStatus = "experimental"
	CapabilityStatusPlanned      CapabilityStatus = "planned"
	CapabilityStatusUnsupported  CapabilityStatus = "unsupported"
)

// CapabilityID is a stable feature identifier shared by backends and callers.
type CapabilityID string

const (
	CapabilityModelLoad       CapabilityID = "model.load"
	CapabilityGenerate        CapabilityID = "generate"
	CapabilityChat            CapabilityID = "chat"
	CapabilityClassify        CapabilityID = "classify"
	CapabilityBatchGenerate   CapabilityID = "batch.generate"
	CapabilityTokenizer       CapabilityID = "tokenizer"
	CapabilityChatTemplate    CapabilityID = "chat.template"
	CapabilityLoRAInference   CapabilityID = "lora.inference"
	CapabilityLoRATraining    CapabilityID = "lora.training"
	CapabilityStateBundle     CapabilityID = "state.bundle"
	CapabilityKVSnapshot      CapabilityID = "kv.snapshot"
	CapabilityPromptCache     CapabilityID = "prompt.cache"
	CapabilityKVCachePlanning CapabilityID = "kv.cache.planning"
	CapabilityMemoryPlanning  CapabilityID = "memory.planning"
	CapabilityModelFit        CapabilityID = "model.fit"
	CapabilityBenchmark       CapabilityID = "benchmark"
	CapabilityEvaluation      CapabilityID = "evaluation"
	CapabilityDistillation    CapabilityID = "distillation"
	CapabilityGRPO            CapabilityID = "grpo"
	CapabilityQuantization    CapabilityID = "quantization"
	CapabilityModelMerge      CapabilityID = "model.merge"
	CapabilityProbeEvents     CapabilityID = "probe.events"
	CapabilityAttentionProbe  CapabilityID = "probe.attention"
	CapabilityLogitProbe      CapabilityID = "probe.logits"
)

// Capability describes one backend feature without importing that backend.
type Capability struct {
	ID     CapabilityID      `json:"id"`
	Group  CapabilityGroup   `json:"group,omitempty"`
	Status CapabilityStatus  `json:"status"`
	Detail string            `json:"detail,omitempty"`
	Labels map[string]string `json:"labels,omitempty"`
}

// CapabilityReport is the portable backend/model feature report consumed by
// go-ml, go-ai, and any package that must avoid backend-specific imports.
type CapabilityReport struct {
	Runtime       RuntimeIdentity   `json:"runtime"`
	Model         ModelIdentity     `json:"model,omitempty"`
	Tokenizer     TokenizerIdentity `json:"tokenizer,omitempty"`
	Adapter       AdapterIdentity   `json:"adapter,omitempty"`
	Available     bool              `json:"available"`
	Architectures []string          `json:"architectures,omitempty"`
	Quantizations []string          `json:"quantizations,omitempty"`
	CacheModes    []string          `json:"cache_modes,omitempty"`
	Capabilities  []Capability      `json:"capabilities,omitempty"`
	Labels        map[string]string `json:"labels,omitempty"`
}

// CapabilityReporter is implemented by backends and loaded models that can
// expose their native feature surface without leaking concrete package types.
type CapabilityReporter interface {
	Capabilities() CapabilityReport
}

// RuntimeMemoryLimits is a backend-neutral request/response for runtime memory
// caps. Zero request values mean "leave unchanged"; previous values are filled
// by backends that can report them.
type RuntimeMemoryLimits struct {
	CacheLimitBytes          uint64 `json:"cache_limit_bytes,omitempty"`
	MemoryLimitBytes         uint64 `json:"memory_limit_bytes,omitempty"`
	PreviousCacheLimitBytes  uint64 `json:"previous_cache_limit_bytes,omitempty"`
	PreviousMemoryLimitBytes uint64 `json:"previous_memory_limit_bytes,omitempty"`
}

// RuntimeMemoryLimiter is implemented by native runtimes that expose allocator
// limits without requiring callers to import the concrete runtime package.
type RuntimeMemoryLimiter interface {
	SetRuntimeMemoryLimits(limits RuntimeMemoryLimits) RuntimeMemoryLimits
}

// SetRuntimeMemoryLimits applies memory limits to a registered backend when it
// supports [RuntimeMemoryLimiter]. The boolean is false when the backend is not
// registered or does not support this operation.
func SetRuntimeMemoryLimits(backendName string, limits RuntimeMemoryLimits) (RuntimeMemoryLimits, bool) {
	backend, ok := Get(backendName)
	if !ok {
		return RuntimeMemoryLimits{}, false
	}
	limiter, ok := backend.(RuntimeMemoryLimiter)
	if !ok {
		return RuntimeMemoryLimits{}, false
	}
	return limiter.SetRuntimeMemoryLimits(limits), true
}

// NewCapability creates a single capability entry.
func NewCapability(id CapabilityID, group CapabilityGroup, status CapabilityStatus, detail string) Capability {
	return Capability{ID: id, Group: group, Status: status, Detail: detail}
}

// SupportedCapability creates a capability entry for a stable feature.
func SupportedCapability(id CapabilityID, group CapabilityGroup) Capability {
	return NewCapability(id, group, CapabilityStatusSupported, "")
}

// ExperimentalCapability creates a capability entry for a usable but unstable feature.
func ExperimentalCapability(id CapabilityID, group CapabilityGroup, detail string) Capability {
	return NewCapability(id, group, CapabilityStatusExperimental, detail)
}

// PlannedCapability creates a capability entry for an intentionally exposed
// roadmap item that is not usable yet.
func PlannedCapability(id CapabilityID, group CapabilityGroup, detail string) Capability {
	return NewCapability(id, group, CapabilityStatusPlanned, detail)
}

// UnsupportedCapability creates a capability entry for an unavailable feature.
func UnsupportedCapability(id CapabilityID, group CapabilityGroup, detail string) Capability {
	return NewCapability(id, group, CapabilityStatusUnsupported, detail)
}

// Usable reports whether a capability can be used by callers today.
func (cap Capability) Usable() bool {
	return cap.Status == CapabilityStatusSupported || cap.Status == CapabilityStatusExperimental
}

// Capability returns the first entry with id.
func (report CapabilityReport) Capability(id CapabilityID) (Capability, bool) {
	for _, capability := range report.Capabilities {
		if capability.ID == id {
			return cloneCapability(capability), true
		}
	}
	return Capability{}, false
}

// Supports reports whether id is present and usable.
func (report CapabilityReport) Supports(id CapabilityID) bool {
	capability, ok := report.Capability(id)
	return ok && capability.Usable()
}

// SupportedCapabilityIDs returns stable IDs for all usable capabilities.
func (report CapabilityReport) SupportedCapabilityIDs() []CapabilityID {
	ids := make([]CapabilityID, 0, len(report.Capabilities))
	for _, capability := range report.Capabilities {
		if capability.Usable() {
			ids = append(ids, capability.ID)
		}
	}
	slices.Sort(ids)
	return slices.Compact(ids)
}

// CapabilityIDs returns stable IDs for every reported capability.
func (report CapabilityReport) CapabilityIDs() []CapabilityID {
	ids := make([]CapabilityID, 0, len(report.Capabilities))
	for _, capability := range report.Capabilities {
		ids = append(ids, capability.ID)
	}
	slices.Sort(ids)
	return slices.Compact(ids)
}

// CapabilitiesOf returns an explicit or inferred capability report for value.
func CapabilitiesOf(value any) (CapabilityReport, bool) {
	if value == nil {
		return CapabilityReport{}, false
	}
	if reporter, ok := value.(CapabilityReporter); ok {
		return reporter.Capabilities(), true
	}
	switch typed := value.(type) {
	case Backend:
		return BackendCapabilities(typed), true
	case TextModel:
		return TextModelCapabilities(RuntimeIdentity{}, typed), true
	default:
		return CapabilityReport{}, false
	}
}

// BackendCapabilities infers the minimal report every registered backend can expose.
func BackendCapabilities(backend Backend) CapabilityReport {
	if backend == nil {
		return CapabilityReport{}
	}
	capabilities := []Capability{SupportedCapability(CapabilityModelLoad, CapabilityGroupRuntime)}
	if _, ok := backend.(ModelFitPlanner); ok {
		capabilities = append(capabilities, SupportedCapability(CapabilityModelFit, CapabilityGroupRuntime))
	}
	return CapabilityReport{
		Runtime:      RuntimeIdentity{Backend: backend.Name()},
		Available:    backend.Available(),
		Capabilities: capabilities,
	}
}

// TextModelCapabilities infers a report from optional interfaces implemented by
// a loaded model.
func TextModelCapabilities(runtime RuntimeIdentity, model TextModel) CapabilityReport {
	if model == nil {
		return CapabilityReport{Runtime: runtime}
	}
	info := model.Info()
	report := CapabilityReport{
		Runtime:   runtime,
		Available: true,
		Model: ModelIdentity{
			Architecture: info.Architecture,
			VocabSize:    info.VocabSize,
			NumLayers:    info.NumLayers,
			HiddenSize:   info.HiddenSize,
			QuantBits:    info.QuantBits,
			QuantGroup:   info.QuantGroup,
		},
		Capabilities: []Capability{
			SupportedCapability(CapabilityGenerate, CapabilityGroupModel),
			SupportedCapability(CapabilityChat, CapabilityGroupModel),
			SupportedCapability(CapabilityClassify, CapabilityGroupModel),
			SupportedCapability(CapabilityBatchGenerate, CapabilityGroupModel),
		},
	}
	if tokenizer, ok := model.(TokenizerModel); ok {
		report.Capabilities = append(report.Capabilities,
			SupportedCapability(CapabilityTokenizer, CapabilityGroupModel),
			SupportedCapability(CapabilityChatTemplate, CapabilityGroupModel),
		)
		_ = tokenizer
	}
	if adapter, ok := model.(AdapterModel); ok {
		report.Adapter = adapter.ActiveAdapter()
		report.Capabilities = append(report.Capabilities, SupportedCapability(CapabilityLoRAInference, CapabilityGroupModel))
	}
	if _, ok := model.(StatefulModel); ok {
		report.Capabilities = append(report.Capabilities, SupportedCapability(CapabilityStateBundle, CapabilityGroupRuntime))
	}
	if _, ok := model.(ProbeableModel); ok {
		report.Capabilities = append(report.Capabilities, SupportedCapability(CapabilityProbeEvents, CapabilityGroupProbe))
	}
	if _, ok := model.(AttentionInspector); ok {
		report.Capabilities = append(report.Capabilities, SupportedCapability(CapabilityAttentionProbe, CapabilityGroupProbe))
	}
	if _, ok := model.(BenchableModel); ok {
		report.Capabilities = append(report.Capabilities, SupportedCapability(CapabilityBenchmark, CapabilityGroupRuntime))
	}
	if _, ok := model.(Evaluator); ok {
		report.Capabilities = append(report.Capabilities, SupportedCapability(CapabilityEvaluation, CapabilityGroupRuntime))
	}
	if _, ok := model.(SFTTrainer); ok {
		report.Capabilities = append(report.Capabilities, SupportedCapability(CapabilityLoRATraining, CapabilityGroupTraining))
	}
	if _, ok := model.(DistillTrainer); ok {
		report.Capabilities = append(report.Capabilities, SupportedCapability(CapabilityDistillation, CapabilityGroupTraining))
	}
	if _, ok := model.(GRPOTrainer); ok {
		report.Capabilities = append(report.Capabilities, SupportedCapability(CapabilityGRPO, CapabilityGroupTraining))
	}
	if _, ok := model.(ModelFitPlanner); ok {
		report.Capabilities = append(report.Capabilities, SupportedCapability(CapabilityModelFit, CapabilityGroupRuntime))
	}
	return report
}

func cloneCapability(capability Capability) Capability {
	capability.Labels = maps.Clone(capability.Labels)
	return capability
}

// TokenizerModel exposes native tokenisation and chat-template handling.
type TokenizerModel interface {
	Encode(text string) []int32
	Decode(ids []int32) string
	ApplyChatTemplate(messages []Message) (string, error)
}

// AdapterModel exposes LoRA adapter lifecycle operations for inference.
type AdapterModel interface {
	LoadAdapter(path string) (AdapterIdentity, error)
	UnloadAdapter() error
	ActiveAdapter() AdapterIdentity
}

// StatefulModel exposes portable model-state capture and restore.
type StatefulModel interface {
	CaptureState(ctx context.Context, prompt string, opts ...GenerateOption) (*StateBundle, error)
	RestoreState(ctx context.Context, bundle *StateBundle) error
}

// ProbeableModel accepts a typed probe sink for inference or training events.
type ProbeableModel interface {
	SetProbeSink(sink ProbeSink)
}

// BenchableModel runs local benchmark workloads.
type BenchableModel interface {
	Benchmark(ctx context.Context, cfg BenchConfig) (*BenchReport, error)
}

// ModelFitPlanner estimates whether a model fits a memory budget.
type ModelFitPlanner interface {
	PlanModelFit(ctx context.Context, model ModelIdentity, memoryBytes uint64) (*ModelFitReport, error)
}

// SFTTrainer trains a model or adapter with supervised fine tuning.
type SFTTrainer interface {
	TrainSFT(ctx context.Context, dataset DatasetStream, cfg TrainingConfig) (*TrainingResult, error)
}

// DistillTrainer trains a student model from teacher outputs.
type DistillTrainer interface {
	Distill(ctx context.Context, dataset DatasetStream, cfg DistillConfig) (*TrainingResult, error)
}

// GRPOTrainer trains grouped reasoning rollouts.
type GRPOTrainer interface {
	TrainGRPO(ctx context.Context, dataset DatasetStream, cfg GRPOConfig) (*TrainingResult, error)
}
