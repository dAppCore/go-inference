// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"context"
	"maps"
	"slices"

	core "dappco.re/go"
)

// ModelComponent identifies a logical part of a model pack that can be kept
// local, moved to a remote worker, or indexed for research queries.
type ModelComponent string

const (
	ModelComponentManifest   ModelComponent = "manifest"
	ModelComponentTokenizer  ModelComponent = "tokenizer"
	ModelComponentLabels     ModelComponent = "labels"
	ModelComponentEmbeddings ModelComponent = "embeddings"
	ModelComponentNorms      ModelComponent = "norms"
	ModelComponentAttention  ModelComponent = "attention"
	ModelComponentFFN        ModelComponent = "ffn"
	ModelComponentGate       ModelComponent = "gate"
	ModelComponentDownMeta   ModelComponent = "down_meta"
	ModelComponentRouter     ModelComponent = "router"
	ModelComponentExperts    ModelComponent = "experts"
	ModelComponentLMHead     ModelComponent = "lm_head"
)

// ModelExtractLevel names the amount of model structure required for a slice
// or research index.
type ModelExtractLevel string

const (
	ModelExtractLevelCustom    ModelExtractLevel = "custom"
	ModelExtractLevelBrowse    ModelExtractLevel = "browse"
	ModelExtractLevelAttention ModelExtractLevel = "attention"
	ModelExtractLevelInference ModelExtractLevel = "inference"
	ModelExtractLevelAll       ModelExtractLevel = "all"
)

// ModelSlicePreset names a repeatable model split topology. The presets mirror
// LarQL's research layout without forcing callers to use LarQL's file format.
type ModelSlicePreset string

const (
	ModelSlicePresetCustom       ModelSlicePreset = "custom"
	ModelSlicePresetFull         ModelSlicePreset = "full"
	ModelSlicePresetClient       ModelSlicePreset = "client"
	ModelSlicePresetAttention    ModelSlicePreset = "attention"
	ModelSlicePresetAttn         ModelSlicePreset = ModelSlicePresetAttention
	ModelSlicePresetEmbed        ModelSlicePreset = "embed"
	ModelSlicePresetServer       ModelSlicePreset = "server"
	ModelSlicePresetBrowse       ModelSlicePreset = "browse"
	ModelSlicePresetRouter       ModelSlicePreset = "router"
	ModelSlicePresetExpertServer ModelSlicePreset = "expert_server"
)

// ModelSliceRequest asks a backend or planner for a portable split plan.
type ModelSliceRequest struct {
	Preset     ModelSlicePreset  `json:"preset,omitempty"`
	Components []ModelComponent  `json:"components,omitempty"`
	Model      ModelIdentity     `json:"model"`
	Adapter    AdapterIdentity   `json:"adapter"`
	OutputPath string            `json:"output_path,omitempty"`
	Labels     map[string]string `json:"labels,omitempty"`
}

// ModelSlicePlan is the backend-neutral result of slicing a model into logical
// components. Actual backends decide how each component maps to tensors/files.
type ModelSlicePlan struct {
	Preset             ModelSlicePreset  `json:"preset,omitempty"`
	ExtractLevel       ModelExtractLevel `json:"extract_level,omitempty"`
	Components         []ModelComponent  `json:"components,omitempty"`
	SourcePath         string            `json:"source_path,omitempty"`
	OutputPath         string            `json:"output_path,omitempty"`
	Model              ModelIdentity     `json:"model"`
	Adapter            AdapterIdentity   `json:"adapter"`
	AttentionLocal     bool              `json:"attention_local,omitempty"`
	FFNRemoteCandidate bool              `json:"ffn_remote_candidate,omitempty"`
	Notes              []string          `json:"notes,omitempty"`
	Labels             map[string]string `json:"labels,omitempty"`
}

// HasComponent reports whether plan contains component.
func (plan ModelSlicePlan) HasComponent(component ModelComponent) bool {
	return slices.Contains(plan.Components, component)
}

// ModelSlicePlanner is implemented by runtimes that can cheaply plan a model
// slice without copying tensors or loading the full model.
type ModelSlicePlanner interface {
	PlanModelSlice(context.Context, ModelSliceRequest) (*ModelSlicePlan, error)
}

// ModelSlicer is implemented by runtimes that can materialise a model slice.
type ModelSlicer interface {
	SliceModel(context.Context, ModelSliceRequest) (*ModelSlicePlan, error)
}

// SplitEndpointRole names the work performed by a remote split-inference
// endpoint.
type SplitEndpointRole string

const (
	SplitEndpointRoleEmbeddings SplitEndpointRole = "embeddings"
	SplitEndpointRoleAttention  SplitEndpointRole = "attention"
	SplitEndpointRoleFFN        SplitEndpointRole = "ffn"
	SplitEndpointRoleRouter     SplitEndpointRole = "router"
	SplitEndpointRoleExpert     SplitEndpointRole = "expert"
)

// SplitInferenceMode names the high-level execution topology.
type SplitInferenceMode string

const (
	SplitInferenceModeLocal          SplitInferenceMode = "local"
	SplitInferenceModeRemoteFFN      SplitInferenceMode = "remote_ffn"
	SplitInferenceModeRemoteEmbedFFN SplitInferenceMode = "remote_embed_ffn"
	SplitInferenceModeRemoteExperts  SplitInferenceMode = "remote_experts"
)

// SplitEndpoint identifies a remote service that owns part of a model.
type SplitEndpoint struct {
	ID          string            `json:"id,omitempty"`
	Role        SplitEndpointRole `json:"role,omitempty"`
	URL         string            `json:"url,omitempty"`
	LayerStart  int               `json:"layer_start,omitempty"`
	LayerEnd    int               `json:"layer_end,omitempty"`
	ExpertStart int               `json:"expert_start,omitempty"`
	ExpertEnd   int               `json:"expert_end,omitempty"`
	WeightShard string            `json:"weight_shard,omitempty"`
	Labels      map[string]string `json:"labels,omitempty"`
}

// SplitInferencePlan describes how a loaded model should place attention,
// embeddings, and FFN/expert work across local and remote workers.
type SplitInferencePlan struct {
	Mode       SplitInferenceMode `json:"mode,omitempty"`
	Model      ModelIdentity      `json:"model"`
	Adapter    AdapterIdentity    `json:"adapter"`
	LocalSlice ModelSlicePlan     `json:"local_slice"`
	Endpoints  []SplitEndpoint    `json:"endpoints,omitempty"`
	Labels     map[string]string  `json:"labels,omitempty"`
}

// SplitPlanner is implemented by runtimes that can turn local hardware facts
// and remote endpoints into a concrete split-inference plan.
type SplitPlanner interface {
	PlanSplitInference(context.Context, SplitInferenceRequest) (*SplitInferencePlan, error)
}

// SplitInferenceRequest asks a backend to plan a split-inference topology.
type SplitInferenceRequest struct {
	Model       ModelIdentity      `json:"model"`
	Adapter     AdapterIdentity    `json:"adapter"`
	LocalPreset ModelSlicePreset   `json:"local_preset,omitempty"`
	Mode        SplitInferenceMode `json:"mode,omitempty"`
	Endpoints   []SplitEndpoint    `json:"endpoints,omitempty"`
	Labels      map[string]string  `json:"labels,omitempty"`
}

// PlanModelSlice expands a slice preset into portable model components.
func PlanModelSlice(req ModelSliceRequest) (ModelSlicePlan, error) {
	preset := req.Preset
	if preset == "" {
		if len(req.Components) > 0 {
			preset = ModelSlicePresetCustom
		} else {
			preset = ModelSlicePresetFull
		}
	}

	components, level, err := modelSlicePresetComponents(preset)
	if err != nil {
		return ModelSlicePlan{}, err
	}
	if preset == ModelSlicePresetCustom {
		components = compactModelComponents(req.Components)
		if len(components) == 0 {
			return ModelSlicePlan{}, core.NewError("inference: custom model slice requires at least one component")
		}
		level = ModelExtractLevelCustom
	}

	plan := ModelSlicePlan{
		Preset:             preset,
		ExtractLevel:       level,
		Components:         components,
		SourcePath:         req.Model.Path,
		OutputPath:         req.OutputPath,
		Model:              req.Model,
		Adapter:            req.Adapter,
		AttentionLocal:     slices.Contains(components, ModelComponentAttention),
		FFNRemoteCandidate: slices.Contains(components, ModelComponentAttention) && !slices.Contains(components, ModelComponentFFN),
		Labels:             maps.Clone(req.Labels),
	}
	return plan, nil
}

// ValidateSplitInferencePlan checks that a split topology is structurally
// usable before a backend spends time loading weights.
func ValidateSplitInferencePlan(plan SplitInferencePlan) error {
	mode := plan.Mode
	if mode == "" {
		mode = SplitInferenceModeLocal
	}
	switch mode {
	case SplitInferenceModeLocal:
		return nil
	case SplitInferenceModeRemoteFFN:
		if !plan.LocalSlice.HasComponent(ModelComponentAttention) {
			return core.NewError("inference: remote_ffn split requires local attention")
		}
		if !splitPlanHasEndpointRole(plan.Endpoints, SplitEndpointRoleFFN) {
			return core.NewError("inference: remote_ffn split requires an ffn endpoint")
		}
	case SplitInferenceModeRemoteEmbedFFN:
		if !plan.LocalSlice.HasComponent(ModelComponentAttention) {
			return core.NewError("inference: remote_embed_ffn split requires local attention")
		}
		if !splitPlanHasEndpointRole(plan.Endpoints, SplitEndpointRoleEmbeddings) {
			return core.NewError("inference: remote_embed_ffn split requires an embeddings endpoint")
		}
		if !splitPlanHasEndpointRole(plan.Endpoints, SplitEndpointRoleFFN) {
			return core.NewError("inference: remote_embed_ffn split requires an ffn endpoint")
		}
	case SplitInferenceModeRemoteExperts:
		if !plan.LocalSlice.HasComponent(ModelComponentAttention) {
			return core.NewError("inference: remote_experts split requires local attention")
		}
		if !splitPlanHasEndpointRole(plan.Endpoints, SplitEndpointRoleExpert) {
			return core.NewError("inference: remote_experts split requires an expert endpoint")
		}
	default:
		return core.Errorf("inference: unknown split inference mode %q", mode)
	}
	if err := validateSplitEndpoints(plan.Endpoints); err != nil {
		return err
	}
	return nil
}

func modelSlicePresetComponents(preset ModelSlicePreset) ([]ModelComponent, ModelExtractLevel, error) {
	switch preset {
	case ModelSlicePresetCustom:
		return nil, ModelExtractLevelCustom, nil
	case ModelSlicePresetFull:
		return []ModelComponent{
			ModelComponentManifest,
			ModelComponentTokenizer,
			ModelComponentLabels,
			ModelComponentEmbeddings,
			ModelComponentNorms,
			ModelComponentAttention,
			ModelComponentFFN,
			ModelComponentGate,
			ModelComponentDownMeta,
			ModelComponentRouter,
			ModelComponentExperts,
			ModelComponentLMHead,
		}, ModelExtractLevelAll, nil
	case ModelSlicePresetClient:
		return []ModelComponent{
			ModelComponentManifest,
			ModelComponentTokenizer,
			ModelComponentLabels,
			ModelComponentEmbeddings,
			ModelComponentNorms,
			ModelComponentAttention,
			ModelComponentLMHead,
		}, ModelExtractLevelAttention, nil
	case ModelSlicePresetAttention:
		return []ModelComponent{
			ModelComponentManifest,
			ModelComponentNorms,
			ModelComponentAttention,
			ModelComponentLabels,
		}, ModelExtractLevelAttention, nil
	case ModelSlicePresetEmbed:
		return []ModelComponent{
			ModelComponentManifest,
			ModelComponentTokenizer,
			ModelComponentLabels,
			ModelComponentEmbeddings,
		}, ModelExtractLevelBrowse, nil
	case ModelSlicePresetServer:
		return []ModelComponent{
			ModelComponentManifest,
			ModelComponentTokenizer,
			ModelComponentLabels,
			ModelComponentEmbeddings,
			ModelComponentNorms,
			ModelComponentFFN,
			ModelComponentGate,
			ModelComponentDownMeta,
			ModelComponentRouter,
			ModelComponentExperts,
			ModelComponentLMHead,
		}, ModelExtractLevelInference, nil
	case ModelSlicePresetBrowse:
		return []ModelComponent{
			ModelComponentManifest,
			ModelComponentTokenizer,
			ModelComponentLabels,
			ModelComponentEmbeddings,
			ModelComponentGate,
			ModelComponentDownMeta,
			ModelComponentRouter,
		}, ModelExtractLevelBrowse, nil
	case ModelSlicePresetRouter:
		return []ModelComponent{
			ModelComponentManifest,
			ModelComponentTokenizer,
			ModelComponentLabels,
			ModelComponentRouter,
		}, ModelExtractLevelBrowse, nil
	case ModelSlicePresetExpertServer:
		return []ModelComponent{
			ModelComponentManifest,
			ModelComponentNorms,
			ModelComponentFFN,
			ModelComponentRouter,
			ModelComponentExperts,
		}, ModelExtractLevelInference, nil
	default:
		return nil, "", core.Errorf("inference: unknown slice preset %q", preset)
	}
}

func compactModelComponents(components []ModelComponent) []ModelComponent {
	if len(components) == 0 {
		return nil
	}
	seen := map[ModelComponent]bool{}
	compacted := make([]ModelComponent, 0, len(components))
	for _, component := range components {
		if component == "" || seen[component] {
			continue
		}
		seen[component] = true
		compacted = append(compacted, component)
	}
	return compacted
}

func splitPlanHasEndpointRole(endpoints []SplitEndpoint, role SplitEndpointRole) bool {
	for _, endpoint := range endpoints {
		if endpoint.Role == role {
			return true
		}
	}
	return false
}

func validateSplitEndpoints(endpoints []SplitEndpoint) error {
	for _, endpoint := range endpoints {
		if endpoint.Role == "" {
			return core.NewError("inference: split endpoint requires a role")
		}
		if endpoint.ID == "" && endpoint.URL == "" {
			return core.NewError("inference: split endpoint requires an id or url")
		}
		if endpoint.LayerEnd > 0 && endpoint.LayerStart > endpoint.LayerEnd {
			return core.NewError("inference: split endpoint layer range is invalid")
		}
		if endpoint.ExpertEnd > 0 && endpoint.ExpertStart > endpoint.ExpertEnd {
			return core.NewError("inference: split endpoint expert range is invalid")
		}
	}
	return nil
}
