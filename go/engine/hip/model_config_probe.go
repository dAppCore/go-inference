// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"encoding/json"
	"os"

	core "dappco.re/go"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const ROCmModelConfigProbeContract = rocmmodel.ConfigProbeContract

// ROCmModelConfigProbe is the pre-load route contract for raw config.json
// metadata. It is the ROCm counterpart to go-mlx's probeModelType +
// model-loader lookup path, with sequence-mixer catalogue metadata included for
// config-composed and hybrid checkpoints.
type ROCmModelConfigProbe struct {
	Contract                 string                        `json:"contract,omitempty"`
	ModelType                string                        `json:"model_type,omitempty"`
	TextTowerModelType       string                        `json:"text_tower_model_type,omitempty"`
	Architectures            []string                      `json:"architectures,omitempty"`
	ArchitectureResolution   ROCmArchitectureResolution    `json:"architecture_resolution,omitempty"`
	LoaderRoute              ROCmModelLoaderRoute          `json:"loader_route,omitempty"`
	RuntimeContractRoute     ROCmModelRuntimeContractRoute `json:"runtime_contract_route,omitempty"`
	SequenceMixerLayers      []SequenceMixerLayerPlan      `json:"sequence_mixer_layers,omitempty"`
	SequenceMixerCache       SequenceMixerCachePlan        `json:"sequence_mixer_cache,omitempty"`
	SequenceMixerLayerTypes  []string                      `json:"sequence_mixer_layer_types,omitempty"`
	SequenceMixerLayerSource string                        `json:"sequence_mixer_layer_source,omitempty"`
	SequenceMixerPlanStatus  string                        `json:"sequence_mixer_plan_status,omitempty"`
	SequenceMixerPlanError   string                        `json:"sequence_mixer_plan_error,omitempty"`
	Registered               bool                          `json:"registered,omitempty"`
	AttachedOnly             bool                          `json:"attached_only,omitempty"`
	Standalone               bool                          `json:"standalone,omitempty"`
	Staged                   bool                          `json:"staged,omitempty"`
	MetadataOnly             bool                          `json:"metadata_only,omitempty"`
	TextGenerate             bool                          `json:"text_generate,omitempty"`
	ConfigComposed           bool                          `json:"config_composed,omitempty"`
	Labels                   map[string]string             `json:"labels,omitempty"`
}

func (probe ROCmModelConfigProbe) clone() ROCmModelConfigProbe {
	probe.Architectures = append([]string(nil), probe.Architectures...)
	probe.ArchitectureResolution = probe.ArchitectureResolution.clone()
	probe.LoaderRoute = probe.LoaderRoute.Clone()
	probe.RuntimeContractRoute = probe.RuntimeContractRoute.Clone()
	probe.SequenceMixerLayers = cloneSequenceMixerLayerPlans(probe.SequenceMixerLayers)
	probe.SequenceMixerCache = cloneSequenceMixerCachePlan(probe.SequenceMixerCache)
	probe.SequenceMixerLayerTypes = append([]string(nil), probe.SequenceMixerLayerTypes...)
	probe.Labels = cloneStringMap(probe.Labels)
	return probe
}

func ProbeROCmModelConfigFile(path string) (ROCmModelConfigProbe, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ROCmModelConfigProbe{}, core.E("rocm.ModelConfigProbe", "read config", err)
	}
	return ProbeROCmModelConfig(data)
}

func ProbeROCmModelConfig(data []byte) (ROCmModelConfigProbe, error) {
	var cfg rocmModelPackConfigProbe
	if err := json.Unmarshal(data, &cfg); err != nil {
		return ROCmModelConfigProbe{}, core.E("rocm.ModelConfigProbe", "parse config", err)
	}
	return probeROCmModelConfig(cfg), nil
}

func probeROCmModelConfig(cfg rocmModelPackConfigProbe) ROCmModelConfigProbe {
	return rocmModelConfigProbeFromModel(rocmmodel.ProbeConfig(rocmModelConfigProbeInput(cfg)))
}

func rocmModelConfigProbeInput(cfg rocmModelPackConfigProbe) rocmmodel.ConfigProbeInput {
	return rocmmodel.ConfigProbeInput{
		ModelType:           cfg.ModelType,
		TextTowerModelType:  cfg.TextConfig.ModelType,
		Architectures:       append([]string(nil), cfg.Architectures...),
		TextArchitectures:   append([]string(nil), cfg.TextConfig.Architectures...),
		LayerTypes:          append([]string(nil), cfg.LayerTypes...),
		TextLayerTypes:      append([]string(nil), cfg.TextConfig.LayerTypes...),
		NumHiddenLayers:     cfg.NumHiddenLayers,
		NumLayers:           cfg.NumLayers,
		TextNumHiddenLayers: cfg.TextConfig.NumHiddenLayers,
		TextNumLayers:       cfg.TextConfig.NumLayers,
	}
}

func rocmSequenceMixerConfigInput(cfg rocmModelPackConfigProbe) rocmmodel.SequenceMixerConfigInput {
	return rocmmodel.SequenceMixerConfigInput{
		ModelType:           cfg.ModelType,
		TextModelType:       cfg.TextConfig.ModelType,
		LayerTypes:          append([]string(nil), cfg.LayerTypes...),
		TextLayerTypes:      append([]string(nil), cfg.TextConfig.LayerTypes...),
		NumHiddenLayers:     cfg.NumHiddenLayers,
		NumLayers:           cfg.NumLayers,
		TextNumHiddenLayers: cfg.TextConfig.NumHiddenLayers,
		TextNumLayers:       cfg.TextConfig.NumLayers,
	}
}

func rocmModelConfigProbeFromModel(probe rocmmodel.ConfigProbe) ROCmModelConfigProbe {
	return ROCmModelConfigProbe{
		Contract:                 probe.Contract,
		ModelType:                probe.ModelType,
		TextTowerModelType:       probe.TextTowerModelType,
		Architectures:            append([]string(nil), probe.Architectures...),
		ArchitectureResolution:   rocmArchitectureResolutionFromProfile(probe.ArchitectureResolution),
		LoaderRoute:              rocmModelLoaderRouteFromModel(probe.LoaderRoute),
		RuntimeContractRoute:     probe.RuntimeContractRoute.Clone(),
		SequenceMixerLayers:      probe.SequenceMixer.Layers,
		SequenceMixerCache:       probe.SequenceMixer.Cache,
		SequenceMixerLayerTypes:  append([]string(nil), probe.SequenceMixer.LayerTypes...),
		SequenceMixerLayerSource: probe.SequenceMixer.LayerSource,
		SequenceMixerPlanStatus:  probe.SequenceMixer.PlanStatus,
		SequenceMixerPlanError:   probe.SequenceMixer.PlanError,
		Registered:               probe.Registered,
		AttachedOnly:             probe.AttachedOnly,
		Standalone:               probe.Standalone,
		Staged:                   probe.Staged,
		MetadataOnly:             probe.MetadataOnly,
		TextGenerate:             probe.TextGenerate,
		ConfigComposed:           probe.SequenceMixer.Composed,
		Labels:                   cloneStringMap(probe.Labels),
	}.clone()
}

func rocmApplyModelConfigProbeLabels(labels map[string]string, cfg rocmModelPackConfigProbe) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	probe := probeROCmModelConfig(cfg)
	for key, value := range probe.Labels {
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}
