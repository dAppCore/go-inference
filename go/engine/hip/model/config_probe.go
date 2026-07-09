// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"strconv"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference/engine/hip/profile"
)

const (
	ConfigProbeContract            = "rocm-model-config-probe-v1"
	ArchitectureResolutionContract = "rocm-architecture-resolution-v1"
)

// ConfigProbeInput is the model-owned subset of config.json metadata needed to
// resolve the architecture, loader route, runtime contracts, and config-composed
// sequence-mixer plan before loading weights.
type ConfigProbeInput struct {
	ModelType           string
	TextTowerModelType  string
	Architectures       []string
	TextArchitectures   []string
	LayerTypes          []string
	TextLayerTypes      []string
	NumHiddenLayers     int
	NumLayers           int
	TextNumHiddenLayers int
	TextNumLayers       int
}

// ConfigProbe is the model-owned pre-load dispatch contract. It mirrors
// go-mlx's config probe plus loader lookup path while keeping ROCm root API
// wrappers out of the planning core.
type ConfigProbe struct {
	Contract               string                         `json:"contract,omitempty"`
	ModelType              string                         `json:"model_type,omitempty"`
	TextTowerModelType     string                         `json:"text_tower_model_type,omitempty"`
	Architectures          []string                       `json:"architectures,omitempty"`
	ArchitectureResolution profile.ArchitectureResolution `json:"architecture_resolution"`
	LoaderRoute            LoaderRoute                    `json:"loader_route"`
	RuntimeContractRoute   RuntimeContractRoute           `json:"runtime_contract_route"`
	SequenceMixer          SequenceMixerConfigProbe       `json:"sequence_mixer"`
	Registered             bool                           `json:"registered,omitempty"`
	AttachedOnly           bool                           `json:"attached_only,omitempty"`
	Standalone             bool                           `json:"standalone,omitempty"`
	Staged                 bool                           `json:"staged,omitempty"`
	MetadataOnly           bool                           `json:"metadata_only,omitempty"`
	TextGenerate           bool                           `json:"text_generate,omitempty"`
	Labels                 map[string]string              `json:"labels,omitempty"`
}

func (probe ConfigProbe) Clone() ConfigProbe {
	probe.Architectures = append([]string(nil), probe.Architectures...)
	probe.ArchitectureResolution = probe.ArchitectureResolution.Clone()
	probe.LoaderRoute = probe.LoaderRoute.Clone()
	probe.RuntimeContractRoute = probe.RuntimeContractRoute.Clone()
	probe.SequenceMixer = probe.SequenceMixer.Clone()
	probe.Labels = cloneStringMap(probe.Labels)
	return probe
}

func ProbeConfig(input ConfigProbeInput) ConfigProbe {
	architectures := append([]string(nil), input.Architectures...)
	architectures = append(architectures, input.TextArchitectures...)
	resolution := profile.ResolveArchitecture(input.ModelType, input.TextTowerModelType, architectures)
	probe := ConfigProbe{
		Contract:               ConfigProbeContract,
		ModelType:              strings.TrimSpace(input.ModelType),
		TextTowerModelType:     strings.TrimSpace(input.TextTowerModelType),
		Architectures:          profile.CleanArchitectureSignals(architectures),
		ArchitectureResolution: resolution,
	}
	if resolution.Matched() {
		if route, ok := LoaderRouteForArchitecture(resolution.Architecture); ok {
			probe.LoaderRoute = route
			probe.Registered = route.Registered
			probe.AttachedOnly = route.AttachedOnly
			probe.Standalone = route.Standalone
			probe.Staged = route.Staged
			probe.MetadataOnly = route.MetadataOnly
			probe.TextGenerate = route.TextGenerate
		}
		if route, ok := RuntimeContractRouteForArchitecture(resolution.Architecture); ok {
			probe.RuntimeContractRoute = route
		}
	}
	probe.SequenceMixer = ProbeSequenceMixerConfig(sequenceMixerConfigInput(input))
	probe.Labels = ConfigProbeLabels(probe)
	return probe.Clone()
}

func ConfigProbeLabels(probe ConfigProbe) map[string]string {
	labels := map[string]string{
		"engine_config_probe_contract":       firstNonEmpty(probe.Contract, ConfigProbeContract),
		"engine_config_loader_registered":    strconv.FormatBool(probe.Registered),
		"engine_config_attached_only":        strconv.FormatBool(probe.AttachedOnly),
		"engine_config_standalone":           strconv.FormatBool(probe.Standalone),
		"engine_config_staged":               strconv.FormatBool(probe.Staged),
		"engine_config_metadata_only":        strconv.FormatBool(probe.MetadataOnly),
		"engine_config_text_generate":        strconv.FormatBool(probe.TextGenerate),
		"engine_config_composed":             strconv.FormatBool(probe.SequenceMixer.Composed),
		"engine_config_runtime_contract":     strconv.FormatBool(probe.RuntimeContractRoute.Matched()),
		"sequence_mixer_registry_contract":   SequenceMixerRegistryContract,
		"sequence_mixer_registry_kinds":      core.Join(",", SequenceMixerFamilyKinds()...),
		"sequence_mixer_cache_factory":       SequenceMixerCacheFactoryContract,
		"sequence_mixer_cache_factory_modes": core.Join(",", DefaultSequenceMixerCacheFactoryModes()...),
	}
	if probe.ModelType != "" {
		labels["engine_config_model_type"] = probe.ModelType
	}
	if probe.TextTowerModelType != "" {
		labels["engine_config_text_tower_model_type"] = probe.TextTowerModelType
	}
	if len(probe.Architectures) > 0 {
		labels["engine_config_architecture_count"] = strconv.Itoa(len(probe.Architectures))
	}
	if probe.ArchitectureResolution.Matched() {
		labels["engine_config_architecture_resolved"] = probe.ArchitectureResolution.Architecture
		labels["engine_config_architecture_source"] = probe.ArchitectureResolution.Source
		labels["architecture_resolution_contract"] = ArchitectureResolutionContract
		labels["architecture_resolved"] = probe.ArchitectureResolution.Architecture
		labels["architecture_resolution_source"] = probe.ArchitectureResolution.Source
	}
	if probe.LoaderRoute.Matched() {
		labels["engine_config_loader"] = probe.LoaderRoute.Loader
		labels["engine_config_loader_runtime"] = probe.LoaderRoute.Runtime
		labels["engine_config_loader_status"] = probe.LoaderRoute.Status
		labels["engine_loader_contract"] = probe.LoaderRoute.Contract
	}
	if probe.RuntimeContractRoute.Matched() {
		labels["engine_config_runtime_contract_count"] = strconv.Itoa(len(probe.RuntimeContractRoute.ContractIDs))
		if len(probe.RuntimeContractRoute.ContractIDs) > 0 {
			labels["engine_config_runtime_contract_ids"] = RuntimeContractIDsCSV(probe.RuntimeContractRoute.ContractIDs)
		}
		for key, value := range RuntimeContractRouteLabels(probe.RuntimeContractRoute) {
			if value != "" {
				labels[key] = value
			}
		}
	}
	if probe.SequenceMixer.LayerSource != "" {
		labels["sequence_mixer_layer_types_source"] = probe.SequenceMixer.LayerSource
	}
	if len(probe.SequenceMixer.LayerTypes) > 0 {
		labels["attention_layer_types"] = core.Join(",", probe.SequenceMixer.LayerTypes...)
		labels["sequence_mixer_declared_kinds"] = core.Join(",", SequenceMixerUniqueKinds(probe.SequenceMixer.LayerTypes)...)
	}
	if probe.SequenceMixer.PlanStatus != "" {
		labels["sequence_mixer_config_plan_status"] = probe.SequenceMixer.PlanStatus
	}
	if probe.SequenceMixer.PlanError != "" {
		labels["sequence_mixer_config_plan_error"] = probe.SequenceMixer.PlanError
	}
	if len(probe.SequenceMixer.Layers) > 0 {
		labels["sequence_mixer_config_plan_layers"] = strconv.Itoa(len(probe.SequenceMixer.Layers))
		labels["sequence_mixer_config_plan_entries"] = SequenceMixerLoadPlanCSV(probe.SequenceMixer.Layers)
	}
	if len(probe.SequenceMixer.Cache.Layers) > 0 {
		labels["sequence_mixer_cache_plan_contract"] = probe.SequenceMixer.Cache.Contract
		labels["sequence_mixer_cache_plan_layers"] = strconv.Itoa(len(probe.SequenceMixer.Cache.Layers))
		labels["sequence_mixer_cache_plan_entries"] = SequenceMixerCachePlanCSV(probe.SequenceMixer.Cache.Layers)
	}
	return labels
}

func sequenceMixerConfigInput(input ConfigProbeInput) SequenceMixerConfigInput {
	return SequenceMixerConfigInput{
		ModelType:           input.ModelType,
		TextModelType:       input.TextTowerModelType,
		LayerTypes:          append([]string(nil), input.LayerTypes...),
		TextLayerTypes:      append([]string(nil), input.TextLayerTypes...),
		NumHiddenLayers:     input.NumHiddenLayers,
		NumLayers:           input.NumLayers,
		TextNumHiddenLayers: input.TextNumHiddenLayers,
		TextNumLayers:       input.TextNumLayers,
	}
}
