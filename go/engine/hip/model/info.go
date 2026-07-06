// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"strconv"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/profile"
)

const ModelInfoReporterContract = "rocm-model-info-reporter-v1"

// ModelInfoReporter mirrors go-mlx's model-owned metadata capability in ROCm
// form. Family packages can implement it without extending a root type switch.
type ModelInfoReporter interface {
	FillModelInfo(*inference.ModelInfo)
}

type ModelInfoRequest struct {
	Path      string
	ModelType string
	Info      inference.ModelInfo
	Identity  inference.ModelIdentity
	Labels    map[string]string
	Reporter  ModelInfoReporter
}

type ModelInfoReport struct {
	Contract     string                  `json:"contract,omitempty"`
	Source       string                  `json:"source,omitempty"`
	Path         string                  `json:"path,omitempty"`
	Architecture string                  `json:"architecture,omitempty"`
	Info         inference.ModelInfo     `json:"info"`
	Identity     inference.ModelIdentity `json:"identity"`
	Labels       map[string]string       `json:"labels,omitempty"`
}

func (report ModelInfoReport) Matched() bool {
	return report.Contract != "" && report.Architecture != ""
}

func (report ModelInfoReport) Clone() ModelInfoReport {
	report.Identity.Labels = cloneStringMap(report.Identity.Labels)
	report.Labels = cloneStringMap(report.Labels)
	return report
}

func ResolveModelInfo(req ModelInfoRequest) ModelInfoReport {
	info := req.Info
	source := "loaded_info"
	if info.Architecture == "" {
		info.Architecture = req.ModelType
	}
	if req.Reporter != nil {
		req.Reporter.FillModelInfo(&info)
		source = "model_info_reporter"
	}

	identity := cloneModelIdentity(req.Identity)
	if identity.Path == "" {
		identity.Path = req.Path
	}
	labels := mergeInfoLabels(req.Labels, identity.Labels)
	identity.Labels = labels

	info = mergeInfoWithIdentity(info, identity, req.ModelType)
	architecture := firstNonEmpty(
		labels["engine_architecture_resolved"],
		labels["architecture_resolved"],
		info.Architecture,
		identity.Architecture,
		req.ModelType,
	)
	architecture = profile.ArchitectureID(architecture)
	info.Architecture = architecture
	identity.Architecture = architecture
	identity = mergeIdentityWithInfo(identity, info)
	if identity.Path == "" {
		identity.Path = req.Path
	}
	if identity.QuantType == "" {
		identity.QuantType = firstNonEmpty(labels["quant_type"], labels["gemma4_quant_mode"])
	}

	reportLabels := modelInfoLabels(labels, source, info, identity)
	identity.Labels = reportLabels
	return ModelInfoReport{
		Contract:     ModelInfoReporterContract,
		Source:       source,
		Path:         identity.Path,
		Architecture: architecture,
		Info:         info,
		Identity:     identity,
		Labels:       reportLabels,
	}.Clone()
}

func ModelInfoFromIdentity(path string, identity inference.ModelIdentity) inference.ModelInfo {
	if identity.Path == "" {
		identity.Path = path
	}
	return ResolveModelInfo(ModelInfoRequest{Path: path, Identity: identity}).Info
}

func ModelInfoIdentity(path string, info inference.ModelInfo, labels map[string]string) inference.ModelIdentity {
	report := ResolveModelInfo(ModelInfoRequest{
		Path:   path,
		Info:   info,
		Labels: labels,
	})
	return report.Identity
}

func mergeInfoWithIdentity(info inference.ModelInfo, identity inference.ModelIdentity, modelType string) inference.ModelInfo {
	info.Architecture = firstNonEmpty(info.Architecture, identity.Architecture, modelType)
	if info.VocabSize == 0 {
		info.VocabSize = identity.VocabSize
	}
	if info.NumLayers == 0 {
		info.NumLayers = identity.NumLayers
	}
	if info.HiddenSize == 0 {
		info.HiddenSize = identity.HiddenSize
	}
	if info.QuantBits == 0 {
		info.QuantBits = identity.QuantBits
	}
	if info.QuantGroup == 0 {
		info.QuantGroup = identity.QuantGroup
	}
	return info
}

func mergeIdentityWithInfo(identity inference.ModelIdentity, info inference.ModelInfo) inference.ModelIdentity {
	identity.Architecture = firstNonEmpty(identity.Architecture, info.Architecture)
	if identity.VocabSize == 0 {
		identity.VocabSize = info.VocabSize
	}
	if identity.NumLayers == 0 {
		identity.NumLayers = info.NumLayers
	}
	if identity.HiddenSize == 0 {
		identity.HiddenSize = info.HiddenSize
	}
	if identity.QuantBits == 0 {
		identity.QuantBits = info.QuantBits
	}
	if identity.QuantGroup == 0 {
		identity.QuantGroup = info.QuantGroup
	}
	return identity
}

func mergeInfoLabels(primary, secondary map[string]string) map[string]string {
	labels := cloneStringMap(primary)
	if labels == nil {
		labels = map[string]string{}
	}
	for key, value := range secondary {
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}

func modelInfoLabels(labels map[string]string, source string, info inference.ModelInfo, identity inference.ModelIdentity) map[string]string {
	labels = cloneStringMap(labels)
	if labels == nil {
		labels = map[string]string{}
	}
	setDefault := func(key, value string) {
		if labels[key] == "" && value != "" {
			labels[key] = value
		}
	}
	setDefault("engine_model_info_contract", ModelInfoReporterContract)
	setDefault("engine_model_info_source", source)
	setDefault("engine_model_info_reactive", "true")
	setDefault("engine_model_info_architecture", info.Architecture)
	setDefault("engine_model_info_path", identity.Path)
	setDefault("engine_model_info_vocab_size", strconv.Itoa(info.VocabSize))
	setDefault("engine_model_info_num_layers", strconv.Itoa(info.NumLayers))
	setDefault("engine_model_info_hidden_size", strconv.Itoa(info.HiddenSize))
	setDefault("engine_model_info_quant_bits", strconv.Itoa(info.QuantBits))
	setDefault("engine_model_info_quant_group", strconv.Itoa(info.QuantGroup))
	return labels
}
