// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import "dappco.re/go/inference"

func rocmCapabilityReportForWrappedModel(model inference.TextModel) inference.CapabilityReport {
	if model == nil {
		return inference.CapabilityReport{Runtime: inference.RuntimeIdentity{Backend: "rocm"}}
	}
	var report inference.CapabilityReport
	if reporter, ok := model.(inference.CapabilityReporter); ok {
		report = rocmCloneCapabilityReport(reporter.Capabilities())
	} else {
		report = inference.TextModelCapabilities(inference.RuntimeIdentity{Backend: "rocm"}, model)
		report = rocmCloneCapabilityReport(report)
	}
	return rocmCapabilityReportWithReactiveProfile(report, model)
}

func rocmCapabilityReportWithReactiveProfile(report inference.CapabilityReport, model inference.TextModel) inference.CapabilityReport {
	if model == nil {
		return report
	}
	identity := rocmDecodeModelIdentity(model)
	if !rocmModelIdentityIsZero(identity) {
		report.Model = rocmMergeCapabilityReportModelIdentity(report.Model, identity)
	}
	profile, ok := ResolveROCmModelProfileForModel(model)
	if !ok || !profile.Matched() {
		return report
	}
	if !rocmModelIdentityIsZero(profile.Model) {
		report.Model = rocmMergeCapabilityReportModelIdentity(report.Model, profile.Model)
	}
	labels := ApplyROCmModelProfileLabels(nil, profile)
	labels = ApplyROCmModelRoutePlanLabels(labels, ROCmModelRoutePlanForProfileAndModel(profile, model))
	rocmCapabilityReportApplyLabels(&report, labels)
	rocmCapabilityReportEnsureEngineFeatureCapabilities(&report, profile.EngineFeatures, labels)
	return report
}

func rocmMergeCapabilityReportModelIdentity(current, richer inference.ModelIdentity) inference.ModelIdentity {
	if rocmModelIdentityIsZero(current) {
		return rocmCloneModelIdentity(richer)
	}
	if current.ID == "" {
		current.ID = richer.ID
	}
	if current.Path == "" {
		current.Path = richer.Path
	}
	if current.Architecture == "" {
		current.Architecture = richer.Architecture
	}
	if current.Revision == "" {
		current.Revision = richer.Revision
	}
	if current.Hash == "" {
		current.Hash = richer.Hash
	}
	if current.QuantBits == 0 {
		current.QuantBits = richer.QuantBits
	}
	if current.QuantGroup == 0 {
		current.QuantGroup = richer.QuantGroup
	}
	if current.QuantType == "" {
		current.QuantType = richer.QuantType
	}
	if current.ContextLength == 0 {
		current.ContextLength = richer.ContextLength
	}
	if current.NumLayers == 0 {
		current.NumLayers = richer.NumLayers
	}
	if current.HiddenSize == 0 {
		current.HiddenSize = richer.HiddenSize
	}
	if current.VocabSize == 0 {
		current.VocabSize = richer.VocabSize
	}
	current.Labels = mergeStringMaps(richer.Labels, current.Labels)
	return current
}

func rocmCapabilityReportEnsureEngineFeatureCapabilities(report *inference.CapabilityReport, features ROCmEngineFeatures, labels map[string]string) {
	if report == nil {
		return
	}
	for _, id := range features.EnabledCapabilities() {
		capability, ok := report.Capability(id)
		if !ok {
			capability = rocmCapabilityForEngineFeature(id)
		}
		capability.Labels = mergeStringMaps(capability.Labels, labels)
		rocmCapabilityReportSetCapability(report, capability)
	}
}

func rocmCapabilityForEngineFeature(id inference.CapabilityID) inference.Capability {
	switch id {
	case inference.CapabilityChatTemplate:
		return inference.ExperimentalCapability(id, inference.CapabilityGroupModel, "registry-declared chat template is available for the loaded model profile")
	default:
		return inference.SupportedCapability(id, inference.CapabilityGroupModel)
	}
}

func rocmCloneCapabilityReport(report inference.CapabilityReport) inference.CapabilityReport {
	report.Runtime.Labels = cloneStringMap(report.Runtime.Labels)
	report.Model = cloneModelIdentity(report.Model)
	report.Tokenizer = cloneTokenizerIdentity(report.Tokenizer)
	report.Adapter = cloneAdapterIdentity(report.Adapter)
	report.Architectures = append([]string(nil), report.Architectures...)
	report.Quantizations = append([]string(nil), report.Quantizations...)
	report.CacheModes = append([]string(nil), report.CacheModes...)
	if len(report.Capabilities) > 0 {
		capabilities := make([]inference.Capability, len(report.Capabilities))
		for index, capability := range report.Capabilities {
			capabilities[index] = rocmCloneCapability(capability)
		}
		report.Capabilities = capabilities
	}
	report.Labels = cloneStringMap(report.Labels)
	return report
}

func rocmCloneCapability(capability inference.Capability) inference.Capability {
	capability.Labels = cloneStringMap(capability.Labels)
	return capability
}

func rocmCapabilityReportSetCapability(report *inference.CapabilityReport, capability inference.Capability) {
	if report == nil || capability.ID == "" {
		return
	}
	capability = rocmCloneCapability(capability)
	for index := range report.Capabilities {
		if report.Capabilities[index].ID == capability.ID {
			report.Capabilities[index] = capability
			return
		}
	}
	report.Capabilities = append(report.Capabilities, capability)
}

func rocmCapabilityReportApplyLabels(report *inference.CapabilityReport, labels map[string]string) {
	if report == nil || len(labels) == 0 {
		return
	}
	report.Labels = mergeStringMaps(report.Labels, labels)
	for index := range report.Capabilities {
		report.Capabilities[index].Labels = mergeStringMaps(report.Capabilities[index].Labels, labels)
	}
}
