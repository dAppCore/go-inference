// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"strconv"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func rocmAdapterIdentityForModel(identity inference.AdapterIdentity, model inference.ModelIdentity) inference.AdapterIdentity {
	identity = cloneAdapterIdentity(identity)
	if adapterIdentityIsZero(identity) || !rocmIsGemma4SizeQuantIdentity(model.Architecture) {
		return identity
	}
	model = rocmGemma4ModelWithInferredPathQuant(model)
	if identity.Labels == nil {
		identity.Labels = map[string]string{}
	}
	rocmApplyGemma4SizeQuantSupportLabels(identity.Labels, model)
	rocmApplyGemma4ProductionQuantLabels(identity.Labels, model)
	rocmAddAdapterBaseProductionQuantLabels(identity.Labels)
	identity.Labels["adapter_base_architecture"] = model.Architecture
	if model.Path != "" {
		identity.Labels["adapter_base_model_path"] = model.Path
	}
	if model.Hash != "" {
		if identity.BaseModelHash == "" {
			identity.BaseModelHash = model.Hash
		}
		identity.Labels["adapter_base_model_hash"] = model.Hash
	}
	if size := identity.Labels["gemma4_size"]; size != "" {
		identity.Labels["adapter_base_gemma4_size"] = size
	}
	if mode := identity.Labels["gemma4_quant_mode"]; mode != "" {
		identity.Labels["adapter_base_gemma4_quant_mode"] = mode
	}
	if model.QuantGroup > 0 {
		group := core.Sprintf("%d", model.QuantGroup)
		identity.Labels["gemma4_quant_group"] = group
		identity.Labels["adapter_base_gemma4_quant_group"] = group
	}
	if runtime := identity.Labels["gemma4_runtime"]; runtime != "" {
		identity.Labels["adapter_base_gemma4_runtime"] = runtime
	}
	if status := identity.Labels["gemma4_generate_status"]; status != "" {
		identity.Labels["adapter_base_gemma4_generate_status"] = status
	}
	if supported := identity.Labels["gemma4_pack_supported"]; supported != "" {
		identity.Labels["adapter_base_gemma4_pack_supported"] = supported
	}
	if runnable := identity.Labels["gemma4_runnable_on_card"]; runnable != "" {
		identity.Labels["adapter_base_gemma4_runnable_on_card"] = runnable
	}
	return identity
}

func checkROCmAdapterModelCompatibility(operation string, model inference.ModelIdentity, adapter inference.AdapterIdentity) error {
	if adapterIdentityIsZero(adapter) {
		return nil
	}
	model = rocmGemma4ModelWithInferredPathQuant(model)
	if adapter.BaseModelHash != "" && model.Hash != "" && adapter.BaseModelHash != model.Hash {
		return core.E(operation, "adapter base model hash mismatch", nil)
	}
	adapterArchitecture := firstNonEmptyString(adapter.Labels["adapter_base_architecture"], adapter.Labels["base_architecture"])
	if adapterArchitecture != "" && model.Architecture != "" && normalizeROCmArchitecture(adapterArchitecture) != normalizeROCmArchitecture(model.Architecture) {
		return core.E(operation, "adapter base model architecture mismatch", nil)
	}
	if err := checkROCmAdapterProductionQuantCompatibility(operation, model, adapter); err != nil {
		return err
	}
	adapterSize := rocmAdapterGemma4BaseSize(adapter)
	adapterMode := rocmAdapterGemma4BaseQuantMode(adapter)
	if adapterSize == "" && adapterMode == "" {
		if rocmAdapterHasGemma4BaseMetadata(adapter) {
			return core.E(operation, "adapter base Gemma4 identity is incomplete", nil)
		}
		return nil
	}
	if adapterSize == "" || adapterMode == "" {
		return core.E(operation, "adapter base Gemma4 identity is incomplete", nil)
	}
	if !rocmIsGemma4SizeQuantIdentity(model.Architecture) {
		return core.E(operation, "adapter Gemma4 base model mismatch", nil)
	}
	modelSize := rocmGemma4ModelPackSize(model, model.Path)
	modelMode := rocmGemma4ModelPackQuantModeForPath(model, model.Path)
	modelMode = rocmGemma4NormalizeSizeQuantMode(modelSize, modelMode)
	if adapterSize != "" && adapterSize != modelSize {
		return core.E(operation, "adapter base Gemma4 size mismatch", nil)
	}
	if adapterMode != "" && adapterMode != modelMode {
		return core.E(operation, "adapter base Gemma4 quant mismatch", nil)
	}
	if groupLabel := firstNonEmptyString(adapter.Labels["adapter_base_gemma4_quant_group"], adapter.Labels["gemma4_quant_group"]); groupLabel != "" {
		group, err := strconv.Atoi(core.Trim(groupLabel))
		if err != nil || group <= 0 {
			return core.E(operation, "adapter base Gemma4 quant group is invalid", err)
		}
		if model.QuantGroup <= 0 || group != model.QuantGroup {
			return core.E(operation, "adapter base Gemma4 quant group mismatch", nil)
		}
	}
	expectedLabels := map[string]string{}
	rocmApplyGemma4SizeQuantSupportLabels(expectedLabels, model)
	if runtime := firstNonEmptyString(adapter.Labels["adapter_base_gemma4_runtime"], adapter.Labels["gemma4_runtime"]); runtime != "" && expectedLabels["gemma4_runtime"] != "" && runtime != expectedLabels["gemma4_runtime"] {
		return core.E(operation, "adapter base Gemma4 runtime mismatch", nil)
	}
	if status := firstNonEmptyString(adapter.Labels["adapter_base_gemma4_generate_status"], adapter.Labels["gemma4_generate_status"]); status != "" && expectedLabels["gemma4_generate_status"] != "" && status != expectedLabels["gemma4_generate_status"] {
		return core.E(operation, "adapter base Gemma4 generate status mismatch", nil)
	}
	if supported := firstNonEmptyString(adapter.Labels["adapter_base_gemma4_pack_supported"], adapter.Labels["gemma4_pack_supported"]); supported != "" && expectedLabels["gemma4_pack_supported"] != "" && core.Lower(core.Trim(supported)) != expectedLabels["gemma4_pack_supported"] {
		return core.E(operation, "adapter base Gemma4 pack support mismatch", nil)
	}
	if runnable := firstNonEmptyString(adapter.Labels["adapter_base_gemma4_runnable_on_card"], adapter.Labels["gemma4_runnable_on_card"]); runnable != "" && expectedLabels["gemma4_runnable_on_card"] != "" && core.Lower(core.Trim(runnable)) != expectedLabels["gemma4_runnable_on_card"] {
		return core.E(operation, "adapter base Gemma4 runnable status mismatch", nil)
	}
	return nil
}

func checkROCmAdapterProductionQuantCompatibility(operation string, model inference.ModelIdentity, adapter inference.AdapterIdentity) error {
	if !rocmAdapterHasProductionQuantBaseMetadata(adapter) {
		return nil
	}
	if !rocmIsGemma4SizeQuantIdentity(model.Architecture) {
		return core.E(operation, "adapter Gemma4 production quant base model mismatch", nil)
	}
	expected := map[string]string{}
	rocmApplyGemma4ProductionQuantLabels(expected, model)
	for _, check := range []struct {
		name        string
		expectedKey string
		actualKeys  []string
	}{
		{name: "model", expectedKey: "production_quant_model", actualKeys: []string{"adapter_base_production_quant_model", "production_quant_model"}},
		{name: "locked model", expectedKey: "production_quant_locked_model", actualKeys: []string{"adapter_base_production_quant_locked_model", "production_quant_locked_model"}},
		{name: "pack", expectedKey: "production_quant_pack", actualKeys: []string{"adapter_base_production_quant_pack", "production_quant_pack"}},
		{name: "tier", expectedKey: "production_quant_tier", actualKeys: []string{"adapter_base_production_quant_tier", "production_quant_tier"}},
		{name: "target model", expectedKey: "production_quant_target_model", actualKeys: []string{"adapter_base_production_quant_target_model", "production_quant_target_model"}},
		{name: "assistant model", expectedKey: "production_quant_assistant_model", actualKeys: []string{"adapter_base_production_quant_assistant_model", "production_quant_assistant_model"}},
		{name: "MTP assistant", expectedKey: "production_quant_mtp_assistant", actualKeys: []string{"adapter_base_production_quant_mtp_assistant", "production_quant_mtp_assistant"}},
		{name: "target family", expectedKey: "production_quant_target_family", actualKeys: []string{"adapter_base_production_quant_target_family", "production_quant_target_family"}},
	} {
		actual := firstNonEmptyStringFromKeys(adapter.Labels, check.actualKeys...)
		if actual == "" {
			continue
		}
		if expected[check.expectedKey] == "" || normalizeProductionQuantAdapterLabel(actual) != normalizeProductionQuantAdapterLabel(expected[check.expectedKey]) {
			return core.E(operation, "adapter base production quant "+check.name+" mismatch", nil)
		}
	}
	return nil
}

func rocmAdapterHasProductionQuantBaseMetadata(adapter inference.AdapterIdentity) bool {
	for _, key := range []string{
		"adapter_base_production_quant_model",
		"production_quant_model",
		"adapter_base_production_quant_locked_model",
		"production_quant_locked_model",
		"adapter_base_production_quant_pack",
		"production_quant_pack",
		"adapter_base_production_quant_tier",
		"production_quant_tier",
		"adapter_base_production_quant_target_model",
		"production_quant_target_model",
		"adapter_base_production_quant_assistant_model",
		"production_quant_assistant_model",
		"adapter_base_production_quant_mtp_assistant",
		"production_quant_mtp_assistant",
		"adapter_base_production_quant_target_family",
		"production_quant_target_family",
	} {
		if core.Trim(adapter.Labels[key]) != "" {
			return true
		}
	}
	return false
}

func firstNonEmptyStringFromKeys(labels map[string]string, keys ...string) string {
	for _, key := range keys {
		if value := core.Trim(labels[key]); value != "" {
			return value
		}
	}
	return ""
}

func normalizeProductionQuantAdapterLabel(value string) string {
	return core.Lower(core.Trim(value))
}

func rocmAdapterGemma4BaseSize(adapter inference.AdapterIdentity) string {
	size := firstNonEmptyString(adapter.Labels["adapter_base_gemma4_size"], adapter.Labels["gemma4_size"])
	if size == "" {
		base := rocmAdapterBaseModelIdentity(adapter)
		size = rocmGemma4ModelPackSize(base, base.Path)
	}
	return rocmGemma4CanonicalSize(size)
}

func rocmAdapterGemma4BaseQuantMode(adapter inference.AdapterIdentity) string {
	mode := firstNonEmptyString(adapter.Labels["adapter_base_gemma4_quant_mode"], adapter.Labels["gemma4_quant_mode"])
	size := rocmAdapterGemma4BaseSize(adapter)
	if mode == "" {
		base := rocmAdapterBaseModelIdentity(adapter)
		mode = rocmGemma4ModelPackQuantModeForPath(base, base.Path)
	}
	return rocmGemma4CanonicalQuantMode(size, mode)
}

func rocmAdapterHasGemma4BaseMetadata(adapter inference.AdapterIdentity) bool {
	architecture := firstNonEmptyString(adapter.Labels["adapter_base_architecture"], adapter.Labels["base_architecture"])
	if rocmIsGemma4SizeQuantIdentity(architecture) {
		return true
	}
	for _, key := range []string{
		"adapter_base_gemma4_size",
		"gemma4_size",
		"adapter_base_gemma4_quant_mode",
		"gemma4_quant_mode",
		"adapter_base_gemma4_quant_group",
		"gemma4_quant_group",
		"adapter_base_gemma4_runtime",
		"gemma4_runtime",
		"adapter_base_gemma4_generate_status",
		"gemma4_generate_status",
		"adapter_base_gemma4_pack_supported",
		"gemma4_pack_supported",
		"adapter_base_gemma4_runnable_on_card",
		"gemma4_runnable_on_card",
	} {
		if core.Trim(adapter.Labels[key]) != "" {
			return true
		}
	}
	return false
}

func rocmAdapterBaseModelIdentity(adapter inference.AdapterIdentity) inference.ModelIdentity {
	path := firstNonEmptyString(
		adapter.Labels["adapter_base_model_path"],
		adapter.Labels["base_model_path"],
		adapter.Labels["adapter_base_path"],
		adapter.Labels["adapter_base_production_quant_model"],
		adapter.Labels["production_quant_model"],
		adapter.Labels["adapter_base_production_quant_assistant_model"],
		adapter.Labels["production_quant_assistant_model"],
		adapter.Labels["adapter_base_production_quant_locked_model"],
		adapter.Labels["production_quant_locked_model"],
		adapter.Labels["base_model"],
	)
	architecture := firstNonEmptyString(adapter.Labels["adapter_base_architecture"], adapter.Labels["base_architecture"])
	if architecture == "" && rocmAdapterBaseProductionQuantAssistantModel(adapter) != "" {
		architecture = officialGemma4E2BAssistantArchitecture
	}
	if architecture == "" && rocmAdapterBasePathLooksLikeGemma4Assistant(path) {
		architecture = officialGemma4E2BAssistantArchitecture
	}
	return inference.ModelIdentity{
		Path:         path,
		Architecture: normalizeROCmArchitecture(architecture),
	}
}

func rocmAdapterBaseProductionQuantAssistantModel(adapter inference.AdapterIdentity) string {
	return firstNonEmptyString(
		adapter.Labels["adapter_base_production_quant_assistant_model"],
		adapter.Labels["production_quant_assistant_model"],
	)
}

func rocmAdapterBasePathLooksLikeGemma4Assistant(path string) bool {
	path = strings.ToLower(strings.TrimSpace(path))
	return strings.Contains(path, "gemma-4") && strings.Contains(path, "assistant")
}

func rocmAddStateBundleAdapterLabels(labels map[string]string, adapter inference.AdapterIdentity) {
	if labels == nil || adapterIdentityIsZero(adapter) {
		return
	}
	labels["state_adapter"] = "metadata_only"
	rocmAddAdapterMetadataLabels(labels, adapter)
}

func rocmAddCapabilityAdapterLabels(labels map[string]string, adapter inference.AdapterIdentity) {
	if labels == nil || adapterIdentityIsZero(adapter) {
		return
	}
	labels["active_adapter"] = "true"
	rocmAddAdapterMetadataLabels(labels, adapter)
}

func rocmApplyCapabilityAdapterLabels(capabilities []inference.Capability, adapter inference.AdapterIdentity) {
	if adapterIdentityIsZero(adapter) {
		return
	}
	for i := range capabilities {
		if capabilities[i].Labels == nil {
			capabilities[i].Labels = map[string]string{}
		}
		rocmAddCapabilityAdapterLabels(capabilities[i].Labels, adapter)
	}
}

func rocmAddAdapterBaseProductionQuantLabels(labels map[string]string) {
	if labels == nil {
		return
	}
	for source, target := range map[string]string{
		"production_quant_model":             "adapter_base_production_quant_model",
		"production_quant_locked_model":      "adapter_base_production_quant_locked_model",
		"production_quant_pack":              "adapter_base_production_quant_pack",
		"production_quant_tier":              "adapter_base_production_quant_tier",
		"production_quant_target_model":      "adapter_base_production_quant_target_model",
		"production_quant_assistant_model":   "adapter_base_production_quant_assistant_model",
		"production_quant_archived_baseline": "adapter_base_production_quant_archived_baseline",
		"production_quant_mtp_assistant":     "adapter_base_production_quant_mtp_assistant",
		"production_quant_target_family":     "adapter_base_production_quant_target_family",
	} {
		if value := labels[source]; value != "" {
			labels[target] = value
		}
	}
}

func rocmAddAdapterMetadataLabels(labels map[string]string, adapter inference.AdapterIdentity) {
	if labels == nil || adapterIdentityIsZero(adapter) {
		return
	}
	if adapter.Path != "" {
		labels["adapter_path"] = adapter.Path
	}
	if adapter.Hash != "" {
		labels["adapter_hash"] = adapter.Hash
	}
	if adapter.Format != "" {
		labels["adapter_format"] = adapter.Format
	}
	for _, key := range []string{
		"adapter_base_architecture",
		"adapter_base_model_hash",
		"adapter_base_model_path",
		"adapter_base_gemma4_size",
		"adapter_base_gemma4_quant_mode",
		"adapter_base_gemma4_quant_group",
		"adapter_base_gemma4_runtime",
		"adapter_base_gemma4_generate_status",
		"adapter_base_gemma4_pack_supported",
		"adapter_base_gemma4_runnable_on_card",
		"adapter_base_production_quant_model",
		"adapter_base_production_quant_locked_model",
		"adapter_base_production_quant_pack",
		"adapter_base_production_quant_tier",
		"adapter_base_production_quant_target_model",
		"adapter_base_production_quant_assistant_model",
		"adapter_base_production_quant_archived_baseline",
		"adapter_base_production_quant_mtp_assistant",
		"adapter_base_production_quant_target_family",
	} {
		if value := adapter.Labels[key]; value != "" {
			labels[key] = value
		}
	}
}
