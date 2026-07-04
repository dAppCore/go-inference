// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
)

type ProductionQuantizationPackSupport = modelgemma4.ProductionQuantizationPackSupport

// DefaultProductionQuantizationPackSupport returns every Gemma 4 pack type ROCm
// recognises for product selection, benchmark selection, or R&D validation.
// q6/q8/q4 remain the app-facing E2B ladder; E4B and 12B entries are explicit
// larger local targets, while 26B-A4B and 31B stay metadata/status-only on the
// pinned card.
func DefaultProductionQuantizationPackSupport() []ProductionQuantizationPackSupport {
	return modelgemma4.DefaultProductionQuantizationPackSupport()
}

// ProductionQuantizationPackByName resolves a supported pack by short name
// ("6bit", "mxfp8") or full model ID.
func ProductionQuantizationPackByName(name string) (ProductionQuantizationPackSupport, bool) {
	return modelgemma4.ProductionQuantizationPackByName(name)
}

func ProductionQuantizationPacksBySize(size string) []ProductionQuantizationPackSupport {
	return modelgemma4.ProductionQuantizationPacksBySize(size)
}

func ApplyProductionQuantizationPackSupportLabels(labels map[string]string) {
	modelgemma4.ApplyProductionQuantizationPackSupportLabels(labels)
}

func productionQuantizationPackLabelName(pack ProductionQuantizationPackSupport) string {
	return modelgemma4.ProductionQuantizationPackLabelName(pack)
}

func rocmGemma4ProductionQuantPackAlias(name string) (ProductionQuantizationPackSupport, bool) {
	return modelgemma4.ProductionQuantizationPackAlias(name)
}

func rocmGemma4ProductionQuantGGUFPackAlias(name, size, mode string) (ProductionQuantizationPackSupport, bool) {
	return modelgemma4.ProductionQuantizationGGUFPackAlias(name, size, mode)
}

func rocmGemma4ProductionQuantPackForModel(model inference.ModelIdentity) (ProductionQuantizationPackSupport, bool) {
	return modelgemma4.ProductionQuantizationPackForModel(model)
}

func rocmGemma4ProductionQuantAssistantPackForModel(model inference.ModelIdentity) (ProductionQuantizationPackSupport, bool) {
	return modelgemma4.ProductionQuantizationAssistantPackForModel(model)
}

func rocmGemma4ProductionQuantPackMode(pack ProductionQuantizationPackSupport) string {
	return modelgemma4.ProductionQuantizationPackMode(pack)
}

func rocmApplyGemma4ProductionQuantPackLabels(labels map[string]string, pack ProductionQuantizationPackSupport) {
	if labels == nil {
		return
	}
	labels["production_quant_size"] = pack.Size
	labels["production_quant_pack"] = productionQuantizationPackLabelName(pack)
	labels["production_quant_pack_name"] = pack.Name
	labels["production_quant_tier"] = pack.ProductRole
	labels["production_quant_model"] = pack.ModelID
	if pack.SourceCollection != "" {
		labels["production_quant_collection"] = pack.SourceCollection
	}
	if pack.LockedModelID != "" {
		labels["production_quant_locked_model"] = pack.LockedModelID
	}
	labels["production_quant_mode"] = rocmGemma4ProductionQuantPackMode(pack)
	labels["production_quant_bits"] = core.Sprintf("%d", pack.Bits)
	if pack.QuantGroup > 0 {
		labels["production_quant_group"] = core.Sprintf("%d", pack.QuantGroup)
	}
	if pack.Runtime != "" {
		labels["production_quant_runtime"] = pack.Runtime
	}
	if pack.GenerateStatus != "" {
		labels["production_quant_generate_status"] = pack.GenerateStatus
	}
	labels["production_quant_supported"] = core.Sprintf("%t", pack.Supported)
	labels["production_quant_runnable_on_card"] = core.Sprintf("%t", pack.RunnableOnCard)
	if pack.RequiresBench {
		labels["production_quant_requires_bench"] = "true"
	}
	if pack.RequiresNative {
		labels["production_quant_requires_native"] = "true"
	}
	if pack.ProductRole != "mtp-assistant" {
		if target, ok := rocmGemma4ProductionQuantPackBySizeRole(pack.Size, "default"); ok {
			labels["production_quant_target_model"] = target.ModelID
		} else if pack.ProductRole == "largest-local-target" {
			labels["production_quant_target_model"] = pack.ModelID
		}
		if quality, ok := rocmGemma4ProductionQuantPackBySizeRole(pack.Size, "quality"); ok {
			labels["production_quant_quality_model"] = quality.ModelID
		}
		if constrained, ok := rocmGemma4ProductionQuantPackBySizeRole(pack.Size, "constrained"); ok {
			labels["production_quant_archived_baseline"] = constrained.ModelID
		}
	}
	switch pack.ProductRole {
	case "quality":
		labels["production_quant_quality_first"] = "true"
		if pack.Size == "E2B" {
			rocmApplyGemma4StaticProductionQuantTierLabels(labels, pack.Bits)
		}
	case "default":
		labels["production_quant_product_default"] = "true"
		labels["production_quant_size_default"] = "true"
		if pack.Size == "E2B" {
			rocmApplyGemma4StaticProductionQuantTierLabels(labels, pack.Bits)
		}
	case "constrained":
		labels["production_quant_constrained_only"] = "true"
		if pack.ModelID == ProductionLaneArchivedBaselineModelID || pack.ModelID == ProductionLaneCurrentConstrainedModelID {
			labels["production_quant_archived_control"] = "true"
			rocmApplyGemma4StaticProductionQuantTierLabels(labels, pack.Bits)
		}
	case "largest-local-target":
		labels["production_quant_size_default"] = "true"
	case "mtp-assistant":
		labels["production_quant_mtp_assistant"] = "true"
		labels["production_quant_assistant_model"] = pack.ModelID
		labels["production_quant_target_family"] = "gemma4"
	}
}

func rocmGemma4ProductionQuantPackBySizeRole(size, role string) (ProductionQuantizationPackSupport, bool) {
	return modelgemma4.ProductionQuantizationPackBySizeRole(size, role)
}

func appendUniqueString(values []string, value string) []string {
	if value == "" {
		return values
	}
	for _, existing := range values {
		if existing == value {
			return values
		}
	}
	return append(values, value)
}
