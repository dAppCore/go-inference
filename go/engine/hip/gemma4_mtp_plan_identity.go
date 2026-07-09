// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"strconv"
	"strings"

	"dappco.re/go/inference"
)

func productionMTPPlanTargetIdentity(plan AttachedDrafterPlan) inference.ModelIdentity {
	return productionMTPPlanGemma4Identity(plan.Target, plan.Labels, true)
}

func productionMTPPlanDraftIdentity(plan AttachedDrafterPlan) inference.ModelIdentity {
	return productionMTPPlanGemma4Identity(plan.Draft, plan.Labels, false)
}

func productionMTPPlanGemma4Identity(info inference.ModelInfo, labels map[string]string, target bool) inference.ModelIdentity {
	identity := productionMTPModelInfoIdentity(info)
	identity.Labels = mergeStringMaps(identity.Labels, productionMTPPlanGemma4Labels(labels, target))
	if group := productionMTPPlanGemma4QuantGroup(labels, target); group > 0 {
		identity.QuantGroup = group
	}
	return rocmGemma4ModelWithInferredPathQuant(identity)
}

func productionMTPPlanGemma4Labels(labels map[string]string, target bool) map[string]string {
	out := map[string]string{}
	for _, suffix := range []string{"size", "quant_mode", "runtime", "generate_status", "pack_supported", "runnable_on_card"} {
		if value := productionMTPPlanGemma4Label(labels, target, suffix); value != "" {
			out["gemma4_"+suffix] = value
		}
	}
	return out
}

func productionMTPPlanGemma4QuantGroup(labels map[string]string, target bool) int {
	value := productionMTPPlanGemma4Label(labels, target, "quant_group")
	if value == "" {
		return 0
	}
	parsed, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || parsed <= 0 {
		return 0
	}
	return parsed
}

func productionMTPPlanGemma4Label(labels map[string]string, target bool, suffix string) string {
	aliases := []string{"attached_drafter_target_gemma4_" + suffix, "attached.drafter.target.gemma4_" + suffix}
	if !target {
		aliases = []string{
			"assistant_gemma4_" + suffix,
			"draft_gemma4_" + suffix,
			"attached_drafter_assistant_gemma4_" + suffix,
			"attached_drafter_draft_gemma4_" + suffix,
			"attached.drafter.assistant.gemma4_" + suffix,
			"attached.drafter.draft.gemma4_" + suffix,
		}
	} else {
		aliases = append([]string{"target_gemma4_" + suffix}, aliases...)
	}
	_, value := productionFirstLabel(labels, aliases)
	return value
}
