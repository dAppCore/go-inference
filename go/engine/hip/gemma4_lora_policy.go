// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"strings"

	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
	rocmprofile "dappco.re/go/inference/engine/hip/profile"
)

type Gemma4LoRATargetPolicy = rocmprofile.LoRATargetPolicy

// ROCmLoRATargetPolicyForArchitecture returns the loader-neutral LoRA target
// policy declared by the model registry for architecture.
func ROCmLoRATargetPolicyForArchitecture(architecture string) (Gemma4LoRATargetPolicy, bool) {
	return rocmLoRATargetPolicyForArchitecture(architecture)
}

func ROCmLoRADefaultTargets(architecture string) []string {
	policy, ok := ROCmLoRATargetPolicyForArchitecture(architecture)
	if !ok {
		return nil
	}
	return cloneGemma4LoRAStringSlice(policy.DefaultTargets)
}

func ROCmLoRATargetPath(architecture, target string) (string, bool) {
	policy, ok := ROCmLoRATargetPolicyForArchitecture(architecture)
	if !ok {
		return "", false
	}
	path, ok := policy.TargetPaths[strings.TrimSpace(target)]
	return path, ok
}

func ROCmLoRASafeTarget(architecture, target string) bool {
	policy, ok := ROCmLoRATargetPolicyForArchitecture(architecture)
	if !ok {
		return false
	}
	path, ok := policy.TargetPaths[strings.TrimSpace(target)]
	if !ok {
		return false
	}
	for _, extended := range policy.ExtendedTargets {
		if path == extended {
			return false
		}
	}
	return true
}

func ROCmLoRAExtendedTarget(architecture, target string) bool {
	policy, ok := ROCmLoRATargetPolicyForArchitecture(architecture)
	if !ok {
		return false
	}
	path, ok := policy.TargetPaths[strings.TrimSpace(target)]
	if !ok {
		return false
	}
	for _, extended := range policy.ExtendedTargets {
		if path == extended {
			return true
		}
	}
	return false
}

func ROCmLoRACanonicalTarget(architecture, target string) (string, bool) {
	target = strings.TrimSpace(target)
	if target == "" {
		return "", false
	}
	parts := strings.Split(target, ".")
	if len(parts) >= 2 {
		short := parts[len(parts)-2] + "." + parts[len(parts)-1]
		if canonical, ok := ROCmLoRATargetPath(architecture, short); ok {
			return joinGemma4LoRACanonicalTarget(parts[:len(parts)-2], canonical), true
		}
		if len(parts) == 2 {
			return "", false
		}
	}
	short := parts[len(parts)-1]
	if canonical, ok := ROCmLoRATargetPath(architecture, short); ok {
		return joinGemma4LoRACanonicalTarget(parts[:len(parts)-1], canonical), true
	}
	return "", false
}

func Gemma4LoRATargetPolicyForArchitecture(architecture string) (Gemma4LoRATargetPolicy, bool) {
	return modelgemma4.LoRATargetPolicyForArchitecture(architecture)
}

func cloneGemma4LoRATargetPolicy(policy Gemma4LoRATargetPolicy) Gemma4LoRATargetPolicy {
	return modelgemma4.CloneLoRATargetPolicy(policy)
}

func rocmApplyGemma4LoRAPolicyLabels(labels map[string]string, architecture string, policy Gemma4LoRATargetPolicy) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if len(policy.DefaultTargets) == 0 && len(policy.SafeTargets) == 0 && len(policy.ExtendedTargets) == 0 && len(policy.TargetPaths) == 0 {
		var ok bool
		policy, ok = Gemma4LoRATargetPolicyForArchitecture(architecture)
		if !ok {
			return labels
		}
	}
	targets := append(cloneGemma4LoRAStringSlice(policy.SafeTargets), policy.ExtendedTargets...)
	labels["engine_lora_policy"] = "gemma4"
	labels["engine_lora_policy_source"] = "model_registry"
	labels["engine_lora_target_family"] = "gemma4"
	labels["engine_lora_targets"] = strings.Join(targets, ",")
	labels["engine_lora_default_targets"] = strings.Join(policy.DefaultTargets, ",")
	labels["engine_lora_safe_targets"] = strings.Join(policy.SafeTargets, ",")
	labels["engine_lora_extended_targets"] = strings.Join(policy.ExtendedTargets, ",")
	labels["engine_lora_extended_targets_require_opt_in"] = "true"
	labels["gemma4_lora_policy"] = "model_registry"
	labels["gemma4_lora_targets"] = strings.Join(targets, ",")
	labels["gemma4_lora_default_targets"] = strings.Join(policy.DefaultTargets, ",")
	labels["gemma4_lora_safe_targets"] = strings.Join(policy.SafeTargets, ",")
	labels["gemma4_lora_extended_targets"] = strings.Join(policy.ExtendedTargets, ",")
	labels["gemma4_lora_extended_targets_require_opt_in"] = "true"
	return labels
}

func Gemma4LoRADefaultTargets(architecture string) []string {
	return modelgemma4.LoRADefaultTargets(architecture)
}

func Gemma4LoRATargetPath(architecture, target string) (string, bool) {
	return modelgemma4.LoRATargetPath(architecture, target)
}

func Gemma4LoRASafeTarget(architecture, target string) bool {
	return modelgemma4.LoRASafeTarget(architecture, target)
}

func Gemma4LoRAExtendedTarget(architecture, target string) bool {
	return modelgemma4.LoRAExtendedTarget(architecture, target)
}

func Gemma4LoRACanonicalTarget(architecture, target string) (string, bool) {
	return modelgemma4.LoRACanonicalTarget(architecture, target)
}

func Gemma4CanonicalWeightName(architecture, name string) (string, bool) {
	return modelgemma4.CanonicalWeightName(architecture, name)
}

func joinGemma4LoRACanonicalTarget(prefix []string, canonical string) string {
	if len(prefix) == 0 {
		return canonical
	}
	parts := make([]string, 0, len(prefix)+strings.Count(canonical, ".")+1)
	parts = append(parts, prefix...)
	parts = append(parts, strings.Split(canonical, ".")...)
	return strings.Join(parts, ".")
}

func unwrapGemma4WeightName(name string) string {
	return modelgemma4.UnwrapWeightName(name)
}

func trimOneGemma4WeightWrapper(name string) (string, bool) {
	return modelgemma4.TrimOneWeightWrapper(name)
}

func cloneGemma4LoRAStringSlice(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	return append([]string(nil), values...)
}
