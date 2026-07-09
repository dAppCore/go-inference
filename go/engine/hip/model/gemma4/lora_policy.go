// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import rocmprofile "dappco.re/go/inference/engine/hip/profile"

type LoRATargetPolicy = rocmprofile.LoRATargetPolicy

func LoRATargetPolicyForArchitecture(architecture string) (LoRATargetPolicy, bool) {
	return rocmprofile.Gemma4LoRATargetPolicyForArchitecture(architecture)
}

func CloneLoRATargetPolicy(policy LoRATargetPolicy) LoRATargetPolicy {
	return rocmprofile.CloneLoRATargetPolicy(policy)
}

func LoRADefaultTargets(architecture string) []string {
	return rocmprofile.Gemma4LoRADefaultTargets(architecture)
}

func LoRATargetPath(architecture, target string) (string, bool) {
	return rocmprofile.Gemma4LoRATargetPath(architecture, target)
}

func LoRASafeTarget(architecture, target string) bool {
	return rocmprofile.Gemma4LoRASafeTarget(architecture, target)
}

func LoRAExtendedTarget(architecture, target string) bool {
	return rocmprofile.Gemma4LoRAExtendedTarget(architecture, target)
}

func LoRACanonicalTarget(architecture, target string) (string, bool) {
	return rocmprofile.Gemma4LoRACanonicalTarget(architecture, target)
}
