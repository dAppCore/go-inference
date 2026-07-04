// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import rocmprofile "dappco.re/go/inference/engine/hip/profile"

// CanonicalWeightName applies the Gemma-4 architecture registry's checkpoint
// weight-name rules. Unknown architectures pass through unchanged.
func CanonicalWeightName(architecture, name string) (string, bool) {
	return rocmprofile.CanonicalWeightName(architecture, name)
}

// TrimWeightWrapperPrefix removes one registered checkpoint wrapper prefix from
// name, reporting whether a Gemma-4 wrapper matched.
func TrimWeightWrapperPrefix(architecture, name string) (string, bool) {
	return rocmprofile.TrimWeightWrapperPrefix(architecture, name)
}

// UnwrapWeightName strips all Gemma-4 checkpoint wrapper prefixes from name.
func UnwrapWeightName(name string) string {
	return rocmprofile.UnwrapGemma4WeightName(name)
}

// TrimOneWeightWrapper strips one Gemma-4 checkpoint wrapper prefix from name.
func TrimOneWeightWrapper(name string) (string, bool) {
	return rocmprofile.TrimOneGemma4WeightWrapper(name)
}

// WeightWrapperPrefixes returns the checkpoint wrapper prefixes used by Gemma-4
// weight canonicalization.
func WeightWrapperPrefixes() []string {
	return rocmprofile.Gemma4WeightWrapperPrefixes()
}
