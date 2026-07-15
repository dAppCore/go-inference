// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"dappco.re/go/inference"
	rocmprofile "dappco.re/go/inference/engine/hip/profile"
)

// ROCmAlgorithmProfile describes one backend-neutral algorithm or runtime
// feature surface in ROCm terms.
type ROCmAlgorithmProfile = rocmprofile.AlgorithmProfile

const ROCmAlgorithmProfileRegistryContract = rocmprofile.AlgorithmProfileRegistryContract

// DefaultROCmAlgorithmProfiles returns the built-in algorithm matrix exposed by
// discovery, daemon registry, and API consumers.
func DefaultROCmAlgorithmProfiles() []ROCmAlgorithmProfile {
	return rocmprofile.BuiltinAlgorithmProfiles()
}

// ROCmAlgorithmProfileByID returns the registered profile for id.
func ROCmAlgorithmProfileByID(id inference.CapabilityID) (ROCmAlgorithmProfile, bool) {
	return rocmprofile.LookupAlgorithmProfile(id)
}

// ROCmAlgorithmCapabilities returns the algorithm matrix as capability rows.
func ROCmAlgorithmCapabilities() []inference.Capability {
	return rocmprofile.AlgorithmCapabilities()
}
