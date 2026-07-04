// SPDX-Licence-Identifier: EUPL-1.2

// Runnable usage examples for the algorithm feature-matrix surface — the
// metadata side of the profile package consumed by capability reports and
// backend planning. Each Example mirrors how a CapabilityReport assembler or a
// backend planner actually reads the matrix: look a capability up, read its
// implementation state, or fold the whole list into the shared Capability
// slice. Output is pinned to deterministic scalar fields (never a map or %+v
// dump) so the examples double as compiled assertions.

package profile_test

import (
	"fmt"

	"dappco.re/go/inference"
	prof "dappco.re/go/inference/profile"
)

// LookupAlgorithmProfile resolves one capability ID to its built-in profile.
// A backend planner reads the runtime state to decide whether a feature is
// wired natively, available only as metadata, or still planned.
func ExampleLookupAlgorithmProfile() {
	p, ok := prof.LookupAlgorithmProfile(inference.CapabilityScheduler)
	fmt.Println(ok, p.Algorithm, p.RuntimeStatus)
	// Output: true scheduler native
}

// An experimental capability advertises its algorithm name and an
// experimental runtime state — the report surface labels it accordingly.
func ExampleLookupAlgorithmProfile_experimental() {
	p, _ := prof.LookupAlgorithmProfile(inference.CapabilityQuantization)
	fmt.Println(p.Algorithm, p.RuntimeStatus, p.CapabilityStatus)
	// Output: auto-round experimental experimental
}

// A capability ID that names no built-in algorithm returns ok=false and the
// zero profile, so callers branch on ok rather than guessing.
func ExampleLookupAlgorithmProfile_miss() {
	p, ok := prof.LookupAlgorithmProfile(inference.CapabilityID("not-a-real-capability"))
	fmt.Printf("%v %q\n", ok, p.Algorithm)
	// Output: false ""
}

// AlgorithmCapabilities folds the whole built-in matrix into the
// inference.Capability slice a CapabilityReport appends. Each capability
// carries its runtime state as a label, read here for the scheduler entry.
func ExampleAlgorithmCapabilities() {
	for _, capability := range prof.AlgorithmCapabilities() {
		if capability.ID == inference.CapabilityScheduler {
			fmt.Println(capability.ID, capability.Labels["runtime_status"], capability.Labels["algorithm"])
		}
	}
	// Output: scheduler native scheduler
}

// BuiltinAlgorithmProfiles returns a defensive clone — mutating the returned
// slice never touches the registry singleton, so a later call sees the
// original algorithm name.
func ExampleBuiltinAlgorithmProfiles() {
	first := prof.BuiltinAlgorithmProfiles()
	first[0].Algorithm = "mutated-by-caller"
	again := prof.BuiltinAlgorithmProfiles()
	fmt.Println(again[0].Algorithm == "mutated-by-caller")
	// Output: false
}
