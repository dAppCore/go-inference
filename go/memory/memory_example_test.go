// SPDX-Licence-Identifier: EUPL-1.2

package memory_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/memory"
)

// Runnable examples for the public memory-planner API. Each invokes the real
// symbol and prints a single deterministic field so the // Output line is
// stable across machines.

// ExampleNewPlan derives a runtime policy from a measured 96GB Apple device and
// prints the machine class the planner selected.
func ExampleNewPlan() {
	plan := memory.NewPlan(memory.Input{
		Device: memory.DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		},
	})
	core.Println(string(plan.MachineClass))
	// Output: apple-silicon-96gb
}

// ExampleClassForBytes maps a raw byte count to its memory tier without building
// a full plan.
func ExampleClassForBytes() {
	core.Println(string(memory.ClassForBytes(96 * memory.GiB)))
	// Output: apple-silicon-96gb
}

// ExampleIsKnownKVCacheMode reports whether a KV-cache mode is part of the
// public contract — the empty default is a member, a made-up mode is not.
func ExampleIsKnownKVCacheMode() {
	core.Println(memory.IsKnownKVCacheMode(memory.KVCacheModeQ8))
	core.Println(memory.IsKnownKVCacheMode(memory.KVCacheMode("not-a-mode")))
	// Output:
	// true
	// false
}
