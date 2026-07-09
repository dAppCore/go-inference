// SPDX-Licence-Identifier: EUPL-1.2

// Runnable examples for eviction.go — kept separate from eviction_test.go so
// the godoc-attached usage snippets stay readable.

package lora

import core "dappco.re/go"

// ExampleNewLRUEvictionPolicy shows the recency contract: marking b after a
// makes a the least-recently-used, so a is the victim selected among the
// candidates offered by the Pool.
func ExampleNewLRUEvictionPolicy() {
	pol := NewLRUEvictionPolicy()
	pol.MarkUsed("a")
	pol.MarkUsed("b")

	victim, ok := pol.SelectVictim([]string{"a", "b"})
	core.Println(victim, ok)

	// Output:
	// a true
}
