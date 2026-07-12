// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	core "dappco.re/go"
)

// ExampleServeService shows a freshly built serve service before any poll: the
// daemon reads as down and unmanaged until Start spawns one or a poll finds an
// externally-running serve.
func ExampleServeService() {
	svc := NewServeService(":36911", "lem", "/tmp/models")
	snap := svc.GetSnapshot()

	core.Println(snap.Up, snap.Managed)
	// Output:
	// false false
}
