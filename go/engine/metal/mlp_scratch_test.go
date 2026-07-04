// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
)

func TestMLPScratchComposedConstantAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	old := customLibraryLoaded
	customLibraryLoaded = false
	t.Cleanup(func() { customLibraryLoaded = old })
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF = 64, 128
	sc := newMLPScratch(dModel, dFF)
	if sc.c044 == nil || sc.c079 == nil || sc.c1 == nil || sc.c05 == nil {
		t.Fatal("newMLPScratch composed constants were not allocated")
	}
	forceNativeGC()

	allocs := testing.AllocsPerRun(5, func() {
		sc := newMLPScratch(dModel, dFF)
		if sc.c044 == nil || sc.c079 == nil || sc.c1 == nil || sc.c05 == nil {
			t.Fatal("newMLPScratch composed constants were not allocated")
		}
	})
	if allocs > 180 {
		t.Fatalf("newMLPScratch composed allocations = %.0f, want <= 180", allocs)
	}
}
