// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestDispatchProfileOneDispatch(t *testing.T) {
	requireNativeRuntime(t)

	encode, run, _, err := dispatchProfile(1, 64)
	if err != nil {
		t.Fatalf("dispatchProfile: %v", err)
	}
	if encode <= 0 {
		t.Fatalf("encode duration = %v, want positive", encode)
	}
	if run <= 0 {
		t.Fatalf("run duration = %v, want positive", run)
	}
}

func TestRebindCostProbe(t *testing.T) {
	requireNativeRuntime(t)

	d, err := rebindCostProbe(4)
	if err != nil {
		t.Fatalf("rebindCostProbe: %v", err)
	}
	if d <= 0 {
		t.Fatalf("rebind duration = %v, want positive", d)
	}
}
