// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestProfileForwardStateIsPackageLocal(t *testing.T) {
	oldEnabled, oldGPU := profileForward, profForwardGPUSec
	defer func() {
		profileForward, profForwardGPUSec = oldEnabled, oldGPU
	}()

	profileForward = true
	profForwardGPUSec = 1.25
	if !profileForward {
		t.Fatal("profileForward was not set")
	}
	if profForwardGPUSec != 1.25 {
		t.Fatalf("profForwardGPUSec = %v, want 1.25", profForwardGPUSec)
	}
}
