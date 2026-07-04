//go:build linux && amd64

package hip

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestRegisterRocm_BackendRegistration_Good(t *testing.T) {
	backend, ok := inference.Get("rocm")
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "rocm", backend.Name())
	core.AssertEqual(t, ROCmAvailable(), backend.Available())
}

func TestRegisterRocm_ROCmAvailable_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	available := ROCmAvailable()
	core.AssertEqual(t, available, ROCmAvailable())
	core.AssertEqual(t, (&rocmBackend{}).Available(), available)
}

func TestRegisterRocm_ROCmAvailable_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	available := ROCmAvailable()
	core.AssertNotEqual(t, "", core.Sprintf("%v", available))
	core.AssertEqual(t, "linux", "linux")
}

func TestRegisterRocm_ROCmAvailable_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	first := ROCmAvailable()
	second := ROCmAvailable()
	core.AssertEqual(t, first, second)
}
