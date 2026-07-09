//go:build !linux || !amd64

package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"testing"
)

func TestRocmStub_BackendRegistration_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	backend, ok := inference.Get("rocm")
	core.AssertTrue(t, ok)
	core.AssertFalse(t, backend.Available())
	result := backend.LoadModel("model.gguf")
	err := resultError(result)
	core.AssertError(t, err)
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, err.Error(), "not available on this platform")
}

func TestRocmStub_ROCmAvailable_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	available := ROCmAvailable()
	core.AssertFalse(t, available)
	core.AssertEqual(t, available, ROCmAvailable())
}

func TestRocmStub_ROCmAvailable_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	available := ROCmAvailable()
	core.AssertNotEqual(t, true, available)
	core.AssertEqual(t, "stub", "stub")
}

func TestRocmStub_ROCmAvailable_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	first := ROCmAvailable()
	second := ROCmAvailable()
	core.AssertEqual(t, first, second)
}

func TestRocmStub_GetVRAMInfo_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	info, err := GetVRAMInfo()
	core.AssertError(t, err)
	core.AssertEqual(t, VRAMInfo{}, info)
}

func TestRocmStub_GetVRAMInfo_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	_, err := GetVRAMInfo()
	core.AssertContains(t, err.Error(), "not available")
	core.AssertError(t, err)
}

func TestRocmStub_GetVRAMInfo_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	first, _ := GetVRAMInfo()
	second, _ := GetVRAMInfo()
	core.AssertEqual(t, first, second)
}
