// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/engine"
)

func TestHIPInferenceMemoryReporter_Good(t *testing.T) {
	driver := &fakeHIPDriver{device: nativeDeviceInfo{MemoryBytes: 16 << 30, FreeBytes: 6 << 30}}
	model := newHipTokenModel(&hipLoadedModel{driver: driver}, hipInferenceModelFixtureTokenizer(), "gemma4")
	var reporter engine.MemoryReporter = model

	hipMemoryWatermarkReset(driver)
	core.AssertEqual(t, uint64(10<<30), reporter.ActiveMemoryBytes())
	driver.device.FreeBytes = 4 << 30
	core.AssertEqual(t, uint64(12<<30), reporter.PeakMemoryBytes())
}

func TestHIPInferenceMemoryReporter_Bad(t *testing.T) {
	var model *hipTokenModel
	core.AssertEqual(t, uint64(0), model.ActiveMemoryBytes())
	core.AssertEqual(t, uint64(0), model.PeakMemoryBytes())
}

func TestHIPInferenceMemoryReporter_Ugly(t *testing.T) {
	driver := &fakeHIPDriver{device: nativeDeviceInfo{MemoryBytes: 8 << 30, FreeBytes: 9 << 30}}
	model := newHipTokenModel(&hipLoadedModel{driver: driver}, hipInferenceModelFixtureTokenizer(), "gemma4")

	hipMemoryWatermarkReset(driver)
	core.AssertEqual(t, uint64(0), model.ActiveMemoryBytes())
	core.AssertEqual(t, uint64(0), model.PeakMemoryBytes())
}
