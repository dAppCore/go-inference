// SPDX-Licence-Identifier: EUPL-1.2

package inference

import "testing"

type deviceInfoBackend struct {
	stubBackend
	info DeviceInfo
}

func (backend *deviceInfoBackend) DeviceInfo() DeviceInfo { return backend.info }

func TestDevice_BackendDeviceInfo_Good(t *testing.T) {
	resetBackends(t)
	want := DeviceInfo{
		Name:                         "Apple M3 Ultra",
		Architecture:                 "applegpu_g15d",
		MemorySize:                   96 << 30,
		MaxBufferLength:              48 << 30,
		MaxRecommendedWorkingSetSize: 72 << 30,
	}
	Register(&deviceInfoBackend{stubBackend: stubBackend{name: "metal", available: true}, info: want})

	got, ok := BackendDeviceInfo("metal")

	checkTrue(t, ok)
	checkEqual(t, want, got)
}

func TestDevice_BackendDeviceInfo_BadMissing(t *testing.T) {
	resetBackends(t)

	got, ok := BackendDeviceInfo("metal")

	checkFalse(t, ok)
	checkEqual(t, DeviceInfo{}, got)
}

func TestDevice_BackendDeviceInfo_UglyUnsupported(t *testing.T) {
	resetBackends(t)
	Register(&stubBackend{name: "metal", available: true})

	got, ok := BackendDeviceInfo("metal")

	checkFalse(t, ok)
	checkEqual(t, DeviceInfo{}, got)
}
