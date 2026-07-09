// SPDX-Licence-Identifier: EUPL-1.2

package inference

// DeviceInfo describes the accelerator a backend runs on. Engine-neutral —
// the Metal engine reports the Apple GPU today; hip/cuda backends report
// theirs. Zero-valued fields mean the backend could not determine them.
//
//	if info, ok := inference.BackendDeviceInfo("metal"); ok {
//	    fmt.Printf("%s (%s), %d GiB\n", info.Name, info.Architecture, info.MemorySize>>30)
//	}
type DeviceInfo struct {
	Name                         string // e.g. "Apple M3 Ultra"
	Architecture                 string // e.g. "applegpu_g15d"
	MemorySize                   uint64 // total device memory in bytes
	MaxBufferLength              uint64 // largest single allocation the device allows
	MaxRecommendedWorkingSetSize uint64 // device-recommended working-set ceiling
}

// DeviceInfoProvider is the optional capability a [Backend] implements when
// it can describe its accelerator without loading a model.
type DeviceInfoProvider interface {
	// DeviceInfo reports the accelerator this backend would run on.
	DeviceInfo() DeviceInfo
}

// BackendDeviceInfo reports the accelerator behind a registered backend when
// it supports [DeviceInfoProvider]. The boolean is false when the backend is
// not registered or does not expose device information.
//
//	info, ok := inference.BackendDeviceInfo("metal")
func BackendDeviceInfo(backendName string) (DeviceInfo, bool) {
	backend, ok := Get(backendName)
	if !ok {
		return DeviceInfo{}, false
	}
	provider, ok := backend.(DeviceInfoProvider)
	if !ok {
		return DeviceInfo{}, false
	}
	return provider.DeviceInfo(), true
}
