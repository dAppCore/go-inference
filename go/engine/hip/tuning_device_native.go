// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"strconv"

	"dappco.re/go/inference"
)

func rocmMachineDiscoveryDevice(b *rocmBackend) inference.MachineDeviceInfo {
	var device nativeDeviceInfo
	if b != nil {
		device = b.nativeRuntime().DeviceInfo()
	} else {
		device = newSystemNativeRuntime().DeviceInfo()
	}
	labels := map[string]string{
		"backend":        "rocm",
		"machine_class":  "rocm",
		"native_runtime": "true",
	}
	if device.Driver != "" {
		labels["driver"] = device.Driver
	}
	if device.FreeBytes > 0 {
		labels["free_bytes"] = strconv.FormatUint(device.FreeBytes, 10)
	}
	return inference.MachineDeviceInfo{
		Name:                         device.Name,
		Architecture:                 firstNonEmptyString(device.Name, "rocm"),
		MemorySize:                   device.MemoryBytes,
		MaxRecommendedWorkingSetSize: device.FreeBytes,
		Labels:                       labels,
	}
}
