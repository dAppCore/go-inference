// SPDX-Licence-Identifier: EUPL-1.2

//go:build !linux || !amd64 || rocm_legacy_server

package hip

import "dappco.re/go/inference"

func rocmMachineDiscoveryDevice(_ *rocmBackend) inference.MachineDeviceInfo {
	return inference.MachineDeviceInfo{
		Name:         "rocm",
		Architecture: "portable_metadata",
		Labels: map[string]string{
			"backend":        "rocm",
			"machine_class":  "portable_metadata",
			"native_runtime": "portable_metadata",
		},
	}
}
