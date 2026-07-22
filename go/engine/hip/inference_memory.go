// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"sync/atomic"

	"dappco.re/go/inference/engine"
)

var (
	_ engine.MemoryReporter = (*hipTokenModel)(nil)

	hipMemoryPeakBytes atomic.Uint64
)

type hipMemoryDevice interface {
	DeviceInfo() nativeDeviceInfo
}

func hipDeviceAllocatedBytes(device hipMemoryDevice) uint64 {
	if device == nil {
		return 0
	}
	info := device.DeviceInfo()
	if info.MemoryBytes == 0 || info.FreeBytes > info.MemoryBytes {
		return 0
	}
	return info.MemoryBytes - info.FreeBytes
}

func hipMemoryWatermarkReset(device hipMemoryDevice) {
	hipMemoryPeakBytes.Store(hipDeviceAllocatedBytes(device))
}

func hipMemoryWatermarkSample(device hipMemoryDevice) {
	now := hipDeviceAllocatedBytes(device)
	for {
		peak := hipMemoryPeakBytes.Load()
		if now <= peak || hipMemoryPeakBytes.CompareAndSwap(peak, now) {
			return
		}
	}
}

func hipMemoryWatermarkPeak(device hipMemoryDevice) uint64 {
	hipMemoryWatermarkSample(device)
	return hipMemoryPeakBytes.Load()
}

func (m *hipTokenModel) ActiveMemoryBytes() uint64 {
	if m == nil || m.loaded == nil {
		return 0
	}
	return hipDeviceAllocatedBytes(m.loaded.driver)
}

func (m *hipTokenModel) PeakMemoryBytes() uint64 {
	if m == nil || m.loaded == nil {
		return 0
	}
	return hipMemoryWatermarkPeak(m.loaded.driver)
}
