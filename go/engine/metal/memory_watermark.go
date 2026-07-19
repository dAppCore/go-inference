// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "sync/atomic"

// The engine.MemoryReporter capability for the metal engine (mantis #1843):
// Active reads MTLDevice.currentAllocatedSize live; Peak is a device-global
// high-water the prefill/append seams reset at operation entry and sample at
// exit, folded with "now" at read time. The watermark tracks MTLBuffer
// allocation (weights, caches, the array pool) — the number a resident-memory
// budget cares about — not transient threadgroup memory inside a kernel.
// Device-global is the honest scope: allocation is a device-wide fact, so
// concurrent generations share the watermark.

var memPeakBytes atomic.Uint64

// deviceAllocatedBytes reads the device's current MTLBuffer allocation; zero
// before the device initialises (or off-device builds).
func deviceAllocatedBytes() uint64 {
	if device.ID == 0 {
		return 0
	}
	return uint64(device.CurrentAllocatedSize())
}

// memWatermarkReset starts an operation's high-water at the current
// allocation. Called at the prefill/append seams every generation crosses.
func memWatermarkReset() {
	memPeakBytes.Store(deviceAllocatedBytes())
}

// memWatermarkSample folds the current allocation into the high-water.
func memWatermarkSample() {
	now := deviceAllocatedBytes()
	for {
		peak := memPeakBytes.Load()
		if now <= peak || memPeakBytes.CompareAndSwap(peak, now) {
			return
		}
	}
}

// memWatermarkPeak returns the high-water with "now" folded in — metrics read
// this right after decode, so the post-decode state always counts.
func memWatermarkPeak() uint64 {
	memWatermarkSample()
	return memPeakBytes.Load()
}

// ActiveMemoryBytes implements engine.MemoryReporter.
func (m *NativeTokenModel) ActiveMemoryBytes() uint64 { return deviceAllocatedBytes() }

// PeakMemoryBytes implements engine.MemoryReporter.
func (m *NativeTokenModel) PeakMemoryBytes() uint64 { return memWatermarkPeak() }

// ActiveMemoryBytes implements engine.MemoryReporter.
func (m *sessionTextModel) ActiveMemoryBytes() uint64 { return deviceAllocatedBytes() }

// PeakMemoryBytes implements engine.MemoryReporter.
func (m *sessionTextModel) PeakMemoryBytes() uint64 { return memWatermarkPeak() }
