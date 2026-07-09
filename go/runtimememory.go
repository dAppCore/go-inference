// SPDX-Licence-Identifier: EUPL-1.2

package inference

// Runtime memory read-back + maintenance — completing the runtime-memory
// contract (docs/engine-merge.md Tier-0: the composition core's memory verb
// set). [RuntimeMemoryLimiter] covers the limit-setting half; these cover
// observation (active/peak/cache bytes) and maintenance (clear caches, reset
// the peak counter), per-backend-dispatched exactly like
// [SetRuntimeMemoryLimits] and [BackendDeviceInfo].

// MemoryUsage reports a runtime's current allocator state. Zero-valued
// fields mean the runtime could not determine them.
type MemoryUsage struct {
	ActiveBytes uint64 `json:"active_bytes,omitempty"` // live allocations
	PeakBytes   uint64 `json:"peak_bytes,omitempty"`   // high-water mark since load or last reset
	CacheBytes  uint64 `json:"cache_bytes,omitempty"`  // allocator-retained reusable memory
}

// RuntimeMemoryReporter is implemented by native runtimes that expose
// allocator observability without callers importing the concrete runtime.
type RuntimeMemoryReporter interface {
	// RuntimeMemoryUsage reports the runtime's current allocator state.
	RuntimeMemoryUsage() MemoryUsage
}

// RuntimeMemoryMaintainer is implemented by native runtimes that support
// allocator maintenance between operations.
type RuntimeMemoryMaintainer interface {
	// ClearRuntimeCache releases the allocator's retained reusable memory.
	ClearRuntimeCache()
	// ResetRuntimePeakMemory resets the high-water mark to current usage.
	ResetRuntimePeakMemory()
}

// RuntimeMemoryUsage reports the allocator state of a registered backend when
// it supports [RuntimeMemoryReporter]. The boolean is false when the backend
// is not registered or does not expose observability.
//
//	if usage, ok := inference.RuntimeMemoryUsage("metal"); ok {
//	    fmt.Printf("active %d MiB, peak %d MiB\n", usage.ActiveBytes>>20, usage.PeakBytes>>20)
//	}
func RuntimeMemoryUsage(backendName string) (MemoryUsage, bool) {
	backend, ok := Get(backendName)
	if !ok {
		return MemoryUsage{}, false
	}
	reporter, ok := backend.(RuntimeMemoryReporter)
	if !ok {
		return MemoryUsage{}, false
	}
	return reporter.RuntimeMemoryUsage(), true
}

// ClearRuntimeCache asks a registered backend to release allocator-retained
// memory. The boolean is false when the backend is not registered or does
// not support maintenance.
func ClearRuntimeCache(backendName string) bool {
	backend, ok := Get(backendName)
	if !ok {
		return false
	}
	maintainer, ok := backend.(RuntimeMemoryMaintainer)
	if !ok {
		return false
	}
	maintainer.ClearRuntimeCache()
	return true
}

// ResetRuntimePeakMemory resets a registered backend's high-water mark. The
// boolean is false when the backend is not registered or does not support
// maintenance.
func ResetRuntimePeakMemory(backendName string) bool {
	backend, ok := Get(backendName)
	if !ok {
		return false
	}
	maintainer, ok := backend.(RuntimeMemoryMaintainer)
	if !ok {
		return false
	}
	maintainer.ResetRuntimePeakMemory()
	return true
}
