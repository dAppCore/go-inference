// SPDX-Licence-Identifier: EUPL-1.2

package inference

import "testing"

type memoryReportingBackend struct {
	stubBackend
	usage   MemoryUsage
	cleared bool
	reset   bool
}

func (b *memoryReportingBackend) RuntimeMemoryUsage() MemoryUsage { return b.usage }
func (b *memoryReportingBackend) ClearRuntimeCache()              { b.cleared = true }
func (b *memoryReportingBackend) ResetRuntimePeakMemory()         { b.reset = true }

func TestRuntimeMemory_RuntimeMemoryUsage_Good(t *testing.T) {
	resetBackends(t)
	want := MemoryUsage{ActiveBytes: 1 << 30, PeakBytes: 2 << 30, CacheBytes: 512 << 20}
	Register(&memoryReportingBackend{stubBackend: stubBackend{name: "metal", available: true}, usage: want})

	got, ok := RuntimeMemoryUsage("metal")

	checkTrue(t, ok)
	checkEqual(t, want, got)
}

func TestRuntimeMemory_RuntimeMemoryUsage_BadMissing(t *testing.T) {
	resetBackends(t)

	got, ok := RuntimeMemoryUsage("metal")

	checkFalse(t, ok)
	checkEqual(t, MemoryUsage{}, got)
}

func TestRuntimeMemory_RuntimeMemoryUsage_UglyUnsupported(t *testing.T) {
	resetBackends(t)
	Register(&stubBackend{name: "metal", available: true})

	_, ok := RuntimeMemoryUsage("metal")

	checkFalse(t, ok)
}

func TestRuntimeMemory_ClearRuntimeCache_Good(t *testing.T) {
	resetBackends(t)
	backend := &memoryReportingBackend{stubBackend: stubBackend{name: "metal", available: true}}
	Register(backend)

	checkTrue(t, ClearRuntimeCache("metal"))
	checkTrue(t, backend.cleared)
}

func TestRuntimeMemory_ResetRuntimePeakMemory_Good(t *testing.T) {
	resetBackends(t)
	backend := &memoryReportingBackend{stubBackend: stubBackend{name: "metal", available: true}}
	Register(backend)

	checkTrue(t, ResetRuntimePeakMemory("metal"))
	checkTrue(t, backend.reset)
}

func TestRuntimeMemory_Maintenance_BadMissing(t *testing.T) {
	resetBackends(t)

	checkFalse(t, ClearRuntimeCache("metal"))
	checkFalse(t, ResetRuntimePeakMemory("metal"))
}
