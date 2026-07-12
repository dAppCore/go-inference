// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the runtime-memory observability dispatch. Per AX-11 — these
// are on-demand maintenance/observability endpoints (a monitoring poll, an
// admin "clear cache" action), not a per-token path, so the multiplier is low;
// the bench records the dispatch cost itself: a registry lookup plus an
// interface type-assertion plus the delegated call, isolated from any real
// backend by a fake reporter.
//
// Run:    go test -bench=RuntimeMemory -benchmem -run='^$' .
package inference

import "testing"

var runtimeMemBenchSinkUsage MemoryUsage
var runtimeMemBenchSinkBool bool

func BenchmarkRuntimeMemoryUsage_Hit(b *testing.B) {
	Register(&memoryReportingBackend{
		stubBackend: stubBackend{name: "bench-mem", available: true},
		usage:       MemoryUsage{ActiveBytes: 1 << 30, PeakBytes: 2 << 30, CacheBytes: 512 << 20},
	})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runtimeMemBenchSinkUsage, runtimeMemBenchSinkBool = RuntimeMemoryUsage("bench-mem")
	}
}

// Miss path: unregistered backend — the early (zero, false) return, the shape
// a poll for a not-yet-loaded engine takes.
func BenchmarkRuntimeMemoryUsage_Miss(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runtimeMemBenchSinkUsage, runtimeMemBenchSinkBool = RuntimeMemoryUsage("bench-absent")
	}
}

func BenchmarkClearRuntimeCache_Hit(b *testing.B) {
	Register(&memoryReportingBackend{stubBackend: stubBackend{name: "bench-clear", available: true}})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runtimeMemBenchSinkBool = ClearRuntimeCache("bench-clear")
	}
}
