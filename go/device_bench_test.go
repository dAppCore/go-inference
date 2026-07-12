// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the device-info dispatch. Per AX-11 — BackendDeviceInfo is an
// on-demand query (a status surface, a boot notice), not a per-token path, so
// the multiplier is low; the bench isolates the dispatch cost — registry
// lookup plus interface type-assertion plus the delegated call — behind a fake
// provider so no GPU is touched.
//
// Run:    go test -bench=BackendDeviceInfo -benchmem -run='^$' .
package inference

import "testing"

var deviceBenchSinkInfo DeviceInfo
var deviceBenchSinkBool bool

func BenchmarkBackendDeviceInfo_Hit(b *testing.B) {
	Register(&deviceInfoBackend{
		stubBackend: stubBackend{name: "bench-dev", available: true},
		info:        DeviceInfo{Name: "Apple M3 Ultra", Architecture: "applegpu_g15d", MemorySize: 64 << 30},
	})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		deviceBenchSinkInfo, deviceBenchSinkBool = BackendDeviceInfo("bench-dev")
	}
}

func BenchmarkBackendDeviceInfo_Miss(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		deviceBenchSinkInfo, deviceBenchSinkBool = BackendDeviceInfo("bench-absent")
	}
}
