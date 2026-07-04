// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && cgo && !rocm_legacy_server

package hip

import "testing"

var benchmarkCGOHIPLaunchArgModeSink cgoHIPLaunchArgMode
var benchmarkCGOHIPLaunchArgCopySink byte

func BenchmarkCGOHIPLaunchArgModeConfig_Hot(b *testing.B) {
	_ = cgoHIPLaunchArgModeConfig()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchmarkCGOHIPLaunchArgModeSink = cgoHIPLaunchArgModeConfig()
	}
}

func BenchmarkCGOHIPLaunchArgCopy_96B(b *testing.B) {
	host := make([]byte, 256)
	args := make([]byte, 96)
	for index := range args {
		args[index] = byte(index)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		copy(host, args)
	}
	benchmarkCGOHIPLaunchArgCopySink = host[len(args)-1]
}
