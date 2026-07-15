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

func TestCGOHIPKernelBatchLayout_Good(t *testing.T) {
	configs := []hipKernelLaunchConfig{
		{Name: "first", Args: make([]byte, 3), GridX: 1, GridY: 1, GridZ: 1, BlockX: 1, BlockY: 1, BlockZ: 1},
		{Name: "second", Args: make([]byte, 17), GridX: 1, GridY: 1, GridZ: 1, BlockX: 1, BlockY: 1, BlockZ: 1},
		{Name: "third", Args: make([]byte, 184), GridX: 1, GridY: 1, GridZ: 1, BlockX: 1, BlockY: 1, BlockZ: 1},
	}
	offsets, total, err := cgoHIPKernelBatchLayout(configs, nil)
	if err != nil {
		t.Fatalf("cgoHIPKernelBatchLayout: %v", err)
	}
	wantOffsets := []uint64{0, 16, 48}
	if len(offsets) != len(wantOffsets) {
		t.Fatalf("offset count = %d, want %d", len(offsets), len(wantOffsets))
	}
	for index := range wantOffsets {
		if offsets[index] != wantOffsets[index] {
			t.Fatalf("offset[%d] = %d, want %d", index, offsets[index], wantOffsets[index])
		}
	}
	if total != 232 {
		t.Fatalf("total = %d, want 232", total)
	}
}

func TestCGOHIPKernelBatchLayout_Bad(t *testing.T) {
	_, _, err := cgoHIPKernelBatchLayout([]hipKernelLaunchConfig{{}}, nil)
	if err == nil {
		t.Fatal("empty launch packet unexpectedly produced a batch layout")
	}
}

func TestCGOHIPKernelBatchOwnerMatches_Good(t *testing.T) {
	owner := cgoHIPDriver{kernelModulePath: "/tmp/first.hsaco"}
	if !cgoHIPKernelBatchOwnerMatches(owner, "/tmp/first.hsaco", true, owner, "/tmp/first.hsaco") {
		t.Fatal("equal driver and resolved module path did not match")
	}
	if cgoHIPKernelBatchOwnerMatches(owner, "/tmp/first.hsaco", true, cgoHIPDriver{kernelModulePath: "/tmp/second.hsaco"}, "/tmp/second.hsaco") {
		t.Fatal("different driver and resolved module path unexpectedly matched")
	}
	if cgoHIPKernelBatchOwnerMatches(owner, "/tmp/first.hsaco", false, owner, "/tmp/first.hsaco") {
		t.Fatal("unset owner unexpectedly matched")
	}
}
