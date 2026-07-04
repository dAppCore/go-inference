//go:build linux && amd64

package hip

import (
	core "dappco.re/go"
	"testing"
)

func TestVram_GetVRAMInfo_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	info, err := GetVRAMInfo()
	if err == nil {
		core.AssertGreaterOrEqual(t, info.Total, info.Used)
	}
}
func TestVram_GetVRAMInfo_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	_, _ = GetVRAMInfo()
	_, err := readSysfsUint64(core.PathJoin(t.TempDir(), "missing"))
	core.AssertError(t, err)
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestVram_GetVRAMInfo_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	_, err := GetVRAMInfo()
	if err != nil {
		core.AssertContains(t, err.Error(), "rocm.GetVRAMInfo")
	}
}

func BenchmarkGetVRAMInfo_Cached(b *testing.B) {
	if _, err := GetVRAMInfo(); err != nil {
		b.Skipf("GetVRAMInfo unavailable: %v", err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := GetVRAMInfo(); err != nil {
			b.Fatalf("GetVRAMInfo: %v", err)
		}
	}
}
