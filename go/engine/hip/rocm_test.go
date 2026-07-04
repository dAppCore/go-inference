package hip

import (
	core "dappco.re/go"
	"testing"
)

func TestRocm_ModelInfoShape(t *testing.T) {
	info := ModelInfo{Name: "demo", Path: "model.gguf"}
	core.AssertEqual(t, "demo", info.Name)
	core.AssertContains(t, info.Path, ".gguf")
}

func TestRocm_VRAMInfoShape(t *testing.T) {
	info := VRAMInfo{Total: 8, Used: 3, Free: 5}
	core.AssertEqual(t, uint64(8), info.Total)
	core.AssertEqual(t, info.Total-info.Used, info.Free)
}
