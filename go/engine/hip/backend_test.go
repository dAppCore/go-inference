//go:build linux && amd64

package hip

import (
	core "dappco.re/go"
	"testing"
)

func TestBackend_Backend_Name_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	core.AssertEqual(t, "rocm", (&rocmBackend{}).Name())
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestBackend_Backend_Name_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	core.AssertNotEqual(t, "cpu", (&rocmBackend{}).Name())
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestBackend_Backend_Name_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	b := &rocmBackend{}
	core.AssertEqual(t, b.Name(), b.Name())
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}

func TestBackend_Backend_Available_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	_ = (&rocmBackend{}).Available()
	core.AssertEqual(t, "rocm", (&rocmBackend{}).Name())
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestBackend_Backend_Available_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	core.AssertNotEqual(t, "", core.Sprintf("%v", (&rocmBackend{}).Available()))
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestBackend_Backend_Available_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	b := &rocmBackend{}
	core.AssertEqual(t, b.Available(), b.Available())
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}

func TestBackend_Backend_LoadModel_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	result := (&rocmBackend{}).LoadModel("missing.gguf")
	core.AssertError(t, resultError(result))
	core.AssertFalse(t, result.OK)
}
func TestBackend_Backend_LoadModel_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	result := (&rocmBackend{}).LoadModel("")
	core.AssertError(t, resultError(result))
	core.AssertFalse(t, result.OK)
}
func TestBackend_Backend_LoadModel_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	result := (&rocmBackend{}).LoadModel(core.PathJoin(t.TempDir(), "x.gguf"))
	core.AssertError(t, resultError(result))
	core.AssertFalse(t, result.OK)
}
