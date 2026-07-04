package hip

import (
	core "dappco.re/go"
	"testing"
)

func TestDiscover_DiscoverModels_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	dir := t.TempDir()
	models, err := DiscoverModels(dir)
	core.AssertNoError(t, err)
	core.AssertEqual(t, 0, len(models))
}

func TestDiscover_DiscoverModels_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	models, err := DiscoverModels(core.PathJoin(t.TempDir(), "missing"))
	core.AssertNoError(t, err)
	core.AssertEqual(t, 0, len(models))
}

func TestDiscover_DiscoverModels_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	models, err := DiscoverModels("bad[")
	core.AssertError(t, err)
	core.AssertNil(t, models)
}
