// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// fixtureRegistryComposedModelType is a model_type registered ONLY by this test file's init() — standing in
// for "any model package" landing a Composed hook after engine/metal ships. No name in load.go or anywhere
// else in this package spells it: TestLoadTokenModelDir_RegistryResolvesComposedArch proving it loads is the
// guard against ever reintroducing the hardcoded composed-arch model_type switch LoadTokenModelDirWithConfig
// used to carry (deleted once model/composed's registration covered every id it named).
const fixtureRegistryComposedModelType = "engine_metal_test_fixture_composed_arch"

// fixtureComposedTokenModel is the minimal model.TokenModel a Composed hook can return: Vocab is the one
// method this test reads back, proving the fixture's own hook — not some other registered arch — built it.
type fixtureComposedTokenModel struct{ vocab int }

func (f fixtureComposedTokenModel) Vocab() int                  { return f.vocab }
func (f fixtureComposedTokenModel) Embed(int32) ([]byte, error) { return nil, nil }
func (f fixtureComposedTokenModel) DecodeForward([][]byte) ([][]byte, error) {
	return nil, nil
}
func (f fixtureComposedTokenModel) Head([]byte) ([]byte, error) { return nil, nil }

func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{fixtureRegistryComposedModelType},
		Composed: func(map[string]safetensors.Tensor, []byte) (model.TokenModel, error) {
			return fixtureComposedTokenModel{vocab: 7}, nil
		},
	})
}

// TestLoadTokenModelDir_RegistryResolvesComposedArch is the registry-resolution proof goal 1 asks for: a
// composed model_type registered from a model package's init() (simulated by this file's own init() above)
// reaches LoadTokenModelDir and builds through its Composed hook with NO edit to load.go naming it — the
// same neutral path model.LoadComposedDir already gives every registered qwen hybrid.
func TestLoadTokenModelDir_RegistryResolvesComposedArch(t *testing.T) {
	dir := t.TempDir()
	cfg := core.Sprintf(`{"model_type":%q}`, fixtureRegistryComposedModelType)
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), cfg); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	blob, err := safetensors.Encode(map[string]safetensors.Tensor{})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}

	tm, err := LoadTokenModelDir(dir, 4)
	if err != nil {
		t.Fatalf("LoadTokenModelDir: %v", err)
	}
	if tm.Vocab() != 7 {
		t.Fatalf("Vocab() = %d, want 7 (the fixture Composed hook was not reached)", tm.Vocab())
	}
}
