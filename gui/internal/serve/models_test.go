// SPDX-Licence-Identifier: EUPL-1.2

package serve

import (
	core "dappco.re/go"
)

// fakeModelTree writes a discoverable model (config.json + one .safetensors)
// named name under a fresh temp root, and returns that root for discovery.
func fakeModelTree(t *core.T, name, configJSON string) string {
	root := t.TempDir()
	dir := core.PathJoin(root, name)
	if r := core.MkdirAll(dir, 0o755); !r.OK {
		t.Fatal("mkdir model dir: " + r.Error())
	}
	if r := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(configJSON), 0o644); !r.OK {
		t.Fatal("write config.json: " + r.Error())
	}
	if r := core.WriteFile(core.PathJoin(dir, "model.safetensors"), []byte("weights"), 0o644); !r.OK {
		t.Fatal("write safetensors: " + r.Error())
	}
	return root
}

func TestModels_DiscoverModels_Good(t *core.T) {
	root := fakeModelTree(t, "gemma-4-e2b-it-4bit", `{"model_type":"gemma3","quantization":{"bits":4,"group_size":64}}`)

	models := DiscoverModels(root)

	core.AssertLen(t, models, 1)
	core.AssertEqual(t, "gemma-4-e2b-it-4bit", models[0].Name)
	core.AssertEqual(t, "gemma3", models[0].Type)
	core.AssertEqual(t, 4, models[0].Quant)
	core.AssertContains(t, models[0].Path, "gemma-4-e2b-it-4bit")
}

func TestModels_DiscoverModels_Bad(t *core.T) {
	models := DiscoverModels("")

	core.AssertLen(t, models, 0)
}

func TestModels_DiscoverModels_Ugly(t *core.T) {
	// A directory with no config.json is not a model — the walk finds nothing.
	root := t.TempDir()
	if r := core.WriteFile(core.PathJoin(root, "README.md"), []byte("not a model"), 0o644); !r.OK {
		t.Fatal("write junk: " + r.Error())
	}

	models := DiscoverModels(root)

	core.AssertLen(t, models, 0)
}

func TestModels_DefaultModelsDir_Good(t *core.T) {
	dir := DefaultModelsDir()

	core.AssertTrue(t, core.HasSuffix(dir, "Lethean/lem/models"))
}

func TestModels_DefaultModelsDir_Bad(t *core.T) {
	dir := DefaultModelsDir()

	core.AssertContains(t, dir, "Lethean")
}

func TestModels_DefaultModelsDir_Ugly(t *core.T) {
	// Deterministic — repeated calls agree.
	core.AssertEqual(t, DefaultModelsDir(), DefaultModelsDir())
}
