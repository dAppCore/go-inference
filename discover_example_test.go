package inference

import (
	"slices"

	core "dappco.re/go"
)

func ExampleDiscover() {
	baseResult := core.MkdirTemp("", "inference-discover-*")
	if !baseResult.OK {
		core.Println("tempdir failed")
		return
	}
	base := baseResult.Value.(string)
	defer core.RemoveAll(base)

	modelDir := core.Path(base, "gemma3-1b")
	core.MkdirAll(modelDir, 0o755)
	core.WriteFile(core.Path(modelDir, "config.json"), []byte(`{"model_type":"gemma3"}`), 0o644)
	core.WriteFile(core.Path(modelDir, "model.safetensors"), []byte("weights"), 0o644)

	models := slices.Collect(Discover(base))
	core.Println(models[0].ModelType, models[0].NumFiles)
	// Output: gemma3 1
}
