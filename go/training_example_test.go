package inference

import (
	core "dappco.re/go"
)

func ExampleDefaultLoRAConfig() {
	cfg := DefaultLoRAConfig()
	core.Println(cfg.Rank, cfg.Alpha, cfg.TargetKeys)
	// Output: 8 16 [q_proj v_proj]
}

func ExampleLoadTrainable() {
	resetExampleBackends()
	Register(&trainableBackend{name: "metal", available: true})

	result := LoadTrainable("/models/gemma3")
	model := result.Value.(TrainableModel)
	core.Println(result.OK, model.NumLayers())
	model.Close()
	// Output: true 26
}
