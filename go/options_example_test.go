package inference

import (
	core "dappco.re/go"
)

func ExampleDefaultGenerateConfig() {
	cfg := DefaultGenerateConfig()
	core.Println(cfg.MaxTokens, cfg.Temperature, cfg.RepeatPenalty)
	// Output: 256 0 1
}

func ExampleWithMaxTokens() {
	cfg := ApplyGenerateOpts([]GenerateOption{WithMaxTokens(64)})
	core.Println(cfg.MaxTokens)
	// Output: 64
}

func ExampleWithTemperature() {
	cfg := ApplyGenerateOpts([]GenerateOption{WithTemperature(0.7)})
	core.Println(cfg.Temperature)
	// Output: 0.7
}

func ExampleWithTopK() {
	cfg := ApplyGenerateOpts([]GenerateOption{WithTopK(40)})
	core.Println(cfg.TopK)
	// Output: 40
}

func ExampleWithTopP() {
	cfg := ApplyGenerateOpts([]GenerateOption{WithTopP(0.9)})
	core.Println(cfg.TopP)
	// Output: 0.9
}

func ExampleWithStopTokens() {
	cfg := ApplyGenerateOpts([]GenerateOption{WithStopTokens(2, 1)})
	core.Println(cfg.StopTokens)
	// Output: [2 1]
}

func ExampleWithRepeatPenalty() {
	cfg := ApplyGenerateOpts([]GenerateOption{WithRepeatPenalty(1.1)})
	core.Println(cfg.RepeatPenalty)
	// Output: 1.1
}

func ExampleWithLogits() {
	cfg := ApplyGenerateOpts([]GenerateOption{WithLogits()})
	core.Println(cfg.ReturnLogits)
	// Output: true
}

func ExampleApplyGenerateOpts() {
	cfg := ApplyGenerateOpts([]GenerateOption{
		WithMaxTokens(32),
		WithTemperature(0.5),
		WithStopTokens(2),
	})
	core.Println(cfg.MaxTokens, cfg.Temperature, cfg.StopTokens)
	// Output: 32 0.5 [2]
}

func ExampleWithBackend() {
	cfg := ApplyLoadOpts([]LoadOption{WithBackend("metal")})
	core.Println(cfg.Backend)
	// Output: metal
}

func ExampleWithContextLen() {
	cfg := ApplyLoadOpts([]LoadOption{WithContextLen(4096)})
	core.Println(cfg.ContextLen)
	// Output: 4096
}

func ExampleWithGPULayers() {
	cfg := ApplyLoadOpts([]LoadOption{WithGPULayers(24)})
	core.Println(cfg.GPULayers)
	// Output: 24
}

func ExampleWithParallelSlots() {
	cfg := ApplyLoadOpts([]LoadOption{WithParallelSlots(2)})
	core.Println(cfg.ParallelSlots)
	// Output: 2
}

func ExampleWithAdapterPath() {
	cfg := ApplyLoadOpts([]LoadOption{WithAdapterPath("adapters/domain")})
	core.Println(cfg.AdapterPath)
	// Output: adapters/domain
}

func ExampleApplyLoadOpts() {
	cfg := ApplyLoadOpts([]LoadOption{
		WithBackend("rocm"),
		WithContextLen(8192),
		WithGPULayers(32),
		WithParallelSlots(4),
		WithAdapterPath("adapters/lora"),
	})
	core.Println(cfg.Backend, cfg.ContextLen, cfg.GPULayers, cfg.ParallelSlots, cfg.AdapterPath)
	// Output: rocm 8192 32 4 adapters/lora
}
