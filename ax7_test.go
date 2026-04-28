package inference

import (
	"slices"

	core "dappco.re/go"
)

func TestOptions_DefaultGenerateConfig_Bad(t *core.T) {
	cfg := DefaultGenerateConfig()
	cfg.MaxTokens = 0

	core.AssertEqual(t, 0, cfg.MaxTokens)
	core.AssertEqual(t, float32(0), cfg.Temperature)
	core.AssertEqual(t, float32(1), cfg.RepeatPenalty)
}

func TestOptions_DefaultGenerateConfig_Ugly(t *core.T) {
	cfg := DefaultGenerateConfig()
	cfg.StopTokens = append(cfg.StopTokens, 1)

	core.AssertEqual(t, []int32{1}, cfg.StopTokens)
	core.AssertFalse(t, DefaultGenerateConfig().ReturnLogits)
	core.AssertNil(t, DefaultGenerateConfig().StopTokens)
}

func TestOptions_WithMaxTokens_Ugly(t *core.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithMaxTokens(1 << 20)})
	def := DefaultGenerateConfig()

	core.AssertEqual(t, 1<<20, cfg.MaxTokens)
	core.AssertEqual(t, def.Temperature, cfg.Temperature)
	core.AssertEqual(t, def.RepeatPenalty, cfg.RepeatPenalty)
}

func TestOptions_WithTemperature_Ugly(t *core.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithTemperature(2.5)})
	def := DefaultGenerateConfig()

	core.AssertInDelta(t, 2.5, float64(cfg.Temperature), 0.0001)
	core.AssertEqual(t, def.MaxTokens, cfg.MaxTokens)
	core.AssertFalse(t, cfg.ReturnLogits)
}

func TestOptions_WithTopK_Ugly(t *core.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithTopK(1 << 16)})
	def := DefaultGenerateConfig()

	core.AssertEqual(t, 1<<16, cfg.TopK)
	core.AssertEqual(t, def.MaxTokens, cfg.MaxTokens)
	core.AssertEqual(t, def.TopP, cfg.TopP)
}

func TestOptions_WithLogits_Bad(t *core.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithLogits(), nil})
	def := DefaultGenerateConfig()

	core.AssertTrue(t, cfg.ReturnLogits)
	core.AssertEqual(t, def.MaxTokens, cfg.MaxTokens)
	core.AssertEqual(t, def.RepeatPenalty, cfg.RepeatPenalty)
}

func TestOptions_WithLogits_Ugly(t *core.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithLogits(), WithLogits()})
	def := DefaultGenerateConfig()

	core.AssertTrue(t, cfg.ReturnLogits)
	core.AssertEqual(t, def.TopK, cfg.TopK)
	core.AssertNil(t, cfg.StopTokens)
}

func TestOptions_ApplyGenerateOpts_Bad(t *core.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{nil, nil})
	def := DefaultGenerateConfig()

	core.AssertEqual(t, def, cfg)
	core.AssertFalse(t, cfg.ReturnLogits)
	core.AssertNil(t, cfg.StopTokens)
}

func TestOptions_ApplyLoadOpts_Good(t *core.T) {
	cfg := ApplyLoadOpts([]LoadOption{
		WithBackend("metal"),
		WithContextLen(4096),
		WithGPULayers(24),
		WithParallelSlots(2),
		WithAdapterPath("adapters/domain"),
	})

	core.AssertEqual(t, "metal", cfg.Backend)
	core.AssertEqual(t, 4096, cfg.ContextLen)
	core.AssertEqual(t, 24, cfg.GPULayers)
	core.AssertEqual(t, 2, cfg.ParallelSlots)
	core.AssertEqual(t, "adapters/domain", cfg.AdapterPath)
}

func TestOptions_ApplyLoadOpts_Bad(t *core.T) {
	cfg := ApplyLoadOpts([]LoadOption{nil})
	def := ApplyLoadOpts(nil)

	core.AssertEqual(t, def, cfg)
	core.AssertEqual(t, -1, cfg.GPULayers)
	core.AssertEqual(t, "", cfg.Backend)
}

func TestOptions_WithBackend_Ugly(t *core.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithBackend("metal"), WithBackend("rocm")})
	def := ApplyLoadOpts(nil)

	core.AssertEqual(t, "rocm", cfg.Backend)
	core.AssertEqual(t, def.ContextLen, cfg.ContextLen)
	core.AssertEqual(t, def.GPULayers, cfg.GPULayers)
}

func TestOptions_WithContextLen_Bad(t *core.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithContextLen(0)})
	def := ApplyLoadOpts(nil)

	core.AssertEqual(t, 0, cfg.ContextLen)
	core.AssertEqual(t, def.GPULayers, cfg.GPULayers)
	core.AssertEqual(t, def.Backend, cfg.Backend)
}

func TestOptions_WithContextLen_Ugly(t *core.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithContextLen(-4096)})
	def := ApplyLoadOpts(nil)

	core.AssertEqual(t, -4096, cfg.ContextLen)
	core.AssertEqual(t, def.ParallelSlots, cfg.ParallelSlots)
	core.AssertEqual(t, def.AdapterPath, cfg.AdapterPath)
}

func TestOptions_WithGPULayers_Bad(t *core.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithGPULayers(-2)})
	def := ApplyLoadOpts(nil)

	core.AssertEqual(t, -2, cfg.GPULayers)
	core.AssertEqual(t, def.ContextLen, cfg.ContextLen)
	core.AssertEqual(t, def.AdapterPath, cfg.AdapterPath)
}

func TestOptions_WithParallelSlots_Bad(t *core.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithParallelSlots(-1)})
	def := ApplyLoadOpts(nil)

	core.AssertEqual(t, -1, cfg.ParallelSlots)
	core.AssertEqual(t, def.Backend, cfg.Backend)
	core.AssertEqual(t, def.GPULayers, cfg.GPULayers)
}

func TestOptions_WithParallelSlots_Ugly(t *core.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithParallelSlots(128)})
	def := ApplyLoadOpts(nil)

	core.AssertEqual(t, 128, cfg.ParallelSlots)
	core.AssertEqual(t, def.ContextLen, cfg.ContextLen)
	core.AssertEqual(t, def.AdapterPath, cfg.AdapterPath)
}

func TestOptions_WithAdapterPath_Ugly(t *core.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithAdapterPath(""), WithAdapterPath("/tmp/adapter")})
	def := ApplyLoadOpts(nil)

	core.AssertEqual(t, "/tmp/adapter", cfg.AdapterPath)
	core.AssertEqual(t, def.ContextLen, cfg.ContextLen)
	core.AssertEqual(t, def.GPULayers, cfg.GPULayers)
}

func TestInference_Discover_Good(t *core.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "gemma3-1b"), map[string]any{"model_type": "gemma3"}, 2)

	models := slices.Collect(Discover(base))
	core.AssertLen(t, models, 1)
	core.AssertEqual(t, "gemma3", models[0].ModelType)
	core.AssertEqual(t, 2, models[0].NumFiles)
}

func TestInference_Discover_Bad(t *core.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "empty-model"), map[string]any{"model_type": "gemma3"}, 0)

	models := slices.Collect(Discover(base))
	core.AssertEmpty(t, models)
	core.AssertEqual(t, 0, len(models))
}

func TestInference_Discover_Ugly(t *core.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "no-type"), map[string]any{"vocab_size": 32000}, 1)

	models := slices.Collect(Discover(base))
	core.AssertLen(t, models, 1)
	core.AssertEqual(t, "", models[0].ModelType)
	core.AssertEqual(t, 0, models[0].QuantBits)
}

func TestInference_AttentionSnapshot_HasQueries_Good(t *core.T) {
	snap := &AttentionSnapshot{Queries: [][][]float32{{{1, 2, 3}}}}
	got := snap.HasQueries()

	core.AssertTrue(t, got)
	core.AssertLen(t, snap.Queries, 1)
	core.AssertEqual(t, float32(1), snap.Queries[0][0][0])
}

func TestInference_AttentionSnapshot_HasQueries_Bad(t *core.T) {
	snap := &AttentionSnapshot{Queries: nil}
	got := snap.HasQueries()

	core.AssertFalse(t, got)
	core.AssertNil(t, snap.Queries)
	core.AssertEqual(t, 0, len(snap.Queries))
}

func TestInference_AttentionSnapshot_HasQueries_Ugly(t *core.T) {
	var snap *AttentionSnapshot
	got := snap.HasQueries()

	core.AssertFalse(t, got)
	core.AssertNil(t, snap)
	core.AssertNotPanics(t, func() { _ = snap.HasQueries() })
}

func TestInference_Register_Bad(t *core.T) {
	resetBackends(t)
	Register(nil)

	core.AssertEmpty(t, List())
	core.AssertLen(t, List(), 0)
	core.AssertFalse(t, slices.Contains(List(), "nil"))
}

func TestInference_Get_Ugly(t *core.T) {
	resetBackends(t)
	Register(&stubBackend{name: "", available: true})

	b, ok := Get("")
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "", b.Name())
	core.AssertTrue(t, b.Available())
}

func TestInference_List_Good(t *core.T) {
	resetBackends(t)
	Register(&stubBackend{name: "beta", available: true})
	Register(&stubBackend{name: "alpha", available: true})

	names := List()
	core.AssertEqual(t, []string{"alpha", "beta"}, names)
	core.AssertLen(t, names, 2)
}

func TestInference_List_Bad(t *core.T) {
	resetBackends(t)
	names := List()

	core.AssertEmpty(t, names)
	core.AssertLen(t, names, 0)
	core.AssertNil(t, names)
}

func TestInference_List_Ugly(t *core.T) {
	resetBackends(t)
	Register(&stubBackend{name: "alpha", available: true})

	names := List()
	names[0] = "mutated"
	core.AssertEqual(t, []string{"alpha"}, List())
	core.AssertNotEqual(t, names[0], List()[0])
}

func TestInference_All_Bad(t *core.T) {
	resetBackends(t)
	count := 0

	for range All() {
		count++
	}
	core.AssertEqual(t, 0, count)
	core.AssertEmpty(t, List())
}

func TestInference_All_Ugly(t *core.T) {
	resetBackends(t)
	Register(&stubBackend{name: "first", available: true})
	Register(&stubBackend{name: "second", available: true})

	count := 0
	for range All() {
		count++
		break
	}
	core.AssertEqual(t, 1, count)
}

func TestInference_Default_Good(t *core.T) {
	resetBackends(t)
	Register(&stubBackend{name: "rocm", available: true})
	Register(&stubBackend{name: "metal", available: true})

	b, err := Default()
	core.AssertNoError(t, err)
	core.AssertEqual(t, "metal", b.Name())
}

func TestInference_Default_Bad(t *core.T) {
	resetBackends(t)
	b, err := Default()

	core.AssertError(t, err)
	core.AssertNil(t, b)
	core.AssertContains(t, err.Error(), "no backends registered")
}

func TestInference_Default_Ugly(t *core.T) {
	resetBackends(t)
	Register(&stubBackend{name: "metal", available: false})
	Register(&stubBackend{name: "zz_custom", available: true})

	b, err := Default()
	core.AssertNoError(t, err)
	core.AssertEqual(t, "zz_custom", b.Name())
}

func TestInference_LoadModel_Good(t *core.T) {
	resetBackends(t)
	Register(&stubBackend{name: "metal", available: true})

	model, err := LoadModel("/models/gemma3")
	core.AssertNoError(t, err)
	core.AssertNotNil(t, model)
	core.AssertEqual(t, "stub", model.ModelType())
	core.AssertNoError(t, model.Close())
}

func TestInference_LoadModel_Bad(t *core.T) {
	resetBackends(t)
	model, err := LoadModel("/models/gemma3")

	core.AssertError(t, err)
	core.AssertNil(t, model)
	core.AssertContains(t, err.Error(), "no backends registered")
}

func TestInference_LoadModel_Ugly(t *core.T) {
	resetBackends(t)
	Register(&stubBackend{name: "metal", available: true, nilModel: true})

	model, err := LoadModel("/models/gemma3")
	core.AssertError(t, err)
	core.AssertNil(t, model)
	core.AssertContains(t, err.Error(), "returned a nil model")
}

func TestTraining_DefaultLoRAConfig_Bad(t *core.T) {
	cfg := DefaultLoRAConfig()
	cfg.Rank = 0

	core.AssertEqual(t, 0, cfg.Rank)
	core.AssertEqual(t, float32(16), cfg.Alpha)
	core.AssertEqual(t, []string{"q_proj", "v_proj"}, cfg.TargetKeys)
}

func TestTraining_DefaultLoRAConfig_Ugly(t *core.T) {
	cfg := DefaultLoRAConfig()
	cfg.TargetKeys = append(cfg.TargetKeys, "k_proj")

	core.AssertEqual(t, []string{"q_proj", "v_proj", "k_proj"}, cfg.TargetKeys)
	core.AssertEqual(t, []string{"q_proj", "v_proj"}, DefaultLoRAConfig().TargetKeys)
	core.AssertFalse(t, cfg.BFloat16)
}

func TestTraining_LoadTrainable_Bad(t *core.T) {
	resetBackends(t)
	Register(&stubBackend{name: "metal", available: true})

	model, err := LoadTrainable("/models/gemma3")
	core.AssertError(t, err)
	core.AssertNil(t, model)
	core.AssertContains(t, err.Error(), "does not support training")
}

func TestTraining_LoadTrainable_Ugly(t *core.T) {
	resetBackends(t)
	Register(&trainableBackend{name: "metal", available: true})

	model, err := LoadTrainable("")
	core.AssertNoError(t, err)
	core.AssertNotNil(t, model)
	core.AssertNoError(t, model.Close())
}

func TestInference_Register_Ugly(t *core.T) {
	resetBackends(t)
	Register(&stubBackend{name: "dup", available: false})
	Register(&stubBackend{name: "dup", available: true})

	b, ok := Get("dup")
	core.AssertTrue(t, ok)
	core.AssertTrue(t, b.Available())
	core.AssertEqual(t, []string{"dup"}, List())
}
