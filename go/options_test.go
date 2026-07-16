package inference

import (
	"testing"

	core "dappco.re/go"
)

// --- DefaultGenerateConfig stability ---

func TestOptions_DefaultGenerateConfig_Good_Idempotent(t *testing.T) {
	firstConfig := DefaultGenerateConfig()
	secondConfig := DefaultGenerateConfig()
	checkEqual(t, firstConfig, secondConfig)
}

// --- GenerateConfig defaults ---

func TestOptions_DefaultGenerateConfig_Good(t *testing.T) {
	cfg := DefaultGenerateConfig()
	checkEqual(t, 0, cfg.MaxTokens) // not defaulted — 0 resolves to the model's context at generate time
	checkEqual(t, float32(0.0), cfg.Temperature)
	checkEqual(t, 0, cfg.TopK)
	checkEqual(t, float32(0.0), cfg.TopP)
	checkNil(t, cfg.StopTokens)
	checkEqual(t, float32(1.0), cfg.RepeatPenalty)
	checkFalse(t, cfg.ReturnLogits)
}

// --- WithMaxTokens ---

func TestOptions_WithMaxTokens_Good(t *testing.T) {
	tests := []struct {
		name string
		val  int
		want int
	}{
		{"small", 32, 32},
		{"medium", 512, 512},
		{"large", 4096, 4096},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ApplyGenerateOpts([]GenerateOption{WithMaxTokens(tt.val)})
			checkEqual(t, tt.want, cfg.MaxTokens)
		})
	}
}

func TestOptions_WithMaxTokens_Bad(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithMaxTokens(0)})
	checkEqual(t, 0, cfg.MaxTokens)

	cfg = ApplyGenerateOpts([]GenerateOption{WithMaxTokens(-1)})
	checkEqual(t, -1, cfg.MaxTokens)
}

func TestOptions_WithMaxTokens_Good_OtherFieldsUnchanged(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithMaxTokens(512)})
	def := DefaultGenerateConfig()
	checkEqual(t, 512, cfg.MaxTokens)
	checkEqual(t, def.Temperature, cfg.Temperature)
	checkEqual(t, def.TopK, cfg.TopK)
	checkEqual(t, def.TopP, cfg.TopP)
	checkNil(t, cfg.StopTokens)
	checkEqual(t, def.RepeatPenalty, cfg.RepeatPenalty)
	checkEqual(t, def.ReturnLogits, cfg.ReturnLogits)
}

// --- WithTemperature ---

func TestOptions_WithTemperature_Good(t *testing.T) {
	tests := []struct {
		name string
		val  float32
		want float32
	}{
		{"greedy", 0.0, 0.0},
		{"low", 0.3, 0.3},
		{"default_creative", 0.7, 0.7},
		{"high", 1.5, 1.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ApplyGenerateOpts([]GenerateOption{WithTemperature(tt.val)})
			checkInDelta(t, tt.want, cfg.Temperature, 0.0001)
		})
	}
}

func TestOptions_WithTemperature_Bad(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithTemperature(-0.5)})
	checkInDelta(t, float32(-0.5), cfg.Temperature, 0.0001)
	checkEqual(t, DefaultGenerateConfig().MaxTokens, cfg.MaxTokens)
	checkFalse(t, cfg.ReturnLogits)
}

// --- WithTopK ---

func TestOptions_WithTopK_Good(t *testing.T) {
	tests := []struct {
		name string
		val  int
		want int
	}{
		{"disabled", 0, 0},
		{"small", 10, 10},
		{"typical", 40, 40},
		{"large", 100, 100},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ApplyGenerateOpts([]GenerateOption{WithTopK(tt.val)})
			checkEqual(t, tt.want, cfg.TopK)
		})
	}
}

func TestOptions_WithTopK_Bad(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithTopK(-1)})
	checkEqual(t, -1, cfg.TopK)
	checkEqual(t, DefaultGenerateConfig().MaxTokens, cfg.MaxTokens)
	checkFalse(t, cfg.ReturnLogits)
}

// --- WithTopP ---

func TestOptions_WithTopP_Good(t *testing.T) {
	tests := []struct {
		name string
		val  float32
		want float32
	}{
		{"disabled", 0.0, 0.0},
		{"tight", 0.5, 0.5},
		{"typical", 0.9, 0.9},
		{"full", 1.0, 1.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ApplyGenerateOpts([]GenerateOption{WithTopP(tt.val)})
			checkInDelta(t, tt.want, cfg.TopP, 0.0001)
		})
	}
}

func TestOptions_WithTopP_Bad(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithTopP(-0.1)})
	checkInDelta(t, float32(-0.1), cfg.TopP, 0.0001)
	checkEqual(t, DefaultGenerateConfig().MaxTokens, cfg.MaxTokens)
	checkNil(t, cfg.StopTokens)
}

func TestOptions_WithTopP_Ugly(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithTopP(1.5)})
	checkInDelta(t, float32(1.5), cfg.TopP, 0.0001)
	checkEqual(t, DefaultGenerateConfig().MaxTokens, cfg.MaxTokens)
	checkFalse(t, cfg.ReturnLogits)
}

// --- WithStopTokens ---

func TestOptions_WithStopTokens_Good(t *testing.T) {
	t.Run("single", func(t *testing.T) {
		cfg := ApplyGenerateOpts([]GenerateOption{WithStopTokens(1)})
		checkEqual(t, []int32{1}, cfg.StopTokens)
	})

	t.Run("multiple", func(t *testing.T) {
		cfg := ApplyGenerateOpts([]GenerateOption{WithStopTokens(1, 2, 3)})
		checkEqual(t, []int32{1, 2, 3}, cfg.StopTokens)
	})
}

func TestOptions_WithStopTokens_Bad(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithStopTokens()})
	checkNil(t, cfg.StopTokens)
	checkEqual(t, DefaultGenerateConfig().MaxTokens, cfg.MaxTokens)
	checkEqual(t, DefaultGenerateConfig().RepeatPenalty, cfg.RepeatPenalty)
}

func TestOptions_WithStopTokens_Ugly(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{
		WithStopTokens(1, 2),
		WithStopTokens(3, 4, 5),
	})
	checkEqual(t, []int32{3, 4, 5}, cfg.StopTokens)
}

func TestOptions_WithStopTokens_Good_CopyIsolation(t *testing.T) {
	ids := []int32{7, 8, 9}
	cfg := ApplyGenerateOpts([]GenerateOption{WithStopTokens(ids...)})

	ids[0] = 42
	checkEqual(t, []int32{7, 8, 9}, cfg.StopTokens)
}

// --- WithRepeatPenalty ---

func TestOptions_WithRepeatPenalty_Good(t *testing.T) {
	tests := []struct {
		name string
		val  float32
		want float32
	}{
		{"disabled", 0.0, 0.0},
		{"no_penalty", 1.0, 1.0},
		{"typical", 1.1, 1.1},
		{"strong", 2.0, 2.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ApplyGenerateOpts([]GenerateOption{WithRepeatPenalty(tt.val)})
			checkInDelta(t, tt.want, cfg.RepeatPenalty, 0.0001)
		})
	}
}

func TestOptions_WithRepeatPenalty_Bad(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithRepeatPenalty(-1.0)})
	checkInDelta(t, float32(-1.0), cfg.RepeatPenalty, 0.0001)
	checkEqual(t, DefaultGenerateConfig().MaxTokens, cfg.MaxTokens)
	checkNil(t, cfg.StopTokens)
}

func TestOptions_WithRepeatPenalty_Ugly(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithRepeatPenalty(10.0)})
	checkInDelta(t, float32(10.0), cfg.RepeatPenalty, 0.0001)
	checkEqual(t, DefaultGenerateConfig().Temperature, cfg.Temperature)
	checkFalse(t, cfg.ReturnLogits)
}

// --- WithLogits ---

func TestOptions_WithLogits_Good(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithLogits()})
	checkTrue(t, cfg.ReturnLogits)
	checkEqual(t, DefaultGenerateConfig().MaxTokens, cfg.MaxTokens)
	checkEqual(t, DefaultGenerateConfig().RepeatPenalty, cfg.RepeatPenalty)
}

func TestOptions_WithLogits_Good_DefaultIsFalse(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithMaxTokens(64)})
	checkFalse(t, cfg.ReturnLogits)
	checkEqual(t, 64, cfg.MaxTokens)
	checkEqual(t, DefaultGenerateConfig().RepeatPenalty, cfg.RepeatPenalty)
}

// --- ApplyGenerateOpts ---

func TestOptions_ApplyGenerateOpts_Good(t *testing.T) {
	t.Run("nil_opts_returns_defaults", func(t *testing.T) {
		actualConfig := ApplyGenerateOpts(nil)
		defaultConfig := DefaultGenerateConfig()
		checkEqual(t, defaultConfig, actualConfig)
	})

	t.Run("nil_option_is_ignored", func(t *testing.T) {
		actualConfig := ApplyGenerateOpts([]GenerateOption{nil})
		defaultConfig := DefaultGenerateConfig()
		checkEqual(t, defaultConfig, actualConfig)
	})

	t.Run("empty_opts_returns_defaults", func(t *testing.T) {
		actualConfig := ApplyGenerateOpts([]GenerateOption{})
		defaultConfig := DefaultGenerateConfig()
		checkEqual(t, defaultConfig, actualConfig)
	})

	t.Run("all_options_combined", func(t *testing.T) {
		cfg := ApplyGenerateOpts([]GenerateOption{
			WithMaxTokens(128),
			WithTemperature(0.7),
			WithTopK(40),
			WithTopP(0.9),
			WithStopTokens(1, 2),
			WithRepeatPenalty(1.1),
			WithLogits(),
		})
		checkEqual(t, 128, cfg.MaxTokens)
		checkInDelta(t, float32(0.7), cfg.Temperature, 0.0001)
		checkEqual(t, 40, cfg.TopK)
		checkInDelta(t, float32(0.9), cfg.TopP, 0.0001)
		checkEqual(t, []int32{1, 2}, cfg.StopTokens)
		checkInDelta(t, float32(1.1), cfg.RepeatPenalty, 0.0001)
		checkTrue(t, cfg.ReturnLogits)
	})
}

func TestOptions_ApplyGenerateOpts_Good_PartialOptions(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{
		WithTemperature(0.8),
		WithTopK(50),
	})
	def := DefaultGenerateConfig()
	checkEqual(t, def.MaxTokens, cfg.MaxTokens)
	checkInDelta(t, float32(0.8), cfg.Temperature, 0.0001)
	checkEqual(t, 50, cfg.TopK)
	checkEqual(t, def.TopP, cfg.TopP)
	checkNil(t, cfg.StopTokens)
	checkEqual(t, def.RepeatPenalty, cfg.RepeatPenalty)
	checkFalse(t, cfg.ReturnLogits)
}

func TestOptions_ApplyGenerateOpts_Ugly(t *testing.T) {
	t.Run("last_option_wins", func(t *testing.T) {
		cfg := ApplyGenerateOpts([]GenerateOption{
			WithMaxTokens(100),
			WithMaxTokens(200),
			WithMaxTokens(300),
		})
		checkEqual(t, 300, cfg.MaxTokens)
	})

	t.Run("temperature_override", func(t *testing.T) {
		cfg := ApplyGenerateOpts([]GenerateOption{
			WithTemperature(0.5),
			WithTemperature(1.0),
		})
		checkInDelta(t, float32(1.0), cfg.Temperature, 0.0001)
	})

	t.Run("topk_override", func(t *testing.T) {
		cfg := ApplyGenerateOpts([]GenerateOption{
			WithTopK(10),
			WithTopK(50),
		})
		checkEqual(t, 50, cfg.TopK)
	})

	t.Run("topp_override", func(t *testing.T) {
		cfg := ApplyGenerateOpts([]GenerateOption{
			WithTopP(0.5),
			WithTopP(0.95),
		})
		checkInDelta(t, float32(0.95), cfg.TopP, 0.0001)
	})

	t.Run("repeat_penalty_override", func(t *testing.T) {
		cfg := ApplyGenerateOpts([]GenerateOption{
			WithRepeatPenalty(1.1),
			WithRepeatPenalty(1.5),
		})
		checkInDelta(t, float32(1.5), cfg.RepeatPenalty, 0.0001)
	})
}

// --- LoadConfig defaults ---

func TestOptions_ApplyLoadOpts_Good_Defaults(t *testing.T) {
	cfg := ApplyLoadOpts(nil)
	checkEqual(t, "", cfg.Backend)
	checkEqual(t, 0, cfg.ContextLen)
	checkEqual(t, -1, cfg.GPULayers)
	checkEqual(t, 0, cfg.ParallelSlots)
}

func TestOptions_ApplyLoadOpts_Good_NilOptionIsIgnored(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{nil})
	checkEqual(t, "", cfg.Backend)
	checkEqual(t, 0, cfg.ContextLen)
	checkEqual(t, -1, cfg.GPULayers)
	checkEqual(t, 0, cfg.ParallelSlots)
	checkEqual(t, "", cfg.AdapterPath)
}

// --- WithBackend ---

func TestOptions_WithBackend_Good(t *testing.T) {
	tests := []struct {
		name    string
		backend string
	}{
		{"metal", "metal"},
		{"rocm", "rocm"},
		{"llama_cpp", "llama_cpp"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ApplyLoadOpts([]LoadOption{WithBackend(tt.backend)})
			checkEqual(t, tt.backend, cfg.Backend)
		})
	}
}

func TestOptions_WithBackend_Bad(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithBackend("")})
	checkEqual(t, "", cfg.Backend)
	checkEqual(t, -1, cfg.GPULayers)
	checkEqual(t, 0, cfg.ContextLen)
}

// --- WithContextLen ---

func TestOptions_WithContextLen_Good(t *testing.T) {
	tests := []struct {
		name string
		val  int
		want int
	}{
		{"small", 2048, 2048},
		{"medium", 4096, 4096},
		{"large", 32768, 32768},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ApplyLoadOpts([]LoadOption{WithContextLen(tt.val)})
			checkEqual(t, tt.want, cfg.ContextLen)
		})
	}
}

// --- WithGPULayers ---

func TestOptions_WithGPULayers_Good(t *testing.T) {
	tests := []struct {
		name string
		val  int
		want int
	}{
		{"all", -1, -1},
		{"none", 0, 0},
		{"partial", 24, 24},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ApplyLoadOpts([]LoadOption{WithGPULayers(tt.val)})
			checkEqual(t, tt.want, cfg.GPULayers)
		})
	}
}

func TestOptions_WithGPULayers_Ugly(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithGPULayers(0)})
	checkEqual(t, 0, cfg.GPULayers)
	checkEqual(t, "", cfg.Backend)
	checkEqual(t, 0, cfg.ContextLen)
}

// --- WithParallelSlots ---

func TestOptions_WithParallelSlots_Good(t *testing.T) {
	tests := []struct {
		name string
		val  int
		want int
	}{
		{"default", 0, 0},
		{"one", 1, 1},
		{"four", 4, 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ApplyLoadOpts([]LoadOption{WithParallelSlots(tt.val)})
			checkEqual(t, tt.want, cfg.ParallelSlots)
		})
	}
}

// --- ApplyLoadOpts combined ---

func TestOptions_ApplyLoadOpts_Good_Combined(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{
		WithBackend("rocm"),
		WithContextLen(8192),
		WithGPULayers(32),
		WithParallelSlots(2),
	})
	checkEqual(t, "rocm", cfg.Backend)
	checkEqual(t, 8192, cfg.ContextLen)
	checkEqual(t, 32, cfg.GPULayers)
	checkEqual(t, 2, cfg.ParallelSlots)
}

func TestOptions_ApplyLoadOpts_Good_PartialOptions(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithContextLen(4096)})
	checkEqual(t, "", cfg.Backend)
	checkEqual(t, 4096, cfg.ContextLen)
	checkEqual(t, -1, cfg.GPULayers)
	checkEqual(t, 0, cfg.ParallelSlots)
}

func TestOptions_ApplyLoadOpts_Ugly(t *testing.T) {
	t.Run("last_option_wins", func(t *testing.T) {
		cfg := ApplyLoadOpts([]LoadOption{
			WithBackend("metal"),
			WithBackend("rocm"),
		})
		checkEqual(t, "rocm", cfg.Backend)
	})

	t.Run("empty_slice_returns_defaults", func(t *testing.T) {
		cfg := ApplyLoadOpts([]LoadOption{})
		checkEqual(t, -1, cfg.GPULayers)
		checkEqual(t, "", cfg.Backend)
	})

	t.Run("context_len_override", func(t *testing.T) {
		cfg := ApplyLoadOpts([]LoadOption{
			WithContextLen(2048),
			WithContextLen(8192),
		})
		checkEqual(t, 8192, cfg.ContextLen)
	})

	t.Run("gpu_layers_override", func(t *testing.T) {
		cfg := ApplyLoadOpts([]LoadOption{
			WithGPULayers(24),
			WithGPULayers(0),
		})
		checkEqual(t, 0, cfg.GPULayers)
	})

	t.Run("parallel_slots_override", func(t *testing.T) {
		cfg := ApplyLoadOpts([]LoadOption{
			WithParallelSlots(4),
			WithParallelSlots(1),
		})
		checkEqual(t, 1, cfg.ParallelSlots)
	})

	t.Run("adapter_path_override", func(t *testing.T) {
		cfg := ApplyLoadOpts([]LoadOption{
			WithAdapterPath("/path/a"),
			WithAdapterPath("/path/b"),
		})
		checkEqual(t, "/path/b", cfg.AdapterPath)
	})
}

// --- WithAdapterPath ---

func TestOptions_WithAdapterPath_Good(t *testing.T) {
	tests := []struct {
		name string
		val  string
		want string
	}{
		{"simple", "/path/to/adapter", "/path/to/adapter"},
		{"relative", "adapters/lora-v1", "adapters/lora-v1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ApplyLoadOpts([]LoadOption{WithAdapterPath(tt.val)})
			checkEqual(t, tt.want, cfg.AdapterPath)
		})
	}
}

func TestOptions_WithAdapterPath_Bad(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithAdapterPath("")})
	checkEqual(t, "", cfg.AdapterPath)
	checkEqual(t, -1, cfg.GPULayers)
	checkEqual(t, 0, cfg.ContextLen)
}

func TestOptions_WithAdapterPath_Good_DefaultIsEmpty(t *testing.T) {
	cfg := ApplyLoadOpts(nil)
	checkEqual(t, "", cfg.AdapterPath)
	checkEqual(t, "", cfg.Backend)
	checkEqual(t, -1, cfg.GPULayers)
}

func TestOptions_WithAdapterPath_Good_OtherFieldsUnchanged(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithAdapterPath("/some/path")})
	checkEqual(t, "", cfg.Backend)
	checkEqual(t, 0, cfg.ContextLen)
	checkEqual(t, -1, cfg.GPULayers)
	checkEqual(t, 0, cfg.ParallelSlots)
	checkEqual(t, "/some/path", cfg.AdapterPath)
}

func TestOptions_DefaultGenerateConfig_Bad(t *testing.T) {
	cfg := DefaultGenerateConfig()
	cfg.MaxTokens = 0

	core.AssertEqual(t, 0, cfg.MaxTokens)
	core.AssertEqual(t, float32(0), cfg.Temperature)
	core.AssertEqual(t, float32(1), cfg.RepeatPenalty)
}

func TestOptions_DefaultGenerateConfig_Ugly(t *testing.T) {
	cfg := DefaultGenerateConfig()
	cfg.StopTokens = append(cfg.StopTokens, 1)

	core.AssertEqual(t, []int32{1}, cfg.StopTokens)
	core.AssertFalse(t, DefaultGenerateConfig().ReturnLogits)
	core.AssertNil(t, DefaultGenerateConfig().StopTokens)
}

func TestOptions_WithMaxTokens_Ugly(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithMaxTokens(1 << 20)})
	def := DefaultGenerateConfig()

	core.AssertEqual(t, 1<<20, cfg.MaxTokens)
	core.AssertEqual(t, def.Temperature, cfg.Temperature)
	core.AssertEqual(t, def.RepeatPenalty, cfg.RepeatPenalty)
}

func TestOptions_WithTemperature_Ugly(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithTemperature(2.5)})
	def := DefaultGenerateConfig()

	core.AssertInDelta(t, 2.5, float64(cfg.Temperature), 0.0001)
	core.AssertEqual(t, def.MaxTokens, cfg.MaxTokens)
	core.AssertFalse(t, cfg.ReturnLogits)
}

func TestOptions_WithTopK_Ugly(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithTopK(1 << 16)})
	def := DefaultGenerateConfig()

	core.AssertEqual(t, 1<<16, cfg.TopK)
	core.AssertEqual(t, def.MaxTokens, cfg.MaxTokens)
	core.AssertEqual(t, def.TopP, cfg.TopP)
}

func TestOptions_WithLogits_Bad(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithLogits(), nil})
	def := DefaultGenerateConfig()

	core.AssertTrue(t, cfg.ReturnLogits)
	core.AssertEqual(t, def.MaxTokens, cfg.MaxTokens)
	core.AssertEqual(t, def.RepeatPenalty, cfg.RepeatPenalty)
}

func TestOptions_WithLogits_Ugly(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithLogits(), WithLogits()})
	def := DefaultGenerateConfig()

	core.AssertTrue(t, cfg.ReturnLogits)
	core.AssertEqual(t, def.TopK, cfg.TopK)
	core.AssertNil(t, cfg.StopTokens)
}

func TestOptions_ApplyGenerateOpts_Bad(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{nil, nil})
	def := DefaultGenerateConfig()

	core.AssertEqual(t, def, cfg)
	core.AssertFalse(t, cfg.ReturnLogits)
	core.AssertNil(t, cfg.StopTokens)
}

func TestOptions_ApplyLoadOpts_Good(t *testing.T) {
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

func TestOptions_ApplyLoadOpts_Bad(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{nil})
	def := ApplyLoadOpts(nil)

	core.AssertEqual(t, def, cfg)
	core.AssertEqual(t, -1, cfg.GPULayers)
	core.AssertEqual(t, "", cfg.Backend)
}

func TestOptions_WithBackend_Ugly(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithBackend("metal"), WithBackend("rocm")})
	def := ApplyLoadOpts(nil)

	core.AssertEqual(t, "rocm", cfg.Backend)
	core.AssertEqual(t, def.ContextLen, cfg.ContextLen)
	core.AssertEqual(t, def.GPULayers, cfg.GPULayers)
}

func TestOptions_WithContextLen_Bad(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithContextLen(0)})
	def := ApplyLoadOpts(nil)

	core.AssertEqual(t, 0, cfg.ContextLen)
	core.AssertEqual(t, def.GPULayers, cfg.GPULayers)
	core.AssertEqual(t, def.Backend, cfg.Backend)
}

func TestOptions_WithContextLen_Ugly(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithContextLen(-4096)})
	def := ApplyLoadOpts(nil)

	core.AssertEqual(t, -4096, cfg.ContextLen)
	core.AssertEqual(t, def.ParallelSlots, cfg.ParallelSlots)
	core.AssertEqual(t, def.AdapterPath, cfg.AdapterPath)
}

func TestOptions_WithGPULayers_Bad(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithGPULayers(-2)})
	def := ApplyLoadOpts(nil)

	core.AssertEqual(t, -2, cfg.GPULayers)
	core.AssertEqual(t, def.ContextLen, cfg.ContextLen)
	core.AssertEqual(t, def.AdapterPath, cfg.AdapterPath)
}

func TestOptions_WithParallelSlots_Bad(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithParallelSlots(-1)})
	def := ApplyLoadOpts(nil)

	core.AssertEqual(t, -1, cfg.ParallelSlots)
	core.AssertEqual(t, def.Backend, cfg.Backend)
	core.AssertEqual(t, def.GPULayers, cfg.GPULayers)
}

func TestOptions_WithParallelSlots_Ugly(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithParallelSlots(128)})
	def := ApplyLoadOpts(nil)

	core.AssertEqual(t, 128, cfg.ParallelSlots)
	core.AssertEqual(t, def.ContextLen, cfg.ContextLen)
	core.AssertEqual(t, def.AdapterPath, cfg.AdapterPath)
}

func TestOptions_WithAdapterPath_Ugly(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithAdapterPath(""), WithAdapterPath("/tmp/adapter")})
	def := ApplyLoadOpts(nil)

	core.AssertEqual(t, "/tmp/adapter", cfg.AdapterPath)
	core.AssertEqual(t, def.ContextLen, cfg.ContextLen)
	core.AssertEqual(t, def.GPULayers, cfg.GPULayers)
}

func TestOptions_WithMinP_Good(t *testing.T) {
	tests := []struct {
		name string
		val  float32
		want float32
	}{
		{"disabled", 0, 0},
		{"typical", 0.05, 0.05},
		{"aggressive", 0.2, 0.2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ApplyGenerateOpts([]GenerateOption{WithMinP(tt.val)})
			checkEqual(t, tt.want, cfg.MinP)
		})
	}
}

func TestOptions_WithSeed_Good(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithSeed(42)})
	checkEqual(t, uint64(42), cfg.Seed)
	checkEqual(t, true, cfg.SeedSet)
	// Absent WithSeed, SeedSet stays false so backends keep non-deterministic sampling.
	checkEqual(t, false, ApplyGenerateOpts(nil).SeedSet)
}

// TestOptions_SamplingScalars_SetFlags pins the SeedSet-style companion flags:
// each sampling scalar's With* setter raises its "was this set" flag (so a
// backend folding a model's declared defaults honours the request's value even
// when it is the zero value — an explicit greedy/disabled request), while an
// absent option leaves the flag false so the declared default may apply.
func TestOptions_SamplingScalars_SetFlags(t *testing.T) {
	// Explicit zero still raises the flag — the whole point of the design: a
	// caller asking Temperature 0 (greedy) or TopK/TopP/MinP 0 (disabled) must
	// be distinguishable from a caller who said nothing.
	set := ApplyGenerateOpts([]GenerateOption{WithTemperature(0), WithTopK(0), WithTopP(0), WithMinP(0)})
	checkEqual(t, true, set.TemperatureSet)
	checkEqual(t, true, set.TopKSet)
	checkEqual(t, true, set.TopPSet)
	checkEqual(t, true, set.MinPSet)

	// Absent options leave every flag false (unset).
	unset := ApplyGenerateOpts(nil)
	checkEqual(t, false, unset.TemperatureSet)
	checkEqual(t, false, unset.TopKSet)
	checkEqual(t, false, unset.TopPSet)
	checkEqual(t, false, unset.MinPSet)

	// A direct struct literal leaves the flags false unless it sets them —
	// mirroring Seed/SeedSet, so existing literal users keep old behaviour.
	literal := GenerateConfig{Temperature: 0.7}
	checkEqual(t, false, literal.TemperatureSet)
}

func TestOptions_WithSuppressTokens_Good(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithSuppressTokens(1, 2, 3)})
	checkEqual(t, 3, len(cfg.SuppressTokens))
	checkEqual(t, int32(1), cfg.SuppressTokens[0])
	checkEqual(t, int32(3), cfg.SuppressTokens[2])
}

func TestOptions_WithMinTokensBeforeStop_Good(t *testing.T) {
	tests := []struct {
		name string
		val  int
		want int
	}{
		{"disabled", 0, 0},
		{"short-answer", 8, 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ApplyGenerateOpts([]GenerateOption{WithMinTokensBeforeStop(tt.val)})
			checkEqual(t, tt.want, cfg.MinTokensBeforeStop)
		})
	}
}

// TestOptions_WithEnableThinking_Good pins the reasoning toggle: nil is the
// model default, &true forces on, &false forces off — the pointer identity is
// carried through so a downstream Chat can distinguish "unset" from "off".
func TestOptions_WithEnableThinking_Good(t *testing.T) {
	if cfg := ApplyGenerateOpts([]GenerateOption{WithEnableThinking(nil)}); cfg.EnableThinking != nil {
		t.Fatalf("WithEnableThinking(nil) set %v, want nil (model default)", cfg.EnableThinking)
	}
	on := true
	if cfg := ApplyGenerateOpts([]GenerateOption{WithEnableThinking(&on)}); cfg.EnableThinking == nil || !*cfg.EnableThinking {
		t.Fatalf("WithEnableThinking(&true) = %v, want forced on", cfg.EnableThinking)
	}
	off := false
	if cfg := ApplyGenerateOpts([]GenerateOption{WithEnableThinking(&off)}); cfg.EnableThinking == nil || *cfg.EnableThinking {
		t.Fatalf("WithEnableThinking(&false) = %v, want forced off", cfg.EnableThinking)
	}
}

// TestOptions_WithThinkingBudget_Good pins the thought-channel token cap: 0 is
// unlimited, a positive value bounds the reasoning span.
func TestOptions_WithThinkingBudget_Good(t *testing.T) {
	if cfg := ApplyGenerateOpts([]GenerateOption{WithThinkingBudget(0)}); cfg.ThinkingBudget != 0 {
		t.Fatalf("WithThinkingBudget(0) = %d, want 0 (unlimited)", cfg.ThinkingBudget)
	}
	if cfg := ApplyGenerateOpts([]GenerateOption{WithThinkingBudget(256)}); cfg.ThinkingBudget != 256 {
		t.Fatalf("WithThinkingBudget(256) = %d, want 256", cfg.ThinkingBudget)
	}
}

// TestOptions_WithVisionBudget_Good pins the vision soft-token budget
// override: 0 is the model default (e.g. gemma4's processor_config.json
// max_soft_tokens), a positive value requests one of the model's declared
// budgets explicitly (1120 is gemma4's OCR budget).
func TestOptions_WithVisionBudget_Good(t *testing.T) {
	if cfg := ApplyGenerateOpts([]GenerateOption{WithVisionBudget(0)}); cfg.VisionBudget != 0 {
		t.Fatalf("WithVisionBudget(0) = %d, want 0 (model default)", cfg.VisionBudget)
	}
	if cfg := ApplyGenerateOpts([]GenerateOption{WithVisionBudget(1120)}); cfg.VisionBudget != 1120 {
		t.Fatalf("WithVisionBudget(1120) = %d, want 1120", cfg.VisionBudget)
	}
}

// TestOptions_WithMetricsSink_Good pins the request-scoped metrics receiver:
// the sink lands on the config and is invoked with exactly the metrics a
// backend delivers at stream end.
func TestOptions_WithMetricsSink_Good(t *testing.T) {
	var got GenerateMetrics
	fired := 0
	cfg := ApplyGenerateOpts([]GenerateOption{WithMetricsSink(func(gm GenerateMetrics) {
		got = gm
		fired++
	})})
	checkNotNil(t, cfg.MetricsSink)
	cfg.MetricsSink(GenerateMetrics{PromptTokens: 7, GeneratedTokens: 3})
	checkEqual(t, 1, fired)
	checkEqual(t, 7, got.PromptTokens)
	checkEqual(t, 3, got.GeneratedTokens)
}

// TestOptions_WithMetricsSink_Bad pins the nil sink: the default config
// carries none, and passing nil explicitly keeps it absent (not reported).
func TestOptions_WithMetricsSink_Bad(t *testing.T) {
	checkNil(t, DefaultGenerateConfig().MetricsSink)
	cfg := ApplyGenerateOpts([]GenerateOption{WithMetricsSink(nil)})
	if cfg.MetricsSink != nil {
		t.Fatalf("WithMetricsSink(nil) set a sink, want none")
	}
}
