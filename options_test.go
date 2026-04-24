package inference

import (
	"testing"
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
	checkEqual(t, 256, cfg.MaxTokens)
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
}

func TestOptions_WithTopP_Ugly(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithTopP(1.5)})
	checkInDelta(t, float32(1.5), cfg.TopP, 0.0001)
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
}

func TestOptions_WithRepeatPenalty_Ugly(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithRepeatPenalty(10.0)})
	checkInDelta(t, float32(10.0), cfg.RepeatPenalty, 0.0001)
}

// --- WithLogits ---

func TestOptions_WithLogits_Good(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithLogits()})
	checkTrue(t, cfg.ReturnLogits)
}

func TestOptions_WithLogits_Good_DefaultIsFalse(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithMaxTokens(64)})
	checkFalse(t, cfg.ReturnLogits)
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
}

func TestOptions_WithAdapterPath_Good_DefaultIsEmpty(t *testing.T) {
	cfg := ApplyLoadOpts(nil)
	checkEqual(t, "", cfg.AdapterPath)
}

func TestOptions_WithAdapterPath_Good_OtherFieldsUnchanged(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithAdapterPath("/some/path")})
	checkEqual(t, "", cfg.Backend)
	checkEqual(t, 0, cfg.ContextLen)
	checkEqual(t, -1, cfg.GPULayers)
	checkEqual(t, 0, cfg.ParallelSlots)
	checkEqual(t, "/some/path", cfg.AdapterPath)
}
