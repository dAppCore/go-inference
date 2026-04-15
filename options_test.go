package inference

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- DefaultGenerateConfig stability ---

func TestOptions_DefaultGenerateConfig_Good_Idempotent(t *testing.T) {
	// Calling DefaultGenerateConfig twice should yield identical results.
	firstConfig := DefaultGenerateConfig()
	secondConfig := DefaultGenerateConfig()
	assert.Equal(t, firstConfig, secondConfig, "DefaultGenerateConfig should be idempotent")
}

// --- GenerateConfig defaults ---

func TestOptions_DefaultGenerateConfig_Good(t *testing.T) {
	cfg := DefaultGenerateConfig()
	assert.Equal(t, 256, cfg.MaxTokens, "default MaxTokens should be 256")
	assert.Equal(t, float32(0.0), cfg.Temperature, "default Temperature should be 0.0 (greedy)")
	assert.Equal(t, 0, cfg.TopK, "default TopK should be 0 (disabled)")
	assert.Equal(t, float32(0.0), cfg.TopP, "default TopP should be 0.0 (disabled)")
	assert.Nil(t, cfg.StopTokens, "default StopTokens should be nil")
	assert.Equal(t, float32(0.0), cfg.RepeatPenalty, "default RepeatPenalty should be 0.0 (disabled)")
	assert.False(t, cfg.ReturnLogits, "default ReturnLogits should be false")
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
			assert.Equal(t, tt.want, cfg.MaxTokens)
		})
	}
}

func TestOptions_WithMaxTokens_Bad(t *testing.T) {
	// Zero and negative values are accepted (no validation in options layer).
	cfg := ApplyGenerateOpts([]GenerateOption{WithMaxTokens(0)})
	assert.Equal(t, 0, cfg.MaxTokens)

	cfg = ApplyGenerateOpts([]GenerateOption{WithMaxTokens(-1)})
	assert.Equal(t, -1, cfg.MaxTokens)
}

func TestOptions_WithMaxTokens_Good_OtherFieldsUnchanged(t *testing.T) {
	// Setting MaxTokens should not affect other fields.
	cfg := ApplyGenerateOpts([]GenerateOption{WithMaxTokens(512)})
	def := DefaultGenerateConfig()
	assert.Equal(t, 512, cfg.MaxTokens)
	assert.Equal(t, def.Temperature, cfg.Temperature, "Temperature should remain at default")
	assert.Equal(t, def.TopK, cfg.TopK, "TopK should remain at default")
	assert.Equal(t, def.TopP, cfg.TopP, "TopP should remain at default")
	assert.Nil(t, cfg.StopTokens, "StopTokens should remain nil")
	assert.Equal(t, def.RepeatPenalty, cfg.RepeatPenalty, "RepeatPenalty should remain at default")
	assert.Equal(t, def.ReturnLogits, cfg.ReturnLogits, "ReturnLogits should remain at default")
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
			assert.InDelta(t, tt.want, cfg.Temperature, 0.0001)
		})
	}
}

func TestOptions_WithTemperature_Bad(t *testing.T) {
	// Negative temperature is accepted at the options layer (no validation).
	cfg := ApplyGenerateOpts([]GenerateOption{WithTemperature(-0.5)})
	assert.InDelta(t, -0.5, cfg.Temperature, 0.0001, "negative temperature should be stored as-is")
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
			assert.Equal(t, tt.want, cfg.TopK)
		})
	}
}

func TestOptions_WithTopK_Bad(t *testing.T) {
	// Negative TopK is accepted at the options layer (no validation).
	cfg := ApplyGenerateOpts([]GenerateOption{WithTopK(-1)})
	assert.Equal(t, -1, cfg.TopK, "negative TopK should be stored as-is")
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
			assert.InDelta(t, tt.want, cfg.TopP, 0.0001)
		})
	}
}

func TestOptions_WithTopP_Bad(t *testing.T) {
	// Negative TopP is accepted at the options layer (no validation).
	cfg := ApplyGenerateOpts([]GenerateOption{WithTopP(-0.1)})
	assert.InDelta(t, -0.1, cfg.TopP, 0.0001, "negative TopP should be stored as-is")
}

func TestOptions_WithTopP_Ugly(t *testing.T) {
	// Value >1.0 is technically out of range but accepted at the options layer.
	cfg := ApplyGenerateOpts([]GenerateOption{WithTopP(1.5)})
	assert.InDelta(t, 1.5, cfg.TopP, 0.0001, "TopP >1.0 should be stored as-is; backend validates")
}

// --- WithStopTokens ---

func TestOptions_WithStopTokens_Good(t *testing.T) {
	t.Run("single", func(t *testing.T) {
		cfg := ApplyGenerateOpts([]GenerateOption{WithStopTokens(1)})
		assert.Equal(t, []int32{1}, cfg.StopTokens)
	})

	t.Run("multiple", func(t *testing.T) {
		cfg := ApplyGenerateOpts([]GenerateOption{WithStopTokens(1, 2, 3)})
		assert.Equal(t, []int32{1, 2, 3}, cfg.StopTokens)
	})
}

func TestOptions_WithStopTokens_Bad(t *testing.T) {
	// Empty variadic — Go passes nil to the variadic parameter, so StopTokens
	// is set to nil (same as default). This is a known Go behaviour.
	cfg := ApplyGenerateOpts([]GenerateOption{WithStopTokens()})
	assert.Nil(t, cfg.StopTokens, "empty variadic should set StopTokens to nil")
}

func TestOptions_WithStopTokens_Ugly(t *testing.T) {
	// Last call wins — stop tokens are replaced, not merged.
	cfg := ApplyGenerateOpts([]GenerateOption{
		WithStopTokens(1, 2),
		WithStopTokens(3, 4, 5),
	})
	assert.Equal(t, []int32{3, 4, 5}, cfg.StopTokens, "last WithStopTokens should win")
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
			assert.InDelta(t, tt.want, cfg.RepeatPenalty, 0.0001)
		})
	}
}

func TestOptions_WithRepeatPenalty_Bad(t *testing.T) {
	// Negative RepeatPenalty is accepted at the options layer (no validation).
	cfg := ApplyGenerateOpts([]GenerateOption{WithRepeatPenalty(-1.0)})
	assert.InDelta(t, -1.0, cfg.RepeatPenalty, 0.0001, "negative RepeatPenalty should be stored as-is")
}

func TestOptions_WithRepeatPenalty_Ugly(t *testing.T) {
	// Very high penalty — accepted at options layer; backend decides behaviour.
	cfg := ApplyGenerateOpts([]GenerateOption{WithRepeatPenalty(10.0)})
	assert.InDelta(t, 10.0, cfg.RepeatPenalty, 0.0001, "extreme RepeatPenalty should be stored as-is")
}

// --- WithLogits ---

func TestOptions_WithLogits_Good(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithLogits()})
	assert.True(t, cfg.ReturnLogits)
}

func TestOptions_WithLogits_Good_DefaultIsFalse(t *testing.T) {
	// Without WithLogits, ReturnLogits stays false.
	cfg := ApplyGenerateOpts([]GenerateOption{WithMaxTokens(64)})
	assert.False(t, cfg.ReturnLogits, "ReturnLogits should be false when WithLogits is not applied")
}

// --- ApplyGenerateOpts ---

func TestOptions_ApplyGenerateOpts_Good(t *testing.T) {
	t.Run("nil_opts_returns_defaults", func(t *testing.T) {
		actualConfig := ApplyGenerateOpts(nil)
		defaultConfig := DefaultGenerateConfig()
		assert.Equal(t, defaultConfig, actualConfig)
	})

	t.Run("nil_option_is_ignored", func(t *testing.T) {
		actualConfig := ApplyGenerateOpts([]GenerateOption{nil})
		defaultConfig := DefaultGenerateConfig()
		assert.Equal(t, defaultConfig, actualConfig)
	})

	t.Run("empty_opts_returns_defaults", func(t *testing.T) {
		actualConfig := ApplyGenerateOpts([]GenerateOption{})
		defaultConfig := DefaultGenerateConfig()
		assert.Equal(t, defaultConfig, actualConfig)
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
		assert.Equal(t, 128, cfg.MaxTokens)
		assert.InDelta(t, 0.7, cfg.Temperature, 0.0001)
		assert.Equal(t, 40, cfg.TopK)
		assert.InDelta(t, 0.9, cfg.TopP, 0.0001)
		assert.Equal(t, []int32{1, 2}, cfg.StopTokens)
		assert.InDelta(t, 1.1, cfg.RepeatPenalty, 0.0001)
		assert.True(t, cfg.ReturnLogits)
	})
}

func TestOptions_ApplyGenerateOpts_Good_PartialOptions(t *testing.T) {
	// Applying only some options should preserve defaults for the rest.
	cfg := ApplyGenerateOpts([]GenerateOption{
		WithTemperature(0.8),
		WithTopK(50),
	})
	def := DefaultGenerateConfig()
	assert.Equal(t, def.MaxTokens, cfg.MaxTokens, "MaxTokens should remain at default")
	assert.InDelta(t, 0.8, cfg.Temperature, 0.0001)
	assert.Equal(t, 50, cfg.TopK)
	assert.Equal(t, def.TopP, cfg.TopP, "TopP should remain at default")
	assert.Nil(t, cfg.StopTokens, "StopTokens should remain nil")
	assert.Equal(t, def.RepeatPenalty, cfg.RepeatPenalty, "RepeatPenalty should remain at default")
	assert.False(t, cfg.ReturnLogits, "ReturnLogits should remain false")
}

func TestOptions_ApplyGenerateOpts_Ugly(t *testing.T) {
	t.Run("last_option_wins", func(t *testing.T) {
		cfg := ApplyGenerateOpts([]GenerateOption{
			WithMaxTokens(100),
			WithMaxTokens(200),
			WithMaxTokens(300),
		})
		assert.Equal(t, 300, cfg.MaxTokens, "last WithMaxTokens should win")
	})

	t.Run("temperature_override", func(t *testing.T) {
		cfg := ApplyGenerateOpts([]GenerateOption{
			WithTemperature(0.5),
			WithTemperature(1.0),
		})
		assert.InDelta(t, 1.0, cfg.Temperature, 0.0001, "last WithTemperature should win")
	})

	t.Run("topk_override", func(t *testing.T) {
		cfg := ApplyGenerateOpts([]GenerateOption{
			WithTopK(10),
			WithTopK(50),
		})
		assert.Equal(t, 50, cfg.TopK, "last WithTopK should win")
	})

	t.Run("topp_override", func(t *testing.T) {
		cfg := ApplyGenerateOpts([]GenerateOption{
			WithTopP(0.5),
			WithTopP(0.95),
		})
		assert.InDelta(t, 0.95, cfg.TopP, 0.0001, "last WithTopP should win")
	})

	t.Run("repeat_penalty_override", func(t *testing.T) {
		cfg := ApplyGenerateOpts([]GenerateOption{
			WithRepeatPenalty(1.1),
			WithRepeatPenalty(1.5),
		})
		assert.InDelta(t, 1.5, cfg.RepeatPenalty, 0.0001, "last WithRepeatPenalty should win")
	})
}

// --- LoadConfig defaults ---

func TestOptions_ApplyLoadOpts_Good_Defaults(t *testing.T) {
	cfg := ApplyLoadOpts(nil)
	assert.Equal(t, "", cfg.Backend, "default Backend should be empty (auto-detect)")
	assert.Equal(t, 0, cfg.ContextLen, "default ContextLen should be 0 (model default)")
	assert.Equal(t, -1, cfg.GPULayers, "default GPULayers should be -1 (all layers)")
	assert.Equal(t, 0, cfg.ParallelSlots, "default ParallelSlots should be 0 (server default)")
}

func TestOptions_ApplyLoadOpts_Good_NilOptionIsIgnored(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{nil})
	assert.Equal(t, "", cfg.Backend)
	assert.Equal(t, 0, cfg.ContextLen)
	assert.Equal(t, -1, cfg.GPULayers)
	assert.Equal(t, 0, cfg.ParallelSlots)
	assert.Equal(t, "", cfg.AdapterPath)
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
			assert.Equal(t, tt.backend, cfg.Backend)
		})
	}
}

func TestOptions_WithBackend_Bad(t *testing.T) {
	// Empty string is valid at the options layer (means auto-detect).
	cfg := ApplyLoadOpts([]LoadOption{WithBackend("")})
	assert.Equal(t, "", cfg.Backend)
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
			assert.Equal(t, tt.want, cfg.ContextLen)
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
			assert.Equal(t, tt.want, cfg.GPULayers)
		})
	}
}

func TestOptions_WithGPULayers_Ugly(t *testing.T) {
	// Override the default -1 with 0
	cfg := ApplyLoadOpts([]LoadOption{WithGPULayers(0)})
	assert.Equal(t, 0, cfg.GPULayers, "WithGPULayers(0) should override default -1")
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
			assert.Equal(t, tt.want, cfg.ParallelSlots)
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
	assert.Equal(t, "rocm", cfg.Backend)
	assert.Equal(t, 8192, cfg.ContextLen)
	assert.Equal(t, 32, cfg.GPULayers)
	assert.Equal(t, 2, cfg.ParallelSlots)
}

func TestOptions_ApplyLoadOpts_Good_PartialOptions(t *testing.T) {
	// Applying only some options should preserve defaults for the rest.
	cfg := ApplyLoadOpts([]LoadOption{WithContextLen(4096)})
	assert.Equal(t, "", cfg.Backend, "Backend should remain at default (auto-detect)")
	assert.Equal(t, 4096, cfg.ContextLen)
	assert.Equal(t, -1, cfg.GPULayers, "GPULayers should remain at default (-1)")
	assert.Equal(t, 0, cfg.ParallelSlots, "ParallelSlots should remain at default (0)")
}

func TestOptions_ApplyLoadOpts_Ugly(t *testing.T) {
	t.Run("last_option_wins", func(t *testing.T) {
		cfg := ApplyLoadOpts([]LoadOption{
			WithBackend("metal"),
			WithBackend("rocm"),
		})
		assert.Equal(t, "rocm", cfg.Backend, "last WithBackend should win")
	})

	t.Run("empty_slice_returns_defaults", func(t *testing.T) {
		cfg := ApplyLoadOpts([]LoadOption{})
		require.Equal(t, -1, cfg.GPULayers, "empty opts should keep default GPULayers=-1")
		assert.Equal(t, "", cfg.Backend)
	})

	t.Run("context_len_override", func(t *testing.T) {
		cfg := ApplyLoadOpts([]LoadOption{
			WithContextLen(2048),
			WithContextLen(8192),
		})
		assert.Equal(t, 8192, cfg.ContextLen, "last WithContextLen should win")
	})

	t.Run("gpu_layers_override", func(t *testing.T) {
		cfg := ApplyLoadOpts([]LoadOption{
			WithGPULayers(24),
			WithGPULayers(0),
		})
		assert.Equal(t, 0, cfg.GPULayers, "last WithGPULayers should win")
	})

	t.Run("parallel_slots_override", func(t *testing.T) {
		cfg := ApplyLoadOpts([]LoadOption{
			WithParallelSlots(4),
			WithParallelSlots(1),
		})
		assert.Equal(t, 1, cfg.ParallelSlots, "last WithParallelSlots should win")
	})

	t.Run("adapter_path_override", func(t *testing.T) {
		cfg := ApplyLoadOpts([]LoadOption{
			WithAdapterPath("/path/a"),
			WithAdapterPath("/path/b"),
		})
		assert.Equal(t, "/path/b", cfg.AdapterPath, "last WithAdapterPath should win")
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
			assert.Equal(t, tt.want, cfg.AdapterPath)
		})
	}
}

func TestOptions_WithAdapterPath_Bad(t *testing.T) {
	// Empty string is valid at the options layer (means no adapter).
	cfg := ApplyLoadOpts([]LoadOption{WithAdapterPath("")})
	assert.Equal(t, "", cfg.AdapterPath)
}

func TestOptions_WithAdapterPath_Good_DefaultIsEmpty(t *testing.T) {
	cfg := ApplyLoadOpts(nil)
	assert.Equal(t, "", cfg.AdapterPath, "default AdapterPath should be empty")
}

func TestOptions_WithAdapterPath_Good_OtherFieldsUnchanged(t *testing.T) {
	cfg := ApplyLoadOpts([]LoadOption{WithAdapterPath("/some/path")})
	assert.Equal(t, "", cfg.Backend, "Backend should remain at default")
	assert.Equal(t, 0, cfg.ContextLen, "ContextLen should remain at default")
	assert.Equal(t, -1, cfg.GPULayers, "GPULayers should remain at default")
	assert.Equal(t, 0, cfg.ParallelSlots, "ParallelSlots should remain at default")
	assert.Equal(t, "/some/path", cfg.AdapterPath)
}
