package inference

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultGenerateConfig_Good_Idempotent(t *testing.T) {
	firstConfig := DefaultGenerateConfig()
	secondConfig := DefaultGenerateConfig()
	assert.Equal(t, firstConfig, secondConfig)
}

func TestDefaultGenerateConfig_Good(t *testing.T) {
	config := DefaultGenerateConfig()
	assert.Equal(t, 256, config.MaxTokens)
	assert.Equal(t, float32(0.0), config.Temperature)
	assert.Equal(t, 0, config.TopK)
	assert.Equal(t, float32(0.0), config.TopP)
	assert.Nil(t, config.StopTokens)
	assert.Equal(t, float32(0.0), config.RepeatPenalty)
	assert.False(t, config.ReturnLogits)
}

func TestWithMaxTokens_Good(t *testing.T) {
	tests := []struct {
		name  string
		value int
		want  int
	}{
		{"small", 32, 32},
		{"medium", 512, 512},
		{"large", 4096, 4096},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			config := ApplyGenerateOpts([]GenerateOption{WithMaxTokens(testCase.value)})
			assert.Equal(t, testCase.want, config.MaxTokens)
		})
	}
}

func TestWithMaxTokens_Bad(t *testing.T) {
	config := ApplyGenerateOpts([]GenerateOption{WithMaxTokens(0)})
	assert.Equal(t, 0, config.MaxTokens)

	config = ApplyGenerateOpts([]GenerateOption{WithMaxTokens(-1)})
	assert.Equal(t, -1, config.MaxTokens)
}

func TestWithMaxTokens_Good_OtherFieldsUnchanged(t *testing.T) {
	config := ApplyGenerateOpts([]GenerateOption{WithMaxTokens(512)})
	defaultConfig := DefaultGenerateConfig()
	assert.Equal(t, 512, config.MaxTokens)
	assert.Equal(t, defaultConfig.Temperature, config.Temperature)
	assert.Equal(t, defaultConfig.TopK, config.TopK)
	assert.Equal(t, defaultConfig.TopP, config.TopP)
	assert.Nil(t, config.StopTokens)
	assert.Equal(t, defaultConfig.RepeatPenalty, config.RepeatPenalty)
	assert.Equal(t, defaultConfig.ReturnLogits, config.ReturnLogits)
}

func TestWithTemperature_Good(t *testing.T) {
	tests := []struct {
		name  string
		value float32
		want  float32
	}{
		{"greedy", 0.0, 0.0},
		{"low", 0.3, 0.3},
		{"default_creative", 0.7, 0.7},
		{"high", 1.5, 1.5},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			config := ApplyGenerateOpts([]GenerateOption{WithTemperature(testCase.value)})
			assert.InDelta(t, testCase.want, config.Temperature, 0.0001)
		})
	}
}

func TestWithTemperature_Bad(t *testing.T) {
	config := ApplyGenerateOpts([]GenerateOption{WithTemperature(-0.5)})
	assert.InDelta(t, -0.5, config.Temperature, 0.0001)
}

func TestWithTopK_Good(t *testing.T) {
	tests := []struct {
		name  string
		value int
		want  int
	}{
		{"disabled", 0, 0},
		{"small", 10, 10},
		{"typical", 40, 40},
		{"large", 100, 100},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			config := ApplyGenerateOpts([]GenerateOption{WithTopK(testCase.value)})
			assert.Equal(t, testCase.want, config.TopK)
		})
	}
}

func TestWithTopK_Bad(t *testing.T) {
	config := ApplyGenerateOpts([]GenerateOption{WithTopK(-1)})
	assert.Equal(t, -1, config.TopK)
}

func TestWithTopP_Good(t *testing.T) {
	tests := []struct {
		name  string
		value float32
		want  float32
	}{
		{"disabled", 0.0, 0.0},
		{"tight", 0.5, 0.5},
		{"typical", 0.9, 0.9},
		{"full", 1.0, 1.0},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			config := ApplyGenerateOpts([]GenerateOption{WithTopP(testCase.value)})
			assert.InDelta(t, testCase.want, config.TopP, 0.0001)
		})
	}
}

func TestWithStopTokens_Good(t *testing.T) {
	t.Run("single", func(t *testing.T) {
		config := ApplyGenerateOpts([]GenerateOption{WithStopTokens(1)})
		assert.Equal(t, []int32{1}, config.StopTokens)
	})

	t.Run("multiple", func(t *testing.T) {
		config := ApplyGenerateOpts([]GenerateOption{WithStopTokens(1, 2, 3)})
		assert.Equal(t, []int32{1, 2, 3}, config.StopTokens)
	})
}

func TestWithStopTokens_Bad(t *testing.T) {
	config := ApplyGenerateOpts([]GenerateOption{WithStopTokens()})
	assert.Nil(t, config.StopTokens)
}

func TestWithStopTokens_Ugly(t *testing.T) {
	config := ApplyGenerateOpts([]GenerateOption{
		WithStopTokens(1, 2),
		WithStopTokens(3, 4, 5),
	})
	assert.Equal(t, []int32{3, 4, 5}, config.StopTokens)
}

func TestWithRepeatPenalty_Good(t *testing.T) {
	tests := []struct {
		name  string
		value float32
		want  float32
	}{
		{"disabled", 0.0, 0.0},
		{"no_penalty", 1.0, 1.0},
		{"typical", 1.1, 1.1},
		{"strong", 2.0, 2.0},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			config := ApplyGenerateOpts([]GenerateOption{WithRepeatPenalty(testCase.value)})
			assert.InDelta(t, testCase.want, config.RepeatPenalty, 0.0001)
		})
	}
}

func TestWithLogits_Good(t *testing.T) {
	config := ApplyGenerateOpts([]GenerateOption{WithLogits()})
	assert.True(t, config.ReturnLogits)
}

func TestWithLogits_Good_DefaultIsFalse(t *testing.T) {
	config := ApplyGenerateOpts([]GenerateOption{WithMaxTokens(64)})
	assert.False(t, config.ReturnLogits)
}

func TestApplyGenerateOpts_Good(t *testing.T) {
	t.Run("nil_options_returns_defaults", func(t *testing.T) {
		config := ApplyGenerateOpts(nil)
		defaultConfig := DefaultGenerateConfig()
		assert.Equal(t, defaultConfig, config)
	})

	t.Run("empty_options_returns_defaults", func(t *testing.T) {
		config := ApplyGenerateOpts([]GenerateOption{})
		defaultConfig := DefaultGenerateConfig()
		assert.Equal(t, defaultConfig, config)
	})

	t.Run("all_options_combined", func(t *testing.T) {
		config := ApplyGenerateOpts([]GenerateOption{
			WithMaxTokens(128),
			WithTemperature(0.7),
			WithTopK(40),
			WithTopP(0.9),
			WithStopTokens(1, 2),
			WithRepeatPenalty(1.1),
			WithLogits(),
		})
		assert.Equal(t, 128, config.MaxTokens)
		assert.InDelta(t, 0.7, config.Temperature, 0.0001)
		assert.Equal(t, 40, config.TopK)
		assert.InDelta(t, 0.9, config.TopP, 0.0001)
		assert.Equal(t, []int32{1, 2}, config.StopTokens)
		assert.InDelta(t, 1.1, config.RepeatPenalty, 0.0001)
		assert.True(t, config.ReturnLogits)
	})
}

func TestApplyGenerateOpts_Good_PartialOptions(t *testing.T) {
	config := ApplyGenerateOpts([]GenerateOption{
		WithTemperature(0.8),
		WithTopK(50),
	})
	defaultConfig := DefaultGenerateConfig()
	assert.Equal(t, defaultConfig.MaxTokens, config.MaxTokens)
	assert.InDelta(t, 0.8, config.Temperature, 0.0001)
	assert.Equal(t, 50, config.TopK)
	assert.Equal(t, defaultConfig.TopP, config.TopP)
	assert.Nil(t, config.StopTokens)
	assert.Equal(t, defaultConfig.RepeatPenalty, config.RepeatPenalty)
	assert.False(t, config.ReturnLogits)
}

func TestApplyGenerateOpts_Ugly(t *testing.T) {
	t.Run("last_option_wins", func(t *testing.T) {
		config := ApplyGenerateOpts([]GenerateOption{
			WithMaxTokens(100),
			WithMaxTokens(200),
			WithMaxTokens(300),
		})
		assert.Equal(t, 300, config.MaxTokens)
	})

	t.Run("temperature_override", func(t *testing.T) {
		config := ApplyGenerateOpts([]GenerateOption{
			WithTemperature(0.5),
			WithTemperature(1.0),
		})
		assert.InDelta(t, 1.0, config.Temperature, 0.0001)
	})

	t.Run("topk_override", func(t *testing.T) {
		config := ApplyGenerateOpts([]GenerateOption{
			WithTopK(10),
			WithTopK(50),
		})
		assert.Equal(t, 50, config.TopK)
	})

	t.Run("topp_override", func(t *testing.T) {
		config := ApplyGenerateOpts([]GenerateOption{
			WithTopP(0.5),
			WithTopP(0.95),
		})
		assert.InDelta(t, 0.95, config.TopP, 0.0001)
	})

	t.Run("repeat_penalty_override", func(t *testing.T) {
		config := ApplyGenerateOpts([]GenerateOption{
			WithRepeatPenalty(1.1),
			WithRepeatPenalty(1.5),
		})
		assert.InDelta(t, 1.5, config.RepeatPenalty, 0.0001)
	})
}

func TestApplyLoadOpts_Good_Defaults(t *testing.T) {
	loadConfig := ApplyLoadOpts(nil)
	assert.Equal(t, "", loadConfig.Backend)
	assert.Equal(t, 0, loadConfig.ContextLen)
	assert.Equal(t, -1, loadConfig.GPULayers)
	assert.Equal(t, 0, loadConfig.ParallelSlots)
}

func TestWithBackend_Good(t *testing.T) {
	tests := []struct {
		name    string
		backend string
	}{
		{"metal", "metal"},
		{"rocm", "rocm"},
		{"llama_cpp", "llama_cpp"},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			loadConfig := ApplyLoadOpts([]LoadOption{WithBackend(testCase.backend)})
			assert.Equal(t, testCase.backend, loadConfig.Backend)
		})
	}
}

func TestWithBackend_Bad(t *testing.T) {
	loadConfig := ApplyLoadOpts([]LoadOption{WithBackend("")})
	assert.Equal(t, "", loadConfig.Backend)
}

func TestWithContextLen_Good(t *testing.T) {
	tests := []struct {
		name  string
		value int
		want  int
	}{
		{"small", 2048, 2048},
		{"medium", 4096, 4096},
		{"large", 32768, 32768},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			loadConfig := ApplyLoadOpts([]LoadOption{WithContextLen(testCase.value)})
			assert.Equal(t, testCase.want, loadConfig.ContextLen)
		})
	}
}

func TestWithGPULayers_Good(t *testing.T) {
	tests := []struct {
		name  string
		value int
		want  int
	}{
		{"all", -1, -1},
		{"none", 0, 0},
		{"partial", 24, 24},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			loadConfig := ApplyLoadOpts([]LoadOption{WithGPULayers(testCase.value)})
			assert.Equal(t, testCase.want, loadConfig.GPULayers)
		})
	}
}

func TestWithGPULayers_Ugly(t *testing.T) {
	loadConfig := ApplyLoadOpts([]LoadOption{WithGPULayers(0)})
	assert.Equal(t, 0, loadConfig.GPULayers)
}

func TestWithParallelSlots_Good(t *testing.T) {
	tests := []struct {
		name  string
		value int
		want  int
	}{
		{"default", 0, 0},
		{"one", 1, 1},
		{"four", 4, 4},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			loadConfig := ApplyLoadOpts([]LoadOption{WithParallelSlots(testCase.value)})
			assert.Equal(t, testCase.want, loadConfig.ParallelSlots)
		})
	}
}

func TestApplyLoadOpts_Good_Combined(t *testing.T) {
	loadConfig := ApplyLoadOpts([]LoadOption{
		WithBackend("rocm"),
		WithContextLen(8192),
		WithGPULayers(32),
		WithParallelSlots(2),
	})
	assert.Equal(t, "rocm", loadConfig.Backend)
	assert.Equal(t, 8192, loadConfig.ContextLen)
	assert.Equal(t, 32, loadConfig.GPULayers)
	assert.Equal(t, 2, loadConfig.ParallelSlots)
}

func TestApplyLoadOpts_Good_PartialOptions(t *testing.T) {
	loadConfig := ApplyLoadOpts([]LoadOption{WithContextLen(4096)})
	assert.Equal(t, "", loadConfig.Backend)
	assert.Equal(t, 4096, loadConfig.ContextLen)
	assert.Equal(t, -1, loadConfig.GPULayers)
	assert.Equal(t, 0, loadConfig.ParallelSlots)
}

func TestApplyLoadOpts_Ugly(t *testing.T) {
	t.Run("last_option_wins", func(t *testing.T) {
		loadConfig := ApplyLoadOpts([]LoadOption{
			WithBackend("metal"),
			WithBackend("rocm"),
		})
		assert.Equal(t, "rocm", loadConfig.Backend)
	})

	t.Run("empty_slice_returns_defaults", func(t *testing.T) {
		loadConfig := ApplyLoadOpts([]LoadOption{})
		require.Equal(t, -1, loadConfig.GPULayers)
		assert.Equal(t, "", loadConfig.Backend)
	})

	t.Run("context_len_override", func(t *testing.T) {
		loadConfig := ApplyLoadOpts([]LoadOption{
			WithContextLen(2048),
			WithContextLen(8192),
		})
		assert.Equal(t, 8192, loadConfig.ContextLen)
	})

	t.Run("gpu_layers_override", func(t *testing.T) {
		loadConfig := ApplyLoadOpts([]LoadOption{
			WithGPULayers(24),
			WithGPULayers(0),
		})
		assert.Equal(t, 0, loadConfig.GPULayers)
	})

	t.Run("parallel_slots_override", func(t *testing.T) {
		loadConfig := ApplyLoadOpts([]LoadOption{
			WithParallelSlots(4),
			WithParallelSlots(1),
		})
		assert.Equal(t, 1, loadConfig.ParallelSlots)
	})

	t.Run("adapter_path_override", func(t *testing.T) {
		loadConfig := ApplyLoadOpts([]LoadOption{
			WithAdapterPath("/models/lora/primary"),
			WithAdapterPath("/models/lora/secondary"),
		})
		assert.Equal(t, "/models/lora/secondary", loadConfig.AdapterPath)
	})
}

func TestWithAdapterPath_Good(t *testing.T) {
	tests := []struct {
		name  string
		value string
		want  string
	}{
		{"simple", "/path/to/adapter", "/path/to/adapter"},
		{"relative", "adapters/lora-v1", "adapters/lora-v1"},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			loadConfig := ApplyLoadOpts([]LoadOption{WithAdapterPath(testCase.value)})
			assert.Equal(t, testCase.want, loadConfig.AdapterPath)
		})
	}
}

func TestWithAdapterPath_Bad(t *testing.T) {
	loadConfig := ApplyLoadOpts([]LoadOption{WithAdapterPath("")})
	assert.Equal(t, "", loadConfig.AdapterPath)
}

func TestWithAdapterPath_Good_DefaultIsEmpty(t *testing.T) {
	loadConfig := ApplyLoadOpts(nil)
	assert.Equal(t, "", loadConfig.AdapterPath)
}

func TestWithAdapterPath_Good_OtherFieldsUnchanged(t *testing.T) {
	loadConfig := ApplyLoadOpts([]LoadOption{WithAdapterPath("/some/path")})
	assert.Equal(t, "", loadConfig.Backend)
	assert.Equal(t, 0, loadConfig.ContextLen)
	assert.Equal(t, -1, loadConfig.GPULayers)
	assert.Equal(t, 0, loadConfig.ParallelSlots)
	assert.Equal(t, "/some/path", loadConfig.AdapterPath)
}
