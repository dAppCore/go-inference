package inference

// GenerateConfig holds generation parameters.
type GenerateConfig struct {
	MaxTokens   int
	Temperature float32
	TopK        int
	TopP        float32
	StopTokens  []int32
	RepeatPenalty float32
}

// DefaultGenerateConfig returns sensible defaults.
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		MaxTokens:   256,
		Temperature: 0.0, // greedy
	}
}

// GenerateOption configures text generation.
type GenerateOption func(*GenerateConfig)

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(n int) GenerateOption {
	return func(c *GenerateConfig) { c.MaxTokens = n }
}

// WithTemperature sets the sampling temperature. 0 = greedy.
func WithTemperature(t float32) GenerateOption {
	return func(c *GenerateConfig) { c.Temperature = t }
}

// WithTopK sets top-k sampling. 0 = disabled.
func WithTopK(k int) GenerateOption {
	return func(c *GenerateConfig) { c.TopK = k }
}

// WithTopP sets nucleus sampling threshold. 0 = disabled.
func WithTopP(p float32) GenerateOption {
	return func(c *GenerateConfig) { c.TopP = p }
}

// WithStopTokens sets token IDs that stop generation.
func WithStopTokens(ids ...int32) GenerateOption {
	return func(c *GenerateConfig) { c.StopTokens = ids }
}

// WithRepeatPenalty sets the repetition penalty. 0 = disabled, 1.0 = no penalty.
func WithRepeatPenalty(p float32) GenerateOption {
	return func(c *GenerateConfig) { c.RepeatPenalty = p }
}

// ApplyGenerateOpts builds a GenerateConfig from options.
func ApplyGenerateOpts(opts []GenerateOption) GenerateConfig {
	cfg := DefaultGenerateConfig()
	for _, o := range opts {
		o(&cfg)
	}
	return cfg
}

// LoadConfig holds model loading parameters.
type LoadConfig struct {
	Backend    string // "metal", "rocm", "llama_cpp" (empty = auto-detect)
	ContextLen int    // Context window size (0 = model default)
	GPULayers  int    // Number of layers to offload to GPU (-1 = all, 0 = none)
}

// LoadOption configures model loading.
type LoadOption func(*LoadConfig)

// WithBackend selects a specific inference backend by name.
func WithBackend(name string) LoadOption {
	return func(c *LoadConfig) { c.Backend = name }
}

// WithContextLen sets the context window size.
func WithContextLen(n int) LoadOption {
	return func(c *LoadConfig) { c.ContextLen = n }
}

// WithGPULayers sets how many layers to offload to GPU.
// -1 means all layers (full GPU offload).
func WithGPULayers(n int) LoadOption {
	return func(c *LoadConfig) { c.GPULayers = n }
}

// ApplyLoadOpts builds a LoadConfig from options.
func ApplyLoadOpts(opts []LoadOption) LoadConfig {
	cfg := LoadConfig{
		GPULayers: -1, // default: full GPU offload
	}
	for _, o := range opts {
		o(&cfg)
	}
	return cfg
}
