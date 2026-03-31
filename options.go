package inference

// config := inference.GenerateConfig{MaxTokens: 256, Temperature: 0.7, TopK: 40}
// config := inference.GenerateConfig{MaxTokens: 64, StopTokens: []int32{2}, RepeatPenalty: 1.1}
type GenerateConfig struct {
	MaxTokens     int
	Temperature   float32
	TopK          int
	TopP          float32
	StopTokens    []int32
	RepeatPenalty float32
	ReturnLogits  bool // Include raw logits in ClassifyResult
}

// config := inference.DefaultGenerateConfig() // MaxTokens=256, Temperature=0.0
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		MaxTokens:   256,
		Temperature: 0.0,
	}
}

// options := []inference.GenerateOption{inference.WithMaxTokens(128), inference.WithTemperature(0.7)}
type GenerateOption func(*GenerateConfig)

// inference.WithMaxTokens(128)  // short reply
// inference.WithMaxTokens(2048) // long-form generation
func WithMaxTokens(n int) GenerateOption {
	return func(config *GenerateConfig) { config.MaxTokens = n }
}

// inference.WithTemperature(0.0) // deterministic
// inference.WithTemperature(0.7) // balanced creativity
// inference.WithTemperature(1.5) // high variance
func WithTemperature(t float32) GenerateOption {
	return func(config *GenerateConfig) { config.Temperature = t }
}

// inference.WithTopK(40) // typical value for creative generation
func WithTopK(k int) GenerateOption {
	return func(config *GenerateConfig) { config.TopK = k }
}

// inference.WithTopP(0.9) // typical nucleus sampling threshold
func WithTopP(p float32) GenerateOption {
	return func(config *GenerateConfig) { config.TopP = p }
}

// inference.WithStopTokens(2)       // EOS token only
// inference.WithStopTokens(2, 1, 0) // EOS + pad tokens
func WithStopTokens(ids ...int32) GenerateOption {
	return func(config *GenerateConfig) { config.StopTokens = ids }
}

// inference.WithRepeatPenalty(1.1) // mild repetition suppression
// inference.WithRepeatPenalty(1.5) // strong repetition suppression
func WithRepeatPenalty(p float32) GenerateOption {
	return func(config *GenerateConfig) { config.RepeatPenalty = p }
}

// inference.WithLogits() // enable logit capture for classification scoring
func WithLogits() GenerateOption {
	return func(config *GenerateConfig) { config.ReturnLogits = true }
}

// config := inference.ApplyGenerateOpts(options)
func ApplyGenerateOpts(options []GenerateOption) GenerateConfig {
	generateConfig := DefaultGenerateConfig()
	for _, option := range options {
		option(&generateConfig)
	}
	return generateConfig
}

// config := inference.LoadConfig{Backend: "metal", ContextLen: 4096, GPULayers: -1}
// config := inference.LoadConfig{Backend: "rocm", AdapterPath: "/models/lora/v1"}
type LoadConfig struct {
	Backend       string
	ContextLen    int
	GPULayers     int
	ParallelSlots int
	AdapterPath   string
}

// options := []inference.LoadOption{inference.WithBackend("metal"), inference.WithContextLen(4096)}
type LoadOption func(*LoadConfig)

// inference.WithBackend("metal")     // Apple Silicon GPU
// inference.WithBackend("rocm")      // AMD GPU
// inference.WithBackend("llama_cpp") // CPU fallback
func WithBackend(name string) LoadOption {
	return func(config *LoadConfig) { config.Backend = name }
}

// inference.WithContextLen(4096)  // standard context
// inference.WithContextLen(32768) // extended context
func WithContextLen(n int) LoadOption {
	return func(config *LoadConfig) { config.ContextLen = n }
}

// inference.WithGPULayers(-1) // full GPU offload (default)
// inference.WithGPULayers(0)  // CPU-only inference
// inference.WithGPULayers(24) // partial offload (24 layers to GPU)
func WithGPULayers(n int) LoadOption {
	return func(config *LoadConfig) { config.GPULayers = n }
}

// inference.WithParallelSlots(4) // allow 4 concurrent inference requests
func WithParallelSlots(n int) LoadOption {
	return func(config *LoadConfig) { config.ParallelSlots = n }
}

// inference.WithAdapterPath("/models/lora/domain-v2") // load fine-tuned adapter
func WithAdapterPath(path string) LoadOption {
	return func(config *LoadConfig) { config.AdapterPath = path }
}

// config := inference.ApplyLoadOpts(options)
func ApplyLoadOpts(options []LoadOption) LoadConfig {
	loadConfig := LoadConfig{
		GPULayers: -1,
	}
	for _, option := range options {
		option(&loadConfig)
	}
	return loadConfig
}
