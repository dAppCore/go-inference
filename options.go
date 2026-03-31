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
	ReturnLogits  bool
}

// config := inference.DefaultGenerateConfig() // MaxTokens=256, Temperature=0.0
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		MaxTokens:   256,
		Temperature: 0.0,
	}
}

// generateOptions := []inference.GenerateOption{inference.WithMaxTokens(128), inference.WithTemperature(0.7)}
type GenerateOption func(*GenerateConfig)

// inference.WithMaxTokens(128)  // short reply
// inference.WithMaxTokens(2048) // long-form generation
func WithMaxTokens(maxTokens int) GenerateOption {
	return func(generateConfig *GenerateConfig) { generateConfig.MaxTokens = maxTokens }
}

// inference.WithTemperature(0.0) // deterministic
// inference.WithTemperature(0.7) // balanced creativity
// inference.WithTemperature(1.5) // high variance
func WithTemperature(temperature float32) GenerateOption {
	return func(generateConfig *GenerateConfig) { generateConfig.Temperature = temperature }
}

// inference.WithTopK(40) // typical value for creative generation
func WithTopK(topK int) GenerateOption {
	return func(generateConfig *GenerateConfig) { generateConfig.TopK = topK }
}

// inference.WithTopP(0.9) // typical nucleus sampling threshold
func WithTopP(topP float32) GenerateOption {
	return func(generateConfig *GenerateConfig) { generateConfig.TopP = topP }
}

// inference.WithStopTokens(2)       // EOS token only
// inference.WithStopTokens(2, 1, 0) // EOS + pad tokens
func WithStopTokens(stopTokenIDs ...int32) GenerateOption {
	return func(generateConfig *GenerateConfig) { generateConfig.StopTokens = stopTokenIDs }
}

// inference.WithRepeatPenalty(1.1) // mild repetition suppression
// inference.WithRepeatPenalty(1.5) // strong repetition suppression
func WithRepeatPenalty(repeatPenalty float32) GenerateOption {
	return func(generateConfig *GenerateConfig) { generateConfig.RepeatPenalty = repeatPenalty }
}

// inference.WithLogits() // enable logit capture for classification scoring
func WithLogits() GenerateOption {
	return func(generateConfig *GenerateConfig) { generateConfig.ReturnLogits = true }
}

// generateConfig := inference.ApplyGenerateOpts(generateOptions)
func ApplyGenerateOpts(generateOptions []GenerateOption) GenerateConfig {
	generateConfig := DefaultGenerateConfig()
	for _, generateOption := range generateOptions {
		generateOption(&generateConfig)
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

// loadOptions := []inference.LoadOption{inference.WithBackend("metal"), inference.WithContextLen(4096)}
type LoadOption func(*LoadConfig)

// inference.WithBackend("metal")     // Apple Silicon GPU
// inference.WithBackend("rocm")      // AMD GPU
// inference.WithBackend("llama_cpp") // CPU fallback
func WithBackend(backendName string) LoadOption {
	return func(loadConfig *LoadConfig) { loadConfig.Backend = backendName }
}

// inference.WithContextLen(4096)  // standard context
// inference.WithContextLen(32768) // extended context
func WithContextLen(contextLength int) LoadOption {
	return func(loadConfig *LoadConfig) { loadConfig.ContextLen = contextLength }
}

// inference.WithGPULayers(-1) // full GPU offload (default)
// inference.WithGPULayers(0)  // CPU-only inference
// inference.WithGPULayers(24) // partial offload (24 layers to GPU)
func WithGPULayers(gpuLayerCount int) LoadOption {
	return func(loadConfig *LoadConfig) { loadConfig.GPULayers = gpuLayerCount }
}

// inference.WithParallelSlots(4) // allow 4 concurrent inference requests
func WithParallelSlots(parallelSlotCount int) LoadOption {
	return func(loadConfig *LoadConfig) { loadConfig.ParallelSlots = parallelSlotCount }
}

// inference.WithAdapterPath("/models/lora/domain-v2") // load fine-tuned adapter
func WithAdapterPath(path string) LoadOption {
	return func(loadConfig *LoadConfig) { loadConfig.AdapterPath = path }
}

// loadConfig := inference.ApplyLoadOpts(loadOptions)
func ApplyLoadOpts(loadOptions []LoadOption) LoadConfig {
	loadConfig := LoadConfig{
		GPULayers: -1,
	}
	for _, loadOption := range loadOptions {
		loadOption(&loadConfig)
	}
	return loadConfig
}
