package inference

// inference.GenerateConfig{MaxTokens: 256, Temperature: 0.7, TopK: 40}
// inference.GenerateConfig{MaxTokens: 64, StopTokens: []int32{2}, RepeatPenalty: 1.1}
type GenerateConfig struct {
	MaxTokens     int
	Temperature   float32
	TopK          int
	TopP          float32
	StopTokens    []int32
	RepeatPenalty float32
	ReturnLogits  bool // Return raw logits in ClassifyResult (default false)
}

// cfg := inference.DefaultGenerateConfig() // MaxTokens=256, Temperature=0.0 (greedy), RepeatPenalty=1.0
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		MaxTokens:     256,
		Temperature:   0.0, // greedy
		RepeatPenalty: 1.0, // no penalty
	}
}

// Used by Generate, Chat, Classify, and BatchGenerate.
//
//	m.Generate(ctx, prompt, inference.WithMaxTokens(128), inference.WithTemperature(0.7))
type GenerateOption func(*GenerateConfig)

// WithMaxTokens caps the number of tokens generated.
//
//	inference.WithMaxTokens(128)  // short reply
//	inference.WithMaxTokens(2048) // long-form generation
func WithMaxTokens(n int) GenerateOption {
	return func(c *GenerateConfig) { c.MaxTokens = n }
}

// WithTemperature controls randomness. 0 = deterministic greedy, >1 = high variance.
//
//	inference.WithTemperature(0.0) // deterministic
//	inference.WithTemperature(0.7) // balanced creativity
//	inference.WithTemperature(1.5) // high variance
func WithTemperature(t float32) GenerateOption {
	return func(c *GenerateConfig) { c.Temperature = t }
}

// WithTopK limits sampling to the top-k highest-probability tokens. 0 = disabled.
//
//	inference.WithTopK(40) // typical value for creative generation
func WithTopK(k int) GenerateOption {
	return func(c *GenerateConfig) { c.TopK = k }
}

// WithTopP sets nucleus sampling — only tokens covering cumulative probability p are considered. 0 = disabled.
//
//	inference.WithTopP(0.9) // typical nucleus sampling threshold
func WithTopP(p float32) GenerateOption {
	return func(c *GenerateConfig) { c.TopP = p }
}

// WithStopTokens halts generation as soon as any listed token ID is sampled.
//
//	inference.WithStopTokens(2)       // EOS token only
//	inference.WithStopTokens(2, 1, 0) // EOS + pad tokens
func WithStopTokens(ids ...int32) GenerateOption {
	return func(c *GenerateConfig) { c.StopTokens = ids }
}

// WithRepeatPenalty penalises repeated tokens. 1.0 = no penalty.
//
//	inference.WithRepeatPenalty(1.1) // mild repetition suppression
//	inference.WithRepeatPenalty(1.5) // strong repetition suppression
func WithRepeatPenalty(p float32) GenerateOption {
	return func(c *GenerateConfig) { c.RepeatPenalty = p }
}

// WithLogits requests raw logits in ClassifyResult. Off by default to save memory.
//
//	inference.WithLogits() // enable logit capture for classification scoring
func WithLogits() GenerateOption {
	return func(c *GenerateConfig) { c.ReturnLogits = true }
}

// cfg := inference.ApplyGenerateOpts(opts) // used internally by backends
func ApplyGenerateOpts(opts []GenerateOption) GenerateConfig {
	cfg := DefaultGenerateConfig()
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	return cfg
}

// inference.LoadConfig{Backend: "metal", ContextLen: 4096, GPULayers: -1}
// inference.LoadConfig{Backend: "rocm", AdapterPath: "/models/lora/v1"}
type LoadConfig struct {
	Backend       string // "metal", "rocm", "llama_cpp" (empty = auto-detect)
	ContextLen    int    // Context window size (0 = model default)
	GPULayers     int    // Number of layers to offload to GPU (-1 = all, 0 = none)
	ParallelSlots int    // Number of concurrent inference slots (0 = server default)
	AdapterPath   string // Path to LoRA adapter directory (empty = no adapter)
}

// Used by LoadModel and LoadTrainable.
//
//	inference.LoadModel("/models/gemma3-1b", inference.WithBackend("metal"), inference.WithContextLen(4096))
type LoadOption func(*LoadConfig)

// WithBackend selects a specific inference backend by name.
//
//	inference.WithBackend("metal")     // Apple Silicon GPU
//	inference.WithBackend("rocm")      // AMD GPU
//	inference.WithBackend("llama_cpp") // CPU fallback
func WithBackend(name string) LoadOption {
	return func(c *LoadConfig) { c.Backend = name }
}

// WithContextLen caps the KV cache to n tokens. 0 = use the model's built-in default.
//
//	inference.WithContextLen(4096)  // standard context
//	inference.WithContextLen(32768) // extended context
func WithContextLen(n int) LoadOption {
	return func(c *LoadConfig) { c.ContextLen = n }
}

// WithGPULayers offloads n transformer layers to GPU. -1 = all (default), 0 = CPU-only.
//
//	inference.WithGPULayers(-1) // full GPU offload (default)
//	inference.WithGPULayers(0)  // CPU-only inference
//	inference.WithGPULayers(24) // partial offload (24 layers to GPU)
func WithGPULayers(n int) LoadOption {
	return func(c *LoadConfig) { c.GPULayers = n }
}

// WithParallelSlots allows n concurrent Generate/Chat calls at the cost of VRAM. 0 = backend default.
//
//	inference.WithParallelSlots(4) // allow 4 concurrent inference requests
func WithParallelSlots(n int) LoadOption {
	return func(c *LoadConfig) { c.ParallelSlots = n }
}

// WithAdapterPath injects a LoRA adapter at load time without fusing into the base model.
// The directory must contain adapter_config.json and adapter safetensors files.
//
//	inference.WithAdapterPath("/models/lora/domain-v2") // load fine-tuned adapter
func WithAdapterPath(path string) LoadOption {
	return func(c *LoadConfig) { c.AdapterPath = path }
}

// cfg := inference.ApplyLoadOpts(opts) // used internally by LoadModel
func ApplyLoadOpts(opts []LoadOption) LoadConfig {
	cfg := LoadConfig{
		GPULayers: -1, // default: full GPU offload
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	return cfg
}
