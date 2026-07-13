package inference

import (
	"slices"

	"dappco.re/go/inference/eval/probe"
)

// inference.GenerateConfig{MaxTokens: 256, Temperature: 0.7, TopK: 40}
// inference.GenerateConfig{MaxTokens: 64, StopTokens: []int32{2}, RepeatPenalty: 1.1}
type GenerateConfig struct {
	MaxTokens   int
	Temperature float32
	TopK        int
	TopP        float32
	MinP        float32
	// Temperature/TopK/TopP/MinP each carry a companion "was this set" flag
	// (the SeedSet precedent below): the field's zero value is ALSO a
	// meaningful explicit request — Temperature 0 = greedy, TopK/TopP/MinP 0 =
	// disabled — so the zero value alone cannot distinguish "caller asked for
	// this" from "caller said nothing". A backend folding a model's declared
	// generation_config defaults reads the flag, not the value: a false flag
	// means the caller left the field unset, so the declared default may apply;
	// a true flag means honour the request's value verbatim (even a zero). The
	// [WithTemperature]/[WithTopK]/[WithTopP]/[WithMinP] setters set these; a
	// direct struct literal leaves them false (unset) unless it sets them
	// explicitly, exactly as Seed/SeedSet already behave.
	TemperatureSet bool
	TopKSet        bool
	TopPSet        bool
	MinPSet        bool
	Seed           uint64
	SeedSet        bool
	StopTokens     []int32
	SuppressTokens []int32
	// MinTokensBeforeStop masks stop tokens until at least this many tokens
	// have been emitted, matching backends that avoid immediate turn closure.
	MinTokensBeforeStop int
	RepeatPenalty       float32
	ReturnLogits        bool // Return raw logits in ClassifyResult (default false)
	// EnableThinking toggles reasoning for models that support it (e.g. Gemma 4).
	// nil = model default; &true = on; &false = off. Backends ignore it when the
	// loaded architecture has no thinking mode.
	EnableThinking *bool
	// ThinkingBudget caps tokens spent inside a reasoning model's thought
	// channel; on overrun the backend forces the channel close so a visible
	// answer is produced rather than the whole budget being spent reasoning.
	// 0 = unlimited. Ignored by architectures with no thinking mode.
	ThinkingBudget int
	// Thinking is the resolved thought-channel processing policy (show, hide,
	// or capture reasoning blocks). EnableThinking is the API-level on/off
	// intent; serving layers resolve it into this policy for the engine. The
	// zero value leaves the engine's default handling in place.
	Thinking ThinkingConfig
	// Engine trace + cache-hygiene knobs (engine-neutral operational
	// controls; a backend without the facility ignores them).
	TraceTokenPhases             bool // per-token coarse phase timing to the engine trace log
	TraceTokenText               bool // include decoded token text in the trace (debug only)
	GenerationClearCache         bool // drop device caches between generations
	GenerationClearCacheInterval int  // clear every N tokens while generating; 0 = never
	// ProbeSink receives research-telemetry events emitted while generating
	// (attention/logit probes). nil = probing off.
	ProbeSink probe.Sink
}

// cfg := inference.DefaultGenerateConfig() // Temperature=0.0 (greedy), RepeatPenalty=1.0
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		// MaxTokens is deliberately NOT defaulted. It is a caller-supplied output
		// cap; absent (0) the backend resolves it to the model's context at
		// generation time. A fixed default truncates every generation at a guess.
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
	return func(c *GenerateConfig) {
		c.Temperature = t
		c.TemperatureSet = true
	}
}

// WithTopK limits sampling to the top-k highest-probability tokens. 0 = disabled.
//
//	inference.WithTopK(40) // typical value for creative generation
func WithTopK(k int) GenerateOption {
	return func(c *GenerateConfig) {
		c.TopK = k
		c.TopKSet = true
	}
}

// WithTopP sets nucleus sampling — only tokens covering cumulative probability p are considered. 0 = disabled.
//
//	inference.WithTopP(0.9) // typical nucleus sampling threshold
func WithTopP(p float32) GenerateOption {
	return func(c *GenerateConfig) {
		c.TopP = p
		c.TopPSet = true
	}
}

// WithMinP sets minimum-probability sampling relative to the best token.
//
//	inference.WithMinP(0.05) // drop tokens below 5% of the top-token probability
func WithMinP(p float32) GenerateOption {
	return func(c *GenerateConfig) {
		c.MinP = p
		c.MinPSet = true
	}
}

// WithSeed makes stochastic sampling reproducible for this request.
//
//	inference.WithSeed(42) // deterministic sampling for a fixed seed
func WithSeed(seed uint64) GenerateOption {
	return func(c *GenerateConfig) {
		c.Seed = seed
		c.SeedSet = true
	}
}

// WithStopTokens halts generation as soon as any listed token ID is sampled.
//
//	inference.WithStopTokens(2)       // EOS token only
//	inference.WithStopTokens(2, 1, 0) // EOS + pad tokens
func WithStopTokens(ids ...int32) GenerateOption {
	return func(c *GenerateConfig) { c.StopTokens = slices.Clone(ids) }
}

// WithSuppressTokens masks token IDs out of the sampling distribution.
//
//	inference.WithSuppressTokens(1, 2) // never emit these ids
func WithSuppressTokens(ids ...int32) GenerateOption {
	return func(c *GenerateConfig) { c.SuppressTokens = slices.Clone(ids) }
}

// WithMinTokensBeforeStop suppresses stop tokens until n tokens have been emitted.
//
//	inference.WithMinTokensBeforeStop(8) // force a short visible answer
func WithMinTokensBeforeStop(n int) GenerateOption {
	return func(c *GenerateConfig) { c.MinTokensBeforeStop = n }
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

// WithEnableThinking sets the reasoning toggle for thinking-capable models.
// Pass nil for the model default, &true to force on, &false to force off.
//
//	off := false
//	m.Chat(ctx, msgs, inference.WithEnableThinking(&off)) // disable Gemma 4 reasoning
func WithEnableThinking(v *bool) GenerateOption {
	return func(c *GenerateConfig) { c.EnableThinking = v }
}

// WithThinkingBudget caps tokens spent in the thought channel; 0 = unlimited.
//
//	m.Chat(ctx, msgs, inference.WithThinkingBudget(256)) // think briefly, then answer
func WithThinkingBudget(tokens int) GenerateOption {
	return func(c *GenerateConfig) { c.ThinkingBudget = tokens }
}

// WithThinking sets the resolved thought-channel processing policy — what the
// engine does with reasoning blocks (show, hide, or capture them).
//
//	m.Generate(ctx, prompt, inference.WithThinking(inference.ThinkingConfig{Mode: inference.ThinkingHide}))
func WithThinking(cfg ThinkingConfig) GenerateOption {
	return func(c *GenerateConfig) { c.Thinking = cfg }
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
