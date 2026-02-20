// Package inference defines shared interfaces for text generation backends.
//
// This package is the contract between GPU-specific backends (go-mlx, go-rocm)
// and consumers (go-ml, go-ai, go-i18n). It has zero dependencies and compiles
// on all platforms.
//
// # Backend registration
//
// Backend implementations register via init() with build tags:
//
//	// go-mlx: //go:build darwin && arm64
//	func init() { inference.Register(metal.NewBackend()) }
//
//	// go-rocm: //go:build linux && amd64
//	func init() { inference.Register(rocm.NewBackend()) }
//
// # Loading and generating
//
//	m, err := inference.LoadModel("/path/to/model/")
//	defer m.Close()
//
//	ctx := context.Background()
//	for tok := range m.Generate(ctx, "prompt", inference.WithMaxTokens(128)) {
//	    fmt.Print(tok.Text)
//	}
//	if err := m.Err(); err != nil { log.Fatal(err) }
//
// # Chat, classify, and batch generate
//
// [TextModel] supports multi-turn chat (with model-native templates),
// batch classification (prefill-only, fast path), and batch generation:
//
//	// Chat
//	for tok := range m.Chat(ctx, []inference.Message{
//	    {Role: "user", Content: "Hello"},
//	}, inference.WithMaxTokens(64)) {
//	    fmt.Print(tok.Text)
//	}
//
//	// Classify — single forward pass per prompt
//	results, _ := m.Classify(ctx, prompts, inference.WithTemperature(0))
//
//	// Batch generate — parallel autoregressive decoding
//	batched, _ := m.BatchGenerate(ctx, prompts, inference.WithMaxTokens(32))
//
// # Functional options
//
// Generation and loading are configured via functional options:
//
//	inference.WithMaxTokens(256)     // cap output length
//	inference.WithTemperature(0.7)   // sampling temperature
//	inference.WithTopK(40)           // top-k sampling
//	inference.WithRepeatPenalty(1.1) // discourage repetition
//	inference.WithContextLen(4096)   // limit KV cache memory
//
// # Model discovery
//
// [Discover] scans a directory for model directories (config.json + *.safetensors):
//
//	models, _ := inference.Discover("/path/to/models/")
//	for _, d := range models {
//	    fmt.Printf("%s (%s)\n", d.Path, d.ModelType)
//	}
package inference

import (
	"context"
	"fmt"
	"iter"
	"sync"
	"time"
)

// Token represents a single generated token for streaming.
type Token struct {
	ID   int32
	Text string
}

// Message represents a chat message for multi-turn conversation.
type Message struct {
	Role    string `json:"role"`    // "system", "user", "assistant"
	Content string `json:"content"`
}

// ClassifyResult holds the output for a single prompt in a batch classification.
type ClassifyResult struct {
	Token  Token     // Sampled/greedy token at last prompt position
	Logits []float32 // Raw vocab-sized logits (only when WithLogits is set)
}

// BatchResult holds the output for a single prompt in batch generation.
type BatchResult struct {
	Tokens []Token // All generated tokens for this prompt
	Err    error   // Per-prompt error (context cancel, OOM, etc.)
}

// GenerateMetrics holds performance metrics from the last inference operation.
// Retrieved via TextModel.Metrics() after Generate, Chat, Classify, or BatchGenerate.
type GenerateMetrics struct {
	// Token counts
	PromptTokens    int // Input tokens (sum across batch for batch ops)
	GeneratedTokens int // Output tokens generated

	// Timing
	PrefillDuration time.Duration // Time to process the prompt(s)
	DecodeDuration  time.Duration // Time for autoregressive decoding
	TotalDuration   time.Duration // Wall-clock time for the full operation

	// Throughput (computed)
	PrefillTokensPerSec float64 // PromptTokens / PrefillDuration
	DecodeTokensPerSec  float64 // GeneratedTokens / DecodeDuration

	// Memory (Metal/GPU)
	PeakMemoryBytes   uint64 // Peak GPU memory during this operation
	ActiveMemoryBytes uint64 // Active GPU memory after operation
}

// ModelInfo holds metadata about a loaded model.
type ModelInfo struct {
	Architecture string // e.g. "gemma3", "qwen3", "llama"
	VocabSize    int    // Vocabulary size
	NumLayers    int    // Number of transformer layers
	HiddenSize   int    // Hidden dimension
	QuantBits    int    // Quantisation bits (0 = unquantised, 4 = 4-bit, 8 = 8-bit)
	QuantGroup   int    // Quantisation group size (0 if unquantised)
}

// TextModel generates text from a loaded model.
type TextModel interface {
	// Generate streams tokens for the given prompt.
	Generate(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq[Token]

	// Chat streams tokens from a multi-turn conversation.
	// The model applies its native chat template.
	Chat(ctx context.Context, messages []Message, opts ...GenerateOption) iter.Seq[Token]

	// Classify runs batched prefill-only inference. Each prompt gets a single
	// forward pass and the token at the last position is sampled. This is the
	// fast path for classification tasks (e.g. domain labelling).
	Classify(ctx context.Context, prompts []string, opts ...GenerateOption) ([]ClassifyResult, error)

	// BatchGenerate runs batched autoregressive generation. Each prompt is
	// decoded up to MaxTokens. Returns all generated tokens per prompt.
	BatchGenerate(ctx context.Context, prompts []string, opts ...GenerateOption) ([]BatchResult, error)

	// ModelType returns the architecture identifier (e.g. "gemma3", "qwen3", "llama3").
	ModelType() string

	// Info returns metadata about the loaded model (architecture, quantisation, etc.).
	Info() ModelInfo

	// Metrics returns performance metrics from the last inference operation.
	// Valid after Generate (iterator exhausted), Chat, Classify, or BatchGenerate.
	Metrics() GenerateMetrics

	// Err returns the error from the last Generate/Chat call, if any.
	// Check this after the iterator stops to distinguish EOS from errors.
	Err() error

	// Close releases all resources (GPU memory, caches, subprocess).
	Close() error
}

// Backend is a named inference engine that can load models.
type Backend interface {
	// Name returns the backend identifier (e.g. "metal", "rocm", "llama_cpp").
	Name() string

	// LoadModel loads a model from the given path.
	LoadModel(path string, opts ...LoadOption) (TextModel, error)

	// Available reports whether this backend can run on the current hardware.
	Available() bool
}

var (
	backendsMu sync.RWMutex
	backends   = map[string]Backend{}
)

// Register adds a backend to the registry. Typically called from init().
func Register(b Backend) {
	backendsMu.Lock()
	defer backendsMu.Unlock()
	backends[b.Name()] = b
}

// Get returns a registered backend by name.
func Get(name string) (Backend, bool) {
	backendsMu.RLock()
	defer backendsMu.RUnlock()
	b, ok := backends[name]
	return b, ok
}

// List returns the names of all registered backends.
func List() []string {
	backendsMu.RLock()
	defer backendsMu.RUnlock()
	names := make([]string, 0, len(backends))
	for name := range backends {
		names = append(names, name)
	}
	return names
}

// Default returns the first available backend.
// Prefers "metal" on macOS, "rocm" on Linux, then any registered backend.
func Default() (Backend, error) {
	backendsMu.RLock()
	defer backendsMu.RUnlock()

	// Platform preference order
	for _, name := range []string{"metal", "rocm", "llama_cpp"} {
		if b, ok := backends[name]; ok && b.Available() {
			return b, nil
		}
	}
	// Fall back to any available
	for _, b := range backends {
		if b.Available() {
			return b, nil
		}
	}
	return nil, fmt.Errorf("inference: no backends registered (import a backend package)")
}

// LoadModel loads a model using the specified or default backend.
func LoadModel(path string, opts ...LoadOption) (TextModel, error) {
	cfg := ApplyLoadOpts(opts)
	if cfg.Backend != "" {
		b, ok := Get(cfg.Backend)
		if !ok {
			return nil, fmt.Errorf("inference: backend %q not registered", cfg.Backend)
		}
		if !b.Available() {
			return nil, fmt.Errorf("inference: backend %q not available on this hardware", cfg.Backend)
		}
		return b.LoadModel(path, opts...)
	}
	b, err := Default()
	if err != nil {
		return nil, err
	}
	return b.LoadModel(path, opts...)
}
