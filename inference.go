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
// # Generation options
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
//	for m := range inference.Discover("/path/to/models/") {
//	    fmt.Printf("%s (%s)\n", m.Path, m.ModelType)
//	}
package inference

import (
	"context"
	"iter"
	"maps"
	"slices"
	"sync"
	"time"

	"dappco.re/go/core"
)

//	for tok := range m.Generate(ctx, prompt) {
//	    fmt.Print(tok.Text) // tok.ID holds the vocab index
//	}
type Token struct {
	ID   int32
	Text string
}

//	messages := []inference.Message{
//	    {Role: "system",    Content: "You are a helpful assistant."},
//	    {Role: "user",      Content: "What is 2+2?"},
//	    {Role: "assistant", Content: "4"},
//	    {Role: "user",      Content: "Are you sure?"},
//	}
type Message struct {
	Role    string `json:"role"` // "system", "user", "assistant"
	Content string `json:"content"`
}

// results, _ := m.Classify(ctx, []string{"positive", "negative"})
// label := results[0].Token.Text  // sampled token at last position
// logits := results[0].Logits     // only populated when WithLogits() is set
type ClassifyResult struct {
	Token  Token     // Sampled/greedy token at last prompt position
	Logits []float32 // Raw vocab-sized logits (only when WithLogits is set)
}

// batched, _ := m.BatchGenerate(ctx, prompts, inference.WithMaxTokens(64))
//
//	for i, r := range batched {
//	    if r.Err != nil { continue }
//	    for _, tok := range r.Tokens { fmt.Print(tok.Text) }
//	}
type BatchResult struct {
	Tokens []Token // All generated tokens for this prompt
	Err    error   // Per-prompt error (context cancel, OOM, etc.)
}

// Retrieved via TextModel.Metrics() after Generate, Chat, Classify, or BatchGenerate.
//
//	m := model.Metrics()
//	fmt.Printf("prefill: %.0f tok/s  decode: %.0f tok/s\n", m.PrefillTokensPerSec, m.DecodeTokensPerSec)
//	fmt.Printf("peak GPU memory: %d MiB\n", m.PeakMemoryBytes>>20)
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

// info := model.Info()
// fmt.Printf("%s %d-bit quant, %d layers, vocab %d\n", info.Architecture, info.QuantBits, info.NumLayers, info.VocabSize)
type ModelInfo struct {
	Architecture string // e.g. "gemma3", "qwen3", "llama"
	VocabSize    int    // Vocabulary size
	NumLayers    int    // Number of transformer layers
	HiddenSize   int    // Hidden dimension
	QuantBits    int    // Quantisation bits (0 = unquantised, 4 = 4-bit, 8 = 8-bit)
	QuantGroup   int    // Quantisation group size (0 if unquantised)
}

// Keys is indexed [layer][head][position*head_dim] — flattened per head.
//
//	snap, _ := inspector.InspectAttention(ctx, prompt)
//	layer0Head0 := snap.Keys[0][0] // flat float32 of len seq_len*head_dim
type AttentionSnapshot struct {
	NumLayers     int           `json:"num_layers"`
	NumHeads      int           `json:"num_heads"` // num_kv_heads (may differ from query heads in GQA)
	SeqLen        int           `json:"seq_len"`   // number of tokens in the prompt
	HeadDim       int           `json:"head_dim"`
	NumQueryHeads int           `json:"num_query_heads"` // num_attention_heads (0 = Q not available)
	Keys          [][][]float32 `json:"keys"`            // [layer][head] → flat float32 of len seq_len*head_dim
	Queries       [][][]float32 `json:"queries"`         // [layer][head] → flat float32 (nil if K-only)
	Architecture  string        `json:"architecture"`
}

// if snap.HasQueries() { processQK(snap.Queries, snap.Keys) }
func (s *AttentionSnapshot) HasQueries() bool {
	return s != nil && s.Queries != nil && len(s.Queries) > 0
}

//	if inspector, ok := model.(inference.AttentionInspector); ok {
//	    snap, err := inspector.InspectAttention(ctx, prompt)
//	}
type AttentionInspector interface {
	InspectAttention(ctx context.Context, prompt string, opts ...GenerateOption) (*AttentionSnapshot, error)
}

// TextModel generates text from a loaded model.
//
//	m, _ := inference.LoadModel("/models/gemma3-1b")
//	defer m.Close()
//	for tok := range m.Generate(ctx, "Hello") { fmt.Print(tok.Text) }
type TextModel interface {
	// Generate streams tokens for the given prompt.
	//
	//	for tok := range m.Generate(ctx, "The quick brown fox", inference.WithMaxTokens(64)) {
	//	    fmt.Print(tok.Text)
	//	}
	//	if err := m.Err(); err != nil { return err }
	Generate(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq[Token]

	// Chat streams tokens from a multi-turn conversation using the model's native template.
	//
	//	for tok := range m.Chat(ctx, []inference.Message{{Role: "user", Content: "Hi"}}) {
	//	    fmt.Print(tok.Text)
	//	}
	Chat(ctx context.Context, messages []Message, opts ...GenerateOption) iter.Seq[Token]

	// Classify runs batched prefill-only inference — fast path for classification tasks.
	// Each prompt gets one forward pass; the token at the last position is sampled.
	//
	//	results, _ := m.Classify(ctx, []string{"positive review", "negative review"})
	//	label := results[0].Token.Text
	Classify(ctx context.Context, prompts []string, opts ...GenerateOption) ([]ClassifyResult, error)

	// BatchGenerate runs batched autoregressive generation up to MaxTokens per prompt.
	//
	//	results, _ := m.BatchGenerate(ctx, prompts, inference.WithMaxTokens(128))
	//	for i, r := range results { fmt.Println(i, r.Tokens) }
	BatchGenerate(ctx context.Context, prompts []string, opts ...GenerateOption) ([]BatchResult, error)

	// ModelType is the architecture string from config.json ("gemma3", "qwen3", "llama3").
	//
	//	fmt.Println(m.ModelType()) // "gemma3", "qwen3", "llama3"
	ModelType() string

	// Info returns architecture metadata for the loaded model.
	//
	//	info := m.Info()
	//	fmt.Printf("%s %d-bit quant, %d layers, vocab %d\n", info.Architecture, info.QuantBits, info.NumLayers, info.VocabSize)
	Info() ModelInfo

	// Metrics returns throughput and memory counters from the last completed operation.
	// Valid after Generate (iterator exhausted), Chat, Classify, or BatchGenerate.
	//
	//	fmt.Printf("%.0f tok/s decode\n", m.Metrics().DecodeTokensPerSec)
	Metrics() GenerateMetrics

	// Err holds any error from the last Generate or Chat call.
	// Check after the iterator stops to distinguish normal EOS from errors.
	//
	//	for tok := range m.Generate(ctx, prompt) { ... }
	//	if err := m.Err(); err != nil { return err }
	Err() error

	// Close releases GPU memory, KV caches, and any subprocess.
	//
	//	defer m.Close()
	Close() error
}

// func init() { inference.Register(metal.NewBackend()) } // called from backend packages
type Backend interface {
	// Name is the stable identifier used for registration and selection.
	//
	//	b.Name() // "metal", "rocm", "llama_cpp"
	Name() string

	// LoadModel reads the model directory at path and returns a ready TextModel.
	//
	//	m, err := b.LoadModel("/models/gemma3-1b", inference.WithContextLen(4096))
	LoadModel(path string, opts ...LoadOption) (TextModel, error)

	// Available reports whether the required hardware or driver is present at runtime.
	//
	//	if !b.Available() { skip } // e.g. Metal on non-Apple hardware returns false
	Available() bool
}

var (
	backendsMu            sync.RWMutex
	backends              = map[string]Backend{}
	preferredBackendOrder = []string{"metal", "rocm", "llama_cpp"}
	preferredBackendSet   = map[string]struct{}{
		"metal":     {},
		"rocm":      {},
		"llama_cpp": {},
	}
)

func snapshotBackends() map[string]Backend {
	backendsMu.RLock()
	snap := maps.Clone(backends)
	backendsMu.RUnlock()
	return snap
}

// Register adds b to the global registry, overwriting any existing entry with the same name.
//
//	func init() { inference.Register(metal.NewBackend()) }
func Register(b Backend) {
	if b == nil {
		return
	}
	backendsMu.Lock()
	defer backendsMu.Unlock()
	backends[b.Name()] = b
}

// Get looks up a backend by name. Returns (nil, false) when not registered.
//
//	b, ok := inference.Get("metal")
func Get(name string) (Backend, bool) {
	backendsMu.RLock()
	defer backendsMu.RUnlock()
	b, ok := backends[name]
	return b, ok
}

// names := inference.List() // ["llama_cpp", "metal", "rocm"]
func List() []string {
	return slices.Sorted(maps.Keys(snapshotBackends()))
}

//	for name, b := range inference.All() {
//	    fmt.Println(name, b.Available())
//	}
func All() iter.Seq2[string, Backend] {
	snap := snapshotBackends()
	names := slices.Sorted(maps.Keys(snap))
	return func(yield func(string, Backend) bool) {
		for _, name := range names {
			if !yield(name, snap[name]) {
				return
			}
		}
	}
}

// Default picks the first available backend in preference order: metal → rocm → llama_cpp → any.
//
//	b, err := inference.Default() // returns metal on Apple Silicon if available
func Default() (Backend, error) {
	snap := snapshotBackends()

	// Platform preference order
	for _, name := range preferredBackendOrder {
		if b, ok := snap[name]; ok && b.Available() {
			return b, nil
		}
	}
	// Fall back to any available
	for _, name := range slices.Sorted(maps.Keys(snap)) {
		if _, ok := preferredBackendSet[name]; ok {
			continue
		}
		if backend := snap[name]; backend.Available() {
			return backend, nil
		}
	}
	return nil, core.E("inference.Default", "no backends registered", nil)
}

// m, err := inference.LoadModel("/models/gemma3-1b")
// m, err := inference.LoadModel("/models/qwen3-4b", inference.WithBackend("rocm"), inference.WithContextLen(8192))
func LoadModel(path string, opts ...LoadOption) (TextModel, error) {
	cfg := ApplyLoadOpts(opts)
	if cfg.Backend != "" {
		b, ok := Get(cfg.Backend)
		if !ok {
			return nil, core.E("inference.LoadModel", core.Sprintf("backend %q not registered", cfg.Backend), nil)
		}
		if !b.Available() {
			return nil, core.E("inference.LoadModel", core.Sprintf("backend %q not available on this hardware", cfg.Backend), nil)
		}
		return b.LoadModel(path, opts...)
	}
	b, err := Default()
	if err != nil {
		return nil, err
	}
	return b.LoadModel(path, opts...)
}
