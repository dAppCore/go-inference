// Package inference defines the shared contract for text-generation backends.
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
//	m, err := inference.LoadModel("/models/gemma3-1b")
//	defer m.Close()
//
//	ctx := context.Background()
//	for tok := range m.Generate(ctx, "Hello", inference.WithMaxTokens(128)) {
//	    fmt.Print(tok.Text)
//	}
//	if err := m.Err(); err != nil { log.Fatal(err) }
//
//	for tok := range m.Chat(ctx, []inference.Message{
//	    {Role: "user", Content: "Hello"},
//	}, inference.WithMaxTokens(64)) {
//	    fmt.Print(tok.Text)
//	}
//
//	results, _ := m.Classify(ctx, []string{"positive", "negative"}, inference.WithTemperature(0))
//	batched, _ := m.BatchGenerate(ctx, []string{"First", "Second"}, inference.WithMaxTokens(32))
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
//	for model := range inference.Discover("/path/to/models/") {
//	    fmt.Printf("%s (%s)\n", model.Path, model.ModelType)
//	}
package inference

import (
	"context"
	"fmt"
	"iter"
	"maps"
	"slices"
	"sync"
	"time"
)

// token := inference.Token{ID: 123, Text: "hello"}
type Token struct {
	ID   int32
	Text string
}

//	messages := []inference.Message{
//		{Role: "system", Content: "You are a helpful assistant."},
//		{Role: "user", Content: "What is 2+2?"},
//		{Role: "assistant", Content: "4"},
//		{Role: "user", Content: "Are you sure?"},
//	}
type Message struct {
	Role    string `json:"role"` // "system", "user", "assistant"
	Content string `json:"content"`
}

// results, _ := m.Classify(ctx, []string{"positive", "negative"})
// label := results[0].Token.Text
// logits := results[0].Logits
type ClassifyResult struct {
	Token  Token     // Sampled or greedy token at the last prompt position
	Logits []float32 // Raw vocab-sized logits when WithLogits is set
}

// batched, _ := m.BatchGenerate(ctx, []string{"First", "Second"}, inference.WithMaxTokens(64))
//
//	for i, result := range batched {
//		if result.Err != nil {
//			continue
//		}
//		for _, tok := range result.Tokens {
//			fmt.Print(tok.Text)
//		}
//	}
type BatchResult struct {
	Tokens []Token // Generated tokens for this prompt
	Err    error   // Per-prompt error
}

// metrics := model.Metrics()
// fmt.Printf("prefill: %.0f tok/s  decode: %.0f tok/s\n", metrics.PrefillTokensPerSec, metrics.DecodeTokensPerSec)
// fmt.Printf("peak GPU memory: %d MiB\n", metrics.PeakMemoryBytes>>20)
type GenerateMetrics struct {
	PromptTokens    int // Input token count
	GeneratedTokens int // Output token count

	PrefillDuration time.Duration // Prompt processing time
	DecodeDuration  time.Duration // Autoregressive decode time
	TotalDuration   time.Duration // Whole-operation wall time

	PrefillTokensPerSec float64 // PromptTokens / PrefillDuration
	DecodeTokensPerSec  float64 // GeneratedTokens / DecodeDuration

	PeakMemoryBytes   uint64 // Peak GPU memory
	ActiveMemoryBytes uint64 // GPU memory after the call
}

// info := model.Info()
// fmt.Printf("%s %d-bit quant, %d layers, vocab %d\n", info.Architecture, info.QuantBits, info.NumLayers, info.VocabSize)
type ModelInfo struct {
	Architecture string // Model architecture
	VocabSize    int    // Vocabulary size
	NumLayers    int    // Transformer layer count
	HiddenSize   int    // Hidden dimension
	QuantBits    int    // Quantisation bits (0 = unquantised)
	QuantGroup   int    // Quantisation group size
}

// snap, _ := inspector.InspectAttention(ctx, prompt)
//
//	if snap.HasQueries() {
//		processQK(snap.Queries, snap.Keys)
//	}
type AttentionSnapshot struct {
	NumLayers     int           `json:"num_layers"`
	NumHeads      int           `json:"num_heads"` // num_kv_heads (may differ from query heads in GQA)
	SeqLen        int           `json:"seq_len"`   // Prompt token count
	HeadDim       int           `json:"head_dim"`
	NumQueryHeads int           `json:"num_query_heads"` // num_attention_heads (0 = Q not available)
	Keys          [][][]float32 `json:"keys"`            // [layer][head] -> flattened len seq_len*head_dim
	Queries       [][][]float32 `json:"queries"`         // [layer][head] -> flattened len seq_len*head_dim
	Architecture  string        `json:"architecture"`
}

// if snap.HasQueries() { processQK(snap.Queries, snap.Keys) }
func (s *AttentionSnapshot) HasQueries() bool {
	return s.Queries != nil && len(s.Queries) > 0
}

//	if inspector, ok := model.(inference.AttentionInspector); ok {
//		snap, err := inspector.InspectAttention(ctx, prompt)
//	}
type AttentionInspector interface {
	InspectAttention(ctx context.Context, prompt string, options ...GenerateOption) (*AttentionSnapshot, error)
}

// m, _ := inference.LoadModel("/models/gemma3-1b")
// defer m.Close()
// for tok := range m.Generate(ctx, "Hello") { fmt.Print(tok.Text) }
type TextModel interface {
	// for tok := range m.Generate(ctx, "The quick brown fox", inference.WithMaxTokens(64)) {
	// 	fmt.Print(tok.Text)
	// }
	// if err := m.Err(); err != nil {
	// 	return err
	// }
	Generate(ctx context.Context, prompt string, options ...GenerateOption) iter.Seq[Token]

	// for tok := range m.Chat(ctx, []inference.Message{{Role: "user", Content: "Hi"}}) {
	// 	fmt.Print(tok.Text)
	// }
	Chat(ctx context.Context, messages []Message, options ...GenerateOption) iter.Seq[Token]

	// results, _ := m.Classify(ctx, []string{"positive review", "negative review"})
	// label := results[0].Token.Text
	Classify(ctx context.Context, prompts []string, options ...GenerateOption) ([]ClassifyResult, error)

	// results, _ := m.BatchGenerate(ctx, prompts, inference.WithMaxTokens(128))
	// for i, result := range results { fmt.Println(i, result.Tokens) }
	BatchGenerate(ctx context.Context, prompts []string, options ...GenerateOption) ([]BatchResult, error)

	// fmt.Println(m.ModelType()) // "gemma3", "qwen3", "llama3"
	ModelType() string

	// info := m.Info()
	// fmt.Printf("%s %d-bit quant, %d layers, vocab %d\n", info.Architecture, info.QuantBits, info.NumLayers, info.VocabSize)
	Info() ModelInfo

	// fmt.Printf("%.0f tok/s decode\n", m.Metrics().DecodeTokensPerSec)
	Metrics() GenerateMetrics

	// for tok := range m.Generate(ctx, prompt) { ... }
	// if err := m.Err(); err != nil { return err }
	Err() error

	// defer m.Close()
	Close() error
}

type Backend interface {
	// b.Name() // "metal", "rocm", "llama_cpp"
	Name() string

	// m, err := b.LoadModel("/models/gemma3-1b", inference.WithContextLen(4096))
	LoadModel(path string, options ...LoadOption) (TextModel, error)

	// if !backend.Available() { skip }
	Available() bool
}

var (
	backendsMu sync.RWMutex
	backends   = map[string]Backend{}
)

// func init() { inference.Register(metal.NewBackend()) }
func Register(backend Backend) {
	backendsMu.Lock()
	defer backendsMu.Unlock()
	backends[backend.Name()] = backend
}

// backend, ok := inference.Get("metal")
func Get(name string) (Backend, bool) {
	backendsMu.RLock()
	defer backendsMu.RUnlock()
	backend, ok := backends[name]
	return backend, ok
}

// names := inference.List() // ["llama_cpp", "metal", "rocm"]
func List() []string {
	backendsMu.RLock()
	defer backendsMu.RUnlock()
	return slices.Sorted(maps.Keys(backends))
}

//	for name, backend := range inference.All() {
//	    fmt.Println(name, backend.Available())
//	}
func All() iter.Seq2[string, Backend] {
	backendsMu.RLock()
	snap := maps.Clone(backends)
	backendsMu.RUnlock()
	return maps.All(snap)
}

// backend, err := inference.Default()
func Default() (Backend, error) {
	backendsMu.RLock()
	defer backendsMu.RUnlock()

	for _, name := range []string{"metal", "rocm", "llama_cpp"} {
		if backend, ok := backends[name]; ok && backend.Available() {
			return backend, nil
		}
	}
	for backend := range maps.Values(backends) {
		if backend.Available() {
			return backend, nil
		}
	}
	return nil, fmt.Errorf("inference: no backends registered (import a backend package)")
}

// m, err := inference.LoadModel("/models/gemma3-1b")
// m, err := inference.LoadModel("/models/qwen3-4b", inference.WithBackend("rocm"), inference.WithContextLen(8192))
func LoadModel(path string, options ...LoadOption) (TextModel, error) {
	loadConfig := ApplyLoadOpts(options)
	if loadConfig.Backend != "" {
		backend, ok := Get(loadConfig.Backend)
		if !ok {
			return nil, fmt.Errorf("inference: backend %q not registered", loadConfig.Backend)
		}
		if !backend.Available() {
			return nil, fmt.Errorf("inference: backend %q not available on this hardware", loadConfig.Backend)
		}
		return backend.LoadModel(path, options...)
	}
	backend, err := Default()
	if err != nil {
		return nil, err
	}
	return backend.LoadModel(path, options...)
}
