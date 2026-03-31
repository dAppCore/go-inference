// Package inference provides the shared contract for text-generation backends.
//
// model, err := inference.LoadModel("/models/gemma3-1b")
// if err != nil {
// 	log.Fatal(err)
// }
// defer model.Close()
//
// ctx := context.Background()
// for token := range model.Generate(ctx, "Hello", inference.WithMaxTokens(128)) {
// 	fmt.Print(token.Text)
// }
// if err := model.Err(); err != nil {
// 	log.Fatal(err)
// }
//
// classificationResults, _ := model.Classify(ctx, []string{"positive", "negative"}, inference.WithTemperature(0))
// batchResults, _ := model.BatchGenerate(ctx, []string{"First", "Second"}, inference.WithMaxTokens(32))
//
// if inspector, ok := model.(inference.AttentionInspector); ok {
// 	snapshot, err := inspector.InspectAttention(ctx, "Hello")
// 	_ = snapshot.HasQueries()
// 	_ = err
// }
//
// for discoveredModel := range inference.Discover("/path/to/models") {
// 	fmt.Printf("%s (%s)\n", discoveredModel.Path, discoveredModel.ModelType)
// }
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
	Role    string `json:"role"`
	Content string `json:"content"`
}

// classificationResults, _ := model.Classify(ctx, []string{"positive", "negative"})
// label := classificationResults[0].Token.Text
// logits := classificationResults[0].Logits
type ClassifyResult struct {
	Token  Token
	Logits []float32
}

// batchResults, _ := model.BatchGenerate(ctx, []string{"First", "Second"}, inference.WithMaxTokens(64))
//
//	for index, result := range batchResults {
//		if result.Err != nil {
//			continue
//		}
//		for _, token := range result.Tokens {
//			fmt.Print(token.Text)
//		}
//	}
type BatchResult struct {
	Tokens []Token
	Err    error
}

// metrics := model.Metrics()
// fmt.Printf("prefill: %.0f tokens/s  decode: %.0f tokens/s\n", metrics.PrefillTokensPerSec, metrics.DecodeTokensPerSec)
// fmt.Printf("peak GPU memory: %d MiB\n", metrics.PeakMemoryBytes>>20)
type GenerateMetrics struct {
	PromptTokens    int
	GeneratedTokens int

	PrefillDuration time.Duration
	DecodeDuration  time.Duration
	TotalDuration   time.Duration

	PrefillTokensPerSec float64
	DecodeTokensPerSec  float64

	PeakMemoryBytes   uint64
	ActiveMemoryBytes uint64
}

// modelInfo := model.Info()
// fmt.Printf("%s %d-bit quant, %d layers, vocab %d\n", modelInfo.Architecture, modelInfo.QuantBits, modelInfo.NumLayers, modelInfo.VocabSize)
type ModelInfo struct {
	Architecture string
	VocabSize    int
	NumLayers    int
	HiddenSize   int
	QuantBits    int
	QuantGroup   int
}

// snapshot, _ := inspector.InspectAttention(ctx, prompt)
//
//	if snapshot.HasQueries() {
//		processQK(snapshot.Queries, snapshot.Keys)
//	}
type AttentionSnapshot struct {
	NumLayers     int           `json:"num_layers"`
	NumHeads      int           `json:"num_heads"`
	SeqLen        int           `json:"seq_len"`
	HeadDim       int           `json:"head_dim"`
	NumQueryHeads int           `json:"num_query_heads"`
	Keys          [][][]float32 `json:"keys"`
	Queries       [][][]float32 `json:"queries"`
	Architecture  string        `json:"architecture"`
}

// if snapshot.HasQueries() { processQK(snapshot.Queries, snapshot.Keys) }
func (snapshot *AttentionSnapshot) HasQueries() bool {
	return snapshot.Queries != nil && len(snapshot.Queries) > 0
}

//	if inspector, ok := model.(inference.AttentionInspector); ok {
//		snapshot, err := inspector.InspectAttention(ctx, prompt)
//	}
type AttentionInspector interface {
	InspectAttention(ctx context.Context, prompt string, generateOptions ...GenerateOption) (*AttentionSnapshot, error)
}

// model, _ := inference.LoadModel("/models/gemma3-1b")
// defer model.Close()
// for token := range model.Generate(ctx, "Hello") { fmt.Print(token.Text) }
type TextModel interface {
	// for token := range model.Generate(ctx, "The quick brown fox", inference.WithMaxTokens(64)) {
	// 	fmt.Print(token.Text)
	// }
	// if err := model.Err(); err != nil {
	// 	return err
	// }
	Generate(ctx context.Context, prompt string, generateOptions ...GenerateOption) iter.Seq[Token]

	// for token := range model.Chat(ctx, []inference.Message{{Role: "user", Content: "Hi"}}) {
	// 	fmt.Print(token.Text)
	// }
	Chat(ctx context.Context, messages []Message, generateOptions ...GenerateOption) iter.Seq[Token]

	// classificationResults, _ := model.Classify(ctx, []string{"positive review", "negative review"})
	// label := classificationResults[0].Token.Text
	Classify(ctx context.Context, prompts []string, generateOptions ...GenerateOption) ([]ClassifyResult, error)

	// batchResults, _ := model.BatchGenerate(ctx, prompts, inference.WithMaxTokens(128))
	// for index, result := range batchResults { fmt.Println(index, result.Tokens) }
	BatchGenerate(ctx context.Context, prompts []string, generateOptions ...GenerateOption) ([]BatchResult, error)

	// fmt.Println(model.ModelType()) // "gemma3", "qwen3", "llama3"
	ModelType() string

	// modelInfo := model.Info()
	// fmt.Printf("%s %d-bit quant, %d layers, vocab %d\n", modelInfo.Architecture, modelInfo.QuantBits, modelInfo.NumLayers, modelInfo.VocabSize)
	Info() ModelInfo

	// fmt.Printf("%.0f tokens/s decode\n", model.Metrics().DecodeTokensPerSec)
	Metrics() GenerateMetrics

	// for token := range model.Generate(ctx, prompt) { ... }
	// if err := model.Err(); err != nil { return err }
	Err() error

	// defer model.Close()
	Close() error
}

type Backend interface {
	// backend.Name() // "metal", "rocm", "llama_cpp"
	Name() string

	// model, err := backend.LoadModel("/models/gemma3-1b", inference.WithContextLen(4096))
	LoadModel(path string, loadOptions ...LoadOption) (TextModel, error)

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
	backendSnapshot := maps.Clone(backends)
	backendsMu.RUnlock()
	return maps.All(backendSnapshot)
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

// model, err := inference.LoadModel("/models/gemma3-1b")
// model, err := inference.LoadModel("/models/qwen3-4b", inference.WithBackend("rocm"), inference.WithContextLen(8192))
func LoadModel(path string, loadOptions ...LoadOption) (TextModel, error) {
	loadConfig := ApplyLoadOpts(loadOptions)
	if loadConfig.Backend != "" {
		backend, ok := Get(loadConfig.Backend)
		if !ok {
			return nil, fmt.Errorf("inference: backend %q not registered", loadConfig.Backend)
		}
		if !backend.Available() {
			return nil, fmt.Errorf("inference: backend %q not available on this hardware", loadConfig.Backend)
		}
		return backend.LoadModel(path, loadOptions...)
	}
	backend, err := Default()
	if err != nil {
		return nil, err
	}
	return backend.LoadModel(path, loadOptions...)
}
