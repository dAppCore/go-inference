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
//	r := inference.LoadModel("/path/to/model/")
//	if !r.OK { log.Fatal(r.Error()) }
//	m := r.Value.(inference.TextModel)
//	defer m.Close()
//
//	ctx := context.Background()
//	for tok := range m.Generate(ctx, "prompt", inference.WithMaxTokens(128)) {
//	    fmt.Print(tok.Text)
//	}
//	if r := m.Err(); !r.OK { log.Fatal(r.Error()) }
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
//	cr := m.Classify(ctx, prompts, inference.WithTemperature(0))
//	results := cr.Value.([]inference.ClassifyResult)
//
//	// Batch generate — parallel autoregressive decoding
//	br := m.BatchGenerate(ctx, prompts, inference.WithMaxTokens(32))
//	batched := br.Value.([]inference.BatchResult)
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
	"slices"
	"time"

	core "dappco.re/go"
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
	// Images carries encoded image bytes (PNG/JPEG) attached to this turn,
	// populated by the compat handlers from multimodal content parts. Only
	// engines implementing VisionModel serve image turns; the handlers
	// reject image requests against text-only models.
	Images [][]byte `json:"images,omitempty"`
}

// VisionModel is the optional capability a TextModel implements when the
// LOADED CHECKPOINT accepts image content — the family supporting vision
// does not mean the snapshot shipped the tower, so this is a live probe,
// not a static declaration.
type VisionModel interface {
	AcceptsImages() bool
}

// cr := m.Classify(ctx, []string{"positive", "negative"})
// results := cr.Value.([]inference.ClassifyResult)
// label := results[0].Token.Text  // sampled token at last position
// logits := results[0].Logits     // only populated when WithLogits() is set
type ClassifyResult struct {
	Token  Token     // Sampled/greedy token at last prompt position
	Logits []float32 // Raw vocab-sized logits (only when WithLogits is set)
}

// br := m.BatchGenerate(ctx, prompts, inference.WithMaxTokens(64))
// batched := br.Value.([]inference.BatchResult)
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
	//	if r := m.Err(); !r.OK { return r }
	Generate(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq[Token]

	// Chat streams tokens from a multi-turn conversation using the model's native template.
	//
	//	for tok := range m.Chat(ctx, []inference.Message{{Role: "user", Content: "Hi"}}) {
	//	    fmt.Print(tok.Text)
	//	}
	Chat(ctx context.Context, messages []Message, opts ...GenerateOption) iter.Seq[Token]

	// Classify runs batched prefill-only inference — fast path for classification tasks.
	// Each prompt gets one forward pass; the token at the last position is sampled.
	// The Result carries []ClassifyResult in Value when OK.
	//
	//	cr := m.Classify(ctx, []string{"positive review", "negative review"})
	//	if !cr.OK { return cr }
	//	results := cr.Value.([]inference.ClassifyResult)
	//	label := results[0].Token.Text
	Classify(ctx context.Context, prompts []string, opts ...GenerateOption) core.Result

	// BatchGenerate runs batched autoregressive generation up to MaxTokens per prompt.
	// The Result carries []BatchResult in Value when OK.
	//
	//	br := m.BatchGenerate(ctx, prompts, inference.WithMaxTokens(128))
	//	if !br.OK { return br }
	//	results := br.Value.([]inference.BatchResult)
	//	for i, r := range results { fmt.Println(i, r.Tokens) }
	BatchGenerate(ctx context.Context, prompts []string, opts ...GenerateOption) core.Result

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

	// Err reports any error from the last Generate or Chat call.
	// Check after the iterator stops to distinguish normal EOS from errors.
	// The Result is OK with a nil Value on success, or a failure carrying
	// the error otherwise.
	//
	//	for tok := range m.Generate(ctx, prompt) { ... }
	//	if r := m.Err(); !r.OK { return r }
	Err() core.Result

	// Close releases GPU memory, KV caches, and any subprocess. The Result
	// is OK with a nil Value on success, or a failure carrying the error.
	//
	//	defer m.Close()
	Close() core.Result
}

// func init() { inference.Register(metal.NewBackend()) } // called from backend packages
type Backend interface {
	// Name is the stable identifier used for registration and selection.
	//
	//	b.Name() // "metal", "rocm", "llama_cpp"
	Name() string

	// LoadModel reads the model directory at path and returns a ready
	// TextModel in the Result's Value when OK.
	//
	//	r := b.LoadModel("/models/gemma3-1b", inference.WithContextLen(4096))
	//	if !r.OK { return r }
	//	m := r.Value.(inference.TextModel)
	LoadModel(path string, opts ...LoadOption) core.Result

	// Available reports whether the required hardware or driver is present at runtime.
	//
	//	if !b.Available() { skip } // e.g. Metal on non-Apple hardware returns false
	Available() bool
}

var (
	backendsMu            = core.New().Lock("inference.backends").Mutex
	backends              = map[string]Backend{}
	preferredBackendOrder = []string{"metal", "rocm", "llama_cpp"}
	preferredBackendSet   = map[string]struct{}{
		"metal":     {},
		"rocm":      {},
		"llama_cpp": {},
	}
)

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
//
// Single-pass key copy under RLock — earlier shape did maps.Clone +
// maps.Keys + slices.Sorted (~4 allocs + bucket cost). Direct slice
// build is 1 alloc; empty registry returns nil (preserves the test
// contract that callers can branch on).
func List() []string {
	backendsMu.RLock()
	if len(backends) == 0 {
		backendsMu.RUnlock()
		return nil
	}
	names := make([]string, 0, len(backends))
	for name := range backends {
		names = append(names, name)
	}
	backendsMu.RUnlock()
	slices.Sort(names)
	return names
}

//	for name, b := range inference.All() {
//	    fmt.Println(name, b.Available())
//	}
//
// Builds a slice of (name, backend) pairs under RLock so the returned
// iterator runs without holding any lock — single alloc for the pair
// slice instead of the previous maps.Clone + maps.Keys + slices.Sorted
// cascade.
func All() iter.Seq2[string, Backend] {
	type entry struct {
		name string
		back Backend
	}
	backendsMu.RLock()
	entries := make([]entry, 0, len(backends))
	for name, b := range backends {
		entries = append(entries, entry{name, b})
	}
	backendsMu.RUnlock()
	slices.SortFunc(entries, func(a, b entry) int {
		if a.name < b.name {
			return -1
		}
		if a.name > b.name {
			return 1
		}
		return 0
	})
	return func(yield func(string, Backend) bool) {
		for _, e := range entries {
			if !yield(e.name, e.back) {
				return
			}
		}
	}
}

// Default picks the first available backend in preference order: metal → rocm → llama_cpp → any.
//
//	r := inference.Default() // r.Value is the backend when r.OK
//
// Both preferred-order scan and fallback run against direct map
// lookups under RLock — no clone, no Keys-iterator allocation. The
// happy path (preferred backend available) is 0 allocs.
func Default() core.Result {
	backendsMu.RLock()
	if len(backends) == 0 {
		backendsMu.RUnlock()
		return core.Fail(core.E("inference.Default", "no backends registered", nil))
	}

	// Platform preference order — direct map lookups, no clone.
	for _, name := range preferredBackendOrder {
		if b, ok := backends[name]; ok && b.Available() {
			backendsMu.RUnlock()
			return core.Ok(b)
		}
	}

	// Fall back to any non-preferred backend, in sorted-name order.
	// Snapshot (name, backend) pairs under RLock so Available() runs
	// outside the lock — matches the prior defensive behaviour.
	type entry struct {
		name string
		back Backend
	}
	var fallback []entry
	for name, b := range backends {
		if _, isPreferred := preferredBackendSet[name]; isPreferred {
			continue
		}
		fallback = append(fallback, entry{name, b})
	}
	backendsMu.RUnlock()

	slices.SortFunc(fallback, func(a, b entry) int {
		if a.name < b.name {
			return -1
		}
		if a.name > b.name {
			return 1
		}
		return 0
	})
	for _, e := range fallback {
		if e.back.Available() {
			return core.Ok(e.back)
		}
	}
	return core.Fail(core.E("inference.Default", "no backends available", nil))
}

// r := inference.LoadModel("/models/gemma3-1b")
// r := inference.LoadModel("/models/qwen3-4b", inference.WithBackend("rocm"), inference.WithContextLen(8192))
func LoadModel(path string, opts ...LoadOption) core.Result {
	cfg := ApplyLoadOpts(opts)
	if cfg.Backend != "" {
		b, ok := Get(cfg.Backend)
		if !ok {
			return core.Fail(core.E("inference.LoadModel", core.Sprintf("backend %q not registered", cfg.Backend), nil))
		}
		if !b.Available() {
			return core.Fail(core.E("inference.LoadModel", core.Sprintf("backend %q not available on this hardware", cfg.Backend), nil))
		}
		modelResult := b.LoadModel(path, opts...)
		if !modelResult.OK {
			return core.Fail(core.Wrap(modelResult.Value.(error), "inference.LoadModel", core.Sprintf("backend %q failed to load model", cfg.Backend)))
		}
		model := modelResult.Value
		if model == nil {
			return core.Fail(core.E("inference.LoadModel", core.Sprintf("backend %q returned a nil model", cfg.Backend), nil))
		}
		return core.Ok(model)
	}
	defaultResult := Default()
	if !defaultResult.OK {
		return defaultResult
	}
	b, ok := defaultResult.Value.(Backend)
	if !ok || b == nil {
		return core.Fail(core.E("inference.LoadModel", "default backend result was not a backend", nil))
	}
	modelResult := b.LoadModel(path, opts...)
	if !modelResult.OK {
		return core.Fail(core.Wrap(modelResult.Value.(error), "inference.LoadModel", core.Sprintf("backend %q failed to load model", b.Name())))
	}
	model := modelResult.Value
	if model == nil {
		return core.Fail(core.E("inference.LoadModel", core.Sprintf("backend %q returned a nil model", b.Name()), nil))
	}
	return core.Ok(model)
}
