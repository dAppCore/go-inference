package inference

import (
	"context"
	"iter"
	"sync" // Note: test-only
	"testing"

	core "dappco.re/go"
)

// --- test helpers ---

// resetBackends clears the global registry for test isolation.
func resetBackends(t *testing.T) {
	t.Helper()
	backendsMu.Lock()
	defer backendsMu.Unlock()
	backends = map[string]Backend{}
}

// stubBackend is a minimal Backend implementation for testing.
type stubBackend struct {
	name      string
	available bool
	loadErr   error
	nilModel  bool
}

func (s *stubBackend) Name() string    { return s.name }
func (s *stubBackend) Available() bool { return s.available }
func (s *stubBackend) LoadModel(path string, opts ...LoadOption) (TextModel, error) {
	if s.loadErr != nil {
		return nil, s.loadErr
	}
	if s.nilModel {
		return nil, nil
	}
	return &stubTextModel{backend: s.name, path: path}, nil
}

// capturingBackend records the LoadOption values it received.
type capturingBackend struct {
	name         string
	available    bool
	capturedOpts []LoadOption
}

func (c *capturingBackend) Name() string    { return c.name }
func (c *capturingBackend) Available() bool { return c.available }
func (c *capturingBackend) LoadModel(path string, opts ...LoadOption) (TextModel, error) {
	c.capturedOpts = opts
	return &stubTextModel{backend: c.name, path: path}, nil
}

// stubTextModel is a minimal TextModel for testing LoadModel routing.
type stubTextModel struct {
	backend string
	path    string
}

func (m *stubTextModel) Generate(_ context.Context, _ string, _ ...GenerateOption) iter.Seq[Token] {
	return func(yield func(Token) bool) {}
}
func (m *stubTextModel) Chat(_ context.Context, _ []Message, _ ...GenerateOption) iter.Seq[Token] {
	return func(yield func(Token) bool) {}
}
func (m *stubTextModel) Classify(_ context.Context, _ []string, _ ...GenerateOption) ([]ClassifyResult, error) {
	return nil, nil
}
func (m *stubTextModel) BatchGenerate(_ context.Context, _ []string, _ ...GenerateOption) ([]BatchResult, error) {
	return nil, nil
}
func (m *stubTextModel) ModelType() string        { return "stub" }
func (m *stubTextModel) Info() ModelInfo          { return ModelInfo{} }
func (m *stubTextModel) Metrics() GenerateMetrics { return GenerateMetrics{} }
func (m *stubTextModel) Err() error               { return nil }
func (m *stubTextModel) Close() error             { return nil }

// --- Register ---

func TestInference_Register_Good(t *testing.T) {
	resetBackends(t)

	b := &stubBackend{name: "test_backend", available: true}
	Register(b)

	got, ok := Get("test_backend")
	checkTrue(t, ok)
	checkEqual(t, "test_backend", got.Name())
}

func TestInference_Register_Good_Multiple(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "alpha", available: true})
	Register(&stubBackend{name: "beta", available: true})
	Register(&stubBackend{name: "gamma", available: true})

	checkEqual(t, []string{"alpha", "beta", "gamma"}, List())
}

func TestInference_Register_Ugly_Overwrites(t *testing.T) {
	resetBackends(t)

	b1 := &stubBackend{name: "dup", available: false}
	b2 := &stubBackend{name: "dup", available: true}

	Register(b1)
	Register(b2)

	got, ok := Get("dup")
	checkTrue(t, ok)
	checkTrue(t, got.Available())
}

func TestInference_Register_Ugly_NilBackendNoop(t *testing.T) {
	resetBackends(t)

	Register(nil)

	checkEmpty(t, List())
}

// --- Get ---

func TestInference_Get_Good(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "exists", available: true})

	b, ok := Get("exists")
	checkTrue(t, ok)
	checkEqual(t, "exists", b.Name())
}

func TestInference_Get_Bad(t *testing.T) {
	resetBackends(t)

	b, ok := Get("nonexistent")
	checkFalse(t, ok)
	checkNil(t, b)
}

// --- List ---

func TestInference_List_Good_Empty(t *testing.T) {
	resetBackends(t)

	names := List()
	checkEmpty(t, names)
}

func TestInference_List_Good_Populated(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "a", available: true})
	Register(&stubBackend{name: "b", available: true})

	checkEqual(t, []string{"a", "b"}, List())
}

// --- All ---

func TestInference_All_Good(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "a", available: true})
	Register(&stubBackend{name: "b", available: true})

	found := map[string]Backend{}
	for name, b := range All() {
		found[name] = b
	}

	checkLen(t, found, 2)
	checkContains(t, found, "a")
	checkContains(t, found, "b")
}

func TestInference_All_Good_SortedOrder(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "beta", available: true})
	Register(&stubBackend{name: "alpha", available: true})

	var names []string
	for name := range All() {
		names = append(names, name)
	}

	checkEqual(t, []string{"alpha", "beta"}, names)
}

func TestInference_All_Good_Empty(t *testing.T) {
	resetBackends(t)

	for name := range All() {
		t.Errorf("expected no backends, got %q", name)
	}
}

func TestInference_All_Good_YieldFalse(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "a", available: true})
	Register(&stubBackend{name: "b", available: true})

	count := 0
	for name := range All() {
		count++
		if name == "a" || name == "b" {
			break
		}
	}
	checkEqual(t, 1, count)
}

// --- Default ---

func TestInference_Default_Good_Metal(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})
	Register(&stubBackend{name: "rocm", available: true})
	Register(&stubBackend{name: "llama_cpp", available: true})

	b, err := Default()
	checkNoError(t, err)
	checkEqual(t, "metal", b.Name())
}

func TestInference_Default_Good_Rocm(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "rocm", available: true})
	Register(&stubBackend{name: "llama_cpp", available: true})

	b, err := Default()
	checkNoError(t, err)
	checkEqual(t, "rocm", b.Name())
}

func TestInference_Default_Good_LlamaCpp(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "llama_cpp", available: true})

	b, err := Default()
	checkNoError(t, err)
	checkEqual(t, "llama_cpp", b.Name())
}

func TestInference_Default_Good_AlphabeticalFallback(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "zeta", available: true})
	Register(&stubBackend{name: "alpha", available: true})

	b, err := Default()
	checkNoError(t, err)
	checkEqual(t, "alpha", b.Name())
}

func TestInference_Default_Good_PriorityOrder(t *testing.T) {
	tests := []struct {
		name     string
		backends []stubBackend
		want     string
	}{
		{
			name: "all_available_prefers_metal",
			backends: []stubBackend{
				{name: "llama_cpp", available: true},
				{name: "rocm", available: true},
				{name: "metal", available: true},
			},
			want: "metal",
		},
		{
			name: "metal_unavailable_prefers_rocm",
			backends: []stubBackend{
				{name: "metal", available: false},
				{name: "rocm", available: true},
				{name: "llama_cpp", available: true},
			},
			want: "rocm",
		},
		{
			name: "metal_rocm_unavailable_prefers_llama_cpp",
			backends: []stubBackend{
				{name: "metal", available: false},
				{name: "rocm", available: false},
				{name: "llama_cpp", available: true},
			},
			want: "llama_cpp",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resetBackends(t)
			for i := range tt.backends {
				Register(&tt.backends[i])
			}
			b, err := Default()
			checkNoError(t, err)
			checkEqual(t, tt.want, b.Name())
		})
	}
}

func TestInference_Default_Good_FallbackToAny(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "custom_gpu", available: true})

	b, err := Default()
	checkNoError(t, err)
	checkEqual(t, "custom_gpu", b.Name())
}

func TestInference_Default_Bad_NoBackends(t *testing.T) {
	resetBackends(t)

	_, err := Default()
	checkError(t, err)
	checkContains(t, err.Error(), "no backends registered")
}

func TestInference_Default_Bad_NoneAvailable(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: false})
	Register(&stubBackend{name: "rocm", available: false})

	_, err := Default()
	checkError(t, err)
	checkContains(t, err.Error(), "no backends available")
}

func TestInference_Default_Ugly_SkipsUnavailablePreferred(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: false})
	Register(&stubBackend{name: "custom_gpu", available: true})

	b, err := Default()
	checkNoError(t, err)
	checkEqual(t, "custom_gpu", b.Name())
}

// --- LoadModel ---

func TestInference_LoadModel_Good_DefaultBackend(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})

	m, err := LoadModel("/path/to/model")
	checkNoError(t, err)
	checkNotNil(t, m)

	sm := m.(*stubTextModel)
	checkEqual(t, "metal", sm.backend)
	checkEqual(t, "/path/to/model", sm.path)
	checkNoError(t, m.Close())
}

func TestInference_LoadModel_Good_ExplicitBackend(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})
	Register(&stubBackend{name: "rocm", available: true})

	m, err := LoadModel("/path/to/model", WithBackend("rocm"))
	checkNoError(t, err)
	checkNotNil(t, m)

	sm := m.(*stubTextModel)
	checkEqual(t, "rocm", sm.backend)
	checkNoError(t, m.Close())
}

func TestInference_LoadModel_Bad_NoBackends(t *testing.T) {
	resetBackends(t)

	_, err := LoadModel("/path/to/model")
	checkError(t, err)
	checkContains(t, err.Error(), "no backends registered")
}

func TestInference_LoadModel_Bad_NoBackendsAvailable(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: false})
	Register(&stubBackend{name: "rocm", available: false})

	_, err := LoadModel("/path/to/model")
	checkError(t, err)
	checkContains(t, err.Error(), "no backends available")
}

func TestInference_LoadModel_Bad_ExplicitBackendNotRegistered(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})

	_, err := LoadModel("/path/to/model", WithBackend("rocm"))
	checkError(t, err)
	checkContains(t, err.Error(), "backend \"rocm\" not registered")
}

func TestInference_LoadModel_Bad_ExplicitBackendNotAvailable(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "rocm", available: false})

	_, err := LoadModel("/path/to/model", WithBackend("rocm"))
	checkError(t, err)
	checkContains(t, err.Error(), "backend \"rocm\" not available")
}

func TestInference_LoadModel_Bad_BackendLoadError(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{
		name:      "broken",
		available: true,
		loadErr:   core.E("test", "GPU out of memory", nil),
	})

	_, err := LoadModel("/path/to/model", WithBackend("broken"))
	checkError(t, err)
	checkContains(t, err.Error(), "GPU out of memory")
	checkEqual(t, "inference.LoadModel", core.Operation(err))
}

func TestInference_LoadModel_Good_PassesOptionsThrough(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})

	m, err := LoadModel("/models/gemma3-1b",
		WithContextLen(4096),
		WithGPULayers(24),
	)
	checkNoError(t, err)

	sm := m.(*stubTextModel)
	checkEqual(t, "/models/gemma3-1b", sm.path)
	checkNoError(t, m.Close())
}

func TestInference_LoadModel_Ugly_DefaultBackendLoadError(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{
		name:      "metal",
		available: true,
		loadErr:   core.E("test", "model not found", nil),
	})

	_, err := LoadModel("/nonexistent/model")
	checkError(t, err)
	checkContains(t, err.Error(), "model not found")
	checkEqual(t, "inference.LoadModel", core.Operation(err))
}

func TestInference_LoadModel_Bad_BackendReturnsNilModel(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{
		name:      "metal",
		available: true,
		nilModel:  true,
	})

	_, err := LoadModel("/path/to/model")
	checkError(t, err)
	checkContains(t, err.Error(), "returned a nil model")
}

// --- Type assertions (compile-time checks) ---

func TestInference_InterfaceCompliance_Good(t *testing.T) {
	var _ Backend = (*stubBackend)(nil)
	var _ TextModel = (*stubTextModel)(nil)
	backend := &stubBackend{name: "compile", available: true}
	checkEqual(t, "compile", backend.Name())
	checkTrue(t, backend.Available())
}

// --- AttentionSnapshot ---

func TestInference_AttentionSnapshot_Good(t *testing.T) {
	snap := AttentionSnapshot{
		NumLayers:    28,
		NumHeads:     16,
		SeqLen:       42,
		HeadDim:      64,
		Keys:         make([][][]float32, 28),
		Architecture: "gemma3",
	}
	checkEqual(t, 28, snap.NumLayers)
	checkEqual(t, 16, snap.NumHeads)
	checkEqual(t, 42, snap.SeqLen)
	checkEqual(t, 64, snap.HeadDim)
	checkLen(t, snap.Keys, 28)
	checkEqual(t, "gemma3", snap.Architecture)
}

func TestInference_AttentionInspectorCompliance_Good(t *testing.T) {
	var _ AttentionInspector = (*mockInspector)(nil)
	inspector := &mockInspector{}
	snap, err := inspector.InspectAttention(context.Background(), "hello")
	checkNoError(t, err)
	checkEqual(t, 28, snap.NumLayers)
}

type mockInspector struct{ stubTextModel }

func (m *mockInspector) InspectAttention(_ context.Context, _ string, _ ...GenerateOption) (*AttentionSnapshot, error) {
	return &AttentionSnapshot{NumLayers: 28, NumHeads: 8, SeqLen: 10, HeadDim: 64, Architecture: "qwen3"}, nil
}

func TestInference_AttentionInspector_Good_ReturnsSnapshot(t *testing.T) {
	var inspector AttentionInspector = &mockInspector{}
	snap, err := inspector.InspectAttention(context.Background(), "hello")
	checkNoError(t, err)
	checkEqual(t, 28, snap.NumLayers)
	checkEqual(t, 8, snap.NumHeads)
	checkEqual(t, "qwen3", snap.Architecture)
}

func TestInference_AttentionSnapshot_Good_HasQueries(t *testing.T) {
	snap := AttentionSnapshot{
		NumLayers:     28,
		NumHeads:      8,
		NumQueryHeads: 32,
		SeqLen:        42,
		HeadDim:       128,
		Keys:          make([][][]float32, 28),
		Queries:       make([][][]float32, 28),
		Architecture:  "gemma3",
	}
	checkEqual(t, 32, snap.NumQueryHeads)
	checkTrue(t, snap.HasQueries())

	kOnly := AttentionSnapshot{
		NumLayers: 28,
		NumHeads:  8,
		Keys:      make([][][]float32, 28),
	}
	checkFalse(t, kOnly.HasQueries())
}

// --- Struct types ---

func TestInference_Token_Good(t *testing.T) {
	tok := Token{ID: 42, Text: "hello"}
	checkEqual(t, int32(42), tok.ID)
	checkEqual(t, "hello", tok.Text)
}

func TestInference_Message_Good(t *testing.T) {
	msg := Message{Role: "user", Content: "Hi there"}
	checkEqual(t, "user", msg.Role)
	checkEqual(t, "Hi there", msg.Content)
}

func TestInference_ClassifyResult_Good(t *testing.T) {
	cr := ClassifyResult{
		Token:  Token{ID: 1, Text: "yes"},
		Logits: []float32{0.1, 0.9},
	}
	checkEqual(t, int32(1), cr.Token.ID)
	checkEqual(t, "yes", cr.Token.Text)
	checkLen(t, cr.Logits, 2)
}

func TestInference_BatchResult_Good(t *testing.T) {
	br := BatchResult{
		Tokens: []Token{{ID: 1, Text: "a"}, {ID: 2, Text: "b"}},
		Err:    nil,
	}
	checkLen(t, br.Tokens, 2)
	checkNoError(t, br.Err)
}

func TestInference_BatchResult_Bad(t *testing.T) {
	br := BatchResult{
		Tokens: nil,
		Err:    core.E("test", "OOM", nil),
	}
	checkNil(t, br.Tokens)
	checkError(t, br.Err)
}

func TestInference_ModelInfo_Good(t *testing.T) {
	info := ModelInfo{
		Architecture: "gemma3",
		VocabSize:    256128,
		NumLayers:    26,
		HiddenSize:   1152,
		QuantBits:    4,
		QuantGroup:   64,
	}
	checkEqual(t, "gemma3", info.Architecture)
	checkEqual(t, 256128, info.VocabSize)
	checkEqual(t, 26, info.NumLayers)
	checkEqual(t, 1152, info.HiddenSize)
	checkEqual(t, 4, info.QuantBits)
	checkEqual(t, 64, info.QuantGroup)
}

func TestInference_GenerateMetrics_Good(t *testing.T) {
	metrics := GenerateMetrics{
		PromptTokens:        100,
		GeneratedTokens:     50,
		PrefillTokensPerSec: 1000.0,
		DecodeTokensPerSec:  25.0,
		PeakMemoryBytes:     1 << 30,
		ActiveMemoryBytes:   512 << 20,
	}
	checkEqual(t, 100, metrics.PromptTokens)
	checkEqual(t, 50, metrics.GeneratedTokens)
	checkInDelta(t, 1000.0, metrics.PrefillTokensPerSec, 0.01)
	checkInDelta(t, 25.0, metrics.DecodeTokensPerSec, 0.01)
	checkEqual(t, uint64(1<<30), metrics.PeakMemoryBytes)
	checkEqual(t, uint64(512<<20), metrics.ActiveMemoryBytes)
}

// --- Concurrent registry access ---

func TestInference_Registry_Good_ConcurrentAccess(t *testing.T) {
	resetBackends(t)

	var wg sync.WaitGroup

	for i := range 20 {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			Register(&stubBackend{
				name:      core.Sprintf("backend_%d", id),
				available: true,
			})
		}(i)
	}

	for range 20 {
		wg.Go(func() {
			_ = List()
		})
	}

	for range 20 {
		wg.Go(func() {
			_, _ = Get("backend_0")
		})
	}

	for range 10 {
		wg.Go(func() {
			_, _ = Default()
		})
	}

	wg.Wait()

	names := List()
	checkLen(t, names, 20)
}

// --- Register overwrite count ---

func TestInference_Register_Ugly_OverwriteKeepsCount(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "alpha", available: true})
	Register(&stubBackend{name: "beta", available: true})
	Register(&stubBackend{name: "alpha", available: false})

	names := List()
	checkLen(t, names, 2)
}

// --- Default with all preferred unavailable and custom available ---

func TestInference_Default_Ugly_AllPreferredUnavailableCustomAvailable(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: false})
	Register(&stubBackend{name: "rocm", available: false})
	Register(&stubBackend{name: "llama_cpp", available: false})
	Register(&stubBackend{name: "custom_vulkan", available: true})

	b, err := Default()
	checkNoError(t, err)
	checkEqual(t, "custom_vulkan", b.Name())
}

// --- Default with multiple custom backends ---

func TestInference_Default_Ugly_MultipleCustomBackends(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "custom_a", available: false})
	Register(&stubBackend{name: "custom_b", available: true})

	b, err := Default()
	checkNoError(t, err)
	checkEqual(t, "custom_b", b.Name())
}

// --- LoadModel option forwarding ---

func TestInference_LoadModel_Good_ExplicitBackendForwardsOptions(t *testing.T) {
	resetBackends(t)

	cb := &capturingBackend{name: "cap", available: true}
	Register(cb)

	opts := []LoadOption{
		WithBackend("cap"),
		WithContextLen(4096),
		WithGPULayers(16),
	}
	m, err := LoadModel("/path/to/model", opts...)
	checkNoError(t, err)
	checkNotNil(t, m)

	checkLen(t, cb.capturedOpts, len(opts))

	cfg := ApplyLoadOpts(cb.capturedOpts)
	checkEqual(t, "cap", cfg.Backend)
	checkEqual(t, 4096, cfg.ContextLen)
	checkEqual(t, 16, cfg.GPULayers)
	checkNoError(t, m.Close())
}

func TestInference_LoadModel_Good_DefaultBackendForwardsOptions(t *testing.T) {
	resetBackends(t)

	cb := &capturingBackend{name: "metal", available: true}
	Register(cb)

	opts := []LoadOption{
		WithContextLen(8192),
		WithGPULayers(-1),
		WithParallelSlots(2),
	}
	m, err := LoadModel("/path/to/model", opts...)
	checkNoError(t, err)
	checkNotNil(t, m)

	checkLen(t, cb.capturedOpts, len(opts))

	cfg := ApplyLoadOpts(cb.capturedOpts)
	checkEqual(t, 8192, cfg.ContextLen)
	checkEqual(t, -1, cfg.GPULayers)
	checkEqual(t, 2, cfg.ParallelSlots)
	checkNoError(t, m.Close())
}

// --- Default preference order does not depend on registration order ---

func TestInference_Default_Good_RegistrationOrderIrrelevant(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "llama_cpp", available: true})
	Register(&stubBackend{name: "rocm", available: true})
	Register(&stubBackend{name: "metal", available: true})

	b, err := Default()
	checkNoError(t, err)
	checkEqual(t, "metal", b.Name())

	resetBackends(t)

	Register(&stubBackend{name: "rocm", available: true})
	Register(&stubBackend{name: "metal", available: true})
	Register(&stubBackend{name: "llama_cpp", available: true})

	b, err = Default()
	checkNoError(t, err)
	checkEqual(t, "metal", b.Name())
}

// --- LoadModel with empty path ---

func TestInference_LoadModel_Ugly_EmptyPath(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})

	m, err := LoadModel("")
	checkNoError(t, err)
	sm := m.(*stubTextModel)
	checkEqual(t, "", sm.path)
	checkNoError(t, m.Close())
}

// --- Get after register and overwrite ---

func TestInference_Get_Good_AfterOverwrite(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "gpu", available: false})
	Register(&stubBackend{name: "gpu", available: true})

	b, ok := Get("gpu")
	checkTrue(t, ok)
	checkTrue(t, b.Available())
}

// --- List slice independence ---

func TestInference_List_Good_IndependentSlices(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "alpha", available: true})

	firstList := List()
	secondList := List()
	checkEqual(t, firstList, secondList)

	firstList[0] = "mutated"
	thirdList := List()
	checkNotEqual(t, firstList[0], thirdList[0])
}
