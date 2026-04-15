package inference

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"sync"
	"testing"

	"dappco.re/go/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	require.True(t, ok)
	assert.Equal(t, "test_backend", got.Name())
}

func TestInference_Register_Good_Multiple(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "alpha", available: true})
	Register(&stubBackend{name: "beta", available: true})
	Register(&stubBackend{name: "gamma", available: true})

	assert.Equal(t, []string{"alpha", "beta", "gamma"}, List())
}

func TestInference_Register_Ugly_Overwrites(t *testing.T) {
	resetBackends(t)

	b1 := &stubBackend{name: "dup", available: false}
	b2 := &stubBackend{name: "dup", available: true}

	Register(b1)
	Register(b2)

	got, ok := Get("dup")
	require.True(t, ok)
	assert.True(t, got.Available(), "second registration should overwrite the first")
}

func TestInference_Register_Ugly_NilBackendNoop(t *testing.T) {
	resetBackends(t)

	Register(nil)

	assert.Empty(t, List(), "Register(nil) should be ignored")
}

// --- Get ---

func TestInference_Get_Good(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "exists", available: true})

	b, ok := Get("exists")
	require.True(t, ok)
	assert.Equal(t, "exists", b.Name())
}

func TestInference_Get_Bad(t *testing.T) {
	resetBackends(t)

	b, ok := Get("nonexistent")
	assert.False(t, ok)
	assert.Nil(t, b)
}

// --- List ---

func TestInference_List_Good_Empty(t *testing.T) {
	resetBackends(t)

	names := List()
	assert.Empty(t, names)
}

func TestInference_List_Good_Populated(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "a", available: true})
	Register(&stubBackend{name: "b", available: true})

	assert.Equal(t, []string{"a", "b"}, List())
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

	assert.Len(t, found, 2)
	assert.Contains(t, found, "a")
	assert.Contains(t, found, "b")
}

func TestInference_All_Good_SortedOrder(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "beta", available: true})
	Register(&stubBackend{name: "alpha", available: true})

	var names []string
	for name := range All() {
		names = append(names, name)
	}

	assert.Equal(t, []string{"alpha", "beta"}, names)
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
			break // Stop iteration
		}
	}
	assert.Equal(t, 1, count, "iteration should stop early")
}

// --- Default ---

func TestInference_Default_Good_Metal(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})
	Register(&stubBackend{name: "rocm", available: true})
	Register(&stubBackend{name: "llama_cpp", available: true})

	b, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "metal", b.Name(), "metal should be preferred when available")
}

func TestInference_Default_Good_Rocm(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "rocm", available: true})
	Register(&stubBackend{name: "llama_cpp", available: true})

	b, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "rocm", b.Name(), "rocm should be preferred when metal is absent")
}

func TestInference_Default_Good_LlamaCpp(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "llama_cpp", available: true})

	b, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "llama_cpp", b.Name(), "llama_cpp should be used as fallback")
}

func TestInference_Default_Good_AlphabeticalFallback(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "zeta", available: true})
	Register(&stubBackend{name: "alpha", available: true})

	b, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "alpha", b.Name(), "fallback should be deterministic across non-preferred backends")
}

func TestInference_Default_Good_PriorityOrder(t *testing.T) {
	// Full priority test: metal > rocm > llama_cpp
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
			require.NoError(t, err)
			assert.Equal(t, tt.want, b.Name())
		})
	}
}

func TestInference_Default_Good_FallbackToAny(t *testing.T) {
	resetBackends(t)

	// A custom backend that's not in the priority list.
	Register(&stubBackend{name: "custom_gpu", available: true})

	b, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "custom_gpu", b.Name(), "should fall back to any available backend")
}

func TestInference_Default_Bad_NoBackends(t *testing.T) {
	resetBackends(t)

	_, err := Default()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no backends registered")
}

func TestInference_Default_Bad_NoneAvailable(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: false})
	Register(&stubBackend{name: "rocm", available: false})

	_, err := Default()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no backends available")
}

func TestInference_Default_Ugly_SkipsUnavailablePreferred(t *testing.T) {
	resetBackends(t)

	// metal registered but not available; custom_gpu is available.
	Register(&stubBackend{name: "metal", available: false})
	Register(&stubBackend{name: "custom_gpu", available: true})

	b, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "custom_gpu", b.Name(), "should skip unavailable preferred and use any available")
}

// --- LoadModel ---

func TestInference_LoadModel_Good_DefaultBackend(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})

	m, err := LoadModel("/path/to/model")
	require.NoError(t, err)
	require.NotNil(t, m)

	sm := m.(*stubTextModel)
	assert.Equal(t, "metal", sm.backend)
	assert.Equal(t, "/path/to/model", sm.path)
	require.NoError(t, m.Close())
}

func TestInference_LoadModel_Good_ExplicitBackend(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})
	Register(&stubBackend{name: "rocm", available: true})

	m, err := LoadModel("/path/to/model", WithBackend("rocm"))
	require.NoError(t, err)
	require.NotNil(t, m)

	sm := m.(*stubTextModel)
	assert.Equal(t, "rocm", sm.backend, "should use explicitly requested backend")
	require.NoError(t, m.Close())
}

func TestInference_LoadModel_Bad_NoBackends(t *testing.T) {
	resetBackends(t)

	_, err := LoadModel("/path/to/model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no backends registered")
}

func TestInference_LoadModel_Bad_NoBackendsAvailable(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: false})
	Register(&stubBackend{name: "rocm", available: false})

	_, err := LoadModel("/path/to/model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no backends available")
}

func TestInference_LoadModel_Bad_ExplicitBackendNotRegistered(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})

	_, err := LoadModel("/path/to/model", WithBackend("rocm"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "backend \"rocm\" not registered")
}

func TestInference_LoadModel_Bad_ExplicitBackendNotAvailable(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "rocm", available: false})

	_, err := LoadModel("/path/to/model", WithBackend("rocm"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "backend \"rocm\" not available")
}

func TestInference_LoadModel_Bad_BackendLoadError(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{
		name:      "broken",
		available: true,
		loadErr:   errors.New("GPU out of memory"),
	})

	_, err := LoadModel("/path/to/model", WithBackend("broken"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "GPU out of memory")
	assert.Equal(t, "inference.LoadModel", core.Operation(err))
}

func TestInference_LoadModel_Good_PassesOptionsThrough(t *testing.T) {
	resetBackends(t)

	// Use a backend that captures the load path.
	Register(&stubBackend{name: "metal", available: true})

	m, err := LoadModel("/models/gemma3-1b",
		WithContextLen(4096),
		WithGPULayers(24),
	)
	require.NoError(t, err)

	sm := m.(*stubTextModel)
	assert.Equal(t, "/models/gemma3-1b", sm.path)
	require.NoError(t, m.Close())
}

func TestInference_LoadModel_Ugly_DefaultBackendLoadError(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{
		name:      "metal",
		available: true,
		loadErr:   errors.New("model not found"),
	})

	_, err := LoadModel("/nonexistent/model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "model not found")
	assert.Equal(t, "inference.LoadModel", core.Operation(err))
}

func TestInference_LoadModel_Bad_BackendReturnsNilModel(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{
		name:      "metal",
		available: true,
		nilModel:  true,
	})

	_, err := LoadModel("/path/to/model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "returned a nil model")
}

// --- Type assertions (compile-time checks) ---

func TestInference_InterfaceCompliance_Good(t *testing.T) {
	// Verify stubBackend implements Backend.
	var _ Backend = (*stubBackend)(nil)
	// Verify stubTextModel implements TextModel.
	var _ TextModel = (*stubTextModel)(nil)
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
	assert.Equal(t, 28, snap.NumLayers)
	assert.Equal(t, 16, snap.NumHeads)
	assert.Equal(t, 42, snap.SeqLen)
	assert.Equal(t, 64, snap.HeadDim)
	assert.Len(t, snap.Keys, 28)
	assert.Equal(t, "gemma3", snap.Architecture)
}

func TestInference_AttentionInspectorCompliance_Good(t *testing.T) {
	// Compile-time check: the interface exists and has the right signature.
	var _ AttentionInspector = (*mockInspector)(nil)
}

type mockInspector struct{ stubTextModel }

func (m *mockInspector) InspectAttention(_ context.Context, _ string, _ ...GenerateOption) (*AttentionSnapshot, error) {
	return &AttentionSnapshot{NumLayers: 28, NumHeads: 8, SeqLen: 10, HeadDim: 64, Architecture: "qwen3"}, nil
}

func TestInference_AttentionInspector_Good_ReturnsSnapshot(t *testing.T) {
	var inspector AttentionInspector = &mockInspector{}
	snap, err := inspector.InspectAttention(context.Background(), "hello")
	require.NoError(t, err)
	assert.Equal(t, 28, snap.NumLayers)
	assert.Equal(t, 8, snap.NumHeads)
	assert.Equal(t, "qwen3", snap.Architecture)
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
	assert.Equal(t, 32, snap.NumQueryHeads, "expected 32 query heads")
	assert.True(t, snap.HasQueries(), "HasQueries() should be true when Queries is non-nil")

	kOnly := AttentionSnapshot{
		NumLayers: 28,
		NumHeads:  8,
		Keys:      make([][][]float32, 28),
	}
	assert.False(t, kOnly.HasQueries(), "HasQueries() should be false when Queries is nil")
}

// --- Struct types ---

func TestInference_Token_Good(t *testing.T) {
	tok := Token{ID: 42, Text: "hello"}
	assert.Equal(t, int32(42), tok.ID)
	assert.Equal(t, "hello", tok.Text)
}

func TestInference_Message_Good(t *testing.T) {
	msg := Message{Role: "user", Content: "Hi there"}
	assert.Equal(t, "user", msg.Role)
	assert.Equal(t, "Hi there", msg.Content)
}

func TestInference_ClassifyResult_Good(t *testing.T) {
	cr := ClassifyResult{
		Token:  Token{ID: 1, Text: "yes"},
		Logits: []float32{0.1, 0.9},
	}
	assert.Equal(t, int32(1), cr.Token.ID)
	assert.Equal(t, "yes", cr.Token.Text)
	assert.Len(t, cr.Logits, 2)
}

func TestInference_BatchResult_Good(t *testing.T) {
	br := BatchResult{
		Tokens: []Token{{ID: 1, Text: "a"}, {ID: 2, Text: "b"}},
		Err:    nil,
	}
	assert.Len(t, br.Tokens, 2)
	assert.NoError(t, br.Err)
}

func TestInference_BatchResult_Bad(t *testing.T) {
	br := BatchResult{
		Tokens: nil,
		Err:    errors.New("OOM"),
	}
	assert.Nil(t, br.Tokens)
	assert.Error(t, br.Err)
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
	assert.Equal(t, "gemma3", info.Architecture)
	assert.Equal(t, 256128, info.VocabSize)
	assert.Equal(t, 26, info.NumLayers)
	assert.Equal(t, 1152, info.HiddenSize)
	assert.Equal(t, 4, info.QuantBits)
	assert.Equal(t, 64, info.QuantGroup)
}

func TestInference_GenerateMetrics_Good(t *testing.T) {
	metrics := GenerateMetrics{
		PromptTokens:        100,
		GeneratedTokens:     50,
		PrefillTokensPerSec: 1000.0,
		DecodeTokensPerSec:  25.0,
		PeakMemoryBytes:     1 << 30,   // 1 GiB
		ActiveMemoryBytes:   512 << 20, // 512 MiB
	}
	assert.Equal(t, 100, metrics.PromptTokens)
	assert.Equal(t, 50, metrics.GeneratedTokens)
	assert.InDelta(t, 1000.0, metrics.PrefillTokensPerSec, 0.01)
	assert.InDelta(t, 25.0, metrics.DecodeTokensPerSec, 0.01)
	assert.Equal(t, uint64(1<<30), metrics.PeakMemoryBytes)
	assert.Equal(t, uint64(512<<20), metrics.ActiveMemoryBytes)
}

// --- Concurrent registry access ---

func TestInference_Registry_Good_ConcurrentAccess(t *testing.T) {
	// Verify the registry is safe for concurrent reads and writes.
	// The -race flag will catch data races if the mutex is broken.
	resetBackends(t)

	var wg sync.WaitGroup

	// Concurrent writers.
	for i := range 20 {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			Register(&stubBackend{
				name:      fmt.Sprintf("backend_%d", id),
				available: true,
			})
		}(i)
	}

	// Concurrent readers interleaved with writers.
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

	// After all goroutines finish, verify all 20 backends are registered.
	names := List()
	assert.Len(t, names, 20, "all 20 backends should be registered after concurrent writes")
}

// --- Register overwrite count ---

func TestInference_Register_Ugly_OverwriteKeepsCount(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "alpha", available: true})
	Register(&stubBackend{name: "beta", available: true})
	Register(&stubBackend{name: "alpha", available: false}) // overwrite

	names := List()
	assert.Len(t, names, 2, "overwriting a backend should not increase the count")
}

// --- Default with all preferred unavailable and custom available ---

func TestInference_Default_Ugly_AllPreferredUnavailableCustomAvailable(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: false})
	Register(&stubBackend{name: "rocm", available: false})
	Register(&stubBackend{name: "llama_cpp", available: false})
	Register(&stubBackend{name: "custom_vulkan", available: true})

	b, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "custom_vulkan", b.Name(),
		"should fall back to custom backend when all preferred backends are unavailable")
}

// --- Default with multiple custom backends ---

func TestInference_Default_Ugly_MultipleCustomBackends(t *testing.T) {
	resetBackends(t)

	// Only non-preferred backends registered — one available, one not.
	Register(&stubBackend{name: "custom_a", available: false})
	Register(&stubBackend{name: "custom_b", available: true})

	b, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "custom_b", b.Name(),
		"should find the available custom backend in the fallback loop")
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
	require.NoError(t, err)
	require.NotNil(t, m)

	// The capturing backend should have received all options.
	assert.Len(t, cb.capturedOpts, len(opts),
		"all LoadOptions should be forwarded to the backend")

	// Verify the forwarded options produce the correct config.
	cfg := ApplyLoadOpts(cb.capturedOpts)
	assert.Equal(t, "cap", cfg.Backend)
	assert.Equal(t, 4096, cfg.ContextLen)
	assert.Equal(t, 16, cfg.GPULayers)
	require.NoError(t, m.Close())
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
	require.NoError(t, err)
	require.NotNil(t, m)

	// The default backend should have received all options.
	assert.Len(t, cb.capturedOpts, len(opts),
		"all LoadOptions should be forwarded to the default backend")

	cfg := ApplyLoadOpts(cb.capturedOpts)
	assert.Equal(t, 8192, cfg.ContextLen)
	assert.Equal(t, -1, cfg.GPULayers)
	assert.Equal(t, 2, cfg.ParallelSlots)
	require.NoError(t, m.Close())
}

// --- Default preference order does not depend on registration order ---

func TestInference_Default_Good_RegistrationOrderIrrelevant(t *testing.T) {
	// Register in reverse priority order — metal should still be chosen.
	resetBackends(t)

	Register(&stubBackend{name: "llama_cpp", available: true})
	Register(&stubBackend{name: "rocm", available: true})
	Register(&stubBackend{name: "metal", available: true})

	b, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "metal", b.Name(),
		"metal should win regardless of registration order")

	// Register in yet another order.
	resetBackends(t)

	Register(&stubBackend{name: "rocm", available: true})
	Register(&stubBackend{name: "metal", available: true})
	Register(&stubBackend{name: "llama_cpp", available: true})

	b, err = Default()
	require.NoError(t, err)
	assert.Equal(t, "metal", b.Name(),
		"metal should win regardless of registration order")
}

// --- LoadModel with empty path ---

func TestInference_LoadModel_Ugly_EmptyPath(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})

	// Empty path is accepted at this layer — backend decides what to do.
	m, err := LoadModel("")
	require.NoError(t, err)
	sm := m.(*stubTextModel)
	assert.Equal(t, "", sm.path, "empty path should be forwarded to the backend as-is")
	require.NoError(t, m.Close())
}

// --- Get after register and overwrite ---

func TestInference_Get_Good_AfterOverwrite(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "gpu", available: false})
	Register(&stubBackend{name: "gpu", available: true}) // overwrite

	b, ok := Get("gpu")
	require.True(t, ok)
	assert.True(t, b.Available(), "Get should return the most recently registered backend")
}

// --- List slice independence ---

func TestInference_List_Good_IndependentSlices(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "alpha", available: true})

	firstList := List()
	secondList := List()
	assert.Equal(t, firstList, secondList, "both calls should return the same names")

	// Mutating one slice should not affect the other.
	firstList[0] = "mutated"
	thirdList := List()
	assert.NotEqual(t, firstList[0], thirdList[0], "List should return independent slices")
}
