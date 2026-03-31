package inference

import (
	"context"
	"fmt"
	"iter"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func resetBackends(t *testing.T) {
	t.Helper()
	backendsMu.Lock()
	defer backendsMu.Unlock()
	backends = map[string]Backend{}
}

type stubBackend struct {
	name      string
	available bool
	loadErr   error
}

func (s *stubBackend) Name() string    { return s.name }
func (s *stubBackend) Available() bool { return s.available }
func (s *stubBackend) LoadModel(path string, opts ...LoadOption) (TextModel, error) {
	if s.loadErr != nil {
		return nil, s.loadErr
	}
	return &stubTextModel{backend: s.name, path: path}, nil
}

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

func TestRegister_Good(t *testing.T) {
	resetBackends(t)

	backend := &stubBackend{name: "test_backend", available: true}
	Register(backend)

	registeredBackend, ok := Get("test_backend")
	require.True(t, ok)
	assert.Equal(t, "test_backend", registeredBackend.Name())
}

func TestRegister_Good_Multiple(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "alpha", available: true})
	Register(&stubBackend{name: "beta", available: true})
	Register(&stubBackend{name: "gamma", available: true})

	assert.Equal(t, []string{"alpha", "beta", "gamma"}, List())
}

func TestRegister_Ugly_Overwrites(t *testing.T) {
	resetBackends(t)

	firstBackend := &stubBackend{name: "dup", available: false}
	replacementBackend := &stubBackend{name: "dup", available: true}

	Register(firstBackend)
	Register(replacementBackend)

	got, ok := Get("dup")
	require.True(t, ok)
	assert.True(t, got.Available())
}

func TestGet_Good(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "exists", available: true})

	backend, ok := Get("exists")
	require.True(t, ok)
	assert.Equal(t, "exists", backend.Name())
}

func TestGet_Bad(t *testing.T) {
	resetBackends(t)

	backend, ok := Get("nonexistent")
	assert.False(t, ok)
	assert.Nil(t, backend)
}

func TestList_Good_Empty(t *testing.T) {
	resetBackends(t)

	names := List()
	assert.Empty(t, names)
}

func TestList_Good_Populated(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "a", available: true})
	Register(&stubBackend{name: "b", available: true})

	assert.Equal(t, []string{"a", "b"}, List())
}

func TestAll_Good(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "a", available: true})
	Register(&stubBackend{name: "b", available: true})

	found := map[string]Backend{}
	for name, backend := range All() {
		found[name] = backend
	}

	assert.Len(t, found, 2)
	assert.Contains(t, found, "a")
	assert.Contains(t, found, "b")
}

func TestAll_Good_Empty(t *testing.T) {
	resetBackends(t)

	count := 0
	for range All() {
		count++
	}
	assert.Zero(t, count)
}

func TestAll_Good_YieldFalse(t *testing.T) {
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
	assert.Equal(t, 1, count)
}

func TestDefault_Good_Metal(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})
	Register(&stubBackend{name: "rocm", available: true})
	Register(&stubBackend{name: "llama_cpp", available: true})

	backend, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "metal", backend.Name())
}

func TestDefault_Good_Rocm(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "rocm", available: true})
	Register(&stubBackend{name: "llama_cpp", available: true})

	backend, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "rocm", backend.Name())
}

func TestDefault_Good_LlamaCpp(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "llama_cpp", available: true})

	backend, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "llama_cpp", backend.Name())
}

func TestDefault_Good_PriorityOrder(t *testing.T) {
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
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			resetBackends(t)
			for backendIndex := range testCase.backends {
				Register(&testCase.backends[backendIndex])
			}
			backend, err := Default()
			require.NoError(t, err)
			assert.Equal(t, testCase.want, backend.Name())
		})
	}
}

func TestDefault_Good_FallbackToAny(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "custom_gpu", available: true})

	backend, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "custom_gpu", backend.Name())
}

func TestDefault_Bad_NoBackends(t *testing.T) {
	resetBackends(t)

	_, err := Default()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no backends registered")
}

func TestDefault_Bad_NoneAvailable(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: false})
	Register(&stubBackend{name: "rocm", available: false})

	_, err := Default()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no backends registered")
}

func TestDefault_Ugly_SkipsUnavailablePreferred(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: false})
	Register(&stubBackend{name: "custom_gpu", available: true})

	backend, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "custom_gpu", backend.Name())
}

func TestLoadModel_Good_DefaultBackend(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})

	model, err := LoadModel("/path/to/model")
	require.NoError(t, err)
	require.NotNil(t, model)

	stubModel := model.(*stubTextModel)
	assert.Equal(t, "metal", stubModel.backend)
	assert.Equal(t, "/path/to/model", stubModel.path)
	require.NoError(t, model.Close())
}

func TestLoadModel_Good_ExplicitBackend(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})
	Register(&stubBackend{name: "rocm", available: true})

	model, err := LoadModel("/path/to/model", WithBackend("rocm"))
	require.NoError(t, err)
	require.NotNil(t, model)

	stubModel := model.(*stubTextModel)
	assert.Equal(t, "rocm", stubModel.backend)
	require.NoError(t, model.Close())
}

func TestLoadModel_Bad_NoBackends(t *testing.T) {
	resetBackends(t)

	_, err := LoadModel("/path/to/model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no backends registered")
}

func TestLoadModel_Bad_ExplicitBackendNotRegistered(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})

	_, err := LoadModel("/path/to/model", WithBackend("rocm"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "backend \"rocm\" not registered")
}

func TestLoadModel_Bad_ExplicitBackendNotAvailable(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "rocm", available: false})

	_, err := LoadModel("/path/to/model", WithBackend("rocm"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "backend \"rocm\" not available")
}

func TestLoadModel_Bad_BackendLoadError(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{
		name:      "broken",
		available: true,
		loadErr:   fmt.Errorf("GPU out of memory"),
	})

	_, err := LoadModel("/path/to/model", WithBackend("broken"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "GPU out of memory")
}

func TestLoadModel_Good_PassesOptionsThrough(t *testing.T) {
	resetBackends(t)

	captureBackend := &capturingBackend{name: "metal", available: true}
	Register(captureBackend)

	loadOptions := []LoadOption{
		WithContextLen(4096),
		WithGPULayers(24),
	}
	model, err := LoadModel("/models/gemma3-1b", loadOptions...)
	require.NoError(t, err)
	require.NotNil(t, model)

	require.Len(t, captureBackend.capturedOpts, len(loadOptions))
	loadConfig := ApplyLoadOpts(captureBackend.capturedOpts)
	assert.Equal(t, 4096, loadConfig.ContextLen)
	assert.Equal(t, 24, loadConfig.GPULayers)

	stubModel := model.(*stubTextModel)
	assert.Equal(t, "/models/gemma3-1b", stubModel.path)
	require.NoError(t, model.Close())
}

func TestLoadModel_Ugly_DefaultBackendLoadError(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{
		name:      "metal",
		available: true,
		loadErr:   fmt.Errorf("model not found"),
	})

	_, err := LoadModel("/nonexistent/model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "model not found")
}

func TestTypes_Good_InterfaceCompliance(t *testing.T) {
	var _ Backend = (*stubBackend)(nil)
	var _ TextModel = (*stubTextModel)(nil)
}

func TestAttentionSnapshot_Good(t *testing.T) {
	snapshot := AttentionSnapshot{
		NumLayers:    28,
		NumHeads:     16,
		SeqLen:       42,
		HeadDim:      64,
		Keys:         make([][][]float32, 28),
		Architecture: "gemma3",
	}
	assert.Equal(t, 28, snapshot.NumLayers)
	assert.Equal(t, 16, snapshot.NumHeads)
	assert.Equal(t, 42, snapshot.SeqLen)
	assert.Equal(t, 64, snapshot.HeadDim)
	assert.Len(t, snapshot.Keys, 28)
	assert.Equal(t, "gemma3", snapshot.Architecture)
}

func TestAttentionInspector_Good_InterfaceCompliance(t *testing.T) {
	var _ AttentionInspector = (*mockInspector)(nil)
}

type mockInspector struct{ stubTextModel }

func (m *mockInspector) InspectAttention(_ context.Context, _ string, _ ...GenerateOption) (*AttentionSnapshot, error) {
	return &AttentionSnapshot{NumLayers: 28, NumHeads: 8, SeqLen: 10, HeadDim: 64, Architecture: "qwen3"}, nil
}

func TestAttentionInspector_Good_ReturnsSnapshot(t *testing.T) {
	var inspector AttentionInspector = &mockInspector{}
	snapshot, err := inspector.InspectAttention(context.Background(), "hello")
	require.NoError(t, err)
	assert.Equal(t, 28, snapshot.NumLayers)
	assert.Equal(t, 8, snapshot.NumHeads)
	assert.Equal(t, "qwen3", snapshot.Architecture)
}

func TestAttentionSnapshotWithQueries(t *testing.T) {
	snapshot := AttentionSnapshot{
		NumLayers:     28,
		NumHeads:      8,
		NumQueryHeads: 32,
		SeqLen:        42,
		HeadDim:       128,
		Keys:          make([][][]float32, 28),
		Queries:       make([][][]float32, 28),
		Architecture:  "gemma3",
	}
	require.Equal(t, 32, snapshot.NumQueryHeads)
	require.True(t, snapshot.HasQueries())

	keyOnlySnapshot := AttentionSnapshot{
		NumLayers: 28,
		NumHeads:  8,
		Keys:      make([][][]float32, 28),
	}
	assert.False(t, keyOnlySnapshot.HasQueries())
}

func TestToken_Good(t *testing.T) {
	token := Token{ID: 42, Text: "hello"}
	assert.Equal(t, int32(42), token.ID)
	assert.Equal(t, "hello", token.Text)
}

func TestMessage_Good(t *testing.T) {
	message := Message{Role: "user", Content: "Hi there"}
	assert.Equal(t, "user", message.Role)
	assert.Equal(t, "Hi there", message.Content)
}

func TestClassifyResult_Good(t *testing.T) {
	classifyResult := ClassifyResult{
		Token:  Token{ID: 1, Text: "yes"},
		Logits: []float32{0.1, 0.9},
	}
	assert.Equal(t, int32(1), classifyResult.Token.ID)
	assert.Equal(t, "yes", classifyResult.Token.Text)
	assert.Len(t, classifyResult.Logits, 2)
}

func TestBatchResult_Good(t *testing.T) {
	batchResult := BatchResult{
		Tokens: []Token{{ID: 1, Text: "a"}, {ID: 2, Text: "b"}},
		Err:    nil,
	}
	assert.Len(t, batchResult.Tokens, 2)
	assert.NoError(t, batchResult.Err)
}

func TestBatchResult_Bad(t *testing.T) {
	batchResult := BatchResult{
		Tokens: nil,
		Err:    fmt.Errorf("OOM"),
	}
	assert.Nil(t, batchResult.Tokens)
	assert.Error(t, batchResult.Err)
}

func TestModelInfo_Good(t *testing.T) {
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

func TestGenerateMetrics_Good(t *testing.T) {
	metrics := GenerateMetrics{
		PromptTokens:        100,
		GeneratedTokens:     50,
		PrefillTokensPerSec: 1000.0,
		DecodeTokensPerSec:  25.0,
		PeakMemoryBytes:     1 << 30,
		ActiveMemoryBytes:   512 << 20,
	}
	assert.Equal(t, 100, metrics.PromptTokens)
	assert.Equal(t, 50, metrics.GeneratedTokens)
	assert.InDelta(t, 1000.0, metrics.PrefillTokensPerSec, 0.01)
	assert.InDelta(t, 25.0, metrics.DecodeTokensPerSec, 0.01)
	assert.Equal(t, uint64(1<<30), metrics.PeakMemoryBytes)
	assert.Equal(t, uint64(512<<20), metrics.ActiveMemoryBytes)
}

func TestRegistry_Good_ConcurrentAccess(t *testing.T) {
	resetBackends(t)

	var waitGroup sync.WaitGroup

	for index := range 20 {
		waitGroup.Add(1)
		go func(id int) {
			defer waitGroup.Done()
			Register(&stubBackend{
				name:      fmt.Sprintf("backend_%d", id),
				available: true,
			})
		}(index)
	}

	for range 20 {
		waitGroup.Go(func() {
			_ = List()
		})
	}

	for range 20 {
		waitGroup.Go(func() {
			_, _ = Get("backend_0")
		})
	}

	for range 10 {
		waitGroup.Go(func() {
			_, _ = Default()
		})
	}

	waitGroup.Wait()

	names := List()
	assert.Len(t, names, 20)
}

func TestRegister_Ugly_OverwriteKeepsCount(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "alpha", available: true})
	Register(&stubBackend{name: "beta", available: true})
	Register(&stubBackend{name: "alpha", available: false})

	names := List()
	assert.Len(t, names, 2)
}

func TestDefault_Ugly_AllPreferredUnavailableCustomAvailable(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: false})
	Register(&stubBackend{name: "rocm", available: false})
	Register(&stubBackend{name: "llama_cpp", available: false})
	Register(&stubBackend{name: "custom_vulkan", available: true})

	backend, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "custom_vulkan", backend.Name())
}

func TestDefault_Ugly_MultipleCustomBackends(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "custom_a", available: false})
	Register(&stubBackend{name: "custom_b", available: true})

	backend, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "custom_b", backend.Name())
}

func TestLoadModel_Good_ExplicitBackendForwardsOptions(t *testing.T) {
	resetBackends(t)

	captureBackend := &capturingBackend{name: "cap", available: true}
	Register(captureBackend)

	loadOptions := []LoadOption{
		WithBackend("cap"),
		WithContextLen(4096),
		WithGPULayers(16),
	}
	model, err := LoadModel("/path/to/model", loadOptions...)
	require.NoError(t, err)
	require.NotNil(t, model)

	assert.Len(t, captureBackend.capturedOpts, len(loadOptions))

	loadConfig := ApplyLoadOpts(captureBackend.capturedOpts)
	assert.Equal(t, "cap", loadConfig.Backend)
	assert.Equal(t, 4096, loadConfig.ContextLen)
	assert.Equal(t, 16, loadConfig.GPULayers)
	require.NoError(t, model.Close())
}

func TestLoadModel_Good_DefaultBackendForwardsOptions(t *testing.T) {
	resetBackends(t)

	captureBackend := &capturingBackend{name: "metal", available: true}
	Register(captureBackend)

	loadOptions := []LoadOption{
		WithContextLen(8192),
		WithGPULayers(-1),
		WithParallelSlots(2),
	}
	model, err := LoadModel("/path/to/model", loadOptions...)
	require.NoError(t, err)
	require.NotNil(t, model)

	assert.Len(t, captureBackend.capturedOpts, len(loadOptions))

	loadConfig := ApplyLoadOpts(captureBackend.capturedOpts)
	assert.Equal(t, 8192, loadConfig.ContextLen)
	assert.Equal(t, -1, loadConfig.GPULayers)
	assert.Equal(t, 2, loadConfig.ParallelSlots)
	require.NoError(t, model.Close())
}

func TestDefault_Good_RegistrationOrderIrrelevant(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "llama_cpp", available: true})
	Register(&stubBackend{name: "rocm", available: true})
	Register(&stubBackend{name: "metal", available: true})

	backend, err := Default()
	require.NoError(t, err)
	assert.Equal(t, "metal", backend.Name())

	resetBackends(t)

	Register(&stubBackend{name: "rocm", available: true})
	Register(&stubBackend{name: "metal", available: true})
	Register(&stubBackend{name: "llama_cpp", available: true})

	backend, err = Default()
	require.NoError(t, err)
	assert.Equal(t, "metal", backend.Name())
}

func TestLoadModel_Ugly_EmptyPath(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})

	model, err := LoadModel("")
	require.NoError(t, err)
	stubModel := model.(*stubTextModel)
	assert.Equal(t, "", stubModel.path)
	require.NoError(t, model.Close())
}

func TestGet_Good_AfterOverwrite(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "gpu", available: false})
	Register(&stubBackend{name: "gpu", available: true})

	backend, ok := Get("gpu")
	require.True(t, ok)
	assert.True(t, backend.Available())
}

func TestList_Good_IndependentSlices(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "alpha", available: true})

	firstNames := List()
	secondNames := List()
	assert.Equal(t, firstNames, secondNames)

	firstNames[0] = "mutated"
	thirdNames := List()
	assert.NotEqual(t, firstNames[0], thirdNames[0])
}
