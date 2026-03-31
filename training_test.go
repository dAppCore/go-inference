package inference

import (
	"testing"

	"dappco.re/go/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- DefaultLoRAConfig ---

func TestTraining_DefaultLoRAConfig_Good(t *testing.T) {
	cfg := DefaultLoRAConfig()
	assert.Equal(t, 8, cfg.Rank, "default Rank should be 8")
	assert.InDelta(t, float32(16), cfg.Alpha, 0.0001, "default Alpha should be 16")
	assert.Equal(t, []string{"q_proj", "v_proj"}, cfg.TargetKeys, "default TargetKeys should be q_proj and v_proj")
	assert.False(t, cfg.BFloat16, "default BFloat16 should be false")
}

func TestTraining_DefaultLoRAConfig_Good_Idempotent(t *testing.T) {
	firstConfig := DefaultLoRAConfig()
	secondConfig := DefaultLoRAConfig()
	assert.Equal(t, firstConfig, secondConfig, "DefaultLoRAConfig should be idempotent")
}

// --- LoadTrainable ---

type stubTrainableModel struct {
	stubTextModel
}

func (m *stubTrainableModel) ApplyLoRA(_ LoRAConfig) Adapter { return nil }
func (m *stubTrainableModel) Encode(_ string) []int32        { return nil }
func (m *stubTrainableModel) Decode(_ []int32) string        { return "" }
func (m *stubTrainableModel) NumLayers() int                 { return 26 }

type trainableBackend struct {
	name      string
	available bool
}

func (b *trainableBackend) Name() string    { return b.name }
func (b *trainableBackend) Available() bool { return b.available }
func (b *trainableBackend) LoadModel(_ string, _ ...LoadOption) (TextModel, error) {
	return &stubTrainableModel{stubTextModel: stubTextModel{backend: b.name}}, nil
}

func TestTraining_LoadTrainable_Good(t *testing.T) {
	resetBackends(t)

	Register(&trainableBackend{name: "metal", available: true})

	tm, err := LoadTrainable("/path/to/model")
	require.NoError(t, err)
	require.NotNil(t, tm)
	assert.Equal(t, 26, tm.NumLayers())
	require.NoError(t, tm.Close())
}

func TestTraining_LoadTrainable_Bad_NoBackends(t *testing.T) {
	resetBackends(t)

	_, err := LoadTrainable("/path/to/model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no backends registered")
}

func TestTraining_LoadTrainable_Bad_NotTrainable(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})

	_, err := LoadTrainable("/path/to/model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "does not support training")
}

func TestTraining_LoadTrainable_Bad_LoadError(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{
		name:      "broken",
		available: true,
		loadErr:   core.NewError("GPU out of memory"),
	})

	_, err := LoadTrainable("/path/to/model", WithBackend("broken"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "GPU out of memory")
}

func TestTraining_LoadTrainable_Good_ExplicitBackend(t *testing.T) {
	resetBackends(t)

	Register(&trainableBackend{name: "rocm", available: true})

	tm, err := LoadTrainable("/path/to/model", WithBackend("rocm"))
	require.NoError(t, err)
	require.NotNil(t, tm)
	require.NoError(t, tm.Close())
}

// --- TrainableModel interface compliance ---

func TestTraining_TrainableModel_Good_InterfaceCompliance(t *testing.T) {
	var _ TrainableModel = (*stubTrainableModel)(nil)
}

// --- Ugly: edge cases ---

func TestTraining_LoadTrainable_Ugly_SkipsUnavailableBackend(t *testing.T) {
	resetBackends(t)

	// Preferred backend registered but unavailable; a fallback is available.
	// Default() should skip the unavailable one and return the fallback.
	Register(&trainableBackend{name: "unavailable", available: false})
	Register(&trainableBackend{name: "fallback", available: true})

	// LoadTrainable without explicit backend — Default() picks the available fallback.
	tm, err := LoadTrainable("/path/to/model")
	require.NoError(t, err)
	require.NotNil(t, tm)
	require.NoError(t, tm.Close())
}

func TestTraining_DefaultLoRAConfig_Good_TargetKeysIndependent(t *testing.T) {
	// Mutating the returned TargetKeys should not affect a subsequent call.
	cfg1 := DefaultLoRAConfig()
	cfg1.TargetKeys = append(cfg1.TargetKeys, "o_proj")

	cfg2 := DefaultLoRAConfig()
	assert.Equal(t, []string{"q_proj", "v_proj"}, cfg2.TargetKeys,
		"DefaultLoRAConfig should return independent TargetKeys slices")
	assert.Len(t, cfg1.TargetKeys, 3, "mutated copy should have 3 keys")
}
