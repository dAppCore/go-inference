package inference

import (
	"testing"

	"dappco.re/go/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- DefaultLoRAConfig ---

func TestDefaultLoRAConfig_Good(t *testing.T) {
	cfg := DefaultLoRAConfig()
	assert.Equal(t, 8, cfg.Rank, "default Rank should be 8")
	assert.InDelta(t, float32(16), cfg.Alpha, 0.0001, "default Alpha should be 16")
	assert.Equal(t, []string{"q_proj", "v_proj"}, cfg.TargetKeys, "default TargetKeys should be q_proj and v_proj")
	assert.False(t, cfg.BFloat16, "default BFloat16 should be false")
}

func TestDefaultLoRAConfig_Good_Idempotent(t *testing.T) {
	a := DefaultLoRAConfig()
	b := DefaultLoRAConfig()
	assert.Equal(t, a, b, "DefaultLoRAConfig should be idempotent")
}

// --- LoadTrainable ---

// stubTrainableModel extends stubTextModel with TrainableModel methods.
type stubTrainableModel struct {
	stubTextModel
}

func (m *stubTrainableModel) ApplyLoRA(_ LoRAConfig) Adapter { return nil }
func (m *stubTrainableModel) Encode(_ string) []int32        { return nil }
func (m *stubTrainableModel) Decode(_ []int32) string        { return "" }
func (m *stubTrainableModel) NumLayers() int                 { return 26 }

// trainableBackend returns a stubTrainableModel from LoadModel.
type trainableBackend struct {
	name      string
	available bool
}

func (b *trainableBackend) Name() string    { return b.name }
func (b *trainableBackend) Available() bool { return b.available }
func (b *trainableBackend) LoadModel(_ string, _ ...LoadOption) (TextModel, error) {
	return &stubTrainableModel{stubTextModel: stubTextModel{backend: b.name}}, nil
}

func TestLoadTrainable_Good(t *testing.T) {
	resetBackends(t)

	Register(&trainableBackend{name: "metal", available: true})

	tm, err := LoadTrainable("/path/to/model")
	require.NoError(t, err)
	require.NotNil(t, tm)
	assert.Equal(t, 26, tm.NumLayers())
	require.NoError(t, tm.Close())
}

func TestLoadTrainable_Bad_NoBackends(t *testing.T) {
	resetBackends(t)

	_, err := LoadTrainable("/path/to/model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no backends registered")
}

func TestLoadTrainable_Bad_NotTrainable(t *testing.T) {
	resetBackends(t)

	// stubBackend returns a stubTextModel which does NOT implement TrainableModel.
	Register(&stubBackend{name: "metal", available: true})

	_, err := LoadTrainable("/path/to/model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "does not support training")
}

func TestLoadTrainable_Bad_LoadError(t *testing.T) {
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

func TestLoadTrainable_Good_ExplicitBackend(t *testing.T) {
	resetBackends(t)

	Register(&trainableBackend{name: "rocm", available: true})

	tm, err := LoadTrainable("/path/to/model", WithBackend("rocm"))
	require.NoError(t, err)
	require.NotNil(t, tm)
	require.NoError(t, tm.Close())
}

// --- TrainableModel interface compliance ---

func TestTrainableModel_Good_InterfaceCompliance(t *testing.T) {
	var _ TrainableModel = (*stubTrainableModel)(nil)
}
