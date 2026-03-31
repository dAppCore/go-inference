package inference

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- DefaultLoRAConfig ---

func TestDefaultLoRAConfig_Good(t *testing.T) {
	config := DefaultLoRAConfig()
	assert.Equal(t, 8, config.Rank, "default Rank should be 8")
	assert.InDelta(t, float32(16), config.Alpha, 0.0001, "default Alpha should be 16")
	assert.Equal(t, []string{"q_proj", "v_proj"}, config.TargetKeys, "default TargetKeys should be q_proj and v_proj")
	assert.False(t, config.BFloat16, "default BFloat16 should be false")
}

func TestDefaultLoRAConfig_Good_Idempotent(t *testing.T) {
	firstConfig := DefaultLoRAConfig()
	secondConfig := DefaultLoRAConfig()
	assert.Equal(t, firstConfig, secondConfig, "DefaultLoRAConfig should be idempotent")
}

// --- LoadTrainable ---

// stubTrainableModel extends stubTextModel with TrainableModel methods.
type stubTrainableModel struct {
	stubTextModel
}

func (model *stubTrainableModel) ApplyLoRA(_ LoRAConfig) Adapter { return nil }
func (model *stubTrainableModel) Encode(_ string) []int32        { return nil }
func (model *stubTrainableModel) Decode(_ []int32) string        { return "" }
func (model *stubTrainableModel) NumLayers() int                 { return 26 }

// trainableBackend returns a stubTrainableModel from LoadModel.
type trainableBackend struct {
	name      string
	available bool
}

func (backend *trainableBackend) Name() string    { return backend.name }
func (backend *trainableBackend) Available() bool { return backend.available }
func (backend *trainableBackend) LoadModel(_ string, _ ...LoadOption) (TextModel, error) {
	return &stubTrainableModel{stubTextModel: stubTextModel{backend: backend.name}}, nil
}

func TestLoadTrainable_Good(t *testing.T) {
	resetBackends(t)

	Register(&trainableBackend{name: "metal", available: true})

	trainableModel, err := LoadTrainable("/path/to/model")
	require.NoError(t, err)
	require.NotNil(t, trainableModel)
	assert.Equal(t, 26, trainableModel.NumLayers())
	require.NoError(t, trainableModel.Close())
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
		loadErr:   fmt.Errorf("GPU out of memory"),
	})

	_, err := LoadTrainable("/path/to/model", WithBackend("broken"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "GPU out of memory")
}

func TestLoadTrainable_Good_ExplicitBackend(t *testing.T) {
	resetBackends(t)

	Register(&trainableBackend{name: "rocm", available: true})

	trainableModel, err := LoadTrainable("/path/to/model", WithBackend("rocm"))
	require.NoError(t, err)
	require.NotNil(t, trainableModel)
	require.NoError(t, trainableModel.Close())
}

// --- TrainableModel interface compliance ---

func TestTrainableModel_Good_InterfaceCompliance(t *testing.T) {
	var _ TrainableModel = (*stubTrainableModel)(nil)
}
