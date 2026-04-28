package inference

import (
	"errors"
	"testing"
)

// --- DefaultLoRAConfig ---

func TestTraining_DefaultLoRAConfig_Good(t *testing.T) {
	cfg := DefaultLoRAConfig()
	checkEqual(t, 8, cfg.Rank)
	checkInDelta(t, float32(16), cfg.Alpha, 0.0001)
	checkEqual(t, []string{"q_proj", "v_proj"}, cfg.TargetKeys)
	checkFalse(t, cfg.BFloat16)
}

func TestTraining_DefaultLoRAConfig_Good_Idempotent(t *testing.T) {
	firstConfig := DefaultLoRAConfig()
	secondConfig := DefaultLoRAConfig()
	checkEqual(t, firstConfig, secondConfig)
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
	checkNoError(t, err)
	checkNotNil(t, tm)
	checkEqual(t, 26, tm.NumLayers())
	checkNoError(t, tm.Close())
}

func TestTraining_LoadTrainable_Bad_NoBackends(t *testing.T) {
	resetBackends(t)

	_, err := LoadTrainable("/path/to/model")
	checkError(t, err)
	checkContains(t, err.Error(), "no backends registered")
}

func TestTraining_LoadTrainable_Bad_NotTrainable(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true})

	_, err := LoadTrainable("/path/to/model")
	checkError(t, err)
	checkContains(t, err.Error(), "does not support training")
}

func TestTraining_LoadTrainable_Bad_LoadError(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{
		name:      "broken",
		available: true,
		loadErr:   errors.New("GPU out of memory"),
	})

	_, err := LoadTrainable("/path/to/model", WithBackend("broken"))
	checkError(t, err)
	checkContains(t, err.Error(), "GPU out of memory")
}

func TestTraining_LoadTrainable_Bad_NilModel(t *testing.T) {
	resetBackends(t)

	Register(&stubBackend{name: "metal", available: true, nilModel: true})

	_, err := LoadTrainable("/path/to/model")
	checkError(t, err)
	checkContains(t, err.Error(), "returned a nil model")
}

func TestTraining_LoadTrainable_Good_ExplicitBackend(t *testing.T) {
	resetBackends(t)

	Register(&trainableBackend{name: "rocm", available: true})

	tm, err := LoadTrainable("/path/to/model", WithBackend("rocm"))
	checkNoError(t, err)
	checkNotNil(t, tm)
	checkNoError(t, tm.Close())
}

// --- TrainableModel interface compliance ---

func TestTraining_TrainableModel_Good_InterfaceCompliance(t *testing.T) {
	var _ TrainableModel = (*stubTrainableModel)(nil)
	model := &stubTrainableModel{}
	checkEqual(t, 26, model.NumLayers())
	checkNil(t, model.ApplyLoRA(DefaultLoRAConfig()))
}

// --- Ugly: edge cases ---

func TestTraining_LoadTrainable_Ugly_SkipsUnavailableBackend(t *testing.T) {
	resetBackends(t)

	Register(&trainableBackend{name: "unavailable", available: false})
	Register(&trainableBackend{name: "fallback", available: true})

	tm, err := LoadTrainable("/path/to/model")
	checkNoError(t, err)
	checkNotNil(t, tm)
	checkNoError(t, tm.Close())
}

func TestTraining_DefaultLoRAConfig_Good_TargetKeysIndependent(t *testing.T) {
	cfg1 := DefaultLoRAConfig()
	cfg1.TargetKeys = append(cfg1.TargetKeys, "o_proj")

	cfg2 := DefaultLoRAConfig()
	checkEqual(t, []string{"q_proj", "v_proj"}, cfg2.TargetKeys)
	checkLen(t, cfg1.TargetKeys, 3)
}

// --- LoRAConfig Bad: zero/negative values are accepted at the config layer ---

func TestTraining_LoRAConfig_Bad_ZeroRank(t *testing.T) {
	cfg := LoRAConfig{Rank: 0, Alpha: 16, TargetKeys: []string{"q_proj"}}
	checkEqual(t, 0, cfg.Rank)
	checkInDelta(t, float32(16), cfg.Alpha, 0.0001)
	checkEqual(t, []string{"q_proj"}, cfg.TargetKeys)
}

func TestTraining_LoRAConfig_Bad_NegativeRank(t *testing.T) {
	cfg := LoRAConfig{Rank: -8, Alpha: 16, TargetKeys: []string{"q_proj"}}
	checkEqual(t, -8, cfg.Rank)
	checkInDelta(t, float32(16), cfg.Alpha, 0.0001)
	checkEqual(t, []string{"q_proj"}, cfg.TargetKeys)
}

func TestTraining_LoRAConfig_Bad_ZeroAlpha(t *testing.T) {
	cfg := LoRAConfig{Rank: 8, Alpha: 0, TargetKeys: []string{"q_proj"}}
	checkInDelta(t, float32(0), cfg.Alpha, 0.0001)
	checkEqual(t, 8, cfg.Rank)
	checkEqual(t, []string{"q_proj"}, cfg.TargetKeys)
}

// --- LoRAConfig Ugly: atypical but valid configurations ---

func TestTraining_LoRAConfig_Ugly_EmptyTargetKeys(t *testing.T) {
	cfg := LoRAConfig{Rank: 8, Alpha: 16, TargetKeys: []string{}}
	checkEmpty(t, cfg.TargetKeys)
	checkEqual(t, 8, cfg.Rank)
	checkInDelta(t, float32(16), cfg.Alpha, 0.0001)
}

func TestTraining_LoRAConfig_Ugly_NilTargetKeys(t *testing.T) {
	cfg := LoRAConfig{Rank: 8, Alpha: 16}
	checkNil(t, cfg.TargetKeys)
	checkEqual(t, 8, cfg.Rank)
	checkInDelta(t, float32(16), cfg.Alpha, 0.0001)
}

func TestTraining_LoRAConfig_Ugly_BFloat16WithHighRank(t *testing.T) {
	cfg := LoRAConfig{Rank: 64, Alpha: 128, TargetKeys: []string{"q_proj", "k_proj", "v_proj"}, BFloat16: true}
	checkEqual(t, 64, cfg.Rank)
	checkInDelta(t, float32(128), cfg.Alpha, 0.0001)
	checkTrue(t, cfg.BFloat16)
	checkLen(t, cfg.TargetKeys, 3)
}
