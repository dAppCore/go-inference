// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"math"
	"os"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestHIPLoRATrainer_InitialFactors_Good(t *testing.T) {
	first := hipInitializeLoRAFactorA(8, 4)
	second := hipInitializeLoRAFactorA(8, 4)

	core.AssertEqual(t, first, second)
	core.AssertTrue(t, rocmFloat32SliceFinite(first))
	nonzero := false
	for _, value := range first {
		if value != 0 {
			nonzero = true
			break
		}
	}
	core.AssertTrue(t, nonzero)
}

func TestHIPLoRATrainer_HeadDelta_Good(t *testing.T) {
	logits, err := hipApplyHeadLoRA(
		[]float32{2, 3},
		[]float32{10, 20, 30},
		[]float32{1, 2},
		[]float32{1, 2, 3},
		3, 2, 1, 2,
	)

	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{26, 52, 78}, logits)
}

func TestHIPLoRATrainer_HeadDelta_Bad(t *testing.T) {
	_, err := hipApplyHeadLoRA([]float32{1}, []float32{1, 2}, []float32{1}, []float32{1, 2}, 2, 2, 1, 1)
	core.AssertError(t, err)
}

func TestHIPLoRATrainer_Save_Good(t *testing.T) {
	state, err := NewNativeLoRAAdamWState([]float32{0.25, -0.5}, []float32{0.75, 1.25, -1.5}, 3, 2, 1, DefaultNativeAdamWConfig())
	core.RequireNoError(t, err)
	trainer := &hipLoRATrainer{state: state, rows: 3, cols: 2, rank: 1, alpha: 4}
	path := t.TempDir()

	core.RequireNoError(t, trainer.Save(path))
	adapter, err := loadMetalHeadLoRAAdapter(path)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{0.25, -0.5}, adapter.a)
	core.AssertEqual(t, []float32{0.75, 1.25, -1.5}, adapter.b)
}

func TestHIPLoRATrainer_Save_Ugly_Closed(t *testing.T) {
	trainer := &hipLoRATrainer{closed: true}

	err := trainer.Save(t.TempDir())

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "trainer is closed")
}

func TestHIPLoRATrainer_E2BHardware_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_LORA_TRAINER_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_LORA_TRAINER_TESTS=1 to run the HIP LoRA lifecycle receipt")
	}
	modelPath := strings.TrimSpace(os.Getenv("GO_ROCM_MODEL_PATH"))
	if modelPath == "" || strings.TrimSpace(os.Getenv("GO_ROCM_KERNEL_HSACO")) == "" || !ROCmAvailable() {
		t.Skip("a linked Gemma4 model, HSACO, and ROCm device are required")
	}

	result := (&rocmBackend{}).LoadModel(modelPath, inference.WithContextLen(64))
	if !result.OK {
		t.Fatalf("LoadModel: %v", result.Value)
	}
	model, ok := result.Value.(*rocmModel)
	if !ok || model.engineModel == nil {
		t.Fatalf("LoadModel returned %T without the shared engine model", result.Value)
	}
	defer model.Close()

	trainerValue, err := model.engineModel.OpenTrainer(inference.TrainingConfig{
		LearningRate: 0.01,
		LoRA:         inference.LoRAConfig{Rank: 1, Alpha: 2},
	})
	core.RequireNoError(t, err)
	trainer, ok := trainerValue.(*hipLoRATrainer)
	if !ok {
		t.Fatalf("OpenTrainer returned %T, want *hipLoRATrainer", trainerValue)
	}
	defer trainer.Close()

	ids, err := model.Tokenize("hello world")
	core.RequireNoError(t, err)
	if len(ids) < 2 {
		t.Fatalf("training prompt tokenized to %d tokens", len(ids))
	}
	if len(ids) > 3 {
		ids = ids[:3]
	}
	before := append([]float32(nil), trainer.state.Parameters()...)
	loss, err := trainer.Step(inference.Batch{TokenIDs: [][]int32{ids}})
	core.RequireNoError(t, err)
	if loss <= 0 || math.IsNaN(loss) || math.IsInf(loss, 0) {
		t.Fatalf("training loss = %g, want finite positive", loss)
	}
	changed := false
	for index, value := range trainer.state.Parameters() {
		if value != before[index] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("real HIP trainer step did not update a LoRA parameter")
	}

	adapterPath := t.TempDir()
	core.RequireNoError(t, trainer.Save(adapterPath))
	identity, err := model.LoadAdapter(adapterPath)
	core.RequireNoError(t, err)
	core.AssertEqual(t, "hip_gemma4_lm_head", identity.Labels["adapter_runtime"])

	tokens := collectHIPHardwareTokens(model.Generate(context.Background(), "hello", inference.WithMaxTokens(1)))
	if err := model.Err(); !err.OK {
		t.Fatalf("Generate with reloaded adapter: %v", err.Value)
	}
	if len(tokens) != 1 {
		t.Fatalf("Generate with reloaded adapter returned %d tokens, want 1", len(tokens))
	}
}
