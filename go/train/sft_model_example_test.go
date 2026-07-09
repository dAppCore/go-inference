// SPDX-Licence-Identifier: EUPL-1.2

package train_test

import (
	"context"
	"fmt"
	"iter"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
	"dappco.re/go/inference/train"
	"dappco.re/go/inference/train/dataset"
)

// echoModel is a minimal inference.TextModel usable from outside the train
// package: Generate echoes the prompt as a single token, and OpenTrainer
// forwards an injected engine.Trainer — the engine.TrainerModel seam
// RunSFTModel probes for. It deliberately does NOT implement
// inference.PromptCacheWarmer, so it also serves the "no warm capability"
// path in ExampleRunSSDModel.
type echoModel struct{ trainer engine.Trainer }

func (m *echoModel) Generate(_ context.Context, prompt string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		yield(inference.Token{ID: 1, Text: "echo:" + prompt})
	}
}
func (m *echoModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}
func (m *echoModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}
func (m *echoModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}
func (m *echoModel) ModelType() string                  { return "echo" }
func (m *echoModel) Info() inference.ModelInfo          { return inference.ModelInfo{Architecture: "echo"} }
func (m *echoModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *echoModel) Err() core.Result                   { return core.Ok(nil) }
func (m *echoModel) Close() core.Result                 { return core.Ok(nil) }

// OpenTrainer forwards the injected trainer.
func (m *echoModel) OpenTrainer(inference.TrainingConfig) (engine.Trainer, error) {
	if m.trainer == nil {
		return nil, core.NewError("echo: no trainer")
	}
	return m.trainer, nil
}

// echoTrainer is a minimal engine.Trainer: it counts optimiser updates and
// returns a monotonically falling loss, same shape as a real adapter's early
// training curve.
type echoTrainer struct{ steps int }

func (t *echoTrainer) Step(inference.Batch) (float64, error) {
	t.steps++
	return 1.0 / float64(t.steps), nil
}
func (t *echoTrainer) StepAccumulated([]inference.Batch) (float64, error) {
	t.steps++
	return 1.0 / float64(t.steps), nil
}
func (t *echoTrainer) Loss(inference.Batch) (float64, error) { return 1.0, nil }
func (t *echoTrainer) Save(string) error                     { return nil }
func (t *echoTrainer) Close() error                          { return nil }

// runeEncode tokenises text as one id per rune — deterministic and
// dependency-free, long enough for the ≥2-token causal-target floor.
func runeEncode(text string) []int32 {
	runes := []rune(text)
	ids := make([]int32, len(runes))
	for i, r := range runes {
		ids[i] = int32(r)
	}
	return ids
}

// ExampleRunSFTModel shows the Model-bound SFT entry: RunSFTModel probes the
// loaded model for the engine.TrainerModel seam, opens a trainer, and drives
// the LoRA SFT loop over the training set.
func ExampleRunSFTModel() {
	model := &echoModel{trainer: &echoTrainer{}}
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "hello", Response: "a helpful response"},
		{Prompt: "world", Response: "another response"},
	})
	cfg := train.SFTConfig{
		Config: train.Config{BatchSize: 1, GradientAccumulationSteps: 1, Epochs: 1},
		LoRA:   inference.LoRAConfig{Rank: 8, Alpha: 16},
	}

	result, err := train.RunSFTModel(context.Background(), model, runeEncode, ds, cfg)
	if err != nil {
		panic(err)
	}
	fmt.Println("steps:", result.Steps)
	fmt.Printf("last loss: %.2f\n", result.LastLoss)
	// Output:
	// steps: 2
	// last loss: 0.50
}
