// SPDX-Licence-Identifier: EUPL-1.2

package train_test

import (
	"context"
	"fmt"

	"dappco.re/go/inference/train"
	"dappco.re/go/inference/train/dataset"
)

// ExampleRunSFT shows the engine-neutral SFT loop driven directly over an
// opened engine.Trainer — the seam a driver's own engine implements, and the
// one RunSFTModel opens automatically from a loaded inference.TextModel.
func ExampleRunSFT() {
	trainer := &echoTrainer{}
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "p1", Response: "a helpful response"},
		{Prompt: "p2", Response: "another response"},
	})
	cfg := train.SFTConfig{Config: train.Config{BatchSize: 1, GradientAccumulationSteps: 1, Epochs: 1}}

	result, err := train.RunSFT(context.Background(), trainer, runeEncode, nil, ds, cfg)
	if err != nil {
		panic(err)
	}
	fmt.Println("steps:", result.Steps)
	fmt.Println("samples:", result.Samples)
	// Output:
	// steps: 2
	// samples: 2
}

// ExampleSFTResult_Metrics shows the JSON-friendly summary a dashboard reads
// off a finished (or in-flight) SFT run.
func ExampleSFTResult_Metrics() {
	result := &train.SFTResult{Steps: 4, Epochs: 1, Samples: 8, LastLoss: 0.5}
	cfg := train.SFTConfig{Config: train.Config{BatchSize: 2, GradientAccumulationSteps: 2, LearningRate: 1e-4}}

	m := result.Metrics(cfg)
	fmt.Println("effective batch size:", m.EffectiveBatchSize)
	fmt.Println("last loss:", m.LastLoss)
	// Output:
	// effective batch size: 4
	// last loss: 0.5
}
