// SPDX-Licence-Identifier: EUPL-1.2

package train_test

import (
	"fmt"

	"dappco.re/go/inference/train"
)

// ExampleNormalizeConfig shows the defaults NormalizeConfig fills in for
// a zero-value Config.
func ExampleNormalizeConfig() {
	cfg := train.NormalizeConfig(train.Config{})
	fmt.Println("batch size:", cfg.BatchSize)
	fmt.Println("epochs:", cfg.Epochs)
	fmt.Println("learning rate:", cfg.LearningRate)
	fmt.Println("eval max tokens:", cfg.EvalMaxTokens)
	// Output:
	// batch size: 1
	// epochs: 1
	// learning rate: 1e-05
	// eval max tokens: 96
}

// ExampleEffectiveBatchSize shows the optimizer batch size a driver's own
// training loop reports after gradient accumulation.
func ExampleEffectiveBatchSize() {
	cfg := train.Config{BatchSize: 4, GradientAccumulationSteps: 8}
	fmt.Println("effective batch size:", train.EffectiveBatchSize(cfg))
	// Output:
	// effective batch size: 32
}
