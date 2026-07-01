// SPDX-Licence-Identifier: EUPL-1.2

package distill_test

import (
	"fmt"

	"dappco.re/go/inference/distill"
)

// ExampleBatchLoss computes the KL-divergence distillation loss between a
// teacher and student's per-token logits over a masked batch.
func ExampleBatchLoss() {
	teacher := distill.Logits{{{2, 0, 0}, {0, 2, 0}}}
	student := distill.Logits{{{2, 0, 0}, {0, 0, 2}}}
	mask := [][]float32{{1, 1}}

	loss, err := distill.BatchLoss(teacher, student, mask, distill.Config{
		Loss:        distill.LossKL,
		Temperature: 1,
	})
	if err != nil {
		panic(err)
	}
	fmt.Println("tokens:", loss.Tokens)
	fmt.Println("kl >= 0:", loss.KL >= 0)
	fmt.Println("value == kl:", loss.Value == loss.KL)
	// Output:
	// tokens: 2
	// kl >= 0: true
	// value == kl: true
}

// ExampleBatchCacheKey derives a stable teacher-logit cache key from a
// batch's token IDs, shifted targets, and loss mask.
func ExampleBatchCacheKey() {
	tokens := [][]int{{1, 2, 3}}
	targets := [][]int{{2, 3, 4}}
	mask := [][]float32{{1, 1, 1}}

	first := distill.BatchCacheKey(tokens, targets, mask)
	second := distill.BatchCacheKey(tokens, targets, mask)
	fmt.Println("stable:", first == second)
	fmt.Println("length:", len(first))
	// Output:
	// stable: true
	// length: 64
}

// ExampleNormalizeConfig shows the defaults NormalizeConfig fills in for
// a zero-value Config.
func ExampleNormalizeConfig() {
	cfg := distill.NormalizeConfig(distill.Config{})
	fmt.Println("epochs:", cfg.Epochs)
	fmt.Println("temperature:", cfg.Temperature)
	fmt.Println("loss:", cfg.Loss)
	fmt.Println("batch size:", cfg.Batch.BatchSize)
	// Output:
	// epochs: 1
	// temperature: 1
	// loss: kl
	// batch size: 1
}
