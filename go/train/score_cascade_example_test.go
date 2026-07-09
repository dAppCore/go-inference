// SPDX-Licence-Identifier: EUPL-1.2

package train_test

import (
	"fmt"

	"dappco.re/go/inference/train"
)

// ExampleScoreFunc shows the shape a driver's own scorer implements: score
// one (prompt, response) pair and return the dimensions the cascade reads.
// Step and At are stamped by the cascade itself, not the scorer.
func ExampleScoreFunc() {
	var score train.ScoreFunc = func(prompt, text string) train.ScoreRecord {
		return train.ScoreRecord{LEK: float64(len(text)) / float64(len(prompt)+len(text))}
	}

	rec := score("hello", "a longer helpful reply")
	fmt.Printf("lek: %.2f\n", rec.LEK)
	// Output:
	// lek: 0.81
}
