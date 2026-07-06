// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Example_trainer demonstrates the [Trainer] lifecycle from the trainer.go
// doc comment: open, step the optimiser, save the adapter, close — against a
// fake engine Trainer, since a real one needs live device state. trainer.go
// itself declares only the Trainer/TrainerModel interfaces (no package-level
// funcs of its own to hang an ExampleXxx off), so this is a plain usage
// example rather than a per-symbol one.
func Example_trainer() {
	tr := &fakeTrainer{stepLoss: 0.5}
	defer func() { _ = tr.Close() }()

	loss, err := tr.Step(inference.Batch{TokenIDs: [][]int32{{1, 2, 3}}})
	if err != nil {
		panic(err)
	}
	if err := tr.Save("/models/lora/domain-v1"); err != nil {
		panic(err)
	}
	core.Println(loss)
	// Output: 0.5
}
