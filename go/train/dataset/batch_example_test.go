// SPDX-Licence-Identifier: EUPL-1.2

package dataset_test

import (
	"fmt"

	"dappco.re/go/inference/train/dataset"
)

// ExampleBatchConfig shows the generic batch shape a driver's tokenizer
// reads to build a training, evaluation, or distillation batch: how many
// samples per batch, the max sequence length, whether same-length samples
// get packed into shared sequences, and whether the trailing EOS token is
// omitted.
func ExampleBatchConfig() {
	cfg := dataset.BatchConfig{BatchSize: 8, MaxSeqLen: 2048}
	fmt.Println("batch size:", cfg.BatchSize)
	fmt.Println("max seq len:", cfg.MaxSeqLen)
	fmt.Println("sequence packing:", cfg.SequencePacking)
	// Output:
	// batch size: 8
	// max seq len: 2048
	// sequence packing: false
}
