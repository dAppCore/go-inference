// SPDX-Licence-Identifier: EUPL-1.2

package dataset

// BatchConfig controls how a driver's tokenizer batches dataset samples for
// training, evaluation, or distillation: how many samples per batch, the
// max sequence length a sample is truncated/padded to, whether same-length
// samples are packed into shared sequences, and whether the trailing EOS
// token is omitted. None of that is specific to what trains on the result
// — every consumer of a Dataset needs the same batch shape — so the
// canonical definition lives here rather than being redefined by each
// caller. distill.BatchConfig is a compatibility alias onto this type (see
// dappco.re/go/inference/distill); it previously duplicated these same
// four fields verbatim.
//
//	cfg := dataset.BatchConfig{BatchSize: 8, MaxSeqLen: 2048}
//	batches, err := driverTokenizer.Build(ds, cfg)
type BatchConfig struct {
	BatchSize       int  `json:"batch_size,omitempty"`
	MaxSeqLen       int  `json:"max_seq_len,omitempty"`
	SequencePacking bool `json:"sequence_packing,omitempty"`
	NoEOS           bool `json:"no_eos,omitempty"`
}
