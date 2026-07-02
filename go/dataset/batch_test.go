// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"testing"

	core "dappco.re/go"
)

// --- BatchConfig ---

// Good: BatchConfig marshals to the wire shape distillation checkpoints and
// driver configs already depend on (batch_size, max_seq_len,
// sequence_packing, no_eos) — the exact shape distill.BatchConfig carried
// before it became an alias onto this type — and round-trips exactly.
func TestBatchConfig_JSON_Good(t *testing.T) {
	cfg := BatchConfig{BatchSize: 8, MaxSeqLen: 2048, SequencePacking: true, NoEOS: true}
	data := core.JSONMarshalString(cfg)
	for _, want := range []string{`"batch_size":8`, `"max_seq_len":2048`, `"sequence_packing":true`, `"no_eos":true`} {
		if !core.Contains(data, want) {
			t.Fatalf("BatchConfig JSON = %s, want to contain %s", data, want)
		}
	}
	var got BatchConfig
	if r := core.JSONUnmarshalString(data, &got); !r.OK {
		t.Fatalf("JSONUnmarshalString() error = %s", r.Error())
	}
	if got != cfg {
		t.Fatalf("round trip = %+v, want %+v", got, cfg)
	}
}

// Bad: the zero-value BatchConfig marshals to an empty object — every
// field is omitempty, so a driver that never sets a batch shape gets a
// clean "{}" rather than a wall of explicit zeros.
func TestBatchConfig_JSON_Bad(t *testing.T) {
	data := core.JSONMarshalString(BatchConfig{})
	if data != "{}" {
		t.Fatalf("zero-value BatchConfig JSON = %s, want {}", data)
	}
}

// Ugly: decoding an empty JSON object into an already-populated BatchConfig
// leaves every field untouched — encoding/json only overwrites keys that
// are actually present in the input, so "{}" is a no-op, not a reset.
func TestBatchConfig_JSON_Ugly(t *testing.T) {
	got := BatchConfig{BatchSize: 99, MaxSeqLen: 4096}
	if r := core.JSONUnmarshalString("{}", &got); !r.OK {
		t.Fatalf(`JSONUnmarshalString("{}") error = %s`, r.Error())
	}
	if got.BatchSize != 99 || got.MaxSeqLen != 4096 {
		t.Fatalf("decode of {} onto a populated BatchConfig = %+v, want unchanged (99, 4096)", got)
	}
}
