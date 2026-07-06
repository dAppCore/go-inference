// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	core "dappco.re/go"
)

// echoBackend is a minimal Backend fake: it returns each input unchanged (identity
// decode), or the configured error when one is set — enough to prove the contract
// shape (T inputs in, T outputs out, error propagates) without a real transformer.
type echoBackend struct{ err error }

func (b echoBackend) DecodeForward(inputs [][]byte) ([][]byte, error) {
	if b.err != nil {
		return nil, b.err
	}
	return inputs, nil
}

var _ Backend = echoBackend{} // compile-time: echoBackend satisfies Backend

// TestBackend_DecodeForward_Good covers the ordinary shape: T input embeddings in,
// T output hidden states out, same byte content (the seam's bf16-bytes contract).
func TestBackend_DecodeForward_Good(t *testing.T) {
	var b Backend = echoBackend{}
	in := [][]byte{{1, 2}, {3, 4}, {5, 6}}
	out, err := b.DecodeForward(in)
	if err != nil {
		t.Fatalf("DecodeForward: %v", err)
	}
	if len(out) != len(in) {
		t.Fatalf("DecodeForward returned %d outputs, want %d (one per input)", len(out), len(in))
	}
	for i := range in {
		if string(out[i]) != string(in[i]) {
			t.Fatalf("DecodeForward[%d] = %v, want %v", i, out[i], in[i])
		}
	}
}

// TestBackend_DecodeForward_Bad covers a backend's error path: DecodeForward's error
// return propagates to the caller untouched, so a decode failure surfaces cleanly
// rather than being swallowed.
func TestBackend_DecodeForward_Bad(t *testing.T) {
	wantErr := core.NewError("backend: decode failed")
	var b Backend = echoBackend{err: wantErr}
	if _, err := b.DecodeForward([][]byte{{1}}); err != wantErr {
		t.Fatalf("DecodeForward error = %v, want the backend's own error %v", err, wantErr)
	}
}

// TestBackend_DecodeForward_Ugly covers the degenerate empty-sequence call: zero
// inputs must not panic, and must return zero outputs (nothing to decode).
func TestBackend_DecodeForward_Ugly(t *testing.T) {
	var b Backend = echoBackend{}
	out, err := b.DecodeForward(nil)
	if err != nil {
		t.Fatalf("DecodeForward(nil) = %v, want nil error", err)
	}
	if len(out) != 0 {
		t.Fatalf("DecodeForward(nil) = %v, want empty", out)
	}
}
