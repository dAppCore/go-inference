// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the capture-first lane (#97): the model's raw return is appended to
// the capture sidecar the moment it exists, independent of any scoring. Ported
// from go-mlx/go/train/capture_test.go and adapted to the neutral port —
// appendCaptureRows is driven directly (no model, no Metal).

package train

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// TestCapture_AppendRowsWritesJSONL asserts appendCaptureRows writes one
// well-formed CaptureRow per eval (step, prompt, raw text, non-zero birth
// timestamp) and reports how many rows landed. A missed capture is a data point
// that never existed, so the row shape is the durable contract.
func TestCapture_AppendRowsWritesJSONL(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "captures.jsonl")
	evals := []SFTEvalResult{
		{Step: 1, Prompt: "q1", Text: "echo:q1"},
		{Step: 1, Prompt: "q2", Text: "echo:q2"},
	}
	if n := appendCaptureRows(path, evals); n != 2 {
		t.Fatalf("appendCaptureRows landed %d rows, want 2", n)
	}
	read, err := coreio.Local.Read(path)
	if err != nil {
		t.Fatalf("capture read: %v", err)
	}
	var row CaptureRow
	first := read[:core.Index(read, "\n")]
	if r := core.JSONUnmarshal([]byte(first), &row); !r.OK {
		t.Fatalf("capture row parse: %v", r.Value)
	}
	if row.Step != 1 || row.Prompt != "q1" || row.Text != "echo:q1" || row.At == 0 {
		t.Fatalf("capture row = %+v", row)
	}
}

// TestCapture_AppendRowsEmptyIsNoOp asserts an empty eval slice or an empty path
// writes nothing and reports zero rows — capture is best-effort and never errors
// on nothing to do.
func TestCapture_AppendRowsEmptyIsNoOp(t *testing.T) {
	if n := appendCaptureRows("", []SFTEvalResult{{Step: 1, Prompt: "q", Text: "a"}}); n != 0 {
		t.Fatalf("empty path landed %d rows, want 0", n)
	}
	if n := appendCaptureRows(core.PathJoin(t.TempDir(), "x.jsonl"), nil); n != 0 {
		t.Fatalf("nil evals landed %d rows, want 0", n)
	}
}

// TestCapture_RowJSONShape asserts CaptureRow serialises to the documented JSONL
// schema — the durable record shape the scorer reads later.
func TestCapture_RowJSONShape(t *testing.T) {
	row := CaptureRow{Step: 7, Prompt: "hold a truth", Text: "I look at it straight.", At: 1700000000}
	encoded := core.JSONMarshal(row)
	if !encoded.OK {
		t.Fatalf("JSONMarshal(CaptureRow) failed: %v", encoded.Value)
	}
	var back CaptureRow
	if r := core.JSONUnmarshal(encoded.Value.([]byte), &back); !r.OK {
		t.Fatalf("JSONUnmarshal(CaptureRow) failed: %v", r.Value)
	}
	if back != row {
		t.Fatalf("round-tripped CaptureRow = %+v, want %+v", back, row)
	}
}
