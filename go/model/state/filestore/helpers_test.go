// SPDX-Licence-Identifier: EUPL-1.2

// filestore shared-helper tests: bounded payload writer, full-write loop, context gate and numeric/result utilities.
package filestore

import (
	"context"
	stdio "io"
	"testing"

	core "dappco.re/go"
)

// errWriter is a minimal stdio.Writer that always fails. It drives
// writeAll and limitedPayloadWriter.Write down their underlying-write-
// error branches deterministically — a real *os.File on a regular
// file cannot be made to fail a Write hermetically once opened.
type errWriter struct{ err error }

func (w errWriter) Write(data []byte) (int, error) { return 0, w.err }

// shortWriter is a minimal stdio.Writer that reports fewer bytes
// written than it was given, without an error. This is the classic
// io.Writer short-write contract; Go's internal/poll already loops
// *os.File.Write to completion for regular files, so only a synthetic
// writer can reach this branch.
type shortWriter struct{ n int }

func (w shortWriter) Write(data []byte) (int, error) { return w.n, nil }

func TestHelpers_writeAll_Good_MultiChunkWrite(t *testing.T) {
	var got []byte
	sink := &collectingWriter{dst: &got}
	if err := writeAll(sink, []byte("hello world")); err != nil {
		t.Fatalf("writeAll() error = %v", err)
	}
	if string(got) != "hello world" {
		t.Fatalf("writeAll() wrote %q, want %q", got, "hello world")
	}
	if sink.calls < 2 {
		t.Fatalf("writeAll() called Write %d times, want >= 2 to prove the loop drains partial writes", sink.calls)
	}
}

func TestHelpers_writeAll_Bad_WriterError(t *testing.T) {
	want := core.NewError("boom")
	err := writeAll(errWriter{err: want}, []byte("data"))
	if !core.Is(err, want) {
		t.Fatalf("writeAll() error = %v, want %v", err, want)
	}
}

func TestHelpers_writeAll_Bad_ShortWriteNoProgress(t *testing.T) {
	err := writeAll(shortWriter{n: 0}, []byte("data"))
	if !core.Is(err, stdio.ErrShortWrite) {
		t.Fatalf("writeAll() error = %v, want %v", err, stdio.ErrShortWrite)
	}
}

// collectingWriter splits each Write into at most 3 bytes so writeAll's
// for-loop is provably exercised across multiple iterations rather than
// draining the whole buffer in one call.
type collectingWriter struct {
	dst   *[]byte
	calls int
}

func (w *collectingWriter) Write(data []byte) (int, error) {
	w.calls++
	n := min(len(data), 3)
	*w.dst = append(*w.dst, data[:n]...)
	return n, nil
}

// fullWriter always accepts the entire buffer in one call — the shape
// limitedPayloadWriter expects from its single-shot underlying Write
// (it does not loop; looping is writeAll's job one layer up).
type fullWriter struct{ dst *[]byte }

func (w fullWriter) Write(data []byte) (int, error) {
	*w.dst = append(*w.dst, data...)
	return len(data), nil
}

func TestLimitedPayloadWriter_Write_Good_WithinLimit(t *testing.T) {
	var got []byte
	w := limitedPayloadWriter{file: fullWriter{dst: &got}, remaining: 5}
	n, err := w.Write([]byte("hello"))
	if err != nil {
		t.Fatalf("Write() error = %v", err)
	}
	if n != 5 {
		t.Fatalf("Write() n = %d, want 5", n)
	}
	if w.remaining != 0 {
		t.Fatalf("remaining = %d, want 0", w.remaining)
	}
	if string(got) != "hello" {
		t.Fatalf("underlying data = %q, want %q", got, "hello")
	}
}

func TestLimitedPayloadWriter_Write_Bad_Oversized(t *testing.T) {
	w := limitedPayloadWriter{file: shortWriter{n: 0}, remaining: 2}
	_, err := w.Write([]byte("too long"))
	if !core.Is(err, errPayloadOversize) {
		t.Fatalf("Write() error = %v, want %v", err, errPayloadOversize)
	}
}

func TestLimitedPayloadWriter_Write_Bad_UnderlyingError(t *testing.T) {
	want := core.NewError("disk gone")
	w := limitedPayloadWriter{file: errWriter{err: want}, remaining: 4}
	n, err := w.Write([]byte("data"))
	if !core.Is(err, want) {
		t.Fatalf("Write() error = %v, want %v", err, want)
	}
	if n != 0 {
		t.Fatalf("Write() n = %d, want 0", n)
	}
	if w.remaining != 4 {
		t.Fatalf("remaining = %d after a failed write, want unchanged 4", w.remaining)
	}
}

func TestLimitedPayloadWriter_Write_Bad_ShortWrite(t *testing.T) {
	w := limitedPayloadWriter{file: shortWriter{n: 2}, remaining: 5}
	n, err := w.Write([]byte("hello"))
	if !core.Is(err, stdio.ErrShortWrite) {
		t.Fatalf("Write() error = %v, want %v", err, stdio.ErrShortWrite)
	}
	if n != 2 {
		t.Fatalf("Write() n = %d, want 2", n)
	}
	if w.remaining != 3 {
		t.Fatalf("remaining = %d, want 3 (5 declared - 2 actually written)", w.remaining)
	}
}

func TestCheckContext_Good_NilContext(t *testing.T) {
	if err := checkContext(nil); err != nil {
		t.Fatalf("checkContext(nil) error = %v, want nil", err)
	}
}

func TestCheckContext_Good_LiveContext(t *testing.T) {
	if err := checkContext(context.Background()); err != nil {
		t.Fatalf("checkContext(live) error = %v, want nil", err)
	}
}

func TestCheckContext_Bad_CancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := checkContext(ctx)
	if !core.Is(err, context.Canceled) {
		t.Fatalf("checkContext(cancelled) error = %v, want context.Canceled", err)
	}
}

func TestIntFromUint64_Good_WithinRange(t *testing.T) {
	got, err := intFromUint64(42, "widget")
	if err != nil {
		t.Fatalf("intFromUint64() error = %v", err)
	}
	if got != 42 {
		t.Fatalf("intFromUint64() = %d, want 42", got)
	}
}

func TestIntFromUint64_Bad_Overflow(t *testing.T) {
	_, err := intFromUint64(uint64(maxInt())+1, "widget")
	if err == nil {
		t.Fatal("intFromUint64(overflow) error = nil")
	}
}
