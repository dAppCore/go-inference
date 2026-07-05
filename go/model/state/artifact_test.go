// SPDX-Licence-Identifier: EUPL-1.2

package state

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// fakeArtifactWriter records every Put call it receives (text, options, and
// call count) and returns a caller-configured ref or error — used to prove
// ExportArtifact's Writer interactions precisely rather than only its
// absence of a crash.
type fakeArtifactWriter struct {
	calls   int
	gotText string
	gotOpts PutOptions
	ref     ChunkRef
	err     error
}

func (w *fakeArtifactWriter) Put(_ context.Context, text string, opts PutOptions) (ChunkRef, error) {
	w.calls++
	w.gotText = text
	w.gotOpts = opts
	if w.err != nil {
		return ChunkRef{}, w.err
	}
	return w.ref, nil
}

// TestExportArtifact_Good proves the happy path: the local-save hook runs,
// the record carries every metadata field, Put.Kind defaults from Kind, and
// the returned ChunkRef is backfilled from the Writer.
func TestExportArtifact_Good(t *testing.T) {
	var saveCalls int
	var savedPath string
	writer := &fakeArtifactWriter{ref: ChunkRef{ChunkID: 9, Codec: "test/codec"}}

	payload := map[string]any{"coherence": 0.87, "layers": 4}
	record, err := ExportArtifact(context.Background(), payload, ArtifactOptions{
		Model:     "lem-gemma",
		Prompt:    "trace me",
		Kind:      "go-mlx/session-state",
		LocalPath: "/tmp/state.kvbin",
		Save: func(path string) error {
			saveCalls++
			savedPath = path
			return nil
		},
		Store: writer,
		Put:   PutOptions{URI: "mlx://session/trace-1", Title: "LEM Gemma trace"},
	})
	if err != nil {
		t.Fatalf("ExportArtifact() error = %v", err)
	}

	if saveCalls != 1 || savedPath != "/tmp/state.kvbin" {
		t.Fatalf("Save calls = %d, path = %q, want 1 call at /tmp/state.kvbin", saveCalls, savedPath)
	}
	if record.Version != ArtifactVersion {
		t.Fatalf("Version = %d, want %d", record.Version, ArtifactVersion)
	}
	if record.Kind != "go-mlx/session-state" || record.Model != "lem-gemma" || record.Prompt != "trace me" {
		t.Fatalf("record metadata = %+v", record)
	}
	if record.LocalPath != "/tmp/state.kvbin" {
		t.Fatalf("LocalPath = %q, want /tmp/state.kvbin", record.LocalPath)
	}
	if record.ChunkRef != writer.ref {
		t.Fatalf("ChunkRef = %#v, want backfilled %#v", record.ChunkRef, writer.ref)
	}
	if writer.calls != 1 {
		t.Fatalf("Put calls = %d, want 1", writer.calls)
	}
	if writer.gotOpts.Kind != "go-mlx/session-state" {
		t.Fatalf("PutOptions.Kind = %q, want defaulted from opts.Kind", writer.gotOpts.Kind)
	}
	if writer.gotOpts.URI != "mlx://session/trace-1" || writer.gotOpts.Title != "LEM Gemma trace" {
		t.Fatalf("PutOptions passthrough = %+v", writer.gotOpts)
	}
	if !core.Contains(writer.gotText, `"model": "lem-gemma"`) || !core.Contains(writer.gotText, `"coherence": 0.87`) {
		t.Fatalf("archived text = %q", writer.gotText)
	}
}

// TestExportArtifact_Bad proves a nil payload is rejected up front, and a
// failing local-save hook short-circuits before the Writer is ever touched.
func TestExportArtifact_Bad(t *testing.T) {
	if _, err := ExportArtifact(context.Background(), nil, ArtifactOptions{}); err == nil {
		t.Fatal("nil payload: want error, got nil")
	}

	writer := &fakeArtifactWriter{}
	saveErr := core.NewError("disk full")
	_, err := ExportArtifact(context.Background(), "payload", ArtifactOptions{
		LocalPath: "/tmp/state.kvbin",
		Save:      func(string) error { return saveErr },
		Store:     writer,
	})
	if !core.Is(err, saveErr) {
		t.Fatalf("Save failure error = %v, want %v", err, saveErr)
	}
	if writer.calls != 0 {
		t.Fatalf("Put calls = %d, want 0 (Save failure must short-circuit before Put)", writer.calls)
	}

	// A Put failure (no Save configured) still returns the propagated
	// error after Put was actually attempted.
	putErr := core.NewError("put failed")
	failingWriter := &fakeArtifactWriter{err: putErr}
	_, err = ExportArtifact(context.Background(), "payload", ArtifactOptions{Store: failingWriter})
	if !core.Is(err, putErr) {
		t.Fatalf("Put failure error = %v, want %v", err, putErr)
	}
	if failingWriter.calls != 1 {
		t.Fatalf("Put calls = %d, want 1 (Put itself must be attempted)", failingWriter.calls)
	}
}

// TestExportArtifact_Ugly proves a cancelled context is rejected before any
// work happens, a nil Store still returns a fully populated Artifact with a
// zero ChunkRef, and an unmarshalable payload surfaces a wrapped marshal
// error without reaching the Writer.
func TestExportArtifact_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ExportArtifact(ctx, "payload", ArtifactOptions{}); !core.Is(err, context.Canceled) {
		t.Fatalf("cancelled ctx error = %v, want context.Canceled", err)
	}

	// A nil context is normalised to context.Background() rather than
	// panicking on ctx.Done() — distinct from the cancelled-context case
	// above.
	if _, err := ExportArtifact(nil, "payload", ArtifactOptions{Model: "m"}); err != nil {
		t.Fatalf("nil ctx error = %v, want normalised context", err)
	}

	record, err := ExportArtifact(context.Background(), "payload", ArtifactOptions{Model: "m", Kind: "k"})
	if err != nil {
		t.Fatalf("no-store export error = %v", err)
	}
	if record.ChunkRef != (ChunkRef{}) {
		t.Fatalf("ChunkRef = %#v, want zero value with no Store", record.ChunkRef)
	}
	if record.Model != "m" || record.Kind != "k" {
		t.Fatalf("record = %+v", record)
	}

	writer := &fakeArtifactWriter{}
	_, err = ExportArtifact(context.Background(), func() {}, ArtifactOptions{Store: writer})
	if err == nil {
		t.Fatal("unmarshalable payload: want marshal error, got nil")
	}
	if writer.calls != 0 {
		t.Fatalf("Put calls = %d, want 0 (marshal failure must short-circuit before Put)", writer.calls)
	}
}

// TestArtifactResultError_Good proves a successful Result yields no error,
// and a failed Result carrying an error Value unwraps it verbatim.
func TestArtifactResultError_Good(t *testing.T) {
	if err := artifactResultError(core.Result{OK: true}); err != nil {
		t.Fatalf("artifactResultError(OK) = %v, want nil", err)
	}

	inner := core.NewError("marshal boom")
	if err := artifactResultError(core.Result{OK: false, Value: inner}); !core.Is(err, inner) {
		t.Fatalf("artifactResultError(error value) = %v, want %v", err, inner)
	}
}

// TestArtifactResultError_Bad proves the fallback sentinel is used when a
// failed Result's Value isn't an error at all — a shape the production
// call site in ExportArtifact never actually produces, but the helper
// guards against regardless.
func TestArtifactResultError_Bad(t *testing.T) {
	err := artifactResultError(core.Result{OK: false, Value: "not an error"})
	if !core.Is(err, errArtifactResultFailed) {
		t.Fatalf("artifactResultError(non-error value) = %v, want errArtifactResultFailed", err)
	}
}
