// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model/artifact"
	"dappco.re/go/inference/model/state/session/internal/sessionfake"
)

// TestArtifact_ExportArtifacts_Good captures the retained KV state and
// exports it as an in-memory artifact record (empty Options.KVPath keeps
// the export off disk).
func TestArtifact_ExportArtifacts_Good(t *testing.T) {
	native := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}
	session := &Session{session: native}

	record, err := session.ExportArtifacts(artifact.Options{
		Model:  "gemma4-1b",
		Prompt: "stable context",
	})

	if err != nil {
		t.Fatalf("ExportArtifacts() error = %v", err)
	}
	if record == nil {
		t.Fatal("ExportArtifacts() record = nil")
	}
	if record.Model != "gemma4-1b" || record.Prompt != "stable context" {
		t.Fatalf("record identity = %q/%q, want gemma4-1b/stable context", record.Model, record.Prompt)
	}
	if record.Snapshot.Architecture != "gemma4_text" || record.Snapshot.TokenCount != 2 {
		t.Fatalf("record snapshot = %+v, want gemma4_text/2 tokens", record.Snapshot)
	}
	if record.Analysis == nil || len(record.Features) == 0 {
		t.Fatalf("record analysis/features = %+v/%v, want populated", record.Analysis, record.Features)
	}
	if record.SAMI.Architecture != "gemma4_text" || record.SAMI.Model != "gemma4-1b" {
		t.Fatalf("record SAMI = %+v, want gemma4_text/gemma4-1b", record.SAMI)
	}
}

// TestArtifact_ExportArtifacts_WritesKVPath_Good exercises the KVPath
// branch — the capture is additionally persisted to disk and the path is
// echoed back on the record.
func TestArtifact_ExportArtifacts_WritesKVPath_Good(t *testing.T) {
	native := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}
	session := &Session{session: native}
	path := core.PathJoin(t.TempDir(), "artifact.kvbin")

	record, err := session.ExportArtifacts(artifact.Options{Model: "gemma4-1b", KVPath: path})

	if err != nil {
		t.Fatalf("ExportArtifacts() error = %v", err)
	}
	if record.KVPath != path {
		t.Fatalf("record KVPath = %q, want %q", record.KVPath, path)
	}
	// Verify the written artifact is a real, loadable KV snapshot rather
	// than just trusting the echoed path — mirrors how the session KV
	// round-trip tests confirm SaveKV output.
	loaded, err := kv.Load(path)
	if err != nil {
		t.Fatalf("kv.Load(%q) error = %v", path, err)
	}
	if loaded.Architecture != "gemma4_text" || loaded.SeqLen != 2 {
		t.Fatalf("written snapshot = %+v, want gemma4_text/2", loaded)
	}
}

// TestArtifact_ExportArtifacts_Bad — a nil session cannot capture state,
// so the export fails before it reaches the artifact package.
func TestArtifact_ExportArtifacts_Bad(t *testing.T) {
	var session *Session

	record, err := session.ExportArtifacts(artifact.Options{Model: "gemma4-1b"})

	if err == nil {
		t.Fatal("ExportArtifacts(nil) error = nil, want nil-session error")
	}
	if record != nil {
		t.Fatalf("ExportArtifacts(nil) record = %+v, want nil", record)
	}
}

// TestArtifact_ExportArtifacts_Ugly — the native capture fails, and the
// error propagates out of ExportArtifacts untouched.
func TestArtifact_ExportArtifacts_Ugly(t *testing.T) {
	wantErr := core.NewError("capture exploded")
	session := &Session{session: &sessionfake.Handle{
		KV:         sessionfake.TestKVSnapshot(),
		CaptureErr: wantErr,
	}}

	record, err := session.ExportArtifacts(artifact.Options{Model: "gemma4-1b"})

	if !core.Is(err, wantErr) {
		t.Fatalf("ExportArtifacts() error = %v, want %v", err, wantErr)
	}
	if record != nil {
		t.Fatalf("ExportArtifacts() record = %+v, want nil", record)
	}
}
