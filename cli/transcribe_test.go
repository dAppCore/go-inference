// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"path/filepath"
	"testing"

	core "dappco.re/go"
)

// TestRunTranscribeCommand_Rejects covers the flag/argument guards that return before whisper.Load is
// ever reached: a missing --model, the wrong positional count, and an unreadable audio path. A bogus
// --model value is safe in every case except "unreadable audio" (which needs a real --model string to
// reach the audio-read step; whisper.Load itself is exercised at the library level —
// model/arch/openai/whisper/live_test.go — not re-tested here).
func TestRunTranscribeCommand_Rejects(t *testing.T) {
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
	}{
		{"no --model", []string{"clip.wav"}, 2},
		{"no positional", []string{"--model", "some-dir"}, 2},
		{"two positionals", []string{"--model", "some-dir", "a.wav", "b.wav"}, 2},
		{"unreadable audio", []string{"--model", "some-dir", filepath.Join(t.TempDir(), "nope.wav")}, 1},
		{"help", []string{"--help"}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runTranscribeCommand(context.Background(), tc.args, &stdout, &stderr); code != tc.wantCode {
				t.Fatalf("exit %d, want %d; stderr=%s", code, tc.wantCode, stderr.String())
			}
		})
	}
}

// TestRunTranscribeCommand_BadModelDir proves a --model directory that resolves (unlike "unreadable
// audio" above) but fails to load — no config.json — reports the library's error and exits 1, not 2 (a
// load failure is a runtime error, not a usage error).
func TestRunTranscribeCommand_BadModelDir(t *testing.T) {
	audio := filepath.Join(t.TempDir(), "clip.wav")
	if r := core.WriteFile(audio, []byte("not really a wav"), 0o644); !r.OK {
		t.Fatalf("write fixture audio: %v", r.Err())
	}
	var stdout, stderr bytes.Buffer
	code := runTranscribeCommand(context.Background(), []string{"--model", t.TempDir(), audio}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("exit %d, want 1; stderr=%s", code, stderr.String())
	}
}
