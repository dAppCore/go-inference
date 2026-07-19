// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	core "dappco.re/go"
)

// writeWhisperishConfig writes a minimal config.json declaring model_type "whisper" — enough for
// whisper.IsWhisperCheckpoint to say yes, not enough for whisper.Load to succeed (no generation_config/
// preprocessor_config/tokenizer/safetensors) — the shape detectAndLoadWhisper's fail-closed tests need.
func writeWhisperishConfig(t *testing.T, dir string) {
	t.Helper()
	data := []byte(`{"model_type":"whisper","is_encoder_decoder":true,"d_model":384}`)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), data, 0o600); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
}

// TestDetectAndLoadWhisper_Good proves a non-Whisper directory falls straight through unchanged: not an
// error, not flagged as Whisper — the ordinary chat-model path proceeds exactly as it did before this
// detection existed.
func TestDetectAndLoadWhisper_Good(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"qwen3"}`), 0o600); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	transcriber, isWhisper, err := detectAndLoadWhisper(dir)
	if err != nil || isWhisper || transcriber != nil {
		t.Fatalf("detectAndLoadWhisper(non-whisper dir) = (%v, %v, %v), want (nil, false, nil)", transcriber, isWhisper, err)
	}
}

// TestDetectAndLoadWhisper_Bad proves a genuine Whisper config.json but an otherwise incomplete
// checkpoint fails CLOSED — isWhisper true AND a non-nil error — matching the outbound-policy/embed-model
// "a deployer who asked for X gets X or an honest refusal, never a silent wrong path" precedent
// (loadOutboundPolicy, loadEmbedModel).
func TestDetectAndLoadWhisper_Bad(t *testing.T) {
	dir := t.TempDir()
	writeWhisperishConfig(t, dir)
	transcriber, isWhisper, err := detectAndLoadWhisper(dir)
	if err == nil || !isWhisper || transcriber != nil {
		t.Fatalf("detectAndLoadWhisper(incomplete whisper dir) = (%v, %v, %v), want (nil, true, a load error)", transcriber, isWhisper, err)
	}
}

// TestDetectAndLoadWhisper_Ugly proves an empty modelPath ("" — RunServe's model-less start) is treated
// the same as "not Whisper" rather than erroring — distinct from _Good's populated-but-wrong-type
// directory.
func TestDetectAndLoadWhisper_Ugly(t *testing.T) {
	transcriber, isWhisper, err := detectAndLoadWhisper("")
	if err != nil || isWhisper || transcriber != nil {
		t.Fatalf("detectAndLoadWhisper(\"\") = (%v, %v, %v), want (nil, false, nil)", transcriber, isWhisper, err)
	}
}

// TestRunServe_Whisper_Bad_LoadFailureFailsClosed mirrors TestRunServe_EmbedModel_Bad_LoadFailureFailsClosed
// exactly: --model pointing at an incomplete Whisper checkpoint fails RunServe closed BEFORE any listener
// binds (detectAndLoadWhisper runs ahead of everything else in RunServe — serve.go) — no freeListenAddr
// needed, this never reaches Addr.
func TestRunServe_Whisper_Bad_LoadFailureFailsClosed(t *testing.T) {
	dir := t.TempDir()
	writeWhisperishConfig(t, dir)
	err := RunServe(context.Background(), ServeConfig{
		Addr:      "", // never reached — detectAndLoadWhisper fails before RunServe touches Addr
		Log:       core.NewBuffer(),
		ModelPath: dir,
	})
	if err == nil {
		t.Fatal("RunServe with an incomplete Whisper --model returned nil, want a fail-closed error")
	}
}
