// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import (
	"os"
	"testing"
)

// live_test.go is the #37 slice's live gate against the REAL, locally-cached openai/whisper-tiny
// checkpoint — RUNTIME-gated, reading the checkpoint dir from WHISPER_TINY_DIR (skips cleanly when
// unset) and defaulting to the standard Hugging Face cache location, mirroring gptoss's
// resolveGptOssDir (arch/openai/gptoss/live_test.go): env override, else auto-resolve the hash-named
// snapshot subdirectory. Every other _test.go in this package is hermetic (committed small/synthetic
// fixtures, no external dependency); this file is the ONE that proves the real 39 M-parameter checkpoint
// — real tensor names, real tokenizer, real trained weights — resolves and transcribes correctly.
func resolveWhisperTinyDir(t *testing.T) string {
	t.Helper()
	if dir := os.Getenv("WHISPER_TINY_DIR"); dir != "" {
		return dir
	}
	home := os.Getenv("HOME")
	base := home + "/.cache/huggingface/hub/models--openai--whisper-tiny/snapshots"
	entries, err := os.ReadDir(base)
	if err != nil {
		t.Skipf("openai/whisper-tiny snapshot dir not found (%v) — set WHISPER_TINY_DIR to override, or download it: "+
			"python -c \"from huggingface_hub import snapshot_download; print(snapshot_download('openai/whisper-tiny'))\"", err)
	}
	for _, e := range entries {
		if e.IsDir() {
			return base + "/" + e.Name()
		}
	}
	t.Skip("openai/whisper-tiny snapshots dir has no snapshot subdirectory")
	return ""
}

// TestLive_RealCheckpoint_Load proves Load resolves the REAL checkpoint's actual tensor names/shapes —
// the "never guessed" requirement — not just the hand-built fixtures every other test in this package
// uses. Geometry pinned to the numbers this whole lane was built and verified against.
func TestLive_RealCheckpoint_Load(t *testing.T) {
	dir := resolveWhisperTinyDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	if m.Config.DModel != 384 || m.Config.EncoderLayers != 4 || m.Config.DecoderLayers != 4 || m.Config.VocabSize != 51865 {
		t.Fatalf("real whisper-tiny geometry = d_model %d, encoder_layers %d, decoder_layers %d, vocab %d; want 384/4/4/51865",
			m.Config.DModel, m.Config.EncoderLayers, m.Config.DecoderLayers, m.Config.VocabSize)
	}
	if len(m.Weights.EncoderLayers) != 4 || len(m.Weights.DecoderLayers) != 4 {
		t.Fatalf("real whisper-tiny loaded %d encoder / %d decoder layer weight sets, want 4/4", len(m.Weights.EncoderLayers), len(m.Weights.DecoderLayers))
	}
	if len(m.Weights.EmbedTokens) != 51865*384 {
		t.Fatalf("real whisper-tiny embed_tokens has %d elements, want %d", len(m.Weights.EmbedTokens), 51865*384)
	}
	if len(m.Generation.LangToID) < 90 {
		t.Fatalf("real whisper-tiny generation_config.json lang_to_id has %d entries, want ~99", len(m.Generation.LangToID))
	}
	if en, ok := m.Generation.LanguageTokenID("en"); !ok || en != 50259 {
		t.Fatalf("real whisper-tiny <|en|> token id = %d, ok=%v; want 50259", en, ok)
	}
}

// TestLive_RealCheckpoint_Transcribe is the E2E gate: a committed WAV fixture (real speech, synthesised
// via macOS `say` — "The quick brown fox jumps over the lazy dog.") through the REAL whisper-tiny
// checkpoint, greedy decode, asserting the EXACT reference transcript. Ground truth was captured by
// running the actual reference implementation (transformers.WhisperForConditionalGeneration.generate,
// do_sample=False num_beams=1) on this exact WAV — see testdata/e2e_golden.json's "text" field and this
// package's git history for the capture script. Greedy decode is deterministic, so exact string equality
// is the right assertion (not a similarity/edit-distance heuristic).
func TestLive_RealCheckpoint_Transcribe(t *testing.T) {
	dir := resolveWhisperTinyDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	wav := readTestdata(t, "hello16k.wav")
	golden := readE2EGolden(t)

	result, err := m.Transcribe(wav, Options{})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if result.Text != golden.Text {
		t.Fatalf("Transcribe text = %q, want %q (the reference's exact greedy transcript)", result.Text, golden.Text)
	}
	if result.Language != golden.DetectedLanguageCode() {
		t.Fatalf("Transcribe detected language = %q, want %q", result.Language, golden.DetectedLanguageCode())
	}
}

// TestLive_RealCheckpoint_Transcribe_LanguageOverride proves --language forces the prompt's language
// token instead of auto-detecting — same audio, same content, English forced explicitly rather than
// discovered — and produces the identical transcript (English audio, forced English).
func TestLive_RealCheckpoint_Transcribe_LanguageOverride(t *testing.T) {
	dir := resolveWhisperTinyDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	wav := readTestdata(t, "hello16k.wav")
	golden := readE2EGolden(t)

	result, err := m.Transcribe(wav, Options{Language: "en"})
	if err != nil {
		t.Fatalf("Transcribe with --language en: %v", err)
	}
	if result.Text != golden.Text {
		t.Fatalf("Transcribe(--language en) text = %q, want %q", result.Text, golden.Text)
	}
	if result.Language != "en" {
		t.Fatalf("Transcribe(--language en) echoed language = %q, want \"en\"", result.Language)
	}
}
