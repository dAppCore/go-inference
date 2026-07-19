// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// TestConfig_ParseConfig_Good parses the unmodified config from openai/whisper-tiny.
// Source: https://huggingface.co/openai/whisper-tiny/resolve/main/config.json
func TestConfig_ParseConfig_Good(t *testing.T) {
	data := core.ReadFile(core.PathJoin("testdata", "openai-whisper-tiny-config.json"))
	if !data.OK {
		t.Fatal("read openai/whisper-tiny fixture")
	}
	cfg, err := ParseConfig(data.Value.([]byte))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if cfg.ModelType != "whisper" || !cfg.IsEncoderDecoder {
		t.Fatalf("parsed whisper identity = model_type %q, is_encoder_decoder %v", cfg.ModelType, cfg.IsEncoderDecoder)
	}
	if cfg.DModel != 384 || cfg.EncoderLayers != 4 || cfg.DecoderLayers != 4 {
		t.Fatalf("parsed whisper-tiny geometry = d_model %d, encoder_layers %d, decoder_layers %d", cfg.DModel, cfg.EncoderLayers, cfg.DecoderLayers)
	}
	if cfg.EncoderAttentionHeads != 6 || cfg.DecoderAttentionHeads != 6 {
		t.Fatalf("parsed whisper-tiny heads = encoder %d, decoder %d", cfg.EncoderAttentionHeads, cfg.DecoderAttentionHeads)
	}
	if cfg.NumMelBins != 80 || cfg.MaxSourcePositions != 1500 || cfg.MaxTargetPositions != 448 {
		t.Fatalf("parsed whisper-tiny audio geometry = mel_bins %d, max_source %d, max_target %d", cfg.NumMelBins, cfg.MaxSourcePositions, cfg.MaxTargetPositions)
	}
	if cfg.VocabSize != 51865 {
		t.Fatalf("parsed whisper-tiny vocab_size = %d, want 51865", cfg.VocabSize)
	}
}

func TestConfig_ParseConfig_Bad(t *testing.T) {
	if _, err := ParseConfig([]byte(`{"model_type":`)); err == nil {
		t.Fatal("ParseConfig accepted malformed JSON")
	}
}

// TestConfig_ParseConfig_Ugly proves ParseConfig never validates geometry —
// a syntactically valid but semantically empty document parses fine.
// Distinct from _Bad's syntax error.
func TestConfig_ParseConfig_Ugly(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{}`))
	if err != nil {
		t.Fatalf("ParseConfig must accept a syntactically valid but semantically empty document: %v", err)
	}
	if cfg.ModelType != "" {
		t.Fatalf("empty document produced a non-empty ModelType: %q", cfg.ModelType)
	}
}

// TestConfig_Arch_Good pins the documented "always refuses" behaviour for a
// realistic, fully-populated config: the refusal echoes the config's ACTUAL
// encoder/decoder/mel counts (proving it doesn't fabricate a generic message).
func TestConfig_Arch_Good(t *testing.T) {
	cfg := Config{DModel: 384, EncoderLayers: 4, DecoderLayers: 4, NumMelBins: 80}
	_, err := cfg.Arch()
	if err == nil {
		t.Fatal("Arch: expected a clean ASR-not-implemented refusal, got a resolved architecture")
	}
	if !core.Contains(err.Error(), "384") || !core.Contains(err.Error(), "80") {
		t.Fatalf("Arch refusal %q must echo the config's actual d_model/mel_bins", err.Error())
	}
}

func TestConfig_Arch_Bad(t *testing.T) {
	_, err := (&Config{}).Arch()
	if err == nil {
		t.Fatal("Arch accepted an empty config")
	}
	if !core.Contains(err.Error(), "Whisper ASR encoder-decoder") {
		t.Fatalf("Arch refusal %q must still explain the arch even for an empty config", err.Error())
	}
}

// TestConfig_Arch_Ugly proves Arch performs NO validation at all — even
// nonsensical negative counts are echoed verbatim in the refusal (there is
// only ever one refusal shape, unconditionally) — distinct from _Bad's
// zero-value case.
func TestConfig_Arch_Ugly(t *testing.T) {
	cfg := Config{DModel: -1, EncoderLayers: -1, DecoderLayers: -1, NumMelBins: -1}
	_, err := cfg.Arch()
	if err == nil || !core.Contains(err.Error(), "-1") {
		t.Fatalf("Arch refusal %v must echo even nonsensical negative counts verbatim", err)
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{DModel: 384}
	cfg.InferFromWeights(nil)
	if cfg.DModel != 384 {
		t.Fatalf("InferFromWeights changed config: %+v", cfg)
	}
}

// TestConfig_InferFromWeights_Bad proves the no-op does not make Arch
// succeed — Arch always refuses regardless.
func TestConfig_InferFromWeights_Bad(t *testing.T) {
	cfg := Config{}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("Arch must still refuse after InferFromWeights")
	}
}

// TestConfig_InferFromWeights_Ugly proves the no-op stays inert even given a
// malformed/weird weights map entry — distinct from _Good's nil-weights case.
func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{DModel: 384}
	cfg.InferFromWeights(map[string]safetensors.Tensor{"weird": {}})
	if cfg.DModel != 384 {
		t.Fatalf("InferFromWeights changed config on a malformed weights map: %+v", cfg)
	}
}

// TestIsWhisperCheckpoint_Good proves the real openai/whisper-tiny config.json fixture (the same one
// TestConfig_ParseConfig_Good parses) is recognised — serve's --model detection routes a directory like
// this one into ASR-only serving (serving.detectAndLoadWhisper).
func TestIsWhisperCheckpoint_Good(t *testing.T) {
	fixture := core.ReadFile(core.PathJoin("testdata", "openai-whisper-tiny-config.json"))
	if !fixture.OK {
		t.Fatal("read openai/whisper-tiny config fixture")
	}
	dir := t.TempDir()
	writeFile(t, dir, "config.json", string(fixture.Value.([]byte)))
	if !IsWhisperCheckpoint(dir) {
		t.Fatal("IsWhisperCheckpoint rejected a real openai/whisper-tiny config.json")
	}
}

// TestIsWhisperCheckpoint_Bad proves a directory with no config.json at all reports false rather than
// erroring — a probe, not a validator (mirrors TestLoad_Bad's directory shape).
func TestIsWhisperCheckpoint_Bad(t *testing.T) {
	if IsWhisperCheckpoint(t.TempDir()) {
		t.Fatal("IsWhisperCheckpoint accepted a directory with no config.json")
	}
}

// TestIsWhisperCheckpoint_Ugly proves a config.json declaring a DIFFERENT model_type reports false —
// distinct from _Bad's missing-file case (mirrors TestLoad_Ugly's non-whisper checkpoint).
func TestIsWhisperCheckpoint_Ugly(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "config.json", `{"model_type":"llama","hidden_size":4096}`)
	if IsWhisperCheckpoint(dir) {
		t.Fatal("IsWhisperCheckpoint accepted a non-whisper checkpoint directory")
	}
}
