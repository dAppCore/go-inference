// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import (
	"testing"

	core "dappco.re/go"
)

// TestConfig_WhisperTiny_Good parses the unmodified config from openai/whisper-tiny.
// Source: https://huggingface.co/openai/whisper-tiny/resolve/main/config.json
func TestConfig_WhisperTiny_Good(t *testing.T) {
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

func TestConfig_Bad(t *testing.T) {
	if _, err := ParseConfig([]byte(`{"model_type":`)); err == nil {
		t.Fatal("ParseConfig accepted malformed JSON")
	}
}
