// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import core "dappco.re/go"

func ExampleParseConfig() {
	cfg, err := ParseConfig([]byte(`{"model_type":"whisper","is_encoder_decoder":true,"d_model":384,"encoder_layers":4,"decoder_layers":4}`))
	core.Println(err == nil, cfg.ModelType, cfg.DModel)
	// Output: true whisper 384
}

func ExampleConfig_Arch() {
	cfg := Config{DModel: 384, EncoderLayers: 4, DecoderLayers: 4, NumMelBins: 80}
	_, err := cfg.Arch()
	core.Println(err != nil)
	// Output: true
}

func ExampleConfig_InferFromWeights() {
	cfg := Config{DModel: 384}
	cfg.InferFromWeights(nil) // no-op: Whisper declares every encoder/decoder dim in config.json
	core.Println(cfg.DModel)
	// Output: 384
}

// ExampleIsWhisperCheckpoint shows the probe serve's --model detection uses: no config.json here (this
// package's testdata/ deliberately carries no bare "config.json" — see config_test.go's fixture naming),
// so this predictably reports false without needing a real checkpoint on disk.
func ExampleIsWhisperCheckpoint() {
	core.Println(IsWhisperCheckpoint("testdata"))
	// Output: false
}
