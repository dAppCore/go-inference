// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	core "dappco.re/go"
	enginegemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/gemma4/audio"
	"dappco.re/go/inference/model/safetensors"
)

// gemma4_audio_tower.go is engine/hip's Gemma 4 Conformer audio subsystem: it loads the audio tower +
// mel feature extractor from a safetensors checkpoint and projects a decoded waveform to the tower's
// last_hidden_state plus the soft-token count. The heavy math is the engine-neutral pure-host tower
// (model/gemma4/audio, gated against the HF goldens at cosine >= 0.999); this binds it to hip's audio
// soft-token policy so E2B/E4B audio is reachable through the hip build. hip previously had audio policy
// + labels only and no path to the tower weights (its text load is GGUF); this is that first path.
//
// Scope (round 1): the tower forward. The embed_audio 4-bit projector (1536 -> LM embed width), the WAV
// bytes-in decode, and the serve token-model seam (implementing engine.AudioInputTokenModel over a hip
// token model, with a combined GGUF-text + safetensors-audio load) are later work — see the port report.

// AudioTower holds the assembled Gemma 4 audio tower payload and the mel feature extractor, plus the
// mmap the tower weight byte-views reference. Close it once the tower is no longer needed.
type AudioTower struct {
	loaded    *model.LoadedAudio
	extractor *audio.FeatureExtractor
	mapping   *safetensors.DirMapping
}

// LoadAudioTower assembles the Gemma 4 audio tower + mel extractor from a safetensors model directory.
// Returns (nil, nil) when the checkpoint ships no Conformer audio tower (text-only or non-gemma4), so a
// caller can probe without special-casing. The returned tower owns the checkpoint mmap; Close releases it.
func LoadAudioTower(dir string) (*AudioTower, error) {
	loaded, mapping, err := model.Load(dir)
	if err != nil {
		return nil, core.E("hip.LoadAudioTower", "load model", err)
	}
	if loaded == nil || loaded.Audio == nil || len(loaded.Audio.Layers) == 0 {
		if mapping != nil {
			_ = mapping.Close()
		}
		return nil, nil
	}
	cfg, err := audio.LoadFeatureConfig(dir)
	if err != nil {
		if mapping != nil {
			_ = mapping.Close()
		}
		return nil, core.E("hip.LoadAudioTower", "load audio feature config", err)
	}
	if cfg == nil {
		if mapping != nil {
			_ = mapping.Close()
		}
		return nil, core.NewError("hip.LoadAudioTower: audio tower present but processor_config.json has no feature_extractor")
	}
	extractor, err := audio.NewFeatureExtractor(cfg)
	if err != nil {
		if mapping != nil {
			_ = mapping.Close()
		}
		return nil, core.E("hip.LoadAudioTower", "build mel feature extractor", err)
	}
	return &AudioTower{loaded: loaded.Audio, extractor: extractor, mapping: mapping}, nil
}

// Project runs one decoded waveform (16 kHz mono float32, [-1,1]) through the tower: mel extract →
// Conformer encode → last_hidden_state [softTokens, OutputDim] float32, plus the gemma4 soft-token count
// (frames stride-subsampled twice: ceil(frames/4)). The features are the tower output before the
// embed_audio projection (the gated boundary this round).
func (a *AudioTower) Project(waveform []float32) (features []float32, softTokens int, err error) {
	if a == nil || a.loaded == nil || a.extractor == nil {
		return nil, 0, core.NewError("hip.AudioTower.Project: tower not loaded")
	}
	mel, frames, melBins, err := audio.InputFeatures(waveform, a.extractor)
	if err != nil {
		return nil, 0, core.E("hip.AudioTower.Project", "mel feature extraction", err)
	}
	features, err = audio.Encode(mel, frames, melBins, a.loaded)
	if err != nil {
		return nil, 0, core.E("hip.AudioTower.Project", "conformer encode", err)
	}
	return features, enginegemma4.AudioSoftTokens(frames), nil
}

// OutputDim reports the tower's last_hidden_state width (audio output_proj_dims).
func (a *AudioTower) OutputDim() int {
	if a == nil || a.loaded == nil {
		return 0
	}
	return a.loaded.Cfg.OutputDim
}

// Close releases the checkpoint mmap backing the tower weights.
func (a *AudioTower) Close() error {
	if a == nil || a.mapping == nil {
		return nil
	}
	return a.mapping.Close()
}
