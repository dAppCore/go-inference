// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"math"

	core "dappco.re/go"
	enginegemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/gemma4/audio"
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
)

// gemma4_audio_tower.go is engine/hip's Gemma 4 Conformer audio subsystem: it loads the audio tower +
// mel feature extractor from a safetensors checkpoint and projects a decoded waveform to the tower's
// last_hidden_state plus the soft-token count. The heavy math is the engine-neutral pure-host tower
// (model/gemma4/audio, gated against the HF goldens at cosine >= 0.999); this binds it to hip's audio
// soft-token policy so E2B/E4B audio is reachable through the hip build. hip previously had audio policy
// + labels only and no path to the tower weights (its text load is GGUF); this is that first path.
//
// The tower forward and embed_audio projector are implemented here; inference_model.go binds
// them to the shared audio serving contract and retained HIP prefill.

// AudioTower holds the assembled Gemma 4 audio tower payload and the mel feature extractor, plus the
// mmap the tower weight byte-views reference. Close it once the tower is no longer needed.
type AudioTower struct {
	loaded           *model.LoadedAudio
	extractor        *audio.FeatureExtractor
	projectorWeights []float32
	mapping          *safetensors.DirMapping
	gemm             audio.GEMM
}

// LoadAudioTower assembles the Gemma 4 audio tower + mel extractor from a safetensors model directory.
// Returns (nil, nil) when the checkpoint ships no Conformer audio tower (text-only or non-gemma4), so a
// caller can probe without special-casing. The returned tower owns the checkpoint mmap; Close releases it.
func LoadAudioTower(dir string) (*AudioTower, error) {
	return loadAudioTowerWithGEMM(dir, newSystemHIPAudioGEMM())
}

func loadAudioTowerWithGEMM(dir string, gemm audio.GEMM) (*AudioTower, error) {
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
	projectorWeights, err := hipLoadAudioProjector(loaded.Audio.Projector)
	if err != nil {
		if mapping != nil {
			_ = mapping.Close()
		}
		return nil, core.E("hip.LoadAudioTower", "load embed_audio projector", err)
	}
	return &AudioTower{loaded: loaded.Audio, extractor: extractor, projectorWeights: projectorWeights, mapping: mapping, gemm: gemm}, nil
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
	features, err = audio.EncodeWithGEMM(mel, frames, melBins, a.loaded, nil, a.gemm)
	if err != nil {
		return nil, 0, core.E("hip.AudioTower.Project", "conformer encode", err)
	}
	return features, enginegemma4.AudioSoftTokens(frames), nil
}

// ProjectEmbeddings runs the audio tower and then embed_audio's no-scale RMS normalisation plus its
// dense-BF16 or MLX-affine 4-bit projection. LoadAudioTower widens the projector once into host
// float32; this keeps the binding portable while matching the same affine q*scale+bias convention as
// HIP's q4 kernels.
func (a *AudioTower) ProjectEmbeddings(waveform []float32) (embeddings []float32, softTokens int, err error) {
	features, softTokens, err := a.Project(waveform)
	if err != nil {
		return nil, 0, err
	}
	if len(a.projectorWeights) == 0 {
		return nil, 0, core.NewError("hip.AudioTower.ProjectEmbeddings: projector not loaded")
	}
	embeddings, err = hipAudioProjectorF32(features, softTokens, a.loaded.Projector, a.projectorWeights, 1e-6)
	if err != nil {
		return nil, 0, core.E("hip.AudioTower.ProjectEmbeddings", "embed_audio projection", err)
	}
	return embeddings, softTokens, nil
}

func hipLoadAudioProjector(projector model.LoadedAudioLinear) ([]float32, error) {
	count := projector.OutDim * projector.InDim
	if projector.OutDim <= 0 || projector.InDim <= 0 || len(projector.Weight) == 0 {
		return nil, core.NewError("hip.embed_audio: projector geometry and weight are required")
	}
	if len(projector.Scales) == 0 && len(projector.Biases) == 0 && projector.Kind == "" && projector.Bits == 0 && projector.GroupSize == 0 {
		weights, err := safetensors.DecodeFloat32("BF16", projector.Weight, count)
		if err != nil {
			return nil, core.E("hip.embed_audio", "decode dense BF16 projector", err)
		}
		return weights, nil
	}
	return hipLoadAudioProjectorQ4(projector)
}

func hipLoadAudioProjectorQ4(projector model.LoadedAudioLinear) ([]float32, error) {
	if projector.Kind != mlxaffine.Mode || projector.Bits != 4 || projector.GroupSize <= 0 ||
		len(projector.Weight) == 0 || len(projector.Scales) == 0 || len(projector.Biases) == 0 {
		return nil, core.NewError("hip.embed_audio: complete MLX affine 4-bit projector is required")
	}
	weights, err := mlxaffine.DequantizeTensor(projector.Weight, projector.Scales, projector.Biases,
		projector.OutDim, projector.InDim, projector.Bits, projector.GroupSize)
	if err != nil {
		return nil, core.E("hip.embed_audio", "dequantise projector", err)
	}
	return weights, nil
}

func hipAudioProjectorF32(features []float32, rows int, projector model.LoadedAudioLinear, weights []float32, eps float32) ([]float32, error) {
	if rows <= 0 || projector.InDim <= 0 || projector.OutDim <= 0 || len(features) != rows*projector.InDim ||
		len(weights) != projector.OutDim*projector.InDim {
		return nil, core.NewError("hip.embed_audio: invalid projector geometry")
	}
	if eps < 0 || math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) {
		return nil, core.NewError("hip.embed_audio: epsilon must be non-negative and finite")
	}
	out := make([]float32, rows*projector.OutDim)
	for row := range rows {
		input := features[row*projector.InDim : (row+1)*projector.InDim]
		var squares float32
		for _, value := range input {
			squares += value * value
		}
		invRMS := float32(1 / math.Sqrt(float64(squares/float32(projector.InDim)+eps)))
		for output := range projector.OutDim {
			weight := weights[output*projector.InDim : (output+1)*projector.InDim]
			var sum float32
			for col, value := range input {
				sum += value * invRMS * weight[col]
			}
			out[row*projector.OutDim+output] = sum
		}
	}
	return out, nil
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
