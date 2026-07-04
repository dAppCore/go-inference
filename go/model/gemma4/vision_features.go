// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import core "dappco.re/go"

// Gemma4ImageFeatureConfig mirrors the image_processor / video_processor
// sections of processor_config.json. Backends convert this neutral shape into
// their own pixel/patch input representation.
type Gemma4ImageFeatureConfig struct {
	PatchSize         int32   `json:"patch_size"`
	MaxSoftTokens     int32   `json:"max_soft_tokens"`
	PoolingKernelSize int32   `json:"pooling_kernel_size"`
	RescaleFactor     float64 `json:"rescale_factor"`
	DoResize          bool    `json:"do_resize"`
	DoConvertRGB      bool    `json:"do_convert_rgb"`
	NumFrames         int32   `json:"num_frames"`
}

type gemma4VisionProcessorConfig struct {
	ImageProcessor *Gemma4ImageFeatureConfig `json:"image_processor"`
	VideoProcessor *Gemma4ImageFeatureConfig `json:"video_processor"`
}

// LoadGemma4ImageFeatureConfigs reads processor_config.json image/video
// sections. A directory with no processor config returns nil configs and nil
// error, matching the metal loader's text-serving behaviour.
func LoadGemma4ImageFeatureConfigs(modelPath string) (imageCfg, videoCfg *Gemma4ImageFeatureConfig, err error) {
	read := core.ReadFile(core.PathJoin(modelPath, "processor_config.json"))
	if !read.OK {
		return nil, nil, nil
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return nil, nil, core.E("gemma4.vision", "processor_config.json read returned non-byte data", nil)
	}
	var processor gemma4VisionProcessorConfig
	if r := core.JSONUnmarshal(data, &processor); !r.OK {
		return nil, nil, core.E("gemma4.vision", "parse processor_config.json", nil)
	}
	return normalizeGemma4ImageFeatureConfig(processor.ImageProcessor),
		normalizeGemma4ImageFeatureConfig(processor.VideoProcessor), nil
}

func normalizeGemma4ImageFeatureConfig(cfg *Gemma4ImageFeatureConfig) *Gemma4ImageFeatureConfig {
	if cfg == nil {
		return nil
	}
	out := *cfg
	if out.PatchSize <= 0 {
		out.PatchSize = 16
	}
	if out.MaxSoftTokens <= 0 {
		out.MaxSoftTokens = 280
	}
	if out.PoolingKernelSize <= 0 {
		out.PoolingKernelSize = 3
	}
	if out.RescaleFactor <= 0 {
		out.RescaleFactor = 1.0 / 255.0
	}
	return &out
}
