// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
)

// ImageFeatureConfig mirrors Gemma-4's image_processor / video_processor
// front-end config from processor_config.json.
type ImageFeatureConfig struct {
	PatchSize         int     `json:"patch_size"`
	MaxSoftTokens     int     `json:"max_soft_tokens"`
	PoolingKernelSize int     `json:"pooling_kernel_size"`
	RescaleFactor     float64 `json:"rescale_factor"`
	DoResize          bool    `json:"do_resize"`
	DoConvertRGB      bool    `json:"do_convert_rgb"`
	NumFrames         int     `json:"num_frames"`
}

type ImageFeatureGeometry struct {
	SourceHeight int
	SourceWidth  int
	TargetHeight int
	TargetWidth  int
	PatchGrid    int
	SoftTokens   int
}

// AudioFeatureConfig mirrors Gemma-4's feature_extractor front-end config from
// processor_config.json.
type AudioFeatureConfig struct {
	FeatureSize      int       `json:"feature_size"`
	SamplingRate     int       `json:"sampling_rate"`
	FrameLength      int       `json:"frame_length"`
	HopLength        int       `json:"hop_length"`
	FFTLength        int       `json:"fft_length"`
	NumMelFilters    int       `json:"num_mel_filters"`
	FrameLengthMs    float64   `json:"frame_length_ms"`
	HopLengthMs      float64   `json:"hop_length_ms"`
	FFTOverdrive     bool      `json:"fft_overdrive"`
	MinFrequency     float64   `json:"min_frequency"`
	MaxFrequency     float64   `json:"max_frequency"`
	MelFloor         float64   `json:"mel_floor"`
	Preemphasis      float64   `json:"preemphasis"`
	PreemphasisHTK   bool      `json:"preemphasis_htk_flavor"`
	Dither           float64   `json:"dither"`
	InputScaleFactor float64   `json:"input_scale_factor"`
	PaddingValue     float64   `json:"padding_value"`
	PerBinMean       []float64 `json:"per_bin_mean"`
	PerBinStddev     []float64 `json:"per_bin_stddev"`
	MaxLengthSamples int       `json:"-"`
	PadToMultiple    int       `json:"-"`
	FeatureExtractor string    `json:"feature_extractor_type"`
}

type AudioFeaturePlan struct {
	Config           AudioFeatureConfig
	FFTLength        int
	MaxLengthSamples int
	PadToMultiple    int
}

type ProcessorConfig struct {
	AudioMsPerToken  int                 `json:"audio_ms_per_token"`
	AudioSeqLength   int                 `json:"audio_seq_length"`
	ImageProcessor   *ImageFeatureConfig `json:"image_processor"`
	VideoProcessor   *ImageFeatureConfig `json:"video_processor"`
	FeatureExtractor *AudioFeatureConfig `json:"feature_extractor"`
}

func ParseProcessorConfig(data []byte) (ProcessorConfig, error) {
	var cfg ProcessorConfig
	if len(data) == 0 {
		return cfg, fmt.Errorf("processor_config.json is empty")
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, err
	}
	if cfg.ImageProcessor != nil {
		resolved := NormalizeImageFeatureConfig(*cfg.ImageProcessor)
		cfg.ImageProcessor = &resolved
	}
	if cfg.VideoProcessor != nil {
		resolved := NormalizeImageFeatureConfig(*cfg.VideoProcessor)
		cfg.VideoProcessor = &resolved
	}
	if cfg.FeatureExtractor != nil {
		resolved := NormalizeAudioFeatureConfig(*cfg.FeatureExtractor)
		cfg.FeatureExtractor = &resolved
	}
	return cfg, nil
}

func NormalizeImageFeatureConfig(cfg ImageFeatureConfig) ImageFeatureConfig {
	if cfg.PatchSize <= 0 {
		cfg.PatchSize = 16
	}
	if cfg.MaxSoftTokens <= 0 {
		cfg.MaxSoftTokens = 280
	}
	if cfg.PoolingKernelSize <= 0 {
		cfg.PoolingKernelSize = 3
	}
	if cfg.RescaleFactor <= 0 {
		cfg.RescaleFactor = 1.0 / 255.0
	}
	return cfg
}

func ImageFeatureGeometryOf(sourceHeight, sourceWidth int, cfg ImageFeatureConfig) (ImageFeatureGeometry, error) {
	cfg = NormalizeImageFeatureConfig(cfg)
	if sourceHeight <= 0 || sourceWidth <= 0 {
		return ImageFeatureGeometry{}, fmt.Errorf("invalid image size %dx%d", sourceHeight, sourceWidth)
	}
	maxPatches := cfg.MaxSoftTokens * cfg.PoolingKernelSize * cfg.PoolingKernelSize
	targetHeight := sourceHeight
	targetWidth := sourceWidth
	sideMultiple := cfg.PatchSize * cfg.PoolingKernelSize
	if cfg.DoResize || targetHeight%sideMultiple != 0 || targetWidth%sideMultiple != 0 {
		var err error
		targetHeight, targetWidth, err = AspectPreservingImageSize(sourceHeight, sourceWidth, cfg.PatchSize, maxPatches, cfg.PoolingKernelSize)
		if err != nil {
			return ImageFeatureGeometry{}, err
		}
	}
	patchGrid := (targetHeight / cfg.PatchSize) * (targetWidth / cfg.PatchSize)
	softTokens := patchGrid / (cfg.PoolingKernelSize * cfg.PoolingKernelSize)
	return ImageFeatureGeometry{
		SourceHeight: sourceHeight,
		SourceWidth:  sourceWidth,
		TargetHeight: targetHeight,
		TargetWidth:  targetWidth,
		PatchGrid:    patchGrid,
		SoftTokens:   softTokens,
	}, nil
}

func AspectPreservingImageSize(height, width, patchSize, maxPatches, pool int) (int, int, error) {
	if height <= 0 || width <= 0 {
		return 0, 0, fmt.Errorf("invalid image size %dx%d", height, width)
	}
	if patchSize <= 0 || maxPatches <= 0 || pool <= 0 {
		return 0, 0, fmt.Errorf("invalid patch budget patch=%d max=%d pool=%d", patchSize, maxPatches, pool)
	}
	targetPx := float64(maxPatches) * float64(patchSize) * float64(patchSize)
	factor := math.Sqrt(targetPx / (float64(height) * float64(width)))
	sideMultiple := pool * patchSize

	targetHeight := int(math.Floor(factor*float64(height)/float64(sideMultiple))) * sideMultiple
	targetWidth := int(math.Floor(factor*float64(width)/float64(sideMultiple))) * sideMultiple

	if targetHeight == 0 && targetWidth == 0 {
		return 0, 0, fmt.Errorf("image degenerates to 0x0 under the patch budget")
	}
	maxSide := (maxPatches / (pool * pool)) * sideMultiple
	if targetHeight == 0 {
		targetHeight = sideMultiple
		targetWidth = minInt(int(math.Floor(float64(width)/float64(height)))*sideMultiple, maxSide)
	} else if targetWidth == 0 {
		targetWidth = sideMultiple
		targetHeight = minInt(int(math.Floor(float64(height)/float64(width)))*sideMultiple, maxSide)
	}
	if int64(targetHeight)*int64(targetWidth) > int64(targetPx) {
		return 0, 0, fmt.Errorf("target %dx%d exceeds the %d-patch budget", targetHeight, targetWidth, maxPatches)
	}
	return targetHeight, targetWidth, nil
}

func NormalizeAudioFeatureConfig(cfg AudioFeatureConfig) AudioFeatureConfig {
	if cfg.FeatureSize <= 0 && cfg.NumMelFilters > 0 {
		cfg.FeatureSize = cfg.NumMelFilters
	}
	if cfg.FeatureSize <= 0 {
		cfg.FeatureSize = 128
	}
	if cfg.SamplingRate <= 0 {
		cfg.SamplingRate = 16000
	}
	msToSamples := func(ms float64) int {
		return int(math.Round(float64(cfg.SamplingRate) * ms / 1000.0))
	}
	if cfg.FrameLength <= 0 && cfg.FrameLengthMs > 0 {
		cfg.FrameLength = msToSamples(cfg.FrameLengthMs)
	}
	if cfg.FrameLength <= 0 {
		cfg.FrameLength = msToSamples(20.0)
	}
	if cfg.HopLength <= 0 && cfg.HopLengthMs > 0 {
		cfg.HopLength = msToSamples(cfg.HopLengthMs)
	}
	if cfg.HopLength <= 0 {
		cfg.HopLength = msToSamples(10.0)
	}
	if cfg.MaxFrequency <= 0 {
		cfg.MaxFrequency = 8000.0
	}
	if cfg.MelFloor <= 0 {
		cfg.MelFloor = 1e-3
	}
	if cfg.InputScaleFactor == 0 {
		cfg.InputScaleFactor = 1
	}
	return cfg
}

func AudioFeaturePlanOf(cfg AudioFeatureConfig) (AudioFeaturePlan, error) {
	cfg = NormalizeAudioFeatureConfig(cfg)
	fftLength := cfg.FFTLength
	if fftLength <= 0 {
		fftLength = 1 << int(math.Ceil(math.Log2(float64(cfg.FrameLength))))
		if cfg.FFTOverdrive {
			fftLength *= 2
		}
	}
	if fftLength&(fftLength-1) != 0 || fftLength < cfg.FrameLength {
		return AudioFeaturePlan{}, fmt.Errorf("fft_length %d must be a power of two >= frame_length %d", fftLength, cfg.FrameLength)
	}
	if cfg.MaxFrequency <= cfg.MinFrequency {
		return AudioFeaturePlan{}, fmt.Errorf("mel band [%v, %v] is empty", cfg.MinFrequency, cfg.MaxFrequency)
	}
	cfg.FFTLength = fftLength
	maxSamples := cfg.MaxLengthSamples
	if maxSamples <= 0 {
		maxSamples = 480000
	}
	padMultiple := cfg.PadToMultiple
	if padMultiple <= 0 {
		padMultiple = 128
	}
	return AudioFeaturePlan{
		Config:           cfg,
		FFTLength:        fftLength,
		MaxLengthSamples: maxSamples,
		PadToMultiple:    padMultiple,
	}, nil
}

func AudioFrameCount(sampleCount int, plan AudioFeaturePlan) (int, error) {
	if sampleCount <= 0 {
		return 0, fmt.Errorf("empty waveform")
	}
	if plan.Config.FrameLength <= 0 || plan.Config.HopLength <= 0 {
		return 0, fmt.Errorf("audio feature plan is not resolved")
	}
	if sampleCount > plan.MaxLengthSamples {
		sampleCount = plan.MaxLengthSamples
	}
	padded := sampleCount
	if rem := padded % plan.PadToMultiple; rem != 0 {
		padded += plan.PadToMultiple - rem
	}
	waveLen := plan.Config.FrameLength/2 + padded
	frameSize := plan.Config.FrameLength + 1
	if waveLen-frameSize < 0 {
		return 0, fmt.Errorf("waveform too short: %d samples < frame %d", sampleCount, frameSize)
	}
	return (waveLen-frameSize)/plan.Config.HopLength + 1, nil
}

func AudioSoftTokens(melFrames int) int {
	if melFrames <= 0 {
		return 0
	}
	half := func(n int) int { return (n + 1) / 2 }
	return half(half(melFrames))
}

func ApplyProcessorConfigLabels(labels map[string]string, cfg ProcessorConfig) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	setPositiveIntLabel(labels, "processor_audio_ms_per_token", cfg.AudioMsPerToken)
	setPositiveIntLabel(labels, "processor_audio_seq_length", cfg.AudioSeqLength)
	if cfg.ImageProcessor != nil {
		labels["image_processor"] = "true"
		applyImageProcessorLabels(labels, "image_processor", *cfg.ImageProcessor)
	}
	if cfg.VideoProcessor != nil {
		labels["video_processor"] = "true"
		applyImageProcessorLabels(labels, "video_processor", *cfg.VideoProcessor)
	}
	if cfg.FeatureExtractor != nil {
		labels["audio_feature_extractor"] = "true"
		applyAudioFeatureLabels(labels, *cfg.FeatureExtractor)
	}
	return labels
}

func applyImageProcessorLabels(labels map[string]string, prefix string, cfg ImageFeatureConfig) {
	cfg = NormalizeImageFeatureConfig(cfg)
	setPositiveIntLabel(labels, prefix+"_patch_size", cfg.PatchSize)
	setPositiveIntLabel(labels, prefix+"_max_soft_tokens", cfg.MaxSoftTokens)
	setPositiveIntLabel(labels, prefix+"_pooling_kernel_size", cfg.PoolingKernelSize)
	setPositiveFloatLabel(labels, prefix+"_rescale_factor", cfg.RescaleFactor)
	labels[prefix+"_do_resize"] = strconv.FormatBool(cfg.DoResize)
	labels[prefix+"_do_convert_rgb"] = strconv.FormatBool(cfg.DoConvertRGB)
	setPositiveIntLabel(labels, prefix+"_num_frames", cfg.NumFrames)
}

func applyAudioFeatureLabels(labels map[string]string, cfg AudioFeatureConfig) {
	plan, err := AudioFeaturePlanOf(cfg)
	if err == nil {
		cfg = plan.Config
		setPositiveIntLabel(labels, "audio_feature_fft_length", plan.FFTLength)
		setPositiveIntLabel(labels, "audio_feature_max_length_samples", plan.MaxLengthSamples)
		setPositiveIntLabel(labels, "audio_feature_pad_to_multiple", plan.PadToMultiple)
	} else {
		cfg = NormalizeAudioFeatureConfig(cfg)
	}
	setPositiveIntLabel(labels, "audio_feature_size", cfg.FeatureSize)
	setPositiveIntLabel(labels, "audio_feature_sampling_rate", cfg.SamplingRate)
	setPositiveIntLabel(labels, "audio_feature_frame_length", cfg.FrameLength)
	setPositiveIntLabel(labels, "audio_feature_hop_length", cfg.HopLength)
	setPositiveFloatLabel(labels, "audio_feature_min_frequency", cfg.MinFrequency)
	setPositiveFloatLabel(labels, "audio_feature_max_frequency", cfg.MaxFrequency)
	setPositiveFloatLabel(labels, "audio_feature_mel_floor", cfg.MelFloor)
	setPositiveFloatLabel(labels, "audio_feature_input_scale_factor", cfg.InputScaleFactor)
	if cfg.FeatureExtractor != "" {
		labels["audio_feature_extractor_type"] = normalizeConfigLabelToken(cfg.FeatureExtractor)
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
