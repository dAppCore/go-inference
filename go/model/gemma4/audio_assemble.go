// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	"dappco.re/go/inference/profile"
)

// audio_assemble.go is the pure-Go sibling of pkg/metal/model/gemma4/audio_encoder_load.go. It
// canonicalises and gathers Gemma 4 audio-tower bytes into model.LoadedAudio so no-cgo backends can
// build the Conformer path without importing pkg/metal.

const (
	Gemma4BOAToken   = "<|audio>"
	Gemma4AudioToken = "<|audio|>"
	Gemma4EOAToken   = "<audio|>"
)

func canonicalGemma4AudioWeightName(name string) (string, bool) {
	trimmed := name
	for {
		next, changed := profile.TrimWeightWrapperPrefix("gemma4", trimmed)
		if !changed {
			break
		}
		trimmed = next
	}
	if core.HasPrefix(trimmed, "embed_audio.") || core.HasPrefix(trimmed, "audio_tower.") {
		return trimmed, true
	}
	return "", false
}

// SanitizeAudioWeights returns the Gemma 4 audio weights keyed by their canonical names.
func SanitizeAudioWeights(raw map[string]safetensors.Tensor) map[string]safetensors.Tensor {
	audio := make(map[string]safetensors.Tensor)
	for name, t := range raw {
		if canonical, ok := canonicalGemma4AudioWeightName(name); ok {
			audio[canonical] = t
		}
	}
	return audio
}

func HasAudioTowerWeights(weights map[string]safetensors.Tensor) bool {
	_, ok := model.WeightAny(weights, "audio_tower.subsample_conv_projection.input_proj_linear.weight")
	return ok
}

func HasAudioProjectionWeights(weights map[string]safetensors.Tensor) bool {
	_, ok := model.WeightAny(weights,
		"embed_audio.embedding_projection.weight",
		"embed_audio.embedding_projection.linear.weight",
	)
	return ok
}

func audioWeight(weights map[string]safetensors.Tensor, names ...string) []byte {
	if t, ok := model.WeightAny(weights, names...); ok {
		return t.Data
	}
	return nil
}

func audioTensor(weights map[string]safetensors.Tensor, names ...string) (safetensors.Tensor, bool) {
	return model.WeightAny(weights, names...)
}

func audioLinear(weights map[string]safetensors.Tensor, prefixes ...string) model.LoadedAudioLinear {
	for _, p := range prefixes {
		if w := audioWeight(weights, p+".weight", p+".linear.weight"); w != nil {
			return model.LoadedAudioLinear{Weight: w, Clip: audioClipPair(weights, p)}
		}
	}
	return model.LoadedAudioLinear{}
}

func audioLinearWithInputDim(weights map[string]safetensors.Tensor, inDim int, prefixes ...string) model.LoadedAudioLinear {
	for _, p := range prefixes {
		for _, candidate := range []string{p, p + ".linear"} {
			lin := model.LoadLinear(weights, candidate, inDim, "affine")
			if lin == nil {
				continue
			}
			return model.LoadedAudioLinear{
				Weight:    lin.Weight,
				Scales:    lin.Scales,
				Biases:    lin.Biases,
				Clip:      audioClipPair(weights, p),
				OutDim:    lin.OutDim,
				InDim:     lin.InDim,
				GroupSize: lin.GroupSize,
				Bits:      lin.Bits,
				Kind:      lin.Kind,
			}
		}
	}
	return model.LoadedAudioLinear{}
}

func audioClippable(weights map[string]safetensors.Tensor, prefix string) model.LoadedAudioLinear {
	lin := audioLinear(weights, prefix+".linear")
	if lin.Weight == nil {
		return model.LoadedAudioLinear{}
	}
	lin.Clip = audioClipPair(weights, prefix)
	return lin
}

func audioClipPair(weights map[string]safetensors.Tensor, prefix string) model.LoadedAudioClipPair {
	return model.LoadedAudioClipPair{
		In:  audioClipBound(weights, prefix+".input_min", prefix+".input_max"),
		Out: audioClipBound(weights, prefix+".output_min", prefix+".output_max"),
	}
}

func audioClipBound(weights map[string]safetensors.Tensor, minName, maxName string) model.LoadedAudioClipBound {
	minT, okMin := weights[minName]
	maxT, okMax := weights[maxName]
	if !okMin || !okMax {
		return model.LoadedAudioClipBound{}
	}
	minV, okMin := audioScalar(minT)
	maxV, okMax := audioScalar(maxT)
	if !okMin || !okMax {
		return model.LoadedAudioClipBound{}
	}
	return model.LoadedAudioClipBound{Min: minV, Max: maxV, Present: true}
}

func audioScalar(t safetensors.Tensor) (float32, bool) {
	vals, ok := audioF32Values(t)
	if !ok || len(vals) == 0 {
		return 0, false
	}
	return vals[0], true
}

func audioF32Values(t safetensors.Tensor) ([]float32, bool) {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	switch t.Dtype {
	case "BF16":
		if len(t.Data) < n*2 {
			return nil, false
		}
		out := make([]float32, n)
		for i := range out {
			b := uint32(t.Data[i*2+1])<<8 | uint32(t.Data[i*2])
			out[i] = math.Float32frombits(b << 16)
		}
		return out, true
	case "F32":
		if len(t.Data) < n*4 {
			return nil, false
		}
		out := make([]float32, n)
		for i := range out {
			bits := uint32(t.Data[i*4]) |
				uint32(t.Data[i*4+1])<<8 |
				uint32(t.Data[i*4+2])<<16 |
				uint32(t.Data[i*4+3])<<24
			out[i] = math.Float32frombits(bits)
		}
		return out, true
	default:
		return nil, false
	}
}

func audioZerosBF16(n int) []byte {
	return make([]byte, n*2)
}

func audioConv2dToOHWI(t safetensors.Tensor) ([]byte, error) {
	if len(t.Shape) != 4 {
		return nil, core.NewError("gemma4.AssembleAudio: conv2d weight must be rank 4")
	}
	outC, inC, kh, kw := t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3]
	if len(t.Data) != outC*inC*kh*kw*2 {
		return nil, core.NewError("gemma4.AssembleAudio: conv2d byte length mismatch")
	}
	out := make([]byte, len(t.Data))
	for oc := 0; oc < outC; oc++ {
		for ic := 0; ic < inC; ic++ {
			for y := 0; y < kh; y++ {
				for x := 0; x < kw; x++ {
					src := (((oc*inC+ic)*kh+y)*kw + x) * 2
					dst := (((oc*kh+y)*kw+x)*inC + ic) * 2
					copy(out[dst:dst+2], t.Data[src:src+2])
				}
			}
		}
	}
	return out, nil
}

func audioDepthwiseToNLC(t safetensors.Tensor) ([]byte, error) {
	if len(t.Shape) != 3 {
		return nil, core.NewError("gemma4.AssembleAudio: depthwise conv1d weight must be rank 3")
	}
	ch := t.Shape[0]
	if len(t.Data) != t.Shape[0]*t.Shape[1]*t.Shape[2]*2 {
		return nil, core.NewError("gemma4.AssembleAudio: depthwise conv1d byte length mismatch")
	}
	switch {
	case t.Shape[1] == 1:
		k := t.Shape[2]
		out := make([]byte, ch*k*2)
		for c := 0; c < ch; c++ {
			for i := 0; i < k; i++ {
				src := ((c*t.Shape[1]+0)*k + i) * 2
				dst := (c*k + i) * 2
				copy(out[dst:dst+2], t.Data[src:src+2])
			}
		}
		return out, nil
	case t.Shape[2] == 1:
		k := t.Shape[1]
		out := make([]byte, ch*k*2)
		for c := 0; c < ch; c++ {
			for i := 0; i < k; i++ {
				src := ((c*k+i)*t.Shape[2] + 0) * 2
				dst := (c*k + i) * 2
				copy(out[dst:dst+2], t.Data[src:src+2])
			}
		}
		return out, nil
	default:
		return nil, core.NewError("gemma4.AssembleAudio: depthwise conv1d weight must be [channels,1,kernel] or [channels,kernel,1]")
	}
}

func audioFoldPerDimScale(t safetensors.Tensor, headDim int) ([]float32, error) {
	vals, ok := audioF32Values(t)
	if !ok {
		return nil, core.NewError("gemma4.AssembleAudio: per_dim_scale must be BF16 or F32")
	}
	if len(vals) < headDim {
		return nil, core.NewError("gemma4.AssembleAudio: per_dim_scale shorter than head_dim")
	}
	qScale := float32(1 / (math.Sqrt(float64(headDim)) * math.Ln2))
	out := make([]float32, headDim)
	for i := range out {
		out[i] = float32(math.Log1p(math.Exp(float64(vals[i])))) * qScale
	}
	return out, nil
}

func audioPositionTable(count, hidden int) []float32 {
	half := hidden / 2
	logIncrement := math.Log(10000.0) / float64(max(half-1, 1))
	vals := make([]float32, count*hidden)
	for p := 0; p < count; p++ {
		position := float64(count - 1 - p)
		row := p * hidden
		for i := 0; i < half; i++ {
			scaled := position * math.Exp(float64(i)*-logIncrement)
			vals[row+i] = float32(math.Sin(scaled))
			vals[row+half+i] = float32(math.Cos(scaled))
		}
	}
	return vals
}

func validateGemma4AudioConfigForAssemble(cfg *Gemma4AudioConfig) error {
	switch {
	case cfg.HiddenSize <= 0,
		cfg.NumHiddenLayers <= 0,
		cfg.NumAttentionHeads <= 0,
		cfg.AttentionChunkSize <= 0,
		cfg.AttentionContextLeft <= 0,
		cfg.ConvKernelSize <= 0,
		len(cfg.SubsamplingConvChannels) != 2,
		cfg.OutputProjDims <= 0,
		cfg.ResidualWeight == 0,
		cfg.AttentionLogitCap == 0:
		return core.E("gemma4.audio", core.Sprintf(
			"audio_config incomplete for the Conformer encoder: hidden=%d layers=%d heads=%d chunk=%d left=%d kernel=%d channels=%v proj=%d residual=%v cap=%v",
			cfg.HiddenSize, cfg.NumHiddenLayers, cfg.NumAttentionHeads,
			cfg.AttentionChunkSize, cfg.AttentionContextLeft, cfg.ConvKernelSize,
			cfg.SubsamplingConvChannels, cfg.OutputProjDims, cfg.ResidualWeight, cfg.AttentionLogitCap), nil)
	}
	if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
		return core.E("gemma4.audio", core.Sprintf("hidden_size %d not divisible by heads %d", cfg.HiddenSize, cfg.NumAttentionHeads), nil)
	}
	return nil
}

func loadedAudioConfig(cfg *Gemma4AudioConfig, textCfg *Gemma4TextConfig, ffInter int) model.LoadedAudioConfig {
	if cfg == nil {
		cfg = normalizeGemma4AudioConfig(&Gemma4AudioConfig{})
	}
	hidden := int(cfg.HiddenSize)
	headDim := 0
	if cfg.NumAttentionHeads > 0 {
		headDim = hidden / int(cfg.NumAttentionHeads)
	}
	out := model.LoadedAudioConfig{
		Hidden:          hidden,
		FFInter:         ffInter,
		Channels:        hidden,
		KernelSize:      int(cfg.ConvKernelSize),
		Eps:             cfg.RMSNormEps,
		Act:             cfg.HiddenAct,
		FFResidual:      cfg.ResidualWeight,
		ClipMin:         -cfg.GradientClipping,
		ClipMax:         cfg.GradientClipping,
		NumHeads:        int(cfg.NumAttentionHeads),
		HeadDim:         headDim,
		ChunkSize:       int(cfg.AttentionChunkSize),
		PastHorizon:     int(cfg.AttentionContextLeft) - 1,
		FutureHorizon:   int(cfg.AttentionContextRight),
		KScale:          float32(math.Log(1+math.E) / math.Ln2),
		LogitCap:        cfg.AttentionLogitCap,
		InvalidLogit:    cfg.AttentionInvalidLogitsValue,
		OutputDim:       int(cfg.OutputProjDims),
		AudioBeginToken: Gemma4BOAToken,
		AudioToken:      Gemma4AudioToken,
		AudioEndToken:   Gemma4EOAToken,
	}
	if textCfg != nil {
		out.AudioTokenID = int(textCfg.AudioTokenID)
	}
	return out
}

// AssembleAudio gathers the Gemma 4 audio tower/projector payload. Projector-only audio packs return a
// non-nil payload with no layers; text-only packs return (nil, nil).
func AssembleAudio(weights map[string]safetensors.Tensor, textCfg *Gemma4TextConfig) (*model.LoadedAudio, error) {
	if !HasAudioTowerWeights(weights) && !HasAudioProjectionWeights(weights) {
		return nil, nil
	}
	audioCfg := normalizeGemma4AudioConfig(&Gemma4AudioConfig{})
	if textCfg != nil && textCfg.AudioConfig != nil {
		copied := *textCfg.AudioConfig
		audioCfg = normalizeGemma4AudioConfig(&copied)
	}
	out := &model.LoadedAudio{
		Projector: audioLinearWithInputDim(weights, int(audioCfg.OutputProjDims), "embed_audio.embedding_projection"),
		Cfg:       loadedAudioConfig(audioCfg, textCfg, 0),
	}
	if !HasAudioTowerWeights(weights) {
		return out, nil
	}
	if textCfg == nil || textCfg.AudioConfig == nil {
		return nil, core.NewError("gemma4: audio tower weights present but config declares no audio_config")
	}
	if err := validateGemma4AudioConfigForAssemble(audioCfg); err != nil {
		return nil, err
	}

	sub, err := assembleAudioSubsample(weights)
	if err != nil {
		return nil, err
	}
	out.Subsample = sub
	out.OutputProj = audioWeight(weights, "audio_tower.output_proj.weight", "audio_tower.output_proj.linear.weight")
	if out.OutputProj == nil {
		return nil, core.NewError("gemma4: audio tower missing output_proj")
	}
	headDim := int(audioCfg.HiddenSize / audioCfg.NumAttentionHeads)
	pos := audioPositionTable(int(audioCfg.AttentionContextLeft), int(audioCfg.HiddenSize))
	out.Layers = make([]model.LoadedAudioLayer, int(audioCfg.NumHiddenLayers))
	ffInter := 0
	for i := range out.Layers {
		layer, layerFF, layerErr := assembleAudioLayer(weights, audioCfg, i, headDim, pos)
		if layerErr != nil {
			return nil, layerErr
		}
		if ffInter == 0 {
			ffInter = layerFF
		}
		out.Layers[i] = layer
	}
	out.Cfg = loadedAudioConfig(audioCfg, textCfg, ffInter)
	return out, nil
}

func assembleAudioSubsample(weights map[string]safetensors.Tensor) (model.LoadedAudioSubsample, error) {
	layer := func(idx int) ([]byte, []byte, []byte, error) {
		base := core.Sprintf("audio_tower.subsample_conv_projection.layer%d", idx)
		convT, ok := audioTensor(weights, base+".conv.weight")
		if !ok {
			return nil, nil, nil, core.E("gemma4.audio", core.Sprintf("subsample layer%d conv/norm weights missing", idx), nil)
		}
		norm := audioWeight(weights, base+".norm.weight")
		if norm == nil {
			return nil, nil, nil, core.E("gemma4.audio", core.Sprintf("subsample layer%d conv/norm weights missing", idx), nil)
		}
		conv, err := audioConv2dToOHWI(convT)
		if err != nil {
			return nil, nil, nil, err
		}
		return conv, norm, audioZerosBF16(convT.Shape[0]), nil
	}
	conv0, norm0, bias0, err := layer(0)
	if err != nil {
		return model.LoadedAudioSubsample{}, err
	}
	conv1, norm1, bias1, err := layer(1)
	if err != nil {
		return model.LoadedAudioSubsample{}, err
	}
	proj := audioLinear(weights, "audio_tower.subsample_conv_projection.input_proj_linear")
	if proj.Weight == nil {
		return model.LoadedAudioSubsample{}, core.NewError("gemma4: audio subsample input_proj_linear missing")
	}
	return model.LoadedAudioSubsample{
		Conv0: conv0, Norm0W: norm0, Norm0B: bias0,
		Conv1: conv1, Norm1W: norm1, Norm1B: bias1,
		InputProj: proj,
	}, nil
}

func assembleAudioLayer(weights map[string]safetensors.Tensor, cfg *Gemma4AudioConfig, idx, headDim int, pos []float32) (model.LoadedAudioLayer, int, error) {
	base := core.Sprintf("audio_tower.layers.%d.", idx)
	norm := func(name string) []byte { return audioWeight(weights, base+name+".weight") }
	ff := func(name string) (model.LoadedAudioFeedForward, int, error) {
		prefix := base + name
		ffw1 := audioClippable(weights, prefix+".ffw_layer_1")
		ffw2 := audioClippable(weights, prefix+".ffw_layer_2")
		pre := norm(name + ".pre_layer_norm")
		post := norm(name + ".post_layer_norm")
		if ffw1.Weight == nil || ffw2.Weight == nil || pre == nil || post == nil {
			return model.LoadedAudioFeedForward{}, 0, core.E("gemma4.audio", core.Sprintf("layer %d %s incomplete", idx, name), nil)
		}
		inter := 0
		if t, ok := audioTensor(weights, prefix+".ffw_layer_1.linear.weight"); ok && len(t.Shape) > 0 {
			inter = t.Shape[0]
		}
		return model.LoadedAudioFeedForward{PreNorm: pre, PostNorm: post, FFW1: ffw1, FFW2: ffw2}, inter, nil
	}
	ff1, ffInter, err := ff("feed_forward1")
	if err != nil {
		return model.LoadedAudioLayer{}, 0, err
	}
	ff2, _, err := ff("feed_forward2")
	if err != nil {
		return model.LoadedAudioLayer{}, 0, err
	}

	perDim, ok := audioTensor(weights, base+"self_attn.per_dim_scale")
	if !ok {
		return model.LoadedAudioLayer{}, 0, core.E("gemma4.audio", core.Sprintf("layer %d self_attn incomplete", idx), nil)
	}
	qScale, err := audioFoldPerDimScale(perDim, headDim)
	if err != nil {
		return model.LoadedAudioLayer{}, 0, err
	}
	attn := model.LoadedAudioAttention{
		Q:             audioClippable(weights, base+"self_attn.q_proj"),
		K:             audioClippable(weights, base+"self_attn.k_proj"),
		V:             audioClippable(weights, base+"self_attn.v_proj"),
		Post:          audioClippable(weights, base+"self_attn.post"),
		RelativeKProj: audioWeight(weights, base+"self_attn.relative_k_proj.weight", base+"self_attn.relative_k_proj.linear.weight"),
		QScalePerDim:  qScale,
		PosEmbed:      pos,
		PosCount:      int(cfg.AttentionContextLeft),
	}
	if attn.Q.Weight == nil || attn.K.Weight == nil || attn.V.Weight == nil || attn.Post.Weight == nil || attn.RelativeKProj == nil {
		return model.LoadedAudioLayer{}, 0, core.E("gemma4.audio", core.Sprintf("layer %d self_attn incomplete", idx), nil)
	}

	depthwise, ok := audioTensor(weights, base+"lconv1d.depthwise_conv1d.weight")
	if !ok {
		return model.LoadedAudioLayer{}, 0, core.E("gemma4.audio", core.Sprintf("layer %d lconv1d incomplete", idx), nil)
	}
	dw, err := audioDepthwiseToNLC(depthwise)
	if err != nil {
		return model.LoadedAudioLayer{}, 0, err
	}
	lconv := model.LoadedAudioLightConv{
		LinearStart:     audioClippable(weights, base+"lconv1d.linear_start"),
		LinearEnd:       audioClippable(weights, base+"lconv1d.linear_end"),
		PreNorm:         norm("lconv1d.pre_layer_norm"),
		ConvNorm:        norm("lconv1d.conv_norm"),
		DepthwiseWeight: dw,
	}
	if lconv.LinearStart.Weight == nil || lconv.LinearEnd.Weight == nil || lconv.PreNorm == nil || lconv.ConvNorm == nil {
		return model.LoadedAudioLayer{}, 0, core.E("gemma4.audio", core.Sprintf("layer %d lconv1d incomplete", idx), nil)
	}

	layer := model.LoadedAudioLayer{
		FF1:          ff1,
		FF2:          ff2,
		Attn:         attn,
		LConv:        lconv,
		NormPreAttn:  norm("norm_pre_attn"),
		NormPostAttn: norm("norm_post_attn"),
		NormOut:      norm("norm_out"),
	}
	if layer.NormPreAttn == nil || layer.NormPostAttn == nil || layer.NormOut == nil {
		return model.LoadedAudioLayer{}, 0, core.E("gemma4.audio", core.Sprintf("layer %d block norms incomplete", idx), nil)
	}
	return layer, ffInter, nil
}
