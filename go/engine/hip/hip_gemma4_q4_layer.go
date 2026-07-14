// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"

	core "dappco.re/go"
)

const (
	hipGemma4Q4Layer0Operation = "rocm.hip.Gemma4Q4Layer0"

	hipGemma4Q4PerLayerCombineScale float32 = 0.70710678118654752440
)

type hipGemma4Q4Layer0Config struct {
	Layer              int
	LayerType          string
	Embedding          hipDeviceEmbeddingLookupConfig
	HiddenSize         int
	EmbeddingScale     float32
	VocabSize          int
	GroupSize          int
	HeadDim            int
	QueryHeads         int
	KeyHeads           int
	IntermediateSize   int
	RoPEBase           float32
	RoPERotaryDim      int
	RoPEFrequencyScale float32
	SlidingWindow      int
	AttentionKEqV      bool
	FinalLogitSoftcap  float32
	LayerScalar        float32
	PerLayerInput      hipGemma4Q4PerLayerInputConfig

	InputNorm           hipRMSNormDeviceWeightConfig
	QueryNorm           hipRMSNormDeviceWeightConfig
	KeyNorm             hipRMSNormDeviceWeightConfig
	PostAttentionNorm   hipRMSNormDeviceWeightConfig
	PreFeedForwardNorm  hipRMSNormDeviceWeightConfig
	PostFeedForwardNorm hipRMSNormDeviceWeightConfig
	FinalNorm           hipRMSNormDeviceWeightConfig

	QueryProjection  hipMLXQ4DeviceWeightConfig
	KeyProjection    hipMLXQ4DeviceWeightConfig
	ValueProjection  hipMLXQ4DeviceWeightConfig
	OutputProjection hipMLXQ4DeviceWeightConfig
	GateProjection   hipMLXQ4DeviceWeightConfig
	UpProjection     hipMLXQ4DeviceWeightConfig
	DownProjection   hipMLXQ4DeviceWeightConfig
	LMHeadProjection hipMLXQ4DeviceWeightConfig
	MoE              *hipGemma4MoELayerConfig
}

type hipBF16DeviceWeightConfig struct {
	WeightPointer nativeDevicePointer
	WeightBytes   uint64
	Rows          int
	Cols          int
}

type hipGemma4Q4PerLayerInputConfig struct {
	InputSize            int
	EmbeddingScale       float32
	ModelProjectionScale float32
	Embedding            hipDeviceEmbeddingLookupConfig
	ModelProjection      hipBF16DeviceWeightConfig
	ProjectionNorm       hipRMSNormDeviceWeightConfig
	InputGate            hipMLXQ4DeviceWeightConfig
	Projection           hipMLXQ4DeviceWeightConfig
	PostInputNorm        hipRMSNormDeviceWeightConfig
}

type hipGemma4Q4Layer0Request struct {
	TokenID  int32
	Position int
	RoPEBase float32
	Epsilon  float32
}

type hipGemma4Q4DecoderLayerRequest struct {
	Position              int
	RoPEBase              float32
	Epsilon               float32
	PriorKeys             []float32
	PriorValues           []float32
	DeviceKVAttention     bool
	DeviceKVMode          string
	EngineConfig          hipGemma4Q4EngineConfig
	PriorDeviceKV         *rocmDeviceKVCache
	PriorDescriptorTable  *rocmDeviceKVDescriptorTable
	KeepDeviceKV          bool
	PerLayerInput         []float32
	PerLayerInputDevice   *hipDeviceByteBuffer
	SharedKeys            []float32
	SharedValues          []float32
	SharedDeviceKV        *rocmDeviceKVCache
	SharedDescriptorTable *rocmDeviceKVDescriptorTable
	LayerInputDevice      *hipDeviceByteBuffer
	NextInputNorm         *hipRMSNormDeviceWeightConfig
	NextInputNormValue    hipRMSNormDeviceWeightConfig
	HasNextInputNorm      bool
	FinalHiddenOutput     *hipDeviceByteBuffer
	NextLayerInputOutput  *hipDeviceByteBuffer
	AttentionWorkspace    *hipAttentionHeadsChunkedWorkspace
	OmitDebugTensors      bool
	ReturnDeviceHidden    bool
	OmitHostKV            bool
}

type hipGemma4Q4ForwardConfig struct {
	Layers          []hipGemma4Q4Layer0Config
	KVSharedLayers  int
	SharedKVSources []int
}

type hipGemma4Q4ForwardRequest struct {
	TokenID                 int32
	Position                int
	RoPEBase                float32
	Epsilon                 float32
	DeviceKVAttention       bool
	DeviceKVMode            string
	EngineConfig            hipGemma4Q4EngineConfig
	PriorDeviceState        *hipGemma4Q4DeviceDecodeState
	ReturnDeviceState       bool
	DeviceFinalSample       bool
	DeviceFinalScores       bool
	DeviceFinalTopKSample   bool
	DeferFinalSampleRead    bool
	FinalCandidateCount     int
	FinalTemperature        float32
	FinalTopP               float32
	FinalDraw               float64
	SkipFinalSample         bool
	FinalGreedyBuffer       *hipDeviceByteBuffer
	TokenIDDeviceBuffer     *hipDeviceByteBuffer
	SuppressTokens          []int32
	AttentionWorkspace      *hipAttentionHeadsChunkedWorkspace
	OmitDebugTensors        bool
	OmitLabels              bool
	OmitHostState           bool
	ReturnDeviceFinalHidden bool
}

type hipGemma4Q4LayerKVState struct {
	Keys   []float32
	Values []float32
}

type hipGemma4Q4PerLayerInputDeviceSet struct {
	driver           nativeHIPDriver
	layerCount       int
	layerStrideBytes uint64
	layerValueCount  int
	viewLabel        string
	borrowedBacking  bool
	view             hipDeviceByteBuffer
	Backing          []*hipDeviceByteBuffer
}

func (set *hipGemma4Q4PerLayerInputDeviceSet) LayerCount() int {
	if set == nil {
		return 0
	}
	return set.layerCount
}

func (set *hipGemma4Q4PerLayerInputDeviceSet) Layer(index int) *hipDeviceByteBuffer {
	if set == nil || index < 0 || index >= set.layerCount || set.layerStrideBytes == 0 || set.layerValueCount <= 0 ||
		len(set.Backing) == 0 || set.Backing[0] == nil || set.Backing[0].Pointer() == 0 {
		return nil
	}
	offset := nativeDevicePointer(uint64(index) * set.layerStrideBytes)
	set.view = hipDeviceByteBuffer{
		driver:    set.driver,
		pointer:   set.Backing[0].Pointer() + offset,
		count:     set.layerValueCount,
		sizeBytes: set.layerStrideBytes,
		borrowed:  true,
		label:     set.viewLabel,
	}
	return &set.view
}

func (set *hipGemma4Q4PerLayerInputDeviceSet) Close() error {
	if set == nil {
		return nil
	}
	var lastErr error
	if !set.borrowedBacking {
		for _, buffer := range set.Backing {
			if err := buffer.Close(); err != nil {
				lastErr = err
			}
		}
	}
	return lastErr
}

type hipGemma4Q4DecodeState struct {
	Layers []hipGemma4Q4LayerKVState
}

type hipGemma4Q4GreedyDecodeRequest struct {
	PromptTokenIDs    []int32
	MaxNewTokens      int
	Position          int
	RoPEBase          float32
	Epsilon           float32
	MirrorDeviceKV    bool
	DeviceKVMode      string
	EngineConfig      hipGemma4Q4EngineConfig
	DeviceKVAttention bool
}

type hipGemma4Q4Layer0Result struct {
	Embedding           []float32
	ScaledEmbedding     []float32
	LayerInput          []float32
	AttentionOutput     []float32
	AttentionProjection []float32
	AttentionResidual   []float32
	MLPOutput           []float32
	FinalHidden         []float32
	Logits              []float32
	Greedy              hipGreedySampleResult
	Labels              map[string]string
}

type hipGemma4Q4DecoderLayerResult struct {
	LayerInput                   []float32
	Key                          []float32
	Value                        []float32
	UpdatedKeys                  []float32
	UpdatedValues                []float32
	DeviceKVAttention            string
	DeviceLayer                  hipGemma4Q4DeviceLayerKVState
	DeviceLayerValid             bool
	AttentionOutput              []float32
	AttentionProjection          []float32
	AttentionResidual            []float32
	MLPOutput                    []float32
	FinalHidden                  []float32
	DeviceFinalHidden            *hipDeviceByteBuffer
	DeviceFinalHiddenBorrowed    bool
	DeviceNextLayerInput         *hipDeviceByteBuffer
	DeviceNextLayerInputBorrowed bool
}

type hipGemma4Q4ForwardResult struct {
	Embedding                 []float32
	ScaledEmbedding           []float32
	LayerResults              []hipGemma4Q4DecoderLayerResult
	FinalHidden               []float32
	Logits                    []float32
	Greedy                    hipGreedySampleResult
	GreedyDevice              *hipDeviceByteBuffer
	Candidates                []hipGreedySampleResult
	DeviceFinalHidden         *hipDeviceByteBuffer
	DeviceFinalHiddenBorrowed bool
	DeviceState               *hipGemma4Q4DeviceDecodeState
	Labels                    map[string]string
}

type hipGemma4Q4GreedyDecodeResult struct {
	Generated   []hipGreedySampleResult
	StepResults []hipGemma4Q4ForwardResult
	State       hipGemma4Q4DecodeState
	DeviceState *hipGemma4Q4DeviceDecodeState
	Labels      map[string]string
}

func (cfg hipGemma4Q4Layer0Config) keyValueDim() int {
	if cfg.KeyHeads <= 0 {
		return cfg.HeadDim
	}
	return cfg.KeyHeads * cfg.HeadDim
}

func (model *hipLoadedModel) loadedGemma4Q4Layer0Config() (hipGemma4Q4Layer0Config, error) {
	return model.loadedGemma4Q4LayerConfig(0)
}

func (model *hipLoadedModel) loadedGemma4Q4LayerConfig(layer int) (hipGemma4Q4Layer0Config, error) {
	return model.loadedGemma4Q4LayerConfigWithSharedKV(layer, nil)
}

func (model *hipLoadedModel) loadedGemma4Q4LayerConfigWithSharedKV(layer int, sharedKV *hipGemma4Q4Layer0Config) (hipGemma4Q4Layer0Config, error) {
	if model == nil {
		return hipGemma4Q4Layer0Config{}, core.E(hipGemma4Q4Layer0Operation, "loaded model is required", nil)
	}
	if model.driver == nil || !model.driver.Available() {
		return hipGemma4Q4Layer0Config{}, core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if !hipLoadedGemma4Q4GenerateLinked(model) {
		if model.gemma4TextConfig.EnableMoEBlock {
			return hipGemma4Q4Layer0Config{}, core.E(hipGemma4Q4Layer0Operation, "loaded Gemma4 MoE runtime is not ready", hipLoadedGemma4MoERuntimeError(model))
		}
		return hipGemma4Q4Layer0Config{}, core.E(hipGemma4Q4Layer0Operation, "loaded Gemma4 MLX affine 4/6/8-bit model is required", nil)
	}
	if layer < 0 {
		return hipGemma4Q4Layer0Config{}, core.E(hipGemma4Q4Layer0Operation, "layer index must be non-negative", nil)
	}
	hidden := model.modelInfo.HiddenSize
	vocab := model.modelInfo.VocabSize
	groupSize := model.modelInfo.QuantGroup
	if groupSize == 0 {
		groupSize = 64
	}
	if hidden <= 0 || vocab <= 0 || groupSize <= 0 {
		return hipGemma4Q4Layer0Config{}, core.E(hipGemma4Q4Layer0Operation, "model hidden, vocab, and MLX affine group sizes must be positive", nil)
	}

	embedding, err := model.loadedGemma4Q4EmbeddingConfig(groupSize)
	if err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}
	layerPrefix := core.Sprintf("language_model.model.layers.%d", layer)
	query, queryRows, queryCols, err := model.loadedGemma4Q4ProjectionConfig(layerPrefix+".self_attn.q_proj", "q_proj", groupSize)
	if err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}
	var key, value hipMLXQ4DeviceWeightConfig
	var keyRows, keyCols, valueRows, valueCols int
	layerType := ""
	if sharedKV != nil {
		key = sharedKV.KeyProjection
		keyRows = key.Rows
		keyCols = key.Cols
		value = sharedKV.ValueProjection
		valueRows = value.Rows
		valueCols = value.Cols
		layerType = sharedKV.LayerType
	} else {
		key, keyRows, keyCols, err = model.loadedGemma4Q4ProjectionConfig(layerPrefix+".self_attn.k_proj", "k_proj", groupSize)
		if err != nil {
			return hipGemma4Q4Layer0Config{}, err
		}
		layerType = model.loadedGemma4Q4LayerType(layer, keyRows)
	}
	headDim := model.loadedGemma4Q4LayerHeadDim(layerType, queryRows, keyRows)
	attentionKEqV := model.loadedGemma4Q4LayerAttentionKEqV(layerPrefix, layerType)
	if sharedKV == nil && attentionKEqV {
		value = key
		valueRows = keyRows
		valueCols = keyCols
	} else if sharedKV == nil {
		value, valueRows, valueCols, err = model.loadedGemma4Q4ProjectionConfig(layerPrefix+".self_attn.v_proj", "v_proj", groupSize)
		if err != nil {
			return hipGemma4Q4Layer0Config{}, err
		}
	}
	output, outputRows, outputCols, err := model.loadedGemma4Q4ProjectionConfig(layerPrefix+".self_attn.o_proj", "o_proj", groupSize)
	if err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}
	gate, gateRows, gateCols, err := model.loadedGemma4Q4ProjectionConfig(layerPrefix+".mlp.gate_proj", "mlp.gate_proj", groupSize)
	if err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}
	up, upRows, upCols, err := model.loadedGemma4Q4ProjectionConfig(layerPrefix+".mlp.up_proj", "mlp.up_proj", groupSize)
	if err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}
	down, downRows, downCols, err := model.loadedGemma4Q4ProjectionConfig(layerPrefix+".mlp.down_proj", "mlp.down_proj", groupSize)
	if err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}
	lmHead, lmRows, lmCols, err := model.loadedGemma4Q4LMHeadProjectionConfig(groupSize)
	if err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}
	if embedding.VocabSize != vocab || embedding.HiddenSize != hidden ||
		queryCols != hidden || keyCols != hidden || valueCols != hidden ||
		headDim <= 0 || keyRows <= 0 || valueRows != keyRows ||
		queryRows%headDim != 0 || keyRows%headDim != 0 ||
		outputRows != hidden || outputCols != queryRows ||
		gateCols != hidden || upCols != hidden || gateRows != upRows ||
		downRows != hidden || downCols != gateRows ||
		lmRows != vocab || lmCols != hidden {
		return hipGemma4Q4Layer0Config{}, core.E(hipGemma4Q4Layer0Operation, "Gemma4 q4 layer-0 tensor shapes are inconsistent", nil)
	}
	queryHeads := queryRows / headDim
	keyHeads := keyRows / headDim
	intermediate := gateRows
	ropeBase, ropeRotaryDim, ropeFrequencyScale := model.loadedGemma4Q4LayerRoPE(layerType, headDim)
	slidingWindow := model.loadedGemma4Q4EffectiveSlidingWindow(layerType, headDim)

	inputNorm, err := model.loadedGemma4NormConfig(layerPrefix+".input_layernorm.weight", "input_layernorm", hidden)
	if err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}
	queryNorm, err := model.loadedGemma4NormConfig(layerPrefix+".self_attn.q_norm.weight", "q_norm", headDim)
	if err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}
	var keyNorm hipRMSNormDeviceWeightConfig
	if sharedKV != nil {
		keyNorm = sharedKV.KeyNorm
	} else {
		keyNorm, err = model.loadedGemma4NormConfig(layerPrefix+".self_attn.k_norm.weight", "k_norm", headDim)
		if err != nil {
			return hipGemma4Q4Layer0Config{}, err
		}
	}
	postAttentionNorm, err := model.loadedGemma4NormConfig(layerPrefix+".post_attention_layernorm.weight", "post_attention_layernorm", hidden)
	if err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}
	preFeedForwardNorm, err := model.loadedGemma4NormConfig(layerPrefix+".pre_feedforward_layernorm.weight", "pre_feedforward_layernorm", hidden)
	if err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}
	postFeedForwardNorm, err := model.loadedGemma4NormConfig(layerPrefix+".post_feedforward_layernorm.weight", "post_feedforward_layernorm", hidden)
	if err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}
	finalNorm, err := model.loadedGemma4NormConfig("language_model.model.norm.weight", "final_norm", hidden)
	if err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}
	layerScalar, err := model.loadedGemma4Q4LayerScalar(layer)
	if err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}
	perLayerInput, err := model.loadedGemma4Q4PerLayerInputConfig(layerPrefix, layer, groupSize, hidden)
	if err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}

	cfg := hipGemma4Q4Layer0Config{
		Layer:               layer,
		LayerType:           layerType,
		Embedding:           embedding,
		HiddenSize:          hidden,
		VocabSize:           vocab,
		GroupSize:           embedding.GroupSize,
		HeadDim:             headDim,
		QueryHeads:          queryHeads,
		KeyHeads:            keyHeads,
		IntermediateSize:    intermediate,
		RoPEBase:            ropeBase,
		RoPERotaryDim:       ropeRotaryDim,
		RoPEFrequencyScale:  ropeFrequencyScale,
		SlidingWindow:       slidingWindow,
		AttentionKEqV:       attentionKEqV,
		FinalLogitSoftcap:   model.loadedGemma4Q4FinalLogitSoftcap(),
		LayerScalar:         layerScalar,
		PerLayerInput:       perLayerInput,
		InputNorm:           inputNorm,
		QueryNorm:           queryNorm,
		KeyNorm:             keyNorm,
		PostAttentionNorm:   postAttentionNorm,
		PreFeedForwardNorm:  preFeedForwardNorm,
		PostFeedForwardNorm: postFeedForwardNorm,
		FinalNorm:           finalNorm,
		QueryProjection:     query,
		KeyProjection:       key,
		ValueProjection:     value,
		OutputProjection:    output,
		GateProjection:      gate,
		UpProjection:        up,
		DownProjection:      down,
		LMHeadProjection:    lmHead,
	}
	moe, err := model.loadedGemma4MoELayerConfig(layer, hidden)
	if err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}
	cfg.MoE = moe
	cfg.finalizeScales()
	if err := cfg.validate(); err != nil {
		return hipGemma4Q4Layer0Config{}, err
	}
	return cfg, nil
}

func (model *hipLoadedModel) loadedGemma4Q4ForwardConfig(layerCount int) (hipGemma4Q4ForwardConfig, error) {
	if layerCount <= 0 {
		return hipGemma4Q4ForwardConfig{}, core.E(hipGemma4Q4Layer0Operation, "layer count must be positive", nil)
	}
	if model == nil {
		return hipGemma4Q4ForwardConfig{}, core.E(hipGemma4Q4Layer0Operation, "loaded model is required", nil)
	}
	if model.modelInfo.NumLayers > 0 && layerCount > model.modelInfo.NumLayers {
		return hipGemma4Q4ForwardConfig{}, core.E(hipGemma4Q4Layer0Operation, "layer count exceeds loaded model layer count", nil)
	}
	if sharedSpecs, ok := model.sharedGemma4LayerSpecs(layerCount); ok {
		layers := make([]hipGemma4Q4Layer0Config, 0, layerCount)
		sharedSources := make([]int, layerCount)
		kvSharedLayers := 0
		for layer, spec := range sharedSpecs {
			source := spec.KVShareFrom
			if source < 0 || source > layer {
				return hipGemma4Q4ForwardConfig{}, core.E(hipGemma4Q4Layer0Operation, core.Sprintf("shared Arch layer %d has invalid KV owner %d", layer, source), nil)
			}
			sharedSources[layer] = source
			var sharedKV *hipGemma4Q4Layer0Config
			if source != layer {
				kvSharedLayers++
				sharedKV = &layers[source]
			}
			cfg, err := model.loadedGemma4Q4LayerConfigWithSharedKV(layer, sharedKV)
			if err != nil {
				return hipGemma4Q4ForwardConfig{}, core.E(hipGemma4Q4Layer0Operation, core.Sprintf("load shared-Arch layer %d config", layer), err)
			}
			if cfg.LayerType != spec.TypeName() {
				return hipGemma4Q4ForwardConfig{}, core.E(hipGemma4Q4Layer0Operation, core.Sprintf("layer %d type %s does not match shared Arch %s", layer, cfg.LayerType, spec.TypeName()), nil)
			}
			if spec.HeadDim > 0 && cfg.HeadDim != spec.HeadDim {
				return hipGemma4Q4ForwardConfig{}, core.E(hipGemma4Q4Layer0Operation, core.Sprintf("layer %d head dim %d does not match shared Arch %d", layer, cfg.HeadDim, spec.HeadDim), nil)
			}
			if spec.KVHeads > 0 && cfg.KeyHeads != spec.KVHeads {
				return hipGemma4Q4ForwardConfig{}, core.E(hipGemma4Q4Layer0Operation, core.Sprintf("layer %d KV heads %d does not match shared Arch %d", layer, cfg.KeyHeads, spec.KVHeads), nil)
			}
			if (cfg.MoE != nil) != spec.MoE {
				return hipGemma4Q4ForwardConfig{}, core.E(hipGemma4Q4Layer0Operation, core.Sprintf("layer %d MoE route does not match shared Arch", layer), nil)
			}
			layers = append(layers, cfg)
		}
		forward := hipGemma4Q4ForwardConfig{
			Layers:          layers,
			KVSharedLayers:  kvSharedLayers,
			SharedKVSources: sharedSources,
		}
		if err := forward.validate(); err != nil {
			return hipGemma4Q4ForwardConfig{}, err
		}
		return forward, nil
	}
	kvSharedLayers := model.loadedGemma4Q4KVSharedLayers(layerCount)
	firstSharedLayer := layerCount - kvSharedLayers
	layers := make([]hipGemma4Q4Layer0Config, 0, layerCount)
	latestKVSourceByType := map[string]int{}
	for layer := 0; layer < layerCount; layer++ {
		layerType := model.loadedGemma4Q4LayerType(layer, 0)
		ownsKV := layer < firstSharedLayer
		var sharedKV *hipGemma4Q4Layer0Config
		if !ownsKV {
			if source, ok := latestKVSourceByType[layerType]; ok {
				sharedKV = &layers[source]
			} else {
				ownsKV = true
			}
		}
		cfg, err := model.loadedGemma4Q4LayerConfigWithSharedKV(layer, sharedKV)
		if err != nil {
			return hipGemma4Q4ForwardConfig{}, core.E(hipGemma4Q4Layer0Operation, core.Sprintf("load layer %d config", layer), err)
		}
		layers = append(layers, cfg)
		if ownsKV {
			latestKVSourceByType[cfg.LayerType] = layer
		}
	}
	forward := hipGemma4Q4ForwardConfig{
		Layers:         layers,
		KVSharedLayers: kvSharedLayers,
	}
	forward.SharedKVSources = hipGemma4Q4BuildSharedKVSourceByLayer(forward)
	if err := forward.validate(); err != nil {
		return hipGemma4Q4ForwardConfig{}, err
	}
	return forward, nil
}

func (model *hipLoadedModel) cachedGemma4Q4ForwardConfig(layerCount int) (hipGemma4Q4ForwardConfig, error) {
	if layerCount <= 0 {
		return hipGemma4Q4ForwardConfig{}, core.E(hipGemma4Q4Layer0Operation, "layer count must be positive", nil)
	}
	if model == nil {
		return hipGemma4Q4ForwardConfig{}, core.E(hipGemma4Q4Layer0Operation, "loaded model is required", nil)
	}
	model.q4ConfigMu.Lock()
	defer model.q4ConfigMu.Unlock()
	if model.q4ConfigOK && model.q4Layers == layerCount {
		return model.q4Config, nil
	}
	cfg, err := model.loadedGemma4Q4ForwardConfig(layerCount)
	if err != nil {
		return hipGemma4Q4ForwardConfig{}, err
	}
	model.q4Config = cfg
	model.q4Layers = layerCount
	model.q4ConfigOK = true
	return cfg, nil
}

func hipRunGemma4Q4Layer0(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, req hipGemma4Q4Layer0Request) (hipGemma4Q4Layer0Result, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipGemma4Q4Layer0Result{}, err
	}
	if driver == nil || !driver.Available() {
		return hipGemma4Q4Layer0Result{}, core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if err := cfg.validate(); err != nil {
		return hipGemma4Q4Layer0Result{}, err
	}
	if err := req.validate(cfg); err != nil {
		return hipGemma4Q4Layer0Result{}, err
	}

	embedding, err := hipRunEmbeddingLookupKernelWithDeviceTable(ctx, driver, []int32{req.TokenID}, cfg.Embedding)
	if err != nil {
		return hipGemma4Q4Layer0Result{}, err
	}
	scaledEmbedding, err := hipRunVectorScaleKernel(ctx, driver, hipVectorScaleRequest{
		Input: embedding,
		Scale: cfg.embeddingScale(),
	})
	if err != nil {
		return hipGemma4Q4Layer0Result{}, err
	}
	perLayerInput, err := hipRunGemma4Q4PerLayerInputForLayer(ctx, driver, cfg, req.TokenID, scaledEmbedding, req.Epsilon)
	if err != nil {
		return hipGemma4Q4Layer0Result{}, err
	}
	layer, err := hipRunGemma4Q4DecoderLayer(ctx, driver, cfg, scaledEmbedding, hipGemma4Q4DecoderLayerRequest{
		Position:      req.Position,
		RoPEBase:      req.RoPEBase,
		Epsilon:       req.Epsilon,
		PerLayerInput: perLayerInput,
	})
	if err != nil {
		return hipGemma4Q4Layer0Result{}, err
	}
	finalNormCfg := cfg.FinalNorm
	finalNormCfg.Epsilon = req.Epsilon
	finalNorm, err := hipRunRMSNormKernelWithDeviceWeightConfig(ctx, driver, layer.FinalHidden, finalNormCfg)
	if err != nil {
		return hipGemma4Q4Layer0Result{}, err
	}
	logits, err := hipRunMLXQ4ProjectionKernelWithDeviceWeightConfig(ctx, driver, finalNorm, cfg.LMHeadProjection)
	if err != nil {
		return hipGemma4Q4Layer0Result{}, err
	}
	logits, err = hipGemma4Q4SoftcapLogits(logits, cfg.FinalLogitSoftcap)
	if err != nil {
		return hipGemma4Q4Layer0Result{}, err
	}
	greedy, err := hipRunGreedyKernel(ctx, driver, hipGreedySampleRequest{Logits: logits})
	if err != nil {
		return hipGemma4Q4Layer0Result{}, err
	}
	return hipGemma4Q4Layer0Result{
		Embedding:           embedding,
		ScaledEmbedding:     scaledEmbedding,
		LayerInput:          layer.LayerInput,
		AttentionOutput:     layer.AttentionOutput,
		AttentionProjection: layer.AttentionProjection,
		AttentionResidual:   layer.AttentionResidual,
		MLPOutput:           layer.MLPOutput,
		FinalHidden:         layer.FinalHidden,
		Logits:              logits,
		Greedy:              greedy,
		Labels:              hipGemma4Q4Layer0Labels(cfg, req),
	}, nil
}

func hipRunGemma4Q4SingleTokenForward(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, req hipGemma4Q4ForwardRequest) (hipGemma4Q4ForwardResult, error) {
	result, _, err := hipRunGemma4Q4SingleTokenForwardWithState(ctx, driver, cfg, hipGemma4Q4DecodeState{}, req)
	return result, err
}

func hipRunGemma4Q4SingleTokenForwardWithState(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, state hipGemma4Q4DecodeState, req hipGemma4Q4ForwardRequest) (hipGemma4Q4ForwardResult, hipGemma4Q4DecodeState, error) {
	return hipRunGemma4Q4SingleTokenForwardWithStateInternal(ctx, driver, cfg, state, req, true)
}

func hipRunGemma4Q4SingleTokenForwardWithStateInternal(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, state hipGemma4Q4DecodeState, req hipGemma4Q4ForwardRequest, validate bool) (hipGemma4Q4ForwardResult, hipGemma4Q4DecodeState, error) {
	routeMetrics := hipBeginDecodeRouteMetrics()
	defer hipFinishDecodeRouteMetrics(routeMetrics)
	if err := hipContextErr(ctx); err != nil {
		return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
	}
	if driver == nil || !driver.Available() {
		return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if validate {
		if err := cfg.validate(); err != nil {
			return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
		}
		if err := state.validate(cfg); err != nil {
			return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
		}
	}
	if req.PriorDeviceState != nil {
		if !req.DeviceKVAttention {
			return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, core.E(hipGemma4Q4Layer0Operation, "prior device state requires device KV attention", nil)
		}
		if validate {
			if err := req.PriorDeviceState.CompatibleWithHostState(cfg, state, req.DeviceKVMode); err != nil {
				return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
			}
		}
	}
	if req.ReturnDeviceState && !req.DeviceKVAttention {
		return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, core.E(hipGemma4Q4Layer0Operation, "returning device state requires device KV attention", nil)
	}
	if req.DeviceFinalSample && req.SkipFinalSample {
		return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, core.E(hipGemma4Q4Layer0Operation, "final sample cannot be both requested and skipped", nil)
	}
	if req.DeferFinalSampleRead && !req.DeviceFinalSample {
		return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, core.E(hipGemma4Q4Layer0Operation, "deferred final sample requires device final sample", nil)
	}
	if req.DeviceFinalScores && req.SkipFinalSample {
		return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, core.E(hipGemma4Q4Layer0Operation, "final scores cannot be both requested and skipped", nil)
	}
	if req.DeviceFinalTopKSample && req.SkipFinalSample {
		return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, core.E(hipGemma4Q4Layer0Operation, "final top-k sample cannot be both requested and skipped", nil)
	}
	if req.DeviceFinalSample && (req.DeviceFinalScores || req.DeviceFinalTopKSample) {
		return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, core.E(hipGemma4Q4Layer0Operation, "final greedy sample and final scores/sample are mutually exclusive", nil)
	}
	if req.DeviceFinalScores && req.DeviceFinalTopKSample {
		return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, core.E(hipGemma4Q4Layer0Operation, "final scores and final top-k sample are mutually exclusive", nil)
	}
	if (req.DeviceFinalScores || req.DeviceFinalTopKSample) && req.FinalCandidateCount <= 0 {
		return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, core.E(hipGemma4Q4Layer0Operation, "final score candidate count must be positive", nil)
	}
	first := cfg.Layers[0]
	if validate {
		if err := (hipGemma4Q4Layer0Request{
			TokenID:  req.TokenID,
			Position: req.Position,
			RoPEBase: req.RoPEBase,
			Epsilon:  req.Epsilon,
		}).validate(first); err != nil {
			return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
		}
	}
	var err error
	var embedding []float32
	var scaledEmbedding []float32
	var hidden []float32
	var hiddenBuffer *hipDeviceByteBuffer
	hiddenBufferBorrowed := false
	var perLayerInputs [][]float32
	var perLayerInputDevices *hipGemma4Q4PerLayerInputDeviceSet
	if req.OmitDebugTensors {
		if req.AttentionWorkspace != nil {
			hiddenBuffer, err = req.AttentionWorkspace.EnsureScaledEmbedding(driver, first.HiddenSize)
			if err == nil && req.TokenIDDeviceBuffer != nil {
				err = hipRunEmbeddingLookupKernelWithDeviceTableGreedyTokenScaledOutputWithWorkspace(ctx, driver, first.Embedding, req.TokenIDDeviceBuffer, hiddenBuffer, first.embeddingScale(), req.AttentionWorkspace)
			} else if err == nil {
				tokenBuffer, tokenErr := req.AttentionWorkspace.EnsureTokenIDValue(driver, req.TokenID, first.Embedding.VocabSize)
				if tokenErr != nil {
					return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, tokenErr
				}
				err = hipRunEmbeddingLookupKernelWithDeviceTableTokenBufferScaledOutputWithWorkspace(ctx, driver, first.Embedding, tokenBuffer, hiddenBuffer, first.embeddingScale(), req.AttentionWorkspace)
			}
			hiddenBufferBorrowed = err == nil
		} else {
			var embeddingBuffer *hipDeviceByteBuffer
			embeddingBuffer, err = hipRunEmbeddingLookupKernelWithDeviceTableBuffer(ctx, driver, []int32{req.TokenID}, first.Embedding)
			if err != nil {
				return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
			}
			defer embeddingBuffer.Close()
			hiddenBuffer, err = hipRunVectorScaleDeviceKernel(ctx, driver, embeddingBuffer, first.embeddingScale())
		}
		if err != nil {
			return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
		}
		perLayerInputDevices, err = hipRunGemma4Q4PerLayerInputDeviceSet(ctx, driver, cfg, req.TokenID, req.TokenIDDeviceBuffer, hiddenBuffer, req.Epsilon, req.AttentionWorkspace)
		if err != nil {
			return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
		}
		defer perLayerInputDevices.Close()
	} else {
		embedding, err = hipRunEmbeddingLookupKernelWithDeviceTable(ctx, driver, []int32{req.TokenID}, first.Embedding)
		if err != nil {
			return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
		}
		scaledEmbedding, err = hipRunVectorScaleKernel(ctx, driver, hipVectorScaleRequest{
			Input: embedding,
			Scale: first.embeddingScale(),
		})
		if err != nil {
			return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
		}
		perLayerInputs, err = hipRunGemma4Q4PerLayerInputs(ctx, driver, cfg, req.TokenID, scaledEmbedding, req.Epsilon)
		if err != nil {
			return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
		}
		hidden = scaledEmbedding
	}
	sharedSources := hipGemma4Q4SharedKVSourceByLayer(cfg)
	defer func() {
		if hiddenBuffer != nil && !hiddenBufferBorrowed {
			_ = hiddenBuffer.Close()
		}
	}()
	var layerResults []hipGemma4Q4DecoderLayerResult
	if !req.OmitDebugTensors {
		layerResults = make([]hipGemma4Q4DecoderLayerResult, 0, len(cfg.Layers))
	}
	nextState := hipGemma4Q4DecodeState{}
	if !req.OmitHostState {
		nextState.Layers = make([]hipGemma4Q4LayerKVState, len(cfg.Layers))
	}
	var nextDeviceState *hipGemma4Q4DeviceDecodeState
	if req.ReturnDeviceState {
		mode := firstNonEmptyString(req.DeviceKVMode, rocmKVCacheModeFP16)
		if req.PriorDeviceState != nil && req.PriorDeviceState.mode != "" {
			mode = firstNonEmptyString(req.DeviceKVMode, req.PriorDeviceState.mode)
		}
		nextDeviceState = hipNewGemma4Q4DeviceDecodeState(mode, len(cfg.Layers))
	}
	success := false
	defer func() {
		if !success && nextDeviceState != nil {
			_ = nextDeviceState.Close()
		}
	}()
	deviceAppendLayers := 0
	deviceRemirrorLayers := 0
	deviceSharedLayers := 0
	sharedKVLayers := 0
	useDeviceSharedKV := req.DeviceKVAttention && req.ReturnDeviceState && req.OmitDebugTensors
	precomputedLayerInputBuffer := (*hipDeviceByteBuffer)(nil)
	precomputedLayerInputBorrowed := false
	defer func() {
		if precomputedLayerInputBuffer != nil && !precomputedLayerInputBorrowed {
			_ = precomputedLayerInputBuffer.Close()
		}
	}()
	var hostKVRequiredByLayer []bool
	if !useDeviceSharedKV {
		hostKVRequiredByLayer = make([]bool, len(cfg.Layers))
		for index, source := range sharedSources {
			if source != index && source >= 0 && source < len(hostKVRequiredByLayer) {
				hostKVRequiredByLayer[source] = true
				hostKVRequiredByLayer[index] = true
			}
		}
	}
	for index, layerCfg := range cfg.Layers {
		if routeMetrics != nil {
			routeMetrics.setLayer(index, layerCfg.LayerType)
		}
		layerState := state.layer(index)
		var priorDeviceKV *rocmDeviceKVCache
		var priorDescriptorTable *rocmDeviceKVDescriptorTable
		if req.PriorDeviceState != nil {
			priorDeviceKV = req.PriorDeviceState.layerCache(index)
			priorDescriptorTable = req.PriorDeviceState.layerDescriptorTable(index)
		}
		layerReq := hipGemma4Q4DecoderLayerRequest{
			Position:             req.Position,
			RoPEBase:             req.RoPEBase,
			Epsilon:              req.Epsilon,
			PriorKeys:            layerState.Keys,
			PriorValues:          layerState.Values,
			DeviceKVAttention:    req.DeviceKVAttention,
			DeviceKVMode:         req.DeviceKVMode,
			EngineConfig:         req.EngineConfig,
			PriorDeviceKV:        priorDeviceKV,
			PriorDescriptorTable: priorDescriptorTable,
			KeepDeviceKV:         req.ReturnDeviceState,
			AttentionWorkspace:   req.AttentionWorkspace,
			OmitDebugTensors:     req.OmitDebugTensors,
			ReturnDeviceHidden:   req.OmitDebugTensors,
			OmitHostKV:           req.DeviceKVAttention && req.ReturnDeviceState && req.OmitDebugTensors && (len(hostKVRequiredByLayer) == 0 || !hostKVRequiredByLayer[index]),
		}
		if precomputedLayerInputBuffer != nil {
			layerReq.LayerInputDevice = precomputedLayerInputBuffer
		}
		if req.OmitDebugTensors {
			var nextInputNormCfg hipRMSNormDeviceWeightConfig
			if index+1 < len(cfg.Layers) {
				nextInputNormCfg = cfg.Layers[index+1].InputNorm
			} else {
				if req.SkipFinalSample {
					nextInputNormCfg = hipRMSNormDeviceWeightConfig{}
				} else if req.DeviceFinalSample || req.DeviceFinalScores || req.DeviceFinalTopKSample {
					nextInputNormCfg = layerCfg.FinalNorm
				}
			}
			if nextInputNormCfg.Count > 0 {
				nextInputNormCfg.Epsilon = req.Epsilon
				layerReq.NextInputNormValue = nextInputNormCfg
				layerReq.HasNextInputNorm = true
			}
			if req.AttentionWorkspace != nil {
				slot := index & 1
				layerReq.FinalHiddenOutput, err = req.AttentionWorkspace.EnsureFinalHiddenOutput(driver, layerCfg.HiddenSize, slot)
				if err != nil {
					return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
				}
				if layerReq.HasNextInputNorm {
					layerReq.NextLayerInputOutput, err = req.AttentionWorkspace.EnsureNextInputOutput(driver, layerReq.NextInputNormValue.Count, slot)
					if err != nil {
						return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
					}
				}
			}
		}
		if perLayerInputDevices != nil {
			layerReq.PerLayerInputDevice = perLayerInputDevices.Layer(index)
		}
		if layerReq.PerLayerInputDevice == nil && len(perLayerInputs) > index {
			layerReq.PerLayerInput = perLayerInputs[index]
		}
		if len(sharedSources) > index && sharedSources[index] != index {
			source := sharedSources[index]
			if useDeviceSharedKV {
				if nextDeviceState == nil || source < 0 || source >= len(nextDeviceState.layers) || nextDeviceState.layers[source].cache == nil {
					return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, core.E(hipGemma4Q4Layer0Operation, "shared device KV source layer is unavailable", nil)
				}
				layerReq.SharedDeviceKV = nextDeviceState.layers[source].cache
				layerReq.SharedDescriptorTable = nextDeviceState.layers[source].descriptorTable
			} else {
				if source < 0 || source >= len(nextState.Layers) || len(nextState.Layers[source].Keys) == 0 {
					return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, core.E(hipGemma4Q4Layer0Operation, "shared KV source layer is unavailable", nil)
				}
				layerReq.SharedKeys = nextState.Layers[source].Keys
				layerReq.SharedValues = nextState.Layers[source].Values
			}
			sharedKVLayers++
		}
		consumedLayerInputBuffer := precomputedLayerInputBuffer
		consumedLayerInputBorrowed := precomputedLayerInputBorrowed
		precomputedLayerInputBuffer = nil
		precomputedLayerInputBorrowed = false
		layer, err := hipRunGemma4Q4DecoderLayerInternalWithDeviceInput(ctx, driver, layerCfg, hidden, hiddenBuffer, layerReq, false)
		if consumedLayerInputBuffer != nil && !consumedLayerInputBorrowed {
			_ = consumedLayerInputBuffer.Close()
		}
		if err != nil {
			return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
		}
		nextHiddenBuffer := layer.DeviceFinalHidden
		nextHiddenBorrowed := layer.DeviceFinalHiddenBorrowed
		layer.DeviceFinalHidden = nil
		precomputedLayerInputBuffer = layer.DeviceNextLayerInput
		precomputedLayerInputBorrowed = layer.DeviceNextLayerInputBorrowed
		layer.DeviceNextLayerInput = nil
		switch layer.DeviceKVAttention {
		case "append_existing_device":
			deviceAppendLayers++
		case "remirror_host_kv":
			deviceRemirrorLayers++
		case "shared_device_kv":
			deviceSharedLayers++
		}
		if req.ReturnDeviceState {
			if !layer.DeviceLayerValid {
				if nextHiddenBuffer != nil && !nextHiddenBorrowed {
					_ = nextHiddenBuffer.Close()
				}
				return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, core.E(hipGemma4Q4Layer0Operation, "decoder layer did not return device KV state", nil)
			}
			nextDeviceState.layers = append(nextDeviceState.layers, layer.DeviceLayer)
			layer.DeviceLayer = hipGemma4Q4DeviceLayerKVState{}
			layer.DeviceLayerValid = false
		}
		if !req.OmitDebugTensors {
			layerResults = append(layerResults, layer)
		}
		if !req.OmitHostState {
			nextState.Layers[index] = hipGemma4Q4LayerKVState{Keys: layer.UpdatedKeys, Values: layer.UpdatedValues}
		}
		if hiddenBuffer != nil && !hiddenBufferBorrowed {
			_ = hiddenBuffer.Close()
		}
		hiddenBuffer = nil
		hiddenBufferBorrowed = false
		if nextHiddenBuffer != nil {
			hiddenBuffer = nextHiddenBuffer
			hiddenBufferBorrowed = nextHiddenBorrowed
			hidden = nil
		} else {
			hidden = layer.FinalHidden
		}
	}
	last := cfg.Layers[len(cfg.Layers)-1]
	finalNormCfg := last.FinalNorm
	finalNormCfg.Epsilon = req.Epsilon
	var logits []float32
	var greedy hipGreedySampleResult
	var greedyDevice *hipDeviceByteBuffer
	var candidates []hipGreedySampleResult
	var deviceFinalHidden *hipDeviceByteBuffer
	deviceFinalHiddenBorrowed := false
	if req.SkipFinalSample {
		// Prompt prefill only needs updated KV state; sampling every intermediate
		// prompt token wastes a full LM-head projection.
	} else if req.DeviceFinalSample || req.DeviceFinalScores || req.DeviceFinalTopKSample {
		finalHiddenBuffer := hiddenBuffer
		if finalHiddenBuffer == nil {
			finalHiddenBuffer, err = hipUploadGemma4Q4Float32Input(driver, "Gemma4 q4 final hidden", hidden)
			if err != nil {
				return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
			}
			defer finalHiddenBuffer.Close()
		}
		finalNormBuffer := precomputedLayerInputBuffer
		if finalNormBuffer == nil {
			finalNormBuffer, err = hipRunRMSNormKernelWithDeviceInputWeightConfig(ctx, driver, finalHiddenBuffer, finalNormCfg)
			if err != nil {
				return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
			}
			defer finalNormBuffer.Close()
		}
		if receipts := hipActiveLogitSpreadReceipts(); receipts != nil && !req.DeviceFinalTopKSample {
			// The fused greedy kernel owns softcap, so its sampler input is the raw
			// LM-head projection. This armed-only twin projection exposes that exact
			// input without changing the production kernel or its unarmed cost.
			rawBuffer, receiptErr := hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, finalNormBuffer, last.LMHeadProjection)
			if receiptErr != nil {
				return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, receiptErr
			}
			rawLogits, receiptErr := hipReadFloat32DeviceOutput(rawBuffer, "rocm.hip.LogitSpread", "fused greedy projection scores", last.LMHeadProjection.Rows)
			_ = rawBuffer.Close()
			if receiptErr != nil {
				return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, receiptErr
			}
			receipts.recordNext("fused-greedy", "sampler-input-pre-softcap", rawLogits)
		}
		if req.DeviceFinalTopKSample {
			greedy, greedyDevice, err = hipRunMLXQ4ProjectionSoftcapSampleKernelWithDeviceInputBufferSuppress(ctx, driver, finalNormBuffer, last.LMHeadProjection, last.FinalLogitSoftcap, req.FinalCandidateCount, req.FinalTemperature, req.FinalTopP, req.FinalDraw, req.FinalGreedyBuffer, req.SuppressTokens, req.AttentionWorkspace)
			if err != nil {
				return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
			}
		} else if req.DeviceFinalScores {
			candidates, err = hipRunMLXQ4ProjectionSoftcapScoreKernelWithDeviceInputBufferSuppress(ctx, driver, finalNormBuffer, last.LMHeadProjection, last.FinalLogitSoftcap, req.FinalCandidateCount, req.SuppressTokens, req.AttentionWorkspace)
			if err != nil {
				return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
			}
			if len(candidates) > 0 {
				greedy = candidates[0]
			}
		} else if req.DeferFinalSampleRead {
			greedyDevice, err = hipRunMLXQ4ProjectionSoftcapGreedyTokenKernelWithDeviceInputBufferSuppressDevice(ctx, driver, finalNormBuffer, last.LMHeadProjection, last.FinalLogitSoftcap, req.FinalGreedyBuffer, req.SuppressTokens, req.AttentionWorkspace)
			if err != nil {
				return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
			}
		} else {
			greedy, greedyDevice, err = hipRunMLXQ4ProjectionSoftcapGreedyTokenKernelWithDeviceInputBufferSuppressResult(ctx, driver, finalNormBuffer, last.LMHeadProjection, last.FinalLogitSoftcap, req.FinalGreedyBuffer, req.SuppressTokens, req.AttentionWorkspace)
			if err != nil {
				return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
			}
		}
	} else {
		if hidden == nil && hiddenBuffer != nil {
			hidden, err = hipReadFloat32DeviceOutput(hiddenBuffer, hipGemma4Q4Layer0Operation, "final hidden output", last.HiddenSize)
			if err != nil {
				return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
			}
		}
		finalNorm, err := hipRunRMSNormKernelWithDeviceWeightConfig(ctx, driver, hidden, finalNormCfg)
		if err != nil {
			return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
		}
		logits, err = hipRunMLXQ4ProjectionKernelWithDeviceWeightConfig(ctx, driver, finalNorm, last.LMHeadProjection)
		if err != nil {
			return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
		}
		logits, err = hipGemma4Q4SoftcapLogits(logits, last.FinalLogitSoftcap)
		if err != nil {
			return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
		}
		greedy, err = hipRunGreedyKernel(ctx, driver, hipGreedySampleRequest{Logits: logits})
		if err != nil {
			return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
		}
	}
	if req.ReturnDeviceFinalHidden {
		if hiddenBuffer == nil {
			hiddenBuffer, err = hipUploadGemma4Q4Float32Input(driver, "Gemma4 q4 final hidden", hidden)
			if err != nil {
				return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
			}
			hiddenBufferBorrowed = false
		}
		deviceFinalHidden = hiddenBuffer
		deviceFinalHiddenBorrowed = hiddenBufferBorrowed
		if !deviceFinalHiddenBorrowed {
			hiddenBuffer = nil
			hiddenBufferBorrowed = false
		}
	}
	if nextDeviceState != nil {
		nextDeviceState.appendLayers = deviceAppendLayers
		nextDeviceState.remirrorLayers = deviceRemirrorLayers
		if err := hipFinalizeGemma4Q4ForwardDeviceState(req.PriorDeviceState, nextDeviceState); err != nil {
			return hipGemma4Q4ForwardResult{}, hipGemma4Q4DecodeState{}, err
		}
	}
	var labels map[string]string
	if !req.OmitLabels {
		labels = hipGemma4Q4ForwardLabels(cfg, req)
		if req.DeviceFinalSample {
			labels["gemma4_q4_final_sample"] = "device_q4_projection_softcap_greedy"
		}
		if req.DeviceFinalScores {
			labels["gemma4_q4_final_sample"] = "device_q4_projection_softcap_scores"
		}
		if req.DeviceFinalTopKSample {
			labels["gemma4_q4_final_sample"] = "device_q4_projection_softcap_topk_sample"
		}
		if req.DeferFinalSampleRead {
			labels["gemma4_q4_final_sample"] = "device_q4_projection_softcap_greedy_deferred"
		}
		if req.SkipFinalSample {
			labels["gemma4_q4_final_sample"] = "skipped"
		}
		if req.ReturnDeviceFinalHidden {
			labels["gemma4_q4_device_final_hidden"] = "returned"
			labels["gemma4_q4_device_final_hidden_borrowed"] = boolLabel(deviceFinalHiddenBorrowed)
		}
		if req.OmitDebugTensors {
			labels["gemma4_q4_debug_tensors"] = "omitted"
		}
		if req.DeviceKVAttention {
			labels["attention_kv_append_layers"] = core.Sprintf("%d", deviceAppendLayers)
			labels["attention_kv_remirror_layers"] = core.Sprintf("%d", deviceRemirrorLayers)
			labels["attention_kv_shared_device_layers"] = core.Sprintf("%d", deviceSharedLayers)
		}
		if cfg.KVSharedLayers > 0 {
			labels["gemma4_q4_kv_shared_layers"] = core.Sprintf("%d", cfg.KVSharedLayers)
			labels["gemma4_q4_kv_shared_runtime_layers"] = core.Sprintf("%d", sharedKVLayers)
		}
		if nextDeviceState != nil {
			labels["gemma4_q4_forward_device_state"] = "returned"
			labels["gemma4_q4_device_kv_append_layers"] = core.Sprintf("%d", deviceAppendLayers)
			labels["gemma4_q4_device_kv_remirror_layers"] = core.Sprintf("%d", deviceRemirrorLayers)
			labels["gemma4_q4_device_kv_shared_layers"] = core.Sprintf("%d", deviceSharedLayers)
		}
	}
	success = true
	result := hipGemma4Q4ForwardResult{
		LayerResults:              layerResults,
		Logits:                    logits,
		Greedy:                    greedy,
		GreedyDevice:              greedyDevice,
		Candidates:                candidates,
		DeviceFinalHidden:         deviceFinalHidden,
		DeviceFinalHiddenBorrowed: deviceFinalHiddenBorrowed,
		DeviceState:               nextDeviceState,
		Labels:                    labels,
	}
	if !req.OmitDebugTensors {
		result.Embedding = embedding
		result.ScaledEmbedding = scaledEmbedding
		result.FinalHidden = hidden
	}
	return result, nextState, nil
}

func hipRunGemma4Q4GreedyDecode(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, req hipGemma4Q4GreedyDecodeRequest) (hipGemma4Q4GreedyDecodeResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipGemma4Q4GreedyDecodeResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipGemma4Q4GreedyDecodeResult{}, core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if err := cfg.validate(); err != nil {
		return hipGemma4Q4GreedyDecodeResult{}, err
	}
	if err := req.validate(cfg); err != nil {
		return hipGemma4Q4GreedyDecodeResult{}, err
	}

	state := hipGemma4Q4DecodeState{}
	var deviceState *hipGemma4Q4DeviceDecodeState
	stepResults := make([]hipGemma4Q4ForwardResult, 0, len(req.PromptTokenIDs)+req.MaxNewTokens-1)
	position := req.Position
	var current hipGemma4Q4ForwardResult
	forwardOwnsDeviceState := req.MirrorDeviceKV && req.DeviceKVAttention
	for index, tokenID := range req.PromptTokenIDs {
		var err error
		previousState := state
		skipFinalSample := index+1 < len(req.PromptTokenIDs)
		current, state, err = hipRunGemma4Q4SingleTokenForwardWithStateInternal(ctx, driver, cfg, state, hipGemma4Q4ForwardRequest{
			TokenID:           tokenID,
			Position:          position,
			RoPEBase:          req.RoPEBase,
			Epsilon:           req.Epsilon,
			DeviceKVAttention: req.DeviceKVAttention,
			DeviceKVMode:      req.DeviceKVMode,
			EngineConfig:      req.EngineConfig,
			PriorDeviceState:  hipGemma4Q4PriorDeviceStateForForward(req, deviceState),
			ReturnDeviceState: forwardOwnsDeviceState,
			SkipFinalSample:   skipFinalSample,
		}, false)
		if err != nil {
			_ = deviceState.Close()
			return hipGemma4Q4GreedyDecodeResult{}, err
		}
		if req.MirrorDeviceKV {
			if forwardOwnsDeviceState {
				if current.DeviceState == nil {
					_ = deviceState.Close()
					return hipGemma4Q4GreedyDecodeResult{}, core.E(hipGemma4Q4Layer0Operation, "forward did not return device KV state", nil)
				}
				previousDeviceState := deviceState
				deviceState = current.DeviceState
				current.DeviceState = nil
				hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
			} else {
				previousDeviceState := deviceState
				nextDeviceState, err := hipUpdateGemma4Q4DeviceDecodeState(driver, cfg, previousState, state, deviceState, req.DeviceKVMode)
				if err != nil {
					_ = deviceState.Close()
					return hipGemma4Q4GreedyDecodeResult{}, err
				}
				deviceState = nextDeviceState
				hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
			}
		}
		stepResults = append(stepResults, current)
		position++
	}

	generated := make([]hipGreedySampleResult, 0, req.MaxNewTokens)
	for len(generated) < req.MaxNewTokens {
		generated = append(generated, current.Greedy)
		if len(generated) == req.MaxNewTokens {
			break
		}
		var err error
		previousState := state
		current, state, err = hipRunGemma4Q4SingleTokenForwardWithStateInternal(ctx, driver, cfg, state, hipGemma4Q4ForwardRequest{
			TokenID:           int32(current.Greedy.TokenID),
			Position:          position,
			RoPEBase:          req.RoPEBase,
			Epsilon:           req.Epsilon,
			DeviceKVAttention: req.DeviceKVAttention,
			DeviceKVMode:      req.DeviceKVMode,
			EngineConfig:      req.EngineConfig,
			PriorDeviceState:  hipGemma4Q4PriorDeviceStateForForward(req, deviceState),
			ReturnDeviceState: forwardOwnsDeviceState,
		}, false)
		if err != nil {
			_ = deviceState.Close()
			return hipGemma4Q4GreedyDecodeResult{}, err
		}
		if req.MirrorDeviceKV {
			if forwardOwnsDeviceState {
				if current.DeviceState == nil {
					_ = deviceState.Close()
					return hipGemma4Q4GreedyDecodeResult{}, core.E(hipGemma4Q4Layer0Operation, "forward did not return device KV state", nil)
				}
				previousDeviceState := deviceState
				deviceState = current.DeviceState
				current.DeviceState = nil
				hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
			} else {
				previousDeviceState := deviceState
				nextDeviceState, err := hipUpdateGemma4Q4DeviceDecodeState(driver, cfg, previousState, state, deviceState, req.DeviceKVMode)
				if err != nil {
					_ = deviceState.Close()
					return hipGemma4Q4GreedyDecodeResult{}, err
				}
				deviceState = nextDeviceState
				hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
			}
		}
		stepResults = append(stepResults, current)
		position++
	}

	labels := hipGemma4Q4GreedyDecodeLabels(cfg, req, state)
	if deviceState != nil {
		for key, value := range deviceState.Labels() {
			labels[key] = value
		}
	}
	return hipGemma4Q4GreedyDecodeResult{
		Generated:   generated,
		StepResults: stepResults,
		State:       state,
		DeviceState: deviceState,
		Labels:      labels,
	}, nil
}

func hipGemma4Q4PriorDeviceStateForForward(req hipGemma4Q4GreedyDecodeRequest, state *hipGemma4Q4DeviceDecodeState) *hipGemma4Q4DeviceDecodeState {
	if !req.MirrorDeviceKV || !req.DeviceKVAttention {
		return nil
	}
	return state
}

func hipRunGemma4Q4DecoderLayer(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input []float32, req hipGemma4Q4DecoderLayerRequest) (hipGemma4Q4DecoderLayerResult, error) {
	return hipRunGemma4Q4DecoderLayerInternal(ctx, driver, cfg, input, req, true)
}

func hipRunGemma4Q4DecoderLayerInternal(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input []float32, req hipGemma4Q4DecoderLayerRequest, validate bool) (hipGemma4Q4DecoderLayerResult, error) {
	return hipRunGemma4Q4DecoderLayerInternalWithDeviceInput(ctx, driver, cfg, input, nil, req, validate)
}

func hipRunGemma4Q4DecoderLayerInternalWithDeviceInput(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input []float32, inputDevice *hipDeviceByteBuffer, req hipGemma4Q4DecoderLayerRequest, validate bool) (hipGemma4Q4DecoderLayerResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipGemma4Q4DecoderLayerResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipGemma4Q4DecoderLayerResult{}, core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if validate {
		if err := cfg.validate(); err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		if err := req.validate(cfg); err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
	}
	ropeBase, err := req.effectiveRoPEBase(cfg)
	if err != nil {
		return hipGemma4Q4DecoderLayerResult{}, err
	}
	var inputBuffer *hipDeviceByteBuffer
	if inputDevice != nil {
		if inputDevice.Pointer() == 0 || inputDevice.Count() != cfg.HiddenSize || inputDevice.SizeBytes() != uint64(cfg.HiddenSize*4) {
			return hipGemma4Q4DecoderLayerResult{}, core.E(hipGemma4Q4Layer0Operation, "decoder layer device input shape mismatch", nil)
		}
		inputBuffer = inputDevice
	} else {
		if len(input) != cfg.HiddenSize {
			return hipGemma4Q4DecoderLayerResult{}, core.E(hipGemma4Q4Layer0Operation, "decoder layer input length must match hidden size", nil)
		}
		var err error
		inputBuffer, err = hipUploadGemma4Q4Float32Input(driver, "Gemma4 q4 decoder layer input", input)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		defer inputBuffer.Close()
	}
	inputNormCfg := cfg.InputNorm
	inputNormCfg.Epsilon = req.Epsilon
	var layerInputBuffer *hipDeviceByteBuffer
	layerInputBorrowed := false
	if req.LayerInputDevice != nil {
		if req.LayerInputDevice.Pointer() == 0 || req.LayerInputDevice.Count() != cfg.HiddenSize || req.LayerInputDevice.SizeBytes() != uint64(cfg.HiddenSize*4) {
			return hipGemma4Q4DecoderLayerResult{}, core.E(hipGemma4Q4Layer0Operation, "decoder layer precomputed input norm shape mismatch", nil)
		}
		layerInputBuffer = req.LayerInputDevice
		layerInputBorrowed = true
	} else if req.AttentionWorkspace != nil && req.OmitDebugTensors {
		layerInputBuffer, err = req.AttentionWorkspace.EnsureRMSNormOutput(driver, inputNormCfg.Count)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		if err := hipRunRMSNormDeviceToDeviceKernelWithWorkspace(ctx, driver, inputBuffer.Pointer(), inputBuffer.SizeBytes(), layerInputBuffer.Pointer(), layerInputBuffer.SizeBytes(), inputNormCfg, req.AttentionWorkspace); err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		layerInputBorrowed = true
	} else {
		layerInputBuffer, err = hipRunRMSNormKernelWithDeviceInputWeightConfig(ctx, driver, inputBuffer, inputNormCfg)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
	}
	if !layerInputBorrowed {
		defer layerInputBuffer.Close()
	}
	var layerInput []float32
	ensureLayerInput := func() ([]float32, error) {
		if layerInput != nil {
			return layerInput, nil
		}
		read, err := hipReadFloat32DeviceOutput(layerInputBuffer, hipGemma4Q4Layer0Operation, "layer input output", cfg.HiddenSize)
		if err != nil {
			return nil, err
		}
		layerInput = read
		return layerInput, nil
	}
	var ropeQueries [][]float32
	var ropeQueryBuffer *hipDeviceByteBuffer
	var qkvOutputBuffer *hipDeviceByteBuffer
	var queryBuffer *hipDeviceByteBuffer
	var keyBuffer *hipDeviceByteBuffer
	var valueBuffer *hipDeviceByteBuffer
	var queryBufferView hipDeviceByteBuffer
	var keyBufferView hipDeviceByteBuffer
	var valueBufferView hipDeviceByteBuffer
	projectLocalKV := req.SharedDeviceKV == nil && len(req.SharedKeys) == 0
	if projectLocalKV &&
		!cfg.AttentionKEqV &&
		cfg.QueryProjection.Cols == cfg.KeyProjection.Cols && cfg.QueryProjection.Cols == cfg.ValueProjection.Cols &&
		cfg.QueryProjection.GroupSize == cfg.KeyProjection.GroupSize && cfg.QueryProjection.GroupSize == cfg.ValueProjection.GroupSize {
		if req.AttentionWorkspace != nil && req.OmitDebugTensors {
			qkvCount := cfg.QueryProjection.Rows + cfg.KeyProjection.Rows + cfg.ValueProjection.Rows
			qkvOutputBuffer, err = req.AttentionWorkspace.EnsureQKVOutput(driver, qkvCount)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			queryBufferView, keyBufferView, valueBufferView, err = hipRunMLXQ4TripleProjectionKernelWithDeviceInputViewsOutputWithWorkspace(ctx, driver, layerInputBuffer, cfg.QueryProjection, cfg.KeyProjection, cfg.ValueProjection, qkvOutputBuffer, req.AttentionWorkspace)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
		} else {
			qkvOutputBuffer, queryBufferView, keyBufferView, valueBufferView, err = hipRunMLXQ4TripleProjectionKernelWithDeviceInputViews(ctx, driver, layerInputBuffer, cfg.QueryProjection, cfg.KeyProjection, cfg.ValueProjection)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			defer qkvOutputBuffer.Close()
		}
		queryBuffer = &queryBufferView
		keyBuffer = &keyBufferView
		valueBuffer = &valueBufferView
	} else if projectLocalKV &&
		cfg.AttentionKEqV &&
		cfg.QueryProjection.Cols == cfg.KeyProjection.Cols &&
		cfg.QueryProjection.GroupSize == cfg.KeyProjection.GroupSize {
		if req.AttentionWorkspace != nil && req.OmitDebugTensors {
			qkvCount := cfg.QueryProjection.Rows + cfg.KeyProjection.Rows
			qkvOutputBuffer, err = req.AttentionWorkspace.EnsureQKVOutput(driver, qkvCount)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			queryBufferView, keyBufferView, err = hipRunMLXQ4PairProjectionKernelWithDeviceInputViewsOutputWithWorkspace(ctx, driver, layerInputBuffer, cfg.QueryProjection, cfg.KeyProjection, qkvOutputBuffer, req.AttentionWorkspace)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
		} else {
			qkvOutputBuffer, queryBufferView, keyBufferView, err = hipRunMLXQ4PairProjectionKernelWithDeviceInputViews(ctx, driver, layerInputBuffer, cfg.QueryProjection, cfg.KeyProjection)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			defer qkvOutputBuffer.Close()
		}
		queryBuffer = &queryBufferView
		keyBuffer = &keyBufferView
		valueBuffer = keyBuffer
	} else {
		if req.AttentionWorkspace != nil && req.OmitDebugTensors {
			queryBuffer, err = req.AttentionWorkspace.EnsureProjectionOutput(driver, cfg.QueryProjection.Rows)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			if err := hipRunMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace(ctx, driver, layerInputBuffer, cfg.QueryProjection, queryBuffer, req.AttentionWorkspace); err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
		} else {
			queryBuffer, err = hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, layerInputBuffer, cfg.QueryProjection)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			defer queryBuffer.Close()
		}
	}
	queryNormCfg := hipGemma4Q4RoPENormConfig(cfg.QueryNorm, req.Epsilon, cfg.HeadDim)
	ropeFrequencyDim, ropeRotaryCount := hipGemma4Q4RoPEKernelDims(cfg)
	ropeFrequencyScale := cfg.effectiveRoPEFrequencyScale()
	pairLocalQKNormRoPE := projectLocalKV && req.AttentionWorkspace != nil && req.OmitDebugTensors
	if req.AttentionWorkspace != nil && req.OmitDebugTensors {
		ropeQueryBuffer, err = req.AttentionWorkspace.EnsureRMSRoPEOutput(driver, queryBuffer.Count())
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		if !pairLocalQKNormRoPE {
			if err := hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutputFrequencyScaleWithWorkspace(ctx, driver, queryBuffer, queryNormCfg, cfg.QueryHeads, req.Position, ropeBase, ropeFrequencyDim, ropeRotaryCount, ropeFrequencyScale, ropeQueryBuffer, req.AttentionWorkspace); err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
		}
	} else {
		ropeQueryBuffer, err = hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigFrequencyScale(ctx, driver, queryBuffer, queryNormCfg, cfg.QueryHeads, req.Position, ropeBase, ropeFrequencyDim, ropeRotaryCount, ropeFrequencyScale)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		defer ropeQueryBuffer.Close()
	}
	var ropeKey []float32
	var value []float32
	var ropeKeyDevice *hipDeviceByteBuffer
	var valueDevice *hipDeviceByteBuffer
	var updatedKeys []float32
	var updatedValues []float32
	keyHeads := firstPositiveInt(cfg.KeyHeads, 1)
	kvDim := cfg.keyValueDim()
	if req.SharedDeviceKV != nil {
		// Shared-KV layers use the source layer's current device cache directly in
		// generation mode. Debug/host-state paths continue to use SharedKeys.
	} else if len(req.SharedKeys) > 0 {
		updatedKeys = append([]float32(nil), req.SharedKeys...)
		updatedValues = append([]float32(nil), req.SharedValues...)
		ropeKey, value, err = hipGemma4Q4LastKVToken(updatedKeys, updatedValues, kvDim)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
	} else if cfg.RoPERotaryDim == cfg.HeadDim {
		if keyBuffer == nil {
			keyBuffer, err = hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, layerInputBuffer, cfg.KeyProjection)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			defer keyBuffer.Close()
		}
		if valueBuffer == nil {
			if cfg.AttentionKEqV {
				valueBuffer = keyBuffer
			} else {
				valueBuffer, err = hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, layerInputBuffer, cfg.ValueProjection)
				if err != nil {
					return hipGemma4Q4DecoderLayerResult{}, err
				}
				defer valueBuffer.Close()
			}
		}
		useDeviceKVToken := req.OmitHostKV && req.DeviceKVAttention
		if req.AttentionWorkspace != nil && req.OmitDebugTensors {
			valueDevice, err = req.AttentionWorkspace.EnsureRMSNoScaleOutput(driver, valueBuffer.Count())
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			if err := hipRunGemma4Q4ValueNormDeviceKernelOutputWithWorkspace(ctx, driver, valueBuffer, valueDevice, cfg.HeadDim, keyHeads, req.Epsilon, req.AttentionWorkspace); err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
		} else {
			valueDevice, err = hipRunGemma4Q4ValueNormDeviceKernel(ctx, driver, valueBuffer, cfg.HeadDim, keyHeads, req.Epsilon)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			defer valueDevice.Close()
		}
		if !useDeviceKVToken {
			value, err = hipReadFloat32DeviceOutput(valueDevice, hipGemma4Q4Layer0Operation, "RMSNormNoScale output", valueDevice.Count())
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
		}
		keyNormCfg := hipGemma4Q4RoPENormConfig(cfg.KeyNorm, req.Epsilon, cfg.HeadDim)
		ropeKeyBuffer := (*hipDeviceByteBuffer)(nil)
		if req.AttentionWorkspace != nil && req.OmitDebugTensors {
			ropeKeyBuffer, err = req.AttentionWorkspace.EnsureKeyRMSRoPEOutput(driver, keyBuffer.Count())
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			if pairLocalQKNormRoPE {
				if err := hipRunRMSNormRoPEHeadsPairKernelWithDeviceInputWeightConfigOutputFrequencyScaleWithWorkspace(ctx, driver, queryBuffer, keyBuffer, queryNormCfg, keyNormCfg, cfg.QueryHeads, keyHeads, req.Position, ropeBase, ropeFrequencyDim, ropeRotaryCount, ropeFrequencyScale, ropeQueryBuffer, ropeKeyBuffer, req.AttentionWorkspace); err != nil {
					return hipGemma4Q4DecoderLayerResult{}, err
				}
			} else {
				if err := hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutputFrequencyScaleWithWorkspace(ctx, driver, keyBuffer, keyNormCfg, keyHeads, req.Position, ropeBase, 0, 0, ropeFrequencyScale, ropeKeyBuffer, req.AttentionWorkspace); err != nil {
					return hipGemma4Q4DecoderLayerResult{}, err
				}
			}
		} else {
			ropeKeyBuffer, err = hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigFrequencyScale(ctx, driver, keyBuffer, keyNormCfg, keyHeads, req.Position, ropeBase, 0, 0, ropeFrequencyScale)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			defer ropeKeyBuffer.Close()
		}
		ropeKeyDevice = ropeKeyBuffer
		if !useDeviceKVToken {
			ropeKey, err = hipReadFloat32DeviceOutput(ropeKeyBuffer, hipGemma4Q4Layer0Operation, "RoPE key output", kvDim)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
		}
		if !req.OmitHostKV {
			updatedKeys = hipGemma4Q4AppendKV(req.PriorKeys, ropeKey)
			updatedValues = hipGemma4Q4AppendKV(req.PriorValues, value)
			updatedKeys, updatedValues = hipGemma4Q4TrimKVWindow(updatedKeys, updatedValues, kvDim, cfg.SlidingWindow)
		}
	} else {
		if keyBuffer == nil {
			keyBuffer, err = hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, layerInputBuffer, cfg.KeyProjection)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			defer keyBuffer.Close()
		}
		if valueBuffer == nil {
			if cfg.AttentionKEqV {
				valueBuffer = keyBuffer
			} else {
				valueBuffer, err = hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, layerInputBuffer, cfg.ValueProjection)
				if err != nil {
					return hipGemma4Q4DecoderLayerResult{}, err
				}
				defer valueBuffer.Close()
			}
		}
		useDeviceKVToken := req.OmitHostKV && req.DeviceKVAttention
		if req.AttentionWorkspace != nil && req.OmitDebugTensors {
			valueDevice, err = req.AttentionWorkspace.EnsureRMSNoScaleOutput(driver, valueBuffer.Count())
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			if err := hipRunGemma4Q4ValueNormDeviceKernelOutputWithWorkspace(ctx, driver, valueBuffer, valueDevice, cfg.HeadDim, keyHeads, req.Epsilon, req.AttentionWorkspace); err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
		} else {
			valueDevice, err = hipRunGemma4Q4ValueNormDeviceKernel(ctx, driver, valueBuffer, cfg.HeadDim, keyHeads, req.Epsilon)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			defer valueDevice.Close()
		}
		if !useDeviceKVToken {
			value, err = hipReadFloat32DeviceOutput(valueDevice, hipGemma4Q4Layer0Operation, "RMSNormNoScale output", valueDevice.Count())
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
		}
		keyNormCfg := hipGemma4Q4RoPENormConfig(cfg.KeyNorm, req.Epsilon, cfg.HeadDim)
		ropeKeyBuffer := (*hipDeviceByteBuffer)(nil)
		if req.AttentionWorkspace != nil && req.OmitDebugTensors {
			ropeKeyBuffer, err = req.AttentionWorkspace.EnsureKeyRMSRoPEOutput(driver, keyBuffer.Count())
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			if pairLocalQKNormRoPE {
				if err := hipRunRMSNormRoPEHeadsPairKernelWithDeviceInputWeightConfigOutputFrequencyScaleWithWorkspace(ctx, driver, queryBuffer, keyBuffer, queryNormCfg, keyNormCfg, cfg.QueryHeads, keyHeads, req.Position, ropeBase, ropeFrequencyDim, ropeRotaryCount, ropeFrequencyScale, ropeQueryBuffer, ropeKeyBuffer, req.AttentionWorkspace); err != nil {
					return hipGemma4Q4DecoderLayerResult{}, err
				}
			} else {
				if err := hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutputFrequencyScaleWithWorkspace(ctx, driver, keyBuffer, keyNormCfg, keyHeads, req.Position, ropeBase, cfg.HeadDim, cfg.RoPERotaryDim, ropeFrequencyScale, ropeKeyBuffer, req.AttentionWorkspace); err != nil {
					return hipGemma4Q4DecoderLayerResult{}, err
				}
			}
		} else {
			ropeKeyBuffer, err = hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigFrequencyScale(ctx, driver, keyBuffer, keyNormCfg, keyHeads, req.Position, ropeBase, cfg.HeadDim, cfg.RoPERotaryDim, ropeFrequencyScale)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			defer ropeKeyBuffer.Close()
		}
		ropeKeyDevice = ropeKeyBuffer
		if !useDeviceKVToken {
			ropeKey, err = hipReadFloat32DeviceOutput(ropeKeyBuffer, hipGemma4Q4Layer0Operation, "RoPE key output", kvDim)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
		}
		if !req.OmitHostKV {
			updatedKeys = hipGemma4Q4AppendKV(req.PriorKeys, ropeKey)
			updatedValues = hipGemma4Q4AppendKV(req.PriorValues, value)
			updatedKeys, updatedValues = hipGemma4Q4TrimKVWindow(updatedKeys, updatedValues, kvDim, cfg.SlidingWindow)
		}
	}

	var deviceKV *rocmDeviceKVCache
	var descriptorTable *rocmDeviceKVDescriptorTable
	borrowedDeviceKV := false
	borrowedDescriptorTable := false
	deviceKVAttention := ""
	var retainedDeviceLayer hipGemma4Q4DeviceLayerKVState
	retainedDeviceLayerValid := false
	retainedDeviceLayerSuccess := false
	defer func() {
		if retainedDeviceLayerValid && !retainedDeviceLayerSuccess {
			_ = retainedDeviceLayer.Close()
		}
	}()
	if req.DeviceKVAttention {
		borrowedPageCount := 0
		if req.SharedDeviceKV != nil {
			if req.SharedDeviceKV.closed {
				return hipGemma4Q4DecoderLayerResult{}, core.E(hipGemma4Q4Layer0Operation, "shared device KV source is closed", nil)
			}
			deviceKV = req.SharedDeviceKV
			borrowedDeviceKV = true
			if req.SharedDescriptorTable != nil {
				if req.SharedDescriptorTable.closed || req.SharedDescriptorTable.Pointer() == 0 {
					return hipGemma4Q4DecoderLayerResult{}, core.E(hipGemma4Q4Layer0Operation, "shared device KV descriptor table is closed", nil)
				}
				descriptorTable = req.SharedDescriptorTable
				borrowedDescriptorTable = true
			}
			deviceKVAttention = "shared_device_kv"
		} else if req.OmitHostKV && req.PriorDeviceKV != nil && ropeKeyDevice != nil && valueDevice != nil {
			deviceKV, err = req.PriorDeviceKV.withAppendedDeviceTokenWindowWithWorkspaceAndEngineConfig(ctx, ropeKeyDevice, valueDevice, cfg.SlidingWindow, req.AttentionWorkspace, req.EngineConfig)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			deviceKVAttention = "append_existing_device"
			borrowedPageCount = req.PriorDeviceKV.PageCount()
		} else if req.OmitHostKV && ropeKeyDevice != nil && valueDevice != nil {
			deviceKV, err = newROCmDeviceKVCacheFromDeviceTokenWithWorkspace(ctx, driver, firstNonEmptyString(req.DeviceKVMode, rocmKVCacheModeFP16), req.EngineConfig.deviceKVBlockSizeForSlidingWindow(cfg.SlidingWindow), ropeKeyDevice, valueDevice, cfg.SlidingWindow, req.AttentionWorkspace)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			deviceKVAttention = "new_device_kv"
		} else if req.OmitHostKV {
			return hipGemma4Q4DecoderLayerResult{}, core.E(hipGemma4Q4Layer0Operation, "device-only KV path requires device token buffers or shared device KV", nil)
		} else if req.PriorDeviceKV != nil && hipGemma4Q4LayerStateCanAppendDeviceKV(cfg,
			hipGemma4Q4LayerKVState{Keys: req.PriorKeys, Values: req.PriorValues},
			hipGemma4Q4LayerKVState{Keys: updatedKeys, Values: updatedValues}) {
			deviceKV, err = req.PriorDeviceKV.withAppendedToken(ropeKey, value)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			deviceKVAttention = "append_existing_device"
			borrowedPageCount = req.PriorDeviceKV.PageCount()
		} else {
			host, err := newROCmKVCache(firstNonEmptyString(req.DeviceKVMode, rocmKVCacheModeFP16), req.EngineConfig.deviceKVBlockSizeForSlidingWindow(cfg.SlidingWindow))
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			keys := updatedKeys
			values := updatedValues
			if req.OmitHostKV {
				keys = ropeKey
				values = value
			}
			if err := host.AppendVectors(0, kvDim, kvDim, keys, values); err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			deviceKV, err = host.MirrorToDevice(driver)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			deviceKVAttention = "remirror_host_kv"
		}
		if deviceKVAttention == "append_existing_device" && req.PriorDeviceKV != nil && req.PriorDescriptorTable != nil {
			descriptorTable, err = deviceKV.KernelDescriptorTableFromAppendedTokenWithWorkspace(ctx, req.PriorDeviceKV, req.PriorDescriptorTable, req.AttentionWorkspace)
		}
		if descriptorTable == nil && err == nil {
			descriptorTable, err = deviceKV.kernelDescriptorTableLabeled("rocm.KVCache.DeviceDescriptor", deviceKVAttention)
		}
		if err != nil {
			if borrowedDeviceKV {
				// Source owner layer keeps the shared cache alive.
			} else if borrowedPageCount > 0 {
				_ = deviceKV.closePagesFrom(borrowedPageCount)
			} else {
				_ = deviceKV.Close()
			}
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		launch, err := deviceKV.KernelLaunchDescriptor(descriptorTable)
		if err != nil {
			if !borrowedDescriptorTable {
				_ = descriptorTable.Close()
			}
			if borrowedDeviceKV {
				// Source owner layer keeps the shared cache alive.
			} else if borrowedPageCount > 0 {
				_ = deviceKV.closePagesFrom(borrowedPageCount)
			} else {
				_ = deviceKV.Close()
			}
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		if req.KeepDeviceKV {
			retainedDeviceLayer = hipGemma4Q4DeviceLayerKVState{
				cache:                   deviceKV,
				descriptorTable:         descriptorTable,
				launch:                  launch,
				borrowedCache:           borrowedDeviceKV,
				borrowedDescriptorTable: borrowedDescriptorTable,
			}
			retainedDeviceLayerValid = true
		} else {
			if !borrowedDescriptorTable {
				defer descriptorTable.Close()
			}
			if borrowedDeviceKV {
				// Source owner layer keeps the shared cache alive.
			} else if borrowedPageCount > 0 {
				defer deviceKV.closePagesFrom(borrowedPageCount)
			} else {
				defer deviceKV.Close()
			}
		}
	}

	var attentionOutputBuffer *hipDeviceByteBuffer
	if req.AttentionWorkspace != nil && req.OmitDebugTensors {
		attentionOutputBuffer, err = req.AttentionWorkspace.EnsureAttentionOutput(driver, cfg.QueryHeads, cfg.HeadDim)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
	} else {
		attentionOutputBuffer, err = hipAllocateByteBuffer(driver, hipGemma4Q4Layer0Operation, "Gemma4 q4 attention concat output", uint64(cfg.QueryHeads*cfg.HeadDim*4), cfg.QueryHeads*cfg.HeadDim)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		defer attentionOutputBuffer.Close()
	}
	if ropeQueryBuffer == nil {
		ropeQueryConcat := make([]float32, 0, cfg.QueryHeads*cfg.HeadDim)
		for _, ropeQuery := range ropeQueries {
			ropeQueryConcat = append(ropeQueryConcat, ropeQuery...)
		}
		ropeQueryBuffer, err = hipUploadGemma4Q4Float32Input(driver, "Gemma4 q4 RoPE query concat", ropeQueryConcat)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		defer ropeQueryBuffer.Close()
	}
	attentionReq := hipAttentionRequest{
		QueryDim:   cfg.HeadDim,
		KeyHeads:   keyHeads,
		Keys:       updatedKeys,
		Values:     updatedValues,
		WindowSize: cfg.SlidingWindow,
		Scale:      hipGemma4Q4AttentionScale(cfg.HeadDim),
	}
	if req.DeviceKVAttention {
		attentionReq.Keys = nil
		attentionReq.Values = nil
		attentionReq.DeviceKV = deviceKV
		attentionReq.DescriptorTable = descriptorTable
	}
	if err := hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernelWithWorkspace(ctx, driver, attentionReq, ropeQueryBuffer, cfg.QueryHeads, attentionOutputBuffer, req.AttentionWorkspace); err != nil {
		return hipGemma4Q4DecoderLayerResult{}, err
	}
	var attentionProjectionBuffer *hipDeviceByteBuffer
	if req.AttentionWorkspace != nil && req.OmitDebugTensors {
		attentionProjectionBuffer, err = req.AttentionWorkspace.EnsureProjectionOutput(driver, cfg.OutputProjection.Rows)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		if err := hipRunMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace(ctx, driver, attentionOutputBuffer, cfg.OutputProjection, attentionProjectionBuffer, req.AttentionWorkspace); err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
	} else {
		attentionProjectionBuffer, err = hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, attentionOutputBuffer, cfg.OutputProjection)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		defer attentionProjectionBuffer.Close()
	}
	var attentionProjection []float32
	var attentionOutput []float32
	if !req.OmitDebugTensors {
		attentionProjection, err = hipReadFloat32DeviceOutput(attentionProjectionBuffer, hipGemma4Q4Layer0Operation, "attention projection output", cfg.HiddenSize)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		attentionOutput, err = hipReadFloat32DeviceOutput(attentionOutputBuffer, hipGemma4Q4Layer0Operation, "attention concat output", cfg.QueryHeads*cfg.HeadDim)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
	}
	postAttentionNormCfg := cfg.PostAttentionNorm
	postAttentionNormCfg.Epsilon = req.Epsilon
	preFeedForwardNormCfg := cfg.PreFeedForwardNorm
	preFeedForwardNormCfg.Epsilon = req.Epsilon
	var attentionResidualBuffer *hipDeviceByteBuffer
	var preFeedForwardBuffer *hipDeviceByteBuffer
	if req.AttentionWorkspace != nil && req.OmitDebugTensors {
		attentionResidualBuffer, err = req.AttentionWorkspace.EnsureRMSResidualOutput(driver, postAttentionNormCfg.Count)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		preFeedForwardBuffer, err = req.AttentionWorkspace.EnsureRMSNormOutput(driver, preFeedForwardNormCfg.Count)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		if err := hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfigOutputWithWorkspace(ctx, driver, attentionProjectionBuffer, inputBuffer, postAttentionNormCfg, preFeedForwardNormCfg, attentionResidualBuffer, preFeedForwardBuffer, 1, req.AttentionWorkspace); err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
	} else {
		attentionResidualBuffer, preFeedForwardBuffer, err = hipRunRMSNormResidualAddNormKernelWithDeviceInputWeightConfig(ctx, driver, attentionProjectionBuffer, inputBuffer, postAttentionNormCfg, preFeedForwardNormCfg)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		defer attentionResidualBuffer.Close()
		defer preFeedForwardBuffer.Close()
	}
	var attentionResidual []float32
	if !req.OmitDebugTensors {
		attentionResidual, err = hipReadFloat32DeviceOutput(attentionResidualBuffer, hipGemma4Q4Layer0Operation, "attention residual output", cfg.HiddenSize)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
	}
	var mlpOutputBuffer *hipDeviceByteBuffer
	if cfg.MoE != nil {
		mlpOutputBuffer, err = hipRunGemma4MoEDeviceMLPWithWorkspace(ctx, driver, attentionResidualBuffer, preFeedForwardBuffer, cfg, req.Epsilon, req.AttentionWorkspace)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		defer mlpOutputBuffer.Close()
	} else if req.AttentionWorkspace != nil && req.OmitDebugTensors {
		mlpOutputBuffer, err = req.AttentionWorkspace.EnsureProjectionOutput(driver, cfg.DownProjection.Rows)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		if err := hipRunGemma4Q4DeviceGELUTanhMLPWithDeviceInputOutput(ctx, driver, preFeedForwardBuffer, cfg.GateProjection, cfg.UpProjection, cfg.DownProjection, mlpOutputBuffer, req.AttentionWorkspace); err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
	} else {
		mlpOutputBuffer, err = hipRunGemma4Q4DeviceGELUTanhMLPWithDeviceInput(ctx, driver, preFeedForwardBuffer, cfg.GateProjection, cfg.UpProjection, cfg.DownProjection)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		defer mlpOutputBuffer.Close()
	}
	var mlpOutput []float32
	if !req.OmitDebugTensors {
		mlpOutput, err = hipReadFloat32DeviceOutput(mlpOutputBuffer, hipGemma4Q4Layer0Operation, "GELU tanh MLP output", cfg.DownProjection.Rows)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
	}
	postFeedForwardNormCfg := cfg.PostFeedForwardNorm
	postFeedForwardNormCfg.Epsilon = req.Epsilon
	var returnedFinalHiddenBuffer *hipDeviceByteBuffer
	var nextLayerInputBuffer *hipDeviceByteBuffer
	nextLayerInputBorrowed := false
	nextLayerInputReturned := false
	defer func() {
		if nextLayerInputBuffer != nil && !nextLayerInputReturned && !nextLayerInputBorrowed {
			_ = nextLayerInputBuffer.Close()
		}
	}()
	layerScalar := cfg.effectiveLayerScalar()
	hasPerLayerInput := req.PerLayerInputDevice != nil || len(req.PerLayerInput) > 0
	postFeedForwardOutputScale := float32(1)
	if !hasPerLayerInput {
		postFeedForwardOutputScale = layerScalar
	}
	var finalHiddenBuffer *hipDeviceByteBuffer
	finalHiddenBorrowed := false
	nextInputNorm, hasNextInputNorm := req.nextInputNormConfig()
	if hasNextInputNorm && !hasPerLayerInput {
		if req.OmitDebugTensors && req.FinalHiddenOutput != nil && req.NextLayerInputOutput != nil {
			if err := hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfigOutputWithWorkspace(ctx, driver, mlpOutputBuffer, attentionResidualBuffer, postFeedForwardNormCfg, nextInputNorm, req.FinalHiddenOutput, req.NextLayerInputOutput, postFeedForwardOutputScale, req.AttentionWorkspace); err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			finalHiddenBuffer = req.FinalHiddenOutput
			finalHiddenBorrowed = true
			nextLayerInputBuffer = req.NextLayerInputOutput
			nextLayerInputBorrowed = true
		} else {
			finalHiddenBuffer, nextLayerInputBuffer, err = hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfig(ctx, driver, mlpOutputBuffer, attentionResidualBuffer, postFeedForwardNormCfg, nextInputNorm, postFeedForwardOutputScale)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
		}
	} else if hasPerLayerInput && req.AttentionWorkspace != nil && req.OmitDebugTensors {
		finalHiddenBuffer, err = req.AttentionWorkspace.EnsureIntermediateOutput(driver, postFeedForwardNormCfg.Count)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		if err := hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfigOutputWithWorkspace(ctx, driver, mlpOutputBuffer, attentionResidualBuffer, postFeedForwardNormCfg, finalHiddenBuffer, postFeedForwardOutputScale, req.AttentionWorkspace); err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		finalHiddenBorrowed = true
	} else {
		if req.OmitDebugTensors && req.FinalHiddenOutput != nil {
			if err := hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfigOutputWithWorkspace(ctx, driver, mlpOutputBuffer, attentionResidualBuffer, postFeedForwardNormCfg, req.FinalHiddenOutput, postFeedForwardOutputScale, req.AttentionWorkspace); err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			finalHiddenBuffer = req.FinalHiddenOutput
			finalHiddenBorrowed = true
		} else {
			finalHiddenBuffer, err = hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfig(ctx, driver, mlpOutputBuffer, attentionResidualBuffer, postFeedForwardNormCfg, postFeedForwardOutputScale)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
		}
	}
	defer func(buffer *hipDeviceByteBuffer, borrowed bool) {
		if buffer != returnedFinalHiddenBuffer && !borrowed {
			_ = buffer.Close()
		}
	}(finalHiddenBuffer, finalHiddenBorrowed)
	if hasPerLayerInput {
		var perLayerProjectionBuffer *hipDeviceByteBuffer
		if req.PerLayerInputDevice != nil && req.AttentionWorkspace != nil && req.OmitDebugTensors {
			perLayerProjectionBuffer, err = req.AttentionWorkspace.EnsureProjectionOutput(driver, cfg.PerLayerInput.Projection.Rows)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			if err := hipRunGemma4Q4DeviceGELUTanhProjectionWithDeviceMultiplierOutput(ctx, driver, finalHiddenBuffer, req.PerLayerInputDevice, cfg.PerLayerInput.InputGate, cfg.PerLayerInput.Projection, perLayerProjectionBuffer, req.AttentionWorkspace); err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
		} else {
			if req.PerLayerInputDevice != nil {
				perLayerProjectionBuffer, err = hipRunGemma4Q4DeviceGELUTanhProjectionWithDeviceMultiplier(ctx, driver, finalHiddenBuffer, req.PerLayerInputDevice, cfg.PerLayerInput.InputGate, cfg.PerLayerInput.Projection)
			} else {
				perLayerProjectionBuffer, err = hipRunGemma4Q4DeviceGELUTanhProjectionWithDeviceInput(ctx, driver, finalHiddenBuffer, req.PerLayerInput, cfg.PerLayerInput.InputGate, cfg.PerLayerInput.Projection)
			}
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
			defer perLayerProjectionBuffer.Close()
		}
		perLayerNormCfg := cfg.PerLayerInput.PostInputNorm
		perLayerNormCfg.Epsilon = req.Epsilon
		var perLayerFinalHiddenBuffer *hipDeviceByteBuffer
		perLayerFinalHiddenBorrowed := false
		if hasNextInputNorm {
			if req.OmitDebugTensors && req.FinalHiddenOutput != nil && req.NextLayerInputOutput != nil {
				if err := hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfigOutputWithWorkspace(ctx, driver, perLayerProjectionBuffer, finalHiddenBuffer, perLayerNormCfg, nextInputNorm, req.FinalHiddenOutput, req.NextLayerInputOutput, layerScalar, req.AttentionWorkspace); err != nil {
					return hipGemma4Q4DecoderLayerResult{}, err
				}
				perLayerFinalHiddenBuffer = req.FinalHiddenOutput
				perLayerFinalHiddenBorrowed = true
				nextLayerInputBuffer = req.NextLayerInputOutput
				nextLayerInputBorrowed = true
			} else {
				perLayerFinalHiddenBuffer, nextLayerInputBuffer, err = hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfig(ctx, driver, perLayerProjectionBuffer, finalHiddenBuffer, perLayerNormCfg, nextInputNorm, layerScalar)
				if err != nil {
					return hipGemma4Q4DecoderLayerResult{}, err
				}
			}
		} else {
			if req.OmitDebugTensors && req.FinalHiddenOutput != nil {
				if err := hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfigOutputWithWorkspace(ctx, driver, perLayerProjectionBuffer, finalHiddenBuffer, perLayerNormCfg, req.FinalHiddenOutput, layerScalar, req.AttentionWorkspace); err != nil {
					return hipGemma4Q4DecoderLayerResult{}, err
				}
				perLayerFinalHiddenBuffer = req.FinalHiddenOutput
				perLayerFinalHiddenBorrowed = true
			} else {
				perLayerFinalHiddenBuffer, err = hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfig(ctx, driver, perLayerProjectionBuffer, finalHiddenBuffer, perLayerNormCfg, layerScalar)
				if err != nil {
					return hipGemma4Q4DecoderLayerResult{}, err
				}
			}
		}
		defer func(buffer *hipDeviceByteBuffer, borrowed bool) {
			if buffer != returnedFinalHiddenBuffer && !borrowed {
				_ = buffer.Close()
			}
		}(perLayerFinalHiddenBuffer, perLayerFinalHiddenBorrowed)
		finalHiddenBuffer = perLayerFinalHiddenBuffer
		finalHiddenBorrowed = perLayerFinalHiddenBorrowed
	}
	var finalHidden []float32
	var deviceFinalHidden *hipDeviceByteBuffer
	if req.ReturnDeviceHidden {
		returnedFinalHiddenBuffer = finalHiddenBuffer
		deviceFinalHidden = finalHiddenBuffer
		if !req.OmitDebugTensors {
			finalHidden, err = hipReadFloat32DeviceOutput(finalHiddenBuffer, hipGemma4Q4Layer0Operation, "final hidden output", cfg.HiddenSize)
			if err != nil {
				return hipGemma4Q4DecoderLayerResult{}, err
			}
		}
	} else {
		finalHidden, err = hipReadFloat32DeviceOutput(finalHiddenBuffer, hipGemma4Q4Layer0Operation, "final hidden output", cfg.HiddenSize)
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
	}
	retainedDeviceLayerSuccess = true
	result := hipGemma4Q4DecoderLayerResult{
		Key:                          ropeKey,
		Value:                        value,
		UpdatedKeys:                  updatedKeys,
		UpdatedValues:                updatedValues,
		DeviceKVAttention:            deviceKVAttention,
		DeviceLayer:                  retainedDeviceLayer,
		DeviceLayerValid:             retainedDeviceLayerValid,
		FinalHidden:                  finalHidden,
		DeviceFinalHidden:            deviceFinalHidden,
		DeviceFinalHiddenBorrowed:    finalHiddenBorrowed,
		DeviceNextLayerInput:         nextLayerInputBuffer,
		DeviceNextLayerInputBorrowed: nextLayerInputBorrowed,
	}
	if !req.OmitDebugTensors {
		layerInput, err = ensureLayerInput()
		if err != nil {
			return hipGemma4Q4DecoderLayerResult{}, err
		}
		result.LayerInput = layerInput
		result.AttentionOutput = attentionOutput
		result.AttentionProjection = attentionProjection
		result.AttentionResidual = attentionResidual
		result.MLPOutput = mlpOutput
	}
	nextLayerInputReturned = nextLayerInputBuffer != nil
	return result, nil
}

func hipRunGemma4Q4DeviceGELUTanhMLP(ctx context.Context, driver nativeHIPDriver, input []float32, gateCfg, upCfg, downCfg hipMLXQ4DeviceWeightConfig) ([]float32, error) {
	inputBuffer, err := hipUploadGemma4Q4Float32Input(driver, "GELU tanh MLP input", input)
	if err != nil {
		return nil, err
	}
	defer inputBuffer.Close()
	output, err := hipRunGemma4Q4DeviceGELUTanhMLPWithDeviceInput(ctx, driver, inputBuffer, gateCfg, upCfg, downCfg)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	return hipReadFloat32DeviceOutput(output, hipGemma4Q4Layer0Operation, "GELU tanh MLP output", downCfg.Rows)
}

func hipRunGemma4Q4DeviceGELUTanhMLPWithDeviceInput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, gateCfg, upCfg, downCfg hipMLXQ4DeviceWeightConfig) (*hipDeviceByteBuffer, error) {
	output, err := hipAllocateByteBuffer(driver, hipGemma4Q4Layer0Operation, "GELU tanh MLP output", uint64(downCfg.Rows*4), downCfg.Rows)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunGemma4Q4DeviceGELUTanhMLPWithDeviceInputOutput(ctx, driver, input, gateCfg, upCfg, downCfg, output, nil); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunGemma4Q4DeviceGELUTanhMLPWithDeviceInputOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, gateCfg, upCfg, downCfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if workspace != nil && hipMLXQ4GELUTanhMLPPersistentRouteEnabled && hipMLXQ4GELUTanhMLPPersistentCompatible(input, gateCfg, upCfg, downCfg) {
		activated, err := workspace.EnsureActivationOutput(driver, gateCfg.Rows)
		if err != nil {
			return err
		}
		barrier, err := workspace.EnsureGELUTanhMLPBarrier(driver)
		if err != nil {
			return err
		}
		return hipRunMLXQ4GELUTanhMLPPersistentKernelWithDeviceInputOutputWithWorkspace(ctx, driver, input, gateCfg, upCfg, downCfg, activated, output, barrier, workspace)
	}
	var activated *hipDeviceByteBuffer
	closeActivated := false
	if workspace != nil {
		var err error
		activated, err = workspace.EnsureActivationOutput(driver, gateCfg.Rows)
		if err != nil {
			return err
		}
		if err := hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInputOutputWithWorkspace(ctx, driver, input, gateCfg, upCfg, activated, workspace); err != nil {
			return err
		}
	} else {
		var err error
		activated, err = hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInput(ctx, driver, input, gateCfg, upCfg)
		if err != nil {
			return err
		}
		closeActivated = true
	}
	if closeActivated {
		defer activated.Close()
	}
	return hipRunMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace(ctx, driver, activated, downCfg, output, workspace)
}

func hipRunGemma4Q4DeviceGELUTanhProjection(ctx context.Context, driver nativeHIPDriver, input, multiplyBy []float32, gateCfg, projectionCfg hipMLXQ4DeviceWeightConfig) ([]float32, error) {
	inputBuffer, err := hipUploadGemma4Q4Float32Input(driver, "GELU tanh projection input", input)
	if err != nil {
		return nil, err
	}
	defer inputBuffer.Close()
	output, err := hipRunGemma4Q4DeviceGELUTanhProjectionWithDeviceInput(ctx, driver, inputBuffer, multiplyBy, gateCfg, projectionCfg)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	return hipReadFloat32DeviceOutput(output, hipGemma4Q4Layer0Operation, "GELU tanh projection output", projectionCfg.Rows)
}

func hipRunGemma4Q4DeviceGELUTanhProjectionWithDeviceInput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, multiplyBy []float32, gateCfg, projectionCfg hipMLXQ4DeviceWeightConfig) (*hipDeviceByteBuffer, error) {
	multiplyBuffer, err := hipUploadGemma4Q4Float32Input(driver, "GELU tanh projection multiplier", multiplyBy)
	if err != nil {
		return nil, err
	}
	defer multiplyBuffer.Close()
	return hipRunGemma4Q4DeviceGELUTanhProjectionWithDeviceMultiplier(ctx, driver, input, multiplyBuffer, gateCfg, projectionCfg)
}

func hipRunGemma4Q4DeviceGELUTanhProjectionWithDeviceMultiplier(ctx context.Context, driver nativeHIPDriver, input, multiplyBuffer *hipDeviceByteBuffer, gateCfg, projectionCfg hipMLXQ4DeviceWeightConfig) (*hipDeviceByteBuffer, error) {
	if multiplyBuffer == nil || multiplyBuffer.Pointer() == 0 || multiplyBuffer.Count() != gateCfg.Rows || multiplyBuffer.SizeBytes() != uint64(gateCfg.Rows*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "GELU tanh projection multiplier device buffer shape mismatch", nil)
	}
	activated, err := hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplier(ctx, driver, input, multiplyBuffer, gateCfg)
	if err != nil {
		return nil, err
	}
	defer activated.Close()
	output, err := hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, activated, projectionCfg)
	if err != nil {
		return nil, err
	}
	return output, nil
}

func hipRunGemma4Q4DeviceGELUTanhProjectionWithDeviceMultiplierOutput(ctx context.Context, driver nativeHIPDriver, input, multiplyBuffer *hipDeviceByteBuffer, gateCfg, projectionCfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if multiplyBuffer == nil || multiplyBuffer.Pointer() == 0 || multiplyBuffer.Count() != gateCfg.Rows || multiplyBuffer.SizeBytes() != uint64(gateCfg.Rows*4) {
		return core.E(hipGemma4Q4Layer0Operation, "GELU tanh projection multiplier device buffer shape mismatch", nil)
	}
	var activated *hipDeviceByteBuffer
	closeActivated := false
	if workspace != nil {
		var err error
		activated, err = workspace.EnsureActivationOutput(driver, gateCfg.Rows)
		if err != nil {
			return err
		}
		if err := hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplierOutputWithWorkspace(ctx, driver, input, multiplyBuffer, gateCfg, activated, workspace); err != nil {
			return err
		}
	} else {
		var err error
		activated, err = hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplier(ctx, driver, input, multiplyBuffer, gateCfg)
		if err != nil {
			return err
		}
		closeActivated = true
	}
	if closeActivated {
		defer activated.Close()
	}
	return hipRunMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace(ctx, driver, activated, projectionCfg, output, workspace)
}

func hipUploadGemma4Q4Float32Input(driver nativeHIPDriver, label string, input []float32) (*hipDeviceByteBuffer, error) {
	if len(input) == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, label+" is required", nil)
	}
	if !rocmFloat32SliceFinite(input) {
		return nil, core.E(hipGemma4Q4Layer0Operation, label+" values must be finite", nil)
	}
	payload, err := hipFloat32Payload(input)
	if err != nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "encode "+label, err)
	}
	return hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label, payload, len(input))
}

func (cfg hipGemma4Q4Layer0Config) validate() error {
	if cfg.Layer < 0 {
		return core.E(hipGemma4Q4Layer0Operation, "layer index must be non-negative", nil)
	}
	if cfg.LayerType != "" && !hipGemma4Q4LayerTypeSupported(cfg.LayerType) {
		return core.E(hipGemma4Q4Layer0Operation, "unsupported Gemma4 q4 layer type", nil)
	}
	keyHeads := firstPositiveInt(cfg.KeyHeads, 1)
	if cfg.HiddenSize <= 0 || cfg.VocabSize <= 0 || cfg.GroupSize <= 0 ||
		cfg.HeadDim <= 0 || cfg.QueryHeads <= 0 || keyHeads <= 0 || cfg.IntermediateSize <= 0 {
		return core.E(hipGemma4Q4Layer0Operation, "hidden, vocab, group, head, and intermediate sizes must be positive", nil)
	}
	if keyHeads > cfg.QueryHeads || cfg.QueryHeads%keyHeads != 0 {
		return core.E(hipGemma4Q4Layer0Operation, "key head count must divide query head count", nil)
	}
	if cfg.RoPEBase <= 0 || math.IsNaN(float64(cfg.RoPEBase)) || math.IsInf(float64(cfg.RoPEBase), 0) {
		return core.E(hipGemma4Q4Layer0Operation, "layer RoPE base must be positive and finite", nil)
	}
	if cfg.RoPERotaryDim <= 0 || cfg.RoPERotaryDim > cfg.HeadDim || cfg.RoPERotaryDim%2 != 0 {
		return core.E(hipGemma4Q4Layer0Operation, "layer RoPE rotary dimension must be positive, even, and no larger than head dimension", nil)
	}
	if cfg.effectiveRoPEFrequencyScale() <= 0 {
		return core.E(hipGemma4Q4Layer0Operation, "layer RoPE frequency scale must be positive and finite", nil)
	}
	if cfg.SlidingWindow < 0 {
		return core.E(hipGemma4Q4Layer0Operation, "sliding window must be non-negative", nil)
	}
	if cfg.AttentionKEqV && cfg.LayerType != "full_attention" {
		return core.E(hipGemma4Q4Layer0Operation, "K=V attention is only valid for full-attention layers", nil)
	}
	if cfg.FinalLogitSoftcap < 0 || math.IsNaN(float64(cfg.FinalLogitSoftcap)) || math.IsInf(float64(cfg.FinalLogitSoftcap), 0) {
		return core.E(hipGemma4Q4Layer0Operation, "final logit softcap must be non-negative and finite", nil)
	}
	if scalar := cfg.effectiveLayerScalar(); math.IsNaN(float64(scalar)) || math.IsInf(float64(scalar), 0) {
		return core.E(hipGemma4Q4Layer0Operation, "layer scalar must be finite", nil)
	}
	if cfg.Embedding.TableEncoding != hipEmbeddingTableEncodingMLXQ4 ||
		cfg.Embedding.VocabSize != cfg.VocabSize ||
		cfg.Embedding.HiddenSize != cfg.HiddenSize ||
		cfg.Embedding.GroupSize != cfg.GroupSize {
		return core.E(hipGemma4Q4Layer0Operation, "embedding config must match Gemma4 q4 dimensions", nil)
	}
	if err := cfg.Embedding.validate([]int32{0}); err != nil {
		return core.E(hipGemma4Q4Layer0Operation, "embedding config", err)
	}
	if cfg.QueryProjection.Rows != cfg.QueryHeads*cfg.HeadDim ||
		cfg.KeyProjection.Rows != keyHeads*cfg.HeadDim ||
		cfg.ValueProjection.Rows != keyHeads*cfg.HeadDim ||
		cfg.OutputProjection.Rows != cfg.HiddenSize ||
		cfg.GateProjection.Rows != cfg.IntermediateSize ||
		cfg.UpProjection.Rows != cfg.IntermediateSize ||
		cfg.DownProjection.Rows != cfg.HiddenSize ||
		cfg.LMHeadProjection.Rows != cfg.VocabSize {
		return core.E(hipGemma4Q4Layer0Operation, "projection row counts do not match Gemma4 layer geometry", nil)
	}
	for label, projection := range map[string]struct {
		cfg  hipMLXQ4DeviceWeightConfig
		cols int
	}{
		"q_proj":               {cfg: cfg.QueryProjection, cols: cfg.HiddenSize},
		"k_proj":               {cfg: cfg.KeyProjection, cols: cfg.HiddenSize},
		"v_proj":               {cfg: cfg.ValueProjection, cols: cfg.HiddenSize},
		"o_proj":               {cfg: cfg.OutputProjection, cols: cfg.QueryHeads * cfg.HeadDim},
		"mlp.gate_proj":        {cfg: cfg.GateProjection, cols: cfg.HiddenSize},
		"mlp.up_proj":          {cfg: cfg.UpProjection, cols: cfg.HiddenSize},
		"mlp.down_proj":        {cfg: cfg.DownProjection, cols: cfg.IntermediateSize},
		"embed_tokens_lm_head": {cfg: cfg.LMHeadProjection, cols: cfg.HiddenSize},
	} {
		if err := projection.cfg.validateInputCount(projection.cols); err != nil {
			return core.E(hipGemma4Q4Layer0Operation, label+" config", err)
		}
	}
	for label, norm := range map[string]struct {
		cfg   hipRMSNormDeviceWeightConfig
		count int
	}{
		"input_layernorm":            {cfg: cfg.InputNorm, count: cfg.HiddenSize},
		"q_norm":                     {cfg: cfg.QueryNorm, count: cfg.HeadDim},
		"k_norm":                     {cfg: cfg.KeyNorm, count: cfg.HeadDim},
		"post_attention_layernorm":   {cfg: cfg.PostAttentionNorm, count: cfg.HiddenSize},
		"pre_feedforward_layernorm":  {cfg: cfg.PreFeedForwardNorm, count: cfg.HiddenSize},
		"post_feedforward_layernorm": {cfg: cfg.PostFeedForwardNorm, count: cfg.HiddenSize},
		"final_norm":                 {cfg: cfg.FinalNorm, count: cfg.HiddenSize},
	} {
		if err := hipValidateGemma4Q4NormConfig(label, norm.cfg, norm.count); err != nil {
			return err
		}
	}
	if err := cfg.validatePerLayerInput(); err != nil {
		return err
	}
	if err := cfg.MoE.validate(cfg.HiddenSize); err != nil {
		return err
	}
	return nil
}

func (cfg hipGemma4Q4Layer0Config) effectiveLayerScalar() float32 {
	if cfg.LayerScalar == 0 {
		return 1
	}
	return cfg.LayerScalar
}

func (cfg hipGemma4Q4Layer0Config) validatePerLayerInput() error {
	perLayer := cfg.PerLayerInput
	if perLayer.isZero() {
		return nil
	}
	if perLayer.layerApplyConfigured() {
		if !perLayer.hasLayerApply() {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer input gate, projection, post norm, and input size must be configured together", nil)
		}
		if perLayer.InputSize <= 0 {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer input size must be positive", nil)
		}
		if perLayer.InputGate.Rows != perLayer.InputSize || perLayer.InputGate.Cols != cfg.HiddenSize {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer input gate shape does not match layer geometry", nil)
		}
		if perLayer.Projection.Rows != cfg.HiddenSize || perLayer.Projection.Cols != perLayer.InputSize {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer projection shape does not match layer geometry", nil)
		}
		if err := perLayer.InputGate.validateInputCount(cfg.HiddenSize); err != nil {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer input gate config", err)
		}
		if err := perLayer.Projection.validateInputCount(perLayer.InputSize); err != nil {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer projection config", err)
		}
		if err := hipValidateGemma4Q4NormConfig("post_per_layer_input_norm", perLayer.PostInputNorm, cfg.HiddenSize); err != nil {
			return err
		}
	}
	if perLayer.globalPrecomputeConfigured() {
		if !perLayer.hasGlobalPrecompute() {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer embedding, model projection, and projection norm must be configured together", nil)
		}
		if !perLayer.hasLayerApply() {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer input precompute requires per-layer gate/projection tensors", nil)
		}
		if err := perLayer.Embedding.validate([]int32{0}); err != nil {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer embedding config", err)
		}
		if err := perLayer.ModelProjection.validate(hipProjectionWeightEncodingBF16); err != nil {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer model projection config", err)
		}
		if perLayer.ModelProjection.Rows != perLayer.Embedding.HiddenSize ||
			perLayer.ModelProjection.Cols != cfg.HiddenSize ||
			perLayer.ModelProjection.Rows%perLayer.InputSize != 0 {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer global projection shape does not match layer geometry", nil)
		}
		if layerCount := perLayer.ModelProjection.Rows / perLayer.InputSize; cfg.Layer >= layerCount {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer input layer index is outside global projection rows", nil)
		}
		if err := hipValidateGemma4Q4NormConfig("per_layer_projection_norm", perLayer.ProjectionNorm, perLayer.InputSize); err != nil {
			return err
		}
	}
	return nil
}

func (cfg hipGemma4Q4PerLayerInputConfig) isZero() bool {
	return !cfg.layerApplyConfigured() && !cfg.globalPrecomputeConfigured()
}

func (cfg hipGemma4Q4PerLayerInputConfig) layerApplyConfigured() bool {
	return cfg.InputSize != 0 ||
		cfg.InputGate.WeightPointer != 0 ||
		cfg.InputGate.ScalePointer != 0 ||
		cfg.InputGate.BiasPointer != 0 ||
		cfg.Projection.WeightPointer != 0 ||
		cfg.Projection.ScalePointer != 0 ||
		cfg.Projection.BiasPointer != 0 ||
		cfg.PostInputNorm.WeightPointer != 0
}

func (cfg hipGemma4Q4PerLayerInputConfig) hasLayerApply() bool {
	return cfg.InputSize > 0 &&
		cfg.InputGate.WeightPointer != 0 &&
		cfg.InputGate.ScalePointer != 0 &&
		cfg.InputGate.BiasPointer != 0 &&
		cfg.Projection.WeightPointer != 0 &&
		cfg.Projection.ScalePointer != 0 &&
		cfg.Projection.BiasPointer != 0 &&
		cfg.PostInputNorm.WeightPointer != 0
}

func (cfg hipGemma4Q4PerLayerInputConfig) globalPrecomputeConfigured() bool {
	return cfg.Embedding.EmbeddingPointer != 0 ||
		cfg.Embedding.ScalePointer != 0 ||
		cfg.Embedding.BiasPointer != 0 ||
		cfg.ModelProjection.WeightPointer != 0 ||
		cfg.ProjectionNorm.WeightPointer != 0
}

func (cfg hipGemma4Q4PerLayerInputConfig) hasGlobalPrecompute() bool {
	return cfg.Embedding.EmbeddingPointer != 0 &&
		cfg.Embedding.ScalePointer != 0 &&
		cfg.Embedding.BiasPointer != 0 &&
		cfg.ModelProjection.WeightPointer != 0 &&
		cfg.ProjectionNorm.WeightPointer != 0
}

func (cfg *hipGemma4Q4Layer0Config) finalizeScales() {
	if cfg == nil {
		return
	}
	cfg.EmbeddingScale = hipGemma4Q4EmbeddingScale(cfg.HiddenSize)
	cfg.PerLayerInput.finalizeScales()
}

func (cfg hipGemma4Q4Layer0Config) embeddingScale() float32 {
	if cfg.EmbeddingScale != 0 {
		return cfg.EmbeddingScale
	}
	return hipGemma4Q4EmbeddingScale(cfg.HiddenSize)
}

func (cfg *hipGemma4Q4PerLayerInputConfig) finalizeScales() {
	if cfg == nil {
		return
	}
	cfg.EmbeddingScale = hipGemma4Q4EmbeddingScale(cfg.InputSize)
	if cfg.ModelProjection.Cols > 0 {
		cfg.ModelProjectionScale = float32(math.Pow(float64(cfg.ModelProjection.Cols), -0.5))
	} else {
		cfg.ModelProjectionScale = 0
	}
}

func (cfg hipGemma4Q4PerLayerInputConfig) embeddingScale() float32 {
	if cfg.EmbeddingScale != 0 {
		return cfg.EmbeddingScale
	}
	return hipGemma4Q4EmbeddingScale(cfg.InputSize)
}

func (cfg hipGemma4Q4PerLayerInputConfig) modelProjectionScale() float32 {
	if cfg.ModelProjectionScale != 0 {
		return cfg.ModelProjectionScale
	}
	if cfg.ModelProjection.Cols <= 0 {
		return 0
	}
	return float32(math.Pow(float64(cfg.ModelProjection.Cols), -0.5))
}

func hipGemma4Q4EmbeddingScale(hiddenSize int) float32 {
	if hiddenSize <= 0 {
		return 0
	}
	scale := float32(math.Sqrt(float64(hiddenSize)))
	return hipBFloat16ToFloat32(hipFloat32ToBFloat16(scale))
}

func (cfg hipBF16DeviceWeightConfig) validate(encoding uint32) error {
	if cfg.WeightPointer == 0 {
		return core.E("rocm.hip.ProjectionLaunch", "projection weight pointer is required", nil)
	}
	if cfg.Rows <= 0 || cfg.Cols <= 0 {
		return core.E("rocm.hip.ProjectionLaunch", "projection rows and cols must be positive", nil)
	}
	weightElements, err := hipProjectionDeviceWeightElementCount(cfg.WeightBytes, encoding)
	if err != nil {
		return err
	}
	if err := validateHIPProjectionShape(cfg.Cols, weightElements, 0, cfg.Rows, cfg.Cols); err != nil {
		return err
	}
	return nil
}

func (cfg hipGemma4Q4ForwardConfig) validate() error {
	if len(cfg.Layers) == 0 {
		return core.E(hipGemma4Q4Layer0Operation, "at least one Gemma4 q4 layer config is required", nil)
	}
	if cfg.KVSharedLayers < 0 || cfg.KVSharedLayers > len(cfg.Layers) {
		return core.E(hipGemma4Q4Layer0Operation, "KV shared layer count must fit forward layer count", nil)
	}
	if len(cfg.SharedKVSources) > 0 {
		if len(cfg.SharedKVSources) != len(cfg.Layers) {
			return core.E(hipGemma4Q4Layer0Operation, "shared KV source table must match layer count", nil)
		}
		for index, source := range cfg.SharedKVSources {
			if source < 0 || source >= len(cfg.Layers) || source > index {
				return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("shared KV source for layer %d is invalid", index), nil)
			}
		}
	}
	first := cfg.Layers[0]
	if err := first.validate(); err != nil {
		return err
	}
	for index, layer := range cfg.Layers[1:] {
		if err := layer.validate(); err != nil {
			return err
		}
		if layer.HiddenSize != first.HiddenSize ||
			layer.VocabSize != first.VocabSize ||
			layer.GroupSize != first.GroupSize {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("layer %d geometry does not match layer 0", index+1), nil)
		}
		if first.PerLayerInput.hasGlobalPrecompute() && !layer.PerLayerInput.hasLayerApply() {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("layer %d per-layer input config is missing", index+1), nil)
		}
	}
	return nil
}

func hipGemma4Q4SharedKVSourceByLayer(cfg hipGemma4Q4ForwardConfig) []int {
	if len(cfg.SharedKVSources) == len(cfg.Layers) {
		return cfg.SharedKVSources
	}
	return hipGemma4Q4BuildSharedKVSourceByLayer(cfg)
}

func hipGemma4Q4BuildSharedKVSourceByLayer(cfg hipGemma4Q4ForwardConfig) []int {
	sources := make([]int, len(cfg.Layers))
	for index := range sources {
		sources[index] = index
	}
	if len(cfg.Layers) == 0 || cfg.KVSharedLayers <= 0 {
		return sources
	}
	firstShared := len(cfg.Layers) - cfg.KVSharedLayers
	if firstShared < 0 {
		firstShared = 0
	}
	latestByType := map[string]int{}
	for index, layer := range cfg.Layers {
		layerType := firstNonEmptyString(layer.LayerType, hipGemma4Q4LayerTypeFromHeadDim(layer.HeadDim))
		ownsCache := index < firstShared
		if !ownsCache {
			if previous, ok := latestByType[layerType]; ok {
				sources[index] = previous
			} else {
				ownsCache = true
			}
		}
		if ownsCache {
			sources[index] = index
			latestByType[layerType] = index
		}
	}
	return sources
}

func (state hipGemma4Q4DecodeState) validate(cfg hipGemma4Q4ForwardConfig) error {
	if len(state.Layers) == 0 {
		return nil
	}
	if len(state.Layers) != len(cfg.Layers) {
		return core.E(hipGemma4Q4Layer0Operation, "decode state layer count must match forward config", nil)
	}
	for index, layerState := range state.Layers {
		layerCfg := cfg.Layers[index]
		if err := hipGemma4Q4ValidateKVState(layerState.Keys, layerState.Values, layerCfg.HeadDim); err != nil {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("decode state layer %d", index), err)
		}
	}
	return nil
}

func (state hipGemma4Q4DecodeState) layer(index int) hipGemma4Q4LayerKVState {
	if len(state.Layers) == 0 {
		return hipGemma4Q4LayerKVState{}
	}
	return state.Layers[index]
}

func (state hipGemma4Q4DecodeState) tokenCount(headDim int) int {
	if len(state.Layers) == 0 || headDim <= 0 {
		return 0
	}
	return len(state.Layers[0].Keys) / headDim
}

func (state hipGemma4Q4DecodeState) tokenCountForConfig(cfg hipGemma4Q4ForwardConfig) int {
	if len(state.Layers) == 0 || len(cfg.Layers) == 0 {
		return 0
	}
	maxTokens := 0
	for index, layerState := range state.Layers {
		headDim := cfg.Layers[0].HeadDim
		if index < len(cfg.Layers) && cfg.Layers[index].HeadDim > 0 {
			headDim = cfg.Layers[index].HeadDim
		}
		if headDim <= 0 || len(layerState.Keys) == 0 {
			continue
		}
		tokens := len(layerState.Keys) / headDim
		if tokens > maxTokens {
			maxTokens = tokens
		}
	}
	return maxTokens
}

func (req hipGemma4Q4Layer0Request) validate(cfg hipGemma4Q4Layer0Config) error {
	if req.TokenID < 0 || int(req.TokenID) >= cfg.VocabSize {
		return core.E(hipGemma4Q4Layer0Operation, "token ID is outside vocabulary", nil)
	}
	return (hipGemma4Q4DecoderLayerRequest{
		Position: req.Position,
		RoPEBase: req.RoPEBase,
		Epsilon:  req.Epsilon,
	}).validate(cfg)
}

func (req hipGemma4Q4DecoderLayerRequest) validate(cfg hipGemma4Q4Layer0Config) error {
	if req.Position < 0 {
		return core.E(hipGemma4Q4Layer0Operation, "position must be non-negative", nil)
	}
	kvDim := cfg.keyValueDim()
	if _, err := req.effectiveRoPEBase(cfg); err != nil {
		return err
	}
	if req.Epsilon < 0 || math.IsNaN(float64(req.Epsilon)) || math.IsInf(float64(req.Epsilon), 0) {
		return core.E(hipGemma4Q4Layer0Operation, "epsilon must be non-negative and finite", nil)
	}
	if len(req.SharedKeys) > 0 || len(req.SharedValues) > 0 {
		if err := hipGemma4Q4ValidateKVState(req.SharedKeys, req.SharedValues, kvDim); err != nil {
			return core.E(hipGemma4Q4Layer0Operation, "shared key/value state", err)
		}
		if len(req.SharedKeys) == 0 {
			return core.E(hipGemma4Q4Layer0Operation, "shared key/value state must be non-empty", nil)
		}
		if len(req.SharedKeys)%kvDim != 0 {
			return core.E(hipGemma4Q4Layer0Operation, "shared key/value lengths must align with head dimension", nil)
		}
		if req.Position+1 != len(req.SharedKeys)/kvDim {
			return core.E(hipGemma4Q4Layer0Operation, "shared key/value token count must include current position", nil)
		}
	}
	if len(req.PerLayerInput) > 0 {
		if !cfg.PerLayerInput.hasLayerApply() {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer input requires configured gate/projection tensors", nil)
		}
		if len(req.PerLayerInput) != cfg.PerLayerInput.InputSize {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer input length must match configured input size", nil)
		}
		if !rocmFloat32SliceFinite(req.PerLayerInput) {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer input values must be finite", nil)
		}
	}
	if req.PerLayerInputDevice != nil {
		if len(req.PerLayerInput) > 0 {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer input cannot mix host and device buffers", nil)
		}
		if !cfg.PerLayerInput.hasLayerApply() {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer input requires configured gate/projection tensors", nil)
		}
		if req.PerLayerInputDevice.Pointer() == 0 ||
			req.PerLayerInputDevice.Count() != cfg.PerLayerInput.InputSize ||
			req.PerLayerInputDevice.SizeBytes() != uint64(cfg.PerLayerInput.InputSize*4) {
			return core.E(hipGemma4Q4Layer0Operation, "per-layer input device buffer shape mismatch", nil)
		}
	}
	if req.LayerInputDevice != nil {
		if req.LayerInputDevice.Pointer() == 0 ||
			req.LayerInputDevice.Count() != cfg.HiddenSize ||
			req.LayerInputDevice.SizeBytes() != uint64(cfg.HiddenSize*4) {
			return core.E(hipGemma4Q4Layer0Operation, "precomputed layer input device buffer shape mismatch", nil)
		}
	}
	if nextInputNorm, hasNextInputNorm := req.nextInputNormConfig(); hasNextInputNorm {
		if nextInputNorm.Count != cfg.HiddenSize {
			return core.E(hipGemma4Q4Layer0Operation, "next input norm count must match hidden size", nil)
		}
		if err := hipValidateRMSNormDeviceWeightConfig("Gemma4Q4NextInputNorm", nextInputNorm); err != nil {
			return err
		}
	}
	if err := hipGemma4Q4ValidateKVState(req.PriorKeys, req.PriorValues, kvDim); err != nil {
		return core.E(hipGemma4Q4Layer0Operation, "prior key/value state", err)
	}
	if req.PriorDeviceKV != nil && !req.DeviceKVAttention {
		return core.E(hipGemma4Q4Layer0Operation, "prior device KV requires device KV attention", nil)
	}
	if req.SharedDeviceKV != nil && !req.DeviceKVAttention {
		return core.E(hipGemma4Q4Layer0Operation, "shared device KV requires device KV attention", nil)
	}
	if req.SharedDeviceKV != nil && (len(req.SharedKeys) > 0 || len(req.SharedValues) > 0) {
		return core.E(hipGemma4Q4Layer0Operation, "shared device KV cannot be combined with host shared KV", nil)
	}
	if req.SharedDescriptorTable != nil {
		if req.SharedDeviceKV == nil {
			return core.E(hipGemma4Q4Layer0Operation, "shared descriptor table requires shared device KV", nil)
		}
		if err := req.SharedDescriptorTable.CompatibleWith(req.SharedDeviceKV); err != nil {
			return core.E(hipGemma4Q4Layer0Operation, "shared descriptor table", err)
		}
	}
	if req.KeepDeviceKV && !req.DeviceKVAttention {
		return core.E(hipGemma4Q4Layer0Operation, "keeping device KV requires device KV attention", nil)
	}
	if req.PriorDeviceKV != nil {
		if req.PriorDeviceKV.closed {
			return core.E(hipGemma4Q4Layer0Operation, "prior device KV is closed", nil)
		}
		mode := firstNonEmptyString(req.DeviceKVMode, req.PriorDeviceKV.mode)
		if mode == "" {
			mode = rocmKVCacheModeFP16
		}
		if req.PriorDeviceKV.mode != "" && req.PriorDeviceKV.mode != mode {
			return core.E(hipGemma4Q4Layer0Operation, "prior device KV mode mismatch", nil)
		}
		if !req.OmitHostKV || len(req.PriorKeys) > 0 {
			hostTokens := 0
			if len(req.PriorKeys) > 0 {
				hostTokens = len(req.PriorKeys) / kvDim
			}
			if req.PriorDeviceKV.TokenCount() != hostTokens {
				return core.E(hipGemma4Q4Layer0Operation, "prior device KV token count mismatch", nil)
			}
		}
		keyWidth, valueWidth, ok := req.PriorDeviceKV.LastVectorWidths()
		if !ok || keyWidth != kvDim || valueWidth != kvDim {
			return core.E(hipGemma4Q4Layer0Operation, "prior device KV width mismatch", nil)
		}
	}
	if req.PriorDescriptorTable != nil {
		if req.PriorDeviceKV == nil {
			return core.E(hipGemma4Q4Layer0Operation, "prior descriptor table requires prior device KV", nil)
		}
		if err := req.PriorDescriptorTable.CompatibleWith(req.PriorDeviceKV); err != nil {
			return core.E(hipGemma4Q4Layer0Operation, "prior descriptor table", err)
		}
	}
	if req.SharedDeviceKV != nil {
		if req.SharedDeviceKV.closed {
			return core.E(hipGemma4Q4Layer0Operation, "shared device KV is closed", nil)
		}
		mode := firstNonEmptyString(req.DeviceKVMode, req.SharedDeviceKV.mode)
		if mode == "" {
			mode = rocmKVCacheModeFP16
		}
		if req.SharedDeviceKV.mode != "" && req.SharedDeviceKV.mode != mode {
			return core.E(hipGemma4Q4Layer0Operation, "shared device KV mode mismatch", nil)
		}
		keyWidth, valueWidth, ok := req.SharedDeviceKV.LastVectorWidths()
		if !ok || keyWidth != kvDim || valueWidth != kvDim {
			return core.E(hipGemma4Q4Layer0Operation, "shared device KV width mismatch", nil)
		}
	}
	return nil
}

func (req hipGemma4Q4GreedyDecodeRequest) validate(cfg hipGemma4Q4ForwardConfig) error {
	if len(req.PromptTokenIDs) == 0 {
		return core.E(hipGemma4Q4Layer0Operation, "at least one prompt token is required", nil)
	}
	if req.MaxNewTokens <= 0 {
		return core.E(hipGemma4Q4Layer0Operation, "max new tokens must be positive", nil)
	}
	for _, tokenID := range req.PromptTokenIDs {
		if err := (hipGemma4Q4Layer0Request{
			TokenID:  tokenID,
			Position: req.Position,
			RoPEBase: req.RoPEBase,
			Epsilon:  req.Epsilon,
		}).validate(cfg.Layers[0]); err != nil {
			return err
		}
	}
	if req.DeviceKVMode != "" && !isROCmKVCacheMode(req.DeviceKVMode) {
		return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("unsupported device KV cache mode %q", req.DeviceKVMode), nil)
	}
	return nil
}

func (req hipGemma4Q4DecoderLayerRequest) effectiveRoPEBase(cfg hipGemma4Q4Layer0Config) (float32, error) {
	base := req.RoPEBase
	if base == 0 {
		base = cfg.RoPEBase
	}
	if base <= 0 || math.IsNaN(float64(base)) || math.IsInf(float64(base), 0) {
		return 0, core.E(hipGemma4Q4Layer0Operation, "RoPE base must be positive and finite", nil)
	}
	return base, nil
}

func (req hipGemma4Q4DecoderLayerRequest) nextInputNormConfig() (hipRMSNormDeviceWeightConfig, bool) {
	if req.HasNextInputNorm {
		return req.NextInputNormValue, true
	}
	if req.NextInputNorm != nil {
		return *req.NextInputNorm, true
	}
	return hipRMSNormDeviceWeightConfig{}, false
}

func hipGemma4Q4ValidateKVState(keys, values []float32, headDim int) error {
	if len(keys) != len(values) {
		return core.E(hipGemma4Q4Layer0Operation, "keys and values must have matching lengths", nil)
	}
	if len(keys) == 0 {
		return nil
	}
	if headDim <= 0 || len(keys)%headDim != 0 {
		return core.E(hipGemma4Q4Layer0Operation, "key/value lengths must align with head dimension", nil)
	}
	return nil
}

func hipGemma4Q4AppendKV(prior, current []float32) []float32 {
	output := make([]float32, 0, len(prior)+len(current))
	output = append(output, prior...)
	output = append(output, current...)
	return output
}

func hipGemma4Q4LastKVToken(keys, values []float32, headDim int) ([]float32, []float32, error) {
	if err := hipGemma4Q4ValidateKVState(keys, values, headDim); err != nil {
		return nil, nil, err
	}
	if len(keys) == 0 {
		return nil, nil, core.E(hipGemma4Q4Layer0Operation, "key/value state has no tokens", nil)
	}
	start := len(keys) - headDim
	return append([]float32(nil), keys[start:]...), append([]float32(nil), values[start:]...), nil
}

func hipGemma4Q4TrimKVWindow(keys, values []float32, headDim, window int) ([]float32, []float32) {
	if window <= 0 || headDim <= 0 {
		return keys, values
	}
	maxValues := headDim * window
	if len(keys) <= maxValues {
		return keys, values
	}
	trimmedKeys := append([]float32(nil), keys[len(keys)-maxValues:]...)
	trimmedValues := append([]float32(nil), values[len(values)-maxValues:]...)
	return trimmedKeys, trimmedValues
}

func hipGemma4Q4SoftcapLogits(logits []float32, softcap float32) ([]float32, error) {
	if softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "final logit softcap must be non-negative and finite", nil)
	}
	if softcap == 0 {
		return logits, nil
	}
	for index, value := range logits {
		logits[index] = float32(math.Tanh(float64(value/softcap))) * softcap
	}
	return logits, nil
}

func hipRunGemma4Q4RoPEVector(ctx context.Context, driver nativeHIPDriver, input []float32, position int, base float32, rotaryDim int) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if rotaryDim <= 0 || rotaryDim > len(input) || rotaryDim%2 != 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "RoPE rotary dimension must be positive, even, and no larger than input length", nil)
	}
	if rotaryDim == len(input) {
		return hipRunRoPEKernel(ctx, driver, hipRoPERequest{Input: input, Position: position, Base: base})
	}
	rotated, err := hipRunRoPEKernel(ctx, driver, hipRoPERequest{Input: append([]float32(nil), input[:rotaryDim]...), Position: position, Base: base, FrequencyDim: len(input)})
	if err != nil {
		return nil, err
	}
	output := make([]float32, len(input))
	copy(output, rotated)
	copy(output[rotaryDim:], input[rotaryDim:])
	return output, nil
}

func hipRunGemma4Q4RMSNormNoScale(ctx context.Context, driver nativeHIPDriver, input []float32, epsilon float32) ([]float32, error) {
	if len(input) == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "RMSNormNoScale input is required", nil)
	}
	ones := make([]float32, len(input))
	for index := range ones {
		ones[index] = 1
	}
	return hipRunRMSNormKernel(ctx, driver, hipRMSNormRequest{
		Input:   input,
		Weight:  ones,
		Epsilon: epsilon,
	})
}

func hipRunGemma4Q4RMSNormNoScaleWithDeviceInput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, epsilon float32) ([]float32, error) {
	output, err := hipRunGemma4Q4RMSNormNoScaleDeviceKernel(ctx, driver, input, epsilon)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	return hipReadFloat32DeviceOutput(output, hipGemma4Q4Layer0Operation, "RMSNormNoScale output", input.Count())
}

func hipRunGemma4Q4RMSNormNoScaleDeviceKernel(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, epsilon float32) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if input == nil || input.Pointer() == 0 || input.Count() <= 0 || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "RMSNormNoScale device input is required", nil)
	}
	output, err := hipAllocateByteBuffer(driver, hipGemma4Q4Layer0Operation, "RMSNormNoScale output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, err
	}
	cfg := hipRMSNormDeviceWeightConfig{
		Count:          input.Count(),
		Epsilon:        epsilon,
		WeightEncoding: hipRMSNormWeightEncodingNone,
	}
	if err := hipRunRMSNormDeviceToDeviceKernel(ctx, driver, input.Pointer(), input.SizeBytes(), output.Pointer(), output.SizeBytes(), cfg); err != nil {
		_ = output.Close()
		return nil, err
	}
	return output, nil
}

func hipRunGemma4Q4RMSNormNoScaleDeviceKernelOutput(ctx context.Context, driver nativeHIPDriver, input, output *hipDeviceByteBuffer, epsilon float32) error {
	return hipRunGemma4Q4RMSNormNoScaleDeviceKernelOutputWithWorkspace(ctx, driver, input, output, epsilon, nil)
}

// hipRunGemma4Q4ValueNormDeviceKernel applies gemma4's no-scale value RMSNorm
// PER HEAD: one RMSNorm over each headDim-length kv head. This mirrors the
// prefill value-norm (hipRunGemma4Q4PrefillValueNormBatch) and the metal
// reference. Normalising over the whole kvDim instead folds every kv head under
// a single RMS, which is identical only when headCount == 1 (E2B); for the
// dense 12B's 8-kv-head sliding layers it shrinks V by ~sqrt(headCount) and
// rebalances the heads, starving attention into repetition.
func hipRunGemma4Q4ValueNormDeviceKernel(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, headDim, headCount int, epsilon float32) (*hipDeviceByteBuffer, error) {
	return hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, driver, input, hipGemma4Q4ValueNormConfig(headDim, epsilon), headCount)
}

func hipRunGemma4Q4ValueNormDeviceKernelOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, input, output *hipDeviceByteBuffer, headDim, headCount int, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace) error {
	return hipRunRMSNormHeadsKernelWithDeviceInputWeightConfigOutputWithWorkspace(ctx, driver, input, hipGemma4Q4ValueNormConfig(headDim, epsilon), headCount, output, workspace)
}

func hipGemma4Q4ValueNormConfig(headDim int, epsilon float32) hipRMSNormDeviceWeightConfig {
	return hipRMSNormDeviceWeightConfig{
		Count:          headDim,
		Epsilon:        epsilon,
		WeightEncoding: hipRMSNormWeightEncodingNone,
	}
}

func hipRunGemma4Q4RMSNormNoScaleDeviceKernelOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, input, output *hipDeviceByteBuffer, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if input == nil || input.Pointer() == 0 || input.Count() <= 0 || input.SizeBytes() != uint64(input.Count()*4) {
		return core.E(hipGemma4Q4Layer0Operation, "RMSNormNoScale device input is required", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != input.Count() || output.SizeBytes() != input.SizeBytes() {
		return core.E(hipGemma4Q4Layer0Operation, "RMSNormNoScale device output shape mismatch", nil)
	}
	cfg := hipRMSNormDeviceWeightConfig{
		Count:          input.Count(),
		Epsilon:        epsilon,
		WeightEncoding: hipRMSNormWeightEncodingNone,
	}
	return hipRunRMSNormDeviceToDeviceKernelWithWorkspace(ctx, driver, input.Pointer(), input.SizeBytes(), output.Pointer(), output.SizeBytes(), cfg, workspace)
}

func hipRunGemma4Q4PerLayerInputForLayer(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, tokenID int32, hidden []float32, epsilon float32) ([]float32, error) {
	if !cfg.PerLayerInput.hasGlobalPrecompute() {
		return nil, nil
	}
	inputs, err := hipRunGemma4Q4PerLayerInputSet(ctx, driver, cfg.PerLayerInput, tokenID, hidden, epsilon)
	if err != nil {
		return nil, err
	}
	if cfg.Layer < 0 || cfg.Layer >= len(inputs) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input layer index is outside computed inputs", nil)
	}
	return inputs[cfg.Layer], nil
}

func hipRunGemma4Q4PerLayerInputs(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokenID int32, hidden []float32, epsilon float32) ([][]float32, error) {
	if len(cfg.Layers) == 0 || !cfg.Layers[0].PerLayerInput.hasGlobalPrecompute() {
		return nil, nil
	}
	inputs, err := hipRunGemma4Q4PerLayerInputSet(ctx, driver, cfg.Layers[0].PerLayerInput, tokenID, hidden, epsilon)
	if err != nil {
		return nil, err
	}
	if len(inputs) < len(cfg.Layers) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "computed per-layer input count is smaller than forward layer count", nil)
	}
	return inputs, nil
}

func hipRunGemma4Q4PerLayerInputSet(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4PerLayerInputConfig, tokenID int32, hidden []float32, epsilon float32) ([][]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if !cfg.hasGlobalPrecompute() {
		return nil, nil
	}
	if !cfg.hasLayerApply() {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input precompute requires per-layer gate/projection tensors", nil)
	}
	if len(hidden) != cfg.ModelProjection.Cols {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input hidden length must match model projection cols", nil)
	}
	if cfg.InputSize <= 0 || cfg.ModelProjection.Rows%cfg.InputSize != 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input rows must align with input size", nil)
	}
	layerCount := cfg.ModelProjection.Rows / cfg.InputSize
	if layerCount <= 0 || cfg.Embedding.HiddenSize != cfg.ModelProjection.Rows {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input global shape mismatch", nil)
	}
	perLayerEmbedding, err := hipRunEmbeddingLookupKernelWithDeviceTable(ctx, driver, []int32{tokenID}, cfg.Embedding)
	if err != nil {
		return nil, err
	}
	perLayerEmbedding, err = hipRunVectorScaleKernel(ctx, driver, hipVectorScaleRequest{
		Input: perLayerEmbedding,
		Scale: cfg.embeddingScale(),
	})
	if err != nil {
		return nil, err
	}
	projected, err := hipRunProjectionKernelWithDeviceWeightEncoding(
		ctx,
		driver,
		hidden,
		cfg.ModelProjection.WeightPointer,
		cfg.ModelProjection.WeightBytes,
		cfg.ModelProjection.Rows,
		cfg.ModelProjection.Cols,
		hipProjectionWeightEncodingBF16,
	)
	if err != nil {
		return nil, err
	}
	projected, err = hipRunVectorScaleKernel(ctx, driver, hipVectorScaleRequest{
		Input: projected,
		Scale: cfg.modelProjectionScale(),
	})
	if err != nil {
		return nil, err
	}
	outputs := make([][]float32, 0, layerCount)
	for layer := 0; layer < layerCount; layer++ {
		start := layer * cfg.InputSize
		end := start + cfg.InputSize
		normCfg := cfg.ProjectionNorm
		normCfg.Epsilon = epsilon
		projectedNorm, err := hipRunRMSNormKernelWithDeviceWeightConfig(ctx, driver, projected[start:end], normCfg)
		if err != nil {
			return nil, err
		}
		combined, err := hipRunVectorAddKernel(ctx, driver, hipVectorAddRequest{
			Left:  projectedNorm,
			Right: perLayerEmbedding[start:end],
		})
		if err != nil {
			return nil, err
		}
		combined, err = hipRunVectorScaleKernel(ctx, driver, hipVectorScaleRequest{
			Input: combined,
			Scale: hipGemma4Q4PerLayerCombineScale,
		})
		if err != nil {
			return nil, err
		}
		outputs = append(outputs, combined)
	}
	return outputs, nil
}

func hipRunGemma4Q4PerLayerInputDeviceSet(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokenID int32, tokenIDDeviceBuffer, hidden *hipDeviceByteBuffer, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipGemma4Q4PerLayerInputDeviceSet, error) {
	if len(cfg.Layers) == 0 || !cfg.Layers[0].PerLayerInput.hasGlobalPrecompute() {
		return nil, nil
	}
	inputs, err := hipRunGemma4Q4PerLayerInputConfigDeviceSet(ctx, driver, cfg.Layers[0].PerLayerInput, tokenID, tokenIDDeviceBuffer, hidden, epsilon, workspace)
	if err != nil {
		return nil, err
	}
	if inputs.LayerCount() < len(cfg.Layers) {
		_ = inputs.Close()
		return nil, core.E(hipGemma4Q4Layer0Operation, "computed per-layer input count is smaller than forward layer count", nil)
	}
	return inputs, nil
}

func hipRunGemma4Q4PerLayerInputConfigDeviceSet(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4PerLayerInputConfig, tokenID int32, tokenIDDeviceBuffer, hidden *hipDeviceByteBuffer, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipGemma4Q4PerLayerInputDeviceSet, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if !cfg.hasGlobalPrecompute() {
		return nil, nil
	}
	if !cfg.hasLayerApply() {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input precompute requires per-layer gate/projection tensors", nil)
	}
	if hidden == nil || hidden.Pointer() == 0 || hidden.Count() != cfg.ModelProjection.Cols || hidden.SizeBytes() != uint64(cfg.ModelProjection.Cols*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input hidden device buffer shape mismatch", nil)
	}
	if cfg.InputSize <= 0 || cfg.ModelProjection.Rows%cfg.InputSize != 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input rows must align with input size", nil)
	}
	layerCount := cfg.ModelProjection.Rows / cfg.InputSize
	if layerCount <= 0 || cfg.Embedding.HiddenSize != cfg.ModelProjection.Rows {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input global shape mismatch", nil)
	}
	var err error
	var perLayerEmbeddingScaled *hipDeviceByteBuffer
	if workspace != nil {
		perLayerEmbeddingScaled, err = workspace.EnsurePerLayerScaled(driver, cfg.ModelProjection.Rows)
		if err == nil && tokenIDDeviceBuffer != nil {
			err = hipRunEmbeddingLookupKernelWithDeviceTableGreedyTokenScaledOutputWithWorkspace(ctx, driver, cfg.Embedding, tokenIDDeviceBuffer, perLayerEmbeddingScaled, cfg.embeddingScale(), workspace)
		} else if err == nil {
			tokenBuffer, tokenErr := workspace.EnsureTokenIDValue(driver, tokenID, cfg.Embedding.VocabSize)
			if tokenErr != nil {
				return nil, tokenErr
			}
			err = hipRunEmbeddingLookupKernelWithDeviceTableTokenBufferScaledOutputWithWorkspace(ctx, driver, cfg.Embedding, tokenBuffer, perLayerEmbeddingScaled, cfg.embeddingScale(), workspace)
		}
		if err != nil {
			return nil, err
		}
	} else {
		var perLayerEmbedding *hipDeviceByteBuffer
		perLayerEmbedding, err = hipRunEmbeddingLookupKernelWithDeviceTableBuffer(ctx, driver, []int32{tokenID}, cfg.Embedding)
		if err != nil {
			return nil, err
		}
		defer perLayerEmbedding.Close()
		perLayerEmbeddingScaled, err = hipRunVectorScaleDeviceKernel(ctx, driver, perLayerEmbedding, cfg.embeddingScale())
		if err != nil {
			return nil, err
		}
		defer perLayerEmbeddingScaled.Close()
	}
	var projected *hipDeviceByteBuffer
	if workspace != nil {
		projected, err = workspace.EnsurePerLayerProjected(driver, cfg.ModelProjection.Rows)
		if err == nil {
			err = hipRunProjectionKernelWithDeviceInputWeightEncodingOutput(
				ctx,
				driver,
				hidden,
				cfg.ModelProjection.WeightPointer,
				cfg.ModelProjection.WeightBytes,
				cfg.ModelProjection.Rows,
				cfg.ModelProjection.Cols,
				hipProjectionWeightEncodingBF16,
				projected,
			)
		}
	} else {
		projected, err = hipRunProjectionKernelWithDeviceInputWeightEncoding(
			ctx,
			driver,
			hidden,
			cfg.ModelProjection.WeightPointer,
			cfg.ModelProjection.WeightBytes,
			cfg.ModelProjection.Rows,
			cfg.ModelProjection.Cols,
			hipProjectionWeightEncodingBF16,
		)
	}
	if err != nil {
		return nil, err
	}
	if workspace == nil {
		defer projected.Close()
	}
	var projectedScaled *hipDeviceByteBuffer
	if workspace != nil {
		projectedScaled = projected
		err = hipRunVectorScaleDeviceKernelOutputWithWorkspace(ctx, driver, projected, cfg.modelProjectionScale(), projectedScaled, workspace)
	} else {
		projectedScaled, err = hipRunVectorScaleDeviceKernel(ctx, driver, projected, cfg.modelProjectionScale())
	}
	if err != nil {
		return nil, err
	}
	if workspace == nil {
		defer projectedScaled.Close()
	}

	normCfg := cfg.ProjectionNorm
	normCfg.Epsilon = epsilon
	normCfg.Count = cfg.InputSize
	var projectedNorm *hipDeviceByteBuffer
	if workspace != nil {
		projectedNorm = projectedScaled
		err = hipRunRMSNormHeadsKernelWithDeviceInputWeightConfigOutputWithWorkspace(ctx, driver, projectedScaled, normCfg, layerCount, projectedNorm, workspace)
	} else {
		projectedNorm, err = hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, driver, projectedScaled, normCfg, layerCount)
	}
	if err != nil {
		return nil, err
	}
	if workspace == nil {
		defer projectedNorm.Close()
	}
	addScale := hipGemma4Q4PerLayerCombineScale
	var scaled *hipDeviceByteBuffer
	if workspace != nil {
		scaled = projectedNorm
		err = hipRunVectorAddScaledDeviceKernelOutputWithWorkspace(ctx, driver, projectedNorm, perLayerEmbeddingScaled, addScale, scaled, workspace)
	} else {
		scaled, err = hipRunVectorAddScaledDeviceKernel(ctx, driver, projectedNorm, perLayerEmbeddingScaled, addScale)
	}
	if err != nil {
		return nil, err
	}

	if workspace != nil {
		return workspace.BorrowPerLayerInputDeviceSet(driver, layerCount, cfg.InputSize, scaled)
	}
	outputs := &hipGemma4Q4PerLayerInputDeviceSet{
		driver:           driver,
		layerCount:       layerCount,
		layerStrideBytes: uint64(cfg.InputSize * 4),
		layerValueCount:  cfg.InputSize,
		viewLabel:        "per-layer input slice",
		borrowedBacking:  workspace != nil,
		Backing:          []*hipDeviceByteBuffer{scaled},
	}
	success := false
	defer func() {
		if !success {
			_ = outputs.Close()
		}
	}()
	success = true
	return outputs, nil
}

func hipGemma4Q4HostGELU(input []float32) ([]float32, error) {
	if len(input) == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "GELU input is required", nil)
	}
	if !rocmFloat32SliceFinite(input) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "GELU input values must be finite", nil)
	}
	output := make([]float32, len(input))
	const sqrt2OverPi = 0.7978845608028654
	const coeff = 0.044715
	for index, value := range input {
		x := float64(value)
		output[index] = float32(0.5 * x * (1 + math.Tanh(sqrt2OverPi*(x+coeff*x*x*x))))
	}
	return output, nil
}

func hipGemma4Q4HostMultiply(left, right []float32) ([]float32, error) {
	if len(left) == 0 || len(left) != len(right) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "multiply inputs must have matching positive lengths", nil)
	}
	if !rocmFloat32SliceFinite(left) || !rocmFloat32SliceFinite(right) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "multiply inputs must be finite", nil)
	}
	output := make([]float32, len(left))
	for index := range left {
		output[index] = left[index] * right[index]
	}
	return output, nil
}

func hipGemma4Q4DecodeQuantLabel(cfg hipGemma4Q4Layer0Config) string {
	bits := cfg.Embedding.QuantBits
	if bits == 0 {
		bits = cfg.QueryProjection.Bits
	}
	if bits == 0 {
		bits = cfg.LMHeadProjection.Bits
	}
	return core.Sprintf("mlx_q%d", hipMLXQ4ProjectionBitsOrDefault(bits))
}

func hipGemma4Q4Layer0Labels(cfg hipGemma4Q4Layer0Config, req hipGemma4Q4Layer0Request) map[string]string {
	labels := map[string]string{
		"gemma4_q4_layer0_kernel": hipKernelStatusLinked,
		"gemma4_q4_layer0_name":   "rocm_gemma4_q4_layer0_smoke",
		"decode_architecture":     "gemma4",
		"decode_tensor_backing":   "loaded_device",
		"decode_quant":            hipGemma4Q4DecodeQuantLabel(cfg),
		"decode_layer":            core.Sprintf("%d", cfg.Layer),
		"decode_position":         core.Sprintf("%d", req.Position),
		"decode_vocab_size":       core.Sprintf("%d", cfg.VocabSize),
		"decode_hidden_size":      core.Sprintf("%d", cfg.HiddenSize),
		"final_logit_softcap":     core.Sprintf("%g", cfg.FinalLogitSoftcap),
		"decode_primitives":       "embedding_lookup,vector_scale,rms_norm,mlx_q4_projection,rope,attention,vector_add,gelu_tanh_mlp,logit_softcap,greedy",
		"gemma4_mlp_activation":   "device_gelu_tanh_multiply",
		"production_decode":       hipKernelStatusNotLinked,
	}
	if cfg.PerLayerInput.hasGlobalPrecompute() {
		labels["gemma4_per_layer_inputs"] = hipKernelStatusLinked
		labels["gemma4_per_layer_input_size"] = core.Sprintf("%d", cfg.PerLayerInput.InputSize)
		labels["gemma4_per_layer_input_activation"] = "device_gelu_tanh_multiply"
		labels["decode_primitives"] += ",gemma4_per_layer_input"
	}
	if cfg.LayerType != "" {
		labels["gemma4_q4_layer_type"] = cfg.LayerType
	}
	return labels
}

func hipGemma4Q4ForwardLabels(cfg hipGemma4Q4ForwardConfig, req hipGemma4Q4ForwardRequest) map[string]string {
	first := cfg.Layers[0]
	labels := map[string]string{
		"gemma4_q4_forward_kernel": hipKernelStatusLinked,
		"gemma4_q4_forward_name":   "rocm_gemma4_q4_single_token_forward_smoke",
		"decode_architecture":      "gemma4",
		"decode_tensor_backing":    "loaded_device",
		"decode_quant":             hipGemma4Q4DecodeQuantLabel(first),
		"decode_layers":            core.Sprintf("%d", len(cfg.Layers)),
		"decode_position":          core.Sprintf("%d", req.Position),
		"decode_vocab_size":        core.Sprintf("%d", first.VocabSize),
		"decode_hidden_size":       core.Sprintf("%d", first.HiddenSize),
		"final_logit_softcap":      core.Sprintf("%g", first.FinalLogitSoftcap),
		"decode_primitives":        "embedding_lookup,vector_scale,rms_norm,mlx_q4_projection,rope,attention,vector_add,gelu_tanh_mlp,logit_softcap,greedy",
		"gemma4_mlp_activation":    "device_gelu_tanh_multiply",
		"production_decode":        hipKernelStatusNotLinked,
	}
	if req.DeviceKVAttention {
		labels["attention_kv_backing"] = "hip_device_descriptor"
		labels["attention_kv_mode"] = firstNonEmptyString(req.DeviceKVMode, rocmKVCacheModeFP16)
		labels["production_kv_cache_backing"] = hipKernelStatusNotLinked
	}
	if first.PerLayerInput.hasGlobalPrecompute() {
		labels["gemma4_per_layer_inputs"] = hipKernelStatusLinked
		labels["gemma4_per_layer_input_size"] = core.Sprintf("%d", first.PerLayerInput.InputSize)
		labels["gemma4_per_layer_input_activation"] = "device_gelu_tanh_multiply"
		labels["decode_primitives"] += ",gemma4_per_layer_input"
	}
	if cfg.KVSharedLayers > 0 {
		labels["gemma4_q4_kv_shared_layers"] = core.Sprintf("%d", cfg.KVSharedLayers)
		labels["decode_primitives"] += ",gemma4_shared_kv"
	}
	return labels
}

func hipGemma4Q4GreedyDecodeLabels(cfg hipGemma4Q4ForwardConfig, req hipGemma4Q4GreedyDecodeRequest, state hipGemma4Q4DecodeState) map[string]string {
	first := cfg.Layers[0]
	labels := map[string]string{
		"gemma4_q4_decode_kernel":     hipKernelStatusLinked,
		"gemma4_q4_decode_name":       "rocm_gemma4_q4_greedy_decode_smoke",
		"decode_architecture":         "gemma4",
		"decode_tensor_backing":       "loaded_device",
		"decode_quant":                hipGemma4Q4DecodeQuantLabel(first),
		"decode_layers":               core.Sprintf("%d", len(cfg.Layers)),
		"decode_prompt_tokens":        core.Sprintf("%d", len(req.PromptTokenIDs)),
		"decode_generated_tokens":     core.Sprintf("%d", req.MaxNewTokens),
		"decode_forward_steps":        core.Sprintf("%d", len(req.PromptTokenIDs)+req.MaxNewTokens-1),
		"decode_state_tokens":         core.Sprintf("%d", state.tokenCountForConfig(cfg)),
		"decode_vocab_size":           core.Sprintf("%d", first.VocabSize),
		"decode_hidden_size":          core.Sprintf("%d", first.HiddenSize),
		"final_logit_softcap":         core.Sprintf("%g", first.FinalLogitSoftcap),
		"decode_primitives":           "embedding_lookup,vector_scale,rms_norm,mlx_q4_projection,rope,attention,kv_state,vector_add,gelu_tanh_mlp,logit_softcap,greedy",
		"gemma4_mlp_activation":       "device_gelu_tanh_multiply",
		"production_decode":           hipKernelStatusNotLinked,
		"production_kv_cache_backing": hipKernelStatusNotLinked,
	}
	if first.PerLayerInput.hasGlobalPrecompute() {
		labels["gemma4_per_layer_inputs"] = hipKernelStatusLinked
		labels["gemma4_per_layer_input_size"] = core.Sprintf("%d", first.PerLayerInput.InputSize)
		labels["gemma4_per_layer_input_activation"] = "device_gelu_tanh_multiply"
		labels["decode_primitives"] += ",gemma4_per_layer_input"
	}
	if cfg.KVSharedLayers > 0 {
		labels["gemma4_q4_kv_shared_layers"] = core.Sprintf("%d", cfg.KVSharedLayers)
		labels["decode_primitives"] += ",gemma4_shared_kv"
	}
	return labels
}

func hipValidateGemma4Q4NormConfig(label string, cfg hipRMSNormDeviceWeightConfig, count int) error {
	if cfg.WeightPointer == 0 {
		return core.E(hipGemma4Q4Layer0Operation, label+" weight pointer is required", nil)
	}
	if cfg.Count != count {
		return core.E(hipGemma4Q4Layer0Operation, label+" count does not match layer geometry", nil)
	}
	switch cfg.WeightEncoding {
	case hipRMSNormWeightEncodingF32:
		if cfg.WeightBytes != uint64(count*4) {
			return core.E(hipGemma4Q4Layer0Operation, label+" F32 weight byte count mismatch", nil)
		}
	case hipRMSNormWeightEncodingBF16:
		if cfg.WeightBytes != uint64(count*2) {
			return core.E(hipGemma4Q4Layer0Operation, label+" BF16 weight byte count mismatch", nil)
		}
	default:
		return core.E(hipGemma4Q4Layer0Operation, label+" weight encoding must be F32 or BF16", nil)
	}
	if cfg.Flags&hipRMSNormLaunchFlagAddUnitWeight != 0 {
		return core.E(hipGemma4Q4Layer0Operation, label+" must use raw Gemma4 RMSNorm weights", nil)
	}
	return nil
}

func (model *hipLoadedModel) loadedGemma4Q4EmbeddingConfig(groupSize int) (hipDeviceEmbeddingLookupConfig, error) {
	weight, err := model.requiredHIPTensor("language_model.model.embed_tokens.weight", "embed_tokens weight")
	if err != nil {
		return hipDeviceEmbeddingLookupConfig{}, err
	}
	scales, err := model.requiredHIPTensor("language_model.model.embed_tokens.scales", "embed_tokens scales")
	if err != nil {
		return hipDeviceEmbeddingLookupConfig{}, err
	}
	biases, err := model.requiredHIPTensor("language_model.model.embed_tokens.biases", "embed_tokens biases")
	if err != nil {
		return hipDeviceEmbeddingLookupConfig{}, err
	}
	vocab := model.modelInfo.VocabSize
	hidden := model.modelInfo.HiddenSize
	bits, rows, cols, effectiveGroupSize, groups, packedCols, err := hipInferMLXAffineBitsFromTensorShapes(weight, scales, biases, groupSize, model.modelInfo.QuantBits, "embed_tokens")
	if err != nil {
		return hipDeviceEmbeddingLookupConfig{}, err
	}
	if rows != vocab || cols != hidden {
		return hipDeviceEmbeddingLookupConfig{}, core.E(hipGemma4Q4Layer0Operation, "embed_tokens dimensions must match Gemma4 config", nil)
	}
	if err := hipValidateGemma4Q4Tensor(weight, "embed_tokens weight", "U32", vocab, packedCols, uint64(vocab)*uint64(packedCols)*4); err != nil {
		return hipDeviceEmbeddingLookupConfig{}, err
	}
	if err := hipValidateGemma4Q4Tensor(scales, "embed_tokens scales", "BF16", vocab, groups, uint64(vocab)*uint64(groups)*2); err != nil {
		return hipDeviceEmbeddingLookupConfig{}, err
	}
	if err := hipValidateGemma4Q4Tensor(biases, "embed_tokens biases", "BF16", vocab, groups, uint64(vocab)*uint64(groups)*2); err != nil {
		return hipDeviceEmbeddingLookupConfig{}, err
	}
	return hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: weight.pointer,
		EmbeddingBytes:   weight.info.ByteSize,
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		VocabSize:        vocab,
		HiddenSize:       hidden,
		GroupSize:        effectiveGroupSize,
		QuantBits:        bits,
		ScalePointer:     scales.pointer,
		BiasPointer:      biases.pointer,
		ScaleBytes:       scales.info.ByteSize,
		BiasBytes:        biases.info.ByteSize,
	}, nil
}

func (model *hipLoadedModel) loadedGemma4Q4PerLayerInputConfig(layerPrefix string, layer, groupSize, hidden int) (hipGemma4Q4PerLayerInputConfig, error) {
	if model == nil {
		return hipGemma4Q4PerLayerInputConfig{}, core.E(hipGemma4Q4Layer0Operation, "loaded model is required", nil)
	}
	globalName := "language_model.model.embed_tokens_per_layer.weight"
	layerGateName := layerPrefix + ".per_layer_input_gate.weight"
	if !model.hasHIPTensor(globalName) && !model.hasHIPTensor(layerGateName) {
		return hipGemma4Q4PerLayerInputConfig{}, nil
	}
	if model.modelInfo.NumLayers <= 0 {
		return hipGemma4Q4PerLayerInputConfig{}, core.E(hipGemma4Q4Layer0Operation, "per-layer inputs require model layer count", nil)
	}
	if hidden <= 0 || groupSize <= 0 {
		return hipGemma4Q4PerLayerInputConfig{}, core.E(hipGemma4Q4Layer0Operation, "per-layer input hidden and group sizes must be positive", nil)
	}
	embedding, inputSize, err := model.loadedGemma4Q4PerLayerEmbeddingConfig(groupSize, model.modelInfo.NumLayers)
	if err != nil {
		return hipGemma4Q4PerLayerInputConfig{}, err
	}
	modelProjection, err := model.loadedGemma4BF16ProjectionConfig(
		"language_model.model.per_layer_model_projection.weight",
		"per_layer_model_projection",
		embedding.HiddenSize,
		hidden,
	)
	if err != nil {
		return hipGemma4Q4PerLayerInputConfig{}, err
	}
	projectionNorm, err := model.loadedGemma4NormConfig("language_model.model.per_layer_projection_norm.weight", "per_layer_projection_norm", inputSize)
	if err != nil {
		return hipGemma4Q4PerLayerInputConfig{}, err
	}
	inputGate, gateRows, gateCols, err := model.loadedGemma4Q4ProjectionConfig(layerPrefix+".per_layer_input_gate", "per_layer_input_gate", groupSize)
	if err != nil {
		return hipGemma4Q4PerLayerInputConfig{}, err
	}
	projection, projectionRows, projectionCols, err := model.loadedGemma4Q4ProjectionConfig(layerPrefix+".per_layer_projection", "per_layer_projection", groupSize)
	if err != nil {
		return hipGemma4Q4PerLayerInputConfig{}, err
	}
	postNorm, err := model.loadedGemma4NormConfig(layerPrefix+".post_per_layer_input_norm.weight", "post_per_layer_input_norm", hidden)
	if err != nil {
		return hipGemma4Q4PerLayerInputConfig{}, err
	}
	if gateRows != inputSize || gateCols != hidden ||
		projectionRows != hidden || projectionCols != inputSize ||
		layer < 0 || layer >= model.modelInfo.NumLayers {
		return hipGemma4Q4PerLayerInputConfig{}, core.E(hipGemma4Q4Layer0Operation, "per-layer input tensor shapes are inconsistent", nil)
	}
	cfg := hipGemma4Q4PerLayerInputConfig{
		InputSize:       inputSize,
		Embedding:       embedding,
		ModelProjection: modelProjection,
		ProjectionNorm:  projectionNorm,
		InputGate:       inputGate,
		Projection:      projection,
		PostInputNorm:   postNorm,
	}
	cfg.finalizeScales()
	if err := (hipGemma4Q4Layer0Config{
		Layer:         layer,
		HiddenSize:    hidden,
		VocabSize:     model.modelInfo.VocabSize,
		GroupSize:     groupSize,
		PerLayerInput: cfg,
	}).validatePerLayerInput(); err != nil {
		return hipGemma4Q4PerLayerInputConfig{}, err
	}
	return cfg, nil
}

func (model *hipLoadedModel) loadedGemma4Q4PerLayerEmbeddingConfig(groupSize, numLayers int) (hipDeviceEmbeddingLookupConfig, int, error) {
	weight, err := model.requiredHIPTensor("language_model.model.embed_tokens_per_layer.weight", "embed_tokens_per_layer weight")
	if err != nil {
		return hipDeviceEmbeddingLookupConfig{}, 0, err
	}
	scales, err := model.requiredHIPTensor("language_model.model.embed_tokens_per_layer.scales", "embed_tokens_per_layer scales")
	if err != nil {
		return hipDeviceEmbeddingLookupConfig{}, 0, err
	}
	biases, err := model.requiredHIPTensor("language_model.model.embed_tokens_per_layer.biases", "embed_tokens_per_layer biases")
	if err != nil {
		return hipDeviceEmbeddingLookupConfig{}, 0, err
	}
	if weight.info.TypeName != "U32" || len(weight.info.Dimensions) != 2 {
		return hipDeviceEmbeddingLookupConfig{}, 0, core.E(hipGemma4Q4Layer0Operation, "embed_tokens_per_layer weight must be U32 rank-2 MLX affine packed tensor", nil)
	}
	bits, vocab, hiddenTotal, effectiveGroupSize, groups, packedCols, err := hipInferMLXAffineBitsFromTensorShapes(weight, scales, biases, groupSize, model.modelInfo.QuantBits, "embed_tokens_per_layer")
	if err != nil {
		return hipDeviceEmbeddingLookupConfig{}, 0, err
	}
	if vocab <= 0 || hiddenTotal <= 0 || numLayers <= 0 || hiddenTotal%numLayers != 0 {
		return hipDeviceEmbeddingLookupConfig{}, 0, core.E(hipGemma4Q4Layer0Operation, "embed_tokens_per_layer dimensions must align with layer count", nil)
	}
	inputSize := hiddenTotal / numLayers
	if model.gemma4TextConfig.HiddenSizePerLayerInput > 0 && inputSize != model.gemma4TextConfig.HiddenSizePerLayerInput {
		return hipDeviceEmbeddingLookupConfig{}, 0, core.E(hipGemma4Q4Layer0Operation, "embed_tokens_per_layer hidden size does not match Gemma4 config", nil)
	}
	if model.gemma4TextConfig.VocabSizePerLayerInput > 0 && vocab != model.gemma4TextConfig.VocabSizePerLayerInput {
		return hipDeviceEmbeddingLookupConfig{}, 0, core.E(hipGemma4Q4Layer0Operation, "embed_tokens_per_layer vocab size does not match Gemma4 config", nil)
	}
	if err := hipValidateGemma4Q4Tensor(weight, "embed_tokens_per_layer weight", "U32", vocab, packedCols, uint64(vocab)*uint64(packedCols)*4); err != nil {
		return hipDeviceEmbeddingLookupConfig{}, 0, err
	}
	if err := hipValidateGemma4Q4Tensor(scales, "embed_tokens_per_layer scales", "BF16", vocab, groups, uint64(vocab)*uint64(groups)*2); err != nil {
		return hipDeviceEmbeddingLookupConfig{}, 0, err
	}
	if err := hipValidateGemma4Q4Tensor(biases, "embed_tokens_per_layer biases", "BF16", vocab, groups, uint64(vocab)*uint64(groups)*2); err != nil {
		return hipDeviceEmbeddingLookupConfig{}, 0, err
	}
	return hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: weight.pointer,
		EmbeddingBytes:   weight.info.ByteSize,
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		VocabSize:        vocab,
		HiddenSize:       hiddenTotal,
		GroupSize:        effectiveGroupSize,
		QuantBits:        bits,
		ScalePointer:     scales.pointer,
		BiasPointer:      biases.pointer,
		ScaleBytes:       scales.info.ByteSize,
		BiasBytes:        biases.info.ByteSize,
	}, inputSize, nil
}

func (model *hipLoadedModel) loadedGemma4BF16ProjectionConfig(name, label string, rows, cols int) (hipBF16DeviceWeightConfig, error) {
	tensor, err := model.requiredHIPTensor(name, label)
	if err != nil {
		return hipBF16DeviceWeightConfig{}, err
	}
	if tensor.info.TypeName != "BF16" ||
		len(tensor.info.Dimensions) != 2 ||
		tensor.info.Dimensions[0] != uint64(rows) ||
		tensor.info.Dimensions[1] != uint64(cols) ||
		tensor.info.ByteSize != uint64(rows)*uint64(cols)*2 {
		return hipBF16DeviceWeightConfig{}, core.E(hipGemma4Q4Layer0Operation, label+" tensor shape/type mismatch", nil)
	}
	cfg := hipBF16DeviceWeightConfig{
		WeightPointer: tensor.pointer,
		WeightBytes:   tensor.info.ByteSize,
		Rows:          rows,
		Cols:          cols,
	}
	if err := cfg.validate(hipProjectionWeightEncodingBF16); err != nil {
		return hipBF16DeviceWeightConfig{}, core.E(hipGemma4Q4Layer0Operation, label+" config", err)
	}
	return cfg, nil
}

func (model *hipLoadedModel) loadedGemma4Q4ProjectionConfig(baseName, label string, groupSize int) (hipMLXQ4DeviceWeightConfig, int, int, error) {
	weight, err := model.requiredHIPTensor(baseName+".weight", label+" weight")
	if err != nil {
		return hipMLXQ4DeviceWeightConfig{}, 0, 0, err
	}
	scales, err := model.requiredHIPTensor(baseName+".scales", label+" scales")
	if err != nil {
		return hipMLXQ4DeviceWeightConfig{}, 0, 0, err
	}
	biases, err := model.requiredHIPTensor(baseName+".biases", label+" biases")
	if err != nil {
		return hipMLXQ4DeviceWeightConfig{}, 0, 0, err
	}
	if weight.pointer == 0 || scales.pointer == 0 || biases.pointer == 0 {
		return hipMLXQ4DeviceWeightConfig{}, 0, 0, core.E(hipGemma4Q4Layer0Operation, label+" MLX affine tensor pointers are required", nil)
	}
	bits, rows, cols, effectiveGroupSize, groups, packedCols, err := hipInferMLXAffineBitsFromTensorShapes(weight, scales, biases, groupSize, model.modelInfo.QuantBits, label)
	if err != nil {
		return hipMLXQ4DeviceWeightConfig{}, 0, 0, err
	}
	if err := hipValidateGemma4Q4Tensor(weight, label+" weight", "U32", rows, packedCols, uint64(rows)*uint64(packedCols)*4); err != nil {
		return hipMLXQ4DeviceWeightConfig{}, 0, 0, err
	}
	if err := hipValidateGemma4Q4Tensor(scales, label+" scales", "BF16", rows, groups, uint64(rows)*uint64(groups)*2); err != nil {
		return hipMLXQ4DeviceWeightConfig{}, 0, 0, err
	}
	if err := hipValidateGemma4Q4Tensor(biases, label+" biases", "BF16", rows, groups, uint64(rows)*uint64(groups)*2); err != nil {
		return hipMLXQ4DeviceWeightConfig{}, 0, 0, err
	}
	cfg := hipMLXQ4DeviceWeightConfig{
		WeightPointer: weight.pointer,
		ScalePointer:  scales.pointer,
		BiasPointer:   biases.pointer,
		WeightBytes:   weight.info.ByteSize,
		ScaleBytes:    scales.info.ByteSize,
		BiasBytes:     biases.info.ByteSize,
		Rows:          rows,
		Cols:          cols,
		GroupSize:     effectiveGroupSize,
		Bits:          bits,
	}
	if err := cfg.validateInputCount(cols); err != nil {
		return hipMLXQ4DeviceWeightConfig{}, 0, 0, core.E(hipGemma4Q4Layer0Operation, label+" MLX affine config", err)
	}
	return cfg, rows, cols, nil
}

func hipInferMLXAffineBitsFromTensorShapes(weight, scales, biases hipTensor, groupSize, preferredBits int, label string) (int, int, int, int, int, int, error) {
	if groupSize <= 0 {
		return 0, 0, 0, 0, 0, 0, core.E(hipGemma4Q4Layer0Operation, label+" MLX affine group size must be positive", nil)
	}
	if weight.info.TypeName != "U32" || len(weight.info.Dimensions) != 2 {
		return 0, 0, 0, 0, 0, 0, core.E(hipGemma4Q4Layer0Operation, label+" weight must be U32 rank-2 MLX affine packed tensor", nil)
	}
	if scales.info.TypeName != "BF16" || len(scales.info.Dimensions) != 2 {
		return 0, 0, 0, 0, 0, 0, core.E(hipGemma4Q4Layer0Operation, label+" scales must be BF16 rank-2 MLX affine tensor", nil)
	}
	if biases.info.TypeName != "BF16" || len(biases.info.Dimensions) != 2 {
		return 0, 0, 0, 0, 0, 0, core.E(hipGemma4Q4Layer0Operation, label+" biases must be BF16 rank-2 MLX affine tensor", nil)
	}
	rows := int(weight.info.Dimensions[0])
	packedCols := int(weight.info.Dimensions[1])
	scaleRows := int(scales.info.Dimensions[0])
	groups := int(scales.info.Dimensions[1])
	biasRows := int(biases.info.Dimensions[0])
	biasGroups := int(biases.info.Dimensions[1])
	if rows <= 0 || packedCols <= 0 || groups <= 0 {
		return 0, 0, 0, 0, 0, 0, core.E(hipGemma4Q4Layer0Operation, label+" MLX affine dimensions must be positive", nil)
	}
	if scaleRows != rows || biasRows != rows || biasGroups != groups {
		return 0, 0, 0, 0, 0, 0, core.E(hipGemma4Q4Layer0Operation, label+" MLX affine tensor shapes must agree", nil)
	}
	for _, candidateGroupSize := range hipMLXAffineCandidateGroupSizes(groupSize) {
		cols := groups * candidateGroupSize
		if cols <= 0 || cols/candidateGroupSize != groups {
			continue
		}
		for _, bits := range hipMLXAffineCandidateBits(preferredBits) {
			wantPackedCols, err := hipMLXAffinePackedCols(cols, bits)
			if err == nil && wantPackedCols == packedCols {
				return bits, rows, cols, candidateGroupSize, groups, packedCols, nil
			}
		}
	}
	return 0, 0, 0, 0, 0, 0, core.E(hipGemma4Q4Layer0Operation, label+" MLX affine packed shape does not match supported bit widths", nil)
}

func hipMLXAffineCandidateBits(preferredBits int) []int {
	preferredBits = hipMLXQ4ProjectionBitsOrDefault(preferredBits)
	out := make([]int, 0, 4)
	for _, bits := range []int{preferredBits, 4, 6, 8} {
		if !hipMLXAffineSupportedBits(bits) {
			continue
		}
		seen := false
		for _, existing := range out {
			if existing == bits {
				seen = true
				break
			}
		}
		if !seen {
			out = append(out, bits)
		}
	}
	return out
}

func hipMLXAffineCandidateGroupSizes(preferredGroupSize int) []int {
	out := make([]int, 0, 4)
	for _, groupSize := range []int{preferredGroupSize, 16, 32, 64} {
		if groupSize <= 0 {
			continue
		}
		seen := false
		for _, existing := range out {
			if existing == groupSize {
				seen = true
				break
			}
		}
		if !seen {
			out = append(out, groupSize)
		}
	}
	return out
}

func (model *hipLoadedModel) loadedGemma4Q4LMHeadProjectionConfig(groupSize int) (hipMLXQ4DeviceWeightConfig, int, int, error) {
	for _, baseName := range []string{
		"language_model.lm_head",
		"language_model.model.lm_head",
		"lm_head",
	} {
		if model.hasHIPTensor(baseName + ".weight") {
			return model.loadedGemma4Q4ProjectionConfig(baseName, "lm_head", groupSize)
		}
	}
	return model.loadedGemma4Q4ProjectionConfig("language_model.model.embed_tokens", "embed_tokens_lm_head", groupSize)
}

func (model *hipLoadedModel) loadedGemma4NormConfig(name, label string, count int) (hipRMSNormDeviceWeightConfig, error) {
	tensor, err := model.requiredHIPTensor(name, label)
	if err != nil {
		return hipRMSNormDeviceWeightConfig{}, err
	}
	encoding, expectedBytes, err := hipGemma4NormWeightEncodingAndBytes(tensor.info, label, count)
	if err != nil {
		return hipRMSNormDeviceWeightConfig{}, err
	}
	if len(tensor.info.Dimensions) != 1 ||
		tensor.info.Dimensions[0] != uint64(count) ||
		tensor.info.ByteSize != expectedBytes {
		return hipRMSNormDeviceWeightConfig{}, core.E(hipGemma4Q4Layer0Operation, label+" tensor shape/type mismatch", nil)
	}
	if tensor.pointer == 0 {
		return hipRMSNormDeviceWeightConfig{}, core.E(hipGemma4Q4Layer0Operation, label+" tensor pointer is required", nil)
	}
	if err := hipValidateGemma4Q4TensorBytes(label, tensor.info.ByteSize, expectedBytes); err != nil {
		return hipRMSNormDeviceWeightConfig{}, err
	}
	cfg := hipRMSNormDeviceWeightConfig{
		WeightPointer:  tensor.pointer,
		WeightBytes:    tensor.info.ByteSize,
		Count:          count,
		WeightEncoding: encoding,
	}
	if err := hipValidateGemma4Q4NormConfig(label, cfg, count); err != nil {
		return hipRMSNormDeviceWeightConfig{}, err
	}
	return cfg, nil
}

func hipGemma4NormWeightEncodingAndBytes(info nativeTensorInfo, label string, count int) (uint32, uint64, error) {
	if count <= 0 {
		return 0, 0, core.E(hipGemma4Q4Layer0Operation, label+" count must be positive", nil)
	}
	switch core.Upper(info.TypeName) {
	case "F32":
		return hipRMSNormWeightEncodingF32, uint64(count) * 4, nil
	case "BF16":
		return hipRMSNormWeightEncodingBF16, uint64(count) * 2, nil
	default:
		return 0, 0, core.E(hipGemma4Q4Layer0Operation, label+" tensor must be F32 or BF16", nil)
	}
}

func (model *hipLoadedModel) loadedGemma4Q4LayerScalar(layer int) (float32, error) {
	if model == nil || model.driver == nil {
		return 0, core.E(hipGemma4Q4Layer0Operation, "loaded model is required", nil)
	}
	name := core.Sprintf("language_model.model.layers.%d.layer_scalar", layer)
	tensor, ok := model.tensors[name]
	if !ok {
		return 1, nil
	}
	scalarBytes, err := hipGemma4LayerScalarBytes(tensor.info, "layer scalar tensor")
	if err != nil {
		return 0, err
	}
	if tensor.pointer == 0 {
		return 0, core.E(hipGemma4Q4Layer0Operation, "layer scalar tensor pointer is required", nil)
	}
	payload := make([]byte, scalarBytes)
	if err := model.driver.CopyDeviceToHost(tensor.pointer, payload); err != nil {
		return 0, core.E(hipGemma4Q4Layer0Operation, "copy layer scalar", err)
	}
	return hipGemma4LayerScalarValue(tensor.info, payload)
}

func hipGemma4LayerScalarBytes(info nativeTensorInfo, label string) (int, error) {
	if len(info.Dimensions) != 1 || info.Dimensions[0] != 1 {
		return 0, core.E(hipGemma4Q4Layer0Operation, label+" must be F32 or BF16 [1]", nil)
	}
	switch core.Upper(info.TypeName) {
	case "F32":
		if info.ByteSize != 4 {
			return 0, core.E(hipGemma4Q4Layer0Operation, label+" F32 byte count mismatch", nil)
		}
		return 4, nil
	case "BF16":
		if info.ByteSize != 2 {
			return 0, core.E(hipGemma4Q4Layer0Operation, label+" BF16 byte count mismatch", nil)
		}
		return 2, nil
	default:
		return 0, core.E(hipGemma4Q4Layer0Operation, label+" must be F32 or BF16 [1]", nil)
	}
}

func hipGemma4LayerScalarValue(info nativeTensorInfo, payload []byte) (float32, error) {
	switch core.Upper(info.TypeName) {
	case "F32":
		if len(payload) != 4 {
			return 0, core.E(hipGemma4Q4Layer0Operation, "layer scalar F32 payload byte count mismatch", nil)
		}
		return math.Float32frombits(binary.LittleEndian.Uint32(payload)), nil
	case "BF16":
		if len(payload) != 2 {
			return 0, core.E(hipGemma4Q4Layer0Operation, "layer scalar BF16 payload byte count mismatch", nil)
		}
		return hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload)), nil
	default:
		return 0, core.E(hipGemma4Q4Layer0Operation, "layer scalar tensor must be F32 or BF16", nil)
	}
}

func (model *hipLoadedModel) hasHIPTensor(name string) bool {
	_, ok := model.lookupHIPTensor(name)
	return ok
}

func (model *hipLoadedModel) requiredHIPTensor(name, label string) (hipTensor, error) {
	if model == nil {
		return hipTensor{}, core.E(hipGemma4Q4Layer0Operation, "loaded model is required", nil)
	}
	tensor, ok := model.lookupHIPTensor(name)
	if !ok {
		return hipTensor{}, core.E(hipGemma4Q4Layer0Operation, "loaded Gemma4 q4 model is missing "+label+" tensor", nil)
	}
	if tensor.pointer == 0 {
		return hipTensor{}, core.E(hipGemma4Q4Layer0Operation, label+" tensor pointer is required", nil)
	}
	return tensor, nil
}

func (model *hipLoadedModel) lookupHIPTensor(name string) (hipTensor, bool) {
	if model == nil {
		return hipTensor{}, false
	}
	if tensor, ok := model.tensors[name]; ok {
		return tensor, true
	}
	if !model.hipGGUFTensorAliasesEnabled() {
		return hipTensor{}, false
	}
	for _, alias := range hipGemma4GGUFTensorAliases(name) {
		if tensor, ok := model.tensors[alias]; ok {
			return tensor, true
		}
	}
	return hipTensor{}, false
}

func (model *hipLoadedModel) hipGGUFTensorAliasesEnabled() bool {
	if model == nil {
		return false
	}
	if rocmGemma4SourceFormatGGUF(model.modelLabels) {
		return true
	}
	path := core.Lower(core.Trim(model.modelPath))
	return core.HasSuffix(path, ".gguf") || core.Contains(path, "-gguf")
}

func hipGemma4GGUFTensorAliases(name string) []string {
	switch name {
	case "language_model.model.embed_tokens.weight":
		return []string{"token_embd.weight"}
	case "language_model.model.embed_tokens_per_layer.weight":
		return []string{"per_layer_token_embd.weight"}
	case "language_model.model.per_layer_model_projection.weight":
		return []string{"per_layer_model_proj.weight"}
	case "language_model.model.per_layer_projection_norm.weight":
		return []string{"per_layer_proj_norm.weight"}
	case "language_model.model.norm.weight":
		return []string{"output_norm.weight"}
	}

	const layerPrefix = "language_model.model.layers."
	if !core.HasPrefix(name, layerPrefix) {
		return nil
	}
	parts := core.Split(core.TrimPrefix(name, layerPrefix), ".")
	if len(parts) < 2 || parts[0] == "" {
		return nil
	}
	suffix := core.Join(".", parts[1:]...)
	ggufSuffix := ""
	switch suffix {
	case "self_attn.q_proj.weight":
		ggufSuffix = "attn_q.weight"
	case "self_attn.k_proj.weight":
		ggufSuffix = "attn_k.weight"
	case "self_attn.v_proj.weight":
		ggufSuffix = "attn_v.weight"
	case "self_attn.o_proj.weight":
		ggufSuffix = "attn_output.weight"
	case "self_attn.q_norm.weight":
		ggufSuffix = "attn_q_norm.weight"
	case "self_attn.k_norm.weight":
		ggufSuffix = "attn_k_norm.weight"
	case "input_layernorm.weight":
		ggufSuffix = "attn_norm.weight"
	case "post_attention_layernorm.weight":
		ggufSuffix = "post_attention_norm.weight"
	case "pre_feedforward_layernorm.weight":
		ggufSuffix = "ffn_norm.weight"
	case "post_feedforward_layernorm.weight":
		ggufSuffix = "post_ffw_norm.weight"
	case "pre_feedforward_layernorm_2.weight":
		ggufSuffix = "pre_ffw_norm_2.weight"
	case "post_feedforward_layernorm_1.weight":
		ggufSuffix = "post_ffw_norm_1.weight"
	case "post_feedforward_layernorm_2.weight":
		ggufSuffix = "post_ffw_norm_2.weight"
	case "router.scale":
		ggufSuffix = "ffn_gate_inp.scale"
	case "router.per_expert_scale":
		ggufSuffix = "ffn_down_exps.scale"
	case "mlp.gate_proj.weight":
		ggufSuffix = "ffn_gate.weight"
	case "mlp.up_proj.weight":
		ggufSuffix = "ffn_up.weight"
	case "mlp.down_proj.weight":
		ggufSuffix = "ffn_down.weight"
	case "per_layer_input_gate.weight":
		ggufSuffix = "inp_gate.weight"
	case "per_layer_projection.weight":
		ggufSuffix = "proj.weight"
	case "post_per_layer_input_norm.weight":
		ggufSuffix = "post_norm.weight"
	}
	if ggufSuffix == "" {
		return nil
	}
	return []string{"blk." + parts[0] + "." + ggufSuffix}
}

func hipValidateGemma4Q4Tensor(tensor hipTensor, label, typeName string, rows, cols int, bytes uint64) error {
	if tensor.info.TypeName != typeName ||
		len(tensor.info.Dimensions) != 2 ||
		tensor.info.Dimensions[0] != uint64(rows) ||
		tensor.info.Dimensions[1] != uint64(cols) ||
		tensor.info.ByteSize != bytes {
		return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("%s tensor shape/type mismatch", label), nil)
	}
	return nil
}

func hipValidateGemma4Q4TensorBytes(label string, got, want uint64) error {
	if got != want {
		return core.E(hipGemma4Q4Layer0Operation, label+" byte count mismatch", nil)
	}
	return nil
}

func hipGemma4Q4LayerRoPEBase(headDim int) float32 {
	if headDim >= 512 {
		return 1000000
	}
	return 10000
}

func hipGemma4Q4LayerRoPERotaryDim(headDim int) int {
	if headDim >= 512 {
		return headDim / 4
	}
	return headDim
}

func (model *hipLoadedModel) loadedGemma4Q4LayerType(layer, headDim int) string {
	if model != nil && layer >= 0 && layer < len(model.gemma4TextConfig.LayerTypes) {
		layerType := model.gemma4TextConfig.LayerTypes[layer]
		if hipGemma4Q4LayerTypeSupported(layerType) && layerType != "" {
			return layerType
		}
	}
	return hipGemma4Q4LayerTypeFromHeadDim(headDim)
}

func (model *hipLoadedModel) loadedGemma4Q4LayerHeadDim(layerType string, queryRows, keyRows int) int {
	headDim := 0
	if model != nil {
		switch layerType {
		case "full_attention":
			headDim = model.gemma4TextConfig.GlobalHeadDim
		default:
			headDim = model.gemma4TextConfig.HeadDim
		}
		if headDim <= 0 && layerType == "full_attention" {
			headDim = model.gemma4TextConfig.HeadDim
		}
	}
	if headDim > 0 && queryRows%headDim == 0 && keyRows%headDim == 0 {
		return headDim
	}
	for _, candidate := range []int{512, 256, 128, 64, 32, 16, 8, 4, 2, 1} {
		if queryRows%candidate == 0 && keyRows%candidate == 0 {
			return candidate
		}
	}
	return 0
}

func (model *hipLoadedModel) loadedGemma4Q4LayerRoPE(layerType string, headDim int) (float32, int, float32) {
	params := nativeGemma4RoPEParameters{}
	if model != nil && model.gemma4TextConfig.RoPEParameters != nil {
		params = model.gemma4TextConfig.RoPEParameters[layerType]
	}
	base := params.RopeTheta
	if base <= 0 {
		switch layerType {
		case "full_attention":
			base = 1000000
		default:
			base = 10000
		}
	}
	factor := params.PartialRotaryFactor
	if factor <= 0 {
		switch layerType {
		case "full_attention":
			factor = 0.25
		default:
			factor = 1
		}
	}
	frequencyScale := float32(1)
	if params.RopeType == "proportional" && params.Factor > 0 && !math.IsNaN(params.Factor) && !math.IsInf(params.Factor, 0) {
		frequencyScale = float32(1 / params.Factor)
	}
	return float32(base), hipGemma4Q4RoPERotaryDimFromFactor(headDim, factor), frequencyScale
}

func hipGemma4Q4RoPERotaryDimFromFactor(headDim int, factor float64) int {
	if headDim <= 0 {
		return 0
	}
	if factor <= 0 {
		factor = 1
	}
	rotaryDim := int(math.Round(float64(headDim) * factor))
	if rotaryDim <= 0 {
		rotaryDim = headDim
	}
	if rotaryDim > headDim {
		rotaryDim = headDim
	}
	if rotaryDim%2 != 0 {
		rotaryDim--
	}
	if rotaryDim <= 0 {
		return headDim
	}
	return rotaryDim
}

func hipGemma4Q4RoPENormConfig(cfg hipRMSNormDeviceWeightConfig, epsilon float32, count int) hipRMSNormDeviceWeightConfig {
	cfg.Epsilon = epsilon
	cfg.Count = count
	cfg.Flags |= hipRMSNormLaunchFlagRoPENeoX
	return cfg
}

func hipGemma4Q4RoPEKernelDims(cfg hipGemma4Q4Layer0Config) (frequencyDim, rotaryCount int) {
	if cfg.RoPERotaryDim != cfg.HeadDim {
		return cfg.HeadDim, cfg.RoPERotaryDim
	}
	return 0, 0
}

func (cfg hipGemma4Q4Layer0Config) effectiveRoPEFrequencyScale() float32 {
	scale := cfg.RoPEFrequencyScale
	if scale == 0 {
		scale = 1
	}
	if scale <= 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return 0
	}
	return scale
}

func hipGemma4Q4LayerSlidingWindow(headDim int) int {
	if headDim >= 512 {
		return 0
	}
	return 512
}

func hipGemma4Q4EffectiveSlidingWindow(headDim, contextSize int) int {
	window := hipGemma4Q4LayerSlidingWindow(headDim)
	if contextSize <= 0 {
		return window
	}
	if window > 0 && contextSize < window {
		return contextSize
	}
	return window
}

func (model *hipLoadedModel) loadedGemma4Q4EffectiveSlidingWindow(layerType string, headDim int) int {
	if layerType != "sliding_attention" {
		return 0
	}
	window := 0
	if model != nil {
		window = model.gemma4TextConfig.SlidingWindow
	}
	if window <= 0 {
		window = 512
		if hipGemma4Q4LayerSlidingWindow(headDim) > 0 {
			window = hipGemma4Q4LayerSlidingWindow(headDim)
		}
	}
	if model != nil && model.contextSize > 0 && model.contextSize < window {
		return model.contextSize
	}
	return window
}

func (model *hipLoadedModel) loadedGemma4Q4AttentionKEqV(layerType string) bool {
	return layerType == "full_attention" && model != nil && model.gemma4TextConfig.AttentionKEqV
}

func (model *hipLoadedModel) loadedGemma4Q4LayerAttentionKEqV(layerPrefix, layerType string) bool {
	if model.loadedGemma4Q4AttentionKEqV(layerType) {
		return true
	}
	if model == nil || layerType != "full_attention" || !model.hipGGUFTensorAliasesEnabled() {
		return false
	}
	keyName := layerPrefix + ".self_attn.k_proj.weight"
	valueName := layerPrefix + ".self_attn.v_proj.weight"
	return model.hasHIPTensor(keyName) && !model.hasHIPTensor(valueName)
}

func hipGemma4Q4AttentionScale(_ int) float32 {
	return 1
}

func hipGemma4Q4LayerTypeFromHeadDim(headDim int) string {
	if headDim >= 512 {
		return "full_attention"
	}
	return "sliding_attention"
}

func hipGemma4Q4LayerTypeSupported(layerType string) bool {
	switch layerType {
	case "", "sliding_attention", "full_attention":
		return true
	default:
		return false
	}
}

func hipGemma4Q4DefaultKVSharedLayers(layerCount int) int {
	if layerCount > 20 {
		return 20
	}
	return 0
}

func (model *hipLoadedModel) loadedGemma4Q4KVSharedLayers(layerCount int) int {
	if model != nil && model.gemma4TextConfig.KVSharedLayersSet {
		if model.gemma4TextConfig.KVSharedLayers < 0 {
			return 0
		}
		if model.gemma4TextConfig.KVSharedLayers > layerCount {
			return layerCount
		}
		return model.gemma4TextConfig.KVSharedLayers
	}
	return hipGemma4Q4DefaultKVSharedLayers(layerCount)
}

func hipGemma4Q4FinalLogitSoftcap() float32 {
	return 30
}

func (model *hipLoadedModel) loadedGemma4Q4FinalLogitSoftcap() float32 {
	if model != nil && model.gemma4TextConfig.FinalLogitSoftcap > 0 &&
		!math.IsNaN(model.gemma4TextConfig.FinalLogitSoftcap) &&
		!math.IsInf(model.gemma4TextConfig.FinalLogitSoftcap, 0) {
		return float32(model.gemma4TextConfig.FinalLogitSoftcap)
	}
	return hipGemma4Q4FinalLogitSoftcap()
}
