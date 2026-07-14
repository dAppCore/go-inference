// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/gguf"
	rocmprofile "dappco.re/go/inference/engine/hip/profile"
	"dappco.re/go/inference/model"
	sharedgemma4 "dappco.re/go/inference/model/gemma4"
)

const Gemma4ArchitectureDeclarationContract = "rocm-gemma4-architecture-declaration-v1"

// Gemma4ArchitectureTopology is the normalized shared-KV topology projected
// from model.Arch. It gives HIP callers the cache-owner identity without
// re-parsing or executing a model.
type Gemma4ArchitectureTopology struct {
	LayerTypes  []string `json:"layer_types,omitempty"`
	KVShareFrom []int    `json:"kv_share_from,omitempty"`
	CacheIndex  []int    `json:"cache_index,omitempty"`
}

// Gemma4ArchitectureDeclaration pairs the shared model declaration with the
// canonical HIP profile selected from the same config signals.
type Gemma4ArchitectureDeclaration struct {
	Contract   string                     `json:"contract,omitempty"`
	Resolution ROCmArchitectureResolution `json:"resolution"`
	Arch       model.Arch                 `json:"arch"`
	Topology   Gemma4ArchitectureTopology `json:"topology"`
}

// Matched reports whether shared model identity and topology agree with the
// canonical HIP profile resolution.
func (declaration Gemma4ArchitectureDeclaration) Matched() bool {
	if !declaration.Resolution.Matched() || declaration.Resolution.Profile.ID != declaration.Resolution.Architecture {
		return false
	}
	if len(declaration.Arch.Layer) == 0 ||
		len(declaration.Topology.LayerTypes) != len(declaration.Arch.Layer) ||
		len(declaration.Topology.KVShareFrom) != len(declaration.Arch.Layer) ||
		len(declaration.Topology.CacheIndex) != len(declaration.Arch.Layer) {
		return false
	}
	for index, layer := range declaration.Arch.Layer {
		if declaration.Topology.LayerTypes[index] != layer.TypeName() ||
			declaration.Topology.KVShareFrom[index] != layer.KVShareFrom ||
			declaration.Topology.CacheIndex[index] != layer.CacheIndex {
			return false
		}
	}
	return true
}

// ResolveGemma4ArchitectureDeclaration resolves config metadata through the
// shared model registry, then pairs its Arch declaration with the HIP profile.
// It is deliberately a declaration-only adapter: it never loads weights or
// invokes an ArchSpec.Composed callback.
func ResolveGemma4ArchitectureDeclaration(data []byte) (Gemma4ArchitectureDeclaration, error) {
	var probe struct {
		Architectures []string `json:"architectures"`
		TextConfig    struct {
			Architectures []string `json:"architectures"`
		} `json:"text_config"`
	}
	if result := core.JSONUnmarshal(data, &probe); !result.OK {
		return Gemma4ArchitectureDeclaration{}, core.NewError("rocm Gemma4 architecture declaration: malformed config: " + result.Error())
	}

	modelType, textModelType := model.ProbeModelTypes(data)
	architectures := append([]string(nil), probe.Architectures...)
	architectures = append(architectures, probe.TextConfig.Architectures...)
	resolution := ResolveROCmArchitecture(modelType, textModelType, architectures)
	if !resolution.Matched() {
		return Gemma4ArchitectureDeclaration{}, core.NewError("rocm Gemma4 architecture declaration: unknown architecture")
	}
	if !rocmprofile.IsGemma4TargetArchitecture(resolution.Architecture) {
		return Gemma4ArchitectureDeclaration{}, core.NewError("rocm Gemma4 architecture declaration: unsupported architecture " + resolution.Architecture)
	}

	spec, ok := model.LookupArch(resolution.Architecture)
	if !ok || spec.Parse == nil {
		return Gemma4ArchitectureDeclaration{}, core.NewError("rocm Gemma4 architecture declaration: no shared ArchSpec for " + resolution.Architecture)
	}
	if spec.Composed != nil {
		return Gemma4ArchitectureDeclaration{}, core.NewError("rocm Gemma4 architecture declaration: composed execution is not supported")
	}
	cfg, err := spec.Parse(data)
	if err != nil {
		return Gemma4ArchitectureDeclaration{}, core.E("rocm Gemma4 architecture declaration", "parse shared config", err)
	}
	if cfg == nil {
		return Gemma4ArchitectureDeclaration{}, core.NewError("rocm Gemma4 architecture declaration: shared parser returned nil config")
	}
	arch, err := cfg.Arch()
	if err != nil {
		return Gemma4ArchitectureDeclaration{}, core.E("rocm Gemma4 architecture declaration", "derive shared Arch", err)
	}
	if len(arch.Layer) == 0 {
		return Gemma4ArchitectureDeclaration{}, core.NewError("rocm Gemma4 architecture declaration: shared Arch has no layers")
	}

	return gemma4ArchitectureDeclarationFromArch(resolution, arch)
}

func resolveGemma4ModelPackArchitectureDeclaration(path string) (Gemma4ArchitectureDeclaration, error) {
	root, err := rocmModelPackRoot(path)
	if err != nil {
		return Gemma4ArchitectureDeclaration{}, err
	}
	read := core.ReadFile(core.PathJoin(root, "config.json"))
	if !read.OK {
		return Gemma4ArchitectureDeclaration{}, read.Value.(error)
	}
	return ResolveGemma4ArchitectureDeclaration(read.Value.([]byte))
}

// ResolveGemma4GGUFArchitectureDeclaration lifts GGUF metadata through the
// shared Gemma4 config so HIP consumes the same Arch declaration for GGUF and
// safetensors checkpoints.
func ResolveGemma4GGUFArchitectureDeclaration(metadata gguf.Metadata, info inference.ModelInfo) (Gemma4ArchitectureDeclaration, error) {
	return resolveGemma4GGUFArchitectureDeclarationWithTensors(metadata, info, nil)
}

func resolveGemma4GGUFArchitectureDeclarationWithTensors(metadata gguf.Metadata, info inference.ModelInfo, tensors []nativeTensorInfo) (Gemma4ArchitectureDeclaration, error) {
	architecture := firstNonEmptyString(info.Architecture, normalizeROCmArchitecture(metadata.Architecture))
	resolution := ResolveROCmArchitecture(architecture, "", nil)
	if !resolution.Matched() || !rocmprofile.IsGemma4TargetArchitecture(resolution.Architecture) {
		return Gemma4ArchitectureDeclaration{}, core.NewError("rocm Gemma4 GGUF architecture declaration: unsupported architecture " + architecture)
	}
	native := nativeGemma4TextConfigFromGGUFMetadata(metadata)
	hidden := firstPositiveInt(info.HiddenSize, int(metadata.EmbeddingLength), int(metadata.EmbeddingLengthOut))
	vocab := firstPositiveInt(info.VocabSize, len(metadata.TokenizerTokens))
	if hidden <= 0 || vocab <= 0 || metadata.AttentionHeadCount == 0 {
		return Gemma4ArchitectureDeclaration{}, core.NewError("rocm Gemma4 GGUF architecture declaration: hidden, vocab, and attention heads are required")
	}
	measuredSlidingKVHeads, measuredGlobalKVHeads, err := gemma4GGUFKVHeadsFromTensors(tensors, native.LayerTypes, native.HeadDim, native.GlobalHeadDim)
	if err != nil {
		return Gemma4ArchitectureDeclaration{}, core.E("rocm Gemma4 GGUF architecture declaration", "infer KV heads from weights", err)
	}
	slidingKVHeads := firstPositiveInt(measuredSlidingKVHeads, int(metadata.AttentionHeadCountKV), measuredGlobalKVHeads)
	globalKVHeads := firstPositiveInt(measuredGlobalKVHeads, int(metadata.AttentionHeadCountKV), slidingKVHeads)
	if slidingKVHeads <= 0 || globalKVHeads <= 0 {
		return Gemma4ArchitectureDeclaration{}, core.NewError("rocm Gemma4 GGUF architecture declaration: KV heads are absent from metadata and weight geometry")
	}
	rope := make(map[string]sharedgemma4.RopeParam, len(native.RoPEParameters))
	for layerType, params := range native.RoPEParameters {
		rope[layerType] = sharedgemma4.RopeParam{
			RopeTheta:           float32(params.RopeTheta),
			PartialRotaryFactor: float32(params.PartialRotaryFactor),
			RopeType:            params.RopeType,
			Factor:              float32(params.Factor),
		}
	}
	config := sharedgemma4.Config{
		HiddenSize:              hidden,
		NumHiddenLayers:         int(metadata.BlockCount),
		IntermediateSize:        int(metadata.FeedForwardLength),
		NumAttentionHeads:       int(metadata.AttentionHeadCount),
		NumKeyValueHeads:        slidingKVHeads,
		NumGlobalKeyValueHeads:  globalKVHeads,
		HeadDim:                 native.HeadDim,
		GlobalHeadDim:           native.GlobalHeadDim,
		VocabSize:               vocab,
		SlidingWindow:           native.SlidingWindow,
		MaxPositionEmbeddings:   int(metadata.ContextLength),
		NumKVSharedLayers:       native.KVSharedLayers,
		LayerTypes:              append([]string(nil), native.LayerTypes...),
		AttentionKEqV:           native.AttentionKEqV,
		RopeParameters:          rope,
		HiddenSizePerLayerInput: native.HiddenSizePerLayerInput,
		VocabSizePerLayerInput:  native.VocabSizePerLayerInput,
		EnableMoEBlock:          native.EnableMoEBlock,
		NumExperts:              native.NumExperts,
		TopKExperts:             native.TopKExperts,
		MoEIntermediateSize:     native.MoEIntermediateSize,
		FinalLogitSoftcapping:   float32(native.FinalLogitSoftcap),
	}
	arch, err := config.Arch()
	if err != nil {
		return Gemma4ArchitectureDeclaration{}, core.E("rocm Gemma4 GGUF architecture declaration", "derive shared Arch", err)
	}
	return gemma4ArchitectureDeclarationFromArch(resolution, arch)
}

func gemma4GGUFKVHeadsFromTensors(tensors []nativeTensorInfo, layerTypes []string, slidingHeadDim, globalHeadDim int) (int, int, error) {
	const (
		prefix = "blk."
		suffix = ".attn_k.weight"
	)
	slidingKVHeads := 0
	globalKVHeads := 0
	for _, tensor := range tensors {
		if !core.HasPrefix(tensor.Name, prefix) || !core.HasSuffix(tensor.Name, suffix) {
			continue
		}
		layerText := core.TrimSuffix(core.TrimPrefix(tensor.Name, prefix), suffix)
		parsed := core.ParseInt(layerText, 10, 64)
		if !parsed.OK {
			continue
		}
		layer := int(parsed.Value.(int64))
		if layer < 0 || layer >= len(layerTypes) || len(tensor.Dimensions) != 2 {
			continue
		}
		layerType := layerTypes[layer]
		headDim := slidingHeadDim
		if layerType == "full_attention" {
			headDim = globalHeadDim
		}
		rows := tensor.Dimensions[1]
		if headDim <= 0 || rows == 0 || rows%uint64(headDim) != 0 {
			return 0, 0, core.NewError(core.Sprintf("layer %d %s K projection width %d is not divisible by head dim %d", layer, layerType, rows, headDim))
		}
		kvHeads := int(rows / uint64(headDim))
		target := &slidingKVHeads
		if layerType == "full_attention" {
			target = &globalKVHeads
		}
		if *target > 0 && *target != kvHeads {
			return 0, 0, core.NewError(core.Sprintf("layer %d %s K projection declares %d KV heads after %d", layer, layerType, kvHeads, *target))
		}
		*target = kvHeads
	}
	return slidingKVHeads, globalKVHeads, nil
}

func gemma4ArchitectureDeclarationFromArch(resolution ROCmArchitectureResolution, arch model.Arch) (Gemma4ArchitectureDeclaration, error) {
	if len(arch.Layer) == 0 {
		return Gemma4ArchitectureDeclaration{}, core.NewError("rocm Gemma4 architecture declaration: shared Arch has no layers")
	}
	topology := Gemma4ArchitectureTopology{
		LayerTypes:  make([]string, len(arch.Layer)),
		KVShareFrom: make([]int, len(arch.Layer)),
		CacheIndex:  make([]int, len(arch.Layer)),
	}
	for index, layer := range arch.Layer {
		topology.LayerTypes[index] = layer.TypeName()
		topology.KVShareFrom[index] = layer.KVShareFrom
		topology.CacheIndex[index] = layer.CacheIndex
	}
	declaration := Gemma4ArchitectureDeclaration{
		Contract:   Gemma4ArchitectureDeclarationContract,
		Resolution: resolution,
		Arch:       arch,
		Topology:   topology,
	}
	if !declaration.Matched() {
		return Gemma4ArchitectureDeclaration{}, core.NewError("rocm Gemma4 architecture declaration: shared topology does not match HIP resolution")
	}
	return declaration, nil
}

func cloneGemma4ArchitectureDeclaration(declaration Gemma4ArchitectureDeclaration) Gemma4ArchitectureDeclaration {
	declaration.Resolution = declaration.Resolution.clone()
	declaration.Arch.Layer = append([]model.LayerSpec(nil), declaration.Arch.Layer...)
	declaration.Arch.RopeFreqs = append([]float32(nil), declaration.Arch.RopeFreqs...)
	declaration.Arch.RopeShortFreqs = append([]float32(nil), declaration.Arch.RopeShortFreqs...)
	if declaration.Arch.TieWordEmbeddings != nil {
		tied := *declaration.Arch.TieWordEmbeddings
		declaration.Arch.TieWordEmbeddings = &tied
	}
	declaration.Topology.LayerTypes = append([]string(nil), declaration.Topology.LayerTypes...)
	declaration.Topology.KVShareFrom = append([]int(nil), declaration.Topology.KVShareFrom...)
	declaration.Topology.CacheIndex = append([]int(nil), declaration.Topology.CacheIndex...)
	return declaration
}

func (loaded *hipLoadedModel) sharedGemma4LayerSpecs(layerCount int) ([]model.LayerSpec, bool) {
	if loaded == nil || layerCount <= 0 || !loaded.gemma4Architecture.Matched() || len(loaded.gemma4Architecture.Arch.Layer) != layerCount {
		return nil, false
	}
	return loaded.gemma4Architecture.Arch.Layer, true
}
