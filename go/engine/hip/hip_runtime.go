// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"io"
	"iter"
	"sort"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const nativeTensorCopyChunkBytes = 16 << 20

type nativeDevicePointer uintptr

type nativeHIPDriver interface {
	Available() bool
	DeviceInfo() nativeDeviceInfo
	Malloc(size uint64) (nativeDevicePointer, error)
	Free(pointer nativeDevicePointer) error
	CopyHostToDevice(pointer nativeDevicePointer, data []byte) error
	CopyDeviceToHost(pointer nativeDevicePointer, data []byte) error
}

type nativeHIPAsyncHostToDevice interface {
	CopyHostToDeviceAsync(pointer nativeDevicePointer, data []byte) error
}

type nativeHIPLabeledHostToDevice interface {
	CopyHostToDeviceLabeled(pointer nativeDevicePointer, data []byte, operation, label string) error
}

type nativeHIPDeviceMemset interface {
	MemsetAsync(pointer nativeDevicePointer, value byte, size uint64) error
}

type nativeHIPKernelFunctionPrewarmer interface {
	PrewarmKernelFunctions(kernelNames []string)
}

type nativeHIPDriverUnwrapper interface {
	rocmUnwrapNativeHIPDriver() nativeHIPDriver
}

func hipCopyHostToDevice(driver nativeHIPDriver, pointer nativeDevicePointer, data []byte) error {
	if async, ok := driver.(nativeHIPAsyncHostToDevice); ok {
		return async.CopyHostToDeviceAsync(pointer, data)
	}
	return driver.CopyHostToDevice(pointer, data)
}

func hipCopyHostToDeviceLabeled(driver nativeHIPDriver, pointer nativeDevicePointer, data []byte, operation, label string) error {
	if labeled, ok := driver.(nativeHIPLabeledHostToDevice); ok {
		return labeled.CopyHostToDeviceLabeled(pointer, data, operation, label)
	}
	return hipCopyHostToDevice(driver, pointer, data)
}

func hipMemsetDevice(driver nativeHIPDriver, pointer nativeDevicePointer, value byte, size uint64) error {
	if size == 0 {
		return nil
	}
	if pointer == 0 {
		return core.E("rocm.hip.MemsetDevice", "device pointer is nil", nil)
	}
	if memset, ok := driver.(nativeHIPDeviceMemset); ok {
		return memset.MemsetAsync(pointer, value, size)
	}
	if size > uint64(int(^uint(0)>>1)) {
		return core.E("rocm.hip.MemsetDevice", "device memset size is out of range", nil)
	}
	payload := make([]byte, int(size))
	if value != 0 {
		for index := range payload {
			payload[index] = value
		}
	}
	return hipCopyHostToDevice(driver, pointer, payload)
}

type hipRuntime struct {
	driver nativeHIPDriver
}

func newSystemNativeRuntime() nativeRuntime {
	return newHIPRuntime(newSystemHIPDriver())
}

func newHIPRuntime(driver nativeHIPDriver) *hipRuntime {
	return &hipRuntime{driver: driver}
}

func (runtime *hipRuntime) Available() bool {
	return runtime != nil && runtime.driver != nil && runtime.driver.Available()
}

func (runtime *hipRuntime) DeviceInfo() nativeDeviceInfo {
	if runtime == nil || runtime.driver == nil {
		return nativeDeviceInfo{}
	}
	return runtime.driver.DeviceInfo()
}

func (runtime *hipRuntime) KernelStatus() hipKernelStatus {
	if runtime == nil || runtime.driver == nil || !runtime.driver.Available() {
		return defaultHIPKernelStatus()
	}
	return normalizeHIPKernelStatus(newHIPRuntimeKernelSet(runtime.driver).Status())
}

func (runtime *hipRuntime) LoadModel(path string, cfg nativeLoadConfig) (nativeModel, error) {
	if runtime == nil || runtime.driver == nil {
		return nil, core.E("rocm.hip.LoadModel", "HIP driver is nil", nil)
	}
	if !runtime.driver.Available() {
		return nil, core.E("rocm.hip.LoadModel", "HIP driver is not available", nil)
	}
	architecture := rocmNativeModelLoaderArchitecture(cfg)
	if route, ok := ROCmModelLoaderRouteForArchitecture(architecture); ok {
		if route.AttachedOnly {
			if cfg.AllowAttachedOnly && route.NativeRuntime && route.Runtime == rocmModelLoaderRuntimeHIP && !route.MetadataOnly {
				return loadHIPDefaultNativeModel(runtime, path, cfg)
			}
			return nil, core.E("rocm.hip.LoadModel", architecture+" is an attached drafter, not a standalone model; load it beside its target via LoadAttachedDrafterPairAsTextModel", nil)
		}
		if !rocmNativeModelLoaderRouteHasStandaloneLoader(route) {
			return nil, core.E("rocm.hip.LoadModel", architecture+" has no standalone HIP model loader; route status is "+string(route.Status), nil)
		}
		if loader, ok := lookupROCmNativeModelLoader(architecture); ok {
			return loader.load(runtime, path, cfg)
		}
		return nil, core.E("rocm.hip.LoadModel", "no native model loader registered for "+architecture, nil)
	}
	if loader, ok := lookupROCmNativeModelLoader(architecture); ok {
		return loader.load(runtime, path, cfg)
	}
	return loadHIPDefaultNativeModel(runtime, path, cfg)
}

func loadHIPDefaultNativeModel(runtime *hipRuntime, path string, cfg nativeLoadConfig) (nativeModel, error) {
	if err := validateHIPLoadConfig(cfg); err != nil {
		return nil, core.E("rocm.hip.LoadModel", "validate tensor plan", err)
	}
	if err := validateHIPTensorFileRanges(path, cfg); err != nil {
		return nil, core.E("rocm.hip.LoadModel", "validate tensor file ranges", err)
	}
	engineConfig := defaultHIPGemma4Q4EngineConfig()
	if cfg.DeviceKVMode != "" {
		engineConfig.DeviceKVMode = cfg.DeviceKVMode
	}
	if _, err := engineConfig.deviceKVMode(); err != nil {
		return nil, core.E("rocm.hip.LoadModel", "validate Gemma4 engine config", err)
	}
	modelLabels := cloneStringMap(cfg.ModelLabels)
	if isROCmGemma4Architecture(cfg.ModelInfo.Architecture) {
		modelLabels = rocmApplyGemma4NativeConfigFeatureLabels(modelLabels, cfg.Gemma4TextConfig)
	}
	tokenText := cfg.TokenText
	if tokenText == nil {
		tokenText = loadHIPTokenTextDecoderIfPresent(cfg.TokenizerPath)
	}
	model := &hipLoadedModel{
		driver:             runtime.driver,
		kernels:            newHIPRuntimeKernelSet(runtime.driver),
		modelPath:          path,
		modelInfo:          cfg.ModelInfo,
		modelLabels:        modelLabels,
		engineProfile:      cfg.EngineProfile.clone(),
		gemma4Q4Config:     engineConfig,
		sequenceMixerPlan:  cloneSequenceMixerLoadPlan(cfg.SequenceMixerPlan),
		contextSize:        cfg.ContextSize,
		gemma4TextConfig:   cloneNativeGemma4TextConfig(cfg.Gemma4TextConfig),
		gemma4Architecture: cloneGemma4ArchitectureDeclaration(cfg.Gemma4Architecture),
		tensors:            make(map[string]hipTensor, len(cfg.Tensors)),
		hostTensors:        make(map[string]nativeTensorInfo),
		tokenText:          tokenText,
		createdAt:          time.Now(),
	}
	var tensorCopyBuffer []byte
	tensorFiles := map[string]*core.OSFile{}
	defer closeTensorSourceFiles(tensorFiles)
	for _, tensor := range cfg.Tensors {
		if tensor.ByteSize == 0 {
			continue
		}
		if hipGemma4HostResidentExpertTensor(cfg, tensor) {
			hostTensor := tensor
			hostTensor.SourcePath = hipTensorSourcePath(path, tensor)
			hostTensor.DataOffset = hipTensorDataOffset(cfg, tensor)
			model.hostTensors[tensor.Name] = hostTensor
			continue
		}
		pointer, err := runtime.driver.Malloc(tensor.ByteSize)
		if err != nil {
			model.Close()
			return nil, core.E("rocm.hip.LoadModel", "allocate tensor "+tensor.Name, err)
		}
		loaded := hipTensor{info: tensor, pointer: pointer}
		model.tensors[tensor.Name] = loaded
		tensorCopyBuffer, err = copyTensorToDevice(runtime.driver, path, cfg.DataOffset, loaded, tensorCopyBuffer, tensorFiles)
		if err != nil {
			model.Close()
			return nil, core.E("rocm.hip.LoadModel", "copy tensor "+tensor.Name, err)
		}
	}
	if err := model.synthesizeGemma4GGUFAffineTensors(); err != nil {
		model.Close()
		return nil, core.E("rocm.hip.LoadModel", "synthesize Gemma4 GGUF affine tensors", err)
	}
	if model.sequenceMixerPlan != nil {
		if err := model.bindSequenceMixerPlan(); err != nil {
			model.Close()
			return nil, core.E("rocm.hip.LoadModel", "bind sequence mixer plan", err)
		}
	}
	if cfg.VisionModelPath != "" {
		gemm := newHIPAudioGEMM(runtime.driver)
		tower, err := loadUnifiedVisionTowerWithGEMM(cfg.VisionModelPath, gemm)
		if err != nil {
			model.Close()
			return nil, core.E("rocm.hip.LoadModel", "load Gemma 4 unified vision tower", err)
		}
		visionRuntime := "gemma4_unified"
		if tower != nil {
			if cfg.ModelInfo.HiddenSize <= 0 || tower.loaded.Cfg.TextHidden != cfg.ModelInfo.HiddenSize {
				_ = tower.Close()
				model.Close()
				return nil, core.E("rocm.hip.LoadModel", "vision projector output does not match text hidden size", nil)
			}
			model.unifiedVision = tower
			if tower.loaded.Cfg.BidirectionalImages {
				model.gemma4Q4Config.BidirectionalSpanTokens = [2]int32{
					tower.loaded.Cfg.ImageTokenID,
					tower.loaded.Cfg.VideoTokenID,
				}
			}
		} else {
			encoder, err := loadHIPVisionEncoderTowerWithGEMM(cfg.VisionModelPath, gemm)
			if err != nil {
				model.Close()
				return nil, core.E("rocm.hip.LoadModel", "load Gemma 4 encoder vision tower", err)
			}
			if encoder == nil {
				model.Close()
				return nil, core.E("rocm.hip.LoadModel", "vision model path has no supported Gemma 4 vision tower", nil)
			}
			if cfg.ModelInfo.HiddenSize <= 0 || encoder.outputDim() != cfg.ModelInfo.HiddenSize {
				_ = encoder.Close()
				model.Close()
				return nil, core.E("rocm.hip.LoadModel", "vision projector output does not match text hidden size", nil)
			}
			model.vision = encoder
			visionRuntime = "gemma4_siglip"
		}
		if model.modelLabels == nil {
			model.modelLabels = map[string]string{}
		}
		model.modelLabels["vision_model_path"] = cfg.VisionModelPath
		model.modelLabels["vision_runtime"] = visionRuntime
	}
	if cfg.AudioModelPath != "" {
		tower, err := loadAudioTowerWithGEMM(cfg.AudioModelPath, newHIPAudioGEMM(runtime.driver))
		if err != nil {
			model.Close()
			return nil, core.E("rocm.hip.LoadModel", "load Gemma 4 audio tower", err)
		}
		if tower == nil {
			model.Close()
			return nil, core.E("rocm.hip.LoadModel", "audio model path has no Gemma 4 audio tower", nil)
		}
		if cfg.ModelInfo.HiddenSize <= 0 || tower.loaded.Projector.OutDim != cfg.ModelInfo.HiddenSize {
			_ = tower.Close()
			model.Close()
			return nil, core.E("rocm.hip.LoadModel", "audio projector output does not match text hidden size", nil)
		}
		model.audio = tower
		if model.modelLabels == nil {
			model.modelLabels = map[string]string{}
		}
		model.modelLabels["audio_model_path"] = cfg.AudioModelPath
		model.modelLabels["audio_runtime"] = "gemma4_conformer"
	}
	hipPrewarmGemma4Q4TokenFilters(model)
	hipPrewarmGemma4Q4KernelFunctions(model.driver)
	hipPrewarmGemma4Q4LaunchPacketPools()
	hipPrewarmGemma4Q4DeviceByteBuffers(model)
	hipPrewarmGemma4Q4DeviceDecodeStates(model)
	hipPrewarmGemma4Q4PrefillForwardLayerBatches(model)
	rocmPrewarmDeviceKVHostPools()
	hipPrewarmGemma4Q4DeviceKVDescriptorPointers(model)
	hipPrewarmGemma4Q4DeviceKVTensorPointers(model)
	hipPrewarmAttentionHeadsChunkedWorkspacePool()
	hipPrewarmGemma4Q4AttentionWorkspaceDeviceBuffersForModel(model)
	hipPrewarmGemma4Q4DefaultSuppressTokenBufferForModel(model)
	return model, nil
}

func hipGemma4HostResidentExpertTensor(cfg nativeLoadConfig, tensor nativeTensorInfo) bool {
	if len(tensor.Dimensions) != 3 {
		return false
	}
	if isROCmGemma4Architecture(cfg.ModelInfo.Architecture) && core.Contains(tensor.Name, "_exps.weight") {
		return true
	}
	if !isROCmGemma4BackboneArchitecture(cfg.ModelInfo.Architecture) {
		return false
	}
	return core.HasSuffix(tensor.Name, ".experts.gate_up_proj.weight") ||
		core.HasSuffix(tensor.Name, ".experts.gate_up_proj.scales") ||
		core.HasSuffix(tensor.Name, ".experts.gate_up_proj.biases") ||
		core.HasSuffix(tensor.Name, ".experts.down_proj.weight") ||
		core.HasSuffix(tensor.Name, ".experts.down_proj.scales") ||
		core.HasSuffix(tensor.Name, ".experts.down_proj.biases")
}

var hipGemma4Q4WarmKernelNames = []string{
	hipKernelNameKVEncodeToken,
	hipKernelNameKVEncodeTokenValueNormDescriptorAppend,
	hipKernelNameKVDescriptorAppend,
	hipKernelNameProjection,
	hipKernelNameProjectionBatch,
	hipKernelNameMLXQ4Proj,
	hipKernelNameMLXQ4ProjQ4G32Rows3840Cols15360,
	hipKernelNameMLXQ4ProjQ4G64Rows3840Cols15360Row16,
	hipKernelNameMLXQ4ProjQ4G64E4BRow8,
	hipKernelNameMLXQ4ProjQ8G64Row8,
	hipKernelNameMLXQ4ProjCols256,
	hipKernelNameMLXQ4ProjQ6G16Row16,
	hipKernelNameMLXQ4ProjQ6Row16,
	hipKernelNameMLXQ4ProjQ6Row64,
	hipKernelNameMLXQ4ProjBatch,
	hipKernelNameMLXQ4ProjBatchQ4G64Tokens16,
	hipKernelNameMLXQ4ProjBatchQ4G64Rows2816Cols704,
	hipKernelNameMLXQ4ProjBatchQ8G64Row16Tokens16,
	hipKernelNameMLXQ4ProjBatchQ8G64Row16Tokens16Shared,
	hipKernelNameMLXQ4ProjBatchQ8G64Row16Tokens64Shared,
	hipKernelNameMLXQ4ProjBatchQ8G64Row32Tokens64Shared,
	hipKernelNameMLXQ4ProjBatchQ8G64Row64Tokens64Shared,
	hipKernelNameMLXQ4ProjBatchQ8G64Row64Tokens64Aligned,
	hipKernelNameMLXQ4ProjBatchQ6Row16,
	hipKernelNameMLXQ4ProjGreedy,
	hipKernelNameMLXQ4ProjGreedyQ6Row64,
	hipKernelNameMLXQ4ProjGreedyBatch,
	hipKernelNameMLXQ4ProjGreedyBatchQ6Row64,
	hipKernelNameMLXQ4ProjScores,
	hipKernelNameMLXQ4ProjScoresQ6Row64,
	hipKernelNameMLXQ4ProjSelectedGreedy,
	hipKernelNameMLXQ4ProjSelectedGreedyQ6Row64,
	hipKernelNameOrderedEmbeddingCandidates,
	hipKernelNamePackedTopK,
	hipKernelNamePackedTopKSample,
	hipKernelNameMLXQ4TripleProj,
	hipKernelNameMLXQ4TripleProjQ6Row64,
	hipKernelNameMLXQ4GELUTanhMul,
	hipKernelNameMLXQ4GELUTanhMulQ4G32Cols1536Row16,
	hipKernelNameMLXQ4GELUTanhMulQ4G32Rows15360Cols3840,
	hipKernelNameMLXQ4GELUTanhMulQ4G32Rows15360Cols3840Row8,
	hipKernelNameMLXQ4GELUTanhMulQ4G64Rows15360Cols3840Row8,
	hipKernelNameMLXQ4GELUTanhMulQ8G64Row8,
	hipKernelNameMLXQ4GELUTanhMLPQ4G32Cols1536Persistent,
	hipKernelNameMLXQ4GELUTanhMulQ6Cols1536,
	hipKernelNameMLXQ4GELUTanhMulQ6Cols1536Row64,
	hipKernelNameMLXQ4GELUTanhMulBatch,
	hipKernelNameMLXQ4GELUTanhMulBatchQ4G64Cols2816Row8,
	hipKernelNameMLXQ4GELUTanhMulBatchQ8G64Row16,
	hipKernelNameMLXQ4GELUTanhMulBatchQ8G64Rows2112T32,
	hipKernelNameMLXQ4GELUTanhProj,
	hipKernelNameMLXQ4GELUTanhProjQ6Row16,
	hipKernelNameMLXQ4GELUTanhProjBatch,
	hipKernelNameRMSNormResidualAddGELUTanhProj,
	hipKernelNameRMSNorm,
	hipKernelNameRMSNormResidualAdd,
	hipKernelNameRMSNormResAddNorm,
	hipKernelNameRMSNormHeads,
	hipKernelNameRMSNormRoPEHeads,
	hipKernelNameRMSNormRoPEHeadsPair,
	hipKernelNameRMSNormRoPEHeadsBatch,
	hipKernelNameRMSNormRoPEHeadsPairLaneBatch,
	hipKernelNameMoECombineNorms,
	hipKernelNameMoEBatchGatherRows,
	hipKernelNameMoEBatchScatterRoutes,
	hipKernelNameMoEBatchReduceRoutes,
	hipKernelNameMoEMLXAffineRoutes,
	hipKernelNameGGUFQ4KExpandMetadata,
	hipKernelNameAttentionHeads,
	hipKernelNameAttentionHeadsBatchCausal,
	hipKernelNameAttentionHeadsLaneBatch,
	hipKernelNameAttentionHeadsBatchCausalQueryRMSRoPE,
	hipKernelNameAttentionHeadsChunkedStage1,
	hipKernelNameAttentionHeadsChunkedStage2,
	hipKernelNameAttentionHeadsBatchChunkedStage1,
	hipKernelNameAttentionHeadsBatchChunkedStage1GQA2,
	hipKernelNameAttentionHeadsBatchChunkedStage1GQA4,
	hipKernelNameAttentionHeadsBatchChunkedStage1GQA8,
	hipKernelNameAttentionHeadsBatchChunkedStage2,
	hipKernelNameVectorAddScaled,
	hipKernelNameVectorScale,
	hipKernelNamePerLayerInputTranspose,
	hipKernelNameEmbedLookup,
	hipKernelNameEmbedLookupGreedyToken,
	hipKernelNameDiffusionExpectedEmbeddingAffineG64Rows16,
	hipKernelNameDiffusionExpectedEmbeddingQ8G64Dims4Rows4,
	hipKernelNameDiffusionExpectedEmbeddingQ8G64SubgroupRows64,
	hipKernelNameDiffusionExpectedEmbeddingQ8G64SubgroupRows64Probability4,
	hipKernelNameDiffusionExpectedEmbeddingQ8G64Tile32x64,
	hipKernelNameDiffusionSampleProbabilities,
}

func hipPrewarmGemma4Q4KernelFunctions(driver nativeHIPDriver) {
	for driver != nil {
		if prewarmer, ok := driver.(nativeHIPKernelFunctionPrewarmer); ok {
			prewarmer.PrewarmKernelFunctions(hipGemma4Q4WarmKernelNames)
			return
		}
		unwrapper, ok := driver.(nativeHIPDriverUnwrapper)
		if !ok {
			return
		}
		unwrapped := unwrapper.rocmUnwrapNativeHIPDriver()
		if unwrapped == driver {
			return
		}
		driver = unwrapped
	}
}

var hipGemma4Q4WarmLaunchPacketSizes = []int{
	hipKVEncodeTokenLaunchArgsBytes,
	hipKVEncodeTokenValueNormDescriptorAppendLaunchArgsBytes,
	hipKVDescriptorAppendLaunchArgsBytes,
	hipMLXQ4ProjectionLaunchArgsBytes,
	hipMLXQ4ProjectionBatchLaunchArgsBytes,
	hipMLXQ4TripleProjLaunchArgsBytes,
	hipMLXQ4GELUTanhMulLaunchArgsBytes,
	hipMLXQ4GELUTanhMLPPersistentLaunchArgsBytes,
	hipMLXQ4GELUTanhMulBatchLaunchArgsBytes,
	hipMLXQ4GELUTanhProjLaunchArgsBytes,
	hipMLXQ4GELUTanhProjBatchLaunchArgsBytes,
	hipRMSResidualAddGELUTanhProjLaunchArgsBytes,
	hipRMSNormLaunchArgsBytes,
	hipRMSNormResidualAddArgsBytes,
	hipRMSNormResAddNormArgsBytes,
	hipRMSNormHeadsLaunchArgsBytes,
	hipRMSNormRoPEHeadsLaunchArgsBytes,
	hipRMSNormRoPEHeadsPairLaunchArgsBytes,
	hipRMSNormRoPEHeadsBatchLaunchArgsBytes,
	hipRMSNormRoPEHeadsPairLaneBatchLaunchArgsBytes,
	hipMoECombineNormsLaunchArgsBytes,
	hipMoEBatchRouteRowsLaunchArgsBytes,
	hipMoEBatchReduceLaunchArgsBytes,
	hipGGUFQ4KExpandLaunchArgsBytes,
	hipAttentionHeadsLaunchArgsBytes,
	hipAttentionHeadsBatchCausalLaunchArgsBytes,
	hipAttentionHeadsLaneBatchLaunchArgsBytes,
	hipAttentionHeadsBatchCausalQueryRMSRoPELaunchArgsBytes,
	hipAttentionHeadsChunkedLaunchArgsBytes,
	hipAttentionHeadsBatchChunkedLaunchArgsBytes,
	hipVectorAddScaledLaunchArgsBytes,
	hipVectorScaleLaunchArgsBytes,
	hipPerLayerInputTransposeLaunchArgsBytes,
	hipEmbeddingLookupLaunchArgsBytes,
	hipSoftcapGreedyLaunchArgsBytes,
	hipDiffusionSampleLaunchArgsBytes,
}

func hipPrewarmGemma4Q4LaunchPacketPools() {
	hipPrewarmLaunchPacketPools(hipGemma4Q4WarmLaunchPacketSizes, 4)
}

func hipPrewarmGemma4Q4DeviceByteBuffers(model *hipLoadedModel) {
	if model == nil ||
		!hipLoadedGemma4Q4GenerateLinked(model) {
		return
	}
	hipPrewarmDeviceByteBufferPool(model.driver, hipMLXQ4ProjectionBestBytes, 4)
}

func hipPrewarmGemma4Q4DeviceDecodeStates(model *hipLoadedModel) {
	if model == nil ||
		!hipLoadedGemma4Q4GenerateLinked(model) ||
		model.modelInfo.NumLayers <= 0 {
		return
	}
	hipPrewarmGemma4Q4DeviceDecodeStatePool(model.modelInfo.NumLayers, 4)
	hipPrewarmGemma4Q4DeviceLayerStatePool(model.modelInfo.NumLayers, 1)
}

func hipPrewarmGemma4Q4PrefillForwardLayerBatches(model *hipLoadedModel) {
	if model == nil ||
		!hipLoadedGemma4Q4GenerateLinked(model) ||
		model.modelInfo.NumLayers <= 0 {
		return
	}
	hipPrewarmGemma4Q4PrefillForwardLayerBatchPool(model.modelInfo.NumLayers, 4)
}

func hipPrewarmGemma4Q4DeviceKVDescriptorPointers(model *hipLoadedModel) {
	if model == nil || model.driver == nil ||
		!rocmDeviceKVTensorPoolDefaultDriverEnabled(model.driver) ||
		!hipLoadedGemma4Q4GenerateLinked(model) ||
		model.modelInfo.NumLayers <= 0 {
		return
	}
	layerCount := model.modelInfo.NumLayers
	rocmPrewarmDeviceKVDescriptorPointerPool(model.driver, layerCount*2, layerCount)
}

func hipPrewarmGemma4Q4DeviceKVTensorPointers(model *hipLoadedModel) {
	if model == nil || model.driver == nil ||
		!rocmDeviceKVTensorPoolDefaultDriverEnabled(model.driver) ||
		!hipLoadedGemma4Q4GenerateLinked(model) ||
		model.modelInfo.NumLayers <= 0 {
		return
	}
	cfg, err := model.cachedGemma4Q4ForwardConfig(model.modelInfo.NumLayers)
	if err != nil {
		return
	}
	engineConfig := model.gemma4Q4EngineConfig()
	mode, err := engineConfig.deviceKVMode()
	if err != nil {
		return
	}
	counts := hipGemma4Q4DeviceKVTensorPrewarmCountsForContextWithEngineConfig(cfg, mode, model.contextSize, engineConfig)
	sizes := make([]uint64, 0, len(counts))
	for sizeBytes, count := range counts {
		if count > 0 {
			sizes = append(sizes, sizeBytes)
		}
	}
	sort.Slice(sizes, func(i, j int) bool { return sizes[i] < sizes[j] })
	for _, sizeBytes := range sizes {
		count := counts[sizeBytes]
		rocmPrewarmDeviceKVTensorPool(model.driver, sizeBytes, count)
	}
}

func hipPrewarmGemma4Q4AttentionWorkspaceDeviceBuffersForModel(model *hipLoadedModel) {
	if model == nil || model.driver == nil ||
		!hipLoadedGemma4Q4GenerateLinked(model) ||
		model.modelInfo.NumLayers <= 0 {
		return
	}
	cfg, err := model.cachedGemma4Q4ForwardConfig(model.modelInfo.NumLayers)
	if err != nil {
		return
	}
	_ = hipPrewarmGemma4Q4AttentionWorkspaceDeviceBuffers(model.driver, cfg, model.contextSize)
}

func hipPrewarmGemma4Q4AttentionWorkspaceModelHiddenBuffers(driver nativeHIPDriver, hiddenSize int) error {
	if driver == nil || !driver.Available() || hiddenSize <= 0 {
		return nil
	}
	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	if _, err := workspace.EnsureScaledEmbedding(driver, hiddenSize); err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		return err
	}
	if _, err := workspace.EnsurePrefillInputNormOutput(driver, hiddenSize); err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		return err
	}
	if _, err := workspace.EnsureIntermediateOutput(driver, hiddenSize); err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		return err
	}
	if _, err := workspace.EnsureFinalHiddenOutput(driver, hiddenSize, 0); err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		return err
	}
	if _, err := workspace.EnsureNextInputOutput(driver, hiddenSize, 0); err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		return err
	}
	workspace.resetBorrowedViews()
	if hipReleaseAttentionHeadsChunkedWorkspace(workspace) {
		return nil
	}
	return hipRecycleAttentionHeadsChunkedWorkspace(workspace)
}

func hipPrewarmGemma4Q4DefaultSuppressTokenBufferForModel(model *hipLoadedModel) {
	if model == nil || model.driver == nil ||
		!hipLoadedGemma4Q4GenerateLinked(model) ||
		model.tokenText == nil {
		return
	}
	tokens := hipGemma4Q4GenerationSuppressTokenIDs(model, nil)
	if len(tokens) == 0 {
		return
	}
	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	if _, err := workspace.EnsureSuppressTokenBuffer(model.driver, tokens); err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		return
	}
	if !hipReleaseAttentionHeadsChunkedWorkspace(workspace) {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
	}
}

func hipGemma4Q4DeviceKVTensorPrewarmCounts(cfg hipGemma4Q4ForwardConfig, mode string) map[uint64]int {
	return hipGemma4Q4DeviceKVTensorPrewarmCountsForContext(cfg, mode, 0)
}

func hipGemma4Q4DeviceKVTensorPrewarmCountsForContext(cfg hipGemma4Q4ForwardConfig, mode string, contextSize int) map[uint64]int {
	return hipGemma4Q4DeviceKVTensorPrewarmCountsForContextWithEngineConfig(cfg, mode, contextSize, defaultHIPGemma4Q4EngineConfig())
}

func hipGemma4Q4DeviceKVTensorPrewarmCountsForContextWithEngineConfig(cfg hipGemma4Q4ForwardConfig, mode string, contextSize int, engineConfig hipGemma4Q4EngineConfig) map[uint64]int {
	keyEncoding, valueEncoding, ok := rocmKVInterleavedEncodingsForMode(mode)
	if !ok || len(cfg.Layers) == 0 {
		return nil
	}
	counts := make(map[uint64]int, 2)
	var globalSizeBytes uint64
	for _, layer := range cfg.Layers {
		blockSize := engineConfig.deviceKVBlockSizeForSlidingWindow(layer.SlidingWindow)
		if blockSize <= 0 {
			continue
		}
		keyStride, err := rocmKVInterleavedRowStride(keyEncoding, layer.HeadDim)
		if err != nil {
			continue
		}
		valueStride, err := rocmKVInterleavedRowStride(valueEncoding, layer.HeadDim)
		if err != nil {
			continue
		}
		sizeBytes := (keyStride + valueStride) * uint64(blockSize)
		if sizeBytes <= rocmDeviceKVTensorPoolDefaultBytes {
			counts[sizeBytes]++
			if layer.SlidingWindow <= 0 {
				globalSizeBytes = sizeBytes
			}
		}
	}
	if globalSizeBytes > 0 && cfg.KVSharedLayers > 0 {
		counts[globalSizeBytes] += cfg.KVSharedLayers
	}
	hipAddGemma4Q4DeviceKVAppendTokenPrewarmCounts(counts, cfg, mode)
	if contextSize <= 0 {
		return counts
	}

	sources := hipGemma4Q4SharedKVSourceByLayer(cfg)
	contextCounts := make(map[uint64]int, len(counts))
	ownerSlack := make(map[uint64]int, len(counts))
	for index, layer := range cfg.Layers {
		if index < len(sources) && sources[index] != index {
			continue
		}
		blockSize := engineConfig.deviceKVBlockSizeForSlidingWindow(layer.SlidingWindow)
		if blockSize <= 0 {
			continue
		}
		keyStride, err := rocmKVInterleavedRowStride(keyEncoding, layer.HeadDim)
		if err != nil {
			continue
		}
		valueStride, err := rocmKVInterleavedRowStride(valueEncoding, layer.HeadDim)
		if err != nil {
			continue
		}
		sizeBytes := (keyStride + valueStride) * uint64(blockSize)
		if sizeBytes > rocmDeviceKVTensorPoolDefaultBytes {
			continue
		}
		tokenCount := contextSize
		if layer.SlidingWindow > 0 && tokenCount > layer.SlidingWindow {
			tokenCount = layer.SlidingWindow
		}
		pageCount := (tokenCount + blockSize - 1) / blockSize
		contextCounts[sizeBytes] += pageCount
		if layer.SlidingWindow > 0 && contextSize >= layer.SlidingWindow {
			ownerSlack[sizeBytes]++
		}
	}
	for sizeBytes, count := range contextCounts {
		if count > counts[sizeBytes] {
			counts[sizeBytes] = count
		}
	}
	for sizeBytes, count := range ownerSlack {
		counts[sizeBytes] += count
	}
	return counts
}

func hipAddGemma4Q4DeviceKVAppendTokenPrewarmCounts(counts map[uint64]int, cfg hipGemma4Q4ForwardConfig, mode string) {
	if counts == nil || len(cfg.Layers) == 0 {
		return
	}
	keyEncoding, valueEncoding := rocmKVEncodingsForMode(mode)
	for _, layer := range cfg.Layers {
		if layer.HeadDim <= 0 {
			continue
		}
		keyBytes, err := rocmKVTensorDeviceByteCount(keyEncoding, layer.HeadDim)
		if err == nil && keyBytes <= rocmDeviceKVTensorPoolDefaultBytes {
			counts[keyBytes]++
		}
		valueBytes, err := rocmKVTensorDeviceByteCount(valueEncoding, layer.HeadDim)
		if err == nil && valueBytes <= rocmDeviceKVTensorPoolDefaultBytes {
			counts[valueBytes]++
		}
	}
}

type hipTensor struct {
	info    nativeTensorInfo
	pointer nativeDevicePointer
}

type hipLoadedModel struct {
	driver                nativeHIPDriver
	kernels               hipKernelSet
	modelPath             string
	modelInfo             inference.ModelInfo
	modelLabels           map[string]string
	engineProfile         ROCmModelProfile
	gemma4Q4Config        hipGemma4Q4EngineConfig
	sequenceMixerPlan     *SequenceMixerLoadPlan
	sequenceMixerBindings *hipSequenceMixerBindings
	contextSize           int
	gemma4TextConfig      nativeGemma4TextConfig
	gemma4Architecture    Gemma4ArchitectureDeclaration
	tensors               map[string]hipTensor
	hostTensors           map[string]nativeTensorInfo
	expertCacheMu         sync.Mutex
	expertCache           *hipGemma4ExpertCache
	adapter               inference.AdapterIdentity
	tinyLoRA              *hipLoadedTinyLoRAAdapter
	smallLoRA             *hipLoadedSmallLoRAAdapter
	gemma4LoRA            *hipLoadedSmallLoRAAdapter
	classLoRA             *hipLoadedClassifierLoRAAdapter
	tokenText             *hipTokenTextDecoder
	audio                 *AudioTower
	vision                *HIPVisionEncoderTower
	unifiedVision         *UnifiedVisionTower
	q4ConfigMu            sync.Mutex
	q4Config              hipGemma4Q4ForwardConfig
	q4Layers              int
	q4ConfigOK            bool
	q4Suppress            []int32
	q4Stop                []int32
	q4SuppressStop        []int32
	q4SuppressStopOK      bool
	attachedDrafterMu     sync.Mutex
	attachedDrafter       *hipAttachedDrafterRuntime
	smallPriorKeys        []float32
	smallPriorValues      []float32
	tinyPriorKeys         []float32
	tinyPriorValues       []float32
	createdAt             time.Time
	closed                bool
}

func (model *hipLoadedModel) AcceptsAudioInput() bool {
	return model != nil && ((model.unifiedVision != nil && model.unifiedVision.AcceptsAudio()) ||
		(model.audio != nil && model.audio.loaded != nil))
}

func (model *hipLoadedModel) AcceptsImageInput() bool {
	return model != nil && ((model.unifiedVision != nil && model.unifiedVision.loaded != nil) ||
		(model.vision != nil && model.vision.loaded != nil))
}

func (model *hipLoadedModel) gemma4Q4EngineConfig() hipGemma4Q4EngineConfig {
	cfg := defaultHIPGemma4Q4EngineConfig()
	if model == nil {
		return cfg
	}
	if model.gemma4Q4Config.DeviceKVMode != "" {
		cfg.DeviceKVMode = model.gemma4Q4Config.DeviceKVMode
	}
	cfg.BidirectionalSpanTokens = model.gemma4Q4Config.BidirectionalSpanTokens
	return cfg
}

func (model *hipLoadedModel) modelIdentity() inference.ModelIdentity {
	if model == nil {
		return inference.ModelIdentity{}
	}
	info := model.modelInfo
	identity := inference.ModelIdentity{
		Path:          model.modelPath,
		Architecture:  firstNonEmptyString(info.Architecture, model.engineProfile.Architecture),
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: model.contextSize,
		Labels:        cloneStringMap(model.modelLabels),
	}
	if len(identity.Labels) > 0 && identity.QuantType == "" {
		identity.QuantType = identity.Labels["quant_type"]
	}
	if len(identity.Labels) > 0 && identity.QuantType == "" && rocmIsGemma4SizeQuantIdentity(identity.Architecture) {
		identity.QuantType = identity.Labels["gemma4_quant_mode"]
	}
	return rocmGemma4ModelWithInferredPathQuant(identity)
}

func (model *hipLoadedModel) ModelProfile() ROCmModelProfile {
	if model == nil {
		return ROCmModelProfile{}
	}
	identity := model.modelIdentity()
	profile := model.engineProfile
	if !profile.Matched() {
		var ok bool
		profile, ok = ResolveROCmModelProfile(identity.Path, identity)
		if !ok {
			return ROCmModelProfile{}
		}
	}
	profile.Model = identity
	return profile.clone()
}

func (model *hipLoadedModel) ROCmEngineFeatures() ROCmEngineFeatures {
	profile := model.ModelProfile()
	if !profile.Matched() {
		return ROCmEngineFeatures{}
	}
	features := profile.EngineFeatures
	if features.empty() {
		features = ROCmEngineFeaturesForProfile(profile)
	}
	return features.clone()
}

func (model *hipLoadedModel) ModelRoutePlan() ROCmModelRoutePlan {
	profile := model.ModelProfile()
	if !profile.Matched() {
		return ROCmModelRoutePlan{}
	}
	return ROCmModelRoutePlanForProfile(profile)
}

func (model *hipLoadedModel) Generate(ctx context.Context, prompt string, cfg inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	return model.kernelSet().Generate(ctx, model, prompt, cfg)
}

func (model *hipLoadedModel) Chat(ctx context.Context, messages []inference.Message, cfg inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	return model.kernelSet().Chat(ctx, model, messages, cfg)
}

func (model *hipLoadedModel) Classify(ctx context.Context, prompts []string, cfg inference.GenerateConfig) ([]inference.ClassifyResult, error) {
	return model.kernelSet().Classify(ctx, model, prompts, cfg)
}

func (model *hipLoadedModel) BatchGenerate(ctx context.Context, prompts []string, cfg inference.GenerateConfig) ([]inference.BatchResult, error) {
	return model.kernelSet().BatchGenerate(ctx, model, prompts, cfg)
}

func (model *hipLoadedModel) Project(ctx context.Context, req hipProjectionRequest) ([]float32, error) {
	return model.kernelSet().Project(ctx, model, req)
}

func (model *hipLoadedModel) Prefill(ctx context.Context, req hipPrefillRequest) (hipPrefillResult, error) {
	return model.kernelSet().Prefill(ctx, model, req)
}

func (model *hipLoadedModel) DecodeToken(ctx context.Context, req hipDecodeRequest) (hipDecodeResult, error) {
	return model.kernelSet().Decode(ctx, model, req)
}

func hipAttachedDrafterTargetRetainedDecodeStatus(model *hipLoadedModel) string {
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) ||
		!hipLoadedGemma4Q4GenerateLinked(model) ||
		model.modelInfo.NumLayers <= 0 {
		return hipKernelStatusNotLinked
	}
	if _, err := model.cachedGemma4Q4ForwardConfig(model.modelInfo.NumLayers); err != nil {
		return hipKernelStatusNotLinked
	}
	return hipKernelStatusLinked
}

func (model *hipLoadedModel) AttachAttachedDrafter(draft nativeModel, plan AttachedDrafterPlan) (AttachedDrafterAttachment, error) {
	if model == nil {
		return AttachedDrafterAttachment{}, core.E("rocm.hip.AttachAttachedDrafter", "target model is nil", nil)
	}
	draftModel, ok := draft.(*hipLoadedModel)
	if !ok || draftModel == nil {
		return AttachedDrafterAttachment{}, core.E("rocm.hip.AttachAttachedDrafter", "draft model must be a loaded HIP Gemma4 assistant", nil)
	}
	if err := validateProductionMTPAttachedDrafterPlan(plan); err != nil {
		return AttachedDrafterAttachment{}, core.E("rocm.hip.AttachAttachedDrafter", "validate plan", err)
	}
	if !isROCmGemma4Architecture(model.modelInfo.Architecture) {
		return AttachedDrafterAttachment{}, core.E("rocm.hip.AttachAttachedDrafter", "target model must be a Gemma4 text model", nil)
	}
	if !isROCmGemma4AssistantArchitecture(draftModel.modelInfo.Architecture) {
		return AttachedDrafterAttachment{}, core.E("rocm.hip.AttachAttachedDrafter", "draft model must be a Gemma4 assistant attached MTP drafter", nil)
	}
	if model.modelInfo.HiddenSize > 0 && draftModel.modelInfo.HiddenSize > 0 && model.modelInfo.HiddenSize != draftModel.modelInfo.HiddenSize {
		targetIdentity := rocmGemma4ModelWithInferredPathQuant(model.modelIdentity())
		draftIdentity := rocmGemma4ModelWithInferredPathQuant(draftModel.modelIdentity())
		backboneHidden, backboneOK := hipAttachedDrafterAssistantIntLabelValue([]map[string]string{
			draftIdentity.Labels,
			draftModel.modelLabels,
			targetIdentity.Labels,
			plan.Labels,
		},
			"attached_drafter_assistant_backbone_hidden_size",
			"attached.drafter.assistant.backbone_hidden_size",
			"engine_attached_drafter_assistant_backbone_hidden_size",
		)
		if !backboneOK {
			return AttachedDrafterAttachment{}, core.E("rocm.hip.AttachAttachedDrafter", core.Sprintf("draft hidden size %d differs from target hidden size %d and assistant backbone hidden size is missing", draftModel.modelInfo.HiddenSize, model.modelInfo.HiddenSize), nil)
		}
		if backboneHidden != model.modelInfo.HiddenSize {
			return AttachedDrafterAttachment{}, core.E("rocm.hip.AttachAttachedDrafter", core.Sprintf("assistant backbone hidden size %d does not match target hidden size %d", backboneHidden, model.modelInfo.HiddenSize), nil)
		}
	}
	if model.modelInfo.VocabSize > 0 && draftModel.modelInfo.VocabSize > 0 && model.modelInfo.VocabSize != draftModel.modelInfo.VocabSize {
		return AttachedDrafterAttachment{}, core.E("rocm.hip.AttachAttachedDrafter", core.Sprintf("draft vocab size %d does not match target vocab size %d", draftModel.modelInfo.VocabSize, model.modelInfo.VocabSize), nil)
	}
	labels := cloneStringMap(plan.Labels)
	if labels == nil {
		labels = map[string]string{}
	}
	targetRetainedDecode := hipAttachedDrafterTargetRetainedDecodeStatus(model)
	nativeHandoff := attachedDrafterNativeHandoffPendingTargetDecode
	if targetRetainedDecode == hipKernelStatusLinked {
		nativeHandoff = attachedDrafterNativeHandoffTargetDecodeOnly
	}
	assistantVerify := hipKernelStatusNotLinked
	assistantPreflight := hipAttachedDrafterAssistantVerifierPreflightFor(model, draftModel, plan.Labels)
	for key, value := range assistantPreflight.Labels() {
		labels[key] = value
	}
	assistantPlan, assistantPlanErr := hipAttachedDrafterAssistantVerifierPlanFor(model, draftModel, plan.Labels)
	assistantPlanStatus := assistantPlan.Status
	inputPlan := hipAttachedDrafterAssistantDraftStepInputPlan{}
	softcap := draftModel.loadedGemma4Q4FinalLogitSoftcap()
	if assistantPlanErr != nil {
		assistantPlanStatus = attachedDrafterAssistantVerifierPlanUnsupported
		labels["attached_drafter_assistant_verifier_plan"] = assistantPlanStatus
		labels["attached_drafter_assistant_verifier_plan_reason"] = assistantPlanErr.Error()
		labels["attached_drafter_assistant_verifier_kernel"] = "not_linked"
	} else {
		for key, value := range assistantPlan.Labels() {
			labels[key] = value
		}
		for key, value := range hipAttachedDrafterAssistantLayerRuntimeLabels(assistantPlan) {
			labels[key] = value
		}
		if targetRetainedDecode == hipKernelStatusLinked && assistantPlan.Status == attachedDrafterAssistantVerifierPlanTensorBound {
			inputPlan = hipAttachedDrafterAssistantDraftStepInputPlanForModel(model, assistantPlan)
			for key, value := range inputPlan.Labels() {
				labels[key] = value
			}
			for key, value := range hipAttachedDrafterAssistantDraftStepHiddenRuntimeLabels(assistantPlan, inputPlan) {
				labels[key] = value
			}
			for key, value := range hipAttachedDrafterAssistantDraftStepProposalRuntimeLabels(assistantPlan, inputPlan, softcap) {
				labels[key] = value
			}
		}
	}
	linked := targetRetainedDecode == hipKernelStatusLinked &&
		assistantPlanErr == nil &&
		assistantPlan.Status == attachedDrafterAssistantVerifierPlanTensorBound &&
		inputPlan.Status == attachedDrafterAssistantDraftStepInputLinked &&
		hipAttachedDrafterAssistantDraftStepProposalPlanInvalidReason(assistantPlan, softcap) == nil
	if linked {
		nativeHandoff = attachedDrafterNativeHandoffRetainedStateVerifier
		assistantVerify = hipKernelStatusLinked
	}
	nativeAttachment := hipKernelStatusNotLinked
	var linkedRuntime *hipAttachedDrafterRuntime
	if linked {
		nativeAttachment = hipKernelStatusLinked
		linkedRuntime = &hipAttachedDrafterRuntime{
			draft:         draftModel,
			assistantPlan: assistantPlan,
			inputPlan:     inputPlan,
			softcap:       softcap,
		}
	}
	labels["attached_drafter_native_attachment"] = nativeAttachment
	labels["attached_drafter_native_handoff"] = nativeHandoff
	labels["attached_drafter_prompt_replay_fallback"] = "forbidden"
	labels["attached_drafter_retained_state_entrypoint"] = hipKernelStatusLinked
	labels["attached_drafter_retained_state_required"] = "true"
	labels["attached_drafter_runtime"] = "hip"
	labels["attached_drafter_state_source"] = "rocm_state_session_runtime_kv"
	labels["attached_drafter_target_retained_decode"] = targetRetainedDecode
	labels["attached_drafter_target_retained_state_decode"] = targetRetainedDecode
	labels["attached_drafter_assistant_verify"] = assistantVerify
	labels["attached_drafter_assistant_state_verify"] = assistantVerify
	attachment := AttachedDrafterAttachment{
		Plan:             plan,
		Target:           rocmNormalizeModelInfo(model.modelInfo),
		Draft:            rocmNormalizeModelInfo(draftModel.modelInfo),
		NativeAttachment: nativeAttachment,
		Labels:           labels,
	}
	if linkedRuntime != nil {
		linkedRuntime.attachment = cloneAttachedDrafterAttachment(attachment)
		model.storeAttachedDrafterRuntime(linkedRuntime)
	} else {
		model.storeAttachedDrafterRuntime(nil)
	}
	return attachment, attachedDrafterAttachError(linked, targetRetainedDecode, assistantVerify, assistantPreflight.Status, assistantPlanStatus)
}

func (model *hipLoadedModel) Encode(text string) []int32 {
	if model != nil && model.tokenText != nil {
		return model.tokenText.Encode(text)
	}
	return approximateTokenIDs(text)
}

func (model *hipLoadedModel) Decode(ids []int32) string {
	if model != nil && model.tokenText != nil {
		return model.tokenText.Decode(ids)
	}
	if len(ids) == 0 {
		return ""
	}
	return core.Sprintf("%d tokens", len(ids))
}

func (model *hipLoadedModel) ApplyChatTemplate(messages []inference.Message) (string, error) {
	if model != nil && isROCmGemma4Architecture(model.modelInfo.Architecture) {
		return formatGemma4ChatTemplate(messages), nil
	}
	if model != nil {
		if prompt, ok := formatHIPArchitectureChatTemplate(messages, model.modelInfo.Architecture, nil); ok {
			return prompt, nil
		}
	}
	return formatFallbackChatTemplate(messages), nil
}

func (model *hipLoadedModel) applyChatTemplateWithGenerateConfig(messages []inference.Message, cfg inference.GenerateConfig) (string, error) {
	if model != nil && isROCmGemma4Architecture(model.modelInfo.Architecture) {
		return formatGemma4ChatTemplateWithConfig(messages, model.gemma4ChatTemplateConfig(cfg, false)), nil
	}
	if model != nil {
		if prompt, ok := formatHIPArchitectureChatTemplate(messages, model.modelInfo.Architecture, cfg.EnableThinking); ok {
			return prompt, nil
		}
	}
	return formatFallbackChatTemplate(messages), nil
}

func (model *hipLoadedModel) LoadAdapter(path string) (inference.AdapterIdentity, error) {
	if core.Trim(path) == "" {
		return inference.AdapterIdentity{}, core.E("rocm.hip.LoadAdapter", "adapter path is required", nil)
	}
	if model != nil && hipLoadedGemma4Q4GenerateLinked(model) && isROCmGemma4Architecture(model.modelInfo.Architecture) {
		adapter, identity, err := model.loadGemma4HeadLoRAAdapter(path)
		if err != nil {
			return inference.AdapterIdentity{}, err
		}
		model.gemma4LoRA = adapter
		model.smallLoRA = nil
		model.tinyLoRA = nil
		model.classLoRA = nil
		model.adapter = cloneAdapterIdentity(identity)
		return cloneAdapterIdentity(identity), nil
	}
	if model == nil || normalizeHIPKernelStatus(model.KernelStatus()).LoRA != hipKernelStatusLinked {
		return inference.AdapterIdentity{}, core.E("rocm.hip.LoadAdapter", "native LoRA adapter application is not linked yet: "+path, nil)
	}
	if smallCfg, err := model.loadedSmallDecodeConfig(); err == nil {
		adapter, identity, err := model.loadSmallLoRAAdapter(path, smallCfg)
		if err != nil {
			return inference.AdapterIdentity{}, err
		}
		model.smallLoRA = adapter
		model.gemma4LoRA = nil
		model.tinyLoRA = nil
		model.classLoRA = nil
		model.adapter = cloneAdapterIdentity(identity)
		return cloneAdapterIdentity(identity), nil
	}
	if _, err := model.loadedTinyLMConfig(); err == nil {
		adapter, identity, err := model.loadTinyLoRAAdapter(path)
		if err != nil {
			return inference.AdapterIdentity{}, err
		}
		model.tinyLoRA = adapter
		model.smallLoRA = nil
		model.gemma4LoRA = nil
		model.classLoRA = nil
		model.adapter = cloneAdapterIdentity(identity)
		return cloneAdapterIdentity(identity), nil
	}
	classifier, hasClassifier, err := model.loadedSequenceClassifierConfig()
	if err != nil {
		return inference.AdapterIdentity{}, err
	}
	if hasClassifier {
		adapter, identity, err := model.loadClassifierLoRAAdapter(path, classifier)
		if err != nil {
			return inference.AdapterIdentity{}, err
		}
		model.classLoRA = adapter
		model.tinyLoRA = nil
		model.smallLoRA = nil
		model.gemma4LoRA = nil
		model.adapter = cloneAdapterIdentity(identity)
		return cloneAdapterIdentity(identity), nil
	}
	return inference.AdapterIdentity{}, core.E("rocm.hip.LoadAdapter", "no loaded LoRA adapter target supports this model", nil)
}

func (model *hipLoadedModel) UnloadAdapter() error {
	model.adapter = inference.AdapterIdentity{}
	model.tinyLoRA = nil
	model.smallLoRA = nil
	model.gemma4LoRA = nil
	model.classLoRA = nil
	return nil
}

func validateHIPLoadConfig(cfg nativeLoadConfig) error {
	if !hipSupportedModelQuantization(cfg.ModelInfo) {
		return core.E("rocm.hip.Validate", "unsupported quantization", nil)
	}
	if cfg.DeviceKVMode != "" && !isROCmKVCacheMode(cfg.DeviceKVMode) {
		return core.E("rocm.hip.Validate", core.Sprintf("unsupported device KV cache mode %q", cfg.DeviceKVMode), nil)
	}
	if cfg.DataOffset < 0 {
		return core.E("rocm.hip.Validate", "data offset must be non-negative", nil)
	}
	if len(cfg.Tensors) == 0 {
		return core.E("rocm.hip.Validate", "missing token embedding tensor", nil)
	}
	hasEmbedding := false
	hasOutput := false
	layerIDs := map[string]struct{}{}
	tensorNames := map[string]struct{}{}
	for _, tensor := range cfg.Tensors {
		if core.Trim(tensor.Name) == "" {
			return core.E("rocm.hip.Validate", "tensor name is required", nil)
		}
		name := core.Lower(tensor.Name)
		if _, exists := tensorNames[name]; exists {
			return core.E("rocm.hip.Validate", "duplicate tensor name "+tensor.Name, nil)
		}
		tensorNames[name] = struct{}{}
		if err := validateHIPTensorDataOffset(hipTensorDataOffset(cfg, tensor), tensor); err != nil {
			return err
		}
		if !hipSupportedTensorDType(tensor) {
			return core.E("rocm.hip.Validate", "unsupported tensor dtype "+tensor.Name, nil)
		}
		if err := validateHIPTensorShape(cfg.ModelInfo, tensor); err != nil {
			return err
		}
		if isHIPEmbeddingTensor(name) {
			if tensor.ByteSize == 0 {
				return core.E("rocm.hip.Validate", "required tensor has zero byte size "+tensor.Name, nil)
			}
			hasEmbedding = true
		}
		if isHIPOutputTensor(name) {
			if tensor.ByteSize == 0 {
				return core.E("rocm.hip.Validate", "required tensor has zero byte size "+tensor.Name, nil)
			}
			hasOutput = true
		}
		if layerID := hipLayerID(name); layerID != "" {
			layerIDs[layerID] = struct{}{}
		}
	}
	if !hasEmbedding {
		return core.E("rocm.hip.Validate", "missing token embedding tensor", nil)
	}
	if !hasOutput && hipLoadConfigRequiresOutputHead(cfg) {
		return core.E("rocm.hip.Validate", "missing output head tensor", nil)
	}
	if cfg.ModelInfo.NumLayers > 0 && len(layerIDs) > 0 && len(layerIDs) != cfg.ModelInfo.NumLayers {
		return core.E("rocm.hip.Validate", core.Sprintf("mismatched layer count: metadata=%d tensors=%d", cfg.ModelInfo.NumLayers, len(layerIDs)), nil)
	}
	return nil
}

func validateHIPTensorDataOffset(dataOffset int64, tensor nativeTensorInfo) error {
	const maxInt64 = int64(1<<63 - 1)
	if tensor.Offset > uint64(maxInt64-dataOffset) {
		return core.E("rocm.hip.Validate", "tensor data offset overflows int64 "+tensor.Name, nil)
	}
	return nil
}

func validateHIPTensorFileRanges(path string, cfg nativeLoadConfig) error {
	for _, tensor := range cfg.Tensors {
		if tensor.ByteSize == 0 {
			continue
		}
		sourcePath := hipTensorSourcePath(path, tensor)
		stat := core.Stat(sourcePath)
		if !stat.OK {
			return stat.Value.(error)
		}
		size := stat.Value.(core.FsFileInfo).Size()
		if size < 0 {
			return core.E("rocm.hip.Validate", "model file size is invalid", nil)
		}
		start := hipTensorDataOffset(cfg, tensor) + int64(tensor.Offset)
		end, err := hipTensorFileEnd(start, tensor.ByteSize)
		if err != nil {
			return core.E("rocm.hip.Validate", "tensor byte range "+tensor.Name, err)
		}
		if end > size {
			return core.E("rocm.hip.Validate", "tensor byte range exceeds file size "+tensor.Name, nil)
		}
	}
	return nil
}

func hipTensorSourcePath(defaultPath string, tensor nativeTensorInfo) string {
	if tensor.SourcePath != "" {
		return tensor.SourcePath
	}
	return defaultPath
}

func hipTensorDataOffset(cfg nativeLoadConfig, tensor nativeTensorInfo) int64 {
	if tensor.SourcePath != "" || tensor.DataOffset != 0 {
		return tensor.DataOffset
	}
	return cfg.DataOffset
}

func hipTensorFileEnd(start int64, byteSize uint64) (int64, error) {
	const maxInt64 = int64(1<<63 - 1)
	if start < 0 {
		return 0, core.E("rocm.hip.TensorRange", "start offset is negative", nil)
	}
	if byteSize > uint64(maxInt64-start) {
		return 0, core.E("rocm.hip.TensorRange", "end offset overflows int64", nil)
	}
	return start + int64(byteSize), nil
}

func validateHIPTensorShape(info inference.ModelInfo, tensor nativeTensorInfo) error {
	if len(tensor.Dimensions) == 0 {
		return nil
	}
	elements, err := hipTensorElementCount(tensor.Dimensions)
	if err != nil {
		return core.E("rocm.hip.Validate", "invalid tensor dimensions "+tensor.Name, err)
	}
	if expectedBytes, ok := hipExpectedTensorBytes(tensor.Type, elements); ok && tensor.ByteSize > 0 && tensor.ByteSize != expectedBytes {
		return core.E("rocm.hip.Validate", core.Sprintf("tensor byte size mismatch %s: metadata=%d expected=%d", tensor.Name, tensor.ByteSize, expectedBytes), nil)
	}
	name := core.Lower(tensor.Name)
	if !isHIPEmbeddingTensor(name) && !isHIPOutputTensor(name) {
		return nil
	}
	if len(tensor.Dimensions) != 2 {
		return core.E("rocm.hip.Validate", "projection tensor must be rank 2 "+tensor.Name, nil)
	}
	if info.HiddenSize > 0 && !hipTensorDimensionsContainLogical(tensor, uint64(info.HiddenSize), info) {
		return core.E("rocm.hip.Validate", core.Sprintf("projection tensor %s missing hidden size %d", tensor.Name, info.HiddenSize), nil)
	}
	if info.VocabSize > 0 && !hipDimensionsContain(tensor.Dimensions, uint64(info.VocabSize)) {
		return core.E("rocm.hip.Validate", core.Sprintf("projection tensor %s missing vocab size %d", tensor.Name, info.VocabSize), nil)
	}
	return nil
}

func isHIPEmbeddingTensor(name string) bool {
	if isHIPPerLayerEmbeddingTensor(name) {
		return false
	}
	return core.Contains(name, "tok_embeddings.weight") ||
		core.Contains(name, "token_embd.weight") ||
		core.Contains(name, "embed_tokens.weight") ||
		core.Contains(name, "word_embeddings.weight")
}

func isHIPPerLayerEmbeddingTensor(name string) bool {
	name = core.Lower(name)
	return core.Contains(name, "per_layer_token_embd.weight") ||
		core.Contains(name, "embed_tokens_per_layer.weight")
}

func isHIPOutputTensor(name string) bool {
	name = core.Lower(name)
	return name == "output.weight" ||
		name == "lm_head.weight" ||
		core.HasSuffix(name, ".output.weight") ||
		core.HasSuffix(name, ".lm_head.weight")
}

func hipLoadConfigRequiresOutputHead(cfg nativeLoadConfig) bool {
	if cfg.TiedWordEmbeddings {
		return false
	}
	return normalizeROCmArchitecture(cfg.ModelInfo.Architecture) != "bert"
}

func hipTensorElementCount(dimensions []uint64) (uint64, error) {
	if len(dimensions) == 0 {
		return 0, core.E("rocm.hip.TensorShape", "tensor has no dimensions", nil)
	}
	elements := uint64(1)
	for _, dimension := range dimensions {
		if dimension == 0 {
			return 0, core.E("rocm.hip.TensorShape", "tensor has a zero dimension", nil)
		}
		if elements > ^uint64(0)/dimension {
			return 0, core.E("rocm.hip.TensorShape", "tensor element count overflows uint64", nil)
		}
		elements *= dimension
	}
	return elements, nil
}

func hipExpectedTensorBytes(tensorType uint32, elements uint64) (uint64, bool) {
	blockSize, typeSize, ok := hipTensorBlockSize(tensorType)
	if !ok {
		return 0, false
	}
	blocks := (elements + blockSize - 1) / blockSize
	if blocks > ^uint64(0)/typeSize {
		return 0, false
	}
	return blocks * typeSize, true
}

func hipTensorBlockSize(tensorType uint32) (blockSize, typeSize uint64, ok bool) {
	switch tensorType {
	case 0:
		return 1, 4, true
	case 1, 30:
		return 1, 2, true
	case 2:
		return 32, 18, true
	case 3:
		return 32, 20, true
	case 6:
		return 32, 22, true
	case 7:
		return 32, 24, true
	case 8:
		return 32, 34, true
	case 10:
		return 256, 84, true
	case 11:
		return 256, 110, true
	case 12:
		return 256, 144, true
	case 13:
		return 256, 176, true
	case 14:
		return 256, 210, true
	case 15:
		return 256, 292, true
	case 24:
		return 1, 1, true
	case 25:
		return 1, 2, true
	case 26:
		return 1, 4, true
	case 27, 28:
		return 1, 8, true
	default:
		return 0, 0, false
	}
}

func hipDimensionsContain(dimensions []uint64, value uint64) bool {
	for _, dimension := range dimensions {
		if dimension == value {
			return true
		}
	}
	return false
}

func hipTensorDimensionsContainLogical(tensor nativeTensorInfo, value uint64, info inference.ModelInfo) bool {
	if hipDimensionsContain(tensor.Dimensions, value) {
		return true
	}
	if hipMLXAffineSupportedBits(info.QuantBits) && (tensor.Type == 26 || core.Upper(tensor.TypeName) == "U32") {
		for _, dimension := range tensor.Dimensions {
			if dimension > uint64(int(^uint(0)>>1)) {
				continue
			}
			for _, bits := range hipMLXAffineCandidateBits(info.QuantBits) {
				cols, err := hipMLXAffineColsFromPackedCols(int(dimension), bits)
				if err == nil && uint64(cols) == value {
					return true
				}
			}
		}
	}
	return false
}

func hipSupportedModelQuantization(info inference.ModelInfo) bool {
	if info.QuantBits == 0 && info.QuantGroup == 0 {
		return true
	}
	return info.QuantBits == 0 || info.QuantBits == 2 || info.QuantBits == 3 || info.QuantBits == 4 || info.QuantBits == 5 || info.QuantBits == 6 || info.QuantBits == 8 || info.QuantBits == 16 || info.QuantBits == 32
}

func hipSupportedTensorDType(tensor nativeTensorInfo) bool {
	if _, _, ok := hipTensorBlockSize(tensor.Type); ok {
		return true
	}
	name := core.Lower(tensor.TypeName)
	if name == "" {
		return false
	}
	return name == "f32" || name == "f16" || name == "q8_0" || name == "q4_k" || name == "q4_k_m" ||
		core.Contains(name, "jangtq") || core.Contains(name, "mxtq") ||
		core.Contains(name, "codebook") || core.Contains(name, "vq")
}

func hipLayerID(name string) string {
	const marker = "layers."
	index := core.Index(name, marker)
	if index < 0 {
		return ""
	}
	rest := name[index+len(marker):]
	end := 0
	for end < len(rest) && rest[end] >= '0' && rest[end] <= '9' {
		end++
	}
	if end == 0 {
		return ""
	}
	return rest[:end]
}

func (model *hipLoadedModel) ActiveAdapter() inference.AdapterIdentity {
	if model == nil {
		return inference.AdapterIdentity{}
	}
	return cloneAdapterIdentity(model.adapter)
}

func (model *hipLoadedModel) KernelStatus() hipKernelStatus {
	return model.tinyLoadedKernelStatus(model.kernelSet().Status())
}

func (model *hipLoadedModel) Metrics() inference.GenerateMetrics {
	if model == nil {
		return inference.GenerateMetrics{}
	}
	metrics := inference.GenerateMetrics{ActiveMemoryBytes: model.deviceBytes()}
	metrics.PeakMemoryBytes = metrics.ActiveMemoryBytes
	return metrics
}

func (model *hipLoadedModel) Close() error {
	if model == nil || model.closed {
		return nil
	}
	var lastErr error
	if err := model.audio.Close(); err != nil {
		lastErr = core.E("rocm.hip.Close", "close audio tower", err)
	}
	model.audio = nil
	if err := model.vision.Close(); err != nil {
		lastErr = core.E("rocm.hip.Close", "close vision encoder tower", err)
	}
	model.vision = nil
	if err := model.unifiedVision.Close(); err != nil {
		lastErr = core.E("rocm.hip.Close", "close unified vision tower", err)
	}
	model.unifiedVision = nil
	model.expertCacheMu.Lock()
	if err := model.expertCache.Close(); err != nil {
		lastErr = core.E("rocm.hip.Close", "close expert cache", err)
	}
	model.expertCache = nil
	model.expertCacheMu.Unlock()
	for name, tensor := range model.tensors {
		if err := model.driver.Free(tensor.pointer); err != nil {
			lastErr = core.E("rocm.hip.Close", "free tensor "+name, err)
		}
		delete(model.tensors, name)
	}
	for name := range model.hostTensors {
		delete(model.hostTensors, name)
	}
	model.adapter = inference.AdapterIdentity{}
	model.tinyLoRA = nil
	model.smallLoRA = nil
	model.gemma4LoRA = nil
	model.classLoRA = nil
	model.storeAttachedDrafterRuntime(nil)
	model.closed = true
	return lastErr
}

func (model *hipLoadedModel) deviceBytes() uint64 {
	var total uint64
	for _, tensor := range model.tensors {
		total += tensor.info.ByteSize
	}
	return total
}

func closeTensorSourceFiles(files map[string]*core.OSFile) {
	for path, file := range files {
		_ = file.Close()
		delete(files, path)
	}
}

func copyTensorToDevice(driver nativeHIPDriver, path string, dataOffset int64, tensor hipTensor, buffer []byte, fileCache map[string]*core.OSFile) ([]byte, error) {
	sourcePath := tensor.info.SourcePath
	if sourcePath == "" {
		sourcePath = path
	} else {
		dataOffset = tensor.info.DataOffset
	}
	if tensor.info.SourcePath == "" && tensor.info.DataOffset != 0 {
		dataOffset = tensor.info.DataOffset
	}
	file := fileCache[sourcePath]
	closeFile := false
	if file == nil {
		fileResult := core.Open(sourcePath)
		if !fileResult.OK {
			return buffer, fileResult.Value.(error)
		}
		file = fileResult.Value.(*core.OSFile)
		if fileCache != nil {
			fileCache[sourcePath] = file
		} else {
			closeFile = true
		}
	}
	if closeFile {
		defer file.Close()
	}

	start := dataOffset + int64(tensor.info.Offset)
	if _, err := file.Seek(start, io.SeekStart); err != nil {
		return buffer, err
	}

	remaining := tensor.info.ByteSize
	bufferBytes := int(min(uint64(nativeTensorCopyChunkBytes), remaining))
	if cap(buffer) < bufferBytes {
		buffer = make([]byte, bufferBytes)
	} else {
		buffer = buffer[:bufferBytes]
	}
	var copied uint64
	for remaining > 0 {
		chunk := int(min(uint64(len(buffer)), remaining))
		if _, err := io.ReadFull(file, buffer[:chunk]); err != nil {
			return buffer, err
		}
		if err := hipCopyPinnedHostToDevice(driver, tensor.pointer+nativeDevicePointer(copied), buffer[:chunk]); err != nil {
			return buffer, err
		}
		copied += uint64(chunk)
		remaining -= uint64(chunk)
	}
	return buffer, nil
}
