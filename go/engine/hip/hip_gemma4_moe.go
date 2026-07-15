// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"
	"sync"
	"syscall"

	core "dappco.re/go"
)

const (
	hipGGUFQ4_0ProjectionLaunchArgsVersion        uint32 = 1
	hipGGUFQ4_0ProjectionLaunchArgsBytes                 = 64
	hipGGUFQ4KExpandLaunchArgsVersion             uint32 = 1
	hipGGUFQ4KExpandLaunchArgsBytes                      = 48
	hipGGUFQ4_0ProjectionBlockSize                uint32 = 256
	hipGGUFQ4_0ProjectionRowsPerBlock             uint32 = 8
	hipGGUFQ4_0SelectedExpertsLaunchArgsVersion   uint32 = 1
	hipGGUFQ4_0SelectedExpertsLaunchArgsBytes            = 240
	hipGGUFQ4_0SelectedExpertsMaxTopK                    = 8
	hipGGUFQ4_0SelectedExpertsPair16RowsPerBlock  uint32 = 16
	hipGGUFQ4KSelectedExpertsSplitRowsPerBlock    uint32 = 8
	hipGGUFQ5_1SelectedExpertsExpert8RowsPerBlock uint32 = 2
	hipGGUFQ4KExpandedBlockBytes                         = 152
	hipGGUFExpertFormatQ4_0                       uint32 = 1
	hipGGUFExpertFormatQ4K                        uint32 = 2
	hipGGUFExpertFormatQ5_1                       uint32 = 3
	hipGGUFExpertFormatQ8_0                       uint32 = 4
	hipGGUFExpertFormatQ4KExpanded                uint32 = 5
)

const (
	hipGemma4SelectedExpertPair16Env      = "GO_ROCM_GEMMA4_Q4_SELECTED_EXPERT_PAIR16"
	hipGemma4SelectedExpertGateUpSplitEnv = "GO_ROCM_GEMMA4_Q4_K_GATE_UP_SPLIT"
	hipGemma4SelectedExpertDownExpert8Env = "GO_ROCM_GEMMA4_Q5_1_DOWN_EXPERT8"
	hipGemma4Q4KExpandedEnv               = "GO_ROCM_GEMMA4_Q4_K_EXPANDED"
)

type hipGGUFQ4_0ProjectionLaunchArgs struct {
	InputPointer  nativeDevicePointer
	WeightPointer nativeDevicePointer
	OutputPointer nativeDevicePointer
	Rows          int
	Cols          int
	RowOffset     int
	WeightRows    int
	InputBytes    uint64
	WeightBytes   uint64
	OutputBytes   uint64
}

type hipGGUFQ4KExpandLaunchArgs struct {
	RawPointer      nativeDevicePointer
	ExpandedPointer nativeDevicePointer
	BlockCount      int
	RawBytes        uint64
	ExpandedBytes   uint64
}

type hipGGUFQ4_0SelectedExpertsLaunchArgs struct {
	InputPointer      nativeDevicePointer
	ActivationPointer nativeDevicePointer
	OutputPointer     nativeDevicePointer
	GateUpPointers    [hipGGUFQ4_0SelectedExpertsMaxTopK]nativeDevicePointer
	DownPointers      [hipGGUFQ4_0SelectedExpertsMaxTopK]nativeDevicePointer
	RouteWeights      [hipGGUFQ4_0SelectedExpertsMaxTopK]float32
	TopK              int
	HiddenSize        int
	ExpertFF          int
	GateUpRows        int
	DownRows          int
	InputBytes        uint64
	ActivationBytes   uint64
	OutputBytes       uint64
	GateUpBytes       uint64
	DownBytes         uint64
	GateUpFormat      uint32
	DownFormat        uint32
}

const (
	hipGemma4ExpertCacheDefaultBytes = 6 * memoryGiB
	hipGemma4ExpertCacheMaximumBytes = 8 * memoryGiB
	hipGemma4ExpertCacheReserveBytes = 2 * memoryGiB
	hipGemma4ExpertCacheRefreshEvery = 256
)

type hipGemma4ExpertCacheKey struct {
	Layer  int
	Expert int
}

type hipGemma4MoEExpertStorage uint8

const (
	hipGemma4MoEExpertStorageGGUF hipGemma4MoEExpertStorage = iota
	hipGemma4MoEExpertStorageMLXAffine
)

type hipGemma4MLXAffineTensorSet struct {
	Weight nativeTensorInfo
	Scales nativeTensorInfo
	Biases nativeTensorInfo
}

type hipGemma4MLXAffineExpertSource struct {
	GateUp hipGemma4MLXAffineTensorSet
	Down   hipGemma4MLXAffineTensorSet
}

type hipGemma4ExpertCacheEntry struct {
	Storage         hipGemma4MoEExpertStorage
	GateUp          *hipDeviceByteBuffer
	Down            *hipDeviceByteBuffer
	GateUpRows      int
	GateUpCols      int
	DownRows        int
	DownCols        int
	GateUpFormat    uint32
	DownFormat      uint32
	MLXGateUpWeight *hipDeviceByteBuffer
	MLXGateUpScales *hipDeviceByteBuffer
	MLXGateUpBiases *hipDeviceByteBuffer
	MLXDownWeight   *hipDeviceByteBuffer
	MLXDownScales   *hipDeviceByteBuffer
	MLXDownBiases   *hipDeviceByteBuffer
	MLXGate         hipMLXQ4DeviceWeightConfig
	MLXUp           hipMLXQ4DeviceWeightConfig
	MLXDown         hipMLXQ4DeviceWeightConfig
	bytes           uint64
	lastUse         uint64
}

type hipGemma4MappedExpertSource struct {
	file *core.OSFile
	data []byte
}

type hipGemma4ExpertCacheStats struct {
	Hits              uint64
	Misses            uint64
	Evictions         uint64
	HostMappings      uint64
	HostMappedBytes   uint64
	H2DBytes          uint64
	AllocationRetries uint64
	BudgetRefreshes   uint64
}

type hipGemma4MoERouterProjectionConfig struct {
	WeightPointer nativeDevicePointer
	WeightBytes   uint64
	Rows          int
	Cols          int
}

type hipGemma4MoELayerConfig struct {
	Layer                  int
	NumExperts             int
	TopKExperts            int
	ExpertIntermediateSize int
	PreFeedForwardNorm2    hipRMSNormDeviceWeightConfig
	PostFeedForwardNorm1   hipRMSNormDeviceWeightConfig
	PostFeedForwardNorm2   hipRMSNormDeviceWeightConfig
	RouterNorm             hipRMSNormDeviceWeightConfig
	RouterProjection       hipGemma4MoERouterProjectionConfig
	RouterProjectionMLX    hipMLXQ4DeviceWeightConfig
	PerExpertScale         []float32
	ExpertCache            *hipGemma4ExpertCache
	ExpertStorage          hipGemma4MoEExpertStorage
	GateUpInfo             nativeTensorInfo
	DownInfo               nativeTensorInfo
	MLXExperts             hipGemma4MLXAffineExpertSource
	MLXHiddenSize          int
	MLXGroupSize           int
	MLXPreferredBits       int
}

type hipGemma4ExpertCache struct {
	driver                          nativeHIPDriver
	maxBytes                        uint64
	bytes                           uint64
	clock                           uint64
	adaptive                        bool
	minimumEntries                  int
	entries                         map[hipGemma4ExpertCacheKey]*hipGemma4ExpertCacheEntry
	gateUpStaging                   *hipDeviceByteBuffer
	expandedGPUDisabled             bool
	sources                         map[string]*hipGemma4MappedExpertSource
	stats                           hipGemma4ExpertCacheStats
	releaseTransientPoolSuppression func()
	mu                              sync.Mutex
}

const hipGemma4MoERouterMaximumOutputBytes = hipGGUFQ4_0SelectedExpertsMaxTopK*8 + 4

type hipGemma4MoEWorkspace struct {
	HiddenPairFixed      hipDeviceByteBuffer
	HiddenPairCap        int
	HiddenViews          [2]hipDeviceByteBuffer
	BatchOutputFixed     hipDeviceByteBuffer
	BatchOutputCap       int
	BatchOutputView      hipDeviceByteBuffer
	ExpertDownFixed      hipDeviceByteBuffer
	ExpertDownCap        int
	ExpertDownView       hipDeviceByteBuffer
	RouterScoresFixed    hipDeviceByteBuffer
	RouterScoresCap      int
	RouterScoresView     hipDeviceByteBuffer
	RouterOutputFixed    hipDeviceByteBuffer
	RouterOutputView     hipDeviceByteBuffer
	RouterIDView         hipDeviceByteBuffer
	RouterProbView       hipDeviceByteBuffer
	RouterStatusView     hipDeviceByteBuffer
	RouterBuffers        hipMoERouterDeviceBuffers
	RouterArgs           [hipMoERouterLaunchArgsBytes]byte
	RouterProjectionArgs [hipProjectionLaunchArgsBytes]byte
	SelectedExpertArgs   [hipGGUFQ4_0SelectedExpertsLaunchArgsBytes]byte
	CombineNormsArgs     [hipMoECombineNormsLaunchArgsBytes]byte
	RouterPayload        [hipGemma4MoERouterMaximumOutputBytes]byte
	Routes               [hipGGUFQ4_0SelectedExpertsMaxTopK]rocmExpertRoute
	Entries              [hipGGUFQ4_0SelectedExpertsMaxTopK]*hipGemma4ExpertCacheEntry
	RouteWeights         [hipGGUFQ4_0SelectedExpertsMaxTopK]float32
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureMoEHiddenOutput(driver nativeHIPDriver, count, slot int) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "attention workspace is required", nil)
	}
	if slot < 0 || slot >= len(workspace.MoE.HiddenViews) {
		return nil, core.E("rocm.hip.Gemma4MoE", "MoE hidden output slot is out of range", nil)
	}
	return workspace.ensureFixedPairOutputReusableCapacity(
		driver,
		&workspace.MoE.HiddenPairFixed,
		&workspace.MoE.HiddenPairCap,
		&workspace.MoE.HiddenViews,
		count,
		slot,
		"MoE hidden output pair",
		"MoE hidden output view",
	)
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureMoERouterScores(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "attention workspace is required", nil)
	}
	return workspace.ensureFixedOutputReusableCapacity(
		driver,
		&workspace.MoE.RouterScoresFixed,
		&workspace.MoE.RouterScoresCap,
		&workspace.MoE.RouterScoresView,
		count,
		"MoE router scores",
		"MoE router score view",
	)
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureMoEBatchOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "attention workspace is required", nil)
	}
	return workspace.ensureFixedOutputReusableCapacity(
		driver,
		&workspace.MoE.BatchOutputFixed,
		&workspace.MoE.BatchOutputCap,
		&workspace.MoE.BatchOutputView,
		count,
		"MoE batch output",
		"MoE batch output view",
	)
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureMoEExpertDownOutput(driver nativeHIPDriver, count int) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "attention workspace is required", nil)
	}
	return workspace.ensureFixedOutputReusableCapacity(
		driver,
		&workspace.MoE.ExpertDownFixed,
		&workspace.MoE.ExpertDownCap,
		&workspace.MoE.ExpertDownView,
		count,
		"MoE expert down output",
		"MoE expert down output view",
	)
}

func (workspace *hipAttentionHeadsChunkedWorkspace) prepareMoERouterBuffers(driver nativeHIPDriver, logits *hipDeviceByteBuffer, topK, layer int) (*hipMoERouterDeviceBuffers, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "attention workspace is required", nil)
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.MoERouterLaunch", "HIP driver is not available", nil)
	}
	if logits == nil || logits.Pointer() == 0 || logits.Count() <= 0 || logits.SizeBytes() != uint64(logits.Count()*4) {
		return nil, core.E("rocm.hip.MoERouterLaunch", "router logits device buffer shape mismatch", nil)
	}
	if topK <= 0 || topK > logits.Count() || topK > hipGGUFQ4_0SelectedExpertsMaxTopK || layer < 0 {
		return nil, core.E("rocm.hip.MoERouterLaunch", "top-k and layer must fit the router workspace", nil)
	}
	moe := &workspace.MoE
	if moe.RouterOutputFixed.Pointer() == 0 || moe.RouterOutputFixed.driver != driver ||
		moe.RouterOutputFixed.SizeBytes() < hipGemma4MoERouterMaximumOutputBytes {
		if err := moe.RouterOutputFixed.Close(); err != nil {
			return nil, err
		}
		output, err := hipAllocateByteBufferValue(
			driver,
			"rocm.hip.MoERouterLaunch",
			"packed router output workspace",
			hipGemma4MoERouterMaximumOutputBytes,
			hipGemma4MoERouterMaximumOutputBytes,
		)
		if err != nil {
			return nil, err
		}
		moe.RouterOutputFixed = output
	}
	idBytes := uint64(topK * 4)
	probBytes := uint64(topK * 4)
	totalBytes := idBytes + probBytes + 4
	moe.RouterOutputView = hipBorrowDeviceByteBufferValue(driver, "packed router output view", moe.RouterOutputFixed.Pointer(), totalBytes, int(totalBytes))
	moe.RouterIDView = hipBorrowDeviceByteBufferValue(driver, "router id output view", moe.RouterOutputView.Pointer(), idBytes, topK)
	moe.RouterProbView = hipBorrowDeviceByteBufferValue(driver, "router probability output view", moe.RouterOutputView.Pointer()+nativeDevicePointer(idBytes), probBytes, topK)
	moe.RouterStatusView = hipBorrowDeviceByteBufferValue(driver, "router status output view", moe.RouterOutputView.Pointer()+nativeDevicePointer(idBytes+probBytes), 4, 1)
	moe.RouterBuffers = hipMoERouterDeviceBuffers{
		Logits:         logits,
		Output:         &moe.RouterOutputView,
		IDs:            &moe.RouterIDView,
		Probs:          &moe.RouterProbView,
		Status:         &moe.RouterStatusView,
		ExpertCount:    logits.Count(),
		TopK:           topK,
		Layer:          layer,
		BorrowedLogits: true,
	}
	return &moe.RouterBuffers, nil
}

func (workspace *hipGemma4MoEWorkspace) resetBorrowedViews() {
	if workspace == nil {
		return
	}
	workspace.HiddenViews = [2]hipDeviceByteBuffer{}
	workspace.BatchOutputView = hipDeviceByteBuffer{}
	workspace.ExpertDownView = hipDeviceByteBuffer{}
	workspace.RouterScoresView = hipDeviceByteBuffer{}
	workspace.RouterOutputView = hipDeviceByteBuffer{}
	workspace.RouterIDView = hipDeviceByteBuffer{}
	workspace.RouterProbView = hipDeviceByteBuffer{}
	workspace.RouterStatusView = hipDeviceByteBuffer{}
	workspace.RouterBuffers = hipMoERouterDeviceBuffers{}
}

func (workspace *hipGemma4MoEWorkspace) Close() error {
	if workspace == nil {
		return nil
	}
	var lastErr error
	if err := workspace.HiddenPairFixed.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.BatchOutputFixed.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.ExpertDownFixed.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.RouterScoresFixed.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.RouterOutputFixed.Close(); err != nil {
		lastErr = err
	}
	workspace.HiddenPairFixed = hipDeviceByteBuffer{}
	workspace.HiddenPairCap = 0
	workspace.BatchOutputFixed = hipDeviceByteBuffer{}
	workspace.BatchOutputCap = 0
	workspace.ExpertDownFixed = hipDeviceByteBuffer{}
	workspace.ExpertDownCap = 0
	workspace.RouterScoresFixed = hipDeviceByteBuffer{}
	workspace.RouterScoresCap = 0
	workspace.RouterOutputFixed = hipDeviceByteBuffer{}
	workspace.resetBorrowedViews()
	return lastErr
}

func hipLoadedGemma4MoERuntimeReady(model *hipLoadedModel) bool {
	return hipLoadedGemma4MoERuntimeError(model) == nil
}

func hipLoadedGemma4MoERuntimeError(model *hipLoadedModel) error {
	if model == nil {
		return core.E(hipGemma4Q4Layer0Operation, "loaded model is required", nil)
	}
	if model.driver == nil || !model.driver.Available() {
		return core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if !model.gemma4TextConfig.EnableMoEBlock {
		return core.E(hipGemma4Q4Layer0Operation, "Gemma4 MoE block is not enabled", nil)
	}
	if model.modelInfo.NumLayers <= 0 || model.modelInfo.HiddenSize <= 0 {
		return core.E(hipGemma4Q4Layer0Operation, "model layer count and hidden size must be positive", nil)
	}
	text := model.gemma4TextConfig
	if text.NumExperts <= 0 || text.TopKExperts <= 0 || text.TopKExperts > text.NumExperts || text.MoEIntermediateSize <= 0 {
		return core.E(hipGemma4Q4Layer0Operation, "Gemma4 MoE expert geometry is invalid", nil)
	}
	for layer := 0; layer < model.modelInfo.NumLayers; layer++ {
		prefix := core.Sprintf("language_model.model.layers.%d", layer)
		if model.hasHIPTensor(prefix + ".router.proj.weight") {
			router, rows, cols, err := model.loadedGemma4Q4ProjectionConfig(prefix+".router.proj", "router.proj", model.modelInfo.QuantGroup)
			if err != nil {
				return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("load MLX affine MoE router for layer %d", layer), err)
			}
			if rows != text.NumExperts || cols != model.modelInfo.HiddenSize || !router.hasCompleteWeightStorage() {
				return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("MLX affine MoE router geometry mismatch for layer %d", layer), nil)
			}
			experts, ok, err := model.loadedGemma4MLXAffineExpertSource(prefix)
			if err != nil {
				return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("load MLX affine MoE experts for layer %d", layer), err)
			}
			if !ok {
				return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("missing host-resident MLX affine MoE expert tensors for layer %d", layer), nil)
			}
			if err := hipValidateGemma4MLXAffineExpertSource(experts, text.NumExperts, model.modelInfo.HiddenSize, text.MoEIntermediateSize, model.modelInfo.QuantGroup, model.modelInfo.QuantBits); err != nil {
				return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("validate MLX affine MoE experts for layer %d", layer), err)
			}
			continue
		}
		if !model.hipGGUFTensorAliasesEnabled() {
			return core.E(hipGemma4Q4Layer0Operation, "GGUF tensor aliases are not enabled", nil)
		}
		routerName := core.Sprintf("blk.%d.ffn_gate_inp.weight", layer)
		router, ok := model.tensors[routerName]
		if !ok {
			return core.E(hipGemma4Q4Layer0Operation, "missing MoE router tensor "+routerName, nil)
		}
		if router.pointer == 0 || core.Upper(router.info.TypeName) != "F32" || len(router.info.Dimensions) != 2 ||
			router.info.Dimensions[0] != uint64(model.modelInfo.HiddenSize) || router.info.Dimensions[1] != uint64(text.NumExperts) {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("MoE router tensor %s has unsupported type or shape: type=%s dims=%v", routerName, router.info.TypeName, router.info.Dimensions), nil)
		}
		gateUpName := core.Sprintf("blk.%d.ffn_gate_up_exps.weight", layer)
		downName := core.Sprintf("blk.%d.ffn_down_exps.weight", layer)
		gateUp, gateUpOK := model.hostTensors[gateUpName]
		down, downOK := model.hostTensors[downName]
		if !gateUpOK || !downOK {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("missing host-resident MoE expert tensors for layer %d: gate_up=%t down=%t", layer, gateUpOK, downOK), nil)
		}
		if !hipGemma4ExpertFormatPairSupported(gateUp, down) {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("unsupported MoE expert tensor formats for layer %d: gate_up=%s down=%s", layer, gateUp.TypeName, down.TypeName), nil)
		}
		if len(gateUp.Dimensions) != 3 || len(down.Dimensions) != 3 ||
			gateUp.Dimensions[0] != uint64(model.modelInfo.HiddenSize) || gateUp.Dimensions[1] != uint64(2*text.MoEIntermediateSize) || gateUp.Dimensions[2] != uint64(text.NumExperts) ||
			down.Dimensions[0] != uint64(text.MoEIntermediateSize) || down.Dimensions[1] != uint64(model.modelInfo.HiddenSize) || down.Dimensions[2] != uint64(text.NumExperts) {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("unsupported MoE expert tensor shapes for layer %d: gate_up=%v down=%v", layer, gateUp.Dimensions, down.Dimensions), nil)
		}
	}
	return nil
}

func hipGemma4ExpertFormatPairSupported(gateUp, down nativeTensorInfo) bool {
	return (hipNativeTensorInfoIsGGUFQ4_0(gateUp) && hipNativeTensorInfoIsGGUFQ4_0(down)) ||
		(hipNativeTensorInfoIsGGUFQ4K(gateUp) && (hipNativeTensorInfoIsGGUFQ5_1(down) || hipNativeTensorInfoIsGGUFQ8_0(down)))
}

func hipGGUFExpertFormatForInfo(info nativeTensorInfo) (uint32, bool) {
	switch {
	case hipNativeTensorInfoIsGGUFQ4_0(info):
		return hipGGUFExpertFormatQ4_0, true
	case hipNativeTensorInfoIsGGUFQ4K(info):
		return hipGGUFExpertFormatQ4K, true
	case hipNativeTensorInfoIsGGUFQ5_1(info):
		return hipGGUFExpertFormatQ5_1, true
	case hipNativeTensorInfoIsGGUFQ8_0(info):
		return hipGGUFExpertFormatQ8_0, true
	default:
		return 0, false
	}
}

func hipGGUFExpertBlockGeometry(format uint32) (int, uint64, bool) {
	switch format {
	case hipGGUFExpertFormatQ4_0:
		return hipGGUFQ4_0BlockSize, hipGGUFQ4_0BlockBytes, true
	case hipGGUFExpertFormatQ4K:
		return hipGGUFQ4KBlockSize, hipGGUFQ4KBlockBytes, true
	case hipGGUFExpertFormatQ4KExpanded:
		return hipGGUFQ4KBlockSize, hipGGUFQ4KExpandedBlockBytes, true
	case hipGGUFExpertFormatQ5_1:
		return hipGGUFQ5_1BlockSize, hipGGUFQ5_1BlockBytes, true
	case hipGGUFExpertFormatQ8_0:
		return hipGGUFQ8_0BlockSize, hipGGUFQ8_0BlockBytes, true
	default:
		return 0, 0, false
	}
}

func hipGGUFEffectiveExpertFormat(format uint32) uint32 {
	if format == 0 {
		return hipGGUFExpertFormatQ4_0
	}
	return format
}

func (model *hipLoadedModel) loadedGemma4MoELayerConfig(layer, hidden int) (*hipGemma4MoELayerConfig, error) {
	if model == nil || !model.gemma4TextConfig.EnableMoEBlock {
		return nil, nil
	}
	text := model.gemma4TextConfig
	if layer < 0 || hidden <= 0 || text.NumExperts <= 0 || text.TopKExperts <= 0 ||
		text.TopKExperts > text.NumExperts || text.MoEIntermediateSize <= 0 {
		return nil, core.E("rocm.hip.Gemma4MoEConfig", "Gemma4 MoE geometry is invalid", nil)
	}
	prefix := core.Sprintf("language_model.model.layers.%d", layer)
	pre2, err := model.loadedGemma4NormConfig(prefix+".pre_feedforward_layernorm_2.weight", "pre_feedforward_layernorm_2", hidden)
	if err != nil {
		return nil, err
	}
	post1, err := model.loadedGemma4NormConfig(prefix+".post_feedforward_layernorm_1.weight", "post_feedforward_layernorm_1", hidden)
	if err != nil {
		return nil, err
	}
	post2, err := model.loadedGemma4NormConfig(prefix+".post_feedforward_layernorm_2.weight", "post_feedforward_layernorm_2", hidden)
	if err != nil {
		return nil, err
	}
	routerNorm, err := model.loadedGemma4ScaledNormConfig(prefix+".router.scale", "router scale", hidden, float32(1/math.Sqrt(float64(hidden))))
	if err != nil {
		return nil, err
	}
	var router hipGemma4MoERouterProjectionConfig
	var routerMLX hipMLXQ4DeviceWeightConfig
	if model.hasHIPTensor(prefix + ".router.proj.weight") {
		var routerRows, routerCols int
		routerMLX, routerRows, routerCols, err = model.loadedGemma4Q4ProjectionConfig(prefix+".router.proj", "router.proj", model.modelInfo.QuantGroup)
		if err != nil {
			return nil, err
		}
		if routerRows != text.NumExperts || routerCols != hidden {
			return nil, core.E("rocm.hip.Gemma4MoEConfig", "MLX affine router projection geometry is invalid", nil)
		}
	} else {
		router, err = model.loadedGemma4MoERouterProjectionConfig(layer, hidden, text.NumExperts)
		if err != nil {
			return nil, err
		}
	}
	perExpertScale, err := model.loadedGemma4Float32Vector(prefix+".router.per_expert_scale", "per-expert scale", text.NumExperts)
	if err != nil {
		return nil, err
	}
	storage := hipGemma4MoEExpertStorageGGUF
	mlxExperts, mlxExpertsOK, err := model.loadedGemma4MLXAffineExpertSource(prefix)
	if err != nil {
		return nil, err
	}
	var gateUp, down nativeTensorInfo
	if mlxExpertsOK {
		storage = hipGemma4MoEExpertStorageMLXAffine
	} else {
		var ok bool
		gateUp, ok = model.hostTensors[core.Sprintf("blk.%d.ffn_gate_up_exps.weight", layer)]
		if !ok {
			return nil, core.E("rocm.hip.Gemma4MoEConfig", "expert gate/up tensor is required", nil)
		}
		down, ok = model.hostTensors[core.Sprintf("blk.%d.ffn_down_exps.weight", layer)]
		if !ok {
			return nil, core.E("rocm.hip.Gemma4MoEConfig", "expert down tensor is required", nil)
		}
	}
	model.expertCacheMu.Lock()
	if model.expertCache == nil {
		model.expertCache = newHIPGemma4AdaptiveExpertCache(model.driver, text.TopKExperts)
	}
	cache := model.expertCache
	model.expertCacheMu.Unlock()
	cfg := &hipGemma4MoELayerConfig{
		Layer:                  layer,
		NumExperts:             text.NumExperts,
		TopKExperts:            text.TopKExperts,
		ExpertIntermediateSize: text.MoEIntermediateSize,
		PreFeedForwardNorm2:    pre2,
		PostFeedForwardNorm1:   post1,
		PostFeedForwardNorm2:   post2,
		RouterNorm:             routerNorm,
		RouterProjection:       router,
		RouterProjectionMLX:    routerMLX,
		PerExpertScale:         perExpertScale,
		ExpertCache:            cache,
		ExpertStorage:          storage,
		GateUpInfo:             gateUp,
		DownInfo:               down,
		MLXExperts:             mlxExperts,
		MLXHiddenSize:          hidden,
		MLXGroupSize:           model.modelInfo.QuantGroup,
		MLXPreferredBits:       model.modelInfo.QuantBits,
	}
	if err := cfg.validate(hidden); err != nil {
		return nil, err
	}
	return cfg, nil
}

func (model *hipLoadedModel) loadedGemma4MLXAffineExpertSource(layerPrefix string) (hipGemma4MLXAffineExpertSource, bool, error) {
	for _, expertPrefix := range []string{
		layerPrefix + ".experts",
		layerPrefix + ".experts.switch_glu",
	} {
		gateUpBase := expertPrefix + ".gate_up_proj"
		downBase := expertPrefix + ".down_proj"
		names := []string{
			gateUpBase + ".weight", gateUpBase + ".scales", gateUpBase + ".biases",
			downBase + ".weight", downBase + ".scales", downBase + ".biases",
		}
		found := 0
		for _, name := range names {
			if _, ok := model.hostTensors[name]; ok {
				found++
			}
		}
		if found == 0 {
			continue
		}
		if found != len(names) {
			return hipGemma4MLXAffineExpertSource{}, false, core.E("rocm.hip.Gemma4MoEConfig", "MLX affine expert tensor set is incomplete", nil)
		}
		return hipGemma4MLXAffineExpertSource{
			GateUp: hipGemma4MLXAffineTensorSet{
				Weight: model.hostTensors[names[0]], Scales: model.hostTensors[names[1]], Biases: model.hostTensors[names[2]],
			},
			Down: hipGemma4MLXAffineTensorSet{
				Weight: model.hostTensors[names[3]], Scales: model.hostTensors[names[4]], Biases: model.hostTensors[names[5]],
			},
		}, true, nil
	}
	return hipGemma4MLXAffineExpertSource{}, false, nil
}

func (model *hipLoadedModel) loadedGemma4ScaledNormConfig(name, label string, count int, scale float32) (hipRMSNormDeviceWeightConfig, error) {
	if scale <= 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return hipRMSNormDeviceWeightConfig{}, core.E("rocm.hip.Gemma4MoEConfig", label+" scale must be positive and finite", nil)
	}
	syntheticName := name + ".root_scaled"
	if !model.hasHIPTensor(syntheticName) {
		values, err := model.loadedGemma4Float32Vector(name, label, count)
		if err != nil {
			return hipRMSNormDeviceWeightConfig{}, err
		}
		for index := range values {
			values[index] *= scale
		}
		payload, err := hipFloat32Payload(values)
		if err != nil {
			return hipRMSNormDeviceWeightConfig{}, core.E("rocm.hip.Gemma4MoEConfig", "encode scaled "+label, err)
		}
		if err := model.uploadSyntheticHIPTensor(syntheticName, hipNativeTensorTypeF32, "F32", []uint64{uint64(count)}, payload); err != nil {
			return hipRMSNormDeviceWeightConfig{}, err
		}
	}
	return model.loadedGemma4NormConfig(syntheticName, label, count)
}

func (model *hipLoadedModel) loadedGemma4Float32Vector(name, label string, count int) ([]float32, error) {
	tensor, err := model.requiredHIPTensor(name, label)
	if err != nil {
		return nil, err
	}
	encoding, wantBytes, err := hipGemma4NormWeightEncodingAndBytes(tensor.info, label, count)
	if err != nil {
		return nil, err
	}
	if len(tensor.info.Dimensions) != 1 || tensor.info.Dimensions[0] != uint64(count) || tensor.info.ByteSize != wantBytes {
		return nil, core.E("rocm.hip.Gemma4MoEConfig", label+" tensor shape/type mismatch", nil)
	}
	payload := make([]byte, int(wantBytes))
	if err := model.driver.CopyDeviceToHost(tensor.pointer, payload); err != nil {
		return nil, core.E("rocm.hip.Gemma4MoEConfig", "copy "+label, err)
	}
	values := make([]float32, count)
	for index := range values {
		if encoding == hipRMSNormWeightEncodingF32 {
			values[index] = math.Float32frombits(binary.LittleEndian.Uint32(payload[index*4:]))
		} else {
			values[index] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload[index*2:]))
		}
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E("rocm.hip.Gemma4MoEConfig", label+" values must be finite", nil)
	}
	return values, nil
}

func (model *hipLoadedModel) loadedGemma4MoERouterProjectionConfig(layer, hidden, experts int) (hipGemma4MoERouterProjectionConfig, error) {
	name := core.Sprintf("blk.%d.ffn_gate_inp.weight", layer)
	tensor, ok := model.tensors[name]
	if !ok {
		return hipGemma4MoERouterProjectionConfig{}, core.E("rocm.hip.Gemma4MoEConfig", "F32 GGUF router weight tensor is required", nil)
	}
	if tensor.pointer == 0 || core.Upper(tensor.info.TypeName) != "F32" || len(tensor.info.Dimensions) != 2 ||
		tensor.info.Dimensions[0] != uint64(hidden) || tensor.info.Dimensions[1] != uint64(experts) ||
		tensor.info.ByteSize != uint64(hidden)*uint64(experts)*4 {
		return hipGemma4MoERouterProjectionConfig{}, core.E("rocm.hip.Gemma4MoEConfig", "F32 GGUF router weight tensor shape/type mismatch", nil)
	}
	return hipGemma4MoERouterProjectionConfig{
		WeightPointer: tensor.pointer,
		WeightBytes:   tensor.info.ByteSize,
		Rows:          experts,
		Cols:          hidden,
	}, nil
}

func (cfg *hipGemma4MoELayerConfig) validate(hidden int) error {
	if cfg == nil {
		return nil
	}
	if cfg.Layer < 0 || hidden <= 0 || cfg.NumExperts <= 0 || cfg.TopKExperts <= 0 ||
		cfg.TopKExperts > cfg.NumExperts || cfg.ExpertIntermediateSize <= 0 {
		return core.E("rocm.hip.Gemma4MoEConfig", "MoE layer geometry is invalid", nil)
	}
	for label, norm := range map[string]hipRMSNormDeviceWeightConfig{
		"pre_feedforward_layernorm_2":  cfg.PreFeedForwardNorm2,
		"post_feedforward_layernorm_1": cfg.PostFeedForwardNorm1,
		"post_feedforward_layernorm_2": cfg.PostFeedForwardNorm2,
		"router scale":                 cfg.RouterNorm,
	} {
		if err := hipValidateGemma4Q4NormConfig(label, norm, hidden); err != nil {
			return err
		}
	}
	hasGGUFRouter := cfg.RouterProjection.WeightPointer != 0
	hasMLXRouter := cfg.RouterProjectionMLX.hasCompleteWeightStorage()
	if hasGGUFRouter == hasMLXRouter {
		return core.E("rocm.hip.Gemma4MoEConfig", "exactly one router projection format is required", nil)
	}
	if hasGGUFRouter && (cfg.RouterProjection.Rows != cfg.NumExperts || cfg.RouterProjection.Cols != hidden ||
		cfg.RouterProjection.WeightBytes != uint64(cfg.NumExperts)*uint64(hidden)*4) {
		return core.E("rocm.hip.Gemma4MoEConfig", "router projection geometry is invalid", nil)
	}
	if hasMLXRouter {
		if cfg.RouterProjectionMLX.Rows != cfg.NumExperts || cfg.RouterProjectionMLX.Cols != hidden {
			return core.E("rocm.hip.Gemma4MoEConfig", "MLX affine router projection geometry is invalid", nil)
		}
		if err := cfg.RouterProjectionMLX.validateInputCount(hidden); err != nil {
			return core.E("rocm.hip.Gemma4MoEConfig", "MLX affine router projection config", err)
		}
	}
	if len(cfg.PerExpertScale) != cfg.NumExperts || !rocmFloat32SliceFinite(cfg.PerExpertScale) {
		return core.E("rocm.hip.Gemma4MoEConfig", "per-expert scale must match the expert count", nil)
	}
	if cfg.ExpertCache == nil {
		return core.E("rocm.hip.Gemma4MoEConfig", "expert cache is required", nil)
	}
	if cfg.ExpertStorage == hipGemma4MoEExpertStorageMLXAffine {
		if cfg.MLXHiddenSize != hidden {
			return core.E("rocm.hip.Gemma4MoEConfig", "MLX affine expert hidden size mismatch", nil)
		}
		if err := hipValidateGemma4MLXAffineExpertSource(cfg.MLXExperts, cfg.NumExperts, hidden, cfg.ExpertIntermediateSize, cfg.MLXGroupSize, cfg.MLXPreferredBits); err != nil {
			return err
		}
	} else {
		if !hipGemma4ExpertFormatPairSupported(cfg.GateUpInfo, cfg.DownInfo) || len(cfg.GateUpInfo.Dimensions) != 3 ||
			cfg.GateUpInfo.Dimensions[0] != uint64(hidden) || cfg.GateUpInfo.Dimensions[1] != uint64(2*cfg.ExpertIntermediateSize) ||
			cfg.GateUpInfo.Dimensions[2] != uint64(cfg.NumExperts) {
			return core.E("rocm.hip.Gemma4MoEConfig", "expert gate/up tensor geometry is invalid", nil)
		}
		if len(cfg.DownInfo.Dimensions) != 3 ||
			cfg.DownInfo.Dimensions[0] != uint64(cfg.ExpertIntermediateSize) || cfg.DownInfo.Dimensions[1] != uint64(hidden) ||
			cfg.DownInfo.Dimensions[2] != uint64(cfg.NumExperts) {
			return core.E("rocm.hip.Gemma4MoEConfig", "expert down tensor geometry is invalid", nil)
		}
	}
	return nil
}

func hipValidateGemma4MLXAffineExpertSource(source hipGemma4MLXAffineExpertSource, experts, hidden, expertFF, groupSize, preferredBits int) error {
	gateWeight, err := hipGemma4MLXAffineExpertRank2Tensor(source.GateUp.Weight, experts, "U32", "expert gate/up weight")
	if err != nil {
		return err
	}
	gateScales, err := hipGemma4MLXAffineExpertRank2Tensor(source.GateUp.Scales, experts, "BF16", "expert gate/up scales")
	if err != nil {
		return err
	}
	gateBiases, err := hipGemma4MLXAffineExpertRank2Tensor(source.GateUp.Biases, experts, "BF16", "expert gate/up biases")
	if err != nil {
		return err
	}
	_, gateRows, gateCols, _, _, _, err := hipInferMLXAffineBitsFromTensorShapes(gateWeight, gateScales, gateBiases, groupSize, preferredBits, "expert gate/up")
	if err != nil {
		return err
	}
	if gateRows != 2*expertFF || gateCols != hidden {
		return core.E("rocm.hip.Gemma4MoEConfig", "MLX affine expert gate/up geometry is invalid", nil)
	}
	downWeight, err := hipGemma4MLXAffineExpertRank2Tensor(source.Down.Weight, experts, "U32", "expert down weight")
	if err != nil {
		return err
	}
	downScales, err := hipGemma4MLXAffineExpertRank2Tensor(source.Down.Scales, experts, "BF16", "expert down scales")
	if err != nil {
		return err
	}
	downBiases, err := hipGemma4MLXAffineExpertRank2Tensor(source.Down.Biases, experts, "BF16", "expert down biases")
	if err != nil {
		return err
	}
	_, downRows, downCols, _, _, _, err := hipInferMLXAffineBitsFromTensorShapes(downWeight, downScales, downBiases, groupSize, preferredBits, "expert down")
	if err != nil {
		return err
	}
	if downRows != hidden || downCols != expertFF {
		return core.E("rocm.hip.Gemma4MoEConfig", "MLX affine expert down geometry is invalid", nil)
	}
	return nil
}

func hipGemma4MLXAffineExpertRank2Tensor(info nativeTensorInfo, experts int, expectedType, label string) (hipTensor, error) {
	if experts <= 0 || core.Upper(info.TypeName) != expectedType || len(info.Dimensions) != 3 || info.Dimensions[0] != uint64(experts) {
		return hipTensor{}, core.E("rocm.hip.Gemma4MoEConfig", label+" type or shape mismatch", nil)
	}
	elementBytes := uint64(2)
	if expectedType == "U32" {
		elementBytes = 4
	}
	rows, cols := info.Dimensions[1], info.Dimensions[2]
	if rows == 0 || cols == 0 || rows > ^uint64(0)/cols || rows*cols > ^uint64(0)/elementBytes {
		return hipTensor{}, core.E("rocm.hip.Gemma4MoEConfig", label+" geometry overflows", nil)
	}
	perExpertBytes := rows * cols * elementBytes
	if uint64(experts) > ^uint64(0)/perExpertBytes || info.ByteSize != uint64(experts)*perExpertBytes || info.SourcePath == "" {
		return hipTensor{}, core.E("rocm.hip.Gemma4MoEConfig", label+" byte count or source path mismatch", nil)
	}
	return hipTensor{info: nativeTensorInfo{
		TypeName: expectedType, Dimensions: []uint64{rows, cols}, ByteSize: perExpertBytes,
	}}, nil
}

func (cfg *hipGemma4MoELayerConfig) expertEntry(expert int) (*hipGemma4ExpertCacheEntry, error) {
	if cfg == nil || cfg.ExpertCache == nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "MoE expert cache is required", nil)
	}
	key := hipGemma4ExpertCacheKey{Layer: cfg.Layer, Expert: expert}
	if cfg.ExpertStorage == hipGemma4MoEExpertStorageMLXAffine {
		return cfg.ExpertCache.entryMLXAffine(key, cfg.MLXExperts, cfg.NumExperts, cfg.MLXHiddenSize, cfg.ExpertIntermediateSize, cfg.MLXGroupSize, cfg.MLXPreferredBits)
	}
	return cfg.ExpertCache.entry(key, cfg.GateUpInfo, cfg.DownInfo, cfg.NumExperts)
}

func hipRunGemma4MoEDeviceMLP(ctx context.Context, driver nativeHIPDriver, attentionResidual, localInput *hipDeviceByteBuffer, layer hipGemma4Q4Layer0Config, epsilon float32) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4MoEDeviceMLPWithWorkspace(ctx, driver, attentionResidual, localInput, layer, epsilon, nil)
}

func hipRunGemma4MoEDeviceMLPWithWorkspace(ctx context.Context, driver nativeHIPDriver, attentionResidual, localInput *hipDeviceByteBuffer, layer hipGemma4Q4Layer0Config, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return hipRunGemma4MoEDeviceMLPAllocated(ctx, driver, attentionResidual, localInput, layer, epsilon)
	}
	return hipRunGemma4MoEDeviceMLPWithWorkspaceOutput(ctx, driver, attentionResidual, localInput, layer, epsilon, workspace, nil)
}

func hipRunGemma4MoERouterProjectionWithOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, moe *hipGemma4MoELayerConfig, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if moe == nil {
		return core.E("rocm.hip.Gemma4MoE", "MoE layer config is required", nil)
	}
	if moe.RouterProjectionMLX.hasCompleteWeightStorage() {
		return hipRunMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace(ctx, driver, input, moe.RouterProjectionMLX, output, workspace)
	}
	return hipRunProjectionKernelWithDeviceInputWeightEncodingOutputWithWorkspace(
		ctx, driver, input,
		moe.RouterProjection.WeightPointer, moe.RouterProjection.WeightBytes,
		moe.RouterProjection.Rows, moe.RouterProjection.Cols,
		hipProjectionWeightEncodingF32, output, workspace,
	)
}

func hipRunGemma4MoEDeviceMLPWithWorkspaceOutput(ctx context.Context, driver nativeHIPDriver, attentionResidual, localInput *hipDeviceByteBuffer, layer hipGemma4Q4Layer0Config, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace, output *hipDeviceByteBuffer) (*hipDeviceByteBuffer, error) {
	moe := layer.MoE
	if moe == nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "MoE layer config is required", nil)
	}
	if err := moe.validate(layer.HiddenSize); err != nil {
		return nil, err
	}
	if attentionResidual == nil || localInput == nil || attentionResidual.Count() != layer.HiddenSize || localInput.Count() != layer.HiddenSize {
		return nil, core.E("rocm.hip.Gemma4MoE", "attention residual and local input must match the hidden size", nil)
	}

	localOutput, err := workspace.EnsureProjectionOutput(driver, layer.HiddenSize)
	if err != nil {
		return nil, err
	}
	if err := hipRunGemma4Q4DeviceGELUTanhMLPWithDeviceInputOutput(
		ctx, driver, localInput, layer.GateProjection, layer.UpProjection, layer.DownProjection, localOutput, workspace,
	); err != nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "run local MLP branch", err)
	}

	expertInput, err := workspace.EnsureMoEHiddenOutput(driver, layer.HiddenSize, 0)
	if err != nil {
		return nil, err
	}
	expertNormCfg := moe.PreFeedForwardNorm2
	expertNormCfg.Epsilon = epsilon
	if err := hipRunRMSNormDeviceToDeviceKernelWithWorkspace(
		ctx, driver, attentionResidual.Pointer(), attentionResidual.SizeBytes(), expertInput.Pointer(), expertInput.SizeBytes(), expertNormCfg, workspace,
	); err != nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "run expert input norm", err)
	}

	routerInput, err := workspace.EnsureMoEHiddenOutput(driver, layer.HiddenSize, 1)
	if err != nil {
		return nil, err
	}
	routerNormCfg := moe.RouterNorm
	routerNormCfg.Epsilon = epsilon
	if err := hipRunRMSNormDeviceToDeviceKernelWithWorkspace(
		ctx, driver, attentionResidual.Pointer(), attentionResidual.SizeBytes(), routerInput.Pointer(), routerInput.SizeBytes(), routerNormCfg, workspace,
	); err != nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "run router norm", err)
	}
	routerScores, err := workspace.EnsureMoERouterScores(driver, moe.NumExperts)
	if err != nil {
		return nil, err
	}
	if err := hipRunGemma4MoERouterProjectionWithOutput(ctx, driver, routerInput, moe, routerScores, workspace); err != nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "run router projection", err)
	}
	routed, err := hipRunMoERouterKernelWithDeviceInputWorkspace(ctx, driver, routerScores, moe.TopKExperts, moe.Layer, workspace)
	if err != nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "select experts", err)
	}

	entries := workspace.MoE.Entries[:len(routed.Routes)]
	routeWeights := workspace.MoE.RouteWeights[:len(routed.Routes)]
	for index, route := range routed.Routes {
		entry, entryErr := moe.expertEntry(route.ID)
		if entryErr != nil {
			return nil, core.E("rocm.hip.Gemma4MoE", core.Sprintf("load expert %d", route.ID), entryErr)
		}
		entries[index] = entry
		routeWeights[index] = route.Prob * moe.PerExpertScale[route.ID]
	}
	expertOutput, err := workspace.EnsureMoEHiddenOutput(driver, layer.HiddenSize, 1)
	if err != nil {
		return nil, err
	}
	activationCount := len(entries) * moe.ExpertIntermediateSize
	if moe.ExpertStorage == hipGemma4MoEExpertStorageMLXAffine {
		activationCount = moe.ExpertIntermediateSize
	}
	activation, err := workspace.EnsureActivationOutput(driver, activationCount)
	if err != nil {
		return nil, err
	}
	if moe.ExpertStorage == hipGemma4MoEExpertStorageMLXAffine {
		downOutput, downErr := workspace.EnsureMoEExpertDownOutput(driver, layer.HiddenSize)
		if downErr != nil {
			return nil, downErr
		}
		err = hipRunGemma4MLXAffineSelectedExpertsWithWorkspace(
			ctx, driver, expertInput, entries, routeWeights,
			layer.HiddenSize, moe.ExpertIntermediateSize, activation, downOutput, expertOutput, workspace,
		)
	} else {
		err = hipRunGGUFQ4_0SelectedExpertsKernelWithDeviceInputOutputWithWorkspace(
			ctx, driver, expertInput, entries, routeWeights,
			layer.HiddenSize, moe.ExpertIntermediateSize, activation, expertOutput, workspace,
		)
	}
	if err != nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "run selected experts", err)
	}

	combined := output
	if combined == nil {
		combined, err = workspace.EnsureActivationOutput(driver, layer.HiddenSize)
		if err != nil {
			return nil, err
		}
	}
	if combined.Pointer() == 0 || combined.Count() != layer.HiddenSize || combined.SizeBytes() != uint64(layer.HiddenSize*4) {
		return nil, core.E("rocm.hip.Gemma4MoE", "combined output must match the hidden size", nil)
	}
	localPostCfg := moe.PostFeedForwardNorm1
	localPostCfg.Epsilon = epsilon
	expertPostCfg := moe.PostFeedForwardNorm2
	expertPostCfg.Epsilon = epsilon
	if err := hipRunMoECombineNormsDeviceKernelOutputWithWorkspace(ctx, driver, localOutput, expertOutput, localPostCfg, expertPostCfg, combined, workspace); err != nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "combine local and expert branches", err)
	}
	return combined, nil
}

func hipRunGemma4MoEDeviceMLPBatchWithWorkspace(ctx context.Context, driver nativeHIPDriver, attentionResidual, localInput *hipDeviceByteBuffer, layer hipGemma4Q4Layer0Config, epsilon float32, tokenCount int, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, error) {
	if tokenCount <= 0 || layer.HiddenSize <= 0 {
		return nil, core.E("rocm.hip.Gemma4MoE", "MoE batch geometry must be positive", nil)
	}
	wantCount := tokenCount * layer.HiddenSize
	if attentionResidual == nil || localInput == nil || attentionResidual.Pointer() == 0 || localInput.Pointer() == 0 ||
		attentionResidual.Count() != wantCount || localInput.Count() != wantCount ||
		attentionResidual.SizeBytes() != uint64(wantCount*4) || localInput.SizeBytes() != uint64(wantCount*4) {
		return nil, core.E("rocm.hip.Gemma4MoE", "MoE batch inputs must match token count and hidden size", nil)
	}
	output, err := workspace.EnsureMoEBatchOutput(driver, wantCount)
	if err != nil {
		return nil, err
	}
	rowBytes := uint64(layer.HiddenSize * 4)
	for token := 0; token < tokenCount; token++ {
		offset := nativeDevicePointer(uint64(token) * rowBytes)
		attentionRow := hipBorrowDeviceByteBufferValue(driver, "MoE batch attention residual row", attentionResidual.Pointer()+offset, rowBytes, layer.HiddenSize)
		localRow := hipBorrowDeviceByteBufferValue(driver, "MoE batch local input row", localInput.Pointer()+offset, rowBytes, layer.HiddenSize)
		outputRow := hipBorrowDeviceByteBufferValue(driver, "MoE batch output row", output.Pointer()+offset, rowBytes, layer.HiddenSize)
		if _, err := hipRunGemma4MoEDeviceMLPWithWorkspaceOutput(ctx, driver, &attentionRow, &localRow, layer, epsilon, workspace, &outputRow); err != nil {
			return nil, core.E("rocm.hip.Gemma4MoE", core.Sprintf("run batch row %d", token), err)
		}
	}
	return output, nil
}

func hipRunGemma4MoEDeviceMLPAllocated(ctx context.Context, driver nativeHIPDriver, attentionResidual, localInput *hipDeviceByteBuffer, layer hipGemma4Q4Layer0Config, epsilon float32) (*hipDeviceByteBuffer, error) {
	moe := layer.MoE
	if moe == nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "MoE layer config is required", nil)
	}
	if err := moe.validate(layer.HiddenSize); err != nil {
		return nil, err
	}
	if attentionResidual == nil || localInput == nil || attentionResidual.Count() != layer.HiddenSize || localInput.Count() != layer.HiddenSize {
		return nil, core.E("rocm.hip.Gemma4MoE", "attention residual and local input must match the hidden size", nil)
	}

	localOutput, err := hipRunGemma4Q4DeviceGELUTanhMLPWithDeviceInput(ctx, driver, localInput, layer.GateProjection, layer.UpProjection, layer.DownProjection)
	if err != nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "run local MLP branch", err)
	}
	defer localOutput.Close()

	expertNormCfg := moe.PreFeedForwardNorm2
	expertNormCfg.Epsilon = epsilon
	expertInput, err := hipRunRMSNormKernelWithDeviceInputWeightConfig(ctx, driver, attentionResidual, expertNormCfg)
	if err != nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "run expert input norm", err)
	}
	defer expertInput.Close()

	routerNormCfg := moe.RouterNorm
	routerNormCfg.Epsilon = epsilon
	routerInput, err := hipRunRMSNormKernelWithDeviceInputWeightConfig(ctx, driver, attentionResidual, routerNormCfg)
	if err != nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "run router norm", err)
	}
	defer routerInput.Close()
	routerScores, err := hipAllocateByteBuffer(driver, "rocm.hip.Gemma4MoE", "router scores", uint64(moe.NumExperts*4), moe.NumExperts)
	if err != nil {
		return nil, err
	}
	defer routerScores.Close()
	if err := hipRunGemma4MoERouterProjectionWithOutput(ctx, driver, routerInput, moe, routerScores, nil); err != nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "run router projection", err)
	}
	routed, err := hipRunMoERouterKernelWithDeviceInput(ctx, driver, routerScores, moe.TopKExperts, moe.Layer)
	if err != nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "select experts", err)
	}

	entries := make([]*hipGemma4ExpertCacheEntry, len(routed.Routes))
	routeWeights := make([]float32, len(routed.Routes))
	for index, route := range routed.Routes {
		entry, err := moe.expertEntry(route.ID)
		if err != nil {
			return nil, core.E("rocm.hip.Gemma4MoE", core.Sprintf("load expert %d", route.ID), err)
		}
		entries[index] = entry
		routeWeights[index] = route.Prob * moe.PerExpertScale[route.ID]
	}
	expertOutput, err := hipAllocateByteBuffer(driver, "rocm.hip.Gemma4MoE", "expert branch output", uint64(layer.HiddenSize*4), layer.HiddenSize)
	if err != nil {
		return nil, err
	}
	defer expertOutput.Close()
	activationCount := len(entries) * moe.ExpertIntermediateSize
	if moe.ExpertStorage == hipGemma4MoEExpertStorageMLXAffine {
		activationCount = moe.ExpertIntermediateSize
	}
	activation, err := hipAllocateByteBuffer(driver, "rocm.hip.Gemma4MoE", "selected expert activation", uint64(activationCount*4), activationCount)
	if err != nil {
		return nil, err
	}
	defer activation.Close()
	if moe.ExpertStorage == hipGemma4MoEExpertStorageMLXAffine {
		downOutput, downErr := hipAllocateByteBuffer(driver, "rocm.hip.Gemma4MoE", "selected expert down output", uint64(layer.HiddenSize*4), layer.HiddenSize)
		if downErr != nil {
			return nil, downErr
		}
		err = hipRunGemma4MLXAffineSelectedExpertsWithWorkspace(
			ctx, driver, expertInput, entries, routeWeights,
			layer.HiddenSize, moe.ExpertIntermediateSize, activation, downOutput, expertOutput, nil,
		)
		_ = downOutput.Close()
	} else {
		err = hipRunGGUFQ4_0SelectedExpertsKernelWithDeviceInputOutput(
			ctx, driver, expertInput, entries, routeWeights,
			layer.HiddenSize, moe.ExpertIntermediateSize, activation, expertOutput,
		)
	}
	if err != nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "run selected experts", err)
	}

	localPostCfg := moe.PostFeedForwardNorm1
	localPostCfg.Epsilon = epsilon
	localPost, err := hipRunRMSNormKernelWithDeviceInputWeightConfig(ctx, driver, localOutput, localPostCfg)
	if err != nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "run local branch output norm", err)
	}
	defer localPost.Close()
	expertPostCfg := moe.PostFeedForwardNorm2
	expertPostCfg.Epsilon = epsilon
	expertPost, err := hipRunRMSNormKernelWithDeviceInputWeightConfig(ctx, driver, expertOutput, expertPostCfg)
	if err != nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "run expert branch output norm", err)
	}
	defer expertPost.Close()
	combined, err := hipRunVectorAddScaledDeviceKernel(ctx, driver, localPost, expertPost, 1)
	if err != nil {
		return nil, core.E("rocm.hip.Gemma4MoE", "combine local and expert branches", err)
	}
	return combined, nil
}

func newHIPGemma4ExpertCache(driver nativeHIPDriver, maxBytes uint64) *hipGemma4ExpertCache {
	return &hipGemma4ExpertCache{
		driver:   driver,
		maxBytes: maxBytes,
		entries:  map[hipGemma4ExpertCacheKey]*hipGemma4ExpertCacheEntry{},
		sources:  map[string]*hipGemma4MappedExpertSource{},
	}
}

func newHIPGemma4AdaptiveExpertCache(driver nativeHIPDriver, minimumEntries int) *hipGemma4ExpertCache {
	if minimumEntries < 1 {
		minimumEntries = 1
	}
	cache := newHIPGemma4ExpertCache(driver, hipGemma4ExpertCacheBudget(driver))
	cache.adaptive = true
	cache.minimumEntries = minimumEntries
	cache.releaseTransientPoolSuppression = hipSuppressDeviceByteBufferPool()
	return cache
}

func hipGemma4ExpertCacheBudget(driver nativeHIPDriver) uint64 {
	if driver == nil {
		return hipGemma4ExpertCacheDefaultBytes
	}
	freeBytes := driver.DeviceInfo().FreeBytes
	if freeBytes == 0 {
		return hipGemma4ExpertCacheDefaultBytes
	}
	if freeBytes <= hipGemma4ExpertCacheReserveBytes {
		return freeBytes / 2
	}
	budget := freeBytes - hipGemma4ExpertCacheReserveBytes
	if budget > hipGemma4ExpertCacheMaximumBytes {
		return hipGemma4ExpertCacheMaximumBytes
	}
	return budget
}

func (model *hipLoadedModel) gemma4ExpertCacheEntry(layer, expert int) (*hipGemma4ExpertCacheEntry, error) {
	if model == nil || model.driver == nil {
		return nil, core.E("rocm.hip.Gemma4ExpertCache", "loaded model is required", nil)
	}
	model.expertCacheMu.Lock()
	if model.expertCache == nil {
		model.expertCache = newHIPGemma4AdaptiveExpertCache(model.driver, model.gemma4TextConfig.TopKExperts)
	}
	cache := model.expertCache
	model.expertCacheMu.Unlock()
	gateUp, ok := model.hostTensors[core.Sprintf("blk.%d.ffn_gate_up_exps.weight", layer)]
	if !ok {
		return nil, core.E("rocm.hip.Gemma4ExpertCache", "expert gate/up tensor is required", nil)
	}
	down, ok := model.hostTensors[core.Sprintf("blk.%d.ffn_down_exps.weight", layer)]
	if !ok {
		return nil, core.E("rocm.hip.Gemma4ExpertCache", "expert down tensor is required", nil)
	}
	return cache.entry(hipGemma4ExpertCacheKey{Layer: layer, Expert: expert}, gateUp, down, model.gemma4TextConfig.NumExperts)
}

func (cache *hipGemma4ExpertCache) entry(key hipGemma4ExpertCacheKey, gateUpInfo, downInfo nativeTensorInfo, expectedExperts int) (*hipGemma4ExpertCacheEntry, error) {
	if cache == nil || cache.driver == nil || !cache.driver.Available() {
		return nil, core.E("rocm.hip.Gemma4ExpertCache", "HIP driver is not available", nil)
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	cache.clock++
	if entry := cache.entries[key]; entry != nil {
		cache.stats.Hits++
		entry.lastUse = cache.clock
		if cache.clock%hipGemma4ExpertCacheRefreshEvery == 0 {
			if err := cache.refreshAdaptiveBudget(entry.bytes); err != nil {
				return nil, err
			}
		}
		return entry, nil
	}
	cache.stats.Misses++
	gateUpSlice, gateUpRows, gateUpCols, gateUpFormat, err := cache.expertTensorSlice(gateUpInfo, key.Expert, expectedExperts)
	if err != nil {
		return nil, core.E("rocm.hip.Gemma4ExpertCache", "resolve expert gate/up slice", err)
	}
	downSlice, downRows, downCols, downFormat, err := cache.expertTensorSlice(downInfo, key.Expert, expectedExperts)
	if err != nil {
		return nil, core.E("rocm.hip.Gemma4ExpertCache", "resolve expert down slice", err)
	}
	gateUpBytes := uint64(len(gateUpSlice))
	if gateUpFormat == hipGGUFExpertFormatQ4K && cache.q4KExpansionEnabled() {
		gateUpBytes = uint64(len(gateUpSlice)/hipGGUFQ4KBlockBytes) * hipGGUFQ4KExpandedBlockBytes
	}
	entryBytes := gateUpBytes + uint64(len(downSlice))
	if err := cache.refreshAdaptiveBudget(entryBytes); err != nil {
		return nil, err
	}
	if entryBytes > cache.maxBytes {
		return nil, core.E("rocm.hip.Gemma4ExpertCache", "one expert exceeds the cache byte limit", nil)
	}
	for cache.bytes+entryBytes > cache.maxBytes {
		if err := cache.evictOldest(); err != nil {
			return nil, err
		}
	}
	gateUpBuffer, gateUpFormat, err := cache.uploadGateUpBuffer(gateUpSlice, gateUpFormat)
	if err != nil {
		return nil, err
	}
	entryBytes = gateUpBuffer.SizeBytes() + uint64(len(downSlice))
	downBuffer, err := cache.uploadExpertBuffer("expert down", downSlice, len(downSlice))
	if err != nil {
		_ = gateUpBuffer.Close()
		return nil, err
	}
	entry := &hipGemma4ExpertCacheEntry{
		GateUp: gateUpBuffer, Down: downBuffer,
		GateUpRows: gateUpRows, GateUpCols: gateUpCols,
		DownRows: downRows, DownCols: downCols,
		GateUpFormat: gateUpFormat, DownFormat: downFormat,
		bytes: entryBytes, lastUse: cache.clock,
	}
	cache.entries[key] = entry
	cache.bytes += entryBytes
	return entry, nil
}

func (cache *hipGemma4ExpertCache) entryMLXAffine(key hipGemma4ExpertCacheKey, source hipGemma4MLXAffineExpertSource, expectedExperts, hidden, expertFF, groupSize, preferredBits int) (*hipGemma4ExpertCacheEntry, error) {
	const operation = "rocm.hip.Gemma4ExpertCache"
	if cache == nil || cache.driver == nil || !cache.driver.Available() {
		return nil, core.E(operation, "HIP driver is not available", nil)
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	cache.clock++
	if entry := cache.entries[key]; entry != nil {
		if entry.Storage != hipGemma4MoEExpertStorageMLXAffine {
			return nil, core.E(operation, "cached expert storage format mismatch", nil)
		}
		cache.stats.Hits++
		entry.lastUse = cache.clock
		if cache.clock%hipGemma4ExpertCacheRefreshEvery == 0 {
			if err := cache.refreshAdaptiveBudget(entry.bytes); err != nil {
				return nil, err
			}
		}
		return entry, nil
	}
	cache.stats.Misses++

	gateUpWeight, gateUpWeightRows, gateUpPackedCols, err := cache.mlxAffineExpertTensorSlice(source.GateUp.Weight, key.Expert, expectedExperts, "U32")
	if err != nil {
		return nil, core.E(operation, "resolve MLX affine expert gate/up weight slice", err)
	}
	gateUpScales, gateUpScaleRows, gateUpGroups, err := cache.mlxAffineExpertTensorSlice(source.GateUp.Scales, key.Expert, expectedExperts, "BF16")
	if err != nil {
		return nil, core.E(operation, "resolve MLX affine expert gate/up scale slice", err)
	}
	gateUpBiases, gateUpBiasRows, gateUpBiasGroups, err := cache.mlxAffineExpertTensorSlice(source.GateUp.Biases, key.Expert, expectedExperts, "BF16")
	if err != nil {
		return nil, core.E(operation, "resolve MLX affine expert gate/up bias slice", err)
	}
	downWeight, downWeightRows, downPackedCols, err := cache.mlxAffineExpertTensorSlice(source.Down.Weight, key.Expert, expectedExperts, "U32")
	if err != nil {
		return nil, core.E(operation, "resolve MLX affine expert down weight slice", err)
	}
	downScales, downScaleRows, downGroups, err := cache.mlxAffineExpertTensorSlice(source.Down.Scales, key.Expert, expectedExperts, "BF16")
	if err != nil {
		return nil, core.E(operation, "resolve MLX affine expert down scale slice", err)
	}
	downBiases, downBiasRows, downBiasGroups, err := cache.mlxAffineExpertTensorSlice(source.Down.Biases, key.Expert, expectedExperts, "BF16")
	if err != nil {
		return nil, core.E(operation, "resolve MLX affine expert down bias slice", err)
	}

	gateUpWeightTensor := hipTensor{info: nativeTensorInfo{TypeName: "U32", Dimensions: []uint64{uint64(gateUpWeightRows), uint64(gateUpPackedCols)}, ByteSize: uint64(len(gateUpWeight))}}
	gateUpScaleTensor := hipTensor{info: nativeTensorInfo{TypeName: "BF16", Dimensions: []uint64{uint64(gateUpScaleRows), uint64(gateUpGroups)}, ByteSize: uint64(len(gateUpScales))}}
	gateUpBiasTensor := hipTensor{info: nativeTensorInfo{TypeName: "BF16", Dimensions: []uint64{uint64(gateUpBiasRows), uint64(gateUpBiasGroups)}, ByteSize: uint64(len(gateUpBiases))}}
	gateUpBits, gateUpRows, gateUpCols, gateUpGroupSize, gateUpGroupCount, gateUpPacked, err := hipInferMLXAffineBitsFromTensorShapes(
		gateUpWeightTensor, gateUpScaleTensor, gateUpBiasTensor, groupSize, preferredBits, "expert gate/up",
	)
	if err != nil {
		return nil, err
	}
	downWeightTensor := hipTensor{info: nativeTensorInfo{TypeName: "U32", Dimensions: []uint64{uint64(downWeightRows), uint64(downPackedCols)}, ByteSize: uint64(len(downWeight))}}
	downScaleTensor := hipTensor{info: nativeTensorInfo{TypeName: "BF16", Dimensions: []uint64{uint64(downScaleRows), uint64(downGroups)}, ByteSize: uint64(len(downScales))}}
	downBiasTensor := hipTensor{info: nativeTensorInfo{TypeName: "BF16", Dimensions: []uint64{uint64(downBiasRows), uint64(downBiasGroups)}, ByteSize: uint64(len(downBiases))}}
	downBits, downRows, downCols, downGroupSize, downGroupCount, downPacked, err := hipInferMLXAffineBitsFromTensorShapes(
		downWeightTensor, downScaleTensor, downBiasTensor, groupSize, preferredBits, "expert down",
	)
	if err != nil {
		return nil, err
	}
	if gateUpRows != 2*expertFF || gateUpCols != hidden || downRows != hidden || downCols != expertFF {
		return nil, core.E(operation, "MLX affine expert tensor geometry mismatch", nil)
	}

	entryBytes := uint64(len(gateUpWeight) + len(gateUpScales) + len(gateUpBiases) + len(downWeight) + len(downScales) + len(downBiases))
	if err := cache.refreshAdaptiveBudget(entryBytes); err != nil {
		return nil, err
	}
	if entryBytes > cache.maxBytes {
		return nil, core.E(operation, "one expert exceeds the cache byte limit", nil)
	}
	for cache.bytes+entryBytes > cache.maxBytes {
		if err := cache.evictOldest(); err != nil {
			return nil, err
		}
	}

	var uploaded []*hipDeviceByteBuffer
	upload := func(label string, payload []byte) (*hipDeviceByteBuffer, error) {
		buffer, uploadErr := cache.uploadExpertBuffer(label, payload, len(payload))
		if uploadErr != nil {
			for _, existing := range uploaded {
				_ = existing.Close()
			}
			return nil, uploadErr
		}
		uploaded = append(uploaded, buffer)
		return buffer, nil
	}
	gateUpWeightBuffer, err := upload("MLX affine expert gate/up weight", gateUpWeight)
	if err != nil {
		return nil, err
	}
	gateUpScaleBuffer, err := upload("MLX affine expert gate/up scales", gateUpScales)
	if err != nil {
		return nil, err
	}
	gateUpBiasBuffer, err := upload("MLX affine expert gate/up biases", gateUpBiases)
	if err != nil {
		return nil, err
	}
	downWeightBuffer, err := upload("MLX affine expert down weight", downWeight)
	if err != nil {
		return nil, err
	}
	downScaleBuffer, err := upload("MLX affine expert down scales", downScales)
	if err != nil {
		return nil, err
	}
	downBiasBuffer, err := upload("MLX affine expert down biases", downBiases)
	if err != nil {
		return nil, err
	}

	gateWeightBytes := uint64(expertFF) * uint64(gateUpPacked) * 4
	gateScaleBytes := uint64(expertFF) * uint64(gateUpGroupCount) * 2
	gate := hipMLXQ4DeviceWeightConfig{
		WeightPointer: gateUpWeightBuffer.Pointer(), ScalePointer: gateUpScaleBuffer.Pointer(), BiasPointer: gateUpBiasBuffer.Pointer(),
		WeightBytes: gateWeightBytes, ScaleBytes: gateScaleBytes, BiasBytes: gateScaleBytes,
		Rows: expertFF, Cols: hidden, GroupSize: gateUpGroupSize, Bits: gateUpBits,
	}
	up := gate
	up.WeightPointer += nativeDevicePointer(gateWeightBytes)
	up.ScalePointer += nativeDevicePointer(gateScaleBytes)
	up.BiasPointer += nativeDevicePointer(gateScaleBytes)
	down := hipMLXQ4DeviceWeightConfig{
		WeightPointer: downWeightBuffer.Pointer(), ScalePointer: downScaleBuffer.Pointer(), BiasPointer: downBiasBuffer.Pointer(),
		WeightBytes: uint64(downRows) * uint64(downPacked) * 4,
		ScaleBytes:  uint64(downRows) * uint64(downGroupCount) * 2,
		BiasBytes:   uint64(downRows) * uint64(downGroupCount) * 2,
		Rows:        downRows, Cols: downCols, GroupSize: downGroupSize, Bits: downBits,
	}
	if err := gate.validateInputCount(hidden); err != nil {
		for _, buffer := range uploaded {
			_ = buffer.Close()
		}
		return nil, core.E(operation, "validate MLX affine expert gate projection", err)
	}
	if err := up.validateInputCount(hidden); err != nil {
		for _, buffer := range uploaded {
			_ = buffer.Close()
		}
		return nil, core.E(operation, "validate MLX affine expert up projection", err)
	}
	if err := down.validateInputCount(expertFF); err != nil {
		for _, buffer := range uploaded {
			_ = buffer.Close()
		}
		return nil, core.E(operation, "validate MLX affine expert down projection", err)
	}
	entry := &hipGemma4ExpertCacheEntry{
		Storage:         hipGemma4MoEExpertStorageMLXAffine,
		MLXGateUpWeight: gateUpWeightBuffer, MLXGateUpScales: gateUpScaleBuffer, MLXGateUpBiases: gateUpBiasBuffer,
		MLXDownWeight: downWeightBuffer, MLXDownScales: downScaleBuffer, MLXDownBiases: downBiasBuffer,
		MLXGate: gate, MLXUp: up, MLXDown: down,
		bytes: entryBytes, lastUse: cache.clock,
	}
	cache.entries[key] = entry
	cache.bytes += entryBytes
	return entry, nil
}

func (cache *hipGemma4ExpertCache) q4KExpansionEnabled() bool {
	if cache == nil || cache.expandedGPUDisabled || core.Env(hipGemma4Q4KExpandedEnv) == "0" {
		return false
	}
	_, ok := cache.driver.(nativeHIPKernelLauncher)
	return ok
}

func (cache *hipGemma4ExpertCache) uploadGateUpBuffer(payload []byte, format uint32) (*hipDeviceByteBuffer, uint32, error) {
	if format != hipGGUFExpertFormatQ4K || !cache.q4KExpansionEnabled() {
		buffer, err := cache.uploadExpertBuffer("expert gate/up", payload, len(payload))
		return buffer, format, err
	}
	if len(payload) == 0 || len(payload)%hipGGUFQ4KBlockBytes != 0 {
		return nil, 0, core.E("rocm.hip.Gemma4ExpertCache", "Q4_K gate/up payload must contain complete blocks", nil)
	}
	blockCount := len(payload) / hipGGUFQ4KBlockBytes
	if blockCount > int(^uint(0)>>1)/hipGGUFQ4KExpandedBlockBytes {
		return nil, 0, core.E("rocm.hip.Gemma4ExpertCache", "expanded Q4_K gate/up byte count overflows", nil)
	}
	expandedByteCount := blockCount * hipGGUFQ4KExpandedBlockBytes
	expandedBytes := uint64(expandedByteCount)
	expanded, err := cache.allocateExpertBuffer("expanded expert gate/up", expandedBytes, expandedByteCount)
	if err != nil {
		return nil, 0, err
	}
	staging, err := cache.ensureGateUpStaging(uint64(len(payload)))
	if err != nil {
		_ = expanded.Close()
		return nil, 0, err
	}
	if err := hipCopyHostToDeviceLabeled(cache.driver, staging.Pointer(), payload, "rocm.hip.Gemma4ExpertCache", "raw expert gate/up staging"); err != nil {
		_ = expanded.Close()
		return nil, 0, core.E("rocm.hip.Gemma4ExpertCache", "copy raw expert gate/up staging", err)
	}
	cache.stats.H2DBytes += uint64(len(payload))
	if err := hipRunGGUFQ4KExpandMetadataKernel(context.Background(), cache.driver, staging, uint64(len(payload)), expanded, blockCount); err != nil {
		_ = expanded.Close()
		cache.expandedGPUDisabled = true
		raw, uploadErr := cache.uploadExpertBuffer("expert gate/up", payload, len(payload))
		return raw, format, uploadErr
	}
	return expanded, hipGGUFExpertFormatQ4KExpanded, nil
}

func (cache *hipGemma4ExpertCache) allocateExpertBuffer(label string, sizeBytes uint64, count int) (*hipDeviceByteBuffer, error) {
	const operation = "rocm.hip.Gemma4ExpertCache"
	for {
		buffer, err := hipAllocateByteBuffer(cache.driver, operation, label, sizeBytes, count)
		if err == nil {
			buffer.pooled = false
			return buffer, nil
		}
		if len(cache.entries) == 0 {
			return nil, err
		}
		cache.stats.AllocationRetries++
		if evictErr := cache.evictOldest(); evictErr != nil {
			return nil, evictErr
		}
	}
}

func (cache *hipGemma4ExpertCache) ensureGateUpStaging(sizeBytes uint64) (*hipDeviceByteBuffer, error) {
	if cache.gateUpStaging != nil && cache.gateUpStaging.SizeBytes() >= sizeBytes {
		return cache.gateUpStaging, nil
	}
	staging, err := cache.allocateExpertBuffer("raw expert gate/up staging", sizeBytes, int(sizeBytes))
	if err != nil {
		return nil, err
	}
	if err := cache.gateUpStaging.Close(); err != nil {
		_ = staging.Close()
		return nil, err
	}
	cache.gateUpStaging = staging
	return staging, nil
}

func (cache *hipGemma4ExpertCache) uploadExpertBuffer(label string, payload []byte, count int) (*hipDeviceByteBuffer, error) {
	const operation = "rocm.hip.Gemma4ExpertCache"
	for {
		buffer, err := hipAllocateByteBuffer(cache.driver, operation, label, uint64(len(payload)), count)
		if err != nil {
			if len(cache.entries) == 0 {
				return nil, err
			}
			cache.stats.AllocationRetries++
			if evictErr := cache.evictOldest(); evictErr != nil {
				return nil, evictErr
			}
			continue
		}
		buffer.pooled = false
		if err := hipCopyHostToDeviceLabeled(cache.driver, buffer.pointer, payload, operation, label); err != nil {
			_ = buffer.Close()
			return nil, core.E(operation, "copy "+label, err)
		}
		cache.stats.H2DBytes += uint64(len(payload))
		return buffer, nil
	}
}

func (cache *hipGemma4ExpertCache) refreshAdaptiveBudget(entryBytes uint64) error {
	if cache == nil || !cache.adaptive || cache.driver == nil {
		return nil
	}
	freeBytes := cache.driver.DeviceInfo().FreeBytes
	if freeBytes == 0 {
		return nil
	}
	minimumEntries := cache.minimumEntries
	if minimumEntries < 1 {
		minimumEntries = 1
	}
	if entryBytes > ^uint64(0)/uint64(minimumEntries) {
		return core.E("rocm.hip.Gemma4ExpertCache", "minimum route byte size overflows", nil)
	}
	minimumBytes := entryBytes * uint64(minimumEntries)
	targetBytes := cache.bytes
	if freeBytes >= hipGemma4ExpertCacheReserveBytes {
		headroom := freeBytes - hipGemma4ExpertCacheReserveBytes
		if headroom > ^uint64(0)-targetBytes {
			targetBytes = ^uint64(0)
		} else {
			targetBytes += headroom
		}
	} else {
		deficit := hipGemma4ExpertCacheReserveBytes - freeBytes
		if deficit >= targetBytes {
			targetBytes = 0
		} else {
			targetBytes -= deficit
		}
	}
	if targetBytes > hipGemma4ExpertCacheMaximumBytes {
		targetBytes = hipGemma4ExpertCacheMaximumBytes
	}
	if targetBytes < minimumBytes {
		targetBytes = minimumBytes
	}
	cache.maxBytes = targetBytes
	cache.stats.BudgetRefreshes++
	for cache.bytes > cache.maxBytes {
		if err := cache.evictOldest(); err != nil {
			return err
		}
	}
	return nil
}

func (cache *hipGemma4ExpertCache) evictOldest() error {
	var oldestKey hipGemma4ExpertCacheKey
	var oldest *hipGemma4ExpertCacheEntry
	for key, entry := range cache.entries {
		if oldest == nil || entry.lastUse < oldest.lastUse {
			oldestKey = key
			oldest = entry
		}
	}
	if oldest == nil {
		return core.E("rocm.hip.Gemma4ExpertCache", "cache byte limit cannot satisfy expert allocation", nil)
	}
	if err := oldest.Close(); err != nil {
		return err
	}
	delete(cache.entries, oldestKey)
	cache.bytes -= oldest.bytes
	cache.stats.Evictions++
	return nil
}

func (entry *hipGemma4ExpertCacheEntry) Close() error {
	if entry == nil {
		return nil
	}
	var lastErr error
	if err := entry.GateUp.Close(); err != nil {
		lastErr = err
	}
	if err := entry.Down.Close(); err != nil {
		lastErr = err
	}
	for _, buffer := range []*hipDeviceByteBuffer{
		entry.MLXGateUpWeight, entry.MLXGateUpScales, entry.MLXGateUpBiases,
		entry.MLXDownWeight, entry.MLXDownScales, entry.MLXDownBiases,
	} {
		if buffer == nil {
			continue
		}
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	entry.GateUp = nil
	entry.Down = nil
	entry.MLXGateUpWeight = nil
	entry.MLXGateUpScales = nil
	entry.MLXGateUpBiases = nil
	entry.MLXDownWeight = nil
	entry.MLXDownScales = nil
	entry.MLXDownBiases = nil
	entry.MLXGate = hipMLXQ4DeviceWeightConfig{}
	entry.MLXUp = hipMLXQ4DeviceWeightConfig{}
	entry.MLXDown = hipMLXQ4DeviceWeightConfig{}
	return lastErr
}

func (cache *hipGemma4ExpertCache) Close() error {
	if cache == nil {
		return nil
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	var lastErr error
	for key, entry := range cache.entries {
		if err := entry.Close(); err != nil {
			lastErr = err
		}
		delete(cache.entries, key)
	}
	for path, source := range cache.sources {
		if err := source.Close(); err != nil {
			lastErr = err
		}
		delete(cache.sources, path)
	}
	if err := cache.gateUpStaging.Close(); err != nil {
		lastErr = err
	}
	cache.gateUpStaging = nil
	cache.bytes = 0
	if cache.releaseTransientPoolSuppression != nil {
		cache.releaseTransientPoolSuppression()
		cache.releaseTransientPoolSuppression = nil
	}
	return lastErr
}

func (source *hipGemma4MappedExpertSource) Close() error {
	if source == nil {
		return nil
	}
	var lastErr error
	if len(source.data) > 0 {
		if err := syscall.Munmap(source.data); err != nil {
			lastErr = err
		}
		source.data = nil
	}
	if source.file != nil {
		if err := source.file.Close(); err != nil {
			lastErr = err
		}
		source.file = nil
	}
	return lastErr
}

func (cache *hipGemma4ExpertCache) mappedExpertSource(path string) (*hipGemma4MappedExpertSource, error) {
	if source := cache.sources[path]; source != nil {
		return source, nil
	}
	fileResult := core.Open(path)
	if !fileResult.OK {
		return nil, fileResult.Value.(error)
	}
	file := fileResult.Value.(*core.OSFile)
	info, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return nil, err
	}
	size := info.Size()
	if size <= 0 || uint64(size) > uint64(^uint(0)>>1) {
		_ = file.Close()
		return nil, core.E("rocm.hip.Gemma4ExpertCache", "expert tensor source size is invalid", nil)
	}
	mapping, err := syscall.Mmap(int(file.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		_ = file.Close()
		return nil, err
	}
	source := &hipGemma4MappedExpertSource{file: file, data: mapping}
	cache.sources[path] = source
	cache.stats.HostMappings++
	cache.stats.HostMappedBytes += uint64(size)
	return source, nil
}

func (cache *hipGemma4ExpertCache) expertTensorSlice(info nativeTensorInfo, expert, expectedExperts int) ([]byte, int, int, uint32, error) {
	format, ok := hipGGUFExpertFormatForInfo(info)
	if !ok || len(info.Dimensions) != 3 {
		return nil, 0, 0, 0, core.E("rocm.hip.Gemma4ExpertCache", "expert tensor must use a supported rank-3 GGUF format", nil)
	}
	blockSize, blockBytes, ok := hipGGUFExpertBlockGeometry(format)
	if !ok {
		return nil, 0, 0, 0, core.E("rocm.hip.Gemma4ExpertCache", "expert tensor format geometry is unavailable", nil)
	}
	cols, rows, experts := int(info.Dimensions[0]), int(info.Dimensions[1]), int(info.Dimensions[2])
	if cols <= 0 || rows <= 0 || experts <= 0 || cols%blockSize != 0 {
		return nil, 0, 0, 0, core.E("rocm.hip.Gemma4ExpertCache", "expert tensor geometry is invalid", nil)
	}
	if expectedExperts > 0 && experts != expectedExperts {
		return nil, 0, 0, 0, core.E("rocm.hip.Gemma4ExpertCache", "expert tensor count mismatch", nil)
	}
	if expert < 0 || expert >= experts {
		return nil, 0, 0, 0, core.E("rocm.hip.Gemma4ExpertCache", "expert index is outside the tensor", nil)
	}
	sliceBytes := uint64(rows) * uint64(cols/blockSize) * blockBytes
	if sliceBytes == 0 || info.ByteSize != sliceBytes*uint64(experts) || sliceBytes > uint64(^uint(0)>>1) {
		return nil, 0, 0, 0, core.E("rocm.hip.Gemma4ExpertCache", "expert tensor byte count mismatch", nil)
	}
	path := info.SourcePath
	if path == "" {
		return nil, 0, 0, 0, core.E("rocm.hip.Gemma4ExpertCache", "expert tensor source path is required", nil)
	}
	source, err := cache.mappedExpertSource(path)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	if info.DataOffset < 0 {
		return nil, 0, 0, 0, core.E("rocm.hip.Gemma4ExpertCache", "expert tensor data offset is invalid", nil)
	}
	start := uint64(info.DataOffset)
	if info.Offset > ^uint64(0)-start {
		return nil, 0, 0, 0, core.E("rocm.hip.Gemma4ExpertCache", "expert tensor slice offset overflows", nil)
	}
	start += info.Offset
	expertOffset := uint64(expert) * sliceBytes
	if expertOffset > ^uint64(0)-start {
		return nil, 0, 0, 0, core.E("rocm.hip.Gemma4ExpertCache", "expert tensor slice offset overflows", nil)
	}
	start += expertOffset
	if sliceBytes > ^uint64(0)-start {
		return nil, 0, 0, 0, core.E("rocm.hip.Gemma4ExpertCache", "expert tensor slice range overflows", nil)
	}
	end := start + sliceBytes
	if end > uint64(len(source.data)) {
		return nil, 0, 0, 0, core.E("rocm.hip.Gemma4ExpertCache", "expert tensor slice is truncated", nil)
	}
	payload := source.data[int(start):int(end)]
	return payload, rows, cols, format, nil
}

func (cache *hipGemma4ExpertCache) mlxAffineExpertTensorSlice(info nativeTensorInfo, expert, expectedExperts int, expectedType string) ([]byte, int, int, error) {
	const operation = "rocm.hip.Gemma4ExpertCache"
	elementBytes := uint64(0)
	switch core.Upper(expectedType) {
	case "U32":
		elementBytes = 4
	case "BF16":
		elementBytes = 2
	default:
		return nil, 0, 0, core.E(operation, "unsupported MLX affine expert element type", nil)
	}
	if core.Upper(info.TypeName) != core.Upper(expectedType) || len(info.Dimensions) != 3 {
		return nil, 0, 0, core.E(operation, "MLX affine expert tensor type or rank mismatch", nil)
	}
	experts64, rows64, cols64 := info.Dimensions[0], info.Dimensions[1], info.Dimensions[2]
	maxInt := uint64(^uint(0) >> 1)
	if experts64 == 0 || rows64 == 0 || cols64 == 0 || experts64 > maxInt || rows64 > maxInt || cols64 > maxInt {
		return nil, 0, 0, core.E(operation, "MLX affine expert tensor geometry is invalid", nil)
	}
	experts, rows, cols := int(experts64), int(rows64), int(cols64)
	if expectedExperts > 0 && experts != expectedExperts {
		return nil, 0, 0, core.E(operation, "MLX affine expert tensor count mismatch", nil)
	}
	if expert < 0 || expert >= experts {
		return nil, 0, 0, core.E(operation, "expert index is outside the MLX affine tensor", nil)
	}
	if rows64 > ^uint64(0)/cols64 || rows64*cols64 > ^uint64(0)/elementBytes {
		return nil, 0, 0, core.E(operation, "MLX affine expert tensor byte count overflows", nil)
	}
	sliceBytes := rows64 * cols64 * elementBytes
	if sliceBytes == 0 || experts64 > ^uint64(0)/sliceBytes || info.ByteSize != sliceBytes*experts64 || sliceBytes > maxInt {
		return nil, 0, 0, core.E(operation, "MLX affine expert tensor byte count mismatch", nil)
	}
	if info.SourcePath == "" {
		return nil, 0, 0, core.E(operation, "MLX affine expert tensor source path is required", nil)
	}
	source, err := cache.mappedExpertSource(info.SourcePath)
	if err != nil {
		return nil, 0, 0, err
	}
	if info.DataOffset < 0 {
		return nil, 0, 0, core.E(operation, "MLX affine expert tensor data offset is invalid", nil)
	}
	start := uint64(info.DataOffset)
	if info.Offset > ^uint64(0)-start {
		return nil, 0, 0, core.E(operation, "MLX affine expert tensor slice offset overflows", nil)
	}
	start += info.Offset
	expertOffset := uint64(expert) * sliceBytes
	if expertOffset > ^uint64(0)-start {
		return nil, 0, 0, core.E(operation, "MLX affine expert tensor slice offset overflows", nil)
	}
	start += expertOffset
	if sliceBytes > ^uint64(0)-start {
		return nil, 0, 0, core.E(operation, "MLX affine expert tensor slice range overflows", nil)
	}
	end := start + sliceBytes
	if end > uint64(len(source.data)) {
		return nil, 0, 0, core.E(operation, "MLX affine expert tensor slice is truncated", nil)
	}
	return source.data[int(start):int(end)], rows, cols, nil
}

func (args hipGGUFQ4KExpandLaunchArgs) Binary() ([]byte, error) {
	if args.RawPointer == 0 || args.ExpandedPointer == 0 {
		return nil, core.E("rocm.hip.GGUFQ4KExpand", "raw and expanded pointers are required", nil)
	}
	blockCount, err := rocmDeviceKVPositiveUint32("Q4_K block count", args.BlockCount)
	if err != nil {
		return nil, err
	}
	if args.RawBytes != uint64(blockCount)*hipGGUFQ4KBlockBytes || args.ExpandedBytes != uint64(blockCount)*hipGGUFQ4KExpandedBlockBytes {
		return nil, core.E("rocm.hip.GGUFQ4KExpand", "Q4_K raw or expanded byte count is invalid", nil)
	}
	payload := make([]byte, hipGGUFQ4KExpandLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipGGUFQ4KExpandLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(hipGGUFQ4KExpandLaunchArgsBytes))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.RawPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.ExpandedPointer))
	binary.LittleEndian.PutUint32(payload[24:], blockCount)
	binary.LittleEndian.PutUint64(payload[32:], args.RawBytes)
	binary.LittleEndian.PutUint64(payload[40:], args.ExpandedBytes)
	return payload, nil
}

func hipRunGGUFQ4KExpandMetadataKernel(ctx context.Context, driver nativeHIPDriver, raw *hipDeviceByteBuffer, rawBytes uint64, expanded *hipDeviceByteBuffer, blockCount int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if raw == nil || expanded == nil || rawBytes > raw.SizeBytes() {
		return core.E("rocm.hip.GGUFQ4KExpand", "raw and expanded device buffers are required", nil)
	}
	args := hipGGUFQ4KExpandLaunchArgs{
		RawPointer:      raw.Pointer(),
		ExpandedPointer: expanded.Pointer(),
		BlockCount:      blockCount,
		RawBytes:        rawBytes,
		ExpandedBytes:   expanded.SizeBytes(),
	}
	payload, err := args.Binary()
	if err != nil {
		return err
	}
	if blockCount > int(^uint(0)>>1)/32 {
		return core.E("rocm.hip.GGUFQ4KExpand", "Q4_K expansion work size overflows", nil)
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameGGUFQ4KExpandMetadata, payload, blockCount*32)
	if err != nil {
		return err
	}
	return hipLaunchKernel(driver, config)
}

func (args hipGGUFQ4_0ProjectionLaunchArgs) Binary() ([]byte, error) {
	payload := make([]byte, hipGGUFQ4_0ProjectionLaunchArgsBytes)
	return args.BinaryInto(payload)
}

func (args hipGGUFQ4_0ProjectionLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.WeightPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.GGUFQ4_0ProjectionLaunch", "input, weight, and output pointers are required", nil)
	}
	if len(payload) < hipGGUFQ4_0ProjectionLaunchArgsBytes {
		return nil, core.E("rocm.hip.GGUFQ4_0ProjectionLaunch", "launch arg payload buffer is too small", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("row count", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("column count", args.Cols)
	if err != nil {
		return nil, err
	}
	weightRows, err := rocmDeviceKVPositiveUint32("weight row count", args.WeightRows)
	if err != nil {
		return nil, err
	}
	if args.RowOffset < 0 || uint64(args.RowOffset) > math.MaxUint32 {
		return nil, core.E("rocm.hip.GGUFQ4_0ProjectionLaunch", "row offset exceeds uint32", nil)
	}
	rowOffset := uint32(args.RowOffset)
	if uint64(rowOffset)+uint64(rows) > uint64(weightRows) {
		return nil, core.E("rocm.hip.GGUFQ4_0ProjectionLaunch", "projection row range exceeds weight rows", nil)
	}
	if cols%hipGGUFQ4_0BlockSize != 0 {
		return nil, core.E("rocm.hip.GGUFQ4_0ProjectionLaunch", "column count must align to Q4_0 blocks", nil)
	}
	expectedInputBytes := uint64(cols) * 4
	expectedWeightBytes := uint64(weightRows) * uint64(cols/hipGGUFQ4_0BlockSize) * hipGGUFQ4_0BlockBytes
	expectedOutputBytes := uint64(rows) * 4
	if args.InputBytes != expectedInputBytes || args.WeightBytes != expectedWeightBytes || args.OutputBytes != expectedOutputBytes {
		return nil, core.E("rocm.hip.GGUFQ4_0ProjectionLaunch", "buffer byte count mismatch", nil)
	}
	if args.InputBytes > math.MaxUint32 || args.WeightBytes > math.MaxUint32 || args.OutputBytes > math.MaxUint32 {
		return nil, core.E("rocm.hip.GGUFQ4_0ProjectionLaunch", "buffer byte count exceeds uint32", nil)
	}
	payload = payload[:hipGGUFQ4_0ProjectionLaunchArgsBytes]
	clear(payload)
	binary.LittleEndian.PutUint32(payload[0:], hipGGUFQ4_0ProjectionLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], rows)
	binary.LittleEndian.PutUint32(payload[36:], cols)
	binary.LittleEndian.PutUint32(payload[40:], rowOffset)
	binary.LittleEndian.PutUint32(payload[44:], weightRows)
	binary.LittleEndian.PutUint32(payload[48:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[52:], uint32(args.WeightBytes))
	binary.LittleEndian.PutUint32(payload[56:], uint32(args.OutputBytes))
	return payload, nil
}

func (args hipGGUFQ4_0SelectedExpertsLaunchArgs) Binary() ([]byte, error) {
	payload := make([]byte, hipGGUFQ4_0SelectedExpertsLaunchArgsBytes)
	return args.BinaryInto(payload)
}

func (args hipGGUFQ4_0SelectedExpertsLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if len(payload) < hipGGUFQ4_0SelectedExpertsLaunchArgsBytes {
		return nil, core.E("rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "launch arg payload buffer is too small", nil)
	}
	if args.InputPointer == 0 || args.ActivationPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "input, activation, and output pointers are required", nil)
	}
	if args.TopK <= 0 || args.TopK > hipGGUFQ4_0SelectedExpertsMaxTopK {
		return nil, core.E("rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "top-k must fit the selected expert packet", nil)
	}
	hidden, err := rocmDeviceKVPositiveUint32("hidden size", args.HiddenSize)
	if err != nil {
		return nil, err
	}
	expertFF, err := rocmDeviceKVPositiveUint32("expert intermediate size", args.ExpertFF)
	if err != nil {
		return nil, err
	}
	gateUpRows, err := rocmDeviceKVPositiveUint32("gate/up row count", args.GateUpRows)
	if err != nil {
		return nil, err
	}
	downRows, err := rocmDeviceKVPositiveUint32("down row count", args.DownRows)
	if err != nil {
		return nil, err
	}
	gateBlockSize, gateBlockBytes, gateOK := hipGGUFExpertBlockGeometry(args.GateUpFormat)
	downBlockSize, downBlockBytes, downOK := hipGGUFExpertBlockGeometry(args.DownFormat)
	if !gateOK || !downOK || hidden%uint32(gateBlockSize) != 0 || expertFF%uint32(downBlockSize) != 0 || gateUpRows != 2*expertFF || downRows != hidden {
		return nil, core.E("rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "selected expert geometry is invalid", nil)
	}
	wantInputBytes := uint64(hidden) * 4
	wantActivationBytes := uint64(args.TopK) * uint64(expertFF) * 4
	wantOutputBytes := uint64(hidden) * 4
	wantGateUpBytes := uint64(gateUpRows) * uint64(hidden/uint32(gateBlockSize)) * gateBlockBytes
	wantDownBytes := uint64(downRows) * uint64(expertFF/uint32(downBlockSize)) * downBlockBytes
	for _, value := range []uint64{args.InputBytes, args.ActivationBytes, args.OutputBytes, args.GateUpBytes, args.DownBytes} {
		if value > math.MaxUint32 {
			return nil, core.E("rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "buffer byte count exceeds uint32", nil)
		}
	}
	if args.InputBytes != wantInputBytes || args.ActivationBytes != wantActivationBytes || args.OutputBytes != wantOutputBytes ||
		args.GateUpBytes != wantGateUpBytes || args.DownBytes != wantDownBytes {
		return nil, core.E("rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "selected expert buffer byte count mismatch", nil)
	}
	for index := 0; index < args.TopK; index++ {
		if args.GateUpPointers[index] == 0 || args.DownPointers[index] == 0 ||
			math.IsNaN(float64(args.RouteWeights[index])) || math.IsInf(float64(args.RouteWeights[index]), 0) {
			return nil, core.E("rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "selected expert pointers and finite route weights are required", nil)
		}
	}
	payload = payload[:hipGGUFQ4_0SelectedExpertsLaunchArgsBytes]
	clear(payload)
	binary.LittleEndian.PutUint32(payload[0:], hipGGUFQ4_0SelectedExpertsLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.ActivationPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	for index := 0; index < hipGGUFQ4_0SelectedExpertsMaxTopK; index++ {
		binary.LittleEndian.PutUint64(payload[32+index*8:], uint64(args.GateUpPointers[index]))
		binary.LittleEndian.PutUint64(payload[96+index*8:], uint64(args.DownPointers[index]))
		binary.LittleEndian.PutUint32(payload[160+index*4:], math.Float32bits(args.RouteWeights[index]))
	}
	binary.LittleEndian.PutUint32(payload[192:], uint32(args.TopK))
	binary.LittleEndian.PutUint32(payload[196:], hidden)
	binary.LittleEndian.PutUint32(payload[200:], expertFF)
	binary.LittleEndian.PutUint32(payload[204:], gateUpRows)
	binary.LittleEndian.PutUint32(payload[208:], downRows)
	binary.LittleEndian.PutUint32(payload[212:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[216:], uint32(args.ActivationBytes))
	binary.LittleEndian.PutUint32(payload[220:], uint32(args.OutputBytes))
	binary.LittleEndian.PutUint32(payload[224:], uint32(args.GateUpBytes))
	binary.LittleEndian.PutUint32(payload[228:], uint32(args.DownBytes))
	binary.LittleEndian.PutUint32(payload[232:], args.GateUpFormat)
	binary.LittleEndian.PutUint32(payload[236:], args.DownFormat)
	return payload, nil
}

func hipRunGemma4MLXAffineSelectedExpertsWithWorkspace(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, entries []*hipGemma4ExpertCacheEntry, routeWeights []float32, hidden, expertFF int, activation, downOutput, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	const operation = "rocm.hip.Gemma4MLXAffineSelectedExperts"
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E(operation, "HIP driver is not available", nil)
	}
	if hidden <= 0 || expertFF <= 0 || len(entries) == 0 || len(entries) != len(routeWeights) || len(entries) > hipGGUFQ4_0SelectedExpertsMaxTopK {
		return core.E(operation, "selected expert geometry is invalid", nil)
	}
	if input == nil || input.Pointer() == 0 || input.Count() != hidden || input.SizeBytes() != uint64(hidden*4) {
		return core.E(operation, "selected expert input must match the hidden size", nil)
	}
	if activation == nil || activation.Pointer() == 0 || activation.Count() != expertFF || activation.SizeBytes() != uint64(expertFF*4) {
		return core.E(operation, "selected expert activation must match the expert intermediate size", nil)
	}
	for label, buffer := range map[string]*hipDeviceByteBuffer{"down output": downOutput, "output": output} {
		if buffer == nil || buffer.Pointer() == 0 || buffer.Count() != hidden || buffer.SizeBytes() != uint64(hidden*4) {
			return core.E(operation, label+" must match the hidden size", nil)
		}
	}
	if downOutput.Pointer() == output.Pointer() {
		return core.E(operation, "down output and accumulated output must not alias", nil)
	}
	for index, entry := range entries {
		if entry == nil || entry.Storage != hipGemma4MoEExpertStorageMLXAffine {
			return core.E(operation, core.Sprintf("selected expert %d is not MLX affine", index), nil)
		}
		if math.IsNaN(float64(routeWeights[index])) || math.IsInf(float64(routeWeights[index]), 0) {
			return core.E(operation, "route weights must be finite", nil)
		}
		if entry.MLXGate.Rows != expertFF || entry.MLXGate.Cols != hidden ||
			entry.MLXUp.Rows != expertFF || entry.MLXUp.Cols != hidden ||
			entry.MLXDown.Rows != hidden || entry.MLXDown.Cols != expertFF {
			return core.E(operation, "selected expert projection geometry mismatch", nil)
		}
	}
	if err := hipMemsetDevice(driver, output.Pointer(), 0, output.SizeBytes()); err != nil {
		return core.E(operation, "clear accumulated expert output", err)
	}
	for index, entry := range entries {
		if err := hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInputOutputWithWorkspace(ctx, driver, input, entry.MLXGate, entry.MLXUp, activation, workspace); err != nil {
			return core.E(operation, core.Sprintf("run expert %d gate/up projection", index), err)
		}
		if err := hipRunMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace(ctx, driver, activation, entry.MLXDown, downOutput, workspace); err != nil {
			return core.E(operation, core.Sprintf("run expert %d down projection", index), err)
		}
		if err := hipRunVectorScaleDeviceKernelOutputWithWorkspace(ctx, driver, downOutput, routeWeights[index], downOutput, workspace); err != nil {
			return core.E(operation, core.Sprintf("scale expert %d output", index), err)
		}
		if err := hipRunVectorAddScaledDeviceKernelOutputWithWorkspace(ctx, driver, output, downOutput, 1, output, workspace); err != nil {
			return core.E(operation, core.Sprintf("accumulate expert %d output", index), err)
		}
	}
	return nil
}

func hipRunGGUFQ4_0SelectedExpertsKernelWithDeviceInputOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, entries []*hipGemma4ExpertCacheEntry, routeWeights []float32, hidden, expertFF int, activation, output *hipDeviceByteBuffer) error {
	return hipRunGGUFQ4_0SelectedExpertsKernelWithDeviceInputOutputWithWorkspace(ctx, driver, input, entries, routeWeights, hidden, expertFF, activation, output, nil)
}

func hipRunGGUFQ4_0SelectedExpertsKernelWithDeviceInputOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, entries []*hipGemma4ExpertCacheEntry, routeWeights []float32, hidden, expertFF int, activation, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "HIP driver is not available", nil)
	}
	if len(entries) == 0 || len(entries) != len(routeWeights) || len(entries) > hipGGUFQ4_0SelectedExpertsMaxTopK {
		return core.E("rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "selected expert entries and route weights must agree", nil)
	}
	if input == nil || activation == nil || output == nil || input.Pointer() == 0 || activation.Pointer() == 0 || output.Pointer() == 0 ||
		input.Count() != hidden || input.SizeBytes() != uint64(hidden*4) ||
		activation.Count() != len(entries)*expertFF || activation.SizeBytes() != uint64(len(entries)*expertFF*4) ||
		output.Count() != hidden || output.SizeBytes() != uint64(hidden*4) {
		return core.E("rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "selected expert input or output buffer shape mismatch", nil)
	}
	args := hipGGUFQ4_0SelectedExpertsLaunchArgs{
		InputPointer: input.Pointer(), ActivationPointer: activation.Pointer(), OutputPointer: output.Pointer(),
		TopK: len(entries), HiddenSize: hidden, ExpertFF: expertFF,
		GateUpRows: 2 * expertFF, DownRows: hidden,
		InputBytes: input.SizeBytes(), ActivationBytes: activation.SizeBytes(), OutputBytes: output.SizeBytes(),
	}
	for index, entry := range entries {
		if entry == nil || entry.GateUp == nil || entry.Down == nil || entry.GateUpRows != 2*expertFF || entry.GateUpCols != hidden || entry.DownRows != hidden || entry.DownCols != expertFF {
			return core.E("rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "selected expert cache entry geometry mismatch", nil)
		}
		gateUpFormat := hipGGUFEffectiveExpertFormat(entry.GateUpFormat)
		downFormat := hipGGUFEffectiveExpertFormat(entry.DownFormat)
		if index == 0 {
			args.GateUpBytes = entry.GateUp.SizeBytes()
			args.DownBytes = entry.Down.SizeBytes()
			args.GateUpFormat = gateUpFormat
			args.DownFormat = downFormat
		} else if args.GateUpBytes != entry.GateUp.SizeBytes() || args.DownBytes != entry.Down.SizeBytes() ||
			args.GateUpFormat != gateUpFormat || args.DownFormat != downFormat {
			return core.E("rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "selected expert cache entry byte counts must agree", nil)
		}
		args.GateUpPointers[index] = entry.GateUp.Pointer()
		args.DownPointers[index] = entry.Down.Pointer()
		args.RouteWeights[index] = routeWeights[index]
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = args.BinaryInto(workspace.MoE.SelectedExpertArgs[:])
	} else {
		launchBytes, err = args.Binary()
	}
	if err != nil {
		return err
	}
	gateRows := uint32(len(entries) * expertFF)
	downRows := uint32(hidden)
	gateRowsPerBlock := hipGGUFQ4_0ProjectionRowsPerBlock
	downRowsPerBlock := hipGGUFQ4_0ProjectionRowsPerBlock
	gateKernel, downKernel := hipKernelNameGGUFQ4_0SelectedExpertGateUp, hipKernelNameGGUFQ4_0SelectedExpertDown
	if args.GateUpFormat == hipGGUFExpertFormatQ4_0 && args.DownFormat == hipGGUFExpertFormatQ4_0 && core.Env(hipGemma4SelectedExpertPair16Env) != "0" {
		gateKernel = hipKernelNameGGUFQ4_0SelectedExpertGateUpPair16
		downKernel = hipKernelNameGGUFQ4_0SelectedExpertDownPair16
		gateRowsPerBlock = hipGGUFQ4_0SelectedExpertsPair16RowsPerBlock
		downRowsPerBlock = hipGGUFQ4_0SelectedExpertsPair16RowsPerBlock
	} else if (args.GateUpFormat == hipGGUFExpertFormatQ4K || args.GateUpFormat == hipGGUFExpertFormatQ4KExpanded) &&
		(args.DownFormat == hipGGUFExpertFormatQ5_1 || args.DownFormat == hipGGUFExpertFormatQ8_0) {
		if args.GateUpFormat == hipGGUFExpertFormatQ4KExpanded {
			gateKernel = hipKernelNameGGUFQ4KExpandedSelectedGateUpSplitPair16
			gateRowsPerBlock = hipGGUFQ4KSelectedExpertsSplitRowsPerBlock
			downRowsPerBlock = hipGGUFQ4_0SelectedExpertsPair16RowsPerBlock
			if args.DownFormat == hipGGUFExpertFormatQ5_1 {
				downKernel = hipKernelNameGGUFQ5_1SelectedExpertDownPair16
				if args.TopK == hipGGUFQ4_0SelectedExpertsMaxTopK && core.Env(hipGemma4SelectedExpertDownExpert8Env) != "0" {
					downKernel = hipKernelNameGGUFQ5_1SelectedExpertDownExpert8Pair16
					downRowsPerBlock = hipGGUFQ5_1SelectedExpertsExpert8RowsPerBlock
				}
			} else {
				downKernel = hipKernelNameGGUFQ8_0SelectedExpertDownPair16
			}
		} else if core.Env(hipGemma4SelectedExpertPair16Env) != "0" {
			gateKernel = hipKernelNameGGUFQ4KSelectedExpertGateUpPair16
			gateRowsPerBlock = hipGGUFQ4_0SelectedExpertsPair16RowsPerBlock
			downRowsPerBlock = hipGGUFQ4_0SelectedExpertsPair16RowsPerBlock
			if core.Env(hipGemma4SelectedExpertGateUpSplitEnv) != "0" {
				gateKernel = hipKernelNameGGUFQ4KSelectedExpertGateUpSplitPair16
				gateRowsPerBlock = hipGGUFQ4KSelectedExpertsSplitRowsPerBlock
			}
			if args.DownFormat == hipGGUFExpertFormatQ5_1 {
				downKernel = hipKernelNameGGUFQ5_1SelectedExpertDownPair16
				if args.TopK == hipGGUFQ4_0SelectedExpertsMaxTopK && core.Env(hipGemma4SelectedExpertDownExpert8Env) != "0" {
					downKernel = hipKernelNameGGUFQ5_1SelectedExpertDownExpert8Pair16
					downRowsPerBlock = hipGGUFQ5_1SelectedExpertsExpert8RowsPerBlock
				}
			} else {
				downKernel = hipKernelNameGGUFQ8_0SelectedExpertDownPair16
			}
		} else {
			gateKernel = hipKernelNameGGUFQ4KSelectedExpertGateUp
			if args.DownFormat == hipGGUFExpertFormatQ5_1 {
				downKernel = hipKernelNameGGUFQ5_1SelectedExpertDown
			} else {
				downKernel = hipKernelNameGGUFQ8_0SelectedExpertDown
			}
		}
	} else if args.GateUpFormat != hipGGUFExpertFormatQ4_0 || args.DownFormat != hipGGUFExpertFormatQ4_0 {
		return core.E("rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "selected expert format pair is unsupported", nil)
	}
	for _, config := range []hipKernelLaunchConfig{
		{Name: gateKernel, Args: launchBytes, GridX: (gateRows + gateRowsPerBlock - 1) / gateRowsPerBlock, GridY: 1, GridZ: 1, BlockX: hipGGUFQ4_0ProjectionBlockSize, BlockY: 1, BlockZ: 1},
		{Name: downKernel, Args: launchBytes, GridX: (downRows + downRowsPerBlock - 1) / downRowsPerBlock, GridY: 1, GridZ: 1, BlockX: hipGGUFQ4_0ProjectionBlockSize, BlockY: 1, BlockZ: 1},
	} {
		if err := hipLaunchKernel(driver, config); err != nil {
			return err
		}
	}
	return nil
}

func hipRunGGUFQ4_0ProjectionKernelWithDeviceInputOutput(ctx context.Context, driver nativeHIPDriver, input, weight *hipDeviceByteBuffer, rows, cols, rowOffset, weightRows int, output *hipDeviceByteBuffer) error {
	return hipRunGGUFQ4_0KernelWithDeviceInputOutput(ctx, driver, hipKernelNameGGUFQ4_0Projection, input, weight, rows, cols, rowOffset, weightRows, output, false)
}

func hipRunGGUFQ4_0GELUTanhGateUpKernelWithDeviceInputOutput(ctx context.Context, driver nativeHIPDriver, input, weight *hipDeviceByteBuffer, rows, cols, rowOffset, weightRows int, output *hipDeviceByteBuffer) error {
	return hipRunGGUFQ4_0KernelWithDeviceInputOutput(ctx, driver, hipKernelNameGGUFQ4_0GELUTanhGateUp, input, weight, rows, cols, rowOffset, weightRows, output, true)
}

func hipRunGGUFQ4_0KernelWithDeviceInputOutput(ctx context.Context, driver nativeHIPDriver, kernelName string, input, weight *hipDeviceByteBuffer, rows, cols, rowOffset, weightRows int, output *hipDeviceByteBuffer, fusedGateUp bool) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.GGUFQ4_0ProjectionLaunch", "HIP driver is not available", nil)
	}
	if input == nil || weight == nil || output == nil || input.Pointer() == 0 || weight.Pointer() == 0 || output.Pointer() == 0 {
		return core.E("rocm.hip.GGUFQ4_0ProjectionLaunch", "input, weight, and output device buffers are required", nil)
	}
	if input.Count() != cols || input.SizeBytes() != uint64(cols*4) || output.Count() != rows || output.SizeBytes() != uint64(rows*4) {
		return core.E("rocm.hip.GGUFQ4_0ProjectionLaunch", "input or output device buffer shape mismatch", nil)
	}
	if fusedGateUp && (rowOffset < 0 || rowOffset+2*rows > weightRows) {
		return core.E("rocm.hip.GGUFQ4_0ProjectionLaunch", "fused gate/up row range exceeds weight rows", nil)
	}
	args := hipGGUFQ4_0ProjectionLaunchArgs{
		InputPointer:  input.Pointer(),
		WeightPointer: weight.Pointer(),
		OutputPointer: output.Pointer(),
		Rows:          rows,
		Cols:          cols,
		RowOffset:     rowOffset,
		WeightRows:    weightRows,
		InputBytes:    input.SizeBytes(),
		WeightBytes:   weight.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
	}
	launchBytes, err := args.Binary()
	if err != nil {
		return err
	}
	rowCount := uint32(rows)
	config := hipKernelLaunchConfig{
		Name:   kernelName,
		Args:   launchBytes,
		GridX:  (rowCount + hipGGUFQ4_0ProjectionRowsPerBlock - 1) / hipGGUFQ4_0ProjectionRowsPerBlock,
		GridY:  1,
		GridZ:  1,
		BlockX: hipGGUFQ4_0ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return hipLaunchKernel(driver, config)
}
