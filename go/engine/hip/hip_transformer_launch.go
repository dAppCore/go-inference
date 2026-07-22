// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
)

const (
	hipRMSNormLaunchArgsVersion                               uint32 = 1
	hipRMSNormLaunchArgsBytes                                        = 64
	hipRMSNormResidualAddArgsVersion                          uint32 = 1
	hipRMSNormResidualAddArgsBytes                                   = 80
	hipRMSNormResAddNormArgsVersion                           uint32 = 1
	hipRMSNormResAddNormArgsBytes                                    = 128
	hipRMSNormHeadsLaunchArgsVersion                          uint32 = 1
	hipRMSNormHeadsLaunchArgsBytes                                   = 64
	hipRMSNormRoPEHeadsLaunchArgsVersion                      uint32 = 2
	hipRMSNormRoPEHeadsLaunchArgsBytes                               = 88
	hipRMSNormRoPEHeadsPairLaunchArgsVersion                  uint32 = 1
	hipRMSNormRoPEHeadsPairLaunchArgsBytes                           = 144
	hipRMSNormRoPEHeadsBatchLaunchArgsVersion                 uint32 = 2
	hipRMSNormRoPEHeadsBatchLaunchArgsBytes                          = 96
	hipRMSNormRoPEHeadsPairLaneBatchLaunchArgsVersion         uint32 = 1
	hipRMSNormRoPEHeadsPairLaneBatchLaunchArgsBytes                  = 152
	hipRoPELaunchArgsVersion                                  uint32 = 1
	hipRoPELaunchArgsBytes                                           = 64
	hipRoPEHeadsLaunchArgsVersion                             uint32 = 1
	hipRoPEHeadsLaunchArgsBytes                                      = 64
	hipGreedyLaunchArgsVersion                                uint32 = 1
	hipGreedyLaunchArgsBytes                                         = 64
	hipSoftcapGreedyLaunchArgsVersion                         uint32 = 2
	hipSoftcapGreedyLaunchArgsBytes                                  = 64
	hipGreedyResultBytes                                             = 8
	hipAttentionLaunchArgsVersion                             uint32 = 1
	hipAttentionLaunchArgsBytes                                      = 104
	hipAttentionHeadsLaunchArgsVersion                        uint32 = 1
	hipAttentionHeadsLaunchArgsBytes                                 = 128
	hipAttentionHeadsBatchCausalLaunchArgsVersion             uint32 = 1
	hipAttentionHeadsBatchCausalLaunchArgsBytes                      = 144
	hipAttentionHeadsLaneBatchLaunchArgsVersion               uint32 = 1
	hipAttentionHeadsLaneBatchLaunchArgsBytes                        = 112
	hipAttentionHeadsLaneDescriptorBytes                             = 32
	hipAttentionHeadsBatchCausalQueryRMSRoPELaunchArgsVersion uint32 = 1
	hipAttentionHeadsBatchCausalQueryRMSRoPELaunchArgsBytes          = 184
	hipAttentionHeadsSharedMaxTokens                                 = 2048
	hipAttentionHeadsChunkedLaunchArgsVersion                 uint32 = 1
	hipAttentionHeadsChunkedLaunchArgsBytes                          = 128
	hipAttentionHeadsBatchChunkedLaunchArgsVersion            uint32 = 2
	hipAttentionHeadsBatchChunkedLaunchArgsBytes                     = 136
	hipAttentionHeadsChunkedBlockSize                                = 512
	hipAttentionHeadsDefaultChunkSize                                = 64
	hipAttentionKVSourceContiguous                            uint32 = 0
	hipAttentionKVSourceDevice                                uint32 = 1
	hipVectorAddLaunchArgsVersion                             uint32 = 1
	hipVectorAddLaunchArgsBytes                                      = 64
	hipVectorAddScaledLaunchArgsVersion                       uint32 = 1
	hipVectorAddScaledLaunchArgsBytes                                = 64
	hipVectorScaleLaunchArgsVersion                           uint32 = 1
	hipVectorScaleLaunchArgsBytes                                    = 64
	hipSwiGLULaunchArgsVersion                                uint32 = 1
	hipSwiGLULaunchArgsBytes                                         = 64
	hipGELUTanhMulLaunchArgsVersion                           uint32 = 1
	hipGELUTanhMulLaunchArgsBytes                                    = 64
	hipTinyPrefillLaunchArgsVersion                           uint32 = 1
	hipTinyPrefillLaunchArgsBytes                                    = 160
	hipTinyDecodeLaunchArgsVersion                            uint32 = 1
	hipTinyDecodeLaunchArgsBytes                                     = 160
)

const (
	hipTinyOutputWeightEncodingFP32     uint32 = 1
	hipTinyOutputWeightEncodingFP16     uint32 = 2
	hipTinyOutputWeightEncodingQ8       uint32 = 3
	hipTinyOutputWeightEncodingJANGTQ   uint32 = 4
	hipTinyOutputWeightEncodingCodebook uint32 = 5
)

const (
	hipRMSNormWeightEncodingNone uint32 = 0
	hipRMSNormWeightEncodingF32  uint32 = 1
	hipRMSNormWeightEncodingBF16 uint32 = 2
)

const (
	hipRMSNormLaunchFlagAddUnitWeight uint32 = 1
	hipRMSNormLaunchFlagRoPENeoX      uint32 = 2
	hipRMSNormLaunchFlagMask                 = hipRMSNormLaunchFlagAddUnitWeight | hipRMSNormLaunchFlagRoPENeoX
)

type hipRMSNormRequest struct {
	Input         []float32
	Weight        []float32
	WeightBF16    []uint16
	Epsilon       float32
	AddUnitWeight bool
}

type hipRMSNormDeviceBuffers struct {
	Input          *hipDeviceByteBuffer
	Weight         *hipDeviceByteBuffer
	Output         *hipDeviceByteBuffer
	Count          int
	Epsilon        float32
	WeightEncoding uint32
	Flags          uint32
}

type hipRMSNormLaunchArgs struct {
	InputPointer   nativeDevicePointer
	WeightPointer  nativeDevicePointer
	OutputPointer  nativeDevicePointer
	Count          int
	InputBytes     uint64
	WeightBytes    uint64
	OutputBytes    uint64
	Epsilon        float32
	WeightEncoding uint32
	Flags          uint32
}

type hipRMSNormResidualAddLaunchArgs struct {
	InputPointer    nativeDevicePointer
	WeightPointer   nativeDevicePointer
	ResidualPointer nativeDevicePointer
	OutputPointer   nativeDevicePointer
	Count           int
	InputBytes      uint64
	WeightBytes     uint64
	ResidualBytes   uint64
	OutputBytes     uint64
	Epsilon         float32
	WeightEncoding  uint32
	Flags           uint32
	OutputScale     float32
}

type hipRMSNormResidualAddNormLaunchArgs struct {
	InputPointer          nativeDevicePointer
	WeightPointer         nativeDevicePointer
	ResidualPointer       nativeDevicePointer
	ResidualOutputPointer nativeDevicePointer
	NormWeightPointer     nativeDevicePointer
	NormOutputPointer     nativeDevicePointer
	Count                 int
	InputBytes            uint64
	WeightBytes           uint64
	ResidualBytes         uint64
	ResidualOutputBytes   uint64
	NormWeightBytes       uint64
	NormOutputBytes       uint64
	Epsilon               float32
	WeightEncoding        uint32
	Flags                 uint32
	NormEpsilon           float32
	NormWeightEncoding    uint32
	NormFlags             uint32
	OutputScale           float32
	Q8OutputBytes         uint64
	Q8OutputPointer       nativeDevicePointer
}

type hipRMSNormHeadsLaunchArgs struct {
	InputPointer   nativeDevicePointer
	WeightPointer  nativeDevicePointer
	OutputPointer  nativeDevicePointer
	HeadDim        int
	HeadCount      int
	InputBytes     uint64
	WeightBytes    uint64
	OutputBytes    uint64
	Epsilon        float32
	WeightEncoding uint32
	Flags          uint32
}

type hipRMSNormRoPEHeadsLaunchArgs struct {
	InputPointer   nativeDevicePointer
	WeightPointer  nativeDevicePointer
	OutputPointer  nativeDevicePointer
	HeadDim        int
	HeadCount      int
	InputBytes     uint64
	WeightBytes    uint64
	OutputBytes    uint64
	Epsilon        float32
	WeightEncoding uint32
	Flags          uint32
	Position       int
	Base           float32
	FrequencyDim   int
	RotaryCount    int
	FrequencyScale float32
}

type hipRMSNormRoPEHeadsPairLaunchArgs struct {
	QueryInputPointer   nativeDevicePointer
	QueryWeightPointer  nativeDevicePointer
	QueryOutputPointer  nativeDevicePointer
	KeyInputPointer     nativeDevicePointer
	KeyWeightPointer    nativeDevicePointer
	KeyOutputPointer    nativeDevicePointer
	HeadDim             int
	QueryHeadCount      int
	KeyHeadCount        int
	QueryInputBytes     uint64
	QueryWeightBytes    uint64
	QueryOutputBytes    uint64
	KeyInputBytes       uint64
	KeyWeightBytes      uint64
	KeyOutputBytes      uint64
	QueryEpsilon        float32
	QueryWeightEncoding uint32
	QueryFlags          uint32
	KeyEpsilon          float32
	KeyWeightEncoding   uint32
	KeyFlags            uint32
	Position            int
	Base                float32
	FrequencyDim        int
	RotaryCount         int
	FrequencyScale      float32
}

type hipRMSNormRoPEHeadsBatchLaunchArgs struct {
	InputPointer   nativeDevicePointer
	WeightPointer  nativeDevicePointer
	OutputPointer  nativeDevicePointer
	HeadDim        int
	HeadCount      int
	Batch          int
	InputBytes     uint64
	WeightBytes    uint64
	OutputBytes    uint64
	Epsilon        float32
	WeightEncoding uint32
	Flags          uint32
	StartPosition  int
	Base           float32
	FrequencyDim   int
	RotaryCount    int
	FrequencyScale float32
}

type hipRMSNormRoPEHeadsPairLaneBatchLaunchArgs struct {
	QueryInputPointer   nativeDevicePointer
	QueryWeightPointer  nativeDevicePointer
	QueryOutputPointer  nativeDevicePointer
	KeyInputPointer     nativeDevicePointer
	KeyWeightPointer    nativeDevicePointer
	KeyOutputPointer    nativeDevicePointer
	PositionsPointer    nativeDevicePointer
	HeadDim             int
	QueryHeadCount      int
	KeyHeadCount        int
	Batch               int
	QueryInputBytes     uint64
	QueryWeightBytes    uint64
	QueryOutputBytes    uint64
	KeyInputBytes       uint64
	KeyWeightBytes      uint64
	KeyOutputBytes      uint64
	QueryEpsilon        float32
	QueryWeightEncoding uint32
	QueryFlags          uint32
	KeyEpsilon          float32
	KeyWeightEncoding   uint32
	KeyFlags            uint32
	PositionsBytes      uint64
	Base                float32
	FrequencyDim        int
	RotaryCount         int
	FrequencyScale      float32
}

type hipRoPERequest struct {
	Input        []float32
	Position     int
	Base         float32
	FrequencyDim int
	RotaryCount  int
}

type hipRoPEDeviceBuffers struct {
	Input        *hipDeviceByteBuffer
	Output       *hipDeviceByteBuffer
	Count        int
	Position     int
	Base         float32
	FrequencyDim int
	RotaryCount  int
}

type hipRoPELaunchArgs struct {
	InputPointer  nativeDevicePointer
	OutputPointer nativeDevicePointer
	Count         int
	InputBytes    uint64
	OutputBytes   uint64
	Position      int
	Base          float32
	FrequencyDim  int
	RotaryCount   int
}

type hipRoPEHeadsLaunchArgs struct {
	InputPointer  nativeDevicePointer
	OutputPointer nativeDevicePointer
	HeadDim       int
	HeadCount     int
	InputBytes    uint64
	OutputBytes   uint64
	Position      int
	Base          float32
	FrequencyDim  int
	RotaryCount   int
}

type hipGreedySampleRequest struct {
	Logits []float32
}

type hipGreedySampleDeviceBuffers struct {
	Logits *hipDeviceByteBuffer
	Output *hipDeviceByteBuffer
	Count  int
}

type hipGreedySampleLaunchArgs struct {
	LogitsPointer nativeDevicePointer
	OutputPointer nativeDevicePointer
	Count         int
	LogitsBytes   uint64
	OutputBytes   uint64
}

type hipSoftcapGreedySampleLaunchArgs struct {
	LogitsPointer   nativeDevicePointer
	OutputPointer   nativeDevicePointer
	Count           int
	LogitsBytes     uint64
	OutputBytes     uint64
	Softcap         float32
	SuppressPointer nativeDevicePointer
	SuppressCount   int
}

type hipGreedySampleResult struct {
	TokenID int
	Score   float32
}

type hipAttentionRequest struct {
	Query           []float32
	QueryDim        int
	KeyHeads        int
	Keys            []float32
	Values          []float32
	DeviceKV        *rocmDeviceKVCache
	DescriptorTable *rocmDeviceKVDescriptorTable
	WindowSize      int
	Scale           float32
}

type hipAttentionDeviceBuffers struct {
	Query      *hipDeviceByteBuffer
	Keys       *hipDeviceByteBuffer
	Values     *hipDeviceByteBuffer
	Output     *hipDeviceByteBuffer
	Weights    *hipDeviceByteBuffer
	Dim        int
	TokenCount int
}

type hipAttentionLaunchArgs struct {
	QueryPointer      nativeDevicePointer
	KeyPointer        nativeDevicePointer
	ValuePointer      nativeDevicePointer
	OutputPointer     nativeDevicePointer
	WeightPointer     nativeDevicePointer
	Dim               int
	TokenCount        int
	QueryBytes        uint64
	KeyBytes          uint64
	ValueBytes        uint64
	OutputBytes       uint64
	WeightBytes       uint64
	KVSource          uint32
	Scale             float32
	DescriptorPointer nativeDevicePointer
	DescriptorBytes   uint64
}

type hipAttentionHeadsLaunchArgs struct {
	QueryPointer      nativeDevicePointer
	KeyPointer        nativeDevicePointer
	ValuePointer      nativeDevicePointer
	OutputPointer     nativeDevicePointer
	WeightPointer     nativeDevicePointer
	Dim               int
	TokenCount        int
	HeadCount         int
	KeyHeads          int
	QueryBytes        uint64
	KeyBytes          uint64
	ValueBytes        uint64
	OutputBytes       uint64
	WeightBytes       uint64
	KVSource          uint32
	Scale             float32
	DescriptorPointer nativeDevicePointer
	DescriptorBytes   uint64
	SharedMemBytes    uint64
	WindowSize        int
}

type hipAttentionHeadsBatchCausalLaunchArgs struct {
	QueryPointer      nativeDevicePointer
	KeyPointer        nativeDevicePointer
	ValuePointer      nativeDevicePointer
	OutputPointer     nativeDevicePointer
	WeightPointer     nativeDevicePointer
	Dim               int
	TokenCount        int
	HeadCount         int
	KeyHeads          int
	QueryCount        int
	QueryStartToken   int
	QueryBytes        uint64
	KeyBytes          uint64
	ValueBytes        uint64
	OutputBytes       uint64
	WeightBytes       uint64
	KVSource          uint32
	Scale             float32
	DescriptorPointer nativeDevicePointer
	DescriptorBytes   uint64
	SharedMemBytes    uint64
	WindowSize        int
	VisibleCapPointer nativeDevicePointer
	VisibleCapBytes   uint64
}

type hipAttentionHeadsLaneBatchLaunchArgs struct {
	QueryPointer   nativeDevicePointer
	OutputPointer  nativeDevicePointer
	WeightPointer  nativeDevicePointer
	LanePointer    nativeDevicePointer
	Dim            int
	LaneCount      int
	HeadCount      int
	KeyHeads       int
	MaxTokenCount  int
	QueryBytes     uint64
	OutputBytes    uint64
	WeightBytes    uint64
	LaneBytes      uint64
	Scale          float32
	SharedMemBytes uint64
}

type hipAttentionHeadsBatchCausalQueryRMSRoPELaunchArgs struct {
	QueryInputPointer      nativeDevicePointer
	QueryWeightPointer     nativeDevicePointer
	KeyPointer             nativeDevicePointer
	ValuePointer           nativeDevicePointer
	OutputPointer          nativeDevicePointer
	AttentionWeightPointer nativeDevicePointer
	Dim                    int
	TokenCount             int
	HeadCount              int
	KeyHeads               int
	QueryCount             int
	QueryStartToken        int
	QueryInputBytes        uint64
	QueryWeightBytes       uint64
	KeyBytes               uint64
	ValueBytes             uint64
	OutputBytes            uint64
	AttentionWeightBytes   uint64
	KVSource               uint32
	Scale                  float32
	DescriptorPointer      nativeDevicePointer
	DescriptorBytes        uint64
	SharedMemBytes         uint64
	WindowSize             int
	QueryEpsilon           float32
	QueryWeightEncoding    uint32
	QueryFlags             uint32
	RoPEStartPosition      int
	RoPEBase               float32
	RoPEFrequencyDim       int
	RoPERotaryCount        int
	RoPEFrequencyScale     float32
}

type hipAttentionHeadsChunkedLaunchArgs struct {
	QueryPointer      nativeDevicePointer
	DescriptorPointer nativeDevicePointer
	PartialPointer    nativeDevicePointer
	StatsPointer      nativeDevicePointer
	OutputPointer     nativeDevicePointer
	Dim               int
	TokenCount        int
	HeadCount         int
	KeyHeads          int
	ChunkSize         int
	ChunkCount        int
	QueryBytes        uint64
	DescriptorBytes   uint64
	PartialBytes      uint64
	StatsBytes        uint64
	OutputBytes       uint64
	Scale             float32
	WindowSize        int
}

type hipAttentionHeadsBatchChunkedLaunchArgs struct {
	QueryPointer      nativeDevicePointer
	DescriptorPointer nativeDevicePointer
	PartialPointer    nativeDevicePointer
	StatsPointer      nativeDevicePointer
	OutputPointer     nativeDevicePointer
	VisibleCapPointer nativeDevicePointer
	Dim               int
	TokenCount        int
	HeadCount         int
	KeyHeads          int
	QueryCount        int
	QueryStartToken   int
	ChunkSize         int
	ChunkCount        int
	QueryBytes        uint64
	DescriptorBytes   uint64
	PartialBytes      uint64
	StatsBytes        uint64
	OutputBytes       uint64
	VisibleCapBytes   uint64
	Scale             float32
	WindowSize        int
	ChunkStartToken   int
}

type hipAttentionResult struct {
	Output  []float32
	Weights []float32
}

type hipVectorAddRequest struct {
	Left  []float32
	Right []float32
}

type hipVectorAddDeviceBuffers struct {
	Left   *hipDeviceByteBuffer
	Right  *hipDeviceByteBuffer
	Output *hipDeviceByteBuffer
	Count  int
}

type hipVectorAddLaunchArgs struct {
	LeftPointer   nativeDevicePointer
	RightPointer  nativeDevicePointer
	OutputPointer nativeDevicePointer
	Count         int
	LeftBytes     uint64
	RightBytes    uint64
	OutputBytes   uint64
}

type hipVectorAddScaledLaunchArgs struct {
	LeftPointer   nativeDevicePointer
	RightPointer  nativeDevicePointer
	OutputPointer nativeDevicePointer
	Count         int
	LeftBytes     uint64
	RightBytes    uint64
	OutputBytes   uint64
	Scale         float32
}

type hipVectorScaleRequest struct {
	Input []float32
	Scale float32
}

type hipVectorScaleDeviceBuffers struct {
	Input  *hipDeviceByteBuffer
	Output *hipDeviceByteBuffer
	Count  int
	Scale  float32
}

type hipVectorScaleLaunchArgs struct {
	InputPointer  nativeDevicePointer
	OutputPointer nativeDevicePointer
	Count         int
	InputBytes    uint64
	OutputBytes   uint64
	Scale         float32
}

type hipSwiGLURequest struct {
	Gate []float32
	Up   []float32
}

type hipGELUTanhMultiplyRequest struct {
	Gate []float32
	Up   []float32
}

type hipSwiGLUDeviceBuffers struct {
	Gate   *hipDeviceByteBuffer
	Up     *hipDeviceByteBuffer
	Output *hipDeviceByteBuffer
	Count  int
}

type hipGELUTanhMultiplyDeviceBuffers struct {
	Gate   *hipDeviceByteBuffer
	Up     *hipDeviceByteBuffer
	Output *hipDeviceByteBuffer
	Count  int
}

type hipSwiGLULaunchArgs struct {
	GatePointer   nativeDevicePointer
	UpPointer     nativeDevicePointer
	OutputPointer nativeDevicePointer
	Count         int
	GateBytes     uint64
	UpBytes       uint64
	OutputBytes   uint64
}

type hipGELUTanhMultiplyLaunchArgs struct {
	GatePointer   nativeDevicePointer
	UpPointer     nativeDevicePointer
	OutputPointer nativeDevicePointer
	Count         int
	GateBytes     uint64
	UpBytes       uint64
	OutputBytes   uint64
}

type hipTinyPrefillRequest struct {
	TokenIDs       []int32
	EmbeddingTable []float32
	OutputWeights  []float32
	OutputFP16     []uint16
	OutputQ8       []int8
	Q8Scale        float32
	VocabSize      int
	HiddenSize     int
}

type hipTinyPrefillDeviceBuffers struct {
	Tokens         *hipDeviceByteBuffer
	EmbeddingTable *hipDeviceByteBuffer
	OutputWeights  *hipDeviceByteBuffer
	Logits         *hipDeviceByteBuffer
	Attention      *hipDeviceByteBuffer
	Keys           *hipDeviceByteBuffer
	Values         *hipDeviceByteBuffer
	Result         *hipDeviceByteBuffer
	TokenCount     int
	VocabSize      int
	HiddenSize     int
}

type hipTinyPrefillLaunchArgs struct {
	TokenPointer         nativeDevicePointer
	EmbeddingPointer     nativeDevicePointer
	OutputWeightPointer  nativeDevicePointer
	LogitPointer         nativeDevicePointer
	AttentionPointer     nativeDevicePointer
	ResultPointer        nativeDevicePointer
	KeyPointer           nativeDevicePointer
	ValuePointer         nativeDevicePointer
	TokenCount           int
	VocabSize            int
	HiddenSize           int
	TokenBytes           uint64
	EmbeddingBytes       uint64
	OutputWeightBytes    uint64
	LogitBytes           uint64
	AttentionBytes       uint64
	ResultBytes          uint64
	KeyBytes             uint64
	ValueBytes           uint64
	OutputWeightEncoding uint32
	Q8Scale              float32
}

type hipTinyPrefillResult struct {
	Logits      []float32
	Attention   []float32
	StateKeys   []float32
	StateValues []float32
	NextTokenID int
	NextScore   float32
}

type hipTinyDecodeRequest struct {
	TokenID        int32
	PriorKeys      []float32
	PriorValues    []float32
	EmbeddingTable []float32
	OutputWeights  []float32
	OutputFP16     []uint16
	OutputQ8       []int8
	Q8Scale        float32
	VocabSize      int
	HiddenSize     int
}

type hipTinyDecodeDeviceBuffers struct {
	PriorKeys       *hipDeviceByteBuffer
	PriorValues     *hipDeviceByteBuffer
	EmbeddingTable  *hipDeviceByteBuffer
	OutputWeights   *hipDeviceByteBuffer
	Logits          *hipDeviceByteBuffer
	Attention       *hipDeviceByteBuffer
	UpdatedKeys     *hipDeviceByteBuffer
	UpdatedValues   *hipDeviceByteBuffer
	Result          *hipDeviceByteBuffer
	PriorTokenCount int
	VocabSize       int
	HiddenSize      int
}

type hipTinyDecodeLaunchArgs struct {
	PriorKeyPointer      nativeDevicePointer
	PriorValuePointer    nativeDevicePointer
	EmbeddingPointer     nativeDevicePointer
	OutputWeightPointer  nativeDevicePointer
	LogitPointer         nativeDevicePointer
	AttentionPointer     nativeDevicePointer
	UpdatedKeyPointer    nativeDevicePointer
	UpdatedValuePointer  nativeDevicePointer
	ResultPointer        nativeDevicePointer
	TokenID              int32
	PriorTokenCount      int
	VocabSize            int
	HiddenSize           int
	PriorKeyBytes        uint64
	PriorValueBytes      uint64
	EmbeddingBytes       uint64
	OutputWeightBytes    uint64
	LogitBytes           uint64
	AttentionBytes       uint64
	UpdatedKeyBytes      uint64
	UpdatedValueBytes    uint64
	ResultBytes          uint64
	OutputWeightEncoding uint32
	Q8Scale              float32
}

type hipTinyDecodeResult struct {
	Logits        []float32
	Attention     []float32
	UpdatedKeys   []float32
	UpdatedValues []float32
	NextTokenID   int
	NextScore     float32
}

func (req hipRMSNormRequest) validate() error {
	if len(req.Input) == 0 {
		return core.E("rocm.hip.RMSNormLaunch", "input is required", nil)
	}
	encodings := 0
	if len(req.Weight) > 0 {
		encodings++
	}
	if len(req.WeightBF16) > 0 {
		encodings++
	}
	if encodings != 1 {
		return core.E("rocm.hip.RMSNormLaunch", "exactly one RMSNorm weight encoding is required", nil)
	}
	if len(req.Weight) > 0 && len(req.Weight) != len(req.Input) {
		return core.E("rocm.hip.RMSNormLaunch", "weight length must match input length", nil)
	}
	if len(req.WeightBF16) > 0 && len(req.WeightBF16) != len(req.Input) {
		return core.E("rocm.hip.RMSNormLaunch", "bf16 weight length must match input length", nil)
	}
	if req.Epsilon < 0 || math.IsNaN(float64(req.Epsilon)) || math.IsInf(float64(req.Epsilon), 0) {
		return core.E("rocm.hip.RMSNormLaunch", "epsilon must be non-negative and finite", nil)
	}
	return nil
}

func (req hipRMSNormRequest) deviceBuffers(driver nativeHIPDriver) (*hipRMSNormDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	inputPayload, err := hipFloat32Payload(req.Input)
	if err != nil {
		return nil, core.E("rocm.hip.RMSNormLaunch", "encode input", err)
	}
	input, err := hipUploadByteBuffer(driver, "rocm.hip.RMSNormLaunch", "rms norm input", inputPayload, len(req.Input))
	if err != nil {
		return nil, err
	}
	var flags uint32
	if req.AddUnitWeight {
		flags |= hipRMSNormLaunchFlagAddUnitWeight
	}
	buffers := &hipRMSNormDeviceBuffers{Input: input, Count: len(req.Input), Epsilon: req.Epsilon, Flags: flags}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	switch {
	case len(req.Weight) > 0:
		weightPayload, err := hipFloat32Payload(req.Weight)
		if err != nil {
			return nil, core.E("rocm.hip.RMSNormLaunch", "encode weight", err)
		}
		weight, err := hipUploadByteBuffer(driver, "rocm.hip.RMSNormLaunch", "rms norm weight", weightPayload, len(req.Weight))
		if err != nil {
			return nil, err
		}
		buffers.Weight = weight
		buffers.WeightEncoding = hipRMSNormWeightEncodingF32
	case len(req.WeightBF16) > 0:
		weightPayload, err := hipUint16Payload(req.WeightBF16)
		if err != nil {
			return nil, core.E("rocm.hip.RMSNormLaunch", "encode bf16 weight", err)
		}
		weight, err := hipUploadByteBuffer(driver, "rocm.hip.RMSNormLaunch", "rms norm bf16 weight", weightPayload, len(req.WeightBF16))
		if err != nil {
			return nil, err
		}
		buffers.Weight = weight
		buffers.WeightEncoding = hipRMSNormWeightEncodingBF16
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormLaunch", "rms norm output", uint64(len(req.Input)*4), len(req.Input))
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipRMSNormRequest) launchArgs(buffers *hipRMSNormDeviceBuffers) (hipRMSNormLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipRMSNormLaunchArgs{}, err
	}
	if buffers == nil || buffers.Input == nil || buffers.Weight == nil || buffers.Output == nil {
		return hipRMSNormLaunchArgs{}, core.E("rocm.hip.RMSNormLaunch", "rms norm device buffers are required", nil)
	}
	encoding, err := hipRMSNormWeightEncoding(req)
	if err != nil {
		return hipRMSNormLaunchArgs{}, err
	}
	var flags uint32
	if req.AddUnitWeight {
		flags |= hipRMSNormLaunchFlagAddUnitWeight
	}
	if buffers.Input.Count() != len(req.Input) || buffers.Weight.Count() != len(req.Input) || buffers.Output.Count() != len(req.Input) || buffers.WeightEncoding != encoding || buffers.Flags != flags {
		return hipRMSNormLaunchArgs{}, core.E("rocm.hip.RMSNormLaunch", "rms norm device buffer shape mismatch", nil)
	}
	return hipRMSNormLaunchArgs{
		InputPointer:   buffers.Input.Pointer(),
		WeightPointer:  buffers.Weight.Pointer(),
		OutputPointer:  buffers.Output.Pointer(),
		Count:          len(req.Input),
		InputBytes:     buffers.Input.SizeBytes(),
		WeightBytes:    buffers.Weight.SizeBytes(),
		OutputBytes:    buffers.Output.SizeBytes(),
		Epsilon:        req.Epsilon,
		WeightEncoding: encoding,
		Flags:          flags,
	}, nil
}

func (args hipRMSNormLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipRMSNormLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.RMSNormLaunch", "input and output pointers are required", nil)
	}
	count, err := rocmDeviceKVPositiveUint32("count", args.Count)
	if err != nil {
		return nil, err
	}
	if args.Epsilon < 0 || math.IsNaN(float64(args.Epsilon)) || math.IsInf(float64(args.Epsilon), 0) {
		return nil, core.E("rocm.hip.RMSNormLaunch", "epsilon must be non-negative and finite", nil)
	}
	inputBytes, err := hipAlignedFloat32Bytes("input", args.InputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.RMSNormLaunch", "input byte count", err)
	}
	encoding := args.WeightEncoding
	var weightBytes uint32
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		if args.Flags != 0 {
			return nil, core.E("rocm.hip.RMSNormLaunch", "unit RMSNorm weight does not support flags", nil)
		}
		if args.WeightPointer != 0 || args.WeightBytes != 0 {
			return nil, core.E("rocm.hip.RMSNormLaunch", "unit RMSNorm weight must not provide a weight pointer", nil)
		}
	case hipRMSNormWeightEncodingF32:
		if args.WeightPointer == 0 {
			return nil, core.E("rocm.hip.RMSNormLaunch", "RMSNorm weight pointer is required", nil)
		}
		weightBytes, err = hipAlignedFloat32Bytes("weight", args.WeightBytes, count)
		if err != nil {
			return nil, core.E("rocm.hip.RMSNormLaunch", "weight byte count", err)
		}
	case hipRMSNormWeightEncodingBF16:
		if args.WeightPointer == 0 {
			return nil, core.E("rocm.hip.RMSNormLaunch", "RMSNorm weight pointer is required", nil)
		}
		weightBytes, err = hipExactUint32Bytes("bf16 weight", args.WeightBytes, uint64(count)*2)
		if err != nil {
			return nil, core.E("rocm.hip.RMSNormLaunch", "bf16 weight byte count", err)
		}
	default:
		return nil, core.E("rocm.hip.RMSNormLaunch", core.Sprintf("unsupported RMSNorm weight encoding %d", encoding), nil)
	}
	outputBytes, err := hipAlignedFloat32Bytes("output", args.OutputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.RMSNormLaunch", "output byte count", err)
	}
	if cap(payload) < hipRMSNormLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipRMSNormLaunchArgsBytes)
	} else {
		payload = payload[:hipRMSNormLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipRMSNormLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], count)
	binary.LittleEndian.PutUint32(payload[36:], inputBytes)
	binary.LittleEndian.PutUint32(payload[40:], weightBytes)
	binary.LittleEndian.PutUint32(payload[44:], outputBytes)
	binary.LittleEndian.PutUint32(payload[48:], math.Float32bits(args.Epsilon))
	binary.LittleEndian.PutUint32(payload[52:], encoding)
	binary.LittleEndian.PutUint32(payload[56:], args.Flags)
	return payload, nil
}

func (args hipRMSNormResidualAddLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipRMSNormResidualAddLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.ResidualPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "input, residual, and output pointers are required", nil)
	}
	count, err := rocmDeviceKVPositiveUint32("count", args.Count)
	if err != nil {
		return nil, err
	}
	if args.Epsilon < 0 || math.IsNaN(float64(args.Epsilon)) || math.IsInf(float64(args.Epsilon), 0) {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "epsilon must be non-negative and finite", nil)
	}
	if math.IsNaN(float64(args.OutputScale)) || math.IsInf(float64(args.OutputScale), 0) {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "output scale must be finite", nil)
	}
	inputBytes, err := hipAlignedFloat32Bytes("input", args.InputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "input byte count", err)
	}
	encoding := args.WeightEncoding
	var weightBytes uint32
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		if args.Flags != 0 {
			return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "unit RMSNorm weight does not support flags", nil)
		}
		if args.WeightPointer != 0 || args.WeightBytes != 0 {
			return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "unit RMSNorm weight must not provide a weight pointer", nil)
		}
	case hipRMSNormWeightEncodingF32:
		if args.WeightPointer == 0 {
			return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "RMSNorm weight pointer is required", nil)
		}
		weightBytes, err = hipAlignedFloat32Bytes("weight", args.WeightBytes, count)
		if err != nil {
			return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "weight byte count", err)
		}
	case hipRMSNormWeightEncodingBF16:
		if args.WeightPointer == 0 {
			return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "RMSNorm weight pointer is required", nil)
		}
		weightBytes, err = hipExactUint32Bytes("bf16 weight", args.WeightBytes, uint64(count)*2)
		if err != nil {
			return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "bf16 weight byte count", err)
		}
	default:
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", core.Sprintf("unsupported RMSNorm weight encoding %d", encoding), nil)
	}
	residualBytes, err := hipAlignedFloat32Bytes("residual", args.ResidualBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "residual byte count", err)
	}
	outputBytes, err := hipAlignedFloat32Bytes("output", args.OutputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "output byte count", err)
	}
	if cap(payload) < hipRMSNormResidualAddArgsBytes {
		payload = hipBorrowLaunchPacket(hipRMSNormResidualAddArgsBytes)
	} else {
		payload = payload[:hipRMSNormResidualAddArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipRMSNormResidualAddArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ResidualPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[40:], count)
	binary.LittleEndian.PutUint32(payload[44:], inputBytes)
	binary.LittleEndian.PutUint32(payload[48:], weightBytes)
	binary.LittleEndian.PutUint32(payload[52:], residualBytes)
	binary.LittleEndian.PutUint32(payload[56:], outputBytes)
	binary.LittleEndian.PutUint32(payload[60:], math.Float32bits(args.Epsilon))
	binary.LittleEndian.PutUint32(payload[64:], encoding)
	binary.LittleEndian.PutUint32(payload[68:], args.Flags)
	if args.OutputScale != 0 && args.OutputScale != 1 {
		binary.LittleEndian.PutUint32(payload[72:], math.Float32bits(args.OutputScale))
	}
	return payload, nil
}

func (args hipRMSNormResidualAddNormLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipRMSNormResidualAddNormLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.ResidualPointer == 0 || args.ResidualOutputPointer == 0 || args.NormOutputPointer == 0 {
		return nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "input, residual, residual output, and norm output pointers are required", nil)
	}
	count, err := rocmDeviceKVPositiveUint32("count", args.Count)
	if err != nil {
		return nil, err
	}
	if args.Epsilon < 0 || math.IsNaN(float64(args.Epsilon)) || math.IsInf(float64(args.Epsilon), 0) {
		return nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "epsilon must be non-negative and finite", nil)
	}
	if args.NormEpsilon < 0 || math.IsNaN(float64(args.NormEpsilon)) || math.IsInf(float64(args.NormEpsilon), 0) {
		return nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "norm epsilon must be non-negative and finite", nil)
	}
	if math.IsNaN(float64(args.OutputScale)) || math.IsInf(float64(args.OutputScale), 0) {
		return nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "output scale must be finite", nil)
	}
	inputBytes, err := hipAlignedFloat32Bytes("input", args.InputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "input byte count", err)
	}
	weightBytes, err := hipRMSNormLaunchWeightBytes("RMSNormResidualAddNormLaunch", "weight", args.WeightPointer, args.WeightBytes, count, args.WeightEncoding, args.Flags)
	if err != nil {
		return nil, err
	}
	residualBytes, err := hipAlignedFloat32Bytes("residual", args.ResidualBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "residual byte count", err)
	}
	residualOutputBytes, err := hipAlignedFloat32Bytes("residual output", args.ResidualOutputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "residual output byte count", err)
	}
	normWeightBytes, err := hipRMSNormLaunchWeightBytes("RMSNormResidualAddNormLaunch", "norm weight", args.NormWeightPointer, args.NormWeightBytes, count, args.NormWeightEncoding, args.NormFlags)
	if err != nil {
		return nil, err
	}
	normOutputBytes, err := hipAlignedFloat32Bytes("norm output", args.NormOutputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "norm output byte count", err)
	}
	var q8OutputBytes uint32
	if args.Q8OutputPointer != 0 || args.Q8OutputBytes != 0 {
		if args.Q8OutputPointer == 0 || args.Q8OutputBytes == 0 {
			return nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "Q8_1 output pointer and byte count must be provided together", nil)
		}
		if count%hipQ8_1BlockSize != 0 {
			return nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "Q8_1 output requires a block-aligned count", nil)
		}
		q8OutputBytes, err = hipExactUint32Bytes("Q8_1 output", args.Q8OutputBytes, uint64(count/hipQ8_1BlockSize)*hipQ8_1BlockBytes)
		if err != nil {
			return nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "Q8_1 output byte count", err)
		}
	}
	if cap(payload) < hipRMSNormResAddNormArgsBytes {
		payload = hipBorrowLaunchPacket(hipRMSNormResAddNormArgsBytes)
	} else {
		payload = payload[:hipRMSNormResAddNormArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipRMSNormResAddNormArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ResidualPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.ResidualOutputPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.NormWeightPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.NormOutputPointer))
	binary.LittleEndian.PutUint32(payload[56:], count)
	binary.LittleEndian.PutUint32(payload[60:], inputBytes)
	binary.LittleEndian.PutUint32(payload[64:], weightBytes)
	binary.LittleEndian.PutUint32(payload[68:], residualBytes)
	binary.LittleEndian.PutUint32(payload[72:], residualOutputBytes)
	binary.LittleEndian.PutUint32(payload[76:], normWeightBytes)
	binary.LittleEndian.PutUint32(payload[80:], normOutputBytes)
	binary.LittleEndian.PutUint32(payload[84:], math.Float32bits(args.Epsilon))
	binary.LittleEndian.PutUint32(payload[88:], args.WeightEncoding)
	binary.LittleEndian.PutUint32(payload[92:], args.Flags)
	binary.LittleEndian.PutUint32(payload[96:], math.Float32bits(args.NormEpsilon))
	binary.LittleEndian.PutUint32(payload[100:], args.NormWeightEncoding)
	binary.LittleEndian.PutUint32(payload[104:], args.NormFlags)
	if args.OutputScale != 0 && args.OutputScale != 1 {
		binary.LittleEndian.PutUint32(payload[108:], math.Float32bits(args.OutputScale))
	}
	binary.LittleEndian.PutUint32(payload[112:], q8OutputBytes)
	binary.LittleEndian.PutUint64(payload[120:], uint64(args.Q8OutputPointer))
	return payload, nil
}

func hipRMSNormLaunchWeightBytes(operation, label string, pointer nativeDevicePointer, bytes uint64, count uint32, encoding uint32, flags uint32) (uint32, error) {
	if flags&^hipRMSNormLaunchFlagMask != 0 {
		return 0, core.E("rocm.hip."+operation, "unsupported RMSNorm weight flags", nil)
	}
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		if flags&hipRMSNormLaunchFlagAddUnitWeight != 0 {
			return 0, core.E("rocm.hip."+operation, "unit RMSNorm weight does not support flags", nil)
		}
		if pointer != 0 || bytes != 0 {
			return 0, core.E("rocm.hip."+operation, "unit RMSNorm weight must not provide a weight pointer", nil)
		}
		return 0, nil
	case hipRMSNormWeightEncodingF32:
		if pointer == 0 {
			return 0, core.E("rocm.hip."+operation, "RMSNorm weight pointer is required", nil)
		}
		weightBytes, err := hipAlignedFloat32Bytes(label, bytes, count)
		if err != nil {
			return 0, core.E("rocm.hip."+operation, label+" byte count", err)
		}
		return weightBytes, nil
	case hipRMSNormWeightEncodingBF16:
		if pointer == 0 {
			return 0, core.E("rocm.hip."+operation, "RMSNorm weight pointer is required", nil)
		}
		weightBytes, err := hipExactUint32Bytes(label, bytes, uint64(count)*2)
		if err != nil {
			return 0, core.E("rocm.hip."+operation, label+" byte count", err)
		}
		return weightBytes, nil
	default:
		return 0, core.E("rocm.hip."+operation, core.Sprintf("unsupported RMSNorm weight encoding %d", encoding), nil)
	}
}

func (args hipRMSNormHeadsLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipRMSNormHeadsLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "input and output pointers are required", nil)
	}
	headDim, err := rocmDeviceKVPositiveUint32("head dim", args.HeadDim)
	if err != nil {
		return nil, err
	}
	headCount, err := rocmDeviceKVPositiveUint32("head count", args.HeadCount)
	if err != nil {
		return nil, err
	}
	if args.Epsilon < 0 || math.IsNaN(float64(args.Epsilon)) || math.IsInf(float64(args.Epsilon), 0) {
		return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "epsilon must be non-negative and finite", nil)
	}
	totalCount := uint64(headDim) * uint64(headCount)
	if totalCount > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "total count is out of range", nil)
	}
	inputBytes, err := hipAlignedFloat32Bytes("input", args.InputBytes, uint32(totalCount))
	if err != nil {
		return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "input byte count", err)
	}
	encoding := args.WeightEncoding
	var weightBytes uint32
	switch encoding {
	case hipRMSNormWeightEncodingNone:
		if args.Flags != 0 {
			return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "unit RMSNorm weight does not support flags", nil)
		}
		if args.WeightPointer != 0 || args.WeightBytes != 0 {
			return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "unit RMSNorm weight must not provide a weight pointer", nil)
		}
	case hipRMSNormWeightEncodingF32:
		if args.WeightPointer == 0 {
			return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "RMSNorm weight pointer is required", nil)
		}
		weightBytes, err = hipAlignedFloat32Bytes("weight", args.WeightBytes, headDim)
		if err != nil {
			return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "weight byte count", err)
		}
	case hipRMSNormWeightEncodingBF16:
		if args.WeightPointer == 0 {
			return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "RMSNorm weight pointer is required", nil)
		}
		weightBytes, err = hipExactUint32Bytes("bf16 weight", args.WeightBytes, uint64(headDim)*2)
		if err != nil {
			return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "bf16 weight byte count", err)
		}
	default:
		return nil, core.E("rocm.hip.RMSNormHeadsLaunch", core.Sprintf("unsupported RMSNorm weight encoding %d", encoding), nil)
	}
	outputBytes, err := hipAlignedFloat32Bytes("output", args.OutputBytes, uint32(totalCount))
	if err != nil {
		return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "output byte count", err)
	}
	if cap(payload) < hipRMSNormHeadsLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipRMSNormHeadsLaunchArgsBytes)
	} else {
		payload = payload[:hipRMSNormHeadsLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipRMSNormHeadsLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], headDim)
	binary.LittleEndian.PutUint32(payload[36:], headCount)
	binary.LittleEndian.PutUint32(payload[40:], inputBytes)
	binary.LittleEndian.PutUint32(payload[44:], weightBytes)
	binary.LittleEndian.PutUint32(payload[48:], outputBytes)
	binary.LittleEndian.PutUint32(payload[52:], math.Float32bits(args.Epsilon))
	binary.LittleEndian.PutUint32(payload[56:], encoding)
	binary.LittleEndian.PutUint32(payload[60:], args.Flags)
	return payload, nil
}

func (args hipRMSNormRoPEHeadsLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipRMSNormRoPEHeadsLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	const operation = "RMSNormRoPEHeadsLaunch"
	if args.InputPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip."+operation, "input and output pointers are required", nil)
	}
	headDim, err := rocmDeviceKVPositiveUint32("head dim", args.HeadDim)
	if err != nil {
		return nil, err
	}
	if headDim%2 != 0 {
		return nil, core.E("rocm.hip."+operation, "head dim must be even", nil)
	}
	headCount, err := rocmDeviceKVPositiveUint32("head count", args.HeadCount)
	if err != nil {
		return nil, err
	}
	if args.Epsilon < 0 || math.IsNaN(float64(args.Epsilon)) || math.IsInf(float64(args.Epsilon), 0) {
		return nil, core.E("rocm.hip."+operation, "epsilon must be non-negative and finite", nil)
	}
	if args.Position < 0 {
		return nil, core.E("rocm.hip."+operation, "position must be non-negative", nil)
	}
	if uint64(args.Position) > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip."+operation, "position is out of uint32 range", nil)
	}
	position := uint32(args.Position)
	if args.Base <= 0 || math.IsNaN(float64(args.Base)) || math.IsInf(float64(args.Base), 0) {
		return nil, core.E("rocm.hip."+operation, "base must be positive and finite", nil)
	}
	frequencyScale := args.FrequencyScale
	if frequencyScale == 0 {
		frequencyScale = 1
	}
	if frequencyScale <= 0 || math.IsNaN(float64(frequencyScale)) || math.IsInf(float64(frequencyScale), 0) {
		return nil, core.E("rocm.hip."+operation, "frequency scale must be positive and finite", nil)
	}
	if args.FrequencyDim < 0 || (args.FrequencyDim > 0 && args.FrequencyDim < args.HeadDim) {
		return nil, core.E("rocm.hip."+operation, "frequency dimension must be zero or at least head dim", nil)
	}
	frequencyDim, err := rocmDeviceKVUint32("frequency dimension", args.FrequencyDim)
	if err != nil {
		return nil, err
	}
	if args.RotaryCount < 0 || args.RotaryCount > args.HeadDim || args.RotaryCount%2 != 0 {
		return nil, core.E("rocm.hip."+operation, "rotary count must be zero or an even count no larger than head dim", nil)
	}
	rotaryCount, err := rocmDeviceKVUint32("rotary count", args.RotaryCount)
	if err != nil {
		return nil, err
	}
	totalCount := uint64(headDim) * uint64(headCount)
	if totalCount > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip."+operation, "total count is out of range", nil)
	}
	inputBytes, err := hipAlignedFloat32Bytes("input", args.InputBytes, uint32(totalCount))
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "input byte count", err)
	}
	weightBytes, err := hipRMSNormLaunchWeightBytes(operation, "weight", args.WeightPointer, args.WeightBytes, headDim, args.WeightEncoding, args.Flags)
	if err != nil {
		return nil, err
	}
	outputBytes, err := hipAlignedFloat32Bytes("output", args.OutputBytes, uint32(totalCount))
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "output byte count", err)
	}
	if cap(payload) < hipRMSNormRoPEHeadsLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipRMSNormRoPEHeadsLaunchArgsBytes)
	} else {
		payload = payload[:hipRMSNormRoPEHeadsLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipRMSNormRoPEHeadsLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], headDim)
	binary.LittleEndian.PutUint32(payload[36:], headCount)
	binary.LittleEndian.PutUint32(payload[40:], inputBytes)
	binary.LittleEndian.PutUint32(payload[44:], weightBytes)
	binary.LittleEndian.PutUint32(payload[48:], outputBytes)
	binary.LittleEndian.PutUint32(payload[52:], math.Float32bits(args.Epsilon))
	binary.LittleEndian.PutUint32(payload[56:], args.WeightEncoding)
	binary.LittleEndian.PutUint32(payload[60:], args.Flags)
	binary.LittleEndian.PutUint32(payload[64:], position)
	binary.LittleEndian.PutUint32(payload[68:], math.Float32bits(args.Base))
	binary.LittleEndian.PutUint32(payload[72:], frequencyDim)
	binary.LittleEndian.PutUint32(payload[76:], rotaryCount)
	binary.LittleEndian.PutUint32(payload[80:], math.Float32bits(frequencyScale))
	return payload, nil
}

func (args hipRMSNormRoPEHeadsBatchLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipRMSNormRoPEHeadsPairLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipRMSNormRoPEHeadsPairLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	const operation = "RMSNormRoPEHeadsPairLaunch"
	if args.QueryInputPointer == 0 || args.QueryOutputPointer == 0 || args.KeyInputPointer == 0 || args.KeyOutputPointer == 0 {
		return nil, core.E("rocm.hip."+operation, "query/key input and output pointers are required", nil)
	}
	headDim, err := rocmDeviceKVPositiveUint32("head dim", args.HeadDim)
	if err != nil {
		return nil, err
	}
	if headDim%2 != 0 {
		return nil, core.E("rocm.hip."+operation, "head dim must be even", nil)
	}
	queryHeadCount, err := rocmDeviceKVPositiveUint32("query head count", args.QueryHeadCount)
	if err != nil {
		return nil, err
	}
	keyHeadCount, err := rocmDeviceKVPositiveUint32("key head count", args.KeyHeadCount)
	if err != nil {
		return nil, err
	}
	if args.QueryEpsilon < 0 || math.IsNaN(float64(args.QueryEpsilon)) || math.IsInf(float64(args.QueryEpsilon), 0) ||
		args.KeyEpsilon < 0 || math.IsNaN(float64(args.KeyEpsilon)) || math.IsInf(float64(args.KeyEpsilon), 0) {
		return nil, core.E("rocm.hip."+operation, "epsilon must be non-negative and finite", nil)
	}
	if args.Position < 0 {
		return nil, core.E("rocm.hip."+operation, "position must be non-negative", nil)
	}
	if uint64(args.Position) > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip."+operation, "position is out of uint32 range", nil)
	}
	position := uint32(args.Position)
	if args.Base <= 0 || math.IsNaN(float64(args.Base)) || math.IsInf(float64(args.Base), 0) {
		return nil, core.E("rocm.hip."+operation, "base must be positive and finite", nil)
	}
	frequencyScale := args.FrequencyScale
	if frequencyScale == 0 {
		frequencyScale = 1
	}
	if frequencyScale <= 0 || math.IsNaN(float64(frequencyScale)) || math.IsInf(float64(frequencyScale), 0) {
		return nil, core.E("rocm.hip."+operation, "frequency scale must be positive and finite", nil)
	}
	if args.FrequencyDim < 0 || (args.FrequencyDim > 0 && args.FrequencyDim < args.HeadDim) {
		return nil, core.E("rocm.hip."+operation, "frequency dimension must be zero or at least head dim", nil)
	}
	frequencyDim, err := rocmDeviceKVUint32("frequency dimension", args.FrequencyDim)
	if err != nil {
		return nil, err
	}
	if args.RotaryCount < 0 || args.RotaryCount > args.HeadDim || args.RotaryCount%2 != 0 {
		return nil, core.E("rocm.hip."+operation, "rotary count must be zero or an even count no larger than head dim", nil)
	}
	rotaryCount, err := rocmDeviceKVUint32("rotary count", args.RotaryCount)
	if err != nil {
		return nil, err
	}
	queryCount := uint64(headDim) * uint64(queryHeadCount)
	keyCount := uint64(headDim) * uint64(keyHeadCount)
	if queryCount > uint64(^uint32(0)) || keyCount > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip."+operation, "total count is out of range", nil)
	}
	queryInputBytes, err := hipAlignedFloat32Bytes("query input", args.QueryInputBytes, uint32(queryCount))
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "query input byte count", err)
	}
	queryWeightBytes, err := hipRMSNormLaunchWeightBytes(operation, "query weight", args.QueryWeightPointer, args.QueryWeightBytes, headDim, args.QueryWeightEncoding, args.QueryFlags)
	if err != nil {
		return nil, err
	}
	queryOutputBytes, err := hipAlignedFloat32Bytes("query output", args.QueryOutputBytes, uint32(queryCount))
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "query output byte count", err)
	}
	keyInputBytes, err := hipAlignedFloat32Bytes("key input", args.KeyInputBytes, uint32(keyCount))
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "key input byte count", err)
	}
	keyWeightBytes, err := hipRMSNormLaunchWeightBytes(operation, "key weight", args.KeyWeightPointer, args.KeyWeightBytes, headDim, args.KeyWeightEncoding, args.KeyFlags)
	if err != nil {
		return nil, err
	}
	keyOutputBytes, err := hipAlignedFloat32Bytes("key output", args.KeyOutputBytes, uint32(keyCount))
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "key output byte count", err)
	}
	if cap(payload) < hipRMSNormRoPEHeadsPairLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipRMSNormRoPEHeadsPairLaunchArgsBytes)
	} else {
		payload = payload[:hipRMSNormRoPEHeadsPairLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipRMSNormRoPEHeadsPairLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.QueryInputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.QueryWeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.QueryOutputPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.KeyInputPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.KeyWeightPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.KeyOutputPointer))
	binary.LittleEndian.PutUint32(payload[56:], headDim)
	binary.LittleEndian.PutUint32(payload[60:], queryHeadCount)
	binary.LittleEndian.PutUint32(payload[64:], keyHeadCount)
	binary.LittleEndian.PutUint32(payload[68:], queryInputBytes)
	binary.LittleEndian.PutUint32(payload[72:], queryWeightBytes)
	binary.LittleEndian.PutUint32(payload[76:], queryOutputBytes)
	binary.LittleEndian.PutUint32(payload[80:], keyInputBytes)
	binary.LittleEndian.PutUint32(payload[84:], keyWeightBytes)
	binary.LittleEndian.PutUint32(payload[88:], keyOutputBytes)
	binary.LittleEndian.PutUint32(payload[92:], math.Float32bits(args.QueryEpsilon))
	binary.LittleEndian.PutUint32(payload[96:], args.QueryWeightEncoding)
	binary.LittleEndian.PutUint32(payload[100:], args.QueryFlags)
	binary.LittleEndian.PutUint32(payload[104:], math.Float32bits(args.KeyEpsilon))
	binary.LittleEndian.PutUint32(payload[108:], args.KeyWeightEncoding)
	binary.LittleEndian.PutUint32(payload[112:], args.KeyFlags)
	binary.LittleEndian.PutUint32(payload[116:], position)
	binary.LittleEndian.PutUint32(payload[120:], math.Float32bits(args.Base))
	binary.LittleEndian.PutUint32(payload[124:], frequencyDim)
	binary.LittleEndian.PutUint32(payload[128:], rotaryCount)
	binary.LittleEndian.PutUint32(payload[132:], math.Float32bits(frequencyScale))
	return payload, nil
}

func (args hipRMSNormRoPEHeadsBatchLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	const operation = "RMSNormRoPEHeadsBatchLaunch"
	if args.InputPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip."+operation, "input and output pointers are required", nil)
	}
	headDim, err := rocmDeviceKVPositiveUint32("head dim", args.HeadDim)
	if err != nil {
		return nil, err
	}
	if headDim%2 != 0 {
		return nil, core.E("rocm.hip."+operation, "head dim must be even", nil)
	}
	headCount, err := rocmDeviceKVPositiveUint32("head count", args.HeadCount)
	if err != nil {
		return nil, err
	}
	batch, err := rocmDeviceKVPositiveUint32("batch", args.Batch)
	if err != nil {
		return nil, err
	}
	if args.Epsilon < 0 || math.IsNaN(float64(args.Epsilon)) || math.IsInf(float64(args.Epsilon), 0) {
		return nil, core.E("rocm.hip."+operation, "epsilon must be non-negative and finite", nil)
	}
	if args.StartPosition < 0 {
		return nil, core.E("rocm.hip."+operation, "start position must be non-negative", nil)
	}
	lastPosition := uint64(args.StartPosition) + uint64(batch) - 1
	if lastPosition > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip."+operation, "position range is out of uint32 range", nil)
	}
	startPosition := uint32(args.StartPosition)
	if args.Base <= 0 || math.IsNaN(float64(args.Base)) || math.IsInf(float64(args.Base), 0) {
		return nil, core.E("rocm.hip."+operation, "base must be positive and finite", nil)
	}
	frequencyScale := args.FrequencyScale
	if frequencyScale == 0 {
		frequencyScale = 1
	}
	if frequencyScale <= 0 || math.IsNaN(float64(frequencyScale)) || math.IsInf(float64(frequencyScale), 0) {
		return nil, core.E("rocm.hip."+operation, "frequency scale must be positive and finite", nil)
	}
	if args.FrequencyDim < 0 || (args.FrequencyDim > 0 && args.FrequencyDim < args.HeadDim) {
		return nil, core.E("rocm.hip."+operation, "frequency dimension must be zero or at least head dim", nil)
	}
	frequencyDim, err := rocmDeviceKVUint32("frequency dimension", args.FrequencyDim)
	if err != nil {
		return nil, err
	}
	if args.RotaryCount < 0 || args.RotaryCount > args.HeadDim || args.RotaryCount%2 != 0 {
		return nil, core.E("rocm.hip."+operation, "rotary count must be zero or an even count no larger than head dim", nil)
	}
	rotaryCount, err := rocmDeviceKVUint32("rotary count", args.RotaryCount)
	if err != nil {
		return nil, err
	}
	totalCount := uint64(headDim) * uint64(headCount) * uint64(batch)
	if totalCount > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip."+operation, "total count is out of range", nil)
	}
	inputBytes, err := hipAlignedFloat32Bytes("input", args.InputBytes, uint32(totalCount))
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "input byte count", err)
	}
	weightBytes, err := hipRMSNormLaunchWeightBytes(operation, "weight", args.WeightPointer, args.WeightBytes, headDim, args.WeightEncoding, args.Flags)
	if err != nil {
		return nil, err
	}
	outputBytes, err := hipAlignedFloat32Bytes("output", args.OutputBytes, uint32(totalCount))
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "output byte count", err)
	}
	if cap(payload) < hipRMSNormRoPEHeadsBatchLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipRMSNormRoPEHeadsBatchLaunchArgsBytes)
	} else {
		payload = payload[:hipRMSNormRoPEHeadsBatchLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipRMSNormRoPEHeadsBatchLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], headDim)
	binary.LittleEndian.PutUint32(payload[36:], headCount)
	binary.LittleEndian.PutUint32(payload[40:], batch)
	binary.LittleEndian.PutUint32(payload[44:], inputBytes)
	binary.LittleEndian.PutUint32(payload[48:], weightBytes)
	binary.LittleEndian.PutUint32(payload[52:], outputBytes)
	binary.LittleEndian.PutUint32(payload[56:], math.Float32bits(args.Epsilon))
	binary.LittleEndian.PutUint32(payload[60:], args.WeightEncoding)
	binary.LittleEndian.PutUint32(payload[64:], args.Flags)
	binary.LittleEndian.PutUint32(payload[68:], startPosition)
	binary.LittleEndian.PutUint32(payload[72:], math.Float32bits(args.Base))
	binary.LittleEndian.PutUint32(payload[76:], frequencyDim)
	binary.LittleEndian.PutUint32(payload[80:], rotaryCount)
	binary.LittleEndian.PutUint32(payload[84:], math.Float32bits(frequencyScale))
	return payload, nil
}

func (args hipRMSNormRoPEHeadsPairLaneBatchLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipRMSNormRoPEHeadsPairLaneBatchLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	const operation = "RMSNormRoPEHeadsPairLaneBatchLaunch"
	if args.QueryInputPointer == 0 || args.QueryOutputPointer == 0 || args.KeyInputPointer == 0 || args.KeyOutputPointer == 0 || args.PositionsPointer == 0 {
		return nil, core.E("rocm.hip."+operation, "query/key input/output and positions pointers are required", nil)
	}
	headDim, err := rocmDeviceKVPositiveUint32("head dim", args.HeadDim)
	if err != nil {
		return nil, err
	}
	if headDim%2 != 0 {
		return nil, core.E("rocm.hip."+operation, "head dim must be even", nil)
	}
	queryHeadCount, err := rocmDeviceKVPositiveUint32("query head count", args.QueryHeadCount)
	if err != nil {
		return nil, err
	}
	keyHeadCount, err := rocmDeviceKVPositiveUint32("key head count", args.KeyHeadCount)
	if err != nil {
		return nil, err
	}
	batch, err := rocmDeviceKVPositiveUint32("batch", args.Batch)
	if err != nil {
		return nil, err
	}
	if args.QueryEpsilon < 0 || math.IsNaN(float64(args.QueryEpsilon)) || math.IsInf(float64(args.QueryEpsilon), 0) ||
		args.KeyEpsilon < 0 || math.IsNaN(float64(args.KeyEpsilon)) || math.IsInf(float64(args.KeyEpsilon), 0) {
		return nil, core.E("rocm.hip."+operation, "epsilon must be non-negative and finite", nil)
	}
	if args.Base <= 0 || math.IsNaN(float64(args.Base)) || math.IsInf(float64(args.Base), 0) {
		return nil, core.E("rocm.hip."+operation, "base must be positive and finite", nil)
	}
	frequencyScale := args.FrequencyScale
	if frequencyScale == 0 {
		frequencyScale = 1
	}
	if frequencyScale <= 0 || math.IsNaN(float64(frequencyScale)) || math.IsInf(float64(frequencyScale), 0) {
		return nil, core.E("rocm.hip."+operation, "frequency scale must be positive and finite", nil)
	}
	if args.FrequencyDim < 0 || (args.FrequencyDim > 0 && args.FrequencyDim < args.HeadDim) {
		return nil, core.E("rocm.hip."+operation, "frequency dimension must be zero or at least head dim", nil)
	}
	frequencyDim, err := rocmDeviceKVUint32("frequency dimension", args.FrequencyDim)
	if err != nil {
		return nil, err
	}
	if args.RotaryCount < 0 || args.RotaryCount > args.HeadDim || args.RotaryCount%2 != 0 {
		return nil, core.E("rocm.hip."+operation, "rotary count must be zero or an even count no larger than head dim", nil)
	}
	rotaryCount, err := rocmDeviceKVUint32("rotary count", args.RotaryCount)
	if err != nil {
		return nil, err
	}
	queryCount := uint64(headDim) * uint64(queryHeadCount) * uint64(batch)
	keyCount := uint64(headDim) * uint64(keyHeadCount) * uint64(batch)
	if queryCount > uint64(^uint32(0)) || keyCount > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip."+operation, "total count is out of range", nil)
	}
	queryInputBytes, err := hipAlignedFloat32Bytes("query input", args.QueryInputBytes, uint32(queryCount))
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "query input byte count", err)
	}
	queryWeightBytes, err := hipRMSNormLaunchWeightBytes(operation, "query weight", args.QueryWeightPointer, args.QueryWeightBytes, headDim, args.QueryWeightEncoding, args.QueryFlags)
	if err != nil {
		return nil, err
	}
	queryOutputBytes, err := hipAlignedFloat32Bytes("query output", args.QueryOutputBytes, uint32(queryCount))
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "query output byte count", err)
	}
	keyInputBytes, err := hipAlignedFloat32Bytes("key input", args.KeyInputBytes, uint32(keyCount))
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "key input byte count", err)
	}
	keyWeightBytes, err := hipRMSNormLaunchWeightBytes(operation, "key weight", args.KeyWeightPointer, args.KeyWeightBytes, headDim, args.KeyWeightEncoding, args.KeyFlags)
	if err != nil {
		return nil, err
	}
	keyOutputBytes, err := hipAlignedFloat32Bytes("key output", args.KeyOutputBytes, uint32(keyCount))
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "key output byte count", err)
	}
	positionsBytes, err := hipAlignedFloat32Bytes("positions", args.PositionsBytes, batch)
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "positions byte count", err)
	}
	if cap(payload) < hipRMSNormRoPEHeadsPairLaneBatchLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipRMSNormRoPEHeadsPairLaneBatchLaunchArgsBytes)
	} else {
		payload = payload[:hipRMSNormRoPEHeadsPairLaneBatchLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipRMSNormRoPEHeadsPairLaneBatchLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.QueryInputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.QueryWeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.QueryOutputPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.KeyInputPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.KeyWeightPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.KeyOutputPointer))
	binary.LittleEndian.PutUint64(payload[56:], uint64(args.PositionsPointer))
	binary.LittleEndian.PutUint32(payload[64:], headDim)
	binary.LittleEndian.PutUint32(payload[68:], queryHeadCount)
	binary.LittleEndian.PutUint32(payload[72:], keyHeadCount)
	binary.LittleEndian.PutUint32(payload[76:], batch)
	binary.LittleEndian.PutUint32(payload[80:], queryInputBytes)
	binary.LittleEndian.PutUint32(payload[84:], queryWeightBytes)
	binary.LittleEndian.PutUint32(payload[88:], queryOutputBytes)
	binary.LittleEndian.PutUint32(payload[92:], keyInputBytes)
	binary.LittleEndian.PutUint32(payload[96:], keyWeightBytes)
	binary.LittleEndian.PutUint32(payload[100:], keyOutputBytes)
	binary.LittleEndian.PutUint32(payload[104:], math.Float32bits(args.QueryEpsilon))
	binary.LittleEndian.PutUint32(payload[108:], args.QueryWeightEncoding)
	binary.LittleEndian.PutUint32(payload[112:], args.QueryFlags)
	binary.LittleEndian.PutUint32(payload[116:], math.Float32bits(args.KeyEpsilon))
	binary.LittleEndian.PutUint32(payload[120:], args.KeyWeightEncoding)
	binary.LittleEndian.PutUint32(payload[124:], args.KeyFlags)
	binary.LittleEndian.PutUint32(payload[128:], positionsBytes)
	binary.LittleEndian.PutUint32(payload[132:], math.Float32bits(args.Base))
	binary.LittleEndian.PutUint32(payload[136:], frequencyDim)
	binary.LittleEndian.PutUint32(payload[140:], rotaryCount)
	binary.LittleEndian.PutUint32(payload[144:], math.Float32bits(frequencyScale))
	return payload, nil
}

func hipRMSNormWeightEncoding(req hipRMSNormRequest) (uint32, error) {
	switch {
	case len(req.Weight) > 0 && len(req.WeightBF16) == 0:
		return hipRMSNormWeightEncodingF32, nil
	case len(req.WeightBF16) > 0 && len(req.Weight) == 0:
		return hipRMSNormWeightEncodingBF16, nil
	default:
		return 0, core.E("rocm.hip.RMSNormLaunch", "exactly one RMSNorm weight encoding is required", nil)
	}
}

func (buffers *hipRMSNormDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Weight, buffers.Input} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipRMSNormDeviceBuffers) ReadOutput() ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.RMSNormLaunch", "rms norm output buffer is required", nil)
	}
	return hipReadFloat32DeviceOutput(buffers.Output, "rocm.hip.RMSNormLaunch", "rms norm output", buffers.Count)
}

func (req hipRoPERequest) validate() error {
	if len(req.Input) == 0 || len(req.Input)%2 != 0 {
		return core.E("rocm.hip.RoPELaunch", "input length must be positive and even", nil)
	}
	if req.Position < 0 {
		return core.E("rocm.hip.RoPELaunch", "position must be non-negative", nil)
	}
	if req.Base <= 0 || math.IsNaN(float64(req.Base)) || math.IsInf(float64(req.Base), 0) {
		return core.E("rocm.hip.RoPELaunch", "base must be positive and finite", nil)
	}
	if req.FrequencyDim < 0 || (req.FrequencyDim > 0 && req.FrequencyDim < len(req.Input)) {
		return core.E("rocm.hip.RoPELaunch", "frequency dimension must be zero or at least input length", nil)
	}
	if req.RotaryCount < 0 || req.RotaryCount > len(req.Input) || req.RotaryCount%2 != 0 {
		return core.E("rocm.hip.RoPELaunch", "rotary count must be zero or an even count no larger than input length", nil)
	}
	return nil
}

func (req hipRoPERequest) deviceBuffers(driver nativeHIPDriver) (*hipRoPEDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	inputPayload, err := hipFloat32Payload(req.Input)
	if err != nil {
		return nil, core.E("rocm.hip.RoPELaunch", "encode input", err)
	}
	input, err := hipUploadByteBuffer(driver, "rocm.hip.RoPELaunch", "rope input", inputPayload, len(req.Input))
	if err != nil {
		return nil, err
	}
	buffers := &hipRoPEDeviceBuffers{Input: input, Count: len(req.Input), Position: req.Position, Base: req.Base, FrequencyDim: req.FrequencyDim, RotaryCount: req.RotaryCount}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RoPELaunch", "rope output", uint64(len(req.Input)*4), len(req.Input))
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipRoPERequest) launchArgs(buffers *hipRoPEDeviceBuffers) (hipRoPELaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipRoPELaunchArgs{}, err
	}
	if buffers == nil || buffers.Input == nil || buffers.Output == nil {
		return hipRoPELaunchArgs{}, core.E("rocm.hip.RoPELaunch", "rope device buffers are required", nil)
	}
	if buffers.Input.Count() != len(req.Input) || buffers.Output.Count() != len(req.Input) {
		return hipRoPELaunchArgs{}, core.E("rocm.hip.RoPELaunch", "rope device buffer shape mismatch", nil)
	}
	return hipRoPELaunchArgs{
		InputPointer:  buffers.Input.Pointer(),
		OutputPointer: buffers.Output.Pointer(),
		Count:         len(req.Input),
		InputBytes:    buffers.Input.SizeBytes(),
		OutputBytes:   buffers.Output.SizeBytes(),
		Position:      req.Position,
		Base:          req.Base,
		FrequencyDim:  req.FrequencyDim,
		RotaryCount:   req.RotaryCount,
	}, nil
}

func (args hipRoPELaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipRoPELaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.RoPELaunch", "input and output pointers are required", nil)
	}
	count, err := rocmDeviceKVPositiveUint32("count", args.Count)
	if err != nil {
		return nil, err
	}
	if count%2 != 0 {
		return nil, core.E("rocm.hip.RoPELaunch", "count must be even", nil)
	}
	if args.Position < 0 {
		return nil, core.E("rocm.hip.RoPELaunch", "position must be non-negative", nil)
	}
	if uint64(args.Position) > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip.RoPELaunch", "position is out of uint32 range", nil)
	}
	position := uint32(args.Position)
	if args.Base <= 0 || math.IsNaN(float64(args.Base)) || math.IsInf(float64(args.Base), 0) {
		return nil, core.E("rocm.hip.RoPELaunch", "base must be positive and finite", nil)
	}
	if args.FrequencyDim < 0 || (args.FrequencyDim > 0 && args.FrequencyDim < args.Count) {
		return nil, core.E("rocm.hip.RoPELaunch", "frequency dimension must be zero or at least count", nil)
	}
	frequencyDim, err := rocmDeviceKVUint32("frequency dimension", args.FrequencyDim)
	if err != nil {
		return nil, err
	}
	if args.RotaryCount < 0 || args.RotaryCount > args.Count || args.RotaryCount%2 != 0 {
		return nil, core.E("rocm.hip.RoPELaunch", "rotary count must be zero or an even count no larger than count", nil)
	}
	rotaryCount, err := rocmDeviceKVUint32("rotary count", args.RotaryCount)
	if err != nil {
		return nil, err
	}
	inputBytes, err := hipAlignedFloat32Bytes("input", args.InputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.RoPELaunch", "input byte count", err)
	}
	outputBytes, err := hipAlignedFloat32Bytes("output", args.OutputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.RoPELaunch", "output byte count", err)
	}
	if cap(payload) < hipRoPELaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipRoPELaunchArgsBytes)
	} else {
		payload = payload[:hipRoPELaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipRoPELaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[24:], count)
	binary.LittleEndian.PutUint32(payload[28:], inputBytes)
	binary.LittleEndian.PutUint32(payload[32:], outputBytes)
	binary.LittleEndian.PutUint32(payload[36:], position)
	binary.LittleEndian.PutUint32(payload[40:], math.Float32bits(args.Base))
	binary.LittleEndian.PutUint32(payload[44:], frequencyDim)
	binary.LittleEndian.PutUint32(payload[48:], rotaryCount)
	return payload, nil
}

func (args hipRoPEHeadsLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipRoPEHeadsLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.RoPEHeadsLaunch", "input and output pointers are required", nil)
	}
	headDim, err := rocmDeviceKVPositiveUint32("head dim", args.HeadDim)
	if err != nil {
		return nil, err
	}
	if headDim%2 != 0 {
		return nil, core.E("rocm.hip.RoPEHeadsLaunch", "head dim must be even", nil)
	}
	headCount, err := rocmDeviceKVPositiveUint32("head count", args.HeadCount)
	if err != nil {
		return nil, err
	}
	if args.Position < 0 {
		return nil, core.E("rocm.hip.RoPEHeadsLaunch", "position must be non-negative", nil)
	}
	if uint64(args.Position) > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip.RoPEHeadsLaunch", "position is out of uint32 range", nil)
	}
	position := uint32(args.Position)
	if args.Base <= 0 || math.IsNaN(float64(args.Base)) || math.IsInf(float64(args.Base), 0) {
		return nil, core.E("rocm.hip.RoPEHeadsLaunch", "base must be positive and finite", nil)
	}
	if args.FrequencyDim < 0 || (args.FrequencyDim > 0 && args.FrequencyDim < args.HeadDim) {
		return nil, core.E("rocm.hip.RoPEHeadsLaunch", "frequency dimension must be zero or at least head dim", nil)
	}
	frequencyDim, err := rocmDeviceKVUint32("frequency dimension", args.FrequencyDim)
	if err != nil {
		return nil, err
	}
	if args.RotaryCount < 0 || args.RotaryCount > args.HeadDim || args.RotaryCount%2 != 0 {
		return nil, core.E("rocm.hip.RoPEHeadsLaunch", "rotary count must be zero or an even count no larger than head dim", nil)
	}
	rotaryCount, err := rocmDeviceKVUint32("rotary count", args.RotaryCount)
	if err != nil {
		return nil, err
	}
	totalCount := uint64(headDim) * uint64(headCount)
	if totalCount > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip.RoPEHeadsLaunch", "total count is out of range", nil)
	}
	inputBytes, err := hipAlignedFloat32Bytes("input", args.InputBytes, uint32(totalCount))
	if err != nil {
		return nil, core.E("rocm.hip.RoPEHeadsLaunch", "input byte count", err)
	}
	outputBytes, err := hipAlignedFloat32Bytes("output", args.OutputBytes, uint32(totalCount))
	if err != nil {
		return nil, core.E("rocm.hip.RoPEHeadsLaunch", "output byte count", err)
	}
	if cap(payload) < hipRoPEHeadsLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipRoPEHeadsLaunchArgsBytes)
	} else {
		payload = payload[:hipRoPEHeadsLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipRoPEHeadsLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[24:], headDim)
	binary.LittleEndian.PutUint32(payload[28:], headCount)
	binary.LittleEndian.PutUint32(payload[32:], inputBytes)
	binary.LittleEndian.PutUint32(payload[36:], outputBytes)
	binary.LittleEndian.PutUint32(payload[40:], position)
	binary.LittleEndian.PutUint32(payload[44:], math.Float32bits(args.Base))
	binary.LittleEndian.PutUint32(payload[48:], frequencyDim)
	binary.LittleEndian.PutUint32(payload[52:], rotaryCount)
	return payload, nil
}

func (buffers *hipRoPEDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Input} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipRoPEDeviceBuffers) ReadOutput() ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.RoPELaunch", "rope output buffer is required", nil)
	}
	return hipReadFloat32DeviceOutput(buffers.Output, "rocm.hip.RoPELaunch", "rope output", buffers.Count)
}

func hipAlignedFloat32Bytes(label string, sizeBytes uint64, count uint32) (uint32, error) {
	want := uint64(count) * 4
	if sizeBytes != want {
		return 0, core.E("rocm.hip.LaunchBytes", label+" bytes must match count", nil)
	}
	if sizeBytes > uint64(^uint32(0)) {
		return 0, core.E("rocm.hip.LaunchBytes", label+" bytes are out of uint32 range", nil)
	}
	return uint32(sizeBytes), nil
}

func hipReadFloat32DeviceOutput(buffer *hipDeviceByteBuffer, operation, label string, count int) ([]float32, error) {
	if count <= 0 || buffer.Count() != count || buffer.SizeBytes() != uint64(count)*4 {
		return nil, core.E(operation, label+" byte count mismatch", nil)
	}
	payload := make([]byte, buffer.SizeBytes())
	if err := buffer.driver.CopyDeviceToHost(buffer.Pointer(), payload); err != nil {
		return nil, core.E(operation, "copy "+label, err)
	}
	values, err := hipFloat32PayloadValues(payload)
	if err != nil {
		return nil, err
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E(operation, label+" values must be finite", nil)
	}
	return values, nil
}

func hipReadGreedyResult(buffer *hipDeviceByteBuffer, operation, label string, vocabSize int) (hipGreedySampleResult, error) {
	if vocabSize <= 0 || buffer.Count() != 1 || buffer.SizeBytes() != hipGreedyResultBytes {
		return hipGreedySampleResult{}, core.E(operation, label+" byte count mismatch", nil)
	}
	payload := make([]byte, buffer.SizeBytes())
	if err := buffer.driver.CopyDeviceToHost(buffer.Pointer(), payload); err != nil {
		return hipGreedySampleResult{}, core.E(operation, "copy "+label, err)
	}
	if len(payload) != hipGreedyResultBytes {
		return hipGreedySampleResult{}, core.E(operation, label+" byte count mismatch", nil)
	}
	result := hipGreedySampleResult{
		TokenID: int(int32(binary.LittleEndian.Uint32(payload[0:]))),
		Score:   math.Float32frombits(binary.LittleEndian.Uint32(payload[4:])),
	}
	if result.TokenID < 0 || result.TokenID >= vocabSize {
		return hipGreedySampleResult{}, core.E(operation, label+" token ID out of range", nil)
	}
	if math.IsNaN(float64(result.Score)) || math.IsInf(float64(result.Score), 0) {
		return hipGreedySampleResult{}, core.E(operation, label+" score must be finite", nil)
	}
	return result, nil
}

func hipFloat32SliceProbabilities(values []float32) bool {
	for _, value := range values {
		if value < 0 || value > 1 {
			return false
		}
	}
	return true
}

func (req hipGreedySampleRequest) validate() error {
	if len(req.Logits) == 0 {
		return core.E("rocm.hip.GreedyLaunch", "logits are required", nil)
	}
	return nil
}

func (req hipGreedySampleRequest) deviceBuffers(driver nativeHIPDriver) (*hipGreedySampleDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	logitsPayload, err := hipFloat32Payload(req.Logits)
	if err != nil {
		return nil, core.E("rocm.hip.GreedyLaunch", "encode logits", err)
	}
	logits, err := hipUploadByteBuffer(driver, "rocm.hip.GreedyLaunch", "greedy logits", logitsPayload, len(req.Logits))
	if err != nil {
		return nil, err
	}
	buffers := &hipGreedySampleDeviceBuffers{Logits: logits, Count: len(req.Logits)}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	output, err := hipAllocateByteBuffer(driver, "rocm.hip.GreedyLaunch", "greedy output", hipGreedyResultBytes, 1)
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipGreedySampleRequest) launchArgs(buffers *hipGreedySampleDeviceBuffers) (hipGreedySampleLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipGreedySampleLaunchArgs{}, err
	}
	if buffers == nil || buffers.Logits == nil || buffers.Output == nil {
		return hipGreedySampleLaunchArgs{}, core.E("rocm.hip.GreedyLaunch", "greedy sample device buffers are required", nil)
	}
	if buffers.Logits.Count() != len(req.Logits) || buffers.Output.SizeBytes() != hipGreedyResultBytes {
		return hipGreedySampleLaunchArgs{}, core.E("rocm.hip.GreedyLaunch", "greedy sample device buffer shape mismatch", nil)
	}
	return hipGreedySampleLaunchArgs{
		LogitsPointer: buffers.Logits.Pointer(),
		OutputPointer: buffers.Output.Pointer(),
		Count:         len(req.Logits),
		LogitsBytes:   buffers.Logits.SizeBytes(),
		OutputBytes:   buffers.Output.SizeBytes(),
	}, nil
}

func (args hipGreedySampleLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipGreedySampleLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.LogitsPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.GreedyLaunch", "logits and output pointers are required", nil)
	}
	count, err := rocmDeviceKVPositiveUint32("count", args.Count)
	if err != nil {
		return nil, err
	}
	logitsBytes, err := hipAlignedFloat32Bytes("logits", args.LogitsBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.GreedyLaunch", "logits byte count", err)
	}
	if args.OutputBytes != hipGreedyResultBytes {
		return nil, core.E("rocm.hip.GreedyLaunch", "output byte count mismatch", nil)
	}
	if cap(payload) < hipGreedyLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipGreedyLaunchArgsBytes)
	} else {
		payload = payload[:hipGreedyLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipGreedyLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.LogitsPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[24:], count)
	binary.LittleEndian.PutUint32(payload[28:], logitsBytes)
	binary.LittleEndian.PutUint32(payload[32:], uint32(args.OutputBytes))
	return payload, nil
}

func (args hipSoftcapGreedySampleLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipSoftcapGreedySampleLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.LogitsPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.SoftcapGreedyLaunch", "logits and output pointers are required", nil)
	}
	count, err := rocmDeviceKVPositiveUint32("count", args.Count)
	if err != nil {
		return nil, err
	}
	logitsBytes, err := hipAlignedFloat32Bytes("logits", args.LogitsBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.SoftcapGreedyLaunch", "logits byte count", err)
	}
	if args.OutputBytes != hipGreedyResultBytes {
		return nil, core.E("rocm.hip.SoftcapGreedyLaunch", "output byte count mismatch", nil)
	}
	if args.Softcap < 0 || math.IsNaN(float64(args.Softcap)) || math.IsInf(float64(args.Softcap), 0) {
		return nil, core.E("rocm.hip.SoftcapGreedyLaunch", "softcap must be non-negative and finite", nil)
	}
	var suppressCount uint32
	if args.SuppressCount < 0 {
		return nil, core.E("rocm.hip.SoftcapGreedyLaunch", "suppress token count must be non-negative", nil)
	}
	if args.SuppressCount > 0 {
		if args.SuppressPointer == 0 {
			return nil, core.E("rocm.hip.SoftcapGreedyLaunch", "suppress token pointer is required", nil)
		}
		if args.SuppressCount >= args.Count {
			return nil, core.E("rocm.hip.SoftcapGreedyLaunch", "suppress token count must be smaller than logits count", nil)
		}
		suppressCount, err = rocmDeviceKVPositiveUint32("suppress token count", args.SuppressCount)
		if err != nil {
			return nil, err
		}
	} else if args.SuppressPointer != 0 {
		return nil, core.E("rocm.hip.SoftcapGreedyLaunch", "suppress token count is required with a suppress pointer", nil)
	}
	if cap(payload) < hipSoftcapGreedyLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipSoftcapGreedyLaunchArgsBytes)
	} else {
		payload = payload[:hipSoftcapGreedyLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipSoftcapGreedyLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.LogitsPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[24:], count)
	binary.LittleEndian.PutUint32(payload[28:], logitsBytes)
	binary.LittleEndian.PutUint32(payload[32:], uint32(args.OutputBytes))
	binary.LittleEndian.PutUint32(payload[36:], math.Float32bits(args.Softcap))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.SuppressPointer))
	binary.LittleEndian.PutUint32(payload[48:], suppressCount)
	return payload, nil
}

func (buffers *hipGreedySampleDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Logits} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipGreedySampleDeviceBuffers) ReadOutput() (hipGreedySampleResult, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return hipGreedySampleResult{}, core.E("rocm.hip.GreedyLaunch", "greedy output buffer is required", nil)
	}
	return hipReadGreedyResult(buffers.Output, "rocm.hip.GreedyLaunch", "greedy output", buffers.Count)
}

func (req hipAttentionRequest) queryDim() (int, error) {
	if len(req.Query) > 0 {
		if req.QueryDim > 0 && req.QueryDim != len(req.Query) {
			return 0, core.E("rocm.hip.AttentionLaunch", "query dimension does not match query length", nil)
		}
		return len(req.Query), nil
	}
	if req.QueryDim <= 0 {
		return 0, core.E("rocm.hip.AttentionLaunch", "query is required", nil)
	}
	return req.QueryDim, nil
}

func (req hipAttentionRequest) keyHeadsOrDefault() int {
	if req.KeyHeads <= 0 {
		return 1
	}
	return req.KeyHeads
}

func (req hipAttentionRequest) validate() error {
	dim, err := req.queryDim()
	if err != nil {
		return err
	}
	keyHeads := req.keyHeadsOrDefault()
	if keyHeads <= 0 {
		return core.E("rocm.hip.AttentionLaunch", "key head count must be positive", nil)
	}
	if req.Scale < 0 || math.IsNaN(float64(req.Scale)) || math.IsInf(float64(req.Scale), 0) {
		return core.E("rocm.hip.AttentionLaunch", "scale must be non-negative and finite", nil)
	}
	if req.DeviceKV != nil {
		if req.DescriptorTable == nil {
			return core.E("rocm.hip.AttentionLaunch", "device KV attention requires descriptor table", nil)
		}
		if err := req.DescriptorTable.CompatibleWith(req.DeviceKV); err != nil {
			return core.E("rocm.hip.AttentionLaunch", "descriptor table does not match device KV cache", err)
		}
		keyWidth, valueWidth, ok := req.DeviceKV.LastVectorWidths()
		if !ok {
			return core.E("rocm.hip.AttentionLaunch", "device KV cache has no pages", nil)
		}
		if keyWidth != dim*keyHeads || valueWidth != dim*keyHeads {
			return core.E("rocm.hip.AttentionLaunch", "device KV widths must match query dimension", nil)
		}
		return nil
	}
	if req.DescriptorTable != nil {
		return core.E("rocm.hip.AttentionLaunch", "descriptor table requires device KV cache", nil)
	}
	if len(req.Keys) == 0 || len(req.Values) == 0 {
		return core.E("rocm.hip.AttentionLaunch", "keys and values are required", nil)
	}
	kvDim := dim * keyHeads
	if len(req.Keys)%kvDim != 0 || len(req.Values)%kvDim != 0 {
		return core.E("rocm.hip.AttentionLaunch", "key/value tensor lengths must align with query dimension", nil)
	}
	if len(req.Keys) != len(req.Values) {
		return core.E("rocm.hip.AttentionLaunch", "keys and values must describe the same token count", nil)
	}
	return nil
}

func (req hipAttentionRequest) shape() (int, int, error) {
	if err := req.validate(); err != nil {
		return 0, 0, err
	}
	dim, err := req.queryDim()
	if err != nil {
		return 0, 0, err
	}
	if req.DeviceKV != nil {
		return dim, req.DeviceKV.TokenCount(), nil
	}
	return dim, len(req.Keys) / (dim * req.keyHeadsOrDefault()), nil
}

func (req hipAttentionRequest) deviceBuffers(driver nativeHIPDriver) (*hipAttentionDeviceBuffers, error) {
	dim, tokenCount, err := req.shape()
	if err != nil {
		return nil, err
	}
	if len(req.Query) != dim {
		return nil, core.E("rocm.hip.AttentionLaunch", "query values are required for host-query attention launch", nil)
	}
	queryPayload, err := hipFloat32Payload(req.Query)
	if err != nil {
		return nil, core.E("rocm.hip.AttentionLaunch", "encode query", err)
	}
	query, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionLaunch", "attention query", queryPayload, len(req.Query))
	if err != nil {
		return nil, err
	}
	buffers := &hipAttentionDeviceBuffers{Query: query, Dim: dim, TokenCount: tokenCount}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	if req.DeviceKV == nil {
		keyPayload, err := hipFloat32Payload(req.Keys)
		if err != nil {
			return nil, core.E("rocm.hip.AttentionLaunch", "encode keys", err)
		}
		keys, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionLaunch", "attention keys", keyPayload, len(req.Keys))
		if err != nil {
			return nil, err
		}
		buffers.Keys = keys
		valuePayload, err := hipFloat32Payload(req.Values)
		if err != nil {
			return nil, core.E("rocm.hip.AttentionLaunch", "encode values", err)
		}
		values, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionLaunch", "attention values", valuePayload, len(req.Values))
		if err != nil {
			return nil, err
		}
		buffers.Values = values
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionLaunch", "attention output", uint64(dim*4), dim)
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	weights, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionLaunch", "attention weights", uint64(tokenCount*4), tokenCount)
	if err != nil {
		return nil, err
	}
	buffers.Weights = weights
	success = true
	return buffers, nil
}

func (req hipAttentionRequest) launchArgs(buffers *hipAttentionDeviceBuffers) (hipAttentionLaunchArgs, error) {
	dim, tokenCount, err := req.shape()
	if err != nil {
		return hipAttentionLaunchArgs{}, err
	}
	if buffers == nil || buffers.Query == nil || buffers.Output == nil || buffers.Weights == nil {
		return hipAttentionLaunchArgs{}, core.E("rocm.hip.AttentionLaunch", "attention device buffers are required", nil)
	}
	if buffers.Query.Count() != dim || buffers.Output.Count() != dim || buffers.Weights.Count() != tokenCount {
		return hipAttentionLaunchArgs{}, core.E("rocm.hip.AttentionLaunch", "attention device buffer shape mismatch", nil)
	}
	if req.DeviceKV == nil {
		if buffers.Keys == nil || buffers.Values == nil ||
			buffers.Keys.Count() != tokenCount*dim ||
			buffers.Values.Count() != tokenCount*dim {
			return hipAttentionLaunchArgs{}, core.E("rocm.hip.AttentionLaunch", "attention device buffer shape mismatch", nil)
		}
		return hipAttentionLaunchArgs{
			QueryPointer:  buffers.Query.Pointer(),
			KeyPointer:    buffers.Keys.Pointer(),
			ValuePointer:  buffers.Values.Pointer(),
			OutputPointer: buffers.Output.Pointer(),
			WeightPointer: buffers.Weights.Pointer(),
			Dim:           dim,
			TokenCount:    tokenCount,
			QueryBytes:    buffers.Query.SizeBytes(),
			KeyBytes:      buffers.Keys.SizeBytes(),
			ValueBytes:    buffers.Values.SizeBytes(),
			OutputBytes:   buffers.Output.SizeBytes(),
			WeightBytes:   buffers.Weights.SizeBytes(),
			KVSource:      hipAttentionKVSourceContiguous,
			Scale:         req.Scale,
		}, nil
	}
	if buffers.Keys != nil || buffers.Values != nil {
		return hipAttentionLaunchArgs{}, core.E("rocm.hip.AttentionLaunch", "device KV attention must not upload contiguous KV buffers", nil)
	}
	return hipAttentionLaunchArgs{
		QueryPointer:      buffers.Query.Pointer(),
		OutputPointer:     buffers.Output.Pointer(),
		WeightPointer:     buffers.Weights.Pointer(),
		Dim:               dim,
		TokenCount:        tokenCount,
		QueryBytes:        buffers.Query.SizeBytes(),
		OutputBytes:       buffers.Output.SizeBytes(),
		WeightBytes:       buffers.Weights.SizeBytes(),
		KVSource:          hipAttentionKVSourceDevice,
		Scale:             req.Scale,
		DescriptorPointer: req.DescriptorTable.Pointer(),
		DescriptorBytes:   req.DescriptorTable.SizeBytes(),
	}, nil
}

func (args hipAttentionLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipAttentionLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.QueryPointer == 0 || args.OutputPointer == 0 || args.WeightPointer == 0 {
		return nil, core.E("rocm.hip.AttentionLaunch", "query, output, and weight pointers are required", nil)
	}
	if args.KVSource != hipAttentionKVSourceContiguous && args.KVSource != hipAttentionKVSourceDevice {
		return nil, core.E("rocm.hip.AttentionLaunch", core.Sprintf("unsupported KV source %d", args.KVSource), nil)
	}
	if args.KVSource == hipAttentionKVSourceContiguous && (args.KeyPointer == 0 || args.ValuePointer == 0) {
		return nil, core.E("rocm.hip.AttentionLaunch", "key and value pointers are required", nil)
	}
	if args.KVSource == hipAttentionKVSourceDevice && (args.DescriptorPointer == 0 || args.DescriptorBytes < rocmDeviceKVDescriptorHeaderBytes) {
		return nil, core.E("rocm.hip.AttentionLaunch", "device KV descriptor is required", nil)
	}
	if args.Scale < 0 || math.IsNaN(float64(args.Scale)) || math.IsInf(float64(args.Scale), 0) {
		return nil, core.E("rocm.hip.AttentionLaunch", "scale must be non-negative and finite", nil)
	}
	dim, err := rocmDeviceKVPositiveUint32("dimension", args.Dim)
	if err != nil {
		return nil, err
	}
	tokenCount, err := rocmDeviceKVPositiveUint32("token count", args.TokenCount)
	if err != nil {
		return nil, err
	}
	queryBytes, err := hipAlignedFloat32Bytes("query", args.QueryBytes, dim)
	if err != nil {
		return nil, core.E("rocm.hip.AttentionLaunch", "query byte count", err)
	}
	var keyBytes uint32
	var valueBytes uint32
	if args.KVSource == hipAttentionKVSourceContiguous {
		keyBytes, err = hipAlignedFloat32Bytes("key", args.KeyBytes, dim*tokenCount)
		if err != nil {
			return nil, core.E("rocm.hip.AttentionLaunch", "key byte count", err)
		}
		valueBytes, err = hipAlignedFloat32Bytes("value", args.ValueBytes, dim*tokenCount)
		if err != nil {
			return nil, core.E("rocm.hip.AttentionLaunch", "value byte count", err)
		}
	} else if args.KeyBytes != 0 || args.ValueBytes != 0 || args.KeyPointer != 0 || args.ValuePointer != 0 {
		return nil, core.E("rocm.hip.AttentionLaunch", "device KV attention must not set contiguous KV pointers", nil)
	}
	outputBytes, err := hipAlignedFloat32Bytes("output", args.OutputBytes, dim)
	if err != nil {
		return nil, core.E("rocm.hip.AttentionLaunch", "output byte count", err)
	}
	weightBytes, err := hipAlignedFloat32Bytes("weight", args.WeightBytes, tokenCount)
	if err != nil {
		return nil, core.E("rocm.hip.AttentionLaunch", "weight byte count", err)
	}
	if cap(payload) < hipAttentionLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipAttentionLaunchArgsBytes)
	} else {
		payload = payload[:hipAttentionLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipAttentionLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.QueryPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.KeyPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ValuePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint32(payload[48:], dim)
	binary.LittleEndian.PutUint32(payload[52:], tokenCount)
	binary.LittleEndian.PutUint32(payload[56:], queryBytes)
	binary.LittleEndian.PutUint32(payload[60:], keyBytes)
	binary.LittleEndian.PutUint32(payload[64:], valueBytes)
	binary.LittleEndian.PutUint32(payload[68:], outputBytes)
	binary.LittleEndian.PutUint32(payload[72:], weightBytes)
	binary.LittleEndian.PutUint32(payload[76:], args.KVSource)
	binary.LittleEndian.PutUint32(payload[80:], math.Float32bits(args.Scale))
	binary.LittleEndian.PutUint64(payload[88:], uint64(args.DescriptorPointer))
	binary.LittleEndian.PutUint64(payload[96:], args.DescriptorBytes)
	return payload, nil
}

func (args hipAttentionHeadsLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipAttentionHeadsLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.QueryPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.AttentionHeadsLaunch", "query and output pointers are required", nil)
	}
	if args.KVSource != hipAttentionKVSourceContiguous && args.KVSource != hipAttentionKVSourceDevice {
		return nil, core.E("rocm.hip.AttentionHeadsLaunch", core.Sprintf("unsupported KV source %d", args.KVSource), nil)
	}
	if args.KVSource == hipAttentionKVSourceContiguous && (args.KeyPointer == 0 || args.ValuePointer == 0) {
		return nil, core.E("rocm.hip.AttentionHeadsLaunch", "key and value pointers are required", nil)
	}
	if args.KVSource == hipAttentionKVSourceDevice && (args.DescriptorPointer == 0 || args.DescriptorBytes < rocmDeviceKVDescriptorHeaderBytes) {
		return nil, core.E("rocm.hip.AttentionHeadsLaunch", "device KV descriptor is required", nil)
	}
	if args.Scale < 0 || math.IsNaN(float64(args.Scale)) || math.IsInf(float64(args.Scale), 0) {
		return nil, core.E("rocm.hip.AttentionHeadsLaunch", "scale must be non-negative and finite", nil)
	}
	dim, err := rocmDeviceKVPositiveUint32("dimension", args.Dim)
	if err != nil {
		return nil, err
	}
	tokenCount, err := rocmDeviceKVPositiveUint32("token count", args.TokenCount)
	if err != nil {
		return nil, err
	}
	headCount, err := rocmDeviceKVPositiveUint32("head count", args.HeadCount)
	if err != nil {
		return nil, err
	}
	keyHeads, err := rocmDeviceKVPositiveUint32("key head count", firstPositiveInt(args.KeyHeads, 1))
	if err != nil {
		return nil, err
	}
	if keyHeads > headCount || headCount%keyHeads != 0 {
		return nil, core.E("rocm.hip.AttentionHeadsLaunch", "key head count must divide query head count", nil)
	}
	queryBytes, err := hipAlignedFloat32Bytes("query", args.QueryBytes, dim*headCount)
	if err != nil {
		return nil, core.E("rocm.hip.AttentionHeadsLaunch", "query byte count", err)
	}
	var keyBytes uint32
	var valueBytes uint32
	if args.KVSource == hipAttentionKVSourceContiguous {
		keyBytes, err = hipAlignedFloat32Bytes("key", args.KeyBytes, dim*tokenCount*keyHeads)
		if err != nil {
			return nil, core.E("rocm.hip.AttentionHeadsLaunch", "key byte count", err)
		}
		valueBytes, err = hipAlignedFloat32Bytes("value", args.ValueBytes, dim*tokenCount*keyHeads)
		if err != nil {
			return nil, core.E("rocm.hip.AttentionHeadsLaunch", "value byte count", err)
		}
	} else if args.KeyBytes != 0 || args.ValueBytes != 0 || args.KeyPointer != 0 || args.ValuePointer != 0 {
		return nil, core.E("rocm.hip.AttentionHeadsLaunch", "device KV attention must not set contiguous KV pointers", nil)
	}
	outputBytes, err := hipAlignedFloat32Bytes("output", args.OutputBytes, dim*headCount)
	if err != nil {
		return nil, core.E("rocm.hip.AttentionHeadsLaunch", "output byte count", err)
	}
	windowSize, err := rocmDeviceKVUint32("window size", args.WindowSize)
	if err != nil {
		return nil, err
	}
	var weightBytes uint32
	if args.WeightPointer == 0 {
		if args.WeightBytes != 0 {
			return nil, core.E("rocm.hip.AttentionHeadsLaunch", "shared attention weights must not set weight bytes", nil)
		}
		if args.TokenCount > hipAttentionHeadsSharedMaxTokens {
			return nil, core.E("rocm.hip.AttentionHeadsLaunch", "shared attention weights token count exceeds limit", nil)
		}
	} else {
		weightBytes, err = hipAlignedFloat32Bytes("weight", args.WeightBytes, tokenCount*headCount)
		if err != nil {
			return nil, core.E("rocm.hip.AttentionHeadsLaunch", "weight byte count", err)
		}
	}
	if cap(payload) < hipAttentionHeadsLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipAttentionHeadsLaunchArgsBytes)
	} else {
		payload = payload[:hipAttentionHeadsLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipAttentionHeadsLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.QueryPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.KeyPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ValuePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint32(payload[48:], dim)
	binary.LittleEndian.PutUint32(payload[52:], tokenCount)
	binary.LittleEndian.PutUint32(payload[56:], headCount)
	binary.LittleEndian.PutUint32(payload[60:], queryBytes)
	binary.LittleEndian.PutUint32(payload[64:], keyBytes)
	binary.LittleEndian.PutUint32(payload[68:], valueBytes)
	binary.LittleEndian.PutUint32(payload[72:], outputBytes)
	binary.LittleEndian.PutUint32(payload[76:], weightBytes)
	binary.LittleEndian.PutUint32(payload[80:], args.KVSource)
	binary.LittleEndian.PutUint32(payload[84:], math.Float32bits(args.Scale))
	binary.LittleEndian.PutUint64(payload[88:], uint64(args.DescriptorPointer))
	binary.LittleEndian.PutUint64(payload[96:], args.DescriptorBytes)
	binary.LittleEndian.PutUint64(payload[104:], args.SharedMemBytes)
	binary.LittleEndian.PutUint32(payload[112:], windowSize)
	binary.LittleEndian.PutUint32(payload[116:], keyHeads)
	return payload, nil
}

func (args hipAttentionHeadsBatchCausalLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipAttentionHeadsBatchCausalLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.QueryPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "query and output pointers are required", nil)
	}
	if args.KVSource != hipAttentionKVSourceContiguous && args.KVSource != hipAttentionKVSourceDevice {
		return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", core.Sprintf("unsupported KV source %d", args.KVSource), nil)
	}
	if args.KVSource == hipAttentionKVSourceContiguous && (args.KeyPointer == 0 || args.ValuePointer == 0) {
		return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "key and value pointers are required", nil)
	}
	if args.KVSource == hipAttentionKVSourceDevice && (args.DescriptorPointer == 0 || args.DescriptorBytes < rocmDeviceKVDescriptorHeaderBytes) {
		return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "device KV descriptor is required", nil)
	}
	if args.Scale < 0 || math.IsNaN(float64(args.Scale)) || math.IsInf(float64(args.Scale), 0) {
		return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "scale must be non-negative and finite", nil)
	}
	dim, err := rocmDeviceKVPositiveUint32("dimension", args.Dim)
	if err != nil {
		return nil, err
	}
	tokenCount, err := rocmDeviceKVPositiveUint32("token count", args.TokenCount)
	if err != nil {
		return nil, err
	}
	headCount, err := rocmDeviceKVPositiveUint32("head count", args.HeadCount)
	if err != nil {
		return nil, err
	}
	keyHeads, err := rocmDeviceKVPositiveUint32("key head count", firstPositiveInt(args.KeyHeads, 1))
	if err != nil {
		return nil, err
	}
	if keyHeads > headCount || headCount%keyHeads != 0 {
		return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "key head count must divide query head count", nil)
	}
	queryCount, err := rocmDeviceKVPositiveUint32("query count", args.QueryCount)
	if err != nil {
		return nil, err
	}
	queryStartToken, err := rocmDeviceKVUint32("query start token", args.QueryStartToken)
	if err != nil {
		return nil, err
	}
	windowSize, err := rocmDeviceKVUint32("window size", args.WindowSize)
	if err != nil {
		return nil, err
	}
	if uint64(queryStartToken)+uint64(queryCount) > uint64(tokenCount) {
		return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "causal query window exceeds token count", nil)
	}
	if args.VisibleCapPointer == 0 {
		if args.VisibleCapBytes != 0 {
			return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "visible-token cap bytes require a cap pointer", nil)
		}
	} else if args.VisibleCapBytes != uint64(queryCount)*4 {
		return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "visible-token cap byte count mismatch", nil)
	}
	queryElements := uint64(dim) * uint64(headCount) * uint64(queryCount)
	queryBytes, err := hipExactUint32Bytes("query", args.QueryBytes, queryElements*4)
	if err != nil {
		return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "query byte count", err)
	}
	var keyBytes uint32
	var valueBytes uint32
	if args.KVSource == hipAttentionKVSourceContiguous {
		kvElements := uint64(dim) * uint64(tokenCount) * uint64(keyHeads)
		keyBytes, err = hipExactUint32Bytes("key", args.KeyBytes, kvElements*4)
		if err != nil {
			return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "key byte count", err)
		}
		valueBytes, err = hipExactUint32Bytes("value", args.ValueBytes, kvElements*4)
		if err != nil {
			return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "value byte count", err)
		}
	} else if args.KeyBytes != 0 || args.ValueBytes != 0 || args.KeyPointer != 0 || args.ValuePointer != 0 {
		return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "device KV attention must not set contiguous KV pointers", nil)
	}
	outputBytes, err := hipExactUint32Bytes("output", args.OutputBytes, queryElements*4)
	if err != nil {
		return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "output byte count", err)
	}
	var weightBytes uint32
	if args.WeightPointer == 0 {
		if args.WeightBytes != 0 {
			return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "shared attention weights must not set weight bytes", nil)
		}
		if args.TokenCount > hipAttentionHeadsSharedMaxTokens {
			return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "shared attention weights token count exceeds limit", nil)
		}
	} else {
		weightElements := uint64(queryCount) * uint64(headCount) * uint64(tokenCount)
		weightBytes, err = hipExactUint32Bytes("weight", args.WeightBytes, weightElements*4)
		if err != nil {
			return nil, core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "weight byte count", err)
		}
	}
	if cap(payload) < hipAttentionHeadsBatchCausalLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipAttentionHeadsBatchCausalLaunchArgsBytes)
	} else {
		payload = payload[:hipAttentionHeadsBatchCausalLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipAttentionHeadsBatchCausalLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.QueryPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.KeyPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ValuePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint32(payload[48:], dim)
	binary.LittleEndian.PutUint32(payload[52:], tokenCount)
	binary.LittleEndian.PutUint32(payload[56:], headCount)
	binary.LittleEndian.PutUint32(payload[60:], queryCount)
	binary.LittleEndian.PutUint32(payload[64:], queryStartToken)
	binary.LittleEndian.PutUint32(payload[68:], queryBytes)
	binary.LittleEndian.PutUint32(payload[72:], keyBytes)
	binary.LittleEndian.PutUint32(payload[76:], valueBytes)
	binary.LittleEndian.PutUint32(payload[80:], outputBytes)
	binary.LittleEndian.PutUint32(payload[84:], weightBytes)
	binary.LittleEndian.PutUint32(payload[88:], args.KVSource)
	binary.LittleEndian.PutUint32(payload[92:], math.Float32bits(args.Scale))
	binary.LittleEndian.PutUint64(payload[96:], uint64(args.DescriptorPointer))
	binary.LittleEndian.PutUint64(payload[104:], args.DescriptorBytes)
	binary.LittleEndian.PutUint64(payload[112:], args.SharedMemBytes)
	binary.LittleEndian.PutUint32(payload[120:], windowSize)
	binary.LittleEndian.PutUint32(payload[124:], keyHeads)
	binary.LittleEndian.PutUint64(payload[128:], uint64(args.VisibleCapPointer))
	binary.LittleEndian.PutUint64(payload[136:], args.VisibleCapBytes)
	return payload, nil
}

func (args hipAttentionHeadsLaneBatchLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipAttentionHeadsLaneBatchLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	const operation = "AttentionHeadsLaneBatchLaunch"
	if args.QueryPointer == 0 || args.OutputPointer == 0 || args.LanePointer == 0 {
		return nil, core.E("rocm.hip."+operation, "query, output, and lane pointers are required", nil)
	}
	if args.Scale < 0 || math.IsNaN(float64(args.Scale)) || math.IsInf(float64(args.Scale), 0) {
		return nil, core.E("rocm.hip."+operation, "scale must be non-negative and finite", nil)
	}
	dim, err := rocmDeviceKVPositiveUint32("dimension", args.Dim)
	if err != nil {
		return nil, err
	}
	laneCount, err := rocmDeviceKVPositiveUint32("lane count", args.LaneCount)
	if err != nil {
		return nil, err
	}
	headCount, err := rocmDeviceKVPositiveUint32("head count", args.HeadCount)
	if err != nil {
		return nil, err
	}
	keyHeads, err := rocmDeviceKVPositiveUint32("key head count", firstPositiveInt(args.KeyHeads, 1))
	if err != nil {
		return nil, err
	}
	if keyHeads > headCount || headCount%keyHeads != 0 {
		return nil, core.E("rocm.hip."+operation, "key head count must divide query head count", nil)
	}
	maxTokenCount, err := rocmDeviceKVPositiveUint32("maximum token count", args.MaxTokenCount)
	if err != nil {
		return nil, err
	}
	queryElements := uint64(dim) * uint64(headCount) * uint64(laneCount)
	queryBytes, err := hipExactUint32Bytes("query", args.QueryBytes, queryElements*4)
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "query byte count", err)
	}
	outputBytes, err := hipExactUint32Bytes("output", args.OutputBytes, queryElements*4)
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "output byte count", err)
	}
	laneBytes, err := hipExactUint32Bytes("lane descriptor", args.LaneBytes, uint64(laneCount)*hipAttentionHeadsLaneDescriptorBytes)
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "lane descriptor byte count", err)
	}
	var weightBytes uint32
	if args.WeightPointer == 0 {
		if args.WeightBytes != 0 {
			return nil, core.E("rocm.hip."+operation, "shared attention weights must not set weight bytes", nil)
		}
		if args.MaxTokenCount > hipAttentionHeadsSharedMaxTokens {
			return nil, core.E("rocm.hip."+operation, "shared attention weights token count exceeds limit", nil)
		}
	} else {
		weightElements := uint64(laneCount) * uint64(headCount) * uint64(maxTokenCount)
		weightBytes, err = hipExactUint32Bytes("weight", args.WeightBytes, weightElements*4)
		if err != nil {
			return nil, core.E("rocm.hip."+operation, "weight byte count", err)
		}
	}
	if cap(payload) < hipAttentionHeadsLaneBatchLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipAttentionHeadsLaneBatchLaunchArgsBytes)
	} else {
		payload = payload[:hipAttentionHeadsLaneBatchLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipAttentionHeadsLaneBatchLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.QueryPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.LanePointer))
	binary.LittleEndian.PutUint32(payload[40:], dim)
	binary.LittleEndian.PutUint32(payload[44:], laneCount)
	binary.LittleEndian.PutUint32(payload[48:], headCount)
	binary.LittleEndian.PutUint32(payload[52:], keyHeads)
	binary.LittleEndian.PutUint32(payload[56:], maxTokenCount)
	binary.LittleEndian.PutUint32(payload[60:], queryBytes)
	binary.LittleEndian.PutUint32(payload[64:], outputBytes)
	binary.LittleEndian.PutUint32(payload[68:], weightBytes)
	binary.LittleEndian.PutUint32(payload[72:], laneBytes)
	binary.LittleEndian.PutUint32(payload[76:], math.Float32bits(args.Scale))
	binary.LittleEndian.PutUint64(payload[80:], args.SharedMemBytes)
	return payload, nil
}

func (args hipAttentionHeadsBatchCausalQueryRMSRoPELaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipAttentionHeadsBatchCausalQueryRMSRoPELaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	const operation = "AttentionHeadsBatchCausalQueryRMSRoPELaunch"
	if args.QueryInputPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip."+operation, "query input and output pointers are required", nil)
	}
	if args.KVSource != hipAttentionKVSourceContiguous && args.KVSource != hipAttentionKVSourceDevice {
		return nil, core.E("rocm.hip."+operation, core.Sprintf("unsupported KV source %d", args.KVSource), nil)
	}
	if args.KVSource == hipAttentionKVSourceContiguous && (args.KeyPointer == 0 || args.ValuePointer == 0) {
		return nil, core.E("rocm.hip."+operation, "key and value pointers are required", nil)
	}
	if args.KVSource == hipAttentionKVSourceDevice && (args.DescriptorPointer == 0 || args.DescriptorBytes < rocmDeviceKVDescriptorHeaderBytes) {
		return nil, core.E("rocm.hip."+operation, "device KV descriptor is required", nil)
	}
	if args.Scale < 0 || math.IsNaN(float64(args.Scale)) || math.IsInf(float64(args.Scale), 0) {
		return nil, core.E("rocm.hip."+operation, "scale must be non-negative and finite", nil)
	}
	if args.QueryEpsilon < 0 || math.IsNaN(float64(args.QueryEpsilon)) || math.IsInf(float64(args.QueryEpsilon), 0) {
		return nil, core.E("rocm.hip."+operation, "query epsilon must be non-negative and finite", nil)
	}
	if args.RoPEStartPosition < 0 {
		return nil, core.E("rocm.hip."+operation, "RoPE start position must be non-negative", nil)
	}
	ropeStartPosition, err := rocmDeviceKVUint32("RoPE start position", args.RoPEStartPosition)
	if err != nil {
		return nil, err
	}
	if args.RoPEBase <= 0 || math.IsNaN(float64(args.RoPEBase)) || math.IsInf(float64(args.RoPEBase), 0) {
		return nil, core.E("rocm.hip."+operation, "RoPE base must be positive and finite", nil)
	}
	ropeFrequencyScale := args.RoPEFrequencyScale
	if ropeFrequencyScale == 0 {
		ropeFrequencyScale = 1
	}
	if ropeFrequencyScale <= 0 || math.IsNaN(float64(ropeFrequencyScale)) || math.IsInf(float64(ropeFrequencyScale), 0) {
		return nil, core.E("rocm.hip."+operation, "RoPE frequency scale must be positive and finite", nil)
	}
	dim, err := rocmDeviceKVPositiveUint32("dimension", args.Dim)
	if err != nil {
		return nil, err
	}
	if dim%2 != 0 {
		return nil, core.E("rocm.hip."+operation, "dimension must be even", nil)
	}
	tokenCount, err := rocmDeviceKVPositiveUint32("token count", args.TokenCount)
	if err != nil {
		return nil, err
	}
	headCount, err := rocmDeviceKVPositiveUint32("head count", args.HeadCount)
	if err != nil {
		return nil, err
	}
	keyHeads, err := rocmDeviceKVPositiveUint32("key head count", firstPositiveInt(args.KeyHeads, 1))
	if err != nil {
		return nil, err
	}
	if keyHeads > headCount || headCount%keyHeads != 0 {
		return nil, core.E("rocm.hip."+operation, "key head count must divide query head count", nil)
	}
	queryCount, err := rocmDeviceKVPositiveUint32("query count", args.QueryCount)
	if err != nil {
		return nil, err
	}
	queryStartToken, err := rocmDeviceKVUint32("query start token", args.QueryStartToken)
	if err != nil {
		return nil, err
	}
	windowSize, err := rocmDeviceKVUint32("window size", args.WindowSize)
	if err != nil {
		return nil, err
	}
	if uint64(queryStartToken)+uint64(queryCount) > uint64(tokenCount) {
		return nil, core.E("rocm.hip."+operation, "causal query window exceeds token count", nil)
	}
	if uint64(ropeStartPosition)+uint64(queryCount) > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip."+operation, "RoPE position range is out of uint32 range", nil)
	}
	if args.RoPEFrequencyDim < 0 || (args.RoPEFrequencyDim > 0 && args.RoPEFrequencyDim < args.Dim) {
		return nil, core.E("rocm.hip."+operation, "RoPE frequency dimension must be zero or at least dimension", nil)
	}
	ropeFrequencyDim, err := rocmDeviceKVUint32("RoPE frequency dimension", args.RoPEFrequencyDim)
	if err != nil {
		return nil, err
	}
	if args.RoPERotaryCount < 0 || args.RoPERotaryCount > args.Dim || args.RoPERotaryCount%2 != 0 {
		return nil, core.E("rocm.hip."+operation, "RoPE rotary count must be zero or an even count no larger than dimension", nil)
	}
	ropeRotaryCount, err := rocmDeviceKVUint32("RoPE rotary count", args.RoPERotaryCount)
	if err != nil {
		return nil, err
	}
	queryElements := uint64(dim) * uint64(headCount) * uint64(queryCount)
	queryInputBytes, err := hipExactUint32Bytes("query input", args.QueryInputBytes, queryElements*4)
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "query input byte count", err)
	}
	queryWeightBytes, err := hipRMSNormLaunchWeightBytes(operation, "query weight", args.QueryWeightPointer, args.QueryWeightBytes, dim, args.QueryWeightEncoding, args.QueryFlags)
	if err != nil {
		return nil, err
	}
	var keyBytes uint32
	var valueBytes uint32
	if args.KVSource == hipAttentionKVSourceContiguous {
		kvElements := uint64(dim) * uint64(tokenCount) * uint64(keyHeads)
		keyBytes, err = hipExactUint32Bytes("key", args.KeyBytes, kvElements*4)
		if err != nil {
			return nil, core.E("rocm.hip."+operation, "key byte count", err)
		}
		valueBytes, err = hipExactUint32Bytes("value", args.ValueBytes, kvElements*4)
		if err != nil {
			return nil, core.E("rocm.hip."+operation, "value byte count", err)
		}
	} else if args.KeyBytes != 0 || args.ValueBytes != 0 || args.KeyPointer != 0 || args.ValuePointer != 0 {
		return nil, core.E("rocm.hip."+operation, "device KV attention must not set contiguous KV pointers", nil)
	}
	outputBytes, err := hipExactUint32Bytes("output", args.OutputBytes, queryElements*4)
	if err != nil {
		return nil, core.E("rocm.hip."+operation, "output byte count", err)
	}
	var attentionWeightBytes uint32
	if args.AttentionWeightPointer == 0 {
		if args.AttentionWeightBytes != 0 {
			return nil, core.E("rocm.hip."+operation, "shared attention weights must not set weight bytes", nil)
		}
		if args.TokenCount > hipAttentionHeadsSharedMaxTokens {
			return nil, core.E("rocm.hip."+operation, "shared attention weights token count exceeds limit", nil)
		}
	} else {
		weightElements := uint64(queryCount) * uint64(headCount) * uint64(tokenCount)
		attentionWeightBytes, err = hipExactUint32Bytes("attention weight", args.AttentionWeightBytes, weightElements*4)
		if err != nil {
			return nil, core.E("rocm.hip."+operation, "attention weight byte count", err)
		}
	}
	if cap(payload) < hipAttentionHeadsBatchCausalQueryRMSRoPELaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipAttentionHeadsBatchCausalQueryRMSRoPELaunchArgsBytes)
	} else {
		payload = payload[:hipAttentionHeadsBatchCausalQueryRMSRoPELaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipAttentionHeadsBatchCausalQueryRMSRoPELaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.QueryInputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.QueryWeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.KeyPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.ValuePointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.AttentionWeightPointer))
	binary.LittleEndian.PutUint32(payload[56:], dim)
	binary.LittleEndian.PutUint32(payload[60:], tokenCount)
	binary.LittleEndian.PutUint32(payload[64:], headCount)
	binary.LittleEndian.PutUint32(payload[68:], queryCount)
	binary.LittleEndian.PutUint32(payload[72:], queryStartToken)
	binary.LittleEndian.PutUint32(payload[76:], queryInputBytes)
	binary.LittleEndian.PutUint32(payload[80:], queryWeightBytes)
	binary.LittleEndian.PutUint32(payload[84:], keyBytes)
	binary.LittleEndian.PutUint32(payload[88:], valueBytes)
	binary.LittleEndian.PutUint32(payload[92:], outputBytes)
	binary.LittleEndian.PutUint32(payload[96:], attentionWeightBytes)
	binary.LittleEndian.PutUint32(payload[100:], args.KVSource)
	binary.LittleEndian.PutUint32(payload[104:], math.Float32bits(args.Scale))
	binary.LittleEndian.PutUint32(payload[108:], keyHeads)
	binary.LittleEndian.PutUint64(payload[112:], uint64(args.DescriptorPointer))
	binary.LittleEndian.PutUint64(payload[120:], args.DescriptorBytes)
	binary.LittleEndian.PutUint64(payload[128:], args.SharedMemBytes)
	binary.LittleEndian.PutUint32(payload[136:], windowSize)
	binary.LittleEndian.PutUint32(payload[140:], math.Float32bits(args.QueryEpsilon))
	binary.LittleEndian.PutUint32(payload[144:], args.QueryWeightEncoding)
	binary.LittleEndian.PutUint32(payload[148:], args.QueryFlags)
	binary.LittleEndian.PutUint32(payload[152:], ropeStartPosition)
	binary.LittleEndian.PutUint32(payload[156:], math.Float32bits(args.RoPEBase))
	binary.LittleEndian.PutUint32(payload[160:], ropeFrequencyDim)
	binary.LittleEndian.PutUint32(payload[164:], ropeRotaryCount)
	binary.LittleEndian.PutUint32(payload[168:], math.Float32bits(ropeFrequencyScale))
	return payload, nil
}

func (args hipAttentionHeadsChunkedLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipAttentionHeadsChunkedLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.QueryPointer == 0 || args.DescriptorPointer == 0 || args.PartialPointer == 0 || args.StatsPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "query, descriptor, workspace, and output pointers are required", nil)
	}
	if args.Scale < 0 || math.IsNaN(float64(args.Scale)) || math.IsInf(float64(args.Scale), 0) {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "scale must be non-negative and finite", nil)
	}
	dim, err := rocmDeviceKVPositiveUint32("dimension", args.Dim)
	if err != nil {
		return nil, err
	}
	tokenCount, err := rocmDeviceKVPositiveUint32("token count", args.TokenCount)
	if err != nil {
		return nil, err
	}
	headCount, err := rocmDeviceKVPositiveUint32("head count", args.HeadCount)
	if err != nil {
		return nil, err
	}
	keyHeads, err := rocmDeviceKVPositiveUint32("key head count", firstPositiveInt(args.KeyHeads, 1))
	if err != nil {
		return nil, err
	}
	if keyHeads > headCount || headCount%keyHeads != 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "key head count must divide query head count", nil)
	}
	if keyHeads > 1 && dim&1 != 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "multi-KV q4 attention requires an even head dimension", nil)
	}
	chunkSize, err := rocmDeviceKVPositiveUint32("attention chunk size", args.ChunkSize)
	if err != nil {
		return nil, err
	}
	chunkCount, err := rocmDeviceKVPositiveUint32("attention chunk count", args.ChunkCount)
	if err != nil {
		return nil, err
	}
	windowSize, err := rocmDeviceKVUint32("window size", args.WindowSize)
	if err != nil {
		return nil, err
	}
	if uint64(chunkCount) != (uint64(tokenCount)+uint64(chunkSize)-1)/uint64(chunkSize) {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "chunk count must cover token count", nil)
	}
	queryBytes, err := hipAlignedFloat32Bytes("query", args.QueryBytes, dim*headCount)
	if err != nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "query byte count", err)
	}
	partialCount := uint64(headCount) * uint64(chunkCount) * uint64(dim)
	partialBytes, err := hipExactUint32Bytes("partial", args.PartialBytes, partialCount*4)
	if err != nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "partial byte count", err)
	}
	statsCount := uint64(headCount) * uint64(chunkCount) * 2
	statsBytes, err := hipExactUint32Bytes("stats", args.StatsBytes, statsCount*4)
	if err != nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "stats byte count", err)
	}
	outputBytes, err := hipAlignedFloat32Bytes("output", args.OutputBytes, dim*headCount)
	if err != nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "output byte count", err)
	}
	if args.DescriptorBytes < rocmDeviceKVDescriptorHeaderBytes {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "device KV descriptor is required", nil)
	}
	if cap(payload) < hipAttentionHeadsChunkedLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipAttentionHeadsChunkedLaunchArgsBytes)
	} else {
		payload = payload[:hipAttentionHeadsChunkedLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipAttentionHeadsChunkedLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.QueryPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.DescriptorPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.PartialPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.StatsPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[48:], dim)
	binary.LittleEndian.PutUint32(payload[52:], tokenCount)
	binary.LittleEndian.PutUint32(payload[56:], headCount)
	binary.LittleEndian.PutUint32(payload[60:], chunkSize)
	binary.LittleEndian.PutUint32(payload[64:], chunkCount)
	binary.LittleEndian.PutUint32(payload[68:], queryBytes)
	binary.LittleEndian.PutUint64(payload[72:], args.DescriptorBytes)
	binary.LittleEndian.PutUint32(payload[80:], partialBytes)
	binary.LittleEndian.PutUint32(payload[84:], statsBytes)
	binary.LittleEndian.PutUint32(payload[88:], outputBytes)
	binary.LittleEndian.PutUint32(payload[92:], math.Float32bits(args.Scale))
	binary.LittleEndian.PutUint32(payload[96:], windowSize)
	binary.LittleEndian.PutUint32(payload[100:], keyHeads)
	return payload, nil
}

func (args hipAttentionHeadsBatchChunkedLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipAttentionHeadsBatchChunkedLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.QueryPointer == 0 || args.DescriptorPointer == 0 || args.PartialPointer == 0 || args.StatsPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "query, descriptor, workspace, and output pointers are required", nil)
	}
	if args.Scale < 0 || math.IsNaN(float64(args.Scale)) || math.IsInf(float64(args.Scale), 0) {
		return nil, core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "scale must be non-negative and finite", nil)
	}
	dim, err := rocmDeviceKVPositiveUint32("dimension", args.Dim)
	if err != nil {
		return nil, err
	}
	tokenCount, err := rocmDeviceKVPositiveUint32("token count", args.TokenCount)
	if err != nil {
		return nil, err
	}
	headCount, err := rocmDeviceKVPositiveUint32("head count", args.HeadCount)
	if err != nil {
		return nil, err
	}
	keyHeads, err := rocmDeviceKVPositiveUint32("key head count", firstPositiveInt(args.KeyHeads, 1))
	if err != nil {
		return nil, err
	}
	if keyHeads > headCount || headCount%keyHeads != 0 {
		return nil, core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "key head count must divide query head count", nil)
	}
	if keyHeads > 1 && dim&1 != 0 {
		return nil, core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "multi-KV q4 attention requires an even head dimension", nil)
	}
	queryCount, err := rocmDeviceKVPositiveUint32("query count", args.QueryCount)
	if err != nil {
		return nil, err
	}
	queryStartToken, err := rocmDeviceKVUint32("query start token", args.QueryStartToken)
	if err != nil {
		return nil, err
	}
	windowSize, err := rocmDeviceKVUint32("window size", args.WindowSize)
	if err != nil {
		return nil, err
	}
	chunkStartToken, err := rocmDeviceKVUint32("chunk start token", args.ChunkStartToken)
	if err != nil {
		return nil, err
	}
	if uint64(queryStartToken)+uint64(queryCount) > uint64(tokenCount) {
		return nil, core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "causal query window exceeds token count", nil)
	}
	if args.VisibleCapPointer == 0 {
		if args.VisibleCapBytes != 0 {
			return nil, core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "visible-token cap bytes require a cap pointer", nil)
		}
	} else if args.VisibleCapBytes != uint64(queryCount)*4 {
		return nil, core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "visible-token cap byte count mismatch", nil)
	}
	chunkSize, err := rocmDeviceKVPositiveUint32("attention chunk size", args.ChunkSize)
	if err != nil {
		return nil, err
	}
	chunkCount, err := rocmDeviceKVPositiveUint32("attention chunk count", args.ChunkCount)
	if err != nil {
		return nil, err
	}
	chunkEndToken := int(queryStartToken) + int(queryCount)
	if args.VisibleCapPointer != 0 {
		chunkEndToken = int(tokenCount)
	}
	if chunkEndToken > int(tokenCount) {
		chunkEndToken = int(tokenCount)
	}
	if int(chunkStartToken) > chunkEndToken {
		return nil, core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "chunk start token exceeds active range", nil)
	}
	expectedChunkCount := (chunkEndToken - int(chunkStartToken) + int(chunkSize) - 1) / int(chunkSize)
	if expectedChunkCount <= 0 || int(chunkCount) != expectedChunkCount {
		return nil, core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "chunk count must cover active token range", nil)
	}
	queryElements := uint64(dim) * uint64(headCount) * uint64(queryCount)
	queryBytes, err := hipExactUint32Bytes("query", args.QueryBytes, queryElements*4)
	if err != nil {
		return nil, core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "query byte count", err)
	}
	partialCount := queryElements * uint64(chunkCount)
	partialBytes, err := hipExactUint32Bytes("partial", args.PartialBytes, partialCount*4)
	if err != nil {
		return nil, core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "partial byte count", err)
	}
	statsCount := uint64(queryCount) * uint64(headCount) * uint64(chunkCount) * 2
	statsBytes, err := hipExactUint32Bytes("stats", args.StatsBytes, statsCount*4)
	if err != nil {
		return nil, core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "stats byte count", err)
	}
	outputBytes, err := hipExactUint32Bytes("output", args.OutputBytes, queryElements*4)
	if err != nil {
		return nil, core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "output byte count", err)
	}
	if args.DescriptorBytes < rocmDeviceKVDescriptorHeaderBytes {
		return nil, core.E("rocm.hip.AttentionHeadsBatchChunkedLaunch", "device KV descriptor is required", nil)
	}
	if cap(payload) < hipAttentionHeadsBatchChunkedLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipAttentionHeadsBatchChunkedLaunchArgsBytes)
	} else {
		payload = payload[:hipAttentionHeadsBatchChunkedLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipAttentionHeadsBatchChunkedLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.QueryPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.DescriptorPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.PartialPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.StatsPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[48:], dim)
	binary.LittleEndian.PutUint32(payload[52:], tokenCount)
	binary.LittleEndian.PutUint32(payload[56:], headCount)
	binary.LittleEndian.PutUint32(payload[60:], queryCount)
	binary.LittleEndian.PutUint32(payload[64:], queryStartToken)
	binary.LittleEndian.PutUint32(payload[68:], chunkSize)
	binary.LittleEndian.PutUint32(payload[72:], chunkCount)
	binary.LittleEndian.PutUint32(payload[76:], queryBytes)
	binary.LittleEndian.PutUint64(payload[80:], args.DescriptorBytes)
	binary.LittleEndian.PutUint32(payload[88:], partialBytes)
	binary.LittleEndian.PutUint32(payload[92:], statsBytes)
	binary.LittleEndian.PutUint32(payload[96:], outputBytes)
	binary.LittleEndian.PutUint32(payload[100:], math.Float32bits(args.Scale))
	binary.LittleEndian.PutUint32(payload[104:], windowSize)
	binary.LittleEndian.PutUint32(payload[108:], chunkStartToken)
	binary.LittleEndian.PutUint32(payload[112:], keyHeads)
	binary.LittleEndian.PutUint32(payload[116:], uint32(args.VisibleCapBytes))
	binary.LittleEndian.PutUint64(payload[120:], uint64(args.VisibleCapPointer))
	return payload, nil
}

func (buffers *hipAttentionDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Weights, buffers.Output, buffers.Values, buffers.Keys, buffers.Query} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipAttentionDeviceBuffers) ReadOutput() (hipAttentionResult, error) {
	if buffers == nil || buffers.Output == nil || buffers.Weights == nil || buffers.Output.Pointer() == 0 || buffers.Weights.Pointer() == 0 {
		return hipAttentionResult{}, core.E("rocm.hip.AttentionLaunch", "attention output buffers are required", nil)
	}
	output, err := buffers.ReadOutputOnly()
	if err != nil {
		return hipAttentionResult{}, err
	}
	weights, err := hipReadFloat32DeviceOutput(buffers.Weights, "rocm.hip.AttentionLaunch", "attention weights", buffers.TokenCount)
	if err != nil {
		return hipAttentionResult{}, err
	}
	if !hipFloat32SliceProbabilities(weights) {
		return hipAttentionResult{}, core.E("rocm.hip.AttentionLaunch", "attention weights must be probabilities", nil)
	}
	return hipAttentionResult{Output: output, Weights: weights}, nil
}

func (buffers *hipAttentionDeviceBuffers) ReadOutputOnly() ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.AttentionLaunch", "attention output buffer is required", nil)
	}
	return hipReadFloat32DeviceOutput(buffers.Output, "rocm.hip.AttentionLaunch", "attention output", buffers.Dim)
}

func (req hipVectorAddRequest) validate() error {
	if len(req.Left) == 0 {
		return core.E("rocm.hip.VectorAddLaunch", "left input is required", nil)
	}
	if len(req.Right) != len(req.Left) {
		return core.E("rocm.hip.VectorAddLaunch", "right input length must match left input length", nil)
	}
	if !rocmFloat32SliceFinite(req.Left) || !rocmFloat32SliceFinite(req.Right) {
		return core.E("rocm.hip.VectorAddLaunch", "inputs must be finite", nil)
	}
	return nil
}

func (req hipVectorAddRequest) deviceBuffers(driver nativeHIPDriver) (*hipVectorAddDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	leftPayload, err := hipFloat32Payload(req.Left)
	if err != nil {
		return nil, core.E("rocm.hip.VectorAddLaunch", "encode left input", err)
	}
	left, err := hipUploadByteBuffer(driver, "rocm.hip.VectorAddLaunch", "vector add left input", leftPayload, len(req.Left))
	if err != nil {
		return nil, err
	}
	buffers := &hipVectorAddDeviceBuffers{Left: left, Count: len(req.Left)}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	rightPayload, err := hipFloat32Payload(req.Right)
	if err != nil {
		return nil, core.E("rocm.hip.VectorAddLaunch", "encode right input", err)
	}
	right, err := hipUploadByteBuffer(driver, "rocm.hip.VectorAddLaunch", "vector add right input", rightPayload, len(req.Right))
	if err != nil {
		return nil, err
	}
	buffers.Right = right
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.VectorAddLaunch", "vector add output", uint64(len(req.Left)*4), len(req.Left))
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipVectorAddRequest) launchArgs(buffers *hipVectorAddDeviceBuffers) (hipVectorAddLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipVectorAddLaunchArgs{}, err
	}
	if buffers == nil || buffers.Left == nil || buffers.Right == nil || buffers.Output == nil {
		return hipVectorAddLaunchArgs{}, core.E("rocm.hip.VectorAddLaunch", "vector add device buffers are required", nil)
	}
	if buffers.Left.Count() != len(req.Left) || buffers.Right.Count() != len(req.Left) || buffers.Output.Count() != len(req.Left) {
		return hipVectorAddLaunchArgs{}, core.E("rocm.hip.VectorAddLaunch", "vector add device buffer shape mismatch", nil)
	}
	return hipVectorAddLaunchArgs{
		LeftPointer:   buffers.Left.Pointer(),
		RightPointer:  buffers.Right.Pointer(),
		OutputPointer: buffers.Output.Pointer(),
		Count:         len(req.Left),
		LeftBytes:     buffers.Left.SizeBytes(),
		RightBytes:    buffers.Right.SizeBytes(),
		OutputBytes:   buffers.Output.SizeBytes(),
	}, nil
}

func (args hipVectorAddLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipVectorAddLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.LeftPointer == 0 || args.RightPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.VectorAddLaunch", "left, right, and output pointers are required", nil)
	}
	count, err := rocmDeviceKVPositiveUint32("count", args.Count)
	if err != nil {
		return nil, err
	}
	leftBytes, err := hipAlignedFloat32Bytes("left", args.LeftBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.VectorAddLaunch", "left byte count", err)
	}
	rightBytes, err := hipAlignedFloat32Bytes("right", args.RightBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.VectorAddLaunch", "right byte count", err)
	}
	outputBytes, err := hipAlignedFloat32Bytes("output", args.OutputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.VectorAddLaunch", "output byte count", err)
	}
	if cap(payload) < hipVectorAddLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipVectorAddLaunchArgsBytes)
	} else {
		payload = payload[:hipVectorAddLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipVectorAddLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.LeftPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.RightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], count)
	binary.LittleEndian.PutUint32(payload[36:], leftBytes)
	binary.LittleEndian.PutUint32(payload[40:], rightBytes)
	binary.LittleEndian.PutUint32(payload[44:], outputBytes)
	return payload, nil
}

func (args hipVectorAddScaledLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipVectorAddScaledLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.LeftPointer == 0 || args.RightPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.VectorAddScaledLaunch", "left, right, and output pointers are required", nil)
	}
	if math.IsNaN(float64(args.Scale)) || math.IsInf(float64(args.Scale), 0) {
		return nil, core.E("rocm.hip.VectorAddScaledLaunch", "scale must be finite", nil)
	}
	count, err := rocmDeviceKVPositiveUint32("count", args.Count)
	if err != nil {
		return nil, err
	}
	leftBytes, err := hipAlignedFloat32Bytes("left", args.LeftBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.VectorAddScaledLaunch", "left byte count", err)
	}
	rightBytes, err := hipAlignedFloat32Bytes("right", args.RightBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.VectorAddScaledLaunch", "right byte count", err)
	}
	outputBytes, err := hipAlignedFloat32Bytes("output", args.OutputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.VectorAddScaledLaunch", "output byte count", err)
	}
	if cap(payload) < hipVectorAddScaledLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipVectorAddScaledLaunchArgsBytes)
	} else {
		payload = payload[:hipVectorAddScaledLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipVectorAddScaledLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.LeftPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.RightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], count)
	binary.LittleEndian.PutUint32(payload[36:], leftBytes)
	binary.LittleEndian.PutUint32(payload[40:], rightBytes)
	binary.LittleEndian.PutUint32(payload[44:], outputBytes)
	binary.LittleEndian.PutUint32(payload[48:], math.Float32bits(args.Scale))
	return payload, nil
}

func (buffers *hipVectorAddDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Right, buffers.Left} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipVectorAddDeviceBuffers) ReadOutput() ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.VectorAddLaunch", "vector add output buffer is required", nil)
	}
	return hipReadFloat32DeviceOutput(buffers.Output, "rocm.hip.VectorAddLaunch", "vector add output", buffers.Count)
}

func (req hipVectorScaleRequest) validate() error {
	if len(req.Input) == 0 {
		return core.E("rocm.hip.VectorScaleLaunch", "input is required", nil)
	}
	if !rocmFloat32SliceFinite(req.Input) || math.IsNaN(float64(req.Scale)) || math.IsInf(float64(req.Scale), 0) {
		return core.E("rocm.hip.VectorScaleLaunch", "input and scale must be finite", nil)
	}
	return nil
}

func (req hipVectorScaleRequest) deviceBuffers(driver nativeHIPDriver) (*hipVectorScaleDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	inputPayload, err := hipFloat32Payload(req.Input)
	if err != nil {
		return nil, core.E("rocm.hip.VectorScaleLaunch", "encode input", err)
	}
	input, err := hipUploadByteBuffer(driver, "rocm.hip.VectorScaleLaunch", "vector scale input", inputPayload, len(req.Input))
	if err != nil {
		return nil, err
	}
	buffers := &hipVectorScaleDeviceBuffers{Input: input, Count: len(req.Input), Scale: req.Scale}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	output, err := hipAllocateByteBuffer(driver, "rocm.hip.VectorScaleLaunch", "vector scale output", uint64(len(req.Input)*4), len(req.Input))
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipVectorScaleRequest) launchArgs(buffers *hipVectorScaleDeviceBuffers) (hipVectorScaleLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipVectorScaleLaunchArgs{}, err
	}
	if buffers == nil || buffers.Input == nil || buffers.Output == nil {
		return hipVectorScaleLaunchArgs{}, core.E("rocm.hip.VectorScaleLaunch", "vector scale device buffers are required", nil)
	}
	if buffers.Input.Count() != len(req.Input) || buffers.Output.Count() != len(req.Input) || buffers.Count != len(req.Input) || buffers.Scale != req.Scale {
		return hipVectorScaleLaunchArgs{}, core.E("rocm.hip.VectorScaleLaunch", "vector scale device buffer shape mismatch", nil)
	}
	return hipVectorScaleLaunchArgs{
		InputPointer:  buffers.Input.Pointer(),
		OutputPointer: buffers.Output.Pointer(),
		Count:         len(req.Input),
		InputBytes:    buffers.Input.SizeBytes(),
		OutputBytes:   buffers.Output.SizeBytes(),
		Scale:         req.Scale,
	}, nil
}

func (args hipVectorScaleLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipVectorScaleLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.VectorScaleLaunch", "input and output pointers are required", nil)
	}
	if math.IsNaN(float64(args.Scale)) || math.IsInf(float64(args.Scale), 0) {
		return nil, core.E("rocm.hip.VectorScaleLaunch", "scale must be finite", nil)
	}
	count, err := rocmDeviceKVPositiveUint32("count", args.Count)
	if err != nil {
		return nil, err
	}
	inputBytes, err := hipAlignedFloat32Bytes("input", args.InputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.VectorScaleLaunch", "input byte count", err)
	}
	outputBytes, err := hipAlignedFloat32Bytes("output", args.OutputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.VectorScaleLaunch", "output byte count", err)
	}
	if cap(payload) < hipVectorScaleLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipVectorScaleLaunchArgsBytes)
	} else {
		payload = payload[:hipVectorScaleLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipVectorScaleLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[24:], count)
	binary.LittleEndian.PutUint32(payload[28:], inputBytes)
	binary.LittleEndian.PutUint32(payload[32:], outputBytes)
	binary.LittleEndian.PutUint32(payload[36:], math.Float32bits(args.Scale))
	return payload, nil
}

func (buffers *hipVectorScaleDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Input} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipVectorScaleDeviceBuffers) ReadOutput() ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.VectorScaleLaunch", "vector scale output buffer is required", nil)
	}
	return hipReadFloat32DeviceOutput(buffers.Output, "rocm.hip.VectorScaleLaunch", "vector scale output", buffers.Count)
}

func (req hipSwiGLURequest) validate() error {
	if len(req.Gate) == 0 {
		return core.E("rocm.hip.SwiGLULaunch", "gate input is required", nil)
	}
	if len(req.Up) != len(req.Gate) {
		return core.E("rocm.hip.SwiGLULaunch", "up input length must match gate input length", nil)
	}
	if !rocmFloat32SliceFinite(req.Gate) || !rocmFloat32SliceFinite(req.Up) {
		return core.E("rocm.hip.SwiGLULaunch", "inputs must be finite", nil)
	}
	return nil
}

func (req hipSwiGLURequest) deviceBuffers(driver nativeHIPDriver) (*hipSwiGLUDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	gatePayload, err := hipFloat32Payload(req.Gate)
	if err != nil {
		return nil, core.E("rocm.hip.SwiGLULaunch", "encode gate input", err)
	}
	gate, err := hipUploadByteBuffer(driver, "rocm.hip.SwiGLULaunch", "swiglu gate input", gatePayload, len(req.Gate))
	if err != nil {
		return nil, err
	}
	buffers := &hipSwiGLUDeviceBuffers{Gate: gate, Count: len(req.Gate)}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	upPayload, err := hipFloat32Payload(req.Up)
	if err != nil {
		return nil, core.E("rocm.hip.SwiGLULaunch", "encode up input", err)
	}
	up, err := hipUploadByteBuffer(driver, "rocm.hip.SwiGLULaunch", "swiglu up input", upPayload, len(req.Up))
	if err != nil {
		return nil, err
	}
	buffers.Up = up
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.SwiGLULaunch", "swiglu output", uint64(len(req.Gate)*4), len(req.Gate))
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipSwiGLURequest) launchArgs(buffers *hipSwiGLUDeviceBuffers) (hipSwiGLULaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipSwiGLULaunchArgs{}, err
	}
	if buffers == nil || buffers.Gate == nil || buffers.Up == nil || buffers.Output == nil {
		return hipSwiGLULaunchArgs{}, core.E("rocm.hip.SwiGLULaunch", "swiglu device buffers are required", nil)
	}
	if buffers.Gate.Count() != len(req.Gate) || buffers.Up.Count() != len(req.Gate) || buffers.Output.Count() != len(req.Gate) {
		return hipSwiGLULaunchArgs{}, core.E("rocm.hip.SwiGLULaunch", "swiglu device buffer shape mismatch", nil)
	}
	return hipSwiGLULaunchArgs{
		GatePointer:   buffers.Gate.Pointer(),
		UpPointer:     buffers.Up.Pointer(),
		OutputPointer: buffers.Output.Pointer(),
		Count:         len(req.Gate),
		GateBytes:     buffers.Gate.SizeBytes(),
		UpBytes:       buffers.Up.SizeBytes(),
		OutputBytes:   buffers.Output.SizeBytes(),
	}, nil
}

func (args hipSwiGLULaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipSwiGLULaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.GatePointer == 0 || args.UpPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.SwiGLULaunch", "gate, up, and output pointers are required", nil)
	}
	count, err := rocmDeviceKVPositiveUint32("count", args.Count)
	if err != nil {
		return nil, err
	}
	gateBytes, err := hipAlignedFloat32Bytes("gate", args.GateBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.SwiGLULaunch", "gate byte count", err)
	}
	upBytes, err := hipAlignedFloat32Bytes("up", args.UpBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.SwiGLULaunch", "up byte count", err)
	}
	outputBytes, err := hipAlignedFloat32Bytes("output", args.OutputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.SwiGLULaunch", "output byte count", err)
	}
	if cap(payload) < hipSwiGLULaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipSwiGLULaunchArgsBytes)
	} else {
		payload = payload[:hipSwiGLULaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipSwiGLULaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.GatePointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.UpPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], count)
	binary.LittleEndian.PutUint32(payload[36:], gateBytes)
	binary.LittleEndian.PutUint32(payload[40:], upBytes)
	binary.LittleEndian.PutUint32(payload[44:], outputBytes)
	return payload, nil
}

func (buffers *hipSwiGLUDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Up, buffers.Gate} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipSwiGLUDeviceBuffers) ReadOutput() ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.SwiGLULaunch", "swiglu output buffer is required", nil)
	}
	return hipReadFloat32DeviceOutput(buffers.Output, "rocm.hip.SwiGLULaunch", "swiglu output", buffers.Count)
}

func (req hipGELUTanhMultiplyRequest) validate() error {
	if len(req.Gate) == 0 {
		return core.E("rocm.hip.GELUTanhMultiplyLaunch", "gate input is required", nil)
	}
	if len(req.Up) != len(req.Gate) {
		return core.E("rocm.hip.GELUTanhMultiplyLaunch", "up input length must match gate input length", nil)
	}
	if !rocmFloat32SliceFinite(req.Gate) || !rocmFloat32SliceFinite(req.Up) {
		return core.E("rocm.hip.GELUTanhMultiplyLaunch", "inputs must be finite", nil)
	}
	return nil
}

func (req hipGELUTanhMultiplyRequest) deviceBuffers(driver nativeHIPDriver) (*hipGELUTanhMultiplyDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	gatePayload, err := hipFloat32Payload(req.Gate)
	if err != nil {
		return nil, core.E("rocm.hip.GELUTanhMultiplyLaunch", "encode gate input", err)
	}
	gate, err := hipUploadByteBuffer(driver, "rocm.hip.GELUTanhMultiplyLaunch", "GELU tanh multiply gate input", gatePayload, len(req.Gate))
	if err != nil {
		return nil, err
	}
	buffers := &hipGELUTanhMultiplyDeviceBuffers{Gate: gate, Count: len(req.Gate)}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	upPayload, err := hipFloat32Payload(req.Up)
	if err != nil {
		return nil, core.E("rocm.hip.GELUTanhMultiplyLaunch", "encode up input", err)
	}
	up, err := hipUploadByteBuffer(driver, "rocm.hip.GELUTanhMultiplyLaunch", "GELU tanh multiply up input", upPayload, len(req.Up))
	if err != nil {
		return nil, err
	}
	buffers.Up = up
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.GELUTanhMultiplyLaunch", "GELU tanh multiply output", uint64(len(req.Gate)*4), len(req.Gate))
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipGELUTanhMultiplyRequest) launchArgs(buffers *hipGELUTanhMultiplyDeviceBuffers) (hipGELUTanhMultiplyLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipGELUTanhMultiplyLaunchArgs{}, err
	}
	if buffers == nil ||
		buffers.Gate == nil ||
		buffers.Up == nil ||
		buffers.Output == nil ||
		buffers.Gate.Count() != len(req.Gate) ||
		buffers.Up.Count() != len(req.Gate) ||
		buffers.Output.Count() != len(req.Gate) {
		return hipGELUTanhMultiplyLaunchArgs{}, core.E("rocm.hip.GELUTanhMultiplyLaunch", "GELU tanh multiply device buffer shape mismatch", nil)
	}
	return hipGELUTanhMultiplyLaunchArgsForDeviceBuffers(buffers)
}

func hipGELUTanhMultiplyLaunchArgsForDeviceBuffers(buffers *hipGELUTanhMultiplyDeviceBuffers) (hipGELUTanhMultiplyLaunchArgs, error) {
	if buffers == nil || buffers.Gate == nil || buffers.Up == nil || buffers.Output == nil {
		return hipGELUTanhMultiplyLaunchArgs{}, core.E("rocm.hip.GELUTanhMultiplyLaunch", "GELU tanh multiply device buffers are required", nil)
	}
	if buffers.Count <= 0 ||
		buffers.Gate.Count() != buffers.Count ||
		buffers.Up.Count() != buffers.Count ||
		buffers.Output.Count() != buffers.Count {
		return hipGELUTanhMultiplyLaunchArgs{}, core.E("rocm.hip.GELUTanhMultiplyLaunch", "GELU tanh multiply device buffer shape mismatch", nil)
	}
	return hipGELUTanhMultiplyLaunchArgs{
		GatePointer:   buffers.Gate.Pointer(),
		UpPointer:     buffers.Up.Pointer(),
		OutputPointer: buffers.Output.Pointer(),
		Count:         buffers.Count,
		GateBytes:     buffers.Gate.SizeBytes(),
		UpBytes:       buffers.Up.SizeBytes(),
		OutputBytes:   buffers.Output.SizeBytes(),
	}, nil
}

func (args hipGELUTanhMultiplyLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipGELUTanhMultiplyLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.GatePointer == 0 || args.UpPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.GELUTanhMultiplyLaunch", "gate, up, and output pointers are required", nil)
	}
	count, err := rocmDeviceKVPositiveUint32("count", args.Count)
	if err != nil {
		return nil, err
	}
	gateBytes, err := hipAlignedFloat32Bytes("gate", args.GateBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.GELUTanhMultiplyLaunch", "gate byte count", err)
	}
	upBytes, err := hipAlignedFloat32Bytes("up", args.UpBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.GELUTanhMultiplyLaunch", "up byte count", err)
	}
	outputBytes, err := hipAlignedFloat32Bytes("output", args.OutputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.GELUTanhMultiplyLaunch", "output byte count", err)
	}
	if cap(payload) < hipGELUTanhMulLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipGELUTanhMulLaunchArgsBytes)
	} else {
		payload = payload[:hipGELUTanhMulLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipGELUTanhMulLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.GatePointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.UpPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], count)
	binary.LittleEndian.PutUint32(payload[36:], gateBytes)
	binary.LittleEndian.PutUint32(payload[40:], upBytes)
	binary.LittleEndian.PutUint32(payload[44:], outputBytes)
	return payload, nil
}

func (buffers *hipGELUTanhMultiplyDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Up, buffers.Gate} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipGELUTanhMultiplyDeviceBuffers) ReadOutput() ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.GELUTanhMultiplyLaunch", "GELU tanh multiply output buffer is required", nil)
	}
	return hipReadFloat32DeviceOutput(buffers.Output, "rocm.hip.GELUTanhMultiplyLaunch", "GELU tanh multiply output", buffers.Count)
}

func hipTinyOutputWeightEncoding(fp32 []float32, fp16 []uint16, q8 []int8, q8Scale float32) (uint32, error) {
	encodings := 0
	if len(fp32) > 0 {
		encodings++
	}
	if len(fp16) > 0 {
		encodings++
	}
	if len(q8) > 0 {
		encodings++
	}
	if encodings != 1 {
		return 0, core.E("rocm.hip.TinyOutputWeights", "exactly one output weight encoding is required", nil)
	}
	if len(fp32) > 0 {
		return hipTinyOutputWeightEncodingFP32, nil
	}
	if len(fp16) > 0 {
		return hipTinyOutputWeightEncodingFP16, nil
	}
	if !hipQ8ScaleIsPositiveFinite(q8Scale) {
		return 0, core.E("rocm.hip.TinyOutputWeights", "q8 scale must be positive and finite", nil)
	}
	return hipTinyOutputWeightEncodingQ8, nil
}

func hipTinyOutputWeightCount(fp32 []float32, fp16 []uint16, q8 []int8) int {
	switch {
	case len(fp32) > 0:
		return len(fp32)
	case len(fp16) > 0:
		return len(fp16)
	default:
		return len(q8)
	}
}

func hipTinyOutputWeightPayload(fp32 []float32, fp16 []uint16, q8 []int8, q8Scale float32) ([]byte, int, uint32, float32, error) {
	encoding, err := hipTinyOutputWeightEncoding(fp32, fp16, q8, q8Scale)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	switch encoding {
	case hipTinyOutputWeightEncodingFP32:
		payload, err := hipFloat32Payload(fp32)
		return payload, len(fp32), encoding, 0, err
	case hipTinyOutputWeightEncodingFP16:
		payload, err := hipUint16Payload(fp16)
		return payload, len(fp16), encoding, 0, err
	case hipTinyOutputWeightEncodingQ8:
		return hipInt8Payload(q8), len(q8), encoding, q8Scale, nil
	default:
		return nil, 0, 0, 0, core.E("rocm.hip.TinyOutputWeights", "unsupported output weight encoding", nil)
	}
}

func hipTinyOutputWeightByteCount(encoding uint32, sizeBytes, tableCount uint64, q8Scale float32) (uint32, error) {
	switch encoding {
	case hipTinyOutputWeightEncodingFP32:
		return hipExactUint32Bytes("output weight", sizeBytes, tableCount*4)
	case hipTinyOutputWeightEncodingFP16:
		return hipExactUint32Bytes("output weight", sizeBytes, tableCount*2)
	case hipTinyOutputWeightEncodingQ8:
		if !hipQ8ScaleIsPositiveFinite(q8Scale) {
			return 0, core.E("rocm.hip.TinyOutputWeights", "q8 scale must be positive and finite", nil)
		}
		return hipExactUint32Bytes("output weight", sizeBytes, tableCount)
	default:
		return 0, core.E("rocm.hip.TinyOutputWeights", "unsupported output weight encoding", nil)
	}
}

func hipTinyOutputWeightValues(payload []byte, encoding uint32, q8Scale float32) ([]float32, error) {
	var values []float32
	switch encoding {
	case hipTinyOutputWeightEncodingFP32:
		decoded, err := hipFloat32PayloadValues(payload)
		if err != nil {
			return nil, err
		}
		values = decoded
	case hipTinyOutputWeightEncodingFP16:
		if len(payload) == 0 || len(payload)%2 != 0 {
			return nil, core.E("rocm.hip.TinyOutputWeights", "fp16 payload byte length must be positive and aligned", nil)
		}
		values = make([]float32, len(payload)/2)
		for index := range values {
			values[index] = hipFloat16ToFloat32(binary.LittleEndian.Uint16(payload[index*2:]))
		}
	case hipTinyOutputWeightEncodingQ8:
		if len(payload) == 0 {
			return nil, core.E("rocm.hip.TinyOutputWeights", "q8 payload is empty", nil)
		}
		if !hipQ8ScaleIsPositiveFinite(q8Scale) {
			return nil, core.E("rocm.hip.TinyOutputWeights", "q8 scale must be positive and finite", nil)
		}
		values = make([]float32, len(payload))
		for index, value := range payload {
			values[index] = float32(int8(value)) * q8Scale
		}
	default:
		return nil, core.E("rocm.hip.TinyOutputWeights", "unsupported output weight encoding", nil)
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E("rocm.hip.TinyOutputWeights", "output weight values must be finite", nil)
	}
	return values, nil
}

func hipQ8ScaleIsPositiveFinite(scale float32) bool {
	return scale > 0 && !math.IsNaN(float64(scale)) && !math.IsInf(float64(scale), 0)
}

func (req hipTinyPrefillRequest) validate() error {
	if len(req.TokenIDs) == 0 {
		return core.E("rocm.hip.TinyPrefillLaunch", "token IDs are required", nil)
	}
	if req.VocabSize <= 0 || req.HiddenSize <= 0 {
		return core.E("rocm.hip.TinyPrefillLaunch", "vocab and hidden size must be positive", nil)
	}
	tableCount := req.VocabSize * req.HiddenSize
	if len(req.EmbeddingTable) != tableCount {
		return core.E("rocm.hip.TinyPrefillLaunch", "embedding table length must match vocab*hidden", nil)
	}
	if _, err := hipTinyOutputWeightEncoding(req.OutputWeights, req.OutputFP16, req.OutputQ8, req.Q8Scale); err != nil {
		return core.E("rocm.hip.TinyPrefillLaunch", "output weight encoding", err)
	}
	if hipTinyOutputWeightCount(req.OutputWeights, req.OutputFP16, req.OutputQ8) != tableCount {
		return core.E("rocm.hip.TinyPrefillLaunch", "output weight length must match vocab*hidden", nil)
	}
	for _, id := range req.TokenIDs {
		if id < 0 || int(id) >= req.VocabSize {
			return core.E("rocm.hip.TinyPrefillLaunch", "token ID is outside vocabulary", nil)
		}
	}
	return nil
}

func (req hipTinyPrefillRequest) deviceBuffers(driver nativeHIPDriver) (*hipTinyPrefillDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	tokenPayload, err := hipTokenIDsPayload(req.TokenIDs)
	if err != nil {
		return nil, core.E("rocm.hip.TinyPrefillLaunch", "encode token IDs", err)
	}
	tokens, err := hipUploadByteBuffer(driver, "rocm.hip.TinyPrefillLaunch", "tiny prefill tokens", tokenPayload, len(req.TokenIDs))
	if err != nil {
		return nil, err
	}
	buffers := &hipTinyPrefillDeviceBuffers{
		Tokens:     tokens,
		TokenCount: len(req.TokenIDs),
		VocabSize:  req.VocabSize,
		HiddenSize: req.HiddenSize,
	}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	embeddingPayload, err := hipFloat32Payload(req.EmbeddingTable)
	if err != nil {
		return nil, core.E("rocm.hip.TinyPrefillLaunch", "encode embedding table", err)
	}
	embedding, err := hipUploadByteBuffer(driver, "rocm.hip.TinyPrefillLaunch", "tiny prefill embedding table", embeddingPayload, len(req.EmbeddingTable))
	if err != nil {
		return nil, err
	}
	buffers.EmbeddingTable = embedding
	weightPayload, weightCount, _, _, err := hipTinyOutputWeightPayload(req.OutputWeights, req.OutputFP16, req.OutputQ8, req.Q8Scale)
	if err != nil {
		return nil, core.E("rocm.hip.TinyPrefillLaunch", "encode output weights", err)
	}
	weights, err := hipUploadByteBuffer(driver, "rocm.hip.TinyPrefillLaunch", "tiny prefill output weights", weightPayload, weightCount)
	if err != nil {
		return nil, err
	}
	buffers.OutputWeights = weights
	logits, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyPrefillLaunch", "tiny prefill logits", uint64(req.VocabSize*4), req.VocabSize)
	if err != nil {
		return nil, err
	}
	buffers.Logits = logits
	attention, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyPrefillLaunch", "tiny prefill attention", uint64(len(req.TokenIDs)*4), len(req.TokenIDs))
	if err != nil {
		return nil, err
	}
	buffers.Attention = attention
	stateCount := len(req.TokenIDs) * req.HiddenSize
	keys, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyPrefillLaunch", "tiny prefill keys", uint64(stateCount*4), stateCount)
	if err != nil {
		return nil, err
	}
	buffers.Keys = keys
	values, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyPrefillLaunch", "tiny prefill values", uint64(stateCount*4), stateCount)
	if err != nil {
		return nil, err
	}
	buffers.Values = values
	result, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyPrefillLaunch", "tiny prefill result", hipGreedyResultBytes, 1)
	if err != nil {
		return nil, err
	}
	buffers.Result = result
	success = true
	return buffers, nil
}

func (req hipTinyPrefillRequest) launchArgs(buffers *hipTinyPrefillDeviceBuffers) (hipTinyPrefillLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipTinyPrefillLaunchArgs{}, err
	}
	if buffers == nil || buffers.Tokens == nil || buffers.EmbeddingTable == nil || buffers.OutputWeights == nil ||
		buffers.Logits == nil || buffers.Attention == nil || buffers.Keys == nil || buffers.Values == nil || buffers.Result == nil {
		return hipTinyPrefillLaunchArgs{}, core.E("rocm.hip.TinyPrefillLaunch", "tiny prefill device buffers are required", nil)
	}
	stateCount := len(req.TokenIDs) * req.HiddenSize
	if buffers.Tokens.Count() != len(req.TokenIDs) ||
		buffers.EmbeddingTable.Count() != len(req.EmbeddingTable) ||
		buffers.OutputWeights.Count() != hipTinyOutputWeightCount(req.OutputWeights, req.OutputFP16, req.OutputQ8) ||
		buffers.Logits.Count() != req.VocabSize ||
		buffers.Attention.Count() != len(req.TokenIDs) ||
		buffers.Keys.Count() != stateCount ||
		buffers.Values.Count() != stateCount ||
		buffers.Result.SizeBytes() != hipGreedyResultBytes ||
		buffers.TokenCount != len(req.TokenIDs) ||
		buffers.VocabSize != req.VocabSize ||
		buffers.HiddenSize != req.HiddenSize {
		return hipTinyPrefillLaunchArgs{}, core.E("rocm.hip.TinyPrefillLaunch", "tiny prefill device buffer shape mismatch", nil)
	}
	encoding, err := hipTinyOutputWeightEncoding(req.OutputWeights, req.OutputFP16, req.OutputQ8, req.Q8Scale)
	if err != nil {
		return hipTinyPrefillLaunchArgs{}, core.E("rocm.hip.TinyPrefillLaunch", "output weight encoding", err)
	}
	return hipTinyPrefillLaunchArgs{
		TokenPointer:         buffers.Tokens.Pointer(),
		EmbeddingPointer:     buffers.EmbeddingTable.Pointer(),
		OutputWeightPointer:  buffers.OutputWeights.Pointer(),
		LogitPointer:         buffers.Logits.Pointer(),
		AttentionPointer:     buffers.Attention.Pointer(),
		ResultPointer:        buffers.Result.Pointer(),
		KeyPointer:           buffers.Keys.Pointer(),
		ValuePointer:         buffers.Values.Pointer(),
		TokenCount:           len(req.TokenIDs),
		VocabSize:            req.VocabSize,
		HiddenSize:           req.HiddenSize,
		TokenBytes:           buffers.Tokens.SizeBytes(),
		EmbeddingBytes:       buffers.EmbeddingTable.SizeBytes(),
		OutputWeightBytes:    buffers.OutputWeights.SizeBytes(),
		LogitBytes:           buffers.Logits.SizeBytes(),
		AttentionBytes:       buffers.Attention.SizeBytes(),
		ResultBytes:          buffers.Result.SizeBytes(),
		KeyBytes:             buffers.Keys.SizeBytes(),
		ValueBytes:           buffers.Values.SizeBytes(),
		OutputWeightEncoding: encoding,
		Q8Scale:              req.Q8Scale,
	}, nil
}

func (args hipTinyPrefillLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipTinyPrefillLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.TokenPointer == 0 || args.EmbeddingPointer == 0 || args.OutputWeightPointer == 0 ||
		args.LogitPointer == 0 || args.AttentionPointer == 0 || args.ResultPointer == 0 ||
		args.KeyPointer == 0 || args.ValuePointer == 0 {
		return nil, core.E("rocm.hip.TinyPrefillLaunch", "token, weight, key/value, and output pointers are required", nil)
	}
	tokenCount, err := rocmDeviceKVPositiveUint32("token count", args.TokenCount)
	if err != nil {
		return nil, err
	}
	vocabSize, err := rocmDeviceKVPositiveUint32("vocab size", args.VocabSize)
	if err != nil {
		return nil, err
	}
	hiddenSize, err := rocmDeviceKVPositiveUint32("hidden size", args.HiddenSize)
	if err != nil {
		return nil, err
	}
	tableCount := uint64(vocabSize) * uint64(hiddenSize)
	stateCount := uint64(tokenCount) * uint64(hiddenSize)
	tokenBytes, err := hipExactUint32Bytes("token", args.TokenBytes, uint64(tokenCount)*4)
	if err != nil {
		return nil, core.E("rocm.hip.TinyPrefillLaunch", "token byte count", err)
	}
	embeddingBytes, err := hipExactUint32Bytes("embedding", args.EmbeddingBytes, tableCount*4)
	if err != nil {
		return nil, core.E("rocm.hip.TinyPrefillLaunch", "embedding byte count", err)
	}
	outputWeightBytes, err := hipTinyOutputWeightByteCount(args.OutputWeightEncoding, args.OutputWeightBytes, tableCount, args.Q8Scale)
	if err != nil {
		return nil, core.E("rocm.hip.TinyPrefillLaunch", "output weight byte count", err)
	}
	logitBytes, err := hipExactUint32Bytes("logit", args.LogitBytes, uint64(vocabSize)*4)
	if err != nil {
		return nil, core.E("rocm.hip.TinyPrefillLaunch", "logit byte count", err)
	}
	attentionBytes, err := hipExactUint32Bytes("attention", args.AttentionBytes, uint64(tokenCount)*4)
	if err != nil {
		return nil, core.E("rocm.hip.TinyPrefillLaunch", "attention byte count", err)
	}
	resultBytes, err := hipExactUint32Bytes("result", args.ResultBytes, hipGreedyResultBytes)
	if err != nil {
		return nil, core.E("rocm.hip.TinyPrefillLaunch", "result byte count", err)
	}
	keyBytes, err := hipExactUint32Bytes("key", args.KeyBytes, stateCount*4)
	if err != nil {
		return nil, core.E("rocm.hip.TinyPrefillLaunch", "key byte count", err)
	}
	valueBytes, err := hipExactUint32Bytes("value", args.ValueBytes, stateCount*4)
	if err != nil {
		return nil, core.E("rocm.hip.TinyPrefillLaunch", "value byte count", err)
	}
	if cap(payload) < hipTinyPrefillLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipTinyPrefillLaunchArgsBytes)
	} else {
		payload = payload[:hipTinyPrefillLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipTinyPrefillLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.TokenPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.EmbeddingPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputWeightPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.LogitPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.AttentionPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.ResultPointer))
	binary.LittleEndian.PutUint64(payload[56:], uint64(args.KeyPointer))
	binary.LittleEndian.PutUint64(payload[64:], uint64(args.ValuePointer))
	binary.LittleEndian.PutUint32(payload[72:], tokenCount)
	binary.LittleEndian.PutUint32(payload[76:], vocabSize)
	binary.LittleEndian.PutUint32(payload[80:], hiddenSize)
	binary.LittleEndian.PutUint32(payload[84:], tokenBytes)
	binary.LittleEndian.PutUint32(payload[88:], embeddingBytes)
	binary.LittleEndian.PutUint32(payload[92:], outputWeightBytes)
	binary.LittleEndian.PutUint32(payload[96:], logitBytes)
	binary.LittleEndian.PutUint32(payload[100:], attentionBytes)
	binary.LittleEndian.PutUint32(payload[104:], resultBytes)
	binary.LittleEndian.PutUint32(payload[108:], keyBytes)
	binary.LittleEndian.PutUint32(payload[112:], valueBytes)
	binary.LittleEndian.PutUint32(payload[116:], args.OutputWeightEncoding)
	binary.LittleEndian.PutUint32(payload[120:], math.Float32bits(args.Q8Scale))
	return payload, nil
}

func hipExactUint32Bytes(label string, sizeBytes, want uint64) (uint32, error) {
	if sizeBytes != want {
		return 0, core.E("rocm.hip.LaunchBytes", label+" bytes must match expected byte count", nil)
	}
	if sizeBytes > uint64(^uint32(0)) {
		return 0, core.E("rocm.hip.LaunchBytes", label+" bytes are out of uint32 range", nil)
	}
	return uint32(sizeBytes), nil
}

func (buffers *hipTinyPrefillDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Result, buffers.Values, buffers.Keys, buffers.Attention, buffers.Logits, buffers.OutputWeights, buffers.EmbeddingTable, buffers.Tokens} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipTinyPrefillDeviceBuffers) ReadOutput() (hipTinyPrefillResult, error) {
	if buffers == nil || buffers.Logits == nil || buffers.Attention == nil || buffers.Keys == nil ||
		buffers.Values == nil || buffers.Result == nil || buffers.Logits.Pointer() == 0 ||
		buffers.Attention.Pointer() == 0 || buffers.Keys.Pointer() == 0 ||
		buffers.Values.Pointer() == 0 || buffers.Result.Pointer() == 0 {
		return hipTinyPrefillResult{}, core.E("rocm.hip.TinyPrefillLaunch", "tiny prefill output buffers are required", nil)
	}
	logits, err := hipReadFloat32DeviceOutput(buffers.Logits, "rocm.hip.TinyPrefillLaunch", "tiny prefill logits", buffers.VocabSize)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	attention, err := hipReadFloat32DeviceOutput(buffers.Attention, "rocm.hip.TinyPrefillLaunch", "tiny prefill attention", buffers.TokenCount)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	if !hipFloat32SliceProbabilities(attention) {
		return hipTinyPrefillResult{}, core.E("rocm.hip.TinyPrefillLaunch", "tiny prefill attention must be probabilities", nil)
	}
	stateKeys, err := hipReadFloat32DeviceOutput(buffers.Keys, "rocm.hip.TinyPrefillLaunch", "tiny prefill keys", buffers.TokenCount*buffers.HiddenSize)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	stateValues, err := hipReadFloat32DeviceOutput(buffers.Values, "rocm.hip.TinyPrefillLaunch", "tiny prefill values", buffers.TokenCount*buffers.HiddenSize)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	result, err := hipReadGreedyResult(buffers.Result, "rocm.hip.TinyPrefillLaunch", "tiny prefill result", buffers.VocabSize)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	return hipTinyPrefillResult{
		Logits:      logits,
		Attention:   attention,
		StateKeys:   stateKeys,
		StateValues: stateValues,
		NextTokenID: result.TokenID,
		NextScore:   result.Score,
	}, nil
}

func (req hipTinyDecodeRequest) validate() error {
	if req.TokenID < 0 {
		return core.E("rocm.hip.TinyDecodeLaunch", "token ID must be non-negative", nil)
	}
	if req.VocabSize <= 0 || req.HiddenSize <= 0 {
		return core.E("rocm.hip.TinyDecodeLaunch", "vocab and hidden size must be positive", nil)
	}
	if int(req.TokenID) >= req.VocabSize {
		return core.E("rocm.hip.TinyDecodeLaunch", "token ID is outside vocabulary", nil)
	}
	tableCount := req.VocabSize * req.HiddenSize
	if len(req.EmbeddingTable) != tableCount {
		return core.E("rocm.hip.TinyDecodeLaunch", "embedding table length must match vocab*hidden", nil)
	}
	if _, err := hipTinyOutputWeightEncoding(req.OutputWeights, req.OutputFP16, req.OutputQ8, req.Q8Scale); err != nil {
		return core.E("rocm.hip.TinyDecodeLaunch", "output weight encoding", err)
	}
	if hipTinyOutputWeightCount(req.OutputWeights, req.OutputFP16, req.OutputQ8) != tableCount {
		return core.E("rocm.hip.TinyDecodeLaunch", "output weight length must match vocab*hidden", nil)
	}
	if len(req.PriorKeys) == 0 || len(req.PriorValues) == 0 {
		return core.E("rocm.hip.TinyDecodeLaunch", "prior key/value tensors are required", nil)
	}
	if len(req.PriorKeys) != len(req.PriorValues) {
		return core.E("rocm.hip.TinyDecodeLaunch", "prior key/value tensor lengths must match", nil)
	}
	if len(req.PriorKeys)%req.HiddenSize != 0 {
		return core.E("rocm.hip.TinyDecodeLaunch", "prior key/value tensor lengths must align with hidden size", nil)
	}
	return nil
}

func (req hipTinyDecodeRequest) shape() (int, error) {
	if err := req.validate(); err != nil {
		return 0, err
	}
	return len(req.PriorKeys) / req.HiddenSize, nil
}

func (req hipTinyDecodeRequest) deviceBuffers(driver nativeHIPDriver) (*hipTinyDecodeDeviceBuffers, error) {
	priorTokenCount, err := req.shape()
	if err != nil {
		return nil, err
	}
	keyPayload, err := hipFloat32Payload(req.PriorKeys)
	if err != nil {
		return nil, core.E("rocm.hip.TinyDecodeLaunch", "encode prior keys", err)
	}
	keys, err := hipUploadByteBuffer(driver, "rocm.hip.TinyDecodeLaunch", "tiny decode prior keys", keyPayload, len(req.PriorKeys))
	if err != nil {
		return nil, err
	}
	buffers := &hipTinyDecodeDeviceBuffers{
		PriorKeys:       keys,
		PriorTokenCount: priorTokenCount,
		VocabSize:       req.VocabSize,
		HiddenSize:      req.HiddenSize,
	}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	valuePayload, err := hipFloat32Payload(req.PriorValues)
	if err != nil {
		return nil, core.E("rocm.hip.TinyDecodeLaunch", "encode prior values", err)
	}
	values, err := hipUploadByteBuffer(driver, "rocm.hip.TinyDecodeLaunch", "tiny decode prior values", valuePayload, len(req.PriorValues))
	if err != nil {
		return nil, err
	}
	buffers.PriorValues = values
	embeddingPayload, err := hipFloat32Payload(req.EmbeddingTable)
	if err != nil {
		return nil, core.E("rocm.hip.TinyDecodeLaunch", "encode embedding table", err)
	}
	embedding, err := hipUploadByteBuffer(driver, "rocm.hip.TinyDecodeLaunch", "tiny decode embedding table", embeddingPayload, len(req.EmbeddingTable))
	if err != nil {
		return nil, err
	}
	buffers.EmbeddingTable = embedding
	weightPayload, weightCount, _, _, err := hipTinyOutputWeightPayload(req.OutputWeights, req.OutputFP16, req.OutputQ8, req.Q8Scale)
	if err != nil {
		return nil, core.E("rocm.hip.TinyDecodeLaunch", "encode output weights", err)
	}
	weights, err := hipUploadByteBuffer(driver, "rocm.hip.TinyDecodeLaunch", "tiny decode output weights", weightPayload, weightCount)
	if err != nil {
		return nil, err
	}
	buffers.OutputWeights = weights
	logits, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyDecodeLaunch", "tiny decode logits", uint64(req.VocabSize*4), req.VocabSize)
	if err != nil {
		return nil, err
	}
	buffers.Logits = logits
	attention, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyDecodeLaunch", "tiny decode attention", uint64((priorTokenCount+1)*4), priorTokenCount+1)
	if err != nil {
		return nil, err
	}
	buffers.Attention = attention
	updatedKeys, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyDecodeLaunch", "tiny decode updated keys", uint64((priorTokenCount+1)*req.HiddenSize*4), (priorTokenCount+1)*req.HiddenSize)
	if err != nil {
		return nil, err
	}
	buffers.UpdatedKeys = updatedKeys
	updatedValues, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyDecodeLaunch", "tiny decode updated values", uint64((priorTokenCount+1)*req.HiddenSize*4), (priorTokenCount+1)*req.HiddenSize)
	if err != nil {
		return nil, err
	}
	buffers.UpdatedValues = updatedValues
	result, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyDecodeLaunch", "tiny decode result", hipGreedyResultBytes, 1)
	if err != nil {
		return nil, err
	}
	buffers.Result = result
	success = true
	return buffers, nil
}

func (req hipTinyDecodeRequest) launchArgs(buffers *hipTinyDecodeDeviceBuffers) (hipTinyDecodeLaunchArgs, error) {
	priorTokenCount, err := req.shape()
	if err != nil {
		return hipTinyDecodeLaunchArgs{}, err
	}
	if buffers == nil || buffers.PriorKeys == nil || buffers.PriorValues == nil || buffers.EmbeddingTable == nil ||
		buffers.OutputWeights == nil || buffers.Logits == nil || buffers.Attention == nil ||
		buffers.UpdatedKeys == nil || buffers.UpdatedValues == nil || buffers.Result == nil {
		return hipTinyDecodeLaunchArgs{}, core.E("rocm.hip.TinyDecodeLaunch", "tiny decode device buffers are required", nil)
	}
	updatedCount := (priorTokenCount + 1) * req.HiddenSize
	if buffers.PriorKeys.Count() != len(req.PriorKeys) ||
		buffers.PriorValues.Count() != len(req.PriorValues) ||
		buffers.EmbeddingTable.Count() != len(req.EmbeddingTable) ||
		buffers.OutputWeights.Count() != hipTinyOutputWeightCount(req.OutputWeights, req.OutputFP16, req.OutputQ8) ||
		buffers.Logits.Count() != req.VocabSize ||
		buffers.Attention.Count() != priorTokenCount+1 ||
		buffers.UpdatedKeys.Count() != updatedCount ||
		buffers.UpdatedValues.Count() != updatedCount ||
		buffers.Result.SizeBytes() != hipGreedyResultBytes ||
		buffers.PriorTokenCount != priorTokenCount ||
		buffers.VocabSize != req.VocabSize ||
		buffers.HiddenSize != req.HiddenSize {
		return hipTinyDecodeLaunchArgs{}, core.E("rocm.hip.TinyDecodeLaunch", "tiny decode device buffer shape mismatch", nil)
	}
	encoding, err := hipTinyOutputWeightEncoding(req.OutputWeights, req.OutputFP16, req.OutputQ8, req.Q8Scale)
	if err != nil {
		return hipTinyDecodeLaunchArgs{}, core.E("rocm.hip.TinyDecodeLaunch", "output weight encoding", err)
	}
	return hipTinyDecodeLaunchArgs{
		PriorKeyPointer:      buffers.PriorKeys.Pointer(),
		PriorValuePointer:    buffers.PriorValues.Pointer(),
		EmbeddingPointer:     buffers.EmbeddingTable.Pointer(),
		OutputWeightPointer:  buffers.OutputWeights.Pointer(),
		LogitPointer:         buffers.Logits.Pointer(),
		AttentionPointer:     buffers.Attention.Pointer(),
		UpdatedKeyPointer:    buffers.UpdatedKeys.Pointer(),
		UpdatedValuePointer:  buffers.UpdatedValues.Pointer(),
		ResultPointer:        buffers.Result.Pointer(),
		TokenID:              req.TokenID,
		PriorTokenCount:      priorTokenCount,
		VocabSize:            req.VocabSize,
		HiddenSize:           req.HiddenSize,
		PriorKeyBytes:        buffers.PriorKeys.SizeBytes(),
		PriorValueBytes:      buffers.PriorValues.SizeBytes(),
		EmbeddingBytes:       buffers.EmbeddingTable.SizeBytes(),
		OutputWeightBytes:    buffers.OutputWeights.SizeBytes(),
		LogitBytes:           buffers.Logits.SizeBytes(),
		AttentionBytes:       buffers.Attention.SizeBytes(),
		UpdatedKeyBytes:      buffers.UpdatedKeys.SizeBytes(),
		UpdatedValueBytes:    buffers.UpdatedValues.SizeBytes(),
		ResultBytes:          buffers.Result.SizeBytes(),
		OutputWeightEncoding: encoding,
		Q8Scale:              req.Q8Scale,
	}, nil
}

func (args hipTinyDecodeLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipTinyDecodeLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.PriorKeyPointer == 0 || args.PriorValuePointer == 0 || args.EmbeddingPointer == 0 ||
		args.OutputWeightPointer == 0 || args.LogitPointer == 0 || args.AttentionPointer == 0 ||
		args.UpdatedKeyPointer == 0 || args.UpdatedValuePointer == 0 || args.ResultPointer == 0 {
		return nil, core.E("rocm.hip.TinyDecodeLaunch", "key, weight, and output pointers are required", nil)
	}
	if args.TokenID < 0 {
		return nil, core.E("rocm.hip.TinyDecodeLaunch", "token ID must be non-negative", nil)
	}
	priorTokenCount, err := rocmDeviceKVPositiveUint32("prior token count", args.PriorTokenCount)
	if err != nil {
		return nil, err
	}
	vocabSize, err := rocmDeviceKVPositiveUint32("vocab size", args.VocabSize)
	if err != nil {
		return nil, err
	}
	if uint32(args.TokenID) >= vocabSize {
		return nil, core.E("rocm.hip.TinyDecodeLaunch", "token ID is outside vocabulary", nil)
	}
	hiddenSize, err := rocmDeviceKVPositiveUint32("hidden size", args.HiddenSize)
	if err != nil {
		return nil, err
	}
	priorVectorCount := uint64(priorTokenCount) * uint64(hiddenSize)
	updatedTokenCount := uint64(priorTokenCount) + 1
	updatedVectorCount := updatedTokenCount * uint64(hiddenSize)
	tableCount := uint64(vocabSize) * uint64(hiddenSize)
	priorKeyBytes, err := hipExactUint32Bytes("prior key", args.PriorKeyBytes, priorVectorCount*4)
	if err != nil {
		return nil, core.E("rocm.hip.TinyDecodeLaunch", "prior key byte count", err)
	}
	priorValueBytes, err := hipExactUint32Bytes("prior value", args.PriorValueBytes, priorVectorCount*4)
	if err != nil {
		return nil, core.E("rocm.hip.TinyDecodeLaunch", "prior value byte count", err)
	}
	embeddingBytes, err := hipExactUint32Bytes("embedding", args.EmbeddingBytes, tableCount*4)
	if err != nil {
		return nil, core.E("rocm.hip.TinyDecodeLaunch", "embedding byte count", err)
	}
	outputWeightBytes, err := hipTinyOutputWeightByteCount(args.OutputWeightEncoding, args.OutputWeightBytes, tableCount, args.Q8Scale)
	if err != nil {
		return nil, core.E("rocm.hip.TinyDecodeLaunch", "output weight byte count", err)
	}
	logitBytes, err := hipExactUint32Bytes("logit", args.LogitBytes, uint64(vocabSize)*4)
	if err != nil {
		return nil, core.E("rocm.hip.TinyDecodeLaunch", "logit byte count", err)
	}
	attentionBytes, err := hipExactUint32Bytes("attention", args.AttentionBytes, updatedTokenCount*4)
	if err != nil {
		return nil, core.E("rocm.hip.TinyDecodeLaunch", "attention byte count", err)
	}
	updatedKeyBytes, err := hipExactUint32Bytes("updated key", args.UpdatedKeyBytes, updatedVectorCount*4)
	if err != nil {
		return nil, core.E("rocm.hip.TinyDecodeLaunch", "updated key byte count", err)
	}
	updatedValueBytes, err := hipExactUint32Bytes("updated value", args.UpdatedValueBytes, updatedVectorCount*4)
	if err != nil {
		return nil, core.E("rocm.hip.TinyDecodeLaunch", "updated value byte count", err)
	}
	resultBytes, err := hipExactUint32Bytes("result", args.ResultBytes, hipGreedyResultBytes)
	if err != nil {
		return nil, core.E("rocm.hip.TinyDecodeLaunch", "result byte count", err)
	}
	if cap(payload) < hipTinyDecodeLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipTinyDecodeLaunchArgsBytes)
	} else {
		payload = payload[:hipTinyDecodeLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipTinyDecodeLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.PriorKeyPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.PriorValuePointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.EmbeddingPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.OutputWeightPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.LogitPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.AttentionPointer))
	binary.LittleEndian.PutUint64(payload[56:], uint64(args.UpdatedKeyPointer))
	binary.LittleEndian.PutUint64(payload[64:], uint64(args.UpdatedValuePointer))
	binary.LittleEndian.PutUint64(payload[72:], uint64(args.ResultPointer))
	binary.LittleEndian.PutUint32(payload[80:], uint32(args.TokenID))
	binary.LittleEndian.PutUint32(payload[84:], priorTokenCount)
	binary.LittleEndian.PutUint32(payload[88:], vocabSize)
	binary.LittleEndian.PutUint32(payload[92:], hiddenSize)
	binary.LittleEndian.PutUint32(payload[96:], priorKeyBytes)
	binary.LittleEndian.PutUint32(payload[100:], priorValueBytes)
	binary.LittleEndian.PutUint32(payload[104:], embeddingBytes)
	binary.LittleEndian.PutUint32(payload[108:], outputWeightBytes)
	binary.LittleEndian.PutUint32(payload[112:], logitBytes)
	binary.LittleEndian.PutUint32(payload[116:], attentionBytes)
	binary.LittleEndian.PutUint32(payload[120:], updatedKeyBytes)
	binary.LittleEndian.PutUint32(payload[124:], updatedValueBytes)
	binary.LittleEndian.PutUint32(payload[128:], resultBytes)
	binary.LittleEndian.PutUint32(payload[132:], args.OutputWeightEncoding)
	binary.LittleEndian.PutUint32(payload[136:], math.Float32bits(args.Q8Scale))
	return payload, nil
}

func (buffers *hipTinyDecodeDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Result, buffers.UpdatedValues, buffers.UpdatedKeys, buffers.Attention, buffers.Logits, buffers.OutputWeights, buffers.EmbeddingTable, buffers.PriorValues, buffers.PriorKeys} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipTinyDecodeDeviceBuffers) ReadOutput() (hipTinyDecodeResult, error) {
	if buffers == nil || buffers.Logits == nil || buffers.Attention == nil || buffers.UpdatedKeys == nil ||
		buffers.UpdatedValues == nil || buffers.Result == nil || buffers.Logits.Pointer() == 0 ||
		buffers.Attention.Pointer() == 0 || buffers.UpdatedKeys.Pointer() == 0 ||
		buffers.UpdatedValues.Pointer() == 0 || buffers.Result.Pointer() == 0 {
		return hipTinyDecodeResult{}, core.E("rocm.hip.TinyDecodeLaunch", "tiny decode output buffers are required", nil)
	}
	tokenCount := buffers.PriorTokenCount + 1
	logits, err := hipReadFloat32DeviceOutput(buffers.Logits, "rocm.hip.TinyDecodeLaunch", "tiny decode logits", buffers.VocabSize)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	attention, err := hipReadFloat32DeviceOutput(buffers.Attention, "rocm.hip.TinyDecodeLaunch", "tiny decode attention", tokenCount)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	if !hipFloat32SliceProbabilities(attention) {
		return hipTinyDecodeResult{}, core.E("rocm.hip.TinyDecodeLaunch", "tiny decode attention must be probabilities", nil)
	}
	updatedKeys, err := hipReadFloat32DeviceOutput(buffers.UpdatedKeys, "rocm.hip.TinyDecodeLaunch", "tiny decode updated keys", tokenCount*buffers.HiddenSize)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	updatedValues, err := hipReadFloat32DeviceOutput(buffers.UpdatedValues, "rocm.hip.TinyDecodeLaunch", "tiny decode updated values", tokenCount*buffers.HiddenSize)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	result, err := hipReadGreedyResult(buffers.Result, "rocm.hip.TinyDecodeLaunch", "tiny decode result", buffers.VocabSize)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	return hipTinyDecodeResult{
		Logits:        logits,
		Attention:     attention,
		UpdatedKeys:   updatedKeys,
		UpdatedValues: updatedValues,
		NextTokenID:   result.TokenID,
		NextScore:     result.Score,
	}, nil
}
