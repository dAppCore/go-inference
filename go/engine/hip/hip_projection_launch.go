// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"strconv"
	"sync"
	"sync/atomic"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const (
	hipProjectionLaunchArgsVersion                 uint32 = 1
	hipProjectionLaunchArgsBytes                          = 96
	hipProjectionBatchLaunchArgsVersion            uint32 = 1
	hipProjectionBatchLaunchArgsBytes                     = 104
	hipMLXQ4ProjectionLaunchArgsVersion            uint32 = 1
	hipMLXQ4ProjectionLaunchArgsBytes                     = 96
	hipMLXQ4ProjectionBatchLaunchArgsVersion       uint32 = 1
	hipMLXQ4ProjectionBatchLaunchArgsBytes                = 96
	hipMLXQ4ProjectionGreedyBatchLaunchArgsVersion uint32 = 1
	hipMLXQ4ProjectionGreedyBatchLaunchArgsBytes          = 104
	hipMLXQ4TripleProjLaunchArgsVersion            uint32 = 1
	hipMLXQ4TripleProjLaunchArgsBytes                     = 168
	hipMLXQ4GELUTanhMulLaunchArgsVersion           uint32 = 1
	hipMLXQ4GELUTanhMulLaunchArgsBytes                    = 128
	hipMLXQ4GELUTanhMulBatchLaunchArgsVersion      uint32 = 1
	hipMLXQ4GELUTanhMulBatchLaunchArgsBytes               = 128
	hipMLXQ4GELUTanhMLPPersistentLaunchArgsVersion uint32 = 1
	hipMLXQ4GELUTanhMLPPersistentLaunchArgsBytes          = 184
	hipMLXQ4GELUTanhMLPPersistentRouteEnv                 = "GO_ROCM_ENABLE_EXPERIMENTAL_PERSISTENT_MLP"
	hipMLXQ4Projection12BDownRouteEnv                     = "GO_ROCM_ENABLE_EXPERIMENTAL_12B_DOWN_PROJECTION"
	hipMLXQ4GELUTanh12BGateUpRouteEnv                     = "GO_ROCM_ENABLE_EXPERIMENTAL_12B_GATE_UP"
	hipMLXQ4GELUTanh12BGateUpGeometryEnv                  = "GO_ROCM_EXPERIMENTAL_12B_GATE_UP_GEOMETRY"
	hipMLXQ4Projection12BHeadGridEnv                      = "GO_ROCM_EXPERIMENTAL_12B_HEAD_GRID"
	hipMLXQ4GELUTanhProjLaunchArgsVersion          uint32 = 1
	hipMLXQ4GELUTanhProjLaunchArgsBytes                   = 96
	hipMLXQ4GELUTanhProjBatchLaunchArgsVersion     uint32 = 1
	hipMLXQ4GELUTanhProjBatchLaunchArgsBytes              = 104
	hipRMSResidualAddGELUTanhProjLaunchArgsVersion uint32 = 1
	hipRMSResidualAddGELUTanhProjLaunchArgsBytes          = 160
	hipPackedTopKLaunchArgsVersion                 uint32 = 1
	hipPackedTopKLaunchArgsBytes                          = 48
	hipPackedTopKSampleLaunchArgsVersion           uint32 = 2
	hipPackedTopKSampleLaunchArgsBytes                    = 64
	hipOrderedEmbeddingCandidatesLaunchArgsVersion uint32 = 1
	hipOrderedEmbeddingCandidatesLaunchArgsBytes          = 80
	hipOrderedEmbeddingCandidatesBlockSize         uint32 = 256
	hipMLXQ4ProjectionBits                                = 4
	hipMLXQ4ProjectionBlockSize                    uint32 = 256
	hipMLXQ4ProjectionRowsPerBlock                        = 8
	hipMLXQ4ProjectionRow16RowsPerBlock                   = 16
	hipMLXQ4ProjectionCols256RowsPerBlock                 = 32
	hipMLXQ4ProjectionQ6Row16RowsPerBlock                 = 16
	hipMLXQ4ProjectionQ6Row32RowsPerBlock                 = 32
	hipMLXQ4ProjectionQ6Row64RowsPerBlock                 = 64
	hipMLXQ4GELUTanhQ6Cols1536RowsPerBlock                = 16
	hipMLXQ4GELUTanhQ4G32Cols1536Row16RowsPerBlock        = 16
	hipMLXQ4GELUTanh12BRow8RowsPerBlock                   = 8
	hipMLXQ4GELUTanhMLPPersistentBlocks            uint32 = 120
	hipMLXQ4GELUTanhQ6Cols1536Row32RowsPerBlock           = 32
	hipMLXQ4GELUTanhQ6Cols1536Row64RowsPerBlock           = 64
	hipMLXQ4ProjectionBatchTokensPerBlock                 = 8
	hipMLXQ4ProjectionGreedyRowsPerBlock                  = 32
	hipMLXQ4ProjectionGreedyQ6RowsPerBlock                = 64
	hipMLXQ4ProjectionBestBytes                           = 8
	hipPackedTopKMaxK                                     = 128
	hipPackedTopKBlockSize                         uint32 = 256
	hipPackedTopKChunkSize                                = 4096
)

var (
	hipMLXQ4GELUTanhMLPPersistentRouteEnabled = os.Getenv(hipMLXQ4GELUTanhMLPPersistentRouteEnv) == "1"
	hipMLXQ4Projection12BDownRouteEnabled     = os.Getenv(hipMLXQ4Projection12BDownRouteEnv) == "1"
	hipMLXQ4GELUTanh12BGateUpRouteEnabled     = os.Getenv(hipMLXQ4GELUTanh12BGateUpRouteEnv) == "1"
	hipMLXQ4GELUTanh12BGateUpGeometry         = os.Getenv(hipMLXQ4GELUTanh12BGateUpGeometryEnv)
	hipMLXQ4Projection12BHeadGridBlocks       = hipExperimentalProjectionGridBlocks(hipMLXQ4Projection12BHeadGridEnv)
)

func hipExperimentalProjectionGridBlocks(name string) int {
	blocks, err := strconv.Atoi(os.Getenv(name))
	if err != nil || blocks <= 0 {
		return 0
	}
	return blocks
}

const (
	hipProjectionWeightEncodingFP16 uint32 = 1
	hipProjectionWeightEncodingQ8   uint32 = 2
	hipProjectionWeightEncodingF32  uint32 = 3
	hipProjectionWeightEncodingBF16 uint32 = 4
)

const hipProjectionLaunchFlagBias uint32 = 1

type hipDeviceByteBuffer struct {
	driver    nativeHIPDriver
	pointer   nativeDevicePointer
	count     int
	sizeBytes uint64
	closed    bool
	borrowed  bool
	pooled    bool
	label     string
}

type hipDeviceByteBufferPoolEntry struct {
	driver  nativeHIPDriver
	pointer nativeDevicePointer
}

type hipDeviceByteBufferPoolSingleSlot struct {
	sizeBytes uint64
	entries   [hipDeviceByteBufferPoolSingleSlotCapacity]hipDeviceByteBufferPoolEntry
	count     uint8
}

type hipDeviceAllocationLabelRecorder interface {
	RecordDeviceAllocationLabel(sizeBytes uint64, operation, label string)
}

var hipDeviceByteBufferPool = struct {
	sync.Mutex
	single  [hipDeviceByteBufferPoolSingleSlots]hipDeviceByteBufferPoolSingleSlot
	entries map[uint64][]hipDeviceByteBufferPoolEntry
	bytes   uint64
}{
	entries: make(map[uint64][]hipDeviceByteBufferPoolEntry),
}

var hipDeviceByteBufferPoolSuppressions atomic.Int64

const (
	hipDeviceByteBufferPoolMaxBytes           = 768 << 20
	hipDeviceByteBufferPoolMaxPerSize         = 512
	hipDeviceByteBufferPoolSingleSlots        = 64
	hipDeviceByteBufferPoolSingleSlotCapacity = 3
)

func hipProjectionUint32Bytes(operation, label string, value uint64) error {
	if value > uint64(^uint32(0)) {
		return core.E(operation, label+" are out of uint32 range", nil)
	}
	return nil
}

type hipProjectionDeviceBuffers struct {
	Input    *hipDeviceByteBuffer
	Weights  *hipDeviceByteBuffer
	Bias     *hipDeviceByteBuffer
	Output   *hipDeviceByteBuffer
	Encoding uint32
	Q8Scale  float32
	Rows     int
	Cols     int
}

type hipProjectionLaunchArgs struct {
	InputPointer   nativeDevicePointer
	InputCount     int
	InputBytes     uint64
	WeightPointer  nativeDevicePointer
	WeightBytes    uint64
	BiasPointer    nativeDevicePointer
	BiasBytes      uint64
	OutputPointer  nativeDevicePointer
	OutputBytes    uint64
	Rows           int
	Cols           int
	WeightEncoding uint32
	Flags          uint32
	Q8Scale        float32
}

type hipProjectionBatchLaunchArgs struct {
	InputPointer   nativeDevicePointer
	WeightPointer  nativeDevicePointer
	WeightBytes    uint64
	BiasPointer    nativeDevicePointer
	BiasBytes      uint64
	OutputPointer  nativeDevicePointer
	InputBytes     uint64
	OutputBytes    uint64
	Rows           int
	Cols           int
	Batch          int
	WeightEncoding uint32
	Flags          uint32
	Q8Scale        float32
}

type hipMLXQ4ProjectionRequest struct {
	Input     []float32
	Weight    []uint32
	Scales    []uint16
	Biases    []uint16
	Rows      int
	Cols      int
	GroupSize int
	Bits      int
}

type hipMLXQ4ProjectionDeviceBuffers struct {
	Input     *hipDeviceByteBuffer
	Weight    *hipDeviceByteBuffer
	Scales    *hipDeviceByteBuffer
	Biases    *hipDeviceByteBuffer
	Output    *hipDeviceByteBuffer
	Rows      int
	Cols      int
	GroupSize int
	Bits      int
}

type hipMLXQ4DeviceWeightConfig struct {
	WeightPointer  nativeDevicePointer
	ScalePointer   nativeDevicePointer
	BiasPointer    nativeDevicePointer
	WeightBytes    uint64
	ScaleBytes     uint64
	BiasBytes      uint64
	Rows           int
	Cols           int
	GroupSize      int
	Bits           int
	WeightEncoding uint32
}

type hipMLXQ4ProjectionLaunchArgs struct {
	InputPointer    nativeDevicePointer
	WeightPointer   nativeDevicePointer
	ScalePointer    nativeDevicePointer
	BiasPointer     nativeDevicePointer
	OutputPointer   nativeDevicePointer
	SuppressPointer nativeDevicePointer
	Rows            int
	Cols            int
	GroupSize       int
	Bits            int
	SuppressCount   int
	InputBytes      uint64
	WeightBytes     uint64
	ScaleBytes      uint64
	BiasBytes       uint64
	OutputBytes     uint64
}

type hipPackedTopKLaunchArgs struct {
	InputPointer  nativeDevicePointer
	OutputPointer nativeDevicePointer
	InputCount    int
	OutputCount   int
	TopK          int
	ChunkSize     int
	InputBytes    uint64
	OutputBytes   uint64
}

type hipPackedTopKSampleLaunchArgs struct {
	InputPointer  nativeDevicePointer
	OutputPointer nativeDevicePointer
	InputCount    int
	TopK          int
	InputBytes    uint64
	OutputBytes   uint64
	Temperature   float32
	TopP          float32
	Draw          float64
	Softcap       float32
}

type hipOrderedEmbeddingCandidatesLaunchArgs struct {
	TopKPointer               nativeDevicePointer
	TokenOrderingPointer      nativeDevicePointer
	OutputPointer             nativeDevicePointer
	SuppressPointer           nativeDevicePointer
	TopKCount                 int
	NumCentroids              int
	TokensPerCentroid         int
	TokenOrderingElementBytes int
	TokenOrderingCount        int
	OutputCount               int
	SuppressCount             int
	TopKBytes                 uint64
	TokenOrderingBytes        uint64
	OutputBytes               uint64
}

type hipMLXQ4ProjectionBatchLaunchArgs struct {
	InputPointer  nativeDevicePointer
	WeightPointer nativeDevicePointer
	ScalePointer  nativeDevicePointer
	BiasPointer   nativeDevicePointer
	OutputPointer nativeDevicePointer
	Rows          int
	Cols          int
	Batch         int
	GroupSize     int
	Bits          int
	InputBytes    uint64
	WeightBytes   uint64
	ScaleBytes    uint64
	BiasBytes     uint64
	OutputBytes   uint64
}

type hipMLXQ4ProjectionGreedyBatchLaunchArgs struct {
	InputPointer    nativeDevicePointer
	WeightPointer   nativeDevicePointer
	ScalePointer    nativeDevicePointer
	BiasPointer     nativeDevicePointer
	OutputPointer   nativeDevicePointer
	SuppressPointer nativeDevicePointer
	Rows            int
	Cols            int
	Batch           int
	GroupSize       int
	Bits            int
	SuppressCount   int
	InputBytes      uint64
	WeightBytes     uint64
	ScaleBytes      uint64
	BiasBytes       uint64
	OutputBytes     uint64
}

type hipMLXQ4GELUTanhMulLaunchArgs struct {
	InputPointer      nativeDevicePointer
	GateWeightPointer nativeDevicePointer
	GateScalePointer  nativeDevicePointer
	GateBiasPointer   nativeDevicePointer
	UpWeightPointer   nativeDevicePointer
	UpScalePointer    nativeDevicePointer
	UpBiasPointer     nativeDevicePointer
	OutputPointer     nativeDevicePointer
	Rows              int
	Cols              int
	GroupSize         int
	Bits              int
	InputBytes        uint64
	GateWeightBytes   uint64
	GateScaleBytes    uint64
	GateBiasBytes     uint64
	UpWeightBytes     uint64
	UpScaleBytes      uint64
	UpBiasBytes       uint64
	OutputBytes       uint64
}

type hipMLXQ4GELUTanhMulBatchLaunchArgs struct {
	InputPointer      nativeDevicePointer
	GateWeightPointer nativeDevicePointer
	GateScalePointer  nativeDevicePointer
	GateBiasPointer   nativeDevicePointer
	UpWeightPointer   nativeDevicePointer
	UpScalePointer    nativeDevicePointer
	UpBiasPointer     nativeDevicePointer
	OutputPointer     nativeDevicePointer
	Rows              int
	Cols              int
	GroupSize         int
	Bits              int
	InputBytes        uint64
	GateWeightBytes   uint64
	GateScaleBytes    uint64
	GateBiasBytes     uint64
	UpWeightBytes     uint64
	UpScaleBytes      uint64
	UpBiasBytes       uint64
	OutputBytes       uint64
	Batch             int
}

type hipMLXQ4GELUTanhMLPPersistentLaunchArgs struct {
	InputPointer      nativeDevicePointer
	GateWeightPointer nativeDevicePointer
	GateScalePointer  nativeDevicePointer
	GateBiasPointer   nativeDevicePointer
	UpWeightPointer   nativeDevicePointer
	UpScalePointer    nativeDevicePointer
	UpBiasPointer     nativeDevicePointer
	DownWeightPointer nativeDevicePointer
	DownScalePointer  nativeDevicePointer
	DownBiasPointer   nativeDevicePointer
	ActivationPointer nativeDevicePointer
	OutputPointer     nativeDevicePointer
	BarrierPointer    nativeDevicePointer
	Rows              int
	Cols              int
	DownRows          int
	GroupSize         int
	Bits              int
	InputBytes        uint64
	GateWeightBytes   uint64
	GateScaleBytes    uint64
	GateBiasBytes     uint64
	UpWeightBytes     uint64
	UpScaleBytes      uint64
	UpBiasBytes       uint64
	DownWeightBytes   uint64
	DownScaleBytes    uint64
	DownBiasBytes     uint64
	ActivationBytes   uint64
	OutputBytes       uint64
	BarrierBytes      uint64
}

type hipMLXQ4TripleProjLaunchArgs struct {
	InputPointer        nativeDevicePointer
	OutputPointer       nativeDevicePointer
	FirstWeightPointer  nativeDevicePointer
	FirstScalePointer   nativeDevicePointer
	FirstBiasPointer    nativeDevicePointer
	SecondWeightPointer nativeDevicePointer
	SecondScalePointer  nativeDevicePointer
	SecondBiasPointer   nativeDevicePointer
	ThirdWeightPointer  nativeDevicePointer
	ThirdScalePointer   nativeDevicePointer
	ThirdBiasPointer    nativeDevicePointer
	FirstRows           int
	SecondRows          int
	ThirdRows           int
	Cols                int
	GroupSize           int
	Bits                int
	InputBytes          uint64
	OutputBytes         uint64
	FirstWeightBytes    uint64
	FirstScaleBytes     uint64
	FirstBiasBytes      uint64
	SecondWeightBytes   uint64
	SecondScaleBytes    uint64
	SecondBiasBytes     uint64
	ThirdWeightBytes    uint64
	ThirdScaleBytes     uint64
	ThirdBiasBytes      uint64
}

type hipMLXQ4GELUTanhProjLaunchArgs struct {
	InputPointer      nativeDevicePointer
	WeightPointer     nativeDevicePointer
	ScalePointer      nativeDevicePointer
	BiasPointer       nativeDevicePointer
	MultiplierPointer nativeDevicePointer
	OutputPointer     nativeDevicePointer
	Rows              int
	Cols              int
	GroupSize         int
	Bits              int
	InputBytes        uint64
	WeightBytes       uint64
	ScaleBytes        uint64
	BiasBytes         uint64
	MultiplierBytes   uint64
	OutputBytes       uint64
}

type hipMLXQ4GELUTanhProjBatchLaunchArgs struct {
	InputPointer      nativeDevicePointer
	WeightPointer     nativeDevicePointer
	ScalePointer      nativeDevicePointer
	BiasPointer       nativeDevicePointer
	MultiplierPointer nativeDevicePointer
	OutputPointer     nativeDevicePointer
	Rows              int
	Cols              int
	Batch             int
	GroupSize         int
	Bits              int
	InputBytes        uint64
	WeightBytes       uint64
	ScaleBytes        uint64
	BiasBytes         uint64
	MultiplierBytes   uint64
	OutputBytes       uint64
}

type hipRMSResidualAddGELUTanhProjLaunchArgs struct {
	InputPointer            nativeDevicePointer
	RMSWeightPointer        nativeDevicePointer
	ResidualPointer         nativeDevicePointer
	WeightPointer           nativeDevicePointer
	ScalePointer            nativeDevicePointer
	BiasPointer             nativeDevicePointer
	MultiplierPointer       nativeDevicePointer
	ResidualOutputPointer   nativeDevicePointer
	ActivationOutputPointer nativeDevicePointer
	Rows                    int
	Cols                    int
	GroupSize               int
	Bits                    int
	InputBytes              uint64
	RMSWeightBytes          uint64
	ResidualBytes           uint64
	WeightBytes             uint64
	ScaleBytes              uint64
	BiasBytes               uint64
	MultiplierBytes         uint64
	ResidualOutputBytes     uint64
	ActivationOutputBytes   uint64
	Epsilon                 float32
	RMSWeightEncoding       uint32
	RMSFlags                uint32
	OutputScale             float32
}

func (req hipProjectionRequest) projectionDeviceBuffers(driver nativeHIPDriver) (*hipProjectionDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	inputPayload, err := hipFloat32Payload(req.Input)
	if err != nil {
		return nil, core.E("rocm.hip.ProjectionLaunch", "encode input", err)
	}
	input, err := hipUploadByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection input", inputPayload, len(req.Input))
	if err != nil {
		return nil, err
	}
	buffers := &hipProjectionDeviceBuffers{Input: input, Rows: req.Rows, Cols: req.Cols}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	switch {
	case len(req.F32) > 0:
		weightsPayload, err := hipFloat32Payload(req.F32)
		if err != nil {
			return nil, core.E("rocm.hip.ProjectionLaunch", "encode f32 weights", err)
		}
		weights, err := hipUploadByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection f32 weights", weightsPayload, len(req.F32))
		if err != nil {
			return nil, err
		}
		buffers.Weights = weights
		buffers.Encoding = hipProjectionWeightEncodingF32
	case len(req.FP16) > 0:
		weightsPayload, err := hipUint16Payload(req.FP16)
		if err != nil {
			return nil, core.E("rocm.hip.ProjectionLaunch", "encode fp16 weights", err)
		}
		weights, err := hipUploadByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection fp16 weights", weightsPayload, len(req.FP16))
		if err != nil {
			return nil, err
		}
		buffers.Weights = weights
		buffers.Encoding = hipProjectionWeightEncodingFP16
	case len(req.BF16) > 0:
		weightsPayload, err := hipUint16Payload(req.BF16)
		if err != nil {
			return nil, core.E("rocm.hip.ProjectionLaunch", "encode bf16 weights", err)
		}
		weights, err := hipUploadByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection bf16 weights", weightsPayload, len(req.BF16))
		if err != nil {
			return nil, err
		}
		buffers.Weights = weights
		buffers.Encoding = hipProjectionWeightEncodingBF16
	case len(req.Q8) > 0:
		weightsPayload := hipInt8Payload(req.Q8)
		weights, err := hipUploadByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection q8 weights", weightsPayload, len(req.Q8))
		if err != nil {
			return nil, err
		}
		buffers.Weights = weights
		buffers.Encoding = hipProjectionWeightEncodingQ8
		buffers.Q8Scale = req.Q8Scale
	default:
		return nil, core.E("rocm.hip.ProjectionLaunch", "projection weights are required", nil)
	}

	if len(req.Bias) > 0 {
		biasPayload, err := hipFloat32Payload(req.Bias)
		if err != nil {
			return nil, core.E("rocm.hip.ProjectionLaunch", "encode bias", err)
		}
		bias, err := hipUploadByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection bias", biasPayload, len(req.Bias))
		if err != nil {
			return nil, err
		}
		buffers.Bias = bias
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection output", uint64(req.Rows*4), req.Rows)
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipMLXQ4ProjectionRequest) validate() error {
	return validateHIPMLXAffineProjectionShape(len(req.Input), len(req.Weight), len(req.Scales), len(req.Biases), req.Rows, req.Cols, req.GroupSize, req.Bits)
}

func (cfg hipMLXQ4DeviceWeightConfig) quantBits() int {
	return hipMLXQ4ProjectionBitsOrDefault(cfg.Bits)
}

func (cfg hipMLXQ4DeviceWeightConfig) denseWeightEncoding() (uint32, bool) {
	switch cfg.WeightEncoding {
	case hipProjectionWeightEncodingBF16:
		return cfg.WeightEncoding, true
	default:
		return 0, false
	}
}

func (cfg hipMLXQ4DeviceWeightConfig) hasCompleteWeightStorage() bool {
	if _, dense := cfg.denseWeightEncoding(); dense {
		return cfg.WeightPointer != 0
	}
	return cfg.WeightPointer != 0 && cfg.ScalePointer != 0 && cfg.BiasPointer != 0
}

func (cfg hipMLXQ4DeviceWeightConfig) validateDenseInputCount(inputCount int) error {
	encoding, ok := cfg.denseWeightEncoding()
	if !ok {
		return core.E("rocm.hip.ProjectionLaunch", core.Sprintf("unsupported dense projection weight encoding %d", cfg.WeightEncoding), nil)
	}
	if inputCount != cfg.Cols {
		return core.E("rocm.hip.ProjectionLaunch", "dense projection input count must match cols", nil)
	}
	return (hipBF16DeviceWeightConfig{
		WeightPointer: cfg.WeightPointer,
		WeightBytes:   cfg.WeightBytes,
		Rows:          cfg.Rows,
		Cols:          cfg.Cols,
	}).validate(encoding)
}

func (cfg hipMLXQ4DeviceWeightConfig) validate(input []float32) error {
	return cfg.validateInputCount(len(input))
}

func (cfg hipMLXQ4DeviceWeightConfig) validateInputCount(inputCount int) error {
	if cfg.WeightEncoding != 0 {
		return cfg.validateDenseInputCount(inputCount)
	}
	if cfg.WeightPointer == 0 || cfg.ScalePointer == 0 || cfg.BiasPointer == 0 {
		return core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device weight, scale, and bias pointers are required", nil)
	}
	if cfg.WeightBytes == 0 || cfg.ScaleBytes == 0 || cfg.BiasBytes == 0 {
		return core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device weight, scale, and bias byte counts are required", nil)
	}
	if cfg.WeightBytes%4 != 0 || cfg.ScaleBytes%2 != 0 || cfg.BiasBytes%2 != 0 {
		return core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device byte counts must be element-aligned", nil)
	}
	if cfg.WeightBytes/4 > uint64(int(^uint(0)>>1)) ||
		cfg.ScaleBytes/2 > uint64(int(^uint(0)>>1)) ||
		cfg.BiasBytes/2 > uint64(int(^uint(0)>>1)) {
		return core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device element counts are out of int range", nil)
	}
	return validateHIPMLXAffineProjectionShape(inputCount, int(cfg.WeightBytes/4), int(cfg.ScaleBytes/2), int(cfg.BiasBytes/2), cfg.Rows, cfg.Cols, cfg.GroupSize, cfg.quantBits())
}

func (cfg hipMLXQ4DeviceWeightConfig) validateBatchInputCount(inputCount int, batch int) error {
	if batch <= 0 {
		return core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch size must be positive", nil)
	}
	if inputCount != cfg.Cols*batch {
		return core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch input count mismatch", nil)
	}
	if cfg.WeightEncoding != 0 {
		return cfg.validateDenseInputCount(cfg.Cols)
	}
	if cfg.WeightPointer == 0 || cfg.ScalePointer == 0 || cfg.BiasPointer == 0 {
		return core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection device weight, scale, and bias pointers are required", nil)
	}
	if cfg.WeightBytes == 0 || cfg.ScaleBytes == 0 || cfg.BiasBytes == 0 {
		return core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection device weight, scale, and bias byte counts are required", nil)
	}
	if cfg.WeightBytes%4 != 0 || cfg.ScaleBytes%2 != 0 || cfg.BiasBytes%2 != 0 {
		return core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection device byte counts must be element-aligned", nil)
	}
	if cfg.WeightBytes/4 > uint64(int(^uint(0)>>1)) ||
		cfg.ScaleBytes/2 > uint64(int(^uint(0)>>1)) ||
		cfg.BiasBytes/2 > uint64(int(^uint(0)>>1)) {
		return core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection device element counts are out of int range", nil)
	}
	return validateHIPMLXAffineProjectionShape(cfg.Cols, int(cfg.WeightBytes/4), int(cfg.ScaleBytes/2), int(cfg.BiasBytes/2), cfg.Rows, cfg.Cols, cfg.GroupSize, cfg.quantBits())
}

func (req hipMLXQ4ProjectionRequest) deviceBuffers(driver nativeHIPDriver) (*hipMLXQ4ProjectionDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	inputPayload, err := hipFloat32Payload(req.Input)
	if err != nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "encode input", err)
	}
	input, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection input", inputPayload, len(req.Input))
	if err != nil {
		return nil, err
	}
	buffers := &hipMLXQ4ProjectionDeviceBuffers{Input: input, Rows: req.Rows, Cols: req.Cols, GroupSize: req.GroupSize, Bits: hipMLXQ4ProjectionBitsOrDefault(req.Bits)}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	weightPayload, err := hipUint32Payload(req.Weight)
	if err != nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "encode packed weights", err)
	}
	weights, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection packed weights", weightPayload, len(req.Weight))
	if err != nil {
		return nil, err
	}
	buffers.Weight = weights

	scalePayload, err := hipUint16Payload(req.Scales)
	if err != nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "encode scales", err)
	}
	scales, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection scales", scalePayload, len(req.Scales))
	if err != nil {
		return nil, err
	}
	buffers.Scales = scales

	biasPayload, err := hipUint16Payload(req.Biases)
	if err != nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "encode biases", err)
	}
	biases, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection biases", biasPayload, len(req.Biases))
	if err != nil {
		return nil, err
	}
	buffers.Biases = biases

	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection output", uint64(req.Rows*4), req.Rows)
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipMLXQ4ProjectionRequest) launchArgs(buffers *hipMLXQ4ProjectionDeviceBuffers) (hipMLXQ4ProjectionLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipMLXQ4ProjectionLaunchArgs{}, err
	}
	if buffers == nil || buffers.Input == nil || buffers.Weight == nil || buffers.Scales == nil || buffers.Biases == nil || buffers.Output == nil {
		return hipMLXQ4ProjectionLaunchArgs{}, core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device buffers are required", nil)
	}
	bits := hipMLXQ4ProjectionBitsOrDefault(req.Bits)
	packedPerRow, err := hipMLXAffinePackedCols(req.Cols, bits)
	if err != nil {
		return hipMLXQ4ProjectionLaunchArgs{}, err
	}
	groupsPerRow := req.Cols / req.GroupSize
	if buffers.Input.Count() != req.Cols ||
		buffers.Weight.Count() != req.Rows*packedPerRow ||
		buffers.Scales.Count() != req.Rows*groupsPerRow ||
		buffers.Biases.Count() != req.Rows*groupsPerRow ||
		buffers.Output.Count() != req.Rows ||
		buffers.Rows != req.Rows ||
		buffers.Cols != req.Cols ||
		buffers.GroupSize != req.GroupSize ||
		buffers.Bits != bits {
		return hipMLXQ4ProjectionLaunchArgs{}, core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device buffer shape mismatch", nil)
	}
	return hipMLXQ4ProjectionLaunchArgs{
		InputPointer:  buffers.Input.Pointer(),
		WeightPointer: buffers.Weight.Pointer(),
		ScalePointer:  buffers.Scales.Pointer(),
		BiasPointer:   buffers.Biases.Pointer(),
		OutputPointer: buffers.Output.Pointer(),
		Rows:          req.Rows,
		Cols:          req.Cols,
		GroupSize:     req.GroupSize,
		Bits:          bits,
		InputBytes:    buffers.Input.SizeBytes(),
		WeightBytes:   buffers.Weight.SizeBytes(),
		ScaleBytes:    buffers.Scales.SizeBytes(),
		BiasBytes:     buffers.Biases.SizeBytes(),
		OutputBytes:   buffers.Output.SizeBytes(),
	}, nil
}

func (req hipProjectionRequest) projectionLaunchArgs(buffers *hipProjectionDeviceBuffers) (hipProjectionLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipProjectionLaunchArgs{}, err
	}
	if buffers == nil || buffers.Input == nil || buffers.Weights == nil || buffers.Output == nil {
		return hipProjectionLaunchArgs{}, core.E("rocm.hip.ProjectionLaunch", "projection device buffers are required", nil)
	}
	if buffers.Input.Count() != req.Cols || buffers.Weights.Count() != req.Rows*req.Cols || buffers.Output.Count() != req.Rows {
		return hipProjectionLaunchArgs{}, core.E("rocm.hip.ProjectionLaunch", "projection device buffer shape mismatch", nil)
	}
	var biasPointer nativeDevicePointer
	var biasBytes uint64
	var flags uint32
	if len(req.Bias) > 0 {
		if buffers.Bias == nil || buffers.Bias.Count() != req.Rows {
			return hipProjectionLaunchArgs{}, core.E("rocm.hip.ProjectionLaunch", "projection bias buffer shape mismatch", nil)
		}
		biasPointer = buffers.Bias.Pointer()
		biasBytes = buffers.Bias.SizeBytes()
		flags |= hipProjectionLaunchFlagBias
	}
	encoding, err := hipProjectionWeightEncodingCode(req)
	if err != nil {
		return hipProjectionLaunchArgs{}, err
	}
	if buffers.Encoding != encoding {
		return hipProjectionLaunchArgs{}, core.E("rocm.hip.ProjectionLaunch", "projection weight encoding mismatch", nil)
	}
	return hipProjectionLaunchArgs{
		InputPointer:   buffers.Input.Pointer(),
		InputCount:     buffers.Input.Count(),
		InputBytes:     buffers.Input.SizeBytes(),
		WeightPointer:  buffers.Weights.Pointer(),
		WeightBytes:    buffers.Weights.SizeBytes(),
		BiasPointer:    biasPointer,
		BiasBytes:      biasBytes,
		OutputPointer:  buffers.Output.Pointer(),
		OutputBytes:    buffers.Output.SizeBytes(),
		Rows:           req.Rows,
		Cols:           req.Cols,
		WeightEncoding: encoding,
		Flags:          flags,
		Q8Scale:        req.Q8Scale,
	}, nil
}

func (args hipProjectionLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipProjectionLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.WeightPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "input, weight, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	inputCount, err := rocmDeviceKVPositiveUint32("input count", args.InputCount)
	if err != nil {
		return nil, err
	}
	if inputCount != cols {
		return nil, core.E("rocm.hip.ProjectionLaunch", "input count must match cols", nil)
	}
	if args.InputBytes != uint64(cols)*4 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "input byte count mismatch", nil)
	}
	if args.InputBytes > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip.ProjectionLaunch", "input bytes are out of uint32 range", nil)
	}
	if args.OutputBytes != uint64(rows)*4 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "output byte count mismatch", nil)
	}
	switch args.WeightEncoding {
	case hipProjectionWeightEncodingFP16, hipProjectionWeightEncodingBF16:
		if args.WeightBytes != uint64(rows)*uint64(cols)*2 {
			return nil, core.E("rocm.hip.ProjectionLaunch", "fp16/bf16 weight byte count mismatch", nil)
		}
	case hipProjectionWeightEncodingQ8:
		if args.WeightBytes != uint64(rows)*uint64(cols) {
			return nil, core.E("rocm.hip.ProjectionLaunch", "q8 weight byte count mismatch", nil)
		}
		if !hipQ8ScaleIsPositiveFinite(args.Q8Scale) {
			return nil, core.E("rocm.hip.ProjectionLaunch", "q8 scale must be positive and finite", nil)
		}
	case hipProjectionWeightEncodingF32:
		if args.WeightBytes != uint64(rows)*uint64(cols)*4 {
			return nil, core.E("rocm.hip.ProjectionLaunch", "f32 weight byte count mismatch", nil)
		}
	default:
		return nil, core.E("rocm.hip.ProjectionLaunch", core.Sprintf("unsupported projection weight encoding %d", args.WeightEncoding), nil)
	}
	if args.Flags&hipProjectionLaunchFlagBias != 0 {
		if args.BiasPointer == 0 {
			return nil, core.E("rocm.hip.ProjectionLaunch", "bias pointer is nil", nil)
		}
		if args.BiasBytes != uint64(rows)*4 {
			return nil, core.E("rocm.hip.ProjectionLaunch", "bias byte count mismatch", nil)
		}
	} else if args.BiasPointer != 0 || args.BiasBytes != 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "bias metadata supplied without bias flag", nil)
	}
	if cap(payload) < hipProjectionLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipProjectionLaunchArgsBytes)
	} else {
		payload = payload[:hipProjectionLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipProjectionLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint32(payload[16:], inputCount)
	binary.LittleEndian.PutUint32(payload[20:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[32:], args.WeightBytes)
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[48:], args.BiasBytes)
	binary.LittleEndian.PutUint64(payload[56:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint64(payload[64:], args.OutputBytes)
	binary.LittleEndian.PutUint32(payload[72:], rows)
	binary.LittleEndian.PutUint32(payload[76:], cols)
	binary.LittleEndian.PutUint32(payload[80:], args.WeightEncoding)
	binary.LittleEndian.PutUint32(payload[84:], args.Flags)
	binary.LittleEndian.PutUint32(payload[88:], math.Float32bits(args.Q8Scale))
	return payload, nil
}

func (args hipProjectionBatchLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipProjectionBatchLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.WeightPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", "input, weight, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	batch, err := rocmDeviceKVPositiveUint32("batch", args.Batch)
	if err != nil {
		return nil, err
	}
	if args.InputBytes != uint64(cols)*uint64(batch)*4 {
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", "input byte count mismatch", nil)
	}
	if args.OutputBytes != uint64(rows)*uint64(batch)*4 {
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", "output byte count mismatch", nil)
	}
	switch args.WeightEncoding {
	case hipProjectionWeightEncodingFP16, hipProjectionWeightEncodingBF16:
		if args.WeightBytes != uint64(rows)*uint64(cols)*2 {
			return nil, core.E("rocm.hip.ProjectionBatchLaunch", "fp16/bf16 weight byte count mismatch", nil)
		}
	case hipProjectionWeightEncodingQ8:
		if args.WeightBytes != uint64(rows)*uint64(cols) {
			return nil, core.E("rocm.hip.ProjectionBatchLaunch", "q8 weight byte count mismatch", nil)
		}
		if !hipQ8ScaleIsPositiveFinite(args.Q8Scale) {
			return nil, core.E("rocm.hip.ProjectionBatchLaunch", "q8 scale must be positive and finite", nil)
		}
	case hipProjectionWeightEncodingF32:
		if args.WeightBytes != uint64(rows)*uint64(cols)*4 {
			return nil, core.E("rocm.hip.ProjectionBatchLaunch", "f32 weight byte count mismatch", nil)
		}
	default:
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", core.Sprintf("unsupported projection weight encoding %d", args.WeightEncoding), nil)
	}
	if args.Flags&hipProjectionLaunchFlagBias != 0 {
		if args.BiasPointer == 0 {
			return nil, core.E("rocm.hip.ProjectionBatchLaunch", "bias pointer is nil", nil)
		}
		if args.BiasBytes != uint64(rows)*4 {
			return nil, core.E("rocm.hip.ProjectionBatchLaunch", "bias byte count mismatch", nil)
		}
	} else if args.BiasPointer != 0 || args.BiasBytes != 0 {
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", "bias metadata supplied without bias flag", nil)
	}
	if cap(payload) < hipProjectionBatchLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipProjectionBatchLaunchArgsBytes)
	} else {
		payload = payload[:hipProjectionBatchLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipProjectionBatchLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], args.WeightBytes)
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], args.BiasBytes)
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint64(payload[56:], args.OutputBytes)
	binary.LittleEndian.PutUint32(payload[64:], rows)
	binary.LittleEndian.PutUint32(payload[68:], cols)
	binary.LittleEndian.PutUint32(payload[72:], batch)
	binary.LittleEndian.PutUint32(payload[76:], args.WeightEncoding)
	binary.LittleEndian.PutUint32(payload[80:], args.Flags)
	binary.LittleEndian.PutUint32(payload[84:], math.Float32bits(args.Q8Scale))
	binary.LittleEndian.PutUint64(payload[88:], args.InputBytes)
	return payload, nil
}

func (args hipMLXQ4ProjectionLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipMLXQ4ProjectionLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	return args.binaryInto(hipMLXQ4ProjectionOutputFull, payload)
}

func (args hipMLXQ4ProjectionLaunchArgs) GreedyBinary() ([]byte, error) {
	return args.GreedyBinaryInto(nil)
}

func (args hipMLXQ4ProjectionLaunchArgs) GreedyBinaryInto(payload []byte) ([]byte, error) {
	return args.binaryInto(hipMLXQ4ProjectionOutputBest, payload)
}

func (args hipMLXQ4ProjectionLaunchArgs) ScoresBinary() ([]byte, error) {
	return args.ScoresBinaryInto(nil)
}

func (args hipMLXQ4ProjectionLaunchArgs) ScoresBinaryInto(payload []byte) ([]byte, error) {
	return args.binaryInto(hipMLXQ4ProjectionOutputScores, payload)
}

func (args hipPackedTopKLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipPackedTopKLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.PackedTopKLaunch", "input and output pointers are required", nil)
	}
	inputCount, err := rocmDeviceKVPositiveUint32("packed top-k input count", args.InputCount)
	if err != nil {
		return nil, err
	}
	outputCount, err := rocmDeviceKVPositiveUint32("packed top-k output count", args.OutputCount)
	if err != nil {
		return nil, err
	}
	topK, err := rocmDeviceKVPositiveUint32("packed top-k", args.TopK)
	if err != nil {
		return nil, err
	}
	if topK > hipPackedTopKMaxK {
		return nil, core.E("rocm.hip.PackedTopKLaunch", "top-k exceeds kernel maximum", nil)
	}
	chunkSize, err := rocmDeviceKVPositiveUint32("packed top-k chunk size", args.ChunkSize)
	if err != nil {
		return nil, err
	}
	if args.ChunkSize != hipPackedTopKChunkSize {
		return nil, core.E("rocm.hip.PackedTopKLaunch", "chunk size mismatch", nil)
	}
	chunkCount := (args.InputCount + args.ChunkSize - 1) / args.ChunkSize
	if args.OutputCount != chunkCount*args.TopK {
		return nil, core.E("rocm.hip.PackedTopKLaunch", "output count mismatch", nil)
	}
	if args.InputBytes != uint64(args.InputCount*hipMLXQ4ProjectionBestBytes) {
		return nil, core.E("rocm.hip.PackedTopKLaunch", "input byte count mismatch", nil)
	}
	if args.OutputBytes != uint64(args.OutputCount*hipMLXQ4ProjectionBestBytes) {
		return nil, core.E("rocm.hip.PackedTopKLaunch", "output byte count mismatch", nil)
	}
	if err := hipProjectionUint32Bytes("rocm.hip.PackedTopKLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.PackedTopKLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	if cap(payload) < hipPackedTopKLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipPackedTopKLaunchArgsBytes)
	} else {
		payload = payload[:hipPackedTopKLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipPackedTopKLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[24:], inputCount)
	binary.LittleEndian.PutUint32(payload[28:], outputCount)
	binary.LittleEndian.PutUint32(payload[32:], topK)
	binary.LittleEndian.PutUint32(payload[36:], chunkSize)
	binary.LittleEndian.PutUint32(payload[40:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[44:], uint32(args.OutputBytes))
	return payload, nil
}

func (args hipPackedTopKSampleLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipOrderedEmbeddingCandidatesLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipOrderedEmbeddingCandidatesLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.TopKPointer == 0 || args.TokenOrderingPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "top-k, token ordering, and output pointers are required", nil)
	}
	topK, err := rocmDeviceKVPositiveUint32("ordered embedding top-k count", args.TopKCount)
	if err != nil {
		return nil, err
	}
	if args.TopKCount > hipPackedTopKMaxK {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "top-k exceeds kernel maximum", nil)
	}
	centroids, err := rocmDeviceKVPositiveUint32("ordered embedding centroids", args.NumCentroids)
	if err != nil {
		return nil, err
	}
	tokensPerCentroid, err := rocmDeviceKVPositiveUint32("ordered embedding tokens per centroid", args.TokensPerCentroid)
	if err != nil {
		return nil, err
	}
	orderingCount, err := rocmDeviceKVPositiveUint32("ordered embedding token-ordering count", args.TokenOrderingCount)
	if err != nil {
		return nil, err
	}
	outputCount, err := rocmDeviceKVPositiveUint32("ordered embedding output count", args.OutputCount)
	if err != nil {
		return nil, err
	}
	suppressCount := uint32(0)
	if args.SuppressCount > 0 {
		if args.SuppressPointer == 0 {
			return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "suppress pointer is required when suppress count is set", nil)
		}
		suppressCount, err = rocmDeviceKVPositiveUint32("ordered embedding suppress count", args.SuppressCount)
		if err != nil {
			return nil, err
		}
	}
	elementBytes := uint32(args.TokenOrderingElementBytes)
	if args.TokenOrderingElementBytes != 4 && args.TokenOrderingElementBytes != 8 {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "token-ordering element bytes must be 4 or 8", nil)
	}
	if args.TokenOrderingCount != args.NumCentroids*args.TokensPerCentroid {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "token-ordering count mismatch", nil)
	}
	if args.OutputCount != args.TopKCount*args.TokensPerCentroid {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "output count mismatch", nil)
	}
	if args.TopKBytes != uint64(args.TopKCount*hipMLXQ4ProjectionBestBytes) {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "top-k byte count mismatch", nil)
	}
	if args.TokenOrderingBytes != uint64(args.TokenOrderingCount*args.TokenOrderingElementBytes) {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "token-ordering byte count mismatch", nil)
	}
	if args.OutputBytes != uint64(args.OutputCount*4) {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "output byte count mismatch", nil)
	}
	if err := hipProjectionUint32Bytes("rocm.hip.OrderedEmbeddingCandidatesLaunch", "top-k bytes", args.TopKBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.OrderedEmbeddingCandidatesLaunch", "token-ordering bytes", args.TokenOrderingBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.OrderedEmbeddingCandidatesLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	if cap(payload) < hipOrderedEmbeddingCandidatesLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipOrderedEmbeddingCandidatesLaunchArgsBytes)
	} else {
		payload = payload[:hipOrderedEmbeddingCandidatesLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipOrderedEmbeddingCandidatesLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.TopKPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.TokenOrderingPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.SuppressPointer))
	binary.LittleEndian.PutUint32(payload[40:], topK)
	binary.LittleEndian.PutUint32(payload[44:], centroids)
	binary.LittleEndian.PutUint32(payload[48:], tokensPerCentroid)
	binary.LittleEndian.PutUint32(payload[52:], elementBytes)
	binary.LittleEndian.PutUint32(payload[56:], orderingCount)
	binary.LittleEndian.PutUint32(payload[60:], outputCount)
	binary.LittleEndian.PutUint32(payload[64:], suppressCount)
	binary.LittleEndian.PutUint32(payload[68:], uint32(args.TopKBytes))
	binary.LittleEndian.PutUint32(payload[72:], uint32(args.TokenOrderingBytes))
	binary.LittleEndian.PutUint32(payload[76:], uint32(args.OutputBytes))
	return payload, nil
}

func (args hipPackedTopKSampleLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.PackedTopKSampleLaunch", "input and output pointers are required", nil)
	}
	inputCount, err := rocmDeviceKVPositiveUint32("packed top-k sample input count", args.InputCount)
	if err != nil {
		return nil, err
	}
	topK, err := rocmDeviceKVPositiveUint32("packed top-k sample top-k", args.TopK)
	if err != nil {
		return nil, err
	}
	if args.TopK > hipPackedTopKMaxK || args.TopK > args.InputCount {
		return nil, core.E("rocm.hip.PackedTopKSampleLaunch", "top-k exceeds input or kernel maximum", nil)
	}
	if args.InputBytes != uint64(args.InputCount*hipMLXQ4ProjectionBestBytes) {
		return nil, core.E("rocm.hip.PackedTopKSampleLaunch", "input byte count mismatch", nil)
	}
	if args.OutputBytes != hipMLXQ4ProjectionBestBytes {
		return nil, core.E("rocm.hip.PackedTopKSampleLaunch", "output byte count mismatch", nil)
	}
	if args.Temperature < 0 || math.IsNaN(float64(args.Temperature)) || math.IsInf(float64(args.Temperature), 0) {
		return nil, core.E("rocm.hip.PackedTopKSampleLaunch", "temperature must be non-negative and finite", nil)
	}
	if args.TopP < 0 || args.TopP > 1 || math.IsNaN(float64(args.TopP)) || math.IsInf(float64(args.TopP), 0) {
		return nil, core.E("rocm.hip.PackedTopKSampleLaunch", "top-p must be in [0, 1]", nil)
	}
	if math.IsNaN(args.Draw) || math.IsInf(args.Draw, 0) {
		return nil, core.E("rocm.hip.PackedTopKSampleLaunch", "draw must be finite", nil)
	}
	if args.Softcap < 0 || math.IsNaN(float64(args.Softcap)) || math.IsInf(float64(args.Softcap), 0) {
		return nil, core.E("rocm.hip.PackedTopKSampleLaunch", "softcap must be non-negative and finite", nil)
	}
	if err := hipProjectionUint32Bytes("rocm.hip.PackedTopKSampleLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if cap(payload) < hipPackedTopKSampleLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipPackedTopKSampleLaunchArgsBytes)
	} else {
		payload = payload[:hipPackedTopKSampleLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipPackedTopKSampleLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[24:], inputCount)
	binary.LittleEndian.PutUint32(payload[28:], topK)
	binary.LittleEndian.PutUint32(payload[32:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[36:], uint32(args.OutputBytes))
	binary.LittleEndian.PutUint32(payload[40:], math.Float32bits(args.Temperature))
	binary.LittleEndian.PutUint32(payload[44:], math.Float32bits(args.TopP))
	binary.LittleEndian.PutUint64(payload[48:], math.Float64bits(args.Draw))
	binary.LittleEndian.PutUint32(payload[56:], math.Float32bits(args.Softcap))
	return payload, nil
}

const (
	hipMLXQ4ProjectionOutputFull = iota
	hipMLXQ4ProjectionOutputBest
	hipMLXQ4ProjectionOutputScores
)

func hipMLXAffineLaunchPackedGroups(operation string, cols, groupSize, bits uint32) (uint64, uint64, error) {
	if !hipMLXAffineSupportedBits(int(bits)) {
		return 0, 0, core.E(operation, "only 4-, 6-, and 8-bit MLX affine projection is supported", nil)
	}
	if groupSize == 0 || cols%groupSize != 0 {
		return 0, 0, core.E(operation, "cols must be divisible by group size", nil)
	}
	totalBits := uint64(cols) * uint64(bits)
	if totalBits%32 != 0 {
		return 0, 0, core.E(operation, "cols*bits must be divisible by 32 for MLX affine packing", nil)
	}
	return totalBits / 32, uint64(cols / groupSize), nil
}

func (args hipMLXQ4ProjectionLaunchArgs) binaryInto(outputKind int, payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.WeightPointer == 0 || args.ScalePointer == 0 || args.BiasPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "input, weight, scale, bias, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	packedPerRow, groupsPerRow, err := hipMLXAffineLaunchPackedGroups("rocm.hip.MLXQ4ProjectionLaunch", cols, groupSize, bits)
	if err != nil {
		return nil, err
	}
	if args.InputBytes != uint64(cols)*4 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "input byte count mismatch", nil)
	}
	if args.WeightBytes != uint64(rows)*packedPerRow*4 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "packed weight byte count mismatch", nil)
	}
	if args.ScaleBytes != uint64(rows)*groupsPerRow*2 || args.BiasBytes != uint64(rows)*groupsPerRow*2 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "scale/bias byte count mismatch", nil)
	}
	wantOutputBytes := uint64(rows) * 4
	switch outputKind {
	case hipMLXQ4ProjectionOutputFull:
	case hipMLXQ4ProjectionOutputBest:
		wantOutputBytes = hipMLXQ4ProjectionBestBytes
	case hipMLXQ4ProjectionOutputScores:
		wantOutputBytes = uint64(rows) * hipMLXQ4ProjectionBestBytes
	default:
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "unsupported q4 projection output kind", nil)
	}
	if args.OutputBytes != wantOutputBytes {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "output byte count mismatch", nil)
	}
	suppressCount := uint32(0)
	if args.SuppressCount > 0 {
		if outputKind != hipMLXQ4ProjectionOutputBest && outputKind != hipMLXQ4ProjectionOutputScores {
			return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "suppress tokens require greedy or score output", nil)
		}
		if args.SuppressPointer == 0 {
			return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "suppress token pointer is required", nil)
		}
		value, err := rocmDeviceKVPositiveUint32("suppress token count", args.SuppressCount)
		if err != nil {
			return nil, err
		}
		suppressCount = value
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionLaunch", "weight bytes", args.WeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionLaunch", "scale bytes", args.ScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionLaunch", "bias bytes", args.BiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	if cap(payload) < hipMLXQ4ProjectionLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipMLXQ4ProjectionLaunchArgsBytes)
	} else {
		payload = payload[:hipMLXQ4ProjectionLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipMLXQ4ProjectionLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ScalePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[48:], rows)
	binary.LittleEndian.PutUint32(payload[52:], cols)
	binary.LittleEndian.PutUint32(payload[56:], groupSize)
	binary.LittleEndian.PutUint32(payload[60:], bits)
	binary.LittleEndian.PutUint32(payload[64:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[68:], uint32(args.WeightBytes))
	binary.LittleEndian.PutUint32(payload[72:], uint32(args.ScaleBytes))
	binary.LittleEndian.PutUint32(payload[76:], uint32(args.BiasBytes))
	binary.LittleEndian.PutUint32(payload[80:], uint32(args.OutputBytes))
	binary.LittleEndian.PutUint32(payload[84:], suppressCount)
	binary.LittleEndian.PutUint64(payload[88:], uint64(args.SuppressPointer))
	return payload, nil
}

func (args hipMLXQ4ProjectionBatchLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipMLXQ4ProjectionBatchLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.WeightPointer == 0 || args.ScalePointer == 0 || args.BiasPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "input, weight, scale, bias, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	batch, err := rocmDeviceKVPositiveUint32("batch", args.Batch)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	packedPerRow, groupsPerRow, err := hipMLXAffineLaunchPackedGroups("rocm.hip.MLXQ4ProjectionBatchLaunch", cols, groupSize, bits)
	if err != nil {
		return nil, err
	}
	if args.InputBytes != uint64(batch)*uint64(cols)*4 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "input byte count mismatch", nil)
	}
	if args.WeightBytes != uint64(rows)*packedPerRow*4 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "packed weight byte count mismatch", nil)
	}
	if args.ScaleBytes != uint64(rows)*groupsPerRow*2 || args.BiasBytes != uint64(rows)*groupsPerRow*2 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "scale/bias byte count mismatch", nil)
	}
	if args.OutputBytes != uint64(batch)*uint64(rows)*4 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "output byte count mismatch", nil)
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionBatchLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionBatchLaunch", "weight bytes", args.WeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionBatchLaunch", "scale bytes", args.ScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionBatchLaunch", "bias bytes", args.BiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionBatchLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	if cap(payload) < hipMLXQ4ProjectionBatchLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipMLXQ4ProjectionBatchLaunchArgsBytes)
	} else {
		payload = payload[:hipMLXQ4ProjectionBatchLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipMLXQ4ProjectionBatchLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ScalePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[48:], rows)
	binary.LittleEndian.PutUint32(payload[52:], cols)
	binary.LittleEndian.PutUint32(payload[56:], batch)
	binary.LittleEndian.PutUint32(payload[60:], groupSize)
	binary.LittleEndian.PutUint32(payload[64:], bits)
	binary.LittleEndian.PutUint32(payload[68:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[72:], uint32(args.WeightBytes))
	binary.LittleEndian.PutUint32(payload[76:], uint32(args.ScaleBytes))
	binary.LittleEndian.PutUint32(payload[80:], uint32(args.BiasBytes))
	binary.LittleEndian.PutUint32(payload[84:], uint32(args.OutputBytes))
	return payload, nil
}

func (args hipMLXQ4ProjectionGreedyBatchLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipMLXQ4ProjectionGreedyBatchLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.WeightPointer == 0 || args.ScalePointer == 0 || args.BiasPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "input, weight, scale, bias, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	batch, err := rocmDeviceKVPositiveUint32("batch", args.Batch)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	packedPerRow, groupsPerRow, err := hipMLXAffineLaunchPackedGroups("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", cols, groupSize, bits)
	if err != nil {
		return nil, err
	}
	if args.InputBytes != uint64(batch)*uint64(cols)*4 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "input byte count mismatch", nil)
	}
	if args.WeightBytes != uint64(rows)*packedPerRow*4 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "packed weight byte count mismatch", nil)
	}
	if args.ScaleBytes != uint64(rows)*groupsPerRow*2 || args.BiasBytes != uint64(rows)*groupsPerRow*2 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "scale/bias byte count mismatch", nil)
	}
	if args.OutputBytes != uint64(batch)*hipMLXQ4ProjectionBestBytes {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "output byte count mismatch", nil)
	}
	suppressCount := uint32(0)
	if args.SuppressCount > 0 {
		if args.SuppressPointer == 0 {
			return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "suppress token pointer is required", nil)
		}
		value, err := rocmDeviceKVPositiveUint32("suppress token count", args.SuppressCount)
		if err != nil {
			return nil, err
		}
		suppressCount = value
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "weight bytes", args.WeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "scale bytes", args.ScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "bias bytes", args.BiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	if cap(payload) < hipMLXQ4ProjectionGreedyBatchLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipMLXQ4ProjectionGreedyBatchLaunchArgsBytes)
	} else {
		payload = payload[:hipMLXQ4ProjectionGreedyBatchLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipMLXQ4ProjectionGreedyBatchLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ScalePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.SuppressPointer))
	binary.LittleEndian.PutUint32(payload[56:], rows)
	binary.LittleEndian.PutUint32(payload[60:], cols)
	binary.LittleEndian.PutUint32(payload[64:], batch)
	binary.LittleEndian.PutUint32(payload[68:], groupSize)
	binary.LittleEndian.PutUint32(payload[72:], bits)
	binary.LittleEndian.PutUint32(payload[76:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[80:], uint32(args.WeightBytes))
	binary.LittleEndian.PutUint32(payload[84:], uint32(args.ScaleBytes))
	binary.LittleEndian.PutUint32(payload[88:], uint32(args.BiasBytes))
	binary.LittleEndian.PutUint32(payload[92:], uint32(args.OutputBytes))
	binary.LittleEndian.PutUint32(payload[96:], suppressCount)
	return payload, nil
}

func (args hipMLXQ4TripleProjLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipMLXQ4TripleProjLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.OutputPointer == 0 ||
		args.FirstWeightPointer == 0 || args.FirstScalePointer == 0 || args.FirstBiasPointer == 0 ||
		args.SecondWeightPointer == 0 || args.SecondScalePointer == 0 || args.SecondBiasPointer == 0 {
		return nil, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "input, output, and q4 weight pointers are required", nil)
	}
	firstRows, err := rocmDeviceKVPositiveUint32("first rows", args.FirstRows)
	if err != nil {
		return nil, err
	}
	secondRows, err := rocmDeviceKVPositiveUint32("second rows", args.SecondRows)
	if err != nil {
		return nil, err
	}
	thirdRows, err := rocmDeviceKVUint32("third rows", args.ThirdRows)
	if err != nil {
		return nil, err
	}
	if thirdRows > 0 && (args.ThirdWeightPointer == 0 || args.ThirdScalePointer == 0 || args.ThirdBiasPointer == 0) {
		return nil, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "third q4 weight pointers are required when third rows are non-zero", nil)
	}
	if thirdRows == 0 && (args.ThirdWeightBytes != 0 || args.ThirdScaleBytes != 0 || args.ThirdBiasBytes != 0) {
		return nil, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "third q4 byte counts must be zero when third rows are zero", nil)
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	packedPerRow, groupsPerRow, err := hipMLXAffineLaunchPackedGroups("rocm.hip.MLXQ4TripleProjectionLaunch", cols, groupSize, bits)
	if err != nil {
		return nil, err
	}
	totalRows := uint64(firstRows) + uint64(secondRows) + uint64(thirdRows)
	if args.InputBytes != uint64(cols)*4 {
		return nil, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "input byte count mismatch", nil)
	}
	if args.OutputBytes != totalRows*4 {
		return nil, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "output byte count mismatch", nil)
	}
	checkPart := func(label string, rows uint32, weightBytes, scaleBytes, biasBytes uint64) error {
		if weightBytes != uint64(rows)*packedPerRow*4 {
			return core.E("rocm.hip.MLXQ4TripleProjectionLaunch", label+" packed weight byte count mismatch", nil)
		}
		wantScaleBiasBytes := uint64(rows) * groupsPerRow * 2
		if scaleBytes != wantScaleBiasBytes || biasBytes != wantScaleBiasBytes {
			return core.E("rocm.hip.MLXQ4TripleProjectionLaunch", label+" scale/bias byte count mismatch", nil)
		}
		return nil
	}
	if err := checkPart("first", firstRows, args.FirstWeightBytes, args.FirstScaleBytes, args.FirstBiasBytes); err != nil {
		return nil, err
	}
	if err := checkPart("second", secondRows, args.SecondWeightBytes, args.SecondScaleBytes, args.SecondBiasBytes); err != nil {
		return nil, err
	}
	if thirdRows > 0 {
		if err := checkPart("third", thirdRows, args.ThirdWeightBytes, args.ThirdScaleBytes, args.ThirdBiasBytes); err != nil {
			return nil, err
		}
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "first weight bytes", args.FirstWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "first scale bytes", args.FirstScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "first bias bytes", args.FirstBiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "second weight bytes", args.SecondWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "second scale bytes", args.SecondScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "second bias bytes", args.SecondBiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "third weight bytes", args.ThirdWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "third scale bytes", args.ThirdScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "third bias bytes", args.ThirdBiasBytes); err != nil {
		return nil, err
	}
	if cap(payload) < hipMLXQ4TripleProjLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipMLXQ4TripleProjLaunchArgsBytes)
	} else {
		payload = payload[:hipMLXQ4TripleProjLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipMLXQ4TripleProjLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.FirstWeightPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.FirstScalePointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.FirstBiasPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.SecondWeightPointer))
	binary.LittleEndian.PutUint64(payload[56:], uint64(args.SecondScalePointer))
	binary.LittleEndian.PutUint64(payload[64:], uint64(args.SecondBiasPointer))
	binary.LittleEndian.PutUint64(payload[72:], uint64(args.ThirdWeightPointer))
	binary.LittleEndian.PutUint64(payload[80:], uint64(args.ThirdScalePointer))
	binary.LittleEndian.PutUint64(payload[88:], uint64(args.ThirdBiasPointer))
	binary.LittleEndian.PutUint32(payload[96:], firstRows)
	binary.LittleEndian.PutUint32(payload[100:], secondRows)
	binary.LittleEndian.PutUint32(payload[104:], thirdRows)
	binary.LittleEndian.PutUint32(payload[108:], cols)
	binary.LittleEndian.PutUint32(payload[112:], groupSize)
	binary.LittleEndian.PutUint32(payload[116:], bits)
	binary.LittleEndian.PutUint32(payload[120:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[124:], uint32(args.OutputBytes))
	binary.LittleEndian.PutUint32(payload[128:], uint32(args.FirstWeightBytes))
	binary.LittleEndian.PutUint32(payload[132:], uint32(args.FirstScaleBytes))
	binary.LittleEndian.PutUint32(payload[136:], uint32(args.FirstBiasBytes))
	binary.LittleEndian.PutUint32(payload[140:], uint32(args.SecondWeightBytes))
	binary.LittleEndian.PutUint32(payload[144:], uint32(args.SecondScaleBytes))
	binary.LittleEndian.PutUint32(payload[148:], uint32(args.SecondBiasBytes))
	binary.LittleEndian.PutUint32(payload[152:], uint32(args.ThirdWeightBytes))
	binary.LittleEndian.PutUint32(payload[156:], uint32(args.ThirdScaleBytes))
	binary.LittleEndian.PutUint32(payload[160:], uint32(args.ThirdBiasBytes))
	return payload, nil
}

func (args hipMLXQ4GELUTanhMulLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipMLXQ4GELUTanhMulLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.GateWeightPointer == 0 || args.GateScalePointer == 0 ||
		args.GateBiasPointer == 0 || args.UpWeightPointer == 0 || args.UpScalePointer == 0 ||
		args.UpBiasPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "input, gate/up weights, scale/bias, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	packedPerRow, groupsPerRow, err := hipMLXAffineLaunchPackedGroups("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", cols, groupSize, bits)
	if err != nil {
		return nil, err
	}
	if args.InputBytes != uint64(cols)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "input byte count mismatch", nil)
	}
	wantWeightBytes := uint64(rows) * packedPerRow * 4
	if args.GateWeightBytes != wantWeightBytes || args.UpWeightBytes != wantWeightBytes {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "packed weight byte count mismatch", nil)
	}
	wantScaleBiasBytes := uint64(rows) * groupsPerRow * 2
	if args.GateScaleBytes != wantScaleBiasBytes || args.GateBiasBytes != wantScaleBiasBytes ||
		args.UpScaleBytes != wantScaleBiasBytes || args.UpBiasBytes != wantScaleBiasBytes {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "scale/bias byte count mismatch", nil)
	}
	if args.OutputBytes != uint64(rows)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "output byte count mismatch", nil)
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "gate weight bytes", args.GateWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "gate scale bytes", args.GateScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "gate bias bytes", args.GateBiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "up weight bytes", args.UpWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "up scale bytes", args.UpScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "up bias bytes", args.UpBiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	if cap(payload) < hipMLXQ4GELUTanhMulLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipMLXQ4GELUTanhMulLaunchArgsBytes)
	} else {
		payload = payload[:hipMLXQ4GELUTanhMulLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipMLXQ4GELUTanhMulLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.GateWeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.GateScalePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.GateBiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.UpWeightPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.UpScalePointer))
	binary.LittleEndian.PutUint64(payload[56:], uint64(args.UpBiasPointer))
	binary.LittleEndian.PutUint64(payload[64:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[72:], rows)
	binary.LittleEndian.PutUint32(payload[76:], cols)
	binary.LittleEndian.PutUint32(payload[80:], groupSize)
	binary.LittleEndian.PutUint32(payload[84:], bits)
	binary.LittleEndian.PutUint32(payload[88:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[92:], uint32(args.GateWeightBytes))
	binary.LittleEndian.PutUint32(payload[96:], uint32(args.GateScaleBytes))
	binary.LittleEndian.PutUint32(payload[100:], uint32(args.GateBiasBytes))
	binary.LittleEndian.PutUint32(payload[104:], uint32(args.UpWeightBytes))
	binary.LittleEndian.PutUint32(payload[108:], uint32(args.UpScaleBytes))
	binary.LittleEndian.PutUint32(payload[112:], uint32(args.UpBiasBytes))
	binary.LittleEndian.PutUint32(payload[116:], uint32(args.OutputBytes))
	return payload, nil
}

func (args hipMLXQ4GELUTanhMulBatchLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipMLXQ4GELUTanhMulBatchLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.GateWeightPointer == 0 || args.GateScalePointer == 0 ||
		args.GateBiasPointer == 0 || args.UpWeightPointer == 0 || args.UpScalePointer == 0 ||
		args.UpBiasPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "input, gate/up weights, scale/bias, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	batch, err := rocmDeviceKVPositiveUint32("batch", args.Batch)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	packedPerRow, groupsPerRow, err := hipMLXAffineLaunchPackedGroups("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", cols, groupSize, bits)
	if err != nil {
		return nil, err
	}
	if args.InputBytes != uint64(batch)*uint64(cols)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "input byte count mismatch", nil)
	}
	wantWeightBytes := uint64(rows) * packedPerRow * 4
	if args.GateWeightBytes != wantWeightBytes || args.UpWeightBytes != wantWeightBytes {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "packed weight byte count mismatch", nil)
	}
	wantScaleBiasBytes := uint64(rows) * groupsPerRow * 2
	if args.GateScaleBytes != wantScaleBiasBytes || args.GateBiasBytes != wantScaleBiasBytes ||
		args.UpScaleBytes != wantScaleBiasBytes || args.UpBiasBytes != wantScaleBiasBytes {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "scale/bias byte count mismatch", nil)
	}
	if args.OutputBytes != uint64(batch)*uint64(rows)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "output byte count mismatch", nil)
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "gate weight bytes", args.GateWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "gate scale bytes", args.GateScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "gate bias bytes", args.GateBiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "up weight bytes", args.UpWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "up scale bytes", args.UpScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "up bias bytes", args.UpBiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	if cap(payload) < hipMLXQ4GELUTanhMulBatchLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipMLXQ4GELUTanhMulBatchLaunchArgsBytes)
	} else {
		payload = payload[:hipMLXQ4GELUTanhMulBatchLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipMLXQ4GELUTanhMulBatchLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.GateWeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.GateScalePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.GateBiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.UpWeightPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.UpScalePointer))
	binary.LittleEndian.PutUint64(payload[56:], uint64(args.UpBiasPointer))
	binary.LittleEndian.PutUint64(payload[64:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[72:], rows)
	binary.LittleEndian.PutUint32(payload[76:], cols)
	binary.LittleEndian.PutUint32(payload[80:], groupSize)
	binary.LittleEndian.PutUint32(payload[84:], bits)
	binary.LittleEndian.PutUint32(payload[88:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[92:], uint32(args.GateWeightBytes))
	binary.LittleEndian.PutUint32(payload[96:], uint32(args.GateScaleBytes))
	binary.LittleEndian.PutUint32(payload[100:], uint32(args.GateBiasBytes))
	binary.LittleEndian.PutUint32(payload[104:], uint32(args.UpWeightBytes))
	binary.LittleEndian.PutUint32(payload[108:], uint32(args.UpScaleBytes))
	binary.LittleEndian.PutUint32(payload[112:], uint32(args.UpBiasBytes))
	binary.LittleEndian.PutUint32(payload[116:], uint32(args.OutputBytes))
	binary.LittleEndian.PutUint32(payload[120:], batch)
	return payload, nil
}

func (args hipMLXQ4GELUTanhMLPPersistentLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipMLXQ4GELUTanhMLPPersistentLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.GateWeightPointer == 0 || args.GateScalePointer == 0 ||
		args.GateBiasPointer == 0 || args.UpWeightPointer == 0 || args.UpScalePointer == 0 ||
		args.UpBiasPointer == 0 || args.DownWeightPointer == 0 || args.DownScalePointer == 0 ||
		args.DownBiasPointer == 0 || args.ActivationPointer == 0 || args.OutputPointer == 0 ||
		args.BarrierPointer == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "input, gate/up/down weights, activation, output, and barrier pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	downRows, err := rocmDeviceKVPositiveUint32("down rows", args.DownRows)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	gatePackedPerRow, gateGroupsPerRow, err := hipMLXAffineLaunchPackedGroups("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", cols, groupSize, bits)
	if err != nil {
		return nil, err
	}
	downPackedPerRow, downGroupsPerRow, err := hipMLXAffineLaunchPackedGroups("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", rows, groupSize, bits)
	if err != nil {
		return nil, err
	}
	if args.InputBytes != uint64(cols)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "input byte count mismatch", nil)
	}
	wantGateWeightBytes := uint64(rows) * gatePackedPerRow * 4
	if args.GateWeightBytes != wantGateWeightBytes || args.UpWeightBytes != wantGateWeightBytes {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "gate/up packed weight byte count mismatch", nil)
	}
	wantGateScaleBiasBytes := uint64(rows) * gateGroupsPerRow * 2
	if args.GateScaleBytes != wantGateScaleBiasBytes || args.GateBiasBytes != wantGateScaleBiasBytes ||
		args.UpScaleBytes != wantGateScaleBiasBytes || args.UpBiasBytes != wantGateScaleBiasBytes {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "gate/up scale/bias byte count mismatch", nil)
	}
	if args.DownWeightBytes != uint64(downRows)*downPackedPerRow*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "down packed weight byte count mismatch", nil)
	}
	if args.DownScaleBytes != uint64(downRows)*downGroupsPerRow*2 || args.DownBiasBytes != uint64(downRows)*downGroupsPerRow*2 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "down scale/bias byte count mismatch", nil)
	}
	if args.ActivationBytes != uint64(rows)*4 || args.OutputBytes != uint64(downRows)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "activation/output byte count mismatch", nil)
	}
	if args.BarrierBytes != 8 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "barrier byte count mismatch", nil)
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "gate weight bytes", args.GateWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "gate scale bytes", args.GateScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "gate bias bytes", args.GateBiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "up weight bytes", args.UpWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "up scale bytes", args.UpScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "up bias bytes", args.UpBiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "down weight bytes", args.DownWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "down scale bytes", args.DownScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "down bias bytes", args.DownBiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "activation bytes", args.ActivationBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	if cap(payload) < hipMLXQ4GELUTanhMLPPersistentLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipMLXQ4GELUTanhMLPPersistentLaunchArgsBytes)
	} else {
		payload = payload[:hipMLXQ4GELUTanhMLPPersistentLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipMLXQ4GELUTanhMLPPersistentLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.GateWeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.GateScalePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.GateBiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.UpWeightPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.UpScalePointer))
	binary.LittleEndian.PutUint64(payload[56:], uint64(args.UpBiasPointer))
	binary.LittleEndian.PutUint64(payload[64:], uint64(args.DownWeightPointer))
	binary.LittleEndian.PutUint64(payload[72:], uint64(args.DownScalePointer))
	binary.LittleEndian.PutUint64(payload[80:], uint64(args.DownBiasPointer))
	binary.LittleEndian.PutUint64(payload[88:], uint64(args.ActivationPointer))
	binary.LittleEndian.PutUint64(payload[96:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint64(payload[104:], uint64(args.BarrierPointer))
	binary.LittleEndian.PutUint32(payload[112:], rows)
	binary.LittleEndian.PutUint32(payload[116:], cols)
	binary.LittleEndian.PutUint32(payload[120:], downRows)
	binary.LittleEndian.PutUint32(payload[124:], groupSize)
	binary.LittleEndian.PutUint32(payload[128:], bits)
	binary.LittleEndian.PutUint32(payload[132:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[136:], uint32(args.GateWeightBytes))
	binary.LittleEndian.PutUint32(payload[140:], uint32(args.GateScaleBytes))
	binary.LittleEndian.PutUint32(payload[144:], uint32(args.GateBiasBytes))
	binary.LittleEndian.PutUint32(payload[148:], uint32(args.UpWeightBytes))
	binary.LittleEndian.PutUint32(payload[152:], uint32(args.UpScaleBytes))
	binary.LittleEndian.PutUint32(payload[156:], uint32(args.UpBiasBytes))
	binary.LittleEndian.PutUint32(payload[160:], uint32(args.DownWeightBytes))
	binary.LittleEndian.PutUint32(payload[164:], uint32(args.DownScaleBytes))
	binary.LittleEndian.PutUint32(payload[168:], uint32(args.DownBiasBytes))
	binary.LittleEndian.PutUint32(payload[172:], uint32(args.ActivationBytes))
	binary.LittleEndian.PutUint32(payload[176:], uint32(args.OutputBytes))
	binary.LittleEndian.PutUint32(payload[180:], uint32(args.BarrierBytes))
	return payload, nil
}

func (args hipMLXQ4GELUTanhProjLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipMLXQ4GELUTanhProjLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.WeightPointer == 0 || args.ScalePointer == 0 ||
		args.BiasPointer == 0 || args.MultiplierPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "input, weight, scale, bias, multiplier, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	packedPerRow, groupsPerRow, err := hipMLXAffineLaunchPackedGroups("rocm.hip.MLXQ4GELUTanhProjectionLaunch", cols, groupSize, bits)
	if err != nil {
		return nil, err
	}
	if args.InputBytes != uint64(cols)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "input byte count mismatch", nil)
	}
	if args.WeightBytes != uint64(rows)*packedPerRow*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "packed weight byte count mismatch", nil)
	}
	if args.ScaleBytes != uint64(rows)*groupsPerRow*2 || args.BiasBytes != uint64(rows)*groupsPerRow*2 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "scale/bias byte count mismatch", nil)
	}
	if args.MultiplierBytes != uint64(rows)*4 || args.OutputBytes != uint64(rows)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "multiplier/output byte count mismatch", nil)
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "weight bytes", args.WeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "scale bytes", args.ScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "bias bytes", args.BiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "multiplier bytes", args.MultiplierBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	if cap(payload) < hipMLXQ4GELUTanhProjLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipMLXQ4GELUTanhProjLaunchArgsBytes)
	} else {
		payload = payload[:hipMLXQ4GELUTanhProjLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipMLXQ4GELUTanhProjLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ScalePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.MultiplierPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[56:], rows)
	binary.LittleEndian.PutUint32(payload[60:], cols)
	binary.LittleEndian.PutUint32(payload[64:], groupSize)
	binary.LittleEndian.PutUint32(payload[68:], bits)
	binary.LittleEndian.PutUint32(payload[72:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[76:], uint32(args.WeightBytes))
	binary.LittleEndian.PutUint32(payload[80:], uint32(args.ScaleBytes))
	binary.LittleEndian.PutUint32(payload[84:], uint32(args.BiasBytes))
	binary.LittleEndian.PutUint32(payload[88:], uint32(args.MultiplierBytes))
	binary.LittleEndian.PutUint32(payload[92:], uint32(args.OutputBytes))
	return payload, nil
}

func (args hipMLXQ4GELUTanhProjBatchLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipMLXQ4GELUTanhProjBatchLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.WeightPointer == 0 || args.ScalePointer == 0 ||
		args.BiasPointer == 0 || args.MultiplierPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "input, weight, scale, bias, multiplier, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	batch, err := rocmDeviceKVPositiveUint32("batch", args.Batch)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	packedPerRow, groupsPerRow, err := hipMLXAffineLaunchPackedGroups("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", cols, groupSize, bits)
	if err != nil {
		return nil, err
	}
	if args.InputBytes != uint64(batch)*uint64(cols)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "input byte count mismatch", nil)
	}
	if args.WeightBytes != uint64(rows)*packedPerRow*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "packed weight byte count mismatch", nil)
	}
	if args.ScaleBytes != uint64(rows)*groupsPerRow*2 || args.BiasBytes != uint64(rows)*groupsPerRow*2 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "scale/bias byte count mismatch", nil)
	}
	if args.MultiplierBytes != uint64(batch)*uint64(rows)*4 || args.OutputBytes != uint64(batch)*uint64(rows)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "multiplier/output byte count mismatch", nil)
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "weight bytes", args.WeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "scale bytes", args.ScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "bias bytes", args.BiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "multiplier bytes", args.MultiplierBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	if cap(payload) < hipMLXQ4GELUTanhProjBatchLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipMLXQ4GELUTanhProjBatchLaunchArgsBytes)
	} else {
		payload = payload[:hipMLXQ4GELUTanhProjBatchLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipMLXQ4GELUTanhProjBatchLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ScalePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.MultiplierPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[56:], rows)
	binary.LittleEndian.PutUint32(payload[60:], cols)
	binary.LittleEndian.PutUint32(payload[64:], batch)
	binary.LittleEndian.PutUint32(payload[68:], groupSize)
	binary.LittleEndian.PutUint32(payload[72:], bits)
	binary.LittleEndian.PutUint32(payload[76:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[80:], uint32(args.WeightBytes))
	binary.LittleEndian.PutUint32(payload[84:], uint32(args.ScaleBytes))
	binary.LittleEndian.PutUint32(payload[88:], uint32(args.BiasBytes))
	binary.LittleEndian.PutUint32(payload[92:], uint32(args.MultiplierBytes))
	binary.LittleEndian.PutUint32(payload[96:], uint32(args.OutputBytes))
	return payload, nil
}

func (args hipRMSResidualAddGELUTanhProjLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipRMSResidualAddGELUTanhProjLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.ResidualPointer == 0 || args.WeightPointer == 0 ||
		args.ScalePointer == 0 || args.BiasPointer == 0 || args.MultiplierPointer == 0 ||
		args.ResidualOutputPointer == 0 || args.ActivationOutputPointer == 0 {
		return nil, core.E("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "input, residual, projection, multiplier, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	if err := hipValidateRMSNormDeviceWeightConfig("RMSResidualAddGELUTanhProjectionLaunch", hipRMSNormDeviceWeightConfig{
		WeightPointer:  args.RMSWeightPointer,
		WeightBytes:    args.RMSWeightBytes,
		Count:          int(cols),
		Epsilon:        args.Epsilon,
		WeightEncoding: args.RMSWeightEncoding,
		Flags:          args.RMSFlags,
	}); err != nil {
		return nil, err
	}
	if math.IsNaN(float64(args.OutputScale)) || math.IsInf(float64(args.OutputScale), 0) {
		return nil, core.E("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "output scale must be finite", nil)
	}
	packedPerRow, groupsPerRow, err := hipMLXAffineLaunchPackedGroups("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", cols, groupSize, bits)
	if err != nil {
		return nil, err
	}
	if args.InputBytes != uint64(cols)*4 || args.ResidualBytes != uint64(cols)*4 || args.ResidualOutputBytes != uint64(cols)*4 {
		return nil, core.E("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "residual-add byte count mismatch", nil)
	}
	if args.WeightBytes != uint64(rows)*packedPerRow*4 {
		return nil, core.E("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "packed weight byte count mismatch", nil)
	}
	if args.ScaleBytes != uint64(rows)*groupsPerRow*2 || args.BiasBytes != uint64(rows)*groupsPerRow*2 {
		return nil, core.E("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "scale/bias byte count mismatch", nil)
	}
	if args.MultiplierBytes != uint64(rows)*4 || args.ActivationOutputBytes != uint64(rows)*4 {
		return nil, core.E("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "multiplier/output byte count mismatch", nil)
	}
	if err := hipProjectionUint32Bytes("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "RMS weight bytes", args.RMSWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "residual bytes", args.ResidualBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "weight bytes", args.WeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "scale bytes", args.ScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "bias bytes", args.BiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "multiplier bytes", args.MultiplierBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "residual output bytes", args.ResidualOutputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "activation output bytes", args.ActivationOutputBytes); err != nil {
		return nil, err
	}
	if cap(payload) < hipRMSResidualAddGELUTanhProjLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipRMSResidualAddGELUTanhProjLaunchArgsBytes)
	} else {
		payload = payload[:hipRMSResidualAddGELUTanhProjLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipRMSResidualAddGELUTanhProjLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.RMSWeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ResidualPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.ScalePointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[56:], uint64(args.MultiplierPointer))
	binary.LittleEndian.PutUint64(payload[64:], uint64(args.ResidualOutputPointer))
	binary.LittleEndian.PutUint64(payload[72:], uint64(args.ActivationOutputPointer))
	binary.LittleEndian.PutUint32(payload[80:], rows)
	binary.LittleEndian.PutUint32(payload[84:], cols)
	binary.LittleEndian.PutUint32(payload[88:], groupSize)
	binary.LittleEndian.PutUint32(payload[92:], bits)
	binary.LittleEndian.PutUint32(payload[96:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[100:], uint32(args.RMSWeightBytes))
	binary.LittleEndian.PutUint32(payload[104:], uint32(args.ResidualBytes))
	binary.LittleEndian.PutUint32(payload[108:], uint32(args.WeightBytes))
	binary.LittleEndian.PutUint32(payload[112:], uint32(args.ScaleBytes))
	binary.LittleEndian.PutUint32(payload[116:], uint32(args.BiasBytes))
	binary.LittleEndian.PutUint32(payload[120:], uint32(args.MultiplierBytes))
	binary.LittleEndian.PutUint32(payload[124:], uint32(args.ResidualOutputBytes))
	binary.LittleEndian.PutUint32(payload[128:], uint32(args.ActivationOutputBytes))
	binary.LittleEndian.PutUint32(payload[132:], math.Float32bits(args.Epsilon))
	binary.LittleEndian.PutUint32(payload[136:], args.RMSWeightEncoding)
	binary.LittleEndian.PutUint32(payload[140:], args.RMSFlags)
	binary.LittleEndian.PutUint32(payload[144:], math.Float32bits(args.OutputScale))
	return payload, nil
}

func hipOrderedFloat32Key(value float32) uint32 {
	bits := math.Float32bits(value)
	if bits&0x80000000 != 0 {
		return ^bits
	}
	return bits ^ 0x80000000
}

func hipFloat32FromOrderedKey(key uint32) float32 {
	if key&0x80000000 != 0 {
		return math.Float32frombits(key ^ 0x80000000)
	}
	return math.Float32frombits(^key)
}

func hipPackGreedyBest(score float32, tokenID int) uint64 {
	return uint64(hipOrderedFloat32Key(score))<<32 | uint64(^uint32(tokenID))
}

func hipUnpackGreedyBest(packed uint64, softcap float32, vocabSize int) (hipGreedySampleResult, error) {
	if vocabSize <= 0 {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "vocab size must be positive", nil)
	}
	if packed == 0 {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "greedy projection did not produce a result", nil)
	}
	if softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "softcap must be non-negative and finite", nil)
	}
	tokenID := int(^uint32(packed))
	if tokenID < 0 || tokenID >= vocabSize {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "greedy projection token is out of range", nil)
	}
	score := hipFloat32FromOrderedKey(uint32(packed >> 32))
	if softcap > 0 {
		score = float32(math.Tanh(float64(score/softcap))) * softcap
	}
	return hipGreedySampleResult{TokenID: tokenID, Score: score}, nil
}

func hipUnpackGreedyBestTokenID(packedLow uint32, vocabSize int) (int, error) {
	if vocabSize <= 0 {
		return 0, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "vocab size must be positive", nil)
	}
	tokenID := int(^packedLow)
	if tokenID < 0 || tokenID >= vocabSize {
		return 0, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "greedy projection token is out of range", nil)
	}
	return tokenID, nil
}

func hipProjectionWeightEncodingCode(req hipProjectionRequest) (uint32, error) {
	switch {
	case len(req.F32) > 0 && len(req.FP16) == 0 && len(req.BF16) == 0 && len(req.Q8) == 0:
		return hipProjectionWeightEncodingF32, nil
	case len(req.FP16) > 0 && len(req.F32) == 0 && len(req.BF16) == 0 && len(req.Q8) == 0:
		return hipProjectionWeightEncodingFP16, nil
	case len(req.BF16) > 0 && len(req.F32) == 0 && len(req.FP16) == 0 && len(req.Q8) == 0:
		return hipProjectionWeightEncodingBF16, nil
	case len(req.Q8) > 0 && len(req.F32) == 0 && len(req.FP16) == 0 && len(req.BF16) == 0:
		return hipProjectionWeightEncodingQ8, nil
	default:
		return 0, core.E("rocm.hip.ProjectionLaunch", "exactly one projection weight encoding is required", nil)
	}
}

func hipRunMLXQ4ProjectionKernel(ctx context.Context, driver nativeHIPDriver, req hipMLXQ4ProjectionRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipMLXQ4ProjectionLaunchConfigForShape(launchBytes, req.Rows, req.Cols, req.GroupSize, hipMLXQ4ProjectionBitsOrDefault(req.Bits))
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	output, err := buffers.ReadOutput()
	if err != nil {
		return nil, err
	}
	success = true
	if err := buffers.Close(); err != nil {
		return nil, err
	}
	return output, nil
}

func hipRunMLXQ4ProjectionKernelWithDeviceWeights(ctx context.Context, driver nativeHIPDriver, input []float32, weightPointer, scalePointer, biasPointer nativeDevicePointer, weightBytes, scaleBytes, biasBytes uint64, rows, cols, groupSize int) ([]float32, error) {
	return hipRunMLXQ4ProjectionKernelWithDeviceWeightConfig(ctx, driver, input, hipMLXQ4DeviceWeightConfig{
		WeightPointer: weightPointer,
		ScalePointer:  scalePointer,
		BiasPointer:   biasPointer,
		WeightBytes:   weightBytes,
		ScaleBytes:    scaleBytes,
		BiasBytes:     biasBytes,
		Rows:          rows,
		Cols:          cols,
		GroupSize:     groupSize,
	})
}

func hipRunMLXQ4ProjectionKernelWithDeviceWeightConfig(ctx context.Context, driver nativeHIPDriver, input []float32, cfg hipMLXQ4DeviceWeightConfig) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if err := cfg.validate(input); err != nil {
		return nil, err
	}
	inputPayload, err := hipFloat32Payload(input)
	if err != nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "encode input", err)
	}
	inputBuffer, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection input", inputPayload, len(input))
	if err != nil {
		return nil, err
	}
	defer inputBuffer.Close()
	output, err := hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, inputBuffer, cfg)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	return hipReadFloat32DeviceOutput(output, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection output", cfg.Rows)
}

func hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig) (*hipDeviceByteBuffer, error) {
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device input is required", nil)
	}
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return nil, err
	}
	if input.SizeBytes() != uint64(cfg.Cols*4) {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device input byte count mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection output", uint64(cfg.Rows*4), cfg.Rows)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunMLXQ4ProjectionKernelWithDeviceInputOutput(ctx, driver, input, cfg, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunDenseProjectionSoftcapGreedyWithDeviceInputSuppress(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (hipGreedySampleResult, error) {
	if _, dense := cfg.denseWeightEncoding(); !dense {
		return hipGreedySampleResult{}, core.E("rocm.hip.DenseProjectionGreedyLaunch", "dense projection weights are required", nil)
	}
	logits, err := hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, input, cfg)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	defer logits.Close()
	var suppress *hipDeviceTokenBuffer
	if len(suppressTokens) > 0 {
		if workspace != nil {
			suppress, err = workspace.EnsureSuppressTokenBuffer(driver, suppressTokens)
		} else {
			suppress, err = hipUploadTokenIDs(driver, suppressTokens)
		}
		if err != nil {
			return hipGreedySampleResult{}, err
		}
		if workspace == nil {
			defer suppress.Close()
		}
	}
	return hipRunSoftcapGreedyKernelWithDeviceLogitsSuppressBuffer(ctx, driver, logits, softcap, suppress)
}

func hipRunMLXQ4ProjectionKernelWithDeviceInputOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer) error {
	return hipRunMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace(ctx, driver, input, cfg, output, nil)
}

func hipRunMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.MLXQ4ProjectionLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device input is required", nil)
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return err
	}
	if input.SizeBytes() != uint64(cfg.Cols*4) {
		return core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device input byte count mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != cfg.Rows || output.SizeBytes() != uint64(cfg.Rows*4) {
		return core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection output shape mismatch", nil)
	}
	if encoding, dense := cfg.denseWeightEncoding(); dense {
		return hipRunProjectionKernelWithDeviceInputWeightEncodingOutputWithWorkspace(ctx, driver, input, cfg.WeightPointer, cfg.WeightBytes, cfg.Rows, cfg.Cols, encoding, output, workspace)
	}
	launchArgs := hipMLXQ4ProjectionLaunchArgs{
		InputPointer:  input.Pointer(),
		WeightPointer: cfg.WeightPointer,
		ScalePointer:  cfg.ScalePointer,
		BiasPointer:   cfg.BiasPointer,
		OutputPointer: output.Pointer(),
		Rows:          cfg.Rows,
		Cols:          cfg.Cols,
		GroupSize:     cfg.GroupSize,
		Bits:          cfg.quantBits(),
		InputBytes:    input.SizeBytes(),
		WeightBytes:   cfg.WeightBytes,
		ScaleBytes:    cfg.ScaleBytes,
		BiasBytes:     cfg.BiasBytes,
		OutputBytes:   output.SizeBytes(),
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.ProjectionArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config, err := hipMLXQ4ProjectionLaunchConfigForShape(launchBytes, cfg.Rows, cfg.Cols, cfg.GroupSize, cfg.quantBits())
	if err != nil {
		return err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return err
	}
	return nil
}

func hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, batch int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch device input is required", nil)
	}
	if err := cfg.validateBatchInputCount(input.Count(), batch); err != nil {
		return nil, err
	}
	if input.SizeBytes() != uint64(batch*cfg.Cols*4) {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch device input byte count mismatch", nil)
	}
	outputCount := batch * cfg.Rows
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch output", uint64(outputCount*4), outputCount)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunMLXQ4ProjectionBatchKernelWithDeviceInputOutput(ctx, driver, input, cfg, batch, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunMLXQ4ProjectionBatchKernelWithDeviceInputOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, batch int, output *hipDeviceByteBuffer) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch device input is required", nil)
	}
	if err := cfg.validateBatchInputCount(input.Count(), batch); err != nil {
		return err
	}
	if input.SizeBytes() != uint64(batch*cfg.Cols*4) {
		return core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch device input byte count mismatch", nil)
	}
	outputCount := batch * cfg.Rows
	if output == nil || output.Pointer() == 0 || output.Count() != outputCount || output.SizeBytes() != uint64(outputCount*4) {
		return core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch output shape mismatch", nil)
	}
	if encoding, dense := cfg.denseWeightEncoding(); dense {
		return hipRunProjectionBatchKernelWithDeviceInputWeightEncodingOutput(ctx, driver, input, cfg.WeightPointer, cfg.WeightBytes, cfg.Rows, cfg.Cols, encoding, batch, output)
	}
	launchBytes, err := (hipMLXQ4ProjectionBatchLaunchArgs{
		InputPointer:  input.Pointer(),
		WeightPointer: cfg.WeightPointer,
		ScalePointer:  cfg.ScalePointer,
		BiasPointer:   cfg.BiasPointer,
		OutputPointer: output.Pointer(),
		Rows:          cfg.Rows,
		Cols:          cfg.Cols,
		Batch:         batch,
		GroupSize:     cfg.GroupSize,
		Bits:          cfg.quantBits(),
		InputBytes:    input.SizeBytes(),
		WeightBytes:   cfg.WeightBytes,
		ScaleBytes:    cfg.ScaleBytes,
		BiasBytes:     cfg.BiasBytes,
		OutputBytes:   output.SizeBytes(),
	}).Binary()
	if err != nil {
		return err
	}
	config, err := hipMLXQ4ProjectionBatchLaunchConfigForShape(launchBytes, cfg.Rows, cfg.Cols, cfg.GroupSize, cfg.quantBits(), batch)
	if err != nil {
		return err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return err
	}
	return nil
}

func hipRunMLXQ4TripleProjectionKernelWithDeviceInput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, firstCfg, secondCfg, thirdCfg hipMLXQ4DeviceWeightConfig) (*hipDeviceByteBuffer, *hipDeviceByteBuffer, *hipDeviceByteBuffer, *hipDeviceByteBuffer, error) {
	output, firstView, secondView, thirdView, err := hipRunMLXQ4TripleProjectionKernelWithDeviceInputViews(ctx, driver, input, firstCfg, secondCfg, thirdCfg)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	first := firstView
	second := secondView
	third := thirdView
	return output, &first, &second, &third, nil
}

func hipRunMLXQ4TripleProjectionKernelWithDeviceInputViews(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, firstCfg, secondCfg, thirdCfg hipMLXQ4DeviceWeightConfig) (*hipDeviceByteBuffer, hipDeviceByteBuffer, hipDeviceByteBuffer, hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input == nil || input.Pointer() == 0 {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "MLX q4 triple projection device input is required", nil)
	}
	if firstCfg.Cols != secondCfg.Cols || firstCfg.Cols != thirdCfg.Cols ||
		firstCfg.GroupSize != secondCfg.GroupSize || firstCfg.GroupSize != thirdCfg.GroupSize ||
		firstCfg.quantBits() != secondCfg.quantBits() || firstCfg.quantBits() != thirdCfg.quantBits() {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "triple projection input shapes must match", nil)
	}
	if err := firstCfg.validateInputCount(input.Count()); err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input.SizeBytes() != uint64(firstCfg.Cols*4) {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "MLX q4 triple projection device input byte count mismatch", nil)
	}
	if err := secondCfg.validateInputCount(input.Count()); err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input.SizeBytes() != uint64(secondCfg.Cols*4) {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "MLX q4 triple projection device input byte count mismatch", nil)
	}
	if err := thirdCfg.validateInputCount(input.Count()); err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input.SizeBytes() != uint64(thirdCfg.Cols*4) {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "MLX q4 triple projection device input byte count mismatch", nil)
	}
	totalRows := firstCfg.Rows + secondCfg.Rows + thirdCfg.Rows
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4TripleProjectionLaunch", "MLX q4 triple projection output", uint64(totalRows*4), totalRows)
	if err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	first, second, third, err := hipRunMLXQ4TripleProjectionKernelWithDeviceInputViewsOutput(ctx, driver, input, firstCfg, secondCfg, thirdCfg, output)
	if err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	success = true
	return output, first, second, third, nil
}

func hipRunMLXQ4TripleProjectionKernelWithDeviceInputViewsOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, firstCfg, secondCfg, thirdCfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer) (hipDeviceByteBuffer, hipDeviceByteBuffer, hipDeviceByteBuffer, error) {
	return hipRunMLXQ4TripleProjectionKernelWithDeviceInputViewsOutputWithWorkspace(ctx, driver, input, firstCfg, secondCfg, thirdCfg, output, nil)
}

func hipRunMLXQ4TripleProjectionKernelWithDeviceInputViewsOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, firstCfg, secondCfg, thirdCfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) (hipDeviceByteBuffer, hipDeviceByteBuffer, hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input == nil || input.Pointer() == 0 {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "MLX q4 triple projection device input is required", nil)
	}
	if firstCfg.Cols != secondCfg.Cols || firstCfg.Cols != thirdCfg.Cols ||
		firstCfg.GroupSize != secondCfg.GroupSize || firstCfg.GroupSize != thirdCfg.GroupSize ||
		firstCfg.quantBits() != secondCfg.quantBits() || firstCfg.quantBits() != thirdCfg.quantBits() {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "triple projection input shapes must match", nil)
	}
	if err := firstCfg.validateInputCount(input.Count()); err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input.SizeBytes() != uint64(firstCfg.Cols*4) {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "MLX q4 triple projection device input byte count mismatch", nil)
	}
	if err := secondCfg.validateInputCount(input.Count()); err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input.SizeBytes() != uint64(secondCfg.Cols*4) {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "MLX q4 triple projection device input byte count mismatch", nil)
	}
	if err := thirdCfg.validateInputCount(input.Count()); err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input.SizeBytes() != uint64(thirdCfg.Cols*4) {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "MLX q4 triple projection device input byte count mismatch", nil)
	}
	totalRows := firstCfg.Rows + secondCfg.Rows + thirdCfg.Rows
	if output == nil || output.Pointer() == 0 || output.Count() != totalRows || output.SizeBytes() != uint64(totalRows*4) {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "MLX q4 triple projection output shape mismatch", nil)
	}
	launchArgs := hipMLXQ4TripleProjLaunchArgs{
		InputPointer:        input.Pointer(),
		OutputPointer:       output.Pointer(),
		FirstWeightPointer:  firstCfg.WeightPointer,
		FirstScalePointer:   firstCfg.ScalePointer,
		FirstBiasPointer:    firstCfg.BiasPointer,
		SecondWeightPointer: secondCfg.WeightPointer,
		SecondScalePointer:  secondCfg.ScalePointer,
		SecondBiasPointer:   secondCfg.BiasPointer,
		ThirdWeightPointer:  thirdCfg.WeightPointer,
		ThirdScalePointer:   thirdCfg.ScalePointer,
		ThirdBiasPointer:    thirdCfg.BiasPointer,
		FirstRows:           firstCfg.Rows,
		SecondRows:          secondCfg.Rows,
		ThirdRows:           thirdCfg.Rows,
		Cols:                firstCfg.Cols,
		GroupSize:           firstCfg.GroupSize,
		Bits:                firstCfg.quantBits(),
		InputBytes:          input.SizeBytes(),
		OutputBytes:         output.SizeBytes(),
		FirstWeightBytes:    firstCfg.WeightBytes,
		FirstScaleBytes:     firstCfg.ScaleBytes,
		FirstBiasBytes:      firstCfg.BiasBytes,
		SecondWeightBytes:   secondCfg.WeightBytes,
		SecondScaleBytes:    secondCfg.ScaleBytes,
		SecondBiasBytes:     secondCfg.BiasBytes,
		ThirdWeightBytes:    thirdCfg.WeightBytes,
		ThirdScaleBytes:     thirdCfg.ScaleBytes,
		ThirdBiasBytes:      thirdCfg.BiasBytes,
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.TripleProjectionArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	config, err := hipMLXQ4TripleProjectionLaunchConfigForShape(launchBytes, totalRows, firstCfg.Cols, firstCfg.GroupSize, firstCfg.quantBits())
	if err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	first := hipDeviceByteBuffer{
		driver:    driver,
		pointer:   output.Pointer(),
		count:     firstCfg.Rows,
		sizeBytes: uint64(firstCfg.Rows * 4),
		borrowed:  true,
		label:     "MLX q4 triple projection first output",
	}
	secondOffset := nativeDevicePointer(firstCfg.Rows * 4)
	second := hipDeviceByteBuffer{
		driver:    driver,
		pointer:   output.Pointer() + secondOffset,
		count:     secondCfg.Rows,
		sizeBytes: uint64(secondCfg.Rows * 4),
		borrowed:  true,
		label:     "MLX q4 triple projection second output",
	}
	thirdOffset := nativeDevicePointer((firstCfg.Rows + secondCfg.Rows) * 4)
	third := hipDeviceByteBuffer{
		driver:    driver,
		pointer:   output.Pointer() + thirdOffset,
		count:     thirdCfg.Rows,
		sizeBytes: uint64(thirdCfg.Rows * 4),
		borrowed:  true,
		label:     "MLX q4 triple projection third output",
	}
	return first, second, third, nil
}

func hipRunMLXQ4PairProjectionKernelWithDeviceInputViews(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, firstCfg, secondCfg hipMLXQ4DeviceWeightConfig) (*hipDeviceByteBuffer, hipDeviceByteBuffer, hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input == nil || input.Pointer() == 0 {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "MLX q4 pair projection device input is required", nil)
	}
	if firstCfg.Cols != secondCfg.Cols || firstCfg.GroupSize != secondCfg.GroupSize || firstCfg.quantBits() != secondCfg.quantBits() {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "pair projection input shapes must match", nil)
	}
	if err := firstCfg.validateInputCount(input.Count()); err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input.SizeBytes() != uint64(firstCfg.Cols*4) {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "MLX q4 pair projection device input byte count mismatch", nil)
	}
	if err := secondCfg.validateInputCount(input.Count()); err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input.SizeBytes() != uint64(secondCfg.Cols*4) {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "MLX q4 pair projection device input byte count mismatch", nil)
	}
	totalRows := firstCfg.Rows + secondCfg.Rows
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4PairProjectionLaunch", "MLX q4 pair projection output", uint64(totalRows*4), totalRows)
	if err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	first, second, err := hipRunMLXQ4PairProjectionKernelWithDeviceInputViewsOutput(ctx, driver, input, firstCfg, secondCfg, output)
	if err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	success = true
	return output, first, second, nil
}

func hipRunMLXQ4PairProjectionKernelWithDeviceInputViewsOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, firstCfg, secondCfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer) (hipDeviceByteBuffer, hipDeviceByteBuffer, error) {
	return hipRunMLXQ4PairProjectionKernelWithDeviceInputViewsOutputWithWorkspace(ctx, driver, input, firstCfg, secondCfg, output, nil)
}

func hipRunMLXQ4PairProjectionKernelWithDeviceInputViewsOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, firstCfg, secondCfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) (hipDeviceByteBuffer, hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input == nil || input.Pointer() == 0 {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "MLX q4 pair projection device input is required", nil)
	}
	if firstCfg.Cols != secondCfg.Cols || firstCfg.GroupSize != secondCfg.GroupSize || firstCfg.quantBits() != secondCfg.quantBits() {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "pair projection input shapes must match", nil)
	}
	if err := firstCfg.validateInputCount(input.Count()); err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input.SizeBytes() != uint64(firstCfg.Cols*4) {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "MLX q4 pair projection device input byte count mismatch", nil)
	}
	if err := secondCfg.validateInputCount(input.Count()); err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input.SizeBytes() != uint64(secondCfg.Cols*4) {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "MLX q4 pair projection device input byte count mismatch", nil)
	}
	totalRows := firstCfg.Rows + secondCfg.Rows
	if output == nil || output.Pointer() == 0 || output.Count() != totalRows || output.SizeBytes() != uint64(totalRows*4) {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "MLX q4 pair projection output shape mismatch", nil)
	}
	launchArgs := hipMLXQ4TripleProjLaunchArgs{
		InputPointer:        input.Pointer(),
		OutputPointer:       output.Pointer(),
		FirstWeightPointer:  firstCfg.WeightPointer,
		FirstScalePointer:   firstCfg.ScalePointer,
		FirstBiasPointer:    firstCfg.BiasPointer,
		SecondWeightPointer: secondCfg.WeightPointer,
		SecondScalePointer:  secondCfg.ScalePointer,
		SecondBiasPointer:   secondCfg.BiasPointer,
		FirstRows:           firstCfg.Rows,
		SecondRows:          secondCfg.Rows,
		Cols:                firstCfg.Cols,
		GroupSize:           firstCfg.GroupSize,
		Bits:                firstCfg.quantBits(),
		InputBytes:          input.SizeBytes(),
		OutputBytes:         output.SizeBytes(),
		FirstWeightBytes:    firstCfg.WeightBytes,
		FirstScaleBytes:     firstCfg.ScaleBytes,
		FirstBiasBytes:      firstCfg.BiasBytes,
		SecondWeightBytes:   secondCfg.WeightBytes,
		SecondScaleBytes:    secondCfg.ScaleBytes,
		SecondBiasBytes:     secondCfg.BiasBytes,
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.TripleProjectionArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	config, err := hipMLXQ4PairProjectionLaunchConfig(launchBytes, totalRows)
	if err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	first := hipDeviceByteBuffer{
		driver:    driver,
		pointer:   output.Pointer(),
		count:     firstCfg.Rows,
		sizeBytes: uint64(firstCfg.Rows * 4),
		borrowed:  true,
		label:     "MLX q4 pair projection first output",
	}
	secondOffset := nativeDevicePointer(firstCfg.Rows * 4)
	second := hipDeviceByteBuffer{
		driver:    driver,
		pointer:   output.Pointer() + secondOffset,
		count:     secondCfg.Rows,
		sizeBytes: uint64(secondCfg.Rows * 4),
		borrowed:  true,
		label:     "MLX q4 pair projection second output",
	}
	return first, second, nil
}

func hipRunDenseGELUTanhMultiplyKernelWithDeviceInputOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, gateCfg, upCfg hipMLXQ4DeviceWeightConfig, batch int, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	count := batch * gateCfg.Rows
	gate, err := hipAllocateByteBuffer(driver, "rocm.hip.GELUTanhMultiplyLaunch", "dense gate projection output", uint64(count*4), count)
	if err != nil {
		return err
	}
	defer gate.Close()
	up, err := hipAllocateByteBuffer(driver, "rocm.hip.GELUTanhMultiplyLaunch", "dense up projection output", uint64(count*4), count)
	if err != nil {
		return err
	}
	defer up.Close()
	gateEncoding, _ := gateCfg.denseWeightEncoding()
	upEncoding, _ := upCfg.denseWeightEncoding()
	if batch == 1 {
		err = hipRunProjectionKernelWithDeviceInputWeightEncodingOutputWithWorkspace(ctx, driver, input, gateCfg.WeightPointer, gateCfg.WeightBytes, gateCfg.Rows, gateCfg.Cols, gateEncoding, gate, workspace)
		if err == nil {
			err = hipRunProjectionKernelWithDeviceInputWeightEncodingOutputWithWorkspace(ctx, driver, input, upCfg.WeightPointer, upCfg.WeightBytes, upCfg.Rows, upCfg.Cols, upEncoding, up, workspace)
		}
	} else {
		err = hipRunProjectionBatchKernelWithDeviceInputWeightEncodingOutput(ctx, driver, input, gateCfg.WeightPointer, gateCfg.WeightBytes, gateCfg.Rows, gateCfg.Cols, gateEncoding, batch, gate)
		if err == nil {
			err = hipRunProjectionBatchKernelWithDeviceInputWeightEncodingOutput(ctx, driver, input, upCfg.WeightPointer, upCfg.WeightBytes, upCfg.Rows, upCfg.Cols, upEncoding, batch, up)
		}
	}
	if err != nil {
		return err
	}
	return hipLaunchGELUTanhMultiplyDeviceBuffers(driver, &hipGELUTanhMultiplyDeviceBuffers{
		Gate:   gate,
		Up:     up,
		Output: output,
		Count:  count,
	})
}

func hipRunDenseGELUTanhProjectionKernelWithDeviceMultiplierOutput(ctx context.Context, driver nativeHIPDriver, input, multiplier *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, batch int, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	count := batch * cfg.Rows
	projected, err := hipAllocateByteBuffer(driver, "rocm.hip.GELUTanhProjectionLaunch", "dense gated projection output", uint64(count*4), count)
	if err != nil {
		return err
	}
	defer projected.Close()
	encoding, _ := cfg.denseWeightEncoding()
	if batch == 1 {
		err = hipRunProjectionKernelWithDeviceInputWeightEncodingOutputWithWorkspace(ctx, driver, input, cfg.WeightPointer, cfg.WeightBytes, cfg.Rows, cfg.Cols, encoding, projected, workspace)
	} else {
		err = hipRunProjectionBatchKernelWithDeviceInputWeightEncodingOutput(ctx, driver, input, cfg.WeightPointer, cfg.WeightBytes, cfg.Rows, cfg.Cols, encoding, batch, projected)
	}
	if err != nil {
		return err
	}
	return hipLaunchGELUTanhMultiplyDeviceBuffers(driver, &hipGELUTanhMultiplyDeviceBuffers{
		Gate:   projected,
		Up:     multiplier,
		Output: output,
		Count:  count,
	})
}

func hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, gateCfg, upCfg hipMLXQ4DeviceWeightConfig) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "MLX q4 GELU tanh multiply device input is required", nil)
	}
	if gateCfg.Rows != upCfg.Rows || gateCfg.Cols != upCfg.Cols || gateCfg.GroupSize != upCfg.GroupSize || gateCfg.quantBits() != upCfg.quantBits() {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "gate and up q4 projection shapes must match", nil)
	}
	if err := gateCfg.validateInputCount(input.Count()); err != nil {
		return nil, err
	}
	if err := upCfg.validateInputCount(input.Count()); err != nil {
		return nil, err
	}
	if input.SizeBytes() != uint64(gateCfg.Cols*4) {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "MLX q4 GELU tanh multiply device input byte count mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "MLX q4 GELU tanh multiply output", uint64(gateCfg.Rows*4), gateCfg.Rows)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInputOutput(ctx, driver, input, gateCfg, upCfg, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInputOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, gateCfg, upCfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer) error {
	return hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInputOutputWithWorkspace(ctx, driver, input, gateCfg, upCfg, output, nil)
}

func hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInputOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, gateCfg, upCfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "MLX q4 GELU tanh multiply device input is required", nil)
	}
	if gateCfg.Rows != upCfg.Rows || gateCfg.Cols != upCfg.Cols || gateCfg.GroupSize != upCfg.GroupSize || gateCfg.quantBits() != upCfg.quantBits() {
		return core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "gate and up q4 projection shapes must match", nil)
	}
	if err := gateCfg.validateInputCount(input.Count()); err != nil {
		return err
	}
	if err := upCfg.validateInputCount(input.Count()); err != nil {
		return err
	}
	if input.SizeBytes() != uint64(gateCfg.Cols*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "MLX q4 GELU tanh multiply device input byte count mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != gateCfg.Rows || output.SizeBytes() != uint64(gateCfg.Rows*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "MLX q4 GELU tanh multiply output shape mismatch", nil)
	}
	if gateEncoding, dense := gateCfg.denseWeightEncoding(); dense {
		upEncoding, upDense := upCfg.denseWeightEncoding()
		if !upDense || upEncoding != gateEncoding {
			return core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "gate and up dense projection encodings must match", nil)
		}
		return hipRunDenseGELUTanhMultiplyKernelWithDeviceInputOutput(ctx, driver, input, gateCfg, upCfg, 1, output, workspace)
	}
	launchArgs := hipMLXQ4GELUTanhMulLaunchArgs{
		InputPointer:      input.Pointer(),
		GateWeightPointer: gateCfg.WeightPointer,
		GateScalePointer:  gateCfg.ScalePointer,
		GateBiasPointer:   gateCfg.BiasPointer,
		UpWeightPointer:   upCfg.WeightPointer,
		UpScalePointer:    upCfg.ScalePointer,
		UpBiasPointer:     upCfg.BiasPointer,
		OutputPointer:     output.Pointer(),
		Rows:              gateCfg.Rows,
		Cols:              gateCfg.Cols,
		GroupSize:         gateCfg.GroupSize,
		Bits:              gateCfg.quantBits(),
		InputBytes:        input.SizeBytes(),
		GateWeightBytes:   gateCfg.WeightBytes,
		GateScaleBytes:    gateCfg.ScaleBytes,
		GateBiasBytes:     gateCfg.BiasBytes,
		UpWeightBytes:     upCfg.WeightBytes,
		UpScaleBytes:      upCfg.ScaleBytes,
		UpBiasBytes:       upCfg.BiasBytes,
		OutputBytes:       output.SizeBytes(),
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.GELUTanhMulArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config, err := hipMLXQ4GELUTanhMultiplyLaunchConfigForShape(launchBytes, gateCfg.Rows, gateCfg.Cols, gateCfg.GroupSize, gateCfg.quantBits())
	if err != nil {
		return err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return err
	}
	return nil
}

func hipMLXQ4GELUTanhMLPPersistentCompatible(input *hipDeviceByteBuffer, gateCfg, upCfg, downCfg hipMLXQ4DeviceWeightConfig) bool {
	return input != nil &&
		input.Pointer() != 0 &&
		gateCfg.WeightEncoding == 0 &&
		upCfg.WeightEncoding == 0 &&
		downCfg.WeightEncoding == 0 &&
		input.Count() == 1536 &&
		input.SizeBytes() == uint64(1536*4) &&
		gateCfg.Rows > 0 &&
		gateCfg.Rows%hipMLXQ4GELUTanhQ4G32Cols1536Row16RowsPerBlock == 0 &&
		gateCfg.Rows == upCfg.Rows &&
		gateCfg.Rows == downCfg.Cols &&
		gateCfg.Cols == 1536 &&
		upCfg.Cols == 1536 &&
		downCfg.Rows == 1536 &&
		gateCfg.GroupSize == 32 &&
		upCfg.GroupSize == 32 &&
		downCfg.GroupSize == 32 &&
		gateCfg.quantBits() == 4 &&
		upCfg.quantBits() == 4 &&
		downCfg.quantBits() == 4
}

func hipRunMLXQ4GELUTanhMLPPersistentKernelWithDeviceInputOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, gateCfg, upCfg, downCfg hipMLXQ4DeviceWeightConfig, activation, output, barrier *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "HIP driver is not available", nil)
	}
	if !hipMLXQ4GELUTanhMLPPersistentCompatible(input, gateCfg, upCfg, downCfg) {
		return core.E("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "unsupported persistent MLP shape", nil)
	}
	if err := gateCfg.validateInputCount(input.Count()); err != nil {
		return err
	}
	if err := upCfg.validateInputCount(input.Count()); err != nil {
		return err
	}
	if err := downCfg.validateInputCount(gateCfg.Rows); err != nil {
		return err
	}
	if activation == nil || activation.Pointer() == 0 || activation.Count() != gateCfg.Rows || activation.SizeBytes() != uint64(gateCfg.Rows*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "activation output shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != downCfg.Rows || output.SizeBytes() != uint64(downCfg.Rows*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "MLP output shape mismatch", nil)
	}
	if barrier == nil || barrier.Pointer() == 0 || barrier.SizeBytes() < 8 {
		return core.E("rocm.hip.MLXQ4GELUTanhMLPPersistentLaunch", "persistent barrier buffer is required", nil)
	}
	launchArgs := hipMLXQ4GELUTanhMLPPersistentLaunchArgs{
		InputPointer:      input.Pointer(),
		GateWeightPointer: gateCfg.WeightPointer,
		GateScalePointer:  gateCfg.ScalePointer,
		GateBiasPointer:   gateCfg.BiasPointer,
		UpWeightPointer:   upCfg.WeightPointer,
		UpScalePointer:    upCfg.ScalePointer,
		UpBiasPointer:     upCfg.BiasPointer,
		DownWeightPointer: downCfg.WeightPointer,
		DownScalePointer:  downCfg.ScalePointer,
		DownBiasPointer:   downCfg.BiasPointer,
		ActivationPointer: activation.Pointer(),
		OutputPointer:     output.Pointer(),
		BarrierPointer:    barrier.Pointer(),
		Rows:              gateCfg.Rows,
		Cols:              gateCfg.Cols,
		DownRows:          downCfg.Rows,
		GroupSize:         gateCfg.GroupSize,
		Bits:              gateCfg.quantBits(),
		InputBytes:        input.SizeBytes(),
		GateWeightBytes:   gateCfg.WeightBytes,
		GateScaleBytes:    gateCfg.ScaleBytes,
		GateBiasBytes:     gateCfg.BiasBytes,
		UpWeightBytes:     upCfg.WeightBytes,
		UpScaleBytes:      upCfg.ScaleBytes,
		UpBiasBytes:       upCfg.BiasBytes,
		DownWeightBytes:   downCfg.WeightBytes,
		DownScaleBytes:    downCfg.ScaleBytes,
		DownBiasBytes:     downCfg.BiasBytes,
		ActivationBytes:   activation.SizeBytes(),
		OutputBytes:       output.SizeBytes(),
		BarrierBytes:      barrier.SizeBytes(),
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.GELUTanhMLPPersistentArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config, err := hipMLXQ4GELUTanhMLPPersistentLaunchConfig(launchBytes)
	if err != nil {
		return err
	}
	return hipLaunchKernel(driver, config)
}

func hipRunMLXQ4GELUTanhMultiplyBatchKernelWithDeviceInput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, gateCfg, upCfg hipMLXQ4DeviceWeightConfig, batch int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "MLX q4 GELU tanh multiply batch device input is required", nil)
	}
	if gateCfg.Rows != upCfg.Rows || gateCfg.Cols != upCfg.Cols || gateCfg.GroupSize != upCfg.GroupSize || gateCfg.quantBits() != upCfg.quantBits() {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "gate and up q4 projection shapes must match", nil)
	}
	if err := gateCfg.validateBatchInputCount(input.Count(), batch); err != nil {
		return nil, err
	}
	if err := upCfg.validateBatchInputCount(input.Count(), batch); err != nil {
		return nil, err
	}
	if input.SizeBytes() != uint64(batch*gateCfg.Cols*4) {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "MLX q4 GELU tanh multiply batch device input byte count mismatch", nil)
	}
	outputCount := batch * gateCfg.Rows
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "MLX q4 GELU tanh multiply batch output", uint64(outputCount*4), outputCount)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipMLXQ4GELUTanhMulBatchLaunchArgs{
		InputPointer:      input.Pointer(),
		GateWeightPointer: gateCfg.WeightPointer,
		GateScalePointer:  gateCfg.ScalePointer,
		GateBiasPointer:   gateCfg.BiasPointer,
		UpWeightPointer:   upCfg.WeightPointer,
		UpScalePointer:    upCfg.ScalePointer,
		UpBiasPointer:     upCfg.BiasPointer,
		OutputPointer:     output.Pointer(),
		Rows:              gateCfg.Rows,
		Cols:              gateCfg.Cols,
		GroupSize:         gateCfg.GroupSize,
		Bits:              gateCfg.quantBits(),
		InputBytes:        input.SizeBytes(),
		GateWeightBytes:   gateCfg.WeightBytes,
		GateScaleBytes:    gateCfg.ScaleBytes,
		GateBiasBytes:     gateCfg.BiasBytes,
		UpWeightBytes:     upCfg.WeightBytes,
		UpScaleBytes:      upCfg.ScaleBytes,
		UpBiasBytes:       upCfg.BiasBytes,
		OutputBytes:       output.SizeBytes(),
		Batch:             batch,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipMLXQ4GELUTanhMultiplyBatchLaunchConfig(launchBytes, gateCfg.Rows, batch)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunMLXQ4GELUTanhMultiplyBatchKernelWithDeviceInputOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, gateCfg, upCfg hipMLXQ4DeviceWeightConfig, batch int, output *hipDeviceByteBuffer) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "MLX q4 GELU tanh multiply batch device input is required", nil)
	}
	if gateCfg.Rows != upCfg.Rows || gateCfg.Cols != upCfg.Cols || gateCfg.GroupSize != upCfg.GroupSize || gateCfg.quantBits() != upCfg.quantBits() {
		return core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "gate and up q4 projection shapes must match", nil)
	}
	if err := gateCfg.validateBatchInputCount(input.Count(), batch); err != nil {
		return err
	}
	if err := upCfg.validateBatchInputCount(input.Count(), batch); err != nil {
		return err
	}
	if input.SizeBytes() != uint64(batch*gateCfg.Cols*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "MLX q4 GELU tanh multiply batch device input byte count mismatch", nil)
	}
	outputCount := batch * gateCfg.Rows
	if output == nil || output.Pointer() == 0 || output.Count() != outputCount || output.SizeBytes() != uint64(outputCount*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "MLX q4 GELU tanh multiply batch output shape mismatch", nil)
	}
	if gateEncoding, dense := gateCfg.denseWeightEncoding(); dense {
		upEncoding, upDense := upCfg.denseWeightEncoding()
		if !upDense || upEncoding != gateEncoding {
			return core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "gate and up dense projection encodings must match", nil)
		}
		return hipRunDenseGELUTanhMultiplyKernelWithDeviceInputOutput(ctx, driver, input, gateCfg, upCfg, batch, output, nil)
	}
	launchBytes, err := (hipMLXQ4GELUTanhMulBatchLaunchArgs{
		InputPointer:      input.Pointer(),
		GateWeightPointer: gateCfg.WeightPointer,
		GateScalePointer:  gateCfg.ScalePointer,
		GateBiasPointer:   gateCfg.BiasPointer,
		UpWeightPointer:   upCfg.WeightPointer,
		UpScalePointer:    upCfg.ScalePointer,
		UpBiasPointer:     upCfg.BiasPointer,
		OutputPointer:     output.Pointer(),
		Rows:              gateCfg.Rows,
		Cols:              gateCfg.Cols,
		GroupSize:         gateCfg.GroupSize,
		Bits:              gateCfg.quantBits(),
		InputBytes:        input.SizeBytes(),
		GateWeightBytes:   gateCfg.WeightBytes,
		GateScaleBytes:    gateCfg.ScaleBytes,
		GateBiasBytes:     gateCfg.BiasBytes,
		UpWeightBytes:     upCfg.WeightBytes,
		UpScaleBytes:      upCfg.ScaleBytes,
		UpBiasBytes:       upCfg.BiasBytes,
		OutputBytes:       output.SizeBytes(),
		Batch:             batch,
	}).Binary()
	if err != nil {
		return err
	}
	config, err := hipMLXQ4GELUTanhMultiplyBatchLaunchConfig(launchBytes, gateCfg.Rows, batch)
	if err != nil {
		return err
	}
	return hipLaunchKernel(driver, config)
}

func hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplier(ctx context.Context, driver nativeHIPDriver, input, multiplier *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection device input is required", nil)
	}
	if multiplier == nil || multiplier.Pointer() == 0 || multiplier.Count() != cfg.Rows || multiplier.SizeBytes() != uint64(cfg.Rows*4) {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection multiplier device buffer shape mismatch", nil)
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return nil, err
	}
	if input.SizeBytes() != uint64(cfg.Cols*4) {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection device input byte count mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection output", uint64(cfg.Rows*4), cfg.Rows)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplierOutput(ctx, driver, input, multiplier, cfg, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplierOutput(ctx context.Context, driver nativeHIPDriver, input, multiplier *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer) error {
	return hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplierOutputWithWorkspace(ctx, driver, input, multiplier, cfg, output, nil)
}

func hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplierOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, input, multiplier *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection device input is required", nil)
	}
	if multiplier == nil || multiplier.Pointer() == 0 || multiplier.Count() != cfg.Rows || multiplier.SizeBytes() != uint64(cfg.Rows*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection multiplier device buffer shape mismatch", nil)
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return err
	}
	if input.SizeBytes() != uint64(cfg.Cols*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection device input byte count mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != cfg.Rows || output.SizeBytes() != uint64(cfg.Rows*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection output shape mismatch", nil)
	}
	if _, dense := cfg.denseWeightEncoding(); dense {
		return hipRunDenseGELUTanhProjectionKernelWithDeviceMultiplierOutput(ctx, driver, input, multiplier, cfg, 1, output, workspace)
	}
	launchArgs := hipMLXQ4GELUTanhProjLaunchArgs{
		InputPointer:      input.Pointer(),
		WeightPointer:     cfg.WeightPointer,
		ScalePointer:      cfg.ScalePointer,
		BiasPointer:       cfg.BiasPointer,
		MultiplierPointer: multiplier.Pointer(),
		OutputPointer:     output.Pointer(),
		Rows:              cfg.Rows,
		Cols:              cfg.Cols,
		GroupSize:         cfg.GroupSize,
		Bits:              cfg.quantBits(),
		InputBytes:        input.SizeBytes(),
		WeightBytes:       cfg.WeightBytes,
		ScaleBytes:        cfg.ScaleBytes,
		BiasBytes:         cfg.BiasBytes,
		MultiplierBytes:   multiplier.SizeBytes(),
		OutputBytes:       output.SizeBytes(),
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.GELUTanhProjArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config, err := hipMLXQ4GELUTanhProjectionLaunchConfigForShape(launchBytes, cfg.Rows, cfg.Cols, cfg.GroupSize, cfg.quantBits())
	if err != nil {
		return err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return err
	}
	return nil
}

func hipRunMLXQ4GELUTanhProjectionBatchKernelWithDeviceMultiplier(ctx context.Context, driver nativeHIPDriver, input, multiplier *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, batch int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "HIP driver is not available", nil)
	}
	if batch <= 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch size must be positive", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch device input is required", nil)
	}
	if multiplier == nil || multiplier.Pointer() == 0 || multiplier.Count() != batch*cfg.Rows || multiplier.SizeBytes() != uint64(batch*cfg.Rows*4) {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch multiplier device buffer shape mismatch", nil)
	}
	if err := cfg.validateInputCount(input.Count() / batch); err != nil {
		return nil, err
	}
	if input.Count() != batch*cfg.Cols || input.SizeBytes() != uint64(batch*cfg.Cols*4) {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch device input byte count mismatch", nil)
	}
	outputCount := batch * cfg.Rows
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch output", uint64(outputCount*4), outputCount)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunMLXQ4GELUTanhProjectionBatchKernelWithDeviceMultiplierOutput(ctx, driver, input, multiplier, cfg, batch, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunMLXQ4GELUTanhProjectionBatchKernelWithDeviceMultiplierOutput(ctx context.Context, driver nativeHIPDriver, input, multiplier *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, batch int, output *hipDeviceByteBuffer) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "HIP driver is not available", nil)
	}
	if batch <= 0 {
		return core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch size must be positive", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch device input is required", nil)
	}
	if multiplier == nil || multiplier.Pointer() == 0 || multiplier.Count() != batch*cfg.Rows || multiplier.SizeBytes() != uint64(batch*cfg.Rows*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch multiplier device buffer shape mismatch", nil)
	}
	if err := cfg.validateInputCount(input.Count() / batch); err != nil {
		return err
	}
	if input.Count() != batch*cfg.Cols || input.SizeBytes() != uint64(batch*cfg.Cols*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch device input byte count mismatch", nil)
	}
	outputCount := batch * cfg.Rows
	if output == nil || output.Pointer() == 0 || output.Count() != outputCount || output.SizeBytes() != uint64(outputCount*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch output shape mismatch", nil)
	}
	if _, dense := cfg.denseWeightEncoding(); dense {
		return hipRunDenseGELUTanhProjectionKernelWithDeviceMultiplierOutput(ctx, driver, input, multiplier, cfg, batch, output, nil)
	}
	launchBytes, err := (hipMLXQ4GELUTanhProjBatchLaunchArgs{
		InputPointer:      input.Pointer(),
		WeightPointer:     cfg.WeightPointer,
		ScalePointer:      cfg.ScalePointer,
		BiasPointer:       cfg.BiasPointer,
		MultiplierPointer: multiplier.Pointer(),
		OutputPointer:     output.Pointer(),
		Rows:              cfg.Rows,
		Cols:              cfg.Cols,
		Batch:             batch,
		GroupSize:         cfg.GroupSize,
		Bits:              cfg.quantBits(),
		InputBytes:        input.SizeBytes(),
		WeightBytes:       cfg.WeightBytes,
		ScaleBytes:        cfg.ScaleBytes,
		BiasBytes:         cfg.BiasBytes,
		MultiplierBytes:   multiplier.SizeBytes(),
		OutputBytes:       output.SizeBytes(),
	}).Binary()
	if err != nil {
		return err
	}
	config, err := hipMLXQ4GELUTanhProjectionBatchLaunchConfig(launchBytes, cfg.Rows, batch)
	if err != nil {
		return err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return err
	}
	return nil
}

func hipRunRMSNormResidualAddGELUTanhProjectionKernelWithDeviceMultiplierOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, rmsCfg hipRMSNormDeviceWeightConfig, multiplier *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, residualOutput, activationOutput *hipDeviceByteBuffer, outputScale float32, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 || residual == nil || residual.Pointer() == 0 {
		return core.E("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "input and residual buffers are required", nil)
	}
	if rmsCfg.Count != cfg.Cols {
		return core.E("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "RMS and projection dimensions must match", nil)
	}
	if err := hipValidateRMSNormDeviceWeightConfig("RMSResidualAddGELUTanhProjectionLaunch", rmsCfg); err != nil {
		return err
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return err
	}
	if input.Count() != cfg.Cols || residual.Count() != cfg.Cols ||
		input.SizeBytes() != uint64(cfg.Cols*4) || residual.SizeBytes() != uint64(cfg.Cols*4) {
		return core.E("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "input and residual buffer shape mismatch", nil)
	}
	if multiplier == nil || multiplier.Pointer() == 0 || multiplier.Count() != cfg.Rows || multiplier.SizeBytes() != uint64(cfg.Rows*4) {
		return core.E("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "multiplier buffer shape mismatch", nil)
	}
	if residualOutput == nil || residualOutput.Pointer() == 0 || residualOutput.Count() != cfg.Cols || residualOutput.SizeBytes() != uint64(cfg.Cols*4) {
		return core.E("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "residual output buffer shape mismatch", nil)
	}
	if activationOutput == nil || activationOutput.Pointer() == 0 || activationOutput.Count() != cfg.Rows || activationOutput.SizeBytes() != uint64(cfg.Rows*4) {
		return core.E("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "activation output buffer shape mismatch", nil)
	}
	if math.IsNaN(float64(outputScale)) || math.IsInf(float64(outputScale), 0) {
		return core.E("rocm.hip.RMSResidualAddGELUTanhProjectionLaunch", "output scale must be finite", nil)
	}
	launchArgs := hipRMSResidualAddGELUTanhProjLaunchArgs{
		InputPointer:            input.Pointer(),
		RMSWeightPointer:        rmsCfg.WeightPointer,
		ResidualPointer:         residual.Pointer(),
		WeightPointer:           cfg.WeightPointer,
		ScalePointer:            cfg.ScalePointer,
		BiasPointer:             cfg.BiasPointer,
		MultiplierPointer:       multiplier.Pointer(),
		ResidualOutputPointer:   residualOutput.Pointer(),
		ActivationOutputPointer: activationOutput.Pointer(),
		Rows:                    cfg.Rows,
		Cols:                    cfg.Cols,
		GroupSize:               cfg.GroupSize,
		Bits:                    cfg.quantBits(),
		InputBytes:              input.SizeBytes(),
		RMSWeightBytes:          rmsCfg.WeightBytes,
		ResidualBytes:           residual.SizeBytes(),
		WeightBytes:             cfg.WeightBytes,
		ScaleBytes:              cfg.ScaleBytes,
		BiasBytes:               cfg.BiasBytes,
		MultiplierBytes:         multiplier.SizeBytes(),
		ResidualOutputBytes:     residualOutput.SizeBytes(),
		ActivationOutputBytes:   activationOutput.SizeBytes(),
		Epsilon:                 rmsCfg.Epsilon,
		RMSWeightEncoding:       rmsCfg.WeightEncoding,
		RMSFlags:                rmsCfg.Flags,
		OutputScale:             outputScale,
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.RMSResidualAddGELUTanhProjArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config, err := hipRMSResidualAddGELUTanhProjectionLaunchConfig(launchBytes, cfg.Rows)
	if err != nil {
		return err
	}
	return hipLaunchKernel(driver, config)
}

func hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32) (hipGreedySampleResult, error) {
	return hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBuffer(ctx, driver, input, cfg, softcap, nil)
}

func hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBuffer(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, best *hipDeviceByteBuffer) (hipGreedySampleResult, error) {
	return hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressBuffer(ctx, driver, input, cfg, softcap, best, nil)
}

func hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressBuffer(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, best *hipDeviceByteBuffer, suppress *hipDeviceTokenBuffer) (hipGreedySampleResult, error) {
	return hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressBufferInitialized(ctx, driver, input, cfg, softcap, best, suppress, true)
}

func hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressBufferInitialized(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, best *hipDeviceByteBuffer, suppress *hipDeviceTokenBuffer, initializeBest bool) (hipGreedySampleResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipGreedySampleResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 projection device input is required", nil)
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return hipGreedySampleResult{}, err
	}
	if input.SizeBytes() != uint64(cfg.Cols*4) {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 projection device input byte count mismatch", nil)
	}
	if softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "softcap must be non-negative and finite", nil)
	}
	ownsBest := false
	if best == nil {
		var err error
		best, err = hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 projection greedy best", hipMLXQ4ProjectionBestBytes, 1)
		if err != nil {
			return hipGreedySampleResult{}, err
		}
		ownsBest = true
		initializeBest = true
	} else if best.Pointer() == 0 || best.Count() != 1 || best.SizeBytes() != hipMLXQ4ProjectionBestBytes {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 projection greedy best buffer shape mismatch", nil)
	}
	if suppress != nil && (suppress.Pointer() == 0 || suppress.Count() <= 0 || suppress.SizeBytes() != uint64(suppress.Count()*4)) {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 suppress token buffer shape mismatch", nil)
	}
	if ownsBest {
		defer best.Close()
	}
	if initializeBest {
		if err := hipMemsetDevice(driver, best.Pointer(), 0, best.SizeBytes()); err != nil {
			return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "initialize greedy best", err)
		}
	}
	launchBytes, err := (hipMLXQ4ProjectionLaunchArgs{
		InputPointer:  input.Pointer(),
		WeightPointer: cfg.WeightPointer,
		ScalePointer:  cfg.ScalePointer,
		BiasPointer:   cfg.BiasPointer,
		OutputPointer: best.Pointer(),
		Rows:          cfg.Rows,
		Cols:          cfg.Cols,
		GroupSize:     cfg.GroupSize,
		Bits:          cfg.quantBits(),
		InputBytes:    input.SizeBytes(),
		WeightBytes:   cfg.WeightBytes,
		ScaleBytes:    cfg.ScaleBytes,
		BiasBytes:     cfg.BiasBytes,
		OutputBytes:   best.SizeBytes(),
	}).GreedyBinary()
	if suppress != nil {
		launchBytes, err = (hipMLXQ4ProjectionLaunchArgs{
			InputPointer:    input.Pointer(),
			WeightPointer:   cfg.WeightPointer,
			ScalePointer:    cfg.ScalePointer,
			BiasPointer:     cfg.BiasPointer,
			OutputPointer:   best.Pointer(),
			SuppressPointer: suppress.Pointer(),
			Rows:            cfg.Rows,
			Cols:            cfg.Cols,
			GroupSize:       cfg.GroupSize,
			Bits:            cfg.quantBits(),
			SuppressCount:   suppress.Count(),
			InputBytes:      input.SizeBytes(),
			WeightBytes:     cfg.WeightBytes,
			ScaleBytes:      cfg.ScaleBytes,
			BiasBytes:       cfg.BiasBytes,
			OutputBytes:     best.SizeBytes(),
		}).GreedyBinary()
	}
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	config, err := hipMLXQ4ProjectionGreedyLaunchConfigForShape(launchBytes, cfg.Rows, cfg.Cols, cfg.GroupSize, cfg.quantBits())
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipGreedySampleResult{}, err
	}
	packed, err := hipReadDeviceUint64(driver, best.Pointer())
	if err != nil {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "copy greedy best", err)
	}
	return hipUnpackGreedyBest(packed, softcap, cfg.Rows)
}

func hipRunMLXQ4ProjectionSoftcapGreedyBatchKernelWithDeviceInput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, batch int) ([]hipGreedySampleResult, error) {
	return hipRunMLXQ4ProjectionSoftcapGreedyBatchKernelWithDeviceInputBufferSuppressBufferInitialized(ctx, driver, input, cfg, softcap, batch, nil, nil, true)
}

func hipRunMLXQ4ProjectionSoftcapGreedyBatchKernelWithDeviceInputBufferSuppress(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, batch int, best *hipDeviceByteBuffer, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) ([]hipGreedySampleResult, error) {
	var suppress *hipDeviceTokenBuffer
	if len(suppressTokens) > 0 {
		var err error
		if workspace != nil {
			suppress, err = workspace.EnsureSuppressTokenBuffer(driver, suppressTokens)
		} else {
			suppress, err = hipUploadTokenIDs(driver, suppressTokens)
		}
		if err != nil {
			return nil, err
		}
		if workspace == nil {
			defer suppress.Close()
		}
	}
	return hipRunMLXQ4ProjectionSoftcapGreedyBatchKernelWithDeviceInputBufferSuppressBufferInitialized(ctx, driver, input, cfg, softcap, batch, best, suppress, true)
}

func hipRunMLXQ4ProjectionSoftcapGreedyBatchKernelWithDeviceInputBufferSuppressBufferInitialized(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, batch int, best *hipDeviceByteBuffer, suppress *hipDeviceTokenBuffer, initializeBest bool) ([]hipGreedySampleResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "HIP driver is not available", nil)
	}
	if batch <= 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "MLX q4 projection greedy batch size must be positive", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "MLX q4 projection batch device input is required", nil)
	}
	if err := cfg.validateInputCount(cfg.Cols); err != nil {
		return nil, err
	}
	if input.Count() != batch*cfg.Cols || input.SizeBytes() != uint64(batch*cfg.Cols*4) {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "MLX q4 projection batch device input byte count mismatch", nil)
	}
	if softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "softcap must be non-negative and finite", nil)
	}
	ownsBest := false
	bestBytes := uint64(batch * hipMLXQ4ProjectionBestBytes)
	if best == nil {
		var err error
		best, err = hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "MLX q4 projection greedy batch best", bestBytes, batch)
		if err != nil {
			return nil, err
		}
		ownsBest = true
		initializeBest = true
	} else if best.Pointer() == 0 || best.Count() != batch || best.SizeBytes() != bestBytes {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "MLX q4 projection greedy batch best buffer shape mismatch", nil)
	}
	if suppress != nil && (suppress.Pointer() == 0 || suppress.Count() <= 0 || suppress.SizeBytes() != uint64(suppress.Count()*4)) {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "MLX q4 suppress token buffer shape mismatch", nil)
	}
	if ownsBest {
		defer best.Close()
	}
	if initializeBest {
		if err := hipMemsetDevice(driver, best.Pointer(), 0, best.SizeBytes()); err != nil {
			return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "initialize greedy batch best", err)
		}
	}
	launchArgs := hipMLXQ4ProjectionGreedyBatchLaunchArgs{
		InputPointer:  input.Pointer(),
		WeightPointer: cfg.WeightPointer,
		ScalePointer:  cfg.ScalePointer,
		BiasPointer:   cfg.BiasPointer,
		OutputPointer: best.Pointer(),
		Rows:          cfg.Rows,
		Cols:          cfg.Cols,
		Batch:         batch,
		GroupSize:     cfg.GroupSize,
		Bits:          cfg.quantBits(),
		InputBytes:    input.SizeBytes(),
		WeightBytes:   cfg.WeightBytes,
		ScaleBytes:    cfg.ScaleBytes,
		BiasBytes:     cfg.BiasBytes,
		OutputBytes:   best.SizeBytes(),
	}
	if suppress != nil {
		launchArgs.SuppressPointer = suppress.Pointer()
		launchArgs.SuppressCount = suppress.Count()
	}
	launchBytes, err := launchArgs.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipMLXQ4ProjectionGreedyBatchLaunchConfigForShape(launchBytes, cfg.Rows, cfg.Cols, cfg.GroupSize, cfg.quantBits(), batch)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	packed, err := hipReadUint64DeviceOutput(best, "rocm.hip.MLXQ4ProjectionGreedyBatchLaunch", "MLX q4 projection greedy batch best", batch)
	if err != nil {
		return nil, err
	}
	results := make([]hipGreedySampleResult, batch)
	for index, value := range packed {
		results[index], err = hipUnpackGreedyBest(value, softcap, cfg.Rows)
		if err != nil {
			return nil, err
		}
	}
	return results, nil
}

type nativeHIPDeviceUint64Reader interface {
	CopyDeviceToHostUint64(pointer nativeDevicePointer) (uint64, error)
}

type nativeHIPDeviceUint32Reader interface {
	CopyDeviceToHostUint32(pointer nativeDevicePointer) (uint32, error)
}

func hipReadDeviceUint64(driver nativeHIPDriver, pointer nativeDevicePointer) (uint64, error) {
	if reader, ok := driver.(nativeHIPDeviceUint64Reader); ok {
		return reader.CopyDeviceToHostUint64(pointer)
	}
	var payload [8]byte
	if err := driver.CopyDeviceToHost(pointer, payload[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(payload[:]), nil
}

func hipReadDeviceUint32(driver nativeHIPDriver, pointer nativeDevicePointer) (uint32, error) {
	if reader, ok := driver.(nativeHIPDeviceUint32Reader); ok {
		return reader.CopyDeviceToHostUint32(pointer)
	}
	var payload [4]byte
	if err := driver.CopyDeviceToHost(pointer, payload[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(payload[:]), nil
}

func hipReadUint64DeviceOutput(buffer *hipDeviceByteBuffer, operation, label string, count int) ([]uint64, error) {
	if buffer == nil || buffer.Pointer() == 0 {
		return nil, core.E(operation, label+" device buffer is required", nil)
	}
	if count <= 0 {
		return nil, core.E(operation, label+" count must be positive", nil)
	}
	if buffer.Count() != count || buffer.SizeBytes() != uint64(count*8) {
		return nil, core.E(operation, label+" byte count mismatch", nil)
	}
	payload := make([]byte, count*8)
	if err := buffer.driver.CopyDeviceToHost(buffer.Pointer(), payload); err != nil {
		return nil, core.E(operation, "copy "+label, err)
	}
	values := make([]uint64, count)
	for index := range values {
		values[index] = binary.LittleEndian.Uint64(payload[index*8:])
	}
	return values, nil
}

func hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppress(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, best *hipDeviceByteBuffer, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (hipGreedySampleResult, error) {
	usesBorrowedBest := workspace != nil && best != nil && best.borrowed
	greedy, err := hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressBufferInitialized(ctx, driver, input, cfg, softcap, best, nil, true)
	if err != nil || !hipTokenIsSuppressed(int32(greedy.TokenID), suppressTokens) {
		return greedy, err
	}
	if workspace != nil {
		suppress, err := workspace.EnsureSuppressTokenBuffer(driver, suppressTokens)
		if err != nil {
			return hipGreedySampleResult{}, err
		}
		suppressBest := best
		initializeBest := true
		if usesBorrowedBest {
			suppressBest, err = workspace.BorrowProjectionGreedyBest(driver)
			if err != nil {
				return hipGreedySampleResult{}, err
			}
		}
		return hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressBufferInitialized(ctx, driver, input, cfg, softcap, suppressBest, suppress, initializeBest)
	}
	logitsBuffer, err := hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, input, cfg)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	defer logitsBuffer.Close()
	logits, err := hipReadFloat32DeviceOutput(logitsBuffer, "rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 suppressed projection logits", cfg.Rows)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	logits, err = hipGemma4Q4SoftcapLogits(logits, softcap)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	tokenID, score, err := hipReferenceGreedySampleSuppress(logits, suppressTokens)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	return hipGreedySampleResult{TokenID: tokenID, Score: score}, nil
}

func hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressResult(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, best *hipDeviceByteBuffer, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (hipGreedySampleResult, *hipDeviceByteBuffer, error) {
	resultBuffer := best
	initializeBest := true
	if workspace != nil {
		var err error
		resultBuffer, err = workspace.BorrowProjectionGreedyBest(driver)
		if err != nil {
			return hipGreedySampleResult{}, nil, err
		}
	}
	greedy, err := hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressBufferInitialized(ctx, driver, input, cfg, softcap, resultBuffer, nil, initializeBest)
	if err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	if !hipTokenIsSuppressed(int32(greedy.TokenID), suppressTokens) {
		if resultBuffer == nil {
			return greedy, nil, nil
		}
		return greedy, resultBuffer, nil
	}
	if workspace != nil {
		suppress, err := workspace.EnsureSuppressTokenBuffer(driver, suppressTokens)
		if err != nil {
			return hipGreedySampleResult{}, nil, err
		}
		resultBuffer, err = workspace.BorrowProjectionGreedyBest(driver)
		if err != nil {
			return hipGreedySampleResult{}, nil, err
		}
		greedy, err = hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressBufferInitialized(ctx, driver, input, cfg, softcap, resultBuffer, suppress, true)
		return greedy, resultBuffer, err
	}
	logitsBuffer, err := hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, input, cfg)
	if err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	defer logitsBuffer.Close()
	logits, err := hipReadFloat32DeviceOutput(logitsBuffer, "rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 suppressed projection logits", cfg.Rows)
	if err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	logits, err = hipGemma4Q4SoftcapLogits(logits, softcap)
	if err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	tokenID, score, err := hipReferenceGreedySampleSuppress(logits, suppressTokens)
	if err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	return hipGreedySampleResult{TokenID: tokenID, Score: score}, nil, nil
}

func hipRunMLXQ4ProjectionSoftcapGreedyTokenKernelWithDeviceInputBufferSuppressResult(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, best *hipDeviceByteBuffer, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (hipGreedySampleResult, *hipDeviceByteBuffer, error) {
	resultBuffer := best
	initializeBest := true
	if workspace != nil {
		var err error
		resultBuffer, err = workspace.BorrowProjectionGreedyBest(driver)
		if err != nil {
			return hipGreedySampleResult{}, nil, err
		}
	}
	if resultBuffer == nil {
		greedy, device, err := hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressResult(ctx, driver, input, cfg, softcap, best, suppressTokens, workspace)
		return greedy, device, err
	}
	tokenID, err := hipRunMLXQ4ProjectionSoftcapGreedyTokenKernelWithDeviceInputBufferSuppressBufferInitialized(ctx, driver, input, cfg, softcap, resultBuffer, nil, initializeBest)
	if err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	if !hipTokenIsSuppressed(int32(tokenID), suppressTokens) {
		return hipGreedySampleResult{TokenID: tokenID}, resultBuffer, nil
	}
	if workspace != nil {
		suppress, err := workspace.EnsureSuppressTokenBuffer(driver, suppressTokens)
		if err != nil {
			return hipGreedySampleResult{}, nil, err
		}
		resultBuffer, err = workspace.BorrowProjectionGreedyBest(driver)
		if err != nil {
			return hipGreedySampleResult{}, nil, err
		}
		tokenID, err = hipRunMLXQ4ProjectionSoftcapGreedyTokenKernelWithDeviceInputBufferSuppressBufferInitialized(ctx, driver, input, cfg, softcap, resultBuffer, suppress, true)
		return hipGreedySampleResult{TokenID: tokenID}, resultBuffer, err
	}
	greedy, _, err := hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressResult(ctx, driver, input, cfg, softcap, best, suppressTokens, workspace)
	return greedy, resultBuffer, err
}

func hipRunMLXQ4ProjectionSoftcapGreedyTokenKernelWithDeviceInputBufferSuppressDevice(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, best *hipDeviceByteBuffer, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, error) {
	resultBuffer := best
	initializeBest := true
	var err error
	if workspace != nil {
		resultBuffer, err = workspace.BorrowProjectionGreedyBest(driver)
		if err != nil {
			return nil, err
		}
	}
	if resultBuffer == nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 projection greedy best buffer is required for deferred token read", nil)
	}
	var suppress *hipDeviceTokenBuffer
	if len(suppressTokens) > 0 {
		if workspace != nil {
			suppress, err = workspace.EnsureSuppressTokenBuffer(driver, suppressTokens)
		} else {
			suppress, err = hipUploadTokenIDs(driver, suppressTokens)
		}
		if err != nil {
			return nil, err
		}
		if workspace == nil {
			defer suppress.Close()
		}
	}
	if err := hipLaunchMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressBufferInitialized(ctx, driver, input, cfg, softcap, resultBuffer, suppress, initializeBest); err != nil {
		return nil, err
	}
	return resultBuffer, nil
}

func hipRunMLXQ4ProjectionSoftcapGreedyTokenKernelWithDeviceInputBufferSuppressBufferInitialized(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, best *hipDeviceByteBuffer, suppress *hipDeviceTokenBuffer, initializeBest bool) (int, error) {
	if err := hipLaunchMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressBufferInitialized(ctx, driver, input, cfg, softcap, best, suppress, initializeBest); err != nil {
		return 0, err
	}
	packedLow, err := hipReadDeviceUint32(driver, best.Pointer())
	if err != nil {
		return 0, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "copy greedy token", err)
	}
	return hipUnpackGreedyBestTokenID(packedLow, cfg.Rows)
}

func hipLaunchMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressBufferInitialized(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, best *hipDeviceByteBuffer, suppress *hipDeviceTokenBuffer, initializeBest bool) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 projection device input is required", nil)
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return err
	}
	if input.SizeBytes() != uint64(cfg.Cols*4) {
		return core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 projection device input byte count mismatch", nil)
	}
	if softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) {
		return core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "softcap must be non-negative and finite", nil)
	}
	if best == nil || best.Pointer() == 0 || best.Count() != 1 || best.SizeBytes() != hipMLXQ4ProjectionBestBytes {
		return core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 projection greedy best buffer shape mismatch", nil)
	}
	if suppress != nil && (suppress.Pointer() == 0 || suppress.Count() <= 0 || suppress.SizeBytes() != uint64(suppress.Count()*4)) {
		return core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 suppress token buffer shape mismatch", nil)
	}
	if initializeBest {
		if err := hipMemsetDevice(driver, best.Pointer(), 0, best.SizeBytes()); err != nil {
			return core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "initialize greedy best", err)
		}
	}
	launchArgs := hipMLXQ4ProjectionLaunchArgs{
		InputPointer:  input.Pointer(),
		WeightPointer: cfg.WeightPointer,
		ScalePointer:  cfg.ScalePointer,
		BiasPointer:   cfg.BiasPointer,
		OutputPointer: best.Pointer(),
		Rows:          cfg.Rows,
		Cols:          cfg.Cols,
		GroupSize:     cfg.GroupSize,
		Bits:          cfg.quantBits(),
		InputBytes:    input.SizeBytes(),
		WeightBytes:   cfg.WeightBytes,
		ScaleBytes:    cfg.ScaleBytes,
		BiasBytes:     cfg.BiasBytes,
		OutputBytes:   best.SizeBytes(),
	}
	if suppress != nil {
		launchArgs.SuppressPointer = suppress.Pointer()
		launchArgs.SuppressCount = suppress.Count()
	}
	launchBytes, err := launchArgs.GreedyBinary()
	if err != nil {
		return err
	}
	config, err := hipMLXQ4ProjectionGreedyLaunchConfigForShape(launchBytes, cfg.Rows, cfg.Cols, cfg.GroupSize, cfg.quantBits())
	if err != nil {
		return err
	}
	return hipLaunchKernel(driver, config)
}

func hipRunMLXQ4ProjectionSoftcapSelectedGreedyTokenKernelWithDeviceInputBufferResult(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, selected *hipDeviceTokenBuffer, best *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) (hipGreedySampleResult, *hipDeviceByteBuffer, error) {
	resultBuffer := best
	initializeBest := true
	if workspace != nil {
		var err error
		resultBuffer, err = workspace.BorrowProjectionGreedyBest(driver)
		if err != nil {
			return hipGreedySampleResult{}, nil, err
		}
	}
	tokenID, err := hipRunMLXQ4ProjectionSoftcapSelectedGreedyTokenKernelWithDeviceInputBufferInitialized(ctx, driver, input, cfg, softcap, selected, resultBuffer, initializeBest)
	if err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	return hipGreedySampleResult{TokenID: tokenID}, resultBuffer, nil
}

func hipRunMLXQ4ProjectionSoftcapSelectedGreedyTokenKernelWithDeviceInputBufferDevice(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, selected *hipDeviceTokenBuffer, best *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, error) {
	resultBuffer := best
	initializeBest := true
	var err error
	if workspace != nil {
		resultBuffer, err = workspace.BorrowProjectionGreedyBest(driver)
		if err != nil {
			return nil, err
		}
	}
	if resultBuffer == nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionSelectedGreedyLaunch", "MLX q4 selected projection greedy best buffer is required for deferred token read", nil)
	}
	if _, err := hipRunMLXQ4ProjectionSoftcapSelectedGreedyTokenKernelWithDeviceInputBufferInitialized(ctx, driver, input, cfg, softcap, selected, resultBuffer, initializeBest); err != nil {
		return nil, err
	}
	return resultBuffer, nil
}

func hipRunMLXQ4ProjectionSoftcapSelectedGreedyTokenKernelWithDeviceInputBufferInitialized(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, selected *hipDeviceTokenBuffer, best *hipDeviceByteBuffer, initializeBest bool) (int, error) {
	if err := hipLaunchMLXQ4ProjectionSoftcapSelectedGreedyKernelWithDeviceInputBufferInitialized(ctx, driver, input, cfg, softcap, selected, best, initializeBest); err != nil {
		return 0, err
	}
	packedLow, err := hipReadDeviceUint32(driver, best.Pointer())
	if err != nil {
		return 0, core.E("rocm.hip.MLXQ4ProjectionSelectedGreedyLaunch", "copy selected greedy token", err)
	}
	return hipUnpackGreedyBestTokenID(packedLow, cfg.Rows)
}

func hipLaunchMLXQ4ProjectionSoftcapSelectedGreedyKernelWithDeviceInputBufferInitialized(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, selected *hipDeviceTokenBuffer, best *hipDeviceByteBuffer, initializeBest bool) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.MLXQ4ProjectionSelectedGreedyLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.MLXQ4ProjectionSelectedGreedyLaunch", "MLX q4 projection device input is required", nil)
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return err
	}
	if input.SizeBytes() != uint64(cfg.Cols*4) {
		return core.E("rocm.hip.MLXQ4ProjectionSelectedGreedyLaunch", "MLX q4 projection device input byte count mismatch", nil)
	}
	if softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) {
		return core.E("rocm.hip.MLXQ4ProjectionSelectedGreedyLaunch", "softcap must be non-negative and finite", nil)
	}
	if selected == nil || selected.Pointer() == 0 || selected.Count() <= 0 || selected.SizeBytes() != uint64(selected.Count()*4) {
		return core.E("rocm.hip.MLXQ4ProjectionSelectedGreedyLaunch", "selected token buffer shape mismatch", nil)
	}
	if best == nil || best.Pointer() == 0 || best.Count() != 1 || best.SizeBytes() != hipMLXQ4ProjectionBestBytes {
		return core.E("rocm.hip.MLXQ4ProjectionSelectedGreedyLaunch", "MLX q4 projection greedy best buffer shape mismatch", nil)
	}
	if initializeBest {
		if err := hipMemsetDevice(driver, best.Pointer(), 0, best.SizeBytes()); err != nil {
			return core.E("rocm.hip.MLXQ4ProjectionSelectedGreedyLaunch", "initialize selected greedy best", err)
		}
	}
	launchBytes, err := (hipMLXQ4ProjectionLaunchArgs{
		InputPointer:    input.Pointer(),
		WeightPointer:   cfg.WeightPointer,
		ScalePointer:    cfg.ScalePointer,
		BiasPointer:     cfg.BiasPointer,
		OutputPointer:   best.Pointer(),
		SuppressPointer: selected.Pointer(),
		Rows:            cfg.Rows,
		Cols:            cfg.Cols,
		GroupSize:       cfg.GroupSize,
		Bits:            cfg.quantBits(),
		SuppressCount:   selected.Count(),
		InputBytes:      input.SizeBytes(),
		WeightBytes:     cfg.WeightBytes,
		ScaleBytes:      cfg.ScaleBytes,
		BiasBytes:       cfg.BiasBytes,
		OutputBytes:     best.SizeBytes(),
	}).GreedyBinary()
	if err != nil {
		return err
	}
	config, err := hipMLXQ4ProjectionSelectedGreedyLaunchConfigForShape(launchBytes, selected.Count(), cfg.Cols, cfg.GroupSize, cfg.quantBits())
	if err != nil {
		return err
	}
	return hipLaunchKernel(driver, config)
}

func hipRunOrderedEmbeddingCandidatesKernel(ctx context.Context, driver nativeHIPDriver, topK *hipDeviceByteBuffer, topKCount int, tokenOrderingPointer nativeDevicePointer, tokenOrderingBytes uint64, tokenOrderingElementBytes, numCentroids, tokensPerCentroid int, suppress *hipDeviceTokenBuffer, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceTokenBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "HIP driver is not available", nil)
	}
	if topK == nil || topK.Pointer() == 0 || topKCount <= 0 || topK.Count() != topKCount || topK.SizeBytes() != uint64(topKCount*hipMLXQ4ProjectionBestBytes) {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "top-k device buffer shape mismatch", nil)
	}
	if tokenOrderingPointer == 0 || tokenOrderingBytes == 0 {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "token-ordering device tensor is required", nil)
	}
	if tokenOrderingElementBytes != 4 && tokenOrderingElementBytes != 8 {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "token-ordering element bytes must be 4 or 8", nil)
	}
	if numCentroids <= 0 || tokensPerCentroid <= 0 {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "ordered embedding shape must be positive", nil)
	}
	tokenOrderingCount := numCentroids * tokensPerCentroid
	if tokenOrderingBytes != uint64(tokenOrderingCount*tokenOrderingElementBytes) {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "token-ordering byte count mismatch", nil)
	}
	if suppress != nil && (suppress.Pointer() == 0 || suppress.Count() <= 0 || suppress.SizeBytes() != uint64(suppress.Count()*4)) {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "suppress token buffer shape mismatch", nil)
	}
	if workspace == nil {
		return nil, core.E("rocm.hip.OrderedEmbeddingCandidatesLaunch", "attention workspace is required", nil)
	}
	outputCount := topKCount * tokensPerCentroid
	output, err := workspace.EnsureProjectionCandidateTokenOutput(driver, outputCount)
	if err != nil {
		return nil, err
	}
	launchArgs := hipOrderedEmbeddingCandidatesLaunchArgs{
		TopKPointer:               topK.Pointer(),
		TokenOrderingPointer:      tokenOrderingPointer,
		OutputPointer:             output.Pointer(),
		TopKCount:                 topKCount,
		NumCentroids:              numCentroids,
		TokensPerCentroid:         tokensPerCentroid,
		TokenOrderingElementBytes: tokenOrderingElementBytes,
		TokenOrderingCount:        tokenOrderingCount,
		OutputCount:               outputCount,
		TopKBytes:                 topK.SizeBytes(),
		TokenOrderingBytes:        tokenOrderingBytes,
		OutputBytes:               output.SizeBytes(),
	}
	if suppress != nil {
		launchArgs.SuppressPointer = suppress.Pointer()
		launchArgs.SuppressCount = suppress.Count()
	}
	launchBytes, err := launchArgs.BinaryInto(workspace.OrderedEmbeddingCandidatesArgs[:])
	if err != nil {
		return nil, err
	}
	config, err := hipOrderedEmbeddingCandidatesLaunchConfig(launchBytes, outputCount)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	return output, nil
}

func hipRunPackedTopKKernelWithWorkspace(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, inputCount, topK int, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, int, error) {
	return hipRunPackedTopKKernelWithWorkspaceOutput(ctx, driver, input, inputCount, topK, workspace, false)
}

func hipRunPackedTopKKernelWithWorkspaceOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, inputCount, topK int, workspace *hipAttentionHeadsChunkedWorkspace, workOutput bool) (*hipDeviceByteBuffer, int, error) {
	if input == nil || input.Pointer() == 0 {
		return nil, 0, core.E("rocm.hip.PackedTopKLaunch", "packed score input is required", nil)
	}
	if inputCount <= 0 || input.Count() != inputCount || input.SizeBytes() != uint64(inputCount*hipMLXQ4ProjectionBestBytes) {
		return nil, 0, core.E("rocm.hip.PackedTopKLaunch", "packed score input shape mismatch", nil)
	}
	if topK <= 0 || topK > hipPackedTopKMaxK {
		return nil, 0, core.E("rocm.hip.PackedTopKLaunch", "top-k must be within kernel maximum", nil)
	}
	if workspace == nil {
		return nil, 0, core.E("rocm.hip.PackedTopKLaunch", "attention workspace is required", nil)
	}
	chunkCount := (inputCount + hipPackedTopKChunkSize - 1) / hipPackedTopKChunkSize
	outputCount := chunkCount * topK
	var output *hipDeviceByteBuffer
	var err error
	if workOutput {
		output, err = workspace.EnsureProjectionTopKWorkOutput(driver, outputCount)
	} else {
		output, err = workspace.EnsureProjectionTopKOutput(driver, outputCount)
	}
	if err != nil {
		return nil, 0, err
	}
	launchBytes, err := (hipPackedTopKLaunchArgs{
		InputPointer:  input.Pointer(),
		OutputPointer: output.Pointer(),
		InputCount:    inputCount,
		OutputCount:   outputCount,
		TopK:          topK,
		ChunkSize:     hipPackedTopKChunkSize,
		InputBytes:    input.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
	}).BinaryInto(workspace.ProjectionTopKArgs[:])
	if err != nil {
		return nil, 0, err
	}
	config, err := hipPackedTopKLaunchConfig(launchBytes, chunkCount)
	if err != nil {
		return nil, 0, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, 0, err
	}
	return output, outputCount, nil
}

func hipRunPackedTopKReduceKernelWithWorkspace(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, inputCount, topK int, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, int, error) {
	current := input
	currentCount := inputCount
	workOutput := false
	for {
		output, outputCount, err := hipRunPackedTopKKernelWithWorkspaceOutput(ctx, driver, current, currentCount, topK, workspace, workOutput)
		if err != nil {
			return nil, 0, err
		}
		if outputCount <= topK {
			return output, outputCount, nil
		}
		current = output
		currentCount = outputCount
		workOutput = !workOutput
	}
}

func hipRunMLXQ4ProjectionSoftcapScoreTopKDeviceWithDeviceInputBufferSuppress(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, topK int, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, int, error) {
	if input == nil || input.Pointer() == 0 {
		return nil, 0, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "MLX q4 projection device input is required", nil)
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return nil, 0, err
	}
	if input.SizeBytes() != uint64(cfg.Cols*4) {
		return nil, 0, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "MLX q4 projection device input byte count mismatch", nil)
	}
	if topK <= 0 || topK > cfg.Rows {
		return nil, 0, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "top-k must be within vocabulary size", nil)
	}
	if softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) {
		return nil, 0, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "softcap must be non-negative and finite", nil)
	}
	if workspace == nil {
		return nil, 0, core.E("rocm.hip.PackedTopKLaunch", "attention workspace is required", nil)
	}
	var suppress *hipDeviceTokenBuffer
	var err error
	if len(suppressTokens) > 0 {
		suppress, err = workspace.EnsureSuppressTokenBuffer(driver, suppressTokens)
		if err != nil {
			return nil, 0, err
		}
	}
	scores, err := workspace.EnsureProjectionScoreOutput(driver, cfg.Rows)
	if err != nil {
		return nil, 0, err
	}
	launchArgs := hipMLXQ4ProjectionLaunchArgs{
		InputPointer:  input.Pointer(),
		WeightPointer: cfg.WeightPointer,
		ScalePointer:  cfg.ScalePointer,
		BiasPointer:   cfg.BiasPointer,
		OutputPointer: scores.Pointer(),
		Rows:          cfg.Rows,
		Cols:          cfg.Cols,
		GroupSize:     cfg.GroupSize,
		Bits:          cfg.quantBits(),
		InputBytes:    input.SizeBytes(),
		WeightBytes:   cfg.WeightBytes,
		ScaleBytes:    cfg.ScaleBytes,
		BiasBytes:     cfg.BiasBytes,
		OutputBytes:   scores.SizeBytes(),
	}
	if suppress != nil {
		launchArgs.SuppressPointer = suppress.Pointer()
		launchArgs.SuppressCount = suppress.Count()
	}
	launchBytes, err := launchArgs.ScoresBinaryInto(workspace.ProjectionScoresArgs[:])
	if err != nil {
		return nil, 0, err
	}
	config, err := hipMLXQ4ProjectionScoresLaunchConfigForShape(launchBytes, cfg.Rows, cfg.Cols, cfg.GroupSize, cfg.quantBits())
	if err != nil {
		return nil, 0, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, 0, err
	}
	return hipRunPackedTopKReduceKernelWithWorkspace(ctx, driver, scores, cfg.Rows, topK, workspace)
}

func hipRunPackedTopKSampleKernel(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, inputCount, topK int, generateTemperature, generateTopP, softcap float32, draw float64, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) (hipGreedySampleResult, *hipDeviceByteBuffer, error) {
	if input == nil || input.Pointer() == 0 {
		return hipGreedySampleResult{}, nil, core.E("rocm.hip.PackedTopKSampleLaunch", "packed candidate input is required", nil)
	}
	if inputCount <= 0 || input.Count() != inputCount || input.SizeBytes() != uint64(inputCount*hipMLXQ4ProjectionBestBytes) {
		return hipGreedySampleResult{}, nil, core.E("rocm.hip.PackedTopKSampleLaunch", "packed candidate input shape mismatch", nil)
	}
	if topK <= 0 || topK > inputCount || topK > hipPackedTopKMaxK {
		return hipGreedySampleResult{}, nil, core.E("rocm.hip.PackedTopKSampleLaunch", "top-k must be within input and kernel maximum", nil)
	}
	if workspace == nil {
		return hipGreedySampleResult{}, nil, core.E("rocm.hip.PackedTopKSampleLaunch", "attention workspace is required", nil)
	}
	if output == nil || output.Pointer() == 0 {
		var err error
		output, err = hipAllocateByteBuffer(driver, "rocm.hip.PackedTopKSampleLaunch", "sampled packed top-k", hipMLXQ4ProjectionBestBytes, 1)
		if err != nil {
			return hipGreedySampleResult{}, nil, err
		}
	}
	if output.Count() != 1 || output.SizeBytes() != hipMLXQ4ProjectionBestBytes {
		return hipGreedySampleResult{}, nil, core.E("rocm.hip.PackedTopKSampleLaunch", "sample output shape mismatch", nil)
	}
	launchBytes, err := (hipPackedTopKSampleLaunchArgs{
		InputPointer:  input.Pointer(),
		OutputPointer: output.Pointer(),
		InputCount:    inputCount,
		TopK:          topK,
		InputBytes:    input.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
		Temperature:   generateTemperature,
		TopP:          generateTopP,
		Draw:          draw,
		Softcap:       softcap,
	}).BinaryInto(workspace.ProjectionTopKSampleArgs[:])
	if err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNamePackedTopKSample,
		Args:   launchBytes,
		GridX:  1,
		GridY:  1,
		GridZ:  1,
		BlockX: 1,
		BlockY: 1,
		BlockZ: 1,
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	packed, err := hipReadDeviceUint64(driver, output.Pointer())
	if err != nil {
		return hipGreedySampleResult{}, nil, core.E("rocm.hip.PackedTopKSampleLaunch", "copy sampled packed top-k", err)
	}
	result, err := hipUnpackGreedyBest(packed, 0, math.MaxInt32)
	if err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	return result, output, nil
}

func hipRunMLXQ4ProjectionSoftcapScoreKernelWithDeviceInputBufferSuppress(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, topK int, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) ([]hipGreedySampleResult, error) {
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "MLX q4 projection device input is required", nil)
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return nil, err
	}
	if input.SizeBytes() != uint64(cfg.Cols*4) {
		return nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "MLX q4 projection device input byte count mismatch", nil)
	}
	if topK <= 0 || topK > cfg.Rows {
		return nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "top-k must be within vocabulary size", nil)
	}
	if softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) {
		return nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "softcap must be non-negative and finite", nil)
	}
	var suppress *hipDeviceTokenBuffer
	var err error
	if len(suppressTokens) > 0 {
		if workspace != nil {
			suppress, err = workspace.EnsureSuppressTokenBuffer(driver, suppressTokens)
		} else {
			suppress, err = hipUploadTokenIDs(driver, suppressTokens)
		}
		if err != nil {
			return nil, err
		}
		if workspace == nil {
			defer suppress.Close()
		}
	}
	var scores *hipDeviceByteBuffer
	if workspace != nil {
		scores, err = workspace.EnsureProjectionScoreOutput(driver, cfg.Rows)
		if err != nil {
			return nil, err
		}
	} else {
		scores, err = hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4ProjectionScoresLaunch", "MLX q4 projection packed scores", uint64(cfg.Rows*hipMLXQ4ProjectionBestBytes), cfg.Rows)
		if err != nil {
			return nil, err
		}
		defer scores.Close()
	}
	launchArgs := hipMLXQ4ProjectionLaunchArgs{
		InputPointer:  input.Pointer(),
		WeightPointer: cfg.WeightPointer,
		ScalePointer:  cfg.ScalePointer,
		BiasPointer:   cfg.BiasPointer,
		OutputPointer: scores.Pointer(),
		Rows:          cfg.Rows,
		Cols:          cfg.Cols,
		GroupSize:     cfg.GroupSize,
		Bits:          cfg.quantBits(),
		InputBytes:    input.SizeBytes(),
		WeightBytes:   cfg.WeightBytes,
		ScaleBytes:    cfg.ScaleBytes,
		BiasBytes:     cfg.BiasBytes,
		OutputBytes:   scores.SizeBytes(),
	}
	if suppress != nil {
		launchArgs.SuppressPointer = suppress.Pointer()
		launchArgs.SuppressCount = suppress.Count()
	}
	var launchBytes []byte
	if workspace != nil {
		launchBytes, err = launchArgs.ScoresBinaryInto(workspace.ProjectionScoresArgs[:])
	} else {
		launchBytes, err = launchArgs.ScoresBinary()
	}
	if err != nil {
		return nil, err
	}
	config, err := hipMLXQ4ProjectionScoresLaunchConfigForShape(launchBytes, cfg.Rows, cfg.Cols, cfg.GroupSize, cfg.quantBits())
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	var top []uint64
	if workspace != nil {
		partial, partialCount, err := hipRunPackedTopKReduceKernelWithWorkspace(ctx, driver, scores, cfg.Rows, topK, workspace)
		if err != nil {
			return nil, err
		}
		payload, err := workspace.ProjectionTopKPayload(partialCount)
		if err != nil {
			return nil, err
		}
		if err := driver.CopyDeviceToHost(partial.Pointer(), payload); err != nil {
			return nil, core.E("rocm.hip.PackedTopKLaunch", "copy packed top-k partial scores", err)
		}
		top = hipSortedPackedScoresBytesInto(payload, topK, workspace.ProjectionTopPacked)
		workspace.ProjectionTopPacked = top
	} else {
		packed, err := hipReadUint64DeviceOutput(scores, "rocm.hip.MLXQ4ProjectionScoresLaunch", "MLX q4 projection packed scores", cfg.Rows)
		if err != nil {
			return nil, err
		}
		top = hipTopPackedScores(packed, topK)
	}
	var candidates []hipGreedySampleResult
	if workspace != nil {
		candidates = workspace.ProjectionCandidates[:0]
		if cap(candidates) < len(top) {
			candidates = make([]hipGreedySampleResult, 0, len(top))
		}
	} else {
		candidates = make([]hipGreedySampleResult, 0, len(top))
	}
	for _, value := range top {
		candidate, err := hipUnpackGreedyBest(value, softcap, cfg.Rows)
		if err != nil {
			return nil, err
		}
		candidates = append(candidates, candidate)
	}
	if len(candidates) == 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "score projection did not produce candidates", nil)
	}
	if workspace != nil {
		workspace.ProjectionCandidates = candidates
	}
	return candidates, nil
}

func hipRunMLXQ4ProjectionSoftcapSampleKernelWithDeviceInputBufferSuppress(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, topK int, temperature, topP float32, draw float64, best *hipDeviceByteBuffer, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (hipGreedySampleResult, *hipDeviceByteBuffer, error) {
	if input == nil || input.Pointer() == 0 {
		return hipGreedySampleResult{}, nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "MLX q4 projection device input is required", nil)
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	if input.SizeBytes() != uint64(cfg.Cols*4) {
		return hipGreedySampleResult{}, nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "MLX q4 projection device input byte count mismatch", nil)
	}
	if topK <= 0 || topK > cfg.Rows {
		return hipGreedySampleResult{}, nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "top-k must be within vocabulary size", nil)
	}
	if softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) {
		return hipGreedySampleResult{}, nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "softcap must be non-negative and finite", nil)
	}
	if workspace == nil {
		return hipGreedySampleResult{}, nil, core.E("rocm.hip.PackedTopKSampleLaunch", "attention workspace is required", nil)
	}
	var suppress *hipDeviceTokenBuffer
	var err error
	if len(suppressTokens) > 0 {
		suppress, err = workspace.EnsureSuppressTokenBuffer(driver, suppressTokens)
		if err != nil {
			return hipGreedySampleResult{}, nil, err
		}
	}
	scores, err := workspace.EnsureProjectionScoreOutput(driver, cfg.Rows)
	if err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	launchArgs := hipMLXQ4ProjectionLaunchArgs{
		InputPointer:  input.Pointer(),
		WeightPointer: cfg.WeightPointer,
		ScalePointer:  cfg.ScalePointer,
		BiasPointer:   cfg.BiasPointer,
		OutputPointer: scores.Pointer(),
		Rows:          cfg.Rows,
		Cols:          cfg.Cols,
		GroupSize:     cfg.GroupSize,
		Bits:          cfg.quantBits(),
		InputBytes:    input.SizeBytes(),
		WeightBytes:   cfg.WeightBytes,
		ScaleBytes:    cfg.ScaleBytes,
		BiasBytes:     cfg.BiasBytes,
		OutputBytes:   scores.SizeBytes(),
	}
	if suppress != nil {
		launchArgs.SuppressPointer = suppress.Pointer()
		launchArgs.SuppressCount = suppress.Count()
	}
	launchBytes, err := launchArgs.ScoresBinaryInto(workspace.ProjectionScoresArgs[:])
	if err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	config, err := hipMLXQ4ProjectionScoresLaunchConfigForShape(launchBytes, cfg.Rows, cfg.Cols, cfg.GroupSize, cfg.quantBits())
	if err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	var receiptHostSample hipGreedySampleResult
	if receipts := hipActiveLogitSpreadReceipts(); receipts != nil {
		packedScores, readErr := hipReadUint64DeviceOutput(scores, "rocm.hip.LogitSpread", "packed sampler projection scores", cfg.Rows)
		if readErr != nil {
			return hipGreedySampleResult{}, nil, readErr
		}
		rawScores := make([]float32, len(packedScores))
		for tokenID, packed := range packedScores {
			rawScores[tokenID] = hipFloat32FromOrderedKey(uint32(packed >> 32))
		}
		receipts.recordNext("device-topk", "sampler-input-pre-softcap", rawScores)
		softcapped, softcapErr := hipGemma4Q4SoftcapLogits(rawScores, softcap)
		if softcapErr != nil {
			return hipGreedySampleResult{}, nil, softcapErr
		}
		receiptHostSample, readErr = hipGemma4Q4HostSampleResult(softcapped, inference.GenerateConfig{Temperature: temperature, TopK: topK, TopP: topP, RepeatPenalty: 1}, nil, nil, draw)
		if readErr != nil {
			return hipGreedySampleResult{}, nil, readErr
		}
	}
	partial, partialCount, err := hipRunPackedTopKReduceKernelWithWorkspace(ctx, driver, scores, cfg.Rows, topK, workspace)
	if err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	result, best, err := hipRunPackedTopKSampleKernel(ctx, driver, partial, partialCount, topK, temperature, topP, softcap, draw, best, workspace)
	if err != nil {
		return hipGreedySampleResult{}, nil, err
	}
	if hipActiveLogitSpreadReceipts() != nil {
		core.Println(core.Sprintf("HIP_DEVICE_SAMPLE_ORACLE draw=%.9g device=%d host=%d", draw, result.TokenID, receiptHostSample.TokenID))
	}
	if result.TokenID < 0 || result.TokenID >= cfg.Rows {
		return hipGreedySampleResult{}, nil, core.E("rocm.hip.PackedTopKSampleLaunch", "sampled token is out of range", nil)
	}
	return result, best, nil
}

func hipTopPackedScores(values []uint64, topK int) []uint64 {
	if topK <= 0 || len(values) == 0 {
		return nil
	}
	top := make([]uint64, 0, min(topK, len(values)))
	for _, value := range values {
		if value == 0 {
			continue
		}
		insert := len(top)
		for insert > 0 && value > top[insert-1] {
			insert--
		}
		if insert >= topK {
			continue
		}
		if len(top) < topK {
			top = append(top, 0)
			copy(top[insert+1:], top[insert:])
		} else {
			copy(top[insert+1:], top[insert:len(top)-1])
		}
		top[insert] = value
	}
	return top
}

func hipTopPackedScoresBytes(payload []byte, topK int) []uint64 {
	return hipTopPackedScoresBytesInto(payload, topK, nil)
}

func hipTopPackedScoresBytesInto(payload []byte, topK int, top []uint64) []uint64 {
	if topK <= 0 || len(payload) == 0 {
		return nil
	}
	top = top[:0]
	if cap(top) < min(topK, len(payload)/hipMLXQ4ProjectionBestBytes) {
		top = make([]uint64, 0, min(topK, len(payload)/hipMLXQ4ProjectionBestBytes))
	}
	for offset := 0; offset+hipMLXQ4ProjectionBestBytes <= len(payload); offset += hipMLXQ4ProjectionBestBytes {
		value := binary.LittleEndian.Uint64(payload[offset:])
		if value == 0 {
			continue
		}
		insert := len(top)
		for insert > 0 && value > top[insert-1] {
			insert--
		}
		if insert >= topK {
			continue
		}
		if len(top) < topK {
			top = append(top, 0)
			copy(top[insert+1:], top[insert:])
		} else {
			copy(top[insert+1:], top[insert:len(top)-1])
		}
		top[insert] = value
	}
	return top
}

func hipSortedPackedScoresBytesInto(payload []byte, topK int, top []uint64) []uint64 {
	if topK <= 0 || len(payload) == 0 {
		return nil
	}
	limit := min(topK, len(payload)/hipMLXQ4ProjectionBestBytes)
	top = top[:0]
	if cap(top) < limit {
		top = make([]uint64, 0, limit)
	}
	for offset := 0; offset+hipMLXQ4ProjectionBestBytes <= len(payload) && len(top) < limit; offset += hipMLXQ4ProjectionBestBytes {
		value := binary.LittleEndian.Uint64(payload[offset:])
		if value == 0 {
			continue
		}
		top = append(top, value)
	}
	return top
}

func hipMLXQ4ProjectionLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 projection row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4Proj,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4ProjectionLaunchConfigForShape(args []byte, rows, cols, groupSize, bits int) (hipKernelLaunchConfig, error) {
	if rows == 3840 && cols == 15360 && groupSize == 64 && hipMLXQ4ProjectionBitsOrDefault(bits) == 4 {
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 group64 12B down projection row16 blocks", (rows+hipMLXQ4ProjectionRow16RowsPerBlock-1)/hipMLXQ4ProjectionRow16RowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4ProjQ4G64Rows3840Cols15360Row16,
			Args:   args,
			GridX:  gridX,
			GridY:  1,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	if hipMLXQ4Projection12BDownRouteEnabled && rows == 3840 && cols == 15360 && groupSize == 32 && hipMLXQ4ProjectionBitsOrDefault(bits) == 4 {
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 group32 12B down projection row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4ProjQ4G32Rows3840Cols15360,
			Args:   args,
			GridX:  gridX,
			GridY:  1,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	if cols == 256 && hipMLXQ4ProjectionCols256Supported(groupSize, hipMLXQ4ProjectionBitsOrDefault(bits)) {
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 cols256 projection row blocks", (rows+hipMLXQ4ProjectionCols256RowsPerBlock-1)/hipMLXQ4ProjectionCols256RowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4ProjCols256,
			Args:   args,
			GridX:  gridX,
			GridY:  1,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	if cols > 2048 && groupSize == 16 && hipMLXQ4ProjectionBitsOrDefault(bits) == 6 {
		gridX, err := rocmDeviceKVPositiveUint32("MLX q6 group16 row16 projection row blocks", (rows+hipMLXQ4ProjectionQ6Row16RowsPerBlock-1)/hipMLXQ4ProjectionQ6Row16RowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4ProjQ6G16Row16,
			Args:   args,
			GridX:  gridX,
			GridY:  1,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	if cols >= 1536 && cols <= 2048 && groupSize == 64 && hipMLXQ4ProjectionBitsOrDefault(bits) == 6 {
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 q6 row64 projection row blocks", (rows+hipMLXQ4ProjectionQ6Row64RowsPerBlock-1)/hipMLXQ4ProjectionQ6Row64RowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4ProjQ6Row64,
			Args:   args,
			GridX:  gridX,
			GridY:  1,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	if cols > 2048 && groupSize == 64 && hipMLXQ4ProjectionBitsOrDefault(bits) == 6 {
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 q6 row16 projection row blocks", (rows+hipMLXQ4ProjectionQ6Row16RowsPerBlock-1)/hipMLXQ4ProjectionQ6Row16RowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4ProjQ6Row16,
			Args:   args,
			GridX:  gridX,
			GridY:  1,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	return hipMLXQ4ProjectionLaunchConfig(args, rows)
}

func hipMLXQ4ProjectionCols256Supported(groupSize, bits int) bool {
	if bits == 8 {
		return groupSize == 32
	}
	return (bits == 4 || bits == 6) && (groupSize == 32 || groupSize == 64)
}

func hipMLXQ4ProjectionScoresLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 projection score row blocks", (rows+hipMLXQ4ProjectionGreedyRowsPerBlock-1)/hipMLXQ4ProjectionGreedyRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4ProjScores,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4ProjectionScoresLaunchConfigForShape(args []byte, rows, cols, groupSize, bits int) (hipKernelLaunchConfig, error) {
	if cols >= 1536 && groupSize == 64 && hipMLXQ4ProjectionBitsOrDefault(bits) == 6 {
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 q6 row64 projection score row blocks", (rows+hipMLXQ4ProjectionGreedyQ6RowsPerBlock-1)/hipMLXQ4ProjectionGreedyQ6RowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4ProjScoresQ6Row64,
			Args:   args,
			GridX:  gridX,
			GridY:  1,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	return hipMLXQ4ProjectionScoresLaunchConfig(args, rows)
}

func hipPackedTopKLaunchConfig(args []byte, chunkCount int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("packed top-k chunks", chunkCount)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNamePackedTopK,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipPackedTopKBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipProjectionLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("projection rows", rows)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameProjection,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipProjectionBatchLaunchConfig(args []byte, rows, batch int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("projection batch row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	gridY, err := rocmDeviceKVPositiveUint32("projection batch token blocks", (batch+hipMLXQ4ProjectionBatchTokensPerBlock-1)/hipMLXQ4ProjectionBatchTokensPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameProjectionBatch,
		Args:   args,
		GridX:  gridX,
		GridY:  gridY,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4ProjectionBatchLaunchConfig(args []byte, rows, batch int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 projection batch row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	gridY, err := rocmDeviceKVPositiveUint32("MLX q4 projection batch token blocks", (batch+hipMLXQ4ProjectionBatchTokensPerBlock-1)/hipMLXQ4ProjectionBatchTokensPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4ProjBatch,
		Args:   args,
		GridX:  gridX,
		GridY:  gridY,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4ProjectionBatchLaunchConfigForShape(args []byte, rows, cols, groupSize, bits, batch int) (hipKernelLaunchConfig, error) {
	if cols >= 1536 && groupSize == 64 && hipMLXQ4ProjectionBitsOrDefault(bits) == 6 {
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 q6 row16 projection batch row blocks", (rows+hipMLXQ4ProjectionQ6Row16RowsPerBlock-1)/hipMLXQ4ProjectionQ6Row16RowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		gridY, err := rocmDeviceKVPositiveUint32("MLX q4 q6 projection batch token blocks", (batch+hipMLXQ4ProjectionBatchTokensPerBlock-1)/hipMLXQ4ProjectionBatchTokensPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4ProjBatchQ6Row16,
			Args:   args,
			GridX:  gridX,
			GridY:  gridY,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	return hipMLXQ4ProjectionBatchLaunchConfig(args, rows, batch)
}

func hipMLXQ4ProjectionGreedyLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 projection row blocks", (rows+hipMLXQ4ProjectionGreedyRowsPerBlock-1)/hipMLXQ4ProjectionGreedyRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4ProjGreedy,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4ProjectionGreedyLaunchConfigForShape(args []byte, rows, cols, groupSize, bits int) (hipKernelLaunchConfig, error) {
	if cols >= 1536 && groupSize == 64 && hipMLXQ4ProjectionBitsOrDefault(bits) == 6 {
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 q6 row64 projection greedy row blocks", (rows+hipMLXQ4ProjectionGreedyQ6RowsPerBlock-1)/hipMLXQ4ProjectionGreedyQ6RowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4ProjGreedyQ6Row64,
			Args:   args,
			GridX:  gridX,
			GridY:  1,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	config, err := hipMLXQ4ProjectionGreedyLaunchConfig(args, rows)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	if rows == 262144 && cols == 3840 && groupSize == 16 && hipMLXAffineSupportedBits(bits) && hipMLXQ4Projection12BHeadGridBlocks > 0 {
		gridBlocks := min(hipMLXQ4Projection12BHeadGridBlocks, int(config.GridX))
		config.GridX = uint32(gridBlocks)
	}
	return config, config.Validate()
}

func hipMLXQ4ProjectionSelectedGreedyLaunchConfig(args []byte, selectedCount int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 selected projection row blocks", (selectedCount+hipMLXQ4ProjectionGreedyRowsPerBlock-1)/hipMLXQ4ProjectionGreedyRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4ProjSelectedGreedy,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4ProjectionSelectedGreedyLaunchConfigForShape(args []byte, selectedCount, cols, groupSize, bits int) (hipKernelLaunchConfig, error) {
	if cols >= 1536 && groupSize == 64 && hipMLXQ4ProjectionBitsOrDefault(bits) == 6 {
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 q6 row64 selected projection row blocks", (selectedCount+hipMLXQ4ProjectionGreedyQ6RowsPerBlock-1)/hipMLXQ4ProjectionGreedyQ6RowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4ProjSelectedGreedyQ6Row64,
			Args:   args,
			GridX:  gridX,
			GridY:  1,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	return hipMLXQ4ProjectionSelectedGreedyLaunchConfig(args, selectedCount)
}

func hipOrderedEmbeddingCandidatesLaunchConfig(args []byte, outputCount int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("ordered embedding candidate blocks", (outputCount+int(hipOrderedEmbeddingCandidatesBlockSize)-1)/int(hipOrderedEmbeddingCandidatesBlockSize))
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameOrderedEmbeddingCandidates,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipOrderedEmbeddingCandidatesBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4ProjectionGreedyBatchLaunchConfig(args []byte, rows, batch int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 projection greedy batch row blocks", (rows+hipMLXQ4ProjectionGreedyRowsPerBlock-1)/hipMLXQ4ProjectionGreedyRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	gridY, err := rocmDeviceKVPositiveUint32("MLX q4 projection greedy batch rows", batch)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4ProjGreedyBatch,
		Args:   args,
		GridX:  gridX,
		GridY:  gridY,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4ProjectionGreedyBatchLaunchConfigForShape(args []byte, rows, cols, groupSize, bits, batch int) (hipKernelLaunchConfig, error) {
	if cols >= 1536 && groupSize == 64 && hipMLXQ4ProjectionBitsOrDefault(bits) == 6 {
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 q6 row64 projection greedy batch row blocks", (rows+hipMLXQ4ProjectionGreedyQ6RowsPerBlock-1)/hipMLXQ4ProjectionGreedyQ6RowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		gridY, err := rocmDeviceKVPositiveUint32("MLX q4 q6 projection greedy batch rows", batch)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4ProjGreedyBatchQ6Row64,
			Args:   args,
			GridX:  gridX,
			GridY:  gridY,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	return hipMLXQ4ProjectionGreedyBatchLaunchConfig(args, rows, batch)
}

func hipMLXQ4TripleProjectionLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 triple projection row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4TripleProj,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4PairProjectionLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 pair projection row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4PairProj,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4TripleProjectionLaunchConfigForShape(args []byte, rows, cols, groupSize, bits int) (hipKernelLaunchConfig, error) {
	if cols == 1536 && groupSize == 64 && hipMLXQ4ProjectionBitsOrDefault(bits) == 6 {
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 q6 row64 triple projection row blocks", (rows+hipMLXQ4ProjectionQ6Row64RowsPerBlock-1)/hipMLXQ4ProjectionQ6Row64RowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4TripleProjQ6Row64,
			Args:   args,
			GridX:  gridX,
			GridY:  1,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	if cols > 1536 && groupSize == 64 && hipMLXQ4ProjectionBitsOrDefault(bits) == 6 {
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 q6 row16 triple projection row blocks", (rows+hipMLXQ4ProjectionQ6Row16RowsPerBlock-1)/hipMLXQ4ProjectionQ6Row16RowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4TripleProjQ6Row16,
			Args:   args,
			GridX:  gridX,
			GridY:  1,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	return hipMLXQ4TripleProjectionLaunchConfig(args, rows)
}

func hipMLXQ4GELUTanhMultiplyLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 GELU tanh multiply row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4GELUTanhMul,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4GELUTanhMultiplyLaunchConfigForShape(args []byte, rows, cols, groupSize, bits int) (hipKernelLaunchConfig, error) {
	if rows == 15360 && cols == 3840 && groupSize == 64 && hipMLXQ4ProjectionBitsOrDefault(bits) == 4 {
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 group64 12B GELU tanh multiply row8 blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4GELUTanhMulQ4G64Rows15360Cols3840Row8,
			Args:   args,
			GridX:  gridX,
			GridY:  1,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	if hipMLXQ4GELUTanh12BGateUpRouteEnabled && rows == 15360 && cols == 3840 && groupSize == 32 && hipMLXQ4ProjectionBitsOrDefault(bits) == 4 {
		rowsPerBlock := hipMLXQ4GELUTanhQ4G32Cols1536Row16RowsPerBlock
		name := hipKernelNameMLXQ4GELUTanhMulQ4G32Rows15360Cols3840
		if hipMLXQ4GELUTanh12BGateUpGeometry == "row8" {
			rowsPerBlock = hipMLXQ4GELUTanh12BRow8RowsPerBlock
			name = hipKernelNameMLXQ4GELUTanhMulQ4G32Rows15360Cols3840Row8
		}
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 group32 12B GELU tanh multiply row blocks", (rows+rowsPerBlock-1)/rowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   name,
			Args:   args,
			GridX:  gridX,
			GridY:  1,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	if cols >= 1536 && groupSize == 32 && hipMLXQ4ProjectionBitsOrDefault(bits) == 4 {
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 GELU tanh multiply q4 group32 cols1536 row16 row blocks", (rows+hipMLXQ4GELUTanhQ4G32Cols1536Row16RowsPerBlock-1)/hipMLXQ4GELUTanhQ4G32Cols1536Row16RowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4GELUTanhMulQ4G32Cols1536Row16,
			Args:   args,
			GridX:  gridX,
			GridY:  1,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	if cols == 1536 && groupSize == 64 && hipMLXQ4ProjectionBitsOrDefault(bits) == 6 {
		if rows <= 6144 {
			gridX, err := rocmDeviceKVPositiveUint32("MLX q4 GELU tanh multiply q6 cols1536 row64 row blocks", (rows+hipMLXQ4GELUTanhQ6Cols1536Row64RowsPerBlock-1)/hipMLXQ4GELUTanhQ6Cols1536Row64RowsPerBlock)
			if err != nil {
				return hipKernelLaunchConfig{}, err
			}
			config := hipKernelLaunchConfig{
				Name:   hipKernelNameMLXQ4GELUTanhMulQ6Cols1536Row64,
				Args:   args,
				GridX:  gridX,
				GridY:  1,
				GridZ:  1,
				BlockX: hipMLXQ4ProjectionBlockSize,
				BlockY: 1,
				BlockZ: 1,
			}
			return config, config.Validate()
		}
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 GELU tanh multiply q6 cols1536 row blocks", (rows+hipMLXQ4GELUTanhQ6Cols1536RowsPerBlock-1)/hipMLXQ4GELUTanhQ6Cols1536RowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4GELUTanhMulQ6Cols1536,
			Args:   args,
			GridX:  gridX,
			GridY:  1,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	return hipMLXQ4GELUTanhMultiplyLaunchConfig(args, rows)
}

func hipMLXQ4GELUTanhMLPPersistentLaunchConfig(args []byte) (hipKernelLaunchConfig, error) {
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4GELUTanhMLPQ4G32Cols1536Persistent,
		Args:   args,
		GridX:  hipMLXQ4GELUTanhMLPPersistentBlocks,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4GELUTanhMultiplyBatchLaunchConfig(args []byte, rows, batch int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 GELU tanh multiply batch row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	gridY, err := rocmDeviceKVPositiveUint32("MLX q4 GELU tanh multiply batch token blocks", (batch+hipMLXQ4ProjectionBatchTokensPerBlock-1)/hipMLXQ4ProjectionBatchTokensPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4GELUTanhMulBatch,
		Args:   args,
		GridX:  gridX,
		GridY:  gridY,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4GELUTanhProjectionLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 GELU tanh projection row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4GELUTanhProj,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4GELUTanhProjectionLaunchConfigForShape(args []byte, rows, cols, groupSize, bits int) (hipKernelLaunchConfig, error) {
	if cols >= 1536 && groupSize == 64 && hipMLXQ4ProjectionBitsOrDefault(bits) == 6 {
		gridX, err := rocmDeviceKVPositiveUint32("MLX q4 q6 row16 GELU tanh projection row blocks", (rows+hipMLXQ4ProjectionQ6Row16RowsPerBlock-1)/hipMLXQ4ProjectionQ6Row16RowsPerBlock)
		if err != nil {
			return hipKernelLaunchConfig{}, err
		}
		config := hipKernelLaunchConfig{
			Name:   hipKernelNameMLXQ4GELUTanhProjQ6Row16,
			Args:   args,
			GridX:  gridX,
			GridY:  1,
			GridZ:  1,
			BlockX: hipMLXQ4ProjectionBlockSize,
			BlockY: 1,
			BlockZ: 1,
		}
		return config, config.Validate()
	}
	return hipMLXQ4GELUTanhProjectionLaunchConfig(args, rows)
}

func hipMLXQ4GELUTanhProjectionBatchLaunchConfig(args []byte, rows, batch int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 GELU tanh projection batch row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	gridY, err := rocmDeviceKVPositiveUint32("MLX q4 GELU tanh projection batch token blocks", (batch+hipMLXQ4ProjectionBatchTokensPerBlock-1)/hipMLXQ4ProjectionBatchTokensPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4GELUTanhProjBatch,
		Args:   args,
		GridX:  gridX,
		GridY:  gridY,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipRMSResidualAddGELUTanhProjectionLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("RMS residual-add GELU tanh projection row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameRMSNormResidualAddGELUTanhProj,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipUploadByteBuffer(driver nativeHIPDriver, operation, label string, payload []byte, count int) (*hipDeviceByteBuffer, error) {
	if len(payload) == 0 {
		return nil, core.E(operation, label+" payload is empty", nil)
	}
	buffer, err := hipAllocateByteBuffer(driver, operation, label, uint64(len(payload)), count)
	if err != nil {
		return nil, err
	}
	if err := hipCopyHostToDeviceLabeled(driver, buffer.pointer, payload, operation, label); err != nil {
		_ = buffer.Close()
		return nil, core.E(operation, "copy "+label, err)
	}
	return buffer, nil
}

func hipAllocateByteBuffer(driver nativeHIPDriver, operation, label string, sizeBytes uint64, count int) (*hipDeviceByteBuffer, error) {
	buffer, err := hipAllocateByteBufferValue(driver, operation, label, sizeBytes, count)
	if err != nil {
		return nil, err
	}
	return &buffer, nil
}

func hipAllocateByteBufferValue(driver nativeHIPDriver, operation, label string, sizeBytes uint64, count int) (hipDeviceByteBuffer, error) {
	if driver == nil {
		return hipDeviceByteBuffer{}, core.E(operation, "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return hipDeviceByteBuffer{}, core.E(operation, "HIP driver is not available", nil)
	}
	if sizeBytes == 0 || count <= 0 {
		return hipDeviceByteBuffer{}, core.E(operation, label+" size must be positive", nil)
	}
	if pointer, ok := hipDeviceByteBufferPoolTake(driver, sizeBytes); ok {
		return hipDeviceByteBuffer{
			driver:    driver,
			pointer:   pointer,
			count:     count,
			sizeBytes: sizeBytes,
			pooled:    true,
			label:     label,
		}, nil
	}
	pointer, err := hipMallocLabeled(driver, operation, label, sizeBytes)
	if err != nil {
		return hipDeviceByteBuffer{}, core.E(operation, "allocate "+label, err)
	}
	return hipDeviceByteBuffer{
		driver:    driver,
		pointer:   pointer,
		count:     count,
		sizeBytes: sizeBytes,
		pooled:    hipDeviceByteBufferPoolEnabled(),
		label:     label,
	}, nil
}

func hipMallocLabeled(driver nativeHIPDriver, operation, label string, sizeBytes uint64) (nativeDevicePointer, error) {
	pointer, err := driver.Malloc(sizeBytes)
	if err != nil {
		return 0, err
	}
	hipRecordDeviceAllocationLabel(driver, sizeBytes, operation, label)
	return pointer, nil
}

func hipRecordDeviceAllocationLabel(driver nativeHIPDriver, sizeBytes uint64, operation, label string) {
	if driver == nil || sizeBytes == 0 {
		return
	}
	recorder, ok := driver.(hipDeviceAllocationLabelRecorder)
	if !ok {
		return
	}
	recorder.RecordDeviceAllocationLabel(sizeBytes, operation, label)
}

func hipDeviceByteBufferPoolEnabled() bool {
	return os.Getenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL") != "1" && hipDeviceByteBufferPoolSuppressions.Load() == 0
}

func hipSuppressDeviceByteBufferPool() func() {
	hipDeviceByteBufferPoolSuppressions.Add(1)
	hipDrainDeviceByteBufferPool()
	var once sync.Once
	return func() {
		once.Do(func() {
			hipDeviceByteBufferPoolSuppressions.Add(-1)
		})
	}
}

func hipDrainDeviceByteBufferPool() {
	hipDeviceByteBufferPool.Lock()
	entries := make([]hipDeviceByteBufferPoolEntry, 0)
	for index := range hipDeviceByteBufferPool.single {
		slot := &hipDeviceByteBufferPool.single[index]
		for entryIndex := 0; entryIndex < int(slot.count); entryIndex++ {
			if entry := slot.entries[entryIndex]; entry.driver != nil && entry.pointer != 0 {
				entries = append(entries, entry)
			}
		}
	}
	for _, bucket := range hipDeviceByteBufferPool.entries {
		for _, entry := range bucket {
			if entry.driver != nil && entry.pointer != 0 {
				entries = append(entries, entry)
			}
		}
	}
	hipDeviceByteBufferPool.single = [hipDeviceByteBufferPoolSingleSlots]hipDeviceByteBufferPoolSingleSlot{}
	hipDeviceByteBufferPool.entries = make(map[uint64][]hipDeviceByteBufferPoolEntry)
	hipDeviceByteBufferPool.bytes = 0
	hipDeviceByteBufferPool.Unlock()
	for _, entry := range entries {
		_ = entry.driver.Free(entry.pointer)
	}
}

func hipPrewarmDeviceByteBufferPool(driver nativeHIPDriver, sizeBytes uint64, count int) {
	if driver == nil || !driver.Available() || sizeBytes == 0 || count <= 0 || !hipDeviceByteBufferPoolEnabled() {
		return
	}
	for i := 0; i < count; i++ {
		pointer, err := hipMallocLabeled(driver, "rocm.hip.DeviceByteBufferPool", "prewarm device byte buffer", sizeBytes)
		if err != nil {
			return
		}
		if !hipDeviceByteBufferPoolPut(driver, pointer, sizeBytes) {
			_ = driver.Free(pointer)
			return
		}
	}
}

func hipDeviceByteBufferPoolTake(driver nativeHIPDriver, sizeBytes uint64) (nativeDevicePointer, bool) {
	if !hipDeviceByteBufferPoolEnabled() {
		return 0, false
	}
	hipDeviceByteBufferPool.Lock()
	defer hipDeviceByteBufferPool.Unlock()
	for index := range hipDeviceByteBufferPool.single {
		slot := &hipDeviceByteBufferPool.single[index]
		if slot.sizeBytes != sizeBytes || slot.count == 0 {
			continue
		}
		for entryIndex := int(slot.count) - 1; entryIndex >= 0; entryIndex-- {
			entry := slot.entries[entryIndex]
			if entry.pointer == 0 || entry.driver != driver {
				continue
			}
			pointer := entry.pointer
			lastIndex := int(slot.count) - 1
			slot.entries[entryIndex] = slot.entries[lastIndex]
			slot.entries[lastIndex] = hipDeviceByteBufferPoolEntry{}
			slot.count--
			if slot.count == 0 {
				*slot = hipDeviceByteBufferPoolSingleSlot{}
			}
			if hipDeviceByteBufferPool.bytes >= sizeBytes {
				hipDeviceByteBufferPool.bytes -= sizeBytes
			} else {
				hipDeviceByteBufferPool.bytes = 0
			}
			return pointer, true
		}
	}
	entries := hipDeviceByteBufferPool.entries[sizeBytes]
	for index := len(entries) - 1; index >= 0; index-- {
		entry := entries[index]
		if entry.driver != driver {
			continue
		}
		pointer := entry.pointer
		entries[index] = entries[len(entries)-1]
		entries[len(entries)-1] = hipDeviceByteBufferPoolEntry{}
		entries = entries[:len(entries)-1]
		if hipDeviceByteBufferPool.bytes >= sizeBytes {
			hipDeviceByteBufferPool.bytes -= sizeBytes
		} else {
			hipDeviceByteBufferPool.bytes = 0
		}
		hipDeviceByteBufferPool.entries[sizeBytes] = entries
		return pointer, true
	}
	return 0, false
}

func hipDeviceByteBufferPoolPut(driver nativeHIPDriver, pointer nativeDevicePointer, sizeBytes uint64) bool {
	if !hipDeviceByteBufferPoolEnabled() || driver == nil || pointer == 0 || sizeBytes == 0 {
		return false
	}
	hipDeviceByteBufferPool.Lock()
	defer hipDeviceByteBufferPool.Unlock()
	if !hipDeviceByteBufferPoolEnabled() {
		return false
	}
	if hipDeviceByteBufferPool.bytes+sizeBytes > hipDeviceByteBufferPoolMaxBytes {
		return false
	}
	emptySingle := -1
	for index := range hipDeviceByteBufferPool.single {
		slot := &hipDeviceByteBufferPool.single[index]
		if slot.count == 0 {
			if emptySingle < 0 {
				emptySingle = index
			}
			continue
		}
		if slot.sizeBytes == sizeBytes {
			if int(slot.count) < len(slot.entries) {
				slot.entries[slot.count] = hipDeviceByteBufferPoolEntry{driver: driver, pointer: pointer}
				slot.count++
				hipDeviceByteBufferPool.bytes += sizeBytes
				return true
			}
			emptySingle = -1
			break
		}
	}
	if emptySingle >= 0 {
		hipDeviceByteBufferPool.single[emptySingle] = hipDeviceByteBufferPoolSingleSlot{
			sizeBytes: sizeBytes,
			entries:   [hipDeviceByteBufferPoolSingleSlotCapacity]hipDeviceByteBufferPoolEntry{{driver: driver, pointer: pointer}},
			count:     1,
		}
		hipDeviceByteBufferPool.bytes += sizeBytes
		return true
	}
	entries := hipDeviceByteBufferPool.entries[sizeBytes]
	if len(entries) >= hipDeviceByteBufferPoolMaxPerSize {
		return false
	}
	hipDeviceByteBufferPool.entries[sizeBytes] = append(entries, hipDeviceByteBufferPoolEntry{driver: driver, pointer: pointer})
	hipDeviceByteBufferPool.bytes += sizeBytes
	return true
}

func hipBorrowDeviceByteBuffer(driver nativeHIPDriver, label string, pointer nativeDevicePointer, sizeBytes uint64, count int) *hipDeviceByteBuffer {
	buffer := hipBorrowDeviceByteBufferValue(driver, label, pointer, sizeBytes, count)
	return &buffer
}

func hipBorrowDeviceByteBufferValue(driver nativeHIPDriver, label string, pointer nativeDevicePointer, sizeBytes uint64, count int) hipDeviceByteBuffer {
	return hipDeviceByteBuffer{
		driver:    driver,
		pointer:   pointer,
		count:     count,
		sizeBytes: sizeBytes,
		borrowed:  true,
		label:     label,
	}
}

func (buffer *hipDeviceByteBuffer) Pointer() nativeDevicePointer {
	if buffer == nil || buffer.closed {
		return 0
	}
	return buffer.pointer
}

func (buffer *hipDeviceByteBuffer) Count() int {
	if buffer == nil || buffer.closed {
		return 0
	}
	return buffer.count
}

func (buffer *hipDeviceByteBuffer) SizeBytes() uint64 {
	if buffer == nil || buffer.closed {
		return 0
	}
	return buffer.sizeBytes
}

func (buffer *hipDeviceByteBuffer) Close() error {
	if buffer == nil || buffer.closed {
		return nil
	}
	if buffer.pointer != 0 && !buffer.borrowed {
		if buffer.driver == nil {
			return core.E("rocm.hip.ProjectionLaunch", "HIP driver is nil", nil)
		}
		if buffer.pooled && hipDeviceByteBufferPoolPut(buffer.driver, buffer.pointer, buffer.sizeBytes) {
			buffer.pointer = 0
			buffer.closed = true
			return nil
		}
		if err := buffer.driver.Free(buffer.pointer); err != nil {
			return core.E("rocm.hip.ProjectionLaunch", "free "+firstNonEmptyString(buffer.label, "device buffer"), err)
		}
		buffer.pointer = 0
	}
	buffer.closed = true
	return nil
}

func (buffers *hipProjectionDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Bias, buffers.Weights, buffers.Input} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipProjectionDeviceBuffers) ReadOutput() ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "projection output buffer is required", nil)
	}
	if buffers.Rows <= 0 || buffers.Output.Count() != buffers.Rows || buffers.Output.SizeBytes() != uint64(buffers.Rows*4) {
		return nil, core.E("rocm.hip.ProjectionLaunch", "projection output byte count mismatch", nil)
	}
	payload := make([]byte, buffers.Output.SizeBytes())
	if err := buffers.Output.driver.CopyDeviceToHost(buffers.Output.Pointer(), payload); err != nil {
		return nil, core.E("rocm.hip.ProjectionLaunch", "copy projection output", err)
	}
	values, err := hipFloat32PayloadValues(payload)
	if err != nil {
		return nil, err
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E("rocm.hip.ProjectionLaunch", "projection output values must be finite", nil)
	}
	return values, nil
}

func hipFloat32Payload(values []float32) ([]byte, error) {
	if len(values) == 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "float32 payload is empty", nil)
	}
	payload := make([]byte, len(values)*4)
	return hipFloat32PayloadInto(payload, values)
}

func hipFloat32PayloadInto(payload []byte, values []float32) ([]byte, error) {
	if len(values) == 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "float32 payload is empty", nil)
	}
	if len(payload) < len(values)*4 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "float32 payload buffer is too small", nil)
	}
	payload = payload[:len(values)*4]
	for index, value := range values {
		binary.LittleEndian.PutUint32(payload[index*4:], math.Float32bits(value))
	}
	return payload, nil
}

func hipUint16Payload(values []uint16) ([]byte, error) {
	if len(values) == 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "uint16 payload is empty", nil)
	}
	payload := make([]byte, len(values)*2)
	for index, value := range values {
		binary.LittleEndian.PutUint16(payload[index*2:], value)
	}
	return payload, nil
}

func hipUint32Payload(values []uint32) ([]byte, error) {
	if len(values) == 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "uint32 payload is empty", nil)
	}
	payload := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(payload[index*4:], value)
	}
	return payload, nil
}

func hipInt8Payload(values []int8) []byte {
	payload := make([]byte, len(values))
	for index, value := range values {
		payload[index] = byte(value)
	}
	return payload
}

func (buffers *hipMLXQ4ProjectionDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Biases, buffers.Scales, buffers.Weight, buffers.Input} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipMLXQ4ProjectionDeviceBuffers) ReadOutput() ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection output buffer is required", nil)
	}
	if buffers.Rows <= 0 || buffers.Output.Count() != buffers.Rows || buffers.Output.SizeBytes() != uint64(buffers.Rows*4) {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection output byte count mismatch", nil)
	}
	payload := make([]byte, buffers.Output.SizeBytes())
	if err := buffers.Output.driver.CopyDeviceToHost(buffers.Output.Pointer(), payload); err != nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "copy MLX q4 projection output", err)
	}
	values, err := hipFloat32PayloadValues(payload)
	if err != nil {
		return nil, err
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection output values must be finite", nil)
	}
	return values, nil
}

func hipFloat32PayloadValues(payload []byte) ([]float32, error) {
	if len(payload) == 0 || len(payload)%4 != 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "float32 payload byte length must be positive and aligned", nil)
	}
	values := make([]float32, len(payload)/4)
	return hipFloat32PayloadValuesInto(values, payload)
}

func hipFloat32PayloadValuesInto(values []float32, payload []byte) ([]float32, error) {
	if len(payload) == 0 || len(payload)%4 != 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "float32 payload byte length must be positive and aligned", nil)
	}
	count := len(payload) / 4
	if len(values) < count {
		return nil, core.E("rocm.hip.ProjectionLaunch", "float32 output buffer is too small", nil)
	}
	values = values[:count]
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(payload[index*4:]))
	}
	return values, nil
}
