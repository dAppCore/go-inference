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
	hipEmbeddingMeanPoolLaunchArgsVersion    uint32 = 1
	hipEmbeddingMeanPoolLaunchArgsBytes             = 64
	hipEmbeddingLookupLaunchArgsVersion      uint32 = 1
	hipEmbeddingLookupLaunchArgsBytes               = 104
	hipDiffusionExpectedEmbeddingArgsVersion uint32 = 1
	hipDiffusionExpectedEmbeddingArgsBytes          = 120
	hipRerankCosineLaunchArgsVersion         uint32 = 1
	hipRerankCosineLaunchArgsBytes                  = 64
)

const hipEmbeddingMeanPoolLaunchFlagNormalize uint32 = 1

const (
	hipEmbeddingTableEncodingF32   uint32 = 1
	hipEmbeddingTableEncodingBF16  uint32 = 2
	hipEmbeddingTableEncodingMLXQ4 uint32 = 3
)

const (
	hipDiffusionExpectedEmbeddingAffineG64RowsPerBlock = 16
	hipDiffusionExpectedEmbeddingQ8G64DimsPerThread    = 4
	hipDiffusionExpectedEmbeddingQ8G64RowsPerBlock     = 4
	hipDiffusionExpectedEmbeddingQ8G64SubgroupWidth    = 32
	hipDiffusionExpectedEmbeddingQ8G64RowsPerSubgroup  = 8
	hipDiffusionExpectedEmbeddingQ8G64Subgroups        = 8
	hipDiffusionExpectedEmbeddingQ8G64SubgroupRows     = hipDiffusionExpectedEmbeddingQ8G64RowsPerSubgroup * hipDiffusionExpectedEmbeddingQ8G64Subgroups
	hipDiffusionExpectedEmbeddingQ8G64SubgroupDims     = hipDiffusionExpectedEmbeddingQ8G64SubgroupWidth * hipDiffusionExpectedEmbeddingQ8G64DimsPerThread
	hipDiffusionExpectedEmbeddingQ8G64SubgroupMinRows  = 64
	hipDiffusionExpectedEmbeddingQ8G64TileRows         = 32
	hipDiffusionExpectedEmbeddingQ8G64TileDims         = 64
)

const (
	hipDisableDiffusionExpectedEmbeddingProbability4Env = "GO_ROCM_DISABLE_DIFFUSION_EXPECTED_EMBEDDING_PROBABILITY4"
	hipDisableDiffusionExpectedEmbeddingSubgroupEnv     = "GO_ROCM_DISABLE_DIFFUSION_EXPECTED_EMBEDDING_SUBGROUP"
	hipDisableDiffusionExpectedEmbeddingTileEnv         = "GO_ROCM_DISABLE_DIFFUSION_EXPECTED_EMBEDDING_TILE"
)

type hipEmbeddingMeanPoolRequest struct {
	Tokens     []float32
	TokenCount int
	Dim        int
	Normalize  bool
}

type hipEmbeddingMeanPoolDeviceBuffers struct {
	Tokens     *hipDeviceByteBuffer
	Output     *hipDeviceByteBuffer
	TokenCount int
	Dim        int
}

type hipEmbeddingMeanPoolLaunchArgs struct {
	TokenPointer  nativeDevicePointer
	OutputPointer nativeDevicePointer
	TokenCount    int
	Dim           int
	TokenBytes    uint64
	OutputBytes   uint64
	Flags         uint32
}

type hipEmbeddingLookupRequest struct {
	TokenIDs      []int32
	EmbeddingF32  []float32
	EmbeddingBF16 []uint16
	EmbeddingQ4   []uint32
	Q4Scales      []uint16
	Q4Biases      []uint16
	Q4GroupSize   int
	QuantBits     int
	VocabSize     int
	HiddenSize    int
}

type hipEmbeddingLookupDeviceBuffers struct {
	Tokens      *hipDeviceTokenBuffer
	Embedding   *hipDeviceByteBuffer
	Scales      *hipDeviceByteBuffer
	Biases      *hipDeviceByteBuffer
	Output      *hipDeviceByteBuffer
	TokenCount  int
	VocabSize   int
	HiddenSize  int
	GroupSize   int
	QuantBits   int
	TableEncode uint32
}

type hipEmbeddingLookupLaunchArgs struct {
	TokenPointer     nativeDevicePointer
	EmbeddingPointer nativeDevicePointer
	OutputPointer    nativeDevicePointer
	TokenCount       int
	VocabSize        int
	HiddenSize       int
	TokenBytes       uint64
	EmbeddingBytes   uint64
	OutputBytes      uint64
	TableEncoding    uint32
	GroupSize        int
	ScalePointer     nativeDevicePointer
	BiasPointer      nativeDevicePointer
	ScaleBytes       uint64
	BiasBytes        uint64
	OutputScale      float32
	QuantBits        int
}

type hipDeviceEmbeddingLookupConfig struct {
	EmbeddingPointer nativeDevicePointer
	EmbeddingBytes   uint64
	TableEncoding    uint32
	VocabSize        int
	HiddenSize       int
	GroupSize        int
	ScalePointer     nativeDevicePointer
	BiasPointer      nativeDevicePointer
	ScaleBytes       uint64
	BiasBytes        uint64
	QuantBits        int
}

type hipDiffusionExpectedEmbeddingLaunchArgs struct {
	ProbabilityPointer nativeDevicePointer
	EmbeddingPointer   nativeDevicePointer
	ScalePointer       nativeDevicePointer
	BiasPointer        nativeDevicePointer
	OutputPointer      nativeDevicePointer
	Rows               int
	VocabSize          int
	HiddenSize         int
	TableEncoding      uint32
	GroupSize          int
	QuantBits          int
	ProbabilityBytes   uint64
	EmbeddingBytes     uint64
	ScaleBytes         uint64
	BiasBytes          uint64
	OutputBytes        uint64
	OutputScale        float32
}

func (cfg hipDeviceEmbeddingLookupConfig) validate(tokenIDs []int32) error {
	if len(tokenIDs) == 0 {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "token IDs are required", nil)
	}
	if err := cfg.validateShape(); err != nil {
		return err
	}
	for _, id := range tokenIDs {
		if id < 0 || int(id) >= cfg.VocabSize {
			return core.E("rocm.hip.EmbeddingLookupLaunch", "token ID is outside vocabulary", nil)
		}
	}
	return nil
}

func (cfg hipDeviceEmbeddingLookupConfig) validateSingleToken(tokenID int32) error {
	if err := cfg.validateShape(); err != nil {
		return err
	}
	if tokenID < 0 || int(tokenID) >= cfg.VocabSize {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "token ID is outside vocabulary", nil)
	}
	return nil
}

func (cfg hipDeviceEmbeddingLookupConfig) validateShape() error {
	if cfg.VocabSize <= 0 || cfg.HiddenSize <= 0 {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "vocab and hidden sizes must be positive", nil)
	}
	if cfg.EmbeddingPointer == 0 {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "embedding pointer is required", nil)
	}
	tableCount := uint64(cfg.VocabSize) * uint64(cfg.HiddenSize)
	switch cfg.TableEncoding {
	case hipEmbeddingTableEncodingF32:
		if cfg.EmbeddingBytes != tableCount*4 {
			return core.E("rocm.hip.EmbeddingLookupLaunch", "f32 embedding byte count mismatch", nil)
		}
	case hipEmbeddingTableEncodingBF16:
		if cfg.EmbeddingBytes != tableCount*2 {
			return core.E("rocm.hip.EmbeddingLookupLaunch", "bf16 embedding byte count mismatch", nil)
		}
	case hipEmbeddingTableEncodingMLXQ4:
		if cfg.ScalePointer == 0 || cfg.BiasPointer == 0 {
			return core.E("rocm.hip.EmbeddingLookupLaunch", "q4 scale and bias pointers are required", nil)
		}
		bits := hipMLXQ4ProjectionBitsOrDefault(cfg.QuantBits)
		packedPerRow, err := hipMLXAffinePackedCols(cfg.HiddenSize, bits)
		if err != nil {
			return err
		}
		if cfg.GroupSize <= 0 || cfg.HiddenSize%cfg.GroupSize != 0 {
			return core.E("rocm.hip.EmbeddingLookupLaunch", "hidden size must align with MLX affine group size", nil)
		}
		weightBytes := uint64(cfg.VocabSize) * uint64(packedPerRow) * 4
		if cfg.EmbeddingBytes != weightBytes {
			return core.E("rocm.hip.EmbeddingLookupLaunch", "MLX affine embedding byte count mismatch", nil)
		}
		groupBytes := uint64(cfg.VocabSize) * uint64(cfg.HiddenSize/cfg.GroupSize) * 2
		if cfg.ScaleBytes != groupBytes || cfg.BiasBytes != groupBytes {
			return core.E("rocm.hip.EmbeddingLookupLaunch", "MLX affine scale/bias byte count mismatch", nil)
		}
	default:
		return core.E("rocm.hip.EmbeddingLookupLaunch", core.Sprintf("unsupported embedding table encoding %d", cfg.TableEncoding), nil)
	}
	return nil
}

type hipRerankCosineRequest struct {
	Query         []float32
	Documents     []float32
	DocumentCount int
	Dim           int
}

type hipRerankCosineDeviceBuffers struct {
	Query         *hipDeviceByteBuffer
	Documents     *hipDeviceByteBuffer
	Output        *hipDeviceByteBuffer
	DocumentCount int
	Dim           int
}

type hipRerankCosineLaunchArgs struct {
	QueryPointer    nativeDevicePointer
	DocumentPointer nativeDevicePointer
	OutputPointer   nativeDevicePointer
	DocumentCount   int
	Dim             int
	QueryBytes      uint64
	DocumentBytes   uint64
	OutputBytes     uint64
}

func (req hipEmbeddingMeanPoolRequest) validate() error {
	if req.TokenCount <= 0 || req.Dim <= 0 {
		return core.E("rocm.hip.EmbeddingMeanPoolLaunch", "token count and dimension must be positive", nil)
	}
	if len(req.Tokens) != req.TokenCount*req.Dim {
		return core.E("rocm.hip.EmbeddingMeanPoolLaunch", "token embedding length must match token_count*dim", nil)
	}
	if _, err := rocmReferenceMeanPoolEmbedding(splitFloat32Vectors(req.Tokens, req.Dim), req.Normalize); err != nil {
		return err
	}
	return nil
}

func (req hipEmbeddingMeanPoolRequest) deviceBuffers(driver nativeHIPDriver) (*hipEmbeddingMeanPoolDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	tokenPayload, err := hipFloat32Payload(req.Tokens)
	if err != nil {
		return nil, core.E("rocm.hip.EmbeddingMeanPoolLaunch", "encode token embeddings", err)
	}
	tokens, err := hipUploadByteBuffer(driver, "rocm.hip.EmbeddingMeanPoolLaunch", "embedding tokens", tokenPayload, len(req.Tokens))
	if err != nil {
		return nil, err
	}
	buffers := &hipEmbeddingMeanPoolDeviceBuffers{Tokens: tokens, TokenCount: req.TokenCount, Dim: req.Dim}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.EmbeddingMeanPoolLaunch", "embedding output", uint64(req.Dim*4), req.Dim)
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipEmbeddingMeanPoolRequest) launchArgs(buffers *hipEmbeddingMeanPoolDeviceBuffers) (hipEmbeddingMeanPoolLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipEmbeddingMeanPoolLaunchArgs{}, err
	}
	if buffers == nil || buffers.Tokens == nil || buffers.Output == nil {
		return hipEmbeddingMeanPoolLaunchArgs{}, core.E("rocm.hip.EmbeddingMeanPoolLaunch", "embedding device buffers are required", nil)
	}
	if buffers.Tokens.Count() != req.TokenCount*req.Dim || buffers.Output.Count() != req.Dim ||
		buffers.TokenCount != req.TokenCount || buffers.Dim != req.Dim {
		return hipEmbeddingMeanPoolLaunchArgs{}, core.E("rocm.hip.EmbeddingMeanPoolLaunch", "embedding device buffer shape mismatch", nil)
	}
	var flags uint32
	if req.Normalize {
		flags |= hipEmbeddingMeanPoolLaunchFlagNormalize
	}
	return hipEmbeddingMeanPoolLaunchArgs{
		TokenPointer:  buffers.Tokens.Pointer(),
		OutputPointer: buffers.Output.Pointer(),
		TokenCount:    req.TokenCount,
		Dim:           req.Dim,
		TokenBytes:    buffers.Tokens.SizeBytes(),
		OutputBytes:   buffers.Output.SizeBytes(),
		Flags:         flags,
	}, nil
}

func (args hipEmbeddingMeanPoolLaunchArgs) Binary() ([]byte, error) {
	if args.TokenPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.EmbeddingMeanPoolLaunch", "token and output pointers are required", nil)
	}
	tokenCount, err := rocmDeviceKVPositiveUint32("token count", args.TokenCount)
	if err != nil {
		return nil, err
	}
	dim, err := rocmDeviceKVPositiveUint32("dimension", args.Dim)
	if err != nil {
		return nil, err
	}
	tokenEntries, err := hipUint32Product("embedding token count", tokenCount, dim)
	if err != nil {
		return nil, err
	}
	tokenBytes, err := hipAlignedFloat32Bytes("embedding tokens", args.TokenBytes, tokenEntries)
	if err != nil {
		return nil, core.E("rocm.hip.EmbeddingMeanPoolLaunch", "token byte count", err)
	}
	outputBytes, err := hipAlignedFloat32Bytes("embedding output", args.OutputBytes, dim)
	if err != nil {
		return nil, core.E("rocm.hip.EmbeddingMeanPoolLaunch", "output byte count", err)
	}
	payload := hipBorrowLaunchPacket(hipEmbeddingMeanPoolLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipEmbeddingMeanPoolLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.TokenPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[24:], tokenCount)
	binary.LittleEndian.PutUint32(payload[28:], dim)
	binary.LittleEndian.PutUint32(payload[32:], tokenBytes)
	binary.LittleEndian.PutUint32(payload[36:], outputBytes)
	binary.LittleEndian.PutUint32(payload[40:], args.Flags)
	return payload, nil
}

func (buffers *hipEmbeddingMeanPoolDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Tokens} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipEmbeddingMeanPoolDeviceBuffers) ReadOutput() ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.EmbeddingMeanPoolLaunch", "embedding output buffer is required", nil)
	}
	if buffers.Dim <= 0 || buffers.Output.Count() != buffers.Dim || buffers.Output.SizeBytes() != uint64(buffers.Dim*4) {
		return nil, core.E("rocm.hip.EmbeddingMeanPoolLaunch", "embedding output byte count mismatch", nil)
	}
	payload := make([]byte, buffers.Output.SizeBytes())
	if err := buffers.Output.driver.CopyDeviceToHost(buffers.Output.Pointer(), payload); err != nil {
		return nil, core.E("rocm.hip.EmbeddingMeanPoolLaunch", "copy embedding output", err)
	}
	values, err := hipFloat32PayloadValues(payload)
	if err != nil {
		return nil, err
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E("rocm.hip.EmbeddingMeanPoolLaunch", "embedding output values must be finite", nil)
	}
	return values, nil
}

func hipRunEmbeddingMeanPoolKernel(ctx context.Context, driver nativeHIPDriver, req hipEmbeddingMeanPoolRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameEmbedMean, launchBytes, 1)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func (req hipEmbeddingLookupRequest) validate() error {
	if req.VocabSize <= 0 || req.HiddenSize <= 0 {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "vocabulary and hidden size must be positive", nil)
	}
	if len(req.TokenIDs) == 0 {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "token IDs are required", nil)
	}
	for _, id := range req.TokenIDs {
		if id < 0 || int(id) >= req.VocabSize {
			return core.E("rocm.hip.EmbeddingLookupLaunch", "token ID is outside vocabulary", nil)
		}
	}
	tableCount := req.VocabSize * req.HiddenSize
	encodings := 0
	if len(req.EmbeddingF32) > 0 {
		encodings++
	}
	if len(req.EmbeddingBF16) > 0 {
		encodings++
	}
	if len(req.EmbeddingQ4) > 0 || len(req.Q4Scales) > 0 || len(req.Q4Biases) > 0 {
		encodings++
	}
	if encodings != 1 {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "exactly one embedding table encoding is required", nil)
	}
	if len(req.EmbeddingF32) > 0 && len(req.EmbeddingF32) != tableCount {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "f32 embedding table length must match vocab*hidden", nil)
	}
	if len(req.EmbeddingBF16) > 0 && len(req.EmbeddingBF16) != tableCount {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "bf16 embedding table length must match vocab*hidden", nil)
	}
	if len(req.EmbeddingQ4) > 0 || len(req.Q4Scales) > 0 || len(req.Q4Biases) > 0 {
		if err := validateHIPMLXAffineProjectionShape(req.HiddenSize, len(req.EmbeddingQ4), len(req.Q4Scales), len(req.Q4Biases), req.VocabSize, req.HiddenSize, req.Q4GroupSize, req.QuantBits); err != nil {
			return err
		}
	}
	return nil
}

func (req hipEmbeddingLookupRequest) deviceBuffers(driver nativeHIPDriver) (*hipEmbeddingLookupDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	tokens, err := hipUploadTokenIDs(driver, req.TokenIDs)
	if err != nil {
		return nil, err
	}
	buffers := &hipEmbeddingLookupDeviceBuffers{
		Tokens:     tokens,
		TokenCount: len(req.TokenIDs),
		VocabSize:  req.VocabSize,
		HiddenSize: req.HiddenSize,
		GroupSize:  req.Q4GroupSize,
		QuantBits:  hipMLXQ4ProjectionBitsOrDefault(req.QuantBits),
	}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()
	switch {
	case len(req.EmbeddingF32) > 0:
		payload, err := hipFloat32Payload(req.EmbeddingF32)
		if err != nil {
			return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "encode f32 embedding table", err)
		}
		embedding, err := hipUploadByteBuffer(driver, "rocm.hip.EmbeddingLookupLaunch", "embedding f32 table", payload, len(req.EmbeddingF32))
		if err != nil {
			return nil, err
		}
		buffers.Embedding = embedding
		buffers.TableEncode = hipEmbeddingTableEncodingF32
	case len(req.EmbeddingBF16) > 0:
		payload, err := hipUint16Payload(req.EmbeddingBF16)
		if err != nil {
			return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "encode bf16 embedding table", err)
		}
		embedding, err := hipUploadByteBuffer(driver, "rocm.hip.EmbeddingLookupLaunch", "embedding bf16 table", payload, len(req.EmbeddingBF16))
		if err != nil {
			return nil, err
		}
		buffers.Embedding = embedding
		buffers.TableEncode = hipEmbeddingTableEncodingBF16
	case len(req.EmbeddingQ4) > 0:
		payload, err := hipUint32Payload(req.EmbeddingQ4)
		if err != nil {
			return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "encode MLX q4 embedding table", err)
		}
		embedding, err := hipUploadByteBuffer(driver, "rocm.hip.EmbeddingLookupLaunch", "embedding MLX q4 table", payload, len(req.EmbeddingQ4))
		if err != nil {
			return nil, err
		}
		buffers.Embedding = embedding
		scalesPayload, err := hipUint16Payload(req.Q4Scales)
		if err != nil {
			return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "encode MLX q4 embedding scales", err)
		}
		scales, err := hipUploadByteBuffer(driver, "rocm.hip.EmbeddingLookupLaunch", "embedding MLX q4 scales", scalesPayload, len(req.Q4Scales))
		if err != nil {
			return nil, err
		}
		buffers.Scales = scales
		biasesPayload, err := hipUint16Payload(req.Q4Biases)
		if err != nil {
			return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "encode MLX q4 embedding biases", err)
		}
		biases, err := hipUploadByteBuffer(driver, "rocm.hip.EmbeddingLookupLaunch", "embedding MLX q4 biases", biasesPayload, len(req.Q4Biases))
		if err != nil {
			return nil, err
		}
		buffers.Biases = biases
		buffers.TableEncode = hipEmbeddingTableEncodingMLXQ4
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.EmbeddingLookupLaunch", "embedding lookup output", uint64(len(req.TokenIDs)*req.HiddenSize*4), len(req.TokenIDs)*req.HiddenSize)
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipEmbeddingLookupRequest) launchArgs(buffers *hipEmbeddingLookupDeviceBuffers) (hipEmbeddingLookupLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipEmbeddingLookupLaunchArgs{}, err
	}
	if buffers == nil || buffers.Tokens == nil || buffers.Embedding == nil || buffers.Output == nil {
		return hipEmbeddingLookupLaunchArgs{}, core.E("rocm.hip.EmbeddingLookupLaunch", "embedding lookup device buffers are required", nil)
	}
	encoding, err := hipEmbeddingLookupEncoding(req)
	if err != nil {
		return hipEmbeddingLookupLaunchArgs{}, err
	}
	if buffers.TokenCount != len(req.TokenIDs) ||
		buffers.VocabSize != req.VocabSize ||
		buffers.HiddenSize != req.HiddenSize ||
		buffers.Tokens.Count() != len(req.TokenIDs) ||
		buffers.Output.Count() != len(req.TokenIDs)*req.HiddenSize ||
		buffers.TableEncode != encoding {
		return hipEmbeddingLookupLaunchArgs{}, core.E("rocm.hip.EmbeddingLookupLaunch", "embedding lookup device buffer shape mismatch", nil)
	}
	if encoding == hipEmbeddingTableEncodingMLXQ4 {
		if buffers.Scales == nil || buffers.Biases == nil ||
			buffers.GroupSize != req.Q4GroupSize ||
			buffers.QuantBits != hipMLXQ4ProjectionBitsOrDefault(req.QuantBits) ||
			buffers.Embedding.Count() != len(req.EmbeddingQ4) ||
			buffers.Scales.Count() != len(req.Q4Scales) ||
			buffers.Biases.Count() != len(req.Q4Biases) {
			return hipEmbeddingLookupLaunchArgs{}, core.E("rocm.hip.EmbeddingLookupLaunch", "embedding lookup q4 device buffer shape mismatch", nil)
		}
	} else if buffers.Embedding.Count() != req.VocabSize*req.HiddenSize {
		return hipEmbeddingLookupLaunchArgs{}, core.E("rocm.hip.EmbeddingLookupLaunch", "embedding lookup device buffer shape mismatch", nil)
	}
	launch := hipEmbeddingLookupLaunchArgs{
		TokenPointer:     buffers.Tokens.Pointer(),
		EmbeddingPointer: buffers.Embedding.Pointer(),
		OutputPointer:    buffers.Output.Pointer(),
		TokenCount:       len(req.TokenIDs),
		VocabSize:        req.VocabSize,
		HiddenSize:       req.HiddenSize,
		TokenBytes:       buffers.Tokens.SizeBytes(),
		EmbeddingBytes:   buffers.Embedding.SizeBytes(),
		OutputBytes:      buffers.Output.SizeBytes(),
		TableEncoding:    encoding,
	}
	if encoding == hipEmbeddingTableEncodingMLXQ4 {
		launch.GroupSize = req.Q4GroupSize
		launch.QuantBits = hipMLXQ4ProjectionBitsOrDefault(req.QuantBits)
		launch.ScalePointer = buffers.Scales.Pointer()
		launch.BiasPointer = buffers.Biases.Pointer()
		launch.ScaleBytes = buffers.Scales.SizeBytes()
		launch.BiasBytes = buffers.Biases.SizeBytes()
	}
	return launch, nil
}

func (args hipEmbeddingLookupLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipDiffusionExpectedEmbeddingLaunchArgs) Binary() ([]byte, error) {
	if args.ProbabilityPointer == 0 || args.EmbeddingPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.DiffusionExpectedEmbeddingLaunch", "probability, embedding, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("diffusion expected embedding rows", args.Rows)
	if err != nil {
		return nil, err
	}
	vocab, err := rocmDeviceKVPositiveUint32("diffusion expected embedding vocab size", args.VocabSize)
	if err != nil {
		return nil, err
	}
	hidden, err := rocmDeviceKVPositiveUint32("diffusion expected embedding hidden size", args.HiddenSize)
	if err != nil {
		return nil, err
	}
	if args.ProbabilityBytes != uint64(rows)*uint64(vocab)*4 {
		return nil, core.E("rocm.hip.DiffusionExpectedEmbeddingLaunch", "probability byte count mismatch", nil)
	}
	if args.OutputBytes != uint64(rows)*uint64(hidden)*4 {
		return nil, core.E("rocm.hip.DiffusionExpectedEmbeddingLaunch", "output byte count mismatch", nil)
	}
	var groupSize, bits uint32
	switch args.TableEncoding {
	case hipEmbeddingTableEncodingF32:
		if args.EmbeddingBytes != uint64(vocab)*uint64(hidden)*4 {
			return nil, core.E("rocm.hip.DiffusionExpectedEmbeddingLaunch", "F32 embedding byte count mismatch", nil)
		}
	case hipEmbeddingTableEncodingBF16:
		if args.EmbeddingBytes != uint64(vocab)*uint64(hidden)*2 {
			return nil, core.E("rocm.hip.DiffusionExpectedEmbeddingLaunch", "BF16 embedding byte count mismatch", nil)
		}
	case hipEmbeddingTableEncodingMLXQ4:
		if args.ScalePointer == 0 || args.BiasPointer == 0 {
			return nil, core.E("rocm.hip.DiffusionExpectedEmbeddingLaunch", "MLX affine scale and bias pointers are required", nil)
		}
		groupSize, err = rocmDeviceKVPositiveUint32("diffusion expected embedding group size", args.GroupSize)
		if err != nil {
			return nil, err
		}
		bits, err = rocmDeviceKVPositiveUint32("diffusion expected embedding quant bits", hipMLXQ4ProjectionBitsOrDefault(args.QuantBits))
		if err != nil {
			return nil, err
		}
		packedPerRow, groupsPerRow, err := hipMLXAffineLaunchPackedGroups("rocm.hip.DiffusionExpectedEmbeddingLaunch", hidden, groupSize, bits)
		if err != nil {
			return nil, err
		}
		if args.EmbeddingBytes != uint64(vocab)*packedPerRow*4 ||
			args.ScaleBytes != uint64(vocab)*groupsPerRow*2 ||
			args.BiasBytes != uint64(vocab)*groupsPerRow*2 {
			return nil, core.E("rocm.hip.DiffusionExpectedEmbeddingLaunch", "MLX affine embedding byte count mismatch", nil)
		}
	default:
		return nil, core.E("rocm.hip.DiffusionExpectedEmbeddingLaunch", "unsupported embedding table encoding", nil)
	}
	if math.IsNaN(float64(args.OutputScale)) || math.IsInf(float64(args.OutputScale), 0) {
		return nil, core.E("rocm.hip.DiffusionExpectedEmbeddingLaunch", "output scale must be finite", nil)
	}
	payload := hipBorrowLaunchPacket(hipDiffusionExpectedEmbeddingArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipDiffusionExpectedEmbeddingArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.ProbabilityPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.EmbeddingPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ScalePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[48:], rows)
	binary.LittleEndian.PutUint32(payload[52:], vocab)
	binary.LittleEndian.PutUint32(payload[56:], hidden)
	binary.LittleEndian.PutUint32(payload[60:], args.TableEncoding)
	binary.LittleEndian.PutUint32(payload[64:], groupSize)
	binary.LittleEndian.PutUint32(payload[68:], bits)
	binary.LittleEndian.PutUint64(payload[72:], args.ProbabilityBytes)
	binary.LittleEndian.PutUint64(payload[80:], args.EmbeddingBytes)
	binary.LittleEndian.PutUint64(payload[88:], args.ScaleBytes)
	binary.LittleEndian.PutUint64(payload[96:], args.BiasBytes)
	binary.LittleEndian.PutUint64(payload[104:], args.OutputBytes)
	if args.OutputScale != 0 && args.OutputScale != 1 {
		binary.LittleEndian.PutUint32(payload[112:], math.Float32bits(args.OutputScale))
	}
	return payload, nil
}

func hipRunDiffusionExpectedEmbeddingKernel(ctx context.Context, driver nativeHIPDriver, probabilities []float32, rows int, cfg hipDeviceEmbeddingLookupConfig, outputScale float32) ([]float32, error) {
	const op = "rocm.hip.DiffusionExpectedEmbeddingLaunch"
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E(op, "HIP driver is not available", nil)
	}
	if rows <= 0 || cfg.VocabSize <= 0 || cfg.HiddenSize <= 0 || len(probabilities) != rows*cfg.VocabSize {
		return nil, core.E(op, "probability and embedding geometry mismatch", nil)
	}
	probabilityBuffer, err := hipUploadGemma4Q4Float32Input(driver, "diffusion probabilities", probabilities)
	if err != nil {
		return nil, err
	}
	defer probabilityBuffer.Close()
	output, err := hipRunDiffusionExpectedEmbeddingDeviceKernel(ctx, driver, probabilityBuffer, rows, cfg, outputScale)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	return hipReadFloat32DeviceOutput(output, op, "diffusion expected embedding output", rows*cfg.HiddenSize)
}

func hipRunDiffusionExpectedEmbeddingDeviceKernel(ctx context.Context, driver nativeHIPDriver, probabilityBuffer *hipDeviceByteBuffer, rows int, cfg hipDeviceEmbeddingLookupConfig, outputScale float32) (*hipDeviceByteBuffer, error) {
	const op = "rocm.hip.DiffusionExpectedEmbeddingLaunch"
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E(op, "HIP driver is not available", nil)
	}
	if probabilityBuffer == nil || probabilityBuffer.Pointer() == 0 || rows <= 0 || cfg.VocabSize <= 0 || cfg.HiddenSize <= 0 {
		return nil, core.E(op, "probability and embedding geometry mismatch", nil)
	}
	if probabilityBuffer.SizeBytes() != uint64(rows)*uint64(cfg.VocabSize)*4 {
		return nil, core.E(op, "probability byte count mismatch", nil)
	}
	outputCount := rows * cfg.HiddenSize
	output, err := hipAllocateByteBuffer(driver, op, "diffusion expected embedding output", uint64(outputCount*4), outputCount)
	if err != nil {
		return nil, err
	}
	args, err := (hipDiffusionExpectedEmbeddingLaunchArgs{
		ProbabilityPointer: probabilityBuffer.Pointer(),
		EmbeddingPointer:   cfg.EmbeddingPointer,
		ScalePointer:       cfg.ScalePointer,
		BiasPointer:        cfg.BiasPointer,
		OutputPointer:      output.Pointer(),
		Rows:               rows,
		VocabSize:          cfg.VocabSize,
		HiddenSize:         cfg.HiddenSize,
		TableEncoding:      cfg.TableEncoding,
		GroupSize:          cfg.GroupSize,
		QuantBits:          cfg.QuantBits,
		ProbabilityBytes:   probabilityBuffer.SizeBytes(),
		EmbeddingBytes:     cfg.EmbeddingBytes,
		ScaleBytes:         cfg.ScaleBytes,
		BiasBytes:          cfg.BiasBytes,
		OutputBytes:        output.SizeBytes(),
		OutputScale:        outputScale,
	}).Binary()
	if err != nil {
		_ = output.Close()
		return nil, err
	}
	kernelName := hipKernelNameDiffusionExpectedEmbedding
	gridRows := rows
	gridHidden := cfg.HiddenSize
	bits := hipMLXQ4ProjectionBitsOrDefault(cfg.QuantBits)
	if rows >= hipDiffusionExpectedEmbeddingQ8G64SubgroupMinRows &&
		cfg.TableEncoding == hipEmbeddingTableEncodingMLXQ4 &&
		cfg.GroupSize == 64 &&
		bits == 8 &&
		core.Env(hipDisableDiffusionExpectedEmbeddingSubgroupEnv) != "1" {
		kernelName = hipKernelNameDiffusionExpectedEmbeddingQ8G64SubgroupRows64Probability4
		if core.Env(hipDisableDiffusionExpectedEmbeddingProbability4Env) == "1" {
			kernelName = hipKernelNameDiffusionExpectedEmbeddingQ8G64SubgroupRows64
		}
		gridRows = (rows + hipDiffusionExpectedEmbeddingQ8G64SubgroupRows - 1) / hipDiffusionExpectedEmbeddingQ8G64SubgroupRows
		gridHidden = ((cfg.HiddenSize + hipDiffusionExpectedEmbeddingQ8G64SubgroupDims - 1) / hipDiffusionExpectedEmbeddingQ8G64SubgroupDims) * 256
	} else if rows >= hipDiffusionExpectedEmbeddingQ8G64TileRows &&
		rows <= 2*hipDiffusionExpectedEmbeddingQ8G64TileRows &&
		cfg.TableEncoding == hipEmbeddingTableEncodingMLXQ4 &&
		cfg.GroupSize == 64 &&
		bits == 8 &&
		core.Env(hipDisableDiffusionExpectedEmbeddingTileEnv) != "1" {
		kernelName = hipKernelNameDiffusionExpectedEmbeddingQ8G64Tile32x64
		gridRows = (rows + hipDiffusionExpectedEmbeddingQ8G64TileRows - 1) / hipDiffusionExpectedEmbeddingQ8G64TileRows
		gridHidden = ((cfg.HiddenSize + hipDiffusionExpectedEmbeddingQ8G64TileDims - 1) / hipDiffusionExpectedEmbeddingQ8G64TileDims) * 256
	} else if rows >= hipDiffusionExpectedEmbeddingQ8G64RowsPerBlock &&
		cfg.TableEncoding == hipEmbeddingTableEncodingMLXQ4 &&
		cfg.GroupSize == 64 &&
		bits == 8 {
		kernelName = hipKernelNameDiffusionExpectedEmbeddingQ8G64Dims4Rows4
		gridRows = (rows + hipDiffusionExpectedEmbeddingQ8G64RowsPerBlock - 1) / hipDiffusionExpectedEmbeddingQ8G64RowsPerBlock
		gridHidden = (cfg.HiddenSize + hipDiffusionExpectedEmbeddingQ8G64DimsPerThread - 1) / hipDiffusionExpectedEmbeddingQ8G64DimsPerThread
	} else if rows >= hipDiffusionExpectedEmbeddingAffineG64RowsPerBlock &&
		cfg.TableEncoding == hipEmbeddingTableEncodingMLXQ4 &&
		cfg.GroupSize == 64 &&
		(bits == 4 || bits == 8) {
		kernelName = hipKernelNameDiffusionExpectedEmbeddingAffineG64Rows16
		gridRows = (rows + hipDiffusionExpectedEmbeddingAffineG64RowsPerBlock - 1) / hipDiffusionExpectedEmbeddingAffineG64RowsPerBlock
	}
	gridX, err := rocmDeviceKVPositiveUint32("diffusion expected embedding hidden blocks", (gridHidden+255)/256)
	if err != nil {
		_ = output.Close()
		return nil, err
	}
	gridY, err := rocmDeviceKVPositiveUint32("diffusion expected embedding rows", gridRows)
	if err != nil {
		_ = output.Close()
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, hipKernelLaunchConfig{
		Name:   kernelName,
		Args:   args,
		GridX:  gridX,
		GridY:  gridY,
		GridZ:  1,
		BlockX: 256,
		BlockY: 1,
		BlockZ: 1,
	}); err != nil {
		_ = output.Close()
		return nil, err
	}
	return output, nil
}

func (args hipEmbeddingLookupLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	return args.binaryInto(false, payload)
}

func (args hipEmbeddingLookupLaunchArgs) GreedyTokenBinary() ([]byte, error) {
	return args.GreedyTokenBinaryInto(nil)
}

func (args hipEmbeddingLookupLaunchArgs) GreedyTokenBinaryInto(payload []byte) ([]byte, error) {
	return args.binaryInto(true, payload)
}

func (args hipEmbeddingLookupLaunchArgs) binary(greedyToken bool) ([]byte, error) {
	return args.binaryInto(greedyToken, nil)
}

func (args hipEmbeddingLookupLaunchArgs) binaryInto(greedyToken bool, payload []byte) ([]byte, error) {
	if args.TokenPointer == 0 || args.EmbeddingPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "token, embedding, and output pointers are required", nil)
	}
	tokenCount, err := rocmDeviceKVPositiveUint32("token count", args.TokenCount)
	if err != nil {
		return nil, err
	}
	if greedyToken && tokenCount != 1 {
		return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "greedy token embedding requires exactly one token", nil)
	}
	vocabSize, err := rocmDeviceKVPositiveUint32("vocab size", args.VocabSize)
	if err != nil {
		return nil, err
	}
	hiddenSize, err := rocmDeviceKVPositiveUint32("hidden size", args.HiddenSize)
	if err != nil {
		return nil, err
	}
	wantTokenBytes := uint64(tokenCount) * 4
	if greedyToken {
		wantTokenBytes = hipMLXQ4ProjectionBestBytes
	}
	tokenBytes, err := hipExactUint32Bytes("embedding lookup tokens", args.TokenBytes, wantTokenBytes)
	if err != nil {
		return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "token byte count", err)
	}
	tableCount := uint64(vocabSize) * uint64(hiddenSize)
	var groupSize uint32
	var scaleBytes uint32
	var biasBytes uint32
	var quantBits uint32
	switch args.TableEncoding {
	case hipEmbeddingTableEncodingF32:
		if args.EmbeddingBytes != tableCount*4 {
			return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "f32 embedding byte count mismatch", nil)
		}
	case hipEmbeddingTableEncodingBF16:
		if args.EmbeddingBytes != tableCount*2 {
			return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "bf16 embedding byte count mismatch", nil)
		}
	case hipEmbeddingTableEncodingMLXQ4:
		if args.ScalePointer == 0 || args.BiasPointer == 0 {
			return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "q4 scale and bias pointers are required", nil)
		}
		groupSize, err = rocmDeviceKVPositiveUint32("q4 group size", args.GroupSize)
		if err != nil {
			return nil, err
		}
		quantBits, err = rocmDeviceKVPositiveUint32("MLX affine bits", hipMLXQ4ProjectionBitsOrDefault(args.QuantBits))
		if err != nil {
			return nil, err
		}
		packedPerRow, groupsPerRow, err := hipMLXAffineLaunchPackedGroups("rocm.hip.EmbeddingLookupLaunch", hiddenSize, groupSize, quantBits)
		if err != nil {
			return nil, err
		}
		weightBytes := uint64(vocabSize) * packedPerRow * 4
		if args.EmbeddingBytes != weightBytes {
			return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "MLX affine embedding byte count mismatch", nil)
		}
		groupBytes := uint64(vocabSize) * groupsPerRow * 2
		scaleBytes, err = hipExactUint32Bytes("q4 embedding scales", args.ScaleBytes, groupBytes)
		if err != nil {
			return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "scale byte count", err)
		}
		biasBytes, err = hipExactUint32Bytes("q4 embedding biases", args.BiasBytes, groupBytes)
		if err != nil {
			return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "bias byte count", err)
		}
	default:
		return nil, core.E("rocm.hip.EmbeddingLookupLaunch", core.Sprintf("unsupported embedding table encoding %d", args.TableEncoding), nil)
	}
	outputBytes := uint64(tokenCount) * uint64(hiddenSize) * 4
	if args.OutputBytes != outputBytes {
		return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "output byte count mismatch", nil)
	}
	if math.IsNaN(float64(args.OutputScale)) || math.IsInf(float64(args.OutputScale), 0) {
		return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "output scale must be finite", nil)
	}
	if cap(payload) < hipEmbeddingLookupLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipEmbeddingLookupLaunchArgsBytes)
	} else {
		payload = payload[:hipEmbeddingLookupLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipEmbeddingLookupLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.TokenPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.EmbeddingPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], tokenCount)
	binary.LittleEndian.PutUint32(payload[36:], vocabSize)
	binary.LittleEndian.PutUint32(payload[40:], hiddenSize)
	binary.LittleEndian.PutUint32(payload[44:], tokenBytes)
	binary.LittleEndian.PutUint64(payload[48:], args.EmbeddingBytes)
	binary.LittleEndian.PutUint64(payload[56:], args.OutputBytes)
	binary.LittleEndian.PutUint32(payload[64:], args.TableEncoding)
	binary.LittleEndian.PutUint32(payload[68:], groupSize)
	binary.LittleEndian.PutUint64(payload[72:], uint64(args.ScalePointer))
	binary.LittleEndian.PutUint64(payload[80:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint32(payload[88:], scaleBytes)
	binary.LittleEndian.PutUint32(payload[92:], biasBytes)
	if args.OutputScale != 0 && args.OutputScale != 1 {
		binary.LittleEndian.PutUint32(payload[96:], math.Float32bits(args.OutputScale))
	}
	binary.LittleEndian.PutUint32(payload[100:], quantBits)
	return payload, nil
}

func (buffers *hipEmbeddingLookupDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Biases, buffers.Scales, buffers.Embedding} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	if err := buffers.Tokens.Close(); err != nil {
		lastErr = err
	}
	return lastErr
}

func (buffers *hipEmbeddingLookupDeviceBuffers) ReadOutput() ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "embedding lookup output buffer is required", nil)
	}
	wantCount := buffers.TokenCount * buffers.HiddenSize
	if buffers.TokenCount <= 0 || buffers.HiddenSize <= 0 || buffers.Output.Count() != wantCount || buffers.Output.SizeBytes() != uint64(wantCount*4) {
		return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "embedding lookup output byte count mismatch", nil)
	}
	payload := make([]byte, buffers.Output.SizeBytes())
	if err := buffers.Output.driver.CopyDeviceToHost(buffers.Output.Pointer(), payload); err != nil {
		return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "copy embedding lookup output", err)
	}
	values, err := hipFloat32PayloadValues(payload)
	if err != nil {
		return nil, err
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "embedding lookup output values must be finite", nil)
	}
	return values, nil
}

func hipRunEmbeddingLookupKernel(ctx context.Context, driver nativeHIPDriver, req hipEmbeddingLookupRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameEmbedLookup, launchBytes, req.HiddenSize*len(req.TokenIDs))
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func hipRunEmbeddingLookupKernelWithDeviceTable(ctx context.Context, driver nativeHIPDriver, tokenIDs []int32, cfg hipDeviceEmbeddingLookupConfig) ([]float32, error) {
	output, err := hipRunEmbeddingLookupKernelWithDeviceTableBuffer(ctx, driver, tokenIDs, cfg)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	return (&hipEmbeddingLookupDeviceBuffers{Output: output, TokenCount: len(tokenIDs), HiddenSize: cfg.HiddenSize}).ReadOutput()
}

func hipRunEmbeddingLookupKernelWithDeviceTableBuffer(ctx context.Context, driver nativeHIPDriver, tokenIDs []int32, cfg hipDeviceEmbeddingLookupConfig) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "HIP driver is not available", nil)
	}
	if err := cfg.validate(tokenIDs); err != nil {
		return nil, err
	}
	tokens, err := hipUploadTokenIDs(driver, tokenIDs)
	if err != nil {
		return nil, err
	}
	defer tokens.Close()
	return hipRunEmbeddingLookupKernelWithDeviceTableTokenBatchBuffer(ctx, driver, cfg, tokens)
}

func hipRunEmbeddingLookupKernelWithDeviceTableTokenBatchBuffer(ctx context.Context, driver nativeHIPDriver, cfg hipDeviceEmbeddingLookupConfig, tokenBuffer *hipDeviceTokenBuffer) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "HIP driver is not available", nil)
	}
	if err := cfg.validateShape(); err != nil {
		return nil, err
	}
	if tokenBuffer == nil || tokenBuffer.Pointer() == 0 || tokenBuffer.Count() <= 0 || tokenBuffer.SizeBytes() != uint64(tokenBuffer.Count()*4) {
		return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "token buffer is required", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.EmbeddingLookupLaunch", "embedding lookup output", uint64(tokenBuffer.Count()*cfg.HiddenSize*4), tokenBuffer.Count()*cfg.HiddenSize)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunEmbeddingLookupKernelWithDeviceTableTokenBatchScaledOutput(ctx, driver, cfg, tokenBuffer, output, 0); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunEmbeddingLookupKernelWithDeviceTableBufferScaledOutput(ctx context.Context, driver nativeHIPDriver, tokenIDs []int32, cfg hipDeviceEmbeddingLookupConfig, output *hipDeviceByteBuffer, outputScale float32) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "HIP driver is not available", nil)
	}
	if err := cfg.validate(tokenIDs); err != nil {
		return err
	}
	if output == nil || output.Pointer() == 0 || output.Count() != len(tokenIDs)*cfg.HiddenSize || output.SizeBytes() != uint64(output.Count()*4) {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "embedding output buffer shape mismatch", nil)
	}
	tokens, err := hipUploadTokenIDs(driver, tokenIDs)
	if err != nil {
		return err
	}
	defer tokens.Close()
	return hipRunEmbeddingLookupKernelWithDeviceTableTokenBatchScaledOutput(ctx, driver, cfg, tokens, output, outputScale)
}

func hipRunEmbeddingLookupKernelWithDeviceTableTokenBatchScaledOutput(ctx context.Context, driver nativeHIPDriver, cfg hipDeviceEmbeddingLookupConfig, tokenBuffer *hipDeviceTokenBuffer, output *hipDeviceByteBuffer, outputScale float32) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "HIP driver is not available", nil)
	}
	if err := cfg.validateShape(); err != nil {
		return err
	}
	if tokenBuffer == nil || tokenBuffer.Pointer() == 0 || tokenBuffer.Count() <= 0 || tokenBuffer.SizeBytes() != uint64(tokenBuffer.Count()*4) {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "token buffer is required", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != tokenBuffer.Count()*cfg.HiddenSize || output.SizeBytes() != uint64(output.Count()*4) {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "embedding output buffer shape mismatch", nil)
	}
	launchBytes, err := (hipEmbeddingLookupLaunchArgs{
		TokenPointer:     tokenBuffer.Pointer(),
		EmbeddingPointer: cfg.EmbeddingPointer,
		OutputPointer:    output.Pointer(),
		TokenCount:       tokenBuffer.Count(),
		VocabSize:        cfg.VocabSize,
		HiddenSize:       cfg.HiddenSize,
		TokenBytes:       tokenBuffer.SizeBytes(),
		EmbeddingBytes:   cfg.EmbeddingBytes,
		OutputBytes:      output.SizeBytes(),
		TableEncoding:    cfg.TableEncoding,
		GroupSize:        cfg.GroupSize,
		ScalePointer:     cfg.ScalePointer,
		BiasPointer:      cfg.BiasPointer,
		ScaleBytes:       cfg.ScaleBytes,
		BiasBytes:        cfg.BiasBytes,
		OutputScale:      outputScale,
		QuantBits:        cfg.QuantBits,
	}).Binary()
	if err != nil {
		return err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameEmbedLookup, launchBytes, cfg.HiddenSize*tokenBuffer.Count())
	if err != nil {
		return err
	}
	return hipLaunchKernelContext(ctx, driver, config)
}

func hipRunEmbeddingLookupKernelWithDeviceTableSingleTokenBuffer(ctx context.Context, driver nativeHIPDriver, tokenID int32, cfg hipDeviceEmbeddingLookupConfig, tokenBuffer *hipDeviceByteBuffer) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "HIP driver is not available", nil)
	}
	if tokenBuffer == nil || tokenBuffer.Pointer() == 0 || tokenBuffer.Count() != 1 || tokenBuffer.SizeBytes() != 4 {
		return nil, core.E("rocm.hip.EmbeddingLookupLaunch", "single-token workspace buffer is required", nil)
	}
	if err := cfg.validateSingleToken(tokenID); err != nil {
		return nil, err
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.EmbeddingLookupLaunch", "embedding lookup output", uint64(cfg.HiddenSize*4), cfg.HiddenSize)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunEmbeddingLookupKernelWithDeviceTableSingleTokenBufferOutput(ctx, driver, tokenID, cfg, tokenBuffer, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunEmbeddingLookupKernelWithDeviceTableSingleTokenBufferOutput(ctx context.Context, driver nativeHIPDriver, tokenID int32, cfg hipDeviceEmbeddingLookupConfig, tokenBuffer, output *hipDeviceByteBuffer) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if err := cfg.validateSingleToken(tokenID); err != nil {
		return err
	}
	if tokenBuffer == nil || tokenBuffer.Pointer() == 0 || tokenBuffer.Count() != 1 || tokenBuffer.SizeBytes() != 4 {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "single-token workspace buffer is required", nil)
	}
	if err := hipWriteSingleTokenID(driver, tokenBuffer.Pointer(), tokenID); err != nil {
		return err
	}
	return hipRunEmbeddingLookupKernelWithDeviceTableTokenBufferOutput(ctx, driver, cfg, tokenBuffer, output)
}

func hipRunEmbeddingLookupKernelWithDeviceTableTokenBufferOutput(ctx context.Context, driver nativeHIPDriver, cfg hipDeviceEmbeddingLookupConfig, tokenBuffer, output *hipDeviceByteBuffer) error {
	return hipRunEmbeddingLookupKernelWithDeviceTableTokenBufferScaledOutput(ctx, driver, cfg, tokenBuffer, output, 0)
}

func hipRunEmbeddingLookupKernelWithDeviceTableTokenBufferScaledOutput(ctx context.Context, driver nativeHIPDriver, cfg hipDeviceEmbeddingLookupConfig, tokenBuffer, output *hipDeviceByteBuffer, outputScale float32) error {
	return hipRunEmbeddingLookupKernelWithDeviceTableTokenBufferScaledOutputWithWorkspace(ctx, driver, cfg, tokenBuffer, output, outputScale, nil)
}

func hipRunEmbeddingLookupKernelWithDeviceTableTokenBufferScaledOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipDeviceEmbeddingLookupConfig, tokenBuffer, output *hipDeviceByteBuffer, outputScale float32, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "HIP driver is not available", nil)
	}
	if tokenBuffer == nil || tokenBuffer.Pointer() == 0 || tokenBuffer.Count() != 1 || tokenBuffer.SizeBytes() != 4 {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "single-token workspace buffer is required", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != cfg.HiddenSize || output.SizeBytes() != uint64(cfg.HiddenSize*4) {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "single-token output buffer shape mismatch", nil)
	}
	launchArgs := hipEmbeddingLookupLaunchArgs{
		TokenPointer:     tokenBuffer.Pointer(),
		EmbeddingPointer: cfg.EmbeddingPointer,
		OutputPointer:    output.Pointer(),
		TokenCount:       1,
		VocabSize:        cfg.VocabSize,
		HiddenSize:       cfg.HiddenSize,
		TokenBytes:       tokenBuffer.SizeBytes(),
		EmbeddingBytes:   cfg.EmbeddingBytes,
		OutputBytes:      output.SizeBytes(),
		TableEncoding:    cfg.TableEncoding,
		GroupSize:        cfg.GroupSize,
		ScalePointer:     cfg.ScalePointer,
		BiasPointer:      cfg.BiasPointer,
		ScaleBytes:       cfg.ScaleBytes,
		BiasBytes:        cfg.BiasBytes,
		OutputScale:      outputScale,
		QuantBits:        cfg.QuantBits,
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.EmbeddingLookupArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameEmbedLookup, launchBytes, cfg.HiddenSize)
	if err != nil {
		return err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return err
	}
	return nil
}

func hipRunEmbeddingLookupKernelWithDeviceTableGreedyTokenOutput(ctx context.Context, driver nativeHIPDriver, cfg hipDeviceEmbeddingLookupConfig, greedyToken, output *hipDeviceByteBuffer) error {
	return hipRunEmbeddingLookupKernelWithDeviceTableGreedyTokenScaledOutput(ctx, driver, cfg, greedyToken, output, 0)
}

func hipRunEmbeddingLookupKernelWithDeviceTableGreedyTokenScaledOutput(ctx context.Context, driver nativeHIPDriver, cfg hipDeviceEmbeddingLookupConfig, greedyToken, output *hipDeviceByteBuffer, outputScale float32) error {
	return hipRunEmbeddingLookupKernelWithDeviceTableGreedyTokenScaledOutputWithWorkspace(ctx, driver, cfg, greedyToken, output, outputScale, nil)
}

func hipRunEmbeddingLookupKernelWithDeviceTableGreedyTokenScaledOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipDeviceEmbeddingLookupConfig, greedyToken, output *hipDeviceByteBuffer, outputScale float32, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "HIP driver is not available", nil)
	}
	if greedyToken == nil || greedyToken.Pointer() == 0 || greedyToken.Count() != 1 || greedyToken.SizeBytes() != hipMLXQ4ProjectionBestBytes {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "greedy token buffer is required", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != cfg.HiddenSize || output.SizeBytes() != uint64(cfg.HiddenSize*4) {
		return core.E("rocm.hip.EmbeddingLookupLaunch", "single-token output buffer shape mismatch", nil)
	}
	launchArgs := hipEmbeddingLookupLaunchArgs{
		TokenPointer:     greedyToken.Pointer(),
		EmbeddingPointer: cfg.EmbeddingPointer,
		OutputPointer:    output.Pointer(),
		TokenCount:       1,
		VocabSize:        cfg.VocabSize,
		HiddenSize:       cfg.HiddenSize,
		TokenBytes:       greedyToken.SizeBytes(),
		EmbeddingBytes:   cfg.EmbeddingBytes,
		OutputBytes:      output.SizeBytes(),
		TableEncoding:    cfg.TableEncoding,
		GroupSize:        cfg.GroupSize,
		ScalePointer:     cfg.ScalePointer,
		BiasPointer:      cfg.BiasPointer,
		ScaleBytes:       cfg.ScaleBytes,
		BiasBytes:        cfg.BiasBytes,
		OutputScale:      outputScale,
		QuantBits:        cfg.QuantBits,
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.GreedyTokenBinaryInto(workspace.EmbeddingLookupArgs[:])
	} else {
		launchBytes, err = launchArgs.GreedyTokenBinary()
	}
	if err != nil {
		return err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameEmbedLookupGreedyToken, launchBytes, cfg.HiddenSize)
	if err != nil {
		return err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return err
	}
	return nil
}

func hipEmbeddingLookupEncoding(req hipEmbeddingLookupRequest) (uint32, error) {
	switch {
	case len(req.EmbeddingF32) > 0 && len(req.EmbeddingBF16) == 0 && len(req.EmbeddingQ4) == 0 && len(req.Q4Scales) == 0 && len(req.Q4Biases) == 0:
		return hipEmbeddingTableEncodingF32, nil
	case len(req.EmbeddingBF16) > 0 && len(req.EmbeddingF32) == 0 && len(req.EmbeddingQ4) == 0 && len(req.Q4Scales) == 0 && len(req.Q4Biases) == 0:
		return hipEmbeddingTableEncodingBF16, nil
	case len(req.EmbeddingQ4) > 0 && len(req.Q4Scales) > 0 && len(req.Q4Biases) > 0 && len(req.EmbeddingF32) == 0 && len(req.EmbeddingBF16) == 0:
		return hipEmbeddingTableEncodingMLXQ4, nil
	default:
		return 0, core.E("rocm.hip.EmbeddingLookupLaunch", "exactly one embedding table encoding is required", nil)
	}
}

func (req hipRerankCosineRequest) validate() error {
	if req.DocumentCount <= 0 || req.Dim <= 0 {
		return core.E("rocm.hip.RerankCosineLaunch", "document count and dimension must be positive", nil)
	}
	if len(req.Query) != req.Dim {
		return core.E("rocm.hip.RerankCosineLaunch", "query length must match dimension", nil)
	}
	if len(req.Documents) != req.DocumentCount*req.Dim {
		return core.E("rocm.hip.RerankCosineLaunch", "document vector length must match document_count*dim", nil)
	}
	return nil
}

func (req hipRerankCosineRequest) deviceBuffers(driver nativeHIPDriver) (*hipRerankCosineDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	queryPayload, err := hipFloat32Payload(req.Query)
	if err != nil {
		return nil, core.E("rocm.hip.RerankCosineLaunch", "encode query", err)
	}
	query, err := hipUploadByteBuffer(driver, "rocm.hip.RerankCosineLaunch", "rerank query", queryPayload, len(req.Query))
	if err != nil {
		return nil, err
	}
	buffers := &hipRerankCosineDeviceBuffers{Query: query, DocumentCount: req.DocumentCount, Dim: req.Dim}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()
	documentPayload, err := hipFloat32Payload(req.Documents)
	if err != nil {
		return nil, core.E("rocm.hip.RerankCosineLaunch", "encode documents", err)
	}
	documents, err := hipUploadByteBuffer(driver, "rocm.hip.RerankCosineLaunch", "rerank documents", documentPayload, len(req.Documents))
	if err != nil {
		return nil, err
	}
	buffers.Documents = documents
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RerankCosineLaunch", "rerank output", uint64(req.DocumentCount*4), req.DocumentCount)
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipRerankCosineRequest) launchArgs(buffers *hipRerankCosineDeviceBuffers) (hipRerankCosineLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipRerankCosineLaunchArgs{}, err
	}
	if buffers == nil || buffers.Query == nil || buffers.Documents == nil || buffers.Output == nil {
		return hipRerankCosineLaunchArgs{}, core.E("rocm.hip.RerankCosineLaunch", "rerank device buffers are required", nil)
	}
	if buffers.Query.Count() != req.Dim || buffers.Documents.Count() != req.DocumentCount*req.Dim ||
		buffers.Output.Count() != req.DocumentCount || buffers.DocumentCount != req.DocumentCount || buffers.Dim != req.Dim {
		return hipRerankCosineLaunchArgs{}, core.E("rocm.hip.RerankCosineLaunch", "rerank device buffer shape mismatch", nil)
	}
	return hipRerankCosineLaunchArgs{
		QueryPointer:    buffers.Query.Pointer(),
		DocumentPointer: buffers.Documents.Pointer(),
		OutputPointer:   buffers.Output.Pointer(),
		DocumentCount:   req.DocumentCount,
		Dim:             req.Dim,
		QueryBytes:      buffers.Query.SizeBytes(),
		DocumentBytes:   buffers.Documents.SizeBytes(),
		OutputBytes:     buffers.Output.SizeBytes(),
	}, nil
}

func (args hipRerankCosineLaunchArgs) Binary() ([]byte, error) {
	if args.QueryPointer == 0 || args.DocumentPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.RerankCosineLaunch", "query, document, and output pointers are required", nil)
	}
	documentCount, err := rocmDeviceKVPositiveUint32("document count", args.DocumentCount)
	if err != nil {
		return nil, err
	}
	dim, err := rocmDeviceKVPositiveUint32("dimension", args.Dim)
	if err != nil {
		return nil, err
	}
	documentEntries, err := hipUint32Product("document vector count", documentCount, dim)
	if err != nil {
		return nil, err
	}
	queryBytes, err := hipAlignedFloat32Bytes("rerank query", args.QueryBytes, dim)
	if err != nil {
		return nil, core.E("rocm.hip.RerankCosineLaunch", "query byte count", err)
	}
	documentBytes, err := hipAlignedFloat32Bytes("rerank documents", args.DocumentBytes, documentEntries)
	if err != nil {
		return nil, core.E("rocm.hip.RerankCosineLaunch", "document byte count", err)
	}
	outputBytes, err := hipAlignedFloat32Bytes("rerank output", args.OutputBytes, documentCount)
	if err != nil {
		return nil, core.E("rocm.hip.RerankCosineLaunch", "output byte count", err)
	}
	payload := hipBorrowLaunchPacket(hipRerankCosineLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipRerankCosineLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.QueryPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.DocumentPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], documentCount)
	binary.LittleEndian.PutUint32(payload[36:], dim)
	binary.LittleEndian.PutUint32(payload[40:], queryBytes)
	binary.LittleEndian.PutUint32(payload[44:], documentBytes)
	binary.LittleEndian.PutUint32(payload[48:], outputBytes)
	return payload, nil
}

func (buffers *hipRerankCosineDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Documents, buffers.Query} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipRerankCosineDeviceBuffers) ReadOutput() ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.RerankCosineLaunch", "rerank output buffer is required", nil)
	}
	if buffers.DocumentCount <= 0 || buffers.Output.Count() != buffers.DocumentCount || buffers.Output.SizeBytes() != uint64(buffers.DocumentCount*4) {
		return nil, core.E("rocm.hip.RerankCosineLaunch", "rerank output byte count mismatch", nil)
	}
	payload := make([]byte, buffers.Output.SizeBytes())
	if err := buffers.Output.driver.CopyDeviceToHost(buffers.Output.Pointer(), payload); err != nil {
		return nil, core.E("rocm.hip.RerankCosineLaunch", "copy rerank output", err)
	}
	values, err := hipFloat32PayloadValues(payload)
	if err != nil {
		return nil, err
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E("rocm.hip.RerankCosineLaunch", "rerank output values must be finite", nil)
	}
	return values, nil
}

func hipRunRerankCosineKernel(ctx context.Context, driver nativeHIPDriver, req hipRerankCosineRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameRerank, launchBytes, req.DocumentCount)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func splitFloat32Vectors(flat []float32, dim int) [][]float32 {
	if dim <= 0 {
		return nil
	}
	out := make([][]float32, 0, len(flat)/dim)
	for start := 0; start < len(flat); start += dim {
		end := start + dim
		if end > len(flat) {
			end = len(flat)
		}
		out = append(out, flat[start:end])
	}
	return out
}
