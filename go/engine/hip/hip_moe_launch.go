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
	hipMoERouterLaunchArgsVersion          uint32 = 1
	hipMoERouterLaunchArgsBytes                   = 64
	hipMoERouterLaunchStatusOK             uint32 = 0x4d4f4552
	hipMoERouterBlockSize                  uint32 = 256
	hipMoELazyLaunchArgsVersion            uint32 = 1
	hipMoELazyLaunchArgsBytes                     = 64
	hipMoEBatchRouteRowsLaunchArgsVersion  uint32 = 1
	hipMoEBatchRouteRowsLaunchArgsBytes           = 72
	hipMoEBatchReduceLaunchArgsVersion     uint32 = 1
	hipMoEBatchReduceLaunchArgsBytes              = 56
	hipMoEBatchRouteMetadataBytes                 = 16
	hipMoEBatchRouteBlockSize              uint32 = 256
	hipMoEMLXAffineRoutesLaunchArgsVersion uint32 = 1
	hipMoEMLXAffineRoutesLaunchArgsBytes          = 80
	hipMoEMLXAffineRouteChunkBytes                = 152
	hipMoEMLXAffineRoutesPerChunk                 = 8
	hipMoEMLXAffineRouteRowsPerBlock              = 8
	hipMoEMLXAffineRoutesFlagGateUp        uint32 = 1
)

type hipMoERouterRequest struct {
	Logits []float32
	TopK   int
	Layer  int
}

type hipMoERouterDeviceBuffers struct {
	Logits         *hipDeviceByteBuffer
	Output         *hipDeviceByteBuffer
	IDs            *hipDeviceByteBuffer
	Probs          *hipDeviceByteBuffer
	Status         *hipDeviceByteBuffer
	InputLogits    []float32
	ExpertCount    int
	TopK           int
	Layer          int
	BorrowedLogits bool
}

type hipMoERouterLaunchArgs struct {
	LogitPointer  nativeDevicePointer
	IDPointer     nativeDevicePointer
	ProbPointer   nativeDevicePointer
	StatusPointer nativeDevicePointer
	ExpertCount   int
	TopK          int
	Layer         int
	LogitBytes    uint64
	IDBytes       uint64
	ProbBytes     uint64
}

type hipMoERouterResult struct {
	Routes []rocmExpertRoute
	Layer  int
	Status uint32
}

type hipMoELazyExpertRequest struct {
	ExpertIDs    []int32
	TotalExperts int
}

type hipMoELazyExpertDeviceBuffers struct {
	IDs          *hipDeviceByteBuffer
	Resident     *hipDeviceByteBuffer
	Selected     int
	TotalExperts int
}

type hipMoELazyExpertLaunchArgs struct {
	IDPointer       nativeDevicePointer
	ResidentPointer nativeDevicePointer
	SelectedCount   int
	TotalExperts    int
	IDBytes         uint64
	ResidentBytes   uint64
}

type hipMoELazyExpertResult struct {
	Resident []bool
}

type hipMoEBatchRouteRowsLaunchArgs struct {
	InputPointer    nativeDevicePointer
	MetadataPointer nativeDevicePointer
	OutputPointer   nativeDevicePointer
	RouteCount      int
	RowWidth        int
	InputRows       int
	OutputRows      int
	InputBytes      uint64
	MetadataBytes   uint64
	OutputBytes     uint64
}

type hipMoEBatchReduceLaunchArgs struct {
	InputPointer  nativeDevicePointer
	OutputPointer nativeDevicePointer
	Rows          int
	TopK          int
	RowWidth      int
	InputBytes    uint64
	OutputBytes   uint64
}

type hipMoEMLXAffineRouteChunk struct {
	GateUpWeightPointer nativeDevicePointer
	GateUpScalePointer  nativeDevicePointer
	GateUpBiasPointer   nativeDevicePointer
	DownWeightPointer   nativeDevicePointer
	DownScalePointer    nativeDevicePointer
	DownBiasPointer     nativeDevicePointer
	RouteCount          int
	TokenRows           [hipMoEMLXAffineRoutesPerChunk]int
	PairIndices         [hipMoEMLXAffineRoutesPerChunk]int
	RouteWeights        [hipMoEMLXAffineRoutesPerChunk]float32
}

type hipMoEMLXAffineRoutesLaunchArgs struct {
	InputPointer  nativeDevicePointer
	ChunkPointer  nativeDevicePointer
	OutputPointer nativeDevicePointer
	Rows          int
	Cols          int
	InputRows     int
	PairCount     int
	ChunkCount    int
	GroupSize     int
	Bits          int
	InputBytes    uint64
	ChunkBytes    uint64
	OutputBytes   uint64
	GateUp        bool
}

func (chunk hipMoEMLXAffineRouteChunk) Binary() ([]byte, error) {
	return chunk.BinaryInto(make([]byte, hipMoEMLXAffineRouteChunkBytes))
}

func (chunk hipMoEMLXAffineRouteChunk) BinaryInto(payload []byte) ([]byte, error) {
	const operation = "rocm.hip.MoEMLXAffineRouteChunk"
	if len(payload) < hipMoEMLXAffineRouteChunkBytes {
		return nil, core.E(operation, "route chunk payload buffer is too small", nil)
	}
	if chunk.GateUpWeightPointer == 0 || chunk.GateUpScalePointer == 0 || chunk.GateUpBiasPointer == 0 ||
		chunk.DownWeightPointer == 0 || chunk.DownScalePointer == 0 || chunk.DownBiasPointer == 0 {
		return nil, core.E(operation, "gate/up and down projection pointers are required", nil)
	}
	if chunk.RouteCount <= 0 || chunk.RouteCount > hipMoEMLXAffineRoutesPerChunk {
		return nil, core.E(operation, "route count must fit one route chunk", nil)
	}
	payload = payload[:hipMoEMLXAffineRouteChunkBytes]
	clear(payload)
	binary.LittleEndian.PutUint64(payload[0:], uint64(chunk.GateUpWeightPointer))
	binary.LittleEndian.PutUint64(payload[8:], uint64(chunk.GateUpScalePointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(chunk.GateUpBiasPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(chunk.DownWeightPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(chunk.DownScalePointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(chunk.DownBiasPointer))
	binary.LittleEndian.PutUint32(payload[48:], uint32(chunk.RouteCount))
	for index := 0; index < chunk.RouteCount; index++ {
		if chunk.TokenRows[index] < 0 || uint64(chunk.TokenRows[index]) > uint64(^uint32(0)) ||
			chunk.PairIndices[index] < 0 || uint64(chunk.PairIndices[index]) > uint64(^uint32(0)) {
			return nil, core.E(operation, "token row or pair index exceeds uint32", nil)
		}
		weight := chunk.RouteWeights[index]
		if math.IsNaN(float64(weight)) || math.IsInf(float64(weight), 0) {
			return nil, core.E(operation, "route weights must be finite", nil)
		}
		binary.LittleEndian.PutUint32(payload[56+index*4:], uint32(chunk.TokenRows[index]))
		binary.LittleEndian.PutUint32(payload[88+index*4:], uint32(chunk.PairIndices[index]))
		binary.LittleEndian.PutUint32(payload[120+index*4:], math.Float32bits(weight))
	}
	return payload, nil
}

func hipMoEMLXAffineRouteChunksBinaryInto(chunks []hipMoEMLXAffineRouteChunk, payload []byte) ([]byte, error) {
	const operation = "rocm.hip.MoEMLXAffineRouteChunks"
	if len(chunks) == 0 {
		return nil, core.E(operation, "at least one route chunk is required", nil)
	}
	if len(chunks) > int(^uint32(0))/hipMoEMLXAffineRouteChunkBytes {
		return nil, core.E(operation, "route chunk byte count exceeds uint32", nil)
	}
	want := len(chunks) * hipMoEMLXAffineRouteChunkBytes
	if len(payload) < want {
		return nil, core.E(operation, "route chunk payload buffer is too small", nil)
	}
	payload = payload[:want]
	for index, chunk := range chunks {
		if _, err := chunk.BinaryInto(payload[index*hipMoEMLXAffineRouteChunkBytes:]); err != nil {
			return nil, core.E(operation, core.Sprintf("encode route chunk %d", index), err)
		}
	}
	return payload, nil
}

func (args hipMoEMLXAffineRoutesLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(make([]byte, hipMoEMLXAffineRoutesLaunchArgsBytes))
}

func (args hipMoEMLXAffineRoutesLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	const operation = "rocm.hip.MoEMLXAffineRoutesLaunch"
	if len(payload) < hipMoEMLXAffineRoutesLaunchArgsBytes {
		return nil, core.E(operation, "launch arg payload buffer is too small", nil)
	}
	if args.InputPointer == 0 || args.ChunkPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E(operation, "input, route chunk, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("MoE affine route rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("MoE affine route cols", args.Cols)
	if err != nil {
		return nil, err
	}
	inputRows, err := rocmDeviceKVPositiveUint32("MoE affine route input rows", args.InputRows)
	if err != nil {
		return nil, err
	}
	pairCount, err := rocmDeviceKVPositiveUint32("MoE affine route pair count", args.PairCount)
	if err != nil {
		return nil, err
	}
	chunkCount, err := rocmDeviceKVPositiveUint32("MoE affine route chunk count", args.ChunkCount)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("MoE affine route group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("MoE affine route bits", args.Bits)
	if err != nil {
		return nil, err
	}
	if !hipMLXAffineSupportedBits(args.Bits) || args.GroupSize > args.Cols || args.Cols%args.GroupSize != 0 || (args.Cols*args.Bits)%32 != 0 {
		return nil, core.E(operation, "MLX affine route quantization geometry is invalid", nil)
	}
	wantInputBytes, err := hipMoEByteProduct(operation, "input", uint64(inputRows), uint64(cols), 4)
	if err != nil {
		return nil, err
	}
	wantChunkBytes, err := hipMoEByteProduct(operation, "route chunk", uint64(chunkCount), hipMoEMLXAffineRouteChunkBytes)
	if err != nil {
		return nil, err
	}
	wantOutputBytes, err := hipMoEByteProduct(operation, "output", uint64(pairCount), uint64(rows), 4)
	if err != nil {
		return nil, err
	}
	if args.InputBytes != wantInputBytes || args.ChunkBytes != wantChunkBytes || args.OutputBytes != wantOutputBytes {
		return nil, core.E(operation, "input, route chunk, or output byte count mismatch", nil)
	}
	for label, value := range map[string]uint64{
		"input bytes": args.InputBytes, "route chunk bytes": args.ChunkBytes, "output bytes": args.OutputBytes,
	} {
		if err := hipProjectionUint32Bytes(operation, label, value); err != nil {
			return nil, err
		}
	}
	flags := uint32(0)
	if args.GateUp {
		flags = hipMoEMLXAffineRoutesFlagGateUp
	}
	payload = payload[:hipMoEMLXAffineRoutesLaunchArgsBytes]
	clear(payload)
	binary.LittleEndian.PutUint32(payload[0:], hipMoEMLXAffineRoutesLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.ChunkPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], rows)
	binary.LittleEndian.PutUint32(payload[36:], cols)
	binary.LittleEndian.PutUint32(payload[40:], inputRows)
	binary.LittleEndian.PutUint32(payload[44:], pairCount)
	binary.LittleEndian.PutUint32(payload[48:], chunkCount)
	binary.LittleEndian.PutUint32(payload[52:], groupSize)
	binary.LittleEndian.PutUint32(payload[56:], bits)
	binary.LittleEndian.PutUint32(payload[60:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[64:], uint32(args.ChunkBytes))
	binary.LittleEndian.PutUint32(payload[68:], uint32(args.OutputBytes))
	binary.LittleEndian.PutUint32(payload[72:], flags)
	return payload, nil
}

func hipMoEByteProduct(operation, label string, factors ...uint64) (uint64, error) {
	value := uint64(1)
	for _, factor := range factors {
		if factor != 0 && value > ^uint64(0)/factor {
			return 0, core.E(operation, label+" byte count overflows", nil)
		}
		value *= factor
	}
	return value, nil
}

func hipRunMoEMLXAffineRoutesKernel(ctx context.Context, driver nativeHIPDriver, input, chunks *hipDeviceByteBuffer, rows, cols, inputRows, pairCount, chunkCount, groupSize, bits int, gateUp bool, output *hipDeviceByteBuffer) error {
	return hipRunMoEMLXAffineRoutesKernelWithArgs(ctx, driver, input, chunks, rows, cols, inputRows, pairCount, chunkCount, groupSize, bits, gateUp, output, nil)
}

func hipRunMoEMLXAffineRoutesKernelWithArgs(ctx context.Context, driver nativeHIPDriver, input, chunks *hipDeviceByteBuffer, rows, cols, inputRows, pairCount, chunkCount, groupSize, bits int, gateUp bool, output *hipDeviceByteBuffer, packet []byte) error {
	const operation = "rocm.hip.MoEMLXAffineRoutesLaunch"
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if input == nil || chunks == nil || output == nil || input.Pointer() == 0 || chunks.Pointer() == 0 || output.Pointer() == 0 {
		return core.E(operation, "input, route chunk, and output buffers are required", nil)
	}
	if rows <= 0 || cols <= 0 || inputRows <= 0 || pairCount <= 0 || chunkCount <= 0 ||
		input.Count() != inputRows*cols || input.SizeBytes() != uint64(inputRows*cols*4) ||
		chunks.SizeBytes() != uint64(chunkCount*hipMoEMLXAffineRouteChunkBytes) ||
		output.Count() != pairCount*rows || output.SizeBytes() != uint64(pairCount*rows*4) {
		return core.E(operation, "MLX affine route buffer shape mismatch", nil)
	}
	launch := hipMoEMLXAffineRoutesLaunchArgs{
		InputPointer: input.Pointer(), ChunkPointer: chunks.Pointer(), OutputPointer: output.Pointer(),
		Rows: rows, Cols: cols, InputRows: inputRows, PairCount: pairCount, ChunkCount: chunkCount,
		GroupSize: groupSize, Bits: bits, InputBytes: input.SizeBytes(), ChunkBytes: chunks.SizeBytes(),
		OutputBytes: output.SizeBytes(), GateUp: gateUp,
	}
	var args []byte
	var err error
	if len(packet) >= hipMoEMLXAffineRoutesLaunchArgsBytes {
		args, err = launch.BinaryInto(packet)
	} else {
		args, err = launch.Binary()
	}
	if err != nil {
		return err
	}
	gridX := (uint64(rows) + hipMoEMLXAffineRouteRowsPerBlock - 1) / hipMoEMLXAffineRouteRowsPerBlock
	if gridX == 0 || gridX > uint64(^uint32(0)) || uint64(chunkCount) > uint64(^uint32(0)) {
		return core.E(operation, "MLX affine route grid exceeds uint32", nil)
	}
	return hipLaunchKernelContext(ctx, driver, hipKernelLaunchConfig{
		Name: hipKernelNameMoEMLXAffineRoutes, Args: args,
		GridX: uint32(gridX), GridY: uint32(chunkCount), GridZ: 1,
		BlockX: hipMoEBatchRouteBlockSize, BlockY: 1, BlockZ: 1,
	})
}

func (args hipMoEBatchRouteRowsLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(make([]byte, hipMoEBatchRouteRowsLaunchArgsBytes))
}

func (args hipMoEBatchRouteRowsLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	const operation = "rocm.hip.MoEBatchRouteRowsLaunch"
	if len(payload) < hipMoEBatchRouteRowsLaunchArgsBytes {
		return nil, core.E(operation, "launch arg payload buffer is too small", nil)
	}
	if args.InputPointer == 0 || args.MetadataPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E(operation, "input, metadata, and output pointers are required", nil)
	}
	routeCount, err := rocmDeviceKVPositiveUint32("MoE batch route count", args.RouteCount)
	if err != nil {
		return nil, err
	}
	rowWidth, err := rocmDeviceKVPositiveUint32("MoE batch route row width", args.RowWidth)
	if err != nil {
		return nil, err
	}
	inputRows, err := rocmDeviceKVPositiveUint32("MoE batch route input rows", args.InputRows)
	if err != nil {
		return nil, err
	}
	outputRows, err := rocmDeviceKVPositiveUint32("MoE batch route output rows", args.OutputRows)
	if err != nil {
		return nil, err
	}
	wantInputBytes := uint64(inputRows) * uint64(rowWidth) * 4
	wantMetadataBytes := uint64(routeCount) * hipMoEBatchRouteMetadataBytes
	wantOutputBytes := uint64(outputRows) * uint64(rowWidth) * 4
	if args.InputBytes != wantInputBytes || args.MetadataBytes != wantMetadataBytes || args.OutputBytes != wantOutputBytes {
		return nil, core.E(operation, "input, metadata, or output byte count mismatch", nil)
	}
	payload = payload[:hipMoEBatchRouteRowsLaunchArgsBytes]
	clear(payload)
	binary.LittleEndian.PutUint32(payload[0:], hipMoEBatchRouteRowsLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.MetadataPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], routeCount)
	binary.LittleEndian.PutUint32(payload[36:], rowWidth)
	binary.LittleEndian.PutUint32(payload[40:], inputRows)
	binary.LittleEndian.PutUint32(payload[44:], outputRows)
	binary.LittleEndian.PutUint64(payload[48:], args.InputBytes)
	binary.LittleEndian.PutUint64(payload[56:], args.MetadataBytes)
	binary.LittleEndian.PutUint64(payload[64:], args.OutputBytes)
	return payload, nil
}

func (args hipMoEBatchReduceLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(make([]byte, hipMoEBatchReduceLaunchArgsBytes))
}

func (args hipMoEBatchReduceLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	const operation = "rocm.hip.MoEBatchReduceRoutesLaunch"
	if len(payload) < hipMoEBatchReduceLaunchArgsBytes {
		return nil, core.E(operation, "launch arg payload buffer is too small", nil)
	}
	if args.InputPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E(operation, "input and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("MoE batch reduce rows", args.Rows)
	if err != nil {
		return nil, err
	}
	topK, err := rocmDeviceKVPositiveUint32("MoE batch reduce top-k", args.TopK)
	if err != nil {
		return nil, err
	}
	rowWidth, err := rocmDeviceKVPositiveUint32("MoE batch reduce row width", args.RowWidth)
	if err != nil {
		return nil, err
	}
	wantInputBytes := uint64(rows) * uint64(topK) * uint64(rowWidth) * 4
	wantOutputBytes := uint64(rows) * uint64(rowWidth) * 4
	if args.InputBytes != wantInputBytes || args.OutputBytes != wantOutputBytes {
		return nil, core.E(operation, "input or output byte count mismatch", nil)
	}
	payload = payload[:hipMoEBatchReduceLaunchArgsBytes]
	clear(payload)
	binary.LittleEndian.PutUint32(payload[0:], hipMoEBatchReduceLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[24:], rows)
	binary.LittleEndian.PutUint32(payload[28:], topK)
	binary.LittleEndian.PutUint32(payload[32:], rowWidth)
	binary.LittleEndian.PutUint64(payload[40:], args.InputBytes)
	binary.LittleEndian.PutUint64(payload[48:], args.OutputBytes)
	return payload, nil
}

func hipRunMoEBatchGatherRowsKernel(ctx context.Context, driver nativeHIPDriver, input, metadata *hipDeviceByteBuffer, routeCount, rowWidth, sourceRows int, output *hipDeviceByteBuffer) error {
	return hipRunMoEBatchGatherRowsKernelWithArgs(ctx, driver, input, metadata, routeCount, rowWidth, sourceRows, output, nil)
}

func hipRunMoEBatchGatherRowsKernelWithArgs(ctx context.Context, driver nativeHIPDriver, input, metadata *hipDeviceByteBuffer, routeCount, rowWidth, sourceRows int, output *hipDeviceByteBuffer, packet []byte) error {
	const operation = "rocm.hip.MoEBatchGatherRowsLaunch"
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if input == nil || metadata == nil || output == nil || input.Pointer() == 0 || metadata.Pointer() == 0 || output.Pointer() == 0 {
		return core.E(operation, "input, metadata, and output buffers are required", nil)
	}
	if routeCount <= 0 || rowWidth <= 0 || sourceRows <= 0 ||
		input.Count() != sourceRows*rowWidth || input.SizeBytes() != uint64(sourceRows*rowWidth*4) ||
		metadata.SizeBytes() != uint64(routeCount*hipMoEBatchRouteMetadataBytes) ||
		output.Count() != routeCount*rowWidth || output.SizeBytes() != uint64(routeCount*rowWidth*4) {
		return core.E(operation, "batch gather buffer shape mismatch", nil)
	}
	launchArgs := hipMoEBatchRouteRowsLaunchArgs{
		InputPointer: input.Pointer(), MetadataPointer: metadata.Pointer(), OutputPointer: output.Pointer(),
		RouteCount: routeCount, RowWidth: rowWidth, InputRows: sourceRows, OutputRows: routeCount,
		InputBytes: input.SizeBytes(), MetadataBytes: metadata.SizeBytes(), OutputBytes: output.SizeBytes(),
	}
	var args []byte
	var err error
	if len(packet) >= hipMoEBatchRouteRowsLaunchArgsBytes {
		args, err = launchArgs.BinaryInto(packet)
	} else {
		args, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	return hipLaunchMoEBatchRouteKernel(driver, hipKernelNameMoEBatchGatherRows, args, routeCount, rowWidth)
}

func hipRunMoEBatchScatterRoutesKernel(ctx context.Context, driver nativeHIPDriver, input, metadata *hipDeviceByteBuffer, routeCount, rowWidth, pairCount int, output *hipDeviceByteBuffer) error {
	return hipRunMoEBatchScatterRoutesKernelWithArgs(ctx, driver, input, metadata, routeCount, rowWidth, pairCount, output, nil)
}

func hipRunMoEBatchScatterRoutesKernelWithArgs(ctx context.Context, driver nativeHIPDriver, input, metadata *hipDeviceByteBuffer, routeCount, rowWidth, pairCount int, output *hipDeviceByteBuffer, packet []byte) error {
	const operation = "rocm.hip.MoEBatchScatterRoutesLaunch"
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if input == nil || metadata == nil || output == nil || input.Pointer() == 0 || metadata.Pointer() == 0 || output.Pointer() == 0 {
		return core.E(operation, "input, metadata, and output buffers are required", nil)
	}
	if routeCount <= 0 || rowWidth <= 0 || pairCount <= 0 ||
		input.Count() != routeCount*rowWidth || input.SizeBytes() != uint64(routeCount*rowWidth*4) ||
		metadata.SizeBytes() != uint64(routeCount*hipMoEBatchRouteMetadataBytes) ||
		output.Count() != pairCount*rowWidth || output.SizeBytes() != uint64(pairCount*rowWidth*4) {
		return core.E(operation, "batch scatter buffer shape mismatch", nil)
	}
	launchArgs := hipMoEBatchRouteRowsLaunchArgs{
		InputPointer: input.Pointer(), MetadataPointer: metadata.Pointer(), OutputPointer: output.Pointer(),
		RouteCount: routeCount, RowWidth: rowWidth, InputRows: routeCount, OutputRows: pairCount,
		InputBytes: input.SizeBytes(), MetadataBytes: metadata.SizeBytes(), OutputBytes: output.SizeBytes(),
	}
	var args []byte
	var err error
	if len(packet) >= hipMoEBatchRouteRowsLaunchArgsBytes {
		args, err = launchArgs.BinaryInto(packet)
	} else {
		args, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	return hipLaunchMoEBatchRouteKernel(driver, hipKernelNameMoEBatchScatterRoutes, args, routeCount, rowWidth)
}

func hipRunMoEBatchReduceRoutesKernel(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, rows, topK, rowWidth int, output *hipDeviceByteBuffer) error {
	return hipRunMoEBatchReduceRoutesKernelWithArgs(ctx, driver, input, rows, topK, rowWidth, output, nil)
}

func hipRunMoEBatchReduceRoutesKernelWithArgs(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, rows, topK, rowWidth int, output *hipDeviceByteBuffer, packet []byte) error {
	const operation = "rocm.hip.MoEBatchReduceRoutesLaunch"
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if input == nil || output == nil || input.Pointer() == 0 || output.Pointer() == 0 {
		return core.E(operation, "input and output buffers are required", nil)
	}
	if rows <= 0 || topK <= 0 || rowWidth <= 0 ||
		input.Count() != rows*topK*rowWidth || input.SizeBytes() != uint64(rows*topK*rowWidth*4) ||
		output.Count() != rows*rowWidth || output.SizeBytes() != uint64(rows*rowWidth*4) {
		return core.E(operation, "batch route reduction buffer shape mismatch", nil)
	}
	launchArgs := hipMoEBatchReduceLaunchArgs{
		InputPointer: input.Pointer(), OutputPointer: output.Pointer(),
		Rows: rows, TopK: topK, RowWidth: rowWidth,
		InputBytes: input.SizeBytes(), OutputBytes: output.SizeBytes(),
	}
	var args []byte
	var err error
	if len(packet) >= hipMoEBatchReduceLaunchArgsBytes {
		args, err = launchArgs.BinaryInto(packet)
	} else {
		args, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	return hipLaunchMoEBatchRouteKernel(driver, hipKernelNameMoEBatchReduceRoutes, args, rows, rowWidth)
}

func hipLaunchMoEBatchRouteKernel(driver nativeHIPDriver, name string, args []byte, rows, rowWidth int) error {
	if rows <= 0 || rowWidth <= 0 {
		return core.E("rocm.hip.MoEBatchRouteLaunch", "route kernel geometry must be positive", nil)
	}
	workItems := uint64(rows) * uint64(rowWidth)
	grid := (workItems + uint64(hipMoEBatchRouteBlockSize) - 1) / uint64(hipMoEBatchRouteBlockSize)
	if grid == 0 || grid > uint64(^uint32(0)) {
		return core.E("rocm.hip.MoEBatchRouteLaunch", "route kernel grid exceeds uint32", nil)
	}
	return hipLaunchKernel(driver, hipKernelLaunchConfig{
		Name: name, Args: args,
		GridX: uint32(grid), GridY: 1, GridZ: 1,
		BlockX: hipMoEBatchRouteBlockSize, BlockY: 1, BlockZ: 1,
	})
}

func (req hipMoERouterRequest) validate() error {
	if len(req.Logits) == 0 {
		return core.E("rocm.hip.MoERouterLaunch", "router logits are required", nil)
	}
	if req.TopK <= 0 || req.TopK > len(req.Logits) {
		return core.E("rocm.hip.MoERouterLaunch", "top-k must be within the expert count", nil)
	}
	if !rocmFloat32SliceFinite(req.Logits) {
		return core.E("rocm.hip.MoERouterLaunch", "router logits must be finite", nil)
	}
	if req.Layer < 0 {
		return core.E("rocm.hip.MoERouterLaunch", "layer must be non-negative", nil)
	}
	return nil
}

func (req hipMoERouterRequest) deviceBuffers(driver nativeHIPDriver) (*hipMoERouterDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	logitPayload, err := hipFloat32Payload(req.Logits)
	if err != nil {
		return nil, core.E("rocm.hip.MoERouterLaunch", "encode router logits", err)
	}
	logits, err := hipUploadByteBuffer(driver, "rocm.hip.MoERouterLaunch", "router logits", logitPayload, len(req.Logits))
	if err != nil {
		return nil, err
	}
	buffers := &hipMoERouterDeviceBuffers{
		Logits:      logits,
		InputLogits: append([]float32(nil), req.Logits...),
		ExpertCount: len(req.Logits),
		TopK:        req.TopK,
		Layer:       req.Layer,
	}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	if err := buffers.allocateOutput(driver); err != nil {
		return nil, err
	}
	success = true
	return buffers, nil
}

func (buffers *hipMoERouterDeviceBuffers) allocateOutput(driver nativeHIPDriver) error {
	if buffers == nil || buffers.TopK <= 0 {
		return core.E("rocm.hip.MoERouterLaunch", "router output geometry is required", nil)
	}
	idBytes := uint64(buffers.TopK * 4)
	probBytes := uint64(buffers.TopK * 4)
	totalBytes := idBytes + probBytes + 4
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MoERouterLaunch", "packed router output", totalBytes, int(totalBytes))
	if err != nil {
		return err
	}
	buffers.Output = output
	ids := hipBorrowDeviceByteBufferValue(driver, "router id output", output.Pointer(), idBytes, buffers.TopK)
	probs := hipBorrowDeviceByteBufferValue(driver, "router probability output", output.Pointer()+nativeDevicePointer(idBytes), probBytes, buffers.TopK)
	status := hipBorrowDeviceByteBufferValue(driver, "router status", output.Pointer()+nativeDevicePointer(idBytes+probBytes), 4, 1)
	buffers.IDs = &ids
	buffers.Probs = &probs
	buffers.Status = &status
	return nil
}

func (req hipMoERouterRequest) launchArgs(buffers *hipMoERouterDeviceBuffers) (hipMoERouterLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipMoERouterLaunchArgs{}, err
	}
	if buffers == nil || buffers.Logits == nil || buffers.IDs == nil || buffers.Probs == nil || buffers.Status == nil {
		return hipMoERouterLaunchArgs{}, core.E("rocm.hip.MoERouterLaunch", "router device buffers are required", nil)
	}
	if buffers.ExpertCount != len(req.Logits) || buffers.TopK != req.TopK ||
		buffers.Logits.Count() != len(req.Logits) || buffers.IDs.Count() != req.TopK || buffers.Probs.Count() != req.TopK ||
		buffers.Status.Count() != 1 || buffers.Logits.SizeBytes() != uint64(len(req.Logits)*4) ||
		buffers.IDs.SizeBytes() != uint64(req.TopK*4) || buffers.Probs.SizeBytes() != uint64(req.TopK*4) ||
		buffers.Status.SizeBytes() != 4 {
		return hipMoERouterLaunchArgs{}, core.E("rocm.hip.MoERouterLaunch", "router device buffer shape mismatch", nil)
	}
	return hipMoERouterLaunchArgs{
		LogitPointer:  buffers.Logits.Pointer(),
		IDPointer:     buffers.IDs.Pointer(),
		ProbPointer:   buffers.Probs.Pointer(),
		StatusPointer: buffers.Status.Pointer(),
		ExpertCount:   len(req.Logits),
		TopK:          req.TopK,
		Layer:         req.Layer,
		LogitBytes:    buffers.Logits.SizeBytes(),
		IDBytes:       buffers.IDs.SizeBytes(),
		ProbBytes:     buffers.Probs.SizeBytes(),
	}, nil
}

func (args hipMoERouterLaunchArgs) Binary() ([]byte, error) {
	payload := make([]byte, hipMoERouterLaunchArgsBytes)
	return args.BinaryInto(payload)
}

func (args hipMoERouterLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.LogitPointer == 0 || args.IDPointer == 0 || args.ProbPointer == 0 {
		return nil, core.E("rocm.hip.MoERouterLaunch", "router logits and output pointers are required", nil)
	}
	if len(payload) < hipMoERouterLaunchArgsBytes {
		return nil, core.E("rocm.hip.MoERouterLaunch", "launch arg payload buffer is too small", nil)
	}
	payload = payload[:hipMoERouterLaunchArgsBytes]
	expertCount, err := rocmDeviceKVPositiveUint32("expert count", args.ExpertCount)
	if err != nil {
		return nil, err
	}
	topK, err := rocmDeviceKVPositiveUint32("top-k", args.TopK)
	if err != nil {
		return nil, err
	}
	if topK > expertCount {
		return nil, core.E("rocm.hip.MoERouterLaunch", "top-k must be within the expert count", nil)
	}
	if args.Layer < 0 {
		return nil, core.E("rocm.hip.MoERouterLaunch", "layer must be non-negative", nil)
	}
	logitBytes, err := hipAlignedFloat32Bytes("router logits", args.LogitBytes, expertCount)
	if err != nil {
		return nil, core.E("rocm.hip.MoERouterLaunch", "logit byte count", err)
	}
	idBytes, err := hipAlignedFloat32Bytes("router ids", args.IDBytes, topK)
	if err != nil {
		return nil, core.E("rocm.hip.MoERouterLaunch", "id byte count", err)
	}
	probBytes, err := hipAlignedFloat32Bytes("router probabilities", args.ProbBytes, topK)
	if err != nil {
		return nil, core.E("rocm.hip.MoERouterLaunch", "probability byte count", err)
	}
	if args.StatusPointer == 0 {
		return nil, core.E("rocm.hip.MoERouterLaunch", "router status pointer is required", nil)
	}
	if uint64(args.Layer) > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip.MoERouterLaunch", "layer exceeds uint32", nil)
	}
	layer := uint32(args.Layer)
	binary.LittleEndian.PutUint32(payload[0:], hipMoERouterLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.LogitPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.IDPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ProbPointer))
	binary.LittleEndian.PutUint32(payload[32:], expertCount)
	binary.LittleEndian.PutUint32(payload[36:], topK)
	binary.LittleEndian.PutUint32(payload[40:], logitBytes)
	binary.LittleEndian.PutUint32(payload[44:], idBytes)
	binary.LittleEndian.PutUint32(payload[48:], probBytes)
	binary.LittleEndian.PutUint32(payload[52:], layer)
	binary.LittleEndian.PutUint64(payload[56:], uint64(args.StatusPointer))
	return payload, nil
}

func (buffers *hipMoERouterDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	if err := buffers.Output.Close(); err != nil {
		lastErr = err
	}
	if !buffers.BorrowedLogits {
		if err := buffers.Logits.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipMoERouterDeviceBuffers) ReadOutput() (hipMoERouterResult, error) {
	if buffers == nil {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "router output buffers are required", nil)
	}
	if buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "packed router output buffer is required", nil)
	}
	routes := make([]rocmExpertRoute, buffers.TopK)
	payload := make([]byte, int(buffers.Output.SizeBytes()))
	return buffers.ReadPackedOutputInto(routes, payload)
}

func (buffers *hipMoERouterDeviceBuffers) ReadPackedOutputInto(routes []rocmExpertRoute, payload []byte) (hipMoERouterResult, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "packed router output buffer is required", nil)
	}
	payloadBytes := int(buffers.Output.SizeBytes())
	if len(routes) < buffers.TopK || len(payload) < payloadBytes {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "packed router result buffer is too small", nil)
	}
	payload = payload[:payloadBytes]
	if err := buffers.Output.driver.CopyDeviceToHost(buffers.Output.Pointer(), payload); err != nil {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "copy packed router output", err)
	}
	idBytes := buffers.TopK * 4
	probBytes := buffers.TopK * 4
	return buffers.parseOutput(routes, payload[:idBytes], payload[idBytes:idBytes+probBytes], payload[idBytes+probBytes:])
}

func (buffers *hipMoERouterDeviceBuffers) ReadOutputInto(routes []rocmExpertRoute, idPayload, probPayload, statusPayload []byte) (hipMoERouterResult, error) {
	if buffers == nil || buffers.IDs == nil || buffers.Probs == nil || buffers.Status == nil {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "router output buffers are required", nil)
	}
	if len(routes) < buffers.TopK {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "router result buffer is too small", nil)
	}
	idBytes := int(buffers.IDs.SizeBytes())
	probBytes := int(buffers.Probs.SizeBytes())
	statusBytes := int(buffers.Status.SizeBytes())
	if len(idPayload) < idBytes || len(probPayload) < probBytes || len(statusPayload) < statusBytes {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "router output payload buffer is too small", nil)
	}
	idPayload = idPayload[:idBytes]
	probPayload = probPayload[:probBytes]
	statusPayload = statusPayload[:statusBytes]
	if err := buffers.IDs.driver.CopyDeviceToHost(buffers.IDs.Pointer(), idPayload); err != nil {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "copy router id output", err)
	}
	if err := buffers.Probs.driver.CopyDeviceToHost(buffers.Probs.Pointer(), probPayload); err != nil {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "copy router probability output", err)
	}
	if err := buffers.Status.driver.CopyDeviceToHost(buffers.Status.Pointer(), statusPayload); err != nil {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "copy router status", err)
	}
	return buffers.parseOutput(routes, idPayload, probPayload, statusPayload)
}

func (buffers *hipMoERouterDeviceBuffers) parseOutput(routes []rocmExpertRoute, idPayload, probPayload, statusPayload []byte) (hipMoERouterResult, error) {
	if len(idPayload) != buffers.TopK*4 || len(probPayload) != buffers.TopK*4 || len(statusPayload) != 4 {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "router output byte count mismatch", nil)
	}
	status := binary.LittleEndian.Uint32(statusPayload)
	if status != hipMoERouterLaunchStatusOK {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", core.Sprintf("router status marker mismatch: got 0x%08x want 0x%08x", status, hipMoERouterLaunchStatusOK), nil)
	}
	routes = routes[:buffers.TopK]
	for index := range routes {
		id := int(int32(binary.LittleEndian.Uint32(idPayload[index*4:])))
		if id < 0 || id >= buffers.ExpertCount {
			return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", core.Sprintf("router expert id %d outside expert count %d", id, buffers.ExpertCount), nil)
		}
		prob := math.Float32frombits(binary.LittleEndian.Uint32(probPayload[index*4:]))
		if math.IsNaN(float64(prob)) || math.IsInf(float64(prob), 0) || prob < 0 || prob > 1 {
			return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "router probability must be finite and within [0,1]", nil)
		}
		score := float32(0)
		if len(buffers.InputLogits) == buffers.ExpertCount {
			score = buffers.InputLogits[id]
		}
		routes[index] = rocmExpertRoute{ID: id, Score: score, Prob: prob}
	}
	return hipMoERouterResult{
		Routes: routes,
		Layer:  buffers.Layer,
		Status: status,
	}, nil
}

func hipRunMoERouterKernel(ctx context.Context, driver nativeHIPDriver, req hipMoERouterRequest) (hipMoERouterResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipMoERouterResult{}, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return hipMoERouterResult{}, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return hipMoERouterResult{}, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return hipMoERouterResult{}, err
	}
	config, err := hipSingleBlockLaunchConfig(hipKernelNameMoERouter, launchBytes, hipMoERouterBlockSize)
	if err != nil {
		return hipMoERouterResult{}, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return hipMoERouterResult{}, err
	}
	return buffers.ReadOutput()
}

func hipRunMoERouterKernelWithDeviceInput(ctx context.Context, driver nativeHIPDriver, logits *hipDeviceByteBuffer, topK, layer int) (hipMoERouterResult, error) {
	return hipRunMoERouterKernelWithDeviceInputWorkspace(ctx, driver, logits, topK, layer, nil)
}

func hipRunMoERouterKernelWithDeviceInputWorkspace(ctx context.Context, driver nativeHIPDriver, logits *hipDeviceByteBuffer, topK, layer int, workspace *hipAttentionHeadsChunkedWorkspace) (hipMoERouterResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipMoERouterResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "HIP driver is not available", nil)
	}
	if logits == nil || logits.Pointer() == 0 || logits.Count() <= 0 || logits.SizeBytes() != uint64(logits.Count()*4) {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "router logits device buffer shape mismatch", nil)
	}
	if topK <= 0 || topK > logits.Count() || layer < 0 {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "top-k and layer must fit the router geometry", nil)
	}
	var buffers *hipMoERouterDeviceBuffers
	if workspace != nil {
		var err error
		buffers, err = workspace.prepareMoERouterBuffers(driver, logits, topK, layer)
		if err != nil {
			return hipMoERouterResult{}, err
		}
	} else {
		buffers = &hipMoERouterDeviceBuffers{
			Logits: logits, ExpertCount: logits.Count(), TopK: topK, Layer: layer, BorrowedLogits: true,
		}
		defer buffers.Close()
		if err := buffers.allocateOutput(driver); err != nil {
			return hipMoERouterResult{}, err
		}
	}
	launchArgs := hipMoERouterLaunchArgs{
		LogitPointer:  logits.Pointer(),
		IDPointer:     buffers.IDs.Pointer(),
		ProbPointer:   buffers.Probs.Pointer(),
		StatusPointer: buffers.Status.Pointer(),
		ExpertCount:   logits.Count(),
		TopK:          topK,
		Layer:         layer,
		LogitBytes:    logits.SizeBytes(),
		IDBytes:       buffers.IDs.SizeBytes(),
		ProbBytes:     buffers.Probs.SizeBytes(),
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.MoE.RouterArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return hipMoERouterResult{}, err
	}
	config, err := hipSingleBlockLaunchConfig(hipKernelNameMoERouter, launchBytes, hipMoERouterBlockSize)
	if err != nil {
		return hipMoERouterResult{}, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return hipMoERouterResult{}, err
	}
	if workspace != nil {
		payloadBytes := topK*8 + 4
		return buffers.ReadPackedOutputInto(workspace.MoE.Routes[:topK], workspace.MoE.RouterPayload[:payloadBytes])
	}
	return buffers.ReadOutput()
}

func (req hipMoELazyExpertRequest) validate() error {
	if req.TotalExperts <= 0 {
		return core.E("rocm.hip.MoELazyLaunch", "expert count must be positive", nil)
	}
	if len(req.ExpertIDs) == 0 {
		return core.E("rocm.hip.MoELazyLaunch", "selected expert IDs are required", nil)
	}
	routes := make([]rocmExpertRoute, len(req.ExpertIDs))
	for index, id := range req.ExpertIDs {
		routes[index] = rocmExpertRoute{ID: int(id)}
	}
	if _, err := rocmReferenceLazyExpertResidency(routes, req.TotalExperts); err != nil {
		return err
	}
	return nil
}

func (req hipMoELazyExpertRequest) deviceBuffers(driver nativeHIPDriver) (*hipMoELazyExpertDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	idPayload := make([]byte, len(req.ExpertIDs)*4)
	for index, id := range req.ExpertIDs {
		binary.LittleEndian.PutUint32(idPayload[index*4:], uint32(id))
	}
	ids, err := hipUploadByteBuffer(driver, "rocm.hip.MoELazyLaunch", "selected expert IDs", idPayload, len(req.ExpertIDs))
	if err != nil {
		return nil, err
	}
	buffers := &hipMoELazyExpertDeviceBuffers{IDs: ids, Selected: len(req.ExpertIDs), TotalExperts: req.TotalExperts}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	resident, err := hipAllocateByteBuffer(driver, "rocm.hip.MoELazyLaunch", "resident expert output", uint64(req.TotalExperts), req.TotalExperts)
	if err != nil {
		return nil, err
	}
	buffers.Resident = resident
	success = true
	return buffers, nil
}

func (req hipMoELazyExpertRequest) launchArgs(buffers *hipMoELazyExpertDeviceBuffers) (hipMoELazyExpertLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipMoELazyExpertLaunchArgs{}, err
	}
	if buffers == nil || buffers.IDs == nil || buffers.Resident == nil {
		return hipMoELazyExpertLaunchArgs{}, core.E("rocm.hip.MoELazyLaunch", "lazy expert device buffers are required", nil)
	}
	if buffers.Selected != len(req.ExpertIDs) || buffers.TotalExperts != req.TotalExperts ||
		buffers.IDs.Count() != len(req.ExpertIDs) || buffers.Resident.Count() != req.TotalExperts {
		return hipMoELazyExpertLaunchArgs{}, core.E("rocm.hip.MoELazyLaunch", "lazy expert device buffer shape mismatch", nil)
	}
	return hipMoELazyExpertLaunchArgs{
		IDPointer:       buffers.IDs.Pointer(),
		ResidentPointer: buffers.Resident.Pointer(),
		SelectedCount:   len(req.ExpertIDs),
		TotalExperts:    req.TotalExperts,
		IDBytes:         buffers.IDs.SizeBytes(),
		ResidentBytes:   buffers.Resident.SizeBytes(),
	}, nil
}

func (args hipMoELazyExpertLaunchArgs) Binary() ([]byte, error) {
	payload := make([]byte, hipMoELazyLaunchArgsBytes)
	return args.BinaryInto(payload)
}

func (args hipMoELazyExpertLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.IDPointer == 0 || args.ResidentPointer == 0 {
		return nil, core.E("rocm.hip.MoELazyLaunch", "expert ID and resident output pointers are required", nil)
	}
	if len(payload) < hipMoELazyLaunchArgsBytes {
		return nil, core.E("rocm.hip.MoELazyLaunch", "launch arg payload buffer is too small", nil)
	}
	payload = payload[:hipMoELazyLaunchArgsBytes]
	selected, err := rocmDeviceKVPositiveUint32("selected expert count", args.SelectedCount)
	if err != nil {
		return nil, err
	}
	total, err := rocmDeviceKVPositiveUint32("expert count", args.TotalExperts)
	if err != nil {
		return nil, err
	}
	if args.IDBytes != uint64(selected)*4 {
		return nil, core.E("rocm.hip.MoELazyLaunch", "expert ID byte count mismatch", nil)
	}
	if args.ResidentBytes != uint64(total) {
		return nil, core.E("rocm.hip.MoELazyLaunch", "resident byte count mismatch", nil)
	}
	if args.IDBytes > uint64(^uint32(0)) || args.ResidentBytes > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip.MoELazyLaunch", "lazy expert byte counts are out of uint32 range", nil)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipMoELazyLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.IDPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.ResidentPointer))
	binary.LittleEndian.PutUint32(payload[24:], selected)
	binary.LittleEndian.PutUint32(payload[28:], total)
	binary.LittleEndian.PutUint32(payload[32:], uint32(args.IDBytes))
	binary.LittleEndian.PutUint32(payload[36:], uint32(args.ResidentBytes))
	return payload, nil
}

func (buffers *hipMoELazyExpertDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Resident, buffers.IDs} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipMoELazyExpertDeviceBuffers) ReadOutput() (hipMoELazyExpertResult, error) {
	if buffers == nil {
		return hipMoELazyExpertResult{}, core.E("rocm.hip.MoELazyLaunch", "resident expert output buffer is required", nil)
	}
	payload := make([]byte, buffers.Resident.SizeBytes())
	resident := make([]bool, buffers.TotalExperts)
	return buffers.ReadOutputInto(resident, payload)
}

func (buffers *hipMoELazyExpertDeviceBuffers) ReadOutputInto(resident []bool, payload []byte) (hipMoELazyExpertResult, error) {
	if buffers == nil || buffers.Resident == nil || buffers.Resident.Pointer() == 0 {
		return hipMoELazyExpertResult{}, core.E("rocm.hip.MoELazyLaunch", "resident expert output buffer is required", nil)
	}
	if buffers.TotalExperts <= 0 || buffers.Resident.Count() != buffers.TotalExperts || buffers.Resident.SizeBytes() != uint64(buffers.TotalExperts) {
		return hipMoELazyExpertResult{}, core.E("rocm.hip.MoELazyLaunch", "resident expert output byte count mismatch", nil)
	}
	payloadBytes := int(buffers.Resident.SizeBytes())
	if len(resident) < buffers.TotalExperts || len(payload) < payloadBytes {
		return hipMoELazyExpertResult{}, core.E("rocm.hip.MoELazyLaunch", "resident expert output buffer is too small", nil)
	}
	payload = payload[:payloadBytes]
	if err := buffers.Resident.driver.CopyDeviceToHost(buffers.Resident.Pointer(), payload); err != nil {
		return hipMoELazyExpertResult{}, core.E("rocm.hip.MoELazyLaunch", "copy resident expert output", err)
	}
	resident = resident[:buffers.TotalExperts]
	for index, value := range payload {
		if value != 0 && value != 1 {
			return hipMoELazyExpertResult{}, core.E("rocm.hip.MoELazyLaunch", "resident expert output must contain binary flags", nil)
		}
		resident[index] = value != 0
	}
	return hipMoELazyExpertResult{Resident: resident}, nil
}

func hipRunMoELazyExpertKernel(ctx context.Context, driver nativeHIPDriver, req hipMoELazyExpertRequest) (hipMoELazyExpertResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipMoELazyExpertResult{}, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return hipMoELazyExpertResult{}, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return hipMoELazyExpertResult{}, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return hipMoELazyExpertResult{}, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameMoELazy, launchBytes, req.TotalExperts)
	if err != nil {
		return hipMoELazyExpertResult{}, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return hipMoELazyExpertResult{}, err
	}
	return buffers.ReadOutput()
}
