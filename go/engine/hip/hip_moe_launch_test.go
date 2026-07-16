// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestHIPMoERouterLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipMoERouterRequest{Logits: []float32{0.1, 2, 1, -1}, TopK: 2, Layer: 7}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.RequireNoError(t, err)
	payload, err := launch.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipMoERouterLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipMoERouterLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipMoERouterLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(buffers.Logits.Pointer()), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(buffers.IDs.Pointer()), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint64(buffers.Probs.Pointer()), binary.LittleEndian.Uint64(payload[24:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(payload[32:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[36:]))
	core.AssertEqual(t, uint32(16), binary.LittleEndian.Uint32(payload[40:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[44:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[48:]))
	core.AssertEqual(t, uint32(7), binary.LittleEndian.Uint32(payload[52:]))
	core.AssertEqual(t, uint64(buffers.Status.Pointer()), binary.LittleEndian.Uint64(payload[56:]))
}

func TestHIPMoEBatchRouteRows_Good_GathersScattersAndReduces(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	inputValues := []float32{1, 2, 3, 4, 5, 6}
	inputPayload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, "test", "MoE batch route input", inputPayload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	metadataPayload := make([]byte, 32)
	binary.LittleEndian.PutUint32(metadataPayload[0:], 2)
	binary.LittleEndian.PutUint32(metadataPayload[4:], 4)
	binary.LittleEndian.PutUint32(metadataPayload[8:], math.Float32bits(0.5))
	binary.LittleEndian.PutUint32(metadataPayload[16:], 0)
	binary.LittleEndian.PutUint32(metadataPayload[20:], 1)
	binary.LittleEndian.PutUint32(metadataPayload[24:], math.Float32bits(2))
	metadata, err := hipUploadByteBuffer(driver, "test", "MoE batch route metadata", metadataPayload, 2)
	core.RequireNoError(t, err)
	defer metadata.Close()

	gathered, err := hipAllocateByteBuffer(driver, "test", "MoE gathered rows", 4*4, 4)
	core.RequireNoError(t, err)
	defer gathered.Close()
	core.RequireNoError(t, hipRunMoEBatchGatherRowsKernel(context.Background(), driver, input, metadata, 2, 2, 3, gathered))
	gatheredValues, err := hipReadFloat32DeviceOutput(gathered, "test", "MoE gathered rows", 4)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{5, 6, 1, 2}, gatheredValues, 0)

	routeOutput, err := hipAllocateByteBuffer(driver, "test", "MoE route output", 12*4, 12)
	core.RequireNoError(t, err)
	defer routeOutput.Close()
	core.RequireNoError(t, hipMemsetDevice(driver, routeOutput.Pointer(), 0, routeOutput.SizeBytes()))
	core.RequireNoError(t, hipRunMoEBatchScatterRoutesKernel(context.Background(), driver, gathered, metadata, 2, 2, 6, routeOutput))
	routeValues, err := hipReadFloat32DeviceOutput(routeOutput, "test", "MoE route output", 12)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0, 0, 2, 4, 0, 0, 0, 0, 2.5, 3, 0, 0}, routeValues, 0)

	destination, err := hipAllocateByteBuffer(driver, "test", "MoE reduced rows", 6*4, 6)
	core.RequireNoError(t, err)
	defer destination.Close()
	core.RequireNoError(t, hipRunMoEBatchReduceRoutesKernel(context.Background(), driver, routeOutput, 3, 2, 2, destination))
	got, err := hipReadFloat32DeviceOutput(destination, "test", "MoE reduced rows", 6)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{2, 4, 0, 0, 2.5, 3}, got, 0)
}

func TestHIPMoEMLXAffineRouteChunk_Good(t *testing.T) {
	chunk := hipMoEMLXAffineRouteChunk{
		GateUpWeightPointer: 11,
		GateUpScalePointer:  22,
		GateUpBiasPointer:   33,
		DownWeightPointer:   44,
		DownScalePointer:    55,
		DownBiasPointer:     66,
		RouteCount:          2,
	}
	chunk.TokenRows[0], chunk.TokenRows[1] = 7, 3
	chunk.PairIndices[0], chunk.PairIndices[1] = 9, 4
	chunk.RouteWeights[0], chunk.RouteWeights[1] = 0.75, 0.25
	payload, err := chunk.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipMoEMLXAffineRouteChunkBytes, len(payload))
	core.AssertEqual(t, uint64(11), binary.LittleEndian.Uint64(payload[0:]))
	core.AssertEqual(t, uint64(22), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(33), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint64(44), binary.LittleEndian.Uint64(payload[24:]))
	core.AssertEqual(t, uint64(55), binary.LittleEndian.Uint64(payload[32:]))
	core.AssertEqual(t, uint64(66), binary.LittleEndian.Uint64(payload[40:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[48:]))
	core.AssertEqual(t, uint32(7), binary.LittleEndian.Uint32(payload[56:]))
	core.AssertEqual(t, uint32(3), binary.LittleEndian.Uint32(payload[60:]))
	core.AssertEqual(t, uint32(9), binary.LittleEndian.Uint32(payload[88:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(payload[92:]))
	assertFloat32Near(t, 0.75, math.Float32frombits(binary.LittleEndian.Uint32(payload[120:])))
	assertFloat32Near(t, 0.25, math.Float32frombits(binary.LittleEndian.Uint32(payload[124:])))
}

func TestHIPMoEMLXAffineRoutesLaunchArgs_Good(t *testing.T) {
	args := hipMoEMLXAffineRoutesLaunchArgs{
		InputPointer: 11, ChunkPointer: 22, OutputPointer: 33,
		Rows: 704, Cols: 2816, InputRows: 64, PairCount: 512, ChunkCount: 73,
		GroupSize: 64, Bits: 4, InputBytes: 64 * 2816 * 4,
		ChunkBytes:  73 * hipMoEMLXAffineRouteChunkBytes,
		OutputBytes: 512 * 704 * 4, GateUp: true,
	}
	payload, err := args.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipMoEMLXAffineRoutesLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipMoEMLXAffineRoutesLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipMoEMLXAffineRoutesLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(11), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(22), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint64(33), binary.LittleEndian.Uint64(payload[24:]))
	core.AssertEqual(t, uint32(704), binary.LittleEndian.Uint32(payload[32:]))
	core.AssertEqual(t, uint32(2816), binary.LittleEndian.Uint32(payload[36:]))
	core.AssertEqual(t, uint32(64), binary.LittleEndian.Uint32(payload[40:]))
	core.AssertEqual(t, uint32(512), binary.LittleEndian.Uint32(payload[44:]))
	core.AssertEqual(t, uint32(73), binary.LittleEndian.Uint32(payload[48:]))
	core.AssertEqual(t, uint32(64), binary.LittleEndian.Uint32(payload[52:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(payload[56:]))
	core.AssertEqual(t, uint32(hipMoEMLXAffineRoutesFlagGateUp), binary.LittleEndian.Uint32(payload[72:]))
}

func TestHIPMoEMLXAffineRoutesLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	input := hipBorrowDeviceByteBufferValue(driver, "route input", 11, 2*32*4, 2*32)
	chunks := hipBorrowDeviceByteBufferValue(driver, "route chunks", 22, 2*hipMoEMLXAffineRouteChunkBytes, 2*(hipMoEMLXAffineRouteChunkBytes/4))
	output := hipBorrowDeviceByteBufferValue(driver, "route output", 33, 4*16*4, 4*16)

	core.RequireNoError(t, hipRunMoEMLXAffineRoutesKernel(
		context.Background(), driver, &input, &chunks, 16, 32, 2, 4, 2, 32, 4, true, &output,
	))
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameMoEMLXAffineRoutes, driver.launches[0].Name)
	core.AssertEqual(t, uint32(2), driver.launches[0].GridX)
	core.AssertEqual(t, uint32(2), driver.launches[0].GridY)
	core.AssertEqual(t, uint32(256), driver.launches[0].BlockX)
}

func TestHIPMoEMLXAffineRoutesLaunch_Bad_RejectsChunkShape(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	input := hipBorrowDeviceByteBufferValue(driver, "route input", 11, 2*32*4, 2*32)
	chunks := hipBorrowDeviceByteBufferValue(driver, "short route chunks", 22, hipMoEMLXAffineRouteChunkBytes, hipMoEMLXAffineRouteChunkBytes/4)
	output := hipBorrowDeviceByteBufferValue(driver, "route output", 33, 4*16*4, 4*16)

	err := hipRunMoEMLXAffineRoutesKernel(
		context.Background(), driver, &input, &chunks, 16, 32, 2, 4, 2, 32, 4, true, &output,
	)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "buffer shape")
}

func TestHIPMoEBatchRouteRowsHardware_Good_GathersScattersAndReduces(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MOE_LANE_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MOE_LANE_TESTS=1 to run the HIP MoE route-kernel receipt")
	}
	if strings.TrimSpace(os.Getenv("GO_ROCM_KERNEL_HSACO")) == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to the linked ROCm kernels HSACO")
	}
	driver := newSystemHIPDriver()
	if driver == nil || !driver.Available() {
		t.Skip("ROCm runtime is not available on this host")
	}

	inputPayload, err := hipFloat32Payload([]float32{1, 2, 3, 4, 5, 6})
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, "rocm.hip.MoEBatchRouteRowsHardware", "input rows", inputPayload, 6)
	core.RequireNoError(t, err)
	defer input.Close()
	metadataPayload := make([]byte, 2*hipMoEBatchRouteMetadataBytes)
	binary.LittleEndian.PutUint32(metadataPayload[0:], 2)
	binary.LittleEndian.PutUint32(metadataPayload[4:], 4)
	binary.LittleEndian.PutUint32(metadataPayload[8:], math.Float32bits(0.5))
	binary.LittleEndian.PutUint32(metadataPayload[16:], 0)
	binary.LittleEndian.PutUint32(metadataPayload[20:], 1)
	binary.LittleEndian.PutUint32(metadataPayload[24:], math.Float32bits(2))
	metadata, err := hipUploadByteBuffer(driver, "rocm.hip.MoEBatchRouteRowsHardware", "route metadata", metadataPayload, 2)
	core.RequireNoError(t, err)
	defer metadata.Close()

	gathered, err := hipAllocateByteBuffer(driver, "rocm.hip.MoEBatchRouteRowsHardware", "gathered rows", 4*4, 4)
	core.RequireNoError(t, err)
	defer gathered.Close()
	core.RequireNoError(t, hipRunMoEBatchGatherRowsKernel(context.Background(), driver, input, metadata, 2, 2, 3, gathered))

	routes, err := hipAllocateByteBuffer(driver, "rocm.hip.MoEBatchRouteRowsHardware", "route pairs", 12*4, 12)
	core.RequireNoError(t, err)
	defer routes.Close()
	core.RequireNoError(t, hipMemsetDevice(driver, routes.Pointer(), 0, routes.SizeBytes()))
	core.RequireNoError(t, hipRunMoEBatchScatterRoutesKernel(context.Background(), driver, gathered, metadata, 2, 2, 6, routes))

	reduced, err := hipAllocateByteBuffer(driver, "rocm.hip.MoEBatchRouteRowsHardware", "reduced rows", 6*4, 6)
	core.RequireNoError(t, err)
	defer reduced.Close()
	core.RequireNoError(t, hipRunMoEBatchReduceRoutesKernel(context.Background(), driver, routes, 3, 2, 2, reduced))
	got, err := hipReadFloat32DeviceOutput(reduced, "rocm.hip.MoEBatchRouteRowsHardware", "reduced rows", 6)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{2, 4, 0, 0, 2.5, 3}, got, 0)
}

func TestHIPMoEBatchRouteRows_Bad_RejectsOutputShape(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	input, err := hipAllocateByteBuffer(driver, "test", "MoE batch route input", 6*4, 6)
	core.RequireNoError(t, err)
	defer input.Close()
	metadata, err := hipAllocateByteBuffer(driver, "test", "MoE batch route metadata", 2*16, 2)
	core.RequireNoError(t, err)
	defer metadata.Close()
	output, err := hipAllocateByteBuffer(driver, "test", "MoE gathered rows", 3*4, 3)
	core.RequireNoError(t, err)
	defer output.Close()
	if err := hipRunMoEBatchGatherRowsKernel(context.Background(), driver, input, metadata, 2, 2, 3, output); err == nil {
		t.Fatal("hipRunMoEBatchGatherRowsKernel accepted a truncated output")
	}
}

func TestHIPGGUFQ4_0ProjectionLaunchArgs_Good(t *testing.T) {
	args := hipGGUFQ4_0ProjectionLaunchArgs{
		InputPointer:  11,
		WeightPointer: 22,
		OutputPointer: 33,
		Rows:          704,
		Cols:          2816,
		RowOffset:     704,
		WeightRows:    1408,
		InputBytes:    2816 * 4,
		WeightBytes:   1408 * (2816 / 32) * 18,
		OutputBytes:   704 * 4,
	}
	payload, err := args.Binary()
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipGGUFQ4_0ProjectionLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipGGUFQ4_0ProjectionLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipGGUFQ4_0ProjectionLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(11), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(22), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint64(33), binary.LittleEndian.Uint64(payload[24:]))
	core.AssertEqual(t, uint32(704), binary.LittleEndian.Uint32(payload[32:]))
	core.AssertEqual(t, uint32(2816), binary.LittleEndian.Uint32(payload[36:]))
	core.AssertEqual(t, uint32(704), binary.LittleEndian.Uint32(payload[40:]))
	core.AssertEqual(t, uint32(1408), binary.LittleEndian.Uint32(payload[44:]))
	core.AssertEqual(t, uint32(2816*4), binary.LittleEndian.Uint32(payload[48:]))
	core.AssertEqual(t, uint32(1408*(2816/32)*18), binary.LittleEndian.Uint32(payload[52:]))
	core.AssertEqual(t, uint32(704*4), binary.LittleEndian.Uint32(payload[56:]))
}

func TestHIPQ8_1QuantizeLaunchArgs_Good(t *testing.T) {
	args := hipQ8_1QuantizeLaunchArgs{
		InputPointer:  11,
		OutputPointer: 22,
		Rows:          1,
		Cols:          3840,
		InputBytes:    3840 * 4,
		OutputBytes:   (3840 / hipQ8_1BlockSize) * hipQ8_1BlockBytes,
	}
	payload, err := args.Binary()
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipQ8_1QuantizeLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipQ8_1QuantizeLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipQ8_1QuantizeLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(11), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(22), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint32(1), binary.LittleEndian.Uint32(payload[24:]))
	core.AssertEqual(t, uint32(3840), binary.LittleEndian.Uint32(payload[28:]))
	core.AssertEqual(t, uint64(3840*4), binary.LittleEndian.Uint64(payload[32:]))
	core.AssertEqual(t, uint64((3840/hipQ8_1BlockSize)*hipQ8_1BlockBytes), binary.LittleEndian.Uint64(payload[40:]))
}

func TestHIPGGUFQ4KQ8_1GateUpLaunchArgs_Good(t *testing.T) {
	args := hipGGUFQ4KQ8_1GateUpLaunchArgs{
		InputPointer:  11,
		GatePointer:   22,
		UpPointer:     33,
		OutputPointer: 44,
		Rows:          15360,
		Cols:          3840,
		Batch:         1,
		InputBytes:    (3840 / hipQ8_1BlockSize) * hipQ8_1BlockBytes,
		GateBytes:     15360 * (3840 / hipGGUFQ4KBlockSize) * hipGGUFQ4KBlockBytes,
		UpBytes:       15360 * (3840 / hipGGUFQ4KBlockSize) * hipGGUFQ4KBlockBytes,
		OutputBytes:   15360 * 4,
	}
	payload, err := args.Binary()
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipGGUFQ4KQ8_1GateUpLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipGGUFQ4KQ8_1GateUpLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipGGUFQ4KQ8_1GateUpLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(11), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(22), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint64(33), binary.LittleEndian.Uint64(payload[24:]))
	core.AssertEqual(t, uint64(44), binary.LittleEndian.Uint64(payload[32:]))
	core.AssertEqual(t, uint32(15360), binary.LittleEndian.Uint32(payload[40:]))
	core.AssertEqual(t, uint32(3840), binary.LittleEndian.Uint32(payload[44:]))
	core.AssertEqual(t, uint32(1), binary.LittleEndian.Uint32(payload[48:]))
	core.AssertEqual(t, uint32((3840/hipQ8_1BlockSize)*hipQ8_1BlockBytes), binary.LittleEndian.Uint32(payload[56:]))
	core.AssertEqual(t, uint32(15360*(3840/hipGGUFQ4KBlockSize)*hipGGUFQ4KBlockBytes), binary.LittleEndian.Uint32(payload[60:]))
	core.AssertEqual(t, uint32(15360*(3840/hipGGUFQ4KBlockSize)*hipGGUFQ4KBlockBytes), binary.LittleEndian.Uint32(payload[64:]))
	core.AssertEqual(t, uint32(15360*4), binary.LittleEndian.Uint32(payload[68:]))
}

func TestHIPGGUFQ4KQ8_1GateUpLaunchArgs_Good_Expanded(t *testing.T) {
	args := hipGGUFQ4KQ8_1GateUpLaunchArgs{
		InputPointer:  11,
		GatePointer:   22,
		UpPointer:     33,
		OutputPointer: 44,
		Rows:          8,
		Cols:          256,
		Batch:         1,
		InputBytes:    8 * hipQ8_1BlockBytes,
		GateBytes:     8 * hipGGUFQ4KExpandedBlockBytes,
		UpBytes:       8 * hipGGUFQ4KExpandedBlockBytes,
		OutputBytes:   8 * 4,
	}
	payload, err := args.Binary()
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint32(8*hipGGUFQ4KExpandedBlockBytes), binary.LittleEndian.Uint32(payload[60:]))
	core.AssertEqual(t, uint32(8*hipGGUFQ4KExpandedBlockBytes), binary.LittleEndian.Uint32(payload[64:]))
}

func TestHIPGGUFQ4KExpandLaunchArgs_Good(t *testing.T) {
	args := hipGGUFQ4KExpandLaunchArgs{
		RawPointer:      11,
		ExpandedPointer: 22,
		BlockCount:      17,
		RawBytes:        17 * hipGGUFQ4KBlockBytes,
		ExpandedBytes:   17 * hipGGUFQ4KExpandedBlockBytes,
	}
	payload, err := args.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipGGUFQ4KExpandLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipGGUFQ4KExpandLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipGGUFQ4KExpandLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(11), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(22), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint32(17), binary.LittleEndian.Uint32(payload[24:]))
	core.AssertEqual(t, uint64(17*hipGGUFQ4KBlockBytes), binary.LittleEndian.Uint64(payload[32:]))
	core.AssertEqual(t, uint64(17*hipGGUFQ4KExpandedBlockBytes), binary.LittleEndian.Uint64(payload[40:]))
}

func TestHIPGGUFQ4_0SelectedExpertsLaunchArgs_Good(t *testing.T) {
	args := hipGGUFQ4_0SelectedExpertsLaunchArgs{
		InputPointer:      11,
		ActivationPointer: 22,
		OutputPointer:     33,
		TopK:              2,
		HiddenSize:        32,
		ExpertFF:          32,
		GateUpRows:        64,
		DownRows:          32,
		InputBytes:        128,
		ActivationBytes:   256,
		OutputBytes:       128,
		GateUpBytes:       64 * 18,
		DownBytes:         32 * 18,
		GateUpFormat:      hipGGUFExpertFormatQ4_0,
		DownFormat:        hipGGUFExpertFormatQ4_0,
	}
	args.GateUpPointers[0], args.GateUpPointers[1] = 101, 102
	args.DownPointers[0], args.DownPointers[1] = 201, 202
	args.RouteWeights[0], args.RouteWeights[1] = 0.75, 0.25
	payload, err := args.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipGGUFQ4_0SelectedExpertsLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipGGUFQ4_0SelectedExpertsLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint64(101), binary.LittleEndian.Uint64(payload[32:]))
	core.AssertEqual(t, uint64(102), binary.LittleEndian.Uint64(payload[40:]))
	core.AssertEqual(t, uint64(201), binary.LittleEndian.Uint64(payload[96:]))
	core.AssertEqual(t, uint64(202), binary.LittleEndian.Uint64(payload[104:]))
	assertFloat32Near(t, 0.75, math.Float32frombits(binary.LittleEndian.Uint32(payload[160:])))
	assertFloat32Near(t, 0.25, math.Float32frombits(binary.LittleEndian.Uint32(payload[164:])))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[192:]))
	core.AssertEqual(t, uint32(32), binary.LittleEndian.Uint32(payload[196:]))
	core.AssertEqual(t, hipGGUFExpertFormatQ4_0, binary.LittleEndian.Uint32(payload[232:]))
	core.AssertEqual(t, hipGGUFExpertFormatQ4_0, binary.LittleEndian.Uint32(payload[236:]))
}

func TestHIPGGUFQ4_0SelectedExpertsLaunch_Good(t *testing.T) {
	t.Setenv(hipGemma4SelectedExpertPair16Env, "0")
	driver := &fakeHIPDriver{available: true}
	input := hipBorrowDeviceByteBufferValue(driver, "input", 11, 32*4, 32)
	activation := hipBorrowDeviceByteBufferValue(driver, "activation", 22, 2*32*4, 2*32)
	output := hipBorrowDeviceByteBufferValue(driver, "output", 33, 32*4, 32)
	borrow := func(label string, pointer nativeDevicePointer, bytes uint64) *hipDeviceByteBuffer {
		buffer := hipBorrowDeviceByteBufferValue(driver, label, pointer, bytes, int(bytes))
		return &buffer
	}
	entries := []*hipGemma4ExpertCacheEntry{
		{GateUp: borrow("gate up 0", 101, 64*18), Down: borrow("down 0", 201, 32*18), GateUpRows: 64, GateUpCols: 32, DownRows: 32, DownCols: 32},
		{GateUp: borrow("gate up 1", 102, 64*18), Down: borrow("down 1", 202, 32*18), GateUpRows: 64, GateUpCols: 32, DownRows: 32, DownCols: 32},
	}

	core.RequireNoError(t, hipRunGGUFQ4_0SelectedExpertsKernelWithDeviceInputOutput(context.Background(), driver, &input, entries, []float32{0.75, 0.25}, 32, 32, &activation, &output))
	core.AssertEqual(t, 2, len(driver.launches))
	core.AssertEqual(t, hipKernelNameGGUFQ4_0SelectedExpertGateUp, driver.launches[0].Name)
	core.AssertEqual(t, hipKernelNameGGUFQ4_0SelectedExpertDown, driver.launches[1].Name)
	core.AssertEqual(t, uint32(8), driver.launches[0].GridX)
	core.AssertEqual(t, uint32(4), driver.launches[1].GridX)
}

func TestHIPGGUFQ4_0SelectedExpertsLaunch_Good_Pair16ProductionShape(t *testing.T) {
	t.Setenv(hipGemma4SelectedExpertPair16Env, "")
	const (
		topK     = 8
		hidden   = 2816
		expertFF = 704
	)
	driver := &fakeHIPDriver{available: true}
	input := hipBorrowDeviceByteBufferValue(driver, "input", 11, hidden*4, hidden)
	activation := hipBorrowDeviceByteBufferValue(driver, "activation", 22, topK*expertFF*4, topK*expertFF)
	output := hipBorrowDeviceByteBufferValue(driver, "output", 33, hidden*4, hidden)
	gateUpBytes := uint64(2 * expertFF * (hidden / hipGGUFQ4_0BlockSize) * hipGGUFQ4_0BlockBytes)
	downBytes := uint64(hidden * (expertFF / hipGGUFQ4_0BlockSize) * hipGGUFQ4_0BlockBytes)
	entries := make([]*hipGemma4ExpertCacheEntry, topK)
	routeWeights := make([]float32, topK)
	for index := range entries {
		gateUp := hipBorrowDeviceByteBufferValue(driver, "gate up", nativeDevicePointer(101+index), gateUpBytes, int(gateUpBytes))
		down := hipBorrowDeviceByteBufferValue(driver, "down", nativeDevicePointer(201+index), downBytes, int(downBytes))
		entries[index] = &hipGemma4ExpertCacheEntry{
			GateUp: &gateUp, Down: &down,
			GateUpRows: 2 * expertFF, GateUpCols: hidden,
			DownRows: hidden, DownCols: expertFF,
		}
		routeWeights[index] = 1 / float32(topK)
	}

	core.RequireNoError(t, hipRunGGUFQ4_0SelectedExpertsKernelWithDeviceInputOutput(context.Background(), driver, &input, entries, routeWeights, hidden, expertFF, &activation, &output))
	core.AssertEqual(t, 2, len(driver.launches))
	core.AssertEqual(t, hipKernelNameGGUFQ4_0SelectedExpertGateUpPair16, driver.launches[0].Name)
	core.AssertEqual(t, hipKernelNameGGUFQ4_0SelectedExpertDownPair16, driver.launches[1].Name)
	core.AssertEqual(t, uint32(352), driver.launches[0].GridX)
	core.AssertEqual(t, uint32(176), driver.launches[1].GridX)
	core.AssertEqual(t, uint32(256), driver.launches[0].BlockX)
	core.AssertEqual(t, uint32(256), driver.launches[1].BlockX)
}

func TestHIPGGUFMixedSelectedExpertsLaunch_Good(t *testing.T) {
	t.Setenv(hipGemma4SelectedExpertPair16Env, "0")
	driver := &fakeHIPDriver{available: true}
	input := hipBorrowDeviceByteBufferValue(driver, "input", 11, 256*4, 256)
	activation := hipBorrowDeviceByteBufferValue(driver, "activation", 22, 2*32*4, 2*32)
	output := hipBorrowDeviceByteBufferValue(driver, "output", 33, 256*4, 256)
	borrow := func(label string, pointer nativeDevicePointer, bytes uint64) *hipDeviceByteBuffer {
		buffer := hipBorrowDeviceByteBufferValue(driver, label, pointer, bytes, int(bytes))
		return &buffer
	}
	entries := []*hipGemma4ExpertCacheEntry{
		{GateUp: borrow("gate up 0", 101, 64*144), Down: borrow("down 0", 201, 256*24), GateUpRows: 64, GateUpCols: 256, DownRows: 256, DownCols: 32, GateUpFormat: hipGGUFExpertFormatQ4K, DownFormat: hipGGUFExpertFormatQ5_1},
		{GateUp: borrow("gate up 1", 102, 64*144), Down: borrow("down 1", 202, 256*24), GateUpRows: 64, GateUpCols: 256, DownRows: 256, DownCols: 32, GateUpFormat: hipGGUFExpertFormatQ4K, DownFormat: hipGGUFExpertFormatQ5_1},
	}

	core.RequireNoError(t, hipRunGGUFQ4_0SelectedExpertsKernelWithDeviceInputOutput(context.Background(), driver, &input, entries, []float32{0.75, 0.25}, 256, 32, &activation, &output))
	core.AssertEqual(t, 2, len(driver.launches))
	core.AssertEqual(t, hipKernelNameGGUFQ4KSelectedExpertGateUp, driver.launches[0].Name)
	core.AssertEqual(t, hipKernelNameGGUFQ5_1SelectedExpertDown, driver.launches[1].Name)
}

func TestHIPGGUFMixedSelectedExpertsQ8_0DownLaunch_Good(t *testing.T) {
	t.Setenv(hipGemma4SelectedExpertPair16Env, "0")
	driver := &fakeHIPDriver{available: true}
	input := hipBorrowDeviceByteBufferValue(driver, "input", 11, 256*4, 256)
	activation := hipBorrowDeviceByteBufferValue(driver, "activation", 22, 2*32*4, 2*32)
	output := hipBorrowDeviceByteBufferValue(driver, "output", 33, 256*4, 256)
	borrow := func(label string, pointer nativeDevicePointer, bytes uint64) *hipDeviceByteBuffer {
		buffer := hipBorrowDeviceByteBufferValue(driver, label, pointer, bytes, int(bytes))
		return &buffer
	}
	entries := []*hipGemma4ExpertCacheEntry{
		{GateUp: borrow("gate up 0", 101, 64*144), Down: borrow("down 0", 201, 256*34), GateUpRows: 64, GateUpCols: 256, DownRows: 256, DownCols: 32, GateUpFormat: hipGGUFExpertFormatQ4K, DownFormat: hipGGUFExpertFormatQ8_0},
		{GateUp: borrow("gate up 1", 102, 64*144), Down: borrow("down 1", 202, 256*34), GateUpRows: 64, GateUpCols: 256, DownRows: 256, DownCols: 32, GateUpFormat: hipGGUFExpertFormatQ4K, DownFormat: hipGGUFExpertFormatQ8_0},
	}

	core.RequireNoError(t, hipRunGGUFQ4_0SelectedExpertsKernelWithDeviceInputOutput(context.Background(), driver, &input, entries, []float32{0.75, 0.25}, 256, 32, &activation, &output))
	core.AssertEqual(t, 2, len(driver.launches))
	core.AssertEqual(t, hipKernelNameGGUFQ4KSelectedExpertGateUp, driver.launches[0].Name)
	core.AssertEqual(t, hipKernelNameGGUFQ8_0SelectedExpertDown, driver.launches[1].Name)
}

func TestHIPGGUFMixedSelectedExpertsLaunch_Good_Pair16ProductionShape(t *testing.T) {
	t.Setenv(hipGemma4SelectedExpertPair16Env, "")
	const (
		topK     = 8
		hidden   = 2816
		expertFF = 704
	)
	for _, test := range []struct {
		name       string
		gateFormat uint32
		gateBlock  int
		gateKernel string
		gateGrid   uint32
		downFormat uint32
		downBlock  int
		downKernel string
		downGrid   uint32
	}{
		{name: "Q4_K_Q5_1", gateFormat: hipGGUFExpertFormatQ4K, gateBlock: hipGGUFQ4KBlockBytes, gateKernel: "rocm_gguf_q4_k_selected_expert_gate_up_split_pair16", gateGrid: 704, downFormat: hipGGUFExpertFormatQ5_1, downBlock: hipGGUFQ5_1BlockBytes, downKernel: "rocm_gguf_q5_1_selected_expert_down_expert8_pair16", downGrid: 1408},
		{name: "Q4_K_Q8_0", gateFormat: hipGGUFExpertFormatQ4K, gateBlock: hipGGUFQ4KBlockBytes, gateKernel: "rocm_gguf_q4_k_selected_expert_gate_up_split_pair16", gateGrid: 704, downFormat: hipGGUFExpertFormatQ8_0, downBlock: hipGGUFQ8_0BlockBytes, downKernel: "rocm_gguf_q8_0_selected_expert_down_pair16", downGrid: 176},
		{name: "Q4_K_expanded_Q5_1", gateFormat: hipGGUFExpertFormatQ4KExpanded, gateBlock: hipGGUFQ4KExpandedBlockBytes, gateKernel: "rocm_gguf_q4_k_expanded_selected_expert_gate_up_split_pair16", gateGrid: 704, downFormat: hipGGUFExpertFormatQ5_1, downBlock: hipGGUFQ5_1BlockBytes, downKernel: "rocm_gguf_q5_1_selected_expert_down_expert8_pair16", downGrid: 1408},
		{name: "Q4_K_expanded_Q8_0", gateFormat: hipGGUFExpertFormatQ4KExpanded, gateBlock: hipGGUFQ4KExpandedBlockBytes, gateKernel: "rocm_gguf_q4_k_expanded_selected_expert_gate_up_split_pair16", gateGrid: 704, downFormat: hipGGUFExpertFormatQ8_0, downBlock: hipGGUFQ8_0BlockBytes, downKernel: "rocm_gguf_q8_0_selected_expert_down_pair16", downGrid: 176},
	} {
		t.Run(test.name, func(t *testing.T) {
			driver := &fakeHIPDriver{available: true}
			input := hipBorrowDeviceByteBufferValue(driver, "input", 11, hidden*4, hidden)
			activation := hipBorrowDeviceByteBufferValue(driver, "activation", 22, topK*expertFF*4, topK*expertFF)
			output := hipBorrowDeviceByteBufferValue(driver, "output", 33, hidden*4, hidden)
			gateUpBytes := uint64(2 * expertFF * (hidden / hipGGUFQ4KBlockSize) * test.gateBlock)
			downBytes := uint64(hidden * (expertFF / hipGGUFQ4_0BlockSize) * test.downBlock)
			entries := make([]*hipGemma4ExpertCacheEntry, topK)
			routeWeights := make([]float32, topK)
			for index := range entries {
				gateUp := hipBorrowDeviceByteBufferValue(driver, "gate up", nativeDevicePointer(101+index), gateUpBytes, int(gateUpBytes))
				down := hipBorrowDeviceByteBufferValue(driver, "down", nativeDevicePointer(201+index), downBytes, int(downBytes))
				entries[index] = &hipGemma4ExpertCacheEntry{
					GateUp: &gateUp, Down: &down,
					GateUpRows: 2 * expertFF, GateUpCols: hidden,
					DownRows: hidden, DownCols: expertFF,
					GateUpFormat: test.gateFormat, DownFormat: test.downFormat,
				}
				routeWeights[index] = 1 / float32(topK)
			}

			core.RequireNoError(t, hipRunGGUFQ4_0SelectedExpertsKernelWithDeviceInputOutput(context.Background(), driver, &input, entries, routeWeights, hidden, expertFF, &activation, &output))
			core.AssertEqual(t, 2, len(driver.launches))
			core.AssertEqual(t, test.gateKernel, driver.launches[0].Name)
			core.AssertEqual(t, test.downKernel, driver.launches[1].Name)
			core.AssertEqual(t, test.gateGrid, driver.launches[0].GridX)
			core.AssertEqual(t, test.downGrid, driver.launches[1].GridX)
			core.AssertEqual(t, uint32(256), driver.launches[0].BlockX)
			core.AssertEqual(t, uint32(256), driver.launches[1].BlockX)
		})
	}
}

func TestHIPMLXAffineSelectedExpertsLaunch_Good(t *testing.T) {
	const (
		hidden    = 32
		expertFF  = 32
		groupSize = 32
		bits      = 4
	)
	driver := &fakeHIPDriver{available: true}
	inputValues := make([]float32, hidden)
	for index := range inputValues {
		inputValues[index] = 1
	}
	inputPayload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, "test", "MLX affine selected expert input", inputPayload, hidden)
	core.RequireNoError(t, err)
	defer input.Close()

	var owned []*hipDeviceByteBuffer
	projection := func(rows, cols int, biasValue float32) hipMLXQ4DeviceWeightConfig {
		t.Helper()
		packedCols := cols * bits / 32
		groups := cols / groupSize
		weight, allocErr := hipUploadByteBuffer(driver, "test", "MLX affine selected expert weight", make([]byte, rows*packedCols*4), rows*packedCols)
		core.RequireNoError(t, allocErr)
		scales, allocErr := hipUploadByteBuffer(driver, "test", "MLX affine selected expert scales", make([]byte, rows*groups*2), rows*groups)
		core.RequireNoError(t, allocErr)
		biasPayload := make([]byte, rows*groups*2)
		for index := 0; index < rows*groups; index++ {
			binary.LittleEndian.PutUint16(biasPayload[index*2:], hipFloat32ToBFloat16(biasValue))
		}
		biases, allocErr := hipUploadByteBuffer(driver, "test", "MLX affine selected expert biases", biasPayload, rows*groups)
		core.RequireNoError(t, allocErr)
		owned = append(owned, weight, scales, biases)
		return hipMLXQ4DeviceWeightConfig{
			WeightPointer: weight.Pointer(), ScalePointer: scales.Pointer(), BiasPointer: biases.Pointer(),
			WeightBytes: weight.SizeBytes(), ScaleBytes: scales.SizeBytes(), BiasBytes: biases.SizeBytes(),
			Rows: rows, Cols: cols, GroupSize: groupSize, Bits: bits,
		}
	}
	entries := []*hipGemma4ExpertCacheEntry{
		{Storage: hipGemma4MoEExpertStorageMLXAffine, MLXGate: projection(expertFF, hidden, 1.0/hidden), MLXUp: projection(expertFF, hidden, 2.0/hidden), MLXDown: projection(hidden, expertFF, 1.0/expertFF)},
		{Storage: hipGemma4MoEExpertStorageMLXAffine, MLXGate: projection(expertFF, hidden, 0.5/hidden), MLXUp: projection(expertFF, hidden, 1.0/hidden), MLXDown: projection(hidden, expertFF, 0.5/expertFF)},
	}
	defer func() {
		for _, buffer := range owned {
			core.AssertNoError(t, buffer.Close())
		}
	}()
	activation, err := hipAllocateByteBuffer(driver, "test", "MLX affine selected expert activation", expertFF*4, expertFF)
	core.RequireNoError(t, err)
	defer activation.Close()
	downOutput, err := hipAllocateByteBuffer(driver, "test", "MLX affine selected expert down output", hidden*4, hidden)
	core.RequireNoError(t, err)
	defer downOutput.Close()
	expertOutput, err := hipAllocateByteBuffer(driver, "test", "MLX affine selected expert output", hidden*4, hidden)
	core.RequireNoError(t, err)
	defer expertOutput.Close()
	workspace := hipNewAttentionHeadsChunkedWorkspace()
	defer workspace.Close()

	err = hipRunGemma4MLXAffineSelectedExpertsWithWorkspace(
		context.Background(), driver, input, entries, []float32{0.75, 0.25}, hidden, expertFF,
		activation, downOutput, expertOutput, workspace,
	)
	core.RequireNoError(t, err)
	got, err := hipReadFloat32DeviceOutput(expertOutput, "test", "MLX affine selected expert output", hidden)
	core.RequireNoError(t, err)
	gelu := func(value float64) float64 {
		return 0.5 * value * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(value+0.044715*value*value*value)))
	}
	want := float32(0.75*(gelu(1)*2) + 0.25*(gelu(0.5)*0.5))
	for _, value := range got {
		if math.Abs(float64(value-want)) > 1e-5 {
			t.Fatalf("MLX affine selected expert output=%g want=%g", value, want)
		}
	}
	var geluLaunches, projectionLaunches, scaleLaunches, addLaunches int
	for _, launch := range driver.launches {
		switch launch.Name {
		case hipKernelNameMLXQ4GELUTanhMul:
			geluLaunches++
		case hipKernelNameMLXQ4Proj:
			projectionLaunches++
		case hipKernelNameVectorScale:
			scaleLaunches++
		case hipKernelNameVectorAddScaled:
			addLaunches++
		case hipKernelNameGGUFQ4_0SelectedExpertGateUp, hipKernelNameGGUFQ4_0SelectedExpertDown:
			t.Fatalf("MLX affine path launched GGUF selected-expert kernel %q", launch.Name)
		}
	}
	core.AssertEqual(t, 2, geluLaunches)
	core.AssertEqual(t, 2, projectionLaunches)
	core.AssertEqual(t, 2, scaleLaunches)
	core.AssertEqual(t, 2, addLaunches)
}

func TestHIPGGUFQ4_0ProjectionLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	input := hipBorrowDeviceByteBufferValue(driver, "input", 11, 32*4, 32)
	weight := hipBorrowDeviceByteBufferValue(driver, "weight", 22, 2*18, 2*18)
	output := hipBorrowDeviceByteBufferValue(driver, "output", 33, 4, 1)
	core.RequireNoError(t, hipRunGGUFQ4_0ProjectionKernelWithDeviceInputOutput(context.Background(), driver, &input, &weight, 1, 32, 1, 2, &output))
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameGGUFQ4_0Projection, driver.launches[0].Name)
	core.AssertEqual(t, uint32(256), driver.launches[0].BlockX)
	core.AssertEqual(t, uint32(1), driver.launches[0].GridX)
}

func TestHIPGGUFQ4KQ8_1GateUpLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	input := hipBorrowDeviceByteBufferValue(driver, "input", 11, 256*4, 256)
	quantized := hipBorrowDeviceByteBufferValue(driver, "quantized", 22, 8*hipQ8_1BlockBytes, 8)
	gate := hipBorrowDeviceByteBufferValue(driver, "expanded gate", 33, 4*hipGGUFQ4KExpandedBlockBytes, 4)
	up := hipBorrowDeviceByteBufferValue(driver, "expanded up", 44, 4*hipGGUFQ4KExpandedBlockBytes, 4)
	output := hipBorrowDeviceByteBufferValue(driver, "output", 55, 4*4, 4)

	core.RequireNoError(t, hipRunQ8_1QuantizeKernel(context.Background(), driver, &input, 1, 256, &quantized))
	core.RequireNoError(t, hipRunGGUFQ4KQ8_1GELUTanhGateUpKernelGeometry(context.Background(), driver, hipKernelNameGGUFQ4KExpandedQ8_1GELUTanhGateUpPairRow8, hipGGUFQ4KQ8_1GateUpRow8RowsPerBlock, 1, hipGGUFQ4KExpandedBlockBytes, &quantized, &gate, &up, 4, 256, 1, &output))
	core.AssertEqual(t, 2, len(driver.launches))
	core.AssertEqual(t, hipKernelNameQ8_1QuantizeF32, driver.launches[0].Name)
	core.AssertEqual(t, uint32(hipQ8_1QuantizeBlockSize), driver.launches[0].BlockX)
	core.AssertEqual(t, uint32(1), driver.launches[0].GridX)
	core.AssertEqual(t, hipKernelNameGGUFQ4KExpandedQ8_1GELUTanhGateUpPairRow8, driver.launches[1].Name)
	core.AssertEqual(t, uint32(hipGGUFQ4_0ProjectionBlockSize), driver.launches[1].BlockX)
	core.AssertEqual(t, uint32(1), driver.launches[1].GridX)
}

func TestHIPGGUFQ4KQ8_1GateUpBatchLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	const batch = 17
	quantized := hipBorrowDeviceByteBufferValue(driver, "quantized batch", 22, batch*8*hipQ8_1BlockBytes, batch*8)
	gate := hipBorrowDeviceByteBufferValue(driver, "expanded gate", 33, 8*hipGGUFQ4KExpandedBlockBytes, 8)
	up := hipBorrowDeviceByteBufferValue(driver, "expanded up", 44, 8*hipGGUFQ4KExpandedBlockBytes, 8)
	output := hipBorrowDeviceByteBufferValue(driver, "batch output", 55, batch*8*4, batch*8)

	core.RequireNoError(t, hipRunGGUFQ4KExpandedQ8_1GELUTanhGateUpPairBatchKernel(context.Background(), driver, &quantized, &gate, &up, 8, 256, batch, &output))
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameGGUFQ4KExpandedQ8_1GELUTanhGateUpPairBatchRow8, driver.launches[0].Name)
	core.AssertEqual(t, uint32(hipGGUFQ4_0ProjectionBlockSize), driver.launches[0].BlockX)
	core.AssertEqual(t, uint32(1), driver.launches[0].GridX)
	core.AssertEqual(t, uint32(2), driver.launches[0].GridY)
}

func TestHIPGemma4Q4NativeQ4KGateUpBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	const (
		rows  = 8
		cols  = 256
		batch = 17
	)
	weightBytes := uint64(rows * (cols / hipGGUFQ4KBlockSize) * hipGGUFQ4KExpandedBlockBytes)
	cfg := hipGemma4Q4NativeQ4KGateUpConfig{
		GatePointer: 33, UpPointer: 44, GateBytes: weightBytes, UpBytes: weightBytes, Rows: rows, Cols: cols,
	}
	input := hipBorrowDeviceByteBufferValue(driver, "native batch input", 11, batch*cols*4, batch*cols)

	activation, err := hipRunGemma4Q4NativeQ4KGateUpWithDeviceInput(context.Background(), driver, &input, cfg, batch, workspace)
	core.RequireNoError(t, err)
	core.AssertEqual(t, batch*rows, activation.Count())
	core.AssertEqual(t, 2, len(driver.launches))
	core.AssertEqual(t, hipKernelNameQ8_1QuantizeF32, driver.launches[0].Name)
	core.AssertEqual(t, hipKernelNameGGUFQ4KExpandedQ8_1GELUTanhGateUpPairBatchRow8, driver.launches[1].Name)
	core.AssertEqual(t, uint32(2), driver.launches[1].GridY)
}

func TestHIPGGUFQ4_0ProjectionLaunch_Bad(t *testing.T) {
	_, err := (hipGGUFQ4_0ProjectionLaunchArgs{
		InputPointer: 1, WeightPointer: 2, OutputPointer: 3,
		Rows: 2, Cols: 32, RowOffset: 1, WeightRows: 2,
		InputBytes: 128, WeightBytes: 36, OutputBytes: 8,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "row range")
}

func TestHIPGemma4ExpertCache_Good(t *testing.T) {
	const (
		experts  = 2
		hidden   = 32
		expertFF = 32
		prefix   = 32
	)
	gateUpSliceBytes := 2 * expertFF * (hidden / hipGGUFQ4_0BlockSize) * hipGGUFQ4_0BlockBytes
	downSliceBytes := hidden * (expertFF / hipGGUFQ4_0BlockSize) * hipGGUFQ4_0BlockBytes
	gateUpPayload := make([]byte, experts*gateUpSliceBytes)
	downPayload := make([]byte, experts*downSliceBytes)
	for index := range gateUpPayload {
		gateUpPayload[index] = byte(index/gateUpSliceBytes + 1)
	}
	for index := range downPayload {
		downPayload[index] = byte(index/downSliceBytes + 11)
	}
	filePayload := make([]byte, prefix+len(gateUpPayload)+len(downPayload))
	copy(filePayload[prefix:], gateUpPayload)
	copy(filePayload[prefix+len(gateUpPayload):], downPayload)
	path := core.PathJoin(t.TempDir(), "experts.gguf")
	core.RequireTrue(t, core.WriteFile(path, filePayload, 0o644).OK)
	driver := &fakeHIPDriver{available: true}
	model := &hipLoadedModel{
		driver: driver,
		gemma4TextConfig: nativeGemma4TextConfig{
			NumExperts: experts,
		},
		hostTensors: map[string]nativeTensorInfo{
			"blk.0.ffn_gate_up_exps.weight": {
				Name: "blk.0.ffn_gate_up_exps.weight", Type: hipGGUFQ4_0TensorType, TypeName: "Q4_0",
				Dimensions: []uint64{hidden, 2 * expertFF, experts}, SourcePath: path, DataOffset: prefix,
				ByteSize: uint64(len(gateUpPayload)),
			},
			"blk.0.ffn_down_exps.weight": {
				Name: "blk.0.ffn_down_exps.weight", Type: hipGGUFQ4_0TensorType, TypeName: "Q4_0",
				Dimensions: []uint64{expertFF, hidden, experts}, SourcePath: path, DataOffset: prefix,
				Offset: uint64(len(gateUpPayload)), ByteSize: uint64(len(downPayload)),
			},
		},
	}
	model.expertCache = newHIPGemma4ExpertCache(driver, uint64(gateUpSliceBytes+downSliceBytes))
	entry0, err := model.gemma4ExpertCacheEntry(0, 0)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, len(driver.copies))
	core.AssertEqual(t, uint64(1), model.expertCache.stats.Misses)
	core.AssertEqual(t, uint64(1), model.expertCache.stats.HostMappings)
	core.AssertEqual(t, uint64(len(filePayload)), model.expertCache.stats.HostMappedBytes)
	core.AssertEqual(t, uint64(gateUpSliceBytes+downSliceBytes), model.expertCache.stats.H2DBytes)
	core.AssertEqual(t, 1, len(model.expertCache.sources))
	entry0Again, err := model.gemma4ExpertCacheEntry(0, 0)
	core.RequireNoError(t, err)
	core.AssertEqual(t, entry0, entry0Again)
	core.AssertEqual(t, 2, len(driver.copies))
	core.AssertEqual(t, uint64(1), model.expertCache.stats.Hits)

	entry1, err := model.gemma4ExpertCacheEntry(0, 1)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 4, len(driver.copies))
	core.AssertNotEqual(t, entry0, entry1)
	core.AssertEqual(t, 1, len(model.expertCache.entries))
	core.AssertEqual(t, uint64(2), model.expertCache.stats.Misses)
	core.AssertEqual(t, uint64(1), model.expertCache.stats.Evictions)
	core.AssertEqual(t, uint64(1), model.expertCache.stats.HostMappings)
	core.AssertEqual(t, gateUpPayload[gateUpSliceBytes:], driver.memory[entry1.GateUp.Pointer()])
	core.AssertEqual(t, downPayload[downSliceBytes:], driver.memory[entry1.Down.Pointer()])

	entry0Reloaded, err := model.gemma4ExpertCacheEntry(0, 0)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 6, len(driver.copies))
	core.AssertNotEqual(t, entry0, entry0Reloaded)
	core.AssertEqual(t, uint64(3), model.expertCache.stats.Misses)
	core.AssertEqual(t, uint64(2), model.expertCache.stats.Evictions)
	core.AssertEqual(t, uint64(1), model.expertCache.stats.HostMappings)
	core.AssertEqual(t, uint64(3*(gateUpSliceBytes+downSliceBytes)), model.expertCache.stats.H2DBytes)
	core.AssertEqual(t, gateUpPayload[:gateUpSliceBytes], driver.memory[entry0Reloaded.GateUp.Pointer()])
	core.AssertEqual(t, downPayload[:downSliceBytes], driver.memory[entry0Reloaded.Down.Pointer()])
	core.RequireNoError(t, model.expertCache.Close())
	core.AssertEqual(t, 0, len(model.expertCache.entries))
	core.AssertEqual(t, 0, len(model.expertCache.sources))
	core.AssertEqual(t, uint64(0), model.expertCache.bytes)
}

func TestHIPGemma4ExpertCache_MLXAffine_Good(t *testing.T) {
	const (
		experts   = 2
		hidden    = 64
		expertFF  = 32
		groupSize = 32
		bits      = 4
	)
	type tensorFixture struct {
		info    nativeTensorInfo
		payload []byte
	}
	writeTensor := func(name, typeName string, dimensions []uint64, elementBytes int, seed byte) tensorFixture {
		t.Helper()
		count := 1
		for _, dimension := range dimensions {
			count *= int(dimension)
		}
		payload := make([]byte, count*elementBytes)
		perExpert := len(payload) / experts
		for index := range payload {
			payload[index] = seed + byte(index/perExpert)
		}
		const dataOffset = 7
		const tensorOffset = 11
		filePayload := make([]byte, dataOffset+tensorOffset+len(payload)+5)
		copy(filePayload[dataOffset+tensorOffset:], payload)
		path := core.PathJoin(t.TempDir(), strings.ReplaceAll(name, ".", "_")+".safetensors")
		core.RequireTrue(t, core.WriteFile(path, filePayload, 0o644).OK)
		return tensorFixture{
			info: nativeTensorInfo{
				Name: name, TypeName: typeName, Dimensions: dimensions,
				SourcePath: path, DataOffset: dataOffset, Offset: tensorOffset,
				ByteSize: uint64(len(payload)),
			},
			payload: payload,
		}
	}
	prefix := "language_model.model.layers.0.experts.gate_up_proj"
	gateUpWeight := writeTensor(prefix+".weight", "U32", []uint64{experts, 2 * expertFF, hidden * bits / 32}, 4, 1)
	gateUpScales := writeTensor(prefix+".scales", "BF16", []uint64{experts, 2 * expertFF, hidden / groupSize}, 2, 11)
	gateUpBiases := writeTensor(prefix+".biases", "BF16", []uint64{experts, 2 * expertFF, hidden / groupSize}, 2, 21)
	prefix = "language_model.model.layers.0.experts.down_proj"
	downWeight := writeTensor(prefix+".weight", "U32", []uint64{experts, hidden, expertFF * bits / 32}, 4, 31)
	downScales := writeTensor(prefix+".scales", "BF16", []uint64{experts, hidden, expertFF / groupSize}, 2, 41)
	downBiases := writeTensor(prefix+".biases", "BF16", []uint64{experts, hidden, expertFF / groupSize}, 2, 51)
	source := hipGemma4MLXAffineExpertSource{
		GateUp: hipGemma4MLXAffineTensorSet{Weight: gateUpWeight.info, Scales: gateUpScales.info, Biases: gateUpBiases.info},
		Down:   hipGemma4MLXAffineTensorSet{Weight: downWeight.info, Scales: downScales.info, Biases: downBiases.info},
	}
	entryBytes := uint64((len(gateUpWeight.payload) + len(gateUpScales.payload) + len(gateUpBiases.payload) +
		len(downWeight.payload) + len(downScales.payload) + len(downBiases.payload)) / experts)
	driver := &fakeHIPDriver{available: true}
	cache := newHIPGemma4ExpertCache(driver, entryBytes)
	t.Cleanup(func() { core.AssertNoError(t, cache.Close()) })

	entry1, err := cache.entryMLXAffine(hipGemma4ExpertCacheKey{Layer: 0, Expert: 1}, source, experts, hidden, expertFF, groupSize, bits)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 6, len(driver.copies))
	core.AssertEqual(t, uint64(1), cache.stats.Misses)
	core.AssertEqual(t, uint64(6), cache.stats.HostMappings)
	core.AssertEqual(t, entryBytes, cache.stats.H2DBytes)
	core.AssertEqual(t, entryBytes, cache.bytes)
	core.AssertEqual(t, 1, len(cache.entries))
	core.AssertEqual(t, gateUpWeight.payload[len(gateUpWeight.payload)/experts:], driver.memory[entry1.MLXGateUpWeight.Pointer()])
	core.AssertEqual(t, gateUpScales.payload[len(gateUpScales.payload)/experts:], driver.memory[entry1.MLXGateUpScales.Pointer()])
	core.AssertEqual(t, gateUpBiases.payload[len(gateUpBiases.payload)/experts:], driver.memory[entry1.MLXGateUpBiases.Pointer()])
	core.AssertEqual(t, downWeight.payload[len(downWeight.payload)/experts:], driver.memory[entry1.MLXDownWeight.Pointer()])
	core.AssertEqual(t, downScales.payload[len(downScales.payload)/experts:], driver.memory[entry1.MLXDownScales.Pointer()])
	core.AssertEqual(t, downBiases.payload[len(downBiases.payload)/experts:], driver.memory[entry1.MLXDownBiases.Pointer()])
	core.AssertEqual(t, entry1.MLXGateUpWeight.Pointer(), entry1.MLXGate.WeightPointer)
	core.AssertEqual(t, entry1.MLXGateUpWeight.Pointer()+nativeDevicePointer(entry1.MLXGate.WeightBytes), entry1.MLXUp.WeightPointer)
	core.AssertEqual(t, entry1.MLXGateUpScales.Pointer()+nativeDevicePointer(entry1.MLXGate.ScaleBytes), entry1.MLXUp.ScalePointer)
	core.AssertEqual(t, entry1.MLXGateUpBiases.Pointer()+nativeDevicePointer(entry1.MLXGate.BiasBytes), entry1.MLXUp.BiasPointer)
	core.AssertEqual(t, hidden, entry1.MLXGate.Cols)
	core.AssertEqual(t, expertFF, entry1.MLXGate.Rows)
	core.AssertEqual(t, expertFF, entry1.MLXDown.Cols)
	core.AssertEqual(t, hidden, entry1.MLXDown.Rows)
	core.AssertEqual(t, bits, entry1.MLXGate.Bits)
	core.AssertEqual(t, groupSize, entry1.MLXGate.GroupSize)

	entry1Again, err := cache.entryMLXAffine(hipGemma4ExpertCacheKey{Layer: 0, Expert: 1}, source, experts, hidden, expertFF, groupSize, bits)
	core.RequireNoError(t, err)
	core.AssertEqual(t, entry1, entry1Again)
	core.AssertEqual(t, 6, len(driver.copies))
	core.AssertEqual(t, uint64(1), cache.stats.Hits)

	entry0, err := cache.entryMLXAffine(hipGemma4ExpertCacheKey{Layer: 0, Expert: 0}, source, experts, hidden, expertFF, groupSize, bits)
	core.RequireNoError(t, err)
	core.AssertNotEqual(t, entry1, entry0)
	core.AssertEqual(t, 12, len(driver.copies))
	core.AssertEqual(t, uint64(1), cache.stats.Evictions)
	core.AssertEqual(t, 1, len(cache.entries))
	core.AssertEqual(t, gateUpWeight.payload[:len(gateUpWeight.payload)/experts], driver.memory[entry0.MLXGateUpWeight.Pointer()])
}

func TestHIPGemma4MoELayerConfig_MLXAffine_Good(t *testing.T) {
	const (
		hidden    = 64
		experts   = 2
		expertFF  = 32
		groupSize = 32
	)
	driver := &fakeHIPDriver{available: true}
	model := &hipLoadedModel{
		driver: driver,
		modelInfo: inference.ModelInfo{
			Architecture: "diffusion_gemma", HiddenSize: hidden, NumLayers: 1,
			QuantBits: 4, QuantGroup: groupSize,
		},
		gemma4TextConfig: nativeGemma4TextConfig{
			EnableMoEBlock: true, NumExperts: experts, TopKExperts: 1, MoEIntermediateSize: expertFF,
		},
		tensors:     map[string]hipTensor{},
		hostTensors: map[string]nativeTensorInfo{},
	}
	addDeviceTensor := func(name, typeName string, dimensions []uint64, payload []byte) {
		t.Helper()
		pointer, err := driver.Malloc(uint64(len(payload)))
		core.RequireNoError(t, err)
		core.RequireNoError(t, driver.CopyHostToDevice(pointer, payload))
		model.tensors[name] = hipTensor{pointer: pointer, info: nativeTensorInfo{
			Name: name, TypeName: typeName, Dimensions: dimensions, ByteSize: uint64(len(payload)),
		}}
	}
	bf16 := func(count int, value float32) []byte {
		payload := make([]byte, count*2)
		for index := 0; index < count; index++ {
			binary.LittleEndian.PutUint16(payload[index*2:], hipFloat32ToBFloat16(value))
		}
		return payload
	}
	prefix := "language_model.model.layers.0"
	for _, name := range []string{
		prefix + ".pre_feedforward_layernorm_2.weight",
		prefix + ".post_feedforward_layernorm_1.weight",
		prefix + ".post_feedforward_layernorm_2.weight",
		prefix + ".router.scale",
	} {
		addDeviceTensor(name, "BF16", []uint64{hidden}, bf16(hidden, 1))
	}
	addDeviceTensor(prefix+".router.per_expert_scale", "BF16", []uint64{experts}, bf16(experts, 1))
	routerPackedCols := hidden * 8 / 32
	addDeviceTensor(prefix+".router.proj.weight", "U32", []uint64{experts, uint64(routerPackedCols)}, make([]byte, experts*routerPackedCols*4))
	addDeviceTensor(prefix+".router.proj.scales", "BF16", []uint64{experts, hidden / groupSize}, bf16(experts*(hidden/groupSize), 1))
	addDeviceTensor(prefix+".router.proj.biases", "BF16", []uint64{experts, hidden / groupSize}, bf16(experts*(hidden/groupSize), 0))
	addHostTensor := func(name, typeName string, dimensions []uint64, elementBytes int) {
		count := 1
		for _, dimension := range dimensions {
			count *= int(dimension)
		}
		model.hostTensors[name] = nativeTensorInfo{
			Name: name, TypeName: typeName, Dimensions: dimensions,
			SourcePath: "/tmp/diffusiongemma.safetensors", ByteSize: uint64(count * elementBytes),
		}
	}
	gateUp := prefix + ".experts.gate_up_proj"
	addHostTensor(gateUp+".weight", "U32", []uint64{experts, 2 * expertFF, hidden * 4 / 32}, 4)
	addHostTensor(gateUp+".scales", "BF16", []uint64{experts, 2 * expertFF, hidden / groupSize}, 2)
	addHostTensor(gateUp+".biases", "BF16", []uint64{experts, 2 * expertFF, hidden / groupSize}, 2)
	down := prefix + ".experts.down_proj"
	addHostTensor(down+".weight", "U32", []uint64{experts, hidden, expertFF * 4 / 32}, 4)
	addHostTensor(down+".scales", "BF16", []uint64{experts, hidden, expertFF / groupSize}, 2)
	addHostTensor(down+".biases", "BF16", []uint64{experts, hidden, expertFF / groupSize}, 2)
	t.Cleanup(func() {
		if model.expertCache != nil {
			core.AssertNoError(t, model.expertCache.Close())
		}
		for _, tensor := range model.tensors {
			core.AssertNoError(t, driver.Free(tensor.pointer))
		}
	})

	moe, err := model.loadedGemma4MoELayerConfig(0, hidden)
	core.RequireNoError(t, err)
	core.RequireTrue(t, moe != nil)
	core.AssertEqual(t, hipGemma4MoEExpertStorageMLXAffine, moe.ExpertStorage)
	core.AssertEqual(t, nativeDevicePointer(0), moe.RouterProjection.WeightPointer)
	core.AssertEqual(t, experts, moe.RouterProjectionMLX.Rows)
	core.AssertEqual(t, hidden, moe.RouterProjectionMLX.Cols)
	core.AssertEqual(t, 8, moe.RouterProjectionMLX.Bits)
	core.AssertEqual(t, groupSize, moe.RouterProjectionMLX.GroupSize)
	core.AssertEqual(t, gateUp+".weight", moe.MLXExperts.GateUp.Weight.Name)
	core.AssertEqual(t, down+".weight", moe.MLXExperts.Down.Weight.Name)
	core.RequireTrue(t, moe.ExpertCache != nil)
}

func TestHIPGemma4ExpertCache_Good_EvictsOnDevicePressure(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	const (
		experts  = 2
		hidden   = 32
		expertFF = 32
	)
	gateUpSliceBytes := 2 * expertFF * (hidden / hipGGUFQ4_0BlockSize) * hipGGUFQ4_0BlockBytes
	downSliceBytes := hidden * (expertFF / hipGGUFQ4_0BlockSize) * hipGGUFQ4_0BlockBytes
	entryBytes := gateUpSliceBytes + downSliceBytes
	gateUpPayload := make([]byte, experts*gateUpSliceBytes)
	downPayload := make([]byte, experts*downSliceBytes)
	for index := range gateUpPayload {
		gateUpPayload[index] = byte(index/gateUpSliceBytes + 1)
	}
	for index := range downPayload {
		downPayload[index] = byte(index/downSliceBytes + 11)
	}
	filePayload := append(append(make([]byte, 0, len(gateUpPayload)+len(downPayload)), gateUpPayload...), downPayload...)
	path := core.PathJoin(t.TempDir(), "experts.gguf")
	core.RequireTrue(t, core.WriteFile(path, filePayload, 0o644).OK)
	driver := &fakeHIPDriver{available: true, maxLiveBytes: uint64(entryBytes)}
	cache := newHIPGemma4ExpertCache(driver, uint64(2*entryBytes))
	t.Cleanup(func() { core.AssertNoError(t, cache.Close()) })
	gateUpInfo := nativeTensorInfo{
		Name: "blk.0.ffn_gate_up_exps.weight", Type: hipGGUFQ4_0TensorType, TypeName: "Q4_0",
		Dimensions: []uint64{hidden, 2 * expertFF, experts}, SourcePath: path,
		ByteSize: uint64(len(gateUpPayload)),
	}
	downInfo := nativeTensorInfo{
		Name: "blk.0.ffn_down_exps.weight", Type: hipGGUFQ4_0TensorType, TypeName: "Q4_0",
		Dimensions: []uint64{expertFF, hidden, experts}, SourcePath: path,
		Offset: uint64(len(gateUpPayload)), ByteSize: uint64(len(downPayload)),
	}

	entry0, err := cache.entry(hipGemma4ExpertCacheKey{Layer: 0, Expert: 0}, gateUpInfo, downInfo, experts)
	core.RequireNoError(t, err)
	entry1, err := cache.entry(hipGemma4ExpertCacheKey{Layer: 0, Expert: 1}, gateUpInfo, downInfo, experts)
	core.RequireNoError(t, err)
	core.AssertNotEqual(t, entry0, entry1)
	core.AssertEqual(t, 1, len(cache.entries))
	core.AssertEqual(t, uint64(1), cache.stats.Evictions)
	core.AssertEqual(t, uint64(1), cache.stats.AllocationRetries)
	core.AssertEqual(t, uint64(entryBytes), cache.bytes)
	core.AssertEqual(t, uint64(entryBytes), driver.liveBytes)
	core.AssertEqual(t, gateUpPayload[gateUpSliceBytes:], driver.memory[entry1.GateUp.Pointer()])
	core.AssertEqual(t, downPayload[downSliceBytes:], driver.memory[entry1.Down.Pointer()])
}

func TestHIPGemma4ExpertCache_Good_AdaptsToLiveHeadroom(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	const entryBytes = uint64(1024)
	driver := &fakeHIPDriver{
		available: true,
		device:    nativeDeviceInfo{FreeBytes: 12 * memoryGiB},
	}
	cache := newHIPGemma4AdaptiveExpertCache(driver, 1)
	t.Cleanup(func() { core.AssertNoError(t, cache.Close()) })
	for expert := 0; expert < 3; expert++ {
		gateUp, err := hipAllocateByteBuffer(driver, "rocm.hip.Gemma4ExpertCache", "adaptive gate/up", entryBytes/2, int(entryBytes/2))
		core.RequireNoError(t, err)
		down, err := hipAllocateByteBuffer(driver, "rocm.hip.Gemma4ExpertCache", "adaptive down", entryBytes/2, int(entryBytes/2))
		core.RequireNoError(t, err)
		cache.entries[hipGemma4ExpertCacheKey{Layer: 0, Expert: expert}] = &hipGemma4ExpertCacheEntry{
			GateUp: gateUp, Down: down, bytes: entryBytes, lastUse: uint64(expert + 1),
		}
		cache.bytes += entryBytes
	}
	driver.device.FreeBytes = hipGemma4ExpertCacheReserveBytes - 2*entryBytes
	core.RequireNoError(t, cache.refreshAdaptiveBudget(entryBytes))
	core.AssertEqual(t, 1, len(cache.entries))
	core.AssertEqual(t, uint64(entryBytes), cache.bytes)
	core.AssertEqual(t, uint64(entryBytes), cache.maxBytes)
	core.AssertEqual(t, uint64(2), cache.stats.Evictions)
	core.AssertEqual(t, uint64(1), cache.stats.BudgetRefreshes)
	core.RequireTrue(t, cache.entries[hipGemma4ExpertCacheKey{Layer: 0, Expert: 2}] != nil)
}

func TestHIPGemma4ExpertCache_Good_DefersEvictedBufferRelease(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	driver := &hipGemma4MoEWorkspaceTestDriver{fakeHIPDriver: &fakeHIPDriver{available: true}}
	cache := newHIPGemma4ExpertCache(driver, 64)
	t.Cleanup(func() { core.AssertNoError(t, cache.Close()) })
	gateUp, err := hipAllocateByteBuffer(driver, "test", "deferred gate/up", 32, 32)
	core.RequireNoError(t, err)
	down, err := hipAllocateByteBuffer(driver, "test", "deferred down", 32, 32)
	core.RequireNoError(t, err)
	cache.entries[hipGemma4ExpertCacheKey{Layer: 0, Expert: 0}] = &hipGemma4ExpertCacheEntry{
		GateUp: gateUp, Down: down, bytes: 64, lastUse: 1,
	}
	cache.bytes = 64

	core.RequireNoError(t, cache.beginDeferredEvictions())
	core.RequireNoError(t, cache.evictOldest())
	core.AssertEqual(t, 0, len(cache.entries))
	core.AssertEqual(t, uint64(0), cache.bytes)
	core.AssertEqual(t, 1, len(cache.deferredEvictions))
	core.AssertEqual(t, 0, len(driver.frees))
	core.AssertEqual(t, uint64(64), driver.liveBytes)

	core.RequireNoError(t, hipSynchronizeGemma4MoEExpertUse(driver))
	core.RequireNoError(t, cache.endDeferredEvictions())
	core.AssertEqual(t, 1, driver.synchronizes)
	core.AssertEqual(t, 0, len(cache.deferredEvictions))
	core.AssertEqual(t, 2, len(driver.frees))
	core.AssertEqual(t, uint64(0), driver.liveBytes)
}

func TestHIPGemma4ExpertCache_Good_ReclaimsDeferredBuffersOnAllocationPressure(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	driver := &hipGemma4MoEWorkspaceTestDriver{fakeHIPDriver: &fakeHIPDriver{available: true, maxLiveBytes: 64}}
	cache := newHIPGemma4ExpertCache(driver, 64)
	t.Cleanup(func() { core.AssertNoError(t, cache.Close()) })
	gateUp, err := hipAllocateByteBuffer(driver, "test", "pressure gate/up", 32, 32)
	core.RequireNoError(t, err)
	down, err := hipAllocateByteBuffer(driver, "test", "pressure down", 32, 32)
	core.RequireNoError(t, err)
	cache.entries[hipGemma4ExpertCacheKey{Layer: 0, Expert: 0}] = &hipGemma4ExpertCacheEntry{
		GateUp: gateUp, Down: down, bytes: 64, lastUse: 1,
	}
	cache.bytes = 64

	core.RequireNoError(t, cache.beginDeferredEvictions())
	core.RequireNoError(t, cache.evictOldest())
	replacement, err := cache.allocateExpertBuffer("replacement expert", 64, 64)
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint64(1), cache.stats.AllocationRetries)
	core.AssertEqual(t, 1, driver.synchronizes)
	core.AssertEqual(t, 0, len(cache.deferredEvictions))
	core.AssertEqual(t, 2, len(driver.frees))
	core.AssertEqual(t, uint64(64), driver.liveBytes)
	core.RequireNoError(t, replacement.Close())
	core.RequireNoError(t, cache.endDeferredEvictions())
	core.AssertEqual(t, 0, cache.deferredEvictionDepth)
	core.AssertEqual(t, uint64(0), driver.liveBytes)
}

func TestHIPGemma4ExpertCache_Good_AdaptiveCacheSuppressesTransientPool(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "")
	driver := &fakeHIPDriver{available: true}
	hipDeviceByteBufferPool.Lock()
	hipDeviceByteBufferPool.single = [hipDeviceByteBufferPoolSingleSlots]hipDeviceByteBufferPoolSingleSlot{}
	hipDeviceByteBufferPool.entries = make(map[uint64][]hipDeviceByteBufferPoolEntry)
	hipDeviceByteBufferPool.bytes = 0
	hipDeviceByteBufferPool.Unlock()
	t.Cleanup(func() {
		hipDeviceByteBufferPool.Lock()
		hipDeviceByteBufferPool.single = [hipDeviceByteBufferPoolSingleSlots]hipDeviceByteBufferPoolSingleSlot{}
		hipDeviceByteBufferPool.entries = make(map[uint64][]hipDeviceByteBufferPoolEntry)
		hipDeviceByteBufferPool.bytes = 0
		hipDeviceByteBufferPool.Unlock()
	})
	core.RequireTrue(t, hipDeviceByteBufferPoolPut(driver, 42, 4096))

	cache := newHIPGemma4AdaptiveExpertCache(driver, 1)
	core.RequireTrue(t, !hipDeviceByteBufferPoolEnabled())
	core.AssertEqual(t, []nativeDevicePointer{42}, driver.frees)
	core.RequireTrue(t, !hipDeviceByteBufferPoolPut(driver, 43, 4096))
	core.RequireNoError(t, cache.Close())
	core.RequireTrue(t, hipDeviceByteBufferPoolEnabled())
}

func TestHIPGemma4ExpertCacheBudget_Good(t *testing.T) {
	core.AssertEqual(t, uint64(11*memoryGiB), hipGemma4ExpertCacheBudget(&fakeHIPDriver{available: true, device: nativeDeviceInfo{FreeBytes: 12 * memoryGiB}}))
	core.AssertEqual(t, uint64(10*memoryGiB), hipGemma4ExpertCacheBudget(&fakeHIPDriver{available: true, device: nativeDeviceInfo{FreeBytes: 11 * memoryGiB}}))
	core.AssertEqual(t, uint64(6*memoryGiB), hipGemma4ExpertCacheBudget(&fakeHIPDriver{available: true, device: nativeDeviceInfo{FreeBytes: 7 * memoryGiB}}))
	core.AssertEqual(t, uint64(6*memoryGiB), hipGemma4ExpertCacheBudget(&fakeHIPDriver{available: true}))
}

func TestHIPGemma4MoEWorkspace_Good_ReusesDeviceBuffers(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := hipNewAttentionHeadsChunkedWorkspace()
	defer workspace.Close()

	expertInput, err := workspace.EnsureMoEHiddenOutput(driver, 2816, 0)
	core.RequireNoError(t, err)
	routerInput, err := workspace.EnsureMoEHiddenOutput(driver, 2816, 1)
	core.RequireNoError(t, err)
	routerScores, err := workspace.EnsureMoERouterScores(driver, 8)
	core.RequireNoError(t, err)
	allocationsAfterWarm := len(driver.allocations)

	expertInputAgain, err := workspace.EnsureMoEHiddenOutput(driver, 2816, 0)
	core.RequireNoError(t, err)
	routerInputAgain, err := workspace.EnsureMoEHiddenOutput(driver, 2816, 1)
	core.RequireNoError(t, err)
	routerScoresAgain, err := workspace.EnsureMoERouterScores(driver, 8)
	core.RequireNoError(t, err)

	core.AssertEqual(t, allocationsAfterWarm, len(driver.allocations))
	core.AssertEqual(t, expertInput.Pointer(), expertInputAgain.Pointer())
	core.AssertEqual(t, routerInput.Pointer(), routerInputAgain.Pointer())
	core.AssertEqual(t, routerScores.Pointer(), routerScoresAgain.Pointer())
	core.AssertNotEqual(t, expertInput.Pointer(), routerInput.Pointer())
}

func TestHIPMoERouterLaunch_Good_DeviceInputWorkspaceReusesPackedOutput(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	logitPayload, err := hipFloat32Payload([]float32{0.1, 2, 1, -1})
	core.RequireNoError(t, err)
	logits, err := hipUploadByteBuffer(driver, "test", "router logits", logitPayload, 4)
	core.RequireNoError(t, err)
	defer logits.Close()
	workspace := hipNewAttentionHeadsChunkedWorkspace()
	defer workspace.Close()

	first, err := hipRunMoERouterKernelWithDeviceInputWorkspace(context.Background(), driver, logits, 2, 7, workspace)
	core.RequireNoError(t, err)
	allocationsAfterWarm := len(driver.allocations)
	copyCountAfterWarm := len(driver.copies)
	second, err := hipRunMoERouterKernelWithDeviceInputWorkspace(context.Background(), driver, logits, 2, 7, workspace)
	core.RequireNoError(t, err)

	core.AssertEqual(t, allocationsAfterWarm, len(driver.allocations))
	core.AssertEqual(t, []uint64{20}, driver.copies[copyCountAfterWarm:])
	core.AssertEqual(t, first, second)
	core.AssertEqual(t, 2, len(second.Routes))
	core.AssertEqual(t, 1, second.Routes[0].ID)
	core.AssertEqual(t, 2, second.Routes[1].ID)
}

func TestHIPGemma4MoEWorkspace_Good_WarmForwardReusesDeviceBuffers(t *testing.T) {
	driver := &hipGemma4MoEWorkspaceTestDriver{fakeHIPDriver: &fakeHIPDriver{available: true}}
	workspace := hipNewAttentionHeadsChunkedWorkspace()
	defer workspace.Close()
	attentionResidual, localInput, layer, cleanup := hipGemma4MoEWorkspaceFixture(t, driver, 1)
	defer cleanup()

	first, err := hipRunGemma4MoEDeviceMLPWithWorkspace(context.Background(), driver, attentionResidual, localInput, layer, 1e-6, workspace)
	if err != nil {
		t.Fatalf("warm MoE forward: %v", err)
	}
	core.RequireNoError(t, first.Close())
	allocationsAfterWarm := len(driver.allocations)
	driver.launches = driver.launches[:0]
	second, err := hipRunGemma4MoEDeviceMLPWithWorkspace(context.Background(), driver, attentionResidual, localInput, layer, 1e-6, workspace)
	if err != nil {
		t.Fatalf("reused MoE forward: %v", err)
	}
	core.RequireNoError(t, second.Close())

	core.AssertEqual(t, allocationsAfterWarm, len(driver.allocations))
	combineLaunches := 0
	postNormLaunches := 0
	vectorAddLaunches := 0
	for _, launch := range driver.launches {
		switch launch.Name {
		case hipKernelNameMoECombineNorms:
			combineLaunches++
		case hipKernelNameRMSNorm:
			postNormLaunches++
		case hipKernelNameVectorAddScaled:
			vectorAddLaunches++
		}
	}
	core.AssertEqual(t, 1, combineLaunches)
	core.AssertEqual(t, 2, postNormLaunches)
	core.AssertEqual(t, 0, vectorAddLaunches)
}

func TestHIPGemma4MoEWorkspace_Good_BatchRowsUseDistinctOutput(t *testing.T) {
	const rows = 2
	driver := &hipGemma4MoEWorkspaceTestDriver{fakeHIPDriver: &fakeHIPDriver{available: true}}
	workspace := hipNewAttentionHeadsChunkedWorkspace()
	defer workspace.Close()
	attentionResidual, localInput, layer, cleanup := hipGemma4MoEWorkspaceFixture(t, driver, rows)
	defer cleanup()

	first, err := hipRunGemma4MoEDeviceMLPBatchWithWorkspace(
		context.Background(), driver, attentionResidual, localInput, layer, 1e-6, rows, workspace,
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, rows*layer.HiddenSize, first.Count())
	core.AssertEqual(t, uint64(rows*layer.HiddenSize*4), first.SizeBytes())
	firstPointer := first.Pointer()
	allocationsAfterWarm := len(driver.allocations)
	driver.launches = driver.launches[:0]

	second, err := hipRunGemma4MoEDeviceMLPBatchWithWorkspace(
		context.Background(), driver, attentionResidual, localInput, layer, 1e-6, rows, workspace,
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, firstPointer, second.Pointer())
	core.AssertEqual(t, allocationsAfterWarm, len(driver.allocations))
	core.AssertEqual(t, rows, countLaunchName(driver.launches, hipKernelNameMoECombineNorms))
}

func TestHIPGemma4MoEWorkspace_Good_MLXAffineBatchDefaultsToGroupedExperts(t *testing.T) {
	t.Setenv("GO_ROCM_GEMMA4_MOE_MLX_AFFINE_ROUTES", "")
	const rows = 10
	driver := &hipGemma4MoEWorkspaceTestDriver{fakeHIPDriver: &fakeHIPDriver{available: true}}
	workspace := hipNewAttentionHeadsChunkedWorkspace()
	defer workspace.Close()
	attentionResidual, localInput, layer, cleanup := hipGemma4MoEMLXWorkspaceFixture(t, driver, rows)
	defer cleanup()

	output, err := hipRunGemma4MoEDeviceMLPBatchWithWorkspace(
		context.Background(), driver, attentionResidual, localInput, layer, 1e-6, rows, workspace,
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, rows*layer.HiddenSize, output.Count())
	core.AssertEqual(t, 0, countLaunchName(driver.launches, hipKernelNameMoERouter))
	core.AssertEqual(t, 0, countLaunchName(driver.launches, hipKernelNameMoECombineNorms))
	core.AssertEqual(t, 2, countLaunchName(driver.launches, hipKernelNameMoEBatchGatherRows))
	core.AssertEqual(t, 2, countLaunchName(driver.launches, hipKernelNameMoEBatchScatterRoutes))
	core.AssertEqual(t, 1, countLaunchName(driver.launches, hipKernelNameMoEBatchReduceRoutes))
	core.AssertEqual(t, 4, countLaunchName(driver.launches, hipKernelNameRMSNormHeads))
	core.AssertEqual(t, 3, countLaunchName(driver.launches, hipKernelNameMLXQ4GELUTanhMulBatch))
	core.AssertEqual(t, 4, countLaunchName(driver.launches, hipKernelNameMLXQ4ProjBatch))
	core.AssertEqual(t, 0, countLaunchName(driver.launches, hipKernelNameMoEMLXAffineRoutes))
	core.AssertEqual(t, 1, countLaunchName(driver.launches, hipKernelNameVectorAddScaled))
	core.AssertEqual(t, 1, driver.synchronizes)
}

func TestHIPGemma4MoEWorkspace_Good_MLXAffineBatchCanEnableAllRoutes(t *testing.T) {
	t.Setenv("GO_ROCM_GEMMA4_MOE_MLX_AFFINE_ROUTES", "1")
	const rows = 2
	driver := &hipGemma4MoEWorkspaceTestDriver{fakeHIPDriver: &fakeHIPDriver{available: true}}
	workspace := hipNewAttentionHeadsChunkedWorkspace()
	defer workspace.Close()
	attentionResidual, localInput, layer, cleanup := hipGemma4MoEMLXWorkspaceFixture(t, driver, rows)
	defer cleanup()

	output, err := hipRunGemma4MoEDeviceMLPBatchWithWorkspace(
		context.Background(), driver, attentionResidual, localInput, layer, 1e-6, rows, workspace,
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, rows*layer.HiddenSize, output.Count())
	core.AssertEqual(t, 2, countLaunchName(driver.launches, hipKernelNameMoEMLXAffineRoutes))
	core.AssertEqual(t, 0, countLaunchName(driver.launches, hipKernelNameMoEBatchGatherRows))
	core.AssertEqual(t, 0, countLaunchName(driver.launches, hipKernelNameMoEBatchScatterRoutes))
	core.AssertEqual(t, 1, countLaunchName(driver.launches, hipKernelNameMLXQ4GELUTanhMulBatch))
	core.AssertEqual(t, 2, countLaunchName(driver.launches, hipKernelNameMLXQ4ProjBatch))
	core.AssertEqual(t, 1, driver.synchronizes)
}

func TestHIPGemma4MoEWorkspace_Bad_MLXAffineBatchSyncFailureReleasesScope(t *testing.T) {
	const rows = 2
	driver := &hipGemma4MoEWorkspaceTestDriver{
		fakeHIPDriver:  &fakeHIPDriver{available: true},
		synchronizeErr: core.E("rocm.hip.Test", "device synchronization failed", nil),
	}
	workspace := hipNewAttentionHeadsChunkedWorkspace()
	defer workspace.Close()
	attentionResidual, localInput, layer, cleanup := hipGemma4MoEMLXWorkspaceFixture(t, driver, rows)
	defer cleanup()

	_, err := hipRunGemma4MoEDeviceMLPBatchWithWorkspace(
		context.Background(), driver, attentionResidual, localInput, layer, 1e-6, rows, workspace,
	)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finish grouped expert device work")
	core.AssertEqual(t, 1, driver.synchronizes)
	core.AssertEqual(t, 0, layer.MoE.ExpertCache.deferredEvictionDepth)
}

func TestHIPGemma4MoEBatchRouteGroups_Good_MatchesReference(t *testing.T) {
	const (
		rows    = 3
		experts = 4
		topK    = 2
	)
	scores := []float32{
		1, 4, 2, 3,
		5, 5, -1, 0,
		-2, 1, 7, 3,
	}
	scales := []float32{1, 0.5, 2, 1.5}
	groups, err := hipGemma4MoEBatchRouteGroups(scores, rows, experts, topK, scales, nil)
	core.RequireNoError(t, err)
	core.AssertEqual(t, experts, len(groups))

	for row := 0; row < rows; row++ {
		want, referenceErr := rocmReferenceRouteExperts(scores[row*experts:(row+1)*experts], topK, 0, nil)
		core.RequireNoError(t, referenceErr)
		got := make(map[int]float32, topK)
		for expert, routes := range groups {
			for _, route := range routes {
				if route.Row == row {
					got[expert] = route.Weight
				}
			}
		}
		core.AssertEqual(t, topK, len(got))
		for _, route := range want {
			assertFloat32Near(t, route.Prob*scales[route.ID], got[route.ID])
		}
	}
}

func TestHIPGemma4MoEAffineRouteChunks_Good_PreservesGroupedPairs(t *testing.T) {
	t.Setenv("GO_ROCM_GEMMA4_MOE_MLX_AFFINE_ROUTES", "1")
	const rows = 3
	driver := &hipGemma4MoEWorkspaceTestDriver{fakeHIPDriver: &fakeHIPDriver{available: true}}
	_, _, layer, cleanup := hipGemma4MoEMLXWorkspaceFixture(t, driver, rows)
	defer cleanup()
	groups := make([][]hipGemma4MoEBatchRoute, layer.MoE.NumExperts)
	groups[0] = []hipGemma4MoEBatchRoute{
		{Row: 2, Pair: 1, Weight: 0.75},
		{Row: 0, Pair: 0, Weight: 0.25},
	}
	groups[1] = []hipGemma4MoEBatchRoute{{Row: 1, Pair: 2, Weight: 1}}

	chunks, bits, useAllRoutes, err := hipPrepareGemma4MoEMLXAffineRouteChunks(layer.MoE, groups, rows, 3, nil)
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, useAllRoutes)
	core.AssertEqual(t, 4, bits)
	if len(chunks) != 2 {
		t.Fatalf("hipPrepareGemma4MoEMLXAffineRouteChunks returned %d chunks, want 2", len(chunks))
	}
	core.AssertEqual(t, 2, chunks[0].RouteCount)
	core.AssertEqual(t, 2, chunks[0].TokenRows[0])
	core.AssertEqual(t, 1, chunks[0].PairIndices[0])
	assertFloat32Near(t, 0.75, chunks[0].RouteWeights[0])
	core.AssertEqual(t, 1, chunks[1].RouteCount)
	core.AssertEqual(t, 1, chunks[1].TokenRows[0])
	core.AssertEqual(t, 2, chunks[1].PairIndices[0])
	entry0 := layer.MoE.ExpertCache.entries[hipGemma4ExpertCacheKey{Layer: 0, Expert: 0}]
	entry1 := layer.MoE.ExpertCache.entries[hipGemma4ExpertCacheKey{Layer: 0, Expert: 1}]
	core.AssertEqual(t, entry0.MLXGate.WeightPointer, chunks[0].GateUpWeightPointer)
	core.AssertEqual(t, entry1.MLXDown.WeightPointer, chunks[1].DownWeightPointer)
}

func TestHIPGemma4MoEAffineRouteChunks_Good_FallsBackWhenWorkingSetExceedsCache(t *testing.T) {
	const rows = 2
	driver := &hipGemma4MoEWorkspaceTestDriver{fakeHIPDriver: &fakeHIPDriver{available: true}}
	_, _, layer, cleanup := hipGemma4MoEMLXWorkspaceFixture(t, driver, rows)
	defer cleanup()
	layer.MoE.ExpertCache.maxBytes = 1
	groups := make([][]hipGemma4MoEBatchRoute, layer.MoE.NumExperts)
	groups[0] = []hipGemma4MoEBatchRoute{{Row: 0, Pair: 0, Weight: 1}}
	groups[1] = []hipGemma4MoEBatchRoute{{Row: 1, Pair: 1, Weight: 1}}

	chunks, bits, useAllRoutes, err := hipPrepareGemma4MoEMLXAffineRouteChunks(layer.MoE, groups, rows, 2, nil)
	core.RequireNoError(t, err)
	core.AssertEqual(t, false, useAllRoutes)
	core.AssertEqual(t, 0, bits)
	core.AssertEqual(t, 0, len(chunks))
}

func TestHIPGemma4MoEBatchRouteGroups_Bad_RejectsShape(t *testing.T) {
	if _, err := hipGemma4MoEBatchRouteGroups([]float32{1, 2, 3}, 2, 2, 1, []float32{1, 1}, nil); err == nil {
		t.Fatal("hipGemma4MoEBatchRouteGroups accepted a truncated score slab")
	}
}

func TestHIPGemma4MoEBatchRouteGroups_Ugly_RejectsNonFiniteScores(t *testing.T) {
	if _, err := hipGemma4MoEBatchRouteGroups([]float32{1, float32(math.NaN())}, 1, 2, 1, []float32{1, 1}, nil); err == nil {
		t.Fatal("hipGemma4MoEBatchRouteGroups accepted NaN scores")
	}
}

func TestHIPGemma4MoEBatchHardwareMatchesSerial_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MOE_LANE_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MOE_LANE_TESTS=1 to run the HIP MoE batch receipt")
	}
	modelPath := strings.TrimSpace(os.Getenv("GO_ROCM_MODEL_PATH"))
	if modelPath == "" {
		t.Skip("set GO_ROCM_MODEL_PATH to a linked Gemma-4 26B-A4B Q4 GGUF")
	}
	if strings.TrimSpace(os.Getenv("GO_ROCM_KERNEL_HSACO")) == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to the linked ROCm kernels HSACO")
	}
	if !ROCmAvailable() {
		t.Skip("ROCm runtime is not available on this host")
	}

	loadedResult := (&rocmBackend{}).LoadModel(modelPath, inference.WithContextLen(64))
	if !loadedResult.OK {
		t.Fatalf("production ROCm LoadModel(%q): %v", modelPath, loadedResult.Value)
	}
	model, ok := loadedResult.Value.(*rocmModel)
	if !ok {
		t.Fatalf("production ROCm LoadModel returned %T, want *rocmModel", loadedResult.Value)
	}
	defer func() {
		if result := model.Close(); !result.OK {
			t.Errorf("Close model: %v", result.Value)
		}
	}()
	loaded, ok := model.native.(*hipLoadedModel)
	if !ok || loaded == nil {
		t.Fatalf("production ROCm native model is %T, want *hipLoadedModel", model.native)
	}
	forward, err := loaded.cachedGemma4Q4ForwardConfig(loaded.modelInfo.NumLayers)
	core.RequireNoError(t, err)
	var layer *hipGemma4Q4Layer0Config
	for index := range forward.Layers {
		if forward.Layers[index].MoE != nil {
			layer = &forward.Layers[index]
			break
		}
	}
	if layer == nil {
		t.Fatal("production model has no sparse layer")
	}

	const rows = 2
	hidden := layer.HiddenSize
	attentionValues := make([]float32, rows*hidden)
	localValues := make([]float32, rows*hidden)
	for row := 0; row < rows; row++ {
		for column := 0; column < hidden; column++ {
			index := row*hidden + column
			attentionValues[index] = float32(column%29-14)/64 + float32(row)/32
			localValues[index] = float32(column%17-8)/32 - float32(row)/64
		}
	}
	attentionPayload, err := hipFloat32Payload(attentionValues)
	core.RequireNoError(t, err)
	attention, err := hipUploadByteBuffer(loaded.driver, "rocm.hip.Gemma4MoEBatchHardware", "attention rows", attentionPayload, len(attentionValues))
	core.RequireNoError(t, err)
	defer attention.Close()
	localPayload, err := hipFloat32Payload(localValues)
	core.RequireNoError(t, err)
	local, err := hipUploadByteBuffer(loaded.driver, "rocm.hip.Gemma4MoEBatchHardware", "local rows", localPayload, len(localValues))
	core.RequireNoError(t, err)
	defer local.Close()

	t.Setenv(hipGemma4MoEMLXAffineRoutesEnv, "")
	groupedWorkspace := hipBorrowAttentionHeadsChunkedWorkspace()
	defer func() { core.RequireNoError(t, hipRecycleAttentionHeadsChunkedWorkspace(groupedWorkspace)) }()
	batch, err := hipRunGemma4MoEDeviceMLPBatchWithWorkspace(context.Background(), loaded.driver, attention, local, *layer, 1e-6, rows, groupedWorkspace)
	core.RequireNoError(t, err)
	grouped, err := hipReadFloat32DeviceOutput(batch, "rocm.hip.Gemma4MoEBatchHardware", "grouped batch output", rows*hidden)
	core.RequireNoError(t, err)
	rowBytes := uint64(hidden * 4)
	attentionRow := hipBorrowDeviceByteBufferValue(loaded.driver, "reused serial attention row", attention.Pointer(), rowBytes, hidden)
	localRow := hipBorrowDeviceByteBufferValue(loaded.driver, "reused serial local row", local.Pointer(), rowBytes, hidden)
	reusedSingle, err := hipRunGemma4MoEDeviceMLPWithWorkspaceOutput(context.Background(), loaded.driver, &attentionRow, &localRow, *layer, 1e-6, groupedWorkspace, nil)
	core.RequireNoError(t, err)
	reused, err := hipReadFloat32DeviceOutput(reusedSingle, "rocm.hip.Gemma4MoEBatchHardware", "batch then single output", hidden)
	core.RequireNoError(t, err)

	t.Setenv(hipGemma4MoEMLXAffineRoutesEnv, "1")
	fusedWorkspace := hipNewAttentionHeadsChunkedWorkspace()
	defer fusedWorkspace.Close()
	batch, err = hipRunGemma4MoEDeviceMLPBatchWithWorkspace(context.Background(), loaded.driver, attention, local, *layer, 1e-6, rows, fusedWorkspace)
	core.RequireNoError(t, err)
	fused, err := hipReadFloat32DeviceOutput(batch, "rocm.hip.Gemma4MoEBatchHardware", "fused batch output", rows*hidden)
	core.RequireNoError(t, err)

	t.Setenv(hipGemma4MoEMLXAffineRoutesEnv, "")
	want := make([]float32, 0, rows*hidden)
	for row := 0; row < rows; row++ {
		offset := nativeDevicePointer(uint64(row) * rowBytes)
		attentionRow := hipBorrowDeviceByteBufferValue(loaded.driver, "serial attention row", attention.Pointer()+offset, rowBytes, hidden)
		localRow := hipBorrowDeviceByteBufferValue(loaded.driver, "serial local row", local.Pointer()+offset, rowBytes, hidden)
		workspace := hipNewAttentionHeadsChunkedWorkspace()
		serial, runErr := hipRunGemma4MoEDeviceMLPWithWorkspaceOutput(context.Background(), loaded.driver, &attentionRow, &localRow, *layer, 1e-6, workspace, nil)
		core.RequireNoError(t, runErr)
		values, readErr := hipReadFloat32DeviceOutput(serial, "rocm.hip.Gemma4MoEBatchHardware", "serial output", hidden)
		core.RequireNoError(t, readErr)
		want = append(want, values...)
		core.RequireNoError(t, workspace.Close())
	}
	compare := func(label string, got []float32) {
		t.Helper()
		maxAbs := float64(0)
		maxRel := float64(0)
		maxAbsIndex := 0
		maxRelIndex := 0
		for index := range want {
			difference := math.Abs(float64(got[index] - want[index]))
			scale := math.Max(math.Abs(float64(want[index])), 1)
			if difference > maxAbs {
				maxAbs = difference
				maxAbsIndex = index
			}
			if relative := difference / scale; relative > maxRel {
				maxRel = relative
				maxRelIndex = index
			}
		}
		t.Logf("%s max_abs=%g index=%d max_rel=%g index=%d", label, maxAbs, maxAbsIndex, maxRel, maxRelIndex)
		assertFloat32SlicesNearRelative(t, want, got, 1e-4, 1e-6)
	}
	compare("grouped versus serial", grouped)
	compare("fused versus serial", fused)
	assertFloat32SlicesNearRelative(t, want[:hidden], reused, 1e-4, 1e-6)
}

type hipGemma4MoEWorkspaceTestDriver struct {
	*fakeHIPDriver
	synchronizes   int
	synchronizeErr error
}

func hipGemma4MoEWorkspaceFixture(t *testing.T, driver nativeHIPDriver, rows int) (*hipDeviceByteBuffer, *hipDeviceByteBuffer, hipGemma4Q4Layer0Config, func()) {
	t.Helper()
	const (
		hidden   = 32
		localFF  = 32
		experts  = 2
		topK     = 2
		expertFF = 32
	)
	attentionResidual, err := hipAllocateByteBuffer(driver, "test", "attention residual", uint64(rows*hidden*4), rows*hidden)
	core.RequireNoError(t, err)
	localInput, err := hipAllocateByteBuffer(driver, "test", "local input", uint64(rows*hidden*4), rows*hidden)
	core.RequireNoError(t, err)
	expertCache := newHIPGemma4ExpertCache(driver, 1<<20)
	gateUpBytes := uint64(2 * expertFF * (hidden / hipGGUFQ4_0BlockSize) * hipGGUFQ4_0BlockBytes)
	downBytes := uint64(hidden * (expertFF / hipGGUFQ4_0BlockSize) * hipGGUFQ4_0BlockBytes)
	for expert := 0; expert < experts; expert++ {
		gateUp, allocErr := hipAllocateByteBuffer(driver, "test", "expert gate/up", gateUpBytes, int(gateUpBytes))
		core.RequireNoError(t, allocErr)
		down, allocErr := hipAllocateByteBuffer(driver, "test", "expert down", downBytes, int(downBytes))
		core.RequireNoError(t, allocErr)
		expertCache.entries[hipGemma4ExpertCacheKey{Layer: 0, Expert: expert}] = &hipGemma4ExpertCacheEntry{
			GateUp: gateUp, Down: down,
			GateUpRows: 2 * expertFF, GateUpCols: hidden,
			DownRows: hidden, DownCols: expertFF,
			GateUpFormat: hipGGUFExpertFormatQ4_0, DownFormat: hipGGUFExpertFormatQ4_0,
			bytes: gateUpBytes + downBytes,
		}
	}
	norm := hipRMSNormDeviceWeightConfig{
		WeightPointer: 0x4000, WeightBytes: hidden * 4, Count: hidden,
		Epsilon: 1e-6, WeightEncoding: hipRMSNormWeightEncodingF32,
	}
	projection := func(outputRows, cols int, pointer nativeDevicePointer) hipMLXQ4DeviceWeightConfig {
		groups := outputRows * (cols / 32)
		return hipMLXQ4DeviceWeightConfig{
			WeightPointer: pointer, ScalePointer: pointer + 0x100, BiasPointer: pointer + 0x200,
			WeightBytes: uint64(outputRows * cols / 2), ScaleBytes: uint64(groups * 2), BiasBytes: uint64(groups * 2),
			Rows: outputRows, Cols: cols, GroupSize: 32, Bits: 4,
		}
	}
	layer := hipGemma4Q4Layer0Config{
		HiddenSize:     hidden,
		GateProjection: projection(localFF, hidden, 0x10000),
		UpProjection:   projection(localFF, hidden, 0x20000),
		DownProjection: projection(hidden, localFF, 0x30000),
		MoE: &hipGemma4MoELayerConfig{
			Layer: 0, NumExperts: experts, TopKExperts: topK, ExpertIntermediateSize: expertFF,
			PreFeedForwardNorm2: norm, PostFeedForwardNorm1: norm, PostFeedForwardNorm2: norm, RouterNorm: norm,
			RouterProjection: hipGemma4MoERouterProjectionConfig{WeightPointer: 0x50000, WeightBytes: experts * hidden * 4, Rows: experts, Cols: hidden},
			PerExpertScale:   []float32{1, 1}, ExpertCache: expertCache,
			GateUpInfo: nativeTensorInfo{Type: hipGGUFQ4_0TensorType, TypeName: "Q4_0", Dimensions: []uint64{hidden, 2 * expertFF, experts}, ByteSize: uint64(experts) * gateUpBytes},
			DownInfo:   nativeTensorInfo{Type: hipGGUFQ4_0TensorType, TypeName: "Q4_0", Dimensions: []uint64{expertFF, hidden, experts}, ByteSize: uint64(experts) * downBytes},
		},
	}
	cleanup := func() {
		_ = attentionResidual.Close()
		_ = localInput.Close()
		_ = expertCache.Close()
	}
	return attentionResidual, localInput, layer, cleanup
}

func hipGemma4MoEMLXWorkspaceFixture(t *testing.T, driver nativeHIPDriver, rows int) (*hipDeviceByteBuffer, *hipDeviceByteBuffer, hipGemma4Q4Layer0Config, func()) {
	t.Helper()
	const (
		hidden    = 32
		localFF   = 32
		experts   = 2
		topK      = 2
		expertFF  = 32
		groupSize = 32
		bits      = 4
	)
	attentionResidual, err := hipAllocateByteBuffer(driver, "test", "MLX attention residual", uint64(rows*hidden*4), rows*hidden)
	core.RequireNoError(t, err)
	localInput, err := hipAllocateByteBuffer(driver, "test", "MLX local input", uint64(rows*hidden*4), rows*hidden)
	core.RequireNoError(t, err)
	expertCache := newHIPGemma4ExpertCache(driver, 1<<20)
	projection := func(outputRows, cols int, pointer nativeDevicePointer) hipMLXQ4DeviceWeightConfig {
		groups := outputRows * (cols / groupSize)
		return hipMLXQ4DeviceWeightConfig{
			WeightPointer: pointer, ScalePointer: pointer + 0x100, BiasPointer: pointer + 0x200,
			WeightBytes: uint64(outputRows * cols * bits / 8), ScaleBytes: uint64(groups * 2), BiasBytes: uint64(groups * 2),
			Rows: outputRows, Cols: cols, GroupSize: groupSize, Bits: bits,
		}
	}
	for expert := 0; expert < experts; expert++ {
		base := nativeDevicePointer(0x60000 + expert*0x10000)
		gate := projection(expertFF, hidden, base)
		gate.ScalePointer = base + 0x2000
		gate.BiasPointer = base + 0x3000
		up := gate
		up.WeightPointer += nativeDevicePointer(gate.WeightBytes)
		up.ScalePointer += nativeDevicePointer(gate.ScaleBytes)
		up.BiasPointer += nativeDevicePointer(gate.BiasBytes)
		down := projection(hidden, expertFF, base+0x4000)
		down.ScalePointer = base + 0x6000
		down.BiasPointer = base + 0x7000
		expertCache.entries[hipGemma4ExpertCacheKey{Layer: 0, Expert: expert}] = &hipGemma4ExpertCacheEntry{
			Storage: hipGemma4MoEExpertStorageMLXAffine,
			MLXGate: gate,
			MLXUp:   up,
			MLXDown: down,
		}
	}
	norm := hipRMSNormDeviceWeightConfig{
		WeightPointer: 0x4000, WeightBytes: hidden * 4, Count: hidden,
		Epsilon: 1e-6, WeightEncoding: hipRMSNormWeightEncodingF32,
	}
	tensor := func(shape []uint64, elementBytes uint64) nativeTensorInfo {
		count := uint64(1)
		for _, dimension := range shape {
			count *= dimension
		}
		return nativeTensorInfo{TypeName: map[uint64]string{2: "BF16", 4: "U32"}[elementBytes], Dimensions: shape, SourcePath: "cached", ByteSize: count * elementBytes}
	}
	packedHidden := uint64(hidden * bits / 32)
	packedExpert := uint64(expertFF * bits / 32)
	gateGroups := uint64(hidden / groupSize)
	downGroups := uint64(expertFF / groupSize)
	expertSource := hipGemma4MLXAffineExpertSource{
		GateUp: hipGemma4MLXAffineTensorSet{
			Weight: tensor([]uint64{experts, 2 * expertFF, packedHidden}, 4),
			Scales: tensor([]uint64{experts, 2 * expertFF, gateGroups}, 2),
			Biases: tensor([]uint64{experts, 2 * expertFF, gateGroups}, 2),
		},
		Down: hipGemma4MLXAffineTensorSet{
			Weight: tensor([]uint64{experts, hidden, packedExpert}, 4),
			Scales: tensor([]uint64{experts, hidden, downGroups}, 2),
			Biases: tensor([]uint64{experts, hidden, downGroups}, 2),
		},
	}
	layer := hipGemma4Q4Layer0Config{
		HiddenSize:     hidden,
		GateProjection: projection(localFF, hidden, 0x10000),
		UpProjection:   projection(localFF, hidden, 0x20000),
		DownProjection: projection(hidden, localFF, 0x30000),
		MoE: &hipGemma4MoELayerConfig{
			Layer: 0, NumExperts: experts, TopKExperts: topK, ExpertIntermediateSize: expertFF,
			PreFeedForwardNorm2: norm, PostFeedForwardNorm1: norm, PostFeedForwardNorm2: norm, RouterNorm: norm,
			RouterProjectionMLX: projection(experts, hidden, 0x50000),
			PerExpertScale:      []float32{1, 1}, ExpertCache: expertCache,
			ExpertStorage: hipGemma4MoEExpertStorageMLXAffine, MLXExperts: expertSource,
			MLXHiddenSize: hidden, MLXGroupSize: groupSize, MLXPreferredBits: bits,
		},
	}
	cleanup := func() {
		_ = attentionResidual.Close()
		_ = localInput.Close()
		_ = expertCache.Close()
	}
	return attentionResidual, localInput, layer, cleanup
}

func (driver *hipGemma4MoEWorkspaceTestDriver) LaunchKernel(config hipKernelLaunchConfig) error {
	if config.Name == hipKernelNameMoERouter {
		return driver.fakeHIPDriver.LaunchKernel(config)
	}
	copied := config
	copied.Args = append([]byte(nil), config.Args...)
	driver.launches = append(driver.launches, copied)
	return nil
}

func (driver *hipGemma4MoEWorkspaceTestDriver) DeviceSynchronize() error {
	driver.synchronizes++
	return driver.synchronizeErr
}

func TestHIPGemma4ExpertCache_Good_MixedKQuant(t *testing.T) {
	const (
		experts  = 2
		hidden   = 256
		expertFF = 32
		prefix   = 32
	)
	gateUpSliceBytes := 2 * expertFF * (hidden / hipGGUFQ4KBlockSize) * hipGGUFQ4KBlockBytes
	expandedGateUpSliceBytes := 2 * expertFF * (hidden / hipGGUFQ4KBlockSize) * hipGGUFQ4KExpandedBlockBytes
	downSliceBytes := hidden * (expertFF / hipGGUFQ5_1BlockSize) * hipGGUFQ5_1BlockBytes
	gateUpPayload := make([]byte, experts*gateUpSliceBytes)
	downPayload := make([]byte, experts*downSliceBytes)
	filePayload := make([]byte, prefix+len(gateUpPayload)+len(downPayload))
	copy(filePayload[prefix:], gateUpPayload)
	copy(filePayload[prefix+len(gateUpPayload):], downPayload)
	path := core.PathJoin(t.TempDir(), "mixed-experts.gguf")
	core.RequireTrue(t, core.WriteFile(path, filePayload, 0o644).OK)
	driver := &fakeHIPDriver{available: true}
	cache := newHIPGemma4ExpertCache(driver, uint64(expandedGateUpSliceBytes+downSliceBytes))
	entry, err := cache.entry(hipGemma4ExpertCacheKey{Layer: 0, Expert: 1}, nativeTensorInfo{
		Type: hipGGUFQ4KTensorType, TypeName: "Q4_K", Dimensions: []uint64{hidden, 2 * expertFF, experts},
		SourcePath: path, DataOffset: prefix, ByteSize: uint64(len(gateUpPayload)),
	}, nativeTensorInfo{
		Type: hipGGUFQ5_1TensorType, TypeName: "Q5_1", Dimensions: []uint64{expertFF, hidden, experts},
		SourcePath: path, DataOffset: prefix, Offset: uint64(len(gateUpPayload)), ByteSize: uint64(len(downPayload)),
	}, experts)
	core.RequireNoError(t, err)
	defer cache.Close()
	core.AssertEqual(t, uint32(hipGGUFExpertFormatQ4KExpanded), entry.GateUpFormat)
	core.AssertEqual(t, uint32(hipGGUFExpertFormatQ5_1), entry.DownFormat)
	core.AssertEqual(t, uint64(expandedGateUpSliceBytes), entry.GateUp.SizeBytes())
	core.AssertEqual(t, uint64(downSliceBytes), entry.Down.SizeBytes())
	core.AssertEqual(t, 1, countLaunchName(driver.launches, hipKernelNameGGUFQ4KExpandMetadata))
	core.AssertEqual(t, uint64(gateUpSliceBytes+downSliceBytes), cache.stats.H2DBytes)
}

func TestHIPGemma4ExpertCache_Good_Q4KExpansionFallback(t *testing.T) {
	driver := &fakeHIPDriver{
		available: true,
		launchErr: core.E("rocm.hip.Test", "expanded kernel unavailable", nil),
	}
	cache := newHIPGemma4ExpertCache(driver, 1<<20)
	defer cache.Close()
	payload := make([]byte, hipGGUFQ4KBlockBytes)

	buffer, format, err := cache.uploadGateUpBuffer(payload, hipGGUFExpertFormatQ4K)
	core.RequireNoError(t, err)
	defer buffer.Close()

	core.AssertEqual(t, uint32(hipGGUFExpertFormatQ4K), format)
	core.AssertEqual(t, uint64(hipGGUFQ4KBlockBytes), buffer.SizeBytes())
	core.AssertEqual(t, true, cache.expandedGPUDisabled)
	core.AssertEqual(t, 1, countLaunchName(driver.launches, hipKernelNameGGUFQ4KExpandMetadata))
	core.AssertEqual(t, uint64(2*hipGGUFQ4KBlockBytes), cache.stats.H2DBytes)
}

func TestHIPGemma4ExpertCache_Good_MixedKQuantQ8_0Down(t *testing.T) {
	t.Setenv(hipGemma4Q4KExpandedEnv, "0")
	const (
		experts  = 2
		hidden   = 256
		expertFF = 32
		prefix   = 32
	)
	gateUpSliceBytes := 2 * expertFF * (hidden / hipGGUFQ4KBlockSize) * hipGGUFQ4KBlockBytes
	downSliceBytes := hidden * (expertFF / hipGGUFQ8_0BlockSize) * hipGGUFQ8_0BlockBytes
	gateUpPayload := make([]byte, experts*gateUpSliceBytes)
	downPayload := make([]byte, experts*downSliceBytes)
	filePayload := make([]byte, prefix+len(gateUpPayload)+len(downPayload))
	copy(filePayload[prefix:], gateUpPayload)
	copy(filePayload[prefix+len(gateUpPayload):], downPayload)
	path := core.PathJoin(t.TempDir(), "mixed-q8-down-experts.gguf")
	core.RequireTrue(t, core.WriteFile(path, filePayload, 0o644).OK)
	driver := &fakeHIPDriver{available: true}
	cache := newHIPGemma4ExpertCache(driver, uint64(gateUpSliceBytes+downSliceBytes))
	entry, err := cache.entry(hipGemma4ExpertCacheKey{Layer: 0, Expert: 1}, nativeTensorInfo{
		Type: hipGGUFQ4KTensorType, TypeName: "Q4_K", Dimensions: []uint64{hidden, 2 * expertFF, experts},
		SourcePath: path, DataOffset: prefix, ByteSize: uint64(len(gateUpPayload)),
	}, nativeTensorInfo{
		Type: hipGGUFQ8_0TensorType, TypeName: "Q8_0", Dimensions: []uint64{expertFF, hidden, experts},
		SourcePath: path, DataOffset: prefix, Offset: uint64(len(gateUpPayload)), ByteSize: uint64(len(downPayload)),
	}, experts)
	core.RequireNoError(t, err)
	defer cache.Close()
	core.AssertEqual(t, uint32(hipGGUFExpertFormatQ4K), entry.GateUpFormat)
	core.AssertEqual(t, uint32(hipGGUFExpertFormatQ8_0), entry.DownFormat)
	core.AssertEqual(t, uint64(gateUpSliceBytes), entry.GateUp.SizeBytes())
	core.AssertEqual(t, uint64(downSliceBytes), entry.Down.SizeBytes())
}

func TestHIPMoERouterLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipMoERouterRequest{Logits: []float32{0.1, 2, 1, -1}, TopK: 2, Layer: 7}
	want, err := rocmReferenceRouteExperts(req.Logits, req.TopK, req.Layer, nil)
	core.RequireNoError(t, err)

	got, err := hipRunMoERouterKernel(context.Background(), driver, req)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameMoERouter, driver.launches[0].Name)
	core.AssertEqual(t, uint32(256), driver.launches[0].BlockX)
	core.AssertEqual(t, hipMoERouterLaunchArgsBytes, len(driver.launches[0].Args))
	core.AssertEqual(t, req.Layer, got.Layer)
	core.AssertEqual(t, hipMoERouterLaunchStatusOK, got.Status)
	core.AssertEqual(t, len(want), len(got.Routes))
	for index := range want {
		core.AssertEqual(t, want[index].ID, got.Routes[index].ID)
		assertFloat32Near(t, want[index].Score, got.Routes[index].Score)
		assertFloat32Near(t, want[index].Prob, got.Routes[index].Prob)
	}
}

func TestHIPMoERouterLaunch_Good_DeviceInput(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	payload, err := hipFloat32Payload([]float32{0.1, 2, 1, -1})
	core.RequireNoError(t, err)
	logits, err := hipUploadByteBuffer(driver, "test", "router logits", payload, 4)
	core.RequireNoError(t, err)
	defer logits.Close()
	copyCount := len(driver.copies)

	got, err := hipRunMoERouterKernelWithDeviceInput(context.Background(), driver, logits, 2, 7)
	core.RequireNoError(t, err)

	core.AssertEqual(t, []uint64{20}, driver.copies[copyCount:])
	core.AssertEqual(t, 2, len(got.Routes))
	core.AssertEqual(t, 1, got.Routes[0].ID)
	core.AssertEqual(t, 2, got.Routes[1].ID)
	assertFloat32Near(t, 1, got.Routes[0].Prob+got.Routes[1].Prob)
}

func TestHIPMoERouterLaunch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	_, err := hipRunMoERouterKernel(context.Background(), driver, hipMoERouterRequest{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "router logits")

	_, err = hipRunMoERouterKernel(context.Background(), driver, hipMoERouterRequest{Logits: []float32{1}, TopK: 2})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "top-k")

	_, err = hipRunMoERouterKernel(context.Background(), driver, hipMoERouterRequest{Logits: []float32{1, float32(math.NaN())}, TopK: 1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = (hipMoERouterLaunchArgs{
		LogitPointer:  1,
		IDPointer:     2,
		ProbPointer:   3,
		StatusPointer: 4,
		ExpertCount:   4,
		TopK:          2,
		Layer:         0,
		LogitBytes:    12,
		IDBytes:       8,
		ProbBytes:     8,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "logit byte count")

	_, err = (hipMoERouterLaunchArgs{
		LogitPointer:  1,
		IDPointer:     2,
		ProbPointer:   3,
		StatusPointer: 4,
		ExpertCount:   2,
		TopK:          3,
		LogitBytes:    8,
		IDBytes:       12,
		ProbBytes:     12,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "top-k")

	_, err = (hipMoERouterLaunchArgs{
		LogitPointer: 1,
		IDPointer:    2,
		ProbPointer:  3,
		ExpertCount:  2,
		TopK:         1,
		LogitBytes:   8,
		IDBytes:      4,
		ProbBytes:    4,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "router status pointer")

	_, err = (hipMoERouterLaunchArgs{
		LogitPointer:  1,
		IDPointer:     2,
		ProbPointer:   3,
		StatusPointer: 4,
		ExpertCount:   2,
		TopK:          1,
		LogitBytes:    8,
		IDBytes:       4,
		ProbBytes:     4,
	}).BinaryInto(make([]byte, hipMoERouterLaunchArgsBytes-1))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "launch arg payload buffer is too small")
}

func TestHIPMoERouterLaunchBufferValidation_Bad(t *testing.T) {
	req := hipMoERouterRequest{Logits: []float32{0.1, 2, 1, -1}, TopK: 2, Layer: 7}
	_, err := req.launchArgs(nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "router device buffers are required")

	driver := &fakeHIPDriver{available: true}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.IDs.count++
	_, err = req.launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "router device buffer shape mismatch")

	buffers.IDs.count--
	buffers.Status.sizeBytes++
	_, err = req.launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "router device buffer shape mismatch")
}

func TestHIPMoERouterReadOutputValidation_Bad(t *testing.T) {
	_, err := (*hipMoERouterDeviceBuffers)(nil).ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "router output buffers are required")

	driver := &fakeHIPDriver{available: true}
	req := hipMoERouterRequest{Logits: []float32{0.1, 2, 1, -1}, TopK: 2, Layer: 7}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()

	_, err = buffers.ReadOutput()

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "router status marker mismatch")

	for _, tt := range []struct {
		name  string
		ids   []int32
		probs []float32
		want  string
	}{
		{
			name:  "expert id",
			ids:   []int32{1, 9},
			probs: []float32{0.5, 0.25},
			want:  "outside expert count",
		},
		{
			name:  "probability",
			ids:   []int32{1, 2},
			probs: []float32{0.5, float32(math.NaN())},
			want:  "router probability",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			driver := &fakeHIPDriver{available: true}
			req := hipMoERouterRequest{Logits: []float32{0.1, 2, 1, -1}, TopK: 2, Layer: 7}
			buffers, err := req.deviceBuffers(driver)
			core.RequireNoError(t, err)
			defer buffers.Close()
			idPayload := make([]byte, buffers.IDs.SizeBytes())
			for index, id := range tt.ids {
				binary.LittleEndian.PutUint32(idPayload[index*4:], uint32(id))
			}
			probPayload := make([]byte, buffers.Probs.SizeBytes())
			for index, prob := range tt.probs {
				binary.LittleEndian.PutUint32(probPayload[index*4:], math.Float32bits(prob))
			}
			statusPayload := make([]byte, buffers.Status.SizeBytes())
			binary.LittleEndian.PutUint32(statusPayload, hipMoERouterLaunchStatusOK)
			core.RequireNoError(t, driver.CopyHostToDevice(buffers.IDs.Pointer(), idPayload))
			core.RequireNoError(t, driver.CopyHostToDevice(buffers.Probs.Pointer(), probPayload))
			core.RequireNoError(t, driver.CopyHostToDevice(buffers.Status.Pointer(), statusPayload))

			_, err = buffers.ReadOutput()

			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}

	t.Run("packed output copy", func(t *testing.T) {
		driver := &fakeHIPDriver{available: true}
		req := hipMoERouterRequest{Logits: []float32{0.1, 2, 1, -1}, TopK: 2, Layer: 7}
		buffers, err := req.deviceBuffers(driver)
		core.RequireNoError(t, err)
		defer buffers.Close()
		driver.copyErr = core.NewError("copy failed")
		driver.copyErrAt = len(driver.copies) + 1

		_, err = buffers.ReadOutput()

		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "copy packed router output")
	})
}

func TestHIPMoELazyExpertLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipMoELazyExpertRequest{ExpertIDs: []int32{3, 1}, TotalExperts: 5}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.RequireNoError(t, err)
	payload, err := launch.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipMoELazyLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipMoELazyLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipMoELazyLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(buffers.IDs.Pointer()), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(buffers.Resident.Pointer()), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[24:]))
	core.AssertEqual(t, uint32(5), binary.LittleEndian.Uint32(payload[28:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[32:]))
	core.AssertEqual(t, uint32(5), binary.LittleEndian.Uint32(payload[36:]))
}

func TestHIPMoELazyExpertLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipMoELazyExpertRequest{ExpertIDs: []int32{3, 1}, TotalExperts: 5}
	want, err := rocmReferenceLazyExpertResidency([]rocmExpertRoute{{ID: 3}, {ID: 1}}, req.TotalExperts)
	core.RequireNoError(t, err)

	got, err := hipRunMoELazyExpertKernel(context.Background(), driver, req)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameMoELazy, driver.launches[0].Name)
	core.AssertEqual(t, hipMoELazyLaunchArgsBytes, len(driver.launches[0].Args))
	core.AssertEqual(t, want, got.Resident)
}

func TestHIPMoELazyExpertLaunch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	_, err := hipRunMoELazyExpertKernel(context.Background(), driver, hipMoELazyExpertRequest{ExpertIDs: []int32{1}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "expert count")

	_, err = hipRunMoELazyExpertKernel(context.Background(), driver, hipMoELazyExpertRequest{ExpertIDs: []int32{5}, TotalExperts: 5})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside expert count")

	_, err = (hipMoELazyExpertLaunchArgs{
		IDPointer:       1,
		ResidentPointer: 2,
		SelectedCount:   2,
		TotalExperts:    5,
		IDBytes:         4,
		ResidentBytes:   5,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "expert ID byte count")

	_, err = (hipMoELazyExpertLaunchArgs{
		IDPointer:       1,
		ResidentPointer: 2,
		SelectedCount:   2,
		TotalExperts:    5,
		IDBytes:         8,
		ResidentBytes:   5,
	}).BinaryInto(make([]byte, hipMoELazyLaunchArgsBytes-1))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "launch arg payload buffer is too small")
}

func TestHIPMoELazyExpertLaunchBufferValidation_Bad(t *testing.T) {
	req := hipMoELazyExpertRequest{ExpertIDs: []int32{3, 1}, TotalExperts: 5}
	_, err := req.launchArgs(nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "lazy expert device buffers are required")

	driver := &fakeHIPDriver{available: true}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.Resident.count++
	_, err = req.launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "lazy expert device buffer shape mismatch")
}

func TestHIPMoELazyExpertReadOutputValidation_Bad(t *testing.T) {
	_, err := (*hipMoELazyExpertDeviceBuffers)(nil).ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "resident expert output buffer is required")

	driver := &fakeHIPDriver{available: true}
	req := hipMoELazyExpertRequest{ExpertIDs: []int32{3, 1}, TotalExperts: 5}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.Resident.sizeBytes++
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "resident expert output byte count mismatch")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	core.RequireNoError(t, driver.CopyHostToDevice(buffers.Resident.Pointer(), []byte{0, 1, 2, 0, 1}))
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "binary flags")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	driver.copyErr = core.NewError("copy failed")

	_, err = buffers.ReadOutput()

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy resident expert output")
}

func BenchmarkHIPMoERouterLaunch_Top2Of128(b *testing.B) {
	logits := make([]float32, 128)
	for i := range logits {
		logits[i] = float32(math.Sin(float64(i)*0.11) + math.Cos(float64(i)*0.03))
	}
	req := hipMoERouterRequest{Logits: logits, TopK: 2, Layer: 7}
	driver := &fakeHIPDriver{available: true}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got, err := hipRunMoERouterKernel(context.Background(), driver, req)
		if err != nil {
			b.Fatalf("run MoE router fixture: %v", err)
		}
		if len(got.Routes) != req.TopK || got.Status != hipMoERouterLaunchStatusOK {
			b.Fatalf("router result = %+v, want top-k status OK", got)
		}
	}
}

func BenchmarkHIPMoERouterLaunchPrepared_Top2Of128(b *testing.B) {
	logits := make([]float32, 128)
	for i := range logits {
		logits[i] = float32(math.Sin(float64(i)*0.11) + math.Cos(float64(i)*0.03))
	}
	req := hipMoERouterRequest{Logits: logits, TopK: 2, Layer: 7}
	driver := &fakeHIPDriver{available: true, skipLaunchRecording: true, copies: make([]uint64, 0, 8)}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		b.Fatalf("prepare MoE router buffers: %v", err)
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		b.Fatalf("prepare MoE router launch args: %v", err)
	}
	launchBytes, err := launch.BinaryInto(make([]byte, hipMoERouterLaunchArgsBytes))
	if err != nil {
		b.Fatalf("encode MoE router launch args: %v", err)
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameMoERouter, launchBytes, 1)
	if err != nil {
		b.Fatalf("prepare MoE router launch config: %v", err)
	}
	routes := make([]rocmExpertRoute, req.TopK)
	idPayload := make([]byte, req.TopK*4)
	probPayload := make([]byte, req.TopK*4)
	statusPayload := make([]byte, 4)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipLaunchKernel(driver, config); err != nil {
			b.Fatalf("launch MoE router fixture: %v", err)
		}
		got, err := buffers.ReadOutputInto(routes, idPayload, probPayload, statusPayload)
		if err != nil {
			b.Fatalf("read MoE router fixture: %v", err)
		}
		if len(got.Routes) != req.TopK || got.Status != hipMoERouterLaunchStatusOK {
			b.Fatalf("router result = %+v, want top-k status OK", got)
		}
		driver.copies = driver.copies[:0]
	}
}

func BenchmarkHIPMoELazyExpertLaunch_Top2Of128(b *testing.B) {
	req := hipMoELazyExpertRequest{ExpertIDs: []int32{37, 5}, TotalExperts: 128}
	driver := &fakeHIPDriver{available: true}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got, err := hipRunMoELazyExpertKernel(context.Background(), driver, req)
		if err != nil {
			b.Fatalf("run MoE lazy expert fixture: %v", err)
		}
		if len(got.Resident) != req.TotalExperts || !got.Resident[37] || !got.Resident[5] {
			b.Fatalf("resident result = %+v, want selected experts resident", got.Resident)
		}
	}
}

func BenchmarkHIPMoELazyExpertLaunchPrepared_Top2Of128(b *testing.B) {
	req := hipMoELazyExpertRequest{ExpertIDs: []int32{37, 5}, TotalExperts: 128}
	driver := &fakeHIPDriver{available: true, skipLaunchRecording: true, copies: make([]uint64, 0, 8)}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		b.Fatalf("prepare MoE lazy expert buffers: %v", err)
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		b.Fatalf("prepare MoE lazy expert launch args: %v", err)
	}
	launchBytes, err := launch.BinaryInto(make([]byte, hipMoELazyLaunchArgsBytes))
	if err != nil {
		b.Fatalf("encode MoE lazy expert launch args: %v", err)
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameMoELazy, launchBytes, req.TotalExperts)
	if err != nil {
		b.Fatalf("prepare MoE lazy expert launch config: %v", err)
	}
	resident := make([]bool, req.TotalExperts)
	payload := make([]byte, req.TotalExperts)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipLaunchKernel(driver, config); err != nil {
			b.Fatalf("launch MoE lazy expert fixture: %v", err)
		}
		got, err := buffers.ReadOutputInto(resident, payload)
		if err != nil {
			b.Fatalf("read MoE lazy expert fixture: %v", err)
		}
		if len(got.Resident) != req.TotalExperts || !got.Resident[37] || !got.Resident[5] {
			b.Fatalf("resident result = %+v, want selected experts resident", got.Resident)
		}
		driver.copies = driver.copies[:0]
	}
}
