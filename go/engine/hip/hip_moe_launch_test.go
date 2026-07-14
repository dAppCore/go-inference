// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
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
	core.AssertEqual(t, uint64(8*memoryGiB), hipGemma4ExpertCacheBudget(&fakeHIPDriver{available: true, device: nativeDeviceInfo{FreeBytes: 12 * memoryGiB}}))
	core.AssertEqual(t, uint64(8*memoryGiB), hipGemma4ExpertCacheBudget(&fakeHIPDriver{available: true, device: nativeDeviceInfo{FreeBytes: 11 * memoryGiB}}))
	core.AssertEqual(t, uint64(5*memoryGiB), hipGemma4ExpertCacheBudget(&fakeHIPDriver{available: true, device: nativeDeviceInfo{FreeBytes: 7 * memoryGiB}}))
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
	const (
		hidden   = 32
		localFF  = 32
		experts  = 2
		topK     = 2
		expertFF = 32
	)
	driver := &hipGemma4MoEWorkspaceTestDriver{fakeHIPDriver: &fakeHIPDriver{available: true}}
	workspace := hipNewAttentionHeadsChunkedWorkspace()
	defer workspace.Close()
	attentionResidual, err := hipAllocateByteBuffer(driver, "test", "attention residual", hidden*4, hidden)
	core.RequireNoError(t, err)
	defer attentionResidual.Close()
	localInput, err := hipAllocateByteBuffer(driver, "test", "local input", hidden*4, hidden)
	core.RequireNoError(t, err)
	defer localInput.Close()

	expertCache := newHIPGemma4ExpertCache(driver, 1<<20)
	defer expertCache.Close()
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
	projection := func(rows, cols int, pointer nativeDevicePointer) hipMLXQ4DeviceWeightConfig {
		groups := rows * (cols / 32)
		return hipMLXQ4DeviceWeightConfig{
			WeightPointer: pointer, ScalePointer: pointer + 0x100, BiasPointer: pointer + 0x200,
			WeightBytes: uint64(rows * cols / 2), ScaleBytes: uint64(groups * 2), BiasBytes: uint64(groups * 2),
			Rows: rows, Cols: cols, GroupSize: 32, Bits: 4,
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

type hipGemma4MoEWorkspaceTestDriver struct {
	*fakeHIPDriver
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
