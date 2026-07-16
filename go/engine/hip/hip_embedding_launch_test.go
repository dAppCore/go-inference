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

func TestHIPDiffusionExpectedEmbeddingLaunch_Good_SelectsAffineGroup64Rows16(t *testing.T) {
	const rows, vocab, hidden, groupSize = 32, 3, 64, 64
	driver := &fakeHIPDriver{available: true}
	_, err := hipRunDiffusionExpectedEmbeddingKernel(context.Background(), driver, make([]float32, rows*vocab), rows, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: 0x9000,
		EmbeddingBytes:   vocab * hidden / 2,
		ScalePointer:     0xa000,
		ScaleBytes:       vocab * (hidden / groupSize) * 2,
		BiasPointer:      0xb000,
		BiasBytes:        vocab * (hidden / groupSize) * 2,
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		GroupSize:        groupSize,
		QuantBits:        4,
		VocabSize:        vocab,
		HiddenSize:       hidden,
	}, 1)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameDiffusionExpectedEmbeddingAffineG64Rows16, driver.launches[0].Name)
	core.AssertEqual(t, uint32(1), driver.launches[0].GridX)
	core.AssertEqual(t, uint32(2), driver.launches[0].GridY)
	core.AssertEqual(t, uint32(256), driver.launches[0].BlockX)
}

func TestHIPDiffusionExpectedEmbeddingLaunch_Good_SelectsQ8Group64Dims4Rows4(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DIFFUSION_EXPECTED_EMBEDDING_TILE", "1")
	const rows, vocab, hidden, groupSize = 32, 3, 64, 64
	driver := &fakeHIPDriver{available: true}
	_, err := hipRunDiffusionExpectedEmbeddingKernel(context.Background(), driver, make([]float32, rows*vocab), rows, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: 0x9000,
		EmbeddingBytes:   vocab * hidden,
		ScalePointer:     0xa000,
		ScaleBytes:       vocab * (hidden / groupSize) * 2,
		BiasPointer:      0xb000,
		BiasBytes:        vocab * (hidden / groupSize) * 2,
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		GroupSize:        groupSize,
		QuantBits:        8,
		VocabSize:        vocab,
		HiddenSize:       hidden,
	}, 1)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameDiffusionExpectedEmbeddingQ8G64Dims4Rows4, driver.launches[0].Name)
	core.AssertEqual(t, uint32(1), driver.launches[0].GridX)
	core.AssertEqual(t, uint32(8), driver.launches[0].GridY)
	core.AssertEqual(t, uint32(256), driver.launches[0].BlockX)
}

func TestHIPDiffusionExpectedEmbeddingLaunch_Good_SelectsQ8Group64SubgroupRows64Probability4(t *testing.T) {
	const rows, vocab, hidden, groupSize = 256, 3, 2816, 64
	driver := &fakeHIPDriver{available: true}
	_, err := hipRunDiffusionExpectedEmbeddingKernel(context.Background(), driver, make([]float32, rows*vocab), rows, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: 0x9000,
		EmbeddingBytes:   vocab * hidden,
		ScalePointer:     0xa000,
		ScaleBytes:       vocab * (hidden / groupSize) * 2,
		BiasPointer:      0xb000,
		BiasBytes:        vocab * (hidden / groupSize) * 2,
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		GroupSize:        groupSize,
		QuantBits:        8,
		VocabSize:        vocab,
		HiddenSize:       hidden,
	}, 1)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, "rocm_diffusion_expected_embedding_q8_g64_subgroup32_rows64_prob4", driver.launches[0].Name)
	core.AssertEqual(t, uint32(22), driver.launches[0].GridX)
	core.AssertEqual(t, uint32(4), driver.launches[0].GridY)
	core.AssertEqual(t, uint32(256), driver.launches[0].BlockX)

	t.Setenv(hipDisableDiffusionExpectedEmbeddingProbability4Env, "1")
	driver.launches = nil
	_, err = hipRunDiffusionExpectedEmbeddingKernel(context.Background(), driver, make([]float32, rows*vocab), rows, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: 0x9000,
		EmbeddingBytes:   vocab * hidden,
		ScalePointer:     0xa000,
		ScaleBytes:       vocab * (hidden / groupSize) * 2,
		BiasPointer:      0xb000,
		BiasBytes:        vocab * (hidden / groupSize) * 2,
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		GroupSize:        groupSize,
		QuantBits:        8,
		VocabSize:        vocab,
		HiddenSize:       hidden,
	}, 1)
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipKernelNameDiffusionExpectedEmbeddingQ8G64SubgroupRows64, driver.launches[0].Name)

	t.Setenv(hipDisableDiffusionExpectedEmbeddingSubgroupEnv, "1")
	driver.launches = nil
	_, err = hipRunDiffusionExpectedEmbeddingKernel(context.Background(), driver, make([]float32, rows*vocab), rows, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: 0x9000,
		EmbeddingBytes:   vocab * hidden,
		ScalePointer:     0xa000,
		ScaleBytes:       vocab * (hidden / groupSize) * 2,
		BiasPointer:      0xb000,
		BiasBytes:        vocab * (hidden / groupSize) * 2,
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		GroupSize:        groupSize,
		QuantBits:        8,
		VocabSize:        vocab,
		HiddenSize:       hidden,
	}, 1)
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipKernelNameDiffusionExpectedEmbeddingQ8G64Dims4Rows4, driver.launches[0].Name)
}

func TestHIPDiffusionExpectedEmbeddingLaunch_Good_SelectsQ8Group64Tile32x64(t *testing.T) {
	const rows, vocab, hidden, groupSize = 64, 3, 64, 64
	driver := &fakeHIPDriver{available: true}
	_, err := hipRunDiffusionExpectedEmbeddingKernel(context.Background(), driver, make([]float32, rows*vocab), rows, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: 0x9000,
		EmbeddingBytes:   vocab * hidden,
		ScalePointer:     0xa000,
		ScaleBytes:       vocab * (hidden / groupSize) * 2,
		BiasPointer:      0xb000,
		BiasBytes:        vocab * (hidden / groupSize) * 2,
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		GroupSize:        groupSize,
		QuantBits:        8,
		VocabSize:        vocab,
		HiddenSize:       hidden,
	}, 1)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, "rocm_diffusion_expected_embedding_q8_g64_tile32x64", driver.launches[0].Name)
	core.AssertEqual(t, uint32(1), driver.launches[0].GridX)
	core.AssertEqual(t, uint32(2), driver.launches[0].GridY)
	core.AssertEqual(t, uint32(256), driver.launches[0].BlockX)
}

func TestHIPDiffusionExpectedEmbeddingLaunch_Good_KeepsProbabilitiesOnDevice(t *testing.T) {
	const rows, vocab, hidden = 2, 3, 4
	driver := &fakeHIPDriver{available: true}
	probabilities, err := hipUploadGemma4Q4Float32Input(driver, "resident diffusion probabilities", make([]float32, rows*vocab))
	core.RequireNoError(t, err)
	defer probabilities.Close()
	driver.copies = nil

	output, err := hipRunDiffusionExpectedEmbeddingDeviceKernel(context.Background(), driver, probabilities, rows, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: 0x9000,
		EmbeddingBytes:   vocab * hidden * 4,
		TableEncoding:    hipEmbeddingTableEncodingF32,
		VocabSize:        vocab,
		HiddenSize:       hidden,
	}, 1)
	core.RequireNoError(t, err)
	defer output.Close()

	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameDiffusionExpectedEmbedding, driver.launches[0].Name)
	core.AssertEqual(t, 0, len(driver.copies))
}

func TestHIPEmbeddingMeanPoolLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipEmbeddingMeanPoolRequest{Tokens: []float32{1, 3, 3, 5}, TokenCount: 2, Dim: 2, Normalize: true}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.RequireNoError(t, err)
	payload, err := launch.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipEmbeddingMeanPoolLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipEmbeddingMeanPoolLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipEmbeddingMeanPoolLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(buffers.Tokens.Pointer()), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[24:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[28:]))
	core.AssertEqual(t, uint32(16), binary.LittleEndian.Uint32(payload[32:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[36:]))
	core.AssertEqual(t, hipEmbeddingMeanPoolLaunchFlagNormalize, binary.LittleEndian.Uint32(payload[40:]))
}

func TestHIPEmbeddingMeanPoolLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipEmbeddingMeanPoolRequest{Tokens: []float32{1, 3, 3, 5}, TokenCount: 2, Dim: 2}
	want, err := rocmReferenceMeanPoolEmbedding(splitFloat32Vectors(req.Tokens, req.Dim), req.Normalize)
	core.RequireNoError(t, err)

	got, err := hipRunEmbeddingMeanPoolKernel(context.Background(), driver, req)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameEmbedMean, driver.launches[0].Name)
	core.AssertEqual(t, hipEmbeddingMeanPoolLaunchArgsBytes, len(driver.launches[0].Args))
	assertFloat32SlicesNear(t, want, got, 0)
}

func TestHIPEmbeddingLookupLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipEmbeddingLookupRequest{
		TokenIDs:     []int32{2, 0},
		EmbeddingF32: []float32{1, -2, 0.5, 2, -1, 3},
		VocabSize:    3,
		HiddenSize:   2,
	}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	core.RequireNoError(t, err)
	payload, err := launch.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipEmbeddingLookupLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipEmbeddingLookupLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipEmbeddingLookupLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(buffers.Tokens.Pointer()), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(buffers.Embedding.Pointer()), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(payload[24:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[32:]))
	core.AssertEqual(t, uint32(3), binary.LittleEndian.Uint32(payload[36:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[40:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[44:]))
	core.AssertEqual(t, uint64(24), binary.LittleEndian.Uint64(payload[48:]))
	core.AssertEqual(t, uint64(16), binary.LittleEndian.Uint64(payload[56:]))
	core.AssertEqual(t, hipEmbeddingTableEncodingF32, binary.LittleEndian.Uint32(payload[64:]))
	core.AssertEqual(t, uint32(0), binary.LittleEndian.Uint32(payload[96:]))

	got, err := hipRunEmbeddingLookupKernel(context.Background(), &fakeHIPDriver{available: true}, req)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{-1, 3, 1, -2}, got, 0)

	bf16Req := hipEmbeddingLookupRequest{
		TokenIDs:      []int32{2, 0},
		EmbeddingBF16: []uint16{0x3f80, 0xc000, 0x3f00, 0x4000, 0xbf80, 0x4040},
		VocabSize:     3,
		HiddenSize:    2,
	}
	bf16Got, err := hipRunEmbeddingLookupKernel(context.Background(), &fakeHIPDriver{available: true}, bf16Req)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{-1, 3, 1, -2}, bf16Got, 0)

	deviceBF16Driver := &fakeHIPDriver{available: true}
	deviceBF16Payload, err := hipUint16Payload(bf16Req.EmbeddingBF16)
	core.RequireNoError(t, err)
	deviceBF16, err := hipUploadByteBuffer(deviceBF16Driver, "rocm.hip.EmbeddingLookupLaunch", "device bf16 embedding", deviceBF16Payload, len(bf16Req.EmbeddingBF16))
	core.RequireNoError(t, err)
	defer deviceBF16.Close()
	deviceBF16Got, err := hipRunEmbeddingLookupKernelWithDeviceTable(context.Background(), deviceBF16Driver, bf16Req.TokenIDs, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: deviceBF16.Pointer(),
		EmbeddingBytes:   deviceBF16.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingBF16,
		VocabSize:        bf16Req.VocabSize,
		HiddenSize:       bf16Req.HiddenSize,
	})
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{-1, 3, 1, -2}, deviceBF16Got, 0)
	tokenWorkspace, err := hipAllocateByteBuffer(deviceBF16Driver, "rocm.hip.EmbeddingLookupLaunch", "single token id", 4, 1)
	core.RequireNoError(t, err)
	defer tokenWorkspace.Close()
	deviceBF16Single, err := hipRunEmbeddingLookupKernelWithDeviceTableSingleTokenBuffer(context.Background(), deviceBF16Driver, 2, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: deviceBF16.Pointer(),
		EmbeddingBytes:   deviceBF16.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingBF16,
		VocabSize:        bf16Req.VocabSize,
		HiddenSize:       bf16Req.HiddenSize,
	}, tokenWorkspace)
	core.RequireNoError(t, err)
	defer deviceBF16Single.Close()
	singleValues, err := (&hipEmbeddingLookupDeviceBuffers{Output: deviceBF16Single, TokenCount: 1, HiddenSize: bf16Req.HiddenSize}).ReadOutput()
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{-1, 3}, singleValues, 0)
	deviceBF16NoWriteOutput, err := hipAllocateByteBuffer(deviceBF16Driver, "rocm.hip.EmbeddingLookupLaunch", "single token no-write output", uint64(bf16Req.HiddenSize*4), bf16Req.HiddenSize)
	core.RequireNoError(t, err)
	defer deviceBF16NoWriteOutput.Close()
	err = hipRunEmbeddingLookupKernelWithDeviceTableTokenBufferOutput(context.Background(), deviceBF16Driver, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: deviceBF16.Pointer(),
		EmbeddingBytes:   deviceBF16.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingBF16,
		VocabSize:        bf16Req.VocabSize,
		HiddenSize:       bf16Req.HiddenSize,
	}, tokenWorkspace, deviceBF16NoWriteOutput)
	core.RequireNoError(t, err)
	noWriteValues, err := (&hipEmbeddingLookupDeviceBuffers{Output: deviceBF16NoWriteOutput, TokenCount: 1, HiddenSize: bf16Req.HiddenSize}).ReadOutput()
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{-1, 3}, noWriteValues, 0)
	deviceBF16ScaledOutput, err := hipAllocateByteBuffer(deviceBF16Driver, "rocm.hip.EmbeddingLookupLaunch", "single token scaled output", uint64(bf16Req.HiddenSize*4), bf16Req.HiddenSize)
	core.RequireNoError(t, err)
	defer deviceBF16ScaledOutput.Close()
	err = hipRunEmbeddingLookupKernelWithDeviceTableTokenBufferScaledOutput(context.Background(), deviceBF16Driver, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: deviceBF16.Pointer(),
		EmbeddingBytes:   deviceBF16.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingBF16,
		VocabSize:        bf16Req.VocabSize,
		HiddenSize:       bf16Req.HiddenSize,
	}, tokenWorkspace, deviceBF16ScaledOutput, 0.5)
	core.RequireNoError(t, err)
	core.AssertEqual(t, math.Float32bits(0.5), binary.LittleEndian.Uint32(deviceBF16Driver.launches[len(deviceBF16Driver.launches)-1].Args[96:]))
	scaledValues, err := (&hipEmbeddingLookupDeviceBuffers{Output: deviceBF16ScaledOutput, TokenCount: 1, HiddenSize: bf16Req.HiddenSize}).ReadOutput()
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{-0.5, 1.5}, scaledValues, 0)
	greedyPayload := make([]byte, hipMLXQ4ProjectionBestBytes)
	binary.LittleEndian.PutUint64(greedyPayload, hipPackGreedyBest(1, 2))
	greedyToken, err := hipUploadByteBuffer(deviceBF16Driver, "rocm.hip.EmbeddingLookupLaunch", "greedy token", greedyPayload, 1)
	core.RequireNoError(t, err)
	defer greedyToken.Close()
	greedyOutput, err := hipAllocateByteBuffer(deviceBF16Driver, "rocm.hip.EmbeddingLookupLaunch", "greedy token output", uint64(bf16Req.HiddenSize*4), bf16Req.HiddenSize)
	core.RequireNoError(t, err)
	defer greedyOutput.Close()
	err = hipRunEmbeddingLookupKernelWithDeviceTableGreedyTokenOutput(context.Background(), deviceBF16Driver, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: deviceBF16.Pointer(),
		EmbeddingBytes:   deviceBF16.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingBF16,
		VocabSize:        bf16Req.VocabSize,
		HiddenSize:       bf16Req.HiddenSize,
	}, greedyToken, greedyOutput)
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipKernelNameEmbedLookupGreedyToken, deviceBF16Driver.launches[len(deviceBF16Driver.launches)-1].Name)
	greedyValues, err := (&hipEmbeddingLookupDeviceBuffers{Output: greedyOutput, TokenCount: 1, HiddenSize: bf16Req.HiddenSize}).ReadOutput()
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{-1, 3}, greedyValues, 0)
	greedyScaledOutput, err := hipAllocateByteBuffer(deviceBF16Driver, "rocm.hip.EmbeddingLookupLaunch", "greedy token scaled output", uint64(bf16Req.HiddenSize*4), bf16Req.HiddenSize)
	core.RequireNoError(t, err)
	defer greedyScaledOutput.Close()
	err = hipRunEmbeddingLookupKernelWithDeviceTableGreedyTokenScaledOutput(context.Background(), deviceBF16Driver, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: deviceBF16.Pointer(),
		EmbeddingBytes:   deviceBF16.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingBF16,
		VocabSize:        bf16Req.VocabSize,
		HiddenSize:       bf16Req.HiddenSize,
	}, greedyToken, greedyScaledOutput, 2)
	core.RequireNoError(t, err)
	core.AssertEqual(t, math.Float32bits(2), binary.LittleEndian.Uint32(deviceBF16Driver.launches[len(deviceBF16Driver.launches)-1].Args[96:]))
	greedyScaledValues, err := (&hipEmbeddingLookupDeviceBuffers{Output: greedyScaledOutput, TokenCount: 1, HiddenSize: bf16Req.HiddenSize}).ReadOutput()
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{-2, 6}, greedyScaledValues, 0)

	q4Req := hipEmbeddingLookupRequest{
		TokenIDs:    []int32{2, 0},
		EmbeddingQ4: []uint32{0x76543210, 0x11111111, 0xfedcba98},
		Q4Scales:    []uint16{0x3f80, 0x3f80, 0x3f00},
		Q4Biases:    []uint16{0x0000, 0x0000, 0xbf80},
		Q4GroupSize: 8,
		VocabSize:   3,
		HiddenSize:  8,
	}
	q4Driver := &fakeHIPDriver{available: true}
	q4Buffers, err := q4Req.deviceBuffers(q4Driver)
	core.RequireNoError(t, err)
	defer q4Buffers.Close()
	q4Launch, err := q4Req.launchArgs(q4Buffers)
	core.RequireNoError(t, err)
	q4Payload, err := q4Launch.Binary()
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipEmbeddingTableEncodingMLXQ4, binary.LittleEndian.Uint32(q4Payload[64:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(q4Payload[68:]))
	core.AssertEqual(t, uint64(q4Buffers.Scales.Pointer()), binary.LittleEndian.Uint64(q4Payload[72:]))
	core.AssertEqual(t, uint64(q4Buffers.Biases.Pointer()), binary.LittleEndian.Uint64(q4Payload[80:]))
	core.AssertEqual(t, uint32(6), binary.LittleEndian.Uint32(q4Payload[88:]))
	core.AssertEqual(t, uint32(6), binary.LittleEndian.Uint32(q4Payload[92:]))
	q4Got, err := hipRunEmbeddingLookupKernel(context.Background(), &fakeHIPDriver{available: true}, q4Req)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 0, 1, 2, 3, 4, 5, 6, 7}, q4Got, 0)

	deviceQ4Driver := &fakeHIPDriver{available: true}
	deviceQ4Payload, err := hipUint32Payload(q4Req.EmbeddingQ4)
	core.RequireNoError(t, err)
	deviceQ4, err := hipUploadByteBuffer(deviceQ4Driver, "rocm.hip.EmbeddingLookupLaunch", "device q4 embedding", deviceQ4Payload, len(q4Req.EmbeddingQ4))
	core.RequireNoError(t, err)
	defer deviceQ4.Close()
	deviceQ4ScalePayload, err := hipUint16Payload(q4Req.Q4Scales)
	core.RequireNoError(t, err)
	deviceQ4Scales, err := hipUploadByteBuffer(deviceQ4Driver, "rocm.hip.EmbeddingLookupLaunch", "device q4 scales", deviceQ4ScalePayload, len(q4Req.Q4Scales))
	core.RequireNoError(t, err)
	defer deviceQ4Scales.Close()
	deviceQ4BiasPayload, err := hipUint16Payload(q4Req.Q4Biases)
	core.RequireNoError(t, err)
	deviceQ4Biases, err := hipUploadByteBuffer(deviceQ4Driver, "rocm.hip.EmbeddingLookupLaunch", "device q4 biases", deviceQ4BiasPayload, len(q4Req.Q4Biases))
	core.RequireNoError(t, err)
	defer deviceQ4Biases.Close()
	deviceQ4Got, err := hipRunEmbeddingLookupKernelWithDeviceTable(context.Background(), deviceQ4Driver, q4Req.TokenIDs, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: deviceQ4.Pointer(),
		EmbeddingBytes:   deviceQ4.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		VocabSize:        q4Req.VocabSize,
		HiddenSize:       q4Req.HiddenSize,
		GroupSize:        q4Req.Q4GroupSize,
		ScalePointer:     deviceQ4Scales.Pointer(),
		BiasPointer:      deviceQ4Biases.Pointer(),
		ScaleBytes:       deviceQ4Scales.SizeBytes(),
		BiasBytes:        deviceQ4Biases.SizeBytes(),
	})
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 0, 1, 2, 3, 4, 5, 6, 7}, deviceQ4Got, 0)

	q6Req := hipEmbeddingLookupRequest{
		TokenIDs: []int32{1},
		EmbeddingQ4: hipPackMLXAffineValuesForTest([]uint32{
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
			16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
		}, 16, 6),
		Q4Scales:    []uint16{0x3f80, 0x3f80},
		Q4Biases:    []uint16{0, 0},
		Q4GroupSize: 16,
		QuantBits:   6,
		VocabSize:   2,
		HiddenSize:  16,
	}
	q6Got, err := hipRunEmbeddingLookupKernel(context.Background(), &fakeHIPDriver{available: true}, q6Req)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}, q6Got, 0)
}

func TestHIPRerankCosineLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipRerankCosineRequest{
		Query:         []float32{1, 0},
		Documents:     []float32{0, 1, 1, 1, 1, 0},
		DocumentCount: 3,
		Dim:           2,
	}

	got, err := hipRunRerankCosineKernel(context.Background(), driver, req)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameRerank, driver.launches[0].Name)
	core.AssertEqual(t, hipRerankCosineLaunchArgsBytes, len(driver.launches[0].Args))
	assertFloat32SlicesNear(t, []float32{0, 0.70710677, 1}, got, 0.0001)
}

func TestHIPEmbeddingAndRerankLaunch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	_, err := hipRunEmbeddingMeanPoolKernel(context.Background(), driver, hipEmbeddingMeanPoolRequest{
		Tokens:     []float32{1, 2, 3},
		TokenCount: 2,
		Dim:        2,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token embedding length")

	_, err = (hipEmbeddingMeanPoolLaunchArgs{
		TokenPointer:  1,
		OutputPointer: 2,
		TokenCount:    2,
		Dim:           2,
		TokenBytes:    8,
		OutputBytes:   8,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token byte count")

	_, err = hipRunRerankCosineKernel(context.Background(), driver, hipRerankCosineRequest{
		Query:         []float32{0, 0},
		Documents:     []float32{1, 0},
		DocumentCount: 1,
		Dim:           2,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "zero vector")

	_, err = hipRunEmbeddingLookupKernel(context.Background(), driver, hipEmbeddingLookupRequest{
		TokenIDs:     []int32{3},
		EmbeddingF32: []float32{1, 2, 3, 4},
		VocabSize:    2,
		HiddenSize:   2,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside vocabulary")

	_, err = (hipEmbeddingLookupLaunchArgs{
		TokenPointer:     1,
		EmbeddingPointer: 2,
		OutputPointer:    3,
		TokenCount:       1,
		VocabSize:        2,
		HiddenSize:       2,
		TokenBytes:       4,
		EmbeddingBytes:   6,
		OutputBytes:      8,
		TableEncoding:    hipEmbeddingTableEncodingBF16,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "bf16 embedding byte count")

	_, err = (hipEmbeddingLookupLaunchArgs{
		TokenPointer:     1,
		EmbeddingPointer: 2,
		OutputPointer:    3,
		TokenCount:       1,
		VocabSize:        2,
		HiddenSize:       8,
		TokenBytes:       4,
		EmbeddingBytes:   7,
		OutputBytes:      32,
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		GroupSize:        8,
		ScalePointer:     4,
		BiasPointer:      5,
		ScaleBytes:       4,
		BiasBytes:        4,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "MLX affine embedding byte count")

	_, err = (hipEmbeddingLookupLaunchArgs{
		TokenPointer:     1,
		EmbeddingPointer: 2,
		OutputPointer:    3,
		TokenCount:       1,
		VocabSize:        2,
		HiddenSize:       2,
		TokenBytes:       4,
		EmbeddingBytes:   16,
		OutputBytes:      8,
		TableEncoding:    hipEmbeddingTableEncodingF32,
		OutputScale:      float32(math.NaN()),
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "output scale")

	_, err = hipRunEmbeddingLookupKernelWithDeviceTable(context.Background(), driver, []int32{2}, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: 1,
		EmbeddingBytes:   16,
		TableEncoding:    hipEmbeddingTableEncodingBF16,
		VocabSize:        2,
		HiddenSize:       4,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside vocabulary")
}

func BenchmarkHIPWriteSingleTokenID_ReusedBuffer(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	buffer, err := hipAllocateByteBuffer(driver, "rocm.hip.Tokens", "single token id", 4, 1)
	core.RequireNoError(b, err)
	defer buffer.Close()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipWriteSingleTokenID(driver, buffer.Pointer(), int32(i&1023)); err != nil {
			b.Fatalf("write token id: %v", err)
		}
	}
}

func TestHIPEmbeddingMeanPoolReadOutputValidation_Bad(t *testing.T) {
	_, err := (*hipEmbeddingMeanPoolDeviceBuffers)(nil).ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "embedding output buffer is required")

	req := hipEmbeddingMeanPoolRequest{Tokens: []float32{1, 3, 3, 5}, TokenCount: 2, Dim: 2}
	driver := &fakeHIPDriver{available: true}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.Output.sizeBytes++
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "embedding output byte count mismatch")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	payload, err := hipFloat32Payload([]float32{0, float32(math.NaN())})
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(buffers.Output.Pointer(), payload))
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	driver.copyErr = core.NewError("copy failed")
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy embedding output")
}

func TestHIPRerankCosineReadOutputValidation_Bad(t *testing.T) {
	_, err := (*hipRerankCosineDeviceBuffers)(nil).ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rerank output buffer is required")

	req := hipRerankCosineRequest{
		Query:         []float32{1, 0},
		Documents:     []float32{0, 1, 1, 1, 1, 0},
		DocumentCount: 3,
		Dim:           2,
	}
	driver := &fakeHIPDriver{available: true}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.Output.sizeBytes++
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rerank output byte count mismatch")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	payload, err := hipFloat32Payload([]float32{0, float32(math.Inf(1)), 1})
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(buffers.Output.Pointer(), payload))
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	driver.copyErr = core.NewError("copy failed")
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy rerank output")
}
