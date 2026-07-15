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

func TestHIPMoECombineNormsLaunchArgs_Good(t *testing.T) {
	args := hipMoECombineNormsLaunchArgs{
		LocalInputPointer:    11,
		LocalWeightPointer:   12,
		ExpertInputPointer:   21,
		ExpertWeightPointer:  22,
		OutputPointer:        31,
		Count:                4,
		LocalInputBytes:      16,
		LocalWeightBytes:     8,
		ExpertInputBytes:     16,
		ExpertWeightBytes:    16,
		OutputBytes:          16,
		LocalEpsilon:         1e-6,
		LocalWeightEncoding:  hipRMSNormWeightEncodingBF16,
		LocalFlags:           hipRMSNormLaunchFlagAddUnitWeight,
		ExpertEpsilon:        1e-5,
		ExpertWeightEncoding: hipRMSNormWeightEncodingF32,
	}

	payload, err := args.Binary()
	core.RequireNoError(t, err)
	defer hipReleaseLaunchPacket(payload)

	core.AssertEqual(t, hipMoECombineNormsLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipMoECombineNormsLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipMoECombineNormsLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(11), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(12), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint64(21), binary.LittleEndian.Uint64(payload[24:]))
	core.AssertEqual(t, uint64(22), binary.LittleEndian.Uint64(payload[32:]))
	core.AssertEqual(t, uint64(31), binary.LittleEndian.Uint64(payload[40:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(payload[48:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[56:]))
	core.AssertEqual(t, uint32(16), binary.LittleEndian.Uint32(payload[64:]))
	core.AssertEqual(t, math.Float32bits(1e-6), binary.LittleEndian.Uint32(payload[72:]))
	core.AssertEqual(t, hipRMSNormWeightEncodingBF16, binary.LittleEndian.Uint32(payload[76:]))
	core.AssertEqual(t, hipRMSNormLaunchFlagAddUnitWeight, binary.LittleEndian.Uint32(payload[80:]))
	core.AssertEqual(t, math.Float32bits(1e-5), binary.LittleEndian.Uint32(payload[84:]))
	core.AssertEqual(t, hipRMSNormWeightEncodingF32, binary.LittleEndian.Uint32(payload[88:]))
}

func TestHIPMoECombineNormsDeviceKernelOutput_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	localValues := []float32{3, -4, 2, -1}
	expertValues := []float32{1, 2, -3, 4}
	localWeight := []uint16{
		hipFloat32ToBFloat16(0.5),
		hipFloat32ToBFloat16(-0.25),
		hipFloat32ToBFloat16(0.75),
		hipFloat32ToBFloat16(0.125),
	}
	expertWeight := []float32{1, 0.5, -0.5, 2}
	local, localCfg := hipMoECombineNormsTestInput(t, driver, "local", localValues, localWeight, nil, hipRMSNormLaunchFlagAddUnitWeight, 1e-6)
	defer local.Close()
	expert, expertCfg := hipMoECombineNormsTestInput(t, driver, "expert", expertValues, nil, expertWeight, 0, 1e-5)
	defer expert.Close()
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MoECombineNormsLaunch", "combined output", uint64(len(localValues)*4), len(localValues))
	core.RequireNoError(t, err)
	defer output.Close()

	err = hipRunMoECombineNormsDeviceKernelOutput(context.Background(), driver, local, expert, localCfg, expertCfg, output)
	core.RequireNoError(t, err)
	got, err := hipReadFloat32DeviceOutput(output, "rocm.hip.MoECombineNormsLaunch", "combined output", len(localValues))
	core.RequireNoError(t, err)

	localReferenceWeight := make([]float32, len(localWeight))
	for index, weight := range localWeight {
		localReferenceWeight[index] = hipBFloat16ToFloat32(weight) + 1
	}
	localReference, err := hipReferenceRMSNorm(localValues, localReferenceWeight, localCfg.Epsilon)
	core.RequireNoError(t, err)
	expertReference, err := hipReferenceRMSNorm(expertValues, expertWeight, expertCfg.Epsilon)
	core.RequireNoError(t, err)
	want := make([]float32, len(localReference))
	for index := range want {
		want[index] = localReference[index] + expertReference[index]
	}
	assertFloat32SlicesNear(t, want, got, 0.00001)
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameMoECombineNorms, driver.launches[0].Name)
	core.AssertEqual(t, uint32(1), driver.launches[0].GridX)
	core.AssertEqual(t, uint32(256), driver.launches[0].BlockX)
}

func TestHIPMoECombineNormsDeviceKernelOutput_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	local, localCfg := hipMoECombineNormsTestInput(t, driver, "local", []float32{1, 2, 3, 4}, nil, []float32{1, 1, 1, 1}, 0, 0)
	defer local.Close()
	expert, expertCfg := hipMoECombineNormsTestInput(t, driver, "expert", []float32{1, 2, 3, 4}, nil, []float32{1, 1, 1, 1}, 0, 0)
	defer expert.Close()
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MoECombineNormsLaunch", "combined output", 16, 4)
	core.RequireNoError(t, err)
	defer output.Close()

	expertCfg.Count = 3
	err = hipRunMoECombineNormsDeviceKernelOutput(context.Background(), driver, local, expert, localCfg, expertCfg, output)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "counts must match")

	expertCfg.Count = 4
	expertCfg.Flags = hipRMSNormLaunchFlagRoPENeoX
	err = hipRunMoECombineNormsDeviceKernelOutput(context.Background(), driver, local, expert, localCfg, expertCfg, output)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported RMSNorm weight flags")
}

func TestHIPMoECombineNormsLaunchArgs_Ugly(t *testing.T) {
	args := hipMoECombineNormsLaunchArgs{
		LocalInputPointer:    11,
		ExpertInputPointer:   21,
		OutputPointer:        31,
		Count:                1,
		LocalInputBytes:      4,
		ExpertInputBytes:     4,
		OutputBytes:          4,
		LocalWeightEncoding:  hipRMSNormWeightEncodingNone,
		ExpertWeightEncoding: hipRMSNormWeightEncodingNone,
	}

	payload, err := args.Binary()
	core.RequireNoError(t, err)
	defer hipReleaseLaunchPacket(payload)
	core.AssertEqual(t, uint32(0), binary.LittleEndian.Uint32(payload[16:]))
	core.AssertEqual(t, uint32(0), binary.LittleEndian.Uint32(payload[32:]))
	core.AssertEqual(t, hipRMSNormWeightEncodingNone, binary.LittleEndian.Uint32(payload[76:]))
	core.AssertEqual(t, hipRMSNormWeightEncodingNone, binary.LittleEndian.Uint32(payload[88:]))
}

func hipMoECombineNormsTestInput(t *testing.T, driver nativeHIPDriver, label string, values []float32, bf16 []uint16, f32 []float32, flags uint32, epsilon float32) (*hipDeviceByteBuffer, hipRMSNormDeviceWeightConfig) {
	t.Helper()
	input, err := hipUploadByteBuffer(driver, "rocm.hip.MoECombineNormsLaunch", label+" input", mustHIPFloat32Payload(t, values), len(values))
	core.RequireNoError(t, err)
	cfg := hipRMSNormDeviceWeightConfig{Count: len(values), Epsilon: epsilon, Flags: flags}
	if len(bf16) > 0 {
		weight, weightErr := hipUploadByteBuffer(driver, "rocm.hip.MoECombineNormsLaunch", label+" bf16 weight", hipUint16PayloadForMoECombineNormsTest(t, bf16), len(bf16))
		core.RequireNoError(t, weightErr)
		cfg.WeightPointer = weight.Pointer()
		cfg.WeightBytes = weight.SizeBytes()
		cfg.WeightEncoding = hipRMSNormWeightEncodingBF16
		t.Cleanup(func() { _ = weight.Close() })
	} else {
		weight, weightErr := hipUploadByteBuffer(driver, "rocm.hip.MoECombineNormsLaunch", label+" f32 weight", mustHIPFloat32Payload(t, f32), len(f32))
		core.RequireNoError(t, weightErr)
		cfg.WeightPointer = weight.Pointer()
		cfg.WeightBytes = weight.SizeBytes()
		cfg.WeightEncoding = hipRMSNormWeightEncodingF32
		t.Cleanup(func() { _ = weight.Close() })
	}
	return input, cfg
}

func hipUint16PayloadForMoECombineNormsTest(t *testing.T, values []uint16) []byte {
	t.Helper()
	payload, err := hipUint16Payload(values)
	core.RequireNoError(t, err)
	return payload
}
