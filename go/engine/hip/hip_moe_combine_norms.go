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
	hipMoECombineNormsLaunchArgsVersion uint32 = 1
	hipMoECombineNormsLaunchArgsBytes          = 96
)

// hipMoECombineNormsLaunchArgs describes one local/expert RMS normalization
// pair followed by their elementwise sum. Both vectors have the same count.
type hipMoECombineNormsLaunchArgs struct {
	LocalInputPointer    nativeDevicePointer
	LocalWeightPointer   nativeDevicePointer
	ExpertInputPointer   nativeDevicePointer
	ExpertWeightPointer  nativeDevicePointer
	OutputPointer        nativeDevicePointer
	Count                int
	LocalInputBytes      uint64
	LocalWeightBytes     uint64
	ExpertInputBytes     uint64
	ExpertWeightBytes    uint64
	OutputBytes          uint64
	LocalEpsilon         float32
	LocalWeightEncoding  uint32
	LocalFlags           uint32
	ExpertEpsilon        float32
	ExpertWeightEncoding uint32
	ExpertFlags          uint32
}

func (args hipMoECombineNormsLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipMoECombineNormsLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.LocalInputPointer == 0 || args.ExpertInputPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.MoECombineNormsLaunch", "local input, expert input, and output pointers are required", nil)
	}
	count, err := rocmDeviceKVPositiveUint32("count", args.Count)
	if err != nil {
		return nil, err
	}
	if args.LocalEpsilon < 0 || math.IsNaN(float64(args.LocalEpsilon)) || math.IsInf(float64(args.LocalEpsilon), 0) {
		return nil, core.E("rocm.hip.MoECombineNormsLaunch", "local epsilon must be non-negative and finite", nil)
	}
	if args.ExpertEpsilon < 0 || math.IsNaN(float64(args.ExpertEpsilon)) || math.IsInf(float64(args.ExpertEpsilon), 0) {
		return nil, core.E("rocm.hip.MoECombineNormsLaunch", "expert epsilon must be non-negative and finite", nil)
	}
	localInputBytes, err := hipAlignedFloat32Bytes("local input", args.LocalInputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.MoECombineNormsLaunch", "local input byte count", err)
	}
	localWeightBytes, err := hipMoECombineNormsWeightBytes("local", args.LocalWeightPointer, args.LocalWeightBytes, count, args.LocalWeightEncoding, args.LocalFlags)
	if err != nil {
		return nil, err
	}
	expertInputBytes, err := hipAlignedFloat32Bytes("expert input", args.ExpertInputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.MoECombineNormsLaunch", "expert input byte count", err)
	}
	expertWeightBytes, err := hipMoECombineNormsWeightBytes("expert", args.ExpertWeightPointer, args.ExpertWeightBytes, count, args.ExpertWeightEncoding, args.ExpertFlags)
	if err != nil {
		return nil, err
	}
	outputBytes, err := hipAlignedFloat32Bytes("output", args.OutputBytes, count)
	if err != nil {
		return nil, core.E("rocm.hip.MoECombineNormsLaunch", "output byte count", err)
	}
	if cap(payload) < hipMoECombineNormsLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipMoECombineNormsLaunchArgsBytes)
	} else {
		payload = payload[:hipMoECombineNormsLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipMoECombineNormsLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.LocalInputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.LocalWeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ExpertInputPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.ExpertWeightPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[48:], count)
	binary.LittleEndian.PutUint32(payload[52:], localInputBytes)
	binary.LittleEndian.PutUint32(payload[56:], localWeightBytes)
	binary.LittleEndian.PutUint32(payload[60:], expertInputBytes)
	binary.LittleEndian.PutUint32(payload[64:], expertWeightBytes)
	binary.LittleEndian.PutUint32(payload[68:], outputBytes)
	binary.LittleEndian.PutUint32(payload[72:], math.Float32bits(args.LocalEpsilon))
	binary.LittleEndian.PutUint32(payload[76:], args.LocalWeightEncoding)
	binary.LittleEndian.PutUint32(payload[80:], args.LocalFlags)
	binary.LittleEndian.PutUint32(payload[84:], math.Float32bits(args.ExpertEpsilon))
	binary.LittleEndian.PutUint32(payload[88:], args.ExpertWeightEncoding)
	binary.LittleEndian.PutUint32(payload[92:], args.ExpertFlags)
	return payload, nil
}

func hipMoECombineNormsWeightBytes(label string, pointer nativeDevicePointer, bytes uint64, count uint32, encoding uint32, flags uint32) (uint32, error) {
	if flags&^hipRMSNormLaunchFlagAddUnitWeight != 0 {
		return 0, core.E("rocm.hip.MoECombineNormsLaunch", "unsupported RMSNorm weight flags", nil)
	}
	return hipRMSNormLaunchWeightBytes("MoECombineNormsLaunch", label+" weight", pointer, bytes, count, encoding, flags)
}

// hipRunMoECombineNormsDeviceKernelOutput writes the fused local and expert
// RMS-normalized sum into caller-owned device storage.
func hipRunMoECombineNormsDeviceKernelOutput(ctx context.Context, driver nativeHIPDriver, local, expert *hipDeviceByteBuffer, localCfg, expertCfg hipRMSNormDeviceWeightConfig, output *hipDeviceByteBuffer) error {
	return hipRunMoECombineNormsDeviceKernelOutputWithWorkspace(ctx, driver, local, expert, localCfg, expertCfg, output, nil)
}

func hipRunMoECombineNormsDeviceKernelOutputWithWorkspace(ctx context.Context, driver nativeHIPDriver, local, expert *hipDeviceByteBuffer, localCfg, expertCfg hipRMSNormDeviceWeightConfig, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.MoECombineNormsLaunch", "HIP driver is not available", nil)
	}
	if local == nil || local.Pointer() == 0 || expert == nil || expert.Pointer() == 0 || output == nil || output.Pointer() == 0 {
		return core.E("rocm.hip.MoECombineNormsLaunch", "local input, expert input, and output device buffers are required", nil)
	}
	if localCfg.Count <= 0 || expertCfg.Count <= 0 || localCfg.Count != expertCfg.Count {
		return core.E("rocm.hip.MoECombineNormsLaunch", "local and expert counts must match and be positive", nil)
	}
	count := localCfg.Count
	if local.Count() != count || expert.Count() != count || output.Count() != count ||
		local.SizeBytes() != uint64(count*4) || expert.SizeBytes() != uint64(count*4) || output.SizeBytes() != uint64(count*4) {
		return core.E("rocm.hip.MoECombineNormsLaunch", "local input, expert input, and output device buffer shapes must match count", nil)
	}
	launchArgs := hipMoECombineNormsLaunchArgs{
		LocalInputPointer:    local.Pointer(),
		LocalWeightPointer:   localCfg.WeightPointer,
		ExpertInputPointer:   expert.Pointer(),
		ExpertWeightPointer:  expertCfg.WeightPointer,
		OutputPointer:        output.Pointer(),
		Count:                count,
		LocalInputBytes:      local.SizeBytes(),
		LocalWeightBytes:     localCfg.WeightBytes,
		ExpertInputBytes:     expert.SizeBytes(),
		ExpertWeightBytes:    expertCfg.WeightBytes,
		OutputBytes:          output.SizeBytes(),
		LocalEpsilon:         localCfg.Epsilon,
		LocalWeightEncoding:  localCfg.WeightEncoding,
		LocalFlags:           localCfg.Flags,
		ExpertEpsilon:        expertCfg.Epsilon,
		ExpertWeightEncoding: expertCfg.WeightEncoding,
		ExpertFlags:          expertCfg.Flags,
	}
	var launchBytes []byte
	var err error
	if workspace != nil {
		launchBytes, err = launchArgs.BinaryInto(workspace.MoE.CombineNormsArgs[:])
	} else {
		launchBytes, err = launchArgs.Binary()
	}
	if err != nil {
		return err
	}
	config, err := hipSingleBlockLaunchConfig(hipKernelNameMoECombineNorms, launchBytes, 256)
	if err != nil {
		hipReleaseLaunchPacket(launchBytes)
		return err
	}
	return hipLaunchKernelContext(ctx, driver, config)
}
