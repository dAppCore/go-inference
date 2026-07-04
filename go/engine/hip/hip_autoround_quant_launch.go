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
	hipAutoRoundQuantizeLaunchArgsVersion uint32 = 1
	hipAutoRoundQuantizeLaunchArgsBytes          = 96
)

const (
	hipAutoRoundFormatMXFP4 uint32 = 1
	hipAutoRoundFormatNVFP4 uint32 = 2
	hipAutoRoundFormatFP8   uint32 = 3
	hipAutoRoundFormatMXFP8 uint32 = 4
	hipAutoRoundFormatINT2  uint32 = 5
)

type hipAutoRoundQuantizeRequest struct {
	Weights []float32
	Plan    ProductionAutoRoundCalibrationPlan
	Rows    int
	Cols    int
}

type hipAutoRoundQuantizeDeviceBuffers struct {
	Weights      *hipDeviceByteBuffer
	PackedOutput *hipDeviceByteBuffer
	ScaleOutput  *hipDeviceByteBuffer
	Rows         int
	Cols         int
	FormatCode   uint32
	Bits         int
	GroupSize    int
	GroupsPerRow int
}

type hipAutoRoundQuantizeLaunchArgs struct {
	WeightPointer nativeDevicePointer
	PackedPointer nativeDevicePointer
	ScalePointer  nativeDevicePointer
	Rows          int
	Cols          int
	FormatCode    uint32
	Bits          int
	GroupSize     int
	GroupsPerRow  int
	WeightBytes   uint64
	PackedBytes   uint64
	ScaleBytes    uint64
	NSamples      int
	SeqLen        int
	Iters         int
}

type hipAutoRoundQuantizeResult struct {
	Packed []byte
	Scales []float32
}

func (req hipAutoRoundQuantizeRequest) validate() (uint32, int, error) {
	if req.Rows <= 0 || req.Cols <= 0 {
		return 0, 0, core.E("rocm.hip.AutoRoundQuantizeLaunch", "rows and cols must be positive", nil)
	}
	if len(req.Weights) != req.Rows*req.Cols {
		return 0, 0, core.E("rocm.hip.AutoRoundQuantizeLaunch", "weight length must match rows*cols", nil)
	}
	if !rocmFloat32SliceFinite(req.Weights) {
		return 0, 0, core.E("rocm.hip.AutoRoundQuantizeLaunch", "weight values must be finite", nil)
	}
	if req.Plan.Runtime != "planned_hip" {
		return 0, 0, core.E("rocm.hip.AutoRoundQuantizeLaunch", "AutoRound plan runtime must be planned_hip", nil)
	}
	if req.Plan.HIPKernel != hipKernelStatusNotLinked {
		return 0, 0, core.E("rocm.hip.AutoRoundQuantizeLaunch", "AutoRound HIP quant kernel must remain not_linked until linked", nil)
	}
	if req.Plan.GroupSize <= 0 || req.Cols%req.Plan.GroupSize != 0 {
		return 0, 0, core.E("rocm.hip.AutoRoundQuantizeLaunch", "cols must be divisible by AutoRound group size", nil)
	}
	formatCode, err := hipAutoRoundFormatCode(req.Plan.FloatFormat, req.Plan.Bits)
	if err != nil {
		return 0, 0, err
	}
	groupsPerRow := req.Cols / req.Plan.GroupSize
	return formatCode, groupsPerRow, nil
}

func (req hipAutoRoundQuantizeRequest) deviceBuffers(driver nativeHIPDriver) (*hipAutoRoundQuantizeDeviceBuffers, error) {
	formatCode, groupsPerRow, err := req.validate()
	if err != nil {
		return nil, err
	}
	weightPayload, err := hipFloat32Payload(req.Weights)
	if err != nil {
		return nil, core.E("rocm.hip.AutoRoundQuantizeLaunch", "encode weights", err)
	}
	weights, err := hipUploadByteBuffer(driver, "rocm.hip.AutoRoundQuantizeLaunch", "AutoRound source weights", weightPayload, len(req.Weights))
	if err != nil {
		return nil, err
	}
	buffers := &hipAutoRoundQuantizeDeviceBuffers{
		Weights:      weights,
		Rows:         req.Rows,
		Cols:         req.Cols,
		FormatCode:   formatCode,
		Bits:         req.Plan.Bits,
		GroupSize:    req.Plan.GroupSize,
		GroupsPerRow: groupsPerRow,
	}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()
	packedBytes := hipAutoRoundPackedBytes(req.Plan.Bits, req.Rows*req.Cols)
	packedOutput, err := hipAllocateByteBuffer(driver, "rocm.hip.AutoRoundQuantizeLaunch", "AutoRound packed output", uint64(packedBytes), packedBytes)
	if err != nil {
		return nil, err
	}
	buffers.PackedOutput = packedOutput
	scaleCount := req.Rows * groupsPerRow
	scaleOutput, err := hipAllocateByteBuffer(driver, "rocm.hip.AutoRoundQuantizeLaunch", "AutoRound scale output", uint64(scaleCount*4), scaleCount)
	if err != nil {
		return nil, err
	}
	buffers.ScaleOutput = scaleOutput
	success = true
	return buffers, nil
}

func (req hipAutoRoundQuantizeRequest) launchArgs(buffers *hipAutoRoundQuantizeDeviceBuffers) (hipAutoRoundQuantizeLaunchArgs, error) {
	formatCode, groupsPerRow, err := req.validate()
	if err != nil {
		return hipAutoRoundQuantizeLaunchArgs{}, err
	}
	if buffers == nil || buffers.Weights == nil || buffers.PackedOutput == nil || buffers.ScaleOutput == nil {
		return hipAutoRoundQuantizeLaunchArgs{}, core.E("rocm.hip.AutoRoundQuantizeLaunch", "AutoRound device buffers are required", nil)
	}
	packedBytes := hipAutoRoundPackedBytes(req.Plan.Bits, req.Rows*req.Cols)
	scaleCount := req.Rows * groupsPerRow
	if buffers.Weights.Count() != len(req.Weights) ||
		buffers.PackedOutput.Count() != packedBytes ||
		buffers.ScaleOutput.Count() != scaleCount ||
		buffers.PackedOutput.SizeBytes() != uint64(packedBytes) ||
		buffers.ScaleOutput.SizeBytes() != uint64(scaleCount*4) ||
		buffers.Rows != req.Rows ||
		buffers.Cols != req.Cols ||
		buffers.FormatCode != formatCode ||
		buffers.Bits != req.Plan.Bits ||
		buffers.GroupSize != req.Plan.GroupSize ||
		buffers.GroupsPerRow != groupsPerRow {
		return hipAutoRoundQuantizeLaunchArgs{}, core.E("rocm.hip.AutoRoundQuantizeLaunch", "AutoRound device buffer shape mismatch", nil)
	}
	return hipAutoRoundQuantizeLaunchArgs{
		WeightPointer: buffers.Weights.Pointer(),
		PackedPointer: buffers.PackedOutput.Pointer(),
		ScalePointer:  buffers.ScaleOutput.Pointer(),
		Rows:          req.Rows,
		Cols:          req.Cols,
		FormatCode:    formatCode,
		Bits:          req.Plan.Bits,
		GroupSize:     req.Plan.GroupSize,
		GroupsPerRow:  groupsPerRow,
		WeightBytes:   buffers.Weights.SizeBytes(),
		PackedBytes:   buffers.PackedOutput.SizeBytes(),
		ScaleBytes:    buffers.ScaleOutput.SizeBytes(),
		NSamples:      req.Plan.NSamples,
		SeqLen:        req.Plan.SeqLen,
		Iters:         req.Plan.Iters,
	}, nil
}

func (args hipAutoRoundQuantizeLaunchArgs) Binary() ([]byte, error) {
	payload := make([]byte, hipAutoRoundQuantizeLaunchArgsBytes)
	return args.BinaryInto(payload)
}

func (args hipAutoRoundQuantizeLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.WeightPointer == 0 || args.PackedPointer == 0 || args.ScalePointer == 0 {
		return nil, core.E("rocm.hip.AutoRoundQuantizeLaunch", "weight, packed output, and scale output pointers are required", nil)
	}
	if len(payload) < hipAutoRoundQuantizeLaunchArgsBytes {
		return nil, core.E("rocm.hip.AutoRoundQuantizeLaunch", "launch arg payload buffer is too small", nil)
	}
	payload = payload[:hipAutoRoundQuantizeLaunchArgsBytes]
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	groupsPerRow, err := rocmDeviceKVPositiveUint32("groups per row", args.GroupsPerRow)
	if err != nil {
		return nil, err
	}
	if int(cols)%int(groupSize) != 0 || int(cols)/int(groupSize) != int(groupsPerRow) {
		return nil, core.E("rocm.hip.AutoRoundQuantizeLaunch", "group geometry must match cols", nil)
	}
	if err := hipAutoRoundValidateFormatCode(args.FormatCode, int(bits)); err != nil {
		return nil, err
	}
	weightCount, err := hipUint32Product("AutoRound weights", rows, cols)
	if err != nil {
		return nil, err
	}
	weightBytes, err := hipAlignedFloat32Bytes("AutoRound weights", args.WeightBytes, weightCount)
	if err != nil {
		return nil, core.E("rocm.hip.AutoRoundQuantizeLaunch", "weight byte count", err)
	}
	wantPackedBytes := hipAutoRoundPackedBytes(int(bits), int(weightCount))
	if args.PackedBytes != uint64(wantPackedBytes) || args.PackedBytes > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip.AutoRoundQuantizeLaunch", "packed output byte count mismatch", nil)
	}
	scaleCount, err := hipUint32Product("AutoRound scales", rows, groupsPerRow)
	if err != nil {
		return nil, err
	}
	scaleBytes, err := hipAlignedFloat32Bytes("AutoRound scales", args.ScaleBytes, scaleCount)
	if err != nil {
		return nil, core.E("rocm.hip.AutoRoundQuantizeLaunch", "scale byte count", err)
	}
	nsamples, err := rocmDeviceKVPositiveUint32("AutoRound nsamples", args.NSamples)
	if err != nil {
		return nil, err
	}
	seqlen, err := rocmDeviceKVPositiveUint32("AutoRound seqlen", args.SeqLen)
	if err != nil {
		return nil, err
	}
	iters, err := rocmDeviceKVPositiveUint32("AutoRound iters", args.Iters)
	if err != nil {
		return nil, err
	}
	clear(payload)
	binary.LittleEndian.PutUint32(payload[0:], hipAutoRoundQuantizeLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.PackedPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ScalePointer))
	binary.LittleEndian.PutUint32(payload[32:], rows)
	binary.LittleEndian.PutUint32(payload[36:], cols)
	binary.LittleEndian.PutUint32(payload[40:], args.FormatCode)
	binary.LittleEndian.PutUint32(payload[44:], bits)
	binary.LittleEndian.PutUint32(payload[48:], groupSize)
	binary.LittleEndian.PutUint32(payload[52:], groupsPerRow)
	binary.LittleEndian.PutUint32(payload[56:], weightBytes)
	binary.LittleEndian.PutUint32(payload[60:], uint32(args.PackedBytes))
	binary.LittleEndian.PutUint32(payload[64:], scaleBytes)
	binary.LittleEndian.PutUint32(payload[68:], nsamples)
	binary.LittleEndian.PutUint32(payload[72:], seqlen)
	binary.LittleEndian.PutUint32(payload[76:], iters)
	return payload, nil
}

func (buffers *hipAutoRoundQuantizeDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.ScaleOutput, buffers.PackedOutput, buffers.Weights} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipAutoRoundQuantizeDeviceBuffers) ReadOutput() (hipAutoRoundQuantizeResult, error) {
	if buffers == nil || buffers.PackedOutput == nil || buffers.ScaleOutput == nil {
		return hipAutoRoundQuantizeResult{}, core.E("rocm.hip.AutoRoundQuantizeLaunch", "AutoRound output buffers are required", nil)
	}
	if buffers.PackedOutput.Pointer() == 0 || buffers.ScaleOutput.Pointer() == 0 {
		return hipAutoRoundQuantizeResult{}, core.E("rocm.hip.AutoRoundQuantizeLaunch", "AutoRound output buffer pointers are required", nil)
	}
	if buffers.PackedOutput.Count() != int(buffers.PackedOutput.SizeBytes()) {
		return hipAutoRoundQuantizeResult{}, core.E("rocm.hip.AutoRoundQuantizeLaunch", "AutoRound packed output byte count mismatch", nil)
	}
	if buffers.ScaleOutput.SizeBytes() != uint64(buffers.ScaleOutput.Count()*4) {
		return hipAutoRoundQuantizeResult{}, core.E("rocm.hip.AutoRoundQuantizeLaunch", "AutoRound scale output byte count mismatch", nil)
	}
	packed := make([]byte, buffers.PackedOutput.SizeBytes())
	if err := buffers.PackedOutput.driver.CopyDeviceToHost(buffers.PackedOutput.Pointer(), packed); err != nil {
		return hipAutoRoundQuantizeResult{}, core.E("rocm.hip.AutoRoundQuantizeLaunch", "copy AutoRound packed output", err)
	}
	scalePayload := make([]byte, buffers.ScaleOutput.SizeBytes())
	if err := buffers.ScaleOutput.driver.CopyDeviceToHost(buffers.ScaleOutput.Pointer(), scalePayload); err != nil {
		return hipAutoRoundQuantizeResult{}, core.E("rocm.hip.AutoRoundQuantizeLaunch", "copy AutoRound scale output", err)
	}
	scales := make([]float32, buffers.ScaleOutput.Count())
	for index := range scales {
		scale := math.Float32frombits(binary.LittleEndian.Uint32(scalePayload[index*4:]))
		if !hipQ8ScaleIsPositiveFinite(scale) {
			return hipAutoRoundQuantizeResult{}, core.E("rocm.hip.AutoRoundQuantizeLaunch", "AutoRound scale output must be positive and finite", nil)
		}
		scales[index] = scale
	}
	return hipAutoRoundQuantizeResult{Packed: packed, Scales: scales}, nil
}

func hipRunAutoRoundQuantizeKernel(ctx context.Context, driver nativeHIPDriver, req hipAutoRoundQuantizeRequest) (hipAutoRoundQuantizeResult, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return hipAutoRoundQuantizeResult{}, err
		}
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return hipAutoRoundQuantizeResult{}, err
	}
	defer buffers.Close()
	launchArgs, err := req.launchArgs(buffers)
	if err != nil {
		return hipAutoRoundQuantizeResult{}, err
	}
	packet, err := launchArgs.Binary()
	if err != nil {
		return hipAutoRoundQuantizeResult{}, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameAutoRoundQuantize, packet, req.Rows*buffers.GroupsPerRow)
	if err != nil {
		return hipAutoRoundQuantizeResult{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipAutoRoundQuantizeResult{}, err
	}
	return buffers.ReadOutput()
}

func hipAutoRoundFormatCode(format string, bits int) (uint32, error) {
	switch format {
	case "mxfp4":
		if bits == 4 {
			return hipAutoRoundFormatMXFP4, nil
		}
	case "nvfp4":
		if bits == 4 {
			return hipAutoRoundFormatNVFP4, nil
		}
	case "fp8":
		if bits == 8 {
			return hipAutoRoundFormatFP8, nil
		}
	case "mxfp8":
		if bits == 8 {
			return hipAutoRoundFormatMXFP8, nil
		}
	case "int2":
		if bits == 2 {
			return hipAutoRoundFormatINT2, nil
		}
	}
	return 0, core.E("rocm.hip.AutoRoundQuantizeLaunch", "unsupported AutoRound format and bit layout", nil)
}

func hipAutoRoundValidateFormatCode(code uint32, bits int) error {
	switch code {
	case hipAutoRoundFormatMXFP4, hipAutoRoundFormatNVFP4:
		if bits == 4 {
			return nil
		}
	case hipAutoRoundFormatFP8, hipAutoRoundFormatMXFP8:
		if bits == 8 {
			return nil
		}
	case hipAutoRoundFormatINT2:
		if bits == 2 {
			return nil
		}
	}
	return core.E("rocm.hip.AutoRoundQuantizeLaunch", "unsupported AutoRound format code and bit layout", nil)
}

func hipAutoRoundPackedBytes(bits int, values int) int {
	return (bits*values + 7) / 8
}
