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
	hipJANGTQLaunchArgsVersion uint32 = 1
	hipJANGTQLaunchArgsBytes          = 96
)

const hipJANGTQLaunchFlagBias uint32 = 1

type hipJANGTQProjectionRequest struct {
	Input         []float32
	PackedWeights []byte
	Descriptor    rocmJANGTQDescriptor
	Rows          int
	Cols          int
	Scale         float32
	Bias          []float32
}

type hipJANGTQDeviceBuffers struct {
	Input  *hipDeviceByteBuffer
	Packed *hipDeviceByteBuffer
	Bias   *hipDeviceByteBuffer
	Output *hipDeviceByteBuffer
	Rows   int
	Cols   int
	Bits   int
}

type hipJANGTQLaunchArgs struct {
	InputPointer  nativeDevicePointer
	PackedPointer nativeDevicePointer
	BiasPointer   nativeDevicePointer
	OutputPointer nativeDevicePointer
	InputCount    int
	Rows          int
	Cols          int
	Bits          int
	GroupSize     int
	InputBytes    uint64
	PackedBytes   uint64
	BiasBytes     uint64
	OutputBytes   uint64
	Scale         float32
	Flags         uint32
}

func (req hipJANGTQProjectionRequest) validate() error {
	if err := validateROCmJANGTQDescriptor(req.Descriptor); err != nil {
		return err
	}
	if !hipQ8ScaleIsPositiveFinite(req.Scale) {
		return core.E("rocm.hip.JANGTQLaunch", "scale must be positive and finite", nil)
	}
	if err := validateHIPProjectionShape(len(req.Input), req.Rows*req.Cols, len(req.Bias), req.Rows, req.Cols); err != nil {
		return err
	}
	if !rocmFloat32SliceFinite(req.Input) || !rocmFloat32SliceFinite(req.Bias) {
		return core.E("rocm.hip.JANGTQLaunch", "input and bias values must be finite", nil)
	}
	requiredBytes := packedROCmJANGTQBytes(req.Descriptor.Bits, req.Rows*req.Cols)
	if len(req.PackedWeights) < requiredBytes {
		return core.E("rocm.hip.JANGTQLaunch", core.Sprintf("packed weights need %d bytes, got %d", requiredBytes, len(req.PackedWeights)), nil)
	}
	return nil
}

func (req hipJANGTQProjectionRequest) deviceBuffers(driver nativeHIPDriver) (*hipJANGTQDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	inputPayload, err := hipFloat32Payload(req.Input)
	if err != nil {
		return nil, core.E("rocm.hip.JANGTQLaunch", "encode input", err)
	}
	input, err := hipUploadByteBuffer(driver, "rocm.hip.JANGTQLaunch", "JANGTQ input", inputPayload, len(req.Input))
	if err != nil {
		return nil, err
	}
	buffers := &hipJANGTQDeviceBuffers{Input: input, Rows: req.Rows, Cols: req.Cols, Bits: req.Descriptor.Bits}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	packed, err := hipUploadByteBuffer(driver, "rocm.hip.JANGTQLaunch", "JANGTQ packed weights", req.PackedWeights, len(req.PackedWeights))
	if err != nil {
		return nil, err
	}
	buffers.Packed = packed
	if len(req.Bias) > 0 {
		biasPayload, err := hipFloat32Payload(req.Bias)
		if err != nil {
			return nil, core.E("rocm.hip.JANGTQLaunch", "encode bias", err)
		}
		bias, err := hipUploadByteBuffer(driver, "rocm.hip.JANGTQLaunch", "JANGTQ bias", biasPayload, len(req.Bias))
		if err != nil {
			return nil, err
		}
		buffers.Bias = bias
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.JANGTQLaunch", "JANGTQ output", uint64(req.Rows*4), req.Rows)
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipJANGTQProjectionRequest) launchArgs(buffers *hipJANGTQDeviceBuffers) (hipJANGTQLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipJANGTQLaunchArgs{}, err
	}
	if buffers == nil || buffers.Input == nil || buffers.Packed == nil || buffers.Output == nil {
		return hipJANGTQLaunchArgs{}, core.E("rocm.hip.JANGTQLaunch", "JANGTQ device buffers are required", nil)
	}
	if buffers.Input.Count() != req.Cols || buffers.Packed.Count() != len(req.PackedWeights) || buffers.Output.Count() != req.Rows ||
		buffers.Rows != req.Rows || buffers.Cols != req.Cols || buffers.Bits != req.Descriptor.Bits {
		return hipJANGTQLaunchArgs{}, core.E("rocm.hip.JANGTQLaunch", "JANGTQ device buffer shape mismatch", nil)
	}
	var biasPointer nativeDevicePointer
	var biasBytes uint64
	var flags uint32
	if len(req.Bias) > 0 {
		if buffers.Bias == nil || buffers.Bias.Count() != req.Rows {
			return hipJANGTQLaunchArgs{}, core.E("rocm.hip.JANGTQLaunch", "JANGTQ bias buffer shape mismatch", nil)
		}
		biasPointer = buffers.Bias.Pointer()
		biasBytes = buffers.Bias.SizeBytes()
		flags |= hipJANGTQLaunchFlagBias
	}
	return hipJANGTQLaunchArgs{
		InputPointer:  buffers.Input.Pointer(),
		PackedPointer: buffers.Packed.Pointer(),
		BiasPointer:   biasPointer,
		OutputPointer: buffers.Output.Pointer(),
		InputCount:    buffers.Input.Count(),
		Rows:          req.Rows,
		Cols:          req.Cols,
		Bits:          req.Descriptor.Bits,
		GroupSize:     req.Descriptor.GroupSize,
		InputBytes:    buffers.Input.SizeBytes(),
		PackedBytes:   buffers.Packed.SizeBytes(),
		BiasBytes:     biasBytes,
		OutputBytes:   buffers.Output.SizeBytes(),
		Scale:         req.Scale,
		Flags:         flags,
	}, nil
}

func (args hipJANGTQLaunchArgs) Binary() ([]byte, error) {
	payload := make([]byte, hipJANGTQLaunchArgsBytes)
	return args.BinaryInto(payload)
}

func (args hipJANGTQLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.PackedPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.JANGTQLaunch", "input, packed weight, and output pointers are required", nil)
	}
	if len(payload) < hipJANGTQLaunchArgsBytes {
		return nil, core.E("rocm.hip.JANGTQLaunch", "launch arg payload buffer is too small", nil)
	}
	payload = payload[:hipJANGTQLaunchArgsBytes]
	if err := validateROCmJANGTQDescriptor(rocmJANGTQDescriptor{WeightFormat: "mxtq", Bits: args.Bits, GroupSize: args.GroupSize}); err != nil {
		return nil, err
	}
	if !hipQ8ScaleIsPositiveFinite(args.Scale) {
		return nil, core.E("rocm.hip.JANGTQLaunch", "scale must be positive and finite", nil)
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
		return nil, core.E("rocm.hip.JANGTQLaunch", "input count must match cols", nil)
	}
	inputBytes, err := hipAlignedFloat32Bytes("JANGTQ input", args.InputBytes, cols)
	if err != nil {
		return nil, core.E("rocm.hip.JANGTQLaunch", "input byte count", err)
	}
	outputBytes, err := hipAlignedFloat32Bytes("JANGTQ output", args.OutputBytes, rows)
	if err != nil {
		return nil, core.E("rocm.hip.JANGTQLaunch", "output byte count", err)
	}
	requiredPacked := packedROCmJANGTQBytes(args.Bits, args.Rows*args.Cols)
	if args.PackedBytes < uint64(requiredPacked) {
		return nil, core.E("rocm.hip.JANGTQLaunch", core.Sprintf("packed weights need %d bytes, got %d", requiredPacked, args.PackedBytes), nil)
	}
	if args.PackedBytes > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip.JANGTQLaunch", "packed weight bytes are out of uint32 range", nil)
	}
	var biasBytes uint32
	if args.Flags&hipJANGTQLaunchFlagBias != 0 {
		if args.BiasPointer == 0 {
			return nil, core.E("rocm.hip.JANGTQLaunch", "bias pointer is nil", nil)
		}
		biasBytes, err = hipAlignedFloat32Bytes("JANGTQ bias", args.BiasBytes, rows)
		if err != nil {
			return nil, core.E("rocm.hip.JANGTQLaunch", "bias byte count", err)
		}
	} else if args.BiasPointer != 0 || args.BiasBytes != 0 {
		return nil, core.E("rocm.hip.JANGTQLaunch", "bias metadata supplied without bias flag", nil)
	}
	bits := uint32(args.Bits)
	groupSize := uint32(args.GroupSize)
	binary.LittleEndian.PutUint32(payload[0:], hipJANGTQLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.PackedPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[40:], inputCount)
	binary.LittleEndian.PutUint32(payload[44:], rows)
	binary.LittleEndian.PutUint32(payload[48:], cols)
	binary.LittleEndian.PutUint32(payload[52:], bits)
	binary.LittleEndian.PutUint32(payload[56:], groupSize)
	binary.LittleEndian.PutUint32(payload[60:], inputBytes)
	binary.LittleEndian.PutUint32(payload[64:], uint32(args.PackedBytes))
	binary.LittleEndian.PutUint32(payload[68:], biasBytes)
	binary.LittleEndian.PutUint32(payload[72:], outputBytes)
	binary.LittleEndian.PutUint32(payload[76:], math.Float32bits(args.Scale))
	binary.LittleEndian.PutUint32(payload[80:], args.Flags)
	return payload, nil
}

func (buffers *hipJANGTQDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Bias, buffers.Packed, buffers.Input} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipJANGTQDeviceBuffers) ReadOutput() ([]float32, error) {
	if buffers == nil {
		return nil, core.E("rocm.hip.JANGTQLaunch", "JANGTQ output buffer is required", nil)
	}
	payload := make([]byte, buffers.Rows*4)
	values := make([]float32, buffers.Rows)
	return buffers.ReadOutputInto(values, payload)
}

func (buffers *hipJANGTQDeviceBuffers) ReadOutputInto(values []float32, payload []byte) ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.JANGTQLaunch", "JANGTQ output buffer is required", nil)
	}
	if buffers.Rows <= 0 || buffers.Output.Count() != buffers.Rows || buffers.Output.SizeBytes() != uint64(buffers.Rows*4) {
		return nil, core.E("rocm.hip.JANGTQLaunch", "JANGTQ output byte count mismatch", nil)
	}
	if len(payload) < int(buffers.Output.SizeBytes()) {
		return nil, core.E("rocm.hip.JANGTQLaunch", "JANGTQ output payload buffer is too small", nil)
	}
	payload = payload[:buffers.Output.SizeBytes()]
	if err := buffers.Output.driver.CopyDeviceToHost(buffers.Output.Pointer(), payload); err != nil {
		return nil, core.E("rocm.hip.JANGTQLaunch", "copy JANGTQ output", err)
	}
	values, err := hipFloat32PayloadValuesInto(values, payload)
	if err != nil {
		return nil, err
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E("rocm.hip.JANGTQLaunch", "JANGTQ output values must be finite", nil)
	}
	return values, nil
}

func hipRunJANGTQProjectionKernel(ctx context.Context, driver nativeHIPDriver, req hipJANGTQProjectionRequest) ([]float32, error) {
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
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameJANGTQ, launchBytes, req.Rows)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func packedROCmJANGTQBytes(bits, count int) int {
	return (bits*count + 7) / 8
}
