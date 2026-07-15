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
	hipLoRALaunchArgsVersion uint32 = 1
	hipLoRALaunchArgsBytes          = 128
)

const hipLoRALaunchFlagBias uint32 = 1

type hipLoRAProjectionRequest struct {
	Input      []float32
	BaseWeight []float32
	LoRAA      []float32
	LoRAB      []float32
	Rows       int
	Cols       int
	Rank       int
	Alpha      float32
	Bias       []float32
}

type hipLoRADeviceBuffers struct {
	Input      *hipDeviceByteBuffer
	BaseWeight *hipDeviceByteBuffer
	LoRAA      *hipDeviceByteBuffer
	LoRAB      *hipDeviceByteBuffer
	Bias       *hipDeviceByteBuffer
	Output     *hipDeviceByteBuffer
	Rows       int
	Cols       int
	Rank       int
}

type hipLoRALaunchArgs struct {
	InputPointer      nativeDevicePointer
	BaseWeightPointer nativeDevicePointer
	LoRAAPointer      nativeDevicePointer
	LoRABPointer      nativeDevicePointer
	BiasPointer       nativeDevicePointer
	OutputPointer     nativeDevicePointer
	InputCount        int
	Rows              int
	Cols              int
	Rank              int
	InputBytes        uint64
	BaseWeightBytes   uint64
	LoRAABytes        uint64
	LoRABBytes        uint64
	BiasBytes         uint64
	OutputBytes       uint64
	Alpha             float32
	Flags             uint32
}

func (req hipLoRAProjectionRequest) validate() error {
	if !hipQ8ScaleIsPositiveFinite(req.Alpha) {
		return core.E("rocm.hip.LoRALaunch", "alpha must be positive and finite", nil)
	}
	if _, err := rocmReferenceLoRAProjection(req.Input, req.BaseWeight, req.LoRAA, req.LoRAB, req.Rows, req.Cols, req.Rank, req.Alpha, req.Bias); err != nil {
		return err
	}
	return nil
}

func (req hipLoRAProjectionRequest) deviceBuffers(driver nativeHIPDriver) (*hipLoRADeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	inputPayload, err := hipFloat32Payload(req.Input)
	if err != nil {
		return nil, core.E("rocm.hip.LoRALaunch", "encode input", err)
	}
	input, err := hipUploadByteBuffer(driver, "rocm.hip.LoRALaunch", "LoRA input", inputPayload, len(req.Input))
	if err != nil {
		return nil, err
	}
	buffers := &hipLoRADeviceBuffers{Input: input, Rows: req.Rows, Cols: req.Cols, Rank: req.Rank}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	basePayload, err := hipFloat32Payload(req.BaseWeight)
	if err != nil {
		return nil, core.E("rocm.hip.LoRALaunch", "encode base weights", err)
	}
	base, err := hipUploadByteBuffer(driver, "rocm.hip.LoRALaunch", "LoRA base weights", basePayload, len(req.BaseWeight))
	if err != nil {
		return nil, err
	}
	buffers.BaseWeight = base

	aPayload, err := hipFloat32Payload(req.LoRAA)
	if err != nil {
		return nil, core.E("rocm.hip.LoRALaunch", "encode LoRA A", err)
	}
	loraA, err := hipUploadByteBuffer(driver, "rocm.hip.LoRALaunch", "LoRA A", aPayload, len(req.LoRAA))
	if err != nil {
		return nil, err
	}
	buffers.LoRAA = loraA

	bPayload, err := hipFloat32Payload(req.LoRAB)
	if err != nil {
		return nil, core.E("rocm.hip.LoRALaunch", "encode LoRA B", err)
	}
	loraB, err := hipUploadByteBuffer(driver, "rocm.hip.LoRALaunch", "LoRA B", bPayload, len(req.LoRAB))
	if err != nil {
		return nil, err
	}
	buffers.LoRAB = loraB

	if len(req.Bias) > 0 {
		biasPayload, err := hipFloat32Payload(req.Bias)
		if err != nil {
			return nil, core.E("rocm.hip.LoRALaunch", "encode bias", err)
		}
		bias, err := hipUploadByteBuffer(driver, "rocm.hip.LoRALaunch", "LoRA bias", biasPayload, len(req.Bias))
		if err != nil {
			return nil, err
		}
		buffers.Bias = bias
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.LoRALaunch", "LoRA output", uint64(req.Rows*4), req.Rows)
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipLoRAProjectionRequest) launchArgs(buffers *hipLoRADeviceBuffers) (hipLoRALaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipLoRALaunchArgs{}, err
	}
	if buffers == nil || buffers.Input == nil || buffers.BaseWeight == nil || buffers.LoRAA == nil || buffers.LoRAB == nil || buffers.Output == nil {
		return hipLoRALaunchArgs{}, core.E("rocm.hip.LoRALaunch", "LoRA device buffers are required", nil)
	}
	if buffers.Input.Count() != req.Cols ||
		buffers.BaseWeight.Count() != req.Rows*req.Cols ||
		buffers.LoRAA.Count() != req.Rank*req.Cols ||
		buffers.LoRAB.Count() != req.Rows*req.Rank ||
		buffers.Output.Count() != req.Rows ||
		buffers.Rows != req.Rows ||
		buffers.Cols != req.Cols ||
		buffers.Rank != req.Rank {
		return hipLoRALaunchArgs{}, core.E("rocm.hip.LoRALaunch", "LoRA device buffer shape mismatch", nil)
	}
	var biasPointer nativeDevicePointer
	var biasBytes uint64
	var flags uint32
	if len(req.Bias) > 0 {
		if buffers.Bias == nil || buffers.Bias.Count() != req.Rows {
			return hipLoRALaunchArgs{}, core.E("rocm.hip.LoRALaunch", "LoRA bias buffer shape mismatch", nil)
		}
		biasPointer = buffers.Bias.Pointer()
		biasBytes = buffers.Bias.SizeBytes()
		flags |= hipLoRALaunchFlagBias
	}
	return hipLoRALaunchArgs{
		InputPointer:      buffers.Input.Pointer(),
		BaseWeightPointer: buffers.BaseWeight.Pointer(),
		LoRAAPointer:      buffers.LoRAA.Pointer(),
		LoRABPointer:      buffers.LoRAB.Pointer(),
		BiasPointer:       biasPointer,
		OutputPointer:     buffers.Output.Pointer(),
		InputCount:        buffers.Input.Count(),
		Rows:              req.Rows,
		Cols:              req.Cols,
		Rank:              req.Rank,
		InputBytes:        buffers.Input.SizeBytes(),
		BaseWeightBytes:   buffers.BaseWeight.SizeBytes(),
		LoRAABytes:        buffers.LoRAA.SizeBytes(),
		LoRABBytes:        buffers.LoRAB.SizeBytes(),
		BiasBytes:         biasBytes,
		OutputBytes:       buffers.Output.SizeBytes(),
		Alpha:             req.Alpha,
		Flags:             flags,
	}, nil
}

func (args hipLoRALaunchArgs) Binary() ([]byte, error) {
	payload := make([]byte, hipLoRALaunchArgsBytes)
	return args.BinaryInto(payload)
}

func (args hipLoRALaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.BaseWeightPointer == 0 || args.LoRAAPointer == 0 || args.LoRABPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.LoRALaunch", "input, base, LoRA, and output pointers are required", nil)
	}
	if len(payload) < hipLoRALaunchArgsBytes {
		return nil, core.E("rocm.hip.LoRALaunch", "launch arg payload buffer is too small", nil)
	}
	payload = payload[:hipLoRALaunchArgsBytes]
	if !hipQ8ScaleIsPositiveFinite(args.Alpha) {
		return nil, core.E("rocm.hip.LoRALaunch", "alpha must be positive and finite", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	rank, err := rocmDeviceKVPositiveUint32("rank", args.Rank)
	if err != nil {
		return nil, err
	}
	inputCount, err := rocmDeviceKVPositiveUint32("input count", args.InputCount)
	if err != nil {
		return nil, err
	}
	if inputCount != cols {
		return nil, core.E("rocm.hip.LoRALaunch", "input count must match cols", nil)
	}
	inputBytes, err := hipAlignedFloat32Bytes("LoRA input", args.InputBytes, cols)
	if err != nil {
		return nil, core.E("rocm.hip.LoRALaunch", "input byte count", err)
	}
	baseCount, err := hipUint32Product("base weight count", rows, cols)
	if err != nil {
		return nil, err
	}
	baseBytes, err := hipAlignedFloat32Bytes("LoRA base weights", args.BaseWeightBytes, baseCount)
	if err != nil {
		return nil, core.E("rocm.hip.LoRALaunch", "base weight byte count", err)
	}
	aCount, err := hipUint32Product("LoRA A count", rank, cols)
	if err != nil {
		return nil, err
	}
	aBytes, err := hipAlignedFloat32Bytes("LoRA A", args.LoRAABytes, aCount)
	if err != nil {
		return nil, core.E("rocm.hip.LoRALaunch", "LoRA A byte count", err)
	}
	bCount, err := hipUint32Product("LoRA B count", rows, rank)
	if err != nil {
		return nil, err
	}
	bBytes, err := hipAlignedFloat32Bytes("LoRA B", args.LoRABBytes, bCount)
	if err != nil {
		return nil, core.E("rocm.hip.LoRALaunch", "LoRA B byte count", err)
	}
	outputBytes, err := hipAlignedFloat32Bytes("LoRA output", args.OutputBytes, rows)
	if err != nil {
		return nil, core.E("rocm.hip.LoRALaunch", "output byte count", err)
	}
	var biasBytes uint32
	if args.Flags&hipLoRALaunchFlagBias != 0 {
		if args.BiasPointer == 0 {
			return nil, core.E("rocm.hip.LoRALaunch", "bias pointer is nil", nil)
		}
		biasBytes, err = hipAlignedFloat32Bytes("LoRA bias", args.BiasBytes, rows)
		if err != nil {
			return nil, core.E("rocm.hip.LoRALaunch", "bias byte count", err)
		}
	} else if args.BiasPointer != 0 || args.BiasBytes != 0 {
		return nil, core.E("rocm.hip.LoRALaunch", "bias metadata supplied without bias flag", nil)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipLoRALaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.BaseWeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.LoRAAPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.LoRABPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[56:], inputCount)
	binary.LittleEndian.PutUint32(payload[60:], rows)
	binary.LittleEndian.PutUint32(payload[64:], cols)
	binary.LittleEndian.PutUint32(payload[68:], rank)
	binary.LittleEndian.PutUint32(payload[72:], inputBytes)
	binary.LittleEndian.PutUint32(payload[76:], baseBytes)
	binary.LittleEndian.PutUint32(payload[80:], aBytes)
	binary.LittleEndian.PutUint32(payload[84:], bBytes)
	binary.LittleEndian.PutUint32(payload[88:], biasBytes)
	binary.LittleEndian.PutUint32(payload[92:], outputBytes)
	binary.LittleEndian.PutUint32(payload[96:], math.Float32bits(args.Alpha))
	binary.LittleEndian.PutUint32(payload[100:], args.Flags)
	return payload, nil
}

func hipUint32Product(field string, a, b uint32) (uint32, error) {
	product := uint64(a) * uint64(b)
	if product > uint64(^uint32(0)) {
		return 0, core.E("rocm.hip.LaunchBytes", field+" is out of uint32 range", nil)
	}
	return uint32(product), nil
}

func (buffers *hipLoRADeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Bias, buffers.LoRAB, buffers.LoRAA, buffers.BaseWeight, buffers.Input} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipLoRADeviceBuffers) ReadOutput() ([]float32, error) {
	if buffers == nil {
		return nil, core.E("rocm.hip.LoRALaunch", "LoRA output buffer is required", nil)
	}
	payload := make([]byte, buffers.Rows*4)
	values := make([]float32, buffers.Rows)
	return buffers.ReadOutputInto(values, payload)
}

func (buffers *hipLoRADeviceBuffers) ReadOutputInto(values []float32, payload []byte) ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.LoRALaunch", "LoRA output buffer is required", nil)
	}
	if buffers.Rows <= 0 || buffers.Output.Count() != buffers.Rows || buffers.Output.SizeBytes() != uint64(buffers.Rows*4) {
		return nil, core.E("rocm.hip.LoRALaunch", "LoRA output byte count mismatch", nil)
	}
	outputBytes := int(buffers.Output.SizeBytes())
	if len(payload) < outputBytes {
		return nil, core.E("rocm.hip.LoRALaunch", "LoRA output payload buffer is too small", nil)
	}
	payload = payload[:outputBytes]
	if err := buffers.Output.driver.CopyDeviceToHost(buffers.Output.Pointer(), payload); err != nil {
		return nil, core.E("rocm.hip.LoRALaunch", "copy LoRA output", err)
	}
	values, err := hipFloat32PayloadValuesInto(values, payload)
	if err != nil {
		return nil, err
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E("rocm.hip.LoRALaunch", "LoRA output values must be finite", nil)
	}
	return values, nil
}

func hipRunLoRAProjectionKernel(ctx context.Context, driver nativeHIPDriver, req hipLoRAProjectionRequest) ([]float32, error) {
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
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameLoRA, launchBytes, req.Rows)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}
