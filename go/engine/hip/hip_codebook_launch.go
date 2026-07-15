// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"

	core "dappco.re/go"
)

const (
	hipCodebookLaunchArgsVersion uint32 = 1
	hipCodebookLaunchArgsBytes          = 64
)

type hipCodebookLookupRequest struct {
	Codes    []uint8
	Codebook []float32
	CodeDim  int
}

type hipCodebookDeviceBuffers struct {
	Codes         *hipDeviceByteBuffer
	Codebook      *hipDeviceByteBuffer
	Output        *hipDeviceByteBuffer
	CodeCount     int
	CodebookCount int
	CodeDim       int
}

type hipCodebookLaunchArgs struct {
	CodePointer     nativeDevicePointer
	CodebookPointer nativeDevicePointer
	OutputPointer   nativeDevicePointer
	CodeCount       int
	CodebookCount   int
	CodeDim         int
	CodeBytes       uint64
	CodebookBytes   uint64
	OutputBytes     uint64
}

func (req hipCodebookLookupRequest) validate() error {
	if len(req.Codes) == 0 {
		return core.E("rocm.hip.CodebookLaunch", "codes are required", nil)
	}
	if req.CodeDim <= 0 {
		return core.E("rocm.hip.CodebookLaunch", "code dimension must be positive", nil)
	}
	if len(req.Codebook) == 0 || len(req.Codebook)%req.CodeDim != 0 {
		return core.E("rocm.hip.CodebookLaunch", "codebook shape does not match code dimension", nil)
	}
	if _, err := rocmReferenceCodebookLookup(req.Codes, req.Codebook, req.CodeDim); err != nil {
		return err
	}
	return nil
}

func (req hipCodebookLookupRequest) deviceBuffers(driver nativeHIPDriver) (*hipCodebookDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	codes, err := hipUploadByteBuffer(driver, "rocm.hip.CodebookLaunch", "codebook codes", append([]byte(nil), req.Codes...), len(req.Codes))
	if err != nil {
		return nil, err
	}
	buffers := &hipCodebookDeviceBuffers{
		Codes:         codes,
		CodeCount:     len(req.Codes),
		CodebookCount: len(req.Codebook) / req.CodeDim,
		CodeDim:       req.CodeDim,
	}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	codebookPayload, err := hipFloat32Payload(req.Codebook)
	if err != nil {
		return nil, core.E("rocm.hip.CodebookLaunch", "encode codebook", err)
	}
	codebook, err := hipUploadByteBuffer(driver, "rocm.hip.CodebookLaunch", "codebook table", codebookPayload, len(req.Codebook))
	if err != nil {
		return nil, err
	}
	buffers.Codebook = codebook
	outputCount := len(req.Codes) * req.CodeDim
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.CodebookLaunch", "codebook output", uint64(outputCount*4), outputCount)
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipCodebookLookupRequest) launchArgs(buffers *hipCodebookDeviceBuffers) (hipCodebookLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipCodebookLaunchArgs{}, err
	}
	if buffers == nil || buffers.Codes == nil || buffers.Codebook == nil || buffers.Output == nil {
		return hipCodebookLaunchArgs{}, core.E("rocm.hip.CodebookLaunch", "codebook device buffers are required", nil)
	}
	codebookCount := len(req.Codebook) / req.CodeDim
	outputCount := len(req.Codes) * req.CodeDim
	if buffers.CodeCount != len(req.Codes) || buffers.CodebookCount != codebookCount || buffers.CodeDim != req.CodeDim ||
		buffers.Codes.Count() != len(req.Codes) || buffers.Codebook.Count() != len(req.Codebook) || buffers.Output.Count() != outputCount {
		return hipCodebookLaunchArgs{}, core.E("rocm.hip.CodebookLaunch", "codebook device buffer shape mismatch", nil)
	}
	return hipCodebookLaunchArgs{
		CodePointer:     buffers.Codes.Pointer(),
		CodebookPointer: buffers.Codebook.Pointer(),
		OutputPointer:   buffers.Output.Pointer(),
		CodeCount:       len(req.Codes),
		CodebookCount:   codebookCount,
		CodeDim:         req.CodeDim,
		CodeBytes:       buffers.Codes.SizeBytes(),
		CodebookBytes:   buffers.Codebook.SizeBytes(),
		OutputBytes:     buffers.Output.SizeBytes(),
	}, nil
}

func (args hipCodebookLaunchArgs) Binary() ([]byte, error) {
	payload := make([]byte, hipCodebookLaunchArgsBytes)
	return args.BinaryInto(payload)
}

func (args hipCodebookLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.CodePointer == 0 || args.CodebookPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.CodebookLaunch", "code, codebook, and output pointers are required", nil)
	}
	if len(payload) < hipCodebookLaunchArgsBytes {
		return nil, core.E("rocm.hip.CodebookLaunch", "launch arg payload buffer is too small", nil)
	}
	payload = payload[:hipCodebookLaunchArgsBytes]
	codeCount, err := rocmDeviceKVPositiveUint32("code count", args.CodeCount)
	if err != nil {
		return nil, err
	}
	codebookCount, err := rocmDeviceKVPositiveUint32("codebook count", args.CodebookCount)
	if err != nil {
		return nil, err
	}
	codeDim, err := rocmDeviceKVPositiveUint32("code dimension", args.CodeDim)
	if err != nil {
		return nil, err
	}
	if args.CodeBytes != uint64(codeCount) {
		return nil, core.E("rocm.hip.CodebookLaunch", "code byte count mismatch", nil)
	}
	if args.CodeBytes > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip.CodebookLaunch", "code bytes are out of uint32 range", nil)
	}
	codebookEntries, err := rocmDeviceKVPositiveUint32("codebook entries", args.CodebookCount*args.CodeDim)
	if err != nil {
		return nil, err
	}
	codebookBytes, err := hipAlignedFloat32Bytes("codebook table", args.CodebookBytes, codebookEntries)
	if err != nil {
		return nil, core.E("rocm.hip.CodebookLaunch", "codebook byte count", err)
	}
	outputEntries, err := rocmDeviceKVPositiveUint32("output entries", args.CodeCount*args.CodeDim)
	if err != nil {
		return nil, err
	}
	outputBytes, err := hipAlignedFloat32Bytes("codebook output", args.OutputBytes, outputEntries)
	if err != nil {
		return nil, core.E("rocm.hip.CodebookLaunch", "output byte count", err)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipCodebookLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.CodePointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.CodebookPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], codeCount)
	binary.LittleEndian.PutUint32(payload[36:], codebookCount)
	binary.LittleEndian.PutUint32(payload[40:], codeDim)
	binary.LittleEndian.PutUint32(payload[44:], uint32(args.CodeBytes))
	binary.LittleEndian.PutUint32(payload[48:], codebookBytes)
	binary.LittleEndian.PutUint32(payload[52:], outputBytes)
	return payload, nil
}

func (buffers *hipCodebookDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Codebook, buffers.Codes} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipCodebookDeviceBuffers) ReadOutput() ([]float32, error) {
	if buffers == nil {
		return nil, core.E("rocm.hip.CodebookLaunch", "codebook output buffer is required", nil)
	}
	outputCount := buffers.CodeCount * buffers.CodeDim
	payload := make([]byte, outputCount*4)
	values := make([]float32, outputCount)
	return buffers.ReadOutputInto(values, payload)
}

func (buffers *hipCodebookDeviceBuffers) ReadOutputInto(values []float32, payload []byte) ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.CodebookLaunch", "codebook output buffer is required", nil)
	}
	outputCount := buffers.CodeCount * buffers.CodeDim
	if buffers.CodeCount <= 0 || buffers.CodeDim <= 0 || buffers.Output.Count() != outputCount || buffers.Output.SizeBytes() != uint64(outputCount*4) {
		return nil, core.E("rocm.hip.CodebookLaunch", "codebook output byte count mismatch", nil)
	}
	if len(payload) < int(buffers.Output.SizeBytes()) {
		return nil, core.E("rocm.hip.CodebookLaunch", "codebook output payload buffer is too small", nil)
	}
	payload = payload[:buffers.Output.SizeBytes()]
	if err := buffers.Output.driver.CopyDeviceToHost(buffers.Output.Pointer(), payload); err != nil {
		return nil, core.E("rocm.hip.CodebookLaunch", "copy codebook output", err)
	}
	values, err := hipFloat32PayloadValuesInto(values, payload)
	if err != nil {
		return nil, err
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E("rocm.hip.CodebookLaunch", "codebook output values must be finite", nil)
	}
	return values, nil
}

func hipRunCodebookLookupKernel(ctx context.Context, driver nativeHIPDriver, req hipCodebookLookupRequest) ([]float32, error) {
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
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameCodebook, launchBytes, len(req.Codes)*req.CodeDim)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}
