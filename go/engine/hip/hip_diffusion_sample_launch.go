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
	hipDiffusionSampleLaunchArgsVersion uint32 = 1
	hipDiffusionSampleLaunchArgsBytes          = 72
	hipDiffusionSampleResultBytes              = 16
	hipDiffusionSampleBlockSize                = 32
	hipDiffusionSampleWideBlockSize            = 512
	hipDiffusionSampleStatusOK          uint32 = 0x44534653
	hipDisableDiffusionSampleWideEnv           = "GO_ROCM_DISABLE_DIFFUSION_SAMPLE_WIDE"
)

type hipDiffusionSampleResult struct {
	Sampled int32
	Greedy  int32
	Entropy float32
}

type hipDiffusionSampleLaunchArgs struct {
	LogitsPointer nativeDevicePointer
	DrawsPointer  nativeDevicePointer
	OutputPointer nativeDevicePointer
	Rows          int
	VocabSize     int
	Temperature   float32
	Softcap       float32
	LogitsBytes   uint64
	DrawsBytes    uint64
	OutputBytes   uint64
}

func (args hipDiffusionSampleLaunchArgs) Binary() ([]byte, error) {
	const op = "rocm.hip.DiffusionSampleLaunch"
	if args.LogitsPointer == 0 || args.DrawsPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E(op, "logits, draws, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("diffusion sample rows", args.Rows)
	if err != nil {
		return nil, err
	}
	vocab, err := rocmDeviceKVPositiveUint32("diffusion sample vocab size", args.VocabSize)
	if err != nil {
		return nil, err
	}
	if args.LogitsBytes != uint64(rows)*uint64(vocab)*4 {
		return nil, core.E(op, "logit byte count mismatch", nil)
	}
	if args.DrawsBytes != uint64(rows)*4 {
		return nil, core.E(op, "draw byte count mismatch", nil)
	}
	if args.OutputBytes != uint64(rows)*hipDiffusionSampleResultBytes {
		return nil, core.E(op, "output byte count mismatch", nil)
	}
	if args.Temperature <= 0 || math.IsNaN(float64(args.Temperature)) || math.IsInf(float64(args.Temperature), 0) {
		return nil, core.E(op, "temperature must be positive and finite", nil)
	}
	if args.Softcap < 0 || math.IsNaN(float64(args.Softcap)) || math.IsInf(float64(args.Softcap), 0) {
		return nil, core.E(op, "softcap must be non-negative and finite", nil)
	}

	payload := hipBorrowLaunchPacket(hipDiffusionSampleLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipDiffusionSampleLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.LogitsPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.DrawsPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], rows)
	binary.LittleEndian.PutUint32(payload[36:], vocab)
	binary.LittleEndian.PutUint32(payload[40:], math.Float32bits(args.Temperature))
	binary.LittleEndian.PutUint32(payload[44:], math.Float32bits(args.Softcap))
	binary.LittleEndian.PutUint64(payload[48:], args.LogitsBytes)
	binary.LittleEndian.PutUint64(payload[56:], args.DrawsBytes)
	binary.LittleEndian.PutUint64(payload[64:], args.OutputBytes)
	return payload, nil
}

func hipDiffusionSampleLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	const op = "rocm.hip.DiffusionSampleLaunch"
	if len(args) != hipDiffusionSampleLaunchArgsBytes {
		return hipKernelLaunchConfig{}, core.E(op, "launch argument byte count mismatch", nil)
	}
	gridX, err := rocmDeviceKVPositiveUint32("diffusion sample rows", rows)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	kernelName := hipKernelNameDiffusionSampleProbabilitiesWide
	blockX := uint32(hipDiffusionSampleWideBlockSize)
	if core.Env(hipDisableDiffusionSampleWideEnv) == "1" {
		kernelName = hipKernelNameDiffusionSampleProbabilities
		blockX = hipDiffusionSampleBlockSize
	}
	return hipKernelLaunchConfig{
		Name:   kernelName,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: blockX,
		BlockY: 1,
		BlockZ: 1,
	}, nil
}

func hipRunDiffusionSampleKernel(ctx context.Context, driver nativeHIPDriver, logits *hipDeviceByteBuffer, rows, vocab int, temperature, softcap float32, draws []float32) ([]hipDiffusionSampleResult, error) {
	const op = "rocm.hip.DiffusionSampleLaunch"
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E(op, "HIP driver is not available", nil)
	}
	if logits == nil || logits.Pointer() == 0 {
		return nil, core.E(op, "logits device buffer is required", nil)
	}
	if rows <= 0 || vocab <= 0 || len(draws) != rows {
		return nil, core.E(op, "row, vocabulary, and draw geometry mismatch", nil)
	}
	if logits.SizeBytes() != uint64(rows)*uint64(vocab)*4 {
		return nil, core.E(op, "logit byte count mismatch", nil)
	}
	drawPayload, err := hipFloat32Payload(draws)
	if err != nil {
		return nil, core.E(op, "encode categorical draws", err)
	}
	drawBuffer, err := hipUploadByteBuffer(driver, op, "diffusion categorical draws", drawPayload, rows)
	if err != nil {
		return nil, err
	}
	defer drawBuffer.Close()
	output, err := hipAllocateByteBuffer(driver, op, "diffusion sample results", uint64(rows*hipDiffusionSampleResultBytes), rows)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	args, err := (hipDiffusionSampleLaunchArgs{
		LogitsPointer: logits.Pointer(),
		DrawsPointer:  drawBuffer.Pointer(),
		OutputPointer: output.Pointer(),
		Rows:          rows,
		VocabSize:     vocab,
		Temperature:   temperature,
		Softcap:       softcap,
		LogitsBytes:   logits.SizeBytes(),
		DrawsBytes:    drawBuffer.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipDiffusionSampleLaunchConfig(args, rows)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	payload := make([]byte, rows*hipDiffusionSampleResultBytes)
	if err := driver.CopyDeviceToHost(output.Pointer(), payload); err != nil {
		return nil, core.E(op, "copy diffusion sample results", err)
	}
	results := make([]hipDiffusionSampleResult, rows)
	for row := range rows {
		base := row * hipDiffusionSampleResultBytes
		status := binary.LittleEndian.Uint32(payload[base+12:])
		if status != hipDiffusionSampleStatusOK {
			return nil, core.E(op, core.Sprintf("diffusion sample row %d did not complete", row), nil)
		}
		result := hipDiffusionSampleResult{
			Sampled: int32(binary.LittleEndian.Uint32(payload[base:])),
			Greedy:  int32(binary.LittleEndian.Uint32(payload[base+4:])),
			Entropy: math.Float32frombits(binary.LittleEndian.Uint32(payload[base+8:])),
		}
		if result.Sampled < 0 || int(result.Sampled) >= vocab || result.Greedy < 0 || int(result.Greedy) >= vocab || math.IsNaN(float64(result.Entropy)) || math.IsInf(float64(result.Entropy), 0) {
			return nil, core.E(op, core.Sprintf("diffusion sample row %d returned invalid values", row), nil)
		}
		results[row] = result
	}
	return results, nil
}
