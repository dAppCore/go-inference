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

func TestHIPDiffusionSampleLaunchArgs_Binary_Good(t *testing.T) {
	args := hipDiffusionSampleLaunchArgs{
		LogitsPointer: 0x1000,
		DrawsPointer:  0x2000,
		OutputPointer: 0x3000,
		Rows:          2,
		VocabSize:     64,
		Temperature:   0.75,
		Softcap:       30,
		LogitsBytes:   2 * 64 * 4,
		DrawsBytes:    2 * 4,
		OutputBytes:   2 * hipDiffusionSampleResultBytes,
	}

	payload, err := args.Binary()
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipDiffusionSampleLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipDiffusionSampleLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint64(args.LogitsPointer), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(args.DrawsPointer), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint64(args.OutputPointer), binary.LittleEndian.Uint64(payload[24:]))
	core.AssertEqual(t, uint32(args.Rows), binary.LittleEndian.Uint32(payload[32:]))
	core.AssertEqual(t, uint32(args.VocabSize), binary.LittleEndian.Uint32(payload[36:]))
	core.AssertEqual(t, math.Float32bits(args.Temperature), binary.LittleEndian.Uint32(payload[40:]))
	core.AssertEqual(t, math.Float32bits(args.Softcap), binary.LittleEndian.Uint32(payload[44:]))
	core.AssertEqual(t, args.LogitsBytes, binary.LittleEndian.Uint64(payload[48:]))
	core.AssertEqual(t, args.DrawsBytes, binary.LittleEndian.Uint64(payload[56:]))
	core.AssertEqual(t, args.OutputBytes, binary.LittleEndian.Uint64(payload[64:]))
}

func TestHIPDiffusionSampleLaunchArgs_Binary_Bad(t *testing.T) {
	_, err := (hipDiffusionSampleLaunchArgs{
		LogitsPointer: 1,
		DrawsPointer:  2,
		OutputPointer: 3,
		Rows:          1,
		VocabSize:     4,
		Temperature:   1,
		LogitsBytes:   12,
		DrawsBytes:    4,
		OutputBytes:   hipDiffusionSampleResultBytes,
	}).Binary()
	core.RequireTrue(t, err != nil)
	core.AssertContains(t, err.Error(), "logit byte count mismatch")
}

func TestHIPDiffusionSampleLaunchConfig_Good(t *testing.T) {
	config, err := hipDiffusionSampleLaunchConfig(make([]byte, hipDiffusionSampleLaunchArgsBytes), 256)
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipKernelNameDiffusionSampleProbabilities, config.Name)
	core.AssertEqual(t, uint32(256), config.GridX)
	core.AssertEqual(t, uint32(1), config.GridY)
	core.AssertEqual(t, uint32(hipDiffusionSampleBlockSize), config.BlockX)
}

func TestHIPDiffusionSampleKernel_Good_KeepsLogitsOnDevice(t *testing.T) {
	restorePool := hipSuppressDeviceByteBufferPool()
	defer restorePool()

	base := &fakeHIPDriver{available: true}
	driver := &hipDiffusionSampleStubDriver{fakeHIPDriver: base}
	logits, err := hipUploadGemma4Q4Float32Input(driver, "diffusion sampler logits", []float32{1, 2, 3, 4, 5, 6, 7, 8})
	core.RequireNoError(t, err)
	defer logits.Close()
	base.copies = nil

	got, err := hipRunDiffusionSampleKernel(context.Background(), driver, logits, 2, 4, 0.75, 30, []float32{0.25, 0.5})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, len(got))
	core.AssertEqual(t, int32(1), got[0].Sampled)
	core.AssertEqual(t, int32(0), got[0].Greedy)
	core.AssertEqual(t, float32(0.25), got[0].Entropy)
	core.AssertEqual(t, int32(2), got[1].Sampled)
	core.AssertEqual(t, 1, len(base.launches))
	core.AssertEqual(t, hipKernelNameDiffusionSampleProbabilities, base.launches[0].Name)
	core.AssertEqual(t, []uint64{8, 32}, base.copies)
}

func TestHIPDiffusionSampleKernel_Bad_RejectsLogitShape(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	logits, err := hipAllocateByteBuffer(driver, "rocm.hip.DiffusionSampleLaunch", "short logits", 12, 3)
	core.RequireNoError(t, err)
	defer logits.Close()

	_, err = hipRunDiffusionSampleKernel(context.Background(), driver, logits, 1, 4, 1, 30, []float32{0.5})
	core.RequireTrue(t, err != nil)
	core.AssertContains(t, err.Error(), "logit byte count mismatch")
}

type hipDiffusionSampleStubDriver struct {
	*fakeHIPDriver
}

func (driver *hipDiffusionSampleStubDriver) LaunchKernel(config hipKernelLaunchConfig) error {
	if err := driver.fakeHIPDriver.LaunchKernel(config); err != nil || config.Name != hipKernelNameDiffusionSampleProbabilities {
		return err
	}
	rows := binary.LittleEndian.Uint32(config.Args[32:])
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(config.Args[24:]))
	payload, offset, ok := driver.memoryForPointer(outputPointer, int(rows)*hipDiffusionSampleResultBytes)
	if !ok {
		return core.NewError("hip diffusion sample stub output is missing")
	}
	for row := uint32(0); row < rows; row++ {
		base := offset + int(row)*hipDiffusionSampleResultBytes
		binary.LittleEndian.PutUint32(payload[base:], row+1)
		binary.LittleEndian.PutUint32(payload[base+4:], row)
		binary.LittleEndian.PutUint32(payload[base+8:], math.Float32bits(float32(row)+0.25))
		binary.LittleEndian.PutUint32(payload[base+12:], hipDiffusionSampleStatusOK)
	}
	return nil
}
