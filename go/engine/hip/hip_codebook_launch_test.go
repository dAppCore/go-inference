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

func TestHIPCodebookLookupLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipCodebookLookupRequest{
		Codes:    []uint8{2, 0},
		Codebook: []float32{1, 2, 3, 4, 5, 6},
		CodeDim:  2,
	}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.RequireNoError(t, err)
	payload, err := launch.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipCodebookLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipCodebookLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipCodebookLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(buffers.Codes.Pointer()), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(buffers.Codebook.Pointer()), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(payload[24:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[32:]))
	core.AssertEqual(t, uint32(3), binary.LittleEndian.Uint32(payload[36:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[40:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[44:]))
	core.AssertEqual(t, uint32(24), binary.LittleEndian.Uint32(payload[48:]))
	core.AssertEqual(t, uint32(16), binary.LittleEndian.Uint32(payload[52:]))
}

func TestHIPCodebookLookupLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipCodebookLookupRequest{
		Codes:    []uint8{2, 0},
		Codebook: []float32{1, 2, 3, 4, 5, 6},
		CodeDim:  2,
	}
	want, err := rocmReferenceCodebookLookup(req.Codes, req.Codebook, req.CodeDim)
	core.RequireNoError(t, err)

	got, err := hipRunCodebookLookupKernel(context.Background(), driver, req)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameCodebook, driver.launches[0].Name)
	core.AssertEqual(t, hipCodebookLaunchArgsBytes, len(driver.launches[0].Args))
	assertFloat32SlicesNear(t, want, got, 0)
}

func TestHIPCodebookLookupLaunch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	_, err := hipRunCodebookLookupKernel(context.Background(), driver, hipCodebookLookupRequest{
		Codebook: []float32{1, 2},
		CodeDim:  2,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "codes are required")

	_, err = hipRunCodebookLookupKernel(context.Background(), driver, hipCodebookLookupRequest{
		Codes:    []uint8{3},
		Codebook: []float32{1, 2, 3, 4, 5, 6},
		CodeDim:  2,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside codebook size")

	_, err = hipRunCodebookLookupKernel(context.Background(), driver, hipCodebookLookupRequest{
		Codes:    []uint8{0},
		Codebook: []float32{1, float32(math.Inf(-1))},
		CodeDim:  2,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = (hipCodebookLaunchArgs{
		CodePointer:     1,
		CodebookPointer: 2,
		OutputPointer:   3,
		CodeCount:       2,
		CodebookCount:   3,
		CodeDim:         2,
		CodeBytes:       3,
		CodebookBytes:   24,
		OutputBytes:     16,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "code byte count")
}

func TestHIPCodebookLookupLaunchBufferValidation_Bad(t *testing.T) {
	req := hipCodebookLookupRequest{
		Codes:    []uint8{2, 0},
		Codebook: []float32{1, 2, 3, 4, 5, 6},
		CodeDim:  2,
	}
	_, err := req.launchArgs(nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "codebook device buffers are required")

	driver := &fakeHIPDriver{available: true}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.Output.count++
	_, err = req.launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "codebook device buffer shape mismatch")

	_, err = (hipCodebookLaunchArgs{
		CodePointer:     1,
		CodebookPointer: 2,
		OutputPointer:   3,
		CodeCount:       2,
		CodebookCount:   3,
		CodeDim:         2,
		CodeBytes:       2,
		CodebookBytes:   24,
		OutputBytes:     12,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "output byte count")
}

func TestHIPCodebookLookupReadOutputValidation_Bad(t *testing.T) {
	_, err := (*hipCodebookDeviceBuffers)(nil).ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "codebook output buffer is required")

	driver := &fakeHIPDriver{available: true}
	req := hipCodebookLookupRequest{
		Codes:    []uint8{2, 0},
		Codebook: []float32{1, 2, 3, 4, 5, 6},
		CodeDim:  2,
	}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.Output.sizeBytes++
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "codebook output byte count mismatch")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	payload, err := hipFloat32Payload([]float32{1, float32(math.NaN()), 3, 4})
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
	core.AssertContains(t, err.Error(), "copy codebook output")
}

func BenchmarkHIPCodebookLookupLaunch_Codes512Dim64(b *testing.B) {
	req := hipCodebookLookupRequest{
		Codes:    codebookBenchmarkCodes(512, 128),
		Codebook: codebookBenchmarkTable(128, 64),
		CodeDim:  64,
	}
	driver := &fakeHIPDriver{available: true}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got, err := hipRunCodebookLookupKernel(context.Background(), driver, req)
		if err != nil {
			b.Fatalf("run codebook fixture: %v", err)
		}
		if len(got) != len(req.Codes)*req.CodeDim {
			b.Fatalf("output length = %d, want %d", len(got), len(req.Codes)*req.CodeDim)
		}
	}
}

func BenchmarkHIPCodebookLookupLaunchPrepared_Codes512Dim64(b *testing.B) {
	req := hipCodebookLookupRequest{
		Codes:    codebookBenchmarkCodes(512, 128),
		Codebook: codebookBenchmarkTable(128, 64),
		CodeDim:  64,
	}
	driver := &fakeHIPDriver{available: true, skipLaunchRecording: true, copies: make([]uint64, 0, 8)}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		b.Fatalf("prepare codebook fixture buffers: %v", err)
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		b.Fatalf("prepare codebook fixture launch args: %v", err)
	}
	launchBytes, err := launch.BinaryInto(make([]byte, hipCodebookLaunchArgsBytes))
	if err != nil {
		b.Fatalf("encode codebook fixture launch args: %v", err)
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameCodebook, launchBytes, len(req.Codes)*req.CodeDim)
	if err != nil {
		b.Fatalf("prepare codebook fixture launch config: %v", err)
	}
	outputPayload := make([]byte, len(req.Codes)*req.CodeDim*4)
	outputValues := make([]float32, len(req.Codes)*req.CodeDim)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipLaunchKernel(driver, config); err != nil {
			b.Fatalf("launch codebook fixture: %v", err)
		}
		got, err := buffers.ReadOutputInto(outputValues, outputPayload)
		if err != nil {
			b.Fatalf("read codebook fixture: %v", err)
		}
		if len(got) != len(req.Codes)*req.CodeDim {
			b.Fatalf("output length = %d, want %d", len(got), len(req.Codes)*req.CodeDim)
		}
		driver.copies = driver.copies[:0]
	}
}

func codebookBenchmarkCodes(count, codebookSize int) []uint8 {
	codes := make([]uint8, count)
	for i := range codes {
		codes[i] = uint8((i * 17) % codebookSize)
	}
	return codes
}

func codebookBenchmarkTable(codebookSize, codeDim int) []float32 {
	values := make([]float32, codebookSize*codeDim)
	for i := range values {
		values[i] = float32(i%codeDim) / float32(codeDim)
	}
	return values
}
