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

func TestHIPLoRAProjectionLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipLoRAProjectionRequest{
		Input:      []float32{2, 3},
		BaseWeight: []float32{1, 0, 0, 1},
		LoRAA:      []float32{1, 1},
		LoRAB:      []float32{2, -1},
		Rows:       2,
		Cols:       2,
		Rank:       1,
		Alpha:      0.5,
		Bias:       []float32{0.25, -0.5},
	}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.RequireNoError(t, err)
	payload, err := launch.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipLoRALaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipLoRALaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipLoRALaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(buffers.Input.Pointer()), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(buffers.BaseWeight.Pointer()), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint64(buffers.LoRAA.Pointer()), binary.LittleEndian.Uint64(payload[24:]))
	core.AssertEqual(t, uint64(buffers.LoRAB.Pointer()), binary.LittleEndian.Uint64(payload[32:]))
	core.AssertEqual(t, uint64(buffers.Bias.Pointer()), binary.LittleEndian.Uint64(payload[40:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(payload[48:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[56:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[60:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[64:]))
	core.AssertEqual(t, uint32(1), binary.LittleEndian.Uint32(payload[68:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[72:]))
	core.AssertEqual(t, uint32(16), binary.LittleEndian.Uint32(payload[76:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[80:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[84:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[88:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[92:]))
	core.AssertEqual(t, hipLoRALaunchFlagBias, binary.LittleEndian.Uint32(payload[100:]))
}

func TestHIPLoRAProjectionLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipLoRAProjectionRequest{
		Input:      []float32{2, 3},
		BaseWeight: []float32{1, 0, 0, 1},
		LoRAA:      []float32{1, 1},
		LoRAB:      []float32{2, -1},
		Rows:       2,
		Cols:       2,
		Rank:       1,
		Alpha:      0.5,
		Bias:       []float32{0.25, -0.5},
	}
	want, err := rocmReferenceLoRAProjection(req.Input, req.BaseWeight, req.LoRAA, req.LoRAB, req.Rows, req.Cols, req.Rank, req.Alpha, req.Bias)
	core.RequireNoError(t, err)

	got, err := hipRunLoRAProjectionKernel(context.Background(), driver, req)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameLoRA, driver.launches[0].Name)
	core.AssertEqual(t, hipLoRALaunchArgsBytes, len(driver.launches[0].Args))
	assertFloat32SlicesNear(t, want, got, 0)
}

func TestHIPLoRAProjectionLaunch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	_, err := hipRunLoRAProjectionKernel(context.Background(), driver, hipLoRAProjectionRequest{
		Input:      []float32{1},
		BaseWeight: []float32{1},
		LoRAA:      []float32{1},
		LoRAB:      []float32{1},
		Rows:       1,
		Cols:       1,
		Rank:       0,
		Alpha:      1,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rank must be positive")

	_, err = hipRunLoRAProjectionKernel(context.Background(), driver, hipLoRAProjectionRequest{
		Input:      []float32{1},
		BaseWeight: []float32{1},
		LoRAA:      []float32{1},
		LoRAB:      []float32{1},
		Rows:       1,
		Cols:       1,
		Rank:       1,
		Alpha:      0,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "alpha must be positive")

	_, err = (hipLoRALaunchArgs{
		InputPointer:      1,
		BaseWeightPointer: 2,
		LoRAAPointer:      3,
		LoRABPointer:      4,
		OutputPointer:     5,
		InputCount:        2,
		Rows:              2,
		Cols:              2,
		Rank:              1,
		InputBytes:        4,
		BaseWeightBytes:   16,
		LoRAABytes:        8,
		LoRABBytes:        8,
		OutputBytes:       8,
		Alpha:             1,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input byte count")

	_, err = (hipLoRALaunchArgs{
		InputPointer:      1,
		BaseWeightPointer: 2,
		LoRAAPointer:      3,
		LoRABPointer:      4,
		OutputPointer:     5,
		InputCount:        2,
		Rows:              2,
		Cols:              2,
		Rank:              1,
		InputBytes:        8,
		BaseWeightBytes:   16,
		LoRAABytes:        8,
		LoRABBytes:        8,
		OutputBytes:       8,
		Alpha:             1,
	}).BinaryInto(make([]byte, hipLoRALaunchArgsBytes-1))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "launch arg payload buffer is too small")
}

func TestHIPLoRAProjectionReadOutputValidation_Bad(t *testing.T) {
	_, err := (*hipLoRADeviceBuffers)(nil).ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "LoRA output buffer is required")

	req := hipLoRAProjectionRequest{
		Input:      []float32{2, 3},
		BaseWeight: []float32{1, 0, 0, 1},
		LoRAA:      []float32{1, 1},
		LoRAB:      []float32{2, -1},
		Rows:       2,
		Cols:       2,
		Rank:       1,
		Alpha:      0.5,
	}
	driver := &fakeHIPDriver{available: true}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.Output.sizeBytes++
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "LoRA output byte count mismatch")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	payload, err := hipFloat32Payload([]float32{0, float32(math.Inf(1))})
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
	core.AssertContains(t, err.Error(), "copy LoRA output")
}

func BenchmarkHIPLoRAProjectionLaunch_Rows128Cols256Rank8(b *testing.B) {
	req := loraBenchmarkProjectionRequest(128, 256, 8)
	driver := &fakeHIPDriver{available: true}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got, err := hipRunLoRAProjectionKernel(context.Background(), driver, req)
		if err != nil {
			b.Fatalf("run LoRA fixture: %v", err)
		}
		if len(got) != req.Rows {
			b.Fatalf("output rows = %d, want %d", len(got), req.Rows)
		}
	}
}

func BenchmarkHIPLoRAProjectionLaunchPrepared_Rows128Cols256Rank8(b *testing.B) {
	req := loraBenchmarkProjectionRequest(128, 256, 8)
	driver := &fakeHIPDriver{available: true, skipLaunchRecording: true, copies: make([]uint64, 0, 8)}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		b.Fatalf("prepare LoRA fixture buffers: %v", err)
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		b.Fatalf("prepare LoRA fixture launch args: %v", err)
	}
	launchBytes, err := launch.BinaryInto(make([]byte, hipLoRALaunchArgsBytes))
	if err != nil {
		b.Fatalf("encode LoRA fixture launch args: %v", err)
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameLoRA, launchBytes, req.Rows)
	if err != nil {
		b.Fatalf("prepare LoRA fixture launch config: %v", err)
	}
	outputPayload := make([]byte, req.Rows*4)
	outputValues := make([]float32, req.Rows)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipLaunchKernel(driver, config); err != nil {
			b.Fatalf("launch LoRA fixture: %v", err)
		}
		got, err := buffers.ReadOutputInto(outputValues, outputPayload)
		if err != nil {
			b.Fatalf("read LoRA fixture: %v", err)
		}
		if len(got) != req.Rows {
			b.Fatalf("output rows = %d, want %d", len(got), req.Rows)
		}
		driver.copies = driver.copies[:0]
	}
}

func loraBenchmarkProjectionRequest(rows, cols, rank int) hipLoRAProjectionRequest {
	input := make([]float32, cols)
	for i := range input {
		input[i] = float32(math.Sin(float64(i)*0.017) + math.Cos(float64(i)*0.041))
	}
	baseWeight := make([]float32, rows*cols)
	for i := range baseWeight {
		baseWeight[i] = float32(math.Sin(float64(i)*0.003) * 0.02)
	}
	loraA := make([]float32, rank*cols)
	for i := range loraA {
		loraA[i] = float32(math.Cos(float64(i)*0.007) * 0.01)
	}
	loraB := make([]float32, rows*rank)
	for i := range loraB {
		loraB[i] = float32(math.Sin(float64(i)*0.011) * 0.01)
	}
	bias := make([]float32, rows)
	for i := range bias {
		bias[i] = float32(math.Cos(float64(i)*0.019) * 0.001)
	}
	return hipLoRAProjectionRequest{
		Input:      input,
		BaseWeight: baseWeight,
		LoRAA:      loraA,
		LoRAB:      loraB,
		Rows:       rows,
		Cols:       cols,
		Rank:       rank,
		Alpha:      8,
		Bias:       bias,
	}
}
