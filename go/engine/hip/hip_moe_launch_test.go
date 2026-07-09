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

func TestHIPMoERouterLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipMoERouterRequest{Logits: []float32{0.1, 2, 1, -1}, TopK: 2, Layer: 7}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.RequireNoError(t, err)
	payload, err := launch.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipMoERouterLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipMoERouterLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipMoERouterLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(buffers.Logits.Pointer()), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(buffers.IDs.Pointer()), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint64(buffers.Probs.Pointer()), binary.LittleEndian.Uint64(payload[24:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(payload[32:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[36:]))
	core.AssertEqual(t, uint32(16), binary.LittleEndian.Uint32(payload[40:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[44:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[48:]))
	core.AssertEqual(t, uint32(7), binary.LittleEndian.Uint32(payload[52:]))
	core.AssertEqual(t, uint64(buffers.Status.Pointer()), binary.LittleEndian.Uint64(payload[56:]))
}

func TestHIPMoERouterLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipMoERouterRequest{Logits: []float32{0.1, 2, 1, -1}, TopK: 2, Layer: 7}
	want, err := rocmReferenceRouteExperts(req.Logits, req.TopK, req.Layer, nil)
	core.RequireNoError(t, err)

	got, err := hipRunMoERouterKernel(context.Background(), driver, req)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameMoERouter, driver.launches[0].Name)
	core.AssertEqual(t, hipMoERouterLaunchArgsBytes, len(driver.launches[0].Args))
	core.AssertEqual(t, req.Layer, got.Layer)
	core.AssertEqual(t, hipMoERouterLaunchStatusOK, got.Status)
	core.AssertEqual(t, len(want), len(got.Routes))
	for index := range want {
		core.AssertEqual(t, want[index].ID, got.Routes[index].ID)
		assertFloat32Near(t, want[index].Score, got.Routes[index].Score)
		assertFloat32Near(t, want[index].Prob, got.Routes[index].Prob)
	}
}

func TestHIPMoERouterLaunch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	_, err := hipRunMoERouterKernel(context.Background(), driver, hipMoERouterRequest{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "router logits")

	_, err = hipRunMoERouterKernel(context.Background(), driver, hipMoERouterRequest{Logits: []float32{1}, TopK: 2})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "top-k")

	_, err = hipRunMoERouterKernel(context.Background(), driver, hipMoERouterRequest{Logits: []float32{1, float32(math.NaN())}, TopK: 1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = (hipMoERouterLaunchArgs{
		LogitPointer:  1,
		IDPointer:     2,
		ProbPointer:   3,
		StatusPointer: 4,
		ExpertCount:   4,
		TopK:          2,
		Layer:         0,
		LogitBytes:    12,
		IDBytes:       8,
		ProbBytes:     8,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "logit byte count")

	_, err = (hipMoERouterLaunchArgs{
		LogitPointer:  1,
		IDPointer:     2,
		ProbPointer:   3,
		StatusPointer: 4,
		ExpertCount:   2,
		TopK:          3,
		LogitBytes:    8,
		IDBytes:       12,
		ProbBytes:     12,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "top-k")

	_, err = (hipMoERouterLaunchArgs{
		LogitPointer: 1,
		IDPointer:    2,
		ProbPointer:  3,
		ExpertCount:  2,
		TopK:         1,
		LogitBytes:   8,
		IDBytes:      4,
		ProbBytes:    4,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "router status pointer")

	_, err = (hipMoERouterLaunchArgs{
		LogitPointer:  1,
		IDPointer:     2,
		ProbPointer:   3,
		StatusPointer: 4,
		ExpertCount:   2,
		TopK:          1,
		LogitBytes:    8,
		IDBytes:       4,
		ProbBytes:     4,
	}).BinaryInto(make([]byte, hipMoERouterLaunchArgsBytes-1))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "launch arg payload buffer is too small")
}

func TestHIPMoERouterLaunchBufferValidation_Bad(t *testing.T) {
	req := hipMoERouterRequest{Logits: []float32{0.1, 2, 1, -1}, TopK: 2, Layer: 7}
	_, err := req.launchArgs(nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "router device buffers are required")

	driver := &fakeHIPDriver{available: true}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.IDs.count++
	_, err = req.launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "router device buffer shape mismatch")

	buffers.IDs.count--
	buffers.Status.sizeBytes++
	_, err = req.launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "router device buffer shape mismatch")
}

func TestHIPMoERouterReadOutputValidation_Bad(t *testing.T) {
	_, err := (*hipMoERouterDeviceBuffers)(nil).ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "router output buffers are required")

	driver := &fakeHIPDriver{available: true}
	req := hipMoERouterRequest{Logits: []float32{0.1, 2, 1, -1}, TopK: 2, Layer: 7}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()

	_, err = buffers.ReadOutput()

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "router status marker mismatch")

	for _, tt := range []struct {
		name  string
		ids   []int32
		probs []float32
		want  string
	}{
		{
			name:  "expert id",
			ids:   []int32{1, 9},
			probs: []float32{0.5, 0.25},
			want:  "outside expert count",
		},
		{
			name:  "probability",
			ids:   []int32{1, 2},
			probs: []float32{0.5, float32(math.NaN())},
			want:  "router probability",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			driver := &fakeHIPDriver{available: true}
			req := hipMoERouterRequest{Logits: []float32{0.1, 2, 1, -1}, TopK: 2, Layer: 7}
			buffers, err := req.deviceBuffers(driver)
			core.RequireNoError(t, err)
			defer buffers.Close()
			idPayload := make([]byte, buffers.IDs.SizeBytes())
			for index, id := range tt.ids {
				binary.LittleEndian.PutUint32(idPayload[index*4:], uint32(id))
			}
			probPayload := make([]byte, buffers.Probs.SizeBytes())
			for index, prob := range tt.probs {
				binary.LittleEndian.PutUint32(probPayload[index*4:], math.Float32bits(prob))
			}
			statusPayload := make([]byte, buffers.Status.SizeBytes())
			binary.LittleEndian.PutUint32(statusPayload, hipMoERouterLaunchStatusOK)
			core.RequireNoError(t, driver.CopyHostToDevice(buffers.IDs.Pointer(), idPayload))
			core.RequireNoError(t, driver.CopyHostToDevice(buffers.Probs.Pointer(), probPayload))
			core.RequireNoError(t, driver.CopyHostToDevice(buffers.Status.Pointer(), statusPayload))

			_, err = buffers.ReadOutput()

			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}

	for _, tt := range []struct {
		name string
		want string
	}{
		{name: "id", want: "copy router id output"},
		{name: "probability", want: "copy router probability output"},
		{name: "status", want: "copy router status"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			driver := &fakeHIPDriver{available: true}
			req := hipMoERouterRequest{Logits: []float32{0.1, 2, 1, -1}, TopK: 2, Layer: 7}
			buffers, err := req.deviceBuffers(driver)
			core.RequireNoError(t, err)
			defer buffers.Close()
			driver.copyErr = core.NewError("copy failed")
			driver.copyErrAt = len(driver.copies) + 1
			switch tt.name {
			case "probability":
				driver.copyErrAt++
			case "status":
				driver.copyErrAt += 2
			}

			_, err = buffers.ReadOutput()

			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}
}

func TestHIPMoELazyExpertLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipMoELazyExpertRequest{ExpertIDs: []int32{3, 1}, TotalExperts: 5}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.RequireNoError(t, err)
	payload, err := launch.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipMoELazyLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipMoELazyLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipMoELazyLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(buffers.IDs.Pointer()), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(buffers.Resident.Pointer()), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[24:]))
	core.AssertEqual(t, uint32(5), binary.LittleEndian.Uint32(payload[28:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[32:]))
	core.AssertEqual(t, uint32(5), binary.LittleEndian.Uint32(payload[36:]))
}

func TestHIPMoELazyExpertLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipMoELazyExpertRequest{ExpertIDs: []int32{3, 1}, TotalExperts: 5}
	want, err := rocmReferenceLazyExpertResidency([]rocmExpertRoute{{ID: 3}, {ID: 1}}, req.TotalExperts)
	core.RequireNoError(t, err)

	got, err := hipRunMoELazyExpertKernel(context.Background(), driver, req)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameMoELazy, driver.launches[0].Name)
	core.AssertEqual(t, hipMoELazyLaunchArgsBytes, len(driver.launches[0].Args))
	core.AssertEqual(t, want, got.Resident)
}

func TestHIPMoELazyExpertLaunch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	_, err := hipRunMoELazyExpertKernel(context.Background(), driver, hipMoELazyExpertRequest{ExpertIDs: []int32{1}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "expert count")

	_, err = hipRunMoELazyExpertKernel(context.Background(), driver, hipMoELazyExpertRequest{ExpertIDs: []int32{5}, TotalExperts: 5})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside expert count")

	_, err = (hipMoELazyExpertLaunchArgs{
		IDPointer:       1,
		ResidentPointer: 2,
		SelectedCount:   2,
		TotalExperts:    5,
		IDBytes:         4,
		ResidentBytes:   5,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "expert ID byte count")

	_, err = (hipMoELazyExpertLaunchArgs{
		IDPointer:       1,
		ResidentPointer: 2,
		SelectedCount:   2,
		TotalExperts:    5,
		IDBytes:         8,
		ResidentBytes:   5,
	}).BinaryInto(make([]byte, hipMoELazyLaunchArgsBytes-1))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "launch arg payload buffer is too small")
}

func TestHIPMoELazyExpertLaunchBufferValidation_Bad(t *testing.T) {
	req := hipMoELazyExpertRequest{ExpertIDs: []int32{3, 1}, TotalExperts: 5}
	_, err := req.launchArgs(nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "lazy expert device buffers are required")

	driver := &fakeHIPDriver{available: true}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.Resident.count++
	_, err = req.launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "lazy expert device buffer shape mismatch")
}

func TestHIPMoELazyExpertReadOutputValidation_Bad(t *testing.T) {
	_, err := (*hipMoELazyExpertDeviceBuffers)(nil).ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "resident expert output buffer is required")

	driver := &fakeHIPDriver{available: true}
	req := hipMoELazyExpertRequest{ExpertIDs: []int32{3, 1}, TotalExperts: 5}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.Resident.sizeBytes++
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "resident expert output byte count mismatch")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	core.RequireNoError(t, driver.CopyHostToDevice(buffers.Resident.Pointer(), []byte{0, 1, 2, 0, 1}))
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "binary flags")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	driver.copyErr = core.NewError("copy failed")

	_, err = buffers.ReadOutput()

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy resident expert output")
}

func BenchmarkHIPMoERouterLaunch_Top2Of128(b *testing.B) {
	logits := make([]float32, 128)
	for i := range logits {
		logits[i] = float32(math.Sin(float64(i)*0.11) + math.Cos(float64(i)*0.03))
	}
	req := hipMoERouterRequest{Logits: logits, TopK: 2, Layer: 7}
	driver := &fakeHIPDriver{available: true}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got, err := hipRunMoERouterKernel(context.Background(), driver, req)
		if err != nil {
			b.Fatalf("run MoE router fixture: %v", err)
		}
		if len(got.Routes) != req.TopK || got.Status != hipMoERouterLaunchStatusOK {
			b.Fatalf("router result = %+v, want top-k status OK", got)
		}
	}
}

func BenchmarkHIPMoERouterLaunchPrepared_Top2Of128(b *testing.B) {
	logits := make([]float32, 128)
	for i := range logits {
		logits[i] = float32(math.Sin(float64(i)*0.11) + math.Cos(float64(i)*0.03))
	}
	req := hipMoERouterRequest{Logits: logits, TopK: 2, Layer: 7}
	driver := &fakeHIPDriver{available: true, skipLaunchRecording: true, copies: make([]uint64, 0, 8)}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		b.Fatalf("prepare MoE router buffers: %v", err)
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		b.Fatalf("prepare MoE router launch args: %v", err)
	}
	launchBytes, err := launch.BinaryInto(make([]byte, hipMoERouterLaunchArgsBytes))
	if err != nil {
		b.Fatalf("encode MoE router launch args: %v", err)
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameMoERouter, launchBytes, 1)
	if err != nil {
		b.Fatalf("prepare MoE router launch config: %v", err)
	}
	routes := make([]rocmExpertRoute, req.TopK)
	idPayload := make([]byte, req.TopK*4)
	probPayload := make([]byte, req.TopK*4)
	statusPayload := make([]byte, 4)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipLaunchKernel(driver, config); err != nil {
			b.Fatalf("launch MoE router fixture: %v", err)
		}
		got, err := buffers.ReadOutputInto(routes, idPayload, probPayload, statusPayload)
		if err != nil {
			b.Fatalf("read MoE router fixture: %v", err)
		}
		if len(got.Routes) != req.TopK || got.Status != hipMoERouterLaunchStatusOK {
			b.Fatalf("router result = %+v, want top-k status OK", got)
		}
		driver.copies = driver.copies[:0]
	}
}

func BenchmarkHIPMoELazyExpertLaunch_Top2Of128(b *testing.B) {
	req := hipMoELazyExpertRequest{ExpertIDs: []int32{37, 5}, TotalExperts: 128}
	driver := &fakeHIPDriver{available: true}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got, err := hipRunMoELazyExpertKernel(context.Background(), driver, req)
		if err != nil {
			b.Fatalf("run MoE lazy expert fixture: %v", err)
		}
		if len(got.Resident) != req.TotalExperts || !got.Resident[37] || !got.Resident[5] {
			b.Fatalf("resident result = %+v, want selected experts resident", got.Resident)
		}
	}
}

func BenchmarkHIPMoELazyExpertLaunchPrepared_Top2Of128(b *testing.B) {
	req := hipMoELazyExpertRequest{ExpertIDs: []int32{37, 5}, TotalExperts: 128}
	driver := &fakeHIPDriver{available: true, skipLaunchRecording: true, copies: make([]uint64, 0, 8)}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		b.Fatalf("prepare MoE lazy expert buffers: %v", err)
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		b.Fatalf("prepare MoE lazy expert launch args: %v", err)
	}
	launchBytes, err := launch.BinaryInto(make([]byte, hipMoELazyLaunchArgsBytes))
	if err != nil {
		b.Fatalf("encode MoE lazy expert launch args: %v", err)
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameMoELazy, launchBytes, req.TotalExperts)
	if err != nil {
		b.Fatalf("prepare MoE lazy expert launch config: %v", err)
	}
	resident := make([]bool, req.TotalExperts)
	payload := make([]byte, req.TotalExperts)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipLaunchKernel(driver, config); err != nil {
			b.Fatalf("launch MoE lazy expert fixture: %v", err)
		}
		got, err := buffers.ReadOutputInto(resident, payload)
		if err != nil {
			b.Fatalf("read MoE lazy expert fixture: %v", err)
		}
		if len(got.Resident) != req.TotalExperts || !got.Resident[37] || !got.Resident[5] {
			b.Fatalf("resident result = %+v, want selected experts resident", got.Resident)
		}
		driver.copies = driver.copies[:0]
	}
}
