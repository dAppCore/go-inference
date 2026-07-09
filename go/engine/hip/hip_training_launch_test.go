// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestHIPTrainingCrossEntropyLossLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipCrossEntropyLossRequest{
		Logits:  []float32{2, 0, 0, 2},
		Targets: []int32{0, 1},
		Batch:   2,
		Vocab:   2,
	}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.RequireNoError(t, err)
	payload, err := launch.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipCrossEntropyLossLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipCrossEntropyLossLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipCrossEntropyLossLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(buffers.Logits.Pointer()), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(buffers.Targets.Pointer()), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(payload[24:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[32:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[36:]))
	core.AssertEqual(t, uint32(16), binary.LittleEndian.Uint32(payload[40:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[44:]))
	core.AssertEqual(t, uint32(hipCrossEntropyLossOutputBytes), binary.LittleEndian.Uint32(payload[48:]))

	got, err := hipRunCrossEntropyLossKernel(context.Background(), driver, req)
	core.RequireNoError(t, err)
	assertFloat64Near(t, 0.1269, got.Loss, 0.0001)
	assertFloat64Near(t, 1.1353, got.Perplexity, 0.0001)
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameCrossEntropy, driver.launches[0].Name)
}

func TestHIPTrainingCrossEntropyLossLaunch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	_, err := (hipCrossEntropyLossRequest{
		Logits:  []float32{1, 2},
		Targets: []int32{0},
		Batch:   0,
		Vocab:   2,
	}).deviceBuffers(driver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "positive")

	_, err = (hipCrossEntropyLossRequest{
		Logits:  []float32{1, float32(math.NaN())},
		Targets: []int32{0},
		Batch:   1,
		Vocab:   2,
	}).deviceBuffers(driver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = (hipCrossEntropyLossRequest{
		Logits:  []float32{1, 2},
		Targets: []int32{3},
		Batch:   1,
		Vocab:   2,
	}).deviceBuffers(driver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside vocabulary")

	req := hipCrossEntropyLossRequest{Logits: []float32{1, 2}, Targets: []int32{0}, Batch: 1, Vocab: 2}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	_, err = (hipCrossEntropyLossRequest{Logits: []float32{1, 2, 3, 4}, Targets: []int32{0, 1}, Batch: 2, Vocab: 2}).launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")

	_, err = (hipCrossEntropyLossLaunchArgs{
		LogitPointer:  1,
		TargetPointer: 2,
		OutputPointer: 3,
		Batch:         2,
		Vocab:         2,
		LogitBytes:    8,
		TargetBytes:   8,
		OutputBytes:   hipCrossEntropyLossOutputBytes,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "logit byte count")
}

func TestHIPTrainingCrossEntropyLossReadOutputValidation_Bad(t *testing.T) {
	_, err := (*hipCrossEntropyLossDeviceBuffers)(nil).ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "cross entropy output buffer is required")

	req := hipCrossEntropyLossRequest{Logits: []float32{1, 2}, Targets: []int32{1}, Batch: 1, Vocab: 2}
	driver := &fakeHIPDriver{available: true}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.Output.sizeBytes++
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "cross entropy output byte count mismatch")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	payload := make([]byte, hipCrossEntropyLossOutputBytes)
	binary.LittleEndian.PutUint64(payload[0:], math.Float64bits(math.NaN()))
	binary.LittleEndian.PutUint64(payload[8:], math.Float64bits(1))
	core.RequireNoError(t, driver.CopyHostToDevice(buffers.Output.Pointer(), payload))
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite and valid")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	driver.copyErr = core.NewError("copy failed")
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy cross entropy output")
}

func TestHIPTrainingDistillationKLLossLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipDistillationKLLossRequest{
		StudentLogits: []float32{1, 0},
		TeacherLogits: []float32{2, 0},
		Batch:         1,
		Vocab:         2,
		Temperature:   1,
	}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.RequireNoError(t, err)
	payload, err := launch.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipDistillationKLLossLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipDistillationKLLossLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipDistillationKLLossLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(buffers.StudentLogits.Pointer()), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(buffers.TeacherLogits.Pointer()), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(payload[24:]))
	core.AssertEqual(t, uint32(1), binary.LittleEndian.Uint32(payload[32:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[36:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[40:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[44:]))
	core.AssertEqual(t, uint32(hipDistillationKLLossOutputBytes), binary.LittleEndian.Uint32(payload[48:]))
	core.AssertEqual(t, math.Float64bits(1), binary.LittleEndian.Uint64(payload[56:]))

	got, err := hipRunDistillationKLLossKernel(context.Background(), driver, req)
	core.RequireNoError(t, err)
	assertFloat64Near(t, 0.0671, got.KL, 0.0001)
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameDistillKL, driver.launches[0].Name)
}

func TestHIPTrainingDistillationKLLossLaunch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	_, err := (hipDistillationKLLossRequest{
		StudentLogits: []float32{1, 2},
		TeacherLogits: []float32{1, 2},
		Batch:         1,
		Vocab:         0,
		Temperature:   1,
	}).deviceBuffers(driver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "positive")

	_, err = (hipDistillationKLLossRequest{
		StudentLogits: []float32{1, 2},
		TeacherLogits: []float32{1, 2},
		Batch:         1,
		Vocab:         2,
		Temperature:   math.Inf(1),
	}).deviceBuffers(driver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "temperature")

	_, err = (hipDistillationKLLossRequest{
		StudentLogits: []float32{1, 2},
		TeacherLogits: []float32{1, float32(math.NaN())},
		Batch:         1,
		Vocab:         2,
		Temperature:   1,
	}).deviceBuffers(driver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	req := hipDistillationKLLossRequest{StudentLogits: []float32{1, 2}, TeacherLogits: []float32{1, 2}, Batch: 1, Vocab: 2, Temperature: 1}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	_, err = (hipDistillationKLLossRequest{
		StudentLogits: []float32{1, 2, 3, 4},
		TeacherLogits: []float32{1, 2, 3, 4},
		Batch:         2,
		Vocab:         2,
		Temperature:   1,
	}).launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")

	_, err = (hipDistillationKLLossLaunchArgs{
		StudentPointer: 1,
		TeacherPointer: 2,
		OutputPointer:  3,
		Batch:          1,
		Vocab:          2,
		StudentBytes:   4,
		TeacherBytes:   8,
		OutputBytes:    hipDistillationKLLossOutputBytes,
		Temperature:    1,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "student byte count")
}

func TestHIPTrainingDistillationKLLossReadOutputValidation_Bad(t *testing.T) {
	_, err := (*hipDistillationKLLossDeviceBuffers)(nil).ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "distillation output buffer is required")

	req := hipDistillationKLLossRequest{StudentLogits: []float32{1, 0}, TeacherLogits: []float32{2, 0}, Batch: 1, Vocab: 2, Temperature: 1}
	driver := &fakeHIPDriver{available: true}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.Output.sizeBytes++
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "distillation output byte count mismatch")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	payload := make([]byte, hipDistillationKLLossOutputBytes)
	binary.LittleEndian.PutUint64(payload[0:], math.Float64bits(math.Inf(1)))
	core.RequireNoError(t, driver.CopyHostToDevice(buffers.Output.Pointer(), payload))
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite and valid")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	driver.copyErr = core.NewError("copy failed")
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy distillation output")
}

func TestHIPTrainingGRPOAdvantageLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipGRPOAdvantageRequest{Rewards: []float64{1, 2, 3}, Count: 3}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.RequireNoError(t, err)
	payload, err := launch.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipGRPOAdvantageLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipGRPOAdvantageLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipGRPOAdvantageLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(buffers.Rewards.Pointer()), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint32(3), binary.LittleEndian.Uint32(payload[24:]))
	core.AssertEqual(t, uint32(24), binary.LittleEndian.Uint32(payload[28:]))
	core.AssertEqual(t, uint32(24), binary.LittleEndian.Uint32(payload[32:]))

	got, err := hipRunGRPOAdvantageKernel(context.Background(), driver, req)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameGRPOAdvantage, driver.launches[0].Name)
	assertFloat64Near(t, -1.2247, got[0], 0.0001)
	assertFloat64Near(t, 0, got[1], 0.0001)
	assertFloat64Near(t, 1.2247, got[2], 0.0001)

	zeroVariance, err := hipRunGRPOAdvantageKernel(context.Background(), &fakeHIPDriver{available: true}, hipGRPOAdvantageRequest{Rewards: []float64{5, 5}, Count: 2})
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float64{0, 0}, zeroVariance)
}

func TestHIPTrainingGRPOAdvantageLaunch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	_, err := (hipGRPOAdvantageRequest{Rewards: []float64{1}, Count: 0}).deviceBuffers(driver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "positive")

	_, err = (hipGRPOAdvantageRequest{Rewards: []float64{1, 2}, Count: 1}).deviceBuffers(driver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "length")

	_, err = (hipGRPOAdvantageRequest{Rewards: []float64{1, math.Inf(1)}, Count: 2}).deviceBuffers(driver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	req := hipGRPOAdvantageRequest{Rewards: []float64{1, 2}, Count: 2}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	_, err = (hipGRPOAdvantageRequest{Rewards: []float64{1, 2, 3}, Count: 3}).launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")

	_, err = (hipGRPOAdvantageLaunchArgs{
		RewardPointer: 1,
		OutputPointer: 2,
		Count:         2,
		RewardBytes:   8,
		OutputBytes:   16,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "reward byte count")
}

func TestHIPTrainingGRPOAdvantageReadOutputValidation_Bad(t *testing.T) {
	_, err := (*hipGRPOAdvantageDeviceBuffers)(nil).ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "GRPO advantage output buffer is required")

	req := hipGRPOAdvantageRequest{Rewards: []float64{1, 2}, Count: 2}
	driver := &fakeHIPDriver{available: true}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.Output.sizeBytes++
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "GRPO advantage output byte count mismatch")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	payload := make([]byte, 16)
	binary.LittleEndian.PutUint64(payload[0:], math.Float64bits(0))
	binary.LittleEndian.PutUint64(payload[8:], math.Float64bits(math.NaN()))
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
	core.AssertContains(t, err.Error(), "copy GRPO advantage output")
}

func TestHIPTrainingLoadedModelDistillationKLLossHook_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	model := &hipLoadedModel{driver: driver, kernels: fakeLinkedHIPKernelSet{}}

	if _, ok := any(model).(inference.DistillTrainer); ok {
		t.Fatalf("hipLoadedModel unexpectedly implements public DistillTrainer")
	}
	got, ok, err := model.RunDistillationKLLoss(context.Background(), [][]float32{{1, 0}}, [][]float32{{2, 0}}, 1)
	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	assertFloat64Near(t, 0.0671, got.KL, 0.0001)
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameDistillKL, driver.launches[0].Name)
}

func TestHIPTrainingLoadedModelDistillationKLLossHook_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	model := &hipLoadedModel{driver: driver, kernels: newDefaultHIPKernelSet()}

	got, ok, err := model.RunDistillationKLLoss(context.Background(), [][]float32{{1, 0}}, [][]float32{{2, 0}}, 1)
	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
	core.AssertEqual(t, hipDistillationKLLossResult{}, got)

	model = &hipLoadedModel{driver: driver, kernels: fakeLinkedHIPKernelSet{}}
	_, ok, err = model.RunDistillationKLLoss(context.Background(), [][]float32{{1, 0}}, [][]float32{{2, 0}}, -1)
	core.AssertError(t, err)
	core.AssertTrue(t, ok)
	core.AssertContains(t, err.Error(), "temperature must be positive and finite")

	_, ok, err = model.RunDistillationKLLoss(context.Background(), [][]float32{{1, 0}, {1}}, [][]float32{{2, 0}, {2, 0}}, 1)
	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
}

func TestHIPTrainingLoadedModelGRPOAdvantageHook_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	model := &hipLoadedModel{driver: driver, kernels: fakeLinkedHIPKernelSet{}}

	if _, ok := any(model).(inference.GRPOTrainer); ok {
		t.Fatalf("hipLoadedModel unexpectedly implements public GRPOTrainer")
	}
	got, ok, err := model.RunGRPOAdvantage(context.Background(), []float64{1, 2, 3})
	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, 3, len(got))
	assertFloat64Near(t, -1.2247, got[0], 0.0001)
	assertFloat64Near(t, 0, got[1], 0.0001)
	assertFloat64Near(t, 1.2247, got[2], 0.0001)
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameGRPOAdvantage, driver.launches[0].Name)
}

func TestHIPTrainingLoadedModelGRPOAdvantageHook_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	model := &hipLoadedModel{driver: driver, kernels: newDefaultHIPKernelSet()}

	got, ok, err := model.RunGRPOAdvantage(context.Background(), []float64{1, 2, 3})
	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
	core.AssertEqual(t, []float64(nil), got)

	model = &hipLoadedModel{driver: driver, kernels: fakeLinkedHIPKernelSet{}}
	_, ok, err = model.RunGRPOAdvantage(context.Background(), []float64{1, math.Inf(1)})
	core.AssertError(t, err)
	core.AssertTrue(t, ok)
	core.AssertContains(t, err.Error(), "reward values must be finite")

	_, ok, err = model.RunGRPOAdvantage(context.Background(), nil)
	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
}

func TestHIPTrainingLoadedModelAdamWUpdateHook_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	model := &hipLoadedModel{driver: driver, kernels: fakeOptimizerHIPKernelSet{}}
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "a", Values: []float32{1, 2}},
		{Name: "b", Values: []float32{3}},
	}, NativeAdamWConfig{LearningRate: 0.01, WeightDecay: 0.1, WeightDecaySet: true})
	core.RequireNoError(t, err)
	expected, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "a", Values: []float32{1, 2}},
		{Name: "b", Values: []float32{3}},
	}, NativeAdamWConfig{LearningRate: 0.01, WeightDecay: 0.1, WeightDecaySet: true})
	core.RequireNoError(t, err)
	gradients := [][]float32{{0.5, -0.25}, {0.125}}
	core.RequireNoError(t, expected.StepInPlace(gradients))

	ok, err := model.RunAdamWUpdate(context.Background(), state, gradients)
	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, expected.Step, state.Step)
	for index, want := range expected.Parameters() {
		assertAdamWFloat32Near(t, want, state.Parameters()[index], 0.0001)
	}
	for index, want := range expected.FirstMoment() {
		assertAdamWFloat32Near(t, want, state.FirstMoment()[index], 0.0001)
	}
	for index, want := range expected.SecondMoment() {
		assertAdamWFloat32Near(t, want, state.SecondMoment()[index], 0.00001)
	}
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameAdamWUpdate, driver.launches[0].Name)
}

func TestHIPTrainingLoadedModelAdamWUpdateHook_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "w", Values: []float32{1, 2}},
	}, NativeAdamWConfig{})
	core.RequireNoError(t, err)

	model := &hipLoadedModel{driver: driver, kernels: newDefaultHIPKernelSet()}
	ok, err := model.RunAdamWUpdate(context.Background(), state, [][]float32{{0.1, 0.2}})
	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)

	model = &hipLoadedModel{driver: driver, kernels: fakeOptimizerHIPKernelSet{}}
	ok, err = model.RunAdamWUpdate(context.Background(), state, [][]float32{{0.1}})
	core.AssertError(t, err)
	core.AssertTrue(t, ok)
	core.AssertContains(t, err.Error(), "gradient length")

	ok, err = model.RunAdamWUpdate(context.Background(), state, nil)
	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)

	ok, err = (*hipLoadedModel)(nil).RunAdamWUpdate(context.Background(), state, [][]float32{{0.1, 0.2}})
	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
}

func TestHIPTrainingLoadedModelFixtureHooksRequireSpecificKernelStatus_Ugly(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	model := &hipLoadedModel{driver: driver, kernels: fakeProjectionOnlyHIPKernelSet{}}

	_, crossEntropyOK, err := model.RunEvalCrossEntropyLoss(context.Background(), [][]float32{{1, 0}}, []int{0})
	core.RequireNoError(t, err)
	core.AssertFalse(t, crossEntropyOK)
	_, distillationOK, err := model.RunDistillationKLLoss(context.Background(), [][]float32{{1, 0}}, [][]float32{{2, 0}}, 1)
	core.RequireNoError(t, err)
	core.AssertFalse(t, distillationOK)
	_, grpoOK, err := model.RunGRPOAdvantage(context.Background(), []float64{1, 2, 3})
	core.RequireNoError(t, err)
	core.AssertFalse(t, grpoOK)
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "w", Values: []float32{1, 2}},
	}, NativeAdamWConfig{})
	core.RequireNoError(t, err)
	optimizerOK, err := model.RunAdamWUpdate(context.Background(), state, [][]float32{{0.1, 0.2}})
	core.RequireNoError(t, err)
	core.AssertFalse(t, optimizerOK)
	core.AssertEqual(t, 0, len(driver.launches))
}

type fakeProjectionOnlyHIPKernelSet struct {
	hipKernelStub
}

func (fakeProjectionOnlyHIPKernelSet) Status() hipKernelStatus {
	return hipKernelStatus{
		Projection: hipKernelStatusLinked,
		Reason:     "fake projection-only test kernel",
	}
}

type fakeOptimizerHIPKernelSet struct {
	hipKernelStub
}

func (fakeOptimizerHIPKernelSet) Status() hipKernelStatus {
	return hipKernelStatus{
		Optimizer: hipKernelStatusLinked,
		Reason:    "fake optimizer test kernel",
	}
}
