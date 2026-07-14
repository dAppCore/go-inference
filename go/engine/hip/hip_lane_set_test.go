// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"os"
	"slices"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model"
)

type fakeHIPLaneExecutor struct {
	prepareCalls  int
	forwardCalls  [][]hipLaneForwardInput
	serialForward bool
	closed        bool
}

func (executor *fakeHIPLaneExecutor) Prepare(_ context.Context, spec inference.LaneSpec) (hipPreparedLane, error) {
	executor.prepareCalls++
	sample := hipNewLaneSampleState(spec.Sampler, spec.SampleSeed)
	pending := spec.PromptIDs[len(spec.PromptIDs)-1] + 10
	if sample.enabled {
		pending = int32(sample.draw() * 1_000_000)
	}
	return hipPreparedLane{
		PendingToken: pending,
		Position:     len(spec.PromptIDs),
		Sample:       sample,
	}, nil
}

func (executor *fakeHIPLaneExecutor) Forward(_ context.Context, inputs []hipLaneForwardInput) ([]hipLaneForwardOutput, error) {
	executor.forwardCalls = append(executor.forwardCalls, append([]hipLaneForwardInput(nil), inputs...))
	outputs := make([]hipLaneForwardOutput, len(inputs))
	for index, input := range inputs {
		sample := input.Sample.clone()
		pending := input.PendingToken + 1
		if sample.enabled {
			pending = int32(sample.draw() * 1_000_000)
		}
		outputs[index] = hipLaneForwardOutput{
			PendingToken: pending,
			Position:     input.Position + 1,
			DeviceState:  input.DeviceState,
			Sample:       sample,
		}
	}
	return outputs, nil
}

func (executor *fakeHIPLaneExecutor) Close() error {
	executor.closed = true
	return nil
}

func (executor *fakeHIPLaneExecutor) UsesSharedBatchForward() bool {
	return !executor.serialForward
}

func TestHIPLaneSetStepBatchesSurvivors_Good(t *testing.T) {
	executor := &fakeHIPLaneExecutor{}
	laneSet := newHIPLaneSetWithExecutor(2, executor)
	defer laneSet.Close()

	laneA, err := laneSet.Prepare(context.Background(), inference.LaneSpec{
		PromptIDs:  []int32{1},
		MaxNew:     3,
		StopTokens: []int32{12},
	})
	core.RequireNoError(t, err)
	laneB, err := laneSet.Prepare(context.Background(), inference.LaneSpec{
		PromptIDs: []int32{2, 3},
		MaxNew:    3,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, laneSet.Active())

	first, err := laneSet.Step(context.Background())
	core.RequireNoError(t, err)
	core.AssertEqual(t, []inference.LaneStep{
		{Lane: laneA, Token: 11, HasToken: true},
		{Lane: laneB, Token: 13, HasToken: true},
	}, first)
	core.AssertEqual(t, uint64(1), laneSet.BatchForwardCount())
	if len(executor.forwardCalls) != 1 || len(executor.forwardCalls[0]) != 2 {
		t.Fatalf("first Step forward batches=%v, want one K=2 call", executor.forwardCalls)
	}

	second, err := laneSet.Step(context.Background())
	core.RequireNoError(t, err)
	core.AssertEqual(t, []inference.LaneStep{
		{Lane: laneA, Token: 12, HasToken: true, Terminal: true},
		{Lane: laneB, Token: 14, HasToken: true},
	}, second)
	core.AssertEqual(t, uint64(2), laneSet.BatchForwardCount())
	if len(executor.forwardCalls) != 2 || len(executor.forwardCalls[1]) != 1 || executor.forwardCalls[1][0].PendingToken != 14 {
		t.Fatalf("second Step survivor batch=%v, want lane B only", executor.forwardCalls)
	}

	third, err := laneSet.Step(context.Background())
	core.RequireNoError(t, err)
	core.AssertEqual(t, []inference.LaneStep{{Lane: laneB, Token: 15, HasToken: true, Terminal: true}}, third)
	core.AssertEqual(t, uint64(2), laneSet.BatchForwardCount())
	empty, err := laneSet.Step(context.Background())
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, len(empty))

	core.RequireNoError(t, laneSet.Retire(laneA))
	core.RequireNoError(t, laneSet.Retire(laneB))
	core.AssertEqual(t, 0, laneSet.Active())
	if err := laneSet.Retire(laneA); err == nil {
		t.Fatal("Retire accepted an unknown lane")
	}
}

func TestHIPLaneSetStepSerialForwardKeepsBatchCounterZero_Good(t *testing.T) {
	executor := &fakeHIPLaneExecutor{serialForward: true}
	laneSet := newHIPLaneSetWithExecutor(2, executor)
	defer laneSet.Close()

	_, err := laneSet.Prepare(context.Background(), inference.LaneSpec{PromptIDs: []int32{1}, MaxNew: 2})
	core.RequireNoError(t, err)
	_, err = laneSet.Prepare(context.Background(), inference.LaneSpec{PromptIDs: []int32{2}, MaxNew: 2})
	core.RequireNoError(t, err)
	steps, err := laneSet.Step(context.Background())
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, len(steps))
	core.AssertEqual(t, uint64(0), laneSet.BatchForwardCount())
	core.AssertEqual(t, 1, len(executor.forwardCalls))
}

func TestHIPLaneSetPrepareValidation_Bad(t *testing.T) {
	executor := &fakeHIPLaneExecutor{}
	laneSet := newHIPLaneSetWithExecutor(1, executor)
	defer laneSet.Close()

	if _, err := laneSet.Prepare(context.Background(), inference.LaneSpec{MaxNew: 1}); err == nil {
		t.Fatal("Prepare accepted an empty prompt")
	}
	if _, err := laneSet.Prepare(context.Background(), inference.LaneSpec{PromptIDs: []int32{1}}); err == nil {
		t.Fatal("Prepare accepted MaxNew <= 0")
	}
	if _, err := laneSet.Prepare(context.Background(), inference.LaneSpec{
		PromptIDs:  []int32{1},
		MaxNew:     1,
		Sampler:    inference.SamplerConfig{Temperature: 0.7},
		SampleSeed: 7,
	}); err != nil {
		t.Fatalf("Prepare rejected sampled decoding: %v", err)
	}
	if _, err := laneSet.Prepare(context.Background(), inference.LaneSpec{PromptIDs: []int32{2}, MaxNew: 1}); err == nil {
		t.Fatal("Prepare exceeded MaxLanes")
	}
	core.AssertEqual(t, 1, executor.prepareCalls)
}

func TestHIPLaneSetOverlappedPrepare_Good(t *testing.T) {
	executor := &fakeHIPLaneExecutor{}
	laneSet := newHIPLaneSetWithExecutor(2, executor)
	defer laneSet.Close()

	admitter, ok := any(laneSet).(inference.LaneSetOverlappedAdmitter)
	if !ok {
		t.Fatal("HIP lane set does not implement LaneSetOverlappedAdmitter")
	}
	pending, err := admitter.BeginPrepare(context.Background(), inference.LaneSpec{
		PromptIDs: []int32{1, 2},
		MaxNew:    2,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, laneSet.Active())
	core.AssertEqual(t, 1, executor.prepareCalls)

	handle, err := admitter.CommitPrepare(pending)
	core.RequireNoError(t, err)
	core.RequireTrue(t, handle.Valid())
	core.AssertEqual(t, 1, laneSet.Active())
	steps, err := laneSet.Step(context.Background())
	core.RequireNoError(t, err)
	core.AssertEqual(t, []inference.LaneStep{{Lane: handle, Token: 12, HasToken: true}}, steps)
	if _, err := admitter.CommitPrepare(pending); err == nil {
		t.Fatal("CommitPrepare accepted an already consumed pending lane")
	}
}

func TestHIPLaneSetOverlappedCommitAtCapacity_Bad(t *testing.T) {
	executor := &fakeHIPLaneExecutor{}
	laneSet := newHIPLaneSetWithExecutor(1, executor)
	defer laneSet.Close()

	_, err := laneSet.Prepare(context.Background(), inference.LaneSpec{PromptIDs: []int32{1}, MaxNew: 1})
	core.RequireNoError(t, err)
	pending, err := laneSet.BeginPrepare(context.Background(), inference.LaneSpec{PromptIDs: []int32{2}, MaxNew: 1})
	core.RequireNoError(t, err)
	prepared := pending.(*hipPendingLane)
	device := hipNewGemma4Q4DeviceDecodeState("", 0)
	prepared.lane.deviceState = device

	if _, err := laneSet.CommitPrepare(pending); err == nil {
		t.Fatal("CommitPrepare admitted a lane beyond MaxLanes")
	}
	core.AssertEqual(t, 1, laneSet.Active())
	core.AssertTrue(t, device.closed)
	core.AssertEqual(t, 2, executor.prepareCalls)
}

func TestHIPLaneSetOverlappedDiscard_Ugly(t *testing.T) {
	executor := &fakeHIPLaneExecutor{}
	laneSet := newHIPLaneSetWithExecutor(1, executor)
	defer laneSet.Close()

	pending, err := laneSet.BeginPrepare(context.Background(), inference.LaneSpec{PromptIDs: []int32{1}, MaxNew: 1})
	core.RequireNoError(t, err)
	prepared := pending.(*hipPendingLane)
	device := hipNewGemma4Q4DeviceDecodeState("", 0)
	prepared.lane.deviceState = device

	pending.Discard()
	pending.Discard()
	core.AssertTrue(t, device.closed)
	core.AssertEqual(t, 0, laneSet.Active())
	if _, err := laneSet.CommitPrepare(pending); err == nil {
		t.Fatal("CommitPrepare accepted a discarded pending lane")
	}
}

func TestHIPLaneSetSampledLanesKeepIndependentSeededStreams_Good(t *testing.T) {
	executor := &fakeHIPLaneExecutor{}
	laneSet := newHIPLaneSetWithExecutor(4, executor)
	defer laneSet.Close()

	config := inference.SamplerConfig{Temperature: 0.8, TopK: 8, RepeatPenalty: 1.2}
	specs := []inference.LaneSpec{
		{PromptIDs: []int32{1}, MaxNew: 4, Sampler: config, SampleSeed: 41},
		{PromptIDs: []int32{9, 1}, MaxNew: 4, Sampler: config, SampleSeed: 41},
		{PromptIDs: []int32{1}, MaxNew: 4, Sampler: config, SampleSeed: 42},
		{PromptIDs: []int32{1}, MaxNew: 4, Sampler: inference.SamplerConfig{TopK: 1}, SampleSeed: 41},
	}
	handles := make([]inference.LaneHandle, len(specs))
	for index, spec := range specs {
		var err error
		handles[index], err = laneSet.Prepare(context.Background(), spec)
		core.RequireNoError(t, err)
	}
	streams := hipDrainLaneSet(t, laneSet)
	core.AssertEqual(t, streams[handles[0].ID], streams[handles[1].ID])
	if slices.Equal(streams[handles[0].ID], streams[handles[2].ID]) {
		t.Fatalf("different sample seeds produced the same stream: %v", streams[handles[0].ID])
	}
	core.AssertEqual(t, []int32{11, 12, 13, 14}, streams[handles[3].ID])
	core.AssertEqual(t, uint64(3), laneSet.BatchForwardCount())

	if len(executor.forwardCalls) == 0 || len(executor.forwardCalls[0]) != len(specs) {
		t.Fatalf("first sampled forward batch = %v, want K=%d", executor.forwardCalls, len(specs))
	}
	for index := 0; index < 3; index++ {
		history := executor.forwardCalls[0][index].Sample.history
		core.AssertEqual(t, []int32{streams[handles[index].ID][0]}, history)
	}
	if executor.forwardCalls[0][3].Sample.enabled {
		t.Fatal("TopK-only zero-temperature lane unexpectedly engaged sampling")
	}
}

func TestHIPLaneSampleStateDrawMatchesModelSampler_Good(t *testing.T) {
	for _, seed := range []uint64{0, 1, 41, ^uint64(0)} {
		got := hipNewLaneSampleState(inference.SamplerConfig{Temperature: 1}, seed)
		want := model.NewSampler(seed)
		for draw := 0; draw < 8; draw++ {
			core.AssertEqual(t, float64(want.Draw()), got.draw())
		}
	}
}

func TestHIPLaneSetCloseReleasesExecutor_Ugly(t *testing.T) {
	executor := &fakeHIPLaneExecutor{}
	laneSet := newHIPLaneSetWithExecutor(0, executor)
	_, err := laneSet.Prepare(context.Background(), inference.LaneSpec{PromptIDs: []int32{1}, MaxNew: 1})
	core.RequireNoError(t, err)
	core.RequireNoError(t, laneSet.Close())
	core.AssertTrue(t, executor.closed)
	core.AssertEqual(t, 0, laneSet.Active())
	if _, err := laneSet.Prepare(context.Background(), inference.LaneSpec{PromptIDs: []int32{2}, MaxNew: 1}); err == nil {
		t.Fatal("Prepare accepted a closed lane set")
	}
}

func TestHIPLaneSetE2BHardwareMatchesSingleLanes_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run the HIP lane-set receipt")
	}
	modelPath := strings.TrimSpace(os.Getenv("GO_ROCM_MODEL_PATH"))
	if modelPath == "" {
		t.Skip("set GO_ROCM_MODEL_PATH to a linked Gemma-4 Q4 GGUF")
	}
	if strings.TrimSpace(os.Getenv("GO_ROCM_KERNEL_HSACO")) == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to the linked ROCm kernels HSACO")
	}
	if !ROCmAvailable() {
		t.Skip("ROCm runtime is not available on this host")
	}

	loadedResult := (&rocmBackend{}).LoadModel(modelPath, inference.WithContextLen(256))
	if !loadedResult.OK {
		t.Fatalf("production ROCm LoadModel(%q): %v", modelPath, loadedResult.Value)
	}
	model, ok := loadedResult.Value.(*rocmModel)
	if !ok {
		t.Fatalf("production ROCm LoadModel returned %T, want *rocmModel", loadedResult.Value)
	}
	defer func() {
		if result := model.Close(); !result.OK {
			t.Errorf("Close model: %v", result.Value)
		}
	}()
	if !model.BatchStepAvailable() {
		t.Fatal("production ROCm model did not expose BatchStepAvailable")
	}

	promptA, err := model.Tokenize("Hello")
	core.RequireNoError(t, err)
	promptB, err := model.Tokenize("Hello, briefly explain attention.")
	core.RequireNoError(t, err)
	specs := []inference.LaneSpec{
		{PromptIDs: promptA, MaxNew: 4},
		{PromptIDs: promptB, MaxNew: 4},
	}

	serial := make([][]int32, len(specs))
	for index, spec := range specs {
		set, err := model.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 1})
		core.RequireNoError(t, err)
		handle, err := set.Prepare(context.Background(), spec)
		core.RequireNoError(t, err)
		streams := hipDrainLaneSet(t, set)
		serial[index] = streams[handle.ID]
		core.AssertEqual(t, uint64(spec.MaxNew-1), set.BatchForwardCount())
		core.RequireNoError(t, set.Close())
	}

	set, err := model.OpenLaneSet(inference.LaneSetConfig{MaxLanes: len(specs)})
	core.RequireNoError(t, err)
	handles := make([]inference.LaneHandle, len(specs))
	for index, spec := range specs {
		handles[index], err = set.Prepare(context.Background(), spec)
		core.RequireNoError(t, err)
	}
	batched := hipDrainLaneSet(t, set)
	for index, handle := range handles {
		core.AssertEqual(t, serial[index], batched[handle.ID])
	}
	core.AssertEqual(t, uint64(specs[0].MaxNew-1), set.BatchForwardCount())
	core.RequireNoError(t, set.Close())
}

func TestHIPLaneSetE2BHardwareHeadLoRAMatchesSerial_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run the HIP head-LoRA lane-set receipt")
	}
	modelPath := strings.TrimSpace(os.Getenv("GO_ROCM_MODEL_PATH"))
	if modelPath == "" {
		t.Skip("set GO_ROCM_MODEL_PATH to a linked Gemma-4 model")
	}
	if strings.TrimSpace(os.Getenv("GO_ROCM_KERNEL_HSACO")) == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to the linked ROCm kernels HSACO")
	}
	if !ROCmAvailable() {
		t.Skip("ROCm runtime is not available on this host")
	}

	loadedResult := (&rocmBackend{}).LoadModel(modelPath, inference.WithContextLen(256))
	if !loadedResult.OK {
		t.Fatalf("production ROCm LoadModel(%q): %v", modelPath, loadedResult.Value)
	}
	textModel, ok := loadedResult.Value.(*rocmModel)
	if !ok {
		t.Fatalf("production ROCm LoadModel returned %T, want *rocmModel", loadedResult.Value)
	}
	defer func() {
		if result := textModel.Close(); !result.OK {
			t.Errorf("Close model: %v", result.Value)
		}
	}()
	loaded, ok := textModel.native.(*hipLoadedModel)
	if !ok || loaded == nil {
		t.Fatalf("production ROCm native model is %T, want *hipLoadedModel", textModel.native)
	}

	const rank = 2
	hidden, vocab := loaded.modelInfo.HiddenSize, loaded.modelInfo.VocabSize
	a := make([]float32, rank*hidden)
	for index := 0; index < hidden; index++ {
		a[index] = float32(index%7-3) * 0.001
		a[hidden+index] = float32(index%5-2) * 0.001
	}
	b := make([]float32, vocab*rank)
	for index, signs := range [][2]float32{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}} {
		token := 100 + index
		b[token*rank] = signs[0] * 0.01
		b[token*rank+1] = signs[1] * 0.01
	}
	adapterPath := t.TempDir()
	core.RequireNoError(t, saveMetalHeadLoRAAdapter(adapterPath, a, b, vocab, hidden, rank, rank))
	_, err := textModel.LoadAdapter(adapterPath)
	core.RequireNoError(t, err)
	if !textModel.BatchStepAvailable() {
		t.Fatal("head-LoRA model did not expose BatchStepAvailable")
	}

	const prompt = "Hello"
	serialTokens := collectInferenceTokens(textModel.Generate(context.Background(), prompt, inference.WithMaxTokens(3), inference.WithTemperature(0)))
	core.RequireNoError(t, resultError(textModel.Err()))
	serial := inferenceTokenIDs(serialTokens)
	if len(serial) != 3 {
		t.Fatalf("adapter serial tokens = %v, want three tokens", serial)
	}
	promptIDs, err := textModel.Tokenize(prompt)
	core.RequireNoError(t, err)
	set, err := textModel.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 1})
	core.RequireNoError(t, err)
	executor := set.(*hipLaneSet).executor.(*hipGemma4Q4LaneExecutor)
	core.AssertTrue(t, executor.forward.HeadLoRA == loaded.gemma4LoRA)
	handle, err := set.Prepare(context.Background(), inference.LaneSpec{PromptIDs: promptIDs, MaxNew: 3})
	core.RequireNoError(t, err)
	lane := hipDrainLaneSet(t, set)[handle.ID]
	core.RequireNoError(t, set.Close())
	core.AssertEqual(t, serial, lane)
}

func TestHIPLaneSetE2BHardwareSampledMatchesSingleLanes_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run the HIP sampled lane-set receipt")
	}
	modelPath := strings.TrimSpace(os.Getenv("GO_ROCM_MODEL_PATH"))
	if modelPath == "" {
		t.Skip("set GO_ROCM_MODEL_PATH to a linked Gemma-4 Q4 GGUF")
	}
	if strings.TrimSpace(os.Getenv("GO_ROCM_KERNEL_HSACO")) == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to the linked ROCm kernels HSACO")
	}
	if !ROCmAvailable() {
		t.Skip("ROCm runtime is not available on this host")
	}

	loadedResult := (&rocmBackend{}).LoadModel(modelPath, inference.WithContextLen(256))
	if !loadedResult.OK {
		t.Fatalf("production ROCm LoadModel(%q): %v", modelPath, loadedResult.Value)
	}
	model, ok := loadedResult.Value.(*rocmModel)
	if !ok {
		t.Fatalf("production ROCm LoadModel returned %T, want *rocmModel", loadedResult.Value)
	}
	defer func() {
		if result := model.Close(); !result.OK {
			t.Errorf("Close model: %v", result.Value)
		}
	}()

	promptA, err := model.Tokenize("Hello")
	core.RequireNoError(t, err)
	promptB, err := model.Tokenize("Name one benefit of continuous batching.")
	core.RequireNoError(t, err)
	specs := []inference.LaneSpec{
		{
			PromptIDs:  promptA,
			MaxNew:     3,
			Sampler:    inference.SamplerConfig{Temperature: 0.8, TopK: 8},
			SampleSeed: 7,
		},
		{
			PromptIDs:  promptB,
			MaxNew:     3,
			Sampler:    inference.SamplerConfig{Temperature: 0.9, MinP: 0.1, RepeatPenalty: 1.2},
			SampleSeed: 21,
		},
	}

	serial := make([][]int32, len(specs))
	for index, spec := range specs {
		set, err := model.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 1})
		core.RequireNoError(t, err)
		handle, err := set.Prepare(context.Background(), spec)
		core.RequireNoError(t, err)
		serial[index] = hipDrainLaneSet(t, set)[handle.ID]
		core.AssertEqual(t, uint64(spec.MaxNew-1), set.BatchForwardCount())
		core.RequireNoError(t, set.Close())
	}

	set, err := model.OpenLaneSet(inference.LaneSetConfig{MaxLanes: len(specs)})
	core.RequireNoError(t, err)
	handles := make([]inference.LaneHandle, len(specs))
	for index, spec := range specs {
		handles[index], err = set.Prepare(context.Background(), spec)
		core.RequireNoError(t, err)
	}
	batched := hipDrainLaneSet(t, set)
	for index, handle := range handles {
		core.AssertEqual(t, serial[index], batched[handle.ID])
	}
	core.AssertEqual(t, uint64(specs[0].MaxNew-1), set.BatchForwardCount())
	core.RequireNoError(t, set.Close())
}

func TestHIPLaneSetE2BHardwareOverlappedAdmissionMatchesSingleLanes_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run the HIP overlapped-admission receipt")
	}
	modelPath := strings.TrimSpace(os.Getenv("GO_ROCM_MODEL_PATH"))
	if modelPath == "" {
		t.Skip("set GO_ROCM_MODEL_PATH to a linked Gemma-4 Q4 GGUF")
	}
	if strings.TrimSpace(os.Getenv("GO_ROCM_KERNEL_HSACO")) == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to the linked ROCm kernels HSACO")
	}
	if !ROCmAvailable() {
		t.Skip("ROCm runtime is not available on this host")
	}

	loadedResult := (&rocmBackend{}).LoadModel(modelPath, inference.WithContextLen(256))
	if !loadedResult.OK {
		t.Fatalf("production ROCm LoadModel(%q): %v", modelPath, loadedResult.Value)
	}
	model, ok := loadedResult.Value.(*rocmModel)
	if !ok {
		t.Fatalf("production ROCm LoadModel returned %T, want *rocmModel", loadedResult.Value)
	}
	defer func() {
		if result := model.Close(); !result.OK {
			t.Errorf("Close model: %v", result.Value)
		}
	}()

	promptA, err := model.Tokenize("Briefly define attention.")
	core.RequireNoError(t, err)
	promptB, err := model.Tokenize(strings.Repeat("Explain one useful property of retained inference state. ", 12))
	core.RequireNoError(t, err)
	specs := []inference.LaneSpec{
		{PromptIDs: promptA, MaxNew: 4},
		{PromptIDs: promptB, MaxNew: 3},
	}

	serial := make([][]int32, len(specs))
	for index, spec := range specs {
		set, err := model.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 1})
		core.RequireNoError(t, err)
		handle, err := set.Prepare(context.Background(), spec)
		core.RequireNoError(t, err)
		serial[index] = hipDrainLaneSet(t, set)[handle.ID]
		core.RequireNoError(t, set.Close())
	}

	set, err := model.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 2})
	core.RequireNoError(t, err)
	defer set.Close()
	admitter, ok := set.(inference.LaneSetOverlappedAdmitter)
	if !ok {
		t.Fatal("production HIP lane set does not expose overlapped admission")
	}
	handleA, err := set.Prepare(context.Background(), specs[0])
	core.RequireNoError(t, err)
	type prepareResult struct {
		pending inference.PendingLane
		err     error
	}
	prepared := make(chan prepareResult, 1)
	go func() {
		pending, beginErr := admitter.BeginPrepare(context.Background(), specs[1])
		prepared <- prepareResult{pending: pending, err: beginErr}
	}()

	first, err := set.Step(context.Background())
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(first))
	core.AssertEqual(t, handleA, first[0].Lane)
	result := <-prepared
	core.RequireNoError(t, result.err)
	handleB, err := admitter.CommitPrepare(result.pending)
	core.RequireNoError(t, err)
	streams := hipDrainLaneSet(t, set)
	streams[handleA.ID] = append([]int32{first[0].Token}, streams[handleA.ID]...)
	core.AssertEqual(t, serial[0], streams[handleA.ID])
	core.AssertEqual(t, serial[1], streams[handleB.ID])
}

func TestHIPLaneSet26BMoEHardwareMatchesSingleLanes_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MOE_LANE_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MOE_LANE_TESTS=1 to run the HIP MoE lane-set receipt")
	}
	modelPath := strings.TrimSpace(os.Getenv("GO_ROCM_MODEL_PATH"))
	if modelPath == "" {
		t.Skip("set GO_ROCM_MODEL_PATH to a linked Gemma-4 26B-A4B Q4 GGUF")
	}
	if strings.TrimSpace(os.Getenv("GO_ROCM_KERNEL_HSACO")) == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to the linked ROCm kernels HSACO")
	}
	if !ROCmAvailable() {
		t.Skip("ROCm runtime is not available on this host")
	}

	loadedResult := (&rocmBackend{}).LoadModel(modelPath, inference.WithContextLen(256))
	if !loadedResult.OK {
		t.Fatalf("production ROCm LoadModel(%q): %v", modelPath, loadedResult.Value)
	}
	model, ok := loadedResult.Value.(*rocmModel)
	if !ok {
		t.Fatalf("production ROCm LoadModel returned %T, want *rocmModel", loadedResult.Value)
	}
	defer func() {
		if result := model.Close(); !result.OK {
			t.Errorf("Close model: %v", result.Value)
		}
	}()
	loaded, ok := model.native.(*hipLoadedModel)
	if !ok || loaded == nil {
		t.Fatalf("production ROCm native model is %T, want *hipLoadedModel", model.native)
	}
	forward, err := loaded.cachedGemma4Q4ForwardConfig(loaded.modelInfo.NumLayers)
	core.RequireNoError(t, err)
	moe := false
	for _, layer := range forward.Layers {
		moe = moe || layer.MoE != nil
	}
	if !moe {
		t.Fatalf("production model is not sparse: %+v", loaded.modelInfo)
	}

	promptA, err := model.Tokenize("Hello")
	core.RequireNoError(t, err)
	promptB, err := model.Tokenize("Two plus two is")
	core.RequireNoError(t, err)
	specs := []inference.LaneSpec{
		{PromptIDs: promptA, MaxNew: 2},
		{
			PromptIDs:  promptB,
			MaxNew:     2,
			Sampler:    inference.SamplerConfig{Temperature: 0.8, TopK: 8},
			SampleSeed: 7,
		},
	}

	serial := make([][]int32, len(specs))
	for index, spec := range specs {
		set, err := model.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 1})
		core.RequireNoError(t, err)
		handle, err := set.Prepare(context.Background(), spec)
		core.RequireNoError(t, err)
		serial[index] = hipDrainLaneSet(t, set)[handle.ID]
		if set.BatchForwardCount() == 0 {
			t.Fatal("single-lane MoE decode did not use the shared batch forward route")
		}
		core.RequireNoError(t, set.Close())
	}

	set, err := model.OpenLaneSet(inference.LaneSetConfig{MaxLanes: len(specs)})
	core.RequireNoError(t, err)
	handles := make([]inference.LaneHandle, len(specs))
	for index, spec := range specs {
		handles[index], err = set.Prepare(context.Background(), spec)
		core.RequireNoError(t, err)
	}
	batched := hipDrainLaneSet(t, set)
	for index, handle := range handles {
		core.AssertEqual(t, serial[index], batched[handle.ID])
	}
	if set.BatchForwardCount() == 0 {
		t.Fatal("multi-lane MoE decode did not use the shared batch forward route")
	}
	core.RequireNoError(t, set.Close())
}

func hipDrainLaneSet(t *testing.T, set inference.LaneSet) map[int][]int32 {
	t.Helper()
	streams := make(map[int][]int32)
	for {
		steps, err := set.Step(context.Background())
		core.RequireNoError(t, err)
		if len(steps) == 0 {
			return streams
		}
		for _, step := range steps {
			if step.HasToken {
				streams[step.Lane.ID] = append(streams[step.Lane.ID], step.Token)
			}
			if step.Terminal {
				core.RequireNoError(t, set.Retire(step.Lane))
			}
		}
	}
}
