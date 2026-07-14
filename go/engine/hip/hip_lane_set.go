// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const (
	hipLaneSetOperation       = "rocm.hip.LaneSet"
	defaultHIPLaneSetMaxLanes = 8
)

type hipPreparedLane struct {
	PendingToken int32
	Position     int
	DeviceState  *hipGemma4Q4DeviceDecodeState
	Sample       hipLaneSampleState
}

type hipLaneForwardInput struct {
	PendingToken int32
	Position     int
	DeviceState  *hipGemma4Q4DeviceDecodeState
	Sample       hipLaneSampleState
}

type hipLaneForwardOutput struct {
	PendingToken int32
	Position     int
	DeviceState  *hipGemma4Q4DeviceDecodeState
	Sample       hipLaneSampleState
}

type hipLaneSampleState struct {
	enabled  bool
	rng      uint64
	generate inference.GenerateConfig
	history  []int32
}

type hipLaneExecutor interface {
	Prepare(context.Context, inference.LaneSpec) (hipPreparedLane, error)
	Forward(context.Context, []hipLaneForwardInput) ([]hipLaneForwardOutput, error)
	UsesSharedBatchForward() bool
	Close() error
}

type hipLaneSet struct {
	executor hipLaneExecutor
	maxLanes int
	lanes    map[int]*hipDecodeLane
	order    []int
	nextID   int
	fwdCount uint64
	closed   bool
}

type hipDecodeLane struct {
	id           int
	pendingToken int32
	position     int
	deviceState  *hipGemma4Q4DeviceDecodeState
	sample       hipLaneSampleState
	maxNew       int
	generated    int
	stops        map[int32]bool
	terminal     bool
}

var _ inference.LaneSet = (*hipLaneSet)(nil)
var _ inference.LaneSetOverlappedAdmitter = (*hipLaneSet)(nil)

// hipPendingLane owns a fully prepared lane until exactly one CommitPrepare or
// Discard consumes it. It is deliberately detached from the set's maps so the
// expensive preparation may overlap Step on another goroutine.
type hipPendingLane struct {
	owner *hipLaneSet
	lane  *hipDecodeLane
}

func (pending *hipPendingLane) Discard() {
	if pending == nil || pending.lane == nil {
		return
	}
	lane := pending.lane
	pending.lane = nil
	pending.owner = nil
	_ = lane.deviceState.Close()
}

func newHIPLaneSetWithExecutor(maxLanes int, executor hipLaneExecutor) *hipLaneSet {
	if maxLanes <= 0 {
		maxLanes = defaultHIPLaneSetMaxLanes
	}
	return &hipLaneSet{
		executor: executor,
		maxLanes: maxLanes,
		lanes:    make(map[int]*hipDecodeLane, maxLanes),
	}
}

func (m *hipTokenModel) OpenLaneSet(cfg inference.LaneSetConfig) (inference.LaneSet, error) {
	if m == nil || m.loaded == nil {
		return nil, core.NewError("hip.TokenModel.OpenLaneSet: model is not initialised")
	}
	if !hipLoadedGemma4Q4GenerateLinked(m.loaded) || m.loaded.modelInfo.NumLayers <= 0 {
		return nil, core.NewError("hip.TokenModel.OpenLaneSet: model is not a linked Gemma4 runtime")
	}
	if m.loaded.gemma4LoRA != nil {
		return nil, core.NewError("hip.TokenModel.OpenLaneSet: Gemma4 head LoRA currently requires the serial host-sampling lane")
	}
	forward, err := m.loaded.cachedGemma4Q4ForwardConfig(m.loaded.modelInfo.NumLayers)
	if err != nil {
		return nil, err
	}
	engineConfig := m.loaded.gemma4Q4EngineConfig()
	mode, err := engineConfig.deviceKVMode()
	if err != nil {
		return nil, err
	}
	executor := &hipGemma4Q4LaneExecutor{
		loaded:     m.loaded,
		forward:    forward,
		engine:     engineConfig,
		mode:       mode,
		workspace:  hipNewAttentionHeadsChunkedWorkspace(),
		contextLen: m.loaded.contextSize,
	}
	if executor.contextLen <= 0 {
		executor.contextLen = defaultContextLengthCap
	}
	return newHIPLaneSetWithExecutor(cfg.MaxLanes, executor), nil
}

func (set *hipLaneSet) Prepare(ctx context.Context, spec inference.LaneSpec) (inference.LaneHandle, error) {
	if set == nil || set.closed || set.executor == nil {
		return inference.LaneHandle{}, core.NewError("hip.LaneSet.Prepare: lane set is unavailable or closed")
	}
	if len(set.lanes) >= set.maxLanes {
		return inference.LaneHandle{}, core.NewError("hip.LaneSet.Prepare: lane set is at MaxLanes")
	}
	pending, err := set.BeginPrepare(ctx, spec)
	if err != nil {
		return inference.LaneHandle{}, err
	}
	return set.CommitPrepare(pending)
}

// BeginPrepare performs only independent admission work. In particular it
// does not inspect or mutate lanes, order, or nextID: those remain owned by the
// Step goroutine while this method prefills on another goroutine.
func (set *hipLaneSet) BeginPrepare(ctx context.Context, spec inference.LaneSpec) (inference.PendingLane, error) {
	if set == nil || set.closed || set.executor == nil {
		return nil, core.NewError("hip.LaneSet.BeginPrepare: lane set is unavailable or closed")
	}
	if len(spec.PromptIDs) == 0 {
		return nil, core.NewError("hip.LaneSet.BeginPrepare: empty prompt")
	}
	if spec.MaxNew <= 0 {
		return nil, core.NewError("hip.LaneSet.BeginPrepare: MaxNew must be > 0")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	prepared, err := set.executor.Prepare(ctx, spec)
	if err != nil {
		return nil, err
	}
	lane := &hipDecodeLane{
		pendingToken: prepared.PendingToken,
		position:     prepared.Position,
		deviceState:  prepared.DeviceState,
		sample:       prepared.Sample,
		maxNew:       spec.MaxNew,
		stops:        hipLaneStopSet(spec.StopTokens),
	}
	return &hipPendingLane{owner: set, lane: lane}, nil
}

// CommitPrepare is the cheap owner-goroutine splice. Capacity and set lifetime
// are checked here because BeginPrepare must remain independent of mutable set
// bookkeeping.
func (set *hipLaneSet) CommitPrepare(p inference.PendingLane) (inference.LaneHandle, error) {
	if set == nil {
		return inference.LaneHandle{}, core.NewError("hip.LaneSet.CommitPrepare: nil lane set")
	}
	pending, ok := p.(*hipPendingLane)
	if !ok || pending == nil || pending.lane == nil {
		return inference.LaneHandle{}, core.NewError("hip.LaneSet.CommitPrepare: not a pending HIP lane")
	}
	if pending.owner != set {
		pending.Discard()
		return inference.LaneHandle{}, core.NewError("hip.LaneSet.CommitPrepare: pending lane belongs to another lane set")
	}
	if set.closed || set.executor == nil {
		pending.Discard()
		return inference.LaneHandle{}, core.NewError("hip.LaneSet.CommitPrepare: lane set is unavailable or closed")
	}
	if len(set.lanes) >= set.maxLanes {
		pending.Discard()
		return inference.LaneHandle{}, core.NewError("hip.LaneSet.CommitPrepare: lane set is at MaxLanes")
	}
	lane := pending.lane
	pending.lane = nil
	pending.owner = nil
	set.nextID++
	lane.id = set.nextID
	set.lanes[lane.id] = lane
	set.order = append(set.order, lane.id)
	return inference.LaneHandle{ID: lane.id}, nil
}

func (set *hipLaneSet) Step(ctx context.Context) ([]inference.LaneStep, error) {
	if set == nil || set.closed || set.executor == nil {
		return nil, core.NewError("hip.LaneSet.Step: lane set is unavailable or closed")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	active := set.activeLanes()
	if len(active) == 0 {
		return nil, nil
	}

	results := make([]inference.LaneStep, 0, len(active))
	survivors := make([]*hipDecodeLane, 0, len(active))
	inputs := make([]hipLaneForwardInput, 0, len(active))
	for _, lane := range active {
		generated := lane.generated + 1
		terminal := lane.stops[lane.pendingToken] || generated >= lane.maxNew
		results = append(results, inference.LaneStep{
			Lane:     inference.LaneHandle{ID: lane.id},
			Token:    lane.pendingToken,
			HasToken: true,
			Terminal: terminal,
		})
		if !terminal {
			sample := lane.sample.clone()
			if sample.enabled && sample.generate.RepeatPenalty > 1 {
				sample.history = append(sample.history, lane.pendingToken)
			}
			survivors = append(survivors, lane)
			inputs = append(inputs, hipLaneForwardInput{
				PendingToken: lane.pendingToken,
				Position:     lane.position,
				DeviceState:  lane.deviceState,
				Sample:       sample,
			})
		}
	}

	var outputs []hipLaneForwardOutput
	if len(inputs) > 0 {
		var err error
		outputs, err = set.executor.Forward(ctx, inputs)
		if err != nil {
			return nil, err
		}
		if len(outputs) != len(inputs) {
			for _, output := range outputs {
				_ = output.DeviceState.Close()
			}
			return nil, core.NewError("hip.LaneSet.Step: lane executor output count mismatch")
		}
		for index, output := range outputs {
			if output.Position != inputs[index].Position+1 {
				for _, candidate := range outputs {
					_ = candidate.DeviceState.Close()
				}
				return nil, core.NewError("hip.LaneSet.Step: lane executor position mismatch")
			}
		}
	}

	for resultIndex, lane := range active {
		lane.generated++
		lane.terminal = results[resultIndex].Terminal
	}
	for index, lane := range survivors {
		previous := lane.deviceState
		lane.pendingToken = outputs[index].PendingToken
		lane.position = outputs[index].Position
		lane.deviceState = outputs[index].DeviceState
		lane.sample = outputs[index].Sample
		hipReleaseClosedGemma4Q4DeviceDecodeState(previous)
	}
	if len(inputs) > 0 && set.executor.UsesSharedBatchForward() {
		set.fwdCount++
	}
	return results, nil
}

func (set *hipLaneSet) Retire(handle inference.LaneHandle) error {
	if set == nil || set.closed {
		return core.NewError("hip.LaneSet.Retire: lane set is unavailable or closed")
	}
	lane, ok := set.lanes[handle.ID]
	if !ok {
		return core.NewError("hip.LaneSet.Retire: unknown lane")
	}
	delete(set.lanes, handle.ID)
	for index, id := range set.order {
		if id == handle.ID {
			set.order = append(set.order[:index], set.order[index+1:]...)
			break
		}
	}
	return lane.deviceState.Close()
}

func (set *hipLaneSet) Active() int {
	if set == nil || set.closed {
		return 0
	}
	return len(set.lanes)
}

func (set *hipLaneSet) BatchForwardCount() uint64 {
	if set == nil {
		return 0
	}
	return set.fwdCount
}

func (set *hipLaneSet) Close() error {
	if set == nil || set.closed {
		return nil
	}
	set.closed = true
	var firstErr error
	for _, lane := range set.lanes {
		if err := lane.deviceState.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	set.lanes = nil
	set.order = nil
	if err := set.executor.Close(); err != nil && firstErr == nil {
		firstErr = err
	}
	set.executor = nil
	return firstErr
}

func (set *hipLaneSet) activeLanes() []*hipDecodeLane {
	lanes := make([]*hipDecodeLane, 0, len(set.order))
	for _, id := range set.order {
		if lane := set.lanes[id]; lane != nil && !lane.terminal {
			lanes = append(lanes, lane)
		}
	}
	return lanes
}

func hipLaneSampled(cfg inference.SamplerConfig) bool {
	return cfg.Temperature > 0 || cfg.MinP > 0 || cfg.RepeatPenalty > 1
}

func hipNewLaneSampleState(cfg inference.SamplerConfig, seed uint64) hipLaneSampleState {
	if !hipLaneSampled(cfg) {
		return hipLaneSampleState{}
	}
	return hipLaneSampleState{
		enabled: true,
		rng:     seed,
		generate: inference.GenerateConfig{
			Temperature:   cfg.Temperature,
			TopK:          cfg.TopK,
			TopP:          cfg.TopP,
			MinP:          cfg.MinP,
			RepeatPenalty: cfg.RepeatPenalty,
		},
	}
}

func (state hipLaneSampleState) clone() hipLaneSampleState {
	state.history = append([]int32(nil), state.history...)
	return state
}

func (state *hipLaneSampleState) draw() float64 {
	state.rng += 0x9e3779b97f4a7c15
	z := state.rng
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	z ^= z >> 31
	return float64(float32(z>>40) / float32(1<<24))
}

func hipLaneStopSet(tokens []int32) map[int32]bool {
	if len(tokens) == 0 {
		return nil
	}
	stops := make(map[int32]bool, len(tokens))
	for _, token := range tokens {
		stops[token] = true
	}
	return stops
}

type hipGemma4Q4LaneExecutor struct {
	loaded     *hipLoadedModel
	forward    hipGemma4Q4ForwardConfig
	engine     hipGemma4Q4EngineConfig
	mode       string
	workspace  *hipAttentionHeadsChunkedWorkspace
	contextLen int
	closed     bool
}

func (executor *hipGemma4Q4LaneExecutor) Prepare(ctx context.Context, spec inference.LaneSpec) (prepared hipPreparedLane, runErr error) {
	if executor == nil || executor.closed || executor.loaded == nil {
		return hipPreparedLane{}, core.NewError("hip.LaneSet.Prepare: Gemma4 lane executor is unavailable or closed")
	}
	if len(spec.PromptIDs) > executor.contextLen {
		return hipPreparedLane{}, core.NewError("hip.LaneSet.Prepare: prompt exceeds model context window")
	}
	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	defer func() {
		if err := hipRecycleAttentionHeadsChunkedWorkspace(workspace); err != nil && runErr == nil {
			_ = prepared.DeviceState.Close()
			prepared = hipPreparedLane{}
			runErr = err
		}
	}()
	sample := hipNewLaneSampleState(spec.Sampler, spec.SampleSeed)
	prefill, err := hipRunAttachedDrafterTargetPrefill(ctx, executor.loaded.driver, hipAttachedDrafterTargetPrefillRequest{
		TargetForward: executor.forward,
		DeviceKVMode:  executor.mode,
		EngineConfig:  executor.engine,
		InputTokenIDs: spec.PromptIDs,
		Epsilon:       1e-6,
		Workspace:     workspace,
	})
	if err != nil {
		return hipPreparedLane{}, err
	}
	defer hipReleaseForwardDeviceFinalHidden(&prefill.Current)
	if prefill.DeviceState == nil || prefill.Current.DeviceFinalHidden == nil {
		_ = prefill.DeviceState.Close()
		return hipPreparedLane{}, core.NewError("hip.LaneSet.Prepare: prompt prefill produced no pending hidden or device state")
	}
	pending := int32(prefill.Current.Greedy.TokenID)
	if sample.enabled {
		picked, sampleErr := executor.sampleHiddenRowWithWorkspace(ctx, prefill.Current.DeviceFinalHidden, 1, 0, &sample, workspace)
		if sampleErr != nil {
			_ = prefill.DeviceState.Close()
			return hipPreparedLane{}, sampleErr
		}
		pending = picked
	}
	return hipPreparedLane{PendingToken: pending, Position: prefill.Position, DeviceState: prefill.DeviceState, Sample: sample}, nil
}

func (executor *hipGemma4Q4LaneExecutor) Forward(ctx context.Context, inputs []hipLaneForwardInput) ([]hipLaneForwardOutput, error) {
	if executor == nil || executor.closed || executor.loaded == nil || executor.workspace == nil {
		return nil, core.NewError("hip.LaneSet.Step: Gemma4 lane executor is unavailable or closed")
	}
	if len(inputs) == 0 {
		return nil, nil
	}
	tokens := make([]int32, len(inputs))
	positions := make([]int, len(inputs))
	states := make([]*hipGemma4Q4DeviceDecodeState, len(inputs))
	for index, input := range inputs {
		if input.Position >= executor.contextLen {
			return nil, core.NewError("hip.LaneSet.Step: lane reached model context window")
		}
		tokens[index] = input.PendingToken
		positions[index] = input.Position
		states[index] = input.DeviceState
	}
	batch, err := hipRunGemma4Q4LaneForward(ctx, executor.loaded.driver, executor.forward, hipGemma4Q4LaneForwardRequest{
		Tokens:       tokens,
		Positions:    positions,
		DeviceStates: states,
		Epsilon:      1e-6,
		Mode:         executor.mode,
		Workspace:    executor.workspace,
		EngineConfig: executor.engine,
	})
	if err != nil {
		return nil, err
	}
	if batch == nil || len(batch.Greedy) != len(inputs) || len(batch.DeviceStates) != len(inputs) {
		if batch != nil {
			for _, state := range batch.DeviceStates {
				_ = state.Close()
			}
		}
		return nil, core.NewError("hip.LaneSet.Step: Gemma4 lane forward output count mismatch")
	}
	outputs := make([]hipLaneForwardOutput, len(inputs))
	for index := range inputs {
		sample := inputs[index].Sample.clone()
		pending := int32(batch.Greedy[index].TokenID)
		if sample.enabled {
			pending, err = executor.sampleHiddenRow(ctx, batch.FinalHidden, len(inputs), index, &sample)
			if err != nil {
				for _, state := range batch.DeviceStates {
					_ = state.Close()
				}
				return nil, err
			}
		}
		outputs[index] = hipLaneForwardOutput{
			PendingToken: pending,
			Position:     inputs[index].Position + 1,
			DeviceState:  batch.DeviceStates[index],
			Sample:       sample,
		}
	}
	return outputs, nil
}

func (executor *hipGemma4Q4LaneExecutor) sampleHiddenRow(ctx context.Context, hidden *hipDeviceByteBuffer, rows, row int, sample *hipLaneSampleState) (int32, error) {
	return executor.sampleHiddenRowWithWorkspace(ctx, hidden, rows, row, sample, executor.workspace)
}

func (executor *hipGemma4Q4LaneExecutor) sampleHiddenRowWithWorkspace(ctx context.Context, hidden *hipDeviceByteBuffer, rows, row int, sample *hipLaneSampleState, workspace *hipAttentionHeadsChunkedWorkspace) (int32, error) {
	if sample == nil || !sample.enabled {
		return 0, core.NewError("hip.LaneSet.Sample: sampled lane state is required")
	}
	if workspace == nil {
		return 0, core.NewError("hip.LaneSet.Sample: attention workspace is required")
	}
	last := executor.forward.Layers[len(executor.forward.Layers)-1]
	result, err := hipGemma4Q4SampleBatchedPrefillRow(
		ctx, executor.loaded.driver, last, executor.forward.HeadLoRA, hidden, rows, row, 1e-6,
		sample.generate, nil, sample.history, sample.draw(), nil, workspace, false,
	)
	if err != nil {
		return 0, err
	}
	return int32(result.Greedy.TokenID), nil
}

func (executor *hipGemma4Q4LaneExecutor) UsesSharedBatchForward() bool {
	return executor != nil
}

func (executor *hipGemma4Q4LaneExecutor) Close() error {
	if executor == nil || executor.closed {
		return nil
	}
	executor.closed = true
	if executor.workspace == nil {
		return nil
	}
	workspace := executor.workspace
	executor.workspace = nil
	return workspace.Close()
}
