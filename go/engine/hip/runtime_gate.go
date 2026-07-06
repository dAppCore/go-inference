// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"maps"
	"slices"
	"sync"

	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

// ROCmRuntimeGateID names a live ROCm runtime fast-path gate. The identifiers
// mirror the model package's route-plan IDs while keeping the mutable state in
// the root runtime package.
type ROCmRuntimeGateID = rocmmodel.RuntimeGateID

const (
	ROCmGateDirectGreedyToken           ROCmRuntimeGateID = rocmmodel.GateDirectGreedyToken
	ROCmGateNativeMLPMatVec             ROCmRuntimeGateID = rocmmodel.GateNativeMLPMatVec
	ROCmGateNativeLinearMatVec          ROCmRuntimeGateID = rocmmodel.GateNativeLinearMatVec
	ROCmGateNativeQ6BitstreamMatVec     ROCmRuntimeGateID = rocmmodel.GateNativeQ6BitstreamMatVec
	ROCmGateNativeAttentionOMatVec      ROCmRuntimeGateID = rocmmodel.GateNativeAttentionOMatVec
	ROCmGateGenerationStream            ROCmRuntimeGateID = rocmmodel.GateGenerationStream
	ROCmGateAsyncDecodePrefetch         ROCmRuntimeGateID = rocmmodel.GateAsyncDecodePrefetch
	ROCmGateFixedSlidingCache           ROCmRuntimeGateID = rocmmodel.GateFixedSlidingCache
	ROCmGateFixedSlidingCacheBound      ROCmRuntimeGateID = rocmmodel.GateFixedSlidingCacheBound
	ROCmGateFixedSharedMask             ROCmRuntimeGateID = rocmmodel.GateFixedSharedMask
	ROCmGateNativeFixedSlidingAttention ROCmRuntimeGateID = rocmmodel.GateNativeFixedSlidingAttention
	ROCmGatePagedDecodeFastConcat       ROCmRuntimeGateID = rocmmodel.GatePagedDecodeFastConcat
	ROCmGateNativePagedAttention        ROCmRuntimeGateID = rocmmodel.GateNativePagedAttention
	ROCmGateCacheOnlyChunkPrefill       ROCmRuntimeGateID = rocmmodel.GateCacheOnlyChunkPrefill
	ROCmGateSortedExpertPrefill         ROCmRuntimeGateID = rocmmodel.GateSortedExpertPrefill
	ROCmGateGatherQMMReferenceTests     ROCmRuntimeGateID = rocmmodel.GateGatherQMMReferenceTests
	ROCmGateCompiledMLPDecode           ROCmRuntimeGateID = rocmmodel.GateCompiledMLPDecode
	ROCmGateCompiledLayerDecode         ROCmRuntimeGateID = rocmmodel.GateCompiledLayerDecode
	ROCmGatePipelinedDecode             ROCmRuntimeGateID = rocmmodel.GatePipelinedDecode
	ROCmGateFixedWideSDPAAttention      ROCmRuntimeGateID = rocmmodel.GateFixedWideSDPAAttention
)

var rocmRuntimeGateIDs = []ROCmRuntimeGateID{
	ROCmGateDirectGreedyToken,
	ROCmGateNativeMLPMatVec,
	ROCmGateNativeLinearMatVec,
	ROCmGateNativeQ6BitstreamMatVec,
	ROCmGateNativeAttentionOMatVec,
	ROCmGateGenerationStream,
	ROCmGateAsyncDecodePrefetch,
	ROCmGateFixedSlidingCache,
	ROCmGateFixedSlidingCacheBound,
	ROCmGateFixedSharedMask,
	ROCmGateNativeFixedSlidingAttention,
	ROCmGatePagedDecodeFastConcat,
	ROCmGateNativePagedAttention,
	ROCmGateCacheOnlyChunkPrefill,
	ROCmGateSortedExpertPrefill,
	ROCmGateGatherQMMReferenceTests,
	ROCmGateCompiledMLPDecode,
	ROCmGateCompiledLayerDecode,
	ROCmGatePipelinedDecode,
	ROCmGateFixedWideSDPAAttention,
}

var (
	rocmRuntimeGateMu     sync.RWMutex
	rocmRuntimeGateStates = newROCmRuntimeGateState()
)

func newROCmRuntimeGateState() map[ROCmRuntimeGateID]bool {
	states := make(map[ROCmRuntimeGateID]bool, len(rocmRuntimeGateIDs))
	for _, gate := range rocmRuntimeGateIDs {
		states[gate] = false
	}
	return states
}

// ROCmRuntimeGateIDs returns the known runtime gates in stable order.
func ROCmRuntimeGateIDs() []ROCmRuntimeGateID {
	return append([]ROCmRuntimeGateID(nil), rocmRuntimeGateIDs...)
}

// SetROCmRuntimeGate turns a typed runtime gate on or off and returns a restore
// function that reinstates the previous value. Unknown gates are ignored.
func SetROCmRuntimeGate(gate ROCmRuntimeGateID, on bool) func() {
	rocmRuntimeGateMu.Lock()
	previous, ok := rocmRuntimeGateStates[gate]
	if !ok {
		rocmRuntimeGateMu.Unlock()
		return func() {}
	}
	rocmRuntimeGateStates[gate] = on
	rocmRuntimeGateMu.Unlock()
	return func() {
		rocmRuntimeGateMu.Lock()
		if _, ok := rocmRuntimeGateStates[gate]; ok {
			rocmRuntimeGateStates[gate] = previous
		}
		rocmRuntimeGateMu.Unlock()
	}
}

// ROCmRuntimeGateEnabled reports whether a typed runtime gate is currently on.
func ROCmRuntimeGateEnabled(gate ROCmRuntimeGateID) bool {
	rocmRuntimeGateMu.RLock()
	enabled := rocmRuntimeGateStates[gate]
	rocmRuntimeGateMu.RUnlock()
	return enabled
}

// ROCmRuntimeGateSnapshot returns a defensive copy of the current live gate map.
func ROCmRuntimeGateSnapshot() map[ROCmRuntimeGateID]bool {
	rocmRuntimeGateMu.RLock()
	defer rocmRuntimeGateMu.RUnlock()
	out := make(map[ROCmRuntimeGateID]bool, len(rocmRuntimeGateStates))
	maps.Copy(out, rocmRuntimeGateStates)
	return out
}

// EnabledRuntimeGates returns the typed gates enabled by these engine features.
func (features ROCmEngineFeatures) EnabledRuntimeGates() []ROCmRuntimeGateID {
	gates := make([]ROCmRuntimeGateID, 0, 16)
	add := func(gate ROCmRuntimeGateID, enabled bool) {
		if enabled {
			gates = append(gates, gate)
		}
	}
	add(ROCmGateDirectGreedyToken, features.DirectGreedyToken)
	add(ROCmGateNativeMLPMatVec, features.NativeMLPMatVec)
	add(ROCmGateNativeLinearMatVec, features.NativeLinearMatVec)
	add(ROCmGateNativeQ6BitstreamMatVec, features.NativeQ6BitstreamMatVec)
	add(ROCmGateNativeAttentionOMatVec, features.NativeAttentionOMatVec)
	add(ROCmGateGenerationStream, features.GenerationStream)
	add(ROCmGateAsyncDecodePrefetch, features.AsyncDecodePrefetch)
	add(ROCmGateFixedSlidingCache, features.FixedSlidingCache)
	add(ROCmGateFixedSlidingCacheBound, features.FixedSlidingCacheBound)
	add(ROCmGateNativeFixedSlidingAttention, features.NativeFixedSlidingAttention)
	add(ROCmGateCompiledLayerDecode, features.CompiledLayerDecode)
	add(ROCmGatePipelinedDecode, features.PipelinedDecode)
	return gates
}

// ApplyRuntimeGates turns on the runtime gates declared by this feature set and
// returns a restore function. Disabled fields are untouched, matching go-mlx's
// additive model-owned EngineFeatures.Apply contract.
func (features ROCmEngineFeatures) ApplyRuntimeGates() func() {
	return ApplyROCmRuntimeGates(features.EnabledRuntimeGates())
}

// ApplyROCmRuntimeFeaturesForModel applies a loaded model's declared runtime
// features and route-plan gates. The returned restore is useful for tests and
// probes; production load paths intentionally keep the gates for process life.
func ApplyROCmRuntimeFeaturesForModel(model any) func() {
	restores := make([]func(), 0, 2)
	if features, ok := ROCmEngineFeaturesFor(model); ok {
		restores = append(restores, features.ApplyRuntimeGates())
	}
	if reporter, ok := model.(ROCmModelRoutePlanReporter); ok {
		plan := reporter.ModelRoutePlan()
		if plan.RuntimeGatePlan.Matched() {
			restores = append(restores, ApplyROCmRuntimeGatePlan(plan.RuntimeGatePlan))
		}
	} else if reporter, ok := model.(ROCmModelProfileReporter); ok {
		profile := reporter.ModelProfile()
		if profile.Matched() {
			plan := ROCmModelRoutePlanForProfile(profile)
			if plan.RuntimeGatePlan.Matched() {
				restores = append(restores, ApplyROCmRuntimeGatePlan(plan.RuntimeGatePlan))
			}
		}
	}
	return func() {
		for _, restore := range slices.Backward(restores) {
			restore()
		}
	}
}

// ApplyROCmRuntimeGatePlan turns on every enabled gate in plan and returns a
// restore function. The plan is metadata-only until explicitly applied here.
func ApplyROCmRuntimeGatePlan(plan rocmmodel.RuntimeGatePlan) func() {
	if !plan.Matched() {
		return func() {}
	}
	gates := make([]ROCmRuntimeGateID, 0, len(plan.Gates)+len(plan.GateIDs))
	seen := map[ROCmRuntimeGateID]bool{}
	for _, gate := range plan.Gates {
		if gate.Enabled && !seen[gate.ID] {
			seen[gate.ID] = true
			gates = append(gates, gate.ID)
		}
	}
	if len(gates) == 0 {
		for _, gate := range plan.GateIDs {
			if !seen[gate] {
				seen[gate] = true
				gates = append(gates, gate)
			}
		}
	}
	return ApplyROCmRuntimeGates(gates)
}

// ApplyROCmRuntimeGates turns on gates in order and returns a restore function.
func ApplyROCmRuntimeGates(gates []ROCmRuntimeGateID) func() {
	restores := make([]func(), 0, len(gates))
	for _, gate := range gates {
		restores = append(restores, SetROCmRuntimeGate(gate, true))
	}
	return func() {
		for _, restore := range slices.Backward(restores) {
			restore()
		}
	}
}
