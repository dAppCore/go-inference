// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// decode_forward_arch_icb_q8.go — q8 KV on the recorded-ICB lane (#367 slice
// B, OPT-IN via LTHN_KV_Q8_ICB=1): the GLOBAL owner layers' caches store int8
// codes + f32 group scales instead of bf16, halving the bytes of the
// unbounded deep-context KV scan (the measured depth slope). Sliding layers
// stay bf16 — their scan is window-capped and their ring landing untouched.
//
// The recorded shape per q8 global layer: the K rope/norm and the V
// projection land into FIXED bf16 staging rows (no per-token rebind), then
// one lthn_kv_q8_store_bf16 op each quantises staging → the int8 cache row +
// scale row (those two ops' output offsets are what prepareStepRebind now
// rebinds), and the layer's SDPA reads through the q8 kernel pair
// (lthn_sdpa_vector_q8 / _2pass_1_q8) with the scale planes bound.
//
// V1 coherence: lanes that would write bf16 into (or read bf16 from) the q8
// caches DECLINE — the batched dense pass falls back to the per-token replay
// (prompt prefill + MTP verify go sequential), and KV save/restore errors.
// The chained replay lane IS the decode; it is fully q8-aware by recording.

// kvQ8ICBEnabled is the opt-in gate (LTHN_KV_Q8_ICB=1). Overridable in tests
// via kvQ8ICBForTest (the env is read once at init).
var (
	kvQ8ICBEnabled = os.Getenv("LTHN_KV_Q8_ICB") == "1"
	kvQ8ICBForTest bool
)

func kvQ8ICBOn() bool { return kvQ8ICBEnabled || kvQ8ICBForTest }

// archICBKVQ8 carries the per-layer q8 KV state threaded from the session
// constructor (which owns the cache allocation) into the recorder and replay.
type archICBKVQ8 struct {
	enabled          []bool
	kScales, vScales []metal.MTLBuffer
}

func (q *archICBKVQ8) on(li int) bool {
	return q != nil && li < len(q.enabled) && q.enabled[li]
}

// hasKVQ8 reports whether this replay carries q8 KV layers — the v1 decline
// gate for lanes that would read or write the int8 caches with bf16 byte
// arithmetic (batched dense, KV save/restore).
func (r *archICBReplay) hasKVQ8() bool { return r != nil && r.kvQ8.any() }

func (q *archICBKVQ8) any() bool {
	if q == nil {
		return false
	}
	for _, e := range q.enabled {
		if e {
			return true
		}
	}
	return false
}

// allocArchICBCaches allocates the recorded replay's per-owner KV caches —
// the ONE allocation loop both session constructors share. Sliding owners
// get the bounded ring (SlidingWindow rows); global owners get maxLen rows.
// Under the q8 opt-in, qualifying GLOBAL owners (q8 SDPA instantiations
// exist for hd 256/512; kvDim must be whole 64-groups) allocate int8 caches
// + f32 scale planes instead of bf16, and the returned carrier marks them.
func allocArchICBCaches(specs []model.LayerSpec, nKVHeads, headDim, maxLen, slidingWindow int) (kCaches, vCaches []metal.MTLBuffer, kvQ8 *archICBKVQ8) {
	kCaches = make([]metal.MTLBuffer, len(specs))
	vCaches = make([]metal.MTLBuffer, len(specs))
	q8 := &archICBKVQ8{
		enabled: make([]bool, len(specs)),
		kScales: make([]metal.MTLBuffer, len(specs)),
		vScales: make([]metal.MTLBuffer, len(specs)),
	}
	for li := range specs {
		if !specs[li].OwnsCache() {
			continue
		}
		lkv, lhd := kvHeadsOf(specs[li], nKVHeads), headDimOf(specs[li], headDim)
		kvd := lkv * lhd
		cacheLen := maxLen
		if slidingWindow > 0 && slidingWindow < maxLen && specs[li].Attention != model.GlobalAttention {
			// Bounded ring — the sliding-window KV memory fix: a sliding owner
			// only ever attends its own window, so it only needs SlidingWindow
			// rows (O(window) not O(context)); prepareStepRebind detects the
			// smaller allocation and ring-writes pos%cacheRows.
			cacheLen = slidingWindow
		}
		if kvQ8ICBOn() && specs[li].Attention == model.GlobalAttention &&
			kvd%kvQ8GroupSize == 0 && (lhd == 256 || lhd == 512) {
			kCaches[li] = device.NewBufferWithLengthOptions(uint(cacheLen*kvd), metal.MTLResourceStorageModeShared)
			vCaches[li] = device.NewBufferWithLengthOptions(uint(cacheLen*kvd), metal.MTLResourceStorageModeShared)
			q8.kScales[li] = device.NewBufferWithLengthOptions(uint(cacheLen*(kvd/kvQ8GroupSize)*4), metal.MTLResourceStorageModeShared)
			q8.vScales[li] = device.NewBufferWithLengthOptions(uint(cacheLen*(kvd/kvQ8GroupSize)*4), metal.MTLResourceStorageModeShared)
			q8.enabled[li] = true
			continue
		}
		cacheBytes := uint(cacheLen * kvd * bf16Size)
		kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
		vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
	}
	if !q8.any() {
		return kCaches, vCaches, nil
	}
	return kCaches, vCaches, q8
}

// ---- ICB-capable pipelines (a kernel recorded into an indirect command
// faults without supportIndirectCommandBuffers) ----

var (
	kvQ8ICBPSOMu        sync.Mutex
	kvQ8StoreICBPSO     metal.MTLComputePipelineState
	kvQ8StoreICBErr     error
	kvQ8StoreICBDone    bool
	sdpaQ8ICBPSOCache   = map[int]metal.MTLComputePipelineState{}
	sdpaQ8P1ICBPSOCache = map[[2]int]metal.MTLComputePipelineState{}
)

func kvQ8StorePipelineICB() (metal.MTLComputePipelineState, error) {
	kvQ8ICBPSOMu.Lock()
	defer kvQ8ICBPSOMu.Unlock()
	if kvQ8StoreICBDone {
		return kvQ8StoreICBPSO, kvQ8StoreICBErr
	}
	kvQ8StoreICBDone = true
	if customLibrary == nil || customLibrary.GetID() == 0 {
		kvQ8StoreICBErr = core.NewError("native.kvQ8StorePipelineICB: custom library unavailable")
		return nil, kvQ8StoreICBErr
	}
	fn := customLibrary.NewFunctionWithName("lthn_kv_q8_store_bf16")
	if fn == nil || fn.GetID() == 0 {
		kvQ8StoreICBErr = core.NewError("native.kvQ8StorePipelineICB: kernel lthn_kv_q8_store_bf16 not found")
		return nil, kvQ8StoreICBErr
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	kvQ8StoreICBPSO, kvQ8StoreICBErr = device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	return kvQ8StoreICBPSO, kvQ8StoreICBErr
}

func sdpaVectorQ8PipelineICB(headDim int) (metal.MTLComputePipelineState, error) {
	kvQ8ICBPSOMu.Lock()
	defer kvQ8ICBPSOMu.Unlock()
	if pso, ok := sdpaQ8ICBPSOCache[headDim]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.sdpaVectorQ8PipelineICB: custom library unavailable")
	}
	fn := customLibrary.NewFunctionWithName(sdpaVectorQ8KernelName(headDim))
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.sdpaVectorQ8PipelineICB: kernel " + sdpaVectorQ8KernelName(headDim) + " not found")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, err
	}
	sdpaQ8ICBPSOCache[headDim] = pso
	return pso, nil
}

func sdpaVector2Pass1Q8PipelineICB(headDim int, blocks int32) (metal.MTLComputePipelineState, error) {
	key := [2]int{headDim, int(blocks)}
	kvQ8ICBPSOMu.Lock()
	defer kvQ8ICBPSOMu.Unlock()
	if pso, ok := sdpaQ8P1ICBPSOCache[key]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass1Q8PipelineICB: custom library unavailable")
	}
	fc := metal.NewMTLFunctionConstantValues()
	blk := blocks
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&blk), metal.MTLDataTypeInt, 26)
	name := core.Sprintf("lthn_sdpa_vector_2pass_1_q8_bf16_%d", headDim)
	fn, err := customLibrary.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil || fn == nil || fn.GetID() == 0 {
		return nil, core.E("native.sdpaVector2Pass1Q8PipelineICB", name, err)
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, perr := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if perr != nil {
		return nil, perr
	}
	sdpaQ8P1ICBPSOCache[key] = pso
	return pso, nil
}

// emitKVQ8Store records the staging→cache quantise hop through any sink:
// row(bf16)=0, out(int8)=1 (rebound per token), scales(f32)=2 (rebound per
// token), kvDim=3 (the kvQ8StoreDims struct is one uint32 — a memoised
// scalar buffer serves it on the ICB). One 32-lane threadgroup per 64-group.
func emitKVQ8Store[S dispatchSink](sink S, pso metal.MTLComputePipelineState, row metal.MTLBuffer, out metal.MTLBuffer, outOff uint, scales metal.MTLBuffer, scaleOff uint, kvDim int) {
	sink.setPSO(pso)
	sink.setBuf(row, 0, 0)
	sink.setBuf(out, outOff, 1)
	sink.setBuf(scales, scaleOff, 2)
	sink.setI32(int32(kvDim), 3)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(kvDim / kvQ8GroupSize), Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
}
