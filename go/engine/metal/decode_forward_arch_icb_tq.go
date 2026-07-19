// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// decode_forward_arch_icb_tq.go — TurboQuant KV on the recorded-ICB lane
// (campaign #41 S3, OPT-IN via -kv-cache turboquant[:N]): qualifying GLOBAL
// owner layers' caches store packed Lloyd-Max centroid codes + per-row-per-head
// f32 norms (the kv/turboquant Q_mse wire format — turboquant_device.go is the
// format authority) instead of bf16, cutting the unbounded deep-context KV
// residency to headDim·bits/8 (+4) bytes per row per head. Sliding-window and
// recurrent layers keep their existing bf16 paths — their residency is bounded.
//
// The recorded shape per TQ global layer: the K rope/norm and the V projection
// land into FIXED bf16 staging rows, one lthn_tq_kv_store op each rotates +
// quantises staging → the code cache row + γ row (those two stores' output
// offsets are what prepareStepRebind rebinds per token), and the layer's SDPA
// reads codes through the TQ kernel pair with q pre-rotated once per step and
// the output unrotated once per step (lthn_tq_rot_rows — the O(output) fold).
//
// V1 coherence: every lane that would read or write these caches as bf16 rows
// DECLINES — the batched dense pass falls back to the per-token replay (prompt
// prefill goes sequential), prompt reuse falls back to the whole prefill, KV
// snapshot / -state sleep-wake / the MTP pairing / paged KV / the laneSet /
// the submit-ahead peer refuse loudly. The chained replay lane IS the decode;
// it is fully TQ-aware by recording.

// archICBKVTQ carries the per-layer TurboQuant KV state threaded from the
// session constructor (which owns the cache allocation) into the recorder and
// replay. A layer is EITHER TurboQuant or q8, never both — under a TQ mode the
// q8 ladder is fully off (allocArchICBCachesTQ).
type archICBKVTQ struct {
	enabled          []bool
	kBits, vBits     int
	kGammas, vGammas []metal.MTLBuffer // per-layer [cacheRows × kvHeads] f32 norm planes
	kRowBytes        []int             // per-layer K code-cache row stride (kvHeads·ceil(hd·kBits/8))
	vRowBytes        []int             // per-layer V code-cache row stride (kvHeads·ceil(hd·vBits/8))
	gammaRowBytes    []int             // per-layer γ-plane row stride (kvHeads·4)
}

func (q *archICBKVTQ) on(li int) bool {
	return q != nil && li < len(q.enabled) && q.enabled[li]
}

func (q *archICBKVTQ) any() bool {
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

// hasKVTQ reports whether this replay carries TurboQuant KV layers — the v1
// decline gate for every lane that would touch the code caches as bf16 bytes.
func (r *archICBReplay) hasKVTQ() bool { return r != nil && r.kvTQ.any() }

// releaseTQPlanes frees the per-layer γ planes (the code caches themselves are
// kCaches/vCaches, released by releaseKVCaches' shared loop). Π and centroid
// tables are process-memoised and shared across sessions — never released here.
func (r *archICBReplay) releaseTQPlanes() {
	if r == nil || r.kvTQ == nil {
		return
	}
	releaseDeviceBuffers(r.kvTQ.kGammas...)
	releaseDeviceBuffers(r.kvTQ.vGammas...)
	for i := range r.kvTQ.kGammas {
		r.kvTQ.kGammas[i] = nil
	}
	for i := range r.kvTQ.vGammas {
		r.kvTQ.vGammas[i] = nil
	}
}

// allocArchICBCachesTQ is allocArchICBCaches with the TurboQuant mode applied:
// tq == nil delegates to the existing q8 ladder unchanged (byte-identical
// off-path); under a TQ mode, qualifying GLOBAL owners (instantiated head dim,
// no sliding sharer — a sliding read offset cannot address packed code rows)
// allocate code caches + γ planes, every other owner allocates bf16, and q8 is
// fully off.
func allocArchICBCachesTQ(specs []model.LayerSpec, nKVHeads, headDim, maxLen, slidingWindow int, tq *tqKVConfig) (kCaches, vCaches []metal.MTLBuffer, kvQ8 *archICBKVQ8, kvTQ *archICBKVTQ) {
	if tq == nil {
		kCaches, vCaches, kvQ8 = allocArchICBCaches(specs, nKVHeads, headDim, maxLen, slidingWindow)
		return kCaches, vCaches, kvQ8, nil
	}
	// A sliding sharer rebinds its SDPA read at a bf16 byte offset into the
	// owner's cache (prepareStepRebind) — unaddressable over packed codes, so
	// such an owner stays bf16.
	slidingSharer := make([]bool, len(specs))
	for li := range specs {
		if specs[li].Attention == model.SlidingAttention && specs[li].KVShareFrom != li {
			slidingSharer[specs[li].KVShareFrom] = true
		}
	}
	kCaches = make([]metal.MTLBuffer, len(specs))
	vCaches = make([]metal.MTLBuffer, len(specs))
	tqc := &archICBKVTQ{
		enabled:       make([]bool, len(specs)),
		kBits:         tq.kBits,
		vBits:         tq.vBits,
		kGammas:       make([]metal.MTLBuffer, len(specs)),
		vGammas:       make([]metal.MTLBuffer, len(specs)),
		kRowBytes:     make([]int, len(specs)),
		vRowBytes:     make([]int, len(specs)),
		gammaRowBytes: make([]int, len(specs)),
	}
	for li := range specs {
		if !specs[li].OwnsCache() {
			continue
		}
		lkv, lhd := kvHeadsOf(specs[li], nKVHeads), headDimOf(specs[li], headDim)
		kvd := lkv * lhd
		cacheLen := maxLen
		if slidingWindow > 0 && slidingWindow < maxLen && specs[li].Attention != model.GlobalAttention {
			cacheLen = slidingWindow // bounded ring, exactly as the q8/bf16 ladder
		}
		if specs[li].Attention == model.GlobalAttention && !slidingSharer[li] &&
			tqKVGeometryOK(tq.kBits, tq.vBits, lhd) {
			kRow := lkv * tqBytesPerRow(tq.kBits, lhd)
			vRow := lkv * tqBytesPerRow(tq.vBits, lhd)
			kCaches[li] = device.NewBufferWithLengthOptions(uint(cacheLen*kRow), metal.MTLResourceStorageModeShared)
			vCaches[li] = device.NewBufferWithLengthOptions(uint(cacheLen*vRow), metal.MTLResourceStorageModeShared)
			tqc.kGammas[li] = device.NewBufferWithLengthOptions(uint(cacheLen*lkv*4), metal.MTLResourceStorageModeShared)
			tqc.vGammas[li] = device.NewBufferWithLengthOptions(uint(cacheLen*lkv*4), metal.MTLResourceStorageModeShared)
			tqc.kRowBytes[li] = kRow
			tqc.vRowBytes[li] = vRow
			tqc.gammaRowBytes[li] = lkv * 4
			tqc.enabled[li] = true
			continue
		}
		cacheBytes := uint(cacheLen * kvd * bf16Size)
		kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
		vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
	}
	if !tqc.any() {
		return kCaches, vCaches, nil, nil
	}
	return kCaches, vCaches, nil, tqc
}

// ---- ICB-capable pipelines (a kernel recorded into an indirect command
// faults without supportIndirectCommandBuffers) ----

var (
	tqKVICBPSOMu      sync.Mutex
	tqKVStoreICBPSOs  = map[int]metal.MTLComputePipelineState{}
	tqRotRowsICBPSOs  = map[bool]metal.MTLComputePipelineState{}
	sdpaTQICBPSOCache = map[[3]int]metal.MTLComputePipelineState{}
	sdpaTQP1ICBPSOs   = map[[4]int]metal.MTLComputePipelineState{}
)

func tqKVStorePipelineICB(bits int) (metal.MTLComputePipelineState, error) {
	if !tqKVBitsOK(bits) {
		return nil, core.NewError("native.tqKVStorePipelineICB: unsupported bit width")
	}
	tqKVICBPSOMu.Lock()
	defer tqKVICBPSOMu.Unlock()
	if pso, ok := tqKVStoreICBPSOs[bits]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.tqKVStorePipelineICB: custom library unavailable")
	}
	name := core.Sprintf("lthn_tq_kv_store_bf16_b%d", bits)
	fn := customLibrary.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.tqKVStorePipelineICB: kernel " + name + " not found")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, err
	}
	tqKVStoreICBPSOs[bits] = pso
	return pso, nil
}

func tqRotRowsPipelineICB(transpose bool) (metal.MTLComputePipelineState, error) {
	tqKVICBPSOMu.Lock()
	defer tqKVICBPSOMu.Unlock()
	if pso, ok := tqRotRowsICBPSOs[transpose]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.tqRotRowsPipelineICB: custom library unavailable")
	}
	name := "lthn_tq_rot_rows_bf16"
	if transpose {
		name = "lthn_tq_unrot_rows_bf16"
	}
	fn := customLibrary.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.tqRotRowsPipelineICB: kernel " + name + " not found")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, err
	}
	tqRotRowsICBPSOs[transpose] = pso
	return pso, nil
}

func sdpaVectorTQPipelineICB(headDim, kBits, vBits int) (metal.MTLComputePipelineState, error) {
	if !tqKVGeometryOK(kBits, vBits, headDim) {
		return nil, core.NewError("native.sdpaVectorTQPipelineICB: unsupported (headDim, kBits, vBits)")
	}
	key := [3]int{headDim, kBits, vBits}
	tqKVICBPSOMu.Lock()
	defer tqKVICBPSOMu.Unlock()
	if pso, ok := sdpaTQICBPSOCache[key]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.sdpaVectorTQPipelineICB: custom library unavailable")
	}
	name := core.Sprintf("lthn_sdpa_vector_tq_bf16_%d", headDim)
	fn, err := customLibrary.NewFunctionWithNameConstantValuesError(name, tqSDPAFunctionConstants(0, int32(kBits), int32(vBits), false))
	if err != nil || fn == nil || fn.GetID() == 0 {
		return nil, core.E("native.sdpaVectorTQPipelineICB", name, err)
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, perr := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if perr != nil {
		return nil, perr
	}
	sdpaTQICBPSOCache[key] = pso
	return pso, nil
}

func sdpaVector2Pass1TQPipelineICB(headDim, kBits, vBits int, blocks int32) (metal.MTLComputePipelineState, error) {
	if !tqKVGeometryOK(kBits, vBits, headDim) {
		return nil, core.NewError("native.sdpaVector2Pass1TQPipelineICB: unsupported (headDim, kBits, vBits)")
	}
	key := [4]int{headDim, kBits, vBits, int(blocks)}
	tqKVICBPSOMu.Lock()
	defer tqKVICBPSOMu.Unlock()
	if pso, ok := sdpaTQP1ICBPSOs[key]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass1TQPipelineICB: custom library unavailable")
	}
	name := core.Sprintf("lthn_sdpa_vector_2pass_1_tq_bf16_%d", headDim)
	fn, err := customLibrary.NewFunctionWithNameConstantValuesError(name, tqSDPAFunctionConstants(blocks, int32(kBits), int32(vBits), true))
	if err != nil || fn == nil || fn.GetID() == 0 {
		return nil, core.E("native.sdpaVector2Pass1TQPipelineICB", name, err)
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, perr := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if perr != nil {
		return nil, perr
	}
	sdpaTQP1ICBPSOs[key] = pso
	return pso, nil
}
