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
	// bf16 snapshot mirrors (lazily allocated; -state/save/restore only) —
	// see q8SnapshotMirror.
	kMirrors, vMirrors []metal.MTLBuffer
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

// ---- -state / snapshot support (#367 slice D): q8 layers expose a bf16
// MIRROR through snapshotCacheViews, so every save/restore path — the -state
// sleeps, CaptureKV/RestoreKV, block and TurboQuant payload restores — reads
// and writes the SAME bf16-shaped bytes a bf16 session would (snapshots stay
// portable in both directions). The symmetric per-64 codec makes the
// dequantise→requantise round trip an exact identity, so a q8 sleep/wake is
// byte-lossless against the live q8 state.

// q8SnapshotMirror returns layer li's bf16 mirror, freshly dequantised from
// the int8 cache + scale plane. The mirror is allocated on first use and
// retained (state-using sessions only pay for it once).
func (r *archICBReplay) q8SnapshotMirror(li int) (metal.MTLBuffer, *byte, error) {
	if r == nil || r.kvQ8 == nil || !r.kvQ8.on(li) {
		return nil, nil, core.NewError("native.q8SnapshotMirror: not a q8 layer")
	}
	kvd := r.rowBytes[li] / bf16Size
	rows := r.cacheRows[li]
	if kvd <= 0 || rows <= 0 {
		return nil, nil, core.NewError("native.q8SnapshotMirror: bad q8 layer geometry")
	}
	if r.kvQ8.kMirrors == nil {
		r.kvQ8.kMirrors = make([]metal.MTLBuffer, len(r.kvQ8.enabled))
		r.kvQ8.vMirrors = make([]metal.MTLBuffer, len(r.kvQ8.enabled))
	}
	if r.kvQ8.kMirrors[li] == nil {
		r.kvQ8.kMirrors[li] = device.NewBufferWithLengthOptions(uint(rows*kvd*bf16Size), metal.MTLResourceStorageModeShared)
		r.kvQ8.vMirrors[li] = device.NewBufferWithLengthOptions(uint(rows*kvd*bf16Size), metal.MTLResourceStorageModeShared)
		if r.kvQ8.kMirrors[li] == nil || r.kvQ8.vMirrors[li] == nil {
			return nil, nil, core.NewError("native.q8SnapshotMirror: mirror allocation failed")
		}
	}
	if !r.dequantQ8MirrorsGPU([]int{li}) {
		r.dequantiseQ8Into(li, r.kCaches[li], r.kvQ8.kScales[li], r.kvQ8.kMirrors[li], rows, kvd)
		r.dequantiseQ8Into(li, r.vCaches[li], r.kvQ8.vScales[li], r.kvQ8.vMirrors[li], rows, kvd)
	}
	return r.kvQ8.kMirrors[li], (*byte)(r.kvQ8.kMirrors[li].Contents()), nil
}

// dequantQ8MirrorsGPU refreshes the given layers' K+V mirrors on the GPU in
// one command buffer — the kernel twin of dequantiseQ8Into. The host loop
// costs ~100ms per call at 16K rows, and the drafter's target-KV export pays
// it EVERY draft block; the kernel makes the refresh sub-ms. Returns false
// (nothing written coherently — the caller host-loops) when the pipeline is
// unavailable.
func (r *archICBReplay) dequantQ8MirrorsGPU(layers []int) bool {
	if _, err := kvQ8DequantRowsPipeline(); err != nil {
		return false
	}
	ok := true
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		for _, li := range layers {
			kvd := r.rowBytes[li] / bf16Size
			rows := r.cacheRows[li]
			if encKVQ8DequantRows(enc, r.kCaches[li], r.kvQ8.kScales[li], r.kvQ8.kMirrors[li], rows, kvd) != nil ||
				encKVQ8DequantRows(enc, r.vCaches[li], r.kvQ8.vScales[li], r.kvQ8.vMirrors[li], rows, kvd) != nil {
				ok = false
			}
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
	})
	return ok
}

// q8SnapshotMirrorV returns the V-side mirror buffer+pointer (q8SnapshotMirror
// refreshes both sides; this accessor avoids a second dequant pass).
func (r *archICBReplay) q8SnapshotMirrorV(li int) (metal.MTLBuffer, *byte) {
	return r.kvQ8.vMirrors[li], (*byte)(r.kvQ8.vMirrors[li].Contents())
}

func (r *archICBReplay) dequantiseQ8Into(li int, cache, scales, mirror metal.MTLBuffer, rows, kvd int) {
	codes := unsafe.Slice((*byte)(cache.Contents()), rows*kvd)
	sc := unsafe.Slice((*float32)(scales.Contents()), rows*(kvd/kvQ8GroupSize))
	out := unsafe.Slice((*byte)(mirror.Contents()), rows*kvd*bf16Size)
	for i, c := range codes {
		f := float32(int8(c)) * sc[i/kvQ8GroupSize]
		lo, hi := bf16BytesOfF32(f)
		out[i*2], out[i*2+1] = lo, hi
	}
}

// flushQ8Mirrors quantises every q8 layer's mirror back into its int8 cache +
// scale plane — the post-write hook every restore entry point calls after
// copying snapshot bytes into the mirrors. Rows a restore did not touch
// round-trip to their existing codes (the codec identity), so partial/prefix
// restores are safe.
func (r *archICBReplay) flushQ8Mirrors() {
	if r == nil || r.kvQ8 == nil || r.kvQ8.kMirrors == nil {
		return
	}
	layers := r.q8MirrorLayers()
	if len(layers) == 0 {
		return
	}
	if r.quantQ8MirrorsGPU(layers) {
		return
	}
	for _, li := range layers {
		kvd := r.rowBytes[li] / bf16Size
		rows := r.cacheRows[li]
		r.quantiseQ8From(r.kvQ8.kMirrors[li], r.kCaches[li], r.kvQ8.kScales[li], rows, kvd)
		r.quantiseQ8From(r.kvQ8.vMirrors[li], r.vCaches[li], r.kvQ8.vScales[li], rows, kvd)
	}
}

// q8MirrorLayers lists the q8 layers whose mirrors are materialised — the
// refresh/flush working set (mirrors allocate lazily; state-free sessions
// have none).
func (r *archICBReplay) q8MirrorLayers() []int {
	var layers []int
	for li := range r.kvQ8.enabled {
		if r.kvQ8.on(li) && r.kvQ8.kMirrors[li] != nil {
			layers = append(layers, li)
		}
	}
	return layers
}

// quantQ8MirrorsGPU flushes the given layers' mirrors back into their int8
// caches + scale planes on the GPU in one command buffer — the existing
// batched-landing kernel (lthn_kv_q8_store_rows_bf16, bit-exact against the
// host quantiser) pointed at the whole mirror. Returns false when the
// pipeline is unavailable (the caller host-loops).
func (r *archICBReplay) quantQ8MirrorsGPU(layers []int) bool {
	if !gpuHasKVQ8StoreRows() {
		return false
	}
	ok := true
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		for _, li := range layers {
			kvd := r.rowBytes[li] / bf16Size
			rows := r.cacheRows[li]
			if encKVQ8StoreRows(enc, r.kvQ8.kMirrors[li], r.kCaches[li], 0, r.kvQ8.kScales[li], 0, rows, kvd) != nil ||
				encKVQ8StoreRows(enc, r.kvQ8.vMirrors[li], r.vCaches[li], 0, r.kvQ8.vScales[li], 0, rows, kvd) != nil {
				ok = false
			}
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
	})
	return ok
}

func (r *archICBReplay) quantiseQ8From(mirror, cache, scales metal.MTLBuffer, rows, kvd int) {
	src := unsafe.Slice((*byte)(mirror.Contents()), rows*kvd*bf16Size)
	codes := unsafe.Slice((*byte)(cache.Contents()), rows*kvd)
	sc := unsafe.Slice((*float32)(scales.Contents()), rows*(kvd/kvQ8GroupSize))
	n := rows * kvd
	for g := 0; g < n/kvQ8GroupSize; g++ {
		base := g * kvQ8GroupSize
		var m float32
		for i := 0; i < kvQ8GroupSize; i++ {
			f := bf16ToF32(src[(base+i)*2], src[(base+i)*2+1])
			if f < 0 {
				f = -f
			}
			if f > m {
				m = f
			}
		}
		scale := m * (1.0 / 127.0) // the store kernel's fast-math reciprocal form
		inv := float32(0)
		if scale > 0 {
			inv = 1 / scale
		}
		sc[g] = scale
		for i := 0; i < kvQ8GroupSize; i++ {
			f := bf16ToF32(src[(base+i)*2], src[(base+i)*2+1])
			q := rintF32(f * inv)
			if q > 127 {
				q = 127
			} else if q < -127 {
				q = -127
			}
			codes[base+i] = byte(int8(q))
		}
	}
}

// refreshQ8SnapshotMirrors re-dequantises every ALREADY-MATERIALISED mirror —
// the cached stateLayerViews hold mirror pointers across saves, and the live
// q8 caches move on between sleeps.
func (r *archICBReplay) refreshQ8SnapshotMirrors() error {
	if r == nil || r.kvQ8 == nil || r.kvQ8.kMirrors == nil {
		return nil
	}
	layers := r.q8MirrorLayers()
	if len(layers) == 0 {
		return nil
	}
	if r.dequantQ8MirrorsGPU(layers) {
		return nil
	}
	for _, li := range layers {
		if _, _, err := r.q8SnapshotMirror(li); err != nil {
			return err
		}
	}
	return nil
}
