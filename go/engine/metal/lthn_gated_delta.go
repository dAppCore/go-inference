// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/attn"
	"github.com/tmc/apple/metal"
)

// lthn_gated_delta.go — the device gated delta-rule recurrence (S1 of the hybrid-recurrence
// campaign, docs/design-hybrid-recurrence.md): pipeline resolvers for the two Dk instantiations,
// the encoder emit the device block (S2) reuses, and the host round-trip wrapper the parity tests
// and the pre-integration benches drive. The kernel replaces deltanet.GatedDeltaRuleF32's host
// triple-loop for engine-metal builds; the host form stays the parity reference and every other
// build's implementation.

// gdBlockEnabled gates the device gated-delta BLOCK (conv ring + gates + recurrence + gated norm in
// one command buffer, recurrent state resident on device — #18 S2). Default on; LTHN_GD_BLOCK=0
// leaves the hooks unbound so the mixer runs the host block — the "before" arm of a same-binary A/B.
var gdBlockEnabled = os.Getenv("LTHN_GD_BLOCK") != "0"

// The model/attn device seams the factory host path consults (GatedDelta*DeviceTry): binding them
// here keeps the hooks beside their implementations. These used to be bound from the retired
// composed engine's backend file (#50); the seams themselves are engine-neutral model/attn hooks.
func init() {
	if gdBlockEnabled {
		attn.GatedDeltaBlockDevice = gatedDeltaBlockDeviceHook             // the whole post-projection gated-delta block in one CB, state device-resident (#18 S2)
		attn.GatedDeltaDeviceStateExport = gatedDeltaDeviceStateExportHook // snapshot/clone readback for the resident state
		attn.GatedDeltaQuantLayerDevice = gatedDeltaQuantLayerDeviceHook   // the WHOLE packed layer in one CB — norm + five packed projections + block + FFN tail (#18 S3)
	}
}

var (
	gatedDeltaDK128Once sync.Once
	gatedDeltaDK128PSO  metal.MTLComputePipelineState
	gatedDeltaDK128Err  error
	gatedDeltaDK64Once  sync.Once
	gatedDeltaDK64PSO   metal.MTLComputePipelineState
	gatedDeltaDK64Err   error
)

// gatedDeltaStepPipeline resolves the recurrence kernel for a key head dim. Dk sizes the kernel's
// per-lane register array, so it is a compile-time instantiation (128 = every Qwen 3.5/3.6 hybrid,
// 64 = the small fixture shapes); other dims are runtime arguments.
func gatedDeltaStepPipeline(dk int) (metal.MTLComputePipelineState, error) {
	resolve := func(name string, pso *metal.MTLComputePipelineState, errOut *error) {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			*errOut = core.NewError("native.gatedDeltaStepPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName(name)
		if fn == nil || fn.GetID() == 0 {
			*errOut = core.NewError("native.gatedDeltaStepPipeline: kernel " + name + " not found")
			return
		}
		*pso, *errOut = device.NewComputePipelineStateWithFunctionError(fn)
	}
	switch dk {
	case 128:
		gatedDeltaDK128Once.Do(func() { resolve("lthn_gated_delta_step_f32_dk128", &gatedDeltaDK128PSO, &gatedDeltaDK128Err) })
		return gatedDeltaDK128PSO, gatedDeltaDK128Err
	case 64:
		gatedDeltaDK64Once.Do(func() { resolve("lthn_gated_delta_step_f32_dk64", &gatedDeltaDK64PSO, &gatedDeltaDK64Err) })
		return gatedDeltaDK64PSO, gatedDeltaDK64Err
	default:
		return nil, core.NewError("native.gatedDeltaStepPipeline: unsupported key head dim (want 64 or 128)")
	}
}

// gatedDeltaStepUsable reports whether the device recurrence serves this geometry: an instantiated
// Dk, square-free runtime dims, and a resolvable pipeline. The customLibrary check runs FIRST so a
// pre-init caller cannot latch the sync.Once into a permanent nil-library failure (the #23 lesson).
func gatedDeltaStepUsable(dk, dv, hk, hv int) bool {
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return false
	}
	if (dk != 128 && dk != 64) || dv <= 0 || hk <= 0 || hv <= 0 || hv%hk != 0 {
		return false
	}
	pso, err := gatedDeltaStepPipeline(dk)
	return err == nil && pso != nil && pso.GetID() != 0
}

// encGatedDeltaStepF32 encodes the recurrence over q/k/v [T,Hk|Hv,Dk|Dv], g/beta [T,Hv] and the
// in-place state [kSlots,Hv,Dv,Dk] (slot 0 live), writing y [T,Hv,Dv] — one dispatch, the whole
// T-loop inside the kernel (docs in kernels/lthn_gated_delta.metal). This is the S2-reusable core:
// the caller owns buffer residency and the surrounding command buffer.
func encGatedDeltaStepF32(
	enc metal.MTLComputeCommandEncoder,
	q, k, v, g, beta, state, y metal.MTLBuffer,
	qOff, kOff, vOff, gOff, betaOff, stateOff, yOff uint,
	T, kSlots, Hk, Hv, Dk, Dv int,
) error {
	if T <= 0 || kSlots < 1 || Dv <= 0 || Hk <= 0 || Hv <= 0 || Hv%Hk != 0 {
		return core.NewError("native.encGatedDeltaStepF32: invalid geometry")
	}
	pso, err := gatedDeltaStepPipeline(Dk)
	if err != nil {
		return err
	}
	setPSO(enc, pso)
	setBuf(enc, q, qOff, 0)
	setBuf(enc, k, kOff, 1)
	setBuf(enc, v, vOff, 2)
	setBuf(enc, g, gOff, 3)
	setBuf(enc, beta, betaOff, 4)
	setBuf(enc, state, stateOff, 5)
	setBuf(enc, y, yOff, 6)
	setEncInt32(enc, int32(T), 7)
	setEncInt32(enc, int32(kSlots), 8)
	setEncInt32(enc, int32(Hk), 9)
	setEncInt32(enc, int32(Hv), 10)
	setEncInt32(enc, int32(Dv), 11)
	const dvPerGroup = 4
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: 1, Height: uint((Dv + dvPerGroup - 1) / dvPerGroup), Depth: uint(Hv)},
		metal.MTLSize{Width: 32, Height: dvPerGroup, Depth: 1},
	)
	return nil
}

// gatedDeltaStepScratch is the pooled pinned staging for one GatedDeltaStepDevice shape — the
// test/bench round-trip only; the integrated block (S2) keeps its buffers resident instead.
type gatedDeltaStepScratch struct {
	q, k, v, g, beta, state, y *pinnedNoCopyBytes
}

type gatedDeltaStepKey struct{ T, kSlots, Hk, Hv, Dk, Dv int }

var gatedDeltaStepPools sync.Map // gatedDeltaStepKey -> *sync.Pool

func getGatedDeltaStepScratch(key gatedDeltaStepKey) (*gatedDeltaStepScratch, error) {
	poolAny, ok := gatedDeltaStepPools.Load(key)
	if !ok {
		poolAny, _ = gatedDeltaStepPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*gatedDeltaStepScratch), nil
	}
	sc := &gatedDeltaStepScratch{}
	var err error
	alloc := func(n int) *pinnedNoCopyBytes {
		if err != nil {
			return nil
		}
		var b *pinnedNoCopyBytes
		b, err = newPinnedNoCopyBytes(n)
		return b
	}
	sc.q = alloc(key.T * key.Hk * key.Dk * 4)
	sc.k = alloc(key.T * key.Hk * key.Dk * 4)
	sc.v = alloc(key.T * key.Hv * key.Dv * 4)
	sc.g = alloc(key.T * key.Hv * 4)
	sc.beta = alloc(key.T * key.Hv * 4)
	sc.state = alloc(key.kSlots * key.Hv * key.Dv * key.Dk * 4)
	sc.y = alloc(key.T * key.Hv * key.Dv * 4)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putGatedDeltaStepScratch(key gatedDeltaStepKey, sc *gatedDeltaStepScratch) {
	if v, ok := gatedDeltaStepPools.Load(key); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// GatedDeltaStepDevice runs the recurrence kernel over host slices — upload, one dispatch, wait,
// read back y and the advanced state (in place over state, which must be [kSlots,Hv,Dv,Dk] with
// slot 0 live; kSlots >= 1). q/k arrive pre-normalised exactly as the kernel contract documents.
// This wrapper exists for the parity tests and the pre-integration benches — the S2 device block
// binds encGatedDeltaStepF32 directly with resident buffers and never round-trips.
func GatedDeltaStepDevice(q, k, v, g, beta, state, y []float32, T, kSlots, Hk, Hv, Dk, Dv int) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if !gatedDeltaStepUsable(Dk, Dv, Hk, Hv) {
		return core.NewError("native.GatedDeltaStepDevice: geometry not servable by the device kernel")
	}
	if len(q) != T*Hk*Dk || len(k) != T*Hk*Dk || len(v) != T*Hv*Dv ||
		len(g) != T*Hv || len(beta) != T*Hv ||
		len(state) != kSlots*Hv*Dv*Dk || len(y) != T*Hv*Dv {
		return core.NewError("native.GatedDeltaStepDevice: size mismatch")
	}
	key := gatedDeltaStepKey{T: T, kSlots: kSlots, Hk: Hk, Hv: Hv, Dk: Dk, Dv: Dv}
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getGatedDeltaStepScratch(key)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putGatedDeltaStepScratch(key, sc)
		up := func(dst *pinnedNoCopyBytes, src []float32) metal.MTLBuffer {
			if encErr != nil {
				return nil
			}
			buf, cerr := dst.copyBuffer(float32Bytes(src))
			if cerr != nil {
				encErr = cerr
			}
			return buf
		}
		qb, kb, vb := up(sc.q, q), up(sc.k, k), up(sc.v, v)
		gb, bb, sb := up(sc.g, g), up(sc.beta, beta), up(sc.state, state)
		if encErr != nil {
			return
		}
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if err := encGatedDeltaStepF32(enc, qb, kb, vb, gb, bb, sb, sc.y.buf,
			0, 0, 0, 0, 0, 0, 0, T, kSlots, Hk, Hv, Dk, Dv); err != nil {
			endEncodingFast(enc)
			encErr = err
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(y, unsafe.Slice((*float32)(unsafe.Pointer(&sc.y.bytes[0])), len(y)))
		copy(state, unsafe.Slice((*float32)(unsafe.Pointer(&sc.state.bytes[0])), len(state)))
	})
	return encErr
}

// --- S2: the device gated-delta BLOCK — conv ring → gates → recurrence → gated norm in one CB ---

var (
	gdConvDK128Once, gdConvDK64Once sync.Once
	gdConvDK128PSO, gdConvDK64PSO   metal.MTLComputePipelineState
	gdConvDK128Err, gdConvDK64Err   error
	gdRingOnce, gdGatesOnce         sync.Once
	gdRingPSO, gdGatesPSO           metal.MTLComputePipelineState
	gdRingErr, gdGatesErr           error
	gdNormDV128Once, gdNormDV64Once sync.Once
	gdNormDV128PSO, gdNormDV64PSO   metal.MTLComputePipelineState
	gdNormDV128Err, gdNormDV64Err   error
)

func gdPlainPipeline(name string, pso *metal.MTLComputePipelineState, errOut *error) {
	if customLibrary == nil || customLibrary.GetID() == 0 {
		*errOut = core.NewError("native.gdPlainPipeline: custom library unavailable")
		return
	}
	fn := customLibrary.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		*errOut = core.NewError("native.gdPlainPipeline: kernel " + name + " not found")
		return
	}
	*pso, *errOut = device.NewComputePipelineStateWithFunctionError(fn)
}

func gdConvPipeline(dk int) (metal.MTLComputePipelineState, error) {
	switch dk {
	case 128:
		gdConvDK128Once.Do(func() { gdPlainPipeline("lthn_gd_conv_silu_split_norm_dk128", &gdConvDK128PSO, &gdConvDK128Err) })
		return gdConvDK128PSO, gdConvDK128Err
	case 64:
		gdConvDK64Once.Do(func() { gdPlainPipeline("lthn_gd_conv_silu_split_norm_dk64", &gdConvDK64PSO, &gdConvDK64Err) })
		return gdConvDK64PSO, gdConvDK64Err
	default:
		return nil, core.NewError("native.gdConvPipeline: unsupported head dim")
	}
}

func gdRingPipeline() (metal.MTLComputePipelineState, error) {
	gdRingOnce.Do(func() { gdPlainPipeline("lthn_gd_ring_advance", &gdRingPSO, &gdRingErr) })
	return gdRingPSO, gdRingErr
}

func gdGatesPipeline() (metal.MTLComputePipelineState, error) {
	gdGatesOnce.Do(func() { gdPlainPipeline("lthn_gd_gates", &gdGatesPSO, &gdGatesErr) })
	return gdGatesPSO, gdGatesErr
}

func gdNormPipeline(dv int) (metal.MTLComputePipelineState, error) {
	switch dv {
	case 128:
		gdNormDV128Once.Do(func() { gdPlainPipeline("lthn_gd_gated_rmsnorm_silu_dv128", &gdNormDV128PSO, &gdNormDV128Err) })
		return gdNormDV128PSO, gdNormDV128Err
	case 64:
		gdNormDV64Once.Do(func() { gdPlainPipeline("lthn_gd_gated_rmsnorm_silu_dv64", &gdNormDV64PSO, &gdNormDV64Err) })
		return gdNormDV64PSO, gdNormDV64Err
	default:
		return nil, core.NewError("native.gdNormPipeline: unsupported head dim")
	}
}

// gatedDeltaBlockUsable reports whether the whole device block serves this geometry: the family
// truth Dk == Dv at an instantiated width, and every stage pipeline resolvable. customLibrary is
// checked FIRST (pre-init callers must not latch the sync.Onces — the #23 lesson).
func gatedDeltaBlockUsable(dk, dv, hk, hv, K int) bool {
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return false
	}
	if dk != dv || (dk != 128 && dk != 64) || hk <= 0 || hv <= 0 || hv%hk != 0 || K < 1 || K > 9 {
		return false
	}
	if !gatedDeltaStepUsable(dk, dv, hk, hv) {
		return false
	}
	for _, resolve := range []func() (metal.MTLComputePipelineState, error){
		func() (metal.MTLComputePipelineState, error) { return gdConvPipeline(dk) },
		gdRingPipeline,
		gdGatesPipeline,
		func() (metal.MTLComputePipelineState, error) { return gdNormPipeline(dv) },
	} {
		if pso, err := resolve(); err != nil || pso == nil || pso.GetID() == 0 {
			return false
		}
	}
	return true
}

// gatedDeltaStateToDevice transposes a host delta state [Hv,Dk,Dv] into the kernel layout
// [Hv,Dv,Dk] (and back for ...ToHost) — the only place the two layouts meet.
func gatedDeltaStateToDevice(dst, src []float32, Hv, Dk, Dv int) {
	for h := 0; h < Hv; h++ {
		hs := src[h*Dk*Dv : (h+1)*Dk*Dv]
		hd := dst[h*Dv*Dk : (h+1)*Dv*Dk]
		for kk := 0; kk < Dk; kk++ {
			for vv := 0; vv < Dv; vv++ {
				hd[vv*Dk+kk] = hs[kk*Dv+vv]
			}
		}
	}
}

func gatedDeltaStateToHost(dst, src []float32, Hv, Dk, Dv int) {
	for h := 0; h < Hv; h++ {
		hs := src[h*Dv*Dk : (h+1)*Dv*Dk]
		hd := dst[h*Dk*Dv : (h+1)*Dk*Dv]
		for vv := 0; vv < Dv; vv++ {
			for kk := 0; kk < Dk; kk++ {
				hd[kk*Dv+vv] = hs[vv*Dk+kk]
			}
		}
	}
}

// gatedDeltaDeviceState is the per-(session, layer) resident recurrent state: the delta matrix
// [kSlots,Hv,Dv,Dk] and the conv ring [(K-1),convDim], both living on device across tokens once
// primed. valid=false means the host slices are authoritative and the next block call re-uploads
// (a fresh session, a restored snapshot, or a never-engaged layer).
type gatedDeltaDeviceState struct {
	Hk, Hv, Dk, Dv, K, convDim, kSlots int
	state, ring                        *pinnedNoCopyBytes
	valid                              bool
}

func newGatedDeltaDeviceState(Hk, Hv, Dk, Dv, K, kSlots int) (*gatedDeltaDeviceState, error) {
	h := &gatedDeltaDeviceState{Hk: Hk, Hv: Hv, Dk: Dk, Dv: Dv, K: K, convDim: (2*Hk + Hv) * Dk, kSlots: kSlots}
	var err error
	if h.state, err = newPinnedNoCopyBytes(kSlots * Hv * Dv * Dk * 4); err != nil {
		return nil, err
	}
	if h.ring, err = newPinnedNoCopyBytes((K - 1) * h.convDim * 4); err != nil {
		return nil, err
	}
	return h, nil
}

// prime uploads host state into the device buffers (nil slices mean zero state — the fresh
// sequence). Host layout for delta is [Hv,Dk,Dv]; the ring layout matches directly.
func (h *gatedDeltaDeviceState) prime(priorConv, priorDelta []float32) {
	ringF := unsafe.Slice((*float32)(unsafe.Pointer(&h.ring.bytes[0])), (h.K-1)*h.convDim)
	stateF := unsafe.Slice((*float32)(unsafe.Pointer(&h.state.bytes[0])), h.kSlots*h.Hv*h.Dv*h.Dk)
	if priorConv != nil {
		copy(ringF, priorConv)
	} else {
		clear(ringF)
	}
	if priorDelta != nil {
		gatedDeltaStateToDevice(stateF[:h.Hv*h.Dv*h.Dk], priorDelta, h.Hv, h.Dk, h.Dv)
	} else {
		clear(stateF)
	}
	h.valid = true
}

// export reads the device state back into host-layout slices — the snapshot/clone seam. The
// device stays authoritative (valid remains true).
func (h *gatedDeltaDeviceState) export() (conv, delta []float32) {
	conv = make([]float32, (h.K-1)*h.convDim)
	delta = make([]float32, h.Hv*h.Dk*h.Dv)
	ringF := unsafe.Slice((*float32)(unsafe.Pointer(&h.ring.bytes[0])), len(conv))
	stateF := unsafe.Slice((*float32)(unsafe.Pointer(&h.state.bytes[0])), h.kSlots*h.Hv*h.Dv*h.Dk)
	copy(conv, ringF)
	gatedDeltaStateToHost(delta, stateF[:h.Hv*h.Dv*h.Dk], h.Hv, h.Dk, h.Dv)
	return conv, delta
}

// gatedDeltaBlockScratch is the pooled per-(shape,L) staging for one block call: the four input
// uploads, the device-only intermediates, and the gated readback.
type gatedDeltaBlockScratch struct {
	qkv, z, a, b        *pinnedNoCopyBytes
	qN, kN, vN, g, beta *pinnedNoCopyBytes
	gated               *pinnedNoCopyBytes
}

type gatedDeltaBlockKey struct{ L, Hk, Hv, Dk int }

var gatedDeltaBlockPools sync.Map // gatedDeltaBlockKey -> *sync.Pool

func getGatedDeltaBlockScratch(key gatedDeltaBlockKey) (*gatedDeltaBlockScratch, error) {
	poolAny, ok := gatedDeltaBlockPools.Load(key)
	if !ok {
		poolAny, _ = gatedDeltaBlockPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*gatedDeltaBlockScratch), nil
	}
	sc := &gatedDeltaBlockScratch{}
	var err error
	alloc := func(n int) *pinnedNoCopyBytes {
		if err != nil {
			return nil
		}
		var buf *pinnedNoCopyBytes
		buf, err = newPinnedNoCopyBytes(n)
		return buf
	}
	convDim := (2*key.Hk + key.Hv) * key.Dk
	vDim := key.Hv * key.Dk
	sc.qkv = alloc(key.L * convDim * 4)
	sc.z = alloc(key.L * vDim * 4)
	sc.a = alloc(key.L * key.Hv * 4)
	sc.b = alloc(key.L * key.Hv * 4)
	sc.qN = alloc(key.L * key.Hk * key.Dk * 4)
	sc.kN = alloc(key.L * key.Hk * key.Dk * 4)
	sc.vN = alloc(key.L * vDim * 4)
	sc.g = alloc(key.L * key.Hv * 4)
	sc.beta = alloc(key.L * key.Hv * 4)
	sc.gated = alloc(key.L * vDim * 4)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putGatedDeltaBlockScratch(key gatedDeltaBlockKey, sc *gatedDeltaBlockScratch) {
	if v, ok := gatedDeltaBlockPools.Load(key); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// GatedDeltaBlockDeviceRun runs the whole gated-delta block AFTER the input projections — causal
// conv ring + SiLU + split + ℓ2-norms, the α/β gate transform, the recurrence, and the gated
// RMSNorm·SiLU(z) — in ONE command buffer over the resident device state h. Inputs are the four
// projection outputs (qkv [L,convDim], z [L,vDim], a/b [L,Hv]) plus the layer's small host
// weights; gated [L,vDim] is written for the caller's out_proj. On the first call (or after a
// restore) h is primed from priorConv/priorDelta; afterwards the device state is authoritative
// and the prior slices are ignored.
func GatedDeltaBlockDeviceRun(
	h *gatedDeltaDeviceState,
	qkv, z, a, b []float32,
	convW, convB, aLog, dtBias, normW []float32,
	priorConv, priorDelta []float32,
	L int,
	gated []float32,
) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if h == nil || !gatedDeltaBlockUsable(h.Dk, h.Dv, h.Hk, h.Hv, h.K) {
		return core.NewError("native.GatedDeltaBlockDeviceRun: geometry not servable")
	}
	convDim, vDim := h.convDim, h.Hv*h.Dv
	if L <= 0 || len(qkv) != L*convDim || len(z) != L*vDim || len(a) != L*h.Hv || len(b) != L*h.Hv ||
		len(convW) != convDim*h.K || len(aLog) != h.Hv || len(dtBias) != h.Hv ||
		len(normW) != h.Dv || len(gated) != L*vDim || (convB != nil && len(convB) != convDim) {
		return core.NewError("native.GatedDeltaBlockDeviceRun: size mismatch")
	}
	if !h.valid {
		h.prime(priorConv, priorDelta)
	}
	key := gatedDeltaBlockKey{L: L, Hk: h.Hk, Hv: h.Hv, Dk: h.Dk}
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getGatedDeltaBlockScratch(key)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putGatedDeltaBlockScratch(key, sc)
		up := func(dst *pinnedNoCopyBytes, src []float32) metal.MTLBuffer {
			if encErr != nil {
				return nil
			}
			buf, cerr := dst.copyBuffer(float32Bytes(src))
			if cerr != nil {
				encErr = cerr
			}
			return buf
		}
		qkvB, zB, aB, bB := up(sc.qkv, qkv), up(sc.z, z), up(sc.a, a), up(sc.b, b)
		if encErr != nil {
			return
		}
		wConv := residentFloat32(convW)
		wBias := wConv
		hasBias := 0
		if convB != nil {
			wBias = residentFloat32(convB)
			hasBias = 1
		}
		wALog := residentFloat32(aLog)
		wDt := residentFloat32(dtBias)
		wNorm := residentFloat32(normW)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if err := encGatedDeltaBlockStages(enc, h, gdBlockStageBufs{
			qkv: qkvB, z: zB, a: aB, b: bB,
			qN: sc.qN.buf, kN: sc.kN.buf, vN: sc.vN.buf, g: sc.g.buf, beta: sc.beta.buf,
			gated: sc.gated.buf,
		}, wConv, wBias, hasBias, wALog, wDt, wNorm, L); err != nil {
			encErr = err
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(gated, unsafe.Slice((*float32)(unsafe.Pointer(&sc.gated.bytes[0])), L*vDim))
	})
	return encErr
}

// --- the qwen3 hook adapters (bound in composed_backend.go) ---

// gatedDeltaBlockDeviceHook adapts GatedDeltaBlockDeviceRun to qwen3's declared block seam: the
// resident state handle rides sc.Device (stowed only after a fully successful run, so a first-call
// failure leaves the host path cleanly in charge), geometry gates before any allocation, and the
// gated output is a fresh slice for the caller's out_proj.
func gatedDeltaBlockDeviceHook(sc *attn.GatedDeltaScratch, qkv, z, a, b []float32, w *model.GatedDeltaWeights, cfg model.GatedDeltaConfig, priorConv, priorDelta []float32, L int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	h, _ := sc.Device.(*gatedDeltaDeviceState)
	if h == nil {
		if !gatedDeltaBlockUsable(cfg.HeadDim, cfg.HeadDim, cfg.KeyHeads, cfg.ValueHeads, cfg.ConvKernel) {
			return nil, core.NewError("native.gatedDeltaBlockDeviceHook: geometry not servable")
		}
		nh, err := newGatedDeltaDeviceState(cfg.KeyHeads, cfg.ValueHeads, cfg.HeadDim, cfg.HeadDim, cfg.ConvKernel, 1)
		if err != nil {
			return nil, err
		}
		h = nh
	}
	gated := make([]float32, L*cfg.VDim())
	if err := GatedDeltaBlockDeviceRun(h, qkv, z, a, b, w.ConvWeight, w.ConvBias, w.ALog, w.DtBias, w.Norm, priorConv, priorDelta, L, gated); err != nil {
		return nil, err
	}
	sc.Device = h
	return gated, nil
}

// gatedDeltaDeviceStateExportHook reads a resident handle back into host-layout slices — the
// snapshot/clone seam behind composed's CloneState.
func gatedDeltaDeviceStateExportHook(dev any) ([]float32, []float32, bool) {
	h, ok := dev.(*gatedDeltaDeviceState)
	if !ok || !h.valid {
		return nil, nil, false
	}
	conv, delta := h.export()
	return conv, delta, true
}

// gdBlockStageBufs names the buffers the block's stage chain reads and writes: the four projection
// outputs in, the split/normed q,k,v + the α/β gates as intermediates, and gated [L,vDim] out (the
// recurrence's y, then gated-normed in place).
type gdBlockStageBufs struct {
	qkv, z, a, b        metal.MTLBuffer
	qN, kN, vN, g, beta metal.MTLBuffer
	gated               metal.MTLBuffer
}

// encGatedDeltaBlockStages encodes the whole post-projection gated-delta block into a live encoder:
// conv ring + SiLU + split + ℓ2-norms → ring advance → α/β gate transform → the recurrence (state
// resident, in place on st) → gated RMSNorm·SiLU(z), with explicit barriers between dependent
// stages (the #8-B encoder discipline). The caller owns the command buffer — GatedDeltaBlockDeviceRun
// wraps this as its own CB; the fused quant LAYER (#18 S3) stacks it between its projections.
func encGatedDeltaBlockStages(
	enc metal.MTLComputeCommandEncoder,
	st *gatedDeltaDeviceState,
	b gdBlockStageBufs,
	wConv, wBias metal.MTLBuffer, hasBias int,
	wALog, wDt, wNorm metal.MTLBuffer,
	L int,
) error {
	convPSO, err := gdConvPipeline(st.Dk)
	if err != nil {
		return err
	}
	ringPSO, err := gdRingPipeline()
	if err != nil {
		return err
	}
	gatesPSO, err := gdGatesPipeline()
	if err != nil {
		return err
	}
	normPSO, err := gdNormPipeline(st.Dv)
	if err != nil {
		return err
	}

	// conv + SiLU + split + ℓ2-norm: one simdgroup per (head row, t).
	setPSO(enc, convPSO)
	setBuf(enc, st.ring.buf, 0, 0)
	setBuf(enc, b.qkv, 0, 1)
	setBuf(enc, wConv, 0, 2)
	setBuf(enc, wBias, 0, 3)
	setBuf(enc, b.qN, 0, 4)
	setBuf(enc, b.kN, 0, 5)
	setBuf(enc, b.vN, 0, 6)
	setEncInt32(enc, int32(L), 7)
	setEncInt32(enc, int32(st.K), 8)
	setEncInt32(enc, int32(st.Hk), 9)
	setEncInt32(enc, int32(st.Hv), 10)
	setEncInt32(enc, int32(hasBias), 11)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: 1, Height: uint(2*st.Hk + st.Hv), Depth: uint(L)},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1})
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)

	// ring advance (reads ring+qkv, writes ring; thread-per-channel column ownership) — after the
	// conv consumed the old ring rows.
	setPSO(enc, ringPSO)
	setBuf(enc, st.ring.buf, 0, 0)
	setBuf(enc, b.qkv, 0, 1)
	setEncInt32(enc, int32(L), 2)
	setEncInt32(enc, int32(st.K), 3)
	setEncInt32(enc, int32(st.convDim), 4)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: uint((st.convDim + 255) / 256), Height: 1, Depth: 1},
		metal.MTLSize{Width: 256, Height: 1, Depth: 1})

	// α/β gate transform (independent of the conv chain — no barrier needed before it).
	setPSO(enc, gatesPSO)
	setBuf(enc, b.a, 0, 0)
	setBuf(enc, b.b, 0, 1)
	setBuf(enc, wALog, 0, 2)
	setBuf(enc, wDt, 0, 3)
	setBuf(enc, b.g, 0, 4)
	setBuf(enc, b.beta, 0, 5)
	setEncInt32(enc, int32(L*st.Hv), 6)
	setEncInt32(enc, int32(st.Hv), 7)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: uint((L*st.Hv + 255) / 256), Height: 1, Depth: 1},
		metal.MTLSize{Width: 256, Height: 1, Depth: 1})
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)

	// the recurrence (S1 kernel): state resident in place, y into b.gated.
	if err := encGatedDeltaStepF32(enc, b.qN, b.kN, b.vN, b.g, b.beta,
		st.state.buf, b.gated, 0, 0, 0, 0, 0, 0, 0, L, st.kSlots, st.Hk, st.Hv, st.Dk, st.Dv); err != nil {
		return err
	}
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)

	// gated RMSNorm(o)·SiLU(z), o and out aliased over b.gated (in-register row staging).
	setPSO(enc, normPSO)
	setBuf(enc, b.gated, 0, 0)
	setBuf(enc, b.z, 0, 1)
	setBuf(enc, wNorm, 0, 2)
	setBuf(enc, b.gated, 0, 3)
	setEncInt32(enc, int32(L*st.Hv), 4)
	setBytesF32(enc, 1e-6, 5)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: 1, Height: uint(L * st.Hv), Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1})
	return nil
}

// --- S3: the whole QUANT gated-delta LAYER in one command buffer ---

// gatedDeltaQuantLayerScratch is the pooled staging for one fused layer call: the x upload / y
// readback pair, the five projections' f32+bf16 stages, the block intermediates, and the FFN tail
// stages — everything between is device-only.
type gatedDeltaQuantLayerScratch struct {
	x, normed1, qkv, z, a, b   *pinnedNoCopyBytes // input side (f32)
	n1BF, qkvBF, zBF, aBF, bBF *pinnedNoCopyBytes // input-side bf16 qmv stages
	qN, kN, vN, g, beta, gated *pinnedNoCopyBytes // block intermediates (f32)
	gatedBF, mixBF             *pinnedNoCopyBytes // out_proj bf16 stages
	mix                        *pinnedNoCopyBytes // out_proj output (f32)
	normed2, n2BF, gFF, gFFBF  *pinnedNoCopyBytes // tail stages
	uFF, uFFBF, sFF, out       *pinnedNoCopyBytes
}

type gatedDeltaQuantLayerKey struct{ L, D, FF, Hk, Hv, Dk, K int }

var gatedDeltaQuantLayerPools sync.Map // gatedDeltaQuantLayerKey -> *sync.Pool

func getGatedDeltaQuantLayerScratch(key gatedDeltaQuantLayerKey) (*gatedDeltaQuantLayerScratch, error) {
	poolAny, ok := gatedDeltaQuantLayerPools.Load(key)
	if !ok {
		poolAny, _ = gatedDeltaQuantLayerPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*gatedDeltaQuantLayerScratch), nil
	}
	sc := &gatedDeltaQuantLayerScratch{}
	var err error
	alloc := func(n int) *pinnedNoCopyBytes {
		if err != nil {
			return nil
		}
		var buf *pinnedNoCopyBytes
		buf, err = newPinnedNoCopyBytes(n)
		return buf
	}
	convDim := (2*key.Hk + key.Hv) * key.Dk
	vDim := key.Hv * key.Dk
	L, D, FF := key.L, key.D, key.FF
	sc.x = alloc(L * D * 4)
	sc.normed1 = alloc(L * D * 4)
	sc.qkv = alloc(L * convDim * 4)
	sc.z = alloc(L * vDim * 4)
	sc.a = alloc(L * key.Hv * 4)
	sc.b = alloc(L * key.Hv * 4)
	sc.n1BF = alloc(L * D * bf16Size)
	sc.qkvBF = alloc(L * convDim * bf16Size)
	sc.zBF = alloc(L * vDim * bf16Size)
	sc.aBF = alloc(L * key.Hv * bf16Size)
	sc.bBF = alloc(L * key.Hv * bf16Size)
	sc.qN = alloc(L * key.Hk * key.Dk * 4)
	sc.kN = alloc(L * key.Hk * key.Dk * 4)
	sc.vN = alloc(L * vDim * 4)
	sc.g = alloc(L * key.Hv * 4)
	sc.beta = alloc(L * key.Hv * 4)
	sc.gated = alloc(L * vDim * 4)
	sc.gatedBF = alloc(L * vDim * bf16Size)
	sc.mixBF = alloc(L * D * bf16Size)
	sc.mix = alloc(L * D * 4)
	sc.normed2 = alloc(L * D * 4)
	sc.n2BF = alloc(L * max(D, FF) * bf16Size)
	sc.gFF = alloc(L * FF * 4)
	sc.gFFBF = alloc(L * FF * bf16Size)
	sc.uFF = alloc(L * FF * 4)
	sc.uFFBF = alloc(L * FF * bf16Size)
	sc.sFF = alloc(L * FF * 4)
	sc.out = alloc(L * D * 4)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putGatedDeltaQuantLayerScratch(key gatedDeltaQuantLayerKey, sc *gatedDeltaQuantLayerScratch) {
	if v, ok := gatedDeltaQuantLayerPools.Load(key); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// gatedDeltaQuantLayerRun runs one WHOLE packed gated-delta layer in a single command buffer:
//
//	normed = RMSNorm(x, inputNorm) → in_proj_qkv/z/a/b (affine qmv/qmm_t over codes)
//	→ the gated-delta block stages (conv ring + gates + recurrence + gated norm, state resident)
//	→ out_proj (packed) → the #8-B FFN tail (residual + post-norm + packed SwiGLU + residual)
//
// x [L,D] is the only upload, y [L,D] the only readback — the unfused path pays SEVEN command
// buffers (4 input projections + block + out_proj + tail) with six host crossings per layer.
// Residuals are plain adds (the composed wiring routes here only at residualScale == 1).
func gatedDeltaQuantLayerRun(
	h *gatedDeltaDeviceState,
	x, inputNorm []float32,
	w *model.GatedDeltaWeights,
	postNorm []float32,
	gate, up, down *model.QuantWeight,
	L, D, FF int, eps float32,
	priorConv, priorDelta []float32,
	y []float32,
) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if h == nil || !gatedDeltaBlockUsable(h.Dk, h.Dv, h.Hk, h.Hv, h.K) {
		return core.NewError("native.gatedDeltaQuantLayerRun: geometry not servable")
	}
	convDim, vDim := h.convDim, h.Hv*h.Dv
	if len(x) != L*D || len(y) != L*D || len(inputNorm) != D || len(postNorm) != D ||
		len(w.ALog) != h.Hv || len(w.DtBias) != h.Hv || len(w.Norm) != h.Dv ||
		(w.ConvBias != nil && len(w.ConvBias) != convDim) || len(w.ConvWeight) != convDim*h.K {
		return core.NewError("native.gatedDeltaQuantLayerRun: size mismatch")
	}
	if w.InProjQKVQ == nil || !quantGeometryOK(w.InProjQKVQ, convDim, D) ||
		w.InProjZQ == nil || !quantGeometryOK(w.InProjZQ, vDim, D) ||
		w.InProjAQ == nil || !quantGeometryOK(w.InProjAQ, h.Hv, D) ||
		w.InProjBQ == nil || !quantGeometryOK(w.InProjBQ, h.Hv, D) ||
		w.OutProjQ == nil || !quantGeometryOK(w.OutProjQ, D, vDim) ||
		!quantGeometryOK(gate, FF, D) || !quantGeometryOK(up, FF, D) || !quantGeometryOK(down, D, FF) {
		return core.NewError("native.gatedDeltaQuantLayerRun: unsupported quant geometry")
	}
	if !h.valid {
		h.prime(priorConv, priorDelta)
	}
	rmsName := "rmsfloat32"
	if D > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32"
	}
	psoRMS, err := pipelineFor(rmsName)
	if err != nil {
		return err
	}
	key := gatedDeltaQuantLayerKey{L: L, D: D, FF: FF, Hk: h.Hk, Hv: h.Hv, Dk: h.Dk, K: h.K}
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getGatedDeltaQuantLayerScratch(key)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putGatedDeltaQuantLayerScratch(key, sc)
		xBuf, cerr := sc.x.copyBuffer(float32Bytes(x))
		if cerr != nil {
			encErr = cerr
			return
		}
		wConv := residentFloat32(w.ConvWeight)
		wBias := wConv
		hasBias := 0
		if w.ConvBias != nil {
			wBias = residentFloat32(w.ConvBias)
			hasBias = 1
		}
		wALog := residentFloat32(w.ALog)
		wDt := residentFloat32(w.DtBias)
		wNorm := residentFloat32(w.Norm)
		inNormBuf := residentFloat32(inputNorm)
		postNormBuf := residentFloat32(postNorm)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		fail := func(err error) {
			encErr = err
			endEncodingFast(enc)
		}

		// input RMSNorm → one bf16 cast → the four packed input projections (independent reads of
		// the same cast rows — barriers only around the shared stages).
		emitRMSNormRows(encSink{enc}, psoRMS, xBuf, inNormBuf, sc.normed1.buf, 0, 0, 0, D, eps, L, rmsThreadgroup(D, psoRMS))
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encNarrowF32ToBF16(enc, sc.normed1.buf, sc.n1BF.buf, L*D); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encProjQuantBF16In(enc, w.InProjQKVQ, sc.n1BF.buf, sc.qkvBF.buf, sc.qkv.buf, L, convDim, D); err != nil {
			fail(err)
			return
		}
		if err := encProjQuantBF16In(enc, w.InProjZQ, sc.n1BF.buf, sc.zBF.buf, sc.z.buf, L, vDim, D); err != nil {
			fail(err)
			return
		}
		if err := encProjQuantBF16In(enc, w.InProjAQ, sc.n1BF.buf, sc.aBF.buf, sc.a.buf, L, h.Hv, D); err != nil {
			fail(err)
			return
		}
		if err := encProjQuantBF16In(enc, w.InProjBQ, sc.n1BF.buf, sc.bBF.buf, sc.b.buf, L, h.Hv, D); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)

		// the gated-delta block stages, state resident.
		if err := encGatedDeltaBlockStages(enc, h, gdBlockStageBufs{
			qkv: sc.qkv.buf, z: sc.z.buf, a: sc.a.buf, b: sc.b.buf,
			qN: sc.qN.buf, kN: sc.kN.buf, vN: sc.vN.buf, g: sc.g.buf, beta: sc.beta.buf,
			gated: sc.gated.buf,
		}, wConv, wBias, hasBias, wALog, wDt, wNorm, L); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)

		// out_proj over codes: gated [L,vDim] → mix [L,D].
		if err := encProjQuantF32(enc, w.OutProjQ, sc.gated.buf, sc.gatedBF.buf, sc.mixBF.buf, sc.mix.buf, L, D, vDim); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)

		// the #8-B FFN tail: hplus = x + mix (in place over xBuf) → post-norm → packed SwiGLU →
		// y = hplus + mlpOut.
		if err := encResidualNormMLPQuantTail(enc, quantTailBufs{
			h: xBuf, mix: sc.mix.buf, normed: sc.normed2.buf, nBF: sc.n2BF.buf,
			g: sc.gFF.buf, gBF: sc.gFFBF.buf, u: sc.uFF.buf, uBF: sc.uFFBF.buf, s: sc.sFF.buf, out: sc.out.buf,
		}, postNormBuf, gate, up, down, L, D, FF, eps); err != nil {
			fail(err)
			return
		}

		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(y, unsafe.Slice((*float32)(unsafe.Pointer(&sc.out.bytes[0])), L*D))
	})
	return encErr
}

// gatedDeltaQuantLayerDeviceHook adapts gatedDeltaQuantLayerRun to qwen3's declared layer seam —
// same handle discipline as the block hook: stowed on sc.Device only after a fully successful run.
func gatedDeltaQuantLayerDeviceHook(sc *attn.GatedDeltaScratch, x, inputNorm []float32, w *model.GatedDeltaWeights, cfg model.GatedDeltaConfig, postNorm []float32, gate, up, down *model.QuantWeight, L, D, FF int, eps float32, priorConv, priorDelta []float32) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	h, _ := sc.Device.(*gatedDeltaDeviceState)
	if h == nil {
		if !gatedDeltaBlockUsable(cfg.HeadDim, cfg.HeadDim, cfg.KeyHeads, cfg.ValueHeads, cfg.ConvKernel) {
			return nil, core.NewError("native.gatedDeltaQuantLayerDeviceHook: geometry not servable")
		}
		nh, err := newGatedDeltaDeviceState(cfg.KeyHeads, cfg.ValueHeads, cfg.HeadDim, cfg.HeadDim, cfg.ConvKernel, 1)
		if err != nil {
			return nil, err
		}
		h = nh
	}
	y := make([]float32, L*D)
	if err := gatedDeltaQuantLayerRun(h, x, inputNorm, w, postNorm, gate, up, down, L, D, FF, eps, priorConv, priorDelta, y); err != nil {
		return nil, err
	}
	sc.Device = h
	return y, nil
}
