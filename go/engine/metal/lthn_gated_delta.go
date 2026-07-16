// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model/arch/Qwen/qwen3"
	"github.com/tmc/apple/metal"
)

// lthn_gated_delta.go — the device gated delta-rule recurrence (S1 of the hybrid-recurrence
// campaign, docs/design-hybrid-recurrence.md): pipeline resolvers for the two Dk instantiations,
// the encoder emit the device block (S2) reuses, and the host round-trip wrapper the parity tests
// and the pre-integration benches drive. The kernel replaces deltanet.GatedDeltaRuleF32's host
// triple-loop for engine-metal builds; the host form stays the parity reference and every other
// build's implementation.

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
	gdConvDK128Once, gdConvDK64Once   sync.Once
	gdConvDK128PSO, gdConvDK64PSO     metal.MTLComputePipelineState
	gdConvDK128Err, gdConvDK64Err     error
	gdRingOnce, gdGatesOnce           sync.Once
	gdRingPSO, gdGatesPSO             metal.MTLComputePipelineState
	gdRingErr, gdGatesErr             error
	gdNormDV128Once, gdNormDV64Once   sync.Once
	gdNormDV128PSO, gdNormDV64PSO     metal.MTLComputePipelineState
	gdNormDV128Err, gdNormDV64Err     error
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
	convPSO, err := gdConvPipeline(h.Dk)
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
	normPSO, err := gdNormPipeline(h.Dv)
	if err != nil {
		return err
	}
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

		// conv + SiLU + split + ℓ2-norm: one simdgroup per (head row, t).
		setPSO(enc, convPSO)
		setBuf(enc, h.ring.buf, 0, 0)
		setBuf(enc, qkvB, 0, 1)
		setBuf(enc, wConv, 0, 2)
		setBuf(enc, wBias, 0, 3)
		setBuf(enc, sc.qN.buf, 0, 4)
		setBuf(enc, sc.kN.buf, 0, 5)
		setBuf(enc, sc.vN.buf, 0, 6)
		setEncInt32(enc, int32(L), 7)
		setEncInt32(enc, int32(h.K), 8)
		setEncInt32(enc, int32(h.Hk), 9)
		setEncInt32(enc, int32(h.Hv), 10)
		setEncInt32(enc, int32(hasBias), 11)
		dispatchThreadgroups(enc,
			metal.MTLSize{Width: 1, Height: uint(2*h.Hk + h.Hv), Depth: uint(L)},
			metal.MTLSize{Width: 32, Height: 1, Depth: 1})

		// ring advance (reads ring+qkv, writes ring; thread-per-channel column ownership).
		setPSO(enc, ringPSO)
		setBuf(enc, h.ring.buf, 0, 0)
		setBuf(enc, qkvB, 0, 1)
		setEncInt32(enc, int32(L), 2)
		setEncInt32(enc, int32(h.K), 3)
		setEncInt32(enc, int32(convDim), 4)
		dispatchThreadgroups(enc,
			metal.MTLSize{Width: uint((convDim + 255) / 256), Height: 1, Depth: 1},
			metal.MTLSize{Width: 256, Height: 1, Depth: 1})

		// α/β gate transform.
		setPSO(enc, gatesPSO)
		setBuf(enc, aB, 0, 0)
		setBuf(enc, bB, 0, 1)
		setBuf(enc, wALog, 0, 2)
		setBuf(enc, wDt, 0, 3)
		setBuf(enc, sc.g.buf, 0, 4)
		setBuf(enc, sc.beta.buf, 0, 5)
		setEncInt32(enc, int32(L*h.Hv), 6)
		setEncInt32(enc, int32(h.Hv), 7)
		dispatchThreadgroups(enc,
			metal.MTLSize{Width: uint((L*h.Hv + 255) / 256), Height: 1, Depth: 1},
			metal.MTLSize{Width: 256, Height: 1, Depth: 1})

		// the recurrence (S1 kernel), state resident in-place, y into vN's twin (reuse sc.z? no —
		// z is an input; o gets its own rows in sc.gated then the norm overwrites... o and gated
		// share extent [L,vDim]; the norm reads o and z and writes gated, so o needs its own
		// buffer: reuse sc.vN? vN is the recurrence's v INPUT. Give o = sc.gated, out = sc.gated:
		// norm reads o row into registers before writing the same rows — single dispatch RMW on
		// disjoint rows with in-register staging, safe.
		if err := encGatedDeltaStepF32(enc, sc.qN.buf, sc.kN.buf, sc.vN.buf, sc.g.buf, sc.beta.buf,
			h.state.buf, sc.gated.buf, 0, 0, 0, 0, 0, 0, 0, L, h.kSlots, h.Hk, h.Hv, h.Dk, h.Dv); err != nil {
			encErr = err
			endEncodingFast(enc)
			return
		}

		// gated RMSNorm(o)·SiLU(z), o and out aliased over sc.gated (in-register row staging).
		setPSO(enc, normPSO)
		setBuf(enc, sc.gated.buf, 0, 0)
		setBuf(enc, zB, 0, 1)
		setBuf(enc, wNorm, 0, 2)
		setBuf(enc, sc.gated.buf, 0, 3)
		setEncInt32(enc, int32(L*h.Hv), 4)
		setBytesF32(enc, 1e-6, 5)
		dispatchThreadgroups(enc,
			metal.MTLSize{Width: 1, Height: uint(L * h.Hv), Depth: 1},
			metal.MTLSize{Width: 32, Height: 1, Depth: 1})

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
func gatedDeltaBlockDeviceHook(sc *qwen3.GatedDeltaScratch, qkv, z, a, b []float32, w *qwen3.GatedDeltaWeights, cfg qwen3.GatedDeltaConfig, priorConv, priorDelta []float32, L int) ([]float32, error) {
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
