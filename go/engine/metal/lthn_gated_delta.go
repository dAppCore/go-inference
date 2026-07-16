// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
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
