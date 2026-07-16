// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/arch/Qwen/qwen3"
	"dappco.re/go/inference/model/arch/deltanet"
	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/quant/mlxaffine"
)

// lthn_gated_delta_test.go proves the S1 device recurrence (kernels/lthn_gated_delta.metal) against
// the host reference deltanet.GatedDeltaRuleF32 — the campaign's parity gate before any wiring
// (docs/design-hybrid-recurrence.md). The two sides factor the same math differently by contract:
// the host repeats q/k to value heads, ℓ2-norms k internally and scales q by 1/√Dk inside; the
// kernel takes UNrepeated pre-normalised q/k (GQA is index arithmetic) and scales the OUTPUT. The
// helpers below bridge the factorings so both sides see identical mathematics.

// gdTestPrep builds one random gated-delta problem and both sides' views of it.
type gdTestPrep struct {
	T, Hk, Hv, Dk, Dv    int
	qHost, kHost, vHost  []float32 // repeated to Hv, q ℓ2-normalised (host contract)
	alpha, beta          []float32 // [T,Hv]
	priorHost            []float32 // [Hv,Dk,Dv] (host layout)
	qDev, kDev           []float32 // unrepeated [T,Hk,Dk], both ℓ2-normalised (kernel contract)
	vDev                 []float32 // [T,Hv,Dv]
	stateDev             []float32 // [kSlots,Hv,Dv,Dk] slot 0 = prior (kernel layout)
	kSlots               int
}

func gdL2NormRows(x []float32, rows, d int) []float32 {
	out := make([]float32, len(x))
	for r := 0; r < rows; r++ {
		var ss float64
		for i := 0; i < d; i++ {
			v := float64(x[r*d+i])
			ss += v * v
		}
		inv := 1.0 / math.Sqrt(ss+1e-6)
		for i := 0; i < d; i++ {
			out[r*d+i] = float32(float64(x[r*d+i]) * inv)
		}
	}
	return out
}

// gdStateToHost transposes one kernel-layout state slot [Hv,Dv,Dk] into the host [Hv,Dk,Dv].
func gdStateToHost(dev []float32, Hv, Dk, Dv int) []float32 {
	out := make([]float32, Hv*Dk*Dv)
	for h := 0; h < Hv; h++ {
		for dv := 0; dv < Dv; dv++ {
			for dk := 0; dk < Dk; dk++ {
				out[h*Dk*Dv+dk*Dv+dv] = dev[(h*Dv+dv)*Dk+dk]
			}
		}
	}
	return out
}

func newGDTestPrep(t *testing.T, seed int64, T, Hk, Hv, Dk, Dv, kSlots int) *gdTestPrep {
	t.Helper()
	if Dk != Dv {
		t.Fatalf("host reference is square-state only (Dk=%d Dv=%d)", Dk, Dv)
	}
	rng := rand.New(rand.NewSource(seed))
	fill := func(n int, lo, hi float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = lo + (hi-lo)*rng.Float32()
		}
		return out
	}
	p := &gdTestPrep{T: T, Hk: Hk, Hv: Hv, Dk: Dk, Dv: Dv, kSlots: kSlots}
	qRaw := fill(T*Hk*Dk, -1, 1)
	kRaw := fill(T*Hk*Dk, -1, 1)
	p.vDev = fill(T*Hv*Dv, -1, 1)
	p.alpha = fill(T*Hv, 0.9, 0.999)
	p.beta = fill(T*Hv, 0.1, 0.9)
	priorDev := fill(Hv*Dv*Dk, -0.2, 0.2)

	// Kernel side: ℓ2-normalise q and k, unrepeated.
	p.qDev = gdL2NormRows(qRaw, T*Hk, Dk)
	p.kDev = kRaw // kernel takes k̂; normalise below with the SAME formula the host uses internally
	p.kDev = gdL2NormRows(kRaw, T*Hk, Dk)
	p.stateDev = make([]float32, kSlots*Hv*Dv*Dk)
	copy(p.stateDev[:Hv*Dv*Dk], priorDev)

	// Host side: repeat to value heads; q normalised (host contract), k RAW (host norms inside).
	rep := Hv / Hk
	expand := func(src []float32) []float32 {
		out := make([]float32, T*Hv*Dk)
		for tt := 0; tt < T; tt++ {
			for hv := 0; hv < Hv; hv++ {
				copy(out[(tt*Hv+hv)*Dk:(tt*Hv+hv+1)*Dk], src[(tt*Hk+hv/rep)*Dk:(tt*Hk+hv/rep+1)*Dk])
			}
		}
		return out
	}
	p.qHost = expand(p.qDev)
	p.kHost = expand(kRaw)
	p.vHost = p.vDev
	p.priorHost = gdStateToHost(priorDev, Hv, Dk, Dv)
	return p
}

// gdScaledDiff returns the max |got−want| relative to the tensor's own max |want| — the
// f32-rounding-scale measure (a plain per-element rel diff inflates on near-zero elements and
// reads rounding noise as drift; the #8-B lesson).
func gdScaledDiff(t *testing.T, name string, got, want []float32) float64 {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch %d vs %d", name, len(got), len(want))
	}
	var worst, scale float64
	for i := range got {
		if a := math.Abs(float64(want[i])); a > scale {
			scale = a
		}
		if d := math.Abs(float64(got[i]) - float64(want[i])); d > worst {
			worst = d
		}
	}
	return worst / (scale + 1e-12)
}

func gdRequireKernel(t *testing.T) {
	t.Helper()
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — device gated-delta recurrence")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("Metal runtime unavailable — device gated-delta recurrence: %v", err)
	}
	if !gatedDeltaStepUsable(128, 128, 16, 48) {
		t.Skip("gated-delta kernel unavailable in this metallib")
	}
}

// TestGatedDeltaStepDevice_Good proves pick-for-pick parity with the host recurrence at the real
// Qwen3.6-27B geometry (Hk=16, Hv=48, Dk=Dv=128 — GQA repeat exercised by construction) for both
// the decode shape (T=1) and a draft-block shape (T=4), over a carried non-zero prior state.
func TestGatedDeltaStepDevice_Good(t *testing.T) {
	gdRequireKernel(t)
	for _, T := range []int{1, 4} {
		p := newGDTestPrep(t, 42+int64(T), T, 16, 48, 128, 128, 1)
		scale := float32(1.0 / math.Sqrt(128))
		oHost, sHost, err := deltanet.GatedDeltaRuleF32(p.qHost, p.kHost, p.vHost, p.beta, p.alpha, p.priorHost, T, p.Hv, p.Dk, scale, 0)
		if err != nil {
			t.Fatalf("host reference: %v", err)
		}
		y := make([]float32, T*p.Hv*p.Dv)
		if err := GatedDeltaStepDevice(p.qDev, p.kDev, p.vDev, p.alpha, p.beta, p.stateDev, y, T, 1, p.Hk, p.Hv, p.Dk, p.Dv); err != nil {
			t.Fatalf("GatedDeltaStepDevice(T=%d): %v", T, err)
		}
		yRel := gdScaledDiff(t, "y", y, oHost)
		sRel := gdScaledDiff(t, "state", gdStateToHost(p.stateDev, p.Hv, p.Dk, p.Dv), sHost)
		t.Logf("T=%d parity: scaled max diff y=%.3e state=%.3e", T, yRel, sRel)
		// f32 kernel vs the host's f64 within-step accumulation: rounding-scale drift only.
		if yRel > 5e-4 || sRel > 5e-4 {
			t.Fatalf("T=%d drift beyond f32 rounding scale: y=%.3e state=%.3e", T, yRel, sRel)
		}
	}
}

// TestGatedDeltaStepDevice_Bad pins the rejection shapes: an uninstantiated key head dim, bad
// geometry, and mismatched slice sizes all error before any dispatch.
func TestGatedDeltaStepDevice_Bad(t *testing.T) {
	gdRequireKernel(t)
	y := make([]float32, 4*64)
	ok := make([]float32, 4*64)
	st := make([]float32, 4*64*64)
	if err := GatedDeltaStepDevice(ok, ok, ok, ok, ok, st, y, 1, 1, 2, 4, 96, 64); err == nil {
		t.Fatal("Dk=96 (no instantiation) must error")
	}
	if err := GatedDeltaStepDevice(ok, ok, ok, ok, ok, st, y, 1, 1, 3, 4, 64, 64); err == nil {
		t.Fatal("Hv%Hk != 0 must error")
	}
	if err := GatedDeltaStepDevice(ok[:7], ok, ok, ok, ok, st, y, 1, 1, 1, 4, 64, 64); err == nil {
		t.Fatal("q size mismatch must error")
	}
	if gatedDeltaStepUsable(128, 0, 16, 48) {
		t.Fatal("Dv=0 must not be usable")
	}
}

// TestGatedDeltaStepDevice_Ugly proves the llama.cpp snapshot-slot mapping: with kSlots=4 and T=4,
// slot s holds the state s tokens back (slot 0 = final); with T=2 < kSlots, only slots 0..T-1 are
// rewritten and older slots keep their caller-owned contents byte-for-byte.
func TestGatedDeltaStepDevice_Ugly(t *testing.T) {
	gdRequireKernel(t)
	const T, Hk, Hv, Dk, Dv, kSlots = 4, 2, 4, 64, 64, 4
	p := newGDTestPrep(t, 7, T, Hk, Hv, Dk, Dv, kSlots)
	scale := float32(1.0 / math.Sqrt(float64(Dk)))

	// Host prefixes: state after (T-s) tokens for s = 0..T-1.
	prefix := make([][]float32, T+1)
	for L := 1; L <= T; L++ {
		q := p.qHost[:L*Hv*Dk]
		k := p.kHost[:L*Hv*Dk]
		v := p.vHost[:L*Hv*Dv]
		_, s, err := deltanet.GatedDeltaRuleF32(q, k, v, p.beta[:L*Hv], p.alpha[:L*Hv], p.priorHost, L, Hv, Dk, scale, 0)
		if err != nil {
			t.Fatalf("host prefix L=%d: %v", L, err)
		}
		prefix[L] = s
	}

	y := make([]float32, T*Hv*Dv)
	if err := GatedDeltaStepDevice(p.qDev, p.kDev, p.vDev, p.alpha, p.beta, p.stateDev, y, T, kSlots, Hk, Hv, Dk, Dv); err != nil {
		t.Fatalf("GatedDeltaStepDevice: %v", err)
	}
	slotSize := Hv * Dv * Dk
	for s := 0; s < kSlots; s++ {
		got := gdStateToHost(p.stateDev[s*slotSize:(s+1)*slotSize], Hv, Dk, Dv)
		rel := gdScaledDiff(t, "slot", got, prefix[T-s])
		t.Logf("slot %d (= state %d tokens back): scaled max diff %.3e", s, s, rel)
		if rel > 5e-4 {
			t.Fatalf("slot %d does not match the host state %d tokens back: %.3e", s, s, rel)
		}
	}

	// T=2 < kSlots: slots 2,3 must keep caller contents.
	p2 := newGDTestPrep(t, 9, 2, Hk, Hv, Dk, Dv, kSlots)
	sentinel := float32(123.5)
	for i := 2 * slotSize; i < 4*slotSize; i++ {
		p2.stateDev[i] = sentinel
	}
	y2 := make([]float32, 2*Hv*Dv)
	if err := GatedDeltaStepDevice(p2.qDev, p2.kDev, p2.vDev, p2.alpha, p2.beta, p2.stateDev, y2, 2, kSlots, Hk, Hv, Dk, Dv); err != nil {
		t.Fatalf("GatedDeltaStepDevice(T=2): %v", err)
	}
	for i := 2 * slotSize; i < 4*slotSize; i++ {
		if p2.stateDev[i] != sentinel {
			t.Fatalf("slot beyond T-1 was clobbered at %d: %v", i, p2.stateDev[i])
		}
	}
}

// TestGatedDeltaStepDeviceBeatsHost is the S1 speed receipt at the real 27B decode shape, measured
// in the form S2 actually uses: all 48 gated-delta layers of one token encoded into ONE command
// buffer over resident state (one big [48,Hv,Dv,Dk] buffer, per-layer offsets), one commit+wait —
// against the host recurrence loop stepping the same 48 layers. The single-shot round-trip
// (upload+wait dominated) is logged for context but not gated.
func TestGatedDeltaStepDeviceBeatsHost(t *testing.T) {
	gdRequireKernel(t)
	if testing.Short() {
		t.Skip("timing probe — skipped in -short")
	}
	const T, Hk, Hv, Dk, Dv, layers = 1, 16, 48, 128, 128, 48
	p := newGDTestPrep(t, 11, T, Hk, Hv, Dk, Dv, 1)
	scale := float32(1.0 / math.Sqrt(float64(Dk)))
	y := make([]float32, T*Hv*Dv)

	// Round-trip context number (not the gate).
	if err := GatedDeltaStepDevice(p.qDev, p.kDev, p.vDev, p.alpha, p.beta, p.stateDev, y, T, 1, Hk, Hv, Dk, Dv); err != nil {
		t.Fatalf("warm GatedDeltaStepDevice: %v", err)
	}
	rtStart := time.Now()
	if err := GatedDeltaStepDevice(p.qDev, p.kDev, p.vDev, p.alpha, p.beta, p.stateDev, y, T, 1, Hk, Hv, Dk, Dv); err != nil {
		t.Fatalf("GatedDeltaStepDevice: %v", err)
	}
	roundTrip := time.Since(rtStart)

	// The one-CB 48-layer span over resident buffers.
	alloc := func(n int) *pinnedNoCopyBytes {
		b, err := newPinnedNoCopyBytes(n)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes(%d): %v", n, err)
		}
		return b
	}
	stateSize := Hv * Dv * Dk * 4
	qb := alloc(len(p.qDev) * 4)
	kb := alloc(len(p.kDev) * 4)
	vb := alloc(len(p.vDev) * 4)
	gb := alloc(len(p.alpha) * 4)
	bb := alloc(len(p.beta) * 4)
	sb := alloc(layers * stateSize)
	yb := alloc(layers * T * Hv * Dv * 4)
	copy(qb.bytes, float32Bytes(p.qDev))
	copy(kb.bytes, float32Bytes(p.kDev))
	copy(vb.bytes, float32Bytes(p.vDev))
	copy(gb.bytes, float32Bytes(p.alpha))
	copy(bb.bytes, float32Bytes(p.beta))
	for li := 0; li < layers; li++ {
		copy(sb.bytes[li*stateSize:(li+1)*stateSize], float32Bytes(p.stateDev))
	}
	span := func() time.Duration {
		start := time.Now()
		var innerErr error
		withAutoreleasePool(func() {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			for li := 0; li < layers; li++ {
				if err := encGatedDeltaStepF32(enc, qb.buf, kb.buf, vb.buf, gb.buf, bb.buf, sb.buf, yb.buf,
					0, 0, 0, 0, 0, uint(li*stateSize), uint(li*T*Hv*Dv*4), T, 1, Hk, Hv, Dk, Dv); err != nil {
					innerErr = err
					break
				}
			}
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
		})
		if innerErr != nil {
			t.Fatalf("encode 48-layer span: %v", innerErr)
		}
		return time.Since(start)
	}
	span() // warm
	const iters = 10
	var devTotal time.Duration
	for i := 0; i < iters; i++ {
		devTotal += span()
	}
	devSpan := devTotal / iters

	// Host: the same 48 layer steps.
	if _, _, err := deltanet.GatedDeltaRuleF32(p.qHost, p.kHost, p.vHost, p.beta, p.alpha, p.priorHost, T, Hv, Dk, scale, 0); err != nil {
		t.Fatalf("warm host: %v", err)
	}
	hostStart := time.Now()
	for i := 0; i < iters; i++ {
		for li := 0; li < layers; li++ {
			if _, _, err := deltanet.GatedDeltaRuleF32(p.qHost, p.kHost, p.vHost, p.beta, p.alpha, p.priorHost, T, Hv, Dk, scale, 0); err != nil {
				t.Fatalf("host iter: %v", err)
			}
		}
	}
	hostSpan := time.Since(hostStart) / iters

	speedup := float64(hostSpan) / float64(devSpan)
	t.Logf("27B token (48 gated-delta layers): host %v vs one-CB device span %v (%.1f µs/layer) = ×%.1f; single round-trip %v",
		hostSpan, devSpan, float64(devSpan.Microseconds())/layers, speedup, roundTrip)
	if speedup < 20 {
		t.Fatalf("one-CB device span only ×%.2f over host (host %v, device %v) — want ≥20×", speedup, hostSpan, devSpan)
	}
}

// --- S2: the device block vs qwen3.GatedDeltaForwardScratchFromInputF32 ---

// gdBlockFixture builds one random gated-delta layer (weights + cfg) and a stream of raw
// projection outputs, for driving both the host block and the device block over a carried state.
type gdBlockFixture struct {
	cfg qwen3.GatedDeltaConfig
	w   *qwen3.GatedDeltaWeights
	rng *rand.Rand
}

func newGDBlockFixture(seed int64, Hk, Hv, Dk, K int) *gdBlockFixture {
	rng := rand.New(rand.NewSource(seed))
	cfg := qwen3.GatedDeltaConfig{KeyHeads: Hk, ValueHeads: Hv, HeadDim: Dk, ConvKernel: K, Eps: 1e-6}
	fill := func(n int, lo, hi float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = lo + (hi-lo)*rng.Float32()
		}
		return out
	}
	convDim, vDim := cfg.ConvDim(), cfg.VDim()
	w := &qwen3.GatedDeltaWeights{
		ConvWeight: fill(convDim*K, -0.5, 0.5),
		ConvBias:   fill(convDim, -0.2, 0.2),
		ALog:       fill(Hv, -1, 1.2),
		DtBias:     fill(Hv, -0.5, 0.5),
		Norm:       fill(Dk, 0.5, 1.5),
		OutProj:    fill(vDim, 0, 0), // unused: the block stops before out_proj
	}
	return &gdBlockFixture{cfg: cfg, w: w, rng: rng}
}

func (f *gdBlockFixture) inputs(L int) (qkv, z, a, b []float32) {
	fill := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = -1 + 2*f.rng.Float32()
		}
		return out
	}
	return fill(L * f.cfg.ConvDim()), fill(L * f.cfg.VDim()), fill(L * f.cfg.ValueHeads), fill(L * f.cfg.ValueHeads)
}

// TestGatedDeltaBlockDeviceRun_Good drives the whole device block against the host block across a
// carried-state sequence (three decode steps then an L=4 chunk) at both the fixture shape and the
// real 27B geometry — gated outputs gate each call, the exported ring+delta state gates the end.
func TestGatedDeltaBlockDeviceRun_Good(t *testing.T) {
	gdRequireKernel(t)
	for _, shape := range []struct {
		name           string
		Hk, Hv, Dk, K  int
	}{
		{"fixture", 2, 4, 64, 4},
		{"real27B", 16, 48, 128, 4},
	} {
		t.Run(shape.name, func(t *testing.T) {
			f := newGDBlockFixture(31, shape.Hk, shape.Hv, shape.Dk, shape.K)
			if !gatedDeltaBlockUsable(shape.Dk, shape.Dk, shape.Hk, shape.Hv, shape.K) {
				t.Skip("block kernels unavailable")
			}
			h, err := newGatedDeltaDeviceState(shape.Hk, shape.Hv, shape.Dk, shape.Dk, shape.K, 1)
			if err != nil {
				t.Fatalf("newGatedDeltaDeviceState: %v", err)
			}
			var hostConv, hostDelta []float32
			const D = 256
			for step, L := range []int{1, 1, 1, 4} {
				qkv, z, a, b := f.inputs(L)
				// The host mutates alpha/beta in place — give each side its own copies.
				aHost, bHost := append([]float32(nil), a...), append([]float32(nil), b...)
				gatedHost, vDim, nc, nd, herr := qwen3.GatedDeltaForwardScratchFromInputF32(
					append([]float32(nil), qkv...), z, aHost, bHost, f.w, f.cfg, hostConv, hostDelta, L, D, nil)
				if herr != nil {
					t.Fatalf("host block step %d: %v", step, herr)
				}
				hostConv, hostDelta = nc, nd
				gatedDev := make([]float32, L*vDim)
				if derr := GatedDeltaBlockDeviceRun(h, qkv, z, a, b,
					f.w.ConvWeight, f.w.ConvBias, f.w.ALog, f.w.DtBias, f.w.Norm,
					nil, nil, L, gatedDev); derr != nil {
					t.Fatalf("device block step %d: %v", step, derr)
				}
				rel := gdScaledDiff(t, "gated", gatedDev, gatedHost)
				t.Logf("%s step %d (L=%d): gated scaled max diff %.3e", shape.name, step, L, rel)
				if rel > 5e-4 {
					t.Fatalf("step %d gated drift %.3e beyond f32 rounding scale", step, rel)
				}
			}
			expConv, expDelta := h.export()
			if rel := gdScaledDiff(t, "ring", expConv, hostConv); rel > 5e-4 {
				t.Fatalf("exported conv ring drift %.3e", rel)
			}
			if rel := gdScaledDiff(t, "delta", expDelta, hostDelta); rel > 5e-4 {
				t.Fatalf("exported delta state drift %.3e", rel)
			}
		})
	}
}

// TestGatedDeltaBlockDeviceRun_Bad pins the rejection shapes: nil handle, size mismatches and an
// unservable geometry all error before any dispatch.
func TestGatedDeltaBlockDeviceRun_Bad(t *testing.T) {
	gdRequireKernel(t)
	f := newGDBlockFixture(5, 2, 4, 64, 4)
	h, err := newGatedDeltaDeviceState(2, 4, 64, 64, 4, 1)
	if err != nil {
		t.Fatalf("newGatedDeltaDeviceState: %v", err)
	}
	qkv, z, a, b := f.inputs(1)
	gated := make([]float32, f.cfg.VDim())
	if err := GatedDeltaBlockDeviceRun(nil, qkv, z, a, b, f.w.ConvWeight, f.w.ConvBias, f.w.ALog, f.w.DtBias, f.w.Norm, nil, nil, 1, gated); err == nil {
		t.Fatal("nil handle must error")
	}
	if err := GatedDeltaBlockDeviceRun(h, qkv[:5], z, a, b, f.w.ConvWeight, f.w.ConvBias, f.w.ALog, f.w.DtBias, f.w.Norm, nil, nil, 1, gated); err == nil {
		t.Fatal("qkv size mismatch must error")
	}
	if gatedDeltaBlockUsable(96, 96, 2, 4, 4) {
		t.Fatal("Dk=96 must not be block-usable")
	}
	if gatedDeltaBlockUsable(64, 128, 2, 4, 4) {
		t.Fatal("Dk != Dv must not be block-usable")
	}
}

// TestGatedDeltaBlockDeviceRun_Ugly proves the export/prime seam (the snapshot/clone path): a
// fresh handle primed from a mid-sequence export continues bit-identically with the original —
// device state round-trips through host layout without loss.
func TestGatedDeltaBlockDeviceRun_Ugly(t *testing.T) {
	gdRequireKernel(t)
	const Hk, Hv, Dk, K = 2, 4, 64, 4
	f := newGDBlockFixture(17, Hk, Hv, Dk, K)
	if !gatedDeltaBlockUsable(Dk, Dk, Hk, Hv, K) {
		t.Skip("block kernels unavailable")
	}
	h1, err := newGatedDeltaDeviceState(Hk, Hv, Dk, Dk, K, 1)
	if err != nil {
		t.Fatalf("newGatedDeltaDeviceState: %v", err)
	}
	vDim := f.cfg.VDim()
	// Two steps on h1, then export → prime h2 → identical third step on both.
	for i := 0; i < 2; i++ {
		qkv, z, a, b := f.inputs(1)
		gated := make([]float32, vDim)
		if err := GatedDeltaBlockDeviceRun(h1, qkv, z, a, b, f.w.ConvWeight, f.w.ConvBias, f.w.ALog, f.w.DtBias, f.w.Norm, nil, nil, 1, gated); err != nil {
			t.Fatalf("h1 step %d: %v", i, err)
		}
	}
	expConv, expDelta := h1.export()
	h2, err := newGatedDeltaDeviceState(Hk, Hv, Dk, Dk, K, 1)
	if err != nil {
		t.Fatalf("newGatedDeltaDeviceState h2: %v", err)
	}
	qkv, z, a, b := f.inputs(1)
	g1 := make([]float32, vDim)
	g2 := make([]float32, vDim)
	if err := GatedDeltaBlockDeviceRun(h1, qkv, z, a, b, f.w.ConvWeight, f.w.ConvBias, f.w.ALog, f.w.DtBias, f.w.Norm, nil, nil, 1, g1); err != nil {
		t.Fatalf("h1 final step: %v", err)
	}
	if err := GatedDeltaBlockDeviceRun(h2, qkv, z, a, b, f.w.ConvWeight, f.w.ConvBias, f.w.ALog, f.w.DtBias, f.w.Norm, expConv, expDelta, 1, g2); err != nil {
		t.Fatalf("h2 (primed from export) step: %v", err)
	}
	for i := range g1 {
		if g1[i] != g2[i] {
			t.Fatalf("export/prime round-trip diverged at %d: %v vs %v", i, g1[i], g2[i])
		}
	}
}

// --- S3: the whole quant layer CB vs the per-stage quant path, at the composed-session level ---

// gdQuantTestModel builds a one-layer fully-packed gated-delta composed model (every projection +
// the FFN carrying 4-bit affine codes) — the S3 engagement shape.
func gdQuantTestModel(t *testing.T) *composed.ComposedModel {
	t.Helper()
	const D, FF, vocab = 512, 1024, 64
	cfg := qwen3.GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 64, ConvKernel: 4, Eps: 1e-5}
	convDim, vDim := cfg.ConvDim(), cfg.VDim()
	quant := func(seed, outDim, inDim int) *model.QuantWeight {
		packed, scales, biases, err := mlxaffine.QuantizeTensor(cbSyn(outDim*inDim, seed), outDim, inDim, 4, 64)
		if err != nil {
			t.Fatalf("QuantizeTensor(%d,%d): %v", outDim, inDim, err)
		}
		return &model.QuantWeight{Packed: packed, Scales: scales, Biases: biases, Bits: 4, GroupSize: 64, OutDim: outDim, InDim: inDim}
	}
	w := &qwen3.GatedDeltaWeights{
		ConvWeight: cbSyn(convDim*cfg.ConvKernel, 11), ConvBias: cbSyn(convDim, 12),
		ALog: cbSyn(cfg.ValueHeads, 13), DtBias: cbSyn(cfg.ValueHeads, 14), Norm: cbSyn(cfg.HeadDim, 15),
		InProjQKVQ: quant(21, convDim, D), InProjZQ: quant(22, vDim, D),
		InProjAQ: quant(23, cfg.ValueHeads, D), InProjBQ: quant(24, cfg.ValueHeads, D),
		OutProjQ: quant(25, D, vDim),
	}
	return &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6, Quantised: true,
		Layers: []composed.Layer{{
			InputNorm: cbSyn(D, 31), PostAttnNorm: cbSyn(D, 32),
			MLP:   &composed.MLP{GateQ: quant(41, FF, D), UpQ: quant(42, FF, D), DownQ: quant(43, D, FF), FF: FF},
			Mixer: composed.NewGatedDeltaMixer(w, cfg),
		}},
	}
}

// TestGatedDeltaQuantLayerDevice_Good is the S3 session-level A/B: the same packed model stepped
// through the whole-layer CB (hook bound) and through the per-stage quant path (hook nil), carried
// state and all — the hiddens must agree at f32 rounding scale every step.
func TestGatedDeltaQuantLayerDevice_Good(t *testing.T) {
	gdRequireKernel(t)
	if qwen3.GatedDeltaQuantLayerDevice == nil {
		t.Skip("quant layer hook unbound (LTHN_GD_BLOCK=0?)")
	}
	m := gdQuantTestModel(t)
	saved := qwen3.GatedDeltaQuantLayerDevice
	defer func() { qwen3.GatedDeltaQuantLayerDevice = saved }()

	steps := [][]int32{{1, 2, 3, 4}, {5}, {6}, {7}}
	run := func() [][]float32 {
		sess := composed.NewSession(m)
		var outs [][]float32
		for _, ids := range steps {
			hid, err := sess.Forward(ids)
			if err != nil {
				t.Fatalf("Forward(%v): %v", ids, err)
			}
			outs = append(outs, append([]float32(nil), hid...))
		}
		return outs
	}
	qwen3.GatedDeltaQuantLayerDevice = saved
	fused := run()
	qwen3.GatedDeltaQuantLayerDevice = nil
	staged := run()
	for i := range fused {
		rel := gdScaledDiff(t, "hidden", fused[i], staged[i])
		t.Logf("step %d: whole-layer CB vs per-stage scaled max diff %.3e", i, rel)
		if rel > 5e-4 {
			t.Fatalf("step %d diverged: %.3e", i, rel)
		}
	}
}

// TestGatedDeltaQuantLayerDevice_Bad pins the decline shapes: a geometry the kernels cannot serve
// (HeadDim 32 — no Dk instantiation) silently falls to the per-stage path (Forward still serves the
// token), and the direct run rejects wrong sizes before any dispatch.
func TestGatedDeltaQuantLayerDevice_Bad(t *testing.T) {
	gdRequireKernel(t)
	if qwen3.GatedDeltaQuantLayerDevice == nil {
		t.Skip("quant layer hook unbound")
	}
	// HeadDim 32: convDim = 2*2*32 + 4*32 = 256, vDim = 128 — packable, but no dk32 kernel, so the
	// layer hook must decline on its first call and the per-stage branch serves.
	const D, FF, vocab = 256, 512, 32
	cfg := qwen3.GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 32, ConvKernel: 4, Eps: 1e-5}
	convDim, vDim := cfg.ConvDim(), cfg.VDim()
	w := &qwen3.GatedDeltaWeights{
		ConvWeight: cbSyn(convDim*cfg.ConvKernel, 11), ConvBias: cbSyn(convDim, 12),
		ALog: cbSyn(cfg.ValueHeads, 13), DtBias: cbSyn(cfg.ValueHeads, 14), Norm: cbSyn(cfg.HeadDim, 15),
		InProjQKVQ: mustQuant(t, 21, convDim, D), InProjZQ: mustQuant(t, 22, vDim, D),
		InProjAQ: mustQuant(t, 23, cfg.ValueHeads, D), InProjBQ: mustQuant(t, 24, cfg.ValueHeads, D),
		OutProjQ: mustQuant(t, 25, D, vDim),
	}
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6, Quantised: true,
		Layers: []composed.Layer{{
			InputNorm: cbSyn(D, 31), PostAttnNorm: cbSyn(D, 32),
			MLP:   &composed.MLP{GateQ: mustQuant(t, 41, FF, D), UpQ: mustQuant(t, 42, FF, D), DownQ: mustQuant(t, 43, D, FF), FF: FF},
			Mixer: composed.NewGatedDeltaMixer(w, cfg),
		}},
	}
	sess := composed.NewSession(m)
	if _, err := sess.Forward([]int32{1, 2}); err != nil {
		t.Fatalf("decline path must serve the token, got %v", err)
	}

	// Direct run: size mismatch rejected before any dispatch.
	h, err := newGatedDeltaDeviceState(2, 4, 64, 64, 4, 1)
	if err != nil {
		t.Fatalf("newGatedDeltaDeviceState: %v", err)
	}
	w64 := &qwen3.GatedDeltaWeights{
		ConvWeight: cbSyn((2*2*64+4*64)*4, 51), ConvBias: cbSyn(2*2*64+4*64, 52),
		ALog: cbSyn(4, 53), DtBias: cbSyn(4, 54), Norm: cbSyn(64, 55),
		InProjQKVQ: mustQuant(t, 61, 2*2*64+4*64, 512), InProjZQ: mustQuant(t, 62, 4*64, 512),
		InProjAQ: mustQuant(t, 63, 4, 512), InProjBQ: mustQuant(t, 64, 4, 512),
		OutProjQ: mustQuant(t, 65, 512, 4*64),
	}
	y := make([]float32, 512)
	if err := gatedDeltaQuantLayerRun(h, make([]float32, 5), cbSyn(512, 1), w64, cbSyn(512, 2),
		mustQuant(t, 71, 1024, 512), mustQuant(t, 72, 1024, 512), mustQuant(t, 73, 512, 1024),
		1, 512, 1024, 1e-6, nil, nil, y); err == nil {
		t.Fatal("x size mismatch must error")
	}
}

func mustQuant(t *testing.T, seed, outDim, inDim int) *model.QuantWeight {
	t.Helper()
	packed, scales, biases, err := mlxaffine.QuantizeTensor(cbSyn(outDim*inDim, seed), outDim, inDim, 4, 64)
	if err != nil {
		t.Fatalf("QuantizeTensor: %v", err)
	}
	return &model.QuantWeight{Packed: packed, Scales: scales, Biases: biases, Bits: 4, GroupSize: 64, OutDim: outDim, InDim: inDim}
}

// gdBF16TestModel builds a one-layer dense bf16-resident gated-delta composed model — the #26
// layer-fold engagement shape (every projection a raw bf16 view).
func gdBF16TestModel(t *testing.T) *composed.ComposedModel {
	t.Helper()
	const D, FF, vocab = 512, 1024, 64
	cfg := qwen3.GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 64, ConvKernel: 4, Eps: 1e-5}
	convDim, vDim := cfg.ConvDim(), cfg.VDim()
	bw := func(seed, outDim, inDim int) *model.BF16Weight {
		return &model.BF16Weight{Data: f32sToBF16Bytes(cbSyn(outDim*inDim, seed)), OutDim: outDim, InDim: inDim}
	}
	w := &qwen3.GatedDeltaWeights{
		ConvWeight: cbSyn(convDim*cfg.ConvKernel, 11), ConvBias: cbSyn(convDim, 12),
		ALog: cbSyn(cfg.ValueHeads, 13), DtBias: cbSyn(cfg.ValueHeads, 14), Norm: cbSyn(cfg.HeadDim, 15),
		InProjQKVB: bw(21, convDim, D), InProjZB: bw(22, vDim, D),
		InProjAB: bw(23, cfg.ValueHeads, D), InProjBB: bw(24, cfg.ValueHeads, D),
		OutProjB: bw(25, D, vDim),
	}
	return &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6, BF16Resident: true,
		Layers: []composed.Layer{{
			InputNorm: cbSyn(D, 31), PostAttnNorm: cbSyn(D, 32),
			MLP:   &composed.MLP{GateB: bw(41, FF, D), UpB: bw(42, FF, D), DownB: bw(43, D, FF), FF: FF},
			Mixer: composed.NewGatedDeltaMixer(w, cfg),
		}},
	}
}

// TestGatedDeltaBF16LayerDevice_Good is the dense bf16 layer fold's session A/B: the same
// bf16-resident model stepped through the whole-layer CB (hook bound) and the per-stage bf16 path
// (hook nil) — carried state and all, at f32 rounding scale.
func TestGatedDeltaBF16LayerDevice_Good(t *testing.T) {
	gdRequireKernel(t)
	if qwen3.GatedDeltaBF16LayerDevice == nil {
		t.Skip("bf16 layer hook unbound (LTHN_BF16_SEAM=0?)")
	}
	m := gdBF16TestModel(t)
	saved := qwen3.GatedDeltaBF16LayerDevice
	defer func() { qwen3.GatedDeltaBF16LayerDevice = saved }()

	steps := [][]int32{{1, 2, 3, 4}, {5}, {6}, {7}}
	run := func() [][]float32 {
		sess := composed.NewSession(m)
		var outs [][]float32
		for _, ids := range steps {
			hid, err := sess.Forward(ids)
			if err != nil {
				t.Fatalf("Forward(%v): %v", ids, err)
			}
			outs = append(outs, append([]float32(nil), hid...))
		}
		return outs
	}
	qwen3.GatedDeltaBF16LayerDevice = saved
	fused := run()
	qwen3.GatedDeltaBF16LayerDevice = nil
	staged := run()
	for i := range fused {
		rel := gdScaledDiff(t, "hidden", fused[i], staged[i])
		t.Logf("step %d: bf16 layer CB vs per-stage scaled max diff %.3e", i, rel)
		if rel > 5e-4 {
			t.Fatalf("step %d diverged: %.3e", i, rel)
		}
	}
}

// TestGatedDeltaBF16LayerDevice_Bad pins the rejections: bad bf16 geometry declines before any
// dispatch and the direct run rejects size mismatches.
func TestGatedDeltaBF16LayerDevice_Bad(t *testing.T) {
	gdRequireKernel(t)
	if !gatedDeltaBlockUsable(64, 64, 2, 4, 4) {
		t.Skip("block kernels unavailable")
	}
	h, err := newGatedDeltaDeviceState(2, 4, 64, 64, 4, 1)
	if err != nil {
		t.Fatalf("newGatedDeltaDeviceState: %v", err)
	}
	if !bf16GeometryOK(&model.BF16Weight{Data: make([]byte, 8*4*2), OutDim: 8, InDim: 4}, 8, 4) {
		t.Fatal("well-formed bf16 weight must pass")
	}
	if bf16GeometryOK(&model.BF16Weight{Data: make([]byte, 7), OutDim: 8, InDim: 4}, 8, 4) {
		t.Fatal("short data must fail")
	}
	y := make([]float32, 512)
	m := gdBF16TestModel(t)
	gm := m.Layers[0].Mixer
	_ = gm
	w := &qwen3.GatedDeltaWeights{}
	if err := gatedDeltaBF16LayerRun(h, make([]float32, 5), cbSyn(512, 1), w, cbSyn(512, 2),
		nil, nil, nil, 1, 512, 1024, 1e-6, nil, nil, y); err == nil {
		t.Fatal("empty weights must error")
	}
}

// TestAttnBF16FoldDevice_Good is the attention fold's session A/B: a two-layer [gated-delta,
// attention] dense bf16 model stepped with every fold hook bound vs the per-stage bf16 path —
// carried KV cache and recurrent state included.
func TestAttnBF16FoldDevice_Good(t *testing.T) {
	gdRequireKernel(t)
	if composed.AttnBF16FrontDevice == nil || composed.AttnBF16TailDevice == nil {
		t.Skip("attention fold hooks unbound (LTHN_BF16_SEAM=0?)")
	}
	const D, FF, vocab = 512, 1024, 64
	gcfg := qwen3.GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 64, ConvKernel: 4, Eps: 1e-5}
	convDim, vDim := gcfg.ConvDim(), gcfg.VDim()
	bw := func(seed, outDim, inDim int) *model.BF16Weight {
		return &model.BF16Weight{Data: f32sToBF16Bytes(cbSyn(outDim*inDim, seed)), OutDim: outDim, InDim: inDim}
	}
	gw := &qwen3.GatedDeltaWeights{
		ConvWeight: cbSyn(convDim*gcfg.ConvKernel, 11), ConvBias: cbSyn(convDim, 12),
		ALog: cbSyn(gcfg.ValueHeads, 13), DtBias: cbSyn(gcfg.ValueHeads, 14), Norm: cbSyn(gcfg.HeadDim, 15),
		InProjQKVB: bw(21, convDim, D), InProjZB: bw(22, vDim, D),
		InProjAB: bw(23, gcfg.ValueHeads, D), InProjBB: bw(24, gcfg.ValueHeads, D),
		OutProjB: bw(25, D, vDim),
	}
	const AH, AKVH, AHD = 4, 2, 128 // heads*hd == D
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6, BF16Resident: true,
		Layers: []composed.Layer{
			{
				InputNorm: cbSyn(D, 31), PostAttnNorm: cbSyn(D, 32),
				MLP:   &composed.MLP{GateB: bw(41, FF, D), UpB: bw(42, FF, D), DownB: bw(43, D, FF), FF: FF},
				Mixer: composed.NewGatedDeltaMixer(gw, gcfg),
			},
			{
				InputNorm: cbSyn(D, 51), PostAttnNorm: cbSyn(D, 52),
				MLP: &composed.MLP{GateB: bw(61, FF, D), UpB: bw(62, FF, D), DownB: bw(63, D, FF), FF: FF},
				Mixer: composed.NewAttnMixer(&composed.AttnWeights{
					QProjB: bw(71, AH*AHD, D), KProjB: bw(72, AKVH*AHD, D), VProjB: bw(73, AKVH*AHD, D),
					OProjB: bw(74, D, AH*AHD), QNorm: cbSyn(AHD, 75), KNorm: cbSyn(AHD, 76),
				}, composed.AttnConfig{Heads: AH, KVHeads: AKVH, HeadDim: AHD, RotaryDim: AHD, RopeTheta: 1e6, NormEps: 1e-6}),
			},
		},
	}
	savedF, savedT := composed.AttnBF16FrontDevice, composed.AttnBF16TailDevice
	savedL := qwen3.GatedDeltaBF16LayerDevice
	savedFull := composed.AttnBF16FullLayerDevice
	defer func() {
		composed.AttnBF16FrontDevice, composed.AttnBF16TailDevice = savedF, savedT
		qwen3.GatedDeltaBF16LayerDevice = savedL
		composed.AttnBF16FullLayerDevice = savedFull
	}()

	steps := [][]int32{{1, 2, 3, 4}, {5}, {6}, {7}}
	run := func() [][]float32 {
		sess := composed.NewSession(m)
		var outs [][]float32
		for _, ids := range steps {
			hid, err := sess.Forward(ids)
			if err != nil {
				t.Fatalf("Forward(%v): %v", ids, err)
			}
			outs = append(outs, append([]float32(nil), hid...))
		}
		return outs
	}
	composed.AttnBF16FrontDevice, composed.AttnBF16TailDevice = savedF, savedT
	qwen3.GatedDeltaBF16LayerDevice = savedL
	composed.AttnBF16FullLayerDevice = savedFull
	fused := run()
	composed.AttnBF16FrontDevice, composed.AttnBF16TailDevice = nil, nil
	qwen3.GatedDeltaBF16LayerDevice = nil
	composed.AttnBF16FullLayerDevice = nil
	staged := run()
	for i := range fused {
		rel := gdScaledDiff(t, "hidden", fused[i], staged[i])
		t.Logf("step %d: attn+gd folds vs per-stage scaled max diff %.3e", i, rel)
		if rel > 5e-4 {
			t.Fatalf("step %d diverged: %.3e", i, rel)
		}
	}
}

// TestAttnBF16FoldDevice_Bad pins the fold's rejections: geometry mismatches error before any
// dispatch on both seams.
func TestAttnBF16FoldDevice_Bad(t *testing.T) {
	gdRequireKernel(t)
	bw := &model.BF16Weight{Data: make([]byte, 8*4*2), OutDim: 8, InDim: 4}
	if _, _, _, err := AttnBF16FrontDevice(make([]float32, 3), make([]float32, 4), bw, bw, bw, 1, 4, 8, 8, 1e-6); err == nil {
		t.Fatal("x size mismatch must error")
	}
	if _, err := AttnBF16TailDevice(make([]float32, 4), make([]float32, 8), bw, make([]float32, 4), bw, bw, bw, 1, 4, 8, 8, 1e-6); err == nil {
		t.Fatal("tail geometry mismatch must error")
	}
}

// TestAttnQuantFoldDevice_Good is the packed twins' session A/B: a two-layer [gated-delta,
// attention] fully-quantised model stepped with the fold hooks bound vs the per-stage quant path.
func TestAttnQuantFoldDevice_Good(t *testing.T) {
	gdRequireKernel(t)
	if composed.AttnQuantFrontDevice == nil || composed.AttnQuantTailDevice == nil {
		t.Skip("quant attention fold hooks unbound (LTHN_BF16_SEAM=0?)")
	}
	const D, FF, vocab = 512, 1024, 64
	gcfg := qwen3.GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 64, ConvKernel: 4, Eps: 1e-5}
	convDim, vDim := gcfg.ConvDim(), gcfg.VDim()
	gw := &qwen3.GatedDeltaWeights{
		ConvWeight: cbSyn(convDim*gcfg.ConvKernel, 11), ConvBias: cbSyn(convDim, 12),
		ALog: cbSyn(gcfg.ValueHeads, 13), DtBias: cbSyn(gcfg.ValueHeads, 14), Norm: cbSyn(gcfg.HeadDim, 15),
		InProjQKVQ: mustQuant(t, 21, convDim, D), InProjZQ: mustQuant(t, 22, vDim, D),
		InProjAQ: mustQuant(t, 23, gcfg.ValueHeads, D), InProjBQ: mustQuant(t, 24, gcfg.ValueHeads, D),
		OutProjQ: mustQuant(t, 25, D, vDim),
	}
	const AH, AKVH, AHD = 4, 2, 128
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6, Quantised: true,
		Layers: []composed.Layer{
			{
				InputNorm: cbSyn(D, 31), PostAttnNorm: cbSyn(D, 32),
				MLP:   &composed.MLP{GateQ: mustQuant(t, 41, FF, D), UpQ: mustQuant(t, 42, FF, D), DownQ: mustQuant(t, 43, D, FF), FF: FF},
				Mixer: composed.NewGatedDeltaMixer(gw, gcfg),
			},
			{
				InputNorm: cbSyn(D, 51), PostAttnNorm: cbSyn(D, 52),
				MLP: &composed.MLP{GateQ: mustQuant(t, 61, FF, D), UpQ: mustQuant(t, 62, FF, D), DownQ: mustQuant(t, 63, D, FF), FF: FF},
				Mixer: composed.NewAttnMixer(&composed.AttnWeights{
					QProjQ: mustQuant(t, 71, AH*AHD, D), KProjQ: mustQuant(t, 72, AKVH*AHD, D), VProjQ: mustQuant(t, 73, AKVH*AHD, D),
					OProjQ: mustQuant(t, 74, D, AH*AHD), QNorm: cbSyn(AHD, 75), KNorm: cbSyn(AHD, 76),
				}, composed.AttnConfig{Heads: AH, KVHeads: AKVH, HeadDim: AHD, RotaryDim: AHD, RopeTheta: 1e6, NormEps: 1e-6}),
			},
		},
	}
	savedF, savedT := composed.AttnQuantFrontDevice, composed.AttnQuantTailDevice
	defer func() { composed.AttnQuantFrontDevice, composed.AttnQuantTailDevice = savedF, savedT }()

	steps := [][]int32{{1, 2, 3, 4}, {5}, {6}, {7}}
	run := func() [][]float32 {
		sess := composed.NewSession(m)
		var outs [][]float32
		for _, ids := range steps {
			hid, err := sess.Forward(ids)
			if err != nil {
				t.Fatalf("Forward(%v): %v", ids, err)
			}
			outs = append(outs, append([]float32(nil), hid...))
		}
		return outs
	}
	composed.AttnQuantFrontDevice, composed.AttnQuantTailDevice = savedF, savedT
	fused := run()
	composed.AttnQuantFrontDevice, composed.AttnQuantTailDevice = nil, nil
	staged := run()
	for i := range fused {
		rel := gdScaledDiff(t, "hidden", fused[i], staged[i])
		t.Logf("step %d: quant attn fold vs per-stage scaled max diff %.3e", i, rel)
		if rel > 5e-4 {
			t.Fatalf("step %d diverged: %.3e", i, rel)
		}
	}
}
