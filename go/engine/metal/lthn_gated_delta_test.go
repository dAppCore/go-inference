// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	"dappco.re/go/inference/model/arch/deltanet"
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
