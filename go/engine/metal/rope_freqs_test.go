// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"os"
	"testing"
	"unsafe"
)

// ropeClose fails when two bf16 rope outputs differ beyond tol (decoded to f32).
func ropeClose(t *testing.T, got, want []byte, tol float32, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d != %d", label, len(got), len(want))
	}
	for i := 0; i+1 < len(got); i += 2 {
		g := bf16ToF32(got[i], got[i+1])
		w := bf16ToF32(want[i], want[i+1])
		if d := g - w; d > tol || d < -tol {
			t.Fatalf("%s: elem %d freqs=%g vs base=%g (diff %g)", label, i/2, g, w, d)
		}
	}
}

func ropeFixtureX(b, nHeads, headDim int) []byte {
	xf := make([]float32, b*nHeads*headDim)
	for i := range xf {
		xf[i] = float32((i%13)-6) * 0.1
	}
	return toBF16Bytes(xf)
}

// plainRopeInvFreqs is the standard spectrum base^(-2d/rotaryDim), d in [0,rotaryDim/2).
func plainRopeInvFreqs(base float64, rotaryDim int) []float32 {
	f := make([]float32, rotaryDim/2)
	for d := range f {
		f[d] = float32(math.Pow(base, -float64(2*d)/float64(rotaryDim)))
	}
	return f
}

func TestRoPEFreqsBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const batch, nHeads, headDim, rotaryDim = 1, 8, 64, 64
	x := toBF16Bytes(syntheticFloat32(batch*nHeads*headDim, 5))
	invFreqs := plainRopeInvFreqs(10000, rotaryDim)
	if _, err := RoPEFreqsBF16(x, batch, nHeads, headDim, rotaryDim, invFreqs, 1, 7, false); err != nil {
		t.Fatalf("RoPEFreqsBF16 warmup: %v", err)
	}

	var ropeErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, ropeErr = RoPEFreqsBF16(x, batch, nHeads, headDim, rotaryDim, invFreqs, 1, 7, false)
	})
	if ropeErr != nil {
		t.Fatalf("RoPEFreqsBF16: %v", ropeErr)
	}
	if allocs > 10 {
		t.Fatalf("RoPEFreqsBF16 allocations = %.0f, want <= 10", allocs)
	}
}

func TestRoPEFreqsBF16IntoUsesCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const batch, nHeads, headDim, rotaryDim = 1, 8, 64, 32
	x := toBF16Bytes(syntheticFloat32(batch*nHeads*headDim, 5))
	invFreqs := plainRopeInvFreqs(10000, rotaryDim)
	out := make([]byte, len(x))
	for i := range out {
		out[i] = 0xA5
	}

	got, err := RoPEFreqsBF16Into(out, x, batch, nHeads, headDim, rotaryDim, invFreqs, 1, 7, false)
	if err != nil {
		t.Fatalf("RoPEFreqsBF16Into: %v", err)
	}
	if len(got) != len(out) {
		t.Fatalf("RoPEFreqsBF16Into len = %d, want %d", len(got), len(out))
	}
	if unsafe.Pointer(&got[0]) != unsafe.Pointer(&out[0]) {
		t.Fatal("RoPEFreqsBF16Into did not return caller-owned output backing")
	}
	want, err := RoPEFreqsBF16(x, batch, nHeads, headDim, rotaryDim, invFreqs, 1, 7, false)
	if err != nil {
		t.Fatalf("RoPEFreqsBF16 reference: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("RoPEFreqsBF16Into output differs from allocating wrapper")
	}
}

// TestRoPEFreqsBF16_EqualsBase_Good proves the freqs path is correct: handed the
// plain-rope spectrum, rope_single_freqs reproduces rope_single — full rotary and
// partial — so the freqs ABI + the inv_freq=1/period reciprocal are right.
func TestRoPEFreqsBF16_EqualsBase_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const b, nHeads, headDim = 1, 2, 64
	const base = float32(1_000_000)
	const pos = 7
	x := ropeFixtureX(b, nHeads, headDim)

	// full rotary
	wantFull, err := RoPEBF16(x, b, nHeads, headDim, base, 1.0, pos, false)
	if err != nil {
		t.Fatalf("RoPEBF16: %v", err)
	}
	gotFull, err := RoPEFreqsBF16(x, b, nHeads, headDim, headDim, plainRopeInvFreqs(float64(base), headDim), 1.0, pos, false)
	if err != nil {
		t.Fatalf("RoPEFreqsBF16 full: %v", err)
	}
	ropeClose(t, gotFull, wantFull, 2e-2, "full rotary")

	// partial rotary (rotaryDim = headDim/2) — the tail must pass through too
	const rot = headDim / 2
	wantPart, err := RoPEDimsBF16(x, b, nHeads, headDim, rot, base, 1.0, pos, false)
	if err != nil {
		t.Fatalf("RoPEDimsBF16: %v", err)
	}
	gotPart, err := RoPEFreqsBF16(x, b, nHeads, headDim, rot, plainRopeInvFreqs(float64(base), rot), 1.0, pos, false)
	if err != nil {
		t.Fatalf("RoPEFreqsBF16 partial: %v", err)
	}
	ropeClose(t, gotPart, wantPart, 2e-2, "partial rotary")
}

// TestRoPEFreqsBF16_NonPlainDiffers_Good proves the frequency buffer is actually
// consumed: a non-plain spectrum produces a different rotation than the base rope.
func TestRoPEFreqsBF16_NonPlainDiffers_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const b, nHeads, headDim = 1, 2, 64
	const base = float32(1_000_000)
	const pos = 7
	x := ropeFixtureX(b, nHeads, headDim)

	want, err := RoPEBF16(x, b, nHeads, headDim, base, 1.0, pos, false)
	if err != nil {
		t.Fatalf("RoPEBF16: %v", err)
	}
	// halve every frequency → every angle halves, so the high-frequency dims (which
	// rotate visibly even at a small position) must change. (Perturbing only the
	// low-freq dims wouldn't show here — they barely rotate at pos 7, which is
	// exactly why YaRN's interpolation is a long-context effect.)
	inv := plainRopeInvFreqs(float64(base), headDim)
	for d := range inv {
		inv[d] *= 0.5
	}
	got, err := RoPEFreqsBF16(x, b, nHeads, headDim, headDim, inv, 1.0, pos, false)
	if err != nil {
		t.Fatalf("RoPEFreqsBF16: %v", err)
	}
	differs := false
	for i := 0; i+1 < len(got); i += 2 {
		if g, w := bf16ToF32(got[i], got[i+1]), bf16ToF32(want[i], want[i+1]); math.Abs(float64(g-w)) > 1e-2 {
			differs = true
			break
		}
	}
	if !differs {
		t.Fatal("non-plain freqs produced the same output as base rope — freqs buffer not consumed")
	}
}

// TestGlobalRopePeriodsFromFolded_MatchesRawSpectrum_Good pins the folded-base seam at the real
// 12B global geometry (headDim 512, rotaryDim 128, raw theta 1e6): the arch-derived base is
// pre-folded to raw^(rotaryDim/headDim), and the unfolding wrapper must reproduce metal's
// gemma4ProportionalFreqs spectrum raw^(2i/headDim) exactly. Feeding the folded base straight
// into proportionalRopePeriods instead lands every period at the 4th root of the true one — the
// position-growing 12B cross-engine drift this seam exists to prevent.
func TestGlobalRopePeriodsFromFolded_MatchesRawSpectrum_Good(t *testing.T) {
	const headDim, rotaryDim = 512, 128
	const rawBase = 1e6
	foldedBase := float32(math.Pow(rawBase, float64(rotaryDim)/float64(headDim))) // 31.6227766 — what arch.RopeBase carries

	got := globalRopePeriodsFromFolded(headDim, rotaryDim, foldedBase)
	want := proportionalRopePeriods(headDim, rotaryDim, rawBase)

	if len(got) != headDim/2 || len(want) != headDim/2 {
		t.Fatalf("period length = %d/%d, want %d", len(got), len(want), headDim/2)
	}
	for i := 0; i < rotaryDim/2; i++ {
		exact := float32(math.Pow(rawBase, float64(2*i)/float64(headDim)))
		if rel := math.Abs(float64(got[i]-exact)) / float64(exact); rel > 1e-5 {
			t.Fatalf("period[%d] = %g, want %g (rel %.2e) — folded base leaked into the raw spectrum", i, got[i], exact, rel)
		}
		if rel := math.Abs(float64(got[i]-want[i])) / float64(want[i]); rel > 1e-6 {
			t.Fatalf("wrapper period[%d] = %g diverges from raw-base periods %g", i, got[i], want[i])
		}
	}
	for i := rotaryDim / 2; i < headDim/2; i++ {
		if !math.IsInf(float64(got[i]), 1) {
			t.Fatalf("period[%d] = %g, want +Inf (unrotated tail)", i, got[i])
		}
	}
}
