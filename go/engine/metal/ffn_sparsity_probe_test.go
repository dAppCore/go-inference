// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"sort"
	"testing"
)

// effWidth returns the fraction of columns holding 95% of the L2 energy of vals
// (already non-negative contributions) — the "effective width".
func effWidth(vals []float64) float64 {
	n := len(vals)
	if n == 0 {
		return 0
	}
	var l2sq float64
	for _, a := range vals {
		l2sq += a * a
	}
	if l2sq == 0 {
		return 0
	}
	sort.Sort(sort.Reverse(sort.Float64Slice(vals)))
	target := 0.95 * l2sq
	var acc float64
	k := 0
	for _, a := range vals {
		acc += a * a
		k++
		if acc >= target {
			break
		}
	}
	return float64(k) / float64(n)
}

// TestFFNGatedSparsityRealE2B is the #364 ceiling probe: on REAL e2b-4bit decode,
// how few FFN columns actually carry the output? A column's OUTPUT contribution is
// gatedᵢ·down[:,i], magnitude ≈ |gatedᵢ|·‖down[:,i]‖ — so we report BOTH the raw
// |gated| effective width and the ‖down‖-weighted one (the true output sparsity).
// Dead columns cost up[:,i]+down[i,:] (2/3 of the FFN) to produce ~nothing; if the
// effective width is small, skipping them is a real decode-bandwidth win.
//
// Forces the non-ICB stepToken path (icbDisabledForTest, byte-identical) so the
// read-only captureFFNGated hook fires. Instrument-only; no engine behaviour change.
//
//	LEM_REAL_E2B=1 MLX_METALLIB_PATH=... go test -run TestFFNGatedSparsityRealE2B -v ./engine/metal/
func TestFFNGatedSparsityRealE2B(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set LEM_REAL_E2B=1 to run the real e2b-4bit FFN-sparsity probe (loads ~2.7GB)")
	}
	dir := resolveE2B4bitDir(t)
	lm, dm, err := loadRegistered(dir)
	if err != nil {
		t.Fatalf("loadRegistered: %v", err)
	}
	defer func() { _ = dm.Close() }()
	sb, err := buildShardBuffers(dm)
	if err != nil {
		t.Fatalf("buildShardBuffers: %v", err)
	}
	defer func() { _ = sb.Close() }()
	qm, err := loadedToQuant(lm, lm.Embed.GroupSize, lm.Embed.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	sess, err := newArchQuantSessionShards(qm, lm.Arch, 320, sb)
	if err != nil {
		t.Fatalf("newArchQuantSessionShards: %v", err)
	}

	prevICB, prevCap, prevBuf := icbDisabledForTest, captureFFNGated, capturedFFNGated
	icbDisabledForTest = true
	captureFFNGated = true
	capturedFFNGated = nil
	defer func() {
		icbDisabledForTest = prevICB
		captureFFNGated = prevCap
		capturedFFNGated = prevBuf
	}()

	prompt := []int32{2, 1000, 2500, 4000, 8000, 16000}
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	const N = 32
	if _, err := sess.GenerateFromCache(N, -1); err != nil {
		t.Fatalf("generate: %v", err)
	}
	caps := capturedFFNGated
	if len(caps) == 0 {
		t.Fatal("no gated captured — hook not firing (ICB not disabled, or all-MoE layers)")
	}
	L := len(qm.Layers)
	dModel := lm.Arch.Hidden

	// Per-layer down column norms ‖down[:,i]‖ (fixed weights; each gated column's
	// output weight). Down is [dModel × dFF]; derive dFF from the PACKED down weight
	// (authoritative) — the captured buffer is the shared mlpScratch, sized for the
	// WIDEST MatFormer layer, so its tail is stale and must be sliced off.
	colNorm := make([][]float64, L)
	dFFof := make([]int, L)
	for l := 0; l < L; l++ {
		dw := qm.Layers[l].Down
		dFF := len(dw.Packed) * 8 / dw.Bits / dModel // 4-bit: 2 codes/byte
		dFFof[l] = dFF
		if l == 0 {
			t.Logf("geom L0: dModel=%d down.Packed=%dB bits=%d gs=%d -> dFF=%d ; captureLen=%d",
				dModel, len(dw.Packed), dw.Bits, dw.GroupSize, dFF, len(bf16ToF32Slice(caps[0])))
		}
		mat, derr := dequantizeAffineRowsF32(dw.Packed, dw.Scales, dw.Biases, dModel, dFF, dw.GroupSize, dw.Bits)
		if derr != nil {
			t.Fatalf("dequant down L%d (dModel=%d dFF=%d): %v", l, dModel, dFF, derr)
		}
		cn := make([]float64, dFF)
		for i := 0; i < dFF; i++ {
			var s float64
			for j := 0; j < dModel; j++ {
				v := float64(mat[j*dFF+i])
				s += v * v
			}
			cn[i] = math.Sqrt(s)
		}
		colNorm[l] = cn
	}

	var below5, totCols int
	var rawCov, wtdCov float64
	var nFFN int
	for k, gb := range caps {
		l := k % L
		dFF := dFFof[l]
		h := bf16ToF32Slice(gb)
		if len(h) < dFF {
			continue
		}
		h = h[:dFF] // the layer's real gated (scratch tail sliced off)
		absH := make([]float64, dFF)
		contrib := make([]float64, dFF)
		var maxv float64
		for i := 0; i < dFF; i++ {
			a := math.Abs(float64(h[i]))
			absH[i] = a
			contrib[i] = a * colNorm[l][i]
			if a > maxv {
				maxv = a
			}
		}
		if maxv > 0 {
			for _, a := range absH {
				if a/maxv < 0.05 {
					below5++
				}
			}
			totCols += dFF
		}
		rawCov += effWidth(append([]float64(nil), absH...))
		wtdCov += effWidth(contrib)
		nFFN++
	}
	if nFFN == 0 {
		t.Fatal("no usable FFN captures")
	}
	t.Logf("=== #364 FFN sparsity — real e2b-4bit, %d FFNs (%d layers) over %d decode tokens ===", nFFN, L, N)
	t.Logf("  |gated| < 5%% of layer max:                       %.1f%% of columns", 100*float64(below5)/float64(totCols))
	t.Logf("  RAW |gated| effective width (95%% L2):            %.1f%% of dFF", 100*rawCov/float64(nFFN))
	t.Logf("  ‖down‖-WEIGHTED effective width (95%% L2):        %.1f%% of dFF   <- the real output sparsity", 100*wtdCov/float64(nFFN))
	t.Logf("  => ~%.0f%% of columns carry <5%% of the OUTPUT — the honest skippable up+down fraction", 100*(1-wtdCov/float64(nFFN)))
}

// TestFFNDropArgmaxCeilingRealE2B is #364's DEFINITIVE ceiling: actually drop the
// bottom-K% of |gated| columns before the down projection and measure when the
// emitted token sequence first diverges from the full-precision baseline. That
// largest all-match drop-% is the TRUE exploitable sparsity (per-layer error
// compounds over 35 layers, so the ~77% "carry <5%" is an upper bound, not it).
//
//	LEM_REAL_E2B=1 MLX_METALLIB_PATH=... go test -run TestFFNDropArgmaxCeilingRealE2B -v ./engine/metal/
func TestFFNDropArgmaxCeilingRealE2B(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set LEM_REAL_E2B=1 to run the real e2b-4bit FFN-drop ceiling (loads ~2.7GB)")
	}
	dir := resolveE2B4bitDir(t)
	lm, dm, err := loadRegistered(dir)
	if err != nil {
		t.Fatalf("loadRegistered: %v", err)
	}
	defer func() { _ = dm.Close() }()
	sb, err := buildShardBuffers(dm)
	if err != nil {
		t.Fatalf("buildShardBuffers: %v", err)
	}
	defer func() { _ = sb.Close() }()
	qm, err := loadedToQuant(lm, lm.Embed.GroupSize, lm.Embed.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}

	prompt := []int32{2, 1000, 2500, 4000, 8000, 16000}
	const N = 24
	genWith := func(frac float64, byGate bool) []int32 {
		prevFrac, prevICB, prevBy := ffnDropFrac, icbDisabledForTest, ffnDropByGate
		icbDisabledForTest = true // the drop lives in the non-ICB stepToken path
		ffnDropFrac = frac
		ffnDropByGate = byGate
		defer func() { ffnDropFrac = prevFrac; icbDisabledForTest = prevICB; ffnDropByGate = prevBy }()
		s, e := newArchQuantSessionShards(qm, lm.Arch, 320, sb)
		if e != nil {
			t.Fatalf("newArchQuantSessionShards: %v", e)
		}
		if e := s.PrefillTokens(prompt); e != nil {
			t.Fatalf("prefill: %v", e)
		}
		toks, e := s.GenerateFromCache(N, -1)
		if e != nil {
			t.Fatalf("generate (drop %.2f byGate=%v): %v", frac, byGate, e)
		}
		return toks
	}

	base := genWith(0, false)
	t.Logf("=== #364 FFN-drop argmax ceiling — real e2b-4bit, %d tokens ===", N)
	t.Logf("  baseline (drop 0%%): %v", base)
	sweep := func(byGate bool, label string) {
		t.Logf("  --- ranked by %s ---", label)
		for _, frac := range []float64{0.30, 0.35, 0.40, 0.45, 0.50} {
			got := genWith(frac, byGate)
			match := 0
			for match < len(base) && match < len(got) && base[match] == got[match] {
				match++
			}
			verdict := "ALL MATCH"
			if match < len(base) {
				verdict = "diverges"
			}
			t.Logf("    drop %2.0f%%: %2d/%d tokens match — %s", frac*100, match, len(base), verdict)
		}
	}
	// EXACT |gated| = the down-skip ceiling (needs up already computed). gate-only
	// |gelu(gate)| = the predictor a real UP-skipping kernel uses (scored before up);
	// the gap between the two is the cost of predicting deadness from gate alone.
	sweep(false, "|gated| (exact — DOWN-skip ceiling)")
	sweep(true, "|gelu(gate)| (gate-only PREDICTOR — UP+down-skip ceiling)")
	t.Logf("  => how much of the exact ceiling survives when deadness is predicted from gate alone")
}
