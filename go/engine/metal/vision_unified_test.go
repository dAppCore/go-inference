// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/vision"
)

// vision_unified_test.go gates the encoder-free (gemma4_unified) vision
// embedder: UnifiedVisionProject against a float64 host reference of the
// documented composition (LayerNorm → dense+bias → LayerNorm → factorised
// pos add → LayerNorm → scale-free RMSNorm → projection), on both the bf16
// and affine-quant linear paths, plus the real-snapshot assemble smoke.

func unifiedTestBF16(vals []float64) []byte {
	out := make([]byte, len(vals)*bf16Size)
	for i, v := range vals {
		b := f32ToBF16(float32(v))
		out[2*i] = byte(b)
		out[2*i+1] = byte(b >> 8)
	}
	return out
}

func unifiedTestF64(b []byte) []float64 {
	out := make([]float64, len(b)/2)
	for i := range out {
		out[i] = float64(bf16ToF32(b[2*i], b[2*i+1]))
	}
	return out
}

func unifiedTestRand(rng *rand.Rand, n int, scale float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = (rng.Float64() - 0.5) * scale
	}
	return out
}

func unifiedRefLayerNorm(x, w, b []float64, rows, dim int, eps float64) {
	for r := range rows {
		row := x[r*dim : (r+1)*dim]
		var mean float64
		for _, v := range row {
			mean += v
		}
		mean /= float64(dim)
		var vr float64
		for _, v := range row {
			d := v - mean
			vr += d * d
		}
		inv := 1 / math.Sqrt(vr/float64(dim)+eps)
		for c := range row {
			row[c] = (row[c]-mean)*inv*w[c] + b[c]
		}
	}
}

func unifiedRefMatmul(x, w []float64, rows, inDim, outDim int, bias []float64) []float64 {
	out := make([]float64, rows*outDim)
	for r := range rows {
		for o := range outDim {
			var acc float64
			for k := range inDim {
				acc += x[r*inDim+k] * w[o*inDim+k]
			}
			if bias != nil {
				acc += bias[o]
			}
			out[r*outDim+o] = acc
		}
	}
	return out
}

func TestUnifiedVisionProjectParity(t *testing.T) {
	requireNativeRuntime(t)
	rng := rand.New(rand.NewSource(280))
	const n, patchDim, mm, hidden, posemb = 5, 507, 256, 320, 16
	const lnEps, rmsEps = 1e-5, 1e-6

	patches := unifiedTestRand(rng, n*patchDim, 2)
	ln1w := unifiedTestRand(rng, patchDim, 1.5)
	ln1b := unifiedTestRand(rng, patchDim, 0.2)
	denseW := unifiedTestRand(rng, mm*patchDim, 0.08)
	denseB := unifiedTestRand(rng, mm, 0.2)
	ln2w := unifiedTestRand(rng, mm, 1.5)
	ln2b := unifiedTestRand(rng, mm, 0.2)
	pos := unifiedTestRand(rng, posemb*2*mm, 0.4)
	posw := unifiedTestRand(rng, mm, 1.5)
	posb := unifiedTestRand(rng, mm, 0.2)
	projW := unifiedTestRand(rng, hidden*mm, 0.08)
	positions := make([]int32, n*2)
	for r := range n {
		positions[r*2] = int32(r % posemb)
		positions[r*2+1] = int32((r * 3) % posemb)
	}
	positions[(n-1)*2] = -1 // padded row: no row-axis contribution
	// bf16-quantise every operand FIRST so the reference computes on exactly
	// the values the engine consumes.
	req := func(v []float64) []float64 { return unifiedTestF64(unifiedTestBF16(v)) }
	patches, ln1w, ln1b = req(patches), req(ln1w), req(ln1b)
	denseW, denseB, ln2w, ln2b = req(denseW), req(denseB), req(ln2w), req(ln2b)
	pos, posw, posb, projW = req(pos), req(posw), req(posb), req(projW)

	// reference forward in f64
	ref := make([]float64, len(patches))
	copy(ref, patches)
	unifiedRefLayerNorm(ref, ln1w, ln1b, n, patchDim, lnEps)
	ref = unifiedRefMatmul(ref, denseW, n, patchDim, mm, denseB)
	unifiedRefLayerNorm(ref, ln2w, ln2b, n, mm, lnEps)
	for r := range n {
		for axis := range 2 {
			idx := int(positions[r*2+axis])
			if idx < 0 {
				continue
			}
			for c := range mm {
				ref[r*mm+c] += pos[(idx*2+axis)*mm+c]
			}
		}
	}
	unifiedRefLayerNorm(ref, posw, posb, n, mm, lnEps)
	for r := range n {
		var ss float64
		for c := range mm {
			ss += ref[r*mm+c] * ref[r*mm+c]
		}
		inv := 1 / math.Sqrt(ss/float64(mm)+rmsEps)
		for c := range mm {
			ref[r*mm+c] *= inv
		}
	}
	ref = unifiedRefMatmul(ref, projW, n, mm, hidden, nil)

	uv := &vision.Unified{
		PatchLN1W: unifiedTestBF16(ln1w), PatchLN1B: unifiedTestBF16(ln1b),
		PatchDense: vision.Linear{
			Weight: unifiedTestBF16(denseW), Bias: unifiedTestBF16(denseB),
			OutDim: mm, InDim: patchDim,
		},
		PatchLN2W: unifiedTestBF16(ln2w), PatchLN2B: unifiedTestBF16(ln2b),
		PosEmbedding: unifiedTestBF16(pos),
		PosNormW:     unifiedTestBF16(posw), PosNormB: unifiedTestBF16(posb),
		Projection: vision.Linear{
			Weight: unifiedTestBF16(projW),
			OutDim: hidden, InDim: mm,
		},
		Cfg: vision.UnifiedConfig{
			MMEmbedDim: mm, TextHidden: hidden, PosembSize: posemb,
			PatchSize: 16, ModelPatchSize: 13, PoolKernel: 3, MaxSoftTokens: 280,
			LayerNormEps: lnEps, RMSNormEps: rmsEps,
		},
	}
	got, err := UnifiedVisionProject(uv, unifiedTestBF16(patches), positions, n)
	if err != nil {
		t.Fatalf("UnifiedVisionProject: %v", err)
	}
	gotF := unifiedTestF64(got)
	if len(gotF) != len(ref) {
		t.Fatalf("output length = %d, want %d", len(gotF), len(ref))
	}
	var maxDiff float64
	for i := range ref {
		if d := math.Abs(gotF[i] - ref[i]); d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 0.06 {
		t.Fatalf("bf16 lane maxDiff = %.5f vs f64 reference", maxDiff)
	}
	t.Logf("bf16 lane maxDiff = %.5f", maxDiff)
}

// TestAssembleUnifiedVision12B smokes the real 12B snapshot end-to-end through
// model.Load: the unified payload must assemble with the declared geometry and
// project a synthetic patch batch without error.
func TestAssembleUnifiedVision12B(t *testing.T) {
	requireNativeRuntime(t)
	dir := os.Getenv("GEMMA4_12B_DIR")
	if dir == "" {
		t.Skip("GEMMA4_12B_DIR not set")
	}
	lm, dm, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load: %v", err)
	}
	defer func() { _ = dm.Close() }()
	uv := lm.UnifiedVision
	if uv == nil {
		t.Fatal("12B snapshot produced no LoadedUnifiedVision")
	}
	cfg := uv.Cfg
	if cfg.ModelPatchSize != 48 || cfg.MMEmbedDim != 3840 || cfg.PosembSize != 1120 || cfg.MaxSoftTokens != 280 {
		t.Fatalf("unexpected geometry: %+v", cfg)
	}
	n := 4
	patchDim := cfg.ModelPatchSize * cfg.ModelPatchSize * 3
	rng := rand.New(rand.NewSource(12))
	patches := unifiedTestBF16(unifiedTestRand(rng, n*patchDim, 1))
	positions := []int32{0, 0, 0, 1, 1, 0, 1, 1}
	features, err := UnifiedVisionProject(uv, patches, positions, n)
	if err != nil {
		t.Fatalf("UnifiedVisionProject: %v", err)
	}
	if len(features) != n*cfg.TextHidden*bf16Size {
		t.Fatalf("features bytes = %d, want %d", len(features), n*cfg.TextHidden*bf16Size)
	}
	nan, _ := bf16NaNScanBytes(features)
	if nan > 0 {
		t.Fatalf("features carry %d NaN", nan)
	}
}
