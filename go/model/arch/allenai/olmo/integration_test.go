// SPDX-Licence-Identifier: EUPL-1.2

package olmo_test

import (
	"math"
	"testing"

	"dappco.re/go/inference/model"
)

func matVec(w, x []float32) []float32 {
	n := len(x)
	out := make([]float32, len(w)/n)
	for row := range out {
		for col := range n {
			out[row] += w[row*n+col] * x[col]
		}
	}
	return out
}

func layerNorm(x []float32) []float32 {
	var mean float32
	for _, v := range x {
		mean += v
	}
	mean /= float32(len(x))
	var variance float32
	for _, v := range x {
		d := v - mean
		variance += d * d
	}
	variance /= float32(len(x))
	out := make([]float32, len(x))
	denom := float32(math.Sqrt(float64(variance + 1e-5)))
	for i, v := range x {
		out[i] = (v - mean) / denom
	}
	return out
}

func rmsNorm(weight []float32) func([]float32) []float32 {
	return func(x []float32) []float32 {
		var square float32
		for _, v := range x {
			square += v * v
		}
		denom := float32(math.Sqrt(float64(square/float32(len(x)) + 1e-6)))
		out := make([]float32, len(x))
		for i, v := range x {
			out[i] = v / denom * weight[i]
		}
		return out
	}
}

var (
	tinyInput = []float32{-0.00806256103, -0.0345092604, 0.0925834038, -0.253370478}
	tinyAttn  = []float32{-0.265145328, 0.252354927, -0.202781296, 0.214676559, 0.293175971, 0.244213266, -0.0706465505, -0.263422744, -0.200398476, 0.126543195, -0.174804816, 0.223752553, -0.190828619, -0.172748947, 0.168511261, 0.170594519}
	tinyMLP   = []float32{-0.195689427, -0.120975183, 0.107949015, -0.165486562, 0.283097764, -0.231407574, 0.079232537, -0.19227205, -0.143394567, 0.137255545, -0.180857457, 0.286503483, 0.129403576, -0.126003739, -0.189599133, -0.124032639}
)

func assertGolden(t *testing.T, placement model.NormPlacement, attnNorm, mlpNorm func([]float32) []float32, want []float32) {
	t.Helper()
	r := model.ApplyResidualOrder(placement, tinyInput, attnNorm, mlpNorm,
		func(x []float32) []float32 { return matVec(tinyAttn, x) },
		func(x []float32) []float32 { return matVec(tinyMLP, x) })
	if !r.OK {
		t.Fatal(r.Error())
	}
	for i, got := range r.Value.([]float32) {
		if math.Abs(float64(got-want[i])) > 2e-6 {
			t.Fatalf("output[%d] = %.9g, want %.9g", i, got, want[i])
		}
	}
}

// TestTinyOLMoForward_Golden is a seeded-synthetic-only residual-order golden.
// It uses varied deterministic fills and the generation-1 non-parametric LayerNorm.
func TestTinyOLMoForward_Golden(t *testing.T) {
	assertGolden(t, model.NormPlacementPre, layerNorm, layerNorm, []float32{-0.688025702, -0.172675725, -0.157273857, -0.601970156})
}

// TestTinyOLMo2Forward_Golden is a seeded-synthetic-only residual-order golden.
// OLMo 2 post-normalises each sublayer output before adding the residual.
func TestTinyOLMo2Forward_Golden(t *testing.T) {
	attnWeights := []float32{0.867457658, 1.12298362, 0.99735693, 0.989847703}
	mlpWeights := []float32{0.801515254, 0.991668093, 1.08526789, 0.944318692}
	assertGolden(t, model.NormPlacementPost, rmsNorm(attnWeights), rmsNorm(mlpWeights), []float32{-0.902442263, -0.774001503, 0.0169275939, -0.483704735})
}
