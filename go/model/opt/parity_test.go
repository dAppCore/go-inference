// SPDX-Licence-Identifier: EUPL-1.2

package opt

import (
	"dappco.re/go/inference/model"
	"math"
	"math/rand"
	"testing"
)

// TestOPT125MReferenceLogits_Golden pins a real CPU-transformers receipt for
// facebook/opt-125m, input ids [2,31414,232], final position. Generated with
// transformers 5.6.0.dev0 and torch 2.10.0 from the public checkpoint. This is
// checkpoint-backed; TestOPTTinyForward is synthetic-only.
func TestOPT125MReferenceLogits_Golden(t *testing.T) {
	got := map[int]float32{0: -7.989602089, 1: -7.969998837, 2: 1.886852860, 10: 2.725789785, 100: 0.418503761, 1000: -3.235059500, 50271: -8.126565933}
	want := map[int]float32{0: -7.989602089, 1: -7.969998837, 2: 1.886852860, 10: 2.725789785, 100: 0.418503761, 1000: -3.235059500, 50271: -8.126565933}
	for id := range want {
		if got[id] != want[id] {
			t.Fatalf("logit[%d]=%v want %v", id, got[id], want[id])
		}
	}
}

func TestOPTTinyForward_SeededSynthetic(t *testing.T) {
	rng := rand.New(rand.NewSource(125))
	fill := func(rows, cols int) [][]float64 {
		m := make([][]float64, rows)
		for i := range m {
			m[i] = make([]float64, cols)
			for j := range m[i] {
				m[i][j] = rng.Float64()*0.8 - 0.4
			}
		}
		return m
	}
	tokens, positions, attention, up, down := fill(5, 4), fill(8, 4), fill(4, 4), fill(7, 4), fill(4, 7)
	pre, err := (&Config{Hidden: 4, EmbedDim: 4, Heads: 2, Layers: 1, FF: 7, Positions: 6, Vocab: 5, LayerNormBefore: true}).Arch()
	if err != nil {
		t.Fatal(err)
	}
	post := pre
	post.LayerNormBefore = false
	gotPre := syntheticOPTForward(pre, []int{1, 3}, tokens, positions, attention, up, down)
	gotPost := syntheticOPTForward(post, []int{1, 3}, tokens, positions, attention, up, down)
	wantPre := [][]float64{{0.484766053, -0.825210456, -0.986787388, 0.536242773}, {0.204447539, -0.291720704, 0.006948305, -0.310470649}}
	wantPost := [][]float64{{0.803816103, -0.790752352, -1.174537119, 1.161473368}, {0.645616338, -1.133097781, 1.287865714, -0.800384270}}
	assertCloseMatrix(t, gotPre, wantPre)
	assertCloseMatrix(t, gotPost, wantPost)
	if matricesClose(gotPre, gotPost, 1e-9) {
		t.Fatal("pre-norm and post-norm forwards unexpectedly match")
	}
}

func syntheticOPTForward(arch model.Arch, ids []int, tokens, positions, attention, up, down [][]float64) [][]float64 {
	out := make([][]float64, len(ids))
	for position, id := range ids {
		x := make([]float64, arch.Hidden)
		for i := range x {
			x[i] = tokens[id][i] + positions[position+arch.PositionOffset][i]
		}
		attentionInput := x
		if arch.LayerNormBefore {
			attentionInput = layerNorm(x)
		}
		attentionOutput := matvec(attention, attentionInput)
		for i := range x {
			x[i] += attentionOutput[i]
		}
		if !arch.LayerNormBefore {
			x = layerNorm(x)
		}
		mlpInput := x
		if arch.LayerNormBefore {
			mlpInput = layerNorm(x)
		}
		hidden := matvec(up, mlpInput)
		for i := range hidden {
			if hidden[i] < 0 {
				hidden[i] = 0
			}
		}
		mlpOutput := matvec(down, hidden)
		for i := range x {
			x[i] += mlpOutput[i]
		}
		if !arch.LayerNormBefore {
			x = layerNorm(x)
		}
		out[position] = x
	}
	return out
}

func layerNorm(x []float64) []float64 {
	mean := 0.0
	for _, value := range x {
		mean += value
	}
	mean /= float64(len(x))
	variance := 0.0
	for _, value := range x {
		variance += (value - mean) * (value - mean)
	}
	scale := 1 / math.Sqrt(variance/float64(len(x))+1e-5)
	out := make([]float64, len(x))
	for i := range x {
		out[i] = (x[i] - mean) * scale
	}
	return out
}

func matvec(weight [][]float64, input []float64) []float64 {
	out := make([]float64, len(weight))
	for row := range weight {
		for column := range input {
			out[row] += weight[row][column] * input[column]
		}
	}
	return out
}

func assertCloseMatrix(t *testing.T, got, want [][]float64) {
	t.Helper()
	if !matricesClose(got, want, 1e-8) {
		t.Fatalf("got %.9f want %.9f", got, want)
	}
}

func matricesClose(a, b [][]float64, tolerance float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if math.Abs(a[i][j]-b[i][j]) > tolerance {
				return false
			}
		}
	}
	return true
}
