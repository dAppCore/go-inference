// SPDX-Licence-Identifier: EUPL-1.2

package audio

import "testing"

type fixtureGEMM struct {
	out []float32
	ok  bool
}

func (gemm fixtureGEMM) MatMul([]float32, []float32, int, int, int, bool) ([]float32, bool) {
	return append([]float32(nil), gemm.out...), gemm.ok
}

func TestGEMM_Good(t *testing.T) {
	got := matMulNNWith(fixtureGEMM{out: []float32{19, 22, 43, 50}, ok: true},
		[]float32{1, 2, 3, 4}, []float32{5, 6, 7, 8}, 2, 2, 2)
	want := []float32{19, 22, 43, 50}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("device result[%d]=%g, want %g", i, got[i], want[i])
		}
	}
}

func TestGEMM_Bad(t *testing.T) {
	got := matMulNNWith(fixtureGEMM{ok: false}, []float32{1, 2, 3, 4}, []float32{5, 6, 7, 8}, 2, 2, 2)
	want := matMulNN([]float32{1, 2, 3, 4}, []float32{5, 6, 7, 8}, 2, 2, 2)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("fallback result[%d]=%g, want host %g", i, got[i], want[i])
		}
	}
}

func TestGEMM_Ugly(t *testing.T) {
	got := matMulNTWith(fixtureGEMM{out: []float32{1}, ok: true},
		[]float32{1, 2, 3, 4}, []float32{5, 6, 7, 8}, 2, 2, 2)
	want := matMulNT([]float32{1, 2, 3, 4}, []float32{5, 6, 7, 8}, 2, 2, 2)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("short device result did not fall back at [%d]: got %g, want %g", i, got[i], want[i])
		}
	}
}
