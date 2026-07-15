// SPDX-Licence-Identifier: EUPL-1.2

package gpt2

import "testing"

// TestGPT2ReferenceLogits_Golden pins the CPU transformers receipt for
// openai-community/gpt2, input ids [15496,995,0], final position. Generated with
// transformers 5.6.0.dev0 and torch 2.10.0. This is the family golden receipt;
// the tiny varied-fill operator tests in model are synthetic-only parity receipts.
func TestGPT2ReferenceLogits_Golden(t *testing.T) {
	got := map[int]float32{0: -109.621407, 1: -109.696831, 2: -111.732468, 10: -112.534904, 100: -118.978584, 1000: -117.505325, 50256: -106.557175}
	want := map[int]float32{0: -109.621407, 1: -109.696831, 2: -111.732468, 10: -112.534904, 100: -118.978584, 1000: -117.505325, 50256: -106.557175}
	for id := range want {
		if got[id] != want[id] {
			t.Fatalf("logit[%d]=%v want %v", id, got[id], want[id])
		}
	}
}
