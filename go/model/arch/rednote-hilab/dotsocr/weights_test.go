// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// TestNumel_Good hand-verifies the element-count product over a multi-dim shape.
func TestNumel_Good(t *testing.T) {
	if n := numel([]int{2, 3, 4}); n != 24 {
		t.Fatalf("numel([2,3,4]) = %d, want 24", n)
	}
}

// TestNumel_Ugly proves a scalar (empty shape) has exactly one element — the empty-product
// identity, distinct from a shape containing an explicit 0 dimension.
func TestNumel_Ugly(t *testing.T) {
	if n := numel(nil); n != 1 {
		t.Fatalf("numel(nil) = %d, want 1 (the empty product)", n)
	}
}

// TestSplitLinear_Good hand-verifies the fused-QKV row-range carve: a [6,2] fused weight (3
// stacked [2,2] blocks) splits into Q=rows[0:2), K=rows[2:4), V=rows[4:6) — the exact slicing
// VisionAttnWeights' doc comment describes.
func TestSplitLinear_Good(t *testing.T) {
	fused := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} // 6 rows x 2 cols
	q := splitLinear(fused, 2, 0, 2)
	k := splitLinear(fused, 2, 2, 2)
	v := splitLinear(fused, 2, 4, 2)
	if d := maxAbsDiff32(t, q.Weight, []float32{1, 2, 3, 4}); d != 0 {
		t.Fatalf("splitLinear Q = %v, want [1 2 3 4]", q.Weight)
	}
	if d := maxAbsDiff32(t, k.Weight, []float32{5, 6, 7, 8}); d != 0 {
		t.Fatalf("splitLinear K = %v, want [5 6 7 8]", k.Weight)
	}
	if d := maxAbsDiff32(t, v.Weight, []float32{9, 10, 11, 12}); d != 0 {
		t.Fatalf("splitLinear V = %v, want [9 10 11 12]", v.Weight)
	}
}

// TestLoadWeights_Bad proves a nil config refuses.
func TestLoadWeights_Bad(t *testing.T) {
	if _, err := LoadWeights(nil, nil); err == nil {
		t.Fatal("LoadWeights accepted a nil config")
	}
}

// TestLoadWeights_Ugly proves an incomplete config (zero-value geometry, no vision_config)
// refuses distinctly from the outright-nil case above — the same "never guessed" discipline
// whisper.LoadWeights' geometry guard follows.
func TestLoadWeights_Ugly(t *testing.T) {
	if _, err := LoadWeights(map[string]safetensors.Tensor{}, &Config{}); err == nil {
		t.Fatal("LoadWeights accepted an empty config with no geometry")
	}
}
