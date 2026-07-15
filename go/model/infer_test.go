// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// TestInfer_WeightAny_Good covers the ordinary case: the first present name in the list
// wins, even when it isn't the first argument tried in isolation (a later config alias
// resolves through the same call).
func TestInfer_WeightAny_Good(t *testing.T) {
	weights := map[string]safetensors.Tensor{"b.weight": {Shape: []int{4}}}
	got, ok := WeightAny(weights, "a.weight", "b.weight", "c.weight")
	if !ok {
		t.Fatal("WeightAny: not found, want the second candidate name to resolve")
	}
	if got.Shape[0] != 4 {
		t.Fatalf("WeightAny returned %+v, want the b.weight tensor", got)
	}
}

// TestInfer_WeightAny_Bad covers every candidate absent: ok=false and a zero Tensor,
// rather than a nil-shape value mistaken for present.
func TestInfer_WeightAny_Bad(t *testing.T) {
	_, ok := WeightAny(map[string]safetensors.Tensor{}, "a.weight", "b.weight")
	if ok {
		t.Fatal("WeightAny(no candidates present) = ok, want not-found")
	}
}

// TestInfer_WeightAny_Ugly covers a no-names call: nothing to look up, so it must report
// not-found rather than panic on an empty variadic.
func TestInfer_WeightAny_Ugly(t *testing.T) {
	_, ok := WeightAny(map[string]safetensors.Tensor{"a.weight": {}})
	if ok {
		t.Fatal("WeightAny(no names) = ok, want not-found")
	}
}

// TestInfer_InferHeadDim_Good covers the ordinary divide-evenly case: a q_proj of
// [numHeads*headDim, hidden] rows over numHeads gives headDim.
func TestInfer_InferHeadDim_Good(t *testing.T) {
	weights := map[string]safetensors.Tensor{"q_proj.weight": {Shape: []int{8 * 64, 2048}}}
	if got := InferHeadDim(weights, "q_proj.weight", 8); got != 64 {
		t.Fatalf("InferHeadDim = %d, want 64", got)
	}
}

// TestInfer_InferHeadDim_Bad covers the absent-weight case: the caller then keeps
// whatever the config declared, so InferHeadDim returns 0 (never guesses).
func TestInfer_InferHeadDim_Bad(t *testing.T) {
	if got := InferHeadDim(map[string]safetensors.Tensor{}, "q_proj.weight", 8); got != 0 {
		t.Fatalf("InferHeadDim(absent) = %d, want 0", got)
	}
}

// TestInfer_InferHeadDim_Ugly covers a row count that does NOT divide evenly by
// numHeads (a malformed or mismatched checkpoint): the shape can't encode a clean head
// dim, so InferHeadDim returns 0 rather than a truncated/wrong value.
func TestInfer_InferHeadDim_Ugly(t *testing.T) {
	weights := map[string]safetensors.Tensor{"q_proj.weight": {Shape: []int{513, 2048}}}
	if got := InferHeadDim(weights, "q_proj.weight", 8); got != 0 {
		t.Fatalf("InferHeadDim(non-dividing rows) = %d, want 0", got)
	}
	if got := InferHeadDim(weights, "q_proj.weight", 0); got != 0 {
		t.Fatalf("InferHeadDim(numHeads=0) = %d, want 0", got)
	}
}

// TestInfer_InferOutFeaturesPerN_Good covers the ordinary per-layer-stacked projection:
// flattened out-features (product of every dim but the last) divided by n gives the
// per-layer width.
func TestInfer_InferOutFeaturesPerN_Good(t *testing.T) {
	// shape [numLayers=30, perLayerHidden=256, hidden=2048] → outFeatures = 30*256 = 7680
	weights := map[string]safetensors.Tensor{"per_layer_proj.weight": {Shape: []int{30, 256, 2048}}}
	if got := InferOutFeaturesPerN(weights, "per_layer_proj.weight", 30); got != 256 {
		t.Fatalf("InferOutFeaturesPerN = %d, want 256", got)
	}
}

// TestInfer_InferOutFeaturesPerN_Bad covers n<=0: a meaningless divisor, so the function
// returns 0 without dividing by zero.
func TestInfer_InferOutFeaturesPerN_Bad(t *testing.T) {
	weights := map[string]safetensors.Tensor{"per_layer_proj.weight": {Shape: []int{30, 256}}}
	if got := InferOutFeaturesPerN(weights, "per_layer_proj.weight", 0); got != 0 {
		t.Fatalf("InferOutFeaturesPerN(n=0) = %d, want 0", got)
	}
	if got := InferOutFeaturesPerN(weights, "per_layer_proj.weight", -1); got != 0 {
		t.Fatalf("InferOutFeaturesPerN(n<0) = %d, want 0", got)
	}
}

// TestInfer_InferOutFeaturesPerN_Ugly covers a rank-1 shape (no leading dims to flatten,
// so len(shape)<2) and a non-dividing outFeatures — both must return 0, never a wrong
// truncated width.
func TestInfer_InferOutFeaturesPerN_Ugly(t *testing.T) {
	rank1 := map[string]safetensors.Tensor{"vec.weight": {Shape: []int{128}}}
	if got := InferOutFeaturesPerN(rank1, "vec.weight", 4); got != 0 {
		t.Fatalf("InferOutFeaturesPerN(rank-1 shape) = %d, want 0", got)
	}
	nonDividing := map[string]safetensors.Tensor{"per_layer_proj.weight": {Shape: []int{7, 256}}}
	if got := InferOutFeaturesPerN(nonDividing, "per_layer_proj.weight", 30); got != 0 {
		t.Fatalf("InferOutFeaturesPerN(non-dividing) = %d, want 0", got)
	}
}
