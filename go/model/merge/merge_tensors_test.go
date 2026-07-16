// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"math"

	core "dappco.re/go"

	"dappco.re/go/inference/model/modelmgmt"
)

func writeShard(t *core.T, path string, values map[string][]float32) {
	t.Helper()
	tensors := make(map[string]modelmgmt.SafetensorsTensorInfo, len(values))
	data := make(map[string][]byte, len(values))
	for name, vals := range values {
		tensors[name] = modelmgmt.SafetensorsTensorInfo{Dtype: "F32", Shape: []int{len(vals)}}
		data[name] = modelmgmt.EncodeFloat32(vals)
	}
	requireResultOK(t, modelmgmt.WriteSafetensors(path, tensors, data))
}

func TestMergeTensors_IndexWeightFiles_Good(t *core.T) {
	dir := t.TempDir()
	shard0 := core.PathJoin(dir, "model-00000.safetensors")
	shard1 := core.PathJoin(dir, "model-00001.safetensors")
	writeShard(t, shard0, map[string][]float32{"a": {1, 2}})
	writeShard(t, shard1, map[string][]float32{"b": {3, 4}})

	index, err := indexWeightFiles([]string{shard0, shard1})
	core.RequireNoError(t, err)
	core.AssertEqual(t, []string{"a", "b"}, index.Names)
}

func TestMergeTensors_IndexWeightFiles_Bad(t *core.T) {
	index, err := indexWeightFiles([]string{core.PathJoin(t.TempDir(), "missing.safetensors")})
	core.AssertError(t, err, "read safetensors")
	core.AssertNil(t, index.Tensors)
}

func TestMergeTensors_IndexWeightFiles_Ugly(t *core.T) {
	dir := t.TempDir()
	shard0 := core.PathJoin(dir, "model-00000.safetensors")
	shard1 := core.PathJoin(dir, "model-00001.safetensors")
	writeShard(t, shard0, map[string][]float32{"dup": {1, 2}})
	writeShard(t, shard1, map[string][]float32{"dup": {3, 4}})

	_, err := indexWeightFiles([]string{shard0, shard1})
	core.AssertError(t, err, "duplicate tensor")
}

// TestMergeTensors_ShapeElements_Good covers a 1-D, 2-D, and 3-D shape —
// the product-of-dimensions rule generalises past the common matrix case.
func TestMergeTensors_ShapeElements_Good(t *core.T) {
	core.AssertEqual(t, 5, shapeElements([]int{5}))
	core.AssertEqual(t, 6, shapeElements([]int{2, 3}))
	core.AssertEqual(t, 24, shapeElements([]int{2, 3, 4}))
}

// TestMergeTensors_ShapeElements_Bad covers the scalar convention (empty
// shape means 1 element) for a nil shape, an empty-but-non-nil shape, and
// an explicit all-ones shape (same product, different representation).
func TestMergeTensors_ShapeElements_Bad(t *core.T) {
	core.AssertEqual(t, 1, shapeElements(nil))
	core.AssertEqual(t, 1, shapeElements([]int{}))
	core.AssertEqual(t, 1, shapeElements([]int{1, 1}))
}

// TestMergeTensors_ShapeElements_Ugly covers a zero-sized dimension
// collapsing the product to 0, at three different positions in the shape.
func TestMergeTensors_ShapeElements_Ugly(t *core.T) {
	core.AssertEqual(t, 0, shapeElements([]int{0, 5}))
	core.AssertEqual(t, 0, shapeElements([]int{5, 0, 3}))
	core.AssertEqual(t, 0, shapeElements([]int{0}))
}

func TestMergeTensors_GatherTensorEntries_Good(t *core.T) {
	indexes := []sourceIndex{
		{Names: []string{"w"}, Tensors: map[string]tensorEntry{"w": {DType: "F32", Shape: []int{2}}}},
		{Names: []string{"w"}, Tensors: map[string]tensorEntry{"w": {DType: "F32", Shape: []int{2}}}},
	}
	entries, complete := gatherTensorEntries(indexes, "w")
	core.AssertTrue(t, complete)
	core.AssertLen(t, entries, 2)
}

func TestMergeTensors_GatherTensorEntries_Bad(t *core.T) {
	indexes := []sourceIndex{
		{Names: []string{"w"}, Tensors: map[string]tensorEntry{"w": {DType: "F32", Shape: []int{2}}}},
		{Names: []string{}, Tensors: map[string]tensorEntry{}},
	}
	entries, complete := gatherTensorEntries(indexes, "w")
	core.AssertFalse(t, complete)
	core.AssertLen(t, entries, 1)
}

func TestMergeTensors_GatherTensorEntries_Ugly(t *core.T) {
	indexes := []sourceIndex{
		{Names: []string{"w"}, Tensors: map[string]tensorEntry{"w": {DType: "F32", Shape: []int{2}}}},
		{Names: []string{"w"}, Tensors: map[string]tensorEntry{"w": {DType: "F32", Shape: []int{3}}}},
	}
	entries, complete := gatherTensorEntries(indexes, "w")
	core.AssertFalse(t, complete)
	core.AssertLen(t, entries, 2)
}

func TestMergeTensors_DecodeAll_Good(t *core.T) {
	entries := []tensorEntry{
		{DType: "F32", Shape: []int{1}, Raw: modelmgmt.EncodeFloat32([]float32{1})},
		{DType: "F32", Shape: []int{1}, Raw: modelmgmt.EncodeFloat32([]float32{2})},
	}
	values, err := decodeAll(entries)
	core.RequireNoError(t, err)
	core.AssertEqual(t, [][]float32{{1}, {2}}, values)
}

func TestMergeTensors_DecodeAll_Bad(t *core.T) {
	entries := []tensorEntry{{DType: "I64", Shape: []int{1}, Raw: []byte{1, 2, 3, 4}}}
	_, err := decodeAll(entries)
	core.AssertError(t, err)
}

func TestMergeTensors_DecodeAll_Ugly(t *core.T) {
	values, err := decodeAll(nil)
	core.RequireNoError(t, err)
	core.AssertEmpty(t, values)
}

func TestMergeTensors_MergeTensorValues_Good(t *core.T) {
	got, err := mergeTensorValues([][]float32{{1, 2}, {3, 4}}, MethodLinear, 0, []float64{0.5, 0.5})
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{2, 3}, got)
}

func TestMergeTensors_MergeTensorValues_Bad(t *core.T) {
	got, err := mergeTensorValues([][]float32{{1, 0}, {0, 1}}, MethodSLERP, 0, nil)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{1, 0}, got)
}

func TestMergeTensors_MergeTensorValues_Ugly(t *core.T) {
	got, err := mergeTensorValues([][]float32{{1}}, Method("ties"), 0, []float64{1})
	core.AssertError(t, err, "unsupported model merge method")
	core.AssertNil(t, got)
}

func TestMergeTensors_LinearMerge_Good(t *core.T) {
	got, err := linearMerge([][]float32{{1, 2}, {3, 4}}, []float64{0.25, 0.75})
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{2.5, 3.5}, got)
}

func TestMergeTensors_LinearMerge_Bad(t *core.T) {
	got, err := linearMerge([][]float32{{1, 2}}, []float64{0.5, 0.5})
	core.AssertErrorIs(t, err, errWeightsSourceCount)
	core.AssertNil(t, got)
}

func TestMergeTensors_LinearMerge_Ugly(t *core.T) {
	got, err := linearMerge([][]float32{{1, 2}, {3}}, []float64{0.5, 0.5})
	core.AssertErrorIs(t, err, errLinearLenMismatch)
	core.AssertNil(t, got)
}

// TestMergeTensors_LinearMerge_Empty covers both a nil and an empty-but-
// non-nil values slice — both must hit the same errNoTensors guard.
func TestMergeTensors_LinearMerge_Empty(t *core.T) {
	_, err := linearMerge(nil, nil)
	core.AssertErrorIs(t, err, errNoTensors)
	_, err2 := linearMerge([][]float32{}, []float64{})
	core.AssertErrorIs(t, err2, errNoTensors)
}

func TestMergeTensors_SlerpMerge_Good(t *core.T) {
	got, err := slerpMerge([][]float32{{1, 0}, {0, 1}}, 0.5)
	core.RequireNoError(t, err)
	core.AssertInDelta(t, 0.70710678, float64(got[0]), 1e-6)
	core.AssertInDelta(t, 0.70710678, float64(got[1]), 1e-6)
}

// TestMergeTensors_SlerpMerge_Bad covers both sides of the != 2 tensor-
// count guard: one tensor and three tensors.
func TestMergeTensors_SlerpMerge_Bad(t *core.T) {
	_, err := slerpMerge([][]float32{{1, 0}}, 0.5)
	core.AssertErrorIs(t, err, errSLERPNeedTwoTensors)
	_, err2 := slerpMerge([][]float32{{1, 0}, {0, 1}, {1, 1}}, 0.5)
	core.AssertErrorIs(t, err2, errSLERPNeedTwoTensors)
}

func TestMergeTensors_SlerpMerge_Ugly(t *core.T) {
	got, err := slerpMerge([][]float32{{1, 0}, {0, 1, 2}}, 0.5)
	core.AssertErrorIs(t, err, errSLERPLenMismatch)
	core.AssertNil(t, got)
}

func TestMergeTensors_SlerpMerge_Boundaries(t *core.T) {
	a, b := []float32{1, 0}, []float32{0, 1}
	got0, err := slerpMerge([][]float32{a, b}, 0)
	core.RequireNoError(t, err)
	core.AssertInDelta(t, 1, float64(got0[0]), 1e-6)
	core.AssertInDelta(t, 0, float64(got0[1]), 1e-6)

	got1, err := slerpMerge([][]float32{a, b}, 1)
	core.RequireNoError(t, err)
	core.AssertInDelta(t, 0, float64(got1[0]), 1e-6)
	core.AssertInDelta(t, 1, float64(got1[1]), 1e-6)
}

func TestMergeTensors_SlerpMerge_ZeroVectorFallback(t *core.T) {
	got, err := slerpMerge([][]float32{{0, 0}, {1, 1}}, 0.5)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{0.5, 0.5}, got)
}

func TestMergeTensors_SlerpMerge_NearParallelFallback(t *core.T) {
	got, err := slerpMerge([][]float32{{1, 0}, {1, 0.0001}}, 0.5)
	core.RequireNoError(t, err)
	core.AssertInDelta(t, 1.0, float64(got[0]), 0.01)
}

func TestMergeTensors_NormalizedWeights_Good(t *core.T) {
	weights, err := normalizedWeights([]Source{{Weight: 1}, {Weight: 3}})
	core.RequireNoError(t, err)
	core.AssertInDelta(t, 0.25, weights[0], 1e-12)
	core.AssertInDelta(t, 0.75, weights[1], 1e-12)
}

func TestMergeTensors_NormalizedWeights_Bad(t *core.T) {
	weights, err := normalizedWeights([]Source{{}, {}})
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float64{0.5, 0.5}, weights)
}

func TestMergeTensors_NormalizedWeights_Ugly(t *core.T) {
	got, err := normalizedWeights([]Source{{Weight: 1}, {Weight: -1}})
	core.AssertErrorIs(t, err, errMergeWeightsSumZero)
	core.AssertNil(t, got)
}

// TestMergeTensors_NormalizedWeights_NotFinite covers both non-finite
// float64 states the guard rejects: NaN and +Inf.
func TestMergeTensors_NormalizedWeights_NotFinite(t *core.T) {
	_, err := normalizedWeights([]Source{{Weight: math.NaN()}, {Weight: 1}})
	core.AssertErrorIs(t, err, errMergeWeightNotFinite)
	_, err2 := normalizedWeights([]Source{{Weight: math.Inf(1)}, {Weight: 1}})
	core.AssertErrorIs(t, err2, errMergeWeightNotFinite)
}

func TestMergeTensors_WriteProvenance_Good(t *core.T) {
	path := core.PathJoin(t.TempDir(), "provenance.json")
	core.RequireNoError(t, writeProvenance(path, Provenance{Version: 1, Method: MethodLinear, SkippedTensors: []string{"b", "a"}}))

	read := core.ReadFile(path)
	requireResultOK(t, read)
	var got Provenance
	requireResultOK(t, core.JSONUnmarshalString(string(read.Value.([]byte)), &got))
	core.AssertEqual(t, []string{"a", "b"}, got.SkippedTensors)
}

// TestMergeTensors_WriteProvenance_Bad documents that a missing parent
// directory fails the write and leaves no file behind.
func TestMergeTensors_WriteProvenance_Bad(t *core.T) {
	path := core.PathJoin(t.TempDir(), "missing-dir", "provenance.json")
	err := writeProvenance(path, Provenance{})
	core.AssertError(t, err)
	core.AssertFalse(t, coreFileExists(path))
}

func TestMergeTensors_WriteProvenance_Ugly(t *core.T) {
	path := core.PathJoin(t.TempDir(), "provenance.json")
	core.RequireNoError(t, writeProvenance(path, Provenance{}))
	read := core.ReadFile(path)
	requireResultOK(t, read)
	core.AssertContains(t, string(read.Value.([]byte)), `"version":0`)
}
