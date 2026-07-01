// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"

	core "dappco.re/go"

	"dappco.re/go/inference/modelmgmt"
)

func TestCompare_ComparePacks_Good(t *core.T) {
	base := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{
		"changed":   {1, 2, 3, 4},
		"unchanged": {5, 6},
	})
	tuned := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{
		"changed":   {2, 2, 3, 8},
		"unchanged": {5, 6},
	})

	result, err := ComparePacks(context.Background(), CompareOptions{Base: base, FineTuned: tuned})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, result.ChangedTensors)
	core.AssertEqual(t, 1, result.UnchangedTensors)
	core.AssertEqual(t, 2, result.ComparedTensors)
	core.AssertLen(t, result.Tensors, 1)
	core.AssertEqual(t, "changed", result.Tensors[0].Name)
	core.AssertInDelta(t, 1.25, result.Tensors[0].MeanAbsDelta, 1e-6) // |1-2|/4 mean over (1,0,0,4)/4
}

func TestCompare_ComparePacks_Bad(t *core.T) {
	tuned := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {1}})
	_, err := ComparePacks(context.Background(), CompareOptions{Base: Source{}, FineTuned: tuned})
	core.AssertError(t, err, "base")
}

func TestCompare_ComparePacks_Ugly(t *core.T) {
	base := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"only_base": {1, 2}})
	tuned := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"only_tuned": {3, 4}})

	result, err := ComparePacks(context.Background(), CompareOptions{Base: base, FineTuned: tuned})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, result.MissingInFineTuned)
	core.AssertEqual(t, 1, result.ExtraInFineTuned)
	core.AssertLen(t, result.Tensors, 2)
}

func TestCompare_ComparePacks_IncludeUnchanged(t *core.T) {
	base := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {1, 2}})
	tuned := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {1, 2}})

	result, err := ComparePacks(context.Background(), CompareOptions{Base: base, FineTuned: tuned, IncludeUnchanged: true})
	core.RequireNoError(t, err)
	core.AssertLen(t, result.Tensors, 1)
	core.AssertEqual(t, CompareStatusUnchanged, result.Tensors[0].Status)
}

func TestCompare_ComparePacks_MaxTensorReports(t *core.T) {
	base := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{
		"a": {1, 9}, "b": {1, 9}, "c": {1, 9},
	})
	tuned := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{
		"a": {2, 9}, "b": {2, 9}, "c": {2, 9},
	})

	result, err := ComparePacks(context.Background(), CompareOptions{Base: base, FineTuned: tuned, MaxTensorReports: 2})
	core.RequireNoError(t, err)
	core.AssertLen(t, result.Tensors, 2)
	core.AssertEqual(t, 3, result.ChangedTensors)
}

func TestCompare_ComparePacks_ShapeMismatch(t *core.T) {
	base := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {1, 2, 3}})
	tuned := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {1, 2}})

	result, err := ComparePacks(context.Background(), CompareOptions{Base: base, FineTuned: tuned})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, result.ShapeMismatches)
	core.AssertEqual(t, CompareStatusShapeMismatch, result.Tensors[0].Status)
}

func TestCompare_ComparePacks_ContextCancelled(t *core.T) {
	base := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {1}})
	tuned := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {2}})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := ComparePacks(ctx, CompareOptions{Base: base, FineTuned: tuned})
	core.AssertError(t, err)
}

func TestCompare_ValidateComparePack_Good(t *core.T) {
	core.AssertNoError(t, validateComparePack("base", Source{Root: "/x", WeightFiles: []string{"/x/model.safetensors"}}))
}

func TestCompare_ValidateComparePack_Bad(t *core.T) {
	core.AssertError(t, validateComparePack("base", Source{}), "root is required")
}

func TestCompare_ValidateComparePack_Ugly(t *core.T) {
	core.AssertError(t, validateComparePack("fine-tuned", Source{Root: "/x"}), "requires weight files")
}

func TestCompare_CompareTensorEntries_Good(t *core.T) {
	base := tensorEntry{DType: "F32", Shape: []int{2}, Raw: modelmgmt.EncodeFloat32([]float32{1, 2})}
	tuned := tensorEntry{DType: "F32", Shape: []int{2}, Raw: modelmgmt.EncodeFloat32([]float32{2, 4})}
	delta, err := compareTensorEntries("w", base, tuned)
	core.RequireNoError(t, err)
	core.AssertEqual(t, CompareStatusChanged, delta.Status)
	core.AssertInDelta(t, 1.5, delta.MeanAbsDelta, 1e-6)
}

func TestCompare_CompareTensorEntries_Bad(t *core.T) {
	base := tensorEntry{DType: "F32", Shape: []int{2}, Raw: modelmgmt.EncodeFloat32([]float32{1, 2})}
	tuned := tensorEntry{DType: "F32", Shape: []int{3}, Raw: modelmgmt.EncodeFloat32([]float32{1, 2, 3})}
	delta, err := compareTensorEntries("w", base, tuned)
	core.RequireNoError(t, err)
	core.AssertEqual(t, CompareStatusShapeMismatch, delta.Status)
}

func TestCompare_CompareTensorEntries_Ugly(t *core.T) {
	base := tensorEntry{DType: "F32", Shape: []int{2}, Raw: modelmgmt.EncodeFloat32([]float32{1, 2})}
	tuned := tensorEntry{DType: "BF16", Shape: []int{2}, Raw: []byte{0, 0, 0, 0}}
	delta, err := compareTensorEntries("w", base, tuned)
	core.RequireNoError(t, err)
	core.AssertEqual(t, CompareStatusDTypeMismatch, delta.Status)
}

func TestCompare_CompareCosine_Good(t *core.T) {
	core.AssertEqual(t, 1.0, compareCosine(0, 0, 0))
}

func TestCompare_CompareCosine_Bad(t *core.T) {
	core.AssertEqual(t, 0.0, compareCosine(0, 1, 0))
}

func TestCompare_CompareCosine_Ugly(t *core.T) {
	core.AssertInDelta(t, 0, compareCosine(0, 1, 1), 1e-12)
}

func TestCompare_CloneCompareLabels_Good(t *core.T) {
	got := cloneCompareLabels(map[string]string{"a": "b"})
	core.AssertEqual(t, map[string]string{"a": "b"}, got)
}

func TestCompare_CloneCompareLabels_Bad(t *core.T) {
	core.AssertNil(t, cloneCompareLabels(map[string]string{}))
}

func TestCompare_CloneCompareLabels_Ugly(t *core.T) {
	core.AssertNil(t, cloneCompareLabels(nil))
}

func TestCompare_CloneIntSlice_Good(t *core.T) {
	core.AssertEqual(t, []int{1, 2}, cloneIntSlice([]int{1, 2}))
}

func TestCompare_CloneIntSlice_Bad(t *core.T) {
	core.AssertNil(t, cloneIntSlice([]int{}))
}

func TestCompare_CloneIntSlice_Ugly(t *core.T) {
	core.AssertNil(t, cloneIntSlice(nil))
}
