// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"

	core "dappco.re/go"

	"dappco.re/go/inference/model/safetensors"
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

// TestCompare_ComparePacks_Sharded proves ComparePacks works across a genuinely multi-shard
// checkpoint (each source split across 2 safetensors files, the model.safetensors.index.json
// + N-shard HF layout) — the sharded-checkpoint gap tracker #34 flagged as unimplemented.
func TestCompare_ComparePacks_Sharded(t *core.T) {
	shardNames := []string{"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
	base := writeSourceFixtureSharded(t, t.TempDir(), "arch", "tok", map[string][]float32{
		"changed":   {1, 2, 3, 4},
		"unchanged": {5, 6},
	}, shardNames)
	tuned := writeSourceFixtureSharded(t, t.TempDir(), "arch", "tok", map[string][]float32{
		"changed":   {2, 2, 3, 8},
		"unchanged": {5, 6},
	}, shardNames)
	core.AssertLen(t, base.WeightFiles, 2)
	core.AssertLen(t, tuned.WeightFiles, 2)

	result, err := ComparePacks(context.Background(), CompareOptions{Base: base, FineTuned: tuned})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, result.ChangedTensors)
	core.AssertEqual(t, 1, result.UnchangedTensors)
	core.AssertInDelta(t, 1.25, result.Tensors[0].MeanAbsDelta, 1e-6)
}

func TestCompare_ComparePacks_ContextCancelled(t *core.T) {
	base := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {1}})
	tuned := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {2}})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := ComparePacks(ctx, CompareOptions{Base: base, FineTuned: tuned})
	core.AssertError(t, err)
}

// TestCompare_ValidateComparePack_Good documents that a source carrying
// both a root and at least one weight file passes validation regardless of
// which label ("base" or "fine-tuned") it is checked under.
func TestCompare_ValidateComparePack_Good(t *core.T) {
	base := Source{Root: "/x", WeightFiles: []string{"/x/model.safetensors"}}
	core.AssertNoError(t, validateComparePack("base", base))
	tuned := Source{Root: "/y", WeightFiles: []string{"/y/a.safetensors", "/y/b.safetensors"}}
	core.AssertNoError(t, validateComparePack("fine-tuned", tuned))
}

func TestCompare_ValidateComparePack_Bad(t *core.T) {
	err := validateComparePack("base", Source{})
	core.AssertError(t, err, "root is required")
	core.AssertError(t, err, "base")
}

func TestCompare_ValidateComparePack_Ugly(t *core.T) {
	err := validateComparePack("fine-tuned", Source{Root: "/x"})
	core.AssertError(t, err, "requires weight files")
	core.AssertError(t, err, "fine-tuned")
}

// compareEntryPair writes base/tuned as two independent single-tensor "w"
// shard files (t.TempDir) and indexes each through indexWeightFiles,
// returning the two real, file-backed tensorEntry values compareTensorEntries
// now requires (it reads each Ref's payload from disk, not an in-memory
// Raw buffer).
func compareEntryPair(t *core.T, base, tuned safetensors.SafetensorsTensorInfo, baseData, tunedData []byte) (tensorEntry, tensorEntry) {
	t.Helper()
	dir := t.TempDir()
	basePath := core.PathJoin(dir, "base.safetensors")
	tunedPath := core.PathJoin(dir, "tuned.safetensors")
	requireResultOK(t, safetensors.WriteSafetensors(basePath, map[string]safetensors.SafetensorsTensorInfo{"w": base}, map[string][]byte{"w": baseData}))
	requireResultOK(t, safetensors.WriteSafetensors(tunedPath, map[string]safetensors.SafetensorsTensorInfo{"w": tuned}, map[string][]byte{"w": tunedData}))
	baseIdx, err := indexWeightFiles([]string{basePath})
	core.RequireNoError(t, err)
	tunedIdx, err := indexWeightFiles([]string{tunedPath})
	core.RequireNoError(t, err)
	return baseIdx.Tensors["w"], tunedIdx.Tensors["w"]
}

func TestCompare_CompareTensorEntries_Good(t *core.T) {
	base, tuned := compareEntryPair(t,
		safetensors.SafetensorsTensorInfo{Dtype: "F32", Shape: []int{2}}, safetensors.SafetensorsTensorInfo{Dtype: "F32", Shape: []int{2}},
		safetensors.EncodeFloat32([]float32{1, 2}), safetensors.EncodeFloat32([]float32{2, 4}))
	cache := safetensors.NewShardCache()
	defer cache.Close()
	delta, err := compareTensorEntries(cache, "w", base, tuned)
	core.RequireNoError(t, err)
	core.AssertEqual(t, CompareStatusChanged, delta.Status)
	core.AssertInDelta(t, 1.5, delta.MeanAbsDelta, 1e-6)
}

func TestCompare_CompareTensorEntries_Bad(t *core.T) {
	base, tuned := compareEntryPair(t,
		safetensors.SafetensorsTensorInfo{Dtype: "F32", Shape: []int{2}}, safetensors.SafetensorsTensorInfo{Dtype: "F32", Shape: []int{3}},
		safetensors.EncodeFloat32([]float32{1, 2}), safetensors.EncodeFloat32([]float32{1, 2, 3}))
	cache := safetensors.NewShardCache()
	defer cache.Close()
	delta, err := compareTensorEntries(cache, "w", base, tuned)
	core.RequireNoError(t, err)
	core.AssertEqual(t, CompareStatusShapeMismatch, delta.Status)
}

func TestCompare_CompareTensorEntries_Ugly(t *core.T) {
	base, tuned := compareEntryPair(t,
		safetensors.SafetensorsTensorInfo{Dtype: "F32", Shape: []int{2}}, safetensors.SafetensorsTensorInfo{Dtype: "BF16", Shape: []int{2}},
		safetensors.EncodeFloat32([]float32{1, 2}), []byte{0, 0, 0, 0})
	cache := safetensors.NewShardCache()
	defer cache.Close()
	delta, err := compareTensorEntries(cache, "w", base, tuned)
	core.RequireNoError(t, err)
	core.AssertEqual(t, CompareStatusDTypeMismatch, delta.Status)
}

// TestCompare_CompareCosine_Good documents cosine similarity for a
// genuinely aligned pair (dot == both norms means the vectors are
// identical) plus the both-zero convention, which is defined as identical.
func TestCompare_CompareCosine_Good(t *core.T) {
	core.AssertInDelta(t, 1.0, compareCosine(4, 4, 4), 1e-12)
	core.AssertInDelta(t, 1.0, compareCosine(8, 4, 16), 1e-12) // scale-invariant: same direction, different magnitude
	core.AssertEqual(t, 1.0, compareCosine(0, 0, 0))
}

// TestCompare_CompareCosine_Bad exercises both sides of the single-zero-norm
// branch: similarity against a zero-magnitude vector is undefined, so 0 is
// returned whichever operand carries the zero norm, regardless of dot sign.
func TestCompare_CompareCosine_Bad(t *core.T) {
	core.AssertEqual(t, 0.0, compareCosine(0, 1, 0))
	core.AssertEqual(t, 0.0, compareCosine(5, 0, 9))
	core.AssertEqual(t, 0.0, compareCosine(-5, 0, 9))
}

// TestCompare_CompareCosine_Ugly covers orthogonal vectors (dot == 0, both
// norms non-zero), a genuine mid-range negative cosine needing no clamp,
// and clamping of an out-of-[-1,1] cosine caused by floating-point drift.
func TestCompare_CompareCosine_Ugly(t *core.T) {
	core.AssertInDelta(t, 0, compareCosine(0, 1, 1), 1e-12)
	core.AssertInDelta(t, -0.5, compareCosine(-2, 4, 4), 1e-12)
	core.AssertEqual(t, -1.0, compareCosine(-100, 1, 1))
}

// TestCompare_CloneCompareLabels_Good documents Clone's independence
// guarantee: mutating the source map after cloning must not leak into the
// returned copy.
func TestCompare_CloneCompareLabels_Good(t *core.T) {
	src := map[string]string{"a": "b"}
	got := cloneCompareLabels(src)
	core.AssertEqual(t, map[string]string{"a": "b"}, got)
	src["a"] = "mutated"
	core.AssertEqual(t, "b", got["a"])
}

// TestCompare_CloneCompareLabels_Bad documents that an empty-but-non-nil
// map normalises to nil rather than an allocated empty map — saving a
// downstream JSON `"labels":{}` versus an omitted field.
func TestCompare_CloneCompareLabels_Bad(t *core.T) {
	src := map[string]string{}
	got := cloneCompareLabels(src)
	core.AssertNil(t, got)
	core.AssertLen(t, src, 0)
}

func TestCompare_CloneCompareLabels_Ugly(t *core.T) {
	got := cloneCompareLabels(nil)
	core.AssertNil(t, got)
	core.AssertEqual(t, map[string]string(nil), got)
}

// TestCompare_CloneIntSlice_Good documents Clone's independence guarantee:
// mutating the source slice after cloning must not leak into the copy.
func TestCompare_CloneIntSlice_Good(t *core.T) {
	src := []int{1, 2}
	got := cloneIntSlice(src)
	core.AssertEqual(t, []int{1, 2}, got)
	src[0] = 99
	core.AssertEqual(t, 1, got[0])
}

// TestCompare_CloneIntSlice_Bad documents that an empty-but-non-nil slice
// normalises to nil, matching cloneCompareLabels' convention.
func TestCompare_CloneIntSlice_Bad(t *core.T) {
	src := []int{}
	got := cloneIntSlice(src)
	core.AssertNil(t, got)
	core.AssertLen(t, src, 0)
}

func TestCompare_CloneIntSlice_Ugly(t *core.T) {
	got := cloneIntSlice(nil)
	core.AssertNil(t, got)
	core.AssertEqual(t, []int(nil), got)
}
