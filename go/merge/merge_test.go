// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"

	core "dappco.re/go"
)

func TestMerge_Packs_Good(t *core.T) {
	a := writeSourceFixture(t, t.TempDir(), "test-arch", "shared-tokenizer", map[string][]float32{"w": {1, 2, 3, 4}})
	b := writeSourceFixture(t, t.TempDir(), "test-arch", "shared-tokenizer", map[string][]float32{"w": {3, 4, 5, 6}})
	outDir := core.PathJoin(t.TempDir(), "merged")

	// Pass a nil ctx to also cover the nil -> context.Background() default.
	result, err := Packs(nil, Options{
		Sources:    []Source{a, b},
		OutputPath: outDir,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, result.MergedTensors)
	core.AssertEqual(t, 0, result.CopiedTensors)
	core.AssertEmpty(t, result.SkippedTensors)
	core.AssertEqual(t, 1, result.TensorCount)
	core.AssertEqual(t, MethodLinear, result.Method)
	core.AssertTrue(t, coreFileExists(result.WeightPath))
	core.AssertTrue(t, coreFileExists(result.ProvenancePath))

	got := readMergedTensor(t, result.WeightPath, "w")
	core.AssertEqual(t, []float32{2, 3, 4, 5}, got)
}

func TestMerge_Packs_Bad(t *core.T) {
	a := writeSourceFixture(t, t.TempDir(), "test-arch", "tok", map[string][]float32{"w": {1, 2}})
	_, err := Packs(context.Background(), Options{
		Sources:    []Source{a},
		OutputPath: core.PathJoin(t.TempDir(), "merged"),
	})
	core.AssertErrorIs(t, err, errMergeNeedTwoSources)
}

func TestMerge_Packs_Ugly(t *core.T) {
	a := writeSourceFixture(t, t.TempDir(), "test-arch", "tok", map[string][]float32{"w": {1, 2}})
	b := writeSourceFixture(t, t.TempDir(), "test-arch", "tok", map[string][]float32{"w": {3, 4}})
	c := writeSourceFixture(t, t.TempDir(), "test-arch", "tok", map[string][]float32{"w": {5, 6}})
	_, err := Packs(context.Background(), Options{
		Sources:    []Source{a, b, c},
		Method:     MethodSLERP,
		OutputPath: core.PathJoin(t.TempDir(), "merged"),
	})
	core.AssertErrorIs(t, err, errSLERPNeedTwoSources)
}

func TestMerge_Packs_SLERP(t *core.T) {
	a := writeSourceFixture(t, t.TempDir(), "test-arch", "tok", map[string][]float32{"w": {1, 0}})
	b := writeSourceFixture(t, t.TempDir(), "test-arch", "tok", map[string][]float32{"w": {0, 1}})
	result, err := Packs(context.Background(), Options{
		Sources:    []Source{a, b},
		Method:     MethodSLERP,
		T:          0.5,
		OutputPath: core.PathJoin(t.TempDir(), "merged"),
	})
	core.RequireNoError(t, err)
	got := readMergedTensor(t, result.WeightPath, "w")
	core.AssertInDelta(t, 0.70710678, float64(got[0]), 1e-6)
	core.AssertInDelta(t, 0.70710678, float64(got[1]), 1e-6)
}

func TestMerge_Packs_WeightedSources(t *core.T) {
	a := writeSourceFixture(t, t.TempDir(), "test-arch", "tok", map[string][]float32{"w": {2, 4}})
	b := writeSourceFixture(t, t.TempDir(), "test-arch", "tok", map[string][]float32{"w": {10, 20}})
	a.Weight = 0.25
	b.Weight = 0.75
	result, err := Packs(context.Background(), Options{
		Sources:    []Source{a, b},
		OutputPath: core.PathJoin(t.TempDir(), "merged"),
	})
	core.RequireNoError(t, err)
	got := readMergedTensor(t, result.WeightPath, "w")
	core.AssertEqual(t, []float32{8, 16}, got)
}

func TestMerge_Packs_ArchitectureMismatch(t *core.T) {
	a := writeSourceFixture(t, t.TempDir(), "arch-a", "tok", map[string][]float32{"w": {1, 2}})
	b := writeSourceFixture(t, t.TempDir(), "arch-b", "tok", map[string][]float32{"w": {3, 4}})

	_, err := Packs(context.Background(), Options{
		Sources:    []Source{a, b},
		OutputPath: core.PathJoin(t.TempDir(), "merged"),
	})
	core.AssertError(t, err, "architecture mismatch")

	result, err := Packs(context.Background(), Options{
		Sources:                   []Source{a, b},
		OutputPath:                core.PathJoin(t.TempDir(), "merged-allowed"),
		AllowArchitectureMismatch: true,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, result.MergedTensors)
}

func TestMerge_Packs_TokenizerMismatch(t *core.T) {
	a := writeSourceFixture(t, t.TempDir(), "arch", "tokenizer-one", map[string][]float32{"w": {1, 2}})
	b := writeSourceFixture(t, t.TempDir(), "arch", "tokenizer-two", map[string][]float32{"w": {3, 4}})

	_, err := Packs(context.Background(), Options{
		Sources:    []Source{a, b},
		OutputPath: core.PathJoin(t.TempDir(), "merged"),
	})
	core.AssertErrorIs(t, err, errTokenizerMismatch)

	result, err := Packs(context.Background(), Options{
		Sources:                []Source{a, b},
		OutputPath:             core.PathJoin(t.TempDir(), "merged-allowed"),
		AllowTokenizerMismatch: true,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, result.MergedTensors)
}

func TestMerge_Packs_TensorMismatchDisallowed(t *core.T) {
	a := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {1, 2, 3, 4}, "only_in_base": {7, 8}})
	b := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {3, 4, 5, 6}})

	_, err := Packs(context.Background(), Options{
		Sources:    []Source{a, b},
		OutputPath: core.PathJoin(t.TempDir(), "merged"),
	})
	core.AssertError(t, err, "only_in_base")
}

func TestMerge_Packs_TensorMismatchAllowed(t *core.T) {
	a := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {1, 2, 3, 4}, "only_in_base": {7, 8}})
	b := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {3, 4, 5, 6}})

	result, err := Packs(context.Background(), Options{
		Sources:             []Source{a, b},
		OutputPath:          core.PathJoin(t.TempDir(), "merged"),
		AllowTensorMismatch: true,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, result.MergedTensors)
	core.AssertEqual(t, 1, result.CopiedTensors)
	core.AssertEqual(t, []string{"only_in_base"}, result.SkippedTensors)

	got := readMergedTensor(t, result.WeightPath, "only_in_base")
	core.AssertEqual(t, []float32{7, 8}, got)
}

func TestMerge_Packs_OutputSameAsSource(t *core.T) {
	root := t.TempDir()
	a := Source{Root: root, WeightFiles: []string{core.PathJoin(root, "model.safetensors")}}
	b := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {1, 2}})

	_, err := Packs(context.Background(), Options{
		Sources:    []Source{a, b},
		OutputPath: root,
	})
	core.AssertErrorIs(t, err, errOutputSameAsSource)
}

func TestMerge_Packs_OutputNotPackDir(t *core.T) {
	a := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {1, 2}})
	b := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {3, 4}})

	_, err := Packs(context.Background(), Options{
		Sources:    []Source{a, b},
		OutputPath: core.PathJoin(t.TempDir(), "model.safetensors"),
	})
	core.AssertErrorIs(t, err, errOutputNotPackDir)
}

func TestMerge_Packs_OutputHasWeights(t *core.T) {
	a := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {1, 2}})
	b := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {3, 4}})
	outDir := t.TempDir()
	requireResultOK(t, core.WriteFile(core.PathJoin(outDir, "existing.safetensors"), []byte("x"), 0o644))

	_, err := Packs(context.Background(), Options{
		Sources:    []Source{a, b},
		OutputPath: outDir,
	})
	core.AssertErrorIs(t, err, errOutputHasWeights)
}

func TestMerge_Packs_TOutOfRange(t *core.T) {
	a := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {1, 2}})
	b := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {3, 4}})

	_, err := Packs(context.Background(), Options{
		Sources:    []Source{a, b},
		T:          1.5,
		OutputPath: core.PathJoin(t.TempDir(), "merged"),
	})
	core.AssertErrorIs(t, err, errMergeTOutOfRange)
}

func TestMerge_Packs_ContextCancelled(t *core.T) {
	a := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {1, 2}})
	b := writeSourceFixture(t, t.TempDir(), "arch", "tok", map[string][]float32{"w": {3, 4}})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := Packs(ctx, Options{
		Sources:    []Source{a, b},
		OutputPath: core.PathJoin(t.TempDir(), "merged"),
	})
	core.AssertError(t, err)
}

func TestMerge_HasSuffixFold_Good(t *core.T) {
	core.AssertTrue(t, hasSuffixFold("Model.SAFETENSORS", ".safetensors"))
}

func TestMerge_HasSuffixFold_Bad(t *core.T) {
	core.AssertFalse(t, hasSuffixFold("model.bin", ".safetensors"))
}

func TestMerge_HasSuffixFold_Ugly(t *core.T) {
	core.AssertFalse(t, hasSuffixFold("st", ".safetensors"))
}

func TestMerge_SameIntSlice_Good(t *core.T) {
	core.AssertTrue(t, sameIntSlice([]int{1, 2, 3}, []int{1, 2, 3}))
}

func TestMerge_SameIntSlice_Bad(t *core.T) {
	core.AssertFalse(t, sameIntSlice([]int{1, 2}, []int{1, 2, 3}))
}

func TestMerge_SameIntSlice_Ugly(t *core.T) {
	core.AssertFalse(t, sameIntSlice([]int{1, 2, 3}, []int{1, 2, 4}))
}

func TestMerge_ClampFloat64_Good(t *core.T) {
	core.AssertEqual(t, 0.5, clampFloat64(0.5, -1, 1))
}

func TestMerge_ClampFloat64_Bad(t *core.T) {
	core.AssertEqual(t, 1.0, clampFloat64(5, -1, 1))
}

func TestMerge_ClampFloat64_Ugly(t *core.T) {
	core.AssertEqual(t, -1.0, clampFloat64(-5, -1, 1))
}

func TestMerge_ResultError_Good(t *core.T) {
	core.AssertNil(t, resultError(core.Ok("value")))
}

func TestMerge_ResultError_Bad(t *core.T) {
	sentinel := core.NewError("boom")
	core.AssertSame(t, sentinel, resultError(core.Fail(sentinel)))
}

func TestMerge_ResultError_Ugly(t *core.T) {
	core.AssertSame(t, errCoreResultFailed, resultError(core.Result{}))
}

func TestMerge_EqualFold_Good(t *core.T) {
	core.AssertTrue(t, equalFold("Adapter_Provenance.JSON", "adapter_provenance.json"))
}

func TestMerge_EqualFold_Bad(t *core.T) {
	core.AssertFalse(t, equalFold("config.json", "adapter_provenance.json"))
}

func TestMerge_EqualFold_Ugly(t *core.T) {
	core.AssertFalse(t, equalFold("short", "muchlonger"))
}

func TestMerge_ContainsFold_Good(t *core.T) {
	core.AssertTrue(t, containsFold("model.SAFETENSORS.index.json", ".safetensors"))
}

func TestMerge_ContainsFold_Bad(t *core.T) {
	core.AssertFalse(t, containsFold("config.json", ".safetensors"))
}

func TestMerge_ContainsFold_Ugly(t *core.T) {
	core.AssertTrue(t, containsFold("anything", ""))
}

// coreFileExists reports whether path names a regular, readable file — a
// thin core.Stat wrapper kept local to the test package.
func coreFileExists(path string) bool {
	stat := core.Stat(path)
	return stat.OK
}
