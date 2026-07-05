package inference

import (
	"slices"
	"testing"

	core "dappco.re/go"
)

// --- test helpers for discover ---

// createModelDir creates a fake model directory with config.json and n safetensors files.
//
//	createModelDir(t, core.JoinPath(base, "gemma3-1b"), map[string]any{"model_type": "gemma3"}, 1)
func createModelDir(t *testing.T, dir string, config map[string]any, numSafetensors int) {
	t.Helper()

	checkResultOK(t, core.MkdirAll(dir, 0o755))

	if config != nil {
		data := []byte(core.JSONMarshalString(config))
		checkResultOK(t, core.WriteFile(core.JoinPath(dir, "config.json"), data, 0o644))
	}

	for i := range numSafetensors {
		fname := core.JoinPath(dir, core.Sprintf("model-%05d-of-%05d.safetensors", i+1, numSafetensors))
		checkResultOK(t, core.WriteFile(fname, []byte("fake"), 0o644))
	}
}

// --- Discover ---

func TestDiscover_Discover_Good(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "gemma3-1b"), map[string]any{"model_type": "gemma3"}, 2)

	models := slices.Collect(Discover(base))
	core.AssertLen(t, models, 1)
	core.AssertEqual(t, "gemma3", models[0].ModelType)
	core.AssertEqual(t, 2, models[0].NumFiles)
}

func TestDiscover_Discover_Bad(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "empty-model"), map[string]any{"model_type": "gemma3"}, 0)

	models := slices.Collect(Discover(base))
	core.AssertEmpty(t, models)
	core.AssertEqual(t, 0, len(models))
}

func TestDiscover_Discover_Ugly(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "no-type"), map[string]any{"vocab_size": 32000}, 1)

	models := slices.Collect(Discover(base))
	core.AssertLen(t, models, 1)
	core.AssertEqual(t, "", models[0].ModelType)
	core.AssertEqual(t, 0, models[0].QuantBits)
}

func TestDiscover_Good_SingleModel(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "gemma3-1b"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	models := slices.Collect(Discover(base))
	checkLen(t, models, 1)

	discovered := models[0]
	checkEqual(t, "gemma3", discovered.ModelType)
	checkEqual(t, 1, discovered.NumFiles)
	checkEqual(t, 0, discovered.QuantBits)
	checkEqual(t, 0, discovered.QuantGroup)
	checkTrue(t, core.PathIsAbs(discovered.Path))
	checkContains(t, discovered.Path, "gemma3-1b")
}

func TestDiscover_Good_MultipleModels(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "gemma3-1b"), map[string]any{
		"model_type": "gemma3",
	}, 1)
	createModelDir(t, core.JoinPath(base, "qwen3-4b"), map[string]any{
		"model_type": "qwen3",
	}, 4)

	models := slices.Collect(Discover(base))
	checkLen(t, models, 2)

	types := make([]string, len(models))
	for i, m := range models {
		types[i] = m.ModelType
	}
	checkElementsMatch(t, []string{"gemma3", "qwen3"}, types)
}

func TestDiscover_Good_RecursiveNestedModels(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "outer"), map[string]any{
		"model_type": "outer",
	}, 1)
	createModelDir(t, core.JoinPath(base, "outer", "inner"), map[string]any{
		"model_type": "inner",
	}, 1)

	models := slices.Collect(Discover(base))
	checkLen(t, models, 2)
	checkEqual(t, "outer", models[0].ModelType)
	checkEqual(t, "inner", models[1].ModelType)
}

func TestDiscover_Good_Quantised(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "gemma3-1b-4bit"), map[string]any{
		"model_type": "gemma3",
		"quantization": map[string]any{
			"bits":       4,
			"group_size": 64,
		},
	}, 1)

	models := slices.Collect(Discover(base))
	checkLen(t, models, 1)

	discovered := models[0]
	checkEqual(t, 4, discovered.QuantBits)
	checkEqual(t, 64, discovered.QuantGroup)
}

func TestDiscover_Good_BaseDirIsModel(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, base, map[string]any{
		"model_type": "llama",
	}, 2)

	models := slices.Collect(Discover(base))
	checkLen(t, models, 1)

	discovered := models[0]
	checkEqual(t, "llama", discovered.ModelType)
	checkEqual(t, 2, discovered.NumFiles)
}

func TestDiscover_Good_BaseDirPlusSubdir(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, base, map[string]any{
		"model_type": "parent_model",
	}, 1)
	createModelDir(t, core.JoinPath(base, "child"), map[string]any{
		"model_type": "child_model",
	}, 1)

	models := slices.Collect(Discover(base))
	checkLen(t, models, 2)

	checkEqual(t, "parent_model", models[0].ModelType)
	checkEqual(t, "child_model", models[1].ModelType)
}

func TestDiscover_Good_EmptyDir(t *testing.T) {
	base := t.TempDir()

	models := slices.Collect(Discover(base))
	checkEmpty(t, models)
}

func TestDiscover_Bad_NonexistentDir(t *testing.T) {
	models := slices.Collect(Discover("/nonexistent/path/that/should/not/exist"))
	checkEmpty(t, models)
	checkLen(t, models, 0)
	checkNil(t, models)
}

func TestDiscover_Bad_NoSafetensors(t *testing.T) {
	base := t.TempDir()
	dir := core.JoinPath(base, "incomplete")

	checkResultOK(t, core.MkdirAll(dir, 0o755))
	config := map[string]any{"model_type": "gemma3"}
	data := []byte(core.JSONMarshalString(config))
	checkResultOK(t, core.WriteFile(core.JoinPath(dir, "config.json"), data, 0o644))

	models := slices.Collect(Discover(base))
	checkEmpty(t, models)
}

func TestDiscover_Bad_NoConfig(t *testing.T) {
	base := t.TempDir()
	dir := core.JoinPath(base, "no-config")

	checkResultOK(t, core.MkdirAll(dir, 0o755))
	checkResultOK(t, core.WriteFile(core.JoinPath(dir, "model.safetensors"), []byte("fake"), 0o644))

	models := slices.Collect(Discover(base))
	checkEmpty(t, models)
}

func TestDiscover_Bad_InvalidJSON(t *testing.T) {
	base := t.TempDir()
	dir := core.JoinPath(base, "bad-json")

	checkResultOK(t, core.MkdirAll(dir, 0o755))
	checkResultOK(t, core.WriteFile(core.JoinPath(dir, "config.json"), []byte("{invalid}"), 0o644))
	checkResultOK(t, core.WriteFile(core.JoinPath(dir, "model.safetensors"), []byte("fake"), 0o644))

	models := slices.Collect(Discover(base))
	checkLen(t, models, 1)
	checkEmpty(t, models[0].ModelType)
	checkEqual(t, 1, models[0].NumFiles)
	checkEqual(t, 0, models[0].QuantBits)
	checkEqual(t, 0, models[0].QuantGroup)
}

func TestDiscover_Ugly_SkipsRegularFiles(t *testing.T) {
	base := t.TempDir()
	checkResultOK(t, core.WriteFile(core.JoinPath(base, "README.md"), []byte("hello"), 0o644))

	createModelDir(t, core.JoinPath(base, "real-model"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	models := slices.Collect(Discover(base))
	checkLen(t, models, 1)
	checkEqual(t, "gemma3", models[0].ModelType)
}

func TestDiscover_Ugly_MissingModelType(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "no-type"), map[string]any{
		"vocab_size": 32000,
	}, 1)

	models := slices.Collect(Discover(base))
	checkLen(t, models, 1)
	checkEqual(t, "", models[0].ModelType)
}

func TestDiscover_Ugly_NoQuantisation(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "fp16"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	models := slices.Collect(Discover(base))
	checkLen(t, models, 1)
	checkEqual(t, 0, models[0].QuantBits)
	checkEqual(t, 0, models[0].QuantGroup)
}

func TestDiscover_Good_MultipleSafetensors(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "large-model"), map[string]any{
		"model_type": "llama",
	}, 8)

	models := slices.Collect(Discover(base))
	checkLen(t, models, 1)
	checkEqual(t, 8, models[0].NumFiles)
}

func TestDiscover_Good_EarlyBreakOnBaseDir(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, base, map[string]any{
		"model_type": "parent",
	}, 1)
	createModelDir(t, core.JoinPath(base, "child"), map[string]any{
		"model_type": "child",
	}, 1)

	count := 0
	for range Discover(base) {
		count++
		break
	}
	checkEqual(t, 1, count)
}

func TestDiscover_Good_EarlyBreakOnSubdir(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "model-a"), map[string]any{
		"model_type": "a",
	}, 1)
	createModelDir(t, core.JoinPath(base, "model-b"), map[string]any{
		"model_type": "b",
	}, 1)

	count := 0
	for range Discover(base) {
		count++
		break
	}
	checkEqual(t, 1, count)
}

func TestDiscover_Good_AbsolutePath(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "test-model"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	models := slices.Collect(Discover(base))
	checkLen(t, models, 1)
	checkTrue(t, core.PathIsAbs(models[0].Path))
}

func TestDiscover_Good_RelativeBaseDir(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "relative-model"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	cwdResult := core.Getwd()
	checkResultOK(t, cwdResult)
	cwd := cwdResult.Value.(string)

	relBaseResult := core.PathRel(cwd, base)
	checkResultOK(t, relBaseResult)
	relBase := relBaseResult.Value.(string)

	models := slices.Collect(Discover(relBase))
	checkLen(t, models, 1)
	checkTrue(t, core.PathIsAbs(models[0].Path))
	checkEqual(t, "gemma3", models[0].ModelType)
}

func TestDiscover_Good_QuantizationConfigFallback(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.JoinPath(base, "qwen3-4b"), map[string]any{
		"model_type": "qwen3",
		"quantization_config": map[string]any{
			"bits":       8,
			"group_size": 128,
		},
	}, 1)

	models := slices.Collect(Discover(base))
	checkLen(t, models, 1)
	checkEqual(t, 8, models[0].QuantBits)
	checkEqual(t, 128, models[0].QuantGroup)
}

func TestDiscover_Good_RecursiveDeepModels(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.Path(base, "models"), map[string]any{
		"model_type": "parent",
	}, 1)
	createModelDir(t, core.Path(base, "models", "qwen3"), map[string]any{
		"model_type": "qwen3",
	}, 1)
	createModelDir(t, core.Path(base, "models", "qwen3", "fine-tuned"), map[string]any{
		"model_type": "qwen3",
	}, 1)
	createModelDir(t, core.Path(base, "other"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	models := slices.Collect(Discover(base))
	checkLen(t, models, 4)

	gotParent := false
	gotNested := false
	for _, m := range models {
		if m.Path == core.Path(base, "models") {
			gotParent = true
		}
		if m.Path == core.Path(base, "models", "qwen3", "fine-tuned") {
			gotNested = true
		}
	}
	checkTrue(t, gotParent)
	checkTrue(t, gotNested)
}

func TestDiscover_Good_RecursiveEarlyBreak(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.Path(base, "a", "b", "deep"), map[string]any{
		"model_type": "deep",
	}, 1)
	createModelDir(t, core.Path(base, "a", "c"), map[string]any{
		"model_type": "second",
	}, 1)

	count := 0
	for range Discover(base) {
		count++
		break
	}
	checkEqual(t, 1, count)
}

// AX-11: alloc budget locked at the measured baseline. Failing
// this test means a recent change increased the per-call alloc
// count above the documented ceiling — surface for review BEFORE
// the regression hits a downstream backend (every driver that
// imports go-inference for Discover pays this per app boot).
//
// Baselines (Apple M3 Ultra, -benchmem, 10 junk dirs):
//
//	alpha.95 (per-call core.New): 254 allocs / 26616 B
//	sync.Once cached Core:        208 allocs / 24064 B  ← current
//
// The ceiling is set with deliberate headroom — small drift from
// stdlib internals across Go releases is acceptable; a fix that
// drops the alloc count ratchets this number DOWN, not up.
//
// Run a fresh Discover under testing.AllocsPerRun (which forces
// a GC + measures averaged-per-call allocs). The harness already
// produces N=10 dirs identical to BenchmarkDiscover_NoModels_TenJunkDirs
// so the bench output and this gate stay aligned.
func TestDiscover_AllocBudget_NoModels_TenJunkDirs(t *testing.T) {
	base := t.TempDir()
	for i := 0; i < 10; i++ {
		dir := core.Path(base, core.Sprintf("junk-%d", i))
		checkResultOK(t, core.MkdirAll(dir, 0o755))
		checkResultOK(t, core.WriteFile(core.Path(dir, "README.md"), []byte("not a model"), 0o644))
	}

	// AllocsPerRun does an untimed warm-up call then averages over
	// runs — first call's lazy-init noise is excluded. 5 runs is
	// enough to stabilise without making the test slow.
	avg := testing.AllocsPerRun(5, func() {
		for range Discover(base) {
			// drain
		}
	})

	// Ceiling: 215 — current measured (208) plus ~3% headroom for
	// stdlib drift. Was 254→260 pre-sync.Once-Core. Ratchet DOWN
	// when optimisations land; never up without a documented
	// reason in the commit that bumps this.
	const budget = 215.0
	if avg > budget {
		t.Fatalf("Discover alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"This usually means a recent change added a per-call allocation "+
			"that propagates to every consumer (go-mlx, go-rocm, go-cuda).\n"+
			"Profile with: go test -bench=BenchmarkDiscover_NoModels_TenJunkDirs "+
			"-benchmem -memprofile=/tmp/disc.mem && go tool pprof -alloc_objects /tmp/disc.mem",
			avg, budget)
	}
}
