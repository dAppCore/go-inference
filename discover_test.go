package inference

import (
	"os"
	"slices"
	"testing"

	"dappco.re/go/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- test helpers for discover ---

// createModelDir creates a fake model directory with config.json and n safetensors files.
func createModelDir(t *testing.T, dir string, config map[string]any, numSafetensors int) {
	t.Helper()
	require.NoError(t, os.MkdirAll(dir, 0o755))

	if config != nil {
		r := core.JSONMarshal(config)
		require.True(t, r.OK)
		require.NoError(t, os.WriteFile(core.Path(dir, "config.json"), r.Value.([]byte), 0o644))
	}

	for i := range numSafetensors {
		fname := core.Path(dir, core.Sprintf("model-%05d-of-%05d.safetensors", i+1, numSafetensors))
		require.NoError(t, os.WriteFile(fname, []byte("fake"), 0o644))
	}
}

// --- Discover ---

func TestDiscover_Good_SingleModel(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.Path(base, "gemma3-1b"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)

	m := models[0]
	assert.Equal(t, "gemma3", m.ModelType)
	assert.Equal(t, 1, m.NumFiles)
	assert.Equal(t, 0, m.QuantBits)
	assert.Equal(t, 0, m.QuantGroup)
	assert.True(t, core.PathIsAbs(m.Path), "path should be absolute")
	assert.Contains(t, m.Path, "gemma3-1b")
}

func TestDiscover_Good_MultipleModels(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.Path(base, "gemma3-1b"), map[string]any{
		"model_type": "gemma3",
	}, 1)
	createModelDir(t, core.Path(base, "qwen3-4b"), map[string]any{
		"model_type": "qwen3",
	}, 4)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 2)

	// Collect model types for assertion (order may vary due to ReadDir).
	types := make([]string, len(models))
	for i, m := range models {
		types[i] = m.ModelType
	}
	assert.ElementsMatch(t, []string{"gemma3", "qwen3"}, types)
}

func TestDiscover_Good_Quantised(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.Path(base, "gemma3-1b-4bit"), map[string]any{
		"model_type": "gemma3",
		"quantization": map[string]any{
			"bits":       4,
			"group_size": 64,
		},
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)

	m := models[0]
	assert.Equal(t, 4, m.QuantBits)
	assert.Equal(t, 64, m.QuantGroup)
}

func TestDiscover_Good_BaseDirIsModel(t *testing.T) {
	// The base directory itself is a model directory.
	base := t.TempDir()
	createModelDir(t, base, map[string]any{
		"model_type": "llama",
	}, 2)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)

	m := models[0]
	assert.Equal(t, "llama", m.ModelType)
	assert.Equal(t, 2, m.NumFiles)
}

func TestDiscover_Good_BaseDirPlusSubdir(t *testing.T) {
	// Both base dir and a subdir are valid models. Base should appear first.
	base := t.TempDir()
	createModelDir(t, base, map[string]any{
		"model_type": "parent_model",
	}, 1)
	createModelDir(t, core.Path(base, "child"), map[string]any{
		"model_type": "child_model",
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 2)

	// Base dir should appear first (prepended).
	assert.Equal(t, "parent_model", models[0].ModelType, "base dir model should be first")
	assert.Equal(t, "child_model", models[1].ModelType)
}

func TestDiscover_Good_EmptyDir(t *testing.T) {
	base := t.TempDir()

	models := slices.Collect(Discover(base))
	assert.Empty(t, models)
}

func TestDiscover_Bad_NonexistentDir(t *testing.T) {
	models := slices.Collect(Discover("/nonexistent/path/that/should/not/exist"))
	assert.Empty(t, models)
}

func TestDiscover_Bad_NoSafetensors(t *testing.T) {
	// Directory with config.json but no .safetensors files.
	base := t.TempDir()
	dir := core.Path(base, "incomplete")
	require.NoError(t, os.MkdirAll(dir, 0o755))

	config := map[string]any{"model_type": "gemma3"}
	r := core.JSONMarshal(config)
	require.True(t, r.OK)
	require.NoError(t, os.WriteFile(core.Path(dir, "config.json"), r.Value.([]byte), 0o644))

	models := slices.Collect(Discover(base))
	assert.Empty(t, models, "directory without safetensors should be skipped")
}

func TestDiscover_Bad_NoConfig(t *testing.T) {
	// Directory with .safetensors but no config.json.
	base := t.TempDir()
	dir := core.Path(base, "no-config")
	require.NoError(t, os.MkdirAll(dir, 0o755))
	require.NoError(t, os.WriteFile(core.Path(dir, "model.safetensors"), []byte("fake"), 0o644))

	models := slices.Collect(Discover(base))
	assert.Empty(t, models, "directory without config.json should be skipped")
}

func TestDiscover_Bad_InvalidJSON(t *testing.T) {
	// config.json exists but contains invalid JSON. Should NOT count as a model anymore.
	base := t.TempDir()
	dir := core.Path(base, "bad-json")
	require.NoError(t, os.MkdirAll(dir, 0o755))
	require.NoError(t, os.WriteFile(core.Path(dir, "config.json"), []byte("{invalid}"), 0o644))
	require.NoError(t, os.WriteFile(core.Path(dir, "model.safetensors"), []byte("fake"), 0o644))

	models := slices.Collect(Discover(base))
	require.Len(t, models, 0)
}

func TestDiscover_Ugly_SkipsRegularFiles(t *testing.T) {
	// Regular files in base dir should be silently skipped.
	base := t.TempDir()
	require.NoError(t, os.WriteFile(core.Path(base, "README.md"), []byte("hello"), 0o644))

	createModelDir(t, core.Path(base, "real-model"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)
	assert.Equal(t, "gemma3", models[0].ModelType)
}

func TestDiscover_Ugly_MissingModelType(t *testing.T) {
	// config.json without model_type field.
	base := t.TempDir()
	createModelDir(t, core.Path(base, "no-type"), map[string]any{
		"vocab_size": 32000,
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)
	assert.Equal(t, "", models[0].ModelType, "missing model_type should yield empty string")
}

func TestDiscover_Ugly_NoQuantisation(t *testing.T) {
	// config.json without quantization key — QuantBits/QuantGroup should be 0.
	base := t.TempDir()
	createModelDir(t, core.Path(base, "fp16"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)
	assert.Equal(t, 0, models[0].QuantBits)
	assert.Equal(t, 0, models[0].QuantGroup)
}

func TestDiscover_Good_MultipleSafetensors(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.Path(base, "large-model"), map[string]any{
		"model_type": "llama",
	}, 8)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)
	assert.Equal(t, 8, models[0].NumFiles)
}

func TestDiscover_Good_EarlyBreakOnBaseDir(t *testing.T) {
	// Base dir is a model and a subdir is also a model.
	// Breaking after the first yield should stop iteration.
	base := t.TempDir()
	createModelDir(t, base, map[string]any{
		"model_type": "parent",
	}, 1)
	createModelDir(t, core.Path(base, "child"), map[string]any{
		"model_type": "child",
	}, 1)

	count := 0
	for range Discover(base) {
		count++
		break // stop after first model
	}
	assert.Equal(t, 1, count, "iterator should stop after first yield when break is called")
}

func TestDiscover_Good_EarlyBreakOnSubdir(t *testing.T) {
	// Base dir is NOT a model; two subdirs are models.
	// Breaking after the first subdir yield should stop iteration.
	base := t.TempDir()
	createModelDir(t, core.Path(base, "model-a"), map[string]any{
		"model_type": "a",
	}, 1)
	createModelDir(t, core.Path(base, "model-b"), map[string]any{
		"model_type": "b",
	}, 1)

	count := 0
	for range Discover(base) {
		count++
		break
	}
	assert.Equal(t, 1, count, "iterator should stop after first subdir yield when break is called")
}

func TestDiscover_Good_AbsolutePath(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, core.Path(base, "test-model"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)
	assert.True(t, core.PathIsAbs(models[0].Path), "discovered path must be absolute")
}
