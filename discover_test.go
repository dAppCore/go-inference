package inference

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- test helpers for discover ---

// createModelDir creates a fake model directory with config.json and n safetensors files.
//
//	createModelDir(t, filepath.Join(base, "gemma3-1b"), map[string]any{"model_type": "gemma3"}, 1)
func createModelDir(t *testing.T, dir string, config map[string]any, numSafetensors int) {
	t.Helper()

	require.NoError(t, os.MkdirAll(dir, 0o755))

	if config != nil {
		data, err := json.Marshal(config)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(dir, "config.json"), data, 0o644))
	}

	for i := range numSafetensors {
		fname := filepath.Join(dir, fmt.Sprintf("model-%05d-of-%05d.safetensors", i+1, numSafetensors))
		require.NoError(t, os.WriteFile(fname, []byte("fake"), 0o644))
	}
}

// --- Discover ---

func TestDiscover_Good_SingleModel(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, filepath.Join(base, "gemma3-1b"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)

	discovered := models[0]
	assert.Equal(t, "gemma3", discovered.ModelType)
	assert.Equal(t, 1, discovered.NumFiles)
	assert.Equal(t, 0, discovered.QuantBits)
	assert.Equal(t, 0, discovered.QuantGroup)
	assert.True(t, filepath.IsAbs(discovered.Path), "path should be absolute")
	assert.Contains(t, discovered.Path, "gemma3-1b")
}

func TestDiscover_Good_MultipleModels(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, filepath.Join(base, "gemma3-1b"), map[string]any{
		"model_type": "gemma3",
	}, 1)
	createModelDir(t, filepath.Join(base, "qwen3-4b"), map[string]any{
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

func TestDiscover_Good_RecursiveNestedModels(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, filepath.Join(base, "outer"), map[string]any{
		"model_type": "outer",
	}, 1)
	createModelDir(t, filepath.Join(base, "outer", "inner"), map[string]any{
		"model_type": "inner",
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 2)
	assert.Equal(t, "outer", models[0].ModelType)
	assert.Equal(t, "inner", models[1].ModelType)
}

func TestDiscover_Good_Quantised(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, filepath.Join(base, "gemma3-1b-4bit"), map[string]any{
		"model_type": "gemma3",
		"quantization": map[string]any{
			"bits":       4,
			"group_size": 64,
		},
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)

	discovered := models[0]
	assert.Equal(t, 4, discovered.QuantBits)
	assert.Equal(t, 64, discovered.QuantGroup)
}

func TestDiscover_Good_BaseDirIsModel(t *testing.T) {
	// The base directory itself is a model directory.
	base := t.TempDir()
	createModelDir(t, base, map[string]any{
		"model_type": "llama",
	}, 2)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)

	discovered := models[0]
	assert.Equal(t, "llama", discovered.ModelType)
	assert.Equal(t, 2, discovered.NumFiles)
}

func TestDiscover_Good_BaseDirPlusSubdir(t *testing.T) {
	// Both base dir and a subdir are valid models. Base should appear first.
	base := t.TempDir()
	createModelDir(t, base, map[string]any{
		"model_type": "parent_model",
	}, 1)
	createModelDir(t, filepath.Join(base, "child"), map[string]any{
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
	dir := filepath.Join(base, "incomplete")

	require.NoError(t, os.MkdirAll(dir, 0o755))
	config := map[string]any{"model_type": "gemma3"}
	data, err := json.Marshal(config)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(dir, "config.json"), data, 0o644))

	models := slices.Collect(Discover(base))
	assert.Empty(t, models, "directory without safetensors should be skipped")
}

func TestDiscover_Bad_NoConfig(t *testing.T) {
	// Directory with .safetensors but no config.json.
	base := t.TempDir()
	dir := filepath.Join(base, "no-config")

	require.NoError(t, os.MkdirAll(dir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("fake"), 0o644))

	models := slices.Collect(Discover(base))
	assert.Empty(t, models, "directory without config.json should be skipped")
}

func TestDiscover_Bad_InvalidJSON(t *testing.T) {
	// config.json exists but contains invalid JSON. The directory is still
	// discovered, but the parsed metadata fields remain at their zero values.
	base := t.TempDir()
	dir := filepath.Join(base, "bad-json")

	require.NoError(t, os.MkdirAll(dir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "config.json"), []byte("{invalid}"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("fake"), 0o644))

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)
	assert.Empty(t, models[0].ModelType)
	assert.Equal(t, 1, models[0].NumFiles)
	assert.Equal(t, 0, models[0].QuantBits)
	assert.Equal(t, 0, models[0].QuantGroup)
}

func TestDiscover_Ugly_SkipsRegularFiles(t *testing.T) {
	// Regular files in base dir should be silently skipped.
	base := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(base, "README.md"), []byte("hello"), 0o644))

	createModelDir(t, filepath.Join(base, "real-model"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)
	assert.Equal(t, "gemma3", models[0].ModelType)
}

func TestDiscover_Ugly_MissingModelType(t *testing.T) {
	// config.json without model_type field.
	base := t.TempDir()
	createModelDir(t, filepath.Join(base, "no-type"), map[string]any{
		"vocab_size": 32000,
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)
	assert.Equal(t, "", models[0].ModelType, "missing model_type should yield empty string")
}

func TestDiscover_Ugly_NoQuantisation(t *testing.T) {
	// config.json without quantization key — QuantBits/QuantGroup should be 0.
	base := t.TempDir()
	createModelDir(t, filepath.Join(base, "fp16"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)
	assert.Equal(t, 0, models[0].QuantBits)
	assert.Equal(t, 0, models[0].QuantGroup)
}

func TestDiscover_Good_MultipleSafetensors(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, filepath.Join(base, "large-model"), map[string]any{
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
	createModelDir(t, filepath.Join(base, "child"), map[string]any{
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
	createModelDir(t, filepath.Join(base, "model-a"), map[string]any{
		"model_type": "a",
	}, 1)
	createModelDir(t, filepath.Join(base, "model-b"), map[string]any{
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
	createModelDir(t, filepath.Join(base, "test-model"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)
	assert.True(t, filepath.IsAbs(models[0].Path), "discovered path must be absolute")
}

func TestDiscover_Good_RelativeBaseDir(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, filepath.Join(base, "relative-model"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	cwd, err := os.Getwd()
	require.NoError(t, err)

	relBase, err := filepath.Rel(cwd, base)
	require.NoError(t, err)

	models := slices.Collect(Discover(relBase))
	require.Len(t, models, 1)
	assert.True(t, filepath.IsAbs(models[0].Path), "discovered path must be absolute even for relative input")
	assert.Equal(t, "gemma3", models[0].ModelType)
}

func TestDiscover_Good_QuantizationConfigFallback(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, filepath.Join(base, "qwen3-4b"), map[string]any{
		"model_type": "qwen3",
		"quantization_config": map[string]any{
			"bits":       8,
			"group_size": 128,
		},
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)
	assert.Equal(t, 8, models[0].QuantBits)
	assert.Equal(t, 128, models[0].QuantGroup)
}
