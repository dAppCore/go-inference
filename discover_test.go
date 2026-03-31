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

func createModelDir(t *testing.T, dir string, config map[string]any, numSafetensors int) {
	t.Helper()
	require.NoError(t, os.MkdirAll(dir, 0o755))

	if config != nil {
		data, err := json.Marshal(config)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(dir, "config.json"), data, 0o644))
	}

	for fileIndex := range numSafetensors {
		filename := filepath.Join(dir, fmt.Sprintf("model-%05d-of-%05d.safetensors", fileIndex+1, numSafetensors))
		require.NoError(t, os.WriteFile(filename, []byte("fake"), 0o644))
	}
}

func TestDiscover_Good_SingleModel(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, filepath.Join(base, "gemma3-1b"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)

	model := models[0]
	assert.Equal(t, "gemma3", model.ModelType)
	assert.Equal(t, 1, model.NumFiles)
	assert.Equal(t, 0, model.QuantBits)
	assert.Equal(t, 0, model.QuantGroup)
	assert.True(t, filepath.IsAbs(model.Path))
	assert.Contains(t, model.Path, "gemma3-1b")
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

	modelTypes := make([]string, len(models))
	for index, model := range models {
		modelTypes[index] = model.ModelType
	}
	assert.ElementsMatch(t, []string{"gemma3", "qwen3"}, modelTypes)
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

	model := models[0]
	assert.Equal(t, 4, model.QuantBits)
	assert.Equal(t, 64, model.QuantGroup)
}

func TestDiscover_Good_BaseDirIsModel(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, base, map[string]any{
		"model_type": "llama",
	}, 2)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)

	model := models[0]
	assert.Equal(t, "llama", model.ModelType)
	assert.Equal(t, 2, model.NumFiles)
}

func TestDiscover_Good_BaseDirPlusSubdir(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, base, map[string]any{
		"model_type": "parent_model",
	}, 1)
	createModelDir(t, filepath.Join(base, "child"), map[string]any{
		"model_type": "child_model",
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 2)

	assert.Equal(t, "parent_model", models[0].ModelType)
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
	base := t.TempDir()
	dir := filepath.Join(base, "incomplete")
	require.NoError(t, os.MkdirAll(dir, 0o755))

	config := map[string]any{"model_type": "gemma3"}
	data, err := json.Marshal(config)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(dir, "config.json"), data, 0o644))

	models := slices.Collect(Discover(base))
	assert.Empty(t, models)
}

func TestDiscover_Bad_NoConfig(t *testing.T) {
	base := t.TempDir()
	dir := filepath.Join(base, "no-config")
	require.NoError(t, os.MkdirAll(dir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("fake"), 0o644))

	models := slices.Collect(Discover(base))
	assert.Empty(t, models)
}

func TestDiscover_Bad_InvalidJSON(t *testing.T) {
	base := t.TempDir()
	dir := filepath.Join(base, "bad-json")
	require.NoError(t, os.MkdirAll(dir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "config.json"), []byte("{invalid}"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("fake"), 0o644))

	models := slices.Collect(Discover(base))
	require.Len(t, models, 0)
}

func TestDiscover_Ugly_SkipsRegularFiles(t *testing.T) {
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
	base := t.TempDir()
	createModelDir(t, filepath.Join(base, "no-type"), map[string]any{
		"vocab_size": 32000,
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)
	assert.Equal(t, "", models[0].ModelType)
}

func TestDiscover_Ugly_NoQuantisation(t *testing.T) {
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
		break
	}
	assert.Equal(t, 1, count)
}

func TestDiscover_Good_EarlyBreakOnSubdir(t *testing.T) {
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
	assert.Equal(t, 1, count)
}

func TestDiscover_Good_AbsolutePath(t *testing.T) {
	base := t.TempDir()
	createModelDir(t, filepath.Join(base, "test-model"), map[string]any{
		"model_type": "gemma3",
	}, 1)

	models := slices.Collect(Discover(base))
	require.Len(t, models, 1)
	assert.True(t, filepath.IsAbs(models[0].Path))
}
