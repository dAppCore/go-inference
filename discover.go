package inference

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// DiscoveredModel describes a model directory found by Discover.
type DiscoveredModel struct {
	Path         string // Absolute path to the model directory
	ModelType    string // Architecture from config.json (e.g. "gemma3", "qwen3", "llama")
	QuantBits    int    // Quantisation bits (0 if unquantised)
	QuantGroup   int    // Quantisation group size
	NumFiles     int    // Number of safetensors weight files
}

// Discover scans baseDir for model directories. A valid model directory
// contains config.json and at least one .safetensors file.
// Scans one level deep (immediate subdirectories of baseDir).
func Discover(baseDir string) ([]DiscoveredModel, error) {
	entries, err := os.ReadDir(baseDir)
	if err != nil {
		return nil, err
	}

	var models []DiscoveredModel
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		dir := filepath.Join(baseDir, entry.Name())
		m, ok := probeModelDir(dir)
		if ok {
			models = append(models, m)
		}
	}

	// Also check baseDir itself (in case it's a model directory).
	if m, ok := probeModelDir(baseDir); ok {
		// Prepend so the base dir appears first.
		models = append([]DiscoveredModel{m}, models...)
	}

	return models, nil
}

// probeModelDir checks if dir looks like a model directory.
func probeModelDir(dir string) (DiscoveredModel, bool) {
	configPath := filepath.Join(dir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return DiscoveredModel{}, false
	}

	// Count safetensors files.
	matches, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if len(matches) == 0 {
		return DiscoveredModel{}, false
	}

	absDir, _ := filepath.Abs(dir)

	var probe struct {
		ModelType    string `json:"model_type"`
		Quantization *struct {
			Bits      int `json:"bits"`
			GroupSize int `json:"group_size"`
		} `json:"quantization"`
	}
	_ = json.Unmarshal(data, &probe)

	m := DiscoveredModel{
		Path:      absDir,
		ModelType: probe.ModelType,
		NumFiles:  len(matches),
	}
	if probe.Quantization != nil {
		m.QuantBits = probe.Quantization.Bits
		m.QuantGroup = probe.Quantization.GroupSize
	}

	return m, true
}
