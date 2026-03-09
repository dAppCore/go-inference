package inference

import (
	"encoding/json"
	"iter"
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
func Discover(baseDir string) iter.Seq[DiscoveredModel] {
	return func(yield func(DiscoveredModel) bool) {
		baseDir = filepath.Clean(baseDir)
		entries, err := os.ReadDir(baseDir)
		if err != nil {
			return
		}

		// Check baseDir itself (in case it's a model directory).
		if m, ok := probeModelDir(baseDir); ok {
			if !yield(m) {
				return
			}
		}

		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			dir := filepath.Join(baseDir, entry.Name())
			if m, ok := probeModelDir(dir); ok {
				if !yield(m) {
					return
				}
			}
		}
	}
}

// probeModelDir checks if dir looks like a model directory.
func probeModelDir(dir string) (DiscoveredModel, bool) {
	configPath := filepath.Join(dir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return DiscoveredModel{}, false
	}

	// Count safetensors files.
	matches, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil || len(matches) == 0 {
		return DiscoveredModel{}, false
	}

	absDir, err := filepath.Abs(dir)
	if err != nil {
		absDir = dir
	}

	var probe struct {
		ModelType    string `json:"model_type"`
		Quantization *struct {
			Bits      int `json:"bits"`
			GroupSize int `json:"group_size"`
		} `json:"quantization"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return DiscoveredModel{}, false
	}

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
