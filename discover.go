package inference

import (
	"encoding/json"
	"iter"
	"os"
	"path/filepath"
)

//	for model := range inference.Discover("/Volumes/Data/models") {
//	    fmt.Printf("%s  arch=%s  quant=%dbit\n", model.Path, model.ModelType, model.QuantBits)
//	}
type DiscoveredModel struct {
	Path       string // Absolute path to the model directory
	ModelType  string // Architecture from config.json (e.g. "gemma3", "qwen3", "llama")
	QuantBits  int    // Quantisation bits (0 if unquantised)
	QuantGroup int    // Quantisation group size
	NumFiles   int    // Number of safetensors weight files
}

// A valid directory has config.json + at least one .safetensors file.
//
//	for model := range inference.Discover("/Volumes/Data/models") {
//	    loadedModel, _ := inference.LoadModel(model.Path)
//	    _ = loadedModel.Close()
//	}
//
//	// Early exit - stop after finding the first match
//	for model := range inference.Discover(baseDir) {
//	    if model.ModelType == "gemma3" { use(model); break }
//	}
func Discover(baseDir string) iter.Seq[DiscoveredModel] {
	return func(yield func(DiscoveredModel) bool) {
		baseDir = filepath.Clean(baseDir)
		entries, err := os.ReadDir(baseDir)
		if err != nil {
			return
		}

		// Check baseDir itself (in case it's a model directory).
		if model, ok := probeModelDir(baseDir); ok {
			if !yield(model) {
				return
			}
		}

		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			dir := filepath.Join(baseDir, entry.Name())
			if model, ok := probeModelDir(dir); ok {
				if !yield(model) {
					return
				}
			}
		}
	}
}

// Accepts directories that contain config.json and at least one .safetensors file.
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

	model := DiscoveredModel{
		Path:      absDir,
		ModelType: probe.ModelType,
		NumFiles:  len(matches),
	}
	if probe.Quantization != nil {
		model.QuantBits = probe.Quantization.Bits
		model.QuantGroup = probe.Quantization.GroupSize
	}

	return model, true
}
