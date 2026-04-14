package inference

import (
	"encoding/json"
	"io/fs"
	"iter"
	"os"
	"path/filepath"
)

//	for m := range inference.Discover("/Volumes/Data/models") {
//	    fmt.Printf("%s  arch=%s  quant=%dbit\n", m.Path, m.ModelType, m.QuantBits)
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
//	for m := range inference.Discover("/Volumes/Data/models") {
//	    model, _ := inference.LoadModel(m.Path)
//	}
//
//	// Early exit — stop after finding the first match
//	for m := range inference.Discover(dir) {
//	    if m.ModelType == "gemma3" { use(m); break }
//	}
func Discover(baseDir string) iter.Seq[DiscoveredModel] {
	return func(yield func(DiscoveredModel) bool) {
		root, err := filepath.Abs(baseDir)
		if err != nil {
			return
		}

		stopWalk := fs.SkipAll
		walkErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if !d.IsDir() {
				return nil
			}
			if m, ok := probeModelDir(path); ok {
				if !yield(m) {
					return stopWalk
				}
			}
			return nil
		})
		if walkErr != nil && walkErr != stopWalk {
			return
		}
	}
}

// Accepts directories that contain config.json and at least one .safetensors file.
func probeModelDir(dir string) (DiscoveredModel, bool) {
	configPath := filepath.Join(dir, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return DiscoveredModel{}, false
	}

	matches, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil || len(matches) == 0 {
		return DiscoveredModel{}, false
	}

	absDir := dir
	if abs, err := filepath.Abs(dir); err == nil {
		absDir = abs
	}

	var probe struct {
		ModelType    string `json:"model_type"`
		Quantization *struct {
			Bits      int `json:"bits"`
			GroupSize int `json:"group_size"`
		} `json:"quantization"`
		QuantConfig *struct {
			Bits      int `json:"bits"`
			GroupSize int `json:"group_size"`
		} `json:"quantization_config"`
	}
	if err := json.Unmarshal(configData, &probe); err != nil {
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
	} else if probe.QuantConfig != nil {
		model.QuantBits = probe.QuantConfig.Bits
		model.QuantGroup = probe.QuantConfig.GroupSize
	}

	return model, true
}
