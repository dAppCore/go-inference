package inference

import (
	"encoding/json"
	"iter"
	"os"
	"path/filepath"
)

//	model := inference.DiscoveredModel{
//		Path:      "/models/gemma3-1b",
//		ModelType: "gemma3",
//		QuantBits: 4,
//		NumFiles:  4,
//	}
type DiscoveredModel struct {
	Path       string
	ModelType  string
	QuantBits  int
	QuantGroup int
	NumFiles   int
}

//	for model := range inference.Discover("/Volumes/Data/models") {
//		loadedModel, _ := inference.LoadModel(model.Path)
//		_ = loadedModel.Close()
//	}
//
//	for model := range inference.Discover(baseDir) {
//		if model.ModelType == "gemma3" {
//			break
//		}
//	}
func Discover(baseDir string) iter.Seq[DiscoveredModel] {
	return func(yield func(DiscoveredModel) bool) {
		baseDir = filepath.Clean(baseDir)
		entries, err := os.ReadDir(baseDir)
		if err != nil {
			return
		}

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

func probeModelDir(dir string) (DiscoveredModel, bool) {
	configPath := filepath.Join(dir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return DiscoveredModel{}, false
	}

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
