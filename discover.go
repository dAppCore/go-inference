package inference

import (
	"cmp"
	"encoding/json"
	iofs "io/fs"
	"iter"
	"os"
	"path/filepath"
	"slices"
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
		absBase, err := filepath.Abs(baseDir)
		if err != nil {
			return
		}
		discoverDir(absBase, yield)
	}
}

func discoverDir(dir string, yield func(DiscoveredModel) bool) bool {
	if m, ok := probeModelDir(dir); ok {
		if !yield(m) {
			return false
		}
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return true
	}

	slices.SortFunc(entries, func(a, b iofs.DirEntry) int {
		return cmp.Compare(a.Name(), b.Name())
	})

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		if !discoverDir(filepath.Join(dir, entry.Name()), yield) {
			return false
		}
	}

	return true
}

// Accepts directories that contain config.json and at least one .safetensors file.
func probeModelDir(dir string) (DiscoveredModel, bool) {
	configPath := filepath.Join(dir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return DiscoveredModel{}, false
	}

	matches, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if len(matches) == 0 {
		return DiscoveredModel{}, false
	}

	model := DiscoveredModel{
		Path:     absolutePath(dir),
		NumFiles: len(matches),
	}

	var probe struct {
		ModelType    string `json:"model_type"`
		Quantization *struct {
			Bits      int `json:"bits"`
			GroupSize int `json:"group_size"`
		} `json:"quantization"`
		QuantizationConfig *struct {
			Bits      int `json:"bits"`
			GroupSize int `json:"group_size"`
		} `json:"quantization_config"`
	}
	if err := json.Unmarshal(data, &probe); err == nil {
		model.ModelType = probe.ModelType
		if probe.Quantization != nil {
			model.QuantBits = probe.Quantization.Bits
			model.QuantGroup = probe.Quantization.GroupSize
		} else if probe.QuantizationConfig != nil {
			model.QuantBits = probe.QuantizationConfig.Bits
			model.QuantGroup = probe.QuantizationConfig.GroupSize
		}
	}

	return model, true
}

func absolutePath(dir string) string {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return dir
	}
	return abs
}
