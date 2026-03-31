package inference

import (
	"iter"
	"os"
	"path/filepath"

	"dappco.re/go/core"
)

// DiscoveredModel describes a model directory found by Discover.
//
//	for m := range inference.Discover("/Volumes/Data/models") {
//	    fmt.Printf("%s  arch=%s  quant=%dbit\n", m.Path, m.ModelType, m.QuantBits)
//	}
type DiscoveredModel struct {
	Path      string // Absolute path to the model directory
	ModelType string // Architecture from config.json (e.g. "gemma3", "qwen3", "llama")
	QuantBits int    // Quantisation bits (0 if unquantised)
	QuantGroup int   // Quantisation group size
	NumFiles  int    // Number of safetensors weight files
}

// Discover yields model directories one level deep under baseDir.
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
		baseDir = core.CleanPath(baseDir, core.Env("DS"))
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
			dir := core.Path(baseDir, entry.Name())
			if m, ok := probeModelDir(dir); ok {
				if !yield(m) {
					return
				}
			}
		}
	}
}

// probeModelDir returns (model, true) when dir contains config.json + *.safetensors.
func probeModelDir(dir string) (DiscoveredModel, bool) {
	configPath := core.Path(dir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return DiscoveredModel{}, false
	}

	matches := core.PathGlob(core.Path(dir, "*.safetensors"))
	if len(matches) == 0 {
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
	if r := core.JSONUnmarshal(data, &probe); !r.OK {
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
