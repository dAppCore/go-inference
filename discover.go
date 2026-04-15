package inference

import (
	"cmp"
	iofs "io/fs"
	"iter"
	"path/filepath"
	"slices"

	"dappco.re/go/core"
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
		fs := (&core.Fs{}).NewUnrestricted()
		discoverDir(absBase, fs, yield)
	}
}

func discoverDir(dir string, fs *core.Fs, yield func(DiscoveredModel) bool) bool {
	if m, ok := probeModelDir(dir, fs); ok {
		if !yield(m) {
			return false
		}
	}

	r := fs.List(dir)
	if !r.OK {
		return true
	}

	entries := r.Value.([]iofs.DirEntry)
	slices.SortFunc(entries, func(a, b iofs.DirEntry) int {
		return cmp.Compare(a.Name(), b.Name())
	})

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		if !discoverDir(filepath.Join(dir, entry.Name()), fs, yield) {
			return false
		}
	}

	return true
}

// Accepts directories that contain config.json and at least one .safetensors file.
func probeModelDir(dir string, fs *core.Fs) (DiscoveredModel, bool) {
	configPath := filepath.Join(dir, "config.json")
	r := fs.Read(configPath)
	if !r.OK {
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
	if rr := core.JSONUnmarshalString(r.Value.(string), &probe); rr.OK {
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
