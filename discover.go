package inference

import (
	"cmp"
	iofs "io/fs"
	"iter"
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
		fs := (&core.Fs{}).NewUnrestricted()
		discoverDir(baseDir, fs, yield)
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
		if !discoverDir(core.Path(dir, entry.Name()), fs, yield) {
			return false
		}
	}

	return true
}

// Accepts directories that contain config.json and at least one .safetensors file.
func probeModelDir(dir string, fs *core.Fs) (DiscoveredModel, bool) {
	configPath := core.Path(dir, "config.json")
	r := fs.Read(configPath)
	if !r.OK {
		return DiscoveredModel{}, false
	}

	matches := core.PathGlob(core.Path(dir, "*.safetensors"))
	if len(matches) == 0 {
		return DiscoveredModel{}, false
	}

	var probe struct {
		ModelType    string `json:"model_type"`
		Quantization *struct {
			Bits      int `json:"bits"`
			GroupSize int `json:"group_size"`
		} `json:"quantization"`
	}
	if rr := core.JSONUnmarshalString(r.Value.(string), &probe); !rr.OK {
		return DiscoveredModel{}, false
	}

	model := DiscoveredModel{
		Path:      absolutePath(dir),
		ModelType: probe.ModelType,
		NumFiles:  len(matches),
	}
	if probe.Quantization != nil {
		model.QuantBits = probe.Quantization.Bits
		model.QuantGroup = probe.Quantization.GroupSize
	}

	return model, true
}

func absolutePath(dir string) string {
	if core.PathIsAbs(dir) {
		return dir
	}
	return core.Path(dir)
}
