package inference

import (
	iofs "io/fs"
	"iter"

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
		baseDir = core.CleanPath(baseDir, core.Env("DS"))
		fs := (&core.Fs{}).NewUnrestricted()

		r := fs.List(baseDir)
		if !r.OK {
			return
		}
		entries := r.Value.([]iofs.DirEntry)

		if m, ok := probeModelDir(baseDir, fs); ok {
			if !yield(m) {
				return
			}
		}

		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			dir := core.Path(baseDir, entry.Name())
			if m, ok := probeModelDir(dir, fs); ok {
				if !yield(m) {
					return
				}
			}
		}
	}
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

	absDir := dir
	if !core.PathIsAbs(dir) {
		absDir = core.Path(dir)
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
