package inference

import (
	"cmp"
	"iter"
	"slices"

	core "dappco.re/go"
)

//	for m := range inference.Discover("/Volumes/Data/models") {
//	    fmt.Printf("%s  arch=%s  quant=%dbit\n", m.Path, m.ModelType, m.QuantBits)
//	}
type DiscoveredModel struct {
	Path        string // Absolute path to the model directory or GGUF file
	ModelType   string // Architecture from config.json/GGUF metadata
	QuantBits   int    // Quantisation bits (0 if unquantised or unknown)
	QuantGroup  int    // Quantisation group size
	QuantType   string // Quantisation type, when known (e.g. q4_k_m, q8_0)
	QuantFamily string // Quantisation family, when known (e.g. q4, q8)
	NumFiles    int    // Number of weight files
	Format      string // safetensors or gguf when known
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
		c := core.New()
		discoverDir(c.Fs(), absolutePath(baseDir), yield)
	}
}

func discoverDir(fsys *core.Fs, dir string, yield func(DiscoveredModel) bool) bool {
	// Single readDir per directory — the entries feed both
	// probeModelDir's safetensors count AND the recursion. Previously
	// each directory was listed THREE times (probe → countSafetensors
	// → discoverDir's own readDir), with each listing also paying
	// reflect-based conversion. Now once, no reflect.
	entries, ok := readDir(fsys, dir)
	if !ok {
		// We can still try to probe the directory even if listing
		// fails — config.json read may succeed independently.
		entries = nil
	}

	if m, ok := probeModelDir(fsys, dir, entries); ok {
		if !yield(m) {
			return false
		}
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		if !discoverDir(fsys, joinPath(dir, entry.Name()), yield) {
			return false
		}
	}

	return true
}

// Accepts directories that contain config.json and at least one
// .safetensors file. `entries` is the pre-read directory listing —
// avoids the second readDir that countSafetensors used to do.
func probeModelDir(fsys *core.Fs, dir string, entries []core.FsDirEntry) (DiscoveredModel, bool) {
	config := fsys.Read(joinPath(dir, "config.json"))
	if !config.OK {
		return DiscoveredModel{}, false
	}

	numFiles := countSafetensors(entries)
	if numFiles == 0 {
		return DiscoveredModel{}, false
	}

	model := DiscoveredModel{
		Path:     absolutePath(dir),
		NumFiles: numFiles,
		Format:   "safetensors",
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
	if data, ok := config.Value.(string); ok && core.JSONUnmarshalString(data, &probe).OK {
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

// readDir returns the directory's entries sorted by name. The result
// is the raw []core.FsDirEntry from core.Fs.List — no reflect, no
// adapter allocation.
func readDir(fsys *core.Fs, dir string) ([]core.FsDirEntry, bool) {
	result := fsys.List(dir)
	if !result.OK {
		return nil, false
	}

	entries, ok := result.Value.([]core.FsDirEntry)
	if !ok {
		return nil, false
	}

	slices.SortFunc(entries, func(a, b core.FsDirEntry) int {
		return cmp.Compare(a.Name(), b.Name())
	})
	return entries, true
}

func countSafetensors(entries []core.FsDirEntry) int {
	count := 0
	for _, entry := range entries {
		if !entry.IsDir() && core.HasSuffix(entry.Name(), ".safetensors") {
			count++
		}
	}
	return count
}

func absolutePath(dir string) string {
	if core.PathIsAbs(dir) {
		return cleanPath(dir)
	}

	cwd := core.Env("DIR_CWD")
	if cwd == "" {
		return cleanPath(dir)
	}
	return joinPath(cwd, dir)
}

func joinPath(parts ...string) string {
	return core.CleanPath(core.Join(pathSeparator(), parts...), pathSeparator())
}

func cleanPath(path string) string {
	return core.CleanPath(path, pathSeparator())
}

func pathSeparator() string {
	if separator := core.Env("DS"); separator != "" {
		return separator
	}
	return "/"
}
