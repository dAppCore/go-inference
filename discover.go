package inference

import (
	"cmp"
	"iter"
	"reflect"
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
		c := core.New()
		discoverDir(c.Fs(), absolutePath(baseDir), yield)
	}
}

func discoverDir(fsys *core.Fs, dir string, yield func(DiscoveredModel) bool) bool {
	if m, ok := probeModelDir(fsys, dir); ok {
		if !yield(m) {
			return false
		}
	}

	entries, ok := readDir(fsys, dir)
	if !ok {
		return true
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

// Accepts directories that contain config.json and at least one .safetensors file.
func probeModelDir(fsys *core.Fs, dir string) (DiscoveredModel, bool) {
	config := fsys.Read(joinPath(dir, "config.json"))
	if !config.OK {
		return DiscoveredModel{}, false
	}

	numFiles, ok := countSafetensors(fsys, dir)
	if !ok || numFiles == 0 {
		return DiscoveredModel{}, false
	}

	model := DiscoveredModel{
		Path:     absolutePath(dir),
		NumFiles: numFiles,
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

type dirEntry interface {
	Name() string
	IsDir() bool
}

func readDir(fsys *core.Fs, dir string) ([]dirEntry, bool) {
	result := fsys.List(dir)
	if !result.OK {
		return nil, false
	}

	entries, ok := dirEntries(result.Value)
	if !ok {
		return nil, false
	}

	slices.SortFunc(entries, func(a, b dirEntry) int {
		return cmp.Compare(a.Name(), b.Name())
	})
	return entries, true
}

func dirEntries(value any) ([]dirEntry, bool) {
	// core.Fs.List returns standard directory entries; adapt them locally.
	slice := reflect.ValueOf(value)
	if !slice.IsValid() || slice.Kind() != reflect.Slice {
		return nil, false
	}

	entries := make([]dirEntry, 0, slice.Len())
	for i := range slice.Len() {
		entry, ok := slice.Index(i).Interface().(dirEntry)
		if !ok {
			return nil, false
		}
		entries = append(entries, entry)
	}
	return entries, true
}

func countSafetensors(fsys *core.Fs, dir string) (int, bool) {
	entries, ok := readDir(fsys, dir)
	if !ok {
		return 0, false
	}

	count := 0
	for _, entry := range entries {
		if !entry.IsDir() && core.HasSuffix(entry.Name(), ".safetensors") {
			count++
		}
	}
	return count, true
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
