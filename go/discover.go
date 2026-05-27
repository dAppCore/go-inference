package inference

import (
	"cmp"
	"iter"
	"slices"
	"sync"

	core "dappco.re/go"
)

// discoverCore is a package-level Core handle reused across
// Discover calls. Profiling (alpha.95 era) showed core.New() per
// call burned ~51 allocs / ~13% of Discover's total cost — every
// invocation spun up a fresh ServiceRuntime + Registry pair just
// to get an Fs() handle, when the same Fs serves every call
// identically. sync.Once initialises on first use so test code
// that monkey-patches the global Core via core.New() before any
// Discover call still sees a usable instance.
//
// Risk: this couples Discover to the package-level Core lifetime
// (process-wide). Acceptable here because Fs() is stateless — no
// per-call state, no cancellation, no auth scope. If Fs() ever
// grows per-caller context, replace this with an option-pattern
// override on Discover (`WithCore(c)`) without breaking the
// existing zero-arg API.
var (
	discoverCoreOnce sync.Once
	discoverCore     *core.Core
)

func sharedDiscoverCore() *core.Core {
	discoverCoreOnce.Do(func() {
		discoverCore = core.New()
	})
	return discoverCore
}

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
		discoverDir(sharedDiscoverCore().Fs(), absolutePath(baseDir), yield)
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
//
// Order matters: single pass over entries first to count safetensors
// AND verify config.json exists. Only then read config.json. This
// short-circuits the wasted disk Read for junk directories that have
// neither — see Discover_NoModels_TenJunkDirs which used to pay one
// fsys.Read per dir before this gate.
func probeModelDir(fsys *core.Fs, dir string, entries []core.FsDirEntry) (DiscoveredModel, bool) {
	numFiles := 0
	hasConfig := false
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if name == "config.json" {
			hasConfig = true
		} else if core.HasSuffix(name, ".safetensors") {
			numFiles++
		}
	}
	if numFiles == 0 || !hasConfig {
		return DiscoveredModel{}, false
	}

	config := fsys.Read(joinPath(dir, "config.json"))
	if !config.OK {
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
