package inference

import (
	"cmp"
	"iter"
	"slices"
	"sync"

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
//
// The walk reads the host filesystem through core's package-level
// primitives (core.ReadDir over core.DirFS, core.ReadFile) rather
// than a *core.Fs handle. Those two calls are EXACTLY what
// Fs.List/Fs.Read run once validatePath has cleared a path — and with
// the unrestricted "/" root a plain core.New() hands out, validatePath
// is a pass-through, so no sandbox is lost here.
//
// Dropping the handle also drops Fs.path's unconditional "/" prefix
// (core's fs.go: CleanPath("/"+p, PathSeparator)). That is harmless
// for a POSIX absolute path but corrupts a drive-letter one —
// filepath.Clean(`/C:\models\x`) is `\C:\models\x` on Windows, which
// no Win32 open can resolve. It is why every absolute-path Discover
// returned nothing on the windows runner while macOS and linux stayed
// green (run 31101973970). It removes the package-level core.New()
// with it — a sync.Once handle that existed only to reach Fs().
func Discover(baseDir string) iter.Seq[DiscoveredModel] {
	return func(yield func(DiscoveredModel) bool) {
		discoverDir(absolutePath(baseDir), yield)
	}
}

func discoverDir(dir string, yield func(DiscoveredModel) bool) bool {
	// Single readDir per directory — the entries feed both
	// probeModelDir's safetensors count AND the recursion. Previously
	// each directory was listed THREE times (probe → countSafetensors
	// → discoverDir's own readDir), with each listing also paying
	// reflect-based conversion. Now once, no reflect.
	entries, ok := readDir(dir)
	if !ok {
		// We can still try to probe the directory even if listing
		// fails — config.json read may succeed independently.
		entries = nil
	}

	if m, ok := probeModelDir(dir, entries); ok {
		if !yield(m) {
			return false
		}
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		if !discoverDir(joinPath(dir, entry.Name()), yield) {
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
// read per dir before this gate.
func probeModelDir(dir string, entries []core.FsDirEntry) (DiscoveredModel, bool) {
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

	config := core.ReadFile(joinPath(dir, "config.json"))
	if !config.OK {
		return DiscoveredModel{}, false
	}
	raw, ok := config.Value.([]byte)
	if !ok {
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
	// AsString over the freshly-read bytes — they become unreachable
	// after this, so the copy is dead weight (same contract Fs.Read
	// uses internally).
	if data := core.AsString(raw); core.JSONUnmarshalString(data, &probe).OK {
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
// is the raw []core.FsDirEntry from core.ReadDir — no reflect, no
// adapter allocation.
func readDir(dir string) ([]core.FsDirEntry, bool) {
	result := core.ReadDir(core.DirFS(dir), ".")
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
	sep := pathSeparator()
	return core.CleanPath(core.Join(sep, parts...), sep)
}

func cleanPath(path string) string {
	return core.CleanPath(path, pathSeparator())
}

// pathSeparator resolves the directory separator once per process and
// caches the result. The previous shape hit core.Env("DS") on every
// call — joinPath / cleanPath fire deep inside the discover walk
// (one per directory entry, hundreds-to-thousands of calls per
// scan), and Env walks a map fallback to os.Getenv when the key is
// unset (the common case for "DS"). The override is set once at
// process start (typically by tests) and never mutates, so sync.Once
// is the natural fit.
func pathSeparator() string {
	pathSeparatorOnce.Do(func() {
		if separator := core.Env("DS"); separator != "" {
			pathSeparatorCache = separator
			return
		}
		pathSeparatorCache = "/"
	})
	return pathSeparatorCache
}

var (
	pathSeparatorOnce  sync.Once
	pathSeparatorCache string
)
