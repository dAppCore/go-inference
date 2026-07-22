// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	core "dappco.re/go"

	"dappco.re/go/inference/model/quant/jang"
)

// InspectLocalMetadata resolves a local model directory — either a
// `models--org--name` cache root or a specific snapshot within one — into
// its ModelMetadata, reading config.json and any JANG quantisation config
// from the resolved snapshot. It returns the resolved snapshot root
// alongside the metadata so a caller can address individual weight files.
//
//	meta, root, err := hf.InspectLocalMetadata("/models/cache/models--org--name")
//	if err != nil { return err }
//	weights := core.PathJoin(root, meta.Files[0].Name)
func InspectLocalMetadata(path string) (ModelMetadata, string, error) {
	root := ResolveLocalMetadataRoot(path)
	read := core.ReadFile(core.PathJoin(root, "config.json"))
	if !read.OK {
		return ModelMetadata{}, root, core.E("hf.InspectLocalMetadata", "read local config.json", read.Err())
	}
	var config ModelConfig
	if result := core.JSONUnmarshal(read.Bytes(), &config); !result.OK {
		return ModelMetadata{}, root, core.E("hf.InspectLocalMetadata", "parse local config.json", result.Err())
	}
	files := LocalModelFiles(root)
	info, _ := jang.ReadConfig(root)
	return ModelMetadata{
		ID:     LocalModelID(path, root),
		Config: config,
		Files:  files,
		JANG:   info,
	}, root, nil
}

// ResolveLocalMetadataRoot resolves path to the directory that actually
// holds config.json and the weight files. Three shapes are recognised:
//
//   - path is a `models--org--name` cache root with a `snapshots/` child —
//     resolves to the lexically-first snapshot directory (the dominant
//     single-snapshot case resolves in one ReadDir).
//   - path already points at a config.json file — resolves to its parent
//     directory.
//   - anything else — path is returned unchanged, assumed to already be a
//     model directory.
//
// Example:
//
//	hf.ResolveLocalMetadataRoot("/cache/models--org--name")
//	// -> "/cache/models--org--name/snapshots/<lexically-first-rev>"
func ResolveLocalMetadataRoot(path string) string {
	// Replace filepath.Glob(path/snapshots/*/config.json) with a single
	// ReadDir of path/snapshots. Glob runs a readdir then per-match stat
	// *and* allocates the full match path strings plus an outer []string.
	// ReadDir hands back DirEntry values; picking the lexically-first
	// directory name and letting the caller's subsequent ReadFile of
	// config.json surface a missing-file error if the snapshot is
	// incomplete keeps the same observable shape as the previous Glob miss
	// path, at a fraction of the syscalls for the dominant single-snapshot
	// case.
	snapshotsDir := core.PathJoin(path, "snapshots")
	read := core.ReadDir(core.DirFS(snapshotsDir), ".")
	if read.OK {
		entries, ok := read.Value.([]core.FsDirEntry)
		if ok && len(entries) > 0 {
			// Find the lexically-first directory entry. ReadDir on
			// Darwin/Linux returns dirents in arbitrary order, so scan all
			// entries and track the smallest valid name.
			var winner string
			for _, entry := range entries {
				if !entry.IsDir() {
					continue
				}
				name := entry.Name()
				if winner == "" || name < winner {
					winner = name
				}
			}
			if winner != "" {
				return core.PathJoin(snapshotsDir, winner)
			}
		}
	}
	// hasSuffixFold avoids allocating a lowered copy of the full path
	// (paths can be long: ~/.cache/huggingface/hub/...) just to test a
	// 12-byte suffix.
	if hasSuffixFold(path, "config.json") {
		return core.PathDir(path)
	}
	return path
}

// localModelIDSearchOrder is the small array LocalModelID walks — hoisted so
// the slice literal isn't allocated per call.
var localModelIDSearchOrder = [2]int{0, 1}

// LocalModelID derives an "org/name" model id from the HuggingFace cache
// directory convention (`models--org--name`), walking up from root and then
// from inputPath. Falls back to root's base name when no `models--` segment
// is found in either path.
//
//	hf.LocalModelID(snapshot, "/cache/models--mlx-community--gemma-4-e2b-it-4bit")
//	// -> "mlx-community/gemma-4-e2b-it-4bit"
func LocalModelID(inputPath, root string) string {
	paths := [2]string{root, inputPath}
	for _, idx := range localModelIDSearchOrder {
		path := paths[idx]
		for current := path; current != "" && current != "."; {
			base := core.PathBase(current)
			if core.HasPrefix(base, "models--") {
				return core.Replace(core.TrimPrefix(base, "models--"), "--", "/")
			}
			parent := core.PathDir(current)
			if parent == current {
				break
			}
			current = parent
		}
	}
	return core.PathBase(root)
}

// LocalModelFiles lists the weight and tokenizer files directly inside
// root — the shapes InferJANG and WeightFormatAndBytes consume
// (*.safetensors, *.gguf, *.bin, tokenizer.json, tokenizer_config.json).
// Subdirectories and any other file are skipped. A missing or unreadable
// root yields an empty (non-nil) slice rather than an error.
func LocalModelFiles(root string) []ModelFile {
	// Pre-size: a typical pack has 1-4 safetensors shards + tokenizer.json
	// + tokenizer_config.json. 8 is a comfortable initial capacity that
	// avoids growslice for almost every real model.
	files := make([]ModelFile, 0, 8)
	// One ReadDir against the snapshot directory beats five filepath.Glob
	// passes (one per pattern) — each Glob pays its own readdir plus a
	// per-entry filepath.Match allocation.
	read := core.ReadDir(core.DirFS(root), ".")
	if !read.OK {
		return files
	}
	entries, ok := read.Value.([]core.FsDirEntry)
	if !ok {
		return files
	}
	// core.ReadDir (via os.DirFS -> os.ReadDir) already returns entries
	// sorted by name, so filtering preserves that order without a
	// post-pass sort.
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !isLocalModelFileName(name) {
			continue
		}
		var size uint64
		if info, err := entry.Info(); err == nil {
			size = uint64(info.Size())
		}
		files = append(files, ModelFile{Name: name, Size: size})
	}
	return files
}

// isLocalModelFileName reports whether name is one of the weight or
// tokenizer file shapes LocalModelFiles surfaces.
func isLocalModelFileName(name string) bool {
	switch name {
	case "tokenizer.json", "tokenizer_config.json":
		return true
	}
	// Suffix tests on the weight extensions. The most common shape is
	// "*.safetensors" so put that first.
	return hasSuffixFold(name, ".safetensors") ||
		hasSuffixFold(name, ".gguf") ||
		hasSuffixFold(name, ".bin")
}

// hasSuffixFold reports whether s ends with suffix using ASCII case-folding.
// Suffix is required to be lowercase. Pure scan, no allocations.
func hasSuffixFold(s, suffix string) bool {
	if len(s) < len(suffix) {
		return false
	}
	off := len(s) - len(suffix)
	for i := 0; i < len(suffix); i++ {
		c := s[off+i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != suffix[i] {
			return false
		}
	}
	return true
}

// Weight-format tags returned by WeightFormatAndBytes.
const (
	formatSafetensors = "safetensors"
	formatGGUF        = "gguf"
	formatMixed       = "mixed"
	formatBin         = "bin"
)

// WeightFormatAndBytes inspects files and reports the predominant weight
// format — "safetensors", "gguf", "bin", or "mixed" when both safetensors
// and gguf shards are present — plus the summed byte size of recognised
// weight files. Tokenizer/config files are not counted. Empty input returns
// ("", 0).
//
//	format, total := hf.WeightFormatAndBytes(meta.Files)
func WeightFormatAndBytes(files []ModelFile) (string, uint64) {
	if len(files) == 0 {
		return "", 0
	}
	var format string
	var total uint64
	for _, file := range files {
		// hasSuffixFold avoids a per-file Lower alloc — model weight
		// filenames are ASCII so case-folding the suffix is sufficient.
		name := file.filename()
		switch {
		case hasSuffixFold(name, ".safetensors"):
			if format == "" {
				format = formatSafetensors
			} else if format != formatSafetensors {
				format = formatMixed
			}
			total += file.byteSize()
		case hasSuffixFold(name, ".gguf"):
			if format == "" {
				format = formatGGUF
			} else if format != formatGGUF {
				format = formatMixed
			}
			total += file.byteSize()
		case hasSuffixFold(name, ".bin"):
			if format == "" {
				format = formatBin
			}
			total += file.byteSize()
		}
	}
	return format, total
}
