// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"strconv"
	"strings"

	core "dappco.re/go"
)

const (
	ModelPackFileManifestContract = "rocm-model-pack-file-manifest-v1"

	ModelPackFilesStatusReady         = "ready"
	ModelPackFilesStatusMissing       = "missing"
	ModelPackFilesStatusAmbiguousGGUF = "ambiguous_gguf"

	ModelPackFormatGGUF        = "gguf"
	ModelPackFormatSafetensors = "safetensors"
	ModelPackFormatMixed       = "mixed"
	ModelPackFormatMissing     = "missing"
)

// ModelPackWeightFile describes one discovered local model weight file.
type ModelPackWeightFile struct {
	Path   string `json:"path,omitempty"`
	Name   string `json:"name,omitempty"`
	Format string `json:"format,omitempty"`
}

func (file ModelPackWeightFile) Clone() ModelPackWeightFile {
	return file
}

// ModelPackFileManifest is the filesystem side of the model load contract. It
// mirrors go-mlx's model_files.go root/file behaviour while keeping ROCm's
// metadata inspection richer: all weight files are preserved for diagnostics,
// and LoadWeightFiles records the go-mlx-compatible load preference.
type ModelPackFileManifest struct {
	Contract             string                `json:"contract,omitempty"`
	SourcePath           string                `json:"source_path,omitempty"`
	Root                 string                `json:"root,omitempty"`
	SourceIsDir          bool                  `json:"source_is_dir,omitempty"`
	Format               string                `json:"format,omitempty"`
	Status               string                `json:"status,omitempty"`
	WeightFiles          []ModelPackWeightFile `json:"weight_files,omitempty"`
	LoadWeightFiles      []ModelPackWeightFile `json:"load_weight_files,omitempty"`
	GGUFCount            int                   `json:"gguf_count,omitempty"`
	SafetensorsCount     int                   `json:"safetensors_count,omitempty"`
	MissingWeights       bool                  `json:"missing_weights,omitempty"`
	MixedWeights         bool                  `json:"mixed_weights,omitempty"`
	AmbiguousGGUF        bool                  `json:"ambiguous_gguf,omitempty"`
	ConfigPath           string                `json:"config_path,omitempty"`
	TokenizerPath        string                `json:"tokenizer_path,omitempty"`
	TokenizerConfigPath  string                `json:"tokenizer_config_path,omitempty"`
	ProcessorConfigPath  string                `json:"processor_config_path,omitempty"`
	SafetensorsIndexPath string                `json:"safetensors_index_path,omitempty"`
	Labels               map[string]string     `json:"labels,omitempty"`
}

func (manifest ModelPackFileManifest) Clone() ModelPackFileManifest {
	manifest.WeightFiles = cloneModelPackWeightFiles(manifest.WeightFiles)
	manifest.LoadWeightFiles = cloneModelPackWeightFiles(manifest.LoadWeightFiles)
	manifest.Labels = cloneStringMap(manifest.Labels)
	return manifest
}

func (manifest ModelPackFileManifest) WeightPaths() []string {
	return modelPackWeightFilePaths(manifest.WeightFiles)
}

func (manifest ModelPackFileManifest) LoadWeightPaths() []string {
	return modelPackWeightFilePaths(manifest.LoadWeightFiles)
}

// ResolveModelPackRoot returns the directory that owns model metadata and
// weights. File paths resolve to their parent directory; directory paths resolve
// to themselves.
func ResolveModelPackRoot(path string) (string, error) {
	manifest, err := InspectModelPackFiles(path)
	if err != nil {
		return "", err
	}
	return manifest.Root, nil
}

// InspectModelPackFiles discovers local model-pack files without parsing tensor
// payloads. It is safe for CLI/API preflight and shared by runtime inspection.
func InspectModelPackFiles(path string) (ModelPackFileManifest, error) {
	resolvedPath := path
	if abs := core.PathAbs(path); abs.OK {
		resolvedPath = abs.Value.(string)
	}
	stat := core.Stat(resolvedPath)
	if !stat.OK {
		return ModelPackFileManifest{}, stat.Value.(error)
	}
	info := stat.Value.(core.FsFileInfo)
	root := resolvedPath
	if !info.IsDir() {
		root = core.PathDir(resolvedPath)
	}
	manifest := ModelPackFileManifest{
		Contract:    ModelPackFileManifestContract,
		SourcePath:  resolvedPath,
		Root:        root,
		SourceIsDir: info.IsDir(),
	}
	manifest.WeightFiles = discoverModelPackWeightFiles(resolvedPath, info)
	manifest.Format = modelPackFormat(manifest.WeightFiles)
	manifest.GGUFCount, manifest.SafetensorsCount = modelPackWeightFormatCounts(manifest.WeightFiles)
	manifest.MissingWeights = len(manifest.WeightFiles) == 0
	manifest.MixedWeights = manifest.GGUFCount > 0 && manifest.SafetensorsCount > 0
	manifest.LoadWeightFiles = modelPackPreferredLoadWeightFiles(manifest)
	manifest.AmbiguousGGUF = manifest.SafetensorsCount == 0 && manifest.GGUFCount > 1
	switch {
	case manifest.MissingWeights:
		manifest.Status = ModelPackFilesStatusMissing
	case manifest.AmbiguousGGUF:
		manifest.Status = ModelPackFilesStatusAmbiguousGGUF
	default:
		manifest.Status = ModelPackFilesStatusReady
	}
	manifest.ConfigPath = modelPackSidecarPath(root, "config.json")
	manifest.TokenizerPath = modelPackSidecarPath(root, "tokenizer.json")
	manifest.TokenizerConfigPath = modelPackSidecarPath(root, "tokenizer_config.json")
	manifest.ProcessorConfigPath = modelPackSidecarPath(root, "processor_config.json")
	manifest.SafetensorsIndexPath = modelPackSidecarPath(root, "model.safetensors.index.json")
	manifest.Labels = modelPackFileManifestLabels(manifest)
	return manifest.Clone(), nil
}

func discoverModelPackWeightFiles(path string, info core.FsFileInfo) []ModelPackWeightFile {
	if !info.IsDir() {
		if modelPackFileFormat(path) != "" {
			return []ModelPackWeightFile{modelPackWeightFile(path)}
		}
		return nil
	}
	weights := []ModelPackWeightFile{}
	_ = core.PathWalkDir(path, func(current string, entry core.FsDirEntry, err error) error {
		if err != nil {
			return nil
		}
		if entry.IsDir() {
			if current != path && strings.HasPrefix(core.PathBase(current), ".") {
				return core.PathSkipDir
			}
			return nil
		}
		if modelPackFileFormat(current) != "" {
			weights = append(weights, modelPackWeightFile(current))
		}
		return nil
	})
	slices.SortFunc(weights, func(left, right ModelPackWeightFile) int {
		return strings.Compare(left.Path, right.Path)
	})
	return weights
}

func modelPackWeightFile(path string) ModelPackWeightFile {
	return ModelPackWeightFile{
		Path:   path,
		Name:   core.PathBase(path),
		Format: modelPackFileFormat(path),
	}
}

func modelPackFileFormat(path string) string {
	switch strings.ToLower(core.PathExt(path)) {
	case ".gguf":
		return ModelPackFormatGGUF
	case ".safetensors":
		return ModelPackFormatSafetensors
	default:
		return ""
	}
}

func modelPackFormat(weights []ModelPackWeightFile) string {
	gguf, safetensors := modelPackWeightFormatCounts(weights)
	switch {
	case gguf > 0 && safetensors > 0:
		return ModelPackFormatMixed
	case gguf > 0:
		return ModelPackFormatGGUF
	case safetensors > 0:
		return ModelPackFormatSafetensors
	default:
		return ModelPackFormatMissing
	}
}

func modelPackWeightFormatCounts(weights []ModelPackWeightFile) (int, int) {
	gguf := 0
	safetensors := 0
	for _, weight := range weights {
		switch weight.Format {
		case ModelPackFormatGGUF:
			gguf++
		case ModelPackFormatSafetensors:
			safetensors++
		}
	}
	return gguf, safetensors
}

func modelPackPreferredLoadWeightFiles(manifest ModelPackFileManifest) []ModelPackWeightFile {
	if manifest.SafetensorsCount > 0 {
		out := make([]ModelPackWeightFile, 0, manifest.SafetensorsCount)
		for _, weight := range manifest.WeightFiles {
			if weight.Format == ModelPackFormatSafetensors {
				out = append(out, weight.Clone())
			}
		}
		return out
	}
	if manifest.GGUFCount == 1 {
		for _, weight := range manifest.WeightFiles {
			if weight.Format == ModelPackFormatGGUF {
				return []ModelPackWeightFile{weight.Clone()}
			}
		}
	}
	return nil
}

func modelPackSidecarPath(root, name string) string {
	path := core.PathJoin(root, name)
	if stat := core.Stat(path); stat.OK && !stat.Value.(core.FsFileInfo).IsDir() {
		return path
	}
	return ""
}

func modelPackFileManifestLabels(manifest ModelPackFileManifest) map[string]string {
	labels := map[string]string{
		"model_pack_file_manifest_contract": ModelPackFileManifestContract,
		"model_pack_source":                 manifest.SourcePath,
		"model_pack_root":                   manifest.Root,
		"model_pack_source_is_dir":          strconv.FormatBool(manifest.SourceIsDir),
		"model_pack_format":                 manifest.Format,
		"model_pack_file_status":            manifest.Status,
		"model_pack_weight_files":           strconv.Itoa(len(manifest.WeightFiles)),
		"model_pack_load_weight_files":      strconv.Itoa(len(manifest.LoadWeightFiles)),
		"model_pack_gguf_files":             strconv.Itoa(manifest.GGUFCount),
		"model_pack_safetensors_files":      strconv.Itoa(manifest.SafetensorsCount),
		"model_pack_missing_weights":        strconv.FormatBool(manifest.MissingWeights),
		"model_pack_mixed_weights":          strconv.FormatBool(manifest.MixedWeights),
		"model_pack_ambiguous_gguf":         strconv.FormatBool(manifest.AmbiguousGGUF),
		"model_pack_config":                 strconv.FormatBool(manifest.ConfigPath != ""),
		"model_pack_tokenizer_json":         strconv.FormatBool(manifest.TokenizerPath != ""),
		"model_pack_tokenizer_config":       strconv.FormatBool(manifest.TokenizerConfigPath != ""),
		"model_pack_processor_config":       strconv.FormatBool(manifest.ProcessorConfigPath != ""),
		"model_pack_safetensors_index":      strconv.FormatBool(manifest.SafetensorsIndexPath != ""),
	}
	if names := modelPackWeightFileNames(manifest.WeightFiles); names != "" {
		labels["model_pack_weight_file_names"] = names
	}
	if names := modelPackWeightFileNames(manifest.LoadWeightFiles); names != "" {
		labels["model_pack_load_weight_file_names"] = names
	}
	return labels
}

func modelPackWeightFilePaths(weights []ModelPackWeightFile) []string {
	out := make([]string, 0, len(weights))
	for _, weight := range weights {
		if weight.Path != "" {
			out = append(out, weight.Path)
		}
	}
	return out
}

func modelPackWeightFileNames(weights []ModelPackWeightFile) string {
	names := make([]string, 0, len(weights))
	for _, weight := range weights {
		if weight.Name != "" {
			names = append(names, weight.Name)
		}
	}
	return strings.Join(names, ",")
}

func cloneModelPackWeightFiles(weights []ModelPackWeightFile) []ModelPackWeightFile {
	out := make([]ModelPackWeightFile, 0, len(weights))
	for _, weight := range weights {
		out = append(out, weight.Clone())
	}
	return out
}
