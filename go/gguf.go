// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"cmp"
	"io/fs"
	"slices"

	core "dappco.re/go"
	"dappco.re/go/inference/model/gguf"
)

// GGUFInfo summarises GGUF metadata without requiring a concrete runtime.
type GGUFInfo struct {
	Path             string
	Architecture     string
	VocabSize        int
	HiddenSize       int
	NumLayers        int
	ContextLength    int
	QuantBits        int
	QuantGroup       int
	QuantType        string
	QuantFamily      string
	TensorCount      int
	MetadataCount    int
	ValidationIssues []GGUFValidationIssue
}

// Valid reports whether metadata parsing found validation errors.
func (info GGUFInfo) Valid() bool {
	for _, issue := range info.ValidationIssues {
		if issue.Severity == GGUFValidationError {
			return false
		}
	}
	return true
}

// GGUFValidationSeverity classifies GGUF metadata validation findings.
type GGUFValidationSeverity string

const (
	GGUFValidationWarning GGUFValidationSeverity = "warning"
	GGUFValidationError   GGUFValidationSeverity = "error"
)

// GGUFValidationIssue describes one GGUF metadata validation issue.
type GGUFValidationIssue struct {
	Severity GGUFValidationSeverity `json:"severity"`
	Code     string                 `json:"code"`
	Message  string                 `json:"message"`
	Tensor   string                 `json:"tensor,omitempty"`
}

// ReadGGUFInfo reads GGUF header metadata without loading tensors.
//
// The wire parsing is delegated to the gguf package's subset reader —
// only the handful of discovery keys below are decoded, everything else
// (vocab tables, tokenizer config, rope settings) is skipped in place, so
// this stays cheap enough for per-directory discovery sweeps. The narrow
// GGUFInfo field mapping (including the fixed file_type→quantisation
// table) is this package's own and is pinned by its alloc-budget and
// behaviour tests.
func ReadGGUFInfo(modelPath string) (GGUFInfo, error) {
	ggufPath, err := gguf.ResolveFile(modelPath)
	if err != nil {
		return GGUFInfo{}, err
	}
	metadata, tensorCount, err := gguf.MetadataSubset(ggufPath, ggufKeyOfInterest)
	if err != nil {
		return GGUFInfo{}, err
	}
	absolutePath := ggufPath
	if abs := core.PathAbs(ggufPath); abs.OK {
		absolutePath = abs.Value.(string)
	}
	architecture := metadataString(metadata, "general.architecture")
	quantBits, quantGroup, quantType, quantFamily := ggufQuantisationFromMetadata(metadata)
	return GGUFInfo{
		Path:          absolutePath,
		Architecture:  architecture,
		VocabSize:     core.FirstPositive(metadataInt(metadata, architecture+".vocab_size"), metadataInt(metadata, "tokenizer.ggml.tokens")),
		HiddenSize:    metadataInt(metadata, architecture+".embedding_length"),
		NumLayers:     metadataInt(metadata, architecture+".block_count"),
		ContextLength: metadataInt(metadata, architecture+".context_length"),
		QuantBits:     quantBits,
		QuantGroup:    quantGroup,
		QuantType:     quantType,
		QuantFamily:   quantFamily,
		TensorCount:   tensorCount,
		MetadataCount: len(metadata),
	}, nil
}

// ggufKeyOfInterest reports whether ReadGGUFInfo queries this metadata key.
// Every other entry's value bytes are skipped inside gguf.MetadataSubset
// without touching the map — on real GGUF headers (hundreds of tokenizer
// entries) that skip is the difference between a handful of allocations
// and hundreds per model load.
func ggufKeyOfInterest(key string) bool {
	switch key {
	case "general.architecture", "general.file_type", "tokenizer.ggml.tokens":
		return true
	}
	return core.HasSuffix(key, ".vocab_size") ||
		core.HasSuffix(key, ".embedding_length") ||
		core.HasSuffix(key, ".block_count") ||
		core.HasSuffix(key, ".context_length")
}

// DiscoverModels returns safetensors and GGUF models beneath basePath.
func DiscoverModels(basePath string) []DiscoveredModel {
	resolvedPath := basePath
	if abs := core.PathAbs(basePath); abs.OK {
		resolvedPath = abs.Value.(string)
	}
	stat := core.Stat(resolvedPath)
	if !stat.OK {
		return nil
	}
	if !stat.Value.(core.FsFileInfo).IsDir() {
		if core.HasSuffix(core.Lower(resolvedPath), ".gguf") {
			if info, err := ReadGGUFInfo(resolvedPath); err == nil {
				return []DiscoveredModel{discoveredModelFromGGUF(info)}
			}
		}
		return nil
	}

	models := slices.Collect(Discover(resolvedPath))
	if err := core.PathWalkDir(resolvedPath, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil || !entry.IsDir() {
			return nil
		}
		ggufs := core.PathGlob(core.PathJoin(path, "*.gguf"))
		if len(ggufs) != 1 {
			return nil
		}
		info, err := ReadGGUFInfo(ggufs[0])
		if err != nil {
			return nil
		}
		models = append(models, discoveredModelFromGGUF(info))
		return nil
	}); err != nil {
		return nil
	}
	slices.SortFunc(models, func(a, b DiscoveredModel) int {
		return cmp.Compare(a.Path, b.Path)
	})
	return models
}

func discoveredModelFromGGUF(info GGUFInfo) DiscoveredModel {
	return DiscoveredModel{
		Path:        info.Path,
		ModelType:   info.Architecture,
		QuantBits:   info.QuantBits,
		QuantGroup:  info.QuantGroup,
		QuantType:   info.QuantType,
		QuantFamily: info.QuantFamily,
		NumFiles:    1,
		Format:      "gguf",
	}
}

func metadataString(metadata map[string]any, key string) string {
	if value, ok := metadata[key].(string); ok {
		return value
	}
	return ""
}

func metadataInt(metadata map[string]any, key string) int {
	switch value := metadata[key].(type) {
	case uint32:
		return int(value)
	case uint64:
		return int(value)
	default:
		return 0
	}
}

// ggufQuantisationFromMetadata maps general.file_type onto the narrow
// discovery quantisation fields. This fixed four-row table is deliberately
// NOT the gguf package's richer inference (majority tensor-type vote,
// per-type block sizes) — the values below are the discovery contract
// downstream backends were built against and stay as-is.
func ggufQuantisationFromMetadata(metadata map[string]any) (bits, group int, quantType, family string) {
	fileType := metadataInt(metadata, "general.file_type")
	switch fileType {
	case 0:
		return 32, 0, "f32", "f32"
	case 1:
		return 16, 0, "f16", "f16"
	case 7:
		return 8, 32, "q8_0", "q8"
	case 15:
		return 4, 32, "q4_k_m", "q4"
	default:
		return 0, 0, "", ""
	}
}
