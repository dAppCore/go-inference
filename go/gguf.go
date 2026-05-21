// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"cmp"
	"encoding/binary"
	"io"
	"io/fs"
	"slices"

	core "dappco.re/go"
)

const (
	ggufMagic      = 0x46554747
	ggufVersion    = 3
	ggufTypeUint32 = 4
	ggufTypeString = 8
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
func ReadGGUFInfo(modelPath string) (GGUFInfo, error) {
	ggufPath, err := resolveGGUFFile(modelPath)
	if err != nil {
		return GGUFInfo{}, err
	}
	metadata, tensorCount, err := parseGGUFMetadata(ggufPath)
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
		VocabSize:     firstPositiveInt(metadataInt(metadata, architecture+".vocab_size"), metadataInt(metadata, "tokenizer.ggml.tokens")),
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

func resolveGGUFFile(modelPath string) (string, error) {
	if core.HasSuffix(core.Lower(modelPath), ".gguf") {
		return modelPath, nil
	}
	ggufs := core.PathGlob(core.PathJoin(modelPath, "*.gguf"))
	switch len(ggufs) {
	case 0:
		return "", core.NewError("inference: no .gguf file found")
	case 1:
		return ggufs[0], nil
	default:
		return "", core.NewError("inference: multiple .gguf files found")
	}
}

func parseGGUFMetadata(path string) (map[string]any, int, error) {
	open := core.Open(path)
	if !open.OK {
		return nil, 0, core.Errorf("inference: open gguf: %w", open.Value.(error))
	}
	file := open.Value.(*core.OSFile)
	defer file.Close()

	// Header reads use binary.LittleEndian.UintX on a stack-allocated
	// fixed-size buffer instead of binary.Read — binary.Read uses
	// reflect and allocates per call (~1 alloc/value); the direct
	// LittleEndian path is zero-alloc. The header loop fires once per
	// metadata entry, so for a vocab-heavy GGUF that's hundreds of
	// avoidable allocs per model load.
	var hdr [8]byte

	if _, err := io.ReadFull(file, hdr[:4]); err != nil {
		return nil, 0, core.Errorf("inference: read gguf magic: %w", err)
	}
	if magic := binary.LittleEndian.Uint32(hdr[:4]); magic != ggufMagic {
		return nil, 0, core.NewError("inference: invalid gguf magic")
	}
	if _, err := io.ReadFull(file, hdr[:4]); err != nil {
		return nil, 0, core.Errorf("inference: read gguf version: %w", err)
	}
	if version := binary.LittleEndian.Uint32(hdr[:4]); version != ggufVersion {
		return nil, 0, core.Errorf("inference: unsupported gguf version: %d", version)
	}
	if _, err := io.ReadFull(file, hdr[:8]); err != nil {
		return nil, 0, core.Errorf("inference: read gguf tensor count: %w", err)
	}
	tensorCount := binary.LittleEndian.Uint64(hdr[:8])
	if _, err := io.ReadFull(file, hdr[:8]); err != nil {
		return nil, 0, core.Errorf("inference: read gguf metadata count: %w", err)
	}
	metadataCount := binary.LittleEndian.Uint64(hdr[:8])
	metadata := make(map[string]any, metadataCount)
	for range metadataCount {
		key, err := readGGUFString(file, hdr[:8])
		if err != nil {
			return nil, 0, err
		}
		if _, err := io.ReadFull(file, hdr[:4]); err != nil {
			return nil, 0, core.Errorf("inference: read gguf metadata type: %w", err)
		}
		valueType := binary.LittleEndian.Uint32(hdr[:4])
		value, err := readGGUFValue(file, valueType, hdr[:8])
		if err != nil {
			return nil, 0, err
		}
		metadata[key] = value
	}
	return metadata, int(tensorCount), nil
}

// readGGUFValue + readGGUFString accept a caller-owned scratch buffer
// so the reflect-allocating binary.Read path stays out of the per-entry
// inner loop. Callers pass hdr[:8] from the outer parse loop.
func readGGUFValue(reader io.Reader, valueType uint32, scratch []byte) (any, error) {
	switch valueType {
	case ggufTypeString:
		return readGGUFString(reader, scratch)
	case ggufTypeUint32:
		if _, err := io.ReadFull(reader, scratch[:4]); err != nil {
			return nil, core.Errorf("inference: read gguf uint32 metadata: %w", err)
		}
		return binary.LittleEndian.Uint32(scratch[:4]), nil
	default:
		return nil, core.Errorf("inference: unsupported gguf metadata type: %d", valueType)
	}
}

func readGGUFString(reader io.Reader, scratch []byte) (string, error) {
	if _, err := io.ReadFull(reader, scratch[:8]); err != nil {
		return "", core.Errorf("inference: read gguf string length: %w", err)
	}
	length := binary.LittleEndian.Uint64(scratch[:8])
	buf := make([]byte, length)
	if _, err := io.ReadFull(reader, buf); err != nil {
		return "", core.Errorf("inference: read gguf string: %w", err)
	}
	// buf is freshly-allocated and unreachable after this conversion —
	// core.AsString skips the []byte→string copy. A typical GGUF
	// metadata pass calls readGGUFString once per key + once per string
	// value (architecture, tokenizer.ggml.tokens, etc.); large vocabs
	// turn this into hundreds of KB of avoidable copies per load.
	return core.AsString(buf), nil
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

func firstPositiveInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}
