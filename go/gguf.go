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

	var magic uint32
	if err := binary.Read(file, binary.LittleEndian, &magic); err != nil {
		return nil, 0, core.Errorf("inference: read gguf magic: %w", err)
	}
	if magic != ggufMagic {
		return nil, 0, core.NewError("inference: invalid gguf magic")
	}
	var version uint32
	if err := binary.Read(file, binary.LittleEndian, &version); err != nil {
		return nil, 0, core.Errorf("inference: read gguf version: %w", err)
	}
	if version != ggufVersion {
		return nil, 0, core.Errorf("inference: unsupported gguf version: %d", version)
	}
	var tensorCount uint64
	if err := binary.Read(file, binary.LittleEndian, &tensorCount); err != nil {
		return nil, 0, core.Errorf("inference: read gguf tensor count: %w", err)
	}
	var metadataCount uint64
	if err := binary.Read(file, binary.LittleEndian, &metadataCount); err != nil {
		return nil, 0, core.Errorf("inference: read gguf metadata count: %w", err)
	}
	metadata := make(map[string]any, metadataCount)
	for range metadataCount {
		key, err := readGGUFString(file)
		if err != nil {
			return nil, 0, err
		}
		var valueType uint32
		if err := binary.Read(file, binary.LittleEndian, &valueType); err != nil {
			return nil, 0, core.Errorf("inference: read gguf metadata type: %w", err)
		}
		value, err := readGGUFValue(file, valueType)
		if err != nil {
			return nil, 0, err
		}
		metadata[key] = value
	}
	return metadata, int(tensorCount), nil
}

func readGGUFValue(reader io.Reader, valueType uint32) (any, error) {
	switch valueType {
	case ggufTypeString:
		return readGGUFString(reader)
	case ggufTypeUint32:
		var value uint32
		if err := binary.Read(reader, binary.LittleEndian, &value); err != nil {
			return nil, core.Errorf("inference: read gguf uint32 metadata: %w", err)
		}
		return value, nil
	default:
		return nil, core.Errorf("inference: unsupported gguf metadata type: %d", valueType)
	}
}

func readGGUFString(reader io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(reader, binary.LittleEndian, &length); err != nil {
		return "", core.Errorf("inference: read gguf string length: %w", err)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(reader, buf); err != nil {
		return "", core.Errorf("inference: read gguf string: %w", err)
	}
	return string(buf), nil
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
