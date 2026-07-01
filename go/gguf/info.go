// SPDX-Licence-Identifier: EUPL-1.2

// Package gguf reads and writes the GGUF model-checkpoint file format
// (llama.cpp's on-disk format for quantised model weights) with no
// dependency on any concrete inference engine. ReadInfo and Metadata parse
// a .gguf file's header, key/value metadata, and tensor directory; the
// quantisation kernels convert dense float32 safetensors tensors into
// GGUF's Q4_0/Q5_0/Q8_0 and Q2_K..Q8_K block formats, and QuantizeModelPack
// writes the result as a new GGUF model pack.
//
// Loading GGUF tensor data into a concrete engine's array type (MLX, ROCm,
// CPU) — and dequantising GGUF blocks back to float32 for inference —
// stays with each engine; this package only understands the file format
// and the write-side (quantise) maths.
//
//	info, err := gguf.ReadInfo("/models/qwen3-4b-q4_k_m.gguf")
//	if err != nil { return err }
//	core.Println(info.Architecture, info.QuantType)
//
//	result, err := gguf.QuantizeModelPack(ctx, gguf.QuantizeOptions{
//	    SourcePack: gguf.Source{
//	        Root:         "/models/qwen3-4b",
//	        Architecture: "qwen3",
//	        WeightFiles:  []string{"/models/qwen3-4b/model.safetensors"},
//	    },
//	    OutputPath: "/models/qwen3-4b-gguf",
//	    Format:     gguf.QuantizeQ4_K,
//	})
package gguf

import (
	"strconv"

	core "dappco.re/go"
)

const maxGGUFCollectionEntries uint64 = 1 << 20

// Sentinel errors — lifted to package vars so the rare-but-hot-under-
// churn failure paths don't allocate a fresh core.NewError per hit.
var (
	errGGUFNoFile        = core.NewError("gguf: no .gguf file found")
	errGGUFMultipleFiles = core.NewError("gguf: multiple .gguf files found")
	errGGUFInvalidMagic  = core.NewError("gguf: invalid gguf magic")
	errGGUFStringTooLong = core.NewError("gguf: gguf string is unreasonably large")
)

const (
	ggufValueTypeUint8   = 0
	ggufValueTypeInt8    = 1
	ggufValueTypeUint16  = 2
	ggufValueTypeInt16   = 3
	ValueTypeUint32      = 4
	ggufValueTypeInt32   = 5
	ggufValueTypeFloat32 = 6
	ggufValueTypeBool    = 7
	ValueTypeString      = 8
	ggufValueTypeArray   = 9
	ggufValueTypeUint64  = 10
	ggufValueTypeInt64   = 11
	ggufValueTypeFloat64 = 12
)

const (
	ggufTensorTypeF32      = 0
	ggufTensorTypeF16      = 1
	TensorTypeQ4_0         = 2
	ggufTensorTypeQ4_1     = 3
	ggufTensorTypeQ5_0     = 6
	ggufTensorTypeQ5_1     = 7
	TensorTypeQ8_0         = 8
	ggufTensorTypeQ8_1     = 9
	ggufTensorTypeQ2K      = 10
	ggufTensorTypeQ3K      = 11
	ggufTensorTypeQ4K      = 12
	ggufTensorTypeQ5K      = 13
	ggufTensorTypeQ6K      = 14
	ggufTensorTypeQ8K      = 15
	ggufTensorTypeIQ2XXS   = 16
	ggufTensorTypeIQ2XS    = 17
	ggufTensorTypeIQ3XXS   = 18
	ggufTensorTypeIQ1S     = 19
	ggufTensorTypeIQ4NL    = 20
	ggufTensorTypeIQ3S     = 21
	ggufTensorTypeIQ2S     = 22
	ggufTensorTypeIQ4XS    = 23
	ggufTensorTypeI8       = 24
	ggufTensorTypeI16      = 25
	ggufTensorTypeI32      = 26
	ggufTensorTypeI64      = 27
	ggufTensorTypeF64      = 28
	ggufTensorTypeIQ1M     = 29
	ggufTensorTypeBF16     = 30
	ggufTensorTypeQ4_0_4_4 = 31
	ggufTensorTypeQ4_0_4_8 = 32
	ggufTensorTypeQ4_0_8_8 = 33
	ggufTensorTypeTQ1_0    = 34
	ggufTensorTypeTQ2_0    = 35
	ggufTensorTypeMXFP4    = 38
	ggufTensorTypeNVFP4    = 39
)

// Info summarises the metadata of a GGUF checkpoint.
type Info struct {
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
	Quantization     QuantizationInfo
	Tensors          []TensorInfo
	ValidationIssues []ValidationIssue
	TensorCount      int
	MetadataCount    int
}

// Valid reports whether tensor metadata passed basic shape/dtype validation.
func (info Info) Valid() bool {
	for _, issue := range info.ValidationIssues {
		if issue.Severity == GGUFValidationError {
			return false
		}
	}
	return true
}

// ValidationSeverity classifies GGUF metadata validation findings.
type ValidationSeverity string

const (
	GGUFValidationWarning ValidationSeverity = "warning"
	GGUFValidationError   ValidationSeverity = "error"
)

// ValidationIssue describes one GGUF tensor metadata validation issue.
type ValidationIssue struct {
	Severity ValidationSeverity `json:"severity"`
	Code     string             `json:"code"`
	Message  string             `json:"message"`
	Tensor   string             `json:"tensor,omitempty"`
}

// TensorInfo describes one tensor entry from the GGUF directory.
type TensorInfo struct {
	Name      string   `json:"name"`
	Type      uint32   `json:"type"`
	TypeName  string   `json:"type_name,omitempty"`
	DType     string   `json:"dtype,omitempty"`
	Bits      int      `json:"bits,omitempty"`
	BlockSize int      `json:"block_size,omitempty"`
	Shape     []uint64 `json:"shape,omitempty"`
	Elements  uint64   `json:"elements,omitempty"`
	Offset    uint64   `json:"offset,omitempty"`
	Quantized bool     `json:"quantized,omitempty"`
}

// TensorTypeSummary counts tensor dtypes found in a GGUF file.
type TensorTypeSummary struct {
	Type      uint32 `json:"type"`
	Name      string `json:"name"`
	DType     string `json:"dtype,omitempty"`
	Bits      int    `json:"bits,omitempty"`
	BlockSize int    `json:"block_size,omitempty"`
	Count     int    `json:"count"`
	Quantized bool   `json:"quantized,omitempty"`
}

// QuantizationInfo captures GGML quantization metadata beyond bit width.
type QuantizationInfo struct {
	Type         string              `json:"type,omitempty"`
	Family       string              `json:"family,omitempty"`
	Bits         int                 `json:"bits,omitempty"`
	GroupSize    int                 `json:"group_size,omitempty"`
	FileType     int                 `json:"file_type,omitempty"`
	FileTypeName string              `json:"file_type_name,omitempty"`
	Version      int                 `json:"version,omitempty"`
	Mixed        bool                `json:"mixed,omitempty"`
	TensorTypes  []TensorTypeSummary `json:"tensor_types,omitempty"`
}

type modelConfigProbe struct {
	ModelType             string   `json:"model_type"`
	VocabSize             int      `json:"vocab_size"`
	HiddenSize            int      `json:"hidden_size"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	Architectures         []string `json:"architectures"`
	NumLabels             int      `json:"num_labels"`
	TextConfig            struct {
		ModelType             string `json:"model_type"`
		VocabSize             int    `json:"vocab_size"`
		HiddenSize            int    `json:"hidden_size"`
		NumHiddenLayers       int    `json:"num_hidden_layers"`
		MaxPositionEmbeddings int    `json:"max_position_embeddings"`
	} `json:"text_config"`
	Quantization *struct {
		Bits      int `json:"bits"`
		GroupSize int `json:"group_size"`
	} `json:"quantization"`
	QuantizationConfig *struct {
		Bits      int `json:"bits"`
		GroupSize int `json:"group_size"`
	} `json:"quantization_config"`
}

// ReadInfo reads GGUF metadata without loading model weights into a
// concrete engine's array type.
func ReadInfo(modelPath string) (Info, error) {
	ggufPath, err := resolveGGUFFile(modelPath)
	if err != nil {
		return Info{}, err
	}

	metadata, tensors, err := parseGGUF(ggufPath)
	if err != nil {
		return Info{}, err
	}

	absolutePath := ggufPath
	if abs := core.PathAbs(ggufPath); abs.OK {
		absolutePath = abs.Value.(string)
	}

	config, _ := readModelConfig(core.PathDir(ggufPath))
	architecture := firstNonEmpty(
		metadataString(metadata["general.architecture"]),
		config.architecture(),
	)
	quantBits := config.quantBits()
	if quantBits == 0 {
		quantBits = inferQuantBits(tensors)
	}
	tensorInfos, validationIssues := buildGGUFTensorInfos(tensors)
	quantization := inferGGUFQuantization(metadata, tensorInfos)
	if quantization.Bits == 0 {
		quantization.Bits = quantBits
	}
	quantization.GroupSize = firstPositive(config.quantGroup(), quantization.GroupSize, quantizationGroupFromTensorTypes(quantization.TensorTypes))
	if quantBits == 0 {
		quantBits = quantization.Bits
	}

	info := Info{
		Path:             absolutePath,
		Architecture:     architecture,
		VocabSize:        firstPositive(config.vocabSize(), inferGGUFVocabSize(metadata, architecture)),
		HiddenSize:       firstPositive(config.hiddenSize(), inferGGUFHiddenSize(metadata, architecture)),
		NumLayers:        config.numLayers(),
		ContextLength:    firstPositive(config.contextLength(), inferGGUFContextLength(metadata, architecture)),
		QuantBits:        quantBits,
		QuantGroup:       quantization.GroupSize,
		QuantType:        quantization.Type,
		QuantFamily:      quantization.Family,
		Quantization:     quantization,
		Tensors:          tensorInfos,
		ValidationIssues: validationIssues,
		TensorCount:      len(tensors),
		MetadataCount:    len(metadata),
	}
	if info.NumLayers == 0 {
		info.NumLayers = inferLayerCount(metadata, tensors, info.Architecture)
	}

	return info, nil
}

func resolveGGUFFile(modelPath string) (string, error) {
	// Case-insensitive .gguf suffix check without allocating a lowered
	// copy of modelPath. Real callers always pass lowercase paths, but
	// stay lenient to the historical .GGUF spelling.
	if hasASCIIInsensitiveSuffix(modelPath, ".gguf") {
		return modelPath, nil
	}

	ggufs := core.PathGlob(core.PathJoin(modelPath, "*.gguf"))
	switch len(ggufs) {
	case 0:
		return "", errGGUFNoFile
	case 1:
		return ggufs[0], nil
	default:
		return "", errGGUFMultipleFiles
	}
}

// hasASCIIInsensitiveSuffix is a zero-alloc ASCII case-insensitive
// HasSuffix. Used in cold-start path probes where allocating a lowered
// copy of the input just to compare against a literal extension is
// wasteful (a few hundred bytes per ReadInfo at the file-open boundary).
func hasASCIIInsensitiveSuffix(s, suffix string) bool {
	if len(s) < len(suffix) {
		return false
	}
	si := len(s) - len(suffix)
	for i := 0; i < len(suffix); i++ {
		a := s[si+i]
		b := suffix[i]
		if a >= 'A' && a <= 'Z' {
			a += 'a' - 'A'
		}
		if b >= 'A' && b <= 'Z' {
			b += 'a' - 'A'
		}
		if a != b {
			return false
		}
	}
	return true
}

func readModelConfig(dir string) (*modelConfigProbe, error) {
	read := core.ReadFile(core.PathJoin(dir, "config.json"))
	if !read.OK {
		return nil, read.Value.(error)
	}
	var config modelConfigProbe
	if result := core.JSONUnmarshal(read.Value.([]byte), &config); !result.OK {
		return nil, result.Value.(error)
	}
	return &config, nil
}

func (probe *modelConfigProbe) architecture() string {
	if probe == nil {
		return ""
	}
	if probe.ModelType != "" {
		return normalizeArchitectureName(probe.ModelType)
	}
	if probe.TextConfig.ModelType != "" {
		return normalizeArchitectureName(probe.TextConfig.ModelType)
	}
	for _, architecture := range probe.Architectures {
		if modelType := architectureFromTransformersName(architecture); modelType != "" {
			return modelType
		}
	}
	return ""
}

func (probe *modelConfigProbe) numLayers() int {
	if probe == nil {
		return 0
	}
	if probe.NumHiddenLayers > 0 {
		return probe.NumHiddenLayers
	}
	return probe.TextConfig.NumHiddenLayers
}

func (probe *modelConfigProbe) vocabSize() int {
	if probe == nil {
		return 0
	}
	if probe.VocabSize > 0 {
		return probe.VocabSize
	}
	return probe.TextConfig.VocabSize
}

func (probe *modelConfigProbe) hiddenSize() int {
	if probe == nil {
		return 0
	}
	if probe.HiddenSize > 0 {
		return probe.HiddenSize
	}
	return probe.TextConfig.HiddenSize
}

func (probe *modelConfigProbe) contextLength() int {
	if probe == nil {
		return 0
	}
	if probe.MaxPositionEmbeddings > 0 {
		return probe.MaxPositionEmbeddings
	}
	return probe.TextConfig.MaxPositionEmbeddings
}

func (probe *modelConfigProbe) quantBits() int {
	if probe == nil {
		return 0
	}
	if probe.Quantization != nil {
		return probe.Quantization.Bits
	}
	if probe.QuantizationConfig != nil {
		return probe.QuantizationConfig.Bits
	}
	return 0
}

func (probe *modelConfigProbe) quantGroup() int {
	if probe == nil {
		return 0
	}
	if probe.Quantization != nil {
		return probe.Quantization.GroupSize
	}
	if probe.QuantizationConfig != nil {
		return probe.QuantizationConfig.GroupSize
	}
	return 0
}

// architectureSuffixes lists common HuggingFace transformers class-name
// suffixes stripped when deriving an architecture family tag from a
// sibling config.json's "architectures" list. This is a generic,
// format-driven fallback — not a per-model alias table (that policy
// belongs to each engine's own model registry; go-mlx's mlx/profile
// package is the authority there — see AX-8, lib never imports consumer).
var architectureSuffixes = [...]string{
	"ForConditionalGeneration",
	"ForSequenceClassification",
	"ForQuestionAnswering",
	"ForTokenClassification",
	"ForCausalLM",
	"Model",
}

// architectureFromTransformersName derives a lowercase architecture family
// tag from a HuggingFace transformers class name (e.g. "Qwen3ForCausalLM"
// → "qwen3"), or "" when name is empty or reduces to nothing recognisable.
func architectureFromTransformersName(name string) string {
	trimmed := name
	for _, suffix := range architectureSuffixes {
		if core.HasSuffix(trimmed, suffix) {
			trimmed = trimmed[:len(trimmed)-len(suffix)]
			break
		}
	}
	if trimmed == "" {
		return ""
	}
	return normalizeArchitectureName(trimmed)
}

// normalizeArchitectureName canonicalises an architecture identifier:
// lowercased, trimmed, with '-' and '.' folded to '_'.
func normalizeArchitectureName(value string) string {
	value = core.Lower(core.Trim(value))
	value = core.Replace(value, "-", "_")
	value = core.Replace(value, ".", "_")
	return value
}

func metadataString(value any) string {
	switch concrete := value.(type) {
	case string:
		return concrete
	default:
		return ""
	}
}

func metadataInt(value any) int {
	switch concrete := value.(type) {
	case uint8:
		return int(concrete)
	case int8:
		return int(concrete)
	case uint16:
		return int(concrete)
	case int16:
		return int(concrete)
	case uint32:
		return int(concrete)
	case int32:
		return int(concrete)
	case uint64:
		return int(concrete)
	case int64:
		return int(concrete)
	case float32:
		return int(concrete)
	case float64:
		return int(concrete)
	default:
		return 0
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if core.Trim(value) != "" {
			return value
		}
	}
	return ""
}

func firstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func inferGGUFVocabSize(metadata map[string]any, architecture string) int {
	return firstPositive(
		metadataIntForSuffix(metadata, architecture, "vocab_size", "n_vocab"),
		metadataArrayLen(metadata["tokenizer.ggml.tokens"]),
	)
}

func inferGGUFHiddenSize(metadata map[string]any, architecture string) int {
	return metadataIntForSuffix(metadata, architecture, "embedding_length", "hidden_size", "n_embd")
}

func inferGGUFContextLength(metadata map[string]any, architecture string) int {
	return metadataIntForSuffix(metadata, architecture, "context_length", "max_position_embeddings", "n_ctx")
}

func metadataIntForSuffix(metadata map[string]any, architecture string, suffixes ...string) int {
	// Prefix iteration order: split-base, architecture, general.
	// Encode as small fixed array (max 3 prefixes) with explicit length —
	// no slice allocation, no append of variadic-built temporary slices.
	var prefixes [3]string
	n := 0
	if architecture != "" {
		// Inline underscore split: most architectures ("qwen3", "llama",
		// "gemma") have no underscore — skip the core.SplitN alloc on the
		// common path. When present, slice without allocating new strings.
		if idx := core.Index(architecture, "_"); idx > 0 && idx < len(architecture)-1 {
			prefixes[n] = architecture[:idx]
			n++
		}
		prefixes[n] = architecture
		n++
	}
	prefixes[n] = "general"
	n++

	// Build "<prefix>.<suffix>" into a stack-allocated scratch buffer
	// instead of forcing a runtime.concatstring2 alloc per probe. Map
	// lookup via string(scratch[...]) still costs a key copy inside the
	// runtime, but the inputs themselves stay on the stack.
	var scratch [128]byte
	for i := 0; i < n; i++ {
		prefix := prefixes[i]
		for _, suffix := range suffixes {
			total := len(prefix) + 1 + len(suffix)
			if total > len(scratch) {
				// Fallback for unusually long keys — rare; rebuild via
				// alloc-allowed concat.
				if value := metadataInt(metadata[prefix+"."+suffix]); value > 0 {
					return value
				}
				continue
			}
			copy(scratch[:len(prefix)], prefix)
			scratch[len(prefix)] = '.'
			copy(scratch[len(prefix)+1:total], suffix)
			// map lookup with []byte-keyed conversion goes through the
			// runtime's []byte-to-string fast path that doesn't allocate.
			if value := metadataInt(metadata[string(scratch[:total])]); value > 0 {
				return value
			}
		}
	}
	for _, suffix := range suffixes {
		if value := metadataInt(metadata[suffix]); value > 0 {
			return value
		}
	}
	return 0
}

func metadataArrayLen(value any) int {
	switch concrete := value.(type) {
	case ggufStringArrayLen:
		return int(concrete)
	case []any:
		return len(concrete)
	case []string:
		return len(concrete)
	default:
		return 0
	}
}

func inferLayerCount(metadata map[string]any, tensors []TensorInfo, architecture string) int {
	if architecture != "" {
		// Same stack-scratch + m[string(b)] pattern as metadataIntForSuffix —
		// avoids the per-probe concat alloc that runtime.concatstring2 would
		// otherwise produce when escape analysis decides the result needs
		// the heap.
		var scratch [128]byte
		base := len(architecture) + 1
		suffixes := [...]string{"block_count", "n_layer", "num_hidden_layers"}
		if base > len(scratch) {
			// architecture comes from untrusted GGUF metadata; if the prefix
			// ("<architecture>.") cannot fit the stack scratch, fall back to
			// the alloc-allowed concat path rather than indexing out of range.
			// Mirrors the length guard in metadataIntForSuffix.
			for _, suffix := range suffixes {
				if count := metadataInt(metadata[architecture+"."+suffix]); count > 0 {
					return count
				}
			}
		} else {
			copy(scratch[:len(architecture)], architecture)
			scratch[len(architecture)] = '.'
			for _, suffix := range suffixes {
				end := base + len(suffix)
				if end > len(scratch) {
					if count := metadataInt(metadata[architecture+"."+suffix]); count > 0 {
						return count
					}
					continue
				}
				copy(scratch[base:end], suffix)
				if count := metadataInt(metadata[string(scratch[:end])]); count > 0 {
					return count
				}
			}
		}
	}

	maxLayer := -1
	for i := range tensors {
		if index := extractLayerIndex(tensors[i].Name); index > maxLayer {
			maxLayer = index
		}
	}
	if maxLayer >= 0 {
		return maxLayer + 1
	}
	return 0
}

// extractLayerIndexMarkers — pkg-level so we don't rebuild the slice
// on every tensor in inferLayerCount.
var extractLayerIndexMarkers = [...]string{"model.layers.", "layers.", "blk.", "block."}

func extractLayerIndex(name string) int {
	for _, marker := range extractLayerIndexMarkers {
		index := indexString(name, marker)
		if index < 0 {
			continue
		}
		start := index + len(marker)
		end := start
		for end < len(name) && name[end] >= '0' && name[end] <= '9' {
			end++
		}
		if end == start {
			continue
		}
		layer, err := strconv.Atoi(name[start:end])
		if err == nil {
			return layer
		}
	}
	return -1
}

func inferQuantBits(tensors []TensorInfo) int {
	// Bit widths are bounded (1, 2, 3, 4, 5, 6, 8, 16, 32, 64) so a
	// fixed-size array beats a map both in dispatch (direct index) and
	// allocation (none). Index 0 unused, 1..64 covers everything.
	var counts [65]int
	for i := range tensors {
		bits := ggufTensorBits(tensors[i].Type)
		if bits > 0 && bits < len(counts) {
			counts[bits]++
		}
	}

	bestBits := 0
	bestCount := 0
	for bits, count := range counts {
		if count == 0 {
			continue
		}
		if count > bestCount || (count == bestCount && bits > bestBits) {
			bestBits = bits
			bestCount = count
		}
	}
	return bestBits
}
