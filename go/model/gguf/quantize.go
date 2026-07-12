// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"context"
	"math"
	"sort"

	core "dappco.re/go"

	"dappco.re/go/inference/model/safetensors"
)

// QuantizeFormat names the GGUF quantization format requested by the caller.
type QuantizeFormat string

const (
	QuantizeQ8_0   QuantizeFormat = "q8_0"
	QuantizeQ4_0   QuantizeFormat = "q4_0"
	QuantizeQ5_0   QuantizeFormat = "q5_0"
	QuantizeQ4_K_M QuantizeFormat = "q4_k_m"
	QuantizeQ4_K   QuantizeFormat = "q4_k"
	QuantizeQ5_K   QuantizeFormat = "q5_k"
	QuantizeQ6_K   QuantizeFormat = "q6_k"
	QuantizeQ8_K   QuantizeFormat = "q8_k"
	QuantizeQ3_K   QuantizeFormat = "q3_k"
	QuantizeQ2_K   QuantizeFormat = "q2_k"

	ggufQuantizeOutputWeights = "model.gguf"
)

// Source identifies a local dense safetensors model pack to quantise into
// GGUF. This package does not validate model-pack structure itself — each
// engine (go-mlx, go-rocm, go-cpu) owns its own pack inspector and maps the
// result onto a Source before calling QuantizeModelPack. Mirrors
// merge.Source's shape: a minimal local descriptor rather than an import of
// any engine's concrete model-pack type (AX-8, lib never imports consumer).
type Source struct {
	// Root is the model pack's directory. Sibling metadata (config.json,
	// tokenizer files, chat templates) is copied from here into the
	// generated GGUF pack's output directory.
	Root string `json:"root"`

	// Architecture is written into the generated GGUF's
	// general.architecture metadata key and used to prefix the
	// <architecture>.vocab_size / .embedding_length / .block_count /
	// .context_length keys the GGUF spec expects.
	Architecture string `json:"architecture,omitempty"`

	// VocabSize, HiddenSize, NumLayers, and ContextLength — when positive —
	// are written into the corresponding <architecture>.* GGUF metadata
	// keys so a downstream GGUF loader does not need to re-derive them
	// from the tensor directory.
	VocabSize     int `json:"vocab_size,omitempty"`
	HiddenSize    int `json:"hidden_size,omitempty"`
	NumLayers     int `json:"num_layers,omitempty"`
	ContextLength int `json:"context_length,omitempty"`

	// WeightFiles lists the pack's safetensors shard paths — exactly one
	// entry for a single-file pack, more for a sharded export. Every entry
	// must end in ".safetensors".
	WeightFiles []string `json:"weight_files"`
}

// QuantizeOptions configures native Go safetensors-to-GGUF quantization.
type QuantizeOptions struct {
	SourcePack Source            `json:"source_pack"`
	OutputPath string            `json:"output_path"`
	Format     QuantizeFormat    `json:"format,omitempty"`
	Labels     map[string]string `json:"labels,omitempty"`
}

// QuantizeResult reports the paths of the generated GGUF model pack and
// its metadata.
type QuantizeResult struct {
	OutputPath       string         `json:"output_path"`
	WeightPath       string         `json:"weight_path"`
	RequestedFormat  QuantizeFormat `json:"requested_format"`
	Format           QuantizeFormat `json:"format"`
	SourcePack       Source         `json:"source_pack"`
	Info             Info           `json:"info"`
	TensorCount      int            `json:"tensor_count"`
	QuantizedTensors int            `json:"quantized_tensors"`
	Notes            []string       `json:"notes,omitempty"`
}

// denseSafetensor is one decoded dense (unquantised) tensor read from a
// source safetensors pack, ready for GGUF quantisation.
type denseSafetensor struct {
	Name  string
	Shape []uint64
	Data  []float32
}

// Tensor is one tensor record for WriteFile: the GGUF tensor-directory name,
// GGML tensor-type id, shape, and the already-encoded data bytes to place in
// the file's data section. Offset is assigned by the writer (each tensor's
// data starts at the next 32-byte-aligned offset) — caller-set values are
// overwritten.
//
//	tensor := gguf.Tensor{Name: "blk.0.attn_q.weight", Type: gguf.TensorTypeQ8_0, Shape: []uint64{32}, Data: blocks}
type Tensor struct {
	Name   string
	Type   uint32
	Shape  []uint64
	Offset uint64
	Data   []byte
}

// ggufQuantizedTensor is the historical package-internal name for Tensor,
// kept as an alias so the quantise pipeline's tests read unchanged.
type ggufQuantizedTensor = Tensor

// MetadataEntry is one key/value pair written into a GGUF header by
// WriteFile. ValueType names the wire type — ValueTypeString, ValueTypeUint32
// or ValueTypeFloat32 — and Value must hold the matching Go type (uint32
// entries also accept int).
//
//	entry := gguf.MetadataEntry{Key: "general.architecture", ValueType: gguf.ValueTypeString, Value: "gemma3"}
type MetadataEntry struct {
	Key       string
	ValueType uint32
	Value     any
}

// ggufMetadataEntry is the historical package-internal name for MetadataEntry,
// kept as an alias so the quantise pipeline's tests read unchanged.
type ggufMetadataEntry = MetadataEntry

// QuantizeModelPack converts a dense safetensors model pack into a GGUF pack.
//
// Every source weight file is decoded fully into memory (via
// safetensors.ReadSafetensors) rather than streamed — the same tradeoff the
// merge package already makes for the pack sizes go-inference callers
// quantise today. A chunked/streaming variant bounded to a fixed working
// set for multi-GB sharded checkpoints is future work, not implemented
// here.
func QuantizeModelPack(ctx context.Context, opts QuantizeOptions) (*QuantizeResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if opts.SourcePack.Root == "" {
		return nil, core.NewError("gguf: source pack is required")
	}
	if opts.OutputPath == "" {
		return nil, core.NewError("gguf: GGUF output path is required")
	}
	if core.HasSuffix(core.Lower(opts.OutputPath), ".gguf") || core.HasSuffix(core.Lower(opts.OutputPath), ".safetensors") {
		return nil, core.NewError("gguf: GGUF output path must be a model-pack directory")
	}

	requested, format, notes, err := resolveGGUFQuantizeFormat(opts.Format)
	if err != nil {
		return nil, err
	}

	source := opts.SourcePack
	if len(source.WeightFiles) == 0 {
		return nil, core.NewError("gguf: GGUF quantization requires one or more safetensors source weight files")
	}
	for _, weightFile := range source.WeightFiles {
		if !hasASCIIInsensitiveSuffix(weightFile, ".safetensors") {
			return nil, core.NewError("gguf: GGUF quantization currently requires dense safetensors source weights")
		}
	}

	output := opts.OutputPath
	if abs := core.PathAbs(output); abs.OK {
		output = abs.Value.(string)
	}
	if samePath(source.Root, output) {
		return nil, core.NewError("gguf: GGUF output path must differ from source model path")
	}
	if err := ensureEmptyGGUFQuantizeDestination(output); err != nil {
		return nil, err
	}
	if result := core.MkdirAll(output, 0o755); !result.OK {
		return nil, core.E("QuantizeModelPack", "create output directory", result.Err())
	}
	if err := copyModelPackMetadata(source.Root, output); err != nil {
		return nil, err
	}

	tensors, err := loadDenseSafetensors(source.WeightFiles)
	if err != nil {
		return nil, core.E("QuantizeModelPack", "load dense safetensors", err)
	}

	// gemma-4 checkpoints take a dedicated lane: llama.cpp maps the text stack
	// by canonical names and needs the full gemma4.* hyperparameter set plus the
	// embedded tokenizer, none of which the generic pipeline produces. Detected
	// from config.json's model_type; the lane is calibrated to q4_k_m.
	var quantized []Tensor
	var metadata []MetadataEntry
	if configRead := core.ReadFile(core.PathJoin(source.Root, "config.json")); configRead.OK && isGemma4Config(configRead.Value.([]byte)) {
		if requested != QuantizeQ4_K_M {
			return nil, core.NewError("gguf: gemma4 GGUF conversion currently supports only q4_k_m (requested " + string(requested) + ")")
		}
		quantized, metadata, err = quantizeGemma4ModelPack(source, configRead.Value.([]byte), tensors)
		if err != nil {
			return nil, err
		}
	} else {
		quantized, err = quantizeGGUFTensors(ctx, tensors, format)
		if err != nil {
			return nil, err
		}
		metadata = ggufQuantizeMetadata(source, format, opts.Labels)
	}

	weightPath := core.PathJoin(output, ggufQuantizeOutputWeights)
	if err := writeQuantizedGGUF(weightPath, metadata, quantized); err != nil {
		return nil, core.E("QuantizeModelPack", "write GGUF", err)
	}

	info, err := ReadInfo(weightPath)
	if err != nil {
		return nil, core.E("QuantizeModelPack", "read generated GGUF", err)
	}
	if !info.Valid() {
		return nil, core.NewError("gguf: generated GGUF failed metadata validation: " + ValidationSummary(info.ValidationIssues))
	}

	return &QuantizeResult{
		OutputPath:       output,
		WeightPath:       weightPath,
		RequestedFormat:  requested,
		Format:           format,
		SourcePack:       source,
		Info:             info,
		TensorCount:      len(quantized),
		QuantizedTensors: len(quantized),
		Notes:            notes,
	}, nil
}

func resolveGGUFQuantizeFormat(format QuantizeFormat) (requested, used QuantizeFormat, notes []string, err error) {
	if format == "" {
		format = QuantizeQ8_0
	}
	normalized := QuantizeFormat(NormalizeQuantType(string(format)))
	switch normalized {
	case QuantizeQ8_0:
		return normalized, QuantizeQ8_0, nil, nil
	case QuantizeQ4_0:
		return normalized, QuantizeQ4_0, nil, nil
	case QuantizeQ5_0:
		return normalized, QuantizeQ5_0, nil, nil
	case QuantizeQ4_K_M:
		return normalized, QuantizeQ4_K, nil, nil
	case QuantizeQ4_K:
		return normalized, QuantizeQ4_K, nil, nil
	case QuantizeQ5_K:
		return normalized, QuantizeQ5_K, nil, nil
	case QuantizeQ6_K:
		return normalized, QuantizeQ6_K, nil, nil
	case QuantizeQ8_K:
		return normalized, QuantizeQ8_K, nil, nil
	case QuantizeQ3_K:
		return normalized, QuantizeQ3_K, nil, nil
	case QuantizeQ2_K:
		return normalized, QuantizeQ2_K, nil, nil
	default:
		return normalized, "", nil, core.NewError("gguf: unsupported GGUF quantization format: " + string(format))
	}
}

func ensureEmptyGGUFQuantizeDestination(output string) error {
	if stat := core.Stat(output); !stat.OK {
		if core.IsNotExist(stat.Value.(error)) {
			return nil
		}
		return core.E("QuantizeModelPack", "inspect output path", stat.Err())
	}
	weights := append(core.PathGlob(core.PathJoin(output, "*.safetensors")), core.PathGlob(core.PathJoin(output, "*.gguf"))...)
	if len(weights) > 0 {
		return core.NewError("gguf: GGUF output path already contains model weights")
	}
	return nil
}

// loadDenseSafetensors decodes every tensor in paths (in sorted-name order,
// duplicates across shards rejected) to float32 via safetensors.ReadSafetensors.
func loadDenseSafetensors(paths []string) ([]denseSafetensor, error) {
	if len(paths) == 0 {
		return nil, core.NewError("gguf: no safetensors weight files available")
	}
	var out []denseSafetensor
	seen := map[string]struct{}{}
	for _, path := range paths {
		tensors, err := readDenseSafetensorsFile(path)
		if err != nil {
			return nil, err
		}
		for _, tensor := range tensors {
			if _, ok := seen[tensor.Name]; ok {
				return nil, core.NewError("gguf: duplicate tensor in safetensors shards: " + tensor.Name)
			}
			seen[tensor.Name] = struct{}{}
			out = append(out, tensor)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// readDenseSafetensorsFile decodes every tensor in one safetensors file,
// in sorted tensor-name order.
func readDenseSafetensorsFile(path string) ([]denseSafetensor, error) {
	read := safetensors.ReadSafetensors(path)
	if !read.OK {
		return nil, core.E("QuantizeModelPack", "read safetensors "+path, read.Err())
	}
	data := read.Value.(safetensors.SafetensorsData)

	names := make([]string, 0, len(data.Tensors))
	for name := range data.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	tensors := make([]denseSafetensor, 0, len(names))
	for _, name := range names {
		info := data.Tensors[name]
		raw := safetensors.GetTensorData(info, data.Data)
		values, err := safetensors.DecodeFloat32(info.Dtype, raw, safetensorsShapeElements(info.Shape))
		if err != nil {
			return nil, core.E("QuantizeModelPack", "decode "+path+" tensor "+name, err)
		}
		shape := make([]uint64, len(info.Shape))
		for i, dim := range info.Shape {
			shape[i] = uint64(dim)
		}
		tensors = append(tensors, denseSafetensor{Name: name, Shape: shape, Data: values})
	}
	return tensors, nil
}

// safetensorsShapeElements returns the element count a safetensors []int
// shape describes (the product of its dimensions; 1 for a scalar's empty
// shape).
func safetensorsShapeElements(shape []int) int {
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	return n
}

func quantizeGGUFTensors(ctx context.Context, tensors []denseSafetensor, format QuantizeFormat) ([]Tensor, error) {
	out := make([]Tensor, 0, len(tensors))
	for _, tensor := range tensors {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if isMultimodalTowerTensor(tensor.Name) {
			// A text GGUF carries text tensors only — llama.cpp maps the text
			// stack by name and rejects unknown tensors; multimodal towers ship
			// separately in that ecosystem (the mmproj convention). Matches the
			// text-only GGUFs the community publishes for these checkpoints.
			continue
		}
		quantized, err := quantizeGGUFTensor(tensor, format)
		if err != nil {
			return nil, err
		}
		out = append(out, quantized)
	}
	if len(out) == 0 {
		return nil, core.NewError("gguf: no text-stack tensors found in source")
	}
	return out, nil
}

// isMultimodalTowerTensor reports whether name belongs to a non-text tower a
// text GGUF must exclude (audio/vision encoders and their embedding bridges).
func isMultimodalTowerTensor(name string) bool {
	for _, prefix := range []string{"audio_tower.", "vision_tower.", "embed_audio.", "embed_vision.", "multi_modal_projector."} {
		if core.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

func quantizeGGUFTensor(tensor denseSafetensor, format QuantizeFormat) (Tensor, error) {
	tensorType, blockSize, _, err := ggufQuantizeLayout(format)
	if err != nil {
		return Tensor{}, err
	}
	if len(tensor.Data)%blockSize != 0 || len(tensor.Shape) == 0 || tensor.Shape[0]%uint64(blockSize) != 0 {
		// Block-incompatible tensors (scalar clips, tiny biases, odd norms)
		// store as raw F32 — llama.cpp's own quantizer does the same rather
		// than failing the model; readers dispatch per-tensor on Type.
		raw := make([]byte, len(tensor.Data)*4)
		for i, v := range tensor.Data {
			bits := math.Float32bits(v)
			raw[4*i], raw[4*i+1], raw[4*i+2], raw[4*i+3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
		}
		return Tensor{
			Name:  tensor.Name,
			Type:  ggufTensorTypeF32,
			Shape: core.SliceClone(tensor.Shape),
			Data:  raw,
		}, nil
	}
	var data []byte
	switch format {
	case QuantizeQ8_0:
		data = quantizeQ8_0(tensor.Data)
	case QuantizeQ4_0:
		data = quantizeQ4_0(tensor.Data)
	case QuantizeQ5_0:
		data = quantizeQ5_0(tensor.Data)
	case QuantizeQ4_K:
		data = quantizeQ4_K(tensor.Data)
	case QuantizeQ5_K:
		data = quantizeQ5_K(tensor.Data)
	case QuantizeQ6_K:
		data = quantizeQ6_K(tensor.Data)
	case QuantizeQ8_K:
		data = quantizeQ8_K(tensor.Data)
	case QuantizeQ3_K:
		data = quantizeQ3_K(tensor.Data)
	case QuantizeQ2_K:
		data = quantizeQ2_K(tensor.Data)
	}
	return Tensor{
		Name:  tensor.Name,
		Type:  tensorType,
		Shape: core.SliceClone(tensor.Shape),
		Data:  data,
	}, nil
}

func ggufQuantizeLayout(format QuantizeFormat) (tensorType uint32, blockSize int, bytesPerBlock int, err error) {
	switch format {
	case QuantizeQ8_0:
		return TensorTypeQ8_0, 32, 34, nil
	case QuantizeQ4_0:
		return TensorTypeQ4_0, 32, 18, nil
	case QuantizeQ5_0:
		return ggufTensorTypeQ5_0, 32, 24, nil
	case QuantizeQ4_K:
		return ggufTensorTypeQ4K, 256, 144, nil
	case QuantizeQ5_K:
		return ggufTensorTypeQ5K, 256, 176, nil
	case QuantizeQ6_K:
		return ggufTensorTypeQ6K, 256, 210, nil
	case QuantizeQ8_K:
		// Canonical block_q8_K: float32 d + 256 int8 qs + 16 int16 bsums.
		return ggufTensorTypeQ8K, 256, 292, nil
	case QuantizeQ3_K:
		return ggufTensorTypeQ3K, 256, 110, nil
	case QuantizeQ2_K:
		// Canonical block_q2_K is 84 (16 scales + 64 qs + f16 d + f16
		// dmin). The gguflib type-size table's 82 drops dmin; its decoder
		// nonetheless advances 84, and upstream static_assert is 84.
		return ggufTensorTypeQ2K, 256, 84, nil
	default:
		return 0, 0, 0, core.NewError("gguf: unsupported resolved GGUF format: " + string(format))
	}
}

// ValidationSummary joins GGUF validation issue codes into a human-readable
// string. Used by callers that report failures from the gguf validation path.
//
//	msg := gguf.ValidationSummary(info.ValidationIssues)
func ValidationSummary(issues []ValidationIssue) string {
	if len(issues) == 0 {
		return "unknown validation failure"
	}
	parts := make([]string, 0, len(issues))
	for _, issue := range issues {
		if issue.Tensor != "" {
			parts = append(parts, core.Concat(issue.Code, ":", issue.Tensor))
			continue
		}
		parts = append(parts, issue.Code)
	}
	return core.Join(", ", parts...)
}

func samePath(a, b string) bool {
	absA := a
	if resolved := core.PathAbs(a); resolved.OK {
		absA = resolved.Value.(string)
	}
	absB := b
	if resolved := core.PathAbs(b); resolved.OK {
		absB = resolved.Value.(string)
	}
	return absA == absB
}

func copyModelPackMetadata(sourceRoot, outputRoot string) error {
	patterns := []string{"*.json", "*.model", "*.txt"}
	seen := map[string]struct{}{}
	for _, pattern := range patterns {
		for _, sourcePath := range core.PathGlob(core.PathJoin(sourceRoot, pattern)) {
			name := core.PathBase(sourcePath)
			if _, ok := seen[name]; ok {
				continue
			}
			seen[name] = struct{}{}
			if isModelWeightMetadataCopySkip(name) {
				continue
			}
			if err := copyLocalFile(sourcePath, core.PathJoin(outputRoot, name)); err != nil {
				return err
			}
		}
	}
	return nil
}

func isModelWeightMetadataCopySkip(name string) bool {
	lower := core.Lower(name)
	return lower == "adapter_provenance.json" ||
		core.Contains(lower, ".safetensors") ||
		core.Contains(lower, ".gguf") ||
		core.HasSuffix(lower, ".safetensors") ||
		core.HasSuffix(lower, ".gguf")
}

// metadataCopyStreamThreshold is the file size at or below which copyLocalFile
// reads the whole file into one buffer and writes it back (core.ReadFile +
// core.WriteFile), and above which it streams source→destination through
// core.Copy's fixed staging buffer. Below ~128 KiB a single read/write is the
// cheaper path — the slurp buffer is small and a dedicated copy buffer would
// cost more than the read it replaces. Above it the slurp is a large
// transient buffer the size of the whole file (tokenizer.json is multiple MB
// on real checkpoints), so streaming wins on B/op without changing a copied
// byte.
const metadataCopyStreamThreshold = 128 << 10

func copyLocalFile(sourcePath, destinationPath string) error {
	// Size-gate: small files take the direct read/write (byte- and
	// mode-identical to the historical core.ReadFile + core.WriteFile);
	// large files stream. A failed/absent stat falls through to the direct
	// read, whose own failure surfaces the real error — never silently skip.
	if stat := core.Stat(sourcePath); stat.OK {
		if info, ok := stat.Value.(core.FsFileInfo); ok && info.Size() > metadataCopyStreamThreshold {
			return streamLocalFile(sourcePath, destinationPath)
		}
	}
	read := core.ReadFile(sourcePath)
	if !read.OK {
		return read.Err()
	}
	if result := core.WriteFile(destinationPath, read.Value.([]byte), 0o644); !result.OK {
		return result.Err()
	}
	return nil
}

// streamLocalFile copies source→destination through core.Copy (io.Copy's
// fixed ~32 KiB staging buffer, or the kernel copy fast-path between two
// *os.File handles) instead of slurping the whole file into a heap []byte.
// The destination is opened with the same O_WRONLY|O_CREATE|O_TRUNC flags and
// 0o644 mode core.WriteFile used, so the written bytes and file mode are
// identical to the direct path.
func streamLocalFile(sourcePath, destinationPath string) error {
	srcOpen := core.Open(sourcePath)
	if !srcOpen.OK {
		return srcOpen.Err()
	}
	src := srcOpen.Value.(*core.OSFile)
	defer src.Close()
	dstOpen := core.OpenFile(destinationPath, core.O_WRONLY|core.O_CREATE|core.O_TRUNC, 0o644)
	if !dstOpen.OK {
		return dstOpen.Err()
	}
	dst := dstOpen.Value.(*core.OSFile)
	if result := core.Copy(dst, src); !result.OK {
		// The copy already failed; close the partial destination best-effort
		// and surface the copy error, not the close error.
		dst.Close()
		return result.Err()
	}
	if err := dst.Close(); err != nil {
		return err
	}
	return nil
}
