// SPDX-Licence-Identifier: EUPL-1.2

package mlxaffine

import (
	"context"
	"encoding/binary"

	core "dappco.re/go"
	coreio "dappco.re/go/io"

	"dappco.re/go/inference/model/safetensors"
)

// Options configures a snapshot conversion: the affine bit-width and group size the
// eligible weights are quantised at. They land verbatim in the output config.json's
// quantization block, so the engine loads the result with the same geometry.
type Options struct {
	Bits      int
	GroupSize int
}

// Result summarises a completed conversion — enough for a CLI to print an honest
// before→after line without re-scanning the directories.
type Result struct {
	OutputDir        string
	WeightFile       string
	TensorCount      int   // output tensors written (a quantised weight counts its 3 siblings)
	QuantizedWeights int   // source weights that were quantised
	PassthroughCount int   // source tensors copied wide (norms, non-group-aligned)
	SourceBytes      int64 // total source tensor payload bytes
	OutputBytes      int64 // total output tensor payload bytes
}

// snapshotMeta is the safetensors __metadata__ value MLX writes; harmless to the
// reader (skipped by name) and kept for faithfulness to mlx_lm.convert output.
const snapshotMeta = `{"format":"mlx"}`

// weightSuffix is the tensor-name tail a quantisable weight carries; the group
// parameters replace it with .scales / .biases.
const weightSuffix = ".weight"

// hipWideBF16Suffixes names the weight tensors a downstream engine requires to stay
// wide (its source dtype, BF16 for the bf16 packs this lane targets) rather than be
// group-affine quantised — even though their shape is otherwise eligible.
//
// per_layer_model_projection is the Gemma4/Gemma-3n per-layer-input model projection.
// The HIP engine loads it through loadedGemma4BF16ProjectionConfig
// (engine/hip/hip_gemma4_q4_layer.go), which validates dtype == BF16 and a rank-2
// [hiddenPerLayer, hidden] shape; a U32-packed tensor there fails the load. Keeping it
// wide here is the quant-side half of that contract, so `lem quant -bits 4` output loads
// on HIP with no per-tensor workaround. The suffix is matched (not the full name) to
// stay robust to the top-level prefix while never colliding with the per-layer
// per_layer_projection.weight, which is quantised.
var hipWideBF16Suffixes = []string{
	".per_layer_model_projection.weight",
}

// quantiseSkipped reports whether a weight must be passed through wide (kept at its
// source dtype, never quantised) to satisfy a downstream engine's dtype contract —
// see hipWideBF16Suffixes.
func quantiseSkipped(name string) bool {
	for _, suffix := range hipWideBF16Suffixes {
		if core.HasSuffix(name, suffix) {
			return true
		}
	}
	return false
}

// ConvertSnapshot reads an MLX-format bf16 model directory and writes a new directory
// with every eligible weight group-affine quantised (packed uint32 + bf16 .scales /
// .biases), every ineligible tensor (norms, non-group-aligned matrices) passed through
// wide, config.json updated with the quantization block, and the tokenizer / template
// sidecar files copied. The output is the native format the engine loads.
//
// It streams one source tensor at a time — the header is planned from shapes alone (no
// tensor data), then payloads are written in one pass — so peak memory is bounded by
// the largest single tensor, never the whole model.
func ConvertSnapshot(ctx context.Context, srcDir, outDir string, opts Options, progress func(name string, quantised bool, index, total int)) (*Result, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !SupportedBits(opts.Bits) {
		return nil, core.Errorf("mlxaffine: unsupported bits %d (want 2, 4, or 8)", opts.Bits)
	}
	if opts.GroupSize <= 0 {
		return nil, core.NewError("mlxaffine: group size must be positive")
	}
	if core.Trim(srcDir) == "" || core.Trim(outDir) == "" {
		return nil, core.NewError("mlxaffine: source and output directories are required")
	}
	if samePath(srcDir, outDir) {
		return nil, core.NewError("mlxaffine: output directory must differ from the source")
	}

	shards := core.PathGlob(core.PathJoin(srcDir, "*.safetensors"))
	core.SliceSort(shards)
	if len(shards) == 0 {
		return nil, core.Errorf("mlxaffine: no *.safetensors shards in %s", srcDir)
	}
	idx, err := safetensors.IndexFiles(shards)
	if err != nil {
		return nil, core.E("mlxaffine.ConvertSnapshot", "index source shards", err)
	}

	items, res, err := planItems(idx, opts)
	if err != nil {
		return nil, err
	}

	if r := core.MkdirAll(outDir, 0o755); !r.OK {
		return nil, core.E("mlxaffine.ConvertSnapshot", "create output directory", r.Err())
	}
	weightPath := core.PathJoin(outDir, "model.safetensors")
	if err := writeQuantizedSafetensors(ctx, weightPath, items, opts, progress); err != nil {
		return nil, err
	}
	if err := writeConfig(srcDir, outDir, opts); err != nil {
		return nil, err
	}
	if err := copySidecars(srcDir, outDir); err != nil {
		return nil, err
	}

	res.OutputDir = outDir
	res.WeightFile = weightPath
	return res, nil
}

// planComp is one output tensor's header record.
type planComp struct {
	name    string
	dtype   string
	shape   []uint64
	byteLen int64
	start   int64 // data-section offset, assigned during planning
}

// planItem is one source tensor's contribution: either a passthrough (one comp copied
// from src) or a quantised weight (three comps derived from the bf16 source).
type planItem struct {
	src      safetensors.TensorRef
	quantise bool
	outDim   int
	inDim    int
	comps    []planComp
}

// planItems walks the source index in sorted-name order and builds the output plan +
// running byte-length totals. Offsets are assigned contiguously in emission order, so
// the data section is written in a single forward pass.
func planItems(idx safetensors.Index, opts Options) ([]planItem, *Result, error) {
	names := append([]string(nil), idx.Names...)
	core.SliceSort(names)

	items := make([]planItem, 0, len(names))
	res := &Result{}
	var offset int64
	for _, name := range names {
		ref := idx.Tensors[name]
		res.SourceBytes += ref.ByteLen

		if core.HasSuffix(name, weightSuffix) && !quantiseSkipped(name) && EligibleShape(ref.Shape, opts.GroupSize) && isFloatDType(ref.DType) {
			outDim, inDim := int(ref.Shape[0]), int(ref.Shape[1])
			base := name[:len(name)-len(weightSuffix)]
			groups := inDim / opts.GroupSize
			comps := []planComp{
				{name: name, dtype: "U32", shape: []uint64{uint64(outDim), uint64(PackedWords(inDim, opts.Bits))}, byteLen: int64(outDim * PackedWords(inDim, opts.Bits) * 4)},
				{name: base + ".scales", dtype: "BF16", shape: []uint64{uint64(outDim), uint64(groups)}, byteLen: int64(outDim * groups * 2)},
				{name: base + ".biases", dtype: "BF16", shape: []uint64{uint64(outDim), uint64(groups)}, byteLen: int64(outDim * groups * 2)},
			}
			for i := range comps {
				comps[i].start = offset
				offset += comps[i].byteLen
				res.OutputBytes += comps[i].byteLen
			}
			items = append(items, planItem{src: ref, quantise: true, outDim: outDim, inDim: inDim, comps: comps})
			res.QuantizedWeights++
			res.TensorCount += 3
			continue
		}

		comp := planComp{name: name, dtype: ref.DType, shape: ref.Shape, byteLen: ref.ByteLen, start: offset}
		offset += ref.ByteLen
		res.OutputBytes += ref.ByteLen
		items = append(items, planItem{src: ref, comps: []planComp{comp}})
		res.PassthroughCount++
		res.TensorCount++
	}
	return items, res, nil
}

// writeQuantizedSafetensors emits the safetensors file: the 8-byte header length, the
// JSON header (every comp with its [start,end] offsets, plus __metadata__), then the
// payloads in item order — quantising each eligible weight once as it is written.
func writeQuantizedSafetensors(ctx context.Context, path string, items []planItem, opts Options, progress func(string, bool, int, int)) error {
	header := buildHeader(items)
	created := core.OpenFile(path, core.O_CREATE|core.O_WRONLY|core.O_TRUNC, 0o644)
	if !created.OK {
		return core.E("mlxaffine.write", "create weight file", created.Err())
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(header)))
	if err := writeAll(file, lenBuf[:]); err != nil {
		return err
	}
	if err := writeAll(file, header); err != nil {
		return err
	}

	src := safetensors.NewShardCache()
	defer src.Close()
	total := len(items)
	for i, item := range items {
		if err := ctx.Err(); err != nil {
			return err
		}
		if progress != nil {
			progress(item.comps[0].name, item.quantise, i+1, total)
		}
		if !item.quantise {
			raw, err := src.ReadRefRaw(item.src)
			if err != nil {
				return core.E("mlxaffine.write", "read passthrough "+item.src.Name, err)
			}
			if err := writeAll(file, raw); err != nil {
				return err
			}
			continue
		}
		raw, err := src.ReadRefRaw(item.src)
		if err != nil {
			return core.E("mlxaffine.write", "read weight "+item.src.Name, err)
		}
		values, err := safetensors.DecodeFloat32(item.src.DType, raw, item.outDim*item.inDim)
		if err != nil {
			return core.E("mlxaffine.write", "decode "+item.src.Name, err)
		}
		packed, scales, biases, err := QuantizeTensor(values, item.outDim, item.inDim, opts.Bits, opts.GroupSize)
		if err != nil {
			return core.E("mlxaffine.write", "quantise "+item.src.Name, err)
		}
		if err := writeAll(file, packed); err != nil {
			return err
		}
		if err := writeAll(file, scales); err != nil {
			return err
		}
		if err := writeAll(file, biases); err != nil {
			return err
		}
	}
	return nil
}

// buildHeader emits the safetensors JSON header for every planned comp in offset order,
// closing with the MLX __metadata__ marker.
func buildHeader(items []planItem) []byte {
	out := make([]byte, 0, 256)
	out = append(out, '{')
	for _, item := range items {
		for _, c := range item.comps {
			out = appendJSONString(out, c.name)
			out = append(out, ':', '{')
			out = appendJSONString(out, "dtype")
			out = append(out, ':')
			out = appendJSONString(out, c.dtype)
			out = append(out, ',')
			out = appendJSONString(out, "shape")
			out = append(out, ':', '[')
			for j, d := range c.shape {
				if j > 0 {
					out = append(out, ',')
				}
				out = appendInt(out, int64(d))
			}
			out = append(out, ']', ',')
			out = appendJSONString(out, "data_offsets")
			out = append(out, ':', '[')
			out = appendInt(out, c.start)
			out = append(out, ',')
			out = appendInt(out, c.start+c.byteLen)
			out = append(out, ']', '}', ',')
		}
	}
	out = appendJSONString(out, "__metadata__")
	out = append(out, ':')
	out = append(out, snapshotMeta...)
	out = append(out, '}')
	return out
}

// writeConfig writes the output config.json: the source config with the quantization
// block injected (both the "quantization" key the engine reads and the
// "quantization_config" alias mlx_lm.convert also writes).
func writeConfig(srcDir, outDir string, opts Options) error {
	cfgStr, err := coreio.Local.Read(core.PathJoin(srcDir, "config.json"))
	if err != nil {
		return core.E("mlxaffine.config", "read source config.json", err)
	}
	var cfg map[string]any
	if r := core.JSONUnmarshal([]byte(cfgStr), &cfg); !r.OK {
		return core.E("mlxaffine.config", "parse source config.json", r.Err())
	}
	block := map[string]any{"group_size": opts.GroupSize, "bits": opts.Bits, "mode": Mode}
	cfg["quantization"] = block
	cfg["quantization_config"] = block
	marshalled := core.JSONMarshalIndent(cfg, "", "  ")
	if !marshalled.OK {
		return core.E("mlxaffine.config", "marshal config.json", marshalled.Err())
	}
	if r := core.WriteFile(core.PathJoin(outDir, "config.json"), marshalled.Bytes(), 0o644); !r.OK {
		return core.E("mlxaffine.config", "write config.json", r.Err())
	}
	return nil
}

// copySidecars copies every non-hidden source file that is not config.json (rewritten),
// a safetensors shard, or a shard index (both regenerated) — tokenizer, chat template,
// generation/processor config, README — verbatim into the output directory.
func copySidecars(srcDir, outDir string) error {
	for _, srcPath := range core.PathGlob(core.PathJoin(srcDir, "*")) {
		name := core.PathBase(srcPath)
		if name == "config.json" || core.HasSuffix(name, ".safetensors") || core.HasSuffix(name, ".safetensors.index.json") {
			continue
		}
		read := core.ReadFile(srcPath)
		if !read.OK {
			continue // a directory or unreadable entry — skip, not fatal
		}
		if r := core.WriteFile(core.PathJoin(outDir, name), read.Bytes(), 0o644); !r.OK {
			return core.E("mlxaffine.sidecar", "copy "+name, r.Err())
		}
	}
	return nil
}

// isFloatDType reports whether a dtype names one of the dense float weight formats a
// bf16 checkpoint stores — the only tensors group-affine quantisation applies to.
func isFloatDType(dtype string) bool {
	switch core.Upper(dtype) {
	case "BF16", "F16", "F32":
		return true
	default:
		return false
	}
}

// samePath reports whether two directory paths resolve to the same location, tolerating
// trailing slashes and relative spellings via PathAbs.
func samePath(a, b string) bool {
	ra, rb := core.PathAbs(a), core.PathAbs(b)
	if ra.OK && rb.OK {
		return ra.String() == rb.String()
	}
	return a == b
}

// writeAll writes the whole buffer, looping over short writes.
func writeAll(file *core.OSFile, data []byte) error {
	for len(data) > 0 {
		n, err := file.Write(data)
		if err != nil {
			return err
		}
		if n == 0 {
			return core.NewError("mlxaffine: write made no progress")
		}
		data = data[n:]
	}
	return nil
}

// appendJSONString appends a JSON-quoted string. Tensor names and dtype tokens carry no
// characters needing escape, but backslash / quote / control bytes are handled for
// safety and RFC-8259 correctness.
func appendJSONString(dst []byte, s string) []byte {
	dst = append(dst, '"')
	start := 0
	for i := 0; i < len(s); i++ {
		if c := s[i]; c == '"' || c == '\\' || c < 0x20 {
			if start < i {
				dst = append(dst, s[start:i]...)
			}
			switch c {
			case '"':
				dst = append(dst, '\\', '"')
			case '\\':
				dst = append(dst, '\\', '\\')
			case '\n':
				dst = append(dst, '\\', 'n')
			case '\r':
				dst = append(dst, '\\', 'r')
			case '\t':
				dst = append(dst, '\\', 't')
			default:
				const hex = "0123456789abcdef"
				dst = append(dst, '\\', 'u', '0', '0', hex[c>>4], hex[c&0xf])
			}
			start = i + 1
		}
	}
	if start < len(s) {
		dst = append(dst, s[start:]...)
	}
	return append(dst, '"')
}

// appendInt appends v in base-10 with no allocation.
func appendInt(dst []byte, v int64) []byte {
	if v == 0 {
		return append(dst, '0')
	}
	var buf [20]byte
	i := len(buf)
	neg := v < 0
	uv := uint64(v)
	if neg {
		uv = uint64(-v)
	}
	for uv > 0 {
		i--
		buf[i] = byte('0' + uv%10)
		uv /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return append(dst, buf[i:]...)
}
