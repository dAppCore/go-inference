// SPDX-Licence-Identifier: EUPL-1.2

package awq

import (
	"context"
	"encoding/binary"
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

type Result struct {
	OutputDir        string
	WeightFile       string
	ConfigFile       string
	TensorCount      int
	QuantizedWeights int
	PassthroughCount int
	SourceBytes      int64
	OutputBytes      int64
}

type component struct {
	name, dtype string
	shape       []uint64
	start, size int64
}

type item struct {
	ref        safetensors.TensorRef
	quantize   bool
	rows       int
	columns    int
	components []component
}

type headerEntry struct {
	DType       string   `json:"dtype"`
	Shape       []uint64 `json:"shape"`
	DataOffsets []int64  `json:"data_offsets"`
}

func ConvertSnapshot(ctx context.Context, srcDir, outDir string, opts Options, progress func(string, bool, int, int)) (*Result, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if core.Trim(srcDir) == "" || core.Trim(outDir) == "" {
		return nil, core.NewError("awq: source and output directories are required")
	}
	if samePath(srcDir, outDir) {
		return nil, core.NewError("awq: output directory must differ from source")
	}
	opts = normaliseOptions(opts)
	if opts.Bits != 2 && opts.Bits != 4 && opts.Bits != 8 {
		return nil, core.NewError("awq: bits must be 2, 4, or 8")
	}
	if opts.GroupSize != 32 && opts.GroupSize != 64 && opts.GroupSize != 128 && opts.GroupSize != 256 {
		return nil, core.NewError("awq: group size must be 32, 64, 128, or 256")
	}
	shards := core.PathGlob(core.PathJoin(srcDir, "*.safetensors"))
	core.SliceSort(shards)
	if len(shards) == 0 {
		return nil, core.NewError("awq: source has no safetensors shards")
	}
	idx, err := safetensors.IndexFiles(shards)
	if err != nil {
		return nil, core.E("awq.ConvertSnapshot", "index source shards", err)
	}
	items, result, err := plan(idx, opts)
	if err != nil {
		return nil, err
	}
	if r := core.MkdirAll(outDir, 0o755); !r.OK {
		return nil, core.E("awq.ConvertSnapshot", "create output directory", r.Err())
	}
	result.OutputDir = outDir
	result.WeightFile = core.PathJoin(outDir, "model.safetensors")
	result.ConfigFile = core.PathJoin(outDir, "quantize_config.json")
	if err := writeSnapshot(ctx, result.WeightFile, items, opts, progress); err != nil {
		return nil, err
	}
	if err := writeQuantizeConfig(result.ConfigFile, opts); err != nil {
		return nil, err
	}
	if err := copyFiles(srcDir, outDir); err != nil {
		return nil, err
	}
	return result, nil
}

func plan(idx safetensors.Index, opts Options) ([]item, *Result, error) {
	names := core.SliceClone(idx.Names)
	core.SliceSort(names)
	pack := 32 / opts.Bits
	var offset int64
	result := &Result{}
	items := make([]item, 0, len(names))
	for _, name := range names {
		ref := idx.Tensors[name]
		result.SourceBytes += ref.ByteLen
		eligible := core.HasSuffix(name, ".weight") && len(ref.Shape) == 2 && isFloat(ref.DType) && isAWQLinearWeight(name)
		if eligible {
			rows, columns := int(ref.Shape[0]), int(ref.Shape[1])
			eligible = rows%pack == 0 && columns%pack == 0 && columns%opts.GroupSize == 0
			if eligible {
				base := core.TrimSuffix(name, ".weight")
				groups := columns / opts.GroupSize
				components := []component{
					{name: base + ".qweight", dtype: "I32", shape: []uint64{uint64(columns), uint64(rows / pack)}, size: int64(columns * rows / pack * 4)},
					{name: base + ".qzeros", dtype: "I32", shape: []uint64{uint64(groups), uint64(rows / pack)}, size: int64(groups * rows / pack * 4)},
					{name: base + ".scales", dtype: "F16", shape: []uint64{uint64(groups), uint64(rows)}, size: int64(groups * rows * 2)},
				}
				for i := range components {
					components[i].start = offset
					offset += components[i].size
					result.OutputBytes += components[i].size
				}
				items = append(items, item{ref: ref, quantize: true, rows: rows, columns: columns, components: components})
				result.QuantizedWeights++
				result.TensorCount += 3
				continue
			}
		}
		passthrough := component{name: name, dtype: ref.DType, shape: ref.Shape, start: offset, size: ref.ByteLen}
		offset += ref.ByteLen
		result.OutputBytes += ref.ByteLen
		items = append(items, item{ref: ref, components: []component{passthrough}})
		result.PassthroughCount++
		result.TensorCount++
	}
	return items, result, nil
}

func isAWQLinearWeight(name string) bool {
	lower := core.Lower(name)
	return !core.Contains(lower, "embed") && !core.Contains(lower, "lm_head")
}

func writeSnapshot(ctx context.Context, path string, items []item, opts Options, progress func(string, bool, int, int)) error {
	header := make(map[string]headerEntry)
	for _, item := range items {
		for _, component := range item.components {
			header[component.name] = headerEntry{DType: component.dtype, Shape: component.shape, DataOffsets: []int64{component.start, component.start + component.size}}
		}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		return core.E("awq.write", "encode safetensors header", encoded.Err())
	}
	created := core.OpenFile(path, core.O_CREATE|core.O_WRONLY|core.O_TRUNC, 0o644)
	if !created.OK {
		return core.E("awq.write", "create safetensors", created.Err())
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()
	var length [8]byte
	headerBytes := encoded.Value.([]byte)
	binary.LittleEndian.PutUint64(length[:], uint64(len(headerBytes)))
	if err := writeAll(file, length[:]); err != nil {
		return err
	}
	if err := writeAll(file, headerBytes); err != nil {
		return err
	}
	cache := safetensors.NewShardCache()
	defer cache.Close()
	for index, item := range items {
		if err := ctx.Err(); err != nil {
			return err
		}
		if progress != nil {
			progress(item.ref.Name, item.quantize, index+1, len(items))
		}
		raw, err := cache.ReadRefRaw(item.ref)
		if err != nil {
			return core.E("awq.write", "read "+item.ref.Name, err)
		}
		if !item.quantize {
			if err := writeAll(file, raw); err != nil {
				return err
			}
			continue
		}
		values, err := safetensors.DecodeFloat32(item.ref.DType, raw, item.rows*item.columns)
		if err != nil {
			return core.E("awq.write", "decode "+item.ref.Name, err)
		}
		tensor, err := Quantize(values, item.rows, item.columns, opts)
		if err != nil {
			return core.E("awq.write", "quantise "+item.ref.Name, err)
		}
		for _, payload := range [][]byte{encodeU32(tensor.QWeight), encodeU32(tensor.QZeros), encodeF16(tensor.Scales)} {
			if err := writeAll(file, payload); err != nil {
				return err
			}
		}
	}
	return nil
}

func writeQuantizeConfig(path string, opts Options) error {
	config := struct {
		Bits             int    `json:"w_bit"`
		GroupSize        int    `json:"q_group_size"`
		ZeroPoint        bool   `json:"zero_point"`
		QuantMethod      string `json:"quant_method"`
		CheckpointFormat string `json:"checkpoint_format"`
		Version          string `json:"version"`
		ModulesToSkip    any    `json:"modules_to_not_convert"`
		Calibration      string `json:"calibration"`
		DataFree         bool   `json:"data_free"`
		Approximation    string `json:"approximation"`
	}{opts.Bits, opts.GroupSize, opts.ZeroPoint, "awq", "awq", "GEMM", nil, "none", true, "weight-only zero-point quantisation; no activation calibration"}
	encoded := core.JSONMarshalIndent(config, "", "  ")
	if !encoded.OK {
		return core.E("awq.config", "encode quantize_config.json", encoded.Err())
	}
	if result := core.WriteFile(path, encoded.Value.([]byte), 0o644); !result.OK {
		return core.E("awq.config", "write quantize_config.json", result.Err())
	}
	return nil
}

func copyFiles(srcDir, outDir string) error {
	for _, src := range core.PathGlob(core.PathJoin(srcDir, "*")) {
		name := core.PathBase(src)
		if core.HasSuffix(name, ".safetensors") || core.HasSuffix(name, ".safetensors.index.json") || name == "quantize_config.json" {
			continue
		}
		read := core.ReadFile(src)
		if !read.OK {
			continue
		}
		if write := core.WriteFile(core.PathJoin(outDir, name), read.Value.([]byte), 0o644); !write.OK {
			return core.E("awq.sidecar", "copy "+name, write.Err())
		}
	}
	return nil
}

func encodeU32(values []uint32) []byte {
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], value)
	}
	return raw
}

func encodeI32(values []int32) []byte {
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], uint32(value))
	}
	return raw
}

func encodeF16(values []float32) []byte {
	raw := make([]byte, len(values)*2)
	for i, value := range values {
		binary.LittleEndian.PutUint16(raw[i*2:], float32ToFloat16(value))
	}
	return raw
}

func float32ToFloat16(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16(bits >> 16 & 0x8000)
	exponent := int(bits>>23&0xff) - 127 + 15
	mantissa := bits & 0x7fffff
	if exponent <= 0 {
		if exponent < -10 {
			return sign
		}
		mantissa = (mantissa | 0x800000) >> uint(1-exponent)
		return sign | uint16((mantissa+0x1000)>>13)
	}
	if exponent >= 31 {
		return sign | 0x7c00
	}
	return sign | uint16(exponent<<10) | uint16((mantissa+0x1000)>>13)
}

func isFloat(dtype string) bool {
	switch core.Upper(dtype) {
	case "BF16", "F16", "F32":
		return true
	default:
		return false
	}
}

func samePath(left, right string) bool {
	a, b := core.PathAbs(left), core.PathAbs(right)
	return a.OK && b.OK && a.Value.(string) == b.Value.(string)
}

func writeAll(file *core.OSFile, data []byte) error {
	for len(data) > 0 {
		n, err := file.Write(data)
		if err != nil {
			return err
		}
		if n == 0 {
			return core.NewError("awq: write made no progress")
		}
		data = data[n:]
	}
	return nil
}
