// SPDX-Licence-Identifier: EUPL-1.2

// loader.go is the byte-native whole-file loader half of this package (moved from
// go-mlx pkg/safetensors during the engine merge): name → raw bytes via Load / LoadDir
// / LoadMmap / LoadDirMmap, plus the page-aligned Mapping the zero-copy GPU path binds.
// It sits on top of the low-level ref reader (safetensors.go); byte-native engines
// (native/metal, go-rocm) share it because it never materialises into a cgo array.
package safetensors

import (
	"sort"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// Tensor is one safetensors entry: its dtype name (e.g. "BF16", "F32", "U8"), shape, and
// raw little-endian bytes — Data sub-slices the source blob (no copy).
type Tensor struct {
	Dtype string
	Shape []int
	Data  []byte
}

// Mapping is a memory-mapped safetensors file (see LoadMmap). Data is the WHOLE file mapped
// page-aligned, and every entry in Tensors is a view into it (no heap copy of the weights).
// The page-aligned base is what makes the zero-copy GPU path possible: a backend wraps Data
// in ONE no-copy buffer (Metal's bytesNoCopy needs page alignment) and binds each tensor at
// its byte offset into Data — so a multi-GB checkpoint is never duplicated in heap or GPU
// memory. Close unmaps; the Mapping MUST outlive every Tensor view and every GPU buffer
// taken over Data.
type Mapping struct {
	Data    []byte // the whole file, mapped page-aligned (the no-copy buffer's backing)
	Tensors map[string]Tensor
}

// headerEntry is one tensor's header record, decoded straight from the JSON header in Parse.
// Decoding into this typed struct instead of map[string]any avoids boxing every dtype/shape/
// offset value onto the heap. Shape and DataOffsets are []int (not [2]int) so Parse can keep
// rejecting a data_offsets array whose length isn't exactly 2 (json silently drops trailing
// elements when decoding into a fixed-size array). Unknown header keys (e.g. __metadata__'s
// {"format":"pt"}) decode into a zero headerEntry and are skipped by name.
type headerEntry struct {
	Dtype       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets []int  `json:"data_offsets"`
}

// dtypeBytes is the element byte size of the safetensors dtypes gemma4 checkpoints use:
// bf16/f32 weights, and the 4-bit-quant companions (u8/u32 packed codes + bf16 scales).
var dtypeBytes = map[string]int{
	"BF16": 2, "F16": 2, "F32": 4, "F64": 8,
	"I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8, "BOOL": 1,
}

// Parse reads a safetensors blob: an 8-byte little-endian header length, then that many
// bytes of JSON ({name:{dtype,shape,data_offsets:[start,end]}, optional "__metadata__"}),
// then the tensor data. data_offsets are relative to the END of the header. Returns
// name→Tensor with Data sub-slicing blob (no copy). Validates the header length, each
// entry's dtype/shape/offsets, and that the byte span equals dtype × ∏shape.
func Parse(blob []byte) (map[string]Tensor, error) {
	if len(blob) < 8 {
		return nil, core.NewError("safetensors.Parse: blob shorter than the 8-byte header length")
	}
	var hdrLen uint64
	for i := range 8 {
		hdrLen |= uint64(blob[i]) << (8 * uint(i))
	}
	dataStart := 8 + int(hdrLen)
	if hdrLen == 0 || dataStart < 8 || dataStart > len(blob) {
		return nil, core.NewError("safetensors.Parse: header length out of range")
	}
	// Decode the header straight into typed entries rather than map[string]map[string]any:
	// the any-form boxes every dtype string, shape number and offset onto the heap (~26
	// allocs + ~45KB/tensor of interface garbage on a 256-tensor shard — the dominant cost
	// of the whole load). Shape and DataOffsets stay []int (not [2]int) so the
	// len(DataOffsets)==2 rejection below is unchanged: json discards trailing array
	// elements, so [2]int would silently accept a 3-element data_offsets.
	var hdr map[string]headerEntry
	if r := core.JSONUnmarshal(blob[8:dataStart], &hdr); !r.OK {
		return nil, core.NewError("safetensors.Parse: header JSON parse failed")
	}

	out := make(map[string]Tensor, len(hdr))
	for name, e := range hdr {
		if name == "__metadata__" { // the one reserved non-tensor key
			continue
		}
		if e.Dtype == "" {
			return nil, core.NewError("safetensors.Parse: tensor " + name + " missing dtype")
		}
		elem, known := dtypeBytes[e.Dtype]
		if !known {
			return nil, core.NewError("safetensors.Parse: tensor " + name + " unsupported dtype " + e.Dtype)
		}
		if e.Shape == nil { // a missing "shape" key — a present but empty "shape":[] (scalar) decodes non-nil
			return nil, core.NewError("safetensors.Parse: tensor " + name + " missing shape")
		}
		count := 1
		for _, d := range e.Shape {
			if d < 0 {
				return nil, core.NewError("safetensors.Parse: tensor " + name + " bad shape entry")
			}
			count *= d
		}
		if len(e.DataOffsets) != 2 {
			return nil, core.NewError("safetensors.Parse: tensor " + name + " data_offsets must be [start,end]")
		}
		start, end := e.DataOffsets[0], e.DataOffsets[1]
		if start < 0 || end < start || dataStart+end > len(blob) {
			return nil, core.NewError("safetensors.Parse: tensor " + name + " data_offsets out of range")
		}
		if end-start != count*elem {
			return nil, core.NewError("safetensors.Parse: tensor " + name + " byte span != dtype × shape")
		}
		out[name] = Tensor{Dtype: e.Dtype, Shape: e.Shape, Data: blob[dataStart+start : dataStart+end]}
	}
	return out, nil
}

// Encode writes tensors to a safetensors blob — the inverse of Parse: the 8-byte
// little-endian header length, the JSON header ({name:{dtype,shape,data_offsets}}), then
// the tensor data laid out in sorted-name order (deterministic). Validates each tensor's
// dtype + that its bytes match dtype × ∏shape. Parse(Encode(x)) round-trips x.
func Encode(tensors map[string]Tensor) ([]byte, error) {
	names := make([]string, 0, len(tensors))
	totalData := 0
	for n, t := range tensors {
		if n == "__metadata__" {
			return nil, core.NewError("safetensors.Encode: __metadata__ is reserved")
		}
		names = append(names, n)
		totalData += len(t.Data)
	}
	sort.Strings(names) // deterministic layout

	type entry struct {
		Dtype       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets [2]int `json:"data_offsets"`
	}
	hdr := make(map[string]entry, len(names))
	// First pass validates and lays out the offsets — no data copy. The payload is written straight
	// into the final buffer in the second pass below, so the intermediate concat buffer (a full extra
	// copy of every tensor's bytes — totalData, the whole checkpoint) is gone entirely.
	off := 0
	for _, n := range names {
		t := tensors[n]
		elem, ok := dtypeBytes[t.Dtype]
		if !ok {
			return nil, core.NewError("safetensors.Encode: tensor " + n + " unsupported dtype " + t.Dtype)
		}
		count := 1
		for _, d := range t.Shape {
			if d < 0 {
				return nil, core.NewError("safetensors.Encode: tensor " + n + " negative shape")
			}
			count *= d
		}
		if len(t.Data) != count*elem {
			return nil, core.NewError("safetensors.Encode: tensor " + n + " byte span != dtype × shape")
		}
		shape := t.Shape
		if shape == nil {
			shape = []int{}
		}
		hdr[n] = entry{Dtype: t.Dtype, Shape: shape, DataOffsets: [2]int{off, off + len(t.Data)}}
		off += len(t.Data)
	}

	hj := core.JSONMarshal(hdr)
	if !hj.OK {
		return nil, core.NewError("safetensors.Encode: header marshal failed")
	}
	hdrBytes := hj.Value.([]byte)
	out := make([]byte, 8+len(hdrBytes)+totalData)
	n := uint64(len(hdrBytes))
	for i := range 8 {
		out[i] = byte(n >> (8 * uint(i)))
	}
	copy(out[8:], hdrBytes)
	// Copy each tensor's payload directly into its slot in the final buffer (name-sorted, same order
	// the offsets were assigned), so the data is copied exactly once.
	dataStart := 8 + len(hdrBytes)
	for _, name := range names {
		copy(out[dataStart+hdr[name].DataOffsets[0]:], tensors[name].Data)
	}
	return out, nil
}

// Load reads a safetensors file and Parses it. NOTE: it reads the whole file into memory
// (the per-tensor Data then sub-slices it); an mmap variant for multi-GB checkpoints is a
// later optimisation, and loading a real model is a deliberate, memory-heavy operation.
func Load(path string) (map[string]Tensor, error) {
	str, err := coreio.Local.Read(path)
	if err != nil {
		return nil, core.E("safetensors.Load", "read "+path, err)
	}
	// coreio.Local.Read returns a freshly-allocated, caller-owned string (its backing
	// []byte is never referenced again), so AsBytes views it without a second whole-file
	// copy — Parse treats the blob as read-only and Tensor.Data sub-slices it, the same
	// read-only-view contract the mmap path relies on. []byte(str) here would duplicate the
	// entire shard (10MB+ FLAT on a real shard) before Parse even runs.
	return Parse(core.AsBytes(str))
}
