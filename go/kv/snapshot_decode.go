// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"encoding/binary"
	"math"
	"unsafe"

	core "dappco.re/go"
)

// UnmarshalBinary replaces the snapshot with data loaded from the stable binary format.
func (s *Snapshot) UnmarshalBinary(data []byte) error {
	if s == nil {
		return errSnapshotNil
	}
	loaded, err := parseKVSnapshot(data)
	if err != nil {
		return err
	}
	*s = *loaded
	return nil
}

// Load reads a KV snapshot saved by (*Snapshot).Save.
func Load(path string) (*Snapshot, error) {
	return LoadWithOptions(path, LoadOptions{})
}

// LoadWithOptions reads a KV snapshot with explicit decode options.
func LoadWithOptions(path string, opts LoadOptions) (*Snapshot, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("Load", "read snapshot", ResultError(read))
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return nil, core.E("Load", "read snapshot returned non-byte data", nil)
	}
	return parseKVSnapshotWithOptions(data, opts)
}

func parseKVSnapshot(data []byte) (*Snapshot, error) {
	return parseKVSnapshotWithOptions(data, LoadOptions{})
}

func parseKVSnapshotWithOptions(data []byte, opts LoadOptions) (*Snapshot, error) {
	reader := kvSnapshotReader{data: data}
	if magic := string(reader.read(len(kvSnapshotMagic))); magic != kvSnapshotMagic {
		return nil, core.E("Load", "invalid KV snapshot magic", nil)
	}
	version := int(reader.u32())
	if version <= 0 || version > SnapshotVersion {
		return nil, core.E("Load", "unsupported KV snapshot version", nil)
	}
	snapshot := &Snapshot{
		Version:       version,
		Architecture:  reader.string(),
		NumLayers:     int(reader.u32()),
		NumHeads:      int(reader.u32()),
		SeqLen:        int(reader.u32()),
		HeadDim:       int(reader.u32()),
		NumQueryHeads: int(reader.u32()),
	}
	if snapshot.Version >= 2 {
		snapshot.TokenOffset = int(reader.u32())
	}
	tokenCount := int(reader.u32())
	if tokenCount > 0 {
		// Batch the i32 block read so bounds check is paid once.
		chunk := reader.read(tokenCount * 4)
		if chunk != nil {
			// Reinterpret-cast bytes → int32 via memcpy; same pattern as
			// f32s() reader. Single copy vs N×Uint32 + int32 cast.
			snapshot.Tokens = make([]int32, tokenCount)
			dst := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(snapshot.Tokens))), tokenCount*4)
			copy(dst, chunk)
		}
	}
	if snapshot.Version >= 2 {
		generatedCount := int(reader.u32())
		if generatedCount > 0 {
			chunk := reader.read(generatedCount * 4)
			if chunk != nil {
				snapshot.Generated = make([]int32, generatedCount)
				dst := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(snapshot.Generated))), generatedCount*4)
				copy(dst, chunk)
			}
		}
	}
	layerCount := int(reader.u32())
	if layerCount > 0 {
		snapshot.Layers = make([]LayerSnapshot, layerCount)
		// Heads-slab: typical snapshots carry NumHeads heads per layer, so
		// one backing slice sized to layerCount*NumHeads collapses the per-
		// layer make([]HeadSnapshot,...) into a single allocation. Layers
		// with a different head count fall through to the per-layer make.
		var headSlab []HeadSnapshot
		var slabCursor int
		if snapshot.NumHeads > 0 {
			headSlab = make([]HeadSnapshot, layerCount*snapshot.NumHeads)
		}
		for layerIdx := range snapshot.Layers {
			layer := &snapshot.Layers[layerIdx]
			layer.Layer = int(reader.i32())
			layer.CacheIndex = int(reader.i32())
			headCount := int(reader.u32())
			if snapshot.Version >= 5 {
				layer.CacheMode = reader.string()
				payloadCount := int(reader.u32())
				if payloadCount > 0 {
					layer.TurboQuantPayloads = make([][]byte, payloadCount)
					for payloadIdx := range layer.TurboQuantPayloads {
						layer.TurboQuantPayloads[payloadIdx] = reader.bytes()
					}
				}
			}
			if snapshot.Version >= 6 {
				layer.MaxSize = int(reader.u32())
			}
			if snapshot.Version >= 4 {
				layer.KeyShape = reader.i32s()
				key := reader.encodedTensor(LoadOptions{RawKVOnly: true})
				layer.KeyDType = key.DType
				layer.KeyBytes = key.Bytes
				layer.ValueShape = reader.i32s()
				value := reader.encodedTensor(LoadOptions{RawKVOnly: true})
				layer.ValueDType = value.DType
				layer.ValueBytes = value.Bytes
			}
			if headCount > 0 {
				if headSlab != nil && slabCursor+headCount <= len(headSlab) {
					layer.Heads = headSlab[slabCursor : slabCursor+headCount : slabCursor+headCount]
					slabCursor += headCount
				} else {
					layer.Heads = make([]HeadSnapshot, headCount)
				}
				for headIdx := range layer.Heads {
					if snapshot.Version >= 3 {
						key := reader.encodedTensor(opts)
						value := reader.encodedTensor(opts)
						layer.Heads[headIdx].Key = key.Values
						layer.Heads[headIdx].KeyDType = key.DType
						layer.Heads[headIdx].KeyBytes = key.Bytes
						layer.Heads[headIdx].Value = value.Values
						layer.Heads[headIdx].ValueDType = value.DType
						layer.Heads[headIdx].ValueBytes = value.Bytes
					} else {
						layer.Heads[headIdx].Key = reader.f32s()
						layer.Heads[headIdx].Value = reader.f32s()
					}
				}
			}
		}
	}
	if snapshot.Version >= 2 {
		shapeCount := int(reader.u32())
		if shapeCount > 0 {
			chunk := reader.read(shapeCount * 4)
			if chunk != nil {
				// Reinterpret-cast bytes → int32 via memcpy; same pattern
				// as f32s() reader. Single copy vs N×Uint32 + int32 cast.
				snapshot.LogitShape = make([]int32, shapeCount)
				dst := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(snapshot.LogitShape))), shapeCount*4)
				copy(dst, chunk)
			}
		}
		snapshot.Logits = reader.f32s()
	}
	if reader.err != nil {
		return nil, core.E("Load", "parse snapshot", reader.err)
	}
	if err := validateKVSnapshotCompressedPayloads(snapshot); err != nil {
		return nil, core.E("Load", "validate compressed KV payload metadata", err)
	}
	if snapshot.TokenOffset == 0 {
		snapshot.TokenOffset = len(snapshot.Tokens)
	}
	return snapshot, nil
}

func parseKVSnapshotTokens(data []byte) ([]int32, error) {
	reader := kvSnapshotReader{data: data}
	if magic := string(reader.read(len(kvSnapshotMagic))); magic != kvSnapshotMagic {
		return nil, core.E("Load", "invalid KV snapshot magic", nil)
	}
	version := int(reader.u32())
	if version <= 0 || version > SnapshotVersion {
		return nil, core.E("Load", "unsupported KV snapshot version", nil)
	}
	architectureLength := int(reader.u32())
	reader.read(architectureLength)
	for range 5 {
		reader.u32()
	}
	if version >= 2 {
		reader.u32()
	}
	tokenCount := int(reader.u32())
	if tokenCount < 0 || tokenCount > (len(reader.data)-reader.offset)/4 {
		return nil, errStateTokenBlockTokenCount
	}
	tokens := make([]int32, tokenCount)
	if tokenCount > 0 {
		// Batch the token block read so bounds check is paid once
		// regardless of token count.
		chunk := reader.read(tokenCount * 4)
		if chunk != nil {
			// Reinterpret-cast bytes → int32 via memcpy; same pattern as
			// f32s() reader. Single copy vs N×Uint32 + int32 cast.
			dst := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(tokens))), tokenCount*4)
			copy(dst, chunk)
		}
	}
	if reader.err != nil {
		return nil, core.E("Load", "parse State tokens", reader.err)
	}
	return tokens, nil
}

// parseKVSnapshotTokensInto appends the token block from data to dst and
// returns the extended slice. Avoids the per-block []int32 allocation
// LoadPrefixTokensFromStateBlocks otherwise pays through parseKVSnapshotTokens.
func parseKVSnapshotTokensInto(dst []int32, data []byte) ([]int32, error) {
	reader := kvSnapshotReader{data: data}
	if magic := string(reader.read(len(kvSnapshotMagic))); magic != kvSnapshotMagic {
		return dst, errInvalidSnapshotMagic
	}
	version := int(reader.u32())
	if version <= 0 || version > SnapshotVersion {
		return dst, errUnsupportedSnapshotVersion
	}
	architectureLength := int(reader.u32())
	reader.read(architectureLength)
	for range 5 {
		reader.u32()
	}
	if version >= 2 {
		reader.u32()
	}
	tokenCount := int(reader.u32())
	if tokenCount < 0 || tokenCount > (len(reader.data)-reader.offset)/4 {
		return dst, errStateTokenBlockTokenCount
	}
	if tokenCount == 0 {
		return dst, nil
	}
	chunk := reader.read(tokenCount * 4)
	if chunk == nil {
		if reader.err != nil {
			return dst, core.E("Load", "parse State tokens", reader.err)
		}
		return dst, nil
	}
	// Extend dst once for the whole block — avoids per-token append regrow.
	start := len(dst)
	if cap(dst) >= start+tokenCount {
		dst = dst[:start+tokenCount]
	} else {
		grown := make([]int32, start+tokenCount, max(cap(dst)*2, start+tokenCount))
		copy(grown, dst)
		dst = grown
	}
	// Reinterpret-cast bytes → int32 via memcpy; same pattern as
	// f32s() reader. Single copy vs N×Uint32 + int32 cast.
	out := dst[start:]
	outBytes := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(out))), tokenCount*4)
	copy(outBytes, chunk)
	if reader.err != nil {
		return dst, core.E("Load", "parse State tokens", reader.err)
	}
	return dst, nil
}

type kvSnapshotReader struct {
	data   []byte
	offset int
	err    error
}

func (r *kvSnapshotReader) read(n int) []byte {
	if r.err != nil {
		return nil
	}
	if n < 0 || len(r.data)-r.offset < n {
		r.err = errTruncatedSnapshot
		return nil
	}
	chunk := r.data[r.offset : r.offset+n]
	r.offset += n
	return chunk
}

func (r *kvSnapshotReader) u32() uint32 {
	chunk := r.read(4)
	if chunk == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(chunk)
}

func (r *kvSnapshotReader) i32() int32 {
	return int32(r.u32())
}

func (r *kvSnapshotReader) string() string {
	size := int(r.u32())
	return string(r.read(size))
}

// dtypeString reads a length-prefixed dtype tag. KV snapshots use a fixed
// six-token vocabulary ("float32"/"F32", "float16"/"F16", "bfloat16"/"BF16");
// matching bytes-first returns the literal canonical string with zero
// allocation. Unknown dtypes fall back to a fresh string for the validator
// to reject downstream.
func (r *kvSnapshotReader) dtypeString() string {
	size := int(r.u32())
	chunk := r.read(size)
	if chunk == nil {
		return ""
	}
	switch len(chunk) {
	case 3:
		switch string(chunk) {
		case "F32":
			return "F32"
		case "F16":
			return "F16"
		}
	case 4:
		if string(chunk) == "BF16" {
			return "BF16"
		}
	case 7:
		switch string(chunk) {
		case "float32":
			return "float32"
		case "float16":
			return "float16"
		}
	case 8:
		if string(chunk) == "bfloat16" {
			return "bfloat16"
		}
	}
	return string(chunk)
}

func (r *kvSnapshotReader) i32s() []int32 {
	size := int(r.u32())
	if size <= 0 {
		return nil
	}
	// Single bounds check + direct decode amortises the per-element
	// read+slice overhead the per-call r.u32() loop incurred.
	chunk := r.read(size * 4)
	if chunk == nil {
		return nil
	}
	// Reinterpret-cast bytes → int32 via memcpy; same pattern as
	// f32s() reader. Single copy vs N×Uint32 + int32 cast.
	values := make([]int32, size)
	dst := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), size*4)
	copy(dst, chunk)
	return values
}

func (r *kvSnapshotReader) bytes() []byte {
	size := int(r.u32())
	raw := r.read(size)
	if raw == nil {
		return nil
	}
	return raw
}

func (r *kvSnapshotReader) f32s() []float32 {
	size := int(r.u32())
	if size <= 0 {
		return nil
	}
	// Single bounds check + direct decode amortises the per-element
	// read+slice overhead the per-call r.u32() loop incurred.
	chunk := r.read(size * 4)
	if chunk == nil {
		return nil
	}
	// Reinterpret-cast the bytes back into float32 via memcpy: source
	// is little-endian on both Go-supported architectures, matching
	// what f32sRaw wrote. One copy vs N×Uint32+Float32frombits.
	// We copy because chunk references the reader's input buffer
	// (potentially mmap-backed); the returned slice must outlive the
	// reader. Same pattern as f32sRaw on the write side.
	values := make([]float32, size)
	dst := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), size*4)
	copy(dst, chunk)
	return values
}

type kvSnapshotEncodedTensor struct {
	Values []float32
	DType  string
	Bytes  []byte
}

func (r *kvSnapshotReader) encodedF32s() []float32 {
	return r.encodedTensor(LoadOptions{}).Values
}

func (r *kvSnapshotReader) encodedTensor(opts LoadOptions) kvSnapshotEncodedTensor {
	encoding := r.u32()
	size := int(r.u32())
	switch encoding {
	case 0:
		if size <= 0 {
			return kvSnapshotEncodedTensor{Values: []float32{}}
		}
		// Single bounds check via batched read avoids per-element bounds work.
		chunk := r.read(size * 4)
		if chunk == nil {
			return kvSnapshotEncodedTensor{}
		}
		// Reinterpret-cast bytes → float32 via memcpy; same pattern
		// as f32s() above. Single copy vs N×Uint32+Float32frombits.
		values := make([]float32, size)
		dst := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), size*4)
		copy(dst, chunk)
		return kvSnapshotEncodedTensor{Values: values}
	case 1:
		scale := math.Float32frombits(r.u32())
		raw := r.read(size)
		values := make([]float32, size)
		for i, value := range raw {
			values[i] = float32(int8(value)) * scale
		}
		return kvSnapshotEncodedTensor{Values: values}
	case 2:
		dtype := r.dtypeString()
		raw := r.bytes()
		dtype, err := validateKVSnapshotNativeTensor(dtype, raw, size)
		if err != nil {
			r.err = err
			return kvSnapshotEncodedTensor{}
		}
		if opts.RawKVOnly {
			return kvSnapshotEncodedTensor{
				DType: dtype,
				Bytes: raw,
			}
		}
		values, err := decodeKVSnapshotNativeTensor(dtype, raw, size)
		if err != nil {
			r.err = err
			return kvSnapshotEncodedTensor{}
		}
		return kvSnapshotEncodedTensor{
			Values: values,
			DType:  dtype,
			Bytes:  raw,
		}
	default:
		r.err = errUnsupportedTensorEncoding
		return kvSnapshotEncodedTensor{}
	}
}
