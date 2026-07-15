// SPDX-Licence-Identifier: EUPL-1.2

// Package safetensors reads and writes the safetensors checkpoint format
// (HuggingFace's on-disk layout for dense model weights: an 8-byte header
// size, a JSON tensor directory, then raw little-endian tensor bytes) with
// no dependency on any inference engine or model-management layer.
//
// The package is a leaf: it depends only on the core runtime. Format
// conversion policy (MLX→PEFT renames, GGUF quantisation, merge maths)
// lives with the packages that own those workflows — they consume these
// codecs, never the other way round.
//
//	read := safetensors.ReadSafetensors("/models/adapter_model.safetensors")
//	if !read.OK { return read }
//	data := read.Value.(safetensors.SafetensorsData)
//	info := data.Tensors["model.embed_tokens.weight"]
//	raw := safetensors.GetTensorData(info, data.Data)
package safetensors

import (
	"encoding/binary"
	"maps"
	"slices"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// SafetensorsHeader represents the header of a safetensors file.
type SafetensorsHeader struct {
	Metadata map[string]string                `json:"__metadata__,omitempty"`
	Tensors  map[string]SafetensorsTensorInfo `json:"-"`
}

// SafetensorsTensorInfo describes a tensor's dtype, shape, and data location.
type SafetensorsTensorInfo struct {
	Dtype       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

// SafetensorsData carries parsed safetensors metadata and tensor bytes.
type SafetensorsData struct {
	Tensors map[string]SafetensorsTensorInfo
	Data    []byte
}

// ReadSafetensors reads a safetensors file and returns tensor info and raw data.
//
//	read := safetensors.ReadSafetensors(path)
//	if !read.OK { return read }
//	data := read.Value.(safetensors.SafetensorsData)
func ReadSafetensors(path string) core.Result {
	raw, err := coreio.Local.Read(path)
	if err != nil {
		return core.Fail(core.E("safetensors.ReadSafetensors", "read file", err))
	}
	// coreio.Local.Read returns a freshly-allocated, caller-owned string whose backing []byte
	// is never referenced again, so view it read-only rather than copying the whole file a second
	// time — the returned SafetensorsData.Data sub-slices this view and every consumer (decode,
	// quantise, transpose-to-new-buffer) treats it read-only, the same contract Load/Parse rely on.
	// []byte(raw) here duplicated the entire checkpoint (MBs) before any tensor was touched.
	data := core.AsBytes(raw)

	if len(data) < 8 {
		return core.Fail(core.E("safetensors.ReadSafetensors", "file too small", nil))
	}

	headerSize := int(binary.LittleEndian.Uint64(data[:8]))
	if 8+headerSize > len(data) {
		return core.Fail(core.E("safetensors.ReadSafetensors", core.Sprintf("invalid header size %d", headerSize), nil))
	}

	headerJSON := data[8 : 8+headerSize]
	tensorData := data[8+headerSize:]

	var rawHeader map[string]SafetensorsTensorInfo
	if r := core.JSONUnmarshal(headerJSON, &rawHeader); !r.OK {
		return core.Fail(core.E("safetensors.ReadSafetensors", "parse header", r.Value.(error)))
	}
	delete(rawHeader, "__metadata__")

	return core.Ok(SafetensorsData{Tensors: rawHeader, Data: tensorData})
}

// GetTensorData extracts raw bytes for a tensor from the data section.
//
//	raw := safetensors.GetTensorData(data.Tensors["blk.0.weight"], data.Data)
func GetTensorData(info SafetensorsTensorInfo, allData []byte) []byte {
	return allData[info.DataOffsets[0]:info.DataOffsets[1]]
}

// WriteSafetensors writes tensors to a safetensors file.
//
//	r := safetensors.WriteSafetensors(path, tensors, tensorData)
//	if !r.OK { return r }
func WriteSafetensors(path string, tensors map[string]SafetensorsTensorInfo, tensorData map[string][]byte) core.Result {
	keys := slices.Sorted(maps.Keys(tensors))

	// Emit the JSON header directly with the package's hand-rolled emitter (shared with Encode /
	// subsetHeaderEncoded) rather than copying into an updatedTensors map and handing it to the
	// reflection-driven core.JSONMarshalString — that map copy plus per-field boxing was the whole
	// per-tensor allocation cost. Byte-identical to the previous marshal for any input: keys sorted,
	// fields in declaration order (dtype, shape, data_offsets), integers base-10, and a nil Shape
	// emits "null" while an empty []int{} emits "[]" (the encoding/json nil-slice rule the marshal
	// used). Offsets lay out sequentially in key order. TestSafetensors_WriteSafetensors_HeaderGolden
	// pins the exact header bytes including the null/[] distinction.
	estBytes := 2 // {} braces
	for _, k := range keys {
		info := tensors[k]
		estBytes += len(k) + len(info.Dtype) + 12*len(info.Shape) + 64
	}
	headerJSON := make([]byte, 0, estBytes)
	headerJSON = append(headerJSON, '{')
	offset := 0
	for i, k := range keys {
		info := tensors[k]
		data := tensorData[k]
		if i > 0 {
			headerJSON = append(headerJSON, ',')
		}
		headerJSON = appendJSONString(headerJSON, k)
		headerJSON = append(headerJSON, ':', '{', '"', 'd', 't', 'y', 'p', 'e', '"', ':')
		headerJSON = appendJSONString(headerJSON, info.Dtype)
		headerJSON = append(headerJSON, ',', '"', 's', 'h', 'a', 'p', 'e', '"', ':')
		if info.Shape == nil {
			headerJSON = append(headerJSON, 'n', 'u', 'l', 'l')
		} else {
			headerJSON = append(headerJSON, '[')
			for j, d := range info.Shape {
				if j > 0 {
					headerJSON = append(headerJSON, ',')
				}
				headerJSON = appendJSONInt64(headerJSON, int64(d))
			}
			headerJSON = append(headerJSON, ']')
		}
		headerJSON = append(headerJSON, ',', '"', 'd', 'a', 't', 'a', '_', 'o', 'f', 'f', 's', 'e', 't', 's', '"', ':', '[')
		headerJSON = appendJSONInt64(headerJSON, int64(offset))
		headerJSON = append(headerJSON, ',')
		headerJSON = appendJSONInt64(headerJSON, int64(offset+len(data)))
		headerJSON = append(headerJSON, ']', '}')
		offset += len(data)
	}
	headerJSON = append(headerJSON, '}')

	f, err := coreio.Local.Create(path)
	if err != nil {
		return core.Fail(core.E("safetensors.WriteSafetensors", core.Sprintf("create %s", path), err))
	}
	defer f.Close()

	headerSizeBuf := make([]byte, 8)
	binary.LittleEndian.PutUint64(headerSizeBuf, uint64(len(headerJSON)))

	if _, err := f.Write(headerSizeBuf); err != nil {
		return core.Fail(err)
	}
	if _, err := f.Write(headerJSON); err != nil {
		return core.Fail(err)
	}

	for _, k := range keys {
		if _, err := f.Write(tensorData[k]); err != nil {
			return core.Fail(err)
		}
	}

	return core.Ok(nil)
}
