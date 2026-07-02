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
	data := []byte(raw)

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
	if r := core.JSONUnmarshalString(string(headerJSON), &rawHeader); !r.OK {
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

	offset := 0
	updatedTensors := make(map[string]SafetensorsTensorInfo, len(tensors))
	for _, k := range keys {
		info := tensors[k]
		data := tensorData[k]
		info.DataOffsets = [2]int{offset, offset + len(data)}
		updatedTensors[k] = info
		offset += len(data)
	}

	// updatedTensors marshals to identical JSON as a map[string]any copy
	// (json sorts keys; struct values serialise the same boxed or not).
	headerJSON := []byte(core.JSONMarshalString(updatedTensors))

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
