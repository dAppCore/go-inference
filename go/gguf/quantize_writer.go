// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"math"
	"sort"
	"strconv"

	core "dappco.re/go"
)

func ggufQuantizeMetadata(source Source, format QuantizeFormat, labels map[string]string) []MetadataEntry {
	fileType := uint32(7)
	quantizationType := string(QuantizeQ8_0)
	if format == QuantizeQ4_0 {
		fileType = 2
		quantizationType = string(QuantizeQ4_0)
	} else if format == QuantizeQ5_0 {
		fileType = 12
		quantizationType = string(QuantizeQ5_0)
	} else if format == QuantizeQ4_K {
		fileType = 15
		quantizationType = string(QuantizeQ4_K_M)
	} else if format == QuantizeQ5_K {
		fileType = 16
		quantizationType = "q5_k_m"
	} else if format == QuantizeQ6_K {
		fileType = 17
		quantizationType = "q6_k"
	} else if format == QuantizeQ8_K {
		fileType = 18
		quantizationType = "q8_k"
	} else if format == QuantizeQ3_K {
		fileType = 12
		quantizationType = "q3_k"
	} else if format == QuantizeQ2_K {
		fileType = 10
		quantizationType = "q2_k"
	}
	architecture := source.Architecture
	metadata := []MetadataEntry{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: architecture},
		{Key: "general.file_type", ValueType: ValueTypeUint32, Value: fileType},
		{Key: "general.quantization_version", ValueType: ValueTypeUint32, Value: uint32(2)},
		{Key: "general.quantization_type", ValueType: ValueTypeString, Value: quantizationType},
		{Key: "general.alignment", ValueType: ValueTypeUint32, Value: uint32(32)},
	}
	if source.VocabSize > 0 {
		metadata = append(metadata, MetadataEntry{Key: architecture + ".vocab_size", ValueType: ValueTypeUint32, Value: uint32(source.VocabSize)})
	}
	if source.HiddenSize > 0 {
		metadata = append(metadata, MetadataEntry{Key: architecture + ".embedding_length", ValueType: ValueTypeUint32, Value: uint32(source.HiddenSize)})
	}
	if source.NumLayers > 0 {
		metadata = append(metadata, MetadataEntry{Key: architecture + ".block_count", ValueType: ValueTypeUint32, Value: uint32(source.NumLayers)})
	}
	if source.ContextLength > 0 {
		metadata = append(metadata, MetadataEntry{Key: architecture + ".context_length", ValueType: ValueTypeUint32, Value: uint32(source.ContextLength)})
	}
	if len(labels) > 0 {
		keys := make([]string, 0, len(labels))
		for key := range labels {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			metadata = append(metadata, MetadataEntry{Key: "gguf.label." + key, ValueType: ValueTypeString, Value: labels[key]})
		}
	}
	return metadata
}

// WriteFile writes metadata and tensors as a GGUF v3 file at path. Tensor
// offsets are assigned by the writer — each tensor's data starts at the next
// 32-byte-aligned offset and the data section's end is padded to the same
// boundary, matching upstream gguf-py's writer (which pads every tensor,
// including the last).
//
//	err := gguf.WriteFile(out, []gguf.MetadataEntry{
//	    {Key: "general.architecture", ValueType: gguf.ValueTypeString, Value: "gemma3"},
//	    {Key: "adapter.lora.alpha", ValueType: gguf.ValueTypeFloat32, Value: float32(16)},
//	}, tensors)
func WriteFile(path string, metadata []MetadataEntry, tensors []Tensor) error {
	return writeQuantizedGGUF(path, metadata, tensors)
}

// writeQuantizedGGUF writes tensors (already quantised to GGUF block bytes)
// and metadata as a GGUF v3 file at path.
func writeQuantizedGGUF(path string, metadata []MetadataEntry, tensors []Tensor) error {
	created := core.Create(path)
	if !created.OK {
		return quantizeGGUFResultError(created)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	assignGGUFTensorOffsets(tensors, 32)
	if err := writeQuantizedGGUFHeader(file, metadata, tensors); err != nil {
		return err
	}
	var written uint64
	for _, tensor := range tensors {
		if tensor.Offset < written {
			return core.NewError("gguf: GGUF tensor offsets are not monotonic")
		}
		if err := writePadding(file, tensor.Offset-written); err != nil {
			return err
		}
		if _, err := file.Write(tensor.Data); err != nil {
			return err
		}
		written = tensor.Offset + uint64(len(tensor.Data))
	}
	// Pad the data section's end to the alignment boundary. Upstream
	// gguf-py pads after every tensor including the last, so a canonical
	// GGUF file's length is 32-byte aligned; readers index tensor data by
	// offset and never see the trailing zeros. Metadata-only files carry
	// no data section and stay unpadded.
	if len(tensors) > 0 {
		if err := writePadding(file, alignPadding(written, 32)); err != nil {
			return err
		}
	}
	return nil
}

func writeQuantizedGGUFHeader(file *core.OSFile, metadata []MetadataEntry, tensors []Tensor) error {
	// Single 24-byte header: magic(4) + version(4) + tensorCount(8) + metadataCount(8).
	// One write call replaces 4 reflect.Write calls.
	var header [24]byte
	copy(header[:4], "GGUF")
	binary.LittleEndian.PutUint32(header[4:8], 3)
	binary.LittleEndian.PutUint64(header[8:16], uint64(len(tensors)))
	binary.LittleEndian.PutUint64(header[16:24], uint64(len(metadata)))
	if _, err := file.Write(header[:]); err != nil {
		return err
	}
	for _, entry := range metadata {
		if err := writeGGUFMetadataEntry(file, entry); err != nil {
			return err
		}
	}
	for _, tensor := range tensors {
		if err := writeGGUFTensorInfo(file, tensor); err != nil {
			return err
		}
	}
	position, err := file.Seek(0, 1)
	if err != nil {
		return err
	}
	if err := writePadding(file, alignPadding(uint64(position), 32)); err != nil {
		return err
	}
	return nil
}

func assignGGUFTensorOffsets(tensors []Tensor, alignment uint64) {
	var offset uint64
	for i := range tensors {
		offset += alignPadding(offset, alignment)
		tensors[i].Offset = offset
		offset += uint64(len(tensors[i].Data))
	}
}

func writeGGUFMetadataEntry(file *core.OSFile, entry MetadataEntry) error {
	if err := writeGGUFStringValue(file, entry.Key); err != nil {
		return err
	}
	// valueType(4) — direct LE encoding skips reflect dispatch.
	var typeBuf [4]byte
	binary.LittleEndian.PutUint32(typeBuf[:], entry.ValueType)
	if _, err := file.Write(typeBuf[:]); err != nil {
		return err
	}
	return writeGGUFMetadataValue(file, entry.ValueType, entry.Value)
}

func writeGGUFMetadataValue(file *core.OSFile, valueType uint32, value any) error {
	switch valueType {
	case ValueTypeString:
		stringValue, ok := value.(string)
		if !ok {
			return core.NewError("gguf: GGUF metadata value is not a string")
		}
		return writeGGUFStringValue(file, stringValue)
	case ValueTypeUint32:
		var v uint32
		switch concrete := value.(type) {
		case uint32:
			v = concrete
		case int:
			v = uint32(concrete)
		default:
			return core.NewError("gguf: GGUF metadata value is not uint32")
		}
		var buf [4]byte
		binary.LittleEndian.PutUint32(buf[:], v)
		_, err := file.Write(buf[:])
		return err
	case ValueTypeFloat32:
		floatValue, ok := value.(float32)
		if !ok {
			return core.NewError("gguf: GGUF metadata value is not float32")
		}
		var buf [4]byte
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(floatValue))
		_, err := file.Write(buf[:])
		return err
	default:
		return core.NewError("gguf: unsupported GGUF metadata write type " + strconv.FormatUint(uint64(valueType), 10))
	}
}

func writeGGUFTensorInfo(file *core.OSFile, tensor Tensor) error {
	if err := writeGGUFStringValue(file, tensor.Name); err != nil {
		return err
	}
	// Pack ndim(4) + all dim(8 each) + tensorType(4) + offset(8) into
	// one batched write — avoids one binary.Write reflect call per
	// dimension (typically 2-4 per tensor).
	dims := tensor.Shape
	bufLen := 4 + len(dims)*8 + 4 + 8
	// Small scratch on stack for the common 2-4 dim case; fall back to
	// heap for higher rank tensors (rare in real GGUF files).
	var stack [64]byte
	var buf []byte
	if bufLen <= len(stack) {
		buf = stack[:bufLen]
	} else {
		buf = make([]byte, bufLen)
	}
	binary.LittleEndian.PutUint32(buf[:4], uint32(len(dims)))
	pos := 4
	for _, dim := range dims {
		binary.LittleEndian.PutUint64(buf[pos:pos+8], dim)
		pos += 8
	}
	binary.LittleEndian.PutUint32(buf[pos:pos+4], tensor.Type)
	pos += 4
	binary.LittleEndian.PutUint64(buf[pos:pos+8], tensor.Offset)
	_, err := file.Write(buf)
	return err
}

func writeGGUFStringValue(file *core.OSFile, value string) error {
	// Length-prefix in one batched write with the value bytes when the
	// value is small enough to fit on stack. For the common metadata-
	// key case (32-200 bytes) this skips one syscall + one Write call.
	var stack [256]byte
	if len(value)+8 <= len(stack) {
		buf := stack[:8+len(value)]
		binary.LittleEndian.PutUint64(buf[:8], uint64(len(value)))
		copy(buf[8:], value)
		_, err := file.Write(buf)
		return err
	}
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(value)))
	if _, err := file.Write(lenBuf[:]); err != nil {
		return err
	}
	_, err := file.Write(core.AsBytes(value))
	return err
}

// ggufPaddingZeros — package-level read-only zero buffer for writePadding.
// 32 KiB chunk matches the original on-stack size; living at package scope
// avoids a 32 KiB stack-frame allocation per writePadding call.
var ggufPaddingZeros [32 * 1024]byte

func writePadding(file *core.OSFile, n uint64) error {
	for n > 0 {
		size := min(n, uint64(len(ggufPaddingZeros)))
		if _, err := file.Write(ggufPaddingZeros[:size]); err != nil {
			return err
		}
		n -= size
	}
	return nil
}

func alignPadding(offset, alignment uint64) uint64 {
	if alignment == 0 {
		return 0
	}
	return (alignment - (offset % alignment)) % alignment
}
