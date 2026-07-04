// Package gguf provides a GGUF binary metadata parser for reading model headers.
//
// GGUF (GGML Universal File) is the file format used by llama.cpp and other
// GGML-based inference engines. This package reads the metadata key-value pairs
// from the file header without loading tensor data, enabling fast model discovery.
//
// Supports GGUF v2 (uint32 counts) and v3 (uint64 counts).
package gguf

import (
	"encoding/binary"
	"io"
	"math"

	core "dappco.re/go"
)

// ggufMagic is the GGUF file magic number: "GGUF" in little-endian.
const ggufMagic = 0x46554747

// GGUF value type codes.
const (
	typeUint8   uint32 = 0
	typeInt8    uint32 = 1
	typeUint16  uint32 = 2
	typeInt16   uint32 = 3
	typeUint32  uint32 = 4
	typeInt32   uint32 = 5
	typeFloat32 uint32 = 6
	typeBool    uint32 = 7
	typeString  uint32 = 8
	typeArray   uint32 = 9
	typeUint64  uint32 = 10
	typeInt64   uint32 = 11
	typeFloat64 uint32 = 12
)

// Metadata holds the interesting fields extracted from a GGUF file header.
type Metadata struct {
	Architecture  string // "gemma3", "llama", "qwen2"
	Name          string // human-readable model name
	SizeLabel     string // "1B", "8B", etc.
	ContextLength uint32 // native context window
	BlockCount    uint32 // transformer layers
	FileType      uint32 // GGML quantisation file type
	FileSize      int64  // file size on disk in bytes
}

// TensorInfo describes one GGUF tensor entry without loading tensor bytes.
type TensorInfo struct {
	Name       string
	Dimensions []uint64
	Type       uint32
	TypeName   string
	Offset     uint64
	ByteSize   uint64
}

// Info is the parsed GGUF header, including metadata and tensor directory.
type Info struct {
	Metadata   Metadata
	Tensors    []TensorInfo
	Alignment  uint32
	DataOffset int64
}

const defaultAlignment = 32

// fileTypeNames maps GGML quantisation file type numbers to human-readable names.
var fileTypeNames = map[uint32]string{
	0:  "F32",
	1:  "F16",
	2:  "Q4_0",
	3:  "Q4_1",
	7:  "Q8_0",
	8:  "Q5_0",
	9:  "Q5_1",
	10: "Q2_K",
	11: "Q3_K_S",
	12: "Q3_K_M",
	13: "Q3_K_L",
	14: "Q4_K_S",
	15: "Q4_K_M",
	16: "Q5_K_S",
	17: "Q5_K_M",
	18: "Q6_K",
}

var tensorTypeNames = map[uint32]string{
	0:  "F32",
	1:  "F16",
	2:  "Q4_0",
	3:  "Q4_1",
	6:  "Q5_0",
	7:  "Q5_1",
	8:  "Q8_0",
	10: "Q2_K",
	11: "Q3_K",
	12: "Q4_K",
	13: "Q5_K",
	14: "Q6_K",
	15: "Q8_K",
	24: "I8",
	25: "I16",
	26: "I32",
	27: "I64",
	28: "F64",
	30: "BF16",
}

//	name := FileTypeName(15) // "Q4_K_M"
//
// FileTypeName returns a human-readable name for a GGML quantisation file
// type. Unknown types return "type_N" where N is the numeric value.
func FileTypeName(ft uint32) string {
	if name, ok := fileTypeNames[ft]; ok {
		return name
	}
	return core.Sprintf("type_%d", ft)
}

// TensorTypeName returns a human-readable name for a GGML tensor type.
func TensorTypeName(t uint32) string {
	if name, ok := tensorTypeNames[t]; ok {
		return name
	}
	return core.Sprintf("type_%d", t)
}

//	metadata, err := ReadMetadata("/models/gemma3-4b.gguf")
//
// ReadMetadata reads the GGUF header from the file at path and returns the
// extracted metadata. Only metadata KV pairs are read; tensor data is not
// loaded.
func ReadMetadata(path string) (
	Metadata,
	error,
) {
	info, err := readInfo(path, false)
	if err != nil {
		return Metadata{}, err
	}
	return info.Metadata, nil
}

// ReadInfo reads the GGUF header and tensor directory from path. Tensor bytes
// are not loaded.
func ReadInfo(path string) (
	Info,
	error,
) {
	return readInfo(path, true)
}

func readInfo(path string, includeTensors bool) (
	Info,
	error,
) {
	fileResult := core.Open(path)
	if !fileResult.OK {
		return Info{}, core.E("gguf.ReadInfo", "open file", fileResult.Value.(error))
	}
	file := fileResult.Value.(*core.OSFile)
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		return Info{}, core.E("gguf.ReadInfo", "stat file", err)
	}

	reader := &countingReader{r: file}

	// Read and validate magic number.
	var magic uint32
	if err := binary.Read(reader, binary.LittleEndian, &magic); err != nil {
		return Info{}, core.E("gguf.ReadInfo", "reading magic", err)
	}
	if magic != ggufMagic {
		return Info{}, core.E("gguf.ReadInfo", core.Sprintf("invalid magic: 0x%08X (expected 0x%08X)", magic, ggufMagic), nil)
	}

	// Read version.
	var version uint32
	if err := binary.Read(reader, binary.LittleEndian, &version); err != nil {
		return Info{}, core.E("gguf.ReadInfo", "reading version", err)
	}
	if version < 2 || version > 3 {
		return Info{}, core.E("gguf.ReadInfo", core.Sprintf("unsupported GGUF version: %d", version), nil)
	}

	// Read tensor count and KV count. v3 uses uint64, v2 uses uint32.
	var tensorCount, kvCount uint64
	if version == 3 {
		if err := binary.Read(reader, binary.LittleEndian, &tensorCount); err != nil {
			return Info{}, core.E("gguf.ReadInfo", "reading tensor count", err)
		}
		if err := binary.Read(reader, binary.LittleEndian, &kvCount); err != nil {
			return Info{}, core.E("gguf.ReadInfo", "reading kv count", err)
		}
	} else {
		var tensorCount32, kvCount32 uint32
		if err := binary.Read(reader, binary.LittleEndian, &tensorCount32); err != nil {
			return Info{}, core.E("gguf.ReadInfo", "reading tensor count", err)
		}
		if err := binary.Read(reader, binary.LittleEndian, &kvCount32); err != nil {
			return Info{}, core.E("gguf.ReadInfo", "reading kv count", err)
		}
		tensorCount = uint64(tensorCount32)
		kvCount = uint64(kvCount32)
	}

	// Read all KV pairs. We store interesting keys and skip the rest.
	// Architecture-specific keys (e.g. llama.context_length) may appear before
	// the general.architecture key, so we collect all candidates and resolve after.
	var meta Metadata
	meta.FileSize = fileInfo.Size()
	alignment := uint32(defaultAlignment)

	// candidateContextLength and candidateBlockCount store values keyed by
	// their full key name (e.g. "llama.context_length") so we can match them
	// against the architecture once it is known.
	candidateContextLength := make(map[string]uint32)
	candidateBlockCount := make(map[string]uint32)

	for i := uint64(0); i < kvCount; i++ {
		key, err := readString(reader)
		if err != nil {
			return Info{}, core.E("gguf.ReadInfo", core.Sprintf("reading key %d", i), err)
		}

		var valType uint32
		if err := binary.Read(reader, binary.LittleEndian, &valType); err != nil {
			return Info{}, core.E("gguf.ReadInfo", core.Sprintf("reading value type for key %q", key), err)
		}

		// Check whether this is an interesting key before reading the value.
		switch {
		case key == "general.architecture":
			value, err := readTypedValue(reader, valType)
			if err != nil {
				return Info{}, core.E("gguf.ReadInfo", core.Sprintf("reading value for key %q", key), err)
			}
			if s, ok := value.(string); ok {
				meta.Architecture = s
			}

		case key == "general.name":
			value, err := readTypedValue(reader, valType)
			if err != nil {
				return Info{}, core.E("gguf.ReadInfo", core.Sprintf("reading value for key %q", key), err)
			}
			if s, ok := value.(string); ok {
				meta.Name = s
			}

		case key == "general.file_type":
			value, err := readTypedValue(reader, valType)
			if err != nil {
				return Info{}, core.E("gguf.ReadInfo", core.Sprintf("reading value for key %q", key), err)
			}
			if u, ok := value.(uint32); ok {
				meta.FileType = u
			}

		case key == "general.size_label":
			value, err := readTypedValue(reader, valType)
			if err != nil {
				return Info{}, core.E("gguf.ReadInfo", core.Sprintf("reading value for key %q", key), err)
			}
			if s, ok := value.(string); ok {
				meta.SizeLabel = s
			}

		case core.HasSuffix(key, ".context_length"):
			value, err := readTypedValue(reader, valType)
			if err != nil {
				return Info{}, core.E("gguf.ReadInfo", core.Sprintf("reading value for key %q", key), err)
			}
			if u, ok := value.(uint32); ok {
				candidateContextLength[key] = u
			}

		case core.HasSuffix(key, ".block_count"):
			value, err := readTypedValue(reader, valType)
			if err != nil {
				return Info{}, core.E("gguf.ReadInfo", core.Sprintf("reading value for key %q", key), err)
			}
			if u, ok := value.(uint32); ok {
				candidateBlockCount[key] = u
			}

		case key == "general.alignment":
			value, err := readTypedValue(reader, valType)
			if err != nil {
				return Info{}, core.E("gguf.ReadInfo", core.Sprintf("reading value for key %q", key), err)
			}
			if u, ok := value.(uint32); ok && u > 0 {
				alignment = u
			}

		default:
			// Skip uninteresting value.
			if err := skipValue(reader, valType); err != nil {
				return Info{}, core.E("gguf.ReadInfo", core.Sprintf("skipping value for key %q", key), err)
			}
		}
	}

	// Resolve architecture-specific keys.
	if meta.Architecture != "" {
		prefix := meta.Architecture + "."
		if v, ok := candidateContextLength[prefix+"context_length"]; ok {
			meta.ContextLength = v
		}
		if v, ok := candidateBlockCount[prefix+"block_count"]; ok {
			meta.BlockCount = v
		}
	}

	if !includeTensors {
		return Info{
			Metadata:  meta,
			Alignment: alignment,
		}, nil
	}

	tensors := make([]TensorInfo, 0, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		tensor, err := readTensorInfo(reader)
		if err != nil {
			return Info{}, core.E("gguf.ReadInfo", core.Sprintf("reading tensor %d", i), err)
		}
		tensors = append(tensors, tensor)
	}

	return Info{
		Metadata:   meta,
		Tensors:    tensors,
		Alignment:  alignment,
		DataOffset: alignOffset(reader.n, int64(alignment)),
	}, nil
}

// maxStringLength is a sanity limit for GGUF string values. No metadata string
// should ever approach 1 MiB; this prevents memory exhaustion from malformed files.
const maxStringLength = 1 << 20

type ggufFailure interface {
	Error() string
}

type countingReader struct {
	r io.Reader
	n int64
}

func (reader *countingReader) Read(p []byte) (int, error) {
	n, err := reader.r.Read(p)
	reader.n += int64(n)
	return n, err
}

// readString reads a GGUF string: uint64 length followed by that many bytes.
func readString(r io.Reader) (
	string,
	error,
) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > maxStringLength {
		return "", core.E("gguf.readString", core.Sprintf("string length %d exceeds maximum %d", length, maxStringLength), nil)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// readTypedValue reads a value of the given GGUF type and returns it as a Go
// value. String, uint32, and uint64 types return typed values (uint64 is
// downcast to uint32 when it fits). All others are read and discarded.
func readTypedValue(r io.Reader, valType uint32) (
	any,
	error,
) {
	switch valType {
	case typeString:
		return readString(r)
	case typeUint32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case typeUint64:
		var v uint64
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		if v <= math.MaxUint32 {
			return uint32(v), nil
		}
		return v, nil
	default:
		// Read and discard the value, returning nil.
		err := skipValue(r, valType)
		return nil, err
	}
}

// skipValue reads and discards a GGUF value of the given type from r.
func skipValue(r io.Reader, valType uint32) ggufFailure {
	switch valType {
	case typeUint8, typeInt8, typeBool:
		_, err := discardBytes(r, 1)
		return err
	case typeUint16, typeInt16:
		_, err := discardBytes(r, 2)
		return err
	case typeUint32, typeInt32, typeFloat32:
		_, err := discardBytes(r, 4)
		return err
	case typeUint64, typeInt64, typeFloat64:
		_, err := discardBytes(r, 8)
		return err
	case typeString:
		var length uint64
		if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
			return err
		}
		if length > maxStringLength {
			return core.E("gguf.skipValue", core.Sprintf("string length %d exceeds maximum %d", length, maxStringLength), nil)
		}
		_, err := discardBytes(r, int64(length))
		return err
	case typeArray:
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return err
		}
		var count uint64
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return err
		}
		for i := uint64(0); i < count; i++ {
			if err := skipValue(r, elemType); err != nil {
				return err
			}
		}
		return nil
	default:
		return core.E("gguf.skipValue", core.Sprintf("unknown GGUF value type: %d", valType), nil)
	}
}

// discardBytes reads and discards exactly n bytes from r.
func discardBytes(r io.Reader, n int64) (
	int64,
	error,
) {
	return io.CopyN(io.Discard, r, n)
}

func readTensorInfo(r io.Reader) (TensorInfo, error) {
	name, err := readString(r)
	if err != nil {
		return TensorInfo{}, err
	}
	var dimensionCount uint32
	if err := binary.Read(r, binary.LittleEndian, &dimensionCount); err != nil {
		return TensorInfo{}, err
	}
	if dimensionCount > 8 {
		return TensorInfo{}, core.E("gguf.readTensorInfo", core.Sprintf("tensor %q has %d dimensions", name, dimensionCount), nil)
	}
	dimensions := make([]uint64, dimensionCount)
	for i := range dimensions {
		if err := binary.Read(r, binary.LittleEndian, &dimensions[i]); err != nil {
			return TensorInfo{}, err
		}
	}
	var tensorType uint32
	if err := binary.Read(r, binary.LittleEndian, &tensorType); err != nil {
		return TensorInfo{}, err
	}
	var offset uint64
	if err := binary.Read(r, binary.LittleEndian, &offset); err != nil {
		return TensorInfo{}, err
	}
	byteSize, err := TensorByteSize(tensorType, dimensions)
	if err != nil {
		return TensorInfo{}, err
	}
	return TensorInfo{
		Name:       name,
		Dimensions: dimensions,
		Type:       tensorType,
		TypeName:   TensorTypeName(tensorType),
		Offset:     offset,
		ByteSize:   byteSize,
	}, nil
}

// TensorByteSize returns the number of bytes occupied by a GGML tensor type.
func TensorByteSize(tensorType uint32, dimensions []uint64) (uint64, error) {
	elements, err := tensorElementCount(dimensions)
	if err != nil {
		return 0, err
	}
	blockSize, typeSize, ok := tensorBlockSize(tensorType)
	if !ok {
		return 0, core.E("gguf.TensorByteSize", core.Sprintf("unsupported GGUF tensor type: %d", tensorType), nil)
	}
	blocks := (elements + blockSize - 1) / blockSize
	if blocks > math.MaxUint64/typeSize {
		return 0, core.E("gguf.TensorByteSize", "tensor byte size overflows uint64", nil)
	}
	return blocks * typeSize, nil
}

func tensorElementCount(dimensions []uint64) (uint64, error) {
	if len(dimensions) == 0 {
		return 0, core.E("gguf.tensorElementCount", "tensor has no dimensions", nil)
	}
	elements := uint64(1)
	for _, dimension := range dimensions {
		if dimension == 0 {
			return 0, core.E("gguf.tensorElementCount", "tensor has a zero dimension", nil)
		}
		if elements > math.MaxUint64/dimension {
			return 0, core.E("gguf.tensorElementCount", "tensor element count overflows uint64", nil)
		}
		elements *= dimension
	}
	return elements, nil
}

func tensorBlockSize(tensorType uint32) (blockSize, typeSize uint64, ok bool) {
	switch tensorType {
	case 0:
		return 1, 4, true
	case 1, 30:
		return 1, 2, true
	case 2:
		return 32, 18, true
	case 3:
		return 32, 20, true
	case 6:
		return 32, 22, true
	case 7:
		return 32, 24, true
	case 8:
		return 32, 34, true
	case 10:
		return 256, 84, true
	case 11:
		return 256, 110, true
	case 12:
		return 256, 144, true
	case 13:
		return 256, 176, true
	case 14:
		return 256, 210, true
	case 15:
		return 256, 292, true
	case 24:
		return 1, 1, true
	case 25:
		return 1, 2, true
	case 26:
		return 1, 4, true
	case 27, 28:
		return 1, 8, true
	default:
		return 0, 0, false
	}
}

func alignOffset(offset, alignment int64) int64 {
	if alignment <= 0 {
		alignment = defaultAlignment
	}
	remainder := offset % alignment
	if remainder == 0 {
		return offset
	}
	return offset + alignment - remainder
}
