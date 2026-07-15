// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"io"

	core "dappco.re/go"
)

// Metadata returns a .gguf file's key/value metadata map without loading
// any tensor data. Values arrive as the parser's native Go types (string,
// bool, uint32/uint64/int32/int64, float32/float64, []any) — callers
// coerce per key.
//
//	meta, err := gguf.Metadata("/models/gemma-4-31B-it-Q8_0.gguf")
//	arch, _ := meta["general.architecture"].(string)
func Metadata(path string) (map[string]any, error) {
	meta, _, err := parseGGUF(path)
	return meta, err
}

// MetadataSubset reads the metadata entries whose key satisfies keep and
// returns them alongside the header's tensor count, without parsing the
// tensor directory. The value bytes of every non-matching entry are skipped
// in place — no decode, no map insert — which keeps a few-well-known-keys
// probe over a vocab-heavy header (hundreds of tokenizer entries) to a
// handful of allocations. The root inference package's ReadGGUFInfo is the
// canonical caller; its alloc-budget test gates this path.
//
// keep receives a transient view of the key that is valid only for the
// duration of the callback — clone before storing it anywhere.
//
//	meta, tensorCount, err := gguf.MetadataSubset(path, func(key string) bool {
//	    return key == "general.architecture" || core.HasSuffix(key, ".block_count")
//	})
func MetadataSubset(modelPath string, keep func(key string) bool) (map[string]any, int, error) {
	path, err := resolveGGUFFile(modelPath)
	if err != nil {
		return nil, 0, err
	}
	open := core.Open(path)
	if !open.OK {
		return nil, 0, core.Errorf("gguf: open gguf: %w", open.Value.(error))
	}
	file := open.Value.(*core.OSFile)
	defer file.Close()

	// Buffered reader for the same reason parseGGUF uses one: the metadata
	// loop is hundreds of small fixed-width reads, and skipped values are
	// discarded straight out of the buffer (zero syscall for small values).
	reader := core.NewBufReader(file)

	var scratch [64]byte
	if _, err := io.ReadFull(reader, scratch[:24]); err != nil {
		return nil, 0, core.Errorf("gguf: read gguf header: %w", err)
	}
	if core.AsString(scratch[:4]) != "GGUF" {
		return nil, 0, errGGUFInvalidMagic
	}
	version := binary.LittleEndian.Uint32(scratch[4:8])
	if version < 2 {
		return nil, 0, core.Errorf("gguf: unsupported gguf version %d", version)
	}
	tensorCount := binary.LittleEndian.Uint64(scratch[8:16])
	metadataCount := binary.LittleEndian.Uint64(scratch[16:24])
	if tensorCount > maxGGUFCollectionEntries {
		return nil, 0, core.Errorf("gguf: gguf tensor count %d exceeds limit %d", tensorCount, maxGGUFCollectionEntries)
	}
	if metadataCount > maxGGUFCollectionEntries {
		return nil, 0, core.Errorf("gguf: gguf metadata count %d exceeds limit %d", metadataCount, maxGGUFCollectionEntries)
	}

	// Sized for the kept subset, not the header count — callers keep a
	// handful of keys out of hundreds.
	metadata := make(map[string]any, 8)
	var keyBuf []byte
	for range metadataCount {
		keyView, err := readGGUFKeyView(reader, scratch[:8], &keyBuf)
		if err != nil {
			return nil, 0, core.Errorf("gguf: read gguf metadata key: %w", err)
		}
		if _, err := io.ReadFull(reader, scratch[:4]); err != nil {
			return nil, 0, core.Errorf("gguf: read gguf metadata type: %w", err)
		}
		valueType := binary.LittleEndian.Uint32(scratch[:4])
		if !keep(keyView) {
			if err := skipGGUFValue(reader, valueType, scratch[:]); err != nil {
				return nil, 0, err
			}
			continue
		}
		// The kept key outlives keyBuf's next reuse — intern well-known
		// keys (zero alloc), clone the rest to detach from the buffer.
		key, interned := ggufInternedStrings[keyView]
		if !interned {
			key = core.Clone(keyView)
		}
		value, err := readGGUFValue(reader, valueType, scratch[:], nil)
		if err != nil {
			return nil, 0, core.Errorf("gguf: read gguf metadata value for %q: %w", key, err)
		}
		metadata[key] = value
	}
	return metadata, int(tensorCount), nil
}

// readGGUFKeyView reads the next length-prefixed key into a caller-owned
// reusable buffer and returns a zero-copy string view aliasing it. The view
// is valid only until the next call reusing the same buffer; callers clone
// (or intern) before storing the key beyond the loop body.
func readGGUFKeyView(reader io.Reader, scratch []byte, keyBuf *[]byte) (string, error) {
	if _, err := io.ReadFull(reader, scratch[:8]); err != nil {
		return "", core.Errorf("gguf: read gguf string length: %w", err)
	}
	length := binary.LittleEndian.Uint64(scratch[:8])
	if length > 16<<20 {
		return "", errGGUFStringTooLong
	}
	if uint64(cap(*keyBuf)) < length {
		*keyBuf = make([]byte, length)
	} else {
		*keyBuf = (*keyBuf)[:length]
	}
	if _, err := io.ReadFull(reader, *keyBuf); err != nil {
		return "", core.Errorf("gguf: read gguf string: %w", err)
	}
	return core.AsString(*keyBuf), nil
}

// skipGGUFValue advances reader past one metadata value of the given wire
// type without decoding or allocating. Fixed-width scalars and string bytes
// are discarded through the buffered reader (served from the buffer when the
// bytes are present — zero syscall); arrays discard fixed-width elements as
// one sized skip and walk string/nested elements one at a time.
func skipGGUFValue(reader *core.BufReader, valueType uint32, scratch []byte) error {
	if size := ggufValueFixedSize(valueType); size > 0 {
		return discardGGUFBytes(reader, uint64(size))
	}
	switch valueType {
	case ValueTypeString:
		if _, err := io.ReadFull(reader, scratch[:8]); err != nil {
			return core.Errorf("gguf: read gguf string length: %w", err)
		}
		length := binary.LittleEndian.Uint64(scratch[:8])
		if length > 16<<20 {
			return errGGUFStringTooLong
		}
		return discardGGUFBytes(reader, length)
	case ValueTypeArray:
		if _, err := io.ReadFull(reader, scratch[:4]); err != nil {
			return core.Errorf("gguf: read gguf array element type: %w", err)
		}
		elementType := binary.LittleEndian.Uint32(scratch[:4])
		if _, err := io.ReadFull(reader, scratch[:8]); err != nil {
			return core.Errorf("gguf: read gguf array length: %w", err)
		}
		length := binary.LittleEndian.Uint64(scratch[:8])
		if length > maxGGUFCollectionEntries {
			return core.Errorf("gguf: gguf array length %d exceeds limit %d", length, maxGGUFCollectionEntries)
		}
		if size := ggufValueFixedSize(elementType); size > 0 {
			// length ≤ maxGGUFCollectionEntries (2^20) and size ≤ 8, so
			// the product stays far below uint64 overflow.
			return discardGGUFBytes(reader, length*uint64(size))
		}
		for range length {
			if err := skipGGUFValue(reader, elementType, scratch); err != nil {
				return err
			}
		}
		return nil
	default:
		return core.Errorf("gguf: unsupported gguf metadata type: %d", valueType)
	}
}

// ggufValueFixedSize returns the on-wire byte width of a fixed-width GGUF
// scalar value type, or 0 for variable-width types (string, array) and
// unknown ids.
func ggufValueFixedSize(valueType uint32) int {
	switch valueType {
	case ggufValueTypeUint8, ggufValueTypeInt8, ValueTypeBool:
		return 1
	case ggufValueTypeUint16, ggufValueTypeInt16:
		return 2
	case ValueTypeUint32, ggufValueTypeInt32, ValueTypeFloat32:
		return 4
	case ggufValueTypeUint64, ggufValueTypeInt64, ggufValueTypeFloat64:
		return 8
	default:
		return 0
	}
}

// discardGGUFBytes drops exactly n bytes from reader — served from the
// buffer when the bytes are already there, streaming reads when not.
func discardGGUFBytes(reader *core.BufReader, n uint64) error {
	for n > 0 {
		chunk := n
		const maxChunk = 1 << 30
		if chunk > maxChunk {
			chunk = maxChunk
		}
		discarded, err := reader.Discard(int(chunk))
		n -= uint64(discarded)
		if err != nil {
			return core.Errorf("gguf: discard gguf value bytes: %w", err)
		}
	}
	return nil
}
