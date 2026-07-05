// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"io"
	"math"

	core "dappco.re/go"
)

// parseGGUFWithDataStart reads the header + tensor directory and also reports
// dataStart — the aligned file offset where the tensor-data section begins
// (each TensorInfo.Offset is relative to it). LoadTensors needs it to bind
// mmapped tensor bytes; the byte tally comes from ggufCountingReader. Moved
// with the tensor-loading layer from go-mlx gguf during the engine merge.
func parseGGUFWithDataStart(path string) (map[string]any, []TensorInfo, uint64, error) {
	open := core.Open(path)
	if !open.OK {
		return nil, nil, 0, core.Errorf("gguf: open gguf: %w", open.Value.(error))
	}
	file := open.Value.(*core.OSFile)
	defer file.Close()

	// Wrap in a buffered reader — parseGGUF does hundreds of small fixed-
	// width reads (8 / 4 / 12 bytes) per metadata entry + tensor. Without
	// buffering each becomes its own syscall; with bufio (default 4 KiB)
	// the read syscalls collapse to a handful for typical GGUF headers.
	// The counting wrapper tallies bytes so dataStart can be derived below.
	reader := &ggufCountingReader{r: core.NewBufReader(file)}

	// Shared scratch buffer used for the file header, every fixed-width
	// metadata/tensor read, and short string reads (interned-key fast
	// path). 64 B covers all known GGUF metadata keys + the bounded
	// architecture-name vocabulary; longer strings fall through to per-
	// call make. Declaring it once at the top of parseGGUF means
	// io.ReadFull's interface-typed buf parameter forces a single per-
	// call heap escape rather than one per read site (header + trailer
	// each used to allocate their own [N]byte locals).
	var scratch [64]byte

	// First 24 bytes: magic(4) + version(4) + tensorCount(8) + metadataCount(8).
	// Reflect-free read — eliminates 4 binary.Read calls (+4 reflect allocs each).
	if _, err := io.ReadFull(reader, scratch[:24]); err != nil {
		return nil, nil, 0, core.Errorf("gguf: read gguf header: %w", err)
	}
	if core.AsString(scratch[:4]) != "GGUF" {
		return nil, nil, 0, errGGUFInvalidMagic
	}
	version := binary.LittleEndian.Uint32(scratch[4:8])
	if version < 2 {
		return nil, nil, 0, core.Errorf("gguf: unsupported gguf version %d", version)
	}
	tensorCount := binary.LittleEndian.Uint64(scratch[8:16])
	metadataCount := binary.LittleEndian.Uint64(scratch[16:24])
	if tensorCount > maxGGUFCollectionEntries {
		return nil, nil, 0, core.Errorf("gguf: gguf tensor count %d exceeds limit %d", tensorCount, maxGGUFCollectionEntries)
	}
	if metadataCount > maxGGUFCollectionEntries {
		return nil, nil, 0, core.Errorf("gguf: gguf metadata count %d exceeds limit %d", metadataCount, maxGGUFCollectionEntries)
	}

	metadata := make(map[string]any, int(metadataCount))
	// Key arena — most metadata keys hit ggufInternedStrings (zero alloc),
	// but unknown / synthetic / future keys still allocate a fresh string
	// each. Bump-allocating into a per-call slab amortises the miss cost.
	// Sized at 48 B/entry — long-tail tokenizer.* keys peak around 40 B.
	keyArena := make([]byte, 0, int(metadataCount)*48)
	// Value-string arena — string-typed metadata values land here.
	// Sized at 56 B/entry; real-world values (tokenizer names, version
	// strings, descriptions) cluster under 48 B. Lifetime is tied to
	// the metadata map / Info via Go's GC: any string-view that escapes
	// into Info keeps the arena live until that Info is dropped.
	valueArena := make([]byte, 0, int(metadataCount)*56)
	for range metadataCount {
		key, err := readStringIntoArena(reader, scratch[:], &keyArena)
		if err != nil {
			return nil, nil, 0, core.Errorf("gguf: read gguf metadata key: %w", err)
		}
		if _, err := io.ReadFull(reader, scratch[:4]); err != nil {
			return nil, nil, 0, core.Errorf("gguf: read gguf metadata type: %w", err)
		}
		valueType := binary.LittleEndian.Uint32(scratch[:4])
		value, err := readGGUFValue(reader, valueType, scratch[:], &valueArena)
		if err != nil {
			return nil, nil, 0, core.Errorf("gguf: read gguf metadata value for %q: %w", key, err)
		}
		metadata[key] = value
	}

	// Build the public TensorInfo slice directly — there is no separate
	// internal tensor struct any more. parseGGUF fills only the base
	// fields (Name/Type/Shape/Offset) read straight off the wire; the
	// derived fields (TypeName/DType/Bits/BlockSize/Elements/Quantized)
	// are filled in place by buildGGUFTensorInfos.
	tensors := make([]TensorInfo, tensorCount)
	// Shape arena — bump-allocate per-tensor shapes from a single slab
	// instead of one `make([]uint64, ndim)` per tensor. Real GGUF tensors
	// run 1-4 dims (rank-2 weights dominate); 4 is a safe initial budget.
	// Overflow falls back to per-tensor make so the arena never reallocates
	// (which would invalidate already-handed-out slice headers).
	shapeArena := make([]uint64, 0, int(tensorCount)*4)
	// Name arena — bump-allocate per-tensor name bytes from a single slab,
	// then hand out zero-copy core.AsString views. Real GGUF tensor names
	// are 12-30 chars (`blk.<N>.<component>.<weight|bias>`); 40 B/tensor
	// covers the long end with headroom. Overflow falls back to per-
	// tensor make. The arena MUST NOT be appended-past-capacity once any
	// view has been handed out — string views alias the backing array,
	// so a re-allocation would dangle every prior name.
	nameArena := make([]byte, 0, int(tensorCount)*40)
	for i := range tensorCount {
		name, err := readStringIntoArena(reader, scratch[:], &nameArena)
		if err != nil {
			return nil, nil, 0, core.Errorf("gguf: read gguf tensor name: %w", err)
		}
		if _, err := io.ReadFull(reader, scratch[:4]); err != nil {
			return nil, nil, 0, core.Errorf("gguf: read gguf tensor ndim: %w", err)
		}
		ndim := binary.LittleEndian.Uint32(scratch[:4])
		var shape []uint64
		if remaining := cap(shapeArena) - len(shapeArena); int(ndim) <= remaining {
			start := len(shapeArena)
			end := start + int(ndim)
			shapeArena = shapeArena[:end]
			// Three-index slice caps the per-tensor view at exactly `ndim`
			// elements so any future append on this Shape can't bleed into
			// the next tensor's region of the arena.
			shape = shapeArena[start:end:end]
		} else {
			shape = make([]uint64, ndim)
		}
		for d := range ndim {
			if _, err := io.ReadFull(reader, scratch[:8]); err != nil {
				return nil, nil, 0, core.Errorf("gguf: read gguf tensor dimension: %w", err)
			}
			shape[d] = binary.LittleEndian.Uint64(scratch[:8])
		}
		// tensorType(4) + offset(8) = 12 bytes in one read. Reuse the
		// per-call `scratch` arena rather than declaring a per-tensor
		// `[12]byte` local — io.ReadFull's interface-typed `buf` argument
		// would force every iteration's local to escape, costing one
		// heap alloc per tensor (~200 on a qwen3-class model).
		if _, err := io.ReadFull(reader, scratch[:12]); err != nil {
			return nil, nil, 0, core.Errorf("gguf: read gguf tensor type/offset: %w", err)
		}
		tensors[i] = TensorInfo{
			Name:   name,
			Type:   binary.LittleEndian.Uint32(scratch[:4]),
			Shape:  shape,
			Offset: binary.LittleEndian.Uint64(scratch[4:12]),
		}
	}

	// The reader now sits exactly at the end of the tensor-info table; the
	// tensor-data section starts at the next general.alignment boundary.
	headerEnd := reader.n
	dataStart := headerEnd + alignPadding(headerEnd, ggufDataAlignment(metadata))
	return metadata, tensors, dataStart, nil
}

// parseGGUF reads the header + tensor directory, discarding the data-section
// offset. Metadata / Info callers that never touch tensor bytes use this thin
// wrapper; LoadTensors calls parseGGUFWithDataStart directly for the offset.
func parseGGUF(path string) (map[string]any, []TensorInfo, error) {
	metadata, tensors, _, err := parseGGUFWithDataStart(path)
	return metadata, tensors, err
}

// ggufCountingReader wraps an io.Reader and tallies bytes read, so the parse
// can report where the header ends (and thus where the aligned tensor-data
// section begins). Moved with the tensor-loading layer from go-mlx gguf.
type ggufCountingReader struct {
	r io.Reader
	n uint64
}

func (r *ggufCountingReader) Read(p []byte) (int, error) {
	n, err := r.r.Read(p)
	r.n += uint64(n)
	return n, err
}

// ggufInternedStrings — singleton mappings for high-frequency GGUF metadata
// keys + bounded-vocabulary string values (architecture names). Map lookup
// via m[string(b)] uses Go's runtime []byte→string fast path that skips
// the conversion alloc; on hit we return the singleton, on miss we fall
// through to the normal allocate-and-convert path.
//
// Real GGUF metadata keys peak around 32 B (tokenizer.ggml.* family is the
// long end). The 64 B short-string threshold in readGGUFString comfortably
// covers all interned entries.
var ggufInternedStrings = map[string]string{
	// general.* — present in every well-formed GGUF.
	"general.architecture":            "general.architecture",
	"general.name":                    "general.name",
	"general.author":                  "general.author",
	"general.version":                 "general.version",
	"general.url":                     "general.url",
	"general.description":             "general.description",
	"general.license":                 "general.license",
	"general.file_type":               "general.file_type",
	"general.quantization_version":    "general.quantization_version",
	"general.quantization_type":       "general.quantization_type",
	"general.quantization":            "general.quantization",
	"general.quantization_group_size": "general.quantization_group_size",
	"general.alignment":               "general.alignment",
	"quantization.type":               "quantization.type",
	"quantization.name":               "quantization.name",
	"quantization.group_size":         "quantization.group_size",
	// Common architecture *.block_count / *.context_length / *.embedding_length —
	// pre-prefixed per known model family.
	"qwen3.block_count":       "qwen3.block_count",
	"qwen3.context_length":    "qwen3.context_length",
	"qwen3.embedding_length":  "qwen3.embedding_length",
	"qwen3.vocab_size":        "qwen3.vocab_size",
	"qwen2.block_count":       "qwen2.block_count",
	"qwen2.context_length":    "qwen2.context_length",
	"qwen2.embedding_length":  "qwen2.embedding_length",
	"llama.block_count":       "llama.block_count",
	"llama.context_length":    "llama.context_length",
	"llama.embedding_length":  "llama.embedding_length",
	"llama.vocab_size":        "llama.vocab_size",
	"gemma3.block_count":      "gemma3.block_count",
	"gemma3.context_length":   "gemma3.context_length",
	"gemma3.embedding_length": "gemma3.embedding_length",
	"gemma3.vocab_size":       "gemma3.vocab_size",
	"gemma2.block_count":      "gemma2.block_count",
	"phi.block_count":         "phi.block_count",
	"mistral.block_count":     "mistral.block_count",
	"mixtral.block_count":     "mixtral.block_count",
	"bert.block_count":        "bert.block_count",
	// Bounded-vocabulary architecture-name values.
	"qwen3":   "qwen3",
	"qwen2":   "qwen2",
	"llama":   "llama",
	"gemma3":  "gemma3",
	"gemma2":  "gemma2",
	"mistral": "mistral",
	"mixtral": "mixtral",
	"phi":     "phi",
	"bert":    "bert",
}

// readStringIntoArena reads a length-prefixed string and parks the bytes
// in the supplied arena, returning a zero-copy string view. Used for
// short-lived bulk strings (tensor names, metadata keys) where the
// caller wants to amortise allocations across many reads.
//
// First tries ggufInternedStrings for the singleton fast path. If the
// name would push the arena past its reserved capacity, falls back to
// a fresh per-call copy so the existing arena views stay valid.
func readStringIntoArena(reader io.Reader, scratch []byte, arena *[]byte) (string, error) {
	if _, err := io.ReadFull(reader, scratch[:8]); err != nil {
		return "", err
	}
	length := binary.LittleEndian.Uint64(scratch[:8])
	if length > 16<<20 {
		return "", errGGUFStringTooLong
	}
	if length == 0 {
		return "", nil
	}
	buf := *arena
	remaining := cap(buf) - len(buf)
	if int(length) > remaining {
		// Arena overflow: copy through scratch when possible (short
		// strings still hit the intern map); else fresh make.
		if uint64(len(scratch)) >= length {
			if _, err := io.ReadFull(reader, scratch[:length]); err != nil {
				return "", err
			}
			if interned, ok := ggufInternedStrings[string(scratch[:length])]; ok {
				return interned, nil
			}
			return string(scratch[:length]), nil
		}
		dst := make([]byte, length)
		if _, err := io.ReadFull(reader, dst); err != nil {
			return "", err
		}
		return core.AsString(dst), nil
	}
	start := len(buf)
	end := start + int(length)
	buf = buf[:end]
	if _, err := io.ReadFull(reader, buf[start:end]); err != nil {
		return "", err
	}
	// Intern probe — singleton hit means we don't need the arena slot.
	// Roll back the cursor so future calls can reuse the space.
	if interned, ok := ggufInternedStrings[string(buf[start:end])]; ok {
		*arena = buf[:start]
		return interned, nil
	}
	*arena = buf
	return core.AsString(buf[start:end]), nil
}

// readGGUFString reads a length-prefixed string into a fresh []byte.
// `scratch` must be at least 8 bytes — used to decode the uint64 length
// without a reflect.Read alloc. When `scratch` is large enough (≥ length),
// short strings are read into it and checked against ggufInternedStrings;
// interned hits return the singleton with zero per-call heap allocation.
func readGGUFString(reader io.Reader, scratch []byte) (string, error) {
	if _, err := io.ReadFull(reader, scratch[:8]); err != nil {
		return "", err
	}
	length := binary.LittleEndian.Uint64(scratch[:8])
	if length > 16<<20 {
		return "", errGGUFStringTooLong
	}
	if length == 0 {
		return "", nil
	}
	if uint64(len(scratch)) >= length {
		// Caller provided a buffer big enough — read into it and try the
		// intern map. Map lookup uses m[string(slice)] fast path that
		// avoids the per-call conversion alloc; on hit, return the static
		// singleton (zero alloc). On miss, fall back to a heap copy via
		// string() conversion (one alloc, same as the make path below).
		if _, err := io.ReadFull(reader, scratch[:length]); err != nil {
			return "", err
		}
		if interned, ok := ggufInternedStrings[string(scratch[:length])]; ok {
			return interned, nil
		}
		return string(scratch[:length]), nil
	}
	buffer := make([]byte, length)
	if _, err := io.ReadFull(reader, buffer); err != nil {
		return "", err
	}
	// Zero-copy: buffer is freshly built and only the returned string
	// references it — no aliasing risk.
	return core.AsString(buffer), nil
}

// ggufStringArrayLen is a GGUF string-element array parsed for its length
// only — the elements were skipped (see readGGUFValue). ReadInfo needs just
// the count (vocab size); materialising a 200k-token vocab is wasted work it
// immediately discards. metadataArrayLen reports the count.
type ggufStringArrayLen int

// skipGGUFString reads a GGUF string's [uint64 length][bytes] and discards the
// bytes through the shared scratch buffer (zero allocation), advancing reader
// past the string. Used when only the array element COUNT is needed.
func skipGGUFString(reader io.Reader, scratch []byte) error {
	if _, err := io.ReadFull(reader, scratch[:8]); err != nil {
		return err
	}
	length := binary.LittleEndian.Uint64(scratch[:8])
	if length > 16<<20 {
		return errGGUFStringTooLong
	}
	for length > 0 {
		n := uint64(len(scratch))
		if n > length {
			n = length
		}
		if _, err := io.ReadFull(reader, scratch[:n]); err != nil {
			return err
		}
		length -= n
	}
	return nil
}

func readGGUFValue(reader io.Reader, valueType uint32, scratch []byte, strArena *[]byte) (any, error) {
	switch valueType {
	case ggufValueTypeUint8:
		if _, err := io.ReadFull(reader, scratch[:1]); err != nil {
			return uint8(0), err
		}
		return scratch[0], nil
	case ggufValueTypeInt8:
		if _, err := io.ReadFull(reader, scratch[:1]); err != nil {
			return int8(0), err
		}
		return int8(scratch[0]), nil
	case ggufValueTypeUint16:
		if _, err := io.ReadFull(reader, scratch[:2]); err != nil {
			return uint16(0), err
		}
		return binary.LittleEndian.Uint16(scratch[:2]), nil
	case ggufValueTypeInt16:
		if _, err := io.ReadFull(reader, scratch[:2]); err != nil {
			return int16(0), err
		}
		return int16(binary.LittleEndian.Uint16(scratch[:2])), nil
	case ValueTypeUint32:
		if _, err := io.ReadFull(reader, scratch[:4]); err != nil {
			return uint32(0), err
		}
		return binary.LittleEndian.Uint32(scratch[:4]), nil
	case ggufValueTypeInt32:
		if _, err := io.ReadFull(reader, scratch[:4]); err != nil {
			return int32(0), err
		}
		return int32(binary.LittleEndian.Uint32(scratch[:4])), nil
	case ValueTypeFloat32:
		if _, err := io.ReadFull(reader, scratch[:4]); err != nil {
			return float32(0), err
		}
		return math.Float32frombits(binary.LittleEndian.Uint32(scratch[:4])), nil
	case ggufValueTypeBool:
		if _, err := io.ReadFull(reader, scratch[:1]); err != nil {
			return false, err
		}
		return scratch[0] != 0, nil
	case ValueTypeString:
		if strArena != nil {
			return readStringIntoArena(reader, scratch, strArena)
		}
		return readGGUFString(reader, scratch)
	case ggufValueTypeArray:
		if _, err := io.ReadFull(reader, scratch[:4]); err != nil {
			return nil, err
		}
		elementType := binary.LittleEndian.Uint32(scratch[:4])
		if _, err := io.ReadFull(reader, scratch[:8]); err != nil {
			return nil, err
		}
		length := binary.LittleEndian.Uint64(scratch[:8])
		if length > maxGGUFCollectionEntries {
			return nil, core.Errorf("gguf array length %d exceeds limit %d", length, maxGGUFCollectionEntries)
		}
		// String-element arrays (the 200k+ entry tokenizer.ggml.tokens vocab
		// dominates header-parse cost) are parsed for their COUNT only.
		// parseGGUF feeds ReadInfo, which reads this array exclusively through
		// metadataArrayLen (vocab size) — the token strings are never read. So
		// skip the element bytes rather than materialising every token (a 200k
		// vocab was ~200k allocs, all immediately discarded) and return the
		// count as ggufStringArrayLen, which metadataArrayLen understands.
		if elementType == ValueTypeString {
			for range length {
				if err := skipGGUFString(reader, scratch); err != nil {
					return nil, err
				}
			}
			return ggufStringArrayLen(length), nil
		}
		values := make([]any, length)
		for i := range length {
			value, err := readGGUFValue(reader, elementType, scratch, strArena)
			if err != nil {
				return nil, err
			}
			values[i] = value
		}
		return values, nil
	case ggufValueTypeUint64:
		if _, err := io.ReadFull(reader, scratch[:8]); err != nil {
			return uint64(0), err
		}
		return binary.LittleEndian.Uint64(scratch[:8]), nil
	case ggufValueTypeInt64:
		if _, err := io.ReadFull(reader, scratch[:8]); err != nil {
			return int64(0), err
		}
		return int64(binary.LittleEndian.Uint64(scratch[:8])), nil
	case ggufValueTypeFloat64:
		if _, err := io.ReadFull(reader, scratch[:8]); err != nil {
			return float64(0), err
		}
		return math.Float64frombits(binary.LittleEndian.Uint64(scratch[:8])), nil
	default:
		return nil, core.Errorf("unsupported gguf metadata type %d", valueType)
	}
}
