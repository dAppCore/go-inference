// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"context"
	"encoding/binary"

	core "dappco.re/go"
)

const defaultRawChunkBytes = 4 << 20

// Sentinel errors hoisted to package vars (see W9-Y + W10-R lifts).
// These fire on validation paths inside WriteSubset / writeAll; static
// message text means they're safe to share by pointer across callers
// and avoid the per-fire core.NewError alloc.
var (
	errSubsetPathEmpty       = core.NewError("mlx: safetensors subset path is empty")
	errSubsetNoTensors       = core.NewError("mlx: safetensors subset requires at least one tensor")
	errSubsetTensorNameEmpty = core.NewError("mlx: safetensors subset tensor name is empty")
	errWriteNoProgress       = core.NewError("mlx: safetensors write made no progress")
)

// WriteSubset writes a safetensors file containing refs without loading all
// selected tensors into memory. Tensor payloads are copied directly from the
// indexed source files in bounded chunks.
func WriteSubset(ctx context.Context, path string, refs []TensorRef) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if core.Trim(path) == "" {
		return errSubsetPathEmpty
	}
	if len(refs) == 0 {
		return errSubsetNoTensors
	}

	ordered, headerBytes, err := subsetHeaderEncoded(refs)
	if err != nil {
		return err
	}

	parent := core.PathDir(path)
	if result := core.MkdirAll(parent, 0o755); !result.OK {
		return resultError(result)
	}
	created := core.OpenFile(path, core.O_CREATE|core.O_WRONLY|core.O_TRUNC, 0o644)
	if !created.OK {
		return resultError(created)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	var headerLen [8]byte
	binary.LittleEndian.PutUint64(headerLen[:], uint64(len(headerBytes)))
	if err := writeAll(file, headerLen[:]); err != nil {
		return err
	}
	if err := writeAll(file, headerBytes); err != nil {
		return err
	}
	// Reuse a single byte buffer across every per-ref chunked copy.
	// writeRefRawChunks previously allocated its own buffer per call,
	// so a subset of N tensors meant N small-or-large allocations.
	// Each ref's payload size is capped by chunkBytes anyway, so
	// reuse is safe — the buffer is grown on demand by passing
	// through writeRefRawChunksScratch.
	var scratch []byte
	for _, ref := range ordered {
		if err := ctx.Err(); err != nil {
			return err
		}
		scratch, err = writeRefRawChunksScratch(ctx, file, ref, defaultRawChunkBytes, scratch)
		if err != nil {
			return err
		}
	}
	return nil
}

// subsetHeaderEncoded validates the supplied refs, sorts them by name,
// and emits the safetensors JSON header bytes directly. This replaces
// the previous flow (build a map[string]HeaderEntry + Shape/DataOffsets
// slices, then core.JSONMarshal it) — the reflection-driven encoder was
// allocating per-entry struct fields, per-key string conversions and a
// growable bytes.Buffer internally. The hand-rolled emitter writes into
// a single appended buffer that is sized up-front.
//
// Output is bit-exact identical to core.JSONMarshal(map[string]HeaderEntry)
// for any valid input: map keys come out sorted alphabetically, struct
// fields emit in declaration order (dtype, shape, data_offsets), and
// integer values use the same base-10 form. The parity test
// TestParseHeader_Parity_Synthetic round-trips through ReadIndex and
// would fail on any format drift.
func subsetHeaderEncoded(refs []TensorRef) ([]TensorRef, []byte, error) {
	byName := make(map[string]TensorRef, len(refs))
	names := make([]string, 0, len(refs))
	for _, ref := range refs {
		if core.Trim(ref.Name) == "" {
			return nil, nil, errSubsetTensorNameEmpty
		}
		if ref.ByteLen < 0 {
			return nil, nil, core.NewError("mlx: safetensors subset tensor byte length is invalid: " + ref.Name)
		}
		if _, ok := byName[ref.Name]; ok {
			return nil, nil, core.NewError("mlx: safetensors subset contains duplicate tensor: " + ref.Name)
		}
		byName[ref.Name] = ref
		names = append(names, ref.Name)
	}
	core.SliceSort(names)

	// Size the output buffer up-front. Per entry we write at minimum:
	//   "name":{"dtype":"XX","shape":[],"data_offsets":[0,0]},
	// which is roughly 50 bytes plus the name, dtype, and integer
	// widths. Use 80 + name + 16*dims + 40 (offsets) as a conservative
	// upper bound — undersize only causes one extra append-grow which is
	// fine; oversize wastes a handful of bytes.
	estBytes := 2 // {} braces
	for _, name := range names {
		ref := byName[name]
		estBytes += len(name) + len(ref.DType) + 24 + 12*len(ref.Shape) + 50
	}
	out := make([]byte, 0, estBytes)
	out = append(out, '{')

	ordered := make([]TensorRef, 0, len(names))
	var offset int64
	for i, name := range names {
		ref := byName[name]
		if i > 0 {
			out = append(out, ',')
		}
		out = appendJSONString(out, name)
		out = append(out, ':', '{')
		// "dtype":"<UPPER>"
		out = append(out, '"', 'd', 't', 'y', 'p', 'e', '"', ':')
		out = appendJSONString(out, core.Upper(ref.DType))
		// ,"shape":[d0,d1,…]
		out = append(out, ',', '"', 's', 'h', 'a', 'p', 'e', '"', ':', '[')
		for j, dim := range ref.Shape {
			if dim > uint64(maxInt64Value()) {
				return nil, nil, core.NewError("mlx: safetensors subset tensor shape is too large: " + ref.Name)
			}
			if j > 0 {
				out = append(out, ',')
			}
			out = appendJSONInt64(out, int64(dim))
		}
		out = append(out, ']')
		// ,"data_offsets":[begin,end]
		out = append(out, ',', '"', 'd', 'a', 't', 'a', '_', 'o', 'f', 'f', 's', 'e', 't', 's', '"', ':', '[')
		out = appendJSONInt64(out, offset)
		out = append(out, ',')
		out = appendJSONInt64(out, offset+ref.ByteLen)
		out = append(out, ']', '}')
		offset += ref.ByteLen
		ordered = append(ordered, ref)
	}
	out = append(out, '}')
	return ordered, out, nil
}

// appendJSONString appends a JSON-quoted string. The fast path (no
// characters needing escape, which is the case for every real
// safetensors tensor name plus every supported dtype) is a verbatim
// byte append between quotes. The slow path handles \\ and \" and the
// control characters per RFC 8259.
func appendJSONString(dst []byte, s string) []byte {
	dst = append(dst, '"')
	start := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '"' || c == '\\' || c < 0x20 {
			if start < i {
				dst = append(dst, s[start:i]...)
			}
			switch c {
			case '"':
				dst = append(dst, '\\', '"')
			case '\\':
				dst = append(dst, '\\', '\\')
			case '\b':
				dst = append(dst, '\\', 'b')
			case '\f':
				dst = append(dst, '\\', 'f')
			case '\n':
				dst = append(dst, '\\', 'n')
			case '\r':
				dst = append(dst, '\\', 'r')
			case '\t':
				dst = append(dst, '\\', 't')
			default:
				dst = append(dst, '\\', 'u', '0', '0', hexNibble(c>>4), hexNibble(c&0xf))
			}
			start = i + 1
		}
	}
	if start < len(s) {
		dst = append(dst, s[start:]...)
	}
	dst = append(dst, '"')
	return dst
}

func hexNibble(b byte) byte {
	if b < 10 {
		return '0' + b
	}
	return 'a' + b - 10
}

// appendJSONInt64 emits a base-10 representation of v with no leading
// zeros (matching encoding/json + strconv.FormatInt). The implementation
// is a digit-extraction unroll that lands in a fixed 20-byte stack
// buffer, so no heap allocation occurs regardless of v's magnitude.
func appendJSONInt64(dst []byte, v int64) []byte {
	if v == 0 {
		return append(dst, '0')
	}
	var buf [20]byte
	i := len(buf)
	neg := v < 0
	var uv uint64
	if neg {
		uv = uint64(-v)
	} else {
		uv = uint64(v)
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

// writeRefRawChunksScratch streams one tensor's raw payload through a
// caller-supplied byte buffer, returning the (possibly grown) buffer
// for the next call to reuse. Hoisting the buffer up to WriteSubset
// collapses what was N small allocs into one.
func writeRefRawChunksScratch(ctx context.Context, out *core.OSFile, ref TensorRef, chunkBytes int64, scratch []byte) ([]byte, error) {
	if chunkBytes <= 0 {
		chunkBytes = defaultRawChunkBytes
	}
	opened := core.Open(ref.Path)
	if !opened.OK {
		return scratch, resultError(opened)
	}
	in := opened.Value.(*core.OSFile)
	defer in.Close()

	need := minInt64(chunkBytes, ref.ByteLen)
	if int64(cap(scratch)) < need {
		scratch = make([]byte, need)
	} else {
		scratch = scratch[:need]
	}
	remaining := ref.ByteLen
	offset := ref.DataStart
	for remaining > 0 {
		if err := ctx.Err(); err != nil {
			return scratch, err
		}
		want := minInt64(int64(len(scratch)), remaining)
		n, err := in.ReadAt(scratch[:want], offset)
		if err != nil && !(err == core.EOF && int64(n) == want) {
			return scratch, err
		}
		if int64(n) != want {
			return scratch, core.NewError("mlx: safetensors tensor payload is truncated: " + ref.Name)
		}
		if err := writeAll(out, scratch[:want]); err != nil {
			return scratch, err
		}
		offset += want
		remaining -= want
	}
	return scratch, nil
}

func writeAll(file *core.OSFile, data []byte) error {
	for len(data) > 0 {
		n, err := file.Write(data)
		if err != nil {
			return err
		}
		if n == 0 {
			return errWriteNoProgress
		}
		data = data[n:]
	}
	return nil
}

func maxInt64Value() int64 { return int64(^uint64(0) >> 1) }

func minInt64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}
