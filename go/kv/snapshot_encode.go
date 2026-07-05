// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"encoding/binary"
	stdio "io"
	"math"
	"sync"
	"unsafe"

	core "dappco.re/go"
)

// Save writes the snapshot to path using the stable go-mlx KV binary format.
func (s *Snapshot) Save(path string) error {
	return s.SaveWithOptions(path, SaveOptions{})
}

// SaveWithOptions writes the snapshot with explicit K/V tensor encoding.
func (s *Snapshot) SaveWithOptions(path string, opts SaveOptions) error {
	if s == nil {
		return errSnapshotNil
	}
	data, err := s.bytesWithOptions(opts)
	if err != nil {
		return err
	}
	if result := core.WriteFile(path, data, 0o600); !result.OK {
		return core.E("Snapshot.Save", "write snapshot", result.Err())
	}
	return nil
}

// MarshalBinary returns the stable binary representation used by Save.
func (s *Snapshot) MarshalBinary() ([]byte, error) {
	if s == nil {
		return nil, errSnapshotNil
	}
	return s.bytesWithOptions(SaveOptions{})
}

func (s *Snapshot) bytes() ([]byte, error) {
	return s.bytesWithOptions(SaveOptions{})
}

func (s *Snapshot) encodedSizeWithOptions(opts SaveOptions) (int, error) {
	encoding, err := normalizeKVSnapshotEncoding(opts.KVEncoding)
	if err != nil {
		return 0, err
	}
	if err := validateKVSnapshotLayerSchemes(s); err != nil {
		return 0, err
	}
	version := effectiveVersion(s, encoding)
	if version <= 0 || version > SnapshotVersion {
		return 0, core.E("Snapshot.Save", "unsupported KV snapshot version", nil)
	}
	if len(s.Architecture) > int(^uint32(0)) {
		return 0, core.E("Snapshot.Save", "architecture string too large", nil)
	}
	size := len(kvSnapshotMagic)
	size += 4                       // version
	size += 4 + len(s.Architecture) // architecture
	size += 5 * 4                   // layers, heads, seq len, head dim, query heads
	size += 4 + len(s.Tokens)*4     // tokens
	size += 4                       // layer count
	if version >= 2 {
		size += 4                      // token offset
		size += 4 + len(s.Generated)*4 // generated tokens
	}
	for _, layer := range s.Layers {
		size += 12 // layer, cache index, head count
		if version >= 5 {
			size += 4 + len(layer.CacheMode)
			size += 4
			for _, payload := range layer.TurboQuantPayloads {
				size += 4 + len(payload)
			}
		}
		if version >= 6 {
			size += 4 // max size
		}
		if version >= 4 {
			keySize, err := kvSnapshotEncodedTensorSize(nil, layer.KeyDType, layer.KeyBytes, encoding)
			if err != nil {
				return 0, core.E("Snapshot.Save", "encode layer key tensor", err)
			}
			valueSize, err := kvSnapshotEncodedTensorSize(nil, layer.ValueDType, layer.ValueBytes, encoding)
			if err != nil {
				return 0, core.E("Snapshot.Save", "encode layer value tensor", err)
			}
			size += 4 + len(layer.KeyShape)*4
			size += keySize
			size += 4 + len(layer.ValueShape)*4
			size += valueSize
		}
		for _, head := range layer.Heads {
			if version >= 3 {
				keySize, err := kvSnapshotEncodedTensorSize(head.Key, head.KeyDType, head.KeyBytes, encoding)
				if err != nil {
					return 0, core.E("Snapshot.Save", "encode key tensor", err)
				}
				valueSize, err := kvSnapshotEncodedTensorSize(head.Value, head.ValueDType, head.ValueBytes, encoding)
				if err != nil {
					return 0, core.E("Snapshot.Save", "encode value tensor", err)
				}
				size += keySize + valueSize
			} else {
				size += 4 + len(head.Key)*4
				size += 4 + len(head.Value)*4
			}
		}
	}
	if version >= 2 {
		size += 4 + len(s.LogitShape)*4
		size += 4 + len(s.Logits)*4
	}
	return size, nil
}

func (s *Snapshot) bytesWithOptions(opts SaveOptions) ([]byte, error) {
	encoding, err := normalizeKVSnapshotEncoding(opts.KVEncoding)
	if err != nil {
		return nil, err
	}
	size, err := s.encodedSizeWithOptions(opts)
	if err != nil {
		return nil, err
	}
	data := make([]byte, 0, size)
	data = append(data, kvSnapshotMagic...)
	version := effectiveVersion(s, encoding)
	if version <= 0 || version > SnapshotVersion {
		return nil, core.E("Snapshot.Save", "unsupported KV snapshot version", nil)
	}
	data = appendKVU32(data, uint32(version))
	if len(s.Architecture) > int(^uint32(0)) {
		return nil, core.E("Snapshot.Save", "architecture string too large", nil)
	}
	data = appendKVBytes(data, core.AsBytes(s.Architecture))
	data = appendKVU32(data, uint32(s.NumLayers))
	data = appendKVU32(data, uint32(s.NumHeads))
	data = appendKVU32(data, uint32(s.SeqLen))
	data = appendKVU32(data, uint32(s.HeadDim))
	data = appendKVU32(data, uint32(s.NumQueryHeads))
	if version >= 2 {
		tokenOffset := s.TokenOffset
		if tokenOffset == 0 {
			tokenOffset = len(s.Tokens)
		}
		data = appendKVU32(data, uint32(tokenOffset))
	}
	data = appendKVU32(data, uint32(len(s.Tokens)))
	data = appendKVI32sRaw(data, s.Tokens)
	if version >= 2 {
		data = appendKVU32(data, uint32(len(s.Generated)))
		data = appendKVI32sRaw(data, s.Generated)
	}
	data = appendKVU32(data, uint32(len(s.Layers)))
	for _, layer := range s.Layers {
		data = appendKVI32(data, int32(layer.Layer))
		data = appendKVI32(data, int32(layer.CacheIndex))
		data = appendKVU32(data, uint32(len(layer.Heads)))
		if version >= 5 {
			data = appendKVBytes(data, core.AsBytes(layer.CacheMode))
			data = appendKVU32(data, uint32(len(layer.TurboQuantPayloads)))
			for _, payload := range layer.TurboQuantPayloads {
				data = appendKVBytes(data, payload)
			}
		}
		if version >= 6 {
			data = appendKVU32(data, uint32(layer.MaxSize))
		}
		if version >= 4 {
			data = appendKVI32s(data, layer.KeyShape)
			data, err = appendKVEncodedTensor(data, nil, layer.KeyDType, layer.KeyBytes, encoding)
			if err != nil {
				return nil, core.E("Snapshot.Save", "encode layer key tensor", err)
			}
			data = appendKVI32s(data, layer.ValueShape)
			data, err = appendKVEncodedTensor(data, nil, layer.ValueDType, layer.ValueBytes, encoding)
			if err != nil {
				return nil, core.E("Snapshot.Save", "encode layer value tensor", err)
			}
		}
		for _, head := range layer.Heads {
			if version >= 3 {
				data, err = appendKVEncodedTensor(data, head.Key, head.KeyDType, head.KeyBytes, encoding)
				if err != nil {
					return nil, core.E("Snapshot.Save", "encode key tensor", err)
				}
				data, err = appendKVEncodedTensor(data, head.Value, head.ValueDType, head.ValueBytes, encoding)
				if err != nil {
					return nil, core.E("Snapshot.Save", "encode value tensor", err)
				}
			} else {
				data = appendKVF32s(data, head.Key)
				data = appendKVF32s(data, head.Value)
			}
		}
	}
	if version >= 2 {
		data = appendKVU32(data, uint32(len(s.LogitShape)))
		data = appendKVI32sRaw(data, s.LogitShape)
		data = appendKVF32s(data, s.Logits)
	}
	return data, nil
}

func (s *Snapshot) writeWithOptions(writer stdio.Writer, opts SaveOptions) error {
	encoding, err := normalizeKVSnapshotEncoding(opts.KVEncoding)
	if err != nil {
		return err
	}
	if err := validateKVSnapshotLayerSchemes(s); err != nil {
		return err
	}
	version := effectiveVersion(s, encoding)
	// Cheap up-front sanity covers what encodedSizeWithOptions exists to
	// guard at this layer — version range and architecture-string length.
	// Per-tensor validation surfaces naturally through stream.encodedTensor
	// during the write loop; callers (HashSnapshot, state-block stream)
	// treat any error as fatal, so the half-flush is harmless.
	if version <= 0 || version > SnapshotVersion {
		return core.E("Snapshot.Save", "unsupported KV snapshot version", nil)
	}
	if len(s.Architecture) > int(^uint32(0)) {
		return core.E("Snapshot.Save", "architecture string too large", nil)
	}
	stream := acquireKVStreamWriter(writer)
	defer releaseKVStreamWriter(stream)
	stream.bytes(core.AsBytes(kvSnapshotMagic))
	stream.u32(uint32(version))
	stream.bytesWithLength(core.AsBytes(s.Architecture))
	stream.u32(uint32(s.NumLayers))
	stream.u32(uint32(s.NumHeads))
	stream.u32(uint32(s.SeqLen))
	stream.u32(uint32(s.HeadDim))
	stream.u32(uint32(s.NumQueryHeads))
	if version >= 2 {
		tokenOffset := s.TokenOffset
		if tokenOffset == 0 {
			tokenOffset = len(s.Tokens)
		}
		stream.u32(uint32(tokenOffset))
	}
	stream.u32(uint32(len(s.Tokens)))
	stream.i32sRaw(s.Tokens)
	if version >= 2 {
		stream.u32(uint32(len(s.Generated)))
		stream.i32sRaw(s.Generated)
	}
	stream.u32(uint32(len(s.Layers)))
	for _, layer := range s.Layers {
		stream.i32(int32(layer.Layer))
		stream.i32(int32(layer.CacheIndex))
		stream.u32(uint32(len(layer.Heads)))
		if version >= 5 {
			stream.bytesWithLength(core.AsBytes(layer.CacheMode))
			stream.u32(uint32(len(layer.TurboQuantPayloads)))
			for _, payload := range layer.TurboQuantPayloads {
				stream.bytesWithLength(payload)
			}
		}
		if version >= 6 {
			stream.u32(uint32(layer.MaxSize))
		}
		if version >= 4 {
			stream.i32s(layer.KeyShape)
			if err := stream.encodedTensor(nil, layer.KeyDType, layer.KeyBytes, encoding); err != nil {
				return core.E("Snapshot.Save", "encode layer key tensor", err)
			}
			stream.i32s(layer.ValueShape)
			if err := stream.encodedTensor(nil, layer.ValueDType, layer.ValueBytes, encoding); err != nil {
				return core.E("Snapshot.Save", "encode layer value tensor", err)
			}
		}
		for _, head := range layer.Heads {
			if version >= 3 {
				if err := stream.encodedTensor(head.Key, head.KeyDType, head.KeyBytes, encoding); err != nil {
					return core.E("Snapshot.Save", "encode key tensor", err)
				}
				if err := stream.encodedTensor(head.Value, head.ValueDType, head.ValueBytes, encoding); err != nil {
					return core.E("Snapshot.Save", "encode value tensor", err)
				}
			} else {
				stream.f32s(head.Key)
				stream.f32s(head.Value)
			}
		}
	}
	if version >= 2 {
		stream.u32(uint32(len(s.LogitShape)))
		stream.i32sRaw(s.LogitShape)
		stream.f32s(s.Logits)
	}
	return stream.err
}

func normalizeKVSnapshotEncoding(encoding Encoding) (Encoding, error) {
	switch encoding {
	case "", KVSnapshotEncodingFloat32:
		return KVSnapshotEncodingFloat32, nil
	case EncodingQ8, EncodingNative:
		return encoding, nil
	default:
		return "", core.E("Snapshot.Save", "unsupported KV snapshot encoding", nil)
	}
}

func appendKVBytes(dst, src []byte) []byte {
	dst = appendKVU32(dst, uint32(len(src)))
	return append(dst, src...)
}

func appendKVU32(dst []byte, value uint32) []byte {
	return binary.LittleEndian.AppendUint32(dst, value)
}

func appendKVI32(dst []byte, value int32) []byte {
	return appendKVU32(dst, uint32(value))
}

func appendKVI32s(dst []byte, values []int32) []byte {
	dst = appendKVU32(dst, uint32(len(values)))
	return appendKVI32sRaw(dst, values)
}

// appendKVI32sRaw appends int32 values without a length prefix.
// Used by bytesWithOptions when the length has already been written.
func appendKVI32sRaw(dst []byte, values []int32) []byte {
	if len(values) == 0 {
		return dst
	}
	// Reinterpret-cast: int32 is little-endian on both Go-supported
	// architectures, so the byte view of []int32 matches the
	// per-element appendKVU32(uint32(v)) loop output. Single append
	// vs N×PutUint32 — see f32sRaw comment.
	src := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), len(values)*4)
	return append(dst, src...)
}

func appendKVF32s(dst []byte, values []float32) []byte {
	dst = appendKVU32(dst, uint32(len(values)))
	return appendKVF32Raw(dst, values)
}

func appendKVF32Raw(dst []byte, values []float32) []byte {
	if len(values) == 0 {
		return dst
	}
	// Reinterpret-cast: float32 storage is little-endian on both
	// Go-supported architectures (arm64 + amd64), so the byte view of
	// []float32 already matches appendKVU32(math.Float32bits(v)).
	// Single append vs per-element PutUint32 — see f32sRaw comment.
	src := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), len(values)*4)
	return append(dst, src...)
}

func appendKVEncodedTensor(dst []byte, values []float32, dtype string, raw []byte, encoding Encoding) ([]byte, error) {
	if encoding == EncodingNative {
		// Fast path when raw is already present — append directly with
		// no intermediate alloc.
		if len(raw) > 0 {
			rawDType, rawElements, _, ok, err := kvSnapshotNativeTensorInfo(values, dtype, raw)
			if err != nil {
				return nil, err
			}
			if ok {
				dst = appendKVU32(dst, 2)
				dst = appendKVU32(dst, uint32(rawElements))
				dst = appendKVBytes(dst, core.AsBytes(rawDType))
				return appendKVBytes(dst, raw), nil
			}
		} else if len(values) > 0 {
			// Stream float32 values directly into dst — skips the
			// normalizeKVSnapshotNativeTensor intermediate alloc + the
			// follow-on appendKVBytes copy.
			dst = appendKVU32(dst, 2)
			dst = appendKVU32(dst, uint32(len(values)))
			dst = appendKVBytes(dst, core.AsBytes("float32"))
			dst = appendKVU32(dst, uint32(len(values)*4))
			return appendKVF32Raw(dst, values), nil
		}
	}
	if len(values) == 0 && len(raw) > 0 {
		return nil, errRawTensorNeedsNative
	}
	if encoding == EncodingQ8 {
		if maxAbs, ok := kvSnapshotQ8Validate(values); ok {
			// Fused: validate already produced maxAbs, skip the
			// follow-on walk inside quantizeKVSnapshotQ8.
			scale, quantized := quantizeKVSnapshotQ8WithMaxAbs(values, maxAbs)
			dst = appendKVU32(dst, 1)
			dst = appendKVU32(dst, uint32(len(values)))
			dst = appendKVU32(dst, math.Float32bits(scale))
			return append(dst, quantized...), nil
		}
	}
	dst = appendKVU32(dst, 0)
	dst = appendKVU32(dst, uint32(len(values)))
	return appendKVF32Raw(dst, values), nil
}

func appendKVEncodedF32s(dst []byte, values []float32, encoding Encoding) []byte {
	out, err := appendKVEncodedTensor(dst, values, "", nil, encoding)
	if err != nil {
		return dst
	}
	return out
}

func kvSnapshotEncodedTensorSize(values []float32, dtype string, raw []byte, encoding Encoding) (int, error) {
	if encoding == EncodingNative {
		normalisedDType, _, rawBytes, ok, err := kvSnapshotNativeTensorInfo(values, dtype, raw)
		if err != nil {
			return 0, err
		}
		if ok {
			return 16 + len(normalisedDType) + rawBytes, nil
		}
	}
	if len(values) == 0 && len(raw) > 0 {
		return 0, errRawTensorNeedsNative
	}
	if encoding == EncodingQ8 && kvSnapshotCanQuantizeQ8(values) {
		return 12 + len(values), nil
	}
	return 8 + len(values)*4, nil
}

func kvSnapshotNativeTensorInfo(values []float32, dtype string, raw []byte) (string, int, int, bool, error) {
	if len(raw) > 0 {
		dtype, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
		if dtype == "" || bytesPerValue <= 0 {
			return "", 0, 0, false, errUnsupportedNativeTensor
		}
		if len(raw)%bytesPerValue != 0 {
			return "", 0, 0, false, errNativeByteLenMismatch
		}
		elements := len(raw) / bytesPerValue
		if len(values) > 0 && elements != len(values) {
			return "", 0, 0, false, errNativeElementCount
		}
		return dtype, elements, len(raw), true, nil
	}
	if len(values) == 0 {
		return "", 0, 0, false, nil
	}
	return "float32", len(values), len(values) * 4, true, nil
}

type kvSnapshotStreamWriter struct {
	writer stdio.Writer
	err    error
	buf    [4]byte
}

// kvSnapshotStreamWriterPool reuses streamWriter structs across
// writeWithOptions calls — the struct escapes to heap (interface-
// satisfying methods + &stream pointer threading). SaveStateBlocks
// fires writeWithOptions per block hash + per block payload + final
// bundle hash, so a pool collapses 6-8 stream allocs into one across
// a single SaveStateBlocks call.
var kvSnapshotStreamWriterPool = sync.Pool{
	New: func() any { return &kvSnapshotStreamWriter{} },
}

func acquireKVStreamWriter(writer stdio.Writer) *kvSnapshotStreamWriter {
	stream := kvSnapshotStreamWriterPool.Get().(*kvSnapshotStreamWriter)
	stream.writer = writer
	stream.err = nil
	return stream
}

func releaseKVStreamWriter(stream *kvSnapshotStreamWriter) {
	stream.writer = nil
	stream.err = nil
	kvSnapshotStreamWriterPool.Put(stream)
}

func (w *kvSnapshotStreamWriter) bytes(data []byte) {
	if w.err != nil {
		return
	}
	n, err := w.writer.Write(data)
	if err != nil {
		w.err = err
		return
	}
	if n != len(data) {
		w.err = stdio.ErrShortWrite
	}
}

func (w *kvSnapshotStreamWriter) bytesWithLength(data []byte) {
	w.u32(uint32(len(data)))
	w.bytes(data)
}

func (w *kvSnapshotStreamWriter) u32(value uint32) {
	binary.LittleEndian.PutUint32(w.buf[:], value)
	w.bytes(w.buf[:])
}

func (w *kvSnapshotStreamWriter) i32(value int32) {
	w.u32(uint32(value))
}

func (w *kvSnapshotStreamWriter) i32s(values []int32) {
	w.u32(uint32(len(values)))
	w.i32sRaw(values)
}

// i32sRaw writes int32 values without a length prefix. Used by
// writeWithOptions when the length has already been written.
func (w *kvSnapshotStreamWriter) i32sRaw(values []int32) {
	if w.err != nil || len(values) == 0 {
		return
	}
	// Reinterpret-cast write: int32 storage is little-endian on both
	// arm64 and amd64 (Go-supported architectures), so the byte view
	// of []int32 already matches the per-element PutUint32 output.
	// Pass the byte view straight to writer.Write — writers (sha256,
	// PutBytesStream) consume the data within the call, so we don't
	// need a scratch staging copy. Same pattern as f32sRaw.
	src := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), len(values)*4)
	w.bytes(src)
}

func (w *kvSnapshotStreamWriter) f32s(values []float32) {
	w.u32(uint32(len(values)))
	w.f32sRaw(values)
}

// f32sRaw writes float32 values without a length prefix.
func (w *kvSnapshotStreamWriter) f32sRaw(values []float32) {
	if w.err != nil || len(values) == 0 {
		return
	}
	// Reinterpret-cast write: float32 storage is little-endian on both
	// Go-supported architectures (arm64 + amd64), so the byte view of
	// []float32 already matches what PutUint32(buf, Float32bits(v))
	// would write element-by-element. Pass the byte view straight to
	// writer.Write — writers (sha256, PutBytesStream) consume the data
	// within the call, so the staging copy via the previously-pooled
	// scratch buffer was net waste (memcpy into scratch then memcpy
	// into the writer's own buffer). One memcpy vs two.
	src := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), len(values)*4)
	w.bytes(src)
}

func (w *kvSnapshotStreamWriter) encodedTensor(values []float32, dtype string, raw []byte, encoding Encoding) error {
	if encoding == EncodingNative {
		// Fast path when raw is already present — write directly with
		// no intermediate alloc.
		if len(raw) > 0 {
			rawDType, rawElements, _, ok, err := kvSnapshotNativeTensorInfo(values, dtype, raw)
			if err != nil {
				return err
			}
			if ok {
				w.u32(2)
				w.u32(uint32(rawElements))
				w.bytesWithLength(core.AsBytes(rawDType))
				w.bytesWithLength(raw)
				return w.err
			}
		} else if len(values) > 0 {
			// Stream float32 values directly — skips the intermediate
			// normalizeKVSnapshotNativeTensor alloc that the
			// pre-bytesWithOptions sibling path already eliminated.
			w.u32(2)
			w.u32(uint32(len(values)))
			w.bytesWithLength(core.AsBytes("float32"))
			w.u32(uint32(len(values) * 4))
			w.f32sRaw(values)
			return w.err
		}
	}
	if len(values) == 0 && len(raw) > 0 {
		return errRawTensorNeedsNative
	}
	if encoding == EncodingQ8 {
		if maxAbs, ok := kvSnapshotQ8Validate(values); ok {
			// Fused: validate already produced maxAbs, skip the
			// follow-on walk inside quantizeKVSnapshotQ8.
			scale, quantized := quantizeKVSnapshotQ8WithMaxAbs(values, maxAbs)
			w.u32(1)
			w.u32(uint32(len(values)))
			w.u32(math.Float32bits(scale))
			w.bytes(quantized)
			return w.err
		}
	}
	w.u32(0)
	w.u32(uint32(len(values)))
	w.f32sRaw(values)
	return w.err
}
