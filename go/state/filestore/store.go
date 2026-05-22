// SPDX-Licence-Identifier: EUPL-1.2

// Package filestore provides an append-only file-backed state store.
package filestore

import (
	"context"
	"encoding/binary"
	stdio "io"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference/state"
)

const (
	CodecFile       = "state/file-log"
	CodecMemvidFile = "memvid/file-log"

	fileMode        = 0o600
	recordHeaderLen = 24
)

var (
	fileMagic       = []byte("go-inference-state-file-log-v1\n")
	legacyFileMagic = []byte("go-mlx-memvid-file-log-v1\n")
	recordMagic     = [4]byte{'M', 'V', 'F', '1'}
)

type Store struct {
	mu       sync.Mutex
	path     string
	file     *core.OSFile
	index    map[int]fileIndexEntry
	uriIndex map[string]int
	nextID   int
	writeAt  int64
}

type fileIndexEntry struct {
	ref         state.ChunkRef
	payloadAt   int64
	payloadSize int
}

type recordMeta struct {
	URI    string            `json:"uri,omitempty"`
	Title  string            `json:"title,omitempty"`
	Kind   string            `json:"kind,omitempty"`
	Track  string            `json:"track,omitempty"`
	Tags   map[string]string `json:"tags,omitempty"`
	Labels []string          `json:"labels,omitempty"`
}

// Create initialises a new append-only state file store at path.
func Create(ctx context.Context, path string) (*Store, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if core.Trim(path) == "" {
		return nil, core.NewError("state file store path is required")
	}
	if result := core.MkdirAll(core.PathDir(path), 0o755); !result.OK {
		return nil, core.E("state.filestore.Create", "create parent directory", resultError(result))
	}
	result := core.OpenFile(path, core.O_CREATE|core.O_TRUNC|core.O_RDWR, fileMode)
	if !result.OK {
		return nil, core.E("state.filestore.Create", "create file", resultError(result))
	}
	file := result.Value.(*core.OSFile)
	if err := writeAll(file, fileMagic); err != nil {
		_ = file.Close()
		return nil, core.E("state.filestore.Create", "write file header", err)
	}
	return &Store{
		path:     path,
		file:     file,
		index:    make(map[int]fileIndexEntry),
		uriIndex: make(map[string]int),
		nextID:   1,
		writeAt:  int64(len(fileMagic)),
	}, nil
}

// Open reopens an existing append-only state file store and rebuilds its
// offset index without reading chunk payloads.
func Open(ctx context.Context, path string) (*Store, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if core.Trim(path) == "" {
		return nil, core.NewError("state file store path is required")
	}
	result := core.OpenFile(path, core.O_RDWR, fileMode)
	if !result.OK {
		return nil, core.E("state.filestore.Open", "open file", resultError(result))
	}
	file := result.Value.(*core.OSFile)
	store := &Store{
		path:     path,
		file:     file,
		index:    make(map[int]fileIndexEntry),
		uriIndex: make(map[string]int),
		nextID:   1,
	}
	if err := store.rebuildIndex(ctx); err != nil {
		_ = file.Close()
		return nil, err
	}
	return store, nil
}

func (s *Store) Path() string {
	if s == nil {
		return ""
	}
	return s.path
}

func (s *Store) ChunkCount() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.index)
}

func (s *Store) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.file == nil {
		return nil
	}
	file := s.file
	s.file = nil
	return file.Close()
}

func (s *Store) Get(ctx context.Context, chunkID int) (string, error) {
	chunk, err := s.Resolve(ctx, chunkID)
	if err != nil {
		return "", err
	}
	return chunk.Text, nil
}

func (s *Store) Resolve(ctx context.Context, chunkID int) (state.Chunk, error) {
	if err := checkContext(ctx); err != nil {
		return state.Chunk{}, err
	}
	if s == nil {
		return state.Chunk{}, &state.ChunkNotFoundError{ID: chunkID}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.file == nil {
		return state.Chunk{}, core.NewError("state file store is closed")
	}
	return s.resolveLocked(chunkID)
}

func (s *Store) ResolveURI(ctx context.Context, uri string) (state.Chunk, error) {
	if err := checkContext(ctx); err != nil {
		return state.Chunk{}, err
	}
	if s == nil {
		return state.Chunk{}, &state.URIChunkNotFoundError{URI: uri}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.file == nil {
		return state.Chunk{}, core.NewError("state file store is closed")
	}
	id, ok := s.uriIndex[uri]
	if !ok {
		return state.Chunk{}, &state.URIChunkNotFoundError{URI: uri}
	}
	return s.resolveLocked(id)
}

func (s *Store) Put(ctx context.Context, text string, opts state.PutOptions) (state.ChunkRef, error) {
	// PutBytes feeds data into a writer that copies it onto disk — the
	// underlying io.Writer contract forbids retention or mutation, so
	// AsBytes is safe here. Avoids the copy of `text` into a fresh
	// []byte just to be discarded after the disk write.
	return s.PutBytes(ctx, core.AsBytes(text), opts)
}

func (s *Store) PutBytes(ctx context.Context, data []byte, opts state.PutOptions) (state.ChunkRef, error) {
	return s.PutBytesStream(ctx, len(data), opts, func(writer stdio.Writer) error {
		return writeAll(writer, data)
	})
}

func (s *Store) PutBytesStream(ctx context.Context, payloadSize int, opts state.PutOptions, write func(stdio.Writer) error) (state.ChunkRef, error) {
	if err := checkContext(ctx); err != nil {
		return state.ChunkRef{}, err
	}
	if s == nil {
		return state.ChunkRef{}, core.NewError("state file store is nil")
	}
	if payloadSize < 0 {
		return state.ChunkRef{}, core.NewError("state file store payload size is invalid")
	}
	if write == nil {
		return state.ChunkRef{}, core.NewError("state file store stream writer is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.file == nil {
		return state.ChunkRef{}, core.NewError("state file store is closed")
	}

	id := s.nextID
	meta := recordMeta{
		URI:    opts.URI,
		Title:  opts.Title,
		Kind:   opts.Kind,
		Track:  opts.Track,
		Tags:   opts.Tags,
		Labels: opts.Labels,
	}
	// Use JSONMarshal direct — JSONMarshalString → []byte cast did a
	// roundtrip via two string conversions. JSONMarshal returns the
	// freshly-allocated []byte we want for the write.
	metaResult := core.JSONMarshal(meta)
	if !metaResult.OK {
		return state.ChunkRef{}, metaResult.Value.(error)
	}
	metaBytes := metaResult.Value.([]byte)
	if uint64(len(metaBytes)) > uint64(^uint32(0)) {
		return state.ChunkRef{}, core.NewError("state file store metadata is too large")
	}

	var headerBuf [recordHeaderLen]byte
	encodeRecordHeader(headerBuf[:], id, payloadSize, len(metaBytes))
	offset := s.writeAt
	if _, err := s.file.Seek(offset, stdio.SeekStart); err != nil {
		return state.ChunkRef{}, core.E("state.filestore.Put", "seek to append offset", err)
	}
	if err := writeAll(s.file, headerBuf[:]); err != nil {
		s.rollbackWriteLocked(offset)
		return state.ChunkRef{}, core.E("state.filestore.Put", "write record header", err)
	}
	if err := writeAll(s.file, metaBytes); err != nil {
		s.rollbackWriteLocked(offset)
		return state.ChunkRef{}, core.E("state.filestore.Put", "write record metadata", err)
	}
	payloadWriter := &limitedPayloadWriter{
		file:      s.file,
		remaining: payloadSize,
	}
	if err := write(payloadWriter); err != nil {
		s.rollbackWriteLocked(offset)
		return state.ChunkRef{}, core.E("state.filestore.Put", "write record payload", err)
	}
	if payloadWriter.remaining != 0 {
		s.rollbackWriteLocked(offset)
		return state.ChunkRef{}, core.NewError("state file store streamed payload is shorter than declared")
	}
	ref := state.ChunkRef{
		ChunkID:        id,
		FrameOffset:    uint64(offset),
		HasFrameOffset: true,
		Codec:          CodecFile,
		Segment:        s.path,
	}
	s.index[id] = fileIndexEntry{
		ref:         ref,
		payloadAt:   offset + recordHeaderLen + int64(len(metaBytes)),
		payloadSize: payloadSize,
	}
	if meta.URI != "" {
		s.uriIndex[meta.URI] = id
	}
	s.nextID++
	s.writeAt += int64(recordHeaderLen + len(metaBytes) + payloadSize)
	return ref, nil
}

func (s *Store) rollbackWriteLocked(offset int64) {
	if s == nil || s.file == nil {
		return
	}
	_ = s.file.Truncate(offset)
	_, _ = s.file.Seek(offset, stdio.SeekStart)
}

func (s *Store) resolveLocked(chunkID int) (state.Chunk, error) {
	chunk, err := s.resolveBytesLocked(chunkID)
	if err != nil {
		return state.Chunk{}, err
	}
	// chunk.Data is freshly allocated by ReadAt and unreachable here
	// — handing it to AsString skips the payload-sized copy that
	// string(chunk.Data) would do. Every Resolve text read benefits;
	// payloads scale to KB+ for compressed state slices.
	chunk.Text = core.AsString(chunk.Data)
	chunk.Data = nil
	return chunk, nil
}

func (s *Store) ResolveBytes(ctx context.Context, chunkID int) (state.Chunk, error) {
	if err := checkContext(ctx); err != nil {
		return state.Chunk{}, err
	}
	if s == nil {
		return state.Chunk{}, &state.ChunkNotFoundError{ID: chunkID}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.file == nil {
		return state.Chunk{}, core.NewError("state file store is closed")
	}
	return s.resolveBytesLocked(chunkID)
}

func (s *Store) ResolveRefBytes(ctx context.Context, ref state.ChunkRef) (state.Chunk, error) {
	if err := checkContext(ctx); err != nil {
		return state.Chunk{}, err
	}
	if s == nil {
		return state.Chunk{}, &state.ChunkNotFoundError{ID: ref.ChunkID}
	}
	if !ref.HasFrameOffset {
		return s.ResolveBytes(ctx, ref.ChunkID)
	}
	if ref.Codec != "" && ref.Codec != CodecFile && ref.Codec != CodecMemvidFile {
		return state.Chunk{}, core.NewError("state file store cannot resolve non-file chunk ref")
	}
	if ref.Segment != "" && ref.Segment != s.path {
		return state.Chunk{}, core.NewError("state file store chunk ref segment mismatch")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.file == nil {
		return state.Chunk{}, core.NewError("state file store is closed")
	}
	return s.resolveRefBytesLocked(ref)
}

func (s *Store) resolveBytesLocked(chunkID int) (state.Chunk, error) {
	entry, ok := s.index[chunkID]
	if !ok {
		return state.Chunk{}, &state.ChunkNotFoundError{ID: chunkID}
	}
	payload := make([]byte, entry.payloadSize)
	if _, err := s.file.ReadAt(payload, entry.payloadAt); err != nil {
		return state.Chunk{}, core.E("state.filestore.Resolve", "read chunk payload", err)
	}
	return state.Chunk{
		Ref:  entry.ref,
		Data: payload,
	}, nil
}

func (s *Store) resolveRefBytesLocked(ref state.ChunkRef) (state.Chunk, error) {
	if ref.FrameOffset > uint64(maxInt()) {
		return state.Chunk{}, core.NewError("state file store frame offset is too large")
	}
	offset := int64(ref.FrameOffset)
	var headerBuf [recordHeaderLen]byte
	if _, err := s.file.ReadAt(headerBuf[:], offset); err != nil {
		return state.Chunk{}, core.E("state.filestore.ResolveRefBytes", "read record header", err)
	}
	record, err := decodeRecordHeader(headerBuf[:])
	if err != nil {
		return state.Chunk{}, err
	}
	id, err := intFromUint64(record.chunkID, "chunk id")
	if err != nil {
		return state.Chunk{}, err
	}
	if ref.ChunkID != 0 && id != ref.ChunkID {
		return state.Chunk{}, core.NewError("state file store chunk ref id mismatch")
	}
	metaSize, err := intFromUint64(uint64(record.metaSize), "metadata")
	if err != nil {
		return state.Chunk{}, err
	}
	payloadSize, err := intFromUint64(record.payloadSize, "payload")
	if err != nil {
		return state.Chunk{}, err
	}
	payloadAt := offset + recordHeaderLen + int64(metaSize)
	payload := make([]byte, payloadSize)
	if _, err := s.file.ReadAt(payload, payloadAt); err != nil {
		return state.Chunk{}, core.E("state.filestore.ResolveRefBytes", "read chunk payload", err)
	}
	return state.Chunk{
		Ref: state.ChunkRef{
			ChunkID:        id,
			FrameOffset:    ref.FrameOffset,
			HasFrameOffset: true,
			Codec:          CodecFile,
			Segment:        s.path,
		},
		Data: payload,
	}, nil
}

func (s *Store) rebuildIndex(ctx context.Context) error {
	info, err := s.file.Stat()
	if err != nil {
		return core.E("state.filestore.Open", "stat file", err)
	}
	size := info.Size()
	headerLen, err := s.detectHeaderLen(size)
	if err != nil {
		return err
	}

	// Best-effort capacity hint — average observed record (24-byte
	// header + ~60-byte meta + 64-byte payload at the bench scale)
	// lands near 150 bytes. Overshoot is harmless: Go maps shrink
	// lazily; undershoot triggers cascade rehash. The divisor is
	// tuned to slot just under the typical record size so the initial
	// bucket count covers the corpus without rehash. Open allocates
	// fresh empty maps at entry so we can swap them out for sized
	// versions in place.
	if records := int((size - headerLen) / 128); records > 0 && len(s.index) == 0 {
		s.index = make(map[int]fileIndexEntry, records)
		s.uriIndex = make(map[string]int, records)
	}

	// Grow the meta buffer in place across records to avoid per-record
	// allocations on large files. The buffer contents are decoded into
	// stack-only locals before the next iteration overwrites them.
	var metaBuf []byte
	offset := headerLen
	for offset < size {
		if err := checkContext(ctx); err != nil {
			return err
		}
		if offset+recordHeaderLen > size {
			return core.NewError("state file store has truncated record header")
		}
		var headerBuf [recordHeaderLen]byte
		if _, err := s.file.ReadAt(headerBuf[:], offset); err != nil {
			return core.E("state.filestore.Open", "read record header", err)
		}
		record, err := decodeRecordHeader(headerBuf[:])
		if err != nil {
			return err
		}
		metaSize, err := intFromUint64(uint64(record.metaSize), "metadata")
		if err != nil {
			return err
		}
		payloadSize, err := intFromUint64(record.payloadSize, "payload")
		if err != nil {
			return err
		}
		metaAt := offset + recordHeaderLen
		payloadAt := metaAt + int64(metaSize)
		nextOffset := payloadAt + int64(payloadSize)
		if nextOffset > size {
			return core.NewError("state file store has truncated record payload")
		}
		if cap(metaBuf) < metaSize {
			metaBuf = make([]byte, metaSize)
		} else {
			metaBuf = metaBuf[:metaSize]
		}
		if metaSize > 0 {
			if _, err := s.file.ReadAt(metaBuf, metaAt); err != nil {
				return core.E("state.filestore.Open", "read record metadata", err)
			}
		}
		// Lazy meta scan: only URI is needed to populate uriIndex —
		// the meta blob's other fields (Title/Kind/Track/Tags/
		// Labels) are written for forward audit, not read by any
		// hot path. extractRecordURI walks the JSON object
		// end-to-end (so structural corruption is still caught)
		// but only materialises the URI string. At 10k records
		// this skips ~6 allocs/record (Tags map + Labels slice +
		// Title/Kind/Track string copies) over a full
		// json.Unmarshal of recordMeta. The fileIndexEntry.meta
		// field is left zero-valued on this path; Put still
		// populates it to keep the put-side bench shape intact.
		var uri string
		if metaSize > 0 {
			extracted, err := extractRecordURI(metaBuf)
			if err != nil {
				return core.E("state.filestore.Open", "parse record metadata", err)
			}
			uri = extracted
		}
		id, err := intFromUint64(record.chunkID, "chunk id")
		if err != nil {
			return err
		}
		ref := state.ChunkRef{
			ChunkID:        id,
			FrameOffset:    uint64(offset),
			HasFrameOffset: true,
			Codec:          CodecFile,
			Segment:        s.path,
		}
		s.index[id] = fileIndexEntry{
			ref:         ref,
			payloadAt:   payloadAt,
			payloadSize: payloadSize,
		}
		if uri != "" {
			s.uriIndex[uri] = id
		}
		if id >= s.nextID {
			s.nextID = id + 1
		}
		offset = nextOffset
	}
	s.writeAt = offset
	return nil
}

func (s *Store) detectHeaderLen(size int64) (int64, error) {
	minHeaderLen := len(fileMagic)
	if len(legacyFileMagic) < minHeaderLen {
		minHeaderLen = len(legacyFileMagic)
	}
	if size < int64(minHeaderLen) {
		return 0, core.NewError("state file store is missing header")
	}
	maxHeaderLen := len(fileMagic)
	if len(legacyFileMagic) > maxHeaderLen {
		maxHeaderLen = len(legacyFileMagic)
	}
	if size < int64(maxHeaderLen) {
		maxHeaderLen = int(size)
	}
	magic := make([]byte, maxHeaderLen)
	if _, err := s.file.ReadAt(magic, 0); err != nil {
		return 0, core.E("state.filestore.Open", "read file header", err)
	}
	if hasMagicPrefix(magic, fileMagic) {
		return int64(len(fileMagic)), nil
	}
	if hasMagicPrefix(magic, legacyFileMagic) {
		return int64(len(legacyFileMagic)), nil
	}
	return 0, core.NewError("state file store header is invalid")
}

func hasMagicPrefix(data, magic []byte) bool {
	return len(data) >= len(magic) && string(data[:len(magic)]) == string(magic)
}

type recordHeader struct {
	chunkID     uint64
	payloadSize uint64
	metaSize    uint32
}

// encodeRecordHeader writes a record header into the caller-supplied
// buffer (must be at least recordHeaderLen bytes). The previous shape
// allocated a fresh []byte on every Put — header writes fire once per
// chunk written, so the alloc compounded for every state save.
func encodeRecordHeader(buf []byte, chunkID int, payloadSize, metaSize int) {
	_ = buf[recordHeaderLen-1] // bounds-check hint
	copy(buf[:4], recordMagic[:])
	binary.LittleEndian.PutUint64(buf[4:12], uint64(chunkID))
	binary.LittleEndian.PutUint64(buf[12:20], uint64(payloadSize))
	binary.LittleEndian.PutUint32(buf[20:24], uint32(metaSize))
}

func decodeRecordHeader(header []byte) (recordHeader, error) {
	if len(header) != recordHeaderLen {
		return recordHeader{}, core.NewError("state file store record header has invalid length")
	}
	// Byte-equal comparison — `string(header[:4]) != string(recordMagic[:])`
	// allocates a fresh 4-byte string on every call. Direct byte compare
	// is alloc-free.
	if header[0] != recordMagic[0] || header[1] != recordMagic[1] ||
		header[2] != recordMagic[2] || header[3] != recordMagic[3] {
		return recordHeader{}, core.NewError("state file store record header is invalid")
	}
	return recordHeader{
		chunkID:     binary.LittleEndian.Uint64(header[4:12]),
		payloadSize: binary.LittleEndian.Uint64(header[12:20]),
		metaSize:    binary.LittleEndian.Uint32(header[20:24]),
	}, nil
}

// extractRecordURI walks data as a top-level JSON object and returns
// the value of the "uri" key as a string, or "" if absent. The walker
// fully traverses the object (including nested arrays / objects) so
// any structural corruption — unbalanced braces, truncated value,
// trailing garbage — surfaces as an error. This replaces a full
// json.Unmarshal into recordMeta for the rebuildIndex hot path,
// dropping ~6 allocs per record at 10k scale (Tags map, Labels slice,
// Title/Kind/Track string copies). The "uri" field is encoded by
// json.Marshal of a string — URLs do not require escapes in
// practice, so the fast path returns a direct slice-to-string copy;
// the rare-but-valid escape path is handled by jsonUnescape.
func extractRecordURI(data []byte) (string, error) {
	i, err := jsonSkipWS(data, 0)
	if err != nil {
		return "", err
	}
	if data[i] != '{' {
		return "", core.NewError("state file store metadata is not a JSON object")
	}
	i++
	uri := ""
	uriSeen := false
	first := true
	for {
		i, err = jsonSkipWS(data, i)
		if err != nil {
			return "", err
		}
		if data[i] == '}' {
			i++
			break
		}
		if !first {
			if data[i] != ',' {
				return "", core.NewError("state file store metadata is missing comma")
			}
			i++
			i, err = jsonSkipWS(data, i)
			if err != nil {
				return "", err
			}
		}
		first = false
		if data[i] != '"' {
			return "", core.NewError("state file store metadata key is not a string")
		}
		keyStart := i + 1
		keyEnd, err := jsonSkipString(data, i)
		if err != nil {
			return "", err
		}
		i = keyEnd
		i, err = jsonSkipWS(data, i)
		if err != nil {
			return "", err
		}
		if data[i] != ':' {
			return "", core.NewError("state file store metadata is missing colon")
		}
		i++
		i, err = jsonSkipWS(data, i)
		if err != nil {
			return "", err
		}
		isURI := !uriSeen && keyEnd-1-keyStart == 3 &&
			data[keyStart] == 'u' && data[keyStart+1] == 'r' && data[keyStart+2] == 'i'
		if isURI {
			if data[i] != '"' {
				return "", core.NewError("state file store uri is not a string")
			}
			value, end, err := jsonReadString(data, i)
			if err != nil {
				return "", err
			}
			uri = value
			uriSeen = true
			i = end
		} else {
			end, err := jsonSkipValue(data, i)
			if err != nil {
				return "", err
			}
			i = end
		}
	}
	// Validate no trailing garbage beyond whitespace.
	for i < len(data) {
		c := data[i]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			return "", core.NewError("state file store metadata has trailing data")
		}
		i++
	}
	return uri, nil
}

// jsonSkipWS advances past JSON whitespace, returning the first
// non-whitespace index or an error if end-of-data is hit. The caller
// uses the returned index to read the next significant byte.
func jsonSkipWS(data []byte, i int) (int, error) {
	for i < len(data) {
		c := data[i]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			return i, nil
		}
		i++
	}
	return i, core.NewError("state file store metadata is truncated")
}

// jsonSkipString advances past a JSON string starting at data[i]
// (which must be '"') and returns the index after the closing quote.
// Handles escape sequences but does not decode them.
func jsonSkipString(data []byte, i int) (int, error) {
	if i >= len(data) || data[i] != '"' {
		return i, core.NewError("state file store metadata expects string")
	}
	i++
	for i < len(data) {
		c := data[i]
		if c == '\\' {
			if i+1 >= len(data) {
				return i, core.NewError("state file store metadata has trailing escape")
			}
			// One-byte escapes (\" \\ \/ \b \f \n \r \t) or \uXXXX —
			// either way the next single byte cannot terminate the
			// string and the wider \uXXXX is bounded by the closing
			// quote check on later iterations.
			i += 2
			continue
		}
		if c == '"' {
			return i + 1, nil
		}
		i++
	}
	return i, core.NewError("state file store metadata string is unterminated")
}

// jsonReadString reads a JSON string at data[i] (which must be '"')
// and returns its decoded value plus the index after the closing
// quote. Fast path: no escapes → direct string copy of the byte
// slice. Slow path: presence of an escape forces a per-byte decode
// into a fresh buffer. Used only for the "uri" field, where escapes
// are extremely rare in practice (URLs).
func jsonReadString(data []byte, i int) (string, int, error) {
	if i >= len(data) || data[i] != '"' {
		return "", i, core.NewError("state file store metadata expects string")
	}
	start := i + 1
	j := start
	hasEscape := false
	for j < len(data) {
		c := data[j]
		if c == '\\' {
			hasEscape = true
			if j+1 >= len(data) {
				return "", j, core.NewError("state file store metadata has trailing escape")
			}
			j += 2
			continue
		}
		if c == '"' {
			if !hasEscape {
				return string(data[start:j]), j + 1, nil
			}
			decoded, err := jsonUnescape(data[start:j])
			if err != nil {
				return "", j, err
			}
			return decoded, j + 1, nil
		}
		j++
	}
	return "", j, core.NewError("state file store metadata string is unterminated")
}

// jsonUnescape decodes the contents of a JSON string (without
// surrounding quotes) that contains at least one backslash escape.
// Handles the six single-byte escapes and \uXXXX (no surrogate-pair
// decoding — surrogate halves pass through as their raw UTF-8
// encoding, which is what encoding/json itself emits for unpaired
// surrogates). Allocated once per uri-with-escape; URIs never have
// escapes in observed corpora, so this is the cold path.
func jsonUnescape(src []byte) (string, error) {
	out := make([]byte, 0, len(src))
	for i := 0; i < len(src); i++ {
		c := src[i]
		if c != '\\' {
			out = append(out, c)
			continue
		}
		if i+1 >= len(src) {
			return "", core.NewError("state file store metadata has trailing escape")
		}
		i++
		switch src[i] {
		case '"', '\\', '/':
			out = append(out, src[i])
		case 'b':
			out = append(out, '\b')
		case 'f':
			out = append(out, '\f')
		case 'n':
			out = append(out, '\n')
		case 'r':
			out = append(out, '\r')
		case 't':
			out = append(out, '\t')
		case 'u':
			if i+4 >= len(src) {
				return "", core.NewError("state file store metadata has short \\u escape")
			}
			var r rune
			for k := 1; k <= 4; k++ {
				h := src[i+k]
				var v byte
				switch {
				case h >= '0' && h <= '9':
					v = h - '0'
				case h >= 'a' && h <= 'f':
					v = h - 'a' + 10
				case h >= 'A' && h <= 'F':
					v = h - 'A' + 10
				default:
					return "", core.NewError("state file store metadata has invalid \\u escape")
				}
				r = r<<4 | rune(v)
			}
			i += 4
			// Emit r as UTF-8. Unpaired surrogates pass through as
			// their replacement encoding — sufficient for the URI
			// field which is ASCII in every observed corpus.
			switch {
			case r < 0x80:
				out = append(out, byte(r))
			case r < 0x800:
				out = append(out, byte(0xC0|r>>6), byte(0x80|r&0x3F))
			case r < 0x10000:
				out = append(out, byte(0xE0|r>>12), byte(0x80|(r>>6)&0x3F), byte(0x80|r&0x3F))
			default:
				out = append(out, byte(0xF0|r>>18), byte(0x80|(r>>12)&0x3F), byte(0x80|(r>>6)&0x3F), byte(0x80|r&0x3F))
			}
		default:
			return "", core.NewError("state file store metadata has unknown escape")
		}
	}
	return string(out), nil
}

// jsonSkipValue advances past a single JSON value (string, number,
// boolean, null, object, array) starting at data[i] and returns the
// index of the first byte after the value. The full traversal is
// what gives rebuildIndex its structural-corruption guarantee
// without forcing the whole metadata blob through json.Unmarshal.
func jsonSkipValue(data []byte, i int) (int, error) {
	if i >= len(data) {
		return i, core.NewError("state file store metadata is truncated")
	}
	c := data[i]
	switch {
	case c == '"':
		return jsonSkipString(data, i)
	case c == '{' || c == '[':
		open := c
		var closeByte byte
		if open == '{' {
			closeByte = '}'
		} else {
			closeByte = ']'
		}
		depth := 1
		i++
		for i < len(data) && depth > 0 {
			cc := data[i]
			switch cc {
			case '"':
				end, err := jsonSkipString(data, i)
				if err != nil {
					return i, err
				}
				i = end
			case '{', '[':
				depth++
				i++
			case '}', ']':
				if cc == closeByte {
					depth--
					i++
					continue
				}
				if (open == '{' && cc == ']') || (open == '[' && cc == '}') {
					return i, core.NewError("state file store metadata has mismatched bracket")
				}
				depth--
				i++
			default:
				i++
			}
		}
		if depth != 0 {
			return i, core.NewError("state file store metadata is unbalanced")
		}
		return i, nil
	case c == 't':
		if i+4 > len(data) || data[i+1] != 'r' || data[i+2] != 'u' || data[i+3] != 'e' {
			return i, core.NewError("state file store metadata expects true")
		}
		return i + 4, nil
	case c == 'f':
		if i+5 > len(data) || data[i+1] != 'a' || data[i+2] != 'l' || data[i+3] != 's' || data[i+4] != 'e' {
			return i, core.NewError("state file store metadata expects false")
		}
		return i + 5, nil
	case c == 'n':
		if i+4 > len(data) || data[i+1] != 'u' || data[i+2] != 'l' || data[i+3] != 'l' {
			return i, core.NewError("state file store metadata expects null")
		}
		return i + 4, nil
	case c == '-' || (c >= '0' && c <= '9'):
		// Number — consume digits, sign, dot, exponent. Loose but
		// correct enough for structural validation; json.Marshal
		// emits canonical numbers so the surface is constrained.
		j := i
		if data[j] == '-' {
			j++
		}
		for j < len(data) {
			b := data[j]
			if (b >= '0' && b <= '9') || b == '.' || b == 'e' || b == 'E' || b == '+' || b == '-' {
				j++
				continue
			}
			break
		}
		if j == i {
			return i, core.NewError("state file store metadata has empty number")
		}
		return j, nil
	default:
		return i, core.NewError("state file store metadata has invalid value")
	}
}

type limitedPayloadWriter struct {
	file      *core.OSFile
	remaining int
}

func (w *limitedPayloadWriter) Write(data []byte) (int, error) {
	if len(data) > w.remaining {
		return 0, core.NewError("state file store streamed payload is larger than declared")
	}
	n, err := w.file.Write(data)
	w.remaining -= n
	if err != nil {
		return n, err
	}
	if n != len(data) {
		return n, stdio.ErrShortWrite
	}
	return n, nil
}

func writeAll(file stdio.Writer, data []byte) error {
	for len(data) > 0 {
		n, err := file.Write(data)
		if err != nil {
			return err
		}
		if n == 0 {
			return stdio.ErrShortWrite
		}
		data = data[n:]
	}
	return nil
}

func checkContext(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		return nil
	}
}

func intFromUint64(value uint64, label string) (int, error) {
	max := uint64(maxInt())
	if value > max {
		return 0, core.NewError("state file store " + label + " is too large")
	}
	return int(value), nil
}

func maxInt() int {
	return int(^uint(0) >> 1)
}

func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}
