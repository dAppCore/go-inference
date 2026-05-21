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
	meta        recordMeta
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
		meta:        meta,
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
		metaBytes := make([]byte, metaSize)
		if _, err := s.file.ReadAt(metaBytes, metaAt); err != nil {
			return core.E("state.filestore.Open", "read record metadata", err)
		}
		var meta recordMeta
		if len(metaBytes) > 0 {
			result := core.JSONUnmarshal(metaBytes, &meta)
			if !result.OK {
				return core.E("state.filestore.Open", "parse record metadata", resultError(result))
			}
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
			meta:        meta,
		}
		if meta.URI != "" {
			s.uriIndex[meta.URI] = id
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
