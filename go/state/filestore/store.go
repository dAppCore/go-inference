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
	// recordMagicU32 is the little-endian uint32 view of recordMagic,
	// pre-computed once at init. decodeRecordHeader's magic check
	// previously walked the 4-byte header byte-by-byte; rebuildIndex
	// runs that check per record at 10k+ scale during cold Open, so
	// folding the 4-way compare into one Uint32 read trims one ALU
	// op per record.
	recordMagicU32 = binary.LittleEndian.Uint32(recordMagic[:])

	// emptyMetaBytes is the canonical empty-record-meta JSON blob.
	// PutBytesStream shortcuts to this slice when no meta field is
	// populated, skipping core.JSONMarshal entirely — encoding/json
	// allocates an encoder + grow-doubled output buffer per call
	// (~5550 B / 4-9 allocs) even for an all-zero struct. Reference
	// types like this share safely because the surface is read-only
	// across writeAll → file.Write.
	emptyMetaBytes = []byte("{}")

	// errStoreClosed is the canonical post-Close error returned by
	// every Resolve/Put gate. Sharing a single &core.Err{...} skips
	// the per-call heap alloc that core.NewError("...") otherwise
	// fires. The error is read-only after init — Err's Message field
	// is set once here and never mutated; Error() is pure derivation.
	// Callers compare via errors.Is(err, nil) or string-equality on
	// .Error(), neither of which depends on pointer identity, so the
	// sharing is safe across goroutines.
	errStoreClosed           = core.NewError("state file store is closed")
	errStoreNil              = core.NewError("state file store is nil")
	errPayloadSizeInvalid    = core.NewError("state file store payload size is invalid")
	errStreamWriterNil       = core.NewError("state file store stream writer is nil")
	errMetadataTooLarge      = core.NewError("state file store metadata is too large")
	errPayloadShort          = core.NewError("state file store streamed payload is shorter than declared")
	errPayloadOversize       = core.NewError("state file store streamed payload is larger than declared")
	errRefNonFileCodec       = core.NewError("state file store cannot resolve non-file chunk ref")
	errRefSegmentMismatch    = core.NewError("state file store chunk ref segment mismatch")
	errRefFrameOffsetTooBig  = core.NewError("state file store frame offset is too large")
	errRefChunkIDMismatch    = core.NewError("state file store chunk ref id mismatch")
)

type Store struct {
	mu       sync.Mutex
	path     string
	file     *core.OSFile
	index    map[int]fileIndexEntry
	uriIndex map[string]int
	nextID   int
	writeAt  int64
	// payloadWriter is the per-Store streaming bound writer reused
	// across PutBytesStream calls. Holding it on the Store skips
	// the &limitedPayloadWriter{...} alloc every Put paid for the
	// closure dispatch (the writer escaped to heap once per call).
	// The mutex above already serialises PutBytesStream so the
	// embedded writer's remaining counter is single-owner during
	// any one call.
	payloadWriter limitedPayloadWriter
	// headerMetaBuf is the per-Store scratch buffer that
	// encodeRecordHeaderMeta builds the on-disk header + meta
	// JSON into. The previous shape allocated a fresh buffer on
	// every PutBytesStream (~49 B for the Kind-only common shape,
	// up to a few hundred B for label-heavy meta). Reusing the
	// buffer under mu skips the per-Put alloc; the slice header
	// is single-owner during any one Put because the mutex above
	// already serialises the entire write path.
	//
	// Lifetime: the buffer is read by writeAll(file, ...) before
	// PutBytesStream returns, so its content is consumed before
	// the next Put can reuse the storage. Length is reset to zero
	// on entry to encodeRecordHeaderMeta so each Put builds
	// fresh contents over the retained capacity.
	headerMetaBuf []byte
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
		return state.Chunk{}, errStoreClosed
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
		return state.Chunk{}, errStoreClosed
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
		return state.ChunkRef{}, errStoreNil
	}
	if payloadSize < 0 {
		return state.ChunkRef{}, errPayloadSizeInvalid
	}
	if write == nil {
		return state.ChunkRef{}, errStreamWriterNil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.file == nil {
		return state.ChunkRef{}, errStoreClosed
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
	// buildHeaderMeta packs the 24-byte record header and
	// the JSON-encoded recordMeta into the per-Store scratch
	// buffer (s.headerMetaBuf). The previous shape allocated a
	// fresh buffer per Put; reusing under mu skips that. The
	// metaSize uint32 in the header is patched after the meta
	// is appended — single-pass build.
	headerMeta := s.buildHeaderMeta(&meta, id, payloadSize)
	metaSize := len(headerMeta) - recordHeaderLen
	if uint64(metaSize) > uint64(^uint32(0)) {
		return state.ChunkRef{}, errMetadataTooLarge
	}
	offset := s.writeAt
	if _, err := s.file.Seek(offset, stdio.SeekStart); err != nil {
		return state.ChunkRef{}, core.E("state.filestore.Put", "seek to append offset", err)
	}
	if err := writeAll(s.file, headerMeta); err != nil {
		s.rollbackWriteLocked(offset)
		return state.ChunkRef{}, core.E("state.filestore.Put", "write record header and metadata", err)
	}
	s.payloadWriter.file = s.file
	s.payloadWriter.remaining = payloadSize
	if err := write(&s.payloadWriter); err != nil {
		s.rollbackWriteLocked(offset)
		return state.ChunkRef{}, core.E("state.filestore.Put", "write record payload", err)
	}
	if s.payloadWriter.remaining != 0 {
		s.rollbackWriteLocked(offset)
		return state.ChunkRef{}, errPayloadShort
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
		payloadAt:   offset + recordHeaderLen + int64(metaSize),
		payloadSize: payloadSize,
	}
	if meta.URI != "" {
		s.uriIndex[meta.URI] = id
	}
	s.nextID++
	s.writeAt += int64(recordHeaderLen + metaSize + payloadSize)
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
		return state.Chunk{}, errStoreClosed
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
		return state.Chunk{}, errRefNonFileCodec
	}
	if ref.Segment != "" && ref.Segment != s.path {
		return state.Chunk{}, errRefSegmentMismatch
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.file == nil {
		return state.Chunk{}, errStoreClosed
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
		return state.Chunk{}, errRefFrameOffsetTooBig
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
		return state.Chunk{}, errRefChunkIDMismatch
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
	// Magic-prefix check via a single Uint32 read against the
	// pre-computed recordMagicU32 — one ALU op per record at the
	// rebuildIndex 10k-scale cold Open, where the previous 4-byte
	// branching compare emitted 4 cmpb + 3 brand merges. Folding
	// the 32 bits into a single equality test also lets the
	// compiler hoist the magic constant into an immediate operand.
	// `string(header[:4]) != string(recordMagic[:])` would allocate
	// a fresh 4-byte string on every call.
	if binary.LittleEndian.Uint32(header[:4]) != recordMagicU32 {
		return recordHeader{}, core.NewError("state file store record header is invalid")
	}
	return recordHeader{
		chunkID:     binary.LittleEndian.Uint64(header[4:12]),
		payloadSize: binary.LittleEndian.Uint64(header[12:20]),
		metaSize:    binary.LittleEndian.Uint32(header[20:24]),
	}, nil
}

// recordMetaIsEmpty reports whether the record meta has no
// populated field — string fields all empty, Tags map nil or empty,
// Labels slice nil or empty. The PutBytesStream fast path uses this
// to short-circuit JSON marshalling on records that carry no caller
// metadata (the common shape for KV snapshots and sentinel writes).
//
//	if recordMetaIsEmpty(&meta) {
//	    metaBytes = emptyMetaBytes
//	}
func recordMetaIsEmpty(meta *recordMeta) bool {
	return meta.URI == "" &&
		meta.Title == "" &&
		meta.Kind == "" &&
		meta.Track == "" &&
		len(meta.Tags) == 0 &&
		len(meta.Labels) == 0
}

// encodeRecordMeta hand-rolls the JSON for recordMeta into a fresh
// single-allocation buffer. Thin wrapper over appendRecordMeta — kept
// as the package-private "I want the meta bytes" entry point, used
// by the round-trip test surface and any future caller that does
// not also need the record header in the same buffer.
//
// PutBytesStream itself routes through (*Store).buildHeaderMeta which
// folds the meta append into the per-Store scratch buffer, dropping
// the alloc entirely on the warm path.
//
//	buf := encodeRecordMeta(&meta)
//	if uint64(len(buf)) > uint64(^uint32(0)) { /* too large */ }
func encodeRecordMeta(meta *recordMeta) []byte {
	if recordMetaIsEmpty(meta) {
		return emptyMetaBytes
	}
	buf := make([]byte, 0, recordMetaCapHint(meta))
	return appendRecordMeta(buf, meta)
}

// buildHeaderMeta builds the on-disk record header + JSON-encoded
// recordMeta into the per-Store scratch buffer (s.headerMetaBuf),
// returning a slice into that buffer. The previous shape allocated
// a fresh buffer per Put — measurable on the state-checkpoint
// fast path because Put fires per Save during a generation step
// and per KV-snapshot during a session.
//
// PutBytesStream holds s.mu for the full record write, so the
// scratch buffer is single-owner during any one Put; the next Put
// reuses the underlying storage after the previous call's
// writeAll consumed the bytes. encodeRecordHeader (called below)
// is a pure-write helper — no further alloc beyond the slice
// header reuse.
//
// The metaSize uint32 in the header is patched after the meta is
// appended — single-pass build, no double walk over the meta
// fields. The slice retains its growth across Puts so the typical
// meta size + the cap hint converge after a handful of records.
//
// encoding/json.Marshal on recordMeta allocates an encoder state
// machine + grow-doubled output buffer + per-tag key/value copies
// on every Put. The hand-roll lands at zero buffer allocations
// regardless of tag count.
//
// The meta portion is valid JSON, parseable by encoding/json
// (round-trips into recordMeta) and by the store's extractRecordURI
// walker. Field ordering follows recordMeta's struct declaration —
// URI, Title, Kind, Track, Tags, Labels — and the omitempty
// semantics match (zero-value strings, nil/empty maps, nil/empty
// slices are elided). Tag-map keys are emitted in Go map iteration
// order — JSON object key order is not semantically meaningful and
// no read site depends on it.
//
//	buf := s.buildHeaderMeta(&meta, chunkID, payloadSize)
//	writeAll(s.file, buf)
func (s *Store) buildHeaderMeta(meta *recordMeta, chunkID, payloadSize int) []byte {
	need := recordHeaderLen + recordMetaCapHint(meta)
	if cap(s.headerMetaBuf) < need {
		s.headerMetaBuf = make([]byte, recordHeaderLen, need)
	} else {
		s.headerMetaBuf = s.headerMetaBuf[:recordHeaderLen]
	}
	s.headerMetaBuf = appendRecordMeta(s.headerMetaBuf, meta)
	metaSize := len(s.headerMetaBuf) - recordHeaderLen
	encodeRecordHeader(s.headerMetaBuf[:recordHeaderLen], chunkID, payloadSize, metaSize)
	return s.headerMetaBuf
}

// recordMetaCapHint returns a tight upper bound on the JSON byte
// length of meta. Each non-empty field contributes its raw byte
// length plus framing overhead (the surrounding "key":"value",
// pair, with a small slack so the heuristic clears the typical
// ASCII shape in one allocation). Pathological escape-heavy inputs
// (control chars, embedded quotes) let append grow once.
func recordMetaCapHint(meta *recordMeta) int {
	if recordMetaIsEmpty(meta) {
		return 2
	}
	size := 2 // outer braces
	if meta.URI != "" {
		size += 10 + len(meta.URI) // `"uri":"<v>",` = 9 bytes + value, +1 slack
	}
	if meta.Title != "" {
		size += 12 + len(meta.Title) // `"title":"<v>",`
	}
	if meta.Kind != "" {
		size += 11 + len(meta.Kind) // `"kind":"<v>",`
	}
	if meta.Track != "" {
		size += 12 + len(meta.Track) // `"track":"<v>",`
	}
	if len(meta.Tags) > 0 {
		size += 12 // `"tags":{...},`
		for k, v := range meta.Tags {
			size += 6 + len(k) + len(v) // `"k":"v",`
		}
	}
	if len(meta.Labels) > 0 {
		size += 14 // `"labels":[...],`
		for _, l := range meta.Labels {
			size += 4 + len(l) // `"l",`
		}
	}
	return size
}

// appendRecordMeta appends the JSON encoding of meta to buf and
// returns the extended slice. Walks the recordMeta struct in
// declaration order, eliding empty fields to honour the omitempty
// json tag semantics. Single-pass; no allocation beyond the
// caller-supplied buf's eventual grow.
func appendRecordMeta(buf []byte, meta *recordMeta) []byte {
	if recordMetaIsEmpty(meta) {
		return append(buf, '{', '}')
	}
	buf = append(buf, '{')
	first := true
	if meta.URI != "" {
		buf = appendJSONField(buf, "uri", meta.URI, first)
		first = false
	}
	if meta.Title != "" {
		buf = appendJSONField(buf, "title", meta.Title, first)
		first = false
	}
	if meta.Kind != "" {
		buf = appendJSONField(buf, "kind", meta.Kind, first)
		first = false
	}
	if meta.Track != "" {
		buf = appendJSONField(buf, "track", meta.Track, first)
		first = false
	}
	if len(meta.Tags) > 0 {
		if !first {
			buf = append(buf, ',')
		}
		first = false
		buf = append(buf, `"tags":{`...)
		tagFirst := true
		for k, v := range meta.Tags {
			if !tagFirst {
				buf = append(buf, ',')
			}
			tagFirst = false
			buf = appendJSONString(buf, k)
			buf = append(buf, ':')
			buf = appendJSONString(buf, v)
		}
		buf = append(buf, '}')
	}
	if len(meta.Labels) > 0 {
		if !first {
			buf = append(buf, ',')
		}
		buf = append(buf, `"labels":[`...)
		for i, l := range meta.Labels {
			if i > 0 {
				buf = append(buf, ',')
			}
			buf = appendJSONString(buf, l)
		}
		buf = append(buf, ']')
	}
	return append(buf, '}')
}

// appendJSONField appends a "key":"value" pair (prefixed by a comma
// when not the first field) to buf. Key is ASCII-only and not
// escaped — recordMeta keys are compile-time constants.
func appendJSONField(buf []byte, key, value string, first bool) []byte {
	if !first {
		buf = append(buf, ',')
	}
	buf = append(buf, '"')
	buf = append(buf, key...)
	buf = append(buf, '"', ':')
	return appendJSONString(buf, value)
}

// appendJSONString appends a JSON-encoded string to buf — opening
// quote, escaped body, closing quote. Escapes match the subset
// recognised by extractRecordURI's jsonUnescape walker: \" \\ \b
// \f \n \r \t for the canonical mnemonic forms and \u00XX for
// other control chars (< 0x20). All bytes ≥ 0x20 outside the
// quote / backslash pair pass through verbatim — encoding/json's
// default also escapes <, >, & for HTML safety but the read path
// does not, and the on-disk record is not consumed by HTML
// contexts.
//
// The body walk batches runs of non-escape bytes into a single
// append per span, so a typical URI / Title / Kind value (no
// escapes) collapses to one append-string call rather than N
// append-byte calls. encoding/json's own writer emits the no-
// escape path the same way; the per-byte loop here was an artefact
// of the original simple shape.
func appendJSONString(buf []byte, s string) []byte {
	buf = append(buf, '"')
	start := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		// Fast-path predicate: any byte ≥ 0x20 that is neither '"'
		// nor '\\' passes through verbatim. The boolean short-
		// circuits left-to-right and the compiler emits two CMPs
		// + AND, cheaper than the previous per-byte switch dispatch.
		if c >= 0x20 && c != '"' && c != '\\' {
			continue
		}
		// Flush the verbatim span up to but not including the
		// escape byte. The span is empty on the first escape at
		// position 0; append-zero-length is a no-op.
		if start < i {
			buf = append(buf, s[start:i]...)
		}
		switch c {
		case '"':
			buf = append(buf, '\\', '"')
		case '\\':
			buf = append(buf, '\\', '\\')
		case '\b':
			buf = append(buf, '\\', 'b')
		case '\f':
			buf = append(buf, '\\', 'f')
		case '\n':
			buf = append(buf, '\\', 'n')
		case '\r':
			buf = append(buf, '\\', 'r')
		case '\t':
			buf = append(buf, '\\', 't')
		default:
			// c < 0x20 and not one of the mnemonic escapes — emit
			// \u00XX. Hex digits emitted lowercase to match the
			// jsonUnescape reader and encoding/json output.
			buf = append(buf, '\\', 'u', '0', '0', hexChar(c>>4), hexChar(c&0x0f))
		}
		start = i + 1
	}
	if start < len(s) {
		buf = append(buf, s[start:]...)
	}
	return append(buf, '"')
}

// hexChar returns the ASCII hex digit for the low nibble of v.
func hexChar(v byte) byte {
	v &= 0x0f
	if v < 10 {
		return '0' + v
	}
	return 'a' + (v - 10)
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
		return 0, errPayloadOversize
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
