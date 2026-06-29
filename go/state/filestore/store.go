// SPDX-Licence-Identifier: EUPL-1.2

// Package filestore provides an append-only file-backed state store.
package filestore

import (
	"context"
	"encoding/binary"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference/state"
)

const (
	CodecFile       = "state/file-log"
	CodecMemvidFile = "memvid/file-log"

	fileMode              = 0o600
	recordHeaderLen       = 24
	indexHintRecordBytes  = 128
	indexHintMaxFileBytes = 32 * 1024 * 1024
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
	errStoreClosed          = core.NewError("state file store is closed")
	errStoreNil             = core.NewError("state file store is nil")
	errPayloadSizeInvalid   = core.NewError("state file store payload size is invalid")
	errStreamWriterNil      = core.NewError("state file store stream writer is nil")
	errMetadataTooLarge     = core.NewError("state file store metadata is too large")
	errPayloadShort         = core.NewError("state file store streamed payload is shorter than declared")
	errPayloadOversize      = core.NewError("state file store streamed payload is larger than declared")
	errRefNonFileCodec      = core.NewError("state file store cannot resolve non-file chunk ref")
	errRefSegmentMismatch   = core.NewError("state file store chunk ref segment mismatch")
	errRefFrameOffsetTooBig = core.NewError("state file store frame offset is too large")
	errRefChunkIDMismatch   = core.NewError("state file store chunk ref id mismatch")
	errStoreReadOnly        = core.NewError("state file store is read-only")
	errRegionInvalid        = core.NewError("state file store region is invalid")
	errMappedRegionInvalid  = core.NewError("state file store mapped region is invalid")
)

type Store struct {
	mu           sync.Mutex
	path         string
	alias        string
	file         *core.OSFile
	baseAt       int64
	region       int64
	readOnly     bool
	mapped       []byte
	mappedRegion []byte
	index        map[int]fileIndexEntry
	uriIndex     map[string]int
	nextID       int
	writeAt      int64
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
	return openWithSegmentAlias(ctx, path, "")
}

// OpenWithSegmentAlias reopens an existing append-only state file store and
// permits refs whose Segment names canonicalSegment. This keeps relocation
// explicit for container-mounted State files while preserving Open's strict
// default segment validation.
func OpenWithSegmentAlias(ctx context.Context, path string, canonicalSegment string) (*Store, error) {
	return openWithSegmentAlias(ctx, path, core.Trim(canonicalSegment))
}

// OpenRegionWithSegmentAlias opens an append-only state log embedded inside a
// larger file. Frame offsets remain relative to the embedded State payload,
// while Segment validation accepts canonicalSegment for relocated refs.
func OpenRegionWithSegmentAlias(ctx context.Context, path string, payloadOffset int64, payloadBytes int64, canonicalSegment string) (*Store, error) {
	return openRegionWithSegmentAlias(ctx, path, payloadOffset, payloadBytes, core.Trim(canonicalSegment), true)
}

func openWithSegmentAlias(ctx context.Context, path string, canonicalSegment string) (*Store, error) {
	return openRegionWithSegmentAlias(ctx, path, 0, 0, canonicalSegment, false)
}

func openRegionWithSegmentAlias(ctx context.Context, path string, payloadOffset int64, payloadBytes int64, canonicalSegment string, readOnly bool) (*Store, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if core.Trim(path) == "" {
		return nil, core.NewError("state file store path is required")
	}
	if payloadOffset < 0 || payloadBytes < 0 {
		return nil, errRegionInvalid
	}
	flags := core.O_RDWR
	if readOnly {
		flags = core.O_RDONLY
	}
	result := core.OpenFile(path, flags, fileMode)
	if !result.OK {
		return nil, core.E("state.filestore.Open", "open file", resultError(result))
	}
	file := result.Value.(*core.OSFile)
	store := &Store{
		path:     path,
		alias:    canonicalSegment,
		file:     file,
		baseAt:   payloadOffset,
		region:   payloadBytes,
		readOnly: readOnly,
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
	s.unmapRegionLocked()
	file := s.file
	s.file = nil
	return file.Close()
}
