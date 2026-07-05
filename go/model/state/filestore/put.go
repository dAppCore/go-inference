// SPDX-Licence-Identifier: EUPL-1.2

// filestore record write path: Put, PutBytes, PutBytesStream and write rollback.
package filestore

import (
	"context"
	stdio "io"

	core "dappco.re/go"
	"dappco.re/go/inference/model/state"
)

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
	if s.readOnly {
		return state.ChunkRef{}, errStoreReadOnly
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
	physicalOffset, err := s.physicalOffset(offset)
	if err != nil {
		return state.ChunkRef{}, err
	}
	if _, err := s.file.Seek(physicalOffset, stdio.SeekStart); err != nil {
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
	physicalOffset, err := s.physicalOffset(offset)
	if err != nil {
		return
	}
	_ = s.file.Truncate(physicalOffset)
	_, _ = s.file.Seek(physicalOffset, stdio.SeekStart)
}
