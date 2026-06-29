// SPDX-Licence-Identifier: EUPL-1.2

// filestore record read path: Get, Resolve, ResolveURI, ResolveBytes, ResolveRefBytes and the Borrow* helpers.
package filestore

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/state"
)

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

func (s *Store) BorrowBytes(ctx context.Context, chunkID int) (state.BorrowedChunk, error) {
	if err := checkContext(ctx); err != nil {
		return state.BorrowedChunk{}, err
	}
	if s == nil {
		return state.BorrowedChunk{}, &state.ChunkNotFoundError{ID: chunkID}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.file == nil {
		return state.BorrowedChunk{}, errStoreClosed
	}
	entry, ok := s.index[chunkID]
	if !ok {
		return state.BorrowedChunk{}, &state.ChunkNotFoundError{ID: chunkID}
	}
	if s.readOnly {
		payloadAt := entry.payloadAt - s.baseAt
		data, err := s.borrowPayloadLocked(payloadAt, entry.payloadSize)
		if err != nil {
			return state.BorrowedChunk{}, err
		}
		return state.BorrowedChunk{Ref: entry.ref, Data: data}, nil
	}
	chunk, err := s.resolveBytesLocked(chunkID)
	if err != nil {
		return state.BorrowedChunk{}, err
	}
	return state.BorrowedChunk{Ref: chunk.Ref, Data: chunk.Data}, nil
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
	if ref.Segment != "" && ref.Segment != s.path && ref.Segment != s.alias {
		return state.Chunk{}, errRefSegmentMismatch
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.file == nil {
		return state.Chunk{}, errStoreClosed
	}
	return s.resolveRefBytesLocked(ref)
}

func (s *Store) BorrowRefBytes(ctx context.Context, ref state.ChunkRef) (state.BorrowedChunk, error) {
	if err := checkContext(ctx); err != nil {
		return state.BorrowedChunk{}, err
	}
	if s == nil {
		return state.BorrowedChunk{}, &state.ChunkNotFoundError{ID: ref.ChunkID}
	}
	if !ref.HasFrameOffset {
		return s.BorrowBytes(ctx, ref.ChunkID)
	}
	if ref.Codec != "" && ref.Codec != CodecFile && ref.Codec != CodecMemvidFile {
		return state.BorrowedChunk{}, errRefNonFileCodec
	}
	if ref.Segment != "" && ref.Segment != s.path && ref.Segment != s.alias {
		return state.BorrowedChunk{}, errRefSegmentMismatch
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.file == nil {
		return state.BorrowedChunk{}, errStoreClosed
	}
	if !s.readOnly {
		chunk, err := s.resolveRefBytesLocked(ref)
		if err != nil {
			return state.BorrowedChunk{}, err
		}
		return state.BorrowedChunk{Ref: chunk.Ref, Data: chunk.Data}, nil
	}
	return s.borrowRefBytesLocked(ref)
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
	physicalOffset, err := s.physicalOffset(offset)
	if err != nil {
		return state.Chunk{}, err
	}
	var headerBuf [recordHeaderLen]byte
	if _, err := s.file.ReadAt(headerBuf[:], physicalOffset); err != nil {
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
	payloadAt := physicalOffset + recordHeaderLen + int64(metaSize)
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

func (s *Store) borrowRefBytesLocked(ref state.ChunkRef) (state.BorrowedChunk, error) {
	if ref.FrameOffset > uint64(maxInt()) {
		return state.BorrowedChunk{}, errRefFrameOffsetTooBig
	}
	offset := int64(ref.FrameOffset)
	var headerView []byte
	if err := s.ensureMappedRegionLocked(); err == nil {
		if offset < 0 || offset+recordHeaderLen > int64(len(s.mappedRegion)) {
			return state.BorrowedChunk{}, errRegionInvalid
		}
		headerView = s.mappedRegion[offset : offset+recordHeaderLen]
	} else {
		physicalOffset, perr := s.physicalOffset(offset)
		if perr != nil {
			return state.BorrowedChunk{}, perr
		}
		var headerBuf [recordHeaderLen]byte
		if _, rerr := s.file.ReadAt(headerBuf[:], physicalOffset); rerr != nil {
			return state.BorrowedChunk{}, core.E("state.filestore.BorrowRefBytes", "read record header", rerr)
		}
		headerView = headerBuf[:]
	}
	record, err := decodeRecordHeader(headerView)
	if err != nil {
		return state.BorrowedChunk{}, err
	}
	id, err := intFromUint64(record.chunkID, "chunk id")
	if err != nil {
		return state.BorrowedChunk{}, err
	}
	if ref.ChunkID != 0 && id != ref.ChunkID {
		return state.BorrowedChunk{}, errRefChunkIDMismatch
	}
	metaSize, err := intFromUint64(uint64(record.metaSize), "metadata")
	if err != nil {
		return state.BorrowedChunk{}, err
	}
	payloadSize, err := intFromUint64(record.payloadSize, "payload")
	if err != nil {
		return state.BorrowedChunk{}, err
	}
	payloadAt := offset + recordHeaderLen + int64(metaSize)
	data, err := s.borrowPayloadLocked(payloadAt, payloadSize)
	if err != nil {
		return state.BorrowedChunk{}, err
	}
	return state.BorrowedChunk{
		Ref: state.ChunkRef{
			ChunkID:        id,
			FrameOffset:    ref.FrameOffset,
			HasFrameOffset: true,
			Codec:          CodecFile,
			Segment:        s.path,
		},
		Data: data,
	}, nil
}

func (s *Store) borrowPayloadLocked(payloadAt int64, payloadSize int) ([]byte, error) {
	if payloadSize < 0 || payloadAt < 0 {
		return nil, errRegionInvalid
	}
	if err := s.ensureMappedRegionLocked(); err != nil {
		physicalAt, perr := s.physicalOffset(payloadAt)
		if perr != nil {
			return nil, perr
		}
		data := make([]byte, payloadSize)
		if _, rerr := s.file.ReadAt(data, physicalAt); rerr != nil {
			return nil, core.E("state.filestore.BorrowRefBytes", "read chunk payload", rerr)
		}
		return data, nil
	}
	end := payloadAt + int64(payloadSize)
	if end < payloadAt || end > int64(len(s.mappedRegion)) {
		return nil, errRegionInvalid
	}
	return s.mappedRegion[payloadAt:end], nil
}
