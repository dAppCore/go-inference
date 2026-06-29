// SPDX-Licence-Identifier: EUPL-1.2

// filestore index rebuild: cold-Open offset index scan, header detection and region/offset maths.
package filestore

import (
	"context"
	stdio "io"

	core "dappco.re/go"
	"dappco.re/go/inference/state"
)

func indexCapacityHint(size, headerLen int64) int {
	recordBytes := size - headerLen
	if recordBytes <= 0 || recordBytes > indexHintMaxFileBytes {
		return 0
	}
	records := recordBytes / indexHintRecordBytes
	if records <= 0 {
		return 0
	}
	return int(records)
}

func (s *Store) rebuildIndex(ctx context.Context) error {
	info, err := s.file.Stat()
	if err != nil {
		return core.E("state.filestore.Open", "stat file", err)
	}
	size, err := s.regionSize(info.Size())
	if err != nil {
		return err
	}
	headerLen, err := s.detectHeaderLen(size)
	if err != nil {
		return err
	}

	// Best-effort capacity hint for small-record stores. Do not derive map
	// capacity from arbitrarily large State files: packed KV containers can be
	// hundreds of MiB with only a few records, and byte-size preallocation turns
	// store-open into a large heap allocation before any payload is touched.
	if records := indexCapacityHint(size, headerLen); records > 0 && len(s.index) == 0 {
		s.index = make(map[int]fileIndexEntry, records)
		s.uriIndex = make(map[string]int, records)
	}

	// Prefetch buffer — read header + meta in a single ReadAt where
	// possible. Typical records have meta < ~200 bytes (URI + Kind +
	// short Title), so a 512-byte prefetch covers ~95% of records and
	// halves the syscall count over the rebuild. Records with bigger
	// meta fall back to the original two-ReadAt path; the cost there
	// is unchanged.
	//
	// The buffer is stack-allocated (gcflags confirms "does not escape")
	// because every byte read out of it is either parsed into a
	// stack-local recordHeader or copied into the URI string via
	// extractRecordURI. Each iteration overwrites it before the next.
	const prefetchSize = 512
	var prefetchBuf [prefetchSize]byte

	// Fallback meta buffer for records whose meta exceeds prefetchSize.
	// Grows in place across records to avoid per-record allocations on
	// the rare-but-not-impossible big-meta corpus. The buffer contents
	// are decoded into stack-only locals before the next iteration
	// overwrites them.
	var metaBuf []byte
	offset := headerLen
	for offset < size {
		if err := checkContext(ctx); err != nil {
			return err
		}
		if offset+recordHeaderLen > size {
			return core.NewError("state file store has truncated record header")
		}
		// Read header + the first prefetchSize-recordHeaderLen bytes
		// of meta in one syscall. ReadAt returns short at EOF for the
		// final record — that's harmless because n is then used as
		// the length of the readable view and we know the meta size
		// from the parsed header. The kernel page cache makes the
		// extra-bytes cost negligible vs the syscall round-trip cost.
		want := int64(prefetchSize)
		if offset+want > size {
			want = size - offset
		}
		physicalOffset, err := s.physicalOffset(offset)
		if err != nil {
			return err
		}
		n, err := s.file.ReadAt(prefetchBuf[:want], physicalOffset)
		if err != nil && err != stdio.EOF {
			return core.E("state.filestore.Open", "read record prefetch", err)
		}
		if n < recordHeaderLen {
			return core.NewError("state file store has truncated record header")
		}
		record, err := decodeRecordHeader(prefetchBuf[:recordHeaderLen])
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
		// Fast path: prefetch covered both header and meta. Hand
		// extractRecordURI a slice straight into prefetchBuf.
		var metaView []byte
		if metaSize == 0 {
			metaView = nil
		} else if recordHeaderLen+metaSize <= n {
			metaView = prefetchBuf[recordHeaderLen : recordHeaderLen+metaSize]
		} else {
			// Big-meta fallback — meta exceeds the prefetched span.
			// Re-read the meta into the growable metaBuf. Rare in
			// practice; size-grows are amortised across records.
			if cap(metaBuf) < metaSize {
				metaBuf = make([]byte, metaSize)
			} else {
				metaBuf = metaBuf[:metaSize]
			}
			metaPhysicalAt, err := s.physicalOffset(metaAt)
			if err != nil {
				return err
			}
			if _, err := s.file.ReadAt(metaBuf, metaPhysicalAt); err != nil {
				return core.E("state.filestore.Open", "read record metadata", err)
			}
			metaView = metaBuf
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
			extracted, err := extractRecordURI(metaView)
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
			payloadAt:   s.baseAt + payloadAt,
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
	if _, err := s.file.ReadAt(magic, s.baseAt); err != nil {
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

func (s *Store) regionSize(fileSize int64) (int64, error) {
	if s == nil || s.baseAt < 0 || s.region < 0 || s.baseAt > fileSize {
		return 0, errRegionInvalid
	}
	available := fileSize - s.baseAt
	if s.region == 0 {
		return available, nil
	}
	if s.region > available {
		return 0, errRegionInvalid
	}
	return s.region, nil
}

func (s *Store) physicalOffset(logOffset int64) (int64, error) {
	if s == nil || logOffset < 0 {
		return 0, errRegionInvalid
	}
	if s.region > 0 && logOffset > s.region {
		return 0, errRegionInvalid
	}
	if s.baseAt > 0 && logOffset > (1<<63-1)-s.baseAt {
		return 0, errRegionInvalid
	}
	return s.baseAt + logOffset, nil
}

func hasMagicPrefix(data, magic []byte) bool {
	return len(data) >= len(magic) && string(data[:len(magic)]) == string(magic)
}
