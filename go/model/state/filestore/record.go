// SPDX-Licence-Identifier: EUPL-1.2

// filestore on-disk record codec: 24-byte record header and record-meta encode/decode.
package filestore

import (
	"encoding/binary"

	core "dappco.re/go"
)

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
