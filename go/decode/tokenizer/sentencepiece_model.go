// SPDX-Licence-Identifier: EUPL-1.2

package tokenizer

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
)

// SentencePiece token-type codes — the SentencePiece ModelProto's own
// SentencePiece.Type enum. These values are identical to llama.cpp's ggml
// token-type codes (NORMAL=1 … BYTE=6), so an exporter can pass a piece's Type
// straight through into a GGUF tokenizer.ggml.token_type array.
const (
	SPMTokenNormal      int32 = 1
	SPMTokenUnknown     int32 = 2
	SPMTokenControl     int32 = 3
	SPMTokenUserDefined int32 = 4
	SPMTokenUnused      int32 = 5
	SPMTokenByte        int32 = 6
)

// SentencePiece is one vocabulary entry decoded from a SentencePiece
// ModelProto (a checkpoint's tokenizer.model file): the piece text, its
// unigram/merge-rank score, and its token type. It carries exactly the three
// fields a GGUF "llama" (SentencePiece-class) tokenizer header needs per token
// — the id-ordered token list, the score array, and the token_type array.
type SentencePiece struct {
	Piece string
	Score float32
	Type  int32
}

// ReadSentencePieceModel reads a SentencePiece ModelProto file (a checkpoint's
// tokenizer.model) and returns its pieces in vocabulary-id order — pieces[i] is
// the entry for token id i. It is the score/type source for exporting a GGUF
// "llama" tokenizer header (gemma, llama, mistral, …), which HF fast-tokenizer
// JSON (tokenizer.json) does not carry: a SentencePiece BPE model records each
// piece's negative merge rank as its score, and that ranking drives llama.cpp's
// SPM segmentation — a header without it detokenises but mis-tokenises.
//
//	pieces, err := tokenizer.ReadSentencePieceModel("/models/gemma3-1b/tokenizer.model")
//	// pieces[2] == {Piece: "<bos>", Score: 0, Type: SPMTokenControl}
func ReadSentencePieceModel(path string) ([]SentencePiece, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("ReadSentencePieceModel", "read tokenizer.model", read.Err())
	}
	return ParseSentencePieceModel(read.Bytes())
}

// ParseSentencePieceModel decodes the protobuf wire bytes of a SentencePiece
// ModelProto into its pieces. Only the repeated `pieces` field (ModelProto
// field 1) is read; every other top-level field (trainer_spec, normalizer_spec,
// …) is skipped by wire type. Within each SentencePiece submessage the three
// fields the exporter needs are decoded — piece (field 1, string), score
// (field 2, 32-bit float), type (field 3, varint enum) — and any further field
// is skipped. A truncated or malformed message is a loud error rather than a
// partial vocab, so a corrupt tokenizer.model cannot silently yield a
// short/garbled token list.
func ParseSentencePieceModel(data []byte) ([]SentencePiece, error) {
	var pieces []SentencePiece
	pos := 0
	for pos < len(data) {
		field, wire, next, err := spmReadTag(data, pos)
		if err != nil {
			return nil, err
		}
		pos = next
		// ModelProto field 1 = repeated SentencePiece pieces (length-delimited).
		if field == 1 && wire == 2 {
			body, after, err := spmReadBytes(data, pos)
			if err != nil {
				return nil, err
			}
			pos = after
			piece, err := spmParsePiece(body)
			if err != nil {
				return nil, err
			}
			pieces = append(pieces, piece)
			continue
		}
		pos, err = spmSkip(data, pos, wire)
		if err != nil {
			return nil, err
		}
	}
	return pieces, nil
}

// spmParsePiece decodes one SentencePiece submessage. A piece with no explicit
// type field defaults to NORMAL — the proto2 default for SentencePiece.Type —
// so an omitted type is a normal token, not a zero/unset one.
func spmParsePiece(body []byte) (SentencePiece, error) {
	piece := SentencePiece{Type: SPMTokenNormal}
	pos := 0
	for pos < len(body) {
		field, wire, next, err := spmReadTag(body, pos)
		if err != nil {
			return SentencePiece{}, err
		}
		pos = next
		switch {
		case field == 1 && wire == 2: // piece text
			text, after, err := spmReadBytes(body, pos)
			if err != nil {
				return SentencePiece{}, err
			}
			// Copy out of the file buffer so a stored piece does not pin the
			// whole tokenizer.model bytes alive.
			piece.Piece = string(text)
			pos = after
		case field == 2 && wire == 5: // score (fixed32 float)
			if pos+4 > len(body) {
				return SentencePiece{}, errSPMTruncated
			}
			piece.Score = math.Float32frombits(binary.LittleEndian.Uint32(body[pos:]))
			pos += 4
		case field == 3 && wire == 0: // type (varint enum)
			value, after, err := spmReadVarint(body, pos)
			if err != nil {
				return SentencePiece{}, err
			}
			piece.Type = int32(value)
			pos = after
		default:
			pos, err = spmSkip(body, pos, wire)
			if err != nil {
				return SentencePiece{}, err
			}
		}
	}
	return piece, nil
}

// errSPMTruncated is the single malformed-bytes sentinel — every bounds/parse
// failure in this reader surfaces as one error rather than a per-site string.
var errSPMTruncated = core.NewError("tokenizer: SentencePiece model bytes are truncated or malformed")

// spmReadTag decodes a protobuf field tag at pos, returning the field number,
// wire type, and the position just past the tag.
func spmReadTag(data []byte, pos int) (field, wire uint64, next int, err error) {
	tag, after, err := spmReadVarint(data, pos)
	if err != nil {
		return 0, 0, 0, err
	}
	return tag >> 3, tag & 7, after, nil
}

// spmReadVarint decodes a base-128 varint at pos.
func spmReadVarint(data []byte, pos int) (value uint64, next int, err error) {
	v, n := binary.Uvarint(data[pos:])
	if n <= 0 {
		return 0, 0, errSPMTruncated
	}
	return v, pos + n, nil
}

// spmReadBytes decodes a length-delimited field's body at pos (a varint length
// followed by that many bytes), returning the body slice and the position past
// it.
func spmReadBytes(data []byte, pos int) (body []byte, next int, err error) {
	length, after, err := spmReadVarint(data, pos)
	if err != nil {
		return nil, 0, err
	}
	end := after + int(length)
	if int(length) < 0 || end < after || end > len(data) {
		return nil, 0, errSPMTruncated
	}
	return data[after:end], end, nil
}

// spmSkip advances past a field value of the given wire type without decoding
// it — varint, 64-bit, length-delimited, and 32-bit are handled; group wire
// types (3, 4) and any unknown type are rejected (SentencePiece models use
// none).
func spmSkip(data []byte, pos int, wire uint64) (next int, err error) {
	switch wire {
	case 0: // varint
		_, after, err := spmReadVarint(data, pos)
		return after, err
	case 1: // 64-bit
		if pos+8 > len(data) {
			return 0, errSPMTruncated
		}
		return pos + 8, nil
	case 2: // length-delimited
		_, after, err := spmReadBytes(data, pos)
		return after, err
	case 5: // 32-bit
		if pos+4 > len(data) {
			return 0, errSPMTruncated
		}
		return pos + 4, nil
	default:
		return 0, errSPMTruncated
	}
}
