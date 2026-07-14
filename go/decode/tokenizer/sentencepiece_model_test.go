// SPDX-Licence-Identifier: EUPL-1.2

package tokenizer

import (
	"encoding/binary"
	"math"
	"testing"
)

// spmPiece builds one SentencePiece submessage's protobuf bytes: piece (field
// 1, string), score (field 2, fixed32 float) and type (field 3, varint). Scores
// and lengths in the tests stay small enough that every varint is a single
// byte, keeping the fixtures readable.
func spmPiece(piece string, score float32, ptype int32) []byte {
	body := []byte{0x0A, byte(len(piece))}
	body = append(body, piece...)
	var scoreBuf [4]byte
	binary.LittleEndian.PutUint32(scoreBuf[:], math.Float32bits(score))
	body = append(body, 0x15)
	body = append(body, scoreBuf[:]...)
	body = append(body, 0x18, byte(ptype))
	return body
}

// spmPieceNoScoreType builds a piece submessage carrying only the piece text —
// score and type fields omitted — to exercise the proto2 defaults.
func spmPieceNoScoreType(piece string) []byte {
	body := []byte{0x0A, byte(len(piece))}
	return append(body, piece...)
}

// spmModel wraps piece submessages as the repeated field-1 entries of a
// ModelProto.
func spmModel(pieces ...[]byte) []byte {
	var out []byte
	for _, p := range pieces {
		out = append(out, 0x0A, byte(len(p)))
		out = append(out, p...)
	}
	return out
}

func TestSentencePieceModel_ParseSentencePieceModel_Good(t *testing.T) {
	data := spmModel(
		spmPiece("<pad>", 0, SPMTokenControl),
		spmPiece("▁the", -12.5, SPMTokenNormal),
		spmPiece("<0x0A>", -20, SPMTokenByte),
	)
	pieces, err := ParseSentencePieceModel(data)
	if err != nil {
		t.Fatalf("ParseSentencePieceModel: %v", err)
	}
	if len(pieces) != 3 {
		t.Fatalf("got %d pieces, want 3", len(pieces))
	}
	want := []SentencePiece{
		{Piece: "<pad>", Score: 0, Type: SPMTokenControl},
		{Piece: "▁the", Score: -12.5, Type: SPMTokenNormal},
		{Piece: "<0x0A>", Score: -20, Type: SPMTokenByte},
	}
	for i, w := range want {
		if pieces[i] != w {
			t.Errorf("piece[%d] = %+v, want %+v", i, pieces[i], w)
		}
	}
}

func TestSentencePieceModel_ParseSentencePieceModel_Bad(t *testing.T) {
	// A pieces field (0x0A) declaring a 20-byte submessage in a 4-byte buffer.
	data := []byte{0x0A, 0x14, 0x00, 0x00}
	if _, err := ParseSentencePieceModel(data); err == nil {
		t.Fatal("ParseSentencePieceModel accepted truncated bytes, want error")
	}
}

func TestSentencePieceModel_ParseSentencePieceModel_Ugly(t *testing.T) {
	// A piece with no score/type field (proto2 defaults apply) preceded by an
	// unknown top-level field (field 2, varint) the reader must skip.
	unknownTopLevel := []byte{0x10, 0x2A} // field 2, wire 0 (varint) = 42
	data := append(unknownTopLevel, spmModel(spmPieceNoScoreType("lone"))...)
	pieces, err := ParseSentencePieceModel(data)
	if err != nil {
		t.Fatalf("ParseSentencePieceModel: %v", err)
	}
	if len(pieces) != 1 {
		t.Fatalf("got %d pieces, want 1", len(pieces))
	}
	got := pieces[0]
	if got.Piece != "lone" || got.Score != 0 || got.Type != SPMTokenNormal {
		t.Errorf("default piece = %+v, want {lone 0 NORMAL(1)}", got)
	}
}

func TestSentencePieceModel_ReadSentencePieceModel_Missing(t *testing.T) {
	if _, err := ReadSentencePieceModel(t.TempDir() + "/absent.model"); err == nil {
		t.Fatal("ReadSentencePieceModel accepted a missing file, want error")
	}
}
