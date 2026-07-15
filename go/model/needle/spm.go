// SPDX-Licence-Identifier: EUPL-1.2

package needle

import (
	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
)

// metaSpace is SentencePiece's whitespace marker U+2581 ("▁"): the tokenizer
// replaces every space with it (escape_whitespaces) and prepends one as a dummy
// prefix (add_dummy_prefix).
const metaSpace = "▁"

// spmTokenizer is a minimal SentencePiece BPE encoder/decoder built from a
// checkpoint's tokenizer.model. Needle's tokenizer.model is a byte-fallback BPE
// model with add_dummy_prefix, escape_whitespaces and remove_extra_whitespaces
// on, identity char-normalisation. That is exactly the subset implemented here —
// enough to reproduce the reference token stream, not a general SPM.
type spmTokenizer struct {
	pieces      []string       // id -> piece text
	scores      []float32      // id -> BPE merge score (higher merges first)
	types       []int32        // id -> SentencePiece token-type code
	pieceID     map[string]int // piece text -> id (all pieces, for emit)
	firstByteID int            // id of "<0x00>"; byte b -> firstByteID + b
}

// loadSPM reads a tokenizer.model (SentencePiece ModelProto) via the shared
// reader and indexes it for BPE encoding.
//
//	tok, err := loadSPM("/models/needle/tokenizer.model", cfg)
func loadSPM(path string, _ Config) (*spmTokenizer, error) {
	pieces, err := tokenizer.ReadSentencePieceModel(path)
	if err != nil {
		return nil, core.E("needle.loadSPM", "read "+path, err)
	}
	t := &spmTokenizer{
		pieces:      make([]string, len(pieces)),
		scores:      make([]float32, len(pieces)),
		types:       make([]int32, len(pieces)),
		pieceID:     make(map[string]int, len(pieces)),
		firstByteID: -1,
	}
	for i, p := range pieces {
		t.pieces[i] = p.Piece
		t.scores[i] = p.Score
		t.types[i] = p.Type
		t.pieceID[p.Piece] = i
		if t.firstByteID < 0 && p.Type == tokenizer.SPMTokenByte {
			t.firstByteID = i
		}
	}
	if t.firstByteID < 0 {
		return nil, core.E("needle.loadSPM", "tokenizer.model has no byte pieces (byte_fallback expected)", nil)
	}
	return t, nil
}

// normalize applies the SentencePiece front-end for Needle's settings:
// collapse space runs and trim (remove_extra_whitespaces), prepend one space
// (add_dummy_prefix), then map every space to the U+2581 marker
// (escape_whitespaces). Char normalisation is identity, so text is otherwise
// untouched. Only ASCII space (0x20) is treated as whitespace, which is all the
// reference inputs carry.
func (t *spmTokenizer) normalize(text string) string {
	collapsed := make([]byte, 0, len(text))
	prevSpace := true // leading spaces are dropped
	for i := range len(text) {
		if text[i] == ' ' {
			if !prevSpace {
				collapsed = append(collapsed, ' ')
			}
			prevSpace = true
			continue
		}
		collapsed = append(collapsed, text[i])
		prevSpace = false
	}
	if n := len(collapsed); n > 0 && collapsed[n-1] == ' ' {
		collapsed = collapsed[:n-1]
	}
	out := make([]byte, 0, len(collapsed)+len(metaSpace))
	out = append(out, metaSpace...) // dummy prefix
	for i := range len(collapsed) {
		if collapsed[i] == ' ' {
			out = append(out, metaSpace...)
			continue
		}
		out = append(out, collapsed[i])
	}
	return string(out)
}

// encode tokenises text into ids: normalise, split into runes, greedily merge the
// adjacent pair whose merged piece has the highest score (leftmost on ties), then
// emit — a known piece as its id, an unknown residual rune byte-by-byte via the
// byte pieces (byte_fallback). This is the standard SentencePiece BPE encode.
//
//	tok.encode("What is the weather in San Francisco?") // [4279 743 302 ...]
func (t *spmTokenizer) encode(text string) []int {
	norm := t.normalize(text)

	symbols := make([]string, 0, len(norm))
	for _, r := range norm {
		symbols = append(symbols, string(r))
	}

	for len(symbols) > 1 {
		bestScore := float32(0)
		bestIdx := -1
		for i := range len(symbols) - 1 {
			merged := symbols[i] + symbols[i+1]
			id, ok := t.pieceID[merged]
			if !ok || t.types[id] != tokenizer.SPMTokenNormal {
				continue // merges only ever produce learned (NORMAL) pieces
			}
			if bestIdx < 0 || t.scores[id] > bestScore {
				bestScore = t.scores[id]
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		symbols[bestIdx] += symbols[bestIdx+1]
		symbols = append(symbols[:bestIdx+1], symbols[bestIdx+2:]...)
	}

	ids := make([]int, 0, len(symbols))
	for _, s := range symbols {
		if id, ok := t.pieceID[s]; ok {
			ids = append(ids, id)
			continue
		}
		for i := range len(s) { // byte_fallback: raw UTF-8 bytes -> <0xNN>
			ids = append(ids, t.firstByteID+int(s[i]))
		}
	}
	return ids
}

// decode turns ids back into text: byte pieces accumulate raw bytes, control
// tokens (<pad>/<s>/</s>) render nothing, everything else emits its piece with
// the U+2581 marker mapped back to a space. A single leading space (the dummy
// prefix) is stripped, matching SentencePiece's DecodePieces.
//
//	tok.decode([]int{356, 294, 264}) // `[{"`
func (t *spmTokenizer) decode(ids []int) string {
	out := make([]byte, 0, len(ids)*4)
	for _, id := range ids {
		if id < 0 || id >= len(t.pieces) {
			continue
		}
		switch t.types[id] {
		case tokenizer.SPMTokenByte:
			out = append(out, byte(id-t.firstByteID))
		case tokenizer.SPMTokenControl:
			// <pad>/<s>/</s> have no surface form.
		default:
			for _, r := range t.pieces[id] {
				if r == '▁' {
					out = append(out, ' ')
					continue
				}
				out = append(out, string(r)...)
			}
		}
	}
	if len(out) > 0 && out[0] == ' ' {
		out = out[1:]
	}
	return string(out)
}
