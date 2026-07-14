// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
	basegguf "dappco.re/go/inference/model/gguf"
)

// gemma3TestPiece builds one SentencePiece submessage: piece (field 1), score
// (field 2, fixed32 float) and type (field 3, varint). Test pieces stay short
// enough that every length varint is a single byte.
func gemma3TestPiece(piece string, score float32, ptype int32) []byte {
	body := []byte{0x0A, byte(len(piece))}
	body = append(body, piece...)
	var scoreBuf [4]byte
	binary.LittleEndian.PutUint32(scoreBuf[:], math.Float32bits(score))
	body = append(body, 0x15)
	body = append(body, scoreBuf[:]...)
	body = append(body, 0x18, byte(ptype))
	return body
}

// gemma3TestSPMModel wraps piece submessages as ModelProto field-1 entries.
func gemma3TestSPMModel(pieces ...[]byte) []byte {
	var out []byte
	for _, p := range pieces {
		out = append(out, 0x0A, byte(len(p)))
		out = append(out, p...)
	}
	return out
}

// gemma3TestVocab is a seven-piece SentencePiece vocab exercising every
// token-type class: control specials (ids 0-2), the unknown token (id 3), a
// normal token with a real score (id 4), a byte-fallback token (id 5), and a
// user-defined <unused> token (id 6) that the added_tokens_decoder reclassifies.
func gemma3TestVocab() []byte {
	return gemma3TestSPMModel(
		gemma3TestPiece("<pad>", 0, tokenizer.SPMTokenControl),
		gemma3TestPiece("<eos>", 0, tokenizer.SPMTokenControl),
		gemma3TestPiece("<bos>", 0, tokenizer.SPMTokenControl),
		gemma3TestPiece("<unk>", 0, tokenizer.SPMTokenUnknown),
		gemma3TestPiece("▁the", -5.5, tokenizer.SPMTokenNormal),
		gemma3TestPiece("<0x0A>", -10, tokenizer.SPMTokenByte),
		gemma3TestPiece("<unused9>", 0, tokenizer.SPMTokenUserDefined),
	)
}

// gemma3WriteTestPack writes a minimal gemma-3 pack (tokenizer.model +
// tokenizer_config.json + config.json) into dir and returns dir. added is the
// added_tokens_decoder JSON body; when empty the file omits the key.
func gemma3WriteTestPack(t *testing.T, dir string) {
	t.Helper()
	write := func(name string, data []byte) {
		if r := core.WriteFile(core.PathJoin(dir, name), data, 0o644); !r.OK {
			t.Fatalf("write %s: %v", name, r.Err())
		}
	}
	write("tokenizer.model", gemma3TestVocab())
	// <unused9> (id 6) is not flagged special, but does_token_look_special
	// classifies it CONTROL; ▁the (id 4) is not in the decoder and stays NORMAL.
	write("tokenizer_config.json", []byte(`{
	  "add_bos_token": true,
	  "add_eos_token": false,
	  "added_tokens_decoder": {
	    "6": {"content": "<unused9>", "special": false}
	  }
	}`))
	write("config.json", []byte(`{"bos_token_id":2,"eos_token_id":[1,106],"pad_token_id":0}`))
}

func gemma3TokEntry(t *testing.T, entries []basegguf.MetadataEntry, key string) basegguf.MetadataEntry {
	t.Helper()
	for _, e := range entries {
		if e.Key == key {
			return e
		}
	}
	t.Fatalf("missing tokenizer entry %q", key)
	return basegguf.MetadataEntry{}
}

func TestGemma3Tokenizer_gemma3Tokenizer_TokensScores(t *testing.T) {
	dir := t.TempDir()
	gemma3WriteTestPack(t, dir)
	entries, err := gemma3Tokenizer(dir)
	if err != nil {
		t.Fatalf("gemma3Tokenizer: %v", err)
	}
	if e := gemma3TokEntry(t, entries, "tokenizer.ggml.model"); e.Value.(string) != "llama" {
		t.Errorf("model = %v, want llama", e.Value)
	}
	if e := gemma3TokEntry(t, entries, "tokenizer.ggml.pre"); e.Value.(string) != "default" {
		t.Errorf("pre = %v, want default", e.Value)
	}
	tokens := gemma3TokEntry(t, entries, "tokenizer.ggml.tokens").Value.([]string)
	wantTokens := []string{"<pad>", "<eos>", "<bos>", "<unk>", "▁the", "<0x0A>", "<unused9>"}
	if len(tokens) != len(wantTokens) {
		t.Fatalf("got %d tokens, want %d", len(tokens), len(wantTokens))
	}
	for i, w := range wantTokens {
		if tokens[i] != w {
			t.Errorf("token[%d] = %q, want %q", i, tokens[i], w)
		}
	}
	scores := gemma3TokEntry(t, entries, "tokenizer.ggml.scores").Value.([]float32)
	// id 4 keeps its proto score; id 6 is overridden to the added-token sentinel.
	if scores[4] != -5.5 {
		t.Errorf("scores[4] = %v, want -5.5 (proto score)", scores[4])
	}
	if scores[6] != gemma3AddedTokenScore {
		t.Errorf("scores[6] = %v, want %v (added-token sentinel)", scores[6], gemma3AddedTokenScore)
	}
}

func TestGemma3Tokenizer_gemma3Tokenizer_TokenTypes(t *testing.T) {
	dir := t.TempDir()
	gemma3WriteTestPack(t, dir)
	entries, err := gemma3Tokenizer(dir)
	if err != nil {
		t.Fatalf("gemma3Tokenizer: %v", err)
	}
	types := gemma3TokEntry(t, entries, "tokenizer.ggml.token_type").Value.([]int32)
	want := []int32{
		tokenizer.SPMTokenControl, // <pad>
		tokenizer.SPMTokenControl, // <eos>
		tokenizer.SPMTokenControl, // <bos>
		tokenizer.SPMTokenUnknown, // <unk>
		tokenizer.SPMTokenNormal,  // ▁the
		tokenizer.SPMTokenByte,    // <0x0A>
		tokenizer.SPMTokenControl, // <unused9> — looks-special override
	}
	for i, w := range want {
		if types[i] != w {
			t.Errorf("token_type[%d] = %d, want %d", i, types[i], w)
		}
	}
}

func TestGemma3Tokenizer_gemma3Tokenizer_SpecialIDsAndFlags(t *testing.T) {
	dir := t.TempDir()
	gemma3WriteTestPack(t, dir)
	entries, err := gemma3Tokenizer(dir)
	if err != nil {
		t.Fatalf("gemma3Tokenizer: %v", err)
	}
	wantU32 := map[string]uint32{
		"tokenizer.ggml.bos_token_id":     2,
		"tokenizer.ggml.eos_token_id":     1, // first of [1, 106]
		"tokenizer.ggml.padding_token_id": 0,
		"tokenizer.ggml.unknown_token_id": 3, // config omits it → first UNKNOWN piece
	}
	for key, want := range wantU32 {
		if got := gemma3TokEntry(t, entries, key).Value.(uint32); got != want {
			t.Errorf("%s = %d, want %d", key, got, want)
		}
	}
	if v := gemma3TokEntry(t, entries, "tokenizer.ggml.add_bos_token").Value.(bool); !v {
		t.Error("add_bos_token = false, want true")
	}
	if v := gemma3TokEntry(t, entries, "tokenizer.ggml.add_eos_token").Value.(bool); v {
		t.Error("add_eos_token = true, want false")
	}
	if v := gemma3TokEntry(t, entries, "tokenizer.ggml.add_space_prefix").Value.(bool); v {
		t.Error("add_space_prefix = true, want false (gemma3 SPM convention)")
	}
}

func TestGemma3Tokenizer_gemma3Tokenizer_Bad(t *testing.T) {
	// No tokenizer.model — gemma3's SPM export path requires it.
	if _, err := gemma3Tokenizer(t.TempDir()); err == nil {
		t.Fatal("gemma3Tokenizer accepted a pack with no tokenizer.model, want error")
	}
}

func TestGemma3Tokenizer_gemma3LooksSpecial(t *testing.T) {
	special := []string{"<pad>", "<mask>", "<unused42>", "<|im_start|>"}
	normal := []string{"▁the", "cat", "<0x0A>", "hello"}
	for _, s := range special {
		if !gemma3LooksSpecial(s) {
			t.Errorf("gemma3LooksSpecial(%q) = false, want true", s)
		}
	}
	for _, s := range normal {
		if gemma3LooksSpecial(s) {
			t.Errorf("gemma3LooksSpecial(%q) = true, want false", s)
		}
	}
}
