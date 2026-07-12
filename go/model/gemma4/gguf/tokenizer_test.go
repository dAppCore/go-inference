// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "testing"

// gemma4TestTokenizer is a compact tokenizer.json exercising every token-type
// class: control/special added tokens (ids 0-5), one USER_DEFINED non-special
// added token (id 9), a byte-fallback token (id 6) and plain normal tokens.
const gemma4TestTokenizer = `{
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<eos>", "special": true},
    {"id": 2, "content": "<bos>", "special": true},
    {"id": 3, "content": "<unk>", "special": true},
    {"id": 4, "content": "<mask>", "special": true},
    {"id": 5, "content": "<turn|>", "special": true},
    {"id": 9, "content": "<custom>", "special": false}
  ],
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4,
      "<turn|>": 5, "<0x0A>": 6, "▁the": 7, "cat": 8, "<custom>": 9
    },
    "merges": [["▁", "the"], ["c", "at"]]
  }
}`

func gemma4TokenizerEntry(t *testing.T, entries []MetadataEntry, key string) MetadataEntry {
	t.Helper()
	for _, e := range entries {
		if e.Key == key {
			return e
		}
	}
	t.Fatalf("tokenizer key %q not found", key)
	return MetadataEntry{}
}

// TestGemma4Tokenizer_gemma4Tokenizer_Tokens checks the token list is
// id-ordered and the model tag is gemma4.
func TestGemma4Tokenizer_gemma4Tokenizer_Tokens(t *testing.T) {
	entries, err := gemma4Tokenizer([]byte(gemma4TestTokenizer))
	if err != nil {
		t.Fatalf("gemma4Tokenizer: %v", err)
	}
	if m := gemma4TokenizerEntry(t, entries, "tokenizer.ggml.model"); m.Value.(string) != "gemma4" {
		t.Errorf("model = %v, want gemma4", m.Value)
	}
	tokens := gemma4TokenizerEntry(t, entries, "tokenizer.ggml.tokens").Value.([]string)
	want := []string{"<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<turn|>", "<0x0A>", "▁the", "cat", "<custom>"}
	if len(tokens) != len(want) {
		t.Fatalf("tokens len = %d, want %d", len(tokens), len(want))
	}
	for i := range want {
		if tokens[i] != want[i] {
			t.Errorf("tokens[%d] = %q, want %q", i, tokens[i], want[i])
		}
	}
}

// TestGemma4Tokenizer_gemma4Tokenizer_TokenTypes checks every classification:
// byte-fallback BYTE, special added CONTROL, non-special added USER_DEFINED,
// plain NORMAL, plus the uniform score.
func TestGemma4Tokenizer_gemma4Tokenizer_TokenTypes(t *testing.T) {
	entries, err := gemma4Tokenizer([]byte(gemma4TestTokenizer))
	if err != nil {
		t.Fatalf("gemma4Tokenizer: %v", err)
	}
	types := gemma4TokenizerEntry(t, entries, "tokenizer.ggml.token_type").Value.([]int32)
	wantType := []int32{
		gemma4TokenTypeControl,     // 0 <pad>
		gemma4TokenTypeControl,     // 1 <eos>
		gemma4TokenTypeControl,     // 2 <bos>
		gemma4TokenTypeControl,     // 3 <unk>
		gemma4TokenTypeControl,     // 4 <mask>
		gemma4TokenTypeControl,     // 5 <turn|>
		gemma4TokenTypeByte,        // 6 <0x0A>
		gemma4TokenTypeNormal,      // 7 ▁the
		gemma4TokenTypeNormal,      // 8 cat
		gemma4TokenTypeUserDefined, // 9 <custom>
	}
	for i := range wantType {
		if types[i] != wantType[i] {
			t.Errorf("token_type[%d] = %d, want %d", i, types[i], wantType[i])
		}
	}
	scores := gemma4TokenizerEntry(t, entries, "tokenizer.ggml.scores").Value.([]float32)
	for i, s := range scores {
		if s != gemma4TokenScore {
			t.Errorf("scores[%d] = %g, want %g", i, s, gemma4TokenScore)
		}
	}
}

// TestGemma4Tokenizer_gemma4Tokenizer_Merges checks merge pairs are space-joined.
func TestGemma4Tokenizer_gemma4Tokenizer_Merges(t *testing.T) {
	entries, err := gemma4Tokenizer([]byte(gemma4TestTokenizer))
	if err != nil {
		t.Fatalf("gemma4Tokenizer: %v", err)
	}
	merges := gemma4TokenizerEntry(t, entries, "tokenizer.ggml.merges").Value.([]string)
	want := []string{"▁ the", "c at"}
	if len(merges) != len(want) {
		t.Fatalf("merges len = %d, want %d", len(merges), len(want))
	}
	for i := range want {
		if merges[i] != want[i] {
			t.Errorf("merges[%d] = %q, want %q", i, merges[i], want[i])
		}
	}
}

// TestGemma4Tokenizer_gemma4Tokenizer_SpecialIDs checks the special token ids
// resolve from the vocab (eos is the end-of-turn token) and the add-bos /
// add-space-prefix flags carry the gemma defaults.
func TestGemma4Tokenizer_gemma4Tokenizer_SpecialIDs(t *testing.T) {
	entries, err := gemma4Tokenizer([]byte(gemma4TestTokenizer))
	if err != nil {
		t.Fatalf("gemma4Tokenizer: %v", err)
	}
	wantID := map[string]uint32{
		"tokenizer.ggml.bos_token_id":     2,
		"tokenizer.ggml.eos_token_id":     5, // <turn|>, not <eos> (id 1)
		"tokenizer.ggml.unknown_token_id": 3,
		"tokenizer.ggml.padding_token_id": 0,
		"tokenizer.ggml.mask_token_id":    4,
	}
	for key, want := range wantID {
		if got := gemma4TokenizerEntry(t, entries, key).Value.(uint32); got != want {
			t.Errorf("%s = %d, want %d", key, got, want)
		}
	}
	if got := gemma4TokenizerEntry(t, entries, "tokenizer.ggml.add_bos_token").Value.(bool); got != true {
		t.Errorf("add_bos_token = %v, want true", got)
	}
	if got := gemma4TokenizerEntry(t, entries, "tokenizer.ggml.add_space_prefix").Value.(bool); got != false {
		t.Errorf("add_space_prefix = %v, want false", got)
	}
}

// TestGemma4Tokenizer_gemma4Tokenizer_Bad rejects malformed tokenizer.json:
// non-JSON, a vocab id gap, a missing special token, and a malformed merge.
func TestGemma4Tokenizer_gemma4Tokenizer_Bad(t *testing.T) {
	for name, js := range map[string]string{
		"not json":       "{ not json",
		"empty vocab":    `{"model": {"vocab": {}}}`,
		"vocab gap":      `{"model": {"vocab": {"<bos>": 0, "<unk>": 2}}}`,
		"missing bos":    `{"model": {"vocab": {"<eos>": 0, "<turn|>": 1}}}`,
		"bad merge pair": `{"model": {"vocab": {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4, "<turn|>": 5}, "merges": [["only-one"]]}}`,
	} {
		if _, err := gemma4Tokenizer([]byte(js)); err == nil {
			t.Errorf("gemma4Tokenizer(%s): want error, got nil", name)
		}
	}
}

// TestGemma4Tokenizer_gemma4IsByteToken checks byte-fallback token detection.
func TestGemma4Tokenizer_gemma4IsByteToken(t *testing.T) {
	yes := []string{"<0x00>", "<0xFF>", "<0x0a>", "<0xAb>"}
	no := []string{"<0xGG>", "<0x1>", "<0x123>", "▁the", "<bos>", "", "0x0A"}
	for _, s := range yes {
		if !gemma4IsByteToken(s) {
			t.Errorf("gemma4IsByteToken(%q) = false, want true", s)
		}
	}
	for _, s := range no {
		if gemma4IsByteToken(s) {
			t.Errorf("gemma4IsByteToken(%q) = true, want false", s)
		}
	}
}
