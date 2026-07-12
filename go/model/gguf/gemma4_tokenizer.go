// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import core "dappco.re/go"

// gemma-4 GGML token-type codes (llama.cpp's llama_token_type). Norm tokens are
// NORMAL, the SentencePiece byte-fallback tokens (<0xHH>) are BYTE, and added /
// special tokens are CONTROL.
const (
	gemma4TokenTypeNormal      int32   = 1
	gemma4TokenTypeControl     int32   = 3
	gemma4TokenTypeUserDefined int32   = 4
	gemma4TokenTypeByte        int32   = 6
	gemma4TokenScore           float32 = -1000.0
)

// gemma-4 special token names, resolved to ids through the checkpoint's vocab.
// The chat EOS is the end-of-turn closer <turn|>: generation_config.json lists
// eos_token_id [1, 106, 50] but the oracle (and gemma-it generation) uses the
// end-of-turn token, so we resolve it by name rather than pick from the list.
const (
	gemma4BOSTokenName  = "<bos>"
	gemma4EOSTokenName  = "<turn|>"
	gemma4UnkTokenName  = "<unk>"
	gemma4PadTokenName  = "<pad>"
	gemma4MaskTokenName = "<mask>"
)

// gemma4DefaultAddBOSToken / gemma4DefaultAddSpacePrefix are gemma-family
// tokenizer conventions the oracle carries and tokenizer_config.json does not
// override: prepend BOS, do not add a leading space. Documented defaults with
// the oracle values, not derived from the source (the source is silent on both).
const (
	gemma4DefaultAddBOSToken    = true
	gemma4DefaultAddSpacePrefix = false
)

// gemma4TokenizerJSON is the subset of a gemma-4 tokenizer.json the GGUF
// tokenizer block needs: the BPE vocab (token -> id), the merge list, and the
// added/special token records that drive token-type classification.
type gemma4TokenizerJSON struct {
	Model struct {
		Vocab  map[string]int `json:"vocab"`
		Merges [][]string     `json:"merges"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
}

// gemma4Tokenizer builds the tokenizer.ggml.* header block for a gemma-4 text
// GGUF from the checkpoint's tokenizer.json, mirroring the unsloth
// gemma-4-E2B-it-Q4_K_M oracle: the model tag, the id-ordered token list, a
// uniform score array, the per-token type array (byte-fallback tokens BYTE,
// added/special tokens CONTROL, the rest NORMAL), the space-joined merge list,
// the bos/eos/unk/pad/mask ids and the add-bos / add-space-prefix flags.
//
// Divergence from the oracle, documented rather than faked: the oracle further
// splits seven of the added tokens into USER_DEFINED and marks <eos> NORMAL —
// that split comes from the original SentencePiece proto's per-token type,
// which tokenizer.json does not carry (it marks every added token special). The
// difference does not affect tokenisation of ordinary text.
func gemma4Tokenizer(tokenizerJSON []byte) ([]MetadataEntry, error) {
	var tok gemma4TokenizerJSON
	if r := core.JSONUnmarshal(tokenizerJSON, &tok); !r.OK {
		return nil, core.E("gemma4Tokenizer", "parse tokenizer.json", r.Err())
	}
	vocabSize := len(tok.Model.Vocab)
	if vocabSize == 0 {
		return nil, core.NewError("gguf: gemma4 tokenizer.json has an empty vocab")
	}

	// Invert vocab (token -> id) into an id-ordered token list, rejecting any
	// out-of-range id, duplicate id, or gap so the array is exactly the vocab.
	tokens := make([]string, vocabSize)
	seen := make([]bool, vocabSize)
	for token, id := range tok.Model.Vocab {
		if id < 0 || id >= vocabSize {
			return nil, core.Errorf("gguf: gemma4 vocab id %d out of range [0,%d)", id, vocabSize)
		}
		if seen[id] {
			return nil, core.Errorf("gguf: gemma4 vocab id %d assigned to more than one token", id)
		}
		seen[id] = true
		tokens[id] = token
	}
	for id, filled := range seen {
		if !filled {
			return nil, core.Errorf("gguf: gemma4 vocab has no token for id %d", id)
		}
	}

	// Added-token type map: special -> CONTROL, otherwise USER_DEFINED.
	addedType := make(map[int]int32, len(tok.AddedTokens))
	for _, added := range tok.AddedTokens {
		if added.Special {
			addedType[added.ID] = gemma4TokenTypeControl
		} else {
			addedType[added.ID] = gemma4TokenTypeUserDefined
		}
	}

	scores := make([]float32, vocabSize)
	tokenType := make([]int32, vocabSize)
	for id, token := range tokens {
		scores[id] = gemma4TokenScore
		switch {
		case gemma4IsByteToken(token):
			tokenType[id] = gemma4TokenTypeByte
		default:
			if t, ok := addedType[id]; ok {
				tokenType[id] = t
			} else {
				tokenType[id] = gemma4TokenTypeNormal
			}
		}
	}

	merges := make([]string, len(tok.Model.Merges))
	for i, pair := range tok.Model.Merges {
		if len(pair) != 2 {
			return nil, core.Errorf("gguf: gemma4 merge %d is not a pair: %v", i, pair)
		}
		merges[i] = core.Concat(pair[0], " ", pair[1])
	}

	bos, err := gemma4LookupToken(tok.Model.Vocab, gemma4BOSTokenName)
	if err != nil {
		return nil, err
	}
	eos, err := gemma4LookupToken(tok.Model.Vocab, gemma4EOSTokenName)
	if err != nil {
		return nil, err
	}
	unk, err := gemma4LookupToken(tok.Model.Vocab, gemma4UnkTokenName)
	if err != nil {
		return nil, err
	}
	pad, err := gemma4LookupToken(tok.Model.Vocab, gemma4PadTokenName)
	if err != nil {
		return nil, err
	}
	mask, err := gemma4LookupToken(tok.Model.Vocab, gemma4MaskTokenName)
	if err != nil {
		return nil, err
	}

	u32 := func(key string, v int) MetadataEntry {
		return MetadataEntry{Key: key, ValueType: ValueTypeUint32, Value: uint32(v)}
	}
	return []MetadataEntry{
		{Key: "tokenizer.ggml.model", ValueType: ValueTypeString, Value: gemma4Arch},
		{Key: "tokenizer.ggml.tokens", ValueType: ValueTypeArray, Value: tokens},
		{Key: "tokenizer.ggml.scores", ValueType: ValueTypeArray, Value: scores},
		{Key: "tokenizer.ggml.token_type", ValueType: ValueTypeArray, Value: tokenType},
		{Key: "tokenizer.ggml.merges", ValueType: ValueTypeArray, Value: merges},
		u32("tokenizer.ggml.bos_token_id", bos),
		u32("tokenizer.ggml.eos_token_id", eos),
		u32("tokenizer.ggml.unknown_token_id", unk),
		u32("tokenizer.ggml.padding_token_id", pad),
		u32("tokenizer.ggml.mask_token_id", mask),
		{Key: "tokenizer.ggml.add_bos_token", ValueType: ValueTypeBool, Value: gemma4DefaultAddBOSToken},
		{Key: "tokenizer.ggml.add_space_prefix", ValueType: ValueTypeBool, Value: gemma4DefaultAddSpacePrefix},
	}, nil
}

// gemma4LookupToken resolves a token name to its vocab id, erroring if the
// checkpoint's vocab does not carry it.
func gemma4LookupToken(vocab map[string]int, name string) (int, error) {
	if id, ok := vocab[name]; ok {
		return id, nil
	}
	return 0, core.NewError("gguf: gemma4 tokenizer vocab has no token " + name)
}

// gemma4IsByteToken reports whether token is a SentencePiece byte-fallback token
// of the form <0xHH> (two hex digits) — the 256 tokens llama.cpp marks BYTE.
//
//	gemma4IsByteToken("<0x0A>") // true
//	gemma4IsByteToken("▁the")   // false
func gemma4IsByteToken(token string) bool {
	if len(token) != 6 || token[0] != '<' || token[1] != '0' || token[2] != 'x' || token[5] != '>' {
		return false
	}
	return gemma4IsHexDigit(token[3]) && gemma4IsHexDigit(token[4])
}

func gemma4IsHexDigit(c byte) bool {
	return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
}
