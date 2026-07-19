// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
	basegguf "dappco.re/go/inference/model/gguf"
)

// gemma3AddedTokenScore is the score llama.cpp's converter assigns to every
// added/special token (added_tokens.json + tokenizer_config's
// added_tokens_decoder) — a fixed sentinel, since an added token has no
// unigram/merge rank of its own.
const gemma3AddedTokenScore float32 = -1000.0

// gemma3AddedTokenDecoder is one entry of a tokenizer_config.json
// added_tokens_decoder map: the token's surface text and whether the checkpoint
// marks it special (a control token) rather than user-defined.
type gemma3AddedTokenDecoder struct {
	Content string `json:"content"`
	Special bool   `json:"special"`
}

// gemma3TokenizerConfig is the subset of tokenizer_config.json the GGUF
// tokenizer header needs: the add-bos/add-eos flags and the added-token type
// overrides. The flags are pointers so an absent key is distinguishable from an
// explicit false and falls back to the gemma default.
type gemma3TokenizerConfig struct {
	AddBosToken        *bool                              `json:"add_bos_token"`
	AddEosToken        *bool                              `json:"add_eos_token"`
	AddedTokensDecoder map[string]gemma3AddedTokenDecoder `json:"added_tokens_decoder"`
}

// gemma3SpecialIDs is the subset of config.json naming the special token ids.
// Each is decoded as `any` because a checkpoint may store an id either as a
// scalar or as a list (gemma-3's eos_token_id is [1, 106]); gemma3FirstInt
// takes the scalar or the list's first entry.
type gemma3SpecialIDs struct {
	BosTokenID any `json:"bos_token_id"`
	EosTokenID any `json:"eos_token_id"`
	PadTokenID any `json:"pad_token_id"`
	UnkTokenID any `json:"unk_token_id"`
}

// gemma3Tokenizer builds the tokenizer.ggml.* header block for a gemma-3 text
// GGUF from the checkpoint's tokenizer.model (a SentencePiece ModelProto),
// porting llama.cpp convert_hf_to_gguf.py's _set_vocab_sentencepiece: the
// "llama" model tag and "default" pre, the id-ordered token list, the
// per-token score and type arrays from the SentencePiece proto, the
// added_tokens.json and tokenizer_config added_tokens_decoder overrides
// (special/looks-special → CONTROL, otherwise USER_DEFINED, both scored with
// the added-token sentinel), the bos/eos/unk/pad ids and the add-bos/add-eos
// flags. add_space_prefix is false — gemma-3's SentencePiece convention, which
// Gemma3Model.set_vocab sets explicitly.
//
// gemma-3's scores are the SentencePiece BPE model's negative merge ranks, and
// llama.cpp's SPM segmentation ranks merges by them — a header without real
// scores loads but mis-tokenises, so the scores come from the proto, not from
// the score-less HF tokenizer.json.
func gemma3Tokenizer(root string) ([]basegguf.MetadataEntry, error) {
	pieces, err := tokenizer.ReadSentencePieceModel(core.PathJoin(root, "tokenizer.model"))
	if err != nil {
		return nil, core.E("gemma3Tokenizer", "read tokenizer.model (gemma3 requires a SentencePiece vocab)", err)
	}
	vocabSize := len(pieces)
	if vocabSize == 0 {
		return nil, core.NewError("gguf: gemma3 tokenizer.model has an empty vocab")
	}

	tokens := make([]string, vocabSize)
	scores := make([]float32, vocabSize)
	tokenType := make([]int32, vocabSize)
	for id, piece := range pieces {
		tokens[id] = piece.Piece
		scores[id] = piece.Score
		tokenType[id] = piece.Type
	}

	// added_tokens.json (name -> id): a plain user-defined override. Out-of-range
	// ids (e.g. a token that only exists once the embedding table is extended)
	// are ignored, matching the converter.
	if read := core.ReadFile(core.PathJoin(root, "added_tokens.json")); read.OK {
		added := map[string]int{}
		if r := core.JSONUnmarshal(read.Bytes(), &added); r.OK {
			for name, id := range added {
				if id >= 0 && id < vocabSize {
					tokens[id] = name
					scores[id] = gemma3AddedTokenScore
					tokenType[id] = tokenizer.SPMTokenUserDefined
				}
			}
		}
	}

	// tokenizer_config.json's added_tokens_decoder reclassifies added tokens:
	// special (or "looks special") -> CONTROL, otherwise USER_DEFINED.
	config, err := gemma3ReadTokenizerConfig(root)
	if err != nil {
		return nil, err
	}
	for idText, data := range config.AddedTokensDecoder {
		parsed := core.Atoi(idText)
		if !parsed.OK {
			continue
		}
		id := parsed.Int()
		if id < 0 || id >= vocabSize {
			continue
		}
		tokens[id] = data.Content
		scores[id] = gemma3AddedTokenScore
		if data.Special || gemma3LooksSpecial(data.Content) {
			tokenType[id] = tokenizer.SPMTokenControl
		} else {
			tokenType[id] = tokenizer.SPMTokenUserDefined
		}
	}

	entries := []basegguf.MetadataEntry{
		{Key: "tokenizer.ggml.model", ValueType: basegguf.ValueTypeString, Value: "llama"},
		{Key: "tokenizer.ggml.pre", ValueType: basegguf.ValueTypeString, Value: "default"},
		{Key: "tokenizer.ggml.tokens", ValueType: basegguf.ValueTypeArray, Value: tokens},
		{Key: "tokenizer.ggml.scores", ValueType: basegguf.ValueTypeArray, Value: scores},
		{Key: "tokenizer.ggml.token_type", ValueType: basegguf.ValueTypeArray, Value: tokenType},
	}
	entries = append(entries, gemma3SpecialTokenEntries(root, tokenType)...)
	entries = append(entries,
		basegguf.MetadataEntry{Key: "tokenizer.ggml.add_bos_token", ValueType: basegguf.ValueTypeBool, Value: gemma3BoolOr(config.AddBosToken, true)},
		basegguf.MetadataEntry{Key: "tokenizer.ggml.add_eos_token", ValueType: basegguf.ValueTypeBool, Value: gemma3BoolOr(config.AddEosToken, false)},
		basegguf.MetadataEntry{Key: "tokenizer.ggml.add_space_prefix", ValueType: basegguf.ValueTypeBool, Value: false},
	)
	return entries, nil
}

// gemma3SpecialTokenEntries resolves the bos/eos/unk/pad token ids from
// config.json and emits the tokenizer.ggml.*_token_id keys for those that
// resolve. The unknown id falls back to the vocabulary's single UNKNOWN-typed
// piece when config.json does not name one (gemma-3's config omits
// unk_token_id, but the SentencePiece vocab marks exactly one piece UNKNOWN).
// An id the checkpoint does not carry is simply omitted rather than defaulted to
// a wrong token.
func gemma3SpecialTokenEntries(root string, tokenType []int32) []basegguf.MetadataEntry {
	var ids gemma3SpecialIDs
	if read := core.ReadFile(core.PathJoin(root, "config.json")); read.OK {
		_ = core.JSONUnmarshal(read.Bytes(), &ids).OK
	}
	entries := make([]basegguf.MetadataEntry, 0, 4)
	add := func(key string, value any, ok bool) {
		if ok {
			entries = append(entries, basegguf.MetadataEntry{Key: key, ValueType: basegguf.ValueTypeUint32, Value: uint32(gemma3FirstInt(value))})
		}
	}
	add("tokenizer.ggml.bos_token_id", ids.BosTokenID, gemma3HasInt(ids.BosTokenID))
	add("tokenizer.ggml.eos_token_id", ids.EosTokenID, gemma3HasInt(ids.EosTokenID))
	add("tokenizer.ggml.padding_token_id", ids.PadTokenID, gemma3HasInt(ids.PadTokenID))
	if gemma3HasInt(ids.UnkTokenID) {
		add("tokenizer.ggml.unknown_token_id", ids.UnkTokenID, true)
	} else if unk, ok := gemma3FirstUnknownID(tokenType); ok {
		entries = append(entries, basegguf.MetadataEntry{Key: "tokenizer.ggml.unknown_token_id", ValueType: basegguf.ValueTypeUint32, Value: uint32(unk)})
	}
	return entries
}

// gemma3ReadTokenizerConfig parses tokenizer_config.json. An absent file yields
// an empty config (all-default flags, no overrides); a present-but-malformed
// file is a loud error rather than silently dropping the special-token
// reclassification.
func gemma3ReadTokenizerConfig(root string) (gemma3TokenizerConfig, error) {
	var config gemma3TokenizerConfig
	read := core.ReadFile(core.PathJoin(root, "tokenizer_config.json"))
	if !read.OK {
		return config, nil
	}
	if r := core.JSONUnmarshal(read.Bytes(), &config); !r.OK {
		return config, core.E("gemma3ReadTokenizerConfig", "parse tokenizer_config.json", r.Err())
	}
	return config, nil
}

// gemma3FirstUnknownID returns the id of the first UNKNOWN-typed piece, used as
// the unknown-token id when config.json omits one.
func gemma3FirstUnknownID(tokenType []int32) (int, bool) {
	for id, t := range tokenType {
		if t == tokenizer.SPMTokenUnknown {
			return id, true
		}
	}
	return 0, false
}

// gemma3LooksSpecial reproduces convert_hf_to_gguf.py's does_token_look_special:
// the surface forms gemma checkpoints leave un-flagged but which are control
// tokens (<pad>, <mask>, <2mass>, [@BOS@]), the <|…|> / <｜…｜> envelopes, and the
// <unused…> reserved tokens.
//
//	gemma3LooksSpecial("<unused42>") // true
//	gemma3LooksSpecial("▁the")       // false
func gemma3LooksSpecial(token string) bool {
	switch token {
	case "<pad>", "<mask>", "<2mass>", "[@BOS@]":
		return true
	}
	if core.HasPrefix(token, "<|") && core.HasSuffix(token, "|>") {
		return true
	}
	if core.HasPrefix(token, "<｜") && core.HasSuffix(token, "｜>") {
		return true
	}
	return core.HasPrefix(token, "<unused") && core.HasSuffix(token, ">")
}

// gemma3BoolOr returns the value a *bool points at, or fallback when nil.
func gemma3BoolOr(value *bool, fallback bool) bool {
	if value == nil {
		return fallback
	}
	return *value
}

// gemma3HasInt reports whether a JSON value decoded as `any` carries at least
// one integer — a scalar number, or a non-empty list whose first entry is a
// number.
func gemma3HasInt(value any) bool {
	switch v := value.(type) {
	case float64:
		return true
	case []any:
		if len(v) == 0 {
			return false
		}
		_, ok := v[0].(float64)
		return ok
	default:
		return false
	}
}

// gemma3FirstInt coerces a JSON value decoded as `any` to an int — the scalar
// itself, or a list's first entry (gemma-3's eos_token_id [1, 106] resolves to
// 1). Callers guard with gemma3HasInt first.
func gemma3FirstInt(value any) int {
	switch v := value.(type) {
	case float64:
		return int(v)
	case []any:
		if len(v) > 0 {
			if n, ok := v[0].(float64); ok {
				return int(n)
			}
		}
	}
	return 0
}
