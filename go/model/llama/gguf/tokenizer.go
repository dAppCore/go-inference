// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	core "dappco.re/go"
	basegguf "dappco.re/go/inference/model/gguf"
)

type llamaTokenizerJSON struct {
	Model struct {
		Vocab  map[string]int `json:"vocab"`
		Merges [][]string     `json:"merges"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int  `json:"id"`
		Special bool `json:"special"`
	} `json:"added_tokens"`
}

func llamaTokenizer(tokenizerJSON []byte, config llamaConfig) ([]basegguf.MetadataEntry, error) {
	var tokenizer llamaTokenizerJSON
	if result := core.JSONUnmarshal(tokenizerJSON, &tokenizer); !result.OK {
		return nil, core.E("llamaTokenizer", "parse tokenizer.json", result.Err())
	}
	if len(tokenizer.Model.Vocab) == 0 {
		return nil, core.NewError("gguf: llama tokenizer.json has an empty vocab")
	}
	tokens := make([]string, len(tokenizer.Model.Vocab))
	seen := make([]bool, len(tokens))
	for token, id := range tokenizer.Model.Vocab {
		if id < 0 || id >= len(tokens) || seen[id] {
			return nil, core.Errorf("gguf: llama tokenizer vocab has invalid id %d", id)
		}
		tokens[id], seen[id] = token, true
	}
	for id, ok := range seen {
		if !ok {
			return nil, core.Errorf("gguf: llama tokenizer vocab has no token for id %d", id)
		}
	}
	types := make([]int32, len(tokens))
	scores := make([]float32, len(tokens))
	for i := range types {
		types[i] = 1
	}
	for _, added := range tokenizer.AddedTokens {
		if added.ID >= 0 && added.ID < len(types) && added.Special {
			types[added.ID] = 3
		}
	}
	merges := make([]string, len(tokenizer.Model.Merges))
	for i, merge := range tokenizer.Model.Merges {
		if len(merge) != 2 {
			return nil, core.Errorf("gguf: llama tokenizer merge %d is not a pair", i)
		}
		merges[i] = core.Concat(merge[0], " ", merge[1])
	}
	u32 := func(key string, value int) basegguf.MetadataEntry {
		return basegguf.MetadataEntry{Key: key, ValueType: basegguf.ValueTypeUint32, Value: uint32(value)}
	}
	return []basegguf.MetadataEntry{
		{Key: "tokenizer.ggml.model", ValueType: basegguf.ValueTypeString, Value: "gpt2"},
		{Key: "tokenizer.ggml.pre", ValueType: basegguf.ValueTypeString, Value: "llama-bpe"},
		{Key: "tokenizer.ggml.tokens", ValueType: basegguf.ValueTypeArray, Value: tokens},
		{Key: "tokenizer.ggml.scores", ValueType: basegguf.ValueTypeArray, Value: scores},
		{Key: "tokenizer.ggml.token_type", ValueType: basegguf.ValueTypeArray, Value: types},
		{Key: "tokenizer.ggml.merges", ValueType: basegguf.ValueTypeArray, Value: merges},
		u32("tokenizer.ggml.bos_token_id", config.BOSTokenID),
		u32("tokenizer.ggml.eos_token_id", config.EOSTokenID),
		{Key: "tokenizer.ggml.add_bos_token", ValueType: basegguf.ValueTypeBool, Value: true},
	}, nil
}
