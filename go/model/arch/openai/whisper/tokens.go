// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import (
	"math"

	core "dappco.re/go"
)

// tokens.go is the task-token machinery: language auto-detection (one decoder step from
// <|startoftranscript|>, argmax restricted to the language-token ids — the reference method, verified
// byte-for-byte against transformers' WhisperGenerationMixin.detect_language before this file was
// written), the initial prompt builder, and the greedy decode loop with the two suppression lists
// generate() applies even in plain greedy decoding (SuppressTokensLogitsProcessor,
// SuppressTokensAtBeginLogitsProcessor). All three were validated against a hand-rolled Python
// replica of this exact algorithm (no HF generate() internals, same primitives this file uses) —
// it reproduced transformers' official generate() output byte-for-byte on the whisper-tiny checkpoint.

// DetectLanguage runs one decoder step from just <|startoftranscript|>, restricts the argmax to the
// checkpoint's known language-token ids (generation_config.json's lang_to_id), and returns the winning
// token id plus its bare code ("en"). This is a standalone, one-off decoder pass — its own fresh
// SelfAttnCache, discarded once DetectLanguage returns — never shared with GreedyDecode's own cache: the
// hypothetical first content token this step evaluates is not part of the real generation (matching the
// reference: language detection and the real decode are separate forward passes).
func DetectLanguage(crossKV []CrossKV, tenc int, w *Weights, cfg *Config, gen *GenerationConfig) (int32, string, error) {
	if gen == nil || len(gen.LangToID) == 0 {
		return 0, "", core.NewError("whisper.DetectLanguage: generation config carries no language tokens")
	}
	cache := NewSelfAttnCache(len(w.DecoderLayers))
	logits, err := DecodeLogitsStep([]int32{gen.DecoderStartTokenID}, 0, cache, crossKV, tenc, w, cfg)
	if err != nil {
		return 0, "", err
	}
	best := int32(-1)
	bestVal := float32(math.Inf(-1))
	for _, id := range gen.LangToID {
		if int(id) < 0 || int(id) >= len(logits) {
			continue
		}
		if best == -1 || logits[id] > bestVal {
			bestVal, best = logits[id], id
		}
	}
	if best == -1 {
		return 0, "", core.NewError("whisper.DetectLanguage: no language token id fell within the vocabulary")
	}
	return best, gen.LanguageCode(best), nil
}

// BuildInitTokens composes the initial decoder prompt: <|startoftranscript|>, the language token
// (forced via language when non-empty — a bare code like "en" or the bracketed "<|en|>" form both
// resolve; "" auto-detects via DetectLanguage), <|transcribe|> (the only task v1 exposes — translate is
// a documented non-goal), <|notimestamps|> (v1 is plain-text only, per the design's scope). Returns the
// prompt tokens plus the resolved language code for the caller to report.
func BuildInitTokens(crossKV []CrossKV, tenc int, w *Weights, cfg *Config, gen *GenerationConfig, language string) ([]int32, string, error) {
	if gen == nil {
		return nil, "", core.NewError("whisper.BuildInitTokens: nil generation config")
	}
	var langID int32
	var langCode string
	if language != "" {
		id, ok := gen.LanguageTokenID(language)
		if !ok {
			return nil, "", core.NewError("whisper.BuildInitTokens: unknown --language " + language + " (not in this checkpoint's generation_config.json lang_to_id)")
		}
		langID, langCode = id, language
	} else {
		id, code, err := DetectLanguage(crossKV, tenc, w, cfg, gen)
		if err != nil {
			return nil, "", err
		}
		langID, langCode = id, code
	}
	transcribeID, ok := gen.TaskToID["transcribe"]
	if !ok {
		return nil, "", core.NewError("whisper.BuildInitTokens: generation config's task_to_id has no \"transcribe\" entry")
	}
	return []int32{gen.DecoderStartTokenID, langID, transcribeID, gen.NoTimestampsTokenID}, langCode, nil
}

// maxDecodeLength picks the greedy loop's hard stop: the min of the checkpoint's generation policy
// (generation_config.json's max_length) and its architecture bound (config.json's max_target_positions)
// — see GenerationConfig.MaxLength's doc comment.
func maxDecodeLength(cfg *Config, gen *GenerationConfig) int {
	limit := cfg.MaxTargetPositions
	if gen.MaxLength > 0 && int(gen.MaxLength) < limit {
		limit = int(gen.MaxLength)
	}
	return limit
}

// GreedyDecode runs the autoregressive greedy loop from initTokens (BuildInitTokens' output), applying
// the checkpoint's suppress_tokens (every step) and begin_suppress_tokens (only the very first
// generated-content position, input_ids.shape[-1]==len(initTokens) in the reference) exactly as
// generate() does, stopping at EOS or the resolved length bound. Returns the CONTENT token ids only —
// the prompt is not echoed back (mirroring how tokenizer.Decode is about to strip every special token
// anyway; a caller wanting the raw sequence prepends initTokens itself).
//
// KV CACHE: logits are sourced from DecodeLogitsStep, not DecodeLogits — one fresh SelfAttnCache for the
// whole call, fed the entire init prompt as ONE prefill batch on the loop's first iteration (populating
// the cache for positions [0,beginIndex)) and exactly one new token per iteration after. The surrounding
// loop shape (the `for len(ids) < limit` guard, the suppress/beginSuppress application, the EOS-before-
// append check) is UNCHANGED from the original whole-sequence-recompute version — only WHERE logits comes
// from differs — so this produces bit-identical output (see decoder.go's file doc comment for why, and
// decoder_test.go's parity test for the proof).
func GreedyDecode(crossKV []CrossKV, tenc int, w *Weights, cfg *Config, gen *GenerationConfig, initTokens []int32) ([]int32, error) {
	if len(initTokens) == 0 {
		return nil, core.NewError("whisper.GreedyDecode: empty init prompt")
	}
	ids := append([]int32(nil), initTokens...)
	beginIndex := len(initTokens)
	suppress := make(map[int32]bool, len(gen.SuppressTokens))
	for _, t := range gen.SuppressTokens {
		suppress[t] = true
	}
	beginSuppress := make(map[int32]bool, len(gen.BeginSuppressTokens))
	for _, t := range gen.BeginSuppressTokens {
		beginSuppress[t] = true
	}

	limit := maxDecodeLength(cfg, gen)
	cache := NewSelfAttnCache(len(w.DecoderLayers))
	prefilled := false
	var content []int32
	for len(ids) < limit {
		var logits []float32
		var err error
		if !prefilled {
			logits, err = DecodeLogitsStep(initTokens, 0, cache, crossKV, tenc, w, cfg)
			prefilled = true
		} else {
			logits, err = DecodeLogitsStep(ids[len(ids)-1:], len(ids)-1, cache, crossKV, tenc, w, cfg)
		}
		if err != nil {
			return nil, err
		}
		for id := range suppress {
			if int(id) >= 0 && int(id) < len(logits) {
				logits[id] = float32(math.Inf(-1))
			}
		}
		if len(ids) == beginIndex {
			for id := range beginSuppress {
				if int(id) >= 0 && int(id) < len(logits) {
					logits[id] = float32(math.Inf(-1))
				}
			}
		}
		next := argmaxF32(logits)
		if next == gen.EOSTokenID {
			break
		}
		ids = append(ids, next)
		content = append(content, next)
	}
	return content, nil
}

// argmaxF32 returns the index of the largest value (ties keep the first, matching torch.argmax).
func argmaxF32(v []float32) int32 {
	best := int32(0)
	bestVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > bestVal {
			bestVal = v[i]
			best = int32(i)
		}
	}
	return best
}
