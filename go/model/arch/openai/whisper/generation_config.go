// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import core "dappco.re/go"

// generation_config.go reads generation_config.json's task-token machinery straight off the checkpoint
// (never guessed): decoder_start_token_id ("<|startoftranscript|>"), the language-code→token-id table
// (lang_to_id — one entry per supported language, e.g. "<|en|>": 50259), the task-code→token-id table
// (task_to_id — "transcribe"/"translate"), no_timestamps_token_id, eos_token_id, and the two suppression
// lists generate() applies even in plain greedy decoding: suppress_tokens (always masked to -∞) and
// begin_suppress_tokens (masked only at the FIRST generated-content position — see tokens.go).

// GenerationConfig is the task-token subset of a Whisper checkpoint's generation_config.json.
type GenerationConfig struct {
	DecoderStartTokenID int32
	EOSTokenID          int32
	NoTimestampsTokenID int32
	// MaxLength is generate()'s length cap (448 for every published Whisper checkpoint) — GreedyDecode
	// takes the min of this and Config.MaxTargetPositions as its hard stop, so a checkpoint whose
	// generation policy caps shorter than its position-table size (never seen in practice) still can't
	// overrun DecodeLogits' bound check. 0 ⇒ absent, callers fall back to MaxTargetPositions alone.
	MaxLength int32
	// LangToID maps a language token string ("<|en|>") to its vocabulary id — every entry
	// generation_config.json's lang_to_id carries.
	LangToID map[string]int32
	// TaskToID maps a task name ("transcribe", "translate") to its vocabulary id.
	TaskToID map[string]int32
	// SuppressTokens are masked to -∞ at every decode step (non-speech/formatting artefacts the
	// checkpoint was tuned to never emit).
	SuppressTokens []int32
	// BeginSuppressTokens are masked to -∞ ONLY when about to generate the first token after the
	// initial task-token prompt (prevents an immediate blank/EOS transcript).
	BeginSuppressTokens []int32
}

// generationConfigJSON mirrors the subset of generation_config.json this package reads.
type generationConfigJSON struct {
	DecoderStartTokenID int32            `json:"decoder_start_token_id"`
	EOSTokenID          int32            `json:"eos_token_id"`
	NoTimestampsTokenID int32            `json:"no_timestamps_token_id"`
	MaxLength           int32            `json:"max_length"`
	LangToID            map[string]int32 `json:"lang_to_id"`
	TaskToID            map[string]int32 `json:"task_to_id"`
	SuppressTokens      []int32          `json:"suppress_tokens"`
	BeginSuppressTokens []int32          `json:"begin_suppress_tokens"`
}

// LoadGenerationConfig reads generation_config.json from a Whisper checkpoint directory.
func LoadGenerationConfig(dir string) (*GenerationConfig, error) {
	path := core.PathJoin(dir, "generation_config.json")
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("whisper.LoadGenerationConfig", "read "+path, resultErr(read))
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return nil, core.NewError("whisper.LoadGenerationConfig: " + path + " read returned non-byte data")
	}
	var g generationConfigJSON
	if r := core.JSONUnmarshal(data, &g); !r.OK {
		return nil, core.NewError("whisper.LoadGenerationConfig: parse " + path)
	}
	if g.DecoderStartTokenID == 0 || len(g.LangToID) == 0 || len(g.TaskToID) == 0 {
		return nil, core.NewError("whisper.LoadGenerationConfig: " + path + " is missing decoder_start_token_id/lang_to_id/task_to_id")
	}
	if _, ok := g.TaskToID["transcribe"]; !ok {
		return nil, core.NewError("whisper.LoadGenerationConfig: " + path + " task_to_id has no \"transcribe\" entry")
	}
	return &GenerationConfig{
		DecoderStartTokenID: g.DecoderStartTokenID,
		EOSTokenID:          g.EOSTokenID,
		NoTimestampsTokenID: g.NoTimestampsTokenID,
		MaxLength:           g.MaxLength,
		LangToID:            g.LangToID,
		TaskToID:            g.TaskToID,
		SuppressTokens:      g.SuppressTokens,
		BeginSuppressTokens: g.BeginSuppressTokens,
	}, nil
}

// LanguageTokenID resolves a short language code ("en") or a bracketed token ("<|en|>") to its
// vocabulary id. Matches the reference's language_to_id lookup for the common case (a bare ISO code);
// TO_LANGUAGE_CODE's full alias table (e.g. "english"→"en") is NOT ported — out of scope for v1's
// --language override, which documents the bracketed/bare-code forms.
func (g *GenerationConfig) LanguageTokenID(code string) (int32, bool) {
	if id, ok := g.LangToID[code]; ok {
		return id, ok
	}
	id, ok := g.LangToID["<|"+code+"|>"]
	return id, ok
}

// LanguageCode returns the bare code ("en") for a language token id, or "" if id is not a known
// language token. Linear over LangToID (≈99 entries) — called once per transcription, not hot.
func (g *GenerationConfig) LanguageCode(id int32) string {
	for token, tid := range g.LangToID {
		if tid == id {
			return fromBracketedToken(token)
		}
	}
	return ""
}

// fromBracketedToken strips a "<|xx|>" wrapper down to "xx"; anything else passes through unchanged.
func fromBracketedToken(token string) string {
	if len(token) > 4 && token[:2] == "<|" && token[len(token)-2:] == "|>" {
		return token[2 : len(token)-2]
	}
	return token
}
