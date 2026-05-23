// SPDX-Licence-Identifier: EUPL-1.2

// Package decode is the driver-neutral decode-optimisation harness used
// by speculative and prompt-lookup decode benchmarks.
//
// The acceptance algorithm is a generic accept/reject over token streams;
// generation is delegated to caller-supplied GenerateFunc callbacks. The
// package is shared by every backend driver (go-mlx, go-cuda, go-rocm)
// that wants a portable speculative or prompt-lookup decode report.
//
//	result, err := decode.Speculative(ctx, decode.SpeculativeConfig{
//	    Prompt: "Write a haiku.",
//	    MaxTokens: 64,
//	    TargetGenerate: target,
//	    DraftGenerate:  draft,
//	})
package decode

import (
	"context"
	"time"

	core "dappco.re/go"
)

// Token is one element of a generation sequence — ID plus an optional
// surface form. Drivers populate the fields their tokenizer can report.
type Token struct {
	ID    int32  `json:"id,omitempty"`
	Value string `json:"value,omitempty"`
	Text  string `json:"text,omitempty"`
}

// GenerateConfig is the per-call generation request passed to the
// caller-supplied GenerateFunc. Only MaxTokens is consumed by decode;
// drivers may carry extra context inside the closure.
type GenerateConfig struct {
	MaxTokens int `json:"max_tokens"`
}

// Generation is the result the GenerateFunc returns to decode.
type Generation struct {
	Tokens []Token `json:"tokens,omitempty"`
	Text   string  `json:"text,omitempty"`
}

// GenerateFunc is the model-side generation hook. decode supplies the
// prompt + per-call config; the driver decides how to evaluate it.
type GenerateFunc func(context.Context, string, GenerateConfig) (Generation, error)

// SpeculativeConfig configures the speculative-decode reference path.
// Target + draft generators must both be supplied; decode compares their
// outputs token-by-token to produce an acceptance report.
type SpeculativeConfig struct {
	Prompt         string         `json:"prompt,omitempty"`
	MaxTokens      int            `json:"max_tokens,omitempty"`
	DraftTokens    int            `json:"draft_tokens,omitempty"`
	GenerateConfig GenerateConfig `json:"generate_config,omitempty"`
	TargetGenerate GenerateFunc   `json:"-"`
	DraftGenerate  GenerateFunc   `json:"-"`
}

// PromptLookupConfig configures prompt-lookup decoding over a caller-
// supplied token sequence (typically derived from repeated context in
// the prompt).
type PromptLookupConfig struct {
	Prompt         string         `json:"prompt,omitempty"`
	MaxTokens      int            `json:"max_tokens,omitempty"`
	GenerateConfig GenerateConfig `json:"generate_config,omitempty"`
	TargetGenerate GenerateFunc   `json:"-"`
	LookupTokens   []Token        `json:"lookup_tokens,omitempty"`
}

// Result is the common decode-optimisation report.
type Result struct {
	Mode    string  `json:"mode"`
	Prompt  string  `json:"prompt,omitempty"`
	Text    string  `json:"text,omitempty"`
	Tokens  []Token `json:"tokens,omitempty"`
	Metrics Metrics `json:"metrics"`
}

// Metrics records candidate acceptance and call-level timing.
type Metrics struct {
	TargetTokens   int           `json:"target_tokens,omitempty"`
	DraftTokens    int           `json:"draft_tokens,omitempty"`
	LookupTokens   int           `json:"lookup_tokens,omitempty"`
	AcceptedTokens int           `json:"accepted_tokens,omitempty"`
	RejectedTokens int           `json:"rejected_tokens,omitempty"`
	EmittedTokens  int           `json:"emitted_tokens,omitempty"`
	AcceptanceRate float64       `json:"acceptance_rate,omitempty"`
	TargetCalls    int           `json:"target_calls,omitempty"`
	DraftCalls     int           `json:"draft_calls,omitempty"`
	Duration       time.Duration `json:"duration,omitempty"`
	TargetDuration time.Duration `json:"target_duration,omitempty"`
	DraftDuration  time.Duration `json:"draft_duration,omitempty"`
}

// Mode constants identify which decode-optimisation produced a Result.
const (
	ModeSpeculative  = "speculative"
	ModePromptLookup = "prompt_lookup"
)

// DefaultMaxTokens is the fallback when neither the caller nor the
// embedded GenerateConfig supplies a positive max.
const DefaultMaxTokens = 256

// Speculative compares draft-model candidates against target-model
// tokens and reports deterministic acceptance metrics. This is the safe
// reference API; it does not claim a speedup until a backend provides
// native verification that the benchmark can measure.
//
//	result, err := decode.Speculative(ctx, cfg)
func Speculative(ctx context.Context, cfg SpeculativeConfig) (Result, error) {
	if cfg.TargetGenerate == nil {
		return Result{}, core.NewError("decode: speculative decode requires target generator")
	}
	if cfg.DraftGenerate == nil {
		return Result{}, core.NewError("decode: speculative decode requires draft generator")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	maxTokens := normaliseMaxTokens(cfg.MaxTokens, cfg.GenerateConfig.MaxTokens)
	targetCfg := cfg.GenerateConfig
	targetCfg.MaxTokens = maxTokens
	draftCfg := cfg.GenerateConfig
	draftCfg.MaxTokens = cfg.DraftTokens
	if draftCfg.MaxTokens <= 0 || draftCfg.MaxTokens > maxTokens {
		draftCfg.MaxTokens = maxTokens
	}

	start := time.Now()
	draftStart := time.Now()
	draft, err := cfg.DraftGenerate(ctx, cfg.Prompt, draftCfg)
	draftDuration := nonZeroDuration(time.Since(draftStart))
	if err != nil {
		return Result{}, err
	}
	targetStart := time.Now()
	target, err := cfg.TargetGenerate(ctx, cfg.Prompt, targetCfg)
	targetDuration := nonZeroDuration(time.Since(targetStart))
	if err != nil {
		return Result{}, err
	}
	result := buildAcceptanceResult(ModeSpeculative, cfg.Prompt, target.Tokens, draft.Tokens, maxTokens)
	result.Metrics.TargetTokens = len(target.Tokens)
	result.Metrics.DraftTokens = len(draft.Tokens)
	result.Metrics.TargetCalls = 1
	result.Metrics.DraftCalls = 1
	result.Metrics.Duration = nonZeroDuration(time.Since(start))
	result.Metrics.TargetDuration = targetDuration
	result.Metrics.DraftDuration = draftDuration
	return result, nil
}

// PromptLookup compares prompt-derived lookup candidates against the
// target stream and reports how often repeated-context tokens were
// reusable.
//
//	result, err := decode.PromptLookup(ctx, cfg)
func PromptLookup(ctx context.Context, cfg PromptLookupConfig) (Result, error) {
	if cfg.TargetGenerate == nil {
		return Result{}, core.NewError("decode: prompt lookup decode requires target generator")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	maxTokens := normaliseMaxTokens(cfg.MaxTokens, cfg.GenerateConfig.MaxTokens)
	targetCfg := cfg.GenerateConfig
	targetCfg.MaxTokens = maxTokens
	start := time.Now()
	targetStart := time.Now()
	target, err := cfg.TargetGenerate(ctx, cfg.Prompt, targetCfg)
	targetDuration := nonZeroDuration(time.Since(targetStart))
	if err != nil {
		return Result{}, err
	}
	result := buildAcceptanceResult(ModePromptLookup, cfg.Prompt, target.Tokens, cfg.LookupTokens, maxTokens)
	result.Metrics.TargetTokens = len(target.Tokens)
	result.Metrics.LookupTokens = len(cfg.LookupTokens)
	result.Metrics.TargetCalls = 1
	result.Metrics.Duration = nonZeroDuration(time.Since(start))
	result.Metrics.TargetDuration = targetDuration
	return result, nil
}

// TokensText renders a token slice as a concatenated string, preferring
// each token's Text field then falling back to Value. Exported so
// drivers that need the same rendering for non-decode paths can reuse it.
//
//	text := decode.TokensText(result.Tokens)
func TokensText(tokens []Token) string {
	// Pre-grow the builder using each token's actual length. Strings
	// are immutable so reading len() is free; this saves the cascade
	// of doubling allocs the builder would otherwise pay as it grows
	// from 0 → final size. For 2048-token decodes that's ~10 allocs
	// down to 1. Index iteration avoids the per-iter 40-byte Token
	// copy a range-value loop emits.
	total := 0
	for i := range tokens {
		text := tokens[i].Text
		if text == "" {
			text = tokens[i].Value
		}
		total += len(text)
	}
	return tokensTextSized(tokens, total)
}

// tokensTextSized is TokensText with the total length pre-computed by
// the caller. buildAcceptanceResult walks the token stream once during
// the acceptance pass and already knows the rendered length when it
// gets here, so the second len-summing walk is redundant. Exported
// (lowercase) only so the inner loop can elide that walk; external
// callers go through TokensText, which computes total itself.
func tokensTextSized(tokens []Token, total int) string {
	builder := core.NewBuilder()
	builder.Grow(total)
	// Index iteration avoids the per-iter 40-byte Token copy that a
	// range-value loop emits; we only read two string headers from
	// the slice slot, never the int32 ID.
	for i := range tokens {
		text := tokens[i].Text
		if text == "" {
			text = tokens[i].Value
		}
		builder.WriteString(text)
	}
	return builder.String()
}

// CloneTokens returns an independent copy of a token slice.
//
//	out := decode.CloneTokens(in)
func CloneTokens(tokens []Token) []Token {
	out := make([]Token, len(tokens))
	copy(out, tokens)
	return out
}

// TokenEqual reports whether two tokens identify the same surface form.
// IDs must match; if both surface strings are non-empty they must also
// match.
//
//	if decode.TokenEqual(a, b) { … }
func TokenEqual(a, b Token) bool {
	if a.ID != b.ID {
		return false
	}
	aText := tokenSurface(a)
	bText := tokenSurface(b)
	if aText == "" || bText == "" {
		return true
	}
	return aText == bText
}

func buildAcceptanceResult(mode, prompt string, target, candidates []Token, maxTokens int) Result {
	limit := len(target)
	if maxTokens > 0 && maxTokens < limit {
		limit = maxTokens
	}
	// Pre-size + direct index assignment beats append on a known-N
	// loop: the append cap-check + len-bump on every iteration is dead
	// weight when we know we write exactly `limit` tokens. Saves the
	// per-token slice-header bookkeeping over a 2048-token pass.
	out := make([]Token, limit)
	// Track the rendered text length alongside the build loop so the
	// TokensText pre-grow walk fuses with the acceptance pass — the
	// previous shape walked the emitted tokens twice (once to build
	// out, once inside TokensText to sum lengths). At 2048 tokens that
	// halves the walk count over the slice.
	totalText := 0
	var accepted, rejected int
	candidateLen := len(candidates)
	for i := 0; i < limit; i++ {
		// Write the emitted token directly into out[i] from whichever
		// source slice owns it — avoids the intermediate `emitted`
		// stack variable plus the speculative pre-load of
		// `targetToken := target[i]`. Per token this saves two 40-byte
		// struct copies (Token is 40 bytes on arm64 / amd64).
		if i < candidateLen && TokenEqual(candidates[i], target[i]) {
			out[i] = candidates[i]
			accepted++
			text := candidates[i].Text
			if text == "" {
				text = candidates[i].Value
			}
			totalText += len(text)
		} else {
			out[i] = target[i]
			if i < candidateLen {
				rejected++
			}
			text := target[i].Text
			if text == "" {
				text = target[i].Value
			}
			totalText += len(text)
		}
	}
	attempted := accepted + rejected
	metrics := Metrics{
		AcceptedTokens: accepted,
		RejectedTokens: rejected,
		EmittedTokens:  limit,
	}
	if attempted > 0 {
		metrics.AcceptanceRate = float64(accepted) / float64(attempted)
	}
	return Result{
		Mode:    mode,
		Prompt:  prompt,
		Text:    tokensTextSized(out, totalText),
		Tokens:  out,
		Metrics: metrics,
	}
}

func normaliseMaxTokens(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return DefaultMaxTokens
}

// tokenSurface returns the token's surface form, preferring Text over
// Value. Inlined two-arg path used by every accept/reject decision; the
// previous variadic firstNonEmpty allocated a []string per call.
func tokenSurface(t Token) string {
	if hasNonSpace(t.Text) {
		return t.Text
	}
	if hasNonSpace(t.Value) {
		return t.Value
	}
	return ""
}

// hasNonSpace reports whether s contains any non-whitespace byte. Avoids
// strings.TrimSpace's per-call string allocation when the input contains
// leading or trailing whitespace. Falls back to core.Trim on multi-byte
// input to preserve Unicode whitespace semantics.
func hasNonSpace(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 0x80 {
			// Multi-byte rune may include Unicode whitespace
			// (NBSP, ideographic space, etc.); defer to core.Trim.
			return core.Trim(s) != ""
		}
		switch c {
		case ' ', '\t', '\n', '\v', '\f', '\r':
			continue
		default:
			return true
		}
	}
	return false
}

func nonZeroDuration(d time.Duration) time.Duration {
	if d <= 0 {
		return time.Nanosecond
	}
	return d
}
