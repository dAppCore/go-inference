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
	builder := core.NewBuilder()
	// Pre-size the builder to avoid reallocation as the result grows;
	// most tokens fall back to Text first so use it for the estimate.
	total := 0
	for i := range tokens {
		total += len(tokens[i].Text)
		if tokens[i].Text == "" {
			total += len(tokens[i].Value)
		}
	}
	builder.Grow(total)
	for i := range tokens {
		builder.WriteString(tokenSurface(tokens[i]))
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
	out := make([]Token, 0, limit)
	var accepted, rejected int
	for i := 0; i < limit; i++ {
		targetToken := target[i]
		if i < len(candidates) {
			if TokenEqual(candidates[i], targetToken) {
				out = append(out, cloneToken(candidates[i]))
				accepted++
				continue
			}
			rejected++
		}
		out = append(out, cloneToken(targetToken))
	}
	attempted := accepted + rejected
	metrics := Metrics{
		AcceptedTokens: accepted,
		RejectedTokens: rejected,
		EmittedTokens:  len(out),
	}
	if attempted > 0 {
		metrics.AcceptanceRate = float64(accepted) / float64(attempted)
	}
	return Result{
		Mode:    mode,
		Prompt:  prompt,
		Text:    TokensText(out),
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

func cloneToken(token Token) Token {
	return Token{ID: token.ID, Value: token.Value, Text: token.Text}
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
