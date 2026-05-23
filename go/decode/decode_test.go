// SPDX-Licence-Identifier: EUPL-1.2

package decode

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestSpeculative_AcceptsAndRejectsDraftTokens_Good(t *testing.T) {
	targetCalls := 0
	draftCalls := 0
	target := GeneratorFunc(func(context.Context, string, GenerateConfig) (Generation, error) {
		targetCalls++
		return Generation{Tokens: []Token{{ID: 1, Text: "A"}, {ID: 2, Text: "B"}, {ID: 4, Text: "D"}}}, nil
	})
	draft := GeneratorFunc(func(context.Context, string, GenerateConfig) (Generation, error) {
		draftCalls++
		return Generation{Tokens: []Token{{ID: 1, Text: "A"}, {ID: 2, Text: "B"}, {ID: 3, Text: "C"}}}, nil
	})

	result, err := Speculative(context.Background(), SpeculativeConfig{
		Prompt:         "p",
		MaxTokens:      3,
		DraftTokens:    3,
		TargetGenerate: target,
		DraftGenerate:  draft,
	})
	if err != nil {
		t.Fatalf("Speculative() error = %v", err)
	}
	if result.Mode != ModeSpeculative {
		t.Fatalf("Mode = %q, want %q", result.Mode, ModeSpeculative)
	}
	if result.Text != "ABD" {
		t.Fatalf("Text = %q, want ABD", result.Text)
	}
	if result.Metrics.AcceptedTokens != 2 || result.Metrics.RejectedTokens != 1 || result.Metrics.AcceptanceRate != 2.0/3.0 {
		t.Fatalf("metrics = %+v, want two accepted + one rejected", result.Metrics)
	}
	if result.Metrics.TargetCalls != 1 || result.Metrics.DraftCalls != 1 || targetCalls != 1 || draftCalls != 1 {
		t.Fatalf("calls = metrics:%+v target:%d draft:%d, want one each", result.Metrics, targetCalls, draftCalls)
	}
	if result.Metrics.Duration <= 0 || result.Metrics.TargetDuration <= 0 || result.Metrics.DraftDuration <= 0 {
		t.Fatalf("durations not populated: %+v", result.Metrics)
	}
}

func TestPromptLookup_AcceptsRepeatedContextTokens_Good(t *testing.T) {
	target := GeneratorFunc(func(context.Context, string, GenerateConfig) (Generation, error) {
		return Generation{Tokens: []Token{{ID: 10, Text: "go"}, {ID: 11, Text: "-"}, {ID: 12, Text: "mlx"}}}, nil
	})

	result, err := PromptLookup(context.Background(), PromptLookupConfig{
		Prompt:         "go-mlx go-mlx",
		MaxTokens:      3,
		TargetGenerate: target,
		LookupTokens:   []Token{{ID: 10, Text: "go"}, {ID: 99, Text: "?"}, {ID: 12, Text: "mlx"}},
	})
	if err != nil {
		t.Fatalf("PromptLookup() error = %v", err)
	}
	if result.Mode != ModePromptLookup {
		t.Fatalf("Mode = %q, want %q", result.Mode, ModePromptLookup)
	}
	if result.Text != "go-mlx" {
		t.Fatalf("Text = %q, want go-mlx", result.Text)
	}
	if result.Metrics.AcceptedTokens != 2 || result.Metrics.RejectedTokens != 1 || result.Metrics.LookupTokens != 3 {
		t.Fatalf("metrics = %+v, want two accepts + one rejection + 3 lookup tokens", result.Metrics)
	}
	if result.Metrics.TargetCalls != 1 || result.Metrics.DraftCalls != 0 {
		t.Fatalf("calls = %+v, want target=1 draft=0", result.Metrics)
	}
}

func TestSpeculative_RequiresTargetAndDraft_Bad(t *testing.T) {
	if _, err := Speculative(context.Background(), SpeculativeConfig{}); err == nil {
		t.Fatal("Speculative(zero) error = nil, want missing-target")
	}
	dummy := GeneratorFunc(func(context.Context, string, GenerateConfig) (Generation, error) { return Generation{}, nil })
	if _, err := Speculative(context.Background(), SpeculativeConfig{TargetGenerate: dummy}); err == nil {
		t.Fatal("Speculative(target-only) error = nil, want missing-draft")
	}
}

func TestPromptLookup_RequiresTarget_Bad(t *testing.T) {
	if _, err := PromptLookup(context.Background(), PromptLookupConfig{}); err == nil {
		t.Fatal("PromptLookup(zero) error = nil, want missing-target")
	}
}

func TestSpeculative_PropagatesDraftError_Bad(t *testing.T) {
	want := errors.New("draft boom")
	target := GeneratorFunc(func(context.Context, string, GenerateConfig) (Generation, error) {
		return Generation{Tokens: []Token{{ID: 1}}}, nil
	})
	draft := GeneratorFunc(func(context.Context, string, GenerateConfig) (Generation, error) { return Generation{}, want })
	if _, err := Speculative(context.Background(), SpeculativeConfig{
		Prompt: "p", MaxTokens: 4, TargetGenerate: target, DraftGenerate: draft,
	}); err == nil {
		t.Fatal("Speculative() did not propagate draft error")
	}
}

func TestSpeculative_PropagatesTargetError_Bad(t *testing.T) {
	want := errors.New("target boom")
	target := GeneratorFunc(func(context.Context, string, GenerateConfig) (Generation, error) { return Generation{}, want })
	draft := GeneratorFunc(func(context.Context, string, GenerateConfig) (Generation, error) {
		return Generation{Tokens: []Token{{ID: 1}}}, nil
	})
	if _, err := Speculative(context.Background(), SpeculativeConfig{
		Prompt: "p", MaxTokens: 4, TargetGenerate: target, DraftGenerate: draft,
	}); err == nil {
		t.Fatal("Speculative() did not propagate target error")
	}
}

func TestPromptLookup_PropagatesTargetError_Bad(t *testing.T) {
	want := errors.New("target boom")
	target := GeneratorFunc(func(context.Context, string, GenerateConfig) (Generation, error) { return Generation{}, want })
	if _, err := PromptLookup(context.Background(), PromptLookupConfig{
		Prompt: "p", MaxTokens: 4, TargetGenerate: target,
	}); err == nil {
		t.Fatal("PromptLookup() did not propagate target error")
	}
}

func TestSpeculative_NilContextDefaultsToBackground_Good(t *testing.T) {
	target := GeneratorFunc(func(context.Context, string, GenerateConfig) (Generation, error) {
		return Generation{Tokens: []Token{{ID: 1, Text: "x"}}}, nil
	})
	draft := target
	if _, err := Speculative(nil, SpeculativeConfig{
		Prompt: "p", MaxTokens: 1, TargetGenerate: target, DraftGenerate: draft,
	}); err != nil {
		t.Fatalf("Speculative(nil ctx) error = %v", err)
	}
}

func TestPromptLookup_NilContextDefaultsToBackground_Good(t *testing.T) {
	target := GeneratorFunc(func(context.Context, string, GenerateConfig) (Generation, error) {
		return Generation{Tokens: []Token{{ID: 1, Text: "x"}}}, nil
	})
	if _, err := PromptLookup(nil, PromptLookupConfig{
		Prompt: "p", MaxTokens: 1, TargetGenerate: target,
	}); err != nil {
		t.Fatalf("PromptLookup(nil ctx) error = %v", err)
	}
}

func TestTokenEqual_GoodBad(t *testing.T) {
	if !TokenEqual(Token{ID: 1, Text: "a"}, Token{ID: 1, Text: "a"}) {
		t.Fatal("identical tokens reported unequal")
	}
	if TokenEqual(Token{ID: 1, Text: "a"}, Token{ID: 2, Text: "a"}) {
		t.Fatal("different IDs reported equal")
	}
	if TokenEqual(Token{ID: 1, Text: "a"}, Token{ID: 1, Text: "b"}) {
		t.Fatal("different non-empty texts reported equal")
	}
	if !TokenEqual(Token{ID: 1}, Token{ID: 1, Text: "a"}) {
		t.Fatal("empty-text token did not skip text comparison")
	}
	if !TokenEqual(Token{ID: 1, Value: "x"}, Token{ID: 1, Value: "x"}) {
		t.Fatal("Value-only equality not honoured")
	}
}

func TestTokensText_PrefersTextOverValue_Good(t *testing.T) {
	got := TokensText([]Token{{Text: "go"}, {Value: "-"}, {Text: "mlx", Value: "ignored"}})
	if got != "go-mlx" {
		t.Fatalf("TokensText = %q, want go-mlx", got)
	}
}

func TestCloneTokens_IndependentCopy_Good(t *testing.T) {
	src := []Token{{ID: 1, Text: "a"}, {ID: 2, Text: "b"}}
	dst := CloneTokens(src)
	src[0].ID = 99
	if dst[0].ID == 99 {
		t.Fatal("CloneTokens did not produce independent copy")
	}
}

func TestSpeculative_MaxTokensClampsTargetWindow_Good(t *testing.T) {
	target := GeneratorFunc(func(context.Context, string, GenerateConfig) (Generation, error) {
		return Generation{Tokens: []Token{{ID: 1, Text: "A"}, {ID: 2, Text: "B"}, {ID: 3, Text: "C"}}}, nil
	})
	draft := target
	result, err := Speculative(context.Background(), SpeculativeConfig{
		Prompt: "p", MaxTokens: 2, TargetGenerate: target, DraftGenerate: draft,
	})
	if err != nil {
		t.Fatalf("Speculative() error = %v", err)
	}
	if result.Metrics.EmittedTokens != 2 {
		t.Fatalf("EmittedTokens = %d, want 2 (clamped by MaxTokens)", result.Metrics.EmittedTokens)
	}
}

func TestSpeculative_DraftTokensClampedToMaxTokens_Good(t *testing.T) {
	var draftMax int
	target := GeneratorFunc(func(context.Context, string, GenerateConfig) (Generation, error) {
		return Generation{Tokens: []Token{{ID: 1}}}, nil
	})
	draft := GeneratorFunc(func(_ context.Context, _ string, cfg GenerateConfig) (Generation, error) {
		draftMax = cfg.MaxTokens
		return Generation{Tokens: []Token{{ID: 1}}}, nil
	})
	if _, err := Speculative(context.Background(), SpeculativeConfig{
		Prompt: "p", MaxTokens: 4, DraftTokens: 99, TargetGenerate: target, DraftGenerate: draft,
	}); err != nil {
		t.Fatalf("Speculative() error = %v", err)
	}
	if draftMax != 4 {
		t.Fatalf("draft cfg.MaxTokens = %d, want clamped to MaxTokens=4", draftMax)
	}
}

func TestNormaliseMaxTokens_FirstPositiveOrDefault_Good(t *testing.T) {
	if got := normaliseMaxTokens(0, 0, 7); got != 7 {
		t.Fatalf("normaliseMaxTokens(0,0,7) = %d, want 7", got)
	}
	if got := normaliseMaxTokens(0, 0); got != DefaultMaxTokens {
		t.Fatalf("normaliseMaxTokens(0,0) = %d, want DefaultMaxTokens=%d", got, DefaultMaxTokens)
	}
}

func TestNonZeroDuration_ClampsToNanosecond_Ugly(t *testing.T) {
	if got := nonZeroDuration(0); got != time.Nanosecond {
		t.Fatalf("nonZeroDuration(0) = %v, want 1ns", got)
	}
	if got := nonZeroDuration(-5); got != time.Nanosecond {
		t.Fatalf("nonZeroDuration(-5) = %v, want 1ns", got)
	}
	if got := nonZeroDuration(7 * time.Millisecond); got != 7*time.Millisecond {
		t.Fatalf("nonZeroDuration(7ms) = %v, want passthrough", got)
	}
}
