// SPDX-Licence-Identifier: EUPL-1.2

package classify

import (
	"context"
	"iter"
	"testing"

	"dappco.re/go"
	"dappco.re/go/i18n"
	"dappco.re/go/inference"
)

// mockGenerateModel satisfies inference.TextModel for validator testing.
// It returns a predetermined token from Generate based on the prompt.
type mockGenerateModel struct {
	generateFunc func(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token]
	genErr       error // error returned by Err() after generation
}

func (m *mockGenerateModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.generateFunc(ctx, prompt, opts...)
}

func (m *mockGenerateModel) Chat(_ context.Context, _ []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {}
}

func (m *mockGenerateModel) Classify(_ context.Context, _ []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (m *mockGenerateModel) BatchGenerate(_ context.Context, _ []string, _ ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (m *mockGenerateModel) ModelType() string                  { return "mock" }
func (m *mockGenerateModel) Info() inference.ModelInfo          { return inference.ModelInfo{} }
func (m *mockGenerateModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *mockGenerateModel) Err() error                         { return m.genErr }
func (m *mockGenerateModel) Close() error                       { return nil }

// newMockArticleModel creates a mock that returns a fixed article token for any prompt.
func newMockArticleModel(article string) *mockGenerateModel {
	return &mockGenerateModel{
		generateFunc: func(_ context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
			return func(yield func(inference.Token) bool) {
				yield(inference.Token{Text: article})
			}
		},
	}
}

// newMockIrregularModel creates a mock that returns different verb forms
// based on a lookup map keyed by verb.
func newMockIrregularModel(forms map[string]string) *mockGenerateModel {
	return &mockGenerateModel{
		generateFunc: func(_ context.Context, prompt string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
			return func(yield func(inference.Token) bool) {
				// Find the matching verb and return its form
				for verb, form := range forms {
					if containsVerb(prompt, verb) {
						yield(inference.Token{Text: form})
						return
					}
				}
				yield(inference.Token{Text: "unknown"})
			}
		},
	}
}

// containsVerb checks if the prompt contains the verb in the expected format.
func containsVerb(prompt, verb string) bool {
	return len(prompt) > 0 && len(verb) > 0 &&
		contains(prompt, core.Sprintf("'%s'", verb))
}

// contains is a simple substring check (avoids importing strings in test).
func contains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestValidateArticle_Correct(t *testing.T) {
	model := newMockArticleModel("a")
	result, err := valueFromResult[ArticleResult](ValidateArticle(context.Background(), model, "book", "a"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Valid {
		t.Errorf("expected Valid=true, got false (Given=%q, Predicted=%q)", result.Given, result.Predicted)
	}
	if result.Predicted != "a" {
		t.Errorf("Predicted = %q, want %q", result.Predicted, "a")
	}
	if result.Noun != "book" {
		t.Errorf("Noun = %q, want %q", result.Noun, "book")
	}
	if result.Prompt == "" {
		t.Error("Prompt should not be empty")
	}
}

func TestValidateArticle_Wrong(t *testing.T) {
	model := newMockArticleModel("a")
	result, err := valueFromResult[ArticleResult](ValidateArticle(context.Background(), model, "book", "an"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Valid {
		t.Errorf("expected Valid=false, got true")
	}
	if result.Given != "an" {
		t.Errorf("Given = %q, want %q", result.Given, "an")
	}
	if result.Predicted != "a" {
		t.Errorf("Predicted = %q, want %q", result.Predicted, "a")
	}
}

func TestValidateArticle_CaseInsensitive(t *testing.T) {
	model := newMockArticleModel("The")
	result, err := valueFromResult[ArticleResult](ValidateArticle(context.Background(), model, "sun", "THE"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Valid {
		t.Errorf("expected Valid=true (case-insensitive), got false (Given=%q, Predicted=%q)", result.Given, result.Predicted)
	}
}

func TestValidateIrregular_Correct(t *testing.T) {
	model := newMockIrregularModel(map[string]string{"go": "went"})
	result, err := valueFromResult[IrregularResult](ValidateIrregular(context.Background(), model, "go", "past", "went"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Valid {
		t.Errorf("expected Valid=true, got false (Given=%q, Predicted=%q)", result.Given, result.Predicted)
	}
	if result.Verb != "go" {
		t.Errorf("Verb = %q, want %q", result.Verb, "go")
	}
	if result.Tense != "past" {
		t.Errorf("Tense = %q, want %q", result.Tense, "past")
	}
	if result.Prompt == "" {
		t.Error("Prompt should not be empty")
	}
}

func TestValidateIrregular_Wrong(t *testing.T) {
	model := newMockIrregularModel(map[string]string{"go": "went"})
	result, err := valueFromResult[IrregularResult](ValidateIrregular(context.Background(), model, "go", "past", "goed"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Valid {
		t.Errorf("expected Valid=false, got true")
	}
	if result.Given != "goed" {
		t.Errorf("Given = %q, want %q", result.Given, "goed")
	}
	if result.Predicted != "went" {
		t.Errorf("Predicted = %q, want %q", result.Predicted, "went")
	}
}

func TestBatchValidateArticles(t *testing.T) {
	// Mock that returns "a" for any prompt
	model := newMockArticleModel("a")
	pairs := []ArticlePair{
		{Noun: "book", Article: "a"},
		{Noun: "apple", Article: "an"},
		{Noun: "car", Article: "a"},
	}
	results, err := valueFromResult[[]ArticleResult](BatchValidateArticles(context.Background(), model, pairs))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3", len(results))
	}
	// "a" == "a" → valid
	if !results[0].Valid {
		t.Errorf("pair 0: expected Valid=true (a/book)")
	}
	// "an" != "a" → invalid
	if results[1].Valid {
		t.Errorf("pair 1: expected Valid=false (an/apple predicted a)")
	}
	// "a" == "a" → valid
	if !results[2].Valid {
		t.Errorf("pair 2: expected Valid=true (a/car)")
	}
}

func TestBatchValidateIrregulars(t *testing.T) {
	model := newMockIrregularModel(map[string]string{
		"go":  "went",
		"eat": "ate",
		"run": "ran",
	})
	forms := []IrregularForm{
		{Verb: "go", Tense: "past", Form: "went"},
		{Verb: "eat", Tense: "past", Form: "eated"},
		{Verb: "run", Tense: "past", Form: "ran"},
	}
	results, err := valueFromResult[[]IrregularResult](BatchValidateIrregulars(context.Background(), model, forms))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3", len(results))
	}
	if !results[0].Valid {
		t.Errorf("form 0: expected Valid=true (went)")
	}
	if results[1].Valid {
		t.Errorf("form 1: expected Valid=false (eated vs ate)")
	}
	if results[1].Predicted != "ate" {
		t.Errorf("form 1: Predicted = %q, want %q", results[1].Predicted, "ate")
	}
	if !results[2].Valid {
		t.Errorf("form 2: expected Valid=true (ran)")
	}
}

func TestBatchValidateArticles_Empty(t *testing.T) {
	model := newMockArticleModel("a")
	results, err := valueFromResult[[]ArticleResult](BatchValidateArticles(context.Background(), model, nil))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("got %d results, want 0", len(results))
	}
}

func TestBatchValidateIrregulars_Empty(t *testing.T) {
	model := newMockIrregularModel(nil)
	results, err := valueFromResult[[]IrregularResult](BatchValidateIrregulars(context.Background(), model, nil))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("got %d results, want 0", len(results))
	}
}

func TestValidateArticle_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	model := &mockGenerateModel{
		generateFunc: func(ctx context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
			return func(yield func(inference.Token) bool) {
				// Context is cancelled — produce no tokens
				if ctx.Err() != nil {
					return
				}
				yield(inference.Token{Text: "a"})
			}
		},
		genErr: context.Canceled,
	}

	_, err := valueFromResult[ArticleResult](ValidateArticle(ctx, model, "book", "a"))
	if err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
}

func TestValidateIrregular_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	model := &mockGenerateModel{
		generateFunc: func(ctx context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
			return func(yield func(inference.Token) bool) {
				if ctx.Err() != nil {
					return
				}
				yield(inference.Token{Text: "went"})
			}
		},
		genErr: context.Canceled,
	}

	_, err := valueFromResult[IrregularResult](ValidateIrregular(ctx, model, "go", "past", "went"))
	if err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
}

func TestValidateArticle_WhitespaceTrimming(t *testing.T) {
	// Model returns token with leading/trailing whitespace
	model := &mockGenerateModel{
		generateFunc: func(_ context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
			return func(yield func(inference.Token) bool) {
				yield(inference.Token{Text: "  a  "})
			}
		},
	}
	result, err := valueFromResult[ArticleResult](ValidateArticle(context.Background(), model, "book", " a "))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Valid {
		t.Errorf("expected Valid=true after trimming, got false (Given=%q, Predicted=%q)", result.Given, result.Predicted)
	}
}

func TestArticlePrompt(t *testing.T) {
	prompt := articlePrompt("elephant")
	if !contains(prompt, "elephant") {
		t.Errorf("prompt should contain the noun: %q", prompt)
	}
	if !contains(prompt, "a/an/the") {
		t.Errorf("prompt should mention article options: %q", prompt)
	}
}

func TestArticlePromptFrenchLocale(t *testing.T) {
	prev := i18n.Default()
	svc, err := serviceFromResult(i18n.New())
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	i18n.SetDefault(svc)
	t.Cleanup(func() {
		i18n.SetDefault(prev)
	})

	if err := errorFromResult(i18n.SetLanguage("fr")); err != nil {
		t.Fatalf("SetLanguage(fr) failed: %v", err)
	}

	prompt := articlePrompt("livre")
	if !contains(prompt, "livre") {
		t.Errorf("prompt should contain the noun: %q", prompt)
	}
	if !contains(prompt, "le/la/l'/les/du/au/aux/un/une/des") {
		t.Errorf("prompt should mention French article options: %q", prompt)
	}
}

func TestIrregularPrompt(t *testing.T) {
	prompt := irregularPrompt("swim", "past participle")
	if !contains(prompt, "'swim'") {
		t.Errorf("prompt should contain the verb: %q", prompt)
	}
	if !contains(prompt, "past participle") {
		t.Errorf("prompt should contain the tense: %q", prompt)
	}
}

// --- AX-7 canonical triplets ---

func TestValidate_ValidateArticle_Good(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := newMockArticleModel("a")
		got, err := valueFromResult[ArticleResult](ValidateArticle(context.Background(), model, "file", "a"))
		if err != nil || !got.Valid {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	if !called {
		t.Fatal("ValidateArticle was not exercised")
	}
}

func TestValidate_ValidateArticle_Bad(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := newMockArticleModel("an")
		got, err := valueFromResult[ArticleResult](ValidateArticle(context.Background(), model, "file", "a"))
		if err != nil || got.Valid {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	if !called {
		t.Fatal("ValidateArticle was not exercised")
	}
}

func TestValidate_ValidateArticle_Ugly(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := newMockArticleModel("")
		got, err := valueFromResult[ArticleResult](ValidateArticle(context.Background(), model, "", ""))
		if err != nil || !got.Valid {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	if !called {
		t.Fatal("ValidateArticle was not exercised")
	}
}

func TestValidate_ValidateIrregular_Good(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := newMockIrregularModel(map[string]string{"go": "went"})
		got, err := valueFromResult[IrregularResult](ValidateIrregular(context.Background(), model, "go", "past", "went"))
		if err != nil || !got.Valid {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	if !called {
		t.Fatal("ValidateIrregular was not exercised")
	}
}

func TestValidate_ValidateIrregular_Bad(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := newMockIrregularModel(map[string]string{"go": "went"})
		got, err := valueFromResult[IrregularResult](ValidateIrregular(context.Background(), model, "go", "past", "goed"))
		if err != nil || got.Valid {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	if !called {
		t.Fatal("ValidateIrregular was not exercised")
	}
}

func TestValidate_ValidateIrregular_Ugly(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := newMockIrregularModel(map[string]string{"": "unknown"})
		got, err := valueFromResult[IrregularResult](ValidateIrregular(context.Background(), model, "", "", "unknown"))
		if err != nil || !got.Valid {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	if !called {
		t.Fatal("ValidateIrregular was not exercised")
	}
}

func TestValidate_BatchValidateArticles_Good(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := newMockArticleModel("a")
		got, err := valueFromResult[[]ArticleResult](BatchValidateArticles(context.Background(), model, []ArticlePair{{Noun: "file", Article: "a"}}))
		if err != nil || len(got) != 1 {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	if !called {
		t.Fatal("BatchValidateArticles was not exercised")
	}
}

func TestValidate_BatchValidateArticles_Bad(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := newMockArticleModel("an")
		got, err := valueFromResult[[]ArticleResult](BatchValidateArticles(context.Background(), model, []ArticlePair{{Noun: "file", Article: "a"}}))
		if err != nil || got[0].Valid {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	if !called {
		t.Fatal("BatchValidateArticles was not exercised")
	}
}

func TestValidate_BatchValidateArticles_Ugly(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := newMockArticleModel("a")
		got, err := valueFromResult[[]ArticleResult](BatchValidateArticles(context.Background(), model, nil))
		if err != nil || len(got) != 0 {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	if !called {
		t.Fatal("BatchValidateArticles was not exercised")
	}
}

func TestValidate_BatchValidateIrregulars_Good(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := newMockIrregularModel(map[string]string{"go": "went"})
		got, err := valueFromResult[[]IrregularResult](BatchValidateIrregulars(context.Background(), model, []IrregularForm{{Verb: "go", Tense: "past", Form: "went"}}))
		if err != nil || len(got) != 1 {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	if !called {
		t.Fatal("BatchValidateIrregulars was not exercised")
	}
}

func TestValidate_BatchValidateIrregulars_Bad(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := newMockIrregularModel(map[string]string{"go": "went"})
		got, err := valueFromResult[[]IrregularResult](BatchValidateIrregulars(context.Background(), model, []IrregularForm{{Verb: "go", Tense: "past", Form: "goed"}}))
		if err != nil || got[0].Valid {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	if !called {
		t.Fatal("BatchValidateIrregulars was not exercised")
	}
}

func TestValidate_BatchValidateIrregulars_Ugly(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := newMockIrregularModel(map[string]string{"go": "went"})
		got, err := valueFromResult[[]IrregularResult](BatchValidateIrregulars(context.Background(), model, nil))
		if err != nil || len(got) != 0 {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	if !called {
		t.Fatal("BatchValidateIrregulars was not exercised")
	}
}
