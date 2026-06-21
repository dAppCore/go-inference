// SPDX-Licence-Identifier: EUPL-1.2

package classify

import (
	"context"

	"dappco.re/go"
	"dappco.re/go/i18n"
	"dappco.re/go/inference"
	golog "dappco.re/go/log"
)

// ArticlePair holds a noun and its proposed article for validation.
type ArticlePair struct {
	Noun    string
	Article string
}

// ArticleResult reports whether a given article usage is grammatically correct.
type ArticleResult struct {
	Noun      string // the noun being checked
	Given     string // the article provided by the caller
	Predicted string // what the model predicted
	Valid     bool   // Given == Predicted
	Prompt    string // the prompt used (for debugging)
}

// IrregularForm holds a verb, tense, and proposed inflected form for validation.
type IrregularForm struct {
	Verb  string
	Tense string
	Form  string
}

// IrregularResult reports whether a given irregular verb form is correct.
type IrregularResult struct {
	Verb      string // base verb
	Tense     string // tense being checked (e.g. "past", "past participle")
	Given     string // the form provided by the caller
	Predicted string // what the model predicted
	Valid     bool   // Given == Predicted
	Prompt    string // the prompt used (for debugging)
}

// articlePrompt builds a fill-in-the-blank prompt for article prediction.
func articlePrompt(noun string) string {
	return articlePromptForLang(i18n.CurrentLanguage(), noun)
}

func articlePromptForLang(lang, noun string) string {
	noun = core.Trim(noun)
	if isFrenchLanguage(lang) {
		return core.Sprintf(
			"Complete with the correct article (le/la/l'/les/du/au/aux/un/une/des): ___ %s. Answer with just the article:",
			noun,
		)
	}
	return core.Sprintf(
		"Complete with the correct article (a/an/the): ___ %s. Answer with just the article:",
		noun,
	)
}

// irregularPrompt builds a fill-in-the-blank prompt for irregular verb prediction.
func irregularPrompt(verb, tense string) string {
	return core.Sprintf(
		"What is the %s form of the verb '%s'? Answer with just the word:",
		tense, verb,
	)
}

// collectGenerated runs a single-token generation and returns the trimmed, lowercased output.
func collectGenerated(ctx context.Context, m inference.TextModel, prompt string) core.Result {
	sb := core.NewBuilder()
	for tok := range m.Generate(ctx, prompt, inference.WithMaxTokens(1), inference.WithTemperature(0.05)) {
		sb.WriteString(tok.Text)
	}
	if r := m.Err(); !r.OK {
		return r
	}
	return core.Ok(core.Trim(core.Lower(sb.String())))
}

// ValidateArticle checks whether a given article usage is grammatically correct
// by asking the model to predict the correct article in context.
// Uses single-token generation with near-zero temperature for deterministic output.
func ValidateArticle(ctx context.Context, m inference.TextModel, noun string, article string) core.Result {
	prompt := articlePrompt(noun)
	generated := collectGenerated(ctx, m, prompt)
	if !generated.OK {
		return failResult(golog.E("ValidateArticle", "validate: "+noun, core.NewError(generated.Error())))
	}
	predicted := generated.Value.(string)
	given := core.Trim(core.Lower(article))
	return core.Ok(ArticleResult{
		Noun:      noun,
		Given:     given,
		Predicted: predicted,
		Valid:     given == predicted,
		Prompt:    prompt,
	})
}

// ValidateIrregular checks whether a given irregular verb form is correct
// by asking the model to predict the correct form in context.
// Uses single-token generation with near-zero temperature for deterministic output.
func ValidateIrregular(ctx context.Context, m inference.TextModel, verb string, tense string, form string) core.Result {
	prompt := irregularPrompt(verb, tense)
	generated := collectGenerated(ctx, m, prompt)
	if !generated.OK {
		return failResult(golog.E("ValidateIrregular", "validate: "+verb+" ("+tense+")", core.NewError(generated.Error())))
	}
	predicted := generated.Value.(string)
	given := core.Trim(core.Lower(form))
	return core.Ok(IrregularResult{
		Verb:      verb,
		Tense:     tense,
		Given:     given,
		Predicted: predicted,
		Valid:     given == predicted,
		Prompt:    prompt,
	})
}

// BatchValidateArticles validates multiple article-noun pairs efficiently.
// Each pair is validated independently via single-token generation.
func BatchValidateArticles(ctx context.Context, m inference.TextModel, pairs []ArticlePair) core.Result {
	results := make([]ArticleResult, 0, len(pairs))
	for _, p := range pairs {
		r := ValidateArticle(ctx, m, p.Noun, p.Article)
		if !r.OK {
			return r
		}
		results = append(results, r.Value.(ArticleResult))
	}
	return core.Ok(results)
}

// BatchValidateIrregulars validates multiple irregular verb forms efficiently.
// Each form is validated independently via single-token generation.
func BatchValidateIrregulars(ctx context.Context, m inference.TextModel, forms []IrregularForm) core.Result {
	results := make([]IrregularResult, 0, len(forms))
	for _, f := range forms {
		r := ValidateIrregular(ctx, m, f.Verb, f.Tense, f.Form)
		if !r.OK {
			return r
		}
		results = append(results, r.Value.(IrregularResult))
	}
	return core.Ok(results)
}
