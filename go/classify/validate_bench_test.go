package classify

import (
	"context"
	"testing"

	"dappco.re/go"
)

func BenchmarkArticlePromptForLang(b *testing.B) {
	// English (a/an/the) and French (le/la/...) prompt construction.
	cases := []struct{ lang, noun string }{
		{"en", "elephant"},
		{"fr", "livre"},
	}
	for _, c := range cases {
		b.Run(c.lang, func(b *testing.B) {
			b.ReportAllocs()
			var s string
			for i := 0; i < b.N; i++ {
				s = articlePromptForLang(c.lang, c.noun)
			}
			benchStringSink = s
		})
	}
}

func BenchmarkIrregularPrompt(b *testing.B) {
	b.ReportAllocs()
	var s string
	for i := 0; i < b.N; i++ {
		s = irregularPrompt("swim", "past participle")
	}
	benchStringSink = s
}

func BenchmarkCollectGenerated(b *testing.B) {
	model := newMockArticleModel("a")
	ctx := context.Background()
	prompt := articlePrompt("book")
	b.ReportAllocs()
	var r core.Result
	for i := 0; i < b.N; i++ {
		r = collectGenerated(ctx, model, prompt)
	}
	benchResultSink = r
}

func BenchmarkValidateArticle(b *testing.B) {
	model := newMockArticleModel("a")
	ctx := context.Background()
	b.ReportAllocs()
	var r core.Result
	for i := 0; i < b.N; i++ {
		r = ValidateArticle(ctx, model, "book", "a")
	}
	benchResultSink = r
}

func BenchmarkValidateIrregular(b *testing.B) {
	model := newMockIrregularModel(map[string]string{"go": "went"})
	ctx := context.Background()
	b.ReportAllocs()
	var r core.Result
	for i := 0; i < b.N; i++ {
		r = ValidateIrregular(ctx, model, "go", "past", "went")
	}
	benchResultSink = r
}

func BenchmarkBatchValidateArticles(b *testing.B) {
	model := newMockArticleModel("a")
	ctx := context.Background()
	pairs := []ArticlePair{
		{Noun: "book", Article: "a"},
		{Noun: "apple", Article: "an"},
		{Noun: "car", Article: "a"},
		{Noun: "elephant", Article: "an"},
	}
	b.ReportAllocs()
	var r core.Result
	for i := 0; i < b.N; i++ {
		r = BatchValidateArticles(ctx, model, pairs)
	}
	benchResultSink = r
}

func BenchmarkBatchValidateIrregulars(b *testing.B) {
	model := newMockIrregularModel(map[string]string{"go": "went", "eat": "ate", "run": "ran"})
	ctx := context.Background()
	forms := []IrregularForm{
		{Verb: "go", Tense: "past", Form: "went"},
		{Verb: "eat", Tense: "past", Form: "ate"},
		{Verb: "run", Tense: "past", Form: "ran"},
	}
	b.ReportAllocs()
	var r core.Result
	for i := 0; i < b.N; i++ {
		r = BatchValidateIrregulars(ctx, model, forms)
	}
	benchResultSink = r
}
