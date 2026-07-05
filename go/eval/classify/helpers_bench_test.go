package classify

import "testing"

func BenchmarkIsFrenchLanguage(b *testing.B) {
	// "en" and "fr" are the common lowercase-tag inputs; "fr-CA" exercises
	// the prefix branch.
	cases := []string{"en", "fr", "fr-CA"}
	for _, lang := range cases {
		b.Run(lang, func(b *testing.B) {
			b.ReportAllocs()
			var v bool
			for i := 0; i < b.N; i++ {
				v = isFrenchLanguage(lang)
			}
			benchBoolSink = v
		})
	}
}
