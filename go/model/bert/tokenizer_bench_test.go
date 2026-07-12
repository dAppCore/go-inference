// SPDX-Licence-Identifier: EUPL-1.2

package bert

import "testing"

// BenchmarkTokenizer_Encode measures WordPiece tokenisation of a typical
// sentence through the synthetic vocab — the per-request cost before the forward.
func BenchmarkTokenizer_Encode(b *testing.B) {
	tk, err := NewTokenizer([]byte(syntheticVocab), true)
	if err != nil {
		b.Fatalf("NewTokenizer: %v", err)
	}
	text := "The quick brown fox playing reset password!"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tk.Encode(text)
	}
}

func BenchmarkTokenizer_EncodePair(b *testing.B) {
	tk, err := NewTokenizer([]byte(syntheticVocab), true)
	if err != nil {
		b.Fatalf("NewTokenizer: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = tk.EncodePair("reset password?", "the quick brown fox")
	}
}
