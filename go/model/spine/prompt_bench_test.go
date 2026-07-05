// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for PromptChunksToString — the iter.Seq[string] prompt
// assembler. Fires once per prompt that arrives as a chunk sequence
// (template-rendered chat turns, streamed prompt parts), so the per-call
// builder allocation pattern is the prompt-side assembly budget.
//
// Run:    go test -bench='BenchmarkPrompt_' -benchmem -run='^$' ./spine

package spine

import "testing"

// Sink defeats compiler DCE.
var promptBenchSinkString string

// chunkSeqOf returns an iter.Seq[string] yielding the given chunks. The
// closure captures the slice; the builder cost in PromptChunksToString is
// what the bench measures, so the source slice is built once outside the
// timed loop.
func chunkSeqOf(chunks []string) func(yield func(string) bool) {
	return func(yield func(string) bool) {
		for _, c := range chunks {
			if !yield(c) {
				return
			}
		}
	}
}

func BenchmarkPrompt_PromptChunksToString_FewChunks(b *testing.B) {
	// The common chat-template shape: a handful of role/marker/body chunks.
	seq := chunkSeqOf([]string{"<start_of_turn>user\n", "Explain the spine package.", "<end_of_turn>\n"})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		promptBenchSinkString = PromptChunksToString(seq)
	}
}

func BenchmarkPrompt_PromptChunksToString_ManyChunks(b *testing.B) {
	// A long multi-turn prompt streamed as many short chunks — exercises
	// the strings.Builder regrow path (no length is available from an
	// iter.Seq, so the builder grows geometrically).
	chunks := make([]string, 64)
	for i := range chunks {
		chunks[i] = "chunk-of-prompt-text "
	}
	seq := chunkSeqOf(chunks)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		promptBenchSinkString = PromptChunksToString(seq)
	}
}

func BenchmarkPrompt_PromptChunksToString_Nil(b *testing.B) {
	// The early-return path — no builder allocation.
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		promptBenchSinkString = PromptChunksToString(nil)
	}
}
