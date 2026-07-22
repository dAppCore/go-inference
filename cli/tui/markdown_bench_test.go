// SPDX-Licence-Identifier: EUPL-1.2

package tui

import "testing"

var sinkMarkdownTranscript string

func BenchmarkMarkdownTranscript(b *testing.B) {
	renderer := newMarkdownRenderer("midnight")
	turns := make([]string, 20)
	for i := range turns {
		turns[i] = "## Assistant result\n\n- item one\n- item two\n\n```go\nfunc answer() int { return 42 }\n```"
	}
	for i, content := range turns {
		sinkMarkdownTranscript = renderer.Render(benchmarkTurnID(i), content, 78)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		for i, content := range turns {
			sinkMarkdownTranscript = renderer.Render(benchmarkTurnID(i), content, 78)
		}
	}
}

func benchmarkTurnID(index int) string {
	const digits = "0123456789"
	return "benchmark-turn-" + string(digits[index/10]) + string(digits[index%10])
}
