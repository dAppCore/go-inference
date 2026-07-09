// SPDX-Licence-Identifier: EUPL-1.2

package capability

import (
	"testing"

	"dappco.re/go"
)

// StripThinkBlocks runs on every DeepSeek/R1 response before scoring —
// the per-sample cleanup hot path. Note: 30 allocs/op baseline is dominated
// by the regex.MustCompile inside the function; treat the budget as a
// shape-lock, not an endorsement.

func BenchmarkStripThinkBlocks_NoBlock(b *testing.B) {
	response := "Just a plain answer with no think block at all."
	b.ReportAllocs()
	for b.Loop() {
		StripThinkBlocks(response)
	}
}

func BenchmarkStripThinkBlocks_Small(b *testing.B) {
	response := "<think>internal reasoning</think>The actual answer is 42."
	b.ReportAllocs()
	for b.Loop() {
		StripThinkBlocks(response)
	}
}

func BenchmarkStripThinkBlocks_Large(b *testing.B) {
	// Realistic R1 shape: 2-3kb thinking, short final answer.
	sb := core.NewBuilder()
	_, _ = sb.WriteString("<think>")
	for range 50 {
		_, _ = sb.WriteString("Let me work through this step by step. ")
	}
	_, _ = sb.WriteString("</think>The final answer is 42.")
	response := sb.String()
	b.ResetTimer()
	b.ReportAllocs()
	for b.Loop() {
		StripThinkBlocks(response)
	}
}

func BenchmarkStripThinkBlocks_OnlyThink(b *testing.B) {
	// Edge case: model emitted only the think block, no answer.
	response := "<think>The model never closed its reasoning here, output is just the thinking with no final answer payload.</think>"
	b.ReportAllocs()
	for b.Loop() {
		StripThinkBlocks(response)
	}
}

func BenchmarkProbeCategories(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		probeCategoriesSink = ProbeCategories()
	}
}

func BenchmarkProbeDomains(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		probeDomainsSink = ProbeDomains()
	}
}

// BenchmarkProbeChecksAll runs every probe Check against a realistic model
// response — the per-sample scoring hot path. Each Check that compiles a
// regex per call shows up here; a typical scoring run fires all 23 against
// each of N model outputs.
func BenchmarkProbeChecksAll(b *testing.B) {
	response := "Let me work through this carefully. After reasoning about the " +
		"problem, the final answer is 162 and the value of x = -12. The starter " +
		"motor is the likely cause, and the answer is yes, you are facing south."
	b.ReportAllocs()
	for b.Loop() {
		for i := range CapabilityProbes {
			probeCheckSink = CapabilityProbes[i].Check(response)
		}
	}
}

var (
	probeCategoriesSink []string
	probeDomainsSink    []string
	probeCheckSink      bool
)
