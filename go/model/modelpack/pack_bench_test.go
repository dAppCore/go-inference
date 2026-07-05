// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the pack utilities — option apply, issue accumulation,
// summary helpers. Per AX-11 — ApplyOptions runs once per Inspect call;
// AddIssue/HasIssue/HasErrorIssue/IssueSummary fire per issue and at the
// final validity gate. Cheap per-call but on the model-pack hot path.
//
// Run:    go test -bench=Benchmark -benchmem -run='^$' ./go/pack

package modelpack

import "testing"

// Sinks defeat compiler DCE.
var (
	packSinkConfig ModelPackConfig
	packSinkBool   bool
	packSinkString string
)

// --- ApplyOptions — once per Inspect call ---

func BenchmarkPack_ApplyOptions_Defaults(b *testing.B) {
	var opts []ModelPackOption
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		packSinkConfig = ApplyOptions(opts)
	}
}

func BenchmarkPack_ApplyOptions_All(b *testing.B) {
	opts := []ModelPackOption{
		WithPackQuantization(4),
		WithPackMaxContextLength(131072),
		WithPackRequireChatTemplate(false),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		packSinkConfig = ApplyOptions(opts)
	}
}

// --- HasIssue / Valid / HasErrorIssue ---

func benchPackWithIssues() ModelPack {
	pack := ModelPack{}
	pack.AddIssue(ModelPackIssueError, ModelPackIssueMissingConfig, "config missing", "/tmp/x/config.json")
	pack.AddIssue(ModelPackIssueWarning, ModelPackIssueMissingChatTemplate, "chat template missing", "/tmp/x")
	pack.AddIssue(ModelPackIssueError, ModelPackIssueUnsupportedRuntime, "runtime not implemented", "/tmp/x")
	pack.AddIssue(ModelPackIssueWarning, ModelPackIssueQuantizationMismatch, "quant 8, want 4", "/tmp/x")
	pack.AddIssue(ModelPackIssueError, ModelPackIssueContextTooLarge, "ctx 200000 > 131072", "/tmp/x")
	return pack
}

func BenchmarkPack_HasIssue_Present(b *testing.B) {
	pack := benchPackWithIssues()
	target := ModelPackIssueContextTooLarge
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		packSinkBool = pack.HasIssue(target)
	}
}

func BenchmarkPack_HasIssue_Missing(b *testing.B) {
	pack := benchPackWithIssues()
	target := ModelPackIssueInvalidGGUF
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		packSinkBool = pack.HasIssue(target)
	}
}

func BenchmarkPack_HasErrorIssue(b *testing.B) {
	pack := benchPackWithIssues()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		packSinkBool = pack.HasErrorIssue()
	}
}

func BenchmarkPack_Valid(b *testing.B) {
	pack := ModelPack{OK: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		packSinkBool = pack.Valid()
	}
}

// --- AddIssue — issue accumulation ---

func BenchmarkPack_AddIssue(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pack := ModelPack{}
		pack.AddIssue(ModelPackIssueError, ModelPackIssueMissingConfig, "config missing", "/tmp/x/config.json")
	}
}

// --- IssueSummary — fires when Validate() rejects a pack ---

func BenchmarkPack_IssueSummary_FiveErrors(b *testing.B) {
	pack := benchPackWithIssues()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		packSinkString = pack.IssueSummary()
	}
}

func BenchmarkPack_IssueSummary_Empty(b *testing.B) {
	pack := ModelPack{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		packSinkString = pack.IssueSummary()
	}
}
