// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"runtime/debug"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/internal/enginegate"
	"dappco.re/go/inference/tokenizer"
)

const (
	realE2BAssistantDraftTokens = 4
	realE2BAssistantPromptText  = "Explain speculative decoding in simple words."
)

func TestRealE2BAssistantPromptUsesWordedTokenizerInput(t *testing.T) {
	requireNativeRuntime(t)
	targetDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")

	prompt := realE2BAssistantPrompt(t, targetDir)

	if len(prompt) < 6 {
		t.Fatalf("assistant prompt token count = %d, want a few word tokens", len(prompt))
	}
	if idsEqual(prompt, realE2BPrompt()) {
		t.Fatal("assistant prompt uses synthetic token-id fixture, want tokenizer-encoded words")
	}
}

func realE2BAssistantPrompt(tb testing.TB, modelDir string) []int32 {
	tb.Helper()
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(modelDir, "tokenizer.json"))
	if err != nil {
		tb.Fatalf("LoadTokenizer(%s): %v", modelDir, err)
	}
	prompt := tok.Encode(realE2BAssistantPromptText)
	if len(prompt) == 0 {
		tb.Fatal("assistant prompt tokenized to no ids")
	}
	return prompt
}

// BenchmarkRealE2BAssistantPair compares the plain native E2B decode loop with
// the native Gemma 4 assistant-pair route over the same cached E2B model. It is
// opt-in through the Hugging Face cache, matching the live assistant metadata
// test, and reports assistant acceptance/draft counters so the MTP lane can be
// improved against a stable real-model surface instead of synthetic fixtures.
func BenchmarkRealE2BAssistantPair(b *testing.B) {
	requireNativeRuntime(b)
	targetDir := enginegate.HFModelPath(b, "mlx-community/gemma-4-e2b-it-4bit")
	assistantDir := enginegate.HFModelPath(b, "mlx-community/gemma-4-E2B-it-assistant-bf16")
	prompt := realE2BAssistantPrompt(b, targetDir)
	tokensPerOp := len(prompt) + realE2BMaxNew

	b.Run("plain", func(b *testing.B) {
		defer debug.SetMemoryLimit(debug.SetMemoryLimit(60 << 30))
		if b.N > realE2BBenchMax {
			b.Skipf("real-e2b assistant plain bench is capped at -benchtime=%dx (OOM guard); got b.N=%d", realE2BBenchMax, b.N)
		}
		sess, err := LoadDir(targetDir, realE2BMaxLen)
		if err != nil {
			b.Fatalf("LoadDir(%s): %v", targetDir, err)
		}
		defer func() { _ = sess.Close() }()

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			gen, err := sess.Generate(prompt, realE2BMaxNew, -1)
			if err != nil {
				b.Fatalf("Generate (op %d, pos=%d): %v", i, sess.Pos(), err)
			}
			if len(gen) != realE2BMaxNew {
				b.Fatalf("op %d: generated %d tokens, want %d", i, len(gen), realE2BMaxNew)
			}
		}
		b.StopTimer()
		b.ReportMetric(float64(tokensPerOp), "tokens/op")
	})

	b.Run("assistant", func(b *testing.B) {
		defer debug.SetMemoryLimit(debug.SetMemoryLimit(60 << 30))
		if b.N > realE2BBenchMax {
			b.Skipf("real-e2b assistant bench is capped at -benchtime=%dx (OOM guard); got b.N=%d", realE2BBenchMax, b.N)
		}
		sess, err := LoadDir(targetDir, realE2BMaxLen)
		if err != nil {
			b.Fatalf("LoadDir(%s): %v", targetDir, err)
		}
		defer func() { _ = sess.Close() }()
		pair, err := LoadAssistantPairDirs(targetDir, assistantDir)
		if err != nil {
			b.Fatalf("LoadAssistantPairDirs(%s, %s): %v", targetDir, assistantDir, err)
		}
		defer pair.Close()

		var totals AssistantGenerateResult
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res, err := pair.GenerateFromSession(sess, prompt, realE2BMaxNew, -1, realE2BAssistantDraftTokens, nil)
			if err != nil {
				b.Fatalf("GenerateFromSession (op %d, pos=%d): %v", i, sess.Pos(), err)
			}
			if len(res.Tokens) != realE2BMaxNew {
				b.Fatalf("op %d: generated %d tokens, want %d", i, len(res.Tokens), realE2BMaxNew)
			}
			totals.DraftCalls += res.DraftCalls
			totals.DraftTokens += res.DraftTokens
			totals.TargetCalls += res.TargetCalls
			totals.TargetVerifyCalls += res.TargetVerifyCalls
			totals.AcceptedTokens += res.AcceptedTokens
			totals.RejectedTokens += res.RejectedTokens
			totals.TargetTokens += res.TargetTokens
		}
		b.StopTimer()
		b.ReportMetric(float64(tokensPerOp), "tokens/op")
		if b.N > 0 {
			b.ReportMetric(float64(totals.DraftTokens)/float64(b.N), "draft-tokens/op")
			b.ReportMetric(float64(totals.AcceptedTokens)/float64(b.N), "accepted-tokens/op")
			b.ReportMetric(float64(totals.RejectedTokens)/float64(b.N), "rejected-tokens/op")
			b.ReportMetric(float64(totals.TargetCalls)/float64(b.N), "target-calls/op")
			b.ReportMetric(float64(totals.TargetVerifyCalls)/float64(b.N), "target-verify-calls/op")
		}
	})

	b.Run("assistant_stream", func(b *testing.B) {
		defer debug.SetMemoryLimit(debug.SetMemoryLimit(60 << 30))
		if b.N > realE2BBenchMax {
			b.Skipf("real-e2b assistant stream bench is capped at -benchtime=%dx (OOM guard); got b.N=%d", realE2BBenchMax, b.N)
		}
		sess, err := LoadDir(targetDir, realE2BMaxLen)
		if err != nil {
			b.Fatalf("LoadDir(%s): %v", targetDir, err)
		}
		defer func() { _ = sess.Close() }()
		pair, err := LoadAssistantPairDirs(targetDir, assistantDir)
		if err != nil {
			b.Fatalf("LoadAssistantPairDirs(%s, %s): %v", targetDir, assistantDir, err)
		}
		defer pair.Close()

		var totals AssistantGenerateResult
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			streamed := 0
			res, err := pair.GenerateFromSessionEach(sess, prompt, realE2BMaxNew, -1, realE2BAssistantDraftTokens, nil, func(int32) bool {
				streamed++
				return true
			})
			if err != nil {
				b.Fatalf("GenerateFromSessionEach (op %d, pos=%d): %v", i, sess.Pos(), err)
			}
			if len(res.Tokens) != realE2BMaxNew || streamed != realE2BMaxNew {
				b.Fatalf("op %d: generated/streamed %d/%d tokens, want %d", i, len(res.Tokens), streamed, realE2BMaxNew)
			}
			totals.DraftCalls += res.DraftCalls
			totals.DraftTokens += res.DraftTokens
			totals.TargetCalls += res.TargetCalls
			totals.TargetVerifyCalls += res.TargetVerifyCalls
			totals.AcceptedTokens += res.AcceptedTokens
			totals.RejectedTokens += res.RejectedTokens
			totals.TargetTokens += res.TargetTokens
		}
		b.StopTimer()
		b.ReportMetric(float64(tokensPerOp), "tokens/op")
		if b.N > 0 {
			b.ReportMetric(float64(totals.DraftTokens)/float64(b.N), "draft-tokens/op")
			b.ReportMetric(float64(totals.AcceptedTokens)/float64(b.N), "accepted-tokens/op")
			b.ReportMetric(float64(totals.RejectedTokens)/float64(b.N), "rejected-tokens/op")
			b.ReportMetric(float64(totals.TargetCalls)/float64(b.N), "target-calls/op")
			b.ReportMetric(float64(totals.TargetVerifyCalls)/float64(b.N), "target-verify-calls/op")
		}
	})
}
