// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gptoss

import (
	"os"
	"strings"
	"testing"

	"dappco.re/go/inference/decode/tokenizer"
	native "dappco.re/go/inference/engine/metal"
)

// live_generate_test.go is the #37 merge gate: a REAL greedy generation through the full serving
// path — model.Load (this package's registration) → the engine's quant arch session (linear-KV
// sinks lane) → greedy decode — against the locally-cached gpt-oss-20b-MLX-4bit checkpoint.
// Runtime-gated twice: the checkpoint dir (resolveGptOssDir, skips cleanly when absent) and
// MLX_METALLIB_PATH (the orchestrator sets it at merge; without it the engine cannot dispatch).
// The import direction is test-only: package gptoss's TEST binary imports engine/metal; the
// production package never does (engine/metal is the consumer of model/, not the reverse).

// TestLive_RealCheckpoint_GreedyGeneration loads the real checkpoint and asserts BOTH briefed
// outputs: (1) a few-shot capital-city prompt greedily completes with "Paris" — the strongest
// deterministic factual probe a greedy decode has (the pattern-completion shape holds for base and
// chat tunes alike); (2) a second, unrelated prompt produces a NON-DEGENERATE continuation
// (multiple distinct tokens, non-empty text) — the anti-collapse check that catches a decode
// producing one repeated token while the first prompt happens to pass.
func TestLive_RealCheckpoint_GreedyGeneration(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skip("metallib not set — the orchestrator runs this gate at merge")
	}
	dir := resolveGptOssDir(t)

	tok, err := tokenizer.LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	sess, err := native.LoadDir(dir, 512)
	if err != nil {
		t.Fatalf("LoadDir (the full registered load path — parse, Arch, assemble, quant session): %v", err)
	}
	defer sess.Close()

	// Gate 1: the briefed "Paris" assertion, as few-shot pattern completion (greedy).
	prompt := "Q: What is the capital of Germany? A: Berlin. Q: What is the capital of France? A:"
	ids := tok.Encode(prompt)
	if len(ids) == 0 {
		t.Fatal("tokeniser produced no ids for the capital prompt")
	}
	if err := sess.PrefillTokens(ids); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	out, err := sess.GenerateFromCache(12, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache: %v", err)
	}
	text := tok.Decode(out)
	t.Logf("capital prompt greedy continuation (%d tokens): %q", len(out), text)
	if !strings.Contains(text, "Paris") {
		t.Fatalf("greedy continuation %q does not contain \"Paris\" — the briefed factual gate failed", text)
	}

	// Gate 2: a second, unrelated prompt must be non-degenerate — several distinct tokens and
	// non-empty decoded text (a single token repeated 12 times is the classic wrong-scale/wrong-
	// activation collapse shape).
	ids2 := tok.Encode("Once upon a time, in a small village by the sea,")
	if err := sess.PrefillTokens(ids2); err != nil {
		t.Fatalf("PrefillTokens(second prompt): %v", err)
	}
	out2, err := sess.GenerateFromCache(16, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache(second prompt): %v", err)
	}
	if len(out2) < 8 {
		t.Fatalf("second prompt generated only %d tokens, want >= 8", len(out2))
	}
	distinct := map[int32]bool{}
	for _, id := range out2 {
		distinct[id] = true
	}
	if len(distinct) < 3 {
		t.Fatalf("second prompt's %d tokens collapse to %d distinct ids (%v) — degenerate decode", len(out2), len(distinct), out2)
	}
	text2 := tok.Decode(out2)
	if strings.TrimSpace(text2) == "" {
		t.Fatalf("second prompt decoded to empty/whitespace text from ids %v", out2)
	}
	t.Logf("second prompt greedy continuation (%d tokens, %d distinct): %q", len(out2), len(distinct), text2)
}
