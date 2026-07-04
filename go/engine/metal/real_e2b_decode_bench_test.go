// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"runtime/debug"
	"testing"
)

// Real-E2B integrated decode-loop bench (AX-11). The synthetic DecodeForward*ICB micro-benches
// measure isolated ops over fixture weights; this measures the INTEGRATED per-token decode path
// of a real gemma4-E2B-it-4bit checkpoint — real (zero-copy mmap'd) weights, a growing resident
// KV cache, the per-layer-input (PLE) tower, and the greedy sampler closing the loop. That is
// where the per-token serving allocations the micro-benches never see actually live (allocs/op ÷
// tokens = allocs/token, paid on EVERY token of EVERY generation — the cost behind tok/s).
//
// Path measured: ArchSession.Generate (LoadDir → resident quant session). It shares
// stepToken + the head encoder with the literal serve path (LoadTokenModelDir →
// model.Generate → StepWithID), which is where the bulk of per-token allocs are, so the session
// path is a faithful, native-scoped proxy for the per-token serve cost.
//
// AX-11 model-loads gate: this loads ~1.5–2 GB, so it is OPT-IN — it skips unless E2B_Q4_DIR
// names the snapshot dir (mirrors the E2B_Q4_DIR convention the other real-model tests use). It
// stays out of `core go qa` / CI, which never set it. To run it:
//
//	export MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib
//	export E2B_Q4_DIR=~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/<rev>
//	go test -tags metal_runtime -run '^$' -bench '^BenchmarkRealE2BDecodeLoop$' -benchmem \
//	        -benchtime=5x -memprofile=/tmp/e2b.alloc ./pkg/native
//
// OOM guard: the session loads ONCE before ResetTimer and is reused across b.N, so memory is the
// flat ~2 GB working set (no per-iteration load — load-in-the-loop is what blew the M3 Ultra up).
// SetMemoryLimit(60 GiB) is a GC backstop, not a budget. maxLen is sized so the reused session's
// position (which accumulates promptLen+maxNew per op) covers benchtimeMax ops without tripping
// the cache cap.

// realE2BDir returns the E2B-4bit snapshot directory, or "" (skip) if unset.
func realE2BDir() string { return os.Getenv("E2B_Q4_DIR") }

const (
	// A fixed, deterministic prompt — real-ish token ids in [0, vocab) for E2B. Using ids (not
	// text) keeps the tokenizer out of the decode-loop measurement; greedy makes the generated
	// ids reproducible (TestRealE2BDecodeDeterministic pins this).
	realE2BPromptLen = 16
	realE2BMaxNew    = 32 // short decode per op (one growing-cache pass), well under the OOM-prone sweep
	realE2BTokens    = realE2BPromptLen + realE2BMaxNew
	realE2BBenchMax  = 5 // -benchtime=5x ceiling (the OOM guard): maxLen must cover this many ops
	// maxLen covers benchMax reused-session ops (pos grows realE2BTokens per op) with headroom.
	realE2BMaxLen = realE2BBenchMax*realE2BTokens + realE2BTokens // 6×48 = 288 cache rows (~tiny)
)

// realE2BPrompt is a fixed prompt of valid E2B token ids — small, spread, deterministic.
func realE2BPrompt() []int32 {
	p := make([]int32, realE2BPromptLen)
	for i := range p {
		p[i] = int32(2 + i*131) // small ids, comfortably within the 256k vocab, never a special-token gap
	}
	return p
}

// BenchmarkRealE2BDecodeLoop measures the heap allocations of the integrated per-token decode
// loop over a real E2B-4bit checkpoint. allocs/op covers realE2BTokens tokens (prefill + decode);
// allocs/token = allocs/op ÷ realE2BTokens. The session is loaded once and reused, so the only
// allocations counted are the per-token decode-path ones (embed, PLE tower, stepToken host
// scratch, head logits, greedy) — not the one-time load.
func BenchmarkRealE2BDecodeLoop(b *testing.B) {
	requireNativeRuntime(b)
	dir := realE2BDir()
	if dir == "" {
		b.Skip("set E2B_Q4_DIR to the gemma-4-e2b-it-4bit snapshot dir (opt-in real-model bench)")
	}
	defer debug.SetMemoryLimit(debug.SetMemoryLimit(60 << 30)) // GC backstop; restore prior on exit

	if b.N > realE2BBenchMax {
		// Honour the OOM guard even if someone passes a larger -benchtime: the reused session's
		// cache is sized for realE2BBenchMax ops only. (b.N>5 ⇒ pos would exceed maxLen mid-run.)
		b.Skipf("real-e2b decode bench is capped at -benchtime=%dx (OOM guard); got b.N=%d", realE2BBenchMax, b.N)
	}

	sess, err := LoadDir(dir, realE2BMaxLen)
	if err != nil {
		b.Fatalf("LoadDir(%s): %v", dir, err)
	}
	defer func() { _ = sess.Close() }()
	prompt := realE2BPrompt()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gen, err := sess.Generate(prompt, realE2BMaxNew, -1) // greedy, no early-EOS — a full maxNew pass
		if err != nil {
			b.Fatalf("Generate (op %d, pos=%d): %v", i, sess.Pos(), err)
		}
		if len(gen) != realE2BMaxNew {
			b.Fatalf("op %d: generated %d tokens, want %d", i, len(gen), realE2BMaxNew)
		}
	}
	b.StopTimer()
	// allocs/token is the figure of merit: b.ReportAllocs prints allocs/op; divide by tokens.
	b.ReportMetric(float64(realE2BTokens), "tokens/op")
}

// TestRealE2BDecodeDeterministic is the byte-identity precondition for the alloc-reduction
// work the bench feeds: greedy decode of a real E2B checkpoint must produce the SAME token ids
// on two independent fresh sessions. Any alloc fix is validated by re-running the bench and
// confirming the generated ids are unchanged; that check is only meaningful if the ids are
// deterministic to begin with. This pins that — same prompt, fresh session each time, identical
// ids — so a later "same ids before/after" claim rests on a verified-stable baseline. Opt-in
// (E2B_Q4_DIR), like the bench. It loads once-per-session (two loads), short maxLen, no sweep.
func TestRealE2BDecodeDeterministic(t *testing.T) {
	requireNativeRuntime(t)
	dir := realE2BDir()
	if dir == "" {
		t.Skip("set E2B_Q4_DIR to the gemma-4-e2b-it-4bit snapshot dir (opt-in real-model test)")
	}
	defer debug.SetMemoryLimit(debug.SetMemoryLimit(60 << 30))
	prompt := realE2BPrompt()

	gen := func() []int32 {
		s, err := LoadDir(dir, realE2BMaxLen)
		if err != nil {
			t.Fatalf("LoadDir(%s): %v", dir, err)
		}
		defer func() { _ = s.Close() }()
		out, err := s.Generate(prompt, realE2BMaxNew, -1)
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		return out
	}

	a, c := gen(), gen()
	if !idsEqual(a, c) {
		t.Fatalf("greedy decode not deterministic across fresh sessions:\n  run1 = %v\n  run2 = %v", a, c)
	}
	for _, id := range a {
		if id < 0 {
			t.Fatalf("negative token id in greedy decode: %v", a)
		}
	}
	t.Logf("real-e2b greedy decode deterministic over %d fresh-session tokens: %v", len(a), a)
}
