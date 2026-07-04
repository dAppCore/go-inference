// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"runtime/debug"
	"testing"
)

// Real-E2B integrated PREFILL (prompt-processing) bench (AX-11, Pass-2). The sibling decode bench
// (BenchmarkRealE2BDecodeLoop) measures the STEADY-STATE per-token loop; this measures the other
// phase the audit had no bench for — processing a whole 256-token prompt before the first
// generated token. It loads the same real gemma-4-e2b-it-4bit checkpoint through pkg/native and
// runs Generate(prompt, maxNew=1): every prompt id is pushed through the resident decode state and
// the growing KV cache, then exactly ONE token is decoded (one head + one greedy) to close the
// real entry point. allocs/op ÷ promptLen = allocs/prompt-token, the figure the ICB prefill lever
// is sized against; the single decode token is ~1/promptLen of the op (rounding error in object
// count), so no head-netting is needed — divide and note it.
//
// SHAPE — the headline finding, not a footnote. pkg/native is a PER-TOKEN backend: it has NO
// batched / CaptureKV / chunked-prefill path (those live in the cgo pkg/metal engine —
// pkg/metal/model/gemma4/forward.go, kv_snapshot.go — a different backend, out of scope here). The
// session's Generate prefills the prompt by calling archDecodeState.stepToken ONCE PER PROMPT
// TOKEN (gemma4_session.go Generate, the prompt loop), structurally identical to decode minus the
// per-token head + greedy. So "prefill processed at once" does not hold for the no-cgo backend;
// prefill's encoder-call pattern = decode's per-token pattern, head-free, repeated promptLen times.
//
// Why prefill is the STRONGER ICB target (this is what "size the PREFILL side of the lever" means).
// For E2B (a per-layer-input model) stepToken does NOT issue one command buffer per token: each
// PLE-gated layer EndEncoding→Commit→WaitUntilCompleted mid-token to run the host-side gate, then
// resumes a fresh encoder (decode_forward_arch.go stepToken), plus the one final commit+wait. That
// is ~numLayers+1 CPU↔GPU syncs PER TOKEN. In DECODE those syncs are partly unavoidable — the next
// token's input is the greedy-sampled output of this one, a host round-trip the GPU must wait for.
// In PREFILL every prompt id is known UP FRONT: there is no per-token CPU sampling between tokens,
// so the promptLen × (numLayers+1) commit+waits are collapsible into far fewer encoded buffers with
// no intervening host sync — a bigger encode-bypass win than decode can offer.
//
// AX-11 model-loads gate + OOM guard (same contract as the decode bench): OPT-IN via E2B_Q4_DIR
// (skips in core go qa / CI, which never set it); session loaded ONCE before ResetTimer and reused
// across b.N (flat ~2 GB working set, no load-in-the-loop); -benchtime=5x ceiling enforced;
// SetMemoryLimit(60 GiB) GC backstop; maxLen sized to cover the 5-op cap (pos grows promptLen+1 per
// op). Run it:
//
//	export MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib
//	export E2B_Q4_DIR=~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/<rev>
//	go test -tags metal_runtime -run '^$' -bench '^BenchmarkRealE2BPrefill$' -benchmem \
//	        -benchtime=5x -memprofile=/tmp/e2b-prefill.alloc ./pkg/native

const (
	// A 256-token prompt — the prefill phase the decode bench's 16-token prompt only grazes, in the
	// task's 256–512 range. Real-ish ids spread across the vocab; greedy makes the one generated id
	// reproducible per fresh session (TestRealE2BPrefillDeterministic pins it).
	realE2BPrefillLen     = 256
	realE2BPrefillMaxNew  = 1 // prefill the whole prompt, decode exactly one token (the real entry point)
	realE2BPrefillBenchMx = 5 // -benchtime=5x ceiling (OOM guard): maxLen must cover this many ops
	// maxLen covers benchMx reused-session ops; pos grows promptLen+maxNew per op, plus one op headroom.
	realE2BPrefillMaxLen = realE2BPrefillBenchMx*(realE2BPrefillLen+realE2BPrefillMaxNew) + (realE2BPrefillLen + realE2BPrefillMaxNew)
)

// realE2BPrefillPrompt is a fixed prompt of valid E2B token ids — spread across the vocab, small,
// deterministic. Distinct stride from realE2BPrompt so the two benches don't alias one fixture.
func realE2BPrefillPrompt() []int32 {
	p := make([]int32, realE2BPrefillLen)
	for i := range p {
		p[i] = int32(3 + i*61) // 256 ids up to ~15.6k, comfortably within the 256k vocab
	}
	return p
}

// BenchmarkRealE2BPrefill measures the heap allocations of processing a 256-token prompt through
// pkg/native — the real prefill path (Generate, maxNew=1). allocs/op covers promptLen prompt
// tokens + 1 decoded token; allocs/prompt-token = allocs/op ÷ promptLen (the decode token is
// ~1/promptLen, negligible in object count). The session is loaded once and reused, so the counted
// allocations are the per-prompt-token prefill ones (embed, PLE tower, stepToken host scratch +
// the FFI marshalling behind each Metal encoder call), not the one-time load.
func BenchmarkRealE2BPrefill(b *testing.B) {
	requireNativeRuntime(b)
	dir := realE2BDir()
	if dir == "" {
		b.Skip("set E2B_Q4_DIR to the gemma-4-e2b-it-4bit snapshot dir (opt-in real-model bench)")
	}
	defer debug.SetMemoryLimit(debug.SetMemoryLimit(40 << 30)) // GC backstop; 40 GiB keeps GC ahead of the 256-tok prefill's heap churn

	if b.N > realE2BPrefillBenchMx {
		// Honour the OOM guard even if someone passes a larger -benchtime: the reused session's cache
		// is sized for realE2BPrefillBenchMx ops only (pos would exceed maxLen mid-run beyond that).
		b.Skipf("real-e2b prefill bench is capped at -benchtime=%dx (OOM guard); got b.N=%d", realE2BPrefillBenchMx, b.N)
	}

	sess, err := LoadDir(dir, realE2BPrefillMaxLen)
	if err != nil {
		b.Fatalf("LoadDir(%s): %v", dir, err)
	}
	defer func() { _ = sess.Close() }()
	prompt := realE2BPrefillPrompt()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gen, err := sess.Generate(prompt, realE2BPrefillMaxNew, -1) // greedy, one token after the full prefill
		if err != nil {
			b.Fatalf("Generate (op %d, pos=%d): %v", i, sess.Pos(), err)
		}
		// Reused session ⇒ each op prefills at a later cache position with prior context resident, so
		// the generated id legitimately differs op-to-op; only the count is invariant here. Byte-identity
		// is pinned per fresh session in TestRealE2BPrefillDeterministic, not in this reused-session loop.
		if len(gen) != realE2BPrefillMaxNew {
			b.Fatalf("op %d: generated %d tokens, want %d", i, len(gen), realE2BPrefillMaxNew)
		}
	}
	b.StopTimer()
	// allocs/prompt-token is the figure of merit: b.ReportAllocs prints allocs/op; divide by promptLen.
	b.ReportMetric(float64(realE2BPrefillLen), "prompt-tokens/op")
}

// TestRealE2BPrefillDeterministic is the byte-identity precondition for the alloc-reduction work
// the prefill bench feeds: greedy decode of the first token AFTER a real 256-token prefill must
// produce the SAME id on two independent fresh sessions. Any prefill-side alloc fix is validated by
// re-running and confirming this id is unchanged; that check is only meaningful if it is
// deterministic to begin with. Fresh session per call (so position starts at 0 each time, unlike
// the reused-session bench), opt-in (E2B_Q4_DIR), short maxLen, no sweep.
func TestRealE2BPrefillDeterministic(t *testing.T) {
	requireNativeRuntime(t)
	dir := realE2BDir()
	if dir == "" {
		t.Skip("set E2B_Q4_DIR to the gemma-4-e2b-it-4bit snapshot dir (opt-in real-model test)")
	}
	defer debug.SetMemoryLimit(debug.SetMemoryLimit(40 << 30)) // 40 GiB: keep GC ahead of the 256-tok prefill heap churn
	prompt := realE2BPrefillPrompt()

	gen := func() []int32 {
		s, err := LoadDir(dir, realE2BPrefillMaxLen)
		if err != nil {
			t.Fatalf("LoadDir(%s): %v", dir, err)
		}
		defer func() { _ = s.Close() }()
		out, err := s.Generate(prompt, realE2BPrefillMaxNew, -1)
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		return out
	}

	a, c := gen(), gen()
	if !idsEqual(a, c) {
		t.Fatalf("post-prefill greedy decode not deterministic across fresh sessions:\n  run1 = %v\n  run2 = %v", a, c)
	}
	for _, id := range a {
		if id < 0 {
			t.Fatalf("negative token id after prefill: %v", a)
		}
	}
	t.Logf("real-e2b post-256-prefill greedy decode deterministic: %v", a)
}
