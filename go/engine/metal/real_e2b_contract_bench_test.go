// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"runtime/debug"
	"testing"

	"dappco.re/go/inference/model"
)

// Real-E2B CONTRACT token-loop bench (AX-11). The sibling decode/prefill benches measure
// pkg/native's own ArchSession.Generate API; this measures the BACKEND-AGNOSTIC contract path —
// model.Generate / model.GenerateSampled (pkg/model/token.go) driving the real E2B TokenModel
// returned by LoadTokenModelDir. That is the literal serve route: register_native.go's
// nativeTextModel.stream calls model.Generate / model.GenerateSampled over exactly this TokenModel,
// and (since the native token model is a SessionModel whose stepper is StepWithID-aware) the loop
// runs through generateStepwise → StepWithID. No existing bench exercised the contract layer over a
// real model — the decode/prefill benches note they are a "native-scoped proxy" for it. This closes
// that: the per-token allocations counted here include the pkg/model contract loop's own scratch
// (the gen []int32 slice, the per-token step closure, embed/head/pick handoff) on top of the
// native stepper, so a contract-layer alloc fix is measured end to end against the real model.
//
// allocs/op covers contractTokens tokens (prefill + decode); allocs/token = allocs/op ÷
// contractTokens, the figure paid on every served token. The token model loads ONCE before
// ResetTimer and is reused across b.N, so only the per-token contract+decode allocations are
// counted, not the one-time load.
//
// AX-11 model-loads gate + OOM guard (identical contract to the decode/prefill benches): OPT-IN via
// E2B_Q4_DIR (skips in core go qa / CI, which never set it); the token model is loaded once and
// reused (flat ~2 GB working set — load-in-the-loop is what blew the M3 Ultra up); -benchtime=5x
// ceiling enforced; SetMemoryLimit(60 GiB) GC backstop. The contract loop has no persistent cursor
// of its own — generateStepwise opens a FRESH session per call (OpenSession) and Close frees it —
// so each op starts the cache at position 0; maxLen need only cover one prompt+decode pass, not b.N
// of them (unlike the session benches whose reused session accumulates position).
//
// Run it:
//
//	export MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib
//	export E2B_Q4_DIR=~/.cache/huggingface/hub/models--mlx-community--gemma-4-E2B-it-4bit/snapshots/<rev>
//	go test -tags metal_runtime -run '^$' -bench '^BenchmarkRealE2BContract' -benchmem \
//	        -benchtime=5x -memprofile=/tmp/e2b-contract.alloc ./pkg/native

const (
	// A fixed, deterministic prompt of valid E2B token ids — small, spread across the vocab. Greedy
	// makes the generated ids reproducible (TestRealE2BContractDeterministic pins it).
	contractPromptLen = 16
	contractMaxNew    = 32 // short decode per op (one fresh-cache pass), well under the OOM-prone sweep
	contractTokens    = contractPromptLen + contractMaxNew
	contractBenchMax  = 5 // -benchtime=5x ceiling (OOM guard)
	// Each op opens a FRESH session at position 0, so the cache only ever holds one pass — maxLen need
	// cover a single prompt+decode, with headroom. (Unlike the reused-session benches.)
	contractMaxLen = 2 * contractTokens
)

// contractPrompt is a fixed prompt of valid E2B token ids — distinct stride from the decode/prefill
// fixtures so the benches don't alias one another.
func contractPrompt() []int32 {
	p := make([]int32, contractPromptLen)
	for i := range p {
		p[i] = int32(5 + i*97) // small ids, comfortably within the 256k vocab
	}
	return p
}

// loadContractTokenModel loads the real E2B-4bit checkpoint as a model.TokenModel — the contract
// surface model.Generate drives. Shared by the bench and its determinism precondition.
func loadContractTokenModel(tb testing.TB, dir string) model.TokenModel {
	tb.Helper()
	tm, err := LoadTokenModelDir(dir, contractMaxLen)
	if err != nil {
		tb.Fatalf("LoadTokenModelDir(%s): %v", dir, err)
	}
	return tm
}

// BenchmarkRealE2BContractOpenSession isolates the fresh-session setup paid by
// the contract path before the first token is prefetched. It keeps the E2B
// checkpoint loaded once, then opens/closes contract sessions inside the timer.
func BenchmarkRealE2BContractOpenSession(b *testing.B) {
	requireNativeRuntime(b)
	dir := realE2BDir()
	if dir == "" {
		b.Skip("set E2B_Q4_DIR to the gemma-4-e2b-it-4bit snapshot dir (opt-in real-model bench)")
	}
	defer debug.SetMemoryLimit(debug.SetMemoryLimit(60 << 30))
	if b.N > contractBenchMax {
		b.Skipf("real-e2b contract session bench is capped at -benchtime=%dx (OOM guard); got b.N=%d", contractBenchMax, b.N)
	}

	tm := loadContractTokenModel(b, dir)
	if c, ok := tm.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	sm, ok := tm.(model.SessionModel)
	if !ok {
		b.Fatalf("LoadTokenModelDir(%s) = %T, want model.SessionModel", dir, tm)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess, err := sm.OpenSession()
		if err != nil {
			b.Fatalf("OpenSession (op %d): %v", i, err)
		}
		if c, ok := sess.(interface{ Close() error }); ok {
			if err := c.Close(); err != nil {
				b.Fatalf("Close session (op %d): %v", i, err)
			}
		}
	}
}

// BenchmarkRealE2BContractGreedy measures the heap allocations of the GREEDY contract loop
// (model.Generate) over a real E2B-4bit TokenModel — the deterministic serve path (Temperature<=0
// in register_native.go falls to model.Generate). Greedy output is reproducible, so this is the
// byte-identity baseline an alloc fix is validated against. allocs/token = allocs/op ÷
// contractTokens.
func BenchmarkRealE2BContractGreedy(b *testing.B) {
	requireNativeRuntime(b)
	dir := realE2BDir()
	if dir == "" {
		b.Skip("set E2B_Q4_DIR to the gemma-4-e2b-it-4bit snapshot dir (opt-in real-model bench)")
	}
	defer debug.SetMemoryLimit(debug.SetMemoryLimit(60 << 30)) // GC backstop; restore prior on exit
	if b.N > contractBenchMax {
		b.Skipf("real-e2b contract bench is capped at -benchtime=%dx (OOM guard); got b.N=%d", contractBenchMax, b.N)
	}

	tm := loadContractTokenModel(b, dir)
	if c, ok := tm.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	prompt := contractPrompt()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gen, err := model.Generate(tm, prompt, contractMaxNew, -1) // greedy, no early-EOS — a full maxNew pass
		if err != nil {
			b.Fatalf("model.Generate (op %d): %v", i, err)
		}
		if len(gen) != contractMaxNew {
			b.Fatalf("op %d: generated %d tokens, want %d", i, len(gen), contractMaxNew)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(contractTokens), "tokens/op")
}

// BenchmarkRealE2BContractSampled measures the SAMPLED contract loop (model.GenerateSampled) over
// the same real TokenModel — the stochastic serve path (Temperature>0 in register_native.go). The
// sampler is constructed once and reused (its RNG advances per draw, matching the serve shape); a
// fixed seed keeps the token COUNT invariant op-to-op (the ids legitimately vary with the RNG, so
// this is a count/alloc bench — byte-identity of the deterministic loop is pinned on the greedy
// bench above). It exercises the GenerateSampled pick-closure allocation on top of the decode path.
func BenchmarkRealE2BContractSampled(b *testing.B) {
	requireNativeRuntime(b)
	dir := realE2BDir()
	if dir == "" {
		b.Skip("set E2B_Q4_DIR to the gemma-4-e2b-it-4bit snapshot dir (opt-in real-model bench)")
	}
	defer debug.SetMemoryLimit(debug.SetMemoryLimit(60 << 30))
	if b.N > contractBenchMax {
		b.Skipf("real-e2b contract bench is capped at -benchtime=%dx (OOM guard); got b.N=%d", contractBenchMax, b.N)
	}

	tm := loadContractTokenModel(b, dir)
	if c, ok := tm.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	prompt := contractPrompt()
	sampler := model.NewSampler(0x5eed)                                  // fixed seed → invariant token count per op
	params := model.SampleParams{Temperature: 0.8, TopK: 40, TopP: 0.95} // a representative serve config
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gen, err := model.GenerateSampled(tm, sampler, params, prompt, contractMaxNew, -1)
		if err != nil {
			b.Fatalf("model.GenerateSampled (op %d): %v", i, err)
		}
		if len(gen) != contractMaxNew {
			b.Fatalf("op %d: generated %d tokens, want %d", i, len(gen), contractMaxNew)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(contractTokens), "tokens/op")
}

// TestRealE2BContractDeterministic is the byte-identity precondition for the alloc-reduction work
// the greedy contract bench feeds: model.Generate (greedy) over a real E2B TokenModel must produce
// the SAME token ids on two independent fresh token models. Any contract-layer alloc fix is
// validated by re-running and confirming these ids are unchanged; that check is only meaningful if
// they are deterministic to begin with. Opt-in (E2B_Q4_DIR), short maxLen, no sweep.
func TestRealE2BContractDeterministic(t *testing.T) {
	requireNativeRuntime(t)
	dir := realE2BDir()
	if dir == "" {
		t.Skip("set E2B_Q4_DIR to the gemma-4-e2b-it-4bit snapshot dir (opt-in real-model test)")
	}
	defer debug.SetMemoryLimit(debug.SetMemoryLimit(60 << 30))
	prompt := contractPrompt()

	gen := func() []int32 {
		tm := loadContractTokenModel(t, dir)
		if c, ok := tm.(interface{ Close() error }); ok {
			defer func() { _ = c.Close() }()
		}
		out, err := model.Generate(tm, prompt, contractMaxNew, -1)
		if err != nil {
			t.Fatalf("model.Generate: %v", err)
		}
		return out
	}

	a, c := gen(), gen()
	if !idsEqual(a, c) {
		t.Fatalf("greedy contract decode not deterministic across fresh token models:\n  run1 = %v\n  run2 = %v", a, c)
	}
	for _, id := range a {
		if id < 0 {
			t.Fatalf("negative token id in greedy contract decode: %v", a)
		}
	}
	t.Logf("real-e2b greedy CONTRACT decode deterministic over %d tokens: %v", len(a), a)
}
