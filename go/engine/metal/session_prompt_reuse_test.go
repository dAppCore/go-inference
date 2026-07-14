// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

// reuseTestModel builds the synthetic 3-layer bf16 fixture the prompt-cache
// tests use, with the layer types supplied (full_attention everywhere for the
// always-safe cases; a sliding mix + window for the ring-safety gates).
func reuseTestModel(t *testing.T, types []string, slidingWindow, maxLen int) (*BF16Model, model.Arch, int) {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const vocab = 64
	layers := make([]DecodeLayerWeights, len(types))
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV, SlidingWindow: slidingWindow,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	return g, arch, maxLen
}

func reuseSession(t *testing.T, g *BF16Model, arch model.Arch, maxLen int) *ArchSession {
	t.Helper()
	s, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	return s
}

func reuseColdDecode(t *testing.T, g *BF16Model, arch model.Arch, maxLen int, ids []int32, maxNew int) []int32 {
	t.Helper()
	cold := reuseSession(t, g, arch, maxLen)
	defer cold.Close()
	if err := cold.PrefillTokens(ids); err != nil {
		t.Fatalf("cold PrefillTokens: %v", err)
	}
	gen, err := cold.GenerateFromCache(maxNew, -1)
	if err != nil {
		t.Fatalf("cold GenerateFromCache: %v", err)
	}
	return gen
}

// --- PrefillTokensCached -----------------------------------------------------

// TestPrefillTokensCached_Good pins the serve chat shape: prefill a prompt,
// generate a reply (the generated ids join the resident cache), then prefill
// the grown conversation — the whole resident run is reused (pure append, no
// rollback) and the continuation decodes token-identically to a cold session
// given the same full prompt.
func TestPrefillTokensCached_Good(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := reuseTestModel(t, []string{"full_attention", "full_attention", "full_attention"}, 0, 96)

	warm := reuseSession(t, g, arch, maxLen)
	defer warm.Close()
	turn1 := []int32{1, 2, 3, 4, 5}
	reused, err := warm.PrefillTokensCached(turn1)
	if err != nil {
		t.Fatalf("turn 1 PrefillTokensCached: %v", err)
	}
	if reused != 0 {
		t.Fatalf("turn 1 reused = %d, want 0 (cold session)", reused)
	}
	reply, err := warm.GenerateFromCache(4, -1)
	if err != nil {
		t.Fatalf("turn 1 GenerateFromCache: %v", err)
	}

	// turn 2 resends the whole conversation: prompt + the model's own reply +
	// the new user turn — exactly what a stateless chat client sends.
	turn2 := append(append(append([]int32(nil), turn1...), reply...), 6, 7, 8)
	reused, err = warm.PrefillTokensCached(turn2)
	if err != nil {
		t.Fatalf("turn 2 PrefillTokensCached: %v", err)
	}
	if want := len(turn1) + len(reply); reused != want {
		t.Fatalf("turn 2 reused = %d, want %d (prompt + generated reply)", reused, want)
	}
	got, err := warm.GenerateFromCache(8, -1)
	if err != nil {
		t.Fatalf("turn 2 GenerateFromCache: %v", err)
	}

	cold := reuseColdDecode(t, g, arch, maxLen, turn2, 8)
	if len(got) != len(cold) {
		t.Fatalf("length mismatch: warm=%d cold=%d", len(got), len(cold))
	}
	for i := range cold {
		if got[i] != cold[i] {
			t.Fatalf("token %d diverged: warm(reuse)=%d cold=%d", i, got[i], cold[i])
		}
	}
	t.Logf("prompt reuse: %d rows reused across the turn boundary, continuation token-identical over %d tokens", reused, len(got))
}

// TestPrefillTokensCached_DivergenceRollback_Good pins the rollback branch on
// an all-global model (rollback always ring-safe): a prompt diverging inside
// the resident run rolls back to the shared prefix, re-prefills the suffix,
// and decodes token-identically to a cold session.
func TestPrefillTokensCached_DivergenceRollback_Good(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := reuseTestModel(t, []string{"full_attention", "full_attention", "full_attention"}, 0, 96)

	warm := reuseSession(t, g, arch, maxLen)
	defer warm.Close()
	if _, err := warm.PrefillTokensCached([]int32{1, 2, 3, 4, 5, 6, 7}); err != nil {
		t.Fatalf("first PrefillTokensCached: %v", err)
	}
	branch := []int32{1, 2, 3, 9, 10, 11}
	reused, err := warm.PrefillTokensCached(branch)
	if err != nil {
		t.Fatalf("branch PrefillTokensCached: %v", err)
	}
	if reused != 3 {
		t.Fatalf("branch reused = %d, want 3 (the shared prefix)", reused)
	}
	got, err := warm.GenerateFromCache(8, -1)
	if err != nil {
		t.Fatalf("branch GenerateFromCache: %v", err)
	}
	cold := reuseColdDecode(t, g, arch, maxLen, branch, 8)
	for i := range cold {
		if got[i] != cold[i] {
			t.Fatalf("token %d diverged after rollback: warm=%d cold=%d", i, got[i], cold[i])
		}
	}
}

// TestPrefillTokensCached_RingSafety_Ugly pins the sliding-ring rule both
// ways: with the resident run past the window (rings wrapped) a divergent
// prompt takes the COLD path (reused 0) — never a rollback over clobbered
// rows; with the run still inside the window the rollback is taken. Both
// decode token-identically to cold sessions.
func TestPrefillTokensCached_RingSafety_Ugly(t *testing.T) {
	requireNativeRuntime(t)
	const window = 8
	types := []string{"sliding_attention", "full_attention", "sliding_attention"}
	g, arch, maxLen := reuseTestModel(t, types, window, 96)

	t.Run("wrapped ring forces cold", func(t *testing.T) {
		warm := reuseSession(t, g, arch, maxLen)
		defer warm.Close()
		long := make([]int32, 20) // pos 20 > window 8: rings have wrapped
		for i := range long {
			long[i] = int32(i%13 + 1)
		}
		if _, err := warm.PrefillTokensCached(long); err != nil {
			t.Fatalf("long PrefillTokensCached: %v", err)
		}
		branch := append(append([]int32(nil), long[:10]...), 40, 41, 42)
		reused, err := warm.PrefillTokensCached(branch)
		if err != nil {
			t.Fatalf("branch PrefillTokensCached: %v", err)
		}
		if reused != 0 {
			t.Fatalf("reused = %d past a wrapped ring, want 0 (cold path)", reused)
		}
		got, err := warm.GenerateFromCache(6, -1)
		if err != nil {
			t.Fatalf("branch GenerateFromCache: %v", err)
		}
		cold := reuseColdDecode(t, g, arch, maxLen, branch, 6)
		for i := range cold {
			if got[i] != cold[i] {
				t.Fatalf("token %d diverged on the forced-cold path: warm=%d cold=%d", i, got[i], cold[i])
			}
		}
	})

	t.Run("inside the window rolls back", func(t *testing.T) {
		warm := reuseSession(t, g, arch, maxLen)
		defer warm.Close()
		if _, err := warm.PrefillTokensCached([]int32{1, 2, 3, 4, 5}); err != nil { // pos 5 <= window 8
			t.Fatalf("short PrefillTokensCached: %v", err)
		}
		branch := []int32{1, 2, 3, 9, 10}
		reused, err := warm.PrefillTokensCached(branch)
		if err != nil {
			t.Fatalf("branch PrefillTokensCached: %v", err)
		}
		if reused != 3 {
			t.Fatalf("reused = %d inside the window, want 3", reused)
		}
		got, err := warm.GenerateFromCache(6, -1)
		if err != nil {
			t.Fatalf("branch GenerateFromCache: %v", err)
		}
		cold := reuseColdDecode(t, g, arch, maxLen, branch, 6)
		for i := range cold {
			if got[i] != cold[i] {
				t.Fatalf("token %d diverged after an in-window rollback: warm=%d cold=%d", i, got[i], cold[i])
			}
		}
	})
}

// TestPrefillTokensCached_Bad pins the input guards: empty ids and a prompt
// beyond maxLen fail clean without disturbing the resident state.
func TestPrefillTokensCached_Bad(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := reuseTestModel(t, []string{"full_attention", "full_attention", "full_attention"}, 0, 16)
	s := reuseSession(t, g, arch, maxLen)
	defer s.Close()
	if _, err := s.PrefillTokensCached(nil); err == nil {
		t.Fatal("want an error for empty prompt tokens")
	}
	over := make([]int32, maxLen+1)
	for i := range over {
		over[i] = int32(i%7 + 1)
	}
	if _, err := s.PrefillTokensCached(over); err == nil {
		t.Fatal("want an error for a prompt beyond maxLen")
	}
}

// The q8-ICB reuse contract tests (TestPrefillTokensCached_Q8Declines_Ugly and
// the canonical-landing engagement case) live in session_prompt_reuse_q8_test.go
// — they need the metal_runtime q8 KV fixture, which is unavailable in the
// portable build this file also compiles under.
