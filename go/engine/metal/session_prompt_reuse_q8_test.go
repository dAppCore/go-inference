// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import "testing"

// TestPrefillTokensCached_Q8Declines_Ugly pins the NARROWED #1846 decline: a
// q8-ICB session WITHOUT canonical landing still refuses the cached path —
// the batched landing forward is intra-batch tile-position sensitive and the
// q8 store amplifies that wobble into token flips, so the reuse contract is
// only honest as a cold prefill there. reused must be 0 on the turn-2 shape the
// Good case reuses, and the decode must equal a cold session's.
func TestPrefillTokensCached_Q8Declines_Ugly(t *testing.T) {
	requireNativeRuntime(t)
	kvQ8ICBForTest = true
	t.Cleanup(func() { kvQ8ICBForTest = false })

	warm := newKVQ8ICBFixtureLen(t, 256)
	defer warm.Close()
	if !warm.state.icb.hasKVQ8() {
		t.Fatal("q8 fixture did not arm the q8 KV store — the decline was never exercised")
	}
	if warm.reuseCanonicalLanding {
		t.Fatal("fixture must default to non-canonical landing for the decline case")
	}
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
	turn2 := append(append(append([]int32(nil), turn1...), reply...), 6, 7, 8)
	reused, err = warm.PrefillTokensCached(turn2)
	if err != nil {
		t.Fatalf("turn 2 PrefillTokensCached: %v", err)
	}
	if reused != 0 {
		t.Fatalf("turn 2 reused = %d on a non-canonical q8-ICB session, want 0 (declined)", reused)
	}
	got, err := warm.GenerateFromCache(8, -1)
	if err != nil {
		t.Fatalf("turn 2 GenerateFromCache: %v", err)
	}
	cold := newKVQ8ICBFixtureLen(t, 256)
	defer cold.Close()
	if err := cold.PrefillTokens(turn2); err != nil {
		t.Fatalf("cold PrefillTokens: %v", err)
	}
	want, err := cold.GenerateFromCache(8, -1)
	if err != nil {
		t.Fatalf("cold GenerateFromCache: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d diverged on the declined path: warm=%d cold=%d", i, got[i], want[i])
		}
	}
}

// TestPrefillTokensCached_Q8Canonical_Good pins the #1846 fix: a q8-ICB session
// with reuseCanonicalLanding ENGAGES the cached path — every prefill/append
// lands through the position-invariant per-token lane, so the reused landing is
// byte-identical to a fresh whole prefill. reused must be > 0 on the turn-2
// shape, and the continuation must be token-identical to a canonical cold
// session given the same full prompt.
func TestPrefillTokensCached_Q8Canonical_Good(t *testing.T) {
	requireNativeRuntime(t)
	kvQ8ICBForTest = true
	t.Cleanup(func() { kvQ8ICBForTest = false })

	warm := newKVQ8ICBFixtureLen(t, 256)
	warm.reuseCanonicalLanding = true
	defer warm.Close()
	if !warm.state.icb.hasKVQ8() {
		t.Fatal("q8 fixture did not arm the q8 KV store")
	}
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
	turn2 := append(append(append([]int32(nil), turn1...), reply...), 6, 7, 8)
	reused, err = warm.PrefillTokensCached(turn2)
	if err != nil {
		t.Fatalf("turn 2 PrefillTokensCached: %v", err)
	}
	if want := len(turn1) + len(reply); reused != want {
		t.Fatalf("turn 2 reused = %d on a canonical q8-ICB session, want %d (engaged)", reused, want)
	}
	got, err := warm.GenerateFromCache(8, -1)
	if err != nil {
		t.Fatalf("turn 2 GenerateFromCache: %v", err)
	}
	cold := newKVQ8ICBFixtureLen(t, 256)
	cold.reuseCanonicalLanding = true
	defer cold.Close()
	if err := cold.PrefillTokens(turn2); err != nil {
		t.Fatalf("cold PrefillTokens: %v", err)
	}
	want, err := cold.GenerateFromCache(8, -1)
	if err != nil {
		t.Fatalf("cold GenerateFromCache: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d diverged on the engaged canonical path: warm(reuse)=%d cold=%d", i, got[i], want[i])
		}
	}
}

// TestPrefillRetainedTokensBatchedDenseNonCanonicalQ8Engages_Good pins the
// #54 fix's precondition at the batched-dense gate itself (arch_session.go
// prefillRetainedTokensBatchedDense): a q8-ICB session that never had
// SetReuseCanonicalLanding armed — the shape ANY caller gets when it never
// goes through the resident prompt-reuse lane (engine/prompt_reuse.go),
// which is what inference.WithDisablePromptReuse now buys a one-shot
// caller — takes the FAST batched-dense prefill lane above the fold
// threshold. Before #54, the ONLY way to reach a q8-ICB session was through
// that lane, which armed canonical landing unconditionally at session
// creation (#1846) — so this shape (q8 + never-armed) was reachable in code
// but never actually exercised end to end.
func TestPrefillRetainedTokensBatchedDenseNonCanonicalQ8Engages_Good(t *testing.T) {
	requireNativeRuntime(t)
	kvQ8ICBForTest = true
	t.Cleanup(func() { kvQ8ICBForTest = false })

	sess := newKVQ8ICBFixtureLen(t, 256)
	if !sess.state.icb.hasKVQ8() {
		t.Fatal("q8 fixture did not arm the q8 KV store — the gate was never exercised")
	}
	if sess.reuseCanonicalLanding {
		t.Fatal("fixture must default to non-canonical landing (the DisablePromptReuse shape)")
	}
	ids := make([]int32, batchedDenseICBMaxRows+8) // above the fold threshold
	for i := range ids {
		ids[i] = int32(i % 64)
	}
	hidden, ok, err := sess.prefillRetainedTokensBatchedDense(ids, "test")
	if err != nil {
		t.Fatalf("prefillRetainedTokensBatchedDense: %v", err)
	}
	if !ok {
		t.Fatal("batched-dense prefill declined on a non-canonical q8-ICB session above the fold threshold — the #54 fast path regressed")
	}
	if len(hidden) == 0 {
		t.Fatal("batched-dense prefill returned no boundary hidden")
	}
}

// TestPrefillRetainedTokensBatchedDenseCanonicalQ8Declines_Ugly is the
// counterpart pinning #1846 staying intact: arming SetReuseCanonicalLanding
// — what the resident prompt-reuse lane still does for every caller that
// does NOT set DisablePromptReuse — makes the SAME session decline the
// batched-dense lane. #54 only skips the resident lane (and so this gate)
// for callers that opt out of reuse; it does not touch the gate itself.
func TestPrefillRetainedTokensBatchedDenseCanonicalQ8Declines_Ugly(t *testing.T) {
	requireNativeRuntime(t)
	kvQ8ICBForTest = true
	t.Cleanup(func() { kvQ8ICBForTest = false })

	sess := newKVQ8ICBFixtureLen(t, 256)
	sess.SetReuseCanonicalLanding(true)
	ids := make([]int32, batchedDenseICBMaxRows+8)
	for i := range ids {
		ids[i] = int32(i % 64)
	}
	if _, ok, err := sess.prefillRetainedTokensBatchedDense(ids, "test"); err != nil {
		t.Fatalf("prefillRetainedTokensBatchedDense: %v", err)
	} else if ok {
		t.Fatal("batched-dense prefill engaged on a canonical-landing q8-ICB session — #1846 regressed")
	}
}
