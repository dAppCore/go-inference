// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// reuseFakeSession is a fakeSession that additionally implements
// PromptReuseSession, recording every cached-prefill call.
type reuseFakeSession struct {
	fakeSession
	cachedCalls [][]int32
	cachedErr   error
	// block, when non-nil, parks PrefillTokensCached until the channel closes —
	// the busy-slot concurrency gate holds the resident slot open with it.
	block   chan struct{}
	entered chan struct{}
}

func (f *reuseFakeSession) PrefillTokensCached(ids []int32) (int, error) {
	f.cachedCalls = append(f.cachedCalls, append([]int32(nil), ids...))
	if f.entered != nil {
		close(f.entered)
		f.entered = nil
	}
	if f.block != nil {
		<-f.block
	}
	if f.cachedErr != nil {
		return 0, f.cachedErr
	}
	reused := f.pos
	if reused > len(ids) {
		reused = len(ids)
	}
	f.pos = len(ids)
	return reused, nil
}

// reuseFakeTokenModel is a TokenModel double whose sessions declare prompt
// reuse: the first OpenEngineSession calls serve the pinned reuse sessions in
// order, later calls fall back to plain fakeSessions (the fresh path).
type reuseFakeTokenModel struct {
	genIDs   []int32
	sessions []*reuseFakeSession
	plain    []*fakeSession

	openCalls int
}

func (f *reuseFakeTokenModel) OpenEngineSession() (Session, error) {
	f.openCalls++
	if i := f.openCalls - 1; i < len(f.sessions) {
		return f.sessions[i], nil
	}
	s := &fakeSession{genIDs: f.genIDs}
	f.plain = append(f.plain, s)
	return s, nil
}

func (f *reuseFakeTokenModel) Close() error { return nil }

func (f *reuseFakeTokenModel) SessionsReusePrompts() bool { return true }

var (
	_ TokenModel              = (*reuseFakeTokenModel)(nil)
	_ PromptReuseCapableModel = (*reuseFakeTokenModel)(nil)
	_ PromptReuseSession      = (*reuseFakeSession)(nil)
)

func drainGenerate(t *testing.T, m *TextModel, prompt string) []inference.Token {
	t.Helper()
	var got []inference.Token
	for tok := range m.Generate(context.Background(), prompt, inference.WithMaxTokens(3)) {
		got = append(got, tok)
	}
	return got
}

// --- acquireReuseSession / stream reuse lane -------------------------------

// TestModel_TextModel_PromptReuse_Good pins the resident-session lane: two
// stateless Generates share ONE session, both prefills go through
// PrefillTokensCached, and the session is never closed between requests.
func TestModel_TextModel_PromptReuse_Good(t *testing.T) {
	rs := &reuseFakeSession{fakeSession: fakeSession{genIDs: []int32{10, 11, 12}}}
	tm := &reuseFakeTokenModel{genIDs: []int32{10, 11, 12}, sessions: []*reuseFakeSession{rs}}
	m := NewTextModel(tm, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)

	if got := drainGenerate(t, m, "hi"); len(got) != 3 {
		t.Fatalf("first Generate produced %d tokens, want 3", len(got))
	}
	if got := drainGenerate(t, m, "hi there again"); len(got) != 3 {
		t.Fatalf("second Generate produced %d tokens, want 3", len(got))
	}
	if tm.openCalls != 1 {
		t.Fatalf("openCalls = %d, want 1 (one resident session across requests)", tm.openCalls)
	}
	if len(rs.cachedCalls) != 2 {
		t.Fatalf("PrefillTokensCached calls = %d, want 2", len(rs.cachedCalls))
	}
	if len(rs.prefillCalls) != 0 {
		t.Fatalf("PrefillTokens calls = %d, want 0 (reuse lane owns the prefill)", len(rs.prefillCalls))
	}
	if rs.closeCalls != 0 {
		t.Fatalf("resident session closeCalls = %d, want 0 between requests", rs.closeCalls)
	}
	if r := m.Err(); !r.OK {
		t.Fatalf("Err() after reuse-lane generations = %+v, want OK", r)
	}
}

// TestModel_TextModel_PromptReuse_Bad pins the poisoned-resident escape hatch:
// a failing cached prefill drops (closes) the resident session and the request
// still completes through a fresh session's plain PrefillTokens.
func TestModel_TextModel_PromptReuse_Bad(t *testing.T) {
	rs := &reuseFakeSession{cachedErr: core.NewError("cached prefill boom"), fakeSession: fakeSession{genIDs: []int32{10}}}
	tm := &reuseFakeTokenModel{genIDs: []int32{10, 11, 12}, sessions: []*reuseFakeSession{rs}}
	m := NewTextModel(tm, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)

	if got := drainGenerate(t, m, "hi"); len(got) != 3 {
		t.Fatalf("Generate after a failed cached prefill produced %d tokens, want 3", len(got))
	}
	if rs.closeCalls != 1 {
		t.Fatalf("poisoned resident closeCalls = %d, want 1", rs.closeCalls)
	}
	if tm.openCalls != 2 {
		t.Fatalf("openCalls = %d, want 2 (resident + fresh fallback)", tm.openCalls)
	}
	if len(tm.plain) != 1 || len(tm.plain[0].prefillCalls) != 1 {
		t.Fatal("fresh fallback session did not take the plain PrefillTokens path")
	}
	if r := m.Err(); !r.OK {
		t.Fatalf("Err() after the fresh fallback = %+v, want OK", r)
	}
}

// TestModel_TextModel_PromptReuse_Ugly pins the stand-down guards: with a
// conversation-continuity interceptor installed the lane never engages, and
// with the kill switch off it never engages — every request opens and closes
// its own fresh session, exactly the pre-reuse behaviour.
func TestModel_TextModel_PromptReuse_Ugly(t *testing.T) {
	t.Run("continuity interceptor installed", func(t *testing.T) {
		rs := &reuseFakeSession{}
		tm := &reuseFakeTokenModel{genIDs: []int32{10, 11, 12}, sessions: []*reuseFakeSession{rs}}
		m := NewTextModel(tm, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
		m.SetChatInterceptor(func(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) (iter.Seq[inference.Token], bool) {
			return nil, false
		})
		_ = drainGenerate(t, m, "hi")
		if len(rs.cachedCalls) != 0 {
			t.Fatal("reuse lane engaged under an installed continuity interceptor")
		}
	})
	t.Run("kill switch off", func(t *testing.T) {
		old := promptReuseEnabled
		promptReuseEnabled = false
		t.Cleanup(func() { promptReuseEnabled = old })
		rs := &reuseFakeSession{fakeSession: fakeSession{genIDs: []int32{10, 11, 12}}}
		tm := &reuseFakeTokenModel{genIDs: []int32{10, 11, 12}, sessions: []*reuseFakeSession{rs}}
		m := NewTextModel(tm, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
		if got := drainGenerate(t, m, "hi"); len(got) != 3 {
			t.Fatalf("kill-switch Generate produced %d tokens, want 3", len(got))
		}
		if len(rs.cachedCalls) != 0 {
			t.Fatal("reuse lane engaged with LTHN_PROMPT_REUSE=0")
		}
		if len(rs.prefillCalls) != 1 || rs.closeCalls != 1 {
			t.Fatalf("kill-switch request: prefillCalls=%d closeCalls=%d, want the plain open/prefill/close shape", len(rs.prefillCalls), rs.closeCalls)
		}
	})
}

// TestModel_TextModel_PromptReuse_DisablePromptReuseSkipsLane_Good pins #54:
// GenerateConfig.DisablePromptReuse is a THIRD stand-down (distinct from the
// process-wide kill switch and the continuity interceptor above) scoped to
// the calls that set it — a one-shot bench/CLI caller that knows its calls
// never share a prefix opts out per-request rather than for the whole
// process. Every such request opens and closes its own fresh session
// (stream() never calls acquireReuseSession), the pinned resident-lane
// session is used only as a plain fresh session on the first call, and the
// second call falls through to its own independent fresh session exactly as
// the kill-switch-off case does.
func TestModel_TextModel_PromptReuse_DisablePromptReuseSkipsLane_Good(t *testing.T) {
	rs := &reuseFakeSession{fakeSession: fakeSession{genIDs: []int32{10, 11, 12}}}
	tm := &reuseFakeTokenModel{genIDs: []int32{10, 11, 12}, sessions: []*reuseFakeSession{rs}}
	m := NewTextModel(tm, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)

	drain := func(prompt string) []inference.Token {
		var got []inference.Token
		for tok := range m.Generate(context.Background(), prompt, inference.WithMaxTokens(3), inference.WithDisablePromptReuse()) {
			got = append(got, tok)
		}
		return got
	}

	if got := drain("hi"); len(got) != 3 {
		t.Fatalf("first Generate produced %d tokens, want 3", len(got))
	}
	if got := drain("hi there again"); len(got) != 3 {
		t.Fatalf("second Generate produced %d tokens, want 3", len(got))
	}
	if len(rs.cachedCalls) != 0 {
		t.Fatalf("PrefillTokensCached calls = %d, want 0 (DisablePromptReuse must skip the reuse lane)", len(rs.cachedCalls))
	}
	if tm.openCalls != 2 {
		t.Fatalf("openCalls = %d, want 2 (a fresh session per request, no resident reuse)", tm.openCalls)
	}
	if len(rs.prefillCalls) != 1 || rs.closeCalls != 1 {
		t.Fatalf("pinned session: prefillCalls=%d closeCalls=%d, want 1/1 (opened, prefilled and closed once, as the first request's fresh session)", len(rs.prefillCalls), rs.closeCalls)
	}
	if len(tm.plain) != 1 || len(tm.plain[0].prefillCalls) != 1 {
		t.Fatal("second request did not open its own independent fresh session via the plain path")
	}
	if r := m.Err(); !r.OK {
		t.Fatalf("Err() after DisablePromptReuse generations = %+v, want OK", r)
	}
}

// TestModel_TextModel_PromptReuse_BusySlotServesFresh pins the TryLock
// semantics: while one request holds the resident slot, a concurrent request
// takes the fresh-session path instead of queueing behind the cache.
func TestModel_TextModel_PromptReuse_BusySlotServesFresh(t *testing.T) {
	rs := &reuseFakeSession{
		fakeSession: fakeSession{genIDs: []int32{10}},
		block:       make(chan struct{}),
		entered:     make(chan struct{}),
	}
	tm := &reuseFakeTokenModel{genIDs: []int32{10, 11, 12}, sessions: []*reuseFakeSession{rs}}
	m := NewTextModel(tm, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)

	entered := rs.entered
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range m.Generate(context.Background(), "hold the slot", inference.WithMaxTokens(1)) {
		}
	}()
	<-entered // the goroutine now owns the resident slot, parked in PrefillTokensCached

	if got := drainGenerate(t, m, "concurrent request"); len(got) != 3 {
		t.Fatalf("concurrent Generate produced %d tokens, want 3", len(got))
	}
	if len(tm.plain) != 1 {
		t.Fatalf("concurrent request opened %d fresh sessions, want 1", len(tm.plain))
	}
	close(rs.block)
	<-done
	if len(rs.cachedCalls) != 1 {
		t.Fatalf("resident PrefillTokensCached calls = %d, want 1", len(rs.cachedCalls))
	}
}

// --- Close ------------------------------------------------------------------

// TestModel_TextModel_PromptReuse_CloseReleasesResident pins the lifecycle:
// closing the model closes the resident reuse session with it.
func TestModel_TextModel_PromptReuse_CloseReleasesResident(t *testing.T) {
	rs := &reuseFakeSession{fakeSession: fakeSession{genIDs: []int32{10}}}
	tm := &reuseFakeTokenModel{genIDs: []int32{10}, sessions: []*reuseFakeSession{rs}}
	m := NewTextModel(tm, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)

	_ = drainGenerate(t, m, "hi")
	if r := m.Close(); !r.OK {
		t.Fatalf("Close() = %+v, want OK", r)
	}
	if rs.closeCalls != 1 {
		t.Fatalf("resident session closeCalls after model Close = %d, want 1", rs.closeCalls)
	}
}
