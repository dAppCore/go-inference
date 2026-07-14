// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/enginetest"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model"
)

// stateRestorer mirrors model/state/session.nativeSessionRestorer — the durable
// conversation session's unexported wake seam, which is RestoreKV(ctx, snapshot)
// rather than the inference.KVRestorer name RestoreFromKV. Pinning that
// *SessionHandle satisfies this exact shape guards the generate -state wake path:
// WakeAgentMemory type-asserts its handle to this interface, so a rename or
// removal of the alias would silently reintroduce the #291 wake failure
// ("native model session does not support KV restore").
type stateRestorer interface {
	RestoreKV(context.Context, *kv.Snapshot) error
}

var _ stateRestorer = (*SessionHandle)(nil)

type contextAwareFakeSession struct {
	fakeSession
	greedyCtx  context.Context
	sampledCtx context.Context
}

type multiBlockFakeSession struct{ fakeSession }

func (s *multiBlockFakeSession) RangeKVBlocks(_ int, _ kv.CaptureOptions, yield func(kv.Block) (bool, error)) error {
	for index := 0; index < 2; index++ {
		cont, err := yield(kv.Block{Index: index, TokenStart: index})
		if err != nil || !cont {
			return err
		}
	}
	return nil
}

func (s *contextAwareFakeSession) GenerateFromCacheEachContext(ctx context.Context, maxNew, eosID int, yield func(int32) bool) ([]int32, error) {
	s.greedyCtx = ctx
	return s.fakeSession.GenerateFromCacheEach(maxNew, eosID, yield)
}

func (s *contextAwareFakeSession) GenerateSampledFromCacheEachContext(ctx context.Context, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	s.sampledCtx = ctx
	return s.fakeSession.GenerateSampledFromCacheEach(maxNew, stopTokens, sampler, params, transform, yield)
}

func TestSession_ContextAwareDecode_Good(t *testing.T) {
	ctx := context.WithValue(context.Background(), struct{}{}, "decode")
	sess := &contextAwareFakeSession{fakeSession: fakeSession{genIDs: []int32{7}}}
	_, err := generateFromCacheEach(ctx, sess, 1, -1, func(int32) bool { return true })
	core.RequireNoError(t, err)
	core.AssertEqual(t, ctx, sess.greedyCtx)

	_, err = generateSampledFromCacheEach(ctx, sess, 1, nil, model.NewSampler(1), model.SampleParams{Temperature: 1}, nil, func(int32) bool { return true })
	core.RequireNoError(t, err)
	core.AssertEqual(t, ctx, sess.sampledCtx)
}

func TestSession_RangeKVBlocks_Cancellation_Good(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	sess := &multiBlockFakeSession{fakeSession: fakeSession{pos: 2}}
	handle := NewSessionHandle(newTestModel(t, &fakeTokenModel{}), sess)
	blocks := 0
	err := handle.RangeKVBlocks(ctx, 1, kv.CaptureOptions{}, func(kv.Block) (bool, error) {
		blocks++
		cancel()
		return true, nil
	})
	core.AssertErrorIs(t, err, context.Canceled)
	core.AssertEqual(t, 1, blocks)
}

// --- conformance -----------------------------------------------------------

// TestSession_SessionHandle_Conformance runs the shared enginetest.SessionHandle
// suite (the same suite metal/hip run against their real decode) against
// NewSessionHandle wrapping a synthetic Session — proving the lifecycle
// (Prefill, AppendPrompt, Generate, CaptureKV, Fork, Reset, RangeKVBlocks,
// Close, optional KV restore) holds for ANY conformant Session.
func TestSession_SessionHandle_Conformance(t *testing.T) {
	tok := newFixtureTokenizer(t)
	enginetest.SessionHandle(t, func(t *testing.T) inference.SessionHandle {
		m := NewTextModel(&fakeTokenModel{}, tok, "gemma-test", inference.ModelInfo{}, 4096)
		sess, err := m.openSession()
		if err != nil {
			t.Fatalf("openSession: %v", err)
		}
		return NewSessionHandle(m, sess)
	})
}

// --- NewSessionHandle --------------------------------------------------

// TestSession_NewSessionHandle_Good pins the plain constructor: the wrapped
// model + session are wired so Prefill can drive the engine straight away.
func TestSession_NewSessionHandle_Good(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, &fakeSession{})
	if err := handle.Prefill(context.Background(), "hi"); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	if err := handle.Err(); err != nil {
		t.Fatalf("Err() after a clean Prefill = %v, want nil", err)
	}
}

// TestSession_NewSessionHandle_Bad pins the nil-session edge: wrapping a nil
// [Session] fails the shared readiness guard on first use rather than
// panicking.
func TestSession_NewSessionHandle_Bad(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, nil)
	if err := handle.Prefill(context.Background(), "hi"); err == nil {
		t.Fatal("want an error wrapping a nil Session")
	}
}

// TestSession_NewSessionHandle_Ugly pins that the constructor never resets
// the session it is handed: a session with pre-existing retained state can
// Generate immediately, with no Prefill call in between.
func TestSession_NewSessionHandle_Ugly(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	sess := &fakeSession{pos: 3, genIDs: []int32{10}}
	handle := NewSessionHandle(m, sess)
	var got int
	for range handle.Generate(context.Background(), inference.GenerateConfig{MaxTokens: 1}) {
		got++
	}
	if got != 1 {
		t.Fatalf("Generate over a pre-populated session produced %d tokens, want 1", got)
	}
}

// --- Pos ---------------------------------------------------------------

// TestSession_SessionHandle_Pos_Good pins the retained-position accessor the
// continuity layer measures per-turn prefill with: it reports the wrapped
// session's position, and reflects a prefill's advance (the delta the
// no-replay per-turn token count is derived from).
func TestSession_SessionHandle_Pos_Good(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, &fakeSession{pos: 5})
	if got := handle.Pos(); got != 5 {
		t.Fatalf("Pos() = %d, want the wrapped session's retained 5", got)
	}
	before := handle.Pos()
	if err := handle.Prefill(context.Background(), "hi"); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	if handle.Pos() == before {
		t.Fatalf("Pos() = %d unchanged after Prefill, want the advanced position", handle.Pos())
	}
}

// TestSession_SessionHandle_Pos_Bad pins the nil-session guard: a handle
// wrapping a nil Session reports 0 rather than dereferencing nil.
func TestSession_SessionHandle_Pos_Bad(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	if got := NewSessionHandle(m, nil).Pos(); got != 0 {
		t.Fatalf("Pos() on a nil-session handle = %d, want 0", got)
	}
}

// TestSession_SessionHandle_Pos_Ugly pins the nil-receiver guard: a nil
// *SessionHandle reports 0, never a panic.
func TestSession_SessionHandle_Pos_Ugly(t *testing.T) {
	var handle *SessionHandle
	if got := handle.Pos(); got != 0 {
		t.Fatalf("Pos() on a nil *SessionHandle = %d, want 0", got)
	}
}

// --- Prefill -----------------------------------------------------------

// TestSession_SessionHandle_Prefill_Good pins the happy path: tokenising and
// storing the prompt's KV/logit state, replacing any prior retained state.
func TestSession_SessionHandle_Prefill_Good(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	sess := &fakeSession{}
	handle := NewSessionHandle(m, sess)
	if err := handle.Prefill(context.Background(), "hi"); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	if sess.pos == 0 {
		t.Fatal("Prefill did not store any retained state")
	}
}

// TestSession_SessionHandle_Prefill_Bad pins the not-ready guard: a handle
// wrapping a nil session refuses to prefill.
func TestSession_SessionHandle_Prefill_Bad(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, nil)
	if err := handle.Prefill(context.Background(), "hi"); err == nil {
		t.Fatal("want an error prefilling a not-ready handle")
	}
}

// TestSession_SessionHandle_Prefill_Ugly pins the nil-tokenizer edge: a model
// with no tokenizer attached fails clean with a named error rather than
// panicking on Encode.
func TestSession_SessionHandle_Prefill_Ugly(t *testing.T) {
	m := &TextModel{tm: &fakeTokenModel{}}
	handle := NewSessionHandle(m, &fakeSession{})
	err := handle.Prefill(context.Background(), "hi")
	if err == nil || !core.Contains(err.Error(), "tokenizer is nil") {
		t.Fatalf("Prefill err = %v, want the tokenizer-is-nil message", err)
	}
}

// TestSessionHandlePrefillEmptyTokens pins the empty-encode edge distinct from
// the nil-tokenizer case: a tokenizer that produces zero ids fails with its
// own named error.
func TestSessionHandlePrefillEmptyTokens(t *testing.T) {
	m := &TextModel{tm: &fakeTokenModel{}, tok: spTok()}
	handle := NewSessionHandle(m, &fakeSession{})
	err := handle.Prefill(context.Background(), "hello")
	if err == nil || !core.Contains(err.Error(), "empty prompt tokens") {
		t.Fatalf("Prefill err = %v, want the empty-prompt-tokens message", err)
	}
}

// --- AppendPrompt --------------------------------------------------------

// TestSession_SessionHandle_AppendPrompt_Good pins the extend-without-replay
// path: appending after a Prefill grows the retained state.
func TestSession_SessionHandle_AppendPrompt_Good(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	sess := &fakeSession{}
	handle := NewSessionHandle(m, sess)
	if err := handle.Prefill(context.Background(), "hi"); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	before := sess.pos
	if err := handle.AppendPrompt(context.Background(), "there"); err != nil {
		t.Fatalf("AppendPrompt: %v", err)
	}
	if sess.pos <= before {
		t.Fatalf("AppendPrompt did not grow retained state: before=%d after=%d", before, sess.pos)
	}
}

// TestSession_SessionHandle_AppendPrompt_Bad pins the no-retained-prefix
// guard: appending before any Prefill fails clean.
func TestSession_SessionHandle_AppendPrompt_Bad(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, &fakeSession{})
	err := handle.AppendPrompt(context.Background(), "there")
	if err == nil || !core.Contains(err.Error(), "no retained prefix") {
		t.Fatalf("AppendPrompt err = %v, want the no-retained-prefix message", err)
	}
}

// TestSession_SessionHandle_AppendPrompt_Ugly pins engine-error propagation:
// a failing AppendTokens surfaces through AppendPrompt untouched.
func TestSession_SessionHandle_AppendPrompt_Ugly(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	sess := &fakeSession{pos: 1, appendErr: core.NewError("device OOM")}
	handle := NewSessionHandle(m, sess)
	if err := handle.AppendPrompt(context.Background(), "there"); err == nil {
		t.Fatal("want the engine's AppendTokens failure to propagate")
	}
}

// --- Generate ------------------------------------------------------------

// TestSession_SessionHandle_Generate_Good pins the bounded streaming path
// over already-retained state.
func TestSession_SessionHandle_Generate_Good(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, &fakeSession{pos: 1, genIDs: []int32{10, 11, 12}})
	var got int
	for range handle.Generate(context.Background(), inference.GenerateConfig{MaxTokens: 3}) {
		got++
	}
	if got != 3 {
		t.Fatalf("Generate produced %d tokens, want 3", got)
	}
}

// TestSession_SessionHandle_Generate_Bad pins the not-ready guard: no retained
// prefill state (Pos()<=0) refuses to generate.
func TestSession_SessionHandle_Generate_Bad(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, &fakeSession{})
	for range handle.Generate(context.Background(), inference.GenerateConfig{MaxTokens: 3}) {
		t.Fatal("expected no tokens with no retained prefill state")
	}
	if err := handle.Err(); err == nil {
		t.Fatal("want an error generating with no retained prefill state")
	}
}

// TestSession_SessionHandle_Generate_Ugly pins the exhausted-context-window
// edge: a session already at the model's context limit has no room left.
func TestSession_SessionHandle_Generate_Ugly(t *testing.T) {
	m := NewTextModel(&fakeTokenModel{}, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 1)
	handle := NewSessionHandle(m, &fakeSession{pos: 1})
	for range handle.Generate(context.Background(), inference.GenerateConfig{MaxTokens: 4}) {
		t.Fatal("expected no tokens with no room in the context window")
	}
	if err := handle.Err(); err == nil || !core.Contains(err.Error(), "no room to generate") {
		t.Fatalf("Err() = %v, want the no-room message", err)
	}
}

// TestSessionHandleGenerateUsesSampledPath pins the temperature/top-p/repeat-
// penalty branch: any of the three route through GenerateSampledFromCacheEach
// instead of the greedy path.
func TestSessionHandleGenerateUsesSampledPath(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	sess := &fakeSession{pos: 1, genIDs: []int32{10, 11}}
	handle := NewSessionHandle(m, sess)
	for range handle.Generate(context.Background(), inference.GenerateConfig{MaxTokens: 2, Temperature: 0.7}) {
	}
	if sess.sampledCalls != 1 {
		t.Fatalf("sampledCalls = %d, want 1 (Temperature>0 must use the sampled path)", sess.sampledCalls)
	}
}

// TestSessionHandleGenerateAppliesDeclaredSampling pins #1844: the continuity
// lane folds the checkpoint's declared sampling defaults exactly as the
// stateless decode does (request-set > model-declared > engine fallback). A
// request with NO sampling fields against a model declaring temperature 0.7
// must take the SAMPLED branch; an explicit Temperature 0 (flag set) stays
// greedy even with the declaration; and with no declarer the engine fallback
// (greedy) stands — before the fix the first case decoded greedy on this lane
// while the stateless lane sampled.
func TestSessionHandleGenerateAppliesDeclaredSampling(t *testing.T) {
	temp := float32(0.7)
	declared := SamplingDefaults{Temperature: &temp}

	// (a) unset request + declared temperature → sampled, same as stateless.
	m := NewTextModel(samplingDeclarerTokenModel{TokenModel: &fakeTokenModel{}, defaults: declared},
		newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
	sess := &fakeSession{pos: 1, genIDs: []int32{10, 11}}
	for range NewSessionHandle(m, sess).Generate(context.Background(), inference.GenerateConfig{MaxTokens: 2}) {
	}
	if sess.sampledCalls != 1 {
		t.Fatalf("declared 0.7 + unset request: sampledCalls = %d, want 1 (the #1844 divergence)", sess.sampledCalls)
	}

	// (b) explicit greedy wins over the declaration (the *Set flag honoured).
	sess = &fakeSession{pos: 1, genIDs: []int32{10, 11}}
	for range NewSessionHandle(m, sess).Generate(context.Background(), inference.GenerateConfig{MaxTokens: 2, Temperature: 0, TemperatureSet: true}) {
	}
	if sess.sampledCalls != 0 {
		t.Fatalf("explicit Temperature 0: sampledCalls = %d, want 0 (request-set wins)", sess.sampledCalls)
	}

	// (c) no declarer: the engine fallback (greedy) stands, unchanged.
	bare := newTestModel(t, &fakeTokenModel{})
	sess = &fakeSession{pos: 1, genIDs: []int32{10, 11}}
	for range NewSessionHandle(bare, sess).Generate(context.Background(), inference.GenerateConfig{MaxTokens: 2}) {
	}
	if sess.sampledCalls != 0 {
		t.Fatalf("no declarer + unset request: sampledCalls = %d, want 0 (greedy fallback)", sess.sampledCalls)
	}
}

// --- CaptureKV -----------------------------------------------------------

// TestSession_SessionHandle_CaptureKV_Good pins the populated-snapshot path
// over retained state.
func TestSession_SessionHandle_CaptureKV_Good(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, &fakeSession{pos: 2})
	snap, err := handle.CaptureKV(context.Background())
	if err != nil {
		t.Fatalf("CaptureKV: %v", err)
	}
	if snap == nil || snap.SeqLen != 2 {
		t.Fatalf("CaptureKV snapshot = %+v, want SeqLen=2", snap)
	}
}

// TestSession_SessionHandle_CaptureKV_Bad pins the not-ready guard.
func TestSession_SessionHandle_CaptureKV_Bad(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, &fakeSession{})
	if _, err := handle.CaptureKV(context.Background()); err == nil {
		t.Fatal("want an error capturing KV with no retained state")
	}
}

// TestSession_SessionHandle_CaptureKV_Ugly pins engine-error propagation.
func TestSession_SessionHandle_CaptureKV_Ugly(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, &fakeSession{pos: 1, captureErr: core.NewError("capture failed")})
	if _, err := handle.CaptureKV(context.Background()); err == nil {
		t.Fatal("want the engine's CaptureKVWithOptions failure to propagate")
	}
}

// --- RangeKVBlocks ---------------------------------------------------------

// TestSession_SessionHandle_RangeKVBlocks_Good pins the streaming path: at
// least one block over retained state.
func TestSession_SessionHandle_RangeKVBlocks_Good(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, &fakeSession{pos: 4})
	blocks := 0
	err := handle.RangeKVBlocks(context.Background(), 16, kv.CaptureOptions{}, func(kv.Block) (bool, error) {
		blocks++
		return true, nil
	})
	if err != nil {
		t.Fatalf("RangeKVBlocks: %v", err)
	}
	if blocks == 0 {
		t.Fatal("RangeKVBlocks yielded zero blocks over retained state")
	}
}

// TestSession_SessionHandle_RangeKVBlocks_Bad pins the nil-yield guard,
// checked ahead of the readiness guard.
func TestSession_SessionHandle_RangeKVBlocks_Bad(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, &fakeSession{})
	err := handle.RangeKVBlocks(context.Background(), 16, kv.CaptureOptions{}, nil)
	if err == nil || !core.Contains(err.Error(), "nil yield") {
		t.Fatalf("RangeKVBlocks err = %v, want the nil-yield message", err)
	}
}

// TestSession_SessionHandle_RangeKVBlocks_Ugly pins the not-ready guard: a
// valid yield over a never-prefilled session still refuses.
func TestSession_SessionHandle_RangeKVBlocks_Ugly(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, &fakeSession{})
	err := handle.RangeKVBlocks(context.Background(), 16, kv.CaptureOptions{}, func(kv.Block) (bool, error) { return true, nil })
	if err == nil {
		t.Fatal("want an error ranging KV blocks with no retained state")
	}
}

// --- RestoreKV (the durable-state alias) -----------------------------------

// restoreRecordingSession is a minimal engine.Session that records the
// snapshot handed to RestoreFromKV, so the RestoreKV alias's delegation is
// observable in isolation from the fuller fakeSession's other behaviour.
type restoreRecordingSession struct {
	restored *kv.Snapshot
}

func (s *restoreRecordingSession) PrefillTokens([]int32) error { return nil }
func (s *restoreRecordingSession) AppendTokens([]int32) error  { return nil }
func (s *restoreRecordingSession) Pos() int                    { return 0 }
func (s *restoreRecordingSession) GenerateFromCacheEach(int, int, func(int32) bool) ([]int32, error) {
	return nil, nil
}
func (s *restoreRecordingSession) GenerateSampledFromCacheEach(int, []int32, *model.Sampler, model.SampleParams, model.TokenTransform, func(int32) bool) ([]int32, error) {
	return nil, nil
}
func (s *restoreRecordingSession) CaptureKVWithOptions(kv.CaptureOptions) (*kv.Snapshot, error) {
	return nil, nil
}
func (s *restoreRecordingSession) RangeKVBlocks(int, kv.CaptureOptions, func(kv.Block) (bool, error)) error {
	return nil
}
func (s *restoreRecordingSession) RestoreFromKV(_ context.Context, snapshot *kv.Snapshot) error {
	s.restored = snapshot
	return nil
}
func (s *restoreRecordingSession) Close() error { return nil }

// TestSession_SessionHandle_RestoreKV_Good pins the #291 fix: the durable
// session's RestoreKV seam delegates to the engine restore, so waking a
// stored KV prefix through model/state/session reaches the engine rather than
// the unsupported-restore guard.
func TestSession_SessionHandle_RestoreKV_Good(t *testing.T) {
	sess := &restoreRecordingSession{}
	handle := NewSessionHandle(&TextModel{}, sess)
	snap := &kv.Snapshot{Tokens: []int32{1, 2, 3}}
	if err := handle.RestoreKV(context.Background(), snap); err != nil {
		t.Fatalf("RestoreKV: %v", err)
	}
	if sess.restored != snap {
		t.Fatal("RestoreKV did not delegate the snapshot to the engine session")
	}
}

// TestSession_SessionHandle_RestoreKV_Bad pins that the alias inherits
// RestoreFromKV's nil-snapshot guard rather than sidestepping it.
func TestSession_SessionHandle_RestoreKV_Bad(t *testing.T) {
	handle := NewSessionHandle(&TextModel{}, &restoreRecordingSession{})
	if err := handle.RestoreKV(context.Background(), nil); err == nil {
		t.Fatal("want an error delegating a nil snapshot through RestoreKV")
	}
}

// TestSession_SessionHandle_RestoreKV_Ugly pins that the alias inherits the
// shared readiness guard: a not-ready handle refuses through RestoreKV too.
func TestSession_SessionHandle_RestoreKV_Ugly(t *testing.T) {
	handle := NewSessionHandle(&TextModel{}, nil)
	if err := handle.RestoreKV(context.Background(), &kv.Snapshot{}); err == nil {
		t.Fatal("want an error delegating through RestoreKV on a not-ready handle")
	}
}

// --- RestoreFromKV -----------------------------------------------------

// TestSession_SessionHandle_RestoreFromKV_Good pins the successful restore:
// the handle's tracked tokens come from the snapshot and generated/duration
// reset.
func TestSession_SessionHandle_RestoreFromKV_Good(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	sess := &fakeSession{}
	handle := NewSessionHandle(m, sess)
	snap := &kv.Snapshot{Tokens: []int32{1, 2, 3}}
	if err := handle.RestoreFromKV(context.Background(), snap); err != nil {
		t.Fatalf("RestoreFromKV: %v", err)
	}
	if sess.pos != 3 {
		t.Fatalf("engine session pos = %d, want 3 (restored from the snapshot's 3 tokens)", sess.pos)
	}
}

// TestSession_SessionHandle_RestoreFromKV_Bad pins the nil-snapshot guard.
func TestSession_SessionHandle_RestoreFromKV_Bad(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, &fakeSession{})
	if err := handle.RestoreFromKV(context.Background(), nil); err == nil {
		t.Fatal("want an error restoring a nil snapshot")
	}
}

// TestSession_SessionHandle_RestoreFromKV_Ugly pins engine-error propagation.
func TestSession_SessionHandle_RestoreFromKV_Ugly(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	sess := &fakeSession{restoreErr: core.NewError("corrupt snapshot")}
	handle := NewSessionHandle(m, sess)
	if err := handle.RestoreFromKV(context.Background(), &kv.Snapshot{Tokens: []int32{1}}); err == nil {
		t.Fatal("want the engine's RestoreFromKV failure to propagate")
	}
}

// --- Fork ----------------------------------------------------------------

// TestSession_SessionHandle_Fork_Good pins the independent-copy contract:
// advancing the fork must not disturb the parent, and both keep generating.
func TestSession_SessionHandle_Fork_Good(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{genIDs: defaultFakeGenIDs})
	handle := NewSessionHandle(m, &fakeSession{pos: 1, genIDs: defaultFakeGenIDs})
	fork, err := handle.Fork(context.Background())
	if err != nil {
		t.Fatalf("Fork: %v", err)
	}
	defer func() { _ = fork.Close() }()
	if toks := drainTokens(fork, inference.GenerateConfig{MaxTokens: 2}); len(toks) == 0 {
		t.Fatal("fork produced no tokens")
	}
	if toks := drainTokens(handle, inference.GenerateConfig{MaxTokens: 2}); len(toks) == 0 {
		t.Fatal("parent produced no tokens after the fork advanced")
	}
}

// TestSession_SessionHandle_Fork_Bad pins CaptureKV failure propagation: Fork
// cannot start from a session with no retained state to capture.
func TestSession_SessionHandle_Fork_Bad(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, &fakeSession{})
	if _, err := handle.Fork(context.Background()); err == nil {
		t.Fatal("want an error forking a session with no retained state")
	}
}

// TestSession_SessionHandle_Fork_Ugly pins restore-failure cleanup: when the
// fresh fork target fails to restore the captured snapshot, Fork closes the
// half-open fork and returns the failure rather than handing back a broken
// session.
func TestSession_SessionHandle_Fork_Ugly(t *testing.T) {
	original := &fakeSession{pos: 1}
	forkTarget := &fakeSession{restoreErr: core.NewError("restore rejected")}
	// original is wrapped directly (never opened through tm), so the fork's
	// s.model.NewSession() call is the FIRST OpenEngineSession call — pin
	// forkTarget as that first result.
	tm := &fakeTokenModel{sessions: []*fakeSession{forkTarget}}
	m := newTestModel(t, tm)
	handle := NewSessionHandle(m, original)
	if _, err := handle.Fork(context.Background()); err == nil {
		t.Fatal("want Fork to fail when the fork target rejects the restore")
	}
	if forkTarget.closeCalls != 1 {
		t.Fatalf("fork target Close called %d times, want 1 (Fork must clean up the half-open fork)", forkTarget.closeCalls)
	}
}

// drainTokens ranges a SessionHandle's Generate to completion.
func drainTokens(s inference.SessionHandle, cfg inference.GenerateConfig) []inference.Token {
	var out []inference.Token
	for tok := range s.Generate(context.Background(), cfg) {
		out = append(out, tok)
	}
	return out
}

// --- Reset -----------------------------------------------------------------

// TestSession_SessionHandle_Reset_Good pins the fresh-prefill contract: Reset
// releases retained state, closes the old engine session, and opens a new one
// ready for another Prefill.
func TestSession_SessionHandle_Reset_Good(t *testing.T) {
	first := &fakeSession{}
	tm := &fakeTokenModel{sessions: []*fakeSession{first, {}}}
	m := newTestModel(t, tm)
	handle := NewSessionHandle(m, first)
	if err := handle.Prefill(context.Background(), "before reset"); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	handle.Reset()
	if first.closeCalls != 1 {
		t.Fatalf("old session Close called %d times, want 1", first.closeCalls)
	}
	if err := handle.Prefill(context.Background(), "after reset"); err != nil {
		t.Fatalf("Prefill after Reset: %v", err)
	}
}

// TestSession_SessionHandle_Reset_Bad pins the closed-handle no-op: Reset on
// an already-closed handle clears local bookkeeping but does not reopen a
// session (there is nothing left to serve).
func TestSession_SessionHandle_Reset_Bad(t *testing.T) {
	sess := &fakeSession{}
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, sess)
	if err := handle.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	handle.Reset() // must not panic re-closing or reopening
	if err := handle.Prefill(context.Background(), "after reset on closed"); err == nil {
		t.Fatal("want a closed handle to stay closed after Reset")
	}
}

// TestSession_SessionHandle_Reset_Ugly pins open-failure handling: when the
// engine cannot open a replacement session, Reset records the failure and
// leaves the handle without a working session rather than panicking.
func TestSession_SessionHandle_Reset_Ugly(t *testing.T) {
	sess := &fakeSession{pos: 1}
	tm := &fakeTokenModel{}
	m := newTestModel(t, tm)
	handle := NewSessionHandle(m, sess)
	tm.openErr = core.NewError("engine offline")
	handle.Reset()
	if err := handle.Err(); err == nil {
		t.Fatal("want Reset to record the engine's open failure")
	}
}

// --- Close -----------------------------------------------------------------

// TestSession_SessionHandle_Close_Good pins the release contract: the engine
// session is closed exactly once.
func TestSession_SessionHandle_Close_Good(t *testing.T) {
	sess := &fakeSession{}
	handle := NewSessionHandle(newTestModel(t, &fakeTokenModel{}), sess)
	if err := handle.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if sess.closeCalls != 1 {
		t.Fatalf("engine session Close called %d times, want 1", sess.closeCalls)
	}
}

// TestSession_SessionHandle_Close_Bad pins the nil-receiver guard: Close on a
// nil *SessionHandle is a safe no-op.
func TestSession_SessionHandle_Close_Bad(t *testing.T) {
	var handle *SessionHandle
	if err := handle.Close(); err != nil {
		t.Fatalf("Close on a nil handle = %v, want nil", err)
	}
}

// TestSession_SessionHandle_Close_Ugly pins idempotency: a second Close does
// not call the engine session's Close again.
func TestSession_SessionHandle_Close_Ugly(t *testing.T) {
	sess := &fakeSession{}
	handle := NewSessionHandle(newTestModel(t, &fakeTokenModel{}), sess)
	if err := handle.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := handle.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if sess.closeCalls != 1 {
		t.Fatalf("engine session Close called %d times across two handle.Close() calls, want 1", sess.closeCalls)
	}
}

// --- Err -------------------------------------------------------------------

// TestSession_SessionHandle_Err_Good pins the readback: a failed operation's
// error is observable via Err().
func TestSession_SessionHandle_Err_Good(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	handle := NewSessionHandle(m, &fakeSession{})
	_ = handle.AppendPrompt(context.Background(), "no prefix yet")
	if err := handle.Err(); err == nil {
		t.Fatal("want Err() to surface the AppendPrompt failure")
	}
}

// TestSession_SessionHandle_Err_Bad pins the nil-receiver guard: Err on a nil
// *SessionHandle returns nil rather than panicking.
func TestSession_SessionHandle_Err_Bad(t *testing.T) {
	var handle *SessionHandle
	if err := handle.Err(); err != nil {
		t.Fatalf("Err() on a nil handle = %v, want nil", err)
	}
}

// TestSession_SessionHandle_Err_Ugly pins Err()'s sticky-failure edge: Close
// only overwrites s.err when the engine's OWN Close call fails — a CLEAN
// Close leaves a prior real failure in place rather than clearing it, so
// Err() still reports the pre-Close failure afterwards.
func TestSession_SessionHandle_Err_Ugly(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	sess := &fakeSession{}
	handle := NewSessionHandle(m, sess)
	_ = handle.AppendPrompt(context.Background(), "no prefix yet")
	if err := handle.Err(); err == nil {
		t.Fatal("want the AppendPrompt failure recorded before Close")
	}
	if err := handle.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := handle.Err(); err == nil {
		t.Fatal("want Err() to still reflect the pre-Close failure — a clean Close does not clear it")
	}
}
