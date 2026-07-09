// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"context"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/engine/enginetest"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model"
	coreio "dappco.re/go/io"
)

// --- shared test fixtures --------------------------------------------------
//
// The fakes below back every TextModel/SessionHandle/Trainer/vision test in
// this package (model_test.go, session_test.go, vision_test.go and
// trainer_test.go all share package engine, so a fixture defined once here is
// visible to all of them). fakeSession/fakeTokenModel satisfy the real
// [Session]/[TokenModel] contracts directly — no shortcut interfaces — so
// tests exercise the genuine TextModel/SessionHandle wiring against a real
// (if synthetic) engine rather than a hand-waved double.

// fixtureTokenizerJSON is a minimal-but-real tokenizer.json: a <bos>/<eos>
// pair (so Encode always emits at least the BOS token — no test prompt here
// needs to survive BPE segmentation to produce a non-empty id list) plus one
// single-character vocab entry ("z" -> 42) the vision tests use to make an
// image placeholder block tokenise to an exact, countable run of one token id.
const fixtureTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {"z": 42},
    "merges": []
  },
  "added_tokens": [
    {"id": 1, "content": "<bos>", "special": true},
    {"id": 2, "content": "<eos>", "special": true}
  ]
}`

// newFixtureTokenizer loads fixtureTokenizerJSON through the production
// LoadTokenizer path (not a hand-built struct), so Encode/Decode in these
// tests run the genuine BPE machinery rather than a faked tokenizer.
func newFixtureTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	if err := coreio.Local.Write(path, fixtureTokenizerJSON); err != nil {
		t.Fatalf("write fixture tokenizer: %v", err)
	}
	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		t.Fatalf("load fixture tokenizer: %v", err)
	}
	return tok
}

// defaultFakeGenIDs is a generous canned decode stream: enough ids to satisfy
// any budget these tests ask for, none equal to the fixture tokenizer's EOS
// (2) or BOS (1), so a plain budget-bounded generation runs the full budget
// rather than stopping on an accidental stop-token match.
var defaultFakeGenIDs = []int32{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}

// fakeSession is a configurable [Session] double. PrefillTokens/AppendTokens/
// RestoreFromKV track the retained length in pos exactly as a real engine
// session would, so the SessionHandle/TextModel state-machine guards
// (Pos()<=0 meaning "nothing retained yet", etc.) see realistic state.
type fakeSession struct {
	pos int

	prefillErr, appendErr, captureErr, rangeErr, restoreErr, closeErr error
	genErr, sampledErr                                                error

	genIDs []int32

	prefillCalls [][]int32
	appendCalls  [][]int32
	restoreSnap  *kv.Snapshot
	closeCalls   int
	sampledCalls int
}

func (f *fakeSession) PrefillTokens(ids []int32) error {
	f.prefillCalls = append(f.prefillCalls, ids)
	if f.prefillErr != nil {
		return f.prefillErr
	}
	f.pos = len(ids)
	return nil
}

func (f *fakeSession) AppendTokens(ids []int32) error {
	f.appendCalls = append(f.appendCalls, ids)
	if f.appendErr != nil {
		return f.appendErr
	}
	f.pos += len(ids)
	return nil
}

func (f *fakeSession) Pos() int { return f.pos }

// GenerateFromCacheEach replays genIDs, yielding up to maxNew of them. eosID
// mirrors the real contract (< 0 lets the caller's yield own the stop
// decision); a non-negative eosID additionally stops the replay on a match,
// exactly as a real engine's own decode loop would.
func (f *fakeSession) GenerateFromCacheEach(maxNew, eosID int, yield func(int32) bool) ([]int32, error) {
	if f.genErr != nil {
		return nil, f.genErr
	}
	var out []int32
	for _, id := range f.genIDs {
		if len(out) >= maxNew {
			break
		}
		out = append(out, id)
		f.pos++
		cont := yield(id)
		if eosID >= 0 && id == int32(eosID) {
			break
		}
		if !cont {
			break
		}
	}
	return out, nil
}

func (f *fakeSession) GenerateSampledFromCacheEach(maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	f.sampledCalls++
	if f.sampledErr != nil {
		return nil, f.sampledErr
	}
	var out []int32
	for _, id := range f.genIDs {
		if len(out) >= maxNew {
			break
		}
		out = append(out, id)
		f.pos++
		if !yield(id) {
			break
		}
	}
	return out, nil
}

func (f *fakeSession) CaptureKVWithOptions(kv.CaptureOptions) (*kv.Snapshot, error) {
	if f.captureErr != nil {
		return nil, f.captureErr
	}
	tokens := make([]int32, f.pos)
	for i := range tokens {
		tokens[i] = int32(i)
	}
	return &kv.Snapshot{Tokens: tokens, SeqLen: f.pos}, nil
}

func (f *fakeSession) RangeKVBlocks(blockSize int, opts kv.CaptureOptions, yield func(kv.Block) (bool, error)) error {
	if f.rangeErr != nil {
		return f.rangeErr
	}
	if f.pos == 0 {
		return nil
	}
	_, err := yield(kv.Block{Index: 0, TokenStart: 0, TokenCount: f.pos})
	return err
}

func (f *fakeSession) RestoreFromKV(_ context.Context, snapshot *kv.Snapshot) error {
	f.restoreSnap = snapshot
	if f.restoreErr != nil {
		return f.restoreErr
	}
	if snapshot != nil {
		f.pos = len(snapshot.Tokens)
	}
	return nil
}

func (f *fakeSession) Close() error {
	f.closeCalls++
	return f.closeErr
}

var _ Session = (*fakeSession)(nil)

// fakeTokenModel is a configurable [TokenModel] double. OpenEngineSession
// hands out a fresh *fakeSession per call (never the same pointer twice)
// unless nextSession pins a specific instance — so Reset/Fork exercise
// genuinely independent retained state, exactly as a real engine would, while
// a test that needs to configure or inspect the exact session in play can
// still pin one.
type fakeTokenModel struct {
	openErr  error
	closeErr error
	genIDs   []int32

	// nextSession, when set, is returned by every OpenEngineSession call
	// instead of a fresh fakeSession.
	nextSession *fakeSession

	// sessions, when set, supplies a specific *fakeSession per successive
	// OpenEngineSession call (index 0 for the first call, 1 for the second,
	// ...) — for tests that need the Nth session opened (e.g. a Fork target)
	// to behave differently from the first. Calls beyond len(sessions) fall
	// back to a fresh default session.
	sessions []*fakeSession

	openCalls  int
	closeCalls int
}

func (f *fakeTokenModel) OpenEngineSession() (Session, error) {
	f.openCalls++
	if f.openErr != nil {
		return nil, f.openErr
	}
	if f.nextSession != nil {
		return f.nextSession, nil
	}
	if idx := f.openCalls - 1; idx < len(f.sessions) {
		return f.sessions[idx], nil
	}
	ids := f.genIDs
	if ids == nil {
		ids = defaultFakeGenIDs
	}
	return &fakeSession{genIDs: ids}, nil
}

func (f *fakeTokenModel) Close() error {
	f.closeCalls++
	return f.closeErr
}

var _ TokenModel = (*fakeTokenModel)(nil)

// bareTokenModel is a minimal engine.TokenModel with no optional seams — used to
// prove Capabilities reports no cache modes when the engine declares none.
type bareTokenModel struct{}

func (bareTokenModel) OpenEngineSession() (Session, error) { return nil, nil }
func (bareTokenModel) Close() error                        { return nil }

// cacheModeTokenModel adds the CacheModeReporter seam so Capabilities forwards
// the declared modes.
type cacheModeTokenModel struct {
	bareTokenModel
	modes []string
}

func (m cacheModeTokenModel) SupportedCacheModes() []string { return m.modes }

// newTestModel builds a TextModel over tm and the fixture tokenizer, with a
// generous context window — the shared model fixture session_test.go and
// vision_test.go's SessionHandle/chatMultimodal tests open sessions against.
func newTestModel(t *testing.T, tm TokenModel) *TextModel {
	t.Helper()
	return NewTextModel(tm, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
}

// --- NewTextModel ------------------------------------------------------------

// TestModel_NewTextModel_Good pins the constructor: every argument lands on
// the returned model's exported readback surface (ModelType/Info), and the
// fresh model starts in the OK state before any generation has run.
func TestModel_NewTextModel_Good(t *testing.T) {
	info := inference.ModelInfo{Architecture: "gemma-test", VocabSize: 64, NumLayers: 2, HiddenSize: 16}
	m := NewTextModel(&fakeTokenModel{}, newFixtureTokenizer(t), "gemma-test", info, 4096)
	if got := m.ModelType(); got != "gemma-test" {
		t.Fatalf("ModelType() = %q, want %q", got, "gemma-test")
	}
	if got := m.Info(); got != info {
		t.Fatalf("Info() = %+v, want %+v", got, info)
	}
	if r := m.Err(); !r.OK {
		t.Fatalf("fresh model Err() = %+v, want OK", r)
	}
}

// TestModel_NewTextModel_Bad pins the nil-tokenizer edge: a model built
// without a tokenizer tokenises every prompt to nothing, so Generate fails
// clean instead of panicking on a nil dereference.
func TestModel_NewTextModel_Bad(t *testing.T) {
	m := NewTextModel(&fakeTokenModel{}, nil, "gemma-test", inference.ModelInfo{}, 4096)
	for range m.Generate(context.Background(), "hello") {
		t.Fatal("expected no tokens with a nil tokenizer")
	}
	if r := m.Err(); r.OK {
		t.Fatal("want a failure Result when the tokenizer is nil")
	}
}

// TestModel_NewTextModel_Ugly pins the zero-maxLen edge: a model with no
// context window at all rejects generation ("no room") rather than serving a
// prompt into an unbounded budget.
func TestModel_NewTextModel_Ugly(t *testing.T) {
	m := NewTextModel(&fakeTokenModel{}, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 0)
	for range m.Generate(context.Background(), "hi") {
		t.Fatal("expected no tokens with a zero context window")
	}
	if r := m.Err(); r.OK {
		t.Fatal("want a failure Result when maxLen is zero")
	}
}

// --- conformance ---------------------------------------------------------

// TestModel_TextModel_Conformance runs the shared enginetest.TextModel suite
// (the same suite metal/hip run against their real decode) against a
// synthetic TokenModel — proving NewTextModel's wiring (Generate, Chat,
// Metrics, Info, ModelType, Classify, BatchGenerate, Close) holds for ANY
// conformant TokenModel, not just a hand-picked happy path.
func TestModel_TextModel_Conformance(t *testing.T) {
	tok := newFixtureTokenizer(t)
	enginetest.TextModel(t, func(t *testing.T) inference.TextModel {
		return NewTextModel(&fakeTokenModel{}, tok, "gemma-test",
			inference.ModelInfo{Architecture: "gemma-test", VocabSize: 64, NumLayers: 2, HiddenSize: 16}, 4096)
	})
}

// --- OpenTrainer -----------------------------------------------------------

// TestModel_TextModel_OpenTrainer_Good pins the forwarding seam: a TokenModel
// that supports training hands back its own Trainer untouched.
func TestModel_TextModel_OpenTrainer_Good(t *testing.T) {
	tr := &fakeTrainer{stepLoss: 0.1}
	m := NewTextModel(&fakeTrainerModel{trainer: tr}, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
	got, err := m.OpenTrainer(inference.TrainingConfig{Epochs: 1})
	if err != nil {
		t.Fatalf("OpenTrainer: %v", err)
	}
	if got != tr {
		t.Fatal("OpenTrainer did not return the engine's own Trainer")
	}
}

// TestModel_TextModel_OpenTrainer_Bad pins the not-initialised guard: a model
// with no engine attached refuses to open a trainer rather than panicking.
func TestModel_TextModel_OpenTrainer_Bad(t *testing.T) {
	m := &TextModel{}
	if _, err := m.OpenTrainer(inference.TrainingConfig{}); err == nil {
		t.Fatal("want an error opening a trainer with no engine model attached")
	}
}

// TestModel_TextModel_OpenTrainer_Ugly pins the capability probe: an engine
// TokenModel that does not implement TrainerModel reports a clear
// unsupported error rather than a type-assertion panic.
func TestModel_TextModel_OpenTrainer_Ugly(t *testing.T) {
	m := NewTextModel(&fakeTokenModel{}, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
	if _, err := m.OpenTrainer(inference.TrainingConfig{}); err == nil {
		t.Fatal("want an error when the engine does not support training")
	}
}

// --- Generate ----------------------------------------------------------

// TestModel_TextModel_Generate_Good pins the plain streaming path: a bounded
// generation yields exactly the requested token budget from the retained
// session and leaves Err() clean.
func TestModel_TextModel_Generate_Good(t *testing.T) {
	m := NewTextModel(&fakeTokenModel{genIDs: []int32{10, 11, 12}}, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
	var got []inference.Token
	for tok := range m.Generate(context.Background(), "hi", inference.WithMaxTokens(3)) {
		got = append(got, tok)
	}
	if len(got) != 3 {
		t.Fatalf("Generate produced %d tokens, want 3", len(got))
	}
	if r := m.Err(); !r.OK {
		t.Fatalf("Err() after a clean generation = %+v, want OK", r)
	}
}

// TestModel_TextModel_Generate_Bad pins the empty-prompt guard: a tokenizer
// that cannot produce any ids for the prompt fails clean rather than opening
// a session over nothing.
func TestModel_TextModel_Generate_Bad(t *testing.T) {
	m := &TextModel{tok: spTok()}
	for range m.Generate(context.Background(), "hello") {
		t.Fatal("expected no tokens from an empty-tokenisation prompt")
	}
	if r := m.Err(); r.OK {
		t.Fatal("want a failure Result for an empty-tokenisation prompt")
	}
}

// TestModel_TextModel_Generate_Ugly pins session-open failure propagation:
// when the engine cannot open a session, Generate yields nothing and Err()
// carries the wrapped engine error rather than silently succeeding.
func TestModel_TextModel_Generate_Ugly(t *testing.T) {
	wantErr := core.NewError("engine offline")
	m := NewTextModel(&fakeTokenModel{openErr: wantErr}, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
	for range m.Generate(context.Background(), "hi") {
		t.Fatal("expected no tokens when the session cannot open")
	}
	if r := m.Err(); r.OK {
		t.Fatal("want a failure Result when OpenEngineSession fails")
	}
}

// --- Chat --------------------------------------------------------------

// TestModel_TextModel_Chat_Good pins the plain text turn path: a text-only
// message streams the requested budget through the same session machinery as
// Generate.
func TestModel_TextModel_Chat_Good(t *testing.T) {
	m := NewTextModel(&fakeTokenModel{genIDs: []int32{10, 11}}, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
	var got int
	for range m.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}, inference.WithMaxTokens(2)) {
		got++
	}
	if got != 2 {
		t.Fatalf("Chat produced %d tokens, want 2", got)
	}
}

// TestModel_TextModel_Chat_Bad pins the vision rejection: a turn carrying an
// image against an engine with no vision support fails clean with a named
// error rather than silently dropping the image.
func TestModel_TextModel_Chat_Bad(t *testing.T) {
	m := NewTextModel(&fakeTokenModel{}, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
	for range m.Chat(context.Background(), []inference.Message{{Role: "user", Content: "look", Images: [][]byte{{1, 2, 3}}}}) {
		t.Fatal("expected no tokens: engine has no vision support")
	}
	r := m.Err()
	if r.OK || !core.Contains(r.Error(), "does not accept image input") {
		t.Fatalf("Err() = %+v, want the image-rejection message", r)
	}
}

// TestModel_TextModel_Chat_Ugly pins ctx-cancellation propagation: a context
// that is already cancelled before Chat starts yields no tokens and surfaces
// the cancellation as the generation error.
func TestModel_TextModel_Chat_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	m := NewTextModel(&fakeTokenModel{}, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
	for range m.Chat(ctx, []inference.Message{{Role: "user", Content: "hi"}}) {
		t.Fatal("expected no tokens once the context is already cancelled")
	}
	if r := m.Err(); r.OK {
		t.Fatal("want a failure Result when the context is already cancelled")
	}
}

// --- FormatChatPrompt ----------------------------------------------------

// TestModel_TextModel_FormatChatPrompt_Good pins the fresh-turn framing the
// durable -state loop opens a session with: the full gemma turn template,
// trailing open model turn. It is byte-identical to the serve path
// (formatChatTurns, which Chat encodes) — the parity that keeps a stateful
// first turn framed like a stateless serve request.
func TestModel_TextModel_FormatChatPrompt_Good(t *testing.T) {
	m := &TextModel{}
	got := m.FormatChatPrompt([]inference.Message{{Role: "user", Content: "hi"}})
	want := "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n"
	if got != want {
		t.Fatalf("FormatChatPrompt = %q, want %q", got, want)
	}
	if got != formatChatTurns(m.turnTokens(), []inference.Message{{Role: "user", Content: "hi"}}) {
		t.Fatal("FormatChatPrompt must equal the serve-path formatChatTurns render")
	}
}

// TestModel_TextModel_FormatChatPrompt_Bad pins the empty-history edge: no
// turns at all still renders a well-formed trailing model turn to complete,
// rather than an empty or malformed prompt.
func TestModel_TextModel_FormatChatPrompt_Bad(t *testing.T) {
	m := &TextModel{}
	if got, want := m.FormatChatPrompt(nil), "<start_of_turn>model\n"; got != want {
		t.Fatalf("FormatChatPrompt(nil) = %q, want %q", got, want)
	}
}

// TestModel_TextModel_FormatChatPrompt_Ugly pins the assistant/model role
// normalisation across a multi-turn history: both spellings render as the
// gemma "model" turn.
func TestModel_TextModel_FormatChatPrompt_Ugly(t *testing.T) {
	m := &TextModel{}
	got := m.FormatChatPrompt([]inference.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	})
	want := "<start_of_turn>user\nhi<end_of_turn>\n" +
		"<start_of_turn>model\nhello<end_of_turn>\n" +
		"<start_of_turn>model\n"
	if got != want {
		t.Fatalf("FormatChatPrompt = %q, want %q", got, want)
	}
}

// --- FormatChatContinuation ------------------------------------------------

// TestModel_TextModel_FormatChatContinuation_Good pins the woken-turn
// framing: it closes the model turn the restored KV prefix ends on with a
// leading <end_of_turn>, then renders the new user turn and reopens the
// assistant header.
func TestModel_TextModel_FormatChatContinuation_Good(t *testing.T) {
	m := &TextModel{}
	got := m.FormatChatContinuation([]inference.Message{{Role: "user", Content: "and now?"}})
	want := "<end_of_turn>\n<start_of_turn>user\nand now?<end_of_turn>\n<start_of_turn>model\n"
	if got != want {
		t.Fatalf("FormatChatContinuation = %q, want %q", got, want)
	}
}

// TestModel_TextModel_FormatChatContinuation_Bad pins the empty-turn edge: no
// new messages still closes the prior model turn and reopens a fresh one,
// rather than producing a malformed or empty continuation.
func TestModel_TextModel_FormatChatContinuation_Bad(t *testing.T) {
	m := &TextModel{}
	if got, want := m.FormatChatContinuation(nil), "<end_of_turn>\n<start_of_turn>model\n"; got != want {
		t.Fatalf("FormatChatContinuation(nil) = %q, want %q", got, want)
	}
}

// TestModel_TextModel_FormatChatContinuation_Ugly proves the continuation
// frames only what it is handed: two turns in, the render contains both and
// still opens with the single close — it never re-emits earlier history, so
// the restored KV is extended, not replayed.
func TestModel_TextModel_FormatChatContinuation_Ugly(t *testing.T) {
	m := &TextModel{}
	got := m.FormatChatContinuation([]inference.Message{{Role: "user", Content: "q2"}})
	if core.Count(got, "<start_of_turn>user") != 1 {
		t.Fatalf("continuation replayed history: %q", got)
	}
	if !core.HasPrefix(got, "<end_of_turn>\n") {
		t.Fatalf("continuation must open by closing the prior model turn: %q", got)
	}
}

// --- Classify ------------------------------------------------------------

// TestModel_TextModel_Classify_Good pins the prefill-only classify path: one
// sampled boundary token per prompt, in order.
func TestModel_TextModel_Classify_Good(t *testing.T) {
	m := NewTextModel(&fakeTokenModel{genIDs: []int32{7}}, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
	r := m.Classify(context.Background(), []string{"a", "b"})
	if !r.OK {
		t.Fatalf("Classify: %v", r.Error())
	}
	results, ok := r.Value.([]inference.ClassifyResult)
	if !ok || len(results) != 2 {
		t.Fatalf("Classify results = %+v, want 2 []inference.ClassifyResult", r.Value)
	}
	if results[0].Token.ID != 7 {
		t.Fatalf("Classify sampled id = %d, want 7", results[0].Token.ID)
	}
}

// TestModel_TextModel_Classify_Bad pins the empty-prompt guard: a prompt that
// tokenises to nothing fails the whole Classify call clean.
func TestModel_TextModel_Classify_Bad(t *testing.T) {
	m := &TextModel{tok: spTok()}
	r := m.Classify(context.Background(), []string{"hello"})
	if r.OK || !core.Contains(r.Error(), "empty prompt after tokenisation") {
		t.Fatalf("Classify = %+v, want the empty-tokenisation failure", r)
	}
}

// TestModel_TextModel_Classify_Ugly pins session-open failure propagation.
func TestModel_TextModel_Classify_Ugly(t *testing.T) {
	wantErr := core.NewError("engine offline")
	m := NewTextModel(&fakeTokenModel{openErr: wantErr}, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
	r := m.Classify(context.Background(), []string{"hello"})
	if r.OK {
		t.Fatal("want a failure Result when the session cannot open")
	}
}

// --- BatchGenerate ---------------------------------------------------------

// TestModel_TextModel_BatchGenerate_Good pins the per-prompt loop: every
// prompt streams its own budget of tokens into its own BatchResult.
func TestModel_TextModel_BatchGenerate_Good(t *testing.T) {
	m := NewTextModel(&fakeTokenModel{genIDs: []int32{1, 2}}, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
	r := m.BatchGenerate(context.Background(), []string{"a", "b"}, inference.WithMaxTokens(2))
	if !r.OK {
		t.Fatalf("BatchGenerate: %v", r.Error())
	}
	results, ok := r.Value.([]inference.BatchResult)
	if !ok || len(results) != 2 {
		t.Fatalf("BatchGenerate results = %+v, want 2 []inference.BatchResult", r.Value)
	}
	for i, res := range results {
		if len(res.Tokens) != 2 {
			t.Fatalf("prompt %d: got %d tokens, want 2", i, len(res.Tokens))
		}
	}
}

// TestModel_TextModel_BatchGenerate_Bad pins per-prompt error capture: a
// prompt that tokenises to nothing carries its own error in BatchResult.Err
// rather than failing the whole Result.
func TestModel_TextModel_BatchGenerate_Bad(t *testing.T) {
	m := &TextModel{tok: spTok(), tm: &fakeTokenModel{}, maxLen: 4096}
	r := m.BatchGenerate(context.Background(), []string{"hello"}, inference.WithMaxTokens(2))
	if !r.OK {
		t.Fatalf("BatchGenerate: %v", r.Error())
	}
	results := r.Value.([]inference.BatchResult)
	if results[0].Err == nil {
		t.Fatal("want a per-prompt error for a prompt that tokenises to nothing")
	}
}

// TestModel_TextModel_BatchGenerate_Ugly pins the exhausted-context-window
// edge: a model with no room to generate still returns a shaped Result with
// the failure captured per-prompt.
func TestModel_TextModel_BatchGenerate_Ugly(t *testing.T) {
	m := NewTextModel(&fakeTokenModel{}, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 0)
	r := m.BatchGenerate(context.Background(), []string{"hi"}, inference.WithMaxTokens(4))
	if !r.OK {
		t.Fatalf("BatchGenerate: %v", r.Error())
	}
	results := r.Value.([]inference.BatchResult)
	if results[0].Err == nil {
		t.Fatal("want a per-prompt error when the context window has no room")
	}
}

// --- NewSession ------------------------------------------------------------

// TestModel_TextModel_NewSession_Good pins the happy path: a healthy engine
// hands back a usable SessionHandle and leaves Err() clean.
func TestModel_TextModel_NewSession_Good(t *testing.T) {
	m := NewTextModel(&fakeTokenModel{}, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
	sess := m.NewSession()
	if sess == nil {
		t.Fatal("NewSession returned nil for a healthy engine")
	}
	defer func() { _ = sess.Close() }()
	if r := m.Err(); !r.OK {
		t.Fatalf("Err() after a clean NewSession = %+v, want OK", r)
	}
}

// TestModel_TextModel_NewSession_Bad pins session-open failure propagation:
// nil comes back and Err() carries the wrapped engine error.
func TestModel_TextModel_NewSession_Bad(t *testing.T) {
	wantErr := core.NewError("engine offline")
	m := NewTextModel(&fakeTokenModel{openErr: wantErr}, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
	if sess := m.NewSession(); sess != nil {
		t.Fatal("want nil session when the engine cannot open one")
	}
	if r := m.Err(); r.OK {
		t.Fatal("want a failure Result when NewSession cannot open")
	}
}

// TestModel_TextModel_NewSession_Ugly pins the not-yet-initialised edge: a
// zero-value TextModel (no engine attached) fails NewSession cleanly via the
// same nil-tm guard openSession uses everywhere else.
func TestModel_TextModel_NewSession_Ugly(t *testing.T) {
	m := &TextModel{}
	if sess := m.NewSession(); sess != nil {
		t.Fatal("want nil session from a TextModel with no engine attached")
	}
	if r := m.Err(); r.OK {
		t.Fatal("want a failure Result from a TextModel with no engine attached")
	}
}

// --- ModelType -------------------------------------------------------------

// TestModel_TextModel_ModelType_Good pins the plain passthrough: the
// architecture selector handed to NewTextModel comes back unchanged.
func TestModel_TextModel_ModelType_Good(t *testing.T) {
	m := &TextModel{modelType: "gemma4"}
	if got := m.ModelType(); got != "gemma4" {
		t.Fatalf("ModelType() = %q, want %q", got, "gemma4")
	}
}

// TestModel_TextModel_ModelType_Bad pins the zero-value edge: a model with no
// modelType set reports empty rather than a placeholder.
func TestModel_TextModel_ModelType_Bad(t *testing.T) {
	m := &TextModel{}
	if got := m.ModelType(); got != "" {
		t.Fatalf("ModelType() = %q, want empty", got)
	}
}

// TestModel_TextModel_ModelType_Ugly pins that ModelType never normalises or
// validates its input — an unusual value (spacing, mixed separators) survives
// verbatim.
func TestModel_TextModel_ModelType_Ugly(t *testing.T) {
	const weird = "  Weird Arch/v2  "
	m := &TextModel{modelType: weird}
	if got := m.ModelType(); got != weird {
		t.Fatalf("ModelType() = %q, want the exact unmodified string %q", got, weird)
	}
}

// --- Info ------------------------------------------------------------------

// TestModel_TextModel_Info_Good pins the plain passthrough of the engine-built
// model metadata.
func TestModel_TextModel_Info_Good(t *testing.T) {
	info := inference.ModelInfo{Architecture: "gemma3", VocabSize: 262144, NumLayers: 34, HiddenSize: 2048}
	m := &TextModel{info: info}
	if got := m.Info(); got != info {
		t.Fatalf("Info() = %+v, want %+v", got, info)
	}
}

// TestModel_TextModel_Info_Bad pins the zero-value edge: a model with no
// metadata set reports the zero ModelInfo.
func TestModel_TextModel_Info_Bad(t *testing.T) {
	m := &TextModel{}
	if got := m.Info(); got != (inference.ModelInfo{}) {
		t.Fatalf("Info() = %+v, want the zero ModelInfo", got)
	}
}

// TestModel_TextModel_Info_Ugly pins that Info never validates or clamps its
// stored value — nonsensical geometry (negative sizes) still passes through
// exactly as stored.
func TestModel_TextModel_Info_Ugly(t *testing.T) {
	info := inference.ModelInfo{QuantBits: -1, VocabSize: -1}
	m := &TextModel{info: info}
	if got := m.Info(); got != info {
		t.Fatalf("Info() = %+v, want the exact stored value %+v", got, info)
	}
}

// --- Metrics -----------------------------------------------------------

// TestModel_TextModel_Metrics_Good pins the snapshot readback after a
// generation records prompt/generated token counts.
func TestModel_TextModel_Metrics_Good(t *testing.T) {
	m := &TextModel{}
	m.setMetrics(5, 3, 2*time.Millisecond, time.Now())
	got := m.Metrics()
	if got.PromptTokens != 5 || got.GeneratedTokens != 3 {
		t.Fatalf("Metrics() = %+v, want PromptTokens=5 GeneratedTokens=3", got)
	}
}

// TestModel_TextModel_Metrics_Bad pins the zero-value edge: a fresh model
// with no completed generation reports the zero metrics.
func TestModel_TextModel_Metrics_Bad(t *testing.T) {
	m := &TextModel{}
	if got := m.Metrics(); got != (inference.GenerateMetrics{}) {
		t.Fatalf("Metrics() on a fresh model = %+v, want the zero value", got)
	}
}

// TestModel_TextModel_Metrics_Ugly pins the decode-phase-trace fold: a traced
// budget attaches to the metrics snapshot the caller reads via Metrics(),
// even without a prior setMetrics call.
func TestModel_TextModel_Metrics_Ugly(t *testing.T) {
	m := &TextModel{}
	budget := &inference.DecodePhaseBudget{Tokens: 7, TotalPerToken: 2 * time.Millisecond}
	m.setDecodePhases(budget)
	got := m.Metrics().DecodePhases
	if got == nil || got.Tokens != 7 {
		t.Fatalf("Metrics().DecodePhases = %+v, want the 7-token budget", got)
	}
}

// --- Capabilities ----------------------------------------------------------

// TestModel_TextModel_Capabilities_Good pins the capability seam: a
// TokenModel declaring cache modes surfaces them on the report's CacheModes —
// the data `generate` consults for the -kv-cache note.
func TestModel_TextModel_Capabilities_Good(t *testing.T) {
	m := &TextModel{tm: cacheModeTokenModel{modes: []string{"native"}}}
	report := m.Capabilities()
	if len(report.CacheModes) != 1 || report.CacheModes[0] != "native" {
		t.Fatalf("CacheModes = %v, want [native]", report.CacheModes)
	}
}

// TestModel_TextModel_Capabilities_Bad pins the absence: a TokenModel without
// the seam leaves CacheModes empty (no invented modes) while the base
// capability set still reports.
func TestModel_TextModel_Capabilities_Bad(t *testing.T) {
	m := &TextModel{tm: bareTokenModel{}}
	report := m.Capabilities()
	if len(report.CacheModes) != 0 {
		t.Fatalf("CacheModes = %v, want empty", report.CacheModes)
	}
	if len(report.Capabilities) == 0 {
		t.Fatal("base capability set is empty; expected generate/chat/classify")
	}
}

// TestModel_TextModel_Capabilities_Ugly pins the no-engine-attached edge: a
// TextModel with tm entirely nil still returns the base capability set
// rather than panicking on the CacheModeReporter probe.
func TestModel_TextModel_Capabilities_Ugly(t *testing.T) {
	m := &TextModel{}
	report := m.Capabilities()
	if len(report.CacheModes) != 0 {
		t.Fatalf("CacheModes = %v, want empty with no engine attached", report.CacheModes)
	}
	if len(report.Capabilities) == 0 {
		t.Fatal("base capability set is empty even with no engine attached")
	}
}

// --- Err -------------------------------------------------------------------

// TestModel_TextModel_Err_Good pins the OK readback after setOK.
func TestModel_TextModel_Err_Good(t *testing.T) {
	m := &TextModel{}
	m.setOK()
	if r := m.Err(); !r.OK {
		t.Fatalf("Err() = %+v, want OK after setOK", r)
	}
}

// TestModel_TextModel_Err_Bad pins the zero-value edge: a bare struct literal
// (never run through NewTextModel) starts in a failing-looking state until
// the first real operation sets it explicitly.
func TestModel_TextModel_Err_Bad(t *testing.T) {
	m := &TextModel{}
	if r := m.Err(); r.OK {
		t.Fatal("want a failure-shaped zero Result on a bare TextModel literal")
	}
}

// TestModel_TextModel_Err_Ugly pins the failure/recovery transition: setOK
// clears a prior setErr rather than layering onto it.
func TestModel_TextModel_Err_Ugly(t *testing.T) {
	m := &TextModel{}
	m.setErr(core.NewError("boom"))
	if r := m.Err(); r.OK {
		t.Fatal("want a failure Result after setErr")
	}
	m.setOK()
	if r := m.Err(); !r.OK {
		t.Fatal("want setOK to clear a prior failure")
	}
}

// --- Close -------------------------------------------------------------

// TestModel_TextModel_Close_Good pins the delegation: Close releases the
// engine's resident weights exactly once.
func TestModel_TextModel_Close_Good(t *testing.T) {
	tm := &fakeTokenModel{}
	m := NewTextModel(tm, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
	if r := m.Close(); !r.OK {
		t.Fatalf("Close() = %+v, want OK", r)
	}
	if tm.closeCalls != 1 {
		t.Fatalf("tm.Close called %d times, want 1", tm.closeCalls)
	}
}

// TestModel_TextModel_Close_Bad pins engine-close failure propagation.
func TestModel_TextModel_Close_Bad(t *testing.T) {
	tm := &fakeTokenModel{closeErr: core.NewError("unmap failed")}
	m := NewTextModel(tm, newFixtureTokenizer(t), "gemma-test", inference.ModelInfo{}, 4096)
	if r := m.Close(); r.OK {
		t.Fatal("want a failure Result when the engine's Close fails")
	}
}

// TestModel_TextModel_Close_Ugly pins the no-engine-attached edge: Close on a
// model with no engine is a clean no-op rather than a panic.
func TestModel_TextModel_Close_Ugly(t *testing.T) {
	m := &TextModel{}
	if r := m.Close(); !r.OK {
		t.Fatalf("Close() on a model with no engine attached = %+v, want a clean no-op OK", r)
	}
}

// --- unexported helpers ------------------------------------------------

// spTok builds a decode-only SentencePiece-style tokenizer: ▁-led pieces are
// word-leading (the marker IS the space), bare pieces are continuations. Its
// empty vocab also makes Encode() return no tokens for any input — the fixture
// the empty-prompt-after-tokenisation tests above rely on.
func spTok() *tokenizer.Tokenizer {
	return tokenizer.NewForDecode(map[int32]string{
		1: "▁hello",
		2: "▁world",
		3: "!",
		4: "hello",
	})
}

// TestTextModelDecodeKeepsWordBoundarySpace pins the STREAMING decode contract:
// concatenating per-token decode output must reassemble the words WITH their
// boundary spaces. The 2026-07-05 regression served "helloworld" for every
// reply because the stream decoded through DecodeOne, whose Decode-of-one
// semantics strip the ▁ boundary.
func TestTextModelDecodeKeepsWordBoundarySpace(t *testing.T) {
	m := &TextModel{tok: spTok()}
	var got string
	for _, id := range []int32{1, 2, 3} {
		got += m.decode(id)
	}
	if want := " hello world!"; got != want {
		t.Fatalf("streamed concat = %q, want %q", got, want)
	}
	if c := m.decode(4); c != "hello" {
		t.Fatalf("continuation piece = %q, want %q (no invented space)", c, "hello")
	}
}

// TestTextModelDecodeLabelStripsBoundary pins the classification contract:
// a standalone label token decodes clean, boundary space stripped — the
// Decode([]int32{id}) semantics classify wants ("▁world" → "world").
func TestTextModelDecodeLabelStripsBoundary(t *testing.T) {
	m := &TextModel{tok: spTok()}
	if got := m.decodeLabel(2); got != "world" {
		t.Fatalf("decodeLabel = %q, want %q", got, "world")
	}
}

// TestTextModelDecodeNilSafe pins the nil-model / nil-tokenizer guards both
// decode variants share.
func TestTextModelDecodeNilSafe(t *testing.T) {
	var nilModel *TextModel
	if got := nilModel.decode(1); got != "" {
		t.Fatalf("nil model decode = %q, want empty", got)
	}
	m := &TextModel{}
	if got := m.decodeLabel(1); got != "" {
		t.Fatalf("nil tok decodeLabel = %q, want empty", got)
	}
}

// TestStopTokens pins the stop-set assembly: caller-supplied StopTokens plus
// the tokenizer's EOS (when it declares one) — never invented, never dropped.
func TestStopTokens(t *testing.T) {
	m := &TextModel{tok: newFixtureTokenizer(t)}
	got := m.stopTokens(inference.GenerateConfig{StopTokens: []int32{99}})
	if len(got) != 2 || got[0] != 99 || got[1] != 2 {
		t.Fatalf("stopTokens = %v, want [99 2] (2 is the fixture tokenizer's <eos>)", got)
	}
}

// TestStopTokensNoTokenizer pins the nil-tokenizer edge: with no tokenizer to
// ask, only the caller-supplied stop tokens are honoured.
func TestStopTokensNoTokenizer(t *testing.T) {
	m := &TextModel{}
	got := m.stopTokens(inference.GenerateConfig{StopTokens: []int32{5}})
	if len(got) != 1 || got[0] != 5 {
		t.Fatalf("stopTokens = %v, want [5]", got)
	}
}

// TestTokenInSet pins the linear stop-token membership check.
func TestTokenInSet(t *testing.T) {
	set := []int32{1, 2, 3}
	if !tokenInSet(2, set) {
		t.Fatal("tokenInSet(2) = false, want true")
	}
	if tokenInSet(9, set) {
		t.Fatal("tokenInSet(9) = true, want false")
	}
	if tokenInSet(1, nil) {
		t.Fatal("tokenInSet against an empty set = true, want false")
	}
}

// TestChatTurnRole pins the gemma role normalisation: "assistant" and "model"
// both render as the model turn; anything else (including empty) is a user
// turn.
func TestChatTurnRole(t *testing.T) {
	cases := map[string]string{"assistant": "model", "model": "model", "user": "user", "system": "user", "": "user"}
	for in, want := range cases {
		if got := chatTurnRole(in); got != want {
			t.Fatalf("chatTurnRole(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestPrefillSplit pins the coarse duration accessor: it reflects elapsed
// wall time since start, never negative.
func TestPrefillSplit(t *testing.T) {
	m := &TextModel{}
	start := time.Now()
	if d := m.prefillSplit(start); d < 0 {
		t.Fatalf("prefillSplit = %v, want >= 0", d)
	}
}

// gemma4FixtureTokenizerJSON is fixtureTokenizerJSON plus the gemma4 turn
// markers (<|turn>/<turn|> — gemma4 renamed the turn tokens and dropped
// <start_of_turn> from its vocab), so dialect detection + framing run the
// production LoadTokenizer path.
const gemma4FixtureTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {"z": 42},
    "merges": []
  },
  "added_tokens": [
    {"id": 1, "content": "<bos>", "special": true},
    {"id": 2, "content": "<eos>", "special": true},
    {"id": 105, "content": "<|turn>", "special": true},
    {"id": 106, "content": "<turn|>", "special": true}
  ]
}`

func newGemma4FixtureTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	if err := coreio.Local.Write(path, gemma4FixtureTokenizerJSON); err != nil {
		t.Fatalf("write gemma4 fixture tokenizer: %v", err)
	}
	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		t.Fatalf("load gemma4 fixture tokenizer: %v", err)
	}
	return tok
}

// TestModel_DetectTurnTokens_Good pins the gemma4 dialect pick: a vocab
// carrying <|turn> selects the renamed turn markers.
func TestModel_DetectTurnTokens_Good(t *testing.T) {
	turns := DetectTurnTokens(newGemma4FixtureTokenizer(t))
	if turns.Open != "<|turn>" || turns.Close != "<turn|>" {
		t.Fatalf("gemma4 vocab detected %+v, want <|turn>/<turn|>", turns)
	}
}

// TestModel_DetectTurnTokens_Bad pins the legacy fallback: a vocab without
// <|turn> keeps the <start_of_turn> template.
func TestModel_DetectTurnTokens_Bad(t *testing.T) {
	turns := DetectTurnTokens(newFixtureTokenizer(t))
	if turns.Open != "<start_of_turn>" || turns.Close != "<end_of_turn>" {
		t.Fatalf("legacy vocab detected %+v, want <start_of_turn>/<end_of_turn>", turns)
	}
}

// TestModel_DetectTurnTokens_Ugly pins the nil-tokenizer edge: detection
// degrades to the legacy template rather than empty markers.
func TestModel_DetectTurnTokens_Ugly(t *testing.T) {
	turns := DetectTurnTokens(nil)
	if turns.Open != "<start_of_turn>" || turns.Close != "<end_of_turn>" {
		t.Fatalf("nil tokenizer detected %+v, want the legacy template", turns)
	}
}

// TestModel_TextModel_FormatChatPrompt_Gemma4Dialect pins the framing a
// gemma4-vocab model serves with: the renamed markers, byte-for-byte the
// template the checkpoint was tuned on (rendering <start_of_turn> against a
// gemma4 vocab tokenises as plain text and measurably damages replies).
func TestModel_TextModel_FormatChatPrompt_Gemma4Dialect(t *testing.T) {
	m := NewTextModel(nil, newGemma4FixtureTokenizer(t), "gemma4", inference.ModelInfo{}, 8)
	got := m.FormatChatPrompt([]inference.Message{{Role: "user", Content: "hi"}})
	want := "<|turn>user\nhi<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("FormatChatPrompt = %q, want %q", got, want)
	}
	if cont := m.FormatChatContinuation([]inference.Message{{Role: "user", Content: "go"}}); cont != "<turn|>\n<|turn>user\ngo<turn|>\n<|turn>model\n" {
		t.Fatalf("FormatChatContinuation = %q, want the <turn|>-closed continuation", cont)
	}
}

// TestModel_TextModel_StopTokens_TurnClose pins the stop set: the model's
// turn-close id joins <eos> (gemma4 tuned models end assistant turns with
// <turn|>, not <eos> — without it a chat reply runs to the token budget).
func TestModel_TextModel_StopTokens_TurnClose(t *testing.T) {
	tok := newGemma4FixtureTokenizer(t)
	m := NewTextModel(nil, tok, "gemma4", inference.ModelInfo{}, 8)
	stop := m.stopTokens(inference.GenerateConfig{})
	if !tokenInSet(106, stop) {
		t.Fatalf("stop set %v missing the <turn|> id 106", stop)
	}
	if !tokenInSet(tok.EOS(), stop) {
		t.Fatalf("stop set %v missing the <eos> id", stop)
	}
	legacy := NewTextModel(nil, newFixtureTokenizer(t), "x", inference.ModelInfo{}, 8)
	if got := legacy.stopTokens(inference.GenerateConfig{}); tokenInSet(106, got) {
		t.Fatalf("legacy vocab stop set %v gained id 106 without a <turn|> token", got)
	}
}

// TestModel_FormatChatPrompt_ThinkPrelude pins the gemma4 thinking switch:
// enabled thinking renders the <|turn>system\n<|think|>\n prelude the
// checkpoint's chat_template.jinja injects — the trained mechanism a request
// turns reasoning on with (byte-for-byte the HF apply_chat_template render).
func TestModel_FormatChatPrompt_ThinkPrelude(t *testing.T) {
	turns := TurnTokens{Open: "<|turn>", Close: "<turn|>"}
	on := true
	got := formatChatPrompt(turns, []inference.Message{{Role: "user", Content: "Hi"}}, &on, false)
	want := "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("thinking prelude = %q, want %q", got, want)
	}
}

// TestModel_FormatChatPrompt_SystemTurn pins the first-system-message framing:
// gemma4 renders it as the leading <|turn>system turn (with the think marker
// first when thinking is on), never as a user turn.
func TestModel_FormatChatPrompt_SystemTurn(t *testing.T) {
	turns := TurnTokens{Open: "<|turn>", Close: "<turn|>"}
	msgs := []inference.Message{{Role: "system", Content: "Be terse."}, {Role: "user", Content: "Hi"}}
	if got, want := formatChatPrompt(turns, msgs, nil, false),
		"<|turn>system\nBe terse.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"; got != want {
		t.Fatalf("system turn = %q, want %q", got, want)
	}
	on := true
	if got, want := formatChatPrompt(turns, msgs, &on, false),
		"<|turn>system\n<|think|>\nBe terse.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"; got != want {
		t.Fatalf("system+think turn = %q, want %q", got, want)
	}
}

// TestModel_FormatChatPrompt_NoPreludeDefault pins the default paths: thinking
// unset (or off) with no system message renders the plain turn history on
// gemma4, and the legacy dialect NEVER gains a system turn or think marker
// (gemma3-era templates have neither).
func TestModel_FormatChatPrompt_NoPreludeDefault(t *testing.T) {
	gemma4 := TurnTokens{Open: "<|turn>", Close: "<turn|>"}
	msgs := []inference.Message{{Role: "user", Content: "Hi"}}
	off := false
	if got, want := formatChatPrompt(gemma4, msgs, &off, false), formatChatTurns(gemma4, msgs); got != want {
		t.Fatalf("thinking-off prompt = %q, want the plain turns %q", got, want)
	}
	legacy := TurnTokens{Open: "<start_of_turn>", Close: "<end_of_turn>"}
	on := true
	sysMsgs := []inference.Message{{Role: "system", Content: "S"}, {Role: "user", Content: "Hi"}}
	if got, want := formatChatPrompt(legacy, sysMsgs, &on, false), formatChatTurns(legacy, sysMsgs); got != want {
		t.Fatalf("legacy prompt = %q, want the plain turns %q (no system/think concept)", got, want)
	}
	if got, want := formatChatPrompt(legacy, sysMsgs, nil, true), formatChatTurns(legacy, sysMsgs); got != want {
		t.Fatalf("legacy suppressor prompt = %q, want the plain turns %q (legacy vocabs have no channel markers)", got, want)
	}
}

// TestModel_FormatChatPrompt_GhostSuppressor pins the large-variant generation
// cue: with thinking off, a ThoughtSuppressorDeclarer checkpoint's template
// pre-closes an empty thought channel after the trailing model turn —
// byte-for-byte the 12B/26B/31B chat_template.jinja branch
// `{%- if not enable_thinking -%}{{- '<|channel>thought\n<channel|>' -}}` —
// and with thinking ON the suffix must NOT render.
func TestModel_FormatChatPrompt_GhostSuppressor(t *testing.T) {
	turns := TurnTokens{Open: "<|turn>", Close: "<turn|>"}
	msgs := []inference.Message{{Role: "user", Content: "Hi"}}
	want := "<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
	if got := formatChatPrompt(turns, msgs, nil, true); got != want {
		t.Fatalf("suppressor thinking-off = %q, want %q", got, want)
	}
	off := false
	if got := formatChatPrompt(turns, msgs, &off, true); got != want {
		t.Fatalf("suppressor explicit-off = %q, want %q", got, want)
	}
	sysMsgs := []inference.Message{{Role: "system", Content: "Be terse."}, {Role: "user", Content: "Hi"}}
	wantSys := "<|turn>system\nBe terse.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
	if got := formatChatPrompt(turns, sysMsgs, nil, true); got != wantSys {
		t.Fatalf("suppressor system-turn = %q, want %q", got, wantSys)
	}
	on := true
	if got, want := formatChatPrompt(turns, msgs, &on, true),
		"<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"; got != want {
		t.Fatalf("suppressor thinking-on = %q, want no pre-closed channel %q", got, want)
	}
}

// thoughtSuppressorTokenModel is a fake TokenModel that declares the
// pre-closed-thought-channel template capability
// (engine.ThoughtSuppressorDeclarer), for the capability-fold test.
type thoughtSuppressorTokenModel struct {
	TokenModel
	suppressor bool
}

func (s thoughtSuppressorTokenModel) NeedsThoughtChannelSuppressor() bool { return s.suppressor }

// TestModel_NewTextModel_ThoughtSuppressorDeclared pins the capability fold: a
// TokenModel declaring the thought-channel suppressor renders the pre-closed
// channel through TextModel.FormatChatPrompt, a non-declaring one (or a
// declarer answering false) renders the plain generation cue.
func TestModel_NewTextModel_ThoughtSuppressorDeclared(t *testing.T) {
	tok := newGemma4FixtureTokenizer(t)
	msgs := []inference.Message{{Role: "user", Content: "Hi"}}
	declared := NewTextModel(thoughtSuppressorTokenModel{suppressor: true}, tok, "x", inference.ModelInfo{}, 8)
	if got := declared.FormatChatPrompt(msgs); !strings.HasSuffix(got, "<|turn>model\n<|channel>thought\n<channel|>") {
		t.Fatalf("declared suppressor prompt = %q, want the pre-closed thought channel suffix", got)
	}
	small := NewTextModel(thoughtSuppressorTokenModel{suppressor: false}, tok, "x", inference.ModelInfo{}, 8)
	if got := small.FormatChatPrompt(msgs); !strings.HasSuffix(got, "<|turn>model\n") {
		t.Fatalf("false-declarer prompt = %q, want the plain generation cue", got)
	}
	undeclared := NewTextModel(nil, tok, "x", inference.ModelInfo{}, 8)
	if got := undeclared.FormatChatPrompt(msgs); !strings.HasSuffix(got, "<|turn>model\n") {
		t.Fatalf("undeclared prompt = %q, want the plain generation cue", got)
	}
}

// stopDeclarerTokenModel is a fake TokenModel that declares a checkpoint stop
// set (engine.StopTokenDeclarer), for the stop-fold test.
type stopDeclarerTokenModel struct {
	TokenModel
	stops []int32
}

func (s stopDeclarerTokenModel) DeclaredStopTokens() []int32 { return s.stops }

// TestModel_TextModel_StopTokens_Declared pins the generation_config fold: a
// TokenModel declaring its checkpoint stop set contributes every id exactly
// once alongside the derived <eos> + turn-close defaults.
func TestModel_TextModel_StopTokens_Declared(t *testing.T) {
	tok := newGemma4FixtureTokenizer(t)
	m := NewTextModel(stopDeclarerTokenModel{stops: []int32{2, 106, 50}}, tok, "gemma4", inference.ModelInfo{}, 8)
	stop := m.stopTokens(inference.GenerateConfig{})
	for _, id := range []int32{2, 106, 50} {
		if !tokenInSet(id, stop) {
			t.Fatalf("stop set %v missing declared id %d", stop, id)
		}
	}
	seen := 0
	for _, id := range stop {
		if id == 106 {
			seen++
		}
	}
	if seen != 1 {
		t.Fatalf("stop set %v holds id 106 %d times, want exactly once", stop, seen)
	}
}

// TestModel_TextModel_NilReceiverGuards pins that every read-only accessor
// tolerates a nil *TextModel the way Close and Capabilities already do —
// zero values (or a clear failure Result from Err), never a panic.
func TestModel_TextModel_NilReceiverGuards(t *testing.T) {
	var m *TextModel
	if got := m.ModelType(); got != "" {
		t.Fatalf("nil ModelType = %q, want empty", got)
	}
	if got := m.Info(); got != (inference.ModelInfo{}) {
		t.Fatalf("nil Info = %+v, want zero", got)
	}
	if got := m.Metrics(); got.GeneratedTokens != 0 || got.PromptTokens != 0 {
		t.Fatalf("nil Metrics = %+v, want zero", got)
	}
	if r := m.Err(); r.OK {
		t.Fatal("nil Err should report a failure Result")
	}
	if s := m.NewSession(); s != nil {
		t.Fatal("nil NewSession should return nil")
	}
}
