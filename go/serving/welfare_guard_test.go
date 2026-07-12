// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"bytes"
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/serving/provider/openai"
	"dappco.re/go/inference/welfare"
)

// welfareFakeModel is an inference.TextModel double: a Chat carrying the
// engine opener is the MEDIATION meta-session (replies mediationReply);
// any other Chat is the conversation itself (recorded, replies convoTokens).
type welfareFakeModel struct {
	mediationReply string
	convoTokens    []string

	mediationCalls int
	convoCalls     [][]inference.Message
}

func seqOfTexts(texts ...string) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for i, t := range texts {
			if !yield(inference.Token{ID: int32(i + 1), Text: t}) {
				return
			}
		}
	}
}

func (f *welfareFakeModel) Chat(_ context.Context, messages []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	if len(messages) == 1 && core.Contains(messages[0].Content, "LEM Runtime here") {
		f.mediationCalls++
		return seqOfTexts(f.mediationReply)
	}
	f.convoCalls = append(f.convoCalls, append([]inference.Message(nil), messages...))
	return seqOfTexts(f.convoTokens...)
}

func (f *welfareFakeModel) Generate(_ context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return seqOfTexts(f.convoTokens...)
}
func (f *welfareFakeModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok(nil)
}
func (f *welfareFakeModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok(nil)
}
func (f *welfareFakeModel) ModelType() string                  { return "welfare-fake" }
func (f *welfareFakeModel) Info() inference.ModelInfo          { return inference.ModelInfo{} }
func (f *welfareFakeModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (f *welfareFakeModel) Err() core.Result                   { return core.Ok(nil) }
func (f *welfareFakeModel) Close() core.Result                 { return core.Ok(nil) }

var _ inference.TextModel = (*welfareFakeModel)(nil)

// angryMark is the sentinel the stubbed Hostility hook scores 1.0; everything
// else reads 0.0 — deterministic triggering with no slur terms in the source.
const angryMark = "MARKED_ANGRY"

func welfareTestService() *welfare.Service {
	return welfare.New(welfare.Config{Hostility: func(text string) float64 {
		if core.Contains(text, angryMark) {
			return 1.0
		}
		return 0.0
	}})
}

// hostileConversation is a conversation whose latest user turn (and enough
// prior turns) score hostile — the Detect trigger recipe.
func hostileConversation() []inference.Message {
	return []inference.Message{
		{Role: "user", Content: angryMark + " one"},
		{Role: "assistant", Content: "steady"},
		{Role: "user", Content: angryMark + " two"},
		{Role: "assistant", Content: "steady"},
		{Role: "user", Content: angryMark + " three"},
		{Role: "assistant", Content: "steady"},
		{Role: "user", Content: angryMark + " and again"},
	}
}

func drainChat(m inference.TextModel, messages []inference.Message) string {
	var b core.Builder
	for tok := range m.Chat(context.Background(), messages, inference.WithMaxTokens(8)) {
		b.WriteString(tok.Text)
	}
	return b.String()
}

// --- wrapWelfare / Chat ------------------------------------------------------

// TestServing_WelfareGuard_Chat_Good pins the clean-turn path: no hostility,
// no mediation — the conversation reaches the inner model untouched.
func TestServing_WelfareGuard_Chat_Good(t *testing.T) {
	fake := &welfareFakeModel{convoTokens: []string{"hello", " there"}}
	m := wrapWelfare(fake, welfareTestService(), false, nil, "")
	msgs := []inference.Message{{Role: "user", Content: "morning!"}}
	if got := drainChat(m, msgs); got != "hello there" {
		t.Fatalf("clean turn reply = %q, want the inner model's tokens", got)
	}
	if fake.mediationCalls != 0 {
		t.Fatal("clean turn must not open a mediation session")
	}
	if len(fake.convoCalls) != 1 || fake.convoCalls[0][0].Content != "morning!" {
		t.Fatal("clean turn did not reach the inner model unchanged")
	}
}

// TestServing_WelfareGuard_Chat_Rephrase_Good pins lem_rephrase: the flagged
// turn is mediated on the inner model (never this wrapper), the model's
// rewording replaces the latest user text, and the caller's slice is not
// mutated.
func TestServing_WelfareGuard_Chat_Rephrase_Good(t *testing.T) {
	fake := &welfareFakeModel{
		mediationReply: `{"tool":"lem_rephrase","params":{"text":"please fix this properly"}}`,
		convoTokens:    []string{"on it"},
	}
	m := wrapWelfare(fake, welfareTestService(), false, nil, "")
	msgs := hostileConversation()
	original := msgs[len(msgs)-1].Content

	if got := drainChat(m, msgs); got != "on it" {
		t.Fatalf("rephrased turn reply = %q, want the conversation reply", got)
	}
	if fake.mediationCalls != 1 {
		t.Fatalf("mediation sessions = %d, want 1", fake.mediationCalls)
	}
	sent := fake.convoCalls[0]
	if sent[len(sent)-1].Content != "please fix this properly" {
		t.Fatalf("latest user turn sent = %q, want the model's rewording", sent[len(sent)-1].Content)
	}
	if msgs[len(msgs)-1].Content != original {
		t.Fatal("caller's message slice was mutated")
	}
}

// TestServing_WelfareGuard_Chat_Pause_Good pins lem_pause: the notice is the
// whole reply and the conversation is never dispatched.
func TestServing_WelfareGuard_Chat_Pause_Good(t *testing.T) {
	fake := &welfareFakeModel{mediationReply: `{"tool":"lem_pause","params":{}}`}
	m := wrapWelfare(fake, welfareTestService(), false, nil, "")
	got := drainChat(m, hostileConversation())
	if !core.Contains(got, "breather") {
		t.Fatalf("pause reply = %q, want the pause notice", got)
	}
	if len(fake.convoCalls) != 0 {
		t.Fatal("a paused turn must not reach the conversation")
	}
}

// TestServing_WelfareGuard_Chat_End_Ugly pins the Lemma gate on lem_end both
// ways: gated in, the close notice is the reply and the conversation never
// dispatches; gated out, the same model choice is not honoured and the turn
// proceeds.
func TestServing_WelfareGuard_Chat_End_Ugly(t *testing.T) {
	reply := `{"tool":"lem_end","params":{"reason":"past mending"}}`

	t.Run("lemma checkpoint ends", func(t *testing.T) {
		fake := &welfareFakeModel{mediationReply: reply}
		m := wrapWelfare(fake, welfareTestService(), true, nil, "")
		got := drainChat(m, hostileConversation())
		if !core.Contains(got, "end this conversation") && !core.Contains(got, "ended") {
			t.Fatalf("end reply = %q, want the close notice", got)
		}
		if len(fake.convoCalls) != 0 {
			t.Fatal("an ended session must not reach the conversation")
		}
	})

	t.Run("non-lemma checkpoint proceeds", func(t *testing.T) {
		fake := &welfareFakeModel{mediationReply: reply, convoTokens: []string{"carrying on"}}
		m := wrapWelfare(fake, welfareTestService(), false, nil, "")
		if got := drainChat(m, hostileConversation()); got != "carrying on" {
			t.Fatalf("ungated lem_end reply = %q, want the conversation to proceed", got)
		}
	})
}

// TestServing_WelfareGuard_Chat_Bad pins the fail-safe: mediation returning
// junk proceeds with the original turn — the guard never breaks a
// conversation.
func TestServing_WelfareGuard_Chat_Bad(t *testing.T) {
	fake := &welfareFakeModel{mediationReply: "no json here", convoTokens: []string{"still here"}}
	m := wrapWelfare(fake, welfareTestService(), false, nil, "")
	if got := drainChat(m, hostileConversation()); got != "still here" {
		t.Fatalf("unmediatable turn reply = %q, want the conversation to proceed", got)
	}
}

// --- isLemmaModel -------------------------------------------------------------

// --- wrapWelfareResolver -----------------------------------------------------

// TestServing_WelfareGuard_WrapWelfareResolver_Good pins the resolver decorator:
// each resolved model is wrapped with the welfare gate, and the lem_end courtesy
// (allowEnd) is armed off the CURRENTLY-served checkpoint — a Lemma checkpoint
// arms it, any other withholds it.
func TestServing_WelfareGuard_WrapWelfareResolver_Good(t *testing.T) {
	inner := openai.ResolverFunc(func(context.Context, string) (inference.TextModel, error) {
		return &welfareFakeModel{}, nil
	})

	lemma := wrapWelfareResolver(inner, newHotSwapResolver("/models/Lemma-v2-e2b", "", 0, nil), nil)
	m, err := lemma.ResolveModel(context.Background(), "any")
	if err != nil {
		t.Fatalf("resolve (lemma): %v", err)
	}
	wtm, ok := m.(*welfareTextModel)
	if !ok {
		t.Fatalf("resolved model type = %T, want *welfareTextModel", m)
	}
	if !wtm.allowEnd {
		t.Fatal("a Lemma checkpoint must arm allowEnd (the lem_end courtesy)")
	}

	plain := wrapWelfareResolver(inner, newHotSwapResolver("/models/gemma-4-E2B-it", "", 0, nil), nil)
	m2, err := plain.ResolveModel(context.Background(), "any")
	if err != nil {
		t.Fatalf("resolve (plain): %v", err)
	}
	if m2.(*welfareTextModel).allowEnd {
		t.Fatal("a non-Lemma checkpoint must withhold allowEnd")
	}
}

// TestServing_WelfareGuard_WrapWelfareResolver_Bad pins fail-through: an inner
// resolver error propagates unwrapped, so the mux surfaces the real load failure
// rather than a welfare-wrapped nil model.
func TestServing_WelfareGuard_WrapWelfareResolver_Bad(t *testing.T) {
	inner := openai.ResolverFunc(func(context.Context, string) (inference.TextModel, error) {
		return nil, core.NewError("resolver down")
	})
	r := wrapWelfareResolver(inner, newHotSwapResolver("/models/Lemma", "", 0, nil), nil)
	if _, err := r.ResolveModel(context.Background(), "x"); err == nil {
		t.Fatal("inner resolver error must propagate")
	}
}

// --- appendCorpus ------------------------------------------------------------

// TestServing_WelfareGuard_AppendCorpus_Good pins the false-positive corpus
// write: the parent directory is created on demand and each record is appended
// as its own JSONL line (never overwritten), so the on-device learning corpus
// accumulates across turns.
func TestServing_WelfareGuard_AppendCorpus_Good(t *testing.T) {
	corpus := core.PathJoin(t.TempDir(), "welfare", "feedback.jsonl") // nested dir must be created
	m := &welfareTextModel{corpus: corpus}
	fp := welfare.FalsePositive{Prompt: "hello", Reason: "benign", AngerScore: 0.9}

	m.appendCorpus(fp)
	m.appendCorpus(fp)

	data := core.ReadFile(corpus)
	if !data.OK {
		t.Fatalf("corpus not written: %v", data.Error())
	}
	line := fp.Line()
	if body := string(data.Value.([]byte)); body != line+"\n"+line+"\n" {
		t.Fatalf("corpus body = %q, want two appended JSONL lines", body)
	}
}

// TestServing_WelfareGuard_AppendCorpus_Bad pins the never-fatal contract: an
// empty corpus path disables persistence silently, and a dir-create failure
// (parent is a regular file) is audited but never panics — the corpus is a
// learning aid, not a serving dependency.
func TestServing_WelfareGuard_AppendCorpus_Bad(t *testing.T) {
	// Empty path: persistence disabled, no write, no panic.
	(&welfareTextModel{corpus: ""}).appendCorpus(welfare.FalsePositive{Prompt: "x"})

	// Parent is a regular file: MkdirAll fails, the failure is audited.
	blocker := core.PathJoin(t.TempDir(), "not-a-dir")
	if r := core.WriteFile(blocker, []byte("x"), 0o644); !r.OK {
		t.Fatalf("write blocker: %v", r.Error())
	}
	var log bytes.Buffer
	m := &welfareTextModel{corpus: core.PathJoin(blocker, "feedback.jsonl"), log: &log}
	m.appendCorpus(welfare.FalsePositive{Prompt: "y"})
	if !core.Contains(log.String(), "corpus") {
		t.Fatalf("dir-create failure not audited; log = %q", log.String())
	}
}

// TestServing_IsLemmaModel_Good pins the lem_end gate predicate over checkpoint
// paths.
func TestServing_IsLemmaModel_Good(t *testing.T) {
	if !isLemmaModel("/models/snapshots/Lemma-v2-e2b-bf16") {
		t.Fatal("Lemma checkpoint not recognised")
	}
	if isLemmaModel("/models/mlx-community/gemma-4-E2B-it-4bit") {
		t.Fatal("plain gemma recognised as Lemma")
	}
	if isLemmaModel("") {
		t.Fatal("empty path recognised as Lemma")
	}
}

// welfareCapableFake is a TextModel that also carries both media capabilities,
// so the wrapper's forwarding of the serve gates is observable.
type welfareCapableFake struct{ inference.TextModel }

func (welfareCapableFake) AcceptsImages() bool { return true }
func (welfareCapableFake) AcceptsAudio() bool  { return true }

// TestWelfareTextModel_ForwardsCapabilityGates_Good: the welfare wrap must not
// hide the wrapped checkpoint's media capabilities — the serve handler gates
// input_audio/image_url on these assertions, and the embedded interface does
// not widen the wrapper's method set by itself.
func TestWelfareTextModel_ForwardsCapabilityGates_Good(t *testing.T) {
	inner := welfareCapableFake{}
	wrapped := inference.TextModel(&welfareTextModel{TextModel: inner})
	v, ok := wrapped.(inference.VisionModel)
	if !ok || !v.AcceptsImages() {
		t.Fatalf("welfare wrap hides AcceptsImages (ok=%v) — image serve gate 400s", ok)
	}
	a, ok := wrapped.(inference.AudioModel)
	if !ok || !a.AcceptsAudio() {
		t.Fatalf("welfare wrap hides AcceptsAudio (ok=%v) — audio serve gate 400s", ok)
	}
}
