// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
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
