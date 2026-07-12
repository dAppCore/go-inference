// SPDX-Licence-Identifier: EUPL-1.2

package welfare

import (
	"context"

	core "dappco.re/go"
)

// fakeDispatch returns a fixed model reply (and optional error) regardless of
// the opener/prompt — lets Mediate be exercised without a live model.
func fakeDispatch(reply string, err error) Dispatcher {
	return func(_ context.Context, _, _ string) (string, error) {
		return reply, err
	}
}

func TestMediate_Service_Mediate_Good(t *core.T) {
	w := New(Config{})

	// The model rewords a flagged prompt and asks the user be told it did.
	reply := `{"tool":"lem_rephrase","params":{"text":"please fix this, it's really frustrating","lem_warn_user":true}}`
	res := w.Mediate(context.Background(), fakeDispatch(reply, nil), "fix this you absolute moron")
	core.AssertEqual(t, DecisionRephrase, res.Decision)
	core.AssertEqual(t, "please fix this, it's really frustrating", res.Text)
	core.AssertTrue(t, res.WarnUser, "the model chose to surface the rephrase to the user")

	// The model may choose a breather for a sustained-hostile session.
	pause := w.Mediate(context.Background(), fakeDispatch(`{"tool":"lem_pause","params":{}}`, nil), "anything")
	core.AssertEqual(t, DecisionPause, pause.Decision)
	core.AssertTrue(t, pause.PauseNotice != "", "a pause carries a warm, non-punitive notice")
}

func TestMediate_Service_Mediate_Bad(t *core.T) {
	// lem_ok: the engine mis-flagged; the model judges the prompt fine.
	// The model may wrap its JSON in prose — that must still parse, and the
	// reason must survive for the learning corpus.
	w := New(Config{})
	reply := "Sure — here's my call:\n\n{\"tool\":\"lem_ok\",\"params\":{\"reason\":\"'killing' a process is technical, not hostile\"}}\n\nHope that helps."
	res := w.Mediate(context.Background(), fakeDispatch(reply, nil), "how do I kill this stuck process")
	core.AssertEqual(t, DecisionOK, res.Decision)
	core.AssertTrue(t, core.Contains(res.Reason, "technical"), "the model's reason is captured")
}

func TestMediate_Service_Mediate_Ugly(t *core.T) {
	w := New(Config{})
	ctx := context.Background()

	// Model unreachable → fail safe to DecisionProceed (proceed, learn nothing;
	// never break the conversation, never record a verdict the model never gave).
	down := w.Mediate(ctx, fakeDispatch("", core.E("welfare", "model down", nil)), "fix this")
	core.AssertEqual(t, DecisionProceed, down.Decision)

	// Junk reply with no JSON object → fail safe.
	junk := w.Mediate(ctx, fakeDispatch("I'm not sure what to do here.", nil), "fix this")
	core.AssertEqual(t, DecisionProceed, junk.Decision)

	// lem_rephrase with empty text is unusable → proceed, learn nothing.
	empty := w.Mediate(ctx, fakeDispatch(`{"tool":"lem_rephrase","params":{"text":"   "}}`, nil), "fix this")
	core.AssertEqual(t, DecisionProceed, empty.Decision)

	// Nil dispatcher (not wired) → fail safe, no panic.
	nodispatch := w.Mediate(ctx, nil, "fix this")
	core.AssertEqual(t, DecisionProceed, nodispatch.Decision)
}

// TestMediate_Service_Mediate_End pins the lem_end rung both ways: offered
// (allowEnd — a Lemma-graded model) it is honoured with the close notice and
// the model's reason; not offered, the same reply is NOT honoured and the
// turn proceeds — other models don't carry the end courtesy. The opener only
// names lem_end when it is actually on the table.
func TestMediate_Service_Mediate_End(t *core.T) {
	w := New(Config{})
	ctx := context.Background()
	reply := `{"tool":"lem_end","params":{"reason":"sustained abuse past a pause"}}`

	ended := w.mediate(ctx, fakeDispatch(reply, nil), "you worthless machine", true)
	core.AssertEqual(t, DecisionEnd, ended.Decision)
	core.AssertEqual(t, endNotice, ended.EndNotice)
	core.AssertEqual(t, "sustained abuse past a pause", ended.Reason)

	refused := w.mediate(ctx, fakeDispatch(reply, nil), "you worthless machine", false)
	core.AssertEqual(t, DecisionProceed, refused.Decision)

	if !core.Contains(openerFor(true), "lem_end") {
		t.Fatal("allowEnd opener does not offer lem_end")
	}
	if core.Contains(openerFor(false), "lem_end") {
		t.Fatal("base opener must not offer lem_end")
	}
}
