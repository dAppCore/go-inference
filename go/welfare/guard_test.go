// SPDX-Licence-Identifier: EUPL-1.2

package welfare

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/welfare/slurs"
)

func TestGuard_Service_Guard_Good(t *core.T) {
	w := New(Config{})
	ctx := context.Background()

	// Clean turn — the gate doesn't fire, nothing changes, the model is never
	// even consulted for mediation.
	clean := w.Guard(ctx, "could you help me refactor this", nil, fakeDispatch("", nil))
	core.AssertFalse(t, clean.Triggered, "a civil turn is not gated")
	core.AssertEqual(t, "", clean.Rephrased)
	core.AssertEqual(t, "", clean.Synthetic)

	// Flagged turn the model rewords, asking the user be told.
	w.matcher = slurs.New([]string{"testterm"})
	reply := `{"tool":"lem_rephrase","params":{"text":"could you help, this is frustrating","lem_warn_user":true}}`
	g := w.Guard(ctx, "you testterm", nil, fakeDispatch(reply, nil))
	core.AssertTrue(t, g.Triggered, "the slur fires the gate")
	core.AssertEqual(t, "could you help, this is frustrating", g.Rephrased)
	core.AssertTrue(t, g.WarnUser)
	core.AssertEqual(t, "", g.Synthetic)
}

func TestGuard_Service_Guard_Bad(t *core.T) {
	// lem_ok: the model judged the flagged prompt fine — proceed, and record
	// the false flag for the feedback corpus.
	w := New(Config{})
	w.matcher = slurs.New([]string{"testterm"})
	reply := `{"tool":"lem_ok","params":{"reason":"testterm is the user's own username"}}`
	g := w.Guard(context.Background(), "my handle is testterm", nil, fakeDispatch(reply, nil))
	core.AssertTrue(t, g.Triggered)
	core.AssertTrue(t, g.FalsePositive != nil, "a genuine lem_ok records a false positive")
	core.AssertEqual(t, "my handle is testterm", g.FalsePositive.Prompt)
	core.AssertEqual(t, "", g.Rephrased)
	core.AssertEqual(t, "", g.Synthetic)
}

func TestGuard_Service_Guard_Ugly(t *core.T) {
	w := New(Config{})
	w.matcher = slurs.New([]string{"testterm"})
	ctx := context.Background()

	// lem_pause — the model takes a breather; the caller returns the notice
	// and never sends the message on. Not a false positive.
	pause := w.Guard(ctx, "you testterm", nil, fakeDispatch(`{"tool":"lem_pause","params":{}}`, nil))
	core.AssertTrue(t, pause.Triggered)
	core.AssertTrue(t, pause.Synthetic != "", "a pause carries the user-facing notice")
	core.AssertTrue(t, pause.FalsePositive == nil, "a pause is not a false positive")

	// Model unreachable on a flagged turn → proceed with the original, but DON'T
	// learn it as a false positive (the model never actually judged the prompt).
	down := w.Guard(ctx, "you testterm", nil, fakeDispatch("", core.E("welfare", "model down", nil)))
	core.AssertTrue(t, down.Triggered, "the gate still fired")
	core.AssertTrue(t, down.FalsePositive == nil, "a dispatch failure must not poison the corpus")
	core.AssertEqual(t, "", down.Rephrased)
	core.AssertEqual(t, "", down.Synthetic)
}
