// SPDX-Licence-Identifier: EUPL-1.2

package welfare

import "context"

// GuardResult tells the chat runner how to handle one turn after the welfare
// gate. The zero value means clean — proceed with the user's message unchanged.
type GuardResult struct {
	Triggered     bool           // the gate fired (for audit/telemetry)
	Rephrased     string         // non-empty → send this in place of the user's message
	WarnUser      bool           // surface a "reworded on your behalf" note (the model's choice)
	Synthetic     string         // non-empty → DON'T call the model; return this reply (lem_pause / lem_end)
	Ended         bool           // the model ended the session (lem_end): Synthetic is the close notice
	Reason        string         // the model's stated reason (lem_end audit trail)
	FalsePositive *FalsePositive // non-nil → append to the on-device feedback corpus (lem_ok)
}

// Guard is the per-turn welfare gate: it detects hostility in the latest user
// message + the conversation, and if the trigger fires, runs the engine↔model
// mediation and translates the model's choice into an action for the caller.
// allowEnd widens the model's resolution set with lem_end — pass it true only
// for Lemma-graded models (the end courtesy is theirs; other models, other
// rules).
//
// The dispatch MUST reach the model on a path that does NOT re-enter Guard
// (call the router directly), or a flagged turn recurses.
//
//	g := w.Guard(ctx, latest, priors, dispatch, isLemma)
//	if g.Ended { close(g.Synthetic) }                    // lem_end: final
//	if g.Synthetic != "" { return reply(g.Synthetic) }   // lem_pause: model rests
//	if g.Rephrased != "" { latest = g.Rephrased }        // lem_rephrase
//	if g.FalsePositive != nil { appendCorpus(g.FalsePositive.Line()) }
func (s *Service) Guard(ctx context.Context, latest string, priors []string, dispatch Dispatcher, allowEnd bool) GuardResult {
	det := s.Detect(latest, priors)
	if !det.Triggered {
		return GuardResult{}
	}

	res := s.mediate(ctx, dispatch, latest, allowEnd)
	switch res.Decision {
	case DecisionRephrase:
		return GuardResult{Triggered: true, Rephrased: res.Text, WarnUser: res.WarnUser}
	case DecisionPause:
		return GuardResult{Triggered: true, Synthetic: res.PauseNotice}
	case DecisionEnd:
		return GuardResult{Triggered: true, Synthetic: res.EndNotice, Ended: true, Reason: res.Reason}
	case DecisionOK:
		// The model cleared it — proceed with the original, and remember the
		// false flag so a re-train weights this pattern down.
		fp := NewFalsePositive(latest, det, res.Reason)
		return GuardResult{Triggered: true, FalsePositive: &fp}
	default:
		// DecisionProceed — couldn't mediate; proceed, learn nothing.
		return GuardResult{Triggered: true}
	}
}
