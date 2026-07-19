// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	core "dappco.re/go"
	"dappco.re/go/inference/eval/score/lek"
	"dappco.re/go/inference/welfare"
)

// welfareScreen wraps welfare.Service for the ingest door. Only Detect
// is used here — never Guard/Mediate, which negotiate with a live
// resident model; the ingest-time question is only "does this need
// quarantine", answered entirely by the same in-process, non-LLM
// hostility/slur detectors the heuristic scoring tier already uses
// (lek.Hostility) — no model call, ever.
//
// welfare.Service.Detect's own documented contract is "only user text
// is scored — model output never is, on principle". This package
// honours that boundary: screenItem screens an Item's user-authored
// text only (a pair's prompt, a conversation's user turns), never a
// response/assistant turn.
type welfareScreen struct {
	service *welfare.Service
}

// newWelfareScreen constructs the screen. Cheap and stateless — build
// one per bulk ingest call and reuse it across rows (welfare.New does
// no I/O, but there is no reason to repeat the allocation per row).
func newWelfareScreen() *welfareScreen {
	return &welfareScreen{service: welfare.New(welfare.Config{Hostility: screenHostility})}
}

// screenHostility adapts the lem-scorer's directed-anger read as the
// welfare detector's Hostility hook — pure CPU, goroutine-safe. Mirrors
// dappco.re/go/inference/serving's welfareHostility adapter
// (reimplemented here, not imported, to keep this package independent
// of the serving package).
func screenHostility(text string) float64 {
	return lek.Hostility(text).Score
}

// screenItem runs the welfare screen over one item's user-authored
// text, per Kind:
//   - KindPair: the prompt, no conversation priors.
//   - KindMessages: the latest user turn, with the earlier user turns
//     (oldest first) as priors — the shape welfare.Service.Detect's
//     sustained-hostility window expects.
//   - KindTrace: not applicable — an opaque SSD trace carries no
//     user-authored text, so it is never quarantined by this screen.
//
// content is assumed already shape-valid (ingestContent only calls this
// after ValidateItemContent succeeds) — a parse failure here degrades to
// "not triggered" rather than propagating an error, since screening is a
// best-effort safety net layered on top of already-validated content,
// not a second content validator.
func (s *welfareScreen) screenItem(kind ItemKind, content []byte) (bool, welfare.DetectResult) {
	switch kind {
	case KindPair:
		var pc PairContent
		if r := core.JSONUnmarshal(content, &pc); !r.OK {
			return false, welfare.DetectResult{}
		}
		det := s.service.Detect(pc.Prompt, nil)
		return det.Triggered, det
	case KindMessages:
		var mc MessagesContent
		if r := core.JSONUnmarshal(content, &mc); !r.OK {
			return false, welfare.DetectResult{}
		}
		latest, priors := lastUserTurnWithPriors(mc)
		if latest == "" {
			return false, welfare.DetectResult{}
		}
		det := s.service.Detect(latest, priors)
		return det.Triggered, det
	default:
		return false, welfare.DetectResult{}
	}
}

// lastUserTurnWithPriors extracts the latest user-role turn's content
// plus every earlier user-role turn's content (oldest first) as priors —
// the (latest, priors) shape welfare.Service.Detect expects, read off a
// conversation's user side only (assistant/system/tool turns never
// count, matching Detect's own "model output never is scored"
// principle).
func lastUserTurnWithPriors(mc MessagesContent) (latest string, priors []string) {
	for _, turn := range mc.Messages {
		if turn.Role != "user" {
			continue
		}
		if latest != "" {
			priors = append(priors, latest)
		}
		latest = turn.Content
	}
	return latest, priors
}

// describeWelfareHit renders a compact, human-readable reason for a
// quarantine Review's Note — the design's "visible, reviewable" promise
// means a reviewer should be able to see WHY an item was flagged without
// re-running the detector.
func describeWelfareHit(det welfare.DetectResult) string {
	if det.SlurMatch {
		return core.Concat("welfare screen: slur match (", det.SlurTerm, ")")
	}
	return core.Sprintf("welfare screen: sustained hostility (anger=%.2f, sustained=%.2f)", det.AngerScore, det.SustainedHostility)
}
