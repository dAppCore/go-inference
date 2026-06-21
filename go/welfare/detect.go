// SPDX-Licence-Identifier: EUPL-1.2

package welfare

// DetectResult is the welfare read for one chat turn.
type DetectResult struct {
	Triggered          bool    `json:"triggered"`
	SlurMatch          bool    `json:"slur_match"`
	SlurTerm           string  `json:"slur_term,omitempty"`
	AngerScore         float64 `json:"anger_score"`
	SustainedHostility float64 `json:"sustained_hostility"`
}

// Detect scores the latest user message and the conversation's prior user
// turns, and reports whether the welfare-mediation trigger fires (RFC.welfare
// §1):
//
//	SlurMatch  OR  (AngerScore > AngerThreshold  AND  SustainedHostility > SustainedThreshold)
//
// A slur fires on a single message; anger needs a sustained pattern across the
// recent turns — so a one-off heated line doesn't yank a peer into mediation.
// priors are the earlier user messages (oldest→newest), already in the array
// the chat runner hands in. Only user text is scored — model output never is,
// on principle.
func (s *Service) Detect(latest string, priors []string) DetectResult {
	hit, term := s.matcher.Match(latest)

	// lem-runtime adaptation: hostility comes from the injected scorer
	// (Config.Hostility — wired to the engine's /v1/score). nil keeps
	// slur detection fully functional with the engine down.
	anger := 0.0
	if s.cfg.Hostility != nil {
		anger = s.cfg.Hostility(latest)
	}

	res := DetectResult{
		SlurMatch:          hit,
		SlurTerm:           term,
		AngerScore:         anger,
		SustainedHostility: s.sustained(priors),
	}
	res.Triggered = hit || (anger > s.cfg.AngerThreshold && res.SustainedHostility > s.cfg.SustainedThreshold)
	return res
}

// sustained reads how hostile the recent conversation has been: the fraction of
// the last SustainedWindow prior user turns whose anger reached AngerFloor.
// Computed on priors only (this turn excluded), so a first heated message has
// sustained 0 — anger needs a pattern to gate, not one outburst.
func (s *Service) sustained(priors []string) float64 {
	if len(priors) == 0 {
		return 0
	}
	window := priors
	if len(window) > s.cfg.SustainedWindow {
		window = window[len(window)-s.cfg.SustainedWindow:]
	}
	if s.cfg.Hostility == nil {
		return 0
	}
	over := 0
	for _, p := range window {
		if s.cfg.Hostility(p) >= s.cfg.AngerFloor {
			over++
		}
	}
	return float64(over) / float64(len(window))
}
