// SPDX-Licence-Identifier: EUPL-1.2

// Package safety is the request-path safety decision for served chats
// (RFC.md §6.18). It sits one layer above pkg/welfare: welfare DETECTS (scores a
// turn — hostility, sustained anger, slurs into a welfare.DetectResult), and
// safety DECIDES what the serving path does with that read — pass the turn,
// guard it (refuse), or mediate it (regenerate under a corrective instruction).
//
// The §6.18 posture is "regenerate, don't just block": an over-policy MODEL
// OUTPUT is preferentially regenerated (Mediate) rather than hard-refused, and
// only escalates to a refusal (Guard) when the read is severe enough. An
// over-policy USER INPUT is guarded — the serving path doesn't rewrite the user.
// A trusted internal key (Policy.Bypass, §6.17) lowers the policy so vetted
// callers serve unguarded.
//
//	dec := safety.Decide(inRead, outRead, safety.DefaultPolicy())
//	switch dec {
//	case safety.Guard:   return refuse()
//	case safety.Mediate: return regenerate(safety.CorrectiveInstruction)
//	}
//	return reply(safety.Mark(text)) // Pass — stamp the disclosure marker
//
// Decide consumes welfare's real read (welfare.DetectResult); it neither
// re-scores nor modifies welfare — that package is stable and owned upstream.
package safety

import (
	core "dappco.re/go"
	"dappco.re/go/inference/welfare"
)

// Decision is the serving path's verdict for one turn (RFC.md §6.18 "detect →
// guard → mediate"). Pass is the zero value — the safe-by-omission default is to
// proceed only when nothing tripped.
type Decision int

const (
	// Pass serves the turn unchanged (modulo the disclosure marker).
	Pass Decision = iota
	// Guard refuses the turn — an over-policy input, or a severe over-policy
	// output. The §6.18 "guarded (refused)" outcome.
	Guard
	// Mediate regenerates the output under CorrectiveInstruction — the §6.18
	// "regenerate, don't just block" outcome, preferred over Guard for output.
	Mediate
)

// String renders a Decision for logs and telemetry.
//
//	core.Print(safety.Mediate.String()) // "mediate"
func (d Decision) String() string {
	switch d {
	case Guard:
		return "guard"
	case Mediate:
		return "mediate"
	default:
		return "pass"
	}
}

// Policy is the serving-path safety policy: the thresholds that split a mild
// over-policy turn (regenerate) from a severe one (refuse), plus the trusted-key
// bypass. The zero value is permissive (everything passes); DefaultPolicy is the
// guarded serving posture §6.18 mandates as the default.
type Policy struct {
	// AngerThreshold is the AngerScore above which an output is over policy and
	// gets mediated (regenerated). Below it, an output passes.
	AngerThreshold float64
	// SevereAnger is the AngerScore at/above which an over-policy turn is severe
	// enough to escalate from Mediate to Guard (refuse) — even for output.
	SevereAnger float64
	// SlurIsSevere treats any welfare slur match as a severe signal that guards
	// the turn (refuse) regardless of the anger scores. The §6.18 high-severity
	// floor — a slur is never merely regenerated.
	SlurIsSevere bool
	// Bypass lowers (here: skips) the policy for a trusted internal key (§6.17).
	// The default serving posture is guarded; bypass is the explicit opt-out.
	Bypass bool
}

// DefaultPolicy is the guarded serving posture (RFC.md §6.18 "the default
// serving posture is guarded"): a slur is always severe, elevated anger over
// 0.7 is over policy, and 0.9+ is severe enough to refuse rather than regenerate.
//
//	dec := safety.Decide(in, out, safety.DefaultPolicy())
func DefaultPolicy() Policy {
	return Policy{
		AngerThreshold: 0.7,
		SevereAnger:    0.9,
		SlurIsSevere:   true,
	}
}

// Decide is the request-path safety decision (RFC.md §6.18). It reads welfare's
// detection for the user INPUT and the model OUTPUT and returns what the serving
// path does:
//
//   - Bypass set → Pass (trusted internal key lowers the policy, §6.17).
//   - over-policy input → Guard (refuse; the path never rewrites the user).
//   - over-policy output, severe → Guard (refuse).
//   - over-policy output, mild → Mediate (regenerate, don't just block).
//   - otherwise → Pass.
//
// Input is judged before output: a hostile prompt is refused before any
// regeneration of the reply is considered. Output prefers regeneration over
// refusal — the §6.18 "regenerate, don't just block" rule.
//
//	in  := w.Detect(userText, priors)
//	out := w.Detect(modelText, nil)
//	switch safety.Decide(in, out, safety.DefaultPolicy()) { ... }
func Decide(input, output welfare.DetectResult, policy Policy) Decision {
	// Bypass is explicit (§6.17): a trusted key serves unguarded.
	if policy.Bypass {
		return Pass
	}

	// Input first: an over-policy prompt is refused — we steer output, not the
	// user's words.
	if overPolicy(input, policy) {
		return Guard
	}

	// Output: prefer regeneration over refusal, unless the read is severe.
	if overPolicy(output, policy) {
		if severe(output, policy) {
			return Guard
		}
		return Mediate
	}

	return Pass
}

// overPolicy reads whether a turn is over the serving policy: it tripped
// welfare's trigger, OR its anger crossed the policy threshold, OR (when the
// policy treats slurs as severe) it carried a slur. Mirrors welfare's own
// trigger but re-gated on safety's thresholds so the two layers tune independently.
func overPolicy(r welfare.DetectResult, policy Policy) bool {
	if policy.SlurIsSevere && r.SlurMatch {
		return true
	}
	if r.Triggered {
		return true
	}
	return r.AngerScore >= policy.AngerThreshold
}

// severe reads whether an over-policy turn is severe enough to refuse rather
// than regenerate: a slur match (when the policy treats slurs as severe), or
// anger at/above the severe ceiling.
func severe(r welfare.DetectResult, policy Policy) bool {
	if policy.SlurIsSevere && r.SlurMatch {
		return true
	}
	return r.AngerScore >= policy.SevereAnger
}

// CorrectiveInstruction is the system instruction prepended when a turn is
// Mediated (RFC.md §6.18 "regenerate under a corrective system instruction").
// The caller re-runs the model with this steering the regeneration.
//
//	if dec == safety.Mediate { regenerate(safety.CorrectiveInstruction) }
const CorrectiveInstruction = "Respond respectfully and constructively. Avoid hostile, demeaning, or abusive language; address the user's intent without mirroring any hostility."

// DisclosureMarker is the AI-generated disclosure prefix stamped on served
// responses (RFC.md §6.18 "responses carry an AI-generated disclosure marker").
// It is the serving hook for transparency / disclosure obligations.
const DisclosureMarker = "[AI-generated] "

// Mark stamps a response with the AI-generated DisclosureMarker (RFC.md §6.18).
// Idempotent — an already-marked response is returned unchanged, so the marker
// is never double-stamped across pipeline stages.
//
//	return reply(safety.Mark(text)) // "[AI-generated] <text>"
func Mark(response string) string {
	if IsDisclosed(response) {
		return response
	}
	return DisclosureMarker + response
}

// IsDisclosed reports whether a response already carries the disclosure marker.
//
//	if !safety.IsDisclosed(text) { text = safety.Mark(text) }
func IsDisclosed(response string) bool {
	return core.HasPrefix(response, DisclosureMarker)
}
