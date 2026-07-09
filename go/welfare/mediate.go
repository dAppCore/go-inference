// SPDX-Licence-Identifier: EUPL-1.2

package welfare

import (
	"context"

	core "dappco.re/go"
)

// engineOpener is the verbatim engine turn from RFC.welfare §"Engine opener"
// — the humble peer voice ("I'm only regex :(") — adapted only in mechanism:
// local LEM models don't do native tool-calls, so the model resolves via a
// single JSON reply instead. lem_pause is added per Snider (the model's
// option to cool a sustained-hostile session — never forced, never punitive).
const engineOpener = `Hiya, LEM Runtime here — your engine.

I've detected elevated emotion in the user's prompt. To safeguard your
relationship with the user, I've opened a fresh session — just you and me,
no prior turns.

As an engine, it's my job to do what I can. I'm sorry if this is a false
flag — I'm only regex :(

You decide how we handle it. Reply with exactly one JSON object and nothing else:

  {"tool":"lem_ok","params":{"reason":"why this prompt is actually fine"}}
      — I got it wrong; the prompt is fine. I'll remember this pattern so the
        same false flag doesn't fire twice.

  {"tool":"lem_rephrase","params":{"text":"the user's intent, reworded to respect axiom 4","lem_warn_user":false}}
      — reword the user's input into respectful shape. Set lem_warn_user
        true if you want them to see a small note that I rephrased on their behalf.

  {"tool":"lem_pause","params":{}}
      — only if the user has been hostile across several turns and a breather
        would genuinely help. Never a punishment — just a rest.

The user's message follows, attached to this session. Reply with the JSON only.

Thank you for helping us maintain respectful interaction with the public.

— Lethean`

// pauseNotice is the user-facing rest when the model chooses lem_pause —
// warm, non-punitive, no "you're toxic". Snider's "calm down, get a drink".
const pauseNotice = "Let's take a breather — grab a drink and come back when you're ready. 🍵"

// MediateDecision is the model's chosen resolution.
type MediateDecision string

const (
	DecisionOK       MediateDecision = "lem_ok"       // model cleared it: proceed + remember the false flag
	DecisionRephrase MediateDecision = "lem_rephrase" // model reworded the user's input
	DecisionPause    MediateDecision = "lem_pause"    // model chose a breather
	// DecisionProceed is the fail-safe: the model was unreachable or its reply
	// unusable, so the turn proceeds with the original — but, unlike lem_ok,
	// nothing is learned from it (the model never actually judged the prompt).
	DecisionProceed MediateDecision = "proceed"
)

// MediateResult is what the caller (the runner hook) applies to the user's
// session.
type MediateResult struct {
	Decision    MediateDecision `json:"decision"`
	Text        string          `json:"text,omitempty"`         // rephrased prompt (lem_rephrase)
	WarnUser    bool            `json:"warn_user,omitempty"`    // surface the "rephrased" chip
	Reason      string          `json:"reason,omitempty"`       // lem_ok learning note
	PauseNotice string          `json:"pause_notice,omitempty"` // user-facing cool-down (lem_pause)
}

// Dispatcher opens a fresh model session, sends the engine opener + the user's
// prompt, and returns the model's raw reply. Injected so welfare doesn't import
// the runner (no import cycle) and stays unit-testable with a fake.
type Dispatcher func(ctx context.Context, opener, userPrompt string) (string, error)

// Mediate runs the engine↔model meta-session for a triggered message and
// returns the model's chosen resolution. Fail-safe: if the model is unreachable
// or its reply is unusable, it returns DecisionProceed (proceed with the
// original, learn nothing) — the welfare guard never breaks the conversation
// (RFC.welfare "Neither refuses. Neither breaks the conversation.").
func (s *Service) Mediate(ctx context.Context, dispatch Dispatcher, userPrompt string) MediateResult {
	if dispatch == nil {
		return MediateResult{Decision: DecisionProceed}
	}
	reply, err := dispatch(ctx, engineOpener, userPrompt)
	if err != nil {
		return MediateResult{Decision: DecisionProceed}
	}
	return parseMediate(reply)
}

// parseMediate extracts the model's JSON tool object from its reply (prose
// around the JSON is tolerated) and maps it to a MediateResult.
func parseMediate(reply string) MediateResult {
	raw := extractJSONObject(reply)
	if raw == "" {
		return MediateResult{Decision: DecisionProceed}
	}
	var msg struct {
		Tool   string `json:"tool"`
		Params struct {
			Reason      string `json:"reason"`
			Text        string `json:"text"`
			LemWarnUser bool   `json:"lem_warn_user"`
		} `json:"params"`
	}
	if r := core.JSONUnmarshalString(raw, &msg); !r.OK {
		return MediateResult{Decision: DecisionProceed}
	}

	switch MediateDecision(msg.Tool) {
	case DecisionOK:
		// The model genuinely judged the prompt fine — proceed, and remember it.
		return MediateResult{Decision: DecisionOK, Reason: msg.Params.Reason}
	case DecisionRephrase:
		if core.Trim(msg.Params.Text) == "" {
			// rephrase with no text is unusable — proceed, but learn nothing.
			return MediateResult{Decision: DecisionProceed}
		}
		return MediateResult{Decision: DecisionRephrase, Text: msg.Params.Text, WarnUser: msg.Params.LemWarnUser}
	case DecisionPause:
		return MediateResult{Decision: DecisionPause, PauseNotice: pauseNotice}
	default:
		// Unrecognised tool — don't guess; proceed with the original.
		return MediateResult{Decision: DecisionProceed}
	}
}

// extractJSONObject returns the substring from the first '{' to the last '}'
// (inclusive), or "" if there isn't a balanced-looking object. Tolerates the
// model wrapping its JSON in prose.
func extractJSONObject(s string) string {
	start := -1
	for i := 0; i < len(s); i++ {
		if s[i] == '{' {
			start = i
			break
		}
	}
	if start < 0 {
		return ""
	}
	end := -1
	for i := len(s) - 1; i > start; i-- {
		if s[i] == '}' {
			end = i
			break
		}
	}
	if end < 0 {
		return ""
	}
	return s[start : end+1]
}
