// SPDX-Licence-Identifier: EUPL-1.2

package welfare

import (
	core "dappco.re/go"
)

// format.go — the mediator-output formatting pass (#376 follow-up). The
// mediator is a model: its reply may carry a reasoning channel around the JSON,
// and its rephrase text may carry wrapping quotes, control bytes, or channel
// markers. That text replaces the USER'S MESSAGE inside the prompt template, so
// it must reach the template as clean single-purpose text — before this pass it
// was spliced verbatim.

// Gemma-family reasoning-channel markers. The mediator dispatch may return the
// raw decode, in which case the reasoning precedes the visible answer as
// "<|channel>thought\n…<channel|>answer".
const (
	thoughtOpen  = "<|channel>"
	thoughtClose = "<channel|>"
)

// stripThought returns the reply's visible answer: everything after the LAST
// channel-close marker. A thought channel may itself contain '{'/'}' — JSON
// extraction over the raw reply would seize the wrong span, so the thought must
// go before extractJSONObject runs. A reply with an open marker but no close is
// all thought (truncated mid-channel): nothing visible, returns "".
func stripThought(reply string) string {
	if i := core.LastIndex(reply, thoughtClose); i >= 0 {
		return reply[i+len(thoughtClose):]
	}
	if core.Contains(reply, thoughtOpen) {
		return ""
	}
	return reply
}

// formatMediated normalises a mediator-supplied replacement text before it is
// spliced into the conversation as the user's message: channel markers are
// removed (they would corrupt the prompt template), CR/LF is normalised, C0
// control bytes other than \n and \t are dropped, and ONE symmetric wrapping
// quote pair is unwrapped (models habitually quote a rephrase). Empty after
// formatting means the rephrase is unusable — callers proceed with the
// original.
func formatMediated(text string) string {
	text = core.Replace(text, thoughtOpen, "")
	text = core.Replace(text, thoughtClose, "")
	var b core.Builder
	b.Grow(len(text))
	for i := 0; i < len(text); i++ {
		c := text[i]
		if c == '\r' {
			if i+1 >= len(text) || text[i+1] != '\n' {
				b.WriteByte('\n')
			}
			continue
		}
		if c < 0x20 && c != '\n' && c != '\t' {
			continue
		}
		b.WriteByte(c)
	}
	return core.Trim(unwrapQuotes(core.Trim(b.String())))
}

// unwrapQuotes removes ONE symmetric wrapping quote pair — exactly one, never
// recursively, so deliberately quoted content inside the rephrase survives.
func unwrapQuotes(s string) string {
	pairs := [4][2]string{{`"`, `"`}, {"'", "'"}, {"“", "”"}, {"‘", "’"}}
	for _, p := range pairs {
		if len(s) >= len(p[0])+len(p[1]) && core.HasPrefix(s, p[0]) && core.HasSuffix(s, p[1]) {
			return s[len(p[0]) : len(s)-len(p[1])]
		}
	}
	return s
}

// formatReason flattens a mediator-supplied reason to one audit-ready line:
// newlines and tabs become single spaces, other C0 control bytes are dropped,
// runs of spaces collapse. Reasons feed the learning corpus and the audit
// trail, both of which are line-oriented.
func formatReason(reason string) string {
	var b core.Builder
	b.Grow(len(reason))
	space := false
	for i := 0; i < len(reason); i++ {
		c := reason[i]
		if c == '\n' || c == '\r' || c == '\t' || c == ' ' {
			space = true
			continue
		}
		if c < 0x20 {
			continue
		}
		if space && b.Len() > 0 {
			b.WriteByte(' ')
		}
		space = false
		b.WriteByte(c)
	}
	return b.String()
}
