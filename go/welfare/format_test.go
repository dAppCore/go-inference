// SPDX-Licence-Identifier: EUPL-1.2

package welfare

import (
	core "dappco.re/go"
)

func TestFormat_FormatMediated_Good(t *core.T) {
	// A clean rephrase passes through untouched.
	core.AssertEqual(t, "please fix this", formatMediated("please fix this"))
	// A quoted rephrase is unwrapped once — models habitually quote.
	core.AssertEqual(t, "please fix this", formatMediated(`"please fix this"`))
	core.AssertEqual(t, "please fix this", formatMediated("“please fix this”"))
}

func TestFormat_FormatMediated_Bad(t *core.T) {
	// CRLF normalises, C0 control bytes drop, \n and \t survive.
	core.AssertEqual(t, "a\nb", formatMediated("a\r\nb\x00"))
	core.AssertEqual(t, "a\tb", formatMediated(" a\tb\x07 "))
	// A lone CR is a newline, not a lost character.
	core.AssertEqual(t, "a\nb", formatMediated("a\rb"))
	// Channel markers are template-corrupting tokens — removed, words kept.
	core.AssertEqual(t, "hello world", formatMediated("hello <|channel><channel|>world"))
}

func TestFormat_FormatMediated_Ugly(t *core.T) {
	// Unusable shapes format to empty (callers proceed with the original).
	core.AssertEqual(t, "", formatMediated("   "))
	core.AssertEqual(t, "", formatMediated("\x00\x01\x02"))
	core.AssertEqual(t, "", formatMediated(`""`))
	// One unwrap only — inner quoted content survives verbatim.
	core.AssertEqual(t, `"nested"`, formatMediated(`""nested""`))
	// Asymmetric quotes are content, not wrapping.
	core.AssertEqual(t, `"open`, formatMediated(`"open`))
}

func TestFormat_StripThought(t *core.T) {
	// The reasoning channel is cut so brace junk inside it cannot derail JSON
	// extraction — the visible answer after the LAST close marker survives.
	reply := "<|channel>thought\n{not: the, json}\n<channel|>{\"tool\":\"lem_pause\",\"params\":{}}"
	core.AssertEqual(t, `{"tool":"lem_pause","params":{}}`, stripThought(reply))
	// No channel markers → untouched.
	core.AssertEqual(t, "plain", stripThought("plain"))
	// Open without close = truncated mid-thought: nothing visible.
	core.AssertEqual(t, "", stripThought("<|channel>thought only..."))

	// End to end: the thought's braces would have seized the wrong span before
	// the strip — parseMediate now lands the real tool object.
	res := parseMediate(reply, false)
	core.AssertEqual(t, DecisionPause, res.Decision)
}

func TestFormat_FormatReason(t *core.T) {
	// Reasons flatten to one audit line: newlines/tabs to spaces, runs collapse.
	core.AssertEqual(t, "too hostile to keep", formatReason("too hostile\n\tto  keep"))
	core.AssertEqual(t, "", formatReason(" \n\t "))
	core.AssertEqual(t, "plain reason", formatReason("plain reason"))
}
