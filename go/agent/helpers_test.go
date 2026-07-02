// SPDX-Licence-Identifier: EUPL-1.2

package agent

import core "dappco.re/go"

func TestHelpers_repeatStr_Good(t *core.T) {
	got := repeatStr("ab", 3)
	core.AssertEqual(t, "ababab", got)
}

func TestHelpers_repeatStr_Bad(t *core.T) {
	got := repeatStr("x", 0)
	core.AssertEqual(t, "", got)
	got = repeatStr("x", -1)
	core.AssertEqual(t, "", got)
}

func TestHelpers_repeatStr_Ugly(t *core.T) {
	got := repeatStr("", 5)
	core.AssertEqual(t, "", got)
}

func TestHelpers_readAll_Good(t *core.T) {
	r := readAll(core.NewReader("hello"))
	requireResultOK(t, r)
	core.AssertEqual(t, []byte("hello"), r.Value.([]byte))
}

func TestHelpers_readAll_Bad(t *core.T) {
	r := readAll(42)
	assertResultError(t, r)
}

func TestHelpers_readAll_Ugly(t *core.T) {
	r := readAll(core.NewReader(""))
	requireResultOK(t, r)
	core.AssertEqual(t, []byte{}, r.Value.([]byte))
}
