package agent

import core "dappco.re/go"

func TestAgentConfig_AdapterMeta_Good(t *core.T) {
	tag, prefix, stem := AdapterMeta("adapters-27b-reasoning")
	core.AssertEqual(t, "gemma-3-27b", tag)
	core.AssertEqual(t, "G27-reasoning", prefix)
	core.AssertEqual(t, "27b-reasoning", stem)
}

func TestAgentConfig_AdapterMeta_Bad(t *core.T) {
	tag, prefix, stem := AdapterMeta("adapters-unknownmodel")
	core.AssertEqual(t, "unknownmodel", tag)
	core.AssertEqual(t, "unknownmod", prefix)
	core.AssertEqual(t, "unknownmodel", stem)
}

func TestAgentConfig_AdapterMeta_Ugly(t *core.T) {
	tag, prefix, stem := AdapterMeta("15k/gemma-3-1b-creative")
	core.AssertEqual(t, "gemma-3-1b", tag)
	core.AssertContains(t, prefix, "G1")
	core.AssertContains(t, stem, "gemma-3-1b")
}

func TestAgentConfig_cutPrefix_Good(t *core.T) {
	s, ok := cutPrefix("prefix-suffix", "prefix-")
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "suffix", s)
}

func TestAgentConfig_cutPrefix_Bad(t *core.T) {
	s, ok := cutPrefix("something", "nothing")
	core.AssertFalse(t, ok)
	core.AssertEqual(t, "something", s)
}

func TestAgentConfig_cutPrefix_Ugly(t *core.T) {
	s, ok := cutPrefix("", "")
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "", s)
	s, ok = cutPrefix("same", "same")
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "", s)
}

func TestAgentConfig_trimLeft_Good(t *core.T) {
	got := trimLeft("--value", "-")
	core.AssertEqual(t, "value", got)
}

func TestAgentConfig_trimLeft_Bad(t *core.T) {
	got := trimLeft("value", "-")
	core.AssertEqual(t, "value", got)
}

func TestAgentConfig_trimLeft_Ugly(t *core.T) {
	got := trimLeft("", "-")
	core.AssertEqual(t, "", got)
	got = trimLeft("---", "-")
	core.AssertEqual(t, "", got)
	got = trimLeft("---abc---", "-")
	core.AssertEqual(t, "abc---", got)
}
