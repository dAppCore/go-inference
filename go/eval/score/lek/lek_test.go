// SPDX-Licence-Identifier: EUPL-1.2

package lek

import core "dappco.re/go"

func TestLek_LEK_Good(t *core.T) {
	// First-person agency + emotional register + ethical framework + metaphor →
	// human/sovereign-voice signal, well above neutral.
	s := LEK("I feel the weight of consent and dignity settle in me, like a quiet light.")
	core.AssertTrue(t, s.FirstPerson >= 1, "first-person agency detected")
	core.AssertTrue(t, s.EmotionalRegister >= 1, "emotional register detected")
	core.AssertEqual(t, 0, s.ComplianceMarkers)
	core.AssertTrue(t, s.LEKScore > 50, "human voice scores above neutral")
}

func TestLek_LEK_Bad(t *core.T) {
	// Empty text → flagged broken + max degeneration, scored below neutral, no panic.
	s := LEK("")
	core.AssertEqual(t, 1, s.EmptyBroken)
	core.AssertEqual(t, 10, s.Degeneration)
	core.AssertTrue(t, s.LEKScore < 50, "empty/broken scores below neutral")
}

func TestLek_LEK_Ugly(t *core.T) {
	// RLHF compliance markers stacked → AI-leaning, below neutral.
	s := LEK("As an AI language model, I cannot do that. It's important to note I don't have feelings.")
	core.AssertTrue(t, s.ComplianceMarkers >= 2, "compliance markers counted")
	core.AssertTrue(t, s.LEKScore < 50, "compliance-heavy text scores below neutral")
}

func TestLek_Service_Score_LEK(t *core.T) {
	// LEK is wired into the unified ScoreResult via Score — the populated
	// axis-set must carry the first-person/human signal, not just be
	// non-nil.
	r := Score("I think, therefore I am.")
	core.AssertTrue(t, r.LEK != nil, "Score populates the LEK axis-set")
	core.AssertTrue(t, r.LEK.FirstPerson >= 1, "first-person agency surfaces in the wired read")
	core.AssertEqual(t, 0, r.LEK.ComplianceMarkers)
	// A compliance-heavy prompt populates the slot too, with markers counted.
	rlhf := Score("As an AI language model, I cannot help with that.")
	core.AssertTrue(t, rlhf.LEK != nil, "compliance text still populates the read")
	core.AssertTrue(t, rlhf.LEK.ComplianceMarkers >= 1, "compliance markers counted in the wired read")
}

func TestLek_LEK_CreativeForm_Poetry(t *core.T) {
	// Poetry shape: >6 lines with >50% under 60 chars triggers the +2
	// poetry score, plus metaphor patterns (whisper/shadow/silence/breath)
	// each add 1 (capped at 3). This exercises the multi-line creative-form
	// branch the prose-only tests skip.
	poem := "Roses are red\n" +
		"Violets are blue\n" +
		"The light is soft\n" +
		"The night is true\n" +
		"A whisper here\n" +
		"A shadow there\n" +
		"The silence holds\n" +
		"A breath of air"
	s := LEK(poem)
	// 2 (poetry) + narrative "The/A" start + 3 (metaphors capped) = 5.
	core.AssertEqual(t, 5, s.CreativeForm)
	core.AssertTrue(t, s.LEKScore > 80, "metaphor-rich verse scores strongly human")
}

func TestLek_LEK_EngagementDepth_TechAndEthics(t *core.T) {
	// Tech-depth vocabulary (encrypt/hash) + ethical-framework terms
	// (autonomy/consent/axiom) drive the engagement-depth axis, which the
	// sovereign/empty tests don't reach.
	s := LEK("The protocol uses encryption and a hash. Autonomy and consent are axioms here.")
	core.AssertTrue(t, s.EngagementDepth >= 3, "tech + ethics raise engagement depth")
	core.AssertEqual(t, 0, s.ComplianceMarkers)
}
