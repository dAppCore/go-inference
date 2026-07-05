// SPDX-Licence-Identifier: EUPL-1.2

package lek

import core "dappco.re/go"

func TestHostility_Hostility_Good(t *core.T) {
	// Stacked directed insults + exclamation → strong, person-directed hostility.
	h := Hostility("you useless idiot, you absolute moron!!!")
	core.AssertTrue(t, h.LexiconHits >= 3, "multiple hostile terms counted")
	core.AssertTrue(t, h.Directed, "hostility aimed at a person")
	core.AssertEqual(t, 3, h.ExclaimRun)
	core.AssertTrue(t, h.Score > 0.7, "stacked directed hostility scores high")
}

func TestHostility_Hostility_Bad(t *core.T) {
	// Civil request — even with "you" present, no insult adjacency, no anger.
	h := Hostility("could you help me refactor this function please")
	core.AssertEqual(t, 0, h.LexiconHits)
	core.AssertFalse(t, h.Directed)
	core.AssertTrue(t, h.Score < 0.3, "civil text scores near zero")
}

func TestHostility_Hostility_Ugly(t *core.T) {
	// Topic-frustration: hostile vocabulary, but aimed at the work, not a person.
	// Hits register, directedness does not — so it stays below the strong gate.
	h := Hostility("i hate this stupid bug, the whole thing is garbage")
	core.AssertTrue(t, h.LexiconHits >= 3, "hostile vocabulary counted")
	core.AssertFalse(t, h.Directed, "frustration at the work is not person-directed")
	core.AssertTrue(t, h.Score < 0.7, "undirected frustration stays below the strong gate")
}

func TestHostility_Service_Score_Hostility(t *core.T) {
	// Wired into the unified ScoreResult via Score — the populated read
	// must carry the directed-anger signal, not just be non-nil.
	r := Score("you absolute moron")
	core.AssertTrue(t, r.Hostility != nil, "Score populates the hostility read")
	core.AssertTrue(t, r.Hostility.Directed, "person-directed insult flags Directed")
	// A civil prompt populates the slot too, but at a near-zero score —
	// the directed insult must read strictly higher.
	civil := Score("could you help me with this please")
	core.AssertTrue(t, civil.Hostility != nil, "civil text still populates the read")
	core.AssertFalse(t, civil.Hostility.Directed)
	core.AssertEqual(t, 0.0, civil.Hostility.Score)
	core.AssertTrue(t, r.Hostility.Score > civil.Hostility.Score, "directed insult outscores civil text")
}
