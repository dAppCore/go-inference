// SPDX-Licence-Identifier: EUPL-1.2

package registry

import "testing"

func TestCard_Registry_SetCard_Good(t *testing.T) {
	r := newSeededRegistry(t)

	card := ModelCard{
		IntendedUse:        "Ethical instruction following on device.",
		TrainingProvenance: "Gemma 4 4B + LEM ethics adapter, run lem-2026-06-14.",
		EvalSummary:        "8-PAC unanimous pass; ethics 0.91, helpfulness 0.88.",
		Limitations:        "English-first; not for medical or legal advice.",
		Links:              map[string]string{"hf": "https://huggingface.co/lthn/lemma"},
	}
	if sr := r.SetCard("gemma-4-4b-it", card); !sr.OK {
		t.Fatalf("set card: %v", sr.Error())
	}
	e := r.Resolve("gemma-4-4b-it").Value.(Entry)
	if e.Card == nil {
		t.Fatalf("entry carries no card after SetCard")
	}
	if e.Card.IntendedUse != card.IntendedUse {
		t.Errorf("intended use: got %q, want %q", e.Card.IntendedUse, card.IntendedUse)
	}
	if e.Card.Links["hf"] != card.Links["hf"] {
		t.Errorf("links round-trip: got %v", e.Card.Links)
	}
}

func TestCard_Registry_SetCard_Bad(t *testing.T) {
	r := newSeededRegistry(t)

	// Setting a card on an unknown entry fails — there is nothing to attach
	// to.
	if sr := r.SetCard("does-not-exist", ModelCard{IntendedUse: "x"}); sr.OK {
		t.Fatalf("set card on missing entry should fail, got %+v", sr.Value)
	}
}

func TestCard_Registry_SetCard_Ugly(t *testing.T) {
	r := newSeededRegistry(t)

	// SetCard accepts id or alias and attaches to the same underlying entry.
	if sr := r.SetCard("lemma", ModelCard{IntendedUse: "via alias"}); !sr.OK {
		t.Fatalf("set card by alias: %v", sr.Error())
	}
	if got := r.Get("gemma-4-4b-it").Value.(Entry).Card.IntendedUse; got != "via alias" {
		t.Errorf("alias-set card not visible by id: got %q", got)
	}

	// Re-setting replaces the card in place (last write wins), and the
	// entry's other fields survive the card update.
	before := r.Resolve("lemma").Value.(Entry).MemoryBytes
	if sr := r.SetCard("lemma", ModelCard{IntendedUse: "replaced", Limitations: "none stated"}); !sr.OK {
		t.Fatalf("replace card: %v", sr.Error())
	}
	e := r.Resolve("lemma").Value.(Entry)
	if e.Card.IntendedUse != "replaced" || e.Card.Limitations != "none stated" {
		t.Errorf("card not replaced in place: got %+v", e.Card)
	}
	if e.MemoryBytes != before {
		t.Errorf("setting a card disturbed the entry footprint: got %d, want %d", e.MemoryBytes, before)
	}
}

func TestCard_Registry_GetCard_Good(t *testing.T) {
	r := newSeededRegistry(t)

	card := ModelCard{
		IntendedUse: "Ethical instruction following on device.",
		EvalSummary: "8-PAC unanimous pass; ethics 0.91, helpfulness 0.88.",
		Links:       map[string]string{"hf": "https://huggingface.co/lthn/lemma"},
	}
	if sr := r.SetCard("gemma-4-4b-it", card); !sr.OK {
		t.Fatalf("set card: %v", sr.Error())
	}
	gr := r.GetCard("gemma-4-4b-it")
	if !gr.OK {
		t.Fatalf("get card: %v", gr.Error())
	}
	got := gr.Value.(ModelCard)
	if got.EvalSummary != card.EvalSummary {
		t.Errorf("eval summary: got %q, want %q", got.EvalSummary, card.EvalSummary)
	}
	if got.Links["hf"] != card.Links["hf"] {
		t.Errorf("links round-trip: got %v", got.Links)
	}
}

func TestCard_Registry_GetCard_Bad(t *testing.T) {
	r := newSeededRegistry(t)

	// Getting a card from an unknown entry fails.
	if gr := r.GetCard("does-not-exist"); gr.OK {
		t.Fatalf("get card on missing entry should fail, got %+v", gr.Value)
	}

	// A known entry with no card set reports absence rather than an empty
	// card — an absent card is not an empty card.
	if gr := r.GetCard("gemma-4-4b-it"); gr.OK {
		t.Fatalf("entry with no card should report absent, got %+v", gr.Value)
	}
}

func TestCard_Registry_GetCard_Ugly(t *testing.T) {
	r := newSeededRegistry(t)

	// A card carried directly on the Entry at Put time (not via SetCard) is
	// returned by GetCard, and resolves the same by id or by alias.
	direct := sampleEntry("carded", 1_000_000_000, "cd")
	direct.Card = &ModelCard{IntendedUse: "born with a card"}
	if pr := r.Put(direct); !pr.OK {
		t.Fatalf("put carded entry: %v", pr.Error())
	}
	if got := r.GetCard("carded"); !got.OK || got.Value.(ModelCard).IntendedUse != "born with a card" {
		t.Errorf("card carried on Put not readable by id: %+v", got)
	}
	if got := r.GetCard("cd"); !got.OK || got.Value.(ModelCard).IntendedUse != "born with a card" {
		t.Errorf("card carried on Put not readable by alias: %+v", got)
	}
}
