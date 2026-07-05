// SPDX-Licence-Identifier: EUPL-1.2

package registry

import "testing"

func TestRegistry_ModelCard_Good(t *testing.T) {
	r := newSeededRegistry(t)

	card := ModelCard{
		IntendedUse:        "Ethical instruction following on device.",
		TrainingProvenance: "Gemma 4 4B + LEM ethics adapter, run lem-2026-06-14.",
		EvalSummary:        "8-PAC unanimous pass; ethics 0.91, helpfulness 0.88.",
		Limitations:        "English-first; not for medical or legal advice.",
		Links:              map[string]string{"hf": "https://huggingface.co/lthn/lemma"},
	}

	// Attaching a card to a stored entry succeeds and is readable straight back.
	if sr := r.SetCard("gemma-4-4b-it", card); !sr.OK {
		t.Fatalf("set card: %v", sr.Error())
	}
	gr := r.GetCard("gemma-4-4b-it")
	if !gr.OK {
		t.Fatalf("get card: %v", gr.Error())
	}
	got := gr.Value.(ModelCard)
	if got.IntendedUse != card.IntendedUse {
		t.Errorf("intended use: got %q, want %q", got.IntendedUse, card.IntendedUse)
	}
	if got.Links["hf"] != "https://huggingface.co/lthn/lemma" {
		t.Errorf("links round-trip: got %v", got.Links)
	}

	// The card also resolves through the entry by id-or-alias.
	e := r.Resolve("lemma").Value.(Entry)
	if e.Card == nil {
		t.Fatalf("resolved entry carries no card")
	}
	if e.Card.EvalSummary != card.EvalSummary {
		t.Errorf("entry.Card eval summary: got %q, want %q", e.Card.EvalSummary, card.EvalSummary)
	}
}

func TestRegistry_ModelCard_Bad(t *testing.T) {
	r := newSeededRegistry(t)

	// Setting a card on an unknown entry fails — there is nothing to attach to.
	if sr := r.SetCard("does-not-exist", ModelCard{IntendedUse: "x"}); sr.OK {
		t.Fatalf("set card on missing entry should fail, got %+v", sr.Value)
	}

	// Getting a card from an unknown entry fails.
	if gr := r.GetCard("does-not-exist"); gr.OK {
		t.Fatalf("get card on missing entry should fail, got %+v", gr.Value)
	}

	// A known entry with no card set reports absence rather than an empty card.
	if gr := r.GetCard("gemma-4-4b-it"); gr.OK {
		t.Fatalf("entry with no card should report absent, got %+v", gr.Value)
	}
}

func TestRegistry_ModelCard_Ugly(t *testing.T) {
	r := newSeededRegistry(t)

	// SetCard accepts id or alias and attaches to the same underlying entry.
	if sr := r.SetCard("lemma", ModelCard{IntendedUse: "via alias"}); !sr.OK {
		t.Fatalf("set card by alias: %v", sr.Error())
	}
	if got := r.GetCard("gemma-4-4b-it").Value.(ModelCard).IntendedUse; got != "via alias" {
		t.Errorf("alias-set card not visible by id: got %q", got)
	}

	// Re-setting replaces the card in place (last write wins), and the entry's
	// other fields survive the card update.
	before := r.Resolve("lemma").Value.(Entry).MemoryBytes
	if sr := r.SetCard("lemma", ModelCard{IntendedUse: "replaced", Limitations: "none stated"}); !sr.OK {
		t.Fatalf("replace card: %v", sr.Error())
	}
	after := r.Resolve("lemma")
	if !after.OK {
		t.Fatalf("resolve after card replace: %v", after.Error())
	}
	e := after.Value.(Entry)
	if e.Card.IntendedUse != "replaced" || e.Card.Limitations != "none stated" {
		t.Errorf("card not replaced in place: got %+v", e.Card)
	}
	if e.MemoryBytes != before {
		t.Errorf("setting a card disturbed the entry footprint: got %d, want %d", e.MemoryBytes, before)
	}

	// A card set on an entry stored directly (carried on the Entry at Put time)
	// is returned by GetCard without a separate SetCard call.
	direct := sampleEntry("carded", 1_000_000_000, "cd")
	direct.Card = &ModelCard{IntendedUse: "born with a card"}
	if pr := r.Put(direct); !pr.OK {
		t.Fatalf("put carded entry: %v", pr.Error())
	}
	if got := r.GetCard("carded"); !got.OK || got.Value.(ModelCard).IntendedUse != "born with a card" {
		t.Errorf("card carried on Put not readable: %+v", got)
	}
}
