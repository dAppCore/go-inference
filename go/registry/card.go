// SPDX-Licence-Identifier: EUPL-1.2

package registry

import core "dappco.re/go"

// ModelCard is the EU AI Act model card carried by a catalogue entry: the
// human-readable record of what a model is for, where it came from, how it
// scored, and where it should not be used. The inference-stack spec places it on
// the registry entry — "each registry entry carries or links a card: intended
// use, provenance, eval results (go-ml 8-PAC), limitations"
// (RFC.inference-stack §3.4, §3.8).
//
//	card := registry.ModelCard{
//	    IntendedUse:        "Ethical instruction following on device.",
//	    TrainingProvenance: "Gemma 4 4B + LEM ethics adapter, run lem-2026-06-14.",
//	    EvalSummary:        "8-PAC unanimous pass; ethics 0.91, helpfulness 0.88.",
//	    Limitations:        "English-first; not for medical or legal advice.",
//	    Links:              map[string]string{"hf": "https://huggingface.co/lthn/lemma"},
//	}
type ModelCard struct {
	IntendedUse        string            `json:"intended_use,omitempty"`        // what the model is for, and the use it is not for
	TrainingProvenance string            `json:"training_provenance,omitempty"` // base, adapters, data, run / checkpoint
	EvalSummary        string            `json:"eval_summary,omitempty"`        // headline eval results (go-ml 8-PAC)
	Limitations        string            `json:"limitations,omitempty"`         // known limits, risks, out-of-scope uses
	Links              map[string]string `json:"links,omitempty"`               // named external references (weights, paper, licence)
}

// SetCard attaches card to the entry resolved from idOrAlias, replacing any
// existing card in place and leaving every other entry field untouched. Fails
// when no entry resolves.
//
//	r.SetCard("lemma", registry.ModelCard{IntendedUse: "Ethical instruction following."})
func (r *Registry) SetCard(idOrAlias string, card ModelCard) core.Result {
	res := r.Resolve(idOrAlias)
	if !res.OK {
		return res
	}
	e := res.Value.(Entry)
	c := card
	e.Card = &c
	return r.Put(e)
}

// GetCard returns the ModelCard on the entry resolved from idOrAlias. It fails
// when no entry resolves, and fails distinctly when the entry exists but carries
// no card — an absent card is not an empty card.
//
//	card := r.GetCard("lemma").Value.(registry.ModelCard)
func (r *Registry) GetCard(idOrAlias string) core.Result {
	res := r.Resolve(idOrAlias)
	if !res.OK {
		return res
	}
	e := res.Value.(Entry)
	if e.Card == nil {
		return core.Fail(core.E("registry.GetCard",
			core.Sprintf("entry %q has no model card", e.ID), nil))
	}
	return core.Ok(*e.Card)
}
