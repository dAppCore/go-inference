// SPDX-Licence-Identifier: EUPL-1.2

package profile

// ResolveArchitecture maps the signals a model's config.json carries —
// top-level model_type, the text_config.model_type of a multimodal wrapper, and
// the architectures class list — to the registered model id the loader
// dispatches on. It is the single home for the resolution ORDER and for the
// family refinements that previously lived as name-branches in the metal
// loader, so a new family is supported by adding registry data, not loader code.
//
// Order, most authoritative first:
//
//  1. A top-level model_type, canonicalised through NormalizeArchitecture, then
//     refined: a multimodal wrapper resolves to its declared text tower
//     (TextTowerID); a base encoder whose architectures name a same-family
//     cross-encoder resolves to that rerank id.
//  2. Otherwise a text_config.model_type, canonicalised.
//  3. Otherwise the first architectures class name that maps to a known family.
//
// An empty result means none of the signals named a recognised architecture.
//
//	id := profile.ResolveArchitecture("gemma4", "gemma4_text", []string{"Gemma4ForConditionalGeneration"})  // → "gemma4_text"
func ResolveArchitecture(modelType, textTowerModelType string, architectures []string) string {
	if modelType != "" {
		id := NormalizeArchitecture(modelType)
		if tower := textTowerRefinement(id, textTowerModelType); tower != "" {
			return tower
		}
		if rerank := rerankRefinement(id, architectures); rerank != "" {
			return rerank
		}
		return id
	}
	if textTowerModelType != "" {
		return NormalizeArchitecture(textTowerModelType)
	}
	for _, arch := range architectures {
		if id := ArchitectureFromTransformersName(arch); id != "" {
			return id
		}
	}
	return ""
}

// textTowerRefinement resolves a multimodal wrapper id to its declared text
// tower when the config's text_config.model_type names that tower. Only a
// profile that declares a TextTowerID (the Gemma-4 multimodal wrapper) can be
// refined, so every other family — including the unified 12B id and the text
// tower itself — is returned unchanged.
func textTowerRefinement(id, textTowerModelType string) string {
	if textTowerModelType == "" {
		return ""
	}
	base, ok := LookupArchitectureProfileRef(id)
	if !ok || base.TextTowerID == "" {
		return ""
	}
	if NormalizeArchitecture(textTowerModelType) == base.TextTowerID {
		return base.TextTowerID
	}
	return ""
}

// rerankRefinement resolves a base encoder id to a cross-encoder sibling when
// the architectures name one. The sibling is found in the registry — a profile
// in the same family that advertises Rerank and whose class-name aliases the
// architectures match — so the only family this fires for is the one that
// registers such a sibling (BERT → bert_rerank), and a base id that is itself a
// reranker is left alone.
func rerankRefinement(id string, architectures []string) string {
	base, ok := LookupArchitectureProfileRef(id)
	if !ok || base.Rerank {
		return ""
	}
	for _, arch := range architectures {
		cand, ok := LookupArchitectureProfileRef(arch)
		if ok && cand.Rerank && cand.Family == base.Family {
			return cand.ID
		}
	}
	return ""
}
