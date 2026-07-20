// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/mistralai/mistral" // register the mistral loaders
	_ "dappco.re/go/inference/model/gemma4"                 // register the gemma4 loaders for the dispatch check
)

// TestLoadDirReactiveDispatch pins the reactive registry dispatch the generic loader runs:
// model.ProbeModelTypes peeks a config for its top-level model_type and the nested text_config
// model_type (a multimodal wrapper carries both), and model.LookupArch finds a registered ArchSpec
// for either — the arches the backend serves register every alias they declare. An unknown model_type
// resolves to no spec — a clean error, not a panic — so the backend stays model-agnostic: it knows
// the registry, not gemma4.
func TestLoadDirReactiveDispatch(t *testing.T) {
	hasLoader := func(cfg string) bool {
		modelType, textModelType := model.ProbeModelTypes([]byte(cfg))
		if _, ok := model.LookupArch(modelType); ok {
			return true
		}
		_, ok := model.LookupArch(textModelType)
		return ok
	}
	if !hasLoader(`{"model_type":"gemma4"}`) {
		t.Fatal("gemma4 config should dispatch to a registered loader")
	}
	if !hasLoader(`{"model_type":"gemma4_unified","text_config":{"model_type":"gemma4_text"}}`) {
		t.Fatal("gemma4 multimodal wrapper should dispatch to a registered loader")
	}
	if !hasLoader(`{"model_type":"mistral3","architectures":["Mistral3ForConditionalGeneration"],"text_config":{"model_type":"ministral3"}}`) {
		t.Fatal("mistral3 config should dispatch to a registered loader")
	}
	if hasLoader(`{"model_type":"nonesuch_arch"}`) {
		t.Fatal("an unregistered model_type must resolve to no loader")
	}
}

// TestIsQwen35FactoryType_Good pins the full set of Qwen 3.6 hybrid ids that default to the factory
// route: qwen35's ArchSpec.ModelTypes carries all seven under ONE dual-route declaration (#50 archzoo —
// qwen3_6/qwen3_6_moe/qwen3_next moved here from model/composed's own spec), so isQwen35FactoryType must
// recognise every one of them, not just the original qwen3_5* subset.
func TestIsQwen35FactoryType_Good(t *testing.T) {
	for _, mt := range []string{
		"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
		"qwen3_6", "qwen3_6_moe", "qwen3_next",
	} {
		if !isQwen35FactoryType(mt) {
			t.Errorf("isQwen35FactoryType(%q) = false, want true", mt)
		}
	}
}

// TestIsQwen35FactoryType_Bad proves the lever stays scoped to qwen35's own registered ids: a composed
// arch with its OWN separate factory route (mixtral, granitemoe, qwenmoe's qwen2_moe/qwen3_moe — none of
// them qwen35's dual-route declaration), an unrelated arch, and the empty/near-miss strings all resolve
// false — this switch does not generalise to "any dual-route arch", only qwen35's.
func TestIsQwen35FactoryType_Bad(t *testing.T) {
	for _, mt := range []string{
		"", "mixtral", "granitemoe", "qwen2_moe", "qwen3_moe", "gemma4", "qwen3_5x", "qwen3_6_moe_text", "composed", "hybrid",
	} {
		if isQwen35FactoryType(mt) {
			t.Errorf("isQwen35FactoryType(%q) = true, want false", mt)
		}
	}
}
