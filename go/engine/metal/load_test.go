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
