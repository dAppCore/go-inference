// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// init registers glm_ocr and its nested text_config alias glm_ocr_text
// (https://huggingface.co/zai-org/GLM-OCR) as RECOGNISED model_types with a clean refusal, not
// an "unknown model architecture": a user pointing lem generate/serve at a GLM-OCR checkpoint
// gets direction (this IS glm_ocr, here is what's missing), never a mystery. Distinct from
// ../glm4 (dense-text GLM-4, no vision) — GLM-OCR is a separate wrapper arch, not a variant of
// it. Both loader entry points refuse: Parse succeeds and Arch refuses (the model.Load path),
// and Composed refuses directly (the model.LoadComposedDir path engine/metal tries first) — so
// whichever of THOSE entry points a backend reaches, the refusal surfaces with the same
// explanation. This mirrors ../../deepseek-ai/deepseek's registration for the identical
// "recognised, not yet implemented" contract — "not yet implemented" for Arch/Composed
// specifically: the actual OCR forward image-in/text-out IS implemented, as ocr.go's own
// standalone Load/Model.OCR path (never entering model.Assemble — see the package doc comment
// in config.go), the same split ../../openai/whisper uses for its ASR forward.
func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"glm_ocr", "glm_ocr_text"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			return ParseConfig(data)
		},
		Composed: func(_ map[string]safetensors.Tensor, cfgJSON []byte) (model.TokenModel, error) {
			mt, textMT := model.ProbeModelTypes(cfgJSON)
			if mt == "" {
				mt = textMT
			}
			if mt == "" {
				mt = "glm_ocr"
			}
			return nil, core.NewError(mt + " (GLM-OCR) is a recognised OCR vision-language arch; its vision encoder + OCR decoder forward is not yet implemented in this engine")
		},
	})
}
