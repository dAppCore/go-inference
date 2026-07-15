// SPDX-Licence-Identifier: EUPL-1.2

// Package builtin registers the built-in model architectures with the reactive
// loader ([model.Load]) by importing each arch package for its init()-side
// [model.RegisterArch]. A serve composition blank-imports this package once and
// every built-in arch becomes resolvable by model_type — the engine stays
// arch-agnostic (it never imports an arch), and adding one is a config + that
// arch's own init().
//
// This is the go-inference home of the arch wiring that lived in go-mlx's
// register_native.go ("the serve layer now imports them explicitly") — the
// pkg/metal-typed composition root that was retired rather than ported, taking
// the wiring with it.
//
//	import _ "dappco.re/go/inference/model/builtin" // all arches resolvable
//
// composed IS listed: it registers the Qwen 3.6 hybrids (qwen3_5 / qwen3_5_moe /
// qwen3_next) as top-level model_types through [model.ArchSpec].Composed, so a
// serve binary resolves them by model_type here rather than relying on an engine
// blank-import. The remaining component packages (deltanet, rwkv7) carry no
// top-level model_type and ride in transitively through the arches that compose
// them; the recurrent SSM mamba2 is reached by the backend's own hybrid/SSM
// branch, not this registry.
package builtin

import (
	_ "dappco.re/go/inference/model/arch/CohereForAI/cohere"      // cohere / cohere2 dense text
	_ "dappco.re/go/inference/model/arch/HuggingFaceTB/smollm3"   // SmolLM3 GQA / NoPE decoder
	_ "dappco.re/go/inference/model/arch/LGAI-EXAONE/exaone4"     // EXAONE 4 dense text
	_ "dappco.re/go/inference/model/arch/Qwen/qwen2"              // qwen2 / qwen2.5 dense text
	_ "dappco.re/go/inference/model/arch/Qwen/qwen3"              // qwen3
	_ "dappco.re/go/inference/model/arch/Qwen/qwenmoe"            // qwen2_moe / qwen3_moe sparse transformers
	_ "dappco.re/go/inference/model/arch/allenai/olmo"            // OLMo / OLMo 2
	_ "dappco.re/go/inference/model/arch/allenai/olmoe"           // AllenAI OLMoE sparse transformer
	_ "dappco.re/go/inference/model/arch/baidu/ernie45"           // ERNIE 4.5 dense text
	_ "dappco.re/go/inference/model/arch/bigcode/starcoder2"      // starcoder2
	_ "dappco.re/go/inference/model/arch/bloom"                   // bloom
	_ "dappco.re/go/inference/model/arch/databricks/dbrx"         // Databricks DBRX sparse MoE
	_ "dappco.re/go/inference/model/arch/deepseek-ai/deepseek"    // deepseek_v2 / deepseek_v3 (MLA declaration)
	_ "dappco.re/go/inference/model/arch/deepseek-ai/deepseekvl2" // deepseek_vl_v2 (DeepSeek-OCR arch; recognised, forward refused)
	_ "dappco.re/go/inference/model/arch/google/gemma3"           // gemma3
	_ "dappco.re/go/inference/model/arch/gpt2"                    // GPT-2 / GPT-SW3 / GPT-BigCode-StarCoder
	_ "dappco.re/go/inference/model/arch/gptneox"                 // gpt_neox / gptj / gpt_neo
	_ "dappco.re/go/inference/model/arch/ibm-granite/granite"     // IBM Granite dense (excludes Granite MoE/hybrid)
	_ "dappco.re/go/inference/model/arch/ibm-granite/granitemoe"  // IBM Granite sparse MoE
	_ "dappco.re/go/inference/model/arch/jetmoe"                  // JetMoE routed FFN (MoA gap reported explicitly)
	_ "dappco.re/go/inference/model/arch/meta-llama/llama"        // llama (dense text)
	_ "dappco.re/go/inference/model/arch/microsoft/phi"           // phi / phi3
	_ "dappco.re/go/inference/model/arch/mistralai/mistral"       // mistral
	_ "dappco.re/go/inference/model/arch/mistralai/mixtral"       // mixtral sparse MoE
	_ "dappco.re/go/inference/model/arch/mpt"                     // MPT ALiBi / learned-position decoder
	_ "dappco.re/go/inference/model/arch/openai/gptoss"           // gpt_oss — generative MoE causal-LM, recognised + configured (forward not yet validated)
	_ "dappco.re/go/inference/model/arch/openai/privacyfilter"    // openai_privacy_filter — PII token-classifier (MoE), not generative; recognised for direction
	_ "dappco.re/go/inference/model/arch/openai/whisper"          // whisper — ASR encoder-decoder (speech-to-text); recognised, forward not yet implemented
	_ "dappco.re/go/inference/model/arch/opt"                     // OPT learned-position transformer
	_ "dappco.re/go/inference/model/arch/rednote-hilab/dotsocr"   // dots_ocr / dots_ocr_1_5 (DOTS-OCR arch; recognised, forward refused)
	_ "dappco.re/go/inference/model/arch/stabilityai/stablelm"    // StableLM partial-RoPE decoder
	_ "dappco.re/go/inference/model/arch/tencent/hunyuan"         // HunYuan v1 dense text
	_ "dappco.re/go/inference/model/arch/tiiuae/falcon"           // falcon (ALiBi transformer; excludes Falcon-H1)
	_ "dappco.re/go/inference/model/arch/zai-org/glm4"            // GLM-4 dense text
	_ "dappco.re/go/inference/model/arch/zai-org/glmocr"          // glm_ocr / glm_ocr_text (GLM-OCR arch; recognised, forward refused)
	_ "dappco.re/go/inference/model/composed"                     // qwen3_5 / qwen3_5_moe / qwen3_next hybrids (ArchSpec.Composed)
	_ "dappco.re/go/inference/model/gemma4"                       // gemma4 / gemma4_text / gemma4_unified (+ assistant) — NOT yet homed under model/arch/google: the fenced engine/hip imports it (see report)
)
