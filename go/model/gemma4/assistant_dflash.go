// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/mtp"
)

// assistant_dflash.go declares the DFlash block-diffusion drafter to the reactive
// assistant loader, the block-diffusion sibling of assistant.go's MTP -assistant.
// A DFlash speculator (arXiv 2602.06036) carries speculators_model_type "dflash"
// plus its own decoder arch; the engine loads that decoder through the ordinary
// assistant pack loader and this spec stamps mtp.MTPDFlash onto the neutral config
// so the decode path dispatches to the block-parallel draft forward rather than the
// autoregressive MTP one. The block size, fused verifier layers and reduced-vocab
// maps ride in the checkpoint and are read model-free by decode/dflash.ParseConfig —
// this spec owns only the decoder-arch derivation the pack loader needs.
//
// The recognised model_type is "gemma4_dflash_assistant" — a gemma4-family DFlash
// drafter whose decoder is a gemma4 text stack. A z-lab / RedHatAI DFlash checkpoint
// declares its base model_type (qwen3, llama, …) with the dflash marker beside it;
// recognising THAT by the speculators marker (rather than a bespoke model_type) is
// the follow-up once such a checkpoint is on the box — the neutral config and block
// forward are already method-generic.

func init() {
	mtp.RegisterAssistant(mtp.AssistantSpec{
		ModelTypes: []string{"gemma4_dflash_assistant"},
		Method:     mtp.MTPDFlash, // the block-diffusion draft forward, not the MTP one
		Parse:      ParseDFlashAssistantConfig,
	})
}

// dflashAssistantConfig is the raw config.json shape a DFlash drafter adds on top of
// the marker: the target-attachment backbone dim and the drafter's own decoder arch
// (nested under text_config, or flat in early exports — the same tolerance
// ParseAssistantConfig gives the MTP layout).
type dflashAssistantConfig struct {
	ModelType          string `json:"model_type"`
	BackboneHiddenSize int    `json:"backbone_hidden_size"`
	TextConfig         Config `json:"text_config"`
}

// ParseDFlashAssistantConfig derives the neutral mtp.AssistantConfig for a DFlash
// drafter: it resolves the decoder arch (nested-or-flat text_config) and validates
// the load-bearing dims, leaving the method to be stamped mtp.MTPDFlash by the
// registered spec. Registered as that spec's config.json parser.
func ParseDFlashAssistantConfig(data []byte) (mtp.AssistantConfig, error) {
	var raw dflashAssistantConfig
	if r := core.JSONUnmarshal(data, &raw); !r.OK {
		return mtp.AssistantConfig{}, core.NewError("gemma4.dflash config parse failed: " + r.Error())
	}
	text := raw.TextConfig
	if text.HiddenSize <= 0 && text.NumHiddenLayers <= 0 {
		// early exports carry the decoder arch FLAT rather than under text_config.
		var flat Config
		if r := core.JSONUnmarshal(data, &flat); !r.OK {
			return mtp.AssistantConfig{}, core.NewError("gemma4.dflash config parse failed: " + r.Error())
		}
		if flat.HiddenSize > 0 || flat.NumHiddenLayers > 0 {
			text = flat
		}
	}
	if raw.BackboneHiddenSize <= 0 {
		return mtp.AssistantConfig{}, core.NewError("gemma4.dflash config has invalid backbone_hidden_size")
	}
	if text.HiddenSize <= 0 || text.NumHiddenLayers <= 0 || text.NumAttentionHeads <= 0 || text.HeadDim <= 0 {
		return mtp.AssistantConfig{}, core.NewError("gemma4.dflash config has invalid decoder arch (hidden_size / num_hidden_layers / num_attention_heads / head_dim)")
	}
	arch, err := text.Arch()
	if err != nil {
		return mtp.AssistantConfig{}, core.E("gemma4.dflash", "derive decoder arch", err)
	}
	return mtp.AssistantConfig{
		ModelType:      "gemma4_dflash_assistant",
		BackboneHidden: raw.BackboneHiddenSize,
		LayerTypes:     text.LayerTypes,
		Arch:           arch,
		Quant:          text.ResolvedQuant(),
	}, nil
}
