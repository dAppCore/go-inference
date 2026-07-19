// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import core "dappco.re/go"

// config.go reads the two facts about a checkpoint that loader.go genuinely cannot derive from the weight
// TENSORS themselves (norm_eps is a bare number; hidden_act/attn are architecture-choice flags), mirroring
// mamba2's mamba2EpsFromConfig — every other geometry fact (H, head_dim, value_dim, the LoRA ranks,
// intermediate_size, whether pre_norm/bias/v_lora/lm_head exist) is read straight off the checkpoint's own
// tensor shapes and key presence in loader.go, the "never guessed" rule applied to config.json too: a
// stale or community-edited config cannot desync the loaded geometry from the actual weights.

// epsFromConfig reads norm_eps from the checkpoint config, defaulting to 1e-5 — RWKV7Config's own default
// and the value every released RWKV-7 checkpoint (including RWKV7-Goose-World2.8-0.1B-HF) declares.
func epsFromConfig(cfg []byte) float32 {
	var probe struct {
		NormEps float32 `json:"norm_eps"`
	}
	_ = core.JSONUnmarshal(cfg, &probe)
	if probe.NormEps > 0 {
		return probe.NormEps
	}
	return 1e-5
}

// checkUnsupportedConfig refuses a config this host port does not implement: a hybrid attn block
// (config.json's "attn" key non-null — softmax-attention layers interleaved with RWKV-7 time-mix,
// RWKV7Config's hybrid escape hatch) or a channel-mix activation other than squared-ReLU (hidden_act,
// every released checkpoint's "sqrelu" — a different value would silently mis-serve since
// channelmix.go hardcodes sqrelu). Refusing loudly at load beats a wrong forward pass discovered later.
func checkUnsupportedConfig(cfg []byte) error {
	var probe struct {
		Attn      any    `json:"attn"`
		HiddenAct string `json:"hidden_act"`
	}
	_ = core.JSONUnmarshal(cfg, &probe)
	if probe.Attn != nil {
		return core.NewError("rwkv7: hybrid attn config (config.json \"attn\") is not supported by this host port")
	}
	if probe.HiddenAct != "" && probe.HiddenAct != "sqrelu" {
		return core.NewError("rwkv7: hidden_act " + probe.HiddenAct + " is not supported (only sqrelu)")
	}
	return nil
}
