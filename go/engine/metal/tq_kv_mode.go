// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// tq_kv_mode.go — the -kv-cache mode contract for the TurboQuant live KV lane
// (campaign #41 S3). The mode strings are FIXED (both ends of the campaign use
// exactly these): "turboquant" (bare = 3.5), "turboquant:4", "turboquant:3.5",
// "turboquant:3", "turboquant:2". The 3.5 mode is mixed — K at 4 bits, V at 3
// bits (K is the score-sensitive side).

// tqKVConfig is a parsed TurboQuant live-cache request: the per-side bit
// widths the qualifying GLOBAL owner layers store codes at.
type tqKVConfig struct {
	kBits, vBits int
}

// tqCacheModeName is the base mode string SupportedCacheModes reports.
const tqCacheModeName = "turboquant"

// parseTurboQuantCacheMode resolves a -kv-cache mode string: nil for the
// native default ("" / "native"), a tqKVConfig for the TurboQuant family, and
// a loud error for anything else — an unknown mode must never run silently
// native.
func parseTurboQuantCacheMode(mode string) (*tqKVConfig, error) {
	switch core.Lower(core.Trim(mode)) {
	case "", "native":
		return nil, nil
	case tqCacheModeName, tqCacheModeName + ":3.5":
		return &tqKVConfig{kBits: 4, vBits: 3}, nil
	case tqCacheModeName + ":4":
		return &tqKVConfig{kBits: 4, vBits: 4}, nil
	case tqCacheModeName + ":3":
		return &tqKVConfig{kBits: 3, vBits: 3}, nil
	case tqCacheModeName + ":2":
		return &tqKVConfig{kBits: 2, vBits: 2}, nil
	default:
		return nil, core.NewError("native: unknown -kv-cache mode " + mode + " (this engine serves: native, turboquant, turboquant:4, turboquant:3.5, turboquant:3, turboquant:2)")
	}
}

// tqKVArchServable is the load-time qualification for a TurboQuant live-cache
// request against an assembled arch, applying the per-LAYER-KIND matrix
// (docs/design-tq-moe-hybrid.md) — never a family-name blanket. MoE is an FFN
// property, not a cache kind: a MoE stack's standard attention layers qualify
// (it decodes on the state carrier — stepToken cannot record the host router,
// but the KV contract is untouched). Recurrent mixers (gated-delta) hold
// state, not KV rows — they are simply not TQ candidates, never a refusal on
// their own. Two things still refuse arch-wide: gated full attention
// (attn_output_gate — its KV lives in the gated/fused lane's resident state,
// which no TQ path is wired through) and attention sinks (gpt-oss: the sink
// joins the softmax denominator — the TQ read kernels carry no sinks lane).
// A stack the matrix leaves with NO qualifying global owner refuses too — a
// "turboquant" session that would quantise nothing must not pretend. layers
// may be nil (hand-built callers): the sinks check then rides the
// spec-declared arch alone.
func tqKVArchServable(arch model.Arch, layers []model.LoadedLayer, tq *tqKVConfig) error {
	if arch.AttnOutputGate {
		return core.NewError("native: -kv-cache turboquant declines gated-attention archs (attn_output_gate KV lives in the gated/fused decode lane, which has no TurboQuant wiring, v1) — reload without -kv-cache")
	}
	qualifying := 0
	for li := range arch.Layer {
		sp := arch.Layer[li]
		if li < len(layers) && len(layers[li].Sinks) > 0 {
			return core.NewError("native: -kv-cache turboquant declines attention-sinks models (the TQ read kernels carry no sinks lane, v1) — reload without -kv-cache")
		}
		if sp.Mixer != model.MixerAttention {
			continue // recurrent state layer — no KV rows to quantise; native state by construction
		}
		if sp.OwnsCache() && sp.Attention == model.GlobalAttention &&
			tqKVGeometryOK(tq.kBits, tq.vBits, headDimOf(sp, arch.HeadDim)) {
			qualifying++
		}
	}
	if qualifying == 0 {
		return core.NewError("native: -kv-cache turboquant found no qualifying global attention layer (head dim must be 128/256/512) — reload without -kv-cache")
	}
	return nil
}
