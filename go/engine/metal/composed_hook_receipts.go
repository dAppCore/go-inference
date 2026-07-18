// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync/atomic"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/attn"
	"dappco.re/go/inference/model/composed"
)

// ComposedHookPhaseCounts separates batched prefill (L>1) from single-token
// decode (L=1). A count records invocation of the device hook, including an
// invocation which returns an error and lets the caller fall back to the host.
type ComposedHookPhaseCounts struct {
	Prefill uint64
	Decode  uint64
}

// ComposedHookCounts is a snapshot of every device-hook site reachable from a
// composed model. The explicit fields make test failures identify the precise
// seam that engaged (or unexpectedly did not).
type ComposedHookCounts struct {
	Projection      ComposedHookPhaseCounts
	MLP             ComposedHookPhaseCounts
	AttentionQKV    ComposedHookPhaseCounts
	GatedDeltaInput ComposedHookPhaseCounts
	ResidualTail    ComposedHookPhaseCounts
	ProjectionTail  ComposedHookPhaseCounts
	AttentionInput  ComposedHookPhaseCounts
	GatedDeltaFold  ComposedHookPhaseCounts
	Mamba2Input     ComposedHookPhaseCounts
	RWKV7Input      ComposedHookPhaseCounts
	Head            ComposedHookPhaseCounts
	// QuantProjection counts the PACKED-weight matvec seam (composed/attn.ProjQuantMatMulInto). A
	// quantised composed model serves through this per-projection quant path — the f32 fold ladder above
	// is bypassed (it takes f32 weights), so for a packed checkpoint this is the per-projection seam
	// that engages while every f32 fold seam stays zero.
	QuantProjection ComposedHookPhaseCounts
	// QuantResidualTail counts the PACKED-weight fused FFN tail (composed.ResidualNormMLPQuantDevice,
	// #8-B slice 1): ONE command buffer swallowing the layer's gate/up/down quant projections plus the
	// residual/norm/silu glue — each hit here replaced three QuantProjection round trips.
	QuantResidualTail ComposedHookPhaseCounts
	// The whole-layer/half-layer fold seams (#26/#18): one hit = one command buffer that swallowed
	// a full gated-delta layer, an attention front (norm+q/k/v), or an attention tail (o_proj+FFN).
	GatedDeltaLayerFold ComposedHookPhaseCounts
	AttnFrontFold       ComposedHookPhaseCounts
	AttnTailFold        ComposedHookPhaseCounts
	// AttnFullLayerFold counts the device-KV whole-attention-layer seam (AttnBF16FullLayerDevice /
	// AttnQuantFullLayerDevice, #26 device-KV): the whole layer — norm, q/k/v, rope, SDPA over the
	// resident cache, o_proj, FFN tail — in ONE command buffer, superseding AttnFrontFold/
	// AttnTailFold's two-CB split. A hit here with AttnFrontFold/AttnTailFold at zero means the
	// device-KV path served every attention layer directly.
	AttnFullLayerFold ComposedHookPhaseCounts
	// ChainedForward counts ComposedChainBeginDevice: one hit = one WHOLE-TOKEN forward that rode
	// the chain (#26 whole-token chain, bf16 or quant) — every layer's encode landed on ONE retained
	// command buffer, one upload, one wait. A chained forward engages NONE of the per-layer/per-fold
	// hooks above (they are superseded wholesale), so this is the census's top-level signal: a hit
	// here with every other counter at zero means the chain served the token, not the host.
	ChainedForward ComposedHookPhaseCounts
}

const (
	hookProjection = iota
	hookMLP
	hookAttentionQKV
	hookGatedDeltaInput
	hookResidualTail
	hookProjectionTail
	hookAttentionInput
	hookGatedDeltaFold
	hookMamba2Input
	hookRWKV7Input
	hookHead
	hookQuantProjection
	hookQuantResidualTail
	hookGatedDeltaLayerFold
	hookAttnFrontFold
	hookAttnTailFold
	hookAttnFullLayerFold
	hookChainedForward
	hookCount
)

// ComposedHookReceiptGuard owns opt-in counter wrappers. With no guard, init
// binds the production hooks directly to their implementations: there is no
// counter load, branch, or atomic operation on the hot path. Guards are for a
// single debug/test run and must not overlap.
type ComposedHookReceiptGuard struct {
	prefill [hookCount]atomic.Uint64
	decode  [hookCount]atomic.Uint64
	restore func()
}

func (g *ComposedHookReceiptGuard) hit(site, L int) {
	if L > 1 {
		g.prefill[site].Add(1)
		return
	}
	g.decode[site].Add(1)
}

// Close restores the exact hook bindings present when the guard was enabled.
func (g *ComposedHookReceiptGuard) Close() {
	if g != nil && g.restore != nil {
		g.restore()
		g.restore = nil
	}
}

// Snapshot returns a race-safe point-in-time copy of the counters.
func (g *ComposedHookReceiptGuard) Snapshot() ComposedHookCounts {
	phase := func(site int) ComposedHookPhaseCounts {
		return ComposedHookPhaseCounts{Prefill: g.prefill[site].Load(), Decode: g.decode[site].Load()}
	}
	return ComposedHookCounts{
		Projection: phase(hookProjection), MLP: phase(hookMLP), AttentionQKV: phase(hookAttentionQKV),
		GatedDeltaInput: phase(hookGatedDeltaInput), ResidualTail: phase(hookResidualTail),
		ProjectionTail: phase(hookProjectionTail), AttentionInput: phase(hookAttentionInput),
		GatedDeltaFold: phase(hookGatedDeltaFold), Mamba2Input: phase(hookMamba2Input),
		RWKV7Input: phase(hookRWKV7Input), Head: phase(hookHead),
		QuantProjection:     phase(hookQuantProjection),
		QuantResidualTail:   phase(hookQuantResidualTail),
		GatedDeltaLayerFold: phase(hookGatedDeltaLayerFold),
		AttnFrontFold:       phase(hookAttnFrontFold),
		AttnTailFold:        phase(hookAttnTailFold),
		AttnFullLayerFold:   phase(hookAttnFullLayerFold),
		ChainedForward:      phase(hookChainedForward),
	}
}

// EnableComposedHookReceipts replaces each currently-bound composed device hook
// with a counting wrapper. Disabled hooks remain nil, so a receipt also shows
// the effective environment-gated configuration. Call Close before returning.
func EnableComposedHookReceipts() *ComposedHookReceiptGuard {
	g := &ComposedHookReceiptGuard{}
	proj, mlp, aqkv, gdInput := composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice, attn.GatedDeltaInputDevice
	resTail, projTail := composed.ResidualNormMLPDevice, composed.ResidualNormMLPProjDevice
	attnIn, gdFold := composed.ResidualNormMLPProjAttnInputDevice, composed.ResidualNormMLPProjGatedDeltaInputDevice
	mambaIn, rwkvIn, head := composed.ResidualNormMLPProjMamba2InputDevice, composed.ResidualNormMLPProjRWKV7InputDevice, composed.ResidualNormMLPProjHeadDevice
	quantProj, quantProjGD := composed.ProjQuantMatMulInto, attn.ProjQuantMatMulInto
	quantTail := composed.ResidualNormMLPQuantDevice
	gdLayerQ, gdLayerB := attn.GatedDeltaQuantLayerDevice, attn.GatedDeltaBF16LayerDevice
	attnFrontQ, attnTailQ := composed.AttnQuantFrontDevice, composed.AttnQuantTailDevice
	attnFrontB, attnTailB := composed.AttnBF16FrontDevice, composed.AttnBF16TailDevice
	attnFullQ, attnFullB := composed.AttnQuantFullLayerDevice, composed.AttnBF16FullLayerDevice
	chainBegin := composed.ComposedChainBeginDevice
	g.restore = func() {
		composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice, attn.GatedDeltaInputDevice = proj, mlp, aqkv, gdInput
		composed.ResidualNormMLPDevice, composed.ResidualNormMLPProjDevice = resTail, projTail
		composed.ResidualNormMLPProjAttnInputDevice, composed.ResidualNormMLPProjGatedDeltaInputDevice = attnIn, gdFold
		composed.ResidualNormMLPProjMamba2InputDevice, composed.ResidualNormMLPProjRWKV7InputDevice, composed.ResidualNormMLPProjHeadDevice = mambaIn, rwkvIn, head
		composed.ProjQuantMatMulInto, attn.ProjQuantMatMulInto = quantProj, quantProjGD
		composed.ResidualNormMLPQuantDevice = quantTail
		attn.GatedDeltaQuantLayerDevice, attn.GatedDeltaBF16LayerDevice = gdLayerQ, gdLayerB
		composed.AttnQuantFrontDevice, composed.AttnQuantTailDevice = attnFrontQ, attnTailQ
		composed.AttnBF16FrontDevice, composed.AttnBF16TailDevice = attnFrontB, attnTailB
		composed.AttnQuantFullLayerDevice, composed.AttnBF16FullLayerDevice = attnFullQ, attnFullB
		composed.ComposedChainBeginDevice = chainBegin
	}
	if proj != nil {
		composed.ProjMatMulInto = func(out, x, w []float32, M, K, N int) ([]float32, error) {
			g.hit(hookProjection, M)
			return proj(out, x, w, M, K, N)
		}
	}
	if mlp != nil {
		composed.MLPDevice = func(gate, up, down, x []float32, L, D, FF int) ([]float32, error) {
			g.hit(hookMLP, L)
			return mlp(gate, up, down, x, L, D, FF)
		}
	}
	if aqkv != nil {
		composed.AttnQKVDevice = func(h, qw, kw, vw []float32, L, D, qc, kvc int) ([]float32, []float32, []float32, error) {
			g.hit(hookAttentionQKV, L)
			return aqkv(h, qw, kw, vw, L, D, qc, kvc)
		}
	}
	if gdInput != nil {
		attn.GatedDeltaInputDevice = func(x, qw, zw, aw, bw []float32, L, D, cd, vd, vh int) ([]float32, []float32, []float32, []float32, error) {
			g.hit(hookGatedDeltaInput, L)
			return gdInput(x, qw, zw, aw, bw, L, D, cd, vd, vh)
		}
	}
	if resTail != nil {
		composed.ResidualNormMLPDevice = func(h, mix, nw, gate, up, down []float32, L, D, FF int, eps float32) ([]float32, error) {
			g.hit(hookResidualTail, L)
			return resTail(h, mix, nw, gate, up, down, L, D, FF, eps)
		}
	}
	if projTail != nil {
		composed.ResidualNormMLPProjDevice = func(mh, pw, h, nw, gate, up, down []float32, L, D, mc, FF int, eps float32) ([]float32, error) {
			g.hit(hookProjectionTail, L)
			return projTail(mh, pw, h, nw, gate, up, down, L, D, mc, FF, eps)
		}
	}
	if attnIn != nil {
		composed.ResidualNormMLPProjAttnInputDevice = func(mh, pw, h, nw, gate, up, down []float32, L, D, mc, FF int, eps float32, nn, qw, kw, vw []float32, qc, kvc int) ([]float32, []float32, []float32, []float32, error) {
			g.hit(hookAttentionInput, L)
			return attnIn(mh, pw, h, nw, gate, up, down, L, D, mc, FF, eps, nn, qw, kw, vw, qc, kvc)
		}
	}
	if gdFold != nil {
		composed.ResidualNormMLPProjGatedDeltaInputDevice = func(mh, pw, h, nw, gate, up, down []float32, L, D, mc, FF int, eps float32, nn, qw, zw, aw, bw []float32, cd, vd, vh int) ([]float32, []float32, []float32, []float32, []float32, error) {
			g.hit(hookGatedDeltaFold, L)
			return gdFold(mh, pw, h, nw, gate, up, down, L, D, mc, FF, eps, nn, qw, zw, aw, bw, cd, vd, vh)
		}
	}
	if mambaIn != nil {
		composed.ResidualNormMLPProjMamba2InputDevice = func(mh, pw, h, nw, gate, up, down []float32, L, D, mc, FF int, eps float32, nn, iw []float32, pd int) ([]float32, []float32, error) {
			g.hit(hookMamba2Input, L)
			return mambaIn(mh, pw, h, nw, gate, up, down, L, D, mc, FF, eps, nn, iw, pd)
		}
	}
	if rwkvIn != nil {
		composed.ResidualNormMLPProjRWKV7InputDevice = func(mh, pw, h, nw, gate, up, down []float32, L, D, mc, FF int, eps float32, nn, rw, ww, kw, vw, aw, bw []float32, hk, hv int) ([]float32, []float32, []float32, []float32, []float32, []float32, []float32, error) {
			g.hit(hookRWKV7Input, L)
			return rwkvIn(mh, pw, h, nw, gate, up, down, L, D, mc, FF, eps, nn, rw, ww, kw, vw, aw, bw, hk, hv)
		}
	}
	if head != nil {
		composed.ResidualNormMLPProjHeadDevice = func(mh, pw, h, nw, gate, up, down []float32, L, D, mc, FF int, eps float32, nf, hw []float32, vocab int) ([]float32, []float32, error) {
			g.hit(hookHead, L)
			return head(mh, pw, h, nw, gate, up, down, L, D, mc, FF, eps, nf, hw, vocab)
		}
	}
	if quantProj != nil {
		composed.ProjQuantMatMulInto = func(out, x []float32, packed, scales, biases []byte, M, K, N, gs, bits int) ([]float32, error) {
			g.hit(hookQuantProjection, M)
			return quantProj(out, x, packed, scales, biases, M, K, N, gs, bits)
		}
	}
	if quantProjGD != nil {
		attn.ProjQuantMatMulInto = func(out, x []float32, packed, scales, biases []byte, M, K, N, gs, bits int) ([]float32, error) {
			g.hit(hookQuantProjection, M)
			return quantProjGD(out, x, packed, scales, biases, M, K, N, gs, bits)
		}
	}
	if quantTail != nil {
		composed.ResidualNormMLPQuantDevice = func(h, mix, nw []float32, gate, up, down *model.QuantWeight, L, D, FF int, eps float32) ([]float32, error) {
			g.hit(hookQuantResidualTail, L)
			return quantTail(h, mix, nw, gate, up, down, L, D, FF, eps)
		}
	}
	g.wrapFoldHooks()
	return g
}

// wrapFoldHooks installs counting wrappers over the whole-layer/half-layer fold seams — split out
// of EnableComposedHookReceipts only to keep that function readable; called from it.
func (g *ComposedHookReceiptGuard) wrapFoldHooks() {
	if gdQ := attn.GatedDeltaQuantLayerDevice; gdQ != nil {
		attn.GatedDeltaQuantLayerDevice = func(sc *attn.GatedDeltaScratch, x, inputNorm []float32, w *model.GatedDeltaWeights, cfg model.GatedDeltaConfig, postNorm []float32, gate, up, down *model.QuantWeight, L, D, FF int, eps float32, priorConv, priorDelta []float32) ([]float32, error) {
			g.hit(hookGatedDeltaLayerFold, L)
			return gdQ(sc, x, inputNorm, w, cfg, postNorm, gate, up, down, L, D, FF, eps, priorConv, priorDelta)
		}
	}
	if gdB := attn.GatedDeltaBF16LayerDevice; gdB != nil {
		attn.GatedDeltaBF16LayerDevice = func(sc *attn.GatedDeltaScratch, x, inputNorm []float32, w *model.GatedDeltaWeights, cfg model.GatedDeltaConfig, postNorm []float32, gate, up, down *model.BF16Weight, L, D, FF int, eps float32, priorConv, priorDelta []float32) ([]float32, error) {
			g.hit(hookGatedDeltaLayerFold, L)
			return gdB(sc, x, inputNorm, w, cfg, postNorm, gate, up, down, L, D, FF, eps, priorConv, priorDelta)
		}
	}
	if fq := composed.AttnQuantFrontDevice; fq != nil {
		composed.AttnQuantFrontDevice = func(x, inputNorm []float32, qw, kw, vw *model.QuantWeight, L, D, qCols, kvCols int, eps float32) ([]float32, []float32, []float32, error) {
			g.hit(hookAttnFrontFold, L)
			return fq(x, inputNorm, qw, kw, vw, L, D, qCols, kvCols, eps)
		}
	}
	if tq := composed.AttnQuantTailDevice; tq != nil {
		composed.AttnQuantTailDevice = func(h, attnOut []float32, ow *model.QuantWeight, postNorm []float32, gate, up, down *model.QuantWeight, L, D, mixCols, FF int, eps float32) ([]float32, error) {
			g.hit(hookAttnTailFold, L)
			return tq(h, attnOut, ow, postNorm, gate, up, down, L, D, mixCols, FF, eps)
		}
	}
	if fb := composed.AttnBF16FrontDevice; fb != nil {
		composed.AttnBF16FrontDevice = func(x, inputNorm []float32, qw, kw, vw *model.BF16Weight, L, D, qCols, kvCols int, eps float32) ([]float32, []float32, []float32, error) {
			g.hit(hookAttnFrontFold, L)
			return fb(x, inputNorm, qw, kw, vw, L, D, qCols, kvCols, eps)
		}
	}
	if tb := composed.AttnBF16TailDevice; tb != nil {
		composed.AttnBF16TailDevice = func(h, attnOut []float32, ow *model.BF16Weight, postNorm []float32, gate, up, down *model.BF16Weight, L, D, mixCols, FF int, eps float32) ([]float32, error) {
			g.hit(hookAttnTailFold, L)
			return tb(h, attnOut, ow, postNorm, gate, up, down, L, D, mixCols, FF, eps)
		}
	}
	if fq := composed.AttnQuantFullLayerDevice; fq != nil {
		composed.AttnQuantFullLayerDevice = func(dev any, x, inputNorm []float32, qw, kw, vw, ow *model.QuantWeight, qNormW, kNormW, postNorm []float32, gate, up, down *model.QuantWeight, priorK, priorV []float32, L, D, H, KVH, HD, RD, pos0, window, gated, qkNorm, FF int, eps, theta float32) ([]float32, any, error) {
			g.hit(hookAttnFullLayerFold, L)
			return fq(dev, x, inputNorm, qw, kw, vw, ow, qNormW, kNormW, postNorm, gate, up, down, priorK, priorV, L, D, H, KVH, HD, RD, pos0, window, gated, qkNorm, FF, eps, theta)
		}
	}
	if fb := composed.AttnBF16FullLayerDevice; fb != nil {
		composed.AttnBF16FullLayerDevice = func(dev any, x, inputNorm []float32, qw, kw, vw, ow *model.BF16Weight, qNormW, kNormW, postNorm []float32, gate, up, down *model.BF16Weight, priorK, priorV []float32, L, D, H, KVH, HD, RD, pos0, window, gated, qkNorm, FF int, eps, theta float32) ([]float32, any, error) {
			g.hit(hookAttnFullLayerFold, L)
			return fb(dev, x, inputNorm, qw, kw, vw, ow, qNormW, kNormW, postNorm, gate, up, down, priorK, priorV, L, D, H, KVH, HD, RD, pos0, window, gated, qkNorm, FF, eps, theta)
		}
	}
	if cb := composed.ComposedChainBeginDevice; cb != nil {
		composed.ComposedChainBeginDevice = func(h []float32, L, D int) (any, error) {
			g.hit(hookChainedForward, L)
			return cb(h, L, D)
		}
	}
}
