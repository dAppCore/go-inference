// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync/atomic"

	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/qwen3"
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
	// QuantProjection counts the PACKED-weight matvec seam (composed/qwen3.ProjQuantMatMulInto). A
	// quantised composed model serves through this per-projection quant path — the f32 fold ladder above
	// is bypassed (it takes f32 weights; quant fused tails are a later slice), so for a packed checkpoint
	// this is the seam that engages while every f32 fold seam stays zero.
	QuantProjection ComposedHookPhaseCounts
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
		QuantProjection: phase(hookQuantProjection),
	}
}

// EnableComposedHookReceipts replaces each currently-bound composed device hook
// with a counting wrapper. Disabled hooks remain nil, so a receipt also shows
// the effective environment-gated configuration. Call Close before returning.
func EnableComposedHookReceipts() *ComposedHookReceiptGuard {
	g := &ComposedHookReceiptGuard{}
	proj, mlp, aqkv, gdInput := composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice, qwen3.GatedDeltaInputDevice
	resTail, projTail := composed.ResidualNormMLPDevice, composed.ResidualNormMLPProjDevice
	attnIn, gdFold := composed.ResidualNormMLPProjAttnInputDevice, composed.ResidualNormMLPProjGatedDeltaInputDevice
	mambaIn, rwkvIn, head := composed.ResidualNormMLPProjMamba2InputDevice, composed.ResidualNormMLPProjRWKV7InputDevice, composed.ResidualNormMLPProjHeadDevice
	quantProj, quantProjGD := composed.ProjQuantMatMulInto, qwen3.ProjQuantMatMulInto
	g.restore = func() {
		composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice, qwen3.GatedDeltaInputDevice = proj, mlp, aqkv, gdInput
		composed.ResidualNormMLPDevice, composed.ResidualNormMLPProjDevice = resTail, projTail
		composed.ResidualNormMLPProjAttnInputDevice, composed.ResidualNormMLPProjGatedDeltaInputDevice = attnIn, gdFold
		composed.ResidualNormMLPProjMamba2InputDevice, composed.ResidualNormMLPProjRWKV7InputDevice, composed.ResidualNormMLPProjHeadDevice = mambaIn, rwkvIn, head
		composed.ProjQuantMatMulInto, qwen3.ProjQuantMatMulInto = quantProj, quantProjGD
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
		qwen3.GatedDeltaInputDevice = func(x, qw, zw, aw, bw []float32, L, D, cd, vd, vh int) ([]float32, []float32, []float32, []float32, error) {
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
		qwen3.ProjQuantMatMulInto = func(out, x []float32, packed, scales, biases []byte, M, K, N, gs, bits int) ([]float32, error) {
			g.hit(hookQuantProjection, M)
			return quantProjGD(out, x, packed, scales, biases, M, K, N, gs, bits)
		}
	}
	return g
}
