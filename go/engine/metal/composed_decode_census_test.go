// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
)

// TestComposedDecodeRoundTripCensus is the item-1 instrument for the parked composed fuse ladder: it
// counts, per SINGLE decode token (L=1 — the generation hot path, distinct from the L>1 prefill every
// existing engagement test exercises), how many command-buffer round trips each projection/fold seam
// still pays. The fold ladder is gated on L*D*FF >= deviceMinWork; at decode L=1 that reduces to
// D*FF >= 1<<20, so this census also proves whether the whole ladder even ENGAGES at decode or whether
// the floor knocks a token back onto the unfused per-projection path. It reports the census; it changes
// nothing. Fusing decisions are made against its numbers.
func TestComposedDecodeRoundTripCensus(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — composed decode round-trip census")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("Metal runtime unavailable — composed decode round-trip census: %v", err)
	}
	dir := os.Getenv("LTHN_COMPOSED_AB_MODEL")
	if dir == "" {
		dir = composedPrefillABDefaultDir
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("composed census checkpoint absent (%s)", dir)
	}
	tm, err := LoadTokenModelDir(dir, 1024)
	if err != nil {
		t.Fatalf("LoadTokenModelDir(%s): %v", dir, err)
	}
	sm, ok := tm.(model.SessionModel)
	if !ok {
		t.Fatalf("loaded model is %T, want model.SessionModel", tm)
	}
	sess, err := sm.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	bp, ok := sess.(model.BatchPrefillStepper)
	if !ok {
		t.Fatalf("session %T lacks BatchPrefillStepper", sess)
	}

	// Prime the per-layer state with a short prompt so the measured Step is a genuine decode over a
	// warmed KV cache / recurrent state, not a cold layer-0 pass. The census is taken over the Step ONLY.
	const promptLen = 16
	prompt := make([]int32, promptLen)
	for i := range prompt {
		prompt[i] = int32(16 + (i*37)%2048)
	}
	embs := make([][]byte, len(prompt))
	for i, id := range prompt {
		if embs[i], err = tm.Embed(id); err != nil {
			t.Fatalf("Embed(%d): %v", id, err)
		}
	}
	if _, err := bp.PrefillBatch(embs); err != nil {
		t.Fatalf("PrefillBatch: %v", err)
	}

	receipts := EnableComposedHookReceipts()
	if _, err := sess.Step(embs[0]); err != nil {
		receipts.Close()
		t.Fatalf("decode Step: %v", err)
	}
	c := receipts.Snapshot()
	receipts.Close()

	// Fused seams: each is ONE command buffer that swallowed projections which the unfused path would
	// have paid a standalone CB round trip for. Unfused seams: the lower-level per-projection hooks that
	// only fire when a fold fell through (nil hook, sub-floor shape, device error, or an un-foldable
	// mixer/layer transition). In a fully-engaged decode the unfused bucket is the fall-through residue.
	fused := c.ProjectionTail.Decode + c.AttentionInput.Decode + c.GatedDeltaFold.Decode +
		c.Mamba2Input.Decode + c.RWKV7Input.Decode + c.Head.Decode
	unfused := c.Projection.Decode + c.MLP.Decode + c.AttentionQKV.Decode +
		c.GatedDeltaInput.Decode + c.ResidualTail.Decode

	t.Logf("COMPOSED DECODE ROUND-TRIP CENSUS (one L=1 token, %s):", dir)
	t.Logf("  FUSED seams (each = 1 CB swallowing projections):")
	t.Logf("    ProjectionTail (o_proj+tail, no next-fold)      = %d", c.ProjectionTail.Decode)
	t.Logf("    AttentionInput (tail + next attn q/k/v)         = %d", c.AttentionInput.Decode)
	t.Logf("    GatedDeltaFold (tail + next gated-delta input)  = %d", c.GatedDeltaFold.Decode)
	t.Logf("    Mamba2Input    (tail + next mamba2 in_proj)     = %d", c.Mamba2Input.Decode)
	t.Logf("    RWKV7Input     (tail + next rwkv7 input)        = %d", c.RWKV7Input.Decode)
	t.Logf("    Head           (last tail + final norm + LM)    = %d", c.Head.Decode)
	t.Logf("  UNFUSED seams (per-projection fall-through CBs):")
	t.Logf("    Projection (single device GEMM)                 = %d", c.Projection.Decode)
	t.Logf("    MLP        (fused SwiGLU, no proj fold)          = %d", c.MLP.Decode)
	t.Logf("    AttentionQKV (q/k/v, no tail fold)              = %d", c.AttentionQKV.Decode)
	t.Logf("    GatedDeltaInput (in_proj family, no tail fold)  = %d", c.GatedDeltaInput.Decode)
	t.Logf("    ResidualTail (tail only, no proj fold)          = %d", c.ResidualTail.Decode)
	// A PACKED checkpoint serves through the quant matvec seam, bypassing the f32 fold ladder above (which
	// stays all-zero by design — quant fused tails are a later slice), so count it as the engaged seam.
	quant := c.QuantProjection.Decode
	quantTail := c.QuantResidualTail.Decode
	folds := c.GatedDeltaLayerFold.Decode + c.AttnFrontFold.Decode + c.AttnTailFold.Decode
	// Whole-layer / whole-token seams (#26 device-KV + whole-token chain, bf16 or quant): a chained
	// decode engages NEITHER the fold ladder above NOR the per-projection seams — ChainedForward is a
	// hit whenever every layer of the model qualified for the session-wide chain; AttnFullLayerFold
	// covers a model that rides the device-KV whole-layer seam per-layer without qualifying globally.
	chainSeams := c.ChainedForward.Decode + c.AttnFullLayerFold.Decode
	t.Logf("  QUANT seams (packed-weight lanes):")
	t.Logf("    QuantResidualTail (fused tail over codes, #8-B)  = %d", quantTail)
	t.Logf("    QuantProjection   (per-projection fall-through)  = %d", quant)
	t.Logf("  FOLD seams (whole/half-layer CBs, #26/#18):")
	t.Logf("    GatedDeltaLayerFold (norm+projs+block+tail)      = %d", c.GatedDeltaLayerFold.Decode)
	t.Logf("    AttnFrontFold (norm + q/k/v)                     = %d", c.AttnFrontFold.Decode)
	t.Logf("    AttnTailFold (o_proj + FFN tail)                 = %d", c.AttnTailFold.Decode)
	t.Logf("  WHOLE-TOKEN seams (#26 device-KV + chain):")
	t.Logf("    AttnFullLayerFold (whole attn layer, device-KV)  = %d", c.AttnFullLayerFold.Decode)
	t.Logf("    ChainedForward (whole token, one retained CB)    = %d", c.ChainedForward.Decode)
	t.Logf("  TOTALS: fused=%d  unfused=%d  quantTail=%d  quantProj=%d  folds=%d  chainSeams=%d  (device-seam CBs per decode token)", fused, unfused, quantTail, quant, folds, chainSeams)

	if fused+unfused+quant+quantTail+folds+chainSeams == 0 {
		t.Fatalf("decode engaged NO composed device seam — floor knocked the whole token to host? census=%+v", c)
	}
}

// TestComposedDecodeFoldLadderEngagesHermetic is the durable (checkpoint-free) guard that the fold ladder
// engages at DECODE (L=1), not just at the prefill (L>1) every other engagement test exercises. It builds a
// 3-layer [gated-delta, attention, gated-delta] hybrid at D=512, FF=2048 — exactly the 1<<20 tail floor at
// L=1 — prefills, then steps ONE decode token under the receipt guard. Each layer's output-projection +
// FFN-tail + the NEXT layer's input-projection (or, for the last layer, the model's final RMSNorm + LM head)
// must ride ONE fold command buffer: the chain is fully folded, so ProjectionTail (the plain fall-through
// tail) and the lower-level MLP/ResidualTail seams must stay at zero. A retune of deviceMinWork or a change
// to forwardEmb's fold gate that silently disengaged decode would trip this guard where the prefill tests
// would not. Layer 0's own input projection is intentionally left sub-floor (small gated-delta geometry) —
// it has no predecessor tail to fold onto and is the ladder's irreducible floor, so it is reported, not
// asserted.
func TestComposedDecodeFoldLadderEngagesHermetic(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — composed decode fold-ladder guard")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("Metal runtime unavailable — composed decode fold-ladder guard: %v", err)
	}
	const D, FF, vocab = 512, 2048, 128 // D*FF == 1<<20: the tail fold engages at L=1
	newMLP := func(seed int) *composed.MLP {
		return &composed.MLP{Gate: cbSyn(FF*D, seed), Up: cbSyn(FF*D, seed+1), Down: cbSyn(D*FF, seed+2), FF: FF}
	}
	attnLayer := func(seed int) composed.Layer {
		const heads, hd = 4, 128 // heads*hd == D
		return composed.Layer{
			InputNorm: cbSyn(D, seed), PostAttnNorm: cbSyn(D, seed+1), MLP: newMLP(seed + 2),
			Mixer: composed.NewAttnMixer(&composed.AttnWeights{
				QProj: cbSyn(D*D, seed+5), KProj: cbSyn(D*D, seed+6), VProj: cbSyn(D*D, seed+7),
				OProj: cbSyn(D*D, seed+8), QNorm: cbSyn(hd, seed+9), KNorm: cbSyn(hd, seed+10),
			}, composed.AttnConfig{Heads: heads, KVHeads: heads, HeadDim: hd, RotaryDim: hd, RopeTheta: 1e6, NormEps: 1e-6}),
		}
	}
	gdLayer := func(seed int) composed.Layer {
		cfg := model.GatedDeltaConfig{KeyHeads: 4, ValueHeads: 4, HeadDim: 32, ConvKernel: 4, Eps: 1e-5}
		convDim, vDim := cfg.ConvDim(), cfg.VDim()
		return composed.Layer{
			InputNorm: cbSyn(D, seed), PostAttnNorm: cbSyn(D, seed+1), MLP: newMLP(seed + 2),
			Mixer: composed.NewGatedDeltaMixer(&model.GatedDeltaWeights{
				InProjQKV: cbSyn(convDim*D, seed+5), ConvWeight: cbSyn(convDim*cfg.ConvKernel, seed+6), ConvBias: cbSyn(convDim, seed+7),
				InProjA: cbSyn(cfg.ValueHeads*D, seed+8), ALog: cbSyn(cfg.ValueHeads, seed+9), DtBias: cbSyn(cfg.ValueHeads, seed+10),
				InProjB: cbSyn(cfg.ValueHeads*D, seed+11), InProjZ: cbSyn(vDim*D, seed+12), Norm: cbSyn(cfg.HeadDim, seed+13), OutProj: cbSyn(D*vDim, seed+14),
			}, cfg),
		}
	}
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6,
		Layers: []composed.Layer{gdLayer(100), attnLayer(200), gdLayer(300)},
	}

	sess := composed.NewSession(m)
	if _, err := sess.Forward([]int32{1, 2, 3, 4}); err != nil { // prime state
		t.Fatalf("prefill Forward: %v", err)
	}
	receipts := EnableComposedHookReceipts()
	if _, err := sess.Forward([]int32{5}); err != nil { // one L=1 decode token
		receipts.Close()
		t.Fatalf("decode Forward: %v", err)
	}
	c := receipts.Snapshot()
	receipts.Close()

	t.Logf("hermetic decode census: attn-input=%d gated-delta-fold=%d head=%d proj-tail=%d | mlp=%d residual-tail=%d gd-input=%d attn-qkv=%d",
		c.AttentionInput.Decode, c.GatedDeltaFold.Decode, c.Head.Decode, c.ProjectionTail.Decode,
		c.MLP.Decode, c.ResidualTail.Decode, c.GatedDeltaInput.Decode, c.AttentionQKV.Decode)

	// Chain: L0(gd) folds L1(attn) input; L1(attn) folds L2(gd) input; L2(gd) is last → head fold.
	if c.AttentionInput.Decode != 1 || c.GatedDeltaFold.Decode != 1 || c.Head.Decode != 1 {
		t.Fatalf("decode fold ladder did not fully engage: attn-input=%d gated-delta-fold=%d head=%d, want 1,1,1; census=%+v",
			c.AttentionInput.Decode, c.GatedDeltaFold.Decode, c.Head.Decode, c)
	}
	// Every layer folded, so nothing fell through to the plain proj-fused tail or the lower device seams.
	if c.ProjectionTail.Decode != 0 || c.MLP.Decode != 0 || c.ResidualTail.Decode != 0 {
		t.Fatalf("decode fold fell through to a lower seam: proj-tail=%d mlp=%d residual-tail=%d, want 0,0,0; census=%+v",
			c.ProjectionTail.Decode, c.MLP.Decode, c.ResidualTail.Decode, c)
	}
}
