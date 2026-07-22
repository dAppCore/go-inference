// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"math"

	core "dappco.re/go"
)

// hipQwen3MoEHiddenForward runs the qwen3_moe decode stack over T positions of one
// sequence, whole-sequence style: Backend.DecodeForward's documented contract ("the KV
// cache is built per call") — causal masking falls out of a position only ever attending
// K/V accumulated from positions <= itself, appended in position order. hidden is
// [T][HiddenSize] host float32, mutated in place — each layer's attention residual lands
// in hidden[t] before hidden[t] is read for that SAME layer's MoE FFN sub-block (the
// standard pre-norm transformer block order: h += Attn(Norm(h)); h += MoE(Norm(h))).
func hipQwen3MoEHiddenForward(ctx context.Context, driver nativeHIPDriver, cfg hipQwen3MoEConfig, weights *hipQwen3MoEWeights, hidden [][]float32) error {
	if weights == nil || len(weights.Layers) != cfg.NumLayers {
		return core.NewError("rocm.hip.Qwen3MoE: weights do not match layer count")
	}
	for layer := 0; layer < cfg.NumLayers; layer++ {
		lw := &weights.Layers[layer]
		keys := make([]float32, 0, len(hidden)*cfg.KVHeads*cfg.HeadDim)
		values := make([]float32, 0, len(hidden)*cfg.KVHeads*cfg.HeadDim)
		for t := range hidden {
			if err := hipQwen3MoEAttentionStep(ctx, driver, cfg, lw, hidden[t], t, &keys, &values); err != nil {
				return core.E("rocm.hip.Qwen3MoE", core.Sprintf("attention layer %d position %d", layer, t), err)
			}
		}
		for t := range hidden {
			if err := hipQwen3MoEFFNStep(ctx, driver, cfg, lw, hidden[t]); err != nil {
				return core.E("rocm.hip.Qwen3MoE", core.Sprintf("moe ffn layer %d position %d", layer, t), err)
			}
		}
	}
	return nil
}

// hipQwen3MoEAttentionStep runs one position's attention sub-block: input norm -> QKV
// projections -> fused per-head QK-norm+RoPE on Q and K -> append K/V to the layer's
// running (host-array) cache -> causal multi-head attention over every accumulated
// position -> output projection -> residual add into hidden (in place). keys/values grow
// by this position's contribution — the caller threads the SAME slices across every
// position of this layer so position t attends every position <= t.
func hipQwen3MoEAttentionStep(ctx context.Context, driver nativeHIPDriver, cfg hipQwen3MoEConfig, lw *hipQwen3MoELayerWeights, hidden []float32, position int, keys, values *[]float32) error {
	normed, err := hipRunRMSNormKernelWithDeviceWeight(ctx, driver, hidden, lw.InputNorm.Pointer(), lw.InputNorm.SizeBytes(), cfg.HiddenSize, cfg.Epsilon)
	if err != nil {
		return err
	}

	qDim := cfg.Heads * cfg.HeadDim
	kvDim := cfg.KVHeads * cfg.HeadDim

	q, err := hipRunProjectionKernelWithDeviceWeightEncoding(ctx, driver, normed, lw.QProj.Pointer(), lw.QProj.SizeBytes(), qDim, cfg.HiddenSize, hipProjectionWeightEncodingF32)
	if err != nil {
		return err
	}
	k, err := hipRunProjectionKernelWithDeviceWeightEncoding(ctx, driver, normed, lw.KProj.Pointer(), lw.KProj.SizeBytes(), kvDim, cfg.HiddenSize, hipProjectionWeightEncodingF32)
	if err != nil {
		return err
	}
	v, err := hipRunProjectionKernelWithDeviceWeightEncoding(ctx, driver, normed, lw.VProj.Pointer(), lw.VProj.SizeBytes(), kvDim, cfg.HiddenSize, hipProjectionWeightEncodingF32)
	if err != nil {
		return err
	}

	ropedQ, err := hipQwen3MoEApplyQKNormRoPE(ctx, driver, q, lw.QNorm, cfg.Heads, cfg.HeadDim, position, cfg.RopeTheta, cfg.Epsilon)
	if err != nil {
		return err
	}
	ropedK, err := hipQwen3MoEApplyQKNormRoPE(ctx, driver, k, lw.KNorm, cfg.KVHeads, cfg.HeadDim, position, cfg.RopeTheta, cfg.Epsilon)
	if err != nil {
		return err
	}

	*keys = append(*keys, ropedK...)
	*values = append(*values, v...)

	queryBuf, err := hipQwen3MoEUploadFloat32(driver, "qwen3_moe attention query", ropedQ)
	if err != nil {
		return err
	}
	defer func() { _ = queryBuf.Close() }()
	outputBuf, err := hipAllocateByteBuffer(driver, "rocm.hip.Qwen3MoE", "qwen3_moe attention output", uint64(qDim*4), qDim)
	if err != nil {
		return err
	}
	defer func() { _ = outputBuf.Close() }()

	attnReq := hipAttentionRequest{
		QueryDim: cfg.HeadDim,
		KeyHeads: cfg.KVHeads,
		Keys:     *keys,
		Values:   *values,
		Scale:    float32(1 / math.Sqrt(float64(cfg.HeadDim))),
	}
	if err := hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernel(ctx, driver, attnReq, queryBuf, cfg.Heads, outputBuf); err != nil {
		return err
	}
	attnOut, err := hipReadFloat32DeviceOutput(outputBuf, "rocm.hip.Qwen3MoE", "attention output", qDim)
	if err != nil {
		return err
	}

	o, err := hipRunProjectionKernelWithDeviceWeightEncoding(ctx, driver, attnOut, lw.OProj.Pointer(), lw.OProj.SizeBytes(), cfg.HiddenSize, qDim, hipProjectionWeightEncodingF32)
	if err != nil {
		return err
	}
	for i := range hidden {
		hidden[i] += o[i]
	}
	return nil
}

// hipQwen3MoEApplyQKNormRoPE fuses per-head RMSNorm (the checkpoint's q_norm/k_norm
// weight) with rotary position encoding — the SAME device kernel gemma4's decoder layer
// uses for its own QK-norm (hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigFrequencyScale),
// applied here to qwen3_moe's Q and K projections. rotaryCount 0 selects full-head
// rotation (qwen3_moe rotates the whole head_dim, no partial rotary split).
func hipQwen3MoEApplyQKNormRoPE(ctx context.Context, driver nativeHIPDriver, values []float32, norm *hipDeviceByteBuffer, heads, headDim, position int, ropeTheta, eps float32) ([]float32, error) {
	buf, err := hipQwen3MoEUploadFloat32(driver, "qwen3_moe qk-norm+rope input", values)
	if err != nil {
		return nil, err
	}
	defer func() { _ = buf.Close() }()
	normCfg := hipRMSNormDeviceWeightConfig{
		WeightPointer:  norm.Pointer(),
		WeightBytes:    norm.SizeBytes(),
		Count:          headDim,
		Epsilon:        eps,
		WeightEncoding: hipRMSNormWeightEncodingF32,
		// qwen3 rotates the split-half convention (rotate_half: pairs (i, i+headDim/2)),
		// the SAME convention gemma4's own QK-norm+RoPE config sets
		// (hipGemma4Q4RoPENormConfig) — without this flag the kernel's other branch
		// pairs adjacent elements (i, i+1) instead (GPT-J-style), a coherent-but-wrong
		// rotation for this family.
		Flags: hipRMSNormLaunchFlagRoPENeoX,
	}
	out, err := hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigFrequencyScale(ctx, driver, buf, normCfg, heads, position, ropeTheta, 0, 0, 1)
	if err != nil {
		return nil, err
	}
	defer func() { _ = out.Close() }()
	return hipReadFloat32DeviceOutput(out, "rocm.hip.Qwen3MoE", "qk-norm+rope output", len(values))
}

// hipQwen3MoEFFNStep runs one position's MoE feed-forward sub-block: pre-MoE norm ->
// router projection (device) -> top-k selection with the checkpoint's declared combine
// order (host, hipQwen3MoERouterSelect) -> selected-expert SwiGLU matvecs, weighted and
// summed -> residual add into hidden (in place). qwen3_moe carries no shared expert (see
// hipQwen3MoELayerWeights' doc), so the routed sum IS the block's whole contribution.
func hipQwen3MoEFFNStep(ctx context.Context, driver nativeHIPDriver, cfg hipQwen3MoEConfig, lw *hipQwen3MoELayerWeights, hidden []float32) error {
	normed, err := hipRunRMSNormKernelWithDeviceWeight(ctx, driver, hidden, lw.PostAttnNorm.Pointer(), lw.PostAttnNorm.SizeBytes(), cfg.HiddenSize, cfg.Epsilon)
	if err != nil {
		return err
	}

	logits, err := hipRunProjectionKernelWithDeviceWeightEncoding(ctx, driver, normed, lw.Router.Pointer(), lw.Router.SizeBytes(), cfg.NumExperts, cfg.HiddenSize, hipProjectionWeightEncodingF32)
	if err != nil {
		return err
	}

	idx, weights := hipQwen3MoERouterSelect(logits, cfg.TopK, cfg.NormaliseTopK)

	acc := make([]float32, cfg.HiddenSize)
	for i, e := range idx {
		down, err := hipQwen3MoEExpertForward(ctx, driver, cfg, lw, normed, e)
		if err != nil {
			return err
		}
		w := weights[i]
		for d := 0; d < cfg.HiddenSize; d++ {
			acc[d] += w * down[d]
		}
	}
	for i := range hidden {
		hidden[i] += acc[i]
	}
	return nil
}

// hipQwen3MoEExpertForward computes one selected expert's SwiGLU transform on an
// already-normed input: down @ (silu(gate@x) . (up@x)) — gate/up projections (device),
// the SiLU-gate elementwise multiply (device, rocm_swiglu via hipRunSwiGLUKernel — the
// kernel this whole family exists to finally wire a production caller to), down
// projection (device). hipQwen3MoESwiGLUHostReference is the host oracle this is tested
// against.
func hipQwen3MoEExpertForward(ctx context.Context, driver nativeHIPDriver, cfg hipQwen3MoEConfig, lw *hipQwen3MoELayerWeights, normed []float32, expert int) ([]float32, error) {
	if expert < 0 || expert >= len(lw.ExpertGate) {
		return nil, core.NewError("rocm.hip.Qwen3MoE: expert index out of range")
	}
	gate, err := hipRunProjectionKernelWithDeviceWeightEncoding(ctx, driver, normed, lw.ExpertGate[expert].Pointer(), lw.ExpertGate[expert].SizeBytes(), cfg.ExpertFF, cfg.HiddenSize, hipProjectionWeightEncodingF32)
	if err != nil {
		return nil, err
	}
	up, err := hipRunProjectionKernelWithDeviceWeightEncoding(ctx, driver, normed, lw.ExpertUp[expert].Pointer(), lw.ExpertUp[expert].SizeBytes(), cfg.ExpertFF, cfg.HiddenSize, hipProjectionWeightEncodingF32)
	if err != nil {
		return nil, err
	}
	gated, err := hipRunSwiGLUKernel(ctx, driver, hipSwiGLURequest{Gate: gate, Up: up})
	if err != nil {
		return nil, err
	}
	down, err := hipRunProjectionKernelWithDeviceWeightEncoding(ctx, driver, gated, lw.ExpertDown[expert].Pointer(), lw.ExpertDown[expert].SizeBytes(), cfg.HiddenSize, cfg.ExpertFF, hipProjectionWeightEncodingF32)
	if err != nil {
		return nil, err
	}
	return down, nil
}
