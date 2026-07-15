// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// DiffusionLayerKVPrefix exposes the resident session prefix K/V rows in the
// token-major layout consumed by DiffusionDenoiseForwardBF16. Returned byte
// slices borrow the session's cache backing when the visible cache span is
// contiguous; callers must consume them before mutating the session cache.
func (s *ArchSession) DiffusionLayerKVPrefix() ([]DiffusionLayerKV, error) {
	const op = "native.ArchSession.DiffusionLayerKVPrefix"
	if s == nil {
		return nil, core.NewError(op + ": nil session")
	}
	if s.pos <= 0 {
		return nil, core.NewError(op + ": empty cache")
	}
	if s.pos > s.maxLen {
		return nil, core.NewError(op + ": position outside maxLen")
	}
	views, err := s.stateLayerViews()
	if err != nil {
		return nil, err
	}
	out := make([]DiffusionLayerKV, len(s.arch.Layer))
	for _, view := range views {
		if view.layer < 0 || view.layer >= len(out) {
			return nil, core.NewError(op + ": layer view outside arch")
		}
		start, tokenCount := 0, s.pos
		if view.maxSize > 0 && s.pos > view.cacheRows {
			start = s.pos - view.cacheRows
			tokenCount = view.cacheRows
		}
		keyRows, valueRows, err := stateBlockLayerBytes(view, start, tokenCount, s.pos)
		if err != nil {
			return nil, core.E(op, "layer prefix", err)
		}
		out[view.layer] = DiffusionLayerKV{
			K:           keyRows,
			V:           valueRows,
			PrefixStart: start,
			Position:    s.pos,
		}
	}
	for li, spec := range s.arch.Layer {
		if spec.OwnsCache() {
			continue
		}
		owner := spec.KVShareFrom
		if owner < 0 || owner >= len(out) {
			return nil, core.NewError(op + ": invalid shared K/V owner")
		}
		out[li] = out[owner]
	}
	return out, nil
}

func (m *NativeTokenModel) GenerateBlockDiffusionTokens(ctx context.Context, prompt []int32, opts BlockDiffusionOptions, yield func(int32) bool) (DiffusionMetrics, error) {
	const op = "native.NativeTokenModel.GenerateBlockDiffusionTokens"
	var metrics DiffusionMetrics
	if ctx == nil {
		ctx = context.Background()
	}
	if m == nil || m.NativeBackend == nil {
		return metrics, core.NewError(op + ": nil model")
	}
	if m.diffusion == nil {
		return metrics, core.NewError(op + ": model has no diffusion payload")
	}
	if m.bf16 == nil && m.quant == nil {
		return metrics, core.NewError(op + ": model weights are not available")
	}
	if len(prompt) == 0 {
		return metrics, core.NewError(op + ": empty prompt")
	}
	if opts.MaxTokens <= 0 {
		return metrics, core.NewError(op + ": MaxTokens must be > 0")
	}
	stepper, err := m.OpenSession()
	if err != nil {
		return metrics, err
	}
	if c, ok := stepper.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	sess, ok := stepper.(*ArchSession)
	if !ok {
		return metrics, core.NewError(op + ": OpenSession did not return an ArchSession")
	}

	canvasLen := m.diffusion.CanvasLength
	if canvasLen <= 0 {
		canvasLen = DefaultCanvasLength
	}
	if int(canvasLen) > opts.MaxTokens {
		canvasLen = int32(opts.MaxTokens)
	}
	maxCanvases := (opts.MaxTokens + int(canvasLen) - 1) / int(canvasLen)
	stepCfg := DefaultDiffusionStepConfig(int32(m.vocab))
	if opts.SeedSet {
		stepCfg.Seed = opts.Seed
	}
	if opts.Temperature > 0 {
		stepCfg.MinTemperature = opts.Temperature
		stepCfg.MaxTemperature = opts.Temperature
	}
	stopTokens := append([]int32(nil), opts.StopTokens...)
	if len(stopTokens) == 0 {
		stopTokens = append(stopTokens, m.diffusion.EOSTokens...)
	}
	emitted := 0
	cfg := DiffusionGenerateConfig{
		Step:         stepCfg,
		CanvasLength: canvasLen,
		MaxCanvases:  maxCanvases,
		StopTokens:   stopTokens,
	}
	_, metrics, err = RunDiffusionGenerate(ctx, cfg, m.diffusion.EOSTokens, int32(m.vocab), m.arch.SlidingWindow, DiffusionGenerateOps{
		Prefill: func(context.Context) (int, error) {
			if err := sess.PrefillTokens(prompt); err != nil {
				return 0, err
			}
			return sess.Pos(), nil
		},
		CacheOffset: sess.Pos,
		Denoise: func(_ context.Context, req DiffusionDenoiseRequest) (DiffusionStepResult, error) {
			prefixKV, err := sess.DiffusionLayerKVPrefix()
			if err != nil {
				return DiffusionStepResult{}, err
			}
			globalMask, localMask, err := diffusionSessionDenoiseMasks(m.arch, prefixKV, req)
			if err != nil {
				return DiffusionStepResult{}, err
			}
			if m.quant != nil {
				logits, err := DiffusionDenoiseForwardQuant(m.quant, m.diffusion, m.arch, req.Canvas, req.SCEmb, prefixKV, globalMask, localMask)
				if err != nil {
					return DiffusionStepResult{}, err
				}
				head := m.quant.LMHead
				scales := m.quant.LMHeadScales
				biases := m.quant.LMHeadBiases
				if len(head) == 0 {
					head = m.quant.Embed
					scales = m.quant.EmbedScales
					biases = m.quant.EmbedBiases
				}
				headWeight := QuantWeight{Packed: head, Scales: scales, Biases: biases, GroupSize: m.quant.GroupSize, Bits: m.quant.Bits}
				groupSize, bits := quantWeightGeometryForShape(headWeight, m.vocab, m.arch.Hidden, m.quant.GroupSize, m.quant.Bits)
				return DiffusionSampleDenoiseStepQuant(logits, head, scales, biases, req.Canvas, m.vocab, m.arch.Hidden, groupSize, bits, req.Step, req.NoiseProportion, req.StepConfig)
			}
			logits, err := DiffusionDenoiseForwardBF16(m.bf16, m.diffusion, m.arch, req.Canvas, req.SCEmb, prefixKV, globalMask, localMask)
			if err != nil {
				return DiffusionStepResult{}, err
			}
			return DiffusionSampleDenoiseStepBF16(logits, m.bf16.Embed, req.Canvas, m.vocab, m.arch.Hidden, req.Step, req.NoiseProportion, req.StepConfig)
		},
		TruncateTo: func(pos int) error {
			if !sess.TruncateTo(pos) {
				return core.NewError(op + ": session refused truncation")
			}
			return nil
		},
		Commit: func(_ context.Context, kept []int32) error {
			if len(kept) == 0 {
				return nil
			}
			remaining := opts.MaxTokens - emitted
			if remaining <= 0 {
				return nil
			}
			if len(kept) > remaining {
				kept = kept[:remaining]
			}
			if err := sess.AppendTokens(kept); err != nil {
				return err
			}
			for _, id := range kept {
				emitted++
				if yield != nil && !yield(id) {
					return core.NewError(op + ": yield stopped")
				}
			}
			return nil
		},
	})
	metrics.EmittedTokens = emitted
	return metrics, err
}

func diffusionSessionDenoiseMasks(arch model.Arch, layerKV []DiffusionLayerKV, req DiffusionDenoiseRequest) ([]float32, []float32, error) {
	canvasLen := len(req.Canvas)
	if canvasLen <= 0 {
		return nil, nil, core.NewError("native.NativeTokenModel.GenerateBlockDiffusionTokens: empty canvas")
	}
	globalPrefix, err := diffusionPrefixLenForAttention(arch, layerKV, model.GlobalAttention, req.Prefix)
	if err != nil {
		return nil, nil, err
	}
	localPrefix, err := diffusionPrefixLenForAttention(arch, layerKV, model.SlidingAttention, req.Prefix)
	if err != nil {
		return nil, nil, err
	}
	globalKeyLen := globalPrefix + canvasLen
	localKeyLen := localPrefix + canvasLen
	globalMask := req.GlobalMask
	if len(globalMask) != canvasLen*globalKeyLen {
		globalMask, _ = diffusionGlobalCanvasMaskData(1, canvasLen, globalKeyLen)
	}
	localMask := req.LocalMask
	if len(localMask) != canvasLen*localKeyLen {
		localMask, _ = diffusionBlockLocalCanvasMaskData(1, canvasLen, localKeyLen, localPrefix, arch.SlidingWindow)
	}
	return globalMask, localMask, nil
}

func diffusionPrefixLenForAttention(arch model.Arch, layerKV []DiffusionLayerKV, attention model.AttentionType, fallback int) (int, error) {
	for _, spec := range arch.Layer {
		if spec.Attention != attention {
			continue
		}
		owner := spec.KVShareFrom
		if owner < 0 || owner >= len(arch.Layer) || owner >= len(layerKV) {
			return 0, core.NewError("native.NativeTokenModel.GenerateBlockDiffusionTokens: invalid K/V owner")
		}
		ownerSpec := arch.Layer[owner]
		kvDim := kvHeadsOf(ownerSpec, arch.KVHeads) * headDimOf(ownerSpec, arch.HeadDim)
		prefixLen, _, _, err := diffusionLayerKVGeometry(layerKV[owner], kvDim)
		if err != nil {
			return 0, err
		}
		return prefixLen, nil
	}
	if fallback < 0 {
		fallback = 0
	}
	return fallback, nil
}
