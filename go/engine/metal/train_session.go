// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import core "dappco.re/go"

// train_session.go begins the real-model side of native training: the ArchSession's normal forward
// discards every per-layer hidden (it only needs the last one to decode), but a backward pass needs
// the residual stream INTO each layer to recompute that layer's intermediates. ForwardCaptureHiddens
// runs a full-sequence forward and returns the saved residual stream — the activation-saving forward
// the chained full-stack backward consumes. It reuses the per-layer capture the cross-engine diff
// already wires into stepToken (captureLayerHiddens), so the captured hiddens are exactly the engine's
// real layer outputs, not a re-derivation. Single-goroutine (the ArchSession contract).

// ForwardCaptureHiddens forwards ids[0:T] over a FRESH session (it resets pos to 0, overwriting the
// cache, so a training loop can re-run it each step) and returns the residual stream:
//
//	embeds[t]        = the input embedding of token t        ([T] of dModel bf16) — layer 0's input
//	perLayerOut[l]   = layer l's output hidden for all tokens ([nLayers] of T·dModel bf16) — the
//	                   residual stream after layer l (and thus layer l+1's input)
//
// So layer l's INPUT is embeds (l==0) or perLayerOut[l-1] (l>0), and perLayerOut[nLayers-1] is the
// final hidden the head reads. The backward chains layer nLayers-1 → 0 over these. bf16 (the engine's
// forward precision); the f32 VJPs widen as needed.
func (s *ArchSession) ForwardCaptureHiddens(ids []int32) (embeds [][]byte, perLayerOut [][]byte, err error) {
	if len(ids) == 0 {
		return nil, nil, core.NewError("native.ForwardCaptureHiddens: empty ids")
	}
	T := len(ids)
	N := len(s.state.specs)
	if s.pos+T > s.maxLen {
		return nil, nil, core.NewError("native.ForwardCaptureHiddens: sequence exceeds maxLen")
	}
	if s.state.icb != nil {
		return s.forwardCaptureHiddensICB(ids, T, N)
	}

	prevFlag, prevCap := captureLayerHiddens, capturedLayerHiddens
	captureLayerHiddens = true
	capturedLayerHiddens = nil
	defer func() { captureLayerHiddens = prevFlag; capturedLayerHiddens = prevCap }()

	s.pos = 0 // forward the whole sequence from scratch (training re-prefills each step)
	embeds = make([][]byte, T)
	rowBytes := s.arch.Hidden * bf16Size
	var embedSlab []byte
	if s.canUseEmbedScratch() {
		embedSlab = make([]byte, T*rowBytes)
	}
	for t, id := range ids {
		var emb []byte
		var e error
		if embedSlab != nil {
			row := embedSlab[t*rowBytes : (t+1)*rowBytes]
			emb, e = s.embedInto(row, id)
			if e == nil && len(emb) != rowBytes {
				e = core.NewError("native.ForwardCaptureHiddens: embedInto returned wrong hidden size")
			}
		} else {
			emb, e = s.embed(id)
		}
		if e != nil {
			return nil, nil, e
		}
		embeds[t] = emb
		if _, e := s.StepWithID(id, emb); e != nil {
			return nil, nil, e
		}
	}
	// capturedLayerHiddens is token-major: entry [t*N + l] is token t's layer-l output (dModel bf16).
	// Re-pack into per-layer [T, dModel] (the shape the block backward wants).
	if len(capturedLayerHiddens) != T*N {
		return nil, nil, core.NewError("native.ForwardCaptureHiddens: capture count mismatch (per-layer capture not wired?)")
	}
	perLayerOut = make([][]byte, N)
	for l := 0; l < N; l++ {
		buf := make([]byte, T*rowBytes)
		for t := 0; t < T; t++ {
			copy(buf[t*rowBytes:(t+1)*rowBytes], capturedLayerHiddens[t*N+l])
		}
		perLayerOut[l] = buf
	}
	return embeds, perLayerOut, nil
}

func (s *ArchSession) forwardCaptureHiddensICB(ids []int32, T, N int) (embeds [][]byte, perLayerOut [][]byte, err error) {
	rowBytes := s.arch.Hidden * bf16Size
	s.pos = 0
	embeds = make([][]byte, T)
	perLayerOut = make([][]byte, N)
	for l := 0; l < N; l++ {
		perLayerOut[l] = make([]byte, T*rowBytes)
	}
	var embedSlab []byte
	if s.canUseEmbedScratch() {
		embedSlab = make([]byte, T*rowBytes)
	}
	for t, id := range ids {
		var emb []byte
		var e error
		if embedSlab != nil {
			row := embedSlab[t*rowBytes : (t+1)*rowBytes]
			emb, e = s.embedInto(row, id)
			if e == nil && len(emb) != rowBytes {
				e = core.NewError("native.ForwardCaptureHiddens: ICB embedInto returned wrong hidden size")
			}
		} else {
			emb, e = s.embed(id)
		}
		if e != nil {
			return nil, nil, e
		}
		embeds[t] = emb
		var pli []byte
		if s.perLayerInput != nil {
			pli, e = s.perLayerInput(id, emb)
			if e != nil {
				return nil, nil, e
			}
			s.state.perLayerInput = pli
		}
		var layers [][]byte
		withAutoreleasePool(func() {
			_, layers = s.state.icb.stepBodyCapture(emb, s.pos, pli)
		})
		if len(layers) != N {
			return nil, nil, core.NewError("native.ForwardCaptureHiddens: ICB capture count mismatch")
		}
		for l := 0; l < N; l++ {
			if len(layers[l]) != rowBytes {
				return nil, nil, core.NewError("native.ForwardCaptureHiddens: ICB capture row size mismatch")
			}
			copy(perLayerOut[l][t*rowBytes:(t+1)*rowBytes], layers[l])
		}
		s.pos++
	}
	return embeds, perLayerOut, nil
}
