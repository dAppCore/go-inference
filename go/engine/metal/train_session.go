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

// KNOWN DIVERGENCE (2026-07-14, probe_train_forward_split_test.go): on a real PLE arch
// (E2B bf16) this serial capture's hiddens DISAGREE with the engine's serving forward —
// against the batched prefill's boundary hidden the serial ICB capture was off by |Δ|≈34
// while ForwardCaptureFinalHidden matched byte-for-byte. Until that is root-caused, the
// per-layer hiddens below are NOT serving-exact on PLE archs; the head-LoRA trainer rides
// ForwardCaptureFinalHidden instead, and the full-stack backward must re-verify this
// capture against the batched route before trusting per-layer hiddens on a PLE model.
//
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
	for l := range N {
		buf := make([]byte, T*rowBytes)
		for t := range T {
			copy(buf[t*rowBytes:(t+1)*rowBytes], capturedLayerHiddens[t*N+l])
		}
		perLayerOut[l] = buf
	}
	return embeds, perLayerOut, nil
}

// captureFinalHiddenBatchedChunksForTest counts batched-route chunks the capture forward
// engaged — the engagement receipt for its parity test (a fallback that silently went
// serial would still be byte-identical, so identity alone can't prove the fast path ran).
var captureFinalHiddenBatchedChunksForTest int

// ForwardCaptureFinalHidden forwards ids[0:T] over a FRESH session (pos reset to 0, the
// cache overwritten — the training re-prefill, exactly ForwardCaptureHiddens' contract)
// and returns the FINAL residual hidden of every token ([T·dModel] bf16 — layer
// nLayers-1's output, the rows the head reads). The HEAD-LoRA trainer needs only these
// rows, so this forward rides the engine's batched-dense prefill route (weight-read-once
// GEMMs; the wall-split probe measured the serial walk 17.8× slower at T=128 on E2B
// bf16) with per-row result scatter, chunked under the same policy as the retained
// prefill but with the shared-suffix layer skip OFF — a skipped chunk's late layers
// would fabricate the very rows this capture returns. Any decline falls back to the
// serial ForwardCaptureHiddens walk; both routes are byte-identical (the #381
// prefill-parity spine). The full per-layer capture (ForwardCaptureHiddens) remains the
// full-stack backward's forward.
func (s *ArchSession) ForwardCaptureFinalHidden(ids []int32) ([]byte, error) {
	if len(ids) == 0 {
		return nil, core.NewError("native.ForwardCaptureFinalHidden: empty ids")
	}
	T := len(ids)
	if T > s.maxLen {
		return nil, core.NewError("native.ForwardCaptureFinalHidden: sequence exceeds maxLen")
	}
	rowBytes := s.arch.Hidden * bf16Size
	out := make([]byte, T*rowBytes)
	s.pos = 0
	done := 0
	for done < T {
		n := s.batchedDensePrefillChunkLen(T - done)
		if n <= 0 || n > T-done {
			n = T - done
		}
		chunk := ids[done : done+n]
		dstRows := make([][]byte, n)
		for i := range dstRows {
			dstRows[i] = out[(done+i)*rowBytes : (done+i+1)*rowBytes]
		}
		ok, err := s.captureFinalHiddenChunkBatched(chunk, dstRows)
		if err != nil {
			return nil, err
		}
		if !ok {
			// The batched body declined (session shape, kernel availability, test lever).
			// Cold path: re-run the WHOLE sequence on the proven serial capture — pos and
			// the cache are reset by ForwardCaptureHiddens itself, so partial batched
			// progress is simply discarded rather than spliced.
			_, perLayer, serr := s.ForwardCaptureHiddens(ids)
			if serr != nil {
				return nil, serr
			}
			if len(perLayer) == 0 {
				return nil, core.NewError("native.ForwardCaptureFinalHidden: serial capture returned no layers")
			}
			last := perLayer[len(perLayer)-1]
			if len(last) != len(out) {
				return nil, core.NewError("native.ForwardCaptureFinalHidden: serial capture size mismatch")
			}
			copy(out, last)
			return out, nil
		}
		captureFinalHiddenBatchedChunksForTest++
		done += n
	}
	return out, nil
}

// captureFinalHiddenChunkBatched runs ONE capture chunk through the batched-dense body
// with per-row result scatter. It mirrors prefillRetainedTokensBatchedDenseOne's input
// plumbing (device-first inputs, embed scratch, the PLE slab) — kept in lockstep with it;
// only the result shape differs (all rows scattered to dstRows vs last-row-into). ok=false
// is the body's decline, never an error: the caller owns the serial fallback.
func (s *ArchSession) captureFinalHiddenChunkBatched(ids []int32, dstRows [][]byte) (bool, error) {
	if s.state.icb != nil && (len(ids) <= batchedDenseICBMaxRows || batchedMLPFoldDisabledForTest || !gpuHasGeluKernel()) {
		return false, nil
	}
	embBuf, pleBuf, devErr := s.prefillInputsDevice(ids)
	if devErr != nil {
		return false, devErr
	}
	embs := make([][]byte, len(ids))
	var pleSlab []byte
	if embBuf == nil {
		if s.canUseEmbedScratch() {
			rowBytes := s.arch.Hidden * bf16Size
			need := len(ids) * rowBytes
			if cap(s.embedScratch) < need {
				s.embedScratch = make([]byte, need)
			} else {
				s.embedScratch = s.embedScratch[:need]
			}
			for i, id := range ids {
				dst := s.embedScratch[i*rowBytes : (i+1)*rowBytes]
				emb, err := s.embedInto(dst, id)
				if err != nil {
					return false, err
				}
				if len(emb) != rowBytes {
					return false, core.NewError("native.ForwardCaptureFinalHidden: embedInto returned wrong hidden size")
				}
				embs[i] = emb
			}
		} else {
			for i, id := range ids {
				emb, err := s.embed(id)
				if err != nil {
					return false, err
				}
				embs[i] = emb
			}
		}
		var slabErr error
		pleSlab, slabErr = s.pleSlabFor(ids, embs)
		if slabErr != nil {
			return false, slabErr
		}
	}
	var (
		ok  bool
		err error
	)
	withAutoreleasePool(func() {
		if pleBuf != nil {
			s.state.prefillPLESlabDevice = pleBuf
			s.state.prefillEmbedDevice = embBuf
			defer func() {
				s.state.prefillPLESlabDevice = nil
				s.state.prefillEmbedDevice = nil
			}()
			_, ok, err = s.state.stepTokensBatchedDenseIntoPLE(embs, nil, s.pos, dstRows)
		} else if embBuf != nil {
			s.state.prefillEmbedDevice = embBuf
			defer func() { s.state.prefillEmbedDevice = nil }()
			_, ok, err = s.state.stepTokensBatchedDenseInto(embs, s.pos, dstRows)
		} else if pleSlab != nil {
			_, ok, err = s.state.stepTokensBatchedDenseIntoPLE(embs, pleSlab, s.pos, dstRows)
		} else {
			_, ok, err = s.state.stepTokensBatchedDenseInto(embs, s.pos, dstRows)
		}
	})
	if err != nil || !ok {
		return ok, err
	}
	s.pos += len(ids)
	return true, nil
}

func (s *ArchSession) forwardCaptureHiddensICB(ids []int32, T, N int) (embeds [][]byte, perLayerOut [][]byte, err error) {
	rowBytes := s.arch.Hidden * bf16Size
	s.pos = 0
	embeds = make([][]byte, T)
	perLayerOut = make([][]byte, N)
	for l := range N {
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
		for l := range N {
			if len(layers[l]) != rowBytes {
				return nil, nil, core.NewError("native.ForwardCaptureHiddens: ICB capture row size mismatch")
			}
			copy(perLayerOut[l][t*rowBytes:(t+1)*rowBytes], layers[l])
		}
		s.pos++
	}
	return embeds, perLayerOut, nil
}
