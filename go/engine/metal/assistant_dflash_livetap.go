// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import core "dappco.re/go"

// assistant_dflash_livetap.go closes the one evidenced engine gap the DFlash live
// lane still needed: a cheap, NON-CORRUPTING boundary tap for the verifier's
// aux-layer hidden states.
//
// ExtractAuxHiddens (assistant_dflash_proposer.go) reads those hiddens through
// ForwardCaptureHiddens, which resets the session to pos 0 and re-runs the WHOLE
// sequence — fine for a throwaway extraction (tests, verification-time), but it
// overwrites the KV cache, so it cannot tap a LIVE incrementally-decoding serving
// session without corrupting it. ExtractAuxHiddensLive taps the SAME hiddens for
// the session's CURRENT boundary token without resetting pos and without
// perturbing the running cache — the seam a live serving loop extracts through.
//
// It works by re-running ONLY the boundary token's forward at its OWN position
// (anchorPos = pos-1) with the engine's existing per-layer hidden capture armed.
// The re-run reads the live KV cache rows [0, anchorPos] for attention exactly as
// the original decode of that token did, and rewrites the boundary token's own
// cache row idempotently (same token, same position, same context ⇒ byte-identical
// K/V), so pos is unchanged and the next real decode step is unperturbed. Both
// forward routes surface the same layer outputs, so the tap is representation-
// agnostic: the recorded-ICB serving path replays through stepBodyCapture (which
// already carves the recorded per-layer boundaries — the #391 fix), and the
// re-encode path drives stepToken under captureLayerHiddens (the cross-engine diff
// seam). Single-goroutine, borrowing the package capture buffers exactly as
// ForwardCaptureHiddens does (the ArchSession contract).

// ExtractAuxHiddensLive returns the verifier's hidden states at auxLayers for the
// session's current boundary token (the last committed token, at pos-1), taken
// from a non-corrupting re-run of that token's forward. anchorID is the boundary
// token's id (the caller holds it — the live source's context[-1]); it must be the
// token the cache boundary already holds, so the re-run reproduces the identical
// hidden. Each returned row is one aux layer's output, backbone-sized bf16 (the
// target hidden == backbone), in auxLayers order — the fused-context input
// DFlashDrafter.ProposeBlock consumes.
//
//	aux, err := target.ExtractAuxHiddensLive(context[len(context)-1], drafter.AuxLayers())
func (s *ArchSession) ExtractAuxHiddensLive(anchorID int32, auxLayers []int) ([][]byte, error) {
	if s == nil {
		return nil, core.NewError("native.dflash: live aux tap session is nil")
	}
	if s.pos <= 0 {
		return nil, core.NewError("native.dflash: live aux tap needs a decoded boundary token (pos == 0)")
	}
	nLayers := len(s.state.specs)
	if len(auxLayers) == 0 {
		return nil, core.NewError("native.dflash: live aux tap requested no aux layers")
	}
	anchorPos := s.pos - 1
	emb, err := s.embedID(anchorID)
	if err != nil {
		return nil, core.E("native.dflash", "embed boundary token", err)
	}
	// embedID hands back the shared embed scratch; the forward re-run below reuses
	// it, so pin the anchor embedding into its own buffer first.
	anchorEmb := append([]byte(nil), emb...)

	perLayer, err := s.captureBoundaryLayerHiddens(anchorID, anchorEmb, anchorPos)
	if err != nil {
		return nil, err
	}
	if len(perLayer) != nLayers {
		return nil, core.NewError(core.Sprintf("native.dflash: live aux tap captured %d layers, want %d", len(perLayer), nLayers))
	}

	rowBytes := s.arch.Hidden * bf16Size
	out := make([][]byte, len(auxLayers))
	for i, layer := range auxLayers {
		if layer < 0 || layer >= nLayers {
			return nil, core.NewError(core.Sprintf("native.dflash: aux layer %d out of range [0,%d)", layer, nLayers))
		}
		if len(perLayer[layer]) != rowBytes {
			return nil, core.NewError(core.Sprintf("native.dflash: aux layer %d hidden is %d bytes, want %d (backbone)", layer, len(perLayer[layer]), rowBytes))
		}
		out[i] = append([]byte(nil), perLayer[layer]...)
	}
	return out, nil
}

// captureBoundaryLayerHiddens re-runs the boundary token's forward at anchorPos
// with per-layer capture armed and returns every layer's output hidden (one row
// each, in layer order). It routes to the same forward the session decodes on: the
// recorded-ICB replay's stepBodyCapture, or the re-encode stepToken under the
// package captureLayerHiddens flag. Both are idempotent at anchorPos, so the tap
// leaves the running cache and s.pos untouched.
func (s *ArchSession) captureBoundaryLayerHiddens(anchorID int32, anchorEmb []byte, anchorPos int) ([][]byte, error) {
	var pli []byte
	if s.perLayerInput != nil { // PLE models: per-layer inputs from this token's id + embedding
		var err error
		if pli, err = s.perLayerInput(anchorID, anchorEmb); err != nil {
			return nil, core.E("native.dflash", "per-layer input", err)
		}
	}

	if s.state.icb != nil && !icbDisabledForTest {
		var perLayer [][]byte
		withAutoreleasePool(func() {
			_, perLayer = s.state.icb.stepBodyCapture(anchorEmb, anchorPos, pli)
		})
		if len(perLayer) == 0 {
			return nil, core.NewError("native.dflash: ICB live aux tap captured no layers (recorded boundaries missing?)")
		}
		return perLayer, nil
	}

	// Re-encode path: drive stepToken with the package per-layer capture armed
	// (the cross-engine-diff seam ForwardCaptureHiddens reuses too).
	prevFlag, prevCap := captureLayerHiddens, capturedLayerHiddens
	captureLayerHiddens = true
	capturedLayerHiddens = nil
	defer func() { captureLayerHiddens = prevFlag; capturedLayerHiddens = prevCap }()
	if pli != nil {
		s.state.perLayerInput = pli
	}
	var stepErr error
	withAutoreleasePool(func() {
		_, stepErr = s.state.stepToken(anchorEmb, anchorPos)
	})
	if stepErr != nil {
		return nil, core.E("native.dflash", "capture boundary forward", stepErr)
	}
	return capturedLayerHiddens, nil
}
