// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/decode/dflash"
)

// assistant_dflash_proposer.go bridges the engine's block-parallel draft forward to
// the model-free, provably-lossless verify driver in decode/dflash. DFlashDrafter
// produces proposals; decode/dflash.Generate/AcceptBlock verifies them against the
// target and commits only the target's own tokens — so wiring the engine forward
// behind the dflash.BlockProposer seam yields DFlash speculative decode whose output
// is byte-identical to plain decode (the losslessness the driver's own fuzz proves).
// This is the seam serve arms once the engine declares DFlash support.

// DFlashAuxSource supplies a block forward's inputs for the current context: the
// verifier hidden states at the drafter's aux layers (numAux entries, backbone-sized
// bf16), the target embedding of the anchor (last committed) token, and the anchor's
// position. Splitting this out keeps DFlashDrafter engine-pure (it takes hiddens, not
// a session) while letting the live lane feed it from a running target forward.
type DFlashAuxSource func(context []int) (auxHiddens [][]byte, anchorEmbedding []byte, anchorPos int, ok bool)

// dflashProposer adapts a DFlashDrafter + an aux source to decode/dflash.BlockProposer.
type dflashProposer struct {
	drafter *DFlashDrafter
	source  DFlashAuxSource
}

// NewDFlashProposer wires a loaded DFlashDrafter and an aux source into a
// decode/dflash.BlockProposer. Feed the result to dflash.Generate (the verify
// driver) with the target's greedy next-token oracle for lossless speculative decode.
func NewDFlashProposer(drafter *DFlashDrafter, source DFlashAuxSource) dflash.BlockProposer {
	return &dflashProposer{drafter: drafter, source: source}
}

// ProposeBlock runs the engine forward for the current context and returns its block
// of target-vocab candidates. An unavailable source (or a forward error) returns an
// empty block, which the driver treats exactly as a drafter miss — the target then
// decodes one token itself, still lossless.
func (p *dflashProposer) ProposeBlock(context []int) []int {
	if p == nil || p.drafter == nil || p.source == nil {
		return nil
	}
	aux, anchorEmbedding, anchorPos, ok := p.source(context)
	if !ok {
		return nil
	}
	block, err := p.drafter.ProposeBlock(aux, anchorEmbedding, anchorPos)
	if err != nil {
		return nil
	}
	out := make([]int, len(block))
	for i, tok := range block {
		out[i] = int(tok)
	}
	return out
}

// ExtractAuxHiddens taps the verifier's hidden states at the drafter's aux layers for
// the LAST token of ids — the fused-context input DFlash conditions on. It reuses the
// engine's real per-layer capture (ForwardCaptureHiddens / captureLayerHiddens), so
// the hiddens are the engine's actual layer outputs, not a re-derivation.
//
// EVIDENCED GAP (docs/design-dflash.md): ForwardCaptureHiddens resets the session to
// pos 0 and re-runs the whole sequence, OVERWRITING the KV cache — fine for a fresh /
// throwaway session (verification-time extraction, tests), but it cannot tap a LIVE
// incrementally-decoding serving session without corrupting its cache. A cheap,
// non-corrupting boundary tap (capturing the aux-layer hiddens during the ordinary
// decode step, the way retainedHidden already captures the final one) is the one
// engine seam DFlash still needs before the live HTTP lane can extract cheaply; the
// forward itself is complete and takes the hiddens as input.
func ExtractAuxHiddens(target *ArchSession, ids []int32, auxLayers []int) ([][]byte, error) {
	if target == nil {
		return nil, core.NewError("native.dflash: aux extraction target session is nil")
	}
	if len(ids) == 0 {
		return nil, core.NewError("native.dflash: aux extraction needs a non-empty prefix")
	}
	_, perLayerOut, err := target.ForwardCaptureHiddens(ids)
	if err != nil {
		return nil, core.E("native.dflash", "capture verifier hiddens", err)
	}
	rowBytes := target.arch.Hidden * bf16Size
	last := len(ids) - 1
	out := make([][]byte, len(auxLayers))
	for i, layer := range auxLayers {
		if layer < 0 || layer >= len(perLayerOut) {
			return nil, core.NewError(core.Sprintf("native.dflash: aux layer %d out of range [0,%d)", layer, len(perLayerOut)))
		}
		row := perLayerOut[layer]
		if len(row) < (last+1)*rowBytes {
			return nil, core.NewError("native.dflash: captured hidden row is short")
		}
		out[i] = append([]byte(nil), row[last*rowBytes:(last+1)*rowBytes]...)
	}
	return out, nil
}
