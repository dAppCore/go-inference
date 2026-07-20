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
// This is the THROWAWAY-session extractor: ForwardCaptureHiddens resets the session to
// pos 0 and re-runs the whole sequence, OVERWRITING the KV cache — fine for a fresh /
// throwaway session (verification-time extraction, tests), but it cannot tap a LIVE
// incrementally-decoding serving session without corrupting its cache. The
// non-corrupting boundary tap for a live session is ExtractAuxHiddensLive
// (assistant_dflash_livetap.go), which captures the aux-layer hiddens for the current
// boundary token without resetting pos or perturbing the running cache. The block
// forward itself takes the hiddens as input, so it is agnostic to which extractor fed
// it.
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

// ExtractAuxHiddensAllRaw taps the verifier's hidden states at auxLayers for
// EVERY position of ids (not just the last, unlike ExtractAuxHiddens) and
// returns them as ONE f32 array [len(ids), len(auxLayers)*hidden] — each row
// the concatenation, in auxLayers order, of that position's hidden at every
// aux layer (extract_context_feature's torch.cat, the real z-lab convention —
// docs/design-dflash-forward.md §2-§3's targetHiddenRaw, the shape
// DFlashZLabForward and zLabDFlashProposer (assistant_dflash_zlab.go)
// consume).
//
// Reuses the SAME throwaway full-replay primitive ExtractAuxHiddens does
// (ForwardCaptureHiddens: resets pos to 0, re-runs the whole sequence — safe
// for a fresh/throwaway extraction, NOT for a live incrementally-decoding
// session), matching the existing DFlash verify oracle's own full-prefix-
// replay posture (speculativeModel.generateDFlash's/generateDFlashZLab's
// next(), which already re-PrefillTokens the whole prefix every call) rather
// than mixing replay tiers within one round. Moving both the aux tap and the
// verify oracle to the incremental live-session primitives together
// (ExtractAuxHiddensLive, assistant_dflash_livetap.go) is the named perf
// follow-up (docs/design-dflash-forward.md §7 item 5 — "skip re-fusing
// unchanged context rows across rounds"), not required for this lane's
// correctness gate.
func ExtractAuxHiddensAllRaw(target *ArchSession, ids []int32, auxLayers []int) ([]float32, error) {
	if target == nil {
		return nil, core.NewError("native.dflash: aux extraction target session is nil")
	}
	if len(ids) == 0 {
		return nil, core.NewError("native.dflash: aux extraction needs a non-empty prefix")
	}
	if len(auxLayers) == 0 {
		return nil, core.NewError("native.dflash: aux extraction requested no aux layers")
	}
	_, perLayerOut, err := target.ForwardCaptureHiddens(ids)
	if err != nil {
		return nil, core.E("native.dflash", "capture verifier hiddens", err)
	}
	hidden := target.arch.Hidden
	rowBytes := hidden * bf16Size
	ctxLen := len(ids)
	numAux := len(auxLayers)
	out := make([]float32, ctxLen*numAux*hidden)
	for i, layer := range auxLayers {
		if layer < 0 || layer >= len(perLayerOut) {
			return nil, core.NewError(core.Sprintf("native.dflash: aux layer %d out of range [0,%d)", layer, len(perLayerOut)))
		}
		row := perLayerOut[layer]
		if len(row) < ctxLen*rowBytes {
			return nil, core.NewError("native.dflash: captured hidden row is short")
		}
		for t := 0; t < ctxLen; t++ {
			dstOff := (t*numAux + i) * hidden
			copy(out[dstOff:dstOff+hidden], bf16ToF32Slice(row[t*rowBytes:(t+1)*rowBytes]))
		}
	}
	return out, nil
}
