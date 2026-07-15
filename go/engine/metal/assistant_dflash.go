// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/mtp"
)

// assistant_dflash.go is the block-parallel twin of the autoregressive MTP draft
// step (assistant_draft_fused.go / draftBlockFromSessionWithSuppress). Where the
// MTP drafter conditions on the target's SINGLE final boundary hidden and drafts
// one token per forward — feeding each pick back in — a DFlash drafter (arXiv
// 2602.06036, "Block Diffusion for Flash Speculative Decoding") conditions on
// FUSED hidden states drawn from several verifier layers and proposes a whole
// BLOCK of tokens in one non-autoregressive readout. The target then verifies the
// block with the ordinary greedy prefix-accept (decode/dflash.AcceptBlock), so
// the emitted sequence is byte-identical to plain decode WHATEVER this forward
// proposes — the drafter changes only how fast the target's own tokens are
// produced, never which. That losslessness (proven, model-free, in decode/dflash)
// is why this forward is safe to arm even though no public gemma-4 DFlash
// checkpoint exists to measure accept-length against: a weak proposal costs a
// verify pass, never a wrong token.
//
// The forward is expressed against the seams the MTP path already ships, so it is
// a fill-in rather than a redesign (docs/design-dflash.md names them):
//
//   - FUSED CONTEXT (1a): the verifier's hidden states at aux_hidden_state_layer_ids
//     are concatenated and projected (dflash.aux_projection) into one backbone-space
//     context feature; the anchor embedding ⊕ that feature seeds the drafter through
//     the existing pre_projection (DraftInputProjectionInto). A cheap non-corrupting
//     boundary tap that extracts those verifier hiddens for a LIVE decode session is
//     the one evidenced gap (see assistant_dflash_proposer.go) — this forward takes
//     the hiddens as input, so parity + losslessness are provable now.
//   - KV INJECTION (1b): each verifier hidden is projected through the draft layer's
//     own k_proj / v_proj into one injected K/V row, so every draft layer cross-attends
//     a numAux-row target-context memory — the exact AssistantTargetKV seam the MTP
//     draftLayer already consumes (there the rows are the target's live K/V; here they
//     are the fused verifier context). The whole gemma4 sandwich-norm layer is reused
//     verbatim (draftLayerIntoScratch), so only the injection is new maths.
//   - BLOCK-PARALLEL READOUT (1c): the block's positions are decoded in one sweep, each
//     against the SAME shared injected context, differentiated by its rope position
//     (anchorPos+j) — no position depends on another's PREDICTION, so it is genuinely
//     non-autoregressive. Joint intra-block self-attention (positions attending each
//     other's hidden) is the documented refinement; the shared-context readout is the
//     honest first forward and is all the losslessness invariant needs.
//   - REDUCED VOCAB (1d): the draft LM head (dflash.lm_head) is a smaller vocab; d2t
//     maps each drafted id back to the target vocab the verifier accepts against.
//
// Every op is a GPU dispatch through the package's parity-gated bf16 primitives
// (MatVecBF16Into, RMSNormBF16Into via draftLayerIntoScratch, the SDPA inside
// draftAttentionIntoScratch); the forward is validated by a pure-float host
// reference of the same maths (assistant_dflash_test.go), the r2-r5 fixture
// pattern. No cgo — pure Go + Metal dispatch, like the rest of this engine.

// DFlashDrafter is a loaded DFlash block-diffusion drafter: the neutral
// AssistantModel (its decoder stack, loaded through the ordinary reactive pack
// loader) plus the DFlash-specific parameters — the block it proposes per readout,
// how many verifier layers fuse into its context, and the reduced-vocab d2t map.
// One instance per attached drafter; ProposeBlock is single-goroutine (it borrows
// the shared draft-layer scratch), matching the ArchSession contract.
type DFlashDrafter struct {
	m          *AssistantModel
	blockSize  int
	numAux     int     // len(aux_hidden_state_layer_ids) — verifier hiddens fused per block
	auxLayers  []int   // the verifier layer ids the fused context is drawn from
	draftVocab int     // the drafter's reduced vocab (dflash.lm_head rows)
	d2t        []int32 // draft-vocab id → target-vocab id (identity when the pack omits it)
	scratch    assistantDraftLayerScratch
}

// BlockSize is the number of candidate tokens ProposeBlock returns per readout (γ,
// the diffusion block). NumAux is the count of verifier hiddens ProposeBlock
// expects — the fused-context width.
func (d *DFlashDrafter) BlockSize() int { return d.blockSize }
func (d *DFlashDrafter) NumAux() int    { return d.numAux }

// AuxLayers returns the verifier layer ids whose hidden states fuse into the
// drafter's context — the taps a caller extracts from the target forward.
func (d *DFlashDrafter) AuxLayers() []int { return append([]int(nil), d.auxLayers...) }

// ProposeBlock runs the block-parallel draft forward and returns up to BlockSize
// candidate TARGET-vocab token ids. auxHiddens are the verifier's hidden states at
// AuxLayers (numAux entries, each backbone-sized bf16 — the fused-context input);
// anchorEmbedding is the target embedding of the last committed token; anchorPos is
// that token's position (drives the block's rope offsets). The returned ids are the
// drafter's greedy argmax per block position, mapped through d2t into the target
// vocab the verifier accepts against — a proposal, never a commitment; the verify
// driver keeps only the prefix the target agrees with.
func (d *DFlashDrafter) ProposeBlock(auxHiddens [][]byte, anchorEmbedding []byte, anchorPos int) ([]int32, error) {
	if d == nil || d.m == nil {
		return nil, core.NewError("native.dflash: drafter is not loaded")
	}
	m := d.m
	backboneBytes := m.BackboneHiddenSize * bf16Size
	if len(auxHiddens) != d.numAux {
		return nil, core.NewError(core.Sprintf("native.dflash: got %d verifier hiddens, want %d (aux_hidden_state_layer_ids)", len(auxHiddens), d.numAux))
	}
	for i, h := range auxHiddens {
		if len(h) != backboneBytes {
			return nil, core.NewError(core.Sprintf("native.dflash: verifier hidden %d is %d bytes, want %d (backbone)", i, len(h), backboneBytes))
		}
	}
	if len(anchorEmbedding) != backboneBytes {
		return nil, core.NewError(core.Sprintf("native.dflash: anchor embedding is %d bytes, want %d (backbone)", len(anchorEmbedding), backboneBytes))
	}

	// (1a) fuse the verifier hiddens into one backbone-space context feature and
	// seed the drafter's hidden through the existing pre_projection.
	auxContext, err := d.fuseAuxContext(auxHiddens)
	if err != nil {
		return nil, err
	}
	seed, err := m.DraftInputProjectionInto(nil, anchorEmbedding, auxContext)
	if err != nil {
		return nil, core.E("native.dflash", "seed projection", err)
	}

	// (1b) project each verifier hidden into every draft layer's injected K/V memory.
	injected, err := d.injectedKV(auxHiddens)
	if err != nil {
		return nil, err
	}

	// (1c) decode the block's positions in one sweep — shared injected context,
	// differentiated by rope position, no cross-position prediction feedback.
	block := make([]int32, 0, d.blockSize)
	hiddenBytes := m.Arch.Hidden * bf16Size
	for j := 0; j < d.blockSize; j++ {
		h := append([]byte(nil), seed...) // per-position residual stream (seed is shared)
		for li := range m.Arch.Layer {
			kv := injected[li]
			// The rope offset is carried in the injected KV window so the reused
			// draftAttention ropes this position's query at anchorPos+j (qPos =
			// Offset+Length-1); the positionless context rows stay unroped.
			kv.Offset = anchorPos + j - (d.numAux - 1)
			next, lerr := m.draftLayerIntoScratch(d.scratch.bytes(assistantDraftScratchLayerOut, hiddenBytes), li, h, kv, &d.scratch)
			if lerr != nil {
				return nil, core.E("native.dflash", core.Sprintf("draft layer %d", li), lerr)
			}
			h = append([]byte(nil), next...)
		}
		normed, ferr := m.DraftFinalNormInto(nil, h)
		if ferr != nil {
			return nil, core.E("native.dflash", "final norm", ferr)
		}
		// (1d) reduced-vocab head → argmax → d2t into the target vocab.
		tok, herr := d.headArgmax(normed)
		if herr != nil {
			return nil, herr
		}
		block = append(block, tok)
	}
	return block, nil
}

// fuseAuxContext concatenates the verifier hiddens and projects them through
// dflash.aux_projection.weight ([backbone, numAux*backbone]) into one backbone-space
// context feature — DFlash's "fused target context". Pooled: one MatVec dispatch.
func (d *DFlashDrafter) fuseAuxContext(auxHiddens [][]byte) ([]byte, error) {
	backbone := d.m.BackboneHiddenSize
	concat := make([]byte, 0, d.numAux*backbone*bf16Size)
	for _, h := range auxHiddens {
		concat = append(concat, h...)
	}
	proj, err := nativeAssistantBF16Matrix(d.m, dflashAuxProjectionWeight, backbone, d.numAux*backbone)
	if err != nil {
		return nil, core.E("native.dflash", "aux_projection", err)
	}
	out, err := MatVecBF16Into(nil, proj.Data, concat, backbone, d.numAux*backbone)
	if err != nil {
		return nil, core.E("native.dflash", "aux_projection matvec", err)
	}
	return out, nil
}

// injectedKV projects each verifier hidden through every draft layer's own
// k_proj / v_proj into that layer's injected K/V memory — numAux rows of nKV*headDim
// per layer, the target-context the layer cross-attends. Built once per block (the
// rows are position-independent; ProposeBlock stamps the rope Offset per position).
func (d *DFlashDrafter) injectedKV(auxHiddens [][]byte) ([]AssistantTargetKV, error) {
	m := d.m
	headDim := m.Arch.HeadDim
	kvHeads := m.Arch.KVHeads
	if kvHeads <= 0 {
		kvHeads = m.Arch.Heads
	}
	rowElems := kvHeads * headDim
	backbone := m.BackboneHiddenSize
	out := make([]AssistantTargetKV, len(m.Arch.Layer))
	for li := range m.Arch.Layer {
		prefix := core.Sprintf("model.layers.%d.self_attn.", li)
		kProj, err := nativeAssistantBF16Matrix(m, prefix+"k_proj.weight", rowElems, backbone)
		if err != nil {
			return nil, core.E("native.dflash", core.Sprintf("layer %d k_proj (DFlash injection weight)", li), err)
		}
		vProj, err := nativeAssistantBF16Matrix(m, prefix+"v_proj.weight", rowElems, backbone)
		if err != nil {
			return nil, core.E("native.dflash", core.Sprintf("layer %d v_proj (DFlash injection weight)", li), err)
		}
		key := make([]byte, 0, d.numAux*rowElems*bf16Size)
		val := make([]byte, 0, d.numAux*rowElems*bf16Size)
		for _, h := range auxHiddens {
			kRow, kerr := MatVecBF16Into(nil, kProj.Data, h, rowElems, backbone)
			if kerr != nil {
				return nil, core.E("native.dflash", "k_proj matvec", kerr)
			}
			vRow, verr := MatVecBF16Into(nil, vProj.Data, h, rowElems, backbone)
			if verr != nil {
				return nil, core.E("native.dflash", "v_proj matvec", verr)
			}
			key = append(key, kRow...)
			val = append(val, vRow...)
		}
		out[li] = AssistantTargetKV{
			Key:     key,
			Value:   val,
			Length:  d.numAux,
			KVHeads: kvHeads,
			HeadDim: headDim,
		}
	}
	return out, nil
}

// headArgmax runs the reduced-vocab draft LM head over a final-normed hidden, takes
// the greedy argmax draft id, and maps it through d2t into the target vocab. bf16
// argmax over the drafter's small vocab is a cheap host reduction (no GPU top-k head).
func (d *DFlashDrafter) headArgmax(normed []byte) (int32, error) {
	head, err := nativeAssistantBF16Matrix(d.m, dflashLMHeadWeight, d.draftVocab, d.m.Arch.Hidden)
	if err != nil {
		return 0, core.E("native.dflash", "lm_head", err)
	}
	logits, err := MatVecBF16Into(nil, head.Data, normed, d.draftVocab, d.m.Arch.Hidden)
	if err != nil {
		return 0, core.E("native.dflash", "lm_head matvec", err)
	}
	draftID := dflashArgmaxBF16(logits, d.draftVocab)
	return d.mapDraftToTarget(draftID), nil
}

// mapDraftToTarget applies the d2t table (draft-vocab id → target-vocab id). An
// out-of-range id or an absent table falls back to identity — the drafter then
// proposes in the target vocab directly, which the verifier still accepts losslessly.
func (d *DFlashDrafter) mapDraftToTarget(draftID int32) int32 {
	if draftID < 0 || int(draftID) >= len(d.d2t) {
		return draftID
	}
	return d.d2t[draftID]
}

// dflashArgmaxBF16 returns the index of the greatest bf16 logit in the first n
// elements, ties resolving to the lowest index (deterministic, matching the driver).
func dflashArgmaxBF16(logits []byte, n int) int32 {
	f := bf16ToF32Slice(logits)
	if n > len(f) {
		n = len(f)
	}
	best := int32(0)
	bestVal := float32(nativeAssistantLogitsFloor)
	for i := 0; i < n; i++ {
		if f[i] > bestVal {
			bestVal = f[i]
			best = int32(i)
		}
	}
	return best
}

const (
	// dflashAuxProjectionWeight fuses the concatenated verifier hiddens into the
	// backbone-space context feature; dflashLMHeadWeight is the reduced-vocab draft
	// head; dflashD2TTensor is the draft→target vocab map. These are the tensors a
	// DFlash pack adds on top of the ordinary MTP drafter layout.
	dflashAuxProjectionWeight = "dflash.aux_projection.weight"
	dflashLMHeadWeight        = "dflash.lm_head.weight"
	dflashD2TTensor           = "dflash.d2t"
)

// resolveDFlashMethod reports whether a loaded drafter is a DFlash block-diffusion
// checkpoint — the method the neutral config was stamped with by the reactive spec.
func resolveDFlashMethod(m *AssistantModel) bool {
	return m != nil && m.Config.Method == mtp.MTPDFlash
}
