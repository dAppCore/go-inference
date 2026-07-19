// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// assistant_draft_chain.go — the device-chained draft block (#53: rows-head
// device-direct). The per-step drafter (assistant_load.go's draft loop over
// assistantFusedDraft.step) pays a host round-trip PER DRAFT STEP: commit +
// wait, a full-vocab host argmax over the head logits (a 512KB scan at
// gemma4's 262144 vocab), the hidden/normed readback copies, and a host
// embedding dequant for the next step's input. None of that host work is
// load-bearing for a plain-headed greedy drafter — the argmax kernels the
// production direct-greedy head already runs (lthn_bf16_logits_argmax_tiles +
// lthn_argmax_merge, lowest-index tie-break like the host scan) can pick the
// token on-device, and the session's encNextInputsGPU seam (the chained
// decode's gather) can produce the next step's embedding from that token
// buffer without the host seeing it. Here the WHOLE K-step block encodes into
// ONE command buffer with ONE wait: K token ids plus the final recursion
// hidden come back in a single readback per ROUND instead of per-step
// traffic.
//
// Correctness bar: the MTP verify accepts only the target's own greedy, so a
// drafter-side deviation can only ever move the ACCEPTANCE RATE, never the
// emitted stream. There is deliberately none: the gather is the byte-twin of
// the host embed dequant (lthn_embed_gather oracle), the head qmv is the same
// dispatch the per-step path encodes (encodeStepBody is shared verbatim), and
// the argmax kernels break ties on the lowest index exactly as
// draftGreedyTokenWithSuppress does — so the chained block proposes the SAME
// tokens as the per-step path, gated by TestAssistantDraftChainParity.
//
// The chain declines (per-step path keeps running, byte-identically) for:
// ordered-head assistants (the centroid head needs a host top-k between its
// two matmuls), PLE targets (their next-inputs seam computes the per-layer
// tower the drafter must not pay for), non-default target embedders, missing
// argmax/gather/copy pipelines, and the confidence-capture diag lever (it
// needs the full host logits row by design).

// mtpDraftChainDisabled restores the per-step draft loop
// (LTHN_MTP_DRAFT_CHAIN=0) — the repro anchor / A/B lever for this lane.
var mtpDraftChainDisabled = os.Getenv("LTHN_MTP_DRAFT_CHAIN") == "0"

// draftChainScratch is the chained block's device scratch, owned by the
// pair's assistantFusedDraft and grown on demand: the seed token slot the
// gather reads, the K output token slots, the argmax tile stage, and the
// suppress id list.
type draftChainScratch struct {
	kCap, tileCap, suppressCap int
	seed                       metal.MTLBuffer // [1]int32 — the NEXT step's input token (argmax re-merged here)
	tokens                     metal.MTLBuffer // [kCap]int32 — the block's draft ids, read once after the wait
	tileValues, tileIndices    metal.MTLBuffer // [tileCap] f32/i32 — argmax stage-1 tiles
	suppress                   metal.MTLBuffer // [suppressCap]int32 — argmax suppression ids
	plSc                       *plGPUScratch   // target plScratchNew() product (dense seams never read it)
}

// chainScratchFor returns the fused drafter's chain scratch sized for k draft
// steps and the current suppress list, allocating/growing lazily. nil means a
// buffer allocation failed — the caller declines to the per-step path.
func (f *assistantFusedDraft) chainScratchFor(target *ArchSession, k, suppressLen int) *draftChainScratch {
	s := f.chain
	if s == nil {
		s = &draftChainScratch{}
	}
	if s.seed == nil {
		s.seed = device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
	}
	if s.kCap < k {
		s.tokens = device.NewBufferWithLengthOptions(uint(k*4), metal.MTLResourceStorageModeShared)
		s.kCap = k
	}
	tileCount := (f.vocab + bf16LogitsArgmaxRowsPerTile - 1) / bf16LogitsArgmaxRowsPerTile
	if s.tileCap < tileCount {
		s.tileValues = device.NewBufferWithLengthOptions(uint(tileCount*4), metal.MTLResourceStorageModeShared)
		s.tileIndices = device.NewBufferWithLengthOptions(uint(tileCount*4), metal.MTLResourceStorageModeShared)
		s.tileCap = tileCount
	}
	if suppressLen > 0 && s.suppressCap < suppressLen {
		s.suppress = device.NewBufferWithLengthOptions(uint(suppressLen*4), metal.MTLResourceStorageModeShared)
		s.suppressCap = suppressLen
	}
	if s.plSc == nil && target.plScratchNew != nil {
		s.plSc = target.plScratchNew()
	}
	if s.seed == nil || s.tokens == nil || s.tileValues == nil || s.tileIndices == nil ||
		(suppressLen > 0 && s.suppress == nil) || s.plSc == nil {
		return nil
	}
	f.chain = s
	return s
}

// chainReady reports whether this fused drafter can run a whole draft block
// device-side against the target session. Every condition is a working
// constraint, not a preference: a plain GPU head must be armed (the ordered
// centroid head tops-k on the host between matmuls), the target must expose
// the next-inputs gather WITHOUT a PLE tower (perLayerInput marks the tower;
// the drafter's input is concat(emb, hidden) only), the target embedder must
// be the default one the gather byte-tracks, and the argmax + copy pipelines
// must resolve from the custom metallib.
func (f *assistantFusedDraft) chainReady(target *ArchSession) bool {
	if mtpDraftChainDisabled || f == nil || target == nil {
		return false
	}
	if f.centroidsW != nil || f.embedW == nil || f.vocabLogits == nil || f.vocab <= 0 {
		return false
	}
	if target.encNextInputsGPU == nil || target.plScratchNew == nil || target.perLayerInput != nil {
		return false
	}
	if !target.canUseEmbedScratch() {
		return false
	}
	if _, err := bf16LogitsArgmaxTilesPipeline(); err != nil {
		return false
	}
	if _, err := argmaxMergeF32Pipeline(); err != nil {
		return false
	}
	return gpuHasCopyKernel()
}

// draftBlockChained runs maxDraftTokens fused drafter steps in ONE command
// buffer: per step the target-side gather materialises the input embedding
// from the seed token buffer, the shared encodeStepBody encodes the
// transformer + head, the argmax tiles+merge pick the token on-device (merged
// once into the block's token slot, once into the seed slot for the next
// step's gather), and the recursion hidden hands over via a device copy. The
// host writes the round's seed token + boundary hidden before the encode and
// reads K ids + the final hidden after the ONE wait. tokens must have
// capacity for maxDraftTokens; hiddenOut receives the final backbone hidden.
// ok=false (no error) declines to the per-step path — arming and allocation
// failures are benign here because the per-step loop produces the same
// tokens.
func (f *assistantFusedDraft) draftBlockChained(target *ArchSession, lastToken int32, maxDraftTokens int, tokens []int32, hiddenOut []byte, suppress []int32) (out []int32, hidden []byte, ok bool, err error) {
	if !f.chainReady(target) {
		return nil, nil, false, nil
	}
	currentHidden, herr := target.boundaryNormedHiddenScratch()
	if herr != nil {
		return nil, nil, false, core.E("native.assistant draft chain", "target boundary hidden", herr)
	}
	bb := f.backbone * bf16Size
	if len(currentHidden) != bb {
		return nil, nil, false, core.NewError("native.assistant draft chain boundary hidden bytes mismatch")
	}
	s := f.chainScratchFor(target, maxDraftTokens, len(suppress))
	if s == nil {
		return nil, nil, false, nil
	}
	// Host staging: the round's committed token seeds the first gather, the
	// boundary hidden fills the recursion half of the concat input, and the
	// suppress ids (per-request constant, rewritten per round for safety)
	// feed the argmax tiles.
	*(*int32)(s.seed.Contents()) = lastToken
	copy(unsafe.Slice((*byte)(f.inConcat.Contents()), 2*bb)[bb:], currentHidden)
	if len(suppress) > 0 {
		copy(unsafe.Slice((*int32)(s.suppress.Contents()), len(suppress)), suppress)
	}
	tileCount := (f.vocab + bf16LogitsArgmaxRowsPerTile - 1) / bf16LogitsArgmaxRowsPerTile
	var encErr error
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emit := func(err error) {
			if err != nil && encErr == nil {
				encErr = err
			}
		}
		for k := 0; k < maxDraftTokens; k++ {
			// input embedding ← embed[seed token] — the target's own gather
			// (byte-twin of embedID) writing inConcat[0:backbone).
			emit(target.encNextInputsGPU(enc, s.seed, f.inConcat, s.plSc))
			if k > 0 {
				// recursion hidden hand-over: the previous step's backbone
				// hidden becomes this step's concat second half.
				emit(encCopyBF16Contig(enc, f.outHidden, f.inConcat, 0, uint(bb), f.backbone))
			}
			emit(f.encodeStepBody(enc))
			var sup metal.MTLBuffer
			if len(suppress) > 0 {
				sup = s.suppress
			}
			emit(encBF16LogitsArgmaxTilesBF16At(enc, f.vocabLogits, s.tileValues, s.tileIndices, sup, 0, 0, 0, f.vocab, len(suppress)))
			emit(encArgmaxMergeF32At(enc, s.tileValues, s.tileIndices, s.tokens, 0, 0, uint(k*4), tileCount))
			if k < maxDraftTokens-1 {
				// second merge lands the same id in the seed slot the next
				// step's gather reads — cheaper than plumbing a token offset
				// through every builder's next-inputs closure.
				emit(encArgmaxMergeF32At(enc, s.tileValues, s.tileIndices, s.seed, 0, 0, 0, tileCount))
			}
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
	})
	if encErr != nil {
		return nil, nil, false, encErr
	}
	ids := unsafe.Slice((*int32)(s.tokens.Contents()), maxDraftTokens)
	tokens = tokens[:0]
	for i := 0; i < maxDraftTokens; i++ {
		if ids[i] < 0 || int(ids[i]) >= f.vocab {
			return nil, nil, false, core.NewError(core.Sprintf("native.assistant draft chain step %d argmax returned invalid token %d for vocab %d", i, ids[i], f.vocab))
		}
		tokens = append(tokens, ids[i])
	}
	hiddenOut = hiddenOut[:bb]
	copy(hiddenOut, unsafe.Slice((*byte)(f.outHidden.Contents()), bb))
	return tokens, hiddenOut, true, nil
}
